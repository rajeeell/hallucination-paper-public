"""Modal runner for the Fadeeva 2024 baseline (CCP + Max-K + Min-K).

Compute token-level uncertainty baselines on the same controlled
examples used for the perplexity comparisons.

What we compute per (controlled cell, sample):
  - per-token negative log-probability (teacher-forced over the existing
    generated answer)
  - sentence-based Claim-Conditioned Probability (CCP-sent): split the
    generated answer at sentence boundaries; for each sentence compute the
    mean negative log-probability; aggregate via max over sentences (the
    "most uncertain claim" signal, matching Fadeeva's max-over-claims rule).
    Sentence segmentation is the tractable proxy for Fadeeva's atomic-claim
    decomposition --- the paper notes this simplification in §2 and the
    Limitations.
  - Max-K negative log-prob (K = 1, 5, 10): mean of the K largest
    -log p values over the answer's tokens. Fadeeva's "MaxSeq" variant.
  - Min-K negative log-prob (K = 5, 10): the Min-K%-style aggregator from
    Shi et al. 2024, related but reported in the Fadeeva paper as an
    ablation.

Input: the bin-matched JSON files at
  `<cell_root>/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`
mounted into the GPU container via a Modal volume `rehab-binmatched`.

Output: per-cell JSON at
  `/results/fadeeva/<model_slug>/<dataset>/fadeeva_ccp.json`
with one record per sample: {id, perplexity_paper, ccp_sent, maxk_1, maxk_5,
maxk_10, mink_5_pct, mink_10_pct, gold_label}.

Cost estimate (8 cells in parallel on H100): ~$8-12.

Usage:
  # one-time upload of binmatched JSONs to the new volume
  modal volume create rehab-binmatched
  python3 tools/upload_binmatched_to_modal.py   # see below; or inline manually
  modal deploy tools/modal_fadeeva_ccp.py

  # CPU validate first
  modal run tools/modal_fadeeva_ccp.py::validate --model-slug phi --dataset squad

  # one cell
  modal run tools/modal_fadeeva_ccp.py::main \\
      --model-name "microsoft/Phi-3-mini-4k-instruct" --dataset squad

  # full 8-cell sweep
  modal run tools/modal_fadeeva_ccp.py::run_all_cells
"""

from __future__ import annotations

import modal

APP_NAME = "rehab-fadeeva-ccp"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "transformers>=4.50.0,<4.55.0",
        "accelerate>=1.0.0",
        "numpy<2",
        "tqdm",
        "huggingface_hub>=0.24.0",
    )
)

app = modal.App(APP_NAME, image=image)

binmatched_vol = modal.Volume.from_name("rehab-binmatched", create_if_missing=True)
results_vol = modal.Volume.from_name("rehab-results", create_if_missing=True)

import os as _os
_hf_secret = modal.Secret.from_dict({"HF_TOKEN": _os.environ.get("HF_TOKEN", "")})


# (model_name, dataset, binmatched_filename_on_volume)
# The slug after the underscore matches the local repo path so the
# uploaded JSONs are kept distinct per cell.
ALL_CELLS = [
    ("microsoft/Phi-3-mini-4k-instruct",   "squad",  "phi_squad.json"),
    ("microsoft/Phi-3-mini-4k-instruct",   "hotpot", "phi_hotpot.json"),
    ("google/gemma-3-4b-it",               "squad",  "gemma_squad.json"),
    ("google/gemma-3-4b-it",               "hotpot", "gemma_hotpot.json"),
    ("meta-llama/Llama-3.1-8B-Instruct",   "squad",  "llama_squad.json"),
    ("meta-llama/Llama-3.1-8B-Instruct",   "hotpot", "llama_hotpot.json"),
    ("Qwen/Qwen2.5-14B-Instruct",          "squad",  "qwen_squad.json"),
    ("Qwen/Qwen2.5-14B-Instruct",          "hotpot", "qwen_hotpot.json"),
]


# Same generation prompt as `extract_uncertainty` in modal_vary_k.py — we
# must reuse the exact prompt the answers were generated under, otherwise
# the per-token probabilities are not interpretable.
SCORE_PROMPT = (
    "Answer the question based on the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _binmatched_filename(model_name: str, dataset: str) -> str:
    """The on-volume filename for a (model, dataset) cell."""
    model_short = {
        "microsoft/Phi-3-mini-4k-instruct": "phi",
        "google/gemma-3-4b-it": "gemma",
        "meta-llama/Llama-3.1-8B-Instruct": "llama",
        "Qwen/Qwen2.5-14B-Instruct": "qwen",
    }[model_name]
    return f"{model_short}_{dataset}.json"


@app.function(
    gpu="H100",
    volumes={"/binmatched": binmatched_vol, "/results": results_vol},
    secrets=[_hf_secret],
    timeout=3600,
)
def run_fadeeva_ccp(
    model_name: str,
    dataset: str,
    n_samples: int | None = None,
    smoke_test: bool = False,
) -> dict:
    """Teacher-force the binmatched answers, log per-token log-probs,
    compute CCP + Max-K + Min-K aggregations.
    """
    import json
    import math
    import os
    import re
    import time
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    hf_token = os.environ.get("HF_TOKEN") or None
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    bm_filename = _binmatched_filename(model_name, dataset)
    bm_path = f"/binmatched/{bm_filename}"
    print(f"[load] binmatched {bm_path}")
    with open(bm_path) as f:
        bm = json.load(f)
    if smoke_test:
        bm = bm[:50]
    elif n_samples is not None:
        bm = bm[:n_samples]
    print(f"[load] {len(bm)} samples")

    print(f"[load] model {model_name}")
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token, device_map="auto",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.eval()
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[load] model on gpu; total vram={gpu_gb:.1f} GB")

    # Sentence-segmenter: simple regex, language-agnostic enough for our use.
    # We only care that consecutive tokens within a sentence stay together;
    # the exact boundary choice (period vs ;) does not materially shift the
    # aggregate score because we always re-aggregate by tokens within each
    # span.
    SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def score_sample(question: str, context_raw, answer: str) -> dict:
        if not answer or not answer.strip():
            return {"skipped": True}
        context_str = "\n\n".join(context_raw) if isinstance(context_raw, list) else str(context_raw)
        prompt_text = SCORE_PROMPT.format(context=context_str, question=question)
        # Reproduce the convention from extract_uncertainty in modal_vary_k.py:
        # apply the chat template to the prompt, then concatenate the answer.
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = input_text + answer
        prompt_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
        full_ids = tokenizer(full_text, return_tensors="pt").to("cuda")
        prompt_len = int(prompt_ids["input_ids"].shape[1])
        full_len = int(full_ids["input_ids"].shape[1])
        ans_len = full_len - prompt_len
        if ans_len <= 0:
            return {"skipped": True}
        with torch.no_grad():
            out = model(**full_ids)
            logits = out.logits  # (1, full_len, vocab)
        # Position t's logit predicts token t+1, so for answer tokens at
        # positions [prompt_len, full_len) we read logits at [prompt_len-1, full_len-1).
        ans_logits = logits[0, prompt_len - 1 : full_len - 1, :]
        ans_tokens = full_ids["input_ids"][0, prompt_len:full_len]
        log_probs = torch.log_softmax(ans_logits.float(), dim=-1)
        token_log_probs = log_probs[range(ans_len), ans_tokens].cpu().numpy()
        # Per-token negative log-prob = surprisal
        nll = -token_log_probs
        finite = np.isfinite(nll)
        nll = nll[finite]
        if len(nll) == 0:
            return {"skipped": True}

        # Sequence-level perplexity (sanity check; should match the paper's
        # reported number when computed on the same answer + context)
        ppl_paper = float(np.exp(nll.mean()))

        # ---- Max-K aggregations (Fadeeva MaxSeq / Min-K style) ----
        nll_sorted_desc = np.sort(nll)[::-1]  # largest surprisal first
        def topk_mean(k: int) -> float:
            k = min(k, len(nll_sorted_desc))
            return float(nll_sorted_desc[:k].mean()) if k else float("nan")
        maxk_1  = topk_mean(1)
        maxk_5  = topk_mean(5)
        maxk_10 = topk_mean(10)

        # ---- Min-K%-prob: mean of the smallest K% of log-probs ----
        # Shi et al. 2024: lower mean = more confident; higher = more uncertain.
        # We report the negative-log-prob version (higher = more uncertain) for
        # consistency with the other aggregations.
        def mink_pct(pct: float) -> float:
            k = max(1, int(round(len(nll) * pct / 100.0)))
            return float(np.sort(nll)[-k:].mean())  # top-k surprisal among the answer's tokens
        mink_5  = mink_pct(5.0)
        mink_10 = mink_pct(10.0)

        # ---- Sentence-based CCP ----
        # Decode the answer once to find sentence boundaries by character offset,
        # then map back to token indices using the tokenizer's offset_mapping.
        # Simpler approach (less brittle): split the *decoded* answer on
        # sentence boundaries, then for each sentence count its number of
        # answer tokens and take the corresponding slice of token_log_probs.
        answer_decoded = tokenizer.decode(ans_tokens, skip_special_tokens=True)
        sentences = [s.strip() for s in SENT_BOUNDARY.split(answer_decoded) if s.strip()]
        if not sentences:
            sentences = [answer_decoded]
        # Approximate token-per-sentence allocation by proportional length.
        # This is a tractable proxy for Fadeeva's atomic-claim decomposition.
        # It is reported as such in the paper's §2 / Limitations.
        sent_char_lens = np.array([len(s) for s in sentences], dtype=float)
        sent_token_alloc = np.maximum(1, np.round(
            sent_char_lens / sent_char_lens.sum() * len(nll)
        ).astype(int))
        # Reconcile rounding: ensure sum == len(nll)
        delta = len(nll) - sent_token_alloc.sum()
        if delta != 0:
            sent_token_alloc[-1] += delta
            sent_token_alloc[-1] = max(1, sent_token_alloc[-1])
        # Now slice
        sent_scores = []
        idx = 0
        for w in sent_token_alloc:
            chunk = nll[idx:idx + w]
            idx += w
            if len(chunk):
                sent_scores.append(float(chunk.mean()))
        ccp_sent_max = float(max(sent_scores)) if sent_scores else float("nan")
        ccp_sent_min = float(min(sent_scores)) if sent_scores else float("nan")
        ccp_sent_mean = float(np.mean(sent_scores)) if sent_scores else float("nan")

        return {
            "n_tokens": int(len(nll)),
            "n_sentences": int(len(sentences)),
            "ppl_paper": ppl_paper,
            "ccp_sent_max": ccp_sent_max,
            "ccp_sent_min": ccp_sent_min,
            "ccp_sent_mean": ccp_sent_mean,
            "maxk_1": maxk_1,
            "maxk_5": maxk_5,
            "maxk_10": maxk_10,
            "mink_5_pct": mink_5,
            "mink_10_pct": mink_10,
        }

    model_slug = model_name.replace("/", "_")
    out_dir = f"/results/fadeeva/{model_slug}/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    records = []
    skipped = 0
    t0 = time.time()
    for i, sample in enumerate(tqdm(bm)):
        try:
            scores = score_sample(
                question=sample.get("question", ""),
                context_raw=sample.get("context", ""),
                answer=sample.get("answer", ""),
            )
            if scores.get("skipped"):
                skipped += 1
                continue
            records.append({
                "id": i,
                "gold_label": int(sample.get("hallucinated", -1)),
                **scores,
            })
        except Exception as e:
            skipped += 1
            print(f"[warn] sample {i} failed: {type(e).__name__}: {e}")

    out_json = f"{out_dir}/fadeeva_ccp.json"
    if smoke_test:
        out_json = f"{out_dir}/fadeeva_ccp_smoke.json"
    summary = {
        "model": model_name,
        "dataset": dataset,
        "n_samples": len(bm),
        "n_kept": len(records),
        "n_skipped": skipped,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    results_vol.commit()
    print(f"[save] {len(records)} records -> {out_json}")
    print(f"[summary] elapsed={summary['elapsed_seconds']}s, skipped={skipped}")
    return summary


@app.function(
    volumes={"/binmatched": binmatched_vol},
    timeout=600,
)
def validate(model_slug: str = "phi", dataset: str = "squad") -> dict:
    """CPU-only sanity check: confirm the binmatched JSON loads from the volume
    and has the expected schema (question, context, answer, hallucinated).
    """
    import json
    import os

    filename = f"{model_slug}_{dataset}.json"
    path = f"/binmatched/{filename}"
    if not os.path.exists(path):
        return {"ok": False, "error": f"missing file: {path}",
                "available": sorted(os.listdir("/binmatched"))}
    with open(path) as f:
        bm = json.load(f)
    n = len(bm)
    sample = bm[0] if n else {}
    report = {
        "ok": True,
        "n": n,
        "keys": list(sample.keys()) if sample else [],
        "first_question": sample.get("question", "")[:120],
        "first_answer_first_80": sample.get("answer", "")[:80],
        "first_label": sample.get("hallucinated"),
    }
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    dataset: str = "squad",
    smoke_test: bool = False,
):
    result = run_fadeeva_ccp.remote(model_name, dataset, smoke_test=smoke_test)
    import json as _json
    print(_json.dumps(result, indent=2))


@app.local_entrypoint()
def smoke(model_name: str = "microsoft/Phi-3-mini-4k-instruct", dataset: str = "squad"):
    """Run on 50 samples only as a smoke test."""
    result = run_fadeeva_ccp.remote(model_name, dataset, smoke_test=True)
    import json as _json
    print(_json.dumps(result, indent=2))


@app.local_entrypoint()
def run_all_cells():
    """Dispatch all 8 cells in parallel on separate H100s."""
    handles = []
    for (model_name, dataset, _filename) in ALL_CELLS:
        handles.append((model_name, dataset,
                        run_fadeeva_ccp.spawn(model_name, dataset)))
    print(f"[dispatched] {len(handles)} runs")
    out = []
    for m, d, h in handles:
        r = h.get()
        print(f"[done] {m} / {d}: n_kept={r['n_kept']}, elapsed={r['elapsed_seconds']}s")
        out.append(r)
    import json as _json
    print(_json.dumps(out, indent=2))
