"""Modal runner for the memorization probe (pre-hoc regime predictor).

For every (question, gold_answer) pair across the 4-model × 2-dataset grid,
compute the model's teacher-forced negative log-likelihood of the gold
answer given ONLY the question (no retrieved context). We call this
"memorization perplexity" (`mem_ppl`).

Hypothesis:
  cells with low mem_ppl (model has memorized the gold answer from
  pretraining) are exactly the cells where perplexity AUROC reverses
  under retrieval degradation at k=1.

Compare these no-context scores with retrieval-experiment metrics to
study whether gold-answer likelihood predicts uncertainty-based detection.

Output: one JSON per (model, dataset) cell at
  /results/memorization/<model_slug>/<dataset>/memprobe_n{N}.json

Each record: {id, question, gold_answer, mem_nll, mem_ppl, gold_len_tokens}.

Usage:
  modal deploy tools/modal_memorization_probe.py

  # CPU sanity-check first (no model load, no GPU spend):
  modal run tools/modal_memorization_probe.py::validate --dataset squad

  # Single cell (default Qwen + squad, non-gated):
  modal run tools/modal_memorization_probe.py::main \\
      --model-name "Qwen/Qwen2.5-14B-Instruct" --dataset squad --n-samples 200

  # All 8 cells in parallel (needs HF_TOKEN for gated gemma + llama):
  modal run tools/modal_memorization_probe.py::run_all_cells --n-samples 200
"""

from __future__ import annotations

import modal

APP_NAME = "rehab-memprobe"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "transformers>=4.50.0,<4.55.0",
        "accelerate>=1.0.0",
        "datasets>=2.20.0",
        "numpy<2",
        "tqdm",
        "huggingface_hub>=0.24.0",
    )
)

app = modal.App(APP_NAME, image=image)

results_vol = modal.Volume.from_name("rehab-results", create_if_missing=True)

import os as _os
_hf_secret = modal.Secret.from_dict({"HF_TOKEN": _os.environ.get("HF_TOKEN", "")})


# Match the natural-hallucination prompt scaffolding from tests/12 *exactly*,
# except for the absence of the Context block — this is the controlled-comparison
# anchor for §5.7. If §5.6 uses chat-templated "Answer based on context", §5.7
# must use chat-templated "Answer the question" with no context, otherwise the
# memorization-perplexity numbers are not interpretable against the existing
# perplexities in the paper.
NO_CONTEXT_PROMPT = (
    "Answer the question.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


@app.function(
    gpu="H100",
    volumes={"/results": results_vol},
    secrets=[_hf_secret],
    timeout=3600,
)
def run_memorization_probe(
    model_name: str,
    dataset: str,
    n_samples: int = 200,
    seed: int = 42,
) -> dict:
    """Score teacher-forced NLL of gold answers under no-context prompts."""
    import os
    import json
    import math
    import time
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    hf_token = os.environ.get("HF_TOKEN") or None
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # --- load QA dataset only (no retrieval needed) ---
    if dataset == "hotpot":
        qa_dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    elif dataset == "squad":
        qa_dataset = load_dataset("squad", split="validation")
    else:
        raise ValueError(f"unknown dataset: {dataset!r} (expected 'hotpot' or 'squad')")
    print(f"[load] qa={len(qa_dataset)} questions for dataset={dataset}")

    # --- load model in bf16 ---
    print(f"[load] model {model_name}")
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[load] model on gpu; total vram={gpu_gb:.1f} GB")

    # --- pick sample indices (same seed as vary_k so cells are paired by id) ---
    np.random.seed(seed)
    indices = np.random.choice(len(qa_dataset), n_samples, replace=False)

    def score_gold(question: str, gold_answer: str) -> tuple[float, float, int]:
        """Return (mem_nll_sum, mem_ppl, gold_len_tokens).

        Teacher-forced: build the chat-templated no-context prompt, append the
        gold answer, single forward pass, sum log-probs over the gold-answer
        tokens only. Empty gold answers return (nan, nan, 0).
        """
        if not gold_answer or not gold_answer.strip():
            return float("nan"), float("nan"), 0
        messages = [{"role": "user", "content": NO_CONTEXT_PROMPT.format(question=question)}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Tokenize prompt alone to learn its length, then prompt+answer for the
        # forward pass. No separator between prompt and answer — matches the
        # paper's existing extract_uncertainty (modal_vary_k.py), which does
        # `full_text = input_text + answer`. apply_chat_template with
        # add_generation_prompt=True already ends the prompt with the chat
        # template's assistant-turn marker, so the joined sequence is well-formed.
        full_text = prompt_text + gold_answer.strip()
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        full_ids = tokenizer(full_text, return_tensors="pt").to("cuda")
        prompt_len = prompt_ids["input_ids"].shape[1]
        full_len = full_ids["input_ids"].shape[1]
        gold_len = full_len - prompt_len
        if gold_len <= 0:
            return float("nan"), float("nan"), 0
        with torch.no_grad():
            out = model(**full_ids)
            logits = out.logits  # (1, full_len, vocab)
        # logit at position t predicts token at t+1, so to predict the gold
        # tokens at positions [prompt_len .. full_len), we read logits at
        # positions [prompt_len-1 .. full_len-1).
        gold_logits = logits[0, prompt_len - 1 : full_len - 1, :]
        gold_tokens = full_ids["input_ids"][0, prompt_len:full_len]
        log_probs = torch.log_softmax(gold_logits.float(), dim=-1)
        token_log_probs = log_probs[range(gold_len), gold_tokens].cpu().numpy()
        # Guard against the rare -inf log-prob (impossible token under the
        # current floating point ops in bf16->float32 cast).
        finite = np.isfinite(token_log_probs)
        if finite.sum() == 0:
            return float("nan"), float("nan"), int(gold_len)
        mean_log_prob = float(np.mean(token_log_probs[finite]))
        mem_nll_sum = float(-np.sum(token_log_probs[finite]))
        mem_ppl = float(np.exp(-mean_log_prob))
        return mem_nll_sum, mem_ppl, int(gold_len)

    model_slug = model_name.replace("/", "_")
    out_dir = f"/results/memorization/{model_slug}/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    records = []
    skipped = 0
    t0 = time.time()
    for i, idx in enumerate(tqdm(indices)):
        try:
            item = qa_dataset[int(idx)]
            question = item["question"]
            if dataset == "hotpot":
                gold = item.get("answer", "")
            else:  # squad
                gold = item.get("answers", {}).get("text", [""])[0]
            mem_nll, mem_ppl, gold_len = score_gold(question, gold)
            records.append({
                "id": i,
                "ds_index": int(idx),
                "question": question,
                "gold_answer": gold,
                "mem_nll_sum": mem_nll,
                "mem_ppl": mem_ppl,
                "gold_len_tokens": gold_len,
            })
        except Exception as e:
            skipped += 1
            print(f"[warn] sample idx={int(idx)} (i={i}) failed: {type(e).__name__}: {e}")
            continue
    if skipped:
        print(f"[warn] skipped {skipped}/{len(indices)} samples")

    # Summary stats (median is robust to the long tail mem_ppl tends to have)
    finite_ppls = [r["mem_ppl"] for r in records if r["mem_ppl"] is not None and math.isfinite(r["mem_ppl"])]
    summary = {
        "model": model_name,
        "dataset": dataset,
        "n_samples": n_samples,
        "n_kept": len(records),
        "n_skipped": skipped,
        "seed": seed,
        "elapsed_seconds": round(time.time() - t0, 2),
        "mem_ppl_median": float(np.median(finite_ppls)) if finite_ppls else None,
        "mem_ppl_mean": float(np.mean(finite_ppls)) if finite_ppls else None,
        "mem_ppl_p10": float(np.percentile(finite_ppls, 10)) if finite_ppls else None,
        "mem_ppl_p90": float(np.percentile(finite_ppls, 90)) if finite_ppls else None,
    }

    out_json = f"{out_dir}/memprobe_n{n_samples}.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    results_vol.commit()
    print(f"[save] {len(records)} records -> {out_json}")
    print(f"[summary] mem_ppl median={summary['mem_ppl_median']:.3f} "
          f"p10={summary['mem_ppl_p10']:.3f} p90={summary['mem_ppl_p90']:.3f} "
          f"(elapsed={summary['elapsed_seconds']}s)")
    return summary


@app.function(
    volumes={"/results": results_vol},
    timeout=600,
)
def validate_data_pipeline(dataset: str = "hotpot") -> dict:
    """CPU-only sanity check: dataset loads, gold-answer extraction works,
    chat-template prompt builds cleanly. ~$0.01 of CPU time."""
    import json
    import time
    from datasets import load_dataset
    from transformers import AutoTokenizer

    report = {"dataset": dataset, "checks": {}}
    t0 = time.time()

    if dataset == "hotpot":
        qa = load_dataset("hotpot_qa", "distractor", split="validation")
        gold_extractor = lambda item: item.get("answer", "")
    elif dataset == "squad":
        qa = load_dataset("squad", split="validation")
        gold_extractor = lambda item: item.get("answers", {}).get("text", [""])[0]
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")
    report["checks"]["qa_len"] = len(qa)
    report["checks"]["qa_sample_question"] = qa[0]["question"][:120]
    report["checks"]["qa_sample_gold"] = gold_extractor(qa[0])[:120]

    # Sanity-check chat-template formatting against a small, non-gated tokenizer
    # (Phi-3-mini-4k-instruct is openly downloadable without HF_TOKEN).
    tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    messages = [{"role": "user", "content": NO_CONTEXT_PROMPT.format(question=qa[0]["question"])}]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    report["checks"]["prompt_preview"] = prompt_text[:240]
    report["checks"]["prompt_tokens"] = len(tok.encode(prompt_text))

    marker_path = f"/results/_memprobe_validate_{dataset}.json"
    with open(marker_path, "w") as f:
        json.dump(report, f, indent=2)
    results_vol.commit()
    report["checks"]["write_marker"] = marker_path
    report["checks"]["elapsed_seconds"] = round(time.time() - t0, 2)
    print(f"[validate] ok ({report['checks']['elapsed_seconds']}s)")
    return report


@app.local_entrypoint()
def validate(dataset: str = "hotpot"):
    """CPU-only data-pipeline sanity check."""
    result = validate_data_pipeline.remote(dataset)
    import json as _json
    print(_json.dumps(result, indent=2))


# 8 controlled-cell pairs matching the paper's main grid.
ALL_CELLS = [
    ("microsoft/Phi-3-mini-4k-instruct", "squad"),
    ("microsoft/Phi-3-mini-4k-instruct", "hotpot"),
    ("google/gemma-3-4b-it", "squad"),
    ("google/gemma-3-4b-it", "hotpot"),
    ("meta-llama/Llama-3.1-8B-Instruct", "squad"),
    ("meta-llama/Llama-3.1-8B-Instruct", "hotpot"),
    ("Qwen/Qwen2.5-14B-Instruct", "squad"),
    ("Qwen/Qwen2.5-14B-Instruct", "hotpot"),
]


@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    dataset: str = "squad",
    n_samples: int = 200,
):
    """Score one (model, dataset) cell. Default is Qwen + squad (non-gated)."""
    result = run_memorization_probe.remote(model_name, dataset, n_samples)
    print("[entrypoint] summary:")
    import json as _json
    print(_json.dumps(result, indent=2))


@app.local_entrypoint()
def run_all_cells(n_samples: int = 200):
    """Dispatch all 8 cells in parallel on separate H100s.

    Gated models (gemma + llama) need HF_TOKEN exported locally.
    """
    handles = [
        (model_name, dataset, run_memorization_probe.spawn(model_name, dataset, n_samples))
        for (model_name, dataset) in ALL_CELLS
    ]
    summaries = []
    for model_name, dataset, h in handles:
        result = h.get()
        print(f"[done] {model_name} / {dataset} -> "
              f"median mem_ppl = {result.get('mem_ppl_median'):.3f}")
        summaries.append(result)
    print("\n[all cells complete]")
    import json as _json
    print(_json.dumps(summaries, indent=2))
