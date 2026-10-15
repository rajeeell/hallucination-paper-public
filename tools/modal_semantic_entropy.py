"""Modal runner for Semantic Entropy (Kuhn 2023 / Farquhar 2024).

Estimate semantic entropy from multiple generated answers and compare
it with perplexity on paired examples.

Algorithm (Kuhn et al. 2023):
  1. For each (question, context) pair, sample N generations at T=1.0
     (we use N=10, matching the original paper).
  2. Run pairwise bidirectional NLI between the N generations. Two
     generations a, b are "semantically equivalent" iff
     NLI(a, b) == entailment AND NLI(b, a) == entailment.
  3. Cluster generations into equivalence classes via connected components.
  4. Semantic Entropy = Shannon entropy of the cluster-size distribution:
        H = -sum_c (n_c / N) * log(n_c / N)
     Higher H means the model produces more semantically distinct answers,
     i.e., it is "uncertain" in the semantic sense rather than the
     token-uncertainty sense.

For paired comparison with PPL, CCP and Max-K results, record the
perplexity of the FIRST
generation (greedy-then-T=1.0 sampling) so that the AUROCs are computed
on a paired subset.

Cost estimate (8 cells × 200 samples × 10 generations at T=1.0):
  - generation: ~17 min wall time per cell on H100
  - NLI clustering: ~3-5 min per cell on the same H100
  - 8 cells in parallel: ~25 min total wall time
  - ~$8-12 modal compute

Usage:
  modal deploy tools/modal_semantic_entropy.py
  modal run tools/modal_semantic_entropy.py::smoke    # n=20 on Phi+SQuAD
  modal run tools/modal_semantic_entropy.py::main --model-name "..." --dataset "..." --n-samples 200
  modal run tools/modal_semantic_entropy.py::run_all_cells --n-samples 200
"""

from __future__ import annotations

import modal

APP_NAME = "rehab-semantic-entropy"

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


ALL_CELLS = [
    ("microsoft/Phi-3-mini-4k-instruct",   "squad"),
    ("microsoft/Phi-3-mini-4k-instruct",   "hotpot"),
    ("google/gemma-3-4b-it",               "squad"),
    ("google/gemma-3-4b-it",               "hotpot"),
    ("meta-llama/Llama-3.1-8B-Instruct",   "squad"),
    ("meta-llama/Llama-3.1-8B-Instruct",   "hotpot"),
    ("Qwen/Qwen2.5-14B-Instruct",          "squad"),
    ("Qwen/Qwen2.5-14B-Instruct",          "hotpot"),
]


# DeBERTa-v3-large-MNLI: standard NLI model used in semantic-entropy papers.
# 440M params, runs on H100 alongside the generator without memory pressure.
NLI_MODEL = "microsoft/deberta-large-mnli"


# Same generation prompt scaffold as the paper's existing pipeline (see
# modal_vary_k.py:generate_answer). The only differences vs the paper's
# original answer generation are (i) temperature is 1.0 here for diversity
# (vs the paper's 0.3) and (ii) we generate N=10 times instead of once.
GEN_PROMPT = (
    "Answer the question based on the context below. Be concise (1-2 sentences).\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _binmatched_filename(model_name: str, dataset: str) -> str:
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
    timeout=7200,
)
def run_semantic_entropy(
    model_name: str,
    dataset: str,
    n_samples: int = 200,
    n_generations: int = 10,
    sampling_temperature: float = 1.0,
    smoke_test: bool = False,
) -> dict:
    """Sample N generations per question, cluster by bidirectional NLI,
    compute semantic entropy over the cluster distribution.
    """
    import json
    import math
    import os
    import time
    import numpy as np
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        AutoModelForSequenceClassification,
    )
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
        bm = bm[:20]
        n_generations = max(3, n_generations // 3)
    else:
        bm = bm[:n_samples]
    print(f"[load] {len(bm)} samples, n_generations={n_generations}, T={sampling_temperature}")

    print(f"[load] generator {model_name}")
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    gen_model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token, device_map="auto",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    gen_model.eval()

    print(f"[load] NLI model {NLI_MODEL}")
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    # NLI in float32 — DeBERTa's classification head has a known bf16
    # incompatibility with its pooled-output projection. 440M params in
    # fp32 fits comfortably on H100 alongside the generator.
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL, torch_dtype=torch.float32,
    ).to("cuda")
    nli_model.eval()
    # DeBERTa-large-mnli label order: ['CONTRADICTION', 'NEUTRAL', 'ENTAILMENT']
    # Verify so we don't silently misuse the labels.
    id2label = nli_model.config.id2label
    entail_idx = next(i for i, lab in id2label.items() if "ENTAIL" in lab.upper())
    print(f"[load] NLI entailment index = {entail_idx} (labels: {id2label})")

    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[load] gpu vram total = {gpu_gb:.1f} GB")

    def sample_generations(question: str, context_raw, n: int) -> tuple[list[str], list[float]]:
        """Return (n generations, n per-generation perplexities)."""
        context_str = "\n\n".join(context_raw) if isinstance(context_raw, list) else str(context_raw)
        prompt = GEN_PROMPT.format(context=context_str, question=question)
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        gens, ppls = [], []
        with torch.no_grad():
            for _ in range(n):
                out = gen_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=sampling_temperature,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                # Per-token log-probs of the generated sequence
                if len(out.scores) > 0:
                    stacked = torch.stack(out.scores, dim=0)  # (gen_len, 1, vocab)
                    probs = torch.softmax(stacked.float(), dim=-1)
                    chosen = gen_ids[:probs.shape[0]]
                    tok_p = probs[range(probs.shape[0]), 0, chosen].cpu().numpy()
                    tok_p = tok_p[np.isfinite(tok_p) & (tok_p > 0)]
                    if len(tok_p) > 0:
                        ppls.append(float(np.exp(-np.mean(np.log(tok_p)))))
                    else:
                        ppls.append(float("nan"))
                else:
                    ppls.append(float("nan"))
                gens.append(text)
        return gens, ppls

    def nli_entails(a: str, b: str) -> bool:
        """Return True if NLI(a -> b) == entailment."""
        if not a.strip() or not b.strip():
            return False
        inp = nli_tokenizer(a, b, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            logits = nli_model(**inp).logits
        return int(torch.argmax(logits, dim=-1).item()) == entail_idx

    def cluster_by_equivalence(gens: list[str]) -> list[int]:
        """Pairwise bidirectional NLI, then connected components → cluster labels."""
        n = len(gens)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for i in range(n):
            for j in range(i + 1, n):
                if nli_entails(gens[i], gens[j]) and nli_entails(gens[j], gens[i]):
                    union(i, j)
        roots = [find(i) for i in range(n)]
        # Remap roots to 0..K-1
        remap, labels = {}, []
        for r in roots:
            if r not in remap:
                remap[r] = len(remap)
            labels.append(remap[r])
        return labels

    def semantic_entropy(labels: list[int]) -> float:
        n = len(labels)
        if n == 0:
            return float("nan")
        from collections import Counter
        counts = Counter(labels)
        probs = np.array(list(counts.values()), dtype=float) / n
        return float(-(probs * np.log(probs + 1e-12)).sum())

    model_slug = model_name.replace("/", "_")
    out_dir = f"/results/semantic_entropy/{model_slug}/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    records = []
    skipped = 0
    t0 = time.time()
    for i, sample in enumerate(tqdm(bm)):
        try:
            gens, ppls = sample_generations(
                question=sample.get("question", ""),
                context_raw=sample.get("context", ""),
                n=n_generations,
            )
            if not gens or all(not g.strip() for g in gens):
                skipped += 1
                continue
            labels = cluster_by_equivalence(gens)
            se = semantic_entropy(labels)
            finite_ppls = [p for p in ppls if np.isfinite(p)]
            mean_ppl = float(np.mean(finite_ppls)) if finite_ppls else float("nan")
            records.append({
                "id": i,
                "gold_label": int(sample.get("hallucinated", -1)),
                "n_generations": n_generations,
                "n_clusters": len(set(labels)),
                "cluster_sizes": [labels.count(c) for c in sorted(set(labels))],
                "semantic_entropy": se,
                "mean_gen_perplexity": mean_ppl,
                "first_gen_perplexity": float(ppls[0]) if ppls and np.isfinite(ppls[0]) else float("nan"),
                "generations": gens,
            })
        except Exception as e:
            skipped += 1
            print(f"[warn] sample {i} failed: {type(e).__name__}: {e}")

    out_name = "semantic_entropy_smoke.json" if smoke_test else f"semantic_entropy_n{n_samples}.json"
    out_json = f"{out_dir}/{out_name}"
    summary = {
        "model": model_name,
        "dataset": dataset,
        "n_samples_requested": len(bm),
        "n_kept": len(records),
        "n_skipped": skipped,
        "n_generations": n_generations,
        "sampling_temperature": sampling_temperature,
        "nli_model": NLI_MODEL,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    results_vol.commit()
    print(f"[save] {len(records)} records -> {out_json}")
    print(f"[summary] elapsed={summary['elapsed_seconds']}s, skipped={skipped}")
    if records:
        import numpy as _np
        ses = [r["semantic_entropy"] for r in records if _np.isfinite(r["semantic_entropy"])]
        ncls = [r["n_clusters"] for r in records]
        print(f"[stats] semantic_entropy mean={_np.mean(ses):.3f} std={_np.std(ses):.3f}")
        print(f"[stats] n_clusters mean={_np.mean(ncls):.2f} (max possible={n_generations})")
    return summary


@app.local_entrypoint()
def smoke(model_name: str = "microsoft/Phi-3-mini-4k-instruct", dataset: str = "squad"):
    """Smoke test: 20 samples, 3 generations each."""
    result = run_semantic_entropy.remote(model_name, dataset, smoke_test=True)
    import json as _json
    print(_json.dumps(result, indent=2))


@app.local_entrypoint()
def main(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    dataset: str = "squad",
    n_samples: int = 200,
    n_generations: int = 10,
):
    result = run_semantic_entropy.remote(model_name, dataset, n_samples, n_generations)
    import json as _json
    print(_json.dumps(result, indent=2))


@app.local_entrypoint()
def run_all_cells(n_samples: int = 200, n_generations: int = 10):
    """Dispatch all 8 cells in parallel on H100s."""
    handles = []
    for (model_name, dataset) in ALL_CELLS:
        handles.append((model_name, dataset,
                        run_semantic_entropy.spawn(model_name, dataset, n_samples, n_generations)))
    print(f"[dispatched] {len(handles)} runs")
    out = []
    for m, d, h in handles:
        r = h.get()
        print(f"[done] {m} / {d}: n_kept={r['n_kept']}, elapsed={r['elapsed_seconds']}s")
        out.append(r)
    import json as _json
    print(_json.dumps(out, indent=2))
