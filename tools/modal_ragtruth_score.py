"""Modal runner: score RAGTruth's externally-annotated QA hallucinations
with our 4 models' token-uncertainty signals.

RAGTruth (Niu et al. 2024, ACL) provides human-annotated
hallucination labels on real (non-prompt-injected) RAG outputs from 6
source models. We teacher-force each (query, context, output) tuple
through our 4 models and ask: does our perplexity-based detector flag
hallucinations on externally-annotated data the same way it does on our
synthetic length-controlled benchmarks?

Approach (closest to the paper's existing extract_uncertainty pipeline):
  - For each RAGTruth QA-task example, build the same chat-templated
    prompt as `modal_vary_k.py:extract_uncertainty` (no length bin
    matching, no prompt injection).
  - Teacher-force the existing output, log per-token log-probs.
  - Aggregate to sequence-level perplexity, mean_prob, min_prob, plus SDS
    (cosine-distance-style L2 between the all-MiniLM-L6-v2 embedding of
    the output and the embedding of the concatenated context).
  - Binary label: hallucinated = (evident_conflict > 0 OR baseless_info > 0).

Output per cell:
  /results/ragtruth/<our_model>/ragtruth_qa.json
with one record per (rag_id, source_model) pair.

Cost estimate: 4 models × 875 samples = 3500 forward passes, teacher-forced
scoring only, no generation. ~$8-12 on H100s in parallel.
"""

from __future__ import annotations

import modal

APP_NAME = "rehab-ragtruth"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "transformers>=4.50.0,<4.55.0",
        "accelerate>=1.0.0",
        "sentence-transformers>=3.0.0",
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


# Reuse the paper's existing extract_uncertainty prompt scaffold exactly
SCORE_PROMPT = (
    "Answer the question based on the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


OUR_MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]


@app.function(
    gpu="H100",
    volumes={"/results": results_vol},
    secrets=[_hf_secret],
    timeout=3600,
)
def run_ragtruth_score(
    model_name: str,
    n_samples: int | None = None,
    smoke_test: bool = False,
) -> dict:
    """Teacher-force RAGTruth QA outputs through `model_name`, log uncertainty signals."""
    import json
    import os
    import time
    import numpy as np
    import torch
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    hf_token = os.environ.get("HF_TOKEN") or None
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print(f"[load] RAGTruth wandb/RAGTruth-processed (test split, QA, quality=good)")
    ds = load_dataset("wandb/RAGTruth-processed", split="test")
    qa = ds.filter(lambda r: r["task_type"] == "QA" and r["quality"] == "good")
    if smoke_test:
        qa = qa.select(range(20))
    elif n_samples is not None:
        qa = qa.select(range(min(n_samples, len(qa))))
    print(f"[load] {len(qa)} QA samples to score")

    print(f"[load] embedder all-MiniLM-L6-v2")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
    print(f"[load] gpu vram = {gpu_gb:.1f} GB")

    def score_one(question: str, context: str, output: str) -> dict:
        if not output or not output.strip():
            return {"skipped": True}
        prompt_text = SCORE_PROMPT.format(context=context, question=question)
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = input_text + output
        prompt_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=3500).to("cuda")
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=3900).to("cuda")
        prompt_len = int(prompt_ids["input_ids"].shape[1])
        full_len = int(full_ids["input_ids"].shape[1])
        ans_len = full_len - prompt_len
        if ans_len <= 0:
            return {"skipped": True}
        with torch.no_grad():
            out = model(**full_ids)
            logits = out.logits
        ans_logits = logits[0, prompt_len - 1 : full_len - 1, :]
        ans_tokens = full_ids["input_ids"][0, prompt_len:full_len]
        probs = torch.softmax(ans_logits.float(), dim=-1)
        token_probs = probs[range(ans_len), ans_tokens].cpu().numpy()
        token_probs = token_probs[np.isfinite(token_probs) & (token_probs > 0)]
        if len(token_probs) == 0:
            return {"skipped": True}
        log_probs = np.log(token_probs + 1e-10)
        ppl = float(np.exp(-np.mean(log_probs)))
        mean_p = float(np.mean(token_probs))
        min_p = float(np.min(token_probs))
        # SDS = L2 distance between SBERT embeddings of (context, output)
        ctx_emb = embedder.encode([context])[0]
        out_emb = embedder.encode([output])[0]
        sds = float(np.linalg.norm(out_emb - ctx_emb))
        return {
            "perplexity": ppl,
            "mean_prob": mean_p,
            "min_prob": min_p,
            "sds": sds,
            "n_tokens": int(len(token_probs)),
        }

    records = []
    skipped = 0
    t0 = time.time()
    for i, ex in enumerate(tqdm(qa)):
        try:
            scores = score_one(ex["query"], ex["context"], ex["output"])
            if scores.get("skipped"):
                skipped += 1
                continue
            labels = ex.get("hallucination_labels_processed") or {}
            ec = int(labels.get("evident_conflict", 0))
            bi = int(labels.get("baseless_info", 0))
            records.append({
                "ragtruth_id": str(ex["id"]),
                "source_model": ex["model"],
                "task_type": ex["task_type"],
                "evident_conflict": ec,
                "baseless_info": bi,
                "hallucinated": int(ec > 0 or bi > 0),
                **scores,
            })
        except Exception as e:
            skipped += 1
            print(f"[warn] sample {i} failed: {type(e).__name__}: {e}")

    model_slug = model_name.replace("/", "_")
    out_dir = f"/results/ragtruth/{model_slug}"
    os.makedirs(out_dir, exist_ok=True)
    out_name = "ragtruth_qa_smoke.json" if smoke_test else "ragtruth_qa.json"
    out_path = f"{out_dir}/{out_name}"
    summary = {
        "model": model_name,
        "n_samples_requested": len(qa),
        "n_kept": len(records),
        "n_skipped": skipped,
        "pos_rate": float(sum(r["hallucinated"] for r in records) / max(1, len(records))),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    results_vol.commit()
    print(f"[save] {len(records)} records -> {out_path}")
    print(f"[stats] hallucination rate = {summary['pos_rate']:.3f}")
    return summary


@app.local_entrypoint()
def smoke(model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
    """20-sample smoke on Phi-3 to validate the pipeline before the full sweep."""
    result = run_ragtruth_score.remote(model_name, smoke_test=True)
    import json as _json
    print(_json.dumps(result, indent=2))


@app.local_entrypoint()
def main(model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
    """Score the full RAGTruth QA test split (~875 samples) with one model."""
    result = run_ragtruth_score.remote(model_name)
    import json as _json
    print(_json.dumps(result, indent=2))


@app.local_entrypoint()
def run_all_cells():
    """Dispatch all 4 of our models in parallel on H100s."""
    handles = [(m, run_ragtruth_score.spawn(m)) for m in OUR_MODELS]
    print(f"[dispatched] {len(handles)} runs")
    out = []
    for m, h in handles:
        r = h.get()
        print(f"[done] {m}: n_kept={r['n_kept']}, pos_rate={r['pos_rate']:.3f}, elapsed={r['elapsed_seconds']}s")
        out.append(r)
    import json as _json
    print(_json.dumps(out, indent=2))
