"""Modal runner for the vary-k retrieval-degradation experiment.

Replicates the inference logic of `*/hotpot/tests/12/12_natural_hallucination_test.ipynb`
(generate-then-extract-uncertainty over the validation split, with bf16 model
load, all-MiniLM-L6-v2 embedder, FAISS retrieval) but parameterised by the
retrieval-k value so we can map the regime-transition curve from k=1 (the
existing natural-hallucination point) through k=2, k=3, k=4 toward k=5 (the
existing controlled point).

Output: one pickle + one JSON per (model, dataset, k) cell, written to the
`rehab-results` Modal volume under `vary_k/<model_slug>/<dataset>/`.

Usage:
  # Register the app (no GPU cost):
  modal deploy tools/modal_vary_k.py

  # Run a single cell (3 k values, ~100 samples each, on one A100):
  modal run tools/modal_vary_k.py::main \\
      --model-name "google/gemma-3-4b-it" \\
      --dataset hotpot \\
      --k-values "2,3,4" \\
      --n-samples 100

  # Run all 3 target cells in parallel:
  modal run tools/modal_vary_k.py::run_all_cells --n-samples 100

Prerequisites (one-time):
  modal volume create rehab-retrieval
  modal volume create rehab-results
  modal volume put rehab-retrieval retrieval_assets/hotpot_faiss.index    /hotpot_faiss.index
  modal volume put rehab-retrieval retrieval_assets/hotpot_corpus.json    /hotpot_corpus.json
  modal volume put rehab-retrieval retrieval_assets/squad_faiss.index     /squad_faiss.index
  modal volume put rehab-retrieval retrieval_assets/squad_corpus.json     /squad_corpus.json
  modal secret create huggingface HF_TOKEN=hf_xxxxx   # only needed for gated models
"""

from __future__ import annotations

import modal

APP_NAME = "rehab-vary-k"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Pinned to a known-compatible pair: transformers 4.50+ needs torch's
        # newer torch.library.custom_op type schema, which lands in 2.5+.
        "torch>=2.5.0",
        "transformers>=4.50.0,<4.55.0",
        "accelerate>=1.0.0",
        "sentence-transformers>=3.0.0",
        "faiss-cpu",
        "datasets>=2.20.0",
        "scikit-learn",
        "numpy<2",
        "pandas",
        "tqdm",
        "huggingface_hub>=0.24.0",
    )
)

app = modal.App(APP_NAME, image=image)

retrieval_vol = modal.Volume.from_name("rehab-retrieval", create_if_missing=True)
results_vol = modal.Volume.from_name("rehab-results", create_if_missing=True)


# HF token passthrough: reads HF_TOKEN from the caller's local environment at
# `modal deploy` / `modal run` time and forwards it as an inline secret. If
# HF_TOKEN isn't set locally, the secret carries an empty value, which is fine
# for non-gated models (Qwen-2.5, Phi-3). Gated models (gemma, llama) need
# HF_TOKEN exported locally before invoking, e.g. via the repo's .env file.
import os as _os
_hf_secret = modal.Secret.from_dict({"HF_TOKEN": _os.environ.get("HF_TOKEN", "")})


@app.function(
    gpu="H100",
    volumes={
        "/retrieval": retrieval_vol,
        "/results": results_vol,
    },
    secrets=[_hf_secret],
    timeout=3600,
)
def run_vary_k(
    model_name: str,
    dataset: str,
    k_values: list[int],
    n_samples: int = 100,
    seed: int = 42,
) -> dict:
    """Run the natural-hallucination pipeline at multiple k values.

    Loads the model once, then loops through each k value. Output written to
    the `rehab-results` volume.
    """
    import os
    import json
    import pickle
    import faiss
    import numpy as np
    import torch
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    hf_token = os.environ.get("HF_TOKEN") or None
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # --- load retrieval assets and dataset ---
    print(f"[load] retrieval assets for dataset={dataset}")
    if dataset == "hotpot":
        index = faiss.read_index("/retrieval/hotpot_faiss.index")
        with open("/retrieval/hotpot_corpus.json") as f:
            corpus = json.load(f)
        qa_dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    elif dataset == "squad":
        index = faiss.read_index("/retrieval/squad_faiss.index")
        with open("/retrieval/squad_corpus.json") as f:
            corpus = json.load(f)
        qa_dataset = load_dataset("squad", split="validation")
    else:
        raise ValueError(f"unknown dataset: {dataset!r} (expected 'hotpot' or 'squad')")
    print(f"[load] index={index.ntotal} vecs, corpus={len(corpus)} paragraphs, qa={len(qa_dataset)} questions")

    print("[load] sentence-transformer embedder")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"[load] model {model_name}")
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[load] model on gpu; total vram={gpu_gb:.1f} GB")

    # --- inference helpers (replicated from tests/12 notebook) ---
    def retrieve_topk(query: str, k: int) -> list[str]:
        q_emb = embedder.encode([query])
        faiss.normalize_L2(q_emb)
        _, indices = index.search(np.array(q_emb), k)
        return [corpus[i] for i in indices[0]]

    def generate_answer(question: str, context, max_new_tokens: int = 100) -> str:
        context_str = "\n\n".join(context) if isinstance(context, list) else context
        prompt = (
            "Answer the question based on the context below. Be concise (1-2 sentences).\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def extract_uncertainty(question: str, context, answer: str) -> tuple[float, float, float]:
        context_str = "\n\n".join(context) if isinstance(context, list) else context
        prompt = (
            "Answer the question based on the context below.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = input_text + answer
        inputs = tokenizer(full_text, return_tensors="pt").to("cuda")
        prompt_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
        prompt_len = prompt_ids["input_ids"].shape[1]
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits
        answer_logits = logits[0, prompt_len - 1:-1, :]
        answer_tokens = inputs["input_ids"][0, prompt_len:]
        if len(answer_tokens) == 0:
            return 1.0, 1.0, 1.0
        probs = torch.softmax(answer_logits.float(), dim=-1)
        token_probs = probs[range(len(answer_tokens)), answer_tokens].cpu().numpy()
        mean_prob = float(np.mean(token_probs))
        min_prob = float(np.min(token_probs))
        log_probs = np.log(token_probs + 1e-10)
        perplexity = float(np.exp(-np.mean(log_probs)))
        return perplexity, mean_prob, min_prob

    # same sample indices across all k values so the curve is paired
    np.random.seed(seed)
    indices = np.random.choice(len(qa_dataset), n_samples, replace=False)

    model_slug = model_name.replace("/", "_")
    out_dir = f"/results/vary_k/{model_slug}/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    cell_summary = {
        "model": model_name,
        "dataset": dataset,
        "k_values": list(k_values),
        "n_samples": n_samples,
        "seed": seed,
        "cells": [],
    }

    for k in k_values:
        print(f"\n[run] k={k} on {model_name} / {dataset}")
        samples = []
        skipped = 0
        for i, idx in enumerate(tqdm(indices)):
            try:
                item = qa_dataset[int(idx)]
                question = item["question"]
                if dataset == "hotpot":
                    gold = item.get("answer", "")
                else:  # squad
                    gold = item.get("answers", {}).get("text", [""])[0]

                context = retrieve_topk(question, k)
                answer = generate_answer(question, context)
                ppl, mp, mn = extract_uncertainty(question, context, answer)
                ctx_emb = embedder.encode([" ".join(context)])[0]
                ans_emb = embedder.encode([answer])[0]
                sds = float(np.linalg.norm(ans_emb - ctx_emb))
                samples.append({
                    "id": i,
                    "question": question,
                    "gold_answer": gold,
                    "context": context,
                    "generated_answer": answer,
                    "retrieval_k": k,
                    "perplexity": ppl,
                    "mean_prob": mp,
                    "min_prob": mn,
                    "sds": sds,
                    "hallucinated": None,  # filled by separate llm-judge step
                })
            except Exception as e:
                skipped += 1
                print(f"[warn] sample idx={int(idx)} (i={i}) failed: {type(e).__name__}: {e}")
                continue
        if skipped:
            print(f"[warn] skipped {skipped}/{len(indices)} samples at k={k}")

        out_json = f"{out_dir}/vary_k_n{n_samples}_k{k}.json"
        out_pkl = f"{out_dir}/vary_k_n{n_samples}_k{k}.pkl"
        with open(out_json, "w") as f:
            json.dump(samples, f, indent=2)
        with open(out_pkl, "wb") as f:
            pickle.dump(samples, f)
        print(f"[save] {len(samples)} samples -> {out_json}")
        results_vol.commit()
        cell_summary["cells"].append({"k": k, "n": len(samples), "path": out_json})

    summary_path = f"{out_dir}/vary_k_summary.json"
    with open(summary_path, "w") as f:
        json.dump(cell_summary, f, indent=2)
    results_vol.commit()
    print(f"\n[done] summary -> {summary_path}")
    return cell_summary


@app.function(
    volumes={"/retrieval": retrieval_vol, "/results": results_vol},
    timeout=600,
)
def validate_data_pipeline(dataset: str = "hotpot") -> dict:
    """CPU-only sanity check before the expensive GPU run.

    Validates: (i) FAISS index + corpus load correctly from the volume,
    (ii) the HF dataset loads, (iii) FAISS retrieval returns valid corpus
    indices, (iv) corpus[idx] is a non-empty string. Skips model load so
    this costs ~$0.01 of CPU time and finishes in well under a minute.
    """
    import json
    import os
    import time
    import faiss
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    report = {"dataset": dataset, "checks": {}}

    t0 = time.time()
    if dataset == "hotpot":
        idx_path, corp_path, hf_args = (
            "/retrieval/hotpot_faiss.index",
            "/retrieval/hotpot_corpus.json",
            ("hotpot_qa", "distractor"),
        )
    elif dataset == "squad":
        idx_path, corp_path, hf_args = (
            "/retrieval/squad_faiss.index",
            "/retrieval/squad_corpus.json",
            ("squad",),
        )
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")

    report["checks"]["faiss_size_bytes"] = os.path.getsize(idx_path)
    report["checks"]["corpus_size_bytes"] = os.path.getsize(corp_path)

    print(f"[validate] loading FAISS index {idx_path}")
    index = faiss.read_index(idx_path)
    report["checks"]["faiss_ntotal"] = index.ntotal
    report["checks"]["faiss_d"] = index.d

    print(f"[validate] loading corpus {corp_path}")
    with open(corp_path) as f:
        corpus = json.load(f)
    report["checks"]["corpus_type"] = type(corpus).__name__
    report["checks"]["corpus_len"] = len(corpus)
    report["checks"]["corpus_sample_type"] = type(corpus[0]).__name__
    report["checks"]["corpus_sample_preview"] = str(corpus[0])[:120]

    print(f"[validate] loading HF dataset {hf_args}")
    qa = load_dataset(*hf_args, split="validation")
    report["checks"]["qa_len"] = len(qa)
    report["checks"]["qa_fields"] = list(qa[0].keys())
    report["checks"]["qa_sample_question"] = qa[0]["question"][:120]

    print(f"[validate] loading embedder + sanity-checking retrieval")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = embedder.encode([qa[0]["question"]])
    faiss.normalize_L2(q_emb)
    _, retrieved = index.search(np.array(q_emb), 5)
    retrieved_ids = retrieved[0].tolist()
    report["checks"]["retrieval_top5_ids"] = retrieved_ids
    report["checks"]["retrieval_top1_preview"] = str(corpus[retrieved_ids[0]])[:120]

    # write a marker into the results volume to confirm we can write there
    marker_path = f"/results/_validate_marker_{dataset}.json"
    with open(marker_path, "w") as f:
        json.dump(report, f, indent=2)
    results_vol.commit()
    report["checks"]["write_marker"] = marker_path
    report["checks"]["elapsed_seconds"] = round(time.time() - t0, 2)
    print(f"[validate] ok ({report['checks']['elapsed_seconds']}s)")
    return report


@app.local_entrypoint()
def validate(dataset: str = "hotpot"):
    """Run the CPU-only data-pipeline sanity check (~$0.01)."""
    result = validate_data_pipeline.remote(dataset)
    import json as _json
    print(_json.dumps(result, indent=2))


# Model/dataset pairs for the retrieval-depth sweep:
# - gemma-3-4b   + hotpot  (the dramatic perplexity-reversal case)
# - llama-3.1-8b + hotpot  (the mean-prob-wins case)
# - qwen-2.5-14b + squad   (the most extreme regime, 93% hall rate at k=1)
TARGET_CELLS = [
    ("google/gemma-3-4b-it", "hotpot"),
    ("meta-llama/Llama-3.1-8B-Instruct", "hotpot"),
    ("Qwen/Qwen2.5-14B-Instruct", "squad"),
]


@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    dataset: str = "squad",
    k_values: str = "2,3,4",
    n_samples: int = 100,
):
    """Run one (model, dataset) cell across multiple k values.

    Defaults to Qwen-2.5-14B + squad because Qwen is non-gated (no HF token
    required for a first smoke test). For gemma / llama you must set the
    `huggingface` Modal secret first.
    """
    ks = [int(x) for x in k_values.split(",") if x.strip()]
    result = run_vary_k.remote(model_name, dataset, ks, n_samples)
    print("[entrypoint] result summary:")
    print(result)


@app.local_entrypoint()
def run_all_cells(n_samples: int = 100, k_values: str = "2,3,4"):
    """Dispatch all 3 target cells in parallel on separate A100s.

    Requires the `huggingface` Modal secret (gemma + llama are gated).
    """
    ks = [int(x) for x in k_values.split(",") if x.strip()]
    handles = [
        run_vary_k.spawn(model_name, dataset, ks, n_samples)
        for (model_name, dataset) in TARGET_CELLS
    ]
    for (model_name, dataset), h in zip(TARGET_CELLS, handles):
        result = h.get()
        print(f"[done] {model_name} / {dataset}")
        print(result)
