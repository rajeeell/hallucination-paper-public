"""Modal runner for multi-seed k=1 generation across all 8 cells.

Runs the k=1 pipeline with multiple seeds for each model/dataset pair.
The resulting records support comparisons of AUROC variation across seeds
and question samples. The single-seed modal_vary_k.py entrypoint defaults
to seed 42.

Output path includes the seed so files do not collide:
  /results/multiseed_k1/<model_slug>/<dataset>/k1_s{seed}.json
  /results/multiseed_k1/<model_slug>/<dataset>/k1_s{seed}.pkl

Usage:
  modal deploy tools/modal_multiseed_k1.py

  # one cell at one seed:
  modal run tools/modal_multiseed_k1.py::main \\
      --model-name "google/gemma-3-4b-it" --dataset hotpot \\
      --seed 1 --n-samples 100

  # all 8 cells across seeds 1,2,3,4 in parallel (needs HF_TOKEN for gated):
  modal run tools/modal_multiseed_k1.py::run_all_seeds \\
      --seeds "1,2,3,4" --n-samples 100
"""

from __future__ import annotations

import modal

APP_NAME = "rehab-multiseed-k1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "transformers>=4.50.0,<4.55.0",
        "accelerate>=1.0.0",
        "sentence-transformers>=3.0.0",
        "faiss-cpu",
        "datasets>=2.20.0",
        "scikit-learn",
        "numpy<2",
        "tqdm",
        "huggingface_hub>=0.24.0",
    )
)

app = modal.App(APP_NAME, image=image)

retrieval_vol = modal.Volume.from_name("rehab-retrieval", create_if_missing=True)
results_vol = modal.Volume.from_name("rehab-results", create_if_missing=True)

import os as _os
_hf_secret = modal.Secret.from_dict({"HF_TOKEN": _os.environ.get("HF_TOKEN", "")})


@app.function(
    gpu="H100",
    volumes={"/retrieval": retrieval_vol, "/results": results_vol},
    secrets=[_hf_secret],
    timeout=3600,
)
def run_multiseed_k1(
    model_name: str,
    dataset: str,
    seed: int,
    n_samples: int = 100,
) -> dict:
    """Run k=1 generation + uncertainty extraction at one (cell, seed)."""
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
        raise ValueError(f"unknown dataset: {dataset!r}")
    print(f"[load] index={index.ntotal}, corpus={len(corpus)}, qa={len(qa_dataset)}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"[load] model {model_name}")
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token, device_map="auto",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )

    def retrieve_topk(query: str, k: int) -> list:
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
                **inputs, max_new_tokens=max_new_tokens, temperature=0.3,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def extract_uncertainty(question: str, context, answer: str):
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
        answer_logits = logits[0, prompt_len - 1 : -1, :]
        answer_tokens = inputs["input_ids"][0, prompt_len:]
        if len(answer_tokens) == 0:
            return 1.0, 1.0, 1.0
        probs = torch.softmax(answer_logits.float(), dim=-1)
        token_probs = probs[range(len(answer_tokens)), answer_tokens].cpu().numpy()
        log_probs = np.log(token_probs + 1e-10)
        return float(np.exp(-np.mean(log_probs))), float(np.mean(token_probs)), float(np.min(token_probs))

    # Critical: seed numpy AGAIN here so the indices are derived purely from `seed`
    np.random.seed(seed)
    indices = np.random.choice(len(qa_dataset), n_samples, replace=False)

    model_slug = model_name.replace("/", "_")
    out_dir = f"/results/multiseed_k1/{model_slug}/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    samples = []
    skipped = 0
    for i, idx in enumerate(tqdm(indices)):
        try:
            item = qa_dataset[int(idx)]
            question = item["question"]
            gold = item.get("answer", "") if dataset == "hotpot" else item.get("answers", {}).get("text", [""])[0]
            context = retrieve_topk(question, 1)
            answer = generate_answer(question, context)
            ppl, mp, mn = extract_uncertainty(question, context, answer)
            ctx_emb = embedder.encode([" ".join(context)])[0]
            ans_emb = embedder.encode([answer])[0]
            sds = float(np.linalg.norm(ans_emb - ctx_emb))
            samples.append({
                "id": i, "ds_index": int(idx),
                "question": question, "gold_answer": gold,
                "context": context, "generated_answer": answer,
                "retrieval_k": 1, "seed": seed,
                "perplexity": ppl, "mean_prob": mp, "min_prob": mn, "sds": sds,
                "hallucinated": None,
            })
        except Exception as e:
            skipped += 1
            print(f"[warn] sample idx={int(idx)} (i={i}) failed: {type(e).__name__}: {e}")

    if skipped:
        print(f"[warn] skipped {skipped}/{len(indices)} samples")

    out_json = f"{out_dir}/k1_s{seed}.json"
    out_pkl = f"{out_dir}/k1_s{seed}.pkl"
    with open(out_json, "w") as f:
        json.dump(samples, f, indent=2)
    with open(out_pkl, "wb") as f:
        pickle.dump(samples, f)
    results_vol.commit()
    print(f"[save] {len(samples)} samples -> {out_json}")
    return {
        "model": model_name, "dataset": dataset, "seed": seed,
        "n_samples": n_samples, "n_kept": len(samples), "path": out_json,
    }


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
    seed: int = 1,
    n_samples: int = 100,
):
    result = run_multiseed_k1.remote(model_name, dataset, seed, n_samples)
    import json as _json
    print(_json.dumps(result, indent=2))


@app.local_entrypoint()
def run_all_seeds(seeds: str = "1,2,3,4", n_samples: int = 100):
    """Dispatch all 8 cells × len(seeds) seeds in parallel on H100s."""
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    handles = []
    for (m, d) in ALL_CELLS:
        for s in seed_list:
            handles.append((m, d, s, run_multiseed_k1.spawn(m, d, s, n_samples)))
    print(f"[dispatched] {len(handles)} runs")
    results = []
    for m, d, s, h in handles:
        r = h.get()
        print(f"[done] {m} / {d} / seed={s}: kept {r['n_kept']}/{r['n_samples']}")
        results.append(r)
    import json as _json
    print(_json.dumps(results, indent=2))
