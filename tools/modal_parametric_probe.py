"""Modal runner for the parametric-generation probe.

This probe measures confidence in the model's own no-context answer.
The separate modal_memorization_probe.py instead scores the gold answer
without retrieved context.

For every (question, gold_answer) pair across the 8-cell grid, this probe:
  1. Generates an answer WITHOUT retrieved context (same chat template as
     `modal_vary_k.py`, but the Context block is omitted).
  2. Scores that generated answer's perplexity / mean_prob / min_prob via the
     same teacher-forced extraction as the existing pipeline.

Hypothesis:
  cells with low parametric_ppl (model is highly confident in its own
  parametric answer) are the cells where perplexity AUROC reverses
  under retrieval degradation at k=1, because under k=1 the model
  reproduces the same parametric answer with the same low PPL, regardless
  of whether the gold answer is supported by the (degraded) context.

Output: one JSON per (model, dataset) at
  /results/parametric/<model_slug>/<dataset>/paramprobe_n{N}.json

Each record: {id, question, gold_answer, parametric_answer,
              parametric_perplexity, parametric_mean_prob, parametric_min_prob}.

Usage:
  modal deploy tools/modal_parametric_probe.py

  modal run tools/modal_parametric_probe.py::main \\
      --model-name "Qwen/Qwen2.5-14B-Instruct" --dataset squad --n-samples 100

  modal run tools/modal_parametric_probe.py::run_all_cells --n-samples 100
"""

from __future__ import annotations

import modal

APP_NAME = "rehab-paramprobe"

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


# Use exactly the same wording as `modal_vary_k.py:generate_answer` minus
# the Context block. This is the only correct ablation for comparing
# parametric_ppl against k=1 generated_ppl — anything else introduces a
# wording confound.
NO_CONTEXT_GEN_PROMPT = (
    "Answer the question. Be concise (1-2 sentences).\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
NO_CONTEXT_SCORE_PROMPT = (
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
def run_parametric_probe(
    model_name: str,
    dataset: str,
    n_samples: int = 100,
    seed: int = 42,
) -> dict:
    """Generate without context, then score the generated answer's PPL."""
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

    # Load dataset, no retrieval needed
    if dataset == "hotpot":
        qa_dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    elif dataset == "squad":
        qa_dataset = load_dataset("squad", split="validation")
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")
    print(f"[load] qa={len(qa_dataset)} questions for dataset={dataset}")

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

    def generate_no_context(question: str, max_new_tokens: int = 100) -> str:
        messages = [{"role": "user", "content": NO_CONTEXT_GEN_PROMPT.format(question=question)}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
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

    def score_answer(question: str, answer: str) -> tuple[float, float, float]:
        if not answer or not answer.strip():
            return float("nan"), float("nan"), float("nan")
        messages = [{"role": "user", "content": NO_CONTEXT_SCORE_PROMPT.format(question=question)}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + answer
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        full_ids = tokenizer(full_text, return_tensors="pt").to("cuda")
        prompt_len = prompt_ids["input_ids"].shape[1]
        full_len = full_ids["input_ids"].shape[1]
        ans_len = full_len - prompt_len
        if ans_len <= 0:
            return float("nan"), float("nan"), float("nan")
        with torch.no_grad():
            out = model(**full_ids)
            logits = out.logits
        ans_logits = logits[0, prompt_len - 1 : full_len - 1, :]
        ans_tokens = full_ids["input_ids"][0, prompt_len:full_len]
        probs = torch.softmax(ans_logits.float(), dim=-1)
        token_probs = probs[range(ans_len), ans_tokens].cpu().numpy()
        token_probs = token_probs[np.isfinite(token_probs) & (token_probs > 0)]
        if len(token_probs) == 0:
            return float("nan"), float("nan"), float("nan")
        log_probs = np.log(token_probs + 1e-10)
        return float(np.exp(-np.mean(log_probs))), float(np.mean(token_probs)), float(np.min(token_probs))

    np.random.seed(seed)
    indices = np.random.choice(len(qa_dataset), n_samples, replace=False)

    model_slug = model_name.replace("/", "_")
    out_dir = f"/results/parametric/{model_slug}/{dataset}"
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
            else:
                gold = item.get("answers", {}).get("text", [""])[0]
            param_answer = generate_no_context(question)
            param_ppl, param_mp, param_mn = score_answer(question, param_answer)
            records.append({
                "id": i,
                "ds_index": int(idx),
                "question": question,
                "gold_answer": gold,
                "parametric_answer": param_answer,
                "parametric_perplexity": param_ppl,
                "parametric_mean_prob": param_mp,
                "parametric_min_prob": param_mn,
            })
        except Exception as e:
            skipped += 1
            print(f"[warn] sample idx={int(idx)} (i={i}) failed: {type(e).__name__}: {e}")
            continue

    finite = [r["parametric_perplexity"] for r in records
              if r["parametric_perplexity"] is not None and math.isfinite(r["parametric_perplexity"])]
    summary = {
        "model": model_name,
        "dataset": dataset,
        "n_samples": n_samples,
        "n_kept": len(records),
        "n_skipped": skipped,
        "seed": seed,
        "elapsed_seconds": round(time.time() - t0, 2),
        "param_ppl_median": float(np.median(finite)) if finite else None,
        "param_ppl_mean": float(np.mean(finite)) if finite else None,
        "param_ppl_p10": float(np.percentile(finite, 10)) if finite else None,
        "param_ppl_p90": float(np.percentile(finite, 90)) if finite else None,
    }
    out_json = f"{out_dir}/paramprobe_n{n_samples}.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    results_vol.commit()
    print(f"[save] {len(records)} records -> {out_json}")
    print(f"[summary] param_ppl median={summary['param_ppl_median']:.3f} "
          f"p10={summary['param_ppl_p10']:.3f} p90={summary['param_ppl_p90']:.3f}")
    return summary


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
    n_samples: int = 100,
):
    result = run_parametric_probe.remote(model_name, dataset, n_samples)
    import json as _json
    print(_json.dumps(result, indent=2))


@app.local_entrypoint()
def run_all_cells(n_samples: int = 100):
    handles = [
        (m, d, run_parametric_probe.spawn(m, d, n_samples))
        for (m, d) in ALL_CELLS
    ]
    out = []
    for m, d, h in handles:
        r = h.get()
        print(f"[done] {m} / {d} -> median param_ppl = {r.get('param_ppl_median'):.3f}")
        out.append(r)
    import json as _json
    print(_json.dumps(out, indent=2))
