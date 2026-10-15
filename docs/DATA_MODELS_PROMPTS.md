# Data, models and prompts

## Models

| Model folder | Hugging Face model ID |
|---|---|
| `phi` | `microsoft/Phi-3-mini-4k-instruct` |
| `gemma` | `google/gemma-3-4b-it` |
| `llama 8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| `qwen 14b` | `Qwen/Qwen2.5-14B-Instruct` |

Each folder contains `hotpot` and `squad` experiments. Artifact paths and
controlled results are in `configs/experiments.json`. Model revision hashes
are not recorded; setup and dtype settings are specified in each notebook.
Gemma/Llama access may require accepting provider terms and setting `HF_TOKEN`.

## Datasets and retrieval assets

Controlled generation uses the HotpotQA distractor or SQuAD training split.
The retrieval-depth runner uses validation questions. RAGTruth evaluation
scores fixed QA responses with each of the four models; response-generating
models and scoring models are distinct.

Copy `.env.example` to `.env` and set `WORK_DIR` to the repository root.
Notebooks search parent directories for this file. Retrieval assets go in:

```text
<project root>/retrieval_assets/hotpot_faiss.index
<project root>/retrieval_assets/hotpot_corpus.json
<project root>/retrieval_assets/squad_faiss.index
<project root>/retrieval_assets/squad_corpus.json
```

The original asset pairs have these dimensions and approximate combined size:

| Corpus/index pair | Records | Embedding dimensions |
|---|---:|---:|
| HotpotQA | 899,667 | 384 |
| SQuAD | 18,891 | 384 |

The pairs occupy about 1.93 GB in total, including a 1.38 GB HotpotQA index.
They are external inputs and are not bundled with the source code. Downloading
a dataset alone does not recreate its corpus/index pair: selection, embedding
settings and document ordering must match. See
[retrieval setup](REPRODUCIBILITY.md#retrieval-assets) and
[data terms](../DATA_LICENSES.md).

Semantic embeddings use `all-MiniLM-L6-v2`. The retrieval-depth runner
normalizes query embeddings and searches FAISS for the selected number of
passages. Semantic drift is the L2 distance between answer and context
embeddings; context joining is specified in the notebook code.

## Experiment flow

1. `02_generate_length_controlled*.ipynb` generates faithful and unsupported
   answer candidates.
2. `bin-matching-filter.ipynb`, where present, selects matched word-count bins.
3. `tests/09/` extracts token likelihood features and semantic drift.
4. `tests/10/` compares baselines and logistic-regression combinations.
5. `tests/12/` generates natural sparse-retrieval answers and judge labels.
6. `tests/18, 09's pickle/` runs LettuceDetect comparisons.
7. `analysis/*.py` computes metrics and figures from saved artifacts.

Use each model's `RUN_ORDER_*.md` for notebook order and path/configuration
notes, including dataset and retrieval settings. Paths are relative to their
model/dataset folders; similarly named notebooks can serve different runs.

## Prompts and sampling

Controlled generation prompts are in each
`02_generate_length_controlled*.ipynb`. Other conditions define their own
prompts in their notebooks or runner functions.

In `tools/modal_vary_k.py`, `generate_answer` requests a concise 1–2 sentence
answer based on the context. It uses the model's chat template, temperature
0.3 and at most 100 new tokens, with defaults of seed 42 and 100 samples.
`extract_uncertainty` defines a separate token-scoring prompt. Keep both
prompt definitions when comparing with saved runs.

`tools/label_vary_k.py` defines the Claude judge's `SYSTEM_PROMPT`. The
conservative two-vote rule assigns label 0 only when both votes say faithful;
any hallucinated or unparseable vote assigns label 1. Saved outputs identify
the Sonnet/Opus judge variant used. Natural k=1 notebooks also record Groq
judge identities; inspect run records for the judge associated with a result.

## Compute and service access

Saved-result analysis runs on CPU. New generation and feature extraction
require model weights, datasets and GPU resources. Modal runners provision
cloud compute; Groq/Anthropic labeling uses hosted APIs. Configure provider
credentials in the environment before running those workflows and account
for their compute and API costs.
