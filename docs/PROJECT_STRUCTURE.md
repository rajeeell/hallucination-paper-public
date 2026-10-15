# Project structure

## Experiments

- Experiment notebooks for Phi-3-mini, Gemma-3-4B, Llama-3.1-8B and Qwen2.5-14B.
- Saved features, generated-answer records, labels, baseline results and figures.
- Scripts for calibration, retrieval-depth analysis, external RAGTruth
  evaluation and bootstrap confidence intervals.
- CPU analysis dependencies, metric checks and numerical/parser tests.
- Experiment configuration tables, model and prompt documentation, and citation
  metadata for the research paper and related workshop versions.
- Original software under the MIT license.

## Notebook setup

Notebook paths use `WORK_DIR`; service tokens are read from environment
variables. Notebook cell outputs and execution counts are omitted.
Setup cells use notebook-compatible `%pip` syntax where needed.

Generated figures and analysis exports go under `analysis/figures/` and
`analysis/results/`. Corpora, retrieval indexes and model weights are external
inputs; see [data and model setup](DATA_MODELS_PROMPTS.md).

## File manifest

[`PUBLIC_FILES.tsv`](../PUBLIC_FILES.tsv) lists the size and SHA-256 checksum
of each tracked file, excluding the manifest itself.

See [reported configurations](COMBINED_SCORE_PROVENANCE.md) for the Table 1
feature variants and [metric reports](VERIFICATION.md) for analysis outputs.
