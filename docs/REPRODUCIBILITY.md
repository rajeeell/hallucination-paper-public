# Reproducing the analyses

## Saved-result workflow

The repository includes features, labels, baseline outputs and aggregate
results. These support CPU analysis without running new model generations:

```bash
python -X utf8 tools/verify_icnlsp.py
python -X utf8 analysis/calibration_analysis.py
python -X utf8 analysis/make_perplexity_distribution_fig.py
```

Run these from the repository root after installing `requirements.txt`.
The ICNLSP verifier disables Python network connections and writes
`release_verification/icnlsp_metrics.json`. It recalculates controlled
perplexity and combined-score AUROCs, sample counts, calibration metrics,
natural k=1 perplexity AUROCs and RAGTruth headline scores.

To disable Python network connections when running an original analysis script:

```bash
python -X utf8 tools/offline_run.py tools/verify_paper_metrics.py
python -X utf8 tools/offline_run.py analysis/calibration_analysis.py
python -X utf8 tools/offline_run.py analysis/make_perplexity_distribution_fig.py
```

The wrapper also removes API keys from its process environment. It supports
the listed Python analyses; it is not a sandbox for external programs.

## Evaluation settings

| Analysis | Combined features | Logistic-regression C | Perplexity | Aggregation |
|---|---|---:|---|---|
| Controlled Table 1 | Two or four; see row mapping | 0.1 | Clipped to [1, 100] | Mean of five fold AUROCs |
| Controlled calibration | Four | 1.0 | Raw | Out-of-fold probabilities for ECE/Brier |
| RAGTruth | Four | 1.0 | Raw | Pooled out-of-fold AUROC |

The combined features are semantic drift, perplexity, mean token probability
and minimum token probability. The two-feature variant uses drift and
perplexity. The Table 1 row mapping and uniform four-feature results are in
[reported configurations](COMBINED_SCORE_PROVENANCE.md).

The original preprocessing fits `StandardScaler` on the full feature matrix
before cross-validation, so held-out feature values contribute to the scaling
statistics. Fold-specific preprocessing instead fits the scaler within each
training fold. Use the original settings above when comparing saved results.

Controlled unsupported edits, natural sparse-retrieval answers and RAGTruth
are separate evaluation conditions. The controlled 0.81–0.91 perplexity AUROC
range applies to the length-matched unsupported-edit benchmark. Threshold
performance is measured on the saved labels; deployment thresholds need
validation on the target data.

See [metric reports](VERIFICATION.md) for comparison thresholds, the three
two-feature/four-feature Table 1 comparisons and the separate Llama/HotpotQA
perplexity display difference.

## New generation runs

Use the model-specific run-order documents and
[data, models and prompts](DATA_MODELS_PROMPTS.md). Inference requires model
weights, matching retrieval assets and a GPU; hosted judges and cloud runners
also require provider access.

The included Colab/Kaggle notebooks contain platform-specific mounting and
installation cells. Adapt those for local execution. Optional inference and
cloud environments have not been tested as clean end-to-end installations.
The Python 3.12 requirements cover saved-result analysis. Older notebook cells
pinning Transformers 4.46.3 and newer runner image recipes require separate
dependency setups.

New runs can vary with sampling, model revisions, random states, package
versions and GPU execution. Exact original model revision hashes and a
complete original environment lock are not recorded. Existing saved labels
are used for the CPU analyses; new runs require their own labeling step.

The older `tools/verify_paper_metrics.py` also reads previously exported
summary CSV/JSON files. Those comparisons check stored summaries rather than
recalculating every baseline or bootstrap interval from per-sample data.

## Retrieval assets

Use a matching corpus/index pair with the same document ordering. An index
lookup returns positions in that corpus; changing the order changes the
retrieved text. Deduplication via `list(set(corpus))` does not give stable
ordering across environments.

Several notebooks have an optional `BUILD_INDEX` cell for SQuAD/TriviaQA but
no HotpotQA construction branch. Use `BUILD_INDEX=False` with existing assets;
building HotpotQA assets needs a separate construction step.
