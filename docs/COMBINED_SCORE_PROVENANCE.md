# Reported combined-score configurations

The controlled notebooks evaluate two logistic-regression variants:

- **Two features:** semantic drift (SDS) and perplexity.
- **Four features:** semantic drift, perplexity, mean token probability and
  minimum token probability.

Both use perplexity clipped to [1, 100], standardization, C=0.1,
random_state=42 and mean AUROC across five cross-validation folds.
The scaler is fitted before cross-validation; see
[evaluation settings](REPRODUCIBILITY.md#evaluation-settings).

## Table 1 mapping

Five combined-column entries use four features; Llama/SQuAD, Qwen/HotpotQA
and Qwen/SQuAD use drift + perplexity. The method paragraph describes four
features. The table below lists both saved scores and the configuration
matching each reported entry.

| Model / dataset | Published | Four features | Two features | Matching configuration |
|---|---:|---:|---:|---|
| Phi / HotpotQA | 0.842 | 0.841503 | 0.841282 | Four |
| Phi / SQuAD | 0.859 | 0.858757 | 0.861546 | Four |
| Gemma / HotpotQA | 0.911 | 0.911061 | 0.911699 | Four |
| Gemma / SQuAD | 0.868 | 0.867702 | 0.868681 | Four |
| Llama / HotpotQA | 0.846 | 0.846223 | 0.817242 | Four |
| Llama / SQuAD | 0.869 | 0.855127 | 0.868813 | Two |
| Qwen / HotpotQA | 0.824 | 0.816987 | 0.823944 | Two |
| Qwen / SQuAD | 0.878 | 0.870594 | 0.877770 | Two |

The [configuration CSV](../analysis/results/table1_combined_variant_crosswalk.csv)
contains full-precision values and source-notebook paths.

## Uniform four-feature column

Using four features for all eight cells under the controlled notebook
settings gives:

| Model / dataset | Proceedings combined column | Uniform four features |
|---|---:|---:|
| Phi / HotpotQA | 0.842 | 0.842 |
| Phi / SQuAD | 0.859 | 0.859 |
| Gemma / HotpotQA | 0.911 | 0.911 |
| Gemma / SQuAD | 0.868 | 0.868 |
| Llama / HotpotQA | 0.846 | 0.846 |
| Llama / SQuAD | 0.869 | 0.855 |
| Qwen / HotpotQA | 0.824 | 0.817 |
| Qwen / SQuAD | 0.878 | 0.871 |

The uniform column ranges from 0.817 to 0.911. The mean of displayed values
is 0.8586, versus 0.8621 for the proceedings column; the mean of unrounded
uniform scores is 0.858494. The uniform combined score exceeds the strongest
non-uncertainty baseline in Table 1 in all eight cells. PPL exceeds the
uniform combined score in both Qwen cells.

## Calibration and RAGTruth protocols

Calibration and RAGTruth use four features throughout, C=1.0 and raw
perplexity. Calibration uses out-of-fold probabilities for ECE and Brier
score; RAGTruth uses pooled out-of-fold AUROC. Table 1 uses C=0.1, clipped
perplexity and the mean of fold AUROCs.

Implementations:
[`analysis/calibration_analysis.py`](../analysis/calibration_analysis.py) and
[`analysis/ragtruth_analysis.py`](../analysis/ragtruth_analysis.py).

## Recalculate from saved features

```bash
python -X utf8 tools/verify_icnlsp.py
```

The report at `release_verification/icnlsp_metrics.json` includes
`combined_four_feature_notebook_auc` and `combined_two_feature_notebook_auc`
for each controlled cell. Compare these with the CSV and its
`variant_matching_published_rounding` column.

The uniform four-feature comparison reports `FAIL` for Llama/SQuAD,
Qwen/HotpotQA and Qwen/SQuAD and returns a nonzero exit status. These are
the three rows whose reported values use two features. The report includes
both variant refits and checks the four-feature refit against its saved
notebook score. See
[metric reports](VERIFICATION.md) for exact comparison thresholds and statuses.
