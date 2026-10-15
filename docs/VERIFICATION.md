# Checks and metric reports

Run commands from the repository root with the Python 3.12 environment in
`requirements.txt`.

```bash
python -X utf8 tools/check_package.py
python -X utf8 -m pytest -q
python -X utf8 tools/verify_icnlsp.py
```

`check_package.py` validates Python syntax, notebook structure and code cells,
JSON syntax, notebook output state and credential patterns. The tests cover
calibration calculations, threshold selection and judge-output parsing.

## Reference run

Results from 15 October 2026, using Python 3.12.14 and `requirements.txt`:

| Check | Result |
|---|---|
| Python source syntax | 23 files compile |
| Notebook schema and code syntax | 74 notebooks validate and compile |
| JSON parsing | 225 files parse |
| Numerical and judge-parser tests | 10 passed |
| Numeric-pickle loading | 95 artifacts load |
| Controlled notebook results | Both feature variants match saved AUROCs in all eight cells |
| Table 1 configuration mapping | Eight entries match the variants in the configuration CSV at three decimals |
| ICNLSP metric comparisons | 42 PASS, 1 WARN, 3 FAIL |

## Reading the metric report

`tools/verify_icnlsp.py` writes actual values, reference values, deltas and
statuses to `release_verification/icnlsp_metrics.json`.

The three `FAIL` entries compare a uniform four-feature score against the
proceedings entries for Llama/SQuAD, Qwen/HotpotQA and Qwen/SQuAD. Those table
entries use the two-feature variant. The command returns a nonzero exit status
for this comparison. See [reported configurations](COMBINED_SCORE_PROVENANCE.md)
for both variants and the uniform four-feature column. Additional failures or
changes to the saved notebook refits require checking the report's deltas.

The `WARN` entry is the Llama/HotpotQA perplexity display comparison: the
calculated AUROC is 0.882482 and the paper displays 0.883.

### Comparison thresholds

| Comparison | PASS | WARN |
|---|---|---|
| Controlled perplexity vs. paper | Absolute difference ≤0.0005 | Up to 0.001 |
| Controlled combined score vs. paper | Absolute difference ≤0.0005 | None |
| Four-feature refit vs. saved notebook score | Absolute difference ≤1e-10 | None |
| Calibration means and RAGTruth combined AUROC vs. paper | Absolute difference ≤0.005 | Up to 0.020 |
| Natural k=1 and RAGTruth mean PPL AUROCs vs. paper | Absolute difference ≤0.00051 | None |

Sample counts must match exactly. Values outside the configured comparison
thresholds, or missing artifacts, produce `FAIL`. Thresholds describe numeric
comparisons, not confidence intervals; consult the JSON for exact values.

## Calibration and external results

| Metric | Perplexity | Combined |
|---|---:|---:|
| Controlled mean ECE | 0.072362 | 0.035176 |
| Controlled mean Brier score | 0.187053 | 0.149935 |
| RAGTruth mean AUROC | 0.583711 | 0.642786 |

Calibration and RAGTruth use four features, C=1 and raw perplexity. Controlled
Table 1 AUROCs use C=0.1 and perplexity clipped to [1, 100].

`tools/verify_paper_metrics.py` is the older consistency-check script. It uses
earlier table/section numbering, and its controlled refits use C=1 with raw
perplexity. Use `tools/verify_icnlsp.py` for the controlled notebook settings.

These commands operate on saved artifacts. New retrieval, generation, token
scoring and labeling use the separate notebook and inference workflows; see
[reproduction](REPRODUCIBILITY.md).
