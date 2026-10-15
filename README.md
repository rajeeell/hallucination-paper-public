# RAG Hallucination Detection

**Token uncertainty and semantic drift for detecting unsupported answers in RAG.**

[Rajeel Ahmad Ansari](https://github.com/rajeeell) ·
[Tautik Agrahari](https://github.com/tautik)

[Paper](https://aclanthology.org/2026.icnlsp-1.6/) ·
[Experiment settings](docs/DATA_MODELS_PROMPTS.md) ·
[Reproduction guide](docs/REPRODUCIBILITY.md)

[Overview](#method-overview) · [Setup](#installation) ·
[Saved-result analysis](#saved-result-analysis) ·
[New experiments](#new-experiments) · [Citation](#publications-and-citation)

This project investigates when a model's token probabilities help identify
answers unsupported by retrieved evidence. It combines token-level uncertainty
with answer–context semantic drift, evaluates answer-length effects and
retrieval conditions, and compares discrimination with probability calibration.

Experiments cover **Phi-3-mini, Gemma-3-4B, Llama-3.1-8B and Qwen2.5-14B** on
**HotpotQA and SQuAD**, with external evaluation on **RAGTruth QA**.

## Key results

| Evaluation | Result |
|---|---|
| Controlled, length-matched unsupported edits | Perplexity AUROC **0.81–0.91**; length-only AUROC **0.500–0.530** |
| Controlled probability calibration | Mean ECE **0.072 → 0.035** and Brier score **0.187 → 0.150**, comparing calibrated perplexity with the combined score |
| External RAGTruth QA, 875 examples | Mean AUROC **0.584** for perplexity and **0.643** for the combined score |

These are separate evaluation conditions. The
[protocol guide](docs/REPRODUCIBILITY.md#evaluation-settings) gives the feature
sets, classifier settings and cross-validation procedures for each analysis.

## Installation

Use **Python 3.12** for the saved-result analysis environment.

```bash
git clone https://github.com/rajeeell/hallucination-paper-public.git
cd hallucination-paper-public
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv/Scripts/Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install the analysis dependencies:

```bash
python -m pip install -r requirements.txt
```

Saved-result analysis runs on CPU using the included feature arrays and labels.
GPU generation and hosted labeling use the optional dependencies described below.

## Method overview

```text
Question + retrieved context → Answer → Token uncertainty + semantic drift
                                     → Combined score → AUROC and calibration
```

1. **Token uncertainty:** score a generated answer conditioned on its question
   and retrieved context, then compute perplexity, mean token probability and
   minimum token probability.
2. **Semantic drift:** embed the answer and context with `all-MiniLM-L6-v2` and
   measure their L2 distance.
3. **Combined score:** standardize the features and fit a logistic-regression
   classifier. Evaluate AUROC, calibration and precision/recall thresholds.

The controlled benchmark matches faithful and unsupported answers within
word-count bins. Separate experiments examine naturally generated answers
under sparse retrieval, multiple retrieval depths, multiple random seeds,
alternative uncertainty measures and external RAGTruth responses.

## Saved-result analysis

The included feature arrays and labels are enough for the following CPU
analyses. Run the commands from the repository root after installation.

### 1. Recalculate metrics

Reads the saved controlled features, natural-run labels and RAGTruth scores
using the paths in [`configs/experiments.json`](configs/experiments.json).

```bash
python -X utf8 tools/verify_icnlsp.py
```

Output: `release_verification/icnlsp_metrics.json`, containing per-cell AUROCs,
sample counts, calibration metrics and comparisons with the published tables.

**Table 1 configurations:** Three combined-score entries (Llama/SQuAD,
Qwen/HotpotQA and Qwen/SQuAD) use drift + perplexity. With all four features,
the corresponding AUROCs are **0.855, 0.817 and 0.871**. The verifier compares
the uniform four-feature score with the proceedings table, so those three
comparisons report `FAIL` and the command returns a nonzero exit status.
See the [configuration tables](docs/COMBINED_SCORE_PROVENANCE.md) and
[report format](docs/VERIFICATION.md) for the row mapping and exact values.

### 2. Evaluate calibration and thresholds

Reads the eight model/dataset feature pickles under `tests/09/` and compares
calibrated perplexity with the combined uncertainty-and-drift score.

```bash
python -X utf8 analysis/calibration_analysis.py
```

Outputs:

- `analysis/results/calibration_metrics.csv`: per-cell ECE, Brier scores and thresholds.
- `analysis/results/calibration_summary.json`: aggregate calibration metrics.
- `analysis/figures/calibration_reliability.{pdf,png}`: reliability diagrams.

### 3. Plot perplexity distributions

Reads `gemma/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl` and compares
faithful and unsupported answers in the controlled Gemma/HotpotQA condition.

```bash
python -X utf8 analysis/make_perplexity_distribution_fig.py
```

Output: `analysis/figures/gemma_perplexity_distribution_paper.png`.
Additional scripts in `analysis/` cover retrieval-depth comparisons,
protocol sensitivity, uncertainty baselines and RAGTruth scoring results.

## New experiments

### 1. Configure models and retrieval

Copy [`.env.example`](.env.example) to `.env`, set `WORK_DIR` to the absolute
project path and place matching corpus/index pairs in `retrieval_assets/`.
The [data and model guide](docs/DATA_MODELS_PROMPTS.md) lists the model IDs,
required asset names and provider access.

- `requirements-inference.txt`: GPU inference dependencies.
- `requirements-cloud.txt`: Modal and hosted labeling dependencies.

Generation notebooks contain Colab/Kaggle setup cells. Local execution needs
the appropriate GPU environment and platform-specific setup changes; the
tested requirements above cover CPU analysis. Hosted generation and labeling
use provider APIs and can incur charges. Model weights and retrieval assets
are obtained separately.

### 2. Generate controlled answers and match lengths

Open the model/dataset's `02_generate_length_controlled*.ipynb` to generate
faithful and unsupported answer candidates. Apply `bin-matching-filter.ipynb`,
where present, to select comparable word-count bins. These stages produce
answer records and a bin-matched JSON dataset.

### 3. Extract features and compare detectors

The notebooks in `tests/09/` read the matched answers and save token
uncertainty, semantic drift and labels as pickle/CSV files. The notebooks in
`tests/10/` read the matched answer dataset and compute detector baselines.
The CPU analyses above use the saved outputs.

### 4. Run natural-answer experiments

The `tests/12/` notebooks, where present, generate sparse-retrieval answers
and run judge labeling. Cloud runner scripts under `tools/modal_*.py` cover
additional experiment conditions. Their generation and labeling settings
are described in [models, data and prompts](docs/DATA_MODELS_PROMPTS.md).

Choose the model-specific notebook sequence for exact filenames, input/output
paths and configuration notes:

| Model | Notebook guide |
|---|---|
| Phi-3-mini | [HotpotQA](phi/RUN_ORDER_PHI_HOTPOT.md) |
| Gemma-3-4B | [HotpotQA and SQuAD](gemma/RUN_ORDER_GEMMA.md) |
| Llama-3.1-8B | [HotpotQA](<llama 8b/RUN_ORDER_LLAMA8B_HOTPOT.md>) · [SQuAD](<llama 8b/RUN_ORDER_LLAMA8B.md>) |
| Qwen2.5-14B | [HotpotQA and SQuAD](<qwen 14b/RUN_ORDER_QWEN14B.md>) |

Some experiments reuse filenames containing `qwen25` or `phi3`; use the
folder and the notebook's `model_name` to identify the model.

## Repository layout

```text
phi/             # Phi-3-mini notebooks and results
gemma/           # Gemma-3-4B notebooks and results
llama 8b/        # Llama-3.1-8B notebooks and results
qwen 14b/        # Qwen2.5-14B notebooks and results
analysis/        # Analysis scripts, saved results, tables and figures
tools/           # Generation runners, labeling and metric checks
configs/         # Model IDs, artifact paths and experiment configuration
tests/           # Numerical and judge-parser tests
docs/            # Protocols, setup details and result interpretation
```

## Tests

```bash
python -X utf8 tools/check_package.py
python -X utf8 -m pytest -q
```

These check source/notebook syntax, saved-file structure, calibration
calculations, threshold selection and judge-output parsing.

## Publications and citation

```bibtex
@inproceedings{ansari-agrahari-2026-uncertainty,
  title = "When Does Uncertainty Measure {RAG} Hallucination Risk?",
  author = "Ansari, Rajeel Ahmad and Agrahari, Tautik",
  booktitle = "Proceedings of the 9th International Conference on Natural Language and Speech Processing (ICNLSP-2026)",
  year = "2026",
  pages = "53--58",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2026.icnlsp-1.6/"
}
```

Related non-archival workshop versions of the same research:

- [SeT-LLM at KDD 2026](https://openreview.net/forum?id=hdomcrvFMA)
- [GroundLM at EMNLP 2026](https://openreview.net/forum?id=DNy1xF8Mf9)
- [AIMS at COLM 2026](https://openreview.net/forum?id=7ma98SQfZg)

## License

Original code and documentation are licensed under [MIT](LICENSE).
Dataset-derived records, model access and third-party components retain their
respective terms; see [data licenses](DATA_LICENSES.md).

See [project structure](docs/PROJECT_STRUCTURE.md) for experiment components and file hashes.
