# Qwen 14B Experiment Run Order

This document describes the checked Qwen experiment folders under `rehab/qwen 14b`. Qwen has both `hotpot` and `squad` runs.

The checked Qwen notebooks load:

- `Qwen/Qwen2.5-14B-Instruct`

Some filenames still contain copied names such as `qwen25`, `phi3`, `llama32`, or `hotpot` inside SQuAD folders. Treat the folder path as the real context.

## Before running

- Run the notebooks in Google Colab/Kaggle with GPU enabled.
- Make sure `Qwen/Qwen2.5-14B-Instruct` is accessible from Hugging Face.
- Copy `.env.example` to `.env` at the repo root and set `WORK_DIR` to the absolute path of this repo. Notebooks load `WORK_DIR` via `python-dotenv` and resolve every path from it (see the README's "Path setup" section). On Colab, set `WORK_DIR=/content/drive/MyDrive/rehab` and place the four retrieval files under `/content/drive/MyDrive/rehab/retrieval_assets/`.
- Common packages used include `transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `datasets`, `faiss-cpu`, `sklearn`, `bert-score`, `groq`, `lettucedetect`, and `python-dotenv`.

For HotpotQA `02_generate...` generation and natural hallucination tests, the notebooks expect:

- `${WORK_DIR}/retrieval_assets/hotpot_faiss.index`
- `${WORK_DIR}/retrieval_assets/hotpot_corpus.json`

For SQuAD `02_generate...` generation, the notebook expects:

- `${WORK_DIR}/retrieval_assets/squad_faiss.index`
- `${WORK_DIR}/retrieval_assets/squad_corpus.json`

These four files are gitignored (too large for GitHub). When sharing the repo, transfer the `retrieval_assets/` folder out of band (zip, drive link, etc.) and drop it back into the repo root before running. Reusing the originals keeps retrieval behavior aligned across machines; rebuild only if the files are missing/corrupt or you intentionally want a different retrieval corpus.

## Qwen Hotpot Run Order

### 1. Generate the length-controlled HotpotQA dataset

Notebook:

`rehab/qwen 14b/hotpot/02_generate_length_controlled_latest.ipynb`

There is also:

`rehab/qwen 14b/hotpot/02_generate_length_controlled_latest (1).ipynb`

The `(1)` copy includes extra checkpoint-loading cells for continuing from `checkpoint_non_hall_hotpot_qa_deduplicated.json` and existing hall/non-hall checkpoints. Use the plain notebook for a fresh run; use the `(1)` notebook if you intentionally want to continue from those checkpoint files.

Inputs:

- HotpotQA from Hugging Face: `load_dataset("hotpot_qa", "distractor", split="train")`
- FAISS index: `${WORK_DIR}/retrieval_assets/hotpot_faiss.index`
- Corpus JSON: `${WORK_DIR}/retrieval_assets/hotpot_corpus.json`
- Model: `Qwen/Qwen2.5-14B-Instruct`

Main outputs:

- `checkpoint_non_hall_hotpot_qa.json`
- `checkpoint_hall_hotpot_qa.json`
- `length_distribution_analysis.png`
- `hotpot_qwen25_rag_length_controlled_1000.json`

### 2. Use or regenerate the bin-matched Hotpot dataset

Current checked output:

- `rehab/qwen 14b/hotpot/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

There is no `bin-matching-filter.ipynb` currently present in `rehab/qwen 14b/hotpot`, but the bin-matched JSON already exists. If rerunning bin matching, reuse a matching notebook from another model folder, for example:

- `rehab/llama 8b/hotpot/bin-matching-filter.ipynb`
- `rehab/gemma/hotpot/bin-matching-filter.ipynb`
- `rehab/phi/hotpot/bin-matching-filter.ipynb`

Update `input_file` and output paths for `rehab/qwen 14b/hotpot`.

### 3. Run uncertainty + SDS drift experiment

Notebook:

`rehab/qwen 14b/hotpot/tests/09/09_uncertainty_drift_kaggledf1c15092b_(3)_(1)_(1).ipynb`

Input:

- `${WORK_DIR}/qwen 14b/hotpot/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

Temporary checkpoint:

- `uncertainty_checkpoint.pkl`

Main outputs:

- `hotpot_phi3_uncertainty_results.pkl`
- `hotpot_phi3_uncertainty_results.csv`
- `uncertainty_drift_results.png`

Note: the output filename says `phi3`, but this notebook loads Qwen.

### 4. Run baseline comparison

Notebook:

`rehab/qwen 14b/hotpot/tests/10/10_baseline_comparison_kaggle_clean (1).ipynb`

Input:

- `${WORK_DIR}/qwen 14b/hotpot/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

This notebook does not load the 09 pickle. It reuses the bin-matched dataset and computes baseline metrics such as BERTScore, NLI, SDS, and perplexity.

Main output:

- `hotpot_sds_perplexity_results.pkl`

### 5. Generate natural hallucination test set

Notebook:

`rehab/qwen 14b/hotpot/tests/12/12_natural_hallucination_test.ipynb`

Inputs:

- HotpotQA validation split: `load_dataset("hotpot_qa", "distractor", split="validation")`
- FAISS index: `${WORK_DIR}/retrieval_assets/hotpot_faiss.index`
- Corpus JSON: `${WORK_DIR}/retrieval_assets/hotpot_corpus.json`
- Model: `Qwen/Qwen2.5-14B-Instruct`

Configured values:

- `N_SAMPLES = 100`
- `DEGRADATION_MODE = "k1"`

Main outputs:

- `natural_hallucination_checkpoint_k1.json`
- `natural_hallucination_test_k1.json`
- `natural_hallucination_labeling_k1.csv`

### 6. Label natural hallucinations with an LLM judge

Notebook:

`rehab/qwen 14b/hotpot/tests/12/super/llm_label_natural_hallucinations.ipynb`

Input:

- `${WORK_DIR}/qwen 14b/hotpot/tests/12/natural_hallucination_test_k1.json`

Environment variable required:

- `GROQ_API_KEY`

Main outputs:

- `natural_hallucination_llm_label_checkpoint.json`
- `natural_hallucination_test_k1_LLM_LABELED.json`
- `natural_hallucination_test_k1_LLM_LABELED.csv`

Important security note: this notebook contains older cells with a hard-coded Groq key. Remove or skip those cells before sharing/rerunning, and use `GROQ_API_KEY` from the environment instead.

### 7. Run LettuceDetect baseline using the 09 pickle

Notebook:

`rehab/qwen 14b/hotpot/tests/18, 09's pickle/18_lettucedetect_baseline (2).ipynb`

Use this `tests/18, 09's pickle` notebook for the documented Hotpot run. The plain `rehab/qwen 14b/hotpot/tests/18` folder loads the `tests/10/hotpot_sds_perplexity_results.pkl` baseline result instead, so it is a different/alternate LettuceDetect comparison and is not the same as the 09-pickle pipeline.

Inputs:

- `${WORK_DIR}/qwen 14b/hotpot/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`
- `${WORK_DIR}/qwen 14b/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl`

Main output:

- `lettucedetect_hotpotqa_results_corrected.pkl`

## Qwen SQuAD Run Order

### 1. Generate the length-controlled SQuAD dataset

Notebook:

`rehab/qwen 14b/squad/02_generate_length_controlled_kaggle3e225260ed_(5)_(2) (2).ipynb`

Inputs:

- SQuAD from Hugging Face: `load_dataset("squad", split="train")`
- FAISS index: `${WORK_DIR}/retrieval_assets/squad_faiss.index`
- Corpus JSON: `${WORK_DIR}/retrieval_assets/squad_corpus.json`
- Model: `Qwen/Qwen2.5-14B-Instruct`

Main outputs:

- `checkpoint_non_hall_squad.json`
- `checkpoint_hall_squad.json`
- `length_distribution_analysis.png`
- `hotpot_qwen25_rag_length_controlled_1000.json`

### 2. Create the bin-matched SQuAD dataset

Current checked bin-matched output:

- `rehab/qwen 14b/squad/good p/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

The downstream SQuAD notebooks expect the bin-matched dataset under this `good p` path, so this folder/file should be kept if you want to run those notebooks without path edits.

There is no `bin-matching-filter.ipynb` currently present in `rehab/qwen 14b/squad`. If rerunning bin matching from the generated JSON, reuse a `bin-matching-filter.ipynb` from another model folder, for example:

- `rehab/llama 8b/hotpot/bin-matching-filter.ipynb`
- `rehab/gemma/hotpot/bin-matching-filter.ipynb`
- `rehab/phi/hotpot/bin-matching-filter.ipynb`

Update `input_file` and output paths for the Qwen SQuAD dataset. Recommended output path matching the existing notebooks:

- `rehab/qwen 14b/squad/good p/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

Alternative cleaner output path if you also update the downstream notebook paths:

- `rehab/qwen 14b/squad/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

### 3. Run uncertainty + SDS drift experiment

Notebook:

`rehab/qwen 14b/squad/tests/09/09_uncertainty_drift_kaggledf1c15092b_(3)_(1)_(1).ipynb`

Current hard-coded input:

- `${WORK_DIR}/qwen 14b/squad/good p/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

If you store the bin-matched JSON directly under `rehab/qwen 14b/squad`, update the notebook path before running.

Temporary checkpoint:

- `uncertainty_checkpoint.pkl`

Main outputs:

- `hotpot_phi3_uncertainty_results.pkl`
- `hotpot_phi3_uncertainty_results.csv`
- `uncertainty_drift_results.png`

### 4. Run baseline comparison

Notebook:

`rehab/qwen 14b/squad/tests/10/10_baseline_comparison_kaggle_clean (1).ipynb`

Current hard-coded input:

- `${WORK_DIR}/qwen 14b/squad/good p/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

Main output:

- `hotpot_sds_perplexity_results.pkl`

### 5. Generate natural hallucination test set

Notebook:

`rehab/qwen 14b/squad/tests/12/12_natural_hallucination_test.ipynb`

Inputs in the checked notebook:

- FAISS index: `${WORK_DIR}/retrieval_assets/squad_faiss.index`
- Corpus JSON: `${WORK_DIR}/retrieval_assets/squad_corpus.json`
- Model: `Qwen/Qwen2.5-14B-Instruct`

Path/config warning: this notebook uses SQuAD index/corpus paths but calls `load_dataset("hotpot_qa", "distractor", split="validation")`. Verify whether that is intentional before rerunning. If this should be a true SQuAD natural hallucination test, update the dataset-loading cell.

Configured values:

- `N_SAMPLES = 100`
- `DEGRADATION_MODE = "k1"`

Main outputs:

- `natural_hallucination_checkpoint_k1.json`
- `natural_hallucination_test_k1.json`
- `natural_hallucination_labeling_k1.csv`

### 6. Label natural hallucinations with an LLM judge

Notebook:

`rehab/qwen 14b/squad/tests/12/super/llm_label_natural_hallucinations.ipynb`

Input:

- `${WORK_DIR}/qwen 14b/squad/tests/12/natural_hallucination_test_k1.json`

Environment variable required:

- `GROQ_API_KEY`

Main outputs:

- `natural_hallucination_llm_label_checkpoint.json`
- `natural_hallucination_test_k1_LLM_LABELED.json`
- `natural_hallucination_test_k1_LLM_LABELED.csv`

Important security note: this notebook contains older cells with a hard-coded Groq key. Remove or skip those cells before sharing/rerunning, and use `GROQ_API_KEY` from the environment instead.

### 7. Run LettuceDetect baseline using the 09 pickle

Notebook:

`rehab/qwen 14b/squad/tests/18, 09's pickle/18_lettucedetect_baseline (2).ipynb`

Inputs:

- `${WORK_DIR}/qwen 14b/squad/good p/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`
- `${WORK_DIR}/qwen 14b/squad/tests/09/hotpot_phi3_uncertainty_results.pkl`

If you store the bin-matched JSON directly under `rehab/qwen 14b/squad`, update `DATASET_PATH` before running.

Main output:

- `lettucedetect_hotpotqa_results_corrected.pkl`

## Dependency Chain Summary

```text
Qwen Hotpot:

hotpot/02_generate_length_controlled_latest.ipynb
  -> hotpot_qwen25_rag_length_controlled_1000.json

hotpot/hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  -> existing bin-matched dataset

hotpot/tests/09/09_uncertainty_drift_....ipynb
  uses hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  -> hotpot_phi3_uncertainty_results.pkl
  -> hotpot_phi3_uncertainty_results.csv

hotpot/tests/10/10_baseline_comparison_....ipynb
  uses hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  -> hotpot_sds_perplexity_results.pkl

hotpot/tests/12/12_natural_hallucination_test.ipynb
  -> natural_hallucination_test_k1.json
  -> natural_hallucination_labeling_k1.csv

hotpot/tests/12/super/llm_label_natural_hallucinations.ipynb
  uses natural_hallucination_test_k1.json
  -> natural_hallucination_test_k1_LLM_LABELED.json
  -> natural_hallucination_test_k1_LLM_LABELED.csv

hotpot/tests/18, 09's pickle/18_lettucedetect_baseline (2).ipynb
  uses hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  uses tests/09/hotpot_phi3_uncertainty_results.pkl
  -> lettucedetect_hotpotqa_results_corrected.pkl

Qwen SQuAD:

squad/02_generate_length_controlled_kaggle3e225260ed_(5)_(2) (2).ipynb
  -> hotpot_qwen25_rag_length_controlled_1000.json

good p/hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  -> existing bin-matched dataset used by downstream notebooks

squad/tests/09/09_uncertainty_drift_....ipynb
  uses good p/hotpot_qwen25_rag_length_controlled_1000_binmatched.json by default
  -> hotpot_phi3_uncertainty_results.pkl
  -> hotpot_phi3_uncertainty_results.csv

squad/tests/10/10_baseline_comparison_....ipynb
  uses good p/hotpot_qwen25_rag_length_controlled_1000_binmatched.json by default
  -> hotpot_sds_perplexity_results.pkl

squad/tests/12/12_natural_hallucination_test.ipynb
  -> natural_hallucination_test_k1.json
  -> natural_hallucination_labeling_k1.csv

squad/tests/12/super/llm_label_natural_hallucinations.ipynb
  uses natural_hallucination_test_k1.json
  -> natural_hallucination_test_k1_LLM_LABELED.json
  -> natural_hallucination_test_k1_LLM_LABELED.csv

squad/tests/18, 09's pickle/18_lettucedetect_baseline (2).ipynb
  uses good p/hotpot_qwen25_rag_length_controlled_1000_binmatched.json by default
  uses tests/09/hotpot_phi3_uncertainty_results.pkl
  -> lettucedetect_hotpotqa_results_corrected.pkl
```

## Cross-check notes

- Qwen Hotpot already has a bin-matched JSON but no local `bin-matching-filter.ipynb`.
- Qwen SQuAD has a checked bin-matched JSON under `good p`, and the downstream notebooks point there.
- If you prefer the bin-matched JSON directly under `rehab/qwen 14b/squad`, update the notebook paths accordingly.
- Qwen Hotpot plain `tests/18` uses the tests/10 baseline pickle, while `tests/18, 09's pickle` uses the tests/09 uncertainty pickle.
- Qwen Hotpot and SQuAD LLM-label notebooks contain old hard-coded Groq-key cells; remove/skip those and use `GROQ_API_KEY`.
- Qwen SQuAD natural hallucination test uses SQuAD index/corpus paths but HotpotQA validation data in the checked notebook; verify before rerunning.
- Repeated filenames are expected; keep Hotpot and SQuAD files inside their own folders.
