# Llama 8B HotpotQA Experiment Run Order

This document describes the run order for the `rehab/llama 8b/hotpot` experiment and the files each notebook expects or produces. The paths below are written relative to this folder unless an absolute Colab path is shown.

## Before running

- Run the notebooks in Google Colab/Kaggle with GPU enabled.
- Make sure the Hugging Face model `meta-llama/Llama-3.1-8B-Instruct` is accessible.
- Copy `.env.example` to `.env` at the repo root and set `WORK_DIR` to the absolute path of this repo. Notebooks load `WORK_DIR` via `python-dotenv` and resolve every path from it (see the README's "Path setup" section). On Colab, set `WORK_DIR=/content/drive/MyDrive/rehab` and place the four retrieval files under `/content/drive/MyDrive/rehab/retrieval_assets/`.
- Required retrieval files for the HotpotQA `02_generate...` generation notebook and natural hallucination tests:
  - `${WORK_DIR}/retrieval_assets/hotpot_faiss.index`
  - `${WORK_DIR}/retrieval_assets/hotpot_corpus.json`

These files are gitignored (too large for GitHub). When sharing the repo, transfer the `retrieval_assets/` folder out of band (zip, drive link, etc.) and drop it back into the repo root before running. Reusing the originals keeps retrieval behavior aligned across machines; rebuild only if the files are missing/corrupt or you intentionally want a different retrieval corpus.

The notebooks install or import packages such as `transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `datasets`, `faiss-cpu`, `sklearn`, `bert-score`, `groq`, `lettucedetect`, and `python-dotenv`.

## Run Sequence

### 1. Generate the length-controlled HotpotQA dataset

Notebook:

`rehab/llama 8b/hotpot/02_generate_length_controlled_latest.ipynb`

Inputs:

- HotpotQA from Hugging Face: `load_dataset("hotpot_qa", "distractor", split="train")`
- FAISS index: `${WORK_DIR}/retrieval_assets/hotpot_faiss.index`
- Corpus JSON: `${WORK_DIR}/retrieval_assets/hotpot_corpus.json`
- Model: `meta-llama/Llama-3.1-8B-Instruct`

Main outputs:

- `checkpoint_non_hall_hotpot_qa.json`
- `checkpoint_hall_hotpot_qa.json`
- `length_distribution_analysis.png`
- `hotpot_qwen25_rag_length_controlled_1000.json`

Note: even though the output filename says `qwen25`, this Llama 8B notebook is configured to load `meta-llama/Llama-3.1-8B-Instruct`.

### 2. Bin-match/filter the generated dataset

Notebook:

`rehab/llama 8b/hotpot/bin-matching-filter.ipynb`

Input:

- `hotpot_qwen25_rag_length_controlled_1000.json`

Main output:

- `hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

This bin-matched JSON is the main dataset used by the later metric/baseline notebooks.

### 3. Run uncertainty + SDS drift experiment

Notebook:

`rehab/llama 8b/hotpot/tests/09/09_uncertainty_drift_kaggledf1c15092b_(3)_(1)_(1).ipynb`

Input:

- `${WORK_DIR}/llama 8b/hotpot/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

Temporary checkpoint:

- `uncertainty_checkpoint.pkl`

Main outputs:

- `hotpot_phi3_uncertainty_results.pkl`
- `hotpot_phi3_uncertainty_results.csv`
- `uncertainty_drift_results.png`

Note: the output filename says `phi3`, but this notebook is also configured to load `meta-llama/Llama-3.1-8B-Instruct`.

### 4. Run baseline comparison

Notebook:

`rehab/llama 8b/hotpot/tests/10/10_baseline_comparison_kaggle_clean (1).ipynb`

Input:

- `${WORK_DIR}/llama 8b/hotpot/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

This notebook does not load the 09 pickle. It reuses the bin-matched dataset and computes baseline metrics such as BERTScore, NLI, SDS, and perplexity.

Main output:

- `hotpot_sds_perplexity_results.pkl`

Important: in the current listed downstream order, this output is not consumed by the LettuceDetect notebook. The LettuceDetect notebook loads the 09 output pickle directly.

### 5. Generate natural hallucination test set

Notebook:

`rehab/llama 8b/hotpot/tests/12/12_natural_hallucination_test.ipynb`

Inputs:

- HotpotQA validation split: `load_dataset("hotpot_qa", "distractor", split="validation")`
- FAISS index: `${WORK_DIR}/retrieval_assets/hotpot_faiss.index`
- Corpus JSON: `${WORK_DIR}/retrieval_assets/hotpot_corpus.json`
- Model: `meta-llama/Llama-3.1-8B-Instruct`

Configured values:

- `N_SAMPLES = 100`
- `DEGRADATION_MODE = "k1"`

Main outputs:

- `natural_hallucination_checkpoint_k1.json`
- `natural_hallucination_test_k1.json`
- `natural_hallucination_labeling_k1.csv`

The JSON output from this notebook is the input to the next LLM-labeling notebook.

### 6. Label natural hallucinations with an LLM judge

Notebook:

`rehab/llama 8b/hotpot/tests/12/super/llm_label_natural_hallucinations.ipynb`

Input:

- `${WORK_DIR}/llama 8b/hotpot/tests/12/natural_hallucination_test_k1.json`

Environment variable required:

- `GROQ_API_KEY`

Configured judge model:

- `llama-3.1-8b-instant`

Main outputs:

- `natural_hallucination_llm_label_checkpoint.json`
- `natural_hallucination_test_k1_LLM_LABELED.json`
- `natural_hallucination_test_k1_LLM_LABELED.csv`

### 7. Run LettuceDetect baseline using the 09 pickle

Notebook:

`rehab/llama 8b/hotpot/tests/18, 09's pickle/18_lettucedetect_baseline (2).ipynb`

Use this `tests/18, 09's pickle` notebook for the documented run. The plain `rehab/llama 8b/hotpot/tests/18` folder contains an alternate/duplicate LettuceDetect notebook and PKL output, but nothing from that plain `tests/18` folder is required by this run order.

Inputs:

- `${WORK_DIR}/llama 8b/hotpot/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`
- `${WORK_DIR}/llama 8b/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl`

Main output:

- `lettucedetect_hotpotqa_results_corrected.pkl`

This confirms your guess: the LettuceDetect notebook uses the pickle generated by the 09 uncertainty drift notebook.

## Dependency Chain Summary

```text
02_generate_length_controlled_latest.ipynb
  -> hotpot_qwen25_rag_length_controlled_1000.json

bin-matching-filter.ipynb
  -> hotpot_qwen25_rag_length_controlled_1000_binmatched.json

tests/09/09_uncertainty_drift_....ipynb
  uses hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  -> hotpot_phi3_uncertainty_results.pkl
  -> hotpot_phi3_uncertainty_results.csv

tests/10/10_baseline_comparison_....ipynb
  uses hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  -> hotpot_sds_perplexity_results.pkl

tests/12/12_natural_hallucination_test.ipynb
  -> natural_hallucination_test_k1.json
  -> natural_hallucination_labeling_k1.csv

tests/12/super/llm_label_natural_hallucinations.ipynb
  uses natural_hallucination_test_k1.json
  -> natural_hallucination_test_k1_LLM_LABELED.json
  -> natural_hallucination_test_k1_LLM_LABELED.csv

tests/18, 09's pickle/18_lettucedetect_baseline (2).ipynb
  uses hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  uses tests/09/hotpot_phi3_uncertainty_results.pkl
  -> lettucedetect_hotpotqa_results_corrected.pkl
```

## Cross-check notes

- The 10 baseline notebook's prerequisite is the bin-matched dataset, not the 09 pickle.
- The 18 LettuceDetect notebook does use the 09 pickle: `tests/09/hotpot_phi3_uncertainty_results.pkl`.
- The plain `tests/18` folder is not used in this documented pipeline; use `tests/18, 09's pickle/18_lettucedetect_baseline (2).ipynb` instead.
- Several filenames still contain older model names such as `qwen25` or `phi3`; the checked Llama 8B notebooks load `meta-llama/Llama-3.1-8B-Instruct`.
- Path resolution goes through `${WORK_DIR}` (loaded from `.env`); to relocate the repo, just edit `.env` instead of touching notebooks.
