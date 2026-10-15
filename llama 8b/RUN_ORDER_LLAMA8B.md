# Llama 8B Experiment Run Order

This document describes the checked Llama 8B experiment folders under `rehab/llama 8b`. Llama has both `hotpot` and `squad` runs.

The checked Llama notebooks load:

- `meta-llama/Llama-3.1-8B-Instruct`

Some filenames still contain copied names such as `qwen25`, `phi3`, or `llama32`. Treat the folder path as the real context.

## Before running

- Run the notebooks in Google Colab/Kaggle with GPU enabled.
- Make sure `meta-llama/Llama-3.1-8B-Instruct` is accessible from Hugging Face.
- Copy `.env.example` to `.env` at the repo root and set `WORK_DIR` to the absolute path of this repo. Notebooks load `WORK_DIR` via `python-dotenv` and resolve every path from it (see the README's "Path setup" section). On Colab, set `WORK_DIR=/content/drive/MyDrive/rehab` and place the four retrieval files under `/content/drive/MyDrive/rehab/retrieval_assets/`.
- Common packages used include `transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `datasets`, `faiss-cpu`, `sklearn`, `bert-score`, `groq`, `lettucedetect`, and `python-dotenv`.

For HotpotQA `02_generate...` generation and natural hallucination tests, the notebooks expect:

- `${WORK_DIR}/retrieval_assets/hotpot_faiss.index`
- `${WORK_DIR}/retrieval_assets/hotpot_corpus.json`

For SQuAD `02_generate...` generation, the notebook expects:

- `${WORK_DIR}/retrieval_assets/squad_faiss.index`
- `${WORK_DIR}/retrieval_assets/squad_corpus.json`

These four files are gitignored (too large for GitHub). When sharing the repo, transfer the `retrieval_assets/` folder out of band (zip, drive link, etc.) and drop it back into the repo root before running. Reusing the originals keeps retrieval behavior aligned across machines; rebuild only if the files are missing/corrupt or you intentionally want a different retrieval corpus.

## Llama Hotpot Run Order

There is also a Hotpot-only detail doc:

`rehab/llama 8b/RUN_ORDER_LLAMA8B_HOTPOT.md`

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

### 2. Bin-match/filter the Hotpot dataset

Notebook:

`rehab/llama 8b/hotpot/bin-matching-filter.ipynb`

Input:

- `hotpot_qwen25_rag_length_controlled_1000.json`

Main output:

- `hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

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

### 4. Run baseline comparison

Notebook:

`rehab/llama 8b/hotpot/tests/10/10_baseline_comparison_kaggle_clean (1).ipynb`

Input:

- `${WORK_DIR}/llama 8b/hotpot/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

This notebook does not load the 09 pickle. It reuses the bin-matched dataset and computes baseline metrics such as BERTScore, NLI, SDS, and perplexity.

Main output:

- `hotpot_sds_perplexity_results.pkl`

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

### 6. Label natural hallucinations with an LLM judge

Notebook:

`rehab/llama 8b/hotpot/tests/12/super/llm_label_natural_hallucinations.ipynb`

Input:

- `${WORK_DIR}/llama 8b/hotpot/tests/12/natural_hallucination_test_k1.json`

Environment variable required:

- `GROQ_API_KEY`

Main outputs:

- `natural_hallucination_llm_label_checkpoint.json`
- `natural_hallucination_test_k1_LLM_LABELED.json`
- `natural_hallucination_test_k1_LLM_LABELED.csv`

### 7. Run LettuceDetect baseline using the 09 pickle

Notebook:

`rehab/llama 8b/hotpot/tests/18, 09's pickle/18_lettucedetect_baseline (2).ipynb`

Use this `tests/18, 09's pickle` notebook for the documented Hotpot run. The plain `rehab/llama 8b/hotpot/tests/18` folder loads an alternate/duplicate LettuceDetect result and is not required by this pipeline.

Inputs:

- `${WORK_DIR}/llama 8b/hotpot/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`
- `${WORK_DIR}/llama 8b/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl`

Main output:

- `lettucedetect_hotpotqa_results_corrected.pkl`

## Llama SQuAD Run Order

### 1. Generate the length-controlled SQuAD dataset

Notebook:

`rehab/llama 8b/squad/02_generate_length_controlled_latest (1).ipynb`

Inputs:

- SQuAD from Hugging Face: `load_dataset("squad", split="train")`
- FAISS index: `${WORK_DIR}/retrieval_assets/squad_faiss.index`
- Corpus JSON: `${WORK_DIR}/retrieval_assets/squad_corpus.json`
- Model: `meta-llama/Llama-3.1-8B-Instruct`

Main outputs:

- `checkpoint_non_hall_squad.json`
- `checkpoint_hall_squad.json`
- `length_distribution_analysis.png`
- `hotpot_qwen25_rag_length_controlled_1000.json`

Note: the output filename says `hotpot_qwen25`, but this folder/run is SQuAD with Llama.

### 2. Bin-match/filter the SQuAD dataset

Notebook:

`rehab/llama 8b/squad/bin-matching-filter.ipynb`

Input:

- `hotpot_qwen25_rag_length_controlled_1000.json`

Main output:

- `hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

### 3. Run uncertainty + SDS drift experiment

Notebook:

`rehab/llama 8b/squad/tests/09/09_uncertainty_drift_kaggledf1c15092b_(3)_(1)_(1).ipynb`

Input:

- `${WORK_DIR}/llama 8b/squad/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

Temporary checkpoint:

- `uncertainty_checkpoint.pkl`

Main outputs:

- `hotpot_phi3_uncertainty_results.pkl`
- `hotpot_phi3_uncertainty_results.csv`
- `uncertainty_drift_results.png`

### 4. Run baseline comparison

Notebook:

`rehab/llama 8b/squad/tests/10/10_baseline_comparison_kaggle_clean (1).ipynb`

Input:

- `${WORK_DIR}/llama 8b/squad/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

This notebook does not load the 09 pickle. It reuses the bin-matched dataset.

Main output:

- `hotpot_sds_perplexity_results.pkl`

### 5. Run LettuceDetect baseline using the 09 pickle

Notebook:

`rehab/llama 8b/squad/tests/18, 09's pickle/18_lettucedetect_baseline (2).ipynb`

Inputs:

- `${WORK_DIR}/llama 8b/squad/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`
- `${WORK_DIR}/llama 8b/squad/tests/09/hotpot_phi3_uncertainty_results.pkl`

Main output:

- `lettucedetect_hotpotqa_results_corrected.pkl`

Use this `tests/18, 09's pickle` notebook for the 09-pickle comparison. The plain `rehab/llama 8b/squad/tests/18` folder loads the tests/10 baseline pickle instead, so it is a different/alternate LettuceDetect comparison.

## Dependency Chain Summary

```text
Llama Hotpot:

hotpot/02_generate_length_controlled_latest.ipynb
  -> hotpot_qwen25_rag_length_controlled_1000.json

hotpot/bin-matching-filter.ipynb
  -> hotpot_qwen25_rag_length_controlled_1000_binmatched.json

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

Llama SQuAD:

squad/02_generate_length_controlled_latest (1).ipynb
  -> hotpot_qwen25_rag_length_controlled_1000.json

squad/bin-matching-filter.ipynb
  -> hotpot_qwen25_rag_length_controlled_1000_binmatched.json

squad/tests/09/09_uncertainty_drift_....ipynb
  uses hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  -> hotpot_phi3_uncertainty_results.pkl
  -> hotpot_phi3_uncertainty_results.csv

squad/tests/10/10_baseline_comparison_....ipynb
  uses hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  -> hotpot_sds_perplexity_results.pkl

squad/tests/18, 09's pickle/18_lettucedetect_baseline (2).ipynb
  uses hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  uses tests/09/hotpot_phi3_uncertainty_results.pkl
  -> lettucedetect_hotpotqa_results_corrected.pkl
```

## Cross-check notes

- Llama Hotpot has a separate detailed runbook at `rehab/llama 8b/RUN_ORDER_LLAMA8B_HOTPOT.md`.
- Llama SQuAD has no `tests/12` natural hallucination folder in the current repo snapshot.
- Llama SQuAD has its own `bin-matching-filter.ipynb`; no need to borrow it from another model.
- The 10 baseline notebooks use the bin-matched dataset, not the 09 pickle.
- The `tests/18, 09's pickle` notebooks use the 09 uncertainty pickle.
- The plain `tests/18` folders are alternate LettuceDetect comparisons and are not the documented 09-pickle path.
- Repeated filenames are expected; keep Hotpot and SQuAD files inside their own folders.
