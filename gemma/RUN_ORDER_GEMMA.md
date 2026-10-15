# Gemma Experiment Run Order

This document describes the checked Gemma experiment folders under `rehab/gemma`. Gemma has both `hotpot` and `squad` runs. The Hotpot run is closest to the Llama/Phi Hotpot pipeline; the SQuAD run is similar but does not include a `tests/12` natural hallucination folder.

The checked Gemma notebooks load:

- `google/gemma-3-4b-it`

Some filenames still contain older copied names such as `qwen25`, `phi3`, `llama32`, or `hotpot` inside SQuAD folders. Treat the folder path as the real context.

## Before running

- Run the notebooks in Google Colab/Kaggle with GPU enabled.
- Make sure `google/gemma-3-4b-it` is accessible from Hugging Face.
- Copy `.env.example` to `.env` at the repo root and set `WORK_DIR` to the absolute path of this repo. Notebooks load `WORK_DIR` via `python-dotenv` and resolve every path from it (see the README's "Path setup" section). On Colab, set `WORK_DIR=/content/drive/MyDrive/rehab` and place the four retrieval files under `/content/drive/MyDrive/rehab/retrieval_assets/`.
- Common packages used include `transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `datasets`, `faiss-cpu`, `sklearn`, `bert-score`, `groq`, `lettucedetect`, and `python-dotenv`.

For HotpotQA `02_generate...` generation and natural hallucination tests, the notebooks expect:

- `${WORK_DIR}/retrieval_assets/hotpot_faiss.index`
- `${WORK_DIR}/retrieval_assets/hotpot_corpus.json`

For SQuAD `02_generate...` generation, the notebook expects:

- `${WORK_DIR}/retrieval_assets/squad_faiss.index`
- `${WORK_DIR}/retrieval_assets/squad_corpus.json`

These four files are gitignored (too large for GitHub). When sharing the repo, transfer the `retrieval_assets/` folder out of band (zip, drive link, etc.) and drop it back into the repo root before running. Reusing the originals keeps retrieval behavior aligned across machines; rebuild only if the files are missing/corrupt or you intentionally want a different retrieval corpus.

## Gemma Hotpot Run Order

### 1. Generate the length-controlled HotpotQA dataset

Notebook:

`rehab/gemma/hotpot/02_generate_length_controlled_latest.ipynb`

Inputs:

- HotpotQA from Hugging Face: `load_dataset("hotpot_qa", "distractor", split="train")`
- FAISS index: `${WORK_DIR}/retrieval_assets/hotpot_faiss.index`
- Corpus JSON: `${WORK_DIR}/retrieval_assets/hotpot_corpus.json`
- Model: `google/gemma-3-4b-it`

Main outputs:

- `checkpoint_non_hall_hotpot_qa.json`
- `checkpoint_hall_hotpot_qa.json`
- `length_distribution_analysis.png`
- `hotpot_qwen25_rag_length_controlled_1000.json`

Note: the output filename says `qwen25`, but the checked notebook loads Gemma.

### 2. Bin-match/filter the generated dataset

Notebook:

`rehab/gemma/hotpot/bin-matching-filter.ipynb`

Input:

- `hotpot_qwen25_rag_length_controlled_1000.json`

Main output:

- `hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

### 3. Run uncertainty + SDS drift experiment

Notebook:

`rehab/gemma/hotpot/tests/09/09_uncertainty_drift_kaggledf1c15092b_(3)_(1)_(1).ipynb`

Input:

- `${WORK_DIR}/gemma/hotpot/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

Temporary checkpoint:

- `uncertainty_checkpoint.pkl`

Main outputs:

- `hotpot_phi3_uncertainty_results.pkl`
- `hotpot_phi3_uncertainty_results.csv`
- `uncertainty_drift_results.png`

Note: the output filename says `phi3`, but this notebook loads Gemma.

### 4. Run baseline comparison

Notebook:

`rehab/gemma/hotpot/tests/10/super/10_baseline_comparison_kaggle_clean (1).ipynb`

Input:

- `${WORK_DIR}/gemma/hotpot/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

This notebook does not load the 09 pickle. It reuses the bin-matched dataset and computes baseline metrics such as BERTScore, NLI, SDS, and perplexity.

Main output:

- `hotpot_sds_perplexity_results.pkl`

### 5. Generate natural hallucination test set

Notebook:

`rehab/gemma/hotpot/tests/12/12_natural_hallucination_test.ipynb`

Inputs:

- HotpotQA validation split: `load_dataset("hotpot_qa", "distractor", split="validation")`
- FAISS index: `${WORK_DIR}/retrieval_assets/hotpot_faiss.index`
- Corpus JSON: `${WORK_DIR}/retrieval_assets/hotpot_corpus.json`
- Model: `google/gemma-3-4b-it`

Configured values:

- `N_SAMPLES = 100`
- `DEGRADATION_MODE = "k1"`

Main outputs:

- `natural_hallucination_checkpoint_k1.json`
- `natural_hallucination_test_k1.json`
- `natural_hallucination_labeling_k1.csv`

### 6. Label natural hallucinations with an LLM judge

Notebook:

`rehab/gemma/hotpot/tests/12/super/llm_label_natural_hallucinations.ipynb`

Input:

- `${WORK_DIR}/gemma/hotpot/tests/12/natural_hallucination_test_k1.json`

Environment variable required:

- `GROQ_API_KEY`

Main outputs:

- `natural_hallucination_llm_label_checkpoint.json`
- `natural_hallucination_test_k1_LLM_LABELED.json`
- `natural_hallucination_test_k1_LLM_LABELED.csv`

Important security note: this notebook contains older cells with a hard-coded Groq key. Remove or skip those cells before sharing/rerunning, and use `GROQ_API_KEY` from the environment instead.

### 7. Run LettuceDetect baseline using the 09 pickle

Notebook:

`rehab/gemma/hotpot/tests/18, 09's pickle/18_lettucedetect_baseline (2).ipynb`

Use this `tests/18, 09's pickle` notebook for the documented Hotpot run. The plain `rehab/gemma/hotpot/tests/18` folder loads the `tests/10/super/hotpot_sds_perplexity_results.pkl` baseline result instead, so it is a different/alternate LettuceDetect comparison and is not the same as the 09-pickle pipeline.

Inputs:

- `${WORK_DIR}/gemma/hotpot/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`
- `${WORK_DIR}/gemma/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl`

Main output:

- `lettucedetect_hotpotqa_results_corrected.pkl`

## Gemma SQuAD Run Order

### 1. Generate the length-controlled SQuAD dataset

Notebook:

`rehab/gemma/squad/02_generate_length_controlled_latest.ipynb`

Inputs:

- SQuAD from Hugging Face: `load_dataset("squad", split="train")`
- FAISS index: `${WORK_DIR}/retrieval_assets/squad_faiss.index`
- Corpus JSON: `${WORK_DIR}/retrieval_assets/squad_corpus.json`
- Model: `google/gemma-3-4b-it`

Main outputs:

- `checkpoint_non_hall_squad.json`
- `checkpoint_hall_squad.json`
- `length_distribution_analysis.png`
- `hotpot_qwen25_rag_length_controlled_1000.json`

Important folder note: this repo currently contains `hotpot_qwen25_rag_length_controlled_1000 (1).json`, not `hotpot_qwen25_rag_length_controlled_1000.json`, in `rehab/gemma/squad`. If rerunning downstream notebooks, either rename the file or update notebook paths.

### 2. Use the bin-matched SQuAD dataset

Current checked output:

- `rehab/gemma/squad/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

There is no `bin-matching-filter.ipynb` currently present in `rehab/gemma/squad`, but the bin-matched JSON already exists. If rerunning bin matching for this dataset, reuse:

`rehab/gemma/hotpot/bin-matching-filter.ipynb`

Copy it into `rehab/gemma/squad` or open it from the Hotpot folder, then update `input_file` and any output path/name for the Gemma SQuAD dataset. For this run, the input should point to the SQuAD generated JSON, for example:

- `rehab/gemma/squad/hotpot_qwen25_rag_length_controlled_1000.json`
- or, in the current repo snapshot, `rehab/gemma/squad/hotpot_qwen25_rag_length_controlled_1000 (1).json`

The expected output should be:

- `rehab/gemma/squad/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

### 3. Run uncertainty + SDS drift experiment

Notebook:

`rehab/gemma/squad/tests/09/09_uncertainty_drift_kaggledf1c15092b_(3)_(1) (1).ipynb`

Input:

- `${WORK_DIR}/gemma/squad/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

Temporary checkpoint:

- `uncertainty_checkpoint.pkl`

Main outputs:

- `hotpot_phi3_uncertainty_results.pkl`
- `hotpot_phi3_uncertainty_results.csv`
- `uncertainty_drift_results.png`

### 4. Run baseline comparison

Notebook:

`rehab/gemma/squad/tests/10/10_baseline_comparison_kaggle_clean (1).ipynb`

Input:

- `${WORK_DIR}/gemma/squad/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`

Main output:

- `hotpot_sds_perplexity_results.pkl`

### 5. Run LettuceDetect baseline using the 09 pickle

Preferred notebook for the 09-pickle comparison:

`rehab/gemma/squad/tests/18, 09's pickle/18_lettucedetect_baseline (2).ipynb`

Inputs:

- `${WORK_DIR}/gemma/squad/hotpot_qwen25_rag_length_controlled_1000_binmatched.json`
- 09 pickle from the uncertainty notebook.

Path warning: this notebook currently points to `${WORK_DIR}/gemma/squad/09/hotpot_phi3_uncertainty_results.pkl`, but the repo stores that file under `rehab/gemma/squad/tests/09/hotpot_phi3_uncertainty_results.pkl`. Update `YOUR_RESULTS_PATH` before running if your Drive layout follows this repo.

Main output:

- `lettucedetect_hotpotqa_results_corrected.pkl`

Other LettuceDetect folders:

- `rehab/gemma/squad/tests/18` loads the tests/10 baseline pickle, so it is not the same as the 09-pickle comparison.
- `rehab/gemma/squad/tests/18 new` also uses the 09-pickle style, but the documented path above is `tests/18, 09's pickle`.

## Dependency Chain Summary

```text
Gemma Hotpot:

hotpot/02_generate_length_controlled_latest.ipynb
  -> hotpot_qwen25_rag_length_controlled_1000.json

hotpot/bin-matching-filter.ipynb
  -> hotpot_qwen25_rag_length_controlled_1000_binmatched.json

hotpot/tests/09/09_uncertainty_drift_....ipynb
  uses hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  -> hotpot_phi3_uncertainty_results.pkl
  -> hotpot_phi3_uncertainty_results.csv

hotpot/tests/10/super/10_baseline_comparison_....ipynb
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

Gemma SQuAD:

squad/02_generate_length_controlled_latest.ipynb
  -> hotpot_qwen25_rag_length_controlled_1000.json

squad/hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  -> existing bin-matched dataset

squad/tests/09/09_uncertainty_drift_....ipynb
  uses hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  -> hotpot_phi3_uncertainty_results.pkl
  -> hotpot_phi3_uncertainty_results.csv

squad/tests/10/10_baseline_comparison_....ipynb
  uses hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  -> hotpot_sds_perplexity_results.pkl

squad/tests/18, 09's pickle/18_lettucedetect_baseline (2).ipynb
  uses hotpot_qwen25_rag_length_controlled_1000_binmatched.json
  uses tests/09/hotpot_phi3_uncertainty_results.pkl after path correction
  -> lettucedetect_hotpotqa_results_corrected.pkl
```

## Cross-check notes

- Gemma Hotpot `tests/10/super` is the baseline notebook matching the Phi Hotpot structure.
- Gemma Hotpot `tests/18, 09's pickle` uses the 09 uncertainty pickle.
- Gemma Hotpot plain `tests/18` uses the tests/10 baseline pickle, so it is not the documented 09-pickle path.
- Gemma SQuAD `tests/18, 09's pickle` should use the 09 pickle, but its current hard-coded path is missing `tests/09`; update it before running.
- Gemma SQuAD has no `tests/12` folder in the current repo snapshot.
- Repeated filenames are expected; keep Hotpot and SQuAD files inside their own folders.
