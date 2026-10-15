# Data and third-party terms

Original project software is licensed under [MIT](LICENSE). Dataset-derived
records, model access and third-party components retain their own terms.
Model weights, dataset distributions, FAISS indexes and retrieval corpora
are external inputs.

## HotpotQA and SQuAD

- HotpotQA: Yang et al. (2018),
  <https://hotpotqa.github.io/>. The project states CC BY-SA 4.0 for the
  dataset and processed Wikipedia corpus.
- SQuAD: Rajpurkar et al. (2016),
  <https://rajpurkar.github.io/SQuAD-explorer/>. The dataset distribution is
  CC BY-SA 4.0.
- License text: <https://creativecommons.org/licenses/by-sa/4.0/legalcode>.

Saved question, context and reference records derived from these datasets
retain the applicable attribution and share-alike terms. Generated answers,
labels, measured features and record selections are additions to the source
data. Preserve source attribution and the applicable license when
redistributing dataset-derived material. Inclusion does not imply endorsement
by the dataset creators.

Source datasets are identified by the experiment's dataset-loading code and
record provenance; filenames may be reused across experiments.

## RAGTruth

Niu et al. (2024),
<https://aclanthology.org/2024.acl-long.585/>;
dataset repository: <https://github.com/CodingLLM/RAGTruth>.

Included files under `analysis/ragtruth_results/` contain numeric scores,
labels, source-model identifiers and record IDs. They do not contain the
original RAGTruth passages or response text. Use the upstream repository
for source-corpus access and applicable distribution terms.

## Models and dependencies

Model access and generated-output use are subject to the applicable
model/provider terms. Gated models require authorization from their providers.
Third-party packages retain their respective licenses; `requirements.txt`
specifies the dependencies for saved-result analysis.

The MIT license covers the authors' original code and documentation.
Dataset-derived records and other third-party material retain the terms
described above.
