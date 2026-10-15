"""Length-bin sensitivity and question-grouped cross-validation analyses.

R10 computes AUROCs within length-defined sub-quartiles of the matched
faithful/unsupported answer set to measure sensitivity to length-bin selection.

R6 compares question-grouped cross-validation with stratified cross-validation.
Questions are used as group identifiers so answers sharing a question remain
in the same fold. The report includes the resulting AUROC difference.

Inputs:
  - bin-matched answer records under each model/dataset folder
  - token uncertainty and SDS features from tests/09/
  - combined-score results from tests/10/

Outputs:
  - analysis/results/r10_binmatching_sensitivity.csv
  - analysis/results/r6_groupkfold_cv.csv
  - analysis/results/rigor_audit_summary.json
"""

from __future__ import annotations

import json
import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# (model_label, dataset_label, model_root, bm_subdir)
CELLS = [
    ("Phi-3-mini",   "HotpotQA", "phi",        "phi/hotpot"),
    ("Phi-3-mini",   "SQuAD",    "phi",        "phi/squad"),
    ("Gemma-3-4B",   "HotpotQA", "gemma",      "gemma/hotpot"),
    ("Gemma-3-4B",   "SQuAD",    "gemma",      "gemma/squad"),
    ("Llama-3.1-8B", "HotpotQA", "llama 8b",   "llama 8b/hotpot"),
    ("Llama-3.1-8B", "SQuAD",    "llama 8b",   "llama 8b/squad"),
    ("Qwen-2.5-14B", "HotpotQA", "qwen 14b",   "qwen 14b/hotpot"),
    ("Qwen-2.5-14B", "SQuAD",    "qwen 14b",   "qwen 14b/squad/good p"),  # squad lives under "good p"
]


def _resolve_paths(model_root: str, dataset_label: str, bm_subdir: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Resolve the (binmatched_json, tests09_pkl) pair for a cell.

    The squad-Qwen cell is under `qwen 14b/squad/good p` for the binmatched JSON,
    but the tests/09 pickle is at `qwen 14b/squad/tests/09/`. Handle both.
    """
    bm = REPO_ROOT / bm_subdir / "hotpot_qwen25_rag_length_controlled_1000_binmatched.json"
    # tests/09 path drops the "good p" suffix for Qwen+SQuAD
    if "good p" in bm_subdir:
        base = pathlib.Path(bm_subdir).parent  # qwen 14b/squad
        pkl = REPO_ROOT / base / "tests" / "09" / "hotpot_phi3_uncertainty_results.pkl"
    else:
        pkl = REPO_ROOT / bm_subdir / "tests" / "09" / "hotpot_phi3_uncertainty_results.pkl"
    return bm, pkl


def load_cell(model: str, dataset: str, model_root: str, bm_subdir: str) -> pd.DataFrame:
    bm_path, pkl_path = _resolve_paths(model_root, dataset, bm_subdir)
    bm = json.loads(bm_path.read_text())
    with open(pkl_path, "rb") as f:
        u = pickle.load(f)

    n = len(bm)
    assert n == len(u["labels"]), f"length mismatch: bm={n} pkl={len(u['labels'])}"

    # Sanity: labels agree (verified once; this also catches accidental reorderings)
    y_bm = np.array([b["hallucinated"] for b in bm], dtype=int)
    y_pkl = np.asarray(u["labels"], dtype=int)
    assert (y_bm == y_pkl).all(), f"label mismatch between bm and pkl for {model}/{dataset}"

    df = pd.DataFrame({
        "cell":        f"{model}/{dataset}",
        "model":       model,
        "dataset":     dataset,
        "question":    [b["question"] for b in bm],
        "answer":      [b["answer"] for b in bm],
        "label":       y_bm,
        "perplexity":  np.asarray(u["perplexity"], dtype=float),
        "mean_prob":   np.asarray(u["mean_prob"], dtype=float),
        "min_prob":    np.asarray(u["min_prob"], dtype=float),
        "sds":         np.asarray(u["SDS"], dtype=float),
    })
    df["ans_words"] = df["answer"].str.split().str.len()
    return df


def auroc_safe(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


# ---------------- R10 -----------------

def r10_binmatch_sensitivity(df: pd.DataFrame) -> dict:
    """AUROCs of (perplexity, sds+unc combined) on the full binmatched set
    plus three length-defined sub-quartiles (Q1, Q2-Q3, Q4)."""

    def features(d: pd.DataFrame) -> np.ndarray:
        return np.c_[
            np.log(np.clip(d["perplexity"], 1e-6, None)),
            d["mean_prob"], d["min_prob"], d["sds"],
        ]

    y = df["label"].to_numpy()
    X = features(df)

    # SDS+Unc combined: LR with StratifiedKFold (the paper's protocol)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
        pipe.fit(X[tr], y[tr])
        scores[te] = pipe.predict_proba(X[te])[:, 1]

    def row(subset_label: str, mask: np.ndarray) -> dict:
        d_sub = df[mask]
        y_sub = y[mask]
        if len(np.unique(y_sub)) < 2 or len(d_sub) < 20:
            return {"subset": subset_label, "n": int(len(d_sub)),
                    "pos_rate": float(y_sub.mean()) if len(y_sub) else float("nan"),
                    "ans_words_med": float(d_sub["ans_words"].median()) if len(d_sub) else float("nan"),
                    "perplexity_AUROC": float("nan"), "sds_unc_AUROC": float("nan")}
        return {
            "subset": subset_label,
            "n": int(len(d_sub)),
            "pos_rate": float(y_sub.mean()),
            "ans_words_med": float(d_sub["ans_words"].median()),
            "ans_words_min": int(d_sub["ans_words"].min()),
            "ans_words_max": int(d_sub["ans_words"].max()),
            "perplexity_AUROC": auroc_safe(y_sub, np.log(np.clip(d_sub["perplexity"], 1e-6, None).to_numpy())),
            "sds_unc_AUROC":     auroc_safe(y_sub, scores[mask]),
        }

    quartiles = np.quantile(df["ans_words"], [0.25, 0.5, 0.75])
    q1_mask    = df["ans_words"] <= quartiles[0]
    mid_mask   = (df["ans_words"] > quartiles[0]) & (df["ans_words"] <= quartiles[2])
    q4_mask    = df["ans_words"] > quartiles[2]
    all_mask   = np.ones(len(df), dtype=bool)

    rows = [
        row("all",     all_mask),
        row("len_Q1",  q1_mask.to_numpy()),
        row("len_Q23", mid_mask.to_numpy()),
        row("len_Q4",  q4_mask.to_numpy()),
    ]
    return {
        "rows": rows,
        "delta_perplexity_max_minus_min":
            max(r["perplexity_AUROC"] for r in rows[1:] if np.isfinite(r["perplexity_AUROC"]))
          - min(r["perplexity_AUROC"] for r in rows[1:] if np.isfinite(r["perplexity_AUROC"])),
        "delta_sds_unc_max_minus_min":
            max(r["sds_unc_AUROC"] for r in rows[1:] if np.isfinite(r["sds_unc_AUROC"]))
          - min(r["sds_unc_AUROC"] for r in rows[1:] if np.isfinite(r["sds_unc_AUROC"])),
    }


# ---------------- R6 -----------------

def r6_groupkfold_cv(df: pd.DataFrame) -> dict:
    """SDS+Unc combined AUROC under (a) StratifiedKFold (paper protocol) vs
    (b) GroupKFold using `question` as the group identifier.

    With GroupKFold, n_splits is bounded by the number of unique groups; we
    use min(5, n_groups). If GroupKFold can't split (n_groups < 2), we
    return NaN for the group-CV AUROC.
    """

    X = np.c_[
        np.log(np.clip(df["perplexity"], 1e-6, None)),
        df["mean_prob"], df["min_prob"], df["sds"],
    ]
    y = df["label"].to_numpy()
    groups = df["question"].astype("category").cat.codes.to_numpy()

    n_groups = len(np.unique(groups))
    n_samples = len(df)
    questions_repeated = int((df.groupby("question").size() > 1).sum())
    total_repeated_samples = int(df.groupby("question").size().pipe(lambda s: s[s > 1].sum()))

    def lr_cv_score(splitter, X, y, groups=None) -> float:
        scores = np.zeros(len(y))
        splits = splitter.split(X, y, groups=groups) if groups is not None else splitter.split(X, y)
        for tr, te in splits:
            pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
            pipe.fit(X[tr], y[tr])
            scores[te] = pipe.predict_proba(X[te])[:, 1]
        return auroc_safe(y, scores)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_skf = lr_cv_score(skf, X, y)

    if n_groups >= 5:
        gkf = GroupKFold(n_splits=5)
        auc_gkf = lr_cv_score(gkf, X, y, groups=groups)
    elif n_groups >= 2:
        gkf = GroupKFold(n_splits=n_groups)
        auc_gkf = lr_cv_score(gkf, X, y, groups=groups)
    else:
        auc_gkf = float("nan")

    return {
        "n_samples": n_samples,
        "n_unique_questions": int(n_groups),
        "questions_with_repeats": questions_repeated,
        "samples_in_repeated_groups": total_repeated_samples,
        "frac_repeated_samples": round(total_repeated_samples / n_samples, 4) if n_samples else float("nan"),
        "sds_unc_AUROC_StratifiedKFold": auc_skf,
        "sds_unc_AUROC_GroupKFold_by_question": auc_gkf,
        "delta_AUROC_group_minus_strat": (auc_gkf - auc_skf) if np.isfinite(auc_gkf) else float("nan"),
    }


# ---------------- main -----------------

def main() -> None:
    print("loading 8 cells...")
    all_cells = {}
    for model, dataset, model_root, bm_subdir in CELLS:
        df = load_cell(model, dataset, model_root, bm_subdir)
        all_cells[f"{model}/{dataset}"] = df
        print(f"  {model}/{dataset}: n={len(df):>5}  ans_words median={df['ans_words'].median():.0f}  pos_rate={df['label'].mean():.3f}")

    # ---- R10 ----
    print("\n--- R10: bin-matching sensitivity (length quartiles) ---")
    r10_rows = []
    r10_blob = {}
    for cell, df in all_cells.items():
        out = r10_binmatch_sensitivity(df)
        r10_blob[cell] = out
        for r in out["rows"]:
            r10_rows.append({"cell": cell, **r})
    r10_df = pd.DataFrame(r10_rows)
    r10_df.to_csv(RESULTS_DIR / "r10_binmatching_sensitivity.csv", index=False)
    print(r10_df.to_string(index=False))

    # ---- R6 ----
    print("\n--- R6: GroupKFold(question) CV audit ---")
    r6_rows = []
    for cell, df in all_cells.items():
        out = r6_groupkfold_cv(df)
        r6_rows.append({"cell": cell, **out})
    r6_df = pd.DataFrame(r6_rows)
    r6_df.to_csv(RESULTS_DIR / "r6_groupkfold_cv.csv", index=False)
    print(r6_df[["cell", "n_samples", "n_unique_questions", "questions_with_repeats",
                 "frac_repeated_samples",
                 "sds_unc_AUROC_StratifiedKFold", "sds_unc_AUROC_GroupKFold_by_question",
                 "delta_AUROC_group_minus_strat"]].to_string(index=False))

    # summary blob
    summary = {
        "r10_perplexity_max_quartile_delta": float(r10_df.groupby("cell").apply(
            lambda g: g.loc[g["subset"] != "all", "perplexity_AUROC"].max()
                    - g.loc[g["subset"] != "all", "perplexity_AUROC"].min()).max()),
        "r10_sds_unc_max_quartile_delta": float(r10_df.groupby("cell").apply(
            lambda g: g.loc[g["subset"] != "all", "sds_unc_AUROC"].max()
                    - g.loc[g["subset"] != "all", "sds_unc_AUROC"].min()).max()),
        "r6_max_abs_AUROC_delta": float(r6_df["delta_AUROC_group_minus_strat"].abs().max()),
        "r6_mean_AUROC_delta":    float(r6_df["delta_AUROC_group_minus_strat"].mean()),
        "r6_cells_with_meaningful_delta": int((r6_df["delta_AUROC_group_minus_strat"].abs() > 0.02).sum()),
        "n_cells": len(r6_df),
    }
    with open(RESULTS_DIR / "rigor_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n--- headline summary ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
