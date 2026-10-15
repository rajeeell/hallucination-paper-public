"""Calibration + deployment-threshold analysis on the 8 controlled cells.

Computes probability-calibration and threshold-selection metrics:

  - **Expected Calibration Error (ECE)**: how well do predicted hallucination
    probabilities match actual hallucination rates across 10 bins?
  - **Brier score**: strict proper scoring rule on probabilistic predictions.
  - **Reliability diagram**: 10-bin plot of predicted vs actual probability.
  - **Deployment thresholds**: per-cell perplexity threshold and SDS+Unc score
    that achieves 90% / 95% precision flagging hallucinations, plus the recall
    at that threshold.

Uses the existing 8-cell tests/09 pickles (no new compute). Two scoring
sources per cell:

  1. **SDS+Unc** — out-of-fold predictions from the original paper's CV
     protocol (raw PPL + mean_prob + min_prob + SDS → StandardScaler →
     LR(C=1.0, random_state=42) → cross_val_predict with cv=5,
     method='predict_proba'). Directly probabilistic.
  2. **Perplexity** — raw perplexity values → Platt-scaled (sigmoid LR fit
     on the same out-of-fold predictions via cross_val_predict) for an
     apples-to-apples probabilistic comparison.

Outputs:
  - analysis/results/calibration_metrics.csv (per-cell ECE, Brier, threshold table)
  - analysis/figures/calibration_reliability.{pdf,png} (8-panel reliability)
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, precision_recall_curve


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
FIG_DIR = REPO_ROOT / "analysis" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# (model_short, dataset, model_color, dataset_marker, pickle_relpath)
CELLS = [
    ("Phi-3",     "HotpotQA", "phi/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Phi-3",     "SQuAD",    "phi/squad/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Gemma-3",   "HotpotQA", "gemma/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Gemma-3",   "SQuAD",    "gemma/squad/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Llama-3.1", "HotpotQA", "llama 8b/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Llama-3.1", "SQuAD",    "llama 8b/squad/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Qwen-2.5",  "HotpotQA", "qwen 14b/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Qwen-2.5",  "SQuAD",    "qwen 14b/squad/tests/09/hotpot_phi3_uncertainty_results.pkl"),
]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Standard equal-width ECE: |mean_pred - mean_true| per bin, weighted by bin size."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        # Inclusive upper bound for the last bin
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if not in_bin.any():
            continue
        bin_mean_pred = float(y_prob[in_bin].mean())
        bin_mean_true = float(y_true[in_bin].mean())
        bin_weight = in_bin.sum() / n
        ece += bin_weight * abs(bin_mean_pred - bin_mean_true)
    return float(ece)


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple:
    """Return (bin_centers, bin_mean_true, bin_count) for plotting."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, mean_true, counts = [], [], []
    for i in range(n_bins):
        in_bin = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        if i == n_bins - 1:
            in_bin = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        if in_bin.any():
            centers.append((edges[i] + edges[i + 1]) / 2.0)
            mean_true.append(float(y_true[in_bin].mean()))
            counts.append(int(in_bin.sum()))
        else:
            centers.append((edges[i] + edges[i + 1]) / 2.0)
            mean_true.append(float("nan"))
            counts.append(0)
    return np.array(centers), np.array(mean_true), np.array(counts)


def threshold_at_precision(y_true: np.ndarray, y_score: np.ndarray, target_precision: float) -> tuple:
    """Find score threshold achieving target precision (or higher); return (threshold, precision, recall).

    Higher score = more likely hallucinated.
    If no threshold reaches the target precision, return the highest-precision threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns arrays of length n+1 / n+1 / n
    # Pair them safely
    n = len(thresholds)
    # Sort by precision descending to find smallest threshold that meets target
    candidates = []
    for i in range(n):
        if precisions[i] >= target_precision:
            candidates.append((float(thresholds[i]), float(precisions[i]), float(recalls[i])))
    if not candidates:
        # Fall back: use highest-precision threshold
        i_best = int(np.argmax(precisions[:-1]))
        return float(thresholds[i_best]), float(precisions[i_best]), float(recalls[i_best])
    # Among those meeting target, return the one with highest recall
    candidates.sort(key=lambda t: -t[2])
    return candidates[0]


def load_cell(rel_path: str) -> dict:
    path = REPO_ROOT / rel_path
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {
        "y":   np.asarray(data["labels"], dtype=int),
        "ppl": np.asarray(data["perplexity"], dtype=float),
        "mp":  np.asarray(data["mean_prob"], dtype=float),
        "mn":  np.asarray(data["min_prob"], dtype=float),
        "sds": np.asarray(data["SDS"], dtype=float),
    }


def out_of_fold_proba(features: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Return out-of-fold predicted probabilities under the original paper's
    LR + 5-fold CV protocol."""
    Xs = StandardScaler().fit_transform(features)
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    return cross_val_predict(lr, Xs, y, cv=5, method="predict_proba")[:, 1]


def main() -> None:
    rows = []
    bin_data = {}  # cell_name -> (centers, mean_true, counts) for plotting

    print(f"{'cell':<22} {'n':>5} {'PPL ECE':>8} {'PPL Brier':>10} {'SDS+Unc ECE':>12} {'SDS+Unc Brier':>14}")
    print("-" * 80)
    for model, dataset, rel in CELLS:
        d = load_cell(rel)
        y = d["y"]
        n = len(y)
        # ---- SDS+Unc: original 4-feature LR, out-of-fold probabilities ----
        X_full = np.c_[d["ppl"], d["mp"], d["mn"], d["sds"]]
        p_sdsunc = out_of_fold_proba(X_full, y)
        # ---- Perplexity alone: Platt-scaled via out-of-fold LR on PPL only ----
        X_ppl = d["ppl"].reshape(-1, 1)
        p_ppl = out_of_fold_proba(X_ppl, y)

        # ECE + Brier for both
        ece_ppl   = expected_calibration_error(y, p_ppl)
        ece_sds   = expected_calibration_error(y, p_sdsunc)
        brier_ppl = float(brier_score_loss(y, p_ppl))
        brier_sds = float(brier_score_loss(y, p_sdsunc))

        # Deployment thresholds — perplexity raw score, target 95% and 90% precision
        t90_ppl, p90_ppl, r90_ppl = threshold_at_precision(y, d["ppl"], 0.90)
        t95_ppl, p95_ppl, r95_ppl = threshold_at_precision(y, d["ppl"], 0.95)
        # And SDS+Unc probability threshold
        t90_sds, p90_sds, r90_sds = threshold_at_precision(y, p_sdsunc, 0.90)
        t95_sds, p95_sds, r95_sds = threshold_at_precision(y, p_sdsunc, 0.95)

        rows.append({
            "model": model, "dataset": dataset,
            "cell": f"{model}/{dataset}",
            "n": n,
            "pos_rate": float(y.mean()),
            "ppl_ECE": ece_ppl,
            "ppl_Brier": brier_ppl,
            "sdsunc_ECE": ece_sds,
            "sdsunc_Brier": brier_sds,
            # PPL raw thresholds
            "ppl_thresh_90": t90_ppl, "ppl_prec_at_t90": p90_ppl, "ppl_recall_at_t90": r90_ppl,
            "ppl_thresh_95": t95_ppl, "ppl_prec_at_t95": p95_ppl, "ppl_recall_at_t95": r95_ppl,
            # SDS+Unc probability thresholds
            "sdsunc_thresh_90": t90_sds, "sdsunc_prec_at_t90": p90_sds, "sdsunc_recall_at_t90": r90_sds,
            "sdsunc_thresh_95": t95_sds, "sdsunc_prec_at_t95": p95_sds, "sdsunc_recall_at_t95": r95_sds,
        })

        bin_data[f"{model}/{dataset}"] = {
            "ppl": reliability_bins(y, p_ppl),
            "sdsunc": reliability_bins(y, p_sdsunc),
        }
        print(f"{model+'/'+dataset:<22} {n:>5} {ece_ppl:>8.4f} {brier_ppl:>10.4f} {ece_sds:>12.4f} {brier_sds:>14.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "calibration_metrics.csv", index=False)
    print(f"\nwrote {RESULTS_DIR/'calibration_metrics.csv'}")

    # ---- Deployment threshold table ----
    print("\n--- Deployment thresholds: perplexity threshold for 95% precision (raw PPL) ---")
    print(f"{'cell':<22} {'thresh':>7}  {'precision':>10}  {'recall':>7}")
    print("-" * 55)
    for r in rows:
        print(f"{r['cell']:<22} {r['ppl_thresh_95']:>7.3f}  {r['ppl_prec_at_t95']:>10.3f}  {r['ppl_recall_at_t95']:>7.3f}")

    print("\n--- Deployment thresholds: SDS+Unc probability for 95% precision ---")
    for r in rows:
        print(f"{r['cell']:<22} {r['sdsunc_thresh_95']:>7.3f}  {r['sdsunc_prec_at_t95']:>10.3f}  {r['sdsunc_recall_at_t95']:>7.3f}")

    # ---- Figure: 8-panel reliability ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    cells_list = [f"{m}/{d}" for m, d, _ in CELLS]
    for ax, cell in zip(axes.flat, cells_list):
        b = bin_data[cell]
        # Diagonal
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        # PPL bars
        c, t, w = b["ppl"]
        mask = w > 0
        ax.plot(c[mask], t[mask], marker="o", color="#4477AA", label="PPL (Platt-scaled)", linewidth=1.5, markersize=5)
        # SDS+Unc bars
        c2, t2, w2 = b["sdsunc"]
        mask2 = w2 > 0
        ax.plot(c2[mask2], t2[mask2], marker="s", color="#EE6677", label="SDS+Unc", linewidth=1.5, markersize=5)
        ax.set_title(cell, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    for ax in axes[-1, :]:
        ax.set_xlabel("Predicted probability of hallucination")
    for ax in axes[:, 0]:
        ax.set_ylabel("Actual fraction hallucinated")
    axes[0, 3].legend(loc="upper left", fontsize=7, frameon=True)

    fig.suptitle("Reliability diagrams: predicted vs actual hallucination probability (8 cells, controlled benchmarks)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "calibration_reliability.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "calibration_reliability.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {FIG_DIR/'calibration_reliability.pdf'}")
    print(f"wrote {FIG_DIR/'calibration_reliability.png'}")

    # Headline summary
    summary = {
        "mean_ppl_ECE": float(df["ppl_ECE"].mean()),
        "mean_sdsunc_ECE": float(df["sdsunc_ECE"].mean()),
        "mean_ppl_Brier": float(df["ppl_Brier"].mean()),
        "mean_sdsunc_Brier": float(df["sdsunc_Brier"].mean()),
        "max_ppl_ECE": float(df["ppl_ECE"].max()),
        "max_sdsunc_ECE": float(df["sdsunc_ECE"].max()),
        "cells_with_95pct_thresh_achievable": int((df["ppl_prec_at_t95"] >= 0.949).sum()),
        "median_recall_at_95pct_PPL_thresh": float(df["ppl_recall_at_t95"].median()),
    }
    import json
    with open(RESULTS_DIR / "calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n--- summary ---")
    print(f"mean PPL ECE = {summary['mean_ppl_ECE']:.4f}, mean SDS+Unc ECE = {summary['mean_sdsunc_ECE']:.4f}")
    print(f"mean PPL Brier = {summary['mean_ppl_Brier']:.4f}, mean SDS+Unc Brier = {summary['mean_sdsunc_Brier']:.4f}")
    print(f"cells where PPL achieves ≥95% precision: {summary['cells_with_95pct_thresh_achievable']}/8")
    print(f"median recall at 95% precision threshold (PPL): {summary['median_recall_at_95pct_PPL_thresh']:.3f}")


if __name__ == "__main__":
    main()
