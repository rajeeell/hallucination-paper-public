"""External validation on RAGTruth (Niu et al. 2024).

Loads the 4-model teacher-forced scoring of RAGTruth's 875 QA examples,
computes AUROCs / ECE / Brier / 95%-precision thresholds, and a
per-source-model breakdown (does the detector work equally well across the
6 source models that produced the RAGTruth responses?).

Inputs: analysis/ragtruth_results/ragtruth/<our_model>/ragtruth_qa.json
Outputs:
  - analysis/results/ragtruth_metrics.csv (per-our-model headline numbers)
  - analysis/results/ragtruth_by_source_model.csv (per-(our × source) AUROCs)
  - analysis/figures/ragtruth_comparison.{pdf,png} (controlled vs RAGTruth)
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, precision_recall_curve,
)


def boot_auc_ci(y: np.ndarray, score: np.ndarray, B: int = 10000, seed: int = 42) -> tuple:
    """Stratified-bootstrap AUROC with 95% CI. Returns (auc, lo, hi, n_pos, n_neg)."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), float("nan"), float("nan"), int(len(pos)), int(len(neg))
    aurocs = np.empty(B)
    for b in range(B):
        p = rng.choice(pos, len(pos), replace=True)
        n = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([p, n])
        try:
            aurocs[b] = roc_auc_score(y[idx], score[idx])
        except Exception:
            aurocs[b] = np.nan
    aurocs = aurocs[np.isfinite(aurocs)]
    auc = float(roc_auc_score(y, score))
    lo, hi = np.percentile(aurocs, [2.5, 97.5])
    return auc, float(lo), float(hi), int(len(pos)), int(len(neg))


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RT_DIR = REPO_ROOT / "analysis" / "ragtruth_results" / "ragtruth"
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
FIG_DIR = REPO_ROOT / "analysis" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


CELLS = [
    ("Phi-3",     "microsoft_Phi-3-mini-4k-instruct"),
    ("Gemma-3",   "google_gemma-3-4b-it"),
    ("Llama-3.1", "meta-llama_Llama-3.1-8B-Instruct"),
    ("Qwen-2.5",  "Qwen_Qwen2.5-14B-Instruct"),
]


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            m = (p >= edges[i]) & (p < edges[i + 1])
        if m.any():
            err += m.sum() / len(y) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(err)


def threshold_at_precision(y: np.ndarray, score: np.ndarray, target: float = 0.95) -> tuple:
    p, r, t = precision_recall_curve(y, score)
    ok = [(float(t[i]), float(p[i]), float(r[i])) for i in range(len(t)) if p[i] >= target]
    if not ok:
        i_best = int(np.argmax(p[:-1]))
        return float(t[i_best]), float(p[i_best]), float(r[i_best])
    ok.sort(key=lambda x: -x[2])
    return ok[0]


def main() -> None:
    headline_rows = []
    source_rows = []

    print(f"{'our model':<12} {'n':>5} {'pos':>5} {'PPL':>6} {'MeanP':>6} {'MinP':>6} {'SDS':>6} {'SDS+Unc':>8} {'ECE':>5} {'recall@95':>10}")
    print("-" * 90)
    for our_short, our_slug in CELLS:
        path = RT_DIR / our_slug / "ragtruth_qa.json"
        if not path.exists():
            print(f"  [skip] {our_short}: missing {path}")
            continue
        recs = json.loads(path.read_text())["records"]
        y   = np.array([r["hallucinated"] for r in recs])
        ppl = np.array([r["perplexity"] for r in recs])
        mp  = np.array([r["mean_prob"] for r in recs])
        mn  = np.array([r["min_prob"] for r in recs])
        sds = np.array([r["sds"] for r in recs])
        src = np.array([r["source_model"] for r in recs])

        auc_ppl = float(roc_auc_score(y, ppl))
        auc_mp  = float(roc_auc_score(y, -mp))
        auc_mn  = float(roc_auc_score(y, -mn))
        auc_sds = float(roc_auc_score(y, sds))

        X = np.c_[ppl, mp, mn, sds]
        Xs = StandardScaler().fit_transform(X)
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        p_sdsunc = cross_val_predict(lr, Xs, y, cv=5, method="predict_proba")[:, 1]
        auc_sdsunc = float(roc_auc_score(y, p_sdsunc))
        ece_sdsunc = ece(y, p_sdsunc)
        brier_sdsunc = float(brier_score_loss(y, p_sdsunc))

        t95, p95, r95 = threshold_at_precision(y, ppl, 0.95)

        print(f"{our_short:<12} {len(y):>5} {y.mean():>5.2f} {auc_ppl:>6.3f} {auc_mp:>6.3f} {auc_mn:>6.3f} {auc_sds:>6.3f} {auc_sdsunc:>8.3f} {ece_sdsunc:>5.3f} {r95:>10.3f}")

        headline_rows.append({
            "our_model": our_short, "n": len(y), "pos_rate": float(y.mean()),
            "auc_ppl": auc_ppl, "auc_mean_prob": auc_mp,
            "auc_min_prob": auc_mn, "auc_sds": auc_sds,
            "auc_sds_unc": auc_sdsunc,
            "ece_sds_unc": ece_sdsunc, "brier_sds_unc": brier_sdsunc,
            "ppl_thresh_95": t95, "ppl_prec_at_95": p95, "ppl_recall_at_95": r95,
        })

        # Per-source-model breakdown
        for source in sorted(np.unique(src)):
            mask = src == source
            if mask.sum() < 30 or len(set(y[mask])) < 2:
                continue
            source_rows.append({
                "our_model": our_short,
                "source_model": source,
                "n": int(mask.sum()),
                "pos_rate": float(y[mask].mean()),
                "auc_ppl": float(roc_auc_score(y[mask], ppl[mask])),
                "auc_sds_unc": float(roc_auc_score(y[mask], p_sdsunc[mask])),
            })

    pd.DataFrame(headline_rows).to_csv(RESULTS_DIR / "ragtruth_metrics.csv", index=False)
    pd.DataFrame(source_rows).to_csv(RESULTS_DIR / "ragtruth_by_source_model.csv", index=False)
    print(f"\nwrote {RESULTS_DIR/'ragtruth_metrics.csv'}")
    print(f"wrote {RESULTS_DIR/'ragtruth_by_source_model.csv'}")

    # Per-source-model summary table
    print("\n--- per-(our_model × source_model) AUROC (SDS+Unc) ---")
    sdf = pd.DataFrame(source_rows)
    pivot = sdf.pivot_table(index="our_model", columns="source_model", values="auc_sds_unc")
    print(pivot.round(3).to_string())

    # Summary statistics
    means = pd.DataFrame(headline_rows).mean(numeric_only=True)
    summary = {
        "n_cells": len(headline_rows),
        "mean_auc_ppl": float(means["auc_ppl"]),
        "mean_auc_sds_unc": float(means["auc_sds_unc"]),
        "mean_delta_sdsunc_minus_ppl": float(means["auc_sds_unc"] - means["auc_ppl"]),
        "mean_ece_sds_unc": float(means["ece_sds_unc"]),
        "mean_brier_sds_unc": float(means["brier_sds_unc"]),
        "median_recall_at_95pct_PPL": float(pd.Series([r["ppl_recall_at_95"] for r in headline_rows]).median()),
    }
    with open(RESULTS_DIR / "ragtruth_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {RESULTS_DIR/'ragtruth_summary.json'}")
    print(f"\n--- headline summary ---")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
        else:
            print(f"  {k:<35} {v}")

    # ---- Figure: controlled vs RAGTruth AUROC comparison ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Controlled-cell AUROCs from the paper's tab:main-results (hardcoded; verified by verify_paper_metrics.py)
    controlled = {
        ("Phi-3", "HotpotQA"):   {"ppl": 0.811, "sds_unc": 0.842},
        ("Phi-3", "SQuAD"):      {"ppl": 0.881, "sds_unc": 0.859},
        ("Gemma-3", "HotpotQA"): {"ppl": 0.908, "sds_unc": 0.911},
        ("Gemma-3", "SQuAD"):    {"ppl": 0.871, "sds_unc": 0.868},
        ("Llama-3.1", "HotpotQA"): {"ppl": 0.883, "sds_unc": 0.846},
        ("Llama-3.1", "SQuAD"):  {"ppl": 0.885, "sds_unc": 0.869},
        ("Qwen-2.5", "HotpotQA"):{"ppl": 0.818, "sds_unc": 0.824},
        ("Qwen-2.5", "SQuAD"):   {"ppl": 0.878, "sds_unc": 0.878},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    model_color = {"Phi-3": "#4477AA", "Gemma-3": "#EE6677", "Llama-3.1": "#228833", "Qwen-2.5": "#CCBB44"}
    width = 0.35
    x = np.arange(4)
    for i, (our_short, _) in enumerate(CELLS):
        ctrl_aucs_ppl = [v["ppl"] for (m, _), v in controlled.items() if m == our_short]
        ctrl_aucs_sds = [v["sds_unc"] for (m, _), v in controlled.items() if m == our_short]
        rt_row = next((r for r in headline_rows if r["our_model"] == our_short), None)
        if rt_row is None:
            continue
        ctrl_ppl = np.mean(ctrl_aucs_ppl)
        ctrl_sds = np.mean(ctrl_aucs_sds)
        rt_ppl = rt_row["auc_ppl"]
        rt_sds = rt_row["auc_sds_unc"]
        ax.bar(x[i] - width/2, ctrl_sds, width, color=model_color[our_short], edgecolor="black", linewidth=0.5, label="Controlled (SDS+Unc)" if i == 0 else None, alpha=0.85)
        ax.bar(x[i] + width/2, rt_sds, width, color=model_color[our_short], edgecolor="black", linewidth=0.5, hatch="///", label="RAGTruth (SDS+Unc)" if i == 0 else None, alpha=0.85)
        # PPL marker on top
        ax.scatter(x[i] - width/2, ctrl_ppl, color="black", marker="_", s=160, zorder=5, label="PPL" if i == 0 else None)
        ax.scatter(x[i] + width/2, rt_ppl, color="black", marker="_", s=160, zorder=5)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in CELLS])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("SDS+Unc AUROC: controlled-cell mean vs RAGTruth QA")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ragtruth_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ragtruth_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {FIG_DIR/'ragtruth_comparison.pdf'}")


if __name__ == "__main__":
    main()
