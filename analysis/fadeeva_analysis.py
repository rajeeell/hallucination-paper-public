"""Fadeeva 2024 baseline analysis: per-cell AUROC + bootstrap CIs.

Inputs:  analysis/fadeeva_results/fadeeva/<model>/<dataset>/fadeeva_ccp.json
Outputs:
  - analysis/results/fadeeva_aurocs.csv     (per-cell, per-variant AUROCs with CIs)
  - analysis/results/fadeeva_summary.json   (aggregate baseline results)

For each (cell, signal) pair we compute AUROC + 95% stratified-bootstrap CI.
Signals:
  - perplexity (Fadeeva's "seq-PPL" baseline, also our existing headline)
  - CCP-sent-max (sentence-level claim-conditioned probability, max aggregator)
  - CCP-sent-min (min aggregator)
  - CCP-sent-mean (mean aggregator)
  - Max-K-1 / Max-K-5 / Max-K-10 (mean of top-K surprisal)
  - Min-K-5%% / Min-K-10%% (Shi et al. style)

Headline comparison: for each cell, is the best Fadeeva variant *meaningfully*
better than the paper's existing perplexity headline (CIs disjoint)?
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FADEEVA_DIR = REPO_ROOT / "analysis" / "fadeeva_results" / "fadeeva"
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


CELLS = [
    ("Phi-3-mini",   "SQuAD",    "microsoft_Phi-3-mini-4k-instruct/squad"),
    ("Phi-3-mini",   "HotpotQA", "microsoft_Phi-3-mini-4k-instruct/hotpot"),
    ("Gemma-3-4B",   "SQuAD",    "google_gemma-3-4b-it/squad"),
    ("Gemma-3-4B",   "HotpotQA", "google_gemma-3-4b-it/hotpot"),
    ("Llama-3.1-8B", "SQuAD",    "meta-llama_Llama-3.1-8B-Instruct/squad"),
    ("Llama-3.1-8B", "HotpotQA", "meta-llama_Llama-3.1-8B-Instruct/hotpot"),
    ("Qwen-2.5-14B", "SQuAD",    "Qwen_Qwen2.5-14B-Instruct/squad"),
    ("Qwen-2.5-14B", "HotpotQA", "Qwen_Qwen2.5-14B-Instruct/hotpot"),
]

FEATURES = [
    ("perplexity",     "ppl_paper",     "seq-PPL"),
    ("CCP-sent-max",   "ccp_sent_max",  "CCP (sentence-level, max-agg)"),
    ("CCP-sent-mean",  "ccp_sent_mean", "CCP (sentence-level, mean-agg)"),
    ("CCP-sent-min",   "ccp_sent_min",  "CCP (sentence-level, min-agg)"),
    ("MaxK-1",         "maxk_1",        "Max-K=1 surprisal"),
    ("MaxK-5",         "maxk_5",        "Max-K=5 surprisal"),
    ("MaxK-10",        "maxk_10",       "Max-K=10 surprisal"),
    ("MinK-5pct",      "mink_5_pct",    "Min-K%=5%% (Shi et al. style)"),
    ("MinK-10pct",     "mink_10_pct",   "Min-K%=10%%"),
]


def bootstrap_auroc_ci(y: np.ndarray, score: np.ndarray, B: int = 10000, seed: int = 42) -> tuple:
    """Stratified bootstrap of AUROC: resample positives and negatives separately."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return float("nan"), float("nan"), float("nan")
    aurocs = np.empty(B)
    for b in range(B):
        p = rng.choice(pos_idx, len(pos_idx), replace=True)
        n = rng.choice(neg_idx, len(neg_idx), replace=True)
        idx = np.concatenate([p, n])
        try:
            aurocs[b] = roc_auc_score(y[idx], score[idx])
        except Exception:
            aurocs[b] = np.nan
    aurocs = aurocs[np.isfinite(aurocs)]
    point = float(roc_auc_score(y, score))
    lo, hi = np.percentile(aurocs, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    rows = []
    print(f"{'cell':<25} {'feature':<14} {'AUROC':>6}  [{'CI':>5}, {'95%':>5}]")
    print("-" * 70)
    for model, dataset, subdir in CELLS:
        path = FADEEVA_DIR / subdir / "fadeeva_ccp.json"
        if not path.exists():
            print(f"  [skip] {subdir}: missing {path}")
            continue
        data = json.loads(path.read_text())
        recs = data["records"]
        y = np.array([r["gold_label"] for r in recs], dtype=int)
        for feat_label, feat_key, paper_label in FEATURES:
            x = np.array([r[feat_key] for r in recs], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & (y >= 0)
            if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                continue
            auroc, lo, hi = bootstrap_auroc_ci(y[mask], x[mask])
            rows.append({
                "cell": f"{model}/{dataset}",
                "feature": feat_label,
                "feature_paper_label": paper_label,
                "n": int(mask.sum()),
                "pos_rate": float(y[mask].mean()),
                "AUROC": auroc,
                "CI_lo": lo,
                "CI_hi": hi,
            })
            if feat_label in ("perplexity", "CCP-sent-max", "MaxK-10", "MinK-10pct"):
                print(f"{model+'/'+dataset:<25} {feat_label:<14} {auroc:>6.3f}  [{lo:>5.3f}, {hi:>5.3f}]")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "fadeeva_aurocs.csv", index=False)
    print(f"\nwrote {RESULTS_DIR/'fadeeva_aurocs.csv'}  ({len(df)} rows)")

    # --- per-cell head-to-head: best Fadeeva variant vs paper's perplexity ---
    print("\n--- head-to-head: best Fadeeva variant vs paper's seq-PPL ---")
    summary_rows = []
    for cell in df["cell"].unique():
        d = df[df["cell"] == cell]
        ppl_row = d[d["feature"] == "perplexity"].iloc[0]
        # Exclude perplexity itself; find the Fadeeva variant with highest AUROC
        fadeeva_variants = d[d["feature"] != "perplexity"]
        best_idx = fadeeva_variants["AUROC"].idxmax()
        best = fadeeva_variants.loc[best_idx]
        # "Meaningfully better" = lower-CI of Fadeeva exceeds upper-CI of PPL
        cis_disjoint = best["CI_lo"] > ppl_row["CI_hi"]
        cis_disjoint_below = best["CI_hi"] < ppl_row["CI_lo"]
        delta = best["AUROC"] - ppl_row["AUROC"]
        print(f"  {cell:<25} PPL={ppl_row['AUROC']:.3f}  best={best['feature']:<14} "
              f"{best['AUROC']:.3f}  delta={delta:>+.3f}  "
              f"{'(disjoint CIs)' if cis_disjoint else '(overlapping CIs)'}")
        summary_rows.append({
            "cell": cell,
            "ppl_AUROC": ppl_row["AUROC"],
            "ppl_CI": [ppl_row["CI_lo"], ppl_row["CI_hi"]],
            "best_fadeeva_variant": best["feature"],
            "best_fadeeva_AUROC": best["AUROC"],
            "best_fadeeva_CI": [best["CI_lo"], best["CI_hi"]],
            "delta_best_minus_ppl": delta,
            "best_meaningfully_better": bool(cis_disjoint),
            "best_meaningfully_worse": bool(cis_disjoint_below),
        })

    summary = {
        "n_cells": len(summary_rows),
        "per_cell": summary_rows,
        "n_cells_fadeeva_meaningfully_better": int(sum(r["best_meaningfully_better"] for r in summary_rows)),
        "n_cells_fadeeva_meaningfully_worse":  int(sum(r["best_meaningfully_worse"] for r in summary_rows)),
        "mean_delta_best_minus_ppl": float(np.mean([r["delta_best_minus_ppl"] for r in summary_rows])),
        "max_delta_best_minus_ppl":  float(np.max([r["delta_best_minus_ppl"] for r in summary_rows])),
        "min_delta_best_minus_ppl":  float(np.min([r["delta_best_minus_ppl"] for r in summary_rows])),
    }
    with open(RESULTS_DIR / "fadeeva_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {RESULTS_DIR/'fadeeva_summary.json'}")

    print(f"\nHeadline: best Fadeeva variant is meaningfully better than seq-PPL in "
          f"{summary['n_cells_fadeeva_meaningfully_better']}/{summary['n_cells']} cells.")
    print(f"Mean delta (best Fadeeva - seq-PPL): {summary['mean_delta_best_minus_ppl']:>+.4f}")
    print(f"Range: [{summary['min_delta_best_minus_ppl']:>+.4f}, {summary['max_delta_best_minus_ppl']:>+.4f}]")


if __name__ == "__main__":
    main()
