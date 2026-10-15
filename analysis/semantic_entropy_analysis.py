"""Semantic Entropy head-to-head analysis.

Loads SE outputs (Kuhn 2023 / Farquhar 2024 style) on the first 200 samples
of each cell's binmatched set, pairs with the paper's existing per-sample
PPL (from `tests/09/.../hotpot_phi3_uncertainty_results.pkl`) on the same
200-sample subset, and computes head-to-head AUROCs with stratified
bootstrap CIs.

Why the paired subset (rather than full ~1000 samples per cell): SE requires
N=10 generations per question at T=1.0, so we ran it on the first 200
samples per cell to stay within the compute budget. To compare against
PPL on apples-to-apples data, we restrict PPL to the same first 200
indices.

Outputs:
  - analysis/results/semantic_entropy_aurocs.csv (per-cell SE + PPL AUROC + CIs)
  - analysis/results/semantic_entropy_summary.json (head-to-head verdict)
"""

from __future__ import annotations

import json
import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SE_DIR = REPO_ROOT / "analysis" / "semantic_entropy_results" / "semantic_entropy"
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# (model_short, dataset, se_subdir, pkl_path_relative_to_repo_root)
CELLS = [
    ("Phi-3-mini",   "SQuAD",    "microsoft_Phi-3-mini-4k-instruct/squad",
     "phi/squad/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Phi-3-mini",   "HotpotQA", "microsoft_Phi-3-mini-4k-instruct/hotpot",
     "phi/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Gemma-3-4B",   "SQuAD",    "google_gemma-3-4b-it/squad",
     "gemma/squad/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Gemma-3-4B",   "HotpotQA", "google_gemma-3-4b-it/hotpot",
     "gemma/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Llama-3.1-8B", "SQuAD",    "meta-llama_Llama-3.1-8B-Instruct/squad",
     "llama 8b/squad/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Llama-3.1-8B", "HotpotQA", "meta-llama_Llama-3.1-8B-Instruct/hotpot",
     "llama 8b/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Qwen-2.5-14B", "SQuAD",    "Qwen_Qwen2.5-14B-Instruct/squad",
     "qwen 14b/squad/tests/09/hotpot_phi3_uncertainty_results.pkl"),
    ("Qwen-2.5-14B", "HotpotQA", "Qwen_Qwen2.5-14B-Instruct/hotpot",
     "qwen 14b/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl"),
]


def bootstrap_auroc_ci(y: np.ndarray, score: np.ndarray, B: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Stratified bootstrap AUROC."""
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
    summary_rows = []

    print(f"{'cell':<28} {'feature':<22} {'AUROC':>6}  [{'lo':>5}, {'hi':>5}]")
    print("-" * 80)

    for model, dataset, se_subdir, pkl_rel in CELLS:
        se_path = SE_DIR / se_subdir / "semantic_entropy_n200.json"
        if not se_path.exists():
            print(f"  [skip] {model}/{dataset}: missing {se_path}")
            continue
        se_data = json.loads(se_path.read_text())
        recs = se_data["records"]
        # SE was computed on binmatched[0:n_samples]; record IDs are 0..n-1
        se_ids = np.array([r["id"] for r in recs], dtype=int)
        se_score = np.array([r["semantic_entropy"] for r in recs], dtype=float)
        se_first_gen_ppl = np.array([r["first_gen_perplexity"] for r in recs], dtype=float)
        se_labels = np.array([r["gold_label"] for r in recs], dtype=int)

        # Paper's existing per-sample PPL on the SAME indices (first n_samples)
        pkl_path = REPO_ROOT / pkl_rel
        with open(pkl_path, "rb") as f:
            u = pickle.load(f)
        paper_ppl_all = np.asarray(u["perplexity"], dtype=float)
        paper_y_all = np.asarray(u["labels"], dtype=int)
        # Take the subset matching the SE IDs (which are 0..n-1 in binmatched order)
        paper_ppl_subset = paper_ppl_all[se_ids]
        paper_y_subset = paper_y_all[se_ids]

        # Sanity: labels should agree between SE (from binmatched JSON) and tests/09 (from pkl)
        agree = (se_labels == paper_y_subset).mean() if len(se_labels) == len(paper_y_subset) else float("nan")
        if not np.isclose(agree, 1.0):
            print(f"  [warn] {model}/{dataset}: SE-vs-paper label agreement = {agree:.3f} on n={len(se_labels)}")

        y = se_labels

        # Compute AUROCs + CIs for SE and for paper-PPL on this paired 200-sample subset
        for feat_label, feat_key, score_arr in [
            ("semantic_entropy",  "se",     se_score),
            ("paper_perplexity",  "ppl",    paper_ppl_subset),
            ("first_gen_ppl",     "ppl_fg", se_first_gen_ppl),
        ]:
            mask = np.isfinite(score_arr) & np.isfinite(y)
            if mask.sum() < 30 or len(np.unique(y[mask])) < 2:
                continue
            auc, lo, hi = bootstrap_auroc_ci(y[mask], score_arr[mask])
            rows.append({
                "cell": f"{model}/{dataset}",
                "feature": feat_label,
                "n": int(mask.sum()),
                "pos_rate": float(y[mask].mean()),
                "AUROC": auc, "CI_lo": lo, "CI_hi": hi,
            })
            if feat_label in ("semantic_entropy", "paper_perplexity"):
                print(f"{model+'/'+dataset:<28} {feat_label:<22} {auc:>6.3f}  [{lo:>5.3f}, {hi:>5.3f}]")

        # Head-to-head: SE vs paper-PPL on paired subset
        d = pd.DataFrame([r for r in rows if r["cell"] == f"{model}/{dataset}"]).set_index("feature")
        if "semantic_entropy" in d.index and "paper_perplexity" in d.index:
            se_row = d.loc["semantic_entropy"]
            ppl_row = d.loc["paper_perplexity"]
            cis_disjoint = se_row["CI_lo"] > ppl_row["CI_hi"]
            cis_disjoint_below = se_row["CI_hi"] < ppl_row["CI_lo"]
            summary_rows.append({
                "cell": f"{model}/{dataset}",
                "n": int(se_row["n"]),
                "ppl_AUROC": float(ppl_row["AUROC"]),
                "ppl_CI": [float(ppl_row["CI_lo"]), float(ppl_row["CI_hi"])],
                "se_AUROC": float(se_row["AUROC"]),
                "se_CI": [float(se_row["CI_lo"]), float(se_row["CI_hi"])],
                "delta_se_minus_ppl": float(se_row["AUROC"] - ppl_row["AUROC"]),
                "se_meaningfully_better": bool(cis_disjoint),
                "se_meaningfully_worse": bool(cis_disjoint_below),
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "semantic_entropy_aurocs.csv", index=False)
    print(f"\nwrote {RESULTS_DIR/'semantic_entropy_aurocs.csv'}  ({len(df)} rows)")

    summary = {
        "n_cells": len(summary_rows),
        "per_cell": summary_rows,
        "n_cells_se_meaningfully_better": int(sum(r["se_meaningfully_better"] for r in summary_rows)),
        "n_cells_se_meaningfully_worse":  int(sum(r["se_meaningfully_worse"] for r in summary_rows)),
        "mean_delta_se_minus_ppl": float(np.mean([r["delta_se_minus_ppl"] for r in summary_rows])) if summary_rows else float("nan"),
        "min_delta_se_minus_ppl": float(np.min([r["delta_se_minus_ppl"] for r in summary_rows])) if summary_rows else float("nan"),
        "max_delta_se_minus_ppl": float(np.max([r["delta_se_minus_ppl"] for r in summary_rows])) if summary_rows else float("nan"),
    }
    with open(RESULTS_DIR / "semantic_entropy_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {RESULTS_DIR/'semantic_entropy_summary.json'}")

    print("\n--- head-to-head: Semantic Entropy vs paper's perplexity (on paired n=200 subset) ---")
    for r in summary_rows:
        verdict = "(SE better)" if r["se_meaningfully_better"] else (
                  "(SE worse)"  if r["se_meaningfully_worse"]  else "(CIs overlap)")
        print(f"  {r['cell']:<28} PPL={r['ppl_AUROC']:.3f}  SE={r['se_AUROC']:.3f}  delta={r['delta_se_minus_ppl']:>+.3f}  {verdict}")
    print(f"\nSE meaningfully > PPL in {summary['n_cells_se_meaningfully_better']}/{summary['n_cells']} cells.")
    print(f"SE meaningfully < PPL in {summary['n_cells_se_meaningfully_worse']}/{summary['n_cells']} cells.")
    print(f"Mean delta (SE - PPL): {summary['mean_delta_se_minus_ppl']:>+.4f}")
    print(f"Range: [{summary['min_delta_se_minus_ppl']:>+.4f}, {summary['max_delta_se_minus_ppl']:>+.4f}]")


if __name__ == "__main__":
    main()
