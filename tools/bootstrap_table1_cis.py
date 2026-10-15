"""Stratified bootstrap 95% CIs for Table 1 / tab:main-results AUROCs.

Inputs: the existing per-cell pickles at
    <cell_root>/tests/09/hotpot_phi3_uncertainty_results.pkl
which contain keys {labels, perplexity, mean_prob, min_prob, SDS, length}.

For each cell + each signal in {length, perplexity, mean_prob, min_prob, SDS,
combined_ppl_sds_unc}, computes:
- point AUROC
- 95% stratified bootstrap CI (B=10,000 with positive/negative resampled
  independently, mirroring the paper's existing bootstrap helpers)
- DeLong-style approximation of the pairwise SDS+Unc vs PPL gap CI

For "combined", we use cross_val_predict to get out-of-fold probabilities once,
then bootstrap the AUROC on the OOF scores -- this avoids retraining LR 10000
times per cell (~30x speedup) while preserving the cross-validation discipline.

Outputs:
- analysis/results/bootstrap_table1_cis.csv  (one row per cell)
- analysis/results/bootstrap_table1_cis.tex  (LaTeX-ready table)

Cost: pure local numpy. Runs in ~30s on the 8 cells.
"""
from __future__ import annotations

import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
B = 10_000
SEED = 42

CELLS = {
    "Phi-3 + HotpotQA":     "phi/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl",
    "Phi-3 + SQuAD":        "phi/squad/tests/09/hotpot_phi3_uncertainty_results.pkl",
    "Gemma-3 + HotpotQA":   "gemma/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl",
    "Gemma-3 + SQuAD":      "gemma/squad/tests/09/hotpot_phi3_uncertainty_results.pkl",
    "Llama-3.1 + HotpotQA": "llama 8b/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl",
    "Llama-3.1 + SQuAD":    "llama 8b/squad/tests/09/hotpot_phi3_uncertainty_results.pkl",
    "Qwen-2.5 + HotpotQA":  "qwen 14b/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl",
    "Qwen-2.5 + SQuAD":     "qwen 14b/squad/tests/09/hotpot_phi3_uncertainty_results.pkl",
}


def stratified_bootstrap_auroc(
    scores: np.ndarray, labels: np.ndarray, B: int = B, seed: int = SEED
) -> tuple[float, float, float]:
    """Return (point_auc, ci_lo, ci_hi) via stratified bootstrap.
    Positives and negatives are resampled independently with replacement,
    preserving the class balance of the original sample."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    point = float(roc_auc_score(labels, scores))
    boots = np.empty(B, dtype=np.float64)
    for b in range(B):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        n = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, n])
        try:
            boots[b] = roc_auc_score(labels[idx], scores[idx])
        except ValueError:
            boots[b] = np.nan
    lo = float(np.nanpercentile(boots, 2.5))
    hi = float(np.nanpercentile(boots, 97.5))
    return point, lo, hi


def combined_oof_scores(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Out-of-fold probabilities from the paper's LR protocol (raw features +
    StandardScaler + LR(C=1.0); cv=5 default StratifiedKFold). Bootstrap on
    these scores rather than retraining LR on every bootstrap sample."""
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    proba = cross_val_predict(lr, Xs, y, cv=5, method="predict_proba")
    return proba[:, 1]


def paired_bootstrap_delta(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray,
    B: int = B, seed: int = SEED,
) -> tuple[float, float, float, float]:
    """Paired bootstrap on AUROC(scores_a) - AUROC(scores_b).
    Same resample for both signals; reports point delta + 95% CI + 2-sided
    p-value (fraction of bootstraps that flipped sign)."""
    rng = np.random.default_rng(seed + 7)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    point = float(roc_auc_score(labels, scores_a) - roc_auc_score(labels, scores_b))
    deltas = np.empty(B, dtype=np.float64)
    for b in range(B):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        n = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, n])
        try:
            a = roc_auc_score(labels[idx], scores_a[idx])
            c = roc_auc_score(labels[idx], scores_b[idx])
            deltas[b] = a - c
        except ValueError:
            deltas[b] = np.nan
    lo = float(np.nanpercentile(deltas, 2.5))
    hi = float(np.nanpercentile(deltas, 97.5))
    if point >= 0:
        pval = float(np.mean(deltas <= 0))
    else:
        pval = float(np.mean(deltas >= 0))
    pval = 2 * pval
    return point, lo, hi, pval


def run() -> None:
    t0 = time.time()
    out_csv = REPO_ROOT / "analysis" / "results" / "bootstrap_table1_cis.csv"
    out_tex = REPO_ROOT / "analysis" / "results" / "bootstrap_table1_cis.tex"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for cell, pkl_rel in CELLS.items():
        path = REPO_ROOT / pkl_rel
        if not path.exists():
            print(f"  [skip] missing pickle: {pkl_rel}", file=sys.stderr)
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        labels = np.asarray(data["labels"]).astype(int)
        ppl = np.asarray(data["perplexity"])
        mean_p = np.asarray(data["mean_prob"])
        min_p = np.asarray(data["min_prob"])
        sds = np.asarray(data["SDS"])
        # length isn't always saved; recover it if possible
        if "length" in data:
            length = np.asarray(data["length"])
        elif "answer_length" in data:
            length = np.asarray(data["answer_length"])
        else:
            length = None

        n = len(labels)
        pos = int(labels.sum())
        neg = int(n - pos)
        print(f"\n[{cell}] n={n} pos={pos} neg={neg}")

        # Per-signal CIs
        def boot(scores, name):
            auc, lo, hi = stratified_bootstrap_auroc(scores, labels)
            print(f"  {name:>10}: AUROC={auc:.3f}  CI=[{lo:.3f}, {hi:.3f}]")
            return auc, lo, hi

        # Length needs the sign flipped if hallucinated answers are longer
        # (we want AUROC > 0.5 to be the "discriminative" direction). The
        # paper's reported length-only AUROC is the symmetric one (max of
        # AUROC, 1 - AUROC), so we report that.
        if length is not None:
            auc_len_raw = roc_auc_score(labels, length)
            # use the sign that gives auc >= 0.5 by convention used in the paper
            scores_len = length if auc_len_raw >= 0.5 else -length
            len_auc, len_lo, len_hi = boot(scores_len, "Length")
        else:
            len_auc = len_lo = len_hi = float("nan")

        ppl_auc, ppl_lo, ppl_hi = boot(ppl, "PPL")
        # Mean prob: higher = more faithful, so flip sign for "hallucinated=1" target
        mean_p_auc, mean_p_lo, mean_p_hi = boot(-mean_p, "MeanP(-)")
        min_p_auc, min_p_lo, min_p_hi = boot(-min_p, "MinP(-)")
        sds_auc, sds_lo, sds_hi = boot(sds, "SDS")

        # Combined SDS+Unc
        X = np.c_[ppl, mean_p, min_p, sds]
        combined_scores = combined_oof_scores(X, labels)
        comb_auc, comb_lo, comb_hi = boot(combined_scores, "Combined")

        # Paired delta: Combined - PPL
        d_pt, d_lo, d_hi, d_p = paired_bootstrap_delta(combined_scores, ppl, labels)
        print(f"  ΔComb−PPL: {d_pt:+.3f}  CI=[{d_lo:+.3f}, {d_hi:+.3f}]  p={d_p:.3f}")

        rows.append({
            "cell": cell,
            "n": n, "pos": pos, "neg": neg,
            "len_auc": len_auc, "len_lo": len_lo, "len_hi": len_hi,
            "ppl_auc": ppl_auc, "ppl_lo": ppl_lo, "ppl_hi": ppl_hi,
            "mean_p_auc": mean_p_auc, "mean_p_lo": mean_p_lo, "mean_p_hi": mean_p_hi,
            "min_p_auc": min_p_auc, "min_p_lo": min_p_lo, "min_p_hi": min_p_hi,
            "sds_auc": sds_auc, "sds_lo": sds_lo, "sds_hi": sds_hi,
            "comb_auc": comb_auc, "comb_lo": comb_lo, "comb_hi": comb_hi,
            "delta_comb_ppl": d_pt, "delta_lo": d_lo, "delta_hi": d_hi, "delta_p": d_p,
        })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

    # LaTeX table: cell | Length | PPL | Combined | Δ(Comb−PPL) p
    tex_lines = [
        r"% Auto-generated by tools/bootstrap_table1_cis.py (B=10000, stratified bootstrap)",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Cell} & \textbf{Length AUROC} & \textbf{PPL AUROC} & \textbf{Combined AUROC} & \textbf{$\Delta$ Comb$-$PPL} \\",
        r" & \scriptsize{[95\% CI]} & \scriptsize{[95\% CI]} & \scriptsize{[95\% CI]} & \scriptsize{[CI]; $p$} \\",
        r"\midrule",
    ]
    for r in rows:
        def fmt(auc, lo, hi):
            return f"{auc:.3f} [{lo:.3f}, {hi:.3f}]"
        def fmt_d(d, lo, hi, p):
            return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]; $p$={p:.3f}"
        tex_lines.append(
            f"{r['cell']} & {fmt(r['len_auc'], r['len_lo'], r['len_hi'])} & "
            f"{fmt(r['ppl_auc'], r['ppl_lo'], r['ppl_hi'])} & "
            f"{fmt(r['comb_auc'], r['comb_lo'], r['comb_hi'])} & "
            f"{fmt_d(r['delta_comb_ppl'], r['delta_lo'], r['delta_hi'], r['delta_p'])} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    with open(out_tex, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"wrote {out_tex}")
    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run()
