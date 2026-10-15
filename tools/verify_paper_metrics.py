"""Compare saved experiment metrics with reference values.

This legacy checker uses the table and section numbering of the extended
research analyses. It reads pickle, CSV and JSON artifacts and reports numeric
differences with PASS, WARN, FAIL or MISSING statuses. Some comparisons use
stored aggregate tables rather than recomputing every per-sample metric.

For ICNLSP Table 1 and its original controlled classifier settings, use
tools/verify_icnlsp.py; see docs/VERIFICATION.md for the protocol differences.

Run:
  python3 tools/verify_paper_metrics.py
Output:
  analysis/results/metrics_verification_report.md
"""

from __future__ import annotations

import json
import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TOL_HARD = 0.005   # claims within ±0.005 are PASS
TOL_SOFT = 0.020   # within ±0.005..0.020 are PASS-WARN
                   # > 0.020 are FAIL


def status_for(claim: float, derived: float, tol_hard: float = TOL_HARD, tol_soft: float = TOL_SOFT) -> str:
    if not np.isfinite(derived):
        return "MISSING (derived = nan)"
    delta = abs(claim - derived)
    if delta <= tol_hard:
        return "PASS"
    elif delta <= tol_soft:
        return f"WARN (Δ={delta:+.4f})"
    else:
        return f"FAIL (Δ={delta:+.4f})"


def report_line(label: str, claim: float, derived: float, src: str) -> str:
    s = status_for(claim, derived)
    return f"  - {label:<55} claim={claim:<7.4f} derived={derived:<7.4f}  [{s}]  src: {src}"


# =====================================================================
# AUDIT 1: Table 4 main results — Perplexity AUROC for 8 cells
# =====================================================================
# Source: each cell's tests/09 uncertainty pickle has the per-sample
# `perplexity` array + `labels`. AUROC = roc_auc_score(labels, perplexity).
# =====================================================================

T4_CLAIMS_PPL = {
    "Phi-3 + HotpotQA":     ("phi/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl",        0.811),
    "Phi-3 + SQuAD":        ("phi/squad/tests/09/hotpot_phi3_uncertainty_results.pkl",         0.881),
    "Gemma-3 + HotpotQA":   ("gemma/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl",      0.908),
    "Gemma-3 + SQuAD":      ("gemma/squad/tests/09/hotpot_phi3_uncertainty_results.pkl",       0.871),
    "Llama-3.1 + HotpotQA": ("llama 8b/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl",   0.883),
    "Llama-3.1 + SQuAD":    ("llama 8b/squad/tests/09/hotpot_phi3_uncertainty_results.pkl",    0.885),
    "Qwen-2.5 + HotpotQA":  ("qwen 14b/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl",   0.818),
    "Qwen-2.5 + SQuAD":     ("qwen 14b/squad/tests/09/hotpot_phi3_uncertainty_results.pkl",    0.878),
}


def audit_t4_ppl(lines: list) -> None:
    lines.append("\n## Audit 1 — Table 4 (Main Results): Perplexity AUROC per cell")
    lines.append("Source: `<cell>/tests/09/hotpot_phi3_uncertainty_results.pkl`")
    lines.append("")
    for label, (pkl_rel, claim) in T4_CLAIMS_PPL.items():
        path = REPO_ROOT / pkl_rel
        if not path.exists():
            lines.append(f"  - {label:<55} claim={claim:<7.4f} derived=MISSING (no pickle)  src: {pkl_rel}")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        y = np.asarray(data["labels"])
        ppl = np.asarray(data["perplexity"])
        try:
            derived = float(roc_auc_score(y, ppl))
        except Exception as e:
            derived = float("nan")
        lines.append(report_line(label, claim, derived, pkl_rel))


# =====================================================================
# AUDIT 2: Table 4 — SDS+Unc combined AUROC per cell
# =====================================================================
# Source: each cell's tests/10 sds_perplexity pickle has the precomputed
# `auroc` field (saved when the LR detector was originally trained).
# We re-derive instead by re-running the LR with the same StratifiedKFold
# protocol on the per-sample features from tests/09.
# =====================================================================

T4_CLAIMS_SDS_UNC = {
    "Phi-3 + HotpotQA":     ("phi/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl",      0.842),
    "Phi-3 + SQuAD":        ("phi/squad/tests/09/hotpot_phi3_uncertainty_results.pkl",       0.859),
    "Gemma-3 + HotpotQA":   ("gemma/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl",    0.911),
    "Gemma-3 + SQuAD":      ("gemma/squad/tests/09/hotpot_phi3_uncertainty_results.pkl",     0.868),
    "Llama-3.1 + HotpotQA": ("llama 8b/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl", 0.846),
    "Llama-3.1 + SQuAD":    ("llama 8b/squad/tests/09/hotpot_phi3_uncertainty_results.pkl",  0.869),
    "Qwen-2.5 + HotpotQA":  ("qwen 14b/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl", 0.824),
    "Qwen-2.5 + SQuAD":     ("qwen 14b/squad/tests/09/hotpot_phi3_uncertainty_results.pkl",  0.878),
}


def lr_cv_auc_paper_protocol(X: np.ndarray, y: np.ndarray, seed: int = 42) -> float:
    """Reproduce the original paper's SDS+Unc CV protocol exactly:
    raw features (NOT log-transformed) → StandardScaler → LR with C=1.0,
    random_state=42, max_iter=1000 → cross_val_score with cv=5 (default
    StratifiedKFold, no shuffle), scoring='roc_auc' → mean of 5 fold AUROCs.
    Source: `tests/10/10_baseline_comparison_kaggle_clean*.ipynb`.
    """
    from sklearn.model_selection import cross_val_score
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    scores = cross_val_score(lr, Xs, y, cv=5, scoring="roc_auc")
    return float(scores.mean())


def audit_t4_sds_unc(lines: list) -> None:
    lines.append("\n## Audit 2 — Table 4 (Main Results): SDS+Unc combined AUROC per cell")
    lines.append("Method: reproduces the original paper's protocol exactly — raw PPL (NOT log) +")
    lines.append("mean_prob + min_prob + SDS → StandardScaler → LogisticRegression(C=1.0, random_state=42)")
    lines.append("→ cross_val_score(cv=5, scoring='roc_auc'). Source notebook:")
    lines.append("`tests/10/10_baseline_comparison_kaggle_clean*.ipynb`.")
    lines.append("")
    for label, (pkl_rel, claim) in T4_CLAIMS_SDS_UNC.items():
        path = REPO_ROOT / pkl_rel
        if not path.exists():
            lines.append(f"  - {label:<55} claim={claim:<7.4f} derived=MISSING  src: {pkl_rel}")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        y = np.asarray(data["labels"])
        # Original paper used RAW perplexity (not log) in the LR
        X = np.c_[
            np.asarray(data["perplexity"]),
            np.asarray(data["mean_prob"]),
            np.asarray(data["min_prob"]),
            np.asarray(data["SDS"]),
        ]
        try:
            derived = lr_cv_auc_paper_protocol(X, y)
        except Exception as e:
            derived = float("nan")
        lines.append(report_line(label, claim, derived, pkl_rel))


# =====================================================================
# AUDIT 3: Table natural-summary — natural-hall AUROC for the best signal per cell
# =====================================================================
# Source: tests/12/super/*_LLM_LABELED.json per cell. The "Best" column in
# the paper claims which signal is strongest. We verify the AUROC value.
# =====================================================================

T_NH_CLAIMS = {
    "Phi-3 + HotpotQA":     ("phi/hotpot/tests/12/super/natural_hallucination_test_k1_LLM_LABELED.json",     "mean_prob", 0.659, 0.69),
    "Gemma-3 + HotpotQA":   ("gemma/hotpot/tests/12/super/natural_hallucination_test_k1_LLM_LABELED.json",   "sds",       0.732, 0.71),
    "Llama-3.1 + HotpotQA": ("llama 8b/hotpot/tests/12/super/natural_hallucination_test_k1_LLM_LABELED.json","mean_prob", 0.823, 0.68),
    "Qwen-2.5 + HotpotQA":  ("qwen 14b/hotpot/tests/12/super/natural_hallucination_test_k1_LLM_LABELED.json","mean_prob", 0.782, 0.63),
    "Qwen-2.5 + SQuAD":     ("qwen 14b/squad/tests/12/super/natural_hallucination_test_k1_LLM_LABELED.json", "sds",       0.641, 0.93),
}


def audit_natural_summary(lines: list) -> None:
    lines.append("\n## Audit 3 — Table natural-summary: best-signal AUROC + hall rate per cell")
    lines.append("Source: `<cell>/tests/12/super/natural_hallucination_test_k1_LLM_LABELED.json`")
    lines.append("Note: for the best signal in the paper, AUROC depends on whether higher score = more hallucinated.")
    lines.append("      - perplexity, sds: higher = more hallucinated (raw score)")
    lines.append("      - mean_prob, min_prob: higher = less hallucinated (use 1 - prob, i.e. negate)")
    lines.append("")
    for label, (path_rel, signal, claim_auc, claim_hall_rate) in T_NH_CLAIMS.items():
        p = REPO_ROOT / path_rel
        if not p.exists():
            lines.append(f"  - {label:<55} claim_auc={claim_auc:<7.4f} derived=MISSING  src: {path_rel}")
            continue
        data = json.loads(p.read_text())
        y = np.array([d["hallucinated"] for d in data])
        if signal in ("mean_prob", "min_prob"):
            score = -np.array([d[signal] for d in data])
        elif signal == "sds":
            score = np.array([d["sds"] for d in data])
        else:
            score = np.array([d["perplexity"] for d in data])
        try:
            derived = float(roc_auc_score(y, score))
        except Exception as e:
            derived = float("nan")
        hall_rate = float(y.mean())
        s = status_for(claim_auc, derived)
        s_hr = status_for(claim_hall_rate, hall_rate, tol_hard=0.01, tol_soft=0.03)
        lines.append(f"  - {label:<55} signal={signal:<10} claim_auc={claim_auc:<6.3f} derived={derived:<6.3f} [{s}]")
        lines.append(f"  {'':<55}                claim_hall%={claim_hall_rate*100:<5.1f} derived={hall_rate*100:<5.1f} [{s_hr}]  src: {path_rel}")


# =====================================================================
# AUDIT 4: §5.5 reversal numbers — Gemma+HotpotQA 0.285, Qwen+SQuAD 0.200/0.337
# =====================================================================

T_REVERSAL_CLAIMS = {
    "Gemma-3 + HotpotQA k=1 PPL AUROC (§5.5)":     ("gemma/hotpot/tests/12/super/natural_hallucination_test_k1_LLM_LABELED.json", 0.285),
    "Qwen-2.5 + SQuAD k=1 PPL AUROC (§5.5, post-fix to 0.200)": ("qwen 14b/squad/tests/12/super/natural_hallucination_test_k1_LLM_LABELED.json", 0.200),
}


def audit_reversal(lines: list) -> None:
    lines.append("\n## Audit 4 — §5.5 reversal numbers (perplexity AUROC at k=1 on natural-hall)")
    lines.append("Source: tests/12/super/natural_hallucination_test_k1_LLM_LABELED.json")
    lines.append("")
    for label, (path_rel, claim) in T_REVERSAL_CLAIMS.items():
        p = REPO_ROOT / path_rel
        if not p.exists():
            lines.append(f"  - {label:<55} claim={claim:<7.4f} derived=MISSING  src: {path_rel}")
            continue
        data = json.loads(p.read_text())
        y = np.array([d["hallucinated"] for d in data])
        ppl = np.array([d["perplexity"] for d in data])
        try:
            derived = float(roc_auc_score(y, ppl))
        except Exception as e:
            derived = float("nan")
        lines.append(report_line(label, claim, derived, path_rel))


# =====================================================================
# AUDIT 5: §5.6 cross-judge κ = 0.870
# =====================================================================

def audit_kappa(lines: list) -> None:
    lines.append("\n## Audit 5 — §5.6 cross-judge κ = 0.870 (Opus vs Sonnet)")
    p_opus = REPO_ROOT / "analysis" / "vary_k_labeled_claude-opus-4-6" / "vary_k"
    p_sonnet = REPO_ROOT / "analysis" / "vary_k_labeled_claude-sonnet-4-6" / "vary_k"
    if not p_opus.exists() or not p_sonnet.exists():
        lines.append(f"  - cross-judge κ                                  claim=0.8700  derived=MISSING  src: vary_k_labeled_*/")
        return
    opus_labels = []
    sonnet_labels = []
    for opus_file in sorted(p_opus.glob("*/*/vary_k_n100_k[234].json")):
        sonnet_file = p_sonnet / opus_file.relative_to(p_opus)
        if not sonnet_file.exists():
            continue
        op = json.loads(opus_file.read_text())
        so = json.loads(sonnet_file.read_text())
        by_id = {s["id"]: s for s in so}
        for o in op:
            if o["id"] in by_id and o["hallucinated"] is not None and by_id[o["id"]]["hallucinated"] is not None:
                opus_labels.append(o["hallucinated"])
                sonnet_labels.append(by_id[o["id"]]["hallucinated"])
    if not opus_labels:
        lines.append(f"  - cross-judge κ                                  claim=0.8700  derived=MISSING (no pairs)  src: vary_k_labeled_*/")
        return
    from sklearn.metrics import cohen_kappa_score
    derived = float(cohen_kappa_score(opus_labels, sonnet_labels))
    lines.append(f"  - cross-judge κ (Opus vs Sonnet, k=2,3,4 cells)  claim=0.8700  derived={derived:<7.4f}  [{status_for(0.870, derived)}]  n_paired={len(opus_labels)}")


# =====================================================================
# AUDIT 6: §5.7 protocol-sensitivity — Spearman ρ = -0.805
# =====================================================================

def audit_protocol_sensitivity(lines: list) -> None:
    lines.append("\n## Audit 6 — §5.7 protocol-sensitivity: Spearman ρ(hall_rate, k=1 AUROC) = -0.805")
    csv_path = REPO_ROOT / "analysis" / "results" / "protocol_sensitivity_table.csv"
    if not csv_path.exists():
        lines.append(f"  - Spearman ρ                                     claim=-0.8050 derived=MISSING  src: protocol_sensitivity_table.csv")
        return
    df = pd.read_csv(csv_path)
    from scipy.stats import spearmanr
    rho = float(spearmanr(df["hall_rate"], df["k1_AUROC"]).statistic)
    n = len(df)
    lines.append(f"  - Spearman ρ (hall_rate vs k=1 AUROC, n=40 cell-seed) claim=-0.8050 derived={rho:<7.4f}  [{status_for(-0.805, rho)}]  src: protocol_sensitivity_table.csv (n={n})")


# =====================================================================
# AUDIT 7: §5.4 Fadeeva head-to-head numbers
# =====================================================================

FADEEVA_CLAIMS = {
    # cell, feature, claimed_AUROC
    ("Phi-3-mini/SQuAD", "perplexity"):     0.875,
    ("Phi-3-mini/SQuAD", "CCP-sent-max"):   0.872,
    ("Phi-3-mini/SQuAD", "MaxK-10"):        0.874,
    ("Phi-3-mini/SQuAD", "MinK-10pct"):     0.854,
    ("Gemma-3-4B/HotpotQA", "perplexity"):  0.871,
    ("Gemma-3-4B/HotpotQA", "MaxK-10"):     0.888,
    ("Llama-3.1-8B/SQuAD", "perplexity"):   0.929,
    ("Llama-3.1-8B/SQuAD", "MaxK-10"):      0.947,
    ("Qwen-2.5-14B/SQuAD", "perplexity"):   0.869,
    ("Qwen-2.5-14B/SQuAD", "CCP-sent-max"): 0.852,
}


def audit_fadeeva(lines: list) -> None:
    lines.append("\n## Audit 7 — §5.4 Fadeeva head-to-head AUROCs (sample of table cells)")
    csv_path = REPO_ROOT / "analysis" / "results" / "fadeeva_aurocs.csv"
    if not csv_path.exists():
        lines.append(f"  - all Fadeeva claims                              MISSING  src: fadeeva_aurocs.csv")
        return
    df = pd.read_csv(csv_path)
    for (cell, feat), claim in FADEEVA_CLAIMS.items():
        row = df[(df["cell"] == cell) & (df["feature"] == feat)]
        if row.empty:
            derived = float("nan")
        else:
            derived = float(row.iloc[0]["AUROC"])
        lines.append(f"  - {cell+'/'+feat:<55} claim={claim:<7.4f} derived={derived:<7.4f}  [{status_for(claim, derived)}]")


# =====================================================================
# AUDIT 8: §5.4 Semantic Entropy claims
# =====================================================================

SE_CLAIMS = {
    "Phi-3-mini/SQuAD":     {"ppl": 0.878, "se": 0.572, "delta": -0.305},
    "Phi-3-mini/HotpotQA":  {"ppl": 0.738, "se": 0.698, "delta": -0.040},
    "Gemma-3-4B/SQuAD":     {"ppl": 0.816, "se": 0.563, "delta": -0.253},
    "Gemma-3-4B/HotpotQA":  {"ppl": 0.918, "se": 0.592, "delta": -0.326},
    "Llama-3.1-8B/SQuAD":   {"ppl": 0.899, "se": 0.576, "delta": -0.322},
    "Llama-3.1-8B/HotpotQA":{"ppl": 0.868, "se": 0.625, "delta": -0.243},
    "Qwen-2.5-14B/SQuAD":   {"ppl": 0.910, "se": 0.601, "delta": -0.309},
    "Qwen-2.5-14B/HotpotQA":{"ppl": 0.798, "se": 0.669, "delta": -0.129},
}


def audit_semantic_entropy(lines: list) -> None:
    lines.append("\n## Audit 8 — §5.4 Semantic Entropy head-to-head AUROCs (all 8 cells)")
    csv_path = REPO_ROOT / "analysis" / "results" / "semantic_entropy_aurocs.csv"
    if not csv_path.exists():
        lines.append(f"  - all SE claims                                   MISSING  src: semantic_entropy_aurocs.csv")
        return
    df = pd.read_csv(csv_path)
    for cell, claims in SE_CLAIMS.items():
        ppl_row = df[(df["cell"] == cell) & (df["feature"] == "paper_perplexity")]
        se_row = df[(df["cell"] == cell) & (df["feature"] == "semantic_entropy")]
        ppl_d = float(ppl_row.iloc[0]["AUROC"]) if not ppl_row.empty else float("nan")
        se_d = float(se_row.iloc[0]["AUROC"]) if not se_row.empty else float("nan")
        delta_d = se_d - ppl_d
        lines.append(f"  - {cell+' PPL':<55} claim={claims['ppl']:<7.4f} derived={ppl_d:<7.4f}  [{status_for(claims['ppl'], ppl_d)}]")
        lines.append(f"  - {cell+' SE':<55}  claim={claims['se']:<7.4f} derived={se_d:<7.4f}  [{status_for(claims['se'], se_d)}]")
        lines.append(f"  - {cell+' Δ':<55}   claim={claims['delta']:<+7.4f} derived={delta_d:<+7.4f}  [{status_for(claims['delta'], delta_d)}]")


# =====================================================================
# AUDIT 9: Appendix M rigor audit numbers
# =====================================================================

def audit_appendix_m(lines: list) -> None:
    lines.append("\n## Audit 9 — Appendix M (rigor audit): R10 + R6 headline numbers")
    summary_path = REPO_ROOT / "analysis" / "results" / "rigor_audit_summary.json"
    if not summary_path.exists():
        lines.append(f"  - Appendix M claims                               MISSING  src: rigor_audit_summary.json")
        return
    summary = json.loads(summary_path.read_text())
    # Paper claims: max sub-quartile AUROC delta = 0.11 (perplexity), 0.10 (SDS+Unc)
    # max |Δ AUROC| under GroupKFold = 0.0017
    r10_ppl = summary.get("r10_perplexity_max_quartile_delta", float("nan"))
    r10_sds = summary.get("r10_sds_unc_max_quartile_delta", float("nan"))
    r6_max = summary.get("r6_max_abs_AUROC_delta", float("nan"))
    lines.append(f"  - R10 max perplexity quartile delta              claim=0.1100  derived={r10_ppl:<7.4f}  [{status_for(0.110, r10_ppl, 0.01, 0.02)}]")
    lines.append(f"  - R10 max SDS+Unc quartile delta                 claim=0.1010  derived={r10_sds:<7.4f}  [{status_for(0.101, r10_sds, 0.01, 0.02)}]")
    lines.append(f"  - R6 max |Δ AUROC| under GroupKFold              claim=0.0017  derived={r6_max:<7.4f}  [{status_for(0.0017, r6_max, 0.001, 0.005)}]")


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    lines: list[str] = []
    lines.append(f"# Paper-metrics verification report")
    lines.append(f"")
    lines.append(f"Generated by `tools/verify_paper_metrics.py`. Each entry re-derives a numerical")
    lines.append(f"claim from the paper's tables against the raw source data in the repo, and reports")
    lines.append(f"PASS / WARN / FAIL based on the absolute delta.")
    lines.append(f"")
    lines.append(f"- **PASS**: |claim − derived| ≤ 0.005")
    lines.append(f"- **WARN**: 0.005 < |claim − derived| ≤ 0.020")
    lines.append(f"- **FAIL**: |claim − derived| > 0.020")
    lines.append(f"- **MISSING**: source data not present in repo")

    audit_t4_ppl(lines)
    audit_t4_sds_unc(lines)
    audit_natural_summary(lines)
    audit_reversal(lines)
    audit_kappa(lines)
    audit_protocol_sensitivity(lines)
    audit_fadeeva(lines)
    audit_semantic_entropy(lines)
    audit_appendix_m(lines)

    # Summary of statuses
    text = "\n".join(lines)
    passes = text.count("[PASS]")
    warns = text.count("[WARN")
    fails = text.count("[FAIL")
    missings = text.count("MISSING")
    lines.append("\n## Summary")
    lines.append(f"")
    lines.append(f"- PASS: **{passes}**")
    lines.append(f"- WARN: **{warns}**")
    lines.append(f"- FAIL: **{fails}**")
    lines.append(f"- MISSING source: **{missings}**")

    out_path = REPO_ROOT / "analysis" / "results" / "metrics_verification_report.md"
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
