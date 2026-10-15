"""§5.7 protocol-sensitivity analysis: multi-seed AUROC variance + hallucination-rate mediation.

Inputs (labeled with claude-opus-4-6 by tools/label_vary_k.py):
  - 5 seeds × 8 cells of k=1 generation+labels (n=100 each)
      * seeds 1-4: analysis/multiseed_k1_labeled_claude-opus-4-6/multiseed_k1/<model>/<dataset>/k1_s{seed}.json
      * seed 42:   analysis/vary_k_labeled_claude-opus-4-6/vary_k/<model>/<dataset>/vary_k_n100_k1.json
  - per-cell memorization probe (200 samples): analysis/memorization_results/...

Outputs:
  - analysis/results/protocol_sensitivity_table.csv (per-cell-per-seed AUROC + hall rate)
  - analysis/results/protocol_sensitivity_summary.csv (per-cell variance summary)
  - analysis/results/protocol_sensitivity_correlations.json (hall-rate-vs-AUROC + bootstrap)
  - analysis/figures/protocol_sensitivity.{pdf,png} — 2-panel: box-plot per cell + scatter

What this answers
-----------------
1. *AUROC variance across seeds*. How much does k=1 perplexity AUROC move if we
   re-sample 100 questions? Per cell.
2. *Hall-rate mediation*. Does (cell, seed)-pair hall rate predict AUROC? If
   yes, the §5.5 reversal is a function of sample difficulty.
3. *Boundary characterization for §5.5*. At what hall rate does PPL stop
   discriminating?
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
FIG_DIR = REPO_ROOT / "analysis" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MULTISEED_DIR = REPO_ROOT / "analysis" / "multiseed_k1_labeled_claude-opus-4-6" / "multiseed_k1"
SEED_42_DIR = REPO_ROOT / "analysis" / "vary_k_labeled_claude-opus-4-6" / "vary_k"

# Paths to the paper's original natural-hallucination protocol labels — used
# as overlay points on the multi-seed figure to show how far the reported
# numbers diverge from a seeded-random-sampling baseline.
NH_PATHS = {
    "Phi-3-mini/HotpotQA":   REPO_ROOT / "phi" / "hotpot" / "tests" / "12" / "super" / "natural_hallucination_test_k1_LLM_LABELED.json",
    "Gemma-3-4B/HotpotQA":   REPO_ROOT / "gemma" / "hotpot" / "tests" / "12" / "super" / "natural_hallucination_test_k1_LLM_LABELED.json",
    "Llama-3.1-8B/HotpotQA": REPO_ROOT / "llama 8b" / "hotpot" / "tests" / "12" / "super" / "natural_hallucination_test_k1_LLM_LABELED.json",
    "Qwen-2.5-14B/HotpotQA": REPO_ROOT / "qwen 14b" / "hotpot" / "tests" / "12" / "super" / "natural_hallucination_test_k1_LLM_LABELED.json",
    "Qwen-2.5-14B/SQuAD":    REPO_ROOT / "qwen 14b" / "squad" / "tests" / "12" / "super" / "natural_hallucination_test_k1_LLM_LABELED.json",
}


CELLS = [
    ("microsoft_Phi-3-mini-4k-instruct", "squad",  "Phi-3-mini",   "SQuAD"),
    ("microsoft_Phi-3-mini-4k-instruct", "hotpot", "Phi-3-mini",   "HotpotQA"),
    ("google_gemma-3-4b-it",             "squad",  "Gemma-3-4B",   "SQuAD"),
    ("google_gemma-3-4b-it",             "hotpot", "Gemma-3-4B",   "HotpotQA"),
    ("meta-llama_Llama-3.1-8B-Instruct", "squad",  "Llama-3.1-8B", "SQuAD"),
    ("meta-llama_Llama-3.1-8B-Instruct", "hotpot", "Llama-3.1-8B", "HotpotQA"),
    ("Qwen_Qwen2.5-14B-Instruct",        "squad",  "Qwen-2.5-14B", "SQuAD"),
    ("Qwen_Qwen2.5-14B-Instruct",        "hotpot", "Qwen-2.5-14B", "HotpotQA"),
]


def auroc_and_hall(path: pathlib.Path) -> tuple[float, float, int]:
    data = json.loads(path.read_text())
    y = np.array([d["hallucinated"] for d in data], dtype=int)
    ppl = np.array([d["perplexity"] for d in data], dtype=float)
    mask = np.isfinite(ppl)
    y, ppl = y[mask], ppl[mask]
    if len(np.unique(y)) < 2:
        return float("nan"), float(y.mean()), int(len(y))
    return float(roc_auc_score(y, ppl)), float(y.mean()), int(len(y))


def boot_spearman(xs: np.ndarray, ys: np.ndarray, B: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(xs)
    rhos = np.full(B, np.nan)
    for b in range(B):
        idx = rng.choice(n, n, replace=True)
        if len(set(xs[idx])) < 2 or len(set(ys[idx])) < 2:
            continue
        rhos[b], _ = spearmanr(xs[idx], ys[idx])
    rhos = rhos[np.isfinite(rhos)]
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return {
        "rho": float(spearmanr(xs, ys).statistic),
        "boot_lo": float(lo),
        "boot_hi": float(hi),
        "frac_positive": float((rhos > 0).mean()),
        "frac_negative": float((rhos < 0).mean()),
        "B": B,
    }


def main() -> None:
    rows = []
    for model_dir, dataset, model_short, dataset_label in CELLS:
        # seed 42 (existing) — uses the vary_k file naming
        seed42_path = SEED_42_DIR / model_dir / dataset / "vary_k_n100_k1.json"
        if seed42_path.exists():
            auroc, hall, n = auroc_and_hall(seed42_path)
            rows.append({
                "model": model_short, "dataset": dataset_label,
                "cell": f"{model_short}/{dataset_label}",
                "seed": 42, "n": n,
                "k1_AUROC": auroc, "hall_rate": hall,
            })
        # seeds 1-4 (new multi-seed)
        for s in (1, 2, 3, 4):
            p = MULTISEED_DIR / model_dir / dataset / f"k1_s{s}.json"
            if not p.exists():
                continue
            auroc, hall, n = auroc_and_hall(p)
            rows.append({
                "model": model_short, "dataset": dataset_label,
                "cell": f"{model_short}/{dataset_label}",
                "seed": s, "n": n,
                "k1_AUROC": auroc, "hall_rate": hall,
            })

    df = pd.DataFrame(rows).sort_values(["cell", "seed"]).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "protocol_sensitivity_table.csv", index=False)
    print(f"wrote {RESULTS_DIR/'protocol_sensitivity_table.csv'} ({len(df)} rows)")

    # ---- per-cell variance summary
    summary = df.groupby("cell").agg(
        n_seeds=("seed", "count"),
        AUROC_mean=("k1_AUROC", "mean"),
        AUROC_std=("k1_AUROC", "std"),
        AUROC_min=("k1_AUROC", "min"),
        AUROC_max=("k1_AUROC", "max"),
        hall_rate_mean=("hall_rate", "mean"),
        hall_rate_std=("hall_rate", "std"),
        hall_rate_min=("hall_rate", "min"),
        hall_rate_max=("hall_rate", "max"),
    ).reset_index()
    summary["AUROC_range"] = summary["AUROC_max"] - summary["AUROC_min"]
    summary["hall_rate_range"] = summary["hall_rate_max"] - summary["hall_rate_min"]
    summary["reverses_in_any_seed"] = df.groupby("cell")["k1_AUROC"].min().reset_index(drop=True) < 0.5
    summary.to_csv(RESULTS_DIR / "protocol_sensitivity_summary.csv", index=False)
    print(f"wrote {RESULTS_DIR/'protocol_sensitivity_summary.csv'}")

    print("\n--- per-cell variance (5 seeds × 8 cells) ---")
    print(summary.to_string(index=False))

    # ---- hall-rate-vs-AUROC scatter (pooled all (cell, seed) pairs)
    xs = df["hall_rate"].to_numpy()
    ys = df["k1_AUROC"].to_numpy()
    boot_pooled = boot_spearman(xs, ys)

    # ---- per-cell hall-rate-vs-AUROC (within-cell across seeds)
    within_cell = {}
    for cell, g in df.groupby("cell"):
        if len(g) >= 3:
            within_cell[cell] = {
                "n": len(g),
                "spearman_hall_vs_AUROC": float(spearmanr(g["hall_rate"], g["k1_AUROC"]).statistic),
            }

    corr_blob = {
        "pooled_spearman_hall_vs_AUROC": boot_pooled,
        "within_cell_spearman": within_cell,
        "n_rows": len(df),
    }
    with open(RESULTS_DIR / "protocol_sensitivity_correlations.json", "w") as f:
        json.dump(corr_blob, f, indent=2)
    print(f"\nwrote {RESULTS_DIR/'protocol_sensitivity_correlations.json'}")
    print(f"\n--- pooled hall-rate vs AUROC ---")
    print(f"Spearman ρ = {boot_pooled['rho']:>+.3f}")
    print(f"Bootstrap 95% CI = [{boot_pooled['boot_lo']:>+.3f}, {boot_pooled['boot_hi']:>+.3f}]")
    print(f"Fraction of bootstraps positive: {boot_pooled['frac_positive']:.3f}")
    print(f"Fraction of bootstraps negative: {boot_pooled['frac_negative']:.3f}")

    # ---- figure: 2 panels
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_colors = {
        "Phi-3-mini":   "#4477AA",
        "Gemma-3-4B":   "#EE6677",
        "Llama-3.1-8B": "#228833",
        "Qwen-2.5-14B": "#CCBB44",
    }
    dataset_markers = {"SQuAD": "o", "HotpotQA": "s"}

    # load NH protocol AUROCs for overlay
    nh_auroc_by_cell = {}
    for cell, path in NH_PATHS.items():
        if path.exists():
            auroc, _, _ = auroc_and_hall(path)
            nh_auroc_by_cell[cell] = auroc

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: AUROC box-plot per cell, ordered by mean.
    # Red diamonds overlay the original natural-hall protocol AUROC where available.
    ax = axes[0]
    cells_ordered = summary.sort_values("AUROC_mean")["cell"].tolist()
    box_data = [df[df["cell"] == c]["k1_AUROC"].to_numpy() for c in cells_ordered]
    bp = ax.boxplot(box_data, tick_labels=cells_ordered, vert=False, widths=0.55,
                    patch_artist=True, medianprops={"color": "black"})
    for patch, cell in zip(bp["boxes"], cells_ordered):
        model_short = cell.split("/")[0]
        patch.set_facecolor(model_colors.get(model_short, "#cccccc"))
        patch.set_alpha(0.6)
    for i, c in enumerate(cells_ordered, start=1):
        ys_cell = df[df["cell"] == c]["k1_AUROC"]
        ax.scatter(ys_cell, [i] * len(ys_cell),
                   color="black", s=20, zorder=3, alpha=0.7)
        if c in nh_auroc_by_cell:
            ax.scatter(nh_auroc_by_cell[c], i, marker="D",
                       color="red", edgecolors="black", linewidths=0.8,
                       s=85, zorder=4, label="natural-hall protocol" if i == 1 else None)
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("k=1 perplexity AUROC")
    ax.set_title("Per-cell AUROC: 5 seeds (boxes) vs natural-hall protocol (red ◆)")
    ax.grid(axis="x", alpha=0.3)
    # custom legend marker for the NH overlay
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="D", color="w", markerfacecolor="red",
                              markeredgecolor="black", markersize=8, label="natural-hall protocol")],
              loc="lower right", frameon=True, fontsize=7)

    # Panel 2: hall_rate vs AUROC scatter, one point per (cell, seed)
    ax = axes[1]
    for _, row in df.iterrows():
        ax.scatter(
            row["hall_rate"], row["k1_AUROC"],
            marker=dataset_markers[row["dataset"]],
            color=model_colors[row["model"]],
            s=70, edgecolors="black", linewidths=0.6, alpha=0.85, zorder=3,
        )
    # fitted line for visual aid
    x_fit = np.linspace(df["hall_rate"].min(), df["hall_rate"].max(), 100)
    fit = np.polyfit(df["hall_rate"], df["k1_AUROC"], deg=1)
    ax.plot(x_fit, fit[0] * x_fit + fit[1], color="grey", linestyle="--", alpha=0.6,
            label=f"linear fit (slope={fit[0]:+.2f})")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Hallucination rate (per cell × seed)")
    ax.set_ylabel("k=1 perplexity AUROC")
    ax.set_title(f"Hall-rate mediation (Spearman ρ = {boot_pooled['rho']:>+.2f})")
    ax.grid(alpha=0.3)

    # legend
    from matplotlib.lines import Line2D
    handles = (
        [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="black",
                markersize=8, label=name) for name, c in model_colors.items()]
        + [Line2D([0], [0], marker=m, color="w", markerfacecolor="grey", markeredgecolor="black",
                markersize=8, label=f"dataset = {d}") for d, m in dataset_markers.items()]
    )
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=7)

    fig.suptitle("§5.7 protocol-sensitivity: AUROC variance across 5 random seeds × 8 cells")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "protocol_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "protocol_sensitivity.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {FIG_DIR/'protocol_sensitivity.pdf'}")
    print(f"wrote {FIG_DIR/'protocol_sensitivity.png'}")


if __name__ == "__main__":
    main()
