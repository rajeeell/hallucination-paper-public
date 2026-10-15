"""Plot Gemma/HotpotQA perplexity distributions from saved features.

Input:
  gemma/hotpot/tests/09/hotpot_phi3_uncertainty_results.pkl
Output:
  analysis/figures/gemma_perplexity_distribution_paper.png

Run:
  python3 analysis/make_perplexity_distribution_fig.py
"""

from __future__ import annotations

import pathlib
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PKL = REPO_ROOT / "gemma" / "hotpot" / "tests" / "09" / "hotpot_phi3_uncertainty_results.pkl"
OUT = REPO_ROOT / "analysis" / "figures" / "gemma_perplexity_distribution_paper.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    labels = np.asarray(data["labels"], dtype=int)
    ppl = np.asarray(data["perplexity"], dtype=float)
    ppl_clipped = np.clip(ppl, None, np.quantile(ppl, 0.98))

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    bins = np.linspace(ppl_clipped.min(), ppl_clipped.max(), 40)
    ax.hist(ppl_clipped[labels == 0], bins=bins, color="#4477AA", alpha=0.65,
            label="Faithful", edgecolor="black", linewidth=0.4)
    ax.hist(ppl_clipped[labels == 1], bins=bins, color="#EE6677", alpha=0.65,
            label="Hallucinated", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Perplexity")
    ax.set_ylabel("Count")
    ax.set_title("Gemma-3-4B + HotpotQA (controlled)")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"  faithful: n={(labels == 0).sum()}, median_ppl={np.median(ppl[labels == 0]):.3f}")
    print(f"  hallucinated: n={(labels == 1).sum()}, median_ppl={np.median(ppl[labels == 1]):.3f}")


if __name__ == "__main__":
    main()
