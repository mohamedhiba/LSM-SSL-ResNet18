#!/usr/bin/env python
"""Generate 3 slide-ready PNGs from the saved R6/R7 result CSVs (no training).

  1) landcover_multiseed.png   -- with vs without landcover, 3-seed mean +/- SD
  2) canonical_validation.png  -- canonical 0.810 holds at 0.782 +/- 0.025 (3 seeds)
  3) why_multiseed.png         -- single run (0.58) vs 3-seed mean (stable)

Output: figures/R7_landcover_multiseed/slides/
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# The documented single-run draw of the canonical condition (new code + landcover,
# seed 42) that motivated the multi-seed check -- see RESULTS_R4_R6_SUMMARY.md sec 2.
SINGLE_RUN_AUC = 0.5799


def main() -> None:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
    })
    BLUE, ORANGE, GREY = "#2c6fbf", "#e08214", "#888888"

    out = PROJECT_ROOT / "figures/R7_landcover_multiseed/slides"
    out.mkdir(parents=True, exist_ok=True)

    r7 = pd.read_csv(PROJECT_ROOT / "outputs/R7_landcover_multiseed/r7_landcover_multiseed.csv")
    r6 = pd.read_csv(PROJECT_ROOT / "outputs/R6_newcode_landcover_check/r6_multiseed.csv")

    def seed_means(df, **filt):
        d = df
        for k, v in filt.items():
            d = d[d[k] == v]
        return d.groupby("seed")["AUC"].mean()

    # ---------- Figure 1: landcover, multi-seed ----------
    arms = [("with_landcover", "With landcover\n(13 terrain + landcover)", BLUE),
            ("no_landcover", "No landcover\n(13 terrain)", ORANGE)]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for i, (key, label, color) in enumerate(arms):
        sm = seed_means(r7, arm=key)
        mean, sd = float(sm.mean()), float(sm.std(ddof=1))
        ax.bar(i, mean, 0.6, color=color, alpha=0.85, yerr=sd, capsize=8,
               error_kw=dict(elinewidth=2, ecolor="#333"))
        ax.scatter([i] * len(sm), sm.values, color="#222", zorder=5, s=55,
                   label="per-seed mean" if i == 0 else None)
        ax.text(i, mean + sd + 0.012, f"{mean:.3f}\n±{sd:.3f}", ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels([a[1] for a in arms])
    ax.set_ylabel("Downstream AUC  (5-fold spatial CV)")
    ax.set_ylim(0.4, 0.9)
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    a = seed_means(r7, arm="with_landcover").mean(); b = seed_means(r7, arm="no_landcover").mean()
    ax.set_title(f"Landcover effect (3 seeds): +{a-b:.3f} AUC, borderline (p≈0.1)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(); fig.savefig(out / "1_landcover_multiseed.png"); plt.close(fig)

    # ---------- Figure 2: canonical validation ----------
    sm6 = seed_means(r6)
    mean6, sd6 = float(sm6.mean()), float(sm6.std(ddof=1))
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.axhline(0.810, color=GREY, ls="--", lw=2, label="single-run reference (0.810)")
    ax.axhspan(mean6 - sd6, mean6 + sd6, color=BLUE, alpha=0.18,
               label=f"3-seed mean ± SD ({mean6:.3f} ± {sd6:.3f})")
    ax.axhline(mean6, color=BLUE, lw=2)
    xs = np.arange(len(sm6))
    ax.scatter(xs, sm6.values, color=BLUE, s=90, zorder=5)
    for x, (seed, v) in zip(xs, sm6.items()):
        ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=12)
    ax.set_xticks(xs); ax.set_xticklabels([f"seed {s}" for s in sm6.index])
    ax.set_ylabel("Downstream AUC  (5-fold spatial CV)")
    ax.set_ylim(0.70, 0.85)
    ax.set_title(f"Canonical cross-channel holds: 0.810 → {mean6:.3f} ± {sd6:.3f} (3 seeds)")
    ax.legend(loc="lower right", frameon=False, fontsize=12)
    fig.tight_layout(); fig.savefig(out / "2_canonical_validation.png"); plt.close(fig)

    # ---------- Figure 3: why multi-seed matters ----------
    fig, ax = plt.subplots(figsize=(7, 5.2))
    # single run point
    ax.scatter([0], [SINGLE_RUN_AUC], color=ORANGE, s=140, zorder=5)
    ax.text(0, SINGLE_RUN_AUC - 0.022, f"{SINGLE_RUN_AUC:.3f}\n(one unlucky run)", ha="center", fontsize=12, color=ORANGE)
    # 3-seed cloud + mean band
    ax.axhspan(mean6 - sd6, mean6 + sd6, xmin=0.66, xmax=0.97, color=BLUE, alpha=0.18)
    ax.scatter([1.18] * len(sm6), sm6.values, color=BLUE, s=90, zorder=5)
    ax.hlines(mean6, 1.08, 1.28, color=BLUE, lw=2)
    ax.text(1.0, mean6, f"{mean6:.3f}\n± {sd6:.3f}", ha="center", va="center",
            fontsize=12, color=BLUE, fontweight="bold")
    # swing arrow
    ax.annotate("", xy=(0.5, mean6), xytext=(0.5, SINGLE_RUN_AUC),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.8))
    ax.text(0.53, (mean6 + SINGLE_RUN_AUC) / 2, "~0.20 swing\n(same config,\nsame seed 42)",
            fontsize=11, va="center", color="#444")
    ax.set_xticks([0, 1.18]); ax.set_xticklabels(["Single run", "3-seed mean"])
    ax.set_xlim(-0.5, 1.7); ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("Downstream AUC  (5-fold spatial CV)")
    ax.set_title("Single runs swing ±0.15; 3-seed mean is stable (±0.025)")
    fig.tight_layout(); fig.savefig(out / "3_why_multiseed.png"); plt.close(fig)

    print("wrote:")
    for f in sorted(out.glob("*.png")):
        print(" ", f.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
