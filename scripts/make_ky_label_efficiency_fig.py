#!/usr/bin/env python
"""Plot the KY label-efficiency curve: mean AUC vs training-label budget, scratch vs SSL.

Reads outputs/KY_label_efficiency/ky_label_efficiency.csv (method, seed, fraction,
mean_train_n, mean_fold_auc, fold_aucs) and writes:
  fig_label_efficiency_curve.{png,pdf}   AUC vs fraction, one line/method (+-seed SD band)
  fig_label_efficiency_delta.{png,pdf}   (SSL - scratch) vs fraction
  ky_label_efficiency_summary.csv
The scientific question: does the SSL-scratch gap OPEN at low label fractions?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"scratch": "#888888", "masked_recon": "#3a9c6e", "cross_channel": "#2c7fb6",
          "sequential": "#8856a7", "strip_jigsaw": "#d7663a"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=ROOT / "outputs/KY_label_efficiency/ky_label_efficiency.csv")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/R8seq_KY")
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)

    g = df.groupby(["method", "fraction"])
    summ = g["mean_fold_auc"].agg(["mean", "std", "count"]).reset_index()
    summ["train_n"] = g["mean_train_n"].mean().values
    summ.to_csv(args.results.parent / "ky_label_efficiency_summary.csv", index=False)
    print(summ.round(4).to_string())

    methods = [m for m in ["scratch", "masked_recon", "cross_channel", "sequential", "strip_jigsaw"]
               if m in set(summ["method"])]
    fracs = sorted(df["fraction"].unique())

    # --- curve ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in methods:
        s = summ[summ["method"] == m].sort_values("fraction")
        c = COLORS.get(m, "#333")
        ax.plot(s["fraction"], s["mean"], "-o", color=c, label=m, lw=2)
        ax.fill_between(s["fraction"], s["mean"] - s["std"], s["mean"] + s["std"],
                        color=c, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks(fracs); ax.set_xticklabels([f"{int(f*100)}%" for f in fracs])
    ax.set_xlabel("Training-label budget (fraction of available labels; log scale)")
    ax.set_ylabel("Mean test AUC (5-cluster spatial CV, seeds)")
    ax.set_title("Kentucky label-efficiency — does SSL help when labels are scarce?\n"
                 "(shaded = seed-to-seed SD)")
    ax.axhline(0.5, ls="--", c="crimson", lw=1, alpha=0.7)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.fig_dir / "fig_label_efficiency_curve.png", dpi=150)
    fig.savefig(args.fig_dir / "fig_label_efficiency_curve.pdf"); plt.close(fig)

    # --- delta vs scratch ---
    if "scratch" in methods:
        piv = summ.pivot(index="fraction", columns="method", values="mean")
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in [x for x in methods if x != "scratch"]:
            ax.plot(piv.index, piv[m] - piv["scratch"], "-o", color=COLORS.get(m, "#333"),
                    label=f"{m} - scratch", lw=2)
        ax.axhline(0, c="k", lw=1)
        ax.set_xscale("log"); ax.set_xticks(fracs)
        ax.set_xticklabels([f"{int(f*100)}%" for f in fracs])
        ax.set_xlabel("Training-label budget (log scale)")
        ax.set_ylabel("AUC delta vs scratch")
        ax.set_title("Kentucky — SSL minus scratch vs label budget\n"
                     "(positive at low budget = SSL helps when labels are scarce)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.fig_dir / "fig_label_efficiency_delta.png", dpi=150)
        fig.savefig(args.fig_dir / "fig_label_efficiency_delta.pdf"); plt.close(fig)
    print(f"\nwrote figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
