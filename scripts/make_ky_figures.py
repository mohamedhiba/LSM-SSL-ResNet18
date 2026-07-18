#!/usr/bin/env python
"""Build the Kentucky (KGS six-county, ps64) SSL-vs-scratch table + figures.

Reads the per-seed ky_results.csv files (method, seed, mean_fold_auc, fold_aucs=
';'-joined 5 fold AUCs) pulled from the RunPod run, computes per-method mean +/- SD
over seeds and the SSL-scratch delta (real iff |delta| > 2x max seed-SD), and writes
figures matching figures/R8seq_562/ (single 10m resolution here):
  fig1_mean_auc_bars, fig2_ssl_minus_scratch, fig3_perfold_spread, fig4_worst_fold_auc
plus ky_summary.csv.

Usage: python scripts/make_ky_figures.py [--results-dir outputs/KY_kgs6c_ps64_runpod]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["scratch", "masked_recon", "cross_channel", "strip_jigsaw", "sequential"]
SSL_C = "#3a9c6e"
SCR_C = "#c2c2c2"
NEG_C = "#d7663a"


def load(results_dir: Path) -> pd.DataFrame:
    files = glob.glob(str(results_dir / "ky_results_s*.csv")) + \
            glob.glob(str(results_dir / "run_s*/ky_results.csv")) + \
            glob.glob(str(results_dir / "ky_results.csv"))
    if not files:
        raise SystemExit(f"no ky_results csv under {results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["method", "seed"], keep="last")
    # explode per-fold aucs
    folds = df["fold_aucs"].apply(lambda s: [float(x) for x in str(s).split(";")])
    df["fold_list"] = folds
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = [m for m in ORDER if m in set(df["method"])]
    for m in methods:
        sub = df[df["method"] == m]
        seed_means = sub["mean_fold_auc"].astype(float).values
        allfolds = np.concatenate(sub["fold_list"].values)
        rows.append({
            "method": m,
            "n_seeds": len(sub),
            "mean_AUC": float(np.mean(seed_means)),
            "seed_to_seed_SD": float(np.std(seed_means, ddof=0)),
            "worst_fold_AUC": float(np.min(allfolds)),
            "best_fold_AUC": float(np.max(allfolds)),
        })
    t = pd.DataFrame(rows).set_index("method")
    if "scratch" in t.index:
        s_mean = t.loc["scratch", "mean_AUC"]
        s_sd = t.loc["scratch", "seed_to_seed_SD"]
        t["delta_vs_scratch"] = t["mean_AUC"] - s_mean
        t["gate_2xSD"] = 2 * np.maximum(s_sd, t["seed_to_seed_SD"])
        t["verdict"] = [
            "—" if m == "scratch" else
            ("within noise" if abs(d) < g else ("SSL helps" if d > 0 else "SSL hurts"))
            for m, d, g in zip(t.index, t["delta_vs_scratch"], t["gate_2xSD"])
        ]
    return t


def fig_mean_bars(t, fig_dir):
    m = [x for x in ORDER if x in t.index]
    vals = t.loc[m, "mean_AUC"].values
    errs = t.loc[m, "seed_to_seed_SD"].values
    colors = [SCR_C if x == "scratch" else (NEG_C if x == "strip_jigsaw" else SSL_C) for x in m]
    fig, ax = plt.subplots(figsize=(8, 5))
    b = ax.bar(range(len(m)), vals, yerr=errs, capsize=5, color=colors, edgecolor="k")
    ax.axhline(0.5, ls="--", c="crimson", lw=1.2, label="random (0.5)")
    if "scratch" in t.index:
        ax.axhline(t.loc["scratch", "mean_AUC"], ls=":", c="k", lw=1,
                   label="scratch baseline")
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width()/2, v + 0.008, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(m))); ax.set_xticklabels(m, rotation=20, ha="right")
    ax.set_ylabel("Mean test AUC (5-cluster spatial CV, 3 seeds)")
    ax.set_ylim(0.5, max(0.8, vals.max() + 0.05))
    ax.set_title("Kentucky KGS six-county (10m, ps64, 1:2) — SSL vs scratch\n"
                 "error bars = seed-to-seed SD")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig1_mean_auc_bars.png", dpi=150)
    fig.savefig(fig_dir / "fig1_mean_auc_bars.pdf"); plt.close(fig)


def fig_delta(t, fig_dir):
    m = [x for x in ORDER if x in t.index and x != "scratch"]
    d = t.loc[m, "delta_vs_scratch"].values
    g = t.loc[m, "gate_2xSD"].values
    colors = [SSL_C if dv > 0 else NEG_C for dv in d]
    fig, ax = plt.subplots(figsize=(8, 5))
    b = ax.bar(range(len(m)), d, color=colors, edgecolor="k")
    ax.errorbar(range(len(m)), d, yerr=g, fmt="none", ecolor="k", capsize=6,
                label="±2x seed-SD (significance gate)")
    ax.axhline(0, c="k", lw=1)
    for r, dv in zip(b, d):
        ax.text(r.get_x() + r.get_width()/2, dv + (0.002 if dv >= 0 else -0.004),
                f"{dv:+.3f}", ha="center", va="bottom" if dv >= 0 else "top", fontsize=10)
    ax.set_xticks(range(len(m))); ax.set_xticklabels(m, rotation=20, ha="right")
    ax.set_ylabel("AUC delta vs scratch")
    ax.set_title("Kentucky — SSL minus scratch (real only if bar clears the error bar)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_ssl_minus_scratch.png", dpi=150)
    fig.savefig(fig_dir / "fig2_ssl_minus_scratch.pdf"); plt.close(fig)


def fig_spread(df, fig_dir):
    m = [x for x in ORDER if x in set(df["method"])]
    data = [np.concatenate(df[df["method"] == x]["fold_list"].values) for x in m]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=m, showmeans=True, patch_artist=True, widths=0.55)
    for patch, x in zip(bp["boxes"], m):
        patch.set_facecolor(SCR_C if x == "scratch" else (NEG_C if x == "strip_jigsaw" else SSL_C))
        patch.set_alpha(0.6)
    for i, dd in enumerate(data, 1):
        ax.scatter(np.random.normal(i, 0.05, len(dd)), dd, s=14, c="k", alpha=0.4, zorder=3)
    ax.axhline(0.5, ls="--", c="crimson", lw=1.2)
    ax.set_xticklabels(m, rotation=20, ha="right")
    ax.set_ylabel("Per-fold test AUC (5 folds x 3 seeds)")
    ax.set_title("Kentucky — per-fold AUC spread (why we report multi-seed)")
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_perfold_spread.png", dpi=150)
    fig.savefig(fig_dir / "fig3_perfold_spread.pdf"); plt.close(fig)


def fig_worst(t, fig_dir):
    m = [x for x in ORDER if x in t.index]
    vals = t.loc[m, "worst_fold_AUC"].values
    colors = [SCR_C if x == "scratch" else (NEG_C if x == "strip_jigsaw" else SSL_C) for x in m]
    fig, ax = plt.subplots(figsize=(8, 5))
    b = ax.bar(range(len(m)), vals, color=colors, edgecolor="k")
    ax.axhline(0.5, ls="--", c="crimson", lw=1.2, label="random (0.5)")
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width()/2, v + 0.008, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(m))); ax.set_xticklabels(m, rotation=20, ha="right")
    ax.set_ylabel("Worst single-fold AUC (robustness on hard clusters)")
    ax.set_ylim(0.4, max(0.75, vals.max() + 0.05))
    ax.set_title("Kentucky — worst spatial fold (does SSL help where scratch struggles?)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_worst_fold_auc.png", dpi=150)
    fig.savefig(fig_dir / "fig4_worst_fold_auc.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=ROOT / "outputs/KY_kgs6c_ps64_runpod")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/R8seq_KY")
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.results_dir)
    t = summarize(df)
    t.to_csv(args.results_dir / "ky_summary.csv")
    print("=== KY summary (mean AUC +/- seed-SD) ===")
    cols = ["n_seeds", "mean_AUC", "seed_to_seed_SD", "worst_fold_AUC"]
    if "delta_vs_scratch" in t.columns:
        cols += ["delta_vs_scratch", "gate_2xSD", "verdict"]
    print(t[cols].round(4).to_string())
    fig_mean_bars(t, args.fig_dir); fig_delta(t, args.fig_dir)
    fig_spread(df, args.fig_dir); fig_worst(t, args.fig_dir)
    print(f"\nwrote figures to {args.fig_dir} and {args.results_dir/'ky_summary.csv'}")


if __name__ == "__main__":
    main()
