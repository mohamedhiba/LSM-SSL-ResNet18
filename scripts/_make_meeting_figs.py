#!/usr/bin/env python
"""Generate meeting figures from the R8 results + KGS Kentucky inventory."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures/meeting"
FIG.mkdir(parents=True, exist_ok=True)


def fig1_2x2():
    s = pd.read_csv(ROOT / "outputs/R8_resolution_2x2_summary.csv").set_index("arm")
    res = ["30m", "10m"]
    scr = [(s.loc[f"scratch_{r}", "mean_AUC"], s.loc[f"scratch_{r}", "seed_to_seed_SD"]) for r in res]
    ssl = [(s.loc[f"ssl_{r}", "mean_AUC"], s.loc[f"ssl_{r}", "seed_to_seed_SD"]) for r in res]
    x = np.arange(2); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 5))
    b1 = ax.bar(x - w/2, [m for m, _ in scr], w, yerr=[e for _, e in scr], capsize=5,
                label="Scratch (no SSL)", color="#c2c2c2", edgecolor="k")
    b2 = ax.bar(x + w/2, [m for m, _ in ssl], w, yerr=[e for _, e in ssl], capsize=5,
                label="SSL (cross-channel)", color="#3a9c6e", edgecolor="k")
    ax.axhline(0.5, ls="--", c="crimson", lw=1.2, label="random (AUC 0.5)")
    for bars, vals in [(b1, scr), (b2, ssl)]:
        for rect, (m, _) in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width()/2, m + 0.012, f"{m:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(["30 m", "10 m"], fontsize=12)
    ax.set_ylabel("Mean test AUC (5-cluster spatial CV)", fontsize=11)
    ax.set_ylim(0, 0.72)
    ax.set_title("R8 — 10 m vs 30 m terrain features\n"
                 "RESOLUTION is within noise for BOTH scratch and SSL\n"
                 "(error bars = seed-to-seed SD)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    fig.text(0.5, -0.02,
             "Compare resolution WITHIN a method (each pair), not scratch-vs-SSL: "
             "scratch needs LR 1e-4, cross-channel SSL needs LR 2e-5 (it collapses at 1e-4).",
             ha="center", fontsize=8, style="italic", wrap=True)
    fig.tight_layout(); fig.savefig(FIG / "fig1_r8_2x2.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig2_fold_spread():
    parts = []
    for arm, p in [("SSL 30m", "outputs/R8_resolution_30m/r8_30m.csv"),
                   ("SSL 10m", "outputs/R8_resolution_10m/r8_10m.csv")]:
        d = pd.read_csv(ROOT / p); parts.append(pd.DataFrame({"arm": arm, "AUC": d["AUC"]}))
    scr = pd.read_csv(ROOT / "outputs/R8_resolution_scratch/r8_scratch.csv")
    for setup, lab in [("30m", "Scratch 30m"), ("10m", "Scratch 10m")]:
        d = scr[scr["setup"] == setup]; parts.append(pd.DataFrame({"arm": lab, "AUC": d["AUC"].values}))
    df = pd.concat(parts, ignore_index=True)
    order = ["Scratch 30m", "Scratch 10m", "SSL 30m", "SSL 10m"]
    data = [df.loc[df["arm"] == a, "AUC"].values for a in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=order, showmeans=True, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], ["#c2c2c2", "#c2c2c2", "#3a9c6e", "#3a9c6e"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, d in enumerate(data, 1):
        ax.scatter(np.random.normal(i, 0.05, len(d)), d, s=14, c="k", alpha=0.4, zorder=3)
    ax.axhline(0.5, ls="--", c="crimson", lw=1.2)
    ax.set_ylabel("Per-fold test AUC", fontsize=11)
    ax.set_title("Why we report multi-seed: per-fold AUC spans ~0.2–0.9\n"
                 "(each dot = one of 5 spatial folds × seeds; spread >> the 30m–10m gap)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig2_fold_spread.png", dpi=150); plt.close(fig)


def _load_points():
    fc = json.loads((ROOT / "data/kentucky/kgs_july2022_points.geojson").read_text())
    rows = []
    for f in fc["features"]:
        g = f.get("geometry")
        if not g:
            continue
        p = f.get("properties", {})
        rows.append({"lon": g["coordinates"][0], "lat": g["coordinates"][1],
                     "mt": p.get("Movement_Type"), "county": p.get("County"),
                     "fl": p.get("Failure_Location")})
    return pd.DataFrame(rows)


def fig3_map():
    df = _load_points()
    top = [m for m, _ in Counter(df["mt"].dropna()).most_common(4)]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#d7301f", "#fc8d59", "#2c7fb6", "#41ab5d", "#888"]
    for mt, c in zip(top, colors):
        sub = df[df["mt"] == mt]
        ax.scatter(sub["lon"], sub["lat"], s=14, c=c, label=f"{mt} ({len(sub)})", alpha=0.7)
    other = df[~df["mt"].isin(top)]
    ax.scatter(other["lon"], other["lat"], s=10, c="#ccc", label=f"other ({len(other)})", alpha=0.5)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"KGS July 2022 landslide inventory — {len(df)} points\n"
                 "eastern Kentucky (Perry/Breathitt/Owsley/Leslie/Clay)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right"); ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(); fig.savefig(FIG / "fig3_kentucky_map.png", dpi=150); plt.close(fig)


def fig4_distributions():
    df = _load_points()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mt = Counter(df["mt"].dropna()).most_common(6)
    axes[0].barh([k for k, _ in mt][::-1], [v for _, v in mt][::-1], color="#2c7fb6")
    axes[0].set_title("Movement type (landslide kind)"); axes[0].set_xlabel("count")
    fl = Counter([x for x in df["fl"].dropna() if x]).most_common(6)
    axes[1].barh([k for k, _ in fl][::-1], [v for _, v in fl][::-1], color="#d7762b")
    axes[1].set_title("Failure_Location → ROAD-CORRIDOR BIAS\n('above road' dominates)")
    axes[1].set_xlabel("count")
    fig.tight_layout(); fig.savefig(FIG / "fig4_kgs_distributions.png", dpi=150); plt.close(fig)


def fig5_spatial_blocks():
    c = pd.read_csv(ROOT / "data/kentucky/kentucky_positive_centers.csv")
    base = c[c["jitter_idx"] == 0]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for fold in sorted(base["fold_id"].unique()):
        sub = base[base["fold_id"] == fold]
        ax.scatter(sub["x"], sub["y"], s=16, label=f"fold {fold} (n={len(sub)})", alpha=0.7)
    ax.set_xlabel("UTM Easting (m)"); ax.set_ylabel("UTM Northing (m)")
    ax.set_title("Point-to-patch design: 5 spatial-CV blocks (KMeans on location)\n"
                 "each point also spawns K=6 jittered patches (positional-uncertainty aug)", fontsize=11)
    ax.legend(fontsize=8); ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(); fig.savefig(FIG / "fig5_spatial_blocks.png", dpi=150); plt.close(fig)


def main():
    fig1_2x2(); fig2_fold_spread(); fig3_map(); fig4_distributions(); fig5_spatial_blocks()
    print("wrote figures:")
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
