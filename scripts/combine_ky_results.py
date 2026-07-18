#!/usr/bin/env python
"""Combine per-seed KY ky_results.csv files into the 5-method comparison table.

Reads any number of ky_results.csv (each has rows: method, seed, mean_fold_auc, fold_aucs)
from the given dirs/globs, concatenates, and reports per-method mean +/- SD over seeds
plus each SSL method's delta vs scratch (with the >=2x seed-SD "trust" gate).

Works whether the 5 methods were produced in one run or several (identical seeds/folds/
indices make them comparable). Usage:
  python scripts/combine_ky_results.py /workspace/ky/full_s42 /workspace/ky/full_s123 ...
  python scripts/combine_ky_results.py --glob '/workspace/ky/*/ky_results.csv'
"""
from __future__ import annotations

import argparse
import glob as globmod
import sys
from pathlib import Path

import pandas as pd

ORDER = ["scratch", "masked_recon", "cross_channel", "strip_jigsaw", "sequential"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="dirs containing ky_results.csv, or csv files")
    ap.add_argument("--glob", default=None, help="glob for ky_results.csv files")
    ap.add_argument("--out", default=None, help="write combined table csv here")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        p = Path(p)
        files.append(p if p.suffix == ".csv" else p / "ky_results.csv")
    if args.glob:
        files += [Path(f) for f in globmod.glob(args.glob)]
    files = [f for f in files if f.exists()]
    if not files:
        sys.exit("no ky_results.csv found")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["method", "seed"], keep="last")
    print(f"loaded {len(files)} files -> {len(df)} (method,seed) rows\n")

    methods = [m for m in ORDER if m in set(df["method"])] + \
              [m for m in df["method"].unique() if m not in ORDER]
    g = df.groupby("method")["mean_fold_auc"]
    table = pd.DataFrame({"mean_auc": g.mean(), "sd_auc": g.std(ddof=0), "n_seeds": g.count()}).reindex(methods)

    print("=== KENTUCKY KGS6C ps64 — 5-method SSL-vs-scratch (mean +/- SD over seeds) ===")
    for m in methods:
        r = table.loc[m]
        print(f"  {m:14s} AUC = {r.mean_auc:.4f} +/- {r.sd_auc:.4f}  (n={int(r.n_seeds)})")

    if "scratch" in table.index:
        sm, ssd = table.loc["scratch", "mean_auc"], table.loc["scratch", "sd_auc"]
        print("\n  delta vs scratch (trust only if |delta| > 2x max seed-SD):")
        for m in methods:
            if m == "scratch":
                continue
            d = table.loc[m, "mean_auc"] - sm
            sd = max(ssd, table.loc[m, "sd_auc"])
            verdict = "within noise" if abs(d) < 2 * sd else ("SSL HELPS" if d > 0 else "SSL hurts")
            print(f"    {m:14s} {d:+.4f}  (2x SD {2*sd:.4f}) -> {verdict}")

    if args.out:
        table.to_csv(args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
