#!/usr/bin/env python
"""Build a leak-safe SUBSAMPLE cache for the KY flat-CV run (boundless=False, interior only).

Reads with boundless=False (the GDAL boundless=True leak was ~11MB/patch). All KY patches
are interior, so this is byte-identical to the boundless read. Outputs a compact float16
package ready to upload + run with run_kentucky_gpu.py --cache-mode --cv-mode flat.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.patch_dataset import list_raster_files  # noqa: E402

RAST = Path("/Users/mohamedhiba/Projects/processed/rasters_cleaned_10m")
LAB_IDX = ROOT / "data/processed/patches/labeled_patch_index_ps64_kentucky_kgs6c_1to2.csv"
OUT = ROOT / "ky_flat_pkg"; (OUT / "cache").mkdir(parents=True, exist_ok=True); (OUT / "indices").mkdir(exist_ok=True)
PS = 64
N_LAB = int(sys.argv[1]) if len(sys.argv) > 1 else 5000     # subsample size (stratified, keeps 1:2)
N_UNLAB = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
SEED = 42


def read_patch(srcs, r0, c0):
    w = Window(c0, r0, PS, PS)
    return np.stack([s.read(1, window=w, boundless=False).astype("float32") for s in srcs], 0)


def main():
    rf = list_raster_files(RAST)
    srcs = [rasterio.open(p) for p in rf]
    W, H = srcs[0].width, srcs[0].height
    rng = np.random.default_rng(SEED)

    # ---- labeled: interior, stratified subsample ----
    df = pd.read_csv(LAB_IDX)
    inside = ((df.window_row_start >= 0) & (df.window_col_start >= 0)
              & (df.window_row_stop <= H) & (df.window_col_stop <= W))
    df = df[inside].reset_index(drop=True)
    frac = min(1.0, N_LAB / len(df))
    pos = df[df["label"] == 1]; neg = df[df["label"] == 0]
    n_pos = max(2, int(round(len(pos) * frac))); n_neg = max(2, int(round(len(neg) * frac)))
    sub = pd.concat([pos.sample(n=n_pos, random_state=SEED), neg.sample(n=n_neg, random_state=SEED)])
    sub = sub.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"labeled subsample: {len(sub)}  balance={sub['label'].value_counts().to_dict()}", flush=True)
    lab = np.empty((len(sub), len(rf), PS, PS), dtype="float32")
    for i, row in sub.iterrows():
        lab[i] = read_patch(srcs, int(row.window_row_start), int(row.window_col_start))
        if i % 1000 == 0: print(f"  labeled {i}/{len(sub)}", flush=True)
    np.savez_compressed(OUT / "cache/ky_labeled.npz", patches=lab.astype("float16"))
    sub.to_csv(OUT / "indices/ky_labeled_sub.csv", index=False)
    print(f"labeled cache {lab.shape} saved; NaN={int(np.isnan(lab).sum())}", flush=True)

    # ---- unlabeled: random interior ps64 patches ----
    half = PS // 2
    rows, recs = [], []
    seen = set()
    attempts = 0
    while len(rows) < N_UNLAB and attempts < 500000:
        attempts += 1
        r = int(rng.integers(half, H - half)); c = int(rng.integers(half, W - half))
        if (r, c) in seen: continue
        p = read_patch(srcs, r - half, c - half)
        if p.shape != (len(rf), PS, PS): continue
        if not np.isfinite(p).all() or (p <= -9990).all(axis=0).mean() > 0.5: continue  # skip mostly-nodata
        seen.add((r, c)); rows.append(p)
        recs.append({"sample_id": f"KU_{len(rows):06d}", "x": 0.0, "y": 0.0, "label": 0,
                     "source": "unlabeled", "cluster_id": 0,
                     "window_row_start": r - half, "window_row_stop": r - half + PS,
                     "window_col_start": c - half, "window_col_stop": c - half + PS,
                     "valid_patch": True, "nodata_ratio": 0.0})
        if len(rows) % 1000 == 0: print(f"  unlabeled {len(rows)}/{N_UNLAB}", flush=True)
    unl = np.stack(rows, 0)
    np.savez_compressed(OUT / "cache/ky_unlabeled.npz", patches=unl.astype("float16"))
    pd.DataFrame(recs).to_csv(OUT / "indices/ky_unlabeled.csv", index=False)
    print(f"unlabeled cache {unl.shape} saved", flush=True)
    for s in srcs: s.close()
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
