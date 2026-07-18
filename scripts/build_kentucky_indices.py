#!/usr/bin/env python
"""Build the Kentucky (KGS six-county) balanced patch index — the KY analogue of
scripts/build_562_indices.py.

Unlike the NYC 562 set (which arrived pre-balanced with negatives + clusters), the
KY inventory gives POSITIVES ONLY. This script therefore does the extra prep NYC
did not need:

  1. Load the deduplicated KGS positives (all label=1), keep only those whose
     center pixel is valid across all 13 KY terrain rasters.
  2. Build a valid-pixel mask (AND across the 13 rasters) once and cache it.
  3. Draw buffered-random negatives from the valid mask, EXCLUDING a buffer around
     every positive (respects the positional-uncertainty halo + road-corridor bias,
     KENTUCKY_KGS_PLAN.md §4-5). Default ratio 1:2 (pos:neg).
  4. Assign spatial clusters (OrdinaryKMeans on x/y) for spatially-blocked 5-fold CV;
     negatives inherit the nearest positive-centroid cluster.
  5. Assemble the samples table with the columns create_patch_index requires
     (sample_id, x, y, label, source, cluster_id, factor_01..14) — factors are
     zero-filled: the CNN reads rasters, factors are only stored for provenance.
  6. Run create_patch_index at the chosen patch size (default ps64) against the KY
     raster dir and save the index.

This is DATA PREP only — no model training. Training runs later on the rented GPU.

Notes / deliberate simplifications (vs KENTUCKY_KGS_PLAN.md):
  * Negatives are buffered-random, NOT RF PU-bagging. PU-bagging (src/pu_bagging.py)
    is the documented refinement; buffered-random is a correct, fast first version.
  * KY channel scheme = 13 terrain (ksat replaces NYC lithology) + valid-context
    mask = 14-ch input. A NYC-pretrained encoder is NOT channel-compatible with KY.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.patch_dataset import (  # noqa: E402
    FACTOR_NAMES,
    audit_raster_alignment,
    create_patch_index,
    list_raster_files,
    save_patch_index,
)


class OrdinaryKMeans:
    """Minimal KMeans++ (2D spatial blocking). Self-contained to avoid the
    geopandas import chain in src.spatial_cluster_balance; mirrors its behavior."""

    def __init__(self, n_clusters: int, random_state: int, n_init: int = 20,
                 max_iter: int = 300, tol: float = 1e-4) -> None:
        self.n_clusters, self.random_state = int(n_clusters), int(random_state)
        self.n_init, self.max_iter, self.tol = n_init, max_iter, tol
        self.cluster_centers_ = None

    @staticmethod
    def _d2(X, C):
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)

    def _one(self, X, rng):
        n = X.shape[0]
        C = np.empty((self.n_clusters, X.shape[1]))
        C[0] = X[int(rng.integers(0, n))]
        closest = self._d2(X, C[:1]).ravel()
        for k in range(1, self.n_clusters):
            tot = float(closest.sum())
            j = int(rng.integers(0, n)) if tot <= 0 else int(rng.choice(n, p=closest / tot))
            C[k] = X[j]
            closest = np.minimum(closest, self._d2(X, C[k:k + 1]).ravel())
        for _ in range(self.max_iter):
            lab = self._d2(X, C).argmin(1)
            newC = np.array([X[lab == k].mean(0) if np.any(lab == k) else C[k]
                             for k in range(self.n_clusters)])
            if np.max(np.abs(newC - C)) < self.tol:
                C = newC
                break
            C = newC
        lab = self._d2(X, C).argmin(1)
        inertia = float(self._d2(X, C)[np.arange(len(X)), lab].sum())
        return C, lab, inertia

    def fit_predict(self, X):
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            C, lab, inertia = self._one(X, rng)
            if best is None or inertia < best[2]:
                best = (C, lab, inertia)
        self.cluster_centers_ = best[0]
        return best[1]

    def predict(self, X):
        return self._d2(X, self.cluster_centers_).argmin(1)

# --- fixed paths (KY processed data lives OUTSIDE the repo) ----------------------
KY_ROOT = Path("/Users/mohamedhiba/Projects/processed")
RASTER_DIR = KY_ROOT / "rasters_cleaned_10m"
POSITIVES_CSV = (
    KY_ROOT / "samples"
    / "kentucky_kgs_v4_six_county_modeling_candidates_deduplicated.csv"
)
OUT_DIR = PROJECT_ROOT / "data/processed/patches"
CACHE_DIR = PROJECT_ROOT / "data/processed/kentucky_cache"
NODATA = -9999.0


def build_valid_mask(raster_files, cache_path: Path) -> np.ndarray:
    """Boolean mask (H, W): True where every raster is finite and != nodata."""
    if cache_path.exists():
        print(f"  loading cached valid mask {cache_path.name}", flush=True)
        return np.load(cache_path)
    mask = None
    for path in raster_files:
        with rasterio.open(path) as src:
            band = src.read(1)
        good = np.isfinite(band) & (band != NODATA)
        mask = good if mask is None else (mask & good)
        print(f"    AND {path.name:34s} valid so far: {int(mask.sum()):,}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mask)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--neg-ratio", type=float, default=2.0, help="negatives per positive")
    ap.add_argument("--buffer-m", type=float, default=650.0,
                    help="exclude negatives within this distance of any positive (meters)")
    ap.add_argument("--n-clusters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-positives", type=int, default=0,
                    help="0 = use all valid positives")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print("=== building Kentucky KGS patch index ===", flush=True)
    print(f"    patch_size={args.patch_size}  neg_ratio={args.neg_ratio}  "
          f"buffer={args.buffer_m}m  clusters={args.n_clusters}  seed={args.seed}", flush=True)

    raster_files = list_raster_files(RASTER_DIR)
    audit = audit_raster_alignment(raster_files)
    tr = audit.transform
    inv = ~tr
    print(f"  {len(raster_files)} rasters aligned; grid {audit.width}x{audit.height} "
          f"CRS {audit.crs}", flush=True)

    # --- positives: keep center-valid only -------------------------------------
    pos = pd.read_csv(POSITIVES_CSV, low_memory=False)
    xs, ys = pos["x"].to_numpy(), pos["y"].to_numpy()
    cols = np.floor(inv.a * xs + inv.b * ys + inv.c).astype(int)
    rows = np.floor(inv.d * xs + inv.e * ys + inv.f).astype(int)
    valid_mask = build_valid_mask(raster_files, CACHE_DIR / "valid_mask.npy")
    inb = (rows >= 0) & (rows < audit.height) & (cols >= 0) & (cols < audit.width)
    keep = inb.copy()
    keep[inb] &= valid_mask[rows[inb], cols[inb]]
    pos = pos.loc[keep, ["x", "y"]].reset_index(drop=True)
    if args.max_positives and len(pos) > args.max_positives:
        pos = pos.sample(n=args.max_positives, random_state=args.seed).reset_index(drop=True)
    n_pos = len(pos)
    print(f"  positives: {n_pos:,} center-valid (of {len(xs):,})", flush=True)

    # --- spatial clusters on positives (blocking only) -------------------------
    km = OrdinaryKMeans(n_clusters=args.n_clusters, random_state=args.seed)
    pos_xy = pos[["x", "y"]].to_numpy(dtype="float64")
    pos_cluster = km.fit_predict(pos_xy)
    print(f"  positive cluster sizes: "
          f"{ {int(c): int((pos_cluster==c).sum()) for c in range(args.n_clusters)} }",
          flush=True)

    # --- buffered-random negatives ---------------------------------------------
    n_neg = int(round(n_pos * args.neg_ratio))
    buffer_px = args.buffer_m / audit.resolution[0]
    ptree = cKDTree(pos_xy)
    valid_rc = np.argwhere(valid_mask)          # (row, col) of every valid pixel
    print(f"  valid pixel pool: {len(valid_rc):,}; drawing {n_neg:,} negatives "
          f"(buffer {args.buffer_m}m = {buffer_px:.1f}px)", flush=True)
    neg_x, neg_y = [], []
    batch = max(n_neg * 4, 100_000)
    while len(neg_x) < n_neg:
        pick = valid_rc[rng.integers(0, len(valid_rc), size=batch)]
        cx = tr.c + (pick[:, 1] + 0.5) * tr.a + (pick[:, 0] + 0.5) * tr.b
        cy = tr.f + (pick[:, 1] + 0.5) * tr.d + (pick[:, 0] + 0.5) * tr.e
        dist, _ = ptree.query(np.column_stack([cx, cy]), k=1)
        ok = dist > args.buffer_m
        for x, y in zip(cx[ok], cy[ok]):
            neg_x.append(float(x)); neg_y.append(float(y))
            if len(neg_x) >= n_neg:
                break
    neg_xy = np.column_stack([neg_x[:n_neg], neg_y[:n_neg]])
    neg_cluster = km.predict(neg_xy)            # inherit nearest positive-centroid block
    print(f"  negatives: {len(neg_xy):,}; cluster sizes: "
          f"{ {int(c): int((neg_cluster==c).sum()) for c in range(args.n_clusters)} }",
          flush=True)

    # --- assemble samples table (create_patch_index schema) --------------------
    def _rows(xy, cluster, label, prefix, src):
        d = {
            "sample_id": [f"{prefix}_{i:06d}" for i in range(len(xy))],
            "x": xy[:, 0], "y": xy[:, 1],
            "label": label, "source": src, "cluster_id": cluster.astype(int),
        }
        for f in FACTOR_NAMES:
            d[f] = 0.0
        return pd.DataFrame(d)

    samples = pd.concat([
        _rows(pos_xy, pos_cluster, 1, "KGS6C_POS", "kgs_v4"),
        _rows(neg_xy, neg_cluster, 0, "KGS6C_NEG", "buffered_random"),
    ], ignore_index=True)

    print(f"\n  building patch index for {len(samples):,} samples "
          f"at ps{args.patch_size} ...", flush=True)
    idx = create_patch_index(
        samples_df=samples, raster_files=raster_files, patch_size=args.patch_size,
    )
    nvalid = int(idx["valid_patch"].astype(bool).sum())
    tag = f"ps{args.patch_size}_kentucky_kgs6c_1to{int(args.neg_ratio)}"
    out_csv = OUT_DIR / f"labeled_patch_index_{tag}.csv"
    save_patch_index(idx, out_csv)

    # --- report ----------------------------------------------------------------
    v = idx.loc[idx["valid_patch"].astype(bool)]
    print(f"\n=== DONE: {out_csv.name} ===", flush=True)
    print(f"  total rows {len(idx):,}  valid_patch {nvalid:,} "
          f"({100*nvalid/len(idx):.1f}%)", flush=True)
    print(f"  valid by label:\n{v['label'].value_counts().to_string()}", flush=True)
    print(f"  valid by cluster:\n"
          f"{v.groupby(['cluster_id','label']).size().unstack(fill_value=0).to_string()}",
          flush=True)


if __name__ == "__main__":
    main()
