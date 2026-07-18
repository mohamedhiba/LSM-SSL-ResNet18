#!/usr/bin/env python
"""Export a compact KY patch-cache package (NO rasters shipped) — KY analogue of
scripts/export_colab_cache.py.

The KY 10 m rasters (~1.9 GB) can't cross the ~150 KB/s uplink. Instead ship the
raw pre-normalization patch arrays the datasets would have read, plus their index
CSVs; on the pod colab_patch monkeypatches the datasets to serve cache[i] for
index row i (valid_patch-filtered, reset order). Byte-identical to the rasters.

KY is ps64 (4x the pixels of NYC ps32), so patches are big:
  - labeled full = 27,594 patches -> 5.9 GB f32 / 2.9 GB f16 (can't ship).
    => subsample a balanced set (--n-labeled, default 3200) to fit the uplink.
  - unlabeled SSL pool -> cap --n-unlabeled (default 3000), float16.
Both caches float16 to stay compact. Preflight verifies the float32 EXTRACTION is
byte-exact vs rasters (max diff 0.0) before the float16 cast, and checks for NaN.

Outputs into ky_package/:
  cache/cache_ky_{labeled,unlabeled}.npz   float16 raw patches (N,13,64,64)
  indices/{labeled_ky_ps64_sub.csv, ky_unlabeled_ps64_n{N}.csv}
  colab/colab_patch.py                     (copied)
  src/                                     (copied, unchanged)
  manifest.json
Run LOCALLY (rasters on the Mac):
  .venv/bin/python scripts/export_ky_cache.py --n-labeled 3200 --n-unlabeled 3000
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RASTER = Path("/Users/mohamedhiba/Projects/processed/rasters_cleaned_10m")
LABELED_FULL = PROJECT_ROOT / "data/processed/patches/labeled_patch_index_ps64_kentucky_kgs6c_1to2.csv"
PKG = PROJECT_ROOT / "ky_package"
PS = 64
SEED = 42


def _preflight_bytexact(cache_f32, index_df, raster_dir, n_check=8):
    """Re-read n_check patches straight from rasters; assert byte-exact vs cache_f32."""
    import rasterio
    from src.patch_dataset import list_raster_files, read_boundless_patch_from_sources
    files = list_raster_files(raster_dir)
    srcs = [rasterio.open(f) for f in files]
    try:
        rng = np.random.default_rng(0)
        rows = rng.choice(len(index_df), size=min(n_check, len(index_df)), replace=False)
        maxdiff = 0.0
        for i in rows:
            r = index_df.iloc[int(i)]
            window = {k: int(r[k]) for k in
                      ("window_row_start", "window_row_stop", "window_col_start", "window_col_stop")}
            patch = read_boundless_patch_from_sources(srcs, window, -9999.0)
            maxdiff = max(maxdiff, float(np.abs(patch.astype("float32") - cache_f32[int(i)]).max()))
    finally:
        for s in srcs:
            s.close()
    return maxdiff


def build_labeled(n_labeled):
    import pandas as pd
    from src.patch_dataset import RasterPatchDataset
    df = pd.read_csv(LABELED_FULL)
    df = df.loc[df["valid_patch"].astype(bool)].reset_index(drop=True)
    if n_labeled and len(df) > n_labeled:
        # balanced stratified subsample by (cluster_id, label), preserve proportions
        frac = n_labeled / len(df)
        parts = [g.sample(max(2, int(round(len(g) * frac))), random_state=SEED)
                 for _, g in df.groupby(["cluster_id", "label"])]
        df = pd.concat(parts).sort_values("sample_id").reset_index(drop=True)
    sub_csv = PKG / "indices" / "labeled_ky_ps64_sub.csv"
    df.to_csv(sub_csv, index=False)
    ds = RasterPatchDataset(sub_csv, RASTER, PS, nodata_value=-9999, normalize=False,
                            valid_only=True, cache_in_memory=True, with_mask=True)
    ds._build_cache()
    cache = np.asarray(ds._cache, dtype="float32")
    ds.close()
    return cache, df, sub_csv


def build_unlabeled(n_unlabeled):
    from src.ssl_cross_channel import CrossChannelMaskRasterDataset, create_unlabeled_patch_index
    ul_csv = PKG / "indices" / f"ky_unlabeled_ps{PS}_n{n_unlabeled}.csv"
    create_unlabeled_patch_index(raster_dir=RASTER, output_csv=ul_csv, patch_size=PS,
                                 n_patches=n_unlabeled, nodata_value=-9999, random_seed=SEED,
                                 center_only=False)
    ds = CrossChannelMaskRasterDataset(ul_csv, RASTER, PS, normalize=False, cache_in_memory=True)
    ds._build_cache()
    cache = np.asarray(ds._cache, dtype="float32")
    ds.close()
    import pandas as pd
    return cache, pd.read_csv(ul_csv), ul_csv


def main() -> None:
    # The module-level builders read these globals; let CLI args override them.
    global RASTER, LABELED_FULL, PKG
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-labeled", type=int, default=3200)
    ap.add_argument("--n-unlabeled", type=int, default=3000)
    ap.add_argument("--raster-dir", type=Path, default=RASTER,
                    help="dir of the 13 KY 10m rasters (override to build on the pod).")
    ap.add_argument("--labeled-index", type=Path, default=LABELED_FULL,
                    help="full ps64 labeled index to subsample from.")
    ap.add_argument("--pkg-dir", type=Path, default=PKG, help="output package dir.")
    args = ap.parse_args()
    RASTER = args.raster_dir
    LABELED_FULL = args.labeled_index
    PKG = args.pkg_dir

    for sub in ("cache", "indices", "colab"):
        (PKG / sub).mkdir(parents=True, exist_ok=True)
    manifest = {"patch_size": PS, "n_terrain_channels": 13, "dtype": "float16",
                "study_area": "KGS six-county eastern KY (EPSG:3088)", "seed": SEED}

    for kind, builder, n in [("unlabeled", build_unlabeled, args.n_unlabeled),
                             ("labeled", build_labeled, args.n_labeled)]:
        print(f"\n=== building {kind} cache (target n={n}) ===", flush=True)
        cache_f32, idx_df, csv = builder(n)
        assert not np.isnan(cache_f32).any(), f"{kind} cache has NaN!"
        maxdiff = _preflight_bytexact(cache_f32, idx_df, RASTER)
        print(f"  shape={cache_f32.shape}  NaN=0  preflight byte-exact max|diff|={maxdiff:.1f} "
              f"(want 0.0)", flush=True)
        assert maxdiff == 0.0, f"{kind} cache NOT byte-exact vs rasters (maxdiff={maxdiff})"
        cache_f16 = cache_f32.astype("float16")
        f16_err = float(np.abs(cache_f16.astype("float32") - cache_f32).max())
        out = PKG / "cache" / f"cache_ky_{kind}.npz"
        np.savez_compressed(out, patches=cache_f16)
        mb = out.stat().st_size / 1e6
        print(f"  saved {out.name}: {mb:.1f} MB (float16, compressed); "
              f"float16 rounding max|diff|={f16_err:.4g}", flush=True)
        manifest[kind] = {"index_csv": csv.name, "cache_npz": out.name,
                          "n": int(cache_f32.shape[0]), "size_mb": round(mb, 1)}

    shutil.copy2(PROJECT_ROOT / "colab_package/colab/colab_patch.py", PKG / "colab" / "colab_patch.py")
    src_dst = PKG / "src"
    if src_dst.exists():
        shutil.rmtree(src_dst)
    shutil.copytree(PROJECT_ROOT / "src", src_dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (PKG / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(f.stat().st_size for f in (PKG / "cache").glob("*.npz")) / 1e6
    idx_mb = sum(f.stat().st_size for f in (PKG / "indices").glob("*.csv")) / 1e6
    print(f"\n=== PACKAGE READY: ky_package/ ===", flush=True)
    print(f"  caches total {total:.1f} MB + indices {idx_mb:.1f} MB", flush=True)
    print(f"  ~upload @150KB/s: {(total+idx_mb)*1e6/150e3/60:.0f} min", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
