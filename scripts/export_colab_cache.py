#!/usr/bin/env python
"""Export a self-contained Colab package for the sequential-SSL resolution study.

Builds compact raw-patch caches LOCALLY (where the 30 m and 10 m rasters live) so
the ~30 GB 10 m stack never has to reach Colab. The cache is exactly what
``*_build_cache`` reads from the rasters, in the dataset's filtered row order, so
on Colab ``colab_patch`` can serve it transparently and reuse every bit of the
``src`` training/finetune/metrics code unchanged.

Outputs into ``colab_package/``:
  cache/cache_{30m,10m}_{unlabeled,labeled}.npz   raw patches (N, 13, 32, 32) float32
  indices/*.csv                                   the four shipped patch indices
  src/                                            copy of the repo src package
  colab/                                          colab_patch.py + sequential_resolution.py
  manifest.json                                   shapes, row counts, provenance

Run locally (has rasters):
    .venv/bin/python scripts/export_colab_cache.py
    .venv/bin/python scripts/export_colab_cache.py --smoke   # tiny cache, fast sanity
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

PKG = PROJECT_ROOT / "colab_package"
RASTER = {
    "30m": PROJECT_ROOT / "data/processed/rasters_cleaned",
    "10m": PROJECT_ROOT / "data/processed/rasters_cleaned_10m",
}
INDICES = {
    "30m": {
        "unlabeled": PROJECT_ROOT / "data/processed/ssl_unlabeled_indices/unlabeled_patch_index_ps32_n20000.csv",
        "labeled": PROJECT_ROOT / "data/processed/patches/labeled_patch_index_ps32_common_balanced_30m_r8.csv",
    },
    "10m": {
        "unlabeled": PROJECT_ROOT / "data/processed/ssl_unlabeled_indices/unlabeled_patch_index_ps32_n20000_10m.csv",
        "labeled": PROJECT_ROOT / "data/processed/patches/labeled_patch_index_ps32_common_balanced_10m.csv",
    },
}
PATCH_SIZE = 32


def _build_unlabeled_cache(csv: Path, raster_dir: Path, limit: int | None) -> np.ndarray:
    from src.ssl_cross_channel import CrossChannelMaskRasterDataset

    ds = CrossChannelMaskRasterDataset(csv, raster_dir, PATCH_SIZE, normalize=False,
                                       cache_in_memory=True)
    if limit is not None:
        ds.index = ds.index.iloc[:limit].reset_index(drop=True)
    ds._build_cache()
    cache = np.asarray(ds._cache, dtype="float32")
    ds.close()
    return cache


def _build_labeled_cache(csv: Path, raster_dir: Path) -> np.ndarray:
    from src.patch_dataset import RasterPatchDataset

    ds = RasterPatchDataset(csv, raster_dir, PATCH_SIZE, normalize=False,
                            valid_only=True, cache_in_memory=True, with_mask=True)
    ds._build_cache()
    cache = np.asarray(ds._cache, dtype="float32")
    ds.close()
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="cap unlabeled cache to 256 patches")
    ap.add_argument("--arms", nargs="+", default=["30m", "10m"], choices=["30m", "10m"])
    args = ap.parse_args()
    limit = 256 if args.smoke else None

    # Strip any run-artifacts from a prior (smoke) driver run: the Colab driver is
    # idempotent and would SKIP real pretraining if stale encoders were shipped.
    for junk in ("checkpoints", "outputs", "figures"):
        if (PKG / junk).exists():
            shutil.rmtree(PKG / junk)
            print(f"removed stale {junk}/ from package", flush=True)

    (PKG / "cache").mkdir(parents=True, exist_ok=True)
    (PKG / "indices").mkdir(parents=True, exist_ok=True)

    manifest: dict = {"patch_size": PATCH_SIZE, "n_terrain_channels": 13,
                      "mask_channel": True, "smoke": args.smoke, "arms": {}}

    for arm in args.arms:
        print(f"\n=== building caches for {arm} ===", flush=True)
        raster_dir = RASTER[arm]
        info: dict = {}
        for kind, builder in [("unlabeled", _build_unlabeled_cache), ("labeled", _build_labeled_cache)]:
            csv = INDICES[arm][kind]
            print(f"  {arm}/{kind}: reading patches from {csv.name} ...", flush=True)
            if kind == "unlabeled":
                cache = builder(csv, raster_dir, limit)
                # ship an index whose length matches the cache (smoke truncates both)
                if limit is not None:
                    import pandas as pd
                    pd.read_csv(csv).iloc[:cache.shape[0]].to_csv(PKG / "indices" / csv.name, index=False)
                else:
                    shutil.copy2(csv, PKG / "indices" / csv.name)
            else:
                cache = builder(csv, raster_dir)
                shutil.copy2(csv, PKG / "indices" / csv.name)
            out = PKG / "cache" / f"cache_{arm}_{kind}.npz"
            np.savez_compressed(out, patches=cache)
            mb = out.stat().st_size / 1e6
            print(f"    cache shape={cache.shape}  saved {out.name} ({mb:.1f} MB)", flush=True)
            info[kind] = {"index_csv": csv.name, "cache_npz": out.name,
                          "shape": list(cache.shape), "size_mb": round(mb, 1)}
        manifest["arms"][arm] = info

    # copy src package
    src_dst = PKG / "src"
    if src_dst.exists():
        shutil.rmtree(src_dst)
    shutil.copytree(PROJECT_ROOT / "src", src_dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"\ncopied src/ -> {src_dst}", flush=True)

    (PKG / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote manifest.json", flush=True)
    print("\nEXPORT COMPLETE. Upload colab_package/ to Google Drive and open the notebook.", flush=True)


if __name__ == "__main__":
    main()
