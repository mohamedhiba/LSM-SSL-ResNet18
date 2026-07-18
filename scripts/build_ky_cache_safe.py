#!/usr/bin/env python
"""Build the KY labeled+unlabeled float16 caches with the boundless=False fix.

GDAL's boundless=True windowed read leaks ~11MB/patch and ignores GDAL_CACHEMAX, which
OOM-killed the on-pod cache build. All KY patch windows are in-bounds, so boundless=False
is byte-identical. This monkeypatches src.patch_dataset.read_boundless_patch_from_sources
(and its ssl_cross_channel import) to read boundless=False, then reuses the tested
export_ky_cache builders (same 3200-subsample + 3000-unlabeled, SEED=42, byte-exact preflight).

  python scripts/build_ky_cache_safe.py --raster-dir /workspace/ky/processed/rasters_cleaned_10m \
      --labeled-index .../labeled_patch_index_ps64_kentucky_kgs6c_1to2.csv \
      --pkg-dir /workspace/ky/pkg --n-labeled 3200 --n-unlabeled 3000
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


def _install_boundless_false_patch():
    """Replace read_boundless_patch_from_sources with a boundless=False version."""
    import src.patch_dataset as pdm

    def _read_inbounds(sources, window, nodata_value=pdm.DEFAULT_NODATA_VALUE):
        rw = pdm._window_to_rasterio(window)
        return np.stack([s.read(1, window=rw, boundless=False) for s in sources],
                        axis=0).astype("float32", copy=False)

    pdm.read_boundless_patch_from_sources = _read_inbounds
    # ssl_cross_channel imported the name at module load; rebind there too.
    import src.ssl_cross_channel as ccm
    if hasattr(ccm, "read_boundless_patch_from_sources"):
        ccm.read_boundless_patch_from_sources = _read_inbounds
    print("[safe-build] boundless=False patch installed", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster-dir", type=Path, required=True)
    ap.add_argument("--labeled-index", type=Path, required=True)
    ap.add_argument("--pkg-dir", type=Path, required=True)
    ap.add_argument("--n-labeled", type=int, default=3200)
    ap.add_argument("--n-unlabeled", type=int, default=3000)
    args = ap.parse_args()

    _install_boundless_false_patch()
    import scripts.export_ky_cache as X
    X.RASTER = args.raster_dir
    X.LABELED_FULL = args.labeled_index
    X.PKG = args.pkg_dir
    for sub in ("cache", "indices", "colab"):
        (args.pkg_dir / sub).mkdir(parents=True, exist_ok=True)

    manifest = {"patch_size": X.PS, "n_terrain_channels": 13, "dtype": "float16",
                "study_area": "KGS six-county eastern KY (EPSG:3088)", "seed": X.SEED,
                "build": "boundless=False (memory-safe)"}
    for kind, builder, n in [("unlabeled", X.build_unlabeled, args.n_unlabeled),
                             ("labeled", X.build_labeled, args.n_labeled)]:
        print(f"\n=== building {kind} cache (n={n}) ===", flush=True)
        cache_f32, idx_df, csv = builder(n)
        assert not np.isnan(cache_f32).any(), f"{kind} NaN!"
        md = X._preflight_bytexact(cache_f32, idx_df, args.raster_dir)
        print(f"  shape={cache_f32.shape} NaN=0 preflight max|diff|={md} (want 0.0)", flush=True)
        assert md == 0.0, f"{kind} not byte-exact (maxdiff={md})"
        out = args.pkg_dir / "cache" / f"cache_ky_{kind}.npz"
        np.savez_compressed(out, patches=cache_f32.astype("float16"))
        print(f"  saved {out.name}: {out.stat().st_size/1e6:.1f} MB", flush=True)
        manifest[kind] = {"index_csv": csv.name, "cache_npz": out.name, "n": int(cache_f32.shape[0])}

    shutil.copy2(PROJECT_ROOT / "colab_package/colab/colab_patch.py",
                 args.pkg_dir / "colab" / "colab_patch.py")
    (args.pkg_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n=== CACHE PACKAGE READY ===", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    print("CACHE_BUILD_OK", flush=True)


if __name__ == "__main__":
    main()
