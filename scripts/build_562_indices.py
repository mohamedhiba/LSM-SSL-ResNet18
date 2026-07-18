#!/usr/bin/env python
"""Build 30m + 10m single-view patch indices for the 562-sample balanced set.

Qianyi's modeling dataset = data/processed/samples/final_cluster_balanced_dataset.csv
(281 landslide + 281 reliable-nonlandslide, 5 balanced spatial clusters). Our earlier
runs mistakenly used the 344-row common_balanced set; this rebuilds indices from the
correct 562 set so we can re-run the SSL-vs-scratch 2x2 on it.

Outputs (ps32, single-view 32x32 + valid-context mask, matching our pipeline):
  data/processed/patches/labeled_patch_index_ps32_balanced562_30m.csv
  data/processed/patches/labeled_patch_index_ps32_balanced562_10m.csv   (matched-valid only)
  data/processed/patches/labeled_patch_index_ps32_balanced562_30m_matched.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.patch_dataset import create_patch_index, list_raster_files, save_patch_index  # noqa: E402

PS = 32
SAMPLES = PROJECT_ROOT / "data/processed/samples/final_cluster_balanced_dataset.csv"
PATCH_DIR = PROJECT_ROOT / "data/processed/patches"
RASTER_30M = PROJECT_ROOT / "data/processed/rasters_cleaned"
RASTER_10M = PROJECT_ROOT / "data/processed/rasters_cleaned_10m"

OUT_30M = PATCH_DIR / "labeled_patch_index_ps32_balanced562_30m.csv"
OUT_10M = PATCH_DIR / "labeled_patch_index_ps32_balanced562_10m.csv"
OUT_30M_MATCHED = PATCH_DIR / "labeled_patch_index_ps32_balanced562_30m_matched.csv"


def _build(raster_dir: Path, out_csv: Path) -> pd.DataFrame:
    samples = pd.read_csv(SAMPLES)  # includes factor_01..factor_14 that create_patch_index requires
    raster_files = list_raster_files(raster_dir)
    print(f"  {raster_dir.name}: {len(raster_files)} rasters, building windows for {len(samples)} samples ...", flush=True)
    idx = create_patch_index(samples_df=samples, raster_files=raster_files, patch_size=PS)
    save_patch_index(idx, out_csv)
    nvalid = int(idx["valid_patch"].astype(bool).sum())
    print(f"    saved {out_csv.name}: {len(idx)} rows, {nvalid} valid", flush=True)
    return idx


def main() -> None:
    print("=== building 562-sample indices (30m + 10m) ===", flush=True)
    idx30 = _build(RASTER_30M, OUT_30M)
    idx10 = _build(RASTER_10M, OUT_10M)

    # matched intersection: samples valid at BOTH resolutions (clean comparison)
    v30 = set(idx30.loc[idx30["valid_patch"].astype(bool), "sample_id"].astype(str))
    v10 = set(idx10.loc[idx10["valid_patch"].astype(bool), "sample_id"].astype(str))
    matched = sorted(v30 & v10)
    dropped = (v30 | v10) - set(matched)
    print(f"\nvalid@30m={len(v30)}  valid@10m={len(v10)}  matched={len(matched)}  dropped={len(dropped)}", flush=True)

    m30 = idx30[idx30["sample_id"].astype(str).isin(matched)].sort_values("sample_id").reset_index(drop=True)
    m10 = idx10[idx10["sample_id"].astype(str).isin(matched)].sort_values("sample_id").reset_index(drop=True)
    m30.to_csv(OUT_30M_MATCHED, index=False)
    m10.to_csv(OUT_10M, index=False)  # overwrite 10m with matched-only for a clean paired run

    for name, m in [("30m", m30), ("10m", m10)]:
        bal = m.groupby(["cluster_id", "label"]).size().unstack(fill_value=0)
        print(f"\n{name} matched (n={len(m)}) per-cluster x label:\n{bal.to_string()}", flush=True)
    print("\nDONE. Matched 562-set indices ready.", flush=True)


if __name__ == "__main__":
    main()
