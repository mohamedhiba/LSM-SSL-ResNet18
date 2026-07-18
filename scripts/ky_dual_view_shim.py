"""Kentucky adapter for Qianyi's final 10 m dual-view pipeline.

Import this module FIRST in any KY dual-view process. It:
  1. puts `LSM_SSL_ResNet18_10m_dual_view_code/` at the front of sys.path, so
     `import src.*` resolves to QIANYI'S modules (not this repo's src/ — never
     mix the two in one process);
  2. rewrites `FINAL_10M_FACTOR_NAMES` IN PLACE to the Kentucky stack
     (ksat replaces lithology; ndvi filename differs). In-place mutation is
     load-bearing: every `from src.patch_dataset import FINAL_10M_FACTOR_NAMES`
     shares the same list object, and `list_raster_files_10m` builds the
     expected filenames from it — so this one patch realigns filenames,
     channel order, and QA asserts everywhere;
  3. provides KY path/config factories mirroring his data contract.

KY deltas vs the NYC contract this shim encodes:
  - factor 6:  lithology -> ksat  (continuous; his lithology-only categorical
    handling deactivates itself because no filename matches the keywords)
  - factor 7:  ndvi_10yr_mean_2015_2024 -> ndvi_annual_mean_2015_2024
  - rasters:   EPSG:3088 KY stack (default /Users/mohamedhiba/Projects/processed/
    rasters_cleaned_10m, override with KY_RASTER_DIR env var)
  - labeled manifest: our kgs6c index (x/y already in raster CRS; setting
    legacy_raster_dir to the KY raster dir makes the loader's CRS check a no-op)

Aligned protocol values (verified from his code, NOT the Slack paraphrase):
  dual view 15/31; SSL corpus n=20000 (50k is only the PU candidate pool);
  masked-recon: ratio 0.5, blocks 3x3 local / 4x4 global, reconstruct 13 factor
  channels only, LR 1e-4; other SSL tasks LR 1e-3; AdamW wd 1e-4, batch 64,
  50 ep / patience 10; downstream: head LR 1e-3, encoder-LR grid
  [0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4], batch 128, 100 ep / patience 15,
  pos_weight, StratifiedKFold(5, shuffle, random_state=42) on label,
  seeds 42-46; LR=0 freezes weights only unless strict_frozen_encoder=True.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DUAL_VIEW_ROOT = REPO_ROOT / "LSM_SSL_ResNet18_10m_dual_view_code"

KY_RASTER_DIR = Path(os.environ.get(
    "KY_RASTER_DIR", "/Users/mohamedhiba/Projects/processed/rasters_cleaned_10m"))
KY_LABELED_MANIFEST = REPO_ROOT / "data" / "processed" / "patches" / \
    "labeled_patch_index_ps64_kentucky_kgs6c_1to2.csv"
KY_OUT_DIR = REPO_ROOT / "data" / "kentucky_dual_view"

KY_10M_FACTOR_NAMES = [
    "aspect",
    "bulk_density",
    "clay_pct",
    "elevation",
    "field_capacity",
    "ksat",
    "ndvi_annual_mean_2015_2024",
    "plan_curv",
    "profile_curv",
    "sand_pct",
    "slope",
    "spi_dinf",
    "twi_dinf",
]


def _bootstrap() -> None:
    if "src.patch_dataset" in sys.modules:
        mod = sys.modules["src.patch_dataset"]
        if not hasattr(mod, "FINAL_10M_FACTOR_NAMES"):
            raise RuntimeError(
                "This repo's own src/ package is already imported; the KY dual-view "
                "shim must run in a fresh process that never imports the local src/.")
    if str(DUAL_VIEW_ROOT) not in sys.path:
        sys.path.insert(0, str(DUAL_VIEW_ROOT))
    import src.patch_dataset as qpd  # Qianyi's module

    qpd.FINAL_10M_FACTOR_NAMES[:] = KY_10M_FACTOR_NAMES


_bootstrap()


def ky_ten_m_patch_config(target_unlabeled_n: int = 20000):
    """Stage-10 config: KY rasters + kgs6c manifest -> dual-view indices + stats."""
    from src.prepare_10m_patch_indices import TenMPatchConfig

    return TenMPatchConfig(
        project_root=REPO_ROOT,
        raster_dir=KY_RASTER_DIR,
        source_labeled_index_csv=KY_LABELED_MANIFEST,
        # same dir as the reference stack -> source CRS == reference CRS -> no reprojection
        legacy_raster_dir=KY_RASTER_DIR,
        labeled_output_csv=KY_OUT_DIR / "dual_view_padded_patch_index_ky10m.csv",
        removed_labeled_output_csv=KY_OUT_DIR / "dual_view_padded_patch_index_ky10m_removed_samples.csv",
        unlabeled_output_csv=KY_OUT_DIR / f"unlabeled_dual_view_padded_index_ky10m_n{target_unlabeled_n}.csv",
        normalization_json=KY_OUT_DIR / "normalization_stats_ky10m_13factors.json",
        normalization_csv=KY_OUT_DIR / "normalization_stats_ky10m_13factors.csv",
        output_dir=KY_OUT_DIR,
        target_unlabeled_n=target_unlabeled_n,
    ).resolved()
