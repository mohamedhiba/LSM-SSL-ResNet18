#!/usr/bin/env python
"""Reproducibility / regression check: re-finetune the canonical 0.810
cross-channel encoder with the NEW (Task-2) code path, landcover INCLUDED, no
mask channel, stable LR, across MULTIPLE SEEDS.

If the multi-seed mean is ~0.81, the new code is fine and the single 0.58 draw
was unlucky (the metric is just high-variance). If the mean sits well below 0.81,
either the new pipeline regressed or 0.810 never reliably reproduced.

Reuses the existing canonical 14-channel encoder (no re-pretraining). Landcover
is included via a runtime-only override of src.patch_dataset module globals,
restored at exit; no pipeline file is modified. Per-fold checkpoints share one
dir (overwritten each seed) to bound disk use.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = [42, 123, 7]


def main() -> None:
    import numpy as np
    import pandas as pd
    import src.patch_dataset as pdmod
    from src.train_finetune import run_pretrained_resnet18_scv_experiment
    from src.utils import ensure_dir, get_device, set_global_seed

    raster_dir = PROJECT_ROOT / "data/processed/rasters_cleaned"
    labeled = PROJECT_ROOT / "data/processed/patches/labeled_patch_index_ps32_common_balanced.csv"
    encoder = (PROJECT_ROOT / "checkpoints/ssl_pretrained/cross_channel_full"
               / "resnet18_cross_channel_ps32_encoder_best.pt")
    out_dir = ensure_dir(PROJECT_ROOT / "outputs/R6_newcode_landcover_check")
    if not encoder.exists():
        raise FileNotFoundError(f"canonical encoder not found: {encoder}")

    device = get_device()
    print(f"device: {device}; encoder: {encoder.name}; seeds: {SEEDS}")

    de, dx = pdmod.EXCLUDED_RASTER_NAME_STEMS, pdmod.EXPECTED_RASTER_COUNT
    pdmod.EXCLUDED_RASTER_NAME_STEMS = ()   # include landcover (14 channels)
    pdmod.EXPECTED_RASTER_COUNT = 14
    rows = []
    try:
        for seed in SEEDS:
            set_global_seed(seed)
            res = run_pretrained_resnet18_scv_experiment(
                project_root=PROJECT_ROOT,
                model_name="cc_newcode_landcover",
                pretraining="cross_channel",
                patch_index_csv=labeled,
                raster_dir=raster_dir,
                encoder_checkpoint_path=encoder,
                output_root=out_dir / f"seed{seed}",
                figure_root=PROJECT_ROOT / "figures/R6_newcode_landcover_check" / f"seed{seed}",
                checkpoint_dir=PROJECT_ROOT / "checkpoints/finetuned/R6_newcode_landcover_check",
                patch_size=32, random_seed=seed, batch_size=16,
                encoder_learning_rate=5e-6, head_learning_rate=2e-5,
                max_epochs=60, early_stopping_patience=12, dropout=0.4,
                expected_checkpoint_keys={"encoder_state_dict"},
                num_workers=0, device=device, plot_figures=False,
                cache_in_memory=True, with_mask=False,
            )
            fm = res["fold_metrics"]
            for _, r in fm.iterrows():
                rows.append({"seed": seed, "fold": int(r["fold"]), "AUC": float(r["auc"])})
            sm = res["summary_metrics"].iloc[0]
            print(f"  seed {seed}: mean AUC={sm['mean_auc']:.4f} SD={sm['std_auc']:.4f} "
                  f"folds={[round(a,3) for a in fm['auc'].tolist()]}", flush=True)
    finally:
        pdmod.EXCLUDED_RASTER_NAME_STEMS = de
        pdmod.EXPECTED_RASTER_COUNT = dx

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "r6_multiseed.csv", index=False)
    seed_means = df.groupby("seed")["AUC"].mean()
    overall = float(df["AUC"].mean())
    print("\n===== NEW-CODE + LANDCOVER, canonical encoder, 3 seeds =====")
    print(f"per-seed mean AUC : {[round(v,4) for v in seed_means.tolist()]}")
    print(f"overall mean AUC  : {overall:.4f}  (pooled SD over 15 folds {df['AUC'].std(ddof=1):.4f})")
    print(f"seed-to-seed SD of mean: {seed_means.std(ddof=1):.4f}")
    print("canonical old-code reference: 0.810")
    print("=> if overall ~0.81: new code OK, 0.58 was an unlucky single draw (high variance)")
    print("=> if overall ~0.58: new-code regression OR 0.810 never reliably reproduced")


if __name__ == "__main__":
    main()
