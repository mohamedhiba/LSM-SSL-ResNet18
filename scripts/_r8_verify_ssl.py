#!/usr/bin/env python
"""Verify R8 SSL 30m against R7. Re-runs cross-channel SSL finetune at 30m on the
FULL 344-sample index (the exact R7 no-landcover config) AND the with-landcover
config, 3 seeds, to confirm the pipeline reproduces R7 (no-landcover ~0.637,
with-landcover ~0.749) and that the low R8 number is the no-landcover regime,
not a bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = [42, 123, 7]


def main() -> None:
    import pandas as pd
    import src.patch_dataset as pdmod
    from src.train_finetune import run_pretrained_resnet18_scv_experiment
    from src.utils import ensure_dir, get_device, set_global_seed

    raster = PROJECT_ROOT / "data/processed/rasters_cleaned"
    labeled_full = PROJECT_ROOT / "data/processed/patches/labeled_patch_index_ps32_common_balanced.csv"
    CKPT = PROJECT_ROOT / "checkpoints/ssl_pretrained"
    out = ensure_dir(PROJECT_ROOT / "outputs/R8_verify_ssl")
    device = get_device()

    arms = [
        {"name": "no_landcover_344", "exclude": ("landcover",), "expected": 13, "with_mask": True,
         "encoder": CKPT / "cross_channel_newpipe/resnet18_cross_channel_ps32_encoder_best.pt"},
        {"name": "with_landcover_344", "exclude": (), "expected": 14, "with_mask": False,
         "encoder": CKPT / "cross_channel_full/resnet18_cross_channel_ps32_encoder_best.pt"},
    ]
    de, dx = pdmod.EXCLUDED_RASTER_NAME_STEMS, pdmod.EXPECTED_RASTER_COUNT
    rows = []
    try:
        for arm in arms:
            pdmod.EXCLUDED_RASTER_NAME_STEMS = arm["exclude"]
            pdmod.EXPECTED_RASTER_COUNT = arm["expected"]
            for seed in SEEDS:
                set_global_seed(seed)
                res = run_pretrained_resnet18_scv_experiment(
                    project_root=PROJECT_ROOT, model_name=f"verify_{arm['name']}",
                    pretraining="cross_channel", patch_index_csv=labeled_full, raster_dir=raster,
                    encoder_checkpoint_path=arm["encoder"], output_root=out / arm["name"] / f"seed{seed}",
                    figure_root=PROJECT_ROOT / f"figures/R8_verify_ssl/{arm['name']}/seed{seed}",
                    checkpoint_dir=PROJECT_ROOT / f"checkpoints/finetuned/R8_verify_{arm['name']}",
                    patch_size=32, random_seed=seed, batch_size=16,
                    encoder_learning_rate=5e-6, head_learning_rate=2e-5,
                    max_epochs=60, early_stopping_patience=12, dropout=0.4,
                    expected_checkpoint_keys={"encoder_state_dict"}, num_workers=0,
                    device=device, plot_figures=False, cache_in_memory=True, with_mask=arm["with_mask"],
                )
                fm = res["fold_metrics"]
                for _, r in fm.iterrows():
                    rows.append({"arm": arm["name"], "seed": seed, "fold": int(r["fold"]), "AUC": float(r["auc"])})
                print(f"  [{arm['name']}] seed {seed}: mean AUC={fm['auc'].mean():.4f}", flush=True)
    finally:
        pdmod.EXCLUDED_RASTER_NAME_STEMS = de
        pdmod.EXPECTED_RASTER_COUNT = dx

    df = pd.DataFrame(rows); df.to_csv(out / "verify.csv", index=False)
    print("\n===== VERIFY vs R7 =====", flush=True)
    for arm in ("no_landcover_344", "with_landcover_344"):
        a = df[df.arm == arm]
        print(f"{arm}: mean {a['AUC'].mean():.3f}  (R7 ref: "
              f"{'0.637' if 'no' in arm else '0.749'})", flush=True)


if __name__ == "__main__":
    main()
