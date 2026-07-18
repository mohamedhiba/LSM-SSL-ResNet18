#!/usr/bin/env python
"""R7: proper multi-seed landcover comparison (finetuning only, new code).

Two arms, both cross-channel-pretrained, both no mask channel, same stable LR
(head 2e-5 / encoder 5e-6, 60 ep), each finetuned across 3 seeds (42, 123, 7):

  (a) with_landcover : 13 terrain + landcover  (14ch)  -- reuse canonical encoder
  (b) no_landcover   : 13 terrain               (13ch)  -- reuse the lc_nomask encoder

Landcover is included for arm (a) via a runtime-only override of
src.patch_dataset module globals (restored at exit). No pipeline file is
modified. Per-fold checkpoints share one dir (overwritten) to bound disk.

Output:
    outputs/R7_landcover_multiseed/r7_landcover_multiseed.csv
        columns: arm, seed, fold, AUC, PR-AUC
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = [42, 123, 7]
CKPT = PROJECT_ROOT / "checkpoints/ssl_pretrained"
# Both arms use comparably-trained cross-channel encoders and 14-channel inputs.
# Arm A's 14th channel is landcover (informative); arm B's 14th channel is the
# valid-context mask, which is a verified no-op on these all-valid patches
# (constant 1.0, absorbed by BatchNorm) -> a fair stand-in for "13 terrain only".
ARMS = [
    {"name": "with_landcover", "exclude": (), "expected": 14, "with_mask": False,
     "encoder": CKPT / "cross_channel_full" / "resnet18_cross_channel_ps32_encoder_best.pt"},
    {"name": "no_landcover", "exclude": ("landcover",), "expected": 13, "with_mask": True,
     "encoder": CKPT / "cross_channel_newpipe" / "resnet18_cross_channel_ps32_encoder_best.pt"},
]


def main() -> None:
    import numpy as np
    import pandas as pd
    import src.patch_dataset as pdmod
    from src.train_finetune import run_pretrained_resnet18_scv_experiment
    from src.utils import ensure_dir, get_device, set_global_seed

    raster_dir = PROJECT_ROOT / "data/processed/rasters_cleaned"
    labeled = PROJECT_ROOT / "data/processed/patches/labeled_patch_index_ps32_common_balanced.csv"
    out_dir = ensure_dir(PROJECT_ROOT / "outputs/R7_landcover_multiseed")
    summary_csv = out_dir / "r7_landcover_multiseed.csv"

    device = get_device()
    print(f"device: {device}; seeds: {SEEDS}")
    for arm in ARMS:
        if not arm["encoder"].exists():
            raise FileNotFoundError(f"arm {arm['name']} encoder missing: {arm['encoder']}")

    de, dx = pdmod.EXCLUDED_RASTER_NAME_STEMS, pdmod.EXPECTED_RASTER_COUNT
    rows = []
    try:
        for arm in ARMS:
            pdmod.EXCLUDED_RASTER_NAME_STEMS = arm["exclude"]
            pdmod.EXPECTED_RASTER_COUNT = arm["expected"]
            n_terrain = len(pdmod.list_raster_files(raster_dir))
            in_channels = n_terrain + (1 if arm["with_mask"] else 0)
            print(f"\n===== arm {arm['name']} (terrain={n_terrain}, in_channels={in_channels}, "
                  f"with_mask={arm['with_mask']}) encoder={arm['encoder'].parent.name} =====")
            for seed in SEEDS:
                set_global_seed(seed)
                res = run_pretrained_resnet18_scv_experiment(
                    project_root=PROJECT_ROOT,
                    model_name=f"cc_{arm['name']}",
                    pretraining="cross_channel",
                    patch_index_csv=labeled,
                    raster_dir=raster_dir,
                    encoder_checkpoint_path=arm["encoder"],
                    output_root=out_dir / arm["name"] / f"seed{seed}",
                    figure_root=PROJECT_ROOT / "figures/R7_landcover_multiseed" / arm["name"] / f"seed{seed}",
                    checkpoint_dir=PROJECT_ROOT / "checkpoints/finetuned/R7_landcover_multiseed",
                    patch_size=32, random_seed=seed, batch_size=16,
                    encoder_learning_rate=5e-6, head_learning_rate=2e-5,
                    max_epochs=60, early_stopping_patience=12, dropout=0.4,
                    expected_checkpoint_keys={"encoder_state_dict"},
                    num_workers=0, device=device, plot_figures=False,
                    cache_in_memory=True, with_mask=arm["with_mask"],
                )
                fm = res["fold_metrics"]
                for _, r in fm.iterrows():
                    rows.append({"arm": arm["name"], "seed": seed, "fold": int(r["fold"]),
                                 "AUC": float(r["auc"]), "PR-AUC": float(r["pr_auc"])})
                pd.DataFrame(rows).to_csv(summary_csv, index=False)
                print(f"  {arm['name']} seed {seed}: mean AUC={res['summary_metrics'].iloc[0]['mean_auc']:.4f} "
                      f"folds={[round(a,3) for a in fm['auc'].tolist()]}", flush=True)
    finally:
        pdmod.EXCLUDED_RASTER_NAME_STEMS = de
        pdmod.EXPECTED_RASTER_COUNT = dx

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print(f"results: {summary_csv}")
    for name in ["with_landcover", "no_landcover"]:
        d = df.loc[df["arm"] == name, "AUC"]
        seed_means = df.loc[df["arm"] == name].groupby("seed")["AUC"].mean()
        print(f"\nArm {name}: overall mean AUC = {d.mean():.4f} +/- {d.std(ddof=1):.4f}  (n={len(d)})")
        print(f"   per-seed means: {[round(v,4) for v in seed_means.tolist()]}  (seed-SD {seed_means.std(ddof=1):.4f})")
    a = df.loc[df["arm"] == "with_landcover", "AUC"]
    b = df.loc[df["arm"] == "no_landcover", "AUC"]
    a_seed = df.loc[df["arm"] == "with_landcover"].groupby("seed")["AUC"].mean()
    b_seed = df.loc[df["arm"] == "no_landcover"].groupby("seed")["AUC"].mean()
    gap = float(a.mean() - b.mean())
    seed_sd = float(max(a_seed.std(ddof=1), b_seed.std(ddof=1)))
    print(f"\nwith-landcover minus no-landcover gap = {gap:+.4f}")
    print(f"seed-to-seed SD (max of arms) = {seed_sd:.4f}")
    print(f"VERDICT: gap {'EXCEEDS' if abs(gap) > 2 * seed_sd else 'is WITHIN'} 2x seed noise "
          f"=> landcover {'appears to matter' if abs(gap) > 2 * seed_sd else 'effect not distinguishable from noise'}")


if __name__ == "__main__":
    main()
