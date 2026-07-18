#!/usr/bin/env python
"""R5: scratch (no-SSL) landcover ablation, finetuning-only robustness check.

Trains a randomly-initialized ResNet-18 binary classifier under the 5-cluster
spatial-CV protocol with the stable finetune LR (head 2e-5 / encoder 5e-6, 60
epochs), with NO SSL pretraining, comparing:

  A) 13 terrain channels + landcover  (14 total, old channel scheme)
  B) 13 terrain channels, no landcover (current scheme, no mask channel)

Each setup is run with 3 random seeds (42, 123, 7) x 5 folds = 15 runs, to test
whether the ~0.81 vs ~0.53 cross-channel gap reflects real downstream signal
from landcover or normal run-to-run noise.

This is self-contained: it does NOT modify any pipeline file and does NOT touch
SSL/pretraining code. Setup A includes Landcover.tif via a runtime-only override
of ``src.patch_dataset`` module globals (restored at exit); no file edit and no
permanent change to EXCLUDED_RASTER_NAME_STEMS. Datasets are built with
with_mask=False so A is exactly 14 channels (13 terrain + landcover) and B is
exactly 13 terrain channels (no mask channel).

Output:
    outputs/R5_landcover_scratch_ablation/r5_landcover_scratch_ablation.csv
        columns: setup, seed, fold, AUC, PR-AUC, best_threshold

Usage:
    python scripts/scratch_landcover_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = [42, 123, 7]
N_FOLDS = 5
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 12
BATCH_SIZE = 16
VAL_FRACTION = 0.2
ENCODER_LR = 5e-6
HEAD_LR = 2e-5
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
GRAD_CLIP_NORM = 5.0

# Setup A includes landcover (no name exclusion, expect 14 rasters); B excludes
# landcover (the package default, 13 rasters). The exclusion/count are applied
# as a runtime override of src.patch_dataset module globals for this run only.
SETUPS = [
    {"name": "A", "in_channels": 14, "exclude": (), "expected": 14,
     "desc": "13 terrain + landcover (old 14-channel scheme)"},
    {"name": "B", "in_channels": 13, "exclude": ("landcover",), "expected": 13,
     "desc": "13 terrain, no landcover (current scheme, no mask)"},
]


def main() -> None:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import DataLoader, Subset

    import src.patch_dataset as pdmod
    from src.patch_dataset import RasterPatchDataset
    from src.metrics import compute_binary_metrics, find_best_f1_threshold
    from src.models_resnet18 import create_resnet18_binary_classifier
    from src.train_finetune import evaluate_model, train_scv_fold
    from src.utils import ensure_dir, get_device, set_global_seed

    raster_dir = PROJECT_ROOT / "data/processed/rasters_cleaned"
    patch_index_csv = PROJECT_ROOT / "data/processed/patches/labeled_patch_index_ps32_common_balanced.csv"
    out_dir = ensure_dir(PROJECT_ROOT / "outputs/R5_landcover_scratch_ablation")
    ckpt_dir = ensure_dir(out_dir / "_ckpt")
    summary_csv = out_dir / "r5_landcover_scratch_ablation.csv"
    patch_size = 32

    device = get_device()
    print(f"device: {device}")
    pin_memory = device.type == "cuda"

    patch_index = pd.read_csv(patch_index_csv).reset_index(drop=True)
    if sorted(patch_index["cluster_id"].astype(int).unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("cluster_id values must be 0..4.")

    # Remember the package defaults so we can restore them no matter what.
    default_exclude = pdmod.EXCLUDED_RASTER_NAME_STEMS
    default_expected = pdmod.EXPECTED_RASTER_COUNT

    def compute_channel_stats(indices, in_channels):
        stats_ds = RasterPatchDataset(
            patch_index_csv=patch_index_csv, raster_dir=raster_dir, patch_size=patch_size,
            nodata_value=-9999, normalize=False, return_metadata=False, valid_only=True,
            cache_in_memory=True, with_mask=False,
        )
        loader = DataLoader(Subset(stats_ds, list(indices)), batch_size=32, shuffle=False, num_workers=0)
        csum = csq = None
        npix = 0
        for X, _y in loader:
            X = X.float()
            csum = X.sum(dim=(0, 2, 3)) if csum is None else csum + X.sum(dim=(0, 2, 3))
            csq = (X ** 2).sum(dim=(0, 2, 3)) if csq is None else csq + (X ** 2).sum(dim=(0, 2, 3))
            npix += X.shape[0] * X.shape[2] * X.shape[3]
        stats_ds.close()
        means = (csum / npix).numpy().astype("float32")
        var = (csq / npix).numpy() - means ** 2
        stds = np.sqrt(np.maximum(var, 1e-12)).astype("float32")
        if len(means) != in_channels:
            raise ValueError(f"stats produced {len(means)} channels, expected {in_channels}.")
        return means, stds

    rows: list[dict] = []
    try:
        for setup in SETUPS:
            # ---- runtime-only override so list_raster_files sees the right channels ----
            pdmod.EXCLUDED_RASTER_NAME_STEMS = setup["exclude"]
            pdmod.EXPECTED_RASTER_COUNT = setup["expected"]
            raster_files = pdmod.list_raster_files(raster_dir)
            assert len(raster_files) == setup["in_channels"], (
                f"setup {setup['name']}: expected {setup['in_channels']} rasters, got {len(raster_files)}"
            )
            print(f"\n===== Setup {setup['name']}: {setup['desc']} (in_channels={setup['in_channels']}) =====")
            print("channels:", [p.name for p in raster_files])

            for seed in SEEDS:
                set_global_seed(seed)
                for fold in range(N_FOLDS):
                    test_mask = patch_index["cluster_id"].astype(int) == fold
                    train_cand = patch_index.index[~test_mask].to_numpy()
                    test_idx = patch_index.index[test_mask].to_numpy()
                    train_idx, val_idx = train_test_split(
                        train_cand, test_size=VAL_FRACTION, random_state=seed + fold,
                        stratify=patch_index.loc[train_cand, "label"].to_numpy(),
                    )
                    train_idx = np.asarray(train_idx, dtype=int)
                    val_idx = np.asarray(val_idx, dtype=int)
                    test_idx = np.asarray(test_idx, dtype=int)

                    means, stds = compute_channel_stats(train_idx, setup["in_channels"])
                    ds = RasterPatchDataset(
                        patch_index_csv=patch_index_csv, raster_dir=raster_dir, patch_size=patch_size,
                        nodata_value=-9999, normalize=True, channel_means=means, channel_stds=stds,
                        return_metadata=True, valid_only=True, cache_in_memory=True, with_mask=False,
                    )
                    train_loader = DataLoader(Subset(ds, train_idx.tolist()), batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=0, pin_memory=pin_memory)
                    val_loader = DataLoader(Subset(ds, val_idx.tolist()), batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=0, pin_memory=pin_memory)
                    test_loader = DataLoader(Subset(ds, test_idx.tolist()), batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=0, pin_memory=pin_memory)

                    # Random-init ResNet-18; NO pretrained encoder loaded.
                    model = create_resnet18_binary_classifier(
                        in_channels=setup["in_channels"], dropout=DROPOUT,
                        small_patch_stem=True, pretrained=False,
                    ).to(device)
                    encoder_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
                    head_params = [p for n, p in model.named_parameters() if n.startswith("fc.")]
                    optimizer = torch.optim.AdamW(
                        [{"params": encoder_params, "lr": ENCODER_LR},
                         {"params": head_params, "lr": HEAD_LR}],
                        weight_decay=WEIGHT_DECAY,
                    )
                    criterion = nn.BCEWithLogitsLoss()
                    ckpt = ckpt_dir / f"scratch_{setup['name']}_seed{seed}_fold{fold}.pt"

                    train_scv_fold(
                        model=model, train_loader=train_loader, val_loader=val_loader,
                        criterion=criterion, optimizer=optimizer, device=device,
                        checkpoint_path=ckpt, max_epochs=MAX_EPOCHS,
                        early_stopping_patience=EARLY_STOPPING_PATIENCE,
                        monitor_metric="val_auc", grad_clip_norm=GRAD_CLIP_NORM,
                    )
                    # model now holds the best (val-AUC) checkpoint weights.
                    val_res = evaluate_model(model, val_loader, criterion, device)
                    best_thr, _ = find_best_f1_threshold(val_res["y_true"], val_res["y_probs"])
                    test_res = evaluate_model(model, test_loader, criterion, device)
                    m = compute_binary_metrics(test_res["y_true"], test_res["y_probs"], threshold=0.5)

                    rows.append({
                        "setup": setup["name"], "seed": seed, "fold": fold,
                        "AUC": float(m["auc"]), "PR-AUC": float(m["pr_auc"]),
                        "best_threshold": float(best_thr),
                    })
                    pd.DataFrame(rows).to_csv(summary_csv, index=False)  # incremental
                    print(f"  setup {setup['name']} seed {seed} fold {fold}: "
                          f"AUC={m['auc']:.4f} PR-AUC={m['pr_auc']:.4f} thr={best_thr:.3f}", flush=True)
                    ds.close()
    finally:
        pdmod.EXCLUDED_RASTER_NAME_STEMS = default_exclude
        pdmod.EXPECTED_RASTER_COUNT = default_expected

    # ---------------- summary ----------------
    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print(f"results written: {summary_csv}")
    a = df.loc[df["setup"] == "A", "AUC"]
    b = df.loc[df["setup"] == "B", "AUC"]
    a_mean, a_sd = float(a.mean()), float(a.std(ddof=1))
    b_mean, b_sd = float(b.mean()), float(b.std(ddof=1))
    gap = a_mean - b_mean
    pooled_sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
    # seed-to-seed variance: SD of the 3 per-seed mean AUCs within each group
    a_seed_means = df.loc[df["setup"] == "A"].groupby("seed")["AUC"].mean()
    b_seed_means = df.loc[df["setup"] == "B"].groupby("seed")["AUC"].mean()
    a_seed_sd = float(a_seed_means.std(ddof=1))
    b_seed_sd = float(b_seed_means.std(ddof=1))

    print(f"\nSetup A (with landcover, 14ch): mean AUC = {a_mean:.4f}  SD = {a_sd:.4f}  (n={len(a)})")
    print(f"Setup B (no landcover,  13ch): mean AUC = {b_mean:.4f}  SD = {b_sd:.4f}  (n={len(b)})")
    print(f"A - B gap = {gap:+.4f}")
    print(f"per-seed mean AUC  A: {[round(v,4) for v in a_seed_means.tolist()]}  (seed-SD {a_seed_sd:.4f})")
    print(f"per-seed mean AUC  B: {[round(v,4) for v in b_seed_means.tolist()]}  (seed-SD {b_seed_sd:.4f})")
    print(f"\ngap ({gap:+.4f}) vs seed-to-seed SD (A {a_seed_sd:.4f}, B {b_seed_sd:.4f}, "
          f"within-group fold SD ~{(a_sd+b_sd)/2:.4f})")
    bigger = abs(gap) > max(a_seed_sd, b_seed_sd)
    print(f"A-vs-B gap larger than seed-to-seed variance? {bigger}")


if __name__ == "__main__":
    main()
