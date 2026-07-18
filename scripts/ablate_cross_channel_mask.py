#!/usr/bin/env python
"""Mask-channel ablation for cross-channel masking SSL, ps32.

Tests the hypothesis that the cross-channel pretext learns physically meaningful
inter-factor terrain structure: masking a hydro-topographic channel (TWI, slope,
curvature) — which is genuinely predictable from the other terrain derivatives —
should yield a more transferable encoder than masking a near-independent
categorical channel (lithology, land cover).

Efficiency: the masked channel is a training-loop parameter, not part of the
encoder, so the 13-channel patch cache is built ONCE and reused for every
ablation point. Each point: pretrain a cross-channel encoder masking that
channel, then fine-tune it under the stable downstream protocol (reduced LR),
and record the 5-fold spatial-CV AUC.

Outputs:
    outputs/R3_cross_channel_ablation/ablation_summary.csv   (incremental)
    outputs/R3_cross_channel_ablation/ch{idx}/...             (per-channel SCV)
    checkpoints/ssl_pretrained/cross_channel_ablation/ch{idx}/...

Run with a native arm64 PyTorch build so the MPS backend is available.

Usage:
    python scripts/ablate_cross_channel_mask.py
    python scripts/ablate_cross_channel_mask.py --masked-channels 12,10,8,6,5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Physical-category labels for interpretation (by raster index, landcover dropped).
CHANNEL_CATEGORY = {
    0: "topographic",        # aspect
    1: "soil",               # bulk_density
    2: "soil",               # clay_pct
    3: "topographic",        # elevation
    4: "soil",               # field_capacity
    5: "categorical",        # lithology
    6: "vegetation",         # ndvi
    7: "topographic",        # plan_curv
    8: "topographic",        # profile_curv
    9: "soil",               # sand_pct
    10: "topographic",       # slope
    11: "hydro-topographic", # spi_dinf
    12: "hydro-topographic", # twi_dinf
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask-channel ablation for cross-channel masking SSL.")
    parser.add_argument("--masked-channels", type=str, default="12,10,8,6,5",
                        help="Comma-separated raster indices to mask, one ablation point each "
                             "(13-channel indexing, landcover dropped): 12=twi, 10=slope, "
                             "8=profile_curv, 6=ndvi, 5=lithology.")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--n-unlabeled-patches", type=int, default=20000)
    parser.add_argument("--normalization-sample-size", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--pretrain-patience", type=int, default=8)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--finetune-patience", type=int, default=10)
    parser.add_argument("--head-learning-rate", type=float, default=2e-5)
    parser.add_argument("--encoder-learning-rate", type=float, default=5e-6)
    parser.add_argument("--finetune-batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, random_split

    from src.patch_dataset import DEFAULT_NODATA_VALUE, list_raster_files
    from src.ssl_cross_channel import (
        CrossChannelMaskRasterDataset,
        CrossChannelModel,
        compute_ssl_channel_stats,
        create_unlabeled_patch_index,
    )
    from src.train_finetune import run_pretrained_resnet18_scv_experiment
    from src.train_ssl import train_cross_channel_model
    from src.utils import ensure_dir, get_device, set_global_seed

    masked_channels = [int(x) for x in args.masked_channels.split(",") if x.strip() != ""]
    raster_dir = PROJECT_ROOT / "data/processed/rasters_cleaned"
    raster_names = [p.name for p in list_raster_files(raster_dir)]
    unlabeled_index_csv = (
        PROJECT_ROOT / "data/processed/ssl_unlabeled_indices"
        / f"unlabeled_patch_index_ps{args.patch_size}_n{args.n_unlabeled_patches}.csv"
    )
    labeled_index_csv = (
        PROJECT_ROOT / "data/processed/patches"
        / f"labeled_patch_index_ps{args.patch_size}_common_balanced.csv"
    )
    out_root = ensure_dir(PROJECT_ROOT / "outputs/R3_cross_channel_ablation")
    ckpt_root = ensure_dir(PROJECT_ROOT / "checkpoints/ssl_pretrained/cross_channel_ablation")
    summary_csv = out_root / "ablation_summary.csv"

    set_global_seed(args.random_seed)
    device = get_device()
    pin_memory = device.type == "cuda"
    print(f"device: {device}")
    print(f"ablating masked channels: {[(i, raster_names[i]) for i in masked_channels]}", flush=True)

    create_unlabeled_patch_index(
        raster_dir=raster_dir,
        output_csv=unlabeled_index_csv,
        patch_size=args.patch_size,
        n_patches=args.n_unlabeled_patches,
        nodata_value=DEFAULT_NODATA_VALUE,
        max_nodata_ratio=0.0,
        random_seed=args.random_seed,
    )

    # ---- Build the patch cache ONCE (shared raw cache across all ablation points) ----
    print("building shared in-memory patch cache (once)...", flush=True)
    raw_dataset = CrossChannelMaskRasterDataset(
        patch_index_csv=unlabeled_index_csv,
        raster_dir=raster_dir,
        patch_size=args.patch_size,
        normalize=False,
        cache_in_memory=True,
    )
    _ = raw_dataset[0]  # trigger cache build
    channel_means, channel_stds = compute_ssl_channel_stats(
        raw_dataset,
        sample_size=args.normalization_sample_size,
        batch_size=args.pretrain_batch_size,
        random_seed=args.random_seed,
    )
    ssl_dataset = CrossChannelMaskRasterDataset(
        patch_index_csv=unlabeled_index_csv,
        raster_dir=raster_dir,
        patch_size=args.patch_size,
        normalize=True,
        channel_means=channel_means,
        channel_stds=channel_stds,
        cache_in_memory=True,
    )
    ssl_dataset._cache = raw_dataset._cache  # reuse the raw cache; avoid a second build
    print(f"cache ready: {ssl_dataset._cache.shape}", flush=True)

    train_size = int(0.9 * len(ssl_dataset))
    val_size = len(ssl_dataset) - train_size
    train_dataset, val_dataset = random_split(
        ssl_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.random_seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.pretrain_batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.pretrain_batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin_memory)

    rows = []
    for idx in masked_channels:
        name = raster_names[idx]
        category = CHANNEL_CATEGORY.get(idx, "unknown")
        tag = f"ch{idx:02d}"
        print(f"\n===== ablation: mask channel {idx} ({name}) [{category}] =====", flush=True)

        # ---- pretrain cross-channel encoder masking this channel ----
        set_global_seed(args.random_seed)  # identical init/order across ablation points
        model = CrossChannelModel(in_channels=len(raster_names) + 1, out_channels=1).to(device)  # +1 valid-context mask
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
        ckpt_dir = ensure_dir(ckpt_root / tag)
        encoder_best = ckpt_dir / "resnet18_cross_channel_ps32_encoder_best.pt"
        config = {"task": "cross_channel", "masked_channel_index": idx, "masked_channel_raster": name,
                  "ablation": True, "pretrain_epochs": args.pretrain_epochs}
        _, pre_info = train_cross_channel_model(
            model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
            device=device,
            full_model_best_path=ckpt_dir / "resnet18_cross_channel_ps32_full_model_best.pt",
            encoder_best_path=encoder_best,
            last_checkpoint_path=ckpt_dir / "resnet18_cross_channel_ps32_last.pt",
            config=config, channel_means=channel_means, channel_stds=channel_stds,
            max_epochs=args.pretrain_epochs, early_stopping_patience=args.pretrain_patience,
            masked_channel_index=idx,
        )
        print(f"  pretrain best val_loss={pre_info['best_val_loss']:.5f} @ epoch {pre_info['best_epoch']}", flush=True)

        # ---- fine-tune the encoder under the stable (reduced-LR) downstream protocol ----
        result = run_pretrained_resnet18_scv_experiment(
            project_root=PROJECT_ROOT,
            model_name=f"cc_mask{idx:02d}_resnet18",
            pretraining=f"cross_channel_mask{idx:02d}",
            patch_index_csv=labeled_index_csv,
            raster_dir=raster_dir,
            encoder_checkpoint_path=encoder_best,
            output_root=out_root / tag,
            figure_root=PROJECT_ROOT / f"figures/R3_cross_channel_ablation/{tag}",
            checkpoint_dir=PROJECT_ROOT / f"checkpoints/finetuned/cc_ablation/{tag}",
            patch_size=args.patch_size,
            random_seed=args.random_seed,
            batch_size=args.finetune_batch_size,
            encoder_learning_rate=args.encoder_learning_rate,
            head_learning_rate=args.head_learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.finetune_epochs,
            early_stopping_patience=args.finetune_patience,
            dropout=args.dropout,
            expected_checkpoint_keys={"encoder_state_dict"},
            num_workers=0,
            device=device,
            plot_figures=False,
            cache_in_memory=True,
        )
        fm = result["fold_metrics"] if isinstance(result, dict) else None
        fold_auc = fm["auc"].tolist()
        row = {
            "masked_channel_index": idx,
            "masked_channel_raster": name,
            "category": category,
            "pretrain_best_val_loss": float(pre_info["best_val_loss"]),
            "pretrain_best_epoch": int(pre_info["best_epoch"]),
            "mean_auc": float(np.mean(fold_auc)),
            "std_auc": float(np.std(fold_auc, ddof=1)),
            "worst_fold_auc": float(np.min(fold_auc)),
            "mean_pr_auc": float(fm["pr_auc"].mean()),
            "mean_f1_best": float(fm["f1_best_f1"].mean()),
            "fold_auc": ";".join(f"{a:.4f}" for a in fold_auc),
        }
        rows.append(row)
        # incremental write so partial progress is visible
        pd.DataFrame(rows).sort_values("mean_auc", ascending=False).to_csv(summary_csv, index=False)
        print(f"  >>> mask {name} [{category}]: mean AUC={row['mean_auc']:.3f} "
              f"(worst {row['worst_fold_auc']:.3f}, pretext val_loss {row['pretrain_best_val_loss']:.4f})", flush=True)

    raw_dataset.close()
    ssl_dataset.close()
    print(f"\nablation summary written: {summary_csv}", flush=True)
    print(pd.DataFrame(rows).sort_values("mean_auc", ascending=False).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
