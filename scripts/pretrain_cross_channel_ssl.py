#!/usr/bin/env python
"""SSL cross-channel masking pretraining, ps32.

Pretrains a modified ResNet-18 encoder by zeroing one entire channel (default:
TWI, channel index 12 after landcover is dropped) in each normalized 13-channel
patch and predicting that channel with MSE loss. Mirrors notebook 06 (masked
reconstruction) end to end.

This is in-domain transductive SSL pretraining: unlabeled patches are sampled
from the whole cleaned study area. SCV holdout clusters are not excluded
because no landslide/non-landslide labels are used during SSL pretraining.

On Apple Silicon, run with a native arm64 PyTorch build (e.g. torch==2.3.1
macOS arm64 wheel) so the MPS backend is available; the conda environment.yml
pins a CUDA build and does not install on macOS.

Usage:
    python scripts/pretrain_cross_channel_ssl.py
    python scripts/pretrain_cross_channel_ssl.py --masked-channel-index 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSL cross-channel masking pretraining, ps32.")
    parser.add_argument("--masked-channel-index", type=int, default=12, help="Channel to mask (default 12 = TWI, landcover dropped).")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--n-unlabeled-patches", type=int, default=20000)
    parser.add_argument("--normalization-sample-size", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0, help="Keep 0: lazy rasterio handles are not fork-safe.")
    parser.add_argument(
        "--cache-in-memory",
        dest="cache_in_memory",
        action="store_true",
        default=True,
        help="Cache all patches in RAM (default on); avoids re-reading 13 GeoTIFFs per patch each epoch.",
    )
    parser.add_argument("--no-cache-in-memory", dest="cache_in_memory", action="store_false")
    parser.add_argument(
        "--center-only-unlabeled",
        action="store_true",
        help="Sample unlabeled patches by center-only validity (boundary/NoData zero-padded) so the "
        "valid-context mask varies during pretraining; writes a separate '_centeronly' index.",
    )
    parser.add_argument(
        "--no-mask-channel",
        dest="no_mask_channel",
        action="store_true",
        help="Ablate the valid-context mask channel: train on terrain channels only (in_channels = "
        "n_terrain). Used to isolate the landcover-drop effect from the mask channel.",
    )
    parser.add_argument("--run-tag", type=str, default="", help="Optional suffix for checkpoint/output dirs, e.g. 'full'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader, random_split

    from src.patch_dataset import DEFAULT_NODATA_VALUE, audit_raster_alignment, list_raster_files
    from src.ssl_cross_channel import (
        CrossChannelMaskRasterDataset,
        CrossChannelModel,
        compute_ssl_channel_stats,
        create_unlabeled_patch_index,
    )
    from src.train_ssl import train_cross_channel_model
    from src.utils import count_trainable_parameters, ensure_dir, get_device, set_global_seed

    raster_dir = PROJECT_ROOT / "data/processed/rasters_cleaned"
    index_suffix = "_centeronly" if args.center_only_unlabeled else ""
    unlabeled_index_csv = (
        PROJECT_ROOT
        / "data/processed/ssl_unlabeled_indices"
        / f"unlabeled_patch_index_ps{args.patch_size}_n{args.n_unlabeled_patches}{index_suffix}.csv"
    )
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    output_root = PROJECT_ROOT / f"outputs/SSL_cross_channel_ps32{tag_suffix}"
    checkpoint_dir = ensure_dir(PROJECT_ROOT / f"checkpoints/ssl_pretrained/cross_channel{tag_suffix}")
    training_log_dir = ensure_dir(output_root / "training_logs")

    full_model_best_path = checkpoint_dir / "resnet18_cross_channel_ps32_full_model_best.pt"
    encoder_best_path = checkpoint_dir / "resnet18_cross_channel_ps32_encoder_best.pt"
    last_checkpoint_path = checkpoint_dir / "resnet18_cross_channel_ps32_last.pt"
    training_log_csv = training_log_dir / "cross_channel_ps32_training_log.csv"

    set_global_seed(args.random_seed)
    device = get_device()
    pin_memory = device.type == "cuda"
    print(f"device: {device}")

    raster_files = list_raster_files(raster_dir)
    audit_raster_alignment(raster_files, expected_nodata=DEFAULT_NODATA_VALUE)
    if not 0 <= args.masked_channel_index < len(raster_files):
        raise ValueError(
            f"--masked-channel-index must be in [0, {len(raster_files) - 1}], got {args.masked_channel_index}."
        )
    print(f"masked channel index {args.masked_channel_index}: {raster_files[args.masked_channel_index].name}")

    unlabeled_index = create_unlabeled_patch_index(
        raster_dir=raster_dir,
        output_csv=unlabeled_index_csv,
        patch_size=args.patch_size,
        n_patches=args.n_unlabeled_patches,
        nodata_value=DEFAULT_NODATA_VALUE,
        max_nodata_ratio=0.0,
        random_seed=args.random_seed,
        max_attempts=1_000_000,
        center_only=args.center_only_unlabeled,
    )
    print(f"number of valid unlabeled patches: {len(unlabeled_index)}")

    raw_dataset = CrossChannelMaskRasterDataset(
        patch_index_csv=unlabeled_index_csv,
        raster_dir=raster_dir,
        patch_size=args.patch_size,
        normalize=False,
    )
    channel_means, channel_stds = compute_ssl_channel_stats(
        raw_dataset,
        sample_size=args.normalization_sample_size,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
    )
    raw_dataset.close()
    print(f"channel_means shape: {channel_means.shape}")
    print(f"channel_stds shape: {channel_stds.shape}")

    with_mask = not args.no_mask_channel
    ssl_dataset = CrossChannelMaskRasterDataset(
        patch_index_csv=unlabeled_index_csv,
        raster_dir=raster_dir,
        patch_size=args.patch_size,
        normalize=True,
        channel_means=channel_means,
        channel_stds=channel_stds,
        cache_in_memory=args.cache_in_memory,
        with_mask=with_mask,
    )
    train_size = int(0.9 * len(ssl_dataset))
    val_size = len(ssl_dataset) - train_size
    train_dataset, val_dataset = random_split(
        ssl_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.random_seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    print(f"train/val split sizes: {train_size}, {val_size}")

    in_channels = len(raster_files) + (1 if with_mask else 0)  # +1 valid-context mask unless ablated
    model = CrossChannelModel(in_channels=in_channels, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print(f"trainable parameters: {count_trainable_parameters(model)}")

    config = {
        "task": "cross_channel",
        "patch_size": args.patch_size,
        "n_unlabeled_patches": args.n_unlabeled_patches,
        "masked_channel_index": args.masked_channel_index,
        "masked_channel_raster": raster_files[args.masked_channel_index].name,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "gradient_clip_norm": args.gradient_clip_norm,
        "random_seed": args.random_seed,
        "with_mask_channel": with_mask,
        "in_channels": in_channels,
        "center_only_unlabeled": args.center_only_unlabeled,
        "device": str(device),
    }
    print(json.dumps(config, indent=2))

    training_log_df, best_info = train_cross_channel_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        full_model_best_path=full_model_best_path,
        encoder_best_path=encoder_best_path,
        last_checkpoint_path=last_checkpoint_path,
        config=config,
        channel_means=channel_means,
        channel_stds=channel_stds,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        masked_channel_index=args.masked_channel_index,
        grad_clip_norm=args.gradient_clip_norm,
        mask_channel_present=with_mask,
    )
    ssl_dataset.close()

    training_log_df.to_csv(training_log_csv, index=False)
    print(f"training log: {training_log_csv}")
    print(f"best checkpoint: {encoder_best_path}")
    print(f"best info: {best_info}")


if __name__ == "__main__":
    main()
