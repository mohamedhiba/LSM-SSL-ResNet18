#!/usr/bin/env python
"""Sequential two-stage SSL pretraining, ps32.

Stage 1 pretrains a modified ResNet-18 encoder with a stable pretext (masked
reconstruction by default, or contrastive) for N epochs. Stage 2 loads the
Stage-1 encoder weights (no reset), switches the pretext to cross-channel
masking at a reduced learning rate (default 2e-5; cross-channel collapses at
1e-4), and continues for M epochs. Only the encoder carries over between stages;
each stage uses its own task-specific decoder/head. The final Stage-2 encoder
can then be fine-tuned on the downstream landslide SCV task as usual.

This mirrors Te Pei's suggestion to stack pretext tasks in sequence: warm up the
encoder with a stable reconstruction/contrastive objective, then specialize it
with the learning-rate-sensitive cross-channel objective.

Inputs are 14-channel patches (13 terrain factors + 1 valid-context mask).
Run with a native arm64 PyTorch build so the MPS backend is available.

Usage:
    python scripts/pretrain_sequential_ssl.py --stage1-epochs 30 --stage2-epochs 20
    python scripts/pretrain_sequential_ssl.py --stage1-task contrastive --finetune
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential two-stage SSL pretraining, ps32.")
    parser.add_argument("--stage1-task", choices=["masked_recon", "contrastive"], default="masked_recon",
                        help="Stage-1 warm-up pretext (default masked_recon).")
    parser.add_argument("--stage1-epochs", type=int, default=30, help="N: Stage-1 epochs.")
    parser.add_argument("--stage2-epochs", type=int, default=20, help="M: Stage-2 cross-channel epochs.")
    parser.add_argument("--stage1-lr", type=float, default=1e-4, help="Stage-1 learning rate.")
    parser.add_argument("--stage2-lr", type=float, default=2e-5,
                        help="Stage-2 cross-channel learning rate (collapses at 1e-4).")
    parser.add_argument("--masked-channel-index", type=int, default=12,
                        help="Stage-2 terrain channel to mask (default 12 = TWI, landcover dropped).")
    parser.add_argument("--stage1-patience", type=int, default=0,
                        help="Stage-1 early-stopping patience; 0 disables (runs all N epochs).")
    parser.add_argument("--stage2-patience", type=int, default=0,
                        help="Stage-2 early-stopping patience; 0 disables (runs all M epochs).")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--n-unlabeled-patches", type=int, default=20000)
    parser.add_argument("--normalization-sample-size", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--mask-ratio", type=float, default=0.5, help="Stage-1 masked-recon block-mask ratio.")
    parser.add_argument("--block-size", type=int, default=4, help="Stage-1 masked-recon block size.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Stage-1 contrastive NT-Xent temperature.")
    parser.add_argument("--num-workers", type=int, default=0, help="Keep 0: lazy rasterio handles are not fork-safe.")
    parser.add_argument(
        "--cache-in-memory",
        dest="cache_in_memory",
        action="store_true",
        default=True,
        help="Cache all patches in RAM (default on); shared across both stages.",
    )
    parser.add_argument("--no-cache-in-memory", dest="cache_in_memory", action="store_false")
    parser.add_argument("--finetune", action="store_true",
                        help="After Stage 2, run the downstream 5-cluster SCV fine-tuning.")
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--finetune-patience", type=int, default=15)
    parser.add_argument("--finetune-batch-size", type=int, default=16)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--no-figures", action="store_true", help="Skip per-fold figure generation in fine-tuning.")
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
    from src.ssl_masked_recon import MaskedReconstructionModel
    from src.train_ssl import train_cross_channel_model, train_masked_reconstruction_model
    from src.utils import count_trainable_parameters, ensure_dir, get_device, set_global_seed

    raster_dir = PROJECT_ROOT / "data/processed/rasters_cleaned"
    unlabeled_index_csv = (
        PROJECT_ROOT
        / "data/processed/ssl_unlabeled_indices"
        / f"unlabeled_patch_index_ps{args.patch_size}_n{args.n_unlabeled_patches}.csv"
    )
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    output_root = PROJECT_ROOT / f"outputs/SSL_sequential_ps32{tag_suffix}"
    checkpoint_dir = ensure_dir(PROJECT_ROOT / f"checkpoints/ssl_pretrained/sequential{tag_suffix}")
    training_log_dir = ensure_dir(output_root / "training_logs")

    stage1_full = checkpoint_dir / f"resnet18_seq_stage1_{args.stage1_task}_ps32_full_model_best.pt"
    stage1_encoder = checkpoint_dir / f"resnet18_seq_stage1_{args.stage1_task}_ps32_encoder_best.pt"
    stage1_last = checkpoint_dir / f"resnet18_seq_stage1_{args.stage1_task}_ps32_last.pt"
    # The Stage-2 encoder is the final artifact consumed by fine-tuning.
    stage2_full = checkpoint_dir / "resnet18_sequential_ps32_full_model_best.pt"
    stage2_encoder = checkpoint_dir / "resnet18_sequential_ps32_encoder_best.pt"
    stage2_last = checkpoint_dir / "resnet18_sequential_ps32_last.pt"

    set_global_seed(args.random_seed)
    device = get_device()
    pin_memory = device.type == "cuda"
    print(f"device: {device}")

    raster_files = list_raster_files(raster_dir)
    audit_raster_alignment(raster_files, expected_nodata=DEFAULT_NODATA_VALUE)
    n_terrain = len(raster_files)
    in_channels = n_terrain + 1  # +1 valid-context mask
    if not 0 <= args.masked_channel_index < n_terrain:
        raise ValueError(
            f"--masked-channel-index must be a terrain channel in [0, {n_terrain - 1}], "
            f"got {args.masked_channel_index}."
        )
    print(f"stage-2 masked channel index {args.masked_channel_index}: "
          f"{raster_files[args.masked_channel_index].name}")

    create_unlabeled_patch_index(
        raster_dir=raster_dir,
        output_csv=unlabeled_index_csv,
        patch_size=args.patch_size,
        n_patches=args.n_unlabeled_patches,
        nodata_value=DEFAULT_NODATA_VALUE,
        max_nodata_ratio=0.0,
        random_seed=args.random_seed,
        max_attempts=1_000_000,
    )

    # ---- Build the shared in-RAM patch cache ONCE; reuse across both stages ----
    raw_dataset = CrossChannelMaskRasterDataset(
        patch_index_csv=unlabeled_index_csv,
        raster_dir=raster_dir,
        patch_size=args.patch_size,
        normalize=False,
        cache_in_memory=args.cache_in_memory,
    )
    if args.cache_in_memory:
        _ = raw_dataset[0]  # trigger cache build
    channel_means, channel_stds = compute_ssl_channel_stats(
        raw_dataset,
        sample_size=args.normalization_sample_size,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
    )
    print(f"channel stats length (terrain): {len(channel_means)}")

    ssl_dataset = CrossChannelMaskRasterDataset(
        patch_index_csv=unlabeled_index_csv,
        raster_dir=raster_dir,
        patch_size=args.patch_size,
        normalize=True,
        channel_means=channel_means,
        channel_stds=channel_stds,
        cache_in_memory=args.cache_in_memory,
    )
    if args.cache_in_memory:
        ssl_dataset._cache = raw_dataset._cache  # reuse the raw cache; avoid a second build

    train_size = int(0.9 * len(ssl_dataset))
    val_size = len(ssl_dataset) - train_size
    train_dataset, val_dataset = random_split(
        ssl_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.random_seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)
    print(f"train/val split sizes: {train_size}, {val_size}")

    # patience=0 means "run all epochs": use a patience >= epochs so early stopping never trips.
    stage1_patience = args.stage1_patience or (args.stage1_epochs + 1)
    stage2_patience = args.stage2_patience or (args.stage2_epochs + 1)

    # ============================== Stage 1 ==============================
    print(f"\n===== Stage 1: {args.stage1_task} warm-up ({args.stage1_epochs} epochs, lr={args.stage1_lr}) =====")
    if args.stage1_task == "masked_recon":
        stage1_model = MaskedReconstructionModel(in_channels=in_channels, out_channels=n_terrain).to(device)
        stage1_optimizer = torch.optim.AdamW(stage1_model.parameters(), lr=args.stage1_lr,
                                             weight_decay=args.weight_decay)
        print(f"stage-1 trainable parameters: {count_trainable_parameters(stage1_model)}")
        stage1_config = {"task": "masked_reconstruction", "stage": 1, "stage1_lr": args.stage1_lr,
                         "mask_ratio": args.mask_ratio, "block_size": args.block_size}
        stage1_log, stage1_info = train_masked_reconstruction_model(
            model=stage1_model, train_loader=train_loader, val_loader=val_loader,
            optimizer=stage1_optimizer, device=device,
            full_model_best_path=stage1_full, encoder_best_path=stage1_encoder,
            last_checkpoint_path=stage1_last, config=stage1_config,
            channel_means=channel_means, channel_stds=channel_stds,
            max_epochs=args.stage1_epochs, early_stopping_patience=stage1_patience,
            mask_ratio=args.mask_ratio, block_size=args.block_size,
            grad_clip_norm=args.gradient_clip_norm,
        )
    else:  # contrastive
        from src.ssl_contrastive import ContrastiveRasterPatchDataset, ContrastiveResNet18Model
        from src.train_ssl import train_contrastive_model
        contrastive_dataset = ContrastiveRasterPatchDataset(
            patch_index_csv=unlabeled_index_csv, raster_dir=raster_dir, patch_size=args.patch_size,
            normalize=True, channel_means=channel_means, channel_stds=channel_stds, augment=True,
        )
        c_train_size = int(0.9 * len(contrastive_dataset))
        c_val_size = len(contrastive_dataset) - c_train_size
        c_train, c_val = random_split(
            contrastive_dataset, [c_train_size, c_val_size],
            generator=torch.Generator().manual_seed(args.random_seed),
        )
        train_loader_c = DataLoader(c_train, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)
        val_loader_c = DataLoader(c_val, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=pin_memory)
        stage1_model = ContrastiveResNet18Model(in_channels=in_channels).to(device)
        stage1_optimizer = torch.optim.AdamW(stage1_model.parameters(), lr=args.stage1_lr,
                                             weight_decay=args.weight_decay)
        print(f"stage-1 trainable parameters: {count_trainable_parameters(stage1_model)}")
        stage1_config = {"task": "contrastive", "stage": 1, "stage1_lr": args.stage1_lr,
                         "temperature": args.temperature}
        stage1_log, stage1_info = train_contrastive_model(
            model=stage1_model, train_loader=train_loader_c, val_loader=val_loader_c,
            optimizer=stage1_optimizer, device=device,
            full_model_best_path=stage1_full, encoder_best_path=stage1_encoder,
            last_checkpoint_path=stage1_last, config=stage1_config,
            channel_means=channel_means, channel_stds=channel_stds,
            max_epochs=args.stage1_epochs, early_stopping_patience=stage1_patience,
            temperature=args.temperature, batch_size=args.batch_size,
            grad_clip_norm=args.gradient_clip_norm,
        )
    stage1_log.to_csv(training_log_dir / f"sequential_stage1_{args.stage1_task}_training_log.csv", index=False)
    print(f"stage 1 best: {stage1_info}")

    # ============================== Stage 2 ==============================
    print(f"\n===== Stage 2: cross-channel masking ({args.stage2_epochs} epochs, lr={args.stage2_lr}) =====")
    stage2_model = CrossChannelModel(in_channels=in_channels, out_channels=1).to(device)
    # Carry the Stage-1 encoder weights over with NO reset.
    stage1_ckpt = torch.load(stage1_encoder, map_location=device)
    missing, unexpected = stage2_model.encoder.load_state_dict(
        stage1_ckpt["encoder_state_dict"], strict=True
    )
    print(f"loaded Stage-1 encoder into Stage-2 (missing={list(missing)}, unexpected={list(unexpected)})")
    stage2_optimizer = torch.optim.AdamW(stage2_model.parameters(), lr=args.stage2_lr,
                                         weight_decay=args.weight_decay)
    stage2_config = {"task": "cross_channel", "stage": 2, "stage2_lr": args.stage2_lr,
                     "masked_channel_index": args.masked_channel_index,
                     "masked_channel_raster": raster_files[args.masked_channel_index].name,
                     "stage1_task": args.stage1_task, "stage1_epochs": args.stage1_epochs,
                     "stage2_epochs": args.stage2_epochs}
    print(json.dumps(stage2_config, indent=2))
    stage2_log, stage2_info = train_cross_channel_model(
        model=stage2_model, train_loader=train_loader, val_loader=val_loader,
        optimizer=stage2_optimizer, device=device,
        full_model_best_path=stage2_full, encoder_best_path=stage2_encoder,
        last_checkpoint_path=stage2_last, config=stage2_config,
        channel_means=channel_means, channel_stds=channel_stds,
        max_epochs=args.stage2_epochs, early_stopping_patience=stage2_patience,
        masked_channel_index=args.masked_channel_index, grad_clip_norm=args.gradient_clip_norm,
    )
    stage2_log.to_csv(training_log_dir / "sequential_stage2_cross_channel_training_log.csv", index=False)
    print(f"stage 2 best: {stage2_info}")
    print(f"final sequential encoder: {stage2_encoder}")

    raw_dataset.close()
    ssl_dataset.close()

    # ============================== Fine-tune ==============================
    if args.finetune:
        from src.train_finetune import run_pretrained_resnet18_scv_experiment
        labeled_index_csv = (
            PROJECT_ROOT / "data/processed/patches"
            / f"labeled_patch_index_ps{args.patch_size}_common_balanced.csv"
        )
        output_namespace = f"R4_sequential_ssl{tag_suffix}"
        print(f"\n===== Fine-tuning sequential encoder (namespace {output_namespace}) =====")
        run_pretrained_resnet18_scv_experiment(
            project_root=PROJECT_ROOT,
            model_name="sequential_resnet18",
            pretraining=f"sequential_{args.stage1_task}_to_cross_channel",
            patch_index_csv=labeled_index_csv,
            raster_dir=raster_dir,
            encoder_checkpoint_path=stage2_encoder,
            output_root=PROJECT_ROOT / f"outputs/{output_namespace}",
            figure_root=PROJECT_ROOT / f"figures/{output_namespace}",
            checkpoint_dir=PROJECT_ROOT / f"checkpoints/finetuned/sequential_resnet18{tag_suffix}",
            patch_size=args.patch_size,
            random_seed=args.random_seed,
            batch_size=args.finetune_batch_size,
            encoder_learning_rate=args.encoder_learning_rate,
            head_learning_rate=args.head_learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.finetune_epochs,
            early_stopping_patience=args.finetune_patience,
            gradient_clip_norm=args.gradient_clip_norm,
            dropout=args.dropout,
            expected_checkpoint_keys={"encoder_state_dict"},
            num_workers=args.num_workers,
            device=device,
            plot_figures=not args.no_figures,
            cache_in_memory=args.cache_in_memory,
        )


if __name__ == "__main__":
    main()
