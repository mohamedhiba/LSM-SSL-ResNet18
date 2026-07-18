#!/usr/bin/env python
"""Fine-tune the sequential-SSL encoder on the downstream landslide SCV task.

Re-finetunes the encoder produced by ``scripts/pretrain_sequential_ssl.py``
(Stage 1 masked reconstruction -> Stage 2 cross-channel masking) without
re-pretraining. Defaults to the stable, reduced learning rate (head 2e-5,
encoder 5e-6, 60 epochs, patience 12) used for the canonical R2-diag
cross-channel numbers, because cross-channel-finalized encoders collapse under
the standard 1e-4 finetune LR. Outputs go to the R4_sequential_ssl_diag
namespace so they stay separate from the default-LR run.

Usage:
    python scripts/finetune_sequential_ssl.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable-LR SCV fine-tune of the sequential-SSL encoder.")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-learning-rate", type=float, default=5e-6)
    parser.add_argument("--head-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--run-tag", type=str, default="diag",
                        help="Suffix for the R4 namespace and finetuned-checkpoint dir (default 'diag').")
    parser.add_argument("--encoder-run-tag", type=str, default="",
                        help="Suffix that locates the sequential encoder dir (default '': checkpoints/.../sequential).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.train_finetune import run_pretrained_resnet18_scv_experiment
    from src.utils import get_device, set_global_seed

    raster_dir = PROJECT_ROOT / "data/processed/rasters_cleaned"
    patch_index_csv = (
        PROJECT_ROOT / "data/processed/patches"
        / f"labeled_patch_index_ps{args.patch_size}_common_balanced.csv"
    )
    enc_tag = f"_{args.encoder_run_tag}" if args.encoder_run_tag else ""
    encoder_checkpoint_path = (
        PROJECT_ROOT / f"checkpoints/ssl_pretrained/sequential{enc_tag}"
        / "resnet18_sequential_ps32_encoder_best.pt"
    )
    if not encoder_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Sequential encoder not found: {encoder_checkpoint_path}. "
            "Run scripts/pretrain_sequential_ssl.py first."
        )

    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    output_namespace = f"R4_sequential_ssl{tag_suffix}"

    set_global_seed(args.random_seed)
    device = get_device()
    print(f"device: {device}")
    print(f"sequential encoder: {encoder_checkpoint_path}")
    print(f"namespace: {output_namespace} | encoder LR {args.encoder_learning_rate} | head LR {args.head_learning_rate}")

    result = run_pretrained_resnet18_scv_experiment(
        project_root=PROJECT_ROOT,
        model_name="sequential_resnet18",
        pretraining="sequential_masked_recon_to_cross_channel",
        patch_index_csv=patch_index_csv,
        raster_dir=raster_dir,
        encoder_checkpoint_path=encoder_checkpoint_path,
        output_root=PROJECT_ROOT / f"outputs/{output_namespace}",
        figure_root=PROJECT_ROOT / f"figures/{output_namespace}",
        checkpoint_dir=PROJECT_ROOT / f"checkpoints/finetuned/sequential_resnet18{tag_suffix}",
        patch_size=args.patch_size,
        random_seed=args.random_seed,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        encoder_learning_rate=args.encoder_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        gradient_clip_norm=args.gradient_clip_norm,
        dropout=args.dropout,
        expected_checkpoint_keys={"encoder_state_dict"},
        num_workers=args.num_workers,
        device=device,
        plot_figures=not args.no_figures,
        cache_in_memory=True,
    )
    summary = result["summary_metrics"].iloc[0]
    print("\n===== sequential (stable-LR) SCV summary =====")
    print(f"mean AUC      : {summary['mean_auc']:.4f}")
    print(f"AUC SD        : {summary['std_auc']:.4f}")
    print(f"worst-fold AUC: {summary['worst_fold_auc']:.4f}")


if __name__ == "__main__":
    main()
