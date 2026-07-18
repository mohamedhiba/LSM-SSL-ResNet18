#!/usr/bin/env python
"""Supervised SCV fine-tuning for the new SSL pretext tasks, ps32.

Fine-tunes SSL-pretrained ResNet-18 encoders (cross-channel masking and/or 1D
strip jigsaw) on the labeled common-balanced ps32 patches with the same
5-cluster spatial cross-validation protocol as notebooks 05/07/09/12/14, via
``run_pretrained_resnet18_scv_experiment``.

Outputs use the R2 namespace (outputs/R2_new_ssl_tasks, figures/R2_new_ssl_tasks)
to keep the new locally trained results separate from the archived R1 results,
which were produced on different hardware.

Usage:
    python scripts/finetune_new_ssl_tasks.py --task both
    python scripts/finetune_new_ssl_tasks.py --task cross_channel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TASKS = {
    "cross_channel": {
        "model_name": "cross_channel_resnet18",
        "pretraining": "cross_channel",
        "encoder_filename": "resnet18_cross_channel_ps32_encoder_best.pt",
        "scientific_note": (
            "Cross-channel masking SSL: one entire conditioning-factor channel "
            "(default TWI) is zeroed in the normalized input and predicted from "
            "the remaining 13 channels."
        ),
    },
    "strip_jigsaw": {
        "model_name": "strip_jigsaw_resnet18",
        "pretraining": "strip_jigsaw",
        "encoder_filename": "resnet18_strip_jigsaw_ps32_encoder_best.pt",
        "scientific_note": (
            "1D strip jigsaw SSL: each patch is split into 3 horizontal strips "
            "(11/11/10 rows) shuffled by one of 6 fixed permutations; the model "
            "predicts the permutation class."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCV fine-tuning for the new SSL pretext tasks, ps32.")
    parser.add_argument("--task", choices=[*TASKS, "both"], default="both")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--num-workers", type=int, default=0, help="Keep 0: lazy rasterio handles are not fork-safe.")
    parser.add_argument("--no-figures", action="store_true", help="Skip per-fold figure generation.")
    parser.add_argument(
        "--cache-in-memory",
        dest="cache_in_memory",
        action="store_true",
        default=True,
        help="Cache the labeled patches in RAM (default on); avoids re-reading them every epoch.",
    )
    parser.add_argument("--no-cache-in-memory", dest="cache_in_memory", action="store_false")
    parser.add_argument(
        "--no-mask-channel",
        dest="no_mask_channel",
        action="store_true",
        help="Ablate the valid-context mask channel (in_channels = n_terrain). Must match the encoder's pretraining.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional suffix for the R2 output dirs (and, by default, the encoder dirs), e.g. 'full'.",
    )
    parser.add_argument(
        "--encoder-run-tag",
        type=str,
        default=None,
        help="Override the suffix used to LOCATE the SSL encoders. Defaults to --run-tag. "
        "Use e.g. --encoder-run-tag full --run-tag diag to re-finetune the full-run encoders "
        "into a separate diagnostic namespace.",
    )
    return parser.parse_args()


def load_scratch_reference_metrics() -> dict[str, dict[str, float]] | None:
    """Load the archived scratch-baseline summary as reference metrics."""

    import pandas as pd

    summary_csv = PROJECT_ROOT / "results_summary/tables/all_ssl_tasks_ps32_summary_metrics.csv"
    if not summary_csv.exists():
        return None
    summary = pd.read_csv(summary_csv)
    scratch = summary.loc[summary["model_display_name"] == "Scratch"]
    if scratch.empty:
        return None
    row = scratch.iloc[0]
    return {
        "scratch": {
            "mean_auc": float(row["mean_auc"]),
            "std_auc": float(row["std_auc"]),
            "worst_fold_auc": float(row["worst_fold_auc"]),
        }
    }


def main() -> None:
    args = parse_args()

    from src.train_finetune import run_pretrained_resnet18_scv_experiment
    from src.utils import get_device, set_global_seed

    patch_index_csv = (
        PROJECT_ROOT / "data/processed/patches"
        / f"labeled_patch_index_ps{args.patch_size}_common_balanced.csv"
    )
    raster_dir = PROJECT_ROOT / "data/processed/rasters_cleaned"

    set_global_seed(args.random_seed)
    device = get_device()
    print(f"device: {device}")

    reference_metrics = load_scratch_reference_metrics()
    if reference_metrics is not None:
        print(f"archived scratch reference: {reference_metrics['scratch']}")

    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    output_namespace = f"R2_new_ssl_tasks{tag_suffix}"
    encoder_run_tag = args.run_tag if args.encoder_run_tag is None else args.encoder_run_tag
    encoder_tag_suffix = f"_{encoder_run_tag}" if encoder_run_tag else ""

    task_names = list(TASKS) if args.task == "both" else [args.task]
    results = {}
    for task_name in task_names:
        task = TASKS[task_name]
        encoder_checkpoint_path = (
            PROJECT_ROOT
            / f"checkpoints/ssl_pretrained/{task_name}{encoder_tag_suffix}"
            / task["encoder_filename"]
        )
        if not encoder_checkpoint_path.exists():
            raise FileNotFoundError(
                f"SSL encoder checkpoint not found: {encoder_checkpoint_path}. "
                f"Run scripts/pretrain_{task_name}_ssl.py"
                + (f" --run-tag {encoder_run_tag}" if encoder_run_tag else "")
                + " first."
            )

        print(f"\n===== fine-tuning {task['model_name']} (namespace {output_namespace}) =====")
        results[task_name] = run_pretrained_resnet18_scv_experiment(
            project_root=PROJECT_ROOT,
            model_name=task["model_name"],
            pretraining=task["pretraining"],
            patch_index_csv=patch_index_csv,
            raster_dir=raster_dir,
            encoder_checkpoint_path=encoder_checkpoint_path,
            output_root=PROJECT_ROOT / f"outputs/{output_namespace}/{task_name}",
            figure_root=PROJECT_ROOT / f"figures/{output_namespace}/{task_name}",
            checkpoint_dir=PROJECT_ROOT / f"checkpoints/finetuned/{task['model_name']}{tag_suffix}",
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
            reference_metrics=reference_metrics,
            expected_checkpoint_keys={"encoder_state_dict"},
            scientific_note=task["scientific_note"],
            num_workers=args.num_workers,
            device=device,
            plot_figures=not args.no_figures,
            cache_in_memory=args.cache_in_memory,
            with_mask=not args.no_mask_channel,
        )

    for task_name, result in results.items():
        summary = result.get("summary_metrics") if isinstance(result, dict) else None
        print(f"\n===== {task_name} done =====")
        if summary is not None:
            print(summary)


if __name__ == "__main__":
    main()
