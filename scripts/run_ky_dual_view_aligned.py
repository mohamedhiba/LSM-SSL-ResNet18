"""Kentucky run driver for Qianyi's final dual-view protocol (protocol-aligned).

Thin wrapper around HIS code (LSM_SSL_ResNet18_10m_dual_view_code): SSL
pretraining via ``final_ssl_runner.run_final_ssl_pretraining`` and downstream
flat-CV jobs via ``final_flat_cv_lr_sweep_runner._run_one_full_job`` — his
model, optimizer grouping (head 1e-3 / swept encoder LR), pos_weight BCE,
batch 128, AMP, early stopping, QA columns, and resume semantics all come
from his modules unchanged.

What this driver replaces (and why):
  - his top-level wrappers hard-gate on NYC literals (562-row labeled set,
    625-job audited manifest, Step-1 preflight CSVs, a Windows-backslash
    checkpoint-path marker that never matches on Linux) → we build the job
    manifest ourselves;
  - his internal StratifiedKFold split → we use the group-safe
    ``ky_flat_cv_split.csv`` (StratifiedGroupKFold by original_feature_id);
  - ``FinalSSLTrainingConfig`` carries no data paths → we patch
    ``final_ssl_runner._paths`` to point at the KY corpus/stats/rasters;
  - his SSL train/val split hardcodes train_size=18000 (correct only for a
    20k corpus) → patched to a 90/10 split of whatever corpus is given;
  - his lazy labeled dataset re-reads rasters every epoch (fine for 562
    samples, fatal for 18,396) → an in-memory tensor cache that preserves
    the exact normalized tensors, mirroring his unlabeled SSL cache.

MUST run in a process that never imports this repo's own src/ —
ky_dual_view_shim is imported first and puts his package at sys.path[0].
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ky_dual_view_shim as shim  # noqa: E402  (must precede any src.* import)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = shim.REPO_ROOT
KY_DV = Path(os.environ.get("KY_DUAL_VIEW_DIR", REPO / "data" / "kentucky_dual_view"))

LABELED_CSV = KY_DV / "dual_view_padded_patch_index_ky10m_pu_mean_oob_balanced.csv"
SPLIT_CSV = KY_DV / "ky_flat_cv_split.csv"
STATS_JSON = KY_DV / "normalization_stats_ky10m_13factors.json"
UNLABELED_50K = KY_DV / "unlabeled_dual_view_padded_index_ky10m_n50000.csv"

OUT_BASE = REPO / "outputs" / "KY_dual_view_aligned"
CKPT_BASE = REPO / "checkpoints" / "KY_dual_view_aligned"
FIG_BASE = REPO / "figures" / "KY_dual_view_aligned"

TASK_ALIASES = {
    "masked_recon": "dual_view_masked_reconstruction",
    "masked_reconstruction": "dual_view_masked_reconstruction",
    "contrastive": "dual_view_contrastive_learning",
    "jigsaw": "dual_view_jigsaw",
    "rotation": "dual_view_rotation_prediction",
}


def _canon_tasks(names: list[str]) -> list[str]:
    out = []
    for name in names:
        if name == "none":
            continue
        canon = TASK_ALIASES.get(name, name)
        if canon not in TASK_ALIASES.values():
            raise SystemExit(f"Unknown SSL task: {name}")
        out.append(canon)
    return out


def _encoder_checkpoint(task: str, seed: int, *, smoke: bool = False) -> Path:
    base = CKPT_BASE / ("smoke/ssl" if smoke else "ssl")
    return (
        base / "ssl_task_comparison" / task / f"seed_{seed}"
        / f"{task}_seed{seed}_encoder_best.pt"
    )


# ---------------------------------------------------------------- SSL stage

def run_ssl_stage(args, *, labeled_csv: Path, unlabeled_csv: Path) -> None:
    import src.final_ssl_runner as fsr
    from src.final_protocol import FinalProtocolPaths

    ssl_out = OUT_BASE / "ssl" if not args.smoke else OUT_BASE / "smoke" / "ssl"
    ssl_fig = FIG_BASE / "ssl" if not args.smoke else FIG_BASE / "smoke" / "ssl"
    ssl_ckpt = CKPT_BASE / "ssl" if not args.smoke else CKPT_BASE / "smoke" / "ssl"

    def _ky_paths(config):
        return FinalProtocolPaths(
            project_root=REPO,
            labeled_index_csv=labeled_csv,
            unlabeled_index_csv=unlabeled_csv,
            raster_dir=shim.KY_RASTER_DIR,
            normalization_stats_path=STATS_JSON,
            ssl_stats_checkpoint_path=STATS_JSON,
            outputs_root=Path(config.outputs_root),
            figures_root=Path(config.figures_root),
            checkpoints_root=Path(config.checkpoints_root),
        ).resolve()

    def _ky_split_indices(n_samples, seed, train_size=None):
        # his hardcoded train_size=18000 assumes the 20k corpus; keep his
        # 90/10 ratio for any corpus size instead
        train_size = int(round(0.9 * int(n_samples)))
        rng = np.random.default_rng(int(seed))
        indices = rng.permutation(int(n_samples))
        return indices[:train_size].tolist(), indices[train_size:].tolist()

    fsr._paths = _ky_paths
    fsr._split_indices = _ky_split_indices
    fsr.SSL_TASK_ORDER[:] = list(args.ssl_tasks)

    config = fsr.FinalSSLTrainingConfig(
        project_root=REPO,
        seed_list=tuple(args.seeds),
        outputs_root=ssl_out,
        figures_root=ssl_fig,
        checkpoints_root=ssl_ckpt,
        ssl_batch_size=64,
        ssl_max_epochs=args.ssl_epochs,
        ssl_patience=args.ssl_patience,
        num_workers=0,
        use_amp=True,
        cache_unlabeled_ssl_patches=True,
    )
    print(
        f"[ssl] tasks={args.ssl_tasks} seeds={args.seeds} corpus={unlabeled_csv.name} "
        f"epochs<={args.ssl_epochs} patience={args.ssl_patience}",
        flush=True,
    )
    manifest = fsr.run_final_ssl_pretraining(config=config, execute=True)
    print(manifest[["task_name", "seed", "checkpoint_status", "qa_passed"]].to_string(index=False), flush=True)
    bad = manifest.loc[~manifest["qa_passed"].astype(bool)]
    if not bad.empty:
        raise RuntimeError(f"SSL pretraining QA failed:\n{bad}")


# --------------------------------------------------------- downstream stage

class _InMemoryDualViewLabeledDataset:
    """RAM cache of the exact normalized labeled tensors (his lazy dataset
    would re-read 26 raster windows per sample per epoch)."""

    def __init__(self, base_dataset, *, progress_every: int = 2000) -> None:
        import torch

        self.torch = torch
        self.n = len(base_dataset)
        if self.n <= 0:
            raise ValueError("Cannot cache an empty labeled dataset.")
        first = base_dataset[0]
        self.local = torch.empty((self.n, *tuple(first["local"].shape)), dtype=torch.float32)
        self.global_x = torch.empty((self.n, *tuple(first["global"].shape)), dtype=torch.float32)
        self.labels = torch.empty((self.n,), dtype=torch.long)
        self.sample_ids: list[object] = [None] * self.n
        self.cluster_ids: list[int] = [0] * self.n
        self.metadata: list[dict[str, object]] = [{} for _ in range(self.n)]
        self._store(0, first)
        for idx in range(1, self.n):
            self._store(idx, base_dataset[idx])
            if progress_every and (idx + 1) % int(progress_every) == 0:
                print(f"cached_labeled_patches={idx + 1}/{self.n}", flush=True)
        print(f"cached_labeled_patches={self.n}/{self.n}", flush=True)

    def _store(self, idx: int, sample: dict[str, object]) -> None:
        self.local[idx].copy_(sample["local"])
        self.global_x[idx].copy_(sample["global"])
        self.labels[idx] = int(sample["label"])
        self.sample_ids[idx] = sample["sample_id"]
        self.cluster_ids[idx] = int(sample["cluster_id"])
        self.metadata[idx] = dict(sample.get("metadata", {}))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "local": self.local[index],
            "global": self.global_x[index],
            "label": self.labels[index],
            "sample_id": self.sample_ids[index],
            "cluster_id": self.cluster_ids[index],
            "metadata": self.metadata[index],
        }

    def close(self) -> None:
        return None


def _build_jobs(args, config) -> list[dict[str, object]]:
    ckpt_dir = Path(config.checkpoints_root) / "downstream"
    pred_base = Path(config.outputs_root) / "downstream"
    jobs: list[dict[str, object]] = []

    def add(model_name: str, ssl_task: str, seed: int, fold: int, encoder_lr) -> None:
        if encoder_lr == "" or encoder_lr is None:
            tag = "scratch"
            input_ckpt = ""
        else:
            tag = f"{float(encoder_lr):g}".replace("+", "").replace(".", "p")
            input_ckpt = str(_encoder_checkpoint(ssl_task, seed, smoke=args.smoke))
        stem = f"{model_name}_seed{seed}_fold{fold}_encoderlr{tag}"
        pred_dir = pred_base / (
            "scratch_full_model_baseline" if model_name == "scratch_full_model_baseline" else model_name
        ) / "predictions"
        jobs.append(
            {
                "job_type": "full_scratch_baseline" if input_ckpt == "" else "full_ssl_finetune",
                "model_name": model_name,
                "ssl_task": ssl_task,
                "seed": int(seed),
                "fold_id": int(fold),
                "encoder_lr": "" if input_ckpt == "" else float(encoder_lr),
                "input_checkpoint_path": input_ckpt,
                "output_checkpoint_path": str(ckpt_dir / f"{stem}_best.pt"),
                "output_prediction_path": str(pred_dir / f"{stem}_predictions.csv"),
            }
        )

    # scratch first: gives the baseline column earliest
    if args.include_scratch:
        for seed in args.seeds:
            for fold in args.folds:
                add("scratch_full_model_baseline", "", seed, fold, "")
    for task in args.ssl_tasks:
        for lr in args.encoder_lrs:
            for seed in args.seeds:
                for fold in args.folds:
                    add(task, task, seed, fold, lr)
    return jobs


def run_downstream_stage(args, *, labeled_csv: Path, split_csv: Path) -> None:
    import torch

    import src.final_flat_cv_lr_sweep_runner as fcv
    from src.patch_dataset import DualViewPaddedRasterPatchDataset
    from src.train_finetune import _load_dual_view_ssl_normalization_stats

    tag = args.tag
    base = ("smoke/" if args.smoke else "") + f"flat_cv/{tag}"
    config = fcv.FinalFlatCVLRSweepConfig(
        project_root=REPO,
        outputs_root=OUT_BASE / base,
        figures_root=FIG_BASE / base,
        checkpoints_root=CKPT_BASE / base,
        labeled_patch_index=labeled_csv,
        flat_cv_split_assignment=split_csv,
        conditioning_factor_folder=shim.KY_RASTER_DIR,
        normalization_stats=STATS_JSON,
        head_lr=1e-3,
        scratch_lr=1e-3,
        encoder_lr_grid=tuple(float(v) for v in args.encoder_lrs),
        seed_list=tuple(args.seeds),
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        num_workers=0,
        use_amp=True,
        # Slack spec 2026-07-13: LR=0 strictly freezes weights AND BN stats
        # (his code default is False; overridden per Mohamed's directive)
        strict_frozen_encoder=args.strict_frozen_encoder,
    ).resolve()

    jobs = _build_jobs(args, config)
    manifest = pd.DataFrame(jobs)
    manifest_dir = Path(config.outputs_root) / "comparison"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_dir / "ky_run_manifest.csv", index=False)
    (manifest_dir / "ky_run_config.json").write_text(
        json.dumps(fcv._json_ready(fcv.asdict(config)), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[downstream] tag={tag} jobs={len(jobs)} -> {fcv._full_metrics_path(config)}", flush=True)

    if args.train_fraction < 1.0:
        # label-efficiency mode: stratified subsample of each fold's TRAIN set.
        # Seeded by (seed, fold, fraction) only — every method sees the same
        # labeled subset, so SSL-vs-scratch stays a paired comparison.
        frac = float(args.train_fraction)
        base_seed = int(args.seeds[0])
        orig_fold_indices = fcv._fold_indices

        def _frac_fold_indices(labeled_index, split_assignment, fold_id):
            train_idx, eval_idx = orig_fold_indices(labeled_index, split_assignment, fold_id)
            labels = labeled_index.iloc[train_idx]["label"].to_numpy()
            rng = np.random.default_rng(base_seed * 1_000_000 + int(fold_id) * 1000 + int(round(frac * 10000)))
            keep: list[int] = []
            for cls in (0, 1):
                cls_idx = [i for i, lab in zip(train_idx, labels) if lab == cls]
                n_keep = max(8, int(round(len(cls_idx) * frac)))
                keep.extend(rng.choice(cls_idx, size=n_keep, replace=False).tolist())
            return sorted(keep), eval_idx

        fcv._fold_indices = _frac_fold_indices
        print(f"[downstream] label-efficiency mode: train fraction {frac:g}", flush=True)

    means, stds, norm_metadata = _load_dual_view_ssl_normalization_stats(config.normalization_stats)
    if len(means) != 13 or len(stds) != 13:
        raise RuntimeError("KY normalization stats must contain 13 factor channels.")
    labeled_index = pd.read_csv(config.labeled_patch_index)
    if "selected" in labeled_index.columns and not labeled_index["selected"].astype(bool).all():
        raise RuntimeError(
            "Labeled index contains selected=False rows; the lazy dataset would drop "
            "them and positional indices would misalign."
        )
    labeled_index = labeled_index.reset_index(drop=True)
    split_assignment = pd.read_csv(config.flat_cv_split_assignment)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lazy = DualViewPaddedRasterPatchDataset(
        config.labeled_patch_index,
        config.conditioning_factor_folder,
        local_size=config.local_size,
        global_size=config.global_size,
        normalize=True,
        channel_means=means,
        channel_stds=stds,
        return_metadata=True,
    )
    if len(lazy) != len(labeled_index):
        lazy.close()
        raise RuntimeError(f"Dataset rows ({len(lazy)}) != labeled index rows ({len(labeled_index)}).")
    if args.cache_labeled:
        t0 = time.time()
        dataset = _InMemoryDualViewLabeledDataset(lazy)
        lazy.close()
        print(f"[downstream] labeled cache built in {time.time() - t0:.0f}s", flush=True)
    else:
        dataset = lazy

    current = fcv._load_existing_fold_metrics(config)
    start = time.time()
    n_done = 0
    try:
        for job_idx, job in enumerate(jobs, start=1):
            valid, reason = fcv._completed_job_valid(config, job, current)
            if valid:
                print(f"[{job_idx}/{len(jobs)}] skip (resume) {fcv._job_stem(job)}", flush=True)
                continue
            if job["input_checkpoint_path"]:
                enc = Path(job["input_checkpoint_path"])
                while not enc.exists():
                    if not args.wait_encoders:
                        raise FileNotFoundError(f"Missing SSL encoder: {enc}")
                    print(f"[{job_idx}/{len(jobs)}] waiting for encoder {enc.name}", flush=True)
                    time.sleep(120)
            print(
                f"[{job_idx}/{len(jobs)}] running {fcv._job_stem(job)} (reason={reason})",
                flush=True,
            )
            # his line `float(encoder_lr)` crashes on scratch rows (lr="") when
            # strict_frozen_encoder=True; strict freeze is meaningless for
            # scratch, so hand those jobs a non-strict config copy
            job_config = config
            if not job["input_checkpoint_path"] and config.strict_frozen_encoder:
                from dataclasses import replace
                job_config = replace(config, strict_frozen_encoder=False)
            row = fcv._run_one_full_job(
                config=job_config,
                job=job,
                labeled_index=labeled_index,
                split_assignment=split_assignment,
                dataset=dataset,
                means=means,
                stds=stds,
                norm_metadata=norm_metadata,
                device=device,
                pin_memory=device.type == "cuda",
            )
            current = fcv._replace_metric_row(current, row)
            metrics_path = fcv._full_metrics_path(config)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            current.to_csv(metrics_path, index=False)
            n_done += 1
            elapsed = time.time() - start
            print(
                f"[{job_idx}/{len(jobs)}] DONE {fcv._job_stem(job)} auc={row['auc']:.4f} "
                f"pr_auc={row['pr_auc']:.4f} best_epoch={row['best_epoch']} "
                f"elapsed_h={elapsed / 3600:.2f} avg_min_per_job={elapsed / 60 / max(n_done, 1):.1f}",
                flush=True,
            )
    finally:
        dataset.close()
    print(f"[downstream] tag={tag} complete: {len(current)} metric rows", flush=True)


# ---------------------------------------------------------------- smoke mode

def _make_smoke_subsets(n_labeled: int = 1200, n_unlabeled: int = 1500) -> tuple[Path, Path, Path]:
    smoke_dir = OUT_BASE / "smoke" / "data"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    labeled = pd.read_csv(LABELED_CSV)
    sub = pd.concat(
        [
            labeled.loc[labeled["label"] == 1].sample(n=n_labeled // 2, random_state=42),
            labeled.loc[labeled["label"] == 0].sample(n=n_labeled // 2, random_state=42),
        ]
    ).reset_index(drop=True)
    labeled_path = smoke_dir / "smoke_labeled.csv"
    sub.to_csv(labeled_path, index=False)
    split = pd.read_csv(SPLIT_CSV)
    split_sub = split.loc[split["sample_id"].isin(set(sub["sample_id"]))].reset_index(drop=True)
    split_path = smoke_dir / "smoke_split.csv"
    split_sub.to_csv(split_path, index=False)
    unlabeled = pd.read_csv(UNLABELED_50K, nrows=n_unlabeled)
    unlabeled_path = smoke_dir / "smoke_unlabeled.csv"
    unlabeled.to_csv(unlabeled_path, index=False)
    return labeled_path, split_path, unlabeled_path


# ----------------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["ssl", "downstream", "all", "smoke"], required=True)
    parser.add_argument("--tag", default=None, help="downstream output namespace (required unless smoke)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--ssl-tasks", nargs="+", default=["masked_recon", "contrastive"])
    parser.add_argument("--encoder-lrs", type=float, nargs="+", default=[0.0, 1e-5, 1e-4])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--include-scratch", action="store_true")
    parser.add_argument("--unlabeled-index", type=Path, default=UNLABELED_50K)
    parser.add_argument("--ssl-epochs", type=int, default=50)
    parser.add_argument("--ssl-patience", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-cache-labeled", dest="cache_labeled", action="store_false")
    parser.add_argument("--strict-frozen-encoder", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument(
        "--wait-encoders", action="store_true",
        help="poll for missing SSL encoder checkpoints instead of failing",
    )
    args = parser.parse_args()
    args.smoke = args.stage == "smoke"
    args.ssl_tasks = _canon_tasks(args.ssl_tasks)

    for path in [LABELED_CSV, SPLIT_CSV, STATS_JSON]:
        if not path.exists():
            raise SystemExit(f"Missing KY input: {path}")
    if not shim.KY_RASTER_DIR.exists():
        raise SystemExit(f"Missing raster dir: {shim.KY_RASTER_DIR} (set KY_RASTER_DIR)")

    if args.smoke:
        args.tag = "smoke"
        args.seeds = args.seeds[:1]
        args.ssl_tasks = args.ssl_tasks[:1]
        args.encoder_lrs = [0.0, 1e-4]
        args.folds = [0]
        args.include_scratch = True
        args.ssl_epochs, args.ssl_patience = 2, 2
        args.max_epochs, args.patience = 3, 3
        labeled_csv, split_csv, unlabeled_csv = _make_smoke_subsets()
        run_ssl_stage(args, labeled_csv=labeled_csv, unlabeled_csv=unlabeled_csv)
        run_downstream_stage(args, labeled_csv=labeled_csv, split_csv=split_csv)
        print("SMOKE_PASSED", flush=True)
        return

    if args.tag is None:
        raise SystemExit("--tag is required for non-smoke stages")
    if args.stage in ("ssl", "all") and args.ssl_tasks:
        run_ssl_stage(args, labeled_csv=LABELED_CSV, unlabeled_csv=args.unlabeled_index)
    if args.stage in ("downstream", "all"):
        run_downstream_stage(args, labeled_csv=LABELED_CSV, split_csv=SPLIT_CSV)


if __name__ == "__main__":
    main()
