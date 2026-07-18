"""Training and evaluation helpers for supervised fine-tuning baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.metrics import compute_binary_metrics
from src.utils import load_checkpoint, save_checkpoint


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    grad_clip_norm: float | None = None,
) -> dict[str, object]:
    """Train for one epoch and return loss/predictions/metrics."""

    model.train()
    total_loss = 0.0
    y_true_parts = []
    y_logit_parts = []

    for X, y, *_metadata in dataloader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += float(loss.item()) * X.size(0)
        y_true_parts.append(y.detach().cpu().numpy())
        y_logit_parts.append(logits.detach().cpu().numpy())

    y_true = np.concatenate(y_true_parts)
    y_logits = np.concatenate(y_logit_parts)
    y_probs = 1.0 / (1.0 + np.exp(-y_logits))
    metrics = compute_binary_metrics(y_true, y_probs, threshold=0.5)
    return {
        "loss": total_loss / len(dataloader.dataset),
        "y_true": y_true,
        "y_logits": y_logits,
        "y_probs": y_probs,
        "y_pred_05": (y_probs >= 0.5).astype(int),
        "metrics": metrics,
    }


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device) -> dict[str, object]:
    """Evaluate model and return loss, predictions, probabilities, and metrics."""

    model.eval()
    total_loss = 0.0
    y_true_parts = []
    y_logit_parts = []
    metadata_parts = []
    for batch in dataloader:
        X, y = batch[0], batch[1]
        if len(batch) > 2:
            metadata_parts.append(batch[2])
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * X.size(0)
        y_true_parts.append(y.detach().cpu().numpy())
        y_logit_parts.append(logits.detach().cpu().numpy())

    y_true = np.concatenate(y_true_parts)
    y_logits = np.concatenate(y_logit_parts)
    y_probs = 1.0 / (1.0 + np.exp(-y_logits))
    metrics = compute_binary_metrics(y_true, y_probs, threshold=0.5)
    return {
        "loss": total_loss / len(dataloader.dataset),
        "y_true": y_true,
        "y_logits": y_logits,
        "y_probs": y_probs,
        "y_pred_05": (y_probs >= 0.5).astype(int),
        "metrics": metrics,
        "metadata": metadata_parts,
    }


def _monitor_value(eval_result: dict[str, object], monitor_metric: str) -> float:
    if monitor_metric == "val_auc":
        value = float(eval_result["metrics"]["auc"])
        if np.isfinite(value):
            return value
        return -float(eval_result["loss"])
    if monitor_metric == "val_loss":
        return -float(eval_result["loss"])
    raise ValueError(f"Unknown monitor_metric={monitor_metric!r}")


def train_scv_fold(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    checkpoint_path: str | Path,
    max_epochs: int = 100,
    early_stopping_patience: int = 15,
    monitor_metric: str = "val_auc",
    grad_clip_norm: float | None = 5.0,
    checkpoint_extra: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train one SCV fold with early stopping and best checkpoint saving."""

    checkpoint_path = Path(checkpoint_path).resolve()
    best_score = -float("inf")
    best_epoch = -1
    best_payload: dict[str, object] = {}
    patience_counter = 0
    rows = []

    for epoch in range(1, max_epochs + 1):
        train_result = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=grad_clip_norm,
        )
        val_result = evaluate_model(model, val_loader, criterion, device)
        learning_rate = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "train_loss": float(train_result["loss"]),
            "val_loss": float(val_result["loss"]),
            "train_auc": float(train_result["metrics"]["auc"]),
            "val_auc": float(val_result["metrics"]["auc"]),
            "train_f1": float(train_result["metrics"]["f1"]),
            "val_f1": float(val_result["metrics"]["f1"]),
            "learning_rate": learning_rate,
        }
        rows.append(row)

        score = _monitor_value(val_result, monitor_metric)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_payload = {
                "best_epoch": best_epoch,
                "best_val_auc": float(val_result["metrics"]["auc"]),
                "best_val_loss": float(val_result["loss"]),
                "monitor_metric": monitor_metric,
                "monitor_score": score,
            }
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=best_payload,
                extra=checkpoint_extra or {},
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    checkpoint = load_checkpoint(checkpoint_path, model, optimizer=None, map_location=device)
    best_payload.update(checkpoint.get("metrics", {}))
    return pd.DataFrame(rows), best_payload


def run_pretrained_resnet18_scv_experiment(
    *,
    project_root: str | Path,
    model_name: str,
    pretraining: str,
    patch_index_csv: str | Path,
    raster_dir: str | Path,
    encoder_checkpoint_path: str | Path,
    output_root: str | Path,
    figure_root: str | Path,
    checkpoint_dir: str | Path,
    patch_size: int = 32,
    random_seed: int = 42,
    val_fraction: float = 0.2,
    train_label_fraction: float = 1.0,
    cv_mode: str = "scv",  # "scv" = spatial CV by cluster_id; "flat" = StratifiedKFold(5, shuffle) by label
    batch_size: int = 16,
    encoder_learning_rate: float = 1e-5,
    head_learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    early_stopping_patience: int = 15,
    gradient_clip_norm: float = 5.0,
    dropout: float = 0.4,
    reference_metrics: dict[str, dict[str, float]] | None = None,
    expected_checkpoint_keys: set[str] | None = None,
    checkpoint_extra_values: dict[str, object] | None = None,
    scientific_note: str | None = None,
    num_workers: int = 0,
    device=None,
    plot_figures: bool = True,
    cache_in_memory: bool = False,
    with_mask: bool = True,
) -> dict[str, object]:
    """Run supervised 5-fold SCV fine-tuning from an SSL-pretrained encoder.

    ``with_mask`` controls whether the valid-context mask channel is included
    (in_channels = terrain + 1) or ablated (in_channels = terrain).
    """

    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import DataLoader, Subset

    from src.metrics import compute_binary_metrics, find_best_f1_threshold
    from src.models_resnet18 import (
        create_resnet18_binary_classifier,
        load_ssl_encoder_weights_into_classifier,
    )
    from src.patch_dataset import RasterPatchDataset, list_raster_files
    from src.plotting import (
        plot_pr_curves_all_folds,
        plot_roc_curves_all_folds,
        plot_training_curves,
    )
    from src.utils import count_trainable_parameters, ensure_dir, load_checkpoint

    project_root = Path(project_root).resolve()
    patch_index_csv = Path(patch_index_csv).resolve()
    raster_dir = Path(raster_dir).resolve()
    encoder_checkpoint_path = Path(encoder_checkpoint_path).resolve()
    output_root = Path(output_root).resolve()
    figure_root = Path(figure_root).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    prediction_dir = output_root / "predictions"
    metrics_dir = output_root / "metrics"
    training_log_dir = output_root / "training_logs"
    roc_dir = figure_root / "roc_curves"
    pr_dir = figure_root / "pr_curves"
    training_curve_dir = figure_root / "training_curves"
    for directory in [
        prediction_dir,
        metrics_dir,
        training_log_dir,
        roc_dir,
        pr_dir,
        training_curve_dir,
        checkpoint_dir,
    ]:
        ensure_dir(directory)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = bool(torch.cuda.is_available())

    patch_index = pd.read_csv(patch_index_csv).reset_index(drop=True)
    raster_files = list_raster_files(raster_dir)
    model_in_channels = len(raster_files) + (1 if with_mask else 0)
    checkpoint = torch.load(encoder_checkpoint_path, map_location="cpu")
    if expected_checkpoint_keys:
        missing_keys = expected_checkpoint_keys - set(checkpoint.keys())
        if missing_keys:
            raise KeyError(
                f"{encoder_checkpoint_path} missing checkpoint keys: {sorted(missing_keys)}"
            )

    if not patch_index["valid_patch"].astype(bool).all():
        raise ValueError("Patch index contains invalid patches.")
    # Boundary-aware padding: nonzero nodata_ratio is allowed (padded pixels are
    # flagged by the appended valid-context mask channel), so it is not checked.
    if set(patch_index["label"].astype(int).unique().tolist()) != {0, 1}:
        raise ValueError("Labels must be binary 0/1.")
    if cv_mode not in ("scv", "flat"):
        raise ValueError(f"cv_mode must be 'scv' or 'flat', got {cv_mode!r}.")
    if cv_mode == "scv" and sorted(patch_index["cluster_id"].astype(int).unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("cluster_id values must be 0..4 for SCV.")

    # Fold assignment: SCV holds out whole spatial clusters; FLAT uses random
    # stratified 5-fold on the label (Qianyi's protocol: StratifiedKFold n=5,
    # shuffle=True, random_state=seed) — appropriate when positives are spatially
    # clustered so cluster-holdout folds become pathologically imbalanced.
    n_folds = 5
    if cv_mode == "scv":
        fold_of = patch_index["cluster_id"].astype(int).to_numpy()
    else:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        fold_of = np.full(len(patch_index), -1, dtype=int)
        for fid, (_tr, te) in enumerate(skf.split(patch_index.index.to_numpy(),
                                                  patch_index["label"].astype(int).to_numpy())):
            fold_of[te] = fid
    print(f"CV mode: {cv_mode} ({n_folds} folds)", flush=True)

    preview_model = create_resnet18_binary_classifier(
        in_channels=model_in_channels,
        dropout=dropout,
        small_patch_stem=True,
        pretrained=False,
    )
    preview_load_info = load_ssl_encoder_weights_into_classifier(
        preview_model,
        encoder_checkpoint_path,
        strict_encoder=True,
    )
    if len(preview_load_info["loaded_keys"]) < 110:
        raise RuntimeError(
            f"Too few SSL encoder keys loaded: {len(preview_load_info['loaded_keys'])}."
        )
    del preview_model

    def compute_channel_stats_for_indices(indices, batch_size_for_stats=32):
        stats_dataset = RasterPatchDataset(
            patch_index_csv=patch_index_csv,
            raster_dir=raster_dir,
            patch_size=patch_size,
            nodata_value=-9999,
            normalize=False,
            return_metadata=False,
            valid_only=True,
        )
        loader = DataLoader(
            Subset(stats_dataset, list(indices)),
            batch_size=batch_size_for_stats,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        # X is (B, terrain + 1 mask): the last channel is the valid-context mask
        # and padded pixels are already zeroed in the terrain channels, so the
        # terrain stats are accumulated over valid pixels only (counted by mask).
        channel_sum = None
        channel_sumsq = None
        pixel_count = 0.0
        for X, _y in loader:
            X = X.float()
            terrain = X[:, :-1]
            ctx_mask = X[:, -1:]
            if channel_sum is None:
                channel_sum = terrain.sum(dim=(0, 2, 3))
                channel_sumsq = (terrain**2).sum(dim=(0, 2, 3))
            else:
                channel_sum += terrain.sum(dim=(0, 2, 3))
                channel_sumsq += (terrain**2).sum(dim=(0, 2, 3))
            pixel_count += float(ctx_mask.sum().item())
        stats_dataset.close()
        means = (channel_sum / pixel_count).numpy().astype("float32")
        variances = (channel_sumsq / pixel_count).numpy() - means**2
        stds = np.sqrt(np.maximum(variances, 1e-12)).astype("float32")
        return means, stds

    def make_dataset(channel_means, channel_stds):
        return RasterPatchDataset(
            patch_index_csv=patch_index_csv,
            raster_dir=raster_dir,
            patch_size=patch_size,
            nodata_value=-9999,
            normalize=True,
            channel_means=channel_means,
            channel_stds=channel_stds,
            return_metadata=True,
            valid_only=True,
            cache_in_memory=cache_in_memory,
            with_mask=with_mask,
        )

    def build_optimizer(model):
        encoder_params = []
        head_params = []
        for name, parameter in model.named_parameters():
            if name.startswith("fc."):
                head_params.append(parameter)
            else:
                encoder_params.append(parameter)
        return torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": encoder_learning_rate},
                {"params": head_params, "lr": head_learning_rate},
            ],
            weight_decay=weight_decay,
        )

    fold_predictions = []
    training_logs = []
    fold_metric_rows = []
    loaded_keys_by_fold = {}

    for fold_id in range(n_folds):
        print("\n" + "=" * 72)
        held = f"cluster {fold_id}" if cv_mode == "scv" else f"flat fold {fold_id}"
        print(f"Fold {fold_id} ({cv_mode}): held-out test = {held}")
        test_mask = fold_of == fold_id
        train_candidate_indices = patch_index.index[~test_mask].to_numpy()
        test_indices = patch_index.index[test_mask].to_numpy()
        train_candidate_labels = patch_index.loc[train_candidate_indices, "label"].to_numpy()
        train_indices, val_indices = train_test_split(
            train_candidate_indices,
            test_size=val_fraction,
            random_state=random_seed + fold_id,
            stratify=train_candidate_labels,
        )
        train_indices = np.asarray(train_indices, dtype=int)
        val_indices = np.asarray(val_indices, dtype=int)
        test_indices = np.asarray(test_indices, dtype=int)

        # Label-efficiency: keep val/test full, subsample only the TRAINING labels
        # (stratified by class, seeded). fraction<1.0 simulates label scarcity.
        if train_label_fraction < 1.0:
            rng = np.random.default_rng(random_seed + fold_id)
            tr_labels = patch_index.loc[train_indices, "label"].to_numpy()
            kept = []
            for cls in np.unique(tr_labels):
                cls_idx = train_indices[tr_labels == cls]
                n_keep = min(len(cls_idx), max(2, int(round(len(cls_idx) * train_label_fraction))))
                kept.append(rng.choice(cls_idx, size=n_keep, replace=False))
            train_indices = np.sort(np.concatenate(kept)).astype(int)
            _bc = np.bincount(patch_index.loc[train_indices, "label"].to_numpy().astype(int))
            print(f"  [label-efficiency] fraction={train_label_fraction}: "
                  f"train subsampled to n={len(train_indices)} (per-class {_bc.tolist()})")

        split_ids = [
            set(patch_index.loc[indices, "sample_id"].astype(str))
            for indices in [train_indices, val_indices, test_indices]
        ]
        if split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2]:
            raise ValueError(f"Fold {fold_id} has overlapping sample IDs.")
        if cv_mode == "scv":
            if fold_id in set(patch_index.loc[train_indices, "cluster_id"].astype(int)):
                raise ValueError(f"Fold {fold_id}: test cluster leaked into training.")
            if fold_id in set(patch_index.loc[val_indices, "cluster_id"].astype(int)):
                raise ValueError(f"Fold {fold_id}: test cluster leaked into validation.")
        for split_name, split_indices in [
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices),
        ]:
            split = patch_index.loc[split_indices]
            if not split["valid_patch"].astype(bool).all():
                raise ValueError(f"Fold {fold_id} {split_name} contains invalid patches.")
            # nonzero nodata_ratio is allowed under boundary-aware padding.

        print("train clusters:", sorted(patch_index.loc[train_indices, "cluster_id"].unique().tolist()))
        print("validation size:", len(val_indices))
        print("test cluster:", fold_id)
        print("train label counts:")
        print(patch_index.loc[train_indices, "label"].value_counts().sort_index().to_string())
        print("val label counts:")
        print(patch_index.loc[val_indices, "label"].value_counts().sort_index().to_string())
        print("test label counts:")
        print(patch_index.loc[test_indices, "label"].value_counts().sort_index().to_string())

        channel_means, channel_stds = compute_channel_stats_for_indices(train_indices)
        print("channel means and stds shape:", channel_means.shape, channel_stds.shape)
        fold_dataset = make_dataset(channel_means, channel_stds)
        train_loader = DataLoader(
            Subset(fold_dataset, train_indices.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            Subset(fold_dataset, val_indices.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            Subset(fold_dataset, test_indices.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        model = create_resnet18_binary_classifier(
            in_channels=model_in_channels,
            dropout=dropout,
            small_patch_stem=True,
            pretrained=False,
        )
        load_info = load_ssl_encoder_weights_into_classifier(
            model,
            encoder_checkpoint_path,
            strict_encoder=True,
        )
        loaded_keys_by_fold[fold_id] = len(load_info["loaded_keys"])
        if loaded_keys_by_fold[fold_id] < 110:
            raise RuntimeError(f"Fold {fold_id}: too few encoder keys loaded.")
        model = model.to(device)
        print("model parameter count:", sum(p.numel() for p in model.parameters()))
        print("number of trainable parameters:", count_trainable_parameters(model))
        print("number of loaded pretrained encoder keys:", loaded_keys_by_fold[fold_id])

        smoke_batch = next(iter(train_loader))
        smoke_X = smoke_batch[0].to(device, non_blocking=True)
        smoke_y = smoke_batch[1].to(device, non_blocking=True)
        print("device sanity model:", next(model.parameters()).device)
        print("device sanity X:", smoke_X.device)
        print("device sanity y:", smoke_y.device)
        if torch.cuda.is_available():
            assert next(model.parameters()).is_cuda and smoke_X.is_cuda and smoke_y.is_cuda

        optimizer = build_optimizer(model)
        criterion = nn.BCEWithLogitsLoss()
        checkpoint_path = checkpoint_dir / f"{model_name}_ps{patch_size}_fold{fold_id}_best.pt"
        extra_values = dict(checkpoint_extra_values or {})
        checkpoint_extra = {
            "fold": fold_id,
            "patch_size": patch_size,
            "model_name": model_name,
            "pretraining_type": pretraining,
            "encoder_checkpoint_path": str(encoder_checkpoint_path),
            **extra_values,
            "channel_means": channel_means,
            "channel_stds": channel_stds,
            "config": {
                "batch_size": batch_size,
                "encoder_learning_rate": encoder_learning_rate,
                "head_learning_rate": head_learning_rate,
                "weight_decay": weight_decay,
                "max_epochs": max_epochs,
                "early_stopping_patience": early_stopping_patience,
                "gradient_clip_norm": gradient_clip_norm,
                "dropout": dropout,
                "random_seed": random_seed,
                "val_fraction": val_fraction,
            },
        }

        training_log, best_payload = train_scv_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            checkpoint_path=checkpoint_path,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            monitor_metric="val_auc",
            grad_clip_norm=gradient_clip_norm,
            checkpoint_extra=checkpoint_extra,
        )
        training_log["fold"] = fold_id
        training_log["encoder_learning_rate"] = encoder_learning_rate
        training_log["head_learning_rate"] = head_learning_rate
        log_path = training_log_dir / f"{model_name}_ps{patch_size}_fold{fold_id}_training_log.csv"
        training_log.to_csv(log_path, index=False)
        training_logs.append(training_log)

        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_payload.update(
            {
                "best_val_auc": best_payload.get("best_val_auc"),
                "best_val_loss": best_payload.get("best_val_loss"),
                "fold": fold_id,
                "patch_size": patch_size,
                "model_name": model_name,
                "pretraining_type": pretraining,
                "encoder_checkpoint_path": str(encoder_checkpoint_path),
                **extra_values,
                "channel_means": channel_means,
                "channel_stds": channel_stds,
                "config": checkpoint_extra["config"],
            }
        )
        torch.save(checkpoint_payload, checkpoint_path)

        load_checkpoint(checkpoint_path, model, optimizer=None, map_location=device)
        val_result = evaluate_model(model, val_loader, criterion, device)
        best_threshold_f1, _ = find_best_f1_threshold(
            val_result["y_true"],
            val_result["y_probs"],
        )
        test_result = evaluate_model(model, test_loader, criterion, device)
        test_metrics_05 = compute_binary_metrics(
            test_result["y_true"],
            test_result["y_probs"],
            threshold=0.5,
        )
        test_metrics_best = compute_binary_metrics(
            test_result["y_true"],
            test_result["y_probs"],
            threshold=best_threshold_f1,
        )

        test_meta = patch_index.loc[
            test_indices,
            ["sample_id", "x", "y", "label", "source", "cluster_id"],
        ].reset_index(drop=True)
        pred_df = test_meta.copy()
        pred_df["fold"] = fold_id
        pred_df["y_true"] = test_result["y_true"].astype(int)
        pred_df["y_logit"] = test_result["y_logits"].astype(float)
        pred_df["y_prob"] = test_result["y_probs"].astype(float)
        pred_df["y_pred_05"] = (test_result["y_probs"] >= 0.5).astype(int)
        pred_df["y_pred_best_f1"] = (test_result["y_probs"] >= best_threshold_f1).astype(int)
        pred_df["split"] = "test"
        pred_df = pred_df[
            [
                "sample_id",
                "x",
                "y",
                "label",
                "source",
                "cluster_id",
                "fold",
                "y_true",
                "y_logit",
                "y_prob",
                "y_pred_05",
                "y_pred_best_f1",
                "split",
            ]
        ]
        pred_path = prediction_dir / f"{model_name}_ps{patch_size}_fold{fold_id}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        fold_predictions.append(pred_df)
        fold_metric_rows.append(
            {
                "model": model_name,
                "pretraining": pretraining,
                "patch_size": patch_size,
                "fold": fold_id,
                "n_train": len(train_indices),
                "n_val": len(val_indices),
                "n_test": len(test_indices),
                "n_test_pos": int((test_result["y_true"] == 1).sum()),
                "n_test_neg": int((test_result["y_true"] == 0).sum()),
                "auc": test_metrics_05["auc"],
                "pr_auc": test_metrics_05["pr_auc"],
                "accuracy_05": test_metrics_05["accuracy"],
                "precision_05": test_metrics_05["precision"],
                "recall_05": test_metrics_05["recall"],
                "f1_05": test_metrics_05["f1"],
                "best_threshold_f1": best_threshold_f1,
                "accuracy_best_f1": test_metrics_best["accuracy"],
                "precision_best_f1": test_metrics_best["precision"],
                "recall_best_f1": test_metrics_best["recall"],
                "f1_best_f1": test_metrics_best["f1"],
                "tn": test_metrics_best["tn"],
                "fp": test_metrics_best["fp"],
                "fn": test_metrics_best["fn"],
                "tp": test_metrics_best["tp"],
                "best_epoch": best_payload.get("best_epoch"),
                "best_val_auc": best_payload.get("best_val_auc"),
                "best_val_loss": best_payload.get("best_val_loss"),
            }
        )
        print(
            f"Fold {fold_id} test AUC: {test_metrics_05['auc']:.6f}; "
            f"best threshold: {best_threshold_f1:.6f}"
        )
        fold_dataset.close()

    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics_path = metrics_dir / f"{model_name}_ps{patch_size}_fold_metrics.csv"
    fold_metrics.to_csv(fold_metrics_path, index=False)

    summary_metrics = {}
    for metric in [
        "auc",
        "pr_auc",
        "accuracy_05",
        "precision_05",
        "recall_05",
        "f1_05",
        "accuracy_best_f1",
        "precision_best_f1",
        "recall_best_f1",
        "f1_best_f1",
    ]:
        summary_metrics[f"mean_{metric}"] = float(fold_metrics[metric].mean())
        summary_metrics[f"std_{metric}"] = float(fold_metrics[metric].std(ddof=1))

    worst_fold_index = int(fold_metrics["auc"].idxmin())
    best_fold_index = int(fold_metrics["auc"].idxmax())
    summary_metrics.update(
        {
            "model": model_name,
            "pretraining": pretraining,
            "patch_size": patch_size,
            "worst_fold_auc": float(fold_metrics.loc[worst_fold_index, "auc"]),
            "best_fold_auc": float(fold_metrics.loc[best_fold_index, "auc"]),
            "worst_fold": int(fold_metrics.loc[worst_fold_index, "fold"]),
            "best_fold": int(fold_metrics.loc[best_fold_index, "fold"]),
            **extra_values,
        }
    )
    for ref_name, ref in (reference_metrics or {}).items():
        summary_metrics[f"{ref_name}_mean_auc"] = ref["mean_auc"]
        summary_metrics[f"{ref_name}_std_auc"] = ref["std_auc"]
        summary_metrics[f"{ref_name}_worst_fold_auc"] = ref["worst_fold_auc"]
        summary_metrics[f"delta_mean_auc_vs_{ref_name}"] = (
            summary_metrics["mean_auc"] - ref["mean_auc"]
        )
        summary_metrics[f"delta_std_auc_vs_{ref_name}"] = (
            summary_metrics["std_auc"] - ref["std_auc"]
        )
        summary_metrics[f"delta_worst_auc_vs_{ref_name}"] = (
            summary_metrics["worst_fold_auc"] - ref["worst_fold_auc"]
        )

    summary_df = pd.DataFrame([summary_metrics])
    summary_metrics_path = metrics_dir / f"{model_name}_ps{patch_size}_summary_metrics.csv"
    summary_df.to_csv(summary_metrics_path, index=False)

    if plot_figures:
        plot_roc_curves_all_folds(
            fold_predictions,
            roc_dir / f"{model_name}_ps{patch_size}_roc_all_folds.png",
        )
        plot_roc_curves_all_folds(
            fold_predictions,
            roc_dir / f"{model_name}_ps{patch_size}_roc_all_folds.pdf",
        )
        plot_pr_curves_all_folds(
            fold_predictions,
            pr_dir / f"{model_name}_ps{patch_size}_pr_all_folds.png",
        )
        plot_pr_curves_all_folds(
            fold_predictions,
            pr_dir / f"{model_name}_ps{patch_size}_pr_all_folds.pdf",
        )
        plot_training_curves(
            training_logs,
            training_curve_dir / f"{model_name}_ps{patch_size}_training_curves.png",
        )
        plot_training_curves(
            training_logs,
            training_curve_dir / f"{model_name}_ps{patch_size}_training_curves.pdf",
        )

    if scientific_note:
        print(scientific_note)

    return {
        "fold_metrics": fold_metrics,
        "summary_metrics": summary_df,
        "fold_predictions": fold_predictions,
        "training_logs": training_logs,
        "paths": {
            "prediction_dir": prediction_dir,
            "metrics_dir": metrics_dir,
            "training_log_dir": training_log_dir,
            "checkpoint_dir": checkpoint_dir,
            "figure_root": figure_root,
            "fold_metrics": fold_metrics_path,
            "summary_metrics": summary_metrics_path,
        },
        "loaded_keys_by_fold": loaded_keys_by_fold,
        "checkpoint": checkpoint,
        "raster_files": raster_files,
        "patch_index": patch_index,
    }
