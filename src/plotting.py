"""Matplotlib plotting helpers for fold-level baseline results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.metrics import safe_average_precision, safe_roc_auc


def set_publication_plot_style(font_family: str = "Arial", font_size: int = 10) -> str:
    """Set a consistent publication-style Matplotlib theme.

    Returns the font family that was actually applied. If the requested font is
    unavailable, falls back to DejaVu Sans while preserving the requested size.
    """

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    applied_font = font_family
    if font_family not in available_fonts:
        applied_font = "Liberation Sans" if "Liberation Sans" in available_fonts else "DejaVu Sans"
        print(
            f"WARNING: font family {font_family!r} is not available. "
            f"Falling back to {applied_font!r}."
        )

    plt.rcParams.update(
        {
            "font.family": applied_font,
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return applied_font


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _roc_curve_points(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thresholds = np.r_[np.inf, np.sort(np.unique(y_prob))[::-1], -np.inf]
    tpr = []
    fpr = []
    positives = max(int((y_true == 1).sum()), 1)
    negatives = max(int((y_true == 0).sum()), 1)
    for threshold in thresholds:
        pred = y_prob >= threshold
        tp = ((y_true == 1) & pred).sum()
        fp = ((y_true == 0) & pred).sum()
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
    return np.asarray(fpr), np.asarray(tpr)


def _pr_curve_points(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thresholds = np.r_[np.inf, np.sort(np.unique(y_prob))[::-1], -np.inf]
    precision = []
    recall = []
    positives = max(int((y_true == 1).sum()), 1)
    for threshold in thresholds:
        pred = y_prob >= threshold
        tp = ((y_true == 1) & pred).sum()
        fp = ((y_true == 0) & pred).sum()
        precision.append(tp / (tp + fp) if (tp + fp) else 1.0)
        recall.append(tp / positives)
    return np.asarray(recall), np.asarray(precision)


def plot_roc_curves_all_folds(fold_predictions: list[pd.DataFrame], output_path: str | Path) -> Path:
    """Plot ROC curves for all SCV folds."""

    output_path = _ensure_parent(output_path)
    plt.figure(figsize=(6, 5))
    aucs = []
    for pred in fold_predictions:
        fold = int(pred["fold"].iloc[0])
        y_true = pred["y_true"].to_numpy()
        y_prob = pred["y_prob"].to_numpy()
        auc = safe_roc_auc(y_true, y_prob)
        aucs.append(auc)
        fpr, tpr = _roc_curve_points(y_true, y_prob)
        plt.plot(fpr, tpr, linewidth=1.5, label=f"Fold {fold} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curves, mean AUC={np.nanmean(aucs):.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close()
    return output_path


def plot_pr_curves_all_folds(fold_predictions: list[pd.DataFrame], output_path: str | Path) -> Path:
    """Plot precision-recall curves for all SCV folds."""

    output_path = _ensure_parent(output_path)
    plt.figure(figsize=(6, 5))
    aps = []
    for pred in fold_predictions:
        fold = int(pred["fold"].iloc[0])
        y_true = pred["y_true"].to_numpy()
        y_prob = pred["y_prob"].to_numpy()
        ap = safe_average_precision(y_true, y_prob)
        aps.append(ap)
        recall, precision = _pr_curve_points(y_true, y_prob)
        plt.plot(recall, precision, linewidth=1.5, label=f"Fold {fold} AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curves, mean AP={np.nanmean(aps):.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close()
    return output_path


def plot_training_curves(training_logs: list[pd.DataFrame], output_path: str | Path) -> Path:
    """Plot train/validation loss and AUC curves for all folds."""

    output_path = _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for log in training_logs:
        fold = int(log["fold"].iloc[0]) if "fold" in log.columns else len(axes[0].lines)
        axes[0].plot(log["epoch"], log["train_loss"], alpha=0.7, label=f"F{fold} train")
        axes[0].plot(log["epoch"], log["val_loss"], alpha=0.9, linestyle="--", label=f"F{fold} val")
        axes[1].plot(log["epoch"], log["train_auc"], alpha=0.7, label=f"F{fold} train")
        axes[1].plot(log["epoch"], log["val_auc"], alpha=0.9, linestyle="--", label=f"F{fold} val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Training and validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC AUC")
    axes[1].set_title("Training and validation AUC")
    for axis in axes:
        axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return output_path


def plot_ssl_loss_curve(training_log_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot SSL train/validation loss."""

    output_path = _ensure_parent(output_path)
    plt.figure(figsize=(6, 4.5))
    plt.plot(training_log_df["epoch"], training_log_df["train_loss"], label="Train loss")
    plt.plot(training_log_df["epoch"], training_log_df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("SSL loss")
    plt.title("SSL training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close()
    return output_path


def plot_masked_reconstruction_examples(
    X,
    mask,
    X_recon,
    output_path: str | Path,
    channel_indices: list[int] | None = None,
    channel_names: list[str] | None = None,
) -> Path:
    """Plot original, masked, reconstructed, and absolute error examples."""

    output_path = _ensure_parent(output_path)
    channel_indices = channel_indices or [0, 1, 2]
    channel_names = channel_names or [f"factor_{index + 1:02d}" for index in channel_indices]

    X = X.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    X_recon = X_recon.detach().cpu().numpy()
    X_masked = X * (1.0 - mask)
    error = np.abs(X_recon - X)

    fig, axes = plt.subplots(len(channel_indices), 4, figsize=(11, 8))
    for row_index, channel_index in enumerate(channel_indices):
        panels = [
            (X[0, channel_index], "Original"),
            (X_masked[0, channel_index], "Masked"),
            (X_recon[0, channel_index], "Reconstruction"),
            (error[0, channel_index], "Abs. error"),
        ]
        for col_index, (image, title) in enumerate(panels):
            ax = axes[row_index, col_index]
            im = ax.imshow(image, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(title)
            if col_index == 0:
                ax.set_ylabel(channel_names[row_index])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return output_path


def plot_contrastive_augmentation_examples(
    X,
    view1,
    view2,
    output_path: str | Path,
    channel_indices: list[int] | None = None,
    channel_names: list[str] | None = None,
) -> Path:
    """Plot original patch and two contrastive augmented views."""

    output_path = _ensure_parent(output_path)
    channel_indices = channel_indices or [0, 1, 2]
    channel_names = channel_names or [f"factor_{index + 1:02d}" for index in channel_indices]

    X = X.detach().cpu().numpy()
    view1 = view1.detach().cpu().numpy()
    view2 = view2.detach().cpu().numpy()
    difference = np.abs(view1 - view2)

    fig, axes = plt.subplots(len(channel_indices), 4, figsize=(11, 8))
    for row_index, channel_index in enumerate(channel_indices):
        panels = [
            (X[0, channel_index], "Original"),
            (view1[0, channel_index], "View 1"),
            (view2[0, channel_index], "View 2"),
            (difference[0, channel_index], "|View 1 - View 2|"),
        ]
        for col_index, (image, title) in enumerate(panels):
            ax = axes[row_index, col_index]
            im = ax.imshow(image, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(title)
            if col_index == 0:
                ax.set_ylabel(channel_names[row_index])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return output_path


def plot_jigsaw_training_curves(training_log_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot Jigsaw SSL loss and permutation accuracy curves."""

    output_path = _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].plot(training_log_df["epoch"], training_log_df["train_loss"], label="Train loss")
    axes[0].plot(training_log_df["epoch"], training_log_df["val_loss"], label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("(a) Jigsaw loss")
    axes[0].legend(frameon=False)

    axes[1].plot(training_log_df["epoch"], training_log_df["train_acc_top1"], label="Train top-1")
    axes[1].plot(training_log_df["epoch"], training_log_df["val_acc_top1"], label="Validation top-1")
    axes[1].plot(training_log_df["epoch"], training_log_df["train_acc_top5"], linestyle="--", label="Train top-5")
    axes[1].plot(training_log_df["epoch"], training_log_df["val_acc_top5"], linestyle="--", label="Validation top-5")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("(b) Permutation accuracy")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return output_path


def plot_jigsaw_examples(
    X,
    X_jigsaw,
    output_path: str | Path,
    perm_class: int,
    grid_size: int = 4,
    channel_indices: list[int] | None = None,
    channel_names: list[str] | None = None,
) -> Path:
    """Plot original and shuffled Jigsaw patches for representative channels."""

    output_path = _ensure_parent(output_path)
    channel_indices = channel_indices or [0, 1, 2]
    channel_names = channel_names or [f"factor_{index + 1:02d}" for index in channel_indices]
    X = X.detach().cpu().numpy()
    X_jigsaw = X_jigsaw.detach().cpu().numpy()

    fig, axes = plt.subplots(len(channel_indices), 2, figsize=(6.2, 8.0), constrained_layout=True)
    for row_index, channel_index in enumerate(channel_indices):
        panels = [
            (X[0, channel_index], "Original"),
            (X_jigsaw[0, channel_index], f"Shuffled, class {perm_class}"),
        ]
        for col_index, (image, title) in enumerate(panels):
            ax = axes[row_index, col_index]
            im = ax.imshow(image, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(title)
            if col_index == 0:
                ax.set_ylabel(channel_names[row_index])
            height, width = image.shape
            tile_h = height / grid_size
            tile_w = width / grid_size
            for grid_line in range(1, grid_size):
                ax.axhline(grid_line * tile_h - 0.5, color="white", linewidth=0.6, alpha=0.85)
                ax.axvline(grid_line * tile_w - 0.5, color="white", linewidth=0.6, alpha=0.85)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return output_path


def plot_jigsaw_confusion_top_classes(
    y_true,
    y_pred,
    output_path: str | Path,
    n_classes_to_show: int = 20,
) -> Path:
    """Plot normalized confusion matrix for the most frequent Jigsaw classes."""

    output_path = _ensure_parent(output_path)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    classes, counts = np.unique(y_true, return_counts=True)
    selected = classes[np.argsort(-counts)[:n_classes_to_show]]
    selected = np.sort(selected)
    index = {class_id: idx for idx, class_id in enumerate(selected)}
    matrix = np.zeros((len(selected), len(selected)), dtype=float)
    for true_value, pred_value in zip(y_true, y_pred):
        if true_value in index and pred_value in index:
            matrix[index[true_value], index[pred_value]] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=max(float(normalized.max()), 1e-6))
    ax.set_xticks(np.arange(len(selected)))
    ax.set_yticks(np.arange(len(selected)))
    ax.set_xticklabels(selected, rotation=90)
    ax.set_yticklabels(selected)
    ax.set_xlabel("Predicted permutation class")
    ax.set_ylabel("Target permutation class")
    ax.set_title("Jigsaw validation confusion, selected classes")
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized frequency")
    fig.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return output_path


def plot_rotation_training_curves(
    training_log_df: pd.DataFrame,
    output_path: str | Path,
    random_loss: float = np.log(4),
    random_accuracy: float = 0.25,
) -> Path:
    """Plot rotation SSL loss and accuracy curves with random baselines."""

    output_path = _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].plot(training_log_df["epoch"], training_log_df["train_loss"], label="Train loss")
    axes[0].plot(training_log_df["epoch"], training_log_df["val_loss"], label="Validation loss")
    axes[0].axhline(random_loss, color="#777777", linestyle="--", linewidth=1.0, label="Random CE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("(a) Rotation loss")
    axes[0].legend(frameon=False)

    axes[1].plot(training_log_df["epoch"], training_log_df["train_acc_top1"], label="Train top-1")
    axes[1].plot(training_log_df["epoch"], training_log_df["val_acc_top1"], label="Validation top-1")
    axes[1].axhline(random_accuracy, color="#777777", linestyle="--", linewidth=1.0, label="Random acc.")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("(b) Rotation accuracy")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return output_path


def plot_rotation_examples(
    X,
    rotated_examples: list,
    output_path: str | Path,
    channel_indices: list[int] | None = None,
    channel_names: list[str] | None = None,
    rotation_labels: list[str] | None = None,
) -> Path:
    """Plot original and rotated examples for representative channels."""

    output_path = _ensure_parent(output_path)
    channel_indices = channel_indices or [0, 1, 2]
    channel_names = channel_names or [f"factor_{index + 1:02d}" for index in channel_indices]
    rotation_labels = rotation_labels or ["0°", "90°", "180°", "270°"]
    X = X.detach().cpu().numpy()
    rotated_arrays = [example.detach().cpu().numpy() for example in rotated_examples]

    fig, axes = plt.subplots(len(channel_indices), 5, figsize=(12.0, 7.6), constrained_layout=True)
    for row_index, channel_index in enumerate(channel_indices):
        panels = [(X[0, channel_index], "Original")]
        panels.extend(
            [(rotated[0, channel_index], label) for rotated, label in zip(rotated_arrays, rotation_labels)]
        )
        for col_index, (image, title) in enumerate(panels):
            ax = axes[row_index, col_index]
            im = ax.imshow(image, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(title)
            if col_index == 0:
                ax.set_ylabel(channel_names[row_index])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return output_path


def plot_rotation_confusion_matrix(
    y_true,
    y_pred,
    output_path: str | Path,
    class_labels: list[str] | None = None,
) -> Path:
    """Plot row-normalized 4x4 rotation confusion matrix."""

    output_path = _ensure_parent(output_path)
    class_labels = class_labels or ["0°", "90°", "180°", "270°"]
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n_classes = len(class_labels)
    matrix = np.zeros((n_classes, n_classes), dtype=float)
    for true_value, pred_value in zip(y_true, y_pred):
        if 0 <= true_value < n_classes and 0 <= pred_value < n_classes:
            matrix[true_value, pred_value] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(5.0, 4.5), constrained_layout=True)
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted rotation")
    ax.set_ylabel("True rotation")
    ax.set_title("Rotation validation confusion")
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if normalized[i, j] > 0.55 else "#1A1A1A"
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", color=color)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized frequency")
    fig.savefig(output_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return output_path
