"""Binary classification metrics for landslide susceptibility experiments."""

from __future__ import annotations

import numpy as np


def _as_numpy(values) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def safe_roc_auc(y_true, y_prob) -> float:
    """Compute ROC AUC, returning NaN when only one class is present."""

    y_true = _as_numpy(y_true).astype(int)
    y_prob = _as_numpy(y_prob).astype(float)
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    sorted_scores = y_prob[order]
    start = 0
    while start < len(y_prob):
        stop = start + 1
        while stop < len(y_prob) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = 0.5 * (start + 1 + stop)
        ranks[order[start:stop]] = average_rank
        start = stop
    rank_sum_pos = ranks[positives].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def safe_average_precision(y_true, y_prob) -> float:
    """Compute average precision, returning NaN when no positives exist."""

    y_true = _as_numpy(y_true).astype(int)
    y_prob = _as_numpy(y_prob).astype(float)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_prob)
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    precision = tp / (np.arange(len(sorted_true)) + 1)
    return float((precision * (sorted_true == 1)).sum() / n_pos)


def confusion_matrix_metrics(y_true, y_prob, threshold: float):
    """Return TN, FP, FN, TP for a probability threshold."""

    y_true = _as_numpy(y_true).astype(int)
    y_pred = (_as_numpy(y_prob).astype(float) >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Compute core binary metrics at a fixed threshold."""

    tn, fp, fn, tp = confusion_matrix_metrics(y_true, y_prob, threshold)
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": safe_average_precision(y_true, y_prob),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def find_best_f1_threshold(y_true, y_prob) -> tuple[float, float]:
    """Find threshold with maximum F1 using supplied validation predictions."""

    y_true = _as_numpy(y_true).astype(int)
    y_prob = _as_numpy(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")

    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_prob)))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        f1 = compute_binary_metrics(y_true, y_prob, float(threshold))["f1"]
        if f1 > best_f1 or (np.isclose(f1, best_f1) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)
