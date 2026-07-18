#!/usr/bin/env python
"""Build notebooks/16_compare_all_seven_ssl_tasks_ps32.ipynb from notebook 15.

Clones the notebook-15 comparison and extends it to 7 methods: the 5 archived
methods are loaded from results_summary/tables (fold metrics only, no
predictions), and the 2 new MPS-trained methods (cross-channel masking, strip
jigsaw) are loaded from outputs/R2_new_ssl_tasks. Prediction-dependent steps
(validation cross-check, pooled ROC/PR) are gated on predictions_available so
the missing archived predictions are skipped with a caveat instead of raising.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "notebooks/15_compare_all_ssl_tasks_ps32.ipynb"

# R2_NAMESPACE selects which fine-tuning outputs the notebook reads/writes.
# "R2_new_ssl_tasks" = verification (smoke) run; "R2_new_ssl_tasks_full" = full run.
R2_NAMESPACE = os.environ.get("R2_NAMESPACE", "R2_new_ssl_tasks")
NB_SUFFIX = os.environ.get("NB16_SUFFIX", "")
# Optional one-line methodological caveat injected into the purpose and findings
# cells (e.g. to note that the two new methods used a reduced finetune LR).
FINETUNE_NOTE = os.environ.get("FINETUNE_NOTE", "")
DST = PROJECT_ROOT / f"notebooks/16_compare_all_seven_ssl_tasks_ps32{NB_SUFFIX}.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELL0 = """# Final all-task SSL comparison (seven methods), ps32

This notebook extends the five-method ps32 comparison to seven methods by adding two new self-supervised pretext tasks: cross-channel masking and 1D strip jigsaw. It reads archived fold metrics for the five original methods and the newly computed fold metrics and predictions for the two new methods, then writes final summary tables and figures. No model training, fine-tuning, checkpoint modification, or susceptibility mapping is performed here.

Hardware-mix caveat: the five original methods were pretrained and fine-tuned on CUDA (RTX 4090); the two new methods were pretrained on Apple Silicon (MPS). The downstream fine-tuning protocol and 5-cluster spatial cross-validation are identical, but pretraining hardware differs, so treat cross-hardware comparisons as indicative rather than exact."""

CELL1 = """## 1. Purpose and experiment configuration

The comparison includes Scratch, Masked reconstruction, Contrastive learning, Jigsaw, Rotation prediction, Cross-channel masking, and Strip jigsaw. The primary evaluation remains fold-wise spatial cross-validation metrics. Pooled ROC and precision-recall curves are included only for visualization and only for methods whose per-fold prediction files are available (the two new methods); the five archived methods provide fold metrics only.
__FINETUNE_NOTE__"""

CELL2 = '''from pathlib import Path

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
PATCH_SIZE = 32
FOLDS = [0, 1, 2, 3, 4]

# R2_NAMESPACE selects which fine-tuning outputs feed this comparison:
#   "R2_new_ssl_tasks"      = verification (smoke) run
#   "R2_new_ssl_tasks_full" = full 50-epoch pretrain / 100-epoch finetune run
R2_NAMESPACE = "__R2_NAMESPACE__"

ARCHIVED_FOLD_METRICS = PROJECT_ROOT / "results_summary/tables/all_ssl_tasks_ps32_fold_metrics.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / R2_NAMESPACE / "summary_all"
FIGURE_DIR = PROJECT_ROOT / "figures" / R2_NAMESPACE / "summary_all"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FOLD_METRICS_OUT = OUTPUT_DIR / "all_seven_ssl_tasks_ps32_fold_metrics.csv"
SUMMARY_METRICS_OUT = OUTPUT_DIR / "all_seven_ssl_tasks_ps32_summary_metrics.csv"
TRANSFER_SUMMARY_OUT = OUTPUT_DIR / "pretext_transfer_summary_ps32.csv"
MANUSCRIPT_TABLE_OUT = OUTPUT_DIR / "Table_R2_all_seven_ssl_tasks_ps32_manuscript.csv"

print("Project root:", PROJECT_ROOT)
print("R2 namespace:", R2_NAMESPACE)
print("Archived fold metrics:", ARCHIVED_FOLD_METRICS)
print("Output directory:", OUTPUT_DIR)
print("Figure directory:", FIGURE_DIR)'''

CELL6 = '''applied_font = set_publication_plot_style(font_family="Arial", font_size=10)
print("Applied plotting font:", applied_font)

# Archived methods: fold metrics come from the single archived CSV (no per-fold
# prediction files are shipped, so predictions_available is False). New methods:
# fold metrics and per-fold predictions come from the R2 fine-tuning outputs.
MODEL_CONFIGS = [
    {
        "model_display_name": "Scratch",
        "model_key": "scratch_resnet18",
        "pretraining": "none",
        "pretext_category": "None",
        "source": "archived",
        "predictions_available": False,
    },
    {
        "model_display_name": "Masked reconstruction",
        "model_key": "masked_recon_resnet18",
        "pretraining": "masked_recon",
        "pretext_category": "Reconstruction-based",
        "source": "archived",
        "predictions_available": False,
    },
    {
        "model_display_name": "Contrastive learning",
        "model_key": "contrastive_resnet18",
        "pretraining": "contrastive",
        "pretext_category": "Contrastive",
        "source": "archived",
        "predictions_available": False,
    },
    {
        "model_display_name": "Jigsaw",
        "model_key": "jigsaw_resnet18",
        "pretraining": "jigsaw",
        "pretext_category": "Spatial-ordering",
        "source": "archived",
        "predictions_available": False,
    },
    {
        "model_display_name": "Rotation prediction",
        "model_key": "rotation_resnet18",
        "pretraining": "rotation",
        "pretext_category": "Geometric",
        "source": "archived",
        "predictions_available": False,
    },
    {
        "model_display_name": "Cross-channel masking",
        "model_key": "cross_channel_resnet18",
        "pretraining": "cross_channel",
        "pretext_category": "Reconstruction-based",
        "source": "new",
        "predictions_available": True,
        "fold_metrics": PROJECT_ROOT / f"outputs/{R2_NAMESPACE}/cross_channel/metrics/cross_channel_resnet18_ps32_fold_metrics.csv",
        "prediction_template": PROJECT_ROOT / f"outputs/{R2_NAMESPACE}/cross_channel/predictions/cross_channel_resnet18_ps32_fold{{fold}}_predictions.csv",
    },
    {
        "model_display_name": "Strip jigsaw",
        "model_key": "strip_jigsaw_resnet18",
        "pretraining": "strip_jigsaw",
        "pretext_category": "Spatial-ordering",
        "source": "new",
        "predictions_available": True,
        "fold_metrics": PROJECT_ROOT / f"outputs/{R2_NAMESPACE}/strip_jigsaw/metrics/strip_jigsaw_resnet18_ps32_fold_metrics.csv",
        "prediction_template": PROJECT_ROOT / f"outputs/{R2_NAMESPACE}/strip_jigsaw/predictions/strip_jigsaw_resnet18_ps32_fold{{fold}}_predictions.csv",
    },
]

MODEL_ORDER = [config["model_display_name"] for config in MODEL_CONFIGS]
PALETTE = {
    "Scratch": "#4D4D4D",
    "Masked reconstruction": "#0072B2",
    "Contrastive learning": "#D55E00",
    "Jigsaw": "#009E73",
    "Rotation prediction": "#CC79A7",
    "Cross-channel masking": "#56B4E9",
    "Strip jigsaw": "#E69F00",
}
PREDICTION_METHODS = [c["model_display_name"] for c in MODEL_CONFIGS if c["predictions_available"]]
print("Model order:", MODEL_ORDER)
print("Methods with per-fold predictions:", PREDICTION_METHODS)'''

CELL7 = "## 4. Load fold metrics for all seven models"

CELL8 = '''config_columns = ["model_display_name", "model_key", "pretraining", "pretext_category"]

# Archived methods share one fold-metrics CSV that already carries the config columns.
if not ARCHIVED_FOLD_METRICS.exists():
    raise FileNotFoundError(f"Missing archived fold metrics file: {ARCHIVED_FOLD_METRICS}")
archived_fold_metrics = pd.read_csv(ARCHIVED_FOLD_METRICS)

fold_metric_frames = []
for config in MODEL_CONFIGS:
    name = config["model_display_name"]
    if config["source"] == "archived":
        df = archived_fold_metrics[archived_fold_metrics["model_display_name"] == name].copy()
        if df.empty:
            raise ValueError(f"Archived fold metrics has no rows for {name}.")
    else:
        path = config["fold_metrics"]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing fold metrics file for {name}: {path}. Run "
                "scripts/finetune_new_ssl_tasks.py first."
            )
        df = pd.read_csv(path)
    for column in config_columns:
        df[column] = config[column]
    fold_metric_frames.append(df)

combined_fold_metrics = pd.concat(fold_metric_frames, ignore_index=True)
combined_fold_metrics["patch_size"] = combined_fold_metrics["patch_size"].astype(int)
combined_fold_metrics.groupby("model_display_name")[["auc", "pr_auc"]].agg(["mean", "std"])'''

CELL9 = "## 5. Load fold prediction files (new methods only)"

CELL10 = '''# Only the two new methods ship per-fold prediction CSVs. The five archived
# methods provide fold metrics only, so pooled ROC/PR and the per-fold sample-ID
# cross-check are limited to the methods in PREDICTION_METHODS.
prediction_frames = []
missing_prediction_files = []
for config in MODEL_CONFIGS:
    if not config["predictions_available"]:
        continue
    for fold in FOLDS:
        path = Path(str(config["prediction_template"]).format(fold=fold))
        if not path.exists():
            missing_prediction_files.append(str(path))
            continue
        pred = pd.read_csv(path)
        for column in config_columns:
            pred[column] = config[column]
        prediction_frames.append(pred)

if missing_prediction_files:
    raise FileNotFoundError(
        "Missing prediction files for methods that should provide them:\\n"
        + "\\n".join(missing_prediction_files)
    )

if prediction_frames:
    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    print("Loaded prediction rows:", len(combined_predictions))
    display(combined_predictions.groupby(["model_display_name", "fold"]).size().unstack())
else:
    combined_predictions = pd.DataFrame()
    print("No per-fold prediction files available; pooled ROC/PR figures will be skipped.")

print("Archived methods without predictions (fold metrics only):",
      [c["model_display_name"] for c in MODEL_CONFIGS if not c["predictions_available"]])'''

CELL11 = "## 6. Validate metrics and predictions"

CELL12 = '''required_metric_columns = [
    "model_display_name", "model_key", "pretraining", "pretext_category", "patch_size", "fold",
    "n_train", "n_val", "n_test", "n_test_pos", "n_test_neg", "auc", "pr_auc", "accuracy_05",
    "precision_05", "recall_05", "f1_05", "best_threshold_f1", "accuracy_best_f1",
    "precision_best_f1", "recall_best_f1", "f1_best_f1", "tn", "fp", "fn", "tp",
    "best_epoch", "best_val_auc", "best_val_loss",
]
required_prediction_columns = [
    "sample_id", "x", "y", "label", "source", "cluster_id", "fold", "y_true", "y_logit",
    "y_prob", "y_pred_05", "y_pred_best_f1", "split",
]

missing_metric_cols = [col for col in required_metric_columns if col not in combined_fold_metrics.columns]
if missing_metric_cols:
    raise ValueError(f"Missing required metric columns: {missing_metric_cols}")
if not combined_predictions.empty:
    missing_prediction_cols = [col for col in required_prediction_columns if col not in combined_predictions.columns]
    if missing_prediction_cols:
        raise ValueError(f"Missing required prediction columns: {missing_prediction_cols}")

warnings_list = []
for config in MODEL_CONFIGS:
    name = config["model_display_name"]
    metric_folds = sorted(combined_fold_metrics.loc[combined_fold_metrics["model_display_name"] == name, "fold"].astype(int).unique().tolist())
    if metric_folds != FOLDS:
        warnings_list.append(f"{name}: metric folds are {metric_folds}, expected {FOLDS}.")
    if config["predictions_available"]:
        pred_folds = sorted(combined_predictions.loc[combined_predictions["model_display_name"] == name, "fold"].astype(int).unique().tolist())
        if pred_folds != FOLDS:
            warnings_list.append(f"{name}: prediction folds are {pred_folds}, expected {FOLDS}.")

if not combined_predictions.empty:
    if not combined_predictions["y_prob"].between(0, 1).all():
        warnings_list.append("Some predicted probabilities are outside [0, 1].")
    if not set(combined_predictions["y_true"].dropna().astype(int).unique()).issubset({0, 1}):
        warnings_list.append("Prediction y_true values are not binary.")
    for (model, fold), group in combined_predictions.groupby(["model_display_name", "fold"]):
        if group["sample_id"].duplicated().any():
            warnings_list.append(f"{model}, fold {fold}: duplicate sample IDs in predictions.")

    # Cross-check test sample IDs/label counts across the methods that have predictions.
    if len(PREDICTION_METHODS) >= 2:
        for fold in FOLDS:
            id_sets = {}
            n_pos_neg = {}
            for name in PREDICTION_METHODS:
                group = combined_predictions[(combined_predictions["model_display_name"] == name) & (combined_predictions["fold"] == fold)]
                id_sets[name] = set(group["sample_id"].astype(str))
                n_pos_neg[name] = (int((group["y_true"] == 1).sum()), int((group["y_true"] == 0).sum()))
            reference_ids = id_sets[PREDICTION_METHODS[0]]
            reference_counts = n_pos_neg[PREDICTION_METHODS[0]]
            for name in PREDICTION_METHODS:
                if id_sets[name] != reference_ids:
                    warnings_list.append(f"Fold {fold}: test sample IDs differ for {name}.")
                if n_pos_neg[name] != reference_counts:
                    warnings_list.append(f"Fold {fold}: test label counts differ for {name}: {n_pos_neg[name]} vs {reference_counts}.")

print("Note: the five archived methods provide fold metrics only; their predictions were not shipped with this package, so pooled ROC/PR and the per-fold sample-ID cross-check cover the two new methods only.")
if warnings_list:
    print("Validation warnings:")
    for warning in warnings_list:
        print("WARNING:", warning)
else:
    print("Validation passed without warnings.")'''

CELL16 = '''summary_rows = []
for config in MODEL_CONFIGS:
    name = config["model_display_name"]
    df = combined_fold_metrics[combined_fold_metrics["model_display_name"] == name].copy()
    summary_rows.append({
        "model_display_name": name,
        "model_key": config["model_key"],
        "pretraining": config["pretraining"],
        "pretext_category": config["pretext_category"],
        "patch_size": PATCH_SIZE,
        "mean_auc": df["auc"].mean(),
        "std_auc": df["auc"].std(ddof=1),
        "median_auc": df["auc"].median(),
        "min_auc": df["auc"].min(),
        "max_auc": df["auc"].max(),
        "worst_fold_auc": df["auc"].min(),
        "best_fold_auc": df["auc"].max(),
        "mean_pr_auc": df["pr_auc"].mean(),
        "std_pr_auc": df["pr_auc"].std(ddof=1),
        "mean_f1_05": df["f1_05"].mean(),
        "std_f1_05": df["f1_05"].std(ddof=1),
        "mean_recall_05": df["recall_05"].mean(),
        "std_recall_05": df["recall_05"].std(ddof=1),
        "mean_precision_05": df["precision_05"].mean(),
        "std_precision_05": df["precision_05"].std(ddof=1),
        "mean_accuracy_05": df["accuracy_05"].mean(),
        "std_accuracy_05": df["accuracy_05"].std(ddof=1),
        "mean_f1_best": df["f1_best_f1"].mean(),
        "std_f1_best": df["f1_best_f1"].std(ddof=1),
    })

summary_metrics = pd.DataFrame(summary_rows)
scratch_row = summary_metrics.loc[summary_metrics["model_display_name"] == "Scratch"].iloc[0]
summary_metrics["delta_mean_auc_vs_scratch"] = summary_metrics["mean_auc"] - scratch_row["mean_auc"]
summary_metrics["delta_std_auc_vs_scratch"] = summary_metrics["std_auc"] - scratch_row["std_auc"]
summary_metrics["delta_worst_auc_vs_scratch"] = summary_metrics["worst_fold_auc"] - scratch_row["worst_fold_auc"]
summary_metrics["rank_by_mean_auc"] = summary_metrics["mean_auc"].rank(ascending=False, method="min").astype(int)
summary_metrics["rank_by_stability"] = summary_metrics["std_auc"].rank(ascending=True, method="min").astype(int)
summary_metrics["rank_by_worst_fold_auc"] = summary_metrics["worst_fold_auc"].rank(ascending=False, method="min").astype(int)

role_map = {
    "Scratch": "Supervised baseline; high spatial fold variability.",
    "Masked reconstruction": "Most stable SSL transfer; best worst-fold AUC.",
    "Contrastive learning": "Highest mean AUC; effective but less stable than masked reconstruction.",
    "Jigsaw": "Weak SSL transfer; underperforms Scratch.",
    "Rotation prediction": "Pretext task solved, but downstream transfer is weak.",
    "Cross-channel masking": "New reconstruction-style SSL; predicts one held-out terrain factor (default TWI).",
    "Strip jigsaw": "New coarse 3-strip spatial-ordering SSL.",
}
summary_metrics["main_scientific_role"] = summary_metrics["model_display_name"].map(role_map)

summary_columns = [
    "model_display_name", "model_key", "pretraining", "pretext_category", "patch_size", "mean_auc", "std_auc",
    "median_auc", "min_auc", "max_auc", "worst_fold_auc", "best_fold_auc", "mean_pr_auc", "std_pr_auc",
    "mean_f1_05", "std_f1_05", "mean_recall_05", "std_recall_05", "mean_precision_05", "std_precision_05",
    "mean_accuracy_05", "std_accuracy_05", "mean_f1_best", "std_f1_best", "delta_mean_auc_vs_scratch",
    "delta_std_auc_vs_scratch", "delta_worst_auc_vs_scratch", "rank_by_mean_auc", "rank_by_stability",
    "rank_by_worst_fold_auc", "main_scientific_role",
]
summary_metrics = summary_metrics[summary_columns]
summary_metrics["model_display_name"] = pd.Categorical(summary_metrics["model_display_name"], categories=MODEL_ORDER, ordered=True)
summary_metrics = summary_metrics.sort_values("model_display_name").reset_index(drop=True)
summary_metrics.to_csv(SUMMARY_METRICS_OUT, index=False)
print("Saved summary metrics:", SUMMARY_METRICS_OUT)
summary_metrics'''

CELL18 = '''transfer_rows = [
    ["Scratch", "None", "Not applicable", "Reference baseline", "Supervised baseline; no SSL pretraining."],
    ["Masked reconstruction", "Masked reconstruction", "Solved sufficiently for representation learning", "Effective and stable", "Effective SSL; strongest spatial stability and best worst-fold AUC."],
    ["Contrastive learning", "SimCLR-style contrastive learning", "Solved sufficiently for representation learning", "Effective average discrimination", "Effective SSL; highest mean AUC."],
    ["Jigsaw", "Jigsaw permutation prediction", "Near random", "Weak and below Scratch", "Weak or ineffective SSL; pretext task near random and downstream underperforms Scratch."],
    ["Rotation prediction", "Rotation prediction", "Solved almost perfectly", "Weak and below Scratch", "Pretext task solved almost perfectly, but downstream transfer is weak and underperforms Scratch."],
    ["Cross-channel masking", "Cross-channel masking (predict one held-out terrain factor)", "See pretext val loss", "See downstream metrics", "New reconstruction-style SSL aligned with the 14-factor raster stack."],
    ["Strip jigsaw", "1D 3-strip permutation prediction (6 classes)", "See pretext val accuracy", "See downstream metrics", "New coarse spatial-ordering SSL; simpler than the 16-tile jigsaw."],
]
transfer_summary = pd.DataFrame(
    transfer_rows,
    columns=["model_display_name", "pretext_task", "pretext_task_outcome", "downstream_transfer_outcome", "scientific_interpretation"],
).merge(summary_metrics[["model_display_name", "mean_auc", "std_auc", "worst_fold_auc"]], on="model_display_name", how="left")
transfer_summary = transfer_summary[
    ["model_display_name", "pretext_task", "pretext_task_outcome", "downstream_transfer_outcome", "mean_auc", "std_auc", "worst_fold_auc", "scientific_interpretation"]
]
transfer_summary.to_csv(TRANSFER_SUMMARY_OUT, index=False)
print("Saved pretext transfer summary:", TRANSFER_SUMMARY_OUT)
transfer_summary'''

CELL20 = '''finding_map = {
    "Scratch": "Baseline; high spatial fold variability",
    "Masked reconstruction": "Most stable transfer and best worst-fold AUC",
    "Contrastive learning": "Highest mean AUC",
    "Jigsaw": "Weak transfer; below Scratch",
    "Rotation prediction": "Pretext solved, but weak downstream transfer",
    "Cross-channel masking": "New reconstruction-style SSL (predicts a held-out factor)",
    "Strip jigsaw": "New coarse 3-strip spatial-ordering SSL",
}
pretext_display = {
    "Scratch": "None",
    "Masked reconstruction": "Masked reconstruction",
    "Contrastive learning": "Contrastive learning",
    "Jigsaw": "Jigsaw",
    "Rotation prediction": "Rotation prediction",
    "Cross-channel masking": "Cross-channel masking",
    "Strip jigsaw": "Strip jigsaw",
}
# F1 is reported at a validation-tuned decision threshold (mean_f1_best). The
# default 0.5 threshold is degenerate for the two new methods, whose downstream
# probabilities saturate near 0 (so F1@0.5 collapses to 0.0 even though the
# ranking-based AUC is well defined). The validation-tuned threshold is selected
# per fold on the validation split only, consistently for all seven methods, so
# the comparison stays apples-to-apples. F1@0.5 is retained as a secondary
# column to make the saturation visible.
manuscript_table = pd.DataFrame({
    "Model": summary_metrics["model_display_name"].astype(str),
    "Pretext task": summary_metrics["model_display_name"].astype(str).map(pretext_display),
    "Mean AUC": summary_metrics["mean_auc"].round(3),
    "AUC SD": summary_metrics["std_auc"].round(3),
    "Worst-fold AUC": summary_metrics["worst_fold_auc"].round(3),
    "Mean PR-AUC": summary_metrics["mean_pr_auc"].round(3),
    "Mean F1 (tuned thr)": summary_metrics["mean_f1_best"].round(3),
    "Mean F1 at 0.5": summary_metrics["mean_f1_05"].round(3),
    "Main finding": summary_metrics["model_display_name"].astype(str).map(finding_map),
})
manuscript_table.to_csv(MANUSCRIPT_TABLE_OUT, index=False)
print("Saved manuscript-ready table:", MANUSCRIPT_TABLE_OUT)
print(
    "Note: 'Mean F1 (tuned thr)' uses a per-fold validation-selected threshold "
    "(consistent across all methods). 'Mean F1 at 0.5' is degenerate (0.0) for "
    "the two new methods because their downstream probabilities saturate near 0; "
    "AUC and PR-AUC are threshold-independent and unaffected."
)
manuscript_table'''

# Figure R1e (pooled ROC) gated on available predictions.
CELL27 = '''# Figure R2e: pooled ROC curves (methods with per-fold predictions only).
if combined_predictions.empty or not PREDICTION_METHODS:
    print("Skipping pooled ROC figure: no per-fold prediction files available.")
else:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in PREDICTION_METHODS:
        pred = combined_predictions[combined_predictions["model_display_name"] == name]
        y_true = pred["y_true"].to_numpy()
        y_prob = pred["y_prob"].to_numpy()
        auc = safe_roc_auc(y_true, y_prob)
        fpr, tpr = roc_curve_points(y_true, y_prob)
        ax.plot(fpr, tpr, color=PALETTE[name], label=f"{name} AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.9, label="Random baseline")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Pooled ROC curves (new methods)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")
    clean_axes(ax, keep_grid=True)
    fig.tight_layout()
    figure_paths.extend(save_figure(fig, "Fig_R2e_pooled_roc_new_tasks_ps32"))
    print("Pooled ROC and PR curves cover the two new methods only (archived predictions were not shipped); fold-wise SCV metrics remain the primary evaluation.")'''

CELL28 = '''# Figure R2f: pooled precision-recall curves (methods with per-fold predictions only).
if combined_predictions.empty or not PREDICTION_METHODS:
    print("Skipping pooled PR figure: no per-fold prediction files available.")
else:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in PREDICTION_METHODS:
        pred = combined_predictions[combined_predictions["model_display_name"] == name]
        y_true = pred["y_true"].to_numpy()
        y_prob = pred["y_prob"].to_numpy()
        ap = safe_average_precision(y_true, y_prob)
        recall, precision = pr_curve_points(y_true, y_prob)
        ax.plot(recall, precision, color=PALETTE[name], label=f"{name} AP {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Pooled precision-recall curves (new methods)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower left")
    clean_axes(ax, keep_grid=True)
    fig.tight_layout()
    figure_paths.extend(save_figure(fig, "Fig_R2f_pooled_pr_new_tasks_ps32"))
    print("Pooled ROC and PR curves cover the two new methods only (archived predictions were not shipped); fold-wise SCV metrics remain the primary evaluation.")'''

# Figure R1c trade-off: remove the fixed axis limits so 7 methods fit.
CELL25 = '''# Figure R2c: accuracy-stability trade-off.
fig, ax = plt.subplots(figsize=(6.8, 4.8))
for _, row in summary_ordered.iterrows():
    name = str(row["model_display_name"])
    ax.scatter(row["std_auc"], row["mean_auc"], s=70, color=PALETTE[name], edgecolor="black", linewidth=0.6, zorder=3)
    ax.annotate(name, (row["std_auc"], row["mean_auc"]), textcoords="offset points", xytext=(6, 4), ha="left", va="center")
ax.set_xlabel("AUC SD")
ax.set_ylabel("Mean AUC")
ax.margins(0.18)
ax.set_title("Accuracy-stability trade-off")
clean_axes(ax, keep_grid=True)
fig.tight_layout()
figure_paths.extend(save_figure(fig, "Fig_R2c_auc_stability_tradeoff_ps32"))'''

# Figure R1g transfer matrix: extend with the two new methods.
CELL29 = '''# Figure R2g: pretext-task transfer matrix.
transfer_matrix_rows = ["Masked reconstruction", "Contrastive learning", "Jigsaw", "Rotation prediction", "Cross-channel masking", "Strip jigsaw"]
transfer_matrix_columns = ["Mean AUC vs Scratch", "AUC SD vs Scratch", "Worst-fold vs Scratch", "Transfer outcome"]


def _sign_label(value, positive_is_good=True, eps=0.003):
    if value > eps:
        return "Positive" if positive_is_good else "Negative"
    if value < -eps:
        return "Negative" if positive_is_good else "Positive"
    return "Neutral"


transfer_matrix_text = []
for name in transfer_matrix_rows:
    row = summary_metrics.loc[summary_metrics["model_display_name"] == name].iloc[0]
    mean_label = _sign_label(row["delta_mean_auc_vs_scratch"], positive_is_good=True)
    # Lower AUC SD than scratch is good, so a negative delta is the favorable outcome.
    std_label = _sign_label(-row["delta_std_auc_vs_scratch"], positive_is_good=True)
    worst_label = _sign_label(row["delta_worst_auc_vs_scratch"], positive_is_good=True)
    outcome = "Effective" if (row["delta_mean_auc_vs_scratch"] > 0 and row["delta_worst_auc_vs_scratch"] >= 0) else "Weak transfer"
    transfer_matrix_text.append([mean_label, std_label, worst_label, outcome])

value_color = {
    "Positive": "#A6DBA0", "Neutral": "#F0F0F0", "Negative": "#F2D7D5",
    "Effective": "#A6DBA0", "Weak transfer": "#F2D7D5",
}
fig, ax = plt.subplots(figsize=(11.6, 4.2))
ax.set_xlim(0, len(transfer_matrix_columns))
ax.set_ylim(0, len(transfer_matrix_rows))
for i, row_name in enumerate(transfer_matrix_rows):
    for j, col_name in enumerate(transfer_matrix_columns):
        y = len(transfer_matrix_rows) - i - 1
        text = transfer_matrix_text[i][j]
        ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor=value_color.get(text, "white"), edgecolor="white", linewidth=2))
        ax.text(j + 0.5, y + 0.5, text, ha="center", va="center", wrap=True)
ax.set_xticks(np.arange(len(transfer_matrix_columns)) + 0.5)
ax.set_xticklabels(transfer_matrix_columns)
ax.set_yticks(np.arange(len(transfer_matrix_rows)) + 0.5)
ax.set_yticklabels(list(reversed(transfer_matrix_rows)))
ax.set_title("Pretext-task transfer matrix")
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout()
figure_paths.extend(save_figure(fig, "Fig_R2g_pretext_transfer_matrix_ps32"))'''

CELL30 = '''# Figure R2h: metric heatmap. F1 uses the validation-tuned threshold (mean_f1_best)
# so the two new methods' probability saturation does not collapse the F1 column to 0.
heatmap_cols = ["Mean AUC", "AUC SD", "Worst-fold AUC", "Mean PR-AUC", "F1 (tuned)"]
heatmap_data = summary_ordered[["mean_auc", "std_auc", "worst_fold_auc", "mean_pr_auc", "mean_f1_best"]].to_numpy(dtype=float)
color_data = np.zeros_like(heatmap_data)
for j in range(heatmap_data.shape[1]):
    col = heatmap_data[:, j]
    color_data[:, j] = 0.5 if np.isclose(col.max(), col.min()) else (col - col.min()) / (col.max() - col.min())
fig, ax = plt.subplots(figsize=(7.8, 4.6))
im = ax.imshow(color_data, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(np.arange(len(heatmap_cols)))
ax.set_xticklabels(heatmap_cols, rotation=25, ha="right")
ax.set_yticks(np.arange(len(MODEL_ORDER)))
ax.set_yticklabels(MODEL_ORDER)
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        ax.text(j, i, f"{heatmap_data[i, j]:.3f}", ha="center", va="center", color="black")
ax.set_title("Summary of downstream metrics")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.ax.tick_params(labelsize=10)
cbar.set_label("Relative value", fontsize=10)
fig.tight_layout()
figure_paths.extend(save_figure(fig, "Fig_R2h_metric_heatmap_all_tasks_ps32"))'''

CELL33 = '''print("Final ps32 SCV summary (seven methods):")
print(manuscript_table.to_string(index=False))

ranked = summary_metrics.sort_values("mean_auc", ascending=False)
best = ranked.iloc[0]
most_stable = summary_metrics.sort_values("std_auc", ascending=True).iloc[0]
best_worst = summary_metrics.sort_values("worst_fold_auc", ascending=False).iloc[0]
scratch = summary_metrics[summary_metrics["model_display_name"] == "Scratch"].iloc[0]
cross_channel = summary_metrics[summary_metrics["model_display_name"] == "Cross-channel masking"].iloc[0]
strip = summary_metrics[summary_metrics["model_display_name"] == "Strip jigsaw"].iloc[0]

print("\\nKey scientific findings:")
print(f"1. Highest mean AUC: {best['model_display_name']} ({best['mean_auc']:.3f}).")
print(f"2. Most stable (lowest AUC SD): {most_stable['model_display_name']} ({most_stable['std_auc']:.3f}).")
print(f"3. Best worst-fold AUC: {best_worst['model_display_name']} ({best_worst['worst_fold_auc']:.3f}).")
print(f"4. Cross-channel masking: mean AUC {cross_channel['mean_auc']:.3f} (delta vs Scratch {cross_channel['delta_mean_auc_vs_scratch']:+.3f}).")
print(f"5. Strip jigsaw: mean AUC {strip['mean_auc']:.3f} (delta vs Scratch {strip['delta_mean_auc_vs_scratch']:+.3f}).")
print(f"6. Scratch baseline mean AUC: {scratch['mean_auc']:.3f}.")
print(f"7. F1 reported at the validation-tuned threshold: cross-channel {cross_channel['mean_f1_best']:.3f}, strip jigsaw {strip['mean_f1_best']:.3f}. At the default 0.5 threshold both collapse to 0.000 because their downstream probabilities saturate near 0; AUC/PR-AUC are threshold-independent.")
print("\\nHardware-mix caveat: the five archived methods were trained on CUDA; the two new methods were pretrained on Apple Silicon (MPS). The 5-cluster SCV protocol is identical, so cross-hardware comparisons are indicative.")
print("__FINETUNE_NOTE__")'''

CELL35 = '''print("Saved summary CSV files:")
for path in [FOLD_METRICS_OUT, SUMMARY_METRICS_OUT, TRANSFER_SUMMARY_OUT, MANUSCRIPT_TABLE_OUT]:
    print(path, "exists=", path.exists(), "bytes=", path.stat().st_size if path.exists() else None)

print("\\nSaved figure files:")
for path in figure_paths:
    print(path, "exists=", path.exists(), "bytes=", path.stat().st_size if path.exists() else None)

print("\\nQA summary:")
print("No training was performed.")
print("No checkpoints were modified.")
print("No existing metric or prediction files were modified.")
print("Five archived methods contribute fold metrics only; the two new methods contribute fold metrics and predictions.")
print("Pooled ROC and PR curves cover the two new methods only; fold-wise SCV metrics remain the primary evaluation.")
print("Standardized publication plotting style was applied with font:", applied_font)'''


def main() -> None:
    nb = json.loads(SRC.read_text())
    cells = nb["cells"]

    def sub_note(text: str) -> str:
        if FINETUNE_NOTE:
            return text.replace("__FINETUNE_NOTE__", FINETUNE_NOTE)
        # Drop the placeholder line entirely (and any leading blank it leaves).
        return text.replace("\n__FINETUNE_NOTE__", "").replace('print("__FINETUNE_NOTE__")\n', "").replace('print("__FINETUNE_NOTE__")', "")

    replacements = {
        0: md(CELL0),
        1: md(sub_note(CELL1)),
        2: code(CELL2.replace("__R2_NAMESPACE__", R2_NAMESPACE)),
        6: code(CELL6),
        7: md(CELL7),
        8: code(CELL8),
        9: md(CELL9),
        10: code(CELL10),
        11: md(CELL11),
        12: code(CELL12),
        16: code(CELL16),
        18: code(CELL18),
        20: code(CELL20),
        25: code(CELL25),
        30: code(CELL30),
        27: code(CELL27),
        28: code(CELL28),
        29: code(CELL29),
        33: code(sub_note(CELL33)),
        35: code(CELL35),
    }
    for index, cell in replacements.items():
        cells[index] = cell

    # Clear any execution counts/outputs on the carried-over cells.
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            cell.setdefault("metadata", {})

    DST.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Wrote {DST} with {len(cells)} cells.")


if __name__ == "__main__":
    main()
