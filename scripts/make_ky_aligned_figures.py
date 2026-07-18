"""Meeting figure pack for the KY dual-view aligned run (2026-07-17).

Reads outputs/KY_dual_view_aligned/flat_cv/*/comparison/*fold_metrics.csv (+
pod-A SSL pretraining logs) and writes PNGs to figures/KY_dual_view_aligned/.

Color roles are fixed by entity: masked recon = blue, contrastive = amber,
scratch = neutral gray reference (recessive baseline, always direct-labeled).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "outputs" / "KY_dual_view_aligned"
FIG = REPO / "figures" / "KY_dual_view_aligned"
FIG.mkdir(parents=True, exist_ok=True)

C_MASKED = "#2563eb"
C_CONTR = "#d97706"
C_SCRATCH = "#6b7280"
MODEL_COLOR = {
    "dual_view_masked_reconstruction": C_MASKED,
    "dual_view_contrastive_learning": C_CONTR,
    "scratch_full_model_baseline": C_SCRATCH,
}
MODEL_LABEL = {
    "dual_view_masked_reconstruction": "Masked reconstruction",
    "dual_view_contrastive_learning": "Contrastive",
    "scratch_full_model_baseline": "Scratch",
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def load_metrics() -> pd.DataFrame:
    files = [
        f
        for f in BASE.glob("flat_cv/*/comparison/final_10m_flat_cv_lr_sweep_fold_metrics.csv")
        if "smoke" not in str(f)
    ]
    metrics = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    metrics["encoder_lr"] = pd.to_numeric(metrics["encoder_lr"], errors="coerce")
    metrics = metrics.drop_duplicates(
        subset=["model_name", "seed", "fold_id", "encoder_lr"], keep="last"
    ).reset_index(drop=True)
    return metrics


def seed_level(metrics: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        metrics.groupby(["model_name", "encoder_lr", "seed"], dropna=False)["auc"]
        .mean()
        .reset_index(name="seed_auc")
    )
    agg = (
        per_seed.groupby(["model_name", "encoder_lr"], dropna=False)["seed_auc"]
        .agg(mean="mean", sd="std", n="size")
        .reset_index()
    )
    return agg


def save(fig, name: str) -> None:
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO)}")


def fig_lr_sweep(agg: pd.DataFrame) -> None:
    scratch = agg.loc[agg["model_name"].eq("scratch_full_model_baseline")]
    s_mean, s_sd = float(scratch["mean"].iloc[0]), float(scratch["sd"].iloc[0])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(s_mean - 2 * s_sd, s_mean + 2 * s_sd, color=C_SCRATCH, alpha=0.15, zorder=0)
    ax.axhline(s_mean, color=C_SCRATCH, linewidth=2, linestyle="--", zorder=1)
    ax.annotate(
        f"Scratch {s_mean:.3f} (±2 seed-SD band)",
        xy=(0.02, s_mean),
        xycoords=("axes fraction", "data"),
        xytext=(0, 7),
        textcoords="offset points",
        color="#374151",
        fontsize=10,
    )
    for model in ["dual_view_masked_reconstruction", "dual_view_contrastive_learning"]:
        grp = agg.loc[agg["model_name"].eq(model)].sort_values("encoder_lr")
        ax.errorbar(
            grp["encoder_lr"],
            grp["mean"],
            yerr=grp["sd"].fillna(0.0),
            marker="o",
            markersize=7,
            linewidth=2,
            capsize=3,
            color=MODEL_COLOR[model],
            label=MODEL_LABEL[model],
        )
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_xticks([0, 1e-5, 1e-4, 1e-3])
    ax.set_xticklabels(["0\n(strict frozen)", "1e-5", "1e-4", "1e-3"])
    ax.set_xlabel("Encoder learning rate (head LR fixed at 1e-3)")
    ax.set_ylabel("AUC (mean over seeds; error bar = seed SD)")
    ax.set_title("KY aligned protocol: encoder-LR sweep — no collapse, best at 1e-3")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "F1_encoder_lr_sweep")


def fig_best_bars(agg: pd.DataFrame) -> None:
    rows = [
        ("Scratch", "scratch_full_model_baseline", np.nan),
        ("Masked recon\n(enc 1e-3)", "dual_view_masked_reconstruction", 1e-3),
        ("Contrastive\n(enc 1e-3)", "dual_view_contrastive_learning", 1e-3),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for i, (label, model, lr) in enumerate(rows):
        if np.isnan(lr):
            r = agg.loc[agg["model_name"].eq(model)].iloc[0]
        else:
            r = agg.loc[agg["model_name"].eq(model) & np.isclose(agg["encoder_lr"], lr)].iloc[0]
        ax.bar(i, r["mean"], yerr=0 if np.isnan(r["sd"]) else r["sd"], width=0.62,
               color=MODEL_COLOR[model], capsize=4)
        ax.annotate(f"{r['mean']:.3f}", xy=(i, r["mean"]), xytext=(0, 6),
                    textcoords="offset points", ha="center", color="#111827")
    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(0.90, 0.97)
    ax.set_ylabel("AUC (mean ± seed SD)")
    ax.set_title("Best settings: SSL reaches parity with scratch on KY")
    save(fig, "F2_best_setting_bars")


def fig_protocol_effect() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    groups = ["Scratch", "Best SSL"]
    july = [0.733, 0.708]
    aligned = [0.955, 0.955]
    x = np.arange(2)
    w = 0.36
    b1 = ax.bar(x - w / 2, july, w, color="#9ca3af", label="July run (misaligned protocol)")
    b2 = ax.bar(x + w / 2, aligned, w, color=C_MASKED, label="Aligned protocol (today)")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(x, groups)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("AUC (flat CV)")
    ax.set_title("Protocol alignment is worth +0.22 AUC — far more than SSL itself")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "F3_protocol_effect")


def fig_nyc_vs_ky() -> None:
    labels = ["Masked recon\n(NYC-562)", "Sequential\n(NYC-562)", "Cross-channel\n(NYC-562)",
              "Masked recon\n(KY aligned)", "Contrastive\n(KY aligned)"]
    deltas = [0.045, 0.043, 0.017, 0.001, -0.001]
    colors = [C_MASKED, "#60a5fa", "#93c5fd", C_MASKED, C_CONTR]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(range(len(deltas)), deltas, 0.62, color=colors)
    for b, d in zip(bars, deltas):
        ax.annotate(f"{d:+.3f}", xy=(b.get_x() + b.get_width() / 2, d),
                    xytext=(0, 5 if d >= 0 else -14), textcoords="offset points",
                    ha="center", fontsize=10)
    ax.axhline(0, color="#374151", linewidth=1)
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_ylabel("SSL − scratch AUC (best settings)")
    ax.set_title("SSL pays where labels are scarce: NYC (562 labels) vs KY (18,396 labels)")
    ax.axvspan(-0.5, 2.5, color="#f3f4f6", zorder=0)
    ax.annotate("NYC: SSL wins (real vs seed noise)", xy=(1, 0.049), ha="center", fontsize=10, color="#374151")
    ax.annotate("KY: parity", xy=(3.5, 0.007), ha="center", fontsize=10, color="#374151")
    save(fig, "F4_nyc_vs_ky_delta")


def fig_foldwise(metrics: pd.DataFrame) -> None:
    settings = [
        ("Scratch", "scratch_full_model_baseline", np.nan),
        ("Masked frozen", "dual_view_masked_reconstruction", 0.0),
        ("Masked 1e-4", "dual_view_masked_reconstruction", 1e-4),
        ("Masked 1e-3", "dual_view_masked_reconstruction", 1e-3),
        ("Contr. frozen", "dual_view_contrastive_learning", 0.0),
        ("Contr. 1e-4", "dual_view_contrastive_learning", 1e-4),
        ("Contr. 1e-3", "dual_view_contrastive_learning", 1e-3),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    rng = np.random.default_rng(0)
    for i, (label, model, lr) in enumerate(settings):
        if np.isnan(lr):
            vals = metrics.loc[metrics["model_name"].eq(model), "auc"]
        else:
            vals = metrics.loc[metrics["model_name"].eq(model) & np.isclose(metrics["encoder_lr"].fillna(-1), lr), "auc"]
        x = i + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=26, color=MODEL_COLOR[model], alpha=0.75, edgecolors="white", linewidths=1)
        ax.hlines(vals.mean(), i - 0.24, i + 0.24, color="#111827", linewidth=2)
    ax.set_xticks(range(len(settings)), [s[0] for s in settings], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Fold AUC (all seeds, all folds)")
    ax.set_title("Fold-level stability — every fold of every seed (bar = mean)")
    save(fig, "F5_foldwise_stability")


def fig_pretraining_curves() -> None:
    root = BASE / "ssl_pod_a" / "ssl_task_comparison"
    tasks = [
        ("dual_view_masked_reconstruction", "Masked reconstruction", C_MASKED),
        ("dual_view_contrastive_learning", "Contrastive", C_CONTR),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    plotted_any = False
    for ax, (task, label, color) in zip(axes, tasks):
        logs = sorted(root.glob(f"{task}/seed_*/pretraining_log.csv"))
        for k, log_path in enumerate(logs):
            df = pd.read_csv(log_path)
            seed = log_path.parent.name.replace("seed_", "")
            if "val_loss" in df:
                ax.plot(df["epoch"], df["val_loss"], linewidth=2, alpha=0.85,
                        color=color, linestyle=["-", "--", ":"][k % 3], label=f"seed {seed} val")
                plotted_any = True
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{label} SSL pretraining (50k corpus)")
        ax.legend(frameon=False, fontsize=9)
    if plotted_any:
        save(fig, "F6_ssl_pretraining_curves")
    else:
        plt.close(fig)
        print("no pretraining logs found; skipped F6")


def fig_frozen_vs_finetuned(agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    x = np.arange(2)
    w = 0.36
    frozen = [
        float(agg.loc[agg["model_name"].eq(m) & np.isclose(agg["encoder_lr"].fillna(-1), 0.0), "mean"].iloc[0])
        for m in ["dual_view_masked_reconstruction", "dual_view_contrastive_learning"]
    ]
    tuned = [
        float(agg.loc[agg["model_name"].eq(m) & np.isclose(agg["encoder_lr"], 1e-3), "mean"].iloc[0])
        for m in ["dual_view_masked_reconstruction", "dual_view_contrastive_learning"]
    ]
    b1 = ax.bar(x - w / 2, frozen, w, color="#9ca3af", label="Strict frozen (LR 0)")
    b2 = ax.bar(x + w / 2, tuned, w, color=C_MASKED, label="Fine-tuned (enc LR 1e-3)")
    b2[1].set_color(C_CONTR)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(x, ["Masked reconstruction", "Contrastive"])
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("AUC (mean over seeds)")
    ax.set_title("Frozen-probe evaluation undersells SSL — fine-tuning is the value")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "F7_frozen_vs_finetuned")


def main() -> None:
    metrics = load_metrics()
    agg = seed_level(metrics)
    fig_lr_sweep(agg)
    fig_best_bars(agg)
    fig_protocol_effect()
    fig_nyc_vs_ky()
    fig_foldwise(metrics)
    fig_pretraining_curves()
    fig_frozen_vs_finetuned(agg)
    print("done")


if __name__ == "__main__":
    main()
