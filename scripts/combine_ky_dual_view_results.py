"""Combine KY dual-view aligned fold metrics (pulled from the pod) into
seed-level mean +/- SD tables and SSL-minus-scratch deltas.

Plain pandas — safe to run in this repo's venv (no shim / no src imports).

Reads every  outputs/KY_dual_view_aligned/flat_cv/*/comparison/
final_10m_flat_cv_lr_sweep_fold_metrics.csv, concatenates (dropping
duplicate model/seed/fold/lr rows, keeping the last), and writes:
  outputs/KY_dual_view_aligned/comparison/ky_fold_metrics_combined.csv
  outputs/KY_dual_view_aligned/comparison/ky_setting_summary.csv
  outputs/KY_dual_view_aligned/comparison/ky_meeting_table.md   (printed too)

Seed-level protocol (THE RULE): a seed's score = mean AUC over its folds;
report mean +/- SD over seeds. A delta is real only if it clears ~2x the
seed-to-seed SD. PR-AUC is reported but must NOT be compared to NYC
(different class-balance history).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "outputs" / "KY_dual_view_aligned"
OUT = BASE / "comparison"

DISPLAY = {
    "scratch_full_model_baseline": "Scratch (full model, LR 1e-3)",
    "dual_view_masked_reconstruction": "Masked reconstruction",
    "dual_view_contrastive_learning": "Contrastive",
    "dual_view_jigsaw": "Jigsaw",
    "dual_view_rotation_prediction": "Rotation",
}


def _lr_label(v) -> str:
    if pd.isna(v) or v == "":
        return "scratch"
    v = float(v)
    return "frozen (LR 0)" if v == 0.0 else f"enc LR {v:g}"


def main() -> None:
    files = sorted(BASE.glob("flat_cv/*/comparison/final_10m_flat_cv_lr_sweep_fold_metrics.csv"))
    files = [f for f in files if "smoke" not in str(f)]
    if not files:
        raise SystemExit(f"No fold metrics found under {BASE}/flat_cv/*/comparison/")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_tag"] = f.parent.parent.name
        frames.append(df)
    metrics = pd.concat(frames, ignore_index=True)
    metrics["encoder_lr_numeric"] = pd.to_numeric(metrics["encoder_lr"], errors="coerce")
    key = ["model_name", "seed", "fold_id", "encoder_lr_numeric"]
    metrics = metrics.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT / "ky_fold_metrics_combined.csv", index=False)

    # seed-level: mean over folds within seed, then mean/SD over seeds
    rows = []
    for (model, lr), grp in metrics.groupby(["model_name", "encoder_lr_numeric"], dropna=False):
        per_seed = grp.groupby("seed").agg(
            seed_auc=("auc", "mean"), seed_pr=("pr_auc", "mean"), n_folds=("fold_id", "nunique")
        )
        rows.append(
            {
                "model_name": model,
                "encoder_lr": "" if pd.isna(lr) else float(lr),
                "n_seeds": int(len(per_seed)),
                "folds_per_seed": ";".join(str(int(v)) for v in per_seed["n_folds"]),
                "n_fold_rows": int(len(grp)),
                "mean_auc_seedlevel": float(per_seed["seed_auc"].mean()),
                "sd_auc_seedlevel": float(per_seed["seed_auc"].std(ddof=1)) if len(per_seed) > 1 else np.nan,
                "mean_auc_foldlevel": float(grp["auc"].mean()),
                "sd_auc_foldlevel": float(grp["auc"].std(ddof=1)) if len(grp) > 1 else np.nan,
                "worst_fold_auc": float(grp["auc"].min()),
                "mean_pr_auc": float(grp["pr_auc"].mean()),
                "mean_f1_05": float(grp["f1_05"].mean()),
                "mean_best_epoch": float(grp["best_epoch"].mean()),
            }
        )
    setting = pd.DataFrame(rows).sort_values(["model_name", "encoder_lr"], na_position="first")
    setting.to_csv(OUT / "ky_setting_summary.csv", index=False)

    scratch = setting.loc[setting["model_name"].eq("scratch_full_model_baseline")]
    scratch_mean = float(scratch["mean_auc_seedlevel"].iloc[0]) if not scratch.empty else np.nan
    scratch_sd = float(scratch["sd_auc_seedlevel"].iloc[0]) if not scratch.empty else np.nan

    lines = [
        "# KY dual-view aligned — flat CV (group-safe split), head LR 1e-3, batch 128",
        "",
        f"Fold-metric files: {len(files)} | fold rows: {len(metrics)} | "
        f"seeds present: {sorted(metrics['seed'].unique().tolist())}",
        "",
        "| Model | Encoder LR | AUC mean ± SD (seeds) | worst fold | PR-AUC | Δ vs scratch |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in setting.iterrows():
        sd = "n/a" if pd.isna(r["sd_auc_seedlevel"]) else f"{r['sd_auc_seedlevel']:.3f}"
        delta = ""
        if r["model_name"] != "scratch_full_model_baseline" and np.isfinite(scratch_mean):
            d = r["mean_auc_seedlevel"] - scratch_mean
            delta = f"{d:+.3f}"
        lines.append(
            f"| {DISPLAY.get(r['model_name'], r['model_name'])} | {_lr_label(r['encoder_lr'])} | "
            f"{r['mean_auc_seedlevel']:.3f} ± {sd} (n={r['n_seeds']}) | "
            f"{r['worst_fold_auc']:.3f} | {r['mean_pr_auc']:.3f} | {delta} |"
        )
    lines += [
        "",
        f"Scratch reference: {scratch_mean:.3f} ± {scratch_sd if np.isfinite(scratch_sd) else float('nan'):.3f}"
        if np.isfinite(scratch_mean)
        else "Scratch reference: NOT YET AVAILABLE",
        "",
        "Rule: a delta is real only if it clears ~2x the seed-to-seed SD.",
        "Do NOT compare PR-AUC to NYC numbers (different balance history).",
    ]
    table = "\n".join(lines)
    (OUT / "ky_meeting_table.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
