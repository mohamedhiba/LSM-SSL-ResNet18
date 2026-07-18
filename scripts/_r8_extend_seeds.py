#!/usr/bin/env python
"""Extend R8 (30m-vs-10m) from 3 to 5 seeds for a more robust verdict.

Adds seeds 1 and 99 to both arms (reusing the same encoders + matched indices +
trusted finetune engine), appends to outputs/R8_resolution_{arm}/r8_{arm}.csv,
and re-aggregates outputs/R8_resolution_summary.csv over all seeds present.
Finetune-only (encoders reused) -> minutes per seed. Idempotent per seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

NEW_SEEDS = [1, 99]


def main() -> None:
    import pandas as pd
    import _r8_resolution as r8
    from src.train_finetune import run_pretrained_resnet18_scv_experiment
    from src.utils import get_device, set_global_seed

    device = get_device()
    arms = {
        "30m": (r8.RASTER_30M, r8.LABELED_30M_MATCHED, r8.ENC_30M),
        "10m": (r8.RASTER_10M, r8.LABELED_10M, r8.ENC_10M),
    }
    for arm, (rdir, lab, enc) in arms.items():
        csv = PROJECT_ROOT / f"outputs/R8_resolution_{arm}/r8_{arm}.csv"
        rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
        have = {int(r["seed"]) for r in rows}
        for seed in NEW_SEEDS:
            if seed in have:
                print(f"[{arm}] seed {seed} already present, skip", flush=True)
                continue
            set_global_seed(seed)
            res = run_pretrained_resnet18_scv_experiment(
                project_root=PROJECT_ROOT, model_name=f"cc_{arm}", pretraining="cross_channel",
                patch_index_csv=lab, raster_dir=rdir, encoder_checkpoint_path=enc,
                output_root=PROJECT_ROOT / f"outputs/R8_resolution_{arm}/seed{seed}",
                figure_root=PROJECT_ROOT / f"figures/R8_resolution_{arm}/seed{seed}",
                checkpoint_dir=PROJECT_ROOT / f"checkpoints/finetuned/R8_resolution_{arm}",
                patch_size=32, random_seed=seed, batch_size=16,
                encoder_learning_rate=5e-6, head_learning_rate=2e-5,
                max_epochs=60, early_stopping_patience=12, dropout=0.4,
                expected_checkpoint_keys={"encoder_state_dict"}, num_workers=0,
                device=device, plot_figures=False, cache_in_memory=True, with_mask=True,
            )
            fm = res["fold_metrics"]
            for _, fr in fm.iterrows():
                rows.append({"arm": arm, "seed": seed, "fold": int(fr["fold"]),
                             "AUC": float(fr["auc"]), "PR_AUC": float(fr["pr_auc"])})
            print(f"  [{arm}] seed {seed}: mean AUC={fm['auc'].mean():.4f} "
                  f"folds={[round(a,3) for a in fm['auc'].tolist()]}", flush=True)
        df = pd.DataFrame(rows).drop_duplicates(["arm", "seed", "fold"])
        df.to_csv(csv, index=False)

    # re-aggregate over ALL seeds present
    rows = []
    for arm in ("30m", "10m"):
        csv = PROJECT_ROOT / f"outputs/R8_resolution_{arm}/r8_{arm}.csv"
        if not csv.exists():
            continue
        a = pd.read_csv(csv)
        sm = a.groupby("seed")["AUC"].mean()
        rows.append({
            "arm": arm, "n_seeds": int(a["seed"].nunique()),
            "per_seed_means": ";".join(f"{s}:{m:.3f}" for s, m in sm.items()),
            "mean_AUC": round(float(a["AUC"].mean()), 3),
            "SD_folds": round(float(a["AUC"].std(ddof=1)), 3),
            "seed_to_seed_SD": round(float(sm.std(ddof=1)), 3),
            "mean_PR_AUC": round(float(a["PR_AUC"].mean()), 3),
        })
    summary = pd.DataFrame(rows)
    out = PROJECT_ROOT / "outputs/R8_resolution_summary_5seed.csv"
    summary.to_csv(out, index=False)
    print("\n===== R8 RESOLUTION (all seeds) =====", flush=True)
    print(summary.to_string(index=False), flush=True)
    if len(summary) == 2:
        d = summary.set_index("arm")
        gap = d.loc["10m", "mean_AUC"] - d.loc["30m", "mean_AUC"]
        spread = max(d.loc["30m", "seed_to_seed_SD"], d.loc["10m", "seed_to_seed_SD"])
        verdict = "REAL (exceeds seed spread)" if abs(gap) > spread else "WITHIN NOISE"
        print(f"\n10m-30m gap = {gap:+.3f}; max seed-to-seed SD = {spread:.3f}; VERDICT: {verdict}", flush=True)
    print(f"saved: {out}", flush=True)


if __name__ == "__main__":
    main()
