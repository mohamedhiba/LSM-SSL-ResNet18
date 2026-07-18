#!/usr/bin/env python
"""R8 - matched 30m-vs-10m resolution comparison (isolating resolution only).

Both arms use the IDENTICAL current pipeline: 13 terrain channels + valid-context
mask (cross_channel_newpipe lineage), cross-channel SSL pretext (mask TWI), and
the stable-LR multi-seed finetune (head 2e-5 / encoder 5e-6, 60 ep, patience 12)
across seeds 42/123/7. The ONLY difference between arms is raster resolution.

- 30m arm: reuses the existing cross_channel_newpipe encoder (pretrained seed 42,
  50 ep, masked idx 12) and the existing 30m labeled/unlabeled indices.
- 10m arm: pretrains a fresh cross-channel encoder on the 10m rasters with the
  SAME protocol, on regenerated 10m patch indices.

Both arms are finetuned on the SAME set of labeled sample_ids (the intersection
of samples valid at both resolutions) so the comparison is clean. Do NOT compare
either arm against the historical 0.782 (different input scheme).

Idempotent: skips any stage whose outputs already exist. Stages:
  1 prep    regenerate 10m labeled + unlabeled patch indices
  2 pretrain  10m cross-channel SSL encoder
  3 finetune  both arms, 3 seeds
  4 report   side-by-side R6/R7-style table

Usage:
    python scripts/_r8_resolution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = [42, 123, 7]
PATCH_SIZE = 32
MASKED_CHANNEL_INDEX = 12  # TWI, alphabetically last of the 13 terrain channels

RASTER_30M = PROJECT_ROOT / "data/processed/rasters_cleaned"
RASTER_10M = PROJECT_ROOT / "data/processed/rasters_cleaned_10m"
PATCH_DIR = PROJECT_ROOT / "data/processed/patches"
UNLAB_DIR = PROJECT_ROOT / "data/processed/ssl_unlabeled_indices"

LABELED_30M = PATCH_DIR / "labeled_patch_index_ps32_common_balanced.csv"
LABELED_30M_MATCHED = PATCH_DIR / "labeled_patch_index_ps32_common_balanced_30m_r8.csv"
LABELED_10M = PATCH_DIR / "labeled_patch_index_ps32_common_balanced_10m.csv"
UNLAB_10M = UNLAB_DIR / "unlabeled_patch_index_ps32_n20000_10m.csv"
ENC_10M_DIR = PROJECT_ROOT / "checkpoints/ssl_pretrained/cross_channel_10m"
ENC_10M = ENC_10M_DIR / "resnet18_cross_channel_ps32_encoder_best.pt"
ENC_30M = (PROJECT_ROOT / "checkpoints/ssl_pretrained/cross_channel_newpipe"
           / "resnet18_cross_channel_ps32_encoder_best.pt")


# ---------------------------------------------------------------------------
# Stage 1 - regenerate 10m patch indices and build the matched labeled set
# ---------------------------------------------------------------------------
def stage_prep() -> None:
    import pandas as pd
    from src.patch_dataset import create_patch_index, list_raster_files, save_patch_index
    from src.ssl_masked_recon import create_unlabeled_patch_index

    print("\n========== STAGE 1: prep 10m indices ==========", flush=True)

    # --- 10m labeled index for the same 344 sample_ids ---
    if not LABELED_10M.exists():
        src_df = pd.read_csv(LABELED_30M)
        raster_files_10m = list_raster_files(RASTER_10M)
        print(f"regenerating 10m labeled windows for {len(src_df)} samples ...", flush=True)
        idx10 = create_patch_index(
            samples_df=src_df,
            raster_files=raster_files_10m,
            patch_size=PATCH_SIZE,
        )
        save_patch_index(idx10, LABELED_10M)
        print(f"  saved {LABELED_10M.name}", flush=True)
    else:
        print(f"  {LABELED_10M.name} exists, skip", flush=True)

    # --- build the matched (intersection-valid) labeled sets for BOTH arms ---
    df30 = pd.read_csv(LABELED_30M)
    df10 = pd.read_csv(LABELED_10M)
    valid30 = set(df30.loc[df30["valid_patch"].astype(bool), "sample_id"].astype(str))
    valid10 = set(df10.loc[df10["valid_patch"].astype(bool), "sample_id"].astype(str))
    matched = sorted(valid30 & valid10)
    dropped = (valid30 | valid10) - set(matched)
    print(f"valid@30m={len(valid30)}  valid@10m={len(valid10)}  matched(intersection)={len(matched)}", flush=True)
    if dropped:
        print(f"  WARNING: {len(dropped)} sample_ids dropped from one arm: {sorted(dropped)[:10]}", flush=True)

    m30 = df30.loc[df30["sample_id"].astype(str).isin(matched)].copy().sort_values("sample_id").reset_index(drop=True)
    m10 = df10.loc[df10["sample_id"].astype(str).isin(matched)].copy().sort_values("sample_id").reset_index(drop=True)
    m30.to_csv(LABELED_30M_MATCHED, index=False)
    m10.to_csv(LABELED_10M, index=False)  # overwrite with matched-only for a clean run
    for name, m in [("30m", m30), ("10m", m10)]:
        bal = m.groupby(["cluster_id", "label"]).size().unstack(fill_value=0)
        print(f"  {name} matched set: n={len(m)}  per-cluster/label:\n{bal.to_string()}", flush=True)

    # --- 10m unlabeled SSL index (interior-only, n=20000, seed 42) ---
    if not UNLAB_10M.exists():
        print("regenerating 10m unlabeled index (n=20000, interior-only) ...", flush=True)
        create_unlabeled_patch_index(
            raster_dir=RASTER_10M,
            output_csv=UNLAB_10M,
            patch_size=PATCH_SIZE,
            n_patches=20000,
            random_seed=42,
            max_attempts=2_000_000,
            center_only=False,
        )
        print(f"  saved {UNLAB_10M.name}", flush=True)
    else:
        print(f"  {UNLAB_10M.name} exists, skip", flush=True)


# ---------------------------------------------------------------------------
# Stage 2 - pretrain the 10m cross-channel encoder (mirrors newpipe protocol)
# ---------------------------------------------------------------------------
def stage_pretrain() -> None:
    print("\n========== STAGE 2: pretrain 10m cross-channel encoder ==========", flush=True)
    if ENC_10M.exists():
        print(f"  {ENC_10M} exists, skip pretrain", flush=True)
        return

    import json
    import torch
    from torch.utils.data import DataLoader, random_split

    from src.patch_dataset import DEFAULT_NODATA_VALUE, audit_raster_alignment, list_raster_files
    from src.ssl_cross_channel import (
        CrossChannelMaskRasterDataset,
        CrossChannelModel,
        compute_ssl_channel_stats,
    )
    from src.train_ssl import train_cross_channel_model
    from src.utils import count_trainable_parameters, ensure_dir, get_device, set_global_seed

    ensure_dir(ENC_10M_DIR)
    out_log_dir = ensure_dir(PROJECT_ROOT / "outputs/SSL_cross_channel_ps32_10m/training_logs")
    set_global_seed(42)
    device = get_device()
    print(f"device: {device}", flush=True)

    raster_files = list_raster_files(RASTER_10M)
    audit_raster_alignment(raster_files, expected_nodata=DEFAULT_NODATA_VALUE)
    print(f"masked channel index {MASKED_CHANNEL_INDEX}: {raster_files[MASKED_CHANNEL_INDEX].name}", flush=True)

    raw = CrossChannelMaskRasterDataset(UNLAB_10M, RASTER_10M, PATCH_SIZE, normalize=False)
    channel_means, channel_stds = compute_ssl_channel_stats(raw, sample_size=5000, batch_size=64, random_seed=42)
    raw.close()
    print(f"channel_means len: {len(channel_means)}", flush=True)

    ssl_ds = CrossChannelMaskRasterDataset(
        UNLAB_10M, RASTER_10M, PATCH_SIZE, normalize=True,
        channel_means=channel_means, channel_stds=channel_stds,
        cache_in_memory=True, with_mask=True,
    )
    train_size = int(0.9 * len(ssl_ds))
    val_size = len(ssl_ds) - train_size
    train_ds, val_ds = random_split(ssl_ds, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"train/val: {train_size}/{val_size}", flush=True)

    model = CrossChannelModel(in_channels=len(raster_files) + 1, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    print(f"trainable params: {count_trainable_parameters(model)}", flush=True)

    config = {
        "task": "cross_channel", "patch_size": PATCH_SIZE, "n_unlabeled_patches": 20000,
        "masked_channel_index": MASKED_CHANNEL_INDEX,
        "masked_channel_raster": raster_files[MASKED_CHANNEL_INDEX].name,
        "batch_size": 64, "learning_rate": 1e-4, "weight_decay": 1e-4,
        "max_epochs": 50, "early_stopping_patience": 10, "gradient_clip_norm": 5.0,
        "random_seed": 42, "with_mask_channel": True, "in_channels": len(raster_files) + 1,
        "resolution": "10m", "device": str(device),
    }
    print(json.dumps(config, indent=2), flush=True)

    log_df, best = train_cross_channel_model(
        model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
        device=device,
        full_model_best_path=ENC_10M_DIR / "resnet18_cross_channel_ps32_full_model_best.pt",
        encoder_best_path=ENC_10M,
        last_checkpoint_path=ENC_10M_DIR / "resnet18_cross_channel_ps32_last.pt",
        config=config, channel_means=channel_means, channel_stds=channel_stds,
        max_epochs=50, early_stopping_patience=10,
        masked_channel_index=MASKED_CHANNEL_INDEX, grad_clip_norm=5.0, mask_channel_present=True,
    )
    ssl_ds.close()
    log_df.to_csv(out_log_dir / "cross_channel_ps32_training_log.csv", index=False)
    print(f"pretrain done: best={best}", flush=True)


# ---------------------------------------------------------------------------
# Stage 3 - finetune both arms, 3 seeds each (stable LR)
# ---------------------------------------------------------------------------
def _finetune_arm(arm: str, raster_dir: Path, labeled_csv: Path, encoder: Path) -> list[dict]:
    import pandas as pd
    from src.train_finetune import run_pretrained_resnet18_scv_experiment
    from src.utils import ensure_dir, get_device, set_global_seed

    out_root = ensure_dir(PROJECT_ROOT / f"outputs/R8_resolution_{arm}")
    device = get_device()
    rows: list[dict] = []
    for seed in SEEDS:
        set_global_seed(seed)
        res = run_pretrained_resnet18_scv_experiment(
            project_root=PROJECT_ROOT,
            model_name=f"cc_{arm}",
            pretraining="cross_channel",
            patch_index_csv=labeled_csv,
            raster_dir=raster_dir,
            encoder_checkpoint_path=encoder,
            output_root=out_root / f"seed{seed}",
            figure_root=PROJECT_ROOT / f"figures/R8_resolution_{arm}/seed{seed}",
            checkpoint_dir=PROJECT_ROOT / f"checkpoints/finetuned/R8_resolution_{arm}",
            patch_size=PATCH_SIZE, random_seed=seed, batch_size=16,
            encoder_learning_rate=5e-6, head_learning_rate=2e-5,
            max_epochs=60, early_stopping_patience=12, dropout=0.4,
            expected_checkpoint_keys={"encoder_state_dict"},
            num_workers=0, device=device, plot_figures=False,
            cache_in_memory=True, with_mask=True,
        )
        fm = res["fold_metrics"]
        sm = res["summary_metrics"].iloc[0]
        for _, r in fm.iterrows():
            rows.append({"arm": arm, "seed": seed, "fold": int(r["fold"]),
                         "AUC": float(r["auc"]), "PR_AUC": float(r["pr_auc"])})
        print(f"  [{arm}] seed {seed}: mean AUC={sm['mean_auc']:.4f} SD={sm['std_auc']:.4f} "
              f"folds={[round(a,3) for a in fm['auc'].tolist()]}", flush=True)
    pd.DataFrame(rows).to_csv(out_root / f"r8_{arm}.csv", index=False)
    return rows


def stage_finetune() -> list[dict]:
    print("\n========== STAGE 3: finetune both arms (3 seeds) ==========", flush=True)
    all_rows: list[dict] = []
    print("\n--- 30m arm (reuse cross_channel_newpipe) ---", flush=True)
    all_rows += _finetune_arm("30m", RASTER_30M, LABELED_30M_MATCHED, ENC_30M)
    print("\n--- 10m arm (fresh 10m encoder) ---", flush=True)
    all_rows += _finetune_arm("10m", RASTER_10M, LABELED_10M, ENC_10M)
    return all_rows


# ---------------------------------------------------------------------------
# Stage 4 - aggregate + report (R6/R7 table format)
# ---------------------------------------------------------------------------
def stage_report(all_rows: list[dict] | None = None) -> None:
    import pandas as pd
    print("\n========== STAGE 4: report ==========", flush=True)
    if all_rows:
        df = pd.DataFrame(all_rows)
    else:
        parts = []
        for arm in ("30m", "10m"):
            p = PROJECT_ROOT / f"outputs/R8_resolution_{arm}/r8_{arm}.csv"
            if p.exists():
                parts.append(pd.read_csv(p))
        df = pd.concat(parts, ignore_index=True)

    summary_rows = []
    for arm in ("30m", "10m"):
        a = df.loc[df["arm"] == arm]
        if a.empty:
            continue
        seed_means = a.groupby("seed")["AUC"].mean()
        summary_rows.append({
            "arm": arm,
            "seed42": round(float(seed_means.get(42, float('nan'))), 3),
            "seed123": round(float(seed_means.get(123, float('nan'))), 3),
            "seed7": round(float(seed_means.get(7, float('nan'))), 3),
            "mean_AUC_15folds": round(float(a["AUC"].mean()), 3),
            "SD_15folds": round(float(a["AUC"].std(ddof=1)), 3),
            "seed_to_seed_SD": round(float(seed_means.std(ddof=1)), 3),
            "mean_PR_AUC": round(float(a["PR_AUC"].mean()), 3),
        })
    summary = pd.DataFrame(summary_rows)
    out = PROJECT_ROOT / "outputs/R8_resolution_summary.csv"
    summary.to_csv(out, index=False)

    print("\n===== R8 RESOLUTION COMPARISON (matched pipeline, 3 seeds) =====", flush=True)
    print(summary.to_string(index=False), flush=True)
    if len(summary) == 2:
        d = summary.set_index("arm")
        gap = d.loc["10m", "mean_AUC_15folds"] - d.loc["30m", "mean_AUC_15folds"]
        spread = max(d.loc["30m", "seed_to_seed_SD"], d.loc["10m", "seed_to_seed_SD"])
        verdict = ("REAL (exceeds seed spread)" if abs(gap) > spread
                   else "WITHIN NOISE (does not exceed seed-to-seed SD) -> not a real difference")
        print(f"\n10m - 30m mean-AUC gap = {gap:+.3f}; max seed-to-seed SD = {spread:.3f}", flush=True)
        print(f"VERDICT: {verdict}", flush=True)
        print("NOTE: do NOT compare to the historical 0.782 (different input scheme).", flush=True)
    print(f"\nsaved: {out}", flush=True)


def main() -> None:
    stage_prep()
    stage_pretrain()
    rows = stage_finetune()
    stage_report(rows)


if __name__ == "__main__":
    main()
