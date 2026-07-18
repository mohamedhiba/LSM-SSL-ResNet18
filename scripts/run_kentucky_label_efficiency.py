#!/usr/bin/env python
"""Kentucky label-efficiency curve: scratch vs SSL at shrinking TRAINING-label fractions.

The SSL claim is that pretraining helps most when labels are SCARCE. The full-label KY
run showed SSL ~= scratch; this sweeps the training-label budget to see if the SSL-scratch
gap opens at the low-label end.

Design (per seed):
  1. Pretrain each SSL encoder ONCE on the full unlabeled pool (reused across all fractions).
  2. For each method and each fraction f in {0.05, 0.1, 0.25, 0.5, 1.0}, run 5-fold spatial
     CV finetune with train_label_fraction=f: TEST folds stay full, only the per-fold
     TRAINING labels are stratified-subsampled (seeded). Val/test untouched.
Reuses prepare_ssl / pretrain / make_scratch_encoder from run_kentucky_gpu.py so the
pretraining, LRs and cache-mode plumbing are identical to the main KY run.

  python scripts/run_kentucky_label_efficiency.py --cache-mode \
    --methods scratch masked_recon --seeds 42 123 7 \
    --fractions 0.05 0.1 0.25 0.5 1.0 \
    --colab-dir PKG/colab --labeled-index ... --labeled-cache ... \
    --unlabeled-index ... --unlabeled-cache ... --out-dir /workspace/ky/le
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_kentucky_gpu as rkg  # noqa: E402  (prepare_ssl / pretrain / make_scratch_encoder)

ALL_METHODS = rkg.ALL_METHODS
SSL_METHODS = rkg.SSL_METHODS
DEFAULT_INDEX = rkg.DEFAULT_INDEX
DEFAULT_RASTER = rkg.DEFAULT_RASTER
DEFAULT_PERM_BANK = rkg.DEFAULT_PERM_BANK


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raster-dir", type=Path, default=DEFAULT_RASTER)
    p.add_argument("--labeled-index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--perm-bank", type=Path, default=DEFAULT_PERM_BANK)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--methods", nargs="+", default=["scratch", "masked_recon"], choices=ALL_METHODS)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--masked-channel-index", type=int, default=12)
    p.add_argument("--n-unlabeled", type=int, default=20000)
    p.add_argument("--ssl-epochs", type=int, default=30)
    p.add_argument("--ssl-batch-size", type=int, default=64)
    p.add_argument("--ft-max-epochs", type=int, default=60)
    p.add_argument("--ft-patience", type=int, default=12)
    p.add_argument("--ft-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cache-mode", action="store_true")
    p.add_argument("--colab-dir", type=Path, default=None)
    p.add_argument("--labeled-cache", type=Path, default=None)
    p.add_argument("--unlabeled-cache", type=Path, default=None)
    p.add_argument("--unlabeled-index", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs/KY_label_efficiency")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np, pandas as pd
    from src.patch_dataset import audit_raster_alignment, list_raster_files
    from src.train_finetune import run_pretrained_resnet18_scv_experiment
    from src.utils import ensure_dir, get_device

    if args.smoke:
        args.n_unlabeled = 200; args.ssl_epochs = 2; args.ft_max_epochs = 2; args.ft_patience = 2
        args.fractions = [0.1, 1.0]

    if args.cache_mode:
        if args.colab_dir:
            sys.path.insert(0, str(args.colab_dir))
        import colab_patch
        colab_patch.apply()
        colab_patch.register_cache(args.labeled_index, np.load(args.labeled_cache)["patches"])
        colab_patch.register_cache(args.unlabeled_index, np.load(args.unlabeled_cache)["patches"])
        args.raster_dir = "__virtual__"
        print("[cache-mode] registered caches; raster_dir=__virtual__", flush=True)

    device = get_device()
    print(f"device: {device}  methods: {args.methods}  fractions: {args.fractions}", flush=True)
    if args.cache_mode:
        import src.patch_dataset as _pdm
        raster_files = _pdm.list_raster_files(args.raster_dir)
    else:
        raster_files = list_raster_files(args.raster_dir)
        audit_raster_alignment(raster_files)
    in_channels = len(raster_files) + 1
    ckpt_dir = ensure_dir(args.out_dir / "_ckpt")
    fig_dir = ensure_dir(args.out_dir / "_fig")
    ssl_wanted = [m for m in args.methods if m in SSL_METHODS]

    rows: list[dict] = []
    for seed in args.seeds:
        ul_csv = means = stds = None
        if ssl_wanted:
            print(f"\n===== prepare shared SSL data seed {seed} =====", flush=True)
            ul_csv, means, stds = rkg.prepare_ssl(args, device, seed)

        # pretrain each SSL encoder ONCE per seed (reused across all fractions)
        encoders: dict[str, Path] = {}
        for method in args.methods:
            if method == "scratch":
                encoders[method] = rkg.make_scratch_encoder(
                    in_channels, ckpt_dir / f"scratch_enc_s{seed}.pt", seed)
            else:
                print(f"\n===== SSL pretrain {method} seed {seed} (once) =====", flush=True)
                encoders[method] = rkg.pretrain(method, args, device, ckpt_dir, seed,
                                                ul_csv, means, stds, in_channels)

        for method in args.methods:
            if method == "scratch":
                enc_lr, head_lr, ft_ep, ft_pat = 1e-4, 1e-4, 100, 15
            else:
                enc_lr, head_lr, ft_ep, ft_pat = 5e-6, 2e-5, args.ft_max_epochs, args.ft_patience
            for frac in args.fractions:
                tag = f"{method}_s{seed}_f{frac}"
                print(f"\n===== finetune {tag} (train_label_fraction={frac}) =====", flush=True)
                res = run_pretrained_resnet18_scv_experiment(
                    project_root=PROJECT_ROOT, model_name=f"ky_le_{method}", pretraining=method,
                    patch_index_csv=args.labeled_index, raster_dir=args.raster_dir,
                    encoder_checkpoint_path=encoders[method],
                    output_root=args.out_dir / method / f"seed{seed}" / f"f{frac}",
                    figure_root=fig_dir / method / f"seed{seed}" / f"f{frac}",
                    checkpoint_dir=ckpt_dir / method / f"seed{seed}_f{frac}_ft",
                    patch_size=args.patch_size, random_seed=seed,
                    train_label_fraction=frac,
                    batch_size=args.ft_batch_size, encoder_learning_rate=enc_lr,
                    head_learning_rate=head_lr, max_epochs=ft_ep, early_stopping_patience=ft_pat,
                    expected_checkpoint_keys={"encoder_state_dict"},
                    num_workers=args.num_workers, device=device, plot_figures=False,
                    cache_in_memory=True, with_mask=True,
                )
                fm = res["fold_metrics"]
                mean_auc = float(fm["auc"].mean())
                n_tr = int(fm["n_train"].mean()) if "n_train" in fm.columns else -1
                rows.append({"method": method, "seed": seed, "fraction": frac,
                             "mean_train_n": n_tr, "mean_fold_auc": mean_auc,
                             "fold_aucs": ";".join(f"{a:.4f}" for a in fm["auc"].astype(float))})
                pd.DataFrame(rows).to_csv(args.out_dir / "ky_label_efficiency.csv", index=False)
                print(f"  {tag}: mean-fold AUC = {mean_auc:.4f} (train n~{n_tr})", flush=True)

    # ---- summary: per fraction, scratch vs each SSL method ----
    r = pd.DataFrame(rows)
    print("\n===== KY LABEL-EFFICIENCY SUMMARY (mean AUC over seeds) =====", flush=True)
    for frac in args.fractions:
        rf = r[r["fraction"] == frac]
        print(f"\n-- fraction {frac} (train n~{int(rf['mean_train_n'].mean())}) --", flush=True)
        summ = {}
        for method in args.methods:
            a = rf.loc[rf["method"] == method, "mean_fold_auc"]
            if len(a):
                summ[method] = (float(a.mean()), float(a.std(ddof=0)))
                print(f"  {method:14s} AUC = {a.mean():.4f} +/- {a.std(ddof=0):.4f}", flush=True)
        if "scratch" in summ:
            sm, ssd = summ["scratch"]
            for method in args.methods:
                if method == "scratch" or method not in summ:
                    continue
                d = summ[method][0] - sm
                sd = max(ssd, summ[method][1])
                verdict = "within noise" if abs(d) < 2 * sd else ("SSL helps" if d > 0 else "SSL hurts")
                print(f"    {method} - scratch = {d:+.4f} (2x SD {2*sd:.4f}) -> {verdict}", flush=True)
    print(f"\nwrote {args.out_dir / 'ky_label_efficiency.csv'}", flush=True)


if __name__ == "__main__":
    main()
