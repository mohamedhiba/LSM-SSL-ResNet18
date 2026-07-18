#!/usr/bin/env python
"""Kentucky KGS six-county: 5-method SSL-vs-scratch multi-seed spatial-CV, ps64 (GPU driver).

Methods (the paper comparison, matching the NYC table):
  scratch        random-init encoder + uniform LR 1e-4
  masked_recon   masked reconstruction (0.5 ratio, 4x4 blocks)
  cross_channel  cross-channel masking (zero one terrain channel, predict it; idx 12=TWI)
  strip_jigsaw   3-strip permutation (6 classes)
  sequential     masked_recon warm-up -> cross_channel (carry encoder)

All SSL-pretrained encoders finetune at the STABLE LR (enc 5e-6 / head 2e-5); scratch at
1e-4 (the SSL LR starves a from-scratch net). "Apple-to-apple" = same 27,594-sample index,
same 5 spatial clusters, same seeds, same 14-ch input; each method at its own working LR.
Per seed the 4 SSL methods SHARE one unlabeled index + channel stats. Headline = each SSL
method minus scratch, mean +/- SD over >=3 seeds (THE RULE).

Runs NATIVELY on the KY rasters (pulled to the pod from Drive) OR in --cache-mode from
shipped .npz caches. Report is combined across seeds by scripts/combine_ky_results.py.

  python scripts/run_kentucky_gpu.py --seeds 42 123 7 \
      --methods scratch masked_recon cross_channel strip_jigsaw sequential \
      --raster-dir /workspace/ky/dl/processed/rasters_cleaned_10m \
      --labeled-index .../labeled_patch_index_ps64_kentucky_kgs6c_1to2.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RASTER = Path("/Users/mohamedhiba/Projects/processed/rasters_cleaned_10m")
DEFAULT_INDEX = PROJECT_ROOT / "data/processed/patches/labeled_patch_index_ps64_kentucky_kgs6c_1to2.csv"
DEFAULT_PERM_BANK = PROJECT_ROOT / "data/processed/ssl_pretext_configs/strip_jigsaw_permutation_bank_ps32_strips3_K6.csv"
ALL_METHODS = ["scratch", "masked_recon", "cross_channel", "strip_jigsaw", "sequential"]
SSL_METHODS = {"masked_recon", "cross_channel", "strip_jigsaw", "sequential"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raster-dir", type=Path, default=DEFAULT_RASTER)
    p.add_argument("--labeled-index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--perm-bank", type=Path, default=DEFAULT_PERM_BANK)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--masked-channel-index", type=int, default=12, help="12 = twi_dinf in KY")
    p.add_argument("--n-unlabeled", type=int, default=20000)
    p.add_argument("--ssl-epochs", type=int, default=50)
    p.add_argument("--ssl-batch-size", type=int, default=64)
    p.add_argument("--ft-max-epochs", type=int, default=60)
    p.add_argument("--ft-patience", type=int, default=12)
    p.add_argument("--ft-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 for native rasters (not fork-safe); >0 OK in --cache-mode.")
    p.add_argument("--cache-mode", action="store_true",
                   help="Serve patches from shipped caches via colab_patch; raster_dir=__virtual__.")
    p.add_argument("--colab-dir", type=Path, default=None)
    p.add_argument("--labeled-cache", type=Path, default=None)
    p.add_argument("--unlabeled-cache", type=Path, default=None)
    p.add_argument("--unlabeled-index", type=Path, default=None,
                   help="fixed shipped unlabeled index (cache-mode; shared across seeds)")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs/KY_kgs6c_ps64")
    p.add_argument("--cv-mode", choices=["scv", "flat"], default="scv",
                   help="scv = spatial CV by cluster_id; flat = random StratifiedKFold-5 (Qianyi's fix for clustered positives)")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _split_loaders(ds, seed, args, device):
    import torch
    from torch.utils.data import DataLoader, random_split
    ntr = int(0.9 * len(ds)); nva = len(ds) - ntr
    tr, va = random_split(ds, [ntr, nva], generator=torch.Generator().manual_seed(seed))
    pin = device.type == "cuda"
    return (DataLoader(tr, batch_size=args.ssl_batch_size, shuffle=True, num_workers=args.num_workers,
                       pin_memory=pin, drop_last=True),
            DataLoader(va, batch_size=args.ssl_batch_size, shuffle=False, num_workers=args.num_workers,
                       pin_memory=pin), ntr, nva)


def prepare_ssl(args, device, seed):
    """Per-seed shared unlabeled index + channel stats (used by all 4 SSL methods)."""
    from src.ssl_cross_channel import (CrossChannelMaskRasterDataset,
                                        compute_ssl_channel_stats, create_unlabeled_patch_index)
    from src.patch_dataset import DEFAULT_NODATA_VALUE
    from src.utils import ensure_dir
    if args.cache_mode:
        ul_csv = args.unlabeled_index
    else:
        ul_csv = ensure_dir(args.out_dir / "_ssl") / f"ky_unlabeled_ps{args.patch_size}_n{args.n_unlabeled}_s{seed}.csv"
        create_unlabeled_patch_index(raster_dir=args.raster_dir, output_csv=ul_csv,
                                     patch_size=args.patch_size, n_patches=args.n_unlabeled,
                                     nodata_value=DEFAULT_NODATA_VALUE, random_seed=seed, center_only=False)
    raw = CrossChannelMaskRasterDataset(ul_csv, args.raster_dir, args.patch_size, normalize=False)
    means, stds = compute_ssl_channel_stats(raw, sample_size=min(5000, args.n_unlabeled),
                                            batch_size=args.ssl_batch_size, random_seed=seed)
    raw.close()
    return ul_csv, means, stds


def pretrain(method, args, device, ckpt_dir, seed, ul_csv, means, stds, in_channels):
    import torch
    from torch import nn
    from src.ssl_cross_channel import CrossChannelMaskRasterDataset, CrossChannelModel
    from src.ssl_masked_recon import MaskedReconstructionModel
    from src.ssl_strip_jigsaw import StripJigsawRasterPatchDataset, StripJigsawResNet18Model, STRIP_PERMUTATIONS
    from src.train_ssl import (train_masked_reconstruction_model, train_cross_channel_model,
                               train_strip_jigsaw_model)
    from src.utils import ensure_dir
    ck = ensure_dir(ckpt_dir / method)
    enc = ck / f"{method}_s{seed}_encoder_best.pt"
    n_terrain = in_channels - 1
    common = dict(full_model_best_path=ck / "full_best.pt", encoder_best_path=enc,
                  last_checkpoint_path=ck / "last.pt", channel_means=means, channel_stds=stds,
                  max_epochs=args.ssl_epochs, early_stopping_patience=args.ssl_epochs + 1, grad_clip_norm=5.0)

    # cache_in_memory=True is ESSENTIAL in native mode (else every epoch re-reads
    # rasters = I/O crawl). It is inert/safe in cache-mode (monkeypatch replaces
    # _read_raw_patch, so _build_cache is never triggered).
    if method == "strip_jigsaw":
        ds = StripJigsawRasterPatchDataset(ul_csv, args.raster_dir, args.patch_size, normalize=True,
                                           channel_means=means, channel_stds=stds, random_seed=seed,
                                           cache_in_memory=True)
    else:
        ds = CrossChannelMaskRasterDataset(ul_csv, args.raster_dir, args.patch_size, normalize=True,
                                           channel_means=means, channel_stds=stds, with_mask=True,
                                           cache_in_memory=True)
    tl, vl, ntr, nva = _split_loaders(ds, seed, args, device)
    print(f"  [{method} s{seed}] {args.ssl_epochs} ep, train/val={ntr}/{nva}", flush=True)

    if method == "masked_recon":
        model = MaskedReconstructionModel(in_channels=in_channels, out_channels=n_terrain).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        train_masked_reconstruction_model(model=model, train_loader=tl, val_loader=vl, optimizer=opt,
            device=device, config={"task": "masked_recon", "seed": seed}, mask_ratio=0.5, block_size=4, **common)
    elif method == "cross_channel":
        model = CrossChannelModel(in_channels=in_channels, out_channels=1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        train_cross_channel_model(model=model, train_loader=tl, val_loader=vl, optimizer=opt, device=device,
            config={"task": "cross_channel", "seed": seed, "masked_channel_index": args.masked_channel_index},
            masked_channel_index=args.masked_channel_index, mask_channel_present=True, **common)
    elif method == "strip_jigsaw":
        n_cls = len(STRIP_PERMUTATIONS)
        model = StripJigsawResNet18Model(in_channels=in_channels, n_permutation_classes=n_cls).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        train_strip_jigsaw_model(model=model, train_loader=tl, val_loader=vl, criterion=nn.CrossEntropyLoss(),
            optimizer=opt, device=device, config={"task": "strip_jigsaw", "seed": seed, "n_permutation_classes": n_cls},
            permutation_bank_path=args.perm_bank, batch_size=args.ssl_batch_size, n_permutation_classes=n_cls, **common)
    elif method == "sequential":
        # Stage 1 masked_recon warm-up
        s1 = MaskedReconstructionModel(in_channels=in_channels, out_channels=n_terrain).to(device)
        s1_opt = torch.optim.AdamW(s1.parameters(), lr=1e-4, weight_decay=1e-4)
        s1_enc = ck / f"seq_stage1_s{seed}_encoder_best.pt"
        train_masked_reconstruction_model(model=s1, train_loader=tl, val_loader=vl, optimizer=s1_opt,
            device=device, full_model_best_path=ck / "seq_s1_full.pt", encoder_best_path=s1_enc,
            last_checkpoint_path=ck / "seq_s1_last.pt", channel_means=means, channel_stds=stds,
            config={"task": "masked_recon", "stage": 1, "seed": seed}, max_epochs=30,
            early_stopping_patience=31, mask_ratio=0.5, block_size=4, grad_clip_norm=5.0)  # NYC: 30ep stage1
        # Stage 2 cross_channel, carrying the Stage-1 encoder (lower LR 2e-5)
        s2 = CrossChannelModel(in_channels=in_channels, out_channels=1).to(device)
        s2.encoder.load_state_dict(torch.load(s1_enc, map_location=device)["encoder_state_dict"], strict=True)
        s2_opt = torch.optim.AdamW(s2.parameters(), lr=2e-5, weight_decay=1e-4)
        train_cross_channel_model(model=s2, train_loader=tl, val_loader=vl, optimizer=s2_opt, device=device,
            config={"task": "cross_channel", "stage": 2, "seed": seed, "masked_channel_index": args.masked_channel_index,
                    "stage1_task": "masked_recon"}, masked_channel_index=args.masked_channel_index,
            mask_channel_present=True, **{**common, "max_epochs": 20, "early_stopping_patience": 21})  # NYC: 20ep stage2
    ds.close()
    return enc


def make_scratch_encoder(in_channels, ckpt_path, seed):
    import torch
    from src.ssl_cross_channel import CrossChannelModel
    from src.utils import set_global_seed
    set_global_seed(seed)
    m = CrossChannelModel(in_channels=in_channels, out_channels=1)
    torch.save({"encoder_state_dict": m.encoder.state_dict()}, ckpt_path)
    return ckpt_path


def main() -> None:
    args = parse_args()
    import numpy as np, pandas as pd
    from src.patch_dataset import audit_raster_alignment, list_raster_files
    from src.train_finetune import run_pretrained_resnet18_scv_experiment
    from src.utils import ensure_dir, get_device

    if args.smoke:
        args.n_unlabeled = 200; args.ssl_epochs = 2; args.ft_max_epochs = 2; args.ft_patience = 2

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
    print(f"device: {device}  methods: {args.methods}", flush=True)
    if args.cache_mode:
        import src.patch_dataset as _pdm
        raster_files = _pdm.list_raster_files(args.raster_dir)
    else:
        raster_files = list_raster_files(args.raster_dir)
        audit_raster_alignment(raster_files)
    in_channels = len(raster_files) + 1  # 13 terrain + valid-context mask = 14
    ckpt_dir = ensure_dir(args.out_dir / "_ckpt")
    fig_dir = ensure_dir(args.out_dir / "_fig")
    ssl_wanted = [m for m in args.methods if m in SSL_METHODS]

    rows: list[dict] = []
    for seed in args.seeds:
        ul_csv = means = stds = None
        if ssl_wanted:
            print(f"\n===== prepare shared SSL data seed {seed} =====", flush=True)
            ul_csv, means, stds = prepare_ssl(args, device, seed)
        for method in args.methods:
            if method == "scratch":
                enc = make_scratch_encoder(in_channels, ckpt_dir / f"scratch_enc_s{seed}.pt", seed)
                enc_lr, head_lr, ft_ep, ft_pat = 1e-4, 1e-4, 100, 15   # NYC: scratch 100ep/patience15
            else:
                print(f"\n===== SSL pretrain {method} seed {seed} =====", flush=True)
                enc = pretrain(method, args, device, ckpt_dir, seed, ul_csv, means, stds, in_channels)
                enc_lr, head_lr, ft_ep, ft_pat = 5e-6, 2e-5, 60, 12    # NYC: SSL 60ep/patience12
            print(f"\n===== finetune {method} seed {seed} (enc_lr={enc_lr}, head_lr={head_lr}) =====", flush=True)
            res = run_pretrained_resnet18_scv_experiment(
                project_root=PROJECT_ROOT, model_name=f"ky_{method}", pretraining=method,
                patch_index_csv=args.labeled_index, raster_dir=args.raster_dir,
                encoder_checkpoint_path=enc,
                output_root=args.out_dir / method / f"seed{seed}",
                figure_root=fig_dir / method / f"seed{seed}",
                checkpoint_dir=ckpt_dir / method / f"seed{seed}_ft",
                patch_size=args.patch_size, random_seed=seed,
                batch_size=args.ft_batch_size, encoder_learning_rate=enc_lr, head_learning_rate=head_lr,
                max_epochs=ft_ep, early_stopping_patience=ft_pat,
                expected_checkpoint_keys={"encoder_state_dict"},
                num_workers=args.num_workers, device=device, plot_figures=False,
                cache_in_memory=True, with_mask=True, cv_mode=args.cv_mode,
            )
            fm = res["fold_metrics"]
            mean_auc = float(fm["auc"].mean())
            rows.append({"method": method, "seed": seed, "mean_fold_auc": mean_auc,
                         "fold_aucs": ";".join(f"{a:.4f}" for a in fm["auc"].astype(float))})
            pd.DataFrame(rows).to_csv(args.out_dir / "ky_results.csv", index=False)
            print(f"  {method} seed {seed}: mean-fold AUC = {mean_auc:.4f}", flush=True)

    # ---- summary: per method mean +/- SD over seeds, delta vs scratch ----
    r = pd.DataFrame(rows)
    print("\n===== KENTUCKY KGS6C ps64 SUMMARY (mean +/- SD over seeds) =====", flush=True)
    summ = {}
    for method in args.methods:
        a = r.loc[r["method"] == method, "mean_fold_auc"]
        summ[method] = (float(a.mean()), float(a.std(ddof=0)))
        print(f"  {method:14s} AUC = {a.mean():.4f} +/- {a.std(ddof=0):.4f}  (n={len(a)})", flush=True)
    if "scratch" in summ:
        sm, ssd = summ["scratch"]
        for method in args.methods:
            if method == "scratch":
                continue
            d = summ[method][0] - sm
            sd = max(ssd, summ[method][1])
            verdict = "within noise" if abs(d) < 2 * sd else ("SSL helps" if d > 0 else "SSL hurts")
            print(f"  {method} - scratch = {d:+.4f}  (2x max seed-SD {2*sd:.4f}) -> {verdict}", flush=True)
    print(f"\nwrote {args.out_dir / 'ky_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
