#!/usr/bin/env python
"""R8 scratch baseline: from-scratch ResNet-18 at 30m vs 10m (no SSL).

Completes the {30m,10m} x {scratch, SSL} 2x2 for the resolution question. Reuses
the trusted R5 scratch-SCV training loop (random init, stable LR, 5-cluster spatial
CV). Both setups: 13 terrain channels (landcover excluded), with_mask=False (the
mask is a documented no-op on all-valid patches), on the SAME matched 341-sample
indices as the R8 SSL arms. Seeds 42/123/7.

Output: outputs/R8_resolution_scratch/r8_scratch.csv  (setup, seed, fold, AUC, PR_AUC)
        outputs/R8_resolution_2x2_summary.csv          (combined 2x2 vs SSL)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

SEEDS = [42, 123, 7]
N_FOLDS = 5
# Proper from-scratch LR (canonical notebook-05 scratch used lr=1e-4). The SSL
# fine-tune LR (2e-5/5e-6) is for adapting a PRETRAINED encoder and starves a
# from-scratch net (floored it below random). Scratch uses a uniform 1e-4; more
# epochs/patience since random init trains slower.
MAX_EPOCHS, PATIENCE, BATCH, VALFRAC = 100, 15, 16, 0.2
ENC_LR, HEAD_LR, WD, DROPOUT, CLIP = 1e-4, 1e-4, 1e-4, 0.4, 5.0
SCRATCH_LR_NOTE = "scratch LR=1e-4 (method-appropriate; SSL arms use 2e-5)"


def main() -> None:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import DataLoader, Subset

    import _r8_resolution as r8
    from src.patch_dataset import RasterPatchDataset
    from src.metrics import compute_binary_metrics, find_best_f1_threshold
    from src.models_resnet18 import create_resnet18_binary_classifier
    from src.train_finetune import evaluate_model, train_scv_fold
    from src.utils import ensure_dir, get_device, set_global_seed

    setups = [
        {"name": "30m", "raster_dir": r8.RASTER_30M, "labeled": r8.LABELED_30M_MATCHED},
        {"name": "10m", "raster_dir": r8.RASTER_10M, "labeled": r8.LABELED_10M},
    ]
    out_dir = ensure_dir(PROJECT_ROOT / "outputs/R8_resolution_scratch")
    ckpt_dir = ensure_dir(out_dir / "_ckpt")
    summary_csv = out_dir / "r8_scratch.csv"
    device = get_device()
    print(f"device: {device}", flush=True)
    pin = device.type == "cuda"
    ps = 32

    def stats(labeled, rdir, idx):
        ds = RasterPatchDataset(labeled, rdir, ps, nodata_value=-9999, normalize=False,
                                return_metadata=False, valid_only=True, cache_in_memory=True, with_mask=False)
        loader = DataLoader(Subset(ds, list(idx)), batch_size=32, shuffle=False, num_workers=0)
        csum = csq = None; npix = 0
        for X, _ in loader:
            X = X.float()
            csum = X.sum(dim=(0, 2, 3)) if csum is None else csum + X.sum(dim=(0, 2, 3))
            csq = (X**2).sum(dim=(0, 2, 3)) if csq is None else csq + (X**2).sum(dim=(0, 2, 3))
            npix += X.shape[0] * X.shape[2] * X.shape[3]
        ds.close()
        m = (csum / npix).numpy().astype("float32")
        s = np.sqrt(np.maximum((csq / npix).numpy() - m**2, 1e-12)).astype("float32")
        return m, s

    rows: list[dict] = []
    for setup in setups:
        labeled = setup["labeled"]; rdir = setup["raster_dir"]
        pi = pd.read_csv(labeled).reset_index(drop=True)
        in_ch = 13
        print(f"\n===== scratch {setup['name']} (in_channels={in_ch}, n={len(pi)}) =====", flush=True)
        for seed in SEEDS:
            set_global_seed(seed)
            for fold in range(N_FOLDS):
                tmask = pi["cluster_id"].astype(int) == fold
                cand = pi.index[~tmask].to_numpy(); test_idx = pi.index[tmask].to_numpy()
                tr, va = train_test_split(cand, test_size=VALFRAC, random_state=seed + fold,
                                          stratify=pi.loc[cand, "label"].to_numpy())
                tr = np.asarray(tr, int); va = np.asarray(va, int); test_idx = np.asarray(test_idx, int)
                m, s = stats(labeled, rdir, tr)
                ds = RasterPatchDataset(labeled, rdir, ps, nodata_value=-9999, normalize=True,
                                        channel_means=m, channel_stds=s, return_metadata=True,
                                        valid_only=True, cache_in_memory=True, with_mask=False)
                tl = DataLoader(Subset(ds, tr.tolist()), batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=pin)
                vl = DataLoader(Subset(ds, va.tolist()), batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=pin)
                el = DataLoader(Subset(ds, test_idx.tolist()), batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=pin)
                model = create_resnet18_binary_classifier(in_channels=in_ch, dropout=DROPOUT,
                                                          small_patch_stem=True, pretrained=False).to(device)
                enc_p = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
                head_p = [p for n, p in model.named_parameters() if n.startswith("fc.")]
                opt = torch.optim.AdamW([{"params": enc_p, "lr": ENC_LR}, {"params": head_p, "lr": HEAD_LR}], weight_decay=WD)
                crit = nn.BCEWithLogitsLoss()
                ck = ckpt_dir / f"scratch_{setup['name']}_s{seed}_f{fold}.pt"
                train_scv_fold(model=model, train_loader=tl, val_loader=vl, criterion=crit, optimizer=opt,
                               device=device, checkpoint_path=ck, max_epochs=MAX_EPOCHS,
                               early_stopping_patience=PATIENCE, monitor_metric="val_auc", grad_clip_norm=CLIP)
                tr_res = evaluate_model(model, el, crit, device)
                mm = compute_binary_metrics(tr_res["y_true"], tr_res["y_probs"], threshold=0.5)
                rows.append({"setup": setup["name"], "seed": seed, "fold": fold,
                             "AUC": float(mm["auc"]), "PR_AUC": float(mm["pr_auc"])})
                pd.DataFrame(rows).to_csv(summary_csv, index=False)
                print(f"  scratch {setup['name']} seed {seed} fold {fold}: AUC={mm['auc']:.4f}", flush=True)
                ds.close()

    # ---- combined 2x2 summary (scratch vs SSL, 30m vs 10m) ----
    scr = pd.DataFrame(rows)
    out_rows = []
    for setup in ("30m", "10m"):
        a = scr.loc[scr["setup"] == setup]
        sm = a.groupby("seed")["AUC"].mean()
        out_rows.append({"arm": f"scratch_{setup}", "mean_AUC": round(float(a["AUC"].mean()), 3),
                         "seed_to_seed_SD": round(float(sm.std(ddof=1)), 3),
                         "mean_PR_AUC": round(float(a["PR_AUC"].mean()), 3), "n_seeds": int(a["seed"].nunique())})
    # pull SSL arms from the 3-seed R8 csvs
    for setup in ("30m", "10m"):
        p = PROJECT_ROOT / f"outputs/R8_resolution_{setup}/r8_{setup}.csv"
        if p.exists():
            a = pd.read_csv(p); sm = a.groupby("seed")["AUC"].mean()
            out_rows.append({"arm": f"ssl_{setup}", "mean_AUC": round(float(a["AUC"].mean()), 3),
                             "seed_to_seed_SD": round(float(sm.std(ddof=1)), 3),
                             "mean_PR_AUC": round(float(a["PR_AUC"].mean()), 3), "n_seeds": int(a["seed"].nunique())})
    summary = pd.DataFrame(out_rows)
    out = PROJECT_ROOT / "outputs/R8_resolution_2x2_summary.csv"
    summary.to_csv(out, index=False)
    print("\n===== R8 2x2: resolution x (scratch/SSL) =====", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"saved: {out}", flush=True)


if __name__ == "__main__":
    main()
