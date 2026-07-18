# KY ↔ Qianyi dual-view protocol alignment (2026-07-14)

Goal: make our KY runs directly comparable to Qianyi's final 10 m dual-view pipeline
(his directive 2026-07-13), then generate LSM maps. **No training yet** — this documents
the verified spec, what is implemented, and what remains.

## 1. The verified protocol (from his code, NOT the Slack paraphrase)

Extracted from `LSM_SSL_ResNet18_10m_dual_view_code/` with file:line citations
(see CLAUDE.md 07-14 log for the diff-vs-Slack summary):

| Item | Value | Where |
|---|---|---|
| Views | dual, local 15×15 + global 31×31, same center, shared encoder | `patch_dataset.py:1117`, `models_resnet18.py:429` |
| Input | 13 factors + valid-context mask = 14 ch; mask NOT normalized, re-binarized >0.5 | `patch_dataset.py:1451-1466` |
| Normalization | block-wise whole-raster mean/std per factor, (x−μ)/(σ+1e-6), factors only | `prepare_10m_patch_indices.py:440-482` |
| Fusion | `logit = local + 0.2 × global_correction`, fixed alpha | `models_resnet18.py:411-452` |
| Masked recon | ratio 0.5 both views; blocks 3×3 local / 4×4 global; valid-context-only; reconstruct 13 factor ch; MSE on masked px; λ_global=1 | `ssl_masked_recon.py:432-531` |
| SSL corpus | **20,000** unlabeled (18k/2k split). ⚠ 50k = PU candidate pool, NOT SSL | `final_ssl_runner.py:480`; `final_pu_mean_oob_protocol.py:59` |
| SSL hyper | AdamW wd 1e-4, batch 64, ≤50 ep, patience 10; LR **1e-4 masked-recon**, 1e-3 others | `final_ssl_runner.py:42-51` |
| Downstream | head LR **1e-3**; encoder-LR grid **[0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]** (⚠ tops at 1e-4, not 1e-3); batch 128; ≤100 ep patience 15; BCE `pos_weight`; AdamW wd 1e-4 | `final_flat_cv_protocol.py:32-33`, `final_flat_cv_lr_sweep_runner.py:39-88` |
| LR=0 semantics | freezes weights only; **BN running stats still update** unless `strict_frozen_encoder=True` (default False; separate strict pilot exists) | `final_flat_cv_lr_sweep_runner.py:74,1205` |
| Flat CV | StratifiedKFold(5, shuffle, random_state=42) on label; non-nested by design | `final_flat_cv_protocol.py:32-36` |
| Seeds | **[42, 43, 44, 45, 46]** (5 seeds) | `final_protocol.py:23` |
| Labeled set | positives + PU mean-OOB negatives 1:1; rule `mean_oob ≤ 0.5` only; RF 50 iters × 100 trees on 50k candidates, 90 m positive buffer | `final_pu_mean_oob_protocol.py` |
| LSM inference | ensemble ≥20 checkpoints (5 seeds × 5 folds), manifest CSV, mean+std GeoTIFFs | `inference_lsm.py:318-520` |

## 2. KY deltas the shim encodes

- factor 6: `lithology` → **`ksat`** (continuous → his lithology-only categorical logic self-deactivates)
- factor 7 filename: `ndvi_10yr_mean_2015_2024` → **`ndvi_annual_mean_2015_2024`**
- rasters: KY EPSG:3088 stack at `/Users/mohamedhiba/Projects/processed/rasters_cleaned_10m` (`KY_RASTER_DIR` env overrides)
- positives: KY kgs6c (n≈9,198 jittered centers), not NYC's 281 — PU/QA literals generalized to n_pos

## 3. Implemented (this session)

| Piece | File | Status |
|---|---|---|
| Shim (path bootstrap + in-place factor-list patch + KY configs) | `scripts/ky_dual_view_shim.py` | done |
| Stage 10: KY dual-view indices + normalization stats | `scripts/build_ky_dual_view_indices.py` | done; first build running |
| Stage 30: KY PU mean-OOB 1:1 labeled set + group-safe flat-CV split | `scripts/build_ky_pu_labeled.py` | **DONE 07-17, QA passed** — 18,396 (9,198+9,198), split grouped by original_feature_id, zero leaks |
| Stage 10 rerun @ 50k unlabeled (Qianyi's stated corpus) | `..._n50000.csv` | **DONE 07-17, QA passed** |

⚠ Never import this repo's `src/` and his `src/` in one process — the shim inserts his
package at `sys.path[0]`; KY dual-view work happens in dedicated processes.

## 4. Remaining (before any training)

1. **Run stage 30** (`build_ky_pu_labeled.py`; RF on ~9k positives × 50 iters is CPU-heavy —
   consider `--max-positives` or run on the pod).
2. **Run driver**: thin wrapper constructing his `FinalSSLTrainingConfig` +
   `FinalFlatCVLRSweepConfig` with KY paths (indices/stats from `data/kentucky_dual_view/`).
   Note his runners resolve labeled/unlabeled paths via `final_protocol.FinalProtocolPaths` —
   the wrapper must pass KY paths explicitly (or patch those defaults in the shim).
   Full sweep is **625 jobs** (4 SSL × 5 seeds × 5 folds × 6 LRs + 25 scratch); his pilot
   mode (3 jobs) is the smoke test to run first on GPU.
3. **Decide sweep budget**: full 625 KY jobs at ps 15/31 is much cheaper per job than our
   ps64 runs, but still sizeable — consider grid subset {0, 1e-5, 1e-4} × 3 seeds first.
4. **LSM stage 90** afterwards: needs ≥20 downstream checkpoints + manifest CSV.

## 5. Open questions for Qianyi

- Slack said "pretrained on 50,000 unlabeled centers" but the code pretrains on the
  **n20000** index (50k is the PU candidate pool) — which did the final runs use?
- Slack said encoder LR swept "0 to 1e-3" but `ENCODER_LR_GRID` tops at **1e-4** — was
  1e-3 an LR-extension run (`60_run_lr_extension.py`) or a misremember?
- ~~KY positives jittered?~~ **RESOLVED 07-17: they are NOT jittered.** The 9,198 are
  center-valid rows of the kgs_v4 file = **9,242 real, deduplicated KGS inventory records**
  (1,842 original points + 7,400 polygon-derived centroids; the old "1,106 × jitter" set was
  the June dataset, long superseded). Residual grouping is mild — 6,123 original features,
  max 3 cells/feature — and is handled: `build_ky_pu_labeled.py` now emits a
  **StratifiedGroupKFold(5) split grouped by `original_feature_id`** (unmatched positives
  snap to the nearest record ≤50 m, else singleton; negatives singleton), so cells of one
  mapped landslide never straddle train/test. QA asserts zero group leaks.
