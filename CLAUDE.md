# CLAUDE.md — Read Me First

Self-supervised learning (SSL) for landslide susceptibility mapping (LSM) with a
small-patch ResNet-18. This file is the orientation doc for any fresh session:
mission, the scientific question, how the pipeline works, where things live, what
the results currently say, the rules we work by, and what's next.

For detail beyond this file, see the three existing results docs (don't duplicate
them — extend or reference):

- `README.md` — reproducibility package: install, notebook run order, channel
  scheme, the historical R1 5-task results table, Phase-1 and Phase-2 notes.
- `RESULTS_NOTE.md` — Phase 1: the two new pretext tasks (cross-channel masking,
  strip jigsaw), the finetune-LR collapse story, and the mask-channel ablation (R3).
- `RESULTS_R4_R6_SUMMARY.md` — the variance investigation (R4–R7): why single
  runs aren't trustworthy, the multi-seed rule, and the landcover re-evaluation.

---

## 1. Mission & team

Research at the **CCNY Media Lab** under **Prof. YingLi Tian**.

| Person | Role |
|---|---|
| **Mohamed** (me) | ML pipeline lead — this repo |
| **Te Pei** | Supervisor / collaborator; preparing Kentucky & Colorado feature extraction + Google Earth Engine code |
| **Qianyi Liu (Chen Yi)** | Collaborator modeling related study areas; owns the new 10 m terrain features and the local–global dual-view CNN code (a deferred dependency) |
| **Xiaolu Liu** | Collaborator modeling related study areas |

**The scientific question:** *Can SSL pretraining on terrain data improve
downstream landslide prediction, especially when labeled data is scarce?*

The critical caveat the team keeps front-of-mind: **SSL is only worth using if the
supervised baseline struggles** (small/hard dataset). If a from-scratch model
already does well, SSL has no justification. This is a live concern, not a
settled assumption — see §4.

---

## 2. What the pipeline does (end to end)

Multi-channel 30 m terrain raster patches → ResNet-18 → binary landslide /
non-landslide, evaluated under **spatially-blocked 5-fold cross-validation**
(whole geographic clusters are held out, so there is no spatial leakage between
train and test).

**Data prep** (notebooks 01–04; `src/pu_bagging.py`,
`src/spatial_cluster_balance.py`, `src/raster_cleaning.py`,
`src/patch_dataset.py`):
1. **PU-Bagging** selects reliable non-landslide negatives (positives = landslide
   points). *(Exact bagging hyperparameters are inferred from the notebook, not
   verified line-by-line — see §8.)*
2. **KMeans spatial clustering** builds 5 clusters (`cluster_id` 0–4) and a
   label-balanced sample set. *(KMeans params likewise inferred.)*
3. **Patch index** generation for patch sizes 16/32/64 around each sample point.
4. **Common-valid + per-cluster label balancing** → the canonical downstream set
   **`data/processed/patches/labeled_patch_index_ps32_common_balanced.csv`
   (344 patches, 5 balanced clusters).** Unlabeled patches for SSL live in
   `data/processed/ssl_unlabeled_indices/` (n=500 and n=20000).

**Channel scheme (current code).** 14 GeoTIFFs sit in
`data/processed/rasters_cleaned/`, but `list_raster_files` excludes
`Landcover.tif` *by name* → **13 terrain channels** (alphabetical: aspect,
bulk_density, clay, elevation, field_capacity, lithology, ndvi, plan_curv,
profile_curv, sand, slope, spi, twi). The dataset appends a **14th binary
valid-context mask channel** (1 = real pixel, 0 = zero-padded NoData) →
**14-channel input**. Normalization is mask-aware (stats over valid pixels only).

**Backbone** (`src/models_resnet18.py`): modified ResNet-18 with a **small-patch
stem** (3×3 stride-1 conv1, no maxpool) for 32×32 inputs. `BinaryClassifier`
variant = adaptive avgpool → Dropout(0.4) → Linear(512,1), `BCEWithLogitsLoss`.
`Encoder` variant returns feature maps for SSL + a decoder.

**SSL pretraining** (notebooks 06/08/11/13 + `scripts/pretrain_*`;
`src/train_ssl.py`, `src/ssl_*.py`): seven pretext tasks —
- *Reconstruction family:* masked reconstruction, contrastive (NT-Xent),
  **cross-channel masking** (zero one terrain channel, predict it; default target
  index 12 = TWI).
- *Spatial-ordering / geometric:* jigsaw (100 perms), rotation (4-class),
  **strip jigsaw** (3 horizontal strips, 6 perms).

Each writes an `encoder_state_dict` checkpoint under `checkpoints/ssl_pretrained/`.

**Downstream finetune + evaluation**
(`src/train_finetune.py::run_pretrained_resnet18_scv_experiment` — the core
engine): per fold, hold out one cluster as test, stratified train/val split of the
other four (val_fraction 0.2), load the SSL encoder strictly (~120 backbone keys),
train with **AdamW + discriminative LRs** (encoder vs head), early-stop on val
AUC, tune an F1 threshold on validation, write per-fold predictions/metrics.
Metrics (`src/metrics.py`, hand-rolled to dodge sklearn AUC edge cases): AUC,
PR-AUC, F1@0.5, F1@tuned-threshold. Summary reports mean/SD across folds + worst-
fold AUC.

**Comparisons** (notebooks 10/15/16; `scripts/_build_notebook16.py`) build the
5-task and 7-task tables and the `Fig_R1a–h` figures into `results_summary/`.

---

## 3. Directory map

```
.
├── CLAUDE.md                  ← this file (read first)
├── README.md                  ← install / run order / channel scheme / R1 table
├── RESULTS_NOTE.md            ← Phase 1: two new pretext tasks + R3 ablation
├── RESULTS_R4_R6_SUMMARY.md   ← variance investigation (R4–R7) + multi-seed rule
├── environment.yml / requirements.txt   ← original CUDA env (see §6 for MPS)
├── src/
│   ├── patch_dataset.py       ← index generation + RasterPatchDataset (channels, mask)
│   ├── models_resnet18.py     ← ResNet-18 classifier + encoder + weight loading
│   ├── train_ssl.py           ← all 7 SSL training loops
│   ├── train_finetune.py      ← 5-cluster SCV finetune engine (run_pretrained_..._scv_experiment)
│   ├── metrics.py             ← AUC / PR-AUC / F1 (hand-rolled)
│   ├── utils.py               ← seed / device (CUDA→MPS→CPU) / checkpoints
│   ├── ssl_{masked_recon,contrastive,jigsaw,rotation,cross_channel,strip_jigsaw}.py
│   ├── pu_bagging.py, spatial_cluster_balance.py, raster_cleaning.py, plotting.py
├── notebooks/                 ← 01–16 main workflow (data prep → per-task → comparisons)
├── scripts/                   ← CLI drivers (see §6) + `_`-prefixed reproducers (R5/R6/R7)
├── data/
│   ├── raw/                   ← rasters, NY boundary shapefile, landslide samples
│   └── processed/
│       ├── rasters_cleaned/   ← 14 aligned GeoTIFFs (Landcover.tif present but excluded by name)
│       ├── patches/           ← labeled patch indices (ps16/32/64; *_common_balanced = canonical)
│       ├── ssl_unlabeled_indices/   ← unlabeled SSL patches (n500, n20000)
│       ├── ssl_pretext_configs/     ← jigsaw / strip-jigsaw permutation banks
│       ├── pu_bagging/, samples/
├── checkpoints/
│   ├── ssl_pretrained/        ← SSL encoders (see §5 lineage — NOT all tasks present)
│   └── finetuned/             ← per-task per-fold classifier checkpoints
├── outputs/                   ← namespaced result CSVs (R2…R7, SSL_* training logs)
├── figures/                   ← per-namespace figures
└── results_summary/           ← archived canonical R1 tables + figures (frozen)
```

---

## 4. Current results state

- **Pipeline reproduces a canonical ~0.78–0.81 AUC.** Reported originally as a
  single-run **0.810**; the trustworthy multi-seed value is **0.782 ± 0.025**
  (3 seeds, R6).
- **SSL currently gives ~no gain over the scratch baseline (~0.779)** on this
  dataset — the difference is inside the noise band.
- **The NYC / Greater New York study area is suspected to be a poor proving
  ground for SSL** — too small, possibly too easy, and too urban (urban terrain
  breaks the terrain-derivative signal SSL is supposed to exploit). This is the
  main reason the roadmap (§7) moves to other study areas.
- Historical task rankings (5-task R1 and 7-task R2 tables) were **single-run** on
  the **old 14-channel-with-landcover input** and are **not directly comparable**
  to current-code numbers. Treat them as historical. Only R6 and R7 are
  multi-seed validated.

### THE RULE — multi-seed reporting (most important methodology finding)

On this dataset a **single 5-fold mean AUC swings ~±0.15** run-to-run, because
**MPS is non-deterministic even at a fixed seed** and the **5 spatial folds are
tiny and wildly uneven** (per-fold AUC spans ~0.33–0.92 for a *known-good*
encoder). A single 0.58 draw and a 0.78 draw came from the *same* config.

> **Always report mean ± SD over ≥3 seeds (5 is better). Only trust a difference
> if it exceeds the seed-to-seed SD (rule of thumb: ~2×). Never quote a single
> run as a result.** Multi-seed finetune sweeps are cheap (encoders are reused;
> minutes, not the ~2 h of SSL pretraining), so there is no excuse to skip them.

---

## 5. Encoder lineage (which checkpoint is what)

Verified by loading the checkpoint configs. This matters because several
`cross_channel*` dirs look interchangeable but are not:

| `checkpoints/ssl_pretrained/…` | Pipeline | Notes |
|---|---|---|
| **`cross_channel_full`** | OLD: 14 terrain **incl. landcover**, no mask channel | **THE canonical encoder.** Full 50-epoch run (best ep 48, val_loss 0.145, masked target idx 13 = TWI). Produced both the single-run **0.810** (R2-diag) and the **0.782 ± 0.025** 3-seed (R6). *Confirmed from the checkpoint config: `len(channel_means)=14` (landcover included), `conv1 in_channels=14` with 0 appended (no mask channel), and `masked_channel_index=13` → `twi_dinf_30m.tif` (TWI is index 13 only when landcover is in the stack).* |
| `cross_channel` | same OLD setup | discarded **2-epoch smoke** encoder — ignore |
| `cross_channel_newpipe` | NEW/Phase-2: 13 terrain (**landcover dropped**) + valid-context mask | current-code cross-channel encoder (best ep 26, masked idx 12 = TWI). R7's "no_landcover" arm uses this. |
| `cross_channel_ablation/chNN` | NEW | one encoder per masked channel — the R3 mechanistic ablation |
| `strip_jigsaw`, `strip_jigsaw_full` | Phase-1 | smoke / full strip-jigsaw encoders |

- **"newpipe"** = the current Phase-2 input (drop landcover, add valid-context
  mask). **"nomask"** (seen only as `outputs/SSL_cross_channel_ps32_*_nomask/`
  and `R2_*_nomask_*` namespaces) = an ablation that removes the mask channel
  (terrain-only input). **The nomask encoders were deleted** — only the three
  `cross_channel*` dirs above (plus ablation/strip) remain. The nomask runs
  existed to disentangle the landcover-drop effect from the mask-channel-addition
  effect.
- **The R1 five original encoders (scratch / masked recon / contrastive / jigsaw /
  rotation) are absent and not regenerable from this package** without
  re-pretraining from scratch; their results survive only as frozen CSVs/figures
  in `results_summary/`.
- **The sequential-SSL encoder is GONE** (only its logs/metrics/figures remain).
  ⚠️ **To run a sequential finetune you must re-run sequential PRETRAINING from
  scratch (`scripts/pretrain_sequential_ssl.py`, the ~2 h two-stage step).** There
  is no saved encoder to cheaply re-finetune.

---

## 6. How to run (conventions)

**Environment.** Original training was Windows + CUDA RTX 4090 (conda env `DL2`,
torch 2.3.1); `environment.yml` pins that CUDA build and **does not install on
macOS**. This machine is Apple Silicon and uses a native arm64 venv:

```bash
python3.11 -m venv .venv
.venv/bin/pip install torch==2.3.1 torchvision==0.18.1 "numpy<2" \
    pandas scikit-learn rasterio matplotlib tqdm affine scipy nbformat nbclient ipykernel
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"   # True
```

Run everything with `.venv/bin/python`. Device selection is automatic:
CUDA → MPS → CPU (`src/utils.get_device`).

**Hard conventions (don't change without reason):**
- `--num-workers 0` everywhere — lazy rasterio handles are **not fork-safe**.
- `cache_in_memory=True` for training — patch loading is I/O-bound (~34 patches/s
  uncached); the in-RAM cache (~1.15 GB, verified bit-identical) turns hours into
  minutes. DataLoader workers do **not** help (shared GeoTIFFs are the contention).
- Training loaders use `drop_last=True` (BatchNorm safety on tiny final batches).
- **Cross-channel finetuning needs the stable LR** (head 2e-5 / encoder 5e-6, ~60
  ep). At the standard 1e-4 it **collapses** (probabilities saturate, AUC below
  random). This is an LR instability, not a bad encoder — see `RESULTS_NOTE.md` §3.

**Common commands:**

```bash
# SSL pretrain (current pipeline; ~2 h / 50 ep on MPS) → checkpoints/ssl_pretrained/cross_channel<tag>
.venv/bin/python scripts/pretrain_cross_channel_ssl.py --run-tag newpipe
.venv/bin/python scripts/pretrain_strip_jigsaw_ssl.py  --run-tag full

# Sequential pretrain (Stage1 masked-recon → Stage2 cross-channel) + optional finetune
.venv/bin/python scripts/pretrain_sequential_ssl.py --stage1-epochs 50 --stage2-epochs 50 --stage2-lr 2e-5 --finetune

# Downstream 5-cluster SCV finetune of an SSL encoder (cheap; minutes)
.venv/bin/python scripts/finetune_new_ssl_tasks.py --task both --run-tag diag \
    --encoder-run-tag full --head-learning-rate 2e-5 --encoder-learning-rate 5e-6 \
    --max-epochs 60 --early-stopping-patience 12

# Multi-seed reproducers (the trustworthy numbers)
.venv/bin/python scripts/_check_newcode_landcover.py    # R6: canonical encoder, 3 seeds → 0.782 ± 0.025
.venv/bin/python scripts/_landcover_multiseed.py        # R7: landcover with/without, 3 seeds
.venv/bin/python scripts/ablate_cross_channel_mask.py   # R3: mask-channel ablation
```

Output namespaces are tagged by `--run-tag` (e.g. `full`, `diag`, `newpipe`) so
runs don't collide; `--encoder-run-tag` locates the encoder independently of where
results are written.

---

## 7. Roadmap / next steps

> ⚠️ **See the SESSION LOG at the bottom of this file for the current state (as of
> 2026-07-01).** Item 2 below is now DONE (sequential-SSL multi-seed: no gain vs
> scratch). The live lead is aligning to Qianyi's **562-sample** setup (SESSION LOG §4).

1. **Swap in Qianyi's new 10 m terrain features** (replacing the current 30 m) and
   re-evaluate **multi-seed**.
   ⚠️ **Baseline warning:** the canonical **0.782** was produced on the OLD input
   (landcover included, no mask channel — see §5). Comparing new 10 m-pipeline
   numbers directly against 0.782 conflates the **resolution change** with the
   **landcover/mask change**. A clean 10 m evaluation needs a **matched-pipeline
   30 m baseline** (same channel scheme, only the resolution differs) — not the
   0.782 number.
2. **Properly multi-seed the sequential-pretraining encoder vs. the canonical**
   one — close the currently-open gap (sequential has only a single, uninformative
   run so far). ⚠️ Requires re-pretraining the sequential encoder first (§5).
3. **Move to study areas where SSL can actually prove itself:**
   - **Kentucky** — KGS July 2022 storm inventory, **1000+ landslides, natural
     terrain**. Labels are **POINTS, not polygons**, and a point may land anywhere
     within a slide (crown / scarp / middle / toe) → **a point-to-patch sampling
     strategy is needed.**
   - **Colorado** (possible). Te Pei is preparing the feature extraction + Google
     Earth Engine code for both.
4. **Sync with Qianyi** so our NYC numbers align (we model the same area); she also
   owns the **local–global dual-view CNN** code (a deferred dependency).

This cycle's three completed tasks (detail in `RESULTS_R4_R6_SUMMARY.md`):
1. **Dropped landcover** from cross-channel masking (14→13 ch): it's a dead
   pretext *target* (can't be predicted from terrain), but appears to help as a
   downstream *input* feature (~+0.11 AUC, borderline at 3 seeds). The two roles
   are decoupled.
2. **Boundary-aware padding + valid-context mask channel** (13→14 ch): keep
   center-valid patches, zero-pad missing pixels, flag real(1)/pad(0). A **no-op
   on the current all-valid NYC data**; it matters at full scale / edge tiles.
3. **Sequential pretraining** (masked-recon warmup → cross-channel masking): built
   and verified to run; **downstream value not yet multi-seed evaluated** (see
   roadmap item 2).

---

## 8. Certainty notes (what's verified vs. inferred)

- **Verified from code/checkpoints:** channel scheme (13 terrain + mask), 344-patch
  canonical set, encoder lineage in §5, the missing R1/sequential encoders, the
  finetune LR/collapse behavior, num_workers/cache/drop_last conventions, the
  R6/R7 multi-seed numbers.
- **Inferred / not line-by-line verified:** the exact **PU-Bagging** and **KMeans**
  hyperparameters in the data-prep notebooks/modules (read at the workflow level,
  not audited), and **which notebooks were actually executed end-to-end on the
  current code** — `README.md` itself warns saved outputs may be missing even when
  result files exist, so notebook-derived claims should be treated as indicative.
- **Known doc/code discrepancy:** `scripts/_landcover_multiseed.py` (R7) has a
  stale comment saying the no-landcover arm reuses an "lc_nomask" encoder, but the
  executed code points that arm at `cross_channel_newpipe`. The code is what ran.
```

SESSION BRIEF — June 24 2026
Migrating to Kentucky dataset. Qianyi (he/him) has delivered 10m terrain features. Tasks for next 2 days (meeting in 2 days):

Audit what 10m data we have — list all rasters, check channels, confirm coverage over the 8-county eastern KY study area (Wolfe, Lee, Owsley, Breathitt, Knott, Perry, Leslie, Letcher)
Run sequential pretraining (Task 3) multi-seed (≥3 seeds) on the existing 30m NYC data to close the open evaluation gap — this is the matched baseline
Run the same sequential pretraining multi-seed on the 10m Kentucky data
Compare 30m vs 10m with actual mean ± SD AUC numbers
Confirm boundary-aware padding + valid-context mask is active in the pipeline (already implemented, just verify it's wired in)

Critical rules:

Always report mean ± SD over ≥3 seeds, never single-run
Do NOT compare 10m numbers against the old 0.782 canonical — that conflates resolution change with channel scheme change
The matched 30m baseline uses the same current channel scheme (13ch + mask = 14ch input), not the old cross_channel_full encoder
Repo is at /Users/mohamedhiba/Projects/LSM_SSL_ResNet18_repro_package_20260515

---

## SESSION LOG — 2026-06-24 → 07-01 (read this for current state)

New docs from this cycle (don't duplicate — extend/reference):
`MEETING_RESULTS.md` (the 2×2 result + table), `CRAWFORD_BRIEFING.md` (supervised
baseline lit), `NEXT_STEPS_MEETING.md` (todo/talking points), `colab_package/`
(the runnable multi-seed driver + patch cache).

**1. The "10 m data" on disk is NYC, not Kentucky.** `data/processed/rasters_cleaned_10m/`
(13 GeoTIFFs, EPSG:26918/UTM18N, ~8.85 m, lon −75/−71) is the **NYC study area
re-extracted at 10 m** = Qianyi's dual-view resolution stack, NOT KY terrain.
**Kentucky has POINT LABELS ONLY** (`data/kentucky/`: KGS July-2022 inventory, 1,106
pts + 6,630 jittered centers in EPSG:26917/UTM17N) and **no terrain rasters** →
**KY modeling is data-blocked** until a KY terrain stack is delivered.

**2. Sequential-SSL 2×2 is DONE multi-seed (closes old roadmap item 2).** Ran
{30m,10m}×{scratch, sequential-SSL (masked-recon→cross-channel)}, 3 seeds, matched
341-sample set, spatial 5-fold, on **local MPS** (Colab upload impossible — bad
internet). Result (`outputs/R8seq_local_mps/`, writeup `MEETING_RESULTS.md`):
scratch 30m **0.612**, 10m **0.636**; SSL 30m **0.564**, 10m **0.577**. **SSL−scratch
= −0.05 (30m) / −0.06 (10m), both WITHIN NOISE → no SSL benefit**; 10m−30m also within
noise. Scratch is stronger/more stable; SSL noisier (one seed collapsed to 0.45 =
documented finetune fragility). ~0.6 is the honest floor of a small/weak/urban dataset
under strict spatial CV, and it **reproduces the R8 numbers** — not a bug. The old
0.78 had landcover as an input feature (not comparable).

**3. Te Pei's directive (framing):** do NOT chase Crawford et al.'s absolute AUC
(**0.78–0.83**, but on **1.5 m lidar** + classical ML — 2022 *Remote Sens.* 14:6246,
2025 *Nat. Hazards* 121:11633). Build **our own baseline on our own data** and report
**relative** performance (SSL vs scratch). Flag the 1.5 m-vs-10 m resolution confound.

**4. Qianyi's feedback = the current lead.** Our 341 samples were the gap. His "final
10 m" pipeline (`LSM_SSL_ResNet18_10m_dual_view_code/`, code-only, no data) uses a
**562-sample set (281 landslide + 281 non-landslide)** from a **corrected PU mean-OOB
rule** + **dual-view patches (local 15×15 + global 31×31)** + Flat-CV/SCV. Code
hard-enforces `len==562`, `{0:281,1:281}`. **NEXT STEP: rebuild/adopt the 562-sample
set and re-run the SSL-vs-scratch comparison on it.** Now unblocked — the 10 m rasters
were deleted (disk) then **restored from Trash (verified intact, 13 tif, EPSG:26918)**.

**5. Compute.** MPS is too slow (~10 h/run) and non-deterministic; the SSL finetune
collapses run-to-run. Need real CUDA. Two paths being pursued: (a) **SSH to a lab GPU
box** (asking Prof. Tian — the lab already has the RTX 4090 from `DL2`), (b) **rent a
cloud GPU** (I can drive it headless with an API key; ship the ~1.6 GB patch cache, not
the 30 GB rasters — the `colab_package` cache+monkeypatch pattern makes any raster-free
GPU work). Bad internet makes big uploads painful → prefer the lab box, or cache-only
transfer.

**Reusable machinery built this cycle:** `scripts/export_colab_cache.py` (extract raw
patches → compact `.npz`), `colab_package/colab/{colab_patch.py, sequential_resolution.py}`
(monkeypatch datasets to serve the cache; run the 2×2 multi-seed on CUDA/MPS/CPU with
`--methods scratch ssl --arms 30m 10m --seeds ...`).

---

## SESSION LOG — 2026-07-01 (latest; supersedes the 341/2×2 results above)

**A. The NYC result flipped: on the CORRECT dataset, SSL BEATS scratch.** The earlier
"no gain" (341 samples) was the WRONG set. The correct one was already in the repo:
`data/processed/samples/final_cluster_balanced_dataset.csv` = **562 (281 landslide + 281
non-landslide), 5 balanced clusters**. Built ps32 indices with `scripts/build_562_indices.py`
→ `labeled_patch_index_ps32_balanced562_{30m_matched,10m}.csv` (matched 557 valid at both
res). Ran the **5-method × {30m,10m} × 3-seed** comparison (scratch + masked_recon +
cross_channel + strip_jigsaw + sequential) on a rented **RunPod RTX 4090**. Result
(`MEETING_RESULTS_562.md`, `outputs/R8seq_562_runpod/`, figs `figures/R8seq_562/`):
mean AUC 30m/10m — scratch 0.836/0.860, masked_recon 0.865/0.905, cross_channel 0.871/0.877,
sequential 0.874/0.903, strip_jigsaw 0.779/0.865. **SSL−scratch = +0.03…+0.05, clears the
tiny seed-SD (~0.01–0.036) → REAL.** Winners: **sequential + masked_recon** (highest, most
stable); SSL also far more robust on the worst spatial fold (scratch ~0.59 vs SSL ~0.73–0.85);
**strip_jigsaw is the negative control** (hurts at 30m). NYC pod terminated after the run.

**B. RunPod GPU workflow (how we run on real CUDA now).** `runpodctl` + SSH key
`~/.ssh/runpod_ed25519`. Ship the compact **patch cache** (not 30 GB rasters); `colab_patch`
monkeypatches datasets to serve it. The driver `colab_package/colab/sequential_resolution.py`
now supports `--methods scratch masked_recon cross_channel strip_jigsaw sequential`,
`--num-workers` (cache is fork-safe → use the cores), `--ssl-epochs`. ⚠️ ROTATE the RunPod
API key (pasted in chat). ⚠️ On flaky internet: launch jobs **detached (`setsid`)** so they
survive SSH drops; monitor watches **`available`** memory not `free` (page cache makes `free`
look scary — NOT a real OOM).

**C. Kentucky is UNBLOCKED and running (the live task).** KY 10 m terrain rasters arrived
(from Qianyi's Drive `processed_data.zip`; on the KY pod at
`/workspace/ky/dl/processed/rasters_cleaned_10m`, 13 tif, **EPSG:3088**). KY modeling set is
**ps64, 27,594 samples = 9,198 landslide : 18,396 non-landslide (1:2, NOT 1:1)**, 5 spatial
clusters (`labeled_patch_index_ps64_kentucky_kgs6c_1to2.csv`). Driver
`scripts/run_kentucky_gpu.py` mirrors the NYC 5-method design (scratch 100ep/LR1e-4; SSL
60ep/enc5e-6/head2e-5; masked idx 12=TWI). Running on a **2nd RunPod pod (RTX 3090,
`jn5ggk5jdb80kg`, 64.119.209.250:12097)**, 3 processes (one per seed), **native raster mode**
(cache_in_memory rebuilds per fold — slow, ~8–10 h; a cache-mode conversion was attempted and
failed, auto-fell-back to native). ⚠️ **The KY OOM earlier was over-parallelization** (6
concurrent ps64 caches), NOT a code bug — fixed by running ≤3 processes with
`OMP/MKL/OPENBLAS_NUM_THREADS=4`. When KY finishes: `scripts/combine_ky_results.py`, then make
KY figures matching `figures/R8seq_562/`. **KY 1:2 ratio → compare on AUC + SSL−scratch delta,
not absolute PR-AUC.**

**D. Coordination:** a second Claude session did the KY data prep; this session drives the KY
GPU run. Don't have two sessions touch the same pod. Compute path decided: **rent RunPod GPUs**
(cheap 3090 ~$0.22/hr fine — model is tiny, GPU ~20% util, bottleneck is CPU/data).

---

## SESSION LOG — 2026-07-10 (latest; NEW-SESSION HANDOFF — read `KY_HANDOFF.md` LIVE STATE)

**Qianyi's flat-CV fix is implemented; the KY flat-CV run is fully prepped but NOT yet running
— blocked purely on Mac→pod file transfer.**

- **Flat CV added.** `src/train_finetune.py::run_pretrained_resnet18_scv_experiment` now takes
  `cv_mode="scv"|"flat"` (default `scv` → **NYC untouched**). `flat` = `StratifiedKFold(5, shuffle,
  random_state=seed)` on label — Qianyi's fix because KY positives are spatially clustered (SCV folds
  swing 11–64 % positive vs a uniform 33 % for flat, verified). `scripts/run_kentucky_gpu.py --cv-mode
  {scv,flat}`. Run BOTH eventually; **flat is the meeting priority** (de-noises SSL−scratch).
- **Everything prepped in `ky_flat_pkg/`** (persisted in repo): caches `ky_labeled.npz` (244 MB) +
  `ky_unlabeled.npz` (147 MB) built leak-safe (boundless=False, per [[gdal-boundless-read-oom-on-pod]]);
  indices (+ `slim_*`, factor cols dropped, ~603/183 KB); `do_ky_pod.sh` (self-contained pod bootstrap:
  extract code → gdown rasters → rebuild caches on pod → launch training detached) and `run_ky_flat.sh`
  (cache-mode `--cv-mode flat`, ps64, 5 methods × 3 seeds). Builder: `scripts/build_ky_flat_cache.py`.
- **THE BLOCKER: this Mac can't move bulk data to the pod.** scp hangs (exit 143); base64-over-ssh
  works only ≤~400 KB (383 KB code.tar OK, 603 KB+ stalls); 0x0.st host is safety-blocked. **Fix: don't
  push the 391 MB caches — have the pod rebuild them** (gdown rasters ~26 s + boundless=False build
  ~10 min via `do_ky_pod.sh`). Only ~800 KB of indices must reach the pod: try **`runpodctl
  send/receive`** (P2P, different transport) first, else **chunked base64** (`split -b 280000`, one
  chunk per ssh call). Then launch detached, verify GPU >0 % + folds in `cmflat.log`.
- Pod `9l4cvkdiw90lx6` (RTX 4090, `root@209.170.80.132 -p 10921`, key `~/.ssh/runpod_ed25519`) had
  `ky_code.tar` transferred but may have auto-terminated (~4 h) — if gone, start fresh (gdown re-pulls).
  ⚠️ ROTATE the RunPod API key (pasted in chat). ⚠️ **NYC 562-set (`MEETING_RESULTS_562.md`) is DONE —
  a complete result to present on its own if KY doesn't land in time.**
- **Full step-by-step for the fresh session is in `KY_HANDOFF.md` → "LIVE STATE" block.**

---

## SESSION LOG — 2026-07-14 (latest; KY flat-CV ran + QIANYI'S PROTOCOL-ALIGNMENT DIRECTIVE)

**A. KY flat-CV run executed (2026-07-10, RunPod 4090, pod `9l4cvkdiw90lx6`).** The transfer
blocker was beaten by shipping only a 215 KB gzipped payload (slim indices + scripts) via one
base64-over-ssh shot and having the pod rebuild the caches itself (gdown rasters + `do_ky_pod.sh`;
NOTE: pod image lacks `unzip` — script now falls back to `python -m zipfile`). Run: 5 methods ×
3 seeds, `--cv-mode flat`, ps64, 5k labeled / 3k unlabeled caches. Results in
`outputs/KY_flat_runpod/` (partial pulls in `partial/`), writeup = top section of
`MEETING_RESULTS_KY.md`. **Flat CV collapsed fold variance ~5× (fold SD ±0.02 vs ±0.08–0.10
SCV) and SSL still does not beat scratch on KY** (scratch ~0.733; SSL −0.03…−0.07, outside
noise). Consistent with the 07-03 SCV table and the flat label-efficiency curve.

**B. QIANYI'S FEEDBACK (Slack 2026-07-13): our KY setup is NOT directly comparable to his —
substantial protocol differences that could themselves explain the SSL gap.** His final 10 m
pipeline vs ours:

| | Qianyi (dual-view) | Ours (KY flat run) |
|---|---|---|
| SSL pretraining corpus | **50,000 unlabeled centers** | 3,000 |
| Views / patch sizes | dual-view 15×15 local + 31×31 global | single-view 64×64 |
| Channels | 13 factors + binary context mask (same) | same |
| Masked recon | **50 % mask ratio, spatial blocks (3×3 local / 4×4 global), reconstruct only the 13 factor channels** | our masked-recon/cross-channel defaults |
| Downstream LRs | **head fixed 1e-3; encoder swept 0→1e-3** (LR=0 freezes weights AND BN stats) | head 2e-5 / enc 5e-6 |
| Collapse at enc LR 1e-4? | **not observed systematically** (task/fold dependent) | we see collapse at 1e-4 (documented) |
| Labeled balance | 1:1 | 1:2 |

Alignment priority (his order): dual-view patch sizes → 50k pretraining set → normalization/
mask handling → class balance (1:1) → head LR. ⚠️ His "no collapse at enc 1e-4" means our
collapse finding may be an artifact of our low head LR (2e-5 vs his 1e-3) — re-test the LR
grid with head 1e-3 before citing collapse as inherent.

**C. QIANYI'S ROADMAP DIRECTIVE (after alignment):** (1) generate actual **LSM results**
(susceptibility maps); (2) sanity-check the maps are physically reasonable; (3) discuss
results + next steps with the professors; (4) then shift focus to **own research analysis:
LSM maps, spatial patterns, interpretation** — not more protocol tuning.

**Implication for the KY story:** the "SSL doesn't help KY" conclusion now carries an
additional caveat — 3k-vs-50k pretraining corpus and the LR scheme are confounds Qianyi
considers substantial. Before presenting KY SSL as a settled null, either align the protocol
or scope the claim to "our configuration."

**D. ALIGNMENT IMPLEMENTED (2026-07-14, no training launched).** Read **`KY_DUAL_VIEW_ALIGNMENT.md`**
— it has the verified protocol spec (extracted from his code with citations) and the plan.
Slack-vs-code corrections that matter: his SSL corpus is **20,000** (50k = PU candidate pool
only); encoder-LR grid is **[0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]** (tops at 1e-4, NOT 1e-3);
masked-recon pretrain LR is **1e-4**; **LR=0 freezes weights only** (BN stats drift unless
`strict_frozen_encoder=True`); 5 seeds [42-46]. Code: `scripts/ky_dual_view_shim.py`
(sys.path bootstrap + in-place `FINAL_10M_FACTOR_NAMES` patch: lithology→ksat,
ndvi filename; ⚠ never mix his `src/` and ours in one process),
`scripts/build_ky_dual_view_indices.py` (stage 10: KY dual-view indices + norm stats →
`data/kentucky_dual_view/`), `scripts/build_ky_pu_labeled.py` (stage 30: PU mean-OOB 1:1
KY labeled set, his 281-literals generalized to n_pos). **07-17 corrections:** (a) the
"jittered positives" concern was WRONG — the 9,198 are real deduplicated KGS records
(kgs_v4: 9,242 = 1,842 points + 7,400 polygon centroids, 6,123 original features, max 3
cells/feature); stage 30 now uses **StratifiedGroupKFold grouped by original_feature_id**
(zero same-feature train/test leaks, QA-asserted). (b) Per Mohamed: SSL corpus = **50k**
(Qianyi's stated number, not his code's 20k default) — `unlabeled_dual_view_padded_index_ky10m_n50000.csv`
built, QA passed. Stage-10 outputs verified: 27,594 labeled (9,198/18,396), stats finite,
ksat contract correct.

**E. CURRENT STATE (07-17) + NEXT SESSION'S JOB.** All KY data prep for the aligned run is
DONE and QA-passed in `data/kentucky_dual_view/` (50k unlabeled corpus, PU mean-OOB 1:1
labeled set 9,198+9,198, group-safe StratifiedGroupKFold split by `original_feature_id` —
zero same-landslide leaks). Meeting summary: `MEETING_UPDATE_20260717.md`. **Next session:
(1) write the thin run driver around Qianyi's `FinalSSLTrainingConfig` /
`FinalFlatCVLRSweepConfig` pointed at the KY paths (import `scripts/ky_dual_view_shim.py`
FIRST — never mix his src/ with ours in one process; his 3-job pilot mode is the smoke
test); (2) rent a GPU (RunPod workflow in KY_HANDOFF.md; rasters re-pull via gdown, ship
only small indices; detached launches; verify GPU>0% before walking away); (3) run
SSL-vs-scratch under his protocol, ≥3 seeds, report mean±SD; (4) then stage-90 LSM maps.** Also: KY flat run final state = seed 42
complete (scratch 0.738 / cross_ch 0.708 / strip 0.694 / masked_recon 0.665 / sequential
0.644) + seed 7 scratch 0.727; seeds 123 & rest of 7 LOST with the pod (only partial pulls
in `outputs/KY_flat_runpod/partial/`).