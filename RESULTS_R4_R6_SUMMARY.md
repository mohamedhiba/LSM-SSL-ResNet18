# Results Summary — Pipeline Update (Tasks 1–3) + Variance & Landcover Investigation (R4–R7)

This note documents the channel/pretraining changes made to the SSL→finetune
pipeline, and a methodological investigation that followed when the new runs came
out low. The headline lesson is about **evaluation variance**, not any single
result.

---

## 1. Tasks 1–3 — what changed (code verified working)

All three were implemented and smoke-tested; each smoke test passed before moving on.

**Task 1 — Drop landcover (14 → 13 channels).**
`Landcover.tif` is excluded from the CNN channel stack by name in
`list_raster_files` (`EXCLUDED_RASTER_NAME_STEMS`); the file stays on disk. All
`14`→`13` count checks, model `in_channels`, and the cross-channel default
(TWI, now index **12**) were renumbered. *Verified:* `list_raster_files` returns
13, landcover excluded, dataset item `(13,32,32)`, classifier `conv1.in_channels=13`.

**Task 2 — Boundary-aware padding + valid-context mask (13 → 14 channels).**
Patches now require only a **valid center pixel**; boundary/local NoData is
zero-padded and flagged by an appended binary mask channel (1=real, 0=padded).
Shared helpers in `patch_dataset.py` (`read_boundless_patch_from_sources`,
`valid_context_mask`, `center_is_valid`, `apply_norm_and_append_mask`); cache
stays terrain-deep (mask derived per item); stats are mask-aware. *Verified:*
interior→all-ones mask, partial-boundary→correct zeros + terrain zeroed,
cache==no-cache identical, mask-aware mean unbiased by padding, cross-channel
never masks the mask channel.

**Task 3 — Sequential pretraining.** New `scripts/pretrain_sequential_ssl.py`:
Stage 1 (masked reconstruction, default) → load encoder unchanged → Stage 2
(cross-channel masking, LR 2e-5) → optional downstream finetune. *Verified:*
both stages run, encoder carries over (strict load, 0 missing/unexpected keys),
finetune loads 120 encoder keys.

**Deferred:** the local–global dual-view CNN (awaiting collaborator reference code).

---

## 2. The key finding — single-run AUCs are not trustworthy

When the new runs came out at ~0.51–0.58 (vs a canonical ~0.81), we ran a
control: **re-finetune the *exact* canonical cross-channel encoder with the new
code, landcover included, same LR.**

- **A single run gave 0.58** — alarming, looked like a regression.
- **Repeating the identical config across 3 seeds gave 0.759 / 0.778 / 0.808,
  mean 0.782, seed-SD 0.025** — right next to the canonical 0.810.

So the new code is **correct** (it reproduces ~0.78–0.81). The 0.58 was an
unlucky draw: at *fixed seed 42*, one execution gave 0.58 and another gave 0.759.

**Why:** Apple MPS is non-deterministic (fixing the seed does not pin the result),
and the 5 spatial-CV folds have wildly different difficulty (per-fold AUCs span
~0.33–0.92 for a *known-good* encoder). Together, a single run's 5-fold mean AUC
**swings ~±0.15**. The 3-seed *mean* is stable to **~±0.03** for well-converged
conditions (looser, ~±0.08, for less-stable ones).

**Cost:** these comparisons are **finetune-only** (encoders reused), so a 3-seed
× 5-fold sweep runs in roughly **1–20 minutes** depending on how early
early-stopping triggers (e.g., 15 folds in ~70 s when models stop early; ~20 min
when they train longer). The expensive part is *SSL pretraining* (~2 hrs / 50
epochs on 18k patches, MPS) — which is **not** repeated for these comparisons.

**Implication:** every earlier single-run number — including the headline 0.810
and the apparent "landcover −0.28", "sequential", and "mask" effects — sat
*inside* the ±0.15 single-run noise band. They were largely noise.

---

## 3. Landcover — proper multi-seed result (R7)

Two arms, both with comparably-trained cross-channel encoders and 14-channel
inputs, both at the stable finetune LR (head 2e-5 / encoder 5e-6, 60 ep), each
finetuned across 3 seeds (42, 123, 7). Arm A's extra channel is landcover;
arm B's extra channel is the valid-context mask, which is a verified no-op on
these all-valid patches (constant 1.0, absorbed by BatchNorm) — i.e. a fair
"13 terrain only" stand-in.

| Arm | seed 42 | seed 123 | seed 7 | **mean ± SD (15 folds)** | seed-to-seed SD |
|---|---:|---:|---:|---:|---:|
| **with landcover** | 0.759 | 0.772 | 0.716 | **0.749 ± 0.102** | 0.029 |
| **no landcover** | 0.574 | 0.615 | 0.722 | **0.637 ± 0.157** | 0.077 |

- **Gap = +0.112** in landcover's favor. Paired by seed: **+0.185, +0.157, −0.006**
  — landcover helps clearly in 2 of 3 seeds, ties in the third.
- Statistically **borderline**: a t-test on the per-seed means gives ~p ≈ 0.1,
  and the gap does not clear a strict 2×-(seed-noise) bar. Not conclusive at 3 seeds.
- **With-landcover is much more stable across seeds (SD 0.029) than no-landcover
  (SD 0.077)** — itself a hint that landcover adds real signal (the model relies
  less on lucky initialization).

**Plain-language verdict.** Landcover *probably* helps downstream — a **modest
~+0.11 AUC**, not the dramatic +0.28 the first single run suggested. The original
"dropping landcover costs ~0.28" claim was a noise artifact. The honest current
statement: **weak-to-moderate evidence that landcover provides downstream signal;
needs ≥5 seeds to confirm.** This still sits in tension with dropping it on
principle (it is a *poor* cross-channel mask target but a *useful* classifier
input); the two roles can be decoupled.

---

## 4. Supporting diagnostic runs (context)

All single-run unless noted; read as noisy point estimates (±~0.15).

| Run | Setup | Mean AUC | Note |
|---|---|---:|---|
| canonical (R2-diag) | old 14ch (incl. landcover), cross-channel | 0.810 | single run; may not reproduce exactly |
| canonical scratch | old 14ch, no SSL, standard LR | 0.779 | reference |
| R4 sequential | new 13+mask, default finetune LR (1e-4) | 0.612 | collapsed/saturated (LR instability) |
| R4 sequential | new 13+mask, stable LR | 0.537 | single run |
| R6 control | canonical encoder, new code, **3 seeds** | **0.782 ± 0.025** | proves new code is correct |
| R7 with-landcover | **3 seeds** | **0.749 ± 0.029** | see §3 |
| R7 no-landcover | **3 seeds** | **0.637 ± 0.077** | see §3 |

The R4 default-LR collapse (0.612) is the documented cross-channel finetune-LR
instability (saturation), not a data/code problem — the stable LR avoids it.

---

## 5. Recommendation — adopt 3-seed reporting as standard

1. **Never report a single run.** On this dataset a single 5-fold mean AUC
   carries ~±0.15 of run-to-run noise (MPS non-determinism + small spatial folds).
   Report **mean ± SD over ≥3 seeds** (5 is better); only believe differences
   that exceed ~2× the seed-to-seed SD.
2. **It is cheap.** Multi-seed comparisons are finetune-only (reuse encoders) and
   run in minutes. There is no excuse to skip them.
3. **Re-multiseed the headline numbers.** The canonical 0.810 (and the 7-task
   comparison in `notebooks/16_...`) should be re-run across seeds before being
   reported as definitive.
4. **The real fix for reliability is more labeled data.** 344 patches over 5
   spatial folds is the root cause of the variance; the planned full-US extraction
   directly shrinks it.
5. (Optional) For exact reproducibility, consider seeding + CPU/deterministic
   runs for the *final* headline numbers, accepting they'll be slower than MPS.

---

## 6. Artifacts

Result CSVs (kept):
```
outputs/R4_sequential_ssl/metrics/                 sequential, default LR (0.612)
outputs/R4_sequential_ssl_diag/metrics/            sequential, stable LR (0.537)
outputs/R5_landcover_scratch_ablation/r5_*.csv     scratch ablation (note: stable LR floors scratch nets — undertrained)
outputs/R6_newcode_landcover_check/r6_multiseed.csv  3-seed reproducibility (0.782 ± 0.025)
outputs/R7_landcover_multiseed/r7_landcover_multiseed.csv  landcover 3-seed comparison (§3)
```
Figures (only the first sequential run was plotted): `figures/R4_sequential_ssl/`.
Canonical encoders kept under `checkpoints/ssl_pretrained/` (incl. `cross_channel_full`).
Per-fold finetune checkpoints from R6/R7 were deleted (regenerable; metrics retained).

Reproducer scripts (standalone, no pipeline files modified):
`scripts/_check_newcode_landcover.py` (R6), `scripts/_landcover_multiseed.py` (R7),
`scripts/scratch_landcover_ablation.py` (R5).

---

## 7. Canonical 0.810 — multi-seed validation

> Note: a dedicated **5-seed** canonical run was requested but **was never started**
> (no such run/CSV exists; nothing was launched, per the no-new-experiments
> instruction). The validation below uses the **3-seed** R6 data, which *is* the
> canonical condition: the original `cross_channel_full` encoder — the one that
> produced the single-run 0.810 — re-finetuned with the current code, landcover
> included, at the stable LR, across 3 seeds.

| Seed | Mean AUC | Folds |
|---|---:|---|
| 42 | 0.759 | 0.86 / 0.69 / 0.73 / 0.68 / 0.84 |
| 123 | 0.778 | 0.82 / 0.80 / 0.75 / 0.70 / 0.82 |
| 7 | 0.808 | 0.89 / 0.83 / 0.70 / 0.70 / 0.92 |
| **Overall** | **0.782 ± 0.080** (15 folds) | seed-to-seed SD **0.025** |

Single-run reference: **0.810**.

**Verdict: the canonical number holds up.** The 3-seed mean is **0.782 ± 0.025**
(seed-to-seed), stable and right next to the single-run 0.810 — it does **not**
shift dramatically the way the landcover number did (single-run +0.28 → multi-seed
+0.11). The reliable estimate is ~**0.78–0.81**, not a fluke. (0.810 sits at the
top of the 3-seed range, consistent with it being one favorable draw; 0.782 is the
better point estimate.) A true 5-seed run would tighten the interval but is not
expected to change this conclusion.

---

## 8. Future work (noted, not run this session)

- **5-seed (or 10-seed) canonical validation** — finish what §7 started at higher
  seed count to tighten the ~0.78–0.81 interval; finetune-only, cheap.
- **Re-multiseed the full 7-task comparison** (`notebooks/16_...`) and the R2/R3
  tables before reporting them as definitive — they are currently single-run.
- **Landcover at ≥5 seeds** to settle the borderline +0.11 effect (§3), and decouple
  its roles: keep it as a downstream feature (one-hot/embed) while excluding it as a
  cross-channel mask target.
- **More labeled data** (full-US extraction) — the root cure for the ±0.15 single-run
  variance; everything above becomes sharper once the 344-patch bottleneck is lifted.
- **Determinism for headline numbers** — consider CPU/deterministic final runs since
  MPS is non-deterministic even at fixed seed.
