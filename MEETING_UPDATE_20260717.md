# Meeting update — 2026-07-17: Kentucky aligned to Qianyi's dual-view protocol

**One-liner: Qianyi said our KY setup wasn't comparable to his; as of today the Kentucky
data is fully rebuilt in his exact protocol (dual-view, 50k pretraining corpus, PU-selected
1:1 negatives, his LR scheme), all QA-passed, ready to train. No training run yet.**

## 1. Where we left off (last meeting, 07-10)

- NYC-562: **done** — SSL beats scratch (+0.03…+0.05, real vs seed noise). `MEETING_RESULTS_562.md`.
- KY flat-CV run: SSL below scratch (scratch 0.733, SSL −0.03…−0.07) with tight fold noise
  (±0.02) — but the pod died before all seeds were pulled (seed 42 complete + seed 7 scratch
  survived; seed 123 lost). Preliminary, and now superseded by the alignment below.

## 2. Qianyi's feedback (07-13) — why our KY numbers weren't comparable

Our run vs his final pipeline: single-view 64×64 vs **dual-view 15×15 local + 31×31 global**;
3,000 vs **~50,000** unlabeled pretraining samples; 1:2 vs **1:1** class balance; head LR
2e-5 vs **1e-3**; buffered-random vs **PU mean-OOB** negatives. His directive: align these
first, then generate LSM maps, then shift to interpretation/spatial analysis.

We verified his protocol against his actual code (`KY_DUAL_VIEW_ALIGNMENT.md` has the full
spec with citations). Three corrections to the Slack version worth confirming with him:
his code pretrains on **20k** (50k is his PU candidate pool); his encoder-LR grid tops at
**1e-4** (not 1e-3); LR=0 freezes weights but **BN stats still drift** unless his strict
pilot flag is set. We built 50k anyway (his stated number; more can't hurt).

## 3. What was built this week (all local CPU, all his QA gates passed)

| Artifact | Detail |
|---|---|
| KY dual-view labeled index | all 27,594 kgs6c samples center-valid in his 15/31 format |
| Unlabeled SSL corpora | **n=50,000** (his stated number) + n=20,000 (his code default) |
| Normalization stats | whole-raster block-wise per-factor mean/std, his schema (ksat replaces lithology, slot 6) |
| PU mean-OOB negatives | RF 50 iters × 100 trees on 50k candidates; keep only mean-OOB ≤ 0.5 — **harder negatives**, addressing our July "negatives too easy" diagnosis |
| Balanced 1:1 labeled set | **18,396 = 9,198 + 9,198** (`dual_view_padded_patch_index_ky10m_pu_mean_oob_balanced.csv`) |
| Group-safe flat-CV split | StratifiedGroupKFold(5) grouped by KGS `original_feature_id` — **zero same-landslide train/test leaks** (QA-asserted) |

Code: `scripts/ky_dual_view_shim.py` (runs his package on KY data unmodified),
`scripts/build_ky_dual_view_indices.py`, `scripts/build_ky_pu_labeled.py`.
Outputs: `data/kentucky_dual_view/`.

## 4. Data correction worth stating at the meeting

The KY positives are **NOT jittered duplicates** (earlier belief, now disproven): they are
**9,242 real, deduplicated KGS inventory records** (1,842 surveyed points + 7,400
polygon-derived centroids; 6,123 original features, ≤3 cells each). The mild same-feature
grouping is what the group-safe split fixes — under plain random 5-fold, cells of the same
mapped landslide would sit in both train and test (score inflation that also biases
SSL-vs-scratch, since scratch's higher LR memorizes duplicates better).

## 5. Next steps (new session picks this up)

1. Write the thin run driver around his `FinalSSLTrainingConfig` / flat-CV sweep configs
   pointed at `data/kentucky_dual_view/` (his 3-job pilot mode = the GPU smoke test).
2. Rent GPU, run: 4 SSL tasks × seeds × his encoder-LR grid + scratch (full protocol is
   625 jobs — start with a grid subset {0, 1e-5, 1e-4} × 3 seeds).
3. Report mean ± SD, SSL−scratch deltas (≥3 seeds, as always).
4. **Generate whole-region KY LSM maps** (his stage 90, ensemble of downstream checkpoints)
   → sanity-check physical plausibility → then interpretation/spatial analysis (Qianyi's
   directive for where the research focus goes next).

Open questions for Qianyi: 20k-vs-50k in his final runs; encoder-LR 1e-3 claim vs code grid;
whether he wants the group-safe split adopted on his side too (his NYC positives are
unaffected — no polygon-derived multi-cell features there).

## 6. Update (meeting day, morning): run driver DONE and smoke-tested

Step 1 above is complete: `scripts/run_ky_dual_view_aligned.py` runs HIS training code
(SSL pretraining + downstream flat-CV LR-sweep jobs) on the KY data with the group-safe
split; the local 3-job pilot smoke test PASSED on real KY patches (scratch / frozen-encoder
LR 0 / encoder LR 1e-4 — his QA columns verify 120 encoder keys loaded, heads-only vs
two-group optimizers, BN-drift-allowed LR=0 semantics). Two deliberate deltas from his
wrappers, both protocol-neutral: our group-safe flat-CV split file replaces his internal
StratifiedKFold, and his hardcoded 18k SSL train split became 90/10 so the 50k corpus is
fully used. GPU launch package (1.8 MB payload + bootstrap + 4-process launch + result
pullers + combiner) is in `ky_dual_view_pkg/`. Priority grid for tonight: scratch +
masked_recon + contrastive, encoder LR {0, 1e-5, 1e-4}, seeds 42/43/44, batch 128,
head LR 1e-3. Blocked only on RunPod credits at the time of writing.
