# RESULT — KY under Qianyi's FULL aligned protocol (2026-07-17, meeting-day run)

**Headline: with the protocol fully aligned to Qianyi's dual-view pipeline, the SSL
deficit on Kentucky disappears — best SSL settings reach PARITY with scratch
(masked recon 0.956 vs scratch 0.955), but unlike NYC, SSL does not beat scratch.
And absolute AUC jumps from ~0.73 to ~0.95: the protocol (dual-view, PU mean-OOB
negatives, 1:1, head LR 1e-3) was worth +0.22 AUC — far more than SSL itself.**

Setup (per his code + his 07-13 Slack, verified spec in `KY_DUAL_VIEW_ALIGNMENT.md`):
dual-view 15×15/31×31 shared ResNet-18, 13 factors + context mask (ksat in slot 6),
masked recon ratio 0.5 blocks 3×3/4×4 factor-channels-only, contrastive; SSL corpus
**n=50,000** (his Slack number), batch 64, ≤50 ep patience 10; downstream head LR
**1e-3**, encoder-LR sweep **{0, 1e-5, 1e-4, 1e-3}**, batch 128, BCE pos_weight,
AdamW wd 1e-4, ≤100 ep patience 15; **LR=0 = strict freeze (weights + BN, QA-verified
by state hashes)** per his Slack; labeled set = PU mean-OOB 1:1 (9,198+9,198);
**flat CV via the group-safe split** (StratifiedGroupKFold by KGS `original_feature_id`
— zero same-landslide train/test leaks; his protocol is plain StratifiedKFold, our
split is the leak-fixed version we recommend adopting). Seeds 42–45 (his list is 42–46).
Hardware: RunPod RTX 4090 + RTX 3090, ~9 h wall, ~$9. Outputs:
`outputs/KY_dual_view_aligned/`, driver `scripts/run_ky_dual_view_aligned.py`.

## The table (seed-level mean ± SD; final refresh pre-meeting)

<!-- KY_ALIGNED_TABLE_START -->
| Model | Encoder LR | AUC mean ± SD (seeds) | worst fold | Δ vs scratch |
|---|---|---|---|---|
| **Scratch (full model, LR 1e-3)** | — | **0.955 ± 0.001 (n=4)** | 0.949 | — |
| **Masked reconstruction** | **1e-3 (best)** | **0.955 ± 0.001 (n=4)** | 0.951 | **+0.001** |
| Masked reconstruction | 1e-4 | 0.950 ± 0.002 (n=4) | 0.946 | −0.004 |
| Masked reconstruction | 1e-5 | 0.939 ± 0.005 (n=4) | 0.926 | −0.015 |
| Masked reconstruction | frozen (strict) | 0.924 ± 0.001 (n=4) | 0.919 | −0.030 |
| **Contrastive** | **1e-3 (best)** | **0.953 ± 0.001 (n=3)** | 0.950 | **−0.001** |
| Contrastive | 1e-4 | 0.949 ± 0.000 (n=4) | 0.944 | −0.006 |
| Contrastive | 1e-5 | 0.945 ± 0.001 (n=4) | 0.940 | −0.009 |
| Contrastive | frozen (strict) | 0.844 ± 0.009 (n=4) | 0.835 | −0.110 |

(seeds 42–45; seed 42's contrastive-1e-3 cell finishing at time of writing — all other
cells final. Fold-level metrics: `outputs/KY_dual_view_aligned/comparison/`.)
<!-- KY_ALIGNED_TABLE_END -->

## The four slide-ready claims

1. **Protocol >> pretraining.** Aligning to Qianyi's protocol moved KY from ~0.73 to
   ~0.955 AUC (+0.22). The SSL-vs-scratch delta at best settings is ±0.001–0.005.
   The July "SSL hurts on KY" result was a protocol artifact, exactly as Qianyi suspected.
2. **SSL = parity on KY, SSL = win on NYC.** NYC-562: SSL +0.03…+0.05 over scratch
   (real vs seed noise). KY: Δ ≈ 0. Consistent with the thesis condition: SSL pays when
   labeled data is scarce (NYC: 562 samples) and adds nothing when supervision is rich
   (KY: 18.4k samples, ~14.7k train per fold). The natural next probe is a KY
   label-efficiency curve (his `train_label_fraction` machinery).
3. **No encoder-LR collapse under head LR 1e-3** — AUC rises monotonically with encoder
   LR from strict-frozen to 1e-3 for both tasks, best at 1e-3 (top of his stated range).
   Our old "collapse at enc 1e-4" was an artifact of head LR 2e-5, confirming his report.
4. **Frozen-encoder evaluation misleads here** — strict-frozen masked recon loses 0.03,
   contrastive 0.10; fine-tuning the encoder is where the performance lives.

Caveats to state: flat CV is a selection benchmark, not spatial generalization (SCV on
this protocol = follow-up); seeds 42–45 of his 42–46; group-safe split differs from his
plain KFold (strictly harder, so our numbers are if anything conservative); PR-AUC not
comparable to NYC (different class-balance history).

## Label efficiency under the aligned protocol (same-day addendum)

Re-ran the label-efficiency curve with the aligned protocol, reusing the pretrained
encoders (fractions of each fold's train set, stratified, identical subsets across
methods; seeds 42/43, n=2 — preliminary). `F8_label_efficiency.png`,
`outputs/KY_dual_view_aligned/comparison/ky_label_efficiency_summary.csv`:

| train labels/fold | scratch | masked recon | Δ (masked−scratch) |
|---|---|---|---|
| 147 (1%) | 0.820 | 0.829 | **+0.009** |
| 736 (5%) | 0.889 | 0.894 | +0.005 |
| 1.5k (10%) | 0.907 | 0.912 | +0.005 |
| 3.7k (25%) | 0.929 | 0.932 | +0.004 |
| 14.7k (100%) | 0.955 | 0.955 | +0.001 |

**The SSL advantage grows monotonically as labels shrink** — small in absolute terms
but consistent in direction at every fraction, and coherent with NYC-562 (+0.045 at
~450 train labels). n=2 seeds → treat as the preliminary version of the curve; extend
to ≥3 seeds (and 5 like his protocol) as follow-up.

## Reproducibility / what was retained

All fold metrics, predictions paths, training logs, SSL pretraining logs (seeds
42/43), figures, and the run driver are in-repo. **Model checkpoints (SSL encoders +
downstream) were NOT retained** — they died with the pods (terminated to stop
spend). Regenerating any seed's encoder is ~1–2 GPU-hours with
`scripts/run_ky_dual_view_aligned.py --stage ssl`; **stage-90 LSM maps therefore
need a rerun that saves ≥20 downstream checkpoints first** (~$3–4 of GPU).

---

# UPDATE — flat CV (Qianyi's protocol), 2026-07-10 — PRELIMINARY, run in progress (SUPERSEDED by the aligned run above — kept for the protocol-effect comparison)

**Headline: under flat CV the fold noise that plagued every KY comparison is gone
(fold spread ±0.02 vs ±0.08–0.10 under SCV), and the verdict sharpens: SSL still does
not beat scratch on KY — the deficits are now clearly outside noise.**

Setup: same KY KGS six-county 10 m stack (13 terrain + valid-context mask = 14-ch, **ps64**),
**5,000-labeled balanced subsample (1:2 pos:neg preserved) + 3,000 unlabeled**, 60-epoch SSL
pretraining, **flat CV = StratifiedKFold(5, shuffle) on label** (Qianyi's fix: KY positives are
spatially clustered, so SCV folds swing 11–64 % positive; flat folds are a uniform 33 %).
RunPod RTX 4090, CUDA deterministic. Outputs: `outputs/KY_flat_runpod/`.

## Flat-CV table (PRELIMINARY — seeds noted per cell; 3-seed table lands tonight)

| Method | mean AUC (flat) | seeds in | Δ vs scratch | per-fold spread |
|---|---|---|---|---|
| **from-scratch** | **0.733** (0.738 / 0.727) | 2 | — | 0.718–0.763 |
| cross_channel | 0.708 | 1 | −0.030 | 0.692–0.719 |
| strip_jigsaw | 0.694 | 1 | −0.044 | 0.684–0.711 |
| masked_recon | 0.665 | 1 | −0.073 | 0.637–0.703 |
| sequential | (running) | 0 | — | — |

With fold SD ~±0.02, the SSL deficits are outside noise even before all seeds land, and
the *direction* matches the completed 3-seed SCV table below — this is confirmation with
cleaner error bars, not a new phenomenon.

## What flat CV adds (the methodology slide)

| | SCV (spatial folds) | flat CV |
|---|---|---|
| fold positive-rate | 11–64 % (pathological) | uniform 33 % |
| within-seed fold AUC spread | ~0.54–0.87 | 0.72–0.76 |
| scratch AUC | 0.703 ± 0.013 | ~0.733 (prelim) |

- Flat CV is the literature-comparable protocol (what Qianyi and Crawford-style studies use);
  SCV remains the honest spatial-generalization test. **Report both; the ~+0.03 flat−scv gap
  on scratch quantifies spatial-leakage inflation on KY.**
- Flat CV removes the last "it's just fold noise" objection to the KY null result.

## The full-picture story for the meeting (NYC vs KY)

| | NYC (562 labeled) | KY (this cycle) |
|---|---|---|
| SSL−scratch | **+0.03…+0.05 → REAL gain** | ≤0 everywhere; flat CV: −0.03…−0.07 outside noise |
| label-efficiency curve | n/a | **flat — SSL never helps, even at 5 % labels** |
| current explanation | labels scarce → SSL has headroom | negatives too easy → scratch saturates at ~100 labels |

**Next step (unchanged, now better-motivated): harder negatives (PU-bagging / hard-negative
mining) to create a KY regime where the supervised baseline actually struggles — then re-run
SSL vs scratch.** Secondary: one higher-encoder-LR SSL rerun; SCV-vs-flat on the identical
5k subsample for a clean protocol-only comparison.

## Caveats (say these)
- Flat-CV numbers are preliminary (1–2 seeds at slide time; final 3-seed mean ± SD tonight).
- Flat CV permits spatial leakage → absolute AUCs are optimistic by design; the meaningful
  readout is the SSL-vs-scratch delta, which is protocol-consistent.
- The flat run uses a 5,000-sample subsample / 60-ep SSL; the SCV table below used a
  3,198-sample subsample / 30-ep SSL — scratch 0.703→0.733 mixes protocol and sample-set
  changes; the clean protocol-only comparison (same 5k subsample under SCV) is queued.
- KY is 1:2 pos:neg → compare AUC + deltas, not absolute PR-AUC, against NYC's 1:1.

---

# Meeting results — Kentucky (KGS six-county): SSL pretext tasks vs from-scratch

**2026-07-03. Headline: on the Kentucky dataset, SSL pretraining does NOT improve
landslide prediction over a from-scratch baseline** — the opposite of the NYC-562
result. No pretext task clears the seed-to-seed noise; the best (masked-reconstruction)
merely ties scratch, and sequential pretraining significantly *hurts*.

Run: **5 methods × 3 seeds (42/123/7)**, Kentucky KGS six-county inventory (Clay, Leslie,
Perry, Breathitt, Knott, Letcher), **10 m terrain, ps64, 13 terrain + valid-context mask =
14-ch input**, spatially-blocked 5-fold CV. Executed on a rented RunPod **RTX 4090 (CUDA,
deterministic)**. Results: `outputs/KY_kgs6c_ps64_runpod/`, figures `figures/R8seq_KY/`.

⚠️ **This is a first pass on a balanced 3,198-patch subsample** (from the full 27,594 =
9,198 landslide : 18,396 non-landslide, **1:2**), SSL pretrained 30 epochs on 3,000
unlabeled patches. Negatives are buffered-random (not PU-bagged). See Caveats — a
label-efficiency curve is the experiment that will actually explain this result.

## The table — mean AUC over 3 seeds (seed-to-seed SD)

| Method | mean AUC | worst-fold AUC | Δ vs scratch | verdict |
|---|---|---|---|---|
| **from-scratch** | 0.703 (±0.013) | 0.60 | — | baseline |
| masked_recon | **0.710** (±0.013) | 0.56 | +0.007 | within noise |
| cross_channel | 0.684 (±0.024) | 0.54 | −0.019 | within noise |
| strip_jigsaw | 0.661 (±0.031) | 0.54 | −0.043 | within noise* |
| sequential | 0.660 (±0.015) | 0.57 | −0.043 | **SSL hurts** |

Gate: a difference is "real" only if |Δ| > 2× the larger of the two seed-SDs.
\*strip_jigsaw's Δ (−0.043) sits just under its wide gate (0.063) → not significant but
clearly negative-trending. sequential's Δ (−0.043) clears its tighter gate (0.030) → a
real loss.

## Per-seed detail

| method | seed 42 | seed 123 | seed 7 |
|---|---|---|---|
| scratch | 0.696 | 0.722 | 0.692 |
| masked_recon | 0.725 | 0.693 | 0.711 |
| cross_channel | 0.717 | 0.659 | 0.676 |
| strip_jigsaw | 0.619 | 0.669 | 0.694 |
| sequential | 0.678 | 0.642 | 0.660 |

## Verdicts

- **No SSL method beats scratch.** masked_recon ties (+0.007), cross_channel and
  strip_jigsaw trend negative, and **sequential is a significant loss (−0.043)**.
- **SSL is LESS robust on the hard folds here** — scratch's worst spatial fold is 0.60;
  every SSL method's worst fold is *lower* (0.54–0.57). This is the reverse of NYC, where
  SSL held up the hard folds (~0.73–0.85 vs scratch ~0.59).
- The per-seed deltas flip sign method-to-method (e.g. cross_channel +0.021 on seed 42,
  −0.063 on seed 123) — consistent with no true effect plus tiny-fold noise.

## What to say at the meeting

1. **On Kentucky, SSL does not help.** Every pretext task is ≤ from-scratch; sequential
   pretraining actively hurts. This is the clean opposite of the NYC-562 result (where
   masked-recon / cross-channel / sequential all beat scratch by +0.03–0.05).
2. **Scratch is already strong and stable** (0.703 ± 0.013, tight across seeds). When the
   supervised baseline is this solid, SSL has little room to add value — the project's own
   stated precondition ("SSL is only worth it if the supervised baseline struggles").
3. **The likely explanation is label abundance, not a broken pipeline.** KY has ~9k
   positives; SSL's thesis is that it helps when labels are *scarce*. At full labels there
   may be nothing left to gain. → **the label-efficiency curve is the next experiment.**
4. **strip_jigsaw stays the negative control** (worst/near-worst here too), so the ranking
   is internally consistent — this isn't noise masquerading as a result.

## Why KY differs from NYC (state it plainly)

- NYC-562: SSL **+0.03–0.05, clears noise, more robust on hard folds** → SSL helps.
- KY-3198: SSL **≤ 0, within/against noise, less robust on hard folds** → SSL doesn't help.
- Different data regime: KY is a larger, road-corridor-biased, 1:2 inventory at ps64;
  NYC-562 is a small, balanced 1:1 urban set. The contrast is the scientifically
  interesting part, not a contradiction.

## Caveats (say these)

- **First pass on a 3,198-patch subsample** (not the full 27,594) with **30-epoch** SSL
  pretraining — a full-data / longer-pretrain rerun is pending, but float16-cache rounding
  is identical across arms so it does not bias the SSL−scratch delta.
- **Negatives are buffered-random, not PU-bagged.** If they are trivially separable,
  scratch is near-ceiling and SSL has no headroom — a confound to rule out.
- **SSL finetune uses the conservative enc 5e-6 / head 2e-5** (collapse-avoiding LRs). On
  KY this may under-adapt the encoder; one higher-LR rerun would rule that out.
- Compare on **AUC + SSL−scratch delta**, not absolute PR-AUC (1:2 class ratio). Do NOT
  compare to NYC absolute AUC or Crawford's 1.5 m-lidar numbers.

## Next steps (the follow-ups that explain this)

1. **Label-efficiency curve** (headline): scratch vs SSL at 5/10/25/50/100% of labels,
   SSL encoder pretrained once on full unlabeled → does the SSL−scratch gap open at the
   low-label end? This is the paper.
2. **Rule out LR**: one SSL rerun at a higher encoder LR.
3. **Rule out easy-shortcut**: PU-bagged negatives instead of buffered-random; re-inspect
   worst-fold AUCs.

Deliverables: `outputs/KY_kgs6c_ps64_runpod/{ky_results_s*.csv, ky_5method_table.csv,
ky_summary.csv, run_s*.log}`, figures `figures/R8seq_KY/{fig1_mean_auc_bars,
fig2_ssl_minus_scratch, fig3_perfold_spread, fig4_worst_fold_auc}.{png,pdf}`.

---

# UPDATE — label-efficiency curve (2026-07-03)

**The follow-up experiment that was supposed to explain the null result instead deepens
it: SSL provides NO benefit at ANY label budget — the SSL−scratch gap never opens.**

Ran scratch vs masked_recon (the tied-best SSL method) at **5 training-label budgets
(5/10/25/50/100 % → n≈102/204/511/1023/2046 training patches per fold)**, encoder
pretrained once on the full unlabeled pool and reused across budgets, 3 seeds, same spatial
5-fold CV. Results `outputs/KY_label_efficiency/`, figure
`figures/R8seq_KY/fig_label_efficiency_{curve,delta}.png`.

| label budget | train n | scratch | masked_recon | Δ (SSL−scratch) |
|---|---|---|---|---|
| 5 % | 102 | 0.671 ±0.043 | 0.652 ±0.039 | −0.019 |
| 10 % | 204 | 0.680 ±0.015 | 0.661 ±0.025 | −0.019 |
| 25 % | 511 | 0.667 ±0.024 | 0.667 ±0.036 | +0.001 |
| 50 % | 1023 | 0.691 ±0.025 | 0.691 ±0.037 | 0.000 |
| 100 % | 2046 | 0.693 ±0.009 | 0.675 ±0.016 | −0.018 |

## What it means (this is the important part)

1. **The curve is flat and the two lines overlap at every budget.** SSL never beats
   scratch — not even at 5 % labels, where SSL's whole thesis says it should help most.
   The label-abundance hypothesis (follow-up #1) is **ruled out**: there is no label-scarce
   regime here in which SSL adds value.
2. **The task saturates almost immediately.** Scratch reaches ~0.67 with just **102
   training labels** and only crawls to ~0.69 at 2046 — a ~0.02 gain for 20× more labels.
   The model hits its ceiling with ~100 examples.
3. **=> the binding explanation is "the supervised task is too easy," not label scarcity.**
   This points squarely at follow-up #3: the negatives are **buffered-random** (drawn
   area-wide, 650 m from any positive) and are likely **trivially separable** from
   landslide terrain, so a from-scratch net saturates on a handful of labels and SSL has no
   headroom at any budget. The fix is **harder negatives (PU-bagged / hard-negative
   mining)** to create a regime where the baseline actually struggles — only then can SSL's
   value (if any) show up.

## Revised next step

- **Priority = harder negatives (follow-up #3), not more SSL tuning.** Rebuild KY with
  PU-bagged negatives, re-check scratch's label-efficiency curve: if scratch stops
  saturating at 100 labels, re-run SSL vs scratch. This is now the lead, ahead of the
  LR-rerun (follow-up #2), which can't fix a ceiling problem.
- Caveat unchanged: 3198-patch subsample, 30-epoch SSL, conservative LR — but the flat
  curve across a 20× label range is robust to these.

Deliverables (this update): `outputs/KY_label_efficiency/{le_results_s*.csv,
ky_label_efficiency.csv, ky_label_efficiency_summary.csv, le_s*.log}`, figures
`figures/R8seq_KY/fig_label_efficiency_{curve,delta}.{png,pdf}`. Pod (community RTX 3090,
$0.22/hr) terminated; this run cost ~$0.49.
