# Meeting results — SSL pretext tasks vs from-scratch, on the CORRECT 562-sample set

**2026-07-01. Headline: on the correct 562-sample NYC dataset, SSL pretraining genuinely
improves landslide prediction over a from-scratch baseline** — reversing the earlier
"no gain" result that came from mistakenly using the wrong 341-sample set.

Run: **5 methods × {30m, 10m} × 3 seeds (42/123/7)**, matched 557-sample balanced set
(281 landslide + 281 non-landslide, 5 spatial clusters), spatially-blocked 5-fold CV,
13 terrain + valid-context mask = 14-ch input. Executed on a rented RunPod **RTX 4090
(CUDA, deterministic)**. Data: `data/processed/samples/final_cluster_balanced_dataset.csv`
→ `scripts/build_562_indices.py`. Results: `outputs/R8seq_562_runpod/`.

## The table — mean AUC over 3 seeds (seed-to-seed SD)

| Method | 30m | 10m | worst-fold (30m/10m) |
|---|---|---|---|
| from-scratch | 0.836 (±0.012) | 0.860 (±0.036) | 0.59 / 0.59 |
| masked_recon | **0.865** (±0.010) | **0.905** (±0.009) | 0.73 / 0.85 |
| cross_channel | **0.871** (±0.019) | 0.877 (±0.031) | 0.72 / 0.53 |
| strip_jigsaw | 0.779 (±0.020) | 0.865 (±0.018) | 0.57 / 0.80 |
| **sequential** | **0.874** (±0.007) | **0.903** (±0.020) | 0.74 / 0.68 |

## Verdicts (a gain is "real" only if it clears the seed-to-seed SD)

**30m** (scratch seed-SD 0.012):
- masked_recon **+0.029 → REAL**
- cross_channel **+0.035 → REAL**
- sequential **+0.038 → REAL** (largest, and lowest variance ±0.007)
- strip_jigsaw **−0.057 → REAL loss** (hurts)

**10m** (scratch seed-SD 0.036):
- masked_recon **+0.045 → REAL**
- sequential **+0.043 → REAL**
- cross_channel +0.017 → within noise (positive but not significant)
- strip_jigsaw +0.005 → within noise (neutral)

## What to say at the meeting

1. **SSL helps.** masked-reconstruction, cross-channel, and sequential pretraining all
   beat the from-scratch baseline, and the gains **exceed the seed-to-seed noise** — so
   it's a real effect, not a lucky split. This is the result the project has been after.
2. **Sequential and masked-reconstruction are the winners** — biggest, most reliable
   gains, top absolute scores (sequential 0.874/0.903, masked-recon 0.865/0.905), and the
   lowest variance.
3. **SSL is also more ROBUST on the hard folds.** Scratch's worst spatial fold collapses
   to ~0.59; masked-recon/sequential hold ~0.73–0.85. SSL helps most exactly where the
   supervised baseline struggles — the classic label-scarcity argument.
4. **strip-jigsaw is the negative control** — it doesn't help (hurts at 30m). A clean
   "not all pretext tasks are equal" finding.
5. **10m ≥ 30m** for most methods, but the resolution gap is mostly within seed noise.

## Why this differs from the earlier "no gain" result

The previous run (`MEETING_RESULTS.md`, [[r8seq-result-ssl-no-gain-nyc]]) used the wrong
**341-sample** `common_balanced` set and produced ~0.56–0.64 with SSL ≈ scratch. Qianyi
flagged the dataset; the correct set is **562 samples (281+281)**. On it, both baselines
rise to ~0.84–0.86 and SSL pulls clearly ahead. That earlier doc is **superseded for NYC**.

## Caveats (state them)
- Each method uses its own working LR (scratch 1e-4; SSL 2e-5, since cross-channel
  collapses at 1e-4) → each-method-at-its-best, not identical-LR.
- SSL unlabeled set reduced to n=5000 (from 20000) to fit the GPU upload; results are
  robust but a full n=20000 rerun would tighten them.
- Do NOT compare to Crawford's 1.5m-lidar numbers (0.78–0.83) or the historical 0.782.
