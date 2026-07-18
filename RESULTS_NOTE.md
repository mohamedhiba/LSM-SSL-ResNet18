# Results Note — Two New SSL Pretext Tasks (Phase 1)

This note documents the design, evaluation, and a non-obvious training subtlety
for two self-supervised pretext tasks added on top of the original five-method
ps32 ResNet-18 comparison: **cross-channel masking** and **1D strip jigsaw**.
It records the full reasoning arc, because the headline conclusion *changed*
twice during evaluation, and the final claims depend on understanding why.

All experiments use 14-channel 30 m terrain patches (32×32), the existing
5-cluster spatial cross-validation (SCV) protocol, and a modified ResNet-18.
The two new methods were pretrained on Apple Silicon (MPS); the five archived
methods were trained on CUDA. The SCV protocol is identical, so cross-hardware
comparisons are indicative rather than exact.

> **Update — current pipeline (Phase 2).** The R2 and R3 results in this note were
> produced with the **old 14-channel input** (13 terrain factors **including
> landcover**, no mask channel). The current code uses a **different** input:
> **13 terrain channels (landcover dropped) + 1 binary valid-context mask channel
> = 14 total**. Patches now require only a valid center pixel; boundary/local
> NoData is zero-padded and flagged by the mask channel (1 = real, 0 = padded).
> The cross-channel default masked channel is now index 12 (TWI). Because the
> channel set changed, the R2/R3 numbers below are **no longer directly
> comparable** to current-code results, and the archived 14-channel SSL encoders
> no longer load (conv1 shape differs). Note: `create_unlabeled_patch_index` is
> still interior-only, so the mask channel is all-ones during SSL pretraining on
> the current unlabeled index; it varies on the downstream labeled patches.

---

## 1. The two new pretext tasks

**Cross-channel masking** (`src/ssl_cross_channel.py`)
Zero one entire conditioning-factor channel in the normalized 14-channel input
and reconstruct it from the remaining 13; MSE loss on the masked channel only.
The input stays 14-channel (masked channel zeroed) so the encoder transfers
directly to the 14-channel downstream classifier. The masked channel index is
configurable (default 13 = TWI). Reconstruction-based, same family as masked
reconstruction.

**1D strip jigsaw** (`src/ssl_strip_jigsaw.py`)
Split each 32×32 patch into 3 horizontal strips (heights 11/11/10), shuffle by
one of the 3! = 6 fixed permutations, predict the permutation class. A coarse
spatial-ordering task (same family as the existing Jigsaw/Rotation tasks). The
unequal last strip leaves a mild strip-size cue — an accepted limitation,
consistent with how the repo already reports pretext solvability.

---

## 2. Final 7-method comparison (ps32, 5-cluster SCV)

| Rank | Model | Pretext family | Mean AUC | AUC SD | Worst-fold AUC | Mean PR-AUC | F1 (tuned) |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Contrastive learning | Contrastive | 0.838 | 0.085 | 0.704 | 0.831 | 0.675 |
| 2 | Masked reconstruction | Reconstruction | 0.821 | 0.043 | 0.786 | 0.814 | 0.667 |
| 3 | **Cross-channel masking** | **Reconstruction** | **0.810** | 0.104 | 0.679 | 0.810 | 0.722 |
| 4 | Scratch | None | 0.779 | 0.105 | 0.624 | 0.771 | 0.657 |
| 5 | Rotation prediction | Geometric | 0.726 | 0.092 | 0.602 | 0.702 | 0.634 |
| 6 | Jigsaw | Spatial-ordering | 0.723 | 0.105 | 0.606 | 0.715 | 0.634 |
| 7 | **Strip jigsaw** | **Spatial-ordering** | **0.691** | 0.143 | 0.511 | 0.679 | 0.685 |

Source: `outputs/R2_new_ssl_tasks_diag/summary_all/`. Canonical notebook:
`notebooks/16_compare_all_seven_ssl_tasks_ps32.ipynb`.

**Headline findings**
1. **Cross-channel masking is effective** (3rd of 7, +0.031 AUC over Scratch),
   and fits the existing thesis: reconstruction-style pretexts align well with
   14-factor terrain patches (it sits right beside masked reconstruction).
2. **Strip jigsaw is weak** (7th, below Scratch), consistent with the prior
   finding that spatial-ordering pretexts (Jigsaw, Rotation) transfer poorly here.

One didn't work, one did — and the one that did is competitive with the SSL
leaders.

---

## 3. The training subtlety that changed the conclusion

The cross-channel conclusion was **not** stable across finetune settings. Three
runs of the *same* method gave three different stories:

| Run | SSL pretrain | Finetune | head LR | Cross-channel mean AUC |
|---|---|---|---|---:|
| smoke | 2 ep | 6 ep | 1e-4 | 0.744 |
| full | 50 ep | 100 ep | 1e-4 | **0.402** (below random) |
| diag | 50 ep | 60 ep | **2e-5** | **0.810** |

Under the **standard** finetune LR (1e-4, what the archived five used), the
heavily-converged 50-epoch cross-channel encoder **collapsed**: probabilities
saturated near 0 (so F1@0.5 = 0.000), 3/5 folds early-stopped at epoch 1, and
the pooled AUC fell *below random* with inverted ranking. The smoke run's 0.744
was misleadingly healthy only because 6 epochs hadn't yet triggered the collapse.

Diagnosis: the collapse is a **finetune learning-rate instability**, not a bad
pretext. The encoder is fine — at a reduced LR (head 2e-5, encoder 5e-6) it
trains stably (val AUC 0.82–0.90) and reaches 0.810. The diagnostic re-finetuned
the *same* full-run encoders and only changed the LR (`outputs/R2_new_ssl_tasks_diag/`).

Two practical notes:
- **F1@0.5 vs tuned threshold.** Because the probabilities saturate, F1 must be
  read at a validation-selected threshold (`f1_best_f1`), reported as
  "F1 (tuned)" above. F1@0.5 is retained as a secondary column to make the
  saturation visible. AUC/PR-AUC are threshold-independent and unaffected.
- **Fairness caveat.** The two new methods use a lower finetune LR than the
  archived five. A fully hyperparameter-matched comparison would re-finetune all
  seven at a common LR, but the archived five have no checkpoints in this package.
  The new rows are therefore *indicative*. The robust statement is:
  *cross-channel masking's downstream finetuning is more LR-sensitive; under a
  stable LR it is competitive with the leading SSL methods.*

---

## 4. Mechanistic evidence: mask-channel ablation

To test *why* cross-channel masking works — does it learn physical terrain
structure, or just generic statistics? — we masked six different channels (one
ablation point each), pretrained an encoder for each, and fine-tuned under the
stable LR. Script: `scripts/ablate_cross_channel_mask.py`. Outputs:
`outputs/R3_cross_channel_ablation/`.

| Masked channel | Category | Mean AUC | Worst-fold | Pretext val_loss |
|---|---|---:|---:|---:|
| profile curvature | topographic | 0.833 | 0.670 | 0.221 |
| slope | topographic | 0.819 | 0.690 | 0.050 |
| TWI | hydro-topographic | 0.780 | 0.658 | 0.158 |
| lithology | categorical | 0.718 | 0.556 | 0.554 |
| NDVI | vegetation | 0.582 | 0.309 | 0.357 |
| Landcover | categorical | 0.403 | 0.202 | 0.590 |

```
Terrain-derivative masks (slope, curvature, TWI):  mean AUC 0.811  (0.780–0.833)
Non-terrain masks (lithology, NDVI, landcover):    mean AUC 0.568  (0.403–0.718)
```

A **0.24 AUC gap** between the groups. Masking a channel that *is* terrain
physics (slope, curvature, TWI) — recoverable from the other terrain
derivatives, and landslide-relevant — produces strongly transferable encoders.
Masking a terrain-independent channel (Landcover, 0.403, below random) gives a
near-dead pretext: land use cannot be predicted from slope, so there is little
learnable signal.

**Conclusion.** Cross-channel masking works because it forces the encoder to
model inter-factor terrain structure. The transferable signal lives specifically
in the topographically-coupled channels.

**Nuances (for honesty in the write-up):**
- Not a clean monotonic "predictable → transferable" law: lithology is nearly
  unreconstructible (pretext loss 0.554) yet still transfers decently (0.718),
  likely because lithology is *directly* landslide-causal. The precise principle
  is: transfer tracks how much masking that channel forces modeling of
  *landslide-relevant terrain structure*.
- Terrain channels are both "physical" and the most landslide-correlated, so
  "learned terrain physics" and "learned landslide-relevant features" are not
  fully separable here. The terrain-vs-landuse contrast nonetheless makes the
  terrain coupling the clear driver.

---

## 5. Reproduction

Native arm64 PyTorch (MPS) required on macOS; the conda `environment.yml` pins a
CUDA build. See README §13.

```bash
# Pretrain (50 ep each) + finetune (full namespace)
python scripts/pretrain_cross_channel_ssl.py --run-tag full
python scripts/pretrain_strip_jigsaw_ssl.py  --run-tag full
python scripts/finetune_new_ssl_tasks.py --task both --run-tag full

# Stable-LR diagnostic re-finetune of the full encoders (the canonical numbers)
python scripts/finetune_new_ssl_tasks.py --task both \
    --encoder-run-tag full --run-tag diag \
    --max-epochs 60 --early-stopping-patience 12 \
    --head-learning-rate 2e-5 --encoder-learning-rate 5e-6

# Mask-channel ablation
python scripts/ablate_cross_channel_mask.py

# Rebuild the canonical 7-method comparison (reads the diag namespace)
R2_NAMESPACE=R2_new_ssl_tasks_diag python scripts/_build_notebook16.py
# then execute notebooks/16_compare_all_seven_ssl_tasks_ps32.ipynb
```

**Performance note.** Patch loading is I/O-bound (14 GeoTIFF window reads per
patch ≈ 34 patches/s ≈ 8 h per pretrain). An in-RAM patch cache
(`cache_in_memory=True`, ~1.15 GB, verified bit-identical to the uncached path)
collapses loading to seconds; it is the reason these runs complete in ~minutes
of compute on the M3 rather than hours. DataLoader workers do *not* help (the
14 shared GeoTIFFs are the contention point).

---

## 6. Artifact map

| Path | Contents |
|---|---|
| `outputs/R2_new_ssl_tasks_smoke/` | 2+6 epoch sanity run (ignore the numbers) |
| `outputs/R2_new_ssl_tasks_full/` | 50+100 epoch run, standard LR (cross-channel collapse) |
| `outputs/R2_new_ssl_tasks_diag/` | 50 ep + stable-LR finetune — **canonical results** |
| `outputs/R3_cross_channel_ablation/` | mask-channel ablation (mechanistic evidence) |
| `notebooks/16_..._ps32.ipynb` | canonical 7-method comparison (diag) |
| `notebooks/16_..._ps32_full.ipynb` | full-run (collapsed) comparison, preserved |
| `notebooks/16_..._ps32_smoke.ipynb` | smoke comparison, preserved |
