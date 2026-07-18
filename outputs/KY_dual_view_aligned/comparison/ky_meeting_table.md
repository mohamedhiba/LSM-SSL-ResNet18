# KY dual-view aligned — flat CV (group-safe split), head LR 1e-3, batch 128

Fold-metric files: 5 | fold rows: 180 | seeds present: [42, 43, 44, 45]

| Model | Encoder LR | AUC mean ± SD (seeds) | worst fold | PR-AUC | Δ vs scratch |
|---|---|---|---|---|---|
| Contrastive | frozen (LR 0) | 0.844 ± 0.009 (n=4) | 0.835 | 0.842 | -0.110 |
| Contrastive | enc LR 1e-05 | 0.944 ± 0.002 (n=4) | 0.936 | 0.950 | -0.010 |
| Contrastive | enc LR 0.0001 | 0.949 ± 0.000 (n=4) | 0.944 | 0.954 | -0.006 |
| Contrastive | enc LR 0.001 | 0.953 ± 0.001 (n=4) | 0.950 | 0.959 | -0.001 |
| Masked reconstruction | frozen (LR 0) | 0.924 ± 0.001 (n=4) | 0.919 | 0.928 | -0.030 |
| Masked reconstruction | enc LR 1e-05 | 0.939 ± 0.005 (n=4) | 0.926 | 0.945 | -0.015 |
| Masked reconstruction | enc LR 0.0001 | 0.950 ± 0.002 (n=4) | 0.946 | 0.956 | -0.004 |
| Masked reconstruction | enc LR 0.001 | 0.955 ± 0.001 (n=4) | 0.951 | 0.961 | +0.001 |
| Scratch (full model, LR 1e-3) | scratch | 0.955 ± 0.001 (n=4) | 0.949 | 0.960 |  |

Scratch reference: 0.955 ± 0.001

Rule: a delta is real only if it clears ~2x the seed-to-seed SD.
Do NOT compare PR-AUC to NYC numbers (different balance history).
