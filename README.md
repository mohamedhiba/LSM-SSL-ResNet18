# Self-Supervised Learning for Landslide Susceptibility Mapping (ResNet-18)

SSL pretraining on multi-channel terrain rasters for landslide prediction,
evaluated across two study areas (Greater New York, eastern Kentucky) under
multi-seed cross-validation. CCNY Media Lab (Prof. YingLi Tian).

## Start here — current results (2026-07-17)

| Doc | What it is |
|---|---|
| **`KY_ALIGNED_PRESENTATION_20260717.md`** | **Latest full result**: Kentucky under the protocol-aligned dual-view pipeline — SSL reaches parity with scratch (0.955 AUC), protocol alignment worth +0.22, label-efficiency curve. Figures embedded. |
| `MEETING_RESULTS_KY.md` | Kentucky writeup + history (aligned run, July flat-CV/SCV runs) |
| `MEETING_RESULTS_562.md` | NYC 562-sample result: SSL beats scratch +0.03–0.05 |
| `KY_DUAL_VIEW_ALIGNMENT.md` | The verified protocol spec (extracted from collaborator code) |
| `CLAUDE.md` | Full project orientation: pipeline, conventions, session history |

**Headline finding: SSL pays where labels are scarce** — it beats scratch on NYC
(562 labels, +0.03–0.05), reaches parity on Kentucky (18,396 labels, ±0.001), and
its Kentucky advantage grows monotonically as labels shrink (+0.009 at 147
labels/fold).

**Data availability:** raw/processed rasters (~24 GB), patch caches, and model
checkpoints are not in this repo. `data/kentucky_dual_view/` (indices, group-safe
CV split, normalization stats, PU-scored candidates) is included since it is
expensive to regenerate. The Kentucky runs also require the collaborator dual-view
code package (`LSM_SSL_ResNet18_10m_dual_view_code/`, Qianyi Liu — not
redistributed here) placed at the repo root.

---

# Original reproducibility package (NYC 30 m pipeline)

This package contains the code and notebooks needed to reproduce the landslide susceptibility modeling experiments based on multi-channel raster patches and ResNet-18.

> **Pipeline update (current code).** The CNN input is now **13 terrain channels + 1 binary valid-context mask channel = 14 channels total**. Landcover was dropped from the input (categorical, human-modifiable, not recoverable from terrain derivatives); `Landcover.tif` remains on disk but is filtered out by name in `list_raster_files`. Patches now require only a **valid center pixel** — boundary/local NoData is zero-padded and flagged by the appended mask channel (1 = real, 0 = padded). A two-stage sequential pretraining path was added (`scripts/pretrain_sequential_ssl.py`: masked reconstruction → cross-channel masking). The historical **R1/R2/R3** results below were produced with the **old 14-channel setup (with landcover, no mask channel)** and are **not directly comparable** to results from the current code. Note: `create_unlabeled_patch_index` (SSL unlabeled sampling) is still interior-only, so the mask channel is all-ones during SSL pretraining on the current unlabeled index; it varies on the downstream labeled patches.

The project compares five ps32 ResNet-18 models under the same 5-cluster spatial cross-validation protocol:

1. Scratch ResNet-18 baseline
2. Masked-reconstruction-pretrained ResNet-18
3. Contrastive-pretrained ResNet-18
4. Jigsaw-pretrained ResNet-18
5. Rotation-pretrained ResNet-18

## 1. Package Contents

```text
LSM_SSL_ResNet18_repro_package_20260515/
|-- README.md
|-- environment.yml
|-- requirements.txt
|-- src/
|-- notebooks/
|-- data/
|-- results_summary/
|   |-- tables/
|   `-- figures/
`-- data_placeholders/
```

Important note: this package now includes the `data/` folder needed for reproduction. It still does not include the large trained model checkpoints or the full historical output folders. It includes code, notebooks, input/intermediate data, and final summary tables/figures.

## 2. Python Environment Used

The experiments were run with this local conda environment:

```text
Environment name: DL2
Python executable: D:\ProgramData\miniconda3\envs\DL2\python.exe
Python version: 3.9.24
PyTorch: 2.3.1
CUDA used by PyTorch: 11.8
GPU: NVIDIA GeForce RTX 4090
OS: Windows
```

Core package versions:

```text
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1
numpy==1.24.3
pandas==2.3.3
scikit-learn==1.5.1
scipy==1.10.1
rasterio==1.4.3
matplotlib==3.9.2
pillow==11.1.0
nbformat==5.10.4
nbclient==0.10.2
ipykernel==6.30.1
jupyter_core==5.8.1
tqdm==4.67.3
affine==2.4.0
```

For full raw geospatial preprocessing, install these additional GIS packages:

```text
geopandas
shapely
fiona
pyproj
gdal
```

They are mainly needed by early raster/vector preprocessing and PU-Bagging stages. The CNN/SSL training and final comparison stages do not require geopandas directly.

## 3. Recommended Installation

Use conda if possible, especially on Windows and when using CUDA/GIS libraries.

```bash
conda env create -f environment.yml
conda activate LSM_SSL_ResNet18
python -m ipykernel install --user --name LSM_SSL_ResNet18 --display-name "LSM_SSL_ResNet18"
```

If using the original local environment, activate:

```bash
conda activate DL2
```

The `requirements.txt` file is provided as a pip-oriented reference, but conda is preferred for PyTorch CUDA and GIS dependencies.

## 4. Required Data Layout

This transfer package already includes the project `data/` folder. If moving files manually, keep this structure at the project root:

```text
data/
|-- raw/
|   `-- samples/
|       |-- landslide samples.csv
|       `-- landslide_points.csv            # if using older notebook versions
`-- processed/
    |-- rasters_cleaned/
    |   `-- *.tif                           # exactly 14 aligned cleaned GeoTIFFs
    |-- pu_bagging/
    |-- samples/
    |-- patches/
    |-- ssl_unlabeled_indices/
    `-- ssl_pretext_configs/
```

The cleaned raster folder contains the aligned GeoTIFF rasters sorted alphabetically. `Landcover.tif` may remain in this folder but is excluded from the CNN channel stack by name (see `EXCLUDED_RASTER_NAME_STEMS` in `src/patch_dataset.py`), leaving **13 terrain channels**; the dataset appends a 14th binary valid-context mask channel at load time. The tabular `factor_01 .. factor_14` sample columns are separate metadata and are unchanged.

Expected raster assumptions:

```text
same CRS
same transform
same width and height
same resolution
same bounds
nodata = -9999
```

## 5. Notebook Running Order

Run the notebooks in this order if reproducing from scratch:

```text
01_pu_bagging_reliable_negative_selection.ipynb
02_kmeans_cluster_balancing.ipynb
03_patch_dataset_preparation.ipynb
04_common_valid_subset_and_rebalance.ipynb

05_resnet18_scratch_scv_baseline.ipynb

06_ssl_pretrain_masked_reconstruction_ps32.ipynb
07_finetune_masked_recon_resnet18_ps32_scv.ipynb

08_ssl_pretrain_contrastive_ps32.ipynb
09_finetune_contrastive_resnet18_ps32_scv.ipynb

10_compare_ssl_tasks_ps32.ipynb

11_ssl_pretrain_jigsaw_ps32.ipynb
12_finetune_jigsaw_resnet18_ps32_scv.ipynb

13_ssl_pretrain_rotation_ps32.ipynb
14_finetune_rotation_resnet18_ps32_scv.ipynb

15_compare_all_ssl_tasks_ps32.ipynb
```

The `.ipynb` notebooks are the main workflow files. The `.py` files in `src/` are reusable modules imported by the notebooks. In normal use, you run the notebooks; the notebooks automatically call the relevant `.py` functions and classes.

## 6. What Each Stage Does

```text
01-04: data preparation
  01: PU-Bagging reliable negative sample selection
  02: balanced spatial clustering and labeled sample creation
  03: patch index generation for 16, 32, and 64 pixel patches
  04: common-valid, cluster-wise balanced patch subset

05: supervised scratch ResNet-18 baseline

06-07: masked reconstruction SSL pretraining and supervised fine-tuning

08-09: contrastive SSL pretraining and supervised fine-tuning

10: intermediate comparison for Scratch, Masked reconstruction, Contrastive

11-12: Jigsaw SSL pretraining and supervised fine-tuning

13-14: Rotation prediction SSL pretraining and supervised fine-tuning

15: final all-task comparison and manuscript-ready figures/tables
```

## 7. Important Output Files

Final comparison tables:

```text
outputs/R1_ssl_task_comparison/summary_all/all_ssl_tasks_ps32_fold_metrics.csv
outputs/R1_ssl_task_comparison/summary_all/all_ssl_tasks_ps32_summary_metrics.csv
outputs/R1_ssl_task_comparison/summary_all/pretext_transfer_summary_ps32.csv
outputs/R1_ssl_task_comparison/summary_all/Table_R1_all_ssl_tasks_ps32_manuscript.csv
```

Final comparison figures:

```text
figures/R1_ssl_task_comparison/summary_all/Fig_R1a_all_tasks_mean_auc_ps32.png
figures/R1_ssl_task_comparison/summary_all/Fig_R1b_all_tasks_foldwise_auc_ps32.png
figures/R1_ssl_task_comparison/summary_all/Fig_R1c_auc_stability_tradeoff_ps32.png
figures/R1_ssl_task_comparison/summary_all/Fig_R1d_worstfold_stability_ps32.png
figures/R1_ssl_task_comparison/summary_all/Fig_R1e_pooled_roc_all_tasks_ps32.png
figures/R1_ssl_task_comparison/summary_all/Fig_R1f_pooled_pr_all_tasks_ps32.png
figures/R1_ssl_task_comparison/summary_all/Fig_R1g_pretext_transfer_matrix_ps32.png
figures/R1_ssl_task_comparison/summary_all/Fig_R1h_metric_heatmap_all_tasks_ps32.png
```

Copies of these final tables and figures are included in:

```text
results_summary/tables/
results_summary/figures/
```

Large trained checkpoints are not included. To reproduce model checkpoints, rerun the relevant training notebooks:

```text
05_resnet18_scratch_scv_baseline.ipynb
07_finetune_masked_recon_resnet18_ps32_scv.ipynb
09_finetune_contrastive_resnet18_ps32_scv.ipynb
12_finetune_jigsaw_resnet18_ps32_scv.ipynb
14_finetune_rotation_resnet18_ps32_scv.ipynb
```

## 8. Final Experimental Results

All models use patch size 32 and the same 5-cluster spatial cross-validation protocol.

| Model | Pretext task | Mean AUC | AUC SD | Worst-fold AUC | Mean PR-AUC | Mean F1 at 0.5 | Main finding |
|---|---|---:|---:|---:|---:|---:|---|
| Scratch | None | 0.779 | 0.105 | 0.624 | 0.771 | 0.664 | Baseline; high spatial fold variability |
| Masked reconstruction | Masked reconstruction | 0.821 | 0.043 | 0.786 | 0.814 | 0.711 | Most stable transfer and best worst-fold AUC |
| Contrastive learning | Contrastive learning | 0.838 | 0.085 | 0.704 | 0.831 | 0.726 | Highest mean AUC |
| Jigsaw | Jigsaw | 0.723 | 0.105 | 0.606 | 0.715 | 0.641 | Weak transfer; below Scratch |
| Rotation prediction | Rotation prediction | 0.726 | 0.092 | 0.602 | 0.702 | 0.586 | Pretext solved, but weak downstream transfer |

## 9. Main Scientific Conclusions

1. Both masked reconstruction and contrastive learning improve mean AUC over the supervised Scratch baseline.
2. Contrastive learning achieves the highest mean AUC, indicating the best average discrimination.
3. Masked reconstruction achieves the lowest AUC SD and the highest worst-fold AUC, indicating the strongest spatial generalization stability.
4. Jigsaw underperforms Scratch, consistent with its near-random pretext-task performance.
5. Rotation prediction solves its pretext task almost perfectly, but transfers poorly to downstream landslide susceptibility classification.
6. Reconstruction- and contrastive-based SSL tasks are better aligned with 14-factor geospatial raster patches than spatial-ordering or simple geometric rotation tasks.

## 10. Practical Notes for Reproducibility

For a clean rerun, execute notebooks with the kernel using the conda environment above.

If you want the notebooks to show outputs after running, save the executed notebooks and do not clear outputs. A useful instruction is:

```text
Run all cells, save the executed notebook in place, and preserve cell outputs and execution counts.
```

Some early notebooks in the original working folder may not display saved cell outputs even though their output files exist. This does not mean the workflow failed; it means the notebook outputs were not saved into the `.ipynb` file.

## 11. Quick Start for Final Figure/Table Reproduction Only

If all model prediction and metric CSV files already exist, run only:

```text
15_compare_all_ssl_tasks_ps32.ipynb
```

This regenerates the final all-task summary tables and figures without training any models.

## 12. Quick Start for Full Reproduction

1. Create the environment from `environment.yml`.
2. Place raw samples and cleaned rasters under the expected `data/` structure.
3. Run notebooks `01` through `15` in order.
4. Check final results under:

```text
outputs/R1_ssl_task_comparison/summary_all/
figures/R1_ssl_task_comparison/summary_all/
```

## 13. Phase 1 Extension: Two New SSL Pretext Tasks (Cross-channel masking, Strip jigsaw)

Two additional self-supervised pretext tasks were added on top of the original
five-method comparison:

```text
Cross-channel masking  src/ssl_cross_channel.py
  Zero one entire terrain channel (default TWI, channel index 12 after landcover
  is dropped) in the normalized input and predict it from the remaining 12
  terrain channels plus the valid-context mask. MSE loss on the masked channel
  only. The masked channel index is configurable. The input is 14-channel
  (13 terrain + valid-context mask) so the pretrained encoder transfers directly
  to the 14-channel downstream classifier; the mask channel is never masked.

Strip jigsaw           src/ssl_strip_jigsaw.py
  Split each 32x32 patch into 3 horizontal strips (heights 11/11/10), shuffle by
  one of the 3! = 6 fixed permutations, and predict the permutation class.
```

New training loops live in `src/train_ssl.py`
(`train_cross_channel_model`, `train_strip_jigsaw_model`); both reuse the
existing checkpoint/log conventions and save an `encoder_state_dict` checkpoint
compatible with `load_ssl_encoder_weights_into_classifier`.

### 13.1 Apple Silicon (MPS) note

`src/utils.py::get_device()` now selects CUDA, then Apple MPS, then CPU, and
`set_global_seed` seeds the MPS generator. The conda `environment.yml` pins a
CUDA PyTorch build that does not install on macOS; on Apple Silicon, use a
native arm64 PyTorch wheel (e.g. `torch==2.3.1`) so the MPS backend is
available. Example:

```bash
python3.11 -m venv .venv            # native arm64 interpreter
.venv/bin/pip install torch==2.3.1 torchvision==0.18.1 "numpy<2" \
    pandas scikit-learn rasterio matplotlib tqdm affine scipy nbformat nbclient ipykernel
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

### 13.2 Run order for the two new tasks

```bash
python scripts/pretrain_cross_channel_ssl.py        # SSL pretrain (50 ep)
python scripts/pretrain_strip_jigsaw_ssl.py         # SSL pretrain (50 ep)
python scripts/finetune_new_ssl_tasks.py --task both  # 5-cluster SCV fine-tune
# then run notebooks/16_compare_all_seven_ssl_tasks_ps32.ipynb top-to-bottom
```

All scripts use `--num-workers 0` (lazy rasterio handles are not fork-safe) and
`drop_last=True` on training loaders (BatchNorm safety on small final batches).

### 13.3 Seven-method comparison

`notebooks/16_compare_all_seven_ssl_tasks_ps32.ipynb` builds the seven-method
table. It reuses the archived fold metrics
(`results_summary/tables/all_ssl_tasks_ps32_fold_metrics.csv`) for the original
five methods and the newly computed fold metrics/predictions for the two new
methods. Pooled ROC/PR curves cover the two new methods only (per-fold
predictions for the archived five are not shipped); fold-wise SCV metrics remain
the primary evaluation. Outputs:

```text
outputs/R2_new_ssl_tasks/summary_all/all_seven_ssl_tasks_ps32_fold_metrics.csv
outputs/R2_new_ssl_tasks/summary_all/all_seven_ssl_tasks_ps32_summary_metrics.csv
outputs/R2_new_ssl_tasks/summary_all/Table_R2_all_seven_ssl_tasks_ps32_manuscript.csv
figures/R2_new_ssl_tasks/summary_all/
```

Hardware-mix caveat: the five archived methods were trained on CUDA (RTX 4090);
the two new methods were pretrained on Apple Silicon (MPS). The downstream
fine-tuning protocol and 5-cluster SCV are identical, so cross-hardware
comparisons are indicative rather than exact.

## 14. Phase 2: 13-channel input, valid-context mask, sequential pretraining

The current pipeline differs from the R1/R2/R3 experiments above (which used the
old 14-channel input *with landcover and no mask channel*); their numbers are
**not directly comparable** to runs from the current code.

```text
Channels      13 terrain factors + 1 binary valid-context mask = 14 total.
              Landcover dropped (excluded by name; file may stay on disk).
Validity      Only the center pixel must be valid. Boundary/local NoData is
              zero-padded; the mask channel marks 1 = real pixel, 0 = padded.
              Normalization stats are computed over valid pixels only.
Cross-channel Default masked channel is now index 12 (TWI); the mask channel
              (index 13) is never the masked target.
```

Sequential two-stage pretraining (`scripts/pretrain_sequential_ssl.py`): Stage 1
warms up the encoder with a stable pretext (masked reconstruction by default, or
`--stage1-task contrastive`) for N epochs; Stage 2 loads that encoder with no
reset, switches to cross-channel masking at a reduced LR (default 2e-5; it
collapses at 1e-4), and continues for M epochs. Only the encoder carries over.
`--finetune` then runs the 5-cluster SCV fine-tuning into the `R4_sequential_ssl`
namespace.

```bash
python scripts/pretrain_sequential_ssl.py \
    --stage1-epochs 50 --stage2-epochs 50 --stage2-lr 2e-5 --finetune
```

Caveat: `create_unlabeled_patch_index` (SSL unlabeled sampling) is still
interior-only, so during SSL pretraining on the current unlabeled index the mask
channel is all-ones; it varies on the downstream labeled patches, where edge
samples are retained.
