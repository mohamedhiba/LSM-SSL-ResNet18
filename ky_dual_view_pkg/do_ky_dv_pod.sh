#!/bin/bash
# KY dual-view protocol-aligned run — self-contained pod bootstrap.
# Expects /workspace/kydv/ky_dv_payload.tar.gz already transferred.
# Usage: bash do_ky_dv_pod.sh            (bootstrap + smoke only)
#        bash do_ky_dv_pod.sh launch     (bootstrap + smoke + launch full priority runs)
set -e
cd /workspace/kydv

if [ ! -d LSM_SSL_ResNet18_10m_dual_view_code ]; then
  tar xzf ky_dv_payload.tar.gz
fi

pip install -q "numpy<2" pandas scikit-learn scipy rasterio tqdm matplotlib gdown 2>&1 | tail -1 || true

if [ ! -d processed/rasters_cleaned_10m ]; then
  echo "pulling KY rasters via gdown..."
  gdown "${KY_RASTERS_GDRIVE_ID:?set to the KY processed_data.zip Drive file id (ask the data owner)}" -O processed_data.zip
  unzip -q processed_data.zip || python -m zipfile -e processed_data.zip .
fi
NTIF=$(ls processed/rasters_cleaned_10m/*.tif 2>/dev/null | wc -l)
echo "raster_tifs=$NTIF (expect 13)"
[ "$NTIF" -eq 13 ] || { echo "RASTER_COUNT_WRONG"; exit 1; }

export KY_RASTER_DIR=/workspace/kydv/processed/rasters_cleaned_10m
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

echo "running pod smoke test..."
python scripts/run_ky_dual_view_aligned.py --stage smoke > smoke.log 2>&1 || {
  echo "POD_SMOKE_FAILED"; tail -30 smoke.log; exit 1; }
grep -q SMOKE_PASSED smoke.log && echo "POD_SMOKE_PASSED"

if [ "$1" = "launch" ]; then
  bash launch_ky_dv.sh
fi
