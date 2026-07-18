#!/bin/bash
# Launch the KY priority grid as 4 detached processes on one GPU.
#   P0: scratch baseline, 3 seeds x 5 folds (baseline column lands first)
#   P1-3: one per seed — SSL pretrain (masked_recon + contrastive, 50k corpus)
#         then downstream encoder-LR {0, 1e-5, 1e-4} x 5 folds, head LR 1e-3.
# All fds redirected so SSH drops cannot kill or hang anything.
cd /workspace/kydv
export KY_RASTER_DIR=/workspace/kydv/processed/rasters_cleaned_10m
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

nohup python scripts/run_ky_dual_view_aligned.py \
  --stage downstream --tag scratch --include-scratch --ssl-tasks none \
  --seeds 42 43 44 > p0_scratch.log 2>&1 </dev/null &
echo "P0 scratch pid=$!"

for SEED in 42 43 44; do
  nohup python scripts/run_ky_dual_view_aligned.py \
    --stage all --tag seed${SEED} --seeds ${SEED} \
    --ssl-tasks masked_recon contrastive \
    --encoder-lrs 0 1e-5 1e-4 > p_seed${SEED}.log 2>&1 </dev/null &
  echo "P seed${SEED} pid=$!"
done
echo "LAUNCHED_ALL"
