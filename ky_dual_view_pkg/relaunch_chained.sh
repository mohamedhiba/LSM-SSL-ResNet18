#!/bin/bash
# Restructured per-seed launch: masked_recon first END-TO-END (SSL then its
# downstream grid), contrastive only afterwards — so the meeting-critical
# masked_recon column lands hours earlier than the original both-SSL-first
# ordering. Strict LR=0 freeze per the Slack spec. Resume skips anything
# already completed.
# Usage: bash relaunch_chained.sh [unlabeled_index_csv]
CORPUS=${1:-data/kentucky_dual_view/unlabeled_dual_view_padded_index_ky10m_n50000.csv}
cd /workspace/kydv
export KY_RASTER_DIR=/workspace/kydv/processed/rasters_cleaned_10m
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

pkill -f strict_switch.sh 2>/dev/null
pkill -f -- "--tag seed" 2>/dev/null
sleep 5

for SEED in 42 43 44; do
  nohup bash -c "
    python scripts/run_ky_dual_view_aligned.py \
      --stage all --tag seed${SEED} --seeds ${SEED} \
      --ssl-tasks masked_recon --encoder-lrs 0 1e-4 1e-5 \
      --unlabeled-index ${CORPUS} --strict-frozen-encoder &&
    python scripts/run_ky_dual_view_aligned.py \
      --stage all --tag seed${SEED} --seeds ${SEED} \
      --ssl-tasks contrastive --encoder-lrs 0 1e-4 1e-5 \
      --unlabeled-index ${CORPUS} --strict-frozen-encoder
  " > p_seed${SEED}.log 2>&1 </dev/null &
  echo "chained seed${SEED} pid=$!"
done
echo "CHAINED_LAUNCHED corpus=${CORPUS}"
