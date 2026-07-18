#!/bin/bash
# Per-seed strict-freeze switch (Slack spec: LR=0 freezes weights AND BN stats).
# The running seed processes were launched pre-directive with the code-default
# non-strict config. For each seed: wait until its SSL stage completes (the
# "[downstream]" line appears in its log), kill that process before any LR=0
# job can finish, wipe the (at most seconds old) downstream namespace, and
# relaunch downstream-only with --strict-frozen-encoder.
cd /workspace/kydv
export KY_RASTER_DIR=/workspace/kydv/processed/rasters_cleaned_10m
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

for SEED in 42 43 44; do
  (
    while ! grep -q "\[downstream\]" p_seed${SEED}.log 2>/dev/null; do sleep 20; done
    pkill -f -- "--tag seed${SEED}" && echo "killed seed${SEED} pre-downstream" || echo "seed${SEED} kill: no match"
    sleep 5
    rm -rf outputs/KY_dual_view_aligned/flat_cv/seed${SEED} checkpoints/KY_dual_view_aligned/flat_cv/seed${SEED}
    nohup python scripts/run_ky_dual_view_aligned.py \
      --stage downstream --tag seed${SEED} --seeds ${SEED} \
      --ssl-tasks masked_recon contrastive \
      --encoder-lrs 0 1e-5 1e-4 --strict-frozen-encoder \
      > p_seed${SEED}.log 2>&1 </dev/null &
    echo "relaunched seed${SEED} strict pid=$!"
  ) >> strict_switch.log 2>&1 &
done
echo "STRICT_SWITCH_ARMED"
