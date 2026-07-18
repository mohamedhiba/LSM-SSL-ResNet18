#!/bin/bash
# Pull KY dual-view fold metrics + logs from the pod to the Mac (small files,
# one base64-over-ssh shot each — survives the flaky uplink).
# Usage: POD_HOST=1.2.3.4 POD_PORT=12345 bash ky_dual_view_pkg/pull_ky_dv_results.sh
set -u
POD_HOST=${POD_HOST:?set POD_HOST}
POD_PORT=${POD_PORT:?set POD_PORT}
KEY=~/.ssh/runpod_ed25519
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SSH="ssh -i $KEY -p $POD_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@$POD_HOST"

pull() {  # pull <pod-path> <local-path>
  local remote=$1 local_path=$2
  mkdir -p "$(dirname "$local_path")"
  if $SSH "test -f $remote && base64 < $remote" > "$local_path.b64" 2>/dev/null && [ -s "$local_path.b64" ]; then
    base64 -d < "$local_path.b64" > "$local_path" && rm -f "$local_path.b64"
    echo "pulled $(basename "$local_path") ($(wc -c < "$local_path" | tr -d ' ') bytes)"
  else
    rm -f "$local_path.b64"
    echo "missing/failed: $remote"
  fi
}

for TAG in scratch seed42 seed43 seed44 seed45; do
  pull "/workspace/kydv/outputs/KY_dual_view_aligned/flat_cv/$TAG/comparison/final_10m_flat_cv_lr_sweep_fold_metrics.csv" \
       "$REPO/outputs/KY_dual_view_aligned/flat_cv/$TAG/comparison/final_10m_flat_cv_lr_sweep_fold_metrics.csv"
done
for LOG in p0_scratch p_seed42 p_seed43 p_seed44; do
  $SSH "tail -5 /workspace/kydv/$LOG.log 2>/dev/null" > "$REPO/outputs/KY_dual_view_aligned/${LOG}_tail.txt" 2>/dev/null \
    && echo "--- $LOG tail:" && cat "$REPO/outputs/KY_dual_view_aligned/${LOG}_tail.txt"
done
