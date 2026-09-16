#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-1}"
Q4_MIN_FREE_MIB="${Q4_MIN_FREE_MIB:-40000}"
FOLLOWUP_JSON="${FOLLOWUP_JSON:-metrics/upr_lite_screen/q4_followup.json}"

arch="upr_lite_implicit_global_q4"
exp_name="exp_${arch}_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707"
checkpoint="$LOG_ROOT/$exp_name/last.ckpt"
column_checkpoint="$LOG_ROOT/exp_upr_lite_column_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707/last.ckpt"
queue_log="$LOG_ROOT/$exp_name.followup.queue.log"

mkdir -p "$LOG_ROOT" "$(dirname "$FOLLOWUP_JSON")"
exec 9>"$LOG_ROOT/${exp_name}.followup.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] Q4 follow-up queue already active" | tee -a "$queue_log"
  exit 0
fi

# Do not steal GPU1 from the fixed Top-3 screen continuation.
while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$column_checkpoint" \
  --min-epochs 8 \
  --expected-arch upr_lite_column \
  --expected-total-steps 13136 \
  --min-global-step 13136 \
  --expected-delta-t 6 \
  --quiet; do
  echo "[$(date -Is)] wait fixed Top-3 column arm" | tee -a "$queue_log"
  sleep "$SLEEP_SEC"
done

if ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$checkpoint" \
  --min-epochs 2 \
  --expected-arch "$arch" \
  --expected-total-steps 13136 \
  --expected-delta-t 6 \
  --quiet; then
  echo "[$(date -Is)] launch two-epoch Q4 efficiency follow-up" \
    | tee -a "$queue_log"
  ARCH="$arch" \
  EXP_NAME="$exp_name" \
  GPU_CANDIDATES="$GPU_CANDIDATES" \
  MIN_FREE_MIB="$Q4_MIN_FREE_MIB" \
  SLEEP_SEC="$SLEEP_SEC" \
    bash tools/train/run_upr_lite_variant_6h_cloudru.sh \
      >>"$queue_log" 2>&1 &
  screen_queue_pid=$!
  "$PYTHON_BIN" tools/train/stop_after_screen_checkpoint.py \
    --queue-pid "$screen_queue_pid" \
    --experiment "$exp_name" \
    --checkpoint "$checkpoint" \
    --min-epochs 2 \
    --poll-seconds 30 \
    >>"$queue_log" 2>&1
  wait "$screen_queue_pid" || true
fi

"$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$checkpoint" \
  --min-epochs 2 \
  --expected-arch "$arch" \
  --expected-total-steps 13136 \
  --expected-delta-t 6 \
  --quiet
"$PYTHON_BIN" tools/eval/assess_upr_q4_followup.py \
  --log-root "$LOG_ROOT" \
  --output "$FOLLOWUP_JSON" \
  --maximum-relative-rmse 1.02 \
  >>"$queue_log" 2>&1
promoted="$("$PYTHON_BIN" -c '
import json, sys
print(int(json.load(open(sys.argv[1]))["promoted"]))
' "$FOLLOWUP_JSON")"
if [[ "$promoted" -ne 1 ]]; then
  echo "[$(date -Is)] Q4 rejected by two-epoch quality gate" \
    | tee -a "$queue_log"
  exit 0
fi

echo "[$(date -Is)] Q4 promoted; resume to eight epochs" \
  | tee -a "$queue_log"
ARCH="$arch" \
EXP_NAME="$exp_name" \
GPU_CANDIDATES="$GPU_CANDIDATES" \
MIN_FREE_MIB="$Q4_MIN_FREE_MIB" \
SLEEP_SEC="$SLEEP_SEC" \
  bash tools/train/run_upr_lite_variant_6h_cloudru.sh \
    >>"$queue_log" 2>&1
"$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$checkpoint" \
  --min-epochs 8 \
  --expected-arch "$arch" \
  --expected-total-steps 13136 \
  --min-global-step 13136 \
  --expected-delta-t 6 \
  --quiet
echo "[$(date -Is)] Q4 follow-up complete checkpoint=$checkpoint" \
  | tee -a "$queue_log"
