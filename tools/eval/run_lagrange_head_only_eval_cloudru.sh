#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
METRICS_ROOT="${METRICS_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime/metrics}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
SLEEP_SEC="${SLEEP_SEC:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
MAX_UTIL="${MAX_UTIL:-5}"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

TRAIN_COMPLETE="$LOG_ROOT/lagrange_head_only.complete"
BASE_CHECKPOINT="$LOG_ROOT/exp_flow_compact_lagrange_l_headft_lr3e4_9m_6h_s202707_v1/last.ckpt"
EDGE_CHECKPOINT="$LOG_ROOT/exp_flow_compact_lagrange_l_headft_lr1e3_9m_6h_s202707_v1/last.ckpt"
FLOW_CHECKPOINT="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202707_protocol_v2/epoch=3-step=6568.ckpt"
BASE_OUT="metrics/lagrange_head_lr3e4_8dpm_6h_2020"
EDGE_OUT="metrics/lagrange_head_lr1e3_8dpm_6h_2020"
FULL_OUT="metrics/lagrange_head_winner_full_6h_2020"
REPORT="$METRICS_ROOT/lagrange_head_only_6h/economy_selection.json"
FULL_REPORT="$METRICS_ROOT/lagrange_head_only_6h/full_year_selection.json"
QUEUE_LOG="$LOG_ROOT/lagrange_head_only_eval.queue.log"
TERMINAL_MARKER="$LOG_ROOT/lagrange_head_only_eval.terminal"
COMPLETE_MARKER="$LOG_ROOT/lagrange_head_only_eval.complete"

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local free_mib
    local util
    free_mib="$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      return
    fi
    sleep "$SLEEP_SEC"
  done
}

run_base_eval() {
  exec 8>"$LOG_ROOT/.lagrange_head_gpu0.lock"
  flock 8
  wait_for_gpu 0
  export CUDA_VISIBLE_DEVICES=0
  mkdir -p "$BASE_OUT"
  "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "lagrange_head_lr3e4:$BASE_CHECKPOINT:,flow_4ep_ref:$FLOW_CHECKPOINT:" \
    --out-dir "$BASE_OUT" \
    --paper-tag lagrange_head_lr3e4_8dpm_6h_2020 \
    --batch-size 2 \
    --num-workers 2 \
    --samples-per-date 4 \
    --eval-days-per-month 8 \
    --max-tau-hours 6 \
    --eval-hours 1,2,3,4,5 \
    --seen-tau 1,3,5 \
    --unseen-tau 2,4 \
    --keep-n-channels 24 \
    --proper-rmse \
    --no-acc \
    2>&1 | tee -a "$QUEUE_LOG"
}

run_edge_eval() {
  exec 8>"$LOG_ROOT/.lagrange_head_gpu1.lock"
  flock 8
  wait_for_gpu 1
  export CUDA_VISIBLE_DEVICES=1
  mkdir -p "$EDGE_OUT"
  "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "lagrange_head_lr1e3:$EDGE_CHECKPOINT:" \
    --out-dir "$EDGE_OUT" \
    --paper-tag lagrange_head_lr1e3_8dpm_6h_2020 \
    --batch-size 2 \
    --num-workers 2 \
    --samples-per-date 4 \
    --eval-days-per-month 8 \
    --max-tau-hours 6 \
    --eval-hours 1,2,3,4,5 \
    --seen-tau 1,3,5 \
    --unseen-tau 2,4 \
    --keep-n-channels 24 \
    --proper-rmse \
    --no-acc \
    2>&1 | tee -a "$QUEUE_LOG"
}

mkdir -p "$LOG_ROOT" "$(dirname "$REPORT")"
exec 9>"$LOG_ROOT/lagrange_head_only_eval.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] Lagrange head-only evaluation already active" \
    | tee -a "$QUEUE_LOG"
  exit 0
fi
rm -f "$TERMINAL_MARKER" "$COMPLETE_MARKER"
trap 'touch "$TERMINAL_MARKER"' ERR

while [[ ! -e "$TRAIN_COMPLETE" ]]; do
  echo "[$(date -Is)] wait Lagrange head-only fine-tunes" \
    | tee -a "$QUEUE_LOG"
  sleep "$SLEEP_SEC"
done
for checkpoint in "$BASE_CHECKPOINT" "$EDGE_CHECKPOINT" "$FLOW_CHECKPOINT"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "missing evaluation checkpoint: $checkpoint" >&2
    exit 2
  fi
done

run_base_eval &
base_pid=$!
run_edge_eval &
edge_pid=$!
status=0
wait "$base_pid" || status=1
wait "$edge_pid" || status=1
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$PYTHON_BIN" tools/eval/select_base_field_candidate.py \
  --candidate "lagrange_head_lr3e4=$BASE_OUT/lagrange_head_lr3e4.json" \
  --candidate "lagrange_head_lr1e3=$EDGE_OUT/lagrange_head_lr1e3.json" \
  --reference "$BASE_OUT/flow_4ep_ref.json" \
  --base-mean-limit 0 \
  --seen-mean-limit 0 \
  --held-mean-limit 0 \
  --per-tau-limit 0 \
  --surface-mean-limit 0 \
  --output "$REPORT" \
  2>&1 | tee -a "$QUEUE_LOG"

selected="$("$PYTHON_BIN" -c '
import json, sys
print(json.load(open(sys.argv[1]))["selected"] or "")
' "$REPORT")"
if [[ -z "$selected" ]]; then
  touch "$COMPLETE_MARKER"
  trap - ERR
  echo "[$(date -Is)] no strict Lagrange head-only winner" \
    | tee -a "$QUEUE_LOG"
  exit 0
fi

case "$selected" in
  lagrange_head_lr3e4) winner_checkpoint="$BASE_CHECKPOINT" ;;
  lagrange_head_lr1e3) winner_checkpoint="$EDGE_CHECKPOINT" ;;
  *) echo "unknown selected arm: $selected" >&2; exit 2 ;;
esac

exec 8>"$LOG_ROOT/.lagrange_head_gpu0.lock"
flock 8
wait_for_gpu 0
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$FULL_OUT"
"$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
  --memmap-dir /tmp/wb2_0p5_cache \
  --test-year 2020 \
  --stats-path data/json_stats_0p5.nc \
  --surface-stats-path data/surface_stats_0p5.json \
  --static-path data/static_features_0p5.pt \
  --models "lagrange_head_winner:$winner_checkpoint:,flow_4ep_ref:$FLOW_CHECKPOINT:" \
  --out-dir "$FULL_OUT" \
  --paper-tag lagrange_head_winner_full_6h_2020 \
  --batch-size 2 \
  --num-workers 2 \
  --samples-per-date 4 \
  --full-year \
  --max-tau-hours 6 \
  --eval-hours 1,2,3,4,5 \
  --seen-tau 1,3,5 \
  --unseen-tau 2,4 \
  --keep-n-channels 24 \
  --proper-rmse \
  --no-acc \
  2>&1 | tee -a "$QUEUE_LOG"

"$PYTHON_BIN" tools/eval/select_base_field_candidate.py \
  --candidate "lagrange_head_winner=$FULL_OUT/lagrange_head_winner.json" \
  --reference "$FULL_OUT/flow_4ep_ref.json" \
  --base-mean-limit 0 \
  --seen-mean-limit 0 \
  --held-mean-limit 0 \
  --per-tau-limit 0 \
  --surface-mean-limit 0 \
  --output "$FULL_REPORT" \
  2>&1 | tee -a "$QUEUE_LOG"

touch "$COMPLETE_MARKER"
trap - ERR
echo "[$(date -Is)] Lagrange head-only evaluation complete" \
  | tee -a "$QUEUE_LOG"
