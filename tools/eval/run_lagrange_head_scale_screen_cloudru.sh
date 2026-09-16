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

SOURCE_CHECKPOINT="$LOG_ROOT/exp_flow_compact_lagrange_l_headft_lr1e3_9m_6h_s202707_v1/last.ckpt"
SOURCE_SHA256="071db44cc69c97383c0f2ecd08ad88b46dc5628d706e75e6fc7d6a95780c21f2"
FLOW_CHECKPOINT="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202707_protocol_v2/epoch=3-step=6568.ckpt"
SCALED_ROOT="$LOG_ROOT/lagrange_head_scale_checkpoints"
LOW_OUT="metrics/lagrange_head_scale_low_8dpm_6h_2020"
HIGH_OUT="metrics/lagrange_head_scale_high_8dpm_6h_2020"
FULL_OUT="metrics/lagrange_head_scale_winner_full_6h_2020"
OOD_OUT="metrics/lagrange_head_scale_winner_full_6h_2021"
REPORT_ROOT="$METRICS_ROOT/lagrange_head_scale_6h"
ECONOMY_REPORT="$REPORT_ROOT/economy_selection.json"
FULL_REPORT="$REPORT_ROOT/full_year_2020_selection.json"
OOD_REPORT="$REPORT_ROOT/full_year_2021_confirmation.json"
QUEUE_LOG="$LOG_ROOT/lagrange_head_scale_eval.queue.log"
TERMINAL_MARKER="$LOG_ROOT/lagrange_head_scale_eval.terminal"
COMPLETE_MARKER="$LOG_ROOT/lagrange_head_scale_eval.complete"

declare -A CHECKPOINTS=(
  [lagrange_scale_2x]="$SCALED_ROOT/factor2.ckpt"
  [lagrange_scale_4x]="$SCALED_ROOT/factor4.ckpt"
  [lagrange_scale_6x]="$SCALED_ROOT/factor6.ckpt"
  [lagrange_scale_8x]="$SCALED_ROOT/factor8.ckpt"
)

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

generate_scaled_checkpoints() {
  mkdir -p "$SCALED_ROOT"
  printf '%s  %s\n' "$SOURCE_SHA256" "$SOURCE_CHECKPOINT" | sha256sum -c -
  local factor
  for factor in 2 4 6 8; do
    "$PYTHON_BIN" tools/train/scale_lagrange_head_checkpoint.py \
      --source "$SOURCE_CHECKPOINT" \
      --output "$SCALED_ROOT/factor${factor}.ckpt" \
      --factor "$factor"
  done
}

run_economy_eval() {
  local gpu="$1"
  local out_dir="$2"
  local models="$3"
  exec 8>"$LOG_ROOT/.lagrange_head_scale_gpu${gpu}.lock"
  flock 8
  wait_for_gpu "$gpu"
  export CUDA_VISIBLE_DEVICES="$gpu"
  mkdir -p "$out_dir"
  "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "$models" \
    --out-dir "$out_dir" \
    --paper-tag lagrange_head_scale_8dpm_6h_2020 \
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

run_full_eval() {
  local year="$1"
  local candidate="$2"
  local checkpoint="$3"
  local out_dir="$4"
  exec 8>"$LOG_ROOT/.lagrange_head_scale_gpu0.lock"
  flock 8
  wait_for_gpu 0
  export CUDA_VISIBLE_DEVICES=0
  mkdir -p "$out_dir"
  "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year "$year" \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "$candidate:$checkpoint:,flow_4ep_ref:$FLOW_CHECKPOINT:" \
    --out-dir "$out_dir" \
    --paper-tag "lagrange_head_scale_full_6h_${year}" \
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
}

select_strict() {
  local candidate="$1"
  local candidate_json="$2"
  local reference_json="$3"
  local output="$4"
  "$PYTHON_BIN" tools/eval/select_base_field_candidate.py \
    --candidate "$candidate=$candidate_json" \
    --reference "$reference_json" \
    --base-mean-limit 0 \
    --seen-mean-limit 0 \
    --held-mean-limit 0 \
    --per-tau-limit 0 \
    --surface-mean-limit 0 \
    --output "$output" \
    2>&1 | tee -a "$QUEUE_LOG"
}

selected_from() {
  "$PYTHON_BIN" -c '
import json, sys
print(json.load(open(sys.argv[1]))["selected"] or "")
' "$1"
}

mkdir -p "$LOG_ROOT" "$REPORT_ROOT"
exec 9>"$LOG_ROOT/lagrange_head_scale_eval.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] Lagrange head scale screen already active" \
    | tee -a "$QUEUE_LOG"
  exit 0
fi
rm -f "$TERMINAL_MARKER" "$COMPLETE_MARKER"
trap 'touch "$TERMINAL_MARKER"' ERR

for checkpoint in "$SOURCE_CHECKPOINT" "$FLOW_CHECKPOINT"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "missing evaluation checkpoint: $checkpoint" >&2
    exit 2
  fi
done
generate_scaled_checkpoints 2>&1 | tee -a "$QUEUE_LOG"

run_economy_eval 0 "$LOW_OUT" \
  "lagrange_scale_2x:${CHECKPOINTS[lagrange_scale_2x]}:,lagrange_scale_4x:${CHECKPOINTS[lagrange_scale_4x]}:,flow_4ep_ref:$FLOW_CHECKPOINT:" &
low_pid=$!
run_economy_eval 1 "$HIGH_OUT" \
  "lagrange_scale_6x:${CHECKPOINTS[lagrange_scale_6x]}:,lagrange_scale_8x:${CHECKPOINTS[lagrange_scale_8x]}:" &
high_pid=$!
status=0
wait "$low_pid" || status=1
wait "$high_pid" || status=1
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$PYTHON_BIN" tools/eval/select_base_field_candidate.py \
  --candidate "lagrange_scale_2x=$LOW_OUT/lagrange_scale_2x.json" \
  --candidate "lagrange_scale_4x=$LOW_OUT/lagrange_scale_4x.json" \
  --candidate "lagrange_scale_6x=$HIGH_OUT/lagrange_scale_6x.json" \
  --candidate "lagrange_scale_8x=$HIGH_OUT/lagrange_scale_8x.json" \
  --reference "$LOW_OUT/flow_4ep_ref.json" \
  --base-mean-limit 0 \
  --seen-mean-limit 0 \
  --held-mean-limit 0 \
  --per-tau-limit 0 \
  --surface-mean-limit 0 \
  --output "$ECONOMY_REPORT" \
  2>&1 | tee -a "$QUEUE_LOG"

selected="$(selected_from "$ECONOMY_REPORT")"
if [[ -z "$selected" ]]; then
  touch "$COMPLETE_MARKER"
  trap - ERR
  echo "[$(date -Is)] no strict scale-screen winner" | tee -a "$QUEUE_LOG"
  exit 0
fi
winner_checkpoint="${CHECKPOINTS[$selected]}"

run_full_eval 2020 "$selected" "$winner_checkpoint" "$FULL_OUT"
select_strict \
  "$selected" \
  "$FULL_OUT/$selected.json" \
  "$FULL_OUT/flow_4ep_ref.json" \
  "$FULL_REPORT"
if [[ -z "$(selected_from "$FULL_REPORT")" ]]; then
  touch "$COMPLETE_MARKER"
  trap - ERR
  echo "[$(date -Is)] scale winner rejected on full 2020" | tee -a "$QUEUE_LOG"
  exit 0
fi

run_full_eval 2021 "$selected" "$winner_checkpoint" "$OOD_OUT"
select_strict \
  "$selected" \
  "$OOD_OUT/$selected.json" \
  "$OOD_OUT/flow_4ep_ref.json" \
  "$OOD_REPORT"

touch "$COMPLETE_MARKER"
trap - ERR
echo "[$(date -Is)] Lagrange head scale evaluation complete" \
  | tee -a "$QUEUE_LOG"
