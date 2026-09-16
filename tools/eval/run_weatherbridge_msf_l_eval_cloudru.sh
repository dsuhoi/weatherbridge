#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
METRICS_ROOT="${METRICS_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime/metrics}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
GPU="${GPU:-1}"
SLEEP_SEC="${SLEEP_SEC:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
MAX_UTIL="${MAX_UTIL:-5}"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

TRAIN_COMPLETE="$LOG_ROOT/exp_weatherbridge_msf_l_9m_6h_s202707_v2_bs8.complete"
MSF_ROOT="$LOG_ROOT/exp_weatherbridge_msf_l_9m_6h_s202707_v2_bs8"
FLOW_CHECKPOINT="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202707_protocol_v2/epoch=3-step=6568.ckpt"
ECONOMY_OUT="metrics/weatherbridge_msf_l_8dpm_6h_2020"
FULL_OUT="metrics/weatherbridge_msf_l_full_6h_2020"
REPORT_ROOT="$METRICS_ROOT/weatherbridge_msf_l_6h"
ECONOMY_REPORT="$REPORT_ROOT/economy_strict_120cell.json"
FULL_REPORT="$REPORT_ROOT/full_year_2020_strict_120cell.json"
QUEUE_LOG="$LOG_ROOT/weatherbridge_msf_l_eval.queue.log"
COMPLETE_MARKER="$LOG_ROOT/weatherbridge_msf_l_eval.complete"
TERMINAL_MARKER="$LOG_ROOT/weatherbridge_msf_l_eval.terminal"

declare -A CHECKPOINTS=(
  [weatherbridge_msf_l_2ep]="$MSF_ROOT/epoch=1-step=3284.ckpt"
  [weatherbridge_msf_l_4ep]="$MSF_ROOT/epoch=3-step=6568.ckpt"
)

wait_for_gpu() {
  while true; do
    local free_mib
    local util
    free_mib="$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
    if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      return
    fi
    sleep "$SLEEP_SEC"
  done
}

run_eval() {
  local out_dir="$1"
  local models="$2"
  local sampling_args="$3"
  local acc_args="$4"
  mkdir -p "$out_dir"
  export CUDA_VISIBLE_DEVICES="$GPU"
  # shellcheck disable=SC2086
  "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "$models" \
    --out-dir "$out_dir" \
    --paper-tag weatherbridge_msf_l_6h_2020 \
    --batch-size 2 \
    --num-workers 2 \
    --samples-per-date 4 \
    --max-tau-hours 6 \
    --eval-hours 1,2,3,4,5 \
    --seen-tau 1,3,5 \
    --unseen-tau 2,4 \
    --keep-n-channels 24 \
    --proper-rmse \
    $sampling_args \
    $acc_args \
    2>&1 | tee -a "$QUEUE_LOG"
}

select_strict() {
  local output="$1"
  shift
  "$PYTHON_BIN" tools/eval/select_strict_flow_dominance.py \
    "$@" \
    --reference "$FLOW_JSON" \
    --cell-limit 0 \
    --include-q \
    --output "$output" \
    2>&1 | tee -a "$QUEUE_LOG"
}

selected_from() {
  "$PYTHON_BIN" -c '
import json
import sys
print(json.load(open(sys.argv[1]))["selected"] or "")
' "$1"
}

mkdir -p "$LOG_ROOT" "$REPORT_ROOT"
exec 9>"$LOG_ROOT/weatherbridge_msf_l_eval.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] WeatherBridge-MSF-L evaluation already active" \
    | tee -a "$QUEUE_LOG"
  exit 0
fi
rm -f "$COMPLETE_MARKER" "$TERMINAL_MARKER"
trap 'touch "$TERMINAL_MARKER"' ERR

while [[ ! -e "$TRAIN_COMPLETE" ]]; do
  echo "[$(date -Is)] wait WeatherBridge-MSF-L training" | tee -a "$QUEUE_LOG"
  sleep "$SLEEP_SEC"
done
for checkpoint in "${CHECKPOINTS[@]}" "$FLOW_CHECKPOINT"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "missing evaluation checkpoint: $checkpoint" >&2
    exit 2
  fi
done

wait_for_gpu
run_eval \
  "$ECONOMY_OUT" \
  "weatherbridge_msf_l_2ep:${CHECKPOINTS[weatherbridge_msf_l_2ep]}:,weatherbridge_msf_l_4ep:${CHECKPOINTS[weatherbridge_msf_l_4ep]}:,flow_4ep_ref:$FLOW_CHECKPOINT:" \
  "--eval-days-per-month 8" \
  "--no-acc"

FLOW_JSON="$ECONOMY_OUT/flow_4ep_ref.json"
select_strict \
  "$ECONOMY_REPORT" \
  --candidate "weatherbridge_msf_l_2ep=$ECONOMY_OUT/weatherbridge_msf_l_2ep.json" \
  --candidate "weatherbridge_msf_l_4ep=$ECONOMY_OUT/weatherbridge_msf_l_4ep.json"

selected="$(selected_from "$ECONOMY_REPORT")"
if [[ -z "$selected" ]]; then
  touch "$COMPLETE_MARKER"
  trap - ERR
  echo "[$(date -Is)] no strict 120-cell MSF-L winner" | tee -a "$QUEUE_LOG"
  exit 0
fi

winner_checkpoint="${CHECKPOINTS[$selected]}"
wait_for_gpu
run_eval \
  "$FULL_OUT" \
  "$selected:$winner_checkpoint:,flow_4ep_ref:$FLOW_CHECKPOINT:" \
  "--full-year --save-window-metrics" \
  "--lazy-climatology"

FLOW_JSON="$FULL_OUT/flow_4ep_ref.json"
select_strict \
  "$FULL_REPORT" \
  --candidate "$selected=$FULL_OUT/$selected.json" \
  --reference "WeatherBridge-PP3-8ep=metrics/journal_unified/6h_2020/weatherbridge_pp3_14m_6yr_ep8.json" \
  --reference "WeatherDCAE=metrics/journal_unified/6h_2020/weatherdcae_14m_3yr_ep8.json" \
  --reference "PixelAttn-VFI=metrics/journal_unified/6h_2020/atm_vfi_24ch_3yr_ep13.json" \
  --reference "FuXi=metrics/journal_unified/6h_2020/fuxi_24ch_6yr_ep8.json" \
  --reference "ModAFNO=metrics/journal_unified/6h_2020/modafno_24ch_6yr_ep8.json" \
  --reference "S-DYffusion=metrics/journal_unified/6h_2020/sdyff_24ch_6yr_ep8.json"

touch "$COMPLETE_MARKER"
trap - ERR
echo "[$(date -Is)] WeatherBridge-MSF-L evaluation complete" \
  | tee -a "$QUEUE_LOG"
