#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source tools/train/architecture_candidate_profile.sh

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
SELECTION="${SELECTION:-metrics/upr_lite_candidate_selection.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
MAX_UTIL="${MAX_UTIL:-5}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
CLIMATOLOGY="${CLIMATOLOGY:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CLIMATOLOGY_CACHE="${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}"

LOG="$LOG_ROOT/temporal_router_12h_queue.log"
ROUTE="metrics/temporal_router_12h/route_2020.json"
ROUTER_NAME="quality_flow_tau_router_12h"
ROUTER_CHECKPOINT="$LOG_ROOT/temporal_router_quality_flow_12h_2020.ckpt"
TERMINAL="$LOG_ROOT/temporal_router_quality_flow_12h_2020.terminal"
FLOW_CHECKPOINT="$LOG_ROOT/exp_flow_pp3_12h_2017_19_held468_lr1e4_sp2_protocol_v3/last.ckpt"
QUALITY_METRICS="metrics/upr_lite_transfer_12h_2020/quality_transfer.json"
FLOW_METRICS="metrics/upr_lite_transfer_12h_2020/weatherbridge_ref.json"

mkdir -p "$LOG_ROOT" "$CLIMATOLOGY_CACHE" "$(dirname "$ROUTE")"
exec 9>"$LOG_ROOT/.temporal_router_12h_queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] 12h temporal-router queue already active" \
    | tee -a "$LOG"
  exit 0
fi

while [[ ! -s "$SELECTION" ]]; do
  echo "[$(date -Is)] wait 6h selection=$SELECTION" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
quality_candidate="$("$PYTHON_BIN" -c '
from pathlib import Path
import json, sys
from tools.eval.summarize_upr_lite_seeds import (
    choose_transfer_candidate,
    validate_frozen_selection_for_followup,
)
selection = json.loads(Path(sys.argv[1]).read_text())
validate_frozen_selection_for_followup(selection)
print(choose_transfer_candidate(selection, "quality"))
' "$SELECTION")"
quality_layout="$(architecture_candidate_layout "$quality_candidate")"
QUALITY_CHECKPOINT="$LOG_ROOT/exp_${quality_candidate}_24ch_12h_2017_19_held468_lr1e4_sp2_${quality_layout}_s202707/last.ckpt"

wait_field_artifact() {
  local label="$1"
  local artifact="$2"
  local checkpoint="$3"
  while ! "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
    "$artifact" \
    --checkpoint "$checkpoint" \
    --required-taus 1,2,3,4,5,6,7,8,9,10,11 \
    --acc-mode enabled \
    --require-physical-metrics \
    --require-temporal-metrics \
    --quiet >>"$LOG" 2>&1; do
    echo "[$(date -Is)] wait 12h field artifact=$label" | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
}

wait_field_artifact quality_2020 "$QUALITY_METRICS" "$QUALITY_CHECKPOINT"
wait_field_artifact flow_2020 "$FLOW_METRICS" "$FLOW_CHECKPOINT"

"$PYTHON_BIN" tools/eval/select_temporal_expert_route.py \
  --expert "quality:$QUALITY_METRICS:$QUALITY_CHECKPOINT" \
  --expert "flow:$FLOW_METRICS:$FLOW_CHECKPOINT" \
  --default-expert auto \
  --min-relative-gain 0.005 \
  --alpha 0.05 \
  --bootstrap-draws 5000 \
  --block-days 7 \
  --seed 202709 \
  --output "$ROUTE" >>"$LOG" 2>&1
"$PYTHON_BIN" tools/train/build_temporal_router_checkpoint.py \
  --route "$ROUTE" \
  --output "$ROUTER_CHECKPOINT" >>"$LOG" 2>&1
"$PYTHON_BIN" tools/train/temporal_router_checkpoint_status.py \
  "$ROUTER_CHECKPOINT" \
  --required-taus 1,2,3,4,5,6,7,8,9,10,11 \
  --quiet

read -r mixed_batch_size mixed_tau_values < <("$PYTHON_BIN" -c '
import json, sys
route = json.load(open(sys.argv[1]))["route_by_tau"]
first_hour = {}
for hour, expert in sorted(
    ((int(hour), expert) for hour, expert in route.items())
):
    first_hour.setdefault(expert, hour)
hours = list(first_hour.values())
print(len(hours), ",".join(str(hour / 12) for hour in hours))
' "$ROUTE")
all_tau_values="$("$PYTHON_BIN" -c '
print(",".join(str(hour / 12) for hour in range(1, 12)))
')"

gpu=""
lock_fd=""
while [[ -z "$gpu" ]]; do
  for candidate in $GPU_CANDIDATES; do
    exec {candidate_fd}>"$LOG_ROOT/.upr_lite_gpu${candidate}.lock"
    if ! flock -n "$candidate_fd"; then
      exec {candidate_fd}>&-
      continue
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
    if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      gpu="$candidate"
      lock_fd="$candidate_fd"
      break
    fi
    flock -u "$candidate_fd"
    exec {candidate_fd}>&-
  done
  if [[ -z "$gpu" ]]; then
    sleep "$SLEEP_SEC"
  fi
done

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
for year in 2020 2021; do
  field_root="metrics/temporal_router_12h_$year"
  field_result="$field_root/$ROUTER_NAME.json"
  if ! "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
    "$field_result" \
    --checkpoint "$ROUTER_CHECKPOINT" \
    --required-taus 1,2,3,4,5,6,7,8,9,10,11 \
    --acc-mode enabled \
    --require-physical-metrics \
    --require-temporal-metrics \
    --quiet >>"$LOG" 2>&1; then
    "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir /tmp/wb2_0p5_cache \
      --test-year "$year" \
      --climatology "$CLIMATOLOGY" \
      --climatology-cache-dir "$CLIMATOLOGY_CACHE" \
      --lazy-climatology \
      --stats-path data/json_stats_0p5.nc \
      --surface-stats-path data/surface_stats_0p5.json \
      --static-path data/static_features_0p5.pt \
      --models "$ROUTER_NAME:$ROUTER_CHECKPOINT:" \
      --out-dir "$field_root" \
      --paper-tag "temporal_router_12h_${year}" \
      --batch-size 4 \
      --num-workers 2 \
      --samples-per-date 2 \
      --full-year \
      --max-tau-hours 12 \
      --eval-hours 1,2,3,4,5,6,7,8,9,10,11 \
      --seen-tau 1,2,3,5,7,9,10,11 \
      --unseen-tau 4,6,8 \
      --keep-n-channels 24 \
      --proper-rmse \
      --save-window-metrics \
      --save-physical-metrics \
      --save-temporal-metrics \
      >>"$LOG" 2>&1
  fi
  "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
    "$field_result" \
    --checkpoint "$ROUTER_CHECKPOINT" \
    --required-taus 1,2,3,4,5,6,7,8,9,10,11 \
    --acc-mode enabled \
    --require-physical-metrics \
    --require-temporal-metrics \
    --quiet
  "$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year "$year" \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --ckpt "$ROUTER_CHECKPOINT" \
    --model-name "$ROUTER_NAME" \
    --model-kind capmatched \
    --out-dir "metrics/temporal_router_spectra_12h_$year" \
    --taus 4,6,8 \
    --channels all \
    --lmax 359 \
    --keep-n-channels 24 \
    --batch-size 2 \
    --samples-per-date 2 \
    --eval-days-per-month 2 \
    --hf-ell-min 180 \
    --max-tau-hours 12 \
    --skip-existing \
    >>"$LOG" 2>&1
done

"$PYTHON_BIN" -u tools/eval/benchmark_capmatched_inference.py \
  --model "$ROUTER_NAME:$ROUTER_CHECKPOINT" \
  --static-path data/static_features_0p5.pt \
  --batch-size 1 \
  --tau-values "$all_tau_values" \
  --height 360 \
  --width 720 \
  --warmup 11 \
  --iterations 22 \
  --out-json metrics/temporal_router_inference_cost_12h_batch1_all_tau.json \
  >>"$LOG" 2>&1
"$PYTHON_BIN" -u tools/eval/benchmark_capmatched_inference.py \
  --model "$ROUTER_NAME:$ROUTER_CHECKPOINT" \
  --static-path data/static_features_0p5.pt \
  --batch-size "$mixed_batch_size" \
  --tau-values "$mixed_tau_values" \
  --height 360 \
  --width 720 \
  --warmup 3 \
  --iterations 10 \
  --out-json metrics/temporal_router_inference_cost_12h_mixed.json \
  >>"$LOG" 2>&1

flock -u "$lock_fd"
exec {lock_fd}>&-

QUALITY_OOD="metrics/upr_lite_transfer_12h_2021/quality_transfer.json"
FLOW_OOD="metrics/upr_lite_transfer_12h_2021/weatherbridge_ref.json"
wait_field_artifact quality_2021 "$QUALITY_OOD" "$QUALITY_CHECKPOINT"
wait_field_artifact flow_2021 "$FLOW_OOD" "$FLOW_CHECKPOINT"

"$PYTHON_BIN" tools/eval/assess_temporal_expert_router.py \
  --route "$ROUTE" \
  --router-checkpoint "$ROUTER_CHECKPOINT" \
  --router-field-2021 \
  "metrics/temporal_router_12h_2021/$ROUTER_NAME.json" \
  --expert-field-2021 "quality=$QUALITY_OOD" \
  --expert-field-2021 "flow=$FLOW_OOD" \
  --router-spectra-2020 metrics/temporal_router_spectra_12h_2020 \
  --router-name "$ROUTER_NAME" \
  --expert-spectra-2020 metrics/upr_lite_transfer_spectra_12h_2020 \
  --expert-spectrum-prefix quality=quality_transfer \
  --expert-spectrum-prefix flow=weatherbridge_ref \
  --spectral-taus 4,6,8 \
  --min-relative-gain 0.005 \
  --alpha 0.05 \
  --draws 5000 \
  --seed 202710 \
  --block-days 7 \
  --extreme-quantile 0.95 \
  --output metrics/temporal_router_generalization_12h.json \
  >>"$LOG" 2>&1

touch "$TERMINAL"
echo "[$(date -Is)] 12h temporal-router evaluation complete" \
  | tee -a "$LOG"
