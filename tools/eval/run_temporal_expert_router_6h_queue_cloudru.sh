#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
MIN_FREE_MIB="${MIN_FREE_MIB:-48000}"
MAX_UTIL="${MAX_UTIL:-5}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
CLIMATOLOGY="${CLIMATOLOGY:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CLIMATOLOGY_CACHE="${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}"
FROZEN_TRAINER_SHA256="${FROZEN_TRAINER_SHA256:-e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a}"

UPR_NAME="upr_implicit_global_14m"
FLOW_NAME="weatherbridge_ref"
ROUTER_NAME="upr_flow_tau_router"
UPR_CHECKPOINT="$LOG_ROOT/exp_upr_implicit_global_14m_hf_135_14m_6h_s202707_protocol_v2/last.ckpt"
FLOW_CHECKPOINT="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202707_protocol_v2/last.ckpt"
UPR_METRICS="metrics/upr_lite_screen_6h_2020/$UPR_NAME.json"
FLOW_METRICS="metrics/upr_lite_screen_6h_2020/$FLOW_NAME.json"
ROUTE="metrics/temporal_router_6h/route_2020.json"
ROUTER_CHECKPOINT="$LOG_ROOT/temporal_router_upr_flow_6h_2020.ckpt"
LOG="$LOG_ROOT/temporal_router_6h_queue.log"
TERMINAL="$LOG_ROOT/temporal_router_upr_flow_6h_2020.terminal"

mkdir -p "$LOG_ROOT" "$CLIMATOLOGY_CACHE" "$(dirname "$ROUTE")"
exec 9>"$LOG_ROOT/.temporal_router_6h_queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] temporal-router queue already active" | tee -a "$LOG"
  exit 0
fi

matched_protocol_args=(
  --require-training-protocol
  --expected-train-years 2014,2015,2016,2017,2018,2019
  --expected-val-years 2020
  --expected-train-taus 1,3,5
  --expected-eval-taus 1,2,3,4,5
  --expected-seed 202707
  --expected-effective-batch-size 16
  --expected-batch-size-per-device 4
  --expected-accumulate-grad-batches 4
  --expected-samples-per-date-train 4
  --expected-samples-per-date-val 2
  --expected-highpass-boundary periodic_lon_replicate_lat
  --expected-precision bf16-mixed
  --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256"
)

while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$UPR_CHECKPOINT" \
  --min-epochs 8 \
  --expected-arch upr_implicit_global_14m \
  --expected-total-steps 13136 \
  --min-global-step 13136 \
  --expected-delta-t 6 \
  "${matched_protocol_args[@]}" \
  --expected-lambda-hf 0.05 \
  --quiet; do
  sleep "$SLEEP_SEC"
done
while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$FLOW_CHECKPOINT" \
  --min-epochs 8 \
  --expected-arch flow_pp3 \
  --expected-total-steps 13136 \
  --min-global-step 13136 \
  --expected-delta-t 6 \
  "${matched_protocol_args[@]}" \
  --expected-lambda-hf 0 \
  --quiet; do
  echo "[$(date -Is)] wait strict Flow checkpoint" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done

while ! "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
  "$UPR_METRICS" \
  --checkpoint "$UPR_CHECKPOINT" \
  --required-taus 1,2,3,4,5 \
  --acc-mode enabled \
  --require-physical-metrics \
  --require-temporal-metrics \
  --quiet >>"$LOG" 2>&1; do
  echo "[$(date -Is)] wait full-year 2020 UPR metrics" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
while ! "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
  "$FLOW_METRICS" \
  --checkpoint "$FLOW_CHECKPOINT" \
  --required-taus 1,2,3,4,5 \
  --acc-mode enabled \
  --require-physical-metrics \
  --require-temporal-metrics \
  --quiet >>"$LOG" 2>&1; do
  echo "[$(date -Is)] wait full-year 2020 Flow metrics" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done

"$PYTHON_BIN" tools/eval/select_temporal_expert_route.py \
  --expert "upr:$UPR_METRICS:$UPR_CHECKPOINT" \
  --expert "flow:$FLOW_METRICS:$FLOW_CHECKPOINT" \
  --default-expert auto \
  --min-relative-gain 0.005 \
  --alpha 0.05 \
  --bootstrap-draws 5000 \
  --block-days 7 \
  --seed 202707 \
  --output "$ROUTE" >>"$LOG" 2>&1
"$PYTHON_BIN" tools/train/build_temporal_router_checkpoint.py \
  --route "$ROUTE" \
  --output "$ROUTER_CHECKPOINT" >>"$LOG" 2>&1
"$PYTHON_BIN" tools/train/temporal_router_checkpoint_status.py \
  "$ROUTER_CHECKPOINT" \
  --required-taus 1,2,3,4,5 \
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
print(len(hours), ",".join(str(hour / 6) for hour in hours))
' "$ROUTE")

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
  field_root="metrics/temporal_router_6h_$year"
  field_result="$field_root/$ROUTER_NAME.json"
  if ! "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
    "$field_result" \
    --checkpoint "$ROUTER_CHECKPOINT" \
    --required-taus 1,2,3,4,5 \
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
      --paper-tag "temporal_router_6h_${year}" \
      --batch-size 4 \
      --num-workers 2 \
      --samples-per-date 4 \
      --full-year \
      --max-tau-hours 6 \
      --eval-hours 1,2,3,4,5 \
      --seen-tau 1,3,5 \
      --unseen-tau 2,4 \
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
    --required-taus 1,2,3,4,5 \
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
    --out-dir "metrics/temporal_router_spectra_6h_$year" \
    --taus 1,2,3,4,5 \
    --channels all \
    --lmax 359 \
    --keep-n-channels 24 \
    --batch-size 2 \
    --samples-per-date 2 \
    --eval-days-per-month 2 \
    --hf-ell-min 180 \
    --max-tau-hours 6 \
    --skip-existing \
    >>"$LOG" 2>&1
done

"$PYTHON_BIN" -u tools/eval/benchmark_capmatched_inference.py \
  --model "$ROUTER_NAME:$ROUTER_CHECKPOINT" \
  --static-path data/static_features_0p5.pt \
  --batch-size 1 \
  --tau-values \
  0.1666666667,0.3333333333,0.5,0.6666666667,0.8333333333 \
  --height 360 \
  --width 720 \
  --warmup 5 \
  --iterations 10 \
  --out-json metrics/temporal_router_inference_cost_batch1_all_tau.json \
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
  --out-json metrics/temporal_router_inference_cost_batch5_mixed.json \
  >>"$LOG" 2>&1

flock -u "$lock_fd"
exec {lock_fd}>&-

for source_name in "$UPR_NAME" "$FLOW_NAME"; do
  source_checkpoint="$UPR_CHECKPOINT"
  if [[ "$source_name" == "$FLOW_NAME" ]]; then
    source_checkpoint="$FLOW_CHECKPOINT"
  fi
  source_ood="metrics/upr_lite_screen_6h_2021/$source_name.json"
  while ! "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
    "$source_ood" \
    --checkpoint "$source_checkpoint" \
    --required-taus 1,2,3,4,5 \
    --acc-mode enabled \
    --require-physical-metrics \
    --require-temporal-metrics \
    --quiet >>"$LOG" 2>&1; do
    echo "[$(date -Is)] wait source 2021 metrics=$source_name" \
      | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
done

"$PYTHON_BIN" tools/eval/assess_temporal_expert_router.py \
  --route "$ROUTE" \
  --router-checkpoint "$ROUTER_CHECKPOINT" \
  --router-field-2021 \
  "metrics/temporal_router_6h_2021/$ROUTER_NAME.json" \
  --expert-field-2021 \
  "upr=metrics/upr_lite_screen_6h_2021/$UPR_NAME.json" \
  --expert-field-2021 \
  "flow=metrics/upr_lite_screen_6h_2021/$FLOW_NAME.json" \
  --router-spectra-2020 metrics/temporal_router_spectra_6h_2020 \
  --router-name "$ROUTER_NAME" \
  --expert-spectra-2020 metrics/upr_lite_screen_spectra_6h_2020 \
  --expert-spectrum-prefix "upr=$UPR_NAME" \
  --expert-spectrum-prefix "flow=$FLOW_NAME" \
  --min-relative-gain 0.005 \
  --alpha 0.05 \
  --draws 5000 \
  --seed 202708 \
  --block-days 7 \
  --extreme-quantile 0.95 \
  --output metrics/temporal_router_generalization.json \
  >>"$LOG" 2>&1

touch "$TERMINAL"
echo "[$(date -Is)] temporal-router evaluation complete" | tee -a "$LOG"
