#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
SLEEP_SEC="${SLEEP_SEC:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"
MAX_UTIL="${MAX_UTIL:-5}"
GPU_STABLE_SEC="${GPU_STABLE_SEC:-30}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
FROZEN_TRAINER_SHA256="${FROZEN_TRAINER_SHA256:-e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a}"

reference_name="exp_upr_implicit_global_14m_hf_135_14m_6h_s202707_protocol_v2"
reference_checkpoint="$LOG_ROOT/$reference_name/last.ckpt"
control_dir="$LOG_ROOT/upr_endpoint_zero_shot"
control_checkpoint="$control_dir/upr_endpoint_implicit_global_14m.ckpt"
output_dir="metrics/upr_endpoint_zero_shot_2020"
assessment="metrics/upr_lite_screen/upr_endpoint_zero_shot_2020.json"
queue_log="$LOG_ROOT/upr_endpoint_zero_shot.queue.log"

protocol_args=(
  --expected-delta-t 6
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
  --expected-lambda-hf 0.05
  --expected-highpass-boundary periodic_lon_replicate_lat
  --expected-precision bf16-mixed
  --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256"
)

mkdir -p "$LOG_ROOT" "$control_dir"
exec 9>"$LOG_ROOT/upr_endpoint_zero_shot.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] endpoint zero-shot queue already active" \
    | tee -a "$queue_log"
  exit 0
fi

while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$reference_checkpoint" \
  --min-epochs 8 \
  --expected-arch upr_implicit_global_14m \
  --expected-total-steps 13136 \
  --min-global-step 13136 \
  "${protocol_args[@]}" \
  --quiet; do
  sleep "$SLEEP_SEC"
done

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
      sleep "$GPU_STABLE_SEC"
      free_mib="$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
      util="$(nvidia-smi --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
      if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
        gpu="$candidate"
        lock_fd="$candidate_fd"
        break
      fi
    fi
    flock -u "$candidate_fd"
    exec {candidate_fd}>&-
  done
  if [[ -z "$gpu" ]]; then
    sleep "$SLEEP_SEC"
  fi
done

"$PYTHON_BIN" tools/train/make_state_compatible_control_checkpoint.py \
  --source "$reference_checkpoint" \
  --output "$control_checkpoint" \
  --expected-source-arch upr_implicit_global_14m \
  --target-arch upr_endpoint_implicit_global_14m \
  >>"$queue_log" 2>&1

(
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONPATH="$PWD:${PYTHONPATH:-}"
  "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "upr_base:$reference_checkpoint,upr_endpoint:$control_checkpoint" \
    --out-dir "$output_dir" \
    --paper-tag upr_endpoint_zero_shot_allocation \
    --batch-size 2 \
    --num-workers 2 \
    --samples-per-date 2 \
    --eval-days-per-month 4 \
    --proper-rmse \
    --save-window-metrics \
    --max-tau-hours 6 \
    --eval-hours 1,2,3,4,5 \
    --seen-tau 1,3,5 \
    --unseen-tau 2,4 \
    --keep-n-channels 24 \
    --no-acc
) >>"$queue_log" 2>&1

"$PYTHON_BIN" tools/eval/assess_upr_endpoint_zero_shot.py \
  --reference-json "$output_dir/upr_base.json" \
  --candidate-json "$output_dir/upr_endpoint.json" \
  --output "$assessment" \
  >>"$queue_log" 2>&1

flock -u "$lock_fd"
echo "[$(date -Is)] endpoint zero-shot assessment=$assessment" \
  | tee -a "$queue_log"
