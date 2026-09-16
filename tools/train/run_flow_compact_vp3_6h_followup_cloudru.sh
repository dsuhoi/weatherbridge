#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
METRICS_ROOT="${METRICS_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime/metrics}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
MAX_UTIL="${MAX_UTIL:-5}"
GPU_STABLE_SEC="${GPU_STABLE_SEC:-20}"
SLEEP_SEC="${SLEEP_SEC:-15}"
MAX_RETRIES="${MAX_RETRIES:-3}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
FROZEN_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_38ca4e17.py"
FROZEN_TRAINER_SHA256="38ca4e176f84b217a641598f2a2d4cb7c4554ab4ae1d36f53630a8f9fc112ff0"
MODEL_SOURCE="weather_time_interp/model/weatherbridge_flow_model.py"
MODEL_SOURCE_SHA256="48534ac2926f0ec9ebb845efffa9dff9da93bdaa179e41bce9e1634ae1ff0082"
MEMMAP_DATASET="weather_time_interp/memmap_dataset.py"
MEMMAP_DATASET_SHA256="cc5c21f1150e9cdfa9acb1746557db88a080b09da3e7f8caa4207a25d60fc1fa"
REFERENCE_TRAINER_SHA256="e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

declare -A EXP_NAMES=(
  [flow_compact_vp3]="exp_flow_compact_vp3_hf_135_3m_6h_s202707_protocol_v2"
  [flow_compact_vp3_m]="exp_flow_compact_vp3_m_hf_135_5m_6h_s202707_protocol_v1"
)
declare -A COMPUTE_JSONS=(
  [flow_compact_vp3]="$METRICS_ROOT/transport_candidate_compute/flow_compact_vp3_compute_a5000.json"
  [flow_compact_vp3_m]="$METRICS_ROOT/transport_candidate_compute/flow_compact_vp3_m_compute_a5000.json"
)
declare -A MAX_PARAMETERS=(
  [flow_compact_vp3]=3100000
  [flow_compact_vp3_m]=4800000
)
declare -A MAX_INFERENCE_MS=(
  [flow_compact_vp3]=30
  [flow_compact_vp3_m]=40
)
declare -A MAX_TRAINING_MS=(
  [flow_compact_vp3]=90
  [flow_compact_vp3_m]=120
)
declare -A MAX_TRAINING_MIB=(
  [flow_compact_vp3]=1700
  [flow_compact_vp3_m]=2100
)

FLOW_NAME="weatherbridge_ref"
FLOW_EXP="exp_flow_pp3_135_14m_6h_s202707_protocol_v2"
FLOW_CHECKPOINT="$LOG_ROOT/$FLOW_EXP/last.ckpt"
UPR_NAME="upr_lite_implicit_global"
UPR_EXP="exp_upr_lite_implicit_global_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707"
UPR_SCREEN="$METRICS_ROOT/upr_lite_screen/two_epoch_screen.json"
SCREEN_REPORT="$METRICS_ROOT/compact_flow_screen_6h/two_epoch_selection.json"
FINAL_REPORT="$METRICS_ROOT/compact_flow_screen_6h/four_epoch_selection.json"
QUEUE_LOG="$LOG_ROOT/compact_flow_short_budget.queue.log"
TERMINAL_MARKER="$LOG_ROOT/compact_flow_short_budget.terminal"
COMPLETE_MARKER="$LOG_ROOT/compact_flow_short_budget.complete"

common_protocol_args=(
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
  --expected-train-batches-per-epoch 6568
  --expected-optimizer-steps-per-epoch 1642
  --expected-samples-per-date-train 4
  --expected-samples-per-date-val 2
  --expected-precision bf16-mixed
)

candidate_ready() {
  local arch="$1"
  local path="$2"
  local min_epochs="$3"
  local min_step="$4"
  shift 4
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$path" \
    --min-epochs "$min_epochs" \
    --expected-arch "$arch" \
    --expected-total-steps 13136 \
    --min-global-step "$min_step" \
    --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256" \
    --expected-highpass-boundary antipodal_vector_parity \
    --expected-lambda-hf 0.05 \
    "${common_protocol_args[@]}" \
    "$@" \
    --quiet
}

flow_ready() {
  local min_epochs="$1"
  local min_step="$2"
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$FLOW_CHECKPOINT" \
    --min-epochs "$min_epochs" \
    --expected-arch flow_pp3 \
    --expected-total-steps 13136 \
    --min-global-step "$min_step" \
    --expected-trainer-sha256 "$REFERENCE_TRAINER_SHA256" \
    --expected-highpass-boundary periodic_lon_replicate_lat \
    --expected-lambda-hf 0 \
    "${common_protocol_args[@]}" \
    --quiet
}

verify_source() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "source SHA-256 mismatch path=$path actual=$actual" >&2
    exit 2
  fi
}

verify_compute_artifact() {
  local arch="$1"
  "$PYTHON_BIN" -c '
import json, sys
artifact = json.load(open(sys.argv[1]))
result = artifact["result"]
assert artifact["evidence_level"] == "synthetic_compute_only"
assert result["arch"] == sys.argv[2]
assert result["parameters"] <= int(sys.argv[3])
assert result["inference_ms_median"] <= float(sys.argv[4])
assert result["training_ms_median"] <= float(sys.argv[5])
assert result["training_peak_mib"] <= float(sys.argv[6])
' \
    "${COMPUTE_JSONS[$arch]}" \
    "$arch" \
    "${MAX_PARAMETERS[$arch]}" \
    "${MAX_INFERENCE_MS[$arch]}" \
    "${MAX_TRAINING_MS[$arch]}" \
    "${MAX_TRAINING_MIB[$arch]}"
}

acquire_gpu() {
  local candidate
  ACQUIRED_GPU=""
  ACQUIRED_FD=""
  while true; do
    for candidate in $GPU_CANDIDATES; do
      local candidate_fd
      exec {candidate_fd}>"$LOG_ROOT/.upr_lite_gpu${candidate}.lock"
      if ! flock -n "$candidate_fd"; then
        exec {candidate_fd}>&-
        continue
      fi
      local free_mib
      local util
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
        if [[ "$free_mib" -ge "$MIN_FREE_MIB" && \
              "$util" -le "$MAX_UTIL" ]]; then
          ACQUIRED_GPU="$candidate"
          ACQUIRED_FD="$candidate_fd"
          return
        fi
      fi
      flock -u "$candidate_fd"
      exec {candidate_fd}>&-
    done
    sleep "$SLEEP_SEC"
  done
}

train_until() {
  local arch="$1"
  local target_epochs="$2"
  local target_step="$3"
  local exp_name="${EXP_NAMES[$arch]}"
  local checkpoint="$LOG_ROOT/$exp_name/last.ckpt"
  local run_log="$LOG_ROOT/$exp_name.log"
  local queue_log="$LOG_ROOT/$exp_name.queue.log"
  local terminal_marker="$LOG_ROOT/${exp_name}.terminal"

  exec 8>"$LOG_ROOT/${exp_name}.queue.lock"
  if ! flock -n 8; then
    echo "[$(date -Is)] worker already active arch=$arch" \
      | tee -a "$queue_log"
    return 0
  fi
  if candidate_ready "$arch" "$checkpoint" "$target_epochs" "$target_step"; then
    return 0
  fi

  local attempt=0
  while ! candidate_ready \
    "$arch" "$checkpoint" "$target_epochs" "$target_step"; do
    attempt=$((attempt + 1))
    if [[ "$attempt" -gt "$MAX_RETRIES" ]]; then
      touch "$terminal_marker"
      echo "[$(date -Is)] retry limit reached arch=$arch" \
        | tee -a "$queue_log"
      return 2
    fi

    local gpu
    local lock_fd
    acquire_gpu
    gpu="$ACQUIRED_GPU"
    lock_fd="$ACQUIRED_FD"
    local resume=()
    if [[ -s "$checkpoint" ]]; then
      resume=(--ckpt_path "$checkpoint")
    fi
    echo "[$(date -Is)] launch arch=$arch target_epochs=$target_epochs gpu=$gpu" \
      | tee -a "$queue_log"
    set +e
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PYTHON_BIN" -u "$FROZEN_TRAINER" \
        --arch "$arch" \
        --exp_name "$exp_name" \
        --log_root "$LOG_ROOT" \
        --gpus 0 \
        --bs 4 \
        --val_bs 2 \
        --accumulate 4 \
        --workers 4 \
        --val_workers 2 \
        --release_memmap_pages \
        --precision bf16-mixed \
        --seed 202707 \
        --years 2014 2015 2016 2017 2018 2019 \
        --val_years 2020 \
        --max_epochs 8 \
        --lr 1e-4 \
        --window_hours 6 \
        --train_tau_subset 1 3 5 \
        --eval_tau 1 2 3 4 5 \
        --samples_per_date_train 4 \
        --samples_per_date_val 2 \
        --lambda_hf_override 0.05 \
        --train_batches_per_epoch 6568 \
        --ckpt_every_n_epochs 1 \
        "${resume[@]}"
    ) >>"$run_log" 2>&1 &
    local train_pid=$!
    "$PYTHON_BIN" tools/train/stop_after_screen_checkpoint.py \
      --queue-pid "$train_pid" \
      --experiment "$exp_name" \
      --checkpoint "$checkpoint" \
      --min-epochs "$target_epochs" \
      --poll-seconds 30 >>"$queue_log" 2>&1
    local stop_status=$?
    wait "$train_pid"
    local train_status=$?
    set -e
    flock -u "$lock_fd"
    exec {lock_fd}>&-
    echo "[$(date -Is)] finish arch=$arch train=$train_status stop=$stop_status" \
      | tee -a "$queue_log"
    if ! candidate_ready \
      "$arch" "$checkpoint" "$target_epochs" "$target_step"; then
      sleep "$SLEEP_SEC"
    fi
  done
}

metrics_csv() {
  local exp_name="$1"
  local epoch="$2"
  local step="$3"
  "$PYTHON_BIN" tools/train/select_validation_metrics_csv.py \
    "$LOG_ROOT/$exp_name" \
    --required-epoch "$epoch" \
    --required-step "$step"
}

mkdir -p "$LOG_ROOT" "$(dirname "$SCREEN_REPORT")"
rm -f "$TERMINAL_MARKER" "$COMPLETE_MARKER"
trap 'touch "$TERMINAL_MARKER"' ERR
verify_source "$FROZEN_TRAINER" "$FROZEN_TRAINER_SHA256"
verify_source "$MODEL_SOURCE" "$MODEL_SOURCE_SHA256"
verify_source "$MEMMAP_DATASET" "$MEMMAP_DATASET_SHA256"
for arch in flow_compact_vp3 flow_compact_vp3_m; do
  verify_compute_artifact "$arch"
done
flow_ready 4 6568

UPR_SCREEN_CSV="$("$PYTHON_BIN" -c '
from pathlib import Path
import hashlib, json, sys
screen = json.loads(Path(sys.argv[1]).read_text())
assert screen["complete"] is True
assert screen["screen"]["temporal_ood_2021_used"] is False
model = screen["models"]["upr_lite_implicit_global"]
assert model["epoch"] == 1 and model["step"] == 3283
path = Path(model["metrics_file"])
assert hashlib.sha256(path.read_bytes()).hexdigest() == model["metrics_sha256"]
print(path)
' "$UPR_SCREEN")"
UPR_FINAL_CSV="$(metrics_csv "$UPR_EXP" 3 6567)"
FLOW_CSV="$(metrics_csv "$FLOW_EXP" 3 6567)"
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "[$(date -Is)] compact short-budget preflight passed" \
    | tee -a "$QUEUE_LOG"
  trap - ERR
  exit 0
fi

train_until flow_compact_vp3 2 3284 &
small_pid=$!
train_until flow_compact_vp3_m 2 3284 &
medium_pid=$!
status=0
wait "$small_pid" || status=1
wait "$medium_pid" || status=1
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

SMALL_CSV="$(metrics_csv "${EXP_NAMES[flow_compact_vp3]}" 1 3283)"
MEDIUM_CSV="$(metrics_csv "${EXP_NAMES[flow_compact_vp3_m]}" 1 3283)"
"$PYTHON_BIN" tools/eval/select_compact_flow_candidate.py \
  --candidate "$UPR_NAME=$UPR_SCREEN_CSV" \
  --candidate "flow_compact_vp3=$SMALL_CSV" \
  --candidate "flow_compact_vp3_m=$MEDIUM_CSV" \
  --reference-csv "$FLOW_CSV" \
  --reference-name "$FLOW_NAME" \
  --epoch 1 \
  --held-relative-limit 0.03 \
  --all-hour-relative-limit 0.03 \
  --per-hour-relative-limit 0.08 \
  --output "$SCREEN_REPORT" | tee -a "$QUEUE_LOG"
selected="$("$PYTHON_BIN" -c '
import json, sys
print(json.load(open(sys.argv[1]))["selected"] or "")
' "$SCREEN_REPORT")"
if [[ -z "$selected" ]]; then
  echo "[$(date -Is)] no compact candidate passed the two-epoch gate" \
    | tee -a "$QUEUE_LOG"
  touch "$COMPLETE_MARKER"
  trap - ERR
  exit 0
fi

if [[ "$selected" == flow_compact_vp3* ]]; then
  train_until "$selected" 4 6568
fi

final_candidates=(--candidate "$UPR_NAME=$UPR_FINAL_CSV")
if [[ "$selected" == flow_compact_vp3* ]]; then
  selected_csv="$(metrics_csv "${EXP_NAMES[$selected]}" 3 6567)"
  final_candidates+=(--candidate "$selected=$selected_csv")
fi
"$PYTHON_BIN" tools/eval/select_compact_flow_candidate.py \
  "${final_candidates[@]}" \
  --reference-csv "$FLOW_CSV" \
  --reference-name "$FLOW_NAME" \
  --epoch 3 \
  --held-relative-limit 0 \
  --all-hour-relative-limit 0 \
  --per-hour-relative-limit 0.05 \
  --output "$FINAL_REPORT" | tee -a "$QUEUE_LOG"

touch "$COMPLETE_MARKER"
trap - ERR
echo "[$(date -Is)] compact short-budget screen complete report=$FINAL_REPORT" \
  | tee -a "$QUEUE_LOG"
