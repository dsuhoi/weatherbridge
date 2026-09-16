#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
MAX_UTIL="${MAX_UTIL:-5}"
GPU_STABLE_SEC="${GPU_STABLE_SEC:-20}"
SLEEP_SEC="${SLEEP_SEC:-15}"
MAX_RETRIES="${MAX_RETRIES:-2}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

FROZEN_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_7c4c9279.py"
FROZEN_TRAINER_SHA256="7c4c9279f5adf323cd9ee2f5bc40605a7112d273584f1b08f5dfd8160829706b"
MODEL_SOURCE="weather_time_interp/model/weatherbridge_flow_model.py"
MODEL_SOURCE_SHA256="af9dbcffc9201a10e58700b8a7a9ee5d6d3be054c483e9b67f358987a99cdc8d"
MEMMAP_DATASET="weather_time_interp/memmap_dataset.py"
MEMMAP_DATASET_SHA256="cc5c21f1150e9cdfa9acb1746557db88a080b09da3e7f8caa4207a25d60fc1fa"

SCALE_SCREEN_COMPLETE="$LOG_ROOT/lagrange_head_scale_eval.complete"
INITIAL_CHECKPOINT="$LOG_ROOT/exp_flow_compact_lagrange_l_headft_lr1e3_9m_6h_s202707_v1/last.ckpt"
INITIAL_CHECKPOINT_SHA256="071db44cc69c97383c0f2ecd08ad88b46dc5628d706e75e6fc7d6a95780c21f2"
COMPUTE_JSON="/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime/metrics/transport_candidate_compute/flow_compact_lagrange_l_compute_a100.json"

declare -A EXP_NAMES=(
  [base]="exp_flow_compact_lagrange_l_allft_base_lr1e5_9m_6h_s202707_v1"
  [edge]="exp_flow_compact_lagrange_l_allft_edge_lr1e5_9m_6h_s202707_v1"
)
declare -A LOSS_PROFILES=(
  [base]="base_balanced"
  [edge]="base_edge_balanced"
)

QUEUE_LOG="$LOG_ROOT/lagrange_allnet_lastmile.queue.log"
TERMINAL_MARKER="$LOG_ROOT/lagrange_allnet_lastmile.terminal"
COMPLETE_MARKER="$LOG_ROOT/lagrange_allnet_lastmile.complete"

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

verify_initial_checkpoint() {
  verify_source "$INITIAL_CHECKPOINT" "$INITIAL_CHECKPOINT_SHA256"
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$INITIAL_CHECKPOINT" \
    --min-epochs 1 \
    --expected-arch flow_compact_lagrange_l \
    --expected-total-steps 1642 \
    --min-global-step 1642 \
    --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256" \
    --expected-highpass-boundary antipodal_vector_parity \
    --expected-lambda-hf 0 \
    "${common_protocol_args[@]}" \
    --quiet
  "$PYTHON_BIN" -c '
import sys, torch
hparams = torch.load(
    sys.argv[1], map_location="cpu", weights_only=False
)["hyper_parameters"]
assert hparams["trainable_scope"] == "base_knot_head"
assert hparams["loss_profile"] == "base_edge_balanced"
' "$INITIAL_CHECKPOINT"
}

verify_compute_artifact() {
  "$PYTHON_BIN" -c '
import json, sys
result = json.load(open(sys.argv[1]))["result"]
assert result["arch"] == "flow_compact_lagrange_l"
assert result["parameters"] == 8806801
assert result["inference_ms_median"] <= 50
assert result["training_ms_median"] <= 145
assert result["training_peak_mib"] <= 2700
' "$COMPUTE_JSON"
}

verify_finetune_metadata() {
  local checkpoint="$1"
  local arm="$2"
  "$PYTHON_BIN" -c '
import sys, torch
hparams = torch.load(
    sys.argv[1], map_location="cpu", weights_only=False
)["hyper_parameters"]
protocol = hparams["training_protocol"]
lineage = hparams["initialization_lineage"]
assert hparams["loss_profile"] == sys.argv[2]
assert protocol["loss_profile"] == sys.argv[2]
assert hparams["trainable_scope"] == "all"
assert protocol["trainable_scope"] == "all"
assert float(hparams["lr"]) == 1e-5
assert hparams["warmup_steps"] == 100
assert lineage["checkpoint_sha256"] == sys.argv[3]
assert lineage["epoch"] == 0
assert lineage["global_step"] == 1642
assert lineage["model"]["arch"] == "flow_compact_lagrange_l"
assert "compatibility" not in lineage
assert "resume_lineage" not in hparams
' "$checkpoint" "${LOSS_PROFILES[$arm]}" "$INITIAL_CHECKPOINT_SHA256"
}

candidate_ready() {
  local arm="$1"
  local checkpoint="$2"
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$checkpoint" \
    --min-epochs 1 \
    --expected-arch flow_compact_lagrange_l \
    --expected-total-steps 1642 \
    --min-global-step 1642 \
    --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256" \
    --expected-highpass-boundary antipodal_vector_parity \
    --expected-lambda-hf 0.02 \
    "${common_protocol_args[@]}" \
    --quiet &&
    verify_finetune_metadata "$checkpoint" "$arm"
}

acquire_gpu() {
  local candidate
  ACQUIRED_GPU=""
  ACQUIRED_FD=""
  while true; do
    for candidate in $GPU_CANDIDATES; do
      local candidate_fd
      exec {candidate_fd}>"$LOG_ROOT/.lagrange_allnet_gpu${candidate}.lock"
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
        if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
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

train_arm() {
  local arm="$1"
  local exp_name="${EXP_NAMES[$arm]}"
  local checkpoint="$LOG_ROOT/$exp_name/last.ckpt"
  local run_log="$LOG_ROOT/$exp_name.log"
  local queue_log="$LOG_ROOT/$exp_name.queue.log"

  exec 8>"$LOG_ROOT/$exp_name.queue.lock"
  if ! flock -n 8; then
    echo "[$(date -Is)] worker already active arm=$arm" | tee -a "$queue_log"
    return 0
  fi
  if candidate_ready "$arm" "$checkpoint"; then
    return 0
  fi
  if [[ -e "$checkpoint" ]]; then
    echo "invalid existing fine-tune checkpoint: $checkpoint" >&2
    return 2
  fi

  local attempt=0
  while ! candidate_ready "$arm" "$checkpoint"; do
    attempt=$((attempt + 1))
    if [[ "$attempt" -gt "$MAX_RETRIES" ]]; then
      echo "[$(date -Is)] retry limit reached arm=$arm" | tee -a "$queue_log"
      return 2
    fi
    acquire_gpu
    local gpu="$ACQUIRED_GPU"
    local lock_fd="$ACQUIRED_FD"
    echo "[$(date -Is)] launch arm=$arm gpu=$gpu" | tee -a "$queue_log"
    set +e
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PYTHON_BIN" -u "$FROZEN_TRAINER" \
        --arch flow_compact_lagrange_l \
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
        --max_epochs 1 \
        --lr 1e-5 \
        --warmup_steps 100 \
        --window_hours 6 \
        --train_tau_subset 1 3 5 \
        --eval_tau 1 2 3 4 5 \
        --samples_per_date_train 4 \
        --samples_per_date_val 2 \
        --lambda_hf_override 0.02 \
        --loss_profile "${LOSS_PROFILES[$arm]}" \
        --trainable_scope all \
        --init_weights_path "$INITIAL_CHECKPOINT" \
        --train_batches_per_epoch 6568 \
        --ckpt_every_n_epochs 1
    ) >>"$run_log" 2>&1
    local status=$?
    set -e
    flock -u "$lock_fd"
    exec {lock_fd}>&-
    echo "[$(date -Is)] finish arm=$arm status=$status" | tee -a "$queue_log"
    if [[ "$status" -ne 0 ]]; then
      sleep "$SLEEP_SEC"
    fi
  done
}

mkdir -p "$LOG_ROOT"
rm -f "$TERMINAL_MARKER" "$COMPLETE_MARKER"
trap 'touch "$TERMINAL_MARKER"' ERR
verify_source "$FROZEN_TRAINER" "$FROZEN_TRAINER_SHA256"
verify_source "$MODEL_SOURCE" "$MODEL_SOURCE_SHA256"
verify_source "$MEMMAP_DATASET" "$MEMMAP_DATASET_SHA256"
verify_initial_checkpoint
verify_compute_artifact

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "[$(date -Is)] Lagrange all-network preflight passed" \
    | tee -a "$QUEUE_LOG"
  trap - ERR
  exit 0
fi

while [[ ! -e "$SCALE_SCREEN_COMPLETE" ]]; do
  echo "[$(date -Is)] wait Lagrange scale screen" | tee -a "$QUEUE_LOG"
  sleep "$SLEEP_SEC"
done

train_arm base &
base_pid=$!
train_arm edge &
edge_pid=$!
status=0
wait "$base_pid" || status=1
wait "$edge_pid" || status=1
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

touch "$COMPLETE_MARKER"
trap - ERR
echo "[$(date -Is)] Lagrange all-network fine-tunes complete" \
  | tee -a "$QUEUE_LOG"
