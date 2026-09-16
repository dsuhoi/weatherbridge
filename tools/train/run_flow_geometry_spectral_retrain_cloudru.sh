#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
CANDIDATE="${CANDIDATE:?set CANDIDATE to spectral or spherical}"
GPU="${GPU:?set GPU to the physical GPU index}"
MIN_FREE_MIB="${MIN_FREE_MIB:-70000}"
MAX_UTIL="${MAX_UTIL:-5}"
GPU_STABLE_SEC="${GPU_STABLE_SEC:-20}"
SLEEP_SEC="${SLEEP_SEC:-15}"
MAX_RETRIES="${MAX_RETRIES:-2}"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

case "$CANDIDATE" in
  spectral)
    ARCH="flow_pp3"
    EXP_NAME="exp_flow_pp3_spectral_14m_6h_s202707_v1_bs4"
    LAMBDA_HF="0.05"
    LAMBDA_SPEC="0.02"
    SPECTRAL_MASK="advected"
    HIGHPASS_BOUNDARY="periodic_lon_replicate_lat"
    ;;
  spherical)
    ARCH="flow_pp3_spherical"
    EXP_NAME="exp_flow_pp3_spherical_14m_6h_s202707_v1_bs4"
    LAMBDA_HF="0"
    LAMBDA_SPEC="0"
    SPECTRAL_MASK="all"
    HIGHPASS_BOUNDARY="antipodal_vector_parity"
    ;;
  *)
    echo "unsupported CANDIDATE=$CANDIDATE" >&2
    exit 2
    ;;
esac

FROZEN_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_cffe7534.py"
FROZEN_TRAINER_SHA256="cffe7534d55ece0720b26cc33c16fd848d30d64699cf6fd1c658d7a269bc060e"
MODEL_SOURCE="weather_time_interp/model/weatherbridge_flow_model.py"
MODEL_SOURCE_SHA256="8bc7bbeec1dad3b521648a8a85eed3598468d77bbfdbf8447bcf3fedc4cee6ac"
MEMMAP_DATASET="weather_time_interp/memmap_dataset.py"
MEMMAP_DATASET_SHA256="cc5c21f1150e9cdfa9acb1746557db88a080b09da3e7f8caa4207a25d60fc1fa"
TRAINING_PROTOCOL="tools/train/training_protocol.py"
TRAINING_PROTOCOL_SHA256="9810dd55a207a74c3a21722cf444976833356152e32a5e9373d5fcf0e88126fb"
NORMALIZATION="weather_time_interp/normalization.py"
NORMALIZATION_SHA256="b0055d02df62f34f665c56aa5b1d13bb77493d3f3b4e136e17d218e4a6717a7e"

EXP_ROOT="$LOG_ROOT/$EXP_NAME"
CHECKPOINT="$EXP_ROOT/last.ckpt"
RUN_LOG="$LOG_ROOT/$EXP_NAME.log"
QUEUE_LOG="$LOG_ROOT/$EXP_NAME.queue.log"
COMPLETE_MARKER="$LOG_ROOT/$EXP_NAME.complete"
TERMINAL_MARKER="$LOG_ROOT/$EXP_NAME.terminal"

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

verify_checkpoint() {
  local checkpoint="$1"
  "$PYTHON_BIN" -c '
import math
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
hparams = checkpoint["hyper_parameters"]
protocol = hparams["training_protocol"]
state = checkpoint["state_dict"]
assert hparams["arch"] == sys.argv[2]
assert math.isclose(float(hparams["lambda_hf_override"]), float(sys.argv[3]))
assert math.isclose(float(hparams["lambda_spec_override"]), float(sys.argv[4]))
assert hparams["spectral_mask_profile"] == sys.argv[5]
assert protocol["train_years"] == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol["val_years"] == [2020]
assert protocol["train_tau_hours"] == [1, 3, 5]
assert protocol["eval_tau_hours"] == [1, 2, 3, 4, 5]
assert protocol["global_effective_batch_size"] == 16
assert protocol["optimizer_steps_per_epoch"] == 1642
assert protocol["lambda_spec"] == float(sys.argv[4])
assert protocol["spectral_mask_profile"] == sys.argv[5]
assert not any("teacher" in key or "distill" in key for key in state)
net_parameters = sum(
    value.numel()
    for key, value in state.items()
    if key.startswith("net.")
)
assert net_parameters == 14260565
assert hparams["training_code_sha256"]["train_capacity_matched_6h.py"] == sys.argv[6]
assert hparams["training_code_sha256"]["weatherbridge_flow_model.py"] == sys.argv[7]
' "$checkpoint" "$ARCH" "$LAMBDA_HF" "$LAMBDA_SPEC" "$SPECTRAL_MASK" \
    "$FROZEN_TRAINER_SHA256" "$MODEL_SOURCE_SHA256"
}

candidate_ready() {
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$CHECKPOINT" \
    --min-epochs 8 \
    --expected-arch "$ARCH" \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256" \
    --expected-highpass-boundary "$HIGHPASS_BOUNDARY" \
    --expected-lambda-hf "$LAMBDA_HF" \
    --expected-delta-t 6 \
    --require-training-protocol \
    --expected-train-years 2014,2015,2016,2017,2018,2019 \
    --expected-val-years 2020 \
    --expected-train-taus 1,3,5 \
    --expected-eval-taus 1,2,3,4,5 \
    --expected-seed 202707 \
    --expected-effective-batch-size 16 \
    --expected-batch-size-per-device 4 \
    --expected-accumulate-grad-batches 4 \
    --expected-train-batches-per-epoch 6568 \
    --expected-optimizer-steps-per-epoch 1642 \
    --expected-samples-per-date-train 4 \
    --expected-samples-per-date-val 2 \
    --expected-precision bf16-mixed \
    --quiet &&
    verify_checkpoint "$CHECKPOINT"
}

wait_for_gpu() {
  while true; do
    local free_mib
    local util
    free_mib="$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
    if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      sleep "$GPU_STABLE_SEC"
      free_mib="$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
      util="$(nvidia-smi --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
      if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
        return
      fi
    fi
    sleep "$SLEEP_SEC"
  done
}

mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/$EXP_NAME.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] $EXP_NAME already active" | tee -a "$QUEUE_LOG"
  exit 0
fi
rm -f "$COMPLETE_MARKER" "$TERMINAL_MARKER"
trap 'touch "$TERMINAL_MARKER"' ERR

verify_source "$FROZEN_TRAINER" "$FROZEN_TRAINER_SHA256"
verify_source tools/train/train_capacity_matched_6h.py "$FROZEN_TRAINER_SHA256"
verify_source "$MODEL_SOURCE" "$MODEL_SOURCE_SHA256"
verify_source "$MEMMAP_DATASET" "$MEMMAP_DATASET_SHA256"
verify_source "$TRAINING_PROTOCOL" "$TRAINING_PROTOCOL_SHA256"
verify_source "$NORMALIZATION" "$NORMALIZATION_SHA256"

if candidate_ready; then
  touch "$COMPLETE_MARKER"
  trap - ERR
  exit 0
fi

attempt=0
while ! candidate_ready; do
  attempt=$((attempt + 1))
  if [[ "$attempt" -gt "$MAX_RETRIES" ]]; then
    echo "retry limit reached" >&2
    exit 2
  fi
  wait_for_gpu
  resume=()
  if [[ -s "$CHECKPOINT" ]]; then
    resume=(--ckpt_path "$CHECKPOINT")
  fi
  echo "[$(date -Is)] launch candidate=$CANDIDATE gpu=$GPU attempt=$attempt" \
    | tee -a "$QUEUE_LOG"
  set +e
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    "$PYTHON_BIN" -u "$FROZEN_TRAINER" \
      --arch "$ARCH" \
      --exp_name "$EXP_NAME" \
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
      --warmup_steps 500 \
      --window_hours 6 \
      --train_tau_subset 1 3 5 \
      --eval_tau 1 2 3 4 5 \
      --samples_per_date_train 4 \
      --samples_per_date_val 2 \
      --lambda_hf_override "$LAMBDA_HF" \
      --lambda_spec_override "$LAMBDA_SPEC" \
      --spectral_mask_profile "$SPECTRAL_MASK" \
      --loss_profile uniform \
      --trainable_scope all \
      --train_batches_per_epoch 6568 \
      --ckpt_every_n_epochs 2 \
      "${resume[@]}"
  ) >>"$RUN_LOG" 2>&1
  status=$?
  set -e
  echo "[$(date -Is)] finish status=$status" | tee -a "$QUEUE_LOG"
  if [[ "$status" -ne 0 ]]; then
    sleep "$SLEEP_SEC"
  fi
done

touch "$COMPLETE_MARKER"
trap - ERR
echo "[$(date -Is)] complete candidate=$CANDIDATE" | tee -a "$QUEUE_LOG"
