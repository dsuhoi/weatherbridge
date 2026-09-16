#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
MAX_UTIL="${MAX_UTIL:-5}"
GPU_STABLE_SEC="${GPU_STABLE_SEC:-20}"
SLEEP_SEC="${SLEEP_SEC:-15}"
MAX_RETRIES="${MAX_RETRIES:-2}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

FROZEN_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_a3d2bc02.py"
FROZEN_TRAINER_SHA256="a3d2bc024f2a95aa2ac2320fd44368e24faab18e232e1db91d2af73b3db1ee27"
MODEL_SOURCE="weather_time_interp/model/weatherbridge_flow_model.py"
MODEL_SOURCE_SHA256="8bc7bbeec1dad3b521648a8a85eed3598468d77bbfdbf8447bcf3fedc4cee6ac"
MEMMAP_DATASET="weather_time_interp/memmap_dataset.py"
MEMMAP_DATASET_SHA256="cc5c21f1150e9cdfa9acb1746557db88a080b09da3e7f8caa4207a25d60fc1fa"
TRAINING_PROTOCOL="tools/train/training_protocol.py"
TRAINING_PROTOCOL_SHA256="9810dd55a207a74c3a21722cf444976833356152e32a5e9373d5fcf0e88126fb"
NORMALIZATION="weather_time_interp/normalization.py"
NORMALIZATION_SHA256="b0055d02df62f34f665c56aa5b1d13bb77493d3f3b4e136e17d218e4a6717a7e"

EXP_NAME="${EXP_NAME:-exp_weatherbridge_geo_msf_l_9m_6h_s202707_v2_polarstable_bs8}"
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

verify_candidate_metadata() {
  local checkpoint="$1"
  "$PYTHON_BIN" -c '
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
hparams = checkpoint["hyper_parameters"]
protocol = hparams["training_protocol"]
state = checkpoint["state_dict"]
assert hparams["arch"] == "flow_geo_msf_l"
assert hparams["loss_profile"] == "pareto_minimax"
assert hparams["trainable_scope"] == "all"
assert hparams["intrinsic_transport_normalization"]["channel_order"] == [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]
assert protocol["train_years"] == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol["val_years"] == [2020]
assert protocol["train_tau_hours"] == [1, 3, 5]
assert protocol["eval_tau_hours"] == [1, 2, 3, 4, 5]
assert protocol["batch_size_per_device"] == 8
assert protocol["accumulate_grad_batches"] == 2
assert protocol["global_effective_batch_size"] == 16
assert protocol["optimizer_steps_per_epoch"] == 1642
assert not any("teacher" in key or "distill" in key for key in state)
net_parameters = sum(
    value.numel()
    for key, value in state.items()
    if key.startswith("net.")
    and key not in {
        "net.normalization_mean",
        "net.normalization_std",
    }
)
assert net_parameters == 8884205
assert state["net.normalization_mean"].shape == (24,)
assert state["net.normalization_std"].shape == (24,)
assert torch.isfinite(state["net.normalization_mean"]).all()
assert (state["net.normalization_std"] > 0).all()
expected_hashes = {
    "train_capacity_matched_6h.py": sys.argv[2],
    "weatherbridge_flow_model.py": sys.argv[3],
}
for name, expected in expected_hashes.items():
    assert hparams["training_code_sha256"][name] == expected
' "$checkpoint" "$FROZEN_TRAINER_SHA256" "$MODEL_SOURCE_SHA256"
}

candidate_ready() {
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$CHECKPOINT" \
    --min-epochs 4 \
    --expected-arch flow_geo_msf_l \
    --expected-total-steps 6568 \
    --min-global-step 6568 \
    --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256" \
    --expected-highpass-boundary antipodal_vector_parity \
    --expected-lambda-hf 0.05 \
    --expected-delta-t 6 \
    --require-training-protocol \
    --expected-train-years 2014,2015,2016,2017,2018,2019 \
    --expected-val-years 2020 \
    --expected-train-taus 1,3,5 \
    --expected-eval-taus 1,2,3,4,5 \
    --expected-seed 202707 \
    --expected-effective-batch-size 16 \
    --expected-batch-size-per-device 8 \
    --expected-accumulate-grad-batches 2 \
    --expected-train-batches-per-epoch 3284 \
    --expected-optimizer-steps-per-epoch 1642 \
    --expected-samples-per-date-train 4 \
    --expected-samples-per-date-val 2 \
    --expected-precision bf16-mixed \
    --quiet &&
    verify_candidate_metadata "$CHECKPOINT"
}

acquire_gpu() {
  local candidate
  while true; do
    for candidate in $GPU_CANDIDATES; do
      local lock_fd
      exec {lock_fd}>"$LOG_ROOT/.weatherbridge_geo_msf_l_gpu${candidate}.lock"
      if ! flock -n "$lock_fd"; then
        exec {lock_fd}>&-
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
          ACQUIRED_FD="$lock_fd"
          return
        fi
      fi
      flock -u "$lock_fd"
      exec {lock_fd}>&-
    done
    sleep "$SLEEP_SEC"
  done
}

mkdir -p "$LOG_ROOT"
rm -f "$COMPLETE_MARKER" "$TERMINAL_MARKER"
trap 'touch "$TERMINAL_MARKER"' ERR
verify_source "$FROZEN_TRAINER" "$FROZEN_TRAINER_SHA256"
verify_source "$MODEL_SOURCE" "$MODEL_SOURCE_SHA256"
verify_source "$MEMMAP_DATASET" "$MEMMAP_DATASET_SHA256"
verify_source "$TRAINING_PROTOCOL" "$TRAINING_PROTOCOL_SHA256"
verify_source "$NORMALIZATION" "$NORMALIZATION_SHA256"

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "[$(date -Is)] WeatherBridge-GeoMSF-L preflight passed" | tee -a "$QUEUE_LOG"
  trap - ERR
  exit 0
fi

exec 9>"$LOG_ROOT/$EXP_NAME.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] WeatherBridge-GeoMSF-L worker already active" \
    | tee -a "$QUEUE_LOG"
  exit 0
fi
if candidate_ready; then
  touch "$COMPLETE_MARKER"
  trap - ERR
  exit 0
fi
if [[ -e "$CHECKPOINT" ]]; then
  echo "invalid existing checkpoint: $CHECKPOINT" >&2
  exit 2
fi

attempt=0
while ! candidate_ready; do
  attempt=$((attempt + 1))
  if [[ "$attempt" -gt "$MAX_RETRIES" ]]; then
    echo "[$(date -Is)] retry limit reached" | tee -a "$QUEUE_LOG"
    exit 2
  fi
  acquire_gpu
  gpu="$ACQUIRED_GPU"
  lock_fd="$ACQUIRED_FD"
  echo "[$(date -Is)] launch gpu=$gpu attempt=$attempt" | tee -a "$QUEUE_LOG"
  set +e
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" -u "$FROZEN_TRAINER" \
      --arch flow_geo_msf_l \
      --exp_name "$EXP_NAME" \
      --log_root "$LOG_ROOT" \
      --gpus 0 \
      --bs 8 \
      --val_bs 4 \
      --accumulate 2 \
      --workers 4 \
      --val_workers 2 \
      --release_memmap_pages \
      --precision bf16-mixed \
      --seed 202707 \
      --years 2014 2015 2016 2017 2018 2019 \
      --val_years 2020 \
      --max_epochs 4 \
      --lr 1e-4 \
      --warmup_steps 500 \
      --window_hours 6 \
      --train_tau_subset 1 3 5 \
      --eval_tau 1 2 3 4 5 \
      --samples_per_date_train 4 \
      --samples_per_date_val 2 \
      --loss_profile pareto_minimax \
      --trainable_scope all \
      --train_batches_per_epoch 3284 \
      --ckpt_every_n_epochs 2
  ) >>"$RUN_LOG" 2>&1
  status=$?
  set -e
  flock -u "$lock_fd"
  exec {lock_fd}>&-
  echo "[$(date -Is)] finish status=$status" | tee -a "$QUEUE_LOG"
  if [[ "$status" -ne 0 ]]; then
    sleep "$SLEEP_SEC"
  fi
done

touch "$COMPLETE_MARKER"
trap - ERR
echo "[$(date -Is)] WeatherBridge-GeoMSF-L training complete" \
  | tee -a "$QUEUE_LOG"
