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

FROZEN_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_f5b026e1.py"
FROZEN_TRAINER_SHA256="f5b026e17e146417fdf5b8cc89109a4d014646a30556acf0520239f404f9d179"
MODEL_SOURCE="weather_time_interp/model/weatherbridge_flow_model.py"
MODEL_SOURCE_SHA256="af9dbcffc9201a10e58700b8a7a9ee5d6d3be054c483e9b67f358987a99cdc8d"
MEMMAP_DATASET="weather_time_interp/memmap_dataset.py"
MEMMAP_DATASET_SHA256="cc5c21f1150e9cdfa9acb1746557db88a080b09da3e7f8caa4207a25d60fc1fa"
TEACHER_CHECKPOINT="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202707_protocol_v2/epoch=3-step=6568.ckpt"
TEACHER_SHA256="3df6746a7e86f7ead8077aa6a5db0ad2c2bd0a4725d32dde0fb7813f53ab64da"

declare -A EXP_NAMES=(
  [plain]="exp_flow_pp3_compact_l_plain_9m_6h_s202707_v1"
  [distill]="exp_flow_pp3_compact_l_distill02_9m_6h_s202707_v1"
)
declare -A DISTILL_WEIGHTS=(
  [plain]="0"
  [distill]="0.2"
)

QUEUE_LOG="$LOG_ROOT/flow_pp3_compact_l.queue.log"
TERMINAL_MARKER="$LOG_ROOT/flow_pp3_compact_l.terminal"
COMPLETE_MARKER="$LOG_ROOT/flow_pp3_compact_l.complete"

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

verify_teacher() {
  verify_source "$TEACHER_CHECKPOINT" "$TEACHER_SHA256"
  "$PYTHON_BIN" -c '
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
hparams = checkpoint["hyper_parameters"]
protocol = hparams["training_protocol"]
assert hparams["arch"] == "flow_pp3"
assert checkpoint["epoch"] == 3
assert checkpoint["global_step"] == 6568
assert protocol["train_years"] == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol["val_years"] == [2020]
assert protocol["train_tau_hours"] == [1, 3, 5]
assert protocol["eval_tau_hours"] == [1, 2, 3, 4, 5]
' "$TEACHER_CHECKPOINT"
}

verify_candidate_metadata() {
  local checkpoint="$1"
  local arm="$2"
  "$PYTHON_BIN" -c '
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
hparams = checkpoint["hyper_parameters"]
protocol = hparams["training_protocol"]
state = checkpoint["state_dict"]
expected_weight = float(sys.argv[2])
assert hparams["arch"] == "flow_pp3_compact_l"
assert hparams["loss_profile"] == "uniform"
assert hparams["trainable_scope"] == "all"
assert float(hparams["distill_weight"]) == expected_weight
assert float(protocol["distill_weight"]) == expected_weight
assert not any(key.startswith("distill_teacher.") for key in state)
net_params = sum(
    value.numel()
    for key, value in state.items()
    if key.startswith("net.")
)
assert net_params == 8838773
if expected_weight:
    lineage = hparams["distillation_teacher_lineage"]
    assert lineage["checkpoint_sha256"] == sys.argv[3]
    assert lineage["model"]["arch"] == "flow_pp3"
else:
    assert "distillation_teacher_lineage" not in hparams
' "$checkpoint" "${DISTILL_WEIGHTS[$arm]}" "$TEACHER_SHA256"
}

candidate_ready() {
  local arm="$1"
  local checkpoint="$2"
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$checkpoint" \
    --min-epochs 4 \
    --expected-arch flow_pp3_compact_l \
    --expected-total-steps 6568 \
    --min-global-step 6568 \
    --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256" \
    --expected-highpass-boundary periodic_lon_replicate_lat \
    --expected-lambda-hf 0 \
    "${common_protocol_args[@]}" \
    --quiet &&
    verify_candidate_metadata "$checkpoint" "$arm"
}

acquire_gpu() {
  local candidate
  ACQUIRED_GPU=""
  ACQUIRED_FD=""
  while true; do
    for candidate in $GPU_CANDIDATES; do
      local candidate_fd
      exec {candidate_fd}>"$LOG_ROOT/.flow_pp3_compact_l_gpu${candidate}.lock"
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
  local distill_args=()
  if [[ "$arm" == "distill" ]]; then
    distill_args=(
      --distill_teacher_checkpoint "$TEACHER_CHECKPOINT"
      --distill_weight "${DISTILL_WEIGHTS[$arm]}"
    )
  fi

  exec 8>"$LOG_ROOT/$exp_name.queue.lock"
  if ! flock -n 8; then
    echo "[$(date -Is)] worker already active arm=$arm" | tee -a "$queue_log"
    return 0
  fi
  if candidate_ready "$arm" "$checkpoint"; then
    return 0
  fi
  if [[ -e "$checkpoint" ]]; then
    echo "invalid existing compact PP3 checkpoint: $checkpoint" >&2
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
        --arch flow_pp3_compact_l \
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
        --max_epochs 4 \
        --lr 1e-4 \
        --warmup_steps 500 \
        --window_hours 6 \
        --train_tau_subset 1 3 5 \
        --eval_tau 1 2 3 4 5 \
        --samples_per_date_train 4 \
        --samples_per_date_val 2 \
        --lambda_hf_override 0 \
        --loss_profile uniform \
        --trainable_scope all \
        --train_batches_per_epoch 6568 \
        --ckpt_every_n_epochs 2 \
        "${distill_args[@]}"
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
verify_teacher

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "[$(date -Is)] compact PP3 preflight passed" | tee -a "$QUEUE_LOG"
  trap - ERR
  exit 0
fi

train_arm plain &
plain_pid=$!
train_arm distill &
distill_pid=$!
status=0
wait "$plain_pid" || status=1
wait "$distill_pid" || status=1
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

touch "$COMPLETE_MARKER"
trap - ERR
echo "[$(date -Is)] compact PP3 training complete" | tee -a "$QUEUE_LOG"
