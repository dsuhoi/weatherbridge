#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
REFERENCE_ARCH="${REFERENCE_ARCH:?set REFERENCE_ARCH=dcae_14m or atmvfi}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
MAX_UTIL="${MAX_UTIL:-5}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
TRAINER_ENTRYPOINT="legacy/training_snapshots/train_capacity_matched_6h_e8d3bd58.py"
TRAINER_SHA256="e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a"
TRAINING_PROTOCOL="tools/train/training_protocol.py"
TRAINING_PROTOCOL_SNAPSHOT="legacy/training_snapshots/training_protocol.py"
TRAINING_PROTOCOL_SHA256="ce3eac6d2f7eb6146c4880ce0e8ac2c201ed9710242003d86f1d724dc63a1e10"
MEMMAP_DATASET="weather_time_interp/memmap_dataset.py"
MEMMAP_DATASET_SHA256="cc5c21f1150e9cdfa9acb1746557db88a080b09da3e7f8caa4207a25d60fc1fa"

case "$REFERENCE_ARCH" in
  dcae_14m)
    exp_name="exp_weatherdcae_14m_24ch_12h_2017_19_lr1e4_protocol_v3"
    ;;
  atmvfi)
    exp_name="exp_atmvfi_24ch_12h_2017_19_oddskip_lr1e4_protocol_v3"
    ;;
  *)
    echo "unsupported 12h reference architecture: $REFERENCE_ARCH" >&2
    exit 2
    ;;
esac

checkpoint="$LOG_ROOT/$exp_name/last.ckpt"
run_log="$LOG_ROOT/$exp_name.log"
queue_log="$LOG_ROOT/$exp_name.queue.log"
batch_size=4
val_batch_size=2
accumulate=4
optimizer_steps_per_epoch=1093
expected_train_batches_per_epoch=$((optimizer_steps_per_epoch * accumulate))
mkdir -p "$LOG_ROOT"

actual_memmap_sha256="$(sha256sum "$MEMMAP_DATASET" | awk '{print $1}')"
if [[ "$actual_memmap_sha256" != "$MEMMAP_DATASET_SHA256" ]]; then
  echo "12h memmap anchor-index SHA-256 mismatch: $actual_memmap_sha256" >&2
  exit 2
fi
actual_trainer_sha256="$(sha256sum "$TRAINER_ENTRYPOINT" | awk '{print $1}')"
if [[ "$actual_trainer_sha256" != "$TRAINER_SHA256" ]]; then
  echo "trainer snapshot SHA-256 mismatch: $actual_trainer_sha256" >&2
  exit 2
fi
for protocol_path in "$TRAINING_PROTOCOL" "$TRAINING_PROTOCOL_SNAPSHOT"; do
  actual_protocol_sha256="$(sha256sum "$protocol_path" | awk '{print $1}')"
  if [[ "$actual_protocol_sha256" != "$TRAINING_PROTOCOL_SHA256" ]]; then
    echo "training protocol SHA-256 mismatch: $protocol_path $actual_protocol_sha256" >&2
    exit 2
  fi
done

protocol_args=(
  --min-epochs 10
  --expected-arch "$REFERENCE_ARCH"
  --expected-total-steps 10930
  --min-global-step 10930
  --expected-delta-t 12
  --require-training-protocol
  --expected-train-years 2017,2018,2019
  --expected-val-years 2020
  --expected-train-taus 1,2,3,5,7,9,10,11
  --expected-eval-taus 1,2,3,4,5,6,7,8,9,10,11
  --expected-seed 202707
  --expected-effective-batch-size 16
  --expected-optimizer-steps-per-epoch "$optimizer_steps_per_epoch"
  --expected-samples-per-date-train 2
  --expected-samples-per-date-val 2
  --expected-lambda-hf 0
  --expected-precision bf16-mixed
  --expected-trainer-sha256 "$TRAINER_SHA256"
)

exec 9>"$LOG_ROOT/${exp_name}.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] queue already active exp=$exp_name" \
    | tee -a "$queue_log"
  exit 0
fi

while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$checkpoint" "${protocol_args[@]}" \
  --expected-train-batches-per-epoch "$expected_train_batches_per_epoch" \
  --quiet; do
  launched=0
  for gpu in $GPU_CANDIDATES; do
    exec {gpu_fd}>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
    if ! flock -n "$gpu_fd"; then
      exec {gpu_fd}>&-
      continue
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ "$free_mib" -lt "$MIN_FREE_MIB" || "$util" -gt "$MAX_UTIL" ]]; then
      flock -u "$gpu_fd"
      exec {gpu_fd}>&-
      continue
    fi

    launched=1
    resume=()
    if [[ -s "$checkpoint" ]]; then
      resume=(--ckpt_path "$checkpoint")
    fi
    echo "[$(date -Is)] launch arch=$REFERENCE_ARCH gpu=$gpu bs=$batch_size accumulate=$accumulate" \
      | tee -a "$queue_log"
    log_start_bytes="$(stat -c %s "$run_log" 2>/dev/null || echo 0)"
    set +e
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
      "$PYTHON_BIN" -u "$TRAINER_ENTRYPOINT" \
        --arch "$REFERENCE_ARCH" \
        --exp_name "$exp_name" \
        --log_root "$LOG_ROOT" \
        --gpus 0 \
        --bs "$batch_size" \
        --val_bs "$val_batch_size" \
        --accumulate "$accumulate" \
        --workers 4 \
        --val_workers 2 \
        --release_memmap_pages \
        --precision bf16-mixed \
        --seed 202707 \
        --years 2017 2018 2019 \
        --val_years 2020 \
        --max_epochs 10 \
        --lr 1e-4 \
        --window_hours 12 \
        --train_tau_subset 1 2 3 5 7 9 10 11 \
        --eval_tau 1 2 3 4 5 6 7 8 9 10 11 \
        --samples_per_date_train 2 \
        --samples_per_date_val 2 \
        --train_batches_per_epoch "$expected_train_batches_per_epoch" \
        --ckpt_every_n_epochs 1 \
        "${resume[@]}"
    ) >>"$run_log" 2>&1
    status=$?
    set -e
    flock -u "$gpu_fd"
    exec {gpu_fd}>&-
    echo "[$(date -Is)] finish arch=$REFERENCE_ARCH status=$status" \
      | tee -a "$queue_log"

    if "$PYTHON_BIN" tools/train/checkpoint_status.py \
      "$checkpoint" "${protocol_args[@]}" \
      --expected-train-batches-per-epoch \
      "$expected_train_batches_per_epoch" \
      --quiet; then
      break
    fi
    if [[ "$status" -ne 0 ]] && \
      tail -c "+$((log_start_bytes + 1))" "$run_log" | \
        grep -Eqi "out of memory|CUDA error: out of memory"; then
      free_after="$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
      if [[ "$free_after" -lt "$MIN_FREE_MIB" ]]; then
        echo "[$(date -Is)] external GPU contention; retain batch recipe" \
          | tee -a "$queue_log"
      elif [[ "$batch_size" -gt 2 ]]; then
        batch_size=2
        val_batch_size=2
        accumulate=8
        expected_train_batches_per_epoch=$((
          optimizer_steps_per_epoch * accumulate
        ))
        echo "[$(date -Is)] intrinsic OOM fallback bs=2 accumulate=8 batches=$expected_train_batches_per_epoch" \
          | tee -a "$queue_log"
      else
        echo "[$(date -Is)] intrinsic OOM at minimum batch" \
          | tee -a "$queue_log"
        exit "$status"
      fi
    elif [[ "$status" -ne 0 ]]; then
      echo "[$(date -Is)] unrecoverable training failure" \
        | tee -a "$queue_log"
      exit "$status"
    fi
    sleep "$SLEEP_SEC"
    break
  done
  if [[ "$launched" -eq 0 ]]; then
    sleep "$SLEEP_SEC"
  fi
done

echo "[$(date -Is)] complete checkpoint=$checkpoint" | tee -a "$queue_log"
