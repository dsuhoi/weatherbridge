#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
# Keep the exact bs=4/accumulate=4 protocol on an otherwise idle 80 GiB GPU.
MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"
MAX_UTIL="${MAX_UTIL:-5}"
GPU_STABLE_SEC="${GPU_STABLE_SEC:-30}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
WAIT_FOR_FLOW_SPHERICAL_TERMINAL="${WAIT_FOR_FLOW_SPHERICAL_TERMINAL:-1}"
QUEUE_LOG="$LOG_ROOT/primary_reference_retrains.queue.log"
PP3_START_MARKER="$LOG_ROOT/primary_reference_pp3_protocol_v2.started"
ATMVFI_START_MARKER="$LOG_ROOT/primary_reference_atmvfi_protocol_v2.started"
PILOT_TERMINAL_MARKER="$LOG_ROOT/exp_flow_spherical_ep_14m_6h_s202707.terminal"
FROZEN_TRAINER="${FROZEN_TRAINER:-legacy/training_snapshots/train_capacity_matched_6h_e8d3bd58.py}"
FROZEN_TRAINER_SHA256="${FROZEN_TRAINER_SHA256:-e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a}"
TRAINER_ENTRYPOINT="${TRAINER_ENTRYPOINT:-tools/train/train_capacity_matched_6h.py}"

mkdir -p "$LOG_ROOT"
actual_trainer_sha256="$(sha256sum "$FROZEN_TRAINER" | awk '{print $1}')"
entrypoint_sha256="$(sha256sum "$TRAINER_ENTRYPOINT" | awk '{print $1}')"
if [[ "$actual_trainer_sha256" != "$FROZEN_TRAINER_SHA256" || \
      "$entrypoint_sha256" != "$FROZEN_TRAINER_SHA256" ]]; then
  echo "frozen trainer SHA-256 mismatch" >&2
  exit 2
fi
exec 9>"$LOG_ROOT/.primary_reference_retrains.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] primary reference queue already active" \
    | tee -a "$QUEUE_LOG"
  exit 0
fi

checkpoint_complete() {
  local checkpoint="$1"
  local arch="$2"
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$checkpoint" \
    --min-epochs 8 \
    --expected-arch "$arch" \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
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
    --expected-samples-per-date-train 4 \
    --expected-samples-per-date-val 2 \
    --expected-lambda-hf 0 \
    --expected-highpass-boundary periodic_lon_replicate_lat \
    --expected-precision bf16-mixed \
    --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256" \
    --quiet
}

run_reference() {
  local name="$1"
  local arch="$2"
  local exp_name="$3"
  local batch_size="$4"
  local val_batch_size="$5"
  local accumulate="$6"
  local wait_marker="${7:-}"
  local checkpoint="$LOG_ROOT/$exp_name/last.ckpt"
  local run_log="$LOG_ROOT/$exp_name.log"
  if [[ "$batch_size" -ne 4 || "$val_batch_size" -ne 2 || \
        "$accumulate" -ne 4 ]]; then
    echo "reference protocol requires bs=4 val_bs=2 accumulate=4" >&2
    return 2
  fi

  if [[ -n "$wait_marker" ]]; then
    while [[ ! -e "$wait_marker" ]]; do
      sleep "$SLEEP_SEC"
    done
  fi

  while ! checkpoint_complete "$checkpoint" "$arch"; do
    local launched=0
    local gpu
    for gpu in $GPU_CANDIDATES; do
      local lock_fd
      exec {lock_fd}>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
      if ! flock -n "$lock_fd"; then
        exec {lock_fd}>&-
        continue
      fi
      local free_mib util
      free_mib="$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
      util="$(nvidia-smi --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
      if [[ "$free_mib" -lt "$MIN_FREE_MIB" || "$util" -gt "$MAX_UTIL" ]]; then
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        continue
      fi
      sleep "$GPU_STABLE_SEC"
      free_mib="$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
      util="$(nvidia-smi --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
      if [[ "$free_mib" -lt "$MIN_FREE_MIB" || "$util" -gt "$MAX_UTIL" ]]; then
        echo "[$(date -Is)] reject unstable gpu=$gpu reference=$name" \
          | tee -a "$QUEUE_LOG"
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        continue
      fi

      launched=1
      local -a resume=()
      if [[ -s "$checkpoint" ]]; then
        resume=(--ckpt_path "$checkpoint")
      fi
      if [[ "$name" == "weatherbridge_pp3" ]]; then
        touch "$PP3_START_MARKER"
      elif [[ "$name" == "atmvfi" ]]; then
        touch "$ATMVFI_START_MARKER"
      fi
      echo "[$(date -Is)] launch reference=$name gpu=$gpu bs=$batch_size accumulate=$accumulate" \
        | tee -a "$QUEUE_LOG"
      local log_start_bytes
      log_start_bytes="$(stat -c %s "$run_log" 2>/dev/null || echo 0)"
      set +e
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
        "$PYTHON_BIN" -u "$TRAINER_ENTRYPOINT" \
          --arch "$arch" \
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
          --years 2014 2015 2016 2017 2018 2019 \
          --val_years 2020 \
          --max_epochs 8 \
          --lr 1e-4 \
          --window_hours 6 \
          --train_tau_subset 1 3 5 \
          --eval_tau 1 2 3 4 5 \
          --samples_per_date_train 4 \
          --samples_per_date_val 2 \
          --ckpt_every_n_epochs 1 \
          "${resume[@]}"
      ) >>"$run_log" 2>&1
      local status=$?
      set -e
      flock -u "$lock_fd"
      exec {lock_fd}>&-
      echo "[$(date -Is)] finish reference=$name status=$status" \
        | tee -a "$QUEUE_LOG"

      if [[ "$status" -ne 0 ]] && \
        tail -c "+$((log_start_bytes + 1))" "$run_log" | \
          grep -Eqi "out of memory|CUDA error: out of memory"; then
        local free_after
        free_after="$(nvidia-smi --query-gpu=memory.free \
          --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
        if [[ "$free_after" -lt "$MIN_FREE_MIB" ]]; then
          echo "[$(date -Is)] external contention reference=$name; retain bs=4 accumulate=4" \
            | tee -a "$QUEUE_LOG"
        else
          echo "[$(date -Is)] intrinsic OOM violates frozen bs=4 reference=$name" \
            | tee -a "$QUEUE_LOG"
          return "$status"
        fi
      elif [[ "$status" -ne 0 ]]; then
        return "$status"
      fi
      if ! checkpoint_complete "$checkpoint" "$arch"; then
        sleep "$SLEEP_SEC"
      fi
      break
    done
    if [[ "$launched" -eq 0 ]]; then
      sleep "$SLEEP_SEC"
    fi
  done
  if [[ "$name" == "weatherbridge_pp3" ]]; then
    touch "$PP3_START_MARKER"
  elif [[ "$name" == "atmvfi" ]]; then
    touch "$ATMVFI_START_MARKER"
  fi
  echo "[$(date -Is)] complete reference=$name checkpoint=$checkpoint" \
    | tee -a "$QUEUE_LOG"
}

case "$WAIT_FOR_FLOW_SPHERICAL_TERMINAL" in
  0)
    echo "[$(date -Is)] strict references run independently of Flow-Spherical" \
      | tee -a "$QUEUE_LOG"
    ;;
  1)
    while [[ ! -e "$PILOT_TERMINAL_MARKER" ]]; do
      sleep "$SLEEP_SEC"
    done
    ;;
  *)
    echo "WAIT_FOR_FLOW_SPHERICAL_TERMINAL must be 0 or 1" >&2
    exit 2
    ;;
esac

run_reference \
  weatherbridge_pp3 \
  flow_pp3 \
  exp_flow_pp3_135_14m_6h_s202707_protocol_v2 \
  4 2 4 &
pp3_pid=$!

run_reference \
  atmvfi \
  atmvfi \
  exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_s202707_protocol_v2 \
  4 2 4 \
  "$PP3_START_MARKER" &
atmvfi_pid=$!

status=0
wait "$pp3_pid" || status=1
wait "$atmvfi_pid" || status=1
if [[ "$status" -eq 0 ]]; then
  echo "[$(date -Is)] primary reference retrains complete" \
    | tee -a "$QUEUE_LOG"
fi
exit "$status"
