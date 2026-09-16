#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"
MAX_UTIL="${MAX_UTIL:-5}"
GPU_STABLE_SEC="${GPU_STABLE_SEC:-30}"
SLEEP_SEC="${SLEEP_SEC:-15}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
FROZEN_TRAINER="${FROZEN_TRAINER:-legacy/training_snapshots/train_capacity_matched_6h_e8d3bd58.py}"
FROZEN_TRAINER_SHA256="${FROZEN_TRAINER_SHA256:-e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a}"
TRAINER_ENTRYPOINT="${TRAINER_ENTRYPOINT:-tools/train/train_capacity_matched_6h.py}"
PP3_START_MARKER="$LOG_ROOT/primary_reference_pp3_protocol_v2.started"
ATMVFI_START_MARKER="$LOG_ROOT/primary_reference_atmvfi_protocol_v2.started"

arch="upr_implicit_global_14m"
exp_name="exp_upr_implicit_global_14m_hf_135_14m_6h_s202707_protocol_v2"
checkpoint="$LOG_ROOT/$exp_name/last.ckpt"
averaged_checkpoint="$LOG_ROOT/$exp_name/avg_last3.ckpt"
run_log="$LOG_ROOT/$exp_name.log"
queue_log="$LOG_ROOT/$exp_name.queue.log"
started_marker="$LOG_ROOT/${exp_name}.started"
terminal_marker="$LOG_ROOT/${exp_name}.terminal"
screened_marker="$LOG_ROOT/${exp_name}.screened"
source tools/train/architecture_screen_barrier.sh
query_name="exp_upr_query_match_14m_hf_135_14m_6h_s202707_protocol_v1"
query_started="$LOG_ROOT/${query_name}.started"
query_terminal="$LOG_ROOT/${query_name}.terminal"

protocol_args=(
  --expected-arch "$arch"
  --expected-total-steps 13136
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

checkpoint_ready() {
  local path="$1"
  local min_epochs="$2"
  local min_step="$3"
  shift 3
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$path" \
    --min-epochs "$min_epochs" \
    --min-global-step "$min_step" \
    "${protocol_args[@]}" \
    "$@" \
    --quiet
}

mkdir -p "$LOG_ROOT"
mark_failed_exit() {
  local status=$?
  if [[ "$status" -ne 0 ]]; then
    touch "$terminal_marker"
  fi
}
trap mark_failed_exit EXIT

actual_trainer_sha256="$(sha256sum "$FROZEN_TRAINER" | awk '{print $1}')"
entrypoint_sha256="$(sha256sum "$TRAINER_ENTRYPOINT" | awk '{print $1}')"
if [[ "$actual_trainer_sha256" != "$FROZEN_TRAINER_SHA256" || \
      "$entrypoint_sha256" != "$FROZEN_TRAINER_SHA256" ]]; then
  echo "frozen trainer SHA-256 mismatch" >&2
  exit 2
fi

exec 9>"$LOG_ROOT/${exp_name}.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] strict UPR+HF reference queue already active" \
    | tee -a "$queue_log"
  exit 0
fi

for marker in "$PP3_START_MARKER" "$ATMVFI_START_MARKER"; do
  while [[ ! -e "$marker" ]]; do
    echo "[$(date -Is)] wait primary reference marker=$marker" \
      | tee -a "$queue_log"
    sleep "$SLEEP_SEC"
  done
done

if checkpoint_ready "$checkpoint" 8 13136 --require-resume-lineage; then
  touch "$screened_marker"
  touch "$started_marker"
  echo "[$(date -Is)] reuse strict UPR+HF reference=$checkpoint" \
    | tee -a "$queue_log"
else
  if [[ -s "$checkpoint" ]] && \
      ! checkpoint_ready "$checkpoint" 1 1; then
    touch "$terminal_marker"
    echo "[$(date -Is)] incompatible partial UPR+HF checkpoint=$checkpoint" \
      | tee -a "$queue_log"
    exit 2
  fi

  acquire_gpu() {
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
        if [[ "$free_mib" -ge "$MIN_FREE_MIB" && \
              "$util" -le "$MAX_UTIL" ]]; then
          sleep "$GPU_STABLE_SEC"
          free_mib="$(nvidia-smi --query-gpu=memory.free \
            --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
          util="$(nvidia-smi --query-gpu=utilization.gpu \
            --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
          if [[ "$free_mib" -ge "$MIN_FREE_MIB" && \
                "$util" -le "$MAX_UTIL" ]]; then
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
  }

  release_gpu() {
    if [[ -n "${lock_fd:-}" ]]; then
      flock -u "$lock_fd"
      exec {lock_fd}>&-
      lock_fd=""
      gpu=""
    fi
  }

  run_stage() {
    local resume=()
    if [[ -s "$checkpoint" ]]; then
      resume=(--ckpt_path "$checkpoint")
    fi
    touch "$started_marker"
    echo "[$(date -Is)] launch strict UPR+HF reference gpu=$gpu" \
      | tee -a "$queue_log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PYTHON_BIN" -u "$TRAINER_ENTRYPOINT" \
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
        --ckpt_every_n_epochs 1 \
        "${resume[@]}"
    ) >>"$run_log" 2>&1
  }

  if ! checkpoint_ready "$checkpoint" 2 3284; then
    acquire_gpu
    log_start_bytes="$(stat -c %s "$run_log" 2>/dev/null || echo 0)"
    set +e
    run_stage &
    screen_pid=$!
    "$PYTHON_BIN" tools/train/stop_after_screen_checkpoint.py \
      --queue-pid "$screen_pid" \
      --experiment "$exp_name" \
      --checkpoint "$checkpoint" \
      --min-epochs 2 \
      --poll-seconds 30 >>"$queue_log" 2>&1
    stop_status=$?
    wait "$screen_pid"
    train_status=$?
    set -e
    if ! checkpoint_ready "$checkpoint" 2 3284; then
      release_gpu
      touch "$terminal_marker"
      if tail -c "+$((log_start_bytes + 1))" "$run_log" | \
        grep -Eqi "out of memory|CUDA error: out of memory"; then
        echo "[$(date -Is)] intrinsic OOM violates frozen bs=4 UPR+HF reference" \
          | tee -a "$queue_log"
      else
        echo "[$(date -Is)] UPR+HF screen failed train=$train_status stop=$stop_status" \
          | tee -a "$queue_log"
      fi
      exit 2
    fi
  fi
  wait_for_architecture_screen_barrier "$exp_name" "$queue_log"
  log_start_bytes="$(stat -c %s "$run_log" 2>/dev/null || echo 0)"
  set +e
  run_stage
  status=$?
  set -e
  release_gpu
  if [[ "$status" -ne 0 ]]; then
    touch "$terminal_marker"
    if tail -c "+$((log_start_bytes + 1))" "$run_log" | \
      grep -Eqi "out of memory|CUDA error: out of memory"; then
      echo "[$(date -Is)] intrinsic OOM violates frozen bs=4 UPR+HF reference" \
        | tee -a "$queue_log"
    else
      echo "[$(date -Is)] strict UPR+HF reference failed status=$status" \
        | tee -a "$queue_log"
    fi
    exit "$status"
  fi
  checkpoint_ready "$checkpoint" 8 13136 --require-resume-lineage
fi

"$PYTHON_BIN" tools/train/average_checkpoints.py \
  --checkpoint-dir "$LOG_ROOT/$exp_name" \
  --last-n 3 \
  --output "$averaged_checkpoint" >>"$queue_log" 2>&1
checkpoint_ready "$averaged_checkpoint" 8 13136
touch "$terminal_marker"
echo "[$(date -Is)] complete strict UPR+HF checkpoint=$checkpoint" \
  | tee -a "$queue_log"
