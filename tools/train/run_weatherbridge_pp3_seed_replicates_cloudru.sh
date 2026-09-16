#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
SELECTION="${SELECTION:-metrics/upr_lite_candidate_selection.json}"
UPR_SEED_REPORT="${UPR_SEED_REPORT:-metrics/upr_lite_seed_robustness.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"
MAX_UTIL="${MAX_UTIL:-5}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
FROZEN_TRAINER="${FROZEN_TRAINER:-legacy/training_snapshots/train_capacity_matched_6h_e8d3bd58.py}"
FROZEN_TRAINER_SHA256="${FROZEN_TRAINER_SHA256:-e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a}"
TRAINER_ENTRYPOINT="${TRAINER_ENTRYPOINT:-tools/train/train_capacity_matched_6h.py}"
LOG="$LOG_ROOT/weatherbridge_pp3_seed_replicates.queue.log"

mkdir -p "$LOG_ROOT"
actual_trainer_sha256="$(sha256sum "$FROZEN_TRAINER" | awk '{print $1}')"
if [[ "$actual_trainer_sha256" != "$FROZEN_TRAINER_SHA256" ]]; then
  echo "frozen trainer SHA-256 mismatch: $actual_trainer_sha256" >&2
  exit 2
fi
entrypoint_sha256="$(sha256sum "$TRAINER_ENTRYPOINT" | awk '{print $1}')"
if [[ "$entrypoint_sha256" != "$FROZEN_TRAINER_SHA256" ]]; then
  echo "trainer entrypoint SHA-256 mismatch: $entrypoint_sha256" >&2
  exit 2
fi
while [[ ! -s "$SELECTION" ]]; do
  echo "[$(date -Is)] wait selection=$SELECTION" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
selection_mtime="$(stat -c %Y "$SELECTION")"
while [[ ! -s "$UPR_SEED_REPORT" ]] || \
  [[ "$(stat -c %Y "$UPR_SEED_REPORT")" -le "$selection_mtime" ]]; do
  echo "[$(date -Is)] wait UPR seed report=$UPR_SEED_REPORT" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done

run_seed() {
  local seed="$1"
  local exp_name="exp_flow_pp3_135_14m_6h_s${seed}"
  local checkpoint="$LOG_ROOT/$exp_name/last.ckpt"
  local run_log="$LOG_ROOT/$exp_name.log"
  local batch_size=4
  local val_batch_size=2
  local accumulate=4

  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$checkpoint" \
    --min-epochs 8 \
    --expected-arch flow_pp3 \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    --require-training-protocol \
    --expected-train-years 2014,2015,2016,2017,2018,2019 \
    --expected-val-years 2020 \
    --expected-train-taus 1,3,5 \
    --expected-eval-taus 1,2,3,4,5 \
    --expected-seed "$seed" \
    --expected-effective-batch-size 16 \
    --expected-batch-size-per-device 4 \
    --expected-accumulate-grad-batches 4 \
    --expected-samples-per-date-train 4 \
    --expected-samples-per-date-val 2 \
    --expected-lambda-hf 0 \
    --expected-highpass-boundary periodic_lon_replicate_lat \
    --expected-precision bf16-mixed \
    --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256" \
    --quiet; do
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
      launched=1
      local -a resume=()
      if [[ -s "$checkpoint" ]]; then
        resume=(--ckpt_path "$checkpoint")
      fi
      echo "[$(date -Is)] launch PP3 seed=$seed gpu=$gpu bs=$batch_size accumulate=$accumulate" \
        | tee -a "$LOG"
      local log_start_bytes
      log_start_bytes="$(stat -c %s "$run_log" 2>/dev/null || echo 0)"
      set +e
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
        "$PYTHON_BIN" -u "$TRAINER_ENTRYPOINT" \
          --arch flow_pp3 \
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
          --seed "$seed" \
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
      echo "[$(date -Is)] finish PP3 seed=$seed status=$status" | tee -a "$LOG"
      if [[ "$status" -ne 0 ]] && \
        tail -c "+$((log_start_bytes + 1))" "$run_log" | \
          grep -Eqi "out of memory|CUDA error: out of memory"; then
        free_after="$(nvidia-smi --query-gpu=memory.free \
          --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
        if [[ "$free_after" -lt "$MIN_FREE_MIB" ]]; then
          echo "[$(date -Is)] external GPU contention after OOM seed=$seed; keep bs=$batch_size accumulate=$accumulate" \
            | tee -a "$LOG"
        else
          echo "[$(date -Is)] intrinsic OOM violates frozen bs=4 protocol seed=$seed" \
            | tee -a "$LOG"
          return "$status"
        fi
      elif [[ "$status" -ne 0 ]]; then
        echo "[$(date -Is)] PP3 seed=$seed failed status=$status" \
          | tee -a "$LOG"
        return "$status"
      fi
      if ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
        "$checkpoint" \
        --min-epochs 8 \
        --expected-arch flow_pp3 \
        --expected-total-steps 13136 \
        --min-global-step 13136 \
        --expected-delta-t 6 \
        --require-training-protocol \
        --expected-train-years 2014,2015,2016,2017,2018,2019 \
        --expected-val-years 2020 \
        --expected-train-taus 1,3,5 \
        --expected-eval-taus 1,2,3,4,5 \
        --expected-seed "$seed" \
        --expected-effective-batch-size 16 \
        --expected-batch-size-per-device 4 \
        --expected-accumulate-grad-batches 4 \
        --expected-samples-per-date-train 4 \
        --expected-samples-per-date-val 2 \
        --expected-lambda-hf 0 \
        --expected-highpass-boundary periodic_lon_replicate_lat \
        --expected-precision bf16-mixed \
        --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256" \
        --quiet; then
        sleep "$SLEEP_SEC"
      fi
      break
    done
    if [[ "$launched" -eq 0 ]]; then
      sleep "$SLEEP_SEC"
    fi
  done
}

run_seed 202708 &
pid_a=$!
run_seed 202709 &
pid_b=$!
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
if [[ "$status" -eq 0 ]]; then
  echo "[$(date -Is)] PP3 seed controls complete" | tee -a "$LOG"
fi
exit "$status"
