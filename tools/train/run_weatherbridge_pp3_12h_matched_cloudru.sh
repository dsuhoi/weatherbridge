#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
SELECTION="${SELECTION:-metrics/upr_lite_candidate_selection.json}"
SCREEN_JSON="${SCREEN_JSON:-metrics/upr_lite_screen/two_epoch_screen.json}"
UPR_WORKER_MARKER="${UPR_WORKER_MARKER:-$LOG_ROOT/upr_14m_workers2_resume.ready}"
MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"
MAX_UTIL="${MAX_UTIL:-5}"
GPU_STABLE_SEC="${GPU_STABLE_SEC:-30}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
TRAINER_ENTRYPOINT="legacy/training_snapshots/train_capacity_matched_6h_e8d3bd58.py"
TRAINER_SHA256="e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a"
MEMMAP_DATASET="weather_time_interp/memmap_dataset.py"
MEMMAP_DATASET_SHA256="cc5c21f1150e9cdfa9acb1746557db88a080b09da3e7f8caa4207a25d60fc1fa"

EXP_NAME="exp_flow_pp3_12h_2017_19_held468_lr1e4_sp2_protocol_v3"
CHECKPOINT="$LOG_ROOT/$EXP_NAME/last.ckpt"
RUN_LOG="$LOG_ROOT/$EXP_NAME.log"
QUEUE_LOG="$LOG_ROOT/$EXP_NAME.queue.log"
QUEUE_LOCK="$LOG_ROOT/$EXP_NAME.queue.lock"
batch_size=4
val_batch_size=2
accumulate=4
optimizer_steps_per_epoch=1093
expected_train_batches_per_epoch=$((optimizer_steps_per_epoch * accumulate))
mkdir -p "$LOG_ROOT"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

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

exec 8>"$QUEUE_LOCK"
if ! flock -n 8; then
  echo "[$(date -Is)] matched PP3 12h queue already active" \
    | tee -a "$QUEUE_LOG"
  exit 0
fi

while [[ ! -s "$SCREEN_JSON" ]] || [[ "$("$PYTHON_BIN" -c '
import json, sys
print(int(json.load(open(sys.argv[1]))["complete"]))
' "$SCREEN_JSON")" -ne 1 ]]; do
  echo "[$(date -Is)] wait complete screen=$SCREEN_JSON" | tee -a "$QUEUE_LOG"
  sleep "$SLEEP_SEC"
done
screen_mtime="$(stat -c %Y "$SCREEN_JSON")"
while [[ ! -s "$SELECTION" ]] || [[ "$(stat -c %Y "$SELECTION")" -le "$screen_mtime" ]]; do
  echo "[$(date -Is)] wait fresh selection=$SELECTION" | tee -a "$QUEUE_LOG"
  sleep "$SLEEP_SEC"
done
"$PYTHON_BIN" -c '
from pathlib import Path
import json, sys
from tools.eval.summarize_upr_lite_seeds import (
    validate_frozen_selection_for_followup,
)
selection = json.loads(Path(sys.argv[1]).read_text())
validate_frozen_selection_for_followup(selection)
' "$SELECTION"
while [[ ! -e "$UPR_WORKER_MARKER" ]]; do
  echo "[$(date -Is)] wait UPR workers=2 handoff=$UPR_WORKER_MARKER" \
    | tee -a "$QUEUE_LOG"
  sleep "$SLEEP_SEC"
done
while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$CHECKPOINT" \
  --min-epochs 10 \
  --expected-arch flow_pp3 \
  --expected-total-steps 10930 \
  --min-global-step 10930 \
  --expected-delta-t 12 \
  --require-training-protocol \
  --expected-train-years 2017,2018,2019 \
  --expected-val-years 2020 \
  --expected-train-taus 1,2,3,5,7,9,10,11 \
  --expected-eval-taus 1,2,3,4,5,6,7,8,9,10,11 \
  --expected-seed 202707 \
  --expected-effective-batch-size 16 \
  --expected-train-batches-per-epoch "$expected_train_batches_per_epoch" \
  --expected-optimizer-steps-per-epoch "$optimizer_steps_per_epoch" \
  --expected-samples-per-date-train 2 \
  --expected-samples-per-date-val 2 \
  --expected-lambda-hf 0 \
  --expected-precision bf16-mixed \
  --expected-trainer-sha256 "$TRAINER_SHA256" \
  --quiet; do
  launched=0
  for gpu in $GPU_CANDIDATES; do
    exec {gpu_fd}>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
    if ! flock -n "$gpu_fd"; then
      exec {gpu_fd}>&-
      continue
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ "$free_mib" -lt "$MIN_FREE_MIB" || "$util" -gt "$MAX_UTIL" ]]; then
      flock -u "$gpu_fd"
      exec {gpu_fd}>&-
      continue
    fi
    sleep "$GPU_STABLE_SEC"
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ "$free_mib" -lt "$MIN_FREE_MIB" || "$util" -gt "$MAX_UTIL" ]]; then
      echo "[$(date -Is)] reject unstable gpu=$gpu free_mib=$free_mib util=$util" \
        | tee -a "$QUEUE_LOG"
      flock -u "$gpu_fd"
      exec {gpu_fd}>&-
      continue
    fi
    launched=1
    resume=()
    if [[ -s "$CHECKPOINT" ]]; then
      resume=(--ckpt_path "$CHECKPOINT")
    fi
    echo "[$(date -Is)] launch matched PP3 12h gpu=$gpu bs=$batch_size accumulate=$accumulate" \
      | tee -a "$QUEUE_LOG"
    log_start_bytes="$(stat -c %s "$RUN_LOG" 2>/dev/null || echo 0)"
    set +e
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
      "$PYTHON_BIN" -u "$TRAINER_ENTRYPOINT" \
        --arch flow_pp3 \
        --exp_name "$EXP_NAME" \
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
    ) >>"$RUN_LOG" 2>&1
    status=$?
    set -e
    flock -u "$gpu_fd"
    exec {gpu_fd}>&-
    echo "[$(date -Is)] finish matched PP3 12h status=$status" \
      | tee -a "$QUEUE_LOG"
    if ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
      "$CHECKPOINT" \
      --min-epochs 10 \
      --expected-arch flow_pp3 \
      --expected-total-steps 10930 \
      --min-global-step 10930 \
      --expected-delta-t 12 \
      --require-training-protocol \
      --expected-train-years 2017,2018,2019 \
      --expected-val-years 2020 \
      --expected-train-taus 1,2,3,5,7,9,10,11 \
      --expected-eval-taus 1,2,3,4,5,6,7,8,9,10,11 \
      --expected-seed 202707 \
      --expected-effective-batch-size 16 \
      --expected-train-batches-per-epoch "$expected_train_batches_per_epoch" \
      --expected-optimizer-steps-per-epoch "$optimizer_steps_per_epoch" \
      --expected-samples-per-date-train 2 \
      --expected-samples-per-date-val 2 \
      --expected-lambda-hf 0 \
      --expected-precision bf16-mixed \
      --expected-trainer-sha256 "$TRAINER_SHA256" \
      --quiet; then
      if [[ "$status" -ne 0 ]] && \
        tail -c "+$((log_start_bytes + 1))" "$RUN_LOG" | \
          grep -Eqi "out of memory|CUDA error: out of memory"; then
        free_after="$(nvidia-smi --query-gpu=memory.free \
          --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
        if [[ "$free_after" -lt "$MIN_FREE_MIB" ]]; then
          echo "[$(date -Is)] external GPU contention after OOM; keep bs=$batch_size accumulate=$accumulate" \
            | tee -a "$QUEUE_LOG"
        elif [[ "$batch_size" -gt 2 ]]; then
          batch_size=2
          val_batch_size=2
          accumulate=8
          expected_train_batches_per_epoch=$((
            optimizer_steps_per_epoch * accumulate
          ))
          echo "[$(date -Is)] intrinsic OOM fallback bs=2 accumulate=8 batches=$expected_train_batches_per_epoch" \
            | tee -a "$QUEUE_LOG"
        else
          echo "[$(date -Is)] intrinsic OOM at minimum batch" \
            | tee -a "$QUEUE_LOG"
          exit "$status"
        fi
      elif [[ "$status" -ne 0 ]]; then
        echo "[$(date -Is)] matched PP3 12h failed without recoverable OOM" \
          | tee -a "$QUEUE_LOG"
        exit "$status"
      fi
      sleep "$SLEEP_SEC"
    fi
    break
  done
  if [[ "$launched" -eq 0 ]]; then
    sleep "$SLEEP_SEC"
  fi
done

echo "[$(date -Is)] complete checkpoint=$CHECKPOINT" | tee -a "$QUEUE_LOG"
