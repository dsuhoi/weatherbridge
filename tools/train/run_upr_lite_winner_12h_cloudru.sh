#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source tools/train/architecture_candidate_profile.sh

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
SELECTION="${SELECTION:-metrics/upr_lite_candidate_selection.json}"
SCREEN_JSON="${SCREEN_JSON:-metrics/upr_lite_screen/two_epoch_screen.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-55000}"
MAX_UTIL="${MAX_UTIL:-5}"
GPU_STABLE_SEC="${GPU_STABLE_SEC:-30}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
TRANSFER_OBJECTIVE="${TRANSFER_OBJECTIVE:-quality}"
BASE_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_e8d3bd58.py"
BASE_TRAINER_SHA256="e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a"
QUERY_MATCH_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_32f4ce54.py"
QUERY_MATCH_TRAINER_SHA256="32f4ce54ddd3d033ea4ea55112d6e65658caf73961e568680aa627e8794e1f30"
MEMMAP_DATASET="weather_time_interp/memmap_dataset.py"
MEMMAP_DATASET_SHA256="cc5c21f1150e9cdfa9acb1746557db88a080b09da3e7f8caa4207a25d60fc1fa"
LOG="$LOG_ROOT/upr_lite_${TRANSFER_OBJECTIVE}_winner_12h.queue.log"

mkdir -p "$LOG_ROOT"

actual_memmap_sha256="$(sha256sum "$MEMMAP_DATASET" | awk '{print $1}')"
if [[ "$actual_memmap_sha256" != "$MEMMAP_DATASET_SHA256" ]]; then
  echo "12h memmap tau-normalization SHA-256 mismatch: $actual_memmap_sha256" >&2
  exit 2
fi

while [[ ! -s "$SCREEN_JSON" ]] || [[ "$("$PYTHON_BIN" -c '
import json, sys
print(int(json.load(open(sys.argv[1]))["complete"]))
' "$SCREEN_JSON")" -ne 1 ]]; do
  echo "[$(date -Is)] wait complete screen=$SCREEN_JSON" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
screen_mtime="$(stat -c %Y "$SCREEN_JSON")"
while [[ ! -s "$SELECTION" ]] || [[ "$(stat -c %Y "$SELECTION")" -le "$screen_mtime" ]]; do
  echo "[$(date -Is)] wait fresh selection=$SELECTION" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done

candidate="$("$PYTHON_BIN" -c '
from pathlib import Path
import json
from tools.eval.summarize_upr_lite_seeds import (
    choose_transfer_candidate,
    validate_frozen_selection_for_followup,
)
selection = json.loads(Path("'"$SELECTION"'").read_text())
validate_frozen_selection_for_followup(selection)
objective = "'"$TRANSFER_OBJECTIVE"'"
if objective == "flow":
    models = selection.get("models", {})
    print(next(
        (
            name
            for name in ("flow_spherical_ep", "flow_pp3_hf")
            if name in models
        ),
        "",
    ))
else:
    print(choose_transfer_candidate(selection, objective))
')"
if [[ -z "$candidate" ]]; then
  echo "[$(date -Is)] no promoted Flow control in frozen selection" \
    | tee -a "$LOG"
  exit 0
fi
TRAINER_ENTRYPOINT="$(architecture_candidate_trainer "$candidate")"
TRAINER_SHA256="$(architecture_candidate_trainer_sha256 "$candidate")"
actual_trainer_sha256="$(sha256sum "$TRAINER_ENTRYPOINT" | awk '{print $1}')"
if [[ "$actual_trainer_sha256" != "$TRAINER_SHA256" ]]; then
  echo "trainer snapshot SHA-256 mismatch: $actual_trainer_sha256" >&2
  exit 2
fi
model_arch="$(architecture_candidate_model_arch "$candidate")"
batch_size="$(architecture_candidate_batch_size "$candidate")"
val_batch_size="$(architecture_candidate_val_batch_size "$candidate")"
accumulate="$(architecture_candidate_accumulate "$candidate")"
optimizer_steps_per_epoch=1093
layout="$(architecture_candidate_layout "$candidate")"
expected_lambda_hf="$(architecture_candidate_lambda_hf "$candidate")"
expected_highpass_boundary="$(
  architecture_candidate_highpass_boundary "$candidate"
)"
loss_args=()
if [[ "$expected_lambda_hf" == "0" ]]; then
  loss_args=(--lambda_hf_override 0)
fi
if [[ "$candidate" == "amt" || "$candidate" == "amt_residual" || \
      "$candidate" == "flow_pp3_hf" ]]; then
  loss_args=(--lambda_hf_override 0.05)
fi
expected_train_batches_per_epoch=$((optimizer_steps_per_epoch * accumulate))
exp_name="exp_${candidate}_24ch_12h_2017_19_held468_lr1e4_sp2_${layout}_s202707"
checkpoint="$LOG_ROOT/$exp_name/last.ckpt"
run_log="$LOG_ROOT/$exp_name.log"
exec 8>"$LOG_ROOT/${exp_name}.transfer_train.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] transfer training already active candidate=$candidate" \
    | tee -a "$LOG"
  exit 0
fi
if "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$checkpoint" \
  --min-epochs 10 \
  --expected-arch "$model_arch" \
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
  --expected-lambda-hf "$expected_lambda_hf" \
  --expected-highpass-boundary "$expected_highpass_boundary" \
  --expected-precision bf16-mixed \
  --expected-trainer-sha256 "$TRAINER_SHA256" \
  --quiet; then
  echo "[$(date -Is)] skip existing candidate=$candidate checkpoint=$checkpoint" | tee -a "$LOG"
  exit 0
fi

while true; do
  for gpu in $GPU_CANDIDATES; do
    exec {lock_fd}>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
    if ! flock -n "$lock_fd"; then
      exec {lock_fd}>&-
      continue
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ "$free_mib" -lt "$MIN_FREE_MIB" || "$util" -gt "$MAX_UTIL" ]]; then
      flock -u "$lock_fd"
      exec {lock_fd}>&-
      continue
    fi
    sleep "$GPU_STABLE_SEC"
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ "$free_mib" -lt "$MIN_FREE_MIB" || "$util" -gt "$MAX_UTIL" ]]; then
      echo "[$(date -Is)] reject unstable gpu=$gpu objective=$TRANSFER_OBJECTIVE" \
        | tee -a "$LOG"
      flock -u "$lock_fd"
      exec {lock_fd}>&-
      continue
    fi
    echo "[$(date -Is)] launch objective=$TRANSFER_OBJECTIVE candidate=$candidate gpu=$gpu exp=$exp_name" | tee -a "$LOG"
    resume=()
    if [[ -s "$checkpoint" ]]; then
      resume=(--ckpt_path "$checkpoint")
      echo "[$(date -Is)] resume checkpoint=$checkpoint" | tee -a "$LOG"
    fi
    log_start_bytes="$(stat -c %s "$run_log" 2>/dev/null || echo 0)"
    set +e
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
      "$PYTHON_BIN" -u "$TRAINER_ENTRYPOINT" \
        --arch "$model_arch" \
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
        "${loss_args[@]}" \
        --ckpt_every_n_epochs 1 \
        "${resume[@]}"
    ) >>"$run_log" 2>&1
    status=$?
    set -e
    flock -u "$lock_fd"
    exec {lock_fd}>&-
    echo "[$(date -Is)] finish candidate=$candidate status=$status" | tee -a "$LOG"
    if "$PYTHON_BIN" tools/train/checkpoint_status.py \
      "$checkpoint" \
      --min-epochs 10 \
      --expected-arch "$model_arch" \
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
      --expected-lambda-hf "$expected_lambda_hf" \
      --expected-highpass-boundary "$expected_highpass_boundary" \
      --expected-precision bf16-mixed \
      --expected-trainer-sha256 "$TRAINER_SHA256" \
      --quiet; then
      exit 0
    fi
    if [[ "$status" -ne 0 ]] && \
      tail -c "+$((log_start_bytes + 1))" "$run_log" | \
        grep -Eqi "out of memory|CUDA error: out of memory"; then
      free_after="$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
      if [[ "$free_after" -lt "$MIN_FREE_MIB" ]]; then
        echo "[$(date -Is)] external GPU contention after OOM; keep bs=$batch_size accumulate=$accumulate" \
          | tee -a "$LOG"
      elif [[ "$batch_size" -gt 2 ]]; then
        batch_size=$((batch_size / 2))
        val_batch_size=$((val_batch_size > 2 ? 2 : val_batch_size))
        accumulate=$((accumulate * 2))
        expected_train_batches_per_epoch=$((optimizer_steps_per_epoch * accumulate))
        echo "[$(date -Is)] intrinsic OOM fallback bs=$batch_size accumulate=$accumulate" \
          | tee -a "$LOG"
      else
        echo "[$(date -Is)] intrinsic OOM at minimum batch" | tee -a "$LOG"
        exit "$status"
      fi
    elif [[ "$status" -ne 0 ]]; then
      echo "[$(date -Is)] training failed without recoverable OOM" | tee -a "$LOG"
      exit "$status"
    fi
    sleep "$SLEEP_SEC"
    break
  done
  sleep "$SLEEP_SEC"
done
