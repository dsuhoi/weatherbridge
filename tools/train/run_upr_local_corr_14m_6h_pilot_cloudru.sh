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
FROZEN_TRAINER="${FROZEN_TRAINER:-legacy/training_snapshots/train_capacity_matched_6h_d7a6bb0d.py}"
FROZEN_TRAINER_SHA256="${FROZEN_TRAINER_SHA256:-d7a6bb0d0b801eadb3365bf7b84803c6f499f7ff7969dc5d8a8de9895353e733}"
TRAINER_ENTRYPOINT="${TRAINER_ENTRYPOINT:-$FROZEN_TRAINER}"
UPR_LITE_SOURCE="weather_time_interp/model/weatherbridge_upr_lite_model.py"
UPR_LITE_SOURCE_SHA256="b52cdc0407c4e5efce70af7ada523fa2706940519c68ea1b01d1a8bd7ccaf587"
UPR_SCALED_SOURCE="weather_time_interp/model/weatherbridge_upr_scaled_model.py"
UPR_SCALED_SOURCE_SHA256="a36c3092943fe6021d9ea0a710e73e62067db0ab24aaf1b55efb03e8c8333cc7"
REFERENCE_TRAINER_SHA256="${REFERENCE_TRAINER_SHA256:-e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a}"
PP3_START_MARKER="$LOG_ROOT/primary_reference_pp3_protocol_v2.started"
ATMVFI_START_MARKER="$LOG_ROOT/primary_reference_atmvfi_protocol_v2.started"

arch="upr_local_corr_14m"
exp_name="exp_upr_local_corr_14m_hf_135_14m_6h_s202707_protocol_v1"
checkpoint="$LOG_ROOT/$exp_name/last.ckpt"
two_epoch_checkpoint="$LOG_ROOT/$exp_name/epoch=1-step=3284.ckpt"
run_log="$LOG_ROOT/$exp_name.log"
queue_log="$LOG_ROOT/$exp_name.queue.log"
assessment="metrics/upr_lite_screen/upr_local_corr_14m_pilot.json"
terminal_marker="$LOG_ROOT/${exp_name}.terminal"
started_marker="$LOG_ROOT/${exp_name}.started"
screened_marker="$LOG_ROOT/${exp_name}.screened"
source tools/train/architecture_screen_barrier.sh

reference_arch="upr_implicit_global_14m"
reference_name="exp_upr_implicit_global_14m_hf_135_14m_6h_s202707_protocol_v2"
reference_checkpoint="$LOG_ROOT/$reference_name/last.ckpt"
reference_two_epoch="$LOG_ROOT/$reference_name/epoch=1-step=3284.ckpt"
reference_terminal="$LOG_ROOT/${reference_name}.terminal"

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
  --expected-samples-per-date-train 4
  --expected-samples-per-date-val 2
  --expected-lambda-hf 0.05
  --expected-highpass-boundary periodic_lon_replicate_lat
  --expected-precision bf16-mixed
)
candidate_protocol_args=(
  "${common_protocol_args[@]}"
  --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256"
)
reference_protocol_args=(
  "${common_protocol_args[@]}"
  --expected-trainer-sha256 "$REFERENCE_TRAINER_SHA256"
)

candidate_ready() {
  local path="$1"
  local min_epochs="$2"
  local min_step="$3"
  shift 3
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$path" \
    --min-epochs "$min_epochs" \
    --expected-arch "$arch" \
    --expected-total-steps 13136 \
    --min-global-step "$min_step" \
    "${candidate_protocol_args[@]}" \
    "$@" \
    --quiet
}

reference_ready() {
  local path="$1"
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$path" \
    --min-epochs 2 \
    --expected-arch "$reference_arch" \
    --expected-total-steps 13136 \
    --min-global-step 3284 \
    "${reference_protocol_args[@]}" \
    --quiet
}

mkdir -p "$LOG_ROOT"
mark_failed_exit() {
  local status=$?
  trap - EXIT
  if [[ "$status" -ne 0 ]]; then
    touch "$terminal_marker"
  fi
  exit "$status"
}
trap mark_failed_exit EXIT

actual_trainer_sha256="$(sha256sum "$FROZEN_TRAINER" | awk '{print $1}')"
entrypoint_sha256="$(sha256sum "$TRAINER_ENTRYPOINT" | awk '{print $1}')"
upr_lite_sha256="$(sha256sum "$UPR_LITE_SOURCE" | awk '{print $1}')"
upr_scaled_sha256="$(sha256sum "$UPR_SCALED_SOURCE" | awk '{print $1}')"
if [[ "$actual_trainer_sha256" != "$FROZEN_TRAINER_SHA256" || \
      "$entrypoint_sha256" != "$FROZEN_TRAINER_SHA256" || \
      "$upr_lite_sha256" != "$UPR_LITE_SOURCE_SHA256" || \
      "$upr_scaled_sha256" != "$UPR_SCALED_SOURCE_SHA256" ]]; then
  echo "frozen LocalCorr source SHA-256 mismatch" >&2
  exit 2
fi

exec 9>"$LOG_ROOT/${exp_name}.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] LocalCorr pilot queue already active" \
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

if candidate_ready \
  "$checkpoint" 8 13136 --require-resume-lineage; then
  touch "$screened_marker"
  touch "$terminal_marker"
  echo "[$(date -Is)] reuse complete LocalCorr checkpoint=$checkpoint" \
    | tee -a "$queue_log"
  exit 0
fi
if [[ -e "$terminal_marker" && -s "$assessment" ]]; then
  promoted="$("$PYTHON_BIN" tools/eval/pilot_assessment_status.py \
    "$assessment" \
    --expected-candidate "$arch" \
    --expected-reference "$reference_arch" \
    --print-promoted)"
  if [[ "$promoted" -ne 1 ]]; then
    touch "$screened_marker"
    echo "[$(date -Is)] reuse excluded LocalCorr pilot" \
      | tee -a "$queue_log"
    exit 0
  fi
fi

# The model source extends the UPR implementation used by active candidates.
# Wait for their terminal markers before this checkout may be synchronized.
wait_for_architecture_completion "$queue_log"

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
    if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
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

run_stage() {
  local resume=()
  if [[ -s "$checkpoint" ]]; then
    resume=(--ckpt_path "$checkpoint")
  fi
  touch "$started_marker"
  echo "[$(date -Is)] launch LocalCorr epochs=8 gpu=$gpu" \
    | tee -a "$queue_log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
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

while ! candidate_ready "$checkpoint" 2 3284; do
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
  if candidate_ready "$checkpoint" 2 3284; then
    echo "[$(date -Is)] complete LocalCorr two-epoch screen" \
      | tee -a "$queue_log"
    continue
  fi
  if tail -c "+$((log_start_bytes + 1))" "$run_log" | \
    grep -Eqi "out of memory|CUDA error: out of memory"; then
    echo "[$(date -Is)] intrinsic OOM violates frozen bs=4 pilot" \
      | tee -a "$queue_log"
    flock -u "$lock_fd"
    exit 2
  fi
  echo "[$(date -Is)] pilot failed train=$train_status stop=$stop_status; retry" \
    | tee -a "$queue_log"
  sleep "$SLEEP_SEC"
done

while ! reference_ready "$reference_checkpoint"; do
  if [[ -e "$reference_terminal" ]]; then
    echo "[$(date -Is)] strict UPR reference failed" \
      | tee -a "$queue_log"
    exit 2
  fi
  echo "[$(date -Is)] wait two-epoch strict UPR reference for assessment" \
    | tee -a "$queue_log"
  sleep "$SLEEP_SEC"
done

candidate_ready "$two_epoch_checkpoint" 2 3284
reference_ready "$reference_two_epoch"
candidate_csv="$("$PYTHON_BIN" tools/train/select_validation_metrics_csv.py \
  "$LOG_ROOT/$exp_name" --required-epoch 1 --required-step 3283)"
reference_csv="$("$PYTHON_BIN" tools/train/select_validation_metrics_csv.py \
  "$LOG_ROOT/$reference_name" --required-epoch 1 --required-step 3283)"

"$PYTHON_BIN" tools/eval/assess_two_epoch_candidate.py \
  --pilot-csv "$candidate_csv" \
  --reference-csv "$reference_csv" \
  --candidate-name "$arch" \
  --reference-name "$reference_arch" \
  --held-relative-limit 0.02 \
  --all-hour-relative-limit 0.02 \
  --per-hour-relative-limit 0.05 \
  --output "$assessment" >>"$queue_log" 2>&1
promoted="$("$PYTHON_BIN" tools/eval/pilot_assessment_status.py \
  "$assessment" \
  --expected-candidate "$arch" \
  --expected-reference "$reference_arch" \
  --print-promoted)"
touch "$screened_marker"
if [[ "$promoted" -ne 1 ]]; then
  touch "$terminal_marker"
  echo "[$(date -Is)] exclude LocalCorr at two-epoch gate" \
    | tee -a "$queue_log"
  flock -u "$lock_fd"
  exit 0
fi

echo "[$(date -Is)] promote LocalCorr to eight epochs" \
  | tee -a "$queue_log"
while ! candidate_ready \
  "$checkpoint" 8 13136 --require-resume-lineage; do
  log_start_bytes="$(stat -c %s "$run_log" 2>/dev/null || echo 0)"
  if ! run_stage; then
    if tail -c "+$((log_start_bytes + 1))" "$run_log" | \
      grep -Eqi "out of memory|CUDA error: out of memory"; then
      echo "[$(date -Is)] intrinsic OOM violates frozen bs=4 promoted" \
        | tee -a "$queue_log"
      flock -u "$lock_fd"
      exit 2
    fi
    echo "[$(date -Is)] promoted stage failed; retry" \
      | tee -a "$queue_log"
    sleep "$SLEEP_SEC"
  fi
done

touch "$terminal_marker"
flock -u "$lock_fd"
echo "[$(date -Is)] complete LocalCorr checkpoint=$checkpoint" \
  | tee -a "$queue_log"
