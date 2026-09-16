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
model_name="upr_implicit_global_14m_nohf"
exp_name="exp_upr_implicit_global_14m_nohf_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707"
checkpoint="$LOG_ROOT/$exp_name/last.ckpt"
run_log="$LOG_ROOT/$exp_name.log"
queue_log="$LOG_ROOT/$exp_name.queue.log"
assessment="metrics/upr_lite_screen/upr_implicit_global_14m_nohf_pilot.json"
ablation="metrics/upr_lite_screen/upr_implicit_global_14m_hf_vs_nohf_2ep.json"
terminal_marker="$LOG_ROOT/${exp_name}.terminal"
started_marker="$LOG_ROOT/${exp_name}.started"
screened_marker="$LOG_ROOT/${exp_name}.screened"
candidate_two_epoch="$LOG_ROOT/$exp_name/epoch=1-step=3284.ckpt"
source tools/train/architecture_screen_barrier.sh

upr_name="exp_upr_implicit_global_14m_hf_135_14m_6h_s202707_protocol_v2"
upr_two_epoch="$LOG_ROOT/$upr_name/epoch=1-step=3284.ckpt"
upr_terminal="$LOG_ROOT/${upr_name}.terminal"
flow_name="exp_flow_pp3_135_14m_6h_s202707_protocol_v2"
flow_two_epoch="$LOG_ROOT/$flow_name/epoch=1-step=3284.ckpt"
query_name="exp_upr_query_match_14m_hf_135_14m_6h_s202707_protocol_v1"
query_started="$LOG_ROOT/${query_name}.started"
query_terminal="$LOG_ROOT/${query_name}.terminal"

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
  --expected-highpass-boundary periodic_lon_replicate_lat
  --expected-precision bf16-mixed
  --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256"
)

checkpoint_ready() {
  local path="$1"
  local expected_arch="$2"
  local min_epochs="$3"
  local min_step="$4"
  local lambda_hf="$5"
  shift 5
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$path" \
    --min-epochs "$min_epochs" \
    --expected-arch "$expected_arch" \
    --expected-total-steps 13136 \
    --min-global-step "$min_step" \
    "${common_protocol_args[@]}" \
    --expected-lambda-hf "$lambda_hf" \
    "$@" \
    --quiet
}

mkdir -p "$LOG_ROOT"
actual_trainer_sha256="$(sha256sum "$FROZEN_TRAINER" | awk '{print $1}')"
entrypoint_sha256="$(sha256sum "$TRAINER_ENTRYPOINT" | awk '{print $1}')"
if [[ "$actual_trainer_sha256" != "$FROZEN_TRAINER_SHA256" || \
      "$entrypoint_sha256" != "$FROZEN_TRAINER_SHA256" ]]; then
  echo "frozen trainer SHA-256 mismatch" >&2
  exit 2
fi

exec 9>"$LOG_ROOT/${exp_name}.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] UPR-14M-noHF pilot queue already active" \
    | tee -a "$queue_log"
  exit 0
fi
exec 8>"$LOG_ROOT/${exp_name}.nohf_train.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] UPR-14M-noHF training already owned" \
    | tee -a "$queue_log"
  exit 0
fi

for reference_marker in "$PP3_START_MARKER" "$ATMVFI_START_MARKER"; do
  while [[ ! -e "$reference_marker" ]]; do
    echo "[$(date -Is)] wait primary reference priority marker=$reference_marker" \
      | tee -a "$queue_log"
    sleep "$SLEEP_SEC"
  done
done

while ! checkpoint_ready \
  "$upr_two_epoch" "$arch" 2 3284 0.05; do
  if [[ -e "$upr_terminal" ]]; then
    touch "$terminal_marker"
    echo "[$(date -Is)] strict UPR+HF reference failed" \
      | tee -a "$queue_log"
    exit 2
  fi
  echo "[$(date -Is)] wait two-epoch UPR+HF reference" \
    | tee -a "$queue_log"
  sleep "$SLEEP_SEC"
done
while ! checkpoint_ready \
  "$flow_two_epoch" flow_pp3 2 3284 0; do
  echo "[$(date -Is)] wait two-epoch Flow-PP3 reference" \
    | tee -a "$queue_log"
  sleep "$SLEEP_SEC"
done

if checkpoint_ready \
  "$checkpoint" "$arch" 8 13136 0 --require-resume-lineage; then
  touch "$screened_marker"
  touch "$terminal_marker"
  echo "[$(date -Is)] reuse complete UPR-14M-noHF checkpoint=$checkpoint" \
    | tee -a "$queue_log"
  exit 0
fi
if [[ -e "$terminal_marker" && -s "$assessment" ]]; then
  promoted="$("$PYTHON_BIN" tools/eval/pilot_assessment_status.py \
    "$assessment" \
    --expected-candidate "$model_name" \
    --expected-reference flow_pp3_nohf \
    --print-promoted)"
  if [[ "$promoted" -ne 1 ]]; then
    touch "$screened_marker"
    echo "[$(date -Is)] reuse excluded UPR-14M-noHF pilot" \
      | tee -a "$queue_log"
    exit 0
  fi
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

wait_for_architecture_completion "$queue_log"
acquire_gpu

run_stage() {
  local resume=()
  if [[ -s "$checkpoint" ]]; then
    resume=(--ckpt_path "$checkpoint")
  fi
  touch "$started_marker"
  echo "[$(date -Is)] launch UPR-14M-noHF epochs=8 gpu=$gpu" \
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
      --lambda_hf_override 0 \
      --ckpt_every_n_epochs 1 \
      "${resume[@]}"
  ) >>"$run_log" 2>&1
}

while ! checkpoint_ready "$checkpoint" "$arch" 2 3284 0; do
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
  if checkpoint_ready "$checkpoint" "$arch" 2 3284 0; then
    echo "[$(date -Is)] complete UPR-14M-noHF two-epoch screen" \
      | tee -a "$queue_log"
    continue
  fi
  if tail -c "+$((log_start_bytes + 1))" "$run_log" | \
    grep -Eqi "out of memory|CUDA error: out of memory"; then
    echo "[$(date -Is)] intrinsic OOM violates frozen bs=4 pilot" \
      | tee -a "$queue_log"
    touch "$terminal_marker"
    flock -u "$lock_fd"
    exit 2
  fi
  echo "[$(date -Is)] pilot failed train=$train_status stop=$stop_status; retry" \
    | tee -a "$queue_log"
  sleep "$SLEEP_SEC"
done

checkpoint_ready "$candidate_two_epoch" "$arch" 2 3284 0
checkpoint_ready "$upr_two_epoch" "$arch" 2 3284 0.05
checkpoint_ready "$flow_two_epoch" flow_pp3 2 3284 0

candidate_csv="$("$PYTHON_BIN" tools/train/select_validation_metrics_csv.py \
  "$LOG_ROOT/$exp_name" --required-epoch 1 --required-step 3283)"
upr_csv="$("$PYTHON_BIN" tools/train/select_validation_metrics_csv.py \
  "$LOG_ROOT/$upr_name" --required-epoch 1 --required-step 3283)"
flow_csv="$("$PYTHON_BIN" tools/train/select_validation_metrics_csv.py \
  "$LOG_ROOT/$flow_name" --required-epoch 1 --required-step 3283)"

"$PYTHON_BIN" tools/eval/assess_flow_spherical_ep_pilot.py \
  --pilot-csv "$candidate_csv" \
  --reference-csv "$flow_csv" \
  --pilot-name "$model_name" \
  --reference-name flow_pp3_nohf \
  --output "$assessment" >>"$queue_log" 2>&1
"$PYTHON_BIN" tools/eval/summarize_matched_training_curves.py \
  --left-csv "$upr_csv" \
  --right-csv "$candidate_csv" \
  --left-name upr_implicit_global_14m_hf \
  --right-name "$model_name" \
  --output "$ablation" >>"$queue_log" 2>&1

promoted="$("$PYTHON_BIN" tools/eval/pilot_assessment_status.py \
  "$assessment" \
  --expected-candidate "$model_name" \
  --expected-reference flow_pp3_nohf \
  --print-promoted)"
if [[ "$promoted" -ne 1 ]]; then
  touch "$screened_marker"
  touch "$terminal_marker"
  echo "[$(date -Is)] exclude UPR-14M-noHF at two-epoch gate" \
    | tee -a "$queue_log"
  flock -u "$lock_fd"
  exit 0
fi

touch "$screened_marker"
release_gpu
while [[ ! -e "$query_started" && ! -e "$query_terminal" ]]; do
  echo "[$(date -Is)] yield GPU until QueryMatch starts" \
    | tee -a "$queue_log"
  sleep "$SLEEP_SEC"
done
acquire_gpu

echo "[$(date -Is)] promote UPR-14M-noHF to eight epochs" \
  | tee -a "$queue_log"
while ! checkpoint_ready \
  "$checkpoint" "$arch" 8 13136 0 --require-resume-lineage; do
  log_start_bytes="$(stat -c %s "$run_log" 2>/dev/null || echo 0)"
  if ! run_stage; then
    if tail -c "+$((log_start_bytes + 1))" "$run_log" | \
      grep -Eqi "out of memory|CUDA error: out of memory"; then
      echo "[$(date -Is)] intrinsic OOM violates frozen bs=4 promoted" \
        | tee -a "$queue_log"
      touch "$terminal_marker"
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
echo "[$(date -Is)] complete UPR-14M-noHF checkpoint=$checkpoint" \
  | tee -a "$queue_log"
