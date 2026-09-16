#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
# Screen each candidate in isolation under the frozen bs=4 protocol.
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

arch="flow_spherical_ep"
exp_name="exp_flow_spherical_ep_14m_6h_s202707"
checkpoint="$LOG_ROOT/$exp_name/last.ckpt"
run_log="$LOG_ROOT/$exp_name.log"
queue_log="$LOG_ROOT/$exp_name.queue.log"
assessment="metrics/upr_lite_screen/flow_spherical_ep_pilot.json"
polar_dir="metrics/upr_lite_screen/flow_spherical_ep_polar_2020"
terminal_marker="$LOG_ROOT/${exp_name}.terminal"
screened_marker="$LOG_ROOT/${exp_name}.screened"
source tools/train/architecture_screen_barrier.sh
upr_name="exp_upr_implicit_global_14m_hf_135_14m_6h_s202707_protocol_v2"
upr_checkpoint="$LOG_ROOT/$upr_name/last.ckpt"
pilot_two_epoch_checkpoint="$LOG_ROOT/$exp_name/epoch=1-step=3284.ckpt"
upr_two_epoch_checkpoint="$LOG_ROOT/$upr_name/epoch=1-step=3284.ckpt"
batch_size=4
accumulate=4

mark_failed_exit() {
  local status=$?
  trap - EXIT
  if [[ "$status" -ne 0 ]]; then
    touch "$terminal_marker"
  fi
  exit "$status"
}
trap mark_failed_exit EXIT

protocol_args=(
  --expected-delta-t 6
  --require-training-protocol
  --expected-train-years 2014,2015,2016,2017,2018,2019
  --expected-val-years 2020
  --expected-train-taus 1,3,5
  --expected-eval-taus 1,2,3,4,5
  --expected-seed 202707
  --expected-effective-batch-size 16
  --expected-samples-per-date-train 4
  --expected-samples-per-date-val 2
  --expected-lambda-hf 0.05
  --expected-precision bf16-mixed
)
candidate_protocol_args=(
  "${protocol_args[@]}"
  --expected-batch-size-per-device 4
  --expected-accumulate-grad-batches 4
  --expected-highpass-boundary antipodal_vector_parity
  --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256"
)

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
  echo "[$(date -Is)] spherical EP pilot queue already active" \
    | tee -a "$queue_log"
  exit 0
fi

while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$upr_checkpoint" \
  --min-epochs 2 \
  --expected-arch upr_implicit_global_14m \
  --expected-total-steps 13136 \
  --min-global-step 3284 \
  "${protocol_args[@]}" \
  --quiet; do
  echo "[$(date -Is)] wait two-epoch UPR-14M reference before pilot" \
    | tee -a "$queue_log"
  sleep "$SLEEP_SEC"
done

for reference_marker in "$PP3_START_MARKER" "$ATMVFI_START_MARKER"; do
  while [[ ! -e "$reference_marker" ]]; do
    echo "[$(date -Is)] wait primary reference priority marker=$reference_marker" \
      | tee -a "$queue_log"
    sleep "$SLEEP_SEC"
  done
done

if "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$checkpoint" \
  --min-epochs 8 \
  --expected-arch "$arch" \
  --expected-total-steps 13136 \
  --min-global-step 13136 \
  "${candidate_protocol_args[@]}" \
  --require-resume-lineage \
  --quiet; then
  touch "$terminal_marker"
  echo "[$(date -Is)] reuse complete promoted checkpoint=$checkpoint" \
    | tee -a "$queue_log"
  exit 0
fi

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
      if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
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
  local epochs="$1"
  local resume=()
  if [[ -s "$checkpoint" ]]; then
    resume=(--ckpt_path "$checkpoint")
  fi
  echo "[$(date -Is)] launch $arch epochs=$epochs gpu=$gpu" \
    | tee -a "$queue_log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
    "$PYTHON_BIN" -u "$TRAINER_ENTRYPOINT" \
      --arch "$arch" \
      --exp_name "$exp_name" \
      --log_root "$LOG_ROOT" \
      --gpus 0 \
      --bs "$batch_size" \
      --val_bs 2 \
      --accumulate "$accumulate" \
      --workers 4 \
      --val_workers 2 \
      --release_memmap_pages \
      --precision bf16-mixed \
      --seed 202707 \
      --years 2014 2015 2016 2017 2018 2019 \
      --val_years 2020 \
      --max_epochs "$epochs" \
      --lr 1e-4 \
      --window_hours 6 \
      --train_tau_subset 1 3 5 \
      --eval_tau 1 2 3 4 5 \
      --samples_per_date_train 4 \
      --samples_per_date_val 2 \
      --ckpt_every_n_epochs 1 \
      "${resume[@]}"
  ) >>"$run_log" 2>&1
}

while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$checkpoint" \
  --min-epochs 2 \
  --expected-arch "$arch" \
  --expected-total-steps 13136 \
  --min-global-step 3284 \
  "${candidate_protocol_args[@]}" \
  --quiet; do
  log_start_bytes="$(stat -c %s "$run_log" 2>/dev/null || echo 0)"
  set +e
  run_stage 8 &
  screen_pid=$!
  "$PYTHON_BIN" tools/train/stop_after_screen_checkpoint.py \
    --queue-pid "$screen_pid" \
    --experiment "$exp_name" \
    --checkpoint "$checkpoint" \
    --min-epochs 2 \
    --poll-seconds 30 \
    >>"$queue_log" 2>&1
  stop_status=$?
  wait "$screen_pid"
  train_status=$?
  set -e
  if "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$checkpoint" \
    --min-epochs 2 \
    --expected-arch "$arch" \
    --expected-total-steps 13136 \
    --min-global-step 3284 \
    "${candidate_protocol_args[@]}" \
    --quiet; then
    echo "[$(date -Is)] complete matched-schedule two-epoch screen" \
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
  else
    echo "[$(date -Is)] pilot stage failed train_status=$train_status stop_status=$stop_status; retry" \
      | tee -a "$queue_log"
    sleep "$SLEEP_SEC"
  fi
done

"$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$pilot_two_epoch_checkpoint" \
  --min-epochs 2 \
  --expected-arch "$arch" \
  --expected-total-steps 13136 \
  --min-global-step 3284 \
  "${candidate_protocol_args[@]}" \
  --quiet
"$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$upr_two_epoch_checkpoint" \
  --min-epochs 2 \
  --expected-arch upr_implicit_global_14m \
  --expected-total-steps 13136 \
  --min-global-step 3284 \
  "${protocol_args[@]}" \
  --quiet

pilot_csv="$("$PYTHON_BIN" \
  tools/train/select_validation_metrics_csv.py \
  "$LOG_ROOT/$exp_name" \
  --required-epoch 1 \
  --required-step 3283)"
upr_csv="$("$PYTHON_BIN" \
  tools/train/select_validation_metrics_csv.py \
  "$LOG_ROOT/$upr_name" \
  --required-epoch 1 \
  --required-step 3283)"

echo "[$(date -Is)] evaluate fixed 2020 polar two-epoch screen" \
  | tee -a "$queue_log"
(
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
  "$PYTHON_BIN" -u tools/eval/region_season_12h_eval.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "flow_spherical_ep:$pilot_two_epoch_checkpoint,upr_implicit_global_14m:$upr_two_epoch_checkpoint" \
    --out-dir "$polar_dir" \
    --batch-size 4 \
    --num-workers 2 \
    --samples-per-date 2 \
    --eval-days-per-month 8 \
    --max-tau-hours 6 \
    --eval-hours 2,4 \
    --keep-n-channels 24
) >>"$queue_log" 2>&1

"$PYTHON_BIN" tools/eval/assess_flow_spherical_ep_pilot.py \
  --pilot-csv "$pilot_csv" \
  --reference-csv "$upr_csv" \
  --polar-pilot-json "$polar_dir/flow_spherical_ep.json" \
  --polar-reference-json "$polar_dir/upr_implicit_global_14m.json" \
  --polar-pilot-checkpoint "$pilot_two_epoch_checkpoint" \
  --polar-reference-checkpoint "$upr_two_epoch_checkpoint" \
  --output "$assessment" >>"$queue_log" 2>&1
promoted="$("$PYTHON_BIN" tools/eval/pilot_assessment_status.py \
  "$assessment" \
  --expected-candidate flow_spherical_ep \
  --expected-reference upr_implicit_global_14m \
  --allow-geometry-rescue \
  --print-promoted)"
if [[ "$promoted" -ne 1 ]]; then
  touch "$terminal_marker"
  echo "[$(date -Is)] exclude $arch at two-epoch gate" \
    | tee -a "$queue_log"
  flock -u "$lock_fd"
  exit 0
fi

wait_for_architecture_screen_barrier "$exp_name" "$queue_log"
echo "[$(date -Is)] promote $arch to eight epochs" | tee -a "$queue_log"
while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$checkpoint" \
  --min-epochs 8 \
  --expected-arch "$arch" \
  --expected-total-steps 13136 \
  --min-global-step 13136 \
  "${candidate_protocol_args[@]}" \
  --require-resume-lineage \
  --quiet; do
  log_start_bytes="$(stat -c %s "$run_log" 2>/dev/null || echo 0)"
  if ! run_stage 8; then
    if tail -c "+$((log_start_bytes + 1))" "$run_log" | \
      grep -Eqi "out of memory|CUDA error: out of memory"; then
      echo "[$(date -Is)] intrinsic OOM violates frozen bs=4 promoted" \
        | tee -a "$queue_log"
      touch "$terminal_marker"
      flock -u "$lock_fd"
      exit 2
    else
      echo "[$(date -Is)] promoted stage failed; retry" \
        | tee -a "$queue_log"
      sleep "$SLEEP_SEC"
    fi
  fi
done
touch "$terminal_marker"
flock -u "$lock_fd"
echo "[$(date -Is)] complete promoted checkpoint=$checkpoint" \
  | tee -a "$queue_log"
