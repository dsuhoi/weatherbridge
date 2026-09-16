#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source tools/train/architecture_candidate_profile.sh

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
SELECTION="${SELECTION:-metrics/upr_lite_candidate_selection.json}"
SCREEN_JSON="${SCREEN_JSON:-metrics/upr_lite_screen/two_epoch_screen.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"
MAX_UTIL="${MAX_UTIL:-5}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
CLIMATOLOGY="${CLIMATOLOGY:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CLIMATOLOGY_CACHE="${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}"
BASE_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_e8d3bd58.py"
BASE_TRAINER_SHA256="e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a"
QUERY_MATCH_TRAINER="legacy/training_snapshots/train_capacity_matched_6h_32f4ce54.py"
QUERY_MATCH_TRAINER_SHA256="32f4ce54ddd3d033ea4ea55112d6e65658caf73961e568680aa627e8794e1f30"

LOG="$LOG_ROOT/upr_lite_winner_nohf.queue.log"
mkdir -p "$LOG_ROOT" "$CLIMATOLOGY_CACHE"

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
import json
from pathlib import Path
from tools.eval.summarize_upr_lite_seeds import (
    choose_transfer_candidate,
    validate_frozen_selection_for_followup,
)
payload = json.loads(Path("'"$SELECTION"'").read_text())
validate_frozen_selection_for_followup(payload)
print(choose_transfer_candidate(payload, "quality"))
')"
TRAINER_ENTRYPOINT="$(architecture_candidate_trainer "$candidate")"
TRAINER_SHA256="$(architecture_candidate_trainer_sha256 "$candidate")"
actual_trainer_sha256="$(sha256sum "$TRAINER_ENTRYPOINT" | awk '{print $1}')"
if [[ "$actual_trainer_sha256" != "$TRAINER_SHA256" ]]; then
  echo "trainer snapshot SHA-256 mismatch: $actual_trainer_sha256" >&2
  exit 2
fi
if [[ "$candidate" == "upr_lite" || \
      "$candidate" == "upr_implicit_global_14m_nohf" ]]; then
  echo "[$(date -Is)] skip no-HF ablation: candidate already uses lambda_hf=0" \
    | tee -a "$LOG"
  exit 0
fi

declare -A DEFAULT_CHECKPOINTS=(
  [upr_lite]="$LOG_ROOT/exp_upr_lite_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707/last.ckpt"
  [upr_lite_lap]="$LOG_ROOT/exp_upr_lite_lap_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707/last.ckpt"
  [upr_lite_column]="$LOG_ROOT/exp_upr_lite_column_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707/last.ckpt"
  [upr_lite_continuous]="$LOG_ROOT/exp_upr_lite_continuous_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707/last.ckpt"
  [upr_lite_continuous_m]="$LOG_ROOT/exp_upr_lite_continuous_m_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707/last.ckpt"
  [upr_lite_implicit_global]="$LOG_ROOT/exp_upr_lite_implicit_global_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707/last.ckpt"
  [upr_lite_implicit_global_q4]="$LOG_ROOT/exp_upr_lite_implicit_global_q4_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707/last.ckpt"
  [upr_implicit_global_14m]="$LOG_ROOT/exp_upr_implicit_global_14m_hf_135_14m_6h_s202707_protocol_v2/last.ckpt"
  [upr_endpoint_implicit_global_14m]="$LOG_ROOT/exp_upr_endpoint_implicit_global_14m_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707/last.ckpt"
  [upr_spherical_implicit_global_14m]="$LOG_ROOT/exp_upr_spherical_implicit_global_14m_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707/last.ckpt"
  [upr_query_match_14m]="$LOG_ROOT/exp_upr_query_match_14m_hf_135_14m_6h_s202707_protocol_v1/last.ckpt"
  [flow_spherical_ep]="$LOG_ROOT/exp_flow_spherical_ep_14m_6h_s202707/last.ckpt"
  [flow_pp3_hf]="$LOG_ROOT/exp_flow_pp3_hf_135_14m_6h_s202707_protocol_v2/last.ckpt"
  [amt]="$LOG_ROOT/exp_weather_amt_l_14m_6h_s202707_protocol_v4/last.ckpt"
)

model_arch="$(architecture_candidate_model_arch "$candidate")"
batch_size="$(architecture_candidate_batch_size "$candidate")"
val_batch_size="$(architecture_candidate_val_batch_size "$candidate")"
accumulate="$(architecture_candidate_accumulate "$candidate")"
train_batches_per_epoch=0
layout="$(architecture_candidate_layout "$candidate")"
expected_highpass_boundary="$(
  architecture_candidate_highpass_boundary "$candidate"
)"
exp_name="exp_${candidate}_nohf_24ch_6h_2014_19_sparse135_lr1e4_${layout}_s202707"
checkpoint="$LOG_ROOT/$exp_name/last.ckpt"
default_checkpoint="${DEFAULT_CHECKPOINTS[$candidate]}"
run_log="$LOG_ROOT/$exp_name.log"
model_name="${candidate}_nohf"
exec 8>"$LOG_ROOT/${exp_name}.nohf_train.lock"
flock 8
echo "[$(date -Is)] acquired no-HF training/evaluation ownership" \
  | tee -a "$LOG"

gpu=""
lock_fd=""
while [[ -z "$gpu" ]]; do
  for candidate_gpu in $GPU_CANDIDATES; do
    exec {candidate_fd}>"$LOG_ROOT/.upr_lite_gpu${candidate_gpu}.lock"
    if ! flock -n "$candidate_fd"; then
      exec {candidate_fd}>&-
      continue
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$candidate_gpu" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$candidate_gpu" | tr -d ' ')"
    if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      gpu="$candidate_gpu"
      lock_fd="$candidate_fd"
      break
    fi
    flock -u "$candidate_fd"
    exec {candidate_fd}>&-
  done
  if [[ -z "$gpu" ]]; then
    sleep "$SLEEP_SEC"
  fi
done

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
echo "[$(date -Is)] no-HF ablation candidate=$candidate gpu=$gpu" | tee -a "$LOG"

while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$checkpoint" \
  --min-epochs 8 \
  --expected-arch "$model_arch" \
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
  --expected-batch-size-per-device "$batch_size" \
  --expected-accumulate-grad-batches "$accumulate" \
  --expected-samples-per-date-train 4 \
  --expected-samples-per-date-val 2 \
  --expected-lambda-hf 0 \
  --expected-highpass-boundary "$expected_highpass_boundary" \
  --expected-precision bf16-mixed \
  --expected-trainer-sha256 "$TRAINER_SHA256" \
  --quiet; do
  resume=()
  if [[ -s "$checkpoint" ]]; then
    resume=(--ckpt_path "$checkpoint")
  fi
  log_start_bytes="$(stat -c %s "$run_log" 2>/dev/null || echo 0)"
  set +e
  "$PYTHON_BIN" -u "$TRAINER_ENTRYPOINT" \
    --arch "$model_arch" \
    --exp_name "$exp_name" \
    --log_root "$LOG_ROOT" \
    --gpus 0 \
    --bs "$batch_size" \
    --val_bs "$val_batch_size" \
    --accumulate "$accumulate" \
    --train_batches_per_epoch "$train_batches_per_epoch" \
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
    "${resume[@]}" >>"$run_log" 2>&1
  status=$?
  set -e
  echo "[$(date -Is)] no-HF train finish status=$status" | tee -a "$LOG"
  if [[ "$status" -ne 0 ]]; then
    if tail -c "+$((log_start_bytes + 1))" "$run_log" | \
      grep -Eqi "out of memory|CUDA error: out of memory"; then
      free_after="$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
      if [[ "$free_after" -lt "$MIN_FREE_MIB" ]]; then
        echo "[$(date -Is)] external GPU contention after OOM; keep bs=$batch_size accumulate=$accumulate" \
          | tee -a "$LOG"
      else
        echo "[$(date -Is)] intrinsic OOM violates frozen bs=$batch_size accumulate=$accumulate protocol" \
          | tee -a "$LOG"
        exit "$status"
      fi
    else
      echo "[$(date -Is)] no-HF training failed without recoverable OOM" \
        | tee -a "$LOG"
      exit "$status"
    fi
    sleep "$SLEEP_SEC"
  fi
done
if ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$checkpoint" \
  --min-epochs 8 \
  --expected-arch "$model_arch" \
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
  --expected-batch-size-per-device "$batch_size" \
  --expected-accumulate-grad-batches "$accumulate" \
  --expected-samples-per-date-train 4 \
  --expected-samples-per-date-val 2 \
  --expected-lambda-hf 0 \
  --expected-highpass-boundary "$expected_highpass_boundary" \
  --expected-precision bf16-mixed \
  --expected-trainer-sha256 "$TRAINER_SHA256" \
  --quiet; then
  echo "[$(date -Is)] no-HF checkpoint failed protocol validation" \
    | tee -a "$LOG"
  exit 2
fi

for year in 2020 2021; do
  result="metrics/upr_lite_nohf_6h_${year}/${model_name}.json"
  if "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
    "$result" \
    --checkpoint "$checkpoint" \
    --required-taus 1,2,3,4,5 \
    --acc-mode enabled \
    --require-physical-metrics \
    --require-temporal-metrics \
    --quiet >>"$LOG" 2>&1; then
    echo "[$(date -Is)] skip validated no-HF field year=$year" \
      | tee -a "$LOG"
  else
    "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir /tmp/wb2_0p5_cache \
      --test-year "$year" \
      --climatology "$CLIMATOLOGY" \
      --climatology-cache-dir "$CLIMATOLOGY_CACHE" \
      --lazy-climatology \
      --stats-path data/json_stats_0p5.nc \
      --surface-stats-path data/surface_stats_0p5.json \
      --static-path data/static_features_0p5.pt \
      --models "$model_name:$checkpoint:" \
      --out-dir "metrics/upr_lite_nohf_6h_${year}" \
      --paper-tag "upr_lite_nohf_6h_${year}" \
      --batch-size 4 \
      --num-workers 2 \
      --samples-per-date 4 \
      --full-year \
      --max-tau-hours 6 \
      --eval-hours 1,2,3,4,5 \
      --seen-tau 1,3,5 \
      --unseen-tau 2,4 \
      --keep-n-channels 24 \
      --proper-rmse \
      --save-window-metrics \
      --save-physical-metrics \
      --save-temporal-metrics 2>&1 | tee -a "$LOG"
    "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
      "$result" \
      --checkpoint "$checkpoint" \
      --required-taus 1,2,3,4,5 \
      --acc-mode enabled \
      --require-physical-metrics \
      --require-temporal-metrics \
      --quiet
  fi
done

"$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
  --memmap-dir /tmp/wb2_0p5_cache \
  --test-year 2020 \
  --stats-path data/json_stats_0p5.nc \
  --surface-stats-path data/surface_stats_0p5.json \
  --static-path data/static_features_0p5.pt \
  --ckpt "$checkpoint" \
  --model-name "$model_name" \
  --model-kind capmatched \
  --out-dir metrics/upr_lite_nohf_spectra_6h_2020 \
  --taus 1,2,3,4,5 \
  --channels all \
  --lmax 359 \
  --keep-n-channels 24 \
  --batch-size 2 \
  --samples-per-date 2 \
  --eval-days-per-month 2 \
  --hf-ell-min 180 \
  --max-tau-hours 6 \
  --skip-existing 2>&1 | tee -a "$LOG"

for primary_result in \
  "metrics/upr_lite_screen_6h_2021/${candidate}.json" \
  "metrics/upr_lite_screen_6h_2021/weatherbridge_ref.json"; do
  while [[ ! -s "$primary_result" ]]; do
    echo "[$(date -Is)] wait primary OOD artifact=$primary_result" \
      | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
done

"$PYTHON_BIN" -u tools/eval/summarize_highpass_ablation.py \
  --selection "$SELECTION" \
  --primary-root-2020 metrics/upr_lite_screen_6h_2020 \
  --primary-root-2021 metrics/upr_lite_screen_6h_2021 \
  --nohf-root-2020 metrics/upr_lite_nohf_6h_2020 \
  --nohf-root-2021 metrics/upr_lite_nohf_6h_2021 \
  --primary-spectra-root metrics/upr_lite_screen_spectra_6h_2020 \
  --nohf-spectra-root metrics/upr_lite_nohf_spectra_6h_2020 \
  --unseen-taus 2,4 \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --output metrics/upr_lite_highpass_ablation.json \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] no-HF ablation complete candidate=$candidate default=$default_checkpoint" | tee -a "$LOG"
flock -u "$lock_fd"
