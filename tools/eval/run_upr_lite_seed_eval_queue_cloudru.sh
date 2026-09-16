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
BASE_TRAINER_SHA256="e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a"
QUERY_MATCH_TRAINER_SHA256="32f4ce54ddd3d033ea4ea55112d6e65658caf73961e568680aa627e8794e1f30"
CLIMATOLOGY="${CLIMATOLOGY:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CLIMATOLOGY_CACHE="${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}"
LOG="$LOG_ROOT/upr_lite_seed_eval_queue.log"

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
from pathlib import Path
import json
from tools.eval.summarize_upr_lite_seeds import (
    choose_transfer_candidate,
    validate_frozen_selection_for_followup,
)
selection = json.loads(Path("'"$SELECTION"'").read_text())
validate_frozen_selection_for_followup(selection)
print(choose_transfer_candidate(selection, "quality"))
')"
CANDIDATE_TRAINER_SHA256="$(
  architecture_candidate_trainer_sha256 "$candidate"
)"
model_arch="$(architecture_candidate_model_arch "$candidate")"
layout="$(architecture_candidate_layout "$candidate")"
batch_size="$(architecture_candidate_batch_size "$candidate")"
accumulate="$(architecture_candidate_accumulate "$candidate")"
expected_lambda_hf="$(architecture_candidate_lambda_hf "$candidate")"
expected_highpass_boundary="$(
  architecture_candidate_highpass_boundary "$candidate"
)"
declare -A CHECKPOINTS=(
  [202707]="$LOG_ROOT/exp_${candidate}_24ch_6h_2014_19_sparse135_lr1e4_${layout}_confirm_s202707/last.ckpt"
  [202708]="$LOG_ROOT/exp_${candidate}_24ch_6h_2014_19_sparse135_lr1e4_${layout}_confirm_s202708/last.ckpt"
  [202709]="$LOG_ROOT/exp_${candidate}_24ch_6h_2014_19_sparse135_lr1e4_${layout}_confirm_s202709/last.ckpt"
)
for seed in 202707 202708 202709; do
  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[$seed]}" \
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
    --expected-seed "$seed" \
    --expected-effective-batch-size 16 \
    --expected-batch-size-per-device "$batch_size" \
    --expected-accumulate-grad-batches "$accumulate" \
    --expected-samples-per-date-train 4 \
    --expected-samples-per-date-val 2 \
    --expected-lambda-hf "$expected_lambda_hf" \
    --expected-highpass-boundary "$expected_highpass_boundary" \
    --expected-precision bf16-mixed \
    --expected-trainer-sha256 "$CANDIDATE_TRAINER_SHA256" \
    --quiet; do
    echo "[$(date -Is)] wait complete candidate=$candidate seed=$seed" | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
done

gpu=""
lock_fd=""
while [[ -z "$gpu" ]]; do
  for gpu_candidate in $GPU_CANDIDATES; do
    exec {candidate_fd}>"$LOG_ROOT/.upr_lite_gpu${gpu_candidate}.lock"
    if ! flock -n "$candidate_fd"; then
      exec {candidate_fd}>&-
      continue
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_candidate" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_candidate" | tr -d ' ')"
    if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      gpu="$gpu_candidate"
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
models="${candidate}_s202707:${CHECKPOINTS[202707]}:,${candidate}_s202708:${CHECKPOINTS[202708]}:,${candidate}_s202709:${CHECKPOINTS[202709]}:"
echo "[$(date -Is)] seed evaluation start candidate=$candidate gpu=$gpu" | tee -a "$LOG"

for year in 2020 2021; do
  artifacts_valid=1
  for seed in 202707 202708 202709; do
    result="metrics/upr_lite_seed_replicates_6h_${year}/${candidate}_s${seed}.json"
    if ! "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
      "$result" \
      --checkpoint "${CHECKPOINTS[$seed]}" \
      --required-taus 1,2,3,4,5 \
      --acc-mode enabled \
      --require-physical-metrics \
      --require-temporal-metrics \
      --quiet >>"$LOG" 2>&1; then
      artifacts_valid=0
    fi
  done
  if [[ "$artifacts_valid" -eq 1 ]]; then
    echo "[$(date -Is)] skip validated seed fields candidate=$candidate year=$year" \
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
      --models "$models" \
      --out-dir "metrics/upr_lite_seed_replicates_6h_${year}" \
      --paper-tag "upr_lite_seed_replicates_6h_${year}" \
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
      --save-temporal-metrics \
      2>&1 | tee -a "$LOG"
    for seed in 202707 202708 202709; do
      "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
        "metrics/upr_lite_seed_replicates_6h_${year}/${candidate}_s${seed}.json" \
        --checkpoint "${CHECKPOINTS[$seed]}" \
        --required-taus 1,2,3,4,5 \
        --acc-mode enabled \
        --require-physical-metrics \
        --require-temporal-metrics \
        --quiet
    done
  fi
done

for seed in 202707 202708 202709; do
  "$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --ckpt "${CHECKPOINTS[$seed]}" \
    --model-name "${candidate}_s${seed}" \
    --model-kind capmatched \
    --out-dir metrics/upr_lite_seed_spectra_6h_2020 \
    --taus 2,4 \
    --channels all \
    --lmax 359 \
    --keep-n-channels 24 \
    --batch-size 2 \
    --samples-per-date 2 \
    --eval-days-per-month 2 \
    --hf-ell-min 180 \
    --max-tau-hours 6 \
    --skip-existing \
    2>&1 | tee -a "$LOG"
done

"$PYTHON_BIN" -u tools/eval/summarize_upr_lite_seeds.py \
  --selection "$SELECTION" \
  --root-2020 metrics/upr_lite_screen_6h_2020 \
  --root-2021 metrics/upr_lite_screen_6h_2021 \
  --replicate-root-2020 metrics/upr_lite_seed_replicates_6h_2020 \
  --replicate-root-2021 metrics/upr_lite_seed_replicates_6h_2021 \
  --spectra-root-2020 metrics/upr_lite_screen_spectra_6h_2020 \
  --replicate-spectra-root-2020 metrics/upr_lite_seed_spectra_6h_2020 \
  --hf-ell-min 180 \
  --spectral-taus 2,4 \
  --require-temporal \
  --all-seeds-from-replicates \
  --out-json metrics/upr_lite_seed_robustness.json \
  --out-md metrics/upr_lite_seed_robustness.md \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] seed evaluation complete candidate=$candidate gpu=$gpu" | tee -a "$LOG"
flock -u "$lock_fd"
