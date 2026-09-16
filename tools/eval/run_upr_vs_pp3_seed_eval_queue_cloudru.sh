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
FROZEN_TRAINER_SHA256="${FROZEN_TRAINER_SHA256:-e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a}"
CLIMATOLOGY="${CLIMATOLOGY:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CLIMATOLOGY_CACHE="${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}"
LOG="$LOG_ROOT/upr_vs_pp3_seed_eval_queue.log"

mkdir -p "$LOG_ROOT" "$CLIMATOLOGY_CACHE"
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
candidate="$("$PYTHON_BIN" -c '
import json, sys
print(json.load(open(sys.argv[1]))["candidate"])
' "$UPR_SEED_REPORT")"
candidate_lambda_hf=0.05
if [[ "$candidate" == "upr_lite" ]]; then
  candidate_lambda_hf=0
fi

declare -A CHECKPOINTS=(
  [202708]="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202708/last.ckpt"
  [202709]="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202709/last.ckpt"
)
for seed in 202708 202709; do
  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[$seed]}" \
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
    echo "[$(date -Is)] wait complete PP3 seed=$seed" | tee -a "$LOG"
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
    free_mib="$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits -i "$gpu_candidate" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits -i "$gpu_candidate" | tr -d ' ')"
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
models="weatherbridge_ref_s202708:${CHECKPOINTS[202708]}:,weatherbridge_ref_s202709:${CHECKPOINTS[202709]}:"
echo "[$(date -Is)] PP3 seed evaluation start candidate=$candidate gpu=$gpu" \
  | tee -a "$LOG"

for year in 2020 2021; do
  artifacts_valid=1
  for seed in 202708 202709; do
    result="metrics/upr_pp3_seed_replicates_6h_${year}/weatherbridge_ref_s${seed}.json"
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
    echo "[$(date -Is)] skip validated PP3 seed fields year=$year" | tee -a "$LOG"
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
      --out-dir "metrics/upr_pp3_seed_replicates_6h_${year}" \
      --paper-tag "upr_pp3_seed_replicates_6h_${year}" \
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
    for seed in 202708 202709; do
      "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
        "metrics/upr_pp3_seed_replicates_6h_${year}/weatherbridge_ref_s${seed}.json" \
        --checkpoint "${CHECKPOINTS[$seed]}" \
        --required-taus 1,2,3,4,5 \
        --acc-mode enabled \
        --require-physical-metrics \
        --require-temporal-metrics \
        --quiet
    done
  fi
done

for seed in 202708 202709; do
  "$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --ckpt "${CHECKPOINTS[$seed]}" \
    --model-name "weatherbridge_ref_s${seed}" \
    --model-kind capmatched \
    --out-dir metrics/upr_pp3_seed_spectra_6h_2020 \
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

"$PYTHON_BIN" -u tools/eval/summarize_upr_vs_pp3_seeds.py \
  --selection "$SELECTION" \
  --primary-root-2020 metrics/upr_lite_screen_6h_2020 \
  --primary-root-2021 metrics/upr_lite_screen_6h_2021 \
  --candidate-replicate-root-2020 metrics/upr_lite_seed_replicates_6h_2020 \
  --candidate-replicate-root-2021 metrics/upr_lite_seed_replicates_6h_2021 \
  --reference-replicate-root-2020 metrics/upr_pp3_seed_replicates_6h_2020 \
  --reference-replicate-root-2021 metrics/upr_pp3_seed_replicates_6h_2021 \
  --primary-spectra-root-2020 metrics/upr_lite_screen_spectra_6h_2020 \
  --candidate-replicate-spectra-root-2020 metrics/upr_lite_seed_spectra_6h_2020 \
  --reference-replicate-spectra-root-2020 metrics/upr_pp3_seed_spectra_6h_2020 \
  --hf-ell-min 180 \
  --spectral-taus 2,4 \
  --candidate-lambda-hf "$candidate_lambda_hf" \
  --reference-lambda-hf 0 \
  --candidate-primary-seed-from-replicate \
  --out-json metrics/upr_vs_pp3_seed_comparison.json \
  --out-md metrics/upr_vs_pp3_seed_comparison.md \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] PP3 seed evaluation complete candidate=$candidate gpu=$gpu" \
  | tee -a "$LOG"
flock -u "$lock_fd"
