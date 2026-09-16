#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source tools/train/architecture_candidate_profile.sh

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
SELECTION="${SELECTION:-metrics/upr_lite_candidate_selection.json}"
SEED_COMPARISON="${SEED_COMPARISON:-metrics/upr_vs_pp3_seed_comparison.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"
MAX_UTIL="${MAX_UTIL:-5}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
BASE_TRAINER_SHA256="e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a"
QUERY_MATCH_TRAINER_SHA256="32f4ce54ddd3d033ea4ea55112d6e65658caf73961e568680aa627e8794e1f30"
OUT_DIR="metrics/upr_vs_pp3_seed_spectra_6h_2021"
LOG="$LOG_ROOT/upr_vs_pp3_ood_spectra_2021.queue.log"

mkdir -p "$LOG_ROOT" "$OUT_DIR"
exec 9>"$LOG_ROOT/.upr_vs_pp3_ood_spectra_2021.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] OOD spectral queue already active" | tee -a "$LOG"
  exit 0
fi

while [[ ! -s "$SELECTION" ]]; do
  echo "[$(date -Is)] wait selection=$SELECTION" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
selection_mtime="$(stat -c %Y "$SELECTION")"
while [[ ! -s "$SEED_COMPARISON" ]] || \
  [[ "$(stat -c %Y "$SEED_COMPARISON")" -le "$selection_mtime" ]]; do
  echo "[$(date -Is)] wait fresh seed comparison=$SEED_COMPARISON" \
    | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done

candidate="$("$PYTHON_BIN" -c '
import json, sys
from tools.eval.summarize_upr_lite_seeds import (
    choose_transfer_candidate,
    validate_frozen_selection_for_followup,
)
selection = json.load(open(sys.argv[1]))
validate_frozen_selection_for_followup(selection)
print(choose_transfer_candidate(selection, "quality"))
' "$SELECTION")"
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
declare -A CANDIDATE_CHECKPOINTS=(
  [202707]="$LOG_ROOT/exp_${candidate}_24ch_6h_2014_19_sparse135_lr1e4_${layout}_confirm_s202707/last.ckpt"
  [202708]="$LOG_ROOT/exp_${candidate}_24ch_6h_2014_19_sparse135_lr1e4_${layout}_confirm_s202708/last.ckpt"
  [202709]="$LOG_ROOT/exp_${candidate}_24ch_6h_2014_19_sparse135_lr1e4_${layout}_confirm_s202709/last.ckpt"
)
declare -A REFERENCE_CHECKPOINTS=(
  [202707]="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202707_protocol_v2/last.ckpt"
  [202708]="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202708/last.ckpt"
  [202709]="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202709/last.ckpt"
)

check_checkpoint() {
  local checkpoint="$1"
  local arch="$2"
  local seed="$3"
  local lambda_hf="$4"
  local highpass_boundary="$5"
  local trainer_sha256="$6"
  local expected_batch_size="${7:-$batch_size}"
  local expected_accumulate="${8:-$accumulate}"
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$checkpoint" \
    --min-epochs 8 \
    --expected-arch "$arch" \
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
    --expected-batch-size-per-device "$expected_batch_size" \
    --expected-accumulate-grad-batches "$expected_accumulate" \
    --expected-samples-per-date-train 4 \
    --expected-samples-per-date-val 2 \
    --expected-lambda-hf "$lambda_hf" \
    --expected-highpass-boundary "$highpass_boundary" \
    --expected-precision bf16-mixed \
    --expected-trainer-sha256 "$trainer_sha256" \
    --quiet
}

for seed in 202707 202708 202709; do
  while ! check_checkpoint \
    "${CANDIDATE_CHECKPOINTS[$seed]}" "$model_arch" "$seed" \
    "$expected_lambda_hf" "$expected_highpass_boundary" \
    "$CANDIDATE_TRAINER_SHA256"; do
    echo "[$(date -Is)] wait candidate checkpoint seed=$seed" | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
  while ! check_checkpoint \
    "${REFERENCE_CHECKPOINTS[$seed]}" flow_pp3 "$seed" 0 \
    periodic_lon_replicate_lat "$BASE_TRAINER_SHA256" 4 4; do
    echo "[$(date -Is)] wait PP3 checkpoint seed=$seed" | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
done

gpu=""
lock_fd=""
while [[ -z "$gpu" ]]; do
  for candidate_gpu in $GPU_CANDIDATES; do
    exec {candidate_fd}>"$LOG_ROOT/.upr_lite_gpu${candidate_gpu}.lock"
    if ! flock -n "$candidate_fd"; then
      exec {candidate_fd}>&-
      continue
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits -i "$candidate_gpu" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits -i "$candidate_gpu" | tr -d ' ')"
    if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      gpu="$candidate_gpu"
      lock_fd="$candidate_fd"
      break
    fi
    flock -u "$candidate_fd"
    exec {candidate_fd}>&-
  done
  [[ -n "$gpu" ]] || sleep "$SLEEP_SEC"
done

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
echo "[$(date -Is)] OOD spectral confirmation start gpu=$gpu candidate=$candidate" \
  | tee -a "$LOG"
for seed in 202707 202708 202709; do
  candidate_name="${candidate}_s${seed}"
  reference_name="weatherbridge_ref"
  if [[ "$seed" -ne 202707 ]]; then
    reference_name="weatherbridge_ref_s${seed}"
  fi
  for spec in \
    "$candidate_name:${CANDIDATE_CHECKPOINTS[$seed]}" \
    "$reference_name:${REFERENCE_CHECKPOINTS[$seed]}"; do
    model_name="${spec%%:*}"
    checkpoint="${spec#*:}"
    "$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
      --memmap-dir /tmp/wb2_0p5_cache \
      --test-year 2021 \
      --stats-path data/json_stats_0p5.nc \
      --surface-stats-path data/surface_stats_0p5.json \
      --static-path data/static_features_0p5.pt \
      --ckpt "$checkpoint" \
      --model-name "$model_name" \
      --model-kind capmatched \
      --out-dir "$OUT_DIR" \
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
done

"$PYTHON_BIN" -u tools/eval/summarize_upr_vs_pp3_ood_spectra.py \
  --selection "$SELECTION" \
  --spectra-root "$OUT_DIR" \
  --seed-comparison "$SEED_COMPARISON" \
  --seeds 202707,202708,202709 \
  --hf-ell-min 180 \
  --spectral-taus 2,4 \
  --candidate-primary-seed-from-replicate \
  --out-json metrics/upr_vs_pp3_ood_spectra_2021.json \
  --out-md metrics/upr_vs_pp3_ood_spectra_2021.md \
  2>&1 | tee -a "$LOG"
echo "[$(date -Is)] OOD spectral confirmation complete" | tee -a "$LOG"
flock -u "$lock_fd"
