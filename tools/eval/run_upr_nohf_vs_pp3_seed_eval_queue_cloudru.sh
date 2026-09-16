#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source tools/train/architecture_candidate_profile.sh

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
SELECTION="${SELECTION:-metrics/upr_lite_candidate_selection.json}"
ABLATION="${ABLATION:-metrics/upr_lite_highpass_ablation.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"
MAX_UTIL="${MAX_UTIL:-5}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
BASE_TRAINER_SHA256="e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a"
QUERY_MATCH_TRAINER_SHA256="32f4ce54ddd3d033ea4ea55112d6e65658caf73961e568680aa627e8794e1f30"
CLIMATOLOGY="${CLIMATOLOGY:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CLIMATOLOGY_CACHE="${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}"
LOG="$LOG_ROOT/upr_nohf_vs_pp3_seed_eval.queue.log"

mkdir -p "$LOG_ROOT" "$CLIMATOLOGY_CACHE"
while [[ ! -s "$SELECTION" || ! -s "$ABLATION" ]]; do
  echo "[$(date -Is)] wait no-HF promotion inputs" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done

readarray -t promotion < <("$PYTHON_BIN" -c '
import hashlib, json, sys
selection_path, ablation_path = sys.argv[1:]
selection_sha = hashlib.sha256(open(selection_path, "rb").read()).hexdigest()
ablation = json.load(open(ablation_path))
candidate = ablation.get("candidate")
valid = (
    ablation.get("schema_version") == 2
    and ablation.get("selection_sha256") == selection_sha
    and ablation.get("nohf_model") == f"{candidate}_nohf"
    and ablation.get("training_seed_count") == 1
    and ablation.get("architecture_only_single_seed_promotion_passed")
    is True
)
print(candidate or "")
print(int(valid))
' "$SELECTION" "$ABLATION")
candidate="${promotion[0]}"
promoted="${promotion[1]}"
if [[ -z "$candidate" || "$promoted" -ne 1 ]]; then
  echo "[$(date -Is)] stop no-HF seed eval: promotion gate failed" \
    | tee -a "$LOG"
  exit 0
fi
nohf="${candidate}_nohf"
CANDIDATE_TRAINER_SHA256="$(
  architecture_candidate_trainer_sha256 "$candidate"
)"
model_arch="$(architecture_candidate_model_arch "$candidate")"
layout="$(architecture_candidate_layout "$candidate")"
batch_size="$(architecture_candidate_batch_size "$candidate")"
accumulate="$(architecture_candidate_accumulate "$candidate")"
expected_highpass_boundary="$(
  architecture_candidate_highpass_boundary "$candidate"
)"

declare -A CANDIDATE_CHECKPOINTS=(
  [202707]="$LOG_ROOT/exp_${candidate}_nohf_24ch_6h_2014_19_sparse135_lr1e4_${layout}_s202707/last.ckpt"
  [202708]="$LOG_ROOT/exp_${candidate}_nohf_24ch_6h_2014_19_sparse135_lr1e4_${layout}_s202708/last.ckpt"
  [202709]="$LOG_ROOT/exp_${candidate}_nohf_24ch_6h_2014_19_sparse135_lr1e4_${layout}_s202709/last.ckpt"
)
declare -A REFERENCE_CHECKPOINTS=(
  [202707]="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202707_protocol_v2/last.ckpt"
  [202708]="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202708/last.ckpt"
  [202709]="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202709/last.ckpt"
)

checkpoint_valid() {
  local checkpoint="$1"
  local arch="$2"
  local seed="$3"
  local boundary="$4"
  local trainer_sha256="$5"
  local expected_batch_size="${6:-$batch_size}"
  local expected_accumulate="${7:-$accumulate}"
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
    --expected-lambda-hf 0 \
    --expected-highpass-boundary "$boundary" \
    --expected-precision bf16-mixed \
    --expected-trainer-sha256 "$trainer_sha256" \
    --quiet
}

for seed in 202707 202708 202709; do
  while ! checkpoint_valid \
    "${CANDIDATE_CHECKPOINTS[$seed]}" "$model_arch" "$seed" \
    "$expected_highpass_boundary" "$CANDIDATE_TRAINER_SHA256"; do
    echo "[$(date -Is)] wait no-HF checkpoint seed=$seed" | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
  while ! checkpoint_valid \
    "${REFERENCE_CHECKPOINTS[$seed]}" flow_pp3 "$seed" \
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
  if [[ -z "$gpu" ]]; then
    sleep "$SLEEP_SEC"
  fi
done

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
models="${nohf}_s202708:${CANDIDATE_CHECKPOINTS[202708]}:,${nohf}_s202709:${CANDIDATE_CHECKPOINTS[202709]}:"

for year in 2020 2021; do
  valid=1
  for seed in 202708 202709; do
    result="metrics/upr_nohf_seed_replicates_6h_${year}/${nohf}_s${seed}.json"
    if ! "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
      "$result" \
      --checkpoint "${CANDIDATE_CHECKPOINTS[$seed]}" \
      --required-taus 1,2,3,4,5 \
      --acc-mode enabled \
      --require-physical-metrics \
      --require-temporal-metrics \
      --quiet >>"$LOG" 2>&1; then
      valid=0
    fi
  done
  if [[ "$valid" -ne 1 ]]; then
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
      --out-dir "metrics/upr_nohf_seed_replicates_6h_${year}" \
      --paper-tag "upr_nohf_seed_replicates_6h_${year}" \
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
  fi
done

for seed in 202708 202709; do
  "$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --ckpt "${CANDIDATE_CHECKPOINTS[$seed]}" \
    --model-name "${nohf}_s${seed}" \
    --model-kind capmatched \
    --out-dir metrics/upr_nohf_seed_spectra_6h_2020 \
    --taus 2,4 \
    --channels all \
    --lmax 359 \
    --keep-n-channels 24 \
    --batch-size 2 \
    --samples-per-date 2 \
    --eval-days-per-month 2 \
    --hf-ell-min 180 \
    --max-tau-hours 6 \
    --skip-existing 2>&1 | tee -a "$LOG"
done
flock -u "$lock_fd"
exec {lock_fd}>&-

for year in 2020 2021; do
  for seed in 202708 202709; do
    reference_result="metrics/upr_pp3_seed_replicates_6h_${year}/weatherbridge_ref_s${seed}.json"
    while ! "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
      "$reference_result" \
      --checkpoint "${REFERENCE_CHECKPOINTS[$seed]}" \
      --required-taus 1,2,3,4,5 \
      --acc-mode enabled \
      --require-physical-metrics \
      --require-temporal-metrics \
      --quiet; do
      echo "[$(date -Is)] wait PP3 eval seed=$seed year=$year" \
        | tee -a "$LOG"
      sleep "$SLEEP_SEC"
    done
  done
done
for seed in 202708 202709; do
  for tau in 2 4; do
    reference_spectrum="metrics/upr_pp3_seed_spectra_6h_2020/weatherbridge_ref_s${seed}_tau${tau}.npz"
    while [[ ! -s "$reference_spectrum" ]]; do
      echo "[$(date -Is)] wait PP3 spectrum seed=$seed tau=$tau" \
        | tee -a "$LOG"
      sleep "$SLEEP_SEC"
    done
  done
done

"$PYTHON_BIN" -u tools/eval/summarize_upr_vs_pp3_seeds.py \
  --selection "$SELECTION" \
  --primary-root-2020 metrics/upr_lite_nohf_6h_2020 \
  --primary-root-2021 metrics/upr_lite_nohf_6h_2021 \
  --reference-primary-root-2020 metrics/upr_lite_screen_6h_2020 \
  --reference-primary-root-2021 metrics/upr_lite_screen_6h_2021 \
  --candidate-replicate-root-2020 metrics/upr_nohf_seed_replicates_6h_2020 \
  --candidate-replicate-root-2021 metrics/upr_nohf_seed_replicates_6h_2021 \
  --reference-replicate-root-2020 metrics/upr_pp3_seed_replicates_6h_2020 \
  --reference-replicate-root-2021 metrics/upr_pp3_seed_replicates_6h_2021 \
  --primary-spectra-root-2020 metrics/upr_lite_nohf_spectra_6h_2020 \
  --reference-primary-spectra-root-2020 metrics/upr_lite_screen_spectra_6h_2020 \
  --candidate-replicate-spectra-root-2020 metrics/upr_nohf_seed_spectra_6h_2020 \
  --reference-replicate-spectra-root-2020 metrics/upr_pp3_seed_spectra_6h_2020 \
  --candidate-artifact-name "$nohf" \
  --candidate-artifact-mode fresh_postselection_replicates \
  --candidate-lambda-hf 0 \
  --reference-lambda-hf 0 \
  --hf-ell-min 180 \
  --spectral-taus 2,4 \
  --out-json metrics/upr_nohf_vs_pp3_seed_comparison.json \
  --out-md metrics/upr_nohf_vs_pp3_seed_comparison.md \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] no-HF versus PP3 seed comparison complete" | tee -a "$LOG"
