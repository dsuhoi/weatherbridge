#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source tools/train/architecture_candidate_profile.sh

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
SELECTION="${SELECTION:-metrics/upr_lite_candidate_selection.json}"
SEED_COMPARISON="${SEED_COMPARISON:-metrics/upr_nohf_vs_pp3_seed_comparison.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"
MAX_UTIL="${MAX_UTIL:-5}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
BASE_TRAINER_SHA256="e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a"
QUERY_MATCH_TRAINER_SHA256="32f4ce54ddd3d033ea4ea55112d6e65658caf73961e568680aa627e8794e1f30"
CANDIDATE_OUT="metrics/upr_nohf_seed_spectra_6h_2021"
REFERENCE_OUT="metrics/upr_vs_pp3_seed_spectra_6h_2021"
LOG="$LOG_ROOT/upr_nohf_vs_pp3_ood_spectra_2021.queue.log"

mkdir -p "$LOG_ROOT" "$CANDIDATE_OUT"
while [[ ! -s "$SELECTION" || ! -s "$SEED_COMPARISON" ]]; do
  echo "[$(date -Is)] wait no-HF seed comparison" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done

readarray -t report_fields < <("$PYTHON_BIN" -c '
import hashlib, json, sys
selection_path, report_path = sys.argv[1:]
selection_sha = hashlib.sha256(open(selection_path, "rb").read()).hexdigest()
report = json.load(open(report_path))
objective = report.get("training_objective", {})
valid = (
    report.get("schema_version") == 6
    and report.get("selection_sha256") == selection_sha
    and report.get("candidate_seed_artifact_mode")
    == "fresh_postselection_replicates"
    and report.get("reference") == "weatherbridge_ref"
    and report.get("seeds") == [202707, 202708, 202709]
    and report.get("primary_quality_superiority_seed_consistent") is True
    and objective.get("matched") is True
    and objective.get("candidate_lambda_hf") == 0.0
    and objective.get("reference_lambda_hf") == 0.0
)
print(report.get("selected_architecture", ""))
print(report.get("candidate", ""))
print(int(valid))
' "$SELECTION" "$SEED_COMPARISON")
candidate_arch="${report_fields[0]}"
candidate="${report_fields[1]}"
valid="${report_fields[2]}"
if [[ -z "$candidate_arch" || "$candidate" != "${candidate_arch}_nohf" || \
      "$valid" -ne 1 ]]; then
  echo "[$(date -Is)] stop no-HF OOD spectrum: seed gate failed" \
    | tee -a "$LOG"
  exit 0
fi

CANDIDATE_TRAINER_SHA256="$(
  architecture_candidate_trainer_sha256 "$candidate_arch"
)"
model_arch="$(architecture_candidate_model_arch "$candidate_arch")"
layout="$(architecture_candidate_layout "$candidate_arch")"
batch_size="$(architecture_candidate_batch_size "$candidate_arch")"
accumulate="$(architecture_candidate_accumulate "$candidate_arch")"
candidate_boundary="$(
  architecture_candidate_highpass_boundary "$candidate_arch"
)"
declare -A CANDIDATE_CHECKPOINTS=(
  [202707]="$LOG_ROOT/exp_${candidate_arch}_nohf_24ch_6h_2014_19_sparse135_lr1e4_${layout}_s202707/last.ckpt"
  [202708]="$LOG_ROOT/exp_${candidate_arch}_nohf_24ch_6h_2014_19_sparse135_lr1e4_${layout}_s202708/last.ckpt"
  [202709]="$LOG_ROOT/exp_${candidate_arch}_nohf_24ch_6h_2014_19_sparse135_lr1e4_${layout}_s202709/last.ckpt"
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
    "$candidate_boundary" "$CANDIDATE_TRAINER_SHA256"; do
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
  [[ -n "$gpu" ]] || sleep "$SLEEP_SEC"
done

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
for seed in 202707 202708 202709; do
  model_name="$candidate"
  if [[ "$seed" -ne 202707 ]]; then
    model_name="${candidate}_s${seed}"
  fi
  "$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2021 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --ckpt "${CANDIDATE_CHECKPOINTS[$seed]}" \
    --model-name "$model_name" \
    --model-kind capmatched \
    --out-dir "$CANDIDATE_OUT" \
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

for seed in 202707 202708 202709; do
  reference_name=weatherbridge_ref
  if [[ "$seed" -ne 202707 ]]; then
    reference_name="weatherbridge_ref_s${seed}"
  fi
  for tau in 2 4; do
    reference_file="$REFERENCE_OUT/${reference_name}_tau${tau}.npz"
    while [[ ! -s "$reference_file" ]]; do
      echo "[$(date -Is)] wait PP3 OOD spectrum seed=$seed tau=$tau" \
        | tee -a "$LOG"
      sleep "$SLEEP_SEC"
    done
  done
done

"$PYTHON_BIN" -u tools/eval/summarize_upr_vs_pp3_ood_spectra.py \
  --selection "$SELECTION" \
  --spectra-root "$CANDIDATE_OUT" \
  --reference-spectra-root "$REFERENCE_OUT" \
  --seed-comparison "$SEED_COMPARISON" \
  --candidate-artifact-name "$candidate" \
  --candidate-artifact-mode fresh_postselection_replicates \
  --candidate-lambda-hf 0 \
  --reference-lambda-hf 0 \
  --seeds 202707,202708,202709 \
  --hf-ell-min 180 \
  --spectral-taus 2,4 \
  --out-json metrics/upr_nohf_vs_pp3_ood_spectra_2021.json \
  --out-md metrics/upr_nohf_vs_pp3_ood_spectra_2021.md \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] no-HF OOD spectral confirmation complete" | tee -a "$LOG"
