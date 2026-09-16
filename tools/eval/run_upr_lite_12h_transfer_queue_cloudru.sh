#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source tools/train/architecture_candidate_profile.sh

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
SELECTION="${SELECTION:-metrics/upr_lite_candidate_selection.json}"
SCREEN_JSON="${SCREEN_JSON:-metrics/upr_lite_screen/two_epoch_screen.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
MAX_UTIL="${MAX_UTIL:-5}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
CLIMATOLOGY="${CLIMATOLOGY:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CLIMATOLOGY_CACHE="${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}"
FORECAST_ANCHOR_DIR="${FORECAST_ANCHOR_DIR:-/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_hres_2021}"
FORECAST_ANCHOR_MAX_INITS="${FORECAST_ANCHOR_MAX_INITS:-16}"
BASE_TRAINER_SHA256="e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a"
QUERY_MATCH_TRAINER_SHA256="32f4ce54ddd3d033ea4ea55112d6e65658caf73961e568680aa627e8794e1f30"
MEMMAP_DATASET="weather_time_interp/memmap_dataset.py"
MEMMAP_DATASET_SHA256="cc5c21f1150e9cdfa9acb1746557db88a080b09da3e7f8caa4207a25d60fc1fa"
LOG="$LOG_ROOT/upr_lite_12h_transfer_eval.queue.log"

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
mapfile -t transfer_candidates < <("$PYTHON_BIN" -c '
from pathlib import Path
import json
from tools.eval.summarize_upr_lite_seeds import (
    choose_transfer_candidate,
    validate_frozen_selection_for_followup,
)
selection = json.loads(Path("'"$SELECTION"'").read_text())
validate_frozen_selection_for_followup(selection)
print(choose_transfer_candidate(selection, "quality"))
print(choose_transfer_candidate(selection, "efficiency"))
models = selection.get("models", {})
print(next(
    (
        name
        for name in ("flow_spherical_ep", "flow_pp3_hf")
        if name in models
    ),
    "-",
))
')
if [[ "${#transfer_candidates[@]}" -ne 3 ]]; then
  echo "expected quality, efficiency, and Flow-control decisions" >&2
  exit 2
fi
quality_candidate="${transfer_candidates[0]}"
efficiency_candidate="${transfer_candidates[1]}"
flow_candidate="${transfer_candidates[2]}"

candidate_layout() {
  architecture_candidate_layout "$1"
}

candidate_arch() {
  architecture_candidate_model_arch "$1"
}

candidate_lambda_hf() {
  architecture_candidate_lambda_hf "$1"
}

candidate_trainer_sha256() {
  architecture_candidate_trainer_sha256 "$1"
}

declare -A CHECKPOINTS=(
  [weatherbridge_ref]="$LOG_ROOT/exp_flow_pp3_12h_2017_19_held468_lr1e4_sp2_protocol_v3/last.ckpt"
  [weatherdcae_ref]="$LOG_ROOT/exp_weatherdcae_14m_24ch_12h_2017_19_lr1e4_protocol_v3/last.ckpt"
  [atmvfi_ref]="$LOG_ROOT/exp_atmvfi_24ch_12h_2017_19_oddskip_lr1e4_protocol_v3/last.ckpt"
)
declare -A EXPECTED_EPOCHS=(
  [weatherbridge_ref]=10
  [weatherdcae_ref]=10
  [atmvfi_ref]=10
)
declare -A EXPECTED_ARCHES=(
  [weatherbridge_ref]=flow_pp3
  [weatherdcae_ref]=dcae_14m
  [atmvfi_ref]=atmvfi
)
declare -A EXPECTED_LAMBDA_HF=(
  [weatherbridge_ref]=0
  [weatherdcae_ref]=0
  [atmvfi_ref]=0
)
declare -A EXPECTED_TRAINER_SHA256=(
  [weatherbridge_ref]="$BASE_TRAINER_SHA256"
  [weatherdcae_ref]="$BASE_TRAINER_SHA256"
  [atmvfi_ref]="$BASE_TRAINER_SHA256"
)
quality_layout="$(candidate_layout "$quality_candidate")"
CHECKPOINTS[quality_transfer]="$LOG_ROOT/exp_${quality_candidate}_24ch_12h_2017_19_held468_lr1e4_sp2_${quality_layout}_s202707/last.ckpt"
EXPECTED_EPOCHS[quality_transfer]=10
EXPECTED_ARCHES[quality_transfer]="$(candidate_arch "$quality_candidate")"
EXPECTED_LAMBDA_HF[quality_transfer]="$(candidate_lambda_hf "$quality_candidate")"
EXPECTED_TRAINER_SHA256[quality_transfer]="$(candidate_trainer_sha256 "$quality_candidate")"
models=(quality_transfer)
if [[ "$efficiency_candidate" != "$quality_candidate" ]]; then
  efficiency_layout="$(candidate_layout "$efficiency_candidate")"
  CHECKPOINTS[efficiency_transfer]="$LOG_ROOT/exp_${efficiency_candidate}_24ch_12h_2017_19_held468_lr1e4_sp2_${efficiency_layout}_s202707/last.ckpt"
  EXPECTED_EPOCHS[efficiency_transfer]=10
  EXPECTED_ARCHES[efficiency_transfer]="$(candidate_arch "$efficiency_candidate")"
  EXPECTED_LAMBDA_HF[efficiency_transfer]="$(candidate_lambda_hf "$efficiency_candidate")"
  EXPECTED_TRAINER_SHA256[efficiency_transfer]="$(candidate_trainer_sha256 "$efficiency_candidate")"
  models+=(efficiency_transfer)
fi
if [[ "$flow_candidate" != "-" && \
      "$flow_candidate" != "$quality_candidate" && \
      "$flow_candidate" != "$efficiency_candidate" ]]; then
  flow_layout="$(candidate_layout "$flow_candidate")"
  CHECKPOINTS[flow_transfer]="$LOG_ROOT/exp_${flow_candidate}_24ch_12h_2017_19_held468_lr1e4_sp2_${flow_layout}_s202707/last.ckpt"
  EXPECTED_EPOCHS[flow_transfer]=10
  EXPECTED_ARCHES[flow_transfer]="$(candidate_arch "$flow_candidate")"
  EXPECTED_LAMBDA_HF[flow_transfer]="$(candidate_lambda_hf "$flow_candidate")"
  EXPECTED_TRAINER_SHA256[flow_transfer]="$(candidate_trainer_sha256 "$flow_candidate")"
  models+=(flow_transfer)
fi
models+=(weatherbridge_ref weatherdcae_ref)
selection_models=("${models[@]}")

expected_lambda_hf() {
  echo "${EXPECTED_LAMBDA_HF[$1]}"
}

checkpoint_ready() {
  local name="$1"
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[$name]}" \
    --min-epochs "${EXPECTED_EPOCHS[$name]}" \
    --expected-arch "${EXPECTED_ARCHES[$name]}" \
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
    --expected-optimizer-steps-per-epoch 1093 \
    --expected-samples-per-date-train 2 \
    --expected-samples-per-date-val 2 \
    --expected-lambda-hf "$(expected_lambda_hf "$name")" \
    --expected-precision bf16-mixed \
    --expected-trainer-sha256 "${EXPECTED_TRAINER_SHA256[$name]}" \
    --quiet
}

for name in "${selection_models[@]}"; do
  while ! checkpoint_ready "$name"; do
    echo "[$(date -Is)] wait complete checkpoint name=$name path=${CHECKPOINTS[$name]}" | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
done

legacy_diagnostic_models=()
if checkpoint_ready atmvfi_ref; then
  legacy_diagnostic_models=(atmvfi_ref)
  echo "[$(date -Is)] add complete optional 12h ATM-VFI diagnostic" \
    | tee -a "$LOG"
else
  echo "[$(date -Is)] skip incomplete optional 12h ATM-VFI diagnostic" \
    | tee -a "$LOG"
fi
models+=("${legacy_diagnostic_models[@]}")
models_csv="$(IFS=,; echo "${selection_models[*]}")"

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
mkdir -p "$CLIMATOLOGY_CACHE"
echo "[$(date -Is)] 12h transfer evaluation start quality=$quality_candidate efficiency=$efficiency_candidate gpu=$gpu" | tee -a "$LOG"

evaluate_field_year() {
  local name="$1"
  local year="$2"
  local result="metrics/upr_lite_transfer_12h_${year}/${name}.json"
  if "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
    "$result" \
    --checkpoint "${CHECKPOINTS[$name]}" \
    --required-taus 1,2,3,4,5,6,7,8,9,10,11 \
    --acc-mode enabled \
    --require-physical-metrics \
    --require-temporal-metrics \
    --quiet >>"$LOG" 2>&1; then
    echo "[$(date -Is)] skip validated field model=$name year=$year" \
      | tee -a "$LOG"
    return
  fi
  "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year "$year" \
    --climatology "$CLIMATOLOGY" \
    --climatology-cache-dir "$CLIMATOLOGY_CACHE" \
    --lazy-climatology \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "${name}:${CHECKPOINTS[$name]}:" \
    --out-dir "metrics/upr_lite_transfer_12h_${year}" \
    --paper-tag "upr_lite_transfer_12h_${year}" \
    --batch-size 4 \
    --num-workers 2 \
    --samples-per-date 2 \
    --full-year \
    --max-tau-hours 12 \
    --eval-hours 1,2,3,4,5,6,7,8,9,10,11 \
    --seen-tau 1,2,3,5,7,9,10,11 \
    --unseen-tau 4,6,8 \
    --keep-n-channels 24 \
    --proper-rmse \
    --save-window-metrics \
    --save-physical-metrics \
    --save-temporal-metrics \
    2>&1 | tee -a "$LOG"
  "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
    "$result" \
    --checkpoint "${CHECKPOINTS[$name]}" \
    --required-taus 1,2,3,4,5,6,7,8,9,10,11 \
    --acc-mode enabled \
    --require-physical-metrics \
    --require-temporal-metrics \
    --quiet
}

for name in "${models[@]}"; do
  evaluate_field_year "$name" 2020
  "$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --ckpt "${CHECKPOINTS[$name]}" \
    --model-name "$name" \
    --model-kind capmatched \
    --out-dir metrics/upr_lite_transfer_spectra_12h_2020 \
    --taus 4,6,8 \
    --channels all \
    --lmax 359 \
    --keep-n-channels 24 \
    --batch-size 2 \
    --samples-per-date 2 \
    --eval-days-per-month 2 \
    --hf-ell-min 180 \
    --max-tau-hours 12 \
    --skip-existing \
    2>&1 | tee -a "$LOG"
done

benchmark_args=()
region_model_specs=()
for name in "${models[@]}"; do
  benchmark_args+=(--model "${name}:${CHECKPOINTS[$name]}")
  region_model_specs+=("${name}:${CHECKPOINTS[$name]}")
done
region_models_csv="$(IFS=,; echo "${region_model_specs[*]}")"

"$PYTHON_BIN" -u tools/eval/benchmark_capmatched_inference.py \
  "${benchmark_args[@]}" \
  --static-path data/static_features_0p5.pt \
  --batch-size 1 \
  --height 360 \
  --width 720 \
  --warmup 3 \
  --iterations 10 \
  --out-json metrics/upr_lite_transfer_inference_cost_12h.json \
  2>&1 | tee -a "$LOG"

"$PYTHON_BIN" -u tools/eval/select_upr_lite_candidate.py \
  --models "$models_csv" \
  --root-2020 metrics/upr_lite_transfer_12h_2020 \
  --spectra-root metrics/upr_lite_transfer_spectra_12h_2020 \
  --hf-ell-min 180 \
  --spectral-taus 4,6,8 \
  --cost-json metrics/upr_lite_transfer_inference_cost_12h.json \
  --out-json metrics/upr_lite_transfer_selection_12h.json \
  --out-md metrics/upr_lite_transfer_selection_12h.md \
  --freeze-output \
  2>&1 | tee -a "$LOG"

winner="$("$PYTHON_BIN" -c '
from pathlib import Path
import json
selection = json.loads(
    Path("metrics/upr_lite_transfer_selection_12h.json").read_text()
)
if selection.get("ood_attached_at_selection_time") is not False:
    raise SystemExit("12h winner was not frozen before OOD evaluation")
print(selection["winner"])
')"
if [[ -z "${CHECKPOINTS[$winner]:-}" ]]; then
  echo "frozen winner has no checkpoint mapping: $winner" >&2
  exit 2
fi
echo "[$(date -Is)] 12h frozen raw winner=$winner before AVG3 and OOD" \
  | tee -a "$LOG"

averaged_name="${winner}_avg3"
CHECKPOINTS[$averaged_name]="$(dirname "${CHECKPOINTS[$winner]}")/${averaged_name}.ckpt"
EXPECTED_EPOCHS[$averaged_name]="${EXPECTED_EPOCHS[$winner]}"
EXPECTED_ARCHES[$averaged_name]="${EXPECTED_ARCHES[$winner]}"
EXPECTED_LAMBDA_HF[$averaged_name]="${EXPECTED_LAMBDA_HF[$winner]}"
EXPECTED_TRAINER_SHA256[$averaged_name]="${EXPECTED_TRAINER_SHA256[$winner]}"
"$PYTHON_BIN" tools/train/average_checkpoints.py \
  --checkpoint-dir "$(dirname "${CHECKPOINTS[$winner]}")" \
  --last-n 3 \
  --output "${CHECKPOINTS[$averaged_name]}" \
  2>&1 | tee -a "$LOG"
"$PYTHON_BIN" tools/train/checkpoint_status.py \
  "${CHECKPOINTS[$averaged_name]}" \
  --min-epochs "${EXPECTED_EPOCHS[$averaged_name]}" \
  --expected-arch "${EXPECTED_ARCHES[$averaged_name]}" \
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
  --expected-optimizer-steps-per-epoch 1093 \
  --expected-samples-per-date-train 2 \
  --expected-samples-per-date-val 2 \
  --expected-lambda-hf "$(expected_lambda_hf "$averaged_name")" \
  --expected-precision bf16-mixed \
  --expected-trainer-sha256 \
    "${EXPECTED_TRAINER_SHA256[$averaged_name]}" \
  --quiet
echo "[$(date -Is)] built 12h AVG3 deployment auxiliary=$averaged_name" \
  | tee -a "$LOG"

evaluate_field_year "$averaged_name" 2020
"$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
  --memmap-dir /tmp/wb2_0p5_cache \
  --test-year 2020 \
  --stats-path data/json_stats_0p5.nc \
  --surface-stats-path data/surface_stats_0p5.json \
  --static-path data/static_features_0p5.pt \
  --ckpt "${CHECKPOINTS[$averaged_name]}" \
  --model-name "$averaged_name" \
  --model-kind capmatched \
  --out-dir metrics/upr_lite_transfer_spectra_12h_2020 \
  --taus 4,6,8 \
  --channels all \
  --lmax 359 \
  --keep-n-channels 24 \
  --batch-size 2 \
  --samples-per-date 2 \
  --eval-days-per-month 2 \
  --hf-ell-min 180 \
  --max-tau-hours 12 \
  --skip-existing \
  2>&1 | tee -a "$LOG"

for name in "${models[@]}" "$averaged_name"; do
  evaluate_field_year "$name" 2021
done

ood_spectra_root="metrics/upr_lite_transfer_spectra_12h_2021"
for name in "${models[@]}"; do
  "$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2021 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --ckpt "${CHECKPOINTS[$name]}" \
    --model-name "$name" \
    --model-kind capmatched \
    --out-dir "$ood_spectra_root" \
    --taus 4,6,8 \
    --channels all \
    --lmax 359 \
    --keep-n-channels 24 \
    --batch-size 2 \
    --samples-per-date 2 \
    --eval-days-per-month 2 \
    --hf-ell-min 180 \
    --max-tau-hours 12 \
    --skip-existing \
    2>&1 | tee -a "$LOG"
done

"$PYTHON_BIN" -u tools/eval/summarize_frozen_ood_spectra.py \
  --selection metrics/upr_lite_transfer_selection_12h.json \
  --spectra-root "$ood_spectra_root" \
  --models "$models_csv" \
  --delta-t-hours 12 \
  --spectral-taus 4,6,8 \
  --hf-ell-min 180 \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --out-json metrics/upr_lite_transfer_ood_spectra_12h_2021.json \
  2>&1 | tee -a "$LOG"

"$PYTHON_BIN" -u tools/eval/validate_upr_lite_selection.py \
  --selection metrics/upr_lite_transfer_selection_12h.json \
  --models "$models_csv" \
  --root-2020 metrics/upr_lite_transfer_12h_2020 \
  --root-2021 metrics/upr_lite_transfer_12h_2021 \
  --spectra-root metrics/upr_lite_transfer_spectra_12h_2020 \
  --all-tau 1,2,3,4,5,6,7,8,9,10,11 \
  --seen-tau 1,2,3,5,7,9,10,11 \
  --unseen-tau 4,6,8 \
  --spectral-tau 4,6,8 \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --out-json metrics/upr_lite_transfer_validation_12h.json \
  2>&1 | tee -a "$LOG"

"$PYTHON_BIN" -u tools/eval/assess_avg3_deployment.py \
  --selection metrics/upr_lite_transfer_selection_12h.json \
  --root-2020 metrics/upr_lite_transfer_12h_2020 \
  --root-2021 metrics/upr_lite_transfer_12h_2021 \
  --spectra-root metrics/upr_lite_transfer_spectra_12h_2020 \
  --all-taus 1,2,3,4,5,6,7,8,9,10,11 \
  --seen-taus 1,2,3,5,7,9,10,11 \
  --unseen-taus 4,6,8 \
  --spectral-taus 4,6,8 \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --output metrics/upr_lite_transfer_avg3_deployment_12h.json \
  2>&1 | tee -a "$LOG"

forecast_root="metrics/upr_lite_forecast_anchor_12h_2021_v1"
mkdir -p "$forecast_root"
forecast_artifact_args=()
for name in "${models[@]}"; do
  result="$forecast_root/${name}.json"
  "$PYTHON_BIN" -u tools/eval/batch_eval_forecast_anchor.py \
    --forecast-dir "$FORECAST_ANCHOR_DIR" \
    --era5-memmap-dir /tmp/wb2_0p5_cache \
    --checkpoint "${CHECKPOINTS[$name]}" \
    --arch "${EXPECTED_ARCHES[$name]}" \
    --model-name "$name" \
    --out-json "$result" \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --device cuda \
    --delta-t-hours 12 \
    --taus 1 2 3 4 5 6 7 8 9 10 11 \
    --max-inits "$FORECAST_ANCHOR_MAX_INITS" \
    2>&1 | tee -a "$LOG"
  forecast_artifact_args+=(--artifact "${name}=${result}")
done
linear_result="$forecast_root/linear.json"
"$PYTHON_BIN" -u tools/eval/batch_eval_forecast_anchor.py \
  --forecast-dir "$FORECAST_ANCHOR_DIR" \
  --era5-memmap-dir /tmp/wb2_0p5_cache \
  --model-blob linear \
  --model-name linear \
  --out-json "$linear_result" \
  --stats-path data/json_stats_0p5.nc \
  --surface-stats-path data/surface_stats_0p5.json \
  --static-path data/static_features_0p5.pt \
  --delta-t-hours 12 \
  --taus 1 2 3 4 5 6 7 8 9 10 11 \
  --max-inits "$FORECAST_ANCHOR_MAX_INITS" \
  2>&1 | tee -a "$LOG"
forecast_artifact_args+=(--artifact "linear=${linear_result}")
"$PYTHON_BIN" -u tools/eval/summarize_forecast_anchor.py \
  "${forecast_artifact_args[@]}" \
  --selection metrics/upr_lite_transfer_selection_12h.json \
  --seen-taus 1,2,3,5,7,9,10,11 \
  --unseen-taus 4,6,8 \
  --draws 5000 \
  --seed 2027 \
  --output "$forecast_root/summary.json" \
  2>&1 | tee -a "$LOG"

for year in 2020 2021; do
  "$PYTHON_BIN" -u tools/eval/eval_anchor_exchange_consistency.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year "$year" \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "$region_models_csv" \
    --max-tau-hours 12 \
    --eval-hours 1,2,3,4,5,6,7,8,9,10,11 \
    --samples-per-date 2 \
    --eval-days-per-month 2 \
    --batch-size 2 \
    --num-workers 2 \
    --keep-n-channels 24 \
    --out-json "metrics/upr_lite_anchor_exchange_12h_${year}.json" \
    2>&1 | tee -a "$LOG"
done
"$PYTHON_BIN" -u tools/eval/summarize_anchor_exchange_consistency.py \
  --selection metrics/upr_lite_transfer_selection_12h.json \
  --artifact-2020 metrics/upr_lite_anchor_exchange_12h_2020.json \
  --artifact-2021 metrics/upr_lite_anchor_exchange_12h_2021.json \
  --all-taus 1,2,3,4,5,6,7,8,9,10,11 \
  --seen-taus 1,2,3,5,7,9,10,11 \
  --unseen-taus 4,6,8 \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --output metrics/upr_lite_anchor_exchange_summary_12h.json \
  2>&1 | tee -a "$LOG"

for year in 2020 2021; do
  "$PYTHON_BIN" -u tools/eval/region_season_12h_eval.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year "$year" \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "$region_models_csv" \
    --out-dir "metrics/upr_lite_region_season_unseen_12h_${year}" \
    --batch-size 4 \
    --num-workers 2 \
    --samples-per-date 2 \
    --eval-days-per-month 8 \
    --max-tau-hours 12 \
    --eval-hours 4,6,8 \
    --keep-n-channels 24 \
    2>&1 | tee -a "$LOG"
done

"$PYTHON_BIN" -u tools/eval/summarize_region_season_generalization.py \
  --selection metrics/upr_lite_transfer_selection_12h.json \
  --root-2020 metrics/upr_lite_region_season_unseen_12h_2020 \
  --root-2021 metrics/upr_lite_region_season_unseen_12h_2021 \
  --models "$models_csv" \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --output metrics/upr_lite_region_season_summary_12h.json \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] 12h transfer evaluation complete quality=$quality_candidate efficiency=$efficiency_candidate gpu=$gpu" | tee -a "$LOG"
flock -u "$lock_fd"
