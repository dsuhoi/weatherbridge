#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
# Inference is materially lighter than UPR training; retain headroom while
# allowing evaluation on a card with an idle memory-resident process.
MIN_FREE_MIB="${MIN_FREE_MIB:-48000}"
MAX_UTIL="${MAX_UTIL:-5}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
CLIMATOLOGY="${CLIMATOLOGY:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CLIMATOLOGY_CACHE="${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}"
FORECAST_ANCHOR_DIR="${FORECAST_ANCHOR_DIR:-/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_hres_2021}"
FORECAST_ANCHOR_MAX_INITS="${FORECAST_ANCHOR_MAX_INITS:-16}"
SCREEN_JSON="${SCREEN_JSON:-metrics/upr_lite_screen/two_epoch_screen.json}"
FROZEN_TRAINER_SHA256="${FROZEN_TRAINER_SHA256:-e8d3bd5830ff7cc5a15dca986e94570c7e1b6a3707b8b7e3d5709dd17dcd6b6a}"
QUERY_MATCH_TRAINER_SHA256="${QUERY_MATCH_TRAINER_SHA256:-32f4ce54ddd3d033ea4ea55112d6e65658caf73961e568680aa627e8794e1f30}"
LOCAL_CORR_TRAINER_SHA256="${LOCAL_CORR_TRAINER_SHA256:-d7a6bb0d0b801eadb3365bf7b84803c6f499f7ff7969dc5d8a8de9895353e733}"
LOCAL_CORR_LOADER_SHA256="${LOCAL_CORR_LOADER_SHA256:-e2c932e65fa781f7835fafbd10998b8f0da35d1340250ff5a54e4238b102b417}"
LOCAL_CORR_UPR_LITE_SHA256="${LOCAL_CORR_UPR_LITE_SHA256:-b52cdc0407c4e5efce70af7ada523fa2706940519c68ea1b01d1a8bd7ccaf587}"
LOCAL_CORR_UPR_SCALED_SHA256="${LOCAL_CORR_UPR_SCALED_SHA256:-a36c3092943fe6021d9ea0a710e73e62067db0ab24aaf1b55efb03e8c8333cc7}"
AMT_RESIDUAL_TRAINER_SHA256="${AMT_RESIDUAL_TRAINER_SHA256:-62c9566ff7b69db0770afc510ade626b6155b30b5a5b48b26e23138e0694b140}"

declare -A CHECKPOINTS=(
  [upr_lite]="$LOG_ROOT/exp_upr_lite_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707/last.ckpt"
  [upr_lite_lap]="$LOG_ROOT/exp_upr_lite_lap_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707/last.ckpt"
  [upr_lite_column]="$LOG_ROOT/exp_upr_lite_column_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707/last.ckpt"
  [upr_lite_continuous]="$LOG_ROOT/exp_upr_lite_continuous_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707/last.ckpt"
  [upr_lite_continuous_m]="$LOG_ROOT/exp_upr_lite_continuous_m_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707/last.ckpt"
  [upr_lite_implicit_global]="$LOG_ROOT/exp_upr_lite_implicit_global_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707/last.ckpt"
  [upr_lite_implicit_global_q4]="$LOG_ROOT/exp_upr_lite_implicit_global_q4_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707/last.ckpt"
  [upr_implicit_global_14m]="$LOG_ROOT/exp_upr_implicit_global_14m_hf_135_14m_6h_s202707_protocol_v2/last.ckpt"
  [upr_implicit_global_14m_avg3]="$LOG_ROOT/exp_upr_implicit_global_14m_hf_135_14m_6h_s202707_protocol_v2/avg_last3.ckpt"
  [upr_implicit_global_14m_nohf]="$LOG_ROOT/exp_upr_implicit_global_14m_nohf_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707/last.ckpt"
  [upr_endpoint_implicit_global_14m]="$LOG_ROOT/exp_upr_endpoint_implicit_global_14m_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707/last.ckpt"
  [upr_query_match_14m]="$LOG_ROOT/exp_upr_query_match_14m_hf_135_14m_6h_s202707_protocol_v1/last.ckpt"
  [upr_local_corr_14m]="$LOG_ROOT/exp_upr_local_corr_14m_hf_135_14m_6h_s202707_protocol_v1/last.ckpt"
  [upr_spherical_implicit_global_14m]="$LOG_ROOT/exp_upr_spherical_implicit_global_14m_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707/last.ckpt"
  [flow_spherical_ep]="$LOG_ROOT/exp_flow_spherical_ep_14m_6h_s202707/last.ckpt"
  [flow_pp3_hf]="$LOG_ROOT/exp_flow_pp3_hf_135_14m_6h_s202707_protocol_v2/last.ckpt"
  [amt]="$LOG_ROOT/exp_weather_amt_l_14m_6h_s202707_protocol_v4/last.ckpt"
  [amt_residual]="$LOG_ROOT/exp_weather_amt_residual_l_14m_6h_s202707_protocol_v1/last.ckpt"
  [weatherbridge_ref]="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202707_protocol_v2/last.ckpt"
  [atmvfi_ref]="$LOG_ROOT/exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_s202707_protocol_v2/last.ckpt"
)
declare -A EXPECTED_ARCHES=(
  [upr_lite]=upr_lite
  [upr_lite_lap]=upr_lite_lap
  [upr_lite_column]=upr_lite_column
  [upr_lite_continuous]=upr_lite_continuous
  [upr_lite_continuous_m]=upr_lite_continuous_m
  [upr_lite_implicit_global]=upr_lite_implicit_global
  [upr_lite_implicit_global_q4]=upr_lite_implicit_global_q4
  [upr_implicit_global_14m]=upr_implicit_global_14m
  [upr_implicit_global_14m_avg3]=upr_implicit_global_14m
  [upr_implicit_global_14m_nohf]=upr_implicit_global_14m
  [upr_endpoint_implicit_global_14m]=upr_endpoint_implicit_global_14m
  [upr_query_match_14m]=upr_query_match_14m
  [upr_local_corr_14m]=upr_local_corr_14m
  [upr_spherical_implicit_global_14m]=upr_spherical_implicit_global_14m
  [flow_spherical_ep]=flow_spherical_ep
  [flow_pp3_hf]=flow_pp3
  [amt]=amt
  [amt_residual]=amt_residual
  [weatherbridge_ref]=flow_pp3
  [atmvfi_ref]=atmvfi
)

LOG="$LOG_ROOT/upr_lite_6h_eval_queue.log"
matched_protocol_args=(
  --require-training-protocol
  --expected-train-years 2014,2015,2016,2017,2018,2019
  --expected-val-years 2020
  --expected-train-taus 1,3,5
  --expected-eval-taus 1,2,3,4,5
  --expected-seed 202707
  --expected-effective-batch-size 16
  --expected-samples-per-date-train 4
  --expected-samples-per-date-val 2
  --expected-precision bf16-mixed
)

expected_lambda_hf() {
  local name="${1%_avg3}"
  case "$name" in
    weatherbridge_ref|atmvfi_ref|upr_lite|upr_implicit_global_14m_nohf)
      echo 0
      ;;
    *)
      echo 0.05
      ;;
  esac
}

validated_promotion() {
  local artifact="$1"
  local candidate="$2"
  local reference="$3"
  local allow_geometry_rescue="${4:-0}"
  local geometry_args=()
  if [[ "$allow_geometry_rescue" -eq 1 ]]; then
    geometry_args=(--allow-geometry-rescue)
  fi
  "$PYTHON_BIN" tools/eval/pilot_assessment_status.py \
    "$artifact" \
    --expected-candidate "$candidate" \
    --expected-reference "$reference" \
    "${geometry_args[@]}" \
    --print-promoted
}

local_corr_sources_ready() {
  local loader="tools/eval/capmatched_loader.py"
  local upr_lite="weather_time_interp/model/weatherbridge_upr_lite_model.py"
  local upr_scaled="weather_time_interp/model/weatherbridge_upr_scaled_model.py"
  [[ -f "$loader" && -f "$upr_lite" && -f "$upr_scaled" ]] || return 1
  [[ "$(sha256sum "$loader" | awk '{print $1}')" == \
      "$LOCAL_CORR_LOADER_SHA256" ]] || return 1
  [[ "$(sha256sum "$upr_lite" | awk '{print $1}')" == \
      "$LOCAL_CORR_UPR_LITE_SHA256" ]] || return 1
  [[ "$(sha256sum "$upr_scaled" | awk '{print $1}')" == \
      "$LOCAL_CORR_UPR_SCALED_SHA256" ]]
}

checkpoint_extra_args() {
  local name="${1%_avg3}"
  CHECKPOINT_EXTRA_ARGS=()
  case "$name" in
    upr_query_match_14m)
      CHECKPOINT_EXTRA_ARGS=(
        --expected-batch-size-per-device 4
        --expected-accumulate-grad-batches 4
        --expected-highpass-boundary periodic_lon_replicate_lat
        --expected-trainer-sha256 "$QUERY_MATCH_TRAINER_SHA256"
      )
      ;;
    upr_local_corr_14m)
      CHECKPOINT_EXTRA_ARGS=(
        --expected-batch-size-per-device 4
        --expected-accumulate-grad-batches 4
        --expected-highpass-boundary periodic_lon_replicate_lat
        --expected-trainer-sha256 "$LOCAL_CORR_TRAINER_SHA256"
      )
      ;;
    upr_spherical_implicit_global_14m|flow_spherical_ep|amt)
      CHECKPOINT_EXTRA_ARGS=(
        --expected-batch-size-per-device 4
        --expected-accumulate-grad-batches 4
        --expected-highpass-boundary antipodal_vector_parity
        --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256"
      )
      ;;
    amt_residual)
      CHECKPOINT_EXTRA_ARGS=(
        --expected-batch-size-per-device 4
        --expected-accumulate-grad-batches 4
        --expected-highpass-boundary antipodal_vector_parity
        --expected-trainer-sha256 "$AMT_RESIDUAL_TRAINER_SHA256"
      )
      ;;
    upr_implicit_global_14m|upr_endpoint_implicit_global_14m|upr_implicit_global_14m_nohf|flow_pp3_hf|weatherbridge_ref|atmvfi_ref)
      CHECKPOINT_EXTRA_ARGS=(
        --expected-batch-size-per-device 4
        --expected-accumulate-grad-batches 4
        --expected-highpass-boundary periodic_lon_replicate_lat
        --expected-trainer-sha256 "$FROZEN_TRAINER_SHA256"
      )
      ;;
  esac
}

build_avg3_auxiliary() {
  local base_name="$1"
  local averaged_dir
  local averaged_checkpoint
  AVG3_NAME="${base_name}_avg3"
  averaged_dir="$(dirname "${CHECKPOINTS[$base_name]}")"
  averaged_checkpoint="$averaged_dir/avg_last3.ckpt"
  "$PYTHON_BIN" tools/train/average_checkpoints.py \
    --checkpoint-dir "$averaged_dir" \
    --last-n 3 \
    --output "$averaged_checkpoint" >>"$LOG" 2>&1
  CHECKPOINTS[$AVG3_NAME]="$averaged_checkpoint"
  EXPECTED_ARCHES[$AVG3_NAME]="${EXPECTED_ARCHES[$base_name]}"
  checkpoint_extra_args "$base_name"
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$averaged_checkpoint" \
    --min-epochs 8 \
    --expected-arch "${EXPECTED_ARCHES[$AVG3_NAME]}" \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    "${matched_protocol_args[@]}" \
    --expected-lambda-hf "$(expected_lambda_hf "$base_name")" \
    "${CHECKPOINT_EXTRA_ARGS[@]}" \
    --require-resume-lineage \
    --quiet
  echo "[$(date -Is)] added AVG3 deployment auxiliary=$AVG3_NAME" \
    | tee -a "$LOG"
}

mkdir -p "$LOG_ROOT" "$CLIMATOLOGY_CACHE"
exec 9>"$LOG_ROOT/.upr_lite_6h_eval_queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] 6h evaluation queue already active" | tee -a "$LOG"
  exit 0
fi

while true; do
  "$PYTHON_BIN" tools/eval/select_upr_lite_two_epoch_screen.py \
    --log-root "$LOG_ROOT" \
    --output "$SCREEN_JSON" \
    --allow-incomplete >>"$LOG" 2>&1
  screen_complete="$("$PYTHON_BIN" -c '
import json, sys
print(int(json.load(open(sys.argv[1]))["complete"]))
' "$SCREEN_JSON")"
  if [[ "$screen_complete" -eq 1 ]]; then
    break
  fi
  echo "[$(date -Is)] wait complete two-epoch screen" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done

mapfile -t selected < <("$PYTHON_BIN" -c '
import json, sys
print(*json.load(open(sys.argv[1]))["selected"], sep="\n")
' "$SCREEN_JSON")
if [[ "${#selected[@]}" -ne 3 ]]; then
  echo "expected three selected UPR arms, got ${#selected[@]}" >&2
  exit 2
fi
required_models=("${selected[@]}" weatherbridge_ref)

for name in "${required_models[@]}"; do
  lambda_hf="$(expected_lambda_hf "$name")"
  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[$name]}" \
    --min-epochs 8 \
    --expected-arch "${EXPECTED_ARCHES[$name]}" \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    "${matched_protocol_args[@]}" \
    --expected-lambda-hf "$lambda_hf" \
    --quiet; do
    echo "[$(date -Is)] wait complete checkpoint name=$name" | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
done

legacy_diagnostic_models=()
if "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "${CHECKPOINTS[atmvfi_ref]}" \
  --min-epochs 8 \
  --expected-arch "${EXPECTED_ARCHES[atmvfi_ref]}" \
  --expected-total-steps 13136 \
  --min-global-step 13136 \
  --expected-delta-t 6 \
  "${matched_protocol_args[@]}" \
  --expected-lambda-hf 0 \
  --quiet; then
  legacy_diagnostic_models=(atmvfi_ref)
  echo "[$(date -Is)] add complete optional ATM-VFI diagnostic" \
    | tee -a "$LOG"
else
  echo "[$(date -Is)] skip incomplete optional ATM-VFI diagnostic" \
    | tee -a "$LOG"
fi

averaged_base="${selected[0]}"
averaged_name="${averaged_base}_avg3"
averaged_dir="$(dirname "${CHECKPOINTS[$averaged_base]}")"
averaged_checkpoint="$averaged_dir/avg_last3.ckpt"
"$PYTHON_BIN" tools/train/average_checkpoints.py \
  --checkpoint-dir "$averaged_dir" \
  --last-n 3 \
  --output "$averaged_checkpoint" >>"$LOG" 2>&1
CHECKPOINTS[$averaged_name]="$averaged_checkpoint"
EXPECTED_ARCHES[$averaged_name]="${EXPECTED_ARCHES[$averaged_base]}"
"$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$averaged_checkpoint" \
  --min-epochs 8 \
  --expected-arch "${EXPECTED_ARCHES[$averaged_name]}" \
  --expected-total-steps 13136 \
  --min-global-step 13136 \
  --expected-delta-t 6 \
  "${matched_protocol_args[@]}" \
  --expected-lambda-hf "$(expected_lambda_hf "$averaged_base")" \
  --quiet
echo "[$(date -Is)] added AVG3 candidate=$averaged_name checkpoint=$averaged_checkpoint" \
  | tee -a "$LOG"

q4_followup="metrics/upr_lite_screen/q4_followup.json"
while [[ ! -s "$q4_followup" ]]; do
  echo "[$(date -Is)] wait Q4 efficiency follow-up=$q4_followup" \
    | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
q4_promoted="$("$PYTHON_BIN" -c '
import json, sys
print(int(json.load(open(sys.argv[1]))["promoted"]))
' "$q4_followup")"
q4_models=()
q4_selection_models=()
if [[ "$q4_promoted" -eq 1 ]]; then
  q4_name="upr_lite_implicit_global_q4"
  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[$q4_name]}" \
    --min-epochs 8 \
    --expected-arch "${EXPECTED_ARCHES[$q4_name]}" \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    "${matched_protocol_args[@]}" \
    --expected-lambda-hf 0.05 \
    --quiet; do
    echo "[$(date -Is)] wait promoted Q4 checkpoint" | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
  q4_averaged_name="${q4_name}_avg3"
  q4_averaged_dir="$(dirname "${CHECKPOINTS[$q4_name]}")"
  q4_averaged_checkpoint="$q4_averaged_dir/avg_last3.ckpt"
  "$PYTHON_BIN" tools/train/average_checkpoints.py \
    --checkpoint-dir "$q4_averaged_dir" \
    --last-n 3 \
    --output "$q4_averaged_checkpoint" >>"$LOG" 2>&1
  CHECKPOINTS[$q4_averaged_name]="$q4_averaged_checkpoint"
  EXPECTED_ARCHES[$q4_averaged_name]="${EXPECTED_ARCHES[$q4_name]}"
  "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$q4_averaged_checkpoint" \
    --min-epochs 8 \
    --expected-arch "${EXPECTED_ARCHES[$q4_averaged_name]}" \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    "${matched_protocol_args[@]}" \
    --expected-lambda-hf 0.05 \
    --quiet
  q4_models=("$q4_name" "$q4_averaged_name")
  q4_selection_models=("$q4_name")
  echo "[$(date -Is)] added promoted Q4 raw and AVG3 candidates" \
    | tee -a "$LOG"
else
  echo "[$(date -Is)] Q4 excluded by prespecified two-epoch gate" \
    | tee -a "$LOG"
fi

spherical_upr_pilot="metrics/upr_lite_screen/upr_spherical_implicit_global_14m_pilot.json"
spherical_upr_terminal="$LOG_ROOT/exp_upr_spherical_implicit_global_14m_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707.terminal"
while [[ ! -s "$spherical_upr_pilot" && \
         ! -e "$spherical_upr_terminal" ]]; do
  echo "[$(date -Is)] wait spherical UPR pilot=$spherical_upr_pilot" \
    | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
spherical_upr_promoted=0
if [[ -s "$spherical_upr_pilot" ]]; then
  spherical_upr_promoted="$(validated_promotion \
    "$spherical_upr_pilot" \
    upr_spherical_implicit_global_14m \
    upr_implicit_global_14m \
    1)"
else
  echo "[$(date -Is)] exclude spherical UPR: terminal without assessment" \
    | tee -a "$LOG"
fi
spherical_upr_models=()
spherical_upr_aux_models=()
if [[ "$spherical_upr_promoted" -eq 1 ]]; then
  checkpoint_extra_args upr_spherical_implicit_global_14m
  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[upr_spherical_implicit_global_14m]}" \
    --min-epochs 8 \
    --expected-arch upr_spherical_implicit_global_14m \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    "${matched_protocol_args[@]}" \
    --expected-lambda-hf 0.05 \
    "${CHECKPOINT_EXTRA_ARGS[@]}" \
    --require-resume-lineage \
    --quiet; do
    echo "[$(date -Is)] wait promoted spherical UPR checkpoint" \
      | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
  spherical_upr_models=(upr_spherical_implicit_global_14m)
  build_avg3_auxiliary upr_spherical_implicit_global_14m
  spherical_upr_aux_models=("$AVG3_NAME")
  echo "[$(date -Is)] add promoted spherical UPR candidate" \
    | tee -a "$LOG"
else
  echo "[$(date -Is)] exclude spherical UPR at two-epoch gate" \
    | tee -a "$LOG"
fi

endpoint_zero_shot="metrics/upr_lite_screen/upr_endpoint_zero_shot_2020.json"
endpoint_pilot="metrics/upr_lite_screen/upr_endpoint_implicit_global_14m_pilot.json"
endpoint_terminal="$LOG_ROOT/exp_upr_endpoint_implicit_global_14m_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707.terminal"
while [[ ! -s "$endpoint_zero_shot" || ! -e "$endpoint_terminal" ]]; do
  echo "[$(date -Is)] wait endpoint UPR allocation and terminal marker" \
    | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
endpoint_retrain_allocated="$("$PYTHON_BIN" -c '
import json, sys
print(int(json.load(open(sys.argv[1]))["promote_matched_retrain"]))
' "$endpoint_zero_shot")"
endpoint_models=()
endpoint_aux_models=()
if [[ "$endpoint_retrain_allocated" -eq 1 ]]; then
  endpoint_promoted=0
  if [[ -s "$endpoint_pilot" ]]; then
    endpoint_promoted="$(validated_promotion \
      "$endpoint_pilot" \
      upr_endpoint_implicit_global_14m \
      upr_implicit_global_14m)"
  else
    echo "[$(date -Is)] exclude endpoint UPR: terminal without assessment" \
      | tee -a "$LOG"
  fi
  if [[ "$endpoint_promoted" -eq 1 ]]; then
    checkpoint_extra_args upr_endpoint_implicit_global_14m
    while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
      "${CHECKPOINTS[upr_endpoint_implicit_global_14m]}" \
      --min-epochs 8 \
      --expected-arch upr_endpoint_implicit_global_14m \
      --expected-total-steps 13136 \
      --min-global-step 13136 \
      --expected-delta-t 6 \
      "${matched_protocol_args[@]}" \
      --expected-lambda-hf 0.05 \
      "${CHECKPOINT_EXTRA_ARGS[@]}" \
      --require-resume-lineage \
      --quiet; do
      echo "[$(date -Is)] wait promoted endpoint UPR checkpoint" \
        | tee -a "$LOG"
      sleep "$SLEEP_SEC"
    done
    endpoint_models=(upr_endpoint_implicit_global_14m)
    build_avg3_auxiliary upr_endpoint_implicit_global_14m
    endpoint_aux_models=("$AVG3_NAME")
    echo "[$(date -Is)] add promoted endpoint UPR candidate" \
      | tee -a "$LOG"
  else
    echo "[$(date -Is)] exclude endpoint UPR at two-epoch gate" \
      | tee -a "$LOG"
  fi
else
  echo "[$(date -Is)] exclude endpoint UPR at zero-shot gate" \
    | tee -a "$LOG"
fi

flow_pilot="metrics/upr_lite_screen/flow_spherical_ep_pilot.json"
flow_terminal="$LOG_ROOT/exp_flow_spherical_ep_14m_6h_s202707.terminal"
while [[ ! -s "$flow_pilot" && ! -e "$flow_terminal" ]]; do
  echo "[$(date -Is)] wait Flow-Spherical-EP pilot=$flow_pilot" \
    | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
flow_promoted=0
if [[ -s "$flow_pilot" ]]; then
  flow_promoted="$(validated_promotion \
    "$flow_pilot" \
    flow_spherical_ep \
    upr_implicit_global_14m \
    1)"
else
  echo "[$(date -Is)] exclude Flow-Spherical-EP: terminal without assessment" \
    | tee -a "$LOG"
fi
flow_models=()
flow_aux_models=()
if [[ "$flow_promoted" -eq 1 ]]; then
  checkpoint_extra_args flow_spherical_ep
  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[flow_spherical_ep]}" \
    --min-epochs 8 \
    --expected-arch flow_spherical_ep \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    "${matched_protocol_args[@]}" \
    --expected-lambda-hf 0.05 \
    "${CHECKPOINT_EXTRA_ARGS[@]}" \
    --require-resume-lineage \
    --quiet; do
    echo "[$(date -Is)] wait promoted Flow-Spherical-EP checkpoint" \
      | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
  flow_models=(flow_spherical_ep)
  build_avg3_auxiliary flow_spherical_ep
  flow_aux_models=("$AVG3_NAME")
  echo "[$(date -Is)] add promoted Flow-Spherical-EP candidate" \
    | tee -a "$LOG"
else
  echo "[$(date -Is)] exclude Flow-Spherical-EP at two-epoch gate" \
    | tee -a "$LOG"
fi

upr_nohf_pilot="metrics/upr_lite_screen/upr_implicit_global_14m_nohf_pilot.json"
upr_nohf_terminal="$LOG_ROOT/exp_upr_implicit_global_14m_nohf_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707.terminal"
while [[ ! -s "$upr_nohf_pilot" && ! -e "$upr_nohf_terminal" ]]; do
  echo "[$(date -Is)] wait UPR-14M-noHF pilot=$upr_nohf_pilot" \
    | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
upr_nohf_promoted=0
if [[ -s "$upr_nohf_pilot" ]]; then
  upr_nohf_promoted="$(validated_promotion \
    "$upr_nohf_pilot" \
    upr_implicit_global_14m_nohf \
    flow_pp3_nohf)"
else
  echo "[$(date -Is)] exclude UPR-14M-noHF: terminal without assessment" \
    | tee -a "$LOG"
fi
upr_nohf_models=()
upr_nohf_aux_models=()
if [[ "$upr_nohf_promoted" -eq 1 ]]; then
  checkpoint_extra_args upr_implicit_global_14m_nohf
  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[upr_implicit_global_14m_nohf]}" \
    --min-epochs 8 \
    --expected-arch upr_implicit_global_14m \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    "${matched_protocol_args[@]}" \
    --expected-lambda-hf 0 \
    "${CHECKPOINT_EXTRA_ARGS[@]}" \
    --require-resume-lineage \
    --quiet; do
    echo "[$(date -Is)] wait promoted UPR-14M-noHF checkpoint" \
      | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
  upr_nohf_models=(upr_implicit_global_14m_nohf)
  build_avg3_auxiliary upr_implicit_global_14m_nohf
  upr_nohf_aux_models=("$AVG3_NAME")
  echo "[$(date -Is)] add promoted UPR-14M-noHF candidate" \
    | tee -a "$LOG"
else
  echo "[$(date -Is)] exclude UPR-14M-noHF at two-epoch gate" \
    | tee -a "$LOG"
fi

flow_hf_pilot="metrics/upr_lite_screen/flow_pp3_hf_pilot.json"
flow_hf_terminal="$LOG_ROOT/exp_flow_pp3_hf_135_14m_6h_s202707_protocol_v2.terminal"
while [[ ! -s "$flow_hf_pilot" && ! -e "$flow_hf_terminal" ]]; do
  echo "[$(date -Is)] wait Flow-PP3+HF pilot=$flow_hf_pilot" \
    | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
flow_hf_promoted=0
if [[ -s "$flow_hf_pilot" ]]; then
  flow_hf_promoted="$(validated_promotion \
    "$flow_hf_pilot" \
    flow_pp3_hf \
    upr_implicit_global_14m)"
else
  echo "[$(date -Is)] exclude Flow-PP3+HF: terminal without assessment" \
    | tee -a "$LOG"
fi
flow_hf_models=()
flow_hf_aux_models=()
if [[ "$flow_hf_promoted" -eq 1 ]]; then
  checkpoint_extra_args flow_pp3_hf
  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[flow_pp3_hf]}" \
    --min-epochs 8 \
    --expected-arch flow_pp3 \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    "${matched_protocol_args[@]}" \
    --expected-lambda-hf 0.05 \
    "${CHECKPOINT_EXTRA_ARGS[@]}" \
    --require-resume-lineage \
    --quiet; do
    echo "[$(date -Is)] wait promoted Flow-PP3+HF checkpoint" \
      | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
  flow_hf_models=(flow_pp3_hf)
  build_avg3_auxiliary flow_pp3_hf
  flow_hf_aux_models=("$AVG3_NAME")
  echo "[$(date -Is)] add promoted Flow-PP3+HF candidate" \
    | tee -a "$LOG"
else
  echo "[$(date -Is)] exclude Flow-PP3+HF at two-epoch gate" \
    | tee -a "$LOG"
fi

query_match_pilot="metrics/upr_lite_screen/upr_query_match_14m_pilot.json"
query_match_terminal="$LOG_ROOT/exp_upr_query_match_14m_hf_135_14m_6h_s202707_protocol_v1.terminal"
while [[ ! -s "$query_match_pilot" && \
         ! -e "$query_match_terminal" ]]; do
  echo "[$(date -Is)] wait QueryMatch pilot=$query_match_pilot" \
    | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
query_match_promoted=0
if [[ -s "$query_match_pilot" ]]; then
  query_match_promoted="$(validated_promotion \
    "$query_match_pilot" \
    upr_query_match_14m \
    upr_implicit_global_14m)"
else
  echo "[$(date -Is)] exclude QueryMatch: terminal without assessment" \
    | tee -a "$LOG"
fi
query_match_models=()
query_match_aux_models=()
if [[ "$query_match_promoted" -eq 1 ]]; then
  checkpoint_extra_args upr_query_match_14m
  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[upr_query_match_14m]}" \
    --min-epochs 8 \
    --expected-arch upr_query_match_14m \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    "${matched_protocol_args[@]}" \
    --expected-lambda-hf 0.05 \
    "${CHECKPOINT_EXTRA_ARGS[@]}" \
    --require-resume-lineage \
    --quiet; do
    echo "[$(date -Is)] wait promoted QueryMatch checkpoint" \
      | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
  query_match_models=(upr_query_match_14m)
  build_avg3_auxiliary upr_query_match_14m
  query_match_aux_models=("$AVG3_NAME")
  echo "[$(date -Is)] add promoted QueryMatch candidate" \
    | tee -a "$LOG"
else
  echo "[$(date -Is)] exclude QueryMatch at two-epoch gate" \
    | tee -a "$LOG"
fi

local_corr_pilot="metrics/upr_lite_screen/upr_local_corr_14m_pilot.json"
local_corr_terminal="$LOG_ROOT/exp_upr_local_corr_14m_hf_135_14m_6h_s202707_protocol_v1.terminal"
while [[ ! -s "$local_corr_pilot" && \
         ! -e "$local_corr_terminal" ]]; do
  echo "[$(date -Is)] wait LocalCorr pilot=$local_corr_pilot" \
    | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
local_corr_promoted=0
if [[ -s "$local_corr_pilot" ]]; then
  local_corr_promoted="$(validated_promotion \
    "$local_corr_pilot" \
    upr_local_corr_14m \
    upr_implicit_global_14m)"
else
  echo "[$(date -Is)] exclude LocalCorr: terminal without assessment" \
    | tee -a "$LOG"
fi
local_corr_models=()
local_corr_aux_models=()
if [[ "$local_corr_promoted" -eq 1 ]]; then
  while ! local_corr_sources_ready; do
    echo "[$(date -Is)] wait frozen LocalCorr inference sources" \
      | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
  checkpoint_extra_args upr_local_corr_14m
  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[upr_local_corr_14m]}" \
    --min-epochs 8 \
    --expected-arch upr_local_corr_14m \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    "${matched_protocol_args[@]}" \
    --expected-lambda-hf 0.05 \
    "${CHECKPOINT_EXTRA_ARGS[@]}" \
    --require-resume-lineage \
    --quiet; do
    echo "[$(date -Is)] wait promoted LocalCorr checkpoint" \
      | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
  local_corr_models=(upr_local_corr_14m)
  build_avg3_auxiliary upr_local_corr_14m
  local_corr_aux_models=("$AVG3_NAME")
  echo "[$(date -Is)] add promoted LocalCorr candidate" \
    | tee -a "$LOG"
else
  echo "[$(date -Is)] exclude LocalCorr at two-epoch gate" \
    | tee -a "$LOG"
fi

amt_pilot="metrics/upr_lite_screen/weather_amt_l_pilot_v4.json"
amt_terminal="$LOG_ROOT/exp_weather_amt_l_14m_6h_s202707_protocol_v4.terminal"
while [[ ! -s "$amt_pilot" && ! -e "$amt_terminal" ]]; do
  echo "[$(date -Is)] wait WeatherAMT-L pilot=$amt_pilot" \
    | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
amt_promoted=0
if [[ -s "$amt_pilot" ]]; then
  amt_promoted="$(validated_promotion \
    "$amt_pilot" \
    weather_amt_l \
    upr_implicit_global_14m \
    1)"
else
  echo "[$(date -Is)] exclude WeatherAMT-L: terminal without assessment" \
    | tee -a "$LOG"
fi
amt_models=()
amt_aux_models=()
if [[ "$amt_promoted" -eq 1 ]]; then
  checkpoint_extra_args amt
  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[amt]}" \
    --min-epochs 8 \
    --expected-arch amt \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    "${matched_protocol_args[@]}" \
    --expected-lambda-hf 0.05 \
    "${CHECKPOINT_EXTRA_ARGS[@]}" \
    --require-resume-lineage \
    --quiet; do
    echo "[$(date -Is)] wait promoted WeatherAMT-L checkpoint" \
      | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
  amt_models=(amt)
  build_avg3_auxiliary amt
  amt_aux_models=("$AVG3_NAME")
  echo "[$(date -Is)] add promoted WeatherAMT-L candidate" \
    | tee -a "$LOG"
else
  echo "[$(date -Is)] exclude WeatherAMT-L at two-epoch gate" \
    | tee -a "$LOG"
fi

amt_residual_pilot="metrics/upr_lite_screen/weather_amt_residual_l_pilot_v1.json"
amt_residual_terminal="$LOG_ROOT/exp_weather_amt_residual_l_14m_6h_s202707_protocol_v1.terminal"
while [[ ! -s "$amt_residual_pilot" && \
         ! -e "$amt_residual_terminal" ]]; do
  echo "[$(date -Is)] wait WeatherAMT-Residual-L pilot=$amt_residual_pilot" \
    | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
amt_residual_promoted=0
if [[ -s "$amt_residual_pilot" ]]; then
  amt_residual_promoted="$(validated_promotion \
    "$amt_residual_pilot" \
    weather_amt_residual_l \
    upr_implicit_global_14m \
    1)"
else
  echo "[$(date -Is)] exclude WeatherAMT-Residual-L: terminal without assessment" \
    | tee -a "$LOG"
fi
amt_residual_models=()
amt_residual_aux_models=()
if [[ "$amt_residual_promoted" -eq 1 ]]; then
  checkpoint_extra_args amt_residual
  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[amt_residual]}" \
    --min-epochs 8 \
    --expected-arch amt_residual \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    "${matched_protocol_args[@]}" \
    --expected-lambda-hf 0.05 \
    "${CHECKPOINT_EXTRA_ARGS[@]}" \
    --require-resume-lineage \
    --quiet; do
    echo "[$(date -Is)] wait promoted WeatherAMT-Residual-L checkpoint" \
      | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done
  amt_residual_models=(amt_residual)
  build_avg3_auxiliary amt_residual
  amt_residual_aux_models=("$AVG3_NAME")
  echo "[$(date -Is)] add promoted WeatherAMT-Residual-L candidate" \
    | tee -a "$LOG"
else
  echo "[$(date -Is)] exclude WeatherAMT-Residual-L at two-epoch gate" \
    | tee -a "$LOG"
fi

eval_models=(
  "${selected[@]}"
  "$averaged_name"
  "${q4_models[@]}"
  upr_implicit_global_14m
  upr_implicit_global_14m_avg3
  "${upr_nohf_models[@]}"
  "${upr_nohf_aux_models[@]}"
  "${spherical_upr_models[@]}"
  "${spherical_upr_aux_models[@]}"
  "${endpoint_models[@]}"
  "${endpoint_aux_models[@]}"
  "${flow_models[@]}"
  "${flow_aux_models[@]}"
  "${flow_hf_models[@]}"
  "${flow_hf_aux_models[@]}"
  "${query_match_models[@]}"
  "${query_match_aux_models[@]}"
  "${local_corr_models[@]}"
  "${local_corr_aux_models[@]}"
  "${amt_models[@]}"
  "${amt_aux_models[@]}"
  "${amt_residual_models[@]}"
  "${amt_residual_aux_models[@]}"
  "${legacy_diagnostic_models[@]}"
  weatherbridge_ref
)
selection_models=(
  "${selected[@]}"
  "${q4_selection_models[@]}"
  upr_implicit_global_14m
  "${upr_nohf_models[@]}"
  "${spherical_upr_models[@]}"
  "${endpoint_models[@]}"
  "${flow_models[@]}"
  "${flow_hf_models[@]}"
  "${query_match_models[@]}"
  "${local_corr_models[@]}"
  "${amt_models[@]}"
  "${amt_residual_models[@]}"
  weatherbridge_ref
)
models_csv="$(IFS=,; echo "${selection_models[*]}")"
checkpoint_extra_args upr_implicit_global_14m
while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "${CHECKPOINTS[upr_implicit_global_14m]}" \
  --min-epochs 8 \
  --expected-arch upr_implicit_global_14m \
  --expected-total-steps 13136 \
  --min-global-step 13136 \
  --expected-delta-t 6 \
  "${matched_protocol_args[@]}" \
  --expected-lambda-hf 0.05 \
  "${CHECKPOINT_EXTRA_ARGS[@]}" \
  --quiet; do
  echo "[$(date -Is)] wait complete scaled checkpoint before GPU lock" \
    | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done

upr14m_dir="$(dirname "${CHECKPOINTS[upr_implicit_global_14m]}")"
"$PYTHON_BIN" tools/train/average_checkpoints.py \
  --checkpoint-dir "$upr14m_dir" \
  --last-n 3 \
  --output "${CHECKPOINTS[upr_implicit_global_14m_avg3]}" >>"$LOG" 2>&1
"$PYTHON_BIN" tools/train/checkpoint_status.py \
  "${CHECKPOINTS[upr_implicit_global_14m_avg3]}" \
  --min-epochs 8 \
  --expected-arch upr_implicit_global_14m \
  --expected-total-steps 13136 \
  --min-global-step 13136 \
  --expected-delta-t 6 \
  "${matched_protocol_args[@]}" \
  --expected-lambda-hf 0.05 \
  "${CHECKPOINT_EXTRA_ARGS[@]}" \
  --quiet
echo "[$(date -Is)] built UPR-14M AVG3 deployment auxiliary" \
  | tee -a "$LOG"

gpu=""
lock_fd=""
while [[ -z "$gpu" ]]; do
  for candidate in $GPU_CANDIDATES; do
    exec {candidate_fd}>"$LOG_ROOT/.upr_lite_gpu${candidate}.lock"
    if ! flock -n "$candidate_fd"; then
      exec {candidate_fd}>&-
      continue
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
    if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      gpu="$candidate"
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
echo "[$(date -Is)] evaluation start gpu=$gpu" | tee -a "$LOG"

for name in "${eval_models[@]}"; do
  lambda_hf="$(expected_lambda_hf "$name")"
  checkpoint_extra_args "$name"
  lineage_args=()
  if [[ "$name" == upr_implicit_global_14m_nohf* ]]; then
    lineage_args=(--require-resume-lineage)
  elif [[ "$name" == upr_endpoint_implicit_global_14m* || \
          "$name" == upr_spherical_implicit_global_14m* || \
          "$name" == upr_query_match_14m* || \
          "$name" == upr_local_corr_14m* || \
          "$name" == flow_spherical_ep* || \
          "$name" == flow_pp3_hf* || "$name" == amt* ]]; then
    lineage_args=(--require-resume-lineage)
  fi
  while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "${CHECKPOINTS[$name]}" \
    --min-epochs 8 \
    --expected-arch "${EXPECTED_ARCHES[$name]}" \
    --expected-total-steps 13136 \
    --min-global-step 13136 \
    --expected-delta-t 6 \
    "${matched_protocol_args[@]}" \
    --expected-lambda-hf "$lambda_hf" \
    "${CHECKPOINT_EXTRA_ARGS[@]}" \
    "${lineage_args[@]}" \
    --quiet; do
    echo "[$(date -Is)] wait evaluation checkpoint name=$name" \
      | tee -a "$LOG"
    sleep "$SLEEP_SEC"
  done

  result_2020="metrics/upr_lite_screen_6h_2020/${name}.json"
  if "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
    "$result_2020" \
    --checkpoint "${CHECKPOINTS[$name]}" \
    --required-taus 1,2,3,4,5 \
    --acc-mode enabled \
    --require-physical-metrics \
    --require-temporal-metrics \
    --quiet >>"$LOG" 2>&1; then
    echo "[$(date -Is)] skip validated field model=$name year=2020" \
      | tee -a "$LOG"
  else
    "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --climatology "$CLIMATOLOGY" \
    --climatology-cache-dir "$CLIMATOLOGY_CACHE" \
    --lazy-climatology \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "${name}:${CHECKPOINTS[$name]}:" \
    --out-dir metrics/upr_lite_screen_6h_2020 \
    --paper-tag upr_lite_screen_6h_2020 \
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
    "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
      "$result_2020" \
      --checkpoint "${CHECKPOINTS[$name]}" \
      --required-taus 1,2,3,4,5 \
      --acc-mode enabled \
      --require-physical-metrics \
      --require-temporal-metrics \
      --quiet
  fi

  "$PYTHON_BIN" -u tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --ckpt "${CHECKPOINTS[$name]}" \
    --model-name "$name" \
    --model-kind capmatched \
    --out-dir metrics/upr_lite_screen_spectra_6h_2020 \
    --taus 1,2,3,4,5 \
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

benchmark_args=()
for name in "${eval_models[@]}"; do
  benchmark_args+=(--model "${name}:${CHECKPOINTS[$name]}")
done
"$PYTHON_BIN" -u tools/eval/benchmark_capmatched_inference.py \
  "${benchmark_args[@]}" \
  --static-path data/static_features_0p5.pt \
  --batch-size 1 \
  --height 360 \
  --width 720 \
  --warmup 3 \
  --iterations 10 \
  --out-json metrics/upr_lite_inference_cost.json \
  2>&1 | tee -a "$LOG"

"$PYTHON_BIN" -u tools/eval/select_upr_lite_candidate.py \
  --models "$models_csv" \
  --root-2020 metrics/upr_lite_screen_6h_2020 \
  --spectra-root metrics/upr_lite_screen_spectra_6h_2020 \
  --hf-ell-min 180 \
  --spectral-taus 2,4 \
  --cost-json metrics/upr_lite_inference_cost.json \
  --out-json metrics/upr_lite_candidate_selection.json \
  --out-md metrics/upr_lite_candidate_selection.md \
  --freeze-output \
  2>&1 | tee -a "$LOG"

winner="$("$PYTHON_BIN" -c '
from pathlib import Path
import json
selection = json.loads(
    Path("metrics/upr_lite_candidate_selection.json").read_text()
)
if selection.get("ood_attached_at_selection_time") is not False:
    raise SystemExit("winner was not frozen before OOD evaluation")
print(selection["winner"])
')"
echo "[$(date -Is)] frozen 2020 winner=$winner before 2021 OOD" \
  | tee -a "$LOG"

diagnostic_models=(
  "$winner"
  weatherbridge_ref
  "${legacy_diagnostic_models[@]}"
  "${upr_nohf_models[@]}"
  "${spherical_upr_models[@]}"
  "${endpoint_models[@]}"
  "${flow_models[@]}"
  "${flow_hf_models[@]}"
  "${amt_models[@]}"
)
declare -A diagnostic_seen=()
diagnostic_model_specs=()
for name in "${diagnostic_models[@]}"; do
  if [[ -n "${diagnostic_seen[$name]:-}" ]]; then
    continue
  fi
  diagnostic_seen[$name]=1
  diagnostic_model_specs+=("${name}:${CHECKPOINTS[$name]}")
done
diagnostic_models_csv="$(IFS=,; echo "${diagnostic_model_specs[*]}")"

for name in "${eval_models[@]}"; do
  result_2021="metrics/upr_lite_screen_6h_2021/${name}.json"
  if "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
    "$result_2021" \
    --checkpoint "${CHECKPOINTS[$name]}" \
    --required-taus 1,2,3,4,5 \
    --acc-mode enabled \
    --require-physical-metrics \
    --require-temporal-metrics \
    --quiet >>"$LOG" 2>&1; then
    echo "[$(date -Is)] skip validated field model=$name year=2021" \
      | tee -a "$LOG"
  else
    "$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir /tmp/wb2_0p5_cache \
      --test-year 2021 \
      --climatology "$CLIMATOLOGY" \
      --climatology-cache-dir "$CLIMATOLOGY_CACHE" \
      --lazy-climatology \
      --stats-path data/json_stats_0p5.nc \
      --surface-stats-path data/surface_stats_0p5.json \
      --static-path data/static_features_0p5.pt \
      --models "${name}:${CHECKPOINTS[$name]}:" \
      --out-dir metrics/upr_lite_screen_6h_2021 \
      --paper-tag upr_lite_screen_6h_2021_ood \
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
    "$PYTHON_BIN" tools/eval/eval_artifact_status.py \
      "$result_2021" \
      --checkpoint "${CHECKPOINTS[$name]}" \
      --required-taus 1,2,3,4,5 \
      --acc-mode enabled \
      --require-physical-metrics \
      --require-temporal-metrics \
      --quiet
  fi
done

if [[ "$upr_nohf_promoted" -eq 1 && "$flow_hf_promoted" -eq 1 ]]; then
  "$PYTHON_BIN" -u tools/eval/summarize_objective_factorial.py \
    --selection metrics/upr_lite_candidate_selection.json \
    --root-2020 metrics/upr_lite_screen_6h_2020 \
    --root-2021 metrics/upr_lite_screen_6h_2021 \
    --all-taus 1,2,3,4,5 \
    --seen-taus 1,3,5 \
    --unseen-taus 2,4 \
    --block-days 7 \
    --draws 5000 \
    --seed 2027 \
    --out-json metrics/upr_flow_objective_factorial.json \
    2>&1 | tee -a "$LOG"
else
  echo "[$(date -Is)] skip full factorial: one objective arm was excluded" \
    | tee -a "$LOG"
fi

"$PYTHON_BIN" -u tools/eval/validate_upr_lite_selection.py \
  --selection metrics/upr_lite_candidate_selection.json \
  --models "$models_csv" \
  --root-2020 metrics/upr_lite_screen_6h_2020 \
  --root-2021 metrics/upr_lite_screen_6h_2021 \
  --spectra-root metrics/upr_lite_screen_spectra_6h_2020 \
  --spectral-tau 2,4 \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --out-json metrics/upr_lite_candidate_validation.json \
  2>&1 | tee -a "$LOG"

"$PYTHON_BIN" -u tools/eval/assess_avg3_deployment.py \
  --selection metrics/upr_lite_candidate_selection.json \
  --root-2020 metrics/upr_lite_screen_6h_2020 \
  --root-2021 metrics/upr_lite_screen_6h_2021 \
  --spectra-root metrics/upr_lite_screen_spectra_6h_2020 \
  --all-taus 1,2,3,4,5 \
  --seen-taus 1,3,5 \
  --unseen-taus 2,4 \
  --spectral-taus 2,4 \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --output metrics/upr_lite_avg3_deployment.json \
  2>&1 | tee -a "$LOG"

forecast_root="metrics/upr_lite_forecast_anchor_2021_v2"
mkdir -p "$forecast_root"
forecast_artifact_args=()
declare -A forecast_seen=()
for name in "${diagnostic_models[@]}"; do
  if [[ -n "${forecast_seen[$name]:-}" ]]; then
    continue
  fi
  forecast_seen[$name]=1
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
    --delta-t-hours 6 \
    --taus 1 2 3 4 5 \
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
  --delta-t-hours 6 \
  --taus 1 2 3 4 5 \
  --max-inits "$FORECAST_ANCHOR_MAX_INITS" \
  2>&1 | tee -a "$LOG"
forecast_artifact_args+=(--artifact "linear=${linear_result}")
"$PYTHON_BIN" -u tools/eval/summarize_forecast_anchor.py \
  "${forecast_artifact_args[@]}" \
  --selection metrics/upr_lite_candidate_selection.json \
  --draws 5000 \
  --seed 2027 \
  --output "$forecast_root/summary.json" \
  2>&1 | tee -a "$LOG"

exchange_model_specs=()
for name in "${selection_models[@]}"; do
  exchange_model_specs+=("${name}:${CHECKPOINTS[$name]}")
done
exchange_models_csv="$(IFS=,; echo "${exchange_model_specs[*]}")"
for year in 2020 2021; do
  "$PYTHON_BIN" -u tools/eval/eval_anchor_exchange_consistency.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year "$year" \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "$exchange_models_csv" \
    --max-tau-hours 6 \
    --eval-hours 1,2,3,4,5 \
    --samples-per-date 2 \
    --eval-days-per-month 2 \
    --batch-size 2 \
    --num-workers 2 \
    --keep-n-channels 24 \
    --out-json "metrics/upr_lite_anchor_exchange_6h_${year}.json" \
    2>&1 | tee -a "$LOG"
done
"$PYTHON_BIN" -u tools/eval/summarize_anchor_exchange_consistency.py \
  --selection metrics/upr_lite_candidate_selection.json \
  --artifact-2020 metrics/upr_lite_anchor_exchange_6h_2020.json \
  --artifact-2021 metrics/upr_lite_anchor_exchange_6h_2021.json \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --output metrics/upr_lite_anchor_exchange_summary.json \
  2>&1 | tee -a "$LOG"

for year in 2020 2021; do
  "$PYTHON_BIN" -u tools/eval/region_season_12h_eval.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year "$year" \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "$exchange_models_csv" \
    --out-dir "metrics/upr_lite_region_season_unseen_6h_${year}" \
    --batch-size 4 \
    --num-workers 2 \
    --samples-per-date 2 \
    --eval-days-per-month 8 \
    --max-tau-hours 6 \
    --eval-hours 2,4 \
    --keep-n-channels 24 \
    2>&1 | tee -a "$LOG"
done

"$PYTHON_BIN" -u tools/eval/summarize_region_season_generalization.py \
  --selection metrics/upr_lite_candidate_selection.json \
  --root-2020 metrics/upr_lite_region_season_unseen_6h_2020 \
  --root-2021 metrics/upr_lite_region_season_unseen_6h_2021 \
  --models "$models_csv" \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --output metrics/upr_lite_region_season_summary_6h.json \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] evaluation complete gpu=$gpu" | tee -a "$LOG"
flock -u "$lock_fd"
