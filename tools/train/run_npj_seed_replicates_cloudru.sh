#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
TRAIN_SOURCE="${TRAIN_SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_universal_refine_v1_source}"
EVAL_SOURCE="${EVAL_SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_universal_refine_v2_eval_source}"
EVAL_ROOT="${EVAL_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime/metrics/npj_seed_ifs_hres_2021_v3}"
MEMMAP="${MEMMAP:-/tmp/wb2_0p5_cache}"
HRES="${HRES:-/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_hres_2021_canonical_v3}"
STATIC="${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}"
STATS="${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}"
SURFACE_STATS="${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
STATE_DIR="${STATE_DIR:-$LOG_ROOT/npj_seed_replicates_v2.state}"
QUEUE_LOG="${QUEUE_LOG:-$LOG_ROOT/npj_seed_replicates_v2.queue.log}"
MIN_FREE_MIB="${MIN_FREE_MIB:-70000}"
MAX_UTIL="${MAX_UTIL:-5}"
POLL_SECONDS="${POLL_SECONDS:-60}"
IFS_MAX_INITS="${IFS_MAX_INITS:-16}"
WORKER_GPUS="${WORKER_GPUS:-0 1}"

TRAINER_SHA256="880be34d0f72e0d0cbed3159269b681a65238935fa525efec00d1f841cbeb14d"
FLOW_MODEL_SHA256="36c2c0e98a3c18ce34cabbc0f7eba04f7481292f8b65cddf95714f0b84cdd852"

# Every family uses one immutable trainer snapshot. Existing checkpoints are
# reused only after the same source/protocol checks pass. JOBS_OVERRIDE is used
# by the post-selection Detail confirmation without duplicating trainer logic.
DEFAULT_JOBS=(
  linear:6:0
  linear:12:0
  refine:6:202707
  refine:6:202708
  refine:6:202709
  flow_spectral:6:202707
  flow_spectral:6:202708
  flow_spectral:6:202709
  dcae:6:202707
  dcae:6:202708
  dcae:6:202709
  flow_spectral:12:202707
  flow_spectral:12:202708
  flow_spectral:12:202709
  dcae:12:202707
  dcae:12:202708
  dcae:12:202709
  refine:12:202708
  refine:12:202709
  refine:12:202707
)
if [[ -n "${JOBS_OVERRIDE:-}" ]]; then
  read -r -a JOBS <<<"$JOBS_OVERRIDE"
else
  JOBS=("${DEFAULT_JOBS[@]}")
fi

mkdir -p "$LOG_ROOT" "$EVAL_ROOT" "$STATE_DIR"
LAUNCH_LOCK="${LAUNCH_LOCK:-$STATE_DIR/launch.lock}"
exec 8>"$LAUNCH_LOCK"
if ! flock -n 8; then
  echo "[$(date -Is)] seed replicate queue already active" | tee -a "$QUEUE_LOG"
  exit 0
fi
exec > >(tee -a "$QUEUE_LOG") 2>&1

sha256_check() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA-256 mismatch path=$path actual=$actual expected=$expected" >&2
    return 1
  }
}

verify_inputs() {
  (
    cd "$TRAIN_SOURCE"
    sha256sum --quiet -c SOURCE_SHA256SUMS
  )
  (
    cd "$EVAL_SOURCE"
    sha256sum --quiet -c SOURCE_SHA256SUMS
  )
  sha256_check \
    "$TRAIN_SOURCE/tools/train/train_capacity_matched_6h.py" \
    "$TRAINER_SHA256"
  sha256_check \
    "$TRAIN_SOURCE/weather_time_interp/model/weatherbridge_flow_model.py" \
    "$FLOW_MODEL_SHA256"
  for year in 2014 2015 2016 2017 2018 2019 2020 2021; do
    test -s "$MEMMAP/wb2_${year}.json"
  done
  test -s "$STATIC"
  test -s "$STATS"
  test -s "$SURFACE_STATS"
  test -d "$HRES"
  test -s "$HRES/forecast_archive_manifest.json"
  local init_count
  init_count="$(find "$HRES" -maxdepth 1 -type f -name 'init_*.bin' | wc -l)"
  [[ "$init_count" -eq "$IFS_MAX_INITS" ]]
}

configure_job() {
  local job="$1"
  IFS=: read -r FAMILY HORIZON SEED <<<"$job"

  case "$FAMILY" in
    detail)
      ARCH="flow_pp3_detail"
      PARAMETERS=14266733
      LAMBDA_HF=0.05
      LAMBDA_SPEC=0
      LAMBDA_BAND=0.02
      SPECTRAL_MASK=advected
      if [[ "$HORIZON" == 6 ]]; then
        EXP="exp_flow_pp3_detail_14m_6h_refinev1_s${SEED}_bs4"
      else
        EXP="exp_flow_pp3_detail_14m_12h_2017_19_refinev1_s${SEED}_bs4"
        SPECTRAL_MASK=all
      fi
      ;;
    refine)
      ARCH="flow_universal_latent_refine"
      PARAMETERS=13651208
      LAMBDA_HF=0.05
      LAMBDA_SPEC=0
      LAMBDA_BAND=0.02
      SPECTRAL_MASK=all
      if [[ "$HORIZON" == 6 ]]; then
        EXP="exp_flow_universal_latent_refine_14m_6h_s${SEED}_v1_bs4"
      else
        EXP="exp_flow_universal_latent_refine_14m_12h_2017_19_s${SEED}_v1_bs4"
      fi
      ;;
    flow_spectral)
      ARCH="flow_pp3"
      PARAMETERS=14260565
      LAMBDA_HF=0.05
      LAMBDA_SPEC=0.02
      LAMBDA_BAND=0
      SPECTRAL_MASK=advected
      if [[ "$HORIZON" == 6 ]]; then
        EXP="exp_flow_pp3_spectral_14m_6h_refinev1_s${SEED}_bs4"
      else
        EXP="exp_flow_pp3_spectral_14m_12h_2017_19_refinev1_s${SEED}_bs4"
      fi
      ;;
    flow_pixel)
      ARCH="flow_pp3"
      PARAMETERS=14260565
      LAMBDA_HF=0
      LAMBDA_SPEC=0
      LAMBDA_BAND=0
      SPECTRAL_MASK=all
      if [[ "$HORIZON" != 6 ]]; then
        echo "flow_pixel is defined only for the primary 6 h factorial" >&2
        return 2
      fi
      EXP="exp_flow_pp3_pixel_14m_6h_refinev1_s${SEED}_bs4"
      ;;
    dcae)
      ARCH="dcae_14m"
      PARAMETERS=14365049
      LAMBDA_HF=0
      LAMBDA_SPEC=0
      LAMBDA_BAND=0
      SPECTRAL_MASK=all
      if [[ "$HORIZON" == 6 ]]; then
        EXP="exp_weatherdcae_14m_6h_6yr_refinev1_s${SEED}_bs4"
      else
        EXP="exp_weatherdcae_14m_12h_2017_19_refinev1_s${SEED}_bs4"
      fi
      ;;
    dcae_spectral)
      ARCH="dcae_14m"
      PARAMETERS=14365049
      LAMBDA_HF=0.05
      LAMBDA_SPEC=0.02
      LAMBDA_BAND=0
      SPECTRAL_MASK=advected
      if [[ "$HORIZON" != 6 ]]; then
        echo "dcae_spectral is defined only for the primary 6 h factorial" >&2
        return 2
      fi
      EXP="exp_weatherdcae_spectral_14m_6h_refinev1_s${SEED}_bs4"
      ;;
    linear)
      ARCH="linear"
      PARAMETERS=0
      LAMBDA_HF=0
      LAMBDA_SPEC=0
      LAMBDA_BAND=0
      SPECTRAL_MASK=all
      EXP="linear_${HORIZON}h"
      ;;
    *)
      echo "unknown family=$FAMILY" >&2
      return 2
      ;;
  esac

  if [[ "$HORIZON" == 6 ]]; then
    YEARS=(2014 2015 2016 2017 2018 2019)
    TRAIN_TAUS=(1 3 5)
    EVAL_TAUS=(1 2 3 4 5)
    MAX_EPOCHS=8
    TRAIN_BATCHES=6568
    OPTIMIZER_STEPS=1642
    SAMPLES_TRAIN=4
    TOTAL_STEPS=13136
  elif [[ "$HORIZON" == 12 ]]; then
    YEARS=(2017 2018 2019)
    TRAIN_TAUS=(1 2 3 5 7 9 10 11)
    EVAL_TAUS=(1 2 3 4 5 6 7 8 9 10 11)
    MAX_EPOCHS=10
    TRAIN_BATCHES=4372
    OPTIMIZER_STEPS=1093
    SAMPLES_TRAIN=2
    TOTAL_STEPS=10930
  else
    echo "unsupported horizon=$HORIZON" >&2
    return 2
  fi

  OUT="$LOG_ROOT/$EXP"
  CHECKPOINT="$OUT/last.ckpt"
  RUN_LOG="$LOG_ROOT/$EXP.log"
  IFS_DIR="$EVAL_ROOT/${HORIZON}h"
  IFS_JSON="$IFS_DIR/$EXP.json"
}

checkpoint_complete() {
  [[ -s "$CHECKPOINT" ]] || return 1
  "$PY" - "$CHECKPOINT" "$ARCH" "$SEED" "$HORIZON" \
    "$MAX_EPOCHS" "$TOTAL_STEPS" "$TRAIN_BATCHES" "$OPTIMIZER_STEPS" \
    "$SAMPLES_TRAIN" "$LAMBDA_HF" "$LAMBDA_SPEC" "$LAMBDA_BAND" \
    "$SPECTRAL_MASK" "$PARAMETERS" "$TRAINER_SHA256" \
    "$FLOW_MODEL_SHA256" <<'PY'
import math
import sys
import torch

(
    path, arch, seed, horizon, max_epochs, total_steps, train_batches,
    optimizer_steps, samples_train, lambda_hf, lambda_spec, lambda_band,
    spectral_mask, parameters, trainer_hash, flow_hash,
) = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
hparams = checkpoint.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
state = checkpoint.get("state_dict", {})
horizon = int(horizon)

assert hparams.get("arch") == arch
assert int(hparams.get("training_seed", -1)) == int(seed)
assert int(checkpoint.get("epoch", -1)) + 1 >= int(max_epochs)
assert int(checkpoint.get("global_step", -1)) >= int(total_steps)
assert float(hparams.get("delta_t", -1)) == float(horizon)
assert protocol.get("seed") == int(seed)
assert protocol.get("train_years") == (
    list(range(2014, 2020)) if horizon == 6 else [2017, 2018, 2019]
)
assert protocol.get("val_years") == [2020]
assert protocol.get("train_tau_hours") == (
    [1, 3, 5] if horizon == 6 else [1, 2, 3, 5, 7, 9, 10, 11]
)
assert protocol.get("eval_tau_hours") == list(range(1, horizon))
assert protocol.get("global_effective_batch_size") == 16
assert protocol.get("batch_size_per_device") == 4
assert protocol.get("accumulate_grad_batches") == 4
assert protocol.get("train_batches_per_epoch") == int(train_batches)
assert protocol.get("optimizer_steps_per_epoch") == int(optimizer_steps)
assert protocol.get("samples_per_date_train") == int(samples_train)
assert protocol.get("samples_per_date_val") == 2
assert protocol.get("precision") == "bf16-mixed"
assert math.isclose(float(protocol.get("lambda_hf", -1)), float(lambda_hf))
assert math.isclose(float(protocol.get("lambda_spec", -1)), float(lambda_spec))
assert math.isclose(float(protocol.get("lambda_band", -1)), float(lambda_band))
assert protocol.get("spectral_mask_profile") == spectral_mask
assert protocol.get("loss_profile") == "uniform"
assert protocol.get("anchor_swap_probability") == 0.0
assert not any("teacher" in name or "distill" in name for name in state)
net_parameters = sum(
    value.numel() for name, value in state.items() if name.startswith("net.")
)
assert net_parameters == int(parameters)
hashes = hparams.get("training_code_sha256", {})
assert hashes.get("train_capacity_matched_6h.py") == trainer_hash
if arch.startswith("flow_"):
    assert hashes.get("weatherbridge_flow_model.py") == flow_hash
PY
}

checkpoint_resume_compatible() {
  [[ -s "$CHECKPOINT" ]] || return 1
  "$PY" - "$CHECKPOINT" "$ARCH" "$SEED" "$TRAINER_SHA256" <<'PY'
import sys
import torch

path, arch, seed, trainer_hash = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
hparams = checkpoint.get("hyper_parameters", {})
assert hparams.get("arch") == arch
assert int(hparams.get("training_seed", -1)) == int(seed)
assert hparams.get("training_code_sha256", {}).get(
    "train_capacity_matched_6h.py"
) == trainer_hash
PY
}

wait_for_external_experiment() {
  while pgrep -f -- "--exp_name ${EXP}( |$)" >/dev/null 2>&1; do
    echo "[$(date -Is)] wait external experiment=$EXP"
    sleep "$POLL_SECONDS"
  done
}

train_model() {
  checkpoint_complete && return 0
  wait_for_external_experiment
  checkpoint_complete && return 0

  local resume=()
  if [[ -s "$CHECKPOINT" ]]; then
    if ! checkpoint_resume_compatible; then
      echo "incompatible partial checkpoint=$CHECKPOINT" >&2
      return 2
    fi
    resume=(--ckpt_path "$CHECKPOINT")
  fi

  mkdir -p "$OUT"
  echo "[$(date -Is)] train exp=$EXP gpu=$GPU seed=$SEED horizon=$HORIZON"
  (
    cd "$TRAIN_SOURCE"
    export CUDA_VISIBLE_DEVICES="$GPU"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$TRAIN_SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/train/train_capacity_matched_6h.py \
      --arch "$ARCH" \
      --exp_name "$EXP" \
      --log_root "$LOG_ROOT" \
      --gpus 0 \
      --bs 4 \
      --val_bs 2 \
      --accumulate 4 \
      --workers 4 \
      --val_workers 2 \
      --release_memmap_pages \
      --precision bf16-mixed \
      --seed "$SEED" \
      --years "${YEARS[@]}" \
      --val_years 2020 \
      --max_epochs "$MAX_EPOCHS" \
      --lr 1e-4 \
      --warmup_steps 500 \
      --window_hours "$HORIZON" \
      --train_tau_subset "${TRAIN_TAUS[@]}" \
      --eval_tau "${EVAL_TAUS[@]}" \
      --samples_per_date_train "$SAMPLES_TRAIN" \
      --samples_per_date_val 2 \
      --lambda_hf_override "$LAMBDA_HF" \
      --lambda_spec_override "$LAMBDA_SPEC" \
      --lambda_band_override "$LAMBDA_BAND" \
      --spectral_mask_profile "$SPECTRAL_MASK" \
      --loss_profile uniform \
      --trainable_scope all \
      --anchor_swap_probability 0 \
      --train_batches_per_epoch "$TRAIN_BATCHES" \
      --ckpt_every_n_epochs "$MAX_EPOCHS" \
      --memmap_dir "$MEMMAP" \
      --static_path "$STATIC" \
      --stats_path "$STATS" \
      --surface_stats_path "$SURFACE_STATS" \
      "${resume[@]}"
  ) >>"$RUN_LOG" 2>&1 || return $?

  checkpoint_complete
}

ifs_artifact_complete() {
  [[ -s "$IFS_JSON" && -s "${IFS_JSON%.json}.paired.npz" ]] || return 1
  "$PY" - "$IFS_JSON" "$CHECKPOINT" "$FAMILY" "$HORIZON" \
    "$IFS_MAX_INITS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

json_path, checkpoint, family, horizon, max_inits = sys.argv[1:]
payload = json.loads(Path(json_path).read_text())
assert payload["protocol"]["delta_t_hours"] == int(horizon)
assert payload["protocol"]["max_inits"] == int(max_inits)
assert payload["provenance"]["forecast_anchors"]["init_count"] == int(max_inits)
forecast = payload["provenance"]["forecast_anchors"]
assert forecast["grid"]["latitude_order"] == "north_to_south"
assert forecast["archive_manifest"]["path"].endswith("forecast_archive_manifest.json")
model = payload["provenance"]["model"]
if family == "linear":
    assert model["kind"] == "linear"
else:
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    assert model["artifact"]["sha256"] == digest
PY
}

run_ifs_eval() {
  ifs_artifact_complete && return 0
  mkdir -p "$IFS_DIR"
  local eval_log="$IFS_DIR/$EXP.log"
  echo "[$(date -Is)] IFS eval exp=$EXP gpu=$GPU horizon=$HORIZON"
  local model_args
  if [[ "$FAMILY" == linear ]]; then
    model_args=(--model-blob linear --model-name "$EXP")
  else
    model_args=(--checkpoint "$CHECKPOINT" --arch "$ARCH" --model-name "$EXP")
  fi
  (
    cd "$EVAL_SOURCE"
    export CUDA_VISIBLE_DEVICES="$GPU"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$EVAL_SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/eval/batch_eval_forecast_anchor.py \
      --forecast-dir "$HRES" \
      --era5-memmap-dir "$MEMMAP" \
      "${model_args[@]}" \
      --out-json "$IFS_JSON" \
      --stats-path "$STATS" \
      --surface-stats-path "$SURFACE_STATS" \
      --static-path "$STATIC" \
      --device cuda \
      --delta-t-hours "$HORIZON" \
      --taus "${EVAL_TAUS[@]}" \
      --max-inits "$IFS_MAX_INITS"
  ) >>"$eval_log" 2>&1 || return $?
  ifs_artifact_complete
}

job_id() {
  echo "${1//:/_}"
}

claim_next_job() {
  local job id
  exec 7>"$STATE_DIR/claims.lock"
  flock 7
  for job in "${JOBS[@]}"; do
    id="$(job_id "$job")"
    if [[ ! -e "$STATE_DIR/$id.done" \
          && ! -e "$STATE_DIR/$id.failed" \
          && ! -e "$STATE_DIR/$id.claimed" ]]; then
      printf '%s\n' "$$" >"$STATE_DIR/$id.claimed"
      flock -u 7
      echo "$job"
      return 0
    fi
  done
  flock -u 7
  return 1
}

wait_for_gpu() {
  local free util
  while true; do
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
    if [[ "$free" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      sleep 15
      free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
      util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
      [[ "$free" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]] && return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

run_job() {
  local job="$1"
  configure_job "$job" || return $?
  if [[ "$FAMILY" != linear ]]; then
    local exp_lock="$LOG_ROOT/$EXP.queue.lock"
    exec 6>"$exp_lock"
    flock 6
    local attempt status=1
    for attempt in 1 2; do
      train_model
      status=$?
      if [[ "$status" -eq 0 ]]; then
        status=0
        break
      fi
      echo "[$(date -Is)] train failed exp=$EXP attempt=$attempt status=$status"
      sleep "$POLL_SECONDS"
    done
    [[ "$status" -eq 0 ]] || return "$status"
  fi
  if [[ "${RUN_IFS:-1}" == 1 ]]; then
    run_ifs_eval
  fi
}

worker() {
  GPU="$1"
  exec 5>"$LOG_ROOT/.upr_lite_gpu${GPU}.lock"

  local job id status
  while job="$(claim_next_job)"; do
    id="$(job_id "$job")"
    echo "[$(date -Is)] worker gpu=$GPU waiting for exclusive device"
    flock 5
    wait_for_gpu
    echo "[$(date -Is)] worker gpu=$GPU ready"
    echo "[$(date -Is)] start job=$job gpu=$GPU"
    if run_job "$job"; then
      touch "$STATE_DIR/$id.done"
      rm -f "$STATE_DIR/$id.failed"
      echo "[$(date -Is)] done job=$job gpu=$GPU"
    else
      status=$?
      printf '%s\n' "$status" >"$STATE_DIR/$id.failed"
      echo "[$(date -Is)] failed job=$job gpu=$GPU status=$status"
    fi
    rm -f "$STATE_DIR/$id.claimed"
    flock -u 5
  done
  echo "[$(date -Is)] worker gpu=$GPU exhausted queue"
}

verify_inputs
for claim in "$STATE_DIR"/*.claimed; do
  [[ -e "$claim" ]] || continue
  rm -f "$claim"
done

{
  echo "queue_source_sha256=$(sha256sum "$0" | awk '{print $1}')"
  echo "train_source=$TRAIN_SOURCE"
  echo "trainer_sha256=$TRAINER_SHA256"
  echo "eval_source=$EVAL_SOURCE"
  echo "ifs_archive=$HRES"
  echo "ifs_max_inits=$IFS_MAX_INITS"
  printf 'jobs=%s\n' "${JOBS[*]}"
} >"$STATE_DIR/protocol.txt"

status=0
pids=()
for worker_gpu in $WORKER_GPUS; do
  worker "$worker_gpu" &
  pids+=("$!")
done
for worker_pid in "${pids[@]}"; do
  wait "$worker_pid" || status=1
done

failed_count="$(find "$STATE_DIR" -maxdepth 1 -type f -name '*.failed' | wc -l)"
done_count="$(find "$STATE_DIR" -maxdepth 1 -type f -name '*.done' | wc -l)"
echo "[$(date -Is)] queue terminal done=$done_count failed=$failed_count total=${#JOBS[@]}"
[[ "$status" -eq 0 && "$failed_count" -eq 0 && "$done_count" -eq "${#JOBS[@]}" ]]
