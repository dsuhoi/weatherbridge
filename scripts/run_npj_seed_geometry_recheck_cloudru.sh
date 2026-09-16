#!/usr/bin/env bash
# Re-evaluate frozen seed checkpoints with the verified WB2 row geometry.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
MEMMAP=${MEMMAP:-/tmp/wb2_0p5_cache}
HRES=${HRES:-/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_hres_2021_canonical_v3}
STATIC=${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}
STATS=${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}
SURFACE_STATS=${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}
POLL_SECONDS=${POLL_SECONDS:-120}
WORKER_GPUS=${WORKER_GPUS:-"0 1"}
IFS_MAX_INITS=${IFS_MAX_INITS:-16}
HRES_EXPECTED_INITS=${HRES_EXPECTED_INITS:-16}
EVAL_ROOT="$RUNTIME/metrics/npj_seed_ifs_hres_2021_v3"
STATE_DIR=${STATE_DIR:-$EVAL_ROOT/geometry_v3/state}
MARKER=${MARKER:-$EVAL_ROOT/geometry_v3/primary.complete}
LOG=${LOG:-$LOG_ROOT/npj_seed_geometry_v3.queue.log}
PRIMARY_STATES=(
  "$LOG_ROOT/npj_seed_replicates_v2.state"
  "$LOG_ROOT/npj_seed_replicates_v1.state"
)

DEFAULT_JOBS=()
for family in refine flow_spectral dcae; do
  for horizon in 6 12; do
    for seed in 202707 202708 202709; do
      DEFAULT_JOBS+=("$family:$horizon:$seed")
    done
  done
done
DEFAULT_JOBS+=("detail:6:202707" "detail:12:202707")
if [[ -n "${JOBS_OVERRIDE:-}" ]]; then
  read -r -a JOBS <<<"$JOBS_OVERRIDE"
else
  JOBS=("${DEFAULT_JOBS[@]}")
fi

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$STATE_DIR" "$(dirname "$MARKER")"
exec 8>"$STATE_DIR/launch.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] geometry recheck already active" | tee -a "$LOG"
  exit 0
fi
exec > >(tee -a "$LOG") 2>&1

SOURCE_FILES=(
  scripts/run_npj_seed_geometry_recheck_cloudru.sh
  tools/eval/batch_eval_forecast_anchor.py
  tools/data/build_hres_forecast_memmap.py
  tools/eval/capmatched_loader.py
  tools/train/checkpoint_status.py
  tools/train/train_capacity_matched_6h.py
  tools/train/training_protocol.py
  weather_time_interp/grid.py
  weather_time_interp/model/weatherbridge_flow_model.py
  weather_time_interp/model/dcae_adaln_model.py
  weather_time_interp/normalization.py
)
SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
if [[ -n "${EXPECTED_SOURCE_SHA:-}" && "$SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]]; then
  echo "source hash mismatch: $SOURCE_SHA != $EXPECTED_SOURCE_SHA" >&2
  exit 2
fi
printf '%s  %s\n' "$SOURCE_SHA" "${SOURCE_FILES[*]}" >"$STATE_DIR/source.sha256"

verify_source_snapshot() {
  local current_sha
  current_sha=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
  if [[ "$current_sha" != "$SOURCE_SHA" ]]; then
    echo "source changed while geometry recheck was active: $current_sha != $SOURCE_SHA" >&2
    exit 2
  fi
}

wait_for_hres_archive() {
  local manifest="$HRES/forecast_archive_manifest.json"
  while [[ ! -s "$manifest" ]]; do
    echo "[$(date -Is)] waiting for canonical HRES archive $manifest"
    sleep "$POLL_SECONDS"
  done
  "$PY" - "$HRES" "$HRES_EXPECTED_INITS" <<'PY'
import sys
from pathlib import Path

from tools.eval.batch_eval_forecast_anchor import (
    _forecast_manifest_provenance,
)

root = Path(sys.argv[1])
expected = int(sys.argv[2])
init_files = sorted(root.glob("init_*.bin"))
if len(init_files) != expected:
    raise SystemExit(
        f"canonical HRES archive has {len(init_files)} inits, expected {expected}"
    )
provenance = _forecast_manifest_provenance(root, init_files)
if int(provenance["init_count"]) != expected:
    raise SystemExit("canonical HRES provenance has the wrong init count")
print(
    "validated canonical HRES archive "
    f"inits={expected} manifest={provenance['archive_manifest']['sha256']}"
)
PY
}

primary_job_done() {
  local id="$1" state
  for state in "${PRIMARY_STATES[@]}"; do
    [[ -e "$state/$id.done" ]] && return 0
  done
  return 1
}

if [[ -z "${JOBS_OVERRIDE:-}" ]]; then
  for family in refine flow_spectral dcae; do
    for horizon in 6 12; do
      for seed in 202707 202708 202709; do
        id="${family}_${horizon}_${seed}"
        while ! primary_job_done "$id"; do
          echo "[$(date -Is)] waiting for trained seed $id"
          sleep "$POLL_SECONDS"
        done
      done
    done
  done
fi
wait_for_hres_archive
verify_source_snapshot

configure_job() {
  local job="$1"
  IFS=: read -r FAMILY HORIZON SEED <<<"$job"
  case "$FAMILY" in
    detail)
      ARCH=flow_pp3_detail
      LAMBDA_HF=0.05
      LAMBDA_SPEC=0
      LAMBDA_BAND=0.02
      infix=$([[ "$HORIZON" == 12 ]] && echo _2017_19 || true)
      EXP="exp_flow_pp3_detail_14m_${HORIZON}h${infix}_refinev1_s${SEED}_bs4"
      ;;
    refine)
      ARCH=flow_universal_latent_refine
      LAMBDA_HF=0.05
      LAMBDA_SPEC=0
      LAMBDA_BAND=0.02
      infix=$([[ "$HORIZON" == 12 ]] && echo _2017_19 || true)
      EXP="exp_flow_universal_latent_refine_14m_${HORIZON}h${infix}_s${SEED}_v1_bs4"
      ;;
    flow_spectral)
      ARCH=flow_pp3
      LAMBDA_HF=0.05
      LAMBDA_SPEC=0.02
      LAMBDA_BAND=0
      infix=$([[ "$HORIZON" == 12 ]] && echo _2017_19 || true)
      EXP="exp_flow_pp3_spectral_14m_${HORIZON}h${infix}_refinev1_s${SEED}_bs4"
      ;;
    dcae)
      ARCH=dcae_14m
      LAMBDA_HF=0
      LAMBDA_SPEC=0
      LAMBDA_BAND=0
      infix=$([[ "$HORIZON" == 6 ]] && echo _6yr || echo _2017_19)
      EXP="exp_weatherdcae_14m_${HORIZON}h${infix}_refinev1_s${SEED}_bs4"
      ;;
    *) echo "unknown family $FAMILY" >&2; return 2 ;;
  esac
  CHECKPOINT="$LOG_ROOT/$EXP/last.ckpt"
  OUT_DIR="$EVAL_ROOT/${HORIZON}h"
  OUT_JSON="$OUT_DIR/$EXP.json"
  if [[ "$HORIZON" == 6 ]]; then
    TAUS=(1 2 3 4 5)
    TRAIN_YEARS=2014,2015,2016,2017,2018,2019
    TRAIN_TAUS=1,3,5
    MIN_EPOCHS=8
    TOTAL_STEPS=13136
    TRAIN_BATCHES=6568
    OPTIMIZER_STEPS=1642
    SAMPLES_PER_DATE_TRAIN=4
  else
    TAUS=(1 2 3 4 5 6 7 8 9 10 11)
    TRAIN_YEARS=2017,2018,2019
    TRAIN_TAUS=1,2,3,5,7,9,10,11
    MIN_EPOCHS=10
    TOTAL_STEPS=10930
    TRAIN_BATCHES=4372
    OPTIMIZER_STEPS=1093
    SAMPLES_PER_DATE_TRAIN=2
  fi
  EVAL_TAUS=$(printf '%s,' "${TAUS[@]}")
  EVAL_TAUS=${EVAL_TAUS%,}
  SPECTRAL_MASK=all
  if [[ "$FAMILY" == flow_spectral || (
        "$FAMILY" == detail && "$HORIZON" == 6
      ) ]]; then
    SPECTRAL_MASK=advected
  fi
}

checkpoint_current() {
  "$PY" tools/train/checkpoint_status.py "$CHECKPOINT" \
    --quiet \
    --min-epochs "$MIN_EPOCHS" \
    --expected-arch "$ARCH" \
    --expected-total-steps "$TOTAL_STEPS" \
    --expected-delta-t "$HORIZON" \
    --min-global-step "$TOTAL_STEPS" \
    --require-training-protocol \
    --expected-train-years "$TRAIN_YEARS" \
    --expected-val-years 2020 \
    --expected-train-taus "$TRAIN_TAUS" \
    --expected-eval-taus "$EVAL_TAUS" \
    --expected-seed "$SEED" \
    --expected-effective-batch-size 16 \
    --expected-batch-size-per-device 4 \
    --expected-accumulate-grad-batches 4 \
    --expected-train-batches-per-epoch "$TRAIN_BATCHES" \
    --expected-optimizer-steps-per-epoch "$OPTIMIZER_STEPS" \
    --expected-samples-per-date-train "$SAMPLES_PER_DATE_TRAIN" \
    --expected-samples-per-date-val 2 \
    --expected-lambda-hf "$LAMBDA_HF" \
    --expected-lambda-spec "$LAMBDA_SPEC" \
    --expected-lambda-band "$LAMBDA_BAND" \
    --expected-spectral-mask-profile "$SPECTRAL_MASK" \
    --expected-loss-profile uniform \
    --expected-trainable-scope all \
    --expected-anchor-swap-probability 0 \
    --expected-precision bf16-mixed \
    --expected-trainer-sha256 \
      880be34d0f72e0d0cbed3159269b681a65238935fa525efec00d1f841cbeb14d
}

artifact_current() {
  [[ -s "$OUT_JSON" && -s "${OUT_JSON%.json}.paired.npz" ]] || return 1
  "$PY" - "$OUT_JSON" "$CHECKPOINT" "$HORIZON" "$SOURCE_SHA" "$HRES" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from tools.eval.batch_eval_forecast_anchor import (
    _forecast_manifest_provenance,
)

artifact, checkpoint, horizon, source_sha, hres = sys.argv[1:]
payload = json.loads(Path(artifact).read_text())
protocol = payload["protocol"]
assert payload["schema_version"] == 2
assert protocol["delta_t_hours"] == int(horizon)
assert protocol["latitude_grid"] == "wb2_0p25_2x2_block_average_v1"
assert protocol["area_weighting"] == "spherical_latitude_strip_area"
forecast = payload["provenance"]["forecast_anchors"]
hres_path = Path(hres).resolve()
assert Path(forecast["path"]).resolve() == hres_path
assert forecast["grid"]["latitude_order"] == "north_to_south"
assert forecast["archive_manifest"]["path"] == str(
    hres_path / "forecast_archive_manifest.json"
)
current_forecast = _forecast_manifest_provenance(
    hres_path,
    sorted(hres_path.glob("init_*.bin")),
)
assert forecast["manifest_sha256"] == current_forecast["manifest_sha256"]
assert forecast["archive_manifest"]["sha256"] == current_forecast[
    "archive_manifest"
]["sha256"]
assert forecast["init_count"] == current_forecast["init_count"]
assert payload["provenance"]["model"]["artifact"]["sha256"] == hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
assert payload["paired_artifact"]["n_windows"] > 0
assert payload["paired_artifact"]["window_index_sha256"]
code = payload["provenance"]["evaluation_code"]
assert "grid.py" in code
for name, record in code.items():
    source = Path(record["path"])
    assert source.is_file(), (name, source)
    assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
for record in (
    payload["provenance"]["evaluator"],
    payload["provenance"]["model"]["artifact"],
    payload["paired_artifact"],
):
    source = Path(record["path"])
    assert source.is_file(), source
    assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
PY
}

claim_next() {
  local job id
  exec 7>"$STATE_DIR/claims.lock"
  flock 7
  for job in "${JOBS[@]}"; do
    id=${job//:/_}
    if [[ ! -e "$STATE_DIR/$id.done" && ! -e "$STATE_DIR/$id.claimed" ]]; then
      printf '%s\n' "$$" >"$STATE_DIR/$id.claimed"
      flock -u 7
      echo "$job"
      return 0
    fi
  done
  flock -u 7
  return 1
}

worker() {
  local GPU="$1" job id
  exec 6>"$LOG_ROOT/.upr_lite_gpu${GPU}.lock"
  while job=$(claim_next); do
    id=${job//:/_}
    configure_job "$job"
    while [[ ! -s "$CHECKPOINT" ]]; do
      echo "[$(date -Is)] waiting for checkpoint $CHECKPOINT"
      sleep "$POLL_SECONDS"
    done
    if ! checkpoint_current; then
      echo "[$(date -Is)] reject incompatible checkpoint $CHECKPOINT" >&2
      return 2
    fi
    verify_source_snapshot
    flock 6
    export CUDA_VISIBLE_DEVICES="$GPU"
    mkdir -p "$OUT_DIR"
    if ! artifact_current; then
      echo "[$(date -Is)] geometry recheck job=$job gpu=$GPU"
      "$PY" -u tools/eval/batch_eval_forecast_anchor.py \
        --forecast-dir "$HRES" --era5-memmap-dir "$MEMMAP" \
        --checkpoint "$CHECKPOINT" --arch "$ARCH" --model-name "$EXP" \
        --out-json "$OUT_JSON" --stats-path "$STATS" \
        --surface-stats-path "$SURFACE_STATS" --static-path "$STATIC" \
        --device cuda --delta-t-hours "$HORIZON" --taus "${TAUS[@]}" \
        --max-inits "$IFS_MAX_INITS"
    fi
    artifact_current
    verify_source_snapshot
    touch "$STATE_DIR/$id.done"
    rm -f "$STATE_DIR/$id.claimed"
    flock -u 6
  done
}

# Source or checkpoint changes invalidate prior completion markers. Reclaim
# those jobs before workers inspect the queue so a restart is a real audit.
verify_source_snapshot
for job in "${JOBS[@]}"; do
  id=${job//:/_}
  if [[ -e "$STATE_DIR/$id.done" ]]; then
    configure_job "$job"
    if ! artifact_current; then
      echo "[$(date -Is)] reclaiming stale geometry artifact job=$job"
      rm -f "$STATE_DIR/$id.done"
    fi
  fi
done

for claim in "$STATE_DIR"/*.claimed; do
  [[ -e "$claim" ]] && rm -f "$claim"
done
pids=()
for gpu in $WORKER_GPUS; do
  worker "$gpu" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

for job in "${JOBS[@]}"; do
  [[ -e "$STATE_DIR/${job//:/_}.done" ]]
done
verify_source_snapshot
"$PY" - "$MARKER" "$SOURCE_SHA" "${JOBS[*]}" "${SOURCE_FILES[@]}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

marker, source_sha, jobs = sys.argv[1:4]
source_files = [Path(value) for value in sys.argv[4:]]
payload = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "source_sha256": source_sha,
    "source_files_sha256": {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in source_files
    },
    "jobs": jobs.split(),
}
target = Path(marker)
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, target)
PY
echo "[$(date -Is)] seed geometry recheck complete"
