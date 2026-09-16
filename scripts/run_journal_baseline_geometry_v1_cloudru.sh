#!/usr/bin/env bash
# Re-evaluate paper baselines with the corrected WB2 latitude geometry.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
RUNTIME="${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"
LEGACY="${LEGACY:-/home/jovyan/dsuhoi/weather_time_interpolation}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
GPU_LOCK_ROOT="${GPU_LOCK_ROOT:-$LOG_ROOT}"
DATA="${DATA:-/tmp/wb2_0p5_cache}"
CLIM="${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CACHE="${CACHE:-/tmp/climatology_eval_cache_24ch}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
MAX_UTIL="${MAX_UTIL:-5}"
POLL_SECONDS="${POLL_SECONDS:-120}"
HORIZONS="${HORIZONS:-6 12}"
WORKER_ID="${WORKER_ID:-primary}"
FINALIZE="${FINALIZE:-1}"
YEARS_6="${YEARS_6:-2020 2021}"
YEARS_12="${YEARS_12:-2020}"
OUT="$RUNTIME/metrics/journal_unified"
MARKER="$OUT/.baseline_geometry_v1_complete"
LOG="${LOG:-$LOG_ROOT/journal_baseline_geometry_v1.log}"

if [[ ! "$WORKER_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "invalid WORKER_ID: $WORKER_ID" >&2
  exit 2
fi

mkdir -p "$OUT" "$CACHE" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
exec 8>"$LOG_ROOT/.journal_baseline_geometry_v1.${WORKER_ID}.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] baseline geometry queue already active"
  exit 0
fi

export PYTHONPATH="$PWD:$LEGACY:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim:${PYTHONPATH:-}"
export PATH="/home/jovyan/.mlspace/envs/ai_scientist/bin:/usr/local/bin:/usr/bin:/bin"

GPU=""
GPU_LOCK_FD=""
LAST_GPU="none"

acquire_gpu() {
  while [[ -z "$GPU" ]]; do
    for candidate in $GPU_CANDIDATES; do
      exec {candidate_fd}>"$GPU_LOCK_ROOT/.upr_lite_gpu${candidate}.lock"
      if ! flock -n "$candidate_fd"; then
        exec {candidate_fd}>&-
        continue
      fi
      free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
      util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
      if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
        GPU="$candidate"
        LAST_GPU="$candidate"
        GPU_LOCK_FD="$candidate_fd"
        export CUDA_VISIBLE_DEVICES="$GPU"
        break
      fi
      flock -u "$candidate_fd"
      exec {candidate_fd}>&-
    done
    if [[ -z "$GPU" ]]; then
      echo "[$(date -Is)] waiting for a free A100"
      sleep "$POLL_SECONDS"
    fi
  done
}

release_gpu() {
  if [[ -n "$GPU" ]]; then
    flock -u "$GPU_LOCK_FD"
    exec {GPU_LOCK_FD}>&-
    GPU=""
    GPU_LOCK_FD=""
  fi
}

wait_checkpoint() {
  local checkpoint="$1"
  while [[ ! -s "$checkpoint" ]]; do
    echo "[$(date -Is)] waiting for checkpoint $checkpoint"
    sleep "$POLL_SECONDS"
  done
}

has_horizon() {
  [[ " $HORIZONS " == *" $1 "* ]]
}

artifact_valid() {
  local horizon="$1" year="$2" eval_tau="$3" name="$4" checkpoint="$5"
  "$PY" tools/eval/eval_artifact_status.py \
    "$OUT/${horizon}h_${year}/${name}.json" \
    --checkpoint "$checkpoint" --required-taus "$eval_tau" \
    --acc-mode enabled --quiet
}

evaluate_model() {
  local horizon="$1"
  local samples_per_date="$2"
  local seen_tau="$3"
  local unseen_tau="$4"
  local eval_tau="$5"
  local name="$6"
  local checkpoint="$7"
  local model_env="${8:-}"
  local years

  if [[ "$horizon" -eq 6 ]]; then
    years="$YEARS_6"
  else
    years="$YEARS_12"
  fi

  wait_checkpoint "$checkpoint"
  for year in $years; do
    local out="$OUT/${horizon}h_${year}"
    local result="$out/${name}.json"
    if artifact_valid "$horizon" "$year" "$eval_tau" "$name" "$checkpoint"; then
      echo "[$(date -Is)] validated existing $result"
      continue
    fi

    mkdir -p "$out"
    acquire_gpu
    echo "[$(date -Is)] evaluate $name horizon=${horizon}h year=$year GPU=$GPU"
    "$PY" -u tools/eval/batch_eval_12h_memmap_fast.py \
      --memmap-dir "$DATA" --test-year "$year" --climatology "$CLIM" \
      --climatology-cache-dir "$CACHE" --lazy-climatology \
      --stats-path data/json_stats_0p5.nc \
      --surface-stats-path data/surface_stats_0p5.json \
      --static-path data/static_features_0p5.pt \
      --models "${name}:${checkpoint}:${model_env}" --out-dir "$out" \
      --paper-tag "journal_${horizon}h_${year}_full_year_geometry_v1" \
      --batch-size 8 --num-workers 0 \
      --samples-per-date "$samples_per_date" \
      --full-year --max-tau-hours "$horizon" --eval-hours "$eval_tau" \
      --seen-tau "$seen_tau" --unseen-tau "$unseen_tau" \
      --keep-n-channels 24 --proper-rmse --save-window-metrics \
      --device cuda:0
    artifact_valid "$horizon" "$year" "$eval_tau" "$name" "$checkpoint"
  done
}

if has_horizon 6; then
  evaluate_model 6 4 "1,3,5" "2,4" "1,2,3,4,5" \
    fuxi_24ch_6yr_ep8 \
    "$LEGACY/logs/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt"
  evaluate_model 6 4 "1,3,5" "2,4" "1,2,3,4,5" \
    modafno_24ch_6yr_ep8 \
    "$LEGACY/logs/exp_modafno_24ch_6yr/epoch=7-step=35032.ckpt" \
    "MODAFNO_INP_H=360 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
  evaluate_model 6 4 "1,3,5" "2,4" "1,2,3,4,5" \
    sdyff_24ch_6yr_ep8 \
    "$LEGACY/logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt" \
    "SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"
  evaluate_model 6 4 "1,3,5" "2,4" "1,2,3,4,5" \
    atm_vfi_6yr_ep8_matched \
    "$LOG_ROOT/exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_s202707_protocol_v2/last.ckpt"
fi

if has_horizon 12; then
  evaluate_model 12 2 "1,2,3,5,7,9,10,11" "4,6,8" \
    "1,2,3,4,5,6,7,8,9,10,11" fuxi_3yr_ep10 \
    "$LOG_ROOT/exp_fuxi_24ch_12h_2017_19_lr1e4/epoch=9-step=5470.ckpt"
  evaluate_model 12 2 "1,2,3,5,7,9,10,11" "4,6,8" \
    "1,2,3,4,5,6,7,8,9,10,11" modafno_3yr_ep10 \
    "$LEGACY/logs/_12h_migrated/modafno_12h.ckpt" \
    "MODAFNO_INP_H=362 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
  evaluate_model 12 2 "1,2,3,5,7,9,10,11" "4,6,8" \
    "1,2,3,4,5,6,7,8,9,10,11" sdyff_3yr_ep10 \
    "$LEGACY/logs/_12h_migrated/sdyff_12h.ckpt" \
    "SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"
  evaluate_model 12 2 "1,2,3,5,7,9,10,11" "4,6,8" \
    "1,2,3,4,5,6,7,8,9,10,11" atm_vfi_3yr_ep10_matched \
    "$LOG_ROOT/exp_atmvfi_24ch_12h_2017_19_oddskip_lr1e4_protocol_v3/last.ckpt"
fi

release_gpu

if [[ "$FINALIZE" -ne 1 ]]; then
  echo "[$(date -Is)] baseline geometry worker $WORKER_ID complete"
  exit 0
fi

while ! (
  for year in 2020 2021; do
    artifact_valid 6 "$year" "1,2,3,4,5" fuxi_24ch_6yr_ep8 \
      "$LEGACY/logs/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt" &&
    artifact_valid 6 "$year" "1,2,3,4,5" modafno_24ch_6yr_ep8 \
      "$LEGACY/logs/exp_modafno_24ch_6yr/epoch=7-step=35032.ckpt" &&
    artifact_valid 6 "$year" "1,2,3,4,5" sdyff_24ch_6yr_ep8 \
      "$LEGACY/logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt" &&
    artifact_valid 6 "$year" "1,2,3,4,5" atm_vfi_6yr_ep8_matched \
      "$LOG_ROOT/exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_s202707_protocol_v2/last.ckpt" || exit 1
  done
  artifact_valid 12 2020 "1,2,3,4,5,6,7,8,9,10,11" fuxi_3yr_ep10 \
    "$LOG_ROOT/exp_fuxi_24ch_12h_2017_19_lr1e4/epoch=9-step=5470.ckpt" &&
  artifact_valid 12 2020 "1,2,3,4,5,6,7,8,9,10,11" modafno_3yr_ep10 \
    "$LEGACY/logs/_12h_migrated/modafno_12h.ckpt" &&
  artifact_valid 12 2020 "1,2,3,4,5,6,7,8,9,10,11" sdyff_3yr_ep10 \
    "$LEGACY/logs/_12h_migrated/sdyff_12h.ckpt" &&
  artifact_valid 12 2020 "1,2,3,4,5,6,7,8,9,10,11" atm_vfi_3yr_ep10_matched \
    "$LOG_ROOT/exp_atmvfi_24ch_12h_2017_19_oddskip_lr1e4_protocol_v3/last.ckpt"
); do
  echo "[$(date -Is)] waiting for the other baseline geometry worker"
  sleep "$POLL_SECONDS"
done

"$PY" - "$MARKER" "$LAST_GPU" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
sources = (
    Path("scripts/run_journal_baseline_geometry_v1_cloudru.sh"),
    Path("tools/eval/batch_eval_12h_memmap.py"),
    Path("tools/eval/batch_eval_12h_memmap_fast.py"),
    Path("tools/eval/eval_artifact_status.py"),
    Path("weather_time_interp/normalization.py"),
)
payload = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "gpu": None if sys.argv[2] == "none" else int(sys.argv[2]),
    "horizons": [6, 12],
    "years": {"6h": [2020, 2021], "12h": [2020]},
    "source_sha256": {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in sources
    },
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, path)
PY

echo "[$(date -Is)] baseline geometry re-evaluation complete"
