#!/usr/bin/env bash
# Re-evaluate the manuscript baselines with the same grid, index, and metric
# implementation used for the canonical WeatherBridge comparisons.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
DATA=${DATA:-/tmp/wb2_0p5_cache}
POST_DATA=${POST_DATA:-/home/jovyan/shares/SR006.nfs2/dsuhoi/postselection_2022_memmap}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
GPU=${GPU:-1}
MIN_FREE_GIB=${MIN_FREE_GIB:-30}
OUT="$RUNTIME/metrics/corrected_baselines_v1"
LOG="$LOG_ROOT/corrected_baselines_v1.log"

STATIC=/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt
STATS=/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc
SURFACE_STATS=/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json
CLIM=/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr
CACHE=/tmp/wti_climatology_cache_corrected_baselines_v1

SWIN_6=/home/jovyan/dsuhoi/weather_time_interpolation/logs/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt
MODAFNO_6=/home/jovyan/dsuhoi/weather_time_interpolation/logs/exp_modafno_24ch_6yr/epoch=7-step=35032.ckpt
SDYFF_6=/home/jovyan/dsuhoi/weather_time_interpolation/logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt
PIXELATTN_6="$LOG_ROOT/exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_matched/last.ckpt"

SWIN_12="$LOG_ROOT/exp_fuxi_24ch_12h_2017_19_lr1e4/epoch=9-step=5470.ckpt"
MODAFNO_12=/home/jovyan/dsuhoi/weather_time_interpolation/logs/_12h_migrated/modafno_12h.ckpt
SDYFF_12=/home/jovyan/dsuhoi/weather_time_interpolation/logs/_12h_migrated/sdyff_12h.ckpt
PIXELATTN_12="$LOG_ROOT/exp_atmvfi_24ch_12h_2017_19_oddskip_lr1e4_protocol_v3/last.ckpt"

SOURCE_FILES=(
  scripts/run_corrected_baselines_v1_cloudru.sh
  tools/eval/batch_eval_12h_memmap.py
  tools/eval/eval_artifact_status.py
  tools/eval/align_window_artifact.py
  tools/eval/paired_block_bootstrap.py
  tools/eval/hard_window_block_bootstrap.py
  tools/eval/paired_aux_block_bootstrap.py
  tools/eval/physical_block_bootstrap.py
  weather_time_interp/grid.py
  weather_time_interp/eval_runner.py
  weather_time_interp/normalization.py
  weather_time_interp/metrics/physical_consistency.py
)

cd "$SOURCE"
mkdir -p "$OUT" "$CACHE"
exec 8>"$LOG_ROOT/.corrected_baselines_v1.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] corrected baseline queue already active"
  exit 0
fi
exec > >(tee -a "$LOG") 2>&1

SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
verify_source_snapshot() {
  local current
  current=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
  if [[ "$current" != "$SOURCE_SHA" ]]; then
    echo "source changed during corrected baseline evaluation" >&2
    exit 2
  fi
}

CHECKPOINTS=(
  "$SWIN_6" "$MODAFNO_6" "$SDYFF_6" "$PIXELATTN_6"
  "$SWIN_12" "$MODAFNO_12" "$SDYFF_12" "$PIXELATTN_12"
)
for path in \
  "$DATA/wb2_2020.bin" "$DATA/wb2_2020.json" \
  "$DATA/wb2_2021.bin" "$DATA/wb2_2021.json" \
  "$POST_DATA/wb2_2022.bin" "$POST_DATA/wb2_2022.json" \
  "$STATIC" "$STATS" "$SURFACE_STATS"; do
  [[ -s "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done
for checkpoint in "${CHECKPOINTS[@]}"; do
  [[ -s "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 2; }
done

free_kib=$(df -Pk "$RUNTIME" | awk 'NR == 2 {print $4}')
if (( free_kib < MIN_FREE_GIB * 1024 * 1024 )); then
  echo "less than ${MIN_FREE_GIB} GiB free at $RUNTIME" >&2
  exit 2
fi

exec 7>"$LOG_ROOT/.upr_lite_gpu${GPU}.lock"
flock 7
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
export MODAFNO_INP_W=720
export MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720
export SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0

configure_horizon() {
  local horizon="$1"
  if [[ "$horizon" == 6 ]]; then
    export MODAFNO_INP_H=360
  else
    export MODAFNO_INP_H=362
  fi
}

checkpoint_for() {
  local horizon="$1" model="$2"
  case "$horizon:$model" in
    6:swinv2) printf '%s' "$SWIN_6" ;;
    6:modafno) printf '%s' "$MODAFNO_6" ;;
    6:sdyff) printf '%s' "$SDYFF_6" ;;
    6:pixelattn_vfi) printf '%s' "$PIXELATTN_6" ;;
    12:swinv2) printf '%s' "$SWIN_12" ;;
    12:modafno) printf '%s' "$MODAFNO_12" ;;
    12:sdyff) printf '%s' "$SDYFF_12" ;;
    12:pixelattn_vfi) printf '%s' "$PIXELATTN_12" ;;
    *) return 2 ;;
  esac
}

batch_for() {
  local horizon="$1" model="$2"
  case "$horizon:$model" in
    6:swinv2|6:pixelattn_vfi) printf '8' ;;
    6:modafno|6:sdyff) printf '4' ;;
    12:swinv2|12:pixelattn_vfi) printf '4' ;;
    12:modafno|12:sdyff) printf '2' ;;
    *) return 2 ;;
  esac
}

output_valid() {
  local horizon="$1" directory="$2" model="$3" allow_economy="$4"
  local required checkpoint economy_args=()
  if [[ "$horizon" == 6 ]]; then
    required=1,2,3,4,5
  else
    required=1,2,3,4,5,6,7,8,9,10,11
  fi
  [[ "$allow_economy" == 1 ]] && economy_args+=(--allow-economy)
  checkpoint=$(checkpoint_for "$horizon" "$model")
  "$PY" tools/eval/eval_artifact_status.py \
    "$directory/$model.json" --checkpoint "$checkpoint" \
    --required-taus "$required" --acc-mode enabled \
    --require-physical-metrics --require-temporal-metrics \
    "${economy_args[@]}" --quiet
}

validate_outputs() {
  local horizon="$1" directory="$2" allow_economy="$3"
  local model
  for model in swinv2 modafno sdyff pixelattn_vfi; do
    output_valid "$horizon" "$directory" "$model" "$allow_economy" || return 1
  done
}

run_smoke() {
  local horizon="$1" eval_hours seen held out model checkpoint batch
  configure_horizon "$horizon"
  out="$OUT/smoke/${horizon}h"
  mkdir -p "$out"
  if [[ "$horizon" == 6 ]]; then
    eval_hours=1,2,3,4,5; seen=1,3,5; held=2,4
  else
    eval_hours=1,2,3,4,5,6,7,8,9,10,11
    seen=1,2,3,5,7,9,10,11; held=4,6,8
  fi
  for model in swinv2 modafno sdyff pixelattn_vfi; do
    if output_valid "$horizon" "$out" "$model" 1; then
      echo "[$(date -Is)] reuse validated smoke $out/$model.json"
      continue
    fi
    checkpoint=$(checkpoint_for "$horizon" "$model")
    batch=$(batch_for "$horizon" "$model")
    "$PY" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir "$DATA" --test-year 2020 --climatology "$CLIM" \
      --stats-path "$STATS" --surface-stats-path "$SURFACE_STATS" \
      --static-path "$STATIC" --models "$model:$checkpoint" --out-dir "$out" \
      --paper-tag "corrected_baselines_v1_smoke_${horizon}h_${model}" \
      --batch-size "$batch" --num-workers 0 --samples-per-date 1 \
      --days-of-month 1 --proper-rmse --save-window-metrics \
      --save-physical-metrics --save-temporal-metrics \
      --max-tau-hours "$horizon" --eval-hours "$eval_hours" \
      --seen-tau "$seen" --unseen-tau "$held" --device cuda:0 \
      --lazy-climatology --climatology-cache-dir "$CACHE" \
      --keep-n-channels 24
  done
  validate_outputs "$horizon" "$out" 1
}

run_eval() {
  local horizon="$1" year="$2" cohort="$3"
  local eval_hours seen held samples data out extra=() tag
  local model checkpoint batch allow_economy
  configure_horizon "$horizon"
  if [[ "$horizon" == 6 ]]; then
    eval_hours=1,2,3,4,5; seen=1,3,5; held=2,4; samples=4
  else
    eval_hours=1,2,3,4,5,6,7,8,9,10,11
    seen=1,2,3,5,7,9,10,11; held=4,6,8; samples=2
  fi
  if [[ "$cohort" == frozen_extension ]]; then
    data="$POST_DATA"
    out="$OUT/${horizon}h/2022/frozen_extension"
    samples=1
    extra+=(--days-of-month 1,8,15,22)
  else
    data="$DATA"
    out="$OUT/${horizon}h/$year/full_year"
    extra+=(--full-year)
  fi
  mkdir -p "$out"
  tag="corrected_baselines_v1_${horizon}h_${year}_${cohort}"
  allow_economy=$([[ "$cohort" == frozen_extension ]] && echo 1 || echo 0)
  for model in swinv2 modafno sdyff pixelattn_vfi; do
    if output_valid "$horizon" "$out" "$model" "$allow_economy"; then
      echo "[$(date -Is)] reuse validated $out/$model.json"
      continue
    fi
    checkpoint=$(checkpoint_for "$horizon" "$model")
    batch=$(batch_for "$horizon" "$model")
    "$PY" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir "$data" --test-year "$year" --climatology "$CLIM" \
      --stats-path "$STATS" --surface-stats-path "$SURFACE_STATS" \
      --static-path "$STATIC" --models "$model:$checkpoint" --out-dir "$out" \
      --paper-tag "${tag}_${model}" --batch-size "$batch" \
      --num-workers 2 --samples-per-date "$samples" "${extra[@]}" \
      --proper-rmse --save-window-metrics --save-physical-metrics \
      --save-temporal-metrics --max-tau-hours "$horizon" \
      --eval-hours "$eval_hours" --seen-tau "$seen" --unseen-tau "$held" \
      --device cuda:0 --lazy-climatology --climatology-cache-dir "$CACHE" \
      --keep-n-channels 24
  done
  validate_outputs "$horizon" "$out" "$allow_economy"
  compare_outputs "$horizon" "$year" "$cohort" "$out" "$held"
  verify_source_snapshot
}

compare_outputs() {
  local horizon="$1" year="$2" cohort="$3" out="$4" held="$5"
  local core dcae baseline window
  if [[ "$cohort" == frozen_extension ]]; then
    core="$RUNTIME/metrics/postselection_2022_v1/${horizon}h"
    dcae=weatherdcae_14m
  else
    core="$RUNTIME/metrics/detailed_benchmark_v2/${horizon}h/$year/full_year"
    [[ "$horizon" == 6 ]] && dcae=weatherdcae_14m_6yr || dcae=weatherdcae_14m
  fi
  mkdir -p "$out/aligned/window_metrics"
  for baseline in swinv2 modafno sdyff pixelattn_vfi; do
    "$PY" tools/eval/align_window_artifact.py \
      --source-json "$out/$baseline.json" \
      --reference-json "$core/flow_spectral.json" \
      --output-json "$out/aligned/$baseline.json" \
      --output-window "$out/aligned/window_metrics/$baseline.npz"
  done
  window="$out/aligned/window_metrics"
  for baseline in swinv2 modafno sdyff pixelattn_vfi; do
    "$PY" tools/eval/paired_block_bootstrap.py \
      --left "$baseline:$window/$baseline.npz" \
      --right "flow_spectral:$core/window_metrics/flow_spectral.npz" \
      --right "$dcae:$core/window_metrics/$dcae.npz" \
      --draws 10000 --seed 20220809 --cellwise \
      --out-json "$out/paired_rmse_${baseline}.json"
    "$PY" tools/eval/paired_block_bootstrap.py \
      --left "$baseline:$window/$baseline.npz" \
      --right "flow_spectral:$core/window_metrics/flow_spectral.npz" \
      --right "$dcae:$core/window_metrics/$dcae.npz" \
      --taus "$held" --draws 10000 --seed 20220809 --cellwise \
      --out-json "$out/paired_rmse_held_${baseline}.json"
    "$PY" tools/eval/hard_window_block_bootstrap.py \
      --left "$baseline:$window/$baseline.npz" \
      --right "flow_spectral:$core/window_metrics/flow_spectral.npz" \
      --right "$dcae:$core/window_metrics/$dcae.npz" \
      --quantile 0.95 --draws 10000 --seed 20220809 \
      --out-json "$out/paired_hard_window_${baseline}.json"
    for metric in acc temporal_curvature physical; do
      "$PY" tools/eval/paired_aux_block_bootstrap.py \
        --left "$baseline:$window/$baseline.npz" \
        --right "flow_spectral:$core/window_metrics/flow_spectral.npz" \
        --right "$dcae:$core/window_metrics/$dcae.npz" \
        --metric "$metric" --draws 10000 --seed 20220809 \
        --out-json "$out/paired_${metric}_${baseline}.json"
    done
  done
}

run_smoke 6
run_smoke 12
for year in 2020 2021; do
  run_eval 6 "$year" full_year
  run_eval 12 "$year" full_year
done
run_eval 6 2022 frozen_extension
run_eval 12 2022 frozen_extension

verify_source_snapshot
"$PY" - "$OUT/.complete" "$SOURCE_SHA" "${SOURCE_FILES[@]}" -- "${CHECKPOINTS[@]}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

separator = sys.argv.index("--")
target = Path(sys.argv[1])
source_sha = sys.argv[2]
sources = [Path(value) for value in sys.argv[3:separator]]
checkpoints = [Path(value) for value in sys.argv[separator + 1:]]
payload = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "source_composite_sha256": source_sha,
    "source_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    },
    "checkpoint_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in checkpoints
    },
    "years": [2020, 2021, 2022],
    "horizons_hours": [6, 12],
    "models": ["swinv2", "modafno", "sdyff", "pixelattn_vfi"],
    "note": "The 2022 baseline extension is descriptive and was run after the frozen WeatherBridge-versus-WeatherDCAE endpoint was opened.",
}
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, target)
PY

flock -u 7
echo "[$(date -Is)] corrected baseline benchmark complete"
