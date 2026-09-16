#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
ERA5_DIR="${ERA5_DIR:-/tmp/wb2_0p5_cache}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
FORCE="${FORCE:-0}"
POLL_SECONDS="${POLL_SECONDS:-120}"

cd "$REPO_ROOT"
mkdir -p metrics/downstream_physics metrics/downstream_diurnal "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$EXTRA_PYTHONPATH:$REPO_ROOT:${PYTHONPATH:-}"

while [[ ! -s metrics/journal_unified/.final_eval_complete ]]; do
  echo "[$(date -Is)] wait for canonical full-year evaluation"
  sleep "$POLL_SECONDS"
done

exec 9>"$LOG_DIR/weatherbridge_downstream.lock"
if ! flock -n 9; then
  echo "WeatherBridge downstream queue is already active."
  exit 0
fi
exec 8>/tmp/weatherbridge_journal_eval_6h.lock
flock 8

run_physics() {
  local model_blob="$1"
  local model_name="$2"
  local horizon="$3"
  local out_json="$4"
  shift 4
  [[ "$FORCE" != "1" && -s "$out_json" ]] && return
  "$PYTHON_BIN" -u tools/downstream/eval_physics_consistency.py \
    --era5-dir "$ERA5_DIR" \
    --year 2020 \
    --model-blob "$model_blob" \
    --model-name "$model_name" \
    --out-json "$out_json" \
    --device cuda:0 \
    --delta-t-hours "$horizon" \
    --taus "$@" \
    --sample-stride-hours 48 \
    --max-pairs 180 \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json
}

run_diurnal() {
  local model_blob="$1"
  local model_name="$2"
  local horizon="$3"
  local out_json="$4"
  shift 4
  [[ "$FORCE" != "1" && -s "$out_json" ]] && return
  "$PYTHON_BIN" -u tools/downstream/eval_diurnal_amplitude.py \
    --era5-dir "$ERA5_DIR" \
    --year 2020 \
    --model-blob "$model_blob" \
    --model-name "$model_name" \
    --out-json "$out_json" \
    --device cuda:0 \
    --delta-t-hours "$horizon" \
    --taus "$@" \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json
}

echo "[$(date -Is)] downstream queue started on physical GPU $GPU"

run_physics weights/weatherbridge_14m_6h_bare.pt WeatherBridge 6 \
  metrics/downstream_physics/weatherbridge_6h.json 1 2 3 4 5
run_physics linear "Linear Interp." 6 \
  metrics/downstream_physics/linear_6h.json 1 2 3 4 5
run_physics weights/weatherbridge_14m_12h_bare.pt WeatherBridge 12 \
  metrics/downstream_physics/weatherbridge_12h.json 1 2 3 4 5 6 7 8 9 10 11
run_physics linear "Linear Interp." 12 \
  metrics/downstream_physics/linear_12h.json 1 2 3 4 5 6 7 8 9 10 11

run_diurnal weights/weatherbridge_14m_6h_bare.pt WeatherBridge 6 \
  metrics/downstream_diurnal/weatherbridge_6h.json 1 2 3 4 5
run_diurnal linear "Linear Interp." 6 \
  metrics/downstream_diurnal/linear_6h.json 1 2 3 4 5
run_diurnal weights/weatherbridge_14m_12h_bare.pt WeatherBridge 12 \
  metrics/downstream_diurnal/weatherbridge_12h.json 1 2 3 4 5 6 7 8 9 10 11
run_diurnal linear "Linear Interp." 12 \
  metrics/downstream_diurnal/linear_12h.json 1 2 3 4 5 6 7 8 9 10 11

echo "[$(date -Is)] downstream queue complete"
