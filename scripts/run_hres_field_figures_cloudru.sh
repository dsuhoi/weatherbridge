#!/usr/bin/env bash
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
MEMMAP=${MEMMAP:-/tmp/wb2_0p5_cache}
HRES=${HRES:-/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_hres_2021_canonical_v3}
STATS=${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}
SURFACE_STATS=${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}
POLL_SECONDS=${POLL_SECONDS:-120}

EVAL_ROOT="$RUNTIME/metrics/npj_seed_ifs_hres_2021_v3"
GEOMETRY_MARKER="$EVAL_ROOT/geometry_v3/primary.complete"
OUT_ROOT="$EVAL_ROOT/field_figures_v1"
MARKER="$SOURCE/paper/images/.hres_field_figures_complete"
LOG="$LOG_ROOT/hres_field_figures_v1.queue.log"

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$OUT_ROOT" "$(dirname "$MARKER")"
exec 8>"$LOG_ROOT/.hres_field_figures_v1.queue.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] HRES field-figure queue already active"
  exit 0
fi
exec > >(tee -a "$LOG") 2>&1

SOURCE_FILES=(
  scripts/run_hres_field_figures_cloudru.sh
  scripts/make_hres_field_comparison.py
  scripts/paper_plot_style.py
  tools/eval/batch_eval_forecast_anchor.py
  weather_time_interp/grid.py
  weather_time_interp/normalization.py
)
SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')

verify_source_snapshot() {
  local current_sha
  current_sha=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
  if [[ "$current_sha" != "$SOURCE_SHA" ]]; then
    echo "HRES figure source changed while queue was active" >&2
    exit 2
  fi
}

wait_complete() {
  local path="$1"
  until "$PY" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
raise SystemExit(
    0 if json.loads(path.read_text()).get("status") == "complete" else 1
)
PY
  do
    echo "[$(date -Is)] waiting for $path"
    sleep "$POLL_SECONDS"
  done
}

wait_complete "$GEOMETRY_MARKER"
verify_source_snapshot

run_linear() {
  local horizon="$1" output="$2" taus=()
  if [[ "$horizon" == 6 ]]; then
    taus=(1 2 3 4 5)
  else
    taus=(1 2 3 4 5 6 7 8 9 10 11)
  fi
  "$PY" -u tools/eval/batch_eval_forecast_anchor.py \
    --forecast-dir "$HRES" --era5-memmap-dir "$MEMMAP" \
    --model-blob linear --model-name linear --out-json "$output" \
    --stats-path "$STATS" --surface-stats-path "$SURFACE_STATS" \
    --device cpu --delta-t-hours "$horizon" --taus "${taus[@]}" \
    --max-inits 16
}

artifact_path() {
  local family="$1" horizon="$2" infix="" exp
  [[ "$horizon" == 12 ]] && infix=_2017_19
  case "$family" in
    flow_spectral)
      exp="exp_flow_pp3_spectral_14m_${horizon}h${infix}_refinev1_s202707_bs4"
      ;;
    dcae)
      if [[ "$horizon" == 6 ]]; then
        exp="exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4"
      else
        exp="exp_weatherdcae_14m_12h_2017_19_refinev1_s202707_bs4"
      fi
      ;;
    *) return 2 ;;
  esac
  printf '%s/%sh/%s.json\n' "$EVAL_ROOT" "$horizon" "$exp"
}

declare -a FIGURES=()
declare -a MANIFESTS=()
for horizon in 6 12; do
  linear="$OUT_ROOT/linear_${horizon}h.json"
  run_linear "$horizon" "$linear"
  flow=$(artifact_path flow_spectral "$horizon")
  dcae=$(artifact_path dcae "$horizon")
  for artifact in "$flow" "$dcae"; do
    [[ -s "$artifact" ]] || {
      echo "missing primary-seed HRES artifact: $artifact" >&2
      exit 1
    }
  done

  figure="paper/images/fig_hres_fields_${horizon}h.pdf"
  manifest="$OUT_ROOT/fig_hres_fields_${horizon}h.manifest.json"
  "$PY" scripts/make_hres_field_comparison.py \
    --horizon "$horizon" \
    --artifact "Linear Interp.=$linear" \
    --artifact "WeatherDCAE-14M=$dcae" \
    --artifact "WeatherBridge=$flow" \
    --output "$figure" --manifest "$manifest"
  FIGURES+=("$figure")
  MANIFESTS+=("$manifest")
done
verify_source_snapshot

"$PY" - "$MARKER" "$SOURCE_SHA" "${FIGURES[@]}" -- "${MANIFESTS[@]}" -- "${SOURCE_FILES[@]}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

marker = Path(sys.argv[1])
source_sha = sys.argv[2]
first = sys.argv.index("--")
second = sys.argv.index("--", first + 1)
figures = [Path(value) for value in sys.argv[3:first]]
manifests = [Path(value) for value in sys.argv[first + 1:second]]
sources = [Path(value) for value in sys.argv[second + 1:]]
payload = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "central_model": "WeatherBridge",
    "internal_arch": "flow_pp3",
    "color": "#D62728",
    "horizons": [6, 12],
    "forecast_init_count": 16,
    "source_sha256": source_sha,
    "source_files_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    },
    "figure_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in figures
    },
    "manifest_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in manifests
    },
}
temporary = marker.with_suffix(marker.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, marker)
PY
echo "[$(date -Is)] HRES field-figure queue complete"
