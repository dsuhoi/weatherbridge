#!/usr/bin/env bash
# Independently reconstruct and audit normalization after memory-heavy evals.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
CLIMATOLOGY=${CLIMATOLOGY:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5.zarr}
COMPONENT_MARKER=${COMPONENT_MARKER:-$RUNTIME/metrics/weatherbridge_component_ablation_v1/state/.complete}
OUT=${OUT:-$RUNTIME/metrics/normalization_replay_v1}
LOG=${LOG:-$LOG_ROOT/normalization_replay_v1.log}
POLL_SECONDS=${POLL_SECONDS:-60}

cd "$SOURCE"
export PYTHONPATH="$SOURCE"
mkdir -p "$OUT" "$LOG_ROOT"

exec 9>"$LOG_ROOT/.normalization_replay.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] normalization replay already active" | tee -a "$LOG"
  exit 0
fi

while [[ ! -s "$COMPONENT_MARKER" ]]; do
  echo "[$(date -Is)] wait component-ablation memory release" | tee -a "$LOG"
  sleep "$POLL_SECONDS"
done

echo "[$(date -Is)] start normalization replay" | tee -a "$LOG"
"$PY" -u tools/data/generate_normalization_0p5.py \
  --climatology "$CLIMATOLOGY" \
  --pressure-output "$OUT/generated_stats.nc" \
  --surface-output "$OUT/generated_surface.json" 2>&1 | tee -a "$LOG"

"$PY" -u tools/data/verify_normalization_replay.py \
  --climatology "$CLIMATOLOGY" \
  --generated-pressure "$OUT/generated_stats.nc" \
  --generated-surface "$OUT/generated_surface.json" \
  --reference-pressure data/json_stats_0p5.nc \
  --reference-surface data/surface_stats_0p5.json \
  --generator tools/data/generate_normalization_0p5.py \
  --record-mismatch \
  --out "$OUT/report.json" 2>&1 | tee -a "$LOG"

"$PY" - "$OUT/report.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
if report.get("status") not in {"complete", "not_bit_exact"}:
    raise SystemExit("normalization reconstruction audit did not complete")
PY

echo "[$(date -Is)] normalization reconstruction audit recorded" | tee -a "$LOG"
