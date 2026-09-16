#!/usr/bin/env bash
# Full-year 2020 parameter-free transport baselines for the primary 6 h task.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
COMPONENT_OUT=${COMPONENT_OUT:-$RUNTIME/metrics/weatherbridge_component_ablation_v1}
OUT=${OUT:-$RUNTIME/metrics/numerical_baselines_6h_2020_v3}
LOG=${LOG:-$LOG_ROOT/numerical_baselines_6h_2020_v3.log}
GPU=${GPU:-1}
POLL_SECONDS=${POLL_SECONDS:-60}

cd "$SOURCE"
export PYTHONPATH="$SOURCE"
mkdir -p "$OUT" "$OUT/state" "$LOG_ROOT"

exec 9>"$LOG_ROOT/.numerical_baselines_6h_2020_v3.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] numerical baseline queue already active" | tee -a "$LOG"
  exit 0
fi

# The first two lesions release GPU1 before the acceleration lesion starts.
while [[ ! -s "$COMPONENT_OUT/weatherbridge.json" \
      || ! -s "$COMPONENT_OUT/weatherbridge_transport_off.json" ]]; do
  echo "[$(date -Is)] wait first component evaluations" | tee -a "$LOG"
  sleep "$POLL_SECONDS"
done

exec 8>"$LOG_ROOT/.upr_lite_gpu${GPU}.lock"
flock 8
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[$(date -Is)] start numerical baselines GPU=$GPU" | tee -a "$LOG"
"$PY" -u tools/eval/numerical_baseline_eval.py \
  --memmap-dir /tmp/wb2_0p5_cache \
  --test-year 2020 \
  --stats-path data/json_stats_0p5.nc \
  --surface-stats-path data/surface_stats_0p5.json \
  --static-path data/static_features_0p5.pt \
  --out-dir "$OUT" \
  --batch-size 2 \
  --num-workers 2 \
  --samples-per-date 4 \
  --full-year \
  --dt-hours 6 \
  --max-tau-hours 6 \
  --eval-hours 1,2,3,4,5 2>&1 | tee -a "$LOG"

"$PY" - "$OUT" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
methods = (
    "bilinear",
    "semi_lagrangian",
    "hermite_advection",
    "settls",
    "hermite_diffusion",
)
artifacts = {}
for method in methods:
    path = root / f"{method}.json"
    payload = json.loads(path.read_text())
    if payload.get("num_samples") != 1463 * 5:
        raise SystemExit(f"unexpected sample count for {method}")
    protocol = payload.get("evaluation_protocol", {})
    if protocol.get("longitude_reduction") != "mean":
        raise SystemExit(f"invalid longitude reduction for {method}")
    if protocol.get("latitude_grid") != "wb2_0p25_2x2_block_average_v1":
        raise SystemExit(f"invalid latitude grid for {method}")
    if not protocol.get("transport_sampling", "").startswith(
        "canonical_cell_centres_periodic_longitude"
    ):
        raise SystemExit(f"invalid transport sampler for {method}")
    artifacts[method] = hashlib.sha256(path.read_bytes()).hexdigest()
marker = root / "state" / ".complete"
temporary = marker.with_suffix(".tmp")
temporary.write_text(json.dumps({
    "schema_version": 2,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "protocol": (
        "full-year 2020, 6 h, 24 scored channels selected downstream; "
        "canonical block-average centres and periodic longitude sampling"
    ),
    "artifacts_sha256": artifacts,
}, sort_keys=True) + "\n")
os.replace(temporary, marker)
PY

flock -u 8
echo "[$(date -Is)] numerical baselines complete" | tee -a "$LOG"
