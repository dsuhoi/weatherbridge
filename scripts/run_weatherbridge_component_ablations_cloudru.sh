#!/usr/bin/env bash
# Full-year 2020 inference lesions of WeatherBridge components.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
CKPT=${CKPT:-$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt}
OUT=${OUT:-$RUNTIME/metrics/weatherbridge_component_ablation_v1}
LOG=${LOG:-$LOG_ROOT/weatherbridge_component_ablation_v1.log}

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
test -s "$CKPT"
test -s data/json_stats_0p5.nc
test -s data/surface_stats_0p5.json
test -s data/static_features_0p5.pt
test -s /tmp/wb2_0p5_cache/wb2_2020.bin
mkdir -p "$OUT" "$OUT/state"

SOURCE_FILES=(
  scripts/run_weatherbridge_component_ablations_cloudru.sh
  tools/eval/batch_eval_12h_memmap.py
  tools/eval/capmatched_loader.py
  tools/eval/paired_block_bootstrap.py
  weather_time_interp/model/weatherbridge_flow_model.py
)
SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')

run_arm() {
  local gpu="$1"
  local name="$2"
  local acceleration="$3"
  local transport="$4"
  local hydrostatic="$5"
  local result="$OUT/$name.json"
  if [[ -s "$result" ]]; then
    echo "[$(date -Is)] keep existing $result" | tee -a "$LOG"
    return
  fi
  echo "[$(date -Is)] start $name GPU=$gpu source=$SOURCE_SHA" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year 2020 \
    --climatology /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "$name:$CKPT:WEATHERBRIDGE_ABLATE_ACCELERATION=$acceleration WEATHERBRIDGE_ABLATE_TRANSPORT=$transport WEATHERBRIDGE_ABLATE_HYDROSTATIC=$hydrostatic" \
    --out-dir "$OUT" \
    --paper-tag weatherbridge_component_ablation_6h_2020 \
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
    --save-temporal-metrics 2>&1 | tee -a "$LOG"
  test -s "$result"
  echo "[$(date -Is)] done $name" | tee -a "$LOG"
}

exec 8>"$LOG_ROOT/.weatherbridge_component_gpu0.lock"
exec 9>"$LOG_ROOT/.weatherbridge_component_gpu1.lock"
flock 8
flock 9
run_arm 0 weatherbridge 0 0 0 &
pid_baseline=$!
run_arm 1 weatherbridge_transport_off 0 1 0 &
pid_transport=$!
wait "$pid_baseline" "$pid_transport"
run_arm 0 weatherbridge_acceleration_off 1 0 0 &
pid_acceleration=$!
run_arm 1 weatherbridge_hydrostatic_off 0 0 1 &
pid_hydrostatic=$!
wait "$pid_acceleration" "$pid_hydrostatic"

"$PY" tools/eval/paired_block_bootstrap.py \
  --left "hydrostatic_off:$OUT/window_metrics/weatherbridge_hydrostatic_off.npz" \
  --right "weatherbridge:$OUT/window_metrics/weatherbridge.npz" \
  --taus 1,2,3,4,5 \
  --block-days 7 \
  --draws 5000 \
  --seed 2027 \
  --cellwise \
  --out-json "$OUT/hydrostatic_off_vs_weatherbridge.json"

"$PY" - "$OUT/state/.complete" "$SOURCE_SHA" "$CKPT" \
  "${SOURCE_FILES[@]}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

marker_path = Path(sys.argv[1])
source_sha = sys.argv[2]
checkpoint = Path(sys.argv[3])
sources = [Path(value) for value in sys.argv[4:]]
root = marker_path.parent.parent
models = (
    "weatherbridge",
    "weatherbridge_acceleration_off",
    "weatherbridge_transport_off",
    "weatherbridge_hydrostatic_off",
)
for model in models:
    if not (root / f"{model}.json").is_file():
        raise SystemExit(f"missing result: {model}")
comparison = root / "hydrostatic_off_vs_weatherbridge.json"
if not comparison.is_file():
    raise SystemExit(f"missing comparison: {comparison}")
payload = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "protocol": "full-year 2020, 6 h, inference component lesions",
    "models": list(models),
    "comparisons": [comparison.name],
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "source_sha256": source_sha,
    "source_files_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    },
}
temporary = marker_path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, marker_path)
PY
flock -u 8
flock -u 9
echo "[$(date -Is)] WeatherBridge component ablations complete" | tee -a "$LOG"
