#!/usr/bin/env bash
# Confirm the development-guarded Flow adapter on frozen 2022 NWP data.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
HRES=${HRES:-/tmp/wti_hres_nwp_2022_frozen16_v1}
ERA5=${ERA5:-/tmp/wb2_0p5_nwp_2022}
STATIC=${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}
STATS=${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}
SURFACE_STATS=${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}
ADAPTER_ROOT=${ADAPTER_ROOT:-$RUNTIME/adapters/nwp_blend_6h_v4_endpoint_guard}
EVAL_ROOT=${EVAL_ROOT:-$RUNTIME/metrics/nwp_blend_6h_2022_v4_endpoint_guard}
MANIFEST=${MANIFEST:-$SOURCE/repro/nwp_forecast_holdout_2022.json}

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$EVAL_ROOT" "$LOG_ROOT"
exec 9>"$EVAL_ROOT/launch.lock"
if ! flock -n 9; then
  echo "2022 NWP holdout queue already active"
  exit 0
fi
LOG="$LOG_ROOT/nwp_blend_holdout_2022_v4_endpoint_guard.log"
exec > >(tee -a "$LOG") 2>&1

SOURCE_FILES=(
  tools/eval/batch_eval_forecast_anchor.py
  tools/eval/summarize_forecast_anchor.py
  tools/eval/summarize_nwp_blend.py
  tools/eval/capmatched_loader.py
  tools/train/train_capacity_matched_6h.py
  weather_time_interp/model/weatherbridge_flow_model.py
  weather_time_interp/normalization.py
  weather_time_interp/grid.py
  repro/nwp_forecast_holdout_2022.json
)
SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
verify_source() {
  local current
  current=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
  [[ "$current" == "$SOURCE_SHA" ]] || {
    echo "source changed during 2022 NWP holdout" >&2
    exit 2
  }
}

until [[ -s "$HRES/forecast_archive_manifest.json" && -s "$ERA5/wb2_2022.json" ]]; do
  echo "[$(date -Is)] wait frozen 2022 HRES and ERA5 archives"
  sleep 60
done

SELECTION="$ADAPTER_ROOT/selection.json"
ADAPTER="$ADAPTER_ROOT/flow.json"
CHECKPOINT="$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt"
"$PY" - "$MANIFEST" "$HRES" "$ERA5" "$SELECTION" "$ADAPTER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, hres_root, era5_root, selection_path, adapter_path = map(
    Path, sys.argv[1:]
)
manifest = json.loads(manifest_path.read_text())
parent = manifest_path.parent.parent / manifest["parent_manifest"]
parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
hres = json.loads((hres_root / "forecast_archive_manifest.json").read_text())
era5 = json.loads((era5_root / "wb2_2022.json").read_text())
selection = json.loads(selection_path.read_text())
adapter_sha = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
if (
    manifest.get("status") != "frozen_before_nwp_model_evaluation"
    or parent_sha != manifest.get("parent_manifest_sha256")
    or hres.get("init_times") != manifest.get("init_times")
    or era5.get("manifest_sha256")
    != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    or selection.get("schema_version") != 2
    or selection.get("winner") != "flow"
    or 2021 not in selection.get("development_years_opened_before_selection", [])
    or 2022 not in selection.get("confirmatory_years_unopened", [])
    or selection["adapters"]["flow"]["sha256"] != adapter_sha
):
    raise SystemExit("frozen 2022 NWP lineage mismatch")
PY
verify_source

if [[ ! -s "$EVAL_ROOT/linear.json" ]]; then
  "$PY" -u tools/eval/batch_eval_forecast_anchor.py \
    --forecast-dir "$HRES" --target-source era5 \
    --era5-memmap-dir "$ERA5" --model-blob linear --model-name linear \
    --out-json "$EVAL_ROOT/linear.json" --stats-path "$STATS" \
    --surface-stats-path "$SURFACE_STATS" --device cpu \
    --delta-t-hours 6 --taus 1 2 3 4 5 --max-inits 16 &
  pid_linear=$!
else
  pid_linear=""
fi

if [[ ! -s "$EVAL_ROOT/flow_adapted.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/eval/batch_eval_forecast_anchor.py \
    --forecast-dir "$HRES" --target-source era5 \
    --era5-memmap-dir "$ERA5" --checkpoint "$CHECKPOINT" --arch flow_pp3 \
    --blend-adapter "$ADAPTER" --model-name flow_adapted \
    --out-json "$EVAL_ROOT/flow_adapted.json" --stats-path "$STATS" \
    --surface-stats-path "$SURFACE_STATS" --static-path "$STATIC" \
    --device cuda --delta-t-hours 6 --taus 1 2 3 4 5 --max-inits 16 \
    --tau-batch-size 5 &
  pid_flow=$!
else
  pid_flow=""
fi

for pid in "$pid_linear" "$pid_flow"; do
  [[ -z "$pid" ]] || wait "$pid"
done
verify_source

"$PY" tools/eval/summarize_nwp_blend.py \
  --selection "$SELECTION" \
  --artifact "linear=$EVAL_ROOT/linear.json" \
  --artifact "flow_adapted=$EVAL_ROOT/flow_adapted.json" \
  --draws 10000 --seed 2027 --test-year 2022 --test-role independent \
  --output "$EVAL_ROOT/paired_summary.json"

"$PY" - "$EVAL_ROOT" "$SOURCE_SHA" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = [root / name for name in ("linear.json", "flow_adapted.json", "paired_summary.json")]
summary = json.loads(paths[-1].read_text())
payload = {
    "schema_version": 1,
    "status": "complete",
    "source_sha256": sys.argv[2],
    "winner": summary["winner"],
    "selection_independent": summary["selection_independent"],
    "test_split": summary["test_split"],
    "artifacts": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    },
}
temporary = root / ".complete.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "complete.json")
print(json.dumps(payload, indent=2))
PY
echo "[$(date -Is)] 2022 NWP endpoint-guard holdout complete"
