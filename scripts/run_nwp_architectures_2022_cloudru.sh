#!/usr/bin/env bash
# Evaluate the frozen principal architectures on the independent 2022 NWP index.
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
MANIFEST=${MANIFEST:-$SOURCE/repro/nwp_forecast_holdout_2022.json}
OUT_ROOT=${OUT_ROOT:-$RUNTIME/metrics/nwp_architectures_6h_2022_v1}

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
exec 9>"$OUT_ROOT/launch.lock"
if ! flock -n 9; then
  echo "2022 NWP architecture evaluation already active"
  exit 0
fi
LOG="$LOG_ROOT/nwp_architectures_6h_2022_v1.log"
exec > >(tee -a "$LOG") 2>&1

SOURCE_FILES=(
  scripts/run_nwp_architectures_2022_cloudru.sh
  tools/eval/batch_eval_forecast_anchor.py
  tools/eval/capmatched_loader.py
  tools/train/train_capacity_matched_6h.py
  weather_time_interp/model/weatherbridge_flow_model.py
  weather_time_interp/normalization.py
  weather_time_interp/grid.py
  repro/nwp_forecast_holdout_2022.json
)
SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')

"$PY" - "$MANIFEST" "$HRES" "$ERA5" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, hres_root, era5_root = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
parent = manifest_path.parent.parent / manifest["parent_manifest"]
hres = json.loads((hres_root / "forecast_archive_manifest.json").read_text())
era5 = json.loads((era5_root / "wb2_2022.json").read_text())
if (
    manifest.get("status") != "frozen_before_nwp_model_evaluation"
    or hashlib.sha256(parent.read_bytes()).hexdigest()
    != manifest.get("parent_manifest_sha256")
    or hres.get("init_times") != manifest.get("init_times")
    or era5.get("manifest_sha256")
    != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
):
    raise SystemExit("frozen 2022 NWP lineage mismatch")
PY

run_model() {
  local gpu="$1" arch="$2" model_name="$3" checkpoint="$4" output="$5"
  if [[ -s "$output" ]]; then
    echo "[$(date -Is)] reuse $output"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u tools/eval/batch_eval_forecast_anchor.py \
    --forecast-dir "$HRES" --target-source era5 \
    --era5-memmap-dir "$ERA5" --checkpoint "$checkpoint" --arch "$arch" \
    --model-name "$model_name" --out-json "$output" --stats-path "$STATS" \
    --surface-stats-path "$SURFACE_STATS" --static-path "$STATIC" \
    --device cuda --delta-t-hours 6 --taus 1 2 3 4 5 --max-inits 16 \
    --tau-batch-size 5
}

DCAE_CKPT="$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt"
FLOW_CKPT="$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt"
BRIDGE_CKPT="$LOG_ROOT/exp_flow_pp3_detail_14m_6h_refinev1_s202707_bs4/last.ckpt"
for checkpoint in "$DCAE_CKPT" "$FLOW_CKPT" "$BRIDGE_CKPT"; do
  [[ -s "$checkpoint" ]] || {
    echo "missing checkpoint: $checkpoint" >&2
    exit 2
  }
done

run_model 0 dcae_14m weatherdcae_14m "$DCAE_CKPT" "$OUT_ROOT/weatherdcae_14m.json" &
pid_dcae=$!
run_model 1 flow_pp3_detail weatherbridge "$BRIDGE_CKPT" "$OUT_ROOT/weatherbridge.json" &
pid_bridge=$!
wait "$pid_dcae" "$pid_bridge"
run_model 0 flow_pp3 flow_spectral "$FLOW_CKPT" "$OUT_ROOT/flow_spectral.json"

"$PY" - "$OUT_ROOT" "$SOURCE_SHA" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = {
    name: root / f"{name}.json"
    for name in ("weatherdcae_14m", "flow_spectral", "weatherbridge")
}
payloads = {name: json.loads(path.read_text()) for name, path in paths.items()}
indices = {
    payload["paired_artifact"]["window_index_sha256"]
    for payload in payloads.values()
}
protocols = []
for payload in payloads.values():
    protocol = dict(payload["protocol"])
    protocol.pop("inference_tau_batch_size", None)
    protocols.append(protocol)
if len(indices) != 1 or any(protocol != protocols[0] for protocol in protocols[1:]):
    raise SystemExit("2022 architecture artifacts are not paired")
if any(payload["protocol"].get("max_inits") != 16 for payload in payloads.values()):
    raise SystemExit("2022 architecture artifact has the wrong init count")

result = {
    "schema_version": 1,
    "status": "complete",
    "test_split": "2022_independent_descriptive_controls",
    "selection_independent": False,
    "checkpoint_weights_frozen_before_2022": True,
    "window_index_sha256": next(iter(indices)),
    "source_sha256": sys.argv[2],
    "artifacts": {
        name: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in paths.items()
    },
}
temporary = root / ".complete.tmp"
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "complete.json")
print(json.dumps(result, indent=2, sort_keys=True))
PY
echo "[$(date -Is)] 2022 NWP architecture evaluation complete"
