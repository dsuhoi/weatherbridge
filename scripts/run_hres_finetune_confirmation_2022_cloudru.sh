#!/usr/bin/env bash
# Build the frozen 2022 date index and evaluate HRES-fine-tuned checkpoints.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
PROTOCOL=${PROTOCOL:-$SOURCE/repro/hres_finetune_protocol.json}
TRAIN_ROOT=${TRAIN_ROOT:-$LOG_ROOT/hres_finetune_6h_108x24_v2}
HRES=${HRES:-/tmp/wti_hres_finetune_confirm_2022_v1}
ERA5=${ERA5:-/tmp/wb2_0p5_hres_confirmation_2022_108x24_v2}
ERA5_MANIFEST=${ERA5_MANIFEST:-$SOURCE/repro/hres_finetune_confirmation_era5_2022.json}
STATIC=${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}
STATS=${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}
SURFACE_STATS=${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}
OUT_ROOT=${OUT_ROOT:-$RUNTIME/metrics/hres_finetune_confirmation_2022_108x24_v2}
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
MIN_FREE_KIB=${MIN_FREE_KIB:-75000000}

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$OUT_ROOT/rmse" "$OUT_ROOT/spectra" "$LOG_ROOT" "$HRES"
exec 9>"$OUT_ROOT/launch.lock"
if ! flock -n 9; then
  echo "HRES fine-tuning confirmation queue already active"
  exit 0
fi
exec > >(tee -a "$LOG_ROOT/hres_finetune_confirmation_2022_108x24_v2.log") 2>&1

SOURCE_FILES=(
  scripts/run_hres_finetune_confirmation_2022_cloudru.sh
  tools/data/build_hres_forecast_memmap.py
  tools/data/build_postselection_2022_memmap.py
  tools/eval/batch_eval_forecast_anchor.py
  tools/eval/nwp_blend_spectra.py
  tools/eval/summarize_forecast_anchor.py
  tools/eval/assess_hres_finetune_confirmation.py
  tools/eval/capmatched_loader.py
  tools/train/train_capacity_matched_6h.py
  tools/train/training_protocol.py
  weather_time_interp/grid.py
  weather_time_interp/metrics/spherical_spectra.py
  weather_time_interp/model/weatherbridge_flow_model.py
  weather_time_interp/model/dcae_adaln_model.py
  weather_time_interp/normalization.py
  repro/hres_finetune_protocol.json
  repro/hres_finetune_confirmation_era5_2022.json
)
SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')

verify_source() {
  local current
  current=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
  [[ "$current" == "$SOURCE_SHA" ]] || {
    echo "source changed during HRES confirmation" >&2
    exit 2
  }
}

until [[ -s "$TRAIN_ROOT/.training_complete" ]]; do
  echo "[$(date -Is)] waiting for HRES fine-tuning selections"
  sleep 120
done

INIT_CSV=$(
  "$PY" - "$PROTOCOL" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
split = payload["splits"]["date_level_confirmation"]
values = split["init_times"]
if (
    payload.get("status") != "frozen_before_hres_weight_finetuning"
    or split.get("year") != 2022
    or len(values) != 24
    or len(values) != len(set(values))
):
    raise SystemExit("invalid frozen HRES fine-tuning protocol")
print(",".join(values))
PY
)

if [[ ! -s "$HRES/forecast_archive_manifest.json" ]]; then
  free_kib=$(df --output=avail "$HRES" | tail -n 1 | tr -d ' ')
  [[ "$free_kib" -ge "$MIN_FREE_KIB" ]] || {
    echo "need at least $MIN_FREE_KIB KiB free, found $free_kib" >&2
    exit 2
  }
  "$PY" -u tools/data/build_hres_forecast_memmap.py \
    --out-dir "$HRES" --init-times "$INIT_CSV"
fi

"$PY" - "$PROTOCOL" "$HRES/forecast_archive_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

protocol = json.loads(Path(sys.argv[1]).read_text())
manifest = json.loads(Path(sys.argv[2]).read_text())
expected = protocol["splits"]["date_level_confirmation"]["init_times"]
if manifest.get("init_times") != expected or len(manifest.get("files", {})) != 24:
    raise SystemExit("HRES confirmation archive differs from frozen dates")
PY
if [[ ! -s "$ERA5/wb2_2022.json" ]]; then
  "$PY" -u tools/data/build_postselection_2022_memmap.py \
    --manifest "$ERA5_MANIFEST" --out-dir "$ERA5" --workers 8
fi
"$PY" - "$ERA5_MANIFEST" "$ERA5/wb2_2022.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
metadata = json.loads(Path(sys.argv[2]).read_text())
manifest = json.loads(manifest_path.read_text())
expected_hours = len(manifest["init_times"]) * (
    int(manifest["maximum_forecast_lead_hours"]) + 1
)
if (
    metadata.get("sparse_file") is not True
    or metadata.get("selected_window_count") != len(manifest["init_times"])
    or len(metadata.get("selected_relative_hours", [])) != expected_hours
    or metadata.get("manifest_sha256")
    != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
):
    raise SystemExit("dedicated HRES confirmation ERA5 cache is invalid")
PY
test -s "$ERA5/wb2_2022.bin"
test -s "$STATIC"
test -s "$STATS"
test -s "$SURFACE_STATS"
verify_source

RAW_WEATHERBRIDGE=${RAW_WEATHERBRIDGE:-$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt}
RAW_WEATHERDCAE=${RAW_WEATHERDCAE:-$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt}
test -s "$RAW_WEATHERBRIDGE"
test -s "$RAW_WEATHERDCAE"

selected_checkpoint() {
  local family="$1" seed="$2"
  "$PY" - "$TRAIN_ROOT/${family}_s${seed}/selection.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

selection = Path(sys.argv[1])
payload = json.loads(selection.read_text())
checkpoint = Path(payload["checkpoint"]["path"])
if (
    payload.get("status") != "selected_on_2020_validation"
    or not checkpoint.is_file()
    or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    != payload["checkpoint"]["sha256"]
):
    raise SystemExit(f"invalid selected checkpoint: {selection}")
print(checkpoint)
PY
}

run_linear() {
  local output="$OUT_ROOT/rmse/linear.json"
  [[ -s "$output" ]] && return 0
  "$PY" -u tools/eval/batch_eval_forecast_anchor.py \
    --forecast-dir "$HRES" --target-source era5 \
    --era5-memmap-dir "$ERA5" --model-blob linear --model-name linear \
    --out-json "$output" --stats-path "$STATS" \
    --surface-stats-path "$SURFACE_STATS" --device cpu \
    --delta-t-hours 6 --taus 1 2 3 4 5 --max-inits 24 \
    --forecast-lead-stride-hours 24 \
    --forecast-maximum-left-lead-hours 120
}

run_model() {
  local gpu="$1" name="$2" arch="$3" checkpoint="$4"
  local output="$OUT_ROOT/rmse/${name}.json"
  [[ -s "$output" ]] && return 0
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u tools/eval/batch_eval_forecast_anchor.py \
    --forecast-dir "$HRES" --target-source era5 \
    --era5-memmap-dir "$ERA5" --checkpoint "$checkpoint" --arch "$arch" \
    --model-name "$name" --out-json "$output" --stats-path "$STATS" \
    --surface-stats-path "$SURFACE_STATS" --static-path "$STATIC" \
    --device cuda --delta-t-hours 6 --taus 1 2 3 4 5 --max-inits 24 \
    --tau-batch-size 5 --forecast-lead-stride-hours 24 \
    --forecast-maximum-left-lead-hours 120
}

run_linear
run_model "$GPU_A" weatherbridge_raw flow_pp3 "$RAW_WEATHERBRIDGE" &
pid_a=$!
run_model "$GPU_B" weatherdcae_raw dcae_14m "$RAW_WEATHERDCAE" &
pid_b=$!
wait "$pid_a" "$pid_b"

for seed in 202707 202708 202709; do
  bridge=$(selected_checkpoint weatherbridge "$seed")
  dcae=$(selected_checkpoint weatherdcae "$seed")
  run_model "$GPU_A" "weatherbridge_s${seed}" flow_pp3 "$bridge" &
  pid_a=$!
  run_model "$GPU_B" "weatherdcae_s${seed}" dcae_14m "$dcae" &
  pid_b=$!
  wait "$pid_a" "$pid_b"
done
verify_source

PRIMARY_SEED=$(
  "$PY" - "$PROTOCOL" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["optimization"]["primary_seed"])
PY
)
PRIMARY_CHECKPOINT=$(selected_checkpoint weatherbridge "$PRIMARY_SEED")

run_spectra() {
  local gpu="$1" name="$2" checkpoint="$3" output="$4"
  [[ -s "$output/spectra_complete.json" ]] && return 0
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u tools/eval/nwp_blend_spectra.py \
    --forecast-dir "$HRES" --era5-memmap-dir "$ERA5" \
    --checkpoint "$checkpoint" --arch flow_pp3 --model-name "$name" \
    --out-dir "$output" --stats-path "$STATS" \
    --surface-stats-path "$SURFACE_STATS" --static-path "$STATIC" \
    --device cuda --delta-t-hours 6 --taus 2 3 --channels Q850 U850 \
    --lmax 180 --hf-ell-min 80 --max-inits 24 --tau-batch-size 2 \
    --forecast-lead-stride-hours 24 \
    --forecast-maximum-left-lead-hours 120
}

run_spectra "$GPU_A" "weatherbridge_s${PRIMARY_SEED}" \
  "$PRIMARY_CHECKPOINT" "$OUT_ROOT/spectra/fine" &
pid_a=$!
run_spectra "$GPU_B" weatherbridge_raw \
  "$RAW_WEATHERBRIDGE" "$OUT_ROOT/spectra/raw" &
pid_b=$!
wait "$pid_a" "$pid_b"
verify_source

ARTIFACT_ARGS=()
for name in \
  linear weatherbridge_raw weatherdcae_raw \
  weatherbridge_s202707 weatherdcae_s202707 \
  weatherbridge_s202708 weatherdcae_s202708 \
  weatherbridge_s202709 weatherdcae_s202709; do
  ARTIFACT_ARGS+=(--artifact "$name=$OUT_ROOT/rmse/$name.json")
done
"$PY" tools/eval/assess_hres_finetune_confirmation.py \
  --protocol "$PROTOCOL" --training-marker "$TRAIN_ROOT/.training_complete" \
  "${ARTIFACT_ARGS[@]}" \
  --raw-checkpoint "weatherbridge_raw=$RAW_WEATHERBRIDGE" \
  --raw-checkpoint "weatherdcae_raw=$RAW_WEATHERDCAE" \
  --fine-spectra-dir "$OUT_ROOT/spectra/fine" \
  --raw-spectra-dir "$OUT_ROOT/spectra/raw" \
  --draws 10000 --seed 2027 --output "$OUT_ROOT/assessment.json"

"$PY" - "$OUT_ROOT" "$SOURCE_SHA" "${SOURCE_FILES[@]}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
assessment = root / "assessment.json"
payload = json.loads(assessment.read_text())
marker = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "assessment_status": payload["status"],
    "promotion_passed": payload["promotion_passed"],
    "assessment": {
        "path": str(assessment),
        "sha256": hashlib.sha256(assessment.read_bytes()).hexdigest(),
    },
    "source_sha256": sys.argv[2],
    "source_files_sha256": {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in sys.argv[3:]
    },
}
temporary = root / ".complete.tmp"
temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / ".complete")
PY
echo "[$(date -Is)] HRES fine-tuning confirmation complete"
