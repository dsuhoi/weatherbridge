#!/usr/bin/env bash
# Fit tiny HRES-to-ERA5 adapters and evaluate them on the disjoint 2021 archive.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
HRES_CAL=${HRES_CAL:-/tmp/wti_hres_adapt_2017_2020_quarterly_v1}
HRES_TEST=${HRES_TEST:-/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_hres_2021_canonical_v3}
ERA5=${ERA5:-/tmp/wb2_0p5_cache}
STATIC=${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}
STATS=${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}
SURFACE_STATS=${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}
ADAPTER_ROOT=${ADAPTER_ROOT:-$RUNTIME/adapters/nwp_blend_6h_v1}
EVAL_ROOT=${EVAL_ROOT:-$RUNTIME/metrics/nwp_blend_6h_2021_v1}
RUN_TAG=${RUN_TAG:-nwp_blend_adaptation_6h_v1}
TEST_YEAR=${TEST_YEAR:-2021}
TEST_ROLE=${TEST_ROLE:-independent}
DEVELOPMENT_YEARS_OPENED=${DEVELOPMENT_YEARS_OPENED:-}
CONFIRMATORY_YEARS_UNOPENED=${CONFIRMATORY_YEARS_UNOPENED:-2021}

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$ADAPTER_ROOT" "$EVAL_ROOT" "$LOG_ROOT"
exec 9>"$ADAPTER_ROOT/launch.lock"
if ! flock -n 9; then
  echo "NWP blend adaptation queue already active"
  exit 0
fi
LOG="$LOG_ROOT/${RUN_TAG}.log"
exec > >(tee -a "$LOG") 2>&1

SOURCE_FILES=(
  tools/train/fit_nwp_blend_adapter.py
  tools/eval/batch_eval_forecast_anchor.py
  tools/eval/summarize_forecast_anchor.py
  tools/eval/summarize_nwp_blend.py
  tools/eval/capmatched_loader.py
  tools/train/train_capacity_matched_6h.py
  weather_time_interp/model/weatherbridge_flow_model.py
  weather_time_interp/model/dcae_adaln_model.py
  weather_time_interp/normalization.py
  weather_time_interp/grid.py
)
SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
verify_source() {
  local current
  current=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
  [[ "$current" == "$SOURCE_SHA" ]] || {
    echo "source changed during NWP blend adaptation" >&2
    exit 2
  }
}

until [[ -s "$HRES_CAL/forecast_archive_manifest.json" ]]; do
  echo "[$(date -Is)] wait calibration HRES archive"
  sleep 60
done
"$PY" - "$HRES_CAL" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / "forecast_archive_manifest.json").read_text())
years = [int(value[:4]) for value in manifest["init_times"]]
expected = {2017: 2, 2018: 2, 2019: 2, 2020: 2}
if len(years) != 8 or {year: years.count(year) for year in set(years)} != expected:
    raise SystemExit("calibration archive does not match the frozen split")
PY
verify_source

configure() {
  local family="$1"
  case "$family" in
    dcae)
      ARCH=dcae_14m
      CHECKPOINT="$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt"
      ;;
    flow)
      ARCH=flow_pp3
      CHECKPOINT="$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt"
      ;;
    *) return 2 ;;
  esac
  ADAPTER="$ADAPTER_ROOT/${family}.json"
  TEST_JSON="$EVAL_ROOT/${family}_adapted.json"
}

fit_family() {
  local gpu="$1" family="$2"
  configure "$family"
  [[ -s "$CHECKPOINT" ]] || {
    echo "missing checkpoint: $CHECKPOINT" >&2
    return 1
  }
  if [[ -s "$ADAPTER" ]]; then
    echo "[$(date -Is)] reuse adapter family=$family"
    return 0
  fi
  exec 8>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
  flock 8
  verify_source
  echo "[$(date -Is)] fit adapter family=$family gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u tools/train/fit_nwp_blend_adapter.py \
    --forecast-dir "$HRES_CAL" --era5-memmap-dir "$ERA5" \
    --checkpoint "$CHECKPOINT" --arch "$ARCH" --output "$ADAPTER" \
    --fit-years 2017 2018 2019 --validation-years 2020 \
    --delta-t-hours 6 --fit-taus 1 3 5 --eval-taus 1 2 3 4 5 \
    --lead-stride-hours 24 \
    --minimum-lead-relative-rmse-gain 0.001 \
    --stats-path "$STATS" --surface-stats-path "$SURFACE_STATS" \
    --static-path "$STATIC" --device cuda
  verify_source
  flock -u 8
  exec 8>&-
}

fit_family 0 flow &
pid_flow=$!
wait "$pid_flow"
fit_family 0 dcae

"$PY" - "$ADAPTER_ROOT" "$DEVELOPMENT_YEARS_OPENED" \
  "$CONFIRMATORY_YEARS_UNOPENED" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
development_years = [
    int(value) for value in sys.argv[2].split(",") if value
]
confirmatory_years = [
    int(value) for value in sys.argv[3].split(",") if value
]
families = ("dcae", "flow")
paths = {family: root / f"{family}.json" for family in families}
payloads = {family: json.loads(path.read_text()) for family, path in paths.items()}
scores = {
    family: payload["selection"]["validation_macro_rmse"]["adapted_all_taus"]
    for family, payload in payloads.items()
}
winner = min(scores, key=scores.get)
selection = {
    "schema_version": 2,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "selection_split": "2020_validation",
    "development_years_opened_before_selection": development_years,
    "confirmatory_years_unopened": confirmatory_years,
    "criterion": "minimum_macro_field_tau_normalized_RMSE_all_hours",
    "winner": winner,
    "validation_scores": scores,
    "adapters": {
        family: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for family, path in paths.items()
    },
}
temporary = root / ".selection.tmp"
temporary.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "selection.json")
print(json.dumps(selection, indent=2))
PY

test_linear() {
  if [[ -s "$EVAL_ROOT/linear.json" ]]; then
    echo "[$(date -Is)] reuse 2021 linear evaluation"
    return 0
  fi
  echo "[$(date -Is)] test linear baseline"
  "$PY" -u tools/eval/batch_eval_forecast_anchor.py \
    --forecast-dir "$HRES_TEST" --target-source era5 \
    --era5-memmap-dir "$ERA5" --model-blob linear --model-name linear \
    --out-json "$EVAL_ROOT/linear.json" --stats-path "$STATS" \
    --surface-stats-path "$SURFACE_STATS" --device cpu \
    --delta-t-hours 6 --taus 1 2 3 4 5 --max-inits 16
}

test_family() {
  local gpu="$1" family="$2"
  configure "$family"
  if [[ -s "$TEST_JSON" ]]; then
    echo "[$(date -Is)] reuse adapted test family=$family"
    return 0
  fi
  exec 8>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
  flock 8
  verify_source
  echo "[$(date -Is)] test adapted family=$family gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u tools/eval/batch_eval_forecast_anchor.py \
    --forecast-dir "$HRES_TEST" --target-source era5 \
    --era5-memmap-dir "$ERA5" --checkpoint "$CHECKPOINT" --arch "$ARCH" \
    --blend-adapter "$ADAPTER" --model-name "${family}_adapted" \
    --out-json "$TEST_JSON" --stats-path "$STATS" \
    --surface-stats-path "$SURFACE_STATS" --static-path "$STATIC" \
    --device cuda --delta-t-hours 6 --taus 1 2 3 4 5 --max-inits 16 \
    --tau-batch-size 5
  flock -u 8
  exec 8>&-
}

test_linear &
pid_linear=$!
test_family 0 flow &
pid_flow=$!
wait "$pid_linear" "$pid_flow"
test_family 0 dcae

"$PY" tools/eval/summarize_nwp_blend.py \
  --selection "$ADAPTER_ROOT/selection.json" \
  --artifact "linear=$EVAL_ROOT/linear.json" \
  --artifact "dcae_adapted=$EVAL_ROOT/dcae_adapted.json" \
  --artifact "flow_adapted=$EVAL_ROOT/flow_adapted.json" \
  --draws 5000 --seed 2027 --test-year "$TEST_YEAR" \
  --test-role "$TEST_ROLE" --output "$EVAL_ROOT/paired_summary.json"

"$PY" - "$EVAL_ROOT" "$SOURCE_SHA" "$ADAPTER_ROOT/selection.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
source_sha = sys.argv[2]
selection_path = Path(sys.argv[3])
names = ("linear", "dcae_adapted", "flow_adapted")
paths = {name: root / f"{name}.json" for name in names}
payloads = {name: json.loads(path.read_text()) for name, path in paths.items()}
indices = {item["paired_artifact"]["window_index_sha256"] for item in payloads.values()}
if len(indices) != 1:
    raise SystemExit("adapted methods do not share one paired 2021 index")
channels = payloads["linear"]["channels"]
cell_wins = {name: 0 for name in names}
macro = {}
for name, payload in payloads.items():
    model_name = payload["model_name"]
    values = []
    for tau in range(1, 6):
        record = payload["per_tau"][str(tau)][model_name]
        values.extend(record[f"rmse_norm_{channel}"] for channel in channels)
    macro[name] = float(np.mean(values))
for tau in range(1, 6):
    for channel in channels:
        winner = min(
            names,
            key=lambda name: payloads[name]["per_tau"][str(tau)][
                payloads[name]["model_name"]
            ][f"rmse_norm_{channel}"],
        )
        cell_wins[winner] += 1
summary = {
    "schema_version": 1,
    "status": "complete",
    "source_sha256": source_sha,
    "window_index_sha256": indices.pop(),
    "macro_field_tau_rmse": macro,
    "field_tau_wins": cell_wins,
    "preselected_winner": json.loads(
        selection_path.read_text()
    )["winner"],
    "artifacts": {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    },
}
temporary = root / ".complete.tmp"
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "complete.json")
print(json.dumps(summary, indent=2))
PY
echo "[$(date -Is)] NWP blend adaptation complete"
