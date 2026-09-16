#!/usr/bin/env bash
# Train two primary-seed WeatherBridge forecast-error augmentation arms.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
HRES=${HRES:-/tmp/wti_hres_finetune_2017_2020_108x24_v2}
ERA5=${ERA5:-/tmp/wb2_0p5_cache}
STATIC=${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}
STATS=${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}
SURFACE_STATS=${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}
INIT=${INIT:-$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt}
PROTOCOL=${PROTOCOL:-$SOURCE/repro/hres_forecast_error_augmentation_v1.json}
OUT_ROOT=${OUT_ROOT:-$LOG_ROOT/hres_forecast_error_augmentation_6h_v1}

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$OUT_ROOT"
exec 9>"$OUT_ROOT/launch.lock"
if ! flock -n 9; then
  echo "forecast-error augmentation queue already active"
  exit 0
fi

SOURCE_FILES=(
  tools/train/train_capacity_matched_6h.py
  tools/train/training_protocol.py
  weather_time_interp/hres_finetune_dataset.py
  weather_time_interp/model/weatherbridge_flow_model.py
  weather_time_interp/normalization.py
  repro/hres_finetune_protocol.json
  repro/hres_forecast_error_augmentation_v1.json
  scripts/run_hres_forecast_error_augmentation_v1_cloudru.sh
)
SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')

verify_source() {
  local current
  current=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
  [[ "$current" == "$SOURCE_SHA" ]] || {
    echo "source changed during forecast-error augmentation training" >&2
    exit 2
  }
}

test -s "$HRES/forecast_archive_manifest.json"
test -s "$ERA5/wb2_2017.json"
test -s "$ERA5/wb2_2020.json"
test -s "$STATIC"
test -s "$STATS"
test -s "$SURFACE_STATS"
test -s "$INIT"
test -s "$PROTOCOL"

"$PY" - "$SOURCE" "$PROTOCOL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
protocol = json.loads(Path(sys.argv[2]).read_text())
parent = source / protocol["parent_protocol"]["path"]
if (
    protocol.get("status") != "frozen_before_optimizer_step"
    or hashlib.sha256(parent.read_bytes()).hexdigest()
    != protocol["parent_protocol"]["sha256"]
    or protocol["data"]["augmentation"] != {
        "training_only": True,
        "method": "scale the paired HRES-minus-ERA5 endpoint residuals around their matching clean ERA5 anchors",
        "repeats": 3,
        "scale_min": 0.5,
        "scale_max": 1.5,
        "includes_exact_hres_scale": True,
        "independent_endpoint_noise": False,
    }
):
    raise SystemExit("forecast-error augmentation protocol mismatch")
PY

select_best() {
  local run_dir="$1" arm="$2"
  "$PY" - "$run_dir" "$arm" "$PROTOCOL" "$SOURCE_SHA" <<'PY'
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

run_dir, arm, protocol_path, source_sha = (
    Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
)
rows = []
for path in run_dir.glob("lightning_logs/version_*/metrics.csv"):
    for row in csv.DictReader(path.open()):
        if row.get("val/rmse_mean"):
            rows.append((float(row["val/rmse_mean"]), int(float(row["epoch"])), int(float(row["step"])), path))
if not rows:
    raise SystemExit(f"{run_dir}: no validation metrics")
metric, epoch, step, metrics_path = min(rows)
matches = []
for path in run_dir.glob(f"*epoch={epoch}-step=*.ckpt"):
    match = re.search(r"epoch=(\d+)-step=(\d+)\.ckpt$", path.name)
    if match and int(match.group(2)) in (step, step + 1):
        matches.append(path)
if len(matches) != 1:
    raise SystemExit(f"{run_dir}: ambiguous selected checkpoint {matches}")
checkpoint = matches[0]
payload = {
    "schema_version": 1,
    "status": "selected_on_2020_development_validation",
    "arm": arm,
    "architecture": "flow_pp3_hres_aug",
    "seed": 202707,
    "validation_rmse_mean": metric,
    "epoch": epoch,
    "metrics_step": step,
    "checkpoint": {
        "path": str(checkpoint.resolve()),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    },
    "metrics_csv_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
    "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
    "source_bundle_sha256": source_sha,
}
temporary = run_dir / ".selection.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, run_dir / "selection.json")
print(json.dumps(payload, sort_keys=True))
PY
}

train_arm() {
  local gpu="$1" arm="$2" sht="$3"
  local run_dir="$OUT_ROOT/$arm"
  if [[ -s "$run_dir/selection.json" ]]; then
    echo "[$(date -Is)] reuse arm=$arm"
    return 0
  fi
  verify_source
  mkdir -p "$run_dir"
  echo "[$(date -Is)] train arm=$arm gpu=$gpu lambda_sht=$sht"
  local sht_args=(--lambda_sht_override "$sht")
  if [[ "$sht" != "0" ]]; then
    sht_args+=(--sht_ell_min 80 --sht_lmax 180 --sht_start_fraction 0.4 --sht_every_n_steps 8)
  fi
  OUT_DIR="$run_dir" CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u \
    tools/train/train_capacity_matched_6h.py \
    --arch flow_pp3_hres_aug --data_source hres_era5 \
    --forecast_train_dir "$HRES" --forecast_val_dir "$HRES" \
    --memmap_dir "$ERA5" --years 2017 2018 2019 --val_years 2020 \
    --window_hours 6 --train_tau_subset 1 3 5 --eval_tau 1 2 3 4 5 \
    --forecast_lead_stride_hours 24 --forecast_max_left_lead_hours 120 \
    --forecast_error_augmentation_repeats 3 \
    --forecast_error_scale_min 0.5 --forecast_error_scale_max 1.5 \
    --forecast_error_augmentation_seed 202707 \
    --bs 2 --val_bs 1 --accumulate 4 --workers 0 --val_workers 0 \
    --train_batches_per_epoch 808 \
    --max_epochs 10 --lr 2e-5 --warmup_steps 10 \
    --trainable_scope all --anchor_swap_probability 0 \
    --seed 202707 --gpus 0 --precision bf16-mixed \
    --static_path "$STATIC" --stats_path "$STATS" \
    --surface_stats_path "$SURFACE_STATS" \
    --lambda_hf_override 0.05 --lambda_spec_override 0.02 \
    --lambda_band_override 0 --spectral_mask_profile advected \
    "${sht_args[@]}" \
    --init_weights_path "$INIT" --ckpt_every_n_epochs 1 \
    --exp_name "hres_${arm}_s202707"
  verify_source
  select_best "$run_dir" "$arm"
}

train_arm 0 lead_fea_rmse 0 &
pid_rmse=$!
train_arm 1 lead_fea_sht 0.01 &
pid_sht=$!
wait "$pid_rmse" "$pid_sht"

verify_source
"$PY" - "$OUT_ROOT" "$PROTOCOL" "$SOURCE_SHA" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root, protocol, source_sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
selections = [json.loads((root / arm / "selection.json").read_text()) for arm in ("lead_fea_rmse", "lead_fea_sht")]
payload = {
    "status": "training_complete_selection_frozen",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
    "source_bundle_sha256": source_sha,
    "selections": selections,
}
temporary = root / ".training_complete.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / ".training_complete")
PY
echo "[$(date -Is)] forecast-error augmentation training complete"
