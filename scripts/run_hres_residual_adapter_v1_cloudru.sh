#!/usr/bin/env bash
# Train one adapter-only WeatherBridge HRES trajectory-correction candidate.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
HRES=${HRES:-/tmp/wti_hres_finetune_2017_2020_108x24_v2}
ERA5=${ERA5:-/tmp/wb2_0p5_cache}
STATIC=${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}
STATS=${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}
SURFACE_STATS=${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}
INIT=${INIT:-$LOG_ROOT/hres_forecast_error_augmentation_6h_v1/lead_fea_rmse/epoch=4-step=1010.ckpt}
PROTOCOL=${PROTOCOL:-$SOURCE/repro/hres_residual_adapter_v1.json}
OUT_ROOT=${OUT_ROOT:-$LOG_ROOT/hres_residual_adapter_6h_v1}
EVAL_OUT=${EVAL_OUT:-$RUNTIME/metrics/hres_fea_2021_dev_v1}

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$OUT_ROOT"
exec 9>"$OUT_ROOT/launch.lock"
if ! flock -n 9; then
  echo "HRES residual-adapter queue already active"
  exit 0
fi
exec > >(tee -a "$OUT_ROOT/queue.log") 2>&1

SOURCE_FILES=(
  tools/train/train_capacity_matched_6h.py
  tools/train/training_protocol.py
  tools/eval/capmatched_loader.py
  weather_time_interp/hres_finetune_dataset.py
  weather_time_interp/model/weatherbridge_flow_model.py
  weather_time_interp/normalization.py
  repro/hres_forecast_error_augmentation_v1.json
  repro/hres_residual_adapter_v1.json
  scripts/run_hres_residual_adapter_v1_cloudru.sh
)
SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')

verify_source() {
  local current
  current=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
  [[ "$current" == "$SOURCE_SHA" ]] || {
    echo "source changed during residual-adapter campaign" >&2
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

"$PY" - "$SOURCE" "$PROTOCOL" "$INIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source, protocol_path, checkpoint = map(Path, sys.argv[1:])
protocol = json.loads(protocol_path.read_text())
parent = source / protocol["parent_protocol"]["path"]
if (
    protocol.get("status") != "frozen_before_optimizer_step"
    or hashlib.sha256(parent.read_bytes()).hexdigest()
    != protocol["parent_protocol"]["sha256"]
    or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    != protocol["initialization"]["checkpoint_sha256"]
    or protocol["model"]["trainable_scope"] != "hres_residual_adapter"
):
    raise SystemExit("HRES residual-adapter protocol mismatch")
PY

while pgrep -af "batch_eval_forecast_anchor.py.*$EVAL_OUT" >/dev/null; do
  echo "[$(date -Is)] waiting for frozen HRES-2021 diagnostics"
  sleep 60
done

RUN_DIR="$OUT_ROOT/adapter_rmse"
if [[ ! -s "$RUN_DIR/selection.json" ]]; then
  verify_source
  mkdir -p "$RUN_DIR"
  echo "[$(date -Is)] train adapter-only candidate gpu=0"
  OUT_DIR="$RUN_DIR" CUDA_VISIBLE_DEVICES=0 "$PY" -u \
    tools/train/train_capacity_matched_6h.py \
    --arch flow_pp3_hres_residual --data_source hres_era5 \
    --forecast_train_dir "$HRES" --forecast_val_dir "$HRES" \
    --memmap_dir "$ERA5" --years 2017 2018 2019 --val_years 2020 \
    --window_hours 6 --train_tau_subset 1 3 5 --eval_tau 1 2 3 4 5 \
    --forecast_lead_stride_hours 24 --forecast_max_left_lead_hours 120 \
    --forecast_error_augmentation_repeats 3 \
    --forecast_error_scale_min 0.5 --forecast_error_scale_max 1.5 \
    --forecast_error_augmentation_seed 202707 \
    --bs 2 --val_bs 1 --accumulate 4 --workers 0 --val_workers 0 \
    --train_batches_per_epoch 808 \
    --max_epochs 6 --lr 1e-4 --warmup_steps 5 \
    --trainable_scope hres_residual_adapter --anchor_swap_probability 0 \
    --seed 202707 --gpus 0 --precision bf16-mixed \
    --static_path "$STATIC" --stats_path "$STATS" \
    --surface_stats_path "$SURFACE_STATS" \
    --lambda_hf_override 0.05 --lambda_spec_override 0.02 \
    --lambda_band_override 0 --lambda_sht_override 0 \
    --spectral_mask_profile advected \
    --init_weights_path "$INIT" --ckpt_every_n_epochs 1 \
    --exp_name hres_residual_adapter_s202707
fi

verify_source
"$PY" - "$RUN_DIR" "$PROTOCOL" "$SOURCE_SHA" <<'PY'
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import torch

run_dir, protocol_path, source_sha = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
)
rows = []
for metrics_path in run_dir.glob("lightning_logs/version_*/metrics.csv"):
    for row in csv.DictReader(metrics_path.open()):
        if row.get("val/rmse_mean"):
            rows.append(
                (
                    float(row["val/rmse_mean"]),
                    int(float(row["epoch"])),
                    int(float(row["step"])),
                    metrics_path,
                )
            )
if not rows:
    raise SystemExit("no residual-adapter validation metrics")
metric, epoch, step, metrics_path = min(rows)
matches = []
for checkpoint_path in run_dir.glob(f"*epoch={epoch}-step=*.ckpt"):
    match = re.search(r"epoch=(\d+)-step=(\d+)\.ckpt$", checkpoint_path.name)
    if match and int(match.group(2)) in (step, step + 1):
        matches.append(checkpoint_path)
if len(matches) != 1:
    raise SystemExit(f"ambiguous selected checkpoint: {matches}")
checkpoint_path = matches[0]
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
hparams = checkpoint.get("hyper_parameters", {})
training_protocol = hparams.get("training_protocol", {})
if (
    hparams.get("arch") != "flow_pp3_hres_residual"
    or hparams.get("trainable_scope") != "hres_residual_adapter"
    or training_protocol.get("seed") != 202707
    or training_protocol.get("train_tau_hours") != [1, 3, 5]
    or training_protocol.get("eval_tau_hours") != [1, 2, 3, 4, 5]
):
    raise SystemExit("selected checkpoint metadata mismatch")
payload = {
    "schema_version": 1,
    "status": "selected_on_2020_development_validation",
    "criterion": "minimum_val_rmse_mean_all_five_interior_hours",
    "architecture": "flow_pp3_hres_residual",
    "trainable_scope": "hres_residual_adapter",
    "seed": 202707,
    "epoch": epoch,
    "metrics_step": step,
    "validation_rmse_mean": metric,
    "checkpoint": {
        "path": str(checkpoint_path.resolve()),
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "size_bytes": checkpoint_path.stat().st_size,
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

"$PY" - "$OUT_ROOT" "$PROTOCOL" "$SOURCE_SHA" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root, protocol, source_sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
selection = json.loads((root / "adapter_rmse" / "selection.json").read_text())
payload = {
    "status": "training_complete_selection_frozen",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
    "source_bundle_sha256": source_sha,
    "selection": selection,
}
temporary = root / ".training_complete.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / ".training_complete")
PY
echo "[$(date -Is)] HRES residual-adapter training complete"
