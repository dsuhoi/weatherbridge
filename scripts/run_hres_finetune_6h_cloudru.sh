#!/usr/bin/env bash
# Fine-tune WeatherBridge and WeatherDCAE on matched IFS HRES anchor pairs.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
HRES_CAL=${HRES_CAL:-/tmp/wti_hres_finetune_2017_2020_108x24_v2}
ERA5=${ERA5:-/tmp/wb2_0p5_cache}
STATIC=${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}
STATS=${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}
SURFACE_STATS=${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}
OUT_ROOT=${OUT_ROOT:-$LOG_ROOT/hres_finetune_6h_108x24_v2}
PROTOCOL=${PROTOCOL:-$SOURCE/repro/hres_finetune_protocol.json}
MIN_FREE_KIB=${MIN_FREE_KIB:-180000000}

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$OUT_ROOT"
exec 9>"$OUT_ROOT/launch.lock"
if ! flock -n 9; then
  echo "HRES fine-tuning queue already active"
  exit 0
fi

SOURCE_FILES=(
  tools/train/train_capacity_matched_6h.py
  tools/train/training_protocol.py
  tools/train/select_hres_finetune_checkpoint.py
  tools/data/build_hres_forecast_memmap.py
  tools/eval/batch_eval_forecast_anchor.py
  weather_time_interp/hres_finetune_dataset.py
  weather_time_interp/memmap_dataset.py
  weather_time_interp/grid.py
  weather_time_interp/model/weatherbridge_flow_model.py
  weather_time_interp/model/dcae.py
  weather_time_interp/model/dcae_adaln_model.py
  weather_time_interp/normalization.py
  repro/hres_finetune_protocol.json
  scripts/run_hres_finetune_6h_cloudru.sh
)
SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')

verify_source() {
  local current
  current=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
  [[ "$current" == "$SOURCE_SHA" ]] || {
    echo "source changed during HRES fine-tuning" >&2
    exit 2
  }
}

test -s "$PROTOCOL"
test -s "$STATIC"
test -s "$STATS"
test -s "$SURFACE_STATS"
INIT_CSV=$("$PY" - "$PROTOCOL" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
optimization = payload.get("optimization", {})
splits = payload.get("splits", {})
models = payload.get("models", {})
data_contract = payload.get("data_contract", {})
train = splits.get("optimization", {})
validation = splits.get("validation_and_checkpoint_selection", {})
train_inits = train.get("init_times", [])
validation_inits = validation.get("init_times", [])
expected_objectives = {
    "weatherbridge": {
        "lambda_highpass": 0.05,
        "lambda_fft_magnitude": 0.02,
        "lambda_multiband": 0.0,
        "lambda_sht": 0.0,
        "spectral_mask_profile": "advected",
    },
    "weatherdcae": {
        "lambda_highpass": 0.0,
        "lambda_fft_magnitude": 0.0,
        "lambda_multiband": 0.0,
        "lambda_sht": 0.0,
        "spectral_mask_profile": "all",
    },
}
expected_source_qc = {
    "coarsening": "finite-only unweighted 2x2 block mean",
    "minimum_finite_native_values_per_block": 2,
    "maximum_nonfinite_fraction_per_channel_and_initialisation": 1e-6,
    "output_nonfinite_values": 0,
}
if (
    payload.get("schema_version") != 2
    or payload.get("protocol_revision") != 4
    or payload.get("status") != "frozen_before_hres_weight_finetuning"
    or payload.get("scope") != "six_hour_forecast_anchor_weight_finetuning"
    or any(
        models.get(name, {}).get("fine_tuning_objective") != objective
        for name, objective in expected_objectives.items()
    )
    or data_contract.get("source_quality_policy") != expected_source_qc
    or optimization.get("seeds") != [202707, 202708, 202709]
    or optimization.get("primary_seed") != 202707
    or optimization.get("epochs") != 10
    or optimization.get("learning_rate") != 2e-5
    or optimization.get("batch_size_per_device") != 2
    or optimization.get("gradient_accumulation") != 4
    or train.get("years") != [2017, 2018, 2019]
    or train.get("query_hours") != [1, 3, 5]
    or validation.get("years") != [2020]
    or validation.get("query_hours") != [1, 2, 3, 4, 5]
    or len(train_inits) != 108
    or len(validation_inits) != 24
    or len(set(train_inits + validation_inits)) != 132
):
    raise SystemExit("launcher arguments differ from the frozen HRES protocol")
print(",".join(train_inits + validation_inits))
PY
)

mkdir -p "$HRES_CAL"
if [[ ! -s "$HRES_CAL/forecast_archive_manifest.json" ]]; then
  free_kib=$(df --output=avail "$HRES_CAL" | tail -n 1 | tr -d ' ')
  [[ "$free_kib" -ge "$MIN_FREE_KIB" ]] || {
    echo "need at least $MIN_FREE_KIB KiB free, found $free_kib" >&2
    exit 2
  }
  "$PY" -u tools/data/build_hres_forecast_memmap.py \
    --out-dir "$HRES_CAL" --init-times "$INIT_CSV"
fi
"$PY" - "$PROTOCOL" "$HRES_CAL/forecast_archive_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

protocol = json.loads(Path(sys.argv[1]).read_text())
manifest = json.loads(Path(sys.argv[2]).read_text())
splits = protocol["splits"]
expected = (
    splits["optimization"]["init_times"]
    + splits["validation_and_checkpoint_selection"]["init_times"]
)
if manifest.get("init_times") != expected or len(manifest.get("files", {})) != 132:
    raise SystemExit("HRES training archive differs from the frozen 108/24 index")
PY

configure_model() {
  local family="$1"
  case "$family" in
    weatherbridge)
      ARCH=flow_pp3
      INIT="$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt"
      LOSS_ARGS=(
        --lambda_hf_override 0.05
        --lambda_spec_override 0.02
        --lambda_band_override 0
        --lambda_sht_override 0
        --spectral_mask_profile advected
      )
      ;;
    weatherdcae)
      ARCH=dcae_14m
      INIT="$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt"
      LOSS_ARGS=(
        --lambda_hf_override 0
        --lambda_spec_override 0
        --lambda_band_override 0
        --lambda_sht_override 0
        --spectral_mask_profile all
      )
      ;;
    *) return 2 ;;
  esac
}

train_one() {
  local gpu="$1" family="$2" seed="$3"
  configure_model "$family"
  local run_dir="$OUT_ROOT/${family}_s${seed}"
  local selection="$run_dir/selection.json"
  test -s "$INIT"
  if [[ -s "$selection" ]]; then
    "$PY" - "$selection" "$ARCH" "$seed" "$PROTOCOL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
checkpoint = Path(payload.get("checkpoint", {}).get("path", ""))
if (
    payload.get("status") != "selected_on_2020_validation"
    or payload.get("architecture") != sys.argv[2]
    or payload.get("seed") != int(sys.argv[3])
    or payload.get("protocol_sha256")
    != hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest()
    or not checkpoint.is_file()
    or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    != payload.get("checkpoint", {}).get("sha256")
):
    raise SystemExit(f"invalid existing HRES selection: {path}")
PY
    echo "[$(date -Is)] reuse selected family=$family seed=$seed"
    return 0
  fi
  verify_source
  echo "[$(date -Is)] train family=$family seed=$seed gpu=$gpu"
  OUT_DIR="$run_dir" CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u \
    tools/train/train_capacity_matched_6h.py \
    --arch "$ARCH" --data_source hres_era5 \
    --forecast_train_dir "$HRES_CAL" --forecast_val_dir "$HRES_CAL" \
    --memmap_dir "$ERA5" --years 2017 2018 2019 --val_years 2020 \
    --window_hours 6 --train_tau_subset 1 3 5 --eval_tau 1 2 3 4 5 \
    --forecast_lead_stride_hours 24 --forecast_max_left_lead_hours 120 \
    --bs 2 --val_bs 1 --accumulate 4 --workers 0 --val_workers 0 \
    --max_epochs 10 --lr 2e-5 --warmup_steps 10 \
    --trainable_scope all --anchor_swap_probability 0 \
    --seed "$seed" --gpus 0 --precision bf16-mixed \
    --static_path "$STATIC" --stats_path "$STATS" \
    --surface_stats_path "$SURFACE_STATS" \
    "${LOSS_ARGS[@]}" \
    --init_weights_path "$INIT" --ckpt_every_n_epochs 1 \
    --exp_name "hres_finetune_${family}_s${seed}"
  verify_source
  "$PY" tools/train/select_hres_finetune_checkpoint.py \
    --run-dir "$run_dir" --expected-arch "$ARCH" \
    --expected-seed "$seed" --protocol "$PROTOCOL" \
    --output "$selection"
}

for seed in 202707 202708 202709; do
  train_one 0 weatherbridge "$seed" &
  pid_bridge=$!
  train_one 1 weatherdcae "$seed" &
  pid_dcae=$!
  wait "$pid_bridge" "$pid_dcae"
done

verify_source
"$PY" - "$OUT_ROOT" "$PROTOCOL" "$SOURCE_SHA" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
protocol = Path(sys.argv[2])
source_sha = sys.argv[3]
selections = sorted(root.glob("*_s*/selection.json"))
if len(selections) != 6:
    raise SystemExit("expected six selected HRES fine-tuning runs")
payload = {
    "schema_version": 1,
    "status": "training_complete_selection_frozen",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "protocol_path": str(protocol.resolve()),
    "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
    "source_sha256": source_sha,
    "selections": {
        path.parent.name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in selections
    },
    "confirmation_status": "not_evaluated",
}
temporary = root / ".training_complete.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / ".training_complete")
PY

echo "[$(date -Is)] HRES fine-tuning training and 2020 selection complete"
