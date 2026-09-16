#!/usr/bin/env bash
# Degradation-aware HRES fine-tuning, matched two-arm ablation.
#
#   degrade_rmse    flow_pp3_degrade   reliability branch + anchor-error supervision
#   augcontrol_rmse flow_pp3_hres_aug  identical run WITHOUT the branch
#
# Both warm-start from the same full-lead checkpoint and see the same anchor
# pairs, augmentation, seed, learning rate and update budget, so the only
# difference between them is the reliability branch.
set -euo pipefail

PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
SOURCE=/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6
LOG_ROOT=/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs
HRES=${HRES:-/tmp/wti_hres_finetune_2017_2020_108x24_v2}
ERA5=${ERA5:-/tmp/wb2_0p5_cache}
STATIC=/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt
STATS=/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc
SURFACE_STATS=/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json
INIT=${INIT:-$LOG_ROOT/hres_full_lead_finetune_6h_v1/fulllead_rmse/epoch=5-step=1212.ckpt}
OUT_ROOT=${OUT_ROOT:-$LOG_ROOT/hres_degradation_aware_6h_v1}

export ANCHOR_RELIABILITY_WEIGHT=${ANCHOR_RELIABILITY_WEIGHT:-0.05}
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
cd "$SOURCE"
mkdir -p "$OUT_ROOT"
exec 9>"$OUT_ROOT/launch.lock"
if ! flock -n 9; then
  echo "degradation-aware queue already active"
  exit 0
fi
exec > >(tee -a "$OUT_ROOT/queue.log") 2>&1

[[ -s "$INIT" ]] || { echo "missing warm-start checkpoint $INIT" >&2; exit 2; }

train_arm() {
  local gpu="$1" arm="$2" arch="$3"
  local run_dir="$OUT_ROOT/$arm"
  if [[ -s "$run_dir/done" ]]; then
    echo "[$(date -Is)] reuse arm=$arm"
    return 0
  fi
  mkdir -p "$run_dir"
  echo "[$(date -Is)] train arm=$arm arch=$arch gpu=$gpu lambda_rel=$ANCHOR_RELIABILITY_WEIGHT"
  OUT_DIR="$run_dir" CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u \
    tools/train/train_capacity_matched_6h.py \
    --arch "$arch" --data_source hres_era5 \
    --forecast_train_dir "$HRES" --forecast_val_dir "$HRES" \
    --memmap_dir "$ERA5" --years 2017 2018 2019 --val_years 2020 \
    --window_hours 6 --train_tau_subset 1 3 5 --eval_tau 1 2 3 4 5 \
    --forecast_lead_stride_hours 24 --forecast_max_left_lead_hours 240 \
    --forecast_error_augmentation_repeats 3 \
    --forecast_error_scale_min 0.5 --forecast_error_scale_max 1.5 \
    --forecast_error_augmentation_seed 202707 \
    --bs 2 --val_bs 1 --accumulate 4 --workers 0 --val_workers 0 \
    --train_batches_per_epoch 808 \
    --max_epochs 8 --lr 1e-5 --warmup_steps 10 \
    --trainable_scope all --anchor_swap_probability 0 \
    --seed 202707 --gpus 0 --precision bf16-mixed \
    --static_path "$STATIC" --stats_path "$STATS" \
    --surface_stats_path "$SURFACE_STATS" \
    --lambda_hf_override 0.05 --lambda_spec_override 0.02 \
    --lambda_band_override 0 --lambda_sht_override 0 \
    --spectral_mask_profile advected \
    --init_weights_path "$INIT" --ckpt_every_n_epochs 1 \
    --exp_name "hres_${arm}_s202707"
  date -Is > "$run_dir/done"
}

train_arm 0 degrade_rmse flow_pp3_degrade &
pid_a=$!
train_arm 1 augcontrol_rmse flow_pp3_hres_aug &
pid_b=$!
wait "$pid_a" "$pid_b"

# Select the best epoch of each arm on the 2020 development validation.
"$PY" - "$OUT_ROOT" <<'PY'
import csv
import hashlib
import json
import re
from pathlib import Path

root = Path(__import__("sys").argv[1])
summary = {}
for arm in ("degrade_rmse", "augcontrol_rmse"):
    run_dir = root / arm
    rows = []
    for metrics_path in run_dir.glob("lightning_logs/version_*/metrics.csv"):
        for row in csv.DictReader(metrics_path.open()):
            if row.get("val/rmse_mean"):
                rows.append(
                    (
                        float(row["val/rmse_mean"]),
                        int(float(row["epoch"])),
                        int(float(row["step"])),
                    )
                )
    if not rows:
        summary[arm] = {"status": "no_validation_metrics"}
        continue
    metric, epoch, step = min(rows)
    matches = [
        path
        for path in run_dir.glob(f"*epoch={epoch}-step=*.ckpt")
        if (match := re.search(r"-step=(\d+)\.ckpt$", path.name))
        and int(match.group(1)) in (step, step + 1)
    ]
    entry = {
        "validation_rmse_mean": metric,
        "epoch": epoch,
        "metrics_step": step,
        "checkpoint": str(matches[0].resolve()) if len(matches) == 1 else None,
    }
    if len(matches) == 1:
        entry["checkpoint_sha256"] = hashlib.sha256(
            matches[0].read_bytes()
        ).hexdigest()
    summary[arm] = entry
(root / "selection.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
echo "[$(date -Is)] degradation-aware fine-tuning complete"
