#!/usr/bin/env bash
# Evaluate the preregistered 2x2 architecture x objective matrix on identical
# full-year 6 h ERA5 windows. A completion marker is written only after every
# artifact and paired comparison passes validation.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
DATA=${DATA:-/tmp/wb2_0p5_cache}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
MIN_FREE_GIB=${MIN_FREE_GIB:-20}
OUT="$RUNTIME/metrics/loss_architecture_factorial_v1"
STATE="$OUT/state"
LOG="$LOG_ROOT/loss_architecture_factorial_v1.eval.log"

STATIC=/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt
STATS=/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc
SURFACE_STATS=/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json
CLIM=/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr
CACHE=/tmp/wti_climatology_cache_loss_architecture_factorial_v1

SOURCE_FILES=(
  scripts/run_loss_architecture_factorial_eval_cloudru.sh
  tools/eval/batch_eval_12h_memmap.py
  tools/eval/eval_artifact_status.py
  tools/eval/paired_block_bootstrap.py
  tools/eval/hard_window_block_bootstrap.py
  tools/eval/paired_aux_block_bootstrap.py
  tools/eval/capmatched_loader.py
  weather_time_interp/eval_runner.py
  weather_time_interp/grid.py
  weather_time_interp/normalization.py
  weather_time_interp/metrics/physical_consistency.py
)

cd "$SOURCE"
mkdir -p "$OUT" "$STATE" "$CACHE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
exec 9>"$STATE/launch.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] factorial evaluation already active"
  exit 0
fi
exec > >(tee -a "$LOG") 2>&1

for year in 2020 2021; do
  for suffix in bin json; do
    path="$DATA/wb2_${year}.${suffix}"
    [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
  done
done
for path in "$STATIC" "$STATS" "$SURFACE_STATS"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done

free_kib=$(df -Pk "$RUNTIME" | awk 'NR == 2 {print $4}')
if (( free_kib < MIN_FREE_GIB * 1024 * 1024 )); then
  echo "less than ${MIN_FREE_GIB} GiB free at $RUNTIME" >&2
  exit 2
fi

SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
verify_source_snapshot() {
  local current
  current=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
  [[ "$current" == "$SOURCE_SHA" ]] || {
    echo "source changed during factorial evaluation" >&2
    return 2
  }
}

checkpoint_for() {
  local family="$1" seed="$2" run
  case "$family" in
    weatherbridge_spectral) run="exp_flow_pp3_spectral_14m_6h_refinev1_s${seed}_bs4" ;;
    weatherbridge_pixel) run="exp_flow_pp3_pixel_14m_6h_refinev1_s${seed}_bs4" ;;
    weatherdcae_spectral) run="exp_weatherdcae_spectral_14m_6h_refinev1_s${seed}_bs4" ;;
    weatherdcae_pixel) run="exp_weatherdcae_14m_6h_6yr_refinev1_s${seed}_bs4" ;;
    *) return 2 ;;
  esac
  printf '%s/%s/last.ckpt' "$LOG_ROOT" "$run"
}

output_valid() {
  local year="$1" family="$2" seed="$3" checkpoint artifact
  checkpoint=$(checkpoint_for "$family" "$seed")
  artifact="$OUT/$year/${family}_s${seed}.json"
  "$PY" tools/eval/eval_artifact_status.py "$artifact" \
    --checkpoint "$checkpoint" --required-taus 1,2,3,4,5 \
    --acc-mode enabled --require-physical-metrics \
    --require-temporal-metrics --quiet
}

run_one() {
  local gpu="$1" year="$2" family="$3" seed="$4"
  local checkpoint name out marker
  checkpoint=$(checkpoint_for "$family" "$seed")
  name="${family}_s${seed}"
  out="$OUT/$year"
  marker="$STATE/${year}_${name}.done"
  [[ -s "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; return 2; }
  mkdir -p "$out"
  if output_valid "$year" "$family" "$seed"; then
    echo "[$(date -Is)] reuse validated $year $name"
    touch "$marker"
    return
  fi
  rm -f "$marker" "$STATE/${year}_${name}.failed"
  echo "[$(date -Is)] evaluate gpu=$gpu year=$year model=$name"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir "$DATA" --test-year "$year" --climatology "$CLIM" \
    --stats-path "$STATS" --surface-stats-path "$SURFACE_STATS" \
    --static-path "$STATIC" --models "$name:$checkpoint" --out-dir "$out" \
    --paper-tag "loss_architecture_factorial_v1_6h_${year}_${name}" \
    --batch-size 12 --num-workers 4 --samples-per-date 4 --full-year \
    --proper-rmse --save-window-metrics --save-physical-metrics \
    --save-temporal-metrics --max-tau-hours 6 --eval-hours 1,2,3,4,5 \
    --seen-tau 1,3,5 --unseen-tau 2,4 --device cuda:0 \
    --lazy-climatology --climatology-cache-dir "$CACHE/gpu${gpu}" \
    --keep-n-channels 24
  output_valid "$year" "$family" "$seed"
  verify_source_snapshot
  touch "$marker"
}

worker() {
  local gpu="$1" parity="$2" index=0 year family seed
  exec 8>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
  flock 8
  for year in 2020 2021; do
    for seed in 202707 202708 202709; do
      for family in weatherbridge_spectral weatherbridge_pixel weatherdcae_spectral weatherdcae_pixel; do
        if (( index % 2 == parity )); then
          if ! run_one "$gpu" "$year" "$family" "$seed"; then
            touch "$STATE/${year}_${family}_s${seed}.failed"
            return 1
          fi
        fi
        index=$((index + 1))
      done
    done
  done
}

worker 0 0 & worker0=$!
worker 1 1 & worker1=$!
status=0
wait "$worker0" || status=1
wait "$worker1" || status=1
(( status == 0 )) || exit "$status"

pair_labels=(
  loss_weatherbridge
  loss_weatherdcae
  architecture_spectral
  architecture_pixel
)
pair_left=(
  weatherbridge_spectral
  weatherdcae_spectral
  weatherbridge_spectral
  weatherbridge_pixel
)
pair_right=(
  weatherbridge_pixel
  weatherdcae_pixel
  weatherdcae_spectral
  weatherdcae_pixel
)

for year in 2020 2021; do
  mkdir -p "$OUT/$year/paired"
  for seed in 202707 202708 202709; do
    for index in 0 1 2 3; do
      label=${pair_labels[$index]}
      left=${pair_left[$index]}
      right=${pair_right[$index]}
      left_npz="$OUT/$year/window_metrics/${left}_s${seed}.npz"
      right_npz="$OUT/$year/window_metrics/${right}_s${seed}.npz"
      prefix="$OUT/$year/paired/${label}_s${seed}"
      "$PY" tools/eval/paired_block_bootstrap.py \
        --left "$left:$left_npz" --right "$right:$right_npz" \
        --draws 10000 --seed 20220809 --cellwise \
        --out-json "${prefix}_rmse.json"
      "$PY" tools/eval/paired_block_bootstrap.py \
        --left "$left:$left_npz" --right "$right:$right_npz" \
        --taus 2,4 --draws 10000 --seed 20220809 --cellwise \
        --out-json "${prefix}_rmse_held.json"
      "$PY" tools/eval/hard_window_block_bootstrap.py \
        --left "$left:$left_npz" --right "$right:$right_npz" \
        --quantile 0.95 --draws 10000 --seed 20220809 \
        --out-json "${prefix}_hard_window.json"
      for metric in acc temporal_curvature physical; do
        "$PY" tools/eval/paired_aux_block_bootstrap.py \
          --left "$left:$left_npz" --right "$right:$right_npz" \
          --metric "$metric" --draws 10000 --seed 20220809 \
          --out-json "${prefix}_${metric}.json"
      done
    done
  done
done

for year in 2020 2021; do
  for seed in 202707 202708 202709; do
    for family in weatherbridge_spectral weatherbridge_pixel weatherdcae_spectral weatherdcae_pixel; do
      output_valid "$year" "$family" "$seed"
    done
  done
done
failed_count=$(find "$STATE" -maxdepth 1 -type f -name '*.failed' | wc -l)
done_count=$(find "$STATE" -maxdepth 1 -type f -name '*.done' | wc -l)
[[ "$failed_count" -eq 0 && "$done_count" -eq 24 ]] || exit 1
verify_source_snapshot

"$PY" - "$OUT/.complete" "$SOURCE_SHA" "$OUT" "${SOURCE_FILES[@]}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
source_sha = sys.argv[2]
root = Path(sys.argv[3])
sources = [Path(value) for value in sys.argv[4:]]
artifacts = sorted(
    path for path in root.rglob("*")
    if path.is_file() and path.name != ".complete" and "state" not in path.parts
)
payload = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "protocol": {
        "task": "6 h ERA5 temporal interpolation",
        "years": [2020, 2021],
        "query_hours": [1, 2, 3, 4, 5],
        "held_query_hours": [2, 4],
        "channels": 24,
        "seeds": [202707, 202708, 202709],
        "design": "2x2 architecture x objective",
    },
    "source_composite_sha256": source_sha,
    "source_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    },
    "artifact_sha256": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in artifacts
    },
}
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, target)
PY

echo "[$(date -Is)] loss x architecture evaluation complete"
