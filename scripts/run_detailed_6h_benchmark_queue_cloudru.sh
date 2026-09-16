#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

WAIT_PID="${1:-}"
GPU="${GPU:-1}"
PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
RUNTIME="${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"
OUT="$RUNTIME/metrics/detailed_benchmark_v2/6h"
WEIGHTS="$RUNTIME/weights/detailed_benchmark_v2"
DATA="${DATA:-/tmp/wb2_0p5_cache}"
STATIC="${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}"
STATS="${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}"
SURFACE_STATS="${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}"
CLIM="${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CACHE="${CACHE:-/tmp/wti_climatology_cache_detailed_6h}"
DETAIL="$LOG_ROOT/exp_flow_pp3_detail_14m_6h_refinev1_s202707_bs4/last.ckpt"
REFINE="$LOG_ROOT/exp_flow_universal_latent_refine_14m_6h_s202707_v1_bs4/last.ckpt"
FLOW="$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt"
DCAE="$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt"
LOG="$LOG_ROOT/detailed_benchmark_v2_6h.log"
SOURCE_FILES=(
  scripts/run_detailed_6h_benchmark_queue_cloudru.sh
  tools/eval/batch_eval_12h_memmap.py
  tools/eval/capmatched_loader.py
  tools/eval/eval_artifact_status.py
  tools/eval/paired_block_bootstrap.py
  tools/eval/hard_window_block_bootstrap.py
  tools/eval/paired_aux_block_bootstrap.py
  tools/eval/physical_block_bootstrap.py
  tools/eval/region_season_12h_eval.py
  tools/eval/eval_anchor_exchange_consistency.py
  tools/downstream/eval_physics_consistency.py
  tools/downstream/eval_diurnal_amplitude.py
  tools/train/capmatched_to_bare.py
  tools/train/train_capacity_matched_6h.py
  examples/_bare_loader.py
  weather_time_interp/model/weatherbridge_flow_model.py
  weather_time_interp/model/dcae_adaln_model.py
  weather_time_interp/grid.py
  weather_time_interp/normalization.py
)

compute_source_sha() {
  sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}'
}

SOURCE_SHA=$(compute_source_sha)
if [[ -n "${EXPECTED_SOURCE_SHA:-}" && "$SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]]; then
  echo "source hash mismatch: $SOURCE_SHA != $EXPECTED_SOURCE_SHA" >&2
  exit 2
fi

verify_source_snapshot() {
  local current_sha
  current_sha=$(compute_source_sha)
  if [[ "$current_sha" != "$SOURCE_SHA" ]]; then
    echo "source changed while 6h benchmark was active: $current_sha != $SOURCE_SHA" >&2
    exit 2
  fi
}

mkdir -p "$OUT" "$WEIGHTS" "$CACHE"
exec 8>"$LOG_ROOT/.detailed_benchmark_v2_6h.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] 6h detailed benchmark queue already active"
  exit 0
fi
exec > >(tee -a "$LOG") 2>&1

if [[ -n "$WAIT_PID" ]]; then
  echo "[$(date -Is)] waiting for upstream pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
fi
for checkpoint in "$DETAIL" "$REFINE" "$FLOW" "$DCAE"; do
  while [[ ! -s "$checkpoint" ]]; do
    echo "[$(date -Is)] waiting for checkpoint $checkpoint"
    sleep 120
  done
done
verify_source_snapshot

exec 7>"$LOG_ROOT/.upr_lite_gpu${GPU}.lock"
flock 7

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim:$PWD:${PYTHONPATH:-}"

"$PY" tools/train/capmatched_to_bare.py --ckpt "$DETAIL" \
  --arch flow_pp3_detail --out "$WEIGHTS/weatherbridge_detail_6h_bare.pt"
"$PY" tools/train/capmatched_to_bare.py --ckpt "$REFINE" \
  --arch flow_universal_latent_refine \
  --out "$WEIGHTS/weatherbridge_refine_6h_bare.pt"
"$PY" tools/train/capmatched_to_bare.py --ckpt "$FLOW" \
  --arch flow_pp3 --out "$WEIGHTS/weatherbridge_flow_spectral_6h_bare.pt"
"$PY" tools/train/capmatched_to_bare.py --ckpt "$DCAE" \
  --arch dcae_14m --out "$WEIGHTS/weatherdcae_14m_6yr_6h_bare.pt"

MODELS="weatherbridge_detail:$DETAIL,refine:$REFINE,flow_spectral:$FLOW,weatherdcae_14m_6yr:$DCAE"
MODEL_SPECS=(
  "weatherbridge_detail|$DETAIL"
  "refine|$REFINE"
  "flow_spectral|$FLOW"
  "weatherdcae_14m_6yr|$DCAE"
)

validate_full_year() {
  local directory="$1"
  local spec name checkpoint
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r name checkpoint <<<"$spec"
    "$PY" tools/eval/eval_artifact_status.py \
      "$directory/$name.json" --checkpoint "$checkpoint" \
      --required-taus 1,2,3,4,5 --acc-mode enabled \
      --require-physical-metrics --require-temporal-metrics --quiet || return 1
  done
}

for year in 2020 2021; do
  YEAR_OUT="$OUT/$year/full_year"
  mkdir -p "$YEAR_OUT"
  if [[ -e "$YEAR_OUT/.complete" ]] && ! validate_full_year "$YEAR_OUT"; then
    echo "[$(date -Is)] discard stale full-year marker $YEAR_OUT/.complete"
    rm -f "$YEAR_OUT/.complete"
  fi
  if [[ ! -e "$YEAR_OUT/.complete" ]]; then
    "$PY" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir "$DATA" --test-year "$year" --climatology "$CLIM" \
      --stats-path "$STATS" --surface-stats-path "$SURFACE_STATS" \
      --static-path "$STATIC" --models "$MODELS" --out-dir "$YEAR_OUT" \
      --paper-tag "detailed_6h_${year}_full_year" --batch-size 4 \
      --num-workers 2 --samples-per-date 4 --full-year --proper-rmse \
      --save-window-metrics --save-physical-metrics \
      --save-temporal-metrics --max-tau-hours 6 --eval-hours 1,2,3,4,5 \
      --seen-tau 1,3,5 --unseen-tau 2,4 --device cuda:0 \
      --lazy-climatology --climatology-cache-dir "$CACHE" \
      --keep-n-channels 24
    validate_full_year "$YEAR_OUT"
    touch "$YEAR_OUT/.complete"
  fi

  WINDOW="$YEAR_OUT/window_metrics"
  for candidate in weatherbridge_detail refine flow_spectral weatherdcae_14m_6yr; do
    rights=()
    for reference in weatherbridge_detail refine flow_spectral weatherdcae_14m_6yr; do
      [[ "$reference" == "$candidate" ]] && continue
      rights+=(--right "$reference:$WINDOW/$reference.npz")
    done
    "$PY" tools/eval/paired_block_bootstrap.py \
      --left "$candidate:$WINDOW/$candidate.npz" "${rights[@]}" \
      --draws 5000 --seed 2027 --cellwise \
      --out-json "$YEAR_OUT/paired_rmse_${candidate}.json"
    "$PY" tools/eval/hard_window_block_bootstrap.py \
      --left "$candidate:$WINDOW/$candidate.npz" "${rights[@]}" \
      --quantile 0.95 --draws 5000 --seed 2027 \
      --out-json "$YEAR_OUT/paired_hard_window_${candidate}.json"
    for metric in acc temporal_curvature physical; do
      "$PY" tools/eval/paired_aux_block_bootstrap.py \
        --left "$candidate:$WINDOW/$candidate.npz" "${rights[@]}" \
        --metric "$metric" --draws 5000 --seed 2027 \
        --out-json "$YEAR_OUT/paired_${metric}_${candidate}.json"
    done
  done
  cp "$YEAR_OUT/paired_rmse_weatherbridge_detail.json" \
    "$YEAR_OUT/paired_rmse.json"
  for metric in acc temporal_curvature physical; do
    cp "$YEAR_OUT/paired_${metric}_weatherbridge_detail.json" \
      "$YEAR_OUT/paired_${metric}.json"
  done

  REGION_OUT="$OUT/$year/region_season"
  if [[ ! -e "$REGION_OUT/.complete" ]]; then
    "$PY" -u tools/eval/region_season_12h_eval.py \
      --memmap-dir "$DATA" --test-year "$year" --stats-path "$STATS" \
      --surface-stats-path "$SURFACE_STATS" --static-path "$STATIC" \
      --models "$MODELS" --out-dir "$REGION_OUT" --batch-size 4 \
      --num-workers 2 --samples-per-date 2 --eval-days-per-month 8 \
      --max-tau-hours 6 --eval-hours 2,4 --keep-n-channels 24 \
      --device cuda:0
    touch "$REGION_OUT/.complete"
  fi

  EXCHANGE_OUT="$OUT/$year/anchor_exchange.json"
  if [[ ! -s "$EXCHANGE_OUT" ]]; then
    "$PY" -u tools/eval/eval_anchor_exchange_consistency.py \
      --memmap-dir "$DATA" --test-year "$year" --stats-path "$STATS" \
      --surface-stats-path "$SURFACE_STATS" --static-path "$STATIC" \
      --models "$MODELS" --max-tau-hours 6 --samples-per-date 2 \
      --eval-days-per-month 2 --batch-size 2 --num-workers 2 \
      --keep-n-channels 24 --out-json "$EXCHANGE_OUT" --device cuda:0
  fi

  for item in \
    "weatherbridge_detail:$WEIGHTS/weatherbridge_detail_6h_bare.pt" \
    "refine:$WEIGHTS/weatherbridge_refine_6h_bare.pt" \
    "flow_spectral:$WEIGHTS/weatherbridge_flow_spectral_6h_bare.pt" \
    "weatherdcae_14m_6yr:$WEIGHTS/weatherdcae_14m_6yr_6h_bare.pt" \
    "linear:linear"; do
    IFS=: read -r name blob <<<"$item"
    "$PY" -u tools/downstream/eval_physics_consistency.py \
      --era5-dir "$DATA" --year "$year" --model-blob "$blob" \
      --model-name "$name" --out-json "$OUT/$year/physics_${name}.json" \
      --device cuda:0 --delta-t-hours 6 --taus 1 2 3 4 5 \
      --sample-stride-hours 48 --max-pairs 180 --stats-path "$STATS" \
      --surface-stats-path "$SURFACE_STATS" --static-path "$STATIC"
    "$PY" -u tools/downstream/eval_diurnal_amplitude.py \
      --era5-dir "$DATA" --year "$year" --model-blob "$blob" \
      --model-name "$name" --out-json "$OUT/$year/diurnal_${name}.json" \
      --device cuda:0 --delta-t-hours 6 --taus 1 2 3 4 5 \
      --stats-path "$STATS" --surface-stats-path "$SURFACE_STATS" \
      --static-path "$STATIC"
  done
  verify_source_snapshot
done

verify_source_snapshot
"$PY" - "$OUT/.complete" "$SOURCE_SHA" "${SOURCE_FILES[@]}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
source_sha = sys.argv[2]
sources = tuple(Path(value) for value in sys.argv[3:])
payload = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "horizon_hours": 6,
    "years": [2020, 2021],
    "source_composite_sha256": source_sha,
    "source_sha256": {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in sources
    },
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
echo "[$(date -Is)] 6h detailed benchmark complete"
flock -u 7
