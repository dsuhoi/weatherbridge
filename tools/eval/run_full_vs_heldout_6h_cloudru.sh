#!/usr/bin/env bash
# Evaluate full-hour checkpoints against the existing held-out-hour controls.
# Run one cohort per GPU:
#   bash tools/eval/run_full_vs_heldout_6h_cloudru.sh core 0
#   bash tools/eval/run_full_vs_heldout_6h_cloudru.sh baselines 1
#   bash tools/eval/run_full_vs_heldout_6h_cloudru.sh dcae_control 0
set -euo pipefail

COHORT=${1:?expected cohort: core, baselines, or dcae_control}
GPU=${2:?expected physical GPU index}
SOURCE=${SOURCE:-/tmp/wti_train_6a4728b}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
DATA=${DATA:-/tmp/wb2_0p5_cache}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
OUT_ROOT=${OUT_ROOT:-$RUNTIME/metrics/full_vs_heldout_6h_s202707}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}

STATIC=/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt
STATS=/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc
SURFACE_STATS=/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json
CLIM=/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr
CACHE=${CLIMATOLOGY_CACHE:-/tmp/wti_climatology_cache_corrected_baselines_v1}

MATCHED=/home/jovyan/shares/SR006.nfs2/dsuhoi/6h_query_generalization_matchedbudget_s202707
FULL=/home/jovyan/shares/SR006.nfs2/dsuhoi/6h_full_hours_compile_6a4728b
FULL_LEGACY=/home/jovyan/shares/SR006.nfs2/dsuhoi/6h_full_hours_s202707_58819a5

case "$COHORT" in
  core)
    MODELS=(
      "weatherbridge_heldout|$MATCHED/exp_weatherbridge_6h_sparse135_matchedbudget13136_s202707/last.ckpt|16|"
      "weatherbridge_full|$MATCHED/exp_weatherbridge_6h_fullhours_matchedbudget13136_s202707/last.ckpt|16|"
      "weatherdcae_full|$FULL/exp_weatherdcae_14m_6h_fullhours_s202707/last.ckpt|16|"
    )
    ;;
  baselines)
    MODELS=(
      "pixelattn_full|$FULL/exp_pixelattn_vfi_14m_6h_fullhours_s202707/last.ckpt|16|"
      "swinv2_full|$FULL/exp_swinv2_8m_6h_fullhours_s202707/last.ckpt|16|"
      "sdyff_full|$FULL_LEGACY/exp_sdyff_101m_6h_fullhours_s202707/last.ckpt|8|SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"
    )
    ;;
  dcae_control)
    MODELS=(
      "weatherdcae_heldout|$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt|16|"
    )
    ;;
  *)
    echo "unknown cohort: $COHORT (expected core, baselines, or dcae_control)" >&2
    exit 2
    ;;
esac

cd "$SOURCE"
mkdir -p "$OUT_ROOT" "$CACHE" "$LOG_ROOT"
exec 8>"$LOG_ROOT/.full_vs_heldout_6h_${COHORT}.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] cohort $COHORT is already active" >&2
  exit 1
fi
exec > >(tee -a "$OUT_ROOT/${COHORT}.log") 2>&1

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
export SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0

for path in \
  "$DATA/wb2_2020.bin" "$DATA/wb2_2020.json" \
  "$DATA/wb2_2021.bin" "$DATA/wb2_2021.json" \
  "$STATIC" "$STATS" "$SURFACE_STATS"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
cache_days=$(find "$CACHE" -maxdepth 1 -type f -name '*.npy' | wc -l)
if [[ "$cache_days" -ne 366 ]]; then
  echo "expected a complete 366-day climatology cache at $CACHE; found $cache_days" >&2
  exit 2
fi

artifact_valid() {
  local json_path="$1" checkpoint="$2"
  [[ -s "$json_path" ]] || return 1
  "$PY" - "$json_path" "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
payload = json.loads(json_path.read_text())
checkpoint = str(Path(sys.argv[2]))
assert payload.get("checkpoint_provenance", {}).get("path") == checkpoint
assert set(payload.get("per_tau", {})) == {"1", "2", "3", "4", "5"}
assert payload.get("evaluation_protocol", {}).get("full_year") is True
window = Path(payload["window_metrics_file"])
if not window.is_absolute():
    window = json_path.parent / window
assert window.is_file() and window.stat().st_size > 0
PY
}

for year in 2020 2021; do
  out="$OUT_ROOT/$year"
  mkdir -p "$out"
  for spec in "${MODELS[@]}"; do
    IFS='|' read -r name checkpoint batch envs <<<"$spec"
    [[ -s "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 2; }
    if artifact_valid "$out/$name.json" "$checkpoint"; then
      echo "[$(date -Is)] reuse $year $name"
      continue
    fi
    model_spec="$name:$checkpoint"
    [[ -n "$envs" ]] && model_spec="$model_spec:$envs"
    echo "[$(date -Is)] start $year $name on physical GPU $GPU"
    "$PY" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir "$DATA" --test-year "$year" --climatology "$CLIM" \
      --stats-path "$STATS" --surface-stats-path "$SURFACE_STATS" \
      --static-path "$STATIC" --models "$model_spec" --out-dir "$out" \
      --paper-tag "full_vs_heldout_6h_s202707_${year}_${name}" \
      --batch-size "$batch" --num-workers 2 --samples-per-date 4 --full-year \
      --proper-rmse --save-window-metrics \
      --max-tau-hours 6 --eval-hours 1,2,3,4,5 \
      --seen-tau 1,3,5 --unseen-tau 2,4 --device cuda:0 \
      --lazy-climatology --climatology-cache-dir "$CACHE" \
      --keep-n-channels 24
    artifact_valid "$out/$name.json" "$checkpoint"
    echo "[$(date -Is)] done $year $name"
  done
done

touch "$OUT_ROOT/.${COHORT}.complete"
echo "[$(date -Is)] cohort $COHORT complete"
