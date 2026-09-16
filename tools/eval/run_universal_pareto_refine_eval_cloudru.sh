#!/usr/bin/env bash
# Full-year 2020 evaluation and paired tests for Universal-Pareto-Refine.
set -euo pipefail

PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
SOURCE="${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_universal_pareto_refine_v1_source}"
RUNTIME="${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
DATA="${DATA:-/tmp/wb2_0p5_cache}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}"
CLIM="${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
CACHE="${CACHE:-/tmp/wti_climatology_cache_pareto_refine}"
GPU="${GPU:-0}"
OUT="$RUNTIME/metrics/universal_pareto_refine_v1/2020/full_year"
BASE="$RUNTIME/metrics/detailed_benchmark_v2/6h/2020/full_year"
LOG="$LOG_ROOT/universal_pareto_refine_eval_v1.log"
SPARSE="$LOG_ROOT/exp_flow_universal_pareto_refine_14m_6h_sparse_s202710_v1_bs4/last.ckpt"
ALLTAU="$LOG_ROOT/exp_flow_universal_pareto_refine_14m_6h_alltau_s202710_v1_bs4/last.ckpt"

mkdir -p "$OUT" "$CACHE"
exec 9>"$OUT/launch.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] Pareto-Refine evaluation already active"
  exit 0
fi
exec > >(tee -a "$LOG") 2>&1

(
  cd "$SOURCE"
  sha256sum --quiet -c SOURCE_SHA256SUMS
)
test -s "$SPARSE"
test -s "$ALLTAU"
for reference in weatherbridge_detail refine flow_spectral weatherdcae_14m_6yr; do
  test -s "$BASE/window_metrics/$reference.npz"
done

exec 8>"$LOG_ROOT/.upr_lite_gpu${GPU}.lock"
flock 8
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim:${PYTHONPATH:-}"

validate_candidate() {
  local name="$1" checkpoint="$2"
  "$PY" "$SOURCE/tools/eval/eval_artifact_status.py" \
    "$OUT/$name.json" --checkpoint "$checkpoint" \
    --required-taus 1,2,3,4,5 --acc-mode enabled \
    --require-physical-metrics --require-temporal-metrics --quiet
}

if [[ ! -e "$OUT/evaluation.complete" ]] || \
   ! validate_candidate pareto_sparse "$SPARSE" || \
   ! validate_candidate pareto_alltau "$ALLTAU"; then
  rm -f "$OUT/evaluation.complete"
  "$PY" -u "$SOURCE/tools/eval/batch_eval_12h_memmap.py" \
    --memmap-dir "$DATA" --test-year 2020 --climatology "$CLIM" \
    --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
    --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
    --static-path "$DATA_ROOT/static_features_0p5.pt" \
    --models "pareto_sparse:$SPARSE,pareto_alltau:$ALLTAU" \
    --out-dir "$OUT" --paper-tag "universal_pareto_refine_6h_2020" \
    --batch-size 4 --num-workers 2 --samples-per-date 4 --full-year \
    --proper-rmse --save-window-metrics --save-physical-metrics \
    --save-temporal-metrics --max-tau-hours 6 --eval-hours 1,2,3,4,5 \
    --seen-tau 1,3,5 --unseen-tau 2,4 --device cuda:0 \
    --lazy-climatology --climatology-cache-dir "$CACHE" \
    --keep-n-channels 24
  validate_candidate pareto_sparse "$SPARSE"
  validate_candidate pareto_alltau "$ALLTAU"
  touch "$OUT/evaluation.complete"
fi

for candidate in pareto_sparse pareto_alltau; do
  rights=()
  for reference in weatherbridge_detail refine flow_spectral weatherdcae_14m_6yr; do
    rights+=(--right "$reference:$BASE/window_metrics/$reference.npz")
  done
  "$PY" "$SOURCE/tools/eval/paired_block_bootstrap.py" \
    --left "$candidate:$OUT/window_metrics/$candidate.npz" "${rights[@]}" \
    --draws 5000 --seed 2027 --cellwise \
    --out-json "$OUT/paired_rmse_$candidate.json"
  "$PY" "$SOURCE/tools/eval/hard_window_block_bootstrap.py" \
    --left "$candidate:$OUT/window_metrics/$candidate.npz" "${rights[@]}" \
    --quantile 0.95 --draws 5000 --seed 2027 \
    --out-json "$OUT/paired_hard_window_$candidate.json"
  for metric in acc temporal_curvature physical; do
    "$PY" "$SOURCE/tools/eval/paired_aux_block_bootstrap.py" \
      --left "$candidate:$OUT/window_metrics/$candidate.npz" "${rights[@]}" \
      --metric "$metric" --draws 5000 --seed 2027 \
      --out-json "$OUT/paired_${metric}_$candidate.json"
  done
done

touch "$OUT/paired.complete"
echo "[$(date -Is)] Universal-Pareto-Refine full-year evaluation complete"
