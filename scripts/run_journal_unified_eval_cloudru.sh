#!/usr/bin/env bash
# Unified 6 h evaluation used by the journal manuscript.
#
# Runs the same full-year windows, RMSE reduction, ACC climatology, and channel
# order for every model. Results include checkpoint hashes and paired per-window
# arrays for block-bootstrap significance tests.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LEGACY=${LEGACY:-/home/jovyan/dsuhoi/weather_time_interpolation}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
GPU=${GPU:-0}
YEARS=${YEARS:-"2020 2021"}
FORCE=${FORCE:-0}
CLIMATOLOGY_CACHE=${CLIMATOLOGY_CACHE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/climatology_eval_cache_24ch}

cd "$RUNTIME"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$RUNTIME:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
export PATH=/home/jovyan/.mlspace/envs/ai_scientist/bin:/usr/local/bin:/usr/bin:/bin

MODELS="fuxi_24ch_6yr_ep8:$LEGACY/logs/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt:,\
modafno_24ch_6yr_ep8:$LEGACY/logs/exp_modafno_24ch_6yr/epoch=7-step=35032.ckpt:MODAFNO_INP_H=360 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720,\
sdyff_24ch_6yr_ep8:$LEGACY/logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt:SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0,\
weatherbridge_pp3_14m_6yr_ep8:/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs/exp_flow_pp3_135_14m_6h/last.ckpt:"

mkdir -p logs/runner "$CLIMATOLOGY_CACHE" metrics/journal_unified
LOG=logs/runner/journal_unified_eval_6h.log
LOCK=/tmp/weatherbridge_journal_eval_6h.lock

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another unified 6 h evaluation owns $LOCK" >&2
  exit 1
fi

for year in $YEARS; do
  out="metrics/journal_unified/6h_${year}"
  if [[ "$FORCE" != 1 && -s "$out/weatherbridge_pp3_14m_6yr_ep8.json" ]]; then
    echo "[$(date -Is)] skip complete $out" | tee -a "$LOG"
    continue
  fi
  mkdir -p "$out"
  echo "[$(date -Is)] start 6 h year=$year GPU=$GPU" | tee -a "$LOG"
  "$PY" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year "$year" \
    --climatology /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr \
    --climatology-cache-dir "$CLIMATOLOGY_CACHE" \
    --lazy-climatology \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --models "$MODELS" \
    --out-dir "$out" \
    --paper-tag "journal_6h_${year}_full_year" \
    --batch-size 4 --num-workers 2 \
    --samples-per-date 4 --full-year \
    --max-tau-hours 6 --eval-hours 1,2,3,4,5 \
    --seen-tau 1,3,5 --unseen-tau 2,4 \
    --keep-n-channels 24 \
    --proper-rmse --save-window-metrics \
    2>&1 | tee -a "$LOG"
  echo "[$(date -Is)] done 6 h year=$year" | tee -a "$LOG"
done
