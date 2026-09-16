#!/usr/bin/env bash
# One horizon per GPU; preserve all original checkpoint and evaluation cohorts.
set -euo pipefail
ROOT=${ROOT:-/workspace-SR006.nfs2/dsuhoi/sdyff_ens_20260914_n21}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
GPU=${GPU:?set GPU to a free device index}
HORIZON=${HORIZON:?set HORIZON to 6 or 12}
MODE=${MODE:-full}
LEGACY=/home/jovyan/dsuhoi/weather_time_interpolation
SHARED=/workspace-SR006.nfs2/dsuhoi
cd "$ROOT/source"
mkdir -p "$ROOT/logs" "$ROOT/metrics"
exec 7>"$SHARED/capmatched_logs/.upr_lite_gpu${GPU}.lock"
flock -n 7 || { echo "GPU queue lock busy"; exit 2; }
# Check twice under the shared lock; do not preempt uncoordinated GPU jobs.
for check in 1 2; do
  memory=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
  [[ "$memory" -lt 100 ]] || { echo "GPU $GPU is occupied ($memory MiB)"; exit 2; }
  sleep 3
done
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/source:$ROOT/source/legacy/scripts:$SHARED/wti_shim"
export SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
case "$HORIZON" in
  6)
    CKPT="$LEGACY/logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt"
    HASH=ffcd25513bd7a286a4edb583e07ebb3dab9f761f4d5b41e8f99cd3ee076bc760
    HOURS=1,2,3,4,5; SEEN=1,3,5; HELD=2,4; SAMPLES=4 ;;
  12)
    CKPT="$LEGACY/logs/_12h_migrated/sdyff_12h.ckpt"
    HASH=b03a81706c51dfbfc670e917f3e020d1cefd12a43b5f1061d4de234cf716f98b
    HOURS=1,2,3,4,5,6,7,8,9,10,11; SEEN=1,2,3,5,7,9,10,11; HELD=4,6,8; SAMPLES=2 ;;
  *) echo "unsupported horizon"; exit 2 ;;
esac
[[ "$(sha256sum "$CKPT" | cut -d' ' -f1)" == "$HASH" ]] || exit 2
for YEAR in ${YEARS:-2020 2021 2022}; do
  DATA=/tmp/wb2_0p5_cache
  COHORT=full_year
  EXTRA=(--full-year)
  COUNT=$SAMPLES
  if [[ "$YEAR" == 2022 ]]; then
    DATA="$SHARED/postselection_2022_memmap"
    COHORT=frozen_extension
    COUNT=1
    EXTRA=(--days-of-month 1,8,15,22)
  fi
  if [[ "$MODE" == smoke ]]; then
    COHORT=smoke
    COUNT=1
    EXTRA=(--days-of-month 1 --eval-hours 3)
  elif [[ "$MODE" != full ]]; then
    echo "unsupported mode"; exit 2
  fi
  OUT="$ROOT/metrics/${HORIZON}h/$YEAR/$COHORT"
  [[ ! -e "$OUT/sdyff_ens.json" ]] || { echo "refusing to overwrite $OUT"; exit 2; }
  echo "$(date -Is) start ${HORIZON}h $YEAR $COHORT"
  "$PY" -u -m tools.eval.eval_sdyff_ensemble \
    --memmap-dir "$DATA" --test-year "$YEAR" \
    --climatology /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr \
    --stats-path "$LEGACY/data/json_stats_0p5.nc" \
    --surface-stats-path "$LEGACY/data/surface_stats_0p5.json" \
    --static-path "$LEGACY/data/static_features_0p5.pt" \
    --models "sdyff_ens:$CKPT" --out-dir "$OUT" \
    --ensemble-sizes "${SIZES:-1,4,16,21}" --ensemble-seed 20260914 \
    --batch-size "${BATCH_SIZE:-4}" --num-workers 0 --samples-per-date "$COUNT" \
    --proper-rmse --save-window-metrics --max-tau-hours "$HORIZON" \
    --eval-hours "$HOURS" --seen-tau "$SEEN" --unseen-tau "$HELD" \
    --device cuda:0 --lazy-climatology \
    --climatology-cache-dir /tmp/wti_climatology_cache_corrected_baselines_v1 \
    --keep-n-channels 24 --paper-tag "sdyff_ens_${HORIZON}h_${YEAR}_${COHORT}" \
    "${EXTRA[@]}"
  if [[ "$MODE" == full ]]; then
    "$PY" -m tools.eval.summarize_sdyff_ensemble "$OUT/sdyff_ens.json" \
      --reference "$ROOT/reference/${HORIZON}h/$YEAR/$COHORT/sdyff.json"
  fi
  echo "$(date -Is) complete ${HORIZON}h $YEAR $COHORT"
done
touch "$ROOT/logs/${HORIZON}h_${MODE}.complete"
