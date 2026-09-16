#!/usr/bin/env bash
# Launch 12h paper-cited models on 2021 OOD test (cloud.ru).
# 2021 memmap already exists at /tmp/wb2_0p5_cache/wb2_2021.{bin,json}.
#
# Models (paper tab_main_12h, 24ch only):
#   - DC-AE NoSkip 3yr 12h fibo (ep9)
#   - DC-AE Skip 3yr 12h fibo (ep9)
#   - ATM-VFI 12h fibo (ep9)
#   - FuXi 12h fibo (ep9)
#   - ModAFNO 12h fibo (ep9)
#   - S-DYff 12h fibo (ep9)
#   - WeatherDCAE NoSkip 6yr 12h cloudru (last; corresponds to ep10 metrics JSON)
#
# Output: metrics/eval_12h_2021_ood/*.json
set -euo pipefail
cd /home/jovyan/dsuhoi/weather_time_interpolation

PY=/home/jovyan/.mlspace/envs/ai_scientist/bin/python
OUT_DIR=metrics/eval_12h_2021_ood
mkdir -p "$OUT_DIR" logs/runner
LOG=logs/runner/eval_2021_ood_12h_cloudru.log

# Format for batch_eval_12h_memmap.py: comma-separated NAME:CKPT:HORIZON:EXTRA-ENVS.
# Names are stable across 2020/2021 evals (compare-friendly).
#
# For 12h fibo-cloudru cross-cluster: 6 of the 7 paper ckpts live on fibo.
# We expect them to be staged at /tmp/ckpts_12h/ on cloud.ru beforehand
# (rsync from fibo). The wrapper run_eval_2021_ood_full_cloudru.sh handles
# staging.

CKPT_DIR=${CKPT_DIR:-/tmp/ckpts_12h}
# DC-AE_NoSkip_3yr_12h and DC-AE_Skip_3yr_12h are intentionally omitted here —
# their 2021 OOD JSONs already exist at metrics/eval_12h_2021_ood/.
#
# Per-model env vars (NAME:CKPT:ENVS) mirror scripts/run_region_season_12h.sh:
#   - ModAFNO: input H/W + native H/W
#   - SDyff:   nlat/nlon + lat_crop
#   - ATM-VFI: handled via custom loader in batch_eval_12h_memmap.py
MODELS="ATM-VFI_3yr_12h:${CKPT_DIR}/exp_atmvfi_12h_oddskip_epoch9.ckpt:,\
FuXi_3yr_12h:${CKPT_DIR}/exp_12h_oddskip_fuxi_2017_18_19_epoch9.ckpt:,\
ModAFNO_3yr_12h:${CKPT_DIR}/exp_12h_oddskip_modafno_full_2017_18_19_epoch9.ckpt:MODAFNO_INP_H=362 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720,\
SDyff_3yr_12h:${CKPT_DIR}/exp_12h_oddskip_sdyff_dyff_0p5_2017_18_19_epoch9.ckpt:SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0,\
WeatherDCAE_NoSkip_6yr_12h:logs/exp_12h_oddskip_dcae_noskip_cloudru_2014_19/last.ckpt:"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PATH=/home/jovyan/.mlspace/envs/ai_scientist/bin:/usr/local/bin:/usr/bin:/bin

echo "[$(date)] launching 12h OOD eval on year 2021..." | tee -a "$LOG"
$PY -u tools/eval/batch_eval_12h_memmap.py \
  --memmap-dir /tmp/wb2_0p5_cache \
  --test-year 2021 \
  --climatology /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr \
  --stats-path data/json_stats_0p5.nc \
  --surface-stats-path data/surface_stats_0p5.json \
  --static-path data/static_features_0p5.pt \
  --models "$MODELS" \
  --out-dir "$OUT_DIR" \
  --paper-tag 12h_2021_ood \
  --max-tau-hours 12 \
  --eval-hours "1,2,3,4,5,6,7,8,9,10,11" \
  --seen-tau "1,2,3,5,7,9,10,11" \
  --unseen-tau "4,6,8" \
  --batch-size 2 --num-workers 2 \
  --samples-per-date 2 --eval-days-per-month 4 \
  --keep-n-channels 24 \
  >> "$LOG" 2>&1

echo "[$(date)] DONE 12h OOD eval. JSONs in $OUT_DIR" | tee -a "$LOG"
