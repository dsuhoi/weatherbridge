#!/usr/bin/env bash
# Launch 6h paper-cited models on 2021 OOD test (cloud.ru).
#
# DEPRECATED (2026-06-12) — Hydra equivalent:
#   python eval.py +legacy=eval_2020_paper_baseline_24ch \
#       eval.test_year=2021 eval.out_dir=metrics/eval_0p5_2021_ood \
#       eval.out_acc_dir=metrics/acc_0p5_2021_ood
# (The legacy YAML pins the 7-model NAME:CKPT:ENVS list inline; override
# test_year + out_dir at the command line for OOD vs baseline.)
# 2021 memmap already exists at /tmp/wb2_0p5_cache/wb2_2021.{bin,json}.
# Climatology: /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr
#
# Models (paper Table sec_main_table_v2, 24ch only):
#   - DC-AE Skip 14.4M 6yr ep8
#   - ATM-VFI v2 ep8
#   - FuXi 24ch 6yr ep8
#   - S-DYff 24ch 6yr ep8
#   - ModAFNO 24ch 6yr ep8
#
# Output: metrics/eval_0p5_2021_ood/*.json  +  metrics/acc_0p5_2021_ood/*.json
set -euo pipefail
cd /home/jovyan/dsuhoi/weather_time_interpolation

PY=/home/jovyan/.mlspace/envs/ai_scientist/bin/python
OUT_RMSE=metrics/eval_0p5_2021_ood
OUT_ACC=metrics/acc_0p5_2021_ood
mkdir -p "$OUT_RMSE" "$OUT_ACC" logs/runner

LOG=logs/runner/eval_2021_ood_6h_cloudru.log

# Comma-separated NAME:CKPT pairs. Names match metrics/eval_6h_2020_paper_leaderboard/ JSONs (minus _2020 suffix).
# weatherdcae_noskip_24ch_6yr_ep8 and dcae_skip_24ch_6yr_ep8 are intentionally
# omitted here — their 2021 OOD JSONs already exist at metrics/eval_0p5_2021_ood/.
#
# Per-model env vars (NAME:CKPT:ENVS) mirror scripts/run_sh_spectra_6h.sh and
# run_region_season_12h.sh:
#   - ModAFNO: input H/W + native H/W
#   - SDyff:   nlat/nlon + lat_crop
#   - ATM-VFI: handled via custom loader in batch_eval_memmap.py
MODELS="atm_vfi_v2_24ch_6yr_ep8:logs/exp_atm_vfi_24ch_v2_static_3yr_135only_fibo/last.ckpt:,\
fuxi_24ch_6yr_ep8:logs/exp_fuxi_24ch_6yr_v2/last.ckpt:,\
sdyff_24ch_6yr_ep8:logs/exp_sdyff_24ch_6yr/last.ckpt:SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0,\
modafno_24ch_6yr_ep8:logs/exp_modafno_24ch_6yr/last.ckpt:MODAFNO_INP_H=360 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"

# Single-GPU eval. Default to GPU 1 (GPU 0 is typically busy with longer-running
# CRPS / ensemble runs); caller can override via CUDA_VISIBLE_DEVICES env.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PATH=/home/jovyan/.mlspace/envs/ai_scientist/bin:/usr/local/bin:/usr/bin:/bin

echo "[$(date)] launching 6h OOD eval on year 2021..." | tee -a "$LOG"
$PY -u tools/eval/batch_eval_memmap.py \
  --memmap-dir /tmp/wb2_0p5_cache \
  --test-year 2021 \
  --climatology /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr \
  --stats-path data/json_stats_0p5.nc \
  --surface-stats-path data/surface_stats_0p5.json \
  --static-path data/static_features_0p5.pt \
  --models "$MODELS" \
  --out-rmse-dir "$OUT_RMSE" \
  --out-acc-dir "$OUT_ACC" \
  --batch-size 4 --num-workers 2 \
  --samples-per-date 4 --eval-days-per-month 4 \
  --keep-n-channels 24 \
  >> "$LOG" 2>&1

echo "[$(date)] DONE 6h OOD eval. JSONs in $OUT_RMSE and $OUT_ACC" | tee -a "$LOG"
