#!/usr/bin/env bash
# FuXi SwinV2 0.5° SMOKE — single year (2018) cached, no test eval.
# Goal: verify training pipeline works end-to-end on lat=360.
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python -u legacy/train_weather_hermite.py \
  --preset stable \
  --years 2018 2014 \
  --train-years 2018 \
  --test-years 2014 \
  --delta-t-hours 6 \
  --train-hours 1 3 5 \
  --eval-hours 0 1 2 3 4 5 6 \
  --data-dir /workspace-SR006.nfs2/weather_data/time_interpolation_0p5 \
  --epochs 2 \
  --multi-level --pressure-levels 1000 925 850 700 \
  --static-path data/static_features_0p5.pt --n-static-features 3 \
  --surface-data-dir /workspace-SR006.nfs2/weather_data/time_interpolation_0p5 \
  --surface-variables t2m u10 v10 mslp sst tcc tcwv \
  --surface-stats-path data/surface_stats_0p5.json \
  --stats-path data/json_stats_0p5.nc \
  --devices 1 --num-workers 4 \
  --batch-size 1 --log-every 10 \
  --limit-val-batches 0 --sanity-val-steps 0 \
  --lat-weighted-loss \
  --lambda-residual 1.0 --residual-scale-floor 0.05 --residual-scale-init 0.30 \
  --lambda-anchor 0.5 --anchor-every-n-batches 4 \
  --model-type fuxi_swinv2_residual_linear \
  --run-name exp_fuxi_0p5_swinv2_smoke \
  --latent-channels 256 --fuxi-depth 8 --fuxi-num-heads 8 \
  --fuxi-window-size-h 5 --fuxi-window-size-w 9 --fuxi-patch-size 4 --fuxi-drop-path 0.1 \
  --no-physical-scales-loss --use-aurora-weights
