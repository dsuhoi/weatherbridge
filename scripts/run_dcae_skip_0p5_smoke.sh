#!/usr/bin/env bash
# DC-AE Skip 0.5° SMOKE TEST: 1 train year (2018) + cache_in_ram + num_workers 0
# Goal: verify pipeline reaches first batch in <10 min
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 \
python -u legacy/train_weather_hermite.py \
  --preset stable \
  --years 2018 2020 \
  --train-years 2018 \
  --test-years 2020 \
  --delta-t-hours 6 \
  --train-hours 1 3 5 \
  --eval-hours 0 1 2 3 4 5 6 \
  --data-dir /workspace-SR006.nfs2/weather_data/time_interpolation_0p5 \
  --cache-in-ram --epochs 1 \
  --multi-level --pressure-levels 1000 925 850 700 \
  --static-path data/static_features_0p5.pt --n-static-features 3 \
  --surface-data-dir /workspace-SR006.nfs2/weather_data/time_interpolation_0p5 \
  --surface-variables t2m u10 v10 mslp sst tcc tcwv \
  --surface-stats-path data/surface_stats_0p5.json \
  --stats-path data/json_stats_0p5.nc \
  --devices 1 --num-workers 0 \
  --batch-size 1 --log-every 1 \
  --limit-val-batches 0 --sanity-val-steps 0 \
  --lat-weighted-loss --lat-crop 8 \
  --lambda-residual 1.0 --residual-scale-floor 0.05 --residual-scale-init 0.30 \
  --lambda-anchor 0.5 --anchor-every-n-batches 4 \
  --model-type dcae_adaln_skip_residual_linear \
  --run-name exp_dcae_0p5_skip_smoke \
  --latent-channels 32 --no-physical-scales-loss --use-aurora-weights
