#!/usr/bin/env bash
# WB2 ERA5 0.5° download (coarsen from 0.25°) + climatology.
# Years: 2014-2020 (6 train + 1 test).
# Variables: 5 PL × 4 levels [1000, 925, 850, 700] + 6 surface — full 26 channels.
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

LOG_DIR=/home/jovyan/dsuhoi/weather_time_interpolation/logs/runner
mkdir -p "$LOG_DIR"

echo "=== WB2 0.5° DATA download (7 years × 26 channels) ===" | tee "$LOG_DIR/wb2_0p5_download.log"
date | tee -a "$LOG_DIR/wb2_0p5_download.log"

python -u tools/data/download_wb2_0p5.py \
  --years 2020 2019 2018 2017 2016 2015 2014 \
  2>&1 | tee -a "$LOG_DIR/wb2_0p5_download.log"

echo "=== WB2 0.5° CLIMATOLOGY (1990-2019, 6-hourly) ===" | tee -a "$LOG_DIR/wb2_0p5_download.log"
date | tee -a "$LOG_DIR/wb2_0p5_download.log"

python -u tools/data/download_wb2_climatology_0p5.py \
  2>&1 | tee -a "$LOG_DIR/wb2_0p5_download.log"

python -u tools/data/canonicalize_wb2_climatology.py \
  --legacy /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_legacy_labels.zarr \
  --output /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr \
  2>&1 | tee -a "$LOG_DIR/wb2_0p5_download.log"

echo "=== ALL DONE ===" | tee -a "$LOG_DIR/wb2_0p5_download.log"
date | tee -a "$LOG_DIR/wb2_0p5_download.log"
df -h /workspace-SR006.nfs2 | tail -2 | tee -a "$LOG_DIR/wb2_0p5_download.log"
