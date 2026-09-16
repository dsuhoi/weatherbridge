#!/usr/bin/env bash
# CDS native 0.5° download → per-year Zarr pipeline.
# 1. Smoke test: 1 month (Jan 2020) to verify multi-thread CDS works
# 2. Full download: 7 years × 12 months parallel
# 3. Convert NCs → per-year Zarr with integrity check
# 4. Cleanup intermediate NCs after Zarr write verified
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

LOG=/home/jovyan/dsuhoi/weather_time_interpolation/logs/runner/cds_0p5_pipeline.log
echo "=== CDS 0.5° PIPELINE START $(date) ===" | tee "$LOG"

# --- Stage 1: SMOKE TEST (1 month) ---
echo | tee -a "$LOG"
echo "=== Stage 1: SMOKE Jan 2020 (4 parallel workers) ===" | tee -a "$LOG"
python -u tools/data/download_cds_0p5_parallel.py \
  --years 2020 --months 1 --workers 2 2>&1 | tee -a "$LOG"

# --- Stage 2: FULL DOWNLOAD ---
echo | tee -a "$LOG"
echo "=== Stage 2: FULL 2014-2020 × 12 months (5 parallel workers) ===" | tee -a "$LOG"
python -u tools/data/download_cds_0p5_parallel.py \
  --years 2014 2015 2016 2017 2018 2019 2020 \
  --months 1 2 3 4 5 6 7 8 9 10 11 12 \
  --workers 5 2>&1 | tee -a "$LOG"

# --- Stage 3: NC → Zarr + Integrity ---
echo | tee -a "$LOG"
echo "=== Stage 3: NC → Zarr per-year + integrity ===" | tee -a "$LOG"
python -u tools/data/cds_to_zarr_0p5.py \
  --years 2014 2015 2016 2017 2018 2019 2020 2>&1 | tee -a "$LOG"

# --- Stage 4: Final disk report ---
echo | tee -a "$LOG"
echo "=== Stage 4: Final disk usage ===" | tee -a "$LOG"
df -h /workspace-SR006.nfs2 | tail -2 | tee -a "$LOG"
du -sh /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/* 2>/dev/null | tee -a "$LOG"
echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
