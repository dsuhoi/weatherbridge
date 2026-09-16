#!/usr/bin/env bash
# Phase B: refresh S3 dataset to new schema.
#  - PL: levels [1000, 925, 850, 700] (standard LADCast grid; replace legacy [1000,950,900,850])
#  - Surface: t2m, u10, v10, mslp, sst, tcc, tcwv  (no TISR — computed analytically)
#  - One year (2018) keeps TISR for analytic-vs-stored sanity validation.
#  - Output dir: /mnt/s3/bucket8tb/weather_data/time_interpolation_v2/
#
# WARNING: 200+ GB writes to s3fs FUSE; ETA 8-15h depending on Google rate-limit.
# Run AFTER current training queue finishes (or in parallel if you don't mind I/O contention).
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

LOG_DIR=logs/runner
mkdir -p "$LOG_DIR"
PL_LOG="$LOG_DIR/refresh_pl_v2.log"
SURF_LOG="$LOG_DIR/refresh_surface_v2.log"
SURF_TISR_LOG="$LOG_DIR/refresh_surface_v2_tisr_2018.log"

YEARS_ALL="2012 2013 2014 2015 2016 2017 2018 2019 2020"
TISR_KEEP_YEAR=2018

NEW_PL_DIR=/mnt/s3/bucket8tb/weather_data/time_interpolation_v2
mkdir -p "$NEW_PL_DIR"

echo "========================================"
echo "Phase B: refresh S3 dataset → $NEW_PL_DIR"
echo "Years: $YEARS_ALL"
echo "TISR retained for: $TISR_KEEP_YEAR (audit)"
echo "Started: $(date)"
echo "========================================"

# --- 1. PL re-download with [1000, 925, 850, 700] ---
echo
echo "[1/3] Downloading PL zarrs (5 vars × 4 levels) ..."
python -u tools/data/download_weatherbench2_v2.py \
  --config legacy/configs/config_pl_standard_grid.json 2>&1 | tee "$PL_LOG"

# --- 2. Surface (no TISR) for all years ---
echo
echo "[2/3] Downloading surface zarrs (no TISR) for all years ..."
python -u tools/data/download_surface_vars.py \
  --target "$NEW_PL_DIR" \
  --years $YEARS_ALL 2>&1 | tee "$SURF_LOG"

# --- 3. Surface WITH TISR for the audit year (overwrites that one) ---
echo
echo "[3/3] Downloading surface with TISR for $TISR_KEEP_YEAR ..."
python -u tools/data/download_surface_vars.py \
  --target "$NEW_PL_DIR" \
  --years $TISR_KEEP_YEAR \
  --include-tisr --overwrite 2>&1 | tee "$SURF_TISR_LOG"

echo
echo "========================================"
echo "Phase B finished: $(date)"
echo "Verifying:"
echo "  PL years:"
ls -d "$NEW_PL_DIR"/zarr_*.zarr 2>/dev/null | sort
echo "  Surface years:"
ls -d "$NEW_PL_DIR"/surface_*.zarr 2>/dev/null | sort
echo "  Total size:"
du -sh "$NEW_PL_DIR"
echo "========================================"
