#!/usr/bin/env bash
# ACC evaluation at 0.5° resolution.
# Wraps tools/eval/compute_acc.py with proper 0.5° artifacts.
#
# Usage:
#   bash scripts/run_acc_eval_0p5.sh RUN_NAME       # full model + bilinear
#   bash scripts/run_acc_eval_0p5.sh ""             # bilinear baseline only
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

RUN_NAME="${1:-}"
TEST_YEAR="${2:-2020}"
DELTA_T="${3:-6}"
DATA_DIR="${DATA_DIR:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5}"
CLIM_PATH="${CLIM_PATH:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}"
STATS_PATH="${STATS_PATH:-data/json_stats_0p5.nc}"
SURFACE_STATS_PATH="${SURFACE_STATS_PATH:-data/surface_stats_0p5.json}"
STATIC_PATH="${STATIC_PATH:-data/static_features_0p5.pt}"
SAMPLES_PER_DATE="${SAMPLES_PER_DATE:-4}"
EVAL_DAYS_PER_MONTH="${EVAL_DAYS_PER_MONTH:-4}"

LOG_DIR=logs/runner
mkdir -p "$LOG_DIR" "metrics/acc_0p5_${TEST_YEAR}"

CKPT_FLAGS=""
NAME_TAG="bilinear"
if [[ -n "$RUN_NAME" ]]; then
  CKPT="logs/$RUN_NAME/last.ckpt"
  [[ -f "$CKPT" ]] || { echo "ERR: $CKPT missing"; exit 1; }
  CKPT_FLAGS="--model-checkpoint $CKPT"
  NAME_TAG="$RUN_NAME"
fi

OUT_FILE="metrics/acc_0p5_${TEST_YEAR}/${NAME_TAG}.json"
LOG_FILE="$LOG_DIR/acc_0p5_${NAME_TAG}.log"

echo "=== ACC EVAL 0.5° on $TEST_YEAR (${NAME_TAG}) ===" | tee -a "$LOG_FILE"
date | tee -a "$LOG_FILE"

python -u tools/eval/compute_acc.py \
  $CKPT_FLAGS \
  --output "$OUT_FILE" \
  --data-dir "$DATA_DIR" --surface-data-dir "$DATA_DIR" \
  --stats-path "$STATS_PATH" --surface-stats-path "$SURFACE_STATS_PATH" \
  --static-path "$STATIC_PATH" \
  --climatology "$CLIM_PATH" \
  --years "$TEST_YEAR" \
  --pressure-levels 1000 925 850 700 \
  --surface-variables t2m u10 v10 mslp sst tcc tisr \
  --samples-per-date "$SAMPLES_PER_DATE" \
  --eval-days-per-month "$EVAL_DAYS_PER_MONTH" \
  --delta-t-hours "$DELTA_T" \
  --analytic-tisr --cache-in-ram \
  --batch-size 4 --num-workers 4 \
  2>&1 | tee -a "$LOG_FILE"

echo "=== DONE ACC 0.5° $NAME_TAG → $OUT_FILE ===" | tee -a "$LOG_FILE"
date
