#!/usr/bin/env bash
# Полный eval на 2020: все 366 дней × все 4 стартовых часа × все интер-часы.
# Используется для финального leaderboard, медленнее economy-режима ~7×.
#
# Usage:
#   bash scripts/run_full_eval.sh RUN_NAME [DELTA_T=6] [TEST_YEAR=2020]
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

RUN_NAME="${1:?usage: $0 RUN_NAME [DELTA_T=6] [TEST_YEAR=2020]}"
DELTA_T="${2:-6}"
TEST_YEAR="${3:-2020}"
DATA_DIR="${DATA_DIR:-/tmp/zarrs}"
EVAL_HOURS_DEFAULT=$(seq 0 "$DELTA_T" | tr '\n' ' ')
EVAL_HOURS="${EVAL_HOURS:-$EVAL_HOURS_DEFAULT}"
SAMPLES_PER_DATE="${SAMPLES_PER_DATE:-4}"
ENSEMBLE_PASSES="${ENSEMBLE_PASSES:-1}"

CKPT="logs/$RUN_NAME/last.ckpt"
[[ -f "$CKPT" ]] || { echo "ERR: $CKPT missing"; exit 1; }

EVAL_EXTRA=""
if [[ "$ENSEMBLE_PASSES" -gt 1 ]]; then
  EVAL_EXTRA="--ensemble-passes $ENSEMBLE_PASSES --mc-dropout"
fi

LOG_DIR=logs/runner
mkdir -p "$LOG_DIR" metrics/eval_full_${TEST_YEAR}
OUT_FILE="metrics/eval_full_${TEST_YEAR}/${RUN_NAME}.json"
LOG_FILE="$LOG_DIR/${RUN_NAME}_eval_full_${TEST_YEAR}.log"

echo "=== FULL EVAL on $TEST_YEAR (${RUN_NAME}, passes=$ENSEMBLE_PASSES, all days × $SAMPLES_PER_DATE start_hours) ===" | tee -a "$LOG_FILE"
date | tee -a "$LOG_FILE"

python -u evaluate_baselines.py \
  --model-checkpoint "$CKPT" \
  --data-dir "$DATA_DIR" --years "$TEST_YEAR" \
  --batch-size 8 --num-workers 4 --samples-per-date "$SAMPLES_PER_DATE" \
  --in-channels 20 --pressure-levels 1000 925 850 700 \
  --static-path data/static_features.pt --stats-path data/json_stats.nc \
  --surface-data-dir "$DATA_DIR" \
  --surface-variables t2m u10 v10 mslp sst tcc tisr \
  --surface-stats-path data/surface_stats.json \
  --analytic-tisr --cache-in-ram \
  --delta-t-hours "$DELTA_T" --skip-bicubic \
  --per-hour-all --eval-hours $EVAL_HOURS \
  --output "$OUT_FILE" \
  --name "${RUN_NAME}_full" \
  $EVAL_EXTRA 2>&1 | tee -a "$LOG_FILE"

echo "=== DONE FULL EVAL $RUN_NAME → $OUT_FILE ===" | tee -a "$LOG_FILE"
date
