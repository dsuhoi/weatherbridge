#!/usr/bin/env bash
# 0.5° experiment runner: train + eval on 0.5° NFS data.
# Required env: RUN_NAME, MODEL_TYPE
# Optional: EXTRA_ARGS, EPOCHS=8, BATCH=1, WORKERS=4, DEVICES=1
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

: "${RUN_NAME:?RUN_NAME env required}"
: "${MODEL_TYPE:?MODEL_TYPE env required}"
EPOCHS=${EPOCHS:-8}
BATCH=${BATCH:-1}
WORKERS=${WORKERS:-4}
LOG_EVERY=${LOG_EVERY:-25}
DEVICES=${DEVICES:-1}
TRAIN_YEARS=${TRAIN_YEARS:-"2017 2018 2019"}
TEST_YEAR=${TEST_YEAR:-2020}
DELTA_T=${DELTA_T:-6}
TRAIN_HOURS=${TRAIN_HOURS:-"1 3 5"}
EVAL_HOURS_DEFAULT=$(seq 0 $DELTA_T | tr '\n' ' ')
EVAL_HOURS=${EVAL_HOURS:-$EVAL_HOURS_DEFAULT}
USE_KITCHEN_SINK=${USE_KITCHEN_SINK:-1}
ENSEMBLE_PASSES=${ENSEMBLE_PASSES:-1}
DATA_DIR=${DATA_DIR:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5}
STATS_PATH=${STATS_PATH:-data/json_stats_0p5.nc}
SURFACE_STATS_PATH=${SURFACE_STATS_PATH:-data/surface_stats_0p5.json}
STATIC_PATH=${STATIC_PATH:-data/static_features_0p5.pt}
EXTRA_ARGS=${EXTRA_ARGS:-""}
CACHE_IN_RAM=${CACHE_IN_RAM:-0}
CACHE_FLAG=""
[[ "$CACHE_IN_RAM" -eq 1 ]] && CACHE_FLAG="--cache-in-ram"

LOG_DIR=logs/runner
mkdir -p "$LOG_DIR" metrics/eval_0p5_${TEST_YEAR}

KITCHEN_FLAGS=""
if [[ "$USE_KITCHEN_SINK" -eq 1 ]]; then
  KITCHEN_FLAGS="--lambda-residual 1.0 --residual-scale-floor 0.05 --residual-scale-init 0.30 --lambda-anchor 0.5 --anchor-every-n-batches 4"
fi

EVAL_EXTRA=""
if [[ "$ENSEMBLE_PASSES" -gt 1 ]]; then
  EVAL_EXTRA="--ensemble-passes $ENSEMBLE_PASSES --mc-dropout"
fi

echo "=== TRAIN 0.5° $RUN_NAME ($MODEL_TYPE, Δt=${DELTA_T}h, $DEVICES GPU, bs=$BATCH) ===" | tee -a "$LOG_DIR/${RUN_NAME}_train.log"
date | tee -a "$LOG_DIR/${RUN_NAME}_train.log"

python -u legacy/train_weather_hermite.py \
  --preset stable \
  --years $TRAIN_YEARS $TEST_YEAR \
  --train-years $TRAIN_YEARS \
  --test-years $TEST_YEAR \
  --delta-t-hours $DELTA_T \
  --train-hours $TRAIN_HOURS \
  --eval-hours $EVAL_HOURS \
  --data-dir "$DATA_DIR" $CACHE_FLAG --epochs $EPOCHS \
  --multi-level --pressure-levels 1000 925 850 700 \
  --static-path "$STATIC_PATH" --n-static-features 3 \
  --surface-data-dir "$DATA_DIR" \
  --surface-variables t2m u10 v10 mslp sst tcc tcwv \
  --surface-stats-path "$SURFACE_STATS_PATH" \
  --stats-path "$STATS_PATH" \
  --resume-from-last --devices $DEVICES --num-workers $WORKERS \
  --batch-size $BATCH --log-every $LOG_EVERY \
  --limit-val-batches 0 --sanity-val-steps 0 \
  --lat-weighted-loss --lat-crop 8 \
  $KITCHEN_FLAGS \
  --model-type "$MODEL_TYPE" \
  --run-name "$RUN_NAME" \
  $EXTRA_ARGS 2>&1 | tee -a "$LOG_DIR/${RUN_NAME}_train.log"

CKPT="logs/$RUN_NAME/last.ckpt"
[[ -f "$CKPT" ]] || { echo "WARN: $CKPT missing — skip eval"; exit 1; }

echo
echo "=== FINAL EVAL 0.5° on $TEST_YEAR (${RUN_NAME}) ===" | tee -a "$LOG_DIR/${RUN_NAME}_eval0p5_${TEST_YEAR}.log"
date | tee -a "$LOG_DIR/${RUN_NAME}_eval0p5_${TEST_YEAR}.log"

python -u evaluate_baselines.py \
  --model-checkpoint "$CKPT" \
  --data-dir "$DATA_DIR" --years $TEST_YEAR \
  --batch-size 4 --num-workers 4 --samples-per-date 4 \
  --eval-days-per-month 4 \
  --in-channels 20 --pressure-levels 1000 925 850 700 \
  --static-path "$STATIC_PATH" --stats-path "$STATS_PATH" \
  --surface-data-dir "$DATA_DIR" \
  --surface-variables t2m u10 v10 mslp sst tcc tcwv \
  --surface-stats-path "$SURFACE_STATS_PATH" \
  $CACHE_FLAG \
  --delta-t-hours $DELTA_T --skip-bicubic \
  --per-hour-all --eval-hours $EVAL_HOURS \
  --output "metrics/eval_0p5_${TEST_YEAR}/${RUN_NAME}.json" \
  --name "$RUN_NAME" \
  $EVAL_EXTRA 2>&1 | tee -a "$LOG_DIR/${RUN_NAME}_eval0p5_${TEST_YEAR}.log"

echo "=== DONE 0.5° $RUN_NAME ===" | tee -a "$LOG_DIR/${RUN_NAME}_eval0p5_${TEST_YEAR}.log"
date
