#!/usr/bin/env bash
# Unified experiment runner: train + final eval on 2020.
# Replaces run_pair_*.sh, run_train_*.sh, run_dcae_v2.sh, run_channels_6h.sh.
#
# REQUIRED env:
#   RUN_NAME, MODEL_TYPE
# OPTIONAL env:
#   EXTRA_ARGS              extra CLI flags (model-specific)
#   EPOCHS=8, BATCH=2, WORKERS=6, LOG_EVERY=25
#   DEVICES=1               1=single-GPU; 2=DDP-2
#   CUDA_VISIBLE_DEVICES    leave unset to let Lightning pick; set "0" or "1" for parallel
#   TRAIN_YEARS="2017 2018 2019"
#   TEST_YEAR=2020
#   DELTA_T=6               interpolation gap (hours)
#   TRAIN_HOURS="1 3 5"
#   EVAL_HOURS="0 1 2 3 4 5 6"
#   USE_KITCHEN_SINK=1      apply lambda_residual + anchor + scale_floor (default ON)
#   ENSEMBLE_PASSES=1       MC-dropout passes for eval (>1 implies --mc-dropout)
#   DATA_DIR=/tmp/zarrs
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

: "${RUN_NAME:?RUN_NAME env required}"
: "${MODEL_TYPE:?MODEL_TYPE env required}"
EPOCHS=${EPOCHS:-8}
BATCH=${BATCH:-2}
WORKERS=${WORKERS:-6}
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
DATA_DIR=${DATA_DIR:-/tmp/zarrs}
EXTRA_ARGS=${EXTRA_ARGS:-""}

LOG_DIR=logs/runner
mkdir -p "$LOG_DIR" metrics/eval_2020

KITCHEN_FLAGS=""
if [[ "$USE_KITCHEN_SINK" -eq 1 ]]; then
  KITCHEN_FLAGS="--lambda-residual 1.0 --residual-scale-floor 0.05 --residual-scale-init 0.30 --lambda-anchor 0.5 --anchor-every-n-batches 4"
fi

EVAL_EXTRA=""
if [[ "$ENSEMBLE_PASSES" -gt 1 ]]; then
  EVAL_EXTRA="--ensemble-passes $ENSEMBLE_PASSES --mc-dropout"
fi

echo "=== TRAIN $RUN_NAME ($MODEL_TYPE, Δt=${DELTA_T}h, $DEVICES GPU, bs=$BATCH) ===" | tee -a "$LOG_DIR/${RUN_NAME}_train.log"
date | tee -a "$LOG_DIR/${RUN_NAME}_train.log"

python -u legacy/train_weather_hermite.py \
  --preset stable \
  --years $TRAIN_YEARS $TEST_YEAR \
  --train-years $TRAIN_YEARS \
  --test-years $TEST_YEAR \
  --delta-t-hours $DELTA_T \
  --train-hours $TRAIN_HOURS \
  --eval-hours $EVAL_HOURS \
  --data-dir "$DATA_DIR" --cache-in-ram --epochs $EPOCHS \
  --multi-level --pressure-levels 1000 925 850 700 \
  --static-path data/static_features.pt --n-static-features 3 \
  --surface-data-dir "$DATA_DIR" \
  --surface-variables t2m u10 v10 mslp sst tcc tisr \
  --surface-stats-path data/surface_stats.json \
  --analytic-tisr \
  --stats-path data/json_stats.nc \
  --resume-from-last --devices $DEVICES --num-workers $WORKERS \
  --batch-size $BATCH --log-every $LOG_EVERY \
  --limit-val-batches 0 --sanity-val-steps 0 \
  --lat-weighted-loss \
  $KITCHEN_FLAGS \
  --model-type "$MODEL_TYPE" \
  --run-name "$RUN_NAME" \
  $EXTRA_ARGS 2>&1 | tee -a "$LOG_DIR/${RUN_NAME}_train.log"

CKPT="logs/$RUN_NAME/last.ckpt"
[[ -f "$CKPT" ]] || { echo "WARN: $CKPT missing — skip eval"; exit 1; }

echo
echo "=== FINAL EVAL on $TEST_YEAR (${RUN_NAME}, passes=$ENSEMBLE_PASSES) ===" | tee -a "$LOG_DIR/${RUN_NAME}_eval${TEST_YEAR}.log"
date | tee -a "$LOG_DIR/${RUN_NAME}_eval${TEST_YEAR}.log"

python -u evaluate_baselines.py \
  --model-checkpoint "$CKPT" \
  --data-dir "$DATA_DIR" --years $TEST_YEAR \
  --batch-size 8 --num-workers 4 --samples-per-date 4 \
  --eval-days-per-month 4 \
  --in-channels 20 --pressure-levels 1000 925 850 700 \
  --static-path data/static_features.pt --stats-path data/json_stats.nc \
  --surface-data-dir "$DATA_DIR" \
  --surface-variables t2m u10 v10 mslp sst tcc tisr \
  --surface-stats-path data/surface_stats.json \
  --analytic-tisr --cache-in-ram \
  --delta-t-hours $DELTA_T --skip-bicubic \
  --per-hour-all --eval-hours $EVAL_HOURS \
  --output "metrics/eval_${TEST_YEAR}/${RUN_NAME}.json" \
  --name "$RUN_NAME" \
  $EVAL_EXTRA 2>&1 | tee -a "$LOG_DIR/${RUN_NAME}_eval${TEST_YEAR}.log"

echo "=== DONE $RUN_NAME ===" | tee -a "$LOG_DIR/${RUN_NAME}_eval${TEST_YEAR}.log"
date
