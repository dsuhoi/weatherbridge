#!/usr/bin/env bash
# Phase 2 CLEAN retrain: all bug fixes + Aurora weights + tisr exclusion.
#
# Includes:
#   - --no-physical-scales-loss (disable legacy double-scaling on normalized data)
#   - --use-aurora-weights (per-level Aurora-style weights, Bodnar 2024 Table 21)
#   - kitchen-sink anchor + scale_floor + residual_loss
#   - --analytic-tisr (and tisr auto-excluded from loss in trainer)
#   - matched-hour eval (corrected per-hour-all)
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights"

# Champion: DC-AE v7 clean retrain on GPU 0+1 DDP-2
DEVICES=2 BATCH=2 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_dcae_v7_clean \
MODEL_TYPE=dcae_residual_linear \
EXTRA_ARGS="--latent-channels 32 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh

# Runner-up: S-DYff anchor clean retrain
DEVICES=2 BATCH=4 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_sdyff_v7_clean \
MODEL_TYPE=sdyff_residual_linear \
EXTRA_ARGS="--latent-channels 96 --sdyff-num-layers 6 --sdyff-n-modes-lat 16 --sdyff-n-modes-lon 32 --sdyff-dropout 0.1 --sdyff-drop-path 0.1 --precision 32 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=4 \
bash scripts/run_exp.sh

echo "=== ALL CLEAN RETRAINS DONE ==="
date
