#!/usr/bin/env bash
# ModAFNO v9 PAPER-FAITHFUL: direct prediction (no bilinear+residual wrapping),
# как в Leinonen 2024 (arxiv 2410.18904).
# Без --lambda-residual / --lambda-anchor / --residual-scale-floor.
set -eo pipefail
cd "$(dirname "$0")/.."

# CLEAN_EXTRA НЕ включает kitchen-sink (lambda_residual, anchor, scale_floor) —
# kitchen-sink настроен на residual mode, для direct prediction он мешает.
CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights --direct-prediction"

CUDA_VISIBLE_DEVICES=0 \
DEVICES=1 BATCH=2 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_modafno_v9_direct \
MODEL_TYPE=modafno_official_residual_linear \
USE_KITCHEN_SINK=0 \
EXTRA_ARGS="--latent-channels 512 --modafno-depth 12 --modafno-num-blocks 8 --modafno-drop-rate 0.0 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_modafno_v9_direct.log 2>&1 &
P0=$!
echo "ModAFNO v9 DIRECT launched: PID=$P0 on GPU 0"

wait $P0
echo "=== ModAFNO v9 DIRECT DONE ==="
date
