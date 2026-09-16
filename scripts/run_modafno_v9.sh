#!/usr/bin/env bash
# ModAFNO с v9-конфигом: Aurora weights + no-physical-scales + fixed economy eval.
# Архитектура: paper-faithful ModEmbedNet (sinusoidal+MLP → FiLM AdaLN в каждом
# AFNO-блоке) — никаких τ-map хаков.
set -eo pipefail
cd "$(dirname "$0")/.."

CLEAN_EXTRA="--no-physical-scales-loss --use-aurora-weights"

# Параллельный single-GPU: одиночный ModAFNO на GPU 0 для воспроизводимости с v7.
CUDA_VISIBLE_DEVICES=0 \
DEVICES=1 BATCH=2 WORKERS=8 EPOCHS=8 \
RUN_NAME=exp_modafno_v9 \
MODEL_TYPE=modafno_official_residual_linear \
EXTRA_ARGS="--latent-channels 512 --modafno-depth 12 --modafno-num-blocks 8 --modafno-drop-rate 0.0 $CLEAN_EXTRA" \
ENSEMBLE_PASSES=1 \
bash scripts/run_exp.sh > logs/runner/launcher_modafno_v9.log 2>&1 &
P0=$!
echo "ModAFNO v9 launched: PID=$P0 on GPU 0"

wait $P0
echo "=== ModAFNO v9 DONE ==="
date
