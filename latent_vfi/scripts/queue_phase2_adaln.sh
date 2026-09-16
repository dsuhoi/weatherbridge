#!/bin/bash
# Queue runner: wait for Phase 1 AE training to finish, then launch Phase 2 AdaLN-Zero
# with bilinear-residual correction scheme.
set -e
cd /home/jovyan/dsuhoi/weather_time_interpolation

PHASE1_CKPT=/home/jovyan/dsuhoi/weather_time_interpolation/logs/exp_dcae_ae_static_6yr/last.ckpt
PHASE2_EXP=exp_dcae_phase2_adaln_6yr
PHASE2_LOG=/tmp/train_phase2_adaln.log

echo "[$(date)] queue: waiting for Phase 1 AE training to finish..."
while pgrep -f "train_dcae_autoencoder_6yr" > /dev/null 2>&1; do
    sleep 120
done
echo "[$(date)] queue: Phase 1 training process gone."

# Wait extra 60s for ckpt save to flush
sleep 60

if [ ! -f "$PHASE1_CKPT" ]; then
    echo "[$(date)] queue: ERROR - Phase 1 ckpt not found at $PHASE1_CKPT"
    exit 1
fi

# Wait for GPUs to be free
echo "[$(date)] queue: waiting for GPUs to release..."
while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{s=0} {s+=$1} END{print s}') -gt 5000 ]; do
    sleep 30
done
echo "[$(date)] queue: GPUs free, launching Phase 2 AdaLN-Zero"

# Phase 2 AdaLN-Zero — bilinear residual correction scheme:
# x_pred = bilinear(x_0, x_T, τ) + scale · δ
# where δ = decoder(AdaLN-Zero(encoder(x_τ), τ, static))
# Encoder FROZEN, decoder + static_proj + AdaLN-Zero MLP + per-channel scale trainable.
PHASE1_TRAINER=/tmp/train_dcae_autoencoder_6yr.py \
nohup /home/jovyan/.mlspace/envs/ai_scientist/bin/python3 /tmp/train_phase2_adaln_zero.py \
    --phase1_ckpt "$PHASE1_CKPT" \
    --years 2014 2015 2016 2017 2018 2019 \
    --bs 4 --workers 6 \
    --max_epochs 8 --lr 1e-4 \
    --time_dim 128 \
    --gpus 0 1 \
    --exp_name "$PHASE2_EXP" \
    >> "$PHASE2_LOG" 2>&1 &

PID=$!
echo "[$(date)] queue: Phase 2 launched PID=$PID, log=$PHASE2_LOG"
