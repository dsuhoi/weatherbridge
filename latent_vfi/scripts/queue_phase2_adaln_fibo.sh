#!/bin/bash
# Queue runner for fibo (inside vfi-train container):
# wait for Phase 1 AE training to finish, then launch Phase 2 AdaLN-Zero on 2× B300 DDP gloo.
set -e
cd /workspace/code/wti

PHASE1_CKPT=/workspace/code/wti/logs/exp_dcae_ae_f16_lat128_3yr_fibo_ddp/last.ckpt
PHASE2_EXP=exp_dcae_phase2_adaln_3yr_fibo
PHASE2_LOG=/tmp/train_phase2_adaln_fibo.log
PHASE1_TRAINER_PATH=/workspace/code/wti/train_dcae_ae_f16_lat128_fibo.py

echo "[$(date)] queue: waiting for Phase 1 AE training to finish..."
while pgrep -f "train_dcae_ae_f16_lat128_fibo" > /dev/null 2>&1; do
    sleep 120
done
echo "[$(date)] queue: Phase 1 training process gone."

sleep 60

if [ ! -f "$PHASE1_CKPT" ]; then
    echo "[$(date)] queue: ERROR - Phase 1 ckpt not found at $PHASE1_CKPT"
    exit 1
fi

echo "[$(date)] queue: waiting for GPUs to release..."
while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{s=0} {s+=$1} END{print s}') -gt 5000 ]; do
    sleep 30
done
echo "[$(date)] queue: GPUs free, launching Phase 2 AdaLN-Zero"

PHASE1_TRAINER="$PHASE1_TRAINER_PATH" DDP_BACKEND=gloo \
nohup python /workspace/code/wti/train_phase2_adaln_zero.py \
    --phase1_ckpt "$PHASE1_CKPT" \
    --years 2017 2018 2019 \
    --bs 4 --workers 6 \
    --max_epochs 8 --lr 1e-4 \
    --time_dim 128 \
    --gpus 0 1 \
    --exp_name "$PHASE2_EXP" \
    --memmap_dir /tmp/wb2_0p5_cache \
    --log_root /workspace/code/wti/logs \
    >> "$PHASE2_LOG" 2>&1 &

PID=$!
echo "[$(date)] queue: Phase 2 launched PID=$PID, log=$PHASE2_LOG"
