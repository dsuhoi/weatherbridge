#!/bin/bash
# After fibo Phase 1 reaches epoch 10, resume training with max_epochs=20.
# Phase 2 queued AFTER extended training completes.
set -e
cd /workspace/code/wti

PHASE1_CKPT=/workspace/code/wti/logs/exp_dcae_ae_f16_lat128_3yr_fibo_ddp/last.ckpt

echo "[$(date)] queue: waiting for Phase 1 (epoch 10) to finish..."
while pgrep -f "train_dcae_ae_f16_lat128_fibo" > /dev/null 2>&1; do
    sleep 120
done
echo "[$(date)] queue: Phase 1 epoch 10 done. Resuming with max_epochs=20..."

sleep 60

# Resume Phase 1 with extended epochs
DDP_BACKEND=gloo nohup python /workspace/code/wti/train_dcae_ae_f16_lat128_fibo.py \
    --bs 20 --workers 8 --max_epochs 20 --lr 1e-4 \
    --latent_channels 128 \
    --exp_name exp_dcae_ae_f16_lat128_3yr_fibo_ddp \
    --gpus 0 1 \
    > /tmp/train_dcae_f16_lat128_extended.log 2>&1 &

EXT_PID=$!
echo "[$(date)] queue: Phase 1 extended training PID=$EXT_PID"
echo "[$(date)] queue: NOW waiting for extended Phase 1 to finish..."

while kill -0 $EXT_PID 2>/dev/null; do
    sleep 120
done
echo "[$(date)] queue: extended Phase 1 done."

sleep 60
echo "[$(date)] queue: waiting for GPUs free..."
while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{s=0} {s+=$1} END{print s}') -gt 5000 ]; do
    sleep 30
done

echo "[$(date)] queue: launching Phase 2 AdaLN-Zero"
PHASE1_TRAINER=/workspace/code/wti/train_dcae_ae_f16_lat128_fibo.py DDP_BACKEND=gloo \
nohup python /workspace/code/wti/train_phase2_adaln_zero.py \
    --phase1_ckpt "$PHASE1_CKPT" \
    --years 2017 2018 2019 \
    --bs 4 --workers 6 \
    --max_epochs 8 --lr 1e-4 \
    --time_dim 128 \
    --gpus 0 1 \
    --exp_name exp_dcae_phase2_adaln_3yr_fibo \
    --memmap_dir /tmp/wb2_0p5_cache \
    --log_root /workspace/code/wti/logs \
    >> /tmp/train_phase2_adaln_fibo.log 2>&1 &

echo "[$(date)] queue: Phase 2 launched"
