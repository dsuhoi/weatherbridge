#!/usr/bin/env bash
# Two parallel DC-AE ablations on 6yr 0.5° (S2.3a + S2.3b):
#   GPU0: no-skip   (model_type=dcae_adaln_residual_linear)   → S2.3a
#   GPU1: no-scaffold (dcae_adaln_skip + direct_prediction=true) → S2.3b
# Same data + hyperparams as headline DC-AE Skip 6yr (8 epochs).
#
# DEPRECATED (2026-06-12) — Hydra equivalents:
#   CUDA_VISIBLE_DEVICES=0 python train.py +legacy=train_weatherdcae_noskip_24ch_6yr
#   CUDA_VISIBLE_DEVICES=1 python train.py model=dcae_skip ++model.direct_prediction=true \
#       exp_name=exp_dcae_noscaffold_0p5_6yr data.years='[2014,2015,2016,2017,2018,2019]'
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

mkdir -p logs/runner

# Sanity: memmap cache exists
for y in 2014 2015 2016 2017 2018 2019; do
  [ -f "/tmp/wb2_0p5_cache/wb2_${y}.bin" ] || { echo "MISSING /tmp/wb2_0p5_cache/wb2_${y}.bin"; exit 1; }
done

export YEARS="2014 2015 2016 2017 2018 2019"
export MAX_EPOCHS=8
export BATCH_SIZE=1
export LR=1e-4
export LATENT_CHANNELS=256
export NUM_WORKERS=4

# === GPU0: no-skip (DC-AE AdaLN — scaffold still on, skip OFF) ===
NOSKIP_OUT="logs/exp_dcae_noskip_0p5_6yr"
mkdir -p "$NOSKIP_OUT"
echo "=== LAUNCH DC-AE no-skip 6yr on GPU0 ==="; date
CUDA_VISIBLE_DEVICES=0 \
  MODEL_TYPE=dcae_adaln_residual_linear \
  LAT_CROP=-8 \
  OUT_DIR="$NOSKIP_OUT" \
  nohup python -u scripts/train_0p5_memmap.py \
  > logs/runner/train_dcae_noskip_0p5_6yr.log 2>&1 &
NOSKIP_PID=$!
echo "no-skip PID=$NOSKIP_PID"

# === GPU1: no-scaffold (DC-AE AdaLN Skip with direct prediction) ===
NOSC_OUT="logs/exp_dcae_noscaffold_0p5_6yr"
mkdir -p "$NOSC_OUT"
echo "=== LAUNCH DC-AE no-scaffold 6yr on GPU1 ==="; date
CUDA_VISIBLE_DEVICES=1 \
  MODEL_TYPE=dcae_adaln_skip_residual_linear \
  LAT_CROP=-8 \
  EXTRA="direct_prediction=true" \
  OUT_DIR="$NOSC_OUT" \
  nohup python -u scripts/train_0p5_memmap.py \
  > logs/runner/train_dcae_noscaffold_0p5_6yr.log 2>&1 &
NOSC_PID=$!
echo "no-scaffold PID=$NOSC_PID"

echo "=== BOTH LAUNCHED at $(date) ==="
echo "no-skip:      PID=$NOSKIP_PID  log: logs/runner/train_dcae_noskip_0p5_6yr.log"
echo "no-scaffold:  PID=$NOSC_PID    log: logs/runner/train_dcae_noscaffold_0p5_6yr.log"
