#!/usr/bin/env bash
# Orchestrator: stage 12h ckpts from fibo, wait for 12h CorrDiff/FM
# training to complete, then run 6h+12h 2021 OOD eval on cloud.ru.
#
# Usage (on cloud.ru):
#   bash scripts/run_eval_2021_ood_full_cloudru.sh [TRAIN_PID]
#
# If TRAIN_PID is given (default: 443633 — the corrdiff_fm_12h DDP launch),
# we poll until it exits, otherwise we skip the wait and start immediately.
set -euo pipefail
cd /home/jovyan/dsuhoi/weather_time_interpolation

TRAIN_PID=${1:-443633}
CKPT_DIR=${CKPT_DIR:-/tmp/ckpts_12h}
mkdir -p "$CKPT_DIR" logs/runner

LOG=logs/runner/eval_2021_ood_orchestrator.log
echo "[$(date)] orchestrator START — TRAIN_PID=$TRAIN_PID, CKPT_DIR=$CKPT_DIR" | tee -a "$LOG"

# 12h ckpts that live on fibo. Stage them at $CKPT_DIR before eval.
# Format: dest_basename:fibo_source_relative_to_logs.
declare -a FIBO_CKPTS=(
  "exp_12h_oddskip_dcae_noskip_3yr_fibo_epoch9.ckpt|exp_12h_oddskip_dcae_noskip_3yr_fibo/epoch=9-step=10940.ckpt"
  "exp_12h_oddskip_dcae_2017_18_19_epoch9.ckpt|exp_12h_oddskip_dcae_2017_18_19/epoch=9-step=10940.ckpt"
  "exp_atmvfi_12h_oddskip_epoch9.ckpt|exp_atmvfi_12h_oddskip/epoch=9-step=43740.ckpt"
  "exp_12h_oddskip_fuxi_2017_18_19_epoch9.ckpt|exp_12h_oddskip_fuxi_2017_18_19/epoch=9-step=5470.ckpt"
  "exp_12h_oddskip_modafno_full_2017_18_19_epoch9.ckpt|exp_12h_oddskip_modafno_full_2017_18_19/epoch=9-step=21870.ckpt"
  "exp_12h_oddskip_sdyff_dyff_0p5_2017_18_19_epoch9.ckpt|exp_12h_oddskip_sdyff_dyff_0p5_2017_18_19/epoch=9-step=5470.ckpt"
)

# NOTE: the staging step assumes someone has scp'd these from fibo to
# /tmp/ckpts_12h/ already, since cloud.ru has no direct fibo ssh
# access. Verify presence:
echo "[$(date)] verifying 12h ckpt staging in $CKPT_DIR ..." | tee -a "$LOG"
MISSING=0
for entry in "${FIBO_CKPTS[@]}"; do
  dest="${entry%%|*}"
  if [ ! -f "$CKPT_DIR/$dest" ]; then
    echo "  MISSING: $CKPT_DIR/$dest" | tee -a "$LOG"
    MISSING=$((MISSING + 1))
  else
    echo "  OK: $CKPT_DIR/$dest ($(du -h "$CKPT_DIR/$dest" | cut -f1))" | tee -a "$LOG"
  fi
done
if [ "$MISSING" -gt 0 ]; then
  echo "[$(date)] $MISSING ckpt(s) missing — 12h eval will skip those. Continuing." | tee -a "$LOG"
fi

# Wait for the training PID to exit
if kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "[$(date)] waiting for training PID $TRAIN_PID ..." | tee -a "$LOG"
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 300  # check every 5 min
  done
  echo "[$(date)] training PID $TRAIN_PID exited" | tee -a "$LOG"
else
  echo "[$(date)] training PID $TRAIN_PID not running — proceeding" | tee -a "$LOG"
fi

# Run 6h eval (single GPU)
echo "[$(date)] launching 6h OOD eval ..." | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 bash scripts/run_eval_2021_ood_6h_cloudru.sh \
  >> "$LOG" 2>&1 || echo "[$(date)] 6h OOD eval FAILED (continuing to 12h)" | tee -a "$LOG"

# Run 12h eval (single GPU)
echo "[$(date)] launching 12h OOD eval ..." | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 CKPT_DIR="$CKPT_DIR" bash scripts/run_eval_2021_ood_12h_cloudru.sh \
  >> "$LOG" 2>&1 || echo "[$(date)] 12h OOD eval FAILED" | tee -a "$LOG"

echo "[$(date)] orchestrator DONE" | tee -a "$LOG"
