#!/usr/bin/env bash
# Batch ACC eval for all 1°-trained checkpoints on clean 2020.
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

EXPERIMENTS=(
  exp_dcae_v9_skip
  exp_dcae_v9_adaln
  exp_dcae_v9_skip_mini
  exp_dcae_v9_bilinear_xattn
  exp_dcae_v9_bilinear_xattn_proper
  exp_dcae_v9_swin_skip
  exp_dcae_v9_swin_skip_proper
  exp_fuxi_swinv2_v9
  exp_fuxi_swinv2_v9_direct
  exp_fuxi_nano_v9
  exp_modafno_v9
  exp_modafno_v9_slim
  exp_modafno_v9_direct
  exp_sdyff_v9_dyffusion
)
OUT_DIR="metrics/acc_1deg_clean_2020"
mkdir -p "$OUT_DIR" logs/runner

for EXP in "${EXPERIMENTS[@]}"; do
  CKPT="logs/$EXP/last.ckpt"
  OUT_FILE="$OUT_DIR/${EXP}.json"
  LOG_FILE="logs/runner/acc_clean_${EXP}.log"
  if [[ -f "$OUT_FILE" ]]; then
    echo "[skip] $EXP: $OUT_FILE exists"
    continue
  fi
  if [[ ! -f "$CKPT" ]]; then
    echo "[miss] $EXP: $CKPT missing"
    continue
  fi
  echo "=== $EXP ===" | tee -a "$LOG_FILE"
  date | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES=0 python -u tools/eval/compute_acc.py \
    --model-checkpoint "$CKPT" \
    --output "$OUT_FILE" \
    --data-dir /workspace-SR006.nfs2/weather_data/time_interpolation_1deg_clean \
    --surface-data-dir /workspace-SR006.nfs2/weather_data/time_interpolation_1deg_clean \
    --climatology /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_1deg_full.zarr \
    --stats-path data/json_stats.nc \
    --surface-stats-path data/surface_stats.json \
    --static-path data/static_features.pt \
    --years 2020 \
    --pressure-levels 1000 925 850 700 \
    --surface-variables t2m u10 v10 mslp sst tcc tisr \
    --analytic-tisr \
    --samples-per-date 4 --eval-days-per-month 4 \
    --batch-size 4 --num-workers 2 \
    --cache-in-ram 2>&1 | tee -a "$LOG_FILE"
  echo "=== DONE $EXP ===" | tee -a "$LOG_FILE"
  date
done
echo "=== ALL EXPERIMENTS DONE ==="
