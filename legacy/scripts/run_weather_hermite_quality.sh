#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

YEARS_TRAIN="${YEARS_TRAIN:-2016 2017 2018 2019}"
YEARS_EVAL="${YEARS_EVAL:-2020}"
RUN_NAME="${RUN_NAME:-weather_hermite_quality_$(date +%Y%m%d_%H%M%S)}"
DEVICE_ARG="${DEVICE_ARG:-auto}"
DO_EVAL="${DO_EVAL:-1}"

echo "[run] ROOT_DIR=$ROOT_DIR"
echo "[run] RUN_NAME=$RUN_NAME"
echo "[run] YEARS_TRAIN=$YEARS_TRAIN"

python legacy/train_weather_hermite.py   --preset quality   --run-name "${RUN_NAME}"   --years ${YEARS_TRAIN}   --devices "${DEVICE_ARG}"   --save-every 1   --warmup-epochs 4   --min-lr-ratio 0.08   --lambda-latent 0.12   --lambda-reg-d 1e-5   --recon-loss smooth_l1   --smooth-l1-beta 0.02   --gradient-clip-val 0.5   --accumulate-grad-batches 2

BEST_CKPT="$(python - <<PY2
import glob
from pathlib import Path
run = Path('logs') / '${RUN_NAME}'
cands = sorted(glob.glob(str(run / 'model_epoch_*.ckpt')))
print(cands[0] if cands else str(run / 'last.ckpt'))
PY2
)"

echo "[run] BEST_CKPT=${BEST_CKPT}"

if [[ "${DO_EVAL}" == "1" ]]; then
  echo "[run] Evaluating on YEARS_EVAL=${YEARS_EVAL}"
  python evaluate_baselines.py     --years ${YEARS_EVAL}     --model-checkpoint "${BEST_CKPT}"     --per-hour-all     --output "metrics/${RUN_NAME}_eval.json"     --log-dir logs/weather_hermite_eval_auto     --name "${RUN_NAME}_eval"
fi

echo "[done]"
