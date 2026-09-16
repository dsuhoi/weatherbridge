#!/usr/bin/env bash
# Sequential ACC + RMSE eval for all 0.5° trained models on 2020 test.
# Usage: bash scripts/eval_all_0p5.sh
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

mkdir -p metrics/acc_6h_2020_wb11m_only metrics/eval_0p5_2020 logs/runner

ACC_COMMON="--data-dir /workspace-SR006.nfs2/weather_data/time_interpolation_0p5 \
  --surface-data-dir /workspace-SR006.nfs2/weather_data/time_interpolation_0p5 \
  --climatology /workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr \
  --stats-path data/json_stats_0p5.nc --surface-stats-path data/surface_stats_0p5.json \
  --static-path data/static_features_0p5.pt --years 2020 \
  --pressure-levels 1000 925 850 700 \
  --surface-variables t2m u10 v10 mslp sst tcc tcwv \
  --samples-per-date 4 --eval-days-per-month 4 --batch-size 4 --num-workers 0 --cache-in-ram"

RMSE_COMMON="--data-dir /workspace-SR006.nfs2/weather_data/time_interpolation_0p5 \
  --surface-data-dir /workspace-SR006.nfs2/weather_data/time_interpolation_0p5 \
  --stats-path data/json_stats_0p5.nc --surface-stats-path data/surface_stats_0p5.json \
  --static-path data/static_features_0p5.pt --years 2020 \
  --pressure-levels 1000 925 850 700 \
  --surface-variables t2m u10 v10 mslp sst tcc tcwv \
  --in-channels 20 --batch-size 2 --num-workers 0 \
  --samples-per-date 4 --eval-days-per-month 4 --delta-t-hours 6 \
  --per-hour-all --eval-hours 0 1 2 3 4 5 6 --cache-in-ram"

run_acc() {
  NAME=$1; CKPT=$2; shift 2
  OUT="metrics/acc_6h_2020_wb11m_only/${NAME}.json"
  [ -f "$OUT" ] && { echo "[SKIP] ACC $NAME (exists)"; return; }
  [ -f "$CKPT" ] || { echo "[MISS] $CKPT"; return; }
  echo "=== ACC $NAME ===" ; date
  env CUDA_VISIBLE_DEVICES=${ACC_GPU:-0} "$@" python -u tools/eval/compute_acc.py \
    --model-checkpoint "$CKPT" --output "$OUT" $ACC_COMMON \
    2>&1 | tee "logs/runner/acc_eval_${NAME}.log"
  echo "DONE ACC $NAME"
}

run_rmse() {
  NAME=$1; CKPT=$2; shift 2
  OUT="metrics/eval_0p5_2020/${NAME}.json"
  [ -f "$OUT" ] && { echo "[SKIP] RMSE $NAME (exists)"; return; }
  [ -f "$CKPT" ] || { echo "[MISS] $CKPT"; return; }
  echo "=== RMSE $NAME ===" ; date
  env CUDA_VISIBLE_DEVICES=${RMSE_GPU:-1} "$@" python -u evaluate_baselines.py \
    --model-checkpoint "$CKPT" --output "$OUT" --name "$NAME" $RMSE_COMMON \
    2>&1 | tee "logs/runner/rmse_eval_${NAME}.log"
  echo "DONE RMSE $NAME"
}

MODELS=(
  "modafno_0p5_2018:logs/exp_modafno_0p5_2018/last.ckpt:MODAFNO_INP_H=360 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
  "fuxi_0p5_2018:logs/exp_fuxi_0p5_2018_full/last.ckpt:"
  "fuxi_0p5_6yr:logs/exp_fuxi_0p5_6yr_full/last.ckpt:"
  "dcae_skip_0p5_pad:logs/exp_dcae_skip_0p5_2018_pad/last.ckpt:LAT_CROP=-8"
)

MODE=${MODE:-acc}

for entry in "${MODELS[@]}"; do
  IFS=":" read -r NAME CKPT ENVS <<< "$entry"
  if [ "$MODE" = "acc" ]; then
    eval "run_acc $NAME $CKPT $ENVS"
  elif [ "$MODE" = "rmse" ]; then
    eval "run_rmse $NAME $CKPT $ENVS"
  else
    eval "run_acc $NAME $CKPT $ENVS"
    eval "run_rmse $NAME $CKPT $ENVS"
  fi
done

echo "=== ALL EVAL DONE ($MODE) ==="
date
