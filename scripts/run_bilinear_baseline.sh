#!/usr/bin/env bash
# Перерасчёт bilinear/bicubic baseline на test 2020 в двух режимах:
#   - economy: ~K=4 дней/месяц × 4 start_hours
#   - full:    все 366 дней × 4 start_hours
#
# Usage:
#   bash scripts/run_bilinear_baseline.sh [DELTA_T=6] [TEST_YEAR=2020]
set -eo pipefail
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/.mlspace/envs/ai_scientist
cd "$(dirname "$0")/.."

DELTA_T="${1:-6}"
TEST_YEAR="${2:-2020}"
DATA_DIR="${DATA_DIR:-/tmp/zarrs}"
EVAL_HOURS=$(seq 0 "$DELTA_T" | tr '\n' ' ')

mkdir -p metrics/eval_${TEST_YEAR} metrics/eval_full_${TEST_YEAR} logs/runner

COMMON=(
  --data-dir "$DATA_DIR" --years "$TEST_YEAR"
  --batch-size 8 --num-workers 4 --samples-per-date 4
  --in-channels 20 --pressure-levels 1000 925 850 700
  --static-path data/static_features.pt --stats-path data/json_stats.nc
  --surface-data-dir "$DATA_DIR"
  --surface-variables t2m u10 v10 mslp sst tcc tisr
  --surface-stats-path data/surface_stats.json
  --analytic-tisr --cache-in-ram
  --delta-t-hours "$DELTA_T" --skip-bicubic
  --per-hour-all --eval-hours $EVAL_HOURS
)

# ECONOMY (~K=4 дней/месяц): быстро, для leaderboard каждого ран-а
echo "=== ECONOMY BILINEAR baseline (K=4 days/month × 4 start_hours, Δt=${DELTA_T}h) ==="
date
python -u evaluate_baselines.py \
  "${COMMON[@]}" \
  --eval-days-per-month 4 \
  --output "metrics/eval_${TEST_YEAR}/bilinear_econ_${DELTA_T}h.json" \
  --name "bilinear_econ_${DELTA_T}h" 2>&1 | tee logs/runner/bilinear_econ_${DELTA_T}h.log

echo
echo "=== FULL BILINEAR baseline (all 366 days × 4 start_hours, Δt=${DELTA_T}h) ==="
date
python -u evaluate_baselines.py \
  "${COMMON[@]}" \
  --output "metrics/eval_full_${TEST_YEAR}/bilinear_full_${DELTA_T}h.json" \
  --name "bilinear_full_${DELTA_T}h" 2>&1 | tee logs/runner/bilinear_full_${DELTA_T}h.log

echo "=== DONE BILINEAR BASELINES ==="
date
