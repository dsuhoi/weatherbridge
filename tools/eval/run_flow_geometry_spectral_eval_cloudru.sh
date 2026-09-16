#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
METRICS_ROOT="${METRICS_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime/metrics}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
GPU="${GPU:-0}"
SLEEP_SEC="${SLEEP_SEC:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-70000}"
MAX_UTIL="${MAX_UTIL:-5}"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

SPECTRAL_EXP="exp_flow_pp3_spectral_14m_6h_s202707_v1_bs4"
SPHERICAL_EXP="exp_flow_pp3_spherical_14m_6h_s202707_v1_bs4"
FLOW_EXP="exp_flow_pp3_135_14m_6h_s202707_protocol_v2"
OUT_DIR="metrics/flow_geometry_spectral_8dpm_6h_2020"
REPORT_ROOT="$METRICS_ROOT/flow_geometry_spectral_6h"
QUEUE_LOG="$LOG_ROOT/flow_geometry_spectral_eval.queue.log"
COMPLETE_MARKER="$LOG_ROOT/flow_geometry_spectral_eval.complete"
TERMINAL_MARKER="$LOG_ROOT/flow_geometry_spectral_eval.terminal"

declare -A CHECKPOINTS=(
  [spectral_2ep]="$LOG_ROOT/$SPECTRAL_EXP/epoch=1-step=3284.ckpt"
  [spectral_4ep]="$LOG_ROOT/$SPECTRAL_EXP/epoch=3-step=6568.ckpt"
  [spectral_8ep]="$LOG_ROOT/$SPECTRAL_EXP/epoch=7-step=13136.ckpt"
  [spherical_2ep]="$LOG_ROOT/$SPHERICAL_EXP/epoch=1-step=3284.ckpt"
  [spherical_4ep]="$LOG_ROOT/$SPHERICAL_EXP/epoch=3-step=6568.ckpt"
  [spherical_8ep]="$LOG_ROOT/$SPHERICAL_EXP/epoch=7-step=13136.ckpt"
  [flow_2ep]="$LOG_ROOT/$FLOW_EXP/epoch=1-step=3284.ckpt"
  [flow_4ep]="$LOG_ROOT/$FLOW_EXP/epoch=3-step=6568.ckpt"
  [flow_8ep]="$LOG_ROOT/$FLOW_EXP/epoch=7-step=13136.ckpt"
)

wait_for_gpu() {
  while true; do
    local free_mib
    local util
    free_mib="$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
    if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      return
    fi
    sleep "$SLEEP_SEC"
  done
}

mkdir -p "$LOG_ROOT" "$REPORT_ROOT"
exec 9>"$LOG_ROOT/flow_geometry_spectral_eval.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] Flow geometry/spectral evaluation already active" \
    | tee -a "$QUEUE_LOG"
  exit 0
fi
rm -f "$COMPLETE_MARKER" "$TERMINAL_MARKER"
trap 'touch "$TERMINAL_MARKER"' ERR

for experiment in "$SPECTRAL_EXP" "$SPHERICAL_EXP"; do
  while [[ ! -e "$LOG_ROOT/$experiment.complete" ]]; do
    echo "[$(date -Is)] wait $experiment" | tee -a "$QUEUE_LOG"
    sleep "$SLEEP_SEC"
  done
done
for checkpoint in "${CHECKPOINTS[@]}"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "missing evaluation checkpoint: $checkpoint" >&2
    exit 2
  fi
done

wait_for_gpu
export CUDA_VISIBLE_DEVICES="$GPU"
"$PYTHON_BIN" -u tools/eval/batch_eval_12h_memmap.py \
  --memmap-dir /tmp/wb2_0p5_cache \
  --test-year 2020 \
  --stats-path data/json_stats_0p5.nc \
  --surface-stats-path data/surface_stats_0p5.json \
  --static-path data/static_features_0p5.pt \
  --models "spectral_2ep:${CHECKPOINTS[spectral_2ep]}:,spectral_4ep:${CHECKPOINTS[spectral_4ep]}:,spectral_8ep:${CHECKPOINTS[spectral_8ep]}:,spherical_2ep:${CHECKPOINTS[spherical_2ep]}:,spherical_4ep:${CHECKPOINTS[spherical_4ep]}:,spherical_8ep:${CHECKPOINTS[spherical_8ep]}:,flow_2ep:${CHECKPOINTS[flow_2ep]}:,flow_4ep:${CHECKPOINTS[flow_4ep]}:,flow_8ep:${CHECKPOINTS[flow_8ep]}:" \
  --out-dir "$OUT_DIR" \
  --paper-tag flow_geometry_spectral_6h_2020 \
  --batch-size 2 \
  --num-workers 2 \
  --samples-per-date 4 \
  --max-tau-hours 6 \
  --eval-hours 1,2,3,4,5 \
  --seen-tau 1,3,5 \
  --unseen-tau 2,4 \
  --keep-n-channels 24 \
  --proper-rmse \
  --eval-days-per-month 8 \
  --no-acc \
  2>&1 | tee -a "$QUEUE_LOG"

for stage in 2 4 8; do
  "$PYTHON_BIN" tools/eval/select_strict_flow_dominance.py \
    --candidate "spectral_${stage}ep=$OUT_DIR/spectral_${stage}ep.json" \
    --candidate "spherical_${stage}ep=$OUT_DIR/spherical_${stage}ep.json" \
    --reference "$OUT_DIR/flow_${stage}ep.json" \
    --cell-limit 0 \
    --include-q \
    --output "$REPORT_ROOT/economy_${stage}ep_strict_120cell.json" \
    2>&1 | tee -a "$QUEUE_LOG"
done

touch "$COMPLETE_MARKER"
trap - ERR
echo "[$(date -Is)] Flow geometry/spectral evaluation complete" \
  | tee -a "$QUEUE_LOG"
