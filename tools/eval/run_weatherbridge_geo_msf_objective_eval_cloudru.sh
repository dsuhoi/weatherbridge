#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
METRICS_ROOT="${METRICS_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime/metrics}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
GPU="${GPU:-0}"
SLEEP_SEC="${SLEEP_SEC:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
MAX_UTIL="${MAX_UTIL:-5}"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

UNIFORM_EXP="exp_weatherbridge_geo_msf_l_9m_6h_s202707_uniform_v1_bs8"
BASEEDGE_EXP="exp_weatherbridge_geo_msf_l_9m_6h_s202707_baseedge_v1_bs8"
FLOW_CHECKPOINT="$LOG_ROOT/exp_flow_pp3_135_14m_6h_s202707_protocol_v2/epoch=3-step=6568.ckpt"
OUT_DIR="metrics/weatherbridge_geo_msf_objective_8dpm_6h_2020"
REPORT_ROOT="$METRICS_ROOT/weatherbridge_geo_msf_objective_6h"
REPORT="$REPORT_ROOT/economy_strict_120cell.json"
QUEUE_LOG="$LOG_ROOT/weatherbridge_geo_msf_objective_eval.queue.log"
COMPLETE_MARKER="$LOG_ROOT/weatherbridge_geo_msf_objective_eval.complete"
TERMINAL_MARKER="$LOG_ROOT/weatherbridge_geo_msf_objective_eval.terminal"

declare -A CHECKPOINTS=(
  [geo_uniform_2ep]="$LOG_ROOT/$UNIFORM_EXP/epoch=1-step=3284.ckpt"
  [geo_uniform_4ep]="$LOG_ROOT/$UNIFORM_EXP/epoch=3-step=6568.ckpt"
  [geo_baseedge_2ep]="$LOG_ROOT/$BASEEDGE_EXP/epoch=1-step=3284.ckpt"
  [geo_baseedge_4ep]="$LOG_ROOT/$BASEEDGE_EXP/epoch=3-step=6568.ckpt"
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
exec 9>"$LOG_ROOT/weatherbridge_geo_msf_objective_eval.queue.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] GeoMSF objective evaluation already active" \
    | tee -a "$QUEUE_LOG"
  exit 0
fi
rm -f "$COMPLETE_MARKER" "$TERMINAL_MARKER"
trap 'touch "$TERMINAL_MARKER"' ERR

for experiment in "$UNIFORM_EXP" "$BASEEDGE_EXP"; do
  while [[ ! -e "$LOG_ROOT/$experiment.complete" ]]; do
    echo "[$(date -Is)] wait $experiment" | tee -a "$QUEUE_LOG"
    sleep "$SLEEP_SEC"
  done
done
for checkpoint in "${CHECKPOINTS[@]}" "$FLOW_CHECKPOINT"; do
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
  --models "geo_uniform_2ep:${CHECKPOINTS[geo_uniform_2ep]}:,geo_uniform_4ep:${CHECKPOINTS[geo_uniform_4ep]}:,geo_baseedge_2ep:${CHECKPOINTS[geo_baseedge_2ep]}:,geo_baseedge_4ep:${CHECKPOINTS[geo_baseedge_4ep]}:,flow_4ep_ref:$FLOW_CHECKPOINT:" \
  --out-dir "$OUT_DIR" \
  --paper-tag weatherbridge_geo_msf_objective_6h_2020 \
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

"$PYTHON_BIN" tools/eval/select_strict_flow_dominance.py \
  --candidate "geo_uniform_2ep=$OUT_DIR/geo_uniform_2ep.json" \
  --candidate "geo_uniform_4ep=$OUT_DIR/geo_uniform_4ep.json" \
  --candidate "geo_baseedge_2ep=$OUT_DIR/geo_baseedge_2ep.json" \
  --candidate "geo_baseedge_4ep=$OUT_DIR/geo_baseedge_4ep.json" \
  --reference "$OUT_DIR/flow_4ep_ref.json" \
  --cell-limit 0 \
  --include-q \
  --output "$REPORT" \
  2>&1 | tee -a "$QUEUE_LOG"

touch "$COMPLETE_MARKER"
trap - ERR
echo "[$(date -Is)] GeoMSF objective evaluation complete" \
  | tee -a "$QUEUE_LOG"
