#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

EXP_NAME="${EXP_NAME:-exp_weatherbridge_12h_2017_19_held468_lr1e4}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
GPU_LIST="${GPU_LIST:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-${GPU_LIST//,/ }}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
SLEEP_SEC="${SLEEP_SEC:-300}"
ACCUMULATE="${ACCUMULATE:-4}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"

mkdir -p "$LOG_ROOT"
LOCK_FILE="$LOG_ROOT/${EXP_NAME}.queue.lock"
QUEUE_LOG="$LOG_ROOT/${EXP_NAME}.queue.log"
RUN_LOG="$LOG_ROOT/${EXP_NAME}.log"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "queue already active for $EXP_NAME"
  exit 0
fi

echo "[$(date -Is)] queue start exp=$EXP_NAME gpu_list=$GPU_LIST min_free_mib=$MIN_FREE_MIB" | tee -a "$QUEUE_LOG"

while true; do
  active_count="$(pgrep -af "tools/train/train_capacity_matched_6h.py" | grep -v "$EXP_NAME" | wc -l || true)"

  ready=1
  IFS=',' read -ra GPUS <<< "$GPU_LIST"
  for gpu in "${GPUS[@]}"; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    echo "[$(date -Is)] gpu=$gpu free_mib=$free_mib util=$util" | tee -a "$QUEUE_LOG"
    if [ "$free_mib" -lt "$MIN_FREE_MIB" ]; then
      ready=0
    fi
  done
  if [ "$ready" -eq 1 ]; then
    echo "[$(date -Is)] gpu memory ready; active_capacity_processes=$active_count" | tee -a "$QUEUE_LOG"
    break
  fi
  echo "[$(date -Is)] waiting: active_capacity_processes=$active_count" | tee -a "$QUEUE_LOG"
  sleep "$SLEEP_SEC"
done

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"

echo "[$(date -Is)] launch $EXP_NAME" | tee -a "$QUEUE_LOG"
exec "$PYTHON_BIN" -W ignore tools/train/train_capacity_matched_6h.py \
  --arch weatherbridge \
  --years 2017 2018 2019 \
  --val_years 2020 \
  --window_hours 12 \
  --train_tau_subset 1 2 3 5 7 9 10 11 \
  --eval_tau 4 6 8 \
  --bs 4 \
  --accumulate "$ACCUMULATE" \
  --lr 1e-4 \
  --max_epochs 10 \
  --gpus $TRAIN_GPUS \
  --exp_name "$EXP_NAME" \
  --workers 8 \
  --val_workers 4 \
  --ckpt_every_n_epochs 1 \
  --log_root "$LOG_ROOT" \
  > "$RUN_LOG" 2>&1
