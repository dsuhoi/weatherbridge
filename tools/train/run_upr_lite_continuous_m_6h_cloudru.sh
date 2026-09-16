#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
MAX_UTIL="${MAX_UTIL:-100}"
SLEEP_SEC="${SLEEP_SEC:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
EXP_NAME="${EXP_NAME:-exp_upr_lite_continuous_m_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707}"
CHECKPOINT="$LOG_ROOT/$EXP_NAME/last.ckpt"
QUEUE_LOG="$LOG_ROOT/$EXP_NAME.queue.log"
RUN_LOG="$LOG_ROOT/$EXP_NAME.log"
batch_size=8
accumulate=2

mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/${EXP_NAME}.queue.lock"
if ! flock -n 9; then
  echo "queue already active exp=$EXP_NAME"
  exit 0
fi

while ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
  "$CHECKPOINT" --min-epochs 8 --quiet; do
  launched=0
  for gpu in $GPU_CANDIDATES; do
    exec {gpu_fd}>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
    if ! flock -n "$gpu_fd"; then
      exec {gpu_fd}>&-
      continue
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ "$free_mib" -lt "$MIN_FREE_MIB" || "$util" -gt "$MAX_UTIL" ]]; then
      flock -u "$gpu_fd"
      exec {gpu_fd}>&-
      continue
    fi
    launched=1
    resume=()
    if [[ -s "$CHECKPOINT" ]]; then
      resume=(--ckpt_path "$CHECKPOINT")
    fi
    echo "[$(date -Is)] launch gpu=$gpu bs=$batch_size accumulate=$accumulate" | tee -a "$QUEUE_LOG"
    set +e
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
      "$PYTHON_BIN" -u tools/train/train_capacity_matched_6h.py \
        --arch upr_lite_continuous_m \
        --exp_name "$EXP_NAME" \
        --log_root "$LOG_ROOT" \
        --gpus 0 \
        --bs "$batch_size" \
        --val_bs 4 \
        --accumulate "$accumulate" \
        --workers 8 \
        --val_workers 4 \
        --precision bf16-mixed \
        --seed 202707 \
        --years 2014 2015 2016 2017 2018 2019 \
        --val_years 2020 \
        --max_epochs 8 \
        --lr 1e-4 \
        --window_hours 6 \
        --train_tau_subset 1 3 5 \
        --eval_tau 1 2 3 4 5 \
        --samples_per_date_train 4 \
        --samples_per_date_val 2 \
        --ckpt_every_n_epochs 1 \
        "${resume[@]}"
    ) >>"$RUN_LOG" 2>&1
    status=$?
    set -e
    flock -u "$gpu_fd"
    exec {gpu_fd}>&-
    echo "[$(date -Is)] finish status=$status" | tee -a "$QUEUE_LOG"
    if [[ "$status" -ne 0 ]] && grep -Eqi \
      "out of memory|CUDA error: out of memory" "$RUN_LOG"; then
      batch_size=4
      accumulate=4
      echo "[$(date -Is)] OOM fallback bs=4 accumulate=4" | tee -a "$QUEUE_LOG"
    fi
    if ! "$PYTHON_BIN" tools/train/checkpoint_status.py \
      "$CHECKPOINT" --min-epochs 8 --quiet; then
      sleep "$SLEEP_SEC"
    fi
    break
  done
  if [[ "$launched" -eq 0 ]]; then
    sleep "$SLEEP_SEC"
  fi
done

echo "[$(date -Is)] complete checkpoint=$CHECKPOINT" | tee -a "$QUEUE_LOG"
