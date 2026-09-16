#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
MIN_FREE_MIB="${MIN_FREE_MIB:-55000}"
MAX_UTIL="${MAX_UTIL:-100}"
SLEEP_SEC="${SLEEP_SEC:-60}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"

mkdir -p "$LOG_ROOT"

run_arch() {
  local arch="$1"
  local exp_name="$2"
  local queue_log="$LOG_ROOT/${exp_name}.queue.log"
  local run_log="$LOG_ROOT/${exp_name}.log"
  local checkpoint="$LOG_ROOT/${exp_name}/last.ckpt"

  if "$PYTHON_BIN" tools/train/checkpoint_status.py \
    "$checkpoint" --min-epochs 8 --quiet; then
    echo "[$(date -Is)] skip complete checkpoint=$checkpoint" | tee -a "$queue_log"
    return 0
  fi

  echo "[$(date -Is)] queue start arch=$arch exp=$exp_name" | tee -a "$queue_log"
  while true; do
    local gpu
    for gpu in $GPU_CANDIDATES; do
      local lock_file="$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
      local lock_fd
      exec {lock_fd}>"$lock_file"
      if ! flock -n "$lock_fd"; then
        exec {lock_fd}>&-
        continue
      fi

      local free_mib
      local util
      free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
      util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
      echo "[$(date -Is)] probe gpu=$gpu free_mib=$free_mib util=$util" | tee -a "$queue_log"
      if [[ "$free_mib" -lt "$MIN_FREE_MIB" || "$util" -gt "$MAX_UTIL" ]]; then
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        continue
      fi

      echo "[$(date -Is)] launch arch=$arch gpu=$gpu" | tee -a "$queue_log"
      local resume=()
      if [[ -s "$checkpoint" ]]; then
        resume=(--ckpt_path "$checkpoint")
        echo "[$(date -Is)] resume arch=$arch checkpoint=$checkpoint" | tee -a "$queue_log"
      fi
      local start_s
      start_s="$(date +%s)"
      set +e
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PYTHONPATH="$EXTRA_PYTHONPATH:$PWD:${PYTHONPATH:-}"
        "$PYTHON_BIN" -u tools/train/train_capacity_matched_6h.py \
          --arch "$arch" \
          --exp_name "$exp_name" \
          --log_root "$LOG_ROOT" \
          --gpus 0 \
          --bs 8 \
          --val_bs 4 \
          --accumulate 2 \
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
          --ckpt_every_n_epochs 2 \
          "${resume[@]}"
      ) >>"$run_log" 2>&1
      local status=$?
      set -e
      local elapsed_s
      elapsed_s="$(( $(date +%s) - start_s ))"
      echo "[$(date -Is)] finish arch=$arch gpu=$gpu status=$status elapsed_s=$elapsed_s" \
        | tee -a "$queue_log"
      flock -u "$lock_fd"
      exec {lock_fd}>&-
      if "$PYTHON_BIN" tools/train/checkpoint_status.py \
        "$checkpoint" --min-epochs 8 --quiet; then
        return 0
      fi
      sleep "$SLEEP_SEC"
      break
    done
    sleep "$SLEEP_SEC"
  done
}

run_arch \
  upr_lite \
  exp_upr_lite_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707 &
pid_base=$!
run_arch \
  upr_lite_lap \
  exp_upr_lite_lap_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707 &
pid_lap=$!
run_arch \
  upr_lite_column \
  exp_upr_lite_column_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707 &
pid_column=$!

status=0
wait "$pid_base" || status=1
wait "$pid_lap" || status=1
wait "$pid_column" || status=1
exit "$status"
