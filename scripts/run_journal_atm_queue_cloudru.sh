#!/usr/bin/env bash
# Train ATM-VFI under the same data/update budgets used by the journal tables.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
GPU=${GPU:-1}
POLL_SECONDS=${POLL_SECONDS:-120}
MAX_USED_MIB=${MAX_USED_MIB:-1000}

cd "$RUNTIME"
export PYTHONPATH="$RUNTIME:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$LOG_ROOT" logs/runner metrics/journal_unified

exec 9>"/tmp/weatherbridge_journal_atm_gpu${GPU}.lock"
flock 9

wait_for_gpu() {
  local used
  while true; do
    used="$(
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        -i "$GPU" | tr -d ' '
    )"
    if (( used <= MAX_USED_MIB )); then
      return
    fi
    echo "[$(date -Is)] wait GPU=$GPU used=${used}MiB"
    sleep "$POLL_SECONDS"
  done
}

train_if_missing() {
  local experiment="$1"
  local expected_epochs="$2"
  shift 2
  local checkpoint="$LOG_ROOT/$experiment/last.ckpt"
  if [[ -s "$checkpoint" ]] && "$PY" - "$checkpoint" "$expected_epochs" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
raise SystemExit(0 if int(checkpoint.get("epoch", -1)) + 1 >= int(sys.argv[2]) else 1)
PY
  then
    echo "[$(date -Is)] skip existing $checkpoint"
    return
  fi
  local resume=()
  if [[ -s "$checkpoint" ]]; then
    resume=(--ckpt_path "$checkpoint")
  fi
  wait_for_gpu
  echo "[$(date -Is)] start $experiment on GPU=$GPU"
  "$PY" -u tools/train/train_capacity_matched_6h.py \
    --arch atmvfi \
    --exp_name "$experiment" \
    --log_root "$LOG_ROOT" \
    --gpus "$GPU" \
    --bs 4 --val_bs 2 --accumulate 4 \
    --workers 8 --val_workers 4 \
    --precision bf16-mixed \
    "${resume[@]}" \
    "$@"
  echo "[$(date -Is)] done $experiment"
}

train_if_missing exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_matched 8 \
  --years 2014 2015 2016 2017 2018 2019 \
  --val_years 2020 \
  --max_epochs 8 --lr 1e-4 \
  --window_hours 6 \
  --train_tau_subset 1 3 5 \
  --eval_tau 1 2 3 4 5 \
  --samples_per_date_train 4 \
  --samples_per_date_val 2

train_if_missing exp_atmvfi_24ch_12h_2017_19_oddskip_lr1e4_matched 10 \
  --years 2017 2018 2019 \
  --val_years 2020 \
  --max_epochs 10 --lr 1e-4 \
  --window_hours 12 \
  --train_tau_subset 1 2 3 5 7 9 10 11 \
  --eval_tau 1 2 3 4 5 6 7 8 9 10 11 \
  --samples_per_date_train 2 \
  --samples_per_date_val 2

touch metrics/journal_unified/.train_atmvfi_complete
