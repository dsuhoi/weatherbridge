#!/usr/bin/env bash
# Retrain the two unmatched transport ablations under the sparse-τ budget.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
GPU=${GPU:-1}

cd "$RUNTIME"
export PYTHONPATH="$RUNTIME:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$LOG_ROOT" logs/runner

# Canonical table checkpoints take priority over secondary transport
# ablations. Their queue scripts write these markers only after validating
# the expected final epoch.
for marker in \
  metrics/journal_unified/.train_dcae12_complete \
  metrics/journal_unified/.train_dcae_skip_complete \
  metrics/journal_unified/.train_atmvfi_complete; do
  while [[ ! -e "$marker" ]]; do
    sleep 120
  done
done

# Blocking here serializes the remaining GPU 1 work.
exec 9>"/tmp/weatherbridge_journal_atm_gpu${GPU}.lock"
flock 9

train_if_missing() {
  local arch="$1"
  local experiment="$2"
  local expected_epochs="$3"
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
  echo "[$(date -Is)] start $experiment on GPU=$GPU"
  "$PY" -u tools/train/train_capacity_matched_6h.py \
    --arch "$arch" \
    --years 2014 2015 2016 2017 2018 2019 \
    --val_years 2020 \
    --max_epochs 8 --lr 1e-4 \
    --window_hours 6 \
    --train_tau_subset 1 3 5 \
    --eval_tau 1 2 3 4 5 \
    --samples_per_date_train 4 \
    --samples_per_date_val 2 \
    --bs 16 --val_bs 16 \
    --workers 8 --val_workers 4 \
    --precision bf16-mixed \
    --gpus "$GPU" \
    "${resume[@]}" \
    --exp_name "$experiment" \
    --log_root "$LOG_ROOT"
  echo "[$(date -Is)] done $experiment"
}

train_if_missing flow_noskip exp_flow_noskip_135_matched_14m_6h 8
train_if_missing flow_ungated exp_flow_ungated_135_matched_14m_6h 8
