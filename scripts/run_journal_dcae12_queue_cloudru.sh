#!/usr/bin/env bash
# Train the compact 12 h WeatherDCAE reference under the journal oddskip protocol.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
GPU=${GPU:-1}
EXPERIMENT=exp_weatherdcae_14m_24ch_12h_2017_19_lr1e4
CHECKPOINT="$LOG_ROOT/$EXPERIMENT/last.ckpt"

cd "$RUNTIME"
export PYTHONPATH="$RUNTIME:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$LOG_ROOT" logs/runner metrics/journal_unified

# Share the GPU queue with ATM-VFI and the transport ablation retrains.
exec 9>"/tmp/weatherbridge_journal_atm_gpu${GPU}.lock"
flock 9

if [[ -s "$CHECKPOINT" ]] && "$PY" - "$CHECKPOINT" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
raise SystemExit(0 if int(checkpoint.get("epoch", -1)) + 1 >= 10 else 1)
PY
then
  echo "[$(date -Is)] skip existing $CHECKPOINT"
  touch metrics/journal_unified/.train_dcae12_complete
  exit 0
fi

resume=()
if [[ -s "$CHECKPOINT" ]]; then
  resume=(--ckpt_path "$CHECKPOINT")
fi

echo "[$(date -Is)] start $EXPERIMENT on GPU=$GPU"
"$PY" -u tools/train/train_capacity_matched_6h.py \
  --arch weatherdcae \
  --years 2017 2018 2019 \
  --val_years 2020 \
  --max_epochs 10 --lr 1e-4 \
  --window_hours 12 \
  --train_tau_subset 1 2 3 5 7 9 10 11 \
  --eval_tau 1 2 3 4 5 6 7 8 9 10 11 \
  --samples_per_date_train 2 \
  --samples_per_date_val 2 \
  --bs 4 --val_bs 2 --accumulate 4 \
  --workers 8 --val_workers 4 \
  --precision bf16-mixed \
  --gpus "$GPU" \
  "${resume[@]}" \
  --exp_name "$EXPERIMENT" \
  --log_root "$LOG_ROOT"
echo "[$(date -Is)] done $EXPERIMENT"
touch metrics/journal_unified/.train_dcae12_complete
