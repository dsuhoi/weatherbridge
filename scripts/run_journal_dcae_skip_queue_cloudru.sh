#!/usr/bin/env bash
# Train and validate the exact matched WeatherDCAE Skip/NoSkip pair.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
OUT_ROOT=${OUT_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime/metrics/journal_skip_pair_v2}
POLL_SECONDS=${POLL_SECONDS:-120}
MAX_UTIL=${MAX_UTIL:-5}
BASE_EXP=exp_weatherdcae_14m_6h_skip_ablation_noskip_s202707_v2
SKIP_EXP=exp_weatherdcae_14m_6h_skip_ablation_skip_s202707_v2
BASE_CKPT="$LOG_ROOT/$BASE_EXP/last.ckpt"
SKIP_CKPT="$LOG_ROOT/$SKIP_EXP/last.ckpt"
VALIDATION="$OUT_ROOT/checkpoint_pair_validation.json"

cd "$SOURCE"
mkdir -p "$LOG_ROOT" "$OUT_ROOT"
exec 9>/tmp/weatherdcae_skip_pair_v2.lock
flock 9
exec 7>"$LOG_ROOT/.upr_lite_gpu0.lock"
exec 8>"$LOG_ROOT/.upr_lite_gpu1.lock"

checkpoint_complete() {
  local path=$1
  local arch=$2
  [[ -s "$path" ]] || return 1
  "$PY" -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); h=c.get("hyper_parameters",{}); raise SystemExit(0 if c.get("epoch")==7 and c.get("global_step")==6560 and h.get("arch")==sys.argv[2] else 1)' "$path" "$arch"
}

gpu_idle() {
  local gpu=$1
  local util
  util=$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  [[ "$util" -le "$MAX_UTIL" ]]
}

acquire_both_gpu_locks() {
  while true; do
    if flock -n 7; then
      if flock -n 8; then
        if gpu_idle 0 && gpu_idle 1; then
          return 0
        fi
        flock -u 8
      fi
      flock -u 7
    fi
    echo "[$(date -Is)] waiting for both GPUs"
    sleep "$POLL_SECONDS"
  done
}

acquire_both_gpu_locks

run_arm() {
  local gpu=$1
  local arch=$2
  local experiment=$3
  local checkpoint=$4
  if checkpoint_complete "$checkpoint" "$arch"; then
    echo "[$(date -Is)] validated existing $experiment"
    return 0
  fi
  local resume=()
  if [[ -s "$checkpoint" ]]; then
    resume=(--ckpt_path "$checkpoint")
  fi
  "$PY" -u tools/train/train_capacity_matched_6h.py \
    --arch "$arch" --exp_name "$experiment" --log_root "$LOG_ROOT" \
    --gpus "$gpu" --bs 4 --val_bs 2 --accumulate 4 \
    --workers 4 --val_workers 2 --release_memmap_pages \
    --precision bf16-mixed --seed 202707 \
    --years 2017 2018 2019 --val_years 2020 --max_epochs 8 \
    --lr 1e-4 --warmup_steps 500 --window_hours 6 \
    --train_tau_subset 1 3 5 --eval_tau 1 2 3 4 5 \
    --samples_per_date_train 4 --samples_per_date_val 2 \
    --lambda_hf_override 0 --lambda_spec_override 0 --lambda_band_override 0 \
    --loss_profile uniform --trainable_scope all \
    --anchor_swap_probability 0 --train_batches_per_epoch 3280 \
    --ckpt_every_n_epochs 8 "${resume[@]}"
  checkpoint_complete "$checkpoint" "$arch"
}

run_arm 0 dcae_14m "$BASE_EXP" "$BASE_CKPT" \
  >"$LOG_ROOT/$BASE_EXP.queue.log" 2>&1 &
base_pid=$!
run_arm 1 wb_skip "$SKIP_EXP" "$SKIP_CKPT" \
  >"$LOG_ROOT/$SKIP_EXP.queue.log" 2>&1 &
skip_pid=$!
wait "$base_pid"
wait "$skip_pid"

"$PY" tools/eval/validate_dcae_skip_pair.py \
  --noskip "$BASE_CKPT" --skip "$SKIP_CKPT" --output "$VALIDATION"
test -s "$VALIDATION"
echo "[$(date -Is)] matched Skip/NoSkip pair complete"
