#!/usr/bin/env bash
# Conflict-free Q-head distillation with a matched GT-only head control.
set -euo pipefail

PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_distill_qhead_v4_source}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
OUT_ROOT=${OUT_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_qhead_v4}
MEMMAP=${MEMMAP:-/tmp/wb2_0p5_cache}
DATA_ROOT=${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}
CLIM=${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}
POLL_SECONDS=${POLL_SECONDS:-120}
MIN_FREE_MIB=${MIN_FREE_MIB:-70000}
MAX_UTIL=${MAX_UTIL:-5}

CONTROL=$LOG_ROOT/exp_weatherdcae_14m_6h_gt_continuation_s202707_v1_bs4/last.ckpt
FLOW_TEACHER=$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt
STATE=$OUT_ROOT/state
EVAL=$OUT_ROOT/eval_2020_8dpm
LOG=$OUT_ROOT/queue.log
ARMS=(qhead_gt qhead_flow)

mkdir -p "$STATE" "$EVAL" "$LOG_ROOT"
exec >>"$LOG" 2>&1
exec 8>"$STATE/launch.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] Q-head distillation queue already active"
  exit 0
fi

verify_inputs() {
  (cd "$SOURCE" && sha256sum --quiet -c SOURCE_SHA256SUMS)
  for path in "$CONTROL" "$FLOW_TEACHER" \
    "$DATA_ROOT/static_features_0p5.pt" \
    "$DATA_ROOT/json_stats_0p5.nc" \
    "$DATA_ROOT/surface_stats_0p5.json"; do
    test -s "$path"
  done
  test -d "$CLIM"
}

wait_for_gpu() {
  local gpu="$1" free util
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    if [[ "$free" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

experiment_for() {
  case "$1" in
    qhead_gt) echo exp_weatherdcae_14m_6h_qhead_gt_s202707_v4_bs4 ;;
    qhead_flow) echo exp_weatherdcae_14m_6h_qhead_flow_s202707_v4_bs4 ;;
    *) return 2 ;;
  esac
}

claim_arm() {
  local arm
  exec 7>"$STATE/claims.lock"
  flock 7
  for arm in "${ARMS[@]}"; do
    if [[ ! -e "$STATE/$arm.done" && ! -e "$STATE/$arm.claimed" ]]; then
      printf '%s\n' "$$" >"$STATE/$arm.claimed"
      flock -u 7
      printf '%s\n' "$arm"
      return 0
    fi
  done
  flock -u 7
  return 1
}

checkpoint_complete() {
  local checkpoint="$1" arm="$2"
  [[ -s "$checkpoint" ]] || return 1
  "$PY" - "$checkpoint" "$CONTROL" "$arm" <<'PY'
import sys
import torch

candidate = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
control = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
arm = sys.argv[3]
hparams = candidate.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
distill = protocol.get("distillation", {})
state = candidate.get("state_dict", {})
reference = control.get("state_dict", {})
assert hparams.get("arch") == "dcae_14m"
assert sum(v.numel() for k, v in state.items() if k.startswith("net.")) == 14365049
assert int(candidate.get("epoch", -1)) >= 1
assert int(candidate.get("global_step", -1)) >= 3284
assert protocol.get("train_years") == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol.get("val_years") == [2020]
assert protocol.get("train_tau_hours") == [1, 3, 5]
assert protocol.get("eval_tau_hours") == [1, 2, 3, 4, 5]
assert protocol.get("global_effective_batch_size") == 16
assert protocol.get("trainable_scope") == "q_output_head"
assert protocol.get("optimizer_weight_decay") == 0.0
assert distill.get("truth_primary") is True
assert distill.get("high_mask_active_indices") == [12, 13, 14, 15]
assert distill.get("high_weight") == (0.08 if arm == "qhead_flow" else 0.0)
assert not any("teacher" in key or "distill" in key for key in state)

mutable = {
    "net.decoder.conv_out.weight",
    "net.decoder.conv_out.bias",
}
for key, value in state.items():
    if not key.startswith("net.") or key in mutable:
        continue
    assert torch.equal(value, reference[key]), key
for key in mutable:
    assert torch.equal(state[key][:12], reference[key][:12]), key
    assert torch.equal(state[key][16:], reference[key][16:]), key
    assert not torch.equal(state[key][12:16], reference[key][12:16]), key
PY
}

train_arm() {
  local gpu="$1" arm="$2" exp output checkpoint run_log
  exp=$(experiment_for "$arm")
  output=$LOG_ROOT/$exp
  checkpoint=$output/last.ckpt
  run_log=$LOG_ROOT/$exp.log
  if checkpoint_complete "$checkpoint" "$arm"; then
    return 0
  fi
  local -a lineage_args teacher_args
  if [[ -s "$checkpoint" ]]; then
    lineage_args=(--ckpt_path "$checkpoint")
  else
    lineage_args=(--init_weights_path "$CONTROL")
  fi
  if [[ "$arm" == qhead_flow ]]; then
    teacher_args=(
      --distill_high_teacher_checkpoint "$FLOW_TEACHER"
      --distill_high_weight 0.08
    )
  else
    teacher_args=(--distill_high_weight 0)
  fi
  echo "[$(date -Is)] train arm=$arm gpu=$gpu exp=$exp"
  (
    cd "$SOURCE"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/train/train_capacity_matched_6h.py \
      --arch dcae_14m --exp_name "$exp" --log_root "$LOG_ROOT" --gpus 0 \
      --bs 4 --val_bs 2 --accumulate 4 --workers 4 --val_workers 2 \
      --release_memmap_pages --precision bf16-mixed --seed 202707 \
      --years 2014 2015 2016 2017 2018 2019 --val_years 2020 \
      --max_epochs 2 --lr 2e-5 --warmup_steps 100 --window_hours 6 \
      --train_tau_subset 1 3 5 --eval_tau 1 2 3 4 5 \
      --samples_per_date_train 4 --samples_per_date_val 2 \
      --lambda_hf_override 0 --lambda_spec_override 0 \
      --lambda_band_override 0 --lambda_sht_override 0 \
      --spectral_mask_profile all --loss_profile uniform \
      --trainable_scope q_output_head --anchor_swap_probability 0 \
      --train_batches_per_epoch 6568 --ckpt_every_n_epochs 1 \
      --distill_high_mask_profile moisture --distill_every_n_steps 1 \
      --distill_schedule cosine_decay --distill_decay_start_fraction 0 \
      --distill_decay_end_fraction 0.75 \
      --memmap_dir "$MEMMAP" \
      --static_path "$DATA_ROOT/static_features_0p5.pt" \
      --stats_path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface_stats_path "$DATA_ROOT/surface_stats_0p5.json" \
      "${teacher_args[@]}" "${lineage_args[@]}"
  ) >>"$run_log" 2>&1
  checkpoint_complete "$checkpoint" "$arm"
}

worker() {
  local gpu="$1" arm status
  exec 6>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
  while true; do
    flock 6
    wait_for_gpu "$gpu"
    if ! arm=$(claim_arm); then
      flock -u 6
      break
    fi
    if train_arm "$gpu" "$arm"; then
      touch "$STATE/$arm.done"
      status=0
    else
      status=$?
      printf '%s\n' "$status" >"$STATE/$arm.failed"
    fi
    rm -f "$STATE/$arm.claimed"
    flock -u 6
    [[ "$status" -eq 0 ]] || return "$status"
  done
}

run_eval() {
  local models="$1"
  if [[ ! -e "$EVAL/.complete" ]]; then
    (cd "$SOURCE" && "$PY" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir "$MEMMAP" --test-year 2020 --climatology "$CLIM" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --models "$models" --out-dir "$EVAL" \
      --paper-tag qhead_distill_v4_2020 --batch-size 4 --num-workers 2 \
      --samples-per-date 2 --eval-days-per-month 8 --proper-rmse \
      --save-window-metrics --save-physical-metrics --save-temporal-metrics \
      --max-tau-hours 6 --eval-hours 1,2,3,4,5 \
      --seen-tau 1,3,5 --unseen-tau 2,4 --device cuda:0 \
      --lazy-climatology --climatology-cache-dir "$OUT_ROOT/climatology_cache" \
      --keep-n-channels 24)
    touch "$EVAL/.complete"
  fi
}

verify_inputs
worker 0 & PID0=$!
worker 1 & PID1=$!
status=0
wait "$PID0" || status=$?
wait "$PID1" || status=$?
[[ "$status" -eq 0 ]] || exit "$status"

GT=$LOG_ROOT/$(experiment_for qhead_gt)/last.ckpt
FLOW=$LOG_ROOT/$(experiment_for qhead_flow)/last.ckpt
exec 9>"$LOG_ROOT/.upr_lite_gpu0.lock"
flock 9
wait_for_gpu 0
export CUDA_VISIBLE_DEVICES=0
MODELS="control:$CONTROL,qhead_gt:$GT,qhead_flow:$FLOW,flow_teacher:$FLOW_TEACHER"
run_eval "$MODELS"
for candidate in qhead_gt qhead_flow; do
  for group in all moisture non_moisture; do
    channels=()
    if [[ "$group" == moisture ]]; then
      channels=(--channels Q1000,Q925,Q850,Q700)
    elif [[ "$group" == non_moisture ]]; then
      channels=(--channels T1000,T925,T850,T700,U1000,U925,U850,U700,V1000,V925,V850,V700,Z1000,Z925,Z850,Z700,t2m,u10,v10,mslp)
    fi
    (cd "$SOURCE" && "$PY" tools/eval/paired_block_bootstrap.py \
      --left "$candidate:$EVAL/window_metrics/$candidate.npz" \
      --right "control:$EVAL/window_metrics/control.npz" \
      --draws 5000 --seed 2027 --cellwise "${channels[@]}" \
      --out-json "$EVAL/paired_${candidate}_${group}.json")
  done
done
(cd "$SOURCE" && "$PY" tools/eval/select_specialist_distillation.py \
  --control "$EVAL/control.json" \
  --candidate "qhead_gt:$EVAL/qhead_gt.json" \
  --candidate "qhead_flow:$EVAL/qhead_flow.json" \
  --aggregate-tolerance-pct 0.02 --held-tolerance-pct 0.02 \
  --out "$OUT_ROOT/selection.json") || true
if (cd "$SOURCE" && "$PY" tools/eval/select_distilled_champion.py \
  --screen "$OUT_ROOT/selection.json" \
  --output "$OUT_ROOT/selection.strict.json"); then
  touch "$STATE/strict.pass"
else
  touch "$STATE/strict.failed"
fi
flock -u 9
touch "$STATE/screen.complete"
echo "[$(date -Is)] Q-head distillation screen complete"
