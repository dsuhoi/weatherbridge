#!/usr/bin/env bash
# Matched continuation screen for teacher-guided Q-head refinement.
set -euo pipefail

PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_distill_qhead_guided_v8_source}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
OUT_ROOT=${OUT_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_qhead_guided_v8}
MEMMAP=${MEMMAP:-/tmp/wb2_0p5_cache}
DATA_ROOT=${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}
CLIM=${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}
POLL_SECONDS=${POLL_SECONDS:-30}
MIN_FREE_MIB=${MIN_FREE_MIB:-70000}
MAX_UTIL=${MAX_UTIL:-5}

BASE=$LOG_ROOT/exp_weatherdcae_14m_6h_qhead_gt_s202707_v4_bs4/last.ckpt
TEACHER=$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt
CONTROL_EXP=exp_weatherdcae_14m_6h_qhead_gt_cont_s202708_v8_bs4
GUIDED_EXP=exp_weatherdcae_14m_6h_qhead_guided_truth_s202708_v8_bs4
CONTROL=$LOG_ROOT/$CONTROL_EXP/last.ckpt
GUIDED=$LOG_ROOT/$GUIDED_EXP/last.ckpt
STATE=$OUT_ROOT/state
EVAL=$OUT_ROOT/eval_2020_8dpm
LOG=$OUT_ROOT/pipeline.log

mkdir -p "$STATE" "$EVAL"
exec >>"$LOG" 2>&1
exec 8>"$STATE/launch.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] teacher-guided Q-head continuation already active"
  exit 0
fi

(cd "$SOURCE" && sha256sum --quiet -c SOURCE_SHA256SUMS)
for path in "$BASE" "$TEACHER" \
  "$DATA_ROOT/static_features_0p5.pt" \
  "$DATA_ROOT/json_stats_0p5.nc" \
  "$DATA_ROOT/surface_stats_0p5.json"; do
  test -s "$path"
done

checkpoint_complete() {
  local checkpoint=$1
  local mode=$2
  [[ -s "$checkpoint" ]] || return 1
  "$PY" - "$checkpoint" "$BASE" "$mode" <<'PY'
import sys
import torch

candidate = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
base = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
mode = sys.argv[3]
hparams = candidate.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
distill = protocol.get("distillation", {})
state = candidate.get("state_dict", {})
reference = base.get("state_dict", {})
assert hparams.get("arch") == "dcae_14m"
assert sum(v.numel() for k, v in state.items() if k.startswith("net.")) == 14365049
assert int(candidate.get("epoch", -1)) >= 0
assert int(candidate.get("global_step", -1)) >= 821
assert protocol.get("train_years") == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol.get("val_years") == [2020]
assert protocol.get("train_tau_hours") == [1, 3, 5]
assert protocol.get("eval_tau_hours") == [1, 2, 3, 4, 5]
assert protocol.get("global_effective_batch_size") == 16
assert protocol.get("trainable_scope") == "q_output_head"
assert protocol.get("optimizer_weight_decay") == 0.0
if mode == "guided":
    assert distill.get("enabled") is True
    assert distill.get("truth_primary") is True
    assert distill.get("high_mask_active_indices") == [12, 13, 14, 15]
    assert distill.get("high_weight") == 0.04
    assert distill.get("high_gate") == "teacher_better_truth"
    assert distill.get("high_teacher_target") == "teacher_guided_truth_highpass"
else:
    assert distill.get("enabled") is False
assert not any("teacher" in key or "distill" in key for key in state)

mutable = {"net.decoder.conv_out.weight", "net.decoder.conv_out.bias"}
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

wait_for_gpu() {
  local gpu=$1
  while true; do
    local free util
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    if [[ "$free" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      return
    fi
    sleep "$POLL_SECONDS"
  done
}

train_one() {
  local mode=$1
  local gpu=$2
  local exp checkpoint
  local -a distill_args=()
  if [[ "$mode" == guided ]]; then
    exp=$GUIDED_EXP
    checkpoint=$GUIDED
    distill_args=(
      --distill_high_teacher_checkpoint "$TEACHER"
      --distill_high_weight 0.04
      --distill_high_mask_profile moisture
      --distill_high_gate teacher_better_truth
      --distill_every_n_steps 1
      --distill_schedule cosine_decay
      --distill_decay_start_fraction 0
      --distill_decay_end_fraction 0.75
    )
  else
    exp=$CONTROL_EXP
    checkpoint=$CONTROL
  fi
  if checkpoint_complete "$checkpoint" "$mode"; then
    return
  fi
  (
    exec 9>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
    flock 9
    wait_for_gpu "$gpu"
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[$(date -Is)] train mode=$mode gpu=$gpu exp=$exp"
    cd "$SOURCE"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/train/train_capacity_matched_6h.py \
      --arch dcae_14m --exp_name "$exp" --log_root "$LOG_ROOT" --gpus 0 \
      --bs 4 --val_bs 2 --accumulate 4 --workers 4 --val_workers 2 \
      --release_memmap_pages --precision bf16-mixed --seed 202708 \
      --years 2014 2015 2016 2017 2018 2019 --val_years 2020 \
      --max_epochs 1 --lr 5e-6 --warmup_steps 40 --window_hours 6 \
      --train_tau_subset 1 3 5 --eval_tau 1 2 3 4 5 \
      --samples_per_date_train 4 --samples_per_date_val 2 \
      --lambda_hf_override 0 --lambda_spec_override 0 \
      --lambda_band_override 0 --lambda_sht_override 0 \
      --spectral_mask_profile all --loss_profile uniform \
      --trainable_scope q_output_head --anchor_swap_probability 0 \
      --train_batches_per_epoch 3284 --ckpt_every_n_epochs 1 \
      --memmap_dir "$MEMMAP" \
      --static_path "$DATA_ROOT/static_features_0p5.pt" \
      --stats_path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface_stats_path "$DATA_ROOT/surface_stats_0p5.json" \
      --init_weights_path "$BASE" \
      "${distill_args[@]}"
  )
  checkpoint_complete "$checkpoint" "$mode"
}

train_one control 0 &
control_pid=$!
train_one guided 1 &
guided_pid=$!
wait "$control_pid"
wait "$guided_pid"
touch "$STATE/train.complete"

exec 9>"$LOG_ROOT/.upr_lite_gpu0.lock"
flock 9
wait_for_gpu 0
export CUDA_VISIBLE_DEVICES=0
if [[ ! -e "$EVAL/.complete" ]]; then
  echo "[$(date -Is)] evaluate matched continuation pair"
  (
    cd "$SOURCE"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir "$MEMMAP" --test-year 2020 --climatology "$CLIM" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --models "qhead_gt_ext:$CONTROL,qhead_guided_truth:$GUIDED" \
      --out-dir "$EVAL" --paper-tag qhead_guided_v8_2020 \
      --batch-size 4 --num-workers 2 --samples-per-date 2 \
      --eval-days-per-month 8 --proper-rmse --save-window-metrics \
      --save-physical-metrics --save-temporal-metrics \
      --max-tau-hours 6 --eval-hours 1,2,3,4,5 \
      --seen-tau 1,3,5 --unseen-tau 2,4 --device cuda:0 \
      --lazy-climatology --climatology-cache-dir "$OUT_ROOT/climatology_cache" \
      --keep-n-channels 24
  )
  touch "$EVAL/.complete"
fi

SCREEN_ROOT=/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_qhead_v4/eval_2020_8dpm
(cd "$SOURCE" && "$PY" tools/eval/select_specialist_distillation.py \
  --control "$SCREEN_ROOT/qhead_gt.json" \
  --candidate "qhead_guided_truth:$EVAL/qhead_guided_truth.json" \
  --aggregate-tolerance-pct 0.02 --held-tolerance-pct 0.02 \
  --out "$OUT_ROOT/selection.original_gt.json") || true
(cd "$SOURCE" && "$PY" tools/eval/select_specialist_distillation.py \
  --control "$EVAL/qhead_gt_ext.json" \
  --candidate "qhead_guided_truth:$EVAL/qhead_guided_truth.json" \
  --aggregate-tolerance-pct 0.02 --held-tolerance-pct 0.02 \
  --out "$OUT_ROOT/selection.matched_ext.json") || true
if (cd "$SOURCE" && "$PY" tools/eval/select_distilled_champion.py \
  --screen "$OUT_ROOT/selection.original_gt.json" \
  --matched-screen "$OUT_ROOT/selection.matched_ext.json" \
  --output "$OUT_ROOT/selection.strict.json"); then
  touch "$STATE/strict.pass"
  echo "[$(date -Is)] teacher-guided Q-head strict screen passed"
else
  touch "$STATE/strict.failed"
  echo "[$(date -Is)] teacher-guided Q-head strict screen failed"
  exit 2
fi
