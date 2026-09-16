#!/usr/bin/env bash
# Truth-gated Q-head distillation follow-up after the matched v4 screen.
set -euo pipefail

PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_distill_qhead_gated_v7_source}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
OUT_ROOT=${OUT_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_qhead_gated_v7}
MEMMAP=${MEMMAP:-/tmp/wb2_0p5_cache}
DATA_ROOT=${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}
CLIM=${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}
POLL_SECONDS=${POLL_SECONDS:-120}
MIN_FREE_MIB=${MIN_FREE_MIB:-70000}
MAX_UTIL=${MAX_UTIL:-5}
GPU=${GPU:-1}

CONTROL=$LOG_ROOT/exp_weatherdcae_14m_6h_gt_continuation_s202707_v1_bs4/last.ckpt
TEACHER=$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt
EXP=exp_weatherdcae_14m_6h_qhead_flow_gated_s202707_v7_bs4
OUTPUT=$LOG_ROOT/$EXP
CHECKPOINT=$OUTPUT/last.ckpt
STATE=$OUT_ROOT/state
EVAL=$OUT_ROOT/eval_2020_8dpm
LOG=$OUT_ROOT/train.log

mkdir -p "$STATE" "$OUTPUT" "$EVAL"
exec >>"$LOG" 2>&1
exec 8>"$STATE/launch.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] gated Q-head follow-up already active"
  exit 0
fi

(cd "$SOURCE" && sha256sum --quiet -c SOURCE_SHA256SUMS)
for path in "$CONTROL" "$TEACHER" \
  "$DATA_ROOT/static_features_0p5.pt" \
  "$DATA_ROOT/json_stats_0p5.nc" \
  "$DATA_ROOT/surface_stats_0p5.json"; do
  test -s "$path"
done

checkpoint_complete() {
  [[ -s "$CHECKPOINT" ]] || return 1
  "$PY" - "$CHECKPOINT" "$CONTROL" <<'PY'
import sys
import torch

candidate = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
control = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
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
assert distill.get("high_weight") == 0.08
assert distill.get("high_gate") == "teacher_better"
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

train_needed=1
if checkpoint_complete; then
  train_needed=0
fi

exec 9>"$LOG_ROOT/.upr_lite_gpu${GPU}.lock"
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  if [[ "$free" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ "$train_needed" -eq 1 ]]; then
  lineage_args=(--init_weights_path "$CONTROL")
  if [[ -s "$CHECKPOINT" ]]; then
    lineage_args=(--ckpt_path "$CHECKPOINT")
  fi
  echo "[$(date -Is)] train gated Q-head gpu=$GPU exp=$EXP"
  (
    cd "$SOURCE"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/train/train_capacity_matched_6h.py \
      --arch dcae_14m --exp_name "$EXP" --log_root "$LOG_ROOT" --gpus 0 \
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
      --distill_high_teacher_checkpoint "$TEACHER" \
      --distill_high_weight 0.08 --distill_high_mask_profile moisture \
      --distill_high_gate teacher_better --distill_every_n_steps 1 \
      --distill_schedule cosine_decay --distill_decay_start_fraction 0 \
      --distill_decay_end_fraction 0.75 \
      --memmap_dir "$MEMMAP" \
      --static_path "$DATA_ROOT/static_features_0p5.pt" \
      --stats_path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface_stats_path "$DATA_ROOT/surface_stats_0p5.json" \
      "${lineage_args[@]}"
  )
fi
checkpoint_complete
touch "$STATE/train.done"

if [[ ! -e "$EVAL/.complete" ]]; then
  echo "[$(date -Is)] evaluate gated Q-head on ERA5-2020 screen"
  (
    cd "$SOURCE"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir "$MEMMAP" --test-year 2020 --climatology "$CLIM" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --models "qhead_flow_gated:$CHECKPOINT" --out-dir "$EVAL" \
      --paper-tag qhead_distill_gated_v7_2020 \
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
test -s "$EVAL/qhead_flow_gated.json"
test -s "$EVAL/window_metrics/qhead_flow_gated.npz"
touch "$STATE/all.complete"
flock -u 9
echo "[$(date -Is)] gated Q-head training and screen complete"
