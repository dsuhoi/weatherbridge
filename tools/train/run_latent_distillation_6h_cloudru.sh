#!/usr/bin/env bash
# Add a block-latent arm after the output-distillation validation screen.
set -euo pipefail

PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_distill_latent_v3_source}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
OUT_ROOT=${OUT_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_latent_v3}
V2_ROOT=${V2_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_specialist_v2}
MEMMAP=${MEMMAP:-/tmp/wb2_0p5_cache}
DATA_ROOT=${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}
CLIM=${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}
HRES=${HRES:-/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_hres_2021}
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}
POLL_SECONDS=${POLL_SECONDS:-120}
MIN_FREE_MIB=${MIN_FREE_MIB:-70000}
MAX_UTIL=${MAX_UTIL:-5}

DCAE_INIT=$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt
FLOW_TEACHER=$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt
CONTROL=$LOG_ROOT/exp_weatherdcae_14m_6h_gt_continuation_s202707_v1_bs4/last.ckpt
Q_DECAY=$LOG_ROOT/exp_weatherdcae_14m_6h_distill_q_decay_s202707_v2_bs4/last.ckpt
Q_ANCHOR=$LOG_ROOT/exp_weatherdcae_14m_6h_distill_q_anchor_decay_s202707_v2_bs4/last.ckpt
EXP=exp_weatherdcae_14m_6h_distill_q_latent_s202707_v3_bs4
LATENT=$LOG_ROOT/$EXP/last.ckpt
STATE=$OUT_ROOT/state
EVAL_2020=$OUT_ROOT/eval_2020_8dpm
EVAL_2021=$OUT_ROOT/eval_2021_8dpm
SPECTRA=$OUT_ROOT/spectra
LOG=$OUT_ROOT/queue.log
LATENT_REJECTION=$STATE/latent_rejected.json

mkdir -p "$STATE" "$EVAL_2020" "$EVAL_2021" "$SPECTRA" "$LOG_ROOT"
exec >>"$LOG" 2>&1
exec 8>"$STATE/launch.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] latent distillation queue already active"
  exit 0
fi

verify_inputs() {
  (cd "$SOURCE" && sha256sum --quiet -c SOURCE_SHA256SUMS)
  for path in "$DCAE_INIT" "$FLOW_TEACHER" "$CONTROL" \
    "$DATA_ROOT/static_features_0p5.pt" \
    "$DATA_ROOT/json_stats_0p5.nc" \
    "$DATA_ROOT/surface_stats_0p5.json"; do
    test -s "$path"
  done
  test -d "$HRES"
  test -d "$CLIM"
}

wait_for_v2_validation() {
  while [[ ! -s "$V2_ROOT/selection.json" ]]; do
    echo "[$(date -Is)] waiting for output-distillation validation screen"
    sleep "$POLL_SECONDS"
  done
  test -s "$Q_DECAY"
  test -s "$Q_ANCHOR"
}

record_rejected_latent_selection() {
  "$PY" - "$V2_ROOT/selection.json" "$LATENT_REJECTION" \
    "$OUT_ROOT/selection_all.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

selection_path = Path(sys.argv[1])
rejection_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
selection = json.loads(selection_path.read_text())
rejection = json.loads(rejection_path.read_text())
selection["selection_extension_role"] = (
    "era5_2020_validation_only_with_rejected_latent_arm"
)
selection["excluded_candidates"] = {"q_latent": rejection}
selection.setdefault("provenance", {})["latent_rejection"] = {
    "path": str(rejection_path.resolve()),
    "sha256": hashlib.sha256(rejection_path.read_bytes()).hexdigest(),
}
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
temporary.write_text(json.dumps(selection, indent=2) + "\n")
temporary.replace(output_path)
PY
  status=$($PY -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$OUT_ROOT/selection_all.json")
  if [[ "$status" == pass ]]; then
    while [[ ! -e "$V2_ROOT/state/all.complete" ]]; do
      echo "[$(date -Is)] waiting for selected output-arm diagnostics"
      sleep "$POLL_SECONDS"
    done
  fi
  touch "$STATE/all.complete"
  echo "[$(date -Is)] q_latent rejected; forwarded output-arm selection"
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

checkpoint_complete() {
  [[ -s "$LATENT" ]] || return 1
  "$PY" - "$LATENT" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
hparams = checkpoint.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
distill = protocol.get("distillation", {})
latent = distill.get("latent", {})
state = checkpoint.get("state_dict", {})
parameters = sum(v.numel() for k, v in state.items() if k.startswith("net."))
assert hparams.get("arch") == "dcae_14m"
assert parameters == 14365049
assert int(checkpoint.get("epoch", -1)) >= 1
assert int(checkpoint.get("global_step", -1)) >= 3284
assert protocol.get("train_years") == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol.get("val_years") == [2020]
assert protocol.get("train_tau_hours") == [1, 3, 5]
assert protocol.get("eval_tau_hours") == [1, 2, 3, 4, 5]
assert protocol.get("global_effective_batch_size") == 16
assert distill.get("truth_primary") is True
assert distill.get("high_mask_profile") == "moisture"
assert distill.get("high_weight") == 0.08
assert distill.get("schedule") == "cosine_decay"
assert distill.get("decay_end_fraction") == 0.75
assert latent.get("weight") == 0.0005
assert latent.get("every_n_steps") == 2
assert latent.get("schedule") == "cosine_decay"
assert latent.get("decay_end_fraction") == 0.5
assert latent.get("block_decay") == 0.5
assert latent.get("pool_shape") == [45, 90]
assert not any("teacher" in key or "distill" in key for key in state)
PY
}

train_arm() {
  local gpu="$1" run_log=$LOG_ROOT/$EXP.log
  if checkpoint_complete; then
    return 0
  fi
  local -a lineage_args
  if [[ -s "$LATENT" ]]; then
    lineage_args=(--ckpt_path "$LATENT")
  else
    lineage_args=(--init_weights_path "$DCAE_INIT")
  fi
  echo "[$(date -Is)] train q_latent gpu=$gpu exp=$EXP"
  (
    cd "$SOURCE"
    export CUDA_VISIBLE_DEVICES="$gpu"
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
      --trainable_scope all --anchor_swap_probability 0 \
      --train_batches_per_epoch 6568 --ckpt_every_n_epochs 1 \
      --distill_high_teacher_checkpoint "$FLOW_TEACHER" \
      --distill_high_weight 0.08 --distill_low_weight 0 \
      --distill_anchor_weight 0 --distill_high_mask_profile moisture \
      --distill_every_n_steps 1 --distill_schedule cosine_decay \
      --distill_decay_start_fraction 0 --distill_decay_end_fraction 0.75 \
      --distill_latent_teacher_checkpoint "$CONTROL" \
      --distill_latent_weight 0.0005 --distill_latent_every_n_steps 2 \
      --distill_latent_schedule cosine_decay \
      --distill_latent_decay_start_fraction 0 \
      --distill_latent_decay_end_fraction 0.5 \
      --distill_latent_block_decay 0.5 \
      --distill_latent_pool_height 45 --distill_latent_pool_width 90 \
      --memmap_dir "$MEMMAP" \
      --static_path "$DATA_ROOT/static_features_0p5.pt" \
      --stats_path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface_stats_path "$DATA_ROOT/surface_stats_0p5.json" \
      "${lineage_args[@]}"
  ) >>"$run_log" 2>&1
  checkpoint_complete
}

claim_and_train() {
  local gpu="$1" status
  exec 6>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
  flock 6
  wait_for_gpu "$gpu"
  exec 7>"$STATE/train.claim.lock"
  if ! flock -n 7; then
    flock -u 6
    return 0
  fi
  if [[ -e "$STATE/train.done" ]]; then
    flock -u 6
    return 0
  fi
  if train_arm "$gpu"; then
    touch "$STATE/train.done"
    status=0
  else
    status=$?
    printf '%s\n' "$status" >"$STATE/train.failed"
  fi
  flock -u 6
  return "$status"
}

run_eval() {
  local gpu="$1" year="$2" out="$3" models="$4"
  mkdir -p "$out"
  if [[ ! -e "$out/.complete" ]]; then
    (
      cd "$SOURCE"
      export CUDA_VISIBLE_DEVICES="$gpu"
      export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
      "$PY" -u tools/eval/batch_eval_12h_memmap.py \
        --memmap-dir "$MEMMAP" --test-year "$year" --climatology "$CLIM" \
        --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
        --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
        --static-path "$DATA_ROOT/static_features_0p5.pt" \
        --models "$models" --out-dir "$out" \
        --paper-tag "latent_distill_v3_${year}" --batch-size 4 \
        --num-workers 2 --samples-per-date 2 --eval-days-per-month 8 \
        --proper-rmse --save-window-metrics --save-physical-metrics \
        --save-temporal-metrics --max-tau-hours 6 \
        --eval-hours 1,2,3,4,5 --seen-tau 1,3,5 --unseen-tau 2,4 \
        --device cuda:0 --lazy-climatology \
        --climatology-cache-dir "$OUT_ROOT/climatology_cache" \
        --keep-n-channels 24
    )
    touch "$out/.complete"
  fi
}

run_spectra() {
  local gpu="$1" year="$2" name="$3" checkpoint="$4" out=$SPECTRA/$year
  mkdir -p "$out"
  (
    cd "$SOURCE"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/eval/sh_energy_spectra_12h.py \
      --memmap-dir "$MEMMAP" --test-year "$year" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --ckpt "$checkpoint" --model-name "$name" --model-kind capmatched \
      --out-dir "$out" --taus 2,3,4 --channels all --lmax 180 \
      --hf-ell-min 80 --keep-n-channels 24 --batch-size 2 \
      --samples-per-date 2 --eval-days-per-month 2 \
      --max-tau-hours 6 --device cuda:0 --skip-existing
  )
}

verify_inputs
wait_for_v2_validation
if [[ -s "$LATENT_REJECTION" ]]; then
  record_rejected_latent_selection
  exit 0
fi
claim_and_train 0 &
PID0=$!
claim_and_train 1 &
PID1=$!
status=0
wait "$PID0" || status=$?
wait "$PID1" || status=$?
[[ "$status" -eq 0 ]] || exit "$status"
checkpoint_complete

exec 9>"$LOG_ROOT/.upr_lite_gpu1.lock"
flock 9
wait_for_gpu 1
MODELS="control:$CONTROL,q_decay:$Q_DECAY,q_anchor_decay:$Q_ANCHOR,q_latent:$LATENT,base_dcae:$DCAE_INIT,flow_teacher:$FLOW_TEACHER"
run_eval 1 2020 "$EVAL_2020" "$MODELS"
(
  cd "$SOURCE"
  "$PY" tools/eval/paired_block_bootstrap.py \
    --left "q_latent:$EVAL_2020/window_metrics/q_latent.npz" \
    --right "control:$EVAL_2020/window_metrics/control.npz" \
    --right "q_decay:$EVAL_2020/window_metrics/q_decay.npz" \
    --right "q_anchor_decay:$EVAL_2020/window_metrics/q_anchor_decay.npz" \
    --draws 5000 --seed 2027 --cellwise \
    --out-json "$EVAL_2020/paired_rmse_q_latent.json"
  "$PY" tools/eval/select_specialist_distillation.py \
    --control "$EVAL_2020/control.json" \
    --candidate "q_decay:$EVAL_2020/q_decay.json" \
    --candidate "q_anchor_decay:$EVAL_2020/q_anchor_decay.json" \
    --candidate "q_latent:$EVAL_2020/q_latent.json" \
    --aggregate-tolerance-pct 0.02 --held-tolerance-pct 0.02 \
    --out "$OUT_ROOT/selection_all.json"
)
SELECTED=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' "$OUT_ROOT/selection_all.json")
if [[ "$SELECTED" == q_latent ]]; then
  OOD_MODELS="control:$CONTROL,selected:$LATENT,base_dcae:$DCAE_INIT,flow_teacher:$FLOW_TEACHER"
  run_eval 1 2021 "$EVAL_2021" "$OOD_MODELS"
  (
    cd "$SOURCE"
    "$PY" tools/eval/paired_block_bootstrap.py \
      --left "selected:$EVAL_2021/window_metrics/selected.npz" \
      --right "control:$EVAL_2021/window_metrics/control.npz" \
      --right "base_dcae:$EVAL_2021/window_metrics/base_dcae.npz" \
      --draws 5000 --seed 2027 --cellwise \
      --out-json "$EVAL_2021/paired_rmse_selected.json"
  )
  for year in 2020 2021; do
    run_spectra 1 "$year" control "$CONTROL"
    run_spectra 1 "$year" selected "$LATENT"
    for tau in 2 3 4; do
      (
        cd "$SOURCE"
        "$PY" tools/eval/spectral_block_bootstrap.py \
          --left "selected:$SPECTRA/$year/selected_tau${tau}.npz" \
          --right "control:$SPECTRA/$year/control_tau${tau}.npz" \
          --channels all --block-days 7 --draws 5000 --seed 2027 --cellwise \
          --out-json "$SPECTRA/$year/selected_vs_control_tau${tau}.json"
        "$PY" tools/eval/spectral_block_bootstrap.py \
          --left "selected:$SPECTRA/$year/selected_tau${tau}.npz" \
          --right "control:$SPECTRA/$year/control_tau${tau}.npz" \
          --channels Q1000,Q925,Q850,Q700 --block-days 7 \
          --draws 5000 --seed 2027 --cellwise \
          --out-json "$SPECTRA/$year/selected_vs_control_q_tau${tau}.json"
      )
    done
  done
  (
    cd "$SOURCE"
    export CUDA_VISIBLE_DEVICES=1
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/eval/batch_eval_forecast_anchor.py \
      --forecast-dir "$HRES" --era5-memmap-dir "$MEMMAP" \
      --checkpoint "$LATENT" --arch dcae_14m --model-name q_latent \
      --out-json "$OUT_ROOT/q_latent_hres.json" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" --device cuda \
      --delta-t-hours 6 --taus 1 2 3 4 5 --max-inits 16
  )
fi
flock -u 9
touch "$STATE/all.complete"
echo "[$(date -Is)] selected=$SELECTED latent distillation screen complete"
