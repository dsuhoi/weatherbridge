#!/usr/bin/env bash
# Validation-gated WeatherDCAE continuation versus band-aware distillation.
set -euo pipefail

PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_multiteacher_v1_source}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
EVAL_ROOT=${EVAL_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/multiteacher_distillation_v1}
MEMMAP=${MEMMAP:-/tmp/wb2_0p5_cache}
DATA_ROOT=${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}
HRES=${HRES:-/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_hres_2021}
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}
POLL_SECONDS=${POLL_SECONDS:-120}
MIN_FREE_MIB=${MIN_FREE_MIB:-70000}
MAX_UTIL=${MAX_UTIL:-5}
IFS_MAX_INITS=${IFS_MAX_INITS:-16}

DCAE_TEACHER=${DCAE_TEACHER:-$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt}
HIGH_TEACHER=${HIGH_TEACHER:-$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt}
CONTROL_EXP=exp_weatherdcae_14m_6h_gt_continuation_s202707_v1_bs4
DISTILL_EXP=exp_weatherdcae_14m_6h_multiteacher_s202707_v1_bs4
ORACLE_DIR=$EVAL_ROOT/oracle
ORACLE_JSON=$ORACLE_DIR/dcae_flow_spectral_2020.json
STATE_DIR=$EVAL_ROOT/state
QUEUE_LOG=$EVAL_ROOT/queue.log

mkdir -p "$ORACLE_DIR" "$STATE_DIR" "$LOG_ROOT" "$EVAL_ROOT/ifs_6h"
exec >>"$QUEUE_LOG" 2>&1
exec 8>"$STATE_DIR/launch.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] multi-teacher queue already active"
  exit 0
fi

verify_inputs() {
  (
    cd "$SOURCE"
    sha256sum --quiet -c SOURCE_SHA256SUMS
  )
  test -s "$DCAE_TEACHER"
  test -s "$HIGH_TEACHER"
  test -s "$DATA_ROOT/static_features_0p5.pt"
  test -s "$DATA_ROOT/json_stats_0p5.nc"
  test -s "$DATA_ROOT/surface_stats_0p5.json"
  test -s "$MEMMAP/wb2_2020.json"
  test -d "$HRES"
}

wait_for_gpu() {
  local gpu="$1" free util
  while true; do
    free=$(nvidia-smi \
      --query-gpu=memory.free \
      --format=csv,noheader,nounits \
      -i "$gpu" | tr -d ' ')
    util=$(nvidia-smi \
      --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits \
      -i "$gpu" | tr -d ' ')
    if [[ "$free" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      return 0
    fi
    echo "[$(date -Is)] wait gpu=$gpu free_mib=$free util=$util"
    sleep "$POLL_SECONDS"
  done
}

oracle_complete() {
  [[ -s "$ORACLE_JSON" ]] || return 1
  "$PY" - "$ORACLE_JSON" "$DCAE_TEACHER" "$HIGH_TEACHER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["status"] == "pass"
assert payload["gate"]["pass"] is True
for name, checkpoint in zip(
    ("low_teacher", "high_teacher"),
    sys.argv[2:],
    strict=True,
):
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    assert payload["models"][name]["checkpoint"]["sha256"] == digest
PY
}

run_oracle_locked() {
  oracle_complete && return 0
  echo "[$(date -Is)] oracle start"
  (
    cd "$SOURCE"
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/eval/eval_multiteacher_oracle.py \
      --low_checkpoint "$DCAE_TEACHER" \
      --high_checkpoint "$HIGH_TEACHER" \
      --output "$ORACLE_JSON" \
      --memmap_dir "$MEMMAP" \
      --static_path "$DATA_ROOT/static_features_0p5.pt" \
      --stats_path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface_stats_path "$DATA_ROOT/surface_stats_0p5.json" \
      --year 2020 \
      --window_hours 6 \
      --tau_hours 1 2 3 4 5 \
      --max_samples 80 \
      --batch_size 1 \
      --workers 2 \
      --gpu 0 \
      --high_mask_profile advected \
      --aggregate_tolerance 0.001 \
      --held_tolerance 0.002
  )
  oracle_complete
  echo "[$(date -Is)] oracle passed"
}

checkpoint_complete() {
  local checkpoint="$1" mode="$2"
  [[ -s "$checkpoint" ]] || return 1
  "$PY" - "$checkpoint" "$mode" "$DCAE_TEACHER" "$HIGH_TEACHER" <<'PY'
import hashlib
import sys
from pathlib import Path

import torch

path, mode, dcae_teacher, high_teacher = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
hparams = checkpoint.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
distillation = protocol.get("distillation", {})
state = checkpoint.get("state_dict", {})
parameters = sum(
    value.numel()
    for name, value in state.items()
    if name.startswith("net.")
)
assert hparams.get("arch") == "dcae_14m"
assert parameters == 14365049
assert int(checkpoint.get("epoch", -1)) >= 1
assert int(checkpoint.get("global_step", -1)) >= 3284
assert protocol.get("train_years") == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol.get("val_years") == [2020]
assert protocol.get("train_tau_hours") == [1, 3, 5]
assert protocol.get("eval_tau_hours") == [1, 2, 3, 4, 5]
assert protocol.get("global_effective_batch_size") == 16
assert protocol.get("optimizer_steps_per_epoch") == 1642
assert protocol.get("lambda_hf") == 0.0
assert protocol.get("lambda_spec") == 0.0
assert protocol.get("lambda_band") == 0.0
assert not any("teacher" in name or "distill" in name for name in state)
if mode == "distill":
    assert distillation.get("enabled") is True
    assert distillation.get("truth_primary") is True
    assert distillation.get("low_weight") == 0.03
    assert distillation.get("high_weight") == 0.08
    assert distillation.get("high_mask_profile") == "advected"
    assert distillation.get("teacher_parameters_saved") is False
    expected = {
        "low_teacher_lineage": dcae_teacher,
        "high_teacher_lineage": high_teacher,
    }
    for key, teacher_path in expected.items():
        digest = hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest()
        assert distillation[key]["checkpoint_sha256"] == digest
else:
    assert distillation.get("enabled") is False
    assert distillation.get("low_weight") == 0.0
    assert distillation.get("high_weight") == 0.0
PY
}

run_ifs_eval() {
  local gpu="$1" exp="$2" checkpoint="$3"
  local output="$EVAL_ROOT/ifs_6h/$exp.json"
  local paired="${output%.json}.paired.npz"
  if [[ -s "$output" && -s "$paired" ]]; then
    return 0
  fi
  (
    cd "$SOURCE"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/eval/batch_eval_forecast_anchor.py \
      --forecast-dir "$HRES" \
      --era5-memmap-dir "$MEMMAP" \
      --checkpoint "$checkpoint" \
      --arch dcae_14m \
      --model-name "$exp" \
      --out-json "$output" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --device cuda \
      --delta-t-hours 6 \
      --taus 1 2 3 4 5 \
      --max-inits "$IFS_MAX_INITS"
  )
}

run_arm() {
  local gpu="$1" exp="$2" mode="$3" prelocked="${4:-false}"
  local output="$LOG_ROOT/$exp"
  local checkpoint="$output/last.ckpt"
  local log="$LOG_ROOT/$exp.log"
  if [[ "$prelocked" == false ]]; then
    exec 5>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
    echo "[$(date -Is)] arm=$mode waiting for gpu=$gpu"
    flock 5
    wait_for_gpu "$gpu"
  else
    echo "[$(date -Is)] arm=$mode inherited gpu=$gpu lock"
  fi
  if ! checkpoint_complete "$checkpoint" "$mode"; then
    local -a lineage_args distill_args
    if [[ -s "$checkpoint" ]]; then
      lineage_args=(--ckpt_path "$checkpoint")
    else
      lineage_args=(--init_weights_path "$DCAE_TEACHER")
    fi
    if [[ "$mode" == distill ]]; then
      distill_args=(
        --distill_low_teacher_checkpoint "$DCAE_TEACHER"
        --distill_high_teacher_checkpoint "$HIGH_TEACHER"
        --distill_low_weight 0.03
        --distill_high_weight 0.08
        --distill_high_mask_profile advected
        --distill_every_n_steps 1
      )
    else
      distill_args=(
        --distill_low_weight 0
        --distill_high_weight 0
        --distill_high_mask_profile advected
      )
    fi
    echo "[$(date -Is)] arm=$mode train start gpu=$gpu exp=$exp"
    (
      cd "$SOURCE"
      export CUDA_VISIBLE_DEVICES="$gpu"
      export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
      "$PY" -u tools/train/train_capacity_matched_6h.py \
        --arch dcae_14m \
        --exp_name "$exp" \
        --log_root "$LOG_ROOT" \
        --gpus 0 \
        --bs 4 \
        --val_bs 2 \
        --accumulate 4 \
        --workers 4 \
        --val_workers 2 \
        --release_memmap_pages \
        --precision bf16-mixed \
        --seed 202707 \
        --years 2014 2015 2016 2017 2018 2019 \
        --val_years 2020 \
        --max_epochs 2 \
        --lr 2e-5 \
        --warmup_steps 100 \
        --window_hours 6 \
        --train_tau_subset 1 3 5 \
        --eval_tau 1 2 3 4 5 \
        --samples_per_date_train 4 \
        --samples_per_date_val 2 \
        --lambda_hf_override 0 \
        --lambda_spec_override 0 \
        --lambda_band_override 0 \
        --lambda_sht_override 0 \
        --spectral_mask_profile all \
        --loss_profile uniform \
        --trainable_scope all \
        --anchor_swap_probability 0 \
        --train_batches_per_epoch 6568 \
        --ckpt_every_n_epochs 1 \
        --memmap_dir "$MEMMAP" \
        --static_path "$DATA_ROOT/static_features_0p5.pt" \
        --stats_path "$DATA_ROOT/json_stats_0p5.nc" \
        --surface_stats_path "$DATA_ROOT/surface_stats_0p5.json" \
        "${distill_args[@]}" \
        "${lineage_args[@]}"
    ) >>"$log" 2>&1
    checkpoint_complete "$checkpoint" "$mode"
  fi
  echo "[$(date -Is)] arm=$mode HRES/IFS evaluation"
  run_ifs_eval "$gpu" "$exp" "$checkpoint" >>"$log" 2>&1
  touch "$STATE_DIR/$mode.complete"
  if [[ "$prelocked" == false ]]; then
    flock -u 5
  fi
  echo "[$(date -Is)] arm=$mode complete"
}

verify_inputs
exec 9>"$LOG_ROOT/.upr_lite_gpu0.lock"
echo "[$(date -Is)] oracle waiting for gpu=0"
flock 9
wait_for_gpu 0
run_oracle_locked
run_arm 0 "$CONTROL_EXP" control true &
CONTROL_PID=$!
run_arm 1 "$DISTILL_EXP" distill &
DISTILL_PID=$!
status=0
wait "$CONTROL_PID" || status=$?
flock -u 9
exec 9>&-
wait "$DISTILL_PID" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "[$(date -Is)] queue failed status=$status"
  exit "$status"
fi
touch "$STATE_DIR/all.complete"
echo "[$(date -Is)] multi-teacher queue complete"
