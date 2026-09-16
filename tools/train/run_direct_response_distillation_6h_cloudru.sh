#!/usr/bin/env bash
# Matched Flow response distillation from a complementary DCAE teacher.
set -euo pipefail

PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_direct_distill_v13_source}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
OUT_ROOT=${OUT_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_direct_response_v13}
MEMMAP=${MEMMAP:-/tmp/wb2_0p5_cache}
DATA_ROOT=${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}
CLIM=${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}
POLL_SECONDS=${POLL_SECONDS:-30}
MIN_FREE_MIB=${MIN_FREE_MIB:-70000}
MAX_UTIL=${MAX_UTIL:-5}

BASE=$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt
TEACHER=$LOG_ROOT/exp_weatherdcae_14m_6h_direct_q_gt_control_s202710_v12_bs4/last.ckpt
BASE_EVAL=/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_qhead_v4/eval_2020_8dpm/flow_teacher.json
TEACHER_EVAL=/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_direct_response_v12/eval_2020_8dpm/direct_gt_control.json
CONTROL_EXP=exp_flow_pp3_14m_6h_direct_gt_control_s202711_v13_bs4
STUDENT_EXP=exp_flow_pp3_14m_6h_direct_dcae_student_s202711_v13_bs4
CONTROL=$LOG_ROOT/$CONTROL_EXP/last.ckpt
STUDENT=$LOG_ROOT/$STUDENT_EXP/last.ckpt
ROUTE=$OUT_ROOT/direct_route_2020.json
STATE=$OUT_ROOT/state
EVAL=$OUT_ROOT/eval_2020_8dpm
LOG=$OUT_ROOT/pipeline.log

mkdir -p "$STATE" "$EVAL"
exec >>"$LOG" 2>&1
exec 8>"$STATE/launch.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] direct response distillation already active"
  exit 0
fi

(cd "$SOURCE" && sha256sum --quiet -c SOURCE_SHA256SUMS)
for path in "$BASE" "$TEACHER" "$BASE_EVAL" "$TEACHER_EVAL" \
  "$DATA_ROOT/static_features_0p5.pt" \
  "$DATA_ROOT/json_stats_0p5.nc" \
  "$DATA_ROOT/surface_stats_0p5.json"; do
  test -s "$path"
done

if [[ ! -s "$ROUTE" ]]; then
  (
    cd "$SOURCE"
    "$PY" tools/eval/build_direct_distillation_route.py \
      --control "$BASE_EVAL" --teacher "$TEACHER_EVAL" \
      --minimum-improvement-pct 0.5 --routing-granularity field \
      --output "$ROUTE"
  )
fi
"$PY" - "$ROUTE" "$BASE_EVAL" "$TEACHER_EVAL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

route = json.loads(Path(sys.argv[1]).read_text())
assert route["selection_role"] == "era5_2020_validation_only"
assert route["selection_year"] == 2020
assert route["minimum_teacher_improvement_pct"] == 0.5
assert route["routing_granularity"] == "field"
assert route["tau_hours"] == [1, 2, 3, 4, 5]
assert len(route["channel_names"]) == 24
assert route["active_routes"] == 25
active = [index for index, value in enumerate(route["teacher_route"][0]) if value]
assert active == [16, 17, 18, 19, 23]
assert route["control"]["sha256"] == hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
assert route["teacher"]["sha256"] == hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()
PY

checkpoint_complete() {
  local checkpoint=$1
  local mode=$2
  [[ -s "$checkpoint" ]] || return 1
  "$PY" - "$checkpoint" "$BASE" "$TEACHER" "$ROUTE" "$mode" <<'PY'
import hashlib
import sys
from pathlib import Path

import torch

candidate = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
base_path, teacher_path, route_path, mode = sys.argv[2:]
hparams = candidate.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
distill = protocol.get("distillation", {})
state = candidate.get("state_dict", {})
assert hparams.get("arch") == "flow_pp3"
assert sum(v.numel() for k, v in state.items() if k.startswith("net.")) == 14260565
assert int(candidate.get("epoch", -1)) >= 0
assert int(candidate.get("global_step", -1)) >= 410
assert protocol.get("train_years") == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol.get("val_years") == [2020]
assert protocol.get("train_tau_hours") == [1, 3, 5]
assert protocol.get("eval_tau_hours") == [1, 2, 3, 4, 5]
assert protocol.get("global_effective_batch_size") == 16
assert protocol.get("trainable_scope") == "all"
assert protocol.get("optimizer_weight_decay") == 1.0e-4
lineage = hparams["initialization_lineage"]
assert lineage["checkpoint_sha256"] == hashlib.sha256(Path(base_path).read_bytes()).hexdigest()
assert not any("teacher" in key or "distill" in key for key in state)
if mode == "student":
    assert distill.get("enabled") is True
    assert distill.get("truth_primary") is True
    direct = distill["direct"]
    assert direct["blend"] == 0.5
    assert direct["target"] == "routed_truth_teacher_convex_response"
    assert direct["route"]["sha256"] == hashlib.sha256(Path(route_path).read_bytes()).hexdigest()
    assert direct["teacher_lineage"]["checkpoint_sha256"] == hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest()
    assert distill["schedule"] == "cosine_decay"
    assert distill["decay_end_fraction"] == 0.75
else:
    assert distill.get("enabled") is False
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
  local -a direct_args=()
  if [[ "$mode" == student ]]; then
    exp=$STUDENT_EXP
    checkpoint=$STUDENT
    direct_args=(
      --distill_direct_teacher_checkpoint "$TEACHER"
      --distill_direct_route_path "$ROUTE"
      --distill_direct_blend 0.5
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
      --arch flow_pp3 --exp_name "$exp" --log_root "$LOG_ROOT" --gpus 0 \
      --bs 4 --val_bs 2 --accumulate 4 --workers 4 --val_workers 2 \
      --release_memmap_pages --precision bf16-mixed --seed 202711 \
      --years 2014 2015 2016 2017 2018 2019 --val_years 2020 \
      --max_epochs 1 --lr 1e-5 --warmup_steps 100 --window_hours 6 \
      --train_tau_subset 1 3 5 --eval_tau 1 2 3 4 5 \
      --samples_per_date_train 4 --samples_per_date_val 2 \
      --lambda_hf_override 0 --lambda_spec_override 0 \
      --lambda_band_override 0 --lambda_sht_override 0 \
      --spectral_mask_profile all --loss_profile uniform \
      --trainable_scope all --anchor_swap_probability 0 \
      --train_batches_per_epoch 1640 --ckpt_every_n_epochs 1 \
      --distill_schedule cosine_decay --distill_decay_start_fraction 0 \
      --distill_decay_end_fraction 0.75 \
      --memmap_dir "$MEMMAP" \
      --static_path "$DATA_ROOT/static_features_0p5.pt" \
      --stats_path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface_stats_path "$DATA_ROOT/surface_stats_0p5.json" \
      --init_weights_path "$BASE" \
      "${direct_args[@]}"
  )
  checkpoint_complete "$checkpoint" "$mode"
}

train_one control 0 &
control_pid=$!
train_one student 1 &
student_pid=$!
wait "$control_pid"
wait "$student_pid"
touch "$STATE/train.complete"

exec 9>"$LOG_ROOT/.upr_lite_gpu0.lock"
flock 9
wait_for_gpu 0
export CUDA_VISIBLE_DEVICES=0
if [[ ! -e "$EVAL/.complete" ]]; then
  echo "[$(date -Is)] evaluate direct response pair"
  (
    cd "$SOURCE"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir "$MEMMAP" --test-year 2020 --climatology "$CLIM" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --models "flow_gt_control:$CONTROL,flow_dcae_student:$STUDENT" \
      --out-dir "$EVAL" --paper-tag direct_distill_v13_2020 \
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

(cd "$SOURCE" && "$PY" tools/eval/select_specialist_distillation.py \
  --control "$BASE_EVAL" \
  --candidate "flow_dcae_student:$EVAL/flow_dcae_student.json" \
  --aggregate-tolerance-pct 0.02 --held-tolerance-pct 0.02 \
  --out "$OUT_ROOT/selection.starting_base.json") || true
(cd "$SOURCE" && "$PY" tools/eval/select_specialist_distillation.py \
  --control "$EVAL/flow_gt_control.json" \
  --candidate "flow_dcae_student:$EVAL/flow_dcae_student.json" \
  --aggregate-tolerance-pct 0.02 --held-tolerance-pct 0.02 \
  --out "$OUT_ROOT/selection.matched_control.json") || true
if (cd "$SOURCE" && "$PY" tools/eval/select_distilled_champion.py \
  --screen "$OUT_ROOT/selection.starting_base.json" \
  --matched-screen "$OUT_ROOT/selection.matched_control.json" \
  --output "$OUT_ROOT/selection.strict.json"); then
  touch "$STATE/strict.pass"
  echo "[$(date -Is)] direct response student strict screen passed"
else
  touch "$STATE/strict.failed"
  echo "[$(date -Is)] direct response student strict screen failed"
  exit 2
fi
