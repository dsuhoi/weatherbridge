#!/usr/bin/env bash
# Train matched sparse-time and all-hours Universal-Pareto-Refine models.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
SOURCE="${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_universal_pareto_refine_v1_source}"
MEMMAP="${MEMMAP:-/tmp/wb2_0p5_cache}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
STATE_DIR="${STATE_DIR:-$LOG_ROOT/universal_pareto_refine_pair_v1.state}"
QUEUE_LOG="${QUEUE_LOG:-$LOG_ROOT/universal_pareto_refine_pair_v1.queue.log}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MIN_FREE_MIB="${MIN_FREE_MIB:-70000}"
MAX_UTIL="${MAX_UTIL:-5}"
ARCH="flow_universal_pareto_refine"
PARAMETERS=13659644
SEED=202710
TOTAL_STEPS=13136

mkdir -p "$LOG_ROOT" "$STATE_DIR"
exec 9>"$STATE_DIR/launch.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] Universal-Pareto-Refine queue already active"
  exit 0
fi
exec > >(tee -a "$QUEUE_LOG") 2>&1

(
  cd "$SOURCE"
  sha256sum --quiet -c SOURCE_SHA256SUMS
)
for year in 2014 2015 2016 2017 2018 2019 2020; do
  test -s "$MEMMAP/wb2_${year}.json"
done
test -s "$DATA_ROOT/static_features_0p5.pt"
test -s "$DATA_ROOT/json_stats_0p5.nc"
test -s "$DATA_ROOT/surface_stats_0p5.json"

checkpoint_complete() {
  local checkpoint="$1"
  local protocol="$2"
  [[ -s "$checkpoint" ]] || return 1
  "$PY" - "$checkpoint" "$protocol" "$PARAMETERS" "$TOTAL_STEPS" <<'PY'
import sys
import torch

path, expected_protocol, expected_parameters, expected_steps = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
hparams = checkpoint.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
state = checkpoint.get("state_dict", {})
parameters = sum(
    value.numel()
    for name, value in state.items()
    if name.startswith("net.")
)
expected_taus = [1, 3, 5] if expected_protocol == "sparse" else [1, 2, 3, 4, 5]
assert hparams.get("arch") == "flow_universal_pareto_refine"
assert parameters == int(expected_parameters)
assert int(checkpoint.get("global_step", -1)) >= int(expected_steps)
assert int(checkpoint.get("epoch", -1)) >= 7
assert protocol.get("train_years") == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol.get("val_years") == [2020]
assert protocol.get("train_tau_hours") == expected_taus
assert protocol.get("eval_tau_hours") == [1, 2, 3, 4, 5]
assert protocol.get("global_effective_batch_size") == 16
assert protocol.get("optimizer_steps_per_epoch") == 1642
assert protocol.get("loss_profile") == "relative_group_pareto"
assert protocol.get("group_balance_ema_decay") == 0.99
assert protocol.get("group_balance_cvar_fields") == 6
assert protocol.get("lambda_hf") == 0.05
assert protocol.get("lambda_spec") == 0.0
assert protocol.get("lambda_band") == 0.02
assert protocol.get("anchor_swap_probability") == 0.0
distillation = protocol.get("distillation", {})
assert distillation.get("enabled") is False
PY
}

gpu_idle() {
  local gpu="$1" free util
  free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  util="$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
  [[ "$free" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]
}

wait_for_gpu() {
  local gpu="$1"
  until gpu_idle "$gpu"; do
    echo "[$(date -Is)] waiting for idle gpu=$gpu"
    sleep "$POLL_SECONDS"
  done
  sleep 15
  gpu_idle "$gpu"
}

run_arm() {
  local gpu="$1" protocol="$2" experiment train_taus
  if [[ "$protocol" == "sparse" ]]; then
    experiment="exp_flow_universal_pareto_refine_14m_6h_sparse_s${SEED}_v1_bs4"
    train_taus=(1 3 5)
  else
    experiment="exp_flow_universal_pareto_refine_14m_6h_alltau_s${SEED}_v1_bs4"
    train_taus=(1 2 3 4 5)
  fi
  local output="$LOG_ROOT/$experiment"
  local checkpoint="$output/last.ckpt"
  local run_log="$LOG_ROOT/$experiment.log"
  local complete="$STATE_DIR/$protocol.complete"

  exec 7>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
  flock 7
  wait_for_gpu "$gpu"
  if checkpoint_complete "$checkpoint" "$protocol"; then
    touch "$complete"
    echo "[$(date -Is)] validated existing protocol=$protocol checkpoint=$checkpoint"
    return 0
  fi

  local resume=()
  if [[ -s "$checkpoint" ]]; then
    resume=(--ckpt_path "$checkpoint")
  fi
  echo "[$(date -Is)] launch protocol=$protocol gpu=$gpu experiment=$experiment"
  (
    cd "$SOURCE"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/train/train_capacity_matched_6h.py \
      --arch "$ARCH" \
      --exp_name "$experiment" \
      --log_root "$LOG_ROOT" \
      --gpus 0 \
      --bs 4 \
      --val_bs 2 \
      --accumulate 4 \
      --workers 4 \
      --val_workers 2 \
      --release_memmap_pages \
      --precision bf16-mixed \
      --seed "$SEED" \
      --years 2014 2015 2016 2017 2018 2019 \
      --val_years 2020 \
      --max_epochs 8 \
      --lr 1e-4 \
      --warmup_steps 500 \
      --window_hours 6 \
      --train_tau_subset "${train_taus[@]}" \
      --eval_tau 1 2 3 4 5 \
      --samples_per_date_train 4 \
      --samples_per_date_val 2 \
      --lambda_hf_override 0.05 \
      --lambda_spec_override 0 \
      --lambda_band_override 0.02 \
      --spectral_mask_profile all \
      --loss_profile relative_group_pareto \
      --group_balance_ema_decay 0.99 \
      --group_balance_cvar_fields 6 \
      --trainable_scope all \
      --anchor_swap_probability 0 \
      --train_batches_per_epoch 6568 \
      --ckpt_every_n_epochs 2 \
      --memmap_dir "$MEMMAP" \
      --static_path "$DATA_ROOT/static_features_0p5.pt" \
      --stats_path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface_stats_path "$DATA_ROOT/surface_stats_0p5.json" \
      "${resume[@]}"
  ) >>"$run_log" 2>&1
  checkpoint_complete "$checkpoint" "$protocol"
  touch "$complete"
  echo "[$(date -Is)] complete protocol=$protocol checkpoint=$checkpoint"
}

run_arm 0 sparse &
sparse_pid=$!
run_arm 1 alltau &
alltau_pid=$!
wait "$sparse_pid"
wait "$alltau_pid"
touch "$STATE_DIR/pair.complete"
echo "[$(date -Is)] Universal-Pareto-Refine pair complete"
