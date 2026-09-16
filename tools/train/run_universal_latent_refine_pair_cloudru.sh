#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
SNAPSHOT="${SNAPSHOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_universal_refine_v1_source}"
MEMMAP="${MEMMAP:-/tmp/wb2_0p5_cache}"
STATIC="${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}"
STATS="${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}"
SURFACE_STATS="${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
QUEUE_LOG="$LOG_ROOT/universal_refine_pair_6h_v1.queue.log"
ARCH="flow_universal_latent_refine"
PARAMETERS=13651208

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$QUEUE_LOG") 2>&1

if [[ ! -d "$SNAPSHOT" ]]; then
  mkdir -p "$SNAPSHOT"
  cp -a weather_time_interp tools tests "$SNAPSHOT/"
  (
    cd "$SNAPSHOT"
    find weather_time_interp tools tests -type f -name '*.py' -print0 \
      | sort -z \
      | xargs -0 sha256sum > SOURCE_SHA256SUMS
  )
fi
(
  cd "$SNAPSHOT"
  sha256sum --quiet -c SOURCE_SHA256SUMS
)

checkpoint_complete() {
  local checkpoint="$1"
  local expected_swap="$2"
  [[ -s "$checkpoint" ]] || return 1
  "$PY" - "$checkpoint" "$expected_swap" "$PARAMETERS" <<'PY'
import sys
import torch

path, expected_swap, expected_parameters = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
hparams = checkpoint.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
state = checkpoint.get("state_dict", {})
parameters = sum(
    value.numel()
    for name, value in state.items()
    if name.startswith("net.")
)
assert hparams.get("arch") == "flow_universal_latent_refine"
assert parameters == expected_parameters
assert int(checkpoint.get("global_step", -1)) >= 13136
assert int(checkpoint.get("epoch", -1)) >= 7
assert protocol.get("train_years") == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol.get("val_years") == [2020]
assert protocol.get("train_tau_hours") == [1, 3, 5]
assert protocol.get("eval_tau_hours") == [1, 2, 3, 4, 5]
assert protocol.get("global_effective_batch_size") == 16
assert protocol.get("optimizer_steps_per_epoch") == 1642
assert protocol.get("lambda_hf") == 0.05
assert protocol.get("lambda_spec") == 0.0
assert protocol.get("lambda_band") == 0.02
assert protocol.get("anchor_swap_probability") == expected_swap
assert not any("teacher" in name or "distill" in name for name in state)
PY
}

run_candidate() {
  local gpu="$1"
  local swap_probability="$2"
  local experiment="$3"
  local output="$LOG_ROOT/$experiment"
  local checkpoint="$output/last.ckpt"
  local run_log="$LOG_ROOT/$experiment.log"
  local complete="$LOG_ROOT/$experiment.complete"

  exec 9>"$LOG_ROOT/.universal_refine_gpu${gpu}.lock"
  flock 9
  if checkpoint_complete "$checkpoint" "$swap_probability"; then
    touch "$complete"
    echo "[$(date -Is)] already complete experiment=$experiment"
    return
  fi

  local resume=()
  if [[ -s "$checkpoint" ]]; then
    resume=(--ckpt_path "$checkpoint")
  fi
  echo "[$(date -Is)] launch experiment=$experiment gpu=$gpu swap=$swap_probability"
  (
    cd "$SNAPSHOT"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SNAPSHOT:${PYTHONPATH:-}"
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
      --seed 202707 \
      --years 2014 2015 2016 2017 2018 2019 \
      --val_years 2020 \
      --max_epochs 8 \
      --lr 1e-4 \
      --warmup_steps 500 \
      --window_hours 6 \
      --train_tau_subset 1 3 5 \
      --eval_tau 1 2 3 4 5 \
      --samples_per_date_train 4 \
      --samples_per_date_val 2 \
      --lambda_hf_override 0.05 \
      --lambda_spec_override 0 \
      --lambda_band_override 0.02 \
      --spectral_mask_profile all \
      --loss_profile uniform \
      --trainable_scope all \
      --anchor_swap_probability "$swap_probability" \
      --train_batches_per_epoch 6568 \
      --ckpt_every_n_epochs 2 \
      --memmap_dir "$MEMMAP" \
      --static_path "$STATIC" \
      --stats_path "$STATS" \
      --surface_stats_path "$SURFACE_STATS" \
      "${resume[@]}"
  ) >>"$run_log" 2>&1
  checkpoint_complete "$checkpoint" "$swap_probability"
  touch "$complete"
  echo "[$(date -Is)] complete experiment=$experiment"
}

run_candidate \
  0 0.0 exp_flow_universal_latent_refine_14m_6h_s202707_v1_bs4 &
pid_standard=$!
run_candidate \
  1 0.5 exp_flow_universal_latent_refine_swap50_14m_6h_s202707_v1_bs4 &
pid_swap=$!

wait "$pid_standard"
wait "$pid_swap"
echo "[$(date -Is)] Universal-Latent-Refine pair complete"
