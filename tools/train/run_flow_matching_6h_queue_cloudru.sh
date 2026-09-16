#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU="${GPU:-0}"
PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
SNAPSHOT="${SNAPSHOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_universal_latent_v1_source}"
MEMMAP="${MEMMAP:-/tmp/wb2_0p5_cache}"
STATIC="${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}"
STATS="${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}"
SURFACE_STATS="${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
QUEUE_LOG="$LOG_ROOT/universal_latent_vs_fm_6h_v1.queue.log"

declare -A PARAMS=(
  [flow_universal_latent]=12100232
  [flow_universal_fm]=12182152
)
declare -A EXPERIMENTS=(
  [flow_universal_latent]=exp_flow_universal_latent_12m_6h_s202707_v1_bs4
  [flow_universal_fm]=exp_flow_universal_fm_12m_6h_s202707_v1_bs4
)
CANDIDATES=(flow_universal_latent flow_universal_fm)

mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/.universal_latent_6h_gpu${GPU}.lock"
flock 9
exec > >(tee -a "$QUEUE_LOG") 2>&1

if [[ ! -d "$SNAPSHOT" ]]; then
  mkdir -p "$SNAPSHOT"
  cp -a weather_time_interp tools "$SNAPSHOT/"
  (
    cd "$SNAPSHOT"
    find weather_time_interp tools -type f -name '*.py' -print0 \
      | sort -z \
      | xargs -0 sha256sum > SOURCE_SHA256SUMS
  )
fi
(
  cd "$SNAPSHOT"
  sha256sum --quiet -c SOURCE_SHA256SUMS
)

export PYTHONPATH="$EXTRA_PYTHONPATH:$SNAPSHOT:${PYTHONPATH:-}"

checkpoint_complete() {
  local checkpoint="$1"
  local arch="$2"
  local expected_parameters="$3"
  [[ -s "$checkpoint" ]] || return 1
  "$PY" - "$checkpoint" "$arch" "$expected_parameters" <<'PY'
import sys
import torch

path, arch, expected_parameters = sys.argv[1], sys.argv[2], int(sys.argv[3])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
hparams = checkpoint.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
state = checkpoint.get("state_dict", {})
parameters = sum(
    value.numel()
    for name, value in state.items()
    if name.startswith("net.")
)
assert hparams.get("arch") == arch
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
assert not any("teacher" in name or "distill" in name for name in state)
PY
}

for arch in "${CANDIDATES[@]}"; do
  experiment="${EXPERIMENTS[$arch]}"
  output="$LOG_ROOT/$experiment"
  checkpoint="$output/last.ckpt"
  run_log="$LOG_ROOT/$experiment.log"
  complete="$LOG_ROOT/$experiment.complete"
  if checkpoint_complete "$checkpoint" "$arch" "${PARAMS[$arch]}"; then
    touch "$complete"
    echo "[$(date -Is)] already complete arch=$arch"
    continue
  fi

  resume=()
  if [[ -s "$checkpoint" ]]; then
    resume=(--ckpt_path "$checkpoint")
  fi
  echo "[$(date -Is)] launch arch=$arch gpu=$GPU"
  (
    cd "$SNAPSHOT"
    export CUDA_VISIBLE_DEVICES="$GPU"
    "$PY" -u tools/train/train_capacity_matched_6h.py \
      --arch "$arch" \
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
      --train_batches_per_epoch 6568 \
      --ckpt_every_n_epochs 2 \
      --memmap_dir "$MEMMAP" \
      --static_path "$STATIC" \
      --stats_path "$STATS" \
      --surface_stats_path "$SURFACE_STATS" \
      "${resume[@]}"
  ) >>"$run_log" 2>&1
  checkpoint_complete "$checkpoint" "$arch" "${PARAMS[$arch]}"
  touch "$complete"
  echo "[$(date -Is)] complete arch=$arch"
done

echo "[$(date -Is)] universal latent and Flow-Matching candidates complete"
