#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
SNAPSHOT="${SNAPSHOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_universal_refine_v1_source}"
MEMMAP="${MEMMAP:-/tmp/wb2_0p5_cache}"
STATIC="${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}"
STATS="${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}"
SURFACE_STATS="${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"

ARCH="flow_universal_latent_refine"
EXP="exp_flow_universal_latent_refine_14m_12h_2017_19_s202707_v1_bs4"
OUT="$LOG_ROOT/$EXP"
CHECKPOINT="$OUT/last.ckpt"
RUN_LOG="$LOG_ROOT/$EXP.log"
QUEUE_LOG="$LOG_ROOT/$EXP.queue.log"
PARAMETERS=13651208
OPTIMIZER_STEPS_PER_EPOCH=1093
TRAIN_BATCHES_PER_EPOCH=4372
TOTAL_STEPS=10930

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$QUEUE_LOG") 2>&1

checkpoint_complete() {
  [[ -s "$CHECKPOINT" ]] || return 1
  "$PY" - "$CHECKPOINT" <<'PY'
import sys
import torch

path = sys.argv[1]
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
assert parameters == 13_651_208
assert int(checkpoint.get("global_step", -1)) >= 10_930
assert int(checkpoint.get("epoch", -1)) >= 9
assert float(hparams.get("delta_t", -1)) == 12.0
assert protocol.get("train_years") == [2017, 2018, 2019]
assert protocol.get("val_years") == [2020]
assert protocol.get("train_tau_hours") == [1, 2, 3, 5, 7, 9, 10, 11]
assert protocol.get("eval_tau_hours") == list(range(1, 12))
assert protocol.get("global_effective_batch_size") == 16
assert protocol.get("optimizer_steps_per_epoch") == 1093
assert protocol.get("samples_per_date_train") == 2
assert protocol.get("samples_per_date_val") == 2
assert protocol.get("lambda_hf") == 0.05
assert protocol.get("lambda_spec") == 0.0
assert protocol.get("lambda_band") == 0.02
assert protocol.get("anchor_swap_probability") == 0.0
assert not any("teacher" in name or "distill" in name for name in state)
PY
}

if checkpoint_complete; then
  echo "[$(date -Is)] already complete checkpoint=$CHECKPOINT"
  exit 0
fi

(
  cd "$SNAPSHOT"
  sha256sum --quiet -c SOURCE_SHA256SUMS
)
for year in 2017 2018 2019 2020; do
  test -s "$MEMMAP/wb2_${year}.json"
done
test -s "$STATIC"
test -s "$STATS"
test -s "$SURFACE_STATS"

exec 9>"$LOG_ROOT/.universal_refine_12h_gpu${GPU}.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] GPU lock is already held: gpu=$GPU"
  exit 0
fi

resume=()
if [[ -s "$CHECKPOINT" ]]; then
  resume=(--ckpt_path "$CHECKPOINT")
fi

echo "[$(date -Is)] launch exp=$EXP gpu=$GPU steps=$TOTAL_STEPS"
(
  cd "$SNAPSHOT"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export PYTHONPATH="$EXTRA_PYTHONPATH:$SNAPSHOT:${PYTHONPATH:-}"
  "$PY" -u tools/train/train_capacity_matched_6h.py \
    --arch "$ARCH" \
    --exp_name "$EXP" \
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
    --years 2017 2018 2019 \
    --val_years 2020 \
    --max_epochs 10 \
    --lr 1e-4 \
    --warmup_steps 500 \
    --window_hours 12 \
    --train_tau_subset 1 2 3 5 7 9 10 11 \
    --eval_tau 1 2 3 4 5 6 7 8 9 10 11 \
    --samples_per_date_train 2 \
    --samples_per_date_val 2 \
    --lambda_hf_override 0.05 \
    --lambda_spec_override 0 \
    --lambda_band_override 0.02 \
    --spectral_mask_profile all \
    --loss_profile uniform \
    --trainable_scope all \
    --anchor_swap_probability 0 \
    --train_batches_per_epoch "$TRAIN_BATCHES_PER_EPOCH" \
    --ckpt_every_n_epochs 2 \
    --memmap_dir "$MEMMAP" \
    --static_path "$STATIC" \
    --stats_path "$STATS" \
    --surface_stats_path "$SURFACE_STATS" \
    "${resume[@]}"
) >>"$RUN_LOG" 2>&1

checkpoint_complete
echo "[$(date -Is)] complete checkpoint=$CHECKPOINT"
