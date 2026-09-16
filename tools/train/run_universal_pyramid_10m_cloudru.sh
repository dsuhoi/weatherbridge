#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
GPU="${GPU:-1}"
PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
RUNTIME="${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
SNAPSHOT="${SNAPSHOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_universal_pyramid_v1_source}"
MEMMAP="${MEMMAP:-/tmp/wb2_0p5_cache}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
ARCH="upr_universal_latent_q4_10m"
EXPERIMENT="exp_upr_universal_latent_q4_10m_6h_s202707_v1_bs8"
PARAMETERS=9500460
QUEUE_LOG="$LOG_ROOT/universal_pyramid_10m_6h_v1.queue.log"
OUTPUT="$LOG_ROOT/$EXPERIMENT"
CHECKPOINT="$OUTPUT/last.ckpt"

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$QUEUE_LOG") 2>&1

if [[ ! -d "$SNAPSHOT" ]]; then
  mkdir -p "$SNAPSHOT"
  cp -a "$RUNTIME/weather_time_interp" "$RUNTIME/tools" "$RUNTIME/tests" "$SNAPSHOT/"
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

if [[ -n "$WAIT_PID" ]]; then
  echo "[$(date -Is)] waiting for Refine queue pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
fi

# Reuse the active Refine lock so this challenger starts as soon as GPU1 is
# released, independently of the standard Refine run on GPU0.
exec 9>"$LOG_ROOT/.universal_refine_gpu${GPU}.lock"
flock 9

checkpoint_complete() {
  [[ -s "$CHECKPOINT" ]] || return 1
  "$PY" - "$CHECKPOINT" "$PARAMETERS" <<'PY'
import sys
import torch

path, expected_parameters = sys.argv[1], int(sys.argv[2])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
hparams = checkpoint.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
state = checkpoint.get("state_dict", {})
parameters = sum(
    value.numel()
    for name, value in state.items()
    if name.startswith("net.")
)
assert hparams.get("arch") == "upr_universal_latent_q4_10m"
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
assert protocol.get("anchor_swap_probability") == 0.0
assert not any("teacher" in name or "distill" in name for name in state)
PY
}

if checkpoint_complete; then
  echo "[$(date -Is)] already complete experiment=$EXPERIMENT"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$EXTRA_PYTHONPATH:$SNAPSHOT:${PYTHONPATH:-}"

echo "[$(date -Is)] full-grid forward/backward smoke"
(
  cd "$SNAPSHOT"
  "$PY" - <<'PY'
import torch

from tools.train.train_capacity_matched_6h import build_net

torch.manual_seed(202707)
model = build_net("upr_universal_latent_q4_10m", "")[0].cuda().train()
x0 = torch.randn(1, 24, 360, 720, device="cuda", dtype=torch.bfloat16)
x1 = torch.randn_like(x0)
static = torch.randn(1, 3, 360, 720, device="cuda", dtype=torch.bfloat16)
tau = torch.tensor([0.5], device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    prediction, _ = model(x0, x1, tau, static=static)
    loss = prediction.float().square().mean()
loss.backward()
gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
assert torch.isfinite(loss)
assert torch.isfinite(gradient)
print(f"smoke_loss={loss.item():.6f} grad_norm={gradient.item():.6f}")
PY
)

resume=()
if [[ -s "$CHECKPOINT" ]]; then
  resume=(--ckpt_path "$CHECKPOINT")
fi
echo "[$(date -Is)] launch experiment=$EXPERIMENT gpu=$GPU"
(
  cd "$SNAPSHOT"
  "$PY" -u tools/train/train_capacity_matched_6h.py \
    --arch "$ARCH" \
    --exp_name "$EXPERIMENT" \
    --log_root "$LOG_ROOT" \
    --gpus 0 \
    --bs 8 \
    --val_bs 4 \
    --accumulate 2 \
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
    --anchor_swap_probability 0 \
    --train_batches_per_epoch 3284 \
    --ckpt_every_n_epochs 2 \
    --memmap_dir "$MEMMAP" \
    --static_path "$DATA_ROOT/static_features_0p5.pt" \
    --stats_path "$DATA_ROOT/json_stats_0p5.nc" \
    --surface_stats_path "$DATA_ROOT/surface_stats_0p5.json" \
    "${resume[@]}"
) >>"$LOG_ROOT/$EXPERIMENT.log" 2>&1

checkpoint_complete
touch "$LOG_ROOT/$EXPERIMENT.complete"
echo "[$(date -Is)] Universal-Pyramid-9.5M complete"
