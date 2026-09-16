#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
SOURCE="${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_universal_content_refine_v1_source}"
MEMMAP="${MEMMAP:-/tmp/wb2_0p5_cache}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
GPU="${GPU:-1}"
EXPERIMENT="exp_flow_universal_content_refine_14m_6h_s202707_v1_bs4"
OUTPUT="$LOG_ROOT/$EXPERIMENT"
CHECKPOINT="$OUTPUT/last.ckpt"
LOG="$LOG_ROOT/$EXPERIMENT.log"
PARAMETERS=13651386

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG") 2>&1

"$PY" - "$SOURCE" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for line in (root / "SOURCE_SHA256SUMS").read_text().splitlines():
    expected, relative = line.split(None, 1)
    path = root / relative.strip()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"source SHA-256 mismatch: {relative}")
PY

checkpoint_complete() {
  [[ -s "$CHECKPOINT" ]] || return 1
  "$PY" - "$CHECKPOINT" "$PARAMETERS" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected_parameters = int(sys.argv[2])
hparams = checkpoint.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
state = checkpoint.get("state_dict", {})
parameters = sum(
    value.numel()
    for name, value in state.items()
    if name.startswith("net.")
)
assert hparams.get("arch") == "flow_universal_content_refine"
assert parameters == expected_parameters
assert int(checkpoint.get("global_step", -1)) >= 13136
assert int(checkpoint.get("epoch", -1)) >= 7
assert protocol.get("train_years") == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol.get("val_years") == [2020]
assert protocol.get("train_tau_hours") == [1, 3, 5]
assert protocol.get("eval_tau_hours") == [1, 2, 3, 4, 5]
assert protocol.get("global_effective_batch_size") == 16
assert protocol.get("optimizer_steps_per_epoch") == 1642
assert protocol.get("loss_profile") == "uniform"
assert protocol.get("lambda_hf") == 0.05
assert protocol.get("lambda_spec") == 0.0
assert protocol.get("lambda_band") == 0.02
assert protocol.get("anchor_swap_probability") == 0.0
assert not any("teacher" in name or "distill" in name for name in state)
PY
}

exec 9>"$LOG_ROOT/.universal_refine_gpu${GPU}.lock"
flock 9
if checkpoint_complete; then
  echo "[$(date -Is)] already complete experiment=$EXPERIMENT"
  exit 0
fi

resume=()
if [[ -s "$CHECKPOINT" ]]; then
  resume=(--ckpt_path "$CHECKPOINT")
fi

echo "[$(date -Is)] launch experiment=$EXPERIMENT gpu=$GPU"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
"$PY" -u "$SOURCE/tools/train/train_capacity_matched_6h.py" \
  --arch flow_universal_content_refine \
  --exp_name "$EXPERIMENT" \
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
  --anchor_swap_probability 0 \
  --train_batches_per_epoch 6568 \
  --ckpt_every_n_epochs 2 \
  --memmap_dir "$MEMMAP" \
  --static_path "$DATA_ROOT/static_features_0p5.pt" \
  --stats_path "$DATA_ROOT/json_stats_0p5.nc" \
  --surface_stats_path "$DATA_ROOT/surface_stats_0p5.json" \
  "${resume[@]}"

checkpoint_complete
touch "$OUTPUT.complete"
echo "[$(date -Is)] complete experiment=$EXPERIMENT"
