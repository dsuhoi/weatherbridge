#!/usr/bin/env bash
# Stage 1: matched WeatherDCAE-FM/direct-control pilot before seed ensembling.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherdcae_fm_ensemble_v19_source}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
OUT_ROOT=${OUT_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherdcae_fm_ensemble_v19}
MEMMAP=${MEMMAP:-/tmp/wb2_0p5_cache}
DATA_ROOT=${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}
CLIM=${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}
CLIM_CACHE=${CLIM_CACHE:-/tmp/wti_climatology_cache_detailed_6h}
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}
BASE=$LOG_ROOT/exp_weatherdcae_14m_6h_sparse_gt_control_s202712_v17_bs4/last.ckpt
BASE_SHA256=95be470e92006aab43f02a0ddf9b0973453a0317027a0a901585ff1e025f73ec
SEED=${SEED:-202713}
FM_EXP=exp_weatherdcae_fm_14m_6h_s${SEED}_v19_bs4
CONTROL_EXP=exp_weatherdcae_fm_control_14m_6h_s${SEED}_v19_bs4
FM=$LOG_ROOT/$FM_EXP/last.ckpt
CONTROL=$LOG_ROOT/$CONTROL_EXP/last.ckpt
STATE=$OUT_ROOT/state
EVAL=$OUT_ROOT/pilot_2020_8dpm
LOG=$OUT_ROOT/pipeline.log
PRECHECK_ONLY=${PRECHECK_ONLY:-0}
MIN_FREE_MIB=${MIN_FREE_MIB:-70000}
MAX_UTIL=${MAX_UTIL:-5}
POLL_SECONDS=${POLL_SECONDS:-30}

mkdir -p "$STATE" "$EVAL"
exec >>"$LOG" 2>&1
exec 8>"$STATE/launch.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] WeatherDCAE-FM pilot already active"
  exit 0
fi

(cd "$SOURCE" && sha256sum --quiet -c SOURCE_SHA256SUMS)
test "$(sha256sum "$BASE" | awk '{print $1}')" = "$BASE_SHA256"
for path in \
  "$DATA_ROOT/static_features_0p5.pt" \
  "$DATA_ROOT/json_stats_0p5.nc" \
  "$DATA_ROOT/surface_stats_0p5.json"; do
  test -s "$path"
done
test -d "$MEMMAP"
test -d "$CLIM"
test -d "$CLIM_CACHE"

if [[ "$PRECHECK_ONLY" -eq 1 ]]; then
  touch "$STATE/precheck.pass"
  echo "[$(date -Is)] WeatherDCAE-FM pilot precheck complete"
  exit 0
fi

checkpoint_complete() {
  local checkpoint=$1
  local arch=$2
  [[ -s "$checkpoint" ]] || return 1
  "$PY" - "$checkpoint" "$arch" "$BASE" <<'PY'
import hashlib
import sys
from pathlib import Path

import torch

checkpoint_path, expected_arch, base_path = sys.argv[1:]
candidate = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
hparams = candidate.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
state = candidate.get("state_dict", {})
assert hparams.get("arch") == expected_arch
assert sum(v.numel() for k, v in state.items() if k.startswith("net.")) == 14465140
assert int(candidate.get("global_step", -1)) >= 410
assert protocol.get("train_years") == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol.get("val_years") == [2020]
assert protocol.get("train_tau_hours") == [1, 3, 5]
assert protocol.get("eval_tau_hours") == [1, 2, 3, 4, 5]
assert protocol.get("global_effective_batch_size") == 16
assert protocol.get("optimizer_weight_decay") == 1.0e-4
assert protocol.get("trainable_scope") == "all"
assert protocol["distillation"]["enabled"] is False
lineage = hparams["initialization_lineage"]
assert lineage["checkpoint_sha256"] == hashlib.sha256(Path(base_path).read_bytes()).hexdigest()
compatibility = lineage["compatibility"]
assert compatibility["source_arch"] == "dcae_14m"
assert compatibility["target_arch"] == expected_arch
assert len(compatibility["zero_initialized_missing_keys"]) == 6
code = hparams["training_code_sha256"]
assert {"dcae.py", "dcae_adaln_model.py", "dcae_flow_matching_model.py"} <= set(code)
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
  local arch=$1
  local exp=$2
  local checkpoint=$3
  local gpu=$4
  if checkpoint_complete "$checkpoint" "$arch"; then
    return
  fi
  (
    exec 9>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
    flock 9
    wait_for_gpu "$gpu"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    echo "[$(date -Is)] train arch=$arch seed=$SEED gpu=$gpu exp=$exp"
    cd "$SOURCE"
    "$PY" -u tools/train/train_capacity_matched_6h.py \
      --arch "$arch" --exp_name "$exp" --log_root "$LOG_ROOT" --gpus 0 \
      --bs 4 --val_bs 2 --accumulate 4 --workers 4 --val_workers 2 \
      --release_memmap_pages --precision bf16-mixed --seed "$SEED" \
      --years 2014 2015 2016 2017 2018 2019 --val_years 2020 \
      --max_epochs 1 --lr 1e-5 --warmup_steps 100 --window_hours 6 \
      --train_tau_subset 1 3 5 --eval_tau 1 2 3 4 5 \
      --samples_per_date_train 4 --samples_per_date_val 2 \
      --lambda_hf_override 0 --lambda_spec_override 0 \
      --lambda_band_override 0 --lambda_sht_override 0 \
      --spectral_mask_profile all --loss_profile uniform \
      --trainable_scope all --anchor_swap_probability 0 \
      --train_batches_per_epoch 1640 --ckpt_every_n_epochs 1 \
      --memmap_dir "$MEMMAP" \
      --static_path "$DATA_ROOT/static_features_0p5.pt" \
      --stats_path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface_stats_path "$DATA_ROOT/surface_stats_0p5.json" \
      --init_weights_path "$BASE"
  )
  checkpoint_complete "$checkpoint" "$arch"
}

train_one dcae_fm_14m "$FM_EXP" "$FM" 0 &
fm_pid=$!
train_one dcae_fm_control_14m "$CONTROL_EXP" "$CONTROL" 1 &
control_pid=$!
wait "$fm_pid"
wait "$control_pid"
touch "$STATE/train.complete"

exec 9>"$LOG_ROOT/.upr_lite_gpu0.lock"
flock 9
wait_for_gpu 0
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
if [[ ! -e "$EVAL/.complete" ]]; then
  echo "[$(date -Is)] evaluate WeatherDCAE-FM pilot pair"
  cd "$SOURCE"
  "$PY" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir "$MEMMAP" --test-year 2020 --climatology "$CLIM" \
    --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
    --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
    --static-path "$DATA_ROOT/static_features_0p5.pt" \
    --models "weatherdcae_fm:$FM,matched_direct_control:$CONTROL" \
    --out-dir "$EVAL" --paper-tag weatherdcae_fm_v19_pilot_2020 \
    --batch-size 4 --num-workers 2 --samples-per-date 2 \
    --eval-days-per-month 8 --proper-rmse --save-window-metrics \
    --save-physical-metrics --save-temporal-metrics \
    --max-tau-hours 6 --eval-hours 1,2,3,4,5 \
    --seen-tau 1,3,5 --unseen-tau 2,4 --device cuda:0 \
    --lazy-climatology --climatology-cache-dir "$CLIM_CACHE" \
    --keep-n-channels 24
  touch "$EVAL/.complete"
fi

set +e
"$PY" "$SOURCE/tools/eval/select_weatherdcae_fm_pilot.py" \
  --candidate "$EVAL/weatherdcae_fm.json" \
  --control "$EVAL/matched_direct_control.json" \
  --output "$OUT_ROOT/pilot_selection.json"
selection_status=$?
set -e
if [[ "$selection_status" -eq 0 ]]; then
  touch "$STATE/pilot.pass"
  echo "[$(date -Is)] WeatherDCAE-FM pilot passed; seed ensemble is eligible"
else
  touch "$STATE/pilot.failed"
  echo "[$(date -Is)] WeatherDCAE-FM pilot rejected; do not spend seed budget"
  exit 2
fi
