#!/usr/bin/env bash
# Stage 2: train retained seeds, build mean ensembles, and run matched screening.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherdcae_fm_ensemble_v19_stage2_source}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
OUT_ROOT=${OUT_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherdcae_fm_ensemble_v19}
MEMMAP=${MEMMAP:-/tmp/wb2_0p5_cache}
DATA_ROOT=${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}
CLIM=${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}
CLIM_CACHE=${CLIM_CACHE:-/tmp/wti_climatology_cache_detailed_6h}
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}
BASE=$LOG_ROOT/exp_weatherdcae_14m_6h_sparse_gt_control_s202712_v17_bs4/last.ckpt
FLOW_REF=/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_shared_merge_v15/checkpoints/merge_a050.ckpt
BASE_SHA256=95be470e92006aab43f02a0ddf9b0973453a0317027a0a901585ff1e025f73ec
STATE=$OUT_ROOT/state
LOG=$OUT_ROOT/stage2.log
EVAL=$OUT_ROOT/ensemble_2020_8dpm
SEEDS=(202714 202715)
ALL_SEEDS=(202713 202714 202715)
MIN_FREE_MIB=${MIN_FREE_MIB:-70000}
MAX_UTIL=${MAX_UTIL:-5}
POLL_SECONDS=${POLL_SECONDS:-60}

mkdir -p "$STATE" "$EVAL"
exec >>"$LOG" 2>&1
exec 8>"$STATE/stage2.launch.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] WeatherDCAE-FM stage 2 already active"
  exit 0
fi

(cd "$SOURCE" && sha256sum --quiet -c SOURCE_SHA256SUMS)
test "$(sha256sum "$BASE" | awk '{print $1}')" = "$BASE_SHA256"
test -s "$FLOW_REF"
while [[ ! -e "$STATE/pilot.pass" && ! -e "$STATE/pilot.failed" ]]; do
  echo "[$(date -Is)] wait for WeatherDCAE-FM pilot gate"
  sleep "$POLL_SECONDS"
done
if [[ -e "$STATE/pilot.failed" ]]; then
  echo "[$(date -Is)] pilot failed; stage 2 pruned"
  touch "$STATE/stage2.pruned"
  exit 2
fi
"$PY" - "$OUT_ROOT/pilot_selection.json" <<'PY'
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text())
assert selection["selection_role"] == "era5_2020_screen_only"
assert selection["continue_seed_ensemble"] is True
PY

checkpoint_complete() {
  local checkpoint=$1
  local arch=$2
  local seed=$3
  [[ -s "$checkpoint" ]] || return 1
  "$PY" - "$checkpoint" "$arch" "$seed" "$BASE" <<'PY'
import hashlib
import sys
from pathlib import Path

import torch

checkpoint_path, expected_arch, expected_seed, base_path = sys.argv[1:]
candidate = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
hparams = candidate.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
state = candidate.get("state_dict", {})
assert hparams.get("arch") == expected_arch
assert int(hparams.get("training_seed", -1)) == int(expected_seed)
assert sum(v.numel() for k, v in state.items() if k.startswith("net.")) == 14465140
assert int(candidate.get("global_step", -1)) >= 410
assert protocol.get("train_years") == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol.get("val_years") == [2020]
assert protocol.get("train_tau_hours") == [1, 3, 5]
assert protocol.get("eval_tau_hours") == [1, 2, 3, 4, 5]
assert protocol.get("global_effective_batch_size") == 16
lineage = hparams["initialization_lineage"]
assert lineage["checkpoint_sha256"] == hashlib.sha256(Path(base_path).read_bytes()).hexdigest()
assert lineage["compatibility"]["target_arch"] == expected_arch
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
  local seed=$2
  local gpu=$3
  local role checkpoint exp
  role=fm
  if [[ "$arch" == dcae_fm_control_14m ]]; then
    role=fm_control
  fi
  exp=exp_weatherdcae_${role}_14m_6h_s${seed}_v19_bs4
  checkpoint=$LOG_ROOT/$exp/last.ckpt
  if checkpoint_complete "$checkpoint" "$arch" "$seed"; then
    return
  fi
  (
    exec 9>"$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
    flock 9
    wait_for_gpu "$gpu"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    echo "[$(date -Is)] stage2 train arch=$arch seed=$seed gpu=$gpu"
    cd "$SOURCE"
    "$PY" -u tools/train/train_capacity_matched_6h.py \
      --arch "$arch" --exp_name "$exp" --log_root "$LOG_ROOT" --gpus 0 \
      --bs 4 --val_bs 2 --accumulate 4 --workers 4 --val_workers 2 \
      --release_memmap_pages --precision bf16-mixed --seed "$seed" \
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
  checkpoint_complete "$checkpoint" "$arch" "$seed"
}

for seed in "${SEEDS[@]}"; do
  train_one dcae_fm_14m "$seed" 0 &
  fm_pid=$!
  train_one dcae_fm_control_14m "$seed" 1 &
  control_pid=$!
  wait "$fm_pid"
  wait "$control_pid"
done
touch "$STATE/stage2.train.complete"

fm_checkpoints=()
control_checkpoints=()
for seed in "${ALL_SEEDS[@]}"; do
  fm_checkpoints+=("$LOG_ROOT/exp_weatherdcae_fm_14m_6h_s${seed}_v19_bs4/last.ckpt")
  control_checkpoints+=("$LOG_ROOT/exp_weatherdcae_fm_control_14m_6h_s${seed}_v19_bs4/last.ckpt")
done
FM_MANIFEST=$OUT_ROOT/weatherdcae_fm_3seed.ensemble.json
CONTROL_MANIFEST=$OUT_ROOT/weatherdcae_direct_3seed.ensemble.json
"$PY" "$SOURCE/tools/eval/build_capmatched_ensemble_manifest.py" \
  --checkpoints "${fm_checkpoints[@]}" --expected-arch dcae_fm_14m \
  --name WeatherDCAE-FM-3seed --output "$FM_MANIFEST"
"$PY" "$SOURCE/tools/eval/build_capmatched_ensemble_manifest.py" \
  --checkpoints "${control_checkpoints[@]}" \
  --expected-arch dcae_fm_control_14m \
  --name WeatherDCAE-Direct-3seed --output "$CONTROL_MANIFEST"

exec 9>"$LOG_ROOT/.upr_lite_gpu0.lock"
flock 9
wait_for_gpu 0
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
if [[ ! -e "$EVAL/.complete" ]]; then
  cd "$SOURCE"
  "$PY" -u tools/eval/batch_eval_12h_memmap.py \
    --memmap-dir "$MEMMAP" --test-year 2020 --climatology "$CLIM" \
    --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
    --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
    --static-path "$DATA_ROOT/static_features_0p5.pt" \
    --models "weatherdcae_fm_3seed:$FM_MANIFEST,direct_3seed:$CONTROL_MANIFEST,starting_dcae:$BASE,flow_reference:$FLOW_REF" \
    --out-dir "$EVAL" --paper-tag weatherdcae_fm_v19_ensemble_2020 \
    --batch-size 2 --num-workers 2 --samples-per-date 2 \
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
  --candidate "$EVAL/weatherdcae_fm_3seed.json" \
  --control "$EVAL/direct_3seed.json" \
  --base "$EVAL/starting_dcae.json" \
  --reference "flow_reference:$EVAL/flow_reference.json" \
  --output "$OUT_ROOT/ensemble_selection_2020.json"
selection_status=$?
set -e
if [[ "$selection_status" -eq 0 ]]; then
  touch "$STATE/ensemble_2020.pass"
  echo "[$(date -Is)] WeatherDCAE-FM ensemble passed matched 2020 screen"
else
  touch "$STATE/ensemble_2020.failed"
  echo "[$(date -Is)] WeatherDCAE-FM ensemble failed matched 2020 screen"
  exit 2
fi
