#!/usr/bin/env bash
# Validation-only field-head merge screen for the complementary Flow student.
set -euo pipefail

PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_field_merge_v14_source}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
OUT_ROOT=${OUT_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_field_merge_v14}
MEMMAP=${MEMMAP:-/tmp/wb2_0p5_cache}
DATA_ROOT=${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}
CLIM=${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}
POLL_SECONDS=${POLL_SECONDS:-30}

CONTROL=$LOG_ROOT/exp_flow_pp3_14m_6h_direct_gt_control_s202711_v13_bs4/last.ckpt
DISTILLED=$LOG_ROOT/exp_flow_pp3_14m_6h_direct_dcae_student_s202711_v13_bs4/last.ckpt
STARTING_EVAL=/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_qhead_v4/eval_2020_8dpm/flow_teacher.json
MATCHED_EVAL=/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_direct_response_v13/eval_2020_8dpm/flow_gt_control.json
CKPT_DIR=$OUT_ROOT/checkpoints
EVAL=$OUT_ROOT/eval_2020_8dpm
STATE=$OUT_ROOT/state
LOG=$OUT_ROOT/pipeline.log

mkdir -p "$CKPT_DIR" "$EVAL" "$STATE"
exec >>"$LOG" 2>&1
exec 8>"$STATE/launch.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] Flow field merge screen already active"
  exit 0
fi

(cd "$SOURCE" && sha256sum --quiet -c SOURCE_SHA256SUMS)
for path in "$CONTROL" "$DISTILLED" "$STARTING_EVAL" "$MATCHED_EVAL"; do
  test -s "$path"
done

declare -A ALPHAS=(
  [merge_a025]=0.25
  [merge_a050]=0.50
  [merge_a100]=1.00
)
for name in merge_a025 merge_a050 merge_a100; do
  output=$CKPT_DIR/$name.ckpt
  if [[ ! -s "$output" ]]; then
    (cd "$SOURCE" && "$PY" tools/train/merge_flow_field_heads.py \
      --control "$CONTROL" --distilled "$DISTILLED" \
      --alpha "${ALPHAS[$name]}" --output "$output")
  fi
done

"$PY" - "$CONTROL" "$DISTILLED" "$CKPT_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

import torch

control_path = Path(sys.argv[1])
distilled_path = Path(sys.argv[2])
checkpoint_dir = Path(sys.argv[3])
control = torch.load(control_path, map_location="cpu", weights_only=False)
reference = control["state_dict"]
expected = {"merge_a025": 0.25, "merge_a050": 0.5, "merge_a100": 1.0}
for name, alpha in expected.items():
    merged = torch.load(checkpoint_dir / f"{name}.ckpt", map_location="cpu", weights_only=False)
    metadata = merged["checkpoint_field_merge"]
    assert metadata["alpha"] == alpha
    assert metadata["fields"] == ["Z1000", "Z925", "Z850", "Z700", "mslp"]
    assert metadata["control"]["sha256"] == hashlib.sha256(control_path.read_bytes()).hexdigest()
    assert metadata["distilled"]["sha256"] == hashlib.sha256(distilled_path.read_bytes()).hexdigest()
    assert merged["optimizer_states"] == []
    assert merged["lr_schedulers"] == []
    selected = set(metadata["row_keys"]) | set(metadata["hydro_keys"])
    for key, value in merged["state_dict"].items():
        if key not in selected:
            assert torch.equal(value, reference[key]), key
PY
touch "$STATE/checkpoints.complete"

exec 9>"$LOG_ROOT/.upr_lite_gpu0.lock"
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')
  if [[ "$free" -ge 70000 && "$util" -le 5 ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done
export CUDA_VISIBLE_DEVICES=0
if [[ ! -e "$EVAL/.complete" ]]; then
  echo "[$(date -Is)] evaluate Flow field merges"
  (
    cd "$SOURCE"
    export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
    "$PY" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir "$MEMMAP" --test-year 2020 --climatology "$CLIM" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --models "merge_a025:$CKPT_DIR/merge_a025.ckpt,merge_a050:$CKPT_DIR/merge_a050.ckpt,merge_a100:$CKPT_DIR/merge_a100.ckpt" \
      --out-dir "$EVAL" --paper-tag field_merge_v14_2020 \
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

if (cd "$SOURCE" && "$PY" tools/eval/select_field_merge_champion.py \
  --starting-control "$STARTING_EVAL" --matched-control "$MATCHED_EVAL" \
  --candidate "merge_a025:$EVAL/merge_a025.json" \
  --candidate "merge_a050:$EVAL/merge_a050.json" \
  --candidate "merge_a100:$EVAL/merge_a100.json" \
  --output "$OUT_ROOT/selection.strict.json"); then
  touch "$STATE/strict.pass"
  echo "[$(date -Is)] Flow field merge strict screen passed"
else
  touch "$STATE/strict.failed"
  echo "[$(date -Is)] Flow field merge strict screen failed"
  exit 2
fi
