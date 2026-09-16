#!/usr/bin/env bash
# Full post-selection test pipeline for the frozen Flow distilled student.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_flow_distill_postrun_v16_source}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
OUT_ROOT=${OUT_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/distilled_flow_student_postrun_v16}
SCREEN_ROOT=${SCREEN_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_shared_merge_v15}
MEMMAP=${MEMMAP:-/tmp/wb2_0p5_cache}
DATA_ROOT=${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}
CLIM=${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}
CLIM_CACHE=${CLIM_CACHE:-/tmp/wti_climatology_cache_detailed_6h}
HRES=${HRES:-/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_hres_2021}
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}
POLL_SECONDS=${POLL_SECONDS:-120}
MIN_FREE_MIB=${MIN_FREE_MIB:-70000}
MAX_UTIL=${MAX_UTIL:-5}
PRECHECK_ONLY=${PRECHECK_ONLY:-0}

MATCHED_CONTROL=$LOG_ROOT/exp_flow_pp3_14m_6h_direct_gt_control_s202711_v13_bs4/last.ckpt
DIRECT_DISTILLED=$LOG_ROOT/exp_flow_pp3_14m_6h_direct_dcae_student_s202711_v13_bs4/last.ckpt
STARTING_FLOW=$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt
DCAE_TEACHER=$LOG_ROOT/exp_weatherdcae_14m_6h_direct_q_gt_control_s202710_v12_bs4/last.ckpt
SELECTION=$SCREEN_ROOT/selection.strict.json
FREEZE=$OUT_ROOT/selection.freeze.json
LOG=$OUT_ROOT/pipeline.log
STATE=$OUT_ROOT/state

mkdir -p "$OUT_ROOT" "$STATE"
test -d "$CLIM_CACHE"
if [[ ! -e "$OUT_ROOT/climatology_cache" ]]; then
  ln -s "$CLIM_CACHE" "$OUT_ROOT/climatology_cache"
fi
test "$(readlink -f "$OUT_ROOT/climatology_cache")" = "$(readlink -f "$CLIM_CACHE")"
exec >>"$LOG" 2>&1
exec 8>"$STATE/launch.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] distilled-student postrun already active"
  exit 0
fi

(cd "$SOURCE" && sha256sum --quiet -c SOURCE_SHA256SUMS)
export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"

while [[ ! -e "$SCREEN_ROOT/state/strict.pass" ]]; do
  echo "[$(date -Is)] wait Flow distillation strict screen"
  sleep "$POLL_SECONDS"
done
status=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$SELECTION")
if [[ "$status" != pass ]]; then
  echo "[$(date -Is)] distillation selection failed; final testing not admissible"
  printf '%s\n' "$status" >"$STATE/selection.failed"
  exit 2
fi
variant=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' "$SELECTION")
STUDENT=$SCREEN_ROOT/checkpoints/$variant.ckpt
for checkpoint in "$STUDENT" "$MATCHED_CONTROL" "$DIRECT_DISTILLED" \
  "$STARTING_FLOW" "$DCAE_TEACHER"; do
  test -s "$checkpoint"
done

$PY - "$STUDENT" "$MATCHED_CONTROL" "$DIRECT_DISTILLED" "$variant" <<'PY'
import hashlib
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
matched_control = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
direct = torch.load(sys.argv[3], map_location="cpu", weights_only=False)
hparams = checkpoint.get("hyper_parameters", {})
state = checkpoint.get("state_dict", {})
reference = matched_control.get("state_dict", {})
metadata = checkpoint.get("checkpoint_field_merge", {})
assert sys.argv[4] == "merge_a050"
assert hparams.get("arch") == "flow_pp3"
assert sum(v.numel() for k, v in state.items() if k.startswith("net.")) == 14260565
assert metadata.get("method") == "shared_trunk_target_head_linear_parameter_merge"
assert metadata.get("include_shared_trunk") is True
assert metadata.get("alpha") == 0.5
assert metadata.get("fields") == ["Z1000", "Z925", "Z850", "Z700", "mslp"]
assert metadata["control"]["sha256"] == hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
assert metadata["distilled"]["sha256"] == hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest()
assert not any("teacher" in key or "distill" in key for key in state)
for key in metadata["row_keys"]:
    for index in set(range(24)) - set(metadata["field_indices"]):
        assert torch.equal(state[key][index], reference[key][index]), (key, index)
assert any(
    not torch.equal(state[key], reference[key])
    for key in state
    if key.startswith("net.") and key not in set(metadata["row_keys"])
)
assert any(not torch.equal(state[key], direct["state_dict"][key]) for key in state)
print({"variant": sys.argv[4], "global_step": checkpoint["global_step"]})
PY

(cd "$SOURCE" && "$PY" tools/eval/freeze_distilled_student_selection.py \
  --selection "$SELECTION" \
  --model "distilled_student:$STUDENT" \
  --model "matched_control:$MATCHED_CONTROL" \
  --model "starting_flow:$STARTING_FLOW" \
  --model "dcae_teacher:$DCAE_TEACHER" \
  --output "$FREEZE")
if [[ "$PRECHECK_ONLY" -eq 1 ]]; then
  touch "$STATE/precheck.pass"
  echo "[$(date -Is)] Flow distilled-student precheck complete"
  exit 0
fi

wait_for_gpu() {
  local gpu="$1" free util
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    if [[ "$free" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

exec 6>"$LOG_ROOT/.upr_lite_gpu0.lock"
exec 7>"$LOG_ROOT/.upr_lite_gpu1.lock"
flock 6
wait_for_gpu 0
flock 7
wait_for_gpu 1
echo "[$(date -Is)] acquired physical GPUs=0,1 variant=$variant"

MODELS="distilled_student:$STUDENT,matched_control:$MATCHED_CONTROL,starting_flow:$STARTING_FLOW,dcae_teacher:$DCAE_TEACHER"
run_year() {
  local gpu="$1" year="$2"
  (
  export CUDA_VISIBLE_DEVICES="$gpu"
  echo "[$(date -Is)] year=$year physical_gpu=$gpu start"
  YEAR_OUT=$OUT_ROOT/$year
  RMSE_OUT=$YEAR_OUT/full_year
  mkdir -p "$RMSE_OUT"
  valid=1
  for spec in \
    "distilled_student:$STUDENT" "matched_control:$MATCHED_CONTROL" \
    "starting_flow:$STARTING_FLOW" "dcae_teacher:$DCAE_TEACHER"; do
    name=${spec%%:*}
    checkpoint=${spec#*:}
    if ! (cd "$SOURCE" && "$PY" tools/eval/eval_artifact_status.py \
      "$RMSE_OUT/$name.json" --checkpoint "$checkpoint" \
      --required-taus 1,2,3,4,5 --acc-mode enabled \
      --require-physical-metrics --require-temporal-metrics --quiet); then
      valid=0
    fi
  done
  if [[ "$valid" -eq 0 ]]; then
    (cd "$SOURCE" && "$PY" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir "$MEMMAP" --test-year "$year" --climatology "$CLIM" \
      --climatology-cache-dir "$OUT_ROOT/climatology_cache" --lazy-climatology \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --models "$MODELS" --out-dir "$RMSE_OUT" \
      --paper-tag "distilled_student_full_year_${year}" \
      --batch-size 4 --num-workers 2 --samples-per-date 4 --full-year \
      --max-tau-hours 6 --eval-hours 1,2,3,4,5 \
      --seen-tau 1,3,5 --unseen-tau 2,4 --keep-n-channels 24 \
      --proper-rmse --save-window-metrics --save-physical-metrics \
      --save-temporal-metrics)
  fi
  for spec in \
    "distilled_student:$STUDENT" "matched_control:$MATCHED_CONTROL" \
    "starting_flow:$STARTING_FLOW" "dcae_teacher:$DCAE_TEACHER"; do
    name=${spec%%:*}
    checkpoint=${spec#*:}
    (cd "$SOURCE" && "$PY" tools/eval/eval_artifact_status.py \
      "$RMSE_OUT/$name.json" --checkpoint "$checkpoint" \
      --required-taus 1,2,3,4,5 --acc-mode enabled \
      --require-physical-metrics --require-temporal-metrics --quiet)
  done
  WINDOW=$RMSE_OUT/window_metrics
  (cd "$SOURCE" && "$PY" tools/eval/paired_block_bootstrap.py \
    --left "distilled_student:$WINDOW/distilled_student.npz" \
    --right "matched_control:$WINDOW/matched_control.npz" \
    --right "starting_flow:$WINDOW/starting_flow.npz" \
    --right "dcae_teacher:$WINDOW/dcae_teacher.npz" \
    --taus 1,2,3,4,5 --block-days 7 --draws 5000 --seed 2027 \
    --cellwise --out-json "$RMSE_OUT/paired_rmse.json")
  (cd "$SOURCE" && "$PY" tools/eval/paired_block_bootstrap.py \
    --left "distilled_student:$WINDOW/distilled_student.npz" \
    --right "matched_control:$WINDOW/matched_control.npz" \
    --right "starting_flow:$WINDOW/starting_flow.npz" \
    --right "dcae_teacher:$WINDOW/dcae_teacher.npz" \
    --taus 1,2,3,4,5 --channels Q1000,Q925,Q850,Q700 \
    --block-days 7 --draws 5000 --seed 2027 --cellwise \
    --out-json "$RMSE_OUT/paired_rmse_moisture.json")
  (cd "$SOURCE" && "$PY" tools/eval/paired_block_bootstrap.py \
    --left "distilled_student:$WINDOW/distilled_student.npz" \
    --right "matched_control:$WINDOW/matched_control.npz" \
    --right "starting_flow:$WINDOW/starting_flow.npz" \
    --right "dcae_teacher:$WINDOW/dcae_teacher.npz" \
    --taus 1,2,3,4,5 \
    --channels T1000,T925,T850,T700,U1000,U925,U850,U700,V1000,V925,V850,V700,Z1000,Z925,Z850,Z700,t2m,u10,v10,mslp \
    --block-days 7 --draws 5000 --seed 2027 --cellwise \
    --out-json "$RMSE_OUT/paired_rmse_non_moisture.json")
  for metric in acc temporal_curvature physical; do
    (cd "$SOURCE" && "$PY" tools/eval/paired_aux_block_bootstrap.py \
      --left "distilled_student:$WINDOW/distilled_student.npz" \
      --right "matched_control:$WINDOW/matched_control.npz" \
      --right "starting_flow:$WINDOW/starting_flow.npz" \
      --right "dcae_teacher:$WINDOW/dcae_teacher.npz" \
      --metric "$metric" --draws 5000 --seed 2027 \
      --out-json "$RMSE_OUT/paired_${metric}.json")
  done

  if [[ ! -s "$YEAR_OUT/anchor_exchange.json" ]]; then
    (cd "$SOURCE" && "$PY" -u tools/eval/eval_anchor_exchange_consistency.py \
      --memmap-dir "$MEMMAP" --test-year "$year" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --models "$MODELS" --max-tau-hours 6 --eval-hours 1,2,3,4,5 \
      --samples-per-date 2 --eval-days-per-month 2 \
      --batch-size 2 --num-workers 2 --keep-n-channels 24 \
      --out-json "$YEAR_OUT/anchor_exchange.json" --device cuda:0)
  fi

  REGION_OUT=$YEAR_OUT/region_season
  if [[ ! -e "$REGION_OUT/.complete" ]]; then
    (cd "$SOURCE" && "$PY" -u tools/eval/region_season_12h_eval.py \
      --memmap-dir "$MEMMAP" --test-year "$year" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --models "$MODELS" --out-dir "$REGION_OUT" \
      --batch-size 2 --num-workers 2 --samples-per-date 2 \
      --eval-days-per-month 8 --max-tau-hours 6 --eval-hours 2,4 \
      --keep-n-channels 24 --device cuda:0)
    touch "$REGION_OUT/.complete"
  fi

  SPECTRA=$YEAR_OUT/spectra
  mkdir -p "$SPECTRA"
  for spec in \
    "distilled_student:$STUDENT" "matched_control:$MATCHED_CONTROL" \
    "starting_flow:$STARTING_FLOW" "dcae_teacher:$DCAE_TEACHER"; do
    name=${spec%%:*}
    checkpoint=${spec#*:}
    (cd "$SOURCE" && "$PY" -u tools/eval/sh_energy_spectra_12h.py \
      --memmap-dir "$MEMMAP" --test-year "$year" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" \
      --ckpt "$checkpoint" --model-name "$name" --model-kind capmatched \
      --out-dir "$SPECTRA" --taus 1,2,3,4,5 --channels all \
      --lmax 359 --hf-ell-min 180 --keep-n-channels 24 \
      --batch-size 2 --samples-per-date 2 --full-year \
      --max-tau-hours 6 --device cuda:0 --skip-existing)
  done
  for tau in 1 2 3 4 5; do
    (cd "$SOURCE" && "$PY" tools/eval/spectral_block_bootstrap.py \
      --left "distilled_student:$SPECTRA/distilled_student_tau${tau}.npz" \
      --right "matched_control:$SPECTRA/matched_control_tau${tau}.npz" \
      --right "starting_flow:$SPECTRA/starting_flow_tau${tau}.npz" \
      --right "dcae_teacher:$SPECTRA/dcae_teacher_tau${tau}.npz" \
      --channels all --block-days 7 --draws 5000 --seed 2027 --cellwise \
      --out-json "$SPECTRA/paired_scalar_tau${tau}.json")
    (cd "$SOURCE" && "$PY" tools/eval/spectral_block_bootstrap.py \
      --left "distilled_student:$SPECTRA/distilled_student_tau${tau}.npz" \
      --right "matched_control:$SPECTRA/matched_control_tau${tau}.npz" \
      --right "starting_flow:$SPECTRA/starting_flow_tau${tau}.npz" \
      --right "dcae_teacher:$SPECTRA/dcae_teacher_tau${tau}.npz" \
      --channels all --block-days 7 --draws 5000 --seed 2027 \
      --cellwise --vector --out-json "$SPECTRA/paired_vector_tau${tau}.json")
  done
  echo "[$(date -Is)] year=$year physical_gpu=$gpu complete"
  )
}

run_year 0 2020 &
PID2020=$!
run_year 1 2021 &
PID2021=$!
status=0
wait "$PID2020" || status=$?
wait "$PID2021" || status=$?
[[ "$status" -eq 0 ]] || exit "$status"
export CUDA_VISIBLE_DEVICES=0

(cd "$SOURCE" && "$PY" tools/eval/summarize_anchor_exchange_consistency.py \
  --selection "$FREEZE" \
  --artifact-2020 "$OUT_ROOT/2020/anchor_exchange.json" \
  --artifact-2021 "$OUT_ROOT/2021/anchor_exchange.json" \
  --block-days 7 --draws 5000 --seed 2027 \
  --all-taus 1,2,3,4,5 --seen-taus 1,3,5 --unseen-taus 2,4 \
  --output "$OUT_ROOT/anchor_exchange_generalization.json")
(cd "$SOURCE" && "$PY" tools/eval/summarize_region_season_generalization.py \
  --selection "$FREEZE" \
  --root-2020 "$OUT_ROOT/2020/region_season" \
  --root-2021 "$OUT_ROOT/2021/region_season" \
  --models distilled_student,matched_control,starting_flow,dcae_teacher \
  --block-days 7 --draws 5000 --seed 2027 \
  --output "$OUT_ROOT/region_season_generalization.json")

HRES_OUT=$OUT_ROOT/hres_2021
mkdir -p "$HRES_OUT"
for spec in "distilled_student:$STUDENT:flow_pp3" "matched_control:$MATCHED_CONTROL:flow_pp3"; do
  IFS=: read -r name checkpoint arch <<<"$spec"
  if [[ ! -s "$HRES_OUT/$name.json" ]]; then
    (cd "$SOURCE" && "$PY" -u tools/eval/batch_eval_forecast_anchor.py \
      --forecast-dir "$HRES" --era5-memmap-dir "$MEMMAP" \
      --checkpoint "$checkpoint" --arch "$arch" --model-name "$name" \
      --out-json "$HRES_OUT/$name.json" \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt" --device cuda \
      --delta-t-hours 6 --taus 1 2 3 4 5 --max-inits 16)
  fi
done
(cd "$SOURCE" && "$PY" tools/eval/summarize_forecast_anchor.py \
  --artifact "distilled_student=$HRES_OUT/distilled_student.json" \
  --artifact "matched_control=$HRES_OUT/matched_control.json" \
  --selection "$FREEZE" --draws 5000 --seed 2027 \
  --seen-taus 1,3,5 --unseen-taus 2,4 \
  --output "$HRES_OUT/summary.json")

BARE=$OUT_ROOT/bare
mkdir -p "$BARE"
for spec in "distilled_student:$STUDENT" "matched_control:$MATCHED_CONTROL"; do
  name=${spec%%:*}
  checkpoint=${spec#*:}
  [[ -s "$BARE/$name.pt" ]] || (cd "$SOURCE" && "$PY" \
    tools/train/capmatched_to_bare.py --ckpt "$checkpoint" \
    --arch flow_pp3 --out "$BARE/$name.pt")
done
for year in 2020 2021; do
  DOWN=$OUT_ROOT/downstream/$year
  mkdir -p "$DOWN"
  for name in distilled_student matched_control; do
    [[ -s "$DOWN/physics_$name.json" ]] || (cd "$SOURCE" && "$PY" -u \
      tools/downstream/eval_physics_consistency.py \
      --era5-dir "$MEMMAP" --year "$year" --model-blob "$BARE/$name.pt" \
      --model-name "$name" --out-json "$DOWN/physics_$name.json" \
      --device cuda:0 --delta-t-hours 6 --taus 1 2 3 4 5 \
      --sample-stride-hours 48 --max-pairs 180 \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt")
    [[ -s "$DOWN/diurnal_$name.json" ]] || (cd "$SOURCE" && "$PY" -u \
      tools/downstream/eval_diurnal_amplitude.py \
      --era5-dir "$MEMMAP" --year "$year" --model-blob "$BARE/$name.pt" \
      --model-name "$name" --out-json "$DOWN/diurnal_$name.json" \
      --device cuda:0 --delta-t-hours 6 --taus 1 2 3 4 5 \
      --stats-path "$DATA_ROOT/json_stats_0p5.nc" \
      --surface-stats-path "$DATA_ROOT/surface_stats_0p5.json" \
      --static-path "$DATA_ROOT/static_features_0p5.pt")
  done
done
(cd "$SOURCE" && "$PY" -u tools/eval/benchmark_capmatched_inference.py \
  --model "distilled_student:$STUDENT" --model "matched_control:$MATCHED_CONTROL" \
  --model "starting_flow:$STARTING_FLOW" \
  --static-path "$DATA_ROOT/static_features_0p5.pt" --batch-size 1 \
  --warmup 3 --iterations 10 --repeats 5 \
  --tau-values 0.1666667,0.3333333,0.5,0.6666667,0.8333333 \
  --out-json "$OUT_ROOT/inference_cost.json")

$PY - "$OUT_ROOT" "$FREEZE" "$variant" "0,1" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
freeze = Path(sys.argv[2])
required = [
    freeze,
    root / "2020/full_year/distilled_student.json",
    root / "2021/full_year/distilled_student.json",
    root / "2020/full_year/paired_rmse.json",
    root / "2021/full_year/paired_rmse.json",
    root / "2020/full_year/paired_acc.json",
    root / "2021/full_year/paired_acc.json",
    root / "anchor_exchange_generalization.json",
    root / "region_season_generalization.json",
    root / "hres_2021/summary.json",
    root / "inference_cost.json",
    root / "downstream/2020/physics_distilled_student.json",
    root / "downstream/2020/diurnal_distilled_student.json",
    root / "downstream/2021/physics_distilled_student.json",
    root / "downstream/2021/diurnal_distilled_student.json",
]
for year in (2020, 2021):
    for tau in (1, 2, 3, 4, 5):
        required.extend([
            root / f"{year}/spectra/paired_scalar_tau{tau}.json",
            root / f"{year}/spectra/paired_vector_tau{tau}.json",
        ])
for path in required:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing final artifact: {path}")
payload = {
    "schema_version": 2,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "selected_variant": sys.argv[3],
    "physical_gpus": [int(value) for value in sys.argv[4].split(",")],
    "artifacts": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in required
    },
}
target = root / "pipeline.complete.json"
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(temporary, target)
PY
touch "$STATE/all.complete"
flock -u 7
flock -u 6
echo "[$(date -Is)] distilled-student full test pipeline complete"
