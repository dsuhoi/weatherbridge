#!/usr/bin/env bash
# Full post-selection test pipeline for the frozen Q-head distilled student.
set -euo pipefail

PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weatherbridge_distill_qhead_postrun_v6_source}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
OUT_ROOT=${OUT_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/distilled_student_qhead_postrun_v6}
SCREEN_ROOT=${SCREEN_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_qhead_v4}
GATED_ROOT=${GATED_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/distill_qhead_gated_v7}
MEMMAP=${MEMMAP:-/tmp/wb2_0p5_cache}
DATA_ROOT=${DATA_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/data}
CLIM=${CLIM:-/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr}
HRES=${HRES:-/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_hres_2021}
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}
POLL_SECONDS=${POLL_SECONDS:-120}
MIN_FREE_MIB=${MIN_FREE_MIB:-70000}
MAX_UTIL=${MAX_UTIL:-5}

BASE=$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt
CONTROL=$LOG_ROOT/exp_weatherdcae_14m_6h_gt_continuation_s202707_v1_bs4/last.ckpt
FLOW=$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt
QHEAD_GT=$LOG_ROOT/exp_weatherdcae_14m_6h_qhead_gt_s202707_v4_bs4/last.ckpt
QHEAD_FLOW=$LOG_ROOT/exp_weatherdcae_14m_6h_qhead_flow_s202707_v4_bs4/last.ckpt
QHEAD_FLOW_GATED=$LOG_ROOT/exp_weatherdcae_14m_6h_qhead_flow_gated_s202707_v7_bs4/last.ckpt
RAW_SELECTION=$OUT_ROOT/selection.candidates.json
SCREEN_EVAL=$SCREEN_ROOT/eval_2020_8dpm
GATED_EVAL=$GATED_ROOT/eval_2020_8dpm
MATCHED_SELECTION=$OUT_ROOT/selection.matched_control.json
SELECTION=$OUT_ROOT/selection.strict.json
FREEZE=$OUT_ROOT/selection.freeze.json
LOG=$OUT_ROOT/pipeline.log
STATE=$OUT_ROOT/state

mkdir -p "$OUT_ROOT" "$STATE"
exec >>"$LOG" 2>&1
exec 8>"$STATE/launch.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] distilled-student postrun already active"
  exit 0
fi

(cd "$SOURCE" && sha256sum --quiet -c SOURCE_SHA256SUMS)
export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"

while [[ ! -e "$SCREEN_ROOT/state/screen.complete" ]]; do
  echo "[$(date -Is)] wait Q-head distillation screen"
  sleep "$POLL_SECONDS"
done
while [[ ! -e "$GATED_ROOT/state/all.complete" ]]; do
  echo "[$(date -Is)] wait truth-gated Q-head screen"
  sleep "$POLL_SECONDS"
done
(cd "$SOURCE" && "$PY" tools/eval/select_specialist_distillation.py \
  --control "$SCREEN_EVAL/control.json" \
  --candidate "qhead_flow:$SCREEN_EVAL/qhead_flow.json" \
  --candidate "qhead_flow_gated:$GATED_EVAL/qhead_flow_gated.json" \
  --aggregate-tolerance-pct 0.02 --held-tolerance-pct 0.02 \
  --out "$RAW_SELECTION") || true
test -s "$RAW_SELECTION"
(cd "$SOURCE" && "$PY" tools/eval/select_specialist_distillation.py \
  --control "$SCREEN_EVAL/qhead_gt.json" \
  --candidate "qhead_flow:$SCREEN_EVAL/qhead_flow.json" \
  --candidate "qhead_flow_gated:$GATED_EVAL/qhead_flow_gated.json" \
  --aggregate-tolerance-pct 0.02 --held-tolerance-pct 0.02 \
  --out "$MATCHED_SELECTION") || true
test -s "$MATCHED_SELECTION"
if ! (cd "$SOURCE" && "$PY" tools/eval/select_distilled_champion.py \
  --screen "$RAW_SELECTION" --matched-screen "$MATCHED_SELECTION" \
  --output "$SELECTION"); then
  echo "[$(date -Is)] no broadly improving distilled champion"
  touch "$STATE/strict_selection.failed"
  exit 2
fi
status=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$SELECTION")
if [[ "$status" != pass ]]; then
  echo "[$(date -Is)] distillation selection failed; final testing not admissible"
  printf '%s\n' "$status" >"$STATE/selection.failed"
  exit 2
fi
variant=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' "$SELECTION")
case "$variant" in
  qhead_flow) STUDENT=$QHEAD_FLOW ;;
  qhead_flow_gated) STUDENT=$QHEAD_FLOW_GATED ;;
  *) echo "selected variant is not a distilled Q-head student: $variant" >&2; exit 2 ;;
esac
for checkpoint in "$STUDENT" "$CONTROL" "$FLOW" "$BASE"; do
  test -s "$checkpoint"
done

$PY - "$STUDENT" "$CONTROL" "$variant" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
control = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
hparams = checkpoint.get("hyper_parameters", {})
protocol = hparams.get("training_protocol", {})
distill = protocol.get("distillation", {})
state = checkpoint.get("state_dict", {})
reference = control.get("state_dict", {})
assert hparams.get("arch") == "dcae_14m"
assert sum(v.numel() for k, v in state.items() if k.startswith("net.")) == 14365049
assert int(checkpoint.get("global_step", -1)) >= 3284
assert protocol.get("train_years") == [2014, 2015, 2016, 2017, 2018, 2019]
assert protocol.get("val_years") == [2020]
assert protocol.get("train_tau_hours") == [1, 3, 5]
assert protocol.get("eval_tau_hours") == [1, 2, 3, 4, 5]
assert protocol.get("global_effective_batch_size") == 16
assert protocol.get("trainable_scope") == "q_output_head"
assert protocol.get("optimizer_weight_decay") == 0.0
expected_gate = "teacher_better" if sys.argv[3] == "qhead_flow_gated" else None
assert distill.get("high_gate") == expected_gate
assert not any("teacher" in key or "distill" in key for key in state)

mutable = {"net.decoder.conv_out.weight", "net.decoder.conv_out.bias"}
for key, value in state.items():
    if not key.startswith("net.") or key in mutable:
        continue
    assert torch.equal(value, reference[key]), key
for key in mutable:
    assert torch.equal(state[key][:12], reference[key][:12]), key
    assert torch.equal(state[key][16:], reference[key][16:]), key
    assert not torch.equal(state[key][12:16], reference[key][12:16]), key
print({"variant": sys.argv[3], "global_step": checkpoint["global_step"]})
PY

(cd "$SOURCE" && "$PY" tools/eval/freeze_distilled_student_selection.py \
  --selection "$SELECTION" \
  --model "distilled_student:$STUDENT" \
  --model "control:$CONTROL" \
  --model "flow_teacher:$FLOW" \
  --model "base_dcae:$BASE" \
  --output "$FREEZE")

GPU=""
GPU_LOCK_FD=""
while [[ -z "$GPU" ]]; do
  for candidate in 0 1; do
    exec {candidate_fd}>"$LOG_ROOT/.upr_lite_gpu${candidate}.lock"
    if ! flock -n "$candidate_fd"; then
      exec {candidate_fd}>&-
      continue
    fi
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')
    if [[ "$free" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      GPU="$candidate"
      GPU_LOCK_FD="$candidate_fd"
      break
    fi
    flock -u "$candidate_fd"
    exec {candidate_fd}>&-
  done
  [[ -n "$GPU" ]] || sleep "$POLL_SECONDS"
done
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[$(date -Is)] acquired physical GPU=$GPU variant=$variant"

MODELS="distilled_student:$STUDENT,control:$CONTROL,flow_teacher:$FLOW,base_dcae:$BASE"
for year in 2020 2021; do
  YEAR_OUT=$OUT_ROOT/$year
  RMSE_OUT=$YEAR_OUT/full_year
  mkdir -p "$RMSE_OUT"
  valid=1
  for spec in \
    "distilled_student:$STUDENT" "control:$CONTROL" \
    "flow_teacher:$FLOW" "base_dcae:$BASE"; do
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
    "distilled_student:$STUDENT" "control:$CONTROL" \
    "flow_teacher:$FLOW" "base_dcae:$BASE"; do
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
    --right "control:$WINDOW/control.npz" \
    --right "flow_teacher:$WINDOW/flow_teacher.npz" \
    --right "base_dcae:$WINDOW/base_dcae.npz" \
    --taus 1,2,3,4,5 --block-days 7 --draws 5000 --seed 2027 \
    --cellwise --out-json "$RMSE_OUT/paired_rmse.json")
  (cd "$SOURCE" && "$PY" tools/eval/paired_block_bootstrap.py \
    --left "distilled_student:$WINDOW/distilled_student.npz" \
    --right "control:$WINDOW/control.npz" \
    --right "flow_teacher:$WINDOW/flow_teacher.npz" \
    --right "base_dcae:$WINDOW/base_dcae.npz" \
    --taus 1,2,3,4,5 --channels Q1000,Q925,Q850,Q700 \
    --block-days 7 --draws 5000 --seed 2027 --cellwise \
    --out-json "$RMSE_OUT/paired_rmse_moisture.json")
  (cd "$SOURCE" && "$PY" tools/eval/paired_block_bootstrap.py \
    --left "distilled_student:$WINDOW/distilled_student.npz" \
    --right "control:$WINDOW/control.npz" \
    --right "flow_teacher:$WINDOW/flow_teacher.npz" \
    --right "base_dcae:$WINDOW/base_dcae.npz" \
    --taus 1,2,3,4,5 \
    --channels T1000,T925,T850,T700,U1000,U925,U850,U700,V1000,V925,V850,V700,Z1000,Z925,Z850,Z700,t2m,u10,v10,mslp \
    --block-days 7 --draws 5000 --seed 2027 --cellwise \
    --out-json "$RMSE_OUT/paired_rmse_non_moisture.json")
  for metric in acc temporal_curvature physical; do
    (cd "$SOURCE" && "$PY" tools/eval/paired_aux_block_bootstrap.py \
      --left "distilled_student:$WINDOW/distilled_student.npz" \
      --right "control:$WINDOW/control.npz" \
      --right "flow_teacher:$WINDOW/flow_teacher.npz" \
      --right "base_dcae:$WINDOW/base_dcae.npz" \
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
    "distilled_student:$STUDENT" "control:$CONTROL" \
    "flow_teacher:$FLOW" "base_dcae:$BASE"; do
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
      --right "control:$SPECTRA/control_tau${tau}.npz" \
      --right "flow_teacher:$SPECTRA/flow_teacher_tau${tau}.npz" \
      --right "base_dcae:$SPECTRA/base_dcae_tau${tau}.npz" \
      --channels all --block-days 7 --draws 5000 --seed 2027 --cellwise \
      --out-json "$SPECTRA/paired_scalar_tau${tau}.json")
    (cd "$SOURCE" && "$PY" tools/eval/spectral_block_bootstrap.py \
      --left "distilled_student:$SPECTRA/distilled_student_tau${tau}.npz" \
      --right "control:$SPECTRA/control_tau${tau}.npz" \
      --right "flow_teacher:$SPECTRA/flow_teacher_tau${tau}.npz" \
      --right "base_dcae:$SPECTRA/base_dcae_tau${tau}.npz" \
      --channels all --block-days 7 --draws 5000 --seed 2027 \
      --cellwise --vector --out-json "$SPECTRA/paired_vector_tau${tau}.json")
  done
done

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
  --models distilled_student,control,flow_teacher,base_dcae \
  --block-days 7 --draws 5000 --seed 2027 \
  --output "$OUT_ROOT/region_season_generalization.json")

HRES_OUT=$OUT_ROOT/hres_2021
mkdir -p "$HRES_OUT"
for spec in "distilled_student:$STUDENT:dcae_14m" "control:$CONTROL:dcae_14m"; do
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
  --artifact "control=$HRES_OUT/control.json" \
  --selection "$FREEZE" --draws 5000 --seed 2027 \
  --seen-taus 1,3,5 --unseen-taus 2,4 \
  --output "$HRES_OUT/summary.json")

BARE=$OUT_ROOT/bare
mkdir -p "$BARE"
for spec in "distilled_student:$STUDENT" "control:$CONTROL"; do
  name=${spec%%:*}
  checkpoint=${spec#*:}
  [[ -s "$BARE/$name.pt" ]] || (cd "$SOURCE" && "$PY" \
    tools/train/capmatched_to_bare.py --ckpt "$checkpoint" \
    --arch dcae_14m --out "$BARE/$name.pt")
done
for year in 2020 2021; do
  DOWN=$OUT_ROOT/downstream/$year
  mkdir -p "$DOWN"
  for name in distilled_student control; do
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
  --model "distilled_student:$STUDENT" --model "control:$CONTROL" \
  --model "flow_teacher:$FLOW" \
  --static-path "$DATA_ROOT/static_features_0p5.pt" --batch-size 1 \
  --warmup 3 --iterations 10 --repeats 5 \
  --tau-values 0.1666667,0.3333333,0.5,0.6666667,0.8333333 \
  --out-json "$OUT_ROOT/inference_cost.json")

$PY - "$OUT_ROOT" "$FREEZE" "$variant" "$GPU" <<'PY'
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
    root / "2020/full_year/paired_rmse.json",
    root / "2021/full_year/paired_rmse.json",
    root / "anchor_exchange_generalization.json",
    root / "region_season_generalization.json",
    root / "hres_2021/summary.json",
    root / "inference_cost.json",
    root / "downstream/2021/physics_distilled_student.json",
    root / "downstream/2021/diurnal_distilled_student.json",
]
for path in required:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing final artifact: {path}")
payload = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "selected_variant": sys.argv[3],
    "gpu": int(sys.argv[4]),
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
flock -u "$GPU_LOCK_FD"
echo "[$(date -Is)] distilled-student full test pipeline complete"
