#!/usr/bin/env bash
# Evaluate the frozen 2022 holdout only after journal model selection is final.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
DATA=${DATA:-/home/jovyan/shares/SR006.nfs2/dsuhoi/postselection_2022_memmap}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
POLL_SECONDS=${POLL_SECONDS:-120}
GPU_CANDIDATES=${GPU_CANDIDATES:-"0 1"}
MIN_FREE_MIB=${MIN_FREE_MIB:-70000}
MAX_UTIL=${MAX_UTIL:-5}

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"

MANIFEST=repro/postselection_holdout_2022.json
CHAMPION=metrics/journal_champion_v1/final.json
CHAMPION_MARKER=metrics/journal_champion_v1/state/.complete
OUT="$RUNTIME/metrics/postselection_2022_v1"
STATE="$OUT/state"
LOG="$LOG_ROOT/postselection_2022_v1.queue.log"
VERIFICATION=${VERIFICATION:-"$DATA/wb2_2022.verification_v2.json"}
CANDIDATE_ASSESSMENT="$OUT/weatherbridge_assessment.json"
STATIC=/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt
STATS=/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc
SURFACE_STATS=/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json
CLIM=/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr
CACHE=/tmp/wti_climatology_cache_postselection_2022
mkdir -p "$STATE" "$CACHE"

exec 8>"$LOG_ROOT/.postselection_2022_v1.queue.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] post-selection queue already active" | tee -a "$LOG"
  exit 0
fi
exec > >(tee -a "$LOG") 2>&1

SOURCE_FILES=(
  scripts/run_postselection_2022_queue_cloudru.sh
  tools/data/build_postselection_2022_memmap.py
  tools/data/verify_postselection_2022_memmap.py
  tools/eval/assess_postselection_holdout.py
  tools/eval/export_postselection_tex.py
  tools/eval/batch_eval_12h_memmap.py
  tools/eval/capmatched_loader.py
  tools/eval/eval_artifact_status.py
  tools/eval/paired_block_bootstrap.py
  tools/eval/paired_aux_block_bootstrap.py
  tools/eval/hard_window_block_bootstrap.py
  tools/eval/sh_energy_spectra_12h.py
  tools/eval/spectral_block_bootstrap.py
  tools/eval/summarize_spectral_dominance.py
  tools/train/train_capacity_matched_6h.py
  weather_time_interp/grid.py
  weather_time_interp/eval_runner.py
  weather_time_interp/normalization.py
  weather_time_interp/metrics/spherical_spectra.py
  weather_time_interp/metrics/physical_consistency.py
  weather_time_interp/model/weatherbridge_flow_model.py
  weather_time_interp/model/dcae_adaln_model.py
  repro/postselection_holdout_2022.json
)
SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
if [[ -n "${EXPECTED_SOURCE_SHA:-}" && "$SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]]; then
  echo "source hash mismatch: $SOURCE_SHA != $EXPECTED_SOURCE_SHA" >&2
  exit 2
fi
printf '%s  %s\n' "$SOURCE_SHA" "${SOURCE_FILES[*]}" >"$STATE/source.sha256"

verify_source_snapshot() {
  local current_sha
  current_sha=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
  if [[ "$current_sha" != "$SOURCE_SHA" ]]; then
    echo "source changed while post-selection was active: $current_sha != $SOURCE_SHA" >&2
    exit 2
  fi
}

wait_for_file() {
  local path="$1"
  while [[ ! -s "$path" ]]; do
    echo "[$(date -Is)] waiting for $path"
    sleep "$POLL_SECONDS"
  done
}

# The ordering is the anti-leakage barrier: no holdout content verification or
# model inference occurs before the selector has frozen its winner.
wait_for_file "$DATA/wb2_2022.json"
wait_for_file "$CHAMPION_MARKER"
wait_for_file "$CHAMPION"
verify_source_snapshot

"$PY" tools/data/verify_postselection_2022_memmap.py \
  --manifest "$MANIFEST" --memmap-dir "$DATA" --out-json "$VERIFICATION"

DETAIL_6="$LOG_ROOT/exp_flow_pp3_detail_14m_6h_refinev1_s202707_bs4/last.ckpt"
REFINE_6="$LOG_ROOT/exp_flow_universal_latent_refine_14m_6h_s202707_v1_bs4/last.ckpt"
FLOW_6="$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt"
DCAE_6="$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt"
DETAIL_12="$LOG_ROOT/exp_flow_pp3_detail_14m_12h_2017_19_refinev1_s202707_bs4/last.ckpt"
REFINE_12="$LOG_ROOT/exp_flow_universal_latent_refine_14m_12h_2017_19_s202707_v1_bs4/last.ckpt"
FLOW_12="$LOG_ROOT/exp_flow_pp3_spectral_14m_12h_2017_19_refinev1_s202707_bs4/last.ckpt"
DCAE_12="$LOG_ROOT/exp_weatherdcae_14m_12h_2017_19_refinev1_s202707_bs4/last.ckpt"
for checkpoint in \
  "$DETAIL_6" "$REFINE_6" "$FLOW_6" "$DCAE_6" \
  "$DETAIL_12" "$REFINE_12" "$FLOW_12" "$DCAE_12"; do
  wait_for_file "$checkpoint"
done
verify_source_snapshot

acquire_gpu() {
  local candidate candidate_fd free_mib util
  while true; do
    for candidate in $GPU_CANDIDATES; do
      exec {candidate_fd}>"$LOG_ROOT/.upr_lite_gpu${candidate}.lock"
      if ! flock -n "$candidate_fd"; then
        exec {candidate_fd}>&-
        continue
      fi
      free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')
      util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')
      if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
        GPU="$candidate"
        GPU_LOCK_FD="$candidate_fd"
        export CUDA_VISIBLE_DEVICES="$GPU"
        return 0
      fi
      flock -u "$candidate_fd"
      exec {candidate_fd}>&-
    done
    sleep "$POLL_SECONDS"
  done
}

run_horizon() {
  local horizon="$1" detail="$2" refine="$3" flow="$4" dcae="$5"
  local eval_hours seen held horizon_out models candidate checkpoint
  verify_source_snapshot
  if [[ "$horizon" == 6 ]]; then
    eval_hours=1,2,3,4,5
    seen=1,3,5
    held=2,4
  else
    eval_hours=1,2,3,4,5,6,7,8,9,10,11
    seen=1,2,3,5,7,9,10,11
    held=4,6,8
  fi
  horizon_out="$OUT/${horizon}h"
  mkdir -p "$horizon_out"
  models="weatherbridge_detail:$detail,refine:$refine,flow_spectral:$flow,weatherdcae_14m:$dcae"
  local reuse_evaluation=1
  for entry in \
    "weatherbridge_detail:$detail" "refine:$refine" \
    "flow_spectral:$flow" "weatherdcae_14m:$dcae"; do
    IFS=: read -r candidate checkpoint <<<"$entry"
    if ! "$PY" tools/eval/eval_artifact_status.py \
      "$horizon_out/$candidate.json" --checkpoint "$checkpoint" \
      --required-taus "$eval_hours" --acc-mode enabled --allow-economy \
      --require-physical-metrics --require-temporal-metrics --quiet; then
      reuse_evaluation=0
      break
    fi
  done
  if [[ "$reuse_evaluation" -eq 1 ]]; then
    echo "[$(date -Is)] reusing verified ${horizon}h evaluation artifacts"
  else
    "$PY" -u tools/eval/batch_eval_12h_memmap.py \
      --memmap-dir "$DATA" --test-year 2022 --climatology "$CLIM" \
      --stats-path "$STATS" --surface-stats-path "$SURFACE_STATS" \
      --static-path "$STATIC" --models "$models" --out-dir "$horizon_out" \
      --paper-tag "postselection_2022_${horizon}h" --batch-size 2 \
      --num-workers 0 --samples-per-date 1 --days-of-month 1,8,15,22 \
      --proper-rmse --save-window-metrics --save-physical-metrics \
      --save-temporal-metrics --max-tau-hours "$horizon" \
      --eval-hours "$eval_hours" --seen-tau "$seen" --unseen-tau "$held" \
      --device cuda:0 --lazy-climatology --climatology-cache-dir "$CACHE" \
      --keep-n-channels 24
  fi

  for entry in \
    "weatherbridge_detail:$detail" "refine:$refine" \
    "flow_spectral:$flow" "weatherdcae_14m:$dcae"; do
    IFS=: read -r candidate checkpoint <<<"$entry"
    "$PY" tools/eval/eval_artifact_status.py \
      "$horizon_out/$candidate.json" --checkpoint "$checkpoint" \
      --required-taus "$eval_hours" --acc-mode enabled --allow-economy \
      --require-physical-metrics --require-temporal-metrics --quiet
  done

  for candidate in weatherbridge_detail refine flow_spectral; do
    "$PY" tools/eval/paired_block_bootstrap.py \
      --left "$candidate:$horizon_out/window_metrics/$candidate.npz" \
      --right "weatherdcae_14m:$horizon_out/window_metrics/weatherdcae_14m.npz" \
      --draws 10000 --seed 20220809 --cellwise \
      --out-json "$horizon_out/paired_rmse_${candidate}.json"
    "$PY" tools/eval/paired_block_bootstrap.py \
      --left "$candidate:$horizon_out/window_metrics/$candidate.npz" \
      --right "weatherdcae_14m:$horizon_out/window_metrics/weatherdcae_14m.npz" \
      --taus "$held" --draws 10000 --seed 20220809 --cellwise \
      --out-json "$horizon_out/paired_rmse_held_${candidate}.json"
    "$PY" tools/eval/hard_window_block_bootstrap.py \
      --left "$candidate:$horizon_out/window_metrics/$candidate.npz" \
      --right "weatherdcae_14m:$horizon_out/window_metrics/weatherdcae_14m.npz" \
      --quantile 0.95 --draws 10000 --seed 20220809 \
      --out-json "$horizon_out/paired_hard_window_${candidate}.json"
    for metric in acc temporal_curvature physical; do
      "$PY" tools/eval/paired_aux_block_bootstrap.py \
        --left "$candidate:$horizon_out/window_metrics/$candidate.npz" \
        --right "weatherdcae_14m:$horizon_out/window_metrics/weatherdcae_14m.npz" \
        --metric "$metric" --draws 10000 --seed 20220809 \
      --out-json "$horizon_out/paired_${metric}_${candidate}.json"
    done
  done

  local spectra_out="$horizon_out/spectra"
  mkdir -p "$spectra_out"
  for entry in \
    "weatherbridge_detail:$detail" "refine:$refine" \
    "flow_spectral:$flow" "weatherdcae_14m:$dcae"; do
    IFS=: read -r candidate checkpoint <<<"$entry"
    "$PY" -u tools/eval/sh_energy_spectra_12h.py \
      --memmap-dir "$DATA" --test-year 2022 \
      --stats-path "$STATS" --surface-stats-path "$SURFACE_STATS" \
      --static-path "$STATIC" --ckpt "$checkpoint" \
      --model-name "$candidate" --model-kind capmatched \
      --out-dir "$spectra_out" --taus "$eval_hours" --channels all \
      --lmax 180 --hf-ell-min 80 --batch-size 2 \
      --samples-per-date 1 --days-of-month 1,8,15,22 \
      --max-tau-hours "$horizon" --device cuda:0 --skip-existing
  done
  for candidate in weatherbridge_detail refine flow_spectral; do
    for tau in $(seq 1 $((horizon - 1))); do
      "$PY" tools/eval/spectral_block_bootstrap.py \
        --left "$candidate:$spectra_out/${candidate}_tau${tau}.npz" \
        --right "weatherdcae_14m:$spectra_out/weatherdcae_14m_tau${tau}.npz" \
        --block-days 7 --draws 10000 --seed 20220809 --cellwise \
        --out-json "$spectra_out/paired_scalar_${candidate}_tau${tau}.json"
      "$PY" tools/eval/spectral_block_bootstrap.py \
        --left "$candidate:$spectra_out/${candidate}_tau${tau}.npz" \
        --right "weatherdcae_14m:$spectra_out/weatherdcae_14m_tau${tau}.npz" \
        --block-days 7 --draws 10000 --seed 20220809 --cellwise --vector \
        --out-json "$spectra_out/paired_vector_${candidate}_tau${tau}.json"
    done
    "$PY" tools/eval/summarize_spectral_dominance.py \
      --root "$spectra_out" --years 2022 --taus "$eval_hours" \
      --pattern "paired_scalar_${candidate}_tau{tau}.json" \
      --reference weatherdcae_14m \
      --out-json "$spectra_out/global_scalar_${candidate}_vs_weatherdcae_14m.json"
    "$PY" tools/eval/summarize_spectral_dominance.py \
      --root "$spectra_out" --years 2022 --taus "$eval_hours" \
      --pattern "paired_vector_${candidate}_tau{tau}.json" \
      --reference weatherdcae_14m \
      --out-json "$spectra_out/global_vector_${candidate}_vs_weatherdcae_14m.json"
  done
  verify_source_snapshot
}

acquire_gpu
run_horizon 6 "$DETAIL_6" "$REFINE_6" "$FLOW_6" "$DCAE_6"
run_horizon 12 "$DETAIL_12" "$REFINE_12" "$FLOW_12" "$DCAE_12"
verify_source_snapshot
flock -u "$GPU_LOCK_FD"
exec {GPU_LOCK_FD}>&-

"$PY" tools/eval/assess_postselection_holdout.py \
  --manifest "$MANIFEST" --champion "$CHAMPION" \
  --verification "$VERIFICATION" --evaluation-root "$OUT" \
  --out-json "$OUT/final.json"
"$PY" tools/eval/assess_postselection_holdout.py \
  --manifest "$MANIFEST" --champion "$CHAMPION" \
  --verification "$VERIFICATION" --evaluation-root "$OUT" \
  --candidate flow_spectral --out-json "$CANDIDATE_ASSESSMENT"
"$PY" tools/eval/export_postselection_tex.py \
  --assessment "$CANDIDATE_ASSESSMENT" --out-tex paper/postselection_2022.tex
verify_source_snapshot

"$PY" - "$OUT/final.json" "$CANDIDATE_ASSESSMENT" \
  "$STATE/.complete" "$SOURCE_SHA" \
  paper/postselection_2022.tex "$CHAMPION" "$CHAMPION_MARKER" \
  "${SOURCE_FILES[@]}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

result_path = Path(sys.argv[1])
candidate_path = Path(sys.argv[2])
marker_path = Path(sys.argv[3])
source_sha = sys.argv[4]
tex_path = Path(sys.argv[5])
champion_path = Path(sys.argv[6])
champion_marker_path = Path(sys.argv[7])
source_files = [Path(value) for value in sys.argv[8:]]
payload = json.loads(result_path.read_text())
candidate_payload = json.loads(candidate_path.read_text())
marker = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "source_sha256": source_sha,
    "source_files_sha256": {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in source_files
    },
    "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    "candidate_result_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    "tex_sha256": hashlib.sha256(tex_path.read_bytes()).hexdigest(),
    "champion_sha256": hashlib.sha256(champion_path.read_bytes()).hexdigest(),
    "champion_marker_sha256": hashlib.sha256(
        champion_marker_path.read_bytes()
    ).hexdigest(),
    "assessment_status": payload["status"],
    "candidate_assessment_status": candidate_payload["status"],
    "frozen_winner": payload["frozen_winner"],
    "evaluated_candidate": candidate_payload["evaluated_candidate"],
}
target = Path(marker_path)
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(marker, sort_keys=True) + "\n")
os.replace(temporary, target)
PY
echo "[$(date -Is)] frozen 2022 holdout evaluation complete"
