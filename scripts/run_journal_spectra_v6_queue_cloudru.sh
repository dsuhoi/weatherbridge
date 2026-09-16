#!/usr/bin/env bash
# Publication spectral protocol: source-grid scalar/vector SHT and dense blocks.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
LEGACY=${LEGACY:-/home/jovyan/dsuhoi/weather_time_interpolation}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
GPU_CANDIDATES=${GPU_CANDIDATES:-"0 1"}
MIN_FREE_MIB=${MIN_FREE_MIB:-60000}
MAX_UTIL=${MAX_UTIL:-5}
POLL_SECONDS=${POLL_SECONDS:-120}
RUN_2021=${RUN_2021:-1}
LMAX=180
ELL_MIN=80
DRAW_COUNT=${DRAW_COUNT:-10000}
PREBUILT_DETAIL_SPECTRA_PDF=${PREBUILT_DETAIL_SPECTRA_PDF:-}

cd "$RUNTIME"
export PYTHONPATH="$RUNTIME:$LEGACY:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
OUT_ROOT=metrics/journal_spectra_v6
STATE_DIR="$OUT_ROOT/state"
mkdir -p "$STATE_DIR" logs/runner
LOG=logs/runner/journal_spectra_v6_queue.log
MARKER="$STATE_DIR/.complete"
exec 8>"$LOG_ROOT/.journal_spectra_v6_queue.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] corrected spectral queue already active" | tee -a "$LOG"
  exit 0
fi

SOURCE_FILES=(
  scripts/run_journal_spectra_v6_queue_cloudru.sh
  dataset.py
  metrics/__init__.py
  metrics/weather.py
  legacy/scripts/trainer_weather_hermite.py
  weather_time_interp/grid.py
  weather_time_interp/memmap_dataset.py
  weather_time_interp/eval_datasets.py
  weather_time_interp/metrics/spherical_spectra.py
  weather_time_interp/model/fuxi_swinv2_model.py
  weather_time_interp/model/modafno_baseline_model.py
  weather_time_interp/model/sdyff_dyffusion_model.py
  tools/eval/sh_energy_spectra_12h.py
  tools/eval/batch_eval_memmap.py
  tools/eval/capmatched_loader.py
  tools/eval/spectral_block_bootstrap.py
  tools/eval/summarize_spectral_dominance.py
  tools/eval/build_sh_per_channel_table_12h.py
  scripts/export_weatherbridge_spectral_latex_data.py
  scripts/make_fig3_spectra.py
  paper/figs/fig_weatherbridge_spectra_tau23.tex
  legacy/scripts/train_atm_vfi_12h_oddskip.py
)
SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
if [[ -n "${EXPECTED_SOURCE_SHA:-}" && "$SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]]; then
  echo "source hash mismatch: $SOURCE_SHA != $EXPECTED_SOURCE_SHA" | tee -a "$LOG"
  exit 2
fi
printf '%s  %s\n' "$SOURCE_SHA" "${SOURCE_FILES[*]}" >"$STATE_DIR/source.sha256"

verify_source_snapshot() {
  local current_sha
  current_sha=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
  if [[ "$current_sha" != "$SOURCE_SHA" ]]; then
    echo "source changed while spectral queue was active: $current_sha != $SOURCE_SHA" >&2
    exit 2
  fi
}

DETAIL_6="$LOG_ROOT/exp_flow_pp3_detail_14m_6h_refinev1_s202707_bs4/last.ckpt"
REFINE_6="$LOG_ROOT/exp_flow_universal_latent_refine_14m_6h_s202707_v1_bs4/last.ckpt"
SPECTRAL_6="$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt"
DCAE_6="$LOG_ROOT/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt"
DETAIL_12="$LOG_ROOT/exp_flow_pp3_detail_14m_12h_2017_19_refinev1_s202707_bs4/last.ckpt"
REFINE_12="$LOG_ROOT/exp_flow_universal_latent_refine_14m_12h_2017_19_s202707_v1_bs4/last.ckpt"
SPECTRAL_12="$LOG_ROOT/exp_flow_pp3_spectral_14m_12h_2017_19_refinev1_s202707_bs4/last.ckpt"
DCAE_12="$LOG_ROOT/exp_weatherdcae_14m_12h_2017_19_refinev1_s202707_bs4/last.ckpt"

CORE_6=(
  "weatherbridge|$DETAIL_6|capmatched|"
  "refine|$REFINE_6|capmatched|"
  "flow_spectral|$SPECTRAL_6|capmatched|"
  "weatherdcae_14m|$DCAE_6|capmatched|"
)
PREDETAIL_6=(
  "refine|$REFINE_6|capmatched|"
  "flow_spectral|$SPECTRAL_6|capmatched|"
  "weatherdcae_14m|$DCAE_6|capmatched|"
)
CORE_12=(
  "weatherbridge|$DETAIL_12|capmatched|"
  "refine|$REFINE_12|capmatched|"
  "flow_spectral|$SPECTRAL_12|capmatched|"
  "weatherdcae_14m|$DCAE_12|capmatched|"
)
BASELINES_6=(
  "fuxi|$LEGACY/logs/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt|hermite|"
  "modafno|$LEGACY/logs/exp_modafno_24ch_6yr/epoch=7-step=35032.ckpt|hermite|MODAFNO_INP_H=360 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
  "sdyff|$LEGACY/logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt|hermite|SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"
  "pixelattn_vfi|$LOG_ROOT/exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_matched/last.ckpt|atmvfi|"
  "linear|analytic|bilinear|"
)
BASELINES_12=(
  "fuxi|$LOG_ROOT/exp_fuxi_24ch_12h_2017_19_lr1e4/epoch=9-step=5470.ckpt|hermite|"
  "modafno|$LEGACY/logs/_12h_migrated/modafno_12h.ckpt|hermite|MODAFNO_INP_H=362 MODAFNO_INP_W=720 MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720"
  "sdyff|$LEGACY/logs/_12h_migrated/sdyff_12h.ckpt|hermite|SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0"
  "pixelattn_vfi|$LOG_ROOT/exp_atmvfi_24ch_12h_2017_19_oddskip_lr1e4_matched/last.ckpt|atmvfi|"
  "linear|analytic|bilinear|"
)

wait_for_specs() {
  local missing spec name checkpoint kind envs
  while true; do
    missing=0
    for spec in "$@"; do
      IFS='|' read -r name checkpoint kind envs <<<"$spec"
      if [[ "$kind" != bilinear && ! -s "$checkpoint" ]]; then
        echo "[$(date -Is)] wait checkpoint $checkpoint" | tee -a "$LOG"
        missing=1
      fi
    done
    [[ "$missing" -eq 0 ]] && return 0
    sleep "$POLL_SECONDS"
  done
}

release_gpu() {
  flock -u "$GPU_LOCK_FD"
  exec {GPU_LOCK_FD}>&-
}

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

run_spectrum() {
  local year="$1" horizon="$2" taus="$3" spec="$4"
  local name checkpoint kind envs out
  IFS='|' read -r name checkpoint kind envs <<<"$spec"
  out="$OUT_ROOT/${horizon}h_${year}"
  mkdir -p "$out"
  echo "[$(date -Is)] year=$year horizon=$horizon model=$name" | tee -a "$LOG"
  "$PY" -u tools/eval/sh_energy_spectra_12h.py \
    --memmap-dir /tmp/wb2_0p5_cache \
    --test-year "$year" \
    --stats-path data/json_stats_0p5.nc \
    --surface-stats-path data/surface_stats_0p5.json \
    --static-path data/static_features_0p5.pt \
    --ckpt "$checkpoint" --model-name "$name" --model-kind "$kind" \
    --envs "$envs" --out-dir "$out" --taus "$taus" --channels all \
    --lmax "$LMAX" --hf-ell-min "$ELL_MIN" --keep-n-channels 24 \
    --batch-size 2 --samples-per-date 2 --full-year \
    --max-tau-hours "$horizon" --device cuda:0 --skip-existing \
    2>&1 | tee -a "$LOG"
}

run_statistics() {
  local year="$1" horizon="$2" taus="$3" spec name checkpoint kind envs tau
  local left_name summary_scalar summary_vector
  local out="$OUT_ROOT/${horizon}h_${year}"
  shift 3
  local specs=("$@")
  IFS=',' read -ra TAU_LIST <<<"$taus"
  for left_name in weatherbridge refine flow_spectral weatherdcae_14m; do
    for tau in "${TAU_LIST[@]}"; do
      local scalar_rights=() vector_rights=()
      for spec in "${specs[@]}"; do
        IFS='|' read -r name checkpoint kind envs <<<"$spec"
        [[ "$name" == "$left_name" ]] && continue
        scalar_rights+=(--right "$name:$out/${name}_tau${tau}.npz")
        vector_rights+=(--right "$name:$out/${name}_tau${tau}.npz")
      done
      "$PY" tools/eval/spectral_block_bootstrap.py \
        --left "$left_name:$out/${left_name}_tau${tau}.npz" \
        "${scalar_rights[@]}" --channels all --block-days 7 \
        --draws "$DRAW_COUNT" --seed 2027 --cellwise \
        --out-json "$out/${left_name}_pairwise_tau${tau}.json" | tee -a "$LOG"
      "$PY" tools/eval/spectral_block_bootstrap.py \
        --left "$left_name:$out/${left_name}_tau${tau}.npz" \
        "${vector_rights[@]}" --channels all --block-days 7 \
        --draws "$DRAW_COUNT" --seed 2027 --cellwise --vector \
        --out-json "$out/${left_name}_vector_pairwise_tau${tau}.json" | tee -a "$LOG"
    done
    for spec in "${specs[@]}"; do
      IFS='|' read -r name checkpoint kind envs <<<"$spec"
      [[ "$name" == "$left_name" ]] && continue
      summary_scalar="$out/global_scalar_${left_name}_vs_${name}.json"
      summary_vector="$out/global_vector_${left_name}_vs_${name}.json"
      "$PY" tools/eval/summarize_spectral_dominance.py \
        --root "$OUT_ROOT" --years "$year" --taus "$taus" \
        --pattern "${horizon}h_{year}/${left_name}_pairwise_tau{tau}.json" \
        --reference "$name" --out-json "$summary_scalar"
      "$PY" tools/eval/summarize_spectral_dominance.py \
        --root "$OUT_ROOT" --years "$year" --taus "$taus" \
        --pattern "${horizon}h_{year}/${left_name}_vector_pairwise_tau{tau}.json" \
        --reference "$name" --out-json "$summary_vector"
      if [[ "$left_name" == weatherbridge ]]; then
        cp "$summary_scalar" "$out/global_scalar_vs_${name}.json"
        cp "$summary_vector" "$out/global_vector_vs_${name}.json"
      fi
    done
  done
}

run_suite() {
  local year="$1" horizon="$2" taus="$3"
  shift 3
  local spec
  for spec in "$@"; do
    run_spectrum "$year" "$horizon" "$taus" "$spec"
  done
  run_statistics "$year" "$horizon" "$taus" "$@"
}

wait_for_specs "${PREDETAIL_6[@]}" "${BASELINES_6[@]}"
acquire_gpu
echo "[$(date -Is)] acquired physical GPU $GPU for pre-Detail 6h spectra source=$SOURCE_SHA" | tee -a "$LOG"
for spec in "${PREDETAIL_6[@]}" "${BASELINES_6[@]}"; do
  run_spectrum 2020 6 1,2,3,4,5 "$spec"
done
if [[ "$RUN_2021" -eq 1 ]]; then
  for spec in "${PREDETAIL_6[@]}"; do
    run_spectrum 2021 6 1,2,3,4,5 "$spec"
  done
fi
release_gpu

wait_for_specs "${CORE_6[0]}"
acquire_gpu
echo "[$(date -Is)] acquired physical GPU $GPU for Detail 6h spectra" | tee -a "$LOG"
run_spectrum 2020 6 1,2,3,4,5 "${CORE_6[0]}"
if [[ "$RUN_2021" -eq 1 ]]; then
  run_spectrum 2021 6 1,2,3,4,5 "${CORE_6[0]}"
fi
run_statistics 2020 6 1,2,3,4,5 "${CORE_6[@]}" "${BASELINES_6[@]}"
if [[ "$RUN_2021" -eq 1 ]]; then
  run_statistics 2021 6 1,2,3,4,5 "${CORE_6[@]}"
fi
verify_source_snapshot
"$PY" - "$OUT_ROOT" "$STATE_DIR/.complete_6h" "$SOURCE_SHA" "$RUN_2021" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
marker = Path(sys.argv[2])
source_sha = sys.argv[3]
run_2021 = bool(int(sys.argv[4]))
expected = {"6h_2020": 45}
if run_2021:
    expected["6h_2021"] = 20
artifact_count = 0
for directory, count in expected.items():
    paths = sorted((root / directory).glob("*_tau*.npz"))
    if len(paths) != count:
        raise SystemExit(f"{directory}: expected {count} NPZ files, got {len(paths)}")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            if metadata.get("schema_version") != 6:
                raise SystemExit(f"{path}: wrong schema")
            if metadata.get("sample_strategy") != "all_valid_anchor_windows":
                raise SystemExit(f"{path}: sparse sampling is not publication-valid")
            if int(data["n_samples"]) < 700:
                raise SystemExit(f"{path}: insufficient dense windows")
        artifact_count += 1
payload = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "source_sha256": source_sha,
    "artifact_count": artifact_count,
    "run_2021": run_2021,
}
temporary = marker.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, marker)
PY
release_gpu

wait_for_specs "${CORE_12[@]}" "${BASELINES_12[@]}"
verify_source_snapshot
acquire_gpu
echo "[$(date -Is)] acquired physical GPU $GPU for 12h" | tee -a "$LOG"
run_suite 2020 12 1,2,3,4,5,6,7,8,9,10,11 "${CORE_12[@]}" "${BASELINES_12[@]}"
if [[ "$RUN_2021" -eq 1 ]]; then
  run_suite 2021 12 1,2,3,4,5,6,7,8,9,10,11 "${CORE_12[@]}"
fi
release_gpu

"$PY" tools/eval/build_sh_per_channel_table_12h.py \
  --npz-dir "$OUT_ROOT/6h_2020" --ell-min "$ELL_MIN" \
  --taus 1,2,3,4,5 --horizon 6 \
  --out-json "$OUT_ROOT/6h_2020/per_channel_breakdown.json" \
  --out-tex paper/tab_sh_per_channel_6h_v6.tex \
  --label tab:sh_per_channel_6h
"$PY" tools/eval/build_sh_per_channel_table_12h.py \
  --npz-dir "$OUT_ROOT/12h_2020" --ell-min "$ELL_MIN" \
  --taus 1,2,3,4,5,6,7,8,9,10,11 --horizon 12 \
  --out-json "$OUT_ROOT/12h_2020/per_channel_breakdown.json" \
  --out-tex paper/tab_sh_per_channel_12h_v6.tex \
  --label tab:sh_per_channel_12h
"$PY" scripts/export_weatherbridge_spectral_latex_data.py --spectra-only
if command -v pdflatex >/dev/null 2>&1; then
  (
    cd paper/figs
    pdflatex -interaction=nonstopmode -halt-on-error fig_weatherbridge_spectra_tau23.tex
  )
elif [[ -n "$PREBUILT_DETAIL_SPECTRA_PDF" \
  && -s "$PREBUILT_DETAIL_SPECTRA_PDF" ]]; then
  cp "$PREBUILT_DETAIL_SPECTRA_PDF" \
    paper/figs/fig_weatherbridge_spectra_tau23.pdf
else
  echo "pdflatex is unavailable and PREBUILT_DETAIL_SPECTRA_PDF is missing" >&2
  exit 2
fi
cp paper/figs/fig_weatherbridge_spectra_tau23.pdf \
  paper/images/fig_weatherbridge_spectra_tau23.pdf
"$PY" scripts/make_fig3_spectra.py

verify_source_snapshot
"$PY" - "$OUT_ROOT" "$MARKER" "$SOURCE_SHA" "$GPU" "$RUN_2021" \
  "${SOURCE_FILES[@]}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
marker = Path(sys.argv[2])
source_sha = sys.argv[3]
gpu = int(sys.argv[4])
run_2021 = bool(int(sys.argv[5]))
source_files = [Path(value) for value in sys.argv[6:]]
figures = [
    Path("paper/images/fig_weatherbridge_spectra_tau23.pdf"),
    Path("paper/images/fig_spectra_ratio_hard.pdf"),
]
expected = {
    "6h_2020": (9, 5),
    "12h_2020": (9, 11),
}
if run_2021:
    expected.update({"6h_2021": (4, 5), "12h_2021": (4, 11)})

artifacts = []
for directory, (n_models, n_taus) in expected.items():
    paths = sorted((root / directory).glob("*_tau*.npz"))
    if len(paths) != n_models * n_taus:
        raise SystemExit(
            f"{directory}: expected {n_models * n_taus} NPZ files, got {len(paths)}"
        )
    for path in paths:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            if metadata.get("schema_version") != 6:
                raise SystemExit(f"{path}: wrong schema")
            if metadata.get("sht_grid") != "wb2_0p25_2x2_block_average_v1_latitude_strip_area_sht":
                raise SystemExit(f"{path}: wrong SHT grid")
            if metadata.get("sample_strategy") != "all_valid_anchor_windows":
                raise SystemExit(f"{path}: sparse sampling is not publication-valid")
            if int(data["n_samples"]) < 700:
                raise SystemExit(f"{path}: insufficient dense windows")
        artifacts.append(str(path))

for figure in figures:
    if not figure.is_file() or figure.stat().st_size == 0:
        raise SystemExit(f"missing spectral figure: {figure}")

payload = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "source_sha256": source_sha,
    "source_files_sha256": {
        str(source): __import__("hashlib").sha256(
            source.read_bytes()
        ).hexdigest()
        for source in source_files
    },
    "gpu": gpu,
    "lmax": 180,
    "ell_min": 80,
    "artifact_count": len(artifacts),
    "run_2021": run_2021,
    "figure_sha256": {
        figure.name: __import__("hashlib").sha256(figure.read_bytes()).hexdigest()
        for figure in figures
    },
}
temporary = marker.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, marker)
PY
echo "[$(date -Is)] corrected spectral evaluation complete" | tee -a "$LOG"
