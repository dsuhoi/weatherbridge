#!/usr/bin/env bash
# Spectral confirmation of the frozen guarded Flow adapter on the 6h NWP holdout.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
HRES=${HRES:-/tmp/wti_hres_nwp_2022_frozen16_v1}
ERA5=${ERA5:-/tmp/wb2_0p5_nwp_2022}
STATIC=${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}
STATS=${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}
SURFACE_STATS=${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}
ADAPTER_ROOT=${ADAPTER_ROOT:-$RUNTIME/adapters/nwp_blend_6h_v4_endpoint_guard}
RMSE_ROOT=${RMSE_ROOT:-$RUNTIME/metrics/nwp_blend_6h_2022_v4_endpoint_guard}
OUT_ROOT=${OUT_ROOT:-$RUNTIME/metrics/nwp_blend_spectra_6h_2022_v2_endpoint_guard}
GPU_CANDIDATES=${GPU_CANDIDATES:-"0 1"}
MIN_FREE_MIB=${MIN_FREE_MIB:-50000}
MAX_UTIL=${MAX_UTIL:-10}

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
exec 9>"$OUT_ROOT/launch.lock"
if ! flock -n 9; then
  echo "2022 NWP spectral queue already active"
  exit 0
fi
LOG="$LOG_ROOT/nwp_blend_spectra_6h_2022_v2_endpoint_guard.log"
exec > >(tee -a "$LOG") 2>&1

SOURCE_FILES=(
  scripts/run_nwp_blend_spectra_2022_cloudru.sh
  tools/eval/nwp_blend_spectra.py
  tools/eval/batch_eval_forecast_anchor.py
  tools/eval/sh_energy_spectra_12h.py
  tools/eval/spectral_block_bootstrap.py
  tools/eval/capmatched_loader.py
  tools/train/train_capacity_matched_6h.py
  weather_time_interp/metrics/spherical_spectra.py
  weather_time_interp/model/weatherbridge_flow_model.py
  weather_time_interp/normalization.py
  weather_time_interp/grid.py
  repro/nwp_forecast_holdout_2022.json
)
SOURCE_SHA=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
verify_source() {
  local current
  current=$(sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')
  [[ "$current" == "$SOURCE_SHA" ]] || {
    echo "source changed during 2022 NWP spectral evaluation" >&2
    exit 2
  }
}

until [[ -s "$RMSE_ROOT/complete.json" && -s "$ERA5/wb2_2022.json" ]]; do
  echo "[$(date -Is)] wait independent 6h RMSE holdout"
  sleep 60
done

ADAPTER="$ADAPTER_ROOT/flow.json"
CHECKPOINT="$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt"
"$PY" - "$RMSE_ROOT/complete.json" "$ADAPTER_ROOT/selection.json" <<'PY'
import json
import sys
from pathlib import Path

complete = json.loads(Path(sys.argv[1]).read_text())
selection = json.loads(Path(sys.argv[2]).read_text())
if (
    complete.get("status") != "complete"
    or complete.get("selection_independent") is not True
    or complete.get("test_split") != "2022_independent"
    or selection.get("winner") != "flow"
    or 2022 not in selection.get("confirmatory_years_unopened", [])
):
    raise SystemExit("independent 2022 RMSE gate is not valid")
PY
verify_source

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
        return 0
      fi
      flock -u "$candidate_fd"
      exec {candidate_fd}>&-
    done
    sleep 60
  done
}

release_gpu() {
  flock -u "$GPU_LOCK_FD"
  exec {GPU_LOCK_FD}>&-
}

if [[ ! -s "$OUT_ROOT/spectra_complete.json" ]]; then
  acquire_gpu
  echo "[$(date -Is)] use physical GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u tools/eval/nwp_blend_spectra.py \
    --forecast-dir "$HRES" --era5-memmap-dir "$ERA5" \
    --checkpoint "$CHECKPOINT" --arch flow_pp3 --blend-adapter "$ADAPTER" \
    --out-dir "$OUT_ROOT" --stats-path "$STATS" \
    --surface-stats-path "$SURFACE_STATS" --static-path "$STATIC" \
    --device cuda:0 --delta-t-hours 6 --taus 2 3 --channels Q850 U850 \
    --lmax 180 --hf-ell-min 80 --max-inits 16 --tau-batch-size 2
  release_gpu
fi
verify_source

for tau in 2 3; do
  "$PY" tools/eval/spectral_block_bootstrap.py \
    --left "flow_adapted:$OUT_ROOT/flow_adapted_tau${tau}.npz" \
    --right "linear:$OUT_ROOT/linear_tau${tau}.npz" \
    --channels all --block-days 7 --draws 10000 --seed 2027 --cellwise \
    --out-json "$OUT_ROOT/flow_adapted_vs_linear_tau${tau}.json"
done
verify_source

"$PY" - "$OUT_ROOT" "$SOURCE_SHA" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = [
    "spectra_complete.json",
    "linear_tau2.npz",
    "linear_tau3.npz",
    "flow_adapted_tau2.npz",
    "flow_adapted_tau3.npz",
    "flow_adapted_vs_linear_tau2.json",
    "flow_adapted_vs_linear_tau3.json",
]
paths = [root / name for name in names]
if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
    raise SystemExit("missing NWP spectral artifact")
comparisons = {
    str(tau): json.loads((root / f"flow_adapted_vs_linear_tau{tau}.json").read_text())
    for tau in (2, 3)
}
payload = {
    "schema_version": 1,
    "status": "complete",
    "source_sha256": sys.argv[2],
    "task": "primary_6h_NWP_anchor_to_ERA5_interpolation",
    "channels": ["Q850", "U850"],
    "taus": [2, 3],
    "comparisons": comparisons,
    "artifacts": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    },
}
temporary = root / ".complete.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "complete.json")
print(json.dumps(payload, indent=2))
PY
echo "[$(date -Is)] 2022 NWP spectral confirmation complete"
