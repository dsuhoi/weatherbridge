#!/usr/bin/env bash
set -euo pipefail

SOURCE="${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
MAX_UTIL="${MAX_UTIL:-5}"
LOG_DIR="${LOG_DIR:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MARKER="${MARKER:-paper/images/.weatherbridge_rmse_maps_complete}"
FLOW_BARE="${FLOW_BARE:-$RUNTIME_ROOT/weights/detailed_benchmark_v2/weatherbridge_flow_spectral_6h_bare.pt}"
DCAE_CKPT="${DCAE_CKPT:-$LOG_DIR/exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4/last.ckpt}"
LEGACY_LOG_ROOT="${LEGACY_LOG_ROOT:-/home/jovyan/dsuhoi/weather_time_interpolation/logs}"
FUXI_CKPT="${FUXI_CKPT:-$LEGACY_LOG_ROOT/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt}"
SDYFF_CKPT="${SDYFF_CKPT:-$LEGACY_LOG_ROOT/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt}"
ATMVFI_CKPT="${ATMVFI_CKPT:-$LOG_DIR/exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_s202707_protocol_v2/last.ckpt}"
CHAMPION="${CHAMPION:-$SOURCE/metrics/journal_champion_v1/final.json}"
CHAMPION_MARKER="${CHAMPION_MARKER:-$SOURCE/metrics/journal_champion_v1/state/.complete}"

mkdir -p "$LOG_DIR"
cd "$SOURCE"
while pgrep -f '[r]un_weatherbridge_downstream_cloudru.sh' >/dev/null; do
  sleep 60
done
for path in \
  "$FLOW_BARE" "$DCAE_CKPT" \
  "$FUXI_CKPT" "$SDYFF_CKPT" "$ATMVFI_CKPT" \
  "$CHAMPION" "$CHAMPION_MARKER"; do
  while [[ ! -s "$path" ]]; do
    echo "[$(date -Is)] waiting for $path"
    sleep "$POLL_SECONDS"
  done
done
"$PYTHON_BIN" - "$CHAMPION" "$CHAMPION_MARKER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

result = Path(sys.argv[1])
marker = json.loads(Path(sys.argv[2]).read_text())
payload = json.loads(result.read_text())
if (
    marker.get("status") != "complete"
    or not marker.get("winner")
    or payload.get("winner") != marker.get("winner")
    or payload.get("status") not in {"confirmed", "reference_retained"}
):
    raise SystemExit("journal selector is not final")
if marker.get("result_sha256") != hashlib.sha256(result.read_bytes()).hexdigest():
    raise SystemExit("champion result hash mismatch")
PY

export PYTHONPATH="$EXTRA_PYTHONPATH:$SOURCE:${PYTHONPATH:-}"
export RMSE_MAX_SAMPLES="${RMSE_MAX_SAMPLES:-96}"
export SDYFF_NLAT="${SDYFF_NLAT:-360}"
export SDYFF_NLON="${SDYFF_NLON:-720}"
export SDYFF_LAT_CROP="${SDYFF_LAT_CROP:-0}"

GPU=""
GPU_LOCK_FD=""
while [[ -z "$GPU" ]]; do
  for candidate in $GPU_CANDIDATES; do
    exec {candidate_fd}>"$LOG_DIR/.upr_lite_gpu${candidate}.lock"
    if ! flock -n "$candidate_fd"; then
      exec {candidate_fd}>&-
      continue
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits -i "$candidate" | tr -d ' ')"
    if [[ "$free_mib" -ge "$MIN_FREE_MIB" && "$util" -le "$MAX_UTIL" ]]; then
      GPU="$candidate"
      GPU_LOCK_FD="$candidate_fd"
      break
    fi
    flock -u "$candidate_fd"
    exec {candidate_fd}>&-
  done
  if [[ -z "$GPU" ]]; then
    sleep "$POLL_SECONDS"
  fi
done

exec 9>/tmp/weatherbridge_journal_eval_6h.lock
flock 9

echo "[$(date -Is)] WeatherBridge RMSE-map queue started on physical GPU $GPU"
export CUDA_VISIBLE_DEVICES="$GPU"
export RMSE_FORCE="${RMSE_FORCE:-0}"
export BARE_FLOW_SPECTRAL="$FLOW_BARE"
export CKPT_WEATHERDCAE="$DCAE_CKPT"
export CKPT_FUXI="$FUXI_CKPT"
export CKPT_SDYFF="$SDYFF_CKPT"
export CKPT_ATMVFI="$ATMVFI_CKPT"
"$PYTHON_BIN" -u scripts/make_fig_rmse_maps_5tau_per_field.py
"$PYTHON_BIN" - "$MARKER" "$GPU" "$FLOW_BARE" \
  "$DCAE_CKPT" "$FUXI_CKPT" "$SDYFF_CKPT" "$ATMVFI_CKPT" \
  "$CHAMPION" "$CHAMPION_MARKER" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
repo = Path.cwd()
artifacts = [Path(value) for value in sys.argv[3:8]]
champion = Path(sys.argv[8])
champion_marker = Path(sys.argv[9])
selection = json.loads(champion.read_text())
figures = [
    repo / "paper" / "images" / f"fig_rmse_maps_{field}_5tau.pdf"
    for field in ("t2m", "u10", "mslp", "Q1000")
]
for artifact in artifacts + figures:
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise SystemExit(f"missing RMSE-map dependency: {artifact}")
payload = {
    "schema_version": 2,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "gpu": int(sys.argv[2]),
    "central_model": "WeatherBridge",
    "internal_arch": "flow_pp3",
    "selected_model": selection["winner"],
    "selector_status": selection["status"],
    "color": "#D62728",
    "champion_sha256": hashlib.sha256(champion.read_bytes()).hexdigest(),
    "champion_marker_sha256": hashlib.sha256(
        champion_marker.read_bytes()
    ).hexdigest(),
    "artifact_sha256": {
        str(artifact): hashlib.sha256(artifact.read_bytes()).hexdigest()
        for artifact in artifacts
    },
    "figure_sha256": {
        figure.name: hashlib.sha256(figure.read_bytes()).hexdigest()
        for figure in figures
    },
    "source_sha256": {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in (
            repo / "scripts" / "run_weatherbridge_rmse_maps_cloudru.sh",
            repo / "scripts" / "make_fig_rmse_maps_5tau_per_field.py",
            repo / "trainer_weather_hermite.py",
            repo / "legacy" / "scripts" / "trainer_weather_hermite.py",
            repo / "examples" / "_bare_loader.py",
            repo / "tools" / "eval" / "batch_eval_12h_memmap.py",
            repo / "tools" / "eval" / "capmatched_loader.py",
            repo / "tools" / "train" / "train_capacity_matched_6h.py",
            repo / "weather_time_interp" / "model" / "weatherbridge_flow_model.py",
            repo / "weather_time_interp" / "model" / "dcae_adaln_model.py",
            repo / "weather_time_interp" / "model" / "dcae.py",
        )
    },
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
echo "[$(date -Is)] WeatherBridge RMSE-map queue complete"
flock -u "$GPU_LOCK_FD"
