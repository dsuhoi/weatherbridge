#!/usr/bin/env bash
set -euo pipefail

SOURCE="${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-120}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1}"
CHECKPOINT="${CHECKPOINT:-$LOG_ROOT/exp_flow_pp3_spectral_14m_6h_refinev1_s202707_bs4/last.ckpt}"
CHAMPION="metrics/journal_champion_v1/final.json"
CHAMPION_MARKER="metrics/journal_champion_v1/state/.complete"
MARKER="paper/images/.weatherbridge_case_figures_complete"
LOG="$LOG_ROOT/weatherbridge_case_figures.log"

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
SOURCE_FILES=(
  scripts/run_weatherbridge_case_figures_cloudru.sh
  tools/eval/export_weatherbridge_cases.py
  tools/eval/export_case_metrics_tex.py
  scripts/make_fig_case_ida_wind.py
  scripts/make_fig5_haishen_local.py
  scripts/paper_plot_style.py
  tools/eval/capmatched_loader.py
  tools/train/train_capacity_matched_6h.py
  weather_time_interp/model/weatherbridge_flow_model.py
)

compute_source_sha() {
  sha256sum "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}'
}

SOURCE_SHA=$(compute_source_sha)
if [[ -n "${EXPECTED_SOURCE_SHA:-}" && "$SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]]; then
  echo "source hash mismatch: $SOURCE_SHA != $EXPECTED_SOURCE_SHA" >&2
  exit 2
fi

verify_source_snapshot() {
  local current_sha
  current_sha=$(compute_source_sha)
  if [[ "$current_sha" != "$SOURCE_SHA" ]]; then
    echo "source changed while case queue was active: $current_sha != $SOURCE_SHA" >&2
    exit 2
  fi
}
exec 8>"$LOG_ROOT/.weatherbridge_case_figures.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] case-figure queue already active"
  exit 0
fi
exec > >(tee -a "$LOG") 2>&1

for path in "$CHECKPOINT" "$CHAMPION" "$CHAMPION_MARKER"; do
  while [[ ! -s "$path" ]]; do
    echo "[$(date -Is)] waiting for $path"
    sleep "$POLL_SECONDS"
  done
done
verify_source_snapshot
"$PY" - "$CHAMPION" "$CHAMPION_MARKER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

result = Path(sys.argv[1])
marker = json.loads(Path(sys.argv[2]).read_text())
payload = json.loads(result.read_text())
if marker.get("status") != "complete" or not marker.get("winner"):
    raise SystemExit("journal selector is not final")
if marker.get("result_sha256") != hashlib.sha256(result.read_bytes()).hexdigest():
    raise SystemExit("champion result hash mismatch")
if payload.get("winner") != marker.get("winner"):
    raise SystemExit("selector result and marker disagree")
if payload.get("status") not in {"confirmed", "reference_retained"}:
    raise SystemExit("journal selector has no final decision")
PY

GPU=""
while [[ -z "$GPU" ]]; do
  for candidate in $GPU_CANDIDATES; do
    exec {candidate_fd}>"$LOG_ROOT/.upr_lite_gpu${candidate}.lock"
    if flock -n "$candidate_fd"; then
      GPU="$candidate"
      GPU_LOCK_FD="$candidate_fd"
      break
    fi
    exec {candidate_fd}>&-
  done
  [[ -n "$GPU" ]] || sleep "$POLL_SECONDS"
done
export CUDA_VISIBLE_DEVICES="$GPU"
"$PY" tools/eval/export_weatherbridge_cases.py \
  --checkpoint "$CHECKPOINT" --model-label WeatherBridge \
  --expected-arch flow_pp3 \
  --output-root metrics/case_studies_weatherbridge \
  --device cuda:0
flock -u "$GPU_LOCK_FD"
exec {GPU_LOCK_FD}>&-
verify_source_snapshot

PYTHONPATH=.:scripts "$PY" scripts/make_fig_case_ida_wind.py
PYTHONPATH=.:scripts "$PY" scripts/make_fig5_haishen_local.py
"$PY" tools/eval/export_case_metrics_tex.py \
  --root . --weatherbridge-checkpoint "$CHECKPOINT" \
  --out-tex paper/case_metrics_v2.tex
verify_source_snapshot

"$PY" - "$MARKER" "$CHECKPOINT" \
  "$CHAMPION" "$CHAMPION_MARKER" "$SOURCE_SHA" "${SOURCE_FILES[@]}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

marker = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
champion = Path(sys.argv[3])
champion_marker = Path(sys.argv[4])
source_sha = sys.argv[5]
source_files = [Path(value) for value in sys.argv[6:]]
root = Path.cwd()
selection = json.loads(champion.read_text())


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


cases = [
    root / "demo/precomputed/typhoon_haishen/East_Asia__mslp.npz",
    root / "metrics/case_studies_weatherbridge/typhoon_haishen/mslp.npz",
    root / "demo/precomputed_2021/hurricane_ida_2021/North_America__u10.npz",
    root / "demo/precomputed_2021/hurricane_ida_2021/North_America__v10.npz",
    root / "metrics/case_studies_weatherbridge/hurricane_ida_2021/u10.npz",
    root / "metrics/case_studies_weatherbridge/hurricane_ida_2021/v10.npz",
]
summaries = [
    root / "metrics/case_studies_weatherbridge/typhoon_haishen/mslp_summary.json",
    root / "metrics/case_studies_weatherbridge/hurricane_ida_2021/wind_speed_summary.json",
]
metrics_tex = root / "paper/case_metrics_v2.tex"
figures = [
    root / "paper/images/fig_haishen_mslp.pdf",
    root / "paper/images/fig_case_ida_wind.pdf",
]
checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
transport_cases = {
    root / "metrics/case_studies_weatherbridge/typhoon_haishen/mslp.npz": (
        "flow_pp3", checkpoint_sha
    ),
    root / "metrics/case_studies_weatherbridge/hurricane_ida_2021/u10.npz": (
        "flow_pp3", checkpoint_sha
    ),
    root / "metrics/case_studies_weatherbridge/hurricane_ida_2021/v10.npz": (
        "flow_pp3", checkpoint_sha
    ),
}
for case, (arch, expected_checkpoint_sha) in transport_cases.items():
    with np.load(case, allow_pickle=False) as data:
        provenance = json.loads(data["provenance_json"].item())
        if len(data["methods"].tolist()) != 1:
            raise SystemExit(f"wrong model count in {case}")
        if data["model_arch"].item() != arch:
            raise SystemExit(f"wrong architecture in {case}")
        if provenance.get("checkpoint", {}).get("sha256") != expected_checkpoint_sha:
            raise SystemExit(f"checkpoint hash mismatch in {case}")
        if (
            provenance.get("schema_version") != 3
            or provenance.get("grid_convention")
            != "wb2_0p25_pair_average_cell_centres_v1"
        ):
            raise SystemExit(f"stale grid provenance in {case}")
        if provenance.get("latitude_sha256") != array_sha256(data["lat"]):
            raise SystemExit(f"latitude hash mismatch in {case}")
        if provenance.get("longitude_sha256") != array_sha256(data["lon"]):
            raise SystemExit(f"longitude hash mismatch in {case}")
        if provenance.get("target_crop_sha256") != array_sha256(data["truth"]):
            raise SystemExit(f"target hash mismatch in {case}")
        if "typhoon_haishen" in case.parts:
            expected_lat = np.arange(54.875, 19.874, -0.5, dtype=np.float32)
            expected_lon = np.arange(100.125, 145.126, 0.5, dtype=np.float32)
        else:
            expected_lat = np.arange(59.875, 19.874, -0.5, dtype=np.float32)
            expected_lon = np.arange(230.125, 300.126, 0.5, dtype=np.float32)
        if not np.array_equal(data["lat"], expected_lat):
            raise SystemExit(f"wrong latitude centres in {case}")
        if not np.array_equal(data["lon"], expected_lon):
            raise SystemExit(f"wrong longitude centres in {case}")
for figure in figures:
    text = subprocess.check_output(["pdftotext", str(figure), "-"]).decode()
    if "WeatherBridge" not in text:
        raise SystemExit(f"wrong model identity in {figure}")
payload = {
    "schema_version": 3,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "central_model": "WeatherBridge",
    "internal_arch": "flow_pp3",
    "selected_model": selection["winner"],
    "selector_status": selection["status"],
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": checkpoint_sha,
    "champion_sha256": hashlib.sha256(champion.read_bytes()).hexdigest(),
    "champion_marker_sha256": hashlib.sha256(champion_marker.read_bytes()).hexdigest(),
    "case_sha256": {str(case.relative_to(root)): hashlib.sha256(case.read_bytes()).hexdigest() for case in cases},
    "summary_sha256": {str(summary.relative_to(root)): hashlib.sha256(summary.read_bytes()).hexdigest() for summary in summaries},
    "metrics_tex_sha256": hashlib.sha256(metrics_tex.read_bytes()).hexdigest(),
    "figure_sha256": {figure.name: hashlib.sha256(figure.read_bytes()).hexdigest() for figure in figures},
    "source_composite_sha256": source_sha,
    "source_sha256": {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in source_files
    },
}
temporary = marker.with_suffix(marker.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, marker)
PY
echo "[$(date -Is)] WeatherBridge case figures complete"
