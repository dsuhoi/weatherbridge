#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
RUNTIME="${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"
ROOT="$RUNTIME/metrics/detailed_benchmark_v2"
OUT="$ROOT/report"
CANONICAL="$RUNTIME/metrics/journal_unified"
PAPER_ROOT="$PWD/paper"
LOG="${LOG:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs/detailed_benchmark_v2_report.log}"
SOURCE_ARCHIVE_ROOT="${SOURCE_ARCHIVE_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi}"
SOURCE_FILES=(
  scripts/run_detailed_benchmark_report_queue_cloudru.sh
  tools/eval/export_main_metrics_tex.py
  tools/eval/export_downstream_metrics_tex.py
  scripts/export_weatherbridge_spectral_latex_data.py
  scripts/make_fig1_rmse_per_tau.py
  scripts/make_fig2_acc_per_tau.py
  scripts/make_fig_main_scores.py
  scripts/make_fig_specific_channels_phys.py
  scripts/paper_plot_style.py
  tools/eval/align_window_artifact.py
  tools/eval/paired_block_bootstrap.py
  tools/repro/materialize_completion_source_snapshot.py
  weather_time_interp/normalization.py
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
    echo "source changed while report queue was active: $current_sha != $SOURCE_SHA" >&2
    exit 2
  fi
}

exec 8>"${LOG}.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] detailed report queue already active"
  exit 0
fi
exec > >(tee -a "$LOG") 2>&1
wait_for_source_snapshot() {
  local label="$1"
  local marker="$2"
  until [[ -s "$marker" ]] && \
    "$PY" tools/repro/materialize_completion_source_snapshot.py \
      --marker "$marker" \
      --search-root "$SOURCE_ARCHIVE_ROOT"; do
    echo "[$(date -Is)] waiting for $label"
    sleep 120
  done
}

wait_for_source_snapshot "validated 6h detailed benchmark" "$ROOT/6h/.complete"
wait_for_source_snapshot "validated 12h detailed benchmark" "$ROOT/12h/.complete"
wait_for_source_snapshot \
  "corrected baseline geometry" \
  "$CANONICAL/.baseline_geometry_v1_complete"
verify_source_snapshot
# Publish only protocol-compatible full-year WeatherBridge metrics into the
# canonical figure inputs. This prevents the 192-window interim audit from
# being mixed with full-year baseline curves.
"$PY" - "$ROOT" "$CANONICAL" <<'PY'
import json
import sys
from pathlib import Path

from tools.eval.align_window_artifact import align_artifact

root = Path(sys.argv[1])
canonical = Path(sys.argv[2])
specs = (
    (6, 2020, "fuxi_24ch_6yr_ep8.json", "flow_spectral", "weatherbridge_pp3_14m_6yr_ep8.json"),
    (6, 2020, "fuxi_24ch_6yr_ep8.json", "weatherdcae_14m_6yr", "weatherdcae_14m_6yr_ep8_matched.json"),
    (6, 2021, "fuxi_24ch_6yr_ep8.json", "flow_spectral", "weatherbridge_pp3_14m_6yr_ep8.json"),
    (6, 2021, "fuxi_24ch_6yr_ep8.json", "weatherdcae_14m_6yr", "weatherdcae_14m_6yr_ep8_matched.json"),
    (12, 2020, "fuxi_3yr_ep10.json", "flow_spectral", "weatherbridge_14m_3yr_ep10.json"),
    (12, 2020, "fuxi_3yr_ep10.json", "weatherdcae_14m", "weatherdcae_14m_3yr_ep10_matched.json"),
)
for horizon, year, reference_name, source_name, output_name in specs:
    source = root / f"{horizon}h" / str(year) / "full_year" / f"{source_name}.json"
    target_dir = canonical / f"{horizon}h_{year}"
    reference = target_dir / reference_name
    payload = json.loads(source.read_text())
    reference_payload = json.loads(reference.read_text())
    protocol = payload.get("evaluation_protocol", {})
    reference_protocol = reference_payload.get("evaluation_protocol", {})
    expected_taus = list(range(1, horizon))
    if payload.get("years") != [year]:
        raise ValueError(f"wrong test year in {source}: {payload.get('years')}")
    if float(payload.get("delta_t_hours", -1)) != float(horizon):
        raise ValueError(f"wrong horizon in {source}")
    if protocol.get("full_year") is not True:
        raise ValueError(f"refusing non-full-year metric {source}")
    if protocol.get("eval_hours") != expected_taus:
        raise ValueError(f"wrong tau grid in {source}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / output_name
    output_window = target_dir / "window_metrics" / f"{target.stem}.npz"
    aligned = align_artifact(
        source_json=source,
        reference_json=reference,
        output_json=target,
        output_window=output_window,
    )
    if aligned["evaluation_protocol"]["index_sha256"] != reference_protocol.get(
        "index_sha256"
    ):
        raise ValueError(f"aligned window index mismatch: {target} vs {reference}")
    print(f"published aligned {target}")
PY

"$PY" tools/eval/export_main_metrics_tex.py \
  --metrics-root "$CANONICAL" \
  --out-tex "$PAPER_ROOT/main_metrics_v2.tex" \
  --out-table "$PAPER_ROOT/tab_main_12h_v2.tex"
"$PY" tools/eval/export_downstream_metrics_tex.py \
  --root "$ROOT" \
  --physics-tex "$PAPER_ROOT/tab_downstream_physics_v2.tex" \
  --diurnal-tex "$PAPER_ROOT/tab_downstream_diurnal_v2.tex"
"$PY" scripts/export_weatherbridge_spectral_latex_data.py
export WTI_JOURNAL_METRICS_ROOT="$CANONICAL"
"$PY" scripts/make_fig1_rmse_per_tau.py
"$PY" scripts/make_fig2_acc_per_tau.py
"$PY" scripts/make_fig_main_scores.py
"$PY" scripts/make_fig_specific_channels_phys.py --mode all

figures=(
  fig1_rmse_per_tau.pdf
  fig2_acc_per_tau.pdf
  fig_main_scores.pdf
  fig_channels_body_6h_phys.pdf
  fig_channels_app_6h_phys.pdf
  fig_channels_body_12h_phys.pdf
  fig_channels_app_12h_phys.pdf
  fig_channels_body_ood2021.pdf
  fig_channels_app_ood2021.pdf
)
for figure in "${figures[@]}"; do
  path="$PAPER_ROOT/images/$figure"
  test -s "$path"
  pdftotext "$path" - | grep -q 'WeatherBridge'
  if pdftotext "$path" - | grep -Eqi \
    'Flow-Spectral|WeatherBridge-Detail|Detail-bypass ablation'; then
    echo "legacy model name remains in $figure" >&2
    exit 1
  fi
done

verify_source_snapshot
"$PY" - "$PAPER_ROOT/images/.weatherbridge_full_year_figures_complete" \
  "$ROOT" "$CANONICAL" "$SOURCE_SHA" \
  "${figures[@]/#/$PAPER_ROOT/images/}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
root = Path(sys.argv[2])
canonical = Path(sys.argv[3])
source_sha = sys.argv[4]
figures = [Path(value) for value in sys.argv[5:]]
paper_root = path.parent.parent
metrics_tex = paper_root / "main_metrics_v2.tex"
main_table = paper_root / "tab_main_12h_v2.tex"
physics_table = paper_root / "tab_downstream_physics_v2.tex"
diurnal_table = paper_root / "tab_downstream_diurnal_v2.tex"
repo = paper_root.parent
source_files = (
    repo / "scripts/run_detailed_benchmark_report_queue_cloudru.sh",
    repo / "tools/eval/export_main_metrics_tex.py",
    repo / "tools/eval/export_downstream_metrics_tex.py",
    repo / "scripts/export_weatherbridge_spectral_latex_data.py",
    repo / "scripts/make_fig1_rmse_per_tau.py",
    repo / "scripts/make_fig2_acc_per_tau.py",
    repo / "scripts/make_fig_main_scores.py",
    repo / "scripts/make_fig_specific_channels_phys.py",
    repo / "scripts/paper_plot_style.py",
    repo / "tools/eval/align_window_artifact.py",
    repo / "tools/eval/paired_block_bootstrap.py",
    repo / "tools/repro/materialize_completion_source_snapshot.py",
    repo / "weather_time_interp/normalization.py",
)
input_files = [
    root / "6h/.complete",
    root / "12h/.complete",
    canonical / ".baseline_geometry_v1_complete",
]
for horizon, names in (
    (6, ("flow_spectral", "weatherdcae_14m_6yr", "linear")),
    (12, ("flow_spectral", "weatherdcae_14m", "linear")),
):
    for name in names:
        input_files.extend(
            (
                root / f"{horizon}h/2020/physics_{name}.json",
                root / f"{horizon}h/2020/diurnal_{name}.json",
            )
        )
for directory in ("6h_2020", "6h_2021", "12h_2020"):
    input_files.extend(
        path
        for path in sorted((canonical / directory).glob("*.json"))
        if "weatherbridge_detail" not in path.name
        and "weatherbridge_refine" not in path.name
    )
if len(input_files) < 12:
    raise SystemExit("incomplete full-year figure input set")
for dependency in (
    *source_files,
    *input_files,
    metrics_tex,
    main_table,
    physics_table,
    diurnal_table,
    *figures,
):
    if not dependency.is_file() or dependency.stat().st_size == 0:
        raise SystemExit(f"missing full-year figure dependency: {dependency}")
payload = {
    "schema_version": 4,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "central_model": "WeatherBridge",
    "internal_arch": "flow_pp3",
    "color": "#D62728",
    "source_sha256": source_sha,
    "source_files_sha256": {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in source_files
    },
    "input_sha256": {
        str(dependency): hashlib.sha256(dependency.read_bytes()).hexdigest()
        for dependency in input_files
    },
    "figure_sha256": {
        figure.name: hashlib.sha256(figure.read_bytes()).hexdigest()
        for figure in figures
    },
    "main_metrics_sha256": hashlib.sha256(metrics_tex.read_bytes()).hexdigest(),
    "main_table_sha256": hashlib.sha256(main_table.read_bytes()).hexdigest(),
    "physics_table_sha256": hashlib.sha256(
        physics_table.read_bytes()
    ).hexdigest(),
    "diurnal_table_sha256": hashlib.sha256(
        diurnal_table.read_bytes()
    ).hexdigest(),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
echo "[$(date -Is)] detailed benchmark report complete"
