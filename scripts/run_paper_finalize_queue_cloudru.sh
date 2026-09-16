#!/usr/bin/env bash
set -euo pipefail

RUNTIME="${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}"
METRICS_RUNTIME="${METRICS_RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"
PY="${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-120}"
LOG="${LOG:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs/paper_finalize.log}"
MARKER="${MARKER:-paper/.journal_figures_verified}"
POST_VERIFICATION="${POST_VERIFICATION:-/home/jovyan/shares/SR006.nfs2/dsuhoi/postselection_2022_memmap/wb2_2022.verification_v2.json}"

cd "$RUNTIME"
mkdir -p "$(dirname "$LOG")"
SOURCE_FILES=(
  scripts/run_paper_finalize_queue_cloudru.sh
  tools/eval/export_journal_statistics.py
  tools/eval/assess_postselection_holdout.py
  tools/eval/export_postselection_tex.py
  tools/repro/export_postselection_supplementary_data.py
  tools/repro/preflight_npj_submission.py
  tools/repro/package_npj_submission.py
  paper/build_npj_submission.sh
  paper/main.tex
  paper/supplementary.tex
  paper/references.bib
  paper/sn-jnl.cls
  paper/sn-nature.bst
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
    echo "source changed while finalizer was active: $current_sha != $SOURCE_SHA" >&2
    exit 2
  fi
}

exec 8>"${LOG}.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] paper finalizer already active"
  exit 0
fi
exec > >(tee -a "$LOG") 2>&1

wait_complete() {
  local path="$1"
  until "$PY" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
raise SystemExit(0 if json.loads(path.read_text()).get("status") == "complete" else 1)
PY
  do
    echo "[$(date -Is)] waiting for $path"
    sleep "$POLL_SECONDS"
  done
}

wait_complete paper/images/.weatherbridge_full_year_figures_complete
wait_complete paper/images/.weatherbridge_rmse_maps_complete
wait_complete paper/images/.weatherbridge_case_figures_complete
wait_complete paper/images/.hres_field_figures_complete
wait_complete metrics/journal_spectra_v6/state/.complete
wait_complete metrics/journal_champion_v1/state/.complete
wait_complete "$METRICS_RUNTIME/metrics/journal_unified/.baseline_geometry_v1_complete"
wait_complete "$METRICS_RUNTIME/metrics/postselection_2022_v1/state/.complete"
verify_source_snapshot

"$PY" tools/eval/assess_postselection_holdout.py \
  --manifest repro/postselection_holdout_2022.json \
  --champion metrics/journal_champion_v1/final.json \
  --verification "$POST_VERIFICATION" \
  --evaluation-root "$METRICS_RUNTIME/metrics/postselection_2022_v1" \
  --candidate flow_spectral \
  --out-json "$METRICS_RUNTIME/metrics/postselection_2022_v1/weatherbridge_assessment.json"
"$PY" tools/eval/export_postselection_tex.py \
  --assessment "$METRICS_RUNTIME/metrics/postselection_2022_v1/weatherbridge_assessment.json" \
  --out-tex paper/postselection_2022.tex
verify_source_snapshot

"$PY" - "$METRICS_RUNTIME" "$POST_VERIFICATION" <<'PY'
import hashlib
import json
from pathlib import Path

from tools.eval.export_postselection_tex import render as render_postselection_tex


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def relative_source_set(values) -> set[str]:
    root = Path.cwd().resolve()
    result = set()
    for value in values:
        path = Path(value)
        if path.is_absolute():
            path = path.resolve().relative_to(root)
        result.add(path.as_posix())
    return result


def verify_historical_source(
    path_string: str,
    expected: str,
    *,
    snapshot_roots: tuple[Path, ...],
    label: str,
) -> None:
    path = Path(path_string)
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return

    root = Path.cwd().resolve()
    try:
        relative = path.resolve().relative_to(root) if path.is_absolute() else path
    except ValueError:
        relative = Path(*path.parts[1:])
    candidates = [snapshot / relative for snapshot in snapshot_roots]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if existing and all(
        hashlib.sha256(candidate.read_bytes()).hexdigest() == expected
        for candidate in existing
    ):
        return
    raise SystemExit(f"{label} hash mismatch: {path}")


metrics_runtime = Path(__import__("sys").argv[1])
post_verification_path = Path(__import__("sys").argv[2])
benchmark_snapshots = tuple(
    metrics_runtime / f"metrics/detailed_benchmark_v2/{horizon}/source_snapshot"
    for horizon in ("6h", "12h")
)

full = load("paper/images/.weatherbridge_full_year_figures_complete")
if (
    full.get("schema_version") != 4
    or full.get("central_model") != "WeatherBridge"
    or full.get("internal_arch") != "flow_pp3"
    or full.get("color") != "#D62728"
):
    raise SystemExit("invalid full-year figure marker")
expected_full_sources = {
    "scripts/run_detailed_benchmark_report_queue_cloudru.sh",
    "tools/eval/export_main_metrics_tex.py",
    "tools/eval/export_downstream_metrics_tex.py",
    "scripts/export_weatherbridge_spectral_latex_data.py",
    "scripts/make_fig1_rmse_per_tau.py",
    "scripts/make_fig2_acc_per_tau.py",
    "scripts/make_fig_main_scores.py",
    "scripts/make_fig_specific_channels_phys.py",
    "scripts/paper_plot_style.py",
    "tools/eval/align_window_artifact.py",
    "tools/eval/paired_block_bootstrap.py",
    "tools/repro/materialize_completion_source_snapshot.py",
    "weather_time_interp/normalization.py",
}
if relative_source_set(full.get("source_files_sha256", {})) != expected_full_sources:
    raise SystemExit("full-year marker has an unexpected source-file set")
if len(full.get("input_sha256", {})) < 28:
    raise SystemExit("full-year marker has an incomplete metric input set")
for collection_name in ("source_files_sha256", "input_sha256"):
    for path_string, expected in full[collection_name].items():
        path = Path(path_string)
        if collection_name == "source_files_sha256":
            verify_historical_source(
                path_string,
                expected,
                snapshot_roots=benchmark_snapshots,
                label=f"full-year {collection_name}",
            )
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"full-year {collection_name} hash mismatch: {path}")
figure_hashes = full.get("figure_sha256", {})
if len(figure_hashes) != 9:
    raise SystemExit("full-year marker must bind nine figures")
for name, expected in figure_hashes.items():
    path = Path("paper/images") / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"figure hash mismatch: {path}")
metrics_tex = Path("paper/main_metrics_v2.tex")
if hashlib.sha256(metrics_tex.read_bytes()).hexdigest() != full.get(
    "main_metrics_sha256"
):
    raise SystemExit("headline metric macro hash mismatch")
main_table = Path("paper/tab_main_12h_v2.tex")
if hashlib.sha256(main_table.read_bytes()).hexdigest() != full.get(
    "main_table_sha256"
):
    raise SystemExit("main 12h table hash mismatch")
physics_table = Path("paper/tab_downstream_physics_v2.tex")
if hashlib.sha256(physics_table.read_bytes()).hexdigest() != full.get(
    "physics_table_sha256"
):
    raise SystemExit("physical-diagnostic table hash mismatch")
diurnal_table = Path("paper/tab_downstream_diurnal_v2.tex")
if hashlib.sha256(diurnal_table.read_bytes()).hexdigest() != full.get(
    "diurnal_table_sha256"
):
    raise SystemExit("diurnal-diagnostic table hash mismatch")

maps = load("paper/images/.weatherbridge_rmse_maps_complete")
if (
    maps.get("schema_version") != 2
    or len(maps.get("figure_sha256", {})) != 4
    or maps.get("central_model") != "WeatherBridge"
    or maps.get("internal_arch") != "flow_pp3"
    or maps.get("color") != "#D62728"
):
    raise SystemExit("invalid RMSE-map marker")
for name, expected in maps["figure_sha256"].items():
    path = Path("paper/images") / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"RMSE-map hash mismatch: {path}")
for path_string, expected in maps.get("source_sha256", {}).items():
    path = Path(path_string)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"RMSE-map source hash mismatch: {path}")
expected_map_sources = {
    "scripts/run_weatherbridge_rmse_maps_cloudru.sh",
    "scripts/make_fig_rmse_maps_5tau_per_field.py",
    "trainer_weather_hermite.py",
    "legacy/scripts/trainer_weather_hermite.py",
    "examples/_bare_loader.py",
    "tools/eval/batch_eval_12h_memmap.py",
    "tools/eval/capmatched_loader.py",
    "tools/train/train_capacity_matched_6h.py",
    "weather_time_interp/model/weatherbridge_flow_model.py",
    "weather_time_interp/model/dcae_adaln_model.py",
    "weather_time_interp/model/dcae.py",
}
if relative_source_set(maps.get("source_sha256", {})) != expected_map_sources:
    raise SystemExit("RMSE-map marker has an unexpected source-file set")
if len(maps.get("artifact_sha256", {})) != 5:
    raise SystemExit("RMSE-map marker must bind five model artifacts")
for path_string, expected in maps["artifact_sha256"].items():
    path = Path(path_string)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"RMSE-map artifact hash mismatch: {path}")

cases = load("paper/images/.weatherbridge_case_figures_complete")
if (
    cases.get("schema_version") != 3
    or cases.get("status") != "complete"
    or cases.get("central_model") != "WeatherBridge"
    or cases.get("internal_arch") != "flow_pp3"
    or len(cases.get("figure_sha256", {})) != 2
    or len(cases.get("case_sha256", {})) != 6
    or len(cases.get("summary_sha256", {})) != 2
):
    raise SystemExit("invalid case-figure marker")
for name, expected in cases["figure_sha256"].items():
    path = Path("paper/images") / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"case-figure hash mismatch: {path}")
for path_string, expected in cases.get("case_sha256", {}).items():
    path = Path(path_string)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"case artifact hash mismatch: {path}")
for path_string, expected in cases.get("summary_sha256", {}).items():
    path = Path(path_string)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"case summary hash mismatch: {path}")
case_metrics = Path("paper/case_metrics_v2.tex")
if hashlib.sha256(case_metrics.read_bytes()).hexdigest() != cases.get(
    "metrics_tex_sha256"
):
    raise SystemExit("case metric macro hash mismatch")
case_sources = cases.get("source_sha256", {})
expected_case_sources = {
    "scripts/run_weatherbridge_case_figures_cloudru.sh",
    "tools/eval/export_weatherbridge_cases.py",
    "tools/eval/export_case_metrics_tex.py",
    "scripts/make_fig_case_ida_wind.py",
    "scripts/make_fig5_haishen_local.py",
    "scripts/paper_plot_style.py",
    "tools/eval/capmatched_loader.py",
    "tools/train/train_capacity_matched_6h.py",
    "weather_time_interp/model/weatherbridge_flow_model.py",
}
if relative_source_set(case_sources) != expected_case_sources:
    raise SystemExit("case marker has an unexpected source-file set")
for path_string, expected in case_sources.items():
    path = Path(path_string)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"case source hash mismatch: {path}")
case_checkpoint = Path(cases.get("checkpoint_path", ""))
if not case_checkpoint.is_file() or hashlib.sha256(
    case_checkpoint.read_bytes()
).hexdigest() != cases.get("checkpoint_sha256"):
    raise SystemExit("case checkpoint hash mismatch")
hres = load("paper/images/.hres_field_figures_complete")
if (
    hres.get("schema_version") != 1
    or hres.get("status") != "complete"
    or hres.get("central_model") != "WeatherBridge"
    or hres.get("internal_arch") != "flow_pp3"
    or hres.get("color") != "#D62728"
    or hres.get("horizons") != [6, 12]
    or hres.get("forecast_init_count") != 16
    or set(hres.get("figure_sha256", {}))
    != {"fig_hres_fields_6h.pdf", "fig_hres_fields_12h.pdf"}
    or len(hres.get("manifest_sha256", {})) != 2
):
    raise SystemExit("invalid HRES field-figure marker")
expected_hres_sources = {
    "scripts/run_hres_field_figures_cloudru.sh",
    "scripts/make_hres_field_comparison.py",
    "scripts/paper_plot_style.py",
    "tools/eval/batch_eval_forecast_anchor.py",
    "weather_time_interp/grid.py",
    "weather_time_interp/normalization.py",
}
hres_sources = hres.get("source_files_sha256", {})
if relative_source_set(hres_sources) != expected_hres_sources:
    raise SystemExit("HRES figure marker has an unexpected source-file set")
for path_string, expected in hres_sources.items():
    path = Path(path_string)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"HRES figure source hash mismatch: {path}")
for name, expected in hres["figure_sha256"].items():
    path = Path("paper/images") / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"HRES figure hash mismatch: {path}")
for path_string, expected in hres["manifest_sha256"].items():
    path = Path(path_string)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"HRES figure manifest hash mismatch: {path}")
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "complete"
        or payload.get("forecast_init_count") != 16
        or payload.get("models")
        != [
            "Linear Interp.",
            "WeatherDCAE-14M",
            "WeatherBridge",
        ]
        or payload.get("horizon_hours") not in {6, 12}
        or not payload.get("window_index_sha256")
        or not payload.get("data_provenance_sha256")
    ):
        raise SystemExit(f"invalid HRES figure manifest: {path}")
    for record in payload.get("artifact_sha256", {}).values():
        artifact = Path(record["path"])
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != record["sha256"]:
            raise SystemExit(f"HRES input artifact hash mismatch: {artifact}")
    figure = Path(payload["figure"]["path"])
    if hashlib.sha256(figure.read_bytes()).hexdigest() != payload["figure"]["sha256"]:
        raise SystemExit(f"HRES manifest figure hash mismatch: {figure}")

spectral = load("metrics/journal_spectra_v6/state/.complete")
source_sha = Path("metrics/journal_spectra_v6/state/source.sha256").read_text().split()[0]
if (
    spectral.get("status") != "complete"
    or spectral.get("source_sha256") != source_sha
    or spectral.get("lmax") != 180
    or spectral.get("ell_min") != 80
    or spectral.get("run_2021") is not True
    or spectral.get("artifact_count") != 208
):
    raise SystemExit("invalid schema-v6 spectral marker")
spectral_sources = spectral.get("source_files_sha256", {})
spectral_snapshot = Path("metrics/journal_spectra_v6/state/source_snapshot")
expected_spectral_sources = {
    "scripts/run_journal_spectra_v6_queue_cloudru.sh",
    "dataset.py",
    "metrics/__init__.py",
    "metrics/weather.py",
    "legacy/scripts/trainer_weather_hermite.py",
    "weather_time_interp/grid.py",
    "weather_time_interp/memmap_dataset.py",
    "weather_time_interp/eval_datasets.py",
    "weather_time_interp/metrics/spherical_spectra.py",
    "weather_time_interp/model/fuxi_swinv2_model.py",
    "weather_time_interp/model/modafno_baseline_model.py",
    "weather_time_interp/model/sdyff_dyffusion_model.py",
    "tools/eval/sh_energy_spectra_12h.py",
    "tools/eval/batch_eval_memmap.py",
    "tools/eval/capmatched_loader.py",
    "tools/eval/spectral_block_bootstrap.py",
    "tools/eval/summarize_spectral_dominance.py",
    "tools/eval/build_sh_per_channel_table_12h.py",
    "scripts/export_weatherbridge_spectral_latex_data.py",
    "scripts/make_fig3_spectra.py",
    "paper/figs/fig_weatherbridge_spectra_tau23.tex",
    "legacy/scripts/train_atm_vfi_12h_oddskip.py",
}
if set(spectral_sources) != expected_spectral_sources:
    raise SystemExit("spectral marker has an unexpected source-file set")
for path_string, expected in spectral_sources.items():
    verify_historical_source(
        path_string,
        expected,
        snapshot_roots=(spectral_snapshot,),
        label="spectral source",
    )
spectral_figures = spectral.get("figure_sha256", {})
if set(spectral_figures) != {
    "fig_weatherbridge_spectra_tau23.pdf",
    "fig_spectra_ratio_hard.pdf",
}:
    raise SystemExit("spectral marker has an unexpected figure set")
for name, expected in spectral_figures.items():
    path = Path("paper/images") / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"spectral figure hash mismatch: {path}")
for table in (
    Path("paper/tab_sh_per_channel_6h_v6.tex"),
    Path("paper/tab_sh_per_channel_12h_v6.tex"),
):
    if not table.is_file() or table.stat().st_size == 0:
        raise SystemExit(f"missing corrected spectral table: {table}")

champion = load("metrics/journal_champion_v1/state/.complete")
if (
    champion.get("selector_status") not in {"confirmed", "reference_retained"}
    or not champion.get("winner")
):
    raise SystemExit("no finalized journal architecture")
champion_result = Path("metrics/journal_champion_v1/final.json")
if hashlib.sha256(champion_result.read_bytes()).hexdigest() != champion.get(
    "result_sha256"
):
    raise SystemExit("champion result hash mismatch")
champion_payload = json.loads(champion_result.read_text())
if (
    champion_payload.get("winner") != champion["winner"]
    or champion_payload.get("status") != champion["selector_status"]
):
    raise SystemExit("champion result and marker disagree")
champion_input_hashes = set(champion_payload.get("input_sha256", {}).values())
full_input_hashes = set(full["input_sha256"].values())
core_metric_names = {
    "weatherbridge_pp3_14m_6yr_ep8.json",
    "weatherdcae_14m_6yr_ep8_matched.json",
    "weatherbridge_14m_3yr_ep10.json",
    "weatherdcae_14m_3yr_ep10_matched.json",
}
full_inputs_by_name = {
    Path(path_string).name: (Path(path_string), expected)
    for path_string, expected in full["input_sha256"].items()
    if Path(path_string).name in core_metric_names
    and any(part.endswith("_2020") for part in Path(path_string).parts)
}
if set(full_inputs_by_name) != core_metric_names:
    raise SystemExit("headline metrics are missing a core model artifact")
alignment_tool_sha = next(
    expected
    for path_string, expected in full["source_files_sha256"].items()
    if Path(path_string).name == "align_window_artifact.py"
)
for name, (path, expected) in full_inputs_by_name.items():
    if expected in champion_input_hashes:
        continue
    payload = json.loads(path.read_text())
    alignment = payload.get("publication_window_alignment", {})
    if not (
        alignment.get("method") == "exact_key_reorder_v1"
        and alignment.get("key_columns") == ["year", "t0", "tau"]
        and alignment.get("tool_sha256") == alignment_tool_sha
        and alignment.get("source_json_sha256") in champion_input_hashes
        and alignment.get("source_window_sha256") in champion_input_hashes
        and alignment.get("reference_json_sha256") in full_input_hashes
        and isinstance(alignment.get("source_index_sha256"), str)
        and isinstance(alignment.get("reference_index_sha256"), str)
        and alignment.get("window_count", 0) > 0
    ):
        raise SystemExit(f"headline/champion metric lineage mismatch: {name}")
champion_source = Path(
    "metrics/journal_champion_v1/state/source.sha256"
).read_text().split()[0]
if champion.get("source_sha256") != champion_source:
    raise SystemExit("champion source hash mismatch")
champion_sources = champion.get("source_files_sha256", {})
expected_champion_sources = {
    "scripts/run_journal_champion_queue_cloudru.sh",
    "tools/eval/select_journal_champion.py",
    "tools/eval/benchmark_capmatched_inference.py",
    "tools/repro/materialize_completion_source_snapshot.py",
    "tools/train/run_npj_seed_replicates_cloudru.sh",
    "tools/train/train_capacity_matched_6h.py",
    "weather_time_interp/model/weatherbridge_flow_model.py",
    "weather_time_interp/model/dcae_adaln_model.py",
    "scripts/run_npj_seed_geometry_recheck_cloudru.sh",
}
if relative_source_set(champion_sources) != expected_champion_sources:
    raise SystemExit("champion marker has an unexpected source-file set")
for path_string, expected in champion_sources.items():
    path = Path(path_string)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"champion source-file hash mismatch: {path}")
champion_marker_path = Path("metrics/journal_champion_v1/state/.complete")
for label, artifact in (("case", cases), ("RMSE-map", maps)):
    if artifact.get("champion_sha256") != hashlib.sha256(
        champion_result.read_bytes()
    ).hexdigest():
        raise SystemExit(f"{label} champion hash mismatch")
    if artifact.get("champion_marker_sha256") != hashlib.sha256(
        champion_marker_path.read_bytes()
    ).hexdigest():
        raise SystemExit(f"{label} champion marker hash mismatch")
    if artifact.get("selected_model") != champion["winner"]:
        raise SystemExit(f"{label} selected-model mismatch")
    if artifact.get("selector_status") != champion["selector_status"]:
        raise SystemExit(f"{label} selector-status mismatch")

postselection_marker_path = (
    metrics_runtime / "metrics/postselection_2022_v1/state/.complete"
)
postselection_result_path = (
    metrics_runtime / "metrics/postselection_2022_v1/final.json"
)
weatherbridge_assessment_path = (
    metrics_runtime / "metrics/postselection_2022_v1/weatherbridge_assessment.json"
)
postselection = json.loads(postselection_marker_path.read_text())
if (
    postselection.get("status") != "complete"
    or postselection.get("frozen_winner") != champion["winner"]
    or postselection.get("assessment_status")
    not in {
        "confirmed_aggregate",
        "reference_retained",
        "reference_retained_holdout_unopened",
        "postselection_failed",
    }
):
    raise SystemExit("invalid frozen 2022 holdout marker")
if hashlib.sha256(postselection_result_path.read_bytes()).hexdigest() != (
    postselection.get("result_sha256")
):
    raise SystemExit("post-selection result hash mismatch")
postselection_sources = postselection.get("source_files_sha256", {})
expected_postselection_sources = {
    "scripts/run_postselection_2022_queue_cloudru.sh",
    "tools/data/build_postselection_2022_memmap.py",
    "tools/data/verify_postselection_2022_memmap.py",
    "tools/eval/assess_postselection_holdout.py",
    "tools/eval/export_postselection_tex.py",
    "tools/eval/batch_eval_12h_memmap.py",
    "tools/eval/capmatched_loader.py",
    "tools/eval/eval_artifact_status.py",
    "tools/eval/paired_block_bootstrap.py",
    "tools/eval/paired_aux_block_bootstrap.py",
    "tools/eval/hard_window_block_bootstrap.py",
    "tools/eval/sh_energy_spectra_12h.py",
    "tools/eval/spectral_block_bootstrap.py",
    "tools/eval/summarize_spectral_dominance.py",
    "tools/train/train_capacity_matched_6h.py",
    "weather_time_interp/grid.py",
    "weather_time_interp/eval_runner.py",
    "weather_time_interp/normalization.py",
    "weather_time_interp/metrics/spherical_spectra.py",
    "weather_time_interp/metrics/physical_consistency.py",
    "weather_time_interp/model/weatherbridge_flow_model.py",
    "weather_time_interp/model/dcae_adaln_model.py",
    "repro/postselection_holdout_2022.json",
}
if relative_source_set(postselection_sources) != expected_postselection_sources:
    raise SystemExit("post-selection marker has an unexpected source-file set")
if hashlib.sha256(champion_result.read_bytes()).hexdigest() != (
    postselection.get("champion_sha256")
):
    raise SystemExit("post-selection champion hash mismatch")
if hashlib.sha256(champion_marker_path.read_bytes()).hexdigest() != (
    postselection.get("champion_marker_sha256")
):
    raise SystemExit("post-selection champion marker hash mismatch")

weatherbridge_assessment = json.loads(weatherbridge_assessment_path.read_text())
if (
    weatherbridge_assessment.get("status") != "confirmed_aggregate"
    or weatherbridge_assessment.get("evaluated_candidate") != "flow_spectral"
    or weatherbridge_assessment.get("reference") != "weatherdcae_14m"
    or weatherbridge_assessment.get("frozen_winner") != champion["winner"]
    or not weatherbridge_assessment.get("aggregate_confirmation_pass")
    or weatherbridge_assessment.get("universal_dominance_claim_allowed") is not False
):
    raise SystemExit("invalid WeatherBridge 2022 assessment")
if weatherbridge_assessment.get("champion_sha256") != hashlib.sha256(
    champion_result.read_bytes()
).hexdigest():
    raise SystemExit("WeatherBridge assessment champion hash mismatch")
if weatherbridge_assessment.get("manifest_sha256") != hashlib.sha256(
    Path("repro/postselection_holdout_2022.json").read_bytes()
).hexdigest():
    raise SystemExit("WeatherBridge assessment manifest hash mismatch")
if weatherbridge_assessment.get("verification_sha256") != hashlib.sha256(
    post_verification_path.read_bytes()
).hexdigest():
    raise SystemExit("WeatherBridge assessment verification hash mismatch")
if Path("paper/postselection_2022.tex").read_text() != render_postselection_tex(
    weatherbridge_assessment
):
    raise SystemExit("post-selection TeX hash mismatch")
PY

"$PY" tools/eval/export_journal_statistics.py \
  --champion-json metrics/journal_champion_v1/final.json \
  --out-csv paper/supplementary_data_1_statistics.csv \
  --out-manifest paper/supplementary_data_1_statistics.manifest.json \
  --source-dir paper/supplementary_data_1_sources
"$PY" tools/repro/export_postselection_supplementary_data.py \
  --assessment "$METRICS_RUNTIME/metrics/postselection_2022_v1/weatherbridge_assessment.json" \
  --verification "$POST_VERIFICATION" \
  --champion metrics/journal_champion_v1/final.json \
  --holdout-manifest repro/postselection_holdout_2022.json \
  --out-json paper/supplementary_data_2_postselection_2022.json \
  --out-manifest paper/supplementary_data_2_postselection_2022.manifest.json

FINAL_PUBLICATION=1 bash paper/build_npj_submission.sh
verify_source_snapshot

figures=(
  fig1_rmse_per_tau.pdf
  fig2_acc_per_tau.pdf
  fig_channels_body_6h_phys.pdf
  fig_weatherbridge_spectra_tau23.pdf
  fig_spectra_ratio_hard.pdf
  fig_channels_body_12h_phys.pdf
  fig_haishen_mslp.pdf
  fig_channels_body_ood2021.pdf
  fig_channels_app_ood2021.pdf
  fig_case_ida_wind.pdf
  fig_channels_app_12h_phys.pdf
  fig_channels_app_6h_phys.pdf
  fig_rmse_maps_t2m_5tau.pdf
  fig_rmse_maps_u10_5tau.pdf
  fig_rmse_maps_mslp_5tau.pdf
  fig_rmse_maps_Q1000_5tau.pdf
  fig_hres_fields_6h.pdf
  fig_hres_fields_12h.pdf
)
for figure in "${figures[@]}"; do
  test -s "paper/images/$figure"
  figure_text=$(pdftotext "paper/images/$figure" -)
  grep -q 'WeatherBridge' <<<"$figure_text"
  if grep -Eq 'WeatherBridge-(Detail|Flow)|Flow-Spectral' <<<"$figure_text"; then
    echo "legacy model name remains in $figure" >&2
    exit 1
  fi
done

"$PY" - "$MARKER" "$SOURCE_SHA" "${SOURCE_FILES[@]}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
source_sha = sys.argv[2]
source_files = [Path(value) for value in sys.argv[3:]]
paper = Path("paper/manuscript_npj.pdf")
statistics_csv = Path("paper/supplementary_data_1_statistics.csv")
statistics_manifest = Path("paper/supplementary_data_1_statistics.manifest.json")
statistics_exporter = Path("tools/eval/export_journal_statistics.py")
postselection_data = Path("paper/supplementary_data_2_postselection_2022.json")
postselection_manifest = Path(
    "paper/supplementary_data_2_postselection_2022.manifest.json"
)
postselection_exporter = Path(
    "tools/repro/export_postselection_supplementary_data.py"
)
physics_table = Path("paper/tab_downstream_physics_v2.tex")
diurnal_table = Path("paper/tab_downstream_diurnal_v2.tex")
champion = Path("metrics/journal_champion_v1/final.json")
champion_marker = Path("metrics/journal_champion_v1/state/.complete")
selection = json.loads(champion.read_text())
payload = {
    "schema_version": 1,
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "paper_sha256": hashlib.sha256(paper.read_bytes()).hexdigest(),
    "central_model": "WeatherBridge",
    "canonical_arch": "weatherbridge",
    "selected_model": selection["winner"],
    "selector_status": selection["status"],
    "spectral_schema": 6,
    "color": "#D62728",
    "source_sha256": source_sha,
    "source_files_sha256": {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in source_files
    },
    "statistics_csv_sha256": hashlib.sha256(
        statistics_csv.read_bytes()
    ).hexdigest(),
    "statistics_manifest_sha256": hashlib.sha256(
        statistics_manifest.read_bytes()
    ).hexdigest(),
    "statistics_exporter_sha256": hashlib.sha256(
        statistics_exporter.read_bytes()
    ).hexdigest(),
    "postselection_data_sha256": hashlib.sha256(
        postselection_data.read_bytes()
    ).hexdigest(),
    "postselection_manifest_sha256": hashlib.sha256(
        postselection_manifest.read_bytes()
    ).hexdigest(),
    "postselection_exporter_sha256": hashlib.sha256(
        postselection_exporter.read_bytes()
    ).hexdigest(),
    "physics_table_sha256": hashlib.sha256(
        physics_table.read_bytes()
    ).hexdigest(),
    "diurnal_table_sha256": hashlib.sha256(
        diurnal_table.read_bytes()
    ).hexdigest(),
    "champion_sha256": hashlib.sha256(champion.read_bytes()).hexdigest(),
    "champion_marker_sha256": hashlib.sha256(
        champion_marker.read_bytes()
    ).hexdigest(),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
"$PY" tools/repro/package_npj_submission.py \
  --paper-dir paper --out paper/weatherbridge_npj_submission.zip
"$PY" - "$MARKER" "paper/.npj_submission_package_complete" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

final_marker = Path(sys.argv[1])
package_marker = Path(sys.argv[2])
archive = Path("paper/weatherbridge_npj_submission.zip")
checksum = Path("paper/weatherbridge_npj_submission.zip.sha256")
payload = {
    "schema_version": 1,
    "status": "complete",
    "archive_path": str(archive),
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "archive_size_bytes": archive.stat().st_size,
    "checksum_sha256": hashlib.sha256(checksum.read_bytes()).hexdigest(),
    "final_marker_sha256": hashlib.sha256(
        final_marker.read_bytes()
    ).hexdigest(),
}
temporary = package_marker.with_suffix(package_marker.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, package_marker)
PY
echo "[$(date -Is)] paper figures verified and manuscript rebuilt"
