import ast
import re
from pathlib import Path


def test_journal_spectral_queue_uses_corrected_dense_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "scripts" / "run_journal_spectra_v6_queue_cloudru.sh"
    ).read_text()
    evaluator = (root / "tools" / "eval" / "sh_energy_spectra_12h.py").read_text()

    assert "--full-year" in script
    assert "--lmax \"$LMAX\" --hf-ell-min \"$ELL_MIN\"" in script
    assert "LMAX=180" in script
    assert "ELL_MIN=80" in script
    assert "--vector" in script
    assert "summarize_spectral_dominance.py" in script
    assert "journal_spectra_v6" in script
    assert 'export PYTHONPATH="$RUNTIME:$LEGACY:' in script
    assert (
        'metadata.get("sht_grid") '
        '!= "wb2_0p25_2x2_block_average_v1_latitude_strip_area_sht"'
        in script
    )
    assert "scripts/run_journal_spectra_v6_queue_cloudru.sh" in script
    assert "tools/eval/batch_eval_memmap.py" in script
    assert "legacy/scripts/train_atm_vfi_12h_oddskip.py" in script
    assert '"source_files_sha256"' in script
    assert "for left_name in weatherbridge refine flow_spectral" in script
    assert '"$STATE_DIR/.complete_6h"' in script
    assert "wait_for_specs \"${PREDETAIL_6[@]}\"" in script
    assert "wait_for_specs \"${CORE_6[0]}\"" in script
    assert script.index('wait_for_specs "${PREDETAIL_6[@]}"') < script.index(
        'wait_for_specs "${CORE_6[0]}"'
    )
    assert '"spectral_field_units": "physical_anomaly_units_via_channel_std"' in evaluator
    assert "_rescale_to_physical_anomalies" in evaluator
    assert "_load_legacy_trainer_class" in evaluator
    assert 'repo_root / "metrics" / "__init__.py"' in evaluator
    assert 'repo_root / "metrics" / "weather.py"' in evaluator
    assert "legacy/scripts/trainer_weather_hermite.py" in script
    assert "metrics/__init__.py" in script
    assert "metrics/weather.py" in script
    assert "tools/eval/capmatched_loader.py" in script
    assert (
        'modafno|$LEGACY/logs/exp_modafno_24ch_6yr/epoch=7-step=35032.ckpt|hermite|'
        'MODAFNO_INP_H=360 MODAFNO_INP_W=720 '
        'MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720'
    ) in script
    assert (
        'modafno|$LEGACY/logs/_12h_migrated/modafno_12h.ckpt|hermite|'
        'MODAFNO_INP_H=362 MODAFNO_INP_W=720 '
        'MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720'
    ) in script
    assert "source changed while spectral queue was active" in script
    assert script.count("verify_source_snapshot") >= 4
    assert "PREBUILT_DETAIL_SPECTRA_PDF" in script
    assert "command -v pdflatex" in script
    assert '"figure_sha256"' in script


def test_paper_finalizer_requires_complete_v6_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_paper_finalize_queue_cloudru.sh").read_text()

    assert 'spectral.get("artifact_count") != 208' in script
    assert 'spectral.get("run_2021") is not True' in script
    assert 'full.get("internal_arch") != "flow_pp3"' in script
    assert "figure hash mismatch" in script
    assert "journal_champion_v1/state/.complete" in script
    assert "no finalized journal architecture" in script
    assert "baseline_geometry_v1_complete" in script
    assert "postselection_2022_v1/state/.complete" in script
    assert ".weatherbridge_case_figures_complete" in script
    assert "post-selection TeX hash mismatch" in script
    assert "FINAL_PUBLICATION=1 bash paper/build_npj_submission.sh" in script
    assert 'full.get("schema_version") != 4' in script
    assert 'full.get("source_files_sha256", {})' in script
    assert 'full.get("input_sha256", {})' in script
    assert "full-year marker must bind nine figures" in script
    assert "verify_historical_source" in script
    assert "benchmark_snapshots" in script
    assert "spectral_snapshot" in script
    assert '"reference_retained"' in script
    assert "source changed while finalizer was active" in script
    assert script.count("verify_source_snapshot") >= 3
    assert '"source_files_sha256"' in script
    assert "package_npj_submission.py" in script
    assert ".npj_submission_package_complete" in script
    assert '"champion_sha256"' in script
    assert "tab_downstream_physics_v2.tex" in script
    assert "tab_downstream_diurnal_v2.tex" in script
    assert "physical-diagnostic table hash mismatch" in script
    assert 'alignment.get("method") == "exact_key_reorder_v1"' in script
    assert "headline/champion metric lineage mismatch" in script


def test_full_year_figure_queue_binds_sources_and_metric_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "scripts" / "run_detailed_benchmark_report_queue_cloudru.sh"
    ).read_text()

    assert '"schema_version": 4' in script
    assert '"source_files_sha256"' in script
    assert '"input_sha256"' in script
    assert "compute_source_sha" in script
    assert script.count("verify_source_snapshot") >= 3
    assert 'canonical / ".baseline_geometry_v1_complete"' in script
    assert '(canonical / directory).glob("*.json")' in script
    assert "export_downstream_metrics_tex.py" in script
    assert '"physics_table_sha256"' in script
    assert '"diurnal_table_sha256"' in script


def test_baseline_geometry_queue_exposes_legacy_model_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "scripts" / "run_journal_baseline_geometry_v1_cloudru.sh"
    ).read_text()

    assert 'export PYTHONPATH="$PWD:$LEGACY:' in script
    assert "--batch-size 8 --num-workers 0" in script
    assert "batch_eval_12h_memmap_fast.py" in script
    assert "--save-window-metrics" in script
    assert "--save-physical-metrics" not in script
    assert "--save-temporal-metrics" not in script
    assert 'HORIZONS="${HORIZONS:-6 12}"' in script
    assert 'WORKER_ID="${WORKER_ID:-primary}"' in script
    assert 'FINALIZE="${FINALIZE:-1}"' in script
    assert 'GPU_LOCK_ROOT="${GPU_LOCK_ROOT:-$LOG_ROOT}"' in script
    assert "waiting for the other baseline geometry worker" in script


def test_detailed_queues_freeze_sources_and_use_180_physics_windows() -> None:
    root = Path(__file__).resolve().parents[1]
    for horizon in (6, 12):
        script = (
            root / "scripts" / f"run_detailed_{horizon}h_benchmark_queue_cloudru.sh"
        ).read_text()
        assert "--max-pairs 180" in script
        assert "EXPECTED_SOURCE_SHA" in script
        assert "compute_source_sha" in script
        assert script.count("verify_source_snapshot") >= 4
        assert '"source_composite_sha256"' in script
        assert "tools/eval/capmatched_loader.py" in script


def test_geometry_and_postselection_queues_freeze_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    for filename in (
        "run_npj_seed_geometry_recheck_cloudru.sh",
        "run_postselection_2022_queue_cloudru.sh",
    ):
        script = (root / "scripts" / filename).read_text()
        assert "EXPECTED_SOURCE_SHA" in script
        assert "source changed while" in script
        assert script.count("verify_source_snapshot") >= 5


def test_champion_queue_is_fail_closed_and_conditionally_confirms_detail() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "scripts" / "run_journal_champion_queue_cloudru.sh"
    ).read_text()

    assert "select_journal_champion.py" in script
    assert "benchmark_capmatched_inference.py" in script
    assert "train_detail_confirmation" in script
    assert "JOBS_OVERRIDE=\"detail:6:202708" in script
    assert "EXPECTED_SOURCE_SHA" in script
    assert '"source_files_sha256"' in script
    assert "compute_source_sha" in script
    assert "verify_source_snapshot" in script
    assert script.count("verify_source_snapshot") >= 4
    assert "source changed while queue was active" in script


def test_spectral_figure_builders_require_actual_wb2_grid() -> None:
    root = Path(__file__).resolve().parents[1]
    figure = (root / "scripts" / "make_fig3_spectra.py").read_text()
    export = (root / "scripts" / "export_weatherbridge_spectral_latex_data.py").read_text()
    detail_figure = (
        root / "paper" / "figs" / "fig_weatherbridge_spectra_tau23.tex"
    ).read_text()

    assert "WB2_BLOCK_GRID_NAME" in figure
    assert "_latitude_strip_area_sht" in figure
    assert "_latitude_strip_area_sht" in export
    assert 'arrays["signed_cospectrum_l"][index]' in export
    assert 'handle.write("ell power_ratio signed_cospectrum\\n")' in export
    for artifact in (
        "linear_tau{tau}.npz",
        "fuxi_tau{tau}.npz",
        "modafno_tau{tau}.npz",
        "sdyff_tau{tau}.npz",
        "pixelattn_vfi_tau{tau}.npz",
        "weatherdcae_14m_tau{tau}.npz",
        "flow_spectral_tau{tau}.npz",
    ):
        assert artifact in export
    for legend_entry in (
        "Linear",
        "SwinV2",
        "ModAFNO",
        "S-DYff",
        "PixelAttn-VFI",
        "WeatherDCAE-14M",
        "WeatherBridge",
        "ERA5 target",
    ):
        assert f"\\addlegendentry{{{legend_entry}}}" in detail_figure
    assert "ylabel={Signed cospectrum}" in detail_figure
    assert "{signed_cospectrum}" in detail_figure
    assert "Spectral coherence" not in detail_figure
    assert "equiangular_cell_centered_fejer1" not in figure
    assert "equiangular_cell_centered_fejer1" not in export


def test_architecture_figure_matches_weatherbridge_path() -> None:
    root = Path(__file__).resolve().parents[1]
    figure = (
        root / "paper/figs/fig_weatherbridge_architecture.tex"
    ).read_text()
    manuscript = (root / "paper/main.tex").read_text()
    normalized_manuscript = " ".join(manuscript.split())

    for label in (
        "Three-scale lat--lon U-Net",
        "Encoder $E_1$",
        "Decoder $1/4$ res.",
        "Decoder $1/2$ res.",
        "Decoder full res.",
        "Conv block",
        "Skip connection",
        "Anchor transport",
        "Motion and blend heads",
        "$\\beta=\\sigma(b)$",
        "G_t^{\\rm SFNO}",
        "WeatherBridge output",
    ):
        assert label in figure
    assert "x_T-x_0" in manuscript
    assert "circular padding in both tensor dimensions" in normalized_manuscript
    assert "implementation artefact" in normalized_manuscript
    assert "longitude-periodic encoder--decoder" not in manuscript
    assert "Flow Matching" not in figure
    assert "detail bypass" not in figure.lower()
    assert "Training objective" not in figure


def test_case_queue_binds_frozen_weatherbridge_checkpoint_to_figures() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "scripts" / "run_weatherbridge_case_figures_cloudru.sh"
    ).read_text()

    assert "journal_champion_v1/final.json" in script
    assert 'payload.get("winner") != marker.get("winner")' in script
    assert '"selected_model": selection["winner"]' in script
    assert "--expected-arch flow_pp3" in script
    assert "checkpoint hash mismatch" in script
    assert 'provenance.get("schema_version") != 3' in script
    assert '"wb2_0p25_pair_average_cell_centres_v1"' in script
    assert "latitude hash mismatch" in script
    assert "wrong longitude centres" in script
    assert "figure_sha256" in script
    assert "export_case_metrics_tex.py" in script
    assert '"schema_version": 3' in script
    assert '"summary_sha256"' in script
    assert '"metrics_tex_sha256"' in script
    assert "compute_source_sha" in script
    assert script.count("verify_source_snapshot") >= 4


def test_paper_finalizer_embedded_python_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "scripts" / "run_paper_finalize_queue_cloudru.sh"
    ).read_text()

    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", script, flags=re.DOTALL)
    assert len(blocks) == 4
    for block in blocks:
        ast.parse(block)
    assert "paper/supplementary.tex" in script
    assert 'raise SystemExit(f"{label} champion hash mismatch")' in script
    assert 'raise SystemExit(f"{label} champion marker hash mismatch")' in script
    assert "expected_spectral_sources" in script
    assert "spectral marker has an unexpected source-file set" in script
    assert "spectral marker has an unexpected figure set" in script
    assert "spectral figure hash mismatch" in script
    assert "expected_full_sources" in script
    assert "expected_map_sources" in script
    assert "expected_case_sources" in script
    assert "expected_champion_sources" in script
    assert "expected_postselection_sources" in script
    assert "render_postselection_tex" in script
    assert "WeatherBridge assessment verification hash mismatch" in script
    assert "WeatherBridge-(Detail|Flow)|Flow-Spectral" in script
    assert "if pdftotext" not in script


def test_climatology_cache_temp_files_are_process_unique() -> None:
    root = Path(__file__).resolve().parents[1]
    evaluator = (root / "tools/eval/batch_eval_12h_memmap.py").read_text()

    assert 'f".{cache_path.stem}.{os.getpid()}.tmp.npy"' in evaluator


def test_atmvfi_loader_has_no_eager_hermite_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools" / "eval" / "batch_eval_memmap.py").read_text()
    prelude = source.split("def _load_atmvfi_model", maxsplit=1)[0]

    assert "from trainer_weather_hermite import" not in prelude
    assert "from tools.eval.compute_acc import" not in prelude
    assert "legacy" in source
    assert "train_atm_vfi_12h_oddskip.py" in source
    assert "load_capmatched_checkpoint" in source
    assert source.index('if hparams.get("arch")') < source.index(
        "# --- ATM-VFI detection + load"
    )
    assert "incompatible ATM-VFI state" in source


def test_spectral_atm_forward_preserves_capacity_matched_wrapper() -> None:
    root = Path(__file__).resolve().parents[1]
    helper = (root / "tools" / "eval" / "batch_eval_memmap.py").read_text()

    assert "class _ATMVFICapMatchedAdapter" in helper
    assert "class _ATMVFICapMatchedNetAdapter" in helper
    assert "self.net.wrapped(x0, xT, tau, cond, static=static)" in helper
    assert "_ATMVFICapMatchedAdapter(model).to(device).eval()" in helper
    assert 'if not getattr(model, "arch", None):' in helper


def test_public_weatherbridge_name_maps_to_flow_pp3_checkpoint() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = (root / "tools" / "eval" / "wti_model_registry_12h.yaml").read_text()
    weatherbridge_example = (
        root / "examples" / "run_weatherbridge_12h_bare.py"
    ).read_text()

    assert "paper_name: Detail-bypass ablation" not in registry
    assert "eval_name: weatherbridge_detail_14m_3yr_ep10" not in registry
    assert "paper_name: WeatherBridge" in registry
    assert "exp_flow_pp3_spectral_14m_12h" in registry
    assert "weatherbridge_14m_12h_bare.pt" in weatherbridge_example
