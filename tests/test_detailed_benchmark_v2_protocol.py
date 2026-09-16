from pathlib import Path


def _read(name: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "scripts" / name).read_text()


def test_detailed_benchmarks_wait_for_matched_inputs_before_gpu_lock() -> None:
    for name in (
        "run_detailed_6h_benchmark_queue_cloudru.sh",
        "run_detailed_12h_benchmark_queue_cloudru.sh",
    ):
        script = _read(name)
        assert "detailed_benchmark_v2" in script
        assert "refinev1_s202707_bs4/last.ckpt" in script
        assert script.index('for checkpoint in "$DETAIL" "$REFINE" "$FLOW" "$DCAE"') < script.index(
            'exec 7>"$LOG_ROOT/.upr_lite_gpu${GPU}.lock"'
        )
        assert 'refine:$REFINE' in script
        assert "--cellwise" in script
        assert "hard_window_block_bootstrap.py" in script
        assert "--quantile 0.95" in script
        assert "tools/eval/eval_artifact_status.py" in script
        assert "tools/eval/paired_block_bootstrap.py" in script
        assert "tools/eval/paired_aux_block_bootstrap.py" in script
        assert "tools/eval/physical_block_bootstrap.py" in script
        assert "--require-temporal-metrics" in script
        assert '"status": "complete"' in script
        assert "sh_energy_spectra_12h.py" not in script


def test_twelve_hour_benchmark_uses_flow_spectral_identity() -> None:
    script = _read("run_detailed_12h_benchmark_queue_cloudru.sh")
    assert "exp_flow_pp3_spectral_14m_12h_2017_19_refinev1" in script
    assert 'flow_spectral:$FLOW' in script
    assert 'flow:$FLOW' not in script
    assert 'EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-12}"' in script
    assert script.count('--batch-size "$EVAL_BATCH_SIZE"') == 2


def test_report_does_not_publish_legacy_sparse_spectra() -> None:
    script = _read("run_detailed_benchmark_report_queue_cloudru.sh")
    assert "detailed_benchmark_v2" in script
    assert "make_fig3_spectra.py" not in script
    assert "sh_spectra_12h_ep10_24ch" not in script
    assert "WeatherBridge" in script
    assert "figure_sha256" in script
    assert "export_main_metrics_tex.py" in script
    assert "fig_detail_field_hour.pdf" not in script
    assert "weatherbridge_refine_14m" not in script
    assert 'CANONICAL="$RUNTIME/metrics/journal_unified"' in script
    assert 'export WTI_JOURNAL_METRICS_ROOT="$CANONICAL"' in script
    assert '$PAPER_ROOT/main_metrics_v2.tex' in script
    assert "materialize_completion_source_snapshot.py" in script
    assert 'wait_for_source_snapshot "validated 6h detailed benchmark"' in script
    assert '"corrected baseline geometry"' in script
    assert "export_weatherbridge_spectral_latex_data.py" in script
    assert 'export PYTHONPATH="$PWD:${PYTHONPATH:-}"' in script
    assert "weather_time_interp/normalization.py" in script


def test_channel_figure_builder_fails_on_missing_model_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "make_fig_specific_channels_phys.py").read_text()
    assert "refusing an incomplete comparison" in script
    assert "except (KeyError, FileNotFoundError)" not in script
