from pathlib import Path


def _source() -> str:
    return (
        Path(__file__).parents[1]
        / "tools"
        / "eval"
        / "run_weatherdcae_sparse_student_postrun_cloudru.sh"
    ).read_text()


def test_weatherdcae_sparse_postrun_is_frozen_and_fail_closed() -> None:
    source = _source()

    assert 'while [[ ! -e "$SCREEN_ROOT/state/strict.pass" ]]' in source
    assert 'if [[ -e "$SCREEN_ROOT/state/strict.failed" ]]' in source
    assert 'assert sys.argv[6] == "weatherdcae_flow_student"' in source
    assert 'assert direct["blend"] == 0.02' in source
    assert 'assert distill["every_n_steps"] == 4' in source
    assert 'assert distill["interval_corrected"] is True' in source
    assert 'assert control_distill.get("enabled") is False' in source
    assert "freeze_distilled_student_selection.py" in source
    assert 'touch "$STATE/precheck.pass"' in source
    assert source.index("freeze_distilled_student_selection.py") < source.index(
        "run_year 1 2021"
    )


def test_weatherdcae_sparse_postrun_covers_full_pipeline_on_two_gpus() -> None:
    source = _source()

    assert "run_year 0 2020 &" in source
    assert "run_year 1 2021 &" in source
    assert '"distilled_student:$STUDENT:dcae_14m"' in source
    assert "--arch dcae_14m" in source
    assert "--full-year" in source
    assert "--save-physical-metrics" in source
    assert "--save-temporal-metrics" in source
    assert "paired_aux_block_bootstrap.py" in source
    assert "eval_anchor_exchange_consistency.py" in source
    assert "region_season_12h_eval.py" in source
    assert "sh_energy_spectra_12h.py" in source
    assert "--lmax 359" in source
    assert "batch_eval_forecast_anchor.py" in source
    assert "eval_physics_consistency.py" in source
    assert "eval_diurnal_amplitude.py" in source
    assert "benchmark_capmatched_inference.py" in source
    assert 'touch "$STATE/all.complete"' in source


def test_weatherdcae_sparse_postrun_reuses_climatology_cache() -> None:
    source = _source()

    assert "CLIM_CACHE=${CLIM_CACHE:-/tmp/wti_climatology_cache_detailed_6h}" in source
    assert 'ln -s "$CLIM_CACHE" "$OUT_ROOT/climatology_cache"' in source
    assert 'test "$(readlink -f "$OUT_ROOT/climatology_cache")"' in source
