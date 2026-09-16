from pathlib import Path


def test_numerical_baseline_cli_supports_full_year() -> None:
    source = Path("tools/eval/numerical_baseline_eval.py").read_text()
    assert '"--full-year"' in source
    assert "None if args.full_year else args.eval_days_per_month" in source
    assert "normalize_longitude=not args.legacy_summed_longitude_rmse" in source


def test_numerical_baseline_queue_is_full_year_and_gpu_serialized() -> None:
    source = Path("scripts/run_full_year_numerical_baselines_cloudru.sh").read_text()
    assert "weatherbridge_transport_off.json" in source
    assert ".upr_lite_gpu${GPU}.lock" in source
    assert "--full-year" in source
    assert "--eval-hours 1,2,3,4,5" in source
    assert "1463 * 5" in source
    assert 'longitude_reduction") != "mean"' in source
    assert 'latitude_grid") != "wb2_0p25_2x2_block_average_v1"' in source
    assert "numerical_baselines_6h_2020_v3" in source
