from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_msf_training_preserves_matched_protocol_without_teacher() -> None:
    source = (
        REPO_ROOT / "tools/train/run_weatherbridge_msf_l_cloudru.sh"
    ).read_text()

    assert "--arch flow_msf_pareto_l" in source
    assert "--years 2014 2015 2016 2017 2018 2019" in source
    assert "--val_years 2020" in source
    assert "--train_tau_subset 1 3 5" in source
    assert "--eval_tau 1 2 3 4 5" in source
    assert "--max_epochs 4" in source
    assert "--lr 1e-4" in source
    assert "--samples_per_date_train 4" in source
    assert "--bs 8" in source
    assert "--accumulate 2" in source
    assert "--train_batches_per_epoch 3284" in source
    assert "--loss_profile pareto_minimax" in source
    assert "net_params == 8884205" in source
    assert "teacher_checkpoint" not in source
    assert "distill_weight" not in source


def test_msf_runner_pins_all_behavioral_sources() -> None:
    source = (
        REPO_ROOT / "tools/train/run_weatherbridge_msf_l_cloudru.sh"
    ).read_text()

    for path in (
        "legacy/training_snapshots/train_capacity_matched_6h_7b36d562.py",
        "weather_time_interp/model/weatherbridge_flow_model.py",
        "weather_time_interp/memmap_dataset.py",
        "tools/train/training_protocol.py",
        "weather_time_interp/normalization.py",
    ):
        assert path in source
    assert source.count("verify_source ") >= 5


def test_geo_msf_runner_is_native_and_from_scratch() -> None:
    source = (
        REPO_ROOT / "tools/train/run_weatherbridge_geo_msf_l_cloudru.sh"
    ).read_text()

    assert "--arch flow_geo_msf_l" in source
    assert "--years 2014 2015 2016 2017 2018 2019" in source
    assert "--train_tau_subset 1 3 5" in source
    assert "--eval_tau 1 2 3 4 5" in source
    assert "--bs 8" in source
    assert "--accumulate 2" in source
    assert "--train_batches_per_epoch 3284" in source
    assert "intrinsic_transport_normalization" in source
    assert "net.normalization_mean" in source
    assert "net.normalization_std" in source
    assert "teacher_checkpoint" not in source
    assert "distill_weight" not in source


def test_msf_eval_requires_all_fields_hours_before_full_year() -> None:
    source = (
        REPO_ROOT / "tools/eval/run_weatherbridge_msf_l_eval_cloudru.sh"
    ).read_text()

    assert "weatherbridge_msf_l_2ep" in source
    assert "weatherbridge_msf_l_4ep" in source
    assert "--eval-hours 1,2,3,4,5" in source
    assert "--seen-tau 1,3,5" in source
    assert "--unseen-tau 2,4" in source
    assert "--include-q" in source
    assert "--cell-limit 0" in source
    assert source.index('selected="$(selected_from "$ECONOMY_REPORT")"') < (
        source.index('"--full-year --save-window-metrics"')
    )
    assert "--lazy-climatology" in source
    assert "--full-year --save-window-metrics" in source
    for reference in (
        "WeatherBridge-PP3-8ep",
        "WeatherDCAE",
        "PixelAttn-VFI",
        "FuXi",
        "ModAFNO",
        "S-DYffusion",
    ):
        assert f'--reference "{reference}=' in source


def test_geo_msf_eval_uses_same_strict_envelope() -> None:
    source = (
        REPO_ROOT / "tools/eval/run_weatherbridge_geo_msf_l_eval_cloudru.sh"
    ).read_text()

    assert "weatherbridge_geo_msf_l_2ep" in source
    assert "weatherbridge_geo_msf_l_4ep" in source
    assert "--eval-hours 1,2,3,4,5" in source
    assert "--include-q" in source
    assert "--cell-limit 0" in source
    assert source.count("--reference ") >= 6


def test_distillation_runners_are_not_active_tools() -> None:
    assert not (
        REPO_ROOT / "tools/train/run_flow_pp3_compact_l_cloudru.sh"
    ).exists()
    assert not (
        REPO_ROOT / "tools/eval/run_flow_pp3_compact_l_eval_cloudru.sh"
    ).exists()
    assert (
        REPO_ROOT
        / "legacy/distillation_experiments/train"
        / "run_flow_pp3_compact_l_cloudru.sh"
    ).is_file()
