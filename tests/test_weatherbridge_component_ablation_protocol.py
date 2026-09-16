from pathlib import Path


def test_component_ablation_queue_is_paired_and_full_year() -> None:
    source = Path(
        "scripts/run_weatherbridge_component_ablations_cloudru.sh"
    ).read_text()

    assert "--full-year" in source
    assert "--eval-hours 1,2,3,4,5" in source
    assert "--save-window-metrics" in source
    assert "--save-physical-metrics" in source
    assert "--save-temporal-metrics" in source
    assert "weatherbridge_acceleration_off 1 0 0" in source
    assert "weatherbridge_transport_off 0 1 0" in source
    assert "weatherbridge_hydrostatic_off 0 0 1" in source
    assert "weatherbridge 0 0 0" in source
    assert "WEATHERBRIDGE_ABLATE_ACCELERATION" in source
    assert "WEATHERBRIDGE_ABLATE_TRANSPORT" in source
    assert "WEATHERBRIDGE_ABLATE_HYDROSTATIC" in source
