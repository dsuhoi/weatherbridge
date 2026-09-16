from pathlib import Path


def test_normalization_replay_waits_for_component_memory_release() -> None:
    source = Path("scripts/run_normalization_replay_cloudru.sh").read_text()
    assert "weatherbridge_component_ablation_v1/state/.complete" in source
    assert "generate_normalization_0p5.py" in source
    assert "verify_normalization_replay.py" in source
    assert "generated_stats.nc" in source
    assert "report.json" in source
