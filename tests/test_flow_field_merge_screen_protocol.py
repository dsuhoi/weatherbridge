from __future__ import annotations

from pathlib import Path


SCRIPT = Path("tools/eval/run_flow_field_merge_screen_cloudru.sh")


def test_field_merge_screen_is_validation_only_and_fail_closed() -> None:
    source = SCRIPT.read_text()

    assert "[merge_a025]=0.25" in source
    assert "[merge_a050]=0.50" in source
    assert "[merge_a100]=1.00" in source
    assert "--starting-control \"$STARTING_EVAL\"" in source
    assert "--matched-control \"$MATCHED_EVAL\"" in source
    assert "select_field_merge_champion.py" in source
    assert 'touch "$STATE/strict.pass"' in source
    assert 'touch "$STATE/strict.failed"' in source


def test_field_merge_screen_verifies_immutable_shared_parameters() -> None:
    source = SCRIPT.read_text()

    assert 'assert torch.equal(value, reference[key]), key' in source
    assert 'metadata["control"]["sha256"]' in source
    assert 'metadata["distilled"]["sha256"]' in source
