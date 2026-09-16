from __future__ import annotations

from tools.eval.select_field_merge_champion import (
    TARGET_FIELDS,
    assess_field_merges,
)


def _metrics(target: float, excluded: float) -> dict[int, dict[str, float]]:
    channels = (*TARGET_FIELDS, "Q1000")
    return {
        tau: {
            channel: target if channel in TARGET_FIELDS else excluded
            for channel in channels
        }
        for tau in range(1, 6)
    }


def test_selector_requires_improvement_against_both_controls() -> None:
    result = assess_field_merges(
        _metrics(1.0, 2.0),
        _metrics(0.9, 2.0),
        {
            "good": _metrics(0.8, 2.0),
            "only_starting": _metrics(0.95, 2.0),
        },
    )

    assert result["status"] == "pass"
    assert result["selected"] == "good"
    assert result["candidates"]["only_starting"]["pass"] is False


def test_selector_rejects_excluded_regression() -> None:
    result = assess_field_merges(
        _metrics(1.0, 2.0),
        _metrics(1.0, 2.0),
        {"regresses": _metrics(0.8, 2.01)},
    )

    assert result["status"] == "fail"
    assert result["selected"] is None
