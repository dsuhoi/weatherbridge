from __future__ import annotations

from tools.eval.select_specialist_distillation import assess_candidates


def _metrics(value: float, q_value: float | None = None):
    channels = [
        *[f"T{level}" for level in (1000, 925, 850, 700)],
        *[f"U{level}" for level in (1000, 925, 850, 700)],
        *[f"V{level}" for level in (1000, 925, 850, 700)],
        *[f"Q{level}" for level in (1000, 925, 850, 700)],
        *[f"Z{level}" for level in (1000, 925, 850, 700)],
        "t2m",
        "u10",
        "v10",
        "mslp",
    ]
    return {
        tau: {
            channel: q_value if channel.startswith("Q") and q_value else value
            for channel in channels
        }
        for tau in range(1, 6)
    }


def test_selector_prefers_larger_moisture_gain_under_guards() -> None:
    result = assess_candidates(
        _metrics(1.0),
        {
            "q_decay": _metrics(1.0, 0.99),
            "q_anchor": _metrics(1.0, 0.98),
        },
        aggregate_tolerance_pct=0.02,
        held_tolerance_pct=0.02,
    )

    assert result["status"] == "pass"
    assert result["selected"] == "q_anchor"


def test_selector_rejects_moisture_gain_with_aggregate_regression() -> None:
    result = assess_candidates(
        _metrics(1.0),
        {"regressed": _metrics(1.01, 0.99)},
        aggregate_tolerance_pct=0.02,
        held_tolerance_pct=0.02,
    )

    assert result["status"] == "fail"
    assert result["selected"] is None
    assert (
        result["candidates"]["regressed"]["checks"][
            "aggregate_noninferior"
        ]
        is False
    )
