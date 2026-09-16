from __future__ import annotations

from tools.eval.select_distilled_champion import select_strict_champion


def _candidate(aggregate: float, moisture: float, non_q: float, held: float):
    return {
        "pass": True,
        "deltas_vs_control_pct": {
            "aggregate_delta_pct": aggregate,
            "moisture_delta_pct": moisture,
            "non_moisture_delta_pct": non_q,
            "held_delta_pct": held,
        },
    }


def test_selects_broadly_better_candidate_by_aggregate() -> None:
    result = select_strict_champion({
        "selection_role": "era5_2020_validation_only",
        "control": {},
        "candidates": {
            "q_only": _candidate(-0.01, -0.20, 0.01, -0.01),
            "broad": _candidate(-0.03, -0.10, -0.01, -0.02),
            "broad_small": _candidate(-0.01, -0.12, -0.01, -0.01),
        },
    })

    assert result["status"] == "pass"
    assert result["selected"] == "broad"
    assert result["candidates"]["q_only"]["pass"] is False


def test_rejects_specialist_that_is_not_broadly_better() -> None:
    result = select_strict_champion({
        "selection_role": "era5_2020_validation_only",
        "control": {},
        "candidates": {
            "aggregate_regression": _candidate(0.001, -0.2, 0.0, 0.0),
            "held_regression": _candidate(-0.01, -0.2, -0.01, 0.001),
        },
    })

    assert result["status"] == "fail"
    assert result["selected"] is None


def test_matched_control_gate_promotes_only_distilled_arm() -> None:
    original = {
        "selection_role": "era5_2020_validation_only",
        "control": {"name": "original"},
        "candidates": {
            "qhead_gt": _candidate(-0.02, -0.10, 0.0, -0.01),
            "qhead_flow": _candidate(-0.03, -0.12, 0.0, -0.02),
        },
    }
    matched = {
        "selection_role": "era5_2020_validation_only",
        "control": {"name": "qhead_gt"},
        "candidates": {
            "qhead_flow": _candidate(-0.01, -0.02, 0.0, -0.01),
        },
    }

    result = select_strict_champion(original, matched)

    assert result["status"] == "pass"
    assert result["selected"] == "qhead_flow"
    assert result["candidates"]["qhead_gt"]["pass"] is False
    assert result["candidates"]["qhead_flow"][
        "matched_control_assessment"
    ]["pass"] is True


def test_matched_control_gate_rejects_teacher_without_added_value() -> None:
    original = {
        "selection_role": "era5_2020_validation_only",
        "control": {},
        "candidates": {
            "qhead_gt": _candidate(-0.03, -0.12, 0.0, -0.02),
            "qhead_flow": _candidate(-0.02, -0.10, 0.0, -0.01),
        },
    }
    matched = {
        "selection_role": "era5_2020_validation_only",
        "control": {},
        "candidates": {
            "qhead_flow": _candidate(0.01, 0.02, 0.0, 0.01),
        },
    }

    result = select_strict_champion(original, matched)

    assert result["status"] == "fail"
    assert result["selected"] is None
