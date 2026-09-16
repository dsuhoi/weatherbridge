from __future__ import annotations

import numpy as np

from tools.eval.eval_multiteacher_oracle import (
    _stratified_indices,
    assess_oracle_metrics,
)


def test_oracle_gate_requires_advected_improvement_and_held_stability() -> None:
    tau_hours = [1, 2, 3, 4, 5]
    mask = np.ones(24, dtype=np.float64)
    mask[[16, 17, 18, 19, 20, 23]] = 0.0
    low = np.ones((5, 24), dtype=np.float64)
    composite = low.copy()
    composite[:, mask.astype(bool)] = 0.99

    result = assess_oracle_metrics(
        {
            "low_teacher": low,
            "high_teacher": np.full_like(low, 1.1),
            "composite": composite,
        },
        tau_hours,
        mask,
        aggregate_tolerance=0.001,
        held_tolerance=0.002,
    )

    assert result["pass"] is True
    assert all(result["checks"].values())


def test_oracle_gate_fails_on_held_out_regression() -> None:
    tau_hours = [1, 2, 3, 4, 5]
    mask = np.ones(24, dtype=np.float64)
    low = np.ones((5, 24), dtype=np.float64)
    composite = np.full_like(low, 0.99)
    composite[1] = 1.01

    result = assess_oracle_metrics(
        {
            "low_teacher": low,
            "high_teacher": low,
            "composite": composite,
        },
        tau_hours,
        mask,
        aggregate_tolerance=0.02,
        held_tolerance=0.002,
    )

    assert result["pass"] is False
    assert result["checks"]["held_tau_2_noninferior"] is False


def test_oracle_subset_is_balanced_across_tau() -> None:
    index = [
        (2020, day * 24, hour, hour / 6.0)
        for day in range(20)
        for hour in (1, 2, 3, 4, 5)
    ]

    selected = _stratified_indices(index, [1, 2, 3, 4, 5], 17)
    selected_hours = [index[item][2] for item in selected]

    counts = {hour: selected_hours.count(hour) for hour in range(1, 6)}
    assert sorted(counts.values()) == [3, 3, 3, 4, 4]
