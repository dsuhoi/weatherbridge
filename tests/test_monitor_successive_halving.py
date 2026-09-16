from tools.train.monitor_successive_halving import pruning_decision


def _row(value: float) -> dict[str, float]:
    return {"mean": value, "held": value}


def test_requires_two_moderate_regressions() -> None:
    reference = {1: _row(1.0), 2: _row(1.0)}
    assert (
        pruning_decision(
            {1: _row(1.04)},
            reference,
            minimum_epoch=1,
            regression_threshold=0.03,
            hard_regression_threshold=0.08,
            patience=2,
        )
        is None
    )
    decision = pruning_decision(
        {1: _row(1.04), 2: _row(1.05)},
        reference,
        minimum_epoch=1,
        regression_threshold=0.03,
        hard_regression_threshold=0.08,
        patience=2,
    )
    assert decision is not None
    assert decision["decision_epoch"] == 2
    assert not decision["hard_failure"]


def test_hard_regression_prunes_immediately() -> None:
    decision = pruning_decision(
        {1: _row(1.09)},
        {1: _row(1.0)},
        minimum_epoch=1,
        regression_threshold=0.03,
        hard_regression_threshold=0.08,
        patience=2,
    )
    assert decision is not None
    assert decision["decision_epoch"] == 1
    assert decision["hard_failure"]
