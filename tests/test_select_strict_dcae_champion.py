import json

from tools.eval.select_strict_dcae_champion import select_champion


CHANNELS = [f"field_{index}" for index in range(24)]


def _write_metrics(path, *, rmse=1.0, acc=0.9, overrides=None):
    overrides = overrides or {}
    per_tau = {}
    for hour in range(1, 6):
        model = {}
        for channel in CHANNELS:
            cell = f"h{hour}/{channel}"
            cell_rmse, cell_acc = overrides.get(cell, (rmse, acc))
            model[f"rmse_norm_{channel}"] = cell_rmse
            model[f"acc_{channel}"] = cell_acc
        per_tau[str(hour)] = {"model": model}
    path.write_text(
        json.dumps(
            {
                "years": [2020],
                "channel_names": CHANNELS,
                "checkpoint": str(path.with_suffix(".ckpt")),
                "evaluation_protocol": {"index_sha256": "same-index"},
                "per_tau": per_tau,
            }
        )
    )
    return path


def test_strict_candidate_beats_lower_mean_candidate_with_one_regression(tmp_path):
    reference = _write_metrics(tmp_path / "reference.json")
    strict = _write_metrics(tmp_path / "strict.json", rmse=0.95, acc=0.91)
    lower_mean = _write_metrics(
        tmp_path / "lower_mean.json",
        rmse=0.90,
        acc=0.92,
        overrides={"h2/field_3": (1.001, 0.899)},
    )

    result = select_champion(
        [("strict", strict), ("lower_mean", lower_mean)],
        ("weatherdcae_14m_6yr", reference),
    )

    assert result["strict_champion"] == "strict"
    assert result["winner"] == "strict"
    assert result["eligible_for_ood_confirmation"]
    assert result["candidates"]["strict"]["rmse_wins"] == 120
    assert result["candidates"]["strict"]["acc_wins"] == 120
    assert not result["candidates"]["lower_mean"]["strict_pointwise_dominance"]


def test_diagnostic_winner_is_not_promoted_when_no_candidate_dominates(tmp_path):
    reference = _write_metrics(tmp_path / "reference.json")
    one_failure = _write_metrics(
        tmp_path / "one_failure.json",
        rmse=0.95,
        acc=0.91,
        overrides={"h4/field_7": (1.002, 0.91)},
    )
    two_failures = _write_metrics(
        tmp_path / "two_failures.json",
        rmse=0.90,
        acc=0.92,
        overrides={
            "h2/field_1": (1.001, 0.92),
            "h4/field_1": (1.001, 0.92),
        },
    )

    result = select_champion(
        [("one_failure", one_failure), ("two_failures", two_failures)],
        ("weatherdcae_14m_6yr", reference),
    )

    assert result["strict_champion"] is None
    assert result["diagnostic_winner"] == "one_failure"
    assert result["winner"] == "one_failure"
    assert not result["eligible_for_ood_confirmation"]
