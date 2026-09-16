import json

import pytest

from tools.eval.select_universal_pyramid import select


def _write_metrics(path, value, *, year=2020, one_cell=None):
    channels = [f"c{index}" for index in range(24)]
    per_tau = {}
    for hour in range(1, 6):
        model = {
            f"rmse_norm_{channel}": float(value)
            for channel in channels
        }
        if one_cell == hour:
            model["rmse_norm_c0"] = float(value) * 1.2
        model["acc_mean"] = 0.99
        per_tau[str(hour)] = {"model": model}
    path.write_text(
        json.dumps(
            {
                "years": [year],
                "channel_names": channels,
                "checkpoint": f"{path.stem}.ckpt",
                "evaluation_protocol": {"index_sha256": "fixed-index"},
                "per_tau": per_tau,
            }
        )
    )


def _write_inference(path):
    path.write_text(
        json.dumps(
            {
                "models": {
                    "universal_pyramid": {
                        "params_m": 9.50046,
                        "latency_ms": 40.0,
                    },
                    "flow_spectral": {
                        "params_m": 14.260565,
                        "latency_ms": 100.0,
                    },
                }
            }
        )
    )


def test_selects_quality_preserving_fast_candidate(tmp_path):
    candidate = tmp_path / "candidate.json"
    flow = tmp_path / "flow.json"
    quality = tmp_path / "quality.json"
    inference = tmp_path / "inference.json"
    _write_metrics(candidate, 0.09)
    _write_metrics(flow, 0.10)
    _write_metrics(quality, 0.088)
    _write_inference(inference)

    result = select(
        candidate,
        flow,
        quality,
        inference,
        quality_gap_limit=0.05,
        latency_ratio_limit=0.8,
        cell_regression_limit=0.0,
    )

    assert result["pareto_pass"]
    assert result["strict_uniform_pass"]
    assert result["eligible_for_ood_evaluation"]
    assert result["comparison"]["cell_wins_vs_flow"] == 120


def test_separates_aggregate_and_uniform_gates(tmp_path):
    candidate = tmp_path / "candidate.json"
    flow = tmp_path / "flow.json"
    quality = tmp_path / "quality.json"
    inference = tmp_path / "inference.json"
    _write_metrics(candidate, 0.09, one_cell=3)
    _write_metrics(flow, 0.10)
    _write_metrics(quality, 0.088)
    _write_inference(inference)

    result = select(
        candidate,
        flow,
        quality,
        inference,
        quality_gap_limit=0.05,
        latency_ratio_limit=0.8,
        cell_regression_limit=0.0,
    )

    assert result["pareto_pass"]
    assert not result["strict_uniform_pass"]
    assert result["comparison"]["worst_cell"] == "h3/c0"


def test_rejects_ood_year_at_selection(tmp_path):
    candidate = tmp_path / "candidate.json"
    flow = tmp_path / "flow.json"
    quality = tmp_path / "quality.json"
    inference = tmp_path / "inference.json"
    _write_metrics(candidate, 0.09, year=2021)
    _write_metrics(flow, 0.10)
    _write_metrics(quality, 0.088)
    _write_inference(inference)

    with pytest.raises(ValueError, match="only 2020"):
        select(
            candidate,
            flow,
            quality,
            inference,
            quality_gap_limit=0.05,
            latency_ratio_limit=0.8,
            cell_regression_limit=0.0,
        )
