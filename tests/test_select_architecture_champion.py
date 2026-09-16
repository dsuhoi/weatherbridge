from __future__ import annotations

import json

from tools.eval.select_architecture_champion import select_champion


CHANNELS = [f"c{index}" for index in range(24)]


def _metrics(path, *, scale: float, index: str = "shared") -> None:
    per_tau = {}
    for hour in range(1, 6):
        model = {f"rmse_norm_{channel}": scale for channel in CHANNELS}
        model["acc_mean"] = 0.9
        per_tau[str(hour)] = {"model": model}
    path.write_text(
        json.dumps(
            {
                "years": [2020],
                "channel_names": CHANNELS,
                "per_tau": per_tau,
                "evaluation_protocol": {"index_sha256": index},
            }
        )
    )


def test_selector_separates_quality_and_efficiency_winners(tmp_path) -> None:
    flow = tmp_path / "flow.json"
    quality = tmp_path / "quality.json"
    compact = tmp_path / "compact.json"
    refined = tmp_path / "refined.json"
    _metrics(flow, scale=1.0)
    _metrics(quality, scale=0.90)
    _metrics(compact, scale=0.96)
    _metrics(refined, scale=0.92)
    inference = tmp_path / "inference.json"
    inference.write_text(
        json.dumps(
            {
                "models": {
                    "flow_spectral": {"latency_ms": 100.0, "params_m": 14.0},
                    "compact": {"latency_ms": 60.0, "params_m": 9.5},
                    "refined": {"latency_ms": 95.0, "params_m": 13.7},
                }
            }
        )
    )

    result = select_champion(
        [("compact", compact), ("refined", refined)],
        flow,
        quality,
        inference,
        quality_gap_limit=0.10,
        latency_ratio_limit=1.25,
        maximum_parameters_m=15.0,
        cell_regression_limit=0.0,
    )

    assert result["quality_winner"] == "refined"
    assert result["efficiency_winner"] == "compact"
    assert result["uniform_winner"] == "refined"
    assert result["primary"] == "refined"
    assert set(result["pareto_candidates"]) == {"compact", "refined"}


def test_selector_rejects_candidate_worse_than_flow(tmp_path) -> None:
    flow = tmp_path / "flow.json"
    quality = tmp_path / "quality.json"
    candidate = tmp_path / "candidate.json"
    _metrics(flow, scale=1.0)
    _metrics(quality, scale=0.90)
    _metrics(candidate, scale=1.01)
    inference = tmp_path / "inference.json"
    inference.write_text(
        json.dumps(
            {
                "models": {
                    "flow_spectral": {"latency_ms": 100.0, "params_m": 14.0},
                    "candidate": {"latency_ms": 50.0, "params_m": 9.0},
                }
            }
        )
    )

    result = select_champion(
        [("candidate", candidate)],
        flow,
        quality,
        inference,
        quality_gap_limit=0.20,
        latency_ratio_limit=1.25,
        maximum_parameters_m=15.0,
        cell_regression_limit=0.0,
    )

    assert result["primary"] is None
    assert not result["eligible_for_ood_confirmation"]
