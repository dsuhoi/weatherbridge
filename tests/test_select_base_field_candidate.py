from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.eval.select_base_field_candidate import build_selection


FIELDS = [
    "T1000",
    "T925",
    "T850",
    "T700",
    "U1000",
    "U925",
    "U850",
    "U700",
    "V1000",
    "V925",
    "V850",
    "V700",
    "Q1000",
    "Q925",
    "Q850",
    "Q700",
    "Z1000",
    "Z925",
    "Z850",
    "Z700",
    "t2m",
    "u10",
    "v10",
    "mslp",
]


def _artifact(scale: float) -> dict:
    return {
        "years": [2020],
        "n_per_tau": {str(tau): 16 for tau in range(1, 6)},
        "evaluation_protocol": {
            "samples_per_date": 4,
            "eval_days_per_month": 8,
        },
        "channel_names": FIELDS,
        "per_tau": {
            str(tau): {
                "model": {
                    f"rmse_norm_{field}": scale
                    for field in FIELDS
                }
            }
            for tau in range(1, 6)
        },
    }


def _write(path: Path, artifact: dict) -> Path:
    path.write_text(json.dumps(artifact))
    return path


def test_selects_candidate_that_improves_all_base_fields(tmp_path) -> None:
    reference = _write(tmp_path / "reference.json", _artifact(1.0))
    better = _write(tmp_path / "better.json", _artifact(0.98))
    worse = _write(tmp_path / "worse.json", _artifact(1.01))

    report = build_selection(
        {"better": better, "worse": worse},
        reference,
        base_mean_limit=0.0,
        seen_mean_limit=0.0,
        held_mean_limit=0.0,
        per_tau_limit=0.0,
        surface_mean_limit=0.0,
    )

    assert report["selected"] == "better"
    assert report["eligible"] == ["better"]
    result = report["candidates"]["better"]
    assert result["base_field_tau_wins"] == 100
    assert result["base_field_tau_total"] == 100
    assert result["base_mean_relative_delta"] == pytest.approx(-0.02)


def test_rejects_evaluation_protocol_mismatch(tmp_path) -> None:
    reference = _write(tmp_path / "reference.json", _artifact(1.0))
    candidate_artifact = _artifact(0.98)
    candidate_artifact["n_per_tau"]["5"] = 15
    candidate = _write(tmp_path / "candidate.json", candidate_artifact)

    with pytest.raises(ValueError, match="n_per_tau"):
        build_selection(
            {"candidate": candidate},
            reference,
            base_mean_limit=0.0,
            seen_mean_limit=0.0,
            held_mean_limit=0.0,
            per_tau_limit=0.0,
            surface_mean_limit=0.0,
        )
