from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.eval.select_strict_flow_dominance import build_selection


FIELDS = [
    *(f"T{level}" for level in (1000, 925, 850, 700)),
    *(f"U{level}" for level in (1000, 925, 850, 700)),
    *(f"V{level}" for level in (1000, 925, 850, 700)),
    *(f"Q{level}" for level in (1000, 925, 850, 700)),
    *(f"Z{level}" for level in (1000, 925, 850, 700)),
    "t2m",
    "u10",
    "v10",
    "mslp",
]


def _artifact(scale: float) -> dict:
    return {
        "years": [2020],
        "n_per_tau": 8,
        "evaluation_protocol": {"proper_rmse": True},
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


def test_requires_all_100_base_field_hour_cells_to_win(tmp_path) -> None:
    reference = _write(tmp_path / "reference.json", _artifact(1.0))
    winner = _artifact(0.99)
    failure = _artifact(0.99)
    failure["per_tau"]["4"]["model"]["rmse_norm_Z700"] = 1.001

    report = build_selection(
        {
            "winner": _write(tmp_path / "winner.json", winner),
            "failure": _write(tmp_path / "failure.json", failure),
        },
        reference,
        cell_limit=0.0,
    )

    assert report["selected"] == "winner"
    assert report["candidates"]["winner"]["winning_cells"] == 100
    failed = report["candidates"]["failure"]
    assert not failed["eligible"]
    assert failed["winning_cells"] == 99
    assert failed["failing_cells"][0]["tau_hour"] == 4
    assert failed["failing_cells"][0]["field"] == "Z700"


def test_rejects_protocol_mismatch(tmp_path) -> None:
    reference = _write(tmp_path / "reference.json", _artifact(1.0))
    candidate = _artifact(0.9)
    candidate["n_per_tau"] = 7

    with pytest.raises(ValueError, match="protocol mismatch"):
        build_selection(
            {"candidate": _write(tmp_path / "candidate.json", candidate)},
            reference,
            cell_limit=0.0,
        )


def test_include_q_expands_gate_to_all_120_cells(tmp_path) -> None:
    reference = _write(tmp_path / "reference.json", _artifact(1.0))
    candidate = _artifact(0.99)
    candidate["per_tau"]["2"]["model"]["rmse_norm_Q850"] = 1.001

    report = build_selection(
        {"candidate": _write(tmp_path / "candidate.json", candidate)},
        reference,
        cell_limit=0.0,
        include_q=True,
    )

    result = report["candidates"]["candidate"]
    assert result["total_cells"] == 120
    assert result["winning_cells"] == 119
    assert result["failing_cells"][0]["field"] == "Q850"


def test_reference_envelope_uses_best_model_per_cell(tmp_path) -> None:
    flow = _artifact(1.0)
    dcae = _artifact(1.1)
    dcae["per_tau"]["3"]["model"]["rmse_norm_t2m"] = 0.8
    candidate = _artifact(0.9)
    candidate["per_tau"]["3"]["model"]["rmse_norm_t2m"] = 0.85

    report = build_selection(
        {"candidate": _write(tmp_path / "candidate.json", candidate)},
        _write(tmp_path / "flow.json", flow),
        cell_limit=0.0,
        include_q=True,
        additional_references={
            "dcae": _write(tmp_path / "dcae.json", dcae),
        },
    )

    result = report["candidates"]["candidate"]
    assert not result["eligible"]
    assert result["winning_cells"] == 119
    assert result["failing_cells"][0]["field"] == "t2m"
    assert report["reference_envelope_source"]["3"]["t2m"] == "dcae"


def test_ignores_non_rmse_artifact_flags_in_protocol(tmp_path) -> None:
    reference = _artifact(1.0)
    reference["evaluation_protocol"]["save_window_metrics"] = True
    candidate = _artifact(0.9)
    candidate["evaluation_protocol"]["save_window_metrics"] = False
    candidate["evaluation_protocol"]["save_physical_metrics"] = True

    report = build_selection(
        {"candidate": _write(tmp_path / "candidate.json", candidate)},
        _write(tmp_path / "reference.json", reference),
        cell_limit=0.0,
        include_q=True,
    )

    assert report["selected"] == "candidate"
