import json
from pathlib import Path

import tools.eval.assess_avg3_deployment as deployment
from tools.eval.assess_avg3_deployment import assess, summarize


def _result(
    delta: float,
    *,
    p_value: float = 1.0,
    hours: tuple[int, ...] = (),
) -> dict:
    result = {
        "delta_left_minus_right": delta,
        "p_paired_block_permutation": p_value,
    }
    if hours:
        result["per_tau"] = {
            str(hour): _result(delta, p_value=p_value)
            for hour in hours
        }
    return result


def _comparisons(
    hours: tuple[int, ...] = (2, 4),
) -> tuple[dict, dict, dict, dict]:
    averaged = "candidate_avg3"
    field = {
        "2020": {
            "seen": {averaged: _result(0.01)},
            "unseen": {averaged: _result(0.02, hours=hours)},
        },
        "2021": {
            "seen": {averaged: _result(0.01)},
            "unseen": {averaged: _result(0.01, hours=hours)},
        },
    }
    acc = {
        year: {
            "unseen": {averaged: _result(-0.01)},
        }
        for year in ("2020", "2021")
    }
    physical = {
        year: {
            "unseen": {
                averaged: {
                    "diagnostics": {
                        "divergence": _result(0.01),
                        "gradient": _result(0.01),
                    }
                }
            }
        }
        for year in ("2020", "2021")
    }
    spectral = {
        str(hour): {
            averaged: {
                "energy_log_error": _result(0.01),
                "shape_log_error": _result(0.01),
                "coherence": _result(-0.01),
            }
        }
        for hour in hours
    }
    return field, acc, physical, spectral


def test_avg3_is_recommended_after_2020_gain_and_2021_confirmation() -> None:
    field, acc, physical, spectral = _comparisons()

    report = summarize(
        "candidate",
        "candidate_avg3",
        field=field,
        acc=acc,
        physical=physical,
        spectral=spectral,
    )

    assert report["selected_on_2020"] is True
    assert report["confirmed_on_frozen_2021"] is True
    assert report["recommended_name"] == "candidate_avg3"


def test_avg3_is_rejected_on_holm_significant_spectral_regression() -> None:
    field, acc, physical, spectral = _comparisons()
    spectral["2"]["candidate_avg3"]["shape_log_error"] = _result(
        -0.2,
        p_value=0.001,
    )

    report = summarize(
        "candidate",
        "candidate_avg3",
        field=field,
        acc=acc,
        physical=physical,
        spectral=spectral,
    )

    assert report["selected_on_2020"] is False
    assert report["recommended_name"] == "candidate"
    assert report["significant_2020_safety_regressions"][0][
        "metric"
    ] == "spectral_h2/shape_log_error"


def test_frozen_2021_regression_vetoes_avg3_deployment() -> None:
    field, acc, physical, spectral = _comparisons()
    field["2021"]["unseen"]["candidate_avg3"] = _result(
        -0.2,
        p_value=0.001,
        hours=(2, 4),
    )

    report = summarize(
        "candidate",
        "candidate_avg3",
        field=field,
        acc=acc,
        physical=physical,
        spectral=spectral,
    )

    assert report["selected_on_2020"] is True
    assert report["confirmed_on_frozen_2021"] is False
    assert report["recommended_name"] == "candidate"


def test_12h_gate_requires_all_three_held_hours() -> None:
    field, acc, physical, spectral = _comparisons((4, 6, 8))
    field["2020"]["unseen"]["candidate_avg3"]["per_tau"]["6"] = _result(
        -0.01
    )

    report = summarize(
        "candidate",
        "candidate_avg3",
        field=field,
        acc=acc,
        physical=physical,
        spectral=spectral,
        held_hours=(4, 6, 8),
    )

    assert report["held_hours"] == [4, 6, 8]
    assert report["improves_all_2020_held_hours"] is False
    assert report["recommended_name"] == "candidate"


def test_missing_avg3_artifacts_retain_raw_winner(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"winner": "candidate"}))

    report = assess(
        selection,
        tmp_path / "2020",
        tmp_path / "2021",
        tmp_path / "spectra",
        block_days=7,
        draws=100,
        seed=2027,
    )

    assert report["available"] is False
    assert report["recommended_name"] == "candidate"


def test_12h_provenance_loader_uses_all_spectral_hours(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_hash = "a" * 64
    averaged_hash = "b" * 64
    common = {
        "window_index_sha256": "windows",
        "evaluation_input_provenance": {"inputs": "same"},
        "evaluation_dataset_provenance": {"dataset": "same"},
    }

    def fake_field(path: Path) -> dict:
        return {
            **common,
            "checkpoint_sha256": (
                averaged_hash if "avg3" in path.name else raw_hash
            ),
        }

    spectral_calls = []

    def fake_spectral(
        root: Path,
        model: str,
        hf_ell_min: int,
        taus: set[int],
        *,
        required_lmax: int,
        expected_channels: int,
    ) -> dict:
        spectral_calls.append(
            (
                root,
                model,
                hf_ell_min,
                taus,
                required_lmax,
                expected_channels,
            )
        )
        return {
            **common,
            "checkpoint_sha256": (
                averaged_hash if model.endswith("_avg3") else raw_hash
            ),
        }

    monkeypatch.setattr(deployment, "load_field_metrics", fake_field)
    monkeypatch.setattr(deployment, "load_spectral_metrics", fake_spectral)
    result = deployment._validate_metric_provenance(
        {
            "models": {
                "candidate": {"checkpoint_sha256": raw_hash},
            }
        },
        "candidate",
        "candidate_avg3",
        tmp_path / "2020",
        tmp_path / "2021",
        tmp_path / "spectra",
        spectral_taus=(4, 6, 8),
    )

    assert result == {
        "raw_checkpoint_sha256": raw_hash,
        "averaged_checkpoint_sha256": averaged_hash,
    }
    assert len(spectral_calls) == 2
    assert all(call[3] == {4, 6, 8} for call in spectral_calls)
    assert all(call[4:] == (359, 24) for call in spectral_calls)
