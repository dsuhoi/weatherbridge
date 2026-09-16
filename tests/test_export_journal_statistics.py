from __future__ import annotations

import csv
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.eval.export_journal_statistics import _annotate_p_censoring, export_statistics


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(delta: float, p_value: float) -> dict:
    return {
        "better": "lower",
        "left": 1.0 + delta,
        "right": 1.0,
        "delta_left_minus_right": delta,
        "delta_ci95": [delta - 0.01, delta, delta + 0.01],
        "p_paired_block_permutation": p_value,
        "p_holm": min(1.0, 2.0 * p_value),
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    pairwise_path = tmp_path / "6h/2020/full_year/paired_rmse_weatherbridge_detail.json"
    cell = {
        "left_rmse": 0.9,
        "right_rmse": 1.0,
        "delta_left_minus_right": -0.1,
        "delta_ci95": [-0.12, -0.1, -0.08],
        "p_paired_block_permutation": 0.01,
        "p_left_better_one_sided": 0.005,
        "p_left_worse_one_sided": 0.995,
        "p_left_better_holm": 0.01,
        "p_left_worse_holm": 1.0,
    }
    comparison = {
        "taus": [1],
        "n_windows": 20,
        "n_blocks": 4,
        "block_days": 7,
        "bootstrap_draws": 100,
        **cell,
        "per_tau": {"1": cell},
        "per_channel_tau": {"1": {"t2m": cell}},
        "cellwise_family": {
            "alpha": 0.05,
            "n_hypotheses": 1,
        },
    }
    pairwise_hash = _write_json(
        pairwise_path,
        {
            "schema_version": 1,
            "left": "weatherbridge_detail",
            "comparisons": {"weatherdcae_14m_6yr": comparison},
        },
    )

    spectral_pair_path = tmp_path / "spectra/6h_2020/paired_tau1.json"
    spectral_comparison = {
        "tau": 1,
        "n_windows": 20,
        "n_blocks": 4,
        "block_days": 7,
        "draws": 100,
        "per_channel": {
            "t2m": {name: _metric(-0.1, 0.01) for name in (
                "energy_log_error",
                "shape_log_error",
                "coherence",
                "signed_cospectrum",
            )},
            "u10": {name: _metric(0.1, 0.04) for name in (
                "energy_log_error",
                "shape_log_error",
                "coherence",
                "signed_cospectrum",
            )},
        },
    }
    spectral_pair_hash = _write_json(
        spectral_pair_path,
        {
            "schema_version": 1,
            "left": "weatherbridge",
            "comparisons": {"weatherdcae_14m": spectral_comparison},
        },
    )
    spectral_summary_path = (
        tmp_path
        / "spectra/6h_2020/global_scalar_weatherbridge_vs_weatherdcae_14m.json"
    )
    metric_summary = {
        "multiplicity_family": {
            "method": "Holm-Bonferroni",
            "n_hypotheses": 2,
            "alpha": 0.05,
        }
    }
    spectral_summary_hash = _write_json(
        spectral_summary_path,
        {
            "schema_version": 2,
            "reference": "weatherdcae_14m",
            "years": [2020],
            "taus": [1],
            "yearly": {
                "2020": {
                    name: metric_summary
                    for name in (
                        "energy_log_error",
                        "shape_log_error",
                        "coherence",
                        "signed_cospectrum",
                    )
                }
            },
            "input_sha256": {str(spectral_pair_path): spectral_pair_hash},
        },
    )

    champion_path = tmp_path / "champion.json"
    _write_json(
        champion_path,
        {
            "schema_version": 1,
            "winner": "weatherbridge_detail",
            "input_sha256": {
                str(pairwise_path): pairwise_hash,
                str(spectral_summary_path): spectral_summary_hash,
            },
        },
    )
    return champion_path, pairwise_path, spectral_pair_path


def test_export_statistics_flattens_bound_tests(tmp_path: Path) -> None:
    champion_path, _, spectral_pair_path = _fixture(tmp_path)
    csv_path = tmp_path / "statistics.csv"
    manifest_path = tmp_path / "statistics.json"

    result = export_statistics(champion_path, csv_path, manifest_path)

    assert result["status"] == "complete"
    assert result["row_count"] == 11
    assert b"\r\n" not in csv_path.read_bytes()
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    field_row = next(
        row
        for row in rows
        if row["family"] == "rmse" and row["scope"] == "field_hour"
    )
    assert field_row["p_holm_left_worse"] == "1.0"
    assert field_row["p_censored"] == "false"
    spectral = [row for row in rows if row["family"] == "scalar_spectral"]
    assert {float(row["p_holm_two_sided"]) for row in spectral} == {0.02, 0.04}
    assert result["source_sha256"][str(spectral_pair_path)] == hashlib.sha256(
        spectral_pair_path.read_bytes()
    ).hexdigest()


def test_p_value_floor_is_marked_as_censored() -> None:
    row = {"p_raw_two_sided": 1.0 / 5001.0, "draws": 5000}
    _annotate_p_censoring(row)
    assert row["p_censored"] == "true"


def test_export_statistics_rejects_stale_champion_input(tmp_path: Path) -> None:
    champion_path, pairwise_path, _ = _fixture(tmp_path)
    pairwise_path.write_text("{}\n")

    with pytest.raises(ValueError, match="stale champion input"):
        export_statistics(
            champion_path,
            tmp_path / "statistics.csv",
            tmp_path / "statistics.json",
        )


def test_export_statistics_keeps_every_named_pairwise_comparison(
    tmp_path: Path,
) -> None:
    champion_path, pairwise_path, _ = _fixture(tmp_path)
    pairwise = json.loads(pairwise_path.read_text())
    comparison = next(iter(pairwise["comparisons"].values()))
    pairwise["comparisons"]["refine"] = copy.deepcopy(comparison)
    pairwise_hash = _write_json(pairwise_path, pairwise)
    champion = json.loads(champion_path.read_text())
    champion["input_sha256"][str(pairwise_path)] = pairwise_hash
    _write_json(champion_path, champion)

    result = export_statistics(
        champion_path,
        tmp_path / "statistics.csv",
        tmp_path / "statistics.json",
    )

    assert result["row_count"] == 14
    with (tmp_path / "statistics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {
        row["reference"] for row in rows if row["family"] == "rmse"
    } == {"weatherdcae_14m_6yr", "refine"}


def test_export_statistics_can_snapshot_remote_sources(tmp_path: Path) -> None:
    champion_path, _, _ = _fixture(tmp_path)
    source_dir = tmp_path / "paper/supplementary_data_1_sources"

    result = export_statistics(
        champion_path,
        tmp_path / "paper/statistics.csv",
        tmp_path / "paper/statistics.json",
        source_dir,
    )

    assert result["source_sha256"]
    assert result["source_original_paths"]
    assert all(Path(path).suffix == ".json" for path in result["source_sha256"])
    for source_name, expected in result["source_sha256"].items():
        source = Path(source_name)
        assert source.is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected


def test_export_statistics_replays_from_hash_bound_snapshots(tmp_path: Path) -> None:
    champion_path, pairwise_path, spectral_pair_path = _fixture(tmp_path)
    source_dir = tmp_path / "paper/supplementary_data_1_sources"
    first_csv = tmp_path / "paper/statistics.csv"
    export_statistics(
        champion_path,
        first_csv,
        tmp_path / "paper/statistics.json",
        source_dir,
    )
    expected_csv = first_csv.read_bytes()

    spectral_summary = next((tmp_path / "spectra/6h_2020").glob("global_*.json"))
    pairwise_path.unlink()
    spectral_pair_path.unlink()
    spectral_summary.unlink()

    replay_csv = tmp_path / "paper/statistics_replay.csv"
    result = export_statistics(
        champion_path,
        replay_csv,
        tmp_path / "paper/statistics_replay.json",
        source_dir,
    )

    assert result["row_count"] == 11
    assert replay_csv.read_bytes() == expected_csv
