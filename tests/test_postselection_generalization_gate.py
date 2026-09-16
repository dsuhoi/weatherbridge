import hashlib
import json
from pathlib import Path

import pytest

from tools.eval.check_postselection_generalization_gate import check_gate
from tools.eval.region_season_12h_eval import REGIONS, SURFACE_REGIMES
from weather_time_interp.metrics.physical_consistency import (
    CANONICAL_24_CHANNELS,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path: Path, delta_t_hours: int = 12):
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": 12,
                "winner": "candidate",
                "models": {
                    "candidate": {},
                    "reference": {},
                },
                "ood_attached_at_selection_time": False,
            }
        )
    )
    selection_sha256 = _sha256(selection)
    forecast = tmp_path / "forecast.json"
    forecast.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "winner": "candidate",
                "diagnostic_only": True,
                "selection_independent": True,
                "block_unit": "forecast_initialization",
                "draws": 5000,
                "seed": 2027,
                "forecast_data_provenance_sha256": "9" * 64,
                "protocol": {
                    "delta_t_hours": delta_t_hours,
                    "evaluated_taus": (
                        [1, 2, 3, 4, 5]
                        if delta_t_hours == 6
                        else list(range(1, 12))
                    ),
                    "seen_taus": (
                        [1, 3, 5]
                        if delta_t_hours == 6
                        else [1, 2, 3, 5, 7, 9, 10, 11]
                    ),
                    "unseen_taus": (
                        [2, 4]
                        if delta_t_hours == 6
                        else [4, 6, 8]
                    ),
                    "unseen_noninferiority_margin_rmse": 0.02,
                },
                "no_significant_unseen_forecast_regression": True,
                "significant_unseen_forecast_regressions": [],
                "unseen_forecast_noninferior_2pct_rmse": True,
                "unseen_forecast_noninferiority_failures": [],
                "provenance": {
                    "selection": {"sha256": selection_sha256}
                },
            }
        )
    )
    exchange = tmp_path / "exchange.json"
    exchange.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "winner": "candidate",
                "diagnostic_only": True,
                "protocol": {
                    "delta_t_hours": delta_t_hours,
                    "all_taus": (
                        [1, 2, 3, 4, 5]
                        if delta_t_hours == 6
                        else list(range(1, 12))
                    ),
                    "seen_taus": (
                        [1, 3, 5]
                        if delta_t_hours == 6
                        else [1, 2, 3, 5, 7, 9, 10, 11]
                    ),
                    "unseen_taus": (
                        [2, 4]
                        if delta_t_hours == 6
                        else [4, 6, 8]
                    ),
                    "channels": list(CANONICAL_24_CHANNELS),
                    "samples_per_date": 2,
                    "eval_days_per_month": 2,
                    "block_days": 7,
                    "draws": 5000,
                    "seed": 2027,
                    "noninferiority_margin": 0.02,
                },
                "evaluation_input_provenance_sha256": "d" * 64,
                "evaluation_dataset_provenance_sha256": {
                    "2020": "e" * 64,
                    "2021": "f" * 64,
                },
                "no_significant_2021_unseen_exchange_regression": True,
                "significant_2021_unseen_exchange_regressions": [],
                "winner_2021_unseen_exchange_noninferior_2pct": True,
                "winner_2021_unseen_exchange_noninferiority_failures": [],
                "winner_endpoint_rmse": {
                    "2020": {"endpoint0": 0.0, "endpoint1": 0.0},
                    "2021": {"endpoint0": 0.0, "endpoint1": 0.0},
                },
                "selection_manifest": {"sha256": selection_sha256},
            }
        )
    )
    region = tmp_path / "region.json"
    region.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "winner": "candidate",
                "diagnostic_only": True,
                "protocol": {
                    "delta_t_hours": delta_t_hours,
                    "regions": list(REGIONS) + list(SURFACE_REGIMES),
                    "channels": list(CANONICAL_24_CHANNELS),
                    "samples_per_date": 2,
                    "eval_days_per_month": 8,
                    "block_days": 7,
                    "draws": 5000,
                    "seed": 2027,
                    "noninferiority_margin_rmse": 0.05,
                },
                "evaluation_input_provenance_sha256": "a" * 64,
                "evaluation_dataset_provenance_sha256": {
                    "2020": "b" * 64,
                    "2021": "c" * 64,
                },
                "no_significant_2021_region_season_regression": True,
                "significant_2021_region_season_regressions": [],
                "winner_2021_region_season_noninferior_5pct": True,
                "winner_2021_region_season_noninferiority_failures": [],
                "selection_manifest": {"sha256": selection_sha256},
            }
        )
    )
    return selection, forecast, exchange, region


def _seed_artifacts(tmp_path: Path, selection: Path):
    selection_sha256 = _sha256(selection)
    seed_report = tmp_path / "seed_report.json"
    seed_report.write_text(
        json.dumps(
            {
                "schema_version": 6,
                "selection_sha256": selection_sha256,
                "candidate": "candidate",
                "reference": "weatherbridge_ref",
                "seeds": [202707, 202708, 202709],
                "candidate_seed_artifact_mode": (
                    "fresh_postselection_replicates"
                ),
                "training_objective": {
                    "candidate_lambda_hf": 0.05,
                    "reference_lambda_hf": 0.0,
                    "matched": False,
                },
                "primary_quality_superiority_seed_consistent": True,
            }
        )
    )
    ood_spectra = tmp_path / "ood_spectra.json"
    ood_spectra.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "selection_sha256": selection_sha256,
                "candidate": "candidate",
                "reference": "weatherbridge_ref",
                "seeds": [202707, 202708, 202709],
                "candidate_seed_artifact_mode": (
                    "fresh_postselection_replicates"
                ),
                "checkpoint_linkage_verified": True,
                "spectral_protocol": {
                    "confirmation_year": 2021,
                    "used_for_model_selection": False,
                },
                "ood_spectral_superiority_seed_consistent": True,
            }
        )
    )
    return seed_report, ood_spectra


def _ood_spectral_summary(tmp_path: Path, selection: Path):
    path = tmp_path / "ood_spectral_summary.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_sha256": _sha256(selection),
                "winner": "candidate",
                "models": ["candidate", "reference"],
                "protocol": {
                    "delta_t_hours": 12,
                    "confirmation_year": 2021,
                    "used_for_model_selection": False,
                    "spectral_taus": [4, 6, 8],
                    "hf_ell_min": 180,
                    "lmax": 359,
                    "channels": 24,
                },
                "paired_window_index_sha256": "a" * 64,
                "spectral_grid_sha256": "b" * 64,
                "spectral_channel_names_sha256": "c" * 64,
                "evaluation_input_provenance_sha256": "d" * 64,
                "evaluation_dataset_provenance_sha256": "e" * 64,
                "no_significant_ood_spectral_regression": True,
                "significant_ood_spectral_regressions": [],
            }
        )
    )
    return path


def test_gate_requires_all_postselection_diagnostics(
    tmp_path: Path,
) -> None:
    selection, forecast, exchange, region = _artifacts(tmp_path)

    result = check_gate(
        selection,
        forecast,
        exchange,
        region,
        delta_t_hours=12,
    )

    assert result["winner"] == "candidate"
    assert result["delta_t_hours"] == 12
    assert result["anchor_exchange_used_as_gate"] is False
    assert result["anchor_exchange_diagnostic"]["noninferior_2pct"] is True


def test_gate_accepts_seed_and_ood_spectral_confirmation(
    tmp_path: Path,
) -> None:
    selection, forecast, exchange, region = _artifacts(
        tmp_path,
        delta_t_hours=6,
    )
    seed_report, ood_spectra = _seed_artifacts(tmp_path, selection)

    result = check_gate(
        selection,
        forecast,
        exchange,
        region,
        delta_t_hours=6,
        seed_report_path=seed_report,
        ood_spectra_path=ood_spectra,
    )

    assert result["seed_consistent_candidate"] == "candidate"
    assert result["seed_comparison_sha256"] == _sha256(seed_report)
    assert result["ood_spectra_sha256"] == _sha256(ood_spectra)


def test_gate_rejects_missing_ood_spectrum(tmp_path: Path) -> None:
    selection, forecast, exchange, region = _artifacts(
        tmp_path,
        delta_t_hours=6,
    )
    seed_report, _ = _seed_artifacts(tmp_path, selection)

    with pytest.raises(ValueError, match="supplied together"):
        check_gate(
            selection,
            forecast,
            exchange,
            region,
            delta_t_hours=6,
            seed_report_path=seed_report,
        )


def test_gate_rejects_seed_quality_failure(tmp_path: Path) -> None:
    selection, forecast, exchange, region = _artifacts(
        tmp_path,
        delta_t_hours=6,
    )
    seed_report, ood_spectra = _seed_artifacts(tmp_path, selection)
    payload = json.loads(seed_report.read_text())
    payload["primary_quality_superiority_seed_consistent"] = False
    seed_report.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="quality superiority"):
        check_gate(
            selection,
            forecast,
            exchange,
            region,
            delta_t_hours=6,
            seed_report_path=seed_report,
            ood_spectra_path=ood_spectra,
        )


@pytest.mark.parametrize("artifact", ("seed", "ood"))
def test_gate_rejects_reused_selection_seed(
    tmp_path: Path,
    artifact: str,
) -> None:
    selection, forecast, exchange, region = _artifacts(
        tmp_path,
        delta_t_hours=6,
    )
    seed_report, ood_spectra = _seed_artifacts(tmp_path, selection)
    path = seed_report if artifact == "seed" else ood_spectra
    payload = json.loads(path.read_text())
    payload["candidate_seed_artifact_mode"] = (
        "selection_seed_plus_replicates"
    )
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="protocol mismatch"):
        check_gate(
            selection,
            forecast,
            exchange,
            region,
            delta_t_hours=6,
            seed_report_path=seed_report,
            ood_spectra_path=ood_spectra,
        )


def test_gate_rejects_ood_spectral_failure(tmp_path: Path) -> None:
    selection, forecast, exchange, region = _artifacts(
        tmp_path,
        delta_t_hours=6,
    )
    seed_report, ood_spectra = _seed_artifacts(tmp_path, selection)
    payload = json.loads(ood_spectra.read_text())
    payload["ood_spectral_superiority_seed_consistent"] = False
    ood_spectra.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="spectral superiority"):
        check_gate(
            selection,
            forecast,
            exchange,
            region,
            delta_t_hours=6,
            seed_report_path=seed_report,
            ood_spectra_path=ood_spectra,
        )


def test_gate_requires_clean_12h_ood_spectral_summary(
    tmp_path: Path,
) -> None:
    selection, forecast, exchange, region = _artifacts(tmp_path)
    ood_summary = _ood_spectral_summary(tmp_path, selection)

    result = check_gate(
        selection,
        forecast,
        exchange,
        region,
        delta_t_hours=12,
        ood_spectral_summary_path=ood_summary,
    )

    assert result["ood_spectral_summary_sha256"] == _sha256(ood_summary)


def test_gate_rejects_12h_ood_spectral_regression(
    tmp_path: Path,
) -> None:
    selection, forecast, exchange, region = _artifacts(tmp_path)
    ood_summary = _ood_spectral_summary(tmp_path, selection)
    payload = json.loads(ood_summary.read_text())
    payload["no_significant_ood_spectral_regression"] = False
    payload["significant_ood_spectral_regressions"] = [
        {"tau": 6, "challenger": "weatherbridge_ref"}
    ]
    ood_summary.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="spectral generalization"):
        check_gate(
            selection,
            forecast,
            exchange,
            region,
            delta_t_hours=12,
            ood_spectral_summary_path=ood_summary,
        )


def test_gate_rejects_incomplete_ood_spectral_protocol(
    tmp_path: Path,
) -> None:
    selection, forecast, exchange, region = _artifacts(tmp_path)
    ood_summary = _ood_spectral_summary(tmp_path, selection)
    payload = json.loads(ood_summary.read_text())
    payload["protocol"]["spectral_taus"] = [4]
    ood_summary.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="spectral-summary protocol"):
        check_gate(
            selection,
            forecast,
            exchange,
            region,
            delta_t_hours=12,
            ood_spectral_summary_path=ood_summary,
        )


def test_gate_rejects_forecast_regression(tmp_path: Path) -> None:
    selection, forecast, exchange, region = _artifacts(tmp_path)
    payload = json.loads(forecast.read_text())
    payload["no_significant_unseen_forecast_regression"] = False
    payload["significant_unseen_forecast_regressions"] = ["reference"]
    forecast.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="forecast-anchor"):
        check_gate(
            selection,
            forecast,
            exchange,
            region,
            delta_t_hours=12,
        )


@pytest.mark.parametrize(
    ("artifact", "flag", "failures", "error"),
    (
        (
            "forecast",
            "unseen_forecast_noninferior_2pct_rmse",
            "unseen_forecast_noninferiority_failures",
            "forecast-anchor",
        ),
        (
            "region",
            "winner_2021_region_season_noninferior_5pct",
            "winner_2021_region_season_noninferiority_failures",
            "region-season",
        ),
    ),
)
def test_gate_rejects_unproven_noninferiority(
    tmp_path: Path,
    artifact: str,
    flag: str,
    failures: str,
    error: str,
) -> None:
    selection, forecast, exchange, region = _artifacts(tmp_path)
    path = {
        "forecast": forecast,
        "exchange": exchange,
        "region": region,
    }[artifact]
    payload = json.loads(path.read_text())
    payload[flag] = False
    payload[failures] = ["reference"]
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=error):
        check_gate(
            selection,
            forecast,
            exchange,
            region,
            delta_t_hours=12,
        )


def test_gate_reports_but_does_not_veto_anchor_exchange_asymmetry(
    tmp_path: Path,
) -> None:
    selection, forecast, exchange, region = _artifacts(tmp_path)
    payload = json.loads(exchange.read_text())
    payload["no_significant_2021_unseen_exchange_regression"] = False
    payload["significant_2021_unseen_exchange_regressions"] = [
        "reference"
    ]
    payload["winner_2021_unseen_exchange_noninferior_2pct"] = False
    payload["winner_2021_unseen_exchange_noninferiority_failures"] = [
        "reference"
    ]
    exchange.write_text(json.dumps(payload))

    result = check_gate(
        selection,
        forecast,
        exchange,
        region,
        delta_t_hours=12,
    )

    diagnostic = result["anchor_exchange_diagnostic"]
    assert diagnostic["no_significant_2021_unseen_regression"] is False
    assert diagnostic["noninferior_2pct"] is False
    assert diagnostic["noninferiority_failures"] == ["reference"]


def test_gate_rejects_malformed_anchor_exchange_endpoint_report(
    tmp_path: Path,
) -> None:
    selection, forecast, exchange, region = _artifacts(tmp_path)
    payload = json.loads(exchange.read_text())
    payload["winner_endpoint_rmse"]["2021"]["endpoint1"] = "missing"
    exchange.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="anchor-exchange diagnostic"):
        check_gate(
            selection,
            forecast,
            exchange,
            region,
            delta_t_hours=12,
        )


def test_gate_rejects_incomplete_region_protocol(
    tmp_path: Path,
) -> None:
    selection, forecast, exchange, region = _artifacts(tmp_path)
    payload = json.loads(region.read_text())
    payload["protocol"]["regions"].remove("Ocean")
    region.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="region-season diagnostic"):
        check_gate(
            selection,
            forecast,
            exchange,
            region,
            delta_t_hours=12,
        )


def test_gate_rejects_stale_selection_hash(tmp_path: Path) -> None:
    selection, forecast, exchange, region = _artifacts(tmp_path)
    selection.write_text(
        json.dumps(
            {
                "schema_version": 12,
                "winner": "candidate",
                "models": {
                    "candidate": {},
                    "reference": {},
                },
                "ood_attached_at_selection_time": False,
                "changed": True,
            }
        )
    )

    with pytest.raises(ValueError, match="selection hash"):
        check_gate(
            selection,
            forecast,
            exchange,
            region,
            delta_t_hours=12,
        )


def test_gate_rejects_pre_robust_selection_schema(
    tmp_path: Path,
) -> None:
    selection, forecast, exchange, region = _artifacts(tmp_path)
    payload = json.loads(selection.read_text())
    payload["schema_version"] = 11
    selection.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="invalid frozen selection"):
        check_gate(
            selection,
            forecast,
            exchange,
            region,
            delta_t_hours=12,
        )


def test_gate_rejects_region_season_regression(tmp_path: Path) -> None:
    selection, forecast, exchange, region = _artifacts(tmp_path)
    payload = json.loads(region.read_text())
    payload["no_significant_2021_region_season_regression"] = False
    payload["significant_2021_region_season_regressions"] = [
        {"region": "Arctic", "season": "DJF", "model": "reference"}
    ]
    region.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="region-season"):
        check_gate(
            selection,
            forecast,
            exchange,
            region,
            delta_t_hours=12,
        )
