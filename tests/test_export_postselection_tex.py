from tools.eval.export_postselection_tex import render


def test_export_reports_aggregate_but_not_universal_claim() -> None:
    payload = {
        "frozen_winner": "weatherbridge_detail",
        "status": "confirmed_aggregate",
        "aggregate_confirmation_pass": True,
        "universal_dominance_claim_allowed": False,
        "horizons": {
            "6h": {
                "delta_normalized_rmse": -0.001,
                "relative_delta_pct": -1.0,
                "delta_ci95": [-0.002, -0.001, -0.0001],
            },
            "12h": {
                "delta_normalized_rmse": -0.002,
                "relative_delta_pct": -1.5,
                "delta_ci95": [-0.003, -0.002, -0.0002],
            },
        },
    }
    output = render(payload)
    assert "confirming the frozen aggregate endpoint" in output
    assert "before data retrieval" not in output
    assert "no claim of universal diagnostic dominance" in output
    assert "48 disjoint" in output


def test_export_reference_retention_has_no_candidate_table() -> None:
    output = render(
        {
            "frozen_winner": "weatherdcae_14m",
            "status": "reference_retained_holdout_unopened",
        }
    )
    assert "remained the selected reference" in output
    assert "endpoint was not evaluated" in output
    assert "begin{table}" not in output


def test_export_separates_candidate_result_from_retained_selector() -> None:
    output = render(
        {
            "frozen_winner": "weatherdcae_14m",
            "evaluated_candidate": "flow_spectral",
            "selector_confirmation": False,
            "status": "confirmed_aggregate",
            "primary_endpoint_evaluated": True,
            "aggregate_confirmation_pass": True,
            "universal_dominance_claim_allowed": False,
            "horizons": {
                "6h": {
                    "delta_normalized_rmse": -0.0086,
                    "relative_delta_pct": -12.6,
                    "delta_ci95": [-0.0089, -0.0086, -0.0084],
                },
                "12h": {
                    "delta_normalized_rmse": -0.0082,
                    "relative_delta_pct": -7.1,
                    "delta_ci95": [-0.0084, -0.0082, -0.0081],
                },
            },
        }
    )
    assert "Pre-specified 2022 comparison" in output
    assert "confirms its aggregate accuracy advantage" in output
    assert "retained WeatherDCAE-14M" in output
