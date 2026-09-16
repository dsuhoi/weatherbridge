#!/usr/bin/env python3
"""Validate post-selection diagnostics and fail on physical OOD regressions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tools.eval.region_season_12h_eval import REGIONS, SURFACE_REGIMES
from weather_time_interp.metrics.physical_consistency import (
    CANONICAL_24_CHANNELS,
)


EXPECTED_REGIONS = list(REGIONS) + list(SURFACE_REGIMES)
EXPECTED_SPECTRAL_TAUS = {
    6: [2, 4],
    12: [4, 6, 8],
}
EXPECTED_TAU_PROTOCOLS = {
    6: {
        "all": [1, 2, 3, 4, 5],
        "seen": [1, 3, 5],
        "unseen": [2, 4],
    },
    12: {
        "all": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "seen": [1, 2, 3, 5, 7, 9, 10, 11],
        "unseen": [4, 6, 8],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_endpoint_report(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"2020", "2021"}:
        return False
    for endpoints in value.values():
        if (
            not isinstance(endpoints, dict)
            or set(endpoints) != {"endpoint0", "endpoint1"}
        ):
            return False
        for error in endpoints.values():
            if (
                isinstance(error, bool)
                or not isinstance(error, (int, float))
                or not math.isfinite(float(error))
                or float(error) < 0.0
            ):
                return False
    return True


def check_gate(
    selection_path: Path,
    forecast_path: Path,
    exchange_path: Path,
    region_path: Path,
    *,
    delta_t_hours: int,
    seed_report_path: Path | None = None,
    ood_spectra_path: Path | None = None,
    ood_spectral_summary_path: Path | None = None,
) -> dict[str, Any]:
    selection = _load(selection_path)
    forecast = _load(forecast_path)
    exchange = _load(exchange_path)
    region = _load(region_path)
    selection_sha256 = _sha256(selection_path)
    winner = selection.get("winner")
    selection_models = selection.get("models")
    expected_taus = EXPECTED_TAU_PROTOCOLS.get(delta_t_hours)
    if (
        isinstance(selection.get("schema_version"), bool)
        or not isinstance(selection.get("schema_version"), int)
        or selection["schema_version"] < 12
        or not isinstance(winner, str)
        or not isinstance(selection_models, dict)
        or winner not in selection_models
        or expected_taus is None
        or selection.get("ood_attached_at_selection_time") is not False
    ):
        raise ValueError("invalid frozen selection")
    forecast_protocol = forecast.get("protocol", {})
    if (
        forecast.get("schema_version") != 1
        or forecast.get("winner") != winner
        or forecast.get("diagnostic_only") is not True
        or forecast.get("selection_independent") is not True
        or forecast_protocol.get("delta_t_hours") != delta_t_hours
        or forecast_protocol.get("evaluated_taus")
        != expected_taus["all"]
        or forecast_protocol.get("seen_taus") != expected_taus["seen"]
        or forecast_protocol.get("unseen_taus") != expected_taus["unseen"]
        or forecast_protocol.get(
            "unseen_noninferiority_margin_rmse"
        )
        != 0.02
        or forecast.get("block_unit") != "forecast_initialization"
        or forecast.get("draws") != 5000
        or forecast.get("seed") != 2027
        or not _is_sha256(
            forecast.get("forecast_data_provenance_sha256")
        )
    ):
        raise ValueError("forecast diagnostic protocol mismatch")
    if (
        forecast.get("provenance", {})
        .get("selection", {})
        .get("sha256")
        != selection_sha256
    ):
        raise ValueError("forecast diagnostic selection hash mismatch")
    if (
        forecast.get("no_significant_unseen_forecast_regression")
        is not True
        or forecast.get("significant_unseen_forecast_regressions") != []
        or forecast.get("unseen_forecast_noninferior_2pct_rmse")
        is not True
        or forecast.get(
            "unseen_forecast_noninferiority_failures"
        )
        != []
    ):
        raise ValueError("forecast-anchor generalization regression")

    exchange_protocol = exchange.get("protocol", {})
    exchange_dataset_hashes = exchange.get(
        "evaluation_dataset_provenance_sha256"
    )
    if (
        exchange.get("schema_version") != 1
        or exchange.get("winner") != winner
        or exchange.get("diagnostic_only") is not True
        or exchange_protocol.get("delta_t_hours") != delta_t_hours
        or exchange_protocol.get("all_taus") != expected_taus["all"]
        or exchange_protocol.get("seen_taus") != expected_taus["seen"]
        or exchange_protocol.get("unseen_taus")
        != expected_taus["unseen"]
        or exchange_protocol.get("channels")
        != list(CANONICAL_24_CHANNELS)
        or exchange_protocol.get("samples_per_date") != 2
        or exchange_protocol.get("eval_days_per_month") != 2
        or exchange_protocol.get("block_days") != 7
        or exchange_protocol.get("draws") != 5000
        or exchange_protocol.get("seed") != 2027
        or exchange_protocol.get("noninferiority_margin") != 0.02
        or not _is_sha256(
            exchange.get("evaluation_input_provenance_sha256")
        )
        or not isinstance(exchange_dataset_hashes, dict)
        or set(exchange_dataset_hashes) != {"2020", "2021"}
        or not all(
            _is_sha256(value)
            for value in exchange_dataset_hashes.values()
        )
        or not isinstance(
            exchange.get(
                "no_significant_2021_unseen_exchange_regression"
            ),
            bool,
        )
        or not isinstance(
            exchange.get(
                "significant_2021_unseen_exchange_regressions"
            ),
            list,
        )
        or not isinstance(
            exchange.get(
                "winner_2021_unseen_exchange_noninferior_2pct"
            ),
            bool,
        )
        or not isinstance(
            exchange.get(
                "winner_2021_unseen_exchange_noninferiority_failures"
            ),
            list,
        )
        or not _valid_endpoint_report(
            exchange.get("winner_endpoint_rmse")
        )
    ):
        raise ValueError("anchor-exchange diagnostic protocol mismatch")
    if (
        exchange.get("selection_manifest", {}).get("sha256")
        != selection_sha256
    ):
        raise ValueError("anchor-exchange selection hash mismatch")
    region_protocol = region.get("protocol", {})
    region_dataset_hashes = region.get(
        "evaluation_dataset_provenance_sha256"
    )
    if (
        region.get("schema_version") != 1
        or region.get("winner") != winner
        or region.get("diagnostic_only") is not True
        or region_protocol.get("delta_t_hours") != delta_t_hours
        or region_protocol.get("regions") != EXPECTED_REGIONS
        or region_protocol.get("channels")
        != list(CANONICAL_24_CHANNELS)
        or region_protocol.get("samples_per_date") != 2
        or region_protocol.get("eval_days_per_month") != 8
        or region_protocol.get("block_days") != 7
        or region_protocol.get("draws") != 5000
        or region_protocol.get("seed") != 2027
        or region_protocol.get("noninferiority_margin_rmse") != 0.05
        or not _is_sha256(
            region.get("evaluation_input_provenance_sha256")
        )
        or not isinstance(region_dataset_hashes, dict)
        or set(region_dataset_hashes) != {"2020", "2021"}
        or not all(
            _is_sha256(value)
            for value in region_dataset_hashes.values()
        )
    ):
        raise ValueError("region-season diagnostic protocol mismatch")
    if (
        region.get("selection_manifest", {}).get("sha256")
        != selection_sha256
    ):
        raise ValueError("region-season selection hash mismatch")
    if (
        region.get("no_significant_2021_region_season_regression")
        is not True
        or region.get("significant_2021_region_season_regressions") != []
        or region.get(
            "winner_2021_region_season_noninferior_5pct"
        )
        is not True
        or region.get(
            "winner_2021_region_season_noninferiority_failures"
        )
        != []
    ):
        raise ValueError("region-season generalization regression")
    result = {
        "winner": winner,
        "delta_t_hours": delta_t_hours,
        "selection_sha256": selection_sha256,
        "forecast_summary_sha256": _sha256(forecast_path),
        "exchange_summary_sha256": _sha256(exchange_path),
        "region_summary_sha256": _sha256(region_path),
        "anchor_exchange_used_as_gate": False,
        "anchor_exchange_diagnostic": {
            "no_significant_2021_unseen_regression": exchange[
                "no_significant_2021_unseen_exchange_regression"
            ],
            "significant_2021_unseen_regressions": exchange[
                "significant_2021_unseen_exchange_regressions"
            ],
            "noninferior_2pct": exchange[
                "winner_2021_unseen_exchange_noninferior_2pct"
            ],
            "noninferiority_failures": exchange[
                "winner_2021_unseen_exchange_noninferiority_failures"
            ],
            "winner_endpoint_rmse": exchange["winner_endpoint_rmse"],
        },
    }
    if (seed_report_path is None) != (ood_spectra_path is None):
        raise ValueError(
            "seed report and OOD spectrum must be supplied together"
        )
    if seed_report_path is not None and ood_spectra_path is not None:
        seed_report = _load(seed_report_path)
        ood_spectra = _load(ood_spectra_path)
        candidate = seed_report.get("candidate")
        expected_seeds = [202707, 202708, 202709]
        if (
            seed_report.get("schema_version") != 6
            or seed_report.get("selection_sha256") != selection_sha256
            or not isinstance(candidate, str)
            or not candidate
            or seed_report.get("reference") != "weatherbridge_ref"
            or seed_report.get("seeds") != expected_seeds
            or seed_report.get("candidate_seed_artifact_mode")
            != "fresh_postselection_replicates"
            or not isinstance(seed_report.get("training_objective"), dict)
            or seed_report["training_objective"].get(
                "candidate_lambda_hf"
            )
            is None
            or seed_report["training_objective"].get(
                "reference_lambda_hf"
            )
            is None
        ):
            raise ValueError("seed-comparison protocol mismatch")
        if (
            seed_report.get("primary_quality_superiority_seed_consistent")
            is not True
        ):
            raise ValueError("seed-consistent quality superiority failed")
        if (
            ood_spectra.get("schema_version") != 2
            or ood_spectra.get("selection_sha256") != selection_sha256
            or ood_spectra.get("candidate") != candidate
            or ood_spectra.get("reference") != "weatherbridge_ref"
            or ood_spectra.get("seeds") != expected_seeds
            or ood_spectra.get("candidate_seed_artifact_mode")
            != "fresh_postselection_replicates"
            or ood_spectra.get("checkpoint_linkage_verified") is not True
            or ood_spectra.get("spectral_protocol", {}).get(
                "confirmation_year"
            )
            != 2021
            or ood_spectra.get("spectral_protocol", {}).get(
                "used_for_model_selection"
            )
            is not False
        ):
            raise ValueError("OOD spectral-confirmation protocol mismatch")
        if (
            ood_spectra.get(
                "ood_spectral_superiority_seed_consistent"
            )
            is not True
        ):
            raise ValueError("OOD spectral superiority failed")
        result.update(
            {
                "seed_comparison_sha256": _sha256(seed_report_path),
                "ood_spectra_sha256": _sha256(ood_spectra_path),
                "seed_consistent_candidate": candidate,
            }
        )
    if ood_spectral_summary_path is not None:
        ood_summary = _load(ood_spectral_summary_path)
        ood_protocol = ood_summary.get("protocol", {})
        if (
            ood_summary.get("schema_version") != 1
            or ood_summary.get("selection_sha256") != selection_sha256
            or ood_summary.get("winner") != winner
            or set(ood_summary.get("models", ()))
            != set(selection_models)
            or ood_protocol.get("delta_t_hours") != delta_t_hours
            or ood_protocol.get("confirmation_year") != 2021
            or ood_protocol.get("used_for_model_selection") is not False
            or ood_protocol.get("spectral_taus")
            != EXPECTED_SPECTRAL_TAUS.get(delta_t_hours)
            or ood_protocol.get("hf_ell_min") != 180
            or ood_protocol.get("lmax") != 359
            or ood_protocol.get("channels") != 24
            or not _is_sha256(
                ood_summary.get("paired_window_index_sha256")
            )
            or not _is_sha256(
                ood_summary.get("spectral_grid_sha256")
            )
            or not _is_sha256(
                ood_summary.get("spectral_channel_names_sha256")
            )
            or not _is_sha256(
                ood_summary.get(
                    "evaluation_input_provenance_sha256"
                )
            )
            or not _is_sha256(
                ood_summary.get(
                    "evaluation_dataset_provenance_sha256"
                )
            )
        ):
            raise ValueError("OOD spectral-summary protocol mismatch")
        if (
            ood_summary.get("no_significant_ood_spectral_regression")
            is not True
            or ood_summary.get(
                "significant_ood_spectral_regressions"
            )
            != []
        ):
            raise ValueError("OOD spectral generalization regression")
        result["ood_spectral_summary_sha256"] = _sha256(
            ood_spectral_summary_path
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--region", type=Path, required=True)
    parser.add_argument("--delta-t-hours", type=int, required=True)
    parser.add_argument("--seed-report", type=Path)
    parser.add_argument("--ood-spectra", type=Path)
    parser.add_argument("--ood-spectral-summary", type=Path)
    args = parser.parse_args()
    result = check_gate(
        args.selection,
        args.forecast,
        args.exchange,
        args.region,
        delta_t_hours=args.delta_t_hours,
        seed_report_path=args.seed_report,
        ood_spectra_path=args.ood_spectra,
        ood_spectral_summary_path=args.ood_spectral_summary,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
