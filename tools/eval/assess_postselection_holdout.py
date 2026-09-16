#!/usr/bin/env python3
"""Assess the frozen 2022 holdout without reopening model selection."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REFERENCE = "weatherdcae_14m"
PHYSICAL_DIAGNOSTICS = (
    "wind_divergence_nmse",
    "wind_vorticity_nmse",
    "kinetic_energy_nmse",
    "hydrostatic_balance_mse",
    "q_negative_fraction",
    "global_mslp_bias",
    "lower_tropospheric_moisture_bias",
)
SPECTRAL_METRICS = (
    "energy_log_error",
    "shape_log_error",
    "coherence",
    "signed_cospectrum",
)
ALLOWED_WINNERS = {
    "weatherbridge_detail",
    "refine",
    "flow_spectral",
    REFERENCE,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite number")
    return result


def _validate_artifact(
    path: Path,
    *,
    horizon: int,
    manifest: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    protocol = payload.get("evaluation_protocol", {})
    expected_taus = list(range(1, horizon))
    if payload.get("schema_version") != 3:
        raise ValueError(f"{path}: unsupported artifact schema")
    if payload.get("years") != [2022] or float(payload.get("delta_t_hours", -1)) != horizon:
        raise ValueError(f"{path}: wrong year or horizon")
    if payload.get("channel_names") != manifest["fields"]:
        raise ValueError(f"{path}: wrong field order")
    required_protocol = {
        "rmse_reduction": "spherical_strip_area_weighted_spatial_mean",
        "samples_per_date": 1,
        "eval_days_per_month": None,
        "eval_days_of_month": [1, 8, 15, 22],
        "sample_strategy": "explicit_calendar_days",
        "full_year": False,
        "eval_hours": expected_taus,
        "save_window_metrics": True,
        "save_physical_metrics": True,
        "save_temporal_metrics": True,
        "latitude_grid": "wb2_0p25_2x2_block_average_v1",
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"{path}: protocol mismatch for {key}")
    expected_samples = 48 * len(expected_taus)
    if int(payload.get("num_samples", -1)) != expected_samples:
        raise ValueError(f"{path}: wrong window count")
    if payload.get("n_per_tau") != {str(tau): 48 for tau in expected_taus}:
        raise ValueError(f"{path}: incomplete per-hour sample counts")
    if not protocol.get("index_sha256"):
        raise ValueError(f"{path}: missing paired-window identity")
    dataset = payload.get("evaluation_dataset_provenance", {})
    year_provenance = dataset.get("files", {}).get("2022", {})
    if year_provenance.get("metadata_sha256") != verification["metadata_sha256"]:
        raise ValueError(f"{path}: holdout metadata hash mismatch")
    if int(year_provenance.get("size_bytes", -1)) != int(
        verification["logical_size_bytes"]
    ):
        raise ValueError(f"{path}: holdout byte size mismatch")
    if int(year_provenance.get("mtime_ns", -1)) != int(
        verification["data_mtime_ns"]
    ):
        raise ValueError(f"{path}: holdout changed after content verification")
    return payload


def _validate_pairwise(
    path: Path,
    *,
    winner: str,
    index_sha256: str,
    taus: list[int],
    expected_windows: int,
    left_metrics_path: Path,
    reference_metrics_path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("left") != winner
        or payload.get("index_sha256") != index_sha256
    ):
        raise ValueError(f"{path}: wrong pairwise identity")
    if payload.get("left_sha256") != _sha256(left_metrics_path):
        raise ValueError(f"{path}: stale candidate window metrics")
    if payload.get("right_sha256", {}).get(REFERENCE) != _sha256(
        reference_metrics_path
    ):
        raise ValueError(f"{path}: stale reference window metrics")
    comparison = payload.get("comparisons", {}).get(REFERENCE)
    if not isinstance(comparison, dict):
        raise ValueError(f"{path}: missing WeatherDCAE comparison")
    if comparison.get("taus") != taus:
        raise ValueError(f"{path}: wrong query-hour family")
    if int(comparison.get("block_days", -1)) != 7:
        raise ValueError(f"{path}: block length is not seven days")
    if int(comparison.get("bootstrap_draws", -1)) < 10_000:
        raise ValueError(f"{path}: fewer than 10,000 bootstrap draws")
    if int(comparison.get("n_windows", -1)) != expected_windows:
        raise ValueError(f"{path}: wrong paired-window count")
    ci = comparison.get("delta_ci95", [])
    if len(ci) != 3 or not all(math.isfinite(float(value)) for value in ci):
        raise ValueError(f"{path}: invalid primary confidence interval")
    return comparison


def _validate_cellwise_family(
    comparison: dict[str, Any],
    *,
    expected_hypotheses: int,
    context: str,
) -> dict[str, Any]:
    family = comparison.get("cellwise_family", {})
    if (
        family.get("correction") != "Holm-Bonferroni"
        or float(family.get("alpha", -1.0)) != 0.05
        or int(family.get("n_hypotheses", -1)) != expected_hypotheses
    ):
        raise ValueError(f"{context}: incomplete cellwise family")
    return family


def _validate_auxiliary_pairwise(
    path: Path,
    *,
    winner: str,
    metric: str,
    left_metrics_path: Path,
    reference_metrics_path: Path,
    expected_taus: list[int],
    expected_windows: int,
    expected_hypotheses: int | None,
    index_sha256: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 2
        or payload.get("left") != winner
        or payload.get("metric") != metric
    ):
        raise ValueError(f"{path}: wrong auxiliary comparison identity")
    if index_sha256 is not None and payload.get("index_sha256") != index_sha256:
        raise ValueError(f"{path}: auxiliary window index mismatch")
    if payload.get("left_sha256") != _sha256(left_metrics_path):
        raise ValueError(f"{path}: stale candidate window metrics")
    if payload.get("right_sha256", {}).get(REFERENCE) != _sha256(
        reference_metrics_path
    ):
        raise ValueError(f"{path}: stale reference window metrics")
    comparison = payload.get("comparisons", {}).get(REFERENCE)
    if not isinstance(comparison, dict):
        raise ValueError(f"{path}: missing WeatherDCAE comparison")
    if comparison.get("taus") != expected_taus:
        raise ValueError(f"{path}: wrong auxiliary query-hour family")
    if int(comparison.get("n_windows", -1)) != expected_windows:
        raise ValueError(f"{path}: wrong auxiliary window count")
    if int(comparison.get("block_days", -1)) != 7:
        raise ValueError(f"{path}: auxiliary block length is not seven days")
    if int(comparison.get("bootstrap_draws", comparison.get("draws", -1))) < 10_000:
        raise ValueError(f"{path}: fewer than 10,000 auxiliary draws")
    if expected_hypotheses is not None:
        _validate_cellwise_family(
            comparison,
            expected_hypotheses=expected_hypotheses,
            context=str(path),
        )
    return comparison


def _validate_hard_window_pairwise(
    path: Path,
    *,
    winner: str,
    index_sha256: str,
    left_metrics_path: Path,
    reference_metrics_path: Path,
    taus: list[int],
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("left") != winner
        or payload.get("metric") != "hard_window_rmse"
        or payload.get("selection_is_model_independent") is not True
        or float(payload.get("quantile", -1.0)) != 0.95
        or payload.get("index_sha256") != index_sha256
    ):
        raise ValueError(f"{path}: wrong hard-window comparison identity")
    if payload.get("left_sha256") != _sha256(left_metrics_path):
        raise ValueError(f"{path}: stale candidate hard-window input")
    if payload.get("right_sha256", {}).get(REFERENCE) != _sha256(
        reference_metrics_path
    ):
        raise ValueError(f"{path}: stale reference hard-window input")
    selected = payload.get("selected_windows_by_tau", {})
    if set(selected) != {str(tau) for tau in taus} or any(
        int(value) <= 0 for value in selected.values()
    ):
        raise ValueError(f"{path}: incomplete hard-window subset")
    comparison = payload.get("comparisons", {}).get(REFERENCE)
    if not isinstance(comparison, dict) or comparison.get("taus") != taus:
        raise ValueError(f"{path}: missing hard-window WeatherDCAE comparison")
    if int(comparison.get("n_windows", -1)) != sum(
        int(value) for value in selected.values()
    ):
        raise ValueError(f"{path}: hard-window count mismatch")
    _validate_cellwise_family(
        comparison,
        expected_hypotheses=24 * len(taus),
        context=str(path),
    )
    return comparison


def _validate_spectrum(
    path: Path,
    *,
    model_name: str,
    tau: int,
    horizon: int,
    fields: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata_json"].item()))
        if int(artifact["tau"]) != tau or int(artifact["n_samples"]) != 48:
            raise ValueError(f"{path}: wrong tau or spectral sample count")
        if [str(value) for value in artifact["channel_names"]] != fields:
            raise ValueError(f"{path}: wrong spectral field order")
        required = {
            "schema_version": 6,
            "year": 2022,
            "model_name": model_name,
            "model_kind": "capmatched",
            "sample_strategy": "explicit_calendar_days",
            "eval_days_per_month": None,
            "days_of_month": [1, 8, 15, 22],
            "samples_per_date": 1,
            "hf_ell_min": 80,
            "sht_grid": (
                "wb2_0p25_2x2_block_average_v1_latitude_strip_area_sht"
            ),
            "spectral_field_units": (
                "physical_anomaly_units_via_channel_std"
            ),
            "max_tau_hours": horizon,
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(f"{path}: spectral protocol mismatch for {key}")
        year = np.asarray(artifact["window_year"])
        t0 = np.asarray(artifact["window_t0"])
        if year.shape != (48,) or t0.shape != (48,) or not np.all(year == 2022):
            raise ValueError(f"{path}: invalid spectral window index")
        dates = [
            dt.date(2022, 1, 1) + dt.timedelta(hours=int(value))
            for value in t0
        ]
        if {date.day for date in dates} != {1, 8, 15, 22}:
            raise ValueError(f"{path}: spectrum used non-frozen calendar days")
        return year.copy(), t0.copy()


def _validate_spectral_pairwise(
    path: Path,
    *,
    winner: str,
    tau: int,
    diagnostic_space: str,
    expected_cells: int,
    left_spectrum_path: Path,
    reference_spectrum_path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 2
        or payload.get("left") != winner
        or payload.get("diagnostic_space") != diagnostic_space
        or len(payload.get("channels", [])) != expected_cells
    ):
        raise ValueError(f"{path}: wrong spectral comparison identity")
    if payload.get("left_sha256") != _sha256(left_spectrum_path):
        raise ValueError(f"{path}: stale candidate spectral input")
    if payload.get("right_sha256", {}).get(REFERENCE) != _sha256(
        reference_spectrum_path
    ):
        raise ValueError(f"{path}: stale reference spectral input")
    if Path(payload.get("left_path", "")).resolve() != left_spectrum_path.resolve():
        raise ValueError(f"{path}: wrong candidate spectral path")
    if Path(payload.get("right_paths", {}).get(REFERENCE, "")).resolve() != (
        reference_spectrum_path.resolve()
    ):
        raise ValueError(f"{path}: wrong reference spectral path")
    comparison = payload.get("comparisons", {}).get(REFERENCE)
    if not isinstance(comparison, dict):
        raise ValueError(f"{path}: missing spectral reference comparison")
    if (
        int(comparison.get("tau", -1)) != tau
        or int(comparison.get("n_windows", -1)) != 48
        or int(comparison.get("n_unique_days", -1)) != 48
        or int(comparison.get("block_days", -1)) != 7
        or int(comparison.get("draws", -1)) < 10_000
    ):
        raise ValueError(f"{path}: incomplete spectral resampling protocol")
    for metric_name in SPECTRAL_METRICS:
        metric = comparison.get(metric_name, {})
        ci = metric.get("delta_ci95", [])
        if len(ci) != 3 or not all(math.isfinite(float(value)) for value in ci):
            raise ValueError(f"{path}: invalid {metric_name} interval")
        family = comparison.get("cellwise_family", {}).get(metric_name, {})
        if int(family.get("n_hypotheses", -1)) != expected_cells:
            raise ValueError(f"{path}: incomplete {metric_name} cell family")
    return comparison


def _validate_spectral_summary(
    path: Path,
    *,
    taus: list[int],
    expected_channels: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 2
        or payload.get("reference") != REFERENCE
        or payload.get("years") != [2022]
        or payload.get("taus") != taus
    ):
        raise ValueError(f"{path}: wrong global spectral summary identity")
    input_hashes = payload.get("input_sha256", {})
    if len(input_hashes) != len(taus):
        raise ValueError(f"{path}: incomplete global spectral source binding")
    for source_name, expected_hash in input_hashes.items():
        source = Path(source_name)
        if not source.is_file() or _sha256(source) != expected_hash:
            raise ValueError(f"{path}: stale global spectral source {source}")
    yearly = payload.get("yearly", {}).get("2022", {})
    expected_hypotheses = expected_channels * len(taus)
    all_pointwise = True
    for metric_name in SPECTRAL_METRICS:
        metric = yearly.get(metric_name, {})
        family = metric.get("multiplicity_family", {})
        if (
            int(metric.get("cell_count", -1)) != expected_hypotheses
            or family.get("method") != "Holm-Bonferroni"
            or family.get("dimensions") != "tau_x_channel"
            or int(family.get("n_hypotheses", -1)) != expected_hypotheses
            or float(family.get("alpha", -1.0)) != 0.05
        ):
            raise ValueError(f"{path}: invalid global family for {metric_name}")
        all_pointwise &= bool(metric.get("all_pointwise_left_better", False))
    if bool(payload.get("strict_pointwise_spectral_dominance")) != all_pointwise:
        raise ValueError(f"{path}: inconsistent strict spectral summary")
    return payload


def assess(
    manifest_path: Path,
    champion_path: Path,
    verification_path: Path,
    evaluation_root: Path,
    *,
    candidate: str | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    champion = json.loads(champion_path.read_text())
    verification = json.loads(verification_path.read_text())
    if manifest.get("status") != "frozen_before_data_access" or manifest.get("year") != 2022:
        raise ValueError("post-selection manifest is not frozen")
    if not verification.get("verified"):
        raise ValueError("holdout memmap has not passed content verification")
    if verification.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("holdout verification is bound to another manifest")
    winner = champion.get("winner")
    if champion.get("status") not in {"confirmed", "reference_retained"}:
        raise ValueError("journal model selection is not final")
    if winner not in ALLOWED_WINNERS:
        raise ValueError(f"unsupported or absent frozen winner: {winner}")
    evaluated_candidate = candidate or winner
    if candidate is not None and evaluated_candidate not in ALLOWED_WINNERS - {REFERENCE}:
        raise ValueError(
            f"unsupported post-selection comparison candidate: {evaluated_candidate}"
        )
    declared_models = manifest.get("models")
    if declared_models is not None and evaluated_candidate not in declared_models:
        raise ValueError(
            f"comparison candidate was not declared in the frozen manifest: "
            f"{evaluated_candidate}"
        )
    selector_confirmation = evaluated_candidate == winner

    result: dict[str, Any] = {
        "schema_version": 1,
        "selection_was_frozen_before_holdout": True,
        "manifest_sha256": _sha256(manifest_path),
        "champion_sha256": _sha256(champion_path),
        "verification_sha256": _sha256(verification_path),
        "frozen_winner": winner,
        "evaluated_candidate": evaluated_candidate,
        "selector_confirmation": selector_confirmation,
        "reference": REFERENCE,
        "horizons": {},
    }
    if winner == REFERENCE and candidate is None:
        result.update(
            {
                "status": "reference_retained_holdout_unopened",
                "primary_endpoint_evaluated": False,
                "aggregate_confirmation_pass": None,
                "strict_field_hour_dominance": False,
                "strict_declared_diagnostic_dominance": False,
                "universal_dominance_claim_allowed": False,
                "publication_claim": (
                    "WeatherDCAE-14M remains the selected reference because no "
                    "learned alternative passed the retrospective selection gates. "
                    "The frozen 2022 candidate-versus-reference endpoint was not "
                    "evaluated."
                ),
            }
        )
        return result

    primary_passes = []
    strict_field_hour_passes = []
    strict_diagnostic_passes = []
    for horizon in (6, 12):
        horizon_dir = evaluation_root / f"{horizon}h"
        candidate = _validate_artifact(
            horizon_dir / f"{evaluated_candidate}.json",
            horizon=horizon,
            manifest=manifest,
            verification=verification,
        )
        reference = _validate_artifact(
            horizon_dir / f"{REFERENCE}.json",
            horizon=horizon,
            manifest=manifest,
            verification=verification,
        )
        candidate_index = candidate["evaluation_protocol"]["index_sha256"]
        if reference["evaluation_protocol"]["index_sha256"] != candidate_index:
            raise ValueError(f"{horizon}h candidate/reference windows are not paired")
        all_taus = list(range(1, horizon))
        held_taus = [int(value) for value in manifest["held_query_hours"][str(horizon)]]
        candidate_windows = (
            horizon_dir / "window_metrics" / f"{evaluated_candidate}.npz"
        )
        reference_windows = (
            horizon_dir / "window_metrics" / f"{REFERENCE}.npz"
        )
        primary = _validate_pairwise(
            horizon_dir / f"paired_rmse_{evaluated_candidate}.json",
            winner=evaluated_candidate,
            index_sha256=candidate_index,
            taus=all_taus,
            expected_windows=48 * len(all_taus),
            left_metrics_path=candidate_windows,
            reference_metrics_path=reference_windows,
        )
        held = _validate_pairwise(
            horizon_dir / f"paired_rmse_held_{evaluated_candidate}.json",
            winner=evaluated_candidate,
            index_sha256=candidate_index,
            taus=held_taus,
            expected_windows=48 * len(held_taus),
            left_metrics_path=candidate_windows,
            reference_metrics_path=reference_windows,
        )
        expected_hypotheses = 24 * len(all_taus)
        family = _validate_cellwise_family(
            primary,
            expected_hypotheses=expected_hypotheses,
            context=f"{horizon}h RMSE",
        )
        _validate_cellwise_family(
            held,
            expected_hypotheses=24 * len(held_taus),
            context=f"{horizon}h held RMSE",
        )
        hard = _validate_hard_window_pairwise(
            horizon_dir / f"paired_hard_window_{evaluated_candidate}.json",
            winner=evaluated_candidate,
            index_sha256=candidate_index,
            left_metrics_path=candidate_windows,
            reference_metrics_path=reference_windows,
            taus=all_taus,
        )
        acc = _validate_auxiliary_pairwise(
            horizon_dir / f"paired_acc_{evaluated_candidate}.json",
            winner=evaluated_candidate,
            metric="acc",
            left_metrics_path=candidate_windows,
            reference_metrics_path=reference_windows,
            expected_taus=all_taus,
            expected_windows=48 * len(all_taus),
            expected_hypotheses=expected_hypotheses,
            index_sha256=candidate_index,
        )
        temporal = _validate_auxiliary_pairwise(
            horizon_dir / f"paired_temporal_curvature_{evaluated_candidate}.json",
            winner=evaluated_candidate,
            metric="temporal_curvature",
            left_metrics_path=candidate_windows,
            reference_metrics_path=reference_windows,
            expected_taus=[0],
            expected_windows=48,
            expected_hypotheses=24,
        )
        physical = _validate_auxiliary_pairwise(
            horizon_dir / f"paired_physical_{evaluated_candidate}.json",
            winner=evaluated_candidate,
            metric="physical",
            left_metrics_path=candidate_windows,
            reference_metrics_path=reference_windows,
            expected_taus=all_taus,
            expected_windows=48 * len(all_taus),
            expected_hypotheses=None,
            index_sha256=candidate_index,
        )
        physical_rows = physical.get("diagnostics", {})
        if set(physical_rows) != set(PHYSICAL_DIAGNOSTICS):
            raise ValueError(f"{horizon}h: incomplete physical diagnostic family")
        if int(family.get("n_hypotheses", -1)) != expected_hypotheses:
            raise ValueError(f"{horizon}h field-hour multiplicity family is incomplete")
        primary_pass = float(primary["delta_ci95"][2]) < 0.0
        strict_field_hour = bool(family.get("all_pointwise_left_better", False))
        primary_passes.append(primary_pass)
        strict_field_hour_passes.append(strict_field_hour)
        spectral: dict[str, dict[str, Any]] = {"scalar": {}, "vector": {}}
        spectra_dir = horizon_dir / "spectra"
        for tau in all_taus:
            candidate_year, candidate_t0 = _validate_spectrum(
                spectra_dir / f"{evaluated_candidate}_tau{tau}.npz",
                model_name=evaluated_candidate,
                tau=tau,
                horizon=horizon,
                fields=manifest["fields"],
            )
            reference_year, reference_t0 = _validate_spectrum(
                spectra_dir / f"{REFERENCE}_tau{tau}.npz",
                model_name=REFERENCE,
                tau=tau,
                horizon=horizon,
                fields=manifest["fields"],
            )
            if not np.array_equal(candidate_year, reference_year) or not np.array_equal(
                candidate_t0,
                reference_t0,
            ):
                raise ValueError(f"{horizon}h tau={tau}: spectral windows are not paired")
            for label, space, cells in (
                ("scalar", "scalar_sht", 24),
                ("vector", "vector_sht", 10),
            ):
                candidate_spectrum = (
                    spectra_dir / f"{evaluated_candidate}_tau{tau}.npz"
                )
                reference_spectrum = spectra_dir / f"{REFERENCE}_tau{tau}.npz"
                comparison = _validate_spectral_pairwise(
                    spectra_dir
                    / f"paired_{label}_{evaluated_candidate}_tau{tau}.json",
                    winner=evaluated_candidate,
                    tau=tau,
                    diagnostic_space=space,
                    expected_cells=cells,
                    left_spectrum_path=candidate_spectrum,
                    reference_spectrum_path=reference_spectrum,
                )
                spectral[label][str(tau)] = {
                    metric_name: {
                        "delta_left_minus_right": float(
                            comparison[metric_name]["delta_left_minus_right"]
                        ),
                        "delta_ci95": [
                            float(value)
                            for value in comparison[metric_name]["delta_ci95"]
                        ],
                        "pointwise_wins": int(
                            comparison["cellwise_family"][metric_name][
                                "n_pointwise_left_better"
                            ]
                        ),
                        "n_cells": cells,
                    }
                    for metric_name in SPECTRAL_METRICS
                }
        scalar_summary = _validate_spectral_summary(
            spectra_dir
            / f"global_scalar_{evaluated_candidate}_vs_{REFERENCE}.json",
            taus=all_taus,
            expected_channels=24,
        )
        vector_summary = _validate_spectral_summary(
            spectra_dir
            / f"global_vector_{evaluated_candidate}_vs_{REFERENCE}.json",
            taus=all_taus,
            expected_channels=10,
        )
        strict_diagnostic = all(
            (
                strict_field_hour,
                bool(hard["cellwise_family"].get("all_pointwise_left_better")),
                bool(acc["cellwise_family"].get("all_pointwise_left_better")),
                bool(temporal["cellwise_family"].get("all_pointwise_left_better")),
                all(
                    float(physical_rows[name]["delta_left_minus_right"]) < 0.0
                    for name in PHYSICAL_DIAGNOSTICS
                ),
                bool(scalar_summary["strict_pointwise_spectral_dominance"]),
                bool(vector_summary["strict_pointwise_spectral_dominance"]),
            )
        )
        strict_diagnostic_passes.append(strict_diagnostic)
        result["horizons"][f"{horizon}h"] = {
            "delta_normalized_rmse": _finite(primary["delta_left_minus_right"], "RMSE delta"),
            "relative_delta_pct": _finite(primary["relative_delta_pct"], "relative RMSE delta"),
            "delta_ci95": [float(value) for value in primary["delta_ci95"]],
            "p_paired_block_permutation": _finite(primary["p_paired_block_permutation"], "paired p-value"),
            "primary_pass": primary_pass,
            "held_relative_delta_pct": _finite(held["relative_delta_pct"], "held RMSE delta"),
            "held_delta_ci95": [float(value) for value in held["delta_ci95"]],
            "field_hour_cells": expected_hypotheses,
            "pointwise_wins": int(family.get("n_pointwise_left_better", 0)),
            "holm_significant_improvements": int(family.get("n_significant_left_better_holm", 0)),
            "holm_significant_regressions": int(family.get("n_significant_left_worse_holm", 0)),
            "all_field_hours_pointwise_better": strict_field_hour,
            "all_declared_diagnostics_pointwise_better": strict_diagnostic,
            "spectral": spectral,
        }

    aggregate_pass = all(primary_passes)
    strict_field_hour = aggregate_pass and all(strict_field_hour_passes)
    strict_diagnostic = aggregate_pass and all(strict_diagnostic_passes)
    result.update(
        {
            "status": "confirmed_aggregate" if aggregate_pass else "postselection_failed",
            "primary_endpoint_evaluated": True,
            "aggregate_confirmation_pass": aggregate_pass,
            "strict_field_hour_dominance": strict_field_hour,
            "strict_declared_diagnostic_dominance": strict_diagnostic,
            "universal_dominance_claim_allowed": strict_diagnostic,
            "publication_claim": (
                (
                    "The frozen winner improves aggregate normalized RMSE at "
                    "both horizons on the untouched 2022 holdout."
                    if selector_confirmation
                    else "The predeclared WeatherBridge checkpoint improves "
                    "aggregate normalized RMSE at both horizons on the frozen "
                    "2022 sample; the earlier multi-metric selector still "
                    "retains WeatherDCAE-14M."
                )
                if aggregate_pass
                else (
                    "The evaluated candidate did not confirm aggregate "
                    "improvement at both horizons; the manuscript must report "
                    "this failure."
                )
            ),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=sorted(ALLOWED_WINNERS - {REFERENCE}),
        help=(
            "Evaluate a model predeclared in the frozen manifest even when the "
            "retrospective selector retained the reference. This does not "
            "reopen model selection."
        ),
    )
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    result = assess(
        args.manifest,
        args.champion,
        args.verification,
        args.evaluation_root,
        candidate=args.candidate,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "frozen_winner", "aggregate_confirmation_pass", "universal_dominance_claim_allowed")}))


if __name__ == "__main__":
    main()
