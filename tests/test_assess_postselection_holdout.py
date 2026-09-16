from __future__ import annotations

import hashlib
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from tools.eval.assess_postselection_holdout import _validate_spectrum, assess
from tools.eval.summarize_spectral_dominance import summarize


FIELDS = [
    f"{prefix}{level}"
    for prefix in ("T", "U", "V", "Q", "Z")
    for level in (1000, 925, 850, 700)
] + ["t2m", "u10", "v10", "mslp"]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _fixture(tmp_path: Path, upper: float = -0.001) -> tuple[Path, Path, Path, Path]:
    manifest = tmp_path / "manifest.json"
    _write(
        manifest,
        {
            "status": "frozen_before_data_access",
            "year": 2022,
            "fields": FIELDS,
            "held_query_hours": {"6": [2, 4], "12": [4, 6, 8]},
        },
    )
    champion = tmp_path / "champion.json"
    _write(champion, {"status": "confirmed", "winner": "weatherbridge_detail"})
    metadata_hash = "a" * 64
    verification = tmp_path / "verification.json"
    _write(
        verification,
        {
            "verified": True,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "metadata_sha256": metadata_hash,
            "logical_size_bytes": 123456,
            "data_mtime_ns": 987654321,
        },
    )
    root = tmp_path / "evaluation"
    for horizon in (6, 12):
        taus = list(range(1, horizon))
        index_hash = str(horizon) * 64
        artifact = {
            "schema_version": 3,
            "years": [2022],
            "delta_t_hours": float(horizon),
            "channel_names": FIELDS,
            "num_samples": 48 * len(taus),
            "n_per_tau": {str(tau): 48 for tau in taus},
            "evaluation_protocol": {
                "rmse_reduction": "spherical_strip_area_weighted_spatial_mean",
                "samples_per_date": 1,
                "eval_days_per_month": None,
                "eval_days_of_month": [1, 8, 15, 22],
                "sample_strategy": "explicit_calendar_days",
                "full_year": False,
                "eval_hours": taus,
                "save_window_metrics": True,
                "save_physical_metrics": True,
                "save_temporal_metrics": True,
                "latitude_grid": "wb2_0p25_2x2_block_average_v1",
                "index_sha256": index_hash,
            },
            "evaluation_dataset_provenance": {
                "files": {
                    "2022": {
                        "metadata_sha256": metadata_hash,
                        "size_bytes": 123456,
                        "mtime_ns": 987654321,
                    }
                }
            },
        }
        for model in ("weatherbridge_detail", "weatherdcae_14m"):
            _write(root / f"{horizon}h" / f"{model}.json", artifact)
        window_dir = root / f"{horizon}h" / "window_metrics"
        window_dir.mkdir(parents=True, exist_ok=True)
        candidate_windows = window_dir / "weatherbridge_detail.npz"
        reference_windows = window_dir / "weatherdcae_14m.npz"
        np.savez(candidate_windows, token=np.asarray("candidate"))
        np.savez(reference_windows, token=np.asarray("reference"))
        candidate_window_hash = hashlib.sha256(
            candidate_windows.read_bytes()
        ).hexdigest()
        reference_window_hash = hashlib.sha256(
            reference_windows.read_bytes()
        ).hexdigest()

        def strict_family(count: int) -> dict:
            return {
                "correction": "Holm-Bonferroni",
                "alpha": 0.05,
                "n_hypotheses": count,
                "n_pointwise_left_better": count,
                "n_significant_left_better_holm": 4,
                "n_significant_left_worse_holm": 0,
                "all_pointwise_left_better": True,
            }

        for suffix, selected_taus in (
            ("", taus),
            ("_held", [2, 4] if horizon == 6 else [4, 6, 8]),
        ):
            family = strict_family(24 * len(selected_taus))
            comparison = {
                "taus": selected_taus,
                "block_days": 7,
                "bootstrap_draws": 10_000,
                "n_windows": 48 * len(selected_taus),
                "delta_left_minus_right": -0.002,
                "relative_delta_pct": -1.0,
                "delta_ci95": [-0.003, -0.002, upper],
                "p_paired_block_permutation": 0.01,
                "cellwise_family": family,
            }
            _write(
                root / f"{horizon}h" / f"paired_rmse{suffix}_weatherbridge_detail.json",
                {
                    "schema_version": 1,
                    "left": "weatherbridge_detail",
                    "left_sha256": candidate_window_hash,
                    "right_sha256": {
                        "weatherdcae_14m": reference_window_hash
                    },
                    "index_sha256": index_hash,
                    "comparisons": {"weatherdcae_14m": comparison},
                },
            )
        hard_selected = {str(tau): 3 for tau in taus}
        hard_comparison = {
            "taus": taus,
            "block_days": 7,
            "bootstrap_draws": 10_000,
            "n_windows": sum(hard_selected.values()),
            "delta_left_minus_right": -0.002,
            "relative_delta_pct": -1.0,
            "delta_ci95": [-0.003, -0.002, -0.001],
            "p_paired_block_permutation": 0.01,
            "cellwise_family": strict_family(24 * len(taus)),
        }
        _write(
            root / f"{horizon}h" / "paired_hard_window_weatherbridge_detail.json",
            {
                "schema_version": 1,
                "left": "weatherbridge_detail",
                "left_sha256": candidate_window_hash,
                "right_sha256": {"weatherdcae_14m": reference_window_hash},
                "metric": "hard_window_rmse",
                "selection_is_model_independent": True,
                "quantile": 0.95,
                "index_sha256": index_hash,
                "selected_windows_by_tau": hard_selected,
                "comparisons": {"weatherdcae_14m": hard_comparison},
            },
        )
        for metric, metric_taus, windows, family_count in (
            ("acc", taus, 48 * len(taus), 24 * len(taus)),
            ("temporal_curvature", [0], 48, 24),
        ):
            comparison = {
                "metric": "mean_window_acc" if metric == "acc" else metric,
                "better": "higher" if metric == "acc" else "lower",
                "taus": metric_taus,
                "block_days": 7,
                "bootstrap_draws": 10_000,
                "n_windows": windows,
                "delta_left_minus_right": 0.01 if metric == "acc" else -0.01,
                "delta_ci95": (
                    [0.001, 0.01, 0.02]
                    if metric == "acc"
                    else [-0.02, -0.01, -0.001]
                ),
                "p_paired_block_permutation": 0.01,
                "cellwise_family": strict_family(family_count),
            }
            _write(
                root / f"{horizon}h" / f"paired_{metric}_weatherbridge_detail.json",
                {
                    "schema_version": 2,
                    "metric": metric,
                    "left": "weatherbridge_detail",
                    "left_sha256": candidate_window_hash,
                    "right_sha256": {"weatherdcae_14m": reference_window_hash},
                    "index_sha256": index_hash,
                    "comparisons": {"weatherdcae_14m": comparison},
                },
            )
        physical_rows = {
            name: {
                "better": "lower",
                "delta_left_minus_right": -0.01,
                "delta_ci95": [-0.02, -0.01, -0.001],
                "p_paired_block_permutation": 0.01,
            }
            for name in (
                "wind_divergence_nmse",
                "wind_vorticity_nmse",
                "kinetic_energy_nmse",
                "hydrostatic_balance_mse",
                "q_negative_fraction",
                "global_mslp_bias",
                "lower_tropospheric_moisture_bias",
            )
        }
        _write(
            root / f"{horizon}h" / "paired_physical_weatherbridge_detail.json",
            {
                "schema_version": 2,
                "metric": "physical",
                "left": "weatherbridge_detail",
                "left_sha256": candidate_window_hash,
                "right_sha256": {"weatherdcae_14m": reference_window_hash},
                "index_sha256": index_hash,
                "comparisons": {
                    "weatherdcae_14m": {
                        "taus": taus,
                        "n_windows": 48 * len(taus),
                        "n_blocks": 48,
                        "block_days": 7,
                        "draws": 10_000,
                        "diagnostics": physical_rows,
                    }
                },
            },
        )
        dates = [
            dt.date(2022, month, day)
            for month in range(1, 13)
            for day in (1, 8, 15, 22)
        ]
        t0 = np.asarray(
            [(date - dt.date(2022, 1, 1)).days * 24 for date in dates],
            dtype=np.int32,
        )
        spectra = root / f"{horizon}h" / "spectra"
        spectra.mkdir(parents=True, exist_ok=True)
        for tau in taus:
            for model in ("weatherbridge_detail", "weatherdcae_14m"):
                metadata = {
                    "schema_version": 6,
                    "year": 2022,
                    "model_name": model,
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
                np.savez(
                    spectra / f"{model}_tau{tau}.npz",
                    metadata_json=np.asarray(json.dumps(metadata)),
                    tau=np.asarray(tau),
                    n_samples=np.asarray(48),
                    channel_names=np.asarray(FIELDS),
                    window_year=np.full(48, 2022, dtype=np.int16),
                    window_t0=t0,
                )
            for label, space, cells in (
                ("scalar", "scalar_sht", 24),
                ("vector", "vector_sht", 10),
            ):
                metric = {
                    "better": "lower",
                    "delta_left_minus_right": -0.01,
                    "delta_ci95": [-0.02, -0.01, -0.001],
                    "p_paired_block_permutation": 0.01,
                }
                family = {
                    "n_hypotheses": cells,
                    "n_pointwise_left_better": cells,
                }
                comparison = {
                    "tau": tau,
                    "n_windows": 48,
                    "n_unique_days": 48,
                    "block_days": 7,
                    "draws": 10_000,
                    **{
                        name: metric
                        for name in (
                            "energy_log_error",
                            "shape_log_error",
                            "coherence",
                            "signed_cospectrum",
                        )
                    },
                    "cellwise_family": {
                        name: family
                        for name in (
                            "energy_log_error",
                            "shape_log_error",
                            "coherence",
                            "signed_cospectrum",
                        )
                    },
                    "per_channel": {
                        f"cell-{index}": {
                            name: {
                                "better": (
                                    "higher"
                                    if name in ("coherence", "signed_cospectrum")
                                    else "lower"
                                ),
                                "delta_left_minus_right": (
                                    0.01
                                    if name in ("coherence", "signed_cospectrum")
                                    else -0.01
                                ),
                                "p_paired_block_permutation": 0.01,
                                "p_holm": 0.02,
                            }
                            for name in (
                                "energy_log_error",
                                "shape_log_error",
                                "coherence",
                                "signed_cospectrum",
                            )
                        }
                        for index in range(cells)
                    },
                }
                candidate_spectrum = spectra / f"weatherbridge_detail_tau{tau}.npz"
                reference_spectrum = spectra / f"weatherdcae_14m_tau{tau}.npz"
                _write(
                    spectra
                    / f"paired_{label}_weatherbridge_detail_tau{tau}.json",
                    {
                        "schema_version": 2,
                        "left": "weatherbridge_detail",
                        "left_path": str(candidate_spectrum),
                        "left_sha256": hashlib.sha256(
                            candidate_spectrum.read_bytes()
                        ).hexdigest(),
                        "right_paths": {
                            "weatherdcae_14m": str(reference_spectrum)
                        },
                        "right_sha256": {
                            "weatherdcae_14m": hashlib.sha256(
                                reference_spectrum.read_bytes()
                            ).hexdigest()
                        },
                        "channels": [f"cell-{index}" for index in range(cells)],
                        "diagnostic_space": space,
                        "comparisons": {"weatherdcae_14m": comparison},
                    },
                )
        for label in ("scalar", "vector"):
            summary = summarize(
                spectra,
                years=[2022],
                taus=taus,
                pattern=(
                    f"paired_{label}_weatherbridge_detail_tau{{tau}}.json"
                ),
                reference="weatherdcae_14m",
            )
            _write(
                spectra
                / f"global_{label}_weatherbridge_detail_vs_weatherdcae_14m.json",
                summary,
            )
    return manifest, champion, verification, root


def test_assessment_confirms_only_the_frozen_winner(tmp_path: Path) -> None:
    result = assess(*_fixture(tmp_path))
    assert result["frozen_winner"] == "weatherbridge_detail"
    assert result["status"] == "confirmed_aggregate"
    assert result["universal_dominance_claim_allowed"] is True


def test_failed_horizon_blocks_postselection_claim(tmp_path: Path) -> None:
    result = assess(*_fixture(tmp_path, upper=0.001))
    assert result["status"] == "postselection_failed"
    assert result["aggregate_confirmation_pass"] is False
    assert result["universal_dominance_claim_allowed"] is False


def test_one_diagnostic_regression_blocks_universal_claim(tmp_path: Path) -> None:
    manifest, champion, verification, root = _fixture(tmp_path)
    path = root / "6h" / "paired_physical_weatherbridge_detail.json"
    payload = json.loads(path.read_text())
    payload["comparisons"]["weatherdcae_14m"]["diagnostics"][
        "global_mslp_bias"
    ]["delta_left_minus_right"] = 0.001
    _write(path, payload)

    result = assess(manifest, champion, verification, root)

    assert result["aggregate_confirmation_pass"] is True
    assert result["strict_field_hour_dominance"] is True
    assert result["strict_declared_diagnostic_dominance"] is False
    assert result["universal_dominance_claim_allowed"] is False


def test_reference_retention_never_becomes_candidate_dominance(tmp_path: Path) -> None:
    manifest, champion, verification, root = _fixture(tmp_path)
    _write(champion, {"status": "reference_retained", "winner": "weatherdcae_14m"})
    result = assess(manifest, champion, verification, root)
    assert result["status"] == "reference_retained_holdout_unopened"
    assert result["primary_endpoint_evaluated"] is False
    assert result["aggregate_confirmation_pass"] is None
    assert result["universal_dominance_claim_allowed"] is False


def test_predeclared_candidate_can_be_reported_without_reopening_selector(
    tmp_path: Path,
) -> None:
    manifest, champion, verification, root = _fixture(tmp_path)
    _write(champion, {"status": "reference_retained", "winner": "weatherdcae_14m"})

    result = assess(
        manifest,
        champion,
        verification,
        root,
        candidate="weatherbridge_detail",
    )

    assert result["frozen_winner"] == "weatherdcae_14m"
    assert result["evaluated_candidate"] == "weatherbridge_detail"
    assert result["selector_confirmation"] is False
    assert result["status"] == "confirmed_aggregate"
    assert result["aggregate_confirmation_pass"] is True


def test_assessment_rejects_data_changed_after_verification(tmp_path: Path) -> None:
    manifest, champion, verification, root = _fixture(tmp_path)
    artifact_path = root / "6h" / "weatherbridge_detail.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["evaluation_dataset_provenance"]["files"]["2022"][
        "mtime_ns"
    ] += 1
    _write(artifact_path, artifact)

    with pytest.raises(ValueError, match="changed after content verification"):
        assess(manifest, champion, verification, root)


def test_assessment_rejects_standardized_spectral_units(tmp_path: Path) -> None:
    manifest, champion, verification, root = _fixture(tmp_path)
    path = root / "6h" / "spectra" / "weatherbridge_detail_tau1.npz"
    with np.load(path, allow_pickle=False) as artifact:
        values = {key: np.asarray(artifact[key]) for key in artifact.files}
    metadata = json.loads(str(values["metadata_json"].item()))
    metadata["spectral_field_units"] = "standardized_channel_units"
    values["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(path, **values)

    with pytest.raises(ValueError, match="spectral_field_units"):
        _validate_spectrum(
            path,
            model_name="weatherbridge_detail",
            tau=1,
            horizon=6,
            fields=FIELDS,
        )
