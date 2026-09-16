import json

import numpy as np
import pytest

import tools.eval.validate_upr_lite_selection as validation
from tools.eval.paired_block_bootstrap import WindowMetrics
from tools.eval.validate_upr_lite_selection import (
    _annotate_holm,
    _canonical_sha256,
    _file_sha256,
    _window_months,
    extreme_field_comparisons,
    summarize,
    verify_selection_artifacts,
)
from weather_time_interp.metrics.physical_consistency import (
    GENERALIZATION_DIAGNOSTICS,
)


def _field(delta: float, p: float) -> dict:
    return {
        "delta_left_minus_right": delta,
        "relative_delta_pct": delta * 100.0,
        "p_paired_block_permutation": p,
    }


def _physical(delta: float, p: float) -> dict:
    return {
        "diagnostics": {
            name: _field(delta, p)
            for name in GENERALIZATION_DIAGNOSTICS
        }
    }


def test_summary_confirms_robust_winner() -> None:
    groups = {
        "seen": {
            "challenger": _field(-0.1, 0.01),
            "bilinear": _field(-0.2, 0.01),
        },
        "unseen": {
            "challenger": _field(-0.1, 0.01),
            "bilinear": _field(-0.2, 0.01),
        },
    }
    field = {"2020": groups, "2021": groups}
    spectral = {
        "2": {
            "challenger": {
                "energy_log_error": _field(-0.1, 0.01),
                "shape_log_error": _field(-0.1, 0.01),
                "coherence": _field(0.1, 0.01),
            }
        }
    }
    physical = {
        "2020": {
            "seen": {
                "challenger": _physical(-0.1, 0.01),
                "bilinear": _physical(-0.1, 0.01),
            },
            "unseen": {
                "challenger": _physical(-0.1, 0.01),
                "bilinear": _physical(-0.1, 0.01),
            },
        },
        "2021": {
            "seen": {
                "challenger": _physical(-0.1, 0.01),
                "bilinear": _physical(-0.1, 0.01),
            },
            "unseen": {
                "challenger": _physical(-0.1, 0.01),
                "bilinear": _physical(-0.1, 0.01),
            },
        },
    }
    acc = {
        "2020": groups,
        "2021": {
            "seen": groups["seen"],
            "unseen": {
                "challenger": _field(-0.1, 0.01),
                "bilinear": _field(0.2, 0.01),
            },
        },
    }

    result = summarize(field, spectral, physical, acc)

    assert result["selection_confirmed"]
    assert not result["cross_metric_generalization_confirmed"]
    assert not result["significant_field_challenger_wins"]
    assert not result["significant_spectral_dominance"]
    assert not result["significant_spectral_regressions"]
    assert not result["significant_physical_dominance"]
    assert not result["significant_physical_regressions"]
    assert result["significant_acc_challenger_wins"] == [
        {
            "year": "2021",
            "tau_group": "unseen",
            "model": "challenger",
            "relative_delta_pct": -10.0,
            "p_raw": 0.01,
            "p_holm": 0.01,
        }
    ]

    acc["2021"]["unseen"]["challenger"] = _field(0.1, 0.01)
    robust_skill = {
        "tail_skill_2021_unseen": 0.1,
        "worst_season_skill_2021_unseen": 0.2,
    }
    result_without_regression = summarize(
        field,
        spectral,
        physical,
        acc,
        ood_robust_skill=robust_skill,
    )
    assert result_without_regression["selection_confirmed"]
    assert result_without_regression["cross_metric_generalization_confirmed"]
    assert result_without_regression["absolute_ood_robust_skill_passed"]

    robust_skill["tail_skill_2021_unseen"] = -0.01
    tail_regression = summarize(
        field,
        spectral,
        physical,
        acc,
        ood_robust_skill=robust_skill,
    )
    assert tail_regression["selection_confirmed"]
    assert not tail_regression["cross_metric_generalization_confirmed"]
    assert not tail_regression["absolute_ood_robust_skill_passed"]


def test_single_physical_regression_blocks_cross_metric_claim() -> None:
    groups = {
        "seen": {
            "challenger": _field(-0.1, 0.01),
            "bilinear": _field(-0.2, 0.01),
        },
        "unseen": {
            "challenger": _field(-0.1, 0.01),
            "bilinear": _field(-0.2, 0.01),
        },
    }
    field = {"2020": groups, "2021": groups}
    spectral = {
        "2": {
            "challenger": {
                "energy_log_error": _field(-0.1, 0.01),
                "shape_log_error": _field(-0.1, 0.01),
                "coherence": _field(0.1, 0.01),
            }
        }
    }
    physical = {
        year: {
            group: {
                "challenger": _physical(-0.1, 0.01),
                "bilinear": _physical(-0.1, 0.01),
            }
            for group in ("seen", "unseen")
        }
        for year in ("2020", "2021")
    }
    physical["2021"]["unseen"]["challenger"]["diagnostics"][
        "global_mslp_bias"
    ] = _field(0.1, 0.001)

    result = summarize(field, spectral, physical)

    assert result["selection_confirmed"]
    assert not result["cross_metric_generalization_confirmed"]
    assert [
        item["diagnostic"]
        for item in result["significant_physical_regressions"]
    ] == ["global_mslp_bias"]


def test_single_ood_hour_regression_blocks_cross_metric_claim() -> None:
    aggregate_groups = {
        "seen": {
            "challenger": _field(-0.1, 0.01),
            "bilinear": _field(-0.2, 0.01),
        },
        "unseen": {
            "challenger": _field(-0.1, 0.01),
            "bilinear": _field(-0.2, 0.01),
        },
    }
    field = {
        "2020": dict(aggregate_groups),
        "2021": {
            **aggregate_groups,
            "h1": {
                "challenger": _field(0.1, 0.001),
                "bilinear": _field(-0.2, 0.01),
            },
            "h2": {
                "challenger": _field(-0.1, 0.01),
                "bilinear": _field(-0.2, 0.01),
            },
        },
    }
    spectral = {
        "2": {
            "challenger": {
                "energy_log_error": _field(-0.1, 0.01),
                "shape_log_error": _field(-0.1, 0.01),
                "coherence": _field(0.1, 0.01),
            }
        }
    }
    physical = {
        year: {
            group: {
                "challenger": _physical(-0.1, 0.01),
                "bilinear": _physical(-0.1, 0.01),
            }
            for group in ("seen", "unseen")
        }
        for year in ("2020", "2021")
    }

    result = summarize(field, spectral, physical)

    assert result["selection_confirmed"]
    assert not result["cross_metric_generalization_confirmed"]
    assert result["significant_individual_hour_field_regressions"] == [
        {
            "year": "2021",
            "tau_group": "h1",
            "tau": 1,
            "model": "challenger",
            "relative_delta_pct": 10.0,
            "p_raw": 0.001,
            "p_holm": 0.002,
        }
    ]


def test_single_ood_seasonal_regression_blocks_cross_metric_claim() -> None:
    groups = {
        group: {
            "challenger": _field(-0.1, 0.01),
            "bilinear": _field(-0.2, 0.01),
        }
        for group in ("seen", "unseen")
    }
    field = {"2020": groups, "2021": groups}
    spectral = {
        "2": {
            "challenger": {
                "energy_log_error": _field(-0.1, 0.01),
                "shape_log_error": _field(-0.1, 0.01),
                "coherence": _field(0.1, 0.01),
            }
        }
    }
    physical = {
        year: {
            group: {
                "challenger": _physical(-0.1, 0.01),
                "bilinear": _physical(-0.1, 0.01),
            }
            for group in ("seen", "unseen")
        }
        for year in ("2020", "2021")
    }
    seasonal_2021 = {
        season: {
            "challenger": _field(
                0.1 if season == "JJA" else -0.1,
                0.001 if season == "JJA" else 1.0,
            ),
            "bilinear": _field(-0.2, 1.0),
        }
        for season in ("DJF", "MAM", "JJA", "SON")
    }

    result = summarize(
        field,
        spectral,
        physical,
        seasonal={"2020": seasonal_2021, "2021": seasonal_2021},
    )

    assert result["selection_confirmed"]
    assert not result["cross_metric_generalization_confirmed"]
    assert [
        (item["season"], item["model"])
        for item in result["significant_seasonal_regressions"]
    ] == [("JJA", "challenger")]

    stable_seasonal = {
        season: {
            "challenger": _field(-0.1, 1.0),
            "bilinear": _field(-0.2, 1.0),
        }
        for season in ("DJF", "MAM", "JJA", "SON")
    }
    extreme = {
        year: {
            "comparisons": {
                "challenger": _field(0.1, 0.001),
                "bilinear": _field(-0.2, 1.0),
            }
        }
        for year in ("2020", "2021")
    }
    extreme_result = summarize(
        field,
        spectral,
        physical,
        seasonal={"2020": stable_seasonal, "2021": stable_seasonal},
        extreme=extreme,
    )

    assert extreme_result["selection_confirmed"]
    assert not extreme_result["cross_metric_generalization_confirmed"]
    assert [
        item["model"]
        for item in extreme_result["significant_extreme_regressions"]
    ] == ["challenger"]


def test_temporal_curvature_regression_blocks_cross_metric_claim() -> None:
    groups = {
        group: {
            "challenger": _field(-0.1, 0.01),
            "bilinear": _field(-0.2, 0.01),
        }
        for group in ("seen", "unseen")
    }
    field = {"2020": groups, "2021": groups}
    spectral = {
        "2": {
            "challenger": {
                "energy_log_error": _field(-0.1, 0.01),
                "shape_log_error": _field(-0.1, 0.01),
                "coherence": _field(0.1, 0.01),
            }
        }
    }
    physical = {
        year: {
            group: {
                "challenger": _physical(-0.1, 0.01),
                "bilinear": _physical(-0.1, 0.01),
            }
            for group in ("seen", "unseen")
        }
        for year in ("2020", "2021")
    }
    temporal = {
        year: {
            "challenger": _field(0.1, 0.001),
            "bilinear": _field(-0.2, 1.0),
        }
        for year in ("2020", "2021")
    }

    result = summarize(
        field,
        spectral,
        physical,
        temporal=temporal,
    )

    assert result["selection_confirmed"]
    assert not result["cross_metric_generalization_confirmed"]
    assert result["significant_temporal_curvature_regressions"] == [
        {
            "year": "2021",
            "model": "challenger",
            "metric": "temporal_curvature_rmse",
            "relative_delta_pct": 10.0,
            "p_raw": 0.001,
            "p_holm": 0.002,
        }
    ]


def test_window_months_respect_leap_year_offsets() -> None:
    metrics = WindowMetrics(
        year=np.full(4, 2020),
        t0=np.asarray([0, 91 * 24, 182 * 24, 335 * 24]),
        tau=np.ones(4),
        mse=np.ones((4, 1)),
        channels=("t2m",),
    )

    assert _window_months(metrics).tolist() == [1, 4, 7, 12]


def test_extreme_comparison_uses_common_bilinear_hard_subset(
    tmp_path,
) -> None:
    window_root = tmp_path / "window_metrics"
    window_root.mkdir()
    n_dates = 120
    year = np.full(n_dates * 2, 2021, dtype=np.int16)
    t0 = np.repeat(np.arange(n_dates) * 24, 2)
    tau = np.tile(np.asarray([2, 4], dtype=np.int8), n_dates)
    hard = np.repeat(np.arange(n_dates) % 10 == 0, 2)
    baseline_rmse = np.where(hard, 10.0, 1.0)
    winner_rmse = np.where(hard, 8.0, 0.8)
    challenger_rmse = np.where(hard, 4.0, 0.9)

    def mse(values: np.ndarray) -> np.ndarray:
        return np.repeat(values[:, None] ** 2, 2, axis=1)

    np.savez(
        window_root / "winner.npz",
        year=year,
        t0=t0,
        tau=tau,
        mse_norm_model=mse(winner_rmse),
        mse_norm_bilinear=mse(baseline_rmse),
        channel_names=np.asarray(["t2m", "u10"]),
    )
    np.savez(
        window_root / "challenger.npz",
        year=year,
        t0=t0,
        tau=tau,
        mse_norm_model=mse(challenger_rmse),
        channel_names=np.asarray(["t2m", "u10"]),
    )

    result = extreme_field_comparisons(
        "winner",
        ("winner", "challenger"),
        tmp_path,
        taus=np.asarray([2, 4]),
        quantile=0.95,
        block_days=7,
        draws=1000,
        seed=9,
    )

    comparison = result["comparisons"]["challenger"]
    assert result["n_windows"] == 24
    assert comparison["delta_left_minus_right"] == pytest.approx(4.0)
    assert comparison["p_paired_block_permutation"] < 0.05


def test_single_spectral_regression_blocks_cross_metric_claim() -> None:
    groups = {
        group: {
            "challenger": _field(-0.1, 0.01),
            "bilinear": _field(-0.2, 0.01),
        }
        for group in ("seen", "unseen")
    }
    field = {"2020": groups, "2021": groups}
    spectral = {
        "2": {
            "challenger": {
                "energy_log_error": _field(0.1, 0.001),
                "shape_log_error": _field(-0.1, 0.01),
                "coherence": _field(0.1, 0.01),
            }
        }
    }
    physical = {
        year: {
            group: {
                "challenger": _physical(-0.1, 0.01),
                "bilinear": _physical(-0.1, 0.01),
            }
            for group in ("seen", "unseen")
        }
        for year in ("2020", "2021")
    }

    result = summarize(field, spectral, physical)

    assert result["selection_confirmed"]
    assert not result["cross_metric_generalization_confirmed"]
    assert not result["significant_spectral_dominance"]
    assert result["significant_spectral_regressions"] == [
        {
            "tau": 2,
            "model": "challenger",
            "metric": "energy_log_error",
        }
    ]


def test_field_challenger_win_survives_spectral_summary() -> None:
    groups = {
        "seen": {
            "challenger": _field(-0.1, 0.01),
            "bilinear": _field(-0.2, 0.01),
        },
        "unseen": {
            "challenger": _field(-0.1, 0.01),
            "bilinear": _field(-0.2, 0.01),
        },
    }
    field = {
        "2020": groups,
        "2021": {
            "seen": groups["seen"],
            "unseen": {
                "challenger": _field(0.1, 0.001),
                "bilinear": _field(-0.2, 0.001),
            },
        },
    }
    spectral = {
        "2": {
            "challenger": {
                "energy_log_error": _field(-0.1, 0.01),
                "shape_log_error": _field(-0.1, 0.01),
                "coherence": _field(0.1, 0.01),
            }
        }
    }
    physical = {
        year: {
            group: {
                "challenger": _physical(-0.1, 0.01),
                "bilinear": _physical(-0.1, 0.01),
            }
            for group in ("seen", "unseen")
        }
        for year in ("2020", "2021")
    }

    result = summarize(field, spectral, physical)

    assert not result["selection_confirmed"]
    assert result["significant_field_challenger_wins"] == [
        {
            "year": "2021",
            "tau_group": "unseen",
            "model": "challenger",
            "relative_delta_pct": 10.0,
            "p_raw": 0.001,
            "p_holm": 0.001,
        }
    ]


def test_relaxed_selection_gate_cannot_be_confirmed() -> None:
    groups = {
        group: {
            "challenger": _field(-0.1, 0.01),
            "bilinear": _field(-0.2, 0.01),
        }
        for group in ("seen", "unseen")
    }
    field = {"2020": groups, "2021": groups}
    spectral = {
        "2": {
            "challenger": {
                "energy_log_error": _field(-0.1, 0.01),
                "shape_log_error": _field(-0.1, 0.01),
                "coherence": _field(0.1, 0.01),
            }
        }
    }
    physical = {
        year: {
            group: {
                "challenger": _physical(-0.1, 0.01),
                "bilinear": _physical(-0.1, 0.01),
            }
            for group in ("seen", "unseen")
        }
        for year in ("2020", "2021")
    }

    result = summarize(
        field,
        spectral,
        physical,
        selection_quality_gate_passed=False,
    )

    assert not result["selection_quality_gate_passed"]
    assert not result["selection_confirmed"]
    assert not result["cross_metric_generalization_confirmed"]


def test_bilinear_confirmation_corrects_across_years() -> None:
    field = {
        year: {
            group: {
                "challenger": _field(-0.1, 0.20),
                "bilinear": _field(-0.2, 0.03),
            }
            for group in ("seen", "unseen")
        }
        for year in ("2020", "2021")
    }
    spectral = {
        "2": {
            "challenger": {
                "energy_log_error": _field(-0.1, 0.20),
                "shape_log_error": _field(-0.1, 0.20),
                "coherence": _field(0.1, 0.20),
            }
        }
    }
    physical = {
        year: {
            group: {
                "challenger": _physical(-0.1, 0.20),
                "bilinear": _physical(-0.1, 0.20),
            }
            for group in ("seen", "unseen")
        }
        for year in ("2020", "2021")
    }

    result = summarize(field, spectral, physical)

    assert not result["selection_confirmed"]
    assert result["unseen_bilinear_year_family"]["2020"]["p_raw"] == 0.03
    assert result["unseen_bilinear_year_family"]["2020"]["p_holm"] == 0.06
    assert result["unseen_bilinear_year_family"]["2021"]["p_holm"] == 0.06


def test_holm_adjustment_controls_a_test_family() -> None:
    results = [_field(0.1, value) for value in (0.01, 0.03, 0.04)]

    _annotate_holm(results, output_key="p_holm")

    assert [result["p_holm"] for result in results] == pytest.approx(
        [0.03, 0.06, 0.06]
    )


def test_holm_adjustment_rejects_invalid_p_value() -> None:
    with pytest.raises(ValueError, match="invalid p-value"):
        _annotate_holm([_field(0.1, float("nan"))], output_key="p_holm")


def test_preflight_verifies_frozen_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    models = ("winner", "reference")
    roots = {
        year: tmp_path / year
        for year in ("2020", "2021")
    }
    spectra_root = tmp_path / "spectra"
    provenance = {
        "static_features": {"sha256": "static"},
        "pressure_level_stats": {"sha256": "pressure"},
        "surface_stats": {"sha256": "surface"},
        "climatology": {"identity_sha256": "climatology"},
    }
    spectral_provenance = {
        key: provenance[key]
        for key in (
            "static_features",
            "pressure_level_stats",
            "surface_stats",
        )
    }
    dataset_provenance_2020 = {
        "root": "/synthetic/memmap",
        "years": [2020],
        "identity_sha256": "dataset-2020",
    }
    dataset_provenance_2021 = {
        "root": "/synthetic/memmap",
        "years": [2021],
        "identity_sha256": "dataset-2021",
    }
    for root in (*roots.values(), spectra_root):
        root.mkdir()
    for root in roots.values():
        for model in models:
            (root / f"{model}.json").write_text(
                json.dumps({"window_metrics_file": f"{model}.npz"})
            )
            np.savez(root / f"{model}.npz", value=np.asarray([1]))
    for model in models:
        np.savez(
            spectra_root / f"{model}_tau2.npz",
            tau=np.asarray(2),
        )

    field_result = {
        "evaluation_full_year": True,
        "window_index_sha256": "field-index",
        "evaluation_input_provenance": provenance,
        "checkpoint_sha256": "checkpoint",
    }
    spectral_result = {
        "window_index_sha256": "spectral-index",
        "spectral_grid_sha256": "grid",
        "spectral_channel_names_sha256": "channels",
        "evaluation_input_provenance": spectral_provenance,
        "evaluation_dataset_provenance": dataset_provenance_2020,
        "checkpoint_sha256": "checkpoint",
    }
    def load_field(path):
        year = int(path.parent.name)
        return {
            **field_result,
            "evaluation_dataset_provenance": (
                dataset_provenance_2020
                if year == 2020
                else dataset_provenance_2021
            ),
        }

    monkeypatch.setattr(
        validation,
        "load_selection_field_metrics",
        load_field,
    )
    monkeypatch.setattr(
        validation,
        "load_selection_spectral_metrics",
        lambda *args, **kwargs: dict(spectral_result),
    )
    cost_path = tmp_path / "cost.json"
    cost_path.write_text(
        json.dumps(
            {
                "models": {
                    model: {"checkpoint_sha256": "checkpoint"}
                    for model in models
                }
            }
        )
    )
    normalization_path = tmp_path / "normalization_provenance.json"
    normalization_path.write_text("{}")
    selection = {
        "schema_version": 12,
        "models": {
            model: {"checkpoint_sha256": "checkpoint"}
            for model in models
        },
        "paired_window_index_sha256": {
            "field_2020": "field-index",
            "field_2021": "field-index",
            "spectral_2020": "spectral-index",
            "spectral_grid": "grid",
            "spectral_channels": "channels",
            "field_input_provenance": _canonical_sha256(provenance),
            "spectral_input_provenance": _canonical_sha256(
                spectral_provenance
            ),
            "field_dataset_2020": _canonical_sha256(
                dataset_provenance_2020
            ),
            "field_dataset_2021": _canonical_sha256(
                dataset_provenance_2021
            ),
            "spectral_dataset_2020": _canonical_sha256(
                dataset_provenance_2020
            ),
        },
        "selection_rule": {
            "hf_ell_min": 180,
            "required_lmax": 359,
            "expected_spectral_channels": 24,
            "spectral_taus": [2],
        },
        "cost_artifact": {
            "path": str(cost_path),
            "size_bytes": cost_path.stat().st_size,
            "sha256": _file_sha256(cost_path),
            "checkpoint_sha256_by_model": {
                model: "checkpoint" for model in models
            },
        },
        "normalization_provenance": {
            "verified": True,
            "manifest_path": str(normalization_path),
            "manifest_sha256": _file_sha256(normalization_path),
            "declared_period": [1979, 2019],
            "evaluation_years": [2020, 2021],
            "artifacts": {
                "pressure_level_stats": {"sha256": "pressure"},
                "surface_stats": {"sha256": "surface"},
            },
        },
    }

    result = verify_selection_artifacts(
        selection,
        models,
        roots["2020"],
        roots["2021"],
        spectra_root,
        {2},
    )

    assert set(result["field"]["2020"]) == set(models)
    assert set(result["spectral"]) == set(models)
    assert len(
        result["spectral"]["winner"]["npz_sha256_by_tau"]["2"]
    ) == 64

    stale_selection = json.loads(json.dumps(selection))
    stale_selection["schema_version"] = 11
    with pytest.raises(
        ValueError,
        match="unsupported frozen selection schema",
    ):
        verify_selection_artifacts(
            stale_selection,
            models,
            roots["2020"],
            roots["2021"],
            spectra_root,
            {2},
        )

    selection_only = json.loads(json.dumps(selection))
    del selection_only["paired_window_index_sha256"]["field_2021"]
    del selection_only["paired_window_index_sha256"]["field_dataset_2021"]
    selection_only["ood_attached_at_selection_time"] = False
    selection_only_result = verify_selection_artifacts(
        selection_only,
        models,
        roots["2020"],
        roots["2021"],
        spectra_root,
        {2},
    )
    assert set(selection_only_result["field"]["2021"]) == set(models)

    normalization_path.write_text('{"changed": true}')
    with pytest.raises(
        ValueError,
        match="normalization manifest changed",
    ):
        verify_selection_artifacts(
            selection,
            models,
            roots["2020"],
            roots["2021"],
            spectra_root,
            {2},
        )
    normalization_path.write_text("{}")

    def load_changed_field(path):
        result = load_field(path)
        result["window_index_sha256"] = "changed-index"
        return result

    monkeypatch.setattr(
        validation,
        "load_selection_field_metrics",
        load_changed_field,
    )
    with pytest.raises(ValueError, match="window index changed"):
        verify_selection_artifacts(
            selection,
            models,
            roots["2020"],
            roots["2021"],
            spectra_root,
            {2},
        )

    selection["cost_artifact"]["size_bytes"] = None
    with pytest.raises(
        ValueError,
        match="selection cost artifact changed after selection",
    ):
        verify_selection_artifacts(
            selection,
            models,
            roots["2020"],
            roots["2021"],
            spectra_root,
            {2},
        )
