import json
from pathlib import Path

import numpy as np
import pytest

from tools.eval.summarize_upr_lite_seeds import (
    build_seed_report,
    choose_transfer_candidate,
    choose_upr_candidate,
    validate_frozen_selection_for_followup,
)
from tools.train.training_protocol import memmap_dataset_provenance


COMMON_PROVENANCE = {
    "static_features": {"sha256": "static-v1"},
    "pressure_level_stats": {"sha256": "pressure-v1"},
    "surface_stats": {"sha256": "surface-v1"},
}


def _write_metrics(path: Path, rmse: float) -> None:
    year = 2020 if path.parent.name.endswith("20") else 2021
    memmap_root = path.parents[1] / "memmaps"
    memmap_root.mkdir(exist_ok=True)
    metadata = memmap_root / f"wb2_{year}.json"
    if not metadata.exists():
        metadata.write_text(
            json.dumps({"T": 2, "n_channels": 1, "H": 2, "W": 2})
        )
        (np.arange(8, dtype=np.float32) + year).tofile(
            memmap_root / f"wb2_{year}.bin"
        )
    window_dir = path.parent / "window_metrics"
    window_dir.mkdir(exist_ok=True)
    np.savez(
        window_dir / f"{path.stem}.npz",
        year=np.full(8, year),
        t0=np.arange(8) * 24 * 45,
        tau=np.tile(np.array([1, 2]), 4),
        mse_norm_model=np.full((8, 1), rmse ** 2),
        mse_norm_bilinear=np.ones((8, 1)),
    )
    payload = {
        "checkpoint_provenance": {
            "sha256": f"{path.stem}-checkpoint",
        },
        "evaluation_protocol": {
            "full_year": True,
            "save_temporal_metrics": True,
            "temporal_index_sha256": f"temporal-{year}",
            "temporal_centers": [1],
        },
        "temporal_curvature_rmse_norm": {
            "model": rmse,
            "bilinear": 1.0,
        },
        "channel_names": ["t2m"],
        "seen_tau": [1],
        "unseen_tau": [2],
        "per_tau": {
            "1": {
                "model": {
                    "rmse_norm_t2m": rmse,
                    "acc_mean": 0.95,
                    "physical_wind_divergence_nmse": rmse,
                    "physical_wind_vorticity_nmse": rmse,
                    "physical_kinetic_energy_nmse": rmse,
                    "physical_hydrostatic_balance_mse": rmse,
                },
                "bilinear": {
                    "rmse_norm_t2m": 1.0,
                    "acc_mean": 0.0,
                    "physical_wind_divergence_nmse": 1.0,
                    "physical_wind_vorticity_nmse": 1.0,
                    "physical_kinetic_energy_nmse": 1.0,
                    "physical_hydrostatic_balance_mse": 1.0,
                },
            },
            "2": {
                "model": {
                    "rmse_norm_t2m": rmse,
                    "acc_mean": 0.94,
                    "physical_wind_divergence_nmse": rmse,
                    "physical_wind_vorticity_nmse": rmse,
                    "physical_kinetic_energy_nmse": rmse,
                    "physical_hydrostatic_balance_mse": rmse,
                },
                "bilinear": {
                    "rmse_norm_t2m": 1.0,
                    "acc_mean": 0.0,
                    "physical_wind_divergence_nmse": 1.0,
                    "physical_wind_vorticity_nmse": 1.0,
                    "physical_kinetic_energy_nmse": 1.0,
                    "physical_hydrostatic_balance_mse": 1.0,
                },
            },
        },
        "window_metrics_file": f"window_metrics/{path.stem}.npz",
        "evaluation_input_provenance": {
            **COMMON_PROVENANCE,
            "climatology": {"cache_identity_sha256": "climatology-v1"},
        },
        "evaluation_dataset_provenance": memmap_dataset_provenance(
            memmap_root,
            [year],
        ),
    }
    path.write_text(json.dumps(payload))


def _write_spectra(
    root: Path,
    model: str,
    energy_scale: float,
    coherence: float,
) -> None:
    dataset_provenance = memmap_dataset_provenance(
        root.parent / "memmaps",
        [2020],
    )
    for tau in (2, 4):
        np.savez(
            root / f"{model}_tau{tau}.npz",
            ell=np.array([0, 180, 359]),
            pred_El=np.full((1, 3), energy_scale),
            gt_El=np.ones((1, 3)),
            n_samples=np.array(4),
            tau=np.array(tau),
            window_hf_log_shape_error=np.full(
                (4, 1),
                abs(np.log(energy_scale)),
            ),
            window_hf_coherence=np.full((4, 1), coherence),
            window_year=np.full(4, 2020),
            window_t0=np.arange(4) * 24,
            metadata_json=np.array(
                json.dumps(
                    {
                        "evaluation_input_provenance": COMMON_PROVENANCE,
                        "evaluation_dataset_provenance": dataset_provenance,
                        "checkpoint_provenance": {
                            "sha256": f"{model}-checkpoint",
                        },
                    }
                )
            ),
        )


def test_choose_best_upr_when_overall_winner_is_reference() -> None:
    selection = {
        "winner": "weatherbridge_ref",
        "models": {
            "weatherbridge_ref": {"mean_rank": 1.0},
            "upr_lite": {"mean_rank": 3.0},
            "upr_lite_continuous": {"mean_rank": 2.0},
        },
    }
    assert choose_upr_candidate(selection) == "upr_lite_continuous"


def test_validates_schema13_selection_for_followup() -> None:
    selection = {
        "schema_version": 13,
        "ood_attached_at_selection_time": False,
        "winner": "upr",
        "efficiency_winner": "upr",
        "eligible": ["upr"],
        "models": {
            "upr": {"checkpoint_sha256": "checkpoint"},
        },
        "selection_rule": {
            "gate_diagnostics": {
                "field_hour_dominance": {
                    "active": True,
                    "total_cells": 120,
                },
            },
        },
        "paired_window_index_sha256": {
            "field_2020": "field-index",
            "spectral_2020": "spectral-index",
            "field_dataset_2020": "field-data",
            "spectral_dataset_2020": "spectral-data",
        },
    }

    validate_frozen_selection_for_followup(selection)
    selection["selection_rule"]["gate_diagnostics"][
        "field_hour_dominance"
    ]["total_cells"] = 119
    with pytest.raises(ValueError, match="field-by-hour dominance"):
        validate_frozen_selection_for_followup(selection)


def test_choose_best_eligible_upr_when_reference_wins() -> None:
    selection = {
        "winner": "weatherbridge_ref",
        "eligible": ["weatherbridge_ref", "upr_lite_column"],
        "models": {
            "weatherbridge_ref": {"mean_rank": 1.0},
            "upr_lite_implicit_global": {"mean_rank": 2.0},
            "upr_lite_column": {"mean_rank": 3.0},
        },
    }
    assert choose_upr_candidate(selection) == "upr_lite_column"


def test_choose_upr_candidate_maps_average_to_trainable_architecture() -> None:
    selection = {
        "winner": "upr_lite_implicit_global_avg3",
        "models": {
            "upr_lite_implicit_global": {"mean_rank": 2.0},
            "upr_lite_implicit_global_avg3": {"mean_rank": 1.0},
        },
    }

    assert choose_upr_candidate(selection) == "upr_lite_implicit_global"


def test_choose_upr_candidate_maps_q4_average_to_trainable_architecture() -> None:
    selection = {
        "winner": "upr_lite_implicit_global_q4_avg3",
        "models": {
            "upr_lite_implicit_global_q4_avg3": {"mean_rank": 1.0},
        },
    }

    assert (
        choose_upr_candidate(selection)
        == "upr_lite_implicit_global_q4"
    )


def test_choose_capacity_matched_upr_when_reference_wins() -> None:
    selection = {
        "winner": "weatherbridge_ref",
        "models": {
            "weatherbridge_ref": {"mean_rank": 1.0},
            "upr_lite_implicit_global": {"mean_rank": 3.0},
            "upr_implicit_global_14m_avg3": {"mean_rank": 2.0},
        },
    }

    assert choose_upr_candidate(selection) == "upr_implicit_global_14m"


def test_transfer_candidate_can_select_spherical_flow_for_quality() -> None:
    selection = {
        "winner": "flow_spherical_ep",
        "efficiency_winner": "upr_lite_implicit_global_q4",
        "eligible": [
            "flow_spherical_ep",
            "upr_lite_implicit_global_q4",
        ],
        "models": {
            "flow_spherical_ep": {
                "mean_rank": 1.0,
                "quality_mean_rank": 1.0,
                "efficiency_mean_rank": 3.0,
            },
            "upr_lite_implicit_global_q4": {
                "mean_rank": 2.0,
                "quality_mean_rank": 2.0,
                "efficiency_mean_rank": 1.0,
            },
            "weatherbridge_ref": {
                "mean_rank": 3.0,
                "quality_mean_rank": 3.0,
                "efficiency_mean_rank": 2.0,
            },
        },
    }

    assert (
        choose_transfer_candidate(selection, "quality")
        == "flow_spherical_ep"
    )
    assert (
        choose_transfer_candidate(selection, "efficiency")
        == "upr_lite_implicit_global_q4"
    )


def test_transfer_candidate_can_select_spherical_upr_for_quality() -> None:
    selection = {
        "winner": "upr_spherical_implicit_global_14m",
        "efficiency_winner": "upr_lite_column",
        "eligible": [
            "upr_spherical_implicit_global_14m",
            "upr_lite_column",
        ],
        "models": {
            "upr_spherical_implicit_global_14m": {
                "mean_rank": 1.0,
                "quality_mean_rank": 1.0,
                "efficiency_mean_rank": 2.0,
            },
            "upr_lite_column": {
                "mean_rank": 2.0,
                "quality_mean_rank": 2.0,
                "efficiency_mean_rank": 1.0,
            },
        },
    }

    assert (
        choose_transfer_candidate(selection, "quality")
        == "upr_spherical_implicit_global_14m"
    )


def test_transfer_candidate_can_select_amt_for_quality() -> None:
    selection = {
        "winner": "amt",
        "efficiency_winner": "upr_lite_column",
        "eligible": ["amt", "upr_lite_column"],
        "models": {
            "amt": {
                "mean_rank": 1.0,
                "quality_mean_rank": 1.0,
                "efficiency_mean_rank": 3.0,
            },
            "upr_lite_column": {
                "mean_rank": 2.0,
                "quality_mean_rank": 2.0,
                "efficiency_mean_rank": 1.0,
            },
        },
    }

    assert choose_transfer_candidate(selection, "quality") == "amt"
    assert (
        choose_transfer_candidate(selection, "efficiency")
        == "upr_lite_column"
    )


def test_transfer_candidate_can_select_residual_amt_for_quality() -> None:
    selection = {
        "winner": "amt_residual",
        "efficiency_winner": "upr_lite_column",
        "eligible": ["amt_residual", "upr_lite_column"],
        "models": {
            "amt_residual": {
                "mean_rank": 1.0,
                "quality_mean_rank": 1.0,
                "efficiency_mean_rank": 3.0,
            },
            "upr_lite_column": {
                "mean_rank": 2.0,
                "quality_mean_rank": 2.0,
                "efficiency_mean_rank": 1.0,
            },
        },
    }

    assert (
        choose_transfer_candidate(selection, "quality")
        == "amt_residual"
    )
    assert (
        choose_transfer_candidate(selection, "efficiency")
        == "upr_lite_column"
    )


def test_transfer_candidate_falls_back_when_reference_wins() -> None:
    selection = {
        "winner": "weatherbridge_ref",
        "efficiency_winner": "atmvfi_ref",
        "eligible": [
            "weatherbridge_ref",
            "atmvfi_ref",
            "upr_lite_column",
            "flow_spherical_ep",
        ],
        "models": {
            "weatherbridge_ref": {
                "mean_rank": 1.0,
                "quality_mean_rank": 1.0,
                "efficiency_mean_rank": 3.0,
            },
            "atmvfi_ref": {
                "mean_rank": 3.0,
                "quality_mean_rank": 3.0,
                "efficiency_mean_rank": 1.0,
            },
            "upr_lite_column": {
                "mean_rank": 2.0,
                "quality_mean_rank": 2.0,
                "efficiency_mean_rank": 4.0,
            },
            "flow_spherical_ep": {
                "mean_rank": 4.0,
                "quality_mean_rank": 4.0,
                "efficiency_mean_rank": 2.0,
            },
        },
    }

    assert (
        choose_transfer_candidate(selection, "quality")
        == "upr_lite_column"
    )
    assert (
        choose_transfer_candidate(selection, "efficiency")
        == "flow_spherical_ep"
    )


def test_transfer_candidate_rejects_unknown_objective() -> None:
    with pytest.raises(ValueError, match="unsupported transfer objective"):
        choose_transfer_candidate({"models": {}}, "latency")


def test_build_seed_report_uses_base_and_replicate_roots(tmp_path: Path) -> None:
    roots = {
        name: tmp_path / name
        for name in (
            "base20",
            "base21",
            "rep20",
            "rep21",
            "base_spec",
            "rep_spec",
        )
    }
    for root in roots.values():
        root.mkdir()
    candidate = "upr_lite_continuous"
    for year in ("20", "21"):
        _write_metrics(roots[f"base{year}"] / f"{candidate}.json", 0.50)
        _write_metrics(
            roots[f"rep{year}"] / f"{candidate}_s202708.json",
            0.51,
        )
        _write_metrics(
            roots[f"rep{year}"] / f"{candidate}_s202709.json",
            0.49,
        )
    _write_spectra(roots["base_spec"], candidate, 0.98, 0.90)
    np.savez(
        roots["base_spec"] / f"{candidate}_tau1.npz",
        ell=np.array([0, 180, 359]),
        pred_El=np.full((1, 3), 100.0),
        gt_El=np.ones((1, 3)),
        n_samples=np.array(4),
        tau=np.array(1),
        window_hf_log_shape_error=np.full((4, 1), np.log(100.0)),
        window_hf_coherence=np.zeros((4, 1)),
        window_year=np.full(4, 2020),
        window_t0=np.arange(4) * 24,
        metadata_json=np.array(
            json.dumps(
                {
                    "evaluation_input_provenance": COMMON_PROVENANCE,
                    "evaluation_dataset_provenance": (
                        memmap_dataset_provenance(
                            tmp_path / "memmaps",
                            [2020],
                        )
                    ),
                    "checkpoint_provenance": {
                        "sha256": f"{candidate}-checkpoint",
                    },
                }
            )
        ),
    )
    _write_spectra(
        roots["rep_spec"],
        f"{candidate}_s202708",
        1.00,
        0.91,
    )
    _write_spectra(
        roots["rep_spec"],
        f"{candidate}_s202709",
        1.02,
        0.92,
    )
    report = build_seed_report(
        {"winner": candidate, "models": {candidate: {"mean_rank": 1.0}}},
        roots["base20"],
        roots["base21"],
        roots["rep20"],
        roots["rep21"],
        spectra_root_2020=roots["base_spec"],
        replicate_spectra_root_2020=roots["rep_spec"],
        require_temporal=True,
    )
    assert report["candidate"] == candidate
    assert report["schema_version"] == 5
    assert report["three_seed_consistent"]
    assert report["aggregate"]["2021"]["unseen_rmse"]["mean"] == 0.5
    assert report["aggregate"]["spectral_2020"]["hf_energy_ratio"]["mean"] == 1.0
    assert report["aggregate"]["temporal_2021"][
        "curvature_ratio_to_bilinear"
    ]["mean"] == 0.5
    assert report["aggregate"]["spectral_2020"]["hf_log_shape_error"][
        "mean"
    ] == pytest.approx(
        np.mean(
            [
                abs(np.log(0.98)),
                abs(np.log(1.00)),
                abs(np.log(1.02)),
            ]
        )
    )
    assert "spectral_2020" in report["paired_window_index_sha256"]
    assert (
        report["evaluation_input_provenance_sha256"]["field_2020"]
        == report["evaluation_input_provenance_sha256"]["field_2021"]
    )

    for year in ("20", "21"):
        _write_metrics(
            roots[f"rep{year}"] / f"{candidate}_s202707.json",
            0.52,
        )
    _write_spectra(
        roots["rep_spec"],
        f"{candidate}_s202707",
        0.96,
        0.89,
    )
    fresh_report = build_seed_report(
        {"winner": candidate, "models": {candidate: {"mean_rank": 1.0}}},
        roots["base20"],
        roots["base21"],
        roots["rep20"],
        roots["rep21"],
        spectra_root_2020=roots["base_spec"],
        replicate_spectra_root_2020=roots["rep_spec"],
        require_temporal=True,
        all_seeds_from_replicates=True,
    )
    assert (
        fresh_report["seed_artifact_mode"]
        == "fresh_postselection_replicates"
    )
    assert fresh_report["per_seed"]["202707"]["2020"][
        "unseen_rmse"
    ] == pytest.approx(0.52)

    mismatched_path = roots["rep21"] / f"{candidate}_s202709.json"
    mismatched = json.loads(mismatched_path.read_text())
    mismatched["evaluation_input_provenance"]["climatology"][
        "cache_identity_sha256"
    ] = "climatology-v2"
    mismatched_path.write_text(json.dumps(mismatched))
    with pytest.raises(ValueError, match="2021 seed input-provenance"):
        build_seed_report(
            {"winner": candidate, "models": {candidate: {"mean_rank": 1.0}}},
            roots["base20"],
            roots["base21"],
            roots["rep20"],
            roots["rep21"],
        )
    _write_metrics(mismatched_path, 0.49)

    incomplete_path = (
        roots["rep21"] / f"{candidate}_s202709.json"
    )
    incomplete = json.loads(incomplete_path.read_text())
    incomplete["evaluation_protocol"]["full_year"] = False
    incomplete_path.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="2021 field metrics are not full-year"):
        build_seed_report(
            {"winner": candidate, "models": {candidate: {"mean_rank": 1.0}}},
            roots["base20"],
            roots["base21"],
            roots["rep20"],
            roots["rep21"],
        )
