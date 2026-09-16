from pathlib import Path

import pytest

from tools.eval import summarize_upr_vs_pp3_seeds as seed_summary
from tools.eval.summarize_upr_vs_pp3_seeds import (
    _aggregate,
    _metric_comparison,
    build_report,
)


def test_metric_comparison_respects_metric_direction() -> None:
    lower = _metric_comparison(0.8, 1.0, better="lower")
    higher = _metric_comparison(0.9, 0.7, better="higher")

    assert lower["candidate_improvement"] == pytest.approx(0.2)
    assert higher["candidate_improvement"] == pytest.approx(0.2)


def test_aggregate_reports_seed_consistency_and_spread() -> None:
    rows = [
        _metric_comparison(candidate, reference, better="lower")
        for candidate, reference in ((0.8, 1.0), (0.9, 1.0), (0.7, 1.0))
    ]

    result = _aggregate(rows)

    assert result["candidate_mean"] == pytest.approx(0.8)
    assert result["reference_mean"] == pytest.approx(1.0)
    assert result["improvement_mean"] == pytest.approx(0.2)
    assert result["improvement_std"] == pytest.approx(0.1)
    assert result["candidate_better_all_seeds"]


def test_aggregate_rejects_non_finite_or_single_seed() -> None:
    with pytest.raises(ValueError, match="at least two finite"):
        _aggregate([_metric_comparison(0.8, 1.0, better="lower")])
    with pytest.raises(ValueError, match="at least two finite"):
        _aggregate(
            [
                _metric_comparison(0.8, 1.0, better="lower"),
                _metric_comparison(float("nan"), 1.0, better="lower"),
            ]
        )


def test_build_report_pairs_matching_candidate_and_reference_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field_names: list[str] = []
    field_paths: list[Path] = []
    spectral_names: list[str] = []
    spectral_roots: list[Path] = []
    common = {
        "static_features": {"sha256": "static"},
        "pressure_level_stats": {"sha256": "pressure"},
        "surface_stats": {"sha256": "surface"},
    }

    def fake_field(path: Path) -> dict[str, object]:
        field_paths.append(path)
        field_names.append(path.stem)
        candidate = path.stem.startswith("upr_")
        lower = 0.8 if candidate else 1.0
        higher = 0.9 if candidate else 0.7
        return {
            "unseen_rmse": lower,
            "unseen_skill": higher,
            "unseen_acc": higher,
            "unseen_physical_ratio": lower,
            "unseen_physical_ratio_max": lower,
            "unseen_cvar95_rmse": lower,
            "unseen_worst_season_rmse": lower,
            "checkpoint_sha256": (
                "candidate-checkpoint"
                if candidate
                else "reference-checkpoint"
            ),
            "evaluation_full_year": True,
            "window_index_sha256": "paired-field-index",
            "evaluation_input_provenance": {
                **common,
                "climatology": {"cache_identity_sha256": "climatology"},
            },
            "evaluation_dataset_provenance": {
                "root": "/synthetic/memmap",
                "years": [2020],
                "identity_sha256": "dataset",
            },
        }

    def fake_spectral(
        root: Path,
        model: str,
        hf_ell_min: int,
        taus: set[int],
    ) -> dict[str, object]:
        del hf_ell_min, taus
        spectral_roots.append(root)
        spectral_names.append(model)
        candidate = model.startswith("upr_")
        return {
            "hf_log_energy_error": 0.1 if candidate else 0.3,
            "hf_log_shape_error": 0.08 if candidate else 0.25,
            "hf_coherence": 0.9 if candidate else 0.7,
            "checkpoint_sha256": (
                "candidate-checkpoint"
                if candidate
                else "reference-checkpoint"
            ),
            "window_index_sha256": "paired-spectral-index",
            "evaluation_input_provenance": common,
            "evaluation_dataset_provenance": {
                "root": "/synthetic/memmap",
                "years": [2020],
                "identity_sha256": "dataset",
            },
        }

    def fake_temporal(path: Path) -> dict[str, object]:
        candidate = path.stem.startswith("upr_")
        return {
            "curvature_rmse": 0.8 if candidate else 1.0,
            "bilinear_curvature_rmse": 1.2,
            "curvature_ratio_to_bilinear": (
                0.8 / 1.2 if candidate else 1.0 / 1.2
            ),
            "window_index_sha256": "paired-temporal-index",
            "centers": (1, 2, 3, 4, 5),
        }

    monkeypatch.setattr(seed_summary, "load_field_metrics", fake_field)
    monkeypatch.setattr(seed_summary, "load_spectral_metrics", fake_spectral)
    monkeypatch.setattr(
        seed_summary,
        "load_temporal_summary",
        fake_temporal,
    )
    root = Path("/unused")
    report = build_report(
        {
            "winner": "upr_implicit_global_14m",
            "models": {
                "upr_implicit_global_14m": {"mean_rank": 1.0},
                "weatherbridge_ref": {"mean_rank": 2.0},
            },
        },
        primary_root_2020=root,
        primary_root_2021=root,
        candidate_replicate_root_2020=root,
        candidate_replicate_root_2021=root,
        reference_replicate_root_2020=root,
        reference_replicate_root_2021=root,
        primary_spectra_root_2020=root,
        candidate_replicate_spectra_root_2020=root,
        reference_replicate_spectra_root_2020=root,
    )

    assert "upr_implicit_global_14m" in field_names
    assert "weatherbridge_ref" in field_names
    assert "upr_implicit_global_14m_s202708" in field_names
    assert "weatherbridge_ref_s202709" in field_names
    assert "upr_implicit_global_14m_s202709" in spectral_names
    assert "weatherbridge_ref_s202708" in spectral_names
    assert report["primary_quality_superiority_seed_consistent"]
    assert report["system_superiority_seed_consistent"]
    assert report["architecture_superiority_seed_consistent"] is None
    assert report["training_objective"] == {
        "candidate_lambda_hf": 0.05,
        "reference_lambda_hf": 0.0,
        "matched": False,
    }
    assert report["schema_version"] == 6
    assert report["aggregate"]["temporal"]["2021"]["curvature_rmse"][
        "candidate_better_all_seeds"
    ]
    assert report["aggregate"]["spectral_2020"]["hf_log_shape_error"][
        "candidate_better_all_seeds"
    ]
    assert (
        report["paired_evaluation_input_provenance_sha256"]["202707"]["2020"]
        == report["paired_evaluation_input_provenance_sha256"]["202709"]["2021"]
    )

    field_names.clear()
    spectral_names.clear()
    fresh_report = build_report(
        {
            "winner": "upr_implicit_global_14m",
            "models": {
                "upr_implicit_global_14m": {"mean_rank": 1.0},
                "weatherbridge_ref": {"mean_rank": 2.0},
            },
        },
        primary_root_2020=root,
        primary_root_2021=root,
        candidate_replicate_root_2020=root,
        candidate_replicate_root_2021=root,
        reference_replicate_root_2020=root,
        reference_replicate_root_2021=root,
        primary_spectra_root_2020=root,
        candidate_replicate_spectra_root_2020=root,
        reference_replicate_spectra_root_2020=root,
        candidate_primary_seed_from_replicate=True,
    )
    assert "upr_implicit_global_14m_s202707" in field_names
    assert "upr_implicit_global_14m" not in field_names
    assert "weatherbridge_ref" in field_names
    assert "upr_implicit_global_14m_s202707" in spectral_names
    assert (
        fresh_report["candidate_seed_artifact_mode"]
        == "fresh_postselection_replicates"
    )

    matched_objective_report = build_report(
        {
            "winner": "upr_implicit_global_14m",
            "models": {
                "upr_implicit_global_14m": {"mean_rank": 1.0},
                "weatherbridge_ref": {"mean_rank": 2.0},
            },
        },
        primary_root_2020=root,
        primary_root_2021=root,
        candidate_replicate_root_2020=root,
        candidate_replicate_root_2021=root,
        reference_replicate_root_2020=root,
        reference_replicate_root_2021=root,
        primary_spectra_root_2020=root,
        candidate_replicate_spectra_root_2020=root,
        reference_replicate_spectra_root_2020=root,
        candidate_lambda_hf=0.0,
        reference_lambda_hf=0.0,
    )
    assert matched_objective_report["training_objective"]["matched"]
    assert matched_objective_report[
        "architecture_superiority_seed_consistent"
    ]

    nohf_report = build_report(
        {
            "winner": "upr_implicit_global_14m",
            "models": {
                "upr_implicit_global_14m": {"mean_rank": 1.0},
                "weatherbridge_ref": {"mean_rank": 2.0},
            },
        },
        primary_root_2020=Path("/candidate/2020"),
        primary_root_2021=Path("/candidate/2021"),
        candidate_replicate_root_2020=root,
        candidate_replicate_root_2021=root,
        reference_replicate_root_2020=root,
        reference_replicate_root_2021=root,
        primary_spectra_root_2020=Path("/candidate/spectra"),
        candidate_replicate_spectra_root_2020=root,
        reference_replicate_spectra_root_2020=root,
        candidate_lambda_hf=0.0,
        reference_lambda_hf=0.0,
        candidate_artifact_name="upr_implicit_global_14m_nohf",
        candidate_artifact_mode="fresh_postselection_replicates",
        reference_primary_root_2020=Path("/reference/2020"),
        reference_primary_root_2021=Path("/reference/2021"),
        reference_primary_spectra_root_2020=Path("/reference/spectra"),
    )
    assert nohf_report["candidate"] == "upr_implicit_global_14m_nohf"
    assert nohf_report["selected_architecture"] == (
        "upr_implicit_global_14m"
    )
    assert nohf_report["candidate_seed_artifact_mode"] == (
        "fresh_postselection_replicates"
    )
    assert nohf_report["architecture_superiority_seed_consistent"]
    assert Path(
        "/candidate/2020/upr_implicit_global_14m_nohf.json"
    ) in field_paths
    assert Path("/reference/2020/weatherbridge_ref.json") in field_paths
    assert Path("/candidate/spectra") in spectral_roots
    assert Path("/reference/spectra") in spectral_roots

    with pytest.raises(ValueError, match="artifact override"):
        build_report(
            {
                "winner": "upr_implicit_global_14m",
                "models": {
                    "upr_implicit_global_14m": {"mean_rank": 1.0},
                    "weatherbridge_ref": {"mean_rank": 2.0},
                },
            },
            primary_root_2020=root,
            primary_root_2021=root,
            candidate_replicate_root_2020=root,
            candidate_replicate_root_2021=root,
            reference_replicate_root_2020=root,
            reference_replicate_root_2021=root,
            primary_spectra_root_2020=root,
            candidate_replicate_spectra_root_2020=root,
            reference_replicate_spectra_root_2020=root,
            candidate_artifact_name="unrelated_model",
        )
