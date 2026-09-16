from pathlib import Path

import pytest

from tools.eval import summarize_upr_vs_pp3_ood_spectra as summary


def test_build_report_requires_paired_2021_spectra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []
    roots: list[Path] = []

    def fake_load(
        root: Path,
        model: str,
        hf_ell_min: int,
        taus: set[int],
        *,
        required_lmax: int,
        expected_channels: int,
    ) -> dict:
        roots.append(root)
        assert hf_ell_min == 180
        assert taus == {2, 4}
        assert required_lmax == 359
        assert expected_channels == 24
        loaded.append(model)
        candidate = model.startswith("upr_")
        return {
            "hf_log_energy_error": 0.1 if candidate else 0.3,
            "hf_log_shape_error": 0.08 if candidate else 0.25,
            "hf_coherence": 0.9 if candidate else 0.7,
            "window_index_sha256": "paired-2021-index",
            "evaluation_input_provenance": {"static": "same"},
            "evaluation_dataset_provenance": {
                "root": "/data",
                "years": [2021],
                "identity_sha256": "2021",
            },
            "checkpoint_sha256": f"{model}-checkpoint",
        }

    monkeypatch.setattr(summary, "load_spectral_metrics", fake_load)
    report = summary.build_report(
        {
            "winner": "upr_implicit_global_14m",
            "models": {
                "upr_implicit_global_14m": {"mean_rank": 1.0},
                "weatherbridge_ref": {"mean_rank": 2.0},
            },
        },
        Path("/unused"),
    )

    assert "upr_implicit_global_14m_s202709" in loaded
    assert "weatherbridge_ref_s202708" in loaded
    assert report["spectral_protocol"]["used_for_model_selection"] is False
    assert report["ood_spectral_superiority_seed_consistent"]
    assert not report["checkpoint_linkage_verified"]
    assert (
        report["architecture_ood_spectral_superiority_seed_consistent"]
        is None
    )
    assert report["aggregate"]["hf_log_energy_error"][
        "candidate_better_all_seeds"
    ]
    assert report["aggregate"]["hf_log_shape_error"][
        "candidate_better_all_seeds"
    ]

    loaded.clear()
    fresh_report = summary.build_report(
        {
            "winner": "upr_implicit_global_14m",
            "models": {
                "upr_implicit_global_14m": {"mean_rank": 1.0},
                "weatherbridge_ref": {"mean_rank": 2.0},
            },
        },
        Path("/unused"),
        candidate_primary_seed_from_replicate=True,
    )
    assert "upr_implicit_global_14m_s202707" in loaded
    assert "upr_implicit_global_14m" not in loaded
    assert "weatherbridge_ref" in loaded
    assert (
        fresh_report["candidate_seed_artifact_mode"]
        == "fresh_postselection_replicates"
    )

    loaded.clear()
    roots.clear()
    nohf_report = summary.build_report(
        {
            "winner": "upr_implicit_global_14m",
            "models": {
                "upr_implicit_global_14m": {"mean_rank": 1.0},
                "weatherbridge_ref": {"mean_rank": 2.0},
            },
        },
        Path("/candidate"),
        reference_spectra_root=Path("/reference"),
        candidate_artifact_name="upr_implicit_global_14m_nohf",
        candidate_artifact_mode="fresh_postselection_replicates",
        candidate_lambda_hf=0.0,
        reference_lambda_hf=0.0,
        checkpoint_report={
            "schema_version": 6,
            "candidate": "upr_implicit_global_14m_nohf",
            "reference": "weatherbridge_ref",
            "seeds": [202707, 202708, 202709],
            "candidate_seed_artifact_mode": (
                "fresh_postselection_replicates"
            ),
            "per_seed": {
                "202707": {
                    "checkpoint_sha256": {
                        "candidate": (
                            "upr_implicit_global_14m_nohf-checkpoint"
                        ),
                        "reference": "weatherbridge_ref-checkpoint",
                    }
                },
                "202708": {
                    "checkpoint_sha256": {
                        "candidate": (
                            "upr_implicit_global_14m_nohf_s202708-checkpoint"
                        ),
                        "reference": (
                            "weatherbridge_ref_s202708-checkpoint"
                        ),
                    }
                },
                "202709": {
                    "checkpoint_sha256": {
                        "candidate": (
                            "upr_implicit_global_14m_nohf_s202709-checkpoint"
                        ),
                        "reference": (
                            "weatherbridge_ref_s202709-checkpoint"
                        ),
                    }
                },
            },
        },
    )
    assert nohf_report["candidate"] == "upr_implicit_global_14m_nohf"
    assert nohf_report["selected_architecture"] == (
        "upr_implicit_global_14m"
    )
    assert nohf_report["training_objective"]["matched"]
    assert nohf_report["checkpoint_linkage_verified"]
    assert nohf_report[
        "architecture_ood_spectral_superiority_seed_consistent"
    ]
    assert Path("/candidate") in roots
    assert Path("/reference") in roots


def test_build_report_rejects_non_2021_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load(*args, **kwargs) -> dict:
        del args, kwargs
        return {
            "hf_log_energy_error": 0.1,
            "hf_log_shape_error": 0.08,
            "hf_coherence": 0.9,
            "window_index_sha256": "same",
            "evaluation_input_provenance": {"same": True},
            "evaluation_dataset_provenance": {"years": [2020]},
            "checkpoint_sha256": "checkpoint",
        }

    monkeypatch.setattr(summary, "load_spectral_metrics", fake_load)
    with pytest.raises(ValueError, match="expected 2021"):
        summary.build_report(
            {
                "winner": "upr_lite",
                "models": {"upr_lite": {"mean_rank": 1.0}},
            },
            Path("/unused"),
        )
