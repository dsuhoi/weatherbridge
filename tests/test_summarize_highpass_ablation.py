import json
from pathlib import Path

import numpy as np
import pytest

from tools.eval.summarize_highpass_ablation import build_report
from tools.train.training_protocol import memmap_dataset_provenance


INPUT_PROVENANCE = {
    "static_features": {"sha256": "static"},
    "pressure_level_stats": {"sha256": "pressure"},
    "surface_stats": {"sha256": "surface"},
    "climatology": {"cache_identity_sha256": "climatology"},
}


def _dataset(root: Path, year: int) -> dict:
    root.mkdir(exist_ok=True)
    metadata = root / f"wb2_{year}.json"
    if not metadata.exists():
        metadata.write_text(
            json.dumps({"T": 2, "n_channels": 1, "H": 2, "W": 2})
        )
        (np.arange(8, dtype=np.float32) + year).tofile(
            root / f"wb2_{year}.bin"
        )
    return memmap_dataset_provenance(root, [year])


def _write_field(
    root: Path,
    *,
    model: str,
    year: int,
    rmse: float,
    acc: float,
    checkpoint: str,
    dataset_root: Path,
) -> None:
    root.mkdir(exist_ok=True)
    window_root = root / "window_metrics"
    window_root.mkdir(exist_ok=True)
    dates = np.arange(24, dtype=np.int64) * 8 * 24
    t0 = np.repeat(dates, 2)
    tau = np.tile(np.asarray([2, 4], dtype=np.int16), len(dates))
    n = len(t0)
    np.savez(
        window_root / f"{model}.npz",
        year=np.full(n, year, dtype=np.int16),
        t0=t0,
        tau=tau,
        mse_norm_model=np.full((n, 1), rmse**2),
        mse_norm_bilinear=np.ones((n, 1)),
        channel_names=np.asarray(["t2m"]),
        acc_model=np.full((n, 1), acc),
        acc_bilinear=np.zeros((n, 1)),
        acc_channel_names=np.asarray(["t2m"]),
    )

    def method(value: float, acc_value: float) -> dict[str, float]:
        return {
            "rmse_norm_t2m": value,
            "acc_mean": acc_value,
            "physical_wind_divergence_nmse": value,
            "physical_wind_vorticity_nmse": value,
            "physical_kinetic_energy_nmse": value,
            "physical_hydrostatic_balance_mse": value,
        }

    per_tau = {
        str(hour): {
            "model": method(rmse, acc),
            "bilinear": method(1.0, 0.0),
        }
        for hour in (1, 2, 4)
    }
    (root / f"{model}.json").write_text(
        json.dumps(
            {
                "checkpoint_provenance": {"sha256": checkpoint},
                "evaluation_protocol": {"full_year": True},
                "channel_names": ["t2m"],
                "seen_tau": [1],
                "unseen_tau": [2, 4],
                "per_tau": per_tau,
                "window_metrics_file": f"window_metrics/{model}.npz",
                "evaluation_input_provenance": INPUT_PROVENANCE,
                "evaluation_dataset_provenance": _dataset(
                    dataset_root,
                    year,
                ),
                "num_samples": n,
            }
        )
    )


def _write_spectra(
    root: Path,
    *,
    model: str,
    energy_ratio: float,
    shape_error: float,
    coherence: float,
    checkpoint: str,
    dataset_root: Path,
) -> None:
    root.mkdir(exist_ok=True)
    ell = np.arange(360, dtype=np.int16)
    dates = np.arange(24, dtype=np.int64) * 8 * 24
    dataset = _dataset(dataset_root, 2020)
    for tau in (2, 4):
        np.savez(
            root / f"{model}_tau{tau}.npz",
            ell=ell,
            pred_El=np.full((1, len(ell)), energy_ratio),
            gt_El=np.ones((1, len(ell))),
            n_samples=np.asarray(len(dates)),
            tau=np.asarray(tau),
            channel_names=np.asarray(["t2m"]),
            window_year=np.full(len(dates), 2020, dtype=np.int16),
            window_t0=dates,
            window_hf_energy_ratio=np.full(
                (len(dates), 1),
                energy_ratio,
            ),
            window_hf_log_shape_error=np.full(
                (len(dates), 1),
                shape_error,
            ),
            window_hf_coherence=np.full(
                (len(dates), 1),
                coherence,
            ),
            window_hf_signed_cospectrum=np.full(
                (len(dates), 1),
                coherence,
            ),
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "evaluation_input_provenance": INPUT_PROVENANCE,
                        "evaluation_dataset_provenance": dataset,
                        "checkpoint_provenance": {
                            "sha256": checkpoint,
                        },
                    }
                )
            ),
        )


def _fixture(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    roots = {
        name: tmp_path / name
        for name in (
            "primary20",
            "primary21",
            "nohf20",
            "nohf21",
            "primary_spec",
            "nohf_spec",
        )
    }
    dataset_root = tmp_path / "memmaps"
    candidate = "upr_implicit_global_14m"
    nohf = f"{candidate}_nohf"
    reference = "weatherbridge_ref"
    checkpoints = {
        candidate: "a" * 64,
        nohf: "b" * 64,
        reference: "c" * 64,
    }
    for year, primary, nohf_root in (
        (2020, roots["primary20"], roots["nohf20"]),
        (2021, roots["primary21"], roots["nohf21"]),
    ):
        _write_field(
            primary,
            model=candidate,
            year=year,
            rmse=0.40,
            acc=0.95,
            checkpoint=checkpoints[candidate],
            dataset_root=dataset_root,
        )
        _write_field(
            nohf_root,
            model=nohf,
            year=year,
            rmse=0.50,
            acc=0.90,
            checkpoint=checkpoints[nohf],
            dataset_root=dataset_root,
        )
        _write_field(
            primary,
            model=reference,
            year=year,
            rmse=0.80,
            acc=0.50,
            checkpoint=checkpoints[reference],
            dataset_root=dataset_root,
        )
    for model, root, energy, shape, coherence in (
        (candidate, roots["primary_spec"], 1.0, 0.02, 0.95),
        (nohf, roots["nohf_spec"], 1.0, 0.04, 0.90),
        (reference, roots["primary_spec"], 1.4, 0.20, 0.50),
    ):
        _write_spectra(
            root,
            model=model,
            energy_ratio=energy,
            shape_error=shape,
            coherence=coherence,
            checkpoint=checkpoints[model],
            dataset_root=dataset_root,
        )
    selection = {
        "schema_version": 6,
        "ood_attached_at_selection_time": False,
        "winner": candidate,
        "eligible": [candidate, reference],
        "models": {
            candidate: {
                "checkpoint_sha256": checkpoints[candidate],
                "quality_mean_rank": 1.0,
            },
            reference: {
                "checkpoint_sha256": checkpoints[reference],
                "quality_mean_rank": 2.0,
            },
        },
    }
    return selection, roots


def _build(selection: dict, roots: dict[str, Path]) -> dict:
    return build_report(
        selection,
        primary_root_2020=roots["primary20"],
        primary_root_2021=roots["primary21"],
        nohf_root_2020=roots["nohf20"],
        nohf_root_2021=roots["nohf21"],
        primary_spectra_root=roots["primary_spec"],
        nohf_spectra_root=roots["nohf_spec"],
        draws=1000,
        seed=11,
    )


def test_confirms_architecture_gain_separately_from_auxiliary(
    tmp_path: Path,
) -> None:
    selection, roots = _fixture(tmp_path)

    report = _build(selection, roots)

    assert report["schema_version"] == 2
    assert report["training_seed_count"] == 1
    assert (
        report["inference_scope"]
        == "single_seed_conditional_on_checkpoint"
    )
    assert report["architecture_only_single_seed_promotion_passed"]
    assert report["architecture_superiority_seed_consistent"] is None
    assert report["auxiliary_effect_single_seed_supported"]
    assert report["protocol"]["holm_family_size_per_comparison"] == 10
    assert all(
        result["significant_improvement"]
        for result in report["significance"]["architecture_only"]
    )


def test_rejects_nohf_checkpoint_change_between_years(
    tmp_path: Path,
) -> None:
    selection, roots = _fixture(tmp_path)
    path = roots["nohf21"] / "upr_implicit_global_14m_nohf.json"
    payload = json.loads(path.read_text())
    payload["checkpoint_provenance"]["sha256"] = "d" * 64
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="checkpoint changed"):
        _build(selection, roots)
