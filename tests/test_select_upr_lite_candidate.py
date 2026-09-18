import json
from pathlib import Path

import numpy as np
import pytest

from tools.eval.benchmark_capmatched_inference import (
    _hash_required_sources,
)
from tools.eval.select_upr_lite_candidate import (
    _file_sha256,
    _rank,
    _validate_cost_artifact,
    build_report,
    load_spectral_metrics,
    markdown_report,
    select,
)
from tools.train.training_protocol import memmap_dataset_provenance

SYNTHETIC_COMMON_PROVENANCE = {
    "static_features": {"sha256": "static-v1"},
    "pressure_level_stats": {"sha256": "pressure-v1"},
    "surface_stats": {"sha256": "surface-v1"},
}


def test_benchmark_writer_rejects_missing_supporting_source(tmp_path) -> None:
    existing = tmp_path / "existing.py"
    existing.write_text("value = 1\n")

    with pytest.raises(FileNotFoundError, match="missing.py"):
        _hash_required_sources(
            (existing, tmp_path / "missing.py"),
            tmp_path,
        )


def test_peak_memory_breaks_otherwise_equal_efficiency_tie() -> None:
    common = {
        "rmse_2020_unseen": 0.1,
        "acc_2020_unseen": 0.9,
        "hf_log_energy_error": 0.1,
        "hf_log_shape_error": 0.1,
        "hf_coherence": 0.9,
        "physical_2020_unseen_max": 1.0,
        "physical_2020_all_max": 1.0,
        "field_2020_all_bilinear_ratio_max": 1.0,
        "rmse_2020_per_tau": {1: 0.1, 2: 0.1},
        "tail_rmse_2020_unseen": 0.2,
        "tail_skill_2020_unseen": 0.1,
        "worst_season_rmse_2020_unseen": 0.2,
        "worst_season_skill_2020_unseen": 0.1,
        "params_m": 14.0,
        "latency_ms": 50.0,
        "skill_gap_2020": 0.0,
        "skill_2020_unseen": 0.1,
        "hf_log_energy_error_max": 0.2,
        "hf_log_shape_error_p95": 0.2,
        "hf_coherence_p05": 0.8,
    }
    rows = {
        "low_memory": {**common, "peak_memory_mib": 1000.0},
        "high_memory": {**common, "peak_memory_mib": 2000.0},
    }

    winner, eligible, _ = select(rows)

    assert winner == "low_memory"
    assert eligible == ["low_memory", "high_memory"]
    assert (
        rows["low_memory"]["efficiency_mean_rank"]
        < rows["high_memory"]["efficiency_mean_rank"]
    )


def test_ineligible_arm_cannot_change_eligible_winner() -> None:
    def row(
        *,
        rmse: float,
        acc: float,
        energy: float,
        shape: float,
        coherence: float,
        physical: float,
        field: float,
        tail: float,
        season: float,
        skill: float = 0.1,
        tail_skill: float = 0.1,
        season_skill: float = 0.1,
    ) -> dict:
        return {
            "rmse_2020_unseen": rmse,
            "acc_2020_unseen": acc,
            "hf_log_energy_error": energy,
            "hf_log_shape_error": shape,
            "hf_coherence": coherence,
            "physical_2020_all_max": physical,
            "field_2020_all_bilinear_ratio_max": field,
            "tail_rmse_2020_unseen": tail,
            "tail_skill_2020_unseen": tail_skill,
            "worst_season_rmse_2020_unseen": season,
            "worst_season_skill_2020_unseen": season_skill,
            "rmse_2020_per_tau": {1: rmse, 2: rmse},
            "skill_gap_2020": 0.0,
            "skill_2020_unseen": skill,
            "hf_log_energy_error_max": energy,
            "hf_log_shape_error_p95": shape,
            "hf_coherence_p05": coherence,
            "params_m": 1.0,
            "latency_ms": 1.0,
            "peak_memory_mib": 1.0,
        }

    def eligible_rows() -> dict[str, dict]:
        return {
            "a": row(
                rmse=0.100,
                acc=0.92,
                energy=0.10,
                shape=0.10,
                coherence=0.90,
                physical=1.00,
                field=1.00,
                tail=0.12,
                season=0.12,
            ),
            "b": row(
                rmse=0.102,
                acc=0.90,
                energy=0.12,
                shape=0.12,
                coherence=0.88,
                physical=0.80,
                field=0.80,
                tail=0.10,
                season=0.10,
            ),
        }

    rows = eligible_rows()
    winner, eligible, _ = select(rows)
    assert winner == "a"
    assert eligible == ["a", "b"]

    rows = {
        **eligible_rows(),
        "ineligible_decoy": row(
            rmse=0.103,
            acc=0.88,
            energy=0.14,
            shape=0.14,
            coherence=0.86,
            physical=0.90,
            field=0.90,
            tail=0.11,
            season=0.11,
            skill=-0.1,
        ),
    }
    winner, eligible, diagnostics = select(rows)

    assert winner == "a"
    assert eligible == ["a", "b"]
    assert diagnostics["winner_ranking_population"] == (
        "final_eligible_models_only"
    )
    assert (
        rows["b"]["quality_mean_rank"]
        < rows["a"]["quality_mean_rank"]
    )
    assert (
        rows["a"]["selection_quality_mean_rank"]
        < rows["b"]["selection_quality_mean_rank"]
    )
    assert (
        rows["ineligible_decoy"]["selection_quality_mean_rank"]
        is None
    )

    rows = {
        "safe": row(
            rmse=0.10,
            acc=0.90,
            energy=0.10,
            shape=0.10,
            coherence=0.90,
            physical=1.00,
            field=1.00,
            tail=0.10,
            season=0.10,
        ),
        "unsafe_low_rmse": row(
            rmse=0.05,
            acc=0.99,
            energy=0.05,
            shape=0.05,
            coherence=0.99,
            physical=1.00,
            field=1.00,
            tail=0.05,
            season=0.05,
            skill=-0.1,
        ),
    }
    winner, eligible, diagnostics = select(rows)

    assert winner == "safe"
    assert eligible == ["safe"]
    assert diagnostics["best_rmse_2020_unseen"] == pytest.approx(0.10)
    assert diagnostics["rmse_benchmark_population"] == (
        "absolute_safety_eligible_only"
    )

    rows = {
        "robust": row(
            rmse=0.10,
            acc=0.90,
            energy=0.10,
            shape=0.10,
            coherence=0.90,
            physical=1.00,
            field=1.00,
            tail=0.10,
            season=0.10,
        ),
        "tail_regression": row(
            rmse=0.05,
            acc=0.99,
            energy=0.05,
            shape=0.05,
            coherence=0.99,
            physical=1.00,
            field=1.00,
            tail=0.20,
            season=0.20,
            tail_skill=-0.01,
            season_skill=-0.01,
        ),
    }
    winner, eligible, diagnostics = select(rows)

    assert winner == "robust"
    assert eligible == ["robust"]
    assert diagnostics["robust_skill_relative_to_linear_min"] == 0.0
    assert rows["tail_regression"]["absolute_safety_gate_passed"] is False

    rows = {
        "rmse_best": row(
            rmse=0.10,
            acc=0.88,
            energy=0.12,
            shape=0.12,
            coherence=0.88,
            physical=1.00,
            field=1.00,
            tail=0.12,
            season=0.12,
        ),
        "secondary_star": row(
            rmse=0.102,
            acc=0.92,
            energy=0.10,
            shape=0.10,
            coherence=0.90,
            physical=0.80,
            field=0.80,
            tail=0.10,
            season=0.10,
        ),
    }
    winner, eligible, diagnostics = select(rows)

    assert eligible == ["rmse_best", "secondary_star"]
    assert (
        rows["secondary_star"]["selection_quality_mean_rank"]
        < rows["rmse_best"]["selection_quality_mean_rank"]
    )
    assert winner == "rmse_best"
    assert diagnostics["quality_shortlist"] == ["rmse_best"]
    assert diagnostics[
        "primary_rmse_practical_equivalence_margin"
    ] == pytest.approx(0.005)


def test_validate_cost_artifact_binds_code_and_static(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    benchmark = repo_root / "tools/eval/benchmark_capmatched_inference.py"
    supporting_paths = (
        "tools/eval/capmatched_loader.py",
        "tools/train/train_capacity_matched_6h.py",
        "weather_time_interp/model/weatherbridge_flow_model.py",
        "weather_time_interp/model/dcae_adaln_model.py",
        "weather_time_interp/model/dcae_adaln_skip_model.py",
        "legacy/scripts/train_atm_vfi_12h_oddskip.py",
    )
    static_path = tmp_path / "static.pt"
    static_path.write_bytes(b"static")
    checkpoint_path = tmp_path / "candidate.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    payload = {
        "schema_version": 2,
        "device": "NVIDIA A100-SXM4-80GB",
        "device_capability": [8, 0],
        "input_seed": 2027,
        "tau_values": [0.5],
        "evaluation_script_sha256": _file_sha256(benchmark),
        "supporting_code_sha256": {
            relative: _file_sha256(repo_root / relative)
            for relative in supporting_paths
        },
        "static_features": {
            "path": str(static_path),
            "size_bytes": static_path.stat().st_size,
            "sha256": _file_sha256(static_path),
        },
        "models": {
            "candidate": {
                "params_m": 14.0,
                "batch_size": 1,
                "height": 360,
                "width": 720,
                "warmup": 3,
                "iterations": 10,
                "repeats": 5,
                "latency_ms": 10.0,
                "latency_mean_ms": 10.0,
                "latency_std_ms": 0.0,
                "latency_p95_ms": 10.0,
                "repeat_latency_ms": [10.0] * 5,
                "samples_per_second": 100.0,
                "peak_memory_mib": 1024.0,
                "input_seed": 2027,
                "tau_values": [0.5],
                "tau_mode": "constant",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
            }
        },
    }
    cost_path = tmp_path / "cost.json"
    cost_path.write_text(json.dumps(payload))

    assert _validate_cost_artifact(cost_path, ("candidate",)) == payload

    payload["static_features"]["size_bytes"] = None
    cost_path.write_text(json.dumps(payload))
    with pytest.raises(
        ValueError,
        match="benchmark static provenance mismatch",
    ):
        _validate_cost_artifact(cost_path, ("candidate",))


def test_validate_cost_artifact_rejects_unstable_latency(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    benchmark = repo_root / "tools/eval/benchmark_capmatched_inference.py"
    supporting_paths = (
        "tools/eval/capmatched_loader.py",
        "tools/train/train_capacity_matched_6h.py",
        "weather_time_interp/model/weatherbridge_flow_model.py",
        "weather_time_interp/model/dcae_adaln_model.py",
        "weather_time_interp/model/dcae_adaln_skip_model.py",
        "legacy/scripts/train_atm_vfi_12h_oddskip.py",
    )
    static_path = tmp_path / "static.pt"
    static_path.write_bytes(b"static")
    checkpoint_path = tmp_path / "candidate.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    repeats = np.asarray([5.0, 5.0, 5.0, 5.0, 15.0])
    payload = {
        "schema_version": 2,
        "device": "NVIDIA A100-SXM4-80GB",
        "device_capability": [8, 0],
        "input_seed": 2027,
        "tau_values": [0.5],
        "evaluation_script_sha256": _file_sha256(benchmark),
        "supporting_code_sha256": {
            relative: _file_sha256(repo_root / relative)
            for relative in supporting_paths
        },
        "static_features": {
            "path": str(static_path),
            "size_bytes": static_path.stat().st_size,
            "sha256": _file_sha256(static_path),
        },
        "models": {
            "candidate": {
                "params_m": 14.0,
                "batch_size": 1,
                "height": 360,
                "width": 720,
                "warmup": 3,
                "iterations": 10,
                "repeats": len(repeats),
                "latency_ms": float(np.median(repeats)),
                "latency_mean_ms": float(np.mean(repeats)),
                "latency_std_ms": float(np.std(repeats)),
                "latency_p95_ms": float(np.quantile(repeats, 0.95)),
                "repeat_latency_ms": repeats.tolist(),
                "samples_per_second": 1000.0 / float(np.median(repeats)),
                "peak_memory_mib": 1024.0,
                "input_seed": 2027,
                "tau_values": [0.5],
                "tau_mode": "constant",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
            }
        },
    }
    cost_path = tmp_path / "cost.json"
    cost_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unstable benchmark latency"):
        _validate_cost_artifact(cost_path, ("candidate",))


def _dataset_provenance(root: Path, year: int) -> dict:
    root.mkdir(exist_ok=True)
    metadata = root / f"wb2_{year}.json"
    data = root / f"wb2_{year}.bin"
    if not metadata.exists():
        metadata.write_text(
            json.dumps({"T": 2, "n_channels": 1, "H": 2, "W": 2})
        )
        (np.arange(8, dtype=np.float32) + year).tofile(data)
    return memmap_dataset_provenance(root, [year])


def _write_metrics(
    path: Path,
    rmse: float,
    acc: float,
    physical_values: tuple[float, float, float, float] | None = None,
    rmse_per_tau: dict[int, float] | None = None,
) -> None:
    year = int(path.parent.name)
    dataset_provenance = _dataset_provenance(
        path.parents[1] / "memmaps",
        year,
    )
    window_dir = path.parent / "window_metrics"
    window_dir.mkdir(exist_ok=True)
    window_path = window_dir / f"{path.stem}.npz"
    tau_values = np.tile(np.array([1, 2]), 4)
    rmse_by_tau = {1: rmse, 2: rmse}
    if rmse_per_tau is not None:
        rmse_by_tau.update(rmse_per_tau)
    np.savez(
        window_path,
        year=np.full(8, year),
        t0=np.arange(8) * 24 * 45,
        tau=tau_values,
        mse_norm_model=np.asarray(
            [rmse_by_tau[int(tau)] ** 2 for tau in tau_values],
            dtype=np.float64,
        )[:, None],
        mse_norm_bilinear=np.ones((8, 1)),
    )
    physical = physical_values or (rmse, rmse, rmse, rmse)
    payload = {
        "checkpoint_provenance": {
            "sha256": f"{path.stem}-checkpoint",
        },
        "num_samples": 8,
        "channel_names": ["t2m"],
        "seen_tau": [1],
        "unseen_tau": [2],
        "per_tau": {
            str(tau): {
                "model": {
                    "rmse_norm_t2m": rmse_by_tau[tau],
                    "acc_mean": acc,
                    "physical_wind_divergence_nmse": physical[0],
                    "physical_wind_vorticity_nmse": physical[1],
                    "physical_kinetic_energy_nmse": physical[2],
                    "physical_hydrostatic_balance_mse": physical[3],
                },
                "bilinear": {
                    "rmse_norm_t2m": 1.0,
                    "acc_mean": 0.0,
                    "physical_wind_divergence_nmse": 1.0,
                    "physical_wind_vorticity_nmse": 1.0,
                    "physical_kinetic_energy_nmse": 1.0,
                    "physical_hydrostatic_balance_mse": 1.0,
                },
            }
            for tau in (1, 2)
        },
        "window_metrics_file": f"window_metrics/{path.stem}.npz",
        "evaluation_protocol": {
            "full_year": True,
            "index_sha256": "synthetic-full-year",
        },
        "evaluation_input_provenance": {
            **SYNTHETIC_COMMON_PROVENANCE,
            "climatology": {"cache_identity_sha256": "climatology-v1"},
        },
        "evaluation_dataset_provenance": dataset_provenance,
    }
    path.write_text(json.dumps(payload))


def _write_spectrum(
    path: Path,
    energy_scale: float,
    coherence: float,
    tau: int = 1,
    dataset_root: Path | None = None,
    spectrum_values: tuple[float, float, float] | None = None,
) -> None:
    dataset_provenance = _dataset_provenance(
        dataset_root or path.parent / "memmaps",
        2020,
    )
    model_name = path.stem.rsplit("_tau", 1)[0]
    truth = np.ones((1, 3), dtype=np.float64)
    pred = (
        truth * np.asarray(spectrum_values, dtype=np.float64)[None]
        if spectrum_values is not None
        else truth * energy_scale
    )
    shape_error = np.mean(
        np.abs(np.log(np.maximum(pred[:, 1:], 1e-30))),
        axis=1,
    )
    np.savez(
        path,
        ell=np.array([0, 90, 120]),
        pred_El=pred,
        gt_El=truth,
        n_samples=np.array(4),
        window_hf_log_shape_error=np.broadcast_to(
            shape_error,
            (4, 1),
        ),
        window_hf_coherence=np.full((4, 1), coherence),
        window_year=np.full(4, 2020),
        window_t0=np.arange(4) * 24,
        tau=np.array(tau),
        metadata_json=np.array(
            json.dumps(
                {
                    "evaluation_input_provenance": (
                        SYNTHETIC_COMMON_PROVENANCE
                    ),
                    "evaluation_dataset_provenance": dataset_provenance,
                    "checkpoint_provenance": {
                        "sha256": f"{model_name}-checkpoint",
                    },
                }
            )
        ),
    )


def test_selection_uses_hf_shape_not_only_total_energy(
    tmp_path: Path,
) -> None:
    root_2020 = tmp_path / "2020"
    spectra = tmp_path / "spectra"
    root_2020.mkdir()
    spectra.mkdir()
    for model in ("a_bad_shape", "z_good_shape"):
        _write_metrics(root_2020 / f"{model}.json", 0.4, 0.95)
    _write_spectrum(
        spectra / "a_bad_shape_tau2.npz",
        1.0,
        0.9,
        tau=2,
        dataset_root=tmp_path / "memmaps",
        spectrum_values=(1.0, 0.2, 1.8),
    )
    _write_spectrum(
        spectra / "z_good_shape_tau2.npz",
        1.0,
        0.9,
        tau=2,
        dataset_root=tmp_path / "memmaps",
        spectrum_values=(1.0, 1.0, 1.0),
    )

    report = build_report(
        ("a_bad_shape", "z_good_shape"),
        root_2020,
        None,
        spectra,
        hf_ell_min=90,
        spectral_taus={2},
    )

    assert report["models"]["a_bad_shape"]["hf_log_energy_error"] == pytest.approx(
        0.0
    )
    assert report["models"]["z_good_shape"]["hf_log_energy_error"] == pytest.approx(
        0.0
    )
    assert (
        report["models"]["a_bad_shape"]["hf_log_shape_error"]
        > report["models"]["z_good_shape"]["hf_log_shape_error"]
    )
    assert report["winner"] == "z_good_shape"


def test_rank_uses_order_independent_average_for_ties() -> None:
    assert _rank({"a": 1.0, "b": 1.0, "c": 2.0}) == {
        "a": 1.5,
        "b": 1.5,
        "c": 3.0,
    }
    assert _rank({"b": 1.0, "a": 1.0, "c": 2.0}, reverse=True) == {
        "c": 1.0,
        "b": 2.5,
        "a": 2.5,
    }


def test_build_report_selects_generalizing_spectral_candidate(
    tmp_path: Path,
) -> None:
    root_2020 = tmp_path / "2020"
    root_2021 = tmp_path / "2021"
    spectra = tmp_path / "spectra"
    root_2020.mkdir()
    root_2021.mkdir()
    spectra.mkdir()

    for model, rmse_2020, rmse_2021, acc, energy, coherence in (
        ("candidate_a", 0.4, 0.5, 0.95, 1.0, 0.9),
        ("candidate_b", 0.41, 0.8, 0.80, 0.5, 0.4),
    ):
        _write_metrics(root_2020 / f"{model}.json", rmse_2020, acc)
        _write_metrics(root_2021 / f"{model}.json", rmse_2021, acc)
        _write_spectrum(
            spectra / f"{model}_tau1.npz",
            energy,
            coherence,
            dataset_root=tmp_path / "memmaps",
        )
        _write_spectrum(
            spectra / f"{model}_tau2.npz",
            energy,
            coherence,
            tau=2,
            dataset_root=tmp_path / "memmaps",
        )

    report = build_report(
        ("candidate_a", "candidate_b"),
        root_2020,
        root_2021,
        spectra,
        hf_ell_min=90,
        model_params_m={"candidate_a": 10.0, "candidate_b": 0.1},
        model_costs={
            "candidate_a": {
                "latency_ms": 10.0,
                "peak_memory_mib": 1000.0,
            },
            "candidate_b": {
                "latency_ms": 1.0,
                "peak_memory_mib": 2000.0,
            },
        },
        spectral_taus={2},
    )

    assert report["winner"] == "candidate_a"
    assert report["efficiency_winner"] == "candidate_a"
    assert report["eligible"] == ["candidate_a"]
    assert report["models"]["candidate_b"]["field_gate_passed"] is True
    assert report["models"]["candidate_b"]["spectral_gate_passed"] is False
    assert report["selection_rule"]["gate_policy"] == "fail_closed"
    assert report["selection_rule"]["spectral_taus"] == [2]
    assert report["pareto_population"] == "final_eligible_models_only"
    assert report["pareto_front"] == ["candidate_a"]
    assert report["models"]["candidate_a"]["hf_coherence"] == 0.9
    assert report["models"]["candidate_a"]["peak_memory_mib"] == 1000.0
    assert (
        report["models"]["candidate_a"]["efficiency_mean_rank"]
        > report["models"]["candidate_b"]["efficiency_mean_rank"]
    )
    rendered = markdown_report(report)
    assert "Physics20 all max/Bilinear" in rendered
    assert "Quality winner: **candidate_a**" in rendered
    assert "Efficiency winner: **candidate_a**" in rendered
    candidate_a_row = next(
        line for line in rendered.splitlines() if line.startswith("| candidate_a |")
    )
    assert "| 0.4000 |" in candidate_a_row

    _write_metrics(root_2021 / "candidate_a.json", 0.99, -100.0)
    _write_metrics(root_2021 / "candidate_b.json", 0.01, 100.0)
    report_with_reversed_ood = build_report(
        ("candidate_a", "candidate_b"),
        root_2020,
        root_2021,
        spectra,
        hf_ell_min=90,
        model_params_m={"candidate_a": 10.0, "candidate_b": 0.1},
        model_costs={
            "candidate_a": {
                "latency_ms": 10.0,
                "peak_memory_mib": 1000.0,
            },
            "candidate_b": {
                "latency_ms": 1.0,
                "peak_memory_mib": 2000.0,
            },
        },
        spectral_taus={2},
    )

    assert report_with_reversed_ood["winner"] == report["winner"]
    assert (
        report_with_reversed_ood["efficiency_winner"]
        == report["efficiency_winner"]
    )
    assert report_with_reversed_ood["eligible"] == report["eligible"]
    assert report_with_reversed_ood["pareto_front"] == report["pareto_front"]
    assert (
        report_with_reversed_ood["models"]["candidate_b"]["rmse_2021_unseen"]
        < report_with_reversed_ood["models"]["candidate_a"]["rmse_2021_unseen"]
    )

    economy = json.loads((root_2020 / "candidate_a.json").read_text())
    economy["evaluation_protocol"]["full_year"] = False
    (root_2020 / "candidate_a.json").write_text(json.dumps(economy))
    with pytest.raises(ValueError, match="not full-year"):
        build_report(
            ("candidate_a", "candidate_b"),
            root_2020,
            root_2021,
            spectra,
            hf_ell_min=90,
            spectral_taus={2},
        )


def test_selection_can_be_frozen_without_ood_artifacts(
    tmp_path: Path,
) -> None:
    root_2020 = tmp_path / "2020"
    spectra = tmp_path / "spectra"
    root_2020.mkdir()
    spectra.mkdir()
    for model, rmse, acc in (
        ("candidate_a", 0.4, 0.95),
        ("candidate_b", 0.7, 0.80),
    ):
        _write_metrics(root_2020 / f"{model}.json", rmse, acc)
        _write_spectrum(
            spectra / f"{model}_tau2.npz",
            1.0,
            0.9,
            tau=2,
            dataset_root=tmp_path / "memmaps",
        )

    report = build_report(
        ("candidate_a", "candidate_b"),
        root_2020,
        None,
        spectra,
        hf_ell_min=90,
        spectral_taus={2},
    )

    assert report["winner"] == "candidate_a"
    assert report["efficiency_winner"] == "candidate_a"
    assert report["ood_attached_at_selection_time"] is False
    assert "field_2021" not in report["paired_window_index_sha256"]
    assert "field_dataset_2021" not in report["paired_window_index_sha256"]
    assert "rmse_2021_unseen" not in report["models"]["candidate_a"]
    assert (
        "before the selection process loaded any 2021 artifact"
        in markdown_report(report)
    )


def test_spectral_loader_rejects_missing_tau_and_truncated_grid(
    tmp_path: Path,
) -> None:
    _write_spectrum(tmp_path / "candidate_tau2.npz", 1.0, 0.9, tau=2)

    with pytest.raises(FileNotFoundError, match="missing=\\[4\\]"):
        load_spectral_metrics(
            tmp_path,
            "candidate",
            hf_ell_min=90,
            taus={2, 4},
        )
    with pytest.raises(ValueError, match="does not cover every degree"):
        load_spectral_metrics(
            tmp_path,
            "candidate",
            hf_ell_min=90,
            taus={2},
            required_lmax=359,
        )


def test_spectral_loader_rejects_out_of_range_coherence(
    tmp_path: Path,
) -> None:
    _write_spectrum(
        tmp_path / "invalid_tau2.npz",
        1.0,
        1.01,
        tau=2,
    )

    with pytest.raises(ValueError, match="invalid per-window HF coherence"):
        load_spectral_metrics(
            tmp_path,
            "invalid",
            hf_ell_min=90,
            taus={2},
        )


def test_selection_rejects_hidden_individual_physical_regression(
    tmp_path: Path,
) -> None:
    root_2020 = tmp_path / "2020"
    root_2021 = tmp_path / "2021"
    spectra = tmp_path / "spectra"
    root_2020.mkdir()
    root_2021.mkdir()
    spectra.mkdir()
    for year_root in (root_2020, root_2021):
        _write_metrics(
            year_root / "unsafe.json",
            0.40,
            0.95,
            physical_values=(1.20, 0.80, 0.80, 0.80),
        )
        _write_metrics(
            year_root / "safe.json",
            0.41,
            0.94,
            physical_values=(1.00, 1.00, 1.00, 1.00),
        )
    for model in ("unsafe", "safe"):
        _write_spectrum(
            spectra / f"{model}_tau2.npz",
            1.0,
            0.9,
            tau=2,
            dataset_root=tmp_path / "memmaps",
        )

    report = build_report(
        ("unsafe", "safe"),
        root_2020,
        root_2021,
        spectra,
        hf_ell_min=90,
        spectral_taus={2},
    )

    assert report["models"]["unsafe"]["physical_2020_unseen"] == pytest.approx(
        0.9
    )
    assert report["models"]["unsafe"]["physical_2020_unseen_max"] == 1.2
    assert report["eligible"] == ["safe"]
    assert report["winner"] == "safe"


def test_selection_fails_closed_when_no_model_is_field_safe(
    tmp_path: Path,
) -> None:
    root_2020 = tmp_path / "2020"
    spectra = tmp_path / "spectra"
    root_2020.mkdir()
    spectra.mkdir()
    for model in ("unsafe_a", "unsafe_b"):
        _write_metrics(
            root_2020 / f"{model}.json",
            0.40,
            0.95,
            physical_values=(1.20, 1.20, 1.20, 1.20),
        )
        _write_spectrum(
            spectra / f"{model}_tau2.npz",
            1.0,
            0.9,
            tau=2,
            dataset_root=tmp_path / "memmaps",
        )

    with pytest.raises(
        ValueError,
        match="no model passed the all-hour RMSE/physical safety gate",
    ):
        build_report(
            ("unsafe_a", "unsafe_b"),
            root_2020,
            None,
            spectra,
            hf_ell_min=90,
            spectral_taus={2},
        )


def test_selection_rejects_hidden_individual_hour_regression(
    tmp_path: Path,
) -> None:
    root_2020 = tmp_path / "2020"
    spectra = tmp_path / "spectra"
    root_2020.mkdir()
    spectra.mkdir()
    _write_metrics(
        root_2020 / "unsafe_edge.json",
        0.40,
        0.95,
        rmse_per_tau={1: 0.50, 2: 0.40},
    )
    _write_metrics(
        root_2020 / "balanced.json",
        0.41,
        0.94,
        rmse_per_tau={1: 0.40, 2: 0.41},
    )
    for model in ("unsafe_edge", "balanced"):
        _write_spectrum(
            spectra / f"{model}_tau2.npz",
            1.0,
            0.9,
            tau=2,
            dataset_root=tmp_path / "memmaps",
        )

    report = build_report(
        ("unsafe_edge", "balanced"),
        root_2020,
        None,
        spectra,
        hf_ell_min=90,
        spectral_taus={2},
    )

    assert report["models"]["unsafe_edge"]["field_gate_passed"] is False
    assert (
        report["models"]["unsafe_edge"][
            "worst_hour_rmse_ratio_to_best_2020"
        ]
        == 1.25
    )
    assert report["eligible"] == ["balanced"]
    assert report["winner"] == "balanced"


def test_selection_rejects_hidden_channel_regression(
    tmp_path: Path,
) -> None:
    root_2020 = tmp_path / "2020"
    spectra = tmp_path / "spectra"
    root_2020.mkdir()
    spectra.mkdir()
    for model, rmse, channel_values in (
        ("unsafe_channel", 0.70, (0.20, 1.20)),
        ("uniform", 0.72, (0.72, 0.72)),
    ):
        path = root_2020 / f"{model}.json"
        _write_metrics(path, rmse, 0.95)
        payload = json.loads(path.read_text())
        payload["channel_names"] = ["t2m", "u10"]
        for cell in payload["per_tau"].values():
            cell["model"]["rmse_norm_t2m"] = channel_values[0]
            cell["model"]["rmse_norm_u10"] = channel_values[1]
            cell["bilinear"]["rmse_norm_t2m"] = 1.0
            cell["bilinear"]["rmse_norm_u10"] = 1.0
        path.write_text(json.dumps(payload))
        _write_spectrum(
            spectra / f"{model}_tau2.npz",
            1.0,
            0.9,
            tau=2,
            dataset_root=tmp_path / "memmaps",
        )

    report = build_report(
        ("unsafe_channel", "uniform"),
        root_2020,
        None,
        spectra,
        hf_ell_min=90,
        spectral_taus={2},
    )

    assert report["models"]["unsafe_channel"]["field_gate_passed"] is False
    assert (
        report["models"]["unsafe_channel"][
            "field_2020_all_bilinear_ratio_max"
        ]
        == 1.2
    )
    assert report["eligible"] == ["uniform"]
    assert report["winner"] == "uniform"


def test_reports_field_hour_dominance_without_requiring_uniform_winner(
    tmp_path: Path,
) -> None:
    root_2020 = tmp_path / "2020"
    spectra = tmp_path / "spectra"
    root_2020.mkdir()
    spectra.mkdir()
    for model, channel_values in (
        ("temperature_specialist", (0.40, 0.60)),
        ("wind_specialist", (0.50, 0.50)),
    ):
        path = root_2020 / f"{model}.json"
        _write_metrics(path, 0.50, 0.95)
        payload = json.loads(path.read_text())
        payload["channel_names"] = ["t2m", "u10"]
        for cell in payload["per_tau"].values():
            cell["model"]["rmse_norm_t2m"] = channel_values[0]
            cell["model"]["rmse_norm_u10"] = channel_values[1]
            cell["bilinear"]["rmse_norm_t2m"] = 1.0
            cell["bilinear"]["rmse_norm_u10"] = 1.0
        path.write_text(json.dumps(payload))
        _write_spectrum(
            spectra / f"{model}_tau2.npz",
            1.0,
            0.9,
            tau=2,
            dataset_root=tmp_path / "memmaps",
        )

    report = build_report(
        ("temperature_specialist", "wind_specialist"),
        root_2020,
        None,
        spectra,
        hf_ell_min=90,
        spectral_taus={2},
    )

    diagnostic = report["selection_rule"]["gate_diagnostics"][
        "field_hour_dominance"
    ]
    assert diagnostic["active"] is True
    assert diagnostic["total_cells"] == 4
    assert diagnostic["uniform_winners"] == []
    assert (
        report["models"]["temperature_specialist"][
            "field_hour_win_fraction_2020"
        ]
        == 0.5
    )
    assert (
        report["models"]["temperature_specialist"][
            "field_hour_worst_ratio_to_best_2020"
        ]
        == pytest.approx(1.2)
    )
    assert (
        report["models"]["wind_specialist"][
            "field_hour_worst_ratio_to_best_2020"
        ]
        == pytest.approx(1.25)
    )


def test_build_report_rejects_input_provenance_mismatch(
    tmp_path: Path,
) -> None:
    root_2020 = tmp_path / "2020"
    root_2021 = tmp_path / "2021"
    spectra = tmp_path / "spectra"
    root_2020.mkdir()
    root_2021.mkdir()
    spectra.mkdir()
    for model in ("a", "b"):
        _write_metrics(root_2020 / f"{model}.json", 0.5, 0.9)
        _write_metrics(root_2021 / f"{model}.json", 0.5, 0.9)
        _write_spectrum(
            spectra / f"{model}_tau2.npz",
            1.0,
            0.9,
            tau=2,
            dataset_root=tmp_path / "memmaps",
        )
    payload = json.loads((root_2021 / "b.json").read_text())
    payload["evaluation_input_provenance"]["climatology"][
        "cache_identity_sha256"
    ] = "climatology-v2"
    (root_2021 / "b.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="2021 field input-provenance"):
        build_report(
            ("a", "b"),
            root_2020,
            root_2021,
            spectra,
            hf_ell_min=90,
            spectral_taus={2},
        )
