from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.eval.select_journal_champion import (
    CANDIDATES,
    METRIC_FILES,
    PARAMETERS_M,
    SEEDS,
    SPECTRAL_METRICS,
    SPECTRAL_NAMES,
    _collect_file_records,
    _ifs_mean,
    _load_aux_pairwise,
    _seed_stem,
    _validate_completion_marker,
    select_champion,
)
from tools.repro.materialize_completion_source_snapshot import (
    materialize_snapshot,
)

CHANNELS = [f"c{index}" for index in range(24)]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _array_index_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _write_ifs_paired(path: Path, horizon: int) -> None:
    n_windows = {6: 3200, 12: 6864}[horizon]
    n_pairs = n_windows // (horizon - 1)
    init = np.repeat(np.arange(n_pairs, dtype=np.int64), horizon - 1)
    lead = np.repeat(np.arange(n_pairs, dtype=np.int16), horizon - 1)
    tau = np.tile(np.arange(1, horizon, dtype=np.int8), n_pairs)
    np.savez(
        path,
        init_time_hours=init,
        anchor_lead_hours=lead,
        tau_hours=tau,
        squared_error_norm=np.ones((n_windows, 24), dtype=np.float32),
        valid_channel=np.ones((n_windows, 24), dtype=np.bool_),
        channel_names=np.asarray(CHANNELS),
    )


def _bind_inputs(
    payload: dict,
    left_path: Path,
    reference: str,
    right_path: Path,
) -> dict:
    payload.update(
        left_sha256=_sha256(left_path),
        right_sha256={reference: _sha256(right_path)},
    )
    return payload


def _metric_payload(
    horizon: int,
    year: int,
    value: float,
    checkpoint: Path,
) -> dict:
    per_tau = {}
    for tau in range(1, horizon):
        model = {f"rmse_norm_{channel}": value for channel in CHANNELS}
        model.update({f"acc_{channel}": 0.9 for channel in CHANNELS})
        per_tau[str(tau)] = {"model": model}
    return {
        "years": [year],
        "delta_t_hours": float(horizon),
        "channel_names": CHANNELS,
        "evaluation_protocol": {
            "full_year": True,
            "latitude_grid": "wb2_0p25_2x2_block_average_v1",
            "eval_hours": list(range(1, horizon)),
            "index_sha256": f"index-{horizon}-{year}",
        },
        "checkpoint_provenance": _file_record(checkpoint),
        "per_tau": per_tau,
    }


def _rmse_pair(candidate: str, reference: str, horizon: int, year: int) -> dict:
    per_channel_tau = {
        str(tau): {channel: {} for channel in CHANNELS}
        for tau in range(1, horizon)
    }
    return {
        "schema_version": 1,
        "left": candidate,
        "index_sha256": f"index-{horizon}-{year}",
        "comparisons": {
            reference: {
                "per_channel_tau": per_channel_tau,
                "cellwise_family": {
                    "correction": "Holm-Bonferroni",
                    "alpha": 0.05,
                    "n_hypotheses": 24 * (horizon - 1),
                    "n_significant_left_worse_holm": 0,
                    "all_pointwise_left_better": True,
                },
            }
        },
    }


def _acc_pair(
    candidate: str,
    reference: str,
    horizon: int,
    year: int,
) -> dict:
    return {
        "schema_version": 2,
        "metric": "acc",
        "left": candidate,
        "index_sha256": f"index-{horizon}-{year}",
        "comparisons": {
            reference: {
                "delta_ci95": [0.01, 0.02, 0.03],
                "per_channel_tau": {
                    str(tau): {channel: {} for channel in CHANNELS}
                    for tau in range(1, horizon)
                },
                "cellwise_family": {
                    "correction": "Holm-Bonferroni",
                    "alpha": 0.05,
                    "n_hypotheses": 24 * (horizon - 1),
                    "n_significant_left_better_holm": 24 * (horizon - 1),
                    "n_significant_left_worse_holm": 0,
                    "all_pointwise_left_better": True,
                },
            }
        },
    }


def _hard_pair(
    candidate: str,
    reference: str,
    horizon: int,
    year: int,
    *,
    relative_delta_pct: float,
) -> dict:
    return {
        "schema_version": 1,
        "metric": "hard_window_rmse",
        "left": candidate,
        "index_sha256": f"index-{horizon}-{year}",
        "selection_is_model_independent": True,
        "quantile": 0.95,
        "comparisons": {
            reference: {
                "relative_delta_pct": relative_delta_pct,
                "per_channel_tau": {
                    str(tau): {channel: {} for channel in CHANNELS}
                    for tau in range(1, horizon)
                },
                "cellwise_family": {
                    "correction": "Holm-Bonferroni",
                    "alpha": 0.05,
                    "n_hypotheses": 24 * (horizon - 1),
                    "n_significant_left_worse_holm": 0,
                    "all_pointwise_left_better": relative_delta_pct < 0.0,
                },
            }
        },
    }


def _temporal_pair(
    candidate: str,
    reference: str,
    horizon: int,
    year: int,
) -> dict:
    return {
        "schema_version": 2,
        "metric": "temporal_curvature",
        "left": candidate,
        "index_sha256": f"temporal-index-{horizon}-{year}",
        "comparisons": {
            reference: {
                "delta_ci95": [-0.03, -0.02, -0.01],
                "per_channel_tau": {
                    "0": {channel: {} for channel in CHANNELS}
                },
                "cellwise_family": {
                    "correction": "Holm-Bonferroni",
                    "alpha": 0.05,
                    "n_hypotheses": 24,
                    "n_significant_left_better_holm": 24,
                    "n_significant_left_worse_holm": 0,
                    "n_pointwise_left_better": 24,
                    "all_pointwise_left_better": True,
                },
            }
        },
    }


def _physical_pair(
    candidate: str,
    reference: str,
    horizon: int | None = None,
    year: int | None = None,
) -> dict:
    payload = {
        "schema_version": 2,
        "metric": "physical",
        "left": candidate,
        "comparisons": {
            reference: {
                "diagnostics": {
                    "wind_divergence_nmse": {
                        "delta_left_minus_right": -0.1,
                        "p_paired_block_permutation": 0.01,
                    },
                    "hydrostatic_balance_mse": {
                        "delta_left_minus_right": -0.1,
                        "p_paired_block_permutation": 0.01,
                    },
                }
            }
        },
    }
    if horizon is not None and year is not None:
        payload["index_sha256"] = f"index-{horizon}-{year}"
    return payload


def _spectral(
    year: int,
    horizon: int,
    input_sha256: dict[str, str],
) -> dict:
    cell_count = 24 * (horizon - 1)
    return {
        "schema_version": 2,
        "reference": "weatherdcae_14m",
        "years": [year],
        "taus": list(range(1, horizon)),
        "yearly": {
            str(year): {
                metric: {
                    "wins": cell_count,
                    "cell_count": cell_count,
                    "failures": [],
                    "multiplicity_family": {
                        "method": "Holm-Bonferroni",
                        "dimensions": "tau_x_channel",
                        "n_hypotheses": cell_count,
                        "alpha": 0.05,
                    },
                }
                for metric in SPECTRAL_METRICS
            }
        },
        "input_sha256": input_sha256,
    }


def _ifs_payload(
    horizon: int,
    value: float,
    *,
    evaluator: Path,
    source: Path,
    model: Path,
    paired: Path,
) -> dict:
    with np.load(paired, allow_pickle=False) as arrays:
        index_sha256 = _array_index_sha256(
            arrays["init_time_hours"],
            arrays["anchor_lead_hours"],
            arrays["tau_hours"],
        )
    return {
        "schema_version": 2,
        "protocol": {
            "delta_t_hours": horizon,
            "latitude_grid": "wb2_0p25_2x2_block_average_v1",
            "area_weighting": "spherical_latitude_strip_area",
            "max_inits": 16,
        },
        "per_tau": {
            str(tau): {
                "model": {
                    **{
                        f"rmse_norm_{channel}": value
                        for channel in CHANNELS
                    },
                    "n_pairs_any": 10,
                }
            }
            for tau in range(1, horizon)
        },
        "provenance": {
            "evaluator": _file_record(evaluator),
            "evaluation_code": {"source.py": _file_record(source)},
            "model": {"artifact": _file_record(model)},
            "forecast_anchors": {"init_count": 16},
        },
        "paired_artifact": {
            **_file_record(paired),
            "window_index_sha256": index_sha256,
            "n_windows": {6: 3200, 12: 6864}[horizon],
        },
    }


def _tree(
    tmp_path: Path,
    *,
    detail_seed_complete: bool,
    candidate_worse: bool = False,
) -> tuple[Path, ...]:
    detailed = tmp_path / "detailed"
    spectra = tmp_path / "spectra"
    seeds = tmp_path / "seeds"
    marker_source = tmp_path / "marker_source.py"
    marker_source.write_text("# frozen selector fixture\n")
    ifs_evaluator = tmp_path / "ifs_evaluator.py"
    ifs_source = tmp_path / "ifs_source.py"
    ifs_model = tmp_path / "ifs_model.ckpt"
    ifs_evaluator.write_text("# evaluator fixture\n")
    ifs_source.write_text("# source fixture\n")
    ifs_model.write_bytes(b"model-fixture")
    for marker in (detailed / "6h/.complete", detailed / "12h/.complete"):
        _write_json(
            marker,
            {
                "status": "complete",
                "source_sha256": {
                    str(marker_source): _sha256(marker_source)
                },
            },
        )
    _write_json(
        spectra / "state/.complete",
        {
            "status": "complete",
            "source_files_sha256": {
                str(marker_source): _sha256(marker_source)
            },
        },
    )

    values = {
        "weatherbridge_detail": 0.80,
        "refine": 0.85,
        "flow_spectral": 0.75 if detail_seed_complete else 0.90,
    }
    if candidate_worse:
        values = {candidate: 1.1 for candidate in CANDIDATES}
    checkpoints: dict[int, dict[str, Path]] = {}
    for horizon in (6, 12):
        checkpoints[horizon] = {}
        for name in (*CANDIDATES, "dcae"):
            checkpoint = tmp_path / "checkpoints" / f"{name}_{horizon}h.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{name}-{horizon}h-checkpoint".encode())
            checkpoints[horizon][name] = checkpoint
    for horizon in (6, 12):
        ifs_paired = tmp_path / f"ifs_windows_{horizon}.npz"
        _write_ifs_paired(ifs_paired, horizon)
        config = METRIC_FILES[horizon]
        for year in (2020, 2021):
            full = detailed / f"{horizon}h/{year}/full_year"
            window_dir = full / "window_metrics"
            window_dir.mkdir(parents=True, exist_ok=True)
            reference_window = window_dir / f"{config['reference_name']}.npz"
            reference_window.write_bytes(
                f"reference-{horizon}-{year}".encode()
            )
            _write_json(
                full / str(config["reference"]),
                _metric_payload(
                    horizon,
                    year,
                    1.0,
                    checkpoints[horizon]["dcae"],
                ),
            )
            for candidate in CANDIDATES:
                candidate_window = window_dir / f"{candidate}.npz"
                candidate_window.write_bytes(
                    f"{candidate}-{horizon}-{year}".encode()
                )
                _write_json(
                    full / str(config[candidate]),
                    _metric_payload(
                        horizon,
                        year,
                        values[candidate],
                        checkpoints[horizon][candidate],
                    ),
                )
                _write_json(
                    full / f"paired_rmse_{candidate}.json",
                    _bind_inputs(
                        _rmse_pair(
                            candidate,
                            str(config["reference_name"]),
                            horizon,
                            year,
                        ),
                        candidate_window,
                        str(config["reference_name"]),
                        reference_window,
                    ),
                )
                _write_json(
                    full / f"paired_acc_{candidate}.json",
                    _bind_inputs(
                        _acc_pair(
                            candidate,
                            str(config["reference_name"]),
                            horizon,
                            year,
                        ),
                        candidate_window,
                        str(config["reference_name"]),
                        reference_window,
                    ),
                )
                _write_json(
                    full / f"paired_hard_window_{candidate}.json",
                    _bind_inputs(
                        _hard_pair(
                            candidate,
                            str(config["reference_name"]),
                            horizon,
                            year,
                            relative_delta_pct=100.0
                            * (values[candidate] - 1.0),
                        ),
                        candidate_window,
                        str(config["reference_name"]),
                        reference_window,
                    ),
                )
                _write_json(
                    full / f"paired_temporal_curvature_{candidate}.json",
                    _bind_inputs(
                        _temporal_pair(
                            candidate,
                            str(config["reference_name"]),
                            horizon,
                            year,
                        ),
                        candidate_window,
                        str(config["reference_name"]),
                        reference_window,
                    ),
                )
                _write_json(
                    full / f"paired_physical_{candidate}.json",
                    _bind_inputs(
                        _physical_pair(
                            candidate,
                            str(config["reference_name"]),
                            horizon,
                            year,
                        ),
                        candidate_window,
                        str(config["reference_name"]),
                        reference_window,
                    ),
                )
                for kind in ("scalar", "vector"):
                    spectrum_dir = spectra / f"{horizon}h_{year}"
                    source_hashes = {}
                    for tau in range(1, horizon):
                        left_source = spectrum_dir / (
                            f"fixture_{kind}_{candidate}_left_tau{tau}.npz"
                        )
                        right_source = spectrum_dir / (
                            f"fixture_{kind}_dcae_right_tau{tau}.npz"
                        )
                        left_source.parent.mkdir(parents=True, exist_ok=True)
                        left_source.write_bytes(
                            f"{kind}-{candidate}-{tau}".encode()
                        )
                        if not right_source.exists():
                            right_source.write_bytes(
                                f"{kind}-dcae-{tau}".encode()
                            )
                        pair_path = spectrum_dir / (
                            f"fixture_{kind}_{candidate}_tau{tau}.json"
                        )
                        _write_json(
                            pair_path,
                            {
                                "left_path": str(left_source),
                                "left_sha256": _sha256(left_source),
                                "right_paths": {
                                    "weatherdcae_14m": str(right_source)
                                },
                                "right_sha256": {
                                    "weatherdcae_14m": _sha256(right_source)
                                },
                            },
                        )
                        source_hashes[str(pair_path)] = _sha256(pair_path)
                    _write_json(
                        spectrum_dir
                        / (
                            f"global_{kind}_{SPECTRAL_NAMES[candidate]}_"
                            "vs_weatherdcae_14m.json"
                        ),
                        _spectral(year, horizon, source_hashes),
                    )

        for seed in SEEDS:
            _write_json(
                seeds / f"{horizon}h" / (_seed_stem("dcae", horizon, seed) + ".json"),
                _ifs_payload(
                    horizon,
                    1.0,
                    evaluator=ifs_evaluator,
                    source=ifs_source,
                    model=ifs_model,
                    paired=ifs_paired,
                ),
            )
            for candidate in CANDIDATES:
                if candidate == "weatherbridge_detail" and not detail_seed_complete:
                    continue
                _write_json(
                    seeds
                    / f"{horizon}h"
                    / (_seed_stem(candidate, horizon, seed) + ".json"),
                    _ifs_payload(
                        horizon,
                        values[candidate],
                        evaluator=ifs_evaluator,
                        source=ifs_source,
                        model=ifs_model,
                        paired=ifs_paired,
                    ),
                )

    cost = tmp_path / "cost.json"
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_source = repo_root / "tools/eval/benchmark_capmatched_inference.py"
    supporting_paths = (
        repo_root / "tools/eval/capmatched_loader.py",
        repo_root / "tools/train/train_capacity_matched_6h.py",
        repo_root / "weather_time_interp/model/weatherbridge_flow_model.py",
        repo_root / "weather_time_interp/model/dcae_adaln_model.py",
    )
    static_features = tmp_path / "static_features.pt"
    static_features.write_bytes(b"static-feature-fixture")
    cost_models = {}
    for index, name in enumerate((*CANDIDATES, "dcae")):
        checkpoint = checkpoints[6][name]
        cost_models[name] = {
            "params_m": PARAMETERS_M[name],
            "latency_ms": 9.0 if name == "dcae" else 10.0 + index,
            "batch_size": 1,
            "height": 360,
            "width": 720,
            "warmup": 5,
            "iterations": 20,
            "repeats": 7,
            "input_seed": 2027,
            "tau_values": [0.25, 0.5, 0.75],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
        }
    _write_json(
        cost,
        {
            "schema_version": 2,
            "device": "NVIDIA A100-SXM4-80GB",
            "input_seed": 2027,
            "tau_values": [0.25, 0.5, 0.75],
            "static_features": _file_record(static_features),
            "evaluation_script_sha256": _sha256(benchmark_source),
            "supporting_code_sha256": {
                str(path.relative_to(repo_root)): _sha256(path)
                for path in supporting_paths
            },
            "models": cost_models,
        },
    )
    return detailed, spectra, seeds, cost


def test_selector_confirms_three_seed_winner(tmp_path: Path) -> None:
    tree = _tree(tmp_path, detail_seed_complete=True)
    result = select_champion(*tree)

    assert result["status"] == "confirmed"
    assert result["winner"] == "flow_spectral"
    assert result["action"] == "promote"
    assert result["strict_dominance_winner"] == "flow_spectral"
    raw_window = (
        tree[0] / "6h/2020/full_year/window_metrics/flow_spectral.npz"
    )
    assert result["input_sha256"][str(raw_window)] == _sha256(raw_window)
    ifs_source = tmp_path / "ifs_source.py"
    assert result["input_sha256"][str(ifs_source)] == _sha256(ifs_source)


def test_selector_requires_detail_confirmation(tmp_path: Path) -> None:
    result = select_champion(*_tree(tmp_path, detail_seed_complete=False))

    assert result["preliminary_winner"] == "weatherbridge_detail"
    assert result["winner"] is None
    assert result["status"] == "confirmation_required"
    assert result["action"] == "train_detail_confirmation"


def test_selector_retains_dcae_when_all_candidates_regress(tmp_path: Path) -> None:
    result = select_champion(
        *_tree(
            tmp_path,
            detail_seed_complete=True,
            candidate_worse=True,
        )
    )

    assert result["status"] == "reference_retained"
    assert result["action"] == "promote_reference"
    assert result["winner"] == "weatherdcae_14m"


def test_selector_uses_next_candidate_when_metric_leader_fails_ood(
    tmp_path: Path,
) -> None:
    tree = _tree(tmp_path, detail_seed_complete=True)
    detailed = tree[0]
    for horizon in (6, 12):
        path = detailed / f"{horizon}h/2021/full_year/flow_spectral.json"
        _write_json(
            path,
            _metric_payload(
                horizon,
                2021,
                1.1,
                tmp_path / "checkpoints" / f"flow_spectral_{horizon}h.ckpt",
            ),
        )

    result = select_champion(*tree)

    assert "flow_spectral" not in result["ood_eligible_candidates"]
    assert result["winner"] == "weatherbridge_detail"


def test_selector_rejects_stale_bootstrap_source(tmp_path: Path) -> None:
    tree = _tree(tmp_path, detail_seed_complete=True)
    window = (
        tree[0]
        / "6h/2020/full_year/window_metrics/flow_spectral.npz"
    )
    window.write_bytes(b"changed-after-bootstrap")

    with pytest.raises(ValueError, match="stale left window metric"):
        select_champion(*tree)


def test_selector_rejects_stale_cost_checkpoint(tmp_path: Path) -> None:
    tree = _tree(tmp_path, detail_seed_complete=True)
    checkpoint = tmp_path / "checkpoints/flow_spectral_6h.ckpt"
    checkpoint.write_bytes(b"changed-after-cost-benchmark")

    with pytest.raises(ValueError, match="stale flow_spectral inference checkpoint"):
        select_champion(*tree)


def test_selector_rejects_partial_ifs_seed_artifact(tmp_path: Path) -> None:
    tree = _tree(tmp_path, detail_seed_complete=True)
    path = tree[2] / "6h" / (
        _seed_stem("flow_spectral", 6, SEEDS[0]) + ".json"
    )
    payload = json.loads(path.read_text())
    payload["provenance"]["forecast_anchors"]["init_count"] = 15
    _write_json(path, payload)

    with pytest.raises(ValueError, match="does not bind 16 inits"):
        select_champion(*tree)


def test_selector_rejects_incomplete_ifs_window_index(tmp_path: Path) -> None:
    tree = _tree(tmp_path, detail_seed_complete=True)
    path = tree[2] / "12h" / (
        _seed_stem("flow_spectral", 12, SEEDS[0]) + ".json"
    )
    payload = json.loads(path.read_text())
    payload["paired_artifact"]["n_windows"] -= 1
    _write_json(path, payload)

    with pytest.raises(ValueError, match="expected 6864 paired IFS window-hours"):
        select_champion(*tree)


def test_selector_rejects_wrong_measured_parameter_count(tmp_path: Path) -> None:
    tree = _tree(tmp_path, detail_seed_complete=True)
    cost = tree[3]
    payload = json.loads(cost.read_text())
    payload["models"]["flow_spectral"]["params_m"] += 0.01
    _write_json(cost, payload)

    with pytest.raises(ValueError, match="flow_spectral parameter count mismatch"):
        select_champion(*tree)


def test_selector_rejects_incomplete_field_hour_family(tmp_path: Path) -> None:
    tree = _tree(tmp_path, detail_seed_complete=True)
    path = tree[0] / "6h/2020/full_year/paired_rmse_flow_spectral.json"
    payload = json.loads(path.read_text())
    del payload["comparisons"]["weatherdcae_14m_6yr"][
        "per_channel_tau"
    ]["1"][CHANNELS[-1]]
    _write_json(path, payload)

    with pytest.raises(ValueError, match="incomplete cellwise field grid"):
        select_champion(*tree)


def test_selector_rejects_mismatched_auxiliary_index(tmp_path: Path) -> None:
    tree = _tree(tmp_path, detail_seed_complete=True)
    path = tree[0] / "6h/2020/full_year/paired_acc_flow_spectral.json"
    payload = json.loads(path.read_text())
    payload["index_sha256"] = "wrong-index"
    _write_json(path, payload)

    with pytest.raises(ValueError, match="ACC window index mismatch"):
        select_champion(*tree)


def test_selector_rejects_stale_completion_source(tmp_path: Path) -> None:
    tree = _tree(tmp_path, detail_seed_complete=True)
    (tmp_path / "marker_source.py").write_text("# changed\n")

    with pytest.raises(ValueError, match="stale source file"):
        select_champion(*tree)


def test_completion_marker_resolves_source_from_producer_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "producer"
    source = producer / "tools/evaluator.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen evaluator\n")
    marker = producer / "metrics/report/6h/.complete"
    _write_json(
        marker,
        {
            "status": "complete",
            "source_sha256": {"tools/evaluator.py": _sha256(source)},
        },
    )
    monkeypatch.chdir(tmp_path)

    resolved = _validate_completion_marker(marker)

    assert resolved == {str(source.resolve()): _sha256(source)}


def test_completion_marker_uses_materialized_source_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive/source_v1"
    source = archive / "tools/evaluator.py"
    source.parent.mkdir(parents=True)
    source.write_text("# archived evaluator\n")
    marker = tmp_path / "outputs/report/6h/.complete"
    _write_json(
        marker,
        {
            "status": "complete",
            "source_sha256": {"tools/evaluator.py": _sha256(source)},
        },
    )

    manifest = materialize_snapshot(marker, tmp_path / "archive")
    source.write_text("# archive later changed\n")
    rerun_manifest = materialize_snapshot(marker, tmp_path / "archive")
    monkeypatch.chdir(tmp_path)
    resolved = _validate_completion_marker(marker)

    snapshot = marker.parent / "source_snapshot/tools/evaluator.py"
    assert manifest["source_files"]["tools/evaluator.py"]["sha256"] == _sha256(
        snapshot
    )
    assert rerun_manifest["source_files"]["tools/evaluator.py"][
        "discovered_from"
    ] == str(snapshot.resolve())
    assert resolved == {str(snapshot.resolve()): _sha256(snapshot)}


def test_nested_provenance_collector_accepts_structured_fingerprints(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    used: dict[str, str] = {}
    payload = {
        "file": _file_record(source),
        "directory": {
            "path": str(tmp_path),
            "metadata_sha256": "directory-fingerprint",
        },
        "detached_file_hash": {"sha256": "hash-with-sibling-path"},
    }

    _collect_file_records(payload, used=used, context="fixture")

    assert used == {str(source): _sha256(source)}


def test_ifs_metric_rejects_stale_evaluation_source(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator.py"
    source = tmp_path / "source.py"
    model = tmp_path / "model.ckpt"
    paired = tmp_path / "paired.npz"
    for path in (evaluator, source, model):
        path.write_bytes(path.name.encode())
    _write_ifs_paired(paired, 6)
    metric = tmp_path / "ifs.json"
    _write_json(
        metric,
        _ifs_payload(
            6,
            0.8,
            evaluator=evaluator,
            source=source,
            model=model,
            paired=paired,
        ),
    )
    source.write_text("changed after evaluation")

    with pytest.raises(ValueError, match="stale provenance file"):
        _ifs_mean(metric, 6)


def test_physical_auxiliary_gate_uses_holm_correction(tmp_path: Path) -> None:
    path = tmp_path / "physical.json"
    payload = _physical_pair("candidate", "dcae")
    diagnostics = payload["comparisons"]["dcae"]["diagnostics"]
    diagnostics["wind_divergence_nmse"] = {
        "delta_left_minus_right": 0.2,
        "p_paired_block_permutation": 0.01,
    }
    diagnostics["hydrostatic_balance_mse"] = {
        "delta_left_minus_right": 0.1,
        "p_paired_block_permutation": 0.02,
    }
    _write_json(path, payload)

    result = _load_aux_pairwise(
        path,
        left="candidate",
        reference="dcae",
        metric="physical",
    )

    assert result["significant_regressions_holm"] == 2
