from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tools.eval.eval_artifact_status import _evaluation_source_paths, validate_eval_artifact
from tools.train.training_protocol import memmap_dataset_provenance
from weather_time_interp.metrics.physical_consistency import (
    DIAGNOSTIC_COMPONENTS,
)
from weather_time_interp.normalization import STATIC_FEATURES_3


def test_release_evaluation_dependencies_exist() -> None:
    from tools.eval.batch_eval_forecast_anchor import forecast_evaluation_source_paths

    assert all(path.is_file() for path in _evaluation_source_paths().values())
    assert all(path.is_file() for path in forecast_evaluation_source_paths().values())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(root: Path, checkpoint: Path) -> Path:
    source = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "eval"
        / "batch_eval_12h_memmap.py"
    )
    climatology_source = source.with_name("climatology.py")
    window_dir = root / "window_metrics"
    window_dir.mkdir()
    window_path = window_dir / "model.npz"
    year = np.asarray([2020] * 10, dtype=np.int16)
    t0 = np.arange(10, dtype=np.int32)
    tau = np.asarray([1, 2] * 5, dtype=np.int8)
    window_payload = {
        "year": year,
        "t0": t0,
        "tau": tau,
        "channel_names": np.asarray(["t2m"]),
        "acc_channel_names": np.asarray(["t2m"]),
        "mse_norm_model": np.full((10, 1), 0.01, dtype=np.float32),
        "mse_norm_bilinear": np.full((10, 1), 0.04, dtype=np.float32),
        "acc_model": np.full((10, 1), 0.9, dtype=np.float32),
        "acc_bilinear": np.full((10, 1), 0.8, dtype=np.float32),
        "temporal_year": np.full(5, 2020, dtype=np.int16),
        "temporal_t0": np.arange(5, dtype=np.int32),
        "temporal_center_count": np.full(5, 2, dtype=np.int8),
        "temporal_centers": np.asarray([1, 2], dtype=np.int8),
        "temporal_curvature_mse_model": np.full(
            (5, 1),
            0.01,
            dtype=np.float32,
        ),
        "temporal_curvature_mse_bilinear": np.full(
            (5, 1),
            0.04,
            dtype=np.float32,
        ),
    }
    for method in ("model", "bilinear"):
        for diagnostic, components in DIAGNOSTIC_COMPONENTS.items():
            window_payload[f"physical_{diagnostic}_{method}"] = np.full(
                (10, len(components)),
                0.1,
                dtype=np.float32,
            )
    np.savez_compressed(window_path, **window_payload)
    index_text = "\n".join(
        f"{int(y)},{int(start)},{int(hour)}"
        for y, start, hour in zip(year, t0, tau)
    )
    index_sha256 = hashlib.sha256(index_text.encode("utf-8")).hexdigest()
    temporal_index_text = "\n".join(
        f"2020,{start}" for start in range(5)
    )
    temporal_index_sha256 = hashlib.sha256(
        temporal_index_text.encode("utf-8")
    ).hexdigest()
    stat = checkpoint.stat()
    memmap_dir = root / "memmaps"
    memmap_dir.mkdir()
    (memmap_dir / "wb2_2020.json").write_text(
        json.dumps({"T": 2, "n_channels": 1, "H": 2, "W": 2})
    )
    np.arange(8, dtype=np.float32).tofile(memmap_dir / "wb2_2020.bin")
    payload = {
        "num_samples": 10,
        "delta_t_hours": 3.0,
        "years": [2020],
        "channel_names": ["t2m"],
        "n_per_tau": {"1": 5, "2": 5},
        "per_tau": {
            str(tau): {
                method: {
                    "rmse_norm_t2m": 0.1,
                    "rmse_phys_t2m": 1.0,
                    "acc_mean": 0.9,
                    "physical_wind_divergence_nmse": 0.1,
                    "physical_wind_vorticity_nmse": 0.1,
                    "physical_kinetic_energy_nmse": 0.1,
                    "physical_hydrostatic_balance_mse": 0.1,
                }
                for method in ("model", "bilinear")
            }
            for tau in (1, 2)
        },
        "acc_channel_names": ["t2m"],
        "evaluation_protocol": {
            "full_year": True,
            "eval_hours": [1, 2],
            "index_sha256": index_sha256,
            "rmse_reduction": "spherical_strip_area_weighted_spatial_mean",
            "save_physical_metrics": True,
            "save_temporal_metrics": True,
            "temporal_index_sha256": temporal_index_sha256,
            "temporal_centers": [1, 2],
        },
        "temporal_curvature_rmse_norm": {
            "model": 0.1,
            "bilinear": 0.2,
        },
        "checkpoint_provenance": {
            "path": str(checkpoint.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256(checkpoint),
        },
        "evaluation_dataset_provenance": memmap_dataset_provenance(
            memmap_dir,
            [2020],
        ),
        "evaluation_input_provenance": {
            "static_features": {
                "sha256": "static",
                "semantic": {
                    "cosine_latitude_grid": (
                        "legacy_symmetric_89p75_to_minus89p75"
                    ),
                    "offset_from_wb2_block_centres_degrees": -0.125,
                },
            },
            "pressure_level_stats": {"sha256": "pressure"},
            "surface_stats": {"sha256": "surface"},
            "climatology": {
                "cache_identity_sha256": "climatology",
                "semantic": {
                    "declared_window": "1990-2019",
                    "start_year": 1990,
                    "end_year": 2019,
                    "evaluation_years": [2020],
                    "precedes_evaluation": True,
                },
            },
        },
        "static_feature_names": list(STATIC_FEATURES_3),
        "evaluation_code_provenance": {
            "batch_eval_12h_memmap.py": _sha256(source),
            "climatology.py": _sha256(climatology_source),
        },
        "window_metrics_file": "window_metrics/model.npz",
        "window_metrics_provenance": {
            "size_bytes": window_path.stat().st_size,
            "sha256": _sha256(window_path),
            "index_sha256": index_sha256,
        },
    }
    artifact = root / "model.json"
    artifact.write_text(json.dumps(payload))
    return artifact


def _refresh_window_provenance(root: Path, payload: dict) -> None:
    window_path = root / payload["window_metrics_file"]
    payload["window_metrics_provenance"] = {
        "size_bytes": window_path.stat().st_size,
        "sha256": _sha256(window_path),
        "index_sha256": payload["evaluation_protocol"]["index_sha256"],
    }


def test_validates_complete_current_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    artifact = _write_artifact(tmp_path, checkpoint)

    result = validate_eval_artifact(
        artifact,
        checkpoint_path=checkpoint,
        required_taus=(1, 2),
        acc_mode="enabled",
        require_physical_metrics=True,
        require_temporal_metrics=True,
    )

    assert result["valid"]
    assert result["errors"] == []


def test_rejects_stale_checkpoint_and_missing_climatology(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    artifact = _write_artifact(tmp_path, checkpoint)
    checkpoint.write_bytes(b"new-weights")
    payload = json.loads(artifact.read_text())
    payload["evaluation_input_provenance"]["climatology"] = None
    artifact.write_text(json.dumps(payload))

    result = validate_eval_artifact(
        artifact,
        checkpoint_path=checkpoint,
        required_taus=(1, 2),
        acc_mode="enabled",
        require_physical_metrics=True,
    )

    assert not result["valid"]
    assert "checkpoint size mismatch" in result["errors"]
    assert "checkpoint content hash mismatch" in result["errors"]
    assert "missing climatology fingerprint" in result["errors"]


def test_rejects_climatology_with_missing_or_overlapping_semantics(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    artifact = _write_artifact(tmp_path, checkpoint)
    payload = json.loads(artifact.read_text())
    climatology = payload["evaluation_input_provenance"]["climatology"]
    climatology["semantic"] = {
        "declared_window": "1990-2020",
        "start_year": 1990,
        "end_year": 2020,
        "evaluation_years": [2020],
        "precedes_evaluation": True,
    }
    artifact.write_text(json.dumps(payload))

    result = validate_eval_artifact(
        artifact,
        checkpoint_path=checkpoint,
        required_taus=(1, 2),
        acc_mode="enabled",
        expected_climatology_window="1990-2020",
    )

    assert not result["valid"]
    assert "climatology overlaps evaluation period" in result["errors"]

    climatology.pop("semantic")
    artifact.write_text(json.dumps(payload))
    result = validate_eval_artifact(
        artifact,
        checkpoint_path=checkpoint,
        required_taus=(1, 2),
        acc_mode="enabled",
    )
    assert "missing climatology semantic provenance" in result["errors"]


def test_accepts_explicit_no_acc_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    artifact = _write_artifact(tmp_path, checkpoint)
    payload = json.loads(artifact.read_text())
    payload["evaluation_input_provenance"]["climatology"] = None
    payload["acc_channel_names"] = []
    window_path = tmp_path / payload["window_metrics_file"]
    with np.load(window_path, allow_pickle=False) as saved:
        windows = {name: np.asarray(saved[name]) for name in saved.files}
    windows["acc_channel_names"] = np.asarray([], dtype="U16")
    windows["acc_model"] = np.empty((10, 0), dtype=np.float32)
    windows["acc_bilinear"] = np.empty((10, 0), dtype=np.float32)
    np.savez_compressed(window_path, **windows)
    _refresh_window_provenance(tmp_path, payload)
    artifact.write_text(json.dumps(payload))

    result = validate_eval_artifact(
        artifact,
        checkpoint_path=checkpoint,
        required_taus=(1, 2),
        acc_mode="disabled",
        require_physical_metrics=True,
    )

    assert result["valid"]


def test_rejects_corrupt_windows_and_malformed_numeric_fields(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    artifact = _write_artifact(tmp_path, checkpoint)
    payload = json.loads(artifact.read_text())
    payload["evaluation_protocol"]["eval_hours"] = ["not-an-hour"]
    payload["num_samples"] = "not-a-count"
    payload["per_tau"] = ["not", "an", "object"]
    window_path = tmp_path / payload["window_metrics_file"]
    window_path.write_bytes(b"not-an-npz")
    artifact.write_text(json.dumps(payload))

    result = validate_eval_artifact(
        artifact,
        checkpoint_path=checkpoint,
        required_taus=(1, 2),
        acc_mode="enabled",
        require_physical_metrics=True,
    )

    assert not result["valid"]
    assert "evaluation eval_hours contains a non-integer value" in result[
        "errors"
    ]
    assert "num_samples is not an integer" in result["errors"]
    assert "per_tau must be an object" in result["errors"]
    assert any(
        error.startswith("cannot read window metrics:")
        for error in result["errors"]
    )


def test_rejects_changed_memmap_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    artifact = _write_artifact(tmp_path, checkpoint)
    payload = json.loads(artifact.read_text())
    memmap = payload["evaluation_dataset_provenance"]
    data_path = Path(memmap["files"]["2020"]["data_path"])
    changed = np.arange(8, dtype=np.float32)
    changed[0] = 99.0
    changed.tofile(data_path)

    result = validate_eval_artifact(
        artifact,
        checkpoint_path=checkpoint,
        required_taus=(1, 2),
        acc_mode="enabled",
        require_physical_metrics=True,
    )

    assert not result["valid"]
    assert "memmap dataset provenance mismatch" in result["errors"]


def test_rejects_stale_supporting_evaluation_code(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    artifact = _write_artifact(tmp_path, checkpoint)
    payload = json.loads(artifact.read_text())
    payload["evaluation_code_provenance"]["weather_amt_model.py"] = (
        "stale-model-code"
    )
    artifact.write_text(json.dumps(payload))

    result = validate_eval_artifact(
        artifact,
        checkpoint_path=checkpoint,
        required_taus=(1, 2),
        acc_mode="enabled",
        require_physical_metrics=True,
    )

    assert not result["valid"]
    assert "evaluation code hash mismatch: weather_amt_model.py" in result[
        "errors"
    ]


def test_rejects_nonfinite_window_metric_with_matching_hash(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    artifact = _write_artifact(tmp_path, checkpoint)
    payload = json.loads(artifact.read_text())
    window_path = tmp_path / payload["window_metrics_file"]
    with np.load(window_path, allow_pickle=False) as saved:
        windows = {name: np.asarray(saved[name]) for name in saved.files}
    windows["mse_norm_model"] = windows["mse_norm_model"].copy()
    windows["mse_norm_model"][0, 0] = np.nan
    np.savez_compressed(window_path, **windows)
    _refresh_window_provenance(tmp_path, payload)
    artifact.write_text(json.dumps(payload))

    result = validate_eval_artifact(
        artifact,
        checkpoint_path=checkpoint,
        required_taus=(1, 2),
        acc_mode="enabled",
        require_physical_metrics=True,
    )

    assert not result["valid"]
    assert "window metrics mse_norm_model is invalid" in result["errors"]
