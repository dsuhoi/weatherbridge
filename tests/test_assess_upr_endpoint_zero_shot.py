from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tools.eval.assess_upr_endpoint_zero_shot import assess


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_eval(
    root: Path,
    name: str,
    checkpoint: Path,
    mse: np.ndarray,
) -> Path:
    year = np.full(mse.shape[0], 2020, dtype=np.int16)
    tau = np.tile(np.arange(1, 6, dtype=np.int8), mse.shape[0] // 5)
    t0 = np.repeat(
        np.arange(mse.shape[0] // 5, dtype=np.int32) * 15 * 24,
        5,
    )
    window_dir = root / "window_metrics"
    window_dir.mkdir(exist_ok=True)
    window_path = window_dir / f"{name}.npz"
    np.savez_compressed(
        window_path,
        year=year,
        t0=t0,
        tau=tau,
        mse_norm_model=mse,
        channel_names=np.asarray(["t2m", "z500"]),
    )
    index_text = "\n".join(
        f"{int(y)},{int(start)},{int(hour)}"
        for y, start, hour in zip(year, t0, tau)
    )
    payload = {
        "years": [2020],
        "evaluation_protocol": {
            "full_year": False,
            "rmse_reduction": "spherical_strip_area_weighted_spatial_mean",
            "eval_hours": [1, 2, 3, 4, 5],
            "index_sha256": hashlib.sha256(
                index_text.encode("utf-8")
            ).hexdigest(),
        },
        "checkpoint_provenance": {
            "path": str(checkpoint.resolve()),
            "sha256": _sha256(checkpoint),
        },
        "window_metrics_file": str(window_path.relative_to(root)),
    }
    path = root / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def test_endpoint_zero_shot_promotes_edge_gain_without_other_regression(
    tmp_path: Path,
) -> None:
    reference_checkpoint = tmp_path / "reference.ckpt"
    reference_checkpoint.write_bytes(b"reference weights")
    control_checkpoint = tmp_path / "control.ckpt"
    torch.save(
        {
            "architecture_conversion": {
                "kind": "zero_shot_state_compatible_control",
                "source_arch": "upr_implicit_global_14m",
                "target_arch": "upr_endpoint_implicit_global_14m",
                "source_checkpoint_sha256": _sha256(
                    reference_checkpoint
                ),
                "weights_unchanged": True,
            }
        },
        control_checkpoint,
    )
    reference_mse = np.full((120, 2), 4.0, dtype=np.float32)
    candidate_mse = reference_mse.copy()
    tau = np.tile(np.arange(1, 6), 24)
    candidate_mse[np.isin(tau, [1, 5])] = 1.0
    reference_json = _write_eval(
        tmp_path,
        "reference",
        reference_checkpoint,
        reference_mse,
    )
    candidate_json = _write_eval(
        tmp_path,
        "candidate",
        control_checkpoint,
        candidate_mse,
    )

    report = assess(
        reference_json,
        candidate_json,
        draws=500,
        seed=7,
    )

    assert report["state_identical_weights"]
    assert report["comparisons"]["edge"]["delta_left_minus_right"] < 0
    assert report["comparisons"]["held"]["delta_left_minus_right"] == 0
    assert report["promote_matched_retrain"]


def test_endpoint_zero_shot_rejects_unbound_control(tmp_path: Path) -> None:
    reference_checkpoint = tmp_path / "reference.ckpt"
    reference_checkpoint.write_bytes(b"reference weights")
    control_checkpoint = tmp_path / "control.ckpt"
    torch.save(
        {
            "architecture_conversion": {
                "kind": "zero_shot_state_compatible_control",
                "source_arch": "upr_implicit_global_14m",
                "target_arch": "upr_endpoint_implicit_global_14m",
                "source_checkpoint_sha256": "wrong",
                "weights_unchanged": True,
            }
        },
        control_checkpoint,
    )
    mse = np.ones((20, 2), dtype=np.float32)
    reference_json = _write_eval(
        tmp_path,
        "reference",
        reference_checkpoint,
        mse,
    )
    candidate_json = _write_eval(
        tmp_path,
        "candidate",
        control_checkpoint,
        mse,
    )

    try:
        assess(reference_json, candidate_json, draws=20, seed=7)
    except ValueError as error:
        assert "not bound" in str(error)
    else:
        raise AssertionError("unbound control was accepted")


def test_endpoint_zero_shot_rejects_stale_checkpoint(
    tmp_path: Path,
) -> None:
    reference_checkpoint = tmp_path / "reference.ckpt"
    reference_checkpoint.write_bytes(b"reference weights")
    control_checkpoint = tmp_path / "control.ckpt"
    torch.save(
        {
            "architecture_conversion": {
                "kind": "zero_shot_state_compatible_control",
                "source_arch": "upr_implicit_global_14m",
                "target_arch": "upr_endpoint_implicit_global_14m",
                "source_checkpoint_sha256": _sha256(
                    reference_checkpoint
                ),
                "weights_unchanged": True,
            }
        },
        control_checkpoint,
    )
    mse = np.ones((20, 2), dtype=np.float32)
    reference_json = _write_eval(
        tmp_path,
        "reference",
        reference_checkpoint,
        mse,
    )
    candidate_json = _write_eval(
        tmp_path,
        "candidate",
        control_checkpoint,
        mse,
    )
    control_checkpoint.write_bytes(b"mutated")

    try:
        assess(reference_json, candidate_json, draws=20, seed=7)
    except ValueError as error:
        assert "stale checkpoint provenance" in str(error)
    else:
        raise AssertionError("stale checkpoint was accepted")
