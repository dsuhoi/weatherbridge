from pathlib import Path

import numpy as np
import pytest

from tools.eval.physical_block_bootstrap import (
    compare_physical,
    load_physical_windows,
)
from weather_time_interp.metrics.physical_consistency import (
    GENERALIZATION_DIAGNOSTICS,
)


def _write(path: Path, scale: float) -> None:
    payload = {
        "year": np.full(12, 2021),
        "t0": np.arange(12) * 24,
        "tau": np.tile(np.array([2, 4]), 6),
    }
    for diagnostic in GENERALIZATION_DIAGNOSTICS:
        payload[f"physical_{diagnostic}_model"] = np.full((12, 3), scale)
    np.savez(path, **payload)


def test_physical_comparison_detects_consistent_improvement(
    tmp_path: Path,
) -> None:
    left_path = tmp_path / "left.npz"
    right_path = tmp_path / "right.npz"
    _write(left_path, 0.5)
    _write(right_path, 1.0)
    result = compare_physical(
        load_physical_windows(left_path),
        load_physical_windows(right_path),
        taus=np.asarray([2, 4]),
        block_days=2,
        draws=200,
        seed=7,
    )
    for diagnostic in result["diagnostics"].values():
        assert diagnostic["delta_left_minus_right"] == -0.5
        assert diagnostic["p_paired_block_permutation"] < 0.05
    assert set(result["diagnostics"]) == set(GENERALIZATION_DIAGNOSTICS)


def test_physical_comparison_rejects_missing_tau(tmp_path: Path) -> None:
    path = tmp_path / "only_tau2.npz"
    payload = {
        "year": np.full(12, 2021, dtype=np.int16),
        "t0": np.arange(12, dtype=np.int32) * 24,
        "tau": np.full(12, 2, dtype=np.int8),
    }
    for diagnostic in GENERALIZATION_DIAGNOSTICS:
        payload[f"physical_{diagnostic}_model"] = np.ones((12, 3))
    np.savez(path, **payload)
    windows = load_physical_windows(path)

    with pytest.raises(ValueError, match="requested taus"):
        compare_physical(
            windows,
            windows,
            taus=np.asarray([2, 4]),
            block_days=2,
            draws=20,
            seed=7,
        )
