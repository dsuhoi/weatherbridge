from __future__ import annotations

import numpy as np
import pytest

from tools.eval.hard_window_block_bootstrap import hard_window_comparisons


def test_hard_windows_are_selected_from_common_bilinear_error(tmp_path) -> None:
    n_dates = 140
    year = np.full(n_dates * 2, 2021, dtype=np.int16)
    t0 = np.repeat(np.arange(n_dates) * 24, 2)
    tau = np.tile(np.asarray([2, 4], dtype=np.int8), n_dates)
    hard = np.repeat(np.arange(n_dates) % 10 == 0, 2)
    baseline_rmse = np.where(hard, 10.0, 1.0)
    left_rmse = np.where(hard, 8.0, 0.8)
    right_rmse = np.where(hard, 4.0, 0.9)

    def mse(values: np.ndarray) -> np.ndarray:
        return np.repeat(values[:, None] ** 2, 2, axis=1)

    np.savez(
        tmp_path / "left.npz",
        year=year,
        t0=t0,
        tau=tau,
        mse_norm_model=mse(left_rmse),
        mse_norm_bilinear=mse(baseline_rmse),
        channel_names=np.asarray(["t2m", "u10"]),
    )
    np.savez(
        tmp_path / "right.npz",
        year=year,
        t0=t0,
        tau=tau,
        mse_norm_model=mse(right_rmse),
        channel_names=np.asarray(["t2m", "u10"]),
    )

    result = hard_window_comparisons(
        tmp_path / "left.npz",
        [("right", tmp_path / "right.npz")],
        quantile=0.95,
        taus=np.asarray([2, 4]),
        block_days=7,
        draws=1000,
        seed=9,
    )

    comparison = result["comparisons"]["right"]
    assert result["selection_is_model_independent"]
    assert result["n_windows"] == 28
    assert comparison["delta_left_minus_right"] == pytest.approx(4.0)
    assert comparison["cellwise_family"]["n_significant_left_worse_holm"] == 4


def test_hard_window_quantile_is_validated(tmp_path) -> None:
    with pytest.raises(ValueError, match="quantile"):
        hard_window_comparisons(
            tmp_path / "missing.npz",
            [],
            quantile=0.5,
            taus=None,
            block_days=7,
            draws=10,
            seed=1,
        )
