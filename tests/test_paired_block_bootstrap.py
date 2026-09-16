from __future__ import annotations

import numpy as np
import pytest

from tools.eval.paired_block_bootstrap import (
    WindowMetrics,
    WindowScores,
    load_metrics,
    load_temporal_metrics,
    paired_block_bootstrap,
    paired_block_score_test,
    paired_cellwise_score_tests,
    select_channels,
)


def test_paired_block_bootstrap_detects_consistent_improvement() -> None:
    year = np.full(24, 2020, dtype=np.int16)
    t0 = np.repeat(np.arange(0, 12 * 24, 24, dtype=np.int32), 2)
    tau = np.tile(np.array([1, 2], dtype=np.int8), 12)
    right_mse = np.full((24, 3), 4.0, dtype=np.float32)
    left_mse = np.full((24, 3), 1.0, dtype=np.float32)
    left = WindowMetrics(year, t0, tau, left_mse, ("a", "b", "c"))
    right = WindowMetrics(year, t0, tau, right_mse, ("a", "b", "c"))

    result = paired_block_bootstrap(
        left,
        right,
        taus=np.array([1, 2]),
        block_days=1,
        draws=200,
        seed=7,
        include_cellwise=True,
    )

    assert result["left_rmse"] == 1.0
    assert result["right_rmse"] == 2.0
    assert result["delta_ci95"][2] < 0.0
    assert 0.0 < result["p_paired_block_permutation"] < 0.05
    assert 0.0 < result["p_left_better_one_sided"] < 0.05
    assert result["p_left_worse_one_sided"] > 0.95
    assert set(result["per_tau"]) == {"1", "2"}
    assert all(
        0.0 < item["p_left_better_one_sided"] < 0.05
        for item in result["per_tau"].values()
    )
    assert result["cellwise_family"] == {
        "correction": "Holm-Bonferroni",
        "alpha": 0.05,
        "n_hypotheses": 6,
        "n_pointwise_left_better": 6,
        "n_pointwise_left_worse": 0,
        "n_significant_left_better_holm": 6,
        "n_significant_left_worse_holm": 0,
        "all_pointwise_left_better": True,
        "all_left_better_holm": True,
        "any_left_worse_holm": False,
    }
    assert set(result["per_channel_tau"]["1"]) == {"a", "b", "c"}
    assert all(
        cell["p_left_better_holm"] < 0.05
        for by_channel in result["per_channel_tau"].values()
        for cell in by_channel.values()
    )


def test_select_channels_preserves_requested_order() -> None:
    metrics = WindowMetrics(
        year=np.asarray([2020, 2020], dtype=np.int16),
        t0=np.asarray([0, 24], dtype=np.int32),
        tau=np.asarray([1, 2], dtype=np.int8),
        mse=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        channels=("t2m", "Q850", "mslp"),
    )

    selected = select_channels(metrics, ("Q850", "t2m"))

    assert selected.channels == ("Q850", "t2m")
    assert np.array_equal(selected.mse, metrics.mse[:, [1, 0]])
    with pytest.raises(ValueError, match="must be unique"):
        select_channels(metrics, ("Q850", "Q850"))
    with pytest.raises(ValueError, match="missing"):
        select_channels(metrics, ("Q700",))


def test_paired_block_score_test_detects_higher_acc() -> None:
    year = np.full(24, 2021, dtype=np.int16)
    t0 = np.repeat(np.arange(0, 12 * 24, 24, dtype=np.int32), 2)
    tau = np.tile(np.array([2, 4], dtype=np.int8), 12)
    left = WindowScores(
        year,
        t0,
        tau,
        np.full((24, 2), 0.9),
        ("z500", "t850"),
    )
    right = WindowScores(
        year,
        t0,
        tau,
        np.full((24, 2), 0.7),
        ("z500", "t850"),
    )

    result = paired_block_score_test(
        left,
        right,
        taus=np.array([2, 4]),
        block_days=1,
        draws=200,
        seed=7,
        better="higher",
    )

    assert result["metric"] == "mean_window_acc"
    assert result["delta_left_minus_right"] == pytest.approx(0.2)
    assert result["delta_ci95"][0] > 0.0
    assert result["p_paired_block_permutation"] < 0.05


def test_cellwise_score_tests_detect_one_regressing_acc_cell() -> None:
    year = np.full(48, 2021, dtype=np.int16)
    t0 = np.repeat(np.arange(0, 24 * 24, 24, dtype=np.int32), 2)
    tau = np.tile(np.array([2, 4], dtype=np.int8), 24)
    left_values = np.full((48, 2), 0.9)
    left_values[:, 1] = 0.6
    left = WindowScores(year, t0, tau, left_values, ("z500", "t850"))
    right = WindowScores(
        year,
        t0,
        tau,
        np.full((48, 2), 0.7),
        ("z500", "t850"),
    )

    result = paired_cellwise_score_tests(
        left,
        right,
        taus=np.array([2, 4]),
        block_days=1,
        draws=500,
        seed=7,
        better="higher",
    )

    family = result["cellwise_family"]
    assert family["n_hypotheses"] == 4
    assert family["n_significant_left_better_holm"] == 2
    assert family["n_significant_left_worse_holm"] == 2
    assert family["any_left_worse_holm"]


def test_rejects_missing_tau_and_invalid_mse(tmp_path) -> None:
    year = np.full(8, 2020, dtype=np.int16)
    t0 = np.arange(8, dtype=np.int32) * 24
    tau = np.ones(8, dtype=np.int8)
    mse = np.ones((8, 2), dtype=np.float32)
    left = WindowMetrics(year, t0, tau, mse, ("a", "b"))

    with pytest.raises(ValueError, match="requested taus"):
        paired_block_bootstrap(
            left,
            left,
            taus=np.asarray([1, 2]),
            block_days=1,
            draws=20,
            seed=7,
        )

    mse[0, 0] = np.nan
    path = tmp_path / "invalid.npz"
    np.savez(
        path,
        year=year,
        t0=t0,
        tau=tau,
        mse_norm_model=mse,
        channel_names=np.asarray(["a", "b"]),
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        load_metrics(path)


def test_rejects_duplicate_window_index(tmp_path) -> None:
    path = tmp_path / "duplicate.npz"
    np.savez(
        path,
        year=np.asarray([2020, 2020], dtype=np.int16),
        t0=np.asarray([24, 24], dtype=np.int32),
        tau=np.asarray([2, 2], dtype=np.int8),
        mse_norm_model=np.ones((2, 1), dtype=np.float32),
        channel_names=np.asarray(["t2m"]),
    )

    with pytest.raises(ValueError, match="duplicate .* window index"):
        load_metrics(path)


def test_redraws_bootstrap_sample_without_every_tau() -> None:
    year = np.full(2, 2020, dtype=np.int16)
    t0 = np.asarray([0, 24], dtype=np.int32)
    tau = np.asarray([1, 2], dtype=np.int8)
    metrics = WindowMetrics(
        year,
        t0,
        tau,
        np.ones((2, 1), dtype=np.float32),
        ("a",),
    )

    result = paired_block_bootstrap(
        metrics,
        metrics,
        taus=np.asarray([1, 2]),
        block_days=1,
        draws=100,
        seed=7,
    )

    assert result["left_rmse"] == result["right_rmse"] == 1.0
    assert result["delta_left_minus_right"] == 0.0
    assert result["bootstrap_rejected_draws_missing_tau"] > 0


def test_load_temporal_metrics_uses_one_row_per_anchor(tmp_path) -> None:
    path = tmp_path / "temporal.npz"
    np.savez(
        path,
        temporal_year=np.full(4, 2021, dtype=np.int16),
        temporal_t0=np.arange(4, dtype=np.int32) * 24,
        temporal_center_count=np.full(4, 5, dtype=np.int8),
        temporal_curvature_mse_model=np.full(
            (4, 2),
            0.25,
            dtype=np.float32,
        ),
        channel_names=np.asarray(["t2m", "z500"]),
    )

    metrics = load_temporal_metrics(path)

    assert metrics.channels == ("t2m", "z500")
    assert np.array_equal(metrics.tau, np.zeros(4, dtype=np.int8))
    assert np.all(metrics.mse == 0.25)


def test_load_temporal_metrics_rejects_duplicate_anchor(tmp_path) -> None:
    path = tmp_path / "duplicate.npz"
    np.savez(
        path,
        temporal_year=np.full(2, 2021, dtype=np.int16),
        temporal_t0=np.zeros(2, dtype=np.int32),
        temporal_center_count=np.full(2, 5, dtype=np.int8),
        temporal_curvature_mse_model=np.ones((2, 1), dtype=np.float32),
        channel_names=np.asarray(["t2m"]),
    )

    with pytest.raises(ValueError, match="duplicate .* window index"):
        load_temporal_metrics(path)
