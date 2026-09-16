from __future__ import annotations

import numpy as np
import pytest

from tools.eval.spectral_block_bootstrap import (
    SpectralWindows,
    compare,
    load_vector_windows,
    load_windows,
)


def test_spectral_block_bootstrap_detects_better_left() -> None:
    year = np.full(20, 2020, dtype=np.int16)
    t0 = np.arange(20, dtype=np.int32) * 24
    channels = ("u10", "V850")
    left = SpectralWindows(
        year,
        t0,
        np.full((20, 2), 0.95),
        np.full((20, 2), 0.05),
        np.full((20, 2), 0.90),
        np.full((20, 2), 0.85),
        channels,
        8,
    )
    right = SpectralWindows(
        year,
        t0,
        np.full((20, 2), 0.70),
        np.full((20, 2), 0.30),
        np.full((20, 2), 0.60),
        np.full((20, 2), 0.50),
        channels,
        8,
    )

    result = compare(
        left,
        right,
        channel_indices=np.arange(2),
        block_days=2,
        draws=200,
        seed=3,
    )

    assert result["energy_log_error"]["delta_ci95"][2] < 0.0
    assert result["shape_log_error"]["delta_ci95"][2] < 0.0
    assert result["coherence"]["delta_ci95"][0] > 0.0
    assert result["signed_cospectrum"]["delta_ci95"][0] > 0.0
    assert result["energy_log_error"]["p_paired_block_permutation"] < 0.05
    assert result["shape_log_error"]["p_paired_block_permutation"] < 0.05
    assert result["coherence"]["p_paired_block_permutation"] < 0.05
    assert result["signed_cospectrum"]["p_paired_block_permutation"] < 0.05


def test_cellwise_spectral_gate_reports_every_channel() -> None:
    year = np.full(20, 2020, dtype=np.int16)
    t0 = np.arange(20, dtype=np.int32) * 24
    channels = ("u10", "V850")
    left = SpectralWindows(
        year,
        t0,
        np.full((20, 2), 0.98),
        np.full((20, 2), 0.03),
        np.full((20, 2), 0.95),
        np.full((20, 2), 0.90),
        channels,
        2,
    )
    right = SpectralWindows(
        year,
        t0,
        np.full((20, 2), 0.70),
        np.full((20, 2), 0.30),
        np.full((20, 2), 0.60),
        np.full((20, 2), 0.50),
        channels,
        2,
    )

    result = compare(
        left,
        right,
        channel_indices=np.arange(2),
        block_days=2,
        draws=200,
        seed=3,
        cellwise=True,
    )

    assert set(result["per_channel"]) == set(channels)
    for metric in (
        "energy_log_error",
        "shape_log_error",
        "coherence",
        "signed_cospectrum",
    ):
        family = result["cellwise_family"][metric]
        assert family["n_pointwise_left_better"] == 2
        assert family["all_pointwise_left_better"]
        assert family["all_left_better_holm"]


def test_spectral_loader_rejects_nonfinite_values(tmp_path) -> None:
    path = tmp_path / "invalid.npz"
    np.savez(
        path,
        window_year=np.asarray([2020, 2020], dtype=np.int16),
        window_t0=np.asarray([0, 24], dtype=np.int32),
        window_hf_energy_ratio=np.asarray([[1.0], [np.nan]]),
        window_hf_log_shape_error=np.asarray([[0.1], [0.1]]),
        window_hf_coherence=np.asarray([[0.8], [0.8]]),
        window_hf_signed_cospectrum=np.asarray([[0.7], [0.7]]),
        channel_names=np.asarray(["t2m"]),
        tau=np.asarray(2),
    )

    with pytest.raises(ValueError, match="spectral values"):
        load_windows(path)


def test_spectral_loader_rejects_duplicate_window_index(tmp_path) -> None:
    path = tmp_path / "duplicate.npz"
    shape = (2, 1)
    np.savez(
        path,
        window_year=np.asarray([2020, 2020], dtype=np.int16),
        window_t0=np.asarray([24, 24], dtype=np.int32),
        window_hf_energy_ratio=np.ones(shape),
        window_hf_log_shape_error=np.zeros(shape),
        window_hf_coherence=np.full(shape, 0.8),
        window_hf_signed_cospectrum=np.full(shape, 0.7),
        channel_names=np.asarray(["t2m"]),
        tau=np.asarray(2),
    )

    with pytest.raises(ValueError, match="duplicate spectral window index"):
        load_windows(path)


def test_vector_loader_flattens_wind_pairs_and_components(tmp_path) -> None:
    path = tmp_path / "vector.npz"
    shape = (3, 2, 2)
    np.savez(
        path,
        window_year=np.full(3, 2020, dtype=np.int16),
        window_t0=np.arange(3, dtype=np.int32) * 24,
        window_vector_hf_energy_ratio=np.ones(shape),
        window_vector_hf_log_shape_error=np.zeros(shape),
        window_vector_hf_coherence=np.full(shape, 0.8),
        window_vector_hf_signed_cospectrum=np.full(shape, 0.7),
        wind_pair_names=np.asarray(["wind850", "wind10"]),
        vector_component_names=np.asarray(["spheroidal", "toroidal"]),
        tau=np.asarray(2),
    )

    loaded = load_vector_windows(path)

    assert loaded.channels == (
        "wind850:spheroidal",
        "wind850:toroidal",
        "wind10:spheroidal",
        "wind10:toroidal",
    )
    assert loaded.energy_ratio.shape == (3, 4)
