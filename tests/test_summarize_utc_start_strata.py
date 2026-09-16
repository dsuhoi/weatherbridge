from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.eval.summarize_utc_start_strata import summarize_year


def _artifact(path: Path, scale: float) -> None:
    t0 = np.asarray([0, 6, 12, 18], dtype=np.int32)
    mse = np.full((4, 2), scale**2, dtype=np.float32)
    np.savez(
        path,
        year=np.full(4, 2020, dtype=np.int16),
        t0=t0,
        tau=np.ones(4, dtype=np.int8),
        channel_names=np.asarray(["a", "b"]),
        mse_norm_model=mse,
        mse_norm_bilinear=np.full((4, 2), 0.25, dtype=np.float32),
    )


def test_summarize_year_uses_paired_utc_strata(tmp_path: Path) -> None:
    weatherbridge = tmp_path / "weatherbridge.npz"
    dcae = tmp_path / "dcae.npz"
    _artifact(weatherbridge, 0.4)
    _artifact(dcae, 0.5)

    report = summarize_year(weatherbridge, dcae, 2020)

    assert report["rows"]["within_window"]["n_window_hours"] == 2
    assert report["rows"]["crosses_window_boundary"]["n_window_hours"] == 2
    assert report["rows"]["00"]["weatherbridge_rmse"] == pytest.approx(0.4)
    assert report["rows"]["00"]["weatherbridge_gain_vs_dcae_pct"] == pytest.approx(20.0)
