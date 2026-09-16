from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.make_fig_fieldwise_scores import CHANNELS, load_horizon


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = [
    "Linear Interp.",
    "SwinV2",
    "ModAFNO",
    "S-DYff",
    "PixelAttn-VFI",
    "WeatherDCAE-14M",
    "WeatherBridge",
]


def test_fieldwise_scores_cover_every_reported_model_and_field() -> None:
    metrics_root = ROOT / "metrics" / "journal_unified"
    expected_wins = {6: 19, 12: 23}

    for horizon in (6, 12):
        scores = load_horizon(metrics_root, horizon)

        assert scores["names"] == EXPECTED_MODELS
        assert scores["rmse"].shape == (len(EXPECTED_MODELS), len(CHANNELS))
        assert scores["acc_error"].shape == scores["rmse"].shape
        assert scores["rmse_wins"] == expected_wins[horizon]
        assert scores["acc_wins"] == expected_wins[horizon]
        assert np.isfinite(scores["rmse"]).all()
        assert np.isfinite(scores["acc_error"]).all()
        assert (scores["rmse"] > 0).all()
        assert (scores["acc_error"] > 0).all()
