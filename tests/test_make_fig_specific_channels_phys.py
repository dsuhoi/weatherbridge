from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.make_fig_specific_channels_phys import (
    ALL_CHANNELS,
    CHANNELS_ORDER,
    MASS_SURFACE_CHANNELS,
    THERMODYNAMIC_CHANNELS,
    WIND_CHANNELS,
    plot_grid_phys,
    models_6h,
    models_12h,
    load_6h_phys,
)


def test_all_field_figure_uses_canonical_channel_order() -> None:
    plotted_channels = [channel for channel, _ in ALL_CHANNELS]

    assert plotted_channels == CHANNELS_ORDER
    assert len(plotted_channels) == 24
    assert len(set(plotted_channels)) == 24


def test_both_sdyff_versions_are_plotted_from_separate_metrics(tmp_path: Path) -> None:
    for models in (models_6h(), models_12h(), models_6h(True), models_12h(True)):
        sources = {name: filename for name, filename, *_ in models}
        assert sources["S-DYff"] != sources["S-DYff-ENS"]
        assert sources["S-DYff-ENS"] == "sdyff_ens21.json"
    path = tmp_path / "ensemble.json"
    path.write_text(json.dumps({"per_tau": {
        str(tau): {"model": {f"rmse_phys_{ch}": 2 for ch in CHANNELS_ORDER},
                   "ens1": {f"rmse_phys_{ch}": 9 for ch in CHANNELS_ORDER}}
        for tau in range(1, 6)}}))
    values = load_6h_phys(path)
    assert values[0, CHANNELS_ORDER.index("T850")] == 2
    assert values[0, CHANNELS_ORDER.index("Q850")] == 2000


def test_large_figure_blocks_partition_all_channels() -> None:
    groups = (
        THERMODYNAMIC_CHANNELS,
        WIND_CHANNELS,
        MASS_SURFACE_CHANNELS,
    )
    grouped_channels = [
        channel
        for group in groups
        for channel, _ in group
    ]

    assert [len(group) for group in groups] == [8, 10, 6]
    assert len(grouped_channels) == len(set(grouped_channels))
    assert set(grouped_channels) == set(CHANNELS_ORDER)


def test_channel_figure_rejects_mismatched_window_indices(
    tmp_path: Path,
) -> None:
    for name, index in (("left.json", "left"), ("right.json", "right")):
        (tmp_path / name).write_text(
            json.dumps({"evaluation_protocol": {"index_sha256": index}})
        )

    models = [
        ("WeatherBridge", "left.json", "red", "o"),
        ("WeatherDCAE-14M", "right.json", "orange", "s"),
    ]
    with pytest.raises(ValueError, match="unmatched comparison"):
        plot_grid_phys(
            lambda _: None,
            models,
            [1],
            [("T1000", "Temperature")],
            1,
            tmp_path / "figure.pdf",
            "test",
            tmp_path,
        )


def test_twelve_hour_axis_has_hourly_grid_and_even_labels(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import numpy as np

    (tmp_path / "scores.json").write_text(json.dumps({
        "evaluation_protocol": {"index_sha256": "shared-index"}}))
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda *_: None)
    try:
        plot_grid_phys(
            lambda _: np.ones((11, 24)),
            [("WeatherBridge", "scores.json", "red", "o")],
            list(range(1, 12)), [("T850", "Temperature")], 1,
            tmp_path / "figure.pdf", "test", tmp_path,
        )
        ax = plt.gcf().axes[0]
        assert ax.get_xticks().tolist() == [1, 3, 5, 7, 9, 11]
        assert ax.get_xticks(minor=True).tolist() == [2, 4, 6, 8, 10]
        assert all(tick.gridline.get_visible() for tick in ax.xaxis.get_minor_ticks())
        assert ax.get_xlim() == pytest.approx((0.6, 11.4))
    finally:
        close("all")
