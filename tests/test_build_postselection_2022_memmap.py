import numpy as np
import pytest

from tools.data.build_postselection_2022_memmap import (
    _download_pressure_variable,
    selected_relative_hours_from_inits,
)


def test_explicit_init_times_expand_over_fixed_forecast_leads() -> None:
    hours = selected_relative_hours_from_inits(
        2022,
        ["2022-01-01T00", "2022-01-02T00"],
        2,
    )

    assert hours == [0, 1, 2, 24, 25, 26]
    with pytest.raises(ValueError, match="unique, sorted"):
        selected_relative_hours_from_inits(
            2022,
            ["2022-01-02T00", "2022-01-01T00"],
            2,
        )


def test_pressure_download_indexes_only_requested_levels() -> None:
    keys = []

    class OrthogonalIndex:
        def __getitem__(self, key):
            keys.append(key)
            return np.zeros((4, 720, 1440), dtype=np.float32)

    class Array:
        oindex = OrthogonalIndex()

    outputs = list(
        _download_pressure_variable(
            Array(),
            [0],
            2022,
            [1, 2, 3, 4],
            workers=1,
        )
    )

    assert keys[0][1] == [1, 2, 3, 4]
    assert outputs[0][1].shape == (4, 360, 720)
