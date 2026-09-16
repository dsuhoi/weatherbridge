import numpy as np
import pytest

from tools.data.build_hres_forecast_memmap import (
    canonical_source_indices,
    finite_block_mean,
    lead_hours,
    parse_init_times,
)
from weather_time_interp.grid import (
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)


def test_canonical_hres_indices_reverse_before_dropping_south_pole():
    source_latitude = np.linspace(-90.0, 90.0, 721)
    source_longitude = np.arange(1440) * 0.25

    lat_index, lon_index, latitude, longitude = canonical_source_indices(
        source_latitude,
        source_longitude,
    )

    np.testing.assert_array_equal(lat_index[:4], [720, 719, 718, 717])
    np.testing.assert_array_equal(lat_index[-2:], [2, 1])
    np.testing.assert_array_equal(lon_index, np.arange(1440))
    np.testing.assert_array_equal(latitude, wb2_block_average_latitudes())
    np.testing.assert_array_equal(longitude, wb2_block_average_longitudes())


def test_parse_init_times_requires_sorted_unique_values():
    result = parse_init_times("2021-01-01T00,2021-01-21T00")
    assert result == [
        np.datetime64("2021-01-01T00", "h"),
        np.datetime64("2021-01-21T00", "h"),
    ]
    with pytest.raises(ValueError, match="sorted"):
        parse_init_times("2021-01-21T00,2021-01-01T00")
    with pytest.raises(ValueError, match="duplicates"):
        parse_init_times("2021-01-01T00,2021-01-01T00")


def test_lead_hours_accepts_explicit_integer_hour_units():
    class Coordinate:
        values = np.asarray([0, 6, 12], dtype=np.int64)
        attrs = {"units": "hours"}

    np.testing.assert_array_equal(lead_hours(Coordinate()), [0, 6, 12])


def test_lead_hours_rejects_unlabelled_integers():
    class Coordinate:
        values = np.asarray([0, 6, 12], dtype=np.int64)
        attrs = {}

    with pytest.raises(ValueError, match="units=hours"):
        lead_hours(Coordinate())


def test_finite_block_mean_records_one_missing_native_value():
    values = np.ones((1, 720, 1440), dtype=np.float32)
    values[0, 0, 0] = np.nan
    values[0, 0, 1] = 4.0

    output, qc = finite_block_mean(values)

    assert output[0, 0, 0] == pytest.approx(2.0)
    assert qc["source_nonfinite_values"] == 1
    assert qc["affected_output_values"] == 1
    assert qc["minimum_finite_native_values_per_output"] == 3
    assert qc["output_nonfinite_values"] == 0


def test_finite_block_mean_accepts_two_sparse_missing_native_values():
    values = np.ones((2, 720, 1440), dtype=np.float32)
    values[0, 0, :2] = np.nan

    output, qc = finite_block_mean(values)

    assert output[0, 0, 0] == pytest.approx(1.0)
    assert qc["minimum_finite_native_values_per_output"] == 2
    assert qc["source_nonfinite_fraction"] < 1.0e-6


def test_finite_block_mean_rejects_fewer_than_two_native_values():
    values = np.ones((2, 720, 1440), dtype=np.float32)
    values[0, 0, :2] = np.nan
    values[0, 1, 0] = np.nan

    with pytest.raises(ValueError, match="fewer than two"):
        finite_block_mean(values)


def test_finite_block_mean_rejects_excessive_sparse_missingness():
    values = np.ones((1, 720, 1440), dtype=np.float32)
    values[0, 0, :6] = np.nan

    with pytest.raises(ValueError, match="non-finite fraction"):
        finite_block_mean(values)
