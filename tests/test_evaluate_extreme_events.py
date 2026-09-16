from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.eval.evaluate_extreme_events import (
    field_artifact_path,
    latitude_weighted_mean,
    parse_taus,
    spatial_indices,
    validate_manifest,
)


def test_spatial_indices_selects_regular_and_wrapped_boxes() -> None:
    lat = np.asarray([60.0, 30.0, 0.0, -30.0])
    lon = np.asarray([0.0, 90.0, 180.0, 270.0])
    lat_idx, lon_idx = spatial_indices([0.0, 30.0, 80.0, 190.0], lat=lat, lon=lon)
    np.testing.assert_array_equal(lat_idx, [1, 2])
    np.testing.assert_array_equal(lon_idx, [1, 2])

    _, wrapped = spatial_indices([0.0, 30.0, 260.0, 10.0], lat=lat, lon=lon)
    np.testing.assert_array_equal(wrapped, [0, 3])


def test_latitude_weighted_mean_uses_cosine_weights() -> None:
    values = np.asarray([[[1.0]], [[9.0]]]).transpose(1, 0, 2)
    result = latitude_weighted_mean(values, np.asarray([0.0, 60.0]))
    assert result == pytest.approx((1.0 + 9.0 * 0.5) / 1.5)


def test_field_export_arguments_are_stable(tmp_path: Path) -> None:
    assert parse_taus("3,5") == (3, 5)
    assert field_artifact_path(
        tmp_path, "typhoon_rai", "WeatherDCAE-14M", 3
    ).name == "typhoon_rai__weatherdcae_14m__tau3.npz"


def test_repository_extreme_event_manifest_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "repro/extreme_events_2021.json").read_text())
    events = validate_manifest(payload)
    assert len(events) == 6
    assert {event["id"] for event in events} >= {
        "hurricane_ida",
        "texas_freeze",
        "pacific_northwest_heatwave",
    }

    sensitivity = json.loads(
        (root / "repro/ida_bbox_sensitivity_2021.json").read_text()
    )
    boxes = validate_manifest(sensitivity)
    assert len(boxes) == 3
    assert boxes[0]["bbox"] == [20.0, 32.0, 260.0, 285.0]
