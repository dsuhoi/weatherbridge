from __future__ import annotations

import json
from pathlib import Path

from scripts.make_fig_extreme_events import metric_value, select_display_cases


ROOT = Path(__file__).resolve().parents[1]


def test_displayed_maps_require_weatherbridge_to_beat_weatherdcae() -> None:
    metrics = json.loads(
        (ROOT / "metrics/extreme_events_2021/weatherbridge_vs_weatherdcae.json")
        .read_text()
    )

    selected = select_display_cases(metrics, tau=3)
    selected_ids = [case["event_id"] for case in selected]

    assert selected_ids == ["typhoon_rai", "typhoon_surigae", "hurricane_ida"]
    assert "typhoon_chanthu" not in selected_ids
    assert metric_value(metrics, "WeatherBridge", "typhoon_chanthu", 3, "mslp") > (
        metric_value(metrics, "WeatherDCAE-14M", "typhoon_chanthu", 3, "mslp")
    )
