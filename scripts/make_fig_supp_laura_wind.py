#!/usr/bin/env python3
"""Render the provenance-bound Hurricane Laura wind comparison."""
from __future__ import annotations

try:
    from scripts import make_fig_supp_ciara_wind as renderer
except ModuleNotFoundError:  # Direct execution: python scripts/<file>.py
    import make_fig_supp_ciara_wind as renderer


EXPECTED_INIT_TIME = "2020-08-27T03:00:00"
EXPECTED_MODELS = ("WeatherDCAE-14M", "PixelAttn-VFI", "WeatherBridge")


def main() -> None:
    if renderer.LEARNED_METHODS != EXPECTED_MODELS:
        raise ValueError("shared wind renderer model set changed")
    renderer.EXPECTED_INIT_TIME = EXPECTED_INIT_TIME
    renderer.EVENT_NAME = "Hurricane Laura"
    renderer.EVENT_DIR = "hurricane_laura_2020"
    renderer.DISPLAY_TAU = 3
    renderer.DISPLAY_TITLE = (
        r"Hurricane Laura, 27 August 2020, 06 UTC ($\tau=3$ h)"
    )
    renderer.MAP_EXTENT = (-110.0, -70.0, 15.0, 35.0)
    renderer.X_TICKS = (-110.0, -90.0, -70.0)
    renderer.Y_TICKS = (15.0, 25.0, 35.0)
    renderer.X_TICK_LABELS = ("110°W", "90°W", "70°W")
    renderer.Y_TICK_LABELS = ("15°N", "25°N", "35°N")
    renderer.SELECTION_NOTE = (
        "Additional illustrative case fixed by event identity and landfall "
        "window, not selected by model error."
    )
    renderer.OUTPUT = (
        renderer.ROOT / "paper" / "images" / "fig_laura_wind10.pdf"
    )
    renderer.SUMMARY = (
        renderer.ROOT
        / "metrics"
        / "case_studies_weatherbridge"
        / renderer.EVENT_DIR
        / "wind10_summary.json"
    )
    renderer.main()


if __name__ == "__main__":
    main()
