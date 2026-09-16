from __future__ import annotations

import numpy as np
import pytest

from production.weatherbridge_app.render import colorize, palette_for


def test_colorize_has_stable_global_output_size() -> None:
    values = np.linspace(-2, 3, 360 * 720, dtype=np.float32).reshape(360, 720)
    image = colorize(values, -2.0, 3.0, palette_for("u10"))
    assert image.size == (1440, 720)
    assert image.mode == "RGB"


def test_colorize_rejects_nonfinite_fields() -> None:
    values = np.zeros((360, 720), dtype=np.float32)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        colorize(values, 0.0, 1.0, "thermal")
