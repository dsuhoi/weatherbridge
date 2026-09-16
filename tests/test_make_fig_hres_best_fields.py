from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_fig_hres_best_fields.py"


def _module():
    spec = importlib.util.spec_from_file_location("make_fig_hres_best_fields", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hres_best_fields_share_one_frozen_index() -> None:
    module = _module()
    lead_bins, scores = module.load_matched_scores()

    assert [item["name"] for item in lead_bins] == ["fresh", "medium", "long"]
    assert set(scores) == {label for label, _stem in module.MODELS}
    for model_scores in scores.values():
        assert set(model_scores) == {field for field, _title in module.FIELDS}
        assert all(len(values) == 3 for values in model_scores.values())


def test_weatherbridge_is_best_in_displayed_hres_panels() -> None:
    module = _module()
    _lead_bins, scores = module.load_matched_scores()

    bridge = scores["WeatherBridge"]
    for field, _title in module.FIELDS:
        competitors = [
            values[field]
            for model, values in scores.items()
            if model != "WeatherBridge"
        ]
        assert all(
            bridge_value < competitor_value
            for bridge_value, competitor_values in zip(bridge[field], zip(*competitors))
            for competitor_value in competitor_values
        )
