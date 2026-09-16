from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.make_hres_field_comparison import (
    MODEL_ORDER,
    load_comparison,
    make_figure,
)
from weather_time_interp.normalization import PAPER_CHANNELS_24


def _write_artifact(
    path: Path,
    *,
    label: str,
    horizon: int,
    index: str = "shared-index",
) -> None:
    model_name = label.lower().replace(" ", "_")
    per_tau = {}
    model_offset = 0.01 * MODEL_ORDER.index(label)
    for tau in range(1, horizon):
        record = {"n_pairs_any": 32}
        for channel_index, channel in enumerate(PAPER_CHANNELS_24):
            record[f"rmse_norm_{channel}"] = (
                0.05 + model_offset + 0.001 * tau + 0.0001 * channel_index
            )
            record[f"n_pairs_{channel}"] = 32
        per_tau[str(tau)] = {model_name: record}
    payload = {
        "schema_version": 2,
        "model_name": model_name,
        "channels": list(PAPER_CHANNELS_24),
        "per_tau": per_tau,
        "protocol": {
            "delta_t_hours": horizon,
            "taus": list(range(1, horizon)),
            "anchor_pairing": "same_initialization_forecast_leads",
            "target_source": "ERA5_at_intermediate_valid_time",
            "future_analysis_as_input": False,
            "max_inits": 16,
        },
        "provenance": {
            "forecast_anchors": {
                "init_count": 16,
                "manifest_sha256": "forecast-hash",
            },
            "era5_truth": {"dataset": "era5-2021"},
            "pressure_stats": {"sha256": "pressure-hash"},
            "surface_stats": {"sha256": "surface-hash"},
        },
        "paired_artifact": {"window_index_sha256": index},
    }
    path.write_text(json.dumps(payload))


def _artifacts(tmp_path: Path, horizon: int) -> dict[str, Path]:
    paths = {}
    for label in MODEL_ORDER:
        path = tmp_path / f"{label.replace(' ', '_')}.json"
        _write_artifact(path, label=label, horizon=horizon)
        paths[label] = path
    return paths


def test_hres_field_figure_binds_shared_protocol(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, 6)
    output = tmp_path / "hres.pdf"
    manifest = tmp_path / "hres.manifest.json"

    result = make_figure(
        artifacts,
        horizon=6,
        output=output,
        manifest=manifest,
    )

    assert output.stat().st_size > 10_000
    assert result["status"] == "complete"
    assert result["models"] == list(MODEL_ORDER)
    assert result["forecast_init_count"] == 16
    assert result["window_index_sha256"] == "shared-index"
    assert result["held_taus"] == [2, 4]
    assert json.loads(manifest.read_text())["figure"]["sha256"]


def test_hres_field_figure_rejects_unpaired_model(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, 12)
    _write_artifact(
        artifacts["WeatherBridge"],
        label="WeatherBridge",
        horizon=12,
        index="different-index",
    )

    with pytest.raises(ValueError, match="paired HRES window index differs"):
        load_comparison(artifacts, 12)
