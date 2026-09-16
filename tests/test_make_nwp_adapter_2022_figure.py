from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.make_nwp_adapter_2022_figure import make_figure
from weather_time_interp.normalization import PAPER_CHANNELS_24


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(path: Path, model: str, offset: float) -> None:
    per_tau = {}
    for tau in range(1, 6):
        record = {"n_pairs_any": 32}
        for channel_index, channel in enumerate(PAPER_CHANNELS_24):
            value = 0.5 + 0.01 * tau + 0.001 * channel_index + offset
            record[f"rmse_norm_{channel}"] = value
            record[f"n_pairs_{channel}"] = 32
        per_tau[str(tau)] = {model: record}
    payload = {
        "schema_version": 2,
        "model_name": model,
        "channels": list(PAPER_CHANNELS_24),
        "protocol": {
            "delta_t_hours": 6,
            "taus": [1, 2, 3, 4, 5],
            "max_inits": 16,
        },
        "paired_artifact": {"window_index_sha256": "shared-index"},
        "per_tau": per_tau,
    }
    path.write_text(json.dumps(payload))


def _score(rmse: float, improvement: float) -> dict:
    return {
        "scores": {
            "linear": {"rmse_norm": rmse},
            "flow_adapted": {"rmse_norm": rmse - improvement},
        },
        "comparisons_to_winner": {
            "linear": {
                "delta_rmse_percent_challenger_minus_winner": 100.0
                * improvement
                / rmse
            }
        },
    }


def test_nwp_adapter_figure_binds_independent_artifacts(tmp_path: Path) -> None:
    linear = tmp_path / "linear.json"
    adapted = tmp_path / "adapted.json"
    weatherdcae = tmp_path / "weatherdcae.json"
    flow_spectral = tmp_path / "flow_spectral.json"
    controls_complete = tmp_path / "controls_complete.json"
    summary = tmp_path / "summary.json"
    output = tmp_path / "figure.pdf"
    manifest = tmp_path / "figure.manifest.json"
    _write_artifact(linear, "linear", 0.0)
    _write_artifact(adapted, "flow_adapted", -0.005)
    _write_artifact(weatherdcae, "weatherdcae_14m", 0.015)
    _write_artifact(flow_spectral, "flow_spectral", 0.010)
    families = {f"tau_h{tau}": _score(0.7 + tau * 0.01, 0.005) for tau in range(1, 6)}
    families.update(
        {
            "all": _score(0.73, 0.005),
            "lead_fresh_0_48h": _score(0.3, 0.006),
            "lead_medium_48_120h": _score(0.5, 0.002),
            "lead_long_120_240h": _score(0.9, 0.0),
        }
    )
    summary.write_text(
        json.dumps(
            {
                "selection_independent": True,
                "test_split": "2022_independent",
                "winner": "flow_adapted",
                "families": families,
                "provenance": {
                    "artifacts": {
                        "linear": {"json": {"sha256": _sha256(linear)}},
                        "flow_adapted": {
                            "json": {"sha256": _sha256(adapted)}
                        },
                    }
                },
            }
        )
    )
    controls_complete.write_text(
        json.dumps(
            {
                "status": "complete",
                "test_split": "2022_independent_descriptive_controls",
                "selection_independent": False,
                "checkpoint_weights_frozen_before_2022": True,
                "window_index_sha256": "shared-index",
                "artifacts": {
                    "weatherdcae_14m": {"sha256": _sha256(weatherdcae)},
                    "flow_spectral": {"sha256": _sha256(flow_spectral)},
                },
            }
        )
    )

    result = make_figure(
        linear,
        adapted,
        summary,
        {
            "WeatherDCAE-14M": weatherdcae,
            "WeatherBridge": flow_spectral,
        },
        controls_complete,
        output=output,
        manifest=manifest,
    )

    assert output.stat().st_size > 10_000
    assert result["status"] == "complete"
    assert result["selection_independent"] is True
    assert result["figure_kind"] == "per_field_normalised_rmse_trajectories"
    assert result["taus"] == [1, 2, 3, 4, 5]
    assert result["models"] == [
        "Linear Interp.",
        "WeatherDCAE-14M",
        "WeatherBridge",
        "Frozen coefficient adapter",
    ]
    assert result["comparison_roles"] == {
        "confirmatory": ["Linear Interp.", "Frozen coefficient adapter"],
        "descriptive_post_hoc": [
            "WeatherDCAE-14M",
            "WeatherBridge",
        ],
    }
    assert result["wins"] == 120
    assert result["ties"] == 0
    assert result["losses"] == 0
    assert json.loads(manifest.read_text())["figure"]["sha256"]
