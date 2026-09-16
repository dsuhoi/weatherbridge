import hashlib
import json

import numpy as np
import pytest

from tools.eval.batch_eval_forecast_anchor import (
    CHANNELS_ORDER,
    _sha256_arrays,
    forecast_evaluation_source_paths,
)
from tools.eval.summarize_forecast_anchor import (
    load_artifact,
    summarize_artifacts,
    validate_selection_lineage,
)
from weather_time_interp.normalization import file_provenance


def _write_artifact(
    tmp_path,
    name: str,
    error: float,
    *,
    delta_t_hours: int = 6,
    taus: tuple[int, ...] = (1, 2, 3, 4, 5),
):
    n_blocks = 12
    leads = np.asarray([0, 60, 132], dtype=np.int16)
    tau_array = np.asarray(taus, dtype=np.int8)
    block_window_lead = np.repeat(leads, tau_array.size)
    block_window_tau = np.tile(tau_array, leads.size)
    init_time = np.repeat(
        np.arange(n_blocks, dtype=np.int64) * 7 * 24,
        block_window_tau.size,
    )
    lead = np.tile(block_window_lead, n_blocks)
    tau = np.tile(block_window_tau, n_blocks)
    squared_error = np.full(
        (tau.size, len(CHANNELS_ORDER)),
        error,
        dtype=np.float32,
    )
    valid_channel = np.ones_like(squared_error, dtype=bool)
    npz_path = tmp_path / f"{name}.paired.npz"
    np.savez_compressed(
        npz_path,
        init_time_hours=init_time,
        anchor_lead_hours=lead,
        tau_hours=tau,
        squared_error_norm=squared_error,
        valid_channel=valid_channel,
        channel_names=np.asarray(CHANNELS_ORDER),
    )
    json_path = tmp_path / f"{name}.json"
    model_provenance = (
        {"kind": "linear"}
        if name == "linear"
        else {
            "kind": "capmatched_checkpoint",
            "artifact": {
                "sha256": hashlib.sha256(name.encode()).hexdigest()
            },
        }
    )
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model_name": name,
                "protocol": {
                    "delta_t_hours": delta_t_hours,
                    "taus": list(taus),
                    "anchor_pairing": (
                        "same_initialization_forecast_leads"
                    ),
                    "target_source": "ERA5_at_intermediate_valid_time",
                    "future_analysis_as_input": False,
                    "area_weighting": "spherical_latitude_strip_area",
                    "latitude_grid": "wb2_0p25_2x2_block_average_v1",
                    "normalization": "(x - training_mean) / training_std",
                    "missing_anchor_channel_imputation": "training_mean",
                    "missing_channels_excluded_from_metrics": True,
                    "init_selection": "evenly_spaced_over_sorted_archive",
                    "max_inits": 16,
                },
                "channels": CHANNELS_ORDER,
                "provenance": {
                    "evaluation_code": {
                        source_name: file_provenance(source_path)
                        for source_name, source_path
                        in forecast_evaluation_source_paths().items()
                    },
                    "model": model_provenance,
                    "forecast_anchors": {
                        "init_count": 16,
                        "manifest_sha256": "a" * 64,
                        "grid": {"latitude_order": "north_to_south"},
                        "archive_manifest": {"sha256": "d" * 64},
                    },
                    "era5_truth": {"years": [2021]},
                    "pressure_stats": {"sha256": "b" * 64},
                    "surface_stats": {"sha256": "c" * 64},
                },
                "paired_artifact": {
                    **file_provenance(npz_path),
                    "n_windows": int(tau.size),
                    "window_index_sha256": _sha256_arrays(
                        init_time,
                        lead,
                        tau,
                    ),
                },
            }
        )
    )
    return json_path


def test_load_artifact_accepts_provenance_bound_nwp_blend(tmp_path) -> None:
    path = _write_artifact(tmp_path, "adapted", 0.5)
    payload = json.loads(path.read_text())
    payload["provenance"]["model"]["kind"] = (
        "capmatched_checkpoint_with_nwp_blend"
    )
    payload["provenance"]["model"]["blend_adapter"] = {
        "sha256": "e" * 64
    }
    path.write_text(json.dumps(payload))

    artifact = load_artifact("adapted", path)

    assert artifact.model_kind == "capmatched_checkpoint_with_nwp_blend"


def test_forecast_summary_uses_paired_initialization_blocks(tmp_path) -> None:
    paths = {
        "winner": _write_artifact(tmp_path, "winner", 1.0),
        "pp3": _write_artifact(tmp_path, "pp3", 4.0),
        "linear": _write_artifact(tmp_path, "linear", 9.0),
    }
    artifacts = {
        name: load_artifact(name, path)
        for name, path in paths.items()
    }

    summary = summarize_artifacts(
        artifacts,
        "winner",
        draws=5000,
        seed=17,
    )

    all_family = summary["families"]["all"]
    assert all_family["scores"]["winner"]["rmse_norm"] == pytest.approx(1.0)
    assert all_family["scores"]["pp3"]["rmse_norm"] == pytest.approx(2.0)
    comparison = all_family["comparisons_to_winner"]["pp3"]
    assert comparison["delta_mse_challenger_minus_winner"] == pytest.approx(3.0)
    assert comparison["n_blocks"] == 12
    assert comparison["winner_significantly_better"]
    assert not comparison["winner_significantly_worse"]
    assert summary["no_significant_unseen_forecast_regression"]
    assert summary["unseen_forecast_noninferior_2pct_rmse"]
    assert summary["unseen_forecast_noninferiority_failures"] == []
    assert summary["significant_unseen_forecast_regressions"] == []
    assert summary["significant_unseen_forecast_dominance"] == [
        "pp3",
        "linear",
    ]


def test_forecast_summary_rejects_unpaired_indices(tmp_path) -> None:
    winner_path = _write_artifact(tmp_path, "winner", 1.0)
    challenger_path = _write_artifact(tmp_path, "challenger", 2.0)
    winner = load_artifact("winner", winner_path)
    challenger = load_artifact("challenger", challenger_path)
    challenger.init_time[0] += 1

    with pytest.raises(ValueError, match="index|init_time"):
        summarize_artifacts(
            {"winner": winner, "challenger": challenger},
            "winner",
            draws=100,
            seed=1,
        )


def test_forecast_summary_rejects_non_2021_truth(tmp_path) -> None:
    path = _write_artifact(tmp_path, "winner", 1.0)
    payload = json.loads(path.read_text())
    payload["provenance"]["era5_truth"]["years"] = [2020]
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="data provenance"):
        load_artifact("winner", path)


def test_forecast_summary_rejects_future_analysis_input(tmp_path) -> None:
    path = _write_artifact(tmp_path, "winner", 1.0)
    payload = json.loads(path.read_text())
    payload["protocol"]["future_analysis_as_input"] = True
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="invalid forecast protocol"):
        load_artifact("winner", path)


def test_forecast_summary_rejects_stale_evaluation_code(tmp_path) -> None:
    path = _write_artifact(tmp_path, "winner", 1.0)
    payload = json.loads(path.read_text())
    payload["provenance"]["evaluation_code"][
        "capmatched_loader.py"
    ]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="stale forecast evaluation source"):
        load_artifact("winner", path)


def test_forecast_summary_supports_12h_seen_unseen_protocol(tmp_path) -> None:
    taus = tuple(range(1, 12))
    paths = {
        name: _write_artifact(
            tmp_path,
            name,
            error,
            delta_t_hours=12,
            taus=taus,
        )
        for name, error in (("winner", 1.0), ("reference", 2.0))
    }

    summary = summarize_artifacts(
        {
            name: load_artifact(name, path)
            for name, path in paths.items()
        },
        "winner",
        draws=1000,
        seed=11,
        seen_taus=(1, 2, 3, 5, 7, 9, 10, 11),
        unseen_taus=(4, 6, 8),
    )

    assert summary["protocol"]["delta_t_hours"] == 12
    assert summary["families"]["unseen_tau"]["scores"]["winner"][
        "n_windows"
    ] == 12 * 3 * 3


def test_forecast_summary_binds_checkpoints_to_frozen_selection(
    tmp_path,
) -> None:
    paths = {
        name: _write_artifact(tmp_path, name, 1.0)
        for name in ("winner", "reference", "linear")
    }
    artifacts = {
        name: load_artifact(name, path)
        for name, path in paths.items()
    }
    selection = {
        "models": {
            name: {
                "checkpoint_sha256": hashlib.sha256(name.encode()).hexdigest()
            }
            for name in ("winner", "reference")
        }
    }

    validate_selection_lineage(artifacts, selection)
    selection["models"]["reference"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        validate_selection_lineage(artifacts, selection)
