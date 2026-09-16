import json
from pathlib import Path

import numpy as np
import pytest

from tools.eval.eval_anchor_exchange_consistency import (
    _index_sha256,
    _summary,
    anchor_exchange_evaluation_source_paths,
)
from tools.eval.summarize_anchor_exchange_consistency import (
    _sha256,
    compare,
    load_scores,
)
from weather_time_interp.metrics.physical_consistency import (
    CANONICAL_24_CHANNELS,
)


def _write_artifact(
    root: Path,
    year: int,
    *,
    winner_mse: float,
    reference_mse: float,
) -> Path:
    taus = np.tile(np.arange(1, 6), 10)
    t0 = np.repeat(np.arange(10) * 7 * 24, 5)
    window_year = np.full(taus.size, year, dtype=np.int64)
    endpoint_t0 = np.arange(10, dtype=np.int64) * 7 * 24
    endpoint_year = np.full(endpoint_t0.size, year, dtype=np.int64)
    winner_exchange = np.full(
        (taus.size, len(CANONICAL_24_CHANNELS)),
        winner_mse,
    )
    reference_exchange = np.full(
        (taus.size, len(CANONICAL_24_CHANNELS)),
        reference_mse,
    )
    winner_forward = np.ones_like(winner_exchange)
    reference_forward = np.ones_like(reference_exchange)
    winner_endpoint = np.zeros(
        (endpoint_t0.size, len(CANONICAL_24_CHANNELS)),
    )
    reference_endpoint = np.full_like(winner_endpoint, 0.01)
    npz_path = root / f"exchange_{year}.npz"
    np.savez(
        npz_path,
        window_year=window_year,
        window_t0=t0.astype(np.int64),
        window_tau=taus.astype(np.int64),
        endpoint_year=endpoint_year,
        endpoint_t0=endpoint_t0,
        channel_names=np.asarray(CANONICAL_24_CHANNELS),
        winner_exchange_mse=winner_exchange,
        winner_forward_mse=winner_forward,
        winner_endpoint0_mse=winner_endpoint,
        winner_endpoint1_mse=winner_endpoint,
        reference_exchange_mse=reference_exchange,
        reference_forward_mse=reference_forward,
        reference_endpoint0_mse=reference_endpoint,
        reference_endpoint1_mse=reference_endpoint,
    )

    def model_payload(
        checkpoint_sha256: str,
        exchange: np.ndarray,
        forward: np.ndarray,
        endpoint: np.ndarray,
    ) -> dict:
        exchange_summary = _summary(exchange)
        forward_summary = _summary(forward)
        return {
            "model_type": "test",
            "checkpoint_provenance": {"sha256": checkpoint_sha256},
            "exchange": exchange_summary,
            "forward_error": forward_summary,
            "exchange_to_forward_rmse_ratio": (
                exchange_summary["pooled_rmse"]
                / max(forward_summary["pooled_rmse"], 1e-12)
            ),
            "endpoint0": _summary(endpoint),
            "endpoint1": _summary(endpoint),
        }

    artifact = {
        "schema_version": 2,
        "diagnostic_only": True,
        "evaluation_code_provenance": {
            name: _sha256(path)
            for name, path in (
                anchor_exchange_evaluation_source_paths().items()
            )
        },
        "test_year": year,
        "delta_t_hours": 6,
        "eval_hours": [1, 2, 3, 4, 5],
        "channel_names": list(CANONICAL_24_CHANNELS),
        "evaluation_protocol": {
            "samples_per_date": 2,
            "eval_days_per_month": 2,
            "window_index_sha256": _index_sha256(
                window_year,
                t0,
                taus,
            ),
            "endpoint_index_sha256": _index_sha256(
                endpoint_year,
                endpoint_t0,
            ),
            "n_windows": int(window_year.size),
            "n_endpoints": int(endpoint_year.size),
        },
        "evaluation_dataset_provenance": {"years": [year]},
        "evaluation_input_provenance": {
            "static_features": {"sha256": "b" * 64},
            "pressure_level_stats": {"sha256": "c" * 64},
            "surface_stats": {"sha256": "d" * 64},
        },
        "paired_windows_file": npz_path.name,
        "paired_windows_size_bytes": npz_path.stat().st_size,
        "paired_windows_sha256": _sha256(npz_path),
        "models": {
            "winner": model_payload(
                "winner-sha",
                winner_exchange,
                winner_forward,
                winner_endpoint,
            ),
            "reference": model_payload(
                "reference-sha",
                reference_exchange,
                reference_forward,
                reference_endpoint,
            ),
        },
    }
    path = root / f"exchange_{year}.json"
    path.write_text(json.dumps(artifact))
    return path


def _selection() -> dict:
    return {
        "schema_version": 12,
        "winner": "winner",
        "ood_attached_at_selection_time": False,
        "models": {
            "winner": {"checkpoint_sha256": "winner-sha"},
            "reference": {"checkpoint_sha256": "reference-sha"},
        },
    }


def test_compare_reports_significant_ood_exchange_gain(
    tmp_path: Path,
) -> None:
    artifacts = {
        str(year): _write_artifact(
            tmp_path,
            year,
            winner_mse=0.1,
            reference_mse=0.2,
        )
        for year in (2020, 2021)
    }

    report = compare(
        _selection(),
        artifacts,
        block_days=7,
        draws=2000,
        seed=7,
    )

    assert report["no_significant_2021_unseen_exchange_regression"] is True
    assert report["winner_2021_unseen_exchange_noninferior_2pct"] is True
    assert (
        report[
            "winner_2021_unseen_exchange_noninferiority_failures"
        ]
        == []
    )
    assert report["significant_2021_unseen_exchange_dominance"] == [
        "reference"
    ]
    assert report["winner_endpoint_rmse"]["2021"]["endpoint0"] == 0.0


def test_loader_rejects_tampered_paired_artifact(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path,
        2020,
        winner_mse=0.1,
        reference_mse=0.2,
    )
    with np.load(tmp_path / "exchange_2020.npz") as values:
        replacement = {key: values[key] for key in values.files}
    replacement["winner_exchange_mse"] *= 2.0
    np.savez(tmp_path / "exchange_2020.npz", **replacement)

    with pytest.raises(ValueError, match="paired NPZ hash mismatch"):
        load_scores(artifact, _selection())


def test_loader_rejects_tampered_endpoint_summary(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(
        tmp_path,
        2021,
        winner_mse=0.1,
        reference_mse=0.2,
    )
    payload = json.loads(artifact.read_text())
    payload["models"]["winner"]["endpoint0"]["pooled_rmse"] = 0.5
    artifact.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="endpoint0: pooled_rmse mismatch"):
        load_scores(artifact, _selection())


def test_loader_rejects_unbound_window_index(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path,
        2021,
        winner_mse=0.1,
        reference_mse=0.2,
    )
    npz_path = tmp_path / "exchange_2021.npz"
    with np.load(npz_path, allow_pickle=False) as values:
        arrays = {key: values[key] for key in values.files}
    arrays["window_t0"] = arrays["window_t0"] + 1
    np.savez(npz_path, **arrays)
    payload = json.loads(artifact.read_text())
    payload["paired_windows_size_bytes"] = npz_path.stat().st_size
    payload["paired_windows_sha256"] = _sha256(npz_path)
    artifact.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="paired protocol coverage mismatch"):
        load_scores(artifact, _selection())


def test_loader_rejects_stale_evaluation_source(tmp_path: Path) -> None:
    artifact = _write_artifact(
        tmp_path,
        2021,
        winner_mse=0.1,
        reference_mse=0.2,
    )
    payload = json.loads(artifact.read_text())
    source_name = next(iter(payload["evaluation_code_provenance"]))
    payload["evaluation_code_provenance"][source_name] = "0" * 64
    artifact.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="evaluation source hash mismatch"):
        load_scores(artifact, _selection())


def test_compare_rejects_pre_robust_selection() -> None:
    selection = _selection()
    selection["schema_version"] = 11

    with pytest.raises(ValueError, match="invalid frozen selection"):
        compare(
            selection,
            {},
            block_days=7,
            draws=100,
            seed=7,
        )


def test_loader_rejects_incomplete_channel_protocol(
    tmp_path: Path,
) -> None:
    artifact = _write_artifact(
        tmp_path,
        2021,
        winner_mse=0.1,
        reference_mse=0.2,
    )
    payload = json.loads(artifact.read_text())
    payload["channel_names"].remove("mslp")
    artifact.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="evaluation protocol mismatch"):
        load_scores(artifact, _selection())


def test_compare_supports_12h_tau_partition(tmp_path: Path) -> None:
    artifacts = {
        str(year): _write_artifact(
            tmp_path,
            year,
            winner_mse=0.1,
            reference_mse=0.2,
        )
        for year in (2020, 2021)
    }
    for path in artifacts.values():
        payload = json.loads(path.read_text())
        payload["delta_t_hours"] = 12
        payload["eval_hours"] = list(range(1, 12))
        path.write_text(json.dumps(payload))
        npz_path = path.with_suffix(".npz")
        with np.load(npz_path, allow_pickle=False) as values:
            arrays = {key: values[key] for key in values.files}
        taus = np.tile(np.arange(1, 12), 10)
        arrays["window_tau"] = taus
        arrays["window_t0"] = np.repeat(np.arange(10) * 7 * 24, 11)
        arrays["window_year"] = np.full(taus.size, int(path.stem[-4:]))
        arrays["winner_exchange_mse"] = np.full(
            (taus.size, len(CANONICAL_24_CHANNELS)),
            0.1,
        )
        arrays["reference_exchange_mse"] = np.full(
            (taus.size, len(CANONICAL_24_CHANNELS)),
            0.2,
        )
        arrays["winner_forward_mse"] = np.ones_like(
            arrays["winner_exchange_mse"]
        )
        arrays["reference_forward_mse"] = np.ones_like(
            arrays["reference_exchange_mse"]
        )
        np.savez(npz_path, **arrays)
        payload["evaluation_protocol"]["window_index_sha256"] = (
            _index_sha256(
                arrays["window_year"],
                arrays["window_t0"],
                arrays["window_tau"],
            )
        )
        payload["evaluation_protocol"]["n_windows"] = int(taus.size)
        for model in ("winner", "reference"):
            exchange_summary = _summary(
                arrays[f"{model}_exchange_mse"]
            )
            forward_summary = _summary(
                arrays[f"{model}_forward_mse"]
            )
            payload["models"][model]["exchange"] = exchange_summary
            payload["models"][model]["forward_error"] = forward_summary
            payload["models"][model][
                "exchange_to_forward_rmse_ratio"
            ] = (
                exchange_summary["pooled_rmse"]
                / forward_summary["pooled_rmse"]
            )
        payload["paired_windows_size_bytes"] = npz_path.stat().st_size
        payload["paired_windows_sha256"] = _sha256(npz_path)
        path.write_text(json.dumps(payload))

    report = compare(
        _selection(),
        artifacts,
        block_days=7,
        draws=1000,
        seed=7,
        tau_groups={
            "all": np.arange(1, 12),
            "seen": np.asarray([1, 2, 3, 5, 7, 9, 10, 11]),
            "unseen": np.asarray([4, 6, 8]),
        },
    )

    assert report["protocol"]["delta_t_hours"] == 12
    assert report["protocol"]["unseen_taus"] == [4, 6, 8]
