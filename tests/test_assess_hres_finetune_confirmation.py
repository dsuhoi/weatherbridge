import json
from pathlib import Path

import numpy as np
import pytest

from tools.eval.assess_hres_finetune_confirmation import (
    _load_spectral_pair,
    _parse_named_paths,
)


def _write_spectrum(
    path: Path,
    *,
    model_name: str,
    tau: int,
    index_sha: str = "a" * 64,
    shape_error: float,
    coherence: float,
    checkpoint_sha: str = "c" * 64,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        schema_version=np.asarray(6),
        channel_names=np.asarray(["Q850", "U850"]),
        tau=np.asarray(tau),
        hf_ell_min=np.asarray(80),
        window_index_sha256=np.asarray(index_sha),
        window_hf_log_shape_error=np.full((3, 2), shape_error),
        window_hf_coherence=np.full((3, 2), coherence),
        model_name=np.asarray(model_name),
        provenance_json=np.asarray(
            json.dumps(
                {
                    "checkpoint": {"sha256": checkpoint_sha},
                    "forecast_anchors": {"manifest_sha256": "f" * 64},
                    "era5_truth": {"identity_sha256": "e" * 64},
                    "protocol": {
                        "task": "test",
                        "model_name": model_name,
                        "forecast_lead_stride_hours": 24,
                        "maximum_left_forecast_lead_hours_exclusive": 120,
                    },
                },
                sort_keys=True,
            )
        ),
        window_init_time_hours=np.asarray([100, 100, 100]),
        window_anchor_lead_hours=np.asarray([0, 24, 48]),
    )


def test_parse_named_paths_rejects_duplicate_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _parse_named_paths([f"model={tmp_path / 'a'}", f"model={tmp_path / 'b'}"])


def test_spectral_gate_uses_paired_tau_and_channel_artifacts(tmp_path: Path) -> None:
    fine = tmp_path / "fine"
    raw = tmp_path / "raw"
    for tau in (2, 3):
        _write_spectrum(
            fine / f"weatherbridge_s202707_tau{tau}.npz",
            model_name="weatherbridge_s202707",
            tau=tau,
            shape_error=0.11,
            coherence=0.89,
            checkpoint_sha="1" * 64,
        )
        _write_spectrum(
            raw / f"weatherbridge_raw_tau{tau}.npz",
            model_name="weatherbridge_raw",
            tau=tau,
            shape_error=0.10,
            coherence=0.90,
            checkpoint_sha="2" * 64,
        )

    result = _load_spectral_pair(
        fine,
        raw,
        fine_name="weatherbridge_s202707",
        raw_name="weatherbridge_raw",
        taus=[2, 3],
        channels=["Q850", "U850"],
        ell_min=80,
        fine_checkpoint_sha256="1" * 64,
        raw_checkpoint_sha256="2" * 64,
        expected_left_leads=[0, 24, 48],
        expected_init_count=1,
    )

    assert len(result["rows"]) == 4
    assert result["rows"][0]["shape_error_change"] == pytest.approx(0.01)
    assert result["rows"][0]["coherence_change"] == pytest.approx(-0.01)


def test_spectral_gate_rejects_unpaired_indices(tmp_path: Path) -> None:
    fine = tmp_path / "fine"
    raw = tmp_path / "raw"
    _write_spectrum(
        fine / "fine_tau2.npz",
        model_name="fine",
        tau=2,
        index_sha="a" * 64,
        shape_error=0.1,
        coherence=0.9,
        checkpoint_sha="1" * 64,
    )
    _write_spectrum(
        raw / "raw_tau2.npz",
        model_name="raw",
        tau=2,
        index_sha="b" * 64,
        shape_error=0.1,
        coherence=0.9,
        checkpoint_sha="2" * 64,
    )
    with pytest.raises(ValueError, match="incompatible spectral artifacts"):
        _load_spectral_pair(
            fine,
            raw,
            fine_name="fine",
            raw_name="raw",
            taus=[2],
            channels=["Q850", "U850"],
            ell_min=80,
            fine_checkpoint_sha256="1" * 64,
            raw_checkpoint_sha256="2" * 64,
        )
