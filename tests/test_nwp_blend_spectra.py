from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.eval.nwp_blend_spectra import (
    _append_window,
    _finalize_payload,
    _new_accumulator,
    hour_of_year,
    lead_model_scale,
)


def test_lead_model_scale_honours_guarded_linear_bin() -> None:
    adapter = SimpleNamespace(
        lead_scales={"fresh": 1.0, "medium": 0.5, "long": 0.0}
    )
    assert lead_model_scale(adapter, 0) == 1.0
    assert lead_model_scale(adapter, 48) == 0.5
    assert lead_model_scale(adapter, 120) == 0.0


def test_lead_model_scale_supports_schema_one_adapter() -> None:
    assert lead_model_scale(SimpleNamespace(lead_scales=None), 234) == 1.0


def test_lead_model_scale_rejects_out_of_protocol_lead() -> None:
    adapter = SimpleNamespace(
        lead_scales={"fresh": 1.0, "medium": 1.0, "long": 0.0}
    )
    with pytest.raises(ValueError, match="outside adapter bins"):
        lead_model_scale(adapter, 240)


def test_hour_of_year_uses_zero_based_calendar_hours() -> None:
    assert hour_of_year(np.datetime64("2022-01-01T00")) == (2022, 0)
    assert hour_of_year(np.datetime64("2022-02-01T06")) == (2022, 31 * 24 + 6)


def test_finalize_payload_matches_spectral_bootstrap_contract() -> None:
    accumulator = _new_accumulator(2, 3, torch.device("cpu"))
    for offset in (0, 6):
        pred = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            dtype=torch.float64,
        )
        truth = pred * 1.1
        cross = torch.complex(torch.sqrt(pred * truth), torch.zeros_like(pred))
        _append_window(
            accumulator,
            pred_power=pred,
            truth_power=truth,
            cross=cross,
            init_time_hours=100 + offset,
            anchor_lead_hours=offset,
            year=2022,
            t0=10 + offset,
            hf_ell_min=2,
        )

    payload = _finalize_payload(
        accumulator,
        model_name="linear",
        tau=2,
        channels=["Q850", "U850"],
        height=360,
        width=720,
        lmax=3,
        hf_ell_min=2,
        provenance={"test": True},
    )
    assert payload["pred_El"].shape == (2, 4)
    assert payload["window_hf_energy_ratio"].shape == (2, 2)
    assert payload["window_year"].tolist() == [2022, 2022]
    assert payload["channel_names"].tolist() == ["Q850", "U850"]
    assert str(payload["window_index_sha256"])


def test_finalize_payload_rejects_duplicate_valid_times() -> None:
    accumulator = _new_accumulator(1, 2, torch.device("cpu"))
    values = torch.ones(1, 3, dtype=torch.float64)
    cross = torch.complex(values, torch.zeros_like(values))
    for lead in (0, 6):
        _append_window(
            accumulator,
            pred_power=values,
            truth_power=values,
            cross=cross,
            init_time_hours=100,
            anchor_lead_hours=lead,
            year=2022,
            t0=12,
            hf_ell_min=1,
        )
    with pytest.raises(RuntimeError, match="duplicate valid times"):
        _finalize_payload(
            accumulator,
            model_name="linear",
            tau=2,
            channels=["Q850"],
            height=4,
            width=8,
            lmax=2,
            hf_ell_min=1,
            provenance={},
        )


def test_cloudru_launcher_keeps_primary_6h_spectral_protocol() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_nwp_blend_spectra_2022_cloudru.sh"
    ).read_text()
    assert "wait independent 6h RMSE holdout" in script
    assert "--delta-t-hours 6 --taus 2 3 --channels Q850 U850" in script
    assert "--lmax 180 --hf-ell-min 80 --max-inits 16" in script
    assert "--draws 10000 --seed 2027 --cellwise" in script
    assert 'complete.get("test_split") != "2022_independent"' in script
    assert "2022 not in selection.get(\"confirmatory_years_unopened\"" in script
