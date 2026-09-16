import datetime as dt
import json

import numpy as np
import pytest
import torch

from tools.eval.sh_energy_spectra_12h import (
    VECTOR_COMPONENT_NAMES,
    _atomic_savez,
    _band_window_diagnostics,
    _cached_spectrum_is_current,
    _filter_index_by_days_of_month,
    _hash_required_sources,
    _rescale_to_physical_anomalies,
    _spectral_stats,
)


def test_explicit_calendar_days_match_frozen_holdout_schedule():
    index = [
        (2022, day_of_year * 24, 2, None)
        for day_of_year in range(365)
    ]

    filtered = _filter_index_by_days_of_month(index, [1, 8, 15, 22])

    dates = [
        dt.date(2022, 1, 1) + dt.timedelta(days=entry[1] // 24)
        for entry in filtered
    ]
    assert len(dates) == 48
    assert {date.day for date in dates} == {1, 8, 15, 22}


def _metadata() -> dict:
    return {
        "schema_version": 6,
        "year": 2020,
        "model_name": "candidate",
        "model_kind": "capmatched",
        "selection_sha256": "fixed-window-index",
        "samples_per_date": 2,
    }


def _payload(metadata: dict, *, n_samples: int = 3) -> dict:
    channels = np.asarray(["t2m", "T850"])
    spectral_shape = (len(channels), 3)
    window_shape = (n_samples, len(channels))
    vector_shape = (0, len(VECTOR_COMPONENT_NAMES), 3)
    vector_window_shape = (n_samples, 0, len(VECTOR_COMPONENT_NAMES))
    return {
        "ell": np.arange(3, dtype=np.int32),
        "pred_El": np.ones(spectral_shape),
        "gt_El": np.ones(spectral_shape),
        "cross_El_real": np.ones(spectral_shape),
        "cross_El_imag": np.zeros(spectral_shape),
        "coherence_l": np.ones(spectral_shape),
        "signed_cospectrum_l": np.ones(spectral_shape),
        "channel_names": channels,
        "n_samples": np.asarray(n_samples),
        "window_year": np.full(n_samples, 2020, dtype=np.int16),
        "window_t0": np.arange(n_samples, dtype=np.int32) * 24,
        "window_hf_energy_ratio": np.ones(window_shape),
        "window_hf_log_shape_error": np.zeros(window_shape),
        "window_hf_coherence": np.ones(window_shape),
        "window_hf_signed_cospectrum": np.ones(window_shape),
        "wind_pair_names": np.asarray([], dtype="U1"),
        "vector_component_names": np.asarray(VECTOR_COMPONENT_NAMES),
        "vector_pred_El": np.ones(vector_shape),
        "vector_gt_El": np.ones(vector_shape),
        "vector_cross_El_real": np.ones(vector_shape),
        "vector_cross_El_imag": np.zeros(vector_shape),
        "vector_coherence_l": np.ones(vector_shape),
        "vector_signed_cospectrum_l": np.ones(vector_shape),
        "window_vector_hf_energy_ratio": np.ones(vector_window_shape),
        "window_vector_hf_log_shape_error": np.zeros(vector_window_shape),
        "window_vector_hf_coherence": np.ones(vector_window_shape),
        "window_vector_hf_signed_cospectrum": np.ones(vector_window_shape),
        "tau": np.asarray(2),
        "H": np.asarray(4),
        "W": np.asarray(8),
        "lmax": np.asarray(2),
        "model_name": np.asarray("candidate"),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }


def _is_current(path, metadata: dict) -> bool:
    return _cached_spectrum_is_current(
        path,
        tau=2,
        lmax=2,
        channels=["t2m", "T850"],
        expected_n_samples=3,
        expected_metadata=metadata,
    )


def test_spectrum_writer_rejects_missing_supporting_source(tmp_path):
    existing = tmp_path / "existing.py"
    existing.write_text("value = 1\n")

    with pytest.raises(FileNotFoundError, match="missing.py"):
        _hash_required_sources(
            (existing, tmp_path / "missing.py"),
            tmp_path,
        )


def test_cached_spectrum_requires_exact_protocol_and_complete_arrays(tmp_path):
    metadata = _metadata()
    path = tmp_path / "candidate_tau2.npz"
    np.savez(path, **_payload(metadata))
    assert _is_current(path, metadata)

    stale_metadata = dict(metadata, selection_sha256="different-window-index")
    assert not _is_current(path, stale_metadata)

    malformed = _payload(metadata)
    malformed["window_hf_energy_ratio"][0, 0] = np.nan
    np.savez(path, **malformed)
    assert not _is_current(path, metadata)

    path.write_bytes(b"partial npz")
    assert not _is_current(path, metadata)


def test_atomic_savez_replaces_target_without_temporary_file(tmp_path):
    path = tmp_path / "spectrum.npz"
    _atomic_savez(path, {"value": np.asarray([1.0, 2.0])})
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["value"], [1.0, 2.0])
    assert list(tmp_path.glob(".*.tmp")) == []


def test_spectral_stats_restores_negative_longitudinal_modes():
    pred_coeff = torch.tensor(
        [[[[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]]]],
        dtype=torch.complex128,
    )
    truth_coeff = pred_coeff.clone()

    class FakeSHT:
        def __init__(self):
            self.calls = 0

        def __call__(self, _field):
            self.calls += 1
            return pred_coeff if self.calls == 1 else truth_coeff

    pred, truth, cross = _spectral_stats(
        torch.zeros(1, 1, 2, 2),
        torch.zeros(1, 1, 2, 2),
        FakeSHT(),
    )
    expected = torch.tensor([[9.0, 41.0]], dtype=torch.float64)
    torch.testing.assert_close(pred, expected)
    torch.testing.assert_close(truth, expected)
    torch.testing.assert_close(cross.real, expected)
    torch.testing.assert_close(cross.imag, torch.zeros_like(expected))


def test_spectral_fields_are_rescaled_to_physical_anomaly_units():
    standardized = torch.ones(2, 2, 3, 4)
    result = _rescale_to_physical_anomalies(
        standardized,
        torch.tensor([2.0, 5.0]),
    )

    torch.testing.assert_close(result[:, 0], torch.full((2, 3, 4), 2.0))
    torch.testing.assert_close(result[:, 1], torch.full((2, 3, 4), 5.0))


def test_spectral_field_rescaling_rejects_invalid_channel_std():
    with pytest.raises(ValueError, match="strictly positive"):
        _rescale_to_physical_anomalies(
            torch.ones(1, 2, 3, 4),
            torch.tensor([1.0, 0.0]),
        )


def test_band_shape_is_invariant_to_uniform_energy_scale():
    truth = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    prediction = 4.0 * truth
    cross = 2.0 * truth.to(torch.complex128)

    ratio, shape_error, coherence, signed_cospectrum = _band_window_diagnostics(
        prediction,
        truth,
        cross,
        0,
    )

    torch.testing.assert_close(ratio, torch.tensor([4.0], dtype=torch.float64))
    torch.testing.assert_close(shape_error, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(coherence, torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(
        signed_cospectrum,
        torch.ones(1, dtype=torch.float64),
    )


def test_coherence_does_not_hide_antiphase_in_signed_cospectrum():
    truth = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    cross = -truth.to(torch.complex128)

    _, _, coherence, signed_cospectrum = _band_window_diagnostics(
        truth,
        truth,
        cross,
        0,
    )

    torch.testing.assert_close(coherence, torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(
        signed_cospectrum,
        -torch.ones(1, dtype=torch.float64),
    )
