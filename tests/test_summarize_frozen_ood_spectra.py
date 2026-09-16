from pathlib import Path

import numpy as np
import pytest

from tools.eval import summarize_frozen_ood_spectra as summary
from tools.eval.spectral_block_bootstrap import SpectralWindows
from weather_time_interp.metrics.physical_consistency import (
    CANONICAL_24_CHANNELS,
)


def _selection() -> dict:
    return {
        "schema_version": 12,
        "ood_attached_at_selection_time": False,
        "winner": "winner",
        "models": {
            "winner": {"checkpoint_sha256": "winner-sha"},
            "challenger": {"checkpoint_sha256": "challenger-sha"},
        },
    }


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    winner_is_better: bool,
    year: int = 2021,
    canonical_channels: bool = True,
) -> None:
    def fake_metrics(
        root: Path,
        model: str,
        hf_ell_min: int,
        taus: set[int],
        *,
        required_lmax: int,
        expected_channels: int,
    ) -> dict:
        del root
        assert hf_ell_min == 180
        assert taus == {4}
        assert required_lmax == 359
        assert expected_channels == 24
        return {
            "checkpoint_sha256": f"{model}-sha",
            "window_index_sha256": "paired-index",
            "spectral_grid_sha256": "ell-grid",
            "spectral_channel_names_sha256": (
                summary.CANONICAL_CHANNEL_HASH
                if canonical_channels
                else "wrong-channel-order"
            ),
            "evaluation_input_provenance": {"stats": "same"},
            "evaluation_dataset_provenance": {
                "years": [year],
                "identity_sha256": str(year),
            },
        }

    def fake_windows(path: Path) -> SpectralWindows:
        is_winner = path.name.startswith("winner_")
        good = is_winner == winner_is_better
        n_windows = 84
        n_channels = 24
        return SpectralWindows(
            year=np.full(n_windows, year, dtype=np.int16),
            t0=np.arange(n_windows, dtype=np.int32) * 24,
            energy_ratio=np.full(
                (n_windows, n_channels),
                0.98 if good else 0.70,
            ),
            shape_error=np.full(
                (n_windows, n_channels),
                0.04 if good else 0.30,
            ),
            coherence=np.full(
                (n_windows, n_channels),
                0.92 if good else 0.60,
            ),
            signed_cospectrum=np.full(
                (n_windows, n_channels),
                0.90 if good else 0.50,
            ),
            channels=CANONICAL_24_CHANNELS,
            tau=4,
        )

    monkeypatch.setattr(
        summary,
        "load_selection_spectral_metrics",
        fake_metrics,
    )
    monkeypatch.setattr(summary, "load_windows", fake_windows)
    monkeypatch.setattr(summary, "_file_sha256", lambda path: path.name)


def test_ood_spectral_report_confirms_clean_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, winner_is_better=True)

    report = summary.build_report(
        _selection(),
        Path("/unused"),
        ("winner", "challenger"),
        delta_t_hours=12,
        spectral_taus=(4,),
        draws=500,
    )

    assert report["no_significant_ood_spectral_regression"]
    assert report["significant_ood_spectral_regressions"] == []
    assert report["protocol"]["used_for_model_selection"] is False


def test_ood_spectral_report_rejects_pre_robust_selection() -> None:
    selection = _selection()
    selection["schema_version"] = 11

    with pytest.raises(ValueError, match="invalid frozen selection"):
        summary.build_report(
            selection,
            Path("/unused"),
            ("winner", "challenger"),
            delta_t_hours=12,
            spectral_taus=(4,),
        )


def test_ood_spectral_report_detects_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, winner_is_better=False)

    report = summary.build_report(
        _selection(),
        Path("/unused"),
        ("winner", "challenger"),
        delta_t_hours=12,
        spectral_taus=(4,),
        draws=500,
    )

    assert not report["no_significant_ood_spectral_regression"]
    assert {
        row["metric"]
        for row in report["significant_ood_spectral_regressions"]
    } == {
        "energy_log_error",
        "shape_log_error",
        "coherence",
        "signed_cospectrum",
    }


def test_ood_spectral_report_rejects_selection_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, winner_is_better=True)

    with pytest.raises(ValueError, match="model set"):
        summary.build_report(
            _selection(),
            Path("/unused"),
            ("winner",),
            delta_t_hours=12,
            spectral_taus=(4,),
        )


def test_ood_spectral_report_rejects_non_2021_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, winner_is_better=True, year=2020)

    with pytest.raises(ValueError, match="expected independent 2021"):
        summary.build_report(
            _selection(),
            Path("/unused"),
            ("winner", "challenger"),
            delta_t_hours=12,
            spectral_taus=(4,),
        )


def test_ood_spectral_report_rejects_noncanonical_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(
        monkeypatch,
        winner_is_better=True,
        canonical_channels=False,
    )

    with pytest.raises(ValueError, match="non-canonical spectral fields"):
        summary.build_report(
            _selection(),
            Path("/unused"),
            ("winner", "challenger"),
            delta_t_hours=12,
            spectral_taus=(4,),
        )
