"""Smoke tests for the 12 h evaluation infrastructure.

Covers the parts of ``tools/eval/batch_eval_12h_memmap.py`` that don't
require xarray / climatology / a real checkpoint:

  * pure-tensor helpers (``_bilinear_2anchor``, ``_catmull_rom_4anchor``)
    return shape-correct, finite outputs and match the analytical limits.
  * ``BatchModelRunner12h._finalize_payload`` emits a JSON payload with
    the schema declared in the script docstring.
  * ``per_tau_continuous_plot._avg_rmse_norm_per_tau`` averages the
    correct keys and skips channels missing from the JSON.

These tests do **not** spin up the full model — they exercise the
runner's accumulator and finalize logic with a stub climatology and
manually-seeded sums.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.eval.climatology import climatology_time_weights


@pytest.fixture(scope="module")
def m12h():
    """Import the 12 h batch-eval module (skip if heavy deps absent)."""
    try:
        mod = importlib.import_module("tools.eval.batch_eval_12h_memmap")
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"cannot import tools.eval.batch_eval_12h_memmap: {e}")
    return mod


@pytest.fixture(scope="module")
def plot_mod():
    try:
        mod = importlib.import_module("tools.eval.per_tau_continuous_plot")
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"cannot import tools.eval.per_tau_continuous_plot: {e}")
    return mod


# ----------------------------------------------------------------------- #
# Helper tensors                                                          #
# ----------------------------------------------------------------------- #


def _make_xy(B: int = 2, C: int = 3, H: int = 4, W: int = 5):
    torch.manual_seed(0)
    x_prev = torch.randn(B, C, H, W)
    x0 = torch.randn(B, C, H, W)
    x1 = torch.randn(B, C, H, W)
    x_next = torch.randn(B, C, H, W)
    return x_prev, x0, x1, x_next


# ----------------------------------------------------------------------- #
# Pure helpers                                                            #
# ----------------------------------------------------------------------- #


def test_bilinear_2anchor_endpoints(m12h):
    x_prev, x0, x1, x_next = _make_xy()
    B = x0.size(0)
    pred_0 = m12h._bilinear_2anchor(x0, x1, torch.zeros(B))
    pred_1 = m12h._bilinear_2anchor(x0, x1, torch.ones(B))
    assert torch.allclose(pred_0, x0, atol=1e-7)
    assert torch.allclose(pred_1, x1, atol=1e-7)


def test_bilinear_2anchor_midpoint(m12h):
    x_prev, x0, x1, x_next = _make_xy()
    B = x0.size(0)
    pred = m12h._bilinear_2anchor(x0, x1, torch.full((B,), 0.5))
    assert torch.allclose(pred, 0.5 * x0 + 0.5 * x1, atol=1e-7)


def test_evaluation_code_provenance_rejects_missing_source(
    m12h,
    tmp_path,
):
    existing = tmp_path / "existing.py"
    existing.write_text("value = 1\n")
    missing = tmp_path / "missing.py"

    with pytest.raises(FileNotFoundError, match="missing.py"):
        m12h._evaluation_code_provenance(
            {
                "existing.py": existing,
                "missing.py": missing,
            }
        )

    provenance = m12h._evaluation_code_provenance(
        {"existing.py": existing}
    )
    assert provenance == {
        "existing.py": m12h._sha256_file(existing),
    }


def test_evaluator_and_validator_share_source_manifest(m12h):
    from tools.eval.eval_artifact_status import _evaluation_source_paths

    sources = _evaluation_source_paths()
    provenance = m12h._evaluation_code_provenance(sources)

    assert set(provenance) == set(sources)
    assert len(provenance) >= 20


def test_catmull_rom_passes_through_anchors(m12h):
    """At τ=0 the curve must equal x0; at τ=1 it must equal x1."""
    x_prev, x0, x1, x_next = _make_xy()
    B = x0.size(0)
    p0 = m12h._catmull_rom_4anchor(x_prev, x0, x1, x_next, torch.zeros(B))
    p1 = m12h._catmull_rom_4anchor(x_prev, x0, x1, x_next, torch.ones(B))
    assert torch.allclose(p0, x0, atol=1e-6)
    assert torch.allclose(p1, x1, atol=1e-6)


def test_catmull_rom_linear_data_recovers_linear(m12h):
    """If x_{-1}=−1, x0=0, x1=1, x2=2 the spline must give exactly τ at the middle segment."""
    B, C, H, W = 1, 1, 2, 2
    x_prev = torch.full((B, C, H, W), -1.0)
    x0 = torch.full((B, C, H, W), 0.0)
    x1 = torch.full((B, C, H, W), 1.0)
    x_next = torch.full((B, C, H, W), 2.0)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        out = m12h._catmull_rom_4anchor(x_prev, x0, x1, x_next, torch.tensor([t]))
        assert torch.allclose(out, torch.full((B, C, H, W), t), atol=1e-6), (
            f"τ={t}: got {out.flatten()[0].item()}, expected {t}"
        )


def test_temporal_curvature_mse_detects_missing_acceleration(m12h):
    times = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1, 1)
    target = times.square().expand(5, 2, 3, 2, 4)
    prediction = torch.zeros_like(target)
    latitude_weights = torch.full((1, 1, 2, 1), 0.5)

    error = m12h.temporal_curvature_mse(
        prediction,
        target,
        latitude_weights,
    )
    exact = m12h.temporal_curvature_mse(
        target,
        target,
        latitude_weights,
    )

    assert error.shape == (2, 3)
    assert torch.allclose(error, torch.full_like(error, 4.0))
    assert torch.count_nonzero(exact) == 0


def test_temporal_curvature_mse_rejects_short_path(m12h):
    path = torch.zeros(2, 1, 1, 2, 2)
    weights = torch.full((1, 1, 2, 1), 0.5)

    with pytest.raises(ValueError, match="at least three"):
        m12h.temporal_curvature_mse(path, path, weights)


def test_climatology_window_precedes_evaluation(m12h):
    provenance = m12h.validate_climatology_window(
        {"climatology_window": "1990-2019"},
        evaluation_years=[2021, 2020],
        expected_window="1990-2019",
    )

    assert provenance == {
        "declared_window": "1990-2019",
        "start_year": 1990,
        "end_year": 2019,
        "evaluation_years": [2020, 2021],
        "precedes_evaluation": True,
    }


@pytest.mark.parametrize(
    ("attrs", "expected_window", "message"),
    [
        ({}, "1990-2019", "must be a YYYY-YYYY string"),
        (
            {"climatology_window": "1990/2019"},
            "1990-2019",
            "must be a YYYY-YYYY string",
        ),
        (
            {"climatology_window": "1991-2019"},
            "1990-2019",
            "window mismatch",
        ),
        (
            {"climatology_window": "1990-2020"},
            "1990-2020",
            "overlaps evaluation period",
        ),
    ],
)
def test_climatology_window_rejects_unverifiable_or_leaking_period(
    m12h,
    attrs,
    expected_window,
    message,
):
    with pytest.raises(ValueError, match=message):
        m12h.validate_climatology_window(
            attrs,
            evaluation_years=[2020],
            expected_window=expected_window,
        )


@pytest.mark.parametrize(
    ("day_of_year", "year", "expected_day1"),
    [
        (100, 2020, 100),
        (366, 2020, 0),
        (365, 2021, 0),
    ],
)
def test_climatology_late_hour_uses_next_calendar_day(
    day_of_year,
    year,
    expected_day1,
):
    h0, h1, day0, day1, weight = climatology_time_weights(
        day_of_year=day_of_year,
        hour=23.0,
        year=year,
        climatology_days=366,
    )

    assert (h0, h1) == (3, 0)
    assert day0 == day_of_year - 1
    assert day1 == expected_day1
    assert weight == pytest.approx(5.0 / 6.0)


def test_climatology_day_does_not_advance_before_last_slot():
    h0, h1, day0, day1, weight = climatology_time_weights(
        day_of_year=100,
        hour=17.0,
        year=2021,
        climatology_days=366,
    )

    assert (h0, h1, day0, day1) == (2, 3, 99, 99)
    assert weight == pytest.approx(5.0 / 6.0)


# ----------------------------------------------------------------------- #
# Payload schema (runner.finalize)                                        #
# ----------------------------------------------------------------------- #


class _StubClimatology:
    """Minimal stand-in for _ClimatologyLookup so the test doesn't open xarray."""

    def __init__(self):
        self.clim = torch.zeros(1, 4, 366, 1, 1)


def test_finalize_payload_schema(m12h):
    """``_finalize_payload`` must emit the contract used in the docstring."""

    # Build a minimal config-like object with just the fields the finalize
    # path reads.
    class _CfgStub:
        dt_hours = 12.0
        max_tau_hours = 12
        eval_hours = (1, 2, 4, 6, 8, 11)
        test_years = [2020]

    runner = m12h.BatchModelRunner12h.__new__(m12h.BatchModelRunner12h)
    runner.cfg = _CfgStub()
    runner.model = None
    runner.model_type = "dummy"
    runner.ckpt_path = "dummy.ckpt"
    runner.climatology = _StubClimatology()
    runner.acc_indices = [0]
    runner.seen_tau = [1, 2, 11]
    runner.unseen_tau = [4, 6, 8]
    runner.channel_names = ["chA", "chB"]
    runner.device = torch.device("cpu")

    runner._init_accumulators()

    # Seed the accumulators with controlled values for one τ.
    tau = 2
    runner._n_per_tau[tau] = 4
    for method in runner.method_names():
        for cname in runner.channel_names:
            runner._rmse_norm_sum_sq[method][tau][cname] = 16.0
            runner._rmse_phys_sum_sq[method][tau][cname] = 64.0
        # ACC accumulators: emulate a perfect correlation.
        runner._acc_sxy[method][tau].fill_(1.0)
        runner._acc_sxx[method][tau].fill_(1.0)
        runner._acc_syy[method][tau].fill_(1.0)

    payload = runner._finalize_payload()

    # Top-level fields.
    for key in [
        "checkpoint",
        "model_type",
        "delta_t_hours",
        "num_samples",
        "channel_names",
        "acc_channel_names",
        "seen_tau",
        "unseen_tau",
        "per_tau",
    ]:
        assert key in payload, f"missing key: {key}"
    assert payload["delta_t_hours"] == 12.0
    assert payload["num_samples"] == 4
    assert payload["seen_tau"] == [1, 2, 11]
    assert payload["unseen_tau"] == [4, 6, 8]

    # Only τ=2 has accumulated samples → only key in per_tau.
    assert set(payload["per_tau"].keys()) == {"2"}
    block = payload["per_tau"]["2"]
    assert set(block.keys()) == {"model", "bilinear", "bicubic"}
    for method_block in block.values():
        # RMSE per channel (norm + phys).
        for cname in runner.channel_names:
            assert math.isclose(
                method_block[f"rmse_norm_{cname}"], math.sqrt(16.0 / 4.0)
            )
            assert math.isclose(
                method_block[f"rmse_phys_{cname}"], math.sqrt(64.0 / 4.0)
            )
        # ACC on the one acc-channel ("chA" via acc_indices[0]).
        # 1e-9 epsilon in the denominator gives ~1e-9 offset from 1.0.
        assert math.isclose(method_block["acc_chA"], 1.0, abs_tol=1e-8)
        assert math.isclose(method_block["acc_mean"], 1.0, abs_tol=1e-8)


# ----------------------------------------------------------------------- #
# Plot helpers                                                            #
# ----------------------------------------------------------------------- #


def test_avg_rmse_norm_per_tau_picks_correct_keys(plot_mod):
    per_tau = {
        "1": {
            "model": {
                "rmse_norm_chA": 2.0,
                "rmse_norm_chB": 4.0,
                "rmse_phys_chA": 100.0,  # must be ignored
                "acc_chA": 0.9,
            },
            "bilinear": {"rmse_norm_chA": 3.0, "rmse_norm_chB": 5.0},
        },
        "2": {
            "model": {"rmse_norm_chA": 4.0},  # chB missing → averaged over chA only
        },
    }
    avg = plot_mod._avg_rmse_norm_per_tau(per_tau, "model", ["chA", "chB"])
    assert avg == {1: 3.0, 2: 4.0}


def test_degradation_zero_when_seen_equals_unseen(plot_mod):
    curve = {1: 1.0, 2: 1.0, 4: 1.0, 6: 1.0, 11: 1.0}
    deg = plot_mod._compute_degradation(curve, [1, 2, 11], [4, 6])
    assert deg == pytest.approx(0.0)


def test_degradation_positive_when_unseen_worse(plot_mod):
    curve = {1: 1.0, 2: 1.0, 4: 1.5, 6: 1.5, 11: 1.0}
    deg = plot_mod._compute_degradation(curve, [1, 2, 11], [4, 6])
    assert deg == pytest.approx(50.0)


def test_degradation_none_when_missing_tau(plot_mod):
    curve = {1: 1.0, 2: 1.0}
    deg = plot_mod._compute_degradation(curve, [1, 2], [4, 6])
    assert deg is None


# ----------------------------------------------------------------------- #
# JSON round-trip: emitted file is loadable and well-typed                #
# ----------------------------------------------------------------------- #


def test_payload_is_json_serialisable(m12h, tmp_path):
    """Sanity check: the dict we just built can be written + read back."""
    class _CfgStub:
        dt_hours = 12.0
        max_tau_hours = 12
        eval_hours = (1, 11)
        test_years = [2020]

    runner = m12h.BatchModelRunner12h.__new__(m12h.BatchModelRunner12h)
    runner.cfg = _CfgStub()
    runner.model = None
    runner.model_type = "dummy"
    runner.ckpt_path = "dummy.ckpt"
    runner.climatology = _StubClimatology()
    runner.acc_indices = [0]
    runner.seen_tau = [1, 11]
    runner.unseen_tau = []
    runner.channel_names = ["chA"]
    runner.device = torch.device("cpu")

    runner._init_accumulators()
    runner._n_per_tau[1] = 1
    for method in runner.method_names():
        runner._rmse_norm_sum_sq[method][1]["chA"] = 1.0
        runner._rmse_phys_sum_sq[method][1]["chA"] = 1.0
        runner._acc_sxy[method][1].fill_(0.5)
        runner._acc_sxx[method][1].fill_(1.0)
        runner._acc_syy[method][1].fill_(1.0)

    payload = runner._finalize_payload()
    path = tmp_path / "smoke.json"
    with open(path, "w") as f:
        json.dump(payload, f)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["per_tau"]["1"]["model"]["acc_chA"] == pytest.approx(0.5)
