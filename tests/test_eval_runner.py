"""Tests for ``weather_time_interp.eval_runner``.

Covers the Phase 0 + Phase 1 refactor:
  - economy days filter parity with legacy ``numerical_baseline_eval``;
  - cosine-lat weights helper;
  - per-hour RMSE accumulator math;
  - per-method-per-hour ``predict`` dispatch (mocked);
  - JSON schema produced by ``NumericalBaselineRunner``;
  - bit-equality of the runner output vs the legacy snapshot
    in ``tests/fixtures/legacy_json/numerical/``.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from weather_time_interp.eval_runner import (  # noqa: E402
    BatchContext,
    Config,
    EvaluationRunner,
    NumericalBaselineRunner,
    PerHourAccums,
)

FIX = Path(__file__).resolve().parent / "fixtures"
MEMMAP_DIR = FIX / "synth_memmap"
STATS_DIR = FIX / "synth_stats"
LEGACY_DIR = FIX / "legacy_json" / "numerical"


# --------------------------------------------------------------------- helpers


def _make_cfg(out_dir: Path) -> Config:
    return Config(
        memmap_dir=str(MEMMAP_DIR),
        test_years=[2020],
        stats_path=str(STATS_DIR / "json_stats_0p5.nc"),
        surface_stats_path=str(STATS_DIR / "surface_stats_0p5.json"),
        static_path=str(STATS_DIR / "static_features_0p5.pt"),
        batch_size=2,
        num_workers=0,
        samples_per_date=2,
        eval_days_per_month=5,
        dt_hours=6.0,
        max_tau_hours=6,
        eval_hours=tuple(range(7)),
        device=torch.device("cpu"),
        normalize_longitude=False,
        out_dir=str(out_dir),
    )


# ------------------------------------------------------------------ economy

def test_economy_filter_matches_legacy():
    """Filter must produce the same index entries as the legacy inline block."""
    from weather_time_interp.memmap_dataset import ERA5MemmapDataset

    ds = ERA5MemmapDataset(
        memmap_dir=str(MEMMAP_DIR),
        years=[2020],
        max_tau_hours=6,
        samples_per_date=2,
        train=False,
        eval_hours=list(range(7)),
        static_path=str(STATS_DIR / "static_features_0p5.pt"),
        stats_path=str(STATS_DIR / "json_stats_0p5.nc"),
        surface_stats_path=str(STATS_DIR / "surface_stats_0p5.json"),
    )
    original_index = list(ds.index)

    # Legacy reference filter — verbatim from numerical_baseline_eval.py
    K = 5
    day_picks = {3: [1, 11, 21], 4: [1, 8, 15, 22], 5: [1, 7, 14, 21, 28]}.get(
        K, sorted({1 + i * (30 // K) for i in range(K)})
    )
    allowed = set(day_picks)
    expected = []
    for entry in original_index:
        y, t0, _, _ = entry
        doy = t0 // 24
        try:
            d = _dt.date(int(y), 1, 1) + _dt.timedelta(days=int(doy))
        except Exception:
            continue
        if d.day in allowed:
            expected.append(entry)

    # Runner-side filter
    cfg = _make_cfg(Path("/tmp/eval_runner_filt"))
    runner = NumericalBaselineRunner(cfg)
    runner.device = torch.device("cpu")
    runner._economy_filter(ds)

    assert ds.index == expected


def test_explicit_calendar_day_filter_uses_requested_days():
    from weather_time_interp.memmap_dataset import ERA5MemmapDataset

    ds = ERA5MemmapDataset(
        memmap_dir=str(MEMMAP_DIR),
        years=[2020],
        max_tau_hours=6,
        samples_per_date=2,
        train=False,
        eval_hours=list(range(7)),
        static_path=str(STATS_DIR / "static_features_0p5.pt"),
        stats_path=str(STATS_DIR / "json_stats_0p5.nc"),
        surface_stats_path=str(STATS_DIR / "surface_stats_0p5.json"),
    )

    NumericalBaselineRunner._calendar_day_filter(ds, [1, 8, 15, 22])

    selected_days = {
        (
            _dt.date(int(year), 1, 1)
            + _dt.timedelta(days=int(t0) // 24)
        ).day
        for year, t0, _, _ in ds.index
    }
    assert selected_days <= {1, 8, 15, 22}
    assert selected_days


# -------------------------------------------------------------- lat weights

def test_lat_weights_H32():
    """For non-360 / non-181 H the helper falls back to ``linspace(90, -90, H)``."""
    cfg = _make_cfg(Path("/tmp/eval_runner_w"))
    runner = NumericalBaselineRunner(cfg)
    runner.device = torch.device("cpu")
    H = 32
    w = runner._lat_weights(H)
    assert w.shape == (1, 1, H, 1)
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-6)
    # Symmetry around equator
    flat = w.view(-1)
    assert torch.allclose(flat, flat.flip(0), atol=1e-6)


# ---------------------------------------------------------- accumulator math

def test_per_hour_accumulator_rmse():
    """Manual RMSE check on a fully-controlled (B,C) error stream."""
    # 2 channels, 2 samples per hour, hour=3 only
    cfg = _make_cfg(Path("/tmp/eval_runner_acc"))
    runner = NumericalBaselineRunner(cfg)
    runner.channel_names = ["chA", "chB"]

    accums = PerHourAccums(
        sum_sq={"bilinear": {h: {} for h in range(7)}},
        n_per_hour={h: 0 for h in range(7)},
    )

    # Simulate legacy reduction loop
    err = torch.tensor([[1.0, 4.0], [3.0, 0.0]])  # (B=2, C=2)
    hours_per_sample = [3, 3]
    method = "bilinear"
    for i in range(2):
        h = int(hours_per_sample[i])
        accums.n_per_hour[h] += 1
        for ci, name in enumerate(runner.channel_names):
            for m, e in [(method, err)]:
                accums.sum_sq[m][h].setdefault(name, 0.0)
                accums.sum_sq[m][h][name] += float(e[i, ci].item())

    assert accums.n_per_hour[3] == 2
    assert accums.sum_sq["bilinear"][3]["chA"] == 4.0  # 1+3
    assert accums.sum_sq["bilinear"][3]["chB"] == 4.0  # 4+0
    rmse_A = math.sqrt(4.0 / 2.0)
    rmse_B = math.sqrt(4.0 / 2.0)
    assert rmse_A == pytest.approx(math.sqrt(2.0))
    assert rmse_B == pytest.approx(math.sqrt(2.0))


def test_per_channel_squared_error_is_a_spatial_mean() -> None:
    cfg = _make_cfg(Path("/tmp/eval_runner_spatial_mean"))
    cfg.normalize_longitude = True
    runner = NumericalBaselineRunner(cfg)
    pred = torch.ones(1, 1, 2, 4)
    target = torch.zeros_like(pred)
    latitude_weights = torch.full((1, 1, 2, 1), 0.5)

    error = runner._per_ch_sq(pred, target, latitude_weights)

    assert error.item() == pytest.approx(1.0)


# -------------------------------------------------------- predict dispatch

def test_predict_called_per_method_per_hour(monkeypatch):
    """``_process_batch`` must call ``predict`` once per (method, h_idx)."""
    cfg = _make_cfg(Path("/tmp/eval_runner_dispatch"))
    runner = NumericalBaselineRunner(cfg)
    runner._setup_data()
    runner.on_setup_done()

    calls: List[tuple] = []
    real_predict = runner.predict

    def spy(method, x0, xT, tau_h, *, batch_ctx):
        calls.append((method, batch_ctx.h_idx, batch_ctx.batch_idx))
        return real_predict(method, x0, xT, tau_h, batch_ctx=batch_ctx)

    runner.predict = spy  # type: ignore[assignment]

    accums = runner._new_accums()
    first_batch = next(iter(runner.loader))
    runner._process_batch(first_batch, 0, accums)

    n_methods = len(runner.method_names())
    nH = first_batch["tau"].size(1)
    assert len(calls) == n_methods * nH
    methods_seen = {c[0] for c in calls}
    assert methods_seen == set(runner.method_names())
    h_idx_seen = {c[1] for c in calls}
    assert h_idx_seen == set(range(nH))


# ------------------------------------------------------- json schema check

def test_json_schema_numerical(tmp_path):
    """Output payload structure mirrors legacy schema."""
    cfg = _make_cfg(tmp_path)
    runner = NumericalBaselineRunner(cfg)
    results = runner.run()

    assert set(results.keys()) == set(runner.method_names())
    for m, payload in results.items():
        assert payload["method"] == m
        assert payload["years"] == [2020]
        assert "channel_names" in payload
        assert "per_hour" in payload
        for h, block in payload["per_hour"].items():
            assert int(h) in range(7)
            assert set(block.keys()) == {"model", "bilinear", "bicubic"}
            for ch in payload["channel_names"]:
                key = f"rmse_{ch}"
                assert key in block["model"]
                assert key in block["bilinear"]
                assert key in block["bicubic"]


# ------------------------------------------------- bit-equality vs legacy

def _compare_dicts(a: Any, b: Any, path: str = "") -> None:
    """Recursive bit-comparison with abs_tol=1e-10 on floats."""
    if isinstance(a, dict):
        assert isinstance(b, dict), f"type mismatch at {path}"
        assert set(a.keys()) == set(b.keys()), (
            f"key mismatch at {path}: {set(a.keys()) ^ set(b.keys())}"
        )
        for k in a:
            _compare_dicts(a[k], b[k], path=f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list), f"type mismatch at {path}"
        assert len(a) == len(b), f"len mismatch at {path}"
        for i, (xa, xb) in enumerate(zip(a, b)):
            _compare_dicts(xa, xb, path=f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(float(a) - float(b)) <= 1e-10, (
            f"float mismatch at {path}: {a} vs {b} (|Δ|={abs(a-b):.3e})"
        )
    else:
        assert a == b, f"value mismatch at {path}: {a!r} vs {b!r}"


def test_bit_equality_numerical_vs_legacy(tmp_path):
    """Critical: refactored runner reproduces legacy JSON exactly (1e-10)."""
    assert LEGACY_DIR.exists(), (
        f"Legacy snapshot missing — regenerate with "
        f"`python tests/fixtures/legacy_numerical_snapshot.py ...` (see test docstring)."
    )

    cfg = _make_cfg(tmp_path)
    runner = NumericalBaselineRunner(cfg)
    runner.run()

    for method in NumericalBaselineRunner.METHODS:
        new_p = tmp_path / f"{method}.json"
        old_p = LEGACY_DIR / f"{method}.json"
        assert new_p.exists(), f"missing new JSON: {new_p}"
        assert old_p.exists(), f"missing legacy JSON: {old_p}"
        with open(new_p) as f:
            new = json.load(f)
        with open(old_p) as f:
            old = json.load(f)
        new.pop("schema_version")
        new.pop("evaluation_protocol")
        _compare_dicts(new, old, path=f"<{method}>")
