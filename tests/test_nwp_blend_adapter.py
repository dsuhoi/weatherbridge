import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.eval.batch_eval_forecast_anchor import (
    CHANNELS_ORDER,
    ResidualBlendAdapter,
)
from tools.train.fit_nwp_blend_adapter import fit_adapter, fit_lead_scales
from tools.train.guard_nwp_blend_adapter import (
    analyze_query_guard,
    guard_adapter,
)


def _moments(taus: list[int], channels: int = 2):
    shape = (max(taus) + 1, channels)
    count = np.zeros(shape, dtype=np.int64)
    count[taus] = 10
    values = {
        "linear_sq": np.full(shape, np.nan),
        "delta_sq": np.full(shape, np.nan),
        "linear_delta": np.full(shape, np.nan),
        "linear_mean": np.full(shape, np.nan),
        "delta_mean": np.full(shape, np.nan),
        "count": count,
    }
    for name, value in values.items():
        if name != "count":
            value[taus] = 0.0
    return values


def test_adapter_fit_interpolates_held_out_tau_and_improves_validation() -> None:
    fit_taus = [1, 3, 5]
    eval_taus = [1, 2, 3, 4, 5]
    fit = _moments(fit_taus)
    validation = _moments(eval_taus)
    for moments, taus in ((fit, fit_taus), (validation, eval_taus)):
        moments["linear_sq"][taus] = 10.0
        moments["delta_sq"][taus] = 10.0
        moments["linear_delta"][taus] = -5.0

    gate, bias, diagnostics = fit_adapter(
        fit,
        validation,
        fit_taus,
        eval_taus,
    )

    np.testing.assert_allclose(gate[eval_taus], 0.5)
    np.testing.assert_allclose(bias[eval_taus], 0.0)
    assert diagnostics["validation_macro_rmse"]["adapted_all_taus"] < 1.0
    assert diagnostics["validation_macro_rmse"]["adapted_all_taus"] < diagnostics[
        "validation_macro_rmse"
    ]["linear_all_taus"]


def test_runtime_adapter_applies_channel_gate_and_normalized_bias(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "kind": "nwp_linear_residual_blend",
        "arch": "flow_pp3_detail",
        "delta_t_hours": 6,
        "channel_order": CHANNELS_ORDER,
        "base_checkpoint": {"sha256": checkpoint_sha},
        "gates": {str(tau): [0.5] * 24 for tau in range(1, 6)},
        "bias_normalized": {
            str(tau): [1.0] * 24 for tau in range(1, 6)
        },
    }
    adapter_path = tmp_path / "adapter.json"
    adapter_path.write_text(json.dumps(payload))

    class FakeModel:
        def __init__(self) -> None:
            self.checkpoint_metadata = {"epoch": 1}
            self.stats = SimpleNamespace(
                std=np.full(24, 2.0, dtype=np.float32)
            )

        def __call__(self, x0, x1, tau_h, dt):
            return np.full_like(x0, 10.0)

        def predict_taus(self, x0, x1, tau_hours, *, dt, chunk_size):
            assert chunk_size == len(tau_hours)
            return np.stack(
                [np.full_like(x0, 10.0) for _ in tau_hours]
            )

    adapter = ResidualBlendAdapter(
        FakeModel(),
        adapter_path,
        checkpoint,
        "flow_pp3_detail",
        6,
    )
    x0 = np.zeros((24, 1, 1), dtype=np.float32)
    x1 = np.full_like(x0, 2.0)

    predictions = adapter.predict_taus(
        x0,
        x1,
        [1, 3, 5],
        dt=6,
        chunk_size=3,
    )

    np.testing.assert_allclose(
        predictions[:, 0, 0, 0],
        [7.0 + 1.0 / 6.0, 7.5, 7.0 + 5.0 / 6.0],
    )


def test_lead_scale_selection_can_choose_exact_linear_fallback() -> None:
    eval_taus = [1, 2, 3, 4, 5]
    validation = {
        name: _moments(eval_taus)
        for name in ("fresh", "medium", "long")
    }
    for name, moments in validation.items():
        moments["linear_sq"][eval_taus] = 10.0
        moments["delta_sq"][eval_taus] = 10.0
        moments["linear_delta"][eval_taus] = (
            1.0 if name == "long" else -5.0
        )
    gate = np.zeros((6, 2), dtype=np.float64)
    gate[eval_taus] = 0.5
    bias = np.zeros_like(gate)

    records, selected_rmse = fit_lead_scales(
        validation,
        gate,
        bias,
        eval_taus,
    )

    assert records["fresh"]["scale"] == 1.0
    assert records["medium"]["scale"] == 1.0
    assert records["long"]["scale"] == 0.0
    assert np.isfinite(selected_rmse)


def test_lead_scale_selection_rejects_fragile_validation_gain() -> None:
    eval_taus = [1, 2, 3, 4, 5]
    validation = {
        name: _moments(eval_taus)
        for name in ("fresh", "medium", "long")
    }
    for moments in validation.values():
        moments["linear_sq"][eval_taus] = 1.0
        moments["delta_sq"][eval_taus] = 1.0e-6
        moments["linear_delta"][eval_taus] = -1.0e-6
    gate = np.zeros((6, 2), dtype=np.float64)
    gate[eval_taus] = 0.5
    bias = np.zeros_like(gate)

    records, selected_rmse = fit_lead_scales(
        validation,
        gate,
        bias,
        eval_taus,
        min_relative_rmse_gain=1.0e-3,
    )

    assert np.isfinite(selected_rmse)
    for record in records.values():
        assert record["unconstrained_scale"] == 1.0
        assert record["scale"] == 0.0
        assert record["linear_fallback_triggered"] is True


def test_lead_aware_runtime_uses_declared_bin_scale(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    payload = {
        "schema_version": 2,
        "kind": "nwp_linear_residual_blend",
        "arch": "flow_pp3",
        "delta_t_hours": 6,
        "channel_order": CHANNELS_ORDER,
        "base_checkpoint": {
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        },
        "gates": {str(tau): [0.5] * 24 for tau in range(1, 6)},
        "bias_normalized": {
            str(tau): [1.0] * 24 for tau in range(1, 6)
        },
        "lead_scale_bins": {
            "fresh": {"lower_hours": 0, "upper_hours": 48, "scale": 1.0},
            "medium": {"lower_hours": 48, "upper_hours": 120, "scale": 0.5},
            "long": {"lower_hours": 120, "upper_hours": 240, "scale": 0.0},
        },
    }
    adapter_path = tmp_path / "adapter.json"
    adapter_path.write_text(json.dumps(payload))

    class FakeModel:
        def __init__(self) -> None:
            self.checkpoint_metadata = {"epoch": 1}
            self.stats = SimpleNamespace(
                std=np.full(24, 2.0, dtype=np.float32)
            )

        def predict_taus(self, x0, x1, tau_hours, *, dt, chunk_size):
            return np.stack(
                [np.full_like(x0, 10.0) for _ in tau_hours]
            )

    adapter = ResidualBlendAdapter(
        FakeModel(),
        adapter_path,
        checkpoint,
        "flow_pp3",
        6,
    )
    x0 = np.zeros((24, 1, 1), dtype=np.float32)
    x1 = np.full_like(x0, 2.0)

    fresh = adapter.predict_taus(
        x0,
        x1,
        [3],
        dt=6,
        chunk_size=1,
        anchor_lead_hours=0,
    )[0]
    long = adapter.predict_taus(
        x0,
        x1,
        [3],
        dt=6,
        chunk_size=1,
        anchor_lead_hours=120,
    )[0]

    np.testing.assert_allclose(fresh, 7.5)
    np.testing.assert_allclose(long, 1.0)
    with pytest.raises(ValueError, match="requires anchor_lead_hours"):
        adapter.predict_taus(x0, x1, [3], dt=6, chunk_size=1)


def _write_guard_artifact(path, squared_error: np.ndarray) -> None:
    tau = np.asarray([1, 2, 5, 1, 2, 5], dtype=np.int8)
    np.savez(
        path,
        init_time_hours=np.asarray([0, 0, 0, 24, 24, 24]),
        anchor_lead_hours=np.asarray([0, 0, 0, 6, 6, 6]),
        tau_hours=tau,
        squared_error_norm=squared_error,
        channel_names=np.asarray(["Q850", "U850"]),
    )


def test_query_guard_removes_only_endpoint_development_regressions(tmp_path) -> None:
    linear = np.ones((6, 2), dtype=np.float32)
    adapted = np.full((6, 2), 0.8, dtype=np.float32)
    adapted[np.isin([1, 2, 5, 1, 2, 5], [1, 5])] = 1.01
    linear_path = tmp_path / "linear.npz"
    adapted_path = tmp_path / "adapted.npz"
    _write_guard_artifact(linear_path, linear)
    _write_guard_artifact(adapted_path, adapted)

    diagnostics = analyze_query_guard(linear_path, adapted_path, (1, 5))

    assert len(diagnostics["regressions_before_guard"]) == 4
    assert diagnostics["regressions_after_guard"] == []
    assert diagnostics["scores"]["all"]["adapted_after_guard_rmse"] < 1.0


def test_query_guard_rejects_unprotected_regression(tmp_path) -> None:
    linear = np.ones((6, 2), dtype=np.float32)
    adapted = np.full((6, 2), 0.8, dtype=np.float32)
    adapted[[1, 4], 0] = 1.01
    linear_path = tmp_path / "linear.npz"
    adapted_path = tmp_path / "adapted.npz"
    _write_guard_artifact(linear_path, linear)
    _write_guard_artifact(adapted_path, adapted)

    with pytest.raises(ValueError, match="outside guarded taus"):
        analyze_query_guard(linear_path, adapted_path, (1, 5))


def test_guard_adapter_zeroes_gate_and_bias_at_declared_taus() -> None:
    payload = {
        "schema_version": 2,
        "kind": "nwp_linear_residual_blend",
        "delta_t_hours": 6,
        "channel_order": ["a", "b"],
        "gates": {str(tau): [0.5, 0.5] for tau in range(1, 6)},
        "bias_normalized": {
            str(tau): [0.1, 0.1] for tau in range(1, 6)
        },
    }
    guarded = guard_adapter(
        payload,
        (1, 5),
        {"all_regressions_removed_by_exact_linear_guard": True},
        source_adapter={},
        linear_development={},
        adapted_development={},
        tool={},
    )

    assert guarded["gates"]["1"] == [0.0, 0.0]
    assert guarded["bias_normalized"]["5"] == [0.0, 0.0]
    assert guarded["gates"]["3"] == [0.5, 0.5]
    assert payload["gates"]["1"] == [0.5, 0.5]


def test_frozen_v4_adapter_has_endpoint_and_long_lead_guards() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "metrics"
        / "nwp_blend_adapters_6h_v4_endpoint_guard"
    )
    adapter_path = root / "flow.json"
    adapter = json.loads(adapter_path.read_text())
    selection = json.loads((root / "selection.json").read_text())

    assert adapter["query_guard_taus"] == [1, 5]
    for tau in ("1", "5"):
        assert adapter["gates"][tau] == [0.0] * len(CHANNELS_ORDER)
        assert adapter["bias_normalized"][tau] == [0.0] * len(
            CHANNELS_ORDER
        )
    assert any(value != 0.0 for value in adapter["gates"]["3"])
    assert adapter["lead_scale_bins"]["long"]["scale"] == 0.0
    assert selection["adapters"]["flow"]["sha256"] == hashlib.sha256(
        adapter_path.read_bytes()
    ).hexdigest()
    diagnostics = selection["postselection_query_guard"]["diagnostics"]
    assert diagnostics["regressions_before_guard"]
    assert diagnostics["regressions_after_guard"] == []
    assert 2022 in selection["confirmatory_years_unopened"]


def test_runtime_skips_zero_query_and_lead_corrections(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    gates = {str(tau): [0.5] * 24 for tau in range(1, 6)}
    biases = {str(tau): [0.0] * 24 for tau in range(1, 6)}
    for tau in ("1", "5"):
        gates[tau] = [0.0] * 24
    payload = {
        "schema_version": 2,
        "kind": "nwp_linear_residual_blend",
        "arch": "flow_pp3",
        "delta_t_hours": 6,
        "channel_order": CHANNELS_ORDER,
        "base_checkpoint": {
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        },
        "gates": gates,
        "bias_normalized": biases,
        "lead_scale_bins": {
            "fresh": {"lower_hours": 0, "upper_hours": 48, "scale": 1.0},
            "medium": {
                "lower_hours": 48,
                "upper_hours": 120,
                "scale": 1.0,
            },
            "long": {"lower_hours": 120, "upper_hours": 240, "scale": 0.0},
        },
    }
    adapter_path = tmp_path / "adapter.json"
    adapter_path.write_text(json.dumps(payload))

    class FakeModel:
        def __init__(self) -> None:
            self.checkpoint_metadata = {"epoch": 1}
            self.stats = SimpleNamespace(
                std=np.ones(24, dtype=np.float32)
            )
            self.calls: list[list[int]] = []

        def predict_taus(self, x0, x1, tau_hours, *, dt, chunk_size):
            self.calls.append(list(tau_hours))
            assert chunk_size == len(tau_hours)
            return np.stack([np.full_like(x0, 10.0) for _ in tau_hours])

    base = FakeModel()
    adapter = ResidualBlendAdapter(
        base,
        adapter_path,
        checkpoint,
        "flow_pp3",
        6,
    )
    x0 = np.zeros((24, 1, 1), dtype=np.float32)
    x1 = np.full_like(x0, 6.0)

    fresh = adapter.predict_taus(
        x0,
        x1,
        [1, 2, 3, 4, 5],
        dt=6,
        chunk_size=5,
        anchor_lead_hours=0,
    )
    assert base.calls == [[2, 3, 4]]
    np.testing.assert_array_equal(fresh[0], np.ones_like(x0))
    np.testing.assert_array_equal(fresh[4], np.full_like(x0, 5.0))

    base.calls.clear()
    long = adapter.predict_taus(
        x0,
        x1,
        [1, 2, 3, 4, 5],
        dt=6,
        chunk_size=5,
        anchor_lead_hours=120,
    )
    assert base.calls == []
    np.testing.assert_array_equal(
        long[:, 0, 0, 0], np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
    )
