import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tools.eval import capmatched_loader
from tools.eval.assess_temporal_expert_router import (
    block_bootstrap_gain,
    field_gates,
    noninferiority_gate,
    spectral_consistency,
)
from tools.eval.select_temporal_expert_route import (
    file_sha256,
    load_expert,
    select_default_expert,
    select_route,
)
from tools.train.build_temporal_router_checkpoint import build_checkpoint
from tools.train.temporal_router_checkpoint_status import verify
from weather_time_interp.metrics.physical_consistency import (
    SELECTION_DIAGNOSTICS,
)
from weather_time_interp.model.temporal_expert_router import (
    TemporalExpertRouter,
)


class _OffsetExpert(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(offset))

    def forward(self, x0, xT, tau, cond=None, static=None):
        del xT, tau, cond, static
        return x0 + self.offset, {}


def test_temporal_router_dispatches_mixed_tau_once_per_sample() -> None:
    router = TemporalExpertRouter(
        {"flow": _OffsetExpert(1.0), "upr": _OffsetExpert(2.0)},
        {1: "flow", 2: "upr", 3: "upr"},
        delta_t=4,
    )
    x0 = torch.zeros(3, 1, 2, 2)
    output = router(
        x0,
        torch.zeros_like(x0),
        torch.tensor([0.25, 0.5, 0.75]),
    )

    assert torch.equal(
        output[:, 0, 0, 0],
        torch.tensor([1.0, 2.0, 2.0]),
    )
    assert ".cpu()" not in inspect.getsource(TemporalExpertRouter.forward)


def test_temporal_router_rejects_unregistered_target() -> None:
    router = TemporalExpertRouter(
        {"upr": _OffsetExpert(0.0)},
        {1: "upr"},
        delta_t=4,
    )

    with pytest.raises(ValueError, match="no temporal expert route"):
        router(
            torch.zeros(1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1),
            torch.tensor([0.5]),
        )


def _write_field_artifact(
    root: Path,
    name: str,
    checkpoint: Path,
    rmse_by_tau: dict[int, float],
) -> Path:
    metrics_path = root / f"{name}.json"
    window_dir = root / "window_metrics"
    window_dir.mkdir(exist_ok=True)
    taus = np.tile(np.arange(1, 6), 8)
    model_rmse = np.asarray(
        [rmse_by_tau[int(tau)] for tau in taus],
        dtype=np.float64,
    )
    window_path = window_dir / f"{name}.npz"
    window_year = np.full(taus.size, 2020, dtype=np.int16)
    window_t0 = np.repeat(
        np.arange(8, dtype=np.int32) * 24 * 7,
        5,
    )
    np.savez(
        window_path,
        year=window_year,
        t0=window_t0,
        tau=taus,
        channel_names=np.asarray(["t2m"]),
        mse_norm_model=np.square(model_rmse)[:, None],
        mse_norm_bilinear=np.ones((taus.size, 1)),
    )
    index_text = "\n".join(
        f"{year},{t0},{tau}"
        for year, t0, tau in zip(window_year, window_t0, taus)
    )
    index_sha256 = hashlib.sha256(index_text.encode()).hexdigest()
    payload = {
        "checkpoint_provenance": {
            "sha256": file_sha256(checkpoint),
            "arch": name,
        },
        "channel_names": ["t2m"],
        "seen_tau": [1, 3, 5],
        "unseen_tau": [2, 4],
        "per_tau": {
            str(tau): {
                "model": {"rmse_norm_t2m": rmse},
                "bilinear": {"rmse_norm_t2m": 1.0},
            }
            for tau, rmse in rmse_by_tau.items()
        },
        "window_metrics_file": f"window_metrics/{name}.npz",
        "window_metrics_provenance": {
            "size_bytes": window_path.stat().st_size,
            "sha256": file_sha256(window_path),
            "index_sha256": index_sha256,
        },
        "evaluation_protocol": {
            "full_year": True,
            "rmse_reduction": (
                "spherical_strip_area_weighted_spatial_mean"
            ),
            "index_sha256": index_sha256,
        },
        "evaluation_input_provenance": {"static": "same"},
        "evaluation_dataset_provenance": {
            "root": "/synthetic",
            "years": [2020],
        },
    }
    metrics_path.write_text(json.dumps(payload))
    return metrics_path


def test_route_freezes_significant_flow_edges_only(tmp_path: Path) -> None:
    upr_checkpoint = tmp_path / "upr.ckpt"
    flow_checkpoint = tmp_path / "flow.ckpt"
    upr_checkpoint.write_bytes(b"upr")
    flow_checkpoint.write_bytes(b"flow")
    upr_metrics = _write_field_artifact(
        tmp_path,
        "upr",
        upr_checkpoint,
        {1: 0.10, 2: 0.08, 3: 0.08, 4: 0.08, 5: 0.10},
    )
    flow_metrics = _write_field_artifact(
        tmp_path,
        "flow",
        flow_checkpoint,
        {1: 0.09, 2: 0.09, 3: 0.09, 4: 0.09, 5: 0.09},
    )
    experts = {
        "upr": load_expert("upr", upr_metrics, upr_checkpoint),
        "flow": load_expert("flow", flow_metrics, flow_checkpoint),
    }

    assert select_default_expert(experts) == "upr"
    route, diagnostics = select_route(
        experts,
        default_expert="upr",
        min_relative_gain=0.005,
        alpha=0.05,
        bootstrap_draws=500,
        seed=7,
        block_days=7,
    )

    assert route == {
        1: "flow",
        2: "upr",
        3: "upr",
        4: "upr",
        5: "flow",
    }
    assert diagnostics[1]["paired_bootstrap_p_one_sided"] < 0.05
    assert diagnostics[1]["paired_bootstrap_p_holm"] < 0.05
    assert diagnostics[1]["candidate_tests"]["flow"]["p_holm"] >= (
        diagnostics[1]["candidate_tests"]["flow"]["p_one_sided"]
    )
    assert diagnostics[2]["selected"] == "upr"


def test_route_loader_rejects_tampered_window_metrics(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "upr.ckpt"
    checkpoint.write_bytes(b"upr")
    metrics = _write_field_artifact(
        tmp_path,
        "upr",
        checkpoint,
        {tau: 0.1 for tau in range(1, 6)},
    )
    payload = json.loads(metrics.read_text())
    window_path = metrics.parent / payload["window_metrics_file"]
    with window_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="window provenance"):
        load_expert("upr", metrics, checkpoint)


def _write_source_checkpoint(
    path: Path,
    *,
    arch: str,
    offset: float,
) -> None:
    torch.save(
        {
            "state_dict": {
                "net.offset": torch.tensor(offset),
            },
            "hyper_parameters": {
                "arch": arch,
                "delta_t": 4.0,
            },
            "global_step": 10,
            "epoch": 1,
        },
        path,
    )


def test_builder_and_loader_preserve_frozen_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow = tmp_path / "flow.ckpt"
    upr = tmp_path / "upr.ckpt"
    unused = tmp_path / "unused.ckpt"
    _write_source_checkpoint(flow, arch="dummy_flow", offset=1.0)
    _write_source_checkpoint(upr, arch="dummy_upr", offset=2.0)
    _write_source_checkpoint(unused, arch="dummy_unused", offset=3.0)
    selector = (
        Path(__file__).resolve().parents[1]
        / "tools/eval/select_temporal_expert_route.py"
    )
    route_path = tmp_path / "route.json"
    route_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "evidence_level": "frozen_full_year_2020_tau_router",
                "selection_year": 2020,
                "ood_year_loaded": False,
                "selection_code_sha256": file_sha256(selector),
                "route_by_tau": {"1": "flow", "2": "upr", "3": "upr"},
                "experts": {
                    "flow": {
                        "checkpoint_path": str(flow),
                        "checkpoint_sha256": file_sha256(flow),
                        "checkpoint_arch": "dummy_flow",
                    },
                    "upr": {
                        "checkpoint_path": str(upr),
                        "checkpoint_sha256": file_sha256(upr),
                        "checkpoint_arch": "dummy_upr",
                    },
                    "unused": {
                        "checkpoint_path": str(unused),
                        "checkpoint_sha256": file_sha256(unused),
                        "checkpoint_arch": "dummy_unused",
                    },
                },
            }
        )
    )
    packed = build_checkpoint(route_path)
    assert set(packed["hyper_parameters"]["router_experts"]) == {
        "flow",
        "upr",
    }
    packed_path = tmp_path / "router.ckpt"
    torch.save(packed, packed_path)
    static_path = tmp_path / "static.pt"
    torch.save(torch.zeros(3, 1, 1), static_path)

    def build_dummy(arch, net_state, static_path):
        del arch, net_state, static_path
        return _OffsetExpert(0.0)

    monkeypatch.setattr(
        capmatched_loader,
        "_build_net_for_checkpoint",
        build_dummy,
    )
    model, model_type = capmatched_loader.load_capmatched_checkpoint(
        packed_path,
        torch.device("cpu"),
        static_path=static_path,
    )
    x0 = torch.zeros(3, 1, 1, 1)
    output = model(
        x0,
        torch.zeros_like(x0),
        torch.tensor([0.25, 0.5, 0.75]),
    )

    assert model_type == "capmatched_temporal_expert_router"
    assert torch.equal(
        output[:, 0, 0, 0],
        torch.tensor([1.0, 2.0, 2.0]),
    )
    assert hashlib.sha256(route_path.read_bytes()).hexdigest() == packed[
        "hyper_parameters"
    ]["router_route_artifact_sha256"]
    status = verify(packed_path, {1, 2, 3})
    assert status["route_by_tau"] == {1: "flow", 2: "upr", 3: "upr"}


def _assessment_field(rmse_by_tau: dict[int, float]) -> dict:
    taus = np.tile(np.arange(1, 6), 8)
    values = np.asarray([rmse_by_tau[int(tau)] for tau in taus])
    per_tau = {}
    for tau, rmse in rmse_by_tau.items():
        model = {"rmse_norm_t2m": rmse, "acc_mean": 0.9}
        linear = {"rmse_norm_t2m": 1.0, "acc_mean": 0.0}
        for diagnostic in SELECTION_DIAGNOSTICS:
            model[f"physical_{diagnostic}"] = 0.9
            linear[f"physical_{diagnostic}"] = 1.0
        per_tau[str(tau)] = {
            "model": model,
            "bilinear": linear,
        }
    return {
        "payload": {
            "channel_names": ["t2m"],
            "seen_tau": [1, 3, 5],
            "unseen_tau": [2, 4],
            "per_tau": per_tau,
            "temporal_curvature_rmse_norm": {
                "model": 0.9,
                "bilinear": 1.0,
            },
        },
        "year": np.full(taus.size, 2021),
        "t0": np.repeat(np.arange(8) * 24 * 7, 5),
        "tau": taus,
        "mse": np.square(values)[:, None],
        "bilinear_mse": np.ones((taus.size, 1)),
        "channels": ("t2m",),
        "input_provenance_sha256": "paired-input",
        "dataset_provenance_sha256": "paired-dataset",
    }


def test_router_assessment_confirms_ood_gain_and_exact_dispatch(
    tmp_path: Path,
) -> None:
    upr = _assessment_field({tau: 0.10 for tau in range(1, 6)})
    flow = _assessment_field(
        {1: 0.09, 2: 0.11, 3: 0.11, 4: 0.11, 5: 0.09}
    )
    route = {1: "flow", 2: "upr", 3: "upr", 4: "upr", 5: "flow"}
    router = _assessment_field(
        {1: 0.09, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.09}
    )

    gain = block_bootstrap_gain(
        router,
        upr,
        draws=500,
        seed=9,
        block_hours=24 * 7,
    )
    gates = field_gates(
        router,
        {"upr": upr, "flow": flow},
        route,
        "upr",
    )

    assert gain["relative_gain"] > 0.005
    assert gain["p_one_sided"] < 0.05
    assert gain["delta_ci_high"] < 0
    assert gates["pass"] is True
    discontinuous = _assessment_field(
        {1: 0.09, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.09}
    )
    discontinuous["payload"]["temporal_curvature_rmse_norm"]["model"] = 1.1
    discontinuous_gates = field_gates(
        discontinuous,
        {"upr": upr, "flow": flow},
        route,
        "upr",
    )
    assert discontinuous_gates["pass"] is False

    router_spectra = tmp_path / "router"
    expert_spectra = tmp_path / "experts"
    router_spectra.mkdir()
    expert_spectra.mkdir()
    for tau, expert in route.items():
        payload = {
            "ell": np.arange(3),
            "gt_El": np.ones((1, 3)),
            "pred_El": np.full((1, 3), 1.0 + tau / 100),
            "window_year": np.full(2, 2020),
            "window_t0": np.arange(2),
        }
        np.savez(
            router_spectra / f"router_tau{tau}.npz",
            **payload,
        )
        np.savez(
            expert_spectra / f"{expert}_model_tau{tau}.npz",
            **payload,
        )
    spectra = spectral_consistency(
        router_spectra,
        "router",
        expert_spectra,
        {"upr": "upr_model", "flow": "flow_model"},
        route,
    )
    assert spectra["pass"] is True


def test_noninferiority_rejects_an_underpowered_point_estimate() -> None:
    result = {
        "candidate_mean": 1.004,
        "reference_mean": 1.0,
        "relative_gain": -0.004,
        "delta_mean": 0.004,
        "delta_ci_high": 0.02,
        "p_regression_one_sided": 0.2,
    }

    gate = noninferiority_gate(
        result,
        margin_relative=0.005,
        alpha=0.05,
    )

    assert gate["margin_absolute"] == pytest.approx(0.005)
    assert gate["pass"] is False
