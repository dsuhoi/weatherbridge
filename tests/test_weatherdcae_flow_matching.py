from __future__ import annotations

import torch

from tools.train.train_capacity_matched_6h import (
    FLOW_MATCHING_ARCHES,
    build_net,
)
from weather_time_interp.model.dcae_adaln_model import WeatherDCAEAdaLNModel
from weather_time_interp.model.dcae_flow_matching_model import (
    WeatherDCAEFlowMatchingModel,
)


def _model_kwargs() -> dict:
    return {
        "in_channels": 2,
        "out_channels": 2,
        "n_static_features": 1,
        "latent_channels": 4,
        "attention_head_dim": 4,
        "block_type": ("ResBlock", "ResBlock"),
        "qkv_multiscales": ((), ()),
        "lat_crop": 0,
        "block_out_channels": (8, 16),
        "layers_per_block": (1, 1),
        "time_emb_dim": 16,
        "time_freq_dim": 8,
    }


def test_weatherdcae_checkpoint_is_an_exact_fm_warm_start() -> None:
    torch.manual_seed(7)
    base = WeatherDCAEAdaLNModel(**_model_kwargs()).eval()
    fm = WeatherDCAEFlowMatchingModel(
        **_model_kwargs(),
        flow_matching=True,
        flow_matching_steps=4,
    ).eval()
    incompatible = fm.load_state_dict(base.state_dict(), strict=False)

    assert set(incompatible.missing_keys) == {
        "state_adapter.weight",
        "state_adapter.bias",
        "flow_time_mlp.mlp.0.weight",
        "flow_time_mlp.mlp.0.bias",
        "flow_time_mlp.mlp.2.weight",
        "flow_time_mlp.mlp.2.bias",
    }
    assert incompatible.unexpected_keys == []

    x0 = torch.randn(2, 2, 16, 32)
    xT = torch.randn_like(x0)
    static = torch.randn(2, 1, 16, 32)
    tau = torch.tensor([0.25, 0.75])
    with torch.no_grad():
        expected, _ = base(x0, xT, tau, None, static)
        actual, auxiliary = fm(x0, xT, tau, None, static)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert int(auxiliary["flow_matching_steps"].item()) == 4


def test_fm_endpoints_are_exact_and_new_branches_receive_gradients() -> None:
    model = WeatherDCAEFlowMatchingModel(**_model_kwargs(), flow_matching=True)
    x0 = torch.randn(2, 2, 16, 32)
    xT = torch.randn_like(x0)
    static = torch.randn(2, 1, 16, 32)
    tau = torch.tensor([0.0, 1.0])
    with torch.no_grad():
        endpoints, _ = model(x0, xT, tau, static=static)
    torch.testing.assert_close(endpoints[0], x0[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(endpoints[1], xT[1], rtol=0.0, atol=0.0)

    tau = torch.tensor([0.25, 0.75])
    tau_b = tau.view(-1, 1, 1, 1)
    linear = (1.0 - tau_b) * x0 + tau_b * xT
    bridge_state = linear + 0.1 * torch.randn_like(linear)
    velocity, _ = model.flow_matching_velocity(
        x0,
        xT,
        tau,
        bridge_state,
        torch.tensor([0.2, 0.8]),
        static=static,
    )
    velocity.square().mean().backward()

    assert velocity.shape == x0.shape
    assert model.state_adapter.weight.grad is not None
    assert model.flow_time_mlp.mlp[-1].weight.grad is not None
    assert torch.isfinite(model.state_adapter.weight.grad).all()


def test_fm_and_direct_control_have_identical_capacity() -> None:
    fm, _, _ = build_net("dcae_fm_14m", "")
    control, _, _ = build_net("dcae_fm_control_14m", "")

    fm_parameters = sum(parameter.numel() for parameter in fm.parameters())
    control_parameters = sum(parameter.numel() for parameter in control.parameters())
    assert fm_parameters == control_parameters
    assert 14_400_000 < fm_parameters < 14_600_000
    assert "dcae_fm_14m" in FLOW_MATCHING_ARCHES
    assert "dcae_fm_control_14m" not in FLOW_MATCHING_ARCHES
