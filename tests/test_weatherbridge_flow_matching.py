from __future__ import annotations

import torch

from tools.train.train_capacity_matched_6h import build_net
from weather_time_interp.model.weatherbridge_flow_model import (
    SphericalLatentTransformer,
    WeatherBridgeModel,
)


def test_flow_matching_inference_preserves_both_anchors() -> None:
    model = WeatherBridgeModel(
        in_channels=24,
        out_channels=24,
        n_static_features=3,
        hidden=8,
        n_levels=2,
        use_accel=True,
        spectral_branch=False,
        hydro_couple=False,
        endpoint_preserving=True,
        anchor_detail_bypass=True,
        flow_matching=True,
        flow_matching_steps=2,
    ).eval()
    x0 = torch.randn(2, 24, 16, 32)
    xT = torch.randn_like(x0)
    static = torch.randn(2, 3, 16, 32)
    tau = torch.tensor([0.0, 1.0])

    with torch.no_grad():
        output, auxiliary = model(x0, xT, tau, static=static)

    torch.testing.assert_close(output[0], x0[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[1], xT[1], rtol=0.0, atol=0.0)
    assert int(auxiliary["flow_matching_steps"].item()) == 2


def test_flow_matching_velocity_accepts_a_bridge_state() -> None:
    model = WeatherBridgeModel(
        in_channels=24,
        out_channels=24,
        n_static_features=3,
        hidden=8,
        n_levels=2,
        use_accel=True,
        endpoint_preserving=True,
        anchor_detail_bypass=True,
        flow_matching=True,
    )
    x0 = torch.randn(2, 24, 16, 32)
    xT = torch.randn_like(x0)
    tau = torch.tensor([0.25, 0.75])
    tau_b = tau.view(-1, 1, 1, 1)
    bridge_state = (1.0 - tau_b) * x0 + tau_b * xT
    flow_time = torch.tensor([0.2, 0.8])

    velocity, _ = model.flow_matching_velocity(
        x0,
        xT,
        tau,
        bridge_state,
        flow_time,
        static=torch.zeros(2, 3, 16, 32),
    )

    assert velocity.shape == x0.shape
    assert torch.isfinite(velocity).all()


def test_detail_bypass_alone_is_exactly_zero_at_both_anchors() -> None:
    common = {
        "in_channels": 24,
        "out_channels": 24,
        "n_static_features": 3,
        "hidden": 8,
        "n_levels": 2,
        "use_accel": True,
        "spectral_branch": False,
        "hydro_couple": False,
        "endpoint_preserving": False,
    }
    base = WeatherBridgeModel(**common, anchor_detail_bypass=False).eval()
    detail = WeatherBridgeModel(**common, anchor_detail_bypass=True).eval()
    incompatible = detail.load_state_dict(base.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert all(key.startswith("detail_gain_head.") for key in incompatible.missing_keys)
    detail.detail_gain_head[-1].bias.data.fill_(1.0)

    x0 = torch.randn(2, 24, 16, 32)
    xT = torch.randn_like(x0)
    static = torch.randn(2, 3, 16, 32)
    tau = torch.tensor([0.0, 1.0])

    with torch.no_grad():
        base_output, _ = base(x0, xT, tau, static=static)
        detail_output, auxiliary = detail(x0, xT, tau, static=static)

    torch.testing.assert_close(detail_output, base_output, rtol=0.0, atol=0.0)
    assert auxiliary["detail_bypass_correction"] is not None


def test_universal_variants_have_no_field_class_specific_controls() -> None:
    for arch, flow_matching in (
        ("flow_universal_detail", False),
        ("flow_universal_latent", False),
        ("flow_universal_latent_refine", False),
        ("flow_universal_content_refine", False),
        ("flow_universal_pareto_refine", False),
        ("flow_universal_fm", True),
    ):
        model, _, _ = build_net(arch, "")

        assert model.flow_matching is flow_matching
        assert model.shared_field_controls is True
        assert model.blend_head.out_channels == 1
        assert model.warp_gate.numel() == 1
        assert model.scale.numel() == 1
        assert model.detail_gain_head[-1].out_features == 1
        assert model.hydro is None
        if arch in {
            "flow_universal_latent",
            "flow_universal_latent_refine",
            "flow_universal_content_refine",
            "flow_universal_pareto_refine",
            "flow_universal_fm",
        }:
            assert isinstance(model.global_mixer, SphericalLatentTransformer)
        if arch in {
            "flow_universal_latent_refine",
            "flow_universal_content_refine",
            "flow_universal_pareto_refine",
        }:
            assert len(model.dec[0]) == 2
        if arch == "flow_universal_content_refine":
            assert model.content_adaptive_controls
            assert model.content_control_head[0].in_channels == 8
            assert model.content_control_head[-1].out_channels == 2
        elif arch == "flow_universal_pareto_refine":
            assert model.time_content_adaptive_controls
            assert model.content_control_head.content.in_channels == 8
            assert model.content_control_head.output.out_channels == 4
        else:
            assert not model.content_adaptive_controls


def test_content_adaptive_controls_are_endpoint_exact_and_fieldwise() -> None:
    model = WeatherBridgeModel(
        in_channels=24,
        out_channels=24,
        n_static_features=3,
        hidden=8,
        n_levels=2,
        use_accel=True,
        endpoint_preserving=True,
        shared_field_controls=True,
        content_adaptive_controls=True,
    ).eval()
    x0 = torch.randn(2, 24, 16, 32)
    xT = torch.randn_like(x0)
    tau = torch.tensor([0.0, 1.0])

    with torch.no_grad():
        output, auxiliary = model(
            x0,
            xT,
            tau,
            static=torch.zeros(2, 3, 16, 32),
        )

    torch.testing.assert_close(output[0], x0[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[1], xT[1], rtol=0.0, atol=0.0)
    assert auxiliary["adaptive_controls"].shape == (2, 24, 2, 16, 32)


def test_time_content_controls_are_zero_init_and_endpoint_exact() -> None:
    model = WeatherBridgeModel(
        in_channels=24,
        out_channels=24,
        n_static_features=3,
        hidden=8,
        n_levels=2,
        use_accel=True,
        endpoint_preserving=True,
        shared_field_controls=True,
        anchor_detail_bypass=True,
        time_content_adaptive_controls=True,
    ).eval()
    x0 = torch.randn(2, 24, 16, 32)
    xT = torch.randn_like(x0)
    tau = torch.tensor([0.0, 1.0])

    with torch.no_grad():
        output, auxiliary = model(
            x0,
            xT,
            tau,
            static=torch.zeros(2, 3, 16, 32),
        )

    torch.testing.assert_close(output[0], x0[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[1], xT[1], rtol=0.0, atol=0.0)
    assert auxiliary["adaptive_controls"].shape == (2, 24, 4, 16, 32)
    assert torch.count_nonzero(auxiliary["adaptive_controls"]) == 0
    torch.testing.assert_close(
        auxiliary["adaptive_residual_gain"],
        torch.ones_like(auxiliary["adaptive_residual_gain"]),
    )


def test_spherical_latent_transformer_preserves_shape_and_backpropagates() -> None:
    mixer = SphericalLatentTransformer(
        channels=32,
        n_tokens=8,
        token_dim=32,
        time_dim=16,
        depth=2,
        n_heads=4,
    )
    x = torch.randn(2, 32, 4, 8, requires_grad=True)
    temb = torch.randn(2, 16)

    output = mixer(x, temb)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert mixer.project.weight.grad is not None
    assert torch.isfinite(mixer.project.weight.grad).all()
