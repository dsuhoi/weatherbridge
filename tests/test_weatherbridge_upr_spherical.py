import torch

import weather_time_interp.model.weatherbridge_upr_spherical_model as spherical_upr
from weather_time_interp.model.weatherbridge_upr_lite_model import (
    SphereConv2d as LegacySphereConv2d,
)
from weather_time_interp.model.weatherbridge_upr_lite_model import (
    WeatherBridgeUPRLiteModel,
)
from weather_time_interp.model.weatherbridge_upr_scaled_model import (
    upr_scaled_variant_kwargs,
)
from weather_time_interp.model.weatherbridge_upr_spherical_model import (
    AntipodalSphereConv2d,
    SphericalPyramidRefiner,
    WeatherBridgeUPRSphericalModel,
    endpoint_blend,
    spherical_resize,
    spherical_warp,
)


def _small_model(**overrides) -> WeatherBridgeUPRSphericalModel:
    kwargs = {
        "hidden": 16,
        "static_width": 4,
        "n_blocks": 1,
        "n_flow_modes": 3,
        "laplacian_detail": True,
        "query_conditioned": False,
        "cubic_trajectory": True,
        "global_tokens": 4,
        "global_token_dim": 8,
        "hydrostatic_coupling": True,
    }
    kwargs.update(overrides)
    return WeatherBridgeUPRSphericalModel(**kwargs)


def test_spherical_upr_preserves_scaled_capacity_and_state_contract() -> None:
    kwargs = upr_scaled_variant_kwargs("upr_implicit_global_14m")
    legacy = WeatherBridgeUPRLiteModel(**kwargs)
    spherical = WeatherBridgeUPRSphericalModel(**kwargs)

    assert sum(p.numel() for p in spherical.parameters()) == 14_261_409
    assert legacy.state_dict().keys() == spherical.state_dict().keys()
    assert {
        key: tuple(value.shape)
        for key, value in legacy.state_dict().items()
    } == {
        key: tuple(value.shape)
        for key, value in spherical.state_dict().items()
    }
    assert not any(
        isinstance(module, LegacySphereConv2d)
        for module in spherical.modules()
    )


def test_spherical_upr_supports_query_match_contract() -> None:
    kwargs = upr_scaled_variant_kwargs("upr_query_match_14m")
    legacy = WeatherBridgeUPRLiteModel(**kwargs)
    spherical = WeatherBridgeUPRSphericalModel(**kwargs)

    assert sum(p.numel() for p in spherical.parameters()) == 14_258_369
    assert legacy.state_dict().keys() == spherical.state_dict().keys()
    assert spherical.refiner.local_matching
    assert spherical.query_state_modulator is not None


def test_spherical_upr_input_stem_uses_vector_pole_parity() -> None:
    model = _small_model()
    stem = model.frame_stem.proj

    assert isinstance(stem, AntipodalSphereConv2d)
    assert stem.pole_parity is not None
    assert stem.pole_parity[4].item() == -1.0
    assert stem.pole_parity[0].item() == 1.0


def test_spherical_upr_refiner_uses_spherical_feature_warp(
    monkeypatch,
) -> None:
    model = _small_model()
    calls = {"count": 0}
    original = spherical_upr.spherical_warp

    def tracking_warp(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(spherical_upr, "spherical_warp", tracking_warp)
    feature = torch.randn(1, 16, 4, 8)
    state = torch.zeros_like(feature)
    flow = torch.zeros(1, 3, 12, 4, 8)
    static = torch.randn(1, 4, 4, 8)

    model.refiner(
        feature,
        feature,
        state,
        flow,
        static,
        torch.tensor([0.5]).reshape(1, 1, 1, 1),
    )

    assert isinstance(model.refiner, SphericalPyramidRefiner)
    assert calls["count"] == 2


def test_spherical_upr_warp_crosses_pole_with_vector_parity() -> None:
    scalar = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]
    )
    fields = torch.stack((scalar, scalar), dim=1)
    flow = torch.zeros(1, 2, 2, 4)
    flow[:, 1, 0] = -1.0

    warped = spherical_warp(
        fields,
        flow,
        pole_parity=torch.tensor((1.0, -1.0)),
    )

    torch.testing.assert_close(
        warped[:, 0, 0],
        torch.tensor([[2.0, 3.0, 0.0, 1.0]]),
    )
    torch.testing.assert_close(
        warped[:, 1, 0],
        torch.tensor([[-2.0, -3.0, 0.0, -1.0]]),
    )


def test_spherical_upr_warp_supports_repeated_pole_crossings() -> None:
    scalar = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]
    )
    fields = torch.stack((scalar, scalar), dim=1)
    flow = torch.zeros(1, 2, 2, 4)
    flow[:, 1, 0] = -5.0

    warped = spherical_warp(
        fields,
        flow,
        pole_parity=torch.tensor((1.0, -1.0)),
    )

    torch.testing.assert_close(
        warped[:, 0, 0],
        torch.tensor([[2.0, 3.0, 0.0, 1.0]]),
    )
    torch.testing.assert_close(
        warped[:, 1, 0],
        torch.tensor([[-2.0, -3.0, 0.0, -1.0]]),
    )


def test_spherical_upr_resize_is_longitude_equivariant() -> None:
    torch.manual_seed(3)
    fields = torch.randn(1, 3, 5, 8)

    output = spherical_resize(fields, (10, 16))
    shifted = spherical_resize(
        torch.roll(fields, 2, dims=-1),
        (10, 16),
    )

    torch.testing.assert_close(
        shifted,
        torch.roll(output, 4, dims=-1),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_spherical_upr_blend_has_no_clamp_dead_zone() -> None:
    logits = torch.tensor([[[[4.0]]]], requires_grad=True)
    tau = torch.tensor([[[[0.25]]]])

    alpha = endpoint_blend(logits, tau)
    alpha.sum().backward()

    assert 0.0 < alpha.item() < 1.0
    assert logits.grad is not None
    assert logits.grad.item() > 0.0
    torch.testing.assert_close(
        endpoint_blend(torch.zeros_like(logits), tau),
        1.0 - tau,
    )


def test_spherical_upr_forward_backward_and_exact_endpoints() -> None:
    torch.manual_seed(5)
    model = _small_model().train()
    x0 = torch.randn(2, 24, 32, 64, requires_grad=True)
    xT = torch.randn_like(x0)
    static = torch.randn(1, 3, 32, 64)
    tau = torch.tensor([0.0, 0.4])

    prediction, aux = model(x0, xT, tau, static=static)
    prediction.square().mean().backward()

    assert prediction.shape == x0.shape
    assert aux["flow_modes"].shape[:3] == (2, 3, 12)
    torch.testing.assert_close(prediction[0], x0[0], rtol=0, atol=0)
    assert x0.grad is not None
    assert torch.isfinite(x0.grad).all()


def test_spherical_upr_is_aligned_longitude_equivariant() -> None:
    torch.manual_seed(7)
    model = _small_model().eval()
    x0 = torch.randn(1, 24, 32, 64)
    xT = torch.randn_like(x0)
    static = torch.randn(1, 3, 32, 64)

    with torch.no_grad():
        prediction, _ = model(
            x0,
            xT,
            torch.tensor([0.4]),
            static=static,
        )
        shifted, _ = model(
            torch.roll(x0, 8, dims=-1),
            torch.roll(xT, 8, dims=-1),
            torch.tensor([0.4]),
            static=torch.roll(static, 8, dims=-1),
        )

    torch.testing.assert_close(
        shifted,
        torch.roll(prediction, 8, dims=-1),
        rtol=2.0e-4,
        atol=2.0e-4,
    )
