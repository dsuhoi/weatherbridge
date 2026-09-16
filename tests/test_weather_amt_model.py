import torch
import torch.nn as nn

import weather_time_interp.model.weather_amt_model as weather_amt
from weather_time_interp.model.amt_upstream.flow_utils import (
    SphereConv2d,
    SphereConvTranspose2d,
    spherical_avg_pool2d,
    spherical_resize,
    warp,
)
from weather_time_interp.model.amt_upstream.raft import bilinear_sampler
from weather_time_interp.model.weather_amt_model import WeatherAMTModel
from weather_time_interp.model.weather_amt_residual_model import (
    WeatherAMTResidualModel,
)
from tools.eval.capmatched_loader import _build_net_for_checkpoint


def test_weather_amt_has_matched_capacity_and_multiflow_core() -> None:
    model = WeatherAMTModel()
    n_parameters = sum(parameter.numel() for parameter in model.parameters())

    assert n_parameters == 14_255_541
    assert model.corr_levels == 4
    assert model.radius == 3
    assert model.num_flows == 5
    assert model.n_static_features == 3
    assert model.decoder1.field_channels == 24


def test_weather_amt_residual_has_matched_capacity() -> None:
    model = WeatherAMTResidualModel()

    assert sum(parameter.numel() for parameter in model.parameters()) == (
        14_255_565
    )


def test_capmatched_loader_builds_weather_amt_residual() -> None:
    model = _build_net_for_checkpoint(
        "amt_residual",
        {},
        "unused-static-path.pt",
    )

    assert isinstance(model, WeatherAMTResidualModel)


def test_weather_amt_residual_starts_at_exact_linear_interpolation(
    monkeypatch,
) -> None:
    model = WeatherAMTResidualModel()
    x0 = torch.randn(2, 24, 17, 32)
    x1 = torch.randn_like(x0)
    tau = torch.tensor([0.2, 0.7])
    transported = torch.randn_like(x0)

    monkeypatch.setattr(
        WeatherAMTModel,
        "forward",
        lambda *args, **kwargs: transported,
    )
    prediction = model(x0, x1, tau)
    expected = (
        (1.0 - tau[:, None, None, None]) * x0
        + tau[:, None, None, None] * x1
    )

    torch.testing.assert_close(prediction, expected)


def test_weather_amt_residual_gate_receives_initial_gradient(
    monkeypatch,
) -> None:
    model = WeatherAMTResidualModel()
    x0 = torch.randn(2, 24, 17, 32)
    x1 = torch.randn_like(x0)
    tau = torch.tensor([0.3, 0.6])
    transported = torch.randn_like(x0)

    monkeypatch.setattr(
        WeatherAMTModel,
        "forward",
        lambda *args, **kwargs: transported,
    )
    model(x0, x1, tau).square().mean().backward()

    assert model.transport_gain.grad is not None
    assert torch.isfinite(model.transport_gain.grad).all()
    assert model.transport_gain.grad.abs().sum() > 0


def test_weather_amt_routes_static_only_to_synthesis_pyramid(
    monkeypatch,
) -> None:
    model = WeatherAMTModel()
    x0 = torch.randn(2, 24, 17, 32)
    x1 = torch.randn_like(x0)
    static = torch.randn(3, 17, 32)
    captured = {}

    def transport(
        anchor0: torch.Tensor,
        anchor1: torch.Tensor,
        tau: torch.Tensor,
        static_pad: torch.Tensor,
    ):
        captured["static"] = static_pad
        return (1.0 - tau) * anchor0 + tau * anchor1, {}

    monkeypatch.setattr(model, "_transport", transport)
    model(x0, x1, torch.full((2,), 0.4), static=static)

    assert model.feat_encoder.conv1.in_channels == 24
    assert model.encoder.pyramid1[0][0].in_channels == 27
    assert captured["static"].shape == (2, 3, 24, 32)
    torch.testing.assert_close(
        captured["static"][..., 3:20, :],
        static.expand(2, -1, -1, -1),
    )
    torch.testing.assert_close(
        model.static_pole_parity,
        torch.ones(3),
    )


def test_weather_amt_exact_endpoints_without_transport_evaluation() -> None:
    model = WeatherAMTModel()
    x0 = torch.randn(2, 24, 17, 32)
    x1 = torch.randn_like(x0)

    torch.testing.assert_close(model(x0, x1, torch.zeros(2)), x0)
    torch.testing.assert_close(model(x0, x1, torch.ones(2)), x1)


def test_weather_amt_runs_all_pairs_multiflow_and_crops_padding(
    monkeypatch,
) -> None:
    calls = {"correlation": 0}
    upstream = weather_amt.BidirCorrBlock

    class TrackingCorrelation(upstream):
        def __init__(self, *args, **kwargs):
            calls["correlation"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(weather_amt, "BidirCorrBlock", TrackingCorrelation)
    model = WeatherAMTModel().eval()
    x0 = torch.randn(1, 24, 68, 80)
    x1 = torch.randn_like(x0)

    with torch.no_grad():
        prediction, aux = model(
            x0,
            x1,
            torch.tensor([0.4]),
            return_aux=True,
        )

    assert calls["correlation"] == 1
    assert prediction.shape == x0.shape
    assert aux["flow0"].shape == (1, 5, 2, 68, 80)
    assert aux["flow1"].shape == (1, 5, 2, 68, 80)
    assert aux["blend"].shape == (1, 5, 68, 80)
    assert torch.isfinite(prediction).all()


def test_amt_feature_warp_wraps_longitude() -> None:
    fields = torch.arange(4, dtype=torch.float32).view(1, 1, 1, 4)
    flow = torch.zeros(1, 2, 1, 4)
    flow[:, 0] = 1.0

    torch.testing.assert_close(
        warp(fields, flow),
        torch.tensor([[[[1.0, 2.0, 3.0, 0.0]]]]),
    )


def test_amt_feature_warp_crosses_pole_at_antipode() -> None:
    fields = torch.tensor(
        [[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]]
    )
    flow = torch.zeros(1, 2, 2, 4)
    flow[:, 1, 0] = -1.0

    warped = warp(fields, flow)

    torch.testing.assert_close(
        warped[:, :, 0],
        torch.tensor([[[2.0, 3.0, 0.0, 1.0]]]),
    )


def test_amt_correlation_lookup_uses_spherical_boundaries() -> None:
    correlation = torch.tensor(
        [[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]]
    )
    coordinates = torch.tensor([[[[4.0, -1.0]]]])

    sampled = bilinear_sampler(correlation, coordinates)

    torch.testing.assert_close(sampled, torch.tensor([[[[2.0]]]]))


def test_amt_correlation_antipode_interpolates_odd_width() -> None:
    correlation = torch.tensor(
        [[[[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]]]
    )
    coordinates = torch.tensor([[[[0.0, -1.0]]]])

    sampled = bilinear_sampler(correlation, coordinates)

    torch.testing.assert_close(sampled, torch.tensor([[[[2.5]]]]))


def test_amt_spherical_correlation_lookup_has_finite_gradients() -> None:
    correlation = torch.tensor(
        [[[[0.0, 1.0, 4.0, 2.0], [3.0, 7.0, 5.0, 9.0]]]],
        requires_grad=True,
    )
    coordinates = torch.tensor(
        [[[[4.25, -0.75]]]],
        requires_grad=True,
    )

    bilinear_sampler(correlation, coordinates).square().sum().backward()

    assert correlation.grad is not None
    assert coordinates.grad is not None
    assert torch.isfinite(correlation.grad).all()
    assert torch.isfinite(coordinates.grad).all()
    assert correlation.grad.abs().sum() > 0
    assert coordinates.grad.abs().sum() > 0


def test_weather_amt_vector_pole_parity() -> None:
    model = WeatherAMTModel()
    negative = {
        index
        for index, parity in enumerate(model.field_pole_parity.tolist())
        if parity < 0
    }

    assert negative == {*range(4, 12), 21, 22}


def test_amt_sphere_conv_reads_antipodal_pole_row() -> None:
    convolution = SphereConv2d(
        1,
        1,
        kernel_size=3,
        padding=1,
        bias=False,
    )
    with torch.no_grad():
        convolution.weight.zero_()
        convolution.weight[0, 0, 0, 1] = 1.0
    fields = torch.tensor(
        [[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]]
    )

    output = convolution(fields)

    torch.testing.assert_close(
        output[:, :, 0],
        torch.tensor([[[2.0, 3.0, 0.0, 1.0]]]),
    )


def test_amt_sphere_conv_applies_input_vector_parity() -> None:
    convolution = SphereConv2d(
        1,
        1,
        kernel_size=3,
        padding=1,
        bias=False,
        pole_parity=torch.tensor([-1.0]),
    )
    with torch.no_grad():
        convolution.weight.zero_()
        convolution.weight[0, 0, 0, 1] = 1.0
    fields = torch.tensor(
        [[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]]
    )

    output = convolution(fields)

    torch.testing.assert_close(
        output[:, :, 0],
        torch.tensor([[[-2.0, -3.0, 0.0, -1.0]]]),
    )


def test_amt_sphere_conv_is_longitude_equivariant() -> None:
    torch.manual_seed(4)
    convolution = SphereConv2d(3, 5, kernel_size=3, padding=1)
    fields = torch.randn(2, 3, 8, 10)

    shifted = convolution(torch.roll(fields, 3, dims=-1))

    torch.testing.assert_close(
        shifted,
        torch.roll(convolution(fields), 3, dims=-1),
    )


def test_amt_spherical_upsample_is_longitude_equivariant() -> None:
    torch.manual_seed(5)
    fields = torch.randn(1, 3, 5, 7)

    shifted = spherical_resize(
        torch.roll(fields, 2, dims=-1),
        scale_factor=2.0,
    )

    torch.testing.assert_close(
        shifted,
        torch.roll(spherical_resize(fields, 2.0), 4, dims=-1),
    )


def test_amt_transpose_conv_is_longitude_equivariant() -> None:
    torch.manual_seed(6)
    convolution = SphereConvTranspose2d(3, 4)
    fields = torch.randn(1, 3, 5, 7)

    shifted = convolution(torch.roll(fields, 2, dims=-1))

    torch.testing.assert_close(
        shifted,
        torch.roll(convolution(fields), 4, dims=-1),
        atol=1.0e-6,
        rtol=1.0e-5,
    )


def test_amt_correlation_pool_keeps_odd_spherical_extent() -> None:
    fields = torch.arange(
        15,
        dtype=torch.float32,
    ).reshape(1, 1, 3, 5)

    pooled = spherical_avg_pool2d(fields)

    assert pooled.shape == (1, 1, 2, 3)
    torch.testing.assert_close(
        pooled[..., 0, -1],
        torch.tensor([[4.5]]),
    )


def test_weather_amt_input_padding_uses_poles_and_vector_parity() -> None:
    model = WeatherAMTModel()
    fields = torch.zeros(1, 24, 2, 4)
    fields[:, 0, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    fields[:, 4, 0] = fields[:, 0, 0]

    padded, crop = model._pad_to_multiple(fields, multiple=4)

    assert crop == (1, 2, 4)
    torch.testing.assert_close(
        padded[:, 0, 0],
        torch.tensor([[2.0, 3.0, 0.0, 1.0]]),
    )
    torch.testing.assert_close(
        padded[:, 4, 0],
        torch.tensor([[-2.0, -3.0, 0.0, -1.0]]),
    )


def test_weather_amt_multiple_of_eight_grid_is_not_extended() -> None:
    model = WeatherAMTModel()
    fields = torch.empty(1, 24, 16, 24)

    padded, crop = model._pad_to_multiple(fields)

    assert padded is fields
    assert crop == (0, 16, 24)


def test_weather_amt_has_no_planar_spatial_padding() -> None:
    model = WeatherAMTModel()
    planar_convolutions = [
        module
        for module in model.modules()
        if type(module) is nn.Conv2d
        and tuple(module.padding) != (0, 0)
    ]
    planar_transpose_convolutions = [
        module
        for module in model.modules()
        if type(module) is nn.ConvTranspose2d
    ]

    assert planar_convolutions == []
    assert planar_transpose_convolutions == []


def test_weather_amt_small_backward_is_finite() -> None:
    torch.manual_seed(7)
    model = WeatherAMTModel().train()
    x0 = torch.randn(1, 24, 32, 32, requires_grad=True)
    x1 = torch.randn(1, 24, 32, 32, requires_grad=True)

    prediction = model(x0, x1, torch.tensor([0.4]))
    prediction.square().mean().backward()

    assert x0.grad is not None
    assert x1.grad is not None
    assert torch.isfinite(x0.grad).all()
    assert torch.isfinite(x1.grad).all()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_weather_amt_is_longitude_equivariant_on_aligned_shift() -> None:
    torch.manual_seed(8)
    model = WeatherAMTModel().eval()
    x0 = torch.randn(1, 24, 32, 32)
    x1 = torch.randn_like(x0)

    with torch.no_grad():
        prediction = model(x0, x1, torch.tensor([0.4]))
        shifted = model(
            torch.roll(x0, 16, dims=-1),
            torch.roll(x1, 16, dims=-1),
            torch.tensor([0.4]),
        )

    torch.testing.assert_close(
        shifted,
        torch.roll(prediction, 16, dims=-1),
        atol=2.0e-4,
        rtol=2.0e-4,
    )
