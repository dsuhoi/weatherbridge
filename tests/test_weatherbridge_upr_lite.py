import pytest
import torch

from tools.train.capmatched_to_bare import ARCH_META
from weather_time_interp.model.weatherbridge_upr_lite_model import (
    GlobalTokenMixer,
    PyramidRefiner,
    QueryStateModulator,
    WeatherBridgeUPRLiteModel,
    _coordinate_grid,
    _evaluate_trajectory,
    endpoint_preserving_blend,
    periodic_resize,
    periodic_warp,
    upr_lite_variant_kwargs,
)
from weather_time_interp.model.weatherbridge_upr_scaled_model import (
    upr_scaled_variant_kwargs,
)


@pytest.mark.parametrize(
    ("n_flow_modes", "laplacian_detail"),
    ((1, False), (1, True), (3, True)),
)
def test_upr_lite_forward_backward(n_flow_modes, laplacian_detail):
    model = WeatherBridgeUPRLiteModel(
        hidden=16,
        static_width=4,
        n_blocks=1,
        n_flow_modes=n_flow_modes,
        laplacian_detail=laplacian_detail,
    )
    x0 = torch.randn(2, 24, 32, 64, requires_grad=True)
    xT = torch.randn(2, 24, 32, 64)
    static = torch.randn(1, 3, 32, 64)
    tau = torch.tensor([0.25, 0.75])

    prediction, aux = model(x0, xT, tau, static=static)

    assert prediction.shape == x0.shape
    assert aux["flow_modes"].shape[:3] == (2, n_flow_modes, 4)
    prediction.square().mean().backward()
    assert x0.grad is not None
    assert torch.isfinite(x0.grad).all()


def test_continuous_variant_uses_one_trajectory_for_all_query_times():
    torch.manual_seed(11)
    model = WeatherBridgeUPRLiteModel(
        hidden=16,
        static_width=4,
        n_blocks=1,
        n_flow_modes=3,
        laplacian_detail=True,
        query_conditioned=False,
        quadratic_trajectory=True,
    ).eval()
    anchor0 = torch.randn(1, 24, 24, 48).expand(2, -1, -1, -1)
    anchorT = torch.randn(1, 24, 24, 48).expand(2, -1, -1, -1)

    with torch.no_grad():
        _, aux = model(anchor0, anchorT, torch.tensor([0.2, 0.8]))

    torch.testing.assert_close(
        aux["flow_modes"][0],
        aux["flow_modes"][1],
        rtol=0,
        atol=1e-6,
    )
    assert aux["flow_modes"].shape[2] == 8


def test_query_modulation_is_identity_at_initialization():
    modulator = QueryStateModulator(8)
    state = torch.randn(3, 8, 4, 6)
    tau = torch.tensor([0.2, 0.5, 0.8]).view(3, 1, 1, 1)

    output = modulator(state, tau)

    torch.testing.assert_close(output, state, rtol=0, atol=0)


def test_query_modulation_changes_synthesis_smoothly():
    modulator = QueryStateModulator(4)
    with torch.no_grad():
        modulator.scale_coefficients[0].fill_(0.25)
        modulator.shift_coefficients[1].fill_(0.1)
    state = torch.ones(3, 4, 2, 3)
    tau = torch.tensor([0.25, 0.5, 0.75]).view(3, 1, 1, 1)

    output = modulator(state, tau)

    assert torch.isfinite(output).all()
    assert not torch.equal(output[0], output[1])
    assert not torch.equal(output[1], output[2])


def test_local_matching_is_invariant_to_positive_feature_scale():
    left = torch.randn(2, 8, 4, 6)
    right = torch.randn(2, 8, 4, 6)

    reference = PyramidRefiner._matching_features(left, right)
    rescaled = PyramidRefiner._matching_features(3.0 * left, 0.5 * right)

    torch.testing.assert_close(reference, rescaled, rtol=1e-5, atol=1e-5)


def test_query_match_keeps_one_trajectory_for_all_query_times():
    torch.manual_seed(13)
    model = WeatherBridgeUPRLiteModel(
        hidden=16,
        static_width=4,
        n_blocks=1,
        n_flow_modes=3,
        laplacian_detail=True,
        query_conditioned=False,
        cubic_trajectory=True,
        global_tokens=4,
        global_token_dim=8,
        local_matching=True,
        query_state_modulation=True,
    ).eval()
    anchor0 = torch.randn(1, 24, 24, 48).expand(2, -1, -1, -1)
    anchorT = torch.randn(1, 24, 24, 48).expand(2, -1, -1, -1)

    with torch.no_grad():
        _, aux = model(anchor0, anchorT, torch.tensor([0.2, 0.8]))

    torch.testing.assert_close(
        aux["flow_modes"][0],
        aux["flow_modes"][1],
        rtol=0,
        atol=1e-6,
    )
    assert aux["flow_modes"].shape[2] == 12


def test_query_match_specific_paths_receive_gradient_after_head_warmup():
    torch.manual_seed(17)
    hidden = 8
    model = WeatherBridgeUPRLiteModel(
        hidden=hidden,
        static_width=4,
        n_blocks=1,
        n_flow_modes=3,
        laplacian_detail=True,
        query_conditioned=False,
        cubic_trajectory=True,
        global_tokens=4,
        global_token_dim=8,
        local_matching=True,
        query_state_modulation=True,
        periodic_longitude_resize=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    anchor0 = torch.randn(2, 24, 16, 32)
    anchorT = torch.randn(2, 24, 16, 32)
    target = torch.randn(2, 24, 16, 32)
    static = torch.randn(1, 3, 16, 32)
    tau = torch.tensor([0.25, 0.75])

    # The synthesis heads are zero-initialized. One update makes their
    # upstream QueryMatch paths observable to the loss.
    prediction, _ = model(anchor0, anchorT, tau, static=static)
    (prediction - target).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    prediction, _ = model(anchor0, anchorT, tau, static=static)
    (prediction - target).square().mean().backward()

    modulator = model.query_state_modulator
    assert modulator is not None
    assert modulator.scale_coefficients.grad is not None
    assert modulator.shift_coefficients.grad is not None
    assert modulator.scale_coefficients.grad.abs().sum() > 0
    assert modulator.shift_coefficients.grad.abs().sum() > 0

    input_grad = model.refiner.input_proj.weight.grad
    assert input_grad is not None
    matching_slice = input_grad[:, 3 * hidden:4 * hidden]
    assert matching_slice.abs().sum() > 0


def test_quadratic_trajectory_preserves_endpoint_displacements():
    trajectory = torch.zeros(3, 1, 8, 1, 1)
    trajectory[:, :, 0] = 2.0
    trajectory[:, :, 2] = 4.0
    trajectory[:, :, 4] = 6.0
    trajectory[:, :, 6] = 8.0
    tau = torch.tensor([0.0, 0.5, 1.0]).view(3, 1, 1, 1)

    forward, backward = _evaluate_trajectory(trajectory, tau)

    torch.testing.assert_close(
        forward[:, 0, 0, 0, 0],
        torch.tensor([0.0, 2.5, 2.0]),
    )
    torch.testing.assert_close(
        backward[:, 0, 0, 0, 0],
        torch.tensor([4.0, 4.0, 0.0]),
    )


def test_cubic_trajectory_is_endpoint_preserving_and_asymmetric():
    trajectory = torch.zeros(3, 1, 12, 1, 1)
    trajectory[:, :, 0] = 2.0
    trajectory[:, :, 2] = 4.0
    trajectory[:, :, 4] = 6.0
    trajectory[:, :, 6] = 8.0
    trajectory[:, :, 8] = 10.0
    trajectory[:, :, 10] = -12.0
    tau = torch.tensor([0.0, 0.25, 1.0]).view(3, 1, 1, 1)

    forward, backward = _evaluate_trajectory(trajectory, tau)

    torch.testing.assert_close(
        forward[:, 0, 0, 0, 0],
        torch.tensor([0.0, 0.6875, 2.0]),
    )
    torch.testing.assert_close(
        backward[:, 0, 0, 0, 0],
        torch.tensor([4.0, 5.625, 0.0]),
    )


def test_smooth_endpoint_blend_is_exact_and_never_hard_clamps():
    tau = torch.tensor([0.0, 1.0 / 6.0, 0.5, 5.0 / 6.0, 1.0]).view(
        5,
        1,
        1,
        1,
    )
    logits = torch.tensor([-5.0, 5.0, -5.0, -5.0, 5.0]).view(
        5,
        1,
        1,
        1,
    )

    alpha = endpoint_preserving_blend(logits, tau)

    assert alpha[0].item() == 1.0
    assert alpha[-1].item() == 0.0
    assert 0.0 < alpha[1].item() < 1.0
    assert 0.0 < alpha[2].item() < 1.0
    assert 0.0 < alpha[3].item() < 1.0


@pytest.mark.parametrize(
    ("n_flow_modes", "laplacian_detail"),
    ((1, False), (1, True), (3, True)),
)
def test_upr_lite_initializes_to_linear_and_preserves_anchors(
    n_flow_modes,
    laplacian_detail,
):
    torch.manual_seed(7)
    model = WeatherBridgeUPRLiteModel(
        hidden=16,
        static_width=4,
        n_blocks=1,
        n_flow_modes=n_flow_modes,
        laplacian_detail=laplacian_detail,
    ).eval()
    x0 = torch.randn(3, 24, 24, 48)
    xT = torch.randn(3, 24, 24, 48)
    tau = torch.tensor([0.0, 0.4, 1.0])

    with torch.no_grad():
        prediction, _ = model(x0, xT, tau)

    expected = (1.0 - tau[:, None, None, None]) * x0
    expected = expected + tau[:, None, None, None] * xT
    torch.testing.assert_close(prediction, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(prediction[0], x0[0], rtol=0, atol=2e-6)
    torch.testing.assert_close(prediction[2], xT[2], rtol=0, atol=2e-6)


def test_periodic_warp_wraps_longitude():
    x = torch.zeros(1, 1, 4, 8)
    x[:, :, :, 0] = 1.0
    flow = torch.zeros(1, 2, 4, 8)
    flow[:, 0] = 1.0

    warped = periodic_warp(x, flow)

    torch.testing.assert_close(warped[:, :, :, -1], torch.ones(1, 1, 4))


def test_periodic_warp_interpolates_across_fractional_seam():
    field = torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]])
    flow = torch.zeros(1, 2, 1, 4)
    flow[:, 0] = 0.5

    warped = periodic_warp(field, flow)

    expected = torch.tensor([[[[0.5, 1.5, 2.5, 1.5]]]])
    torch.testing.assert_close(warped, expected)


def test_periodic_warp_uses_precise_grid_for_bfloat16():
    x = torch.arange(16, dtype=torch.bfloat16).view(1, 1, 1, 16)
    flow = torch.full((1, 2, 1, 16), 0.25, dtype=torch.bfloat16)
    flow[:, 1] = 0

    warped = periodic_warp(x, flow)

    assert warped.dtype == torch.bfloat16
    assert warped[0, 0, 0, 4].float().item() == pytest.approx(4.25, abs=0.04)


def test_periodic_warp_reuses_coordinate_grid():
    _coordinate_grid.cache_clear()
    field = torch.randn(1, 2, 8, 16)
    flow = torch.zeros(1, 2, 8, 16)

    first = periodic_warp(field, flow)
    second = periodic_warp(field, flow)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert _coordinate_grid.cache_info().hits == 1


def test_grouped_column_warp_matches_per_channel_reference():
    torch.manual_seed(19)
    model = WeatherBridgeUPRLiteModel(
        hidden=16,
        static_width=4,
        n_blocks=1,
        n_flow_modes=3,
        laplacian_detail=True,
    )
    fields = torch.randn(2, 24, 8, 16)
    modes = torch.randn(2, 3, 2, 8, 16) * 0.25
    group_flows = model._group_flows(modes)
    channel_flows = group_flows[:, model.channel_to_group]

    grouped = model._warp_fields(fields, group_flows)
    reference = periodic_warp(fields, channel_flows)

    torch.testing.assert_close(grouped, reference, rtol=0, atol=1e-6)


def test_periodic_resize_is_longitude_shift_equivariant():
    torch.manual_seed(23)
    field = torch.randn(2, 3, 4, 8)

    reference = periodic_resize(field, (8, 16))
    shifted = periodic_resize(torch.roll(field, 2, dims=-1), (8, 16))

    torch.testing.assert_close(
        shifted,
        torch.roll(reference, 4, dims=-1),
        rtol=0,
        atol=1e-6,
    )


def test_resize_flow_preserves_displacement_units():
    model = WeatherBridgeUPRLiteModel(
        hidden=16,
        static_width=4,
        n_blocks=1,
        n_flow_modes=3,
        laplacian_detail=True,
        cubic_trajectory=True,
        periodic_longitude_resize=True,
    )
    flow = torch.empty(1, 3, 12, 4, 8)
    flow[:, :, 0::2].fill_(2.0)
    flow[:, :, 1::2].fill_(-3.0)

    resized = model._resize_flow(flow, (8, 16))

    torch.testing.assert_close(
        resized[:, :, 0::2],
        torch.full_like(resized[:, :, 0::2], 4.0),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        resized[:, :, 1::2],
        torch.full_like(resized[:, :, 1::2], -6.0),
        rtol=0,
        atol=1e-6,
    )


def test_continuous_medium_converter_metadata_matches_capacity_control():
    kwargs = ARCH_META["upr_lite_continuous_m"]["kwargs"]

    assert kwargs == upr_lite_variant_kwargs("upr_lite_continuous_m")
    assert kwargs["hidden"] == 192
    assert kwargs["static_width"] == 24
    assert kwargs["n_flow_modes"] == 3
    assert kwargs["laplacian_detail"]
    assert not kwargs["query_conditioned"]
    assert kwargs["quadratic_trajectory"]

    model = WeatherBridgeUPRLiteModel(**kwargs)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 1_500_000 < n_parameters < 1_700_000


def test_implicit_global_variant_is_cubic_and_globally_conditioned():
    kwargs = ARCH_META["upr_lite_implicit_global"]["kwargs"]

    assert kwargs == upr_lite_variant_kwargs("upr_lite_implicit_global")
    assert kwargs["cubic_trajectory"]
    assert not kwargs["quadratic_trajectory"]
    assert not kwargs["query_conditioned"]
    assert kwargs["global_tokens"] == 16
    assert kwargs["hydrostatic_coupling"]

    model = WeatherBridgeUPRLiteModel(**kwargs)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 1_600_000 < n_parameters < 2_000_000

    x0 = torch.randn(2, 24, 24, 48)
    xT = torch.randn(2, 24, 24, 48)
    tau = torch.tensor([0.0, 1.0])
    with torch.no_grad():
        prediction, aux = model(x0, xT, tau)

    assert aux["flow_modes"].shape[2] == 12
    assert aux["hydrostatic_delta"].shape == (2, 5, 24, 48)
    torch.testing.assert_close(prediction[0], x0[0], rtol=0, atol=2e-6)
    torch.testing.assert_close(prediction[1], xT[1], rtol=0, atol=2e-6)


def test_q4_variant_keeps_capacity_and_stops_motion_pyramid_at_quarter_grid():
    kwargs = ARCH_META["upr_lite_implicit_global_q4"]["kwargs"]

    assert kwargs == upr_lite_variant_kwargs("upr_lite_implicit_global_q4")
    assert kwargs["pyramid_divisors"] == (16, 8, 4)
    model = WeatherBridgeUPRLiteModel(**kwargs).eval()
    reference = WeatherBridgeUPRLiteModel(
        **upr_lite_variant_kwargs("upr_lite_implicit_global")
    )
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in reference.parameters()
    )

    refiner_sizes = []
    handle = model.refiner.register_forward_hook(
        lambda _module, inputs, _output: refiner_sizes.append(inputs[0].shape[-2:])
    )
    with torch.no_grad():
        prediction, _ = model(
            torch.randn(1, 24, 32, 64),
            torch.randn(1, 24, 32, 64),
            torch.tensor([0.5]),
            static=torch.randn(1, 3, 32, 64),
        )
    handle.remove()

    assert prediction.shape == (1, 24, 32, 64)
    assert refiner_sizes == [(2, 4), (4, 8), (8, 16)]


def test_scaled_implicit_global_matches_pp3_parameter_budget():
    kwargs = ARCH_META["upr_implicit_global_14m"]["kwargs"]

    assert kwargs == upr_scaled_variant_kwargs("upr_implicit_global_14m")
    assert kwargs["hidden"] == 504
    assert kwargs["n_blocks"] == 10
    assert kwargs["global_tokens"] == 32
    assert kwargs["global_token_dim"] == 112
    assert kwargs["cubic_trajectory"]
    assert kwargs["hydrostatic_coupling"]

    model = WeatherBridgeUPRLiteModel(**kwargs)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert n_parameters == 14_261_409


def test_universal_latent_q4_is_shared_area_aware_and_sub_10m():
    kwargs = ARCH_META["upr_universal_latent_q4_10m"]["kwargs"]

    assert kwargs == upr_scaled_variant_kwargs("upr_universal_latent_q4_10m")
    assert kwargs["pyramid_divisors"] == (16, 8, 4)
    assert kwargs["n_flow_modes"] == 1
    assert not kwargs["hydrostatic_coupling"]
    assert kwargs["global_area_weighted"]
    assert kwargs["global_transformer_depth"] == 2
    model = WeatherBridgeUPRLiteModel(**kwargs).eval()
    assert isinstance(model.global_mixer, GlobalTokenMixer)
    assert model.global_mixer.area_weighted
    assert model.global_mixer.transformer is not None
    assert len(model.global_mixer.transformer) == 2
    assert sum(parameter.numel() for parameter in model.parameters()) == 9_500_460

    with torch.no_grad():
        prediction, auxiliary = model(
            torch.randn(1, 24, 32, 64),
            torch.randn(1, 24, 32, 64),
            torch.tensor([0.5]),
            static=torch.randn(1, 3, 32, 64),
        )
    assert prediction.shape == (1, 24, 32, 64)
    assert auxiliary["flow_modes"].shape[:3] == (1, 1, 12)


def test_query_match_scaled_variant_matches_pp3_parameter_budget():
    kwargs = ARCH_META["upr_query_match_14m"]["kwargs"]

    assert kwargs == upr_scaled_variant_kwargs("upr_query_match_14m")
    assert kwargs["hidden"] == 500
    assert kwargs["global_token_dim"] == 96
    assert kwargs["local_matching"]
    assert kwargs["query_state_modulation"]
    assert kwargs["periodic_longitude_resize"]
    model = WeatherBridgeUPRLiteModel(**kwargs)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())

    assert n_parameters == 14_258_369
    assert abs(n_parameters - 14_260_565) / 14_260_565 < 0.001


def test_local_correlation_wraps_longitude_seam() -> None:
    refiner = PyramidRefiner(
        hidden=4,
        static_width=1,
        n_flow_modes=1,
        n_blocks=1,
        motion_components=4,
        local_correlation_radius=1,
        matching_width=4,
    )
    assert refiner.match_proj is not None
    with torch.no_grad():
        refiner.match_proj.weight.zero_()
        for channel in range(4):
            refiner.match_proj.weight[channel, channel, 0, 0] = 1.0
    left = torch.zeros(1, 4, 3, 5)
    right = torch.zeros_like(left)
    left[:, 0, 1, 0] = 1.0
    right[:, 0, 1, -1] = 1.0

    correlation = refiner._local_correlation(left, right)

    assert correlation.shape == (1, 9, 3, 5)
    torch.testing.assert_close(
        correlation[0, 3, 1, 0],
        torch.tensor(1.0),
    )


def test_local_correlation_scaled_variant_is_capacity_matched() -> None:
    kwargs = ARCH_META["upr_local_corr_14m"]["kwargs"]

    assert kwargs == upr_scaled_variant_kwargs("upr_local_corr_14m")
    assert kwargs["local_correlation_radius"] == 2
    assert kwargs["matching_width"] == 32
    assert not kwargs["local_matching"]
    assert kwargs["periodic_longitude_resize"]
    model = WeatherBridgeUPRLiteModel(**kwargs)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())

    assert n_parameters == 14_290_137
    assert abs(n_parameters - 14_260_565) / 14_260_565 < 0.005


def test_query_match_inference_loader_avoids_mutable_trainer(monkeypatch):
    from tools.eval.capmatched_loader import _build_net_for_checkpoint
    from tools.train import train_capacity_matched_6h as trainer

    def fail_build(*args, **kwargs):
        del args, kwargs
        raise AssertionError("mutable training entrypoint was used")

    monkeypatch.setattr(trainer, "build_net", fail_build)
    model = _build_net_for_checkpoint(
        "upr_query_match_14m",
        {},
        "unused-static-path",
    )

    assert isinstance(model, WeatherBridgeUPRLiteModel)
    assert model.local_matching is True
    assert model.query_state_modulation is True


def test_local_corr_inference_loader_avoids_mutable_trainer(monkeypatch):
    from tools.eval.capmatched_loader import _build_net_for_checkpoint
    from tools.train import train_capacity_matched_6h as trainer

    def fail_build(*args, **kwargs):
        del args, kwargs
        raise AssertionError("mutable training entrypoint was used")

    monkeypatch.setattr(trainer, "build_net", fail_build)
    model = _build_net_for_checkpoint(
        "upr_local_corr_14m",
        {},
        "unused-static-path",
    )

    assert isinstance(model, WeatherBridgeUPRLiteModel)
    assert model.local_correlation_radius == 2
    assert model.local_matching is False


def test_query_match_initializes_to_linear_and_preserves_anchors():
    torch.manual_seed(37)
    kwargs = upr_scaled_variant_kwargs("upr_query_match_14m")
    kwargs.update(
        hidden=16,
        static_width=4,
        n_blocks=1,
        global_tokens=4,
        global_token_dim=8,
    )
    model = WeatherBridgeUPRLiteModel(**kwargs).eval()
    x0 = torch.randn(3, 24, 24, 48)
    xT = torch.randn(3, 24, 24, 48)
    tau = torch.tensor([0.0, 0.4, 1.0])

    with torch.no_grad():
        prediction, _ = model(x0, xT, tau)

    expected = (1.0 - tau[:, None, None, None]) * x0
    expected = expected + tau[:, None, None, None] * xT
    torch.testing.assert_close(prediction, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(prediction[0], x0[0], rtol=0, atol=2e-6)
    torch.testing.assert_close(prediction[2], xT[2], rtol=0, atol=2e-6)


def test_query_match_is_aligned_longitude_equivariant():
    torch.manual_seed(39)
    kwargs = upr_scaled_variant_kwargs("upr_query_match_14m")
    kwargs.update(
        hidden=16,
        static_width=4,
        n_blocks=1,
        global_tokens=4,
        global_token_dim=8,
    )
    model = WeatherBridgeUPRLiteModel(**kwargs).eval()
    with torch.no_grad():
        model.refiner.flow_head.weight.normal_(std=0.005)
        model.blend_head.weight.normal_(std=0.005)
        model.residual_head.weight.normal_(std=0.005)
        assert model.detail_head is not None
        model.detail_head.weight.normal_(std=0.005)
        assert model.global_mixer is not None
        model.global_mixer.project.weight.normal_(std=0.005)
        assert model.query_state_modulator is not None
        model.query_state_modulator.scale_coefficients.normal_(std=0.05)
        model.query_state_modulator.shift_coefficients.normal_(std=0.05)
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
        rtol=2e-4,
        atol=2e-4,
    )


def test_query_match_branches_receive_gradient_after_head_activation():
    torch.manual_seed(41)
    model = WeatherBridgeUPRLiteModel(
        hidden=16,
        static_width=4,
        n_blocks=1,
        n_flow_modes=3,
        laplacian_detail=True,
        query_conditioned=False,
        cubic_trajectory=True,
        global_tokens=4,
        global_token_dim=8,
        local_matching=True,
        query_state_modulation=True,
    )
    with torch.no_grad():
        model.residual_head.weight.normal_(std=0.01)
    x0 = torch.randn(2, 24, 24, 48)
    xT = torch.randn_like(x0)
    target = torch.randn_like(x0)

    prediction, _ = model(x0, xT, torch.tensor([0.3, 0.7]))
    (prediction - target).square().mean().backward()

    modulator = model.query_state_modulator
    assert modulator is not None
    assert modulator.scale_coefficients.grad is not None
    assert modulator.scale_coefficients.grad.abs().sum().item() > 0.0
    matching_gradient = model.refiner.input_proj.weight.grad[
        :,
        3 * 16 : 4 * 16,
    ]
    assert matching_gradient.abs().sum().item() > 0.0


def test_endpoint_scaled_variant_is_state_compatible():
    base = WeatherBridgeUPRLiteModel(
        **upr_scaled_variant_kwargs("upr_implicit_global_14m")
    )
    endpoint = WeatherBridgeUPRLiteModel(
        **upr_scaled_variant_kwargs("upr_endpoint_implicit_global_14m")
    )

    assert not base.smooth_endpoint_blend
    assert endpoint.smooth_endpoint_blend
    assert tuple(base.state_dict()) == tuple(endpoint.state_dict())
    endpoint.load_state_dict(base.state_dict(), strict=True)
    assert sum(p.numel() for p in endpoint.parameters()) == 14_261_409


def test_implicit_global_motion_is_query_independent():
    torch.manual_seed(29)
    model = WeatherBridgeUPRLiteModel(
        hidden=16,
        static_width=4,
        n_blocks=1,
        n_flow_modes=3,
        laplacian_detail=True,
        query_conditioned=False,
        cubic_trajectory=True,
        global_tokens=4,
        hydrostatic_coupling=True,
    ).eval()
    x0 = torch.randn(1, 24, 24, 48).expand(2, -1, -1, -1)
    xT = torch.randn(1, 24, 24, 48).expand(2, -1, -1, -1)
    static = torch.randn(1, 3, 24, 48).expand(2, -1, -1, -1)

    with torch.no_grad():
        _, aux = model(
            x0,
            xT,
            torch.tensor([0.2, 0.8]),
            static=static,
        )

    torch.testing.assert_close(
        aux["flow_modes"][0],
        aux["flow_modes"][1],
        rtol=0,
        atol=1e-6,
    )


def test_hydrostatic_coupling_only_updates_mass_fields_and_vanishes_at_anchors():
    torch.manual_seed(31)
    model = WeatherBridgeUPRLiteModel(
        hidden=16,
        static_width=4,
        n_blocks=1,
        n_flow_modes=3,
        laplacian_detail=True,
        query_conditioned=False,
        cubic_trajectory=True,
        global_tokens=4,
        hydrostatic_coupling=True,
    ).eval()
    assert model.hydrostatic_head is not None
    with torch.no_grad():
        model.hydrostatic_head.bias.fill_(1.0)

    x0 = torch.randn(3, 24, 24, 48)
    xT = torch.randn(3, 24, 24, 48)
    tau = torch.tensor([0.0, 0.5, 1.0])
    with torch.no_grad():
        prediction, aux = model(x0, xT, tau)

    linear = (
        (1.0 - tau[:, None, None, None]) * x0
        + tau[:, None, None, None] * xT
    )
    expected_delta = torch.zeros_like(prediction)
    expected_delta[1, (16, 17, 18, 19, 23)] = 0.25
    torch.testing.assert_close(
        prediction,
        linear + expected_delta,
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        aux["hydrostatic_delta"][0],
        torch.zeros_like(aux["hydrostatic_delta"][0]),
    )
    torch.testing.assert_close(
        aux["hydrostatic_delta"][2],
        torch.zeros_like(aux["hydrostatic_delta"][2]),
    )
