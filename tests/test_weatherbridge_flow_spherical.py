import torch

from weather_time_interp.model.weatherbridge_flow_model import (
    SphereConv2d,
    SphereConvTranspose2d,
    WeatherBridgeModel,
    _endpoint_blend,
    _evaluate_base_knot_correction,
    _evaluate_endpoint_tangent_correction,
    _spherical_resize,
    intrinsic_spherical_warp,
    spherical_multiband_components,
    warp,
)
from weather_time_interp.metrics.physical_consistency import (
    CANONICAL_24_CHANNELS,
)


def _small_model(**overrides) -> WeatherBridgeModel:
    kwargs = {
        "in_channels": 24,
        "out_channels": 24,
        "n_static_features": 3,
        "hidden": 8,
        "n_levels": 1,
        "time_emb_dim": 8,
        "use_accel": True,
        "spectral_branch": False,
        "hydro_couple": False,
    }
    kwargs.update(overrides)
    return WeatherBridgeModel(**kwargs)


def test_no_difference_ablation_removes_difference_stem_channels() -> None:
    model = _small_model(use_frame_difference=False).eval()
    x0 = torch.randn(1, 24, 8, 16)
    xT = torch.randn_like(x0)
    static = torch.randn(1, 3, 8, 16)
    captured = {}

    def capture_input(_module, args):
        captured["encoder_input"] = args[0].detach().clone()

    handle = model.encoder[0].register_forward_pre_hook(capture_input)
    with torch.no_grad():
        model(x0, xT, torch.tensor([0.5]), static=static)
    handle.remove()

    encoder_input = captured["encoder_input"]
    assert encoder_input.shape[1] == 51
    torch.testing.assert_close(encoder_input[:, :24], x0)
    torch.testing.assert_close(encoder_input[:, 24:48], xT)
    torch.testing.assert_close(encoder_input[:, 48:], static)


@torch.no_grad()
def test_transport_lesion_uses_exact_linear_scaffold() -> None:
    model = _small_model().eval()
    model.ablate_transport = True
    x0 = torch.randn(1, 24, 8, 16)
    xT = torch.randn_like(x0)
    static = torch.randn(1, 3, 8, 16)
    tau = torch.tensor([0.4])

    _, auxiliary = model(x0, xT, tau, static=static)

    expected = 0.6 * x0 + 0.4 * xT
    torch.testing.assert_close(auxiliary["warped"], expected)


@torch.no_grad()
def test_acceleration_lesion_changes_only_quadratic_transport() -> None:
    torch.manual_seed(17)
    model = _small_model().eval()
    with torch.no_grad():
        model.flow_head.bias[4:8].fill_(0.5)
    x0 = torch.randn(1, 24, 8, 16)
    xT = torch.randn_like(x0)
    static = torch.randn(1, 3, 8, 16)
    tau = torch.tensor([0.4])

    baseline, baseline_aux = model(x0, xT, tau, static=static)
    model.ablate_acceleration = True
    lesioned, lesioned_aux = model(x0, xT, tau, static=static)

    torch.testing.assert_close(baseline_aux["flow"], lesioned_aux["flow"])
    assert not torch.allclose(baseline_aux["warped"], lesioned_aux["warped"])
    assert not torch.allclose(baseline, lesioned)


@torch.no_grad()
def test_hydrostatic_lesion_changes_only_coupled_mass_channels() -> None:
    torch.manual_seed(23)
    model = _small_model(hydro_couple=True).eval()
    model.hydro.weight.zero_()
    model.hydro.bias.fill_(0.25)
    x0 = torch.randn(1, 24, 8, 16)
    xT = torch.randn_like(x0)
    static = torch.randn(1, 3, 8, 16)
    tau = torch.tensor([0.4])

    baseline, _ = model(x0, xT, tau, static=static)
    model.ablate_hydrostatic = True
    lesioned, _ = model(x0, xT, tau, static=static)

    coupled = [16, 17, 18, 19, 23]
    uncoupled = [index for index in range(24) if index not in coupled]
    assert not torch.allclose(baseline[:, coupled], lesioned[:, coupled])
    torch.testing.assert_close(baseline[:, uncoupled], lesioned[:, uncoupled])


def test_multiband_components_preserve_shape_and_phase() -> None:
    impulse = torch.zeros(1, 2, 8, 16)
    impulse[:, :, 3, 5] = 1.0

    bands = spherical_multiband_components(
        impulse,
        torch.tensor([1.0, -1.0]),
    )

    assert len(bands) == 3
    assert all(band.shape == impulse.shape for band in bands)
    assert all(torch.isfinite(band).all() for band in bands)
    assert all(band[..., 3, 5].min().item() > 0 for band in bands)


@torch.no_grad()
def test_multiband_calibrator_is_identity_initialized() -> None:
    torch.manual_seed(5)
    model = _small_model(multiband_calibration=True).eval()
    x0 = torch.randn(1, 24, 8, 16)
    xT = torch.randn_like(x0)
    static = torch.randn(1, 3, 8, 16)

    calibrated, aux = model(
        x0,
        xT,
        torch.tensor([0.5]),
        static=static,
    )
    calibrator = model.multiband_calibrator
    model.multiband_calibrator = None
    baseline, _ = model(x0, xT, torch.tensor([0.5]), static=static)

    torch.testing.assert_close(calibrated, baseline)
    torch.testing.assert_close(
        aux["multiband_correction"],
        torch.zeros_like(aux["multiband_correction"]),
    )
    model.multiband_calibrator = calibrator


@torch.no_grad()
def test_spectral_detail_arms_do_not_change_base_at_endpoints() -> None:
    torch.manual_seed(7)
    x0 = torch.randn(1, 24, 8, 16)
    xT = torch.randn_like(x0)
    static = torch.randn(1, 3, 8, 16)
    for option, module_name in (
        ("multiband_calibration", "multiband_calibrator"),
        ("anchor_detail_bypass", "detail_gain_head"),
    ):
        model = _small_model(**{option: True}).eval()
        head = getattr(model, module_name)
        linear = (
            head.gain_head[-1]
            if module_name == "multiband_calibrator"
            else head[-1]
        )
        torch.nn.init.constant_(linear.weight, 0.1)
        torch.nn.init.constant_(linear.bias, 0.1)
        for tau in (0.0, 1.0):
            corrected, _ = model(
                x0,
                xT,
                torch.tensor([tau]),
                static=static,
            )
            setattr(model, module_name, None)
            baseline, _ = model(
                x0,
                xT,
                torch.tensor([tau]),
                static=static,
            )
            setattr(model, module_name, head)
            torch.testing.assert_close(corrected, baseline)


def test_detail_heads_receive_finite_gradients() -> None:
    torch.manual_seed(11)
    x0 = torch.randn(1, 24, 8, 16)
    xT = torch.randn_like(x0)
    static = torch.randn(1, 3, 8, 16)
    for option, module_name in (
        ("multiband_calibration", "multiband_calibrator"),
        ("anchor_detail_bypass", "detail_gain_head"),
    ):
        model = _small_model(**{option: True})
        prediction, _ = model(
            x0,
            xT,
            torch.tensor([0.5]),
            static=static,
        )
        prediction.square().mean().backward()
        head = getattr(model, module_name)
        linear = (
            head.gain_head[-1]
            if module_name == "multiband_calibrator"
            else head[-1]
        )
        gradient = linear.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0


def test_periodic_warp_crosses_longitude_seam() -> None:
    x = torch.arange(4, dtype=torch.float32).view(1, 1, 1, 4)
    flow = torch.zeros(1, 2, 1, 4)
    flow[:, 0] = 1.0

    periodic = warp(x, flow, periodic_longitude=True)
    legacy = warp(x, flow, periodic_longitude=False)

    torch.testing.assert_close(
        periodic,
        torch.tensor([[[[1.0, 2.0, 3.0, 0.0]]]]),
    )
    torch.testing.assert_close(
        legacy,
        torch.tensor([[[[1.0, 2.0, 3.0, 3.0]]]]),
    )


def test_spherical_warp_crosses_pole_at_antipodal_longitude() -> None:
    x = torch.tensor(
        [[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]]
    )
    flow = torch.zeros(1, 2, 2, 4)
    flow[:, 1, 0] = -1.0

    spherical = warp(x, flow, periodic_longitude=True)
    legacy = warp(x, flow, periodic_longitude=False)

    torch.testing.assert_close(
        spherical[:, :, 0],
        torch.tensor([[[2.0, 3.0, 0.0, 1.0]]]),
    )
    torch.testing.assert_close(
        legacy[:, :, 0],
        torch.tensor([[[0.0, 1.0, 2.0, 3.0]]]),
    )


def test_spherical_warp_reverses_vector_components_at_pole() -> None:
    scalar = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]
    )
    x = torch.stack((scalar, scalar), dim=1)
    flow = torch.zeros(1, 2, 2, 4)
    flow[:, 1, 0] = -1.0

    spherical = warp(
        x,
        flow,
        periodic_longitude=True,
        pole_parity=torch.tensor([1.0, -1.0]),
    )

    torch.testing.assert_close(
        spherical[:, 0, 0],
        torch.tensor([[2.0, 3.0, 0.0, 1.0]]),
    )
    torch.testing.assert_close(
        spherical[:, 1, 0],
        torch.tensor([[-2.0, -3.0, 0.0, -1.0]]),
    )


def test_spherical_warp_is_continuous_at_fractional_pole_crossing() -> None:
    x = torch.tensor(
        [[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]]
    )
    flow = torch.zeros(1, 2, 2, 4)
    flow[:, 1, 0] = -0.5

    spherical = warp(x, flow, periodic_longitude=True)

    torch.testing.assert_close(
        spherical[:, :, 0],
        torch.tensor([[[1.0, 2.0, 1.0, 2.0]]]),
    )


def test_intrinsic_spherical_warp_is_identity_at_zero_flow() -> None:
    torch.manual_seed(37)
    fields = torch.randn(2, 24, 8, 16)
    flow = torch.zeros(2, 2, 8, 16)

    warped = intrinsic_spherical_warp(fields, flow)

    torch.testing.assert_close(warped, fields, rtol=1.0e-4, atol=5.0e-6)


def test_intrinsic_warp_applies_vector_rotation_in_physical_units() -> None:
    torch.manual_seed(39)
    fields = torch.randn(1, 24, 8, 16)
    flow = torch.zeros(1, 2, 8, 16)
    mean = torch.linspace(-3.0, 2.0, 24)
    std = torch.linspace(0.5, 4.0, 24)

    warped = intrinsic_spherical_warp(
        fields,
        flow,
        channel_mean=mean,
        channel_std=std,
    )

    torch.testing.assert_close(warped, fields, rtol=1.0e-4, atol=1.0e-5)


def test_intrinsic_spherical_warp_parallel_transports_wind_norm() -> None:
    fields = torch.zeros(1, 24, 12, 24)
    fields[:, 4:8] = 1.0
    fields[:, 21] = 1.0
    flow = torch.zeros(1, 2, 12, 24)
    flow[:, 0] = 2.0

    warped = intrinsic_spherical_warp(fields, flow)

    for east_index, north_index in (
        (4, 8),
        (5, 9),
        (6, 10),
        (7, 11),
        (21, 22),
    ):
        speed = torch.sqrt(
            warped[:, east_index].square()
            + warped[:, north_index].square()
        )
        torch.testing.assert_close(
            speed,
            torch.ones_like(speed),
            rtol=1.0e-5,
            atol=1.0e-5,
        )
    assert warped[:, 8, 0].abs().max().item() > 0.01


def test_intrinsic_spherical_warp_has_finite_zero_flow_gradient() -> None:
    torch.manual_seed(41)
    fields = torch.randn(1, 24, 8, 16)
    flow = torch.zeros(1, 2, 8, 16, requires_grad=True)

    intrinsic_spherical_warp(fields, flow).square().mean().backward()

    assert flow.grad is not None
    assert torch.isfinite(flow.grad).all()
    assert torch.count_nonzero(flow.grad).item() > 0


def test_intrinsic_spherical_warp_has_finite_polar_stress_gradient() -> None:
    torch.manual_seed(47)
    fields = torch.randn(1, 24, 12, 24)
    flow = (8.0 * torch.randn(1, 2, 12, 24)).requires_grad_()

    warped = intrinsic_spherical_warp(fields, flow)
    polar_loss = warped[..., (0, 1, -2, -1), :].square().mean()
    polar_loss.backward()

    assert torch.isfinite(warped).all()
    assert flow.grad is not None
    assert torch.isfinite(flow.grad).all()


def test_spherical_convolution_uses_antipodal_pole_padding() -> None:
    conv = SphereConv2d(1, 1, kernel_size=3, stride=1)
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        conv.weight[0, 0, 0, 1] = 1.0
    x = torch.tensor(
        [[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]]
    )

    output = conv(x)

    torch.testing.assert_close(
        output[:, :, 0],
        torch.tensor([[[2.0, 3.0, 0.0, 1.0]]]),
    )


def test_spherical_convolution_applies_vector_pole_parity() -> None:
    conv = SphereConv2d(
        2,
        1,
        kernel_size=3,
        stride=1,
        pole_parity=torch.tensor([1.0, -1.0]),
    )
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        conv.weight[0, 1, 0, 1] = 1.0
    scalar = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]
    )
    x = torch.stack((scalar, scalar), dim=1)

    output = conv(x)

    torch.testing.assert_close(
        output[:, :, 0],
        torch.tensor([[[-2.0, -3.0, 0.0, -1.0]]]),
    )


def test_model_pole_parity_matches_canonical_vector_channels() -> None:
    model = _small_model(spherical_ops=True)
    vector_indices = {
        index
        for index, name in enumerate(CANONICAL_24_CHANNELS)
        if name.startswith(("U", "V")) or name in {"u10", "v10"}
    }
    negative_indices = {
        index
        for index, parity in enumerate(model.field_pole_parity.tolist())
        if parity < 0
    }

    assert negative_indices == vector_indices


def test_spherical_transpose_convolution_is_longitude_equivariant() -> None:
    torch.manual_seed(5)
    layer = SphereConvTranspose2d(2, 3).eval()
    x = torch.randn(1, 2, 6, 12)

    with torch.no_grad():
        output = layer(x)
        shifted = layer(torch.roll(x, shifts=1, dims=-1))

    assert output.shape == (1, 3, 12, 24)
    torch.testing.assert_close(
        shifted,
        torch.roll(output, shifts=2, dims=-1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_spherical_transpose_convolution_uses_antipodal_ghost() -> None:
    layer = SphereConvTranspose2d(1, 1)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.zero_()
    x = torch.zeros(1, 1, 2, 4)
    x[:, :, 0, 0] = 1.0

    output = layer(x)

    assert output.shape == (1, 1, 4, 8)
    assert torch.count_nonzero(output[:, :, 0, 3:6]).item() > 0


def test_spherical_endpoint_model_reproduces_both_anchors() -> None:
    torch.manual_seed(7)
    model = _small_model(
        spherical_ops=True,
        endpoint_preserving=True,
        query_independent_trajectory=True,
    ).eval()
    x0 = torch.randn(2, 24, 8, 16)
    x1 = torch.randn(2, 24, 8, 16)
    static = torch.randn(2, 3, 8, 16)

    with torch.no_grad():
        at_zero, _ = model(x0, x1, torch.zeros(2), static=static)
        at_one, _ = model(x0, x1, torch.ones(2), static=static)

    torch.testing.assert_close(at_zero, x0, rtol=0, atol=0)
    torch.testing.assert_close(at_one, x1, rtol=0, atol=0)


def test_query_independent_trajectory_predicts_one_motion_field() -> None:
    torch.manual_seed(11)
    model = _small_model(
        spherical_ops=True,
        endpoint_preserving=True,
        query_independent_trajectory=True,
    ).eval()
    x0 = torch.randn(1, 24, 8, 16)
    x1 = torch.randn(1, 24, 8, 16)
    static = torch.randn(1, 3, 8, 16)

    with torch.no_grad():
        _, early = model(x0, x1, torch.tensor([0.25]), static=static)
        _, late = model(x0, x1, torch.tensor([0.75]), static=static)

    torch.testing.assert_close(early["flow"], late["flow"], rtol=0, atol=0)


def test_compact_vp3_is_query_independent_and_endpoint_exact() -> None:
    torch.manual_seed(19)
    model = _small_model(
        use_accel=False,
        spherical_ops=True,
        endpoint_preserving=True,
        query_independent_trajectory=True,
        n_flow_modes=3,
        cubic_trajectory=True,
        global_tokens=2,
        global_token_dim=8,
    ).eval()
    x0 = torch.randn(1, 24, 8, 16)
    x1 = torch.randn_like(x0)
    static = torch.randn(1, 3, 8, 16)

    with torch.no_grad():
        at_zero, _ = model(x0, x1, torch.zeros(1), static=static)
        at_one, _ = model(x0, x1, torch.ones(1), static=static)
        _, early = model(x0, x1, torch.tensor([0.25]), static=static)
        _, late = model(x0, x1, torch.tensor([0.75]), static=static)

    torch.testing.assert_close(at_zero, x0, rtol=0, atol=0)
    torch.testing.assert_close(at_one, x1, rtol=0, atol=0)
    assert early["flow"].shape == (1, 36, 8, 16)
    torch.testing.assert_close(early["flow"], late["flow"], rtol=0, atol=0)


def test_compact_vp3_group_warp_can_move_pressure_levels_separately() -> None:
    model = _small_model(
        use_accel=False,
        spherical_ops=True,
        n_flow_modes=3,
        cubic_trajectory=True,
    ).eval()
    longitude = torch.arange(4, dtype=torch.float32).view(1, 1, 1, 4)
    fields = longitude.expand(1, 24, 2, 4).clone()
    flow = torch.zeros(1, 5, 2, 2, 4)
    flow[:, 0, 0] = 1.0

    warped = model._warp_grouped_fields(fields, flow)

    torch.testing.assert_close(
        warped[:, 0],
        torch.tensor([[[1.0, 2.0, 3.0, 0.0]]]).expand(1, 2, 4),
    )
    torch.testing.assert_close(warped[:, 1], fields[:, 1])


def test_endpoint_tangent_correction_is_asymmetric_and_endpoint_exact() -> None:
    tangents = torch.tensor([[[[2.0]], [[4.0]], [[6.0]], [[8.0]]]])

    at_zero = _evaluate_endpoint_tangent_correction(
        tangents,
        torch.tensor([0.0]),
    )
    at_one = _evaluate_endpoint_tangent_correction(
        tangents,
        torch.tensor([1.0]),
    )
    at_quarter = _evaluate_endpoint_tangent_correction(
        tangents,
        torch.tensor([0.25]),
    )
    expected = 0.25 * 0.75 * (
        0.75 * tangents[:, :2] - 0.25 * tangents[:, 2:]
    )

    torch.testing.assert_close(at_zero, torch.zeros_like(at_zero))
    torch.testing.assert_close(at_one, torch.zeros_like(at_one))
    torch.testing.assert_close(at_quarter, expected)


def test_base_knot_correction_interpolates_seen_hours_and_anchors() -> None:
    knots = torch.tensor(
        [[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]]]
    )

    at_zero = _evaluate_base_knot_correction(knots, torch.tensor([0.0]))
    at_one = _evaluate_base_knot_correction(knots, torch.tensor([1.0]))
    at_h1 = _evaluate_base_knot_correction(knots, torch.tensor([1.0 / 6.0]))
    at_h3 = _evaluate_base_knot_correction(knots, torch.tensor([0.5]))
    at_h5 = _evaluate_base_knot_correction(knots, torch.tensor([5.0 / 6.0]))

    torch.testing.assert_close(at_zero, torch.zeros_like(at_zero))
    torch.testing.assert_close(at_one, torch.zeros_like(at_one))
    torch.testing.assert_close(at_h1, knots[:, :2])
    torch.testing.assert_close(at_h3, knots[:, 2:4])
    torch.testing.assert_close(at_h5, knots[:, 4:])


def test_base_knot_head_corrects_only_non_moisture_fields() -> None:
    torch.manual_seed(29)
    model = _small_model(
        use_accel=False,
        spherical_ops=True,
        endpoint_preserving=True,
        query_independent_trajectory=True,
        n_flow_modes=3,
        cubic_trajectory=True,
        base_field_knots=True,
    ).eval()
    assert model.base_knot_head is not None
    x0 = torch.randn(1, 24, 8, 16)
    x1 = torch.randn_like(x0)
    static = torch.randn(1, 3, 8, 16)
    tau = torch.tensor([1.0 / 6.0])

    with torch.no_grad():
        model.base_knot_head.bias.zero_()
        baseline, _ = model(x0, x1, tau, static=static)
        model.base_knot_head.bias.fill_(1.0)
        corrected, auxiliary = model(x0, x1, tau, static=static)
        at_zero, _ = model(x0, x1, torch.zeros(1), static=static)
        at_one, _ = model(x0, x1, torch.ones(1), static=static)

    difference = corrected - baseline
    torch.testing.assert_close(difference[:, :12], torch.ones_like(difference[:, :12]))
    torch.testing.assert_close(difference[:, 12:16], torch.zeros_like(difference[:, 12:16]))
    torch.testing.assert_close(difference[:, 16:], torch.ones_like(difference[:, 16:]))
    assert auxiliary["base_knot_correction"].shape == (1, 20, 8, 16)
    torch.testing.assert_close(at_zero, x0, rtol=0, atol=0)
    torch.testing.assert_close(at_one, x1, rtol=0, atol=0)


def test_multiscale_experts_keep_physical_field_groups_separate() -> None:
    model = _small_model(
        n_levels=3,
        use_accel=False,
        endpoint_preserving=True,
        query_independent_trajectory=True,
        multiscale_field_experts=True,
    ).eval()
    assert model.multiscale_expert_heads is not None
    features = [
        torch.zeros(1, head[0].in_channels, height, 2 * height)
        for head, height in zip(
            model.multiscale_expert_heads,
            (2, 4, 8),
        )
    ]
    with torch.no_grad():
        for head in model.multiscale_expert_heads:
            head[-1].bias.zero_()
        bias = model.multiscale_expert_heads[0][-1].bias.reshape(6, 3, 4)
        for group in range(6):
            bias[group, 0].fill_(float(group + 1))
        correction = model._multiscale_field_correction(
            features,
            torch.tensor([1.0 / 6.0]),
            (8, 16),
        )

    for group in range(6):
        expected = torch.full_like(
            correction[:, group * 4:(group + 1) * 4],
            float(group + 1),
        )
        torch.testing.assert_close(
            correction[:, group * 4:(group + 1) * 4],
            expected,
        )


def test_multiscale_experts_are_endpoint_exact_with_nonzero_heads() -> None:
    torch.manual_seed(31)
    model = _small_model(
        n_levels=3,
        use_accel=False,
        spherical_ops=True,
        endpoint_preserving=True,
        query_independent_trajectory=True,
        endpoint_tangents=True,
        multiscale_field_experts=True,
    ).eval()
    assert model.multiscale_expert_heads is not None
    x0 = torch.randn(1, 24, 8, 16)
    x1 = torch.randn_like(x0)
    static = torch.randn(1, 3, 8, 16)

    with torch.no_grad():
        for head in model.multiscale_expert_heads:
            head[-1].weight.normal_()
            head[-1].bias.normal_()
        at_zero, aux_zero = model(
            x0,
            x1,
            torch.zeros(1),
            static=static,
        )
        at_one, aux_one = model(
            x0,
            x1,
            torch.ones(1),
            static=static,
        )

    torch.testing.assert_close(at_zero, x0, rtol=0, atol=0)
    torch.testing.assert_close(at_one, x1, rtol=0, atol=0)
    torch.testing.assert_close(
        aux_zero["multiscale_field_correction"],
        torch.zeros_like(aux_zero["multiscale_field_correction"]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        aux_one["multiscale_field_correction"],
        torch.zeros_like(aux_one["multiscale_field_correction"]),
        rtol=0,
        atol=0,
    )


def test_geo_msf_model_is_endpoint_exact() -> None:
    torch.manual_seed(43)
    model = _small_model(
        n_levels=3,
        use_accel=True,
        spherical_ops=True,
        endpoint_preserving=True,
        query_independent_trajectory=True,
        endpoint_tangents=True,
        multiscale_field_experts=True,
        intrinsic_spherical_transport=True,
    ).eval()
    x0 = torch.randn(1, 24, 8, 16)
    x1 = torch.randn_like(x0)
    static = torch.randn(1, 3, 8, 16)

    with torch.no_grad():
        at_zero, _ = model(x0, x1, torch.zeros(1), static=static)
        at_one, _ = model(x0, x1, torch.ones(1), static=static)

    torch.testing.assert_close(at_zero, x0, rtol=0, atol=0)
    torch.testing.assert_close(at_one, x1, rtol=0, atol=0)


def test_endpoint_tangent_head_is_query_independent() -> None:
    torch.manual_seed(23)
    model = _small_model(
        use_accel=False,
        spherical_ops=True,
        endpoint_preserving=True,
        query_independent_trajectory=True,
        n_flow_modes=3,
        cubic_trajectory=True,
        global_tokens=2,
        global_token_dim=8,
        endpoint_tangents=True,
    ).eval()
    x0 = torch.randn(1, 24, 8, 16)
    x1 = torch.randn_like(x0)
    static = torch.randn(1, 3, 8, 16)

    with torch.no_grad():
        at_zero, _ = model(x0, x1, torch.zeros(1), static=static)
        at_one, _ = model(x0, x1, torch.ones(1), static=static)
        _, early = model(x0, x1, torch.tensor([0.25]), static=static)
        _, late = model(x0, x1, torch.tensor([0.75]), static=static)

    torch.testing.assert_close(at_zero, x0, rtol=0, atol=0)
    torch.testing.assert_close(at_one, x1, rtol=0, atol=0)
    assert early["endpoint_tangent"].shape == (1, 48, 8, 16)
    torch.testing.assert_close(
        early["endpoint_tangent"],
        late["endpoint_tangent"],
        rtol=0,
        atol=0,
    )


def test_spherical_variant_preserves_parameter_contract() -> None:
    legacy = _small_model()
    spherical = _small_model(
        spherical_ops=True,
        endpoint_preserving=True,
        query_independent_trajectory=True,
    )
    legacy_state = legacy.state_dict()
    spherical_state = spherical.state_dict()

    assert legacy_state.keys() == spherical_state.keys()
    assert {
        key: tuple(value.shape)
        for key, value in legacy_state.items()
    } == {
        key: tuple(value.shape)
        for key, value in spherical_state.items()
    }


def test_spherical_resize_is_longitude_equivariant() -> None:
    torch.manual_seed(13)
    fields = torch.randn(1, 3, 5, 8)

    output = _spherical_resize(fields, (10, 16))
    shifted = _spherical_resize(
        torch.roll(fields, 2, dims=-1),
        (10, 16),
    )

    torch.testing.assert_close(
        shifted,
        torch.roll(output, 4, dims=-1),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_spherical_resize_applies_vector_pole_parity() -> None:
    scalar = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]
    )
    fields = torch.stack((scalar, scalar), dim=1)

    resized = _spherical_resize(
        fields,
        (4, 4),
        pole_parity=torch.tensor([1.0, -1.0]),
    )

    torch.testing.assert_close(
        resized[:, 0, 0],
        torch.tensor([[0.5, 1.5, 1.5, 2.5]]),
    )
    torch.testing.assert_close(
        resized[:, 1, 0],
        torch.tensor([[-0.5, 0.0, 1.5, 2.0]]),
    )


def test_endpoint_blend_is_bounded_without_dead_clamp_gradient() -> None:
    logits = torch.tensor([[[[4.0]]]], requires_grad=True)
    tau = torch.tensor([[[[0.25]]]])

    alpha = _endpoint_blend(logits, tau)
    alpha.sum().backward()

    assert 0.0 < alpha.item() < 1.0
    assert logits.grad is not None
    assert logits.grad.item() > 0.0
    torch.testing.assert_close(
        _endpoint_blend(torch.zeros_like(logits), tau),
        1.0 - tau,
    )
    torch.testing.assert_close(
        _endpoint_blend(logits.detach(), torch.zeros_like(tau)),
        torch.ones_like(tau),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _endpoint_blend(logits.detach(), torch.ones_like(tau)),
        torch.zeros_like(tau),
        rtol=0,
        atol=0,
    )


def test_spherical_endpoint_model_is_aligned_longitude_equivariant() -> None:
    torch.manual_seed(17)
    model = _small_model(
        n_levels=2,
        spherical_ops=True,
        endpoint_preserving=True,
        query_independent_trajectory=True,
    ).eval()
    x0 = torch.randn(1, 24, 16, 32)
    x1 = torch.randn_like(x0)
    static = torch.randn(1, 3, 16, 32)

    with torch.no_grad():
        prediction, _ = model(
            x0,
            x1,
            torch.tensor([0.4]),
            static=static,
        )
        shifted, _ = model(
            torch.roll(x0, 4, dims=-1),
            torch.roll(x1, 4, dims=-1),
            torch.tensor([0.4]),
            static=torch.roll(static, 4, dims=-1),
        )

    torch.testing.assert_close(
        shifted,
        torch.roll(prediction, 4, dims=-1),
        rtol=2.0e-4,
        atol=2.0e-4,
    )
