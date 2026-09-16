from __future__ import annotations

from argparse import Namespace
import json

import pytest
import torch

from tools.train.training_protocol import (
    checkpoint_resume_lineage,
    memmap_dataset_provenance,
    validate_training_protocol_args,
)
from tools.train.train_capacity_matched_6h import (
    DISTILLATION_HIGH_GATES,
    PAPER_CHANNEL_NAMES,
    apply_trainable_scope,
    build_net,
    canonical_arch_name,
    compose_multiteacher_target,
    distillation_channel_mask,
    distillation_schedule_scale,
    exchange_anchors,
    highpass_component,
    interval_corrected_distillation_blend,
    latent_block_weights,
    load_direct_distillation_route,
    lowpass_component,
    loss_profile_channel_weights,
    loss_profile_tau_weights,
    pareto_minimax_latitude_l1,
    relative_group_pareto_l1,
    normalized_latent_distillation_loss,
    resolve_arch,
    resolve_train_batch_schedule,
    routed_teacher_target,
    spectral_loss_config,
    teacher_advantage_mask,
    trainable_scope_weight_decay,
    weather_field_pole_parity,
    weighted_latitude_l1,
)


def test_anchor_exchange_preserves_target_time_identity() -> None:
    x0 = torch.tensor([[[[0.0]]], [[[10.0]]]])
    x1 = torch.tensor([[[[6.0]]], [[[16.0]]]])
    tau = torch.tensor([1.0 / 6.0, 5.0 / 6.0])
    tau_hour = torch.tensor([1, 5])

    swapped = exchange_anchors(
        x0,
        x1,
        tau,
        tau_hour,
        torch.tensor([True, False]),
        6.0,
    )

    torch.testing.assert_close(swapped[0][0], x1[0])
    torch.testing.assert_close(swapped[1][0], x0[0])
    torch.testing.assert_close(swapped[2], torch.tensor([5.0 / 6.0, 5.0 / 6.0]))
    torch.testing.assert_close(swapped[3], torch.tensor([5, 5]))


def _args(**overrides) -> Namespace:
    values = {
        "years": [2014, 2015, 2016, 2017, 2018, 2019],
        "val_years": [2020],
        "train_tau_subset": [1, 3, 5],
        "eval_tau": [1, 2, 3, 4, 5],
        "window_hours": 6,
        "samples_per_date_train": 4,
        "samples_per_date_val": 2,
        "bs": 4,
        "val_bs": 2,
        "accumulate": 4,
        "max_epochs": 8,
        "workers": 8,
        "val_workers": 4,
        "gpus": [0],
        "lr": 1e-4,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "train_batches_per_epoch": 0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_accepts_matched_sparse_tau_protocol() -> None:
    validate_training_protocol_args(_args())


def test_weatherbridge_public_alias_selects_headline_architecture() -> None:
    assert resolve_arch("weatherbridge") == "flow_pp3"
    assert canonical_arch_name("weatherbridge") == "WeatherBridge"


def test_hres_augmented_weatherbridge_has_zero_init_lead_conditioner() -> None:
    model, _, _ = build_net("flow_pp3_hres_aug", "")

    assert model.forecast_lead_conditioning is True
    assert model.forecast_lead_mlp is not None
    assert torch.count_nonzero(model.forecast_lead_mlp.net[-1].weight) == 0
    assert torch.count_nonzero(model.forecast_lead_mlp.net[-1].bias) == 0


def test_hres_residual_adapter_is_zero_initialized_and_isolated() -> None:
    model, _, _ = build_net("flow_pp3_hres_residual", "")

    assert model.forecast_lead_conditioning is True
    assert model.hres_residual_head is not None
    assert torch.count_nonzero(model.hres_residual_head.weight) == 0
    assert torch.count_nonzero(model.hres_residual_head.bias) == 0

    apply_trainable_scope(model, "hres_residual_adapter")
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        "hres_residual_head.weight",
        "hres_residual_head.bias",
    }


def test_accepts_12h_protocol_with_held_taus_in_validation() -> None:
    validate_training_protocol_args(
        _args(
            years=[2017, 2018, 2019],
            train_tau_subset=[1, 2, 3, 5, 7, 9, 10, 11],
            eval_tau=list(range(1, 12)),
            window_hours=12,
            samples_per_date_train=2,
            max_epochs=10,
        )
    )


def test_schedule_keeps_complete_6h_accumulation_groups() -> None:
    assert resolve_train_batch_schedule(
        dataset_size=26274,
        batch_size_per_device=4,
        accumulate=4,
        devices=1,
    ) == (6568, 1642, 0)


def test_spherical_highpass_uses_antipodal_vector_parity() -> None:
    field = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)
    parity = torch.tensor([1.0, -1.0])
    channel_parity = parity.view(1, 2, 1, 1)
    top = torch.roll(field[..., :1, :], 2, dims=-1) * channel_parity
    bottom = (
        torch.roll(field[..., -1:, :], 2, dims=-1) * channel_parity
    )
    padded = torch.cat((top, field, bottom), dim=-2)
    padded = torch.nn.functional.pad(
        padded,
        (1, 1, 0, 0),
        mode="circular",
    )
    expected = field - torch.nn.functional.avg_pool2d(
        padded,
        kernel_size=3,
        stride=1,
    )

    assert torch.equal(highpass_component(field, parity), expected)


def test_planar_highpass_preserves_legacy_boundary() -> None:
    field = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)
    padded = torch.nn.functional.pad(
        field,
        (1, 1, 0, 0),
        mode="circular",
    )
    padded = torch.nn.functional.pad(
        padded,
        (0, 0, 1, 1),
        mode="replicate",
    )
    expected = field - torch.nn.functional.avg_pool2d(
        padded,
        kernel_size=3,
        stride=1,
    )

    assert torch.equal(highpass_component(field), expected)


def test_multiteacher_composite_changes_only_advected_detail() -> None:
    low_teacher = torch.arange(
        2 * 24 * 4 * 8,
        dtype=torch.float32,
    ).reshape(2, 24, 4, 8)
    high_teacher = torch.flip(low_teacher, dims=(-1,))
    parity = weather_field_pole_parity()
    mask = distillation_channel_mask("advected")

    composite = compose_multiteacher_target(
        low_teacher,
        high_teacher,
        mask,
        parity,
    )

    smooth = torch.tensor([16, 17, 18, 19, 20, 23])
    advected = mask.bool()
    torch.testing.assert_close(composite[:, smooth], low_teacher[:, smooth])
    torch.testing.assert_close(
        composite[:, advected],
        lowpass_component(low_teacher[:, advected], parity[advected])
        + highpass_component(high_teacher[:, advected], parity[advected]),
    )


def test_distillation_parity_marks_only_vector_fields_odd() -> None:
    parity = weather_field_pole_parity()

    assert torch.equal(
        (parity < 0).nonzero().reshape(-1),
        torch.tensor([4, 5, 6, 7, 8, 9, 10, 11, 21, 22]),
    )


def test_moisture_distillation_mask_selects_only_q_levels() -> None:
    mask = distillation_channel_mask("moisture")

    assert torch.equal(mask.nonzero().reshape(-1), torch.arange(12, 16))


def test_teacher_advantage_mask_keeps_only_teacher_improvements() -> None:
    student = torch.tensor([[[[2.0, 0.5, 1.0]]]])
    teacher = torch.tensor([[[[0.5, 2.0, 1.0]]]])
    truth = torch.zeros_like(student)

    mask = teacher_advantage_mask(student, teacher, truth)

    assert torch.equal(mask, torch.tensor([[[[1.0, 0.0, 0.0]]]]))


def test_teacher_guided_truth_gate_is_a_supported_protocol() -> None:
    assert "teacher_better_truth" in DISTILLATION_HIGH_GATES


def test_direct_distillation_route_and_target_preserve_excluded_fields(
    tmp_path,
) -> None:
    route_path = tmp_path / "route.json"
    route_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_role": "era5_2020_validation_only",
                "selection_year": 2020,
                "tau_hours": [1, 3, 5],
                "channel_names": list(PAPER_CHANNEL_NAMES),
                "teacher_route": [
                    [index == 0 for index in range(24)],
                    [False] * 24,
                    [index == 1 for index in range(24)],
                ],
                "control": {"sha256": "control"},
                "teacher": {"sha256": "teacher"},
            }
        )
    )
    route, provenance = load_direct_distillation_route(route_path)
    selected = route[torch.tensor([1, 5])].view(2, 24, 1, 1)
    truth = torch.zeros(2, 24, 1, 1)
    teacher = torch.ones_like(truth)

    target = routed_teacher_target(truth, teacher, selected, 0.25)

    assert target[0, 0].item() == pytest.approx(0.25)
    assert target[1, 1].item() == pytest.approx(0.25)
    assert torch.count_nonzero(target).item() == 2
    assert provenance["active_routes"] == 2


def test_direct_distillation_interval_preserves_mean_blend() -> None:
    values = [
        interval_corrected_distillation_blend(0.02, 4, step, 0.5)
        for step in range(4)
    ]
    assert values == [0.04, 0.0, 0.0, 0.0]
    assert sum(values) / len(values) == pytest.approx(0.01)
    assert interval_corrected_distillation_blend(0.5, 4, 0, 1.0) == 1.0


def test_cosine_distillation_schedule_reaches_zero_before_final_steps() -> None:
    values = [
        distillation_schedule_scale(
            "cosine_decay",
            step,
            100,
            0.0,
            0.8,
        )
        for step in (0, 39, 79, 99)
    ]

    assert values[0] < 1.0
    assert values[0] > values[1] > values[2]
    assert values[2] == 0.0
    assert values[3] == 0.0
    assert distillation_schedule_scale("constant", 99, 100, 0.0, 0.8) == 1.0


def test_latent_block_weights_decay_away_from_bottleneck() -> None:
    weights = latent_block_weights(0.5)

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["encoder"] > weights["encoder.down_blocks.6"]
    assert (
        weights["encoder.down_blocks.6"]
        > weights["encoder.down_blocks.2"]
    )
    assert weights["encoder.down_blocks.6"] == pytest.approx(
        weights["decoder.up_blocks.2"]
    )


def test_latent_loss_is_zero_for_identical_taps_and_has_gradients() -> None:
    weights = latent_block_weights(0.5)
    student = {
        name: torch.randn(2, 3, 4, 5, requires_grad=True)
        for name in weights
    }
    teacher = {name: value.detach().clone() for name, value in student.items()}

    identical, _ = normalized_latent_distillation_loss(
        student,
        teacher,
        0.5,
    )
    assert identical.item() == pytest.approx(0.0)

    shifted_teacher = {
        name: value + 0.25
        for name, value in teacher.items()
    }
    shifted, _ = normalized_latent_distillation_loss(
        student,
        shifted_teacher,
        0.5,
    )
    shifted.backward()
    assert shifted.item() > 0.0
    assert all(value.grad is not None for value in student.values())


def test_uniform_weighted_l1_preserves_legacy_reduction() -> None:
    prediction = torch.arange(32, dtype=torch.float32).reshape(2, 2, 2, 4)
    target = torch.flip(prediction, dims=(-1,))
    latitude = torch.tensor((0.5, 1.5)).view(1, 1, 2, 1)
    expected = ((prediction - target).abs() * latitude).mean()

    actual = weighted_latitude_l1(
        prediction,
        target,
        latitude,
        torch.ones(2),
        torch.ones(2),
    )

    assert torch.equal(actual, expected)


def test_pareto_minimax_loss_targets_worst_four_fields() -> None:
    prediction = torch.ones(1, 24, 1, 1)
    prediction[:, -4:] = 10.0
    target = torch.zeros_like(prediction)

    actual = pareto_minimax_latitude_l1(
        prediction,
        target,
        torch.ones(1, 1, 1, 1),
        torch.ones(1),
    )

    mean_loss = (20.0 + 4.0 * 10.0) / 24.0
    assert actual.item() == pytest.approx(0.5 * mean_loss + 0.5 * 10.0)


def test_pareto_minimax_loss_preserves_uniform_error_scale() -> None:
    prediction = torch.full((2, 24, 2, 2), 3.0)

    actual = pareto_minimax_latitude_l1(
        prediction,
        torch.zeros_like(prediction),
        torch.ones(1, 1, 2, 1),
        torch.tensor((1.0, 2.0)),
    )

    assert actual.item() == pytest.approx(3.0)


def test_relative_group_pareto_balances_by_train_only_scale() -> None:
    errors = torch.ones(2, 24)
    errors[0, -6:] = 4.0
    scales = torch.ones_like(errors)
    scales[0, -6:] = 2.0

    actual = relative_group_pareto_l1(
        errors,
        scales,
        torch.tensor((1.0, 1.0)),
        cvar_fields=6,
    )

    sample0_macro = (18.0 + 6.0 * 4.0) / 24.0
    sample0_nominal_scale = (18.0 + 6.0 * 2.0) / 24.0
    sample0 = 0.5 * sample0_macro + 0.5 * 2.0 * sample0_nominal_scale
    assert actual.item() == pytest.approx(0.5 * (sample0 + 1.0))


def test_relative_group_pareto_rejects_nonpositive_scales() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        relative_group_pareto_l1(
            torch.ones(1, 24),
            torch.zeros(1, 24),
            torch.ones(1),
            cvar_fields=6,
        )


def test_masked_spectral_loss_excludes_smooth_mass_fields() -> None:
    weight, mask, profile = spectral_loss_config(
        "flow_pp3",
        0.02,
        "advected",
    )

    assert weight == pytest.approx(0.02)
    assert profile == "advected"
    assert mask is not None
    assert torch.equal(mask[[16, 17, 18, 19, 20, 23]], torch.zeros(6))
    assert mask[[0, 4, 8, 12, 21, 22]].min().item() == 1.0


def test_spectral_loss_override_rejects_invalid_weight() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        spectral_loss_config("flow_pp3", float("nan"), "all")


def test_base_loss_profile_prioritizes_non_moisture_fields() -> None:
    reconstruction, highpass = loss_profile_channel_weights(
        "base_balanced"
    )

    assert reconstruction.mean().item() == pytest.approx(1.0)
    assert highpass.mean().item() == pytest.approx(1.0)
    assert reconstruction[12:16].max() < reconstruction[:12].min()
    assert reconstruction[16:20].min() > reconstruction[12:16].max()
    assert reconstruction[20] > reconstruction[12]
    assert highpass[16] < highpass[4]
    assert highpass[23] < highpass[8]


def test_edge_profile_reweights_only_seen_endpoint_hours() -> None:
    tau = torch.tensor((1, 3, 5, 1, 3, 5))

    balanced = loss_profile_tau_weights("base_balanced", tau)
    edge = loss_profile_tau_weights("base_edge_balanced", tau)

    assert torch.equal(balanced, torch.ones(6))
    assert torch.equal(edge, torch.tensor((1.35, 1.0, 1.35, 1.35, 1.0, 1.35)))


def test_base_knot_scope_freezes_every_other_parameter() -> None:
    model, _, _ = build_net("flow_compact_lagrange_l", "")

    apply_trainable_scope(model, "base_knot_head")

    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        "base_knot_head.weight",
        "base_knot_head.bias",
    }


def test_q_output_head_scope_masks_every_non_moisture_row() -> None:
    class OutputOnlyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Conv2d(2, 2, 1)
            self.decoder = torch.nn.Module()
            self.decoder.conv_out = torch.nn.Conv2d(2, 24, 3, padding=1)

    model = OutputOnlyNet()
    apply_trainable_scope(model, "q_output_head")
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {
        "decoder.conv_out.weight",
        "decoder.conv_out.bias",
    }

    model.decoder.conv_out(torch.ones(1, 2, 4, 4)).sum().backward()
    weight_gradient = model.decoder.conv_out.weight.grad
    bias_gradient = model.decoder.conv_out.bias.grad
    assert weight_gradient is not None
    assert bias_gradient is not None
    assert torch.count_nonzero(weight_gradient[:12]) == 0
    assert torch.count_nonzero(weight_gradient[12:16]) > 0
    assert torch.count_nonzero(weight_gradient[16:]) == 0
    assert torch.count_nonzero(bias_gradient[:12]) == 0
    assert torch.count_nonzero(bias_gradient[12:16]) > 0
    assert torch.count_nonzero(bias_gradient[16:]) == 0
    assert trainable_scope_weight_decay("q_output_head") == 0.0


@pytest.mark.parametrize(
    (
        "batch_size",
        "accumulate",
        "expected_microbatches",
        "expected_dropped",
    ),
    (
        (8, 2, 2186, 1),
        (4, 4, 4372, 2),
        (2, 8, 8744, 4),
    ),
)
def test_schedule_preserves_12h_optimizer_steps_across_batch_fallbacks(
    batch_size: int,
    accumulate: int,
    expected_microbatches: int,
    expected_dropped: int,
) -> None:
    assert resolve_train_batch_schedule(
        dataset_size=17496,
        batch_size_per_device=batch_size,
        accumulate=accumulate,
        devices=1,
    ) == (expected_microbatches, 1093, expected_dropped)


@pytest.mark.parametrize(
    ("arch", "expected_parameters"),
    (
        ("dcae_14m", 14_365_049),
        ("atmvfi", 14_675_616),
        ("amt", 14_255_541),
        ("amt_residual", 14_255_565),
        ("upr_implicit_global_14m", 14_261_409),
        ("upr_endpoint_implicit_global_14m", 14_261_409),
        ("upr_local_corr_14m", 14_290_137),
        ("upr_query_match_14m", 14_258_369),
        ("upr_universal_latent_q4_10m", 9_500_460),
        ("upr_spherical_implicit_global_14m", 14_261_409),
        ("flow_pp3", 14_260_565),
        ("flow_pp3_nodiff", 14_245_013),
        ("flow_pp3_spherical", 14_260_565),
        ("flow_pp3_multiband", 14_279_069),
        ("flow_pp3_detail", 14_266_733),
        ("flow_pp3_detail_fm", 14_350_381),
        ("flow_universal_latent", 12_100_232),
        ("flow_universal_latent_refine", 13_651_208),
        ("flow_universal_content_refine", 13_651_386),
        ("flow_universal_pareto_refine", 13_659_644),
        ("flow_universal_fm", 12_182_152),
        ("flow_pp3_compact_l", 8_838_773),
        ("flow_msf_pareto_l", 8_884_205),
        ("flow_geo_msf_l", 8_884_205),
        ("flow_spherical_ep", 14_260_565),
        ("flow_compact_vp3", 3_057_013),
        ("flow_compact_vp3_m", 4_626_341),
        ("flow_compact_vp3_l", 8_752_261),
        ("flow_compact_hermite_l", 8_776_501),
        ("flow_compact_lagrange_l", 8_806_801),
        ("lg_wavelet_10m", 9_685_168),
    ),
)
def test_capacity_matched_architecture_parameter_contract(
    arch: str,
    expected_parameters: int,
) -> None:
    model, _, _ = build_net(arch, "")

    assert sum(parameter.numel() for parameter in model.parameters()) == (
        expected_parameters
    )


def test_paper_detail_model_tapers_only_the_detail_arm() -> None:
    model, _, _ = build_net("flow_pp3_detail", "")

    assert model.hidden == 72
    assert model.n_levels == 3
    assert model.use_frame_difference is True
    assert model.use_accel is True
    assert model.spherical_ops is False
    assert model.spectral is not None
    assert model.hydro is not None
    assert model.detail_gain_head is not None
    assert model.endpoint_preserving is False


def test_canonical_architectures_are_available_through_cli() -> None:
    from tools.train.train_capacity_matched_6h import (
        ARCH_CLI_CHOICES,
        CANONICAL_ARCH_NAMES,
    )

    assert set(CANONICAL_ARCH_NAMES) <= set(ARCH_CLI_CHOICES)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"val_years": [2019, 2020]}, "years overlap"),
        (
            {
                "train_tau_subset": [1, 2, 3, 5],
                "eval_tau": [1, 3, 5],
            },
            "subset of eval_tau",
        ),
        ({"train_tau_subset": [1, 3, 6]}, "out-of-range"),
        ({"samples_per_date_train": 5}, "positive divisor of 24"),
        (
            {
                "train_batches_per_epoch": 13,
                "accumulate": 4,
            },
            "complete accumulation groups",
        ),
    ],
)
def test_rejects_protocol_drift(overrides, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_training_protocol_args(_args(**overrides))


def test_memmap_provenance_detects_content_change(tmp_path) -> None:
    metadata = {
        "T": 2,
        "n_channels": 1,
        "H": 2,
        "W": 2,
    }
    (tmp_path / "wb2_2020.json").write_text(
        __import__("json").dumps(metadata)
    )
    data_path = tmp_path / "wb2_2020.bin"
    data_path.write_bytes(bytes(range(32)))

    before = memmap_dataset_provenance(
        tmp_path,
        [2020],
        sampled_bytes_per_file=12,
    )
    data_path.write_bytes(bytes(reversed(range(32))))
    after = memmap_dataset_provenance(
        tmp_path,
        [2020],
        sampled_bytes_per_file=12,
    )

    assert before["sampled_total_bytes"] == 12
    assert before["files"]["2020"]["shape"] == [2, 1, 2, 2]
    assert before["identity_sha256"] != after["identity_sha256"]


def test_checkpoint_resume_lineage_preserves_prior_protocol(tmp_path) -> None:
    path = tmp_path / "last.ckpt"
    torch.save(
        {
            "epoch": 1,
            "global_step": 3284,
            "hyper_parameters": {
                "arch": "upr_implicit_global_14m",
                "total_steps": 13136,
                "delta_t": 6.0,
                "training_seed": 202707,
                "training_protocol": {"train_years": [2014, 2015]},
                "training_code_sha256": {"trainer.py": "old-hash"},
                "training_input_provenance": {
                    "memmap": {"identity_sha256": "input-hash"}
                },
            },
        },
        path,
    )

    lineage = checkpoint_resume_lineage(path)

    assert lineage["checkpoint_path"] == str(path.resolve())
    assert len(lineage["checkpoint_sha256"]) == 64
    assert lineage["epoch"] == 1
    assert lineage["global_step"] == 3284
    assert lineage["model"]["arch"] == "upr_implicit_global_14m"
    assert lineage["previous_training_code_sha256"] == {
        "trainer.py": "old-hash"
    }
    assert lineage["previous_training_input_provenance"]["memmap"][
        "identity_sha256"
    ] == "input-hash"
