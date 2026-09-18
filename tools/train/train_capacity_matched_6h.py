"""Matched-protocol interpolation trainer for WeatherBridge ablations.

The paper presets set the data, query hours, update budget and loss for
WeatherBridge, WeatherDCAE-14M and PixelAttn-VFI. The remaining architecture
flags cover the transport ablations and HRES adaptation controls.

Public CLI names:

  --arch weatherbridge  -> WeatherBridgeModel (headline model)
  --arch weatherdcae    -> WeatherDCAEAdaLNModel (24-channel 14M baseline)

Historical experiment flags such as ``flow_pp3`` remain accepted so old
launch scripts and checkpoints stay reproducible.

Paper architecture, objective and query protocol:
  bash repro/scripts/train_paper_matched.sh weatherbridge 6
The general CLI also supports non-paper query sets and ablations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from tools.train.training_protocol import (
    checkpoint_resume_lineage,
    memmap_dataset_provenance,
    validate_training_protocol_args,
)
from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from weather_time_interp.hres_finetune_dataset import (
    HRESForecastAnchorDataset,
    hres_finetune_dataset_provenance,
)
from weather_time_interp.normalization import file_provenance


CANONICAL_ARCH_NAMES = {
    "dcae_14m": "WeatherDCAE",
    "wb_vanilla": "WeatherDCAE",
    "wb_skip": "WeatherDCAE-Skip",
    "atmvfi": "PixelAttn-VFI",
    "flow_pp3": "WeatherBridge",
    "flow_pp3_hres_aug": "WeatherBridge-HRES-FEA",
    "flow_pp3_hres_residual": "WeatherBridge-HRES-Residual",
    "flow_pp3_nodiff": "WeatherBridge-NoDiff",
    "flow_pp3_detail": "Legacy Detail Ablation",
    "flow_universal_latent_refine": "WeatherBridge-Universal-Latent-Refine",
}

FLOW_COMPACT_ARCHES = {
    "flow_compact_vp3",
    "flow_compact_vp3_m",
    "flow_compact_vp3_l",
    "flow_compact_hermite_l",
    "flow_compact_lagrange_l",
}
FLOW_COMPACT_HIDDEN = {
    "flow_compact_vp3": 32,
    "flow_compact_vp3_m": 40,
    "flow_compact_vp3_l": 56,
    "flow_compact_hermite_l": 56,
    "flow_compact_lagrange_l": 56,
}
FLOW_PP3_COMPACT_ARCHES = {
    "flow_pp3_compact_l",
}
FLOW_MSF_ARCHES = {
    "flow_msf_pareto_l",
    "flow_geo_msf_l",
}
FLOW_INTRINSIC_ARCHES = {"flow_geo_msf_l"}
FLOW_SPHERICAL_ENCODER_ARCHES = {"flow_pp3_spherical"}
FLOW_SPECTRAL_DETAIL_ARCHES = {
    "flow_pp3_multiband",
    "flow_pp3_detail",
    "flow_pp3_detail_fm",
    "flow_universal_detail",
    "flow_universal_latent",
    "flow_universal_latent_refine",
    "flow_universal_content_refine",
    "flow_universal_pareto_refine",
    "flow_universal_fm",
}
FLOW_DETAIL_ARCHES = {
    "flow_pp3_detail",
    "flow_pp3_detail_fm",
    "flow_universal_detail",
    "flow_universal_latent",
    "flow_universal_latent_refine",
    "flow_universal_content_refine",
    "flow_universal_pareto_refine",
    "flow_universal_fm",
}
FLOW_MATCHING_ARCHES = {
    "dcae_fm_14m",
    "flow_pp3_detail_fm",
    "flow_universal_fm",
}
FLOW_UNIVERSAL_ARCHES = {
    "flow_universal_detail",
    "flow_universal_latent",
    "flow_universal_latent_refine",
    "flow_universal_content_refine",
    "flow_universal_pareto_refine",
    "flow_universal_fm",
}
FLOW_LATENT_TRANSFORMER_ARCHES = {
    "flow_universal_latent",
    "flow_universal_latent_refine",
    "flow_universal_content_refine",
    "flow_universal_pareto_refine",
    "flow_universal_fm",
}
FLOW_REFINED_DECODER_ARCHES = {
    "flow_universal_latent_refine",
    "flow_universal_content_refine",
    "flow_universal_pareto_refine",
}
HRES_LEAD_ARCHES = {
    "flow_pp3_hres_aug",
    "flow_pp3_hres_residual",
    "flow_pp3_degrade",
}
# Degradation-aware arm: identical configuration to the FEA arch plus an
# anchor-reliability branch that predicts the local anchor error field and
# withdraws anchor trust where the forecast anchors are unreliable.
HRES_DEGRADE_ARCHES = {"flow_pp3_degrade"}
# Weight of the reliability supervision relative to the reconstruction
# objective. Pinned by the campaign protocol; the branch is train-only.
ANCHOR_RELIABILITY_WEIGHT = float(
    os.environ.get("ANCHOR_RELIABILITY_WEIGHT", "0.05")
)
# Anchor error spans an order of magnitude between short and long forecast
# leads. Supervising the magnitude directly makes the branch regress toward the
# mean and under-predict the long-lead tail exactly where the trust gate matters
# most; a log-space residual equalises the relative error across leads.
# Learning-rate multiplier for the reliability branch's two control
# scalars. At the fine-tuning rate their gradients are far too small to
# move them off initialisation within the update budget, which leaves the
# control path effectively frozen while the predictor itself trains fine.
# Scaffold ablation: pin and freeze the warp gate. A large positive logit
# drives beta to 1, which removes the linear interpolation scaffold from
# the anchor blend and leaves transport plus residual alone.
WARP_GATE_FREEZE_LOGIT = os.environ.get("WARP_GATE_FREEZE_LOGIT")
ANCHOR_GATE_LR_SCALE = float(os.environ.get("ANCHOR_GATE_LR_SCALE", "1"))
# Weight of the query-time null-space penalty (C1). The harmonic query-time
# basis has more directions than the three trained query hours can constrain;
# cos(3 pi t) is the extreme case, exactly zero at tau in {1,3,5}/6 and +-1 at
# the held-out {2,4}/6, so the training loss never sees it while it moves the
# held-out prediction at full amplitude. This penalises exactly the component
# of the time-embedding weight that lies in that blind subspace.
TAU_NULLSPACE_WEIGHT = float(os.environ.get("TAU_NULLSPACE_WEIGHT", "0"))
# Width of the harmonic query-time basis (C2). 8 is the trained default; 4
# leaves a single poorly-determined direction and 2 leaves none, at the cost of
# how sharply the model may vary with query time.
TIME_FREQ_DIM = int(os.environ.get("TIME_FREQ_DIM", "8"))
ANCHOR_RELIABILITY_LOG_SPACE = (
    os.environ.get("ANCHOR_RELIABILITY_LOG_SPACE", "0") == "1"
)
WAVELET_ARCHES = {"lg_wavelet_10m"}

LOSS_PROFILES = (
    "uniform",
    "base_balanced",
    "base_edge_balanced",
    "pareto_minimax",
    "relative_group_pareto",
)
SPECTRAL_MASK_PROFILES = ("auto", "all", "advected")
DISTILLATION_MASK_PROFILES = ("all", "advected", "moisture")
DISTILLATION_SCHEDULES = ("constant", "cosine_decay")
DISTILLATION_HIGH_GATES = (
    "none",
    "teacher_better",
    "teacher_better_truth",
)
DCAE_LATENT_TAPS = (
    ("encoder.down_blocks.2", 2),
    ("encoder.down_blocks.6", 1),
    ("encoder", 0),
    ("decoder.up_blocks.2", 1),
    ("decoder.up_blocks.6", 2),
)
TRAINABLE_SCOPES = (
    "all",
    "base_knot_head",
    "hres_residual_adapter",
    "q_output_head",
)
PAPER_CHANNEL_NAMES = tuple(
    [
        f"{variable}{level}"
        for variable in ("T", "U", "V", "Q", "Z")
        for level in (1000, 925, 850, 700)
    ]
    + ["t2m", "u10", "v10", "mslp"]
)

ARCH_ALIASES = {
    "weatherbridge": "flow_pp3",
    "weatherbridge_universal_latent_refine": "flow_universal_latent_refine",
    "weatherdcae": "dcae_14m",
    "pixelattn_vfi": "atmvfi",
}

ARCH_CLI_CHOICES = tuple(
    sorted(
        {
            *CANONICAL_ARCH_NAMES,
            *ARCH_ALIASES,
            "flow",
            "flow_noskip",
            "flow_ungated",
            "flow_accel",
            "flow_pp",
            "flow_pp2",
            "flow_pp3_degrade",
        }
    )
)


def canonical_arch_name(arch: str) -> str:
    """Return the paper-facing name while preserving legacy CLI flags."""
    return CANONICAL_ARCH_NAMES.get(ARCH_ALIASES.get(arch, arch), arch)


def exchange_anchors(
    x0: torch.Tensor,
    x1: torch.Tensor,
    tau: torch.Tensor,
    tau_hour: torch.Tensor,
    swap_mask: torch.Tensor,
    delta_t: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the exact interpolation identity (x0,x1,t) -> (x1,x0,1-t)."""
    if swap_mask.shape != (x0.size(0),):
        raise ValueError("swap_mask must have one entry per sample")
    if x0.shape != x1.shape:
        raise ValueError("anchor tensors must have identical shapes")
    sample_mask = swap_mask.to(device=x0.device, dtype=torch.bool)
    field_mask = sample_mask.view(-1, 1, 1, 1)
    exchanged_x0 = torch.where(field_mask, x1, x0)
    exchanged_x1 = torch.where(field_mask, x0, x1)
    exchanged_tau = torch.where(sample_mask, 1.0 - tau, tau)
    exchanged_hour = torch.where(
        sample_mask,
        tau_hour.new_tensor(round(delta_t)) - tau_hour,
        tau_hour,
    )
    return exchanged_x0, exchanged_x1, exchanged_tau, exchanged_hour


def resolve_arch(arch: str) -> str:
    """Resolve a paper-facing CLI name to the checkpoint-compatible flag."""
    return ARCH_ALIASES.get(arch, arch)


def apply_public_arch_defaults(args) -> None:
    """Use the published objective for the public WeatherBridge CLI name.

    Historical flags retain their original defaults for checkpoint replay.
    Explicit loss overrides, including zero, always take precedence.
    """
    if args.arch != "weatherbridge":
        return
    for name, value in (
        ("lambda_hf_override", 0.05),
        ("lambda_spec_override", 0.02),
        ("lambda_band_override", 0.0),
    ):
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.spectral_mask_profile == "auto":
        args.spectral_mask_profile = "advected"


def loss_profile_channel_weights(
    profile: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized reconstruction and high-pass channel weights."""
    if profile not in LOSS_PROFILES:
        raise ValueError(f"unknown loss profile: {profile}")
    reconstruction = torch.ones(24, dtype=torch.float32)
    highpass = torch.ones(24, dtype=torch.float32)
    if profile in ("base_balanced", "base_edge_balanced"):
        # T, U/V, Q, Z, then t2m/u10/v10/mslp. Moisture is deliberately
        # downweighted because Q700 already dominates the compact model's
        # gains; the target objective is broad skill on the other 20 fields.
        reconstruction[:4] = 1.10
        reconstruction[4:12] = 1.10
        reconstruction[12:16] = 0.35
        reconstruction[16:20] = 1.35
        reconstruction[20:] = torch.tensor((1.50, 1.20, 1.20, 1.50))
        reconstruction = reconstruction / reconstruction.mean()

        # Do not make smooth mass fields chase local high-pass noise.
        highpass[16:20] = 0.25
        highpass[20] = 0.25
        highpass[23] = 0.25
        highpass = highpass / highpass.mean()
    return reconstruction, highpass


def loss_profile_tau_weights(
    profile: str,
    tau_hour: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample weights while retaining the sparse 1/3/5 protocol."""
    if profile not in LOSS_PROFILES:
        raise ValueError(f"unknown loss profile: {profile}")
    weights = torch.ones_like(tau_hour, dtype=torch.float32)
    if profile == "base_edge_balanced":
        edge = (tau_hour == 1) | (tau_hour == 5)
        weights = torch.where(edge, weights.new_tensor(1.35), weights)
    return weights


def weighted_latitude_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitude_weights: torch.Tensor,
    channel_weights: torch.Tensor,
    sample_weights: torch.Tensor,
    element_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce a fieldwise L1 objective without changing its nominal scale."""
    absolute_error = (prediction - target).abs()
    if element_weights is not None:
        if element_weights.shape != absolute_error.shape:
            raise ValueError("element weights must match the prediction shape")
        absolute_error = absolute_error * element_weights
    per_sample_channel = (absolute_error * latitude_weights).mean(dim=(-2, -1))
    per_sample = (
        per_sample_channel * channel_weights.view(1, -1)
    ).sum(dim=-1) / channel_weights.sum()
    return (per_sample * sample_weights).sum() / sample_weights.sum()


def teacher_advantage_mask(
    student: torch.Tensor,
    teacher: torch.Tensor,
    truth: torch.Tensor,
) -> torch.Tensor:
    """Select elements where the frozen teacher is closer to truth."""
    if student.shape != teacher.shape or student.shape != truth.shape:
        raise ValueError("student, teacher, and truth must have matching shapes")
    return (
        (teacher.detach() - truth.detach()).abs()
        < (student.detach() - truth.detach()).abs()
    ).to(dtype=student.dtype)


def load_direct_distillation_route(
    path: str | Path,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Load a validation-selected tau-by-channel teacher route."""
    route_path = Path(path)
    payload = json.loads(route_path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported direct-distillation route schema")
    if payload.get("selection_role") != "era5_2020_validation_only":
        raise ValueError("direct-distillation route must be validation-only")
    tau_hours = payload.get("tau_hours")
    channel_names = payload.get("channel_names")
    active = payload.get("teacher_route")
    if channel_names != list(PAPER_CHANNEL_NAMES):
        raise ValueError("direct-distillation route channel order mismatch")
    if (
        not isinstance(tau_hours, list)
        or not tau_hours
        or not all(isinstance(hour, int) and 1 <= hour <= 24 for hour in tau_hours)
        or len(set(tau_hours)) != len(tau_hours)
    ):
        raise ValueError("invalid direct-distillation tau hours")
    if (
        not isinstance(active, list)
        or len(active) != len(tau_hours)
        or any(not isinstance(row, list) or len(row) != 24 for row in active)
        or any(not isinstance(value, bool) for row in active for value in row)
    ):
        raise ValueError("invalid direct-distillation route matrix")
    route = torch.zeros(max(tau_hours) + 1, 24, dtype=torch.float32)
    for hour, row in zip(tau_hours, active, strict=True):
        route[hour] = torch.tensor(row, dtype=torch.float32)
    provenance = {
        "path": str(route_path.resolve()),
        "sha256": hashlib.sha256(route_path.read_bytes()).hexdigest(),
        "selection_role": payload["selection_role"],
        "selection_year": payload.get("selection_year"),
        "active_routes": int(route.sum().item()),
        "source_control_sha256": payload.get("control", {}).get("sha256"),
        "source_teacher_sha256": payload.get("teacher", {}).get("sha256"),
    }
    return route, provenance


def routed_teacher_target(
    truth: torch.Tensor,
    teacher: torch.Tensor,
    route: torch.Tensor,
    blend: float | torch.Tensor,
) -> torch.Tensor:
    """Construct a convex response target without altering excluded fields."""
    if truth.shape != teacher.shape:
        raise ValueError("truth and teacher responses must match")
    if route.shape != (truth.size(0), truth.size(1), 1, 1):
        raise ValueError("direct-distillation route must have shape BC11")
    blend_tensor = torch.as_tensor(blend, device=truth.device, dtype=truth.dtype)
    if bool((blend_tensor < 0.0).any() or (blend_tensor > 1.0).any()):
        raise ValueError("direct-distillation blend must lie in [0, 1]")
    return truth + blend_tensor * route * (teacher - truth)


def pareto_minimax_latitude_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitude_weights: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    """Balance mean skill with the worst four normalized fields per sample."""
    per_sample_channel = (
        (prediction - target).abs() * latitude_weights
    ).mean(dim=(-2, -1))
    mean_loss = per_sample_channel.mean(dim=-1)
    worst_loss = per_sample_channel.topk(
        k=min(4, per_sample_channel.size(1)),
        dim=-1,
    ).values.mean(dim=-1)
    per_sample = 0.5 * mean_loss + 0.5 * worst_loss
    return (per_sample * sample_weights).sum() / sample_weights.sum()


def relative_group_pareto_l1(
    group_errors: torch.Tensor,
    group_scales: torch.Tensor,
    sample_weights: torch.Tensor,
    cvar_fields: int,
) -> torch.Tensor:
    """Combine macro error with CVaR over train-normalized field groups."""
    if group_errors.shape != group_scales.shape:
        raise ValueError("group errors and scales must have matching shapes")
    if group_errors.dim() != 2:
        raise ValueError("group errors must have shape batch-by-channel")
    if sample_weights.shape != (group_errors.size(0),):
        raise ValueError("sample weights must have one value per sample")
    if not 1 <= cvar_fields <= group_errors.size(1):
        raise ValueError("cvar_fields must lie within the channel count")
    if not torch.isfinite(group_scales).all() or (group_scales <= 0).any():
        raise ValueError("group scales must be finite and positive")

    macro = group_errors.mean(dim=-1)
    relative = group_errors / group_scales.detach()
    relative_cvar = relative.topk(cvar_fields, dim=-1).values.mean(dim=-1)
    nominal_scale = group_scales.detach().mean(dim=-1)
    per_sample = 0.5 * macro + 0.5 * relative_cvar * nominal_scale
    return (per_sample * sample_weights).sum() / sample_weights.sum()


def spectral_loss_config(
    arch: str,
    lambda_spec_override: float | None,
    mask_profile: str,
) -> tuple[float, torch.Tensor | None, str]:
    """Resolve the spectral weight and channel mask recorded by the run."""
    if mask_profile not in SPECTRAL_MASK_PROFILES:
        raise ValueError(f"unknown spectral mask profile: {mask_profile}")
    default_weight = (
        0.05
        if arch in ("flow_pp", "flow_pp2")
        else 0.02 if arch in FLOW_MSF_ARCHES
        else 0.0
    )
    weight = (
        default_weight
        if lambda_spec_override is None
        else float(lambda_spec_override)
    )
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(
            "lambda_spec_override must be finite and non-negative"
        )
    effective_profile = (
        "advected"
        if mask_profile == "auto" and arch == "flow_pp2"
        else "all" if mask_profile == "auto"
        else mask_profile
    )
    if effective_profile == "all":
        return weight, None, effective_profile
    mask = torch.ones(24, dtype=torch.float32)
    for channel in (16, 17, 18, 19, 20, 23):
        mask[channel] = 0.0
    return weight, mask, effective_profile


def distillation_channel_mask(profile: str) -> torch.Tensor:
    """Return the explicit channel mask used by teacher supervision."""
    if profile not in DISTILLATION_MASK_PROFILES:
        raise ValueError(f"unknown distillation mask profile: {profile}")
    mask = torch.ones(24, dtype=torch.float32)
    if profile == "advected":
        # Smooth mass/thermodynamic fields are deliberately kept on the
        # truth-primary WeatherDCAE path instead of borrowing spectral detail.
        mask[[16, 17, 18, 19, 20, 23]] = 0.0
    elif profile == "moisture":
        mask.zero_()
        mask[12:16] = 1.0
    return mask


def distillation_schedule_scale(
    schedule: str,
    global_step: int,
    total_steps: int,
    decay_start_fraction: float,
    decay_end_fraction: float,
) -> float:
    """Return the deterministic teacher-weight multiplier for one step."""
    if schedule not in DISTILLATION_SCHEDULES:
        raise ValueError(f"unknown distillation schedule: {schedule}")
    if schedule == "constant":
        return 1.0
    progress = min(1.0, (int(global_step) + 1) / max(1, total_steps))
    if progress <= decay_start_fraction:
        return 1.0
    if progress >= decay_end_fraction:
        return 0.0
    phase = (progress - decay_start_fraction) / (
        decay_end_fraction - decay_start_fraction
    )
    return 0.5 * (1.0 + math.cos(math.pi * phase))


def interval_corrected_distillation_blend(
    blend: float,
    every_n_steps: int,
    global_step: int,
    schedule_scale: float,
) -> float:
    """Preserve mean direct-teacher influence while skipping teacher forwards."""
    if every_n_steps < 1:
        raise ValueError("distillation interval must be positive")
    if int(global_step) % every_n_steps:
        return 0.0
    return min(1.0, float(blend) * every_n_steps) * float(schedule_scale)


def latent_block_weights(block_decay: float) -> dict[str, float]:
    """Return normalized DCAE tap weights decaying away from the bottleneck."""
    if not math.isfinite(block_decay) or not 0.0 < block_decay <= 1.0:
        raise ValueError("latent block decay must lie in (0, 1]")
    raw = {
        name: block_decay ** distance
        for name, distance in DCAE_LATENT_TAPS
    }
    normalizer = sum(raw.values())
    return {name: value / normalizer for name, value in raw.items()}


def normalized_latent_distillation_loss(
    student_features: dict[str, torch.Tensor],
    teacher_features: dict[str, torch.Tensor],
    block_decay: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match pooled same-architecture features on the teacher RMS scale."""
    weights = latent_block_weights(block_decay)
    if set(student_features) != set(weights):
        raise ValueError("student latent taps do not match the DCAE protocol")
    if set(teacher_features) != set(weights):
        raise ValueError("teacher latent taps do not match the DCAE protocol")
    losses: dict[str, torch.Tensor] = {}
    total = None
    for name, weight in weights.items():
        student = student_features[name].float()
        teacher = teacher_features[name].detach().float()
        if student.shape != teacher.shape:
            raise ValueError(
                f"latent shape mismatch at {name}: "
                f"student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
            )
        teacher_scale = teacher.square().mean(
            dim=(-2, -1),
            keepdim=True,
        ).sqrt().clamp_min(1.0e-4)
        loss = F.smooth_l1_loss(
            student / teacher_scale,
            teacher / teacher_scale,
            beta=0.5,
        )
        losses[name] = loss
        weighted = weight * loss
        total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError("latent distillation requires at least one tap")
    return total, losses


def apply_trainable_scope(net: nn.Module, scope: str) -> None:
    """Apply an explicit fine-tuning scope without changing model state."""
    if scope not in TRAINABLE_SCOPES:
        raise ValueError(f"unknown trainable scope: {scope}")
    if scope == "all":
        for parameter in net.parameters():
            parameter.requires_grad_(True)
        return
    if scope == "q_output_head":
        decoder = getattr(net, "decoder", None)
        head = getattr(decoder, "conv_out", None)
        if not isinstance(head, nn.Conv2d) or head.out_channels != 24:
            raise ValueError(
                "q_output_head scope requires a 24-channel decoder.conv_out"
            )
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        head.weight.requires_grad_(True)
        if head.bias is not None:
            head.bias.requires_grad_(True)

        def mask_output_gradient(gradient: torch.Tensor) -> torch.Tensor:
            mask = gradient.new_zeros((gradient.size(0),))
            mask[12:16] = 1.0
            return gradient * mask.view(
                gradient.size(0), *([1] * (gradient.ndim - 1))
            )

        handles = [head.weight.register_hook(mask_output_gradient)]
        if head.bias is not None:
            handles.append(head.bias.register_hook(mask_output_gradient))
        net._q_output_gradient_handles = tuple(handles)
        return
    if scope == "hres_residual_adapter":
        head = getattr(net, "hres_residual_head", None)
        if head is None:
            raise ValueError(
                "hres_residual_adapter scope requires an HRES adapter"
            )
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        for parameter in head.parameters():
            parameter.requires_grad_(True)
        return
    head = getattr(net, "base_knot_head", None)
    if head is None:
        raise ValueError(
            "base_knot_head scope requires a Lagrange architecture"
        )
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(True)


def trainable_scope_weight_decay(scope: str) -> float:
    """Avoid changing masked output rows through decoupled weight decay."""
    if scope not in TRAINABLE_SCOPES:
        raise ValueError(f"unknown trainable scope: {scope}")
    return 0.0 if scope == "q_output_head" else 1.0e-4


def resolve_train_batch_schedule(
    *,
    dataset_size: int,
    batch_size_per_device: int,
    accumulate: int,
    devices: int,
    requested_microbatches: int = 0,
    limit_fraction: float = 1.0,
) -> tuple[int, int, int]:
    """Return complete per-device accumulation groups and optimizer steps."""
    available = dataset_size // (batch_size_per_device * devices)
    if available < accumulate:
        raise ValueError("dataset cannot form one complete accumulation group")
    if requested_microbatches:
        if requested_microbatches > available:
            raise ValueError(
                "train_batches_per_epoch exceeds available per-device "
                f"train batches: {requested_microbatches} > {available}"
            )
        selected = requested_microbatches
    else:
        selected = min(
            available,
            max(accumulate, int(available * limit_fraction)),
        )
    aligned = selected - selected % accumulate
    if aligned < accumulate:
        raise ValueError("train schedule has no complete accumulation group")
    return aligned, aligned // accumulate, available - aligned


def _atmvfi_source(repo_root: Path) -> Path:
    """Prefer the checked-in ATM-VFI source over an untracked root mirror."""
    candidates = (
        repo_root / "legacy" / "scripts" / "train_atm_vfi_12h_oddskip.py",
        repo_root / "train_atm_vfi_12h_oddskip.py",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "ATM-VFI implementation not found in legacy/scripts or repository root"
    )


# Opt-in single-conv fast path for every SphereConv2d (3x fewer launches,
# no per-forward weight.clone; poles negligible). Enable with WTI_FAST_SPHERE=1.
if __import__("os").environ.get("WTI_FAST_SPHERE"):
    from weather_time_interp.model.sphere_conv import enable_fast_sphere_conv
    enable_fast_sphere_conv()
    print("[sphere] fast single-conv path ENABLED", flush=True)


class TauRescaleAnd24chWrapper(Dataset):
    """Select the paper fields and rescale tau = tau_hour / delta_t."""

    def __init__(self, base, delta_t=6.0, n_keep=24):
        self.base = base
        self.delta_t = float(delta_t)
        self.n_keep = n_keep

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = self.base[idx]
        if "tau_hour" in out:
            th = out["tau_hour"].float()
            if out["tau"].shape == th.shape:
                out["tau"] = (th / self.delta_t).view_as(out["tau"])
            else:
                out["tau"] = th / self.delta_t
        for k in ("x0", "x1", "target"):
            if k in out and isinstance(out[k], torch.Tensor) and out[k].dim() >= 3:
                out[k] = out[k][..., : self.n_keep, :, :].contiguous()
        return out


def highpass_component(
    field: torch.Tensor,
    pole_parity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a local high-pass field with the requested pole topology."""
    if field.ndim != 4:
        raise ValueError("high-pass input must have shape (B,C,H,W)")
    if pole_parity is None:
        padded = F.pad(field, (1, 1, 0, 0), mode="circular")
        padded = F.pad(padded, (0, 0, 1, 1), mode="replicate")
    else:
        parity = torch.as_tensor(
            pole_parity,
            device=field.device,
            dtype=field.dtype,
        )
        if parity.numel() != field.size(1):
            raise ValueError(
                "pole parity must have one value per high-pass channel"
            )
        width = field.size(-1)
        half_width = width // 2

        def antipodal_shift(value: torch.Tensor) -> torch.Tensor:
            shifted = torch.roll(value, half_width, dims=-1)
            if width % 2:
                shifted = 0.5 * (
                    shifted
                    + torch.roll(value, half_width + 1, dims=-1)
                )
            return shifted

        parity = parity.view(1, -1, 1, 1)
        top = antipodal_shift(field[..., :1, :]) * parity
        bottom = antipodal_shift(field[..., -1:, :]) * parity
        padded = torch.cat((top, field, bottom), dim=-2)
        padded = F.pad(padded, (1, 1, 0, 0), mode="circular")
    return field - F.avg_pool2d(padded, kernel_size=3, stride=1)


def weather_field_pole_parity(channels: int = 24) -> torch.Tensor:
    """Return scalar/vector parity for the paper's fixed field order."""
    if channels != 24:
        raise ValueError("distillation requires the 24-field paper protocol")
    parity = torch.ones(channels, dtype=torch.float32)
    parity[[4, 5, 6, 7, 8, 9, 10, 11, 21, 22]] = -1.0
    return parity


def lowpass_component(
    field: torch.Tensor,
    pole_parity: torch.Tensor,
) -> torch.Tensor:
    """Return the phase-preserving local base band complementary to high-pass."""
    return field - highpass_component(field, pole_parity)


def compose_multiteacher_target(
    low_teacher: torch.Tensor,
    high_teacher: torch.Tensor,
    high_channel_mask: torch.Tensor,
    pole_parity: torch.Tensor,
) -> torch.Tensor:
    """Combine one teacher's base band with another teacher's detail band."""
    if low_teacher.shape != high_teacher.shape:
        raise ValueError("teacher predictions must have identical shapes")
    mask = torch.as_tensor(
        high_channel_mask,
        device=low_teacher.device,
        dtype=low_teacher.dtype,
    ).view(1, -1, 1, 1)
    if mask.size(1) != low_teacher.size(1):
        raise ValueError("teacher mask must have one value per channel")
    composite = lowpass_component(
        low_teacher,
        pole_parity,
    ) + highpass_component(high_teacher, pole_parity)
    return low_teacher + mask * (composite - low_teacher)


def multiband_spectral_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitude_weights: torch.Tensor,
    pole_parity: torch.Tensor,
) -> torch.Tensor:
    """Match spherical detail-band energy and relative spectral shape."""
    from weather_time_interp.model.weatherbridge_flow_model import (
        spherical_multiband_components,
    )

    pred_bands = spherical_multiband_components(
        prediction.float(),
        pole_parity,
    )
    target_bands = spherical_multiband_components(
        target.float(),
        pole_parity,
    )
    pred_energy = torch.stack(
        [
            ((band.square() * latitude_weights).mean(dim=(-2, -1)) + 1.0e-8)
            .sqrt()
            for band in pred_bands
        ],
        dim=-1,
    )
    target_energy = torch.stack(
        [
            ((band.square() * latitude_weights).mean(dim=(-2, -1)) + 1.0e-8)
            .sqrt()
            for band in target_bands
        ],
        dim=-1,
    )
    log_energy = (
        torch.log(pred_energy + 1.0e-4)
        - torch.log(target_energy + 1.0e-4)
    ).abs()
    pred_shape = pred_energy / pred_energy.sum(dim=-1, keepdim=True)
    target_shape = target_energy / target_energy.sum(dim=-1, keepdim=True)
    log_shape = (
        torch.log(pred_shape + 1.0e-4)
        - torch.log(target_shape + 1.0e-4)
    ).abs()
    return 0.5 * log_energy.mean() + 0.5 * log_shape.mean()


def assert_24ch_protocol(base: Dataset, wrapped: Dataset, split: str) -> None:
    """Fail early if the memmap/stat channel order drifts from the paper setup."""
    expected_pl = [f"{v}{lvl}" for v in ["T", "U", "V", "Q", "Z"]
                   for lvl in [1000, 925, 850, 700]]
    expected_surf = ["t2m", "u10", "v10", "mslp", "sst", "tcc", "tcwv"]
    if base.channel_names != expected_pl:
        raise RuntimeError(f"{split}: pressure channel order mismatch: {base.channel_names}")
    if base.surface_variables != expected_surf:
        raise RuntimeError(f"{split}: surface channel order mismatch: {base.surface_variables}")
    sample = wrapped[0]
    for key in ("x0", "x1", "target"):
        if sample[key].shape[-3] != 24:
            raise RuntimeError(f"{split}: {key} has {sample[key].shape[-3]} channels, expected 24")
    print(
        f"  {split}_channels=24 "
        f"pl={expected_pl[0]}..{expected_pl[-1]} "
        f"surface_keep={expected_surf[:4]} "
        f"sample_shape={tuple(sample['x0'].shape)}",
        flush=True,
    )


class ATMVFIStaticNet(nn.Module):
    """Legacy PixelAttn-VFI with static-augmented encoder.

    Encoder sees [frame; static] (24+n_static ch); the bilinear scaffold and
    residual head operate over the 24 prognostic channels. Mirrors the
    paper's static-augmented encoder while keeping the prognostic output on
    the common 24-field protocol.
    forward(x0, xT, tau, cond=None, static=None) -> (B, 24, H, W).
    """

    def __init__(self, hidden=72, n_levels=3, n_static=3):
        super().__init__()
        import importlib.util
        root = Path(__file__).resolve().parents[2]
        source = _atmvfi_source(root)
        spec = importlib.util.spec_from_file_location(
            "train_atm_vfi",
            source,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import ATM-VFI implementation from {source}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.n_static = n_static
        # Encoder input = 24 + static; output residual stays 24ch.
        self.net = mod.PixelAttentionVFINet(in_ch=24 + n_static, hidden=hidden,
                                            n_levels=n_levels)
        ch0 = self.net.out_conv.weight.shape[1]
        self.net.out_conv = nn.Conv2d(ch0, 24, kernel_size=3, padding=1)
        nn.init.zeros_(self.net.out_conv.weight)
        nn.init.zeros_(self.net.out_conv.bias)
        self.net.scale = nn.Parameter(torch.full((24,), 0.10))

    def forward(self, x0, xT, tau, cond=None, static=None):
        n = self.net  # PixelAttentionVFINet
        tau_b = tau.view(-1, 1, 1, 1) if tau.dim() <= 2 else tau
        x_bilinear = (1.0 - tau_b) * x0 + tau_b * xT   # 24ch
        B, _, H, W = x0.shape
        st = static
        if st.dim() == 3:
            st = st.unsqueeze(0)
        if st.size(0) != B:
            st = st.expand(B, -1, -1, -1)
        a0 = torch.cat([x0, st], dim=1)
        aT = torch.cat([xT, st], dim=1)
        f0 = n.encode_frame(a0)
        fT = n.encode_frame(aT)
        h = n.attn(f0[-1], fT[-1])
        h = n.adaln(h, tau.view(-1))
        n_lev = len(n.decoder) // 2
        for i in range(n_lev):
            up = n.decoder[2 * i]
            blk = n.decoder[2 * i + 1]
            h = up(h)
            skip = (f0[-(i + 2)] + fT[-(i + 2)]) / 2
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear",
                                   align_corners=False)
            h = blk(torch.cat([h, skip], dim=1))
        if h.shape[-2:] != x0.shape[-2:]:
            h = F.interpolate(h, size=x0.shape[-2:], mode="bilinear",
                               align_corners=False)
        delta = n.out_conv(h)
        s = torch.tanh(n.scale).view(1, -1, 1, 1)
        return x_bilinear + s * delta


def build_net(arch: str, static_path: str):
    """Return (net, is_atmvfi, needs_cond_static)."""
    arch = resolve_arch(arch)
    if arch not in ARCH_CLI_CHOICES:
        raise ValueError(f"architecture is not part of the paper release: {arch}")
    degradation_aware = arch in HRES_DEGRADE_ARCHES
    if degradation_aware:
        # Reuse the FEA configuration verbatim so the only difference is
        # the reliability branch and its zero-init gains.
        arch = "flow_pp3_hres_aug"
    if arch == "wb_vanilla":
        from weather_time_interp.model.dcae_adaln_model import WeatherDCAEAdaLNModel
        net = WeatherDCAEAdaLNModel(
            in_channels=24, out_channels=24, n_static_features=3,
            latent_channels=32, attention_head_dim=32,
            block_type=("ResBlock", "ResBlock", "EfficientViTBlock"),
            qkv_multiscales=((), (), (5,)), lat_crop=-8,
            block_out_channels=(128, 128, 256, 256), layers_per_block=(2, 2, 2),
        )
        return net, False, True
    if arch == "dcae_14m":
        from weather_time_interp.model.dcae_adaln_model import (
            WeatherDCAEAdaLNModel,
        )
        net = WeatherDCAEAdaLNModel(
            in_channels=24,
            out_channels=24,
            n_static_features=3,
            latent_channels=256,
            attention_head_dim=32,
            block_type=("ResBlock", "ResBlock", "EfficientViTBlock"),
            qkv_multiscales=((), (), (5,)),
            lat_crop=-8,
            block_out_channels=(64, 128, 256),
            layers_per_block=(3, 3, 3),
        )
        return net, False, True
    if arch == "wb_skip":
        # Strict WeatherDCAE Skip arm. Every backbone hyperparameter matches
        # dcae_14m; only two zero-init scalar skip gates are added.
        from weather_time_interp.model.dcae_adaln_skip_model import (
            WeatherDCAEAdaLNSkipModel,
        )
        net = WeatherDCAEAdaLNSkipModel(
            in_channels=24,
            out_channels=24,
            n_static_features=3,
            latent_channels=256,
            attention_head_dim=32,
            block_type=("ResBlock", "ResBlock", "EfficientViTBlock"),
            qkv_multiscales=((), (), (5,)),
            lat_crop=-8,
            block_out_channels=(64, 128, 256),
            layers_per_block=(3, 3, 3),
            skip_lateral_rank=0,
            tau_conditional_gates=False,
        )
        return net, False, True
    if arch == "atmvfi":
        # Static-augmented PixelAttn-VFI: the
        # encoder sees [frame; static] so the model can localise land/sea
        # diurnal response; residual + bilinear scaffold stay 24ch. Trained
        # with static like the WB-family for a fair capacity-matched compare.
        net = ATMVFIStaticNet(hidden=72, n_levels=3, n_static=3)
        return net, False, True
    if arch in (
        "flow",
        "flow_noskip",
        "flow_ungated",
        "flow_accel",
        "flow_pp",
        "flow_pp2",
        "flow_pp3",
        "flow_pp3_hres_aug",
        "flow_pp3_hres_residual",
        "flow_pp3_nodiff",
        *FLOW_SPHERICAL_ENCODER_ARCHES,
        *FLOW_SPECTRAL_DETAIL_ARCHES,
        "flow_dual",
        "flow_spherical_ep",
        *FLOW_MSF_ARCHES,
        *FLOW_PP3_COMPACT_ARCHES,
        *FLOW_COMPACT_ARCHES,
    ):
        # WeatherBridge: transport-aware (flow-warp + learned tau-blend +
        # pyramid residual), frame-diff input, tau-gated skips. hidden=72 →
        # ~14.1M (gated skip), matched to the rest. Ablation arms:
        #   flow          -> tau-gated skips (main candidate), linear F*tau warp
        #   flow_ungated  -> plain skips (reproduces WB-Skip overshoot risk)
        #   flow_noskip   -> no skips
        #   flow_accel    -> gated skips + constant-acceleration warp F*tau+1/2 A*tau^2
        #   flow_pp       -> flow_accel + strengthened (conv-block) residual head
        #                    (+ light spectral loss, set on CapMatchedLit)
        from weather_time_interp.model.weatherbridge_flow_model import (
            WeatherBridgeModel,
        )
        # flow_dual: transport stream + skip-less global decoder stream, gated
        # per-channel fusion. Hidden trimmed 72->64 to fund the 2nd decoder at
        # the matched ~14M budget.
        _hidden = (
            FLOW_COMPACT_HIDDEN[arch]
            if arch in FLOW_COMPACT_ARCHES
            else 56 if arch in FLOW_MSF_ARCHES
            else 56 if arch in FLOW_PP3_COMPACT_ARCHES
            else 64 if arch == "flow_dual"
            else 64 if arch in FLOW_LATENT_TRANSFORMER_ARCHES
            else 72
        )
        net = WeatherBridgeModel(
            in_channels=24, out_channels=24, n_static_features=3,
            hidden=_hidden, n_levels=3,
            use_skip=(arch != "flow_noskip"),
            gated_skip=(arch != "flow_ungated"),
            use_accel=(
                arch
                in (
                    "flow_accel",
                    "flow_pp",
                    "flow_pp2",
                    "flow_pp3",
                    "flow_pp3_hres_aug",
                    "flow_pp3_hres_residual",
                    "flow_pp3_nodiff",
                    *FLOW_SPHERICAL_ENCODER_ARCHES,
                    *FLOW_SPECTRAL_DETAIL_ARCHES,
                    *FLOW_MSF_ARCHES,
                    *FLOW_PP3_COMPACT_ARCHES,
                    "flow_dual",
                    "flow_spherical_ep",
                )
            ),
            # flow_pp3 = clean accel base + SFNO global branch + hydrostatic
            # coupling ONLY. mass_aware_gate (beta->0 for Z/mslp) is dropped:
            # it REGRESSED mass fields (flow_pp2) because synoptic pressure
            # systems DO advect, so the warp helps them. strong_residual dropped
            # to isolate the two new mass levers.
            strong_residual=(arch in ("flow_pp", "flow_pp2")),
            mass_aware_gate=(arch == "flow_pp2"),
            time_freq_dim=TIME_FREQ_DIM,
            spectral_branch=(
                arch
                in (
                    "flow_pp3",
                    "flow_pp3_hres_aug",
                    "flow_pp3_hres_residual",
                    "flow_pp3_nodiff",
                    *FLOW_SPHERICAL_ENCODER_ARCHES,
                    *FLOW_SPECTRAL_DETAIL_ARCHES,
                    "flow_spherical_ep",
                    *FLOW_MSF_ARCHES,
                    *FLOW_PP3_COMPACT_ARCHES,
                )
            ),
            hydro_couple=(
                arch
                in (
                    "flow_pp3",
                    "flow_pp3_hres_aug",
                    "flow_pp3_hres_residual",
                    "flow_pp3_nodiff",
                    *FLOW_SPHERICAL_ENCODER_ARCHES,
                    *FLOW_SPECTRAL_DETAIL_ARCHES,
                    "flow_spherical_ep",
                    *FLOW_MSF_ARCHES,
                    *FLOW_PP3_COMPACT_ARCHES,
                    *FLOW_COMPACT_ARCHES,
                )
                and arch not in FLOW_UNIVERSAL_ARCHES
            ),
            dual_stream=(arch == "flow_dual"),
            spherical_ops=(
                arch == "flow_spherical_ep"
                or arch in FLOW_SPHERICAL_ENCODER_ARCHES
                or arch in FLOW_MSF_ARCHES
                or arch in FLOW_COMPACT_ARCHES
            ),
            endpoint_preserving=(
                arch == "flow_spherical_ep"
                or arch in FLOW_MSF_ARCHES
                or arch in FLOW_COMPACT_ARCHES
                or arch in FLOW_MATCHING_ARCHES
                or arch in FLOW_UNIVERSAL_ARCHES
            ),
            query_independent_trajectory=(
                arch == "flow_spherical_ep"
                or arch in FLOW_MSF_ARCHES
                or arch in FLOW_COMPACT_ARCHES
            ),
            n_flow_modes=3 if arch in FLOW_COMPACT_ARCHES else 1,
            cubic_trajectory=(arch in FLOW_COMPACT_ARCHES),
            global_tokens=8 if arch in FLOW_COMPACT_ARCHES else 0,
            global_token_dim=32,
            endpoint_tangents=(
                arch
                in (
                    "flow_compact_hermite_l",
                    "flow_compact_lagrange_l",
                    *FLOW_MSF_ARCHES,
                )
            ),
            base_field_knots=(arch == "flow_compact_lagrange_l"),
            multiscale_field_experts=(arch in FLOW_MSF_ARCHES),
            intrinsic_spherical_transport=(arch in FLOW_INTRINSIC_ARCHES),
            use_frame_difference=(arch != "flow_pp3_nodiff"),
            multiband_calibration=(arch == "flow_pp3_multiband"),
            anchor_detail_bypass=(arch in FLOW_DETAIL_ARCHES),
            flow_matching=(arch in FLOW_MATCHING_ARCHES),
            flow_matching_steps=4,
            shared_field_controls=(arch in FLOW_UNIVERSAL_ARCHES),
            latent_transformer_tokens=(
                16 if arch in FLOW_LATENT_TRANSFORMER_ARCHES else 0
            ),
            latent_transformer_dim=128,
            latent_transformer_depth=2,
            latent_transformer_heads=4,
            decoder_blocks_per_level=(
                2 if arch in FLOW_REFINED_DECODER_ARCHES else 1
            ),
            content_adaptive_controls=(
                arch == "flow_universal_content_refine"
            ),
            time_content_adaptive_controls=(
                arch == "flow_universal_pareto_refine"
            ),
            forecast_lead_conditioning=(arch in HRES_LEAD_ARCHES),
            hres_residual_adapter=(arch == "flow_pp3_hres_residual"),
            degradation_aware=degradation_aware,
        )
        if WARP_GATE_FREEZE_LOGIT is not None and hasattr(net, "warp_gate"):
            with torch.no_grad():
                net.warp_gate.fill_(float(WARP_GATE_FREEZE_LOGIT))
            net.warp_gate.requires_grad_(False)
        return net, False, True
    raise ValueError(f"unknown arch {arch}")


class PooledFeatureCollector:
    """Capture bounded-memory feature taps without changing model outputs."""

    def __init__(
        self,
        model: nn.Module,
        pool_height: int,
        pool_width: int,
    ) -> None:
        if pool_height < 1 or pool_width < 1:
            raise ValueError("latent pool dimensions must be positive")
        modules = dict(model.named_modules())
        missing = [name for name, _ in DCAE_LATENT_TAPS if name not in modules]
        if missing:
            raise ValueError(f"model lacks DCAE latent taps: {missing}")
        self.pool_height = int(pool_height)
        self.pool_width = int(pool_width)
        self.enabled = False
        self.features: dict[str, torch.Tensor] = {}
        self._handles = [
            modules[name].register_forward_hook(self._hook(name))
            for name, _ in DCAE_LATENT_TAPS
        ]

    def _hook(self, name: str):
        def capture(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: torch.Tensor,
        ) -> None:
            if not self.enabled:
                return
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise RuntimeError(f"latent tap {name} did not return BCHW")
            target = (
                min(self.pool_height, output.size(-2)),
                min(self.pool_width, output.size(-1)),
            )
            self.features[name] = (
                F.adaptive_avg_pool2d(output, target)
                if output.shape[-2:] != target
                else output
            )

        return capture

    def begin(self) -> None:
        self.features = {}
        self.enabled = True

    def end(self, *, detach: bool) -> dict[str, torch.Tensor]:
        self.enabled = False
        result = {
            name: value.detach() if detach else value
            for name, value in self.features.items()
        }
        self.features = {}
        expected = {name for name, _ in DCAE_LATENT_TAPS}
        if set(result) != expected:
            raise RuntimeError("DCAE forward did not populate every latent tap")
        return result


class FrozenTeacherRuntime:
    """A frozen inference model intentionally excluded from student state."""

    def __init__(
        self,
        checkpoint_path: str,
        static_path: str,
        expected_delta_t: float,
    ) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        hparams = dict(checkpoint.get("hyper_parameters", {}))
        self.arch = str(hparams.get("arch", ""))
        if not self.arch:
            raise ValueError(
                f"{checkpoint_path}: teacher checkpoint is missing arch"
            )
        self.delta_t = float(hparams.get("delta_t", expected_delta_t))
        if not math.isclose(self.delta_t, expected_delta_t, abs_tol=1.0e-9):
            raise ValueError(
                f"{checkpoint_path}: teacher delta_t={self.delta_t:g} does "
                f"not match student delta_t={expected_delta_t:g}"
            )
        if self.arch in FLOW_INTRINSIC_ARCHES:
            raise ValueError(
                "intrinsic-transport teachers require checkpoint-specific "
                "normalization and are not supported for distillation"
            )
        state = checkpoint.get("state_dict", checkpoint)
        net_state = {
            name.removeprefix("net."): value
            for name, value in state.items()
            if name.startswith("net.")
        }
        if not net_state:
            raise ValueError(f"{checkpoint_path}: no teacher net state found")
        self.model, self.is_atmvfi, self.needs_cond = build_net(
            self.arch,
            static_path,
        )
        self.model.load_state_dict(net_state, strict=True)
        self.model.requires_grad_(False).eval()
        self.lineage = checkpoint_resume_lineage(checkpoint_path)
        self._latent_collector: PooledFeatureCollector | None = None

    def enable_latent_capture(
        self,
        pool_height: int,
        pool_width: int,
    ) -> None:
        if self.arch != "dcae_14m":
            raise ValueError(
                "latent distillation requires a dcae_14m teacher"
            )
        self._latent_collector = PooledFeatureCollector(
            self.model,
            pool_height,
            pool_width,
        )

    def to(self, device: torch.device) -> None:
        self.model.to(device).eval()

    def predict(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        tau: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        parameter = next(self.model.parameters())
        if parameter.device != x0.device:
            self.to(x0.device)
        with torch.no_grad():
            if self.is_atmvfi:
                output = self.model(x0, x1, tau)
            else:
                batch_size = x0.size(0)
                cond = torch.full(
                    (batch_size,),
                    self.delta_t,
                    device=x0.device,
                    dtype=torch.float32,
                )
                output = self.model(
                    x0,
                    x1,
                    tau,
                    cond,
                    static=static,
                )
        return (output[0] if isinstance(output, tuple) else output).detach()

    def predict_latents(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        tau: torch.Tensor,
        static: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self._latent_collector is None:
            raise RuntimeError("latent capture was not configured")
        self._latent_collector.begin()
        try:
            self.predict(x0, x1, tau, static)
            return self._latent_collector.end(detach=True)
        except BaseException:
            self._latent_collector.enabled = False
            self._latent_collector.features = {}
            raise


class CapMatchedLit(pl.LightningModule):
    def __init__(self, arch: str, static_path: str, lr: float,
                 delta_t: float, eval_taus, train_taus=(), warmup_steps: int = 500,
                 total_steps: int = 100000, training_seed: int = 202707,
                  lambda_hf_override: float | None = None,
                  lambda_spec_override: float | None = None,
                  lambda_band_override: float | None = None,
                  lambda_sht_override: float = 0.0,
                  sht_ell_min: int = 80,
                  sht_lmax: int = 180,
                  sht_start_fraction: float = 0.6,
                  sht_every_n_steps: int = 8,
                 spectral_mask_profile: str = "auto",
                 loss_profile: str = "uniform",
                 group_balance_ema_decay: float = 0.99,
                 group_balance_cvar_fields: int = 6,
                 trainable_scope: str = "all",
                 anchor_swap_probability: float = 0.0,
                 distill_low_teacher_checkpoint: str | None = None,
                 distill_high_teacher_checkpoint: str | None = None,
                 distill_direct_teacher_checkpoint: str | None = None,
                 distill_direct_route_path: str | None = None,
                 distill_direct_blend: float = 0.0,
                 distill_low_weight: float = 0.0,
                 distill_high_weight: float = 0.0,
                 distill_anchor_weight: float = 0.0,
                 distill_high_mask_profile: str = "advected",
                 distill_high_gate: str = "none",
                 distill_every_n_steps: int = 1,
                 distill_schedule: str = "constant",
                 distill_decay_start_fraction: float = 0.0,
                 distill_decay_end_fraction: float = 1.0,
                 distill_latent_teacher_checkpoint: str | None = None,
                 distill_latent_weight: float = 0.0,
                 distill_latent_every_n_steps: int = 2,
                 distill_latent_schedule: str = "constant",
                 distill_latent_decay_start_fraction: float = 0.0,
                 distill_latent_decay_end_fraction: float = 1.0,
                 distill_latent_block_decay: float = 0.5,
                 distill_latent_pool_height: int = 45,
                 distill_latent_pool_width: int = 90):
        super().__init__()
        self.save_hyperparameters()
        self.arch = arch
        # Phase-sensitive local high-pass supervision is substantially cheaper
        # than a full-resolution FFT and cannot be satisfied by adding
        # spectrally plausible noise. The plain UPR-Lite arm retains the exact
        # L1-only recipe as an architecture control.
        default_lambda_hf = 0.05 if arch in (
            "upr_lite_lap",
            "upr_lite_column",
            "upr_lite_continuous",
            "upr_lite_continuous_m",
            "upr_lite_implicit_global",
            "upr_lite_implicit_global_q4",
            "upr_implicit_global_14m",
            "upr_endpoint_implicit_global_14m",
            "upr_local_corr_14m",
            "upr_query_match_14m",
            "upr_universal_latent_q4_10m",
            "upr_spherical_implicit_global_14m",
            "flow_spherical_ep",
            *FLOW_MSF_ARCHES,
            *FLOW_COMPACT_ARCHES,
            *WAVELET_ARCHES,
            *FLOW_SPECTRAL_DETAIL_ARCHES,
            "amt_residual",
        ) else 0.0
        self.lambda_tau_null = TAU_NULLSPACE_WEIGHT
        self.lambda_hf = (
            default_lambda_hf
            if lambda_hf_override is None
            else float(lambda_hf_override)
        )
        if not math.isfinite(self.lambda_hf) or self.lambda_hf < 0.0:
            raise ValueError(
                "lambda_hf_override must be finite and non-negative"
            )
        default_lambda_band = (
            0.02 if arch in FLOW_SPECTRAL_DETAIL_ARCHES else 0.0
        )
        self.lambda_band = (
            default_lambda_band
            if lambda_band_override is None
            else float(lambda_band_override)
        )
        if not math.isfinite(self.lambda_band) or self.lambda_band < 0.0:
            raise ValueError(
                "lambda_band_override must be finite and non-negative"
            )
        self.lambda_sht = float(lambda_sht_override)
        self.sht_ell_min = int(sht_ell_min)
        self.sht_lmax = int(sht_lmax)
        self.sht_start_fraction = float(sht_start_fraction)
        self.sht_every_n_steps = int(sht_every_n_steps)
        if not math.isfinite(self.lambda_sht) or self.lambda_sht < 0.0:
            raise ValueError("lambda_sht_override must be finite and non-negative")
        if not 0.0 <= self.sht_start_fraction < 1.0:
            raise ValueError("sht_start_fraction must lie in [0, 1)")
        if self.sht_every_n_steps < 1:
            raise ValueError("sht_every_n_steps must be positive")
        (
            self.lambda_spec,
            spectral_mask,
            self.spectral_mask_profile,
        ) = spectral_loss_config(
            arch,
            lambda_spec_override,
            spectral_mask_profile,
        )
        if spectral_mask is not None:
            self.register_buffer(
                "spec_mask",
                spectral_mask.view(1, 24, 1, 1),
                persistent=False,
            )
        else:
            self.spec_mask = None
        self.net, self.is_atmvfi, self.needs_cond = build_net(arch, static_path)
        self.trainable_scope = trainable_scope
        self.anchor_swap_probability = float(anchor_swap_probability)
        if (
            not math.isfinite(self.anchor_swap_probability)
            or not 0.0 <= self.anchor_swap_probability <= 1.0
        ):
            raise ValueError(
                "anchor_swap_probability must lie in [0, 1]"
            )
        self.distill_low_weight = float(distill_low_weight)
        self.distill_high_weight = float(distill_high_weight)
        self.distill_anchor_weight = float(distill_anchor_weight)
        self.distill_direct_blend = float(distill_direct_blend)
        self.distill_every_n_steps = int(distill_every_n_steps)
        for label, weight in (
            ("low", self.distill_low_weight),
            ("high", self.distill_high_weight),
            ("anchor", self.distill_anchor_weight),
        ):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"distill_{label}_weight must be finite and non-negative"
                )
        if (
            not math.isfinite(self.distill_direct_blend)
            or not 0.0 <= self.distill_direct_blend <= 1.0
        ):
            raise ValueError("distill_direct_blend must lie in [0, 1]")
        direct_enabled = self.distill_direct_blend > 0.0
        if direct_enabled != bool(distill_direct_teacher_checkpoint):
            raise ValueError(
                "direct blend and direct teacher checkpoint must be enabled together"
            )
        if direct_enabled != bool(distill_direct_route_path):
            raise ValueError(
                "direct blend and direct route path must be enabled together"
            )
        low_teacher_required = (
            self.distill_low_weight > 0.0
            or self.distill_anchor_weight > 0.0
        )
        if low_teacher_required != bool(distill_low_teacher_checkpoint):
            raise ValueError(
                "low/anchor distillation weights and low teacher checkpoint "
                "must be enabled together"
            )
        if (self.distill_high_weight > 0.0) != bool(
            distill_high_teacher_checkpoint
        ):
            raise ValueError(
                "distill_high_weight and high teacher checkpoint must be "
                "enabled together"
            )
        if self.distill_every_n_steps < 1:
            raise ValueError("distill_every_n_steps must be positive")
        if distill_schedule not in DISTILLATION_SCHEDULES:
            raise ValueError(f"unknown distillation schedule: {distill_schedule}")
        self.distill_schedule = distill_schedule
        self.distill_decay_start_fraction = float(
            distill_decay_start_fraction
        )
        self.distill_decay_end_fraction = float(distill_decay_end_fraction)
        if not (
            math.isfinite(self.distill_decay_start_fraction)
            and math.isfinite(self.distill_decay_end_fraction)
            and 0.0 <= self.distill_decay_start_fraction
            < self.distill_decay_end_fraction
            <= 1.0
        ):
            raise ValueError(
                "distillation decay fractions must satisfy 0 <= start < end <= 1"
            )
        self.distill_latent_weight = float(distill_latent_weight)
        self.distill_latent_every_n_steps = int(
            distill_latent_every_n_steps
        )
        self.distill_latent_schedule = distill_latent_schedule
        self.distill_latent_decay_start_fraction = float(
            distill_latent_decay_start_fraction
        )
        self.distill_latent_decay_end_fraction = float(
            distill_latent_decay_end_fraction
        )
        self.distill_latent_block_decay = float(
            distill_latent_block_decay
        )
        self.distill_latent_pool_height = int(distill_latent_pool_height)
        self.distill_latent_pool_width = int(distill_latent_pool_width)
        if not math.isfinite(self.distill_latent_weight) or self.distill_latent_weight < 0.0:
            raise ValueError(
                "distill_latent_weight must be finite and non-negative"
            )
        if (self.distill_latent_weight > 0.0) != bool(
            distill_latent_teacher_checkpoint
        ):
            raise ValueError(
                "latent distillation weight and teacher checkpoint must be "
                "enabled together"
            )
        if self.distill_latent_weight > 0.0 and self.arch != "dcae_14m":
            raise ValueError(
                "latent distillation currently requires a dcae_14m student"
            )
        if self.distill_latent_every_n_steps < 1:
            raise ValueError("distill_latent_every_n_steps must be positive")
        if self.distill_latent_schedule not in DISTILLATION_SCHEDULES:
            raise ValueError(
                f"unknown latent distillation schedule: "
                f"{self.distill_latent_schedule}"
            )
        if not (
            math.isfinite(self.distill_latent_decay_start_fraction)
            and math.isfinite(self.distill_latent_decay_end_fraction)
            and 0.0 <= self.distill_latent_decay_start_fraction
            < self.distill_latent_decay_end_fraction
            <= 1.0
        ):
            raise ValueError(
                "latent distillation decay fractions must satisfy "
                "0 <= start < end <= 1"
            )
        latent_block_weights(self.distill_latent_block_decay)
        if (
            self.distill_latent_pool_height < 1
            or self.distill_latent_pool_width < 1
        ):
            raise ValueError("latent pool dimensions must be positive")
        self.distill_high_mask_profile = distill_high_mask_profile
        if distill_high_gate not in DISTILLATION_HIGH_GATES:
            raise ValueError(f"unknown high-teacher gate: {distill_high_gate}")
        self.distill_high_gate = distill_high_gate
        self.register_buffer(
            "distill_high_channel_mask",
            distillation_channel_mask(distill_high_mask_profile),
            persistent=False,
        )
        if (
            self.distill_anchor_weight > 0.0
            and self.distill_high_channel_mask.min().item() > 0.0
        ):
            raise ValueError(
                "distill_anchor_weight requires a high mask with excluded fields"
            )
        self.register_buffer(
            "distill_pole_parity",
            weather_field_pole_parity(),
            persistent=False,
        )
        if distill_direct_route_path:
            direct_route, direct_route_provenance = (
                load_direct_distillation_route(distill_direct_route_path)
            )
        else:
            direct_route = torch.zeros(1, 24, dtype=torch.float32)
            direct_route_provenance = None
        self.register_buffer(
            "distill_direct_route",
            direct_route,
            persistent=False,
        )
        self.distill_direct_route_provenance = direct_route_provenance
        apply_trainable_scope(self.net, trainable_scope)
        spherical_highpass_arches = {
            "amt",
            "amt_residual",
            "flow_spherical_ep",
            *FLOW_SPHERICAL_ENCODER_ARCHES,
            *FLOW_MSF_ARCHES,
            *FLOW_COMPACT_ARCHES,
            *FLOW_SPECTRAL_DETAIL_ARCHES,
            "upr_spherical_implicit_global_14m",
            *WAVELET_ARCHES,
        }
        if arch in spherical_highpass_arches:
            pole_parity = getattr(self.net, "field_pole_parity", None)
            if (
                not isinstance(pole_parity, torch.Tensor)
                or pole_parity.numel() != 24
            ):
                raise ValueError(
                    f"{arch} lacks the 24-field spherical pole parity"
                )
            self.register_buffer(
                "hf_pole_parity",
                pole_parity.detach().clone(),
                persistent=False,
            )
            self.highpass_boundary = "antipodal_vector_parity"
        else:
            self.register_buffer(
                "hf_pole_parity",
                torch.empty(0),
                persistent=False,
            )
            self.highpass_boundary = "periodic_lon_replicate_lat"
        self.lr = lr
        self.delta_t = float(delta_t)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.loss_profile = loss_profile
        self.group_balance_ema_decay = float(group_balance_ema_decay)
        self.group_balance_cvar_fields = int(group_balance_cvar_fields)
        if not 0.0 <= self.group_balance_ema_decay < 1.0:
            raise ValueError("group balance EMA decay must lie in [0, 1)")
        if not 1 <= self.group_balance_cvar_fields <= 24:
            raise ValueError("group balance CVaR fields must lie in [1, 24]")
        channel_weights, highpass_weights = loss_profile_channel_weights(
            loss_profile
        )
        self.register_buffer(
            "loss_channel_weights",
            channel_weights,
            persistent=False,
        )
        self.register_buffer(
            "loss_highpass_weights",
            highpass_weights,
            persistent=False,
        )

        static = torch.load(static_path, weights_only=False).float()
        lsm = static[0]
        H = lsm.shape[0]
        W = lsm.shape[1]
        self.grid_h = int(H)
        self.grid_w = int(W)
        lat = torch.linspace(89.75, -89.75, H, dtype=torch.float32)
        lw = torch.cos(torch.deg2rad(lat))
        lw = lw / lw.sum() * H
        self.register_buffer("lat_w", lw.view(1, 1, -1, 1), persistent=False)
        self.register_buffer("static3", static[:3].unsqueeze(0), persistent=False)
        self._distill_low_teacher = (
            FrozenTeacherRuntime(
                distill_low_teacher_checkpoint,
                static_path,
                self.delta_t,
            )
            if distill_low_teacher_checkpoint
            else None
        )
        self._distill_high_teacher = (
            FrozenTeacherRuntime(
                distill_high_teacher_checkpoint,
                static_path,
                self.delta_t,
            )
            if distill_high_teacher_checkpoint
            else None
        )
        self._distill_direct_teacher = (
            FrozenTeacherRuntime(
                distill_direct_teacher_checkpoint,
                static_path,
                self.delta_t,
            )
            if distill_direct_teacher_checkpoint
            else None
        )
        self._distill_latent_teacher = (
            FrozenTeacherRuntime(
                distill_latent_teacher_checkpoint,
                static_path,
                self.delta_t,
            )
            if distill_latent_teacher_checkpoint
            else None
        )
        self._student_latent_collector: PooledFeatureCollector | None = None
        if self._distill_latent_teacher is not None:
            if self._distill_latent_teacher.arch != self.arch:
                raise ValueError(
                    "latent teacher and student architectures must match"
                )
            self._distill_latent_teacher.enable_latent_capture(
                self.distill_latent_pool_height,
                self.distill_latent_pool_width,
            )
            self._student_latent_collector = PooledFeatureCollector(
                self.net,
                self.distill_latent_pool_height,
                self.distill_latent_pool_width,
            )
        if self.lambda_sht > 0.0:
            if not 1 <= self.sht_ell_min <= self.sht_lmax < H:
                raise ValueError(
                    "SHT loss requires 1 <= sht_ell_min <= sht_lmax < grid height"
                )
            from weather_time_interp.metrics.spherical_spectra import (
                CellCenteredRealSHT,
            )

            self.sht_loss_operator = CellCenteredRealSHT(
                H,
                W,
                lmax=self.sht_lmax + 1,
                mmax=self.sht_lmax + 1,
            )
        else:
            self.sht_loss_operator = None

        if self.loss_profile == "relative_group_pareto":
            n_tau_slots = int(round(self.delta_t)) + 1
            self.register_buffer(
                "group_loss_ema",
                torch.zeros(n_tau_slots, 24),
            )
            self.register_buffer(
                "group_loss_initialized",
                torch.zeros(n_tau_slots, dtype=torch.bool),
            )
        else:
            self.register_buffer(
                "group_loss_ema",
                torch.empty(0),
                persistent=False,
            )
            self.register_buffer(
                "group_loss_initialized",
                torch.empty(0, dtype=torch.bool),
                persistent=False,
            )

        self.train_taus = list(train_taus)
        self.eval_taus = list(eval_taus)
        # Needs both train_taus and the built net, so it is registered here
        # rather than beside the other loss weights.
        self.register_buffer(
            "tau_null_basis", self._tau_null_basis(), persistent=False
        )
        for h in self.eval_taus:
            self.register_buffer(f"val_sq_h{h}", torch.zeros(24), persistent=False)
            self.register_buffer(f"val_cnt_h{h}", torch.tensor(0.0), persistent=False)

    def _relative_group_objective(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        tau_hour: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        group_errors = (
            (prediction - target).abs() * self.lat_w
        ).mean(dim=(-2, -1))
        hour_indices = tau_hour.long()
        if int(hour_indices.min()) < 0 or int(hour_indices.max()) >= int(
            self.group_loss_ema.size(0)
        ):
            raise ValueError("tau hour is outside the group-balance table")
        with torch.no_grad():
            for hour in torch.unique(hour_indices):
                index = int(hour.item())
                observed = group_errors[hour_indices == hour].mean(dim=0)
                if bool(self.group_loss_initialized[index]):
                    self.group_loss_ema[index].lerp_(
                        observed,
                        1.0 - self.group_balance_ema_decay,
                    )
                else:
                    self.group_loss_ema[index].copy_(observed)
                    self.group_loss_initialized[index] = True
        scales = self.group_loss_ema.index_select(0, hour_indices).clamp_min(
            1.0e-6
        )
        return relative_group_pareto_l1(
            group_errors,
            scales,
            sample_weights,
            self.group_balance_cvar_fields,
        )

    def _forward(self, x0, x1, tau, forecast_lead_hours=None):
        if self.is_atmvfi:
            return self.net(x0, x1, tau)
        B = x0.size(0)
        if self.arch in HRES_LEAD_ARCHES:
            if forecast_lead_hours is None:
                raise ValueError("WeatherBridge-HRES-FEA requires forecast lead")
            cond = forecast_lead_hours.reshape(-1).to(
                device=x0.device,
                dtype=torch.float32,
            )
        else:
            cond = torch.full(
                (B,), self.delta_t, device=x0.device, dtype=torch.float32
            )
        static = self.static3.expand(B, -1, -1, -1)
        out = self.net(x0, x1, tau, cond, static=static)
        if isinstance(out, tuple):
            self._last_forward_aux = out[1]
            return out[0]
        self._last_forward_aux = None
        return out

    def _anchor_reliability_objective(self, batch):
        """Supervise the predicted anchor error with the measured one."""
        target = batch.get("anchor_error")
        aux = getattr(self, "_last_forward_aux", None)
        if target is None or not isinstance(aux, dict):
            return None
        predicted = aux.get("predicted_anchor_error")
        if predicted is None:
            return None
        target = target.to(
            device=predicted.device,
            dtype=predicted.dtype,
        )
        if ANCHOR_RELIABILITY_LOG_SPACE:
            floor = 1.0e-3
            return (
                (
                    torch.log(predicted + floor)
                    - torch.log(target + floor)
                ).abs()
                * self.lat_w
            ).mean()
        return ((predicted - target).abs() * self.lat_w).mean()

    def _direct_distillation_target(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        truth: torch.Tensor,
        tau: torch.Tensor,
        tau_hour: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Blend truth with routed teacher responses during early training."""
        if self._distill_direct_teacher is None:
            return truth, {}
        schedule_scale = self._distillation_schedule_scale()
        blend = interval_corrected_distillation_blend(
            self.distill_direct_blend,
            self.distill_every_n_steps,
            int(self.global_step),
            schedule_scale,
        )
        if blend <= 0.0:
            return truth, {
                "direct_schedule_scale": truth.new_zeros(()),
                "direct_route_fraction": truth.new_zeros(()),
                "direct_effective_blend": truth.new_zeros(()),
            }
        if int(tau_hour.max()) >= self.distill_direct_route.size(0):
            raise ValueError("direct-distillation route lacks a requested tau")
        static = self.static3.expand(x0.size(0), -1, -1, -1)
        teacher = self._distill_direct_teacher.predict(
            x0,
            x1,
            tau,
            static,
        ).detach()
        route = self.distill_direct_route[tau_hour.long()].view(
            truth.size(0), truth.size(1), 1, 1
        )
        target = routed_teacher_target(truth, teacher, route, blend)
        return target, {
            "direct_schedule_scale": truth.new_tensor(schedule_scale),
            "direct_route_fraction": route.mean(),
            "direct_effective_blend": route.mean() * blend,
        }

    def _distillation_objective(
        self,
        prediction: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        target: torch.Tensor,
        tau: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return scheduled low/high teacher supervision."""
        zero = prediction.new_zeros(())
        if (
            self.distill_low_weight <= 0.0
            and self.distill_high_weight <= 0.0
            and self.distill_anchor_weight <= 0.0
        ):
            return zero, {}
        if int(self.global_step) % self.distill_every_n_steps:
            return zero, {}
        schedule_scale = self._distillation_schedule_scale()
        if schedule_scale <= 0.0:
            return zero, {}
        static = self.static3.expand(x0.size(0), -1, -1, -1)
        components: dict[str, torch.Tensor] = {
            "schedule_scale": prediction.new_tensor(schedule_scale),
        }
        objective = zero
        if self._distill_low_teacher is not None:
            low_target = self._distill_low_teacher.predict(
                x0,
                x1,
                tau,
                static,
            )
            low = weighted_latitude_l1(
                lowpass_component(prediction, self.distill_pole_parity),
                lowpass_component(low_target, self.distill_pole_parity),
                self.lat_w,
                torch.ones_like(self.distill_high_channel_mask),
                sample_weights,
            )
            components["low"] = low
            objective = objective + self.distill_low_weight * low
            if self.distill_anchor_weight > 0.0:
                anchor = weighted_latitude_l1(
                    prediction,
                    low_target,
                    self.lat_w,
                    1.0 - self.distill_high_channel_mask,
                    sample_weights,
                )
                components["anchor"] = anchor
                objective = objective + self.distill_anchor_weight * anchor
        if self._distill_high_teacher is not None:
            high_target = self._distill_high_teacher.predict(
                x0,
                x1,
                tau,
                static,
            )
            prediction_high = highpass_component(
                prediction,
                self.distill_pole_parity,
            )
            teacher_high = highpass_component(
                high_target,
                self.distill_pole_parity,
            )
            element_weights = None
            high_loss_target = teacher_high
            if self.distill_high_gate in {
                "teacher_better",
                "teacher_better_truth",
            }:
                truth_high = highpass_component(
                    target,
                    self.distill_pole_parity,
                )
                element_weights = teacher_advantage_mask(
                    prediction_high,
                    teacher_high,
                    truth_high,
                )
                active = self.distill_high_channel_mask.view(1, -1, 1, 1)
                components["high_gate_fraction"] = (
                    (element_weights * active).sum()
                    / active.sum().clamp_min(1.0)
                    / prediction.shape[0]
                    / prediction.shape[-2]
                    / prediction.shape[-1]
                )
                if self.distill_high_gate == "teacher_better_truth":
                    high_loss_target = truth_high
            high = weighted_latitude_l1(
                prediction_high,
                high_loss_target,
                self.lat_w,
                self.distill_high_channel_mask,
                sample_weights,
                element_weights,
            )
            components["high"] = high
            objective = objective + self.distill_high_weight * high
        return (
            objective * self.distill_every_n_steps * schedule_scale,
            components,
        )

    def _distillation_schedule_scale(self) -> float:
        return distillation_schedule_scale(
            self.distill_schedule,
            int(self.global_step),
            self.total_steps,
            self.distill_decay_start_fraction,
            self.distill_decay_end_fraction,
        )

    def _latent_distillation_schedule_scale(self) -> float:
        return distillation_schedule_scale(
            self.distill_latent_schedule,
            int(self.global_step),
            self.total_steps,
            self.distill_latent_decay_start_fraction,
            self.distill_latent_decay_end_fraction,
        )

    def _latent_distillation_due(self) -> bool:
        return bool(
            self.distill_latent_weight > 0.0
            and int(self.global_step) % self.distill_latent_every_n_steps == 0
            and self._latent_distillation_schedule_scale() > 0.0
        )

    def _latent_distillation_objective(
        self,
        student_features: dict[str, torch.Tensor],
        x0: torch.Tensor,
        x1: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not student_features:
            return x0.new_zeros(()), {}
        if self._distill_latent_teacher is None:
            raise RuntimeError("latent features were captured without a teacher")
        static = self.static3.expand(x0.size(0), -1, -1, -1)
        teacher_features = self._distill_latent_teacher.predict_latents(
            x0,
            x1,
            tau,
            static,
        )
        latent, block_losses = normalized_latent_distillation_loss(
            student_features,
            teacher_features,
            self.distill_latent_block_decay,
        )
        schedule_scale = self._latent_distillation_schedule_scale()
        components = {
            "latent": latent,
            "latent_schedule_scale": latent.new_tensor(schedule_scale),
            **{
                f"latent_{name.replace('.', '_')}": value
                for name, value in block_losses.items()
            },
        }
        scaled = (
            self.distill_latent_weight
            * self.distill_latent_every_n_steps
            * schedule_scale
            * latent
        )
        return scaled, components

    def _sht_objective_scale(self) -> float:
        """Return a late-ramped, interval-corrected SHT objective weight."""
        if self.lambda_sht <= 0.0 or self.sht_loss_operator is None:
            return 0.0
        if int(self.global_step) % self.sht_every_n_steps:
            return 0.0
        progress = min(1.0, (int(self.global_step) + 1) / max(1, self.total_steps))
        if progress < self.sht_start_fraction:
            return 0.0
        ramp = (progress - self.sht_start_fraction) / (
            1.0 - self.sht_start_fraction
        )
        return self.lambda_sht * min(1.0, ramp) * self.sht_every_n_steps

    def _step(self, batch, prefix):
        x0, x1, tgt = batch["x0"], batch["x1"], batch["target"]
        tau = batch["tau"]
        if tau.dim() > 1:
            tau = tau.view(-1)
        tau_hour = batch["tau_hour"].view(-1).long()
        forecast_lead_hours = batch.get("forecast_lead_hours")
        if prefix == "train" and self.anchor_swap_probability > 0.0:
            swap_mask = (
                torch.rand(x0.size(0), device=x0.device)
                < self.anchor_swap_probability
            )
            x0, x1, tau, tau_hour = exchange_anchors(
                x0,
                x1,
                tau,
                tau_hour,
                swap_mask,
                self.delta_t,
            )
            self.log(
                "train/anchor_swap_fraction",
                swap_mask.float().mean(),
                sync_dist=True,
            )
        student_latent_features: dict[str, torch.Tensor] = {}
        latent_distillation_due = (
            prefix == "train" and self._latent_distillation_due()
        )
        if latent_distillation_due:
            if self._student_latent_collector is None:
                raise RuntimeError("latent distillation collector is missing")
            self._student_latent_collector.begin()
        if prefix == "train" and self.arch in FLOW_MATCHING_ARCHES:
            batch_size = x0.size(0)
            tau_b = tau.view(-1, 1, 1, 1)
            x_linear = (1.0 - tau_b) * x0 + tau_b * x1
            flow_time = torch.rand(
                batch_size,
                device=x0.device,
                dtype=x0.dtype,
            )
            flow_time_b = flow_time.view(-1, 1, 1, 1)
            bridge_state = (
                (1.0 - flow_time_b) * x_linear
                + flow_time_b * tgt
            )
            cond = torch.full(
                (batch_size,),
                self.delta_t,
                device=x0.device,
                dtype=torch.float32,
            )
            static = self.static3.expand(batch_size, -1, -1, -1)
            velocity, _ = self.net.flow_matching_velocity(
                x0,
                x1,
                tau,
                bridge_state,
                flow_time,
                cond,
                static,
            )
            pred = x_linear + velocity
            self.log(
                "train/flow_time_mean",
                flow_time.mean(),
                sync_dist=True,
            )
        else:
            pred = self._forward(x0, x1, tau, forecast_lead_hours)
        if latent_distillation_due:
            student_latent_features = self._student_latent_collector.end(
                detach=False,
            )
        optimization_target = tgt
        if prefix == "train":
            optimization_target, direct_components = (
                self._direct_distillation_target(
                    x0,
                    x1,
                    tgt,
                    tau,
                    tau_hour,
                )
            )
            for name, value in direct_components.items():
                self.log(f"train/distill_{name}", value, sync_dist=True)
        recon = ((pred - tgt).abs() * self.lat_w).mean()
        self.log(f"{prefix}/recon_l1", recon, sync_dist=True, prog_bar=True)
        if prefix == "train":
            sample_weights = loss_profile_tau_weights(
                self.loss_profile,
                tau_hour,
            )
            if self.loss_profile == "pareto_minimax":
                err = pareto_minimax_latitude_l1(
                    pred,
                    optimization_target,
                    self.lat_w,
                    sample_weights,
                )
            elif self.loss_profile == "relative_group_pareto":
                err = self._relative_group_objective(
                    pred,
                    optimization_target,
                    tau_hour,
                    sample_weights,
                )
            else:
                err = weighted_latitude_l1(
                    pred,
                    optimization_target,
                    self.lat_w,
                    self.loss_channel_weights,
                    sample_weights,
                )
            if self.loss_profile != "uniform":
                self.log("train/objective_l1", err, sync_dist=True)
        else:
            err = recon
        if prefix == "train":
            distill, distill_components = self._distillation_objective(
                pred,
                x0,
                x1,
                tgt,
                tau,
                sample_weights,
            )
            for name, value in distill_components.items():
                self.log(f"train/distill_{name}", value, sync_dist=True)
            if distill_components:
                self.log(
                    "train/distill_interval_scale",
                    torch.as_tensor(
                        self.distill_every_n_steps,
                        device=pred.device,
                    ),
                    sync_dist=True,
                )
                err = err + distill
            latent_distill, latent_components = (
                self._latent_distillation_objective(
                    student_latent_features,
                    x0,
                    x1,
                    tau,
                )
            )
            for name, value in latent_components.items():
                self.log(f"train/distill_{name}", value, sync_dist=True)
            if latent_components:
                self.log(
                    "train/distill_latent_interval_scale",
                    torch.as_tensor(
                        self.distill_latent_every_n_steps,
                        device=pred.device,
                    ),
                    sync_dist=True,
                )
                err = err + latent_distill
            reliability = self._anchor_reliability_objective(batch)
            if reliability is not None:
                self.log(
                    "train/anchor_reliability_l1",
                    reliability,
                    sync_dist=True,
                )
                err = err + ANCHOR_RELIABILITY_WEIGHT * reliability
        if prefix == "train" and self.lambda_spec > 0:
            # Legacy planar all-frequency FFT-magnitude objective. Retained only
            # for exact reproduction of existing WeatherBridge checkpoints.
            pf = torch.fft.rfft2(pred.float(), norm="ortho").abs()
            tf = torch.fft.rfft2(tgt.float(), norm="ortho").abs()
            if self.spec_mask is not None:
                per_ch = (pf - tf).abs().mean(dim=(0, 2, 3))          # (C,)
                w = self.spec_mask.view(-1)
                spec = (per_ch * w).sum() / w.sum()
            else:
                spec = (pf - tf).abs().mean()
            self.log("train/spec", spec, sync_dist=True)
            err = err + self.lambda_spec * spec
        if prefix == "train" and self.lambda_hf > 0:
            pole_parity = (
                self.hf_pole_parity
                if self.hf_pole_parity.numel()
                else None
            )
            pred_hf = highpass_component(pred, pole_parity)
            tgt_hf = highpass_component(tgt, pole_parity)
            if self.loss_profile in {
                "pareto_minimax",
                "relative_group_pareto",
            }:
                hf = pareto_minimax_latitude_l1(
                    pred_hf,
                    tgt_hf,
                    self.lat_w,
                    sample_weights,
                )
            else:
                hf = weighted_latitude_l1(
                    pred_hf,
                    tgt_hf,
                    self.lat_w,
                    self.loss_highpass_weights,
                    sample_weights,
                )
            self.log("train/highpass", hf, sync_dist=True)
            err = err + self.lambda_hf * hf
        if prefix == "train" and self.lambda_band > 0:
            if not self.hf_pole_parity.numel():
                raise RuntimeError(
                    "multiband loss requires spherical field parity"
                )
            band = multiband_spectral_loss(
                pred,
                tgt,
                self.lat_w,
                self.hf_pole_parity,
            )
            self.log("train/multiband", band, sync_dist=True)
            err = err + self.lambda_band * band
        if prefix == "train":
            sht_scale = self._sht_objective_scale()
            if sht_scale > 0.0:
                from weather_time_interp.metrics.spherical_spectra import (
                    aligned_sht_band_objective,
                )

                channel_weights = (
                    self.spec_mask.reshape(-1)
                    if self.spec_mask is not None
                    else None
                )
                components = aligned_sht_band_objective(
                    pred,
                    tgt,
                    self.sht_loss_operator,
                    ell_min=self.sht_ell_min,
                    channel_weights=channel_weights,
                )
                self.log(
                    "train/sht_amplitude",
                    components["amplitude"],
                    sync_dist=True,
                )
                self.log("train/sht_shape", components["shape"], sync_dist=True)
                self.log("train/sht_phase", components["phase"], sync_dist=True)
                self.log(
                    "train/sht_ramp_scale",
                    torch.as_tensor(sht_scale, device=pred.device),
                    sync_dist=True,
                )
                err = err + sht_scale * components["total"].to(err.dtype)
        if prefix == "val":
            sq = (pred.float() - tgt.float()) ** 2
            sq_lw = (sq * self.lat_w).sum(dim=(-2, -1))  # (B, C)
            for h in self.eval_taus:
                mask = tau_hour == h
                if mask.any():
                    getattr(self, f"val_sq_h{h}").add_(sq_lw[mask].sum(dim=0).detach())
                    getattr(self, f"val_cnt_h{h}").add_(mask.float().sum())
        if prefix == "train" and self.lambda_tau_null > 0:
            penalty = self._tau_nullspace_penalty()
            if penalty is not None:
                self.log("train/tau_nullspace", penalty, sync_dist=True)
                err = err + self.lambda_tau_null * penalty.to(err.dtype)
        return err

    def _tau_null_basis(self) -> torch.Tensor:
        """Directions of the query-time basis that the trained hours cannot see.

        Rows of the basis matrix are the harmonic features at each trained
        query hour. Any coefficient direction in its null space leaves every
        training prediction unchanged while moving the held-out ones, so it is
        determined by initialisation and weight decay alone.
        """
        taus = sorted({float(t) for t in self.train_taus})
        if not taus:
            return torch.zeros(0, 0)
        net = self.net
        freq = int(getattr(net, "time_freq_dim", 8))
        half = freq // 2
        t = torch.tensor(taus, dtype=torch.float64) / float(self.delta_t)
        k = torch.arange(1, half + 1, dtype=torch.float64)
        a = t.view(-1, 1) * (math.pi * k).view(1, -1)
        basis = torch.cat([a.sin(), a.cos()], dim=1)          # (n_tau, freq)
        _, sv, vh = torch.linalg.svd(basis, full_matrices=True)
        rank = int((sv > 1e-9).sum())
        return vh[rank:].T.float().contiguous()               # (freq, n_null)

    def _tau_nullspace_penalty(self):
        basis = self.tau_null_basis
        if basis.numel() == 0:
            return None
        weight = self.net.time_mlp.net[0].weight               # (dim, freq)
        blind = weight @ basis.to(weight.dtype).to(weight.device)
        # Scale-free: the share of the layer's energy sitting in the blind
        # subspace, so the weight does not have to be retuned with the model.
        return blind.pow(2).sum() / weight.pow(2).sum().clamp_min(1e-12)

    def training_step(self, b, i):
        return self._step(b, "train")

    def validation_step(self, b, i):
        return self._step(b, "val")

    def on_validation_epoch_end(self):
        if self.trainer is not None and self.trainer.world_size > 1:
            for h in self.eval_taus:
                torch.distributed.all_reduce(getattr(self, f"val_sq_h{h}"))
                torch.distributed.all_reduce(getattr(self, f"val_cnt_h{h}"))
        n_px = self.grid_h * self.grid_w
        rmses = []
        for h in self.eval_taus:
            cnt = getattr(self, f"val_cnt_h{h}").item()
            if cnt > 0:
                mse = getattr(self, f"val_sq_h{h}") / (cnt * n_px)
                rmse = mse.mean().sqrt().item()
                rmses.append(rmse)
                self.log(f"val/rmse_h{h}", rmse, sync_dist=False)
                for channel_name, channel_rmse in zip(
                    PAPER_CHANNEL_NAMES,
                    mse.sqrt(),
                    strict=True,
                ):
                    self.log(
                        f"val/rmse_{channel_name}_h{h}",
                        channel_rmse.item(),
                        sync_dist=False,
                    )
            getattr(self, f"val_sq_h{h}").zero_()
            getattr(self, f"val_cnt_h{h}").zero_()
        if rmses:
            self.log("val/rmse_mean", sum(rmses) / len(rmses), prog_bar=True)

    def configure_optimizers(self):
        trainable = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("no trainable parameters")
        decay = trainable_scope_weight_decay(self.trainable_scope)
        gate_names = {"net.anchor_trust_gain", "net.anchor_residual_gain"}
        gates = [
            parameter
            for name, parameter in self.named_parameters()
            if name in gate_names and parameter.requires_grad
        ]
        if gates and ANCHOR_GATE_LR_SCALE != 1.0:
            gate_ids = {id(parameter) for parameter in gates}
            body = [
                parameter
                for parameter in trainable
                if id(parameter) not in gate_ids
            ]
            opt = torch.optim.AdamW(
                [
                    {"params": body, "lr": self.lr, "weight_decay": decay},
                    {
                        "params": gates,
                        "lr": self.lr * ANCHOR_GATE_LR_SCALE,
                        "weight_decay": 0.0,
                    },
                ]
            )
        else:
            opt = torch.optim.AdamW(
                trainable,
                lr=self.lr,
                weight_decay=decay,
            )

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            prog = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=ARCH_CLI_CHOICES)
    p.add_argument("--years", nargs="+", type=int,
                   default=[2014, 2015, 2016, 2017, 2018, 2019])
    p.add_argument("--val_years", nargs="+", type=int, default=[2020])
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--val_bs", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val_workers", type=int, default=4)
    p.add_argument(
        "--release_memmap_pages",
        action="store_true",
        help=(
            "Release copied file-backed time slices from worker page mappings."
        ),
    )
    p.add_argument("--max_epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--exp_name", required=True)
    p.add_argument("--gpus", nargs="+", type=int, default=[0])
    p.add_argument("--memmap_dir", default="/tmp/wb2_0p5_cache")
    p.add_argument(
        "--data_source",
        choices=("era5", "hres_era5"),
        default="era5",
        help="Use ERA5 anchors or same-trajectory HRES anchors with ERA5 targets.",
    )
    p.add_argument(
        "--forecast_train_dir",
        default=None,
        help="Canonical HRES archive containing training initialisations.",
    )
    p.add_argument(
        "--forecast_val_dir",
        default=None,
        help="Canonical HRES validation archive; defaults to forecast_train_dir.",
    )
    p.add_argument("--forecast_lead_stride_hours", type=int, default=24)
    p.add_argument("--forecast_max_left_lead_hours", type=int, default=120)
    p.add_argument("--forecast_error_augmentation_repeats", type=int, default=1)
    p.add_argument("--forecast_error_scale_min", type=float, default=1.0)
    p.add_argument("--forecast_error_scale_max", type=float, default=1.0)
    p.add_argument("--forecast_error_augmentation_seed", type=int, default=0)
    p.add_argument("--static_path", default="data/static_features_0p5.pt")
    p.add_argument("--stats_path", default="data/json_stats_0p5.nc")
    p.add_argument("--surface_stats_path", default="data/surface_stats_0p5.json")
    p.add_argument("--window_hours", type=int, default=6)
    p.add_argument("--train_tau_subset", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--eval_tau", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--samples_per_date_train", type=int, default=4)
    p.add_argument("--samples_per_date_val", type=int, default=2)
    p.add_argument(
        "--lambda_hf_override",
        type=float,
        default=None,
        help="Override the architecture-default local high-pass loss weight.",
    )
    p.add_argument(
        "--lambda_spec_override",
        type=float,
        default=None,
        help="Override the architecture-default FFT-magnitude loss weight.",
    )
    p.add_argument(
        "--lambda_band_override",
        type=float,
        default=None,
        help="Override the spherical multiband energy/shape loss weight.",
    )
    p.add_argument(
        "--lambda_sht_override",
        type=float,
        default=0.0,
        help="Weight of the cell-centred SHT amplitude/shape/phase objective.",
    )
    p.add_argument("--sht_ell_min", type=int, default=80)
    p.add_argument("--sht_lmax", type=int, default=180)
    p.add_argument(
        "--sht_start_fraction",
        type=float,
        default=0.6,
        help="Training progress at which the SHT objective starts ramping.",
    )
    p.add_argument(
        "--sht_every_n_steps",
        type=int,
        default=8,
        help="Apply SHT supervision every N optimizer steps.",
    )
    p.add_argument(
        "--spectral_mask_profile",
        choices=SPECTRAL_MASK_PROFILES,
        default="auto",
        help="Channels included in FFT-magnitude supervision.",
    )
    p.add_argument(
        "--loss_profile",
        choices=LOSS_PROFILES,
        default="uniform",
        help="Opt-in field and sparse-tau weighting for compact fine-tuning.",
    )
    p.add_argument(
        "--group_balance_ema_decay",
        type=float,
        default=0.99,
        help="Train-only EMA decay for relative field-by-time balancing.",
    )
    p.add_argument(
        "--group_balance_cvar_fields",
        type=int,
        default=6,
        help="Number of relative-error fields in the adaptive CVaR term.",
    )
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Optimizer warmup steps; short fine-tunes should set this explicitly.",
    )
    p.add_argument(
        "--trainable_scope",
        choices=TRAINABLE_SCOPES,
        default="all",
        help="Restrict optimization to an explicitly supported model submodule.",
    )
    p.add_argument(
        "--anchor_swap_probability",
        type=float,
        default=0.0,
        help=(
            "Training-only probability of swapping anchors and replacing "
            "tau by 1-tau; preserves the target and sample selection."
        ),
    )
    p.add_argument(
        "--distill_low_teacher_checkpoint",
        default=None,
        help="Frozen checkpoint supplying the local low-frequency target.",
    )
    p.add_argument(
        "--distill_high_teacher_checkpoint",
        default=None,
        help="Frozen checkpoint supplying the local high-frequency target.",
    )
    p.add_argument(
        "--distill_direct_teacher_checkpoint",
        default=None,
        help="Frozen teacher supplying routed full-response targets.",
    )
    p.add_argument(
        "--distill_direct_route_path",
        default=None,
        help="Validation-only tau-by-channel route for direct distillation.",
    )
    p.add_argument("--distill_direct_blend", type=float, default=0.0)
    p.add_argument("--distill_low_weight", type=float, default=0.0)
    p.add_argument("--distill_high_weight", type=float, default=0.0)
    p.add_argument("--distill_anchor_weight", type=float, default=0.0)
    p.add_argument(
        "--distill_high_mask_profile",
        choices=DISTILLATION_MASK_PROFILES,
        default="advected",
    )
    p.add_argument(
        "--distill_high_gate",
        choices=DISTILLATION_HIGH_GATES,
        default="none",
        help=(
            "Transfer only high-pass elements where the frozen teacher is "
            "closer to the training truth when set to teacher_better."
        ),
    )
    p.add_argument(
        "--distill_every_n_steps",
        type=int,
        default=1,
        help="Evaluate frozen teachers every N optimizer steps.",
    )
    p.add_argument(
        "--distill_schedule",
        choices=DISTILLATION_SCHEDULES,
        default="constant",
    )
    p.add_argument(
        "--distill_decay_start_fraction",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--distill_decay_end_fraction",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--distill_latent_teacher_checkpoint",
        default=None,
        help="Same-architecture WeatherDCAE teacher for block features.",
    )
    p.add_argument("--distill_latent_weight", type=float, default=0.0)
    p.add_argument("--distill_latent_every_n_steps", type=int, default=2)
    p.add_argument(
        "--distill_latent_schedule",
        choices=DISTILLATION_SCHEDULES,
        default="constant",
    )
    p.add_argument(
        "--distill_latent_decay_start_fraction",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--distill_latent_decay_end_fraction",
        type=float,
        default=1.0,
    )
    p.add_argument("--distill_latent_block_decay", type=float, default=0.5)
    p.add_argument("--distill_latent_pool_height", type=int, default=45)
    p.add_argument("--distill_latent_pool_width", type=int, default=90)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--seed", type=int, default=202707)
    p.add_argument("--ckpt_path", default=None)
    p.add_argument(
        "--init_weights_path",
        default=None,
        help=(
            "Initialize model weights from a checkpoint without restoring "
            "optimizer, scheduler, epoch, or global-step state."
        ),
    )
    p.add_argument("--ckpt_every_n_epochs", type=int, default=2)
    p.add_argument("--log_root", default=None)
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument(
        "--train_batches_per_epoch",
        type=int,
        default=0,
        help=(
            "Optional exact microbatch count per epoch. This is used by "
            "OOM fallbacks to drop an incomplete gradient-accumulation group."
        ),
    )
    p.add_argument("--init_only", action="store_true",
                   help="build datasets/model, validate channel protocol, then exit")
    p.add_argument("--accumulate", type=int, default=1,
                   help="grad accumulation; global effective_bs = bs * accumulate * n_gpus")
    args = p.parse_args()
    if args.ckpt_path and args.init_weights_path:
        p.error("--ckpt_path and --init_weights_path are mutually exclusive")
    if args.data_source == "hres_era5":
        if not args.forecast_train_dir:
            p.error("--forecast_train_dir is required for HRES fine-tuning")
        if args.window_hours != 6:
            p.error("HRES weight fine-tuning is restricted to the 6 h task")
        if args.forecast_lead_stride_hours <= 0:
            p.error("--forecast_lead_stride_hours must be positive")
        if args.forecast_max_left_lead_hours <= 0:
            p.error("--forecast_max_left_lead_hours must be positive")
        if args.forecast_error_augmentation_repeats < 1:
            p.error("--forecast_error_augmentation_repeats must be positive")
        if not (
            math.isfinite(args.forecast_error_scale_min)
            and math.isfinite(args.forecast_error_scale_max)
            and 0.0 <= args.forecast_error_scale_min
            <= 1.0
            <= args.forecast_error_scale_max
        ):
            p.error("forecast-error scales must be finite and bracket 1.0")
    elif args.forecast_error_augmentation_repeats != 1:
        p.error("forecast-error augmentation requires --data_source hres_era5")
    if args.arch in HRES_LEAD_ARCHES and args.data_source != "hres_era5":
        p.error(f"{args.arch} requires --data_source hres_era5")
    if args.warmup_steps < 0:
        p.error("--warmup_steps must be non-negative")
    if (
        not math.isfinite(args.anchor_swap_probability)
        or not 0.0 <= args.anchor_swap_probability <= 1.0
    ):
        p.error("--anchor_swap_probability must lie in [0, 1]")
    if not math.isfinite(args.lambda_sht_override) or args.lambda_sht_override < 0:
        p.error("--lambda_sht_override must be finite and non-negative")
    if not 1 <= args.sht_ell_min <= args.sht_lmax:
        p.error("SHT bounds must satisfy 1 <= --sht_ell_min <= --sht_lmax")
    if not 0.0 <= args.sht_start_fraction < 1.0:
        p.error("--sht_start_fraction must lie in [0, 1)")
    if args.sht_every_n_steps < 1:
        p.error("--sht_every_n_steps must be positive")
    if args.distill_every_n_steps < 1:
        p.error("--distill_every_n_steps must be positive")
    for label in ("low", "high", "anchor"):
        weight = float(getattr(args, f"distill_{label}_weight"))
        if not math.isfinite(weight) or weight < 0.0:
            p.error(
                f"--distill_{label}_weight must be finite and non-negative"
            )
    low_required = (
        args.distill_low_weight > 0.0
        or args.distill_anchor_weight > 0.0
    )
    if low_required != bool(args.distill_low_teacher_checkpoint):
        p.error(
            "low/anchor distillation weights and low teacher checkpoint "
            "must be enabled together"
        )
    if (args.distill_high_weight > 0.0) != bool(
        args.distill_high_teacher_checkpoint
    ):
        p.error(
            "--distill_high_weight and --distill_high_teacher_checkpoint "
            "must be enabled together"
        )
    for label in ("low", "high", "latent"):
        checkpoint_path = getattr(args, f"distill_{label}_teacher_checkpoint")
        if checkpoint_path and not Path(checkpoint_path).is_file():
            p.error(f"missing {label} teacher checkpoint: {checkpoint_path}")
    if not (
        math.isfinite(args.distill_decay_start_fraction)
        and math.isfinite(args.distill_decay_end_fraction)
        and 0.0 <= args.distill_decay_start_fraction
        < args.distill_decay_end_fraction
        <= 1.0
    ):
        p.error(
            "distillation decay fractions must satisfy 0 <= start < end <= 1"
        )
    if (
        not math.isfinite(args.distill_latent_weight)
        or args.distill_latent_weight < 0.0
    ):
        p.error("--distill_latent_weight must be finite and non-negative")
    if (args.distill_latent_weight > 0.0) != bool(
        args.distill_latent_teacher_checkpoint
    ):
        p.error(
            "latent distillation weight and teacher checkpoint must be "
            "enabled together"
        )
    if args.distill_latent_every_n_steps < 1:
        p.error("--distill_latent_every_n_steps must be positive")
    if not (
        math.isfinite(args.distill_latent_decay_start_fraction)
        and math.isfinite(args.distill_latent_decay_end_fraction)
        and 0.0 <= args.distill_latent_decay_start_fraction
        < args.distill_latent_decay_end_fraction
        <= 1.0
    ):
        p.error(
            "latent distillation decay fractions must satisfy "
            "0 <= start < end <= 1"
        )
    try:
        latent_block_weights(args.distill_latent_block_decay)
    except ValueError as error:
        p.error(str(error))
    if (
        args.distill_latent_pool_height < 1
        or args.distill_latent_pool_width < 1
    ):
        p.error("latent pool dimensions must be positive")
    requested_arch = args.arch
    apply_public_arch_defaults(args)
    args.arch = resolve_arch(args.arch)
    validate_training_protocol_args(args)
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    print(
        f"=== MATCHED-PROTOCOL {args.window_hours}h TRAIN: "
        f"model={canonical_arch_name(requested_arch)} "
        f"internal_arch={args.arch} ===",
        flush=True,
    )
    print(f"  years={args.years} val={args.val_years} epochs={args.max_epochs} "
          f"bs={args.bs} lr={args.lr} seed={args.seed}", flush=True)
    print(f"  train_tau={args.train_tau_subset} eval_tau={args.eval_tau} "
          f"gpus={args.gpus}", flush=True)
    print(f"  data_source={args.data_source}", flush=True)

    common_provenance = {
        "static_features": file_provenance(args.static_path),
        "pressure_level_stats": file_provenance(args.stats_path),
        "surface_stats": file_provenance(args.surface_stats_path),
    }
    if args.data_source == "era5":
        train_base = ERA5MemmapDataset(
            memmap_dir=args.memmap_dir, years=args.years,
            max_tau_hours=args.window_hours,
            samples_per_date=args.samples_per_date_train, train=True,
            train_hours=args.train_tau_subset,
            release_memmap_pages=args.release_memmap_pages,
            static_path=args.static_path, stats_path=args.stats_path,
            surface_stats_path=args.surface_stats_path)
        val_base = ERA5MemmapDataset(
            memmap_dir=args.memmap_dir, years=args.val_years,
            max_tau_hours=args.window_hours,
            samples_per_date=args.samples_per_date_val, train=False,
            eval_hours=args.eval_tau,
            release_memmap_pages=args.release_memmap_pages,
            static_path=args.static_path, stats_path=args.stats_path,
            surface_stats_path=args.surface_stats_path)
        input_provenance = {
            **common_provenance,
            "memmap": memmap_dataset_provenance(
                args.memmap_dir,
                list(dict.fromkeys([*args.years, *args.val_years])),
            ),
        }
    else:
        validation_forecast_dir = (
            args.forecast_val_dir or args.forecast_train_dir
        )
        train_base = HRESForecastAnchorDataset(
            args.forecast_train_dir,
            args.memmap_dir,
            args.years,
            max_tau_hours=args.window_hours,
            train=True,
            train_hours=args.train_tau_subset,
            lead_stride_hours=args.forecast_lead_stride_hours,
            maximum_left_lead_hours=args.forecast_max_left_lead_hours,
            forecast_error_augmentation_repeats=(
                args.forecast_error_augmentation_repeats
            ),
            forecast_error_scale_min=args.forecast_error_scale_min,
            forecast_error_scale_max=args.forecast_error_scale_max,
            forecast_error_augmentation_seed=(
                args.forecast_error_augmentation_seed
            ),
            expose_anchor_error=(
                resolve_arch(args.arch) in HRES_DEGRADE_ARCHES
            ),
            stats_path=args.stats_path,
            surface_stats_path=args.surface_stats_path,
        )
        val_base = HRESForecastAnchorDataset(
            validation_forecast_dir,
            args.memmap_dir,
            args.val_years,
            max_tau_hours=args.window_hours,
            train=False,
            eval_hours=args.eval_tau,
            lead_stride_hours=args.forecast_lead_stride_hours,
            maximum_left_lead_hours=args.forecast_max_left_lead_hours,
            stats_path=args.stats_path,
            surface_stats_path=args.surface_stats_path,
        )
        input_provenance = {
            **common_provenance,
            "hres_train": hres_finetune_dataset_provenance(train_base),
            "hres_validation": hres_finetune_dataset_provenance(val_base),
        }
    train_ds = TauRescaleAnd24chWrapper(train_base, delta_t=float(args.window_hours))
    val_ds = TauRescaleAnd24chWrapper(val_base, delta_t=float(args.window_hours))
    assert_24ch_protocol(train_base, train_ds, "train")
    assert_24ch_protocol(val_base, val_ds, "val")

    print(f"  train={len(train_ds)} val={len(val_ds)} "
          f"global_effective_bs={args.bs * args.accumulate * max(1, len(args.gpus))}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_bs, shuffle=False,
                            num_workers=args.val_workers, pin_memory=True,
                            persistent_workers=args.val_workers > 0)
    (
        trainer_train_batch_limit,
        steps_per_epoch,
        dropped_train_microbatches,
    ) = resolve_train_batch_schedule(
        dataset_size=len(train_ds),
        batch_size_per_device=args.bs,
        accumulate=args.accumulate,
        devices=max(1, len(args.gpus)),
        requested_microbatches=args.train_batches_per_epoch,
        limit_fraction=args.limit_train_batches,
    )
    total_steps = steps_per_epoch * args.max_epochs
    print(
        "  train_schedule: "
        f"microbatches_per_device={trainer_train_batch_limit} "
        f"optimizer_steps_per_epoch={steps_per_epoch} "
        f"dropped_incomplete_microbatches={dropped_train_microbatches}",
        flush=True,
    )

    model = CapMatchedLit(
        arch=args.arch, static_path=args.static_path, lr=args.lr,
        delta_t=float(args.window_hours), eval_taus=tuple(args.eval_tau),
        train_taus=tuple(args.train_tau_subset),
        warmup_steps=args.warmup_steps, total_steps=total_steps,
        training_seed=args.seed,
        lambda_hf_override=args.lambda_hf_override,
        lambda_spec_override=args.lambda_spec_override,
        lambda_band_override=args.lambda_band_override,
        lambda_sht_override=args.lambda_sht_override,
        sht_ell_min=args.sht_ell_min,
        sht_lmax=args.sht_lmax,
        sht_start_fraction=args.sht_start_fraction,
        sht_every_n_steps=args.sht_every_n_steps,
        spectral_mask_profile=args.spectral_mask_profile,
        loss_profile=args.loss_profile,
        group_balance_ema_decay=args.group_balance_ema_decay,
        group_balance_cvar_fields=args.group_balance_cvar_fields,
        trainable_scope=args.trainable_scope,
        anchor_swap_probability=args.anchor_swap_probability,
        distill_low_teacher_checkpoint=(
            args.distill_low_teacher_checkpoint
        ),
        distill_high_teacher_checkpoint=(
            args.distill_high_teacher_checkpoint
        ),
        distill_direct_teacher_checkpoint=(
            args.distill_direct_teacher_checkpoint
        ),
        distill_direct_route_path=args.distill_direct_route_path,
        distill_direct_blend=args.distill_direct_blend,
        distill_low_weight=args.distill_low_weight,
        distill_high_weight=args.distill_high_weight,
        distill_anchor_weight=args.distill_anchor_weight,
        distill_high_mask_profile=args.distill_high_mask_profile,
        distill_high_gate=args.distill_high_gate,
        distill_every_n_steps=args.distill_every_n_steps,
        distill_schedule=args.distill_schedule,
        distill_decay_start_fraction=(
            args.distill_decay_start_fraction
        ),
        distill_decay_end_fraction=args.distill_decay_end_fraction,
        distill_latent_teacher_checkpoint=(
            args.distill_latent_teacher_checkpoint
        ),
        distill_latent_weight=args.distill_latent_weight,
        distill_latent_every_n_steps=args.distill_latent_every_n_steps,
        distill_latent_schedule=args.distill_latent_schedule,
        distill_latent_decay_start_fraction=(
            args.distill_latent_decay_start_fraction
        ),
        distill_latent_decay_end_fraction=(
            args.distill_latent_decay_end_fraction
        ),
        distill_latent_block_decay=args.distill_latent_block_decay,
        distill_latent_pool_height=args.distill_latent_pool_height,
        distill_latent_pool_width=args.distill_latent_pool_width,
    )
    if args.arch in FLOW_INTRINSIC_ARCHES:
        normalization_mean = torch.cat(
            (
                train_base.mu.reshape(-1),
                train_base.surface_mu[:4].reshape(-1),
            )
        )
        normalization_std = torch.cat(
            (
                train_base.sigma.reshape(-1),
                train_base.surface_sigma[:4].reshape(-1),
            )
        )
        validation_mean = torch.cat(
            (
                val_base.mu.reshape(-1),
                val_base.surface_mu[:4].reshape(-1),
            )
        )
        validation_std = torch.cat(
            (
                val_base.sigma.reshape(-1),
                val_base.surface_sigma[:4].reshape(-1),
            )
        )
        torch.testing.assert_close(
            normalization_mean,
            validation_mean,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            normalization_std,
            validation_std,
            rtol=0,
            atol=0,
        )
        model.net.set_normalization(
            normalization_mean,
            normalization_std,
        )
        model.hparams["intrinsic_transport_normalization"] = {
            "channel_order": [
                *train_base.channel_names,
                *train_base.surface_variables[:4],
            ],
            "pressure_level_stats_sha256": input_provenance[
                "pressure_level_stats"
            ]["sha256"],
            "surface_stats_sha256": input_provenance[
                "surface_stats"
            ]["sha256"],
        }
    if args.init_weights_path:
        initialization = checkpoint_resume_lineage(args.init_weights_path)
        initialized_arch = initialization["model"].get("arch")
        lagrange_expansion = (
            initialized_arch == "flow_compact_hermite_l"
            and args.arch == "flow_compact_lagrange_l"
        )
        dcae_fm_expansion = (
            initialized_arch == "dcae_14m"
            and args.arch in {"dcae_fm_14m", "dcae_fm_control_14m"}
        )
        hres_lead_expansion = (
            initialized_arch == "flow_pp3"
            and args.arch == "flow_pp3_hres_aug"
        )
        hres_residual_expansion = (
            initialized_arch == "flow_pp3_hres_aug"
            and args.arch == "flow_pp3_hres_residual"
        )
        # The degradation-aware arch is the lead-conditioned arch plus a
        # reliability branch whose consumers are zero at initialisation, so a
        # warm start reproduces the parent exactly (verified bit-identical).
        hres_degrade_expansion = (
            initialized_arch == "flow_pp3_hres_aug"
            and args.arch in HRES_DEGRADE_ARCHES
        )
        compatible_expansion = (
            lagrange_expansion
            or dcae_fm_expansion
            or hres_lead_expansion
            or hres_residual_expansion
            or hres_degrade_expansion
        )
        if initialized_arch != args.arch and not compatible_expansion:
            raise ValueError(
                "initialization architecture mismatch: "
                f"{initialized_arch!r} != {args.arch!r}"
            )
        checkpoint = torch.load(
            args.init_weights_path,
            map_location="cpu",
            weights_only=False,
        )
        incompatible = model.load_state_dict(
            checkpoint["state_dict"],
            strict=not compatible_expansion,
        )
        if compatible_expansion:
            if lagrange_expansion:
                expected_missing = {
                    "net.base_knot_head.weight",
                    "net.base_knot_head.bias",
                }
            elif dcae_fm_expansion:
                expected_missing = {
                    "net.state_adapter.weight",
                    "net.state_adapter.bias",
                    "net.flow_time_mlp.mlp.0.weight",
                    "net.flow_time_mlp.mlp.0.bias",
                    "net.flow_time_mlp.mlp.2.weight",
                    "net.flow_time_mlp.mlp.2.bias",
                }
            elif hres_lead_expansion:
                expected_missing = {
                    "net.forecast_lead_mlp.net.0.weight",
                    "net.forecast_lead_mlp.net.0.bias",
                    "net.forecast_lead_mlp.net.2.weight",
                    "net.forecast_lead_mlp.net.2.bias",
                }
            elif hres_degrade_expansion:
                # Reliability branch only; every consumer is neutral at
                # initialisation so the warm start is bit-identical.
                expected_missing = {
                    key
                    for key in incompatible.missing_keys
                    if key.startswith("net.reliability_")
                    or key
                    in {
                        "net.anchor_trust_gain",
                        "net.anchor_residual_gain",
                    }
                }
            else:
                expected_missing = {
                    "net.hres_residual_head.weight",
                    "net.hres_residual_head.bias",
                }
            if (
                set(incompatible.missing_keys) != expected_missing
                or incompatible.unexpected_keys
            ):
                raise ValueError(
                    "unexpected warm-start incompatibility: "
                    f"missing={incompatible.missing_keys} "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            initialization["compatibility"] = {
                "source_arch": initialized_arch,
                "target_arch": args.arch,
                "zero_initialized_missing_keys": sorted(expected_missing),
            }
        model.hparams["initialization_lineage"] = initialization
        print(
            "  initialized_weights="
            f"{initialization['checkpoint_path']} "
            f"epoch={initialization['epoch']} "
            f"step={initialization['global_step']}",
            flush=True,
        )
    print(
        f"  loss_weights: highpass={model.lambda_hf:g} "
        f"spectral={model.lambda_spec:g} "
        f"multiband={model.lambda_band:g} "
        f"sht={model.lambda_sht:g} "
        f"spectral_mask={model.spectral_mask_profile} "
        f"profile={model.loss_profile} "
        f"trainable_scope={model.trainable_scope} "
        f"distill_low={model.distill_low_weight:g} "
        f"distill_high={model.distill_high_weight:g} "
        f"distill_direct={model.distill_direct_blend:g} "
        f"distill_anchor={model.distill_anchor_weight:g} "
        f"distill_mask={model.distill_high_mask_profile} "
        f"distill_interval={model.distill_every_n_steps} "
        f"distill_schedule={model.distill_schedule} "
        f"distill_latent={model.distill_latent_weight:g} "
        f"latent_interval={model.distill_latent_every_n_steps} "
        f"latent_schedule={model.distill_latent_schedule}",
        flush=True,
    )
    model.hparams["training_protocol"] = {
        "requested_arch": requested_arch,
        "canonical_arch": canonical_arch_name(requested_arch),
        "data_source": args.data_source,
        "anchor_source": (
            "IFS_HRES_same_forecast_trajectory"
            if args.data_source == "hres_era5"
            else "ERA5"
        ),
        "target_source": "ERA5_at_intermediate_valid_time",
        "future_analysis_as_input": False,
        "train_years": list(args.years),
        "val_years": list(args.val_years),
        "train_tau_hours": list(args.train_tau_subset),
        "eval_tau_hours": list(args.eval_tau),
        "window_hours": args.window_hours,
        "batch_size_per_device": args.bs,
        "accumulate_grad_batches": args.accumulate,
        "devices": list(args.gpus),
        "global_effective_batch_size": (
            args.bs * args.accumulate * max(1, len(args.gpus))
        ),
        "precision": args.precision,
        "samples_per_date_train": args.samples_per_date_train,
        "samples_per_date_val": args.samples_per_date_val,
        "forecast_lead_stride_hours": (
            args.forecast_lead_stride_hours
            if args.data_source == "hres_era5"
            else None
        ),
        "forecast_max_left_lead_hours": (
            args.forecast_max_left_lead_hours
            if args.data_source == "hres_era5"
            else None
        ),
        "forecast_error_augmentation": (
            {
                "repeats": args.forecast_error_augmentation_repeats,
                "scale_min": args.forecast_error_scale_min,
                "scale_max": args.forecast_error_scale_max,
                "seed": args.forecast_error_augmentation_seed,
            }
            if args.data_source == "hres_era5"
            else None
        ),
        "release_memmap_pages": args.release_memmap_pages,
        "seed": args.seed,
        "loss_profile": model.loss_profile,
        "group_balance_ema_decay": model.group_balance_ema_decay,
        "group_balance_cvar_fields": model.group_balance_cvar_fields,
        "trainable_scope": model.trainable_scope,
        "optimizer_weight_decay": trainable_scope_weight_decay(
            model.trainable_scope
        ),
        "anchor_swap_probability": model.anchor_swap_probability,
        "warmup_steps": args.warmup_steps,
        "lambda_hf": model.lambda_hf,
        "lambda_hf_override": args.lambda_hf_override,
        "lambda_spec": model.lambda_spec,
        "lambda_spec_override": args.lambda_spec_override,
        "lambda_band": model.lambda_band,
        "lambda_band_override": args.lambda_band_override,
        "lambda_sht": model.lambda_sht,
        "lambda_sht_override": args.lambda_sht_override,
        "sht_grid": "equiangular_cell_centered_fejer1",
        "sht_ell_min": model.sht_ell_min,
        "sht_lmax": model.sht_lmax,
        "sht_start_fraction": model.sht_start_fraction,
        "sht_every_n_steps": model.sht_every_n_steps,
        "spectral_mask_profile": model.spectral_mask_profile,
        "highpass_boundary": model.highpass_boundary,
        "distillation": {
            "enabled": bool(
                model.distill_low_weight > 0.0
                or model.distill_high_weight > 0.0
                or model.distill_anchor_weight > 0.0
                or model.distill_direct_blend > 0.0
                or model.distill_latent_weight > 0.0
            ),
            "truth_primary": True,
            "decomposition": "spherical_local_3x3_phase_preserving",
            "low_teacher_target": "complementary_local_lowpass",
            "high_teacher_target": (
                "teacher_guided_truth_highpass"
                if model.distill_high_gate == "teacher_better_truth"
                else "local_highpass"
            ),
            "high_mask_profile": model.distill_high_mask_profile,
            "high_gate": model.distill_high_gate,
            "high_mask_active_indices": (
                model.distill_high_channel_mask.nonzero()
                .reshape(-1)
                .cpu()
                .tolist()
            ),
            "low_weight": model.distill_low_weight,
            "high_weight": model.distill_high_weight,
            "direct": {
                "blend": model.distill_direct_blend,
                "target": "routed_truth_teacher_convex_response",
                "route": model.distill_direct_route_provenance,
                "teacher_lineage": (
                    model._distill_direct_teacher.lineage
                    if model._distill_direct_teacher is not None
                    else None
                ),
            },
            "anchor_weight": model.distill_anchor_weight,
            "anchor_target": "full_output_on_non_high_teacher_fields",
            "every_n_steps": model.distill_every_n_steps,
            "interval_corrected": True,
            "schedule": model.distill_schedule,
            "decay_start_fraction": (
                model.distill_decay_start_fraction
            ),
            "decay_end_fraction": model.distill_decay_end_fraction,
            "teacher_parameters_saved": False,
            "low_teacher_lineage": (
                model._distill_low_teacher.lineage
                if model._distill_low_teacher is not None
                else None
            ),
            "high_teacher_lineage": (
                model._distill_high_teacher.lineage
                if model._distill_high_teacher is not None
                else None
            ),
            "latent": {
                "weight": model.distill_latent_weight,
                "teacher_target": "pooled_same_architecture_features",
                "tap_weights": latent_block_weights(
                    model.distill_latent_block_decay
                ),
                "block_decay": model.distill_latent_block_decay,
                "pool_shape": [
                    model.distill_latent_pool_height,
                    model.distill_latent_pool_width,
                ],
                "every_n_steps": model.distill_latent_every_n_steps,
                "interval_corrected": True,
                "schedule": model.distill_latent_schedule,
                "decay_start_fraction": (
                    model.distill_latent_decay_start_fraction
                ),
                "decay_end_fraction": (
                    model.distill_latent_decay_end_fraction
                ),
                "teacher_lineage": (
                    model._distill_latent_teacher.lineage
                    if model._distill_latent_teacher is not None
                    else None
                ),
            },
        },
        "train_batches_per_epoch": trainer_train_batch_limit,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "dropped_train_microbatches_per_epoch": (
            dropped_train_microbatches
        ),
    }
    model.hparams["training_input_provenance"] = input_provenance
    repo_root = Path(__file__).resolve().parents[2]
    training_sources = {
        "train_capacity_matched_6h.py": Path(__file__).resolve(),
        "training_protocol.py": (
            repo_root / "tools" / "train" / "training_protocol.py"
        ),
        "memmap_dataset.py": (
            repo_root / "weather_time_interp" / "memmap_dataset.py"
        ),
        "normalization.py": (
            repo_root / "weather_time_interp" / "normalization.py"
        ),
        "spherical_spectra.py": (
            repo_root
            / "weather_time_interp"
            / "metrics"
            / "spherical_spectra.py"
        ),
    }
    if args.data_source == "hres_era5":
        training_sources["hres_finetune_dataset.py"] = (
            repo_root
            / "weather_time_interp"
            / "hres_finetune_dataset.py"
        )
        training_sources["batch_eval_forecast_anchor.py"] = (
            repo_root
            / "tools"
            / "eval"
            / "batch_eval_forecast_anchor.py"
        )
    if args.arch in {
        "dcae_14m",
        "dcae_fm_14m",
        "dcae_fm_control_14m",
    }:
        training_sources["dcae.py"] = (
            repo_root / "weather_time_interp" / "model" / "dcae.py"
        )
        training_sources["dcae_adaln_model.py"] = (
            repo_root
            / "weather_time_interp"
            / "model"
            / "dcae_adaln_model.py"
        )
    if args.arch in (
        "flow",
        "flow_noskip",
        "flow_ungated",
        "flow_accel",
        "flow_pp",
        "flow_pp2",
        "flow_pp3",
        "flow_pp3_hres_aug",
        "flow_pp3_hres_residual",
        "flow_pp3_nodiff",
        *FLOW_SPHERICAL_ENCODER_ARCHES,
        *FLOW_SPECTRAL_DETAIL_ARCHES,
        "flow_dual",
        "flow_spherical_ep",
        *FLOW_MSF_ARCHES,
        *FLOW_PP3_COMPACT_ARCHES,
        *FLOW_COMPACT_ARCHES,
    ):
        training_sources["weatherbridge_flow_model.py"] = (
            repo_root
            / "weather_time_interp"
            / "model"
            / "weatherbridge_flow_model.py"
        )
    if args.arch == "atmvfi":
        training_sources["train_atm_vfi_12h_oddskip.py"] = _atmvfi_source(
            Path(__file__).resolve().parents[2]
        )
    model.hparams["training_code_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in training_sources.items()
    }
    if args.ckpt_path:
        model.hparams["resume_lineage"] = checkpoint_resume_lineage(
            args.ckpt_path
        )
    n_p = sum(q.numel() for q in model.net.parameters()) / 1e6
    n_trainable = sum(
        q.numel() for q in model.net.parameters() if q.requires_grad
    ) / 1e6
    print(
        f"  params: {n_p:.2f}M trainable={n_trainable:.3f}M "
        f"total_steps={total_steps}",
        flush=True,
    )
    if args.init_only:
        print("=== INIT OK ===", flush=True)
        return

    default_out = Path(args.log_root) / args.exp_name if args.log_root else Path(f"logs/{args.exp_name}")
    out_dir = Path(os.environ.get("OUT_DIR", str(default_out)))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=out_dir, save_last=True, save_top_k=-1,
                              every_n_epochs=args.ckpt_every_n_epochs,
                              filename="{epoch}-{step}",
                              enable_version_counter=False)
    logger_root = out_dir / "lightning_logs"
    logger_version = 0
    if args.ckpt_path:
        existing_versions = [
            int(path.name.removeprefix("version_"))
            for path in logger_root.glob("version_*")
            if path.name.removeprefix("version_").isdigit()
        ]
        if existing_versions:
            logger_version = max(existing_versions) + 1
    logger = CSVLogger(
        save_dir=str(logger_root),
        name="",
        version=logger_version,
    )
    print(f"  csv_logger_version={logger_version}", flush=True)

    if len(args.gpus) > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(process_group_backend="nccl", find_unused_parameters=False)
    else:
        strategy = "auto"
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator="gpu", devices=args.gpus,
        strategy=strategy, precision=args.precision,
        accumulate_grad_batches=args.accumulate,
        callbacks=[ckpt_cb], logger=logger,
        log_every_n_steps=20, gradient_clip_val=1.0,
        num_sanity_val_steps=0, enable_progress_bar=False,
        limit_train_batches=trainer_train_batch_limit,
        limit_val_batches=args.limit_val_batches)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
