"""Matched-protocol interpolation trainer for WeatherBridge ablations.

All architectures use the same recipe (LR, batch size, epochs, data,
tau sampling, latitude-weighted reconstruction loss, and validation
protocol). The established baselines are approximately 14M parameters;
UPR-Lite is an explicit sub-2M efficiency family whose parameter count and
latency are reported as separate Pareto objectives.

Public CLI names:

  --arch weatherbridge  -> WeatherBridgeModel (headline model)
  --arch weatherdcae    -> WeatherDCAEAdaLNModel (24-channel 14M baseline)

Historical experiment flags such as ``flow_pp3`` remain accepted so old
launch scripts and checkpoints stay reproducible.

Run (LR=1e-4, bs=16, 3yr, 8 epochs):
  PYTHONPATH=... python train_capacity_matched_6h.py \
    --arch weatherbridge --years 2017 2018 2019 \
    --val_years 2020 --bs 16 --lr 1e-4 --max_epochs 8 --gpus 0 \
    --exp_name exp_weatherbridge_14m_6h
"""
from __future__ import annotations

import argparse
import hashlib
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
from weather_time_interp.normalization import file_provenance


CANONICAL_ARCH_NAMES = {
    "dcae_14m": "WeatherDCAE",
    "wb_vanilla": "WeatherDCAE",
    "wb_skip": "WeatherDCAE-Skip",
    "atmvfi": "PixelAttn-VFI",
    "amt": "WeatherAMT-L",
    "amt_residual": "WeatherAMT-Residual-L",
    "flow_pp3": "WeatherBridge",
    "flow_pp3_compact_l": "WeatherBridge-PP3-Compact-L",
    "flow_spherical_ep": "WeatherBridge-Spherical-EP",
    "flow_compact_vp3": "WeatherBridge-Compact-VP3",
    "flow_compact_vp3_m": "WeatherBridge-Compact-VP3-M",
    "flow_compact_vp3_l": "WeatherBridge-Compact-VP3-L",
    "flow_compact_hermite_l": "WeatherBridge-Compact-Hermite-L",
    "flow_compact_lagrange_l": "WeatherBridge-Compact-Lagrange-L",
    "upr_lite": "WeatherBridge-UPR-Lite",
    "upr_lite_lap": "WeatherBridge-UPR-Lite-Lap",
    "upr_lite_column": "WeatherBridge-UPR-Lite-Column",
    "upr_lite_continuous": "WeatherBridge-UPR-Lite-Continuous",
    "upr_lite_continuous_m": "WeatherBridge-UPR-Lite-Continuous-M",
    "upr_lite_implicit_global": "WeatherBridge-UPR-Lite-Implicit-Global",
    "upr_lite_implicit_global_q4": "WeatherBridge-UPR-Lite-Implicit-Global-Q4",
    "upr_implicit_global_14m": "WeatherBridge-UPR-Implicit-Global-14M",
    "upr_endpoint_implicit_global_14m": (
        "WeatherBridge-UPR-Endpoint-Implicit-Global-14M"
    ),
    "upr_local_corr_14m": "WeatherBridge-UPR-LocalCorr-14M",
    "upr_query_match_14m": "WeatherBridge-UPR-QueryMatch-14M",
    "upr_spherical_implicit_global_14m": (
        "WeatherBridge-UPR-Spherical-Implicit-Global-14M"
    ),
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

LOSS_PROFILES = (
    "uniform",
    "base_balanced",
    "base_edge_balanced",
)
TRAINABLE_SCOPES = (
    "all",
    "base_knot_head",
)

ARCH_ALIASES = {
    "weatherbridge": "flow_pp3",
    "weatherdcae": "dcae_14m",
    "pixelattn_vfi": "atmvfi",
    "weatheramt": "amt",
    "weatheramt_residual": "amt_residual",
}


def canonical_arch_name(arch: str) -> str:
    """Return the paper-facing name while preserving legacy CLI flags."""
    return CANONICAL_ARCH_NAMES.get(ARCH_ALIASES.get(arch, arch), arch)


def resolve_arch(arch: str) -> str:
    """Resolve a paper-facing CLI name to the checkpoint-compatible flag."""
    return ARCH_ALIASES.get(arch, arch)


def loss_profile_channel_weights(
    profile: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized reconstruction and high-pass channel weights."""
    if profile not in LOSS_PROFILES:
        raise ValueError(f"unknown loss profile: {profile}")
    reconstruction = torch.ones(24, dtype=torch.float32)
    highpass = torch.ones(24, dtype=torch.float32)
    if profile != "uniform":
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
) -> torch.Tensor:
    """Reduce a fieldwise L1 objective without changing its nominal scale."""
    per_sample_channel = (
        (prediction - target).abs() * latitude_weights
    ).mean(dim=(-2, -1))
    per_sample = (
        per_sample_channel * channel_weights.view(1, -1)
    ).sum(dim=-1) / channel_weights.sum()
    return (per_sample * sample_weights).sum() / sample_weights.sum()


def apply_trainable_scope(net: nn.Module, scope: str) -> None:
    """Apply an explicit fine-tuning scope without changing model state."""
    if scope not in TRAINABLE_SCOPES:
        raise ValueError(f"unknown trainable scope: {scope}")
    if scope == "all":
        for parameter in net.parameters():
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


def assert_24ch_protocol(base: ERA5MemmapDataset, wrapped: Dataset, split: str) -> None:
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
    if arch == "crossframe":
        from weather_time_interp.model.weatherbridge_crossframe_model import (
            WeatherDCAECrossFrameModel,
        )
        net = WeatherDCAECrossFrameModel(
            in_channels=24, out_channels=24, n_static_features=3,
            latent_channels=32, attention_head_dim=32,
            block_type=("ResBlock", "ResBlock", "EfficientViTBlock"),
            qkv_multiscales=((), (), (5,)), lat_crop=-8,
            block_out_channels=(128, 128, 256, 256), layers_per_block=(2, 2, 2),
        )
        return net, False, True
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
    if arch == "amt":
        from weather_time_interp.model.weather_amt_model import WeatherAMTModel

        net = WeatherAMTModel(
            in_channels=24,
            out_channels=24,
            n_static_features=3,
            corr_radius=3,
            corr_levels=4,
            num_flows=5,
            channels=(48, 64, 72, 110),
            skip_channels=48,
            endpoint_envelope=True,
            max_field_displacement=16.0,
        )
        return net, False, True
    if arch == "amt_residual":
        from weather_time_interp.model.weather_amt_residual_model import (
            WeatherAMTResidualModel,
        )

        net = WeatherAMTResidualModel(
            in_channels=24,
            out_channels=24,
            n_static_features=3,
            corr_radius=3,
            corr_levels=4,
            num_flows=5,
            channels=(48, 64, 72, 110),
            skip_channels=48,
            max_field_displacement=16.0,
        )
        return net, False, True
    if arch == "mamba":
        # WeatherBridge-Mamba: VFIMamba Mixed-SSM backbone (NeurIPS 2024).
        # hidden=64 → 13.68M, matched to ATM-VFI 14.67M / WB-XF 13.37M.
        from weather_time_interp.model.weatherbridge_mamba_model import (
            WeatherBridgeMambaModel,
        )
        net = WeatherBridgeMambaModel(
            in_channels=24, out_channels=24, n_static_features=3,
            hidden=64, n_levels=3, d_state=16, lat_crop=0)
        return net, False, True
    if arch in (
        "upr_lite",
        "upr_lite_lap",
        "upr_lite_column",
        "upr_lite_continuous",
        "upr_lite_continuous_m",
        "upr_lite_implicit_global",
        "upr_lite_implicit_global_q4",
    ):
        from weather_time_interp.model.weatherbridge_upr_lite_model import (
            WeatherBridgeUPRLiteModel,
            upr_lite_variant_kwargs,
        )
        net = WeatherBridgeUPRLiteModel(**upr_lite_variant_kwargs(arch))
        return net, False, True
    if arch in (
        "upr_implicit_global_14m",
        "upr_endpoint_implicit_global_14m",
        "upr_local_corr_14m",
        "upr_query_match_14m",
    ):
        from weather_time_interp.model.weatherbridge_upr_lite_model import (
            WeatherBridgeUPRLiteModel,
        )
        from weather_time_interp.model.weatherbridge_upr_scaled_model import (
            upr_scaled_variant_kwargs,
        )
        net = WeatherBridgeUPRLiteModel(
            **upr_scaled_variant_kwargs(arch)
        )
        return net, False, True
    if arch == "upr_spherical_implicit_global_14m":
        from weather_time_interp.model.weatherbridge_upr_scaled_model import (
            upr_scaled_variant_kwargs,
        )
        from weather_time_interp.model.weatherbridge_upr_spherical_model import (
            WeatherBridgeUPRSphericalModel,
        )

        net = WeatherBridgeUPRSphericalModel(
            **upr_scaled_variant_kwargs("upr_implicit_global_14m")
        )
        return net, False, True
    if arch in (
        "flow",
        "flow_noskip",
        "flow_ungated",
        "flow_accel",
        "flow_pp",
        "flow_pp2",
        "flow_pp3",
        "flow_dual",
        "flow_spherical_ep",
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
            else 56 if arch in FLOW_PP3_COMPACT_ARCHES
            else 64 if arch == "flow_dual"
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
            spectral_branch=(
                arch
                in (
                    "flow_pp3",
                    "flow_spherical_ep",
                    *FLOW_PP3_COMPACT_ARCHES,
                )
            ),
            hydro_couple=(
                arch
                in (
                    "flow_pp3",
                    "flow_spherical_ep",
                    *FLOW_PP3_COMPACT_ARCHES,
                    *FLOW_COMPACT_ARCHES,
                )
            ),
            dual_stream=(arch == "flow_dual"),
            spherical_ops=(
                arch == "flow_spherical_ep" or arch in FLOW_COMPACT_ARCHES
            ),
            endpoint_preserving=(
                arch == "flow_spherical_ep" or arch in FLOW_COMPACT_ARCHES
            ),
            query_independent_trajectory=(
                arch == "flow_spherical_ep" or arch in FLOW_COMPACT_ARCHES
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
                )
            ),
            base_field_knots=(arch == "flow_compact_lagrange_l"),
        )
        return net, False, True
    raise ValueError(f"unknown arch {arch}")


class CapMatchedLit(pl.LightningModule):
    def __init__(self, arch: str, static_path: str, lr: float,
                 delta_t: float, eval_taus, warmup_steps: int = 500,
                 total_steps: int = 100000, training_seed: int = 202707,
                 lambda_hf_override: float | None = None,
                 loss_profile: str = "uniform",
                 trainable_scope: str = "all",
                 distill_teacher_checkpoint: str = "",
                 distill_weight: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        if not math.isfinite(distill_weight) or distill_weight < 0.0:
            raise ValueError("distill_weight must be finite and non-negative")
        if bool(distill_teacher_checkpoint) != (distill_weight > 0.0):
            raise ValueError(
                "distillation requires both a teacher checkpoint and "
                "a positive weight"
            )
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
            "upr_spherical_implicit_global_14m",
            "flow_spherical_ep",
            *FLOW_COMPACT_ARCHES,
            "amt_residual",
        ) else 0.0
        self.lambda_hf = (
            default_lambda_hf
            if lambda_hf_override is None
            else float(lambda_hf_override)
        )
        if not math.isfinite(self.lambda_hf) or self.lambda_hf < 0.0:
            raise ValueError(
                "lambda_hf_override must be finite and non-negative"
            )
        # light spectral (FFT-magnitude) loss to sharpen HF content; only for
        # the flow_pp* upgrade arms so the rest of the study is unchanged.
        self.lambda_spec = 0.05 if arch in ("flow_pp", "flow_pp2") else 0.0
        # flow_pp2: mask the spectral loss to ADVECTED channels only. On smooth
        # mass/static fields (Z*, t2m, mslp) an HF-magnitude penalty injects
        # spurious small-scale noise and RAISES their RMSE (observed in flow_pp:
        # mslp +4.7% -> +7.0%). Winds/moisture carry real HF, so keep it there.
        if arch == "flow_pp2":
            m = torch.ones(24)
            for c in (16, 17, 18, 19, 20, 23):  # Z1000..Z700, t2m, mslp
                m[c] = 0.0
            self.register_buffer("spec_mask", m.view(1, 24, 1, 1), persistent=False)
        else:
            self.spec_mask = None
        self.net, self.is_atmvfi, self.needs_cond = build_net(arch, static_path)
        self.trainable_scope = trainable_scope
        apply_trainable_scope(self.net, trainable_scope)
        self.distill_weight = float(distill_weight)
        self.distill_teacher = None
        if self.distill_weight > 0.0:
            teacher_checkpoint = torch.load(
                distill_teacher_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            teacher_hparams = teacher_checkpoint.get(
                "hyper_parameters",
                {},
            )
            if teacher_hparams.get("arch") != "flow_pp3":
                raise ValueError("distillation teacher must be flow_pp3")
            teacher, _, _ = build_net("flow_pp3", static_path)
            teacher_state = {
                key.removeprefix("net."): value
                for key, value in teacher_checkpoint["state_dict"].items()
                if key.startswith("net.")
            }
            incompatible = teacher.load_state_dict(
                teacher_state,
                strict=True,
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError("incompatible distillation teacher state")
            teacher.requires_grad_(False)
            teacher.eval()
            self.distill_teacher = teacher
            self.hparams["distillation_teacher_lineage"] = (
                checkpoint_resume_lineage(distill_teacher_checkpoint)
            )
        spherical_highpass_arches = {
            "amt",
            "amt_residual",
            "flow_spherical_ep",
            *FLOW_COMPACT_ARCHES,
            "upr_spherical_implicit_global_14m",
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

        self.eval_taus = list(eval_taus)
        for h in self.eval_taus:
            self.register_buffer(f"val_sq_h{h}", torch.zeros(24), persistent=False)
            self.register_buffer(f"val_cnt_h{h}", torch.tensor(0.0), persistent=False)

    def _forward(self, x0, x1, tau):
        if self.is_atmvfi:
            return self.net(x0, x1, tau)
        B = x0.size(0)
        cond = torch.full((B,), self.delta_t, device=x0.device, dtype=torch.float32)
        static = self.static3.expand(B, -1, -1, -1)
        out = self.net(x0, x1, tau, cond, static=static)
        return out[0] if isinstance(out, tuple) else out

    def _teacher_forward(self, x0, x1, tau):
        if self.distill_teacher is None:
            raise RuntimeError("distillation teacher is not configured")
        self.distill_teacher.eval()
        batch_size = x0.size(0)
        cond = torch.full(
            (batch_size,),
            self.delta_t,
            device=x0.device,
            dtype=torch.float32,
        )
        static = self.static3.expand(batch_size, -1, -1, -1)
        with torch.no_grad():
            out = self.distill_teacher(
                x0,
                x1,
                tau,
                cond,
                static=static,
            )
        return out[0] if isinstance(out, tuple) else out

    def _step(self, batch, prefix):
        x0, x1, tgt = batch["x0"], batch["x1"], batch["target"]
        tau = batch["tau"]
        if tau.dim() > 1:
            tau = tau.view(-1)
        pred = self._forward(x0, x1, tau)
        recon = ((pred - tgt).abs() * self.lat_w).mean()
        self.log(f"{prefix}/recon_l1", recon, sync_dist=True, prog_bar=True)
        if prefix == "train":
            tau_hour = batch["tau_hour"].view(-1).long()
            sample_weights = loss_profile_tau_weights(
                self.loss_profile,
                tau_hour,
            )
            err = weighted_latitude_l1(
                pred,
                tgt,
                self.lat_w,
                self.loss_channel_weights,
                sample_weights,
            )
            if self.loss_profile != "uniform":
                self.log("train/objective_l1", err, sync_dist=True)
        else:
            err = recon
        if prefix == "train" and self.lambda_spec > 0:
            # FFT-magnitude L1 over the spatial plane (per channel), penalising
            # blurred high-frequency structure. Light weight so RMSE stays primary.
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
            hf = weighted_latitude_l1(
                pred_hf,
                tgt_hf,
                self.lat_w,
                self.loss_highpass_weights,
                sample_weights,
            )
            self.log("train/highpass", hf, sync_dist=True)
            err = err + self.lambda_hf * hf
        if prefix == "train" and self.distill_weight > 0.0:
            teacher_pred = self._teacher_forward(x0, x1, tau)
            distill = weighted_latitude_l1(
                pred,
                teacher_pred,
                self.lat_w,
                self.loss_channel_weights,
                sample_weights,
            )
            self.log("train/distill_l1", distill, sync_dist=True)
            err = err + self.distill_weight * distill
        if prefix == "val":
            sq = (pred.float() - tgt.float()) ** 2
            sq_lw = (sq * self.lat_w).sum(dim=(-2, -1))  # (B, C)
            tau_hour = batch["tau_hour"].view(-1).long()
            for h in self.eval_taus:
                mask = tau_hour == h
                if mask.any():
                    getattr(self, f"val_sq_h{h}").add_(sq_lw[mask].sum(dim=0).detach())
                    getattr(self, f"val_cnt_h{h}").add_(mask.float().sum())
        return err

    def training_step(self, b, i):
        return self._step(b, "train")

    def validation_step(self, b, i):
        return self._step(b, "val")

    def on_save_checkpoint(self, checkpoint):
        for key in list(checkpoint["state_dict"]):
            if key.startswith("distill_teacher."):
                del checkpoint["state_dict"][key]

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
        opt = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=1e-4)

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
    p.add_argument("--arch", required=True, choices=["weatherbridge", "weatherdcae", "pixelattn_vfi", "weatheramt", "weatheramt_residual", "dcae_14m", "crossframe", "atmvfi", "amt", "amt_residual", "wb_vanilla", "wb_skip", "mamba",
                                                      "upr_lite", "upr_lite_lap",
                                                      "upr_lite_column",
                                                      "upr_lite_continuous",
                                                      "upr_lite_continuous_m",
                                                      "upr_lite_implicit_global",
                                                      "upr_lite_implicit_global_q4",
                                                      "upr_implicit_global_14m",
                                                      "upr_endpoint_implicit_global_14m",
                                                      "upr_local_corr_14m",
                                                      "upr_query_match_14m",
                                                      "upr_spherical_implicit_global_14m",
                                                      "flow", "flow_noskip", "flow_ungated",
                                                      "flow_accel", "flow_pp",
                                                      "flow_pp2", "flow_pp3",
                                                      "flow_pp3_compact_l",
                                                      "flow_dual",
                                                      "flow_spherical_ep",
                                                      "flow_compact_vp3",
                                                      "flow_compact_vp3_m",
                                                      "flow_compact_vp3_l",
                                                      "flow_compact_hermite_l",
                                                      "flow_compact_lagrange_l"])
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
        "--loss_profile",
        choices=LOSS_PROFILES,
        default="uniform",
        help="Opt-in field and sparse-tau weighting for compact fine-tuning.",
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
        "--distill_teacher_checkpoint",
        default="",
        help="Optional frozen flow_pp3 teacher used only during training.",
    )
    p.add_argument(
        "--distill_weight",
        type=float,
        default=0.0,
        help="Latitude-weighted L1 coefficient for teacher distillation.",
    )
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
    if args.warmup_steps < 0:
        p.error("--warmup_steps must be non-negative")
    if args.distill_weight < 0.0 or not math.isfinite(args.distill_weight):
        p.error("--distill_weight must be finite and non-negative")
    if bool(args.distill_teacher_checkpoint) != (args.distill_weight > 0.0):
        p.error(
            "distillation requires --distill_teacher_checkpoint and "
            "positive --distill_weight"
        )
    requested_arch = args.arch
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

    input_provenance = {
        "memmap": memmap_dataset_provenance(
            args.memmap_dir,
            list(dict.fromkeys([*args.years, *args.val_years])),
        ),
        "static_features": file_provenance(args.static_path),
        "pressure_level_stats": file_provenance(args.stats_path),
        "surface_stats": file_provenance(args.surface_stats_path),
    }
    train_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=args.years,
        max_tau_hours=args.window_hours,
        samples_per_date=args.samples_per_date_train, train=True,
        train_hours=args.train_tau_subset,
        release_memmap_pages=args.release_memmap_pages,
        static_path=args.static_path, stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path)
    train_ds = TauRescaleAnd24chWrapper(train_base, delta_t=float(args.window_hours))
    val_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=args.val_years,
        max_tau_hours=args.window_hours,
        samples_per_date=args.samples_per_date_val, train=False,
        eval_hours=args.eval_tau,
        release_memmap_pages=args.release_memmap_pages,
        static_path=args.static_path, stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path)
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
        warmup_steps=args.warmup_steps, total_steps=total_steps,
        training_seed=args.seed,
        lambda_hf_override=args.lambda_hf_override,
        loss_profile=args.loss_profile,
        trainable_scope=args.trainable_scope,
        distill_teacher_checkpoint=args.distill_teacher_checkpoint,
        distill_weight=args.distill_weight)
    if args.init_weights_path:
        initialization = checkpoint_resume_lineage(args.init_weights_path)
        initialized_arch = initialization["model"].get("arch")
        compatible_expansion = (
            initialized_arch == "flow_compact_hermite_l"
            and args.arch == "flow_compact_lagrange_l"
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
            expected_missing = {
                "net.base_knot_head.weight",
                "net.base_knot_head.bias",
            }
            if (
                set(incompatible.missing_keys) != expected_missing
                or incompatible.unexpected_keys
            ):
                raise ValueError(
                    "unexpected Lagrange warm-start incompatibility: "
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
        f"profile={model.loss_profile} "
        f"trainable_scope={model.trainable_scope}",
        flush=True,
    )
    model.hparams["training_protocol"] = {
        "requested_arch": requested_arch,
        "canonical_arch": canonical_arch_name(requested_arch),
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
        "release_memmap_pages": args.release_memmap_pages,
        "seed": args.seed,
        "loss_profile": model.loss_profile,
        "trainable_scope": model.trainable_scope,
        "distill_weight": model.distill_weight,
        "distill_teacher_checkpoint": args.distill_teacher_checkpoint,
        "warmup_steps": args.warmup_steps,
        "lambda_hf": model.lambda_hf,
        "lambda_hf_override": args.lambda_hf_override,
        "highpass_boundary": model.highpass_boundary,
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
    }
    if args.arch in (
        "upr_lite",
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
        "upr_spherical_implicit_global_14m",
    ):
        training_sources["weatherbridge_upr_lite_model.py"] = (
            Path(__file__).resolve().parents[2]
            / "weather_time_interp"
            / "model"
            / "weatherbridge_upr_lite_model.py"
        )
    if args.arch in (
        "upr_implicit_global_14m",
        "upr_endpoint_implicit_global_14m",
        "upr_local_corr_14m",
        "upr_query_match_14m",
        "upr_spherical_implicit_global_14m",
    ):
        training_sources["weatherbridge_upr_scaled_model.py"] = (
            Path(__file__).resolve().parents[2]
            / "weather_time_interp"
            / "model"
            / "weatherbridge_upr_scaled_model.py"
        )
    if args.arch == "upr_spherical_implicit_global_14m":
        training_sources["weatherbridge_upr_spherical_model.py"] = (
            Path(__file__).resolve().parents[2]
            / "weather_time_interp"
            / "model"
            / "weatherbridge_upr_spherical_model.py"
        )
    if args.arch in (
        "flow",
        "flow_noskip",
        "flow_ungated",
        "flow_accel",
        "flow_pp",
        "flow_pp2",
        "flow_pp3",
        "flow_dual",
        "flow_spherical_ep",
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
    if args.arch in ("amt", "amt_residual"):
        amt_root = (
            repo_root
            / "weather_time_interp"
            / "model"
            / "amt_upstream"
        )
        training_sources["weather_amt_model.py"] = (
            repo_root
            / "weather_time_interp"
            / "model"
            / "weather_amt_model.py"
        )
        if args.arch == "amt_residual":
            training_sources["weather_amt_residual_model.py"] = (
                repo_root
                / "weather_time_interp"
                / "model"
                / "weather_amt_residual_model.py"
            )
        for source_name in (
            "__init__.py",
            "feat_enc.py",
            "flow_utils.py",
            "ifrnet.py",
            "multi_flow.py",
            "raft.py",
        ):
            training_sources[f"amt_upstream/{source_name}"] = (
                amt_root / source_name
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
