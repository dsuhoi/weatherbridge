"""Lightweight recurrent-pyramid WeatherBridge variants.

The model keeps expensive learned processing below full resolution. A shared
coarse-to-fine refiner estimates bidirectional motion, while the original
anchor fields are warped at full resolution so their small-scale structure is
not reconstructed through a bottleneck.
"""
from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


UPR_LITE_VARIANTS = (
    "upr_lite",
    "upr_lite_lap",
    "upr_lite_column",
    "upr_lite_continuous",
    "upr_lite_continuous_m",
    "upr_lite_implicit_global",
    "upr_lite_implicit_global_q4",
)


def upr_lite_variant_kwargs(arch: str) -> dict[str, object]:
    """Return the canonical constructor arguments for an experiment arm."""
    if arch not in UPR_LITE_VARIANTS:
        raise ValueError(f"unknown UPR-Lite variant {arch}")
    continuous = arch in {
        "upr_lite_continuous",
        "upr_lite_continuous_m",
        "upr_lite_implicit_global",
        "upr_lite_implicit_global_q4",
    }
    column_flow = arch in {
        "upr_lite_column",
        "upr_lite_continuous",
        "upr_lite_continuous_m",
        "upr_lite_implicit_global",
        "upr_lite_implicit_global_q4",
    }
    detail = arch != "upr_lite"
    medium = arch in {
        "upr_lite_continuous_m",
        "upr_lite_implicit_global",
        "upr_lite_implicit_global_q4",
    }
    implicit_global = arch in {
        "upr_lite_implicit_global",
        "upr_lite_implicit_global_q4",
    }
    return {
        "in_channels": 24,
        "out_channels": 24,
        "n_static_features": 3,
        "hidden": 192 if medium else 128,
        "static_width": 24 if medium else 16,
        "n_blocks": 6,
        "n_flow_modes": 3 if column_flow else 1,
        "laplacian_detail": detail,
        "query_conditioned": not continuous,
        "quadratic_trajectory": continuous and not implicit_global,
        "cubic_trajectory": implicit_global,
        "global_tokens": 16 if implicit_global else 0,
        "hydrostatic_coupling": implicit_global,
        "flow_scale": 2.0,
        "pyramid_divisors": (
            (16, 8, 4)
            if arch == "upr_lite_implicit_global_q4"
            else (8, 4, 2)
        ),
    }


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _sphere_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return x
    x = F.pad(x, (pad, pad, 0, 0), mode="circular")
    return F.pad(x, (0, 0, pad, pad), mode="replicate")


class SphereConv2d(nn.Conv2d):
    """Longitude-periodic convolution with replicated pole padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 **kwargs):
        if kernel_size % 2 != 1:
            raise ValueError("SphereConv2d requires an odd kernel size")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            **kwargs,
        )
        self.sphere_padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(_sphere_pad(x, self.sphere_padding))


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        inner = channels * expansion
        self.norm = nn.GroupNorm(_norm_groups(channels), channels)
        self.expand = nn.Conv2d(channels, inner, 1)
        self.depthwise = SphereConv2d(
            inner,
            inner,
            3,
            groups=inner,
        )
        self.project = nn.Conv2d(inner, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(F.silu(self.norm(x)))
        h = self.depthwise(h)
        return x + self.project(F.silu(h))


class FrameStem(nn.Module):
    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        self.proj = SphereConv2d(in_channels, hidden, 3)
        self.blocks = nn.Sequential(
            DepthwiseResidualBlock(hidden),
            DepthwiseResidualBlock(hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


@lru_cache(maxsize=32)
def _coordinate_grid(
    device_type: str,
    device_index: int | None,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = (
        torch.device(device_type)
        if device_index is None
        else torch.device(device_type, device_index)
    )
    return torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )


def periodic_resize(
    x: torch.Tensor,
    size: tuple[int, int] | torch.Size,
) -> torch.Tensor:
    """Bilinear resize with align-corners-false geometry and periodic longitude."""
    out_h, out_w = (int(value) for value in size)
    in_h, in_w = x.shape[-2:]
    if (in_h, in_w) == (out_h, out_w):
        return x
    if min(in_h, in_w, out_h, out_w) < 1:
        raise ValueError("periodic resize requires non-empty grids")

    yy, xx = _coordinate_grid(
        x.device.type,
        x.device.index,
        out_h,
        out_w,
    )
    source_y = ((yy + 0.5) * (in_h / out_h) - 0.5).clamp(
        0,
        in_h - 1,
    )
    source_x = torch.remainder(
        (xx + 0.5) * (in_w / out_w) - 0.5,
        in_w,
    )
    periodic = torch.cat((x, x[..., :1]), dim=-1)
    grid = torch.stack(
        (
            2.0 * source_x / max(in_w, 1) - 1.0,
            2.0 * source_y / max(in_h - 1, 1) - 1.0,
        ),
        dim=-1,
    ).unsqueeze(0).expand(x.size(0), -1, -1, -1)
    with torch.autocast(device_type=x.device.type, enabled=False):
        resized = F.grid_sample(
            periodic.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    return resized.to(dtype=x.dtype)


def periodic_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warp with periodic longitude and clamped latitude.

    ``flow`` can be shared by all channels, ``(B, 2, H, W)``, or supplied per
    channel as ``(B, C, 2, H, W)``.
    """

    batch, channels, height, width = x.shape
    per_channel = flow.dim() == 5
    if per_channel:
        if flow.shape[:2] != (batch, channels):
            raise ValueError(
                f"per-channel flow {tuple(flow.shape)} does not match "
                f"input {tuple(x.shape)}"
            )
        flow_flat = flow.reshape(batch * channels, 2, height, width)
        input_flat = x.reshape(batch * channels, 1, height, width)
    else:
        if flow.shape != (batch, 2, height, width):
            raise ValueError(
                f"shared flow {tuple(flow.shape)} does not match "
                f"input {tuple(x.shape)}"
            )
        flow_flat = flow
        input_flat = x

    flow_float = flow_flat.float()
    yy, xx = _coordinate_grid(
        x.device.type,
        x.device.index,
        height,
        width,
    )
    source_x = torch.remainder(xx[None] + flow_float[:, 0], width)
    source_y = (yy[None] + flow_float[:, 1]).clamp(0, height - 1)
    # Append longitude zero so fractional samples after the last grid point
    # interpolate across the periodic seam instead of using border padding.
    input_periodic = torch.cat((input_flat, input_flat[..., :1]), dim=-1)
    grid_x = 2.0 * source_x / max(width, 1) - 1.0
    grid_y = 2.0 * source_y / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    with torch.autocast(device_type=x.device.type, enabled=False):
        warped = F.grid_sample(
            input_periodic.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    warped = warped.to(input_flat.dtype)
    if per_channel:
        return warped.reshape(batch, channels, height, width)
    return warped


def _evaluate_trajectory(
    trajectory: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate endpoint flow plus optional endpoint-preserving curvature."""
    if trajectory.dim() != 5 or trajectory.size(2) not in (4, 8, 12):
        raise ValueError(
            "trajectory must have shape (B, N, 4|8|12, H, W)"
        )
    time = tau.unsqueeze(1)
    forward = trajectory[:, :, :2] * time
    backward = trajectory[:, :, 2:4] * (1.0 - time)
    if trajectory.size(2) >= 8:
        interior = time * (1.0 - time)
        forward = forward + trajectory[:, :, 4:6] * interior
        backward = backward + trajectory[:, :, 6:8] * interior
    if trajectory.size(2) == 12:
        # The odd interior basis captures directional changes without imposing
        # time-reversal symmetry. It vanishes at both anchors, so endpoint
        # displacements remain exactly constrained.
        jerk_basis = interior * (2.0 * time - 1.0)
        forward = forward + trajectory[:, :, 8:10] * jerk_basis
        backward = backward + trajectory[:, :, 10:12] * jerk_basis
    return forward, backward


def endpoint_preserving_blend(
    logits: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Tilt the linear blend smoothly while retaining exact anchor limits."""
    tau_float = tau.float()
    prior = (1.0 - tau_float).clamp(1.0e-6, 1.0 - 1.0e-6)
    alpha = torch.sigmoid(torch.logit(prior) + 2.0 * logits.float())
    alpha = torch.where(tau_float <= 0.0, torch.ones_like(alpha), alpha)
    alpha = torch.where(tau_float >= 1.0, torch.zeros_like(alpha), alpha)
    return alpha.to(logits.dtype)


class GlobalTokenMixer(nn.Module):
    """Low-rank global matching at the coarsest pyramid resolution.

    Learned queries gather a small set of input-specific motion tokens from
    the whole globe. Every grid point then retrieves a mixture of those
    tokens. This supplies global context with O(HWT) rather than O((HW)^2)
    attention, where T is normally 16.
    """

    def __init__(
        self,
        channels: int,
        n_tokens: int = 16,
        token_dim: int = 48,
        transformer_depth: int = 0,
        n_heads: int = 4,
        area_weighted: bool = False,
    ):
        super().__init__()
        if transformer_depth < 0:
            raise ValueError("transformer_depth must be non-negative")
        if token_dim <= 0 or token_dim % n_heads:
            raise ValueError("token_dim must be positive and divisible by n_heads")
        self.n_tokens = int(n_tokens)
        self.token_dim = int(token_dim)
        self.area_weighted = bool(area_weighted)
        self.norm = nn.GroupNorm(_norm_groups(channels), channels)
        self.key = nn.Conv2d(channels, token_dim, 1, bias=False)
        self.value = nn.Conv2d(channels, token_dim, 1, bias=False)
        self.query = nn.Conv2d(channels, token_dim, 1, bias=False)
        self.token_queries = nn.Parameter(
            torch.randn(n_tokens, token_dim) / token_dim ** 0.5
        )
        self.transformer = (
            nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=token_dim,
                        nhead=n_heads,
                        dim_feedforward=4 * token_dim,
                        dropout=0.0,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(transformer_depth)
                ]
            )
            if transformer_depth > 0
            else None
        )
        self.token_norm = (
            nn.LayerNorm(token_dim)
            if transformer_depth > 0
            else None
        )
        self.token_key = nn.Linear(token_dim, token_dim, bias=False)
        self.token_value = nn.Linear(token_dim, token_dim, bias=False)
        self.project = nn.Conv2d(token_dim, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    @staticmethod
    def _log_cell_area(
        height: int,
        width: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        latitude = (
            torch.pi / 2.0
            - (torch.arange(height, device=device, dtype=torch.float32) + 0.5)
            * (torch.pi / height)
        )
        area = latitude.cos().clamp_min(1.0e-4)
        return area.log().repeat_interleave(width).reshape(1, 1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            normalized = self.norm(x.float())
            keys = self.key(normalized).flatten(2)
            values = self.value(normalized).flatten(2)
            gather_logits = torch.einsum(
                "td,bdn->btn",
                self.token_queries,
                keys,
            ) / self.token_dim ** 0.5
            if self.area_weighted:
                gather_logits = gather_logits + self._log_cell_area(
                    height,
                    width,
                    device=x.device,
                )
            gather = gather_logits.softmax(dim=-1)
            tokens = torch.einsum("btn,bdn->btd", gather, values)
            if self.transformer is not None:
                tokens = tokens + self.token_queries.unsqueeze(0)
                for block in self.transformer:
                    tokens = block(tokens)
                if self.token_norm is None:
                    raise RuntimeError("token norm is missing")
                tokens = self.token_norm(tokens)

            token_keys = self.token_key(tokens)
            token_values = self.token_value(tokens)
            queries = self.query(normalized).flatten(2).transpose(1, 2)
            dispatch_logits = torch.einsum(
                "bnd,btd->bnt",
                queries,
                token_keys,
            ) / self.token_dim ** 0.5
            dispatch = dispatch_logits.softmax(dim=-1)
            context = torch.einsum(
                "bnt,btd->bdn",
                dispatch,
                token_values,
            ).reshape(batch, self.token_dim, height, width)
            correction = self.project(context)
        return x + correction.to(dtype=x.dtype)


class QueryStateModulator(nn.Module):
    """Smooth arbitrary-time modulation for query-independent motion state."""

    def __init__(self, channels: int):
        super().__init__()
        self.scale_coefficients = nn.Parameter(torch.zeros(3, channels))
        self.shift_coefficients = nn.Parameter(torch.zeros(3, channels))

    def forward(
        self,
        state: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=state.device.type, enabled=False):
            time = tau.float()
            centered = 2.0 * time - 1.0
            interior = 4.0 * time * (1.0 - time)
            basis = torch.cat(
                (centered, interior, interior * centered),
                dim=1,
            )
            scale = torch.einsum(
                "bkhw,kc->bchw",
                basis,
                self.scale_coefficients.float(),
            )
            shift = torch.einsum(
                "bkhw,kc->bchw",
                basis,
                self.shift_coefficients.float(),
            )
            modulated = (
                state.float() * (1.0 + torch.tanh(scale))
                + torch.tanh(shift)
            )
        return modulated.to(dtype=state.dtype)


class PyramidRefiner(nn.Module):
    def __init__(
        self,
        hidden: int,
        static_width: int,
        n_flow_modes: int,
        n_blocks: int,
        motion_components: int,
        local_matching: bool = False,
        local_correlation_radius: int = 0,
        matching_width: int = 32,
    ):
        super().__init__()
        self.local_matching = bool(local_matching)
        self.local_correlation_radius = int(local_correlation_radius)
        if self.local_correlation_radius < 0:
            raise ValueError("local_correlation_radius must be non-negative")
        if self.local_matching and self.local_correlation_radius:
            raise ValueError(
                "local_matching and local correlation are mutually exclusive"
            )
        if matching_width <= 0:
            raise ValueError("matching_width must be positive")
        correlation_channels = (
            (2 * self.local_correlation_radius + 1) ** 2
            if self.local_correlation_radius
            else 0
        )
        in_channels = (
            hidden * (5 if self.local_matching else 4)
            + correlation_channels
            + static_width
            + n_flow_modes * motion_components
            + 1
        )
        self.motion_components = motion_components
        self.match_proj = (
            nn.Conv2d(hidden, matching_width, 1, bias=False)
            if self.local_correlation_radius
            else None
        )
        self.input_proj = nn.Conv2d(in_channels, hidden, 1)
        self.blocks = nn.Sequential(
            *(DepthwiseResidualBlock(hidden) for _ in range(n_blocks))
        )
        self.flow_head = SphereConv2d(
            hidden,
            n_flow_modes * motion_components,
            3,
        )
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    @staticmethod
    def _matching_features(
        feature0: torch.Tensor,
        feature1: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(
            device_type=feature0.device.type,
            enabled=False,
        ):
            left = feature0.float()
            right = feature1.float()
            left_scale = left.square().mean(
                dim=1,
                keepdim=True,
            ).add(1.0e-6).rsqrt()
            right_scale = right.square().mean(
                dim=1,
                keepdim=True,
            ).add(1.0e-6).rsqrt()
            matching = left * right * left_scale * right_scale
        return matching.to(dtype=feature0.dtype)

    def _local_correlation(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor,
    ) -> torch.Tensor:
        if self.match_proj is None:
            raise RuntimeError("local correlation projection is disabled")
        radius = self.local_correlation_radius
        kernel = 2 * radius + 1
        projected0 = self.match_proj(feature0)
        projected1 = self.match_proj(feature1)
        with torch.autocast(
            device_type=feature0.device.type,
            enabled=False,
        ):
            left = F.normalize(
                projected0.float(),
                dim=1,
                eps=1.0e-6,
            )
            right = F.normalize(
                projected1.float(),
                dim=1,
                eps=1.0e-6,
            )
            batch, channels, height, width = left.shape
            patches = F.unfold(
                _sphere_pad(right, radius),
                kernel_size=kernel,
            ).view(
                batch,
                channels,
                kernel * kernel,
                height,
                width,
            )
            correlation = (left.unsqueeze(2) * patches).sum(dim=1)
        return correlation.to(dtype=feature0.dtype)

    def forward(
        self,
        feature0: torch.Tensor,
        feature1: torch.Tensor,
        state: torch.Tensor,
        flow_modes: torch.Tensor,
        static: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = feature0.shape
        forward, backward = _evaluate_trajectory(flow_modes[:, :1], tau)
        warped0 = periodic_warp(feature0, forward[:, 0])
        warped1 = periodic_warp(feature1, backward[:, 0])
        tau_map = tau.expand(batch, 1, height, width)
        inputs = [warped0, warped1, warped1 - warped0]
        if self.local_matching:
            inputs.append(self._matching_features(warped0, warped1))
        if self.local_correlation_radius:
            inputs.append(self._local_correlation(warped0, warped1))
        inputs.extend(
            (state, flow_modes.flatten(1, 2), static, tau_map)
        )
        h = torch.cat(inputs, dim=1)
        state = self.blocks(self.input_proj(h))
        delta = torch.tanh(self.flow_head(state))
        delta = delta.view(
            batch,
            flow_modes.size(1),
            self.motion_components,
            height,
            width,
        )
        return state, flow_modes + delta


class WeatherBridgeUPRLiteModel(nn.Module):
    """Shared-pyramid flow estimator with optional spectral-detail transport."""

    _CHANNEL_TO_GROUP = (
        0, 1, 2, 3,  # T
        0, 1, 2, 3,  # U
        0, 1, 2, 3,  # V
        0, 1, 2, 3,  # Q
        0, 1, 2, 3,  # Z
        4, 4, 4, 4,  # surface
    )

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 24,
        n_static_features: int = 3,
        hidden: int = 64,
        static_width: int = 12,
        n_blocks: int = 4,
        n_flow_modes: int = 1,
        laplacian_detail: bool = False,
        query_conditioned: bool = True,
        quadratic_trajectory: bool = False,
        cubic_trajectory: bool = False,
        global_tokens: int = 0,
        global_token_dim: int = 48,
        global_transformer_depth: int = 0,
        global_transformer_heads: int = 4,
        global_area_weighted: bool = False,
        hydrostatic_coupling: bool = False,
        smooth_endpoint_blend: bool = False,
        local_matching: bool = False,
        local_correlation_radius: int = 0,
        matching_width: int = 32,
        query_state_modulation: bool = False,
        periodic_longitude_resize: bool = False,
        flow_scale: float = 2.0,
        pyramid_divisors: tuple[int, int, int] = (8, 4, 2),
    ):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("UPR-Lite requires matching anchor/output channels")
        if in_channels != len(self._CHANNEL_TO_GROUP):
            raise ValueError("UPR-Lite currently implements the canonical 24 fields")
        if n_flow_modes not in (1, 3):
            raise ValueError("n_flow_modes must be 1 or 3")
        if quadratic_trajectory and cubic_trajectory:
            raise ValueError(
                "quadratic_trajectory and cubic_trajectory are mutually exclusive"
            )
        if global_tokens < 0:
            raise ValueError("global_tokens must be non-negative")
        if global_token_dim <= 0:
            raise ValueError("global_token_dim must be positive")
        if (
            len(pyramid_divisors) != 3
            or any(divisor <= 0 for divisor in pyramid_divisors)
            or not (
                pyramid_divisors[0]
                > pyramid_divisors[1]
                > pyramid_divisors[2]
            )
        ):
            raise ValueError(
                "pyramid_divisors must contain three positive, "
                "strictly decreasing divisors"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_static_features = n_static_features
        self.n_flow_modes = n_flow_modes
        self.laplacian_detail = laplacian_detail
        self.query_conditioned = query_conditioned
        self.quadratic_trajectory = quadratic_trajectory
        self.cubic_trajectory = cubic_trajectory
        self.hydrostatic_coupling = hydrostatic_coupling
        self.smooth_endpoint_blend = bool(smooth_endpoint_blend)
        self.local_matching = bool(local_matching)
        self.local_correlation_radius = int(local_correlation_radius)
        self.query_state_modulation = bool(query_state_modulation)
        self.periodic_longitude_resize = bool(periodic_longitude_resize)
        self.motion_components = (
            12
            if cubic_trajectory
            else 8 if quadratic_trajectory else 4
        )
        self.flow_scale = float(flow_scale)
        self.pyramid_divisors = tuple(int(value) for value in pyramid_divisors)

        self.frame_stem = FrameStem(in_channels, hidden)
        self.static_stem = FrameStem(n_static_features, static_width)
        self.refiner = PyramidRefiner(
            hidden=hidden,
            static_width=static_width,
            n_flow_modes=n_flow_modes,
            n_blocks=n_blocks,
            motion_components=self.motion_components,
            local_matching=self.local_matching,
            local_correlation_radius=self.local_correlation_radius,
            matching_width=matching_width,
        )
        self.global_mixer = (
            GlobalTokenMixer(
                hidden,
                n_tokens=global_tokens,
                token_dim=global_token_dim,
                transformer_depth=global_transformer_depth,
                n_heads=global_transformer_heads,
                area_weighted=global_area_weighted,
            )
            if global_tokens > 0
            else None
        )
        self.blend_head = SphereConv2d(hidden, out_channels, 3)
        self.residual_head = SphereConv2d(hidden, out_channels, 3)
        self.detail_head = (
            SphereConv2d(hidden, out_channels, 3)
            if laplacian_detail
            else None
        )
        self.hydrostatic_head = (
            nn.Conv2d(8, 5, 1)
            if hydrostatic_coupling
            else None
        )
        self.query_state_modulator = (
            QueryStateModulator(hidden)
            if self.query_state_modulation
            else None
        )
        for head in (self.blend_head, self.residual_head, self.detail_head):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        if self.hydrostatic_head is not None:
            nn.init.zeros_(self.hydrostatic_head.weight)
            nn.init.zeros_(self.hydrostatic_head.bias)
        self.residual_scale = nn.Parameter(torch.full((out_channels,), 0.1))

        if n_flow_modes == 1:
            basis = torch.ones(5, 1)
        else:
            vertical_coordinate = torch.tensor([1.0, 0.5, 0.0, -1.0, 1.2])
            basis = torch.stack(
                (
                    torch.ones_like(vertical_coordinate),
                    vertical_coordinate,
                    0.5 * (3.0 * vertical_coordinate.square() - 1.0),
                ),
                dim=1,
            )
        self.register_buffer("vertical_basis", basis, persistent=True)
        self.register_buffer(
            "channel_to_group",
            torch.tensor(self._CHANNEL_TO_GROUP, dtype=torch.long),
            persistent=True,
        )
        grouped_order = torch.cat(
            [
                torch.nonzero(
                    self.channel_to_group == group,
                    as_tuple=False,
                ).flatten()
                for group in range(5)
            ]
        )
        self.register_buffer("grouped_order", grouped_order, persistent=True)
        self.register_buffer(
            "inverse_grouped_order",
            torch.argsort(grouped_order),
            persistent=True,
        )
        self.register_buffer(
            "hydrostatic_indices",
            torch.tensor((16, 17, 18, 19, 23), dtype=torch.long),
            persistent=False,
        )

    def _resize_flow(
        self,
        flow: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        old_h, old_w = flow.shape[-2:]
        new_h, new_w = size
        batch, modes, components = flow.shape[:3]
        flat_flow = flow.reshape(
            batch,
            modes * components,
            old_h,
            old_w,
        )
        if self.periodic_longitude_resize:
            flat_flow = periodic_resize(flat_flow, size)
        else:
            flat_flow = F.interpolate(
                flat_flow,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        resized = flat_flow.view(
            batch,
            modes,
            components,
            new_h,
            new_w,
        )
        resized[:, :, 0::2] *= new_w / old_w
        resized[:, :, 1::2] *= new_h / old_h
        return resized

    def _prepare_static(
        self,
        static: torch.Tensor | None,
        batch: int,
        size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if static is None:
            return torch.zeros(
                batch,
                self.n_static_features,
                *size,
                device=device,
                dtype=dtype,
            )
        if static.dim() == 3:
            static = static.unsqueeze(0)
        if static.size(0) == 1:
            static = static.expand(batch, -1, -1, -1)
        if static.size(0) != batch:
            raise ValueError(f"static batch {static.size(0)} != input batch {batch}")
        static = static[:, :self.n_static_features].to(device=device, dtype=dtype)
        return F.interpolate(static, size=size, mode="bilinear", align_corners=False)

    def _group_flows(self, modes: torch.Tensor) -> torch.Tensor:
        basis = self.vertical_basis.to(modes.dtype)
        return torch.einsum("gm,bmkhw->bgkhw", basis, modes)

    def _warp_fields(
        self,
        fields: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        if flow.dim() == 4:
            return periodic_warp(fields, flow)
        batch, channels, height, width = fields.shape
        if flow.shape != (batch, 5, 2, height, width):
            raise ValueError(
                f"group flow {tuple(flow.shape)} does not match "
                f"input {tuple(fields.shape)}"
            )
        ordered = fields.index_select(1, self.grouped_order)
        padding = fields.new_zeros(batch, 1, height, width)
        grouped = torch.cat((ordered, padding), dim=1).reshape(
            batch * 5,
            5,
            height,
            width,
        )
        warped = periodic_warp(
            grouped,
            flow.reshape(batch * 5, 2, height, width),
        )
        warped_ordered = warped.reshape(
            batch,
            25,
            height,
            width,
        )[:, :channels]
        return warped_ordered.index_select(1, self.inverse_grouped_order)

    @staticmethod
    def _smooth(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(_sphere_pad(x, 1), kernel_size=3, stride=1)

    def forward(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau: torch.Tensor,
        cond: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del cond
        batch, _, height, width = x0.shape
        tau = tau.reshape(batch, 1, 1, 1).to(dtype=x0.dtype)
        motion_tau = (
            tau
            if self.query_conditioned
            else torch.full_like(tau, 0.5)
        )
        finest_divisor = self.pyramid_divisors[-1]
        work_size = (
            max(1, height // finest_divisor),
            max(1, width // finest_divisor),
        )
        x0_work = F.interpolate(x0, size=work_size, mode="area")
        xT_work = F.interpolate(xT, size=work_size, mode="area")
        feature0_work = self.frame_stem(x0_work)
        featureT_work = self.frame_stem(xT_work)
        static_work = self._prepare_static(
            static,
            batch,
            work_size,
            x0.device,
            x0.dtype,
        )
        static_feature_work = self.static_stem(static_work)

        pyramid_sizes = [
            (
                max(1, height // divisor),
                max(1, width // divisor),
            )
            for divisor in self.pyramid_divisors
        ]
        state = None
        flow_modes = None
        for level, size in enumerate(pyramid_sizes):
            feature0 = F.adaptive_avg_pool2d(feature0_work, size)
            featureT = F.adaptive_avg_pool2d(featureT_work, size)
            static_feature = F.adaptive_avg_pool2d(static_feature_work, size)
            if state is None:
                state = torch.zeros_like(feature0)
                flow_modes = feature0.new_zeros(
                    batch,
                    self.n_flow_modes,
                    self.motion_components,
                    *size,
                )
            else:
                if self.periodic_longitude_resize:
                    state = periodic_resize(state, size)
                else:
                    state = F.interpolate(
                        state,
                        size=size,
                        mode="bilinear",
                        align_corners=False,
                    )
                flow_modes = self._resize_flow(flow_modes, size)
            state, flow_modes = self.refiner(
                feature0,
                featureT,
                state,
                flow_modes,
                static_feature,
                motion_tau,
            )
            if level == 0 and self.global_mixer is not None:
                state = self.global_mixer(state)

        flow_modes = self._resize_flow(flow_modes, (height, width))
        flow_modes = flow_modes * self.flow_scale
        if self.n_flow_modes == 1:
            forward_flow, backward_flow = _evaluate_trajectory(
                flow_modes,
                tau,
            )
            forward_flow = forward_flow[:, 0]
            backward_flow = backward_flow[:, 0]
        else:
            group_trajectory = self._group_flows(flow_modes)
            forward_flow, backward_flow = _evaluate_trajectory(
                group_trajectory,
                tau,
            )

        synthesis_state = (
            self.query_state_modulator(state, tau)
            if self.query_state_modulator is not None
            else state
        )
        blend_logits = self.blend_head(synthesis_state)
        if self.periodic_longitude_resize:
            blend_logits = periodic_resize(
                blend_logits,
                (height, width),
            )
        else:
            blend_logits = F.interpolate(
                blend_logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        if self.smooth_endpoint_blend:
            alpha = endpoint_preserving_blend(blend_logits, tau)
        else:
            alpha = (
                (1.0 - tau)
                + 2.0 * tau * (1.0 - tau) * torch.tanh(blend_logits)
            ).clamp(0.0, 1.0)

        warped0 = self._warp_fields(x0, forward_flow)
        warpedT = self._warp_fields(xT, backward_flow)
        scaffold = alpha * warped0 + (1.0 - alpha) * warpedT
        if self.detail_head is not None:
            detail = (
                alpha * (warped0 - self._smooth(warped0))
                + (1.0 - alpha) * (warpedT - self._smooth(warpedT))
            )
            detail_logits = self.detail_head(synthesis_state)
            if self.periodic_longitude_resize:
                detail_logits = periodic_resize(
                    detail_logits,
                    (height, width),
                )
            else:
                detail_logits = F.interpolate(
                    detail_logits,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            detail_gain = (
                1.0
                + 2.0 * tau * (1.0 - tau) * torch.tanh(detail_logits)
            )
            scaffold = scaffold + (detail_gain - 1.0) * detail

        residual = self.residual_head(synthesis_state)
        if self.periodic_longitude_resize:
            residual = periodic_resize(
                residual,
                (height, width),
            )
        else:
            residual = F.interpolate(
                residual,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        scale = torch.tanh(self.residual_scale).view(1, -1, 1, 1)
        delta = tau * (1.0 - tau) * scale * residual
        prediction = scaffold + delta
        hydrostatic_delta = prediction.new_zeros(
            batch,
            5,
            height,
            width,
        )
        if self.hydrostatic_head is not None:
            column = torch.cat(
                (prediction[:, 0:4], prediction[:, 12:16]),
                dim=1,
            )
            hydrostatic_delta = (
                tau
                * (1.0 - tau)
                * self.hydrostatic_head(column).to(prediction.dtype)
            )
            prediction = prediction.index_add(
                1,
                self.hydrostatic_indices,
                hydrostatic_delta,
            )
        prediction = torch.where(tau == 0, x0, prediction)
        prediction = torch.where(tau == 1, xT, prediction)
        return prediction, {
            "flow_modes": flow_modes,
            "alpha": alpha,
            "delta": delta,
            "hydrostatic_delta": hydrostatic_delta,
        }
