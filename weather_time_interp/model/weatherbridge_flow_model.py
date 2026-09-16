"""Transport-aware interpolation between two global weather states.

The default input stack is ``[x0, xT, xT - x0, static]``. A strided
circular-convolution U-Net predicts bidirectional displacement, anchor blend,
transport/scaffold gate and multiscale residual heads at low resolution. The
full-resolution output combines warped anchors, the linear interpolation
scaffold and an additive correction. Configuration flags enable the SFNO
low-mode branch, temperature-conditioned mass-field correction, pole-aware
anchor-detail bypass and experimental trajectory variants.

Decoder lateral features can pass through zero-initialised gates conditioned
on normalised query time ``t``. ``gated_skip=False`` selects plain skips and
``use_skip=False`` removes them.

``forward(x0, xT, tau, cond=None, static=None) -> (prediction, diagnostics)``
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- tau embedding
class SinusoidalPosEmb(nn.Module):
    """Low-order harmonic basis over NORMALISED tau in [0, 1].

    tau reaches the net as tau_hour / delta_t (see the trainer's
    TauRescaleAnd24chWrapper), so it always lives in [0, 1]. Using a small set
    of LOW harmonic frequencies {pi, 2pi, ..., half*pi} keeps the
    tau -> modulation map smooth and low-order, so the model INTERPOLATES to
    unseen tau (train {1,3,5} -> eval {2,4}) instead of memorising the training
    tau and zig-zagging in between. This is the fix for the t2m held-out spike:
    t2m carries the strongest diurnal cycle, so its optimal correction varies
    most with tau -> a rich (128-freq) embedding overfit the 3 training tau and
    blew up at {2,4}. A few smooth harmonics still capture the diurnal
    curvature over the 6 h window while generalising across tau."""

    def __init__(self, dim: int, base_period: float = 16.0):
        super().__init__()
        self.dim = dim
        self.base_period = base_period  # kept for signature compat (unused)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        k = torch.arange(1, half + 1, device=t.device, dtype=torch.float32)
        args = t.view(-1, 1).float() * (math.pi * k).view(1, -1)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeMLP(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        freq_dim: int = 8,
        base_period: float = 16.0,
        zero_init_output: bool = False,
    ):
        super().__init__()
        self.emb = SinusoidalPosEmb(freq_dim, base_period)
        self.net = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(),
                                  nn.Linear(dim, dim))
        if zero_init_output:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(self.emb(t))


class AdaLNZero(nn.Module):
    """Zero-init AdaLN modulation on (B, C, H, W) conditioned on tau embedding."""

    def __init__(self, dim: int, time_dim: int = 256):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim * 3))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        g, b, gate = self.mlp(temb).chunk(3, dim=-1)
        g = g[:, :, None, None]; b = b[:, :, None, None]; gate = gate[:, :, None, None]
        return h + gate * ((1 + g) * self.norm(h) + b)


class EndpointMultibandCalibrator(nn.Module):
    """Restore band amplitudes without changing their predicted phase."""

    def __init__(
        self,
        channels: int,
        time_dim: int,
        n_bands: int = 3,
        max_gain: float = 0.25,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_bands = int(n_bands)
        self.max_gain = float(max_gain)
        self.gain_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, self.channels * self.n_bands),
        )
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.zeros_(self.gain_head[-1].bias)

    def forward(
        self,
        field: torch.Tensor,
        tau: torch.Tensor,
        time_embedding: torch.Tensor,
        pole_parity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bands = spherical_multiband_components(
            field,
            pole_parity,
            dilations=(1, 2, 4)[: self.n_bands],
        )
        gains = self.max_gain * torch.tanh(
            self.gain_head(time_embedding)
        ).view(field.size(0), self.n_bands, self.channels, 1, 1)
        correction = sum(
            gains[:, index] * band
            for index, band in enumerate(bands)
        )
        envelope = 4.0 * tau * (1.0 - tau)
        return field + envelope * correction, correction


# ---------------------------------------------------------------- conv blocks
def _pole_pad_latitude(
    x: torch.Tensor,
    pad: int,
    pole_parity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append latitude rows reflected through each pole at the antipode."""
    if pad == 0:
        return x
    if x.size(-1) % 2:
        raise ValueError("spherical padding requires an even longitude width")
    if pad > x.size(-2):
        raise ValueError("spherical padding exceeds the latitude height")
    half_width = x.size(-1) // 2
    top = torch.roll(
        x[..., :pad, :].flip(-2),
        shifts=half_width,
        dims=-1,
    )
    bottom = torch.roll(
        x[..., -pad:, :].flip(-2),
        shifts=half_width,
        dims=-1,
    )
    if pole_parity is not None:
        if pole_parity.numel() != x.size(-3):
            raise ValueError(
                "pole parity must have one value per input channel"
            )
        parity = pole_parity.to(
            device=x.device,
            dtype=x.dtype,
        ).view(1, -1, 1, 1)
        top = top * parity
        bottom = bottom * parity
    return torch.cat((top, x, bottom), dim=-2)


def _sphere_pad(
    x: torch.Tensor,
    pad: int,
    pole_parity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pad an equirectangular field across the true spherical boundaries."""
    x = _pole_pad_latitude(x, pad, pole_parity)
    return F.pad(x, (pad, pad, 0, 0), mode="circular")


def spherical_multiband_components(
    field: torch.Tensor,
    pole_parity: torch.Tensor | None = None,
    dilations: tuple[int, ...] = (1, 2, 4),
) -> tuple[torch.Tensor, ...]:
    """Split a lat-lon field into fixed, phase-preserving detail bands."""
    if field.ndim != 4:
        raise ValueError("multiband input must have shape (B,C,H,W)")
    if not dilations or any(dilation <= 0 for dilation in dilations):
        raise ValueError("multiband dilations must be positive")
    channels = field.size(1)
    kernel = field.new_full((channels, 1, 3, 3), 1.0 / 9.0)
    current = field
    bands = []
    for dilation in dilations:
        smooth = F.conv2d(
            _sphere_pad(current, dilation, pole_parity),
            kernel,
            dilation=dilation,
            groups=channels,
        )
        bands.append(current - smooth)
        current = smooth
    return tuple(bands)


def _spherical_resize(
    x: torch.Tensor,
    size: tuple[int, int] | torch.Size,
    pole_parity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bilinear resize on a periodic longitude/antipodal latitude grid."""
    out_h, out_w = (int(value) for value in size)
    in_h, in_w = x.shape[-2:]
    if (in_h, in_w) == (out_h, out_w):
        return x
    if min(in_h, in_w, out_h, out_w) < 1:
        raise ValueError("spherical resize requires non-empty spatial grids")

    yy, xx = torch.meshgrid(
        torch.arange(out_h, device=x.device, dtype=torch.float32),
        torch.arange(out_w, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    source_y = (yy + 0.5) * (in_h / out_h) - 0.5
    source_x = (xx + 0.5) * (in_w / out_w) - 0.5
    pole_pad = min(1, in_h)
    sample = _pole_pad_latitude(x, pole_pad, pole_parity)
    sample = torch.cat((sample, sample[..., :1]), dim=-1)
    gx = torch.remainder(source_x, in_w) / in_w * 2.0 - 1.0
    gy = (
        (source_y + pole_pad)
        / max(sample.size(-2) - 1, 1)
        * 2.0
        - 1.0
    )
    grid = torch.stack((gx, gy), dim=-1)
    grid = grid.unsqueeze(0).expand(x.size(0), -1, -1, -1)
    with torch.autocast(device_type=x.device.type, enabled=False):
        output = F.grid_sample(
            sample.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    return output.to(x.dtype)


def _endpoint_blend(
    blend_logits: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Smooth learned blend with exact anchor limits and linear zero-init."""
    tau_float = tau.float()
    prior = (1.0 - tau_float).clamp(1.0e-6, 1.0 - 1.0e-6)
    alpha = torch.sigmoid(torch.logit(prior) + 2.0 * blend_logits.float())
    alpha = torch.where(tau_float <= 0.0, torch.ones_like(alpha), alpha)
    alpha = torch.where(tau_float >= 1.0, torch.zeros_like(alpha), alpha)
    return alpha.to(blend_logits.dtype)


class SphereConv2d(nn.Conv2d):
    """Convolution with periodic longitude and antipodal pole padding."""

    def __init__(
        self,
        ci: int,
        co: int,
        kernel_size: int = 3,
        stride: int = 1,
        pole_parity: torch.Tensor | None = None,
    ):
        if kernel_size % 2 != 1:
            raise ValueError("SphereConv2d requires an odd kernel size")
        super().__init__(ci, co, kernel_size, stride=stride, padding=0)
        self.sphere_padding = kernel_size // 2
        self.register_buffer(
            "pole_parity",
            (
                None
                if pole_parity is None
                else torch.as_tensor(pole_parity, dtype=torch.float32)
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(
            _sphere_pad(x, self.sphere_padding, self.pole_parity)
        )


class SphereConvTranspose2d(nn.ConvTranspose2d):
    """Stride-2 transpose convolution without planar decoder boundaries.

    Padding one low-resolution ghost cell adds the neighbour required by a
    4x4, stride-2 transpose convolution. Cropping the corresponding two
    output cells restores the exact shape and alignment of the legacy layer.
    The inherited parameter layout keeps checkpoint keys and capacity fixed.
    """

    def __init__(self, ci: int, co: int):
        super().__init__(ci, co, kernel_size=4, stride=2, padding=1)
        self.sphere_input_padding = 1

    def forward(
        self,
        x: torch.Tensor,
        output_size: list[int] | None = None,
    ) -> torch.Tensor:
        if output_size is not None:
            raise ValueError(
                "SphereConvTranspose2d does not support output_size"
            )
        pad = self.sphere_input_padding
        expanded = _sphere_pad(x, pad)
        output = F.conv_transpose2d(
            expanded,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )
        crop = pad * self.stride[0]
        return output[..., crop:-crop, crop:-crop]


def _conv3x3(
    ci: int,
    co: int,
    stride: int,
    spherical_ops: bool,
    pole_parity: torch.Tensor | None = None,
) -> nn.Module:
    if spherical_ops:
        return SphereConv2d(
            ci,
            co,
            3,
            stride=stride,
            pole_parity=pole_parity,
        )
    return nn.Conv2d(
        ci,
        co,
        3,
        stride,
        1,
        padding_mode="circular",
    )


def conv_block(
    ci,
    co,
    stride=1,
    spherical_ops: bool = False,
    input_pole_parity: torch.Tensor | None = None,
):
    return nn.Sequential(
        _conv3x3(
            ci,
            co,
            stride,
            spherical_ops,
            pole_parity=input_pole_parity,
        ),
        nn.GroupNorm(min(8, co), co), nn.SiLU(),
        _conv3x3(co, co, 1, spherical_ops),
        nn.GroupNorm(min(8, co), co), nn.SiLU(),
    )


class GatedSkip(nn.Module):
    """Fuse a decoder feature with an encoder skip through a zero-init tau-gate.

    out = up + gate(tau) * proj(cat[up, skip]); gate zero-init so training
    starts as if there were no skip (no overshoot), and only opens where the
    interpolation genuinely needs the encoder detail.
    """

    def __init__(
        self,
        dim: int,
        time_dim: int = 256,
        spherical_ops: bool = False,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            _conv3x3(dim * 2, dim, 1, spherical_ops),
            nn.GroupNorm(min(8, dim), dim), nn.SiLU())
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, up, skip, temb):
        g = torch.tanh(self.gate(temb))[:, :, None, None]
        return up + g * self.proj(torch.cat([up, skip], dim=1))


class BottleneckTokenMixer(nn.Module):
    """Low-rank global context at the coarsest encoder resolution."""

    def __init__(self, channels: int, n_tokens: int, token_dim: int):
        super().__init__()
        self.token_dim = int(token_dim)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.key = nn.Conv2d(channels, token_dim, 1, bias=False)
        self.value = nn.Conv2d(channels, token_dim, 1, bias=False)
        self.query = nn.Conv2d(channels, token_dim, 1, bias=False)
        self.token_queries = nn.Parameter(
            torch.randn(n_tokens, token_dim) / token_dim ** 0.5
        )
        self.token_key = nn.Linear(token_dim, token_dim, bias=False)
        self.token_value = nn.Linear(token_dim, token_dim, bias=False)
        self.project = nn.Conv2d(token_dim, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del temb
        batch, _, height, width = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            normalized = self.norm(x.float())
            keys = self.key(normalized).flatten(2)
            values = self.value(normalized).flatten(2)
            gather = (
                torch.einsum(
                    "td,bdn->btn",
                    self.token_queries,
                    keys,
                )
                / self.token_dim ** 0.5
            ).softmax(dim=-1)
            tokens = torch.einsum("btn,bdn->btd", gather, values)
            token_keys = self.token_key(tokens)
            token_values = self.token_value(tokens)
            queries = self.query(normalized).flatten(2).transpose(1, 2)
            dispatch = (
                torch.einsum("bnd,btd->bnt", queries, token_keys)
                / self.token_dim ** 0.5
            ).softmax(dim=-1)
            context = torch.einsum(
                "bnt,btd->bdn",
                dispatch,
                token_values,
            ).reshape(batch, self.token_dim, height, width)
            correction = self.project(context)
        return x + correction.to(dtype=x.dtype)


class SphericalLatentTransformer(nn.Module):
    """Area-aware global attention with a fixed set of latent tokens.

    Cross-attention scales linearly with the number of grid cells. Learned
    latents first gather the 45x90 bottleneck using spherical cell-area
    weights, mix globally in token space, and then broadcast the result back
    to the grid. No absolute longitude embedding is used, preserving cyclic
    longitude equivariance inherited from the convolutional encoder.
    """

    def __init__(
        self,
        channels: int,
        n_tokens: int = 16,
        token_dim: int = 128,
        time_dim: int = 256,
        depth: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        if n_tokens <= 0:
            raise ValueError("n_tokens must be positive")
        if token_dim <= 0 or token_dim % n_heads:
            raise ValueError("token_dim must be positive and divisible by n_heads")
        if depth <= 0:
            raise ValueError("depth must be positive")

        self.token_dim = int(token_dim)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.grid_key_value = nn.Conv2d(
            channels,
            2 * token_dim,
            kernel_size=1,
            bias=False,
        )
        self.grid_query = nn.Conv2d(
            channels,
            token_dim,
            kernel_size=1,
            bias=False,
        )
        self.latents = nn.Parameter(
            torch.randn(n_tokens, token_dim) / token_dim ** 0.5
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, token_dim),
        )
        self.transformer = nn.ModuleList(
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
                for _ in range(depth)
            ]
        )
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_key = nn.Linear(token_dim, token_dim, bias=False)
        self.token_value = nn.Linear(token_dim, token_dim, bias=False)
        self.project = nn.Conv2d(token_dim, channels, kernel_size=1)
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
            math.pi / 2.0
            - (torch.arange(height, device=device, dtype=torch.float32) + 0.5)
            * (math.pi / height)
        )
        area = latitude.cos().clamp_min(1.0e-4)
        return area.log().repeat_interleave(width).reshape(1, 1, -1)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if temb is None:
            raise ValueError("SphericalLatentTransformer requires a time embedding")
        batch, _, height, width = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            normalized = self.norm(x.float())
            keys, values = self.grid_key_value(normalized).chunk(2, dim=1)
            keys = keys.flatten(2)
            values = values.flatten(2)

            latent_queries = self.latents.unsqueeze(0).expand(batch, -1, -1)
            latent_queries = latent_queries + self.time_projection(
                temb.float()
            ).unsqueeze(1)
            gather_logits = (
                torch.einsum("btd,bdn->btn", latent_queries, keys)
                / self.token_dim ** 0.5
            )
            gather_logits = gather_logits + self._log_cell_area(
                height,
                width,
                device=x.device,
            )
            gather = gather_logits.softmax(dim=-1)
            tokens = latent_queries + torch.einsum(
                "btn,bdn->btd",
                gather,
                values,
            )
            for block in self.transformer:
                tokens = block(tokens)
            tokens = self.token_norm(tokens)

            grid_queries = self.grid_query(normalized).flatten(2).transpose(1, 2)
            token_keys = self.token_key(tokens)
            token_values = self.token_value(tokens)
            dispatch = (
                torch.einsum("bnd,btd->bnt", grid_queries, token_keys)
                / self.token_dim ** 0.5
            ).softmax(dim=-1)
            context = torch.einsum(
                "bnt,btd->bdn",
                dispatch,
                token_values,
            ).reshape(batch, self.token_dim, height, width)
            correction = self.project(context)
        return x + correction.to(dtype=x.dtype)


class SharedContentTimeController(nn.Module):
    """Weight-shared low-rank controls conditioned on field content and time."""

    def __init__(
        self,
        input_channels: int = 8,
        hidden_channels: int = 16,
        time_dim: int = 256,
        output_channels: int = 4,
    ):
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        self.content = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        self.time_affine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2 * hidden_channels),
        )
        self.output = nn.Conv2d(
            hidden_channels,
            output_channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        content: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if content.size(0) != time_embedding.size(0):
            raise ValueError("content and time embedding batches must match")
        hidden = self.content(content)
        scale, shift = self.time_affine(time_embedding).chunk(2, dim=-1)
        scale = 0.25 * torch.tanh(scale).unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        hidden = F.silu(hidden * (1.0 + scale) + shift)
        return self.output(hidden)


def _evaluate_endpoint_trajectory(
    trajectory: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a query-independent cubic path with exact anchor limits."""
    if trajectory.dim() != 5 or trajectory.size(2) != 12:
        raise ValueError("cubic trajectory must have shape (B,N,12,H,W)")
    time = tau.unsqueeze(1)
    interior = time * (1.0 - time)
    odd_interior = interior * (2.0 * time - 1.0)
    forward = (
        trajectory[:, :, :2] * time
        + trajectory[:, :, 4:6] * interior
        + trajectory[:, :, 8:10] * odd_interior
    )
    backward = (
        trajectory[:, :, 2:4] * (1.0 - time)
        + trajectory[:, :, 6:8] * interior
        + trajectory[:, :, 10:12] * odd_interior
    )
    return forward, backward


def _evaluate_endpoint_tangent_correction(
    tangents: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Evaluate separate left/right Hermite tangent corrections."""
    if tangents.dim() != 4 or tangents.size(1) % 2:
        raise ValueError("endpoint tangents must have shape (B,2C,H,W)")
    time = tau.reshape(-1, 1, 1, 1)
    left, right = tangents.chunk(2, dim=1)
    return time * (1.0 - time) * (
        (1.0 - time) * left - time * right
    )


def _evaluate_base_knot_correction(
    knots: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Interpolate three interior correction maps with exact anchor zeros."""
    if knots.dim() != 4 or knots.size(1) % 3:
        raise ValueError("base-field knots must have shape (B,3C,H,W)")
    time = tau.reshape(-1, 1, 1, 1)
    maps = knots.chunk(3, dim=1)
    nodes = (1.0 / 6.0, 0.5, 5.0 / 6.0)
    endpoint = time * (1.0 - time)
    correction = torch.zeros_like(maps[0])
    for index, node in enumerate(nodes):
        basis = endpoint / (node * (1.0 - node))
        for other_index, other in enumerate(nodes):
            if other_index != index:
                basis = basis * (time - other) / (node - other)
        correction = correction + basis * maps[index]
    return correction


def warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    *,
    periodic_longitude: bool = False,
    pole_parity: torch.Tensor | None = None,
    sampling_mode: str = "bilinear",
) -> torch.Tensor:
    """Backward-warp ``x`` by pixel-space flow.

    The retained checkpoints use this equirectangular grid-sample operator in
    both axes. The opt-in spherical branch wraps longitude and maps samples
    crossing either pole to the antipodal reflected latitude.
    """
    if sampling_mode not in ("bilinear", "bicubic"):
        raise ValueError("warp sampling_mode must be bilinear or bicubic")
    B, C, H, W = x.shape
    if not periodic_longitude:
        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        gx = (xx + flow[:, 0]) / max(W - 1, 1) * 2 - 1
        gy = (yy + flow[:, 1]) / max(H - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)
        return F.grid_sample(
            x,
            grid,
            mode=sampling_mode,
            padding_mode="border",
            align_corners=True,
        )
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32),
        torch.arange(W, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    flow_float = flow.float()
    # The branch's tanh-bounded velocity and acceleration heads permit at
    # most 12 pixels of displacement. Sixteen antipodal ghost rows therefore
    # cover every production sample while allowing grid_sample to interpolate
    # continuously on both sides of the physical pole at y=-0.5/H-0.5.
    pole_pad = min(16, H)
    sample = _pole_pad_latitude(x, pole_pad, pole_parity)
    sample = torch.cat((sample, sample[..., :1]), dim=-1)
    source_y = yy[None] + flow_float[:, 1] + pole_pad
    source_x = torch.remainder(xx[None] + flow_float[:, 0], W)
    gx = source_x / max(W, 1) * 2 - 1
    gy = source_y / max(sample.size(-2) - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)
    with torch.autocast(device_type=x.device.type, enabled=False):
        warped = F.grid_sample(
            sample.float(),
            grid,
            mode=sampling_mode,
            padding_mode="border",
            align_corners=True,
        )
    return warped.to(x.dtype)


CANONICAL_WIND_PAIRS = (
    (4, 8),
    (5, 9),
    (6, 10),
    (7, 11),
    (21, 22),
)


def intrinsic_spherical_warp(
    x: torch.Tensor,
    tangent_flow: torch.Tensor,
    *,
    vector_pairs: tuple[tuple[int, int], ...] = CANONICAL_WIND_PAIRS,
    channel_mean: torch.Tensor | None = None,
    channel_std: torch.Tensor | None = None,
    geometry_basis: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Backward-warp fields along great circles on the unit sphere.

    ``tangent_flow`` is expressed in equatorial grid-cell arc lengths in the
    local east/south basis. Vector pairs are sampled at the source point,
    parallel-transported along the geodesic, and projected into the target
    east/north basis.
    """
    if x.ndim != 4 or tangent_flow.ndim != 4:
        raise ValueError("spherical warp expects BCHW fields and B2HW flow")
    batch, channels, height, width = x.shape
    if tangent_flow.shape != (batch, 2, height, width):
        raise ValueError("spherical tangent flow must have shape (B,2,H,W)")
    for east_index, north_index in vector_pairs:
        if min(east_index, north_index) < 0 or max(
            east_index,
            north_index,
        ) >= channels:
            raise ValueError("vector pair index exceeds field channels")
    if (channel_mean is None) != (channel_std is None):
        raise ValueError("channel mean and standard deviation must be paired")
    if channel_mean is not None:
        if channel_mean.numel() != channels or channel_std.numel() != channels:
            raise ValueError("normalization must contain one value per channel")

    if geometry_basis is None:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=x.device, dtype=torch.float32),
            torch.arange(width, device=x.device, dtype=torch.float32),
            indexing="ij",
        )
        latitude = math.pi / 2.0 - (yy + 0.5) * (math.pi / height)
        longitude = (xx + 0.5) * (2.0 * math.pi / width)
        cos_latitude = latitude.cos()
        target = torch.stack(
            (
                cos_latitude * longitude.cos(),
                cos_latitude * longitude.sin(),
                latitude.sin(),
            ),
            dim=0,
        ).unsqueeze(0)
        east_basis = torch.stack(
            (
                -longitude.sin(),
                longitude.cos(),
                torch.zeros_like(longitude),
            ),
            dim=0,
        ).unsqueeze(0)
        north_basis = torch.stack(
            (
                -latitude.sin() * longitude.cos(),
                -latitude.sin() * longitude.sin(),
                cos_latitude,
            ),
            dim=0,
        ).unsqueeze(0)
    else:
        target, east_basis, north_basis = geometry_basis
        expected_shape = (1, 3, height, width)
        if any(value.shape != expected_shape for value in geometry_basis):
            raise ValueError("cached spherical geometry has the wrong shape")

    with torch.autocast(device_type=x.device.type, enabled=False):
        flow = tangent_flow.float()
        east_arc = flow[:, 0] * (2.0 * math.pi / width)
        north_arc = -flow[:, 1] * (math.pi / height)
        tangent = (
            east_arc.unsqueeze(1) * east_basis
            + north_arc.unsqueeze(1) * north_basis
        )
        distance = torch.sqrt(
            (
                east_arc.square() + north_arc.square()
            ).clamp_min(1.0e-12)
        )
        sinc = torch.sin(distance) / distance
        source = (
            torch.cos(distance).unsqueeze(1) * target
            + sinc.unsqueeze(1) * tangent
        )
        source = F.normalize(source, dim=1)
        polar_half_cell = math.pi / (2.0 * height)
        max_abs_latitude = math.pi / 2.0 - polar_half_cell
        max_abs_z = math.sin(max_abs_latitude)
        source_latitude = torch.asin(
            source[:, 2].clamp(-max_abs_z, max_abs_z)
        )
        source_xy_norm = torch.sqrt(
            source[:, 0].square() + source[:, 1].square()
        )
        target_xy_norm = torch.sqrt(
            target[:, 0].square() + target[:, 1].square()
        ).clamp_min(1.0e-6)
        fallback_x = target[:, 0] / target_xy_norm
        fallback_y = target[:, 1] / target_xy_norm
        resolved_xy = source_xy_norm >= math.sin(polar_half_cell)
        stable_source_x = torch.where(
            resolved_xy,
            source[:, 0],
            fallback_x * math.sin(polar_half_cell),
        )
        stable_source_y = torch.where(
            resolved_xy,
            source[:, 1],
            fallback_y * math.sin(polar_half_cell),
        )
        source_longitude = torch.atan2(stable_source_y, stable_source_x)
        source_latitude_cos = source_latitude.cos()
        sampled_source = torch.stack(
            (
                source_latitude_cos * source_longitude.cos(),
                source_latitude_cos * source_longitude.sin(),
                source_latitude.sin(),
            ),
            dim=1,
        )
        source_x = torch.remainder(
            source_longitude * (width / (2.0 * math.pi)) - 0.5,
            width,
        )
        source_y = (
            (math.pi / 2.0 - source_latitude)
            * (height / math.pi)
            - 0.5
        )
        periodic_sample = torch.cat((x.float(), x[..., :1].float()), dim=-1)
        grid = torch.stack(
            (
                source_x / max(width, 1) * 2.0 - 1.0,
                source_y / max(height - 1, 1) * 2.0 - 1.0,
            ),
            dim=-1,
        )
        sampled = F.grid_sample(
            periodic_sample,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        if vector_pairs:
            source_longitude_sin = source_longitude.sin()
            source_longitude_cos = source_longitude.cos()
            source_latitude_sin = source_latitude.sin()
            source_latitude_cos = source_latitude.cos()
            source_east_basis = torch.stack(
                (
                    -source_longitude_sin,
                    source_longitude_cos,
                    torch.zeros_like(source_longitude),
                ),
                dim=1,
            )
            source_north_basis = torch.stack(
                (
                    -source_latitude_sin * source_longitude_cos,
                    -source_latitude_sin * source_longitude_sin,
                    source_latitude_cos,
                ),
                dim=1,
            )
            result = sampled.clone()
            denominator = (
                1.0 + (sampled_source * target).sum(dim=1)
            ).clamp_min(1.0e-6)
            path_sum = sampled_source + target
            for east_index, north_index in vector_pairs:
                east_component = sampled[:, east_index]
                north_component = sampled[:, north_index]
                if channel_mean is not None and channel_std is not None:
                    east_component = (
                        east_component * channel_std[east_index]
                        + channel_mean[east_index]
                    )
                    north_component = (
                        north_component * channel_std[north_index]
                        + channel_mean[north_index]
                    )
                vector = (
                    east_component.unsqueeze(1) * source_east_basis
                    + north_component.unsqueeze(1) * source_north_basis
                )
                transported = vector - (
                    (vector * target).sum(dim=1) / denominator
                ).unsqueeze(1) * path_sum
                transported_east = (transported * east_basis).sum(dim=1)
                transported_north = (transported * north_basis).sum(dim=1)
                if channel_mean is not None and channel_std is not None:
                    transported_east = (
                        transported_east - channel_mean[east_index]
                    ) / channel_std[east_index]
                    transported_north = (
                        transported_north - channel_mean[north_index]
                    ) / channel_std[north_index]
                result[:, east_index] = transported_east
                result[:, north_index] = transported_north
            sampled = result
    return sampled.to(x.dtype)


class SphericalConv(nn.Module):
    """Spherical Fourier Neural Operator layer (SFNO, Bonev et al. 2023 /
    NVIDIA). Uses the real Spherical Harmonic Transform (torch_harmonics) to
    project onto the LOW spherical-harmonic modes, mixes them with a learned
    complex weight, then transforms back. Unlike a planar FFT this is the
    global basis on the lat/lon sphere and avoids the planar-FFT pole seam for
    smooth planetary mass fields such as geopotential and MSLP."""

    def __init__(self, in_ch: int, out_ch: int, nlat: int, nlon: int,
                 lmax: int = 20, mmax: int = 20, grid: str = "equiangular"):
        super().__init__()
        from torch_harmonics import RealSHT, InverseRealSHT
        self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid)
        self.isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid)
        s = 1.0 / (in_ch * out_ch)
        self.w = nn.Parameter(s * torch.randn(in_ch, out_ch, lmax, mmax, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.sht(x)                                     # (B,C,lmax,mmax) complex
        c = torch.einsum("bilm,iolm->bolm", c, self.w)
        return self.isht(c)


class SpectralBranch(nn.Module):
    """Global low-wavenumber residual branch on the SPHERE (SFNO). Runs in fp32
    (SHT + complex weights are not autocast-safe). Zero-init output so it starts
    neutral. Construction requires ``torch_harmonics`` to be installed."""

    def __init__(self, base: int, out_channels: int, width: int = 20,
                 nlat: int = 360, nlon: int = 720, modes: int = 20):
        super().__init__()
        self.pin = nn.Conv2d(base, width, 1)
        self.spec = SphericalConv(width, width, nlat, nlon, lmax=modes, mmax=modes)
        self.act = nn.GELU()
        self.pout = nn.Conv2d(width, out_channels, 1)
        nn.init.zeros_(self.pout.weight); nn.init.zeros_(self.pout.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        with torch.autocast(h.device.type, enabled=False):
            z = self.pin(h.float())
            z = self.act(self.spec(z) + z)
            return self.pout(z)


class WeatherBridgeModel(nn.Module):
    _CHANNEL_TO_GROUP = (
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        4, 4, 4, 4,
    )

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 24,
        n_static_features: int = 3,
        hidden: int = 96,
        n_levels: int = 3,
        time_emb_dim: int = 256,
        # Width of the harmonic query-time basis, [sin(k pi t), cos(k pi t)]
        # for k = 1 .. time_freq_dim/2. With three trained query hours the
        # training loss constrains only three directions, so harmonics above
        # the second are poorly determined and cos(3 pi t) is not determined at
        # all: it is exactly zero at tau in {1,3,5}/6 and +-1 at {2,4}/6.
        # Lowering this removes those aliasing modes by construction.
        time_freq_dim: int = 8,
        residual_scale_init: float = 0.10,
        use_skip: bool = True,
        gated_skip: bool = True,
        flow_scale: float = 8.0,
        lat_crop: int = 0,
        use_accel: bool = False,
        strong_residual: bool = False,
        mass_aware_gate: bool = False,
        spectral_branch: bool = False,
        hydro_couple: bool = False,
        dual_stream: bool = False,
        spherical_ops: bool = False,
        endpoint_preserving: bool = False,
        query_independent_trajectory: bool = False,
        n_flow_modes: int = 1,
        cubic_trajectory: bool = False,
        global_tokens: int = 0,
        global_token_dim: int = 32,
        endpoint_tangents: bool = False,
        base_field_knots: bool = False,
        multiscale_field_experts: bool = False,
        intrinsic_spherical_transport: bool = False,
        use_frame_difference: bool = True,
        multiband_calibration: bool = False,
        anchor_detail_bypass: bool = False,
        flow_matching: bool = False,
        flow_matching_steps: int = 4,
        shared_field_controls: bool = False,
        latent_transformer_tokens: int = 0,
        latent_transformer_dim: int = 128,
        latent_transformer_depth: int = 2,
        latent_transformer_heads: int = 4,
        decoder_blocks_per_level: int = 1,
        content_adaptive_controls: bool = False,
        time_content_adaptive_controls: bool = False,
        forecast_lead_conditioning: bool = False,
        forecast_lead_scale_hours: float = 120.0,
        hres_residual_adapter: bool = False,
        degradation_aware: bool = False,
        anchor_error_reference: float = 0.35,
    ):
        super().__init__()
        if n_flow_modes not in (1, 3):
            raise ValueError("n_flow_modes must be 1 or 3")
        if n_flow_modes > 1 and out_channels != len(self._CHANNEL_TO_GROUP):
            raise ValueError("vertical flow modes require the canonical 24 fields")
        if global_tokens < 0:
            raise ValueError("global_tokens must be non-negative")
        if global_token_dim <= 0:
            raise ValueError("global_token_dim must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_static_features = int(n_static_features)
        self.hidden = int(hidden)
        self.n_levels = int(n_levels)
        self.use_skip = bool(use_skip)
        self.gated_skip = bool(gated_skip)
        self.flow_scale = float(flow_scale)
        self.lat_crop = int(lat_crop)
        # Quadratic transport: warp by F(t)*t + 1/2 A(t)*t^2 instead of F(t)*t.
        # F and A remain query-conditioned; this expands the displacement
        # parameterization without claiming one global constant-acceleration
        # trajectory across every query time.
        self.use_accel = bool(use_accel)
        # Evaluation-only component lesions. Checkpoint loading sets these
        # attributes explicitly; training always leaves them disabled.
        self.ablate_acceleration = False
        self.ablate_transport = False
        self.ablate_hydrostatic = False
        # Strong residual: prepend a conv block before the (zero-init) fine
        # residual head so the NON-transport correction has decoder-like
        # capacity. This targets fields where warping does not help — smooth
        # mass (mslp) and the diurnal-local part of t2m — where a thin 1-conv
        # residual trails WeatherDCAE's full DC-AE decoder. Warp still owns
        # the moving fields; this only deepens the additive correction.
        self.strong_residual = bool(strong_residual)
        self.spherical_ops = bool(spherical_ops)
        self.endpoint_preserving = bool(endpoint_preserving)
        self.query_independent_trajectory = bool(
            query_independent_trajectory
        )
        self.n_flow_modes = int(n_flow_modes)
        self.cubic_trajectory = bool(cubic_trajectory)
        self.endpoint_tangents = bool(endpoint_tangents)
        self.base_field_knots = bool(base_field_knots)
        self.multiscale_field_experts = bool(multiscale_field_experts)
        self.intrinsic_spherical_transport = bool(
            intrinsic_spherical_transport
        )
        self.use_frame_difference = bool(use_frame_difference)
        self.multiband_calibration = bool(multiband_calibration)
        self.anchor_detail_bypass = bool(anchor_detail_bypass)
        self.flow_matching = bool(flow_matching)
        self.flow_matching_steps = int(flow_matching_steps)
        self.shared_field_controls = bool(shared_field_controls)
        self.content_adaptive_controls = bool(content_adaptive_controls)
        self.time_content_adaptive_controls = bool(
            time_content_adaptive_controls
        )
        self.forecast_lead_conditioning = bool(forecast_lead_conditioning)
        self.forecast_lead_scale_hours = float(forecast_lead_scale_hours)
        self.hres_residual_adapter = bool(hres_residual_adapter)
        # Degradation-aware interpolation: the anchors are forecast states, so
        # their error grows with forecast lead and is spatially heterogeneous.
        # A single lead scalar cannot express "these anchors are unreliable
        # HERE". The reliability branch predicts the local anchor error field
        # and uses it to withdraw trust from the transported anchors in favour
        # of the smooth scaffold and the learned residual.
        self.degradation_aware = bool(degradation_aware)
        self.anchor_error_reference = float(anchor_error_reference)
        if self.forecast_lead_scale_hours <= 0.0:
            raise ValueError("forecast lead scale must be positive")
        if self.hres_residual_adapter and not self.forecast_lead_conditioning:
            raise ValueError(
                "HRES residual adapter requires forecast lead conditioning"
            )
        if self.degradation_aware and not self.forecast_lead_conditioning:
            raise ValueError(
                "degradation-aware conditioning requires forecast lead"
            )
        if self.anchor_error_reference <= 0.0:
            raise ValueError("anchor error reference must be positive")
        if self.flow_matching_steps < 1:
            raise ValueError("flow_matching_steps must be positive")
        if self.shared_field_controls and self.n_flow_modes != 1:
            raise ValueError("shared field controls require one flow mode")
        if self.content_adaptive_controls and not self.shared_field_controls:
            raise ValueError(
                "content-adaptive controls require shared field controls"
            )
        if (
            self.time_content_adaptive_controls
            and not self.shared_field_controls
        ):
            raise ValueError(
                "time-content controls require shared field controls"
            )
        if (
            self.content_adaptive_controls
            and self.time_content_adaptive_controls
        ):
            raise ValueError(
                "content-only and time-content controls are mutually exclusive"
            )
        if global_tokens > 0 and latent_transformer_tokens > 0:
            raise ValueError(
                "global token mixer and latent transformer are mutually exclusive"
            )
        if decoder_blocks_per_level < 1:
            raise ValueError("decoder_blocks_per_level must be positive")
        if self.multiband_calibration and self.anchor_detail_bypass:
            raise ValueError(
                "multiband calibration and anchor detail bypass are separate arms"
            )
        if self.anchor_detail_bypass and self.n_flow_modes != 1:
            raise ValueError("anchor detail bypass requires one flow mode")
        if self.anchor_detail_bypass and self.intrinsic_spherical_transport:
            raise ValueError(
                "anchor detail bypass does not support intrinsic transport"
            )
        if self.endpoint_tangents and not self.endpoint_preserving:
            raise ValueError("endpoint tangents require endpoint preservation")
        if self.base_field_knots and not self.endpoint_preserving:
            raise ValueError("base-field knots require endpoint preservation")
        if self.base_field_knots and out_channels != 24:
            raise ValueError("base-field knots require the canonical 24 fields")
        if self.multiscale_field_experts and not self.endpoint_preserving:
            raise ValueError(
                "multiscale field experts require endpoint preservation"
            )
        if self.multiscale_field_experts and out_channels != 24:
            raise ValueError(
                "multiscale field experts require the canonical 24 fields"
            )
        if self.multiscale_field_experts and n_levels < 3:
            raise ValueError(
                "multiscale field experts require at least three decoder levels"
            )
        if self.intrinsic_spherical_transport and not self.spherical_ops:
            raise ValueError(
                "intrinsic spherical transport requires spherical operators"
            )
        if self.intrinsic_spherical_transport and self.n_flow_modes != 1:
            raise ValueError(
                "intrinsic spherical transport currently requires one flow mode"
            )
        if self.intrinsic_spherical_transport and out_channels != 24:
            raise ValueError(
                "intrinsic spherical transport requires the canonical 24 fields"
            )
        self.register_buffer(
            "normalization_mean",
            (
                torch.zeros(out_channels)
                if self.intrinsic_spherical_transport
                else torch.empty(0)
            ),
            persistent=self.intrinsic_spherical_transport,
        )
        self.register_buffer(
            "normalization_std",
            (
                torch.ones(out_channels)
                if self.intrinsic_spherical_transport
                else torch.empty(0)
            ),
            persistent=self.intrinsic_spherical_transport,
        )
        if self.intrinsic_spherical_transport:
            yy, xx = torch.meshgrid(
                torch.arange(360, dtype=torch.float32),
                torch.arange(720, dtype=torch.float32),
                indexing="ij",
            )
            latitude = math.pi / 2.0 - (yy + 0.5) * (math.pi / 360)
            longitude = (xx + 0.5) * (2.0 * math.pi / 720)
            cos_latitude = latitude.cos()
            geometry = (
                torch.stack(
                    (
                        cos_latitude * longitude.cos(),
                        cos_latitude * longitude.sin(),
                        latitude.sin(),
                    ),
                    dim=0,
                ).unsqueeze(0),
                torch.stack(
                    (
                        -longitude.sin(),
                        longitude.cos(),
                        torch.zeros_like(longitude),
                    ),
                    dim=0,
                ).unsqueeze(0),
                torch.stack(
                    (
                        -latitude.sin() * longitude.cos(),
                        -latitude.sin() * longitude.sin(),
                        cos_latitude,
                    ),
                    dim=0,
                ).unsqueeze(0),
            )
        else:
            geometry = (torch.empty(0),) * 3
        self.register_buffer(
            "sphere_target",
            geometry[0],
            persistent=False,
        )
        self.register_buffer(
            "sphere_east_basis",
            geometry[1],
            persistent=False,
        )
        self.register_buffer(
            "sphere_north_basis",
            geometry[2],
            persistent=False,
        )
        self.motion_components = (
            12 if self.cubic_trajectory else 8 if self.use_accel else 4
        )
        field_pole_parity = torch.ones(out_channels)
        for index in (*range(4, 12), 21, 22):
            if index < out_channels:
                field_pole_parity[index] = -1.0
        self.register_buffer(
            "field_pole_parity",
            field_pole_parity,
            persistent=False,
        )

        self.time_freq_dim = int(time_freq_dim)
        self.time_mlp = TimeMLP(time_emb_dim, freq_dim=self.time_freq_dim)
        self.forecast_lead_mlp = (
            TimeMLP(time_emb_dim, freq_dim=self.time_freq_dim,
                    zero_init_output=True)
            if self.forecast_lead_conditioning
            else None
        )
        self.flow_time_mlp = (
            TimeMLP(time_emb_dim, freq_dim=self.time_freq_dim)
            if self.flow_matching
            else None
        )
        ch = [hidden * (2 ** i) for i in range(n_levels + 1)]

        # Main input: [x0, xT, xT-x0, static]. The structural NoDiff
        # ablation removes the difference channels and their stem weights.
        n_anchor_inputs = 3 if self.use_frame_difference else 2
        enc_in = (
            n_anchor_inputs * in_channels
            + n_static_features
            + (in_channels if self.flow_matching else 0)
        )
        self.encoder = nn.ModuleList()
        self.encoder.append(
            conv_block(
                enc_in,
                ch[0],
                stride=1,
                spherical_ops=self.spherical_ops,
                input_pole_parity=(
                    torch.cat(
                        (
                            *(
                                field_pole_parity
                                for _ in range(n_anchor_inputs)
                            ),
                            *(
                                (field_pole_parity,)
                                if self.flow_matching
                                else ()
                            ),
                            torch.ones(n_static_features),
                        )
                    )
                    if self.spherical_ops
                    else None
                ),
            )
        )
        for i in range(n_levels):
            self.encoder.append(
                conv_block(
                    ch[i],
                    ch[i + 1],
                    stride=2,
                    spherical_ops=self.spherical_ops,
                )
            )

        self.adaln = AdaLNZero(ch[-1], time_dim=time_emb_dim)
        if latent_transformer_tokens > 0:
            self.global_mixer = SphericalLatentTransformer(
                ch[-1],
                n_tokens=latent_transformer_tokens,
                token_dim=latent_transformer_dim,
                time_dim=time_emb_dim,
                depth=latent_transformer_depth,
                n_heads=latent_transformer_heads,
            )
        elif global_tokens > 0:
            self.global_mixer = BottleneckTokenMixer(
                ch[-1],
                n_tokens=global_tokens,
                token_dim=global_token_dim,
            )
        else:
            self.global_mixer = None

        # decoder
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.skip = nn.ModuleList()
        for i in range(n_levels):
            self.up.append(
                SphereConvTranspose2d(
                    ch[-(i + 1)],
                    ch[-(i + 2)],
                )
                if self.spherical_ops
                else nn.ConvTranspose2d(
                    ch[-(i + 1)],
                    ch[-(i + 2)],
                    4,
                    2,
                    1,
                )
            )
            decoder_channels = ch[-(i + 2)]
            self.dec.append(
                nn.Sequential(
                    *[
                        conv_block(
                            decoder_channels,
                            decoder_channels,
                            spherical_ops=self.spherical_ops,
                        )
                        for _ in range(decoder_blocks_per_level)
                    ]
                )
            )
            if self.use_skip:
                self.skip.append(
                    GatedSkip(
                        ch[-(i + 2)],
                        time_emb_dim,
                        spherical_ops=self.spherical_ops,
                    )
                    if gated_skip
                    else _conv3x3(
                        ch[-(i + 2)] * 2,
                        ch[-(i + 2)],
                        1,
                        self.spherical_ops,
                    )
                )

        base = ch[0]
        # heads (operate on the stem-resolution feature)
        n_flow = self.n_flow_modes * self.motion_components
        self.flow_head = _conv3x3(
            base,
            n_flow,
            1,
            self.spherical_ops,
        )
        control_channels = 1 if self.shared_field_controls else out_channels
        self.blend_head = _conv3x3(
            base,
            control_channels,
            1,
            self.spherical_ops,
        )
        self.res_coarse = _conv3x3(
            ch[1],
            out_channels,
            1,
            self.spherical_ops,
        )
        # optional decoder-capacity block before the zero-init fine head
        self.res_body = (
            conv_block(base, base, spherical_ops=self.spherical_ops)
            if self.strong_residual
            else nn.Identity()
        )
        self.res_fine = _conv3x3(
            base,
            out_channels,
            1,
            self.spherical_ops,
        )
        self.hres_residual_head = (
            nn.Conv2d(base + 2 * out_channels, out_channels, 1)
            if self.hres_residual_adapter
            else None
        )
        if self.hres_residual_head is not None:
            nn.init.zeros_(self.hres_residual_head.weight)
            nn.init.zeros_(self.hres_residual_head.bias)
        # --- anchor-reliability branch (degradation-aware interpolation) ---
        # Predicts the LOCAL anchor error magnitude from decoder features,
        # modulated by the forecast lead through a zero-init FiLM. It is
        # supervised during fine-tuning by the measured anchor error, so at
        # inference the model estimates its own input degradation without
        # labels. Both consumers are zero-init gains, so a warm-started model
        # reproduces its parent exactly before any optimisation step.
        if self.degradation_aware:
            self.reliability_stem = _conv3x3(
                base,
                base,
                1,
                self.spherical_ops,
            )
            self.reliability_norm = nn.GroupNorm(min(8, base), base)
            self.reliability_film = nn.Linear(time_emb_dim, 2 * base)
            nn.init.zeros_(self.reliability_film.weight)
            nn.init.zeros_(self.reliability_film.bias)
            self.reliability_out = _conv3x3(
                base,
                out_channels,
                1,
                self.spherical_ops,
            )
            # Start the branch at the reference anchor error, so the predicted
            # excess is exactly zero and BOTH consumers are strictly neutral at
            # initialisation: a warm-started model reproduces its parent.
            reference_bias = float(
                math.log(math.expm1(self.anchor_error_reference))
            )
            for module in self.reliability_out.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, reference_bias)
            # Withdraw anchor trust where the predicted anchor error is high.
            self.anchor_trust_gain = nn.Parameter(
                torch.zeros(control_channels)
            )
            # Lean harder on the learned residual when the anchors degrade.
            self.anchor_residual_gain = nn.Parameter(
                torch.zeros(out_channels)
            )
        else:
            self.reliability_stem = None
            self.reliability_norm = None
            self.reliability_film = None
            self.reliability_out = None
        self.multiband_calibrator = (
            EndpointMultibandCalibrator(
                out_channels,
                time_emb_dim,
                n_bands=3,
                max_gain=0.25,
            )
            if self.multiband_calibration
            else None
        )
        self.detail_gain_head = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, control_channels),
            )
            if self.anchor_detail_bypass
            else None
        )
        if self.detail_gain_head is not None:
            nn.init.zeros_(self.detail_gain_head[-1].weight)
            nn.init.zeros_(self.detail_gain_head[-1].bias)
        # A single weight-shared controller can route each field according to
        # anchor/warp content without allocating learned parameters by class.
        if self.content_adaptive_controls:
            self.content_control_head = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(16, 2, kernel_size=1),
            )
            nn.init.zeros_(self.content_control_head[-1].weight)
            nn.init.zeros_(self.content_control_head[-1].bias)
        elif self.time_content_adaptive_controls:
            self.content_control_head = SharedContentTimeController(
                input_channels=8,
                hidden_channels=16,
                time_dim=time_emb_dim,
                output_channels=4,
            )
        else:
            self.content_control_head = None
        self.tangent_head = (
            _conv3x3(
                base,
                2 * out_channels,
                1,
                self.spherical_ops,
            )
            if self.endpoint_tangents
            else None
        )
        self.base_knot_head = (
            _conv3x3(
                base,
                3 * 20,
                1,
                self.spherical_ops,
            )
            if self.base_field_knots
            else None
        )
        self.multiscale_expert_heads = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(channels, 6 * 8, kernel_size=1),
                        nn.GroupNorm(6, 6 * 8),
                        nn.SiLU(),
                        nn.Conv2d(
                            6 * 8,
                            6 * 3 * 4,
                            kernel_size=1,
                            groups=6,
                        ),
                    )
                    for channels in (ch[2], ch[1], ch[0])
                ]
            )
            if self.multiscale_field_experts
            else None
        )
        self.register_buffer(
            "multiscale_knot_pole_parity",
            (
                field_pole_parity.repeat(3)
                if self.multiscale_field_experts
                else torch.empty(0)
            ),
            persistent=False,
        )
        self.register_buffer(
            "base_field_indices",
            torch.tensor((*range(12), *range(16, 24)), dtype=torch.long),
            persistent=False,
        )
        initialized_heads = (
            self.flow_head,
            self.blend_head,
            self.res_coarse,
            self.res_fine,
        )
        for m in initialized_heads:
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        if self.tangent_head is not None:
            nn.init.zeros_(self.tangent_head.weight)
            nn.init.zeros_(self.tangent_head.bias)
        if self.base_knot_head is not None:
            nn.init.zeros_(self.base_knot_head.weight)
            nn.init.zeros_(self.base_knot_head.bias)
        if self.multiscale_expert_heads is not None:
            for expert in self.multiscale_expert_heads:
                nn.init.zeros_(expert[-1].weight)
                nn.init.zeros_(expert[-1].bias)
        # global low-wavenumber (FNO) residual for smooth planetary mass fields
        self.spectral = SpectralBranch(base, out_channels) if spectral_branch else None
        # hydrostatic coupling: correct Z1000-700 + mslp from the predicted T
        # column (thickness ~ layer-mean T). 1x1 (local column relation), zero-init.
        self.hydro = nn.Conv2d(4, 5, 1) if hydro_couple else None
        if self.hydro is not None:
            nn.init.zeros_(self.hydro.weight); nn.init.zeros_(self.hydro.bias)
        # ---- dual-stream: a second, SKIP-LESS global decoder from the bottleneck
        # latent (DC-AE-like: no warp, no skips -> global planetary reconstruction)
        # fused with the transport stream by a learned per-channel gate. Winds/
        # moisture route to transport (gate->1), smooth mass to the decoder (->0).
        self.dual_stream = bool(dual_stream)
        if self.dual_stream:
            self.gdec = nn.ModuleList()
            for i in range(n_levels):
                co = ch[-(i + 2)]
                self.gdec.append(nn.Sequential(
                    (
                        SphereConvTranspose2d(ch[-(i + 1)], co)
                        if self.spherical_ops
                        else nn.ConvTranspose2d(
                            ch[-(i + 1)],
                            co,
                            4,
                            2,
                            1,
                        )
                    ),
                    nn.GroupNorm(min(8, co), co), nn.SiLU(),
                    _conv3x3(co, co, 1, self.spherical_ops),
                    nn.GroupNorm(min(8, co), co), nn.SiLU()))
            self.gout = _conv3x3(
                ch[0],
                out_channels,
                1,
                self.spherical_ops,
            )
            nn.init.zeros_(self.gout.weight); nn.init.zeros_(self.gout.bias)
            # per-channel stream gate: sigmoid(0)=0.5 start, learns routing
            self.stream_gate = nn.Parameter(torch.zeros(out_channels))
        self.scale = nn.Parameter(
            torch.full((control_channels,), float(residual_scale_init))
        )
        # Per-channel warp gate: sigmoid(warp_gate)=1 -> trust the flow-warped
        # frames (moving fields: wind, moisture); =0 -> fall back to the plain
        # bilinear scaffold (quasi-static fields: t2m/mslp locked to orography
        # where warping only distorts). Init +2 (~0.88) so training starts near
        # the warp-only behaviour and only pulls static channels down.
        self.warp_gate = nn.Parameter(torch.full((control_channels,), 2.0))
        # Mass-aware init: for quasi-static / mass channels (Z1000-700, t2m,
        # mslp in the 24ch order) start warp_gate at -2 (sigmoid ~0.12) so they
        # ride the smooth bilinear scaffold from step 0 instead of the warped
        # frames. Warping a planetary-scale pressure/geopotential field only
        # injects spurious small-scale structure; these fields are the decoder's
        # (residual's) job, not transport's.
        if mass_aware_gate:
            if self.shared_field_controls:
                raise ValueError(
                    "mass-aware and shared field gates are mutually exclusive"
                )
            # ONLY the true mass/pressure fields (geopotential, mslp) — these are
            # quasi-static and warping them only distorts. t2m is deliberately
            # EXCLUDED: its weakness is the diurnal residual, not warp distortion,
            # and it carries partial temperature-front advection that the warp can
            # help — forcing beta->0 there would bias against a real signal.
            mass_idx = [16, 17, 18, 19, 23]  # Z1000,Z925,Z850,Z700,mslp
            with torch.no_grad():
                for c in mass_idx:
                    if c < out_channels:
                        self.warp_gate[c] = -2.0

        vertical_coordinate = torch.tensor([1.0, 0.5, 0.0, -1.0, 1.2])
        vertical_basis = (
            torch.ones(5, 1)
            if self.n_flow_modes == 1
            else torch.stack(
                (
                    torch.ones_like(vertical_coordinate),
                    vertical_coordinate,
                    0.5 * (
                        3.0 * vertical_coordinate.square() - 1.0
                    ),
                ),
                dim=1,
            )
        )
        channel_to_group = torch.tensor(
            self._CHANNEL_TO_GROUP,
            dtype=torch.long,
        )
        grouped_order = torch.cat(
            [
                torch.nonzero(
                    channel_to_group == group,
                    as_tuple=False,
                ).flatten()
                for group in range(5)
            ]
        )
        self.register_buffer(
            "vertical_basis",
            vertical_basis,
            persistent=False,
        )
        self.register_buffer(
            "grouped_order",
            grouped_order,
            persistent=False,
        )
        self.register_buffer(
            "inverse_grouped_order",
            torch.argsort(grouped_order),
            persistent=False,
        )
        self.register_buffer(
            "grouped_pole_parity",
            torch.tensor((1.0, -1.0, -1.0, 1.0, 1.0)),
            persistent=False,
        )

    def _prep_static(self, static, B, device):
        if self.n_static_features <= 0:
            return None
        if static.dim() == 3:
            static = static.unsqueeze(0)
        if static.size(0) != B:
            static = (static.repeat_interleave(B // static.size(0), 0)
                      if B % static.size(0) == 0 else static.expand(B, -1, -1, -1))
        return static.to(device)

    def _group_trajectory(self, modes: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "gm,bmkhw->bgkhw",
            self.vertical_basis.to(dtype=modes.dtype),
            modes,
        )

    def _warp_grouped_fields(
        self,
        fields: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
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
        warped = warp(
            grouped,
            flow.reshape(batch * 5, 2, height, width),
            periodic_longitude=self.spherical_ops,
            pole_parity=self.grouped_pole_parity,
        )
        warped_ordered = warped.reshape(
            batch,
            25,
            height,
            width,
        )[:, :channels]
        return warped_ordered.index_select(1, self.inverse_grouped_order)

    def _warp_fields(
        self,
        fields: torch.Tensor,
        flow: torch.Tensor,
        *,
        sampling_mode: str = "bilinear",
    ) -> torch.Tensor:
        if self.intrinsic_spherical_transport:
            if sampling_mode != "bilinear":
                raise ValueError(
                    "intrinsic spherical transport supports bilinear sampling"
                )
            geometry_basis = None
            if self.sphere_target.shape[-2:] == fields.shape[-2:]:
                geometry_basis = (
                    self.sphere_target,
                    self.sphere_east_basis,
                    self.sphere_north_basis,
                )
            return intrinsic_spherical_warp(
                fields,
                flow,
                channel_mean=self.normalization_mean,
                channel_std=self.normalization_std,
                geometry_basis=geometry_basis,
            )
        return warp(
            fields,
            flow,
            periodic_longitude=self.spherical_ops,
            pole_parity=self.field_pole_parity,
            sampling_mode=sampling_mode,
        )

    def set_normalization(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        """Bind physical channel units used by vector parallel transport."""
        if not self.intrinsic_spherical_transport:
            raise RuntimeError("normalization is only used by intrinsic transport")
        mean = torch.as_tensor(mean, dtype=self.normalization_mean.dtype)
        std = torch.as_tensor(std, dtype=self.normalization_std.dtype)
        if mean.numel() != self.out_channels or std.numel() != self.out_channels:
            raise ValueError("normalization must contain one value per field")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("normalization must be finite")
        if (std <= 0).any():
            raise ValueError("normalization standard deviations must be positive")
        self.normalization_mean.copy_(mean.reshape_as(self.normalization_mean))
        self.normalization_std.copy_(std.reshape_as(self.normalization_std))

    def _multiscale_field_correction(
        self,
        pyramid_features: list[torch.Tensor],
        tau: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Evaluate low/mid/high field experts at native decoder scales."""
        if self.multiscale_expert_heads is None:
            raise RuntimeError("multiscale field experts are not configured")
        if len(pyramid_features) != 3:
            raise ValueError("multiscale experts require three pyramid features")
        knots = None
        for features, expert in zip(
            pyramid_features,
            self.multiscale_expert_heads,
        ):
            batch, _, height, width = features.shape
            native = expert(features).reshape(
                batch,
                6,
                3,
                4,
                height,
                width,
            ).permute(0, 2, 1, 3, 4, 5).reshape(
                batch,
                3 * self.out_channels,
                height,
                width,
            )
            if native.shape[-2:] != output_size:
                native = (
                    _spherical_resize(
                        native,
                        output_size,
                        pole_parity=self.multiscale_knot_pole_parity,
                    )
                    if self.spherical_ops
                    else F.interpolate(
                        native,
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            knots = native if knots is None else knots + native
        if knots is None:
            raise RuntimeError("multiscale field experts produced no knots")
        return _evaluate_base_knot_correction(
            knots,
            tau,
        )

    def flow_matching_velocity(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau: torch.Tensor,
        bridge_state: torch.Tensor,
        flow_time: torch.Tensor,
        cond: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        """Evaluate the conditional rectified-flow velocity."""
        if not self.flow_matching:
            raise RuntimeError("flow_matching_velocity requires flow_matching=True")
        proposal, aux = self.forward(
            x0,
            xT,
            tau,
            cond,
            static,
            bridge_state=bridge_state,
            flow_time=flow_time,
            integrate_flow=False,
        )
        tau_b = tau.reshape(-1, 1, 1, 1)
        x_linear = (1.0 - tau_b) * x0 + tau_b * xT
        return proposal - x_linear, aux

    def forward(
        self,
        x0,
        xT,
        tau,
        cond=None,
        static=None,
        *,
        bridge_state: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
        integrate_flow: bool = True,
    ):
        B, _, H, W = x0.shape
        tau_b = tau.view(-1, 1, 1, 1) if tau.dim() == 1 else tau.view(B, 1, 1, 1)
        x_bilinear = (1.0 - tau_b) * x0 + tau_b * xT
        if self.flow_matching and bridge_state is None:
            if not integrate_flow:
                raise ValueError("bridge_state is required for velocity evaluation")
            state = x_bilinear
            aux: dict[str, torch.Tensor | None] = {}
            step_size = 1.0 / self.flow_matching_steps
            for step in range(self.flow_matching_steps):
                integration_time = x0.new_full(
                    (B,),
                    (step + 0.5) * step_size,
                )
                velocity, aux = self.flow_matching_velocity(
                    x0,
                    xT,
                    tau.view(-1),
                    state,
                    integration_time,
                    cond,
                    static,
                )
                state = state + step_size * velocity
            if self.endpoint_preserving:
                state = torch.where(tau_b == 0, x0, state)
                state = torch.where(tau_b == 1, xT, state)
            return state, {
                **aux,
                "flow_matching_state": state,
                "flow_matching_steps": x0.new_tensor(
                    self.flow_matching_steps
                ),
            }
        if bridge_state is not None and not self.flow_matching:
            raise ValueError("bridge_state requires flow_matching=True")
        if self.flow_matching:
            if bridge_state is None or bridge_state.shape != x0.shape:
                raise ValueError("bridge_state must match the anchor shape")
            if flow_time is None:
                raise ValueError("flow_time is required with bridge_state")

        trajectory_tau = (
            torch.full_like(tau.view(-1), 0.5)
            if self.query_independent_trajectory
            else tau.view(-1)
        )
        temb = self.time_mlp(trajectory_tau)
        lead_embedding = None
        if self.forecast_lead_mlp is not None:
            if cond is None:
                raise ValueError(
                    "forecast lead conditioning requires anchor lead hours"
                )
            forecast_lead = cond.reshape(-1).float()
            if forecast_lead.size(0) != B:
                raise ValueError("forecast lead batch does not match anchors")
            lead_embedding = self.forecast_lead_mlp(
                forecast_lead / self.forecast_lead_scale_hours
            )
            temb = temb + lead_embedding
        if self.flow_time_mlp is not None:
            flow_embedding = self.flow_time_mlp(flow_time.view(-1))
            if flow_embedding.size(0) != B and B % flow_embedding.size(0) == 0:
                flow_embedding = flow_embedding.repeat_interleave(
                    B // flow_embedding.size(0),
                    0,
                )
            temb = temb + flow_embedding
        if temb.size(0) != B and B % temb.size(0) == 0:
            temb = temb.repeat_interleave(B // temb.size(0), 0)

        st = self._prep_static(static, B, x0.device)
        parts = [x0, xT]
        if self.use_frame_difference:
            parts.append(xT - x0)
        if self.flow_matching:
            parts.append(bridge_state - x_bilinear)
        if st is not None:
            parts.append(st)
        h = torch.cat(parts, dim=1)

        feats = []
        for blk in self.encoder:
            h = blk(h); feats.append(h)
        h = self.adaln(h, temb)
        if self.global_mixer is not None:
            h = self.global_mixer(h, temb)
        bott = h if self.dual_stream else None   # global latent for decoder stream

        coarse_feat = None
        pyramid_features = []
        for i in range(self.n_levels):
            h = self.up[i](h)
            skip = feats[-(i + 2)]
            if h.shape[-2:] != skip.shape[-2:]:
                h = (
                    _spherical_resize(h, skip.shape[-2:])
                    if self.spherical_ops
                    else F.interpolate(
                        h,
                        size=skip.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            if self.use_skip:
                if self.gated_skip:
                    h = self.skip[i](h, skip, temb)
                else:
                    h = self.skip[i](torch.cat([h, skip], dim=1))
            h = self.dec[i](h)
            pyramid_features.append(h)
            if i == self.n_levels - 2:
                coarse_feat = h  # one level below full res -> coarse residual

        # upsample stem feature to full res if needed
        if h.shape[-2:] != (H, W):
            h = (
                _spherical_resize(h, (H, W))
                if self.spherical_ops
                else F.interpolate(
                    h,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
            )

        # --- flow warp (explicit transport) ---
        fa = torch.tanh(self.flow_head(h)) * self.flow_scale
        tf = tau_b                                                  # tau in [0,1]
        tr = 1.0 - tau_b                                            # reverse time
        if self.cubic_trajectory:
            flow_modes = fa.view(
                B,
                self.n_flow_modes,
                self.motion_components,
                H,
                W,
            )
            trajectory = (
                flow_modes
                if self.n_flow_modes == 1
                else self._group_trajectory(flow_modes)
            )
            forward_flow, backward_flow = _evaluate_endpoint_trajectory(
                trajectory,
                tau_b,
            )
            if self.n_flow_modes == 1:
                detail_forward_flow = forward_flow[:, 0]
                detail_backward_flow = backward_flow[:, 0]
                w0 = self._warp_fields(x0, detail_forward_flow)
                wT = self._warp_fields(xT, detail_backward_flow)
            else:
                w0 = self._warp_grouped_fields(x0, forward_flow)
                wT = self._warp_grouped_fields(xT, backward_flow)
        elif self.n_flow_modes > 1:
            flow_modes = fa.view(
                B,
                self.n_flow_modes,
                self.motion_components,
                H,
                W,
            )
            trajectory = self._group_trajectory(flow_modes)
            time = tf.unsqueeze(1)
            reverse_time = tr.unsqueeze(1)
            forward_flow = trajectory[:, :, :2] * time
            backward_flow = trajectory[:, :, 2:4] * reverse_time
            if self.use_accel and not self.ablate_acceleration:
                forward_flow = (
                    forward_flow
                    + 0.5 * trajectory[:, :, 4:6] * time * time
                )
                backward_flow = (
                    backward_flow
                    + 0.5
                    * trajectory[:, :, 6:8]
                    * reverse_time
                    * reverse_time
                )
            w0 = self._warp_grouped_fields(x0, forward_flow)
            wT = self._warp_grouped_fields(xT, backward_flow)
        elif self.use_accel and not self.ablate_acceleration:
            # constant-acceleration displacement: F*t + 1/2 A*t^2 (smooth in tau)
            detail_forward_flow = (
                fa[:, :2] * tf + 0.5 * fa[:, 4:6] * tf * tf
            )
            detail_backward_flow = (
                fa[:, 2:4] * tr + 0.5 * fa[:, 6:8] * tr * tr
            )
            w0 = self._warp_fields(x0, detail_forward_flow)
            wT = self._warp_fields(xT, detail_backward_flow)
        else:
            detail_forward_flow = fa[:, :2] * tf
            detail_backward_flow = fa[:, 2:4] * tr
            w0 = self._warp_fields(x0, detail_forward_flow)
            wT = self._warp_fields(xT, detail_backward_flow)
        flow = fa
        adaptive_controls = None
        if self.content_control_head is not None:
            control_size = (max(1, H // 4), max(1, W // 4))
            control_fields = (
                x0,
                xT,
                xT - x0,
                w0,
                wT,
                x_bilinear,
                w0 - x0,
                wT - xT,
            )
            reduced = [
                F.adaptive_avg_pool2d(field, control_size)
                for field in control_fields
            ]
            control_input = torch.stack(reduced, dim=2).reshape(
                B * self.out_channels,
                len(reduced),
                *control_size,
            )
            if self.time_content_adaptive_controls:
                control_time = temb.repeat_interleave(
                    self.out_channels,
                    dim=0,
                )
                adaptive_controls = self.content_control_head(
                    control_input,
                    control_time,
                )
            else:
                adaptive_controls = self.content_control_head(control_input)
            n_adaptive_controls = adaptive_controls.size(1)
            adaptive_controls = F.interpolate(
                adaptive_controls,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).view(
                B,
                self.out_channels,
                n_adaptive_controls,
                H,
                W,
            )

        # --- learned blend (fixes linear-chord error) ---
        blend_logits = self.blend_head(h)
        if adaptive_controls is not None:
            blend_logits = blend_logits + adaptive_controls[:, :, 0]
        if self.endpoint_preserving:
            alpha = _endpoint_blend(blend_logits, tau_b)
        else:
            alpha = torch.sigmoid(blend_logits + (1.0 - 2.0 * tau_b))
        warped = alpha * w0 + (1.0 - alpha) * wT
        # --- predicted local anchor error (degradation-aware branch) ---
        predicted_anchor_error = None
        anchor_error_excess = None
        if self.reliability_out is not None:
            if lead_embedding is None:
                raise ValueError(
                    "degradation-aware branch requires the lead embedding"
                )
            reliability_features = self.reliability_norm(
                self.reliability_stem(h)
            )
            film = self.reliability_film(lead_embedding).view(B, -1, 1, 1)
            film_scale, film_shift = film.chunk(2, dim=1)
            reliability_features = F.silu(
                reliability_features * (1.0 + film_scale) + film_shift
            )
            predicted_anchor_error = F.softplus(
                self.reliability_out(reliability_features)
            )
            # Centre on the typical anchor error so the gains swing both ways
            # and a mid-range forecast leaves the parent behaviour untouched.
            anchor_error_excess = (
                predicted_anchor_error - self.anchor_error_reference
            )
        # per-channel warp gate: blend warped frames with the un-warped bilinear
        # scaffold so static fields (t2m/mslp) can opt out of transport.
        beta_logits = self.warp_gate.view(1, -1, 1, 1)
        if adaptive_controls is not None:
            beta_logits = beta_logits + adaptive_controls[:, :, 1]
        if anchor_error_excess is not None:
            # Degraded anchors carry unreliable small-scale detail: transporting
            # them propagates forecast error into the interior estimate, so the
            # gate shifts toward the smooth scaffold exactly where the branch
            # predicts large anchor error.
            trust_excess = anchor_error_excess
            if beta_logits.size(1) == 1 and trust_excess.size(1) != 1:
                trust_excess = trust_excess.mean(dim=1, keepdim=True)
            beta_logits = beta_logits - (
                F.softplus(self.anchor_trust_gain).view(1, -1, 1, 1)
                * trust_excess
            )
        beta = torch.sigmoid(beta_logits)
        if self.ablate_transport:
            beta = torch.zeros_like(beta)
        warped = beta * warped + (1.0 - beta) * x_bilinear
        # --- pyramid residual ---
        delta = self.res_fine(self.res_body(h))
        if coarse_feat is not None:
            dc = self.res_coarse(coarse_feat)
            delta = delta + (
                _spherical_resize(
                    dc,
                    (H, W),
                    pole_parity=self.field_pole_parity,
                )
                if self.spherical_ops
                else F.interpolate(
                    dc,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        if self.spectral is not None:            # global low-wavenumber correction
            delta = delta + self.spectral(h).to(delta.dtype)
        adaptive_residual_gain = None
        if adaptive_controls is not None and adaptive_controls.size(2) >= 3:
            adaptive_residual_gain = 1.0 + 0.25 * torch.tanh(
                adaptive_controls[:, :, 2]
            )
            delta = delta * adaptive_residual_gain
        if anchor_error_excess is not None:
            # With unreliable anchors the interior state has to come from
            # learned structure rather than from copied anchor content, so the
            # residual is amplified exactly where the anchors are degraded.
            delta = delta * (
                1.0
                + torch.tanh(self.anchor_residual_gain).view(1, -1, 1, 1)
                * anchor_error_excess
            )
        s = torch.tanh(self.scale).view(1, -1, 1, 1)
        endpoint_factor = (
            4.0 * tau_b * (1.0 - tau_b)
            if self.endpoint_preserving
            else None
        )
        x_hat = (
            warped + endpoint_factor * s * delta
            if endpoint_factor is not None
            else warped + s * delta
        )
        if self.dual_stream:
            # global skip-less decoder stream (DC-AE-like) from the bottleneck
            g = bott
            for blk in self.gdec:
                g = blk(g)
            if g.shape[-2:] != (H, W):
                g = (
                    _spherical_resize(g, (H, W))
                    if self.spherical_ops
                    else F.interpolate(
                        g,
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            global_delta = self.gout(g)
            x_decoder = (
                x_bilinear + endpoint_factor * global_delta
                if endpoint_factor is not None
                else x_bilinear + global_delta
            )
            gate = torch.sigmoid(self.stream_gate).view(1, -1, 1, 1)
            x_hat = gate * x_hat + (1.0 - gate) * x_decoder
        endpoint_tangent = None
        if self.tangent_head is not None:
            endpoint_tangent = self.tangent_head(h)
            x_hat = x_hat + _evaluate_endpoint_tangent_correction(
                endpoint_tangent,
                tau_b,
            )
        base_knot_correction = None
        if self.base_knot_head is not None:
            base_knot_correction = _evaluate_base_knot_correction(
                self.base_knot_head(h),
                tau_b,
            )
            x_hat = x_hat.index_add(
                1,
                self.base_field_indices,
                base_knot_correction,
            )
        multiscale_field_correction = None
        if self.multiscale_expert_heads is not None:
            multiscale_field_correction = self._multiscale_field_correction(
                pyramid_features[-3:],
                tau_b,
                (H, W),
            )
            x_hat = x_hat + multiscale_field_correction
        if self.hydro is not None and not self.ablate_hydrostatic:
            # hydrostatic coupling: nudge Z1000-700 + mslp from the predicted
            # T column (thickness ~ layer-mean T). Zero-init so neutral at start.
            corr = self.hydro(x_hat[:, 0:4]).to(x_hat.dtype)
            if endpoint_factor is not None:
                corr = endpoint_factor * corr
            idx = torch.tensor([16, 17, 18, 19, 23], device=x_hat.device)
            x_hat = x_hat.index_add(1, idx, corr)              # dtype-safe channel add
        multiband_correction = None
        detail_bypass_correction = None
        if self.multiband_calibrator is not None:
            x_hat, multiband_correction = self.multiband_calibrator(
                x_hat,
                tau_b,
                temb,
                self.field_pole_parity,
            )
        if self.detail_gain_head is not None:
            detail0 = spherical_multiband_components(
                x0,
                self.field_pole_parity,
                dilations=(1,),
            )[0]
            detailT = spherical_multiband_components(
                xT,
                self.field_pole_parity,
                dilations=(1,),
            )[0]
            warped_detail0 = self._warp_fields(
                detail0,
                detail_forward_flow,
                sampling_mode="bicubic",
            )
            warped_detailT = self._warp_fields(
                detailT,
                detail_backward_flow,
                sampling_mode="bicubic",
            )
            transported_detail = (
                alpha * warped_detail0 + (1.0 - alpha) * warped_detailT
            )
            linear_detail = (1.0 - tau_b) * detail0 + tau_b * detailT
            anchor_detail = (
                beta * transported_detail + (1.0 - beta) * linear_detail
            )
            predicted_detail = spherical_multiband_components(
                x_hat,
                self.field_pole_parity,
                dilations=(1,),
            )[0]
            detail_gain_logits = self.detail_gain_head(temb).view(
                B,
                -1,
                1,
                1,
            )
            if adaptive_controls is not None and adaptive_controls.size(2) >= 4:
                detail_gain_logits = (
                    detail_gain_logits + adaptive_controls[:, :, 3]
                )
            detail_gain = 0.5 * torch.tanh(detail_gain_logits)
            detail_bypass_correction = detail_gain * (
                anchor_detail - predicted_detail
            )
            calibration_factor = 4.0 * tau_b * (1.0 - tau_b)
            x_hat = x_hat + calibration_factor * detail_bypass_correction
        hres_residual_correction = None
        if self.hres_residual_head is not None:
            hres_residual_correction = self.hres_residual_head(
                torch.cat(
                    (h, x_hat - x_bilinear, xT - x0),
                    dim=1,
                )
            )
            x_hat = x_hat + (
                4.0 * tau_b * (1.0 - tau_b) * hres_residual_correction
            )
        if self.endpoint_preserving:
            x_hat = torch.where(tau_b == 0, x0, x_hat)
            x_hat = torch.where(tau_b == 1, xT, x_hat)
        return x_hat, {
            "warped": warped,
            "delta": delta,
            "flow": flow,
            "alpha": alpha,
            "adaptive_controls": adaptive_controls,
            "adaptive_residual_gain": adaptive_residual_gain,
            "endpoint_tangent": endpoint_tangent,
            "base_knot_correction": base_knot_correction,
            "multiscale_field_correction": multiscale_field_correction,
            "multiband_correction": multiband_correction,
            "detail_bypass_correction": detail_bypass_correction,
            "hres_residual_correction": hres_residual_correction,
            "predicted_anchor_error": predicted_anchor_error,
        }


# Legacy import retained for checkpoints/scripts created before the paper
# nomenclature was finalised. New code should import ``WeatherBridgeModel``.
WeatherBridgeFlowModel = WeatherBridgeModel
