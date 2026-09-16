"""WeatherBridge — transport-aware VFI backbone for temporal
interpolation of reanalysis fields.

Motivation. The capacity-matched study showed the *backbone* barely moves
bulk RMSE at 6h — the bilinear scaffold carries the low-frequency signal and
all learned backbones cluster. The one thing every WeatherBridge variant so
far omits is **explicit transport**: weather advects, but ``bilinear +
additive residual`` never warps. Proper VFI (EMA-VFI / VFIMamba / ATM-VFI)
estimates motion and warps the anchor frames toward the query time. This
backbone adds that, cheaply, while staying simpler and faster than the
DC-AE reference.

Design (all learned work at low resolution — pixel-unshuffle-style stride
stem, so activations shrink ~4-16x and bs scales back up):

    input  = [x0, xT, xT - x0, static]          # frame-diff hands the net motion
    trunk  = strided circular-conv UNet + tau-AdaLN bottleneck
    heads (at stem resolution, upsampled to full res):
      * flow   F0, FT, A0, AT       -> constant-acceleration anchor warps
      * blend  alpha(tau, motion)   -> per-pixel/channel transport fusion
      * gate   beta                 -> transport vs linear-scaffold routing
      * resid  Delta                -> fine + coarse + optional SFNO correction
      * hydro  H(T)                 -> optional Z-column and MSLP correction
    x_tr  = alpha * warp(x0) + (1 - alpha) * warp(xT)
    x_mix = beta * x_tr + (1 - beta) * x_linear
    x_hat = x_mix + tanh(scale) * Delta + optional_hydro_correction

Gated skips. Decoder skips are fused through a **zero-init tau-gate**
(``GatedSkip``): near the anchors (tau=1,5) the gate stays ~0 so the warped
scaffold is trusted; mid-interval (tau=3) it opens for fine correction. This
is exactly the fix for the old ``WB-Skip`` overshoot (ungated high-res skips
injected uncontrolled HF). Set ``gated_skip=False`` / ``use_skip=False`` for
the ablation arms.

forward(x0, xT, tau, cond=None, static=None) -> (x_hat, aux_dict)
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
    def __init__(self, dim: int = 256, freq_dim: int = 8, base_period: float = 16.0):
        super().__init__()
        self.emb = SinusoidalPosEmb(freq_dim, base_period)
        self.net = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(),
                                  nn.Linear(dim, dim))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
) -> torch.Tensor:
    """Backward-warp ``x`` by pixel-space flow.

    The retained checkpoints use this equirectangular grid-sample operator in
    both axes. The opt-in spherical branch wraps longitude and maps samples
    crossing either pole to the antipodal reflected latitude.
    """
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
            mode="bilinear",
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
            mode="bilinear",
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
        source_latitude = torch.asin(source[:, 2].clamp(-1.0, 1.0))
        source_longitude = torch.atan2(source[:, 1], source[:, 0])
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
                1.0 + (source * target).sum(dim=1)
            ).clamp_min(1.0e-6)
            path_sum = source + target
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
    neutral. Falls back to identity-zero if torch_harmonics is unavailable."""

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
        # Constant-acceleration transport: warp by F*t + 1/2 A*t^2 instead of F*t.
        # The tau-dependence of the warp is then a POLYNOMIAL (analytically smooth)
        # so it extrapolates cleanly to held-out tau, and the quadratic term
        # captures curved parcel trajectories (rotating fronts, diurnal drift)
        # that a linear flow forces onto the tau-conditioned residual.
        self.use_accel = bool(use_accel)
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

        self.time_mlp = TimeMLP(time_emb_dim)
        ch = [hidden * (2 ** i) for i in range(n_levels + 1)]

        # input: [x0, xT, xT-x0, static]
        enc_in = 3 * in_channels + n_static_features
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
                            field_pole_parity,
                            field_pole_parity,
                            field_pole_parity,
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
        self.global_mixer = (
            BottleneckTokenMixer(
                ch[-1],
                n_tokens=global_tokens,
                token_dim=global_token_dim,
            )
            if global_tokens > 0
            else None
        )

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
            self.dec.append(
                conv_block(
                    ch[-(i + 2)],
                    ch[-(i + 2)],
                    spherical_ops=self.spherical_ops,
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
        self.blend_head = _conv3x3(
            base,
            out_channels,
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
        self.scale = nn.Parameter(torch.full((out_channels,), float(residual_scale_init)))
        # Per-channel warp gate: sigmoid(warp_gate)=1 -> trust the flow-warped
        # frames (moving fields: wind, moisture); =0 -> fall back to the plain
        # bilinear scaffold (quasi-static fields: t2m/mslp locked to orography
        # where warping only distorts). Init +2 (~0.88) so training starts near
        # the warp-only behaviour and only pulls static channels down.
        self.warp_gate = nn.Parameter(torch.full((out_channels,), 2.0))
        # Mass-aware init: for quasi-static / mass channels (Z1000-700, t2m,
        # mslp in the 24ch order) start warp_gate at -2 (sigmoid ~0.12) so they
        # ride the smooth bilinear scaffold from step 0 instead of the warped
        # frames. Warping a planetary-scale pressure/geopotential field only
        # injects spurious small-scale structure; these fields are the decoder's
        # (residual's) job, not transport's.
        if mass_aware_gate:
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
    ) -> torch.Tensor:
        if self.intrinsic_spherical_transport:
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

    def forward(self, x0, xT, tau, cond=None, static=None):
        B, _, H, W = x0.shape
        tau_b = tau.view(-1, 1, 1, 1) if tau.dim() == 1 else tau.view(B, 1, 1, 1)
        trajectory_tau = (
            torch.full_like(tau.view(-1), 0.5)
            if self.query_independent_trajectory
            else tau.view(-1)
        )
        temb = self.time_mlp(trajectory_tau)
        if temb.size(0) != B and B % temb.size(0) == 0:
            temb = temb.repeat_interleave(B // temb.size(0), 0)

        st = self._prep_static(static, B, x0.device)
        parts = [x0, xT, xT - x0]
        if st is not None:
            parts.append(st)
        h = torch.cat(parts, dim=1)

        feats = []
        for blk in self.encoder:
            h = blk(h); feats.append(h)
        h = self.adaln(h, temb)
        if self.global_mixer is not None:
            h = self.global_mixer(h)
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
                w0 = self._warp_fields(x0, forward_flow[:, 0])
                wT = self._warp_fields(xT, backward_flow[:, 0])
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
            if self.use_accel:
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
        elif self.use_accel:
            # constant-acceleration displacement: F*t + 1/2 A*t^2 (smooth in tau)
            w0 = self._warp_fields(
                x0,
                fa[:, :2] * tf + 0.5 * fa[:, 4:6] * tf * tf,
            )
            wT = self._warp_fields(
                xT,
                fa[:, 2:4] * tr + 0.5 * fa[:, 6:8] * tr * tr,
            )
        else:
            w0 = self._warp_fields(x0, fa[:, :2] * tf)
            wT = self._warp_fields(xT, fa[:, 2:4] * tr)
        flow = fa
        # --- learned blend (fixes linear-chord error) ---
        blend_logits = self.blend_head(h)
        if self.endpoint_preserving:
            alpha = _endpoint_blend(blend_logits, tau_b)
        else:
            alpha = torch.sigmoid(blend_logits + (1.0 - 2.0 * tau_b))
        warped = alpha * w0 + (1.0 - alpha) * wT
        # per-channel warp gate: blend warped frames with the un-warped bilinear
        # scaffold so static fields (t2m/mslp) can opt out of transport.
        x_bilinear = (1.0 - tau_b) * x0 + tau_b * xT
        beta = torch.sigmoid(self.warp_gate).view(1, -1, 1, 1)
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
        if self.hydro is not None:
            # hydrostatic coupling: nudge Z1000-700 + mslp from the predicted
            # T column (thickness ~ layer-mean T). Zero-init so neutral at start.
            corr = self.hydro(x_hat[:, 0:4]).to(x_hat.dtype)
            if endpoint_factor is not None:
                corr = endpoint_factor * corr
            idx = torch.tensor([16, 17, 18, 19, 23], device=x_hat.device)
            x_hat = x_hat.index_add(1, idx, corr)              # dtype-safe channel add
        if self.endpoint_preserving:
            x_hat = torch.where(tau_b == 0, x0, x_hat)
            x_hat = torch.where(tau_b == 1, xT, x_hat)
        return x_hat, {
            "warped": warped,
            "delta": delta,
            "flow": flow,
            "alpha": alpha,
            "endpoint_tangent": endpoint_tangent,
            "base_knot_correction": base_knot_correction,
            "multiscale_field_correction": multiscale_field_correction,
        }


# Legacy import retained for checkpoints/scripts created before the paper
# nomenclature was finalised. New code should import ``WeatherBridgeModel``.
WeatherBridgeFlowModel = WeatherBridgeModel
