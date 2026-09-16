"""Local-global wavelet operator for endpoint-preserving weather interpolation.

The model keeps the anchor fields in an orthonormal three-level Haar pyramid.
It predicts corrections to the linearly interpolated coefficients rather than
reconstructing fine scales from a latent bottleneck. The coarsest 45x90 branch
mixes local spherical convolutions with a spherical-harmonic operator; the two
finer branches refine only the wavelet details at 90x180 and 180x360.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from weather_time_interp.model.weatherbridge_flow_model import (
    SphereConv2d,
    SphericalConv,
    TimeMLP,
    _sphere_pad,
    _spherical_resize,
)


def haar_dwt2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one orthonormal 2D Haar analysis step.

    Details are concatenated as longitude-high, latitude-high, and diagonal
    bands. Both spatial dimensions must be even.
    """
    if x.size(-2) % 2 or x.size(-1) % 2:
        raise ValueError(
            f"Haar analysis requires even dimensions, got {x.shape[-2:]}"
        )
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    low = (a + b + c + d) * 0.5
    lon_high = (a - b + c - d) * 0.5
    lat_high = (a + b - c - d) * 0.5
    diagonal = (a - b - c + d) * 0.5
    return low, torch.cat((lon_high, lat_high, diagonal), dim=1)


def haar_iwt2(low: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    """Invert one orthonormal 2D Haar analysis step."""
    if detail.size(1) != 3 * low.size(1):
        raise ValueError(
            "Haar synthesis expects three detail bands per low-pass channel"
        )
    lon_high, lat_high, diagonal = detail.chunk(3, dim=1)
    a = (low + lon_high + lat_high + diagonal) * 0.5
    b = (low - lon_high + lat_high - diagonal) * 0.5
    c = (low + lon_high - lat_high - diagonal) * 0.5
    d = (low - lon_high - lat_high + diagonal) * 0.5
    output = low.new_empty(
        low.size(0),
        low.size(1),
        low.size(2) * 2,
        low.size(3) * 2,
    )
    output[..., 0::2, 0::2] = a
    output[..., 0::2, 1::2] = b
    output[..., 1::2, 0::2] = c
    output[..., 1::2, 1::2] = d
    return output


def haar_pyramid(
    x: torch.Tensor,
    levels: int = 3,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return the coarsest low-pass and details ordered fine to coarse."""
    low = x
    details: list[torch.Tensor] = []
    for _ in range(levels):
        low, detail = haar_dwt2(low)
        details.append(detail)
    return low, details


def haar_reconstruct(
    low: torch.Tensor,
    details: list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Reconstruct a pyramid whose details are ordered fine to coarse."""
    output = low
    for detail in reversed(details):
        output = haar_iwt2(output, detail)
    return output


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DepthwiseSphereConv2d(nn.Conv2d):
    """Depthwise convolution with spherical latitude/longitude padding."""

    def __init__(self, channels: int, kernel_size: int = 3):
        if kernel_size % 2 != 1:
            raise ValueError("kernel size must be odd")
        super().__init__(
            channels,
            channels,
            kernel_size,
            groups=channels,
            padding=0,
        )
        self.sphere_padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(_sphere_pad(x, self.sphere_padding))


class PairBandStem(nn.Module):
    """Encode the two ordered anchor coefficient bands with shared weights."""

    def __init__(
        self,
        in_channels: int,
        endpoint_width: int,
        out_channels: int,
        pole_parity: torch.Tensor,
    ):
        super().__init__()
        self.endpoint = nn.Sequential(
            SphereConv2d(
                in_channels,
                endpoint_width,
                3,
                pole_parity=pole_parity,
            ),
            nn.GroupNorm(_norm_groups(endpoint_width), endpoint_width),
            nn.SiLU(),
        )
        self.fuse = nn.Conv2d(2 * endpoint_width, out_channels, 1)

    def forward(
        self,
        band0: torch.Tensor,
        band1: torch.Tensor,
    ) -> torch.Tensor:
        return self.fuse(
            torch.cat((self.endpoint(band0), self.endpoint(band1)), dim=1)
        )


class StaticStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            SphereConv2d(in_channels, out_channels, 3),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        return self.net(static)


class LocalGlobalBlock(nn.Module):
    """Factorized local operator with an optional low-mode spherical branch."""

    def __init__(
        self,
        channels: int,
        time_dim: int,
        *,
        nlat: int,
        nlon: int,
        global_width: int = 0,
        global_modes: int = 0,
        expansion: int = 4,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        inner = expansion * channels
        self.norm = nn.GroupNorm(_norm_groups(channels), channels)
        self.local = nn.Sequential(
            nn.Conv2d(channels, inner, 1),
            nn.SiLU(),
            DepthwiseSphereConv2d(inner, 3),
            nn.SiLU(),
            nn.Conv2d(inner, channels, 1),
        )
        self.time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 3 * channels),
        )
        nn.init.zeros_(self.time[-1].weight)
        nn.init.zeros_(self.time[-1].bias)
        self.residual_scale = float(residual_scale)

        if global_width > 0 and global_modes > 0:
            self.global_in = nn.Conv2d(channels, global_width, 1)
            self.global_operator = SphericalConv(
                global_width,
                global_width,
                nlat=nlat,
                nlon=nlon,
                lmax=global_modes,
                mmax=global_modes,
            )
            self.global_out = nn.Conv2d(global_width, channels, 1)
        else:
            self.global_in = None
            self.global_operator = None
            self.global_out = None

    def _global(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_operator is None:
            return torch.zeros_like(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            hidden = self.global_in(x.float())
            hidden = F.silu(self.global_operator(hidden) + hidden)
            output = self.global_out(hidden)
        return output.to(dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm(x)
        update = self.local(normalized) + self._global(normalized)
        scale, shift, gate = self.time(time_embedding).chunk(3, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        gate = 2.0 * torch.sigmoid(gate[:, :, None, None])
        update = gate * ((1.0 + scale) * update + shift)
        return x + self.residual_scale * update


class LocalGlobalWaveletBridgeModel(nn.Module):
    """A roughly 10M-parameter deterministic wavelet interpolation operator."""

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 24,
        n_static_features: int = 3,
        widths: tuple[int, int, int] = (336, 200, 120),
        endpoint_widths: tuple[int, int, int] = (96, 72, 56),
        band_widths: tuple[int, int, int] = (160, 96, 80),
        static_widths: tuple[int, int, int] = (32, 24, 16),
        blocks: tuple[int, int, int] = (4, 2, 2),
        time_dim: int = 256,
        global_width: int = 36,
        global_modes: int = 20,
        base_grid: tuple[int, int] = (360, 720),
    ):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("wavelet coefficient scaffold requires matching I/O")
        if len(widths) != 3 or len(blocks) != 3:
            raise ValueError("the canonical bridge uses exactly three scales")
        if base_grid[0] % 8 or base_grid[1] % 8:
            raise ValueError("base_grid must be divisible by eight")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_static_features = n_static_features
        self.base_grid = tuple(int(value) for value in base_grid)

        field_pole_parity = torch.ones(out_channels)
        for index in (*range(4, 12), 21, 22):
            if index < out_channels:
                field_pole_parity[index] = -1.0
        detail_pole_parity = torch.cat(
            (
                field_pole_parity,
                -field_pole_parity,
                -field_pole_parity,
            )
        )
        self.register_buffer(
            "field_pole_parity",
            field_pole_parity,
            persistent=False,
        )
        self.register_buffer(
            "detail_pole_parity",
            detail_pole_parity,
            persistent=False,
        )

        coarse_h = self.base_grid[0] // 8
        coarse_w = self.base_grid[1] // 8
        self.time_mlp = TimeMLP(time_dim)
        self.low_stem = PairBandStem(
            in_channels,
            endpoint_widths[0],
            band_widths[0],
            field_pole_parity,
        )
        self.detail_stems = nn.ModuleList(
            (
                PairBandStem(
                    3 * in_channels,
                    endpoint_widths[0],
                    band_widths[0],
                    detail_pole_parity,
                ),
                PairBandStem(
                    3 * in_channels,
                    endpoint_widths[1],
                    band_widths[1],
                    detail_pole_parity,
                ),
                PairBandStem(
                    3 * in_channels,
                    endpoint_widths[2],
                    band_widths[2],
                    detail_pole_parity,
                ),
            )
        )
        self.static_stems = nn.ModuleList(
            StaticStem(n_static_features, width) for width in static_widths
        )

        self.coarse_fuse = nn.Conv2d(
            2 * band_widths[0] + static_widths[0],
            widths[0],
            1,
        )
        self.coarse_blocks = nn.ModuleList(
            LocalGlobalBlock(
                widths[0],
                time_dim,
                nlat=coarse_h,
                nlon=coarse_w,
                global_width=global_width,
                global_modes=global_modes,
                residual_scale=1.0 / math.sqrt(blocks[0]),
            )
            for _ in range(blocks[0])
        )

        self.coarse_to_mid = SphereConv2d(widths[0], widths[1], 3)
        self.mid_fuse = nn.Conv2d(
            widths[1] + band_widths[1] + static_widths[1],
            widths[1],
            1,
        )
        self.mid_blocks = nn.ModuleList(
            LocalGlobalBlock(
                widths[1],
                time_dim,
                nlat=coarse_h * 2,
                nlon=coarse_w * 2,
                residual_scale=1.0 / math.sqrt(blocks[1]),
            )
            for _ in range(blocks[1])
        )

        self.mid_to_fine = SphereConv2d(widths[1], widths[2], 3)
        self.fine_fuse = nn.Conv2d(
            widths[2] + band_widths[2] + static_widths[2],
            widths[2],
            1,
        )
        self.fine_blocks = nn.ModuleList(
            LocalGlobalBlock(
                widths[2],
                time_dim,
                nlat=coarse_h * 4,
                nlon=coarse_w * 4,
                residual_scale=1.0 / math.sqrt(blocks[2]),
            )
            for _ in range(blocks[2])
        )

        self.low_head = nn.Conv2d(widths[0], out_channels, 1)
        self.detail_heads = nn.ModuleList(
            (
                nn.Conv2d(widths[0], 3 * out_channels, 1),
                nn.Conv2d(widths[1], 3 * out_channels, 1),
                nn.Conv2d(widths[2], 3 * out_channels, 1),
            )
        )
        for head in (self.low_head, *self.detail_heads):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _static_pyramid(
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fine = F.avg_pool2d(static, 2, 2)
        mid = F.avg_pool2d(fine, 2, 2)
        coarse = F.avg_pool2d(mid, 2, 2)
        return coarse, mid, fine

    @staticmethod
    def _linear_pair(
        value0: torch.Tensor,
        value1: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        return (1.0 - tau) * value0 + tau * value1

    def forward(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau: torch.Tensor,
        cond: torch.Tensor | None = None,
        *,
        static: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del cond
        if x0.shape != xT.shape:
            raise ValueError("anchor tensors must have identical shapes")
        if x0.size(1) != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} fields, got {x0.size(1)}"
            )
        if tuple(x0.shape[-2:]) != self.base_grid:
            raise ValueError(
                f"expected grid {self.base_grid}, got {tuple(x0.shape[-2:])}"
            )
        batch = x0.size(0)
        tau = tau.reshape(batch).to(device=x0.device, dtype=x0.dtype)
        tau4 = tau[:, None, None, None]
        if static is None:
            static = x0.new_zeros(
                batch,
                self.n_static_features,
                *self.base_grid,
            )
        elif static.shape != (
            batch,
            self.n_static_features,
            *self.base_grid,
        ):
            raise ValueError(
                f"invalid static shape {tuple(static.shape)} for batch {batch}"
            )

        low0, details0 = haar_pyramid(x0)
        lowT, detailsT = haar_pyramid(xT)
        static_coarse, static_mid, static_fine = self._static_pyramid(static)
        time_embedding = self.time_mlp(tau.float())

        low_features = self.low_stem(low0, lowT)
        coarse_detail_features = self.detail_stems[0](
            details0[2],
            detailsT[2],
        )
        coarse = self.coarse_fuse(
            torch.cat(
                (
                    low_features,
                    coarse_detail_features,
                    self.static_stems[0](static_coarse),
                ),
                dim=1,
            )
        )
        for block in self.coarse_blocks:
            coarse = block(coarse, time_embedding)

        mid = _spherical_resize(
            coarse,
            details0[1].shape[-2:],
        )
        mid = self.coarse_to_mid(mid)
        mid = self.mid_fuse(
            torch.cat(
                (
                    mid,
                    self.detail_stems[1](details0[1], detailsT[1]),
                    self.static_stems[1](static_mid),
                ),
                dim=1,
            )
        )
        for block in self.mid_blocks:
            mid = block(mid, time_embedding)

        fine = _spherical_resize(
            mid,
            details0[0].shape[-2:],
        )
        fine = self.mid_to_fine(fine)
        fine = self.fine_fuse(
            torch.cat(
                (
                    fine,
                    self.detail_stems[2](details0[0], detailsT[0]),
                    self.static_stems[2](static_fine),
                ),
                dim=1,
            )
        )
        for block in self.fine_blocks:
            fine = block(fine, time_embedding)

        envelope = 4.0 * tau4 * (1.0 - tau4)
        low_delta = self.low_head(coarse)
        low = self._linear_pair(low0, lowT, tau4)
        low = low + envelope * low_delta
        details = [
            self._linear_pair(details0[0], detailsT[0], tau4)
            + envelope * self.detail_heads[2](fine),
            self._linear_pair(details0[1], detailsT[1], tau4)
            + envelope * self.detail_heads[1](mid),
            self._linear_pair(details0[2], detailsT[2], tau4)
            + envelope * self.detail_heads[0](coarse),
        ]
        output = haar_reconstruct(low, details)
        output = torch.where(tau4 <= 0.0, x0, output)
        output = torch.where(tau4 >= 1.0, xT, output)
        return output, {
            "wavelet_envelope": envelope,
            "coarse_low_delta": low_delta,
        }
