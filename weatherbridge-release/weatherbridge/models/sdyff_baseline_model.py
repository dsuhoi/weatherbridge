"""Spherical-DYffusion-style SFNO interpolator (Stage 1 only) for time interpolation.

Based on Cachay et al. 2024 "Probabilistic Emulation of a Global Climate Model
with Spherical DYffusion" (arXiv:2406.14798). Key architectural choices:

- Modified SFNO blocks with InstanceNorm → AdaIN scale-shift (FiLM) → SHT spectral
  filter → MLP.
- Time embedding: sin/cos at 32 frequencies (period 16) → 2-layer MLP → 128-d.
- AdaIN per block (every block has independent FiLM produced from same time emb).
- Dropout + DropPath kept ON at inference (MC ensembling).

Wrapped into the same residual-to-bilinear pipeline as our other baselines:
    x_hat = bilinear(x0, xT, tau) + tanh(scale) * decoder_out
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_harmonics as th


# ============================================================
# Time embedding (DDPM-style sinusoidal + MLP)
# ============================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for τ ∈ [0, 1]."""

    def __init__(self, dim: int, base_period: float = 16.0):
        super().__init__()
        self.dim = dim
        self.base_period = base_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B,) or (B, 1)
        t = t.view(-1)
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(self.base_period) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)


class TimeMLP(nn.Module):
    def __init__(self, time_dim: int = 128, freq_dim: int = 64, base_period: float = 16.0):
        super().__init__()
        self.embed = SinusoidalPosEmb(freq_dim, base_period=base_period)
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(t))  # (B, time_dim)


# ============================================================
# Spherical convolution block (SHT-based)
# ============================================================

class SphericalConv2d(nn.Module):
    """Spherical convolution via SHT: x → SHT → linear in spectral domain → ISHT.

    Uses torch_harmonics.RealSHT / RealISHT. Linear filter is a learned complex
    weight in spectral domain of shape (out_ch, in_ch).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nlat: int,
        nlon: int,
        n_modes_lat: int = 32,
        n_modes_lon: int = 64,
        grid: str = "equiangular",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_lat = n_modes_lat
        self.n_modes_lon = n_modes_lon

        self.sht = th.RealSHT(nlat, nlon, lmax=n_modes_lat, mmax=n_modes_lon, grid=grid).float()
        self.isht = th.InverseRealSHT(nlat, nlon, lmax=n_modes_lat, mmax=n_modes_lon, grid=grid).float()

        # Spectral weights: complex-valued. Real and imaginary parts as separate parameters.
        scale = 1.0 / math.sqrt(in_channels)
        self.weight = nn.Parameter(
            torch.randn(2, out_channels, in_channels, n_modes_lat, n_modes_lon) * scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W) at sphere grid.
        # cuFFT (used inside torch_harmonics.RealSHT) refuses fp16/bf16 when the
        # signal size isn't a power of 2 (720 isn't). Disable autocast and cast
        # to fp32 around the SHT; cast back to the input dtype on exit.
        in_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            x_f = x.float()
            x_sht = self.sht(x_f)
            w = torch.complex(self.weight[0].float(), self.weight[1].float())
            y = torch.einsum("bilm,oilm->bolm", x_sht, w)
            out = self.isht(y)
        return out.to(in_dtype)


# ============================================================
# DYffusion-style SFNO block with AdaIN time conditioning
# ============================================================

class SDyffBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        time_dim: int,
        nlat: int,
        nlon: int,
        n_modes_lat: int = 32,
        n_modes_lon: int = 64,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(embed_dim, affine=True)
        # AdaIN: time emb → scale & shift (per-channel)
        self.time_proj = nn.Linear(time_dim, 2 * embed_dim)
        self.spectral = SphericalConv2d(
            embed_dim, embed_dim, nlat=nlat, nlon=nlon,
            n_modes_lat=n_modes_lat, n_modes_lon=n_modes_lon,
        )
        self.activation = nn.GELU()

        self.norm2 = nn.InstanceNorm2d(embed_dim, affine=True)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, embed_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

        # DropPath (stochastic depth)
        self.drop_path_rate = drop_path

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        # Apply per-sample stochastic depth always (training and inference, like S-DYff).
        if self.drop_path_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor_() / keep_prob

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), t_emb: (B, time_dim)
        # 1. InstanceNorm → FiLM (scale, shift) → spectral filter
        x_norm = self.norm1(x)
        scale_shift = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)  # (B, 2C, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        x_norm = x_norm * (1 + scale) + shift
        x_spec = self.spectral(x_norm)
        x_spec = self.activation(x_spec)
        x = x + self._drop_path(self.dropout(x_spec))

        # 2. InstanceNorm → MLP
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = x + self._drop_path(x_mlp)
        return x


# ============================================================
# Full S-DYff Stage 1 SFNO interpolator wrapper
# ============================================================

class WeatherSDyffusionResidualLinearModel(nn.Module):
    """Spherical DYffusion Stage 1 (interpolator only) + bilinear-residual pipeline.

    Default config matches our scale: 6 blocks at embed_dim=128, ~30M params.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 5,
        embed_dim: int = 64,
        num_layers: int = 4,
        time_dim: int = 128,
        # n_modes determines spectral weight size (out_C × in_C × lat × lon).
        # For 181x360 grid with embed_dim=64 + 4 blocks: ~16M params with these modes.
        n_modes_lat: int = 16,
        n_modes_lon: int = 32,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        # Spatial input is 181x360. Crop to 180 for SHT to use even nlat.
        nlat: int = 180,
        nlon: int = 360,
        lat_crop: int = 1,
        residual_scale_init: float = 0.10,
        residual_scale_learnable: bool = True,
        residual_clip: float | None = None,
        residual_scale_floor: float = 0.0,
    ):
        super().__init__()
        self.lat_crop = max(0, int(lat_crop))
        self.residual_clip = residual_clip
        self.residual_scale_floor = float(residual_scale_floor)
        self.nlat = nlat
        self.nlon = nlon

        # Input is cat([x0, xT], dim=1) — τ goes through time conditioning, not channel.
        self.conv_in = nn.Conv2d(in_channels * 2, embed_dim, kernel_size=1)
        self.time_mlp = TimeMLP(time_dim=time_dim, freq_dim=64, base_period=16.0)

        self.blocks = nn.ModuleList([
            SDyffBlock(
                embed_dim=embed_dim, time_dim=time_dim,
                nlat=nlat, nlon=nlon,
                n_modes_lat=n_modes_lat, n_modes_lon=n_modes_lon,
                mlp_ratio=mlp_ratio, dropout=dropout, drop_path=drop_path,
            )
            for _ in range(num_layers)
        ])

        self.norm_out = nn.InstanceNorm2d(embed_dim, affine=True)
        self.conv_out = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

        init_scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if residual_scale_learnable:
            self.residual_scale = nn.Parameter(init_scale)
        else:
            self.register_buffer("residual_scale", init_scale)

    def _crop_lat(self, x: torch.Tensor) -> torch.Tensor:
        # Crop 181 → 180 for even SHT input.
        if self.lat_crop <= 0:
            return x
        h = x.size(-2)
        if h <= self.lat_crop:
            raise ValueError(f"lat_crop={self.lat_crop} too large for height={h}")
        crop_top = self.lat_crop // 2
        crop_bottom = self.lat_crop - crop_top
        if crop_bottom == 0:
            return x[:, :, crop_top:]
        return x[:, :, crop_top:-crop_bottom]

    def forward(self, x0, xT, tau, cond):
        if tau.dim() == 1:
            tau_ = tau.view(-1, 1, 1, 1)
        elif tau.dim() == 2:
            tau_ = tau.view(tau.size(0), 1, 1, 1)
        else:
            tau_ = tau

        # Concatenate endpoints along channel; τ via time embedding.
        x_in = torch.cat([x0, xT], dim=1)
        x_cropped = self._crop_lat(x_in)
        h = self.conv_in(x_cropped)

        # Time embedding from τ ∈ [0, 1]
        t_emb = self.time_mlp(tau.view(-1))

        for block in self.blocks:
            h = block(h, t_emb)

        h = self.norm_out(h)
        decoder_out_raw = self.conv_out(h)
        # Restore full (181, 360)
        decoder_out = F.interpolate(
            decoder_out_raw, size=x0.shape[-2:], mode="bilinear", align_corners=False
        )

        if self.residual_clip is not None and self.residual_clip > 0:
            decoder_out = torch.clamp(decoder_out, -self.residual_clip, self.residual_clip)

        residual_scale = torch.tanh(self.residual_scale)
        if self.residual_scale_floor > 0.0:
            sign = torch.where(
                residual_scale >= 0,
                torch.ones_like(residual_scale),
                -torch.ones_like(residual_scale),
            )
            magnitude = residual_scale.abs().clamp(min=self.residual_scale_floor)
            residual_scale = sign * magnitude

        x_bilinear = (1.0 - tau_) * x0 + tau_ * xT
        x_hat = x_bilinear + residual_scale * decoder_out

        return x_hat, {
            "z_tau": None,
            "d0": None,
            "d1": None,
            "decoder_out": decoder_out,
            "x_bilinear": x_bilinear,
            "residual_scale": residual_scale.detach(),
        }
