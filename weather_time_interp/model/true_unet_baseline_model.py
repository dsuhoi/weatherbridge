"""True U-Net (Ronneberger 2015 style) with multi-resolution skip connections,
adapted for global ERA5 fields (sphere-conv + circular padding for longitude).

Key difference from `unet_baseline_model.py`: that model is encoder→bottleneck→decoder
WITHOUT skip connections (encoder-AE shape). This is a real U-Net: each decoder
stage receives the encoder output of the same resolution via skip-concat.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SphereConv2d(nn.Module):
    """Conv2d with circular padding on longitude (W) and zero padding on latitude (H)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad: circular on W (longitude), zero on H (latitude).
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.pad, self.pad), mode="constant", value=0)
        return self.conv(x)


class DoubleConv(nn.Module):
    """Two SphereConv blocks with GroupNorm + GELU. Standard U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        groups = min(groups, out_ch)
        while out_ch % groups != 0 and groups > 1:
            groups -= 1
        self.block = nn.Sequential(
            SphereConv2d(in_ch, out_ch, kernel_size=3),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            SphereConv2d(out_ch, out_ch, kernel_size=3),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downsample by 2 + DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Bilinear upsample + DoubleConv with skip connection from encoder."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims mismatch (e.g. odd sizes after rounds of /2 → ×2).
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class WeatherTrueUNetResidualLinearModel(nn.Module):
    """True U-Net with skip connections + bilinear-residual pipeline.

    Architecture:
      conv_in (5*2+1 → 64)
      down1 (64 → 128, /2)
      down2 (128 → 256, /2)
      down3 (256 → 512, /2)
      bottleneck DoubleConv at 512
      up1 (512 + skip 256 → 256, ×2)
      up2 (256 + skip 128 → 128, ×2)
      up3 (128 + skip 64 → 64, ×2)
      conv_out (64 → 5)

    For 181×360 input, downsampling needs even sizes. Use lat_crop=5 → 176×360 → 88×180 → 44×90 → 22×45.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 5,
        base_channels: int = 64,
        num_levels: int = 3,
        n_static_features: int = 0,
        lat_crop: int = 5,  # 181 → 176 = 16*11
        residual_scale_init: float = 0.10,
        residual_scale_learnable: bool = True,
        residual_clip: float | None = None,
        residual_scale_floor: float = 0.0,
        direct_prediction: bool = False,
    ):
        super().__init__()
        self.lat_crop = max(0, int(lat_crop))
        self.residual_clip = residual_clip
        self.residual_scale_floor = float(residual_scale_floor)
        self.direct_prediction = bool(direct_prediction)
        self.n_static_features = int(n_static_features)

        # Input: cat([x0, xT, tau_map, static])
        c_in = in_channels * 2 + 1 + self.n_static_features
        self.conv_in = DoubleConv(c_in, base_channels)

        self.downs = nn.ModuleList()
        ch = base_channels
        for i in range(num_levels):
            self.downs.append(Down(ch, ch * 2))
            ch *= 2

        self.bottleneck = DoubleConv(ch, ch)

        # Decoder with skip connections.
        # We need to remember encoder skip channels in reverse.
        self.ups = nn.ModuleList()
        skip_channels = [base_channels * (2 ** i) for i in range(num_levels)]  # [64, 128, 256]
        for i in reversed(range(num_levels)):
            in_c = ch
            skip_c = skip_channels[i]
            out_c = skip_c
            self.ups.append(Up(in_c, skip_c, out_c))
            ch = out_c

        self.conv_out = SphereConv2d(base_channels, out_channels, kernel_size=1)

        init_scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if residual_scale_learnable and not self.direct_prediction:
            self.residual_scale = nn.Parameter(init_scale)
        else:
            self.register_buffer("residual_scale", init_scale)

    def _crop_lat(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x0, xT, tau, cond, static=None):
        if tau.dim() == 1:
            tau_ = tau.view(-1, 1, 1, 1)
        elif tau.dim() == 2:
            tau_ = tau.view(tau.size(0), 1, 1, 1)
        else:
            tau_ = tau

        tau_map = tau_.expand(-1, 1, x0.size(-2), x0.size(-1))
        parts = [x0, xT, tau_map]
        if self.n_static_features > 0:
            if static is None:
                raise ValueError(f"Model requires static (n_static_features={self.n_static_features}), got None")
            if static.dim() == 3:
                static = static.unsqueeze(0).expand(x0.size(0), -1, -1, -1)
            elif static.size(0) != x0.size(0):
                if x0.size(0) % static.size(0) == 0:
                    n_tau = x0.size(0) // static.size(0)
                    static = static.repeat_interleave(n_tau, dim=0)
                else:
                    static = static.expand(x0.size(0), -1, -1, -1)
            parts.append(static)
        x_in = torch.cat(parts, dim=1)
        x_cropped = self._crop_lat(x_in)

        # Encoder with skip connections
        skips = []
        h = self.conv_in(x_cropped)
        skips.append(h)
        for i, down in enumerate(self.downs):
            h = down(h)
            if i < len(self.downs) - 1:
                skips.append(h)

        h = self.bottleneck(h)

        # Decoder with skip connections (reverse order)
        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip)

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
        if self.direct_prediction:
            x_hat = decoder_out
        else:
            x_hat = x_bilinear + residual_scale * decoder_out

        return x_hat, {
            "z_tau": None,
            "d0": None,
            "d1": None,
            "decoder_out": decoder_out,
            "x_bilinear": x_bilinear,
            "residual_scale": residual_scale.detach(),
        }
