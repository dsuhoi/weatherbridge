"""Spherical Fourier Neural Operator baseline for time interpolation.

Wraps neuralop.models.SFNO into the residual-to-bilinear pipeline. SFNO operates
in spherical harmonic basis — natural for global weather data on a (lat,lon) grid.

Input: cat([x0, xT, tau_map], dim=1) → SFNO → x_core
Output: x_hat = bilinear(x0, xT, tau) + tanh(scale) * x_core
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.models import SFNO


class WeatherSFNOResidualLinearModel(nn.Module):
    """SFNO + bilinear baseline + learnable residual scale.

    SFNO uses spherical harmonics, so it natively respects sphere geometry
    of (lat,lon) data. Should be especially good for periodic longitude.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 5,
        # Default scaled up from (64, n_layers=4) → (128, n_layers=6).
        # neuralop SFNO uses Tucker factorization so spectral weights stay manageable.
        # Targeting ~25-30M params, comparable to DC-AE/UNet for fair comparison.
        hidden_channels: int = 128,
        n_modes: tuple = (64, 64),
        n_layers: int = 6,
        lat_crop: int = 1,
        residual_scale_init: float = 0.10,
        residual_scale_learnable: bool = True,
        residual_clip: float | None = None,
        residual_scale_floor: float = 0.0,
        direct_prediction: bool = False,
        n_static_features: int = 0,
    ):
        super().__init__()
        self.lat_crop = max(0, int(lat_crop))
        self.residual_clip = residual_clip
        self.residual_scale_floor = float(residual_scale_floor)
        self.direct_prediction = bool(direct_prediction)
        self.n_static_features = int(n_static_features)

        self.sfno = SFNO(
            n_modes=n_modes,
            in_channels=in_channels * 2 + 1 + self.n_static_features,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
        )

        init_scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if residual_scale_learnable and not self.direct_prediction:
            self.residual_scale = nn.Parameter(init_scale)
        else:
            self.register_buffer("residual_scale", init_scale)

    def _crop_lat(self, x: torch.Tensor) -> torch.Tensor:
        if self.lat_crop <= 0:
            return x
        height = x.size(-2)
        if height <= self.lat_crop:
            raise ValueError(f"lat_crop={self.lat_crop} too large for height={height}")
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
                raise ValueError(f"Model requires static (n_static={self.n_static_features}), got None")
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
        x_in_cropped = self._crop_lat(x_in)

        decoder_out_raw = self.sfno(x_in_cropped)
        # SFNO preserves spatial dims; restore full (181, 360) if cropped.
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
