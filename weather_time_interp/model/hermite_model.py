"""Legacy Hermite interpolator retained for old checkpoint compatibility."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_decoder import WeatherDecoder, WeatherEncoder


class FiLM(nn.Module):
    """Feature-wise linear modulation for endpoint derivatives."""

    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, num_features * 2),
            nn.SiLU(inplace=True),
            nn.Linear(num_features * 2, num_features * 2),
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = h.shape
        if channels != self.num_features:
            raise ValueError(
                f"expected {self.num_features} feature channels, got {channels}"
            )
        gamma, beta = self.fc(cond).chunk(2, dim=1)
        return gamma.view(batch_size, channels, 1, 1) * h + beta.view(
            batch_size, channels, 1, 1
        )


class HermiteParamNetFiLM(nn.Module):
    """Predict endpoint derivatives in latent space."""

    def __init__(
        self,
        latent_channels: int = 64,
        hidden_channels: int = 128,
        cond_dim: int = 1,
        cond_mlp_dim: int = 32,
    ):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_mlp_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_mlp_dim, cond_mlp_dim),
            nn.SiLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(
            latent_channels * 2, hidden_channels, kernel_size=3, padding=1
        )
        self.film1 = FiLM(cond_mlp_dim, hidden_channels)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.film2 = FiLM(cond_mlp_dim, hidden_channels)
        self.to_d = nn.Conv2d(hidden_channels, 2 * latent_channels, kernel_size=1)
        nn.init.zeros_(self.to_d.weight)
        nn.init.zeros_(self.to_d.bias)

    def forward(
        self, z0: torch.Tensor, zT: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond_emb = self.cond_mlp(cond.reshape(z0.size(0), -1))
        hidden = F.silu(self.film1(self.conv1(torch.cat([z0, zT], 1)), cond_emb))
        hidden = F.silu(self.film2(self.conv2(hidden), cond_emb))
        return self.to_d(hidden).chunk(2, dim=1)


class HermiteLatentInterpolator(nn.Module):
    """Cubic Hermite interpolation in latent space."""

    def forward(
        self,
        z0: torch.Tensor,
        zT: torch.Tensor,
        d0: torch.Tensor,
        d1: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        t = tau.reshape(tau.size(0), 1, 1, 1)
        t2 = t.square()
        t3 = t2 * t
        return (
            (2 * t3 - 3 * t2 + 1) * z0
            + (-2 * t3 + 3 * t2) * zT
            + (t3 - 2 * t2 + t) * d0
            + (t3 - t2) * d1
        )


class WeatherHermiteModel(nn.Module):
    """Historical Hermite model used by archived Lightning checkpoints."""

    def __init__(
        self,
        latent_channels: int = 64,
        cond_dim: int = 1,
        block_out_channels: tuple = (128, 128, 256, 256),
        layers_per_block: tuple = (2, 2, 2, 2),
        hidden_channels: int = 128,
        cond_mlp_dim: int = 32,
        in_channels: int = 5,
        out_channels: int = 5,
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
        self.encoder = WeatherEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )
        self.decoder = WeatherDecoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )
        self.param_net = HermiteParamNetFiLM(
            latent_channels, hidden_channels, cond_dim, cond_mlp_dim
        )
        self.hermite = HermiteLatentInterpolator()
        init_scale = torch.tensor(float(residual_scale_init))
        if residual_scale_learnable:
            self.residual_scale = nn.Parameter(init_scale)
        else:
            self.register_buffer("residual_scale", init_scale)

    def _crop_lat(self, x: torch.Tensor) -> torch.Tensor:
        if self.lat_crop <= 0:
            return x
        if x.size(-2) <= self.lat_crop:
            raise ValueError(
                f"lat_crop={self.lat_crop} is too large for height={x.size(-2)}"
            )
        top = self.lat_crop // 2
        bottom = self.lat_crop - top
        return x[:, :, top:] if bottom == 0 else x[:, :, top:-bottom]

    def encode_field(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._crop_lat(x))

    def forward(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau: torch.Tensor,
        cond: torch.Tensor,
    ):
        z0 = self.encode_field(x0)
        zT = self.encode_field(xT)
        d0, d1 = self.param_net(z0, zT, cond)
        z_tau = self.hermite(z0, zT, d0, d1, tau)
        decoder_out = F.interpolate(
            self.decoder(z_tau),
            size=x0.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if self.residual_clip is not None and self.residual_clip > 0:
            decoder_out = decoder_out.clamp(
                -self.residual_clip, self.residual_clip
            )
        residual_scale = torch.tanh(self.residual_scale)
        if self.residual_scale_floor > 0:
            residual_scale = residual_scale.sign() * residual_scale.abs().clamp(
                min=self.residual_scale_floor
            )
        tau_map = tau.reshape(tau.size(0), 1, 1, 1)
        x_bilinear = (1.0 - tau_map) * x0 + tau_map * xT
        return x_bilinear + residual_scale * decoder_out, {
            "z0": z0,
            "zT": zT,
            "d0": d0,
            "d1": d1,
            "z_tau": z_tau,
            "decoder_out": decoder_out,
            "x_bilinear": x_bilinear,
            "residual_scale": residual_scale.detach(),
        }
