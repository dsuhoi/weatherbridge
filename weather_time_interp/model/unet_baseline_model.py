import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_decoder import WeatherEncoder, WeatherDecoder


class WeatherUNetBaselineModel(nn.Module):
    """UNet-like baseline for temporal interpolation."""

    def __init__(
        self,
        latent_channels: int = 64,
        block_out_channels: tuple = (128, 128, 256, 256),
        layers_per_block: tuple = (2, 2, 2, 2),
        in_channels: int = 5,
        out_channels: int = 5,
        lat_crop: int = 1,
        residual_scale_init: float = 0.10,
        residual_scale_learnable: bool = True,
        residual_clip: float | None = None,
        residual_to_linear: bool = True,
        residual_scale_floor: float = 0.0,
        direct_prediction: bool = False,
        n_static_features: int = 0,
    ):
        super().__init__()
        self.lat_crop = max(0, int(lat_crop))
        self.residual_clip = residual_clip
        self.residual_to_linear = residual_to_linear
        self.residual_scale_floor = float(residual_scale_floor)
        self.direct_prediction = bool(direct_prediction)
        self.n_static_features = int(n_static_features)
        if self.direct_prediction:
            # In direct mode, ignore residual_to_linear (no bilinear baseline).
            self.residual_to_linear = False

        self.encoder = WeatherEncoder(
            in_channels=in_channels * 2 + 1 + self.n_static_features,
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

        init_scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        # In direct mode (no bilinear baseline), residual_scale isn't used in
        # forward path, so register as buffer to avoid DDP unused-param error.
        if residual_to_linear and not self.direct_prediction:
            if residual_scale_learnable:
                self.residual_scale = nn.Parameter(init_scale)
            else:
                self.register_buffer("residual_scale", init_scale)
        else:
            self.register_buffer("residual_scale", torch.tensor(1.0, dtype=torch.float32))

    def _crop_lat(self, x: torch.Tensor) -> torch.Tensor:
        if self.lat_crop <= 0:
            return x

        height = x.size(-2)
        if height <= self.lat_crop:
            raise ValueError(
                f"lat_crop={self.lat_crop} is too large for input height={height}"
            )

        crop_top = self.lat_crop // 2
        crop_bottom = self.lat_crop - crop_top

        if crop_bottom == 0:
            return x[:, :, crop_top:]
        return x[:, :, crop_top:-crop_bottom]

    def forward(
        self,
        x0: torch.Tensor,
        xT: torch.Tensor,
        tau: torch.Tensor,
        cond: torch.Tensor,
        static: torch.Tensor | None = None,
    ):
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

        z_tau = self.encoder(self._crop_lat(x_in))
        decoder_out_raw = self.decoder(z_tau)
        decoder_out = F.interpolate(
            decoder_out_raw,
            size=x0.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if self.residual_to_linear:
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
        else:
            residual_scale = self.residual_scale
            x_bilinear = (1.0 - tau_) * x0 + tau_ * xT
            x_hat = decoder_out

        return x_hat, {
            "z_tau": z_tau,
            "d0": None,
            "d1": None,
            "decoder_out": decoder_out,
            "x_bilinear": x_bilinear,
            "residual_scale": residual_scale.detach(),
        }


class WeatherUNetResidualLinearModel(WeatherUNetBaselineModel):
    def __init__(self, **kwargs):
        super().__init__(residual_to_linear=True, **kwargs)


class WeatherUNetDirectModel(WeatherUNetBaselineModel):
    """Pure direct prediction without bilinear baseline.

    Decoder output IS the prediction. Used to test whether residual-to-bilinear
    structure is necessary or whether the model can learn the full mapping.
    """

    def __init__(self, **kwargs):
        kwargs.pop("residual_to_linear", None)  # ignore — we force direct
        kwargs.pop("direct_prediction", None)
        super().__init__(residual_to_linear=False, direct_prediction=True, **kwargs)
