"""DC-AE based baseline (transformer + residual blocks) for time interpolation.

Wraps DC-AE Encoder/Decoder (Sana-style hybrid: ResBlock + EfficientViTBlock with
linear attention on deeper levels) into the same residual-to-bilinear pipeline as
WeatherUNetBaselineModel.

Input: cat([x0, xT, tau_map], dim=1) → DC-AE encoder → latent → DC-AE decoder → x_core
Output: x_hat = bilinear(x0, xT, tau) + tanh(scale) * x_core
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dcae import Encoder as DCAEEncoder, Decoder as DCAEDecoder


class WeatherDCAEResidualLinearModel(nn.Module):
    """DC-AE encoder/decoder + bilinear baseline + learnable residual.

    Default config uses 4 stages with EfficientViTBlock (linear attention) on the
    last 2 stages — this is the "transformer + residual" recipe.
    """

    def __init__(
        self,
        latent_channels: int = 128,
        # NOTE: DC-AE DCDownBlock requires `prev_ch * 4 % next_ch == 0` between
        # consecutive stages (group_size invariant, see dcae.py:478). Stick to
        # power-of-two friendly progressions like (128, 256, 512).
        # Also: ERA5 grid 360 lon has only 3 factors of 2 — at most 2 downsamples
        # before width becomes odd. So use 3 stages (= 2 downsamples).
        block_out_channels: tuple = (128, 256, 512),
        layers_per_block: tuple = (2, 2, 2),
        block_type: tuple | str = ("ResBlock", "ResBlock", "EfficientViTBlock"),
        qkv_multiscales: tuple = ((), (), (5,)),
        attention_head_dim: int = 32,
        in_channels: int = 5,
        out_channels: int = 5,
        n_static_features: int = 0,  # land_sea_mask, orography, lat_cos
        # lat_crop=5 makes height 181 → 176 = 16*11, divisible by 4.
        lat_crop: int = 5,
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

        # Input: cat([x0, xT, tau_map, static]) → 2*C + 1 + N_static channels.
        self.encoder = DCAEEncoder(
            in_channels=in_channels * 2 + 1 + self.n_static_features,
            latent_channels=latent_channels,
            block_type=block_type,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=tuple(layers_per_block),
            qkv_multiscales=qkv_multiscales,
            attention_head_dim=attention_head_dim,
            downsample_block_type="pixel_unshuffle",
            out_shortcut=True,
        )
        self.decoder = DCAEDecoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_type=block_type,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=tuple(layers_per_block),
            qkv_multiscales=qkv_multiscales,
            attention_head_dim=attention_head_dim,
            upsample_block_type="pixel_shuffle",
            in_shortcut=True,
        )

        init_scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        # In direct mode, residual_scale isn't used in the forward path, so don't
        # make it a Parameter (DDP would complain about unused parameters).
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
                raise ValueError(f"Model requires static (n_static_features={self.n_static_features}), got None")
            # static: (n, H, W) or (B, n, H, W) → broadcast/repeat to match x0 batch
            if static.dim() == 3:
                static = static.unsqueeze(0).expand(x0.size(0), -1, -1, -1)
            elif static.size(0) != x0.size(0):
                # Multi-tau flattens batch from B to B*n_tau; replicate static accordingly.
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
            # Pure direct: no bilinear baseline, decoder_out IS the prediction.
            x_hat = decoder_out
        else:
            x_hat = x_bilinear + residual_scale * decoder_out

        return x_hat, {
            "z_tau": z_tau,
            "d0": None,
            "d1": None,
            "decoder_out": decoder_out,
            "x_bilinear": x_bilinear,
            "residual_scale": residual_scale.detach(),
        }
