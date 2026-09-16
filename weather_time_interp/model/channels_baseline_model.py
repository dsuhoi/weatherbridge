"""Multi-channel output approach: predict all hours τ ∈ {1..max_tau} in one forward.

Unlike τ-conditioned models (ModAFNO/DC-AE which take tau scalar and predict x_τ),
this approach reframes time interpolation as **multi-output regression**:

    input:  (x0, xT, static)
    output: (x_1, x_2, ..., x_{max_tau})  — all interpolated hours simultaneously

Advantages over τ-conditioning:
- One forward pass per (x0, xT) window vs max_tau forwards → ~max_tau× faster training.
- Shared latent representation across all hours forces cross-hour consistency.
- All hour-losses summed → implicit regularization.

Architecture: identical DC-AE encoder/decoder, but:
- decoder outputs (B, max_tau * C, H, W) — reshape to (B, max_tau, C, H, W)
- per-hour bilinear baseline is computed outside model (in trainer), residual_scale
  shared across hours via single scalar tanh-bounded.

forward signature (compat with trainer's `_shared_step`):
    x_hat (B, max_tau, C, H, W), aux  — when tau=None
    x_hat (B, C, H, W), aux             — when tau given (selects single hour for eval compat)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dcae import Encoder as DCAEEncoder, Decoder as DCAEDecoder


class WeatherChannelsResidualLinearModel(nn.Module):
    """DC-AE backbone with multi-hour output head.

    Produces residuals for all hours h ∈ {1..max_tau} in one decoder pass.
    """

    def __init__(
        self,
        latent_channels: int = 128,
        block_out_channels: tuple = (128, 256, 512),
        layers_per_block: tuple = (2, 2, 2),
        block_type: tuple | str = ("ResBlock", "ResBlock", "EfficientViTBlock"),
        qkv_multiscales: tuple = ((), (), (5,)),
        attention_head_dim: int = 32,
        in_channels: int = 5,
        out_channels: int = 5,
        max_tau_hours: int = 6,
        n_static_features: int = 0,
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_tau_hours = int(max_tau_hours)
        # Hours predicted: 1..max_tau (endpoints 0, max_tau are exact via bilinear).
        self.n_pred_hours = self.max_tau_hours - 1

        # Encoder input: cat([x0, xT, static]) — no tau-map (predict all hours).
        encoder_in = in_channels * 2 + self.n_static_features
        # Decoder output: out_channels * n_pred_hours (residuals for h=1..max_tau-1).
        decoder_out = out_channels * self.n_pred_hours

        self.encoder = DCAEEncoder(
            in_channels=encoder_in,
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
            out_channels=decoder_out,
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
        if residual_scale_learnable and not self.direct_prediction:
            self.residual_scale = nn.Parameter(init_scale)
        else:
            self.register_buffer("residual_scale", init_scale)

    def _crop_lat(self, x: torch.Tensor) -> torch.Tensor:
        if self.lat_crop <= 0:
            return x
        height = x.size(-2)
        crop_top = self.lat_crop // 2
        crop_bottom = self.lat_crop - crop_top
        if crop_bottom == 0:
            return x[:, :, crop_top:]
        return x[:, :, crop_top:-crop_bottom]

    def _bilinear_all_hours(self, x0: torch.Tensor, xT: torch.Tensor) -> torch.Tensor:
        """Build per-hour bilinear baselines for h=1..max_tau-1.

        Returns (B, n_pred_hours, C, H, W).
        """
        B, C, H, W = x0.shape
        bilinears = []
        for h in range(1, self.max_tau_hours):
            tau = h / float(self.max_tau_hours)
            bilinears.append((1.0 - tau) * x0 + tau * xT)
        return torch.stack(bilinears, dim=1)  # (B, n_pred_hours, C, H, W)

    def forward(self, x0, xT, tau=None, cond=None, static=None):
        """
        If tau is None: returns x_hat (B, n_pred_hours, C, H, W) — all hours.
        If tau is given (scalar/tensor of hours in normalized [0,1] units): returns single hour.
        """
        B, C, H, W = x0.shape

        # Build encoder input
        parts = [x0, xT]
        if self.n_static_features > 0:
            if static is None:
                raise ValueError(f"Channels model requires static (n_static={self.n_static_features}), got None")
            if static.dim() == 3:
                static = static.unsqueeze(0).expand(B, -1, -1, -1)
            elif static.size(0) != B:
                static = static.expand(B, -1, -1, -1)
            parts.append(static)
        x_in = torch.cat(parts, dim=1)

        # Encode → decode → reshape to (B, n_pred_hours, C, H, W)
        z = self.encoder(self._crop_lat(x_in))
        dec_raw = self.decoder(z)  # (B, out_channels * n_pred_hours, H_dec, W_dec)
        dec_full = F.interpolate(
            dec_raw, size=x0.shape[-2:], mode="bilinear", align_corners=False
        )
        decoder_out = dec_full.view(B, self.n_pred_hours, self.out_channels, H, W)

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

        x_bilinear_all = self._bilinear_all_hours(x0, xT)  # (B, n_pred_hours, C, H, W)
        if self.direct_prediction:
            x_hat_all = decoder_out
        else:
            x_hat_all = x_bilinear_all + residual_scale * decoder_out

        aux = {
            "z_tau": z,
            "d0": None,
            "d1": None,
            "decoder_out": decoder_out,
            "x_bilinear": x_bilinear_all,
            "residual_scale": residual_scale.detach(),
        }

        # If tau provided (eval-time per-hour query), pick specific hour from all-hours output.
        if tau is not None:
            # tau in normalized [0,1] units → hour_idx ∈ {0..max_tau}
            if tau.dim() == 0:
                tau_scalar = tau.float()
            elif tau.dim() == 1:
                tau_scalar = tau.float()
            else:
                tau_scalar = tau.float().view(-1)
            hour = (tau_scalar * self.max_tau_hours).round().long().clamp(0, self.max_tau_hours)
            # Endpoints: h=0 → x0, h=max_tau → xT (exact from bilinear)
            x_hat_out = torch.empty_like(x0)
            for b in range(B):
                h = int(hour[b].item()) if hour.dim() > 0 else int(hour.item())
                if h == 0:
                    x_hat_out[b] = x0[b]
                elif h == self.max_tau_hours:
                    x_hat_out[b] = xT[b]
                else:
                    # h ∈ {1..max_tau-1} → index h-1 in predicted array
                    x_hat_out[b] = x_hat_all[b, h - 1]
            return x_hat_out, aux

        # Training/multi-hour return: full (B, n_pred_hours, C, H, W)
        return x_hat_all, aux
