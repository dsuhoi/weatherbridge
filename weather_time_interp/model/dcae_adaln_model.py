"""DC-AE with native FiLM/AdaLN-style time-modulation (no tau-map channel).

Replaces tau-as-channel-broadcast (v7/v8) with sinusoidal+MLP time embedding fed
through DC-AE's built-in temb-channels path. Each ResBlock/EfficientViTBlock in
DC-AE applies scale+shift modulation: h = h * γ(t_emb) + β(t_emb) — equivalent to
FiLM (Sana/DiT-style AdaLN-Zero is the same modulation pattern with extra gate).

Input to encoder: cat([x0, xT, static]) — tau is NOT in input channels.
Output: x_hat = (1-τ)·x0 + τ·xT + tanh(scale) · decoder_out
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dcae import Encoder as DCAEEncoder, Decoder as DCAEDecoder


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for τ ∈ [0, 1] (DDPM-style)."""

    def __init__(self, dim: int, base_period: float = 16.0):
        super().__init__()
        self.dim = dim
        self.base_period = base_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(self.base_period) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    def __init__(self, time_dim: int = 256, freq_dim: int = 128, base_period: float = 16.0):
        super().__init__()
        self.embed = SinusoidalPosEmb(freq_dim, base_period=base_period)
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(t))


class WeatherDCAEAdaLNModel(nn.Module):
    """DC-AE encoder/decoder with FiLM-style time conditioning + bilinear-residual.

    Same DC-AE backbone as `WeatherDCAEResidualLinearModel` (v7), but τ is fed via
    sinusoidal+MLP embedding through `temb_channels` instead of as an extra input
    channel. DC-AE blocks already apply `h * scale + shift` modulation when
    `temb_channels` is set (see dcae.py ResBlock.forward).
    """

    def __init__(
        self,
        latent_channels: int = 32,
        block_out_channels: tuple = (128, 256, 512),
        layers_per_block: tuple = (2, 2, 2),
        block_type: tuple | str = ("ResBlock", "ResBlock", "EfficientViTBlock"),
        qkv_multiscales: tuple = ((), (), (5,)),
        attention_head_dim: int = 32,
        in_channels: int = 5,
        out_channels: int = 5,
        n_static_features: int = 0,
        lat_crop: int = 5,
        residual_scale_init: float = 0.10,
        residual_scale_learnable: bool = True,
        residual_clip: float | None = None,
        residual_scale_floor: float = 0.0,
        direct_prediction: bool = False,
        time_emb_dim: int = 256,
        time_freq_dim: int = 128,
        time_base_period: float = 16.0,
    ):
        super().__init__()
        # Persist every architectural kwarg so checkpoint normalizers /
        # introspection-based loaders can recover them without sniffing
        # state_dict shapes (see tools/ckpt_normalize.py).
        self.latent_channels = int(latent_channels)
        self.block_out_channels = tuple(block_out_channels)
        self.layers_per_block = tuple(layers_per_block)
        self.block_type = (
            tuple(block_type) if not isinstance(block_type, str) else block_type
        )
        self.qkv_multiscales = tuple(qkv_multiscales)
        self.attention_head_dim = int(attention_head_dim)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.residual_scale_init = float(residual_scale_init)
        self.residual_scale_learnable = bool(residual_scale_learnable)
        self.time_freq_dim = int(time_freq_dim)
        self.time_base_period = float(time_base_period)
        # lat_crop semantics (matches dcae_adaln_skip_model): >0 crop, <0 pad, ==0 no-op
        self.lat_crop = int(lat_crop)
        self.residual_clip = residual_clip
        self.residual_scale_floor = float(residual_scale_floor)
        self.direct_prediction = bool(direct_prediction)
        self.n_static_features = int(n_static_features)
        self.time_emb_dim = int(time_emb_dim)

        self.time_mlp = TimeMLP(
            time_dim=self.time_emb_dim,
            freq_dim=int(time_freq_dim),
            base_period=float(time_base_period),
        )

        # Input: cat([x0, xT, static]) — 2*C + N_static. NO tau channel.
        self.encoder = DCAEEncoder(
            in_channels=in_channels * 2 + self.n_static_features,
            latent_channels=latent_channels,
            temb_channels=self.time_emb_dim,
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
            temb_channels=self.time_emb_dim,
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
        if self.lat_crop == 0:
            return x
        n = abs(int(self.lat_crop))
        if self.lat_crop > 0:
            # CROP mode: drop n rows symmetrically
            crop_top = n // 2; crop_bottom = n - crop_top
            if crop_bottom == 0:
                return x[:, :, crop_top:]
            return x[:, :, crop_top:-crop_bottom]
        # PAD mode (lat_crop < 0): replicate-pad n rows symmetrically
        pad_top = n // 2; pad_bottom = n - pad_top
        return torch.nn.functional.pad(x, (0, 0, pad_top, pad_bottom), mode="replicate")

    def forward(self, x0, xT, tau, cond, static=None):
        if tau.dim() == 1:
            tau_ = tau.view(-1, 1, 1, 1)
        elif tau.dim() == 2:
            tau_ = tau.view(tau.size(0), 1, 1, 1)
        else:
            tau_ = tau

        # Build time embedding from τ — replaces the tau-map channel.
        t_emb = self.time_mlp(tau.view(-1))  # (B, time_emb_dim)
        # Ensure t_emb batch matches x0 batch (multi-tau flatten path).
        if t_emb.size(0) != x0.size(0):
            if x0.size(0) % t_emb.size(0) == 0:
                t_emb = t_emb.repeat_interleave(x0.size(0) // t_emb.size(0), dim=0)
            else:
                raise ValueError(
                    f"tau batch {t_emb.size(0)} cannot align with x0 batch {x0.size(0)}"
                )

        parts = [x0, xT]
        if self.n_static_features > 0:
            if static is None:
                raise ValueError(
                    f"Model requires static (n_static_features={self.n_static_features}), got None"
                )
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

        z_tau = self.encoder(self._crop_lat(x_in), t_emb)
        decoder_out_raw = self.decoder(z_tau, t_emb)
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
            "z_tau": z_tau,
            "d0": None,
            "d1": None,
            "decoder_out": decoder_out,
            "x_bilinear": x_bilinear,
            "residual_scale": residual_scale.detach(),
            "t_emb": t_emb,
        }
