"""FuXi-style SwinV2 transformer backbone for time interpolation.

FuXi (Chen 2023 "FuXi: A cascade machine learning forecasting system") uses
SwinV2 transformer blocks in its UTransformer backbone. This is a simplified
adaptation for the interpolation task:

  - Patch-embed (4×4 conv stride 4): 181×360 → 45×90 tokens, dim=embed_dim
  - Stack of SwinV2Block (W-MSA + SW-MSA alternating, NHWC layout)
  - Between block pairs: FiLM modulation `x = x * (1 + γ(t_emb)) + β(t_emb)`
    where (γ, β) come from sinusoidal+MLP τ-embedding (same pipeline as DC-AE v9
    and S-DYff v9 — paper-faithful AdaLN-Zero style modulation).
  - Patch-unembed (1×1 → PixelShuffle 4×): tokens → 181×360 grid
  - Bilinear+residual wrapping: x_hat = (1-τ)·x0 + τ·xT + tanh(scale) · v_θ

τ is NOT fed as input channel — only via per-block FiLM, like DC-AE v9.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import SwinTransformerBlockV2


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, base_period: float = 16.0):
        super().__init__()
        self.dim = dim
        self.base_period = base_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        half_dim = self.dim // 2
        emb = math.log(self.base_period) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
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


class FiLMModulator(nn.Module):
    """AdaLN-Zero style FiLM: t_emb → (γ, β), apply x * (1 + γ) + β."""

    def __init__(self, time_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(time_dim, 2 * feat_dim)
        # Zero-init last linear so block starts as identity (AdaLN-Zero trick).
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x shape: (B, H, W, C) — SwinV2 NHWC layout
        scale_shift = self.proj(t_emb)                          # (B, 2C)
        scale, shift = scale_shift.chunk(2, dim=-1)             # (B, C), (B, C)
        return x * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]


class WeatherFuXiSwinV2Model(nn.Module):
    """SwinV2 transformer + FiLM time conditioning + bilinear-residual."""

    def __init__(
        self,
        in_channels: int = 27,
        out_channels: int = 27,
        n_static_features: int = 3,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        window_size: tuple = (5, 9),    # (H_window, W_window); 45/5=9 windows lat, 90/9=10 lon
        mlp_ratio: float = 4.0,
        patch_size: int = 4,
        time_emb_dim: int = 256,
        time_freq_dim: int = 128,
        time_base_period: float = 16.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        residual_scale_init: float = 0.10,
        residual_scale_learnable: bool = True,
        residual_clip: float | None = None,
        residual_scale_floor: float = 0.0,
        direct_prediction: bool = False,
        lat_crop: int = 1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_static_features = int(n_static_features)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.lat_crop = max(0, int(lat_crop))
        self.depth = int(depth)
        # Persist remaining init kwargs for introspection-based loaders.
        self.num_heads = int(num_heads)
        self.window_size = tuple(window_size)
        self.mlp_ratio = float(mlp_ratio)
        self.time_emb_dim = int(time_emb_dim)
        self.time_freq_dim = int(time_freq_dim)
        self.time_base_period = float(time_base_period)
        self.attention_dropout = float(attention_dropout)
        self.stochastic_depth_prob = float(stochastic_depth_prob)
        self.residual_scale_init = float(residual_scale_init)
        self.residual_scale_learnable = bool(residual_scale_learnable)
        self.residual_clip = residual_clip
        self.residual_scale_floor = float(residual_scale_floor)
        self.direct_prediction = bool(direct_prediction)

        # Input: cat([x0, xT, static]) — no τ channel.
        in_total = in_channels * 2 + self.n_static_features

        # Patch embed: 4×4 conv stride 4. Cropped input 180×360 → 45×90 tokens.
        self.patch_embed = nn.Conv2d(in_total, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Time embedding (same shape as DC-AE v9 / S-DYff v9).
        self.time_mlp = TimeMLP(time_dim=time_emb_dim, freq_dim=time_freq_dim, base_period=time_base_period)

        # SwinV2 blocks alternating W-MSA / SW-MSA + per-block FiLM modulator.
        self.swin_blocks = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for i in range(depth):
            shift = (0, 0) if i % 2 == 0 else (window_size[0] // 2, window_size[1] // 2)
            self.swin_blocks.append(
                SwinTransformerBlockV2(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=list(window_size),
                    shift_size=list(shift),
                    mlp_ratio=mlp_ratio,
                    attention_dropout=attention_dropout,
                    stochastic_depth_prob=stochastic_depth_prob * i / max(depth - 1, 1),
                )
            )
            self.film_layers.append(FiLMModulator(time_emb_dim, embed_dim))

        self.norm_out = nn.LayerNorm(embed_dim)

        # Patch unembed: 1×1 conv to expand to out_C * patch² channels, then PixelShuffle.
        self.patch_unembed_proj = nn.Conv2d(embed_dim, out_channels * patch_size * patch_size, kernel_size=1)

        init_scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if residual_scale_learnable and not self.direct_prediction:
            self.residual_scale = nn.Parameter(init_scale)
        else:
            self.register_buffer("residual_scale", init_scale)

    def _crop_lat(self, x: torch.Tensor) -> torch.Tensor:
        if self.lat_crop <= 0:
            return x
        h = x.size(-2)
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

        t_emb = self.time_mlp(tau.view(-1))  # (B, time_emb_dim)
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
                    static = static.repeat_interleave(x0.size(0) // static.size(0), dim=0)
                else:
                    static = static.expand(x0.size(0), -1, -1, -1)
            parts.append(static)
        x_in = torch.cat(parts, dim=1)             # (B, C_in, H, W) NCHW

        # Crop H=181→180 (or any even) for clean patch division.
        x_cropped = self._crop_lat(x_in)

        # Patch embed → NCHW with reduced H, W.
        z = self.patch_embed(x_cropped)            # (B, embed_dim, H/patch, W/patch)
        # SwinV2 expects NHWC.
        z = z.permute(0, 2, 3, 1).contiguous()     # (B, H/p, W/p, embed_dim)

        for blk, film in zip(self.swin_blocks, self.film_layers):
            z = blk(z)                              # Swin V2 block: NHWC in/out
            z = film(z, t_emb)                      # FiLM modulation post-block

        z = self.norm_out(z)
        z = z.permute(0, 3, 1, 2).contiguous()     # back to NCHW

        # Unpatch: 1x1 expand + PixelShuffle.
        z = self.patch_unembed_proj(z)              # (B, C_out * p², H/p, W/p)
        z = F.pixel_shuffle(z, self.patch_size)     # (B, C_out, H/p * p, W/p * p)
        decoder_out = F.interpolate(z, size=x0.shape[-2:], mode="bilinear", align_corners=False)

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
            "t_emb": t_emb,
        }
