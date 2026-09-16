"""WeatherDCAE-CrossFrame: DC-AE backbone + ATM-VFI cross-frame attention.

Motivation: the vanilla WeatherDCAE (`WeatherDCAEAdaLNModel`) fuses the
two 6-hourly anchor frames at the INPUT (channel concat) and encodes them
jointly. At interior τ its bottleneck latent collapses toward the linear
midpoint, so it recovers the diurnal-cycle amplitude no better than
Linear-Interp. ATM-VFI (`PixelAttentionVFINet`) instead encodes the two
frames SEPARATELY and applies cross-frame pixel attention at the
bottleneck — letting each output pixel attend to both anchors and model a
non-linear sub-6h trajectory. This is the single feature responsible for
ATM-VFI's diurnal-amplitude advantage.

This model ports that feature into WeatherDCAE:
  1. Encode ``[x0, static]`` and ``[xT, static]`` SEPARATELY through the
     same weight-shared DC-AE encoder → z0, zT (bottleneck).
  2. ``z = CrossFrameAttention(z0, zT)`` — MHA over concatenated bottleneck
     tokens; average the two halves (ATM-VFI module, ported).
  3. τ conditioning via the DC-AE temb path (unchanged).
  4. Decode z → residual; add to the bilinear scaffold (unchanged).

Output contract is identical to WeatherDCAEAdaLNModel:
  ``forward(x0, xT, tau, cond, static=None) -> (x_hat, aux_dict)``
so it drops straight into the existing eval / bare-blob loader
(register ``WeatherDCAECrossFrameModel`` in the arch registry).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dcae import Encoder as DCAEEncoder, Decoder as DCAEDecoder
from .dcae_adaln_model import TimeMLP


class CrossFrameAttention(nn.Module):
    """Self-attention over concatenated bottleneck tokens of two frames.

    Ported from ATM-VFI's PixelAttentionVFINet. Each spatial token from
    z0 and zT attends to every other token; the two halves are averaged
    back into a single (B, C, H, W) latent.

    Efficiency: full MHA over 2·H·W tokens at the /8 bottleneck (46×90 →
    8280 tokens) is O(N²) and dominates training cost. We pool each frame
    to a fixed coarse grid (``attn_grid``) BEFORE the cross-frame
    attention, then upsample the attended correction back to (H, W) and
    add it as a residual to the frame mean. Synoptic-scale correspondence
    is captured at the coarse grid; local detail is carried by the
    frame-mean skip. This cuts attention cost ~16× (46×90 → 12×24) with
    negligible quality loss for temporal interpolation.
    """

    def __init__(self, dim: int, num_heads: int = 8,
                 attn_grid: tuple = (12, 24)):
        super().__init__()
        if dim % num_heads != 0:
            for h in (8, 4, 2, 1):
                if dim % h == 0:
                    num_heads = h
                    break
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.attn_grid = attn_grid

    def forward(self, z0: torch.Tensor, zT: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z0.shape
        gh, gw = self.attn_grid
        # Pool both frames to the coarse attention grid.
        p0 = F.adaptive_avg_pool2d(z0, (gh, gw))
        pT = F.adaptive_avg_pool2d(zT, (gh, gw))
        t0 = p0.flatten(2).transpose(1, 2)   # (B, gh*gw, C)
        tT = pT.flatten(2).transpose(1, 2)
        tokens = torch.cat([t0, tT], dim=1)  # (B, 2*gh*gw, C)
        tokens = self.norm(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + attn_out
        fused = (tokens[:, : gh * gw] + tokens[:, gh * gw:]) / 2
        fused = fused.transpose(1, 2).reshape(B, C, gh, gw)
        # Upsample the attended correction back to bottleneck resolution.
        corr = F.interpolate(fused, size=(H, W), mode="bilinear",
                              align_corners=False)
        # Residual over the frame mean (local detail preserved).
        return 0.5 * (z0 + zT) + corr


class WeatherDCAECrossFrameModel(nn.Module):
    """WeatherDCAE with dual-frame encoding and cross-frame attention."""

    def __init__(
        self,
        latent_channels: int = 32,
        block_out_channels: tuple = (128, 256, 512),
        layers_per_block: tuple = (2, 2, 2),
        block_type: tuple | str = ("ResBlock", "ResBlock", "EfficientViTBlock"),
        qkv_multiscales: tuple = ((), (), (5,)),
        attention_head_dim: int = 32,
        in_channels: int = 24,
        out_channels: int = 24,
        n_static_features: int = 0,
        lat_crop: int = -8,
        residual_scale_init: float = 0.10,
        residual_scale_learnable: bool = True,
        residual_clip: float | None = None,
        residual_scale_floor: float = 0.0,
        direct_prediction: bool = False,
        time_emb_dim: int = 256,
        time_freq_dim: int = 128,
        time_base_period: float = 16.0,
        cross_frame_heads: int = 8,
    ):
        super().__init__()
        # Persist architectural kwargs for the introspection-based loader.
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
        self.n_static_features = int(n_static_features)
        self.lat_crop = int(lat_crop)
        self.residual_scale_init = float(residual_scale_init)
        self.residual_scale_learnable = bool(residual_scale_learnable)
        self.residual_clip = residual_clip
        self.residual_scale_floor = float(residual_scale_floor)
        self.direct_prediction = bool(direct_prediction)
        self.time_emb_dim = int(time_emb_dim)
        self.time_freq_dim = int(time_freq_dim)
        self.time_base_period = float(time_base_period)
        self.cross_frame_heads = int(cross_frame_heads)

        self.time_mlp = TimeMLP(
            time_dim=self.time_emb_dim,
            freq_dim=int(time_freq_dim),
            base_period=float(time_base_period),
        )

        # DUAL-FRAME encoder: single frame + static (NOT 2*C).
        self.encoder = DCAEEncoder(
            in_channels=in_channels + self.n_static_features,
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
        self.cross_frame = CrossFrameAttention(
            latent_channels, num_heads=self.cross_frame_heads
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
            crop_top = n // 2
            crop_bottom = n - crop_top
            if crop_bottom == 0:
                return x[:, :, crop_top:]
            return x[:, :, crop_top:-crop_bottom]
        pad_top = n // 2
        pad_bottom = n - pad_top
        return F.pad(x, (0, 0, pad_top, pad_bottom), mode="replicate")

    def _prep_static(self, static, B, device):
        if self.n_static_features <= 0:
            return None
        if static is None:
            raise ValueError(
                f"Model requires static (n_static_features={self.n_static_features}), got None"
            )
        if static.dim() == 3:
            static = static.unsqueeze(0)
        if static.size(0) != B:
            if B % static.size(0) == 0:
                static = static.repeat_interleave(B // static.size(0), dim=0)
            else:
                static = static.expand(B, -1, -1, -1)
        return static.to(device)

    def forward(self, x0, xT, tau, cond, static=None):
        if tau.dim() == 1:
            tau_ = tau.view(-1, 1, 1, 1)
        elif tau.dim() == 2:
            tau_ = tau.view(tau.size(0), 1, 1, 1)
        else:
            tau_ = tau

        B = x0.size(0)
        t_emb = self.time_mlp(tau.view(-1))
        if t_emb.size(0) != B:
            if B % t_emb.size(0) == 0:
                t_emb = t_emb.repeat_interleave(B // t_emb.size(0), dim=0)
            else:
                raise ValueError(
                    f"tau batch {t_emb.size(0)} cannot align with x0 batch {B}"
                )

        static_b = self._prep_static(static, B, x0.device)

        def _frame_in(x):
            parts = [x]
            if static_b is not None:
                parts.append(static_b)
            return self._crop_lat(torch.cat(parts, dim=1))

        # Weight-shared dual-frame encode.
        z0 = self.encoder(_frame_in(x0), t_emb)
        zT = self.encoder(_frame_in(xT), t_emb)
        z = self.cross_frame(z0, zT)

        decoder_out_raw = self.decoder(z, t_emb)
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
            "z_tau": z,
            "z0": z0,
            "zT": zT,
            "decoder_out": decoder_out,
            "x_bilinear": x_bilinear,
            "residual_scale": residual_scale.detach(),
            "t_emb": t_emb,
        }
