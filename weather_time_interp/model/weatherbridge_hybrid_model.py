"""WeatherDCAE-Hybrid: DC-AE backbone + adaptive skip + cross-frame attention.

Combines the best elements of:
  • WeatherDCAEAdaLNModel (NoSkip):
      - DC-AE pixel-(un)shuffle backbone (compute-efficient (un)downsample)
      - FiLM modulation in every ResBlock (deep τ-conditioning)
      - AdaLN-Zero in EfficientViT bottleneck
      - SphereConv2d circular padding
  • WeatherDCAEAdaLNSkipModel:
      - 3 zero-init τ-conditional gated skip connections (encoder→decoder)
  • ATM-VFI:
      - Separate frame encoding (x_0, x_T independent encoders)
      - Cross-Frame Attention at bottleneck (explicit motion modelling)
      - Per-channel residual scale (s ∈ R^C, not a global scalar)

Forward pass:
  1. Encode x_0 (concat with static) → list of features per stage
  2. Encode x_T (concat with static) → list of features per stage  (shared encoder weights)
  3. Cross-frame attention on bottleneck features fuses motion x_0 ↔ x_T
  4. AdaLN-Zero refinement on fused bottleneck (with t_emb)
  5. Decoder with τ-conditional gated skip connections:
       skip_i = blend(f_0[i], f_T[i], τ)            (bilinear-like blend)
       gate_i = tanh(skip_gate_mlp(t_emb))[i]       (adaptive per-stage gate)
       h     = up(h) + gate_i · skip_i              (zero-init at start)
  6. Pixel output:
       delta   = out_conv(decoder_top)
       x̂_τ    = (1-τ/Δ)·x_0 + (τ/Δ)·x_T + tanh(s_chan)·delta   (per-channel scale)

Target capacity ≈ 14-16M params (between Skip 14M and ATM-VFI 12M).

Hyper-defaults (3-stage):
    block_out_channels = (96, 192, 384)
    layers_per_block   = (2, 2, 2)
    latent_channels    = 192
    block_type         = ("ResBlock", "ResBlock", "EfficientViTBlock")
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import existing DC-AE building blocks. Encoder/Decoder are exported without
# DCAE prefix in dcae.py — alias for readability. DCDownBlock2d marks the
# downsample boundary used by `_get_stage_layout`.
from .dcae import (
    Encoder as DCAEEncoder,
    Decoder as DCAEDecoder,
    DCDownBlock2d,
)
from diffusers.models.normalization import RMSNorm
from .dcae_adaln_model import SinusoidalPosEmb, TimeMLP
# SphereConv2d is used INSIDE DCAEEncoder/DCAEDecoder (their 3x3 ResBlock and
# EfficientViT convs); we don't construct it directly here — skip projections
# are 1x1 which doesn't need pole-aware padding.


class CrossFrameAttention(nn.Module):
    """Lightweight cross-attention between bottleneck features of x_0 and x_T.

    Multi-head attention with shared K/V projection and τ-modulated mixing.
    Shape-invariant: in/out are (B, C, H, W); flattens (H*W) as the token dim.
    """

    def __init__(self, channels: int, n_heads: int = 8, time_emb_dim: int = 256):
        super().__init__()
        assert channels % n_heads == 0, f"{channels} not divisible by {n_heads} heads"
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.norm0 = RMSNorm(channels, 1e-7, elementwise_affine=True, bias=True)
        self.normT = RMSNorm(channels, 1e-7, elementwise_affine=True, bias=True)
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.proj_out = nn.Linear(channels, channels, bias=True)
        # AdaLN-Zero gate (so init = identity ⇒ no risk of breaking pre-trained backbone)
        self.gate_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(self, f0: torch.Tensor, fT: torch.Tensor,
                temb: torch.Tensor) -> torch.Tensor:
        """f0, fT: (B, C, H, W) — bottleneck features. Returns fused (B, C, H, W)."""
        B, C, H, W = f0.shape
        # Tokenise — (B, H*W, C)
        a0 = self.norm0(f0.movedim(1, -1)).flatten(1, 2)        # query side
        aT = self.normT(fT.movedim(1, -1)).flatten(1, 2)        # key/value side
        # Cross-attention: Q from f0 attends to K,V from fT
        qkv0 = self.qkv(a0).chunk(3, dim=-1)
        qkvT = self.qkv(aT).chunk(3, dim=-1)
        q = qkv0[0].view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = qkvT[1].view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = qkvT[2].view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)          # (B, n_heads, T, d)
        attn = attn.transpose(1, 2).reshape(B, H * W, C)
        attn = self.proj_out(attn).view(B, H, W, C).movedim(-1, 1)
        # Symmetric direction: T attends to 0 — sum both for symmetric motion.
        qT = qkvT[0].view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k0 = qkv0[1].view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v0 = qkv0[2].view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        attnT = F.scaled_dot_product_attention(qT, k0, v0)
        attnT = attnT.transpose(1, 2).reshape(B, H * W, C)
        attnT = self.proj_out(attnT).view(B, H, W, C).movedim(-1, 1)
        # Mean fuse, gated by AdaLN-Zero so init = identity.
        gate = self.gate_mlp(temb).view(B, C, 1, 1)
        fused_base = 0.5 * (f0 + fT)
        return fused_base + gate * 0.5 * (attn + attnT)


class WeatherDCAEHybridModel(nn.Module):
    """DC-AE encoder/decoder + cross-frame attention + adaptive skip + per-ch scale.

    Encoder is shared (same weights process x_0 and x_T independently). Skip
    connections feed τ-blended encoder features into the decoder with per-stage
    learned tanh-gates conditioned on t_emb (zero-init).

    Bilinear scaffold + per-channel residual scale:
        x̂_τ = (1-τ/Δ) x_0 + (τ/Δ) x_T + tanh(s_chan) · decoder_out
    """

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 24,
        n_static_features: int = 3,
        latent_channels: int = 192,
        block_out_channels: Tuple[int, ...] = (96, 192, 384),
        layers_per_block: Tuple[int, ...] = (2, 2, 2),
        block_type: Tuple[str, ...] = ("ResBlock", "ResBlock", "EfficientViTBlock"),
        qkv_multiscales: Tuple = ((), (), (5,)),
        time_emb_dim: int = 256,
        time_freq_dim: int = 128,
        time_base_period: float = 16.0,
        cross_attn_heads: int = 8,
        residual_scale_init: float = 0.30,
        residual_scale_floor: float = 0.05,
        residual_scale_learnable: bool = True,
        residual_clip: Optional[float] = None,
        direct_prediction: bool = False,
        lat_crop: int = -8,
        max_tau_hours: int = 6,
    ):
        super().__init__()
        n_stages = len(block_out_channels)
        assert len(layers_per_block) == n_stages == len(block_type)
        self.lat_crop = int(lat_crop)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_static_features = int(n_static_features)
        self.latent_channels = int(latent_channels)
        self.time_emb_dim = int(time_emb_dim)
        self.max_tau_hours = int(max_tau_hours)
        self.direct_prediction = bool(direct_prediction)
        self.residual_scale_floor = float(residual_scale_floor)
        self.residual_clip = residual_clip

        # τ pathway.
        self.time_mlp = TimeMLP(
            time_dim=self.time_emb_dim,
            freq_dim=int(time_freq_dim),
            base_period=float(time_base_period),
        )

        # Shared encoder — processes a single frame (with static) at a time.
        # Input channels = C + n_static (we DO NOT concat 2 frames here; that
        # would defeat cross-attention. Single-frame encoder is the key
        # architectural change vs WeatherDCAEAdaLNModel.)
        self.encoder = DCAEEncoder(
            in_channels=self.in_channels + self.n_static_features,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=tuple(layers_per_block),
            block_type=tuple(block_type),
            qkv_multiscales=tuple(qkv_multiscales),
            latent_channels=self.latent_channels,
            temb_channels=self.time_emb_dim,
            out_shortcut=True,
        )

        # Cross-frame attention on bottleneck latents.
        self.cross_attn = CrossFrameAttention(
            channels=self.latent_channels,
            n_heads=cross_attn_heads,
            time_emb_dim=self.time_emb_dim,
        )

        # Decoder: standard DC-AE decoder shape (mirror).
        self.decoder = DCAEDecoder(
            out_channels=self.out_channels,
            block_out_channels=tuple(block_out_channels),
            layers_per_block=tuple(layers_per_block),
            block_type=tuple(block_type),
            qkv_multiscales=tuple(qkv_multiscales),
            latent_channels=self.latent_channels,
            temb_channels=self.time_emb_dim,
            in_shortcut=True,
        )

        # Adaptive τ-conditional skip gates (zero-init).
        # 3 gates — one per encoder stage; output of skip_gates_mlp is (B, n_stages).
        self.n_skips = n_stages
        self.skip_gates_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.n_skips),
        )
        nn.init.zeros_(self.skip_gates_mlp[-1].weight)
        nn.init.zeros_(self.skip_gates_mlp[-1].bias)

        # 1×1 conv projections — plain Conv2d (no spherical padding needed for
        # 1×1 kernels). SphereConv2d's pole-aware splitting expects ≥3×3.
        # Indexed by encoder-stage index, so `skip_proj[i_stage]` maps
        # `block_out_channels[i_stage]` channels.
        self.skip_proj = nn.ModuleList([
            nn.Conv2d(
                block_out_channels[i], block_out_channels[i],
                kernel_size=1, stride=1, padding=0,
            )
            for i in range(self.n_skips)
        ])

        # Per-channel residual scale (`tanh(scale_chan)` ∈ (-1, 1)).
        if residual_scale_learnable:
            self.residual_scale = nn.Parameter(
                torch.full((self.out_channels,), residual_scale_init)
            )
        else:
            self.register_buffer(
                "residual_scale", torch.full((self.out_channels,), residual_scale_init)
            )
        self.residual_scale_learnable = residual_scale_learnable

    def _crop_lat(self, x: torch.Tensor) -> torch.Tensor:
        if self.lat_crop > 0:
            return x[..., self.lat_crop : -self.lat_crop, :]
        if self.lat_crop < 0:
            pad = -self.lat_crop
            return F.pad(x, (0, 0, pad, pad), mode="replicate")
        return x

    def _uncrop_lat(self, x: torch.Tensor) -> torch.Tensor:
        if self.lat_crop > 0:
            pad = self.lat_crop
            return F.pad(x, (0, 0, pad, pad), mode="replicate")
        if self.lat_crop < 0:
            crop = -self.lat_crop
            return x[..., crop : -crop, :]
        return x

    def _encode_frame_with_skips(
        self, x_frame: torch.Tensor, static: torch.Tensor, t_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the encoder, collecting per-stage-end features for skip use.

        Iteration over `encoder.down_blocks` strictly follows the construction
        layout: for each stage `i`, `n_layers_per_block[i]` consecutive
        ResBlock/EfficientViTBlock entries, then (if not the final stage) one
        `DCDownBlock2d`. The skip feature for stage `i` is the OUTPUT of the
        last layer block (BEFORE the trailing downsample).

        Returns (latent_z, [feat_stage_0, feat_stage_1, ..., feat_stage_{N-1}]).
        Spatial size of feat_stage_i is the encoder input cropped by `lat_crop`
        and reduced by ×2^{i+1} per stage (conv_in already strides by 2).
        """
        # Concat static frame-by-frame (static is shared across both frames).
        if static.ndim == 3:
            static = static.unsqueeze(0).expand(x_frame.size(0), -1, -1, -1)
        x_in = torch.cat([x_frame, static], dim=1)
        x_in = self._crop_lat(x_in)
        h = self.encoder.conv_in(x_in)

        feats: List[torch.Tensor] = []
        idx = 0
        for n_lay, has_down in self._get_stage_layout():
            for _ in range(n_lay):
                h = self.encoder.down_blocks[idx](h, t_emb)
                idx += 1
            feats.append(h)                                    # skip = stage end
            if has_down:
                h = self.encoder.down_blocks[idx](h, t_emb)
                idx += 1

        # Encoder output (latent) — mirror DCAEEncoder.forward's out_shortcut path.
        if self.encoder.out_shortcut:
            x_short = h.unflatten(1, (-1, self.encoder.out_shortcut_average_group_size))
            x_short = x_short.mean(dim=2)
            z = self.encoder.conv_out(h) + x_short
        else:
            z = self.encoder.conv_out(h)
        return z, feats

    def _get_stage_layout(self) -> List[Tuple[int, bool]]:
        """Return [(n_layers_in_stage, has_trailing_downsample), ...] derived from
        the construction-time block_out_channels / layers_per_block.

        Cached on first call.
        """
        if hasattr(self, "_stage_layout_cache"):
            return self._stage_layout_cache
        # `encoder._n_stages` was set by DCAEEncoder; if not present, infer.
        # Safer: read from self.encoder.down_blocks types — DCDownBlock2d marks a
        # downsample, anything else is a layer block.
        layout: List[Tuple[int, bool]] = []
        n_acc = 0
        for blk in self.encoder.down_blocks:
            if isinstance(blk, DCDownBlock2d):
                layout.append((n_acc, True))
                n_acc = 0
            else:
                n_acc += 1
        if n_acc > 0:
            layout.append((n_acc, False))
        self._stage_layout_cache = layout
        return layout

    def _decode_with_skips(
        self, z: torch.Tensor, skips: List[torch.Tensor], t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Run decoder; before each post-upsample stage, inject τ-gated skip.

        Decoder iterates UP from deepest stage. Up-block layout mirrors encoder
        in reverse: for stages i = N-1 .. 0:
            (1 upsample if i < N-1) + (n_lay layer blocks at stage i)
        At stage-end (after n_lay layers) we add gate_i · skip_proj_i(skip_i).
        """
        # Compute τ-conditional gates: shape (B, n_skips).
        gates = torch.tanh(self.skip_gates_mlp(t_emb))  # (B, n_skips)
        # Decoder in_shortcut handling first.
        if self.decoder.in_shortcut:
            x_short = z.repeat_interleave(self.decoder.in_shortcut_repeats, dim=1)
            h = self.decoder.conv_in(z) + x_short
        else:
            h = self.decoder.conv_in(z)
        # Walk up_blocks; track stage index and apply skip at stage end.
        stage_layout = self._get_stage_layout()  # encoder layout
        # Decoder up_blocks layout (reversed): for i = N-1 down to 0,
        # if i < N-1 → 1 upsample; then n_layers layer blocks at stage i.
        idx_up = 0
        import os
        debug = os.environ.get("WB_HYBRID_DEBUG") == "1"
        if debug:
            print(f"[decode] init h={tuple(h.shape)} z={tuple(z.shape)}", flush=True)
        for k, (n_lay, has_down) in enumerate(reversed(stage_layout)):
            # k=0 → deepest stage (i = N-1). No leading upsample for first one.
            i_stage = (len(stage_layout) - 1) - k
            if k > 0:
                # Upsample first.
                h = self.decoder.up_blocks[idx_up](h, t_emb)
                idx_up += 1
                if debug:
                    print(f"[decode] k={k} after up h={tuple(h.shape)}", flush=True)
            # Layer blocks for this stage.
            for _ in range(n_lay):
                h = self.decoder.up_blocks[idx_up](h, t_emb)
                idx_up += 1
            if debug:
                print(f"[decode] k={k} after {n_lay} blocks h={tuple(h.shape)}", flush=True)
            # τ-gated skip injection (after stage's blocks complete).
            skip = skips[i_stage]
            skip_proj_i = self.skip_proj[i_stage]
            gate = gates[:, i_stage].view(-1, 1, 1, 1)
            if debug:
                print(f"[decode] k={k} skip[{i_stage}]={tuple(skip.shape)}", flush=True)
            h = h + gate * skip_proj_i(skip)
        # Final norm/act/conv.
        h = self.decoder.norm_out(h.movedim(1, -1)).movedim(-1, 1)
        h = self.decoder.conv_act(h)
        h = self.decoder.conv_out(h)
        return h

    def forward(
        self, x0: torch.Tensor, xT: torch.Tensor, tau: torch.Tensor,
        cond: Optional[torch.Tensor] = None, static: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass.

        Args:
            x0, xT: (B, C, H, W) anchor frames in normalised stats.
            tau:    (B,) or (B, 1) hour value in [0, max_tau_hours).
            static: (3, H, W) or (B, 3, H, W) static geographic fields.
            cond:   unused (kept for API compatibility).

        Returns (x_hat, aux):
            x_hat: (B, C, H, W)
            aux:   {"decoder_out", "x_bilinear", "z0", "zT", "z_fused", "residual_scale"}
        """
        B = x0.size(0)
        if static is None:
            static = x0.new_zeros((self.n_static_features, x0.size(-2), x0.size(-1)))
        # τ pathway.
        t_emb = self.time_mlp(tau.view(-1).float())   # (B, time_emb_dim)
        # Bilinear scaffold.
        tau_ = tau.view(-1, 1, 1, 1).float() / float(self.max_tau_hours)
        x_bilinear = (1.0 - tau_) * x0 + tau_ * xT
        # Encode each frame independently (shared weights), gathering skips.
        z0, skips0 = self._encode_frame_with_skips(x0, static, t_emb)
        zT, skipsT = self._encode_frame_with_skips(xT, static, t_emb)
        # Cross-frame attention on bottleneck latents.
        z_fused = self.cross_attn(z0, zT, t_emb)
        # Blend skips at feature level (mirror the bilinear blend).
        skips_blended = [
            (1.0 - tau_) * s0 + tau_ * sT for s0, sT in zip(skips0, skipsT)
        ]
        # Decode with τ-gated skip injection.
        decoder_out = self._decode_with_skips(z_fused, skips_blended, t_emb)
        decoder_out = self._uncrop_lat(decoder_out)
        if self.direct_prediction:
            x_hat = decoder_out
            scale_t = None
        else:
            scale = torch.tanh(self.residual_scale)
            # Apply floor symmetrically.
            if self.residual_scale_floor > 0:
                sign = torch.sign(scale.detach()) + (scale.detach() == 0).float()
                scale = torch.maximum(scale.abs(),
                                       torch.tensor(self.residual_scale_floor,
                                                    device=scale.device)) * sign
            scale_t = scale.view(1, -1, 1, 1)
            if self.residual_clip is not None:
                decoder_out = decoder_out.clamp(-float(self.residual_clip),
                                                  float(self.residual_clip))
            x_hat = x_bilinear + scale_t * decoder_out
        aux = {
            "decoder_out": decoder_out,
            "x_bilinear": x_bilinear,
            "z0": z0, "zT": zT, "z_fused": z_fused,
            "residual_scale": scale_t if scale_t is not None else torch.zeros(1, device=x0.device),
        }
        return x_hat, aux


# Backward compatibility for checkpoints and legacy Hydra model names.
WeatherBridgeHybridModel = WeatherDCAEHybridModel
