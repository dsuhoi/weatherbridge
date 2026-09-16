"""DC-AE + AdaLN-Zero (v9) + U-Net-style gated skip connections.

Extends `dcae_adaln_model.WeatherDCAEAdaLNModel` with skip connections between
encoder and decoder at each downsample/upsample boundary. Skips use additive
gated injection with zero-init `tanh(α)` gate — model starts identical to v9
AdaLN, and learns to use skips if helpful.

Skip mechanics:
  encoder forward: save `h_before_downsample` at each `DCDownBlock2d` boundary
  decoder forward: after each `DCUpBlock2d`, inject `+ tanh(α_i) · skip_i`
  α_i: learnable scalar per skip, init=0 (zero-init → strict superset of v9)

Motivation:
  - DC-AE was designed for image-gen compression (Sana). 16× channel bottleneck
    (512→32) kills high-frequency detail critical for weather interpolation.
  - Interpolation output ≈ bilinear + small residual → high-freq detail must
    survive the AE bottleneck. Skip connections (U-Net, Ronneberger 2015) are
    the canonical fix.
  - Gated zero-init makes this a strict superset of v9: model can ignore skips
    if not helpful; can learn graded use if helpful.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dcae import Encoder as DCAEEncoder, Decoder as DCAEDecoder
from .dcae import DCDownBlock2d, DCUpBlock2d
from .dcae_adaln_model import SinusoidalPosEmb, TimeMLP


class WeatherDCAEAdaLNSkipModel(nn.Module):
    """DC-AE with FiLM time-cond + gated zero-init U-Net skips."""

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
        skip_gate_init: float = 0.0,
        skip_lateral_rank: int = 0,
        tau_conditional_gates: bool = False,
    ):
        super().__init__()
        # Persist every architectural kwarg so checkpoint normalizers can
        # recover them via introspection (no state_dict sniffing).
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
        self.skip_gate_init = float(skip_gate_init)
        # lat_crop semantics: >0 crop, <0 pad-replicate, ==0 no-op.
        # Previously max(0,...) silently stripped PAD intent — fixed in this revision.
        self.lat_crop = int(lat_crop)
        self.tau_conditional_gates = bool(tau_conditional_gates)
        self.residual_clip = residual_clip
        self.residual_scale_floor = float(residual_scale_floor)
        self.direct_prediction = bool(direct_prediction)
        self.n_static_features = int(n_static_features)
        self.time_emb_dim = int(time_emb_dim)
        self.skip_lateral_rank = max(0, int(skip_lateral_rank))

        self.time_mlp = TimeMLP(
            time_dim=self.time_emb_dim,
            freq_dim=int(time_freq_dim),
            base_period=float(time_base_period),
        )

        self.encoder = DCAEEncoder(
            in_channels=in_channels * 2 + self.n_static_features,
            latent_channels=latent_channels,
            temb_channels=self.time_emb_dim,
            block_type=block_type,
            block_out_channels=self.block_out_channels,
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
            block_out_channels=self.block_out_channels,
            layers_per_block=tuple(layers_per_block),
            qkv_multiscales=qkv_multiscales,
            attention_head_dim=attention_head_dim,
            upsample_block_type="pixel_shuffle",
            in_shortcut=True,
        )

        # Skip gates: one per pair of (encoder DCDownBlock, decoder DCUpBlock).
        # For 3-stage encoder (block_out_channels of length N), there are N-1 boundaries.
        n_skips = max(0, len(self.block_out_channels) - 1)
        self._n_skips = n_skips
        if self.tau_conditional_gates:
            # Novelty A: gates depend on τ via a small MLP from t_emb.
            # Output zero-init so gates start near 0 (≈ no-skip), letting model
            # learn when to engage skips during training.
            self.skip_gates_mlp = nn.Sequential(
                nn.Linear(self.time_emb_dim, 64),
                nn.GELU(),
                nn.Linear(64, n_skips),
            )
            nn.init.zeros_(self.skip_gates_mlp[-1].weight)
            nn.init.zeros_(self.skip_gates_mlp[-1].bias)
        else:
            self.skip_gates = nn.ParameterList([
                nn.Parameter(torch.tensor(float(skip_gate_init), dtype=torch.float32))
                for _ in range(n_skips)
            ])

        # Optional lateral projection (rank>0 → low-rank LoRA-style adapter).
        # When rank=0, skip is injected as-is (gated). Channel counts already match
        # by encoder/decoder symmetry for vanilla DC-AE.
        self.skip_laterals = nn.ModuleList()
        if self.skip_lateral_rank > 0:
            for i in range(n_skips):
                ch = self.block_out_channels[i]
                lateral = nn.Sequential(
                    nn.Conv2d(ch, self.skip_lateral_rank, kernel_size=1, bias=False),
                    nn.Conv2d(self.skip_lateral_rank, ch, kernel_size=1, bias=True),
                )
                # zero-init final to keep skip ≈ identity at start
                nn.init.zeros_(lateral[-1].weight)
                nn.init.zeros_(lateral[-1].bias)
                self.skip_laterals.append(lateral)

        init_scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if residual_scale_learnable and not self.direct_prediction:
            self.residual_scale = nn.Parameter(init_scale)
        else:
            self.register_buffer("residual_scale", init_scale)

    def _crop_lat(self, x: torch.Tensor) -> torch.Tensor:
        """If lat_crop > 0: crop |lat_crop| rows symmetrically top+bottom.
        If lat_crop < 0: PAD |lat_crop| rows symmetrically top+bottom via replicate.
        Use negative for sphere-respecting size adjustment without data loss.
        """
        if self.lat_crop == 0:
            return x
        n = abs(int(self.lat_crop))
        if self.lat_crop > 0:
            height = x.size(-2)
            if height <= self.lat_crop:
                raise ValueError(f"lat_crop={self.lat_crop} too large for height={height}")
            crop_top = n // 2
            crop_bottom = n - crop_top
            if crop_bottom == 0:
                return x[:, :, crop_top:]
            return x[:, :, crop_top:-crop_bottom]
        # PAD mode: lat_crop < 0
        pad_top = n // 2
        pad_bottom = n - pad_top
        return F.pad(x, (0, 0, pad_top, pad_bottom), mode="replicate")

    def _encode_with_skips(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple:
        """Run encoder, capture pre-downsample features at each DCDownBlock boundary."""
        h = self.encoder.conv_in(x)
        skips = []
        for blk in self.encoder.down_blocks:
            if isinstance(blk, DCDownBlock2d):
                # Save feature BEFORE downsample at this resolution.
                skips.append(h)
                h = blk(h, t_emb)
            else:
                h = blk(h, t_emb)

        if self.encoder.out_shortcut:
            shortcut = h.unflatten(1, (-1, self.encoder.out_shortcut_average_group_size)).mean(dim=2)
            h = self.encoder.conv_out(h) + shortcut
        else:
            h = self.encoder.conv_out(h)
        return h, skips

    def _decode_with_skips(self, z: torch.Tensor, skips: list, t_emb: torch.Tensor) -> torch.Tensor:
        """Run decoder, inject skips after each DCUpBlock boundary (LIFO order)."""
        if self.decoder.in_shortcut:
            shortcut = z.repeat_interleave(self.decoder.in_shortcut_repeats, dim=1)
            h = self.decoder.conv_in(z) + shortcut
        else:
            h = self.decoder.conv_in(z)

        # Decoder up_blocks list: interleaves DCUpBlock2d + ResBlock/EfficientViTBlock.
        # After each DCUpBlock2d, we are at the resolution of the NEXT (shallower)
        # encoder stage — inject the corresponding skip.
        # skips order: [stage_0_skip, stage_1_skip, ..., stage_{N-2}_skip]
        # decoder visits up_blocks in deep→shallow order; first DCUpBlock arrives
        # at stage_{N-2}, last DCUpBlock at stage_0. So inject in REVERSE order.
        skip_ptr = len(skips)  # consume from end

        # Pre-compute per-sample τ-conditional gates if enabled.
        if self.tau_conditional_gates:
            # (B, n_skips) — tanh keeps gates in (-1, 1), zero-init means gate≈0 initially.
            tau_gates = torch.tanh(self.skip_gates_mlp(t_emb))
        else:
            tau_gates = None

        for blk in self.decoder.up_blocks:
            if isinstance(blk, DCUpBlock2d):
                h = blk(h, t_emb)
                skip_ptr -= 1
                if 0 <= skip_ptr < len(skips):
                    skip_feat = skips[skip_ptr]
                    if self.skip_lateral_rank > 0:
                        skip_feat = skip_feat + self.skip_laterals[skip_ptr](skip_feat)
                    if tau_gates is not None:
                        # (B, 1, 1, 1) per-sample gate for this stage
                        gate = tau_gates[:, skip_ptr].view(-1, 1, 1, 1)
                    else:
                        gate = torch.tanh(self.skip_gates[skip_ptr])
                    h = h + gate * skip_feat
            else:
                h = blk(h, t_emb)

        return self.decoder.conv_out(h)

    def forward(self, x0, xT, tau, cond, static=None):
        if tau.dim() == 1:
            tau_ = tau.view(-1, 1, 1, 1)
        elif tau.dim() == 2:
            tau_ = tau.view(tau.size(0), 1, 1, 1)
        else:
            tau_ = tau

        t_emb = self.time_mlp(tau.view(-1))
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

        z_tau, skips = self._encode_with_skips(self._crop_lat(x_in), t_emb)
        decoder_out_raw = self._decode_with_skips(z_tau, skips, t_emb)
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

        if self.tau_conditional_gates:
            # Per-sample gates; report batch-mean per stage for logging.
            with torch.no_grad():
                _tau_gates = torch.tanh(self.skip_gates_mlp(t_emb))  # (B, n_skips)
            skip_gates_detached = [_tau_gates[:, i].mean().detach() for i in range(self._n_skips)]
        else:
            skip_gates_detached = [torch.tanh(g).detach() for g in self.skip_gates]
        return x_hat, {
            "z_tau": z_tau,
            "d0": None,
            "d1": None,
            "decoder_out": decoder_out,
            "x_bilinear": x_bilinear,
            "residual_scale": residual_scale.detach(),
            "t_emb": t_emb,
            "skip_gates": skip_gates_detached,
        }
