"""DC-AE Skip + bilinear-guidance injection at each decoder stage.

Motivation:
  Current pipeline: x_hat = bilinear + tanh(α)·v_θ. Bilinear visible only at
  output. But bilinear is a strong, smooth predictor — decoder benefits from
  seeing it AT EACH SCALE while up-sampling, not just at final layer.

  At each decoder upsample boundary, downsample bilinear to current resolution
  and add via zero-init 1×1 conv adapter:

      bilinear_at_scale = F.interpolate(bilinear_full, size=(H_dec, W_dec))
      decoder_feat = decoder_feat + adapter_i(bilinear_at_scale)

  Adapter is zero-init → strict superset of v9 Skip; learns graded use.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dcae import DCDownBlock2d, DCUpBlock2d
from .dcae_adaln_skip_model import WeatherDCAEAdaLNSkipModel


class WeatherDCAEBilinearXAttnModel(WeatherDCAEAdaLNSkipModel):
    """DC-AE Skip + per-scale bilinear injection. Strictly extends parent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Adapters: one per decoder up-sample boundary + one at deepest stage.
        # Channels match each stage's `block_out_channels[i]`.
        # Input to adapter is bilinear (C=out_channels) downsampled to that res.
        out_ch = self.decoder.conv_out.out_channels
        n_stages = len(self.block_out_channels)
        self.bilinear_adapters = nn.ModuleList()
        for i in range(n_stages):
            ch = self.block_out_channels[i]
            adapter = nn.Conv2d(out_ch, ch, kernel_size=1, bias=True)
            # Zero-init → starts as identity to parent v9 Skip
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)
            self.bilinear_adapters.append(adapter)

    def _decode_with_skips_and_bilinear(
        self,
        z: torch.Tensor,
        skips: list,
        t_emb: torch.Tensor,
        bilinear_full: torch.Tensor,
    ) -> torch.Tensor:
        """Decoder forward injecting bilinear at each scale via zero-init adapters."""
        if self.decoder.in_shortcut:
            shortcut = z.repeat_interleave(self.decoder.in_shortcut_repeats, dim=1)
            h = self.decoder.conv_in(z) + shortcut
        else:
            h = self.decoder.conv_in(z)

        # Inject bilinear at deepest stage (matching encoder bottleneck).
        deepest_idx = len(self.bilinear_adapters) - 1
        bilinear_deep = F.interpolate(bilinear_full, size=h.shape[-2:], mode="bilinear", align_corners=False)
        h = h + self.bilinear_adapters[deepest_idx](bilinear_deep)

        skip_ptr = len(skips)
        # Track which "logical stage" we're at: stage indices for shallower stages.
        # As we walk decoder up_blocks, each DCUpBlock takes us to the next shallower stage.
        # Initial logical stage = n_stages-1 (deepest). After each DCUpBlock: stage--.
        current_logical_stage = len(self.bilinear_adapters) - 1

        for blk in self.decoder.up_blocks:
            if isinstance(blk, DCUpBlock2d):
                h = blk(h, t_emb)
                current_logical_stage -= 1
                skip_ptr -= 1
                if 0 <= skip_ptr < len(skips):
                    skip_feat = skips[skip_ptr]
                    if self.skip_lateral_rank > 0:
                        skip_feat = skip_feat + self.skip_laterals[skip_ptr](skip_feat)
                    gate = torch.tanh(self.skip_gates[skip_ptr])
                    h = h + gate * skip_feat
                # Inject bilinear at this resolution
                if 0 <= current_logical_stage < len(self.bilinear_adapters):
                    bil = F.interpolate(bilinear_full, size=h.shape[-2:], mode="bilinear", align_corners=False)
                    h = h + self.bilinear_adapters[current_logical_stage](bil)
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

        # Bilinear baseline at full resolution
        x_bilinear_full = (1.0 - tau_) * x0 + tau_ * xT

        z_tau, skips = self._encode_with_skips(self._crop_lat(x_in), t_emb)
        decoder_out_raw = self._decode_with_skips_and_bilinear(
            z_tau, skips, t_emb, self._crop_lat(x_bilinear_full)
        )
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

        if self.direct_prediction:
            x_hat = decoder_out
        else:
            x_hat = x_bilinear_full + residual_scale * decoder_out

        skip_gates_detached = [torch.tanh(g).detach() for g in self.skip_gates]
        return x_hat, {
            "z_tau": z_tau,
            "d0": None,
            "d1": None,
            "decoder_out": decoder_out,
            "x_bilinear": x_bilinear_full,
            "residual_scale": residual_scale.detach(),
            "t_emb": t_emb,
            "skip_gates": skip_gates_detached,
        }
