"""PixelAttn-VFI backbone, extracted for inference.

Upstream this lives inside a PyTorch Lightning training script
(``legacy/scripts/train_atm_vfi_12h_oddskip.py``). Only the ``nn.Module``
half is needed to run the model, so the Lightning wrapper, dataset classes
and trainer are left behind and this file depends on torch alone.

The released 6 h checkpoint has asymmetric I/O: its encoder was trained on
``[x; static]`` (27 channels) while the residual it predicts stays 24. That
rewrite is applied by ``weatherbridge._loader`` rather than baked in here, so
the class below stays faithful to the trained definition.

Code is reproduced verbatim from the training script; only the imports and
this docstring differ.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    )


class CrossFrameAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, h0, hT):
        B, C, H, W = h0.shape
        t0 = h0.flatten(2).transpose(1, 2)
        tT = hT.flatten(2).transpose(1, 2)
        tokens = torch.cat([t0, tT], dim=1)
        tokens = self.norm(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + attn_out
        out_t = (tokens[:, :H*W] + tokens[:, H*W:]) / 2
        return out_t.transpose(1, 2).reshape(B, C, H, W)


class AdaLNZero(nn.Module):
    def __init__(self, dim, time_dim=128):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, dim * 3))
        nn.init.normal_(self.time_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.time_mlp[-1].bias)
        with torch.no_grad():
            self.time_mlp[-1].weight[2 * dim:].zero_()
        self.time_dim = time_dim
    def time_embed(self, tau):
        half = self.time_dim // 2
        device = tau.device
        freqs = torch.exp(torch.linspace(0, -math.log(10000), half, device=device))
        args = tau.view(-1, 1) * freqs.view(1, -1)
        return torch.cat([args.sin(), args.cos()], dim=-1)
    def forward(self, h, tau):
        h_n = self.norm(h)
        t = self.time_embed(tau.view(-1).float())
        g, b, gate = self.time_mlp(t).chunk(3, dim=-1)
        g = g.view(-1, h.size(1), 1, 1)
        b = b.view(-1, h.size(1), 1, 1)
        gate = gate.view(-1, h.size(1), 1, 1)
        return h + gate * ((1 + g) * h_n + b)


class PixelAttentionVFINet(nn.Module):
    def __init__(self, in_ch=24, hidden=64, n_levels=3, time_dim=128):
        super().__init__()
        self.in_ch = in_ch
        self.frame_encoder = nn.ModuleList()
        ch = [hidden * (2 ** i) for i in range(n_levels + 1)]
        self.frame_encoder.append(conv_block(in_ch, ch[0], stride=1))
        for i in range(n_levels):
            self.frame_encoder.append(conv_block(ch[i], ch[i+1], stride=2))
        self.attn = CrossFrameAttention(ch[-1])
        self.adaln = AdaLNZero(ch[-1], time_dim=time_dim)
        self.decoder = nn.ModuleList()
        for i in range(n_levels):
            self.decoder.append(nn.ConvTranspose2d(ch[-(i+1)], ch[-(i+2)],
                                                    kernel_size=4, stride=2, padding=1))
            self.decoder.append(conv_block(ch[-(i+2)] * 2, ch[-(i+2)]))
        self.out_conv = nn.Conv2d(ch[0], in_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight); nn.init.zeros_(self.out_conv.bias)
        self.scale = nn.Parameter(torch.full((in_ch,), 0.1))

    def encode_frame(self, x):
        feats = []
        h = x
        for blk in self.frame_encoder:
            h = blk(h); feats.append(h)
        return feats

    def forward(self, x_0, x_T, tau):
        tau_b = tau.view(-1, 1, 1, 1) if tau.dim() <= 2 else tau
        # 12h bilinear scaffold uses tau ∈ [0,1] (already rescaled)
        x_bilinear = (1.0 - tau_b) * x_0 + tau_b * x_T

        f0 = self.encode_frame(x_0)
        fT = self.encode_frame(x_T)
        h = self.attn(f0[-1], fT[-1])
        h = self.adaln(h, tau.view(-1))
        n_levels = len(self.decoder) // 2
        for i in range(n_levels):
            up = self.decoder[2 * i]
            blk = self.decoder[2 * i + 1]
            h = up(h)
            skip = (f0[-(i+2)] + fT[-(i+2)]) / 2
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            h = blk(torch.cat([h, skip], dim=1))
        if h.shape[-2:] != x_0.shape[-2:]:
            h = F.interpolate(h, size=x_0.shape[-2:], mode='bilinear', align_corners=False)
        delta = self.out_conv(h)
        s = torch.tanh(self.scale).view(1, -1, 1, 1)
        return x_bilinear + s * delta

