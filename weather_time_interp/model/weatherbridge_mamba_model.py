"""WeatherBridge-Mamba — VFIMamba-style Mixed-SSM backbone for temporal
interpolation of reanalysis fields.

Replaces the sphere-conv DC-AE backbone (an autoencoder-compressor, not a
VFI network) with a modern efficient VFI design following VFIMamba
(NeurIPS 2024): plain-conv feature pyramids + a Mixed-SSM block that
interleaves tokens from the two anchor frames and applies a selective
state-space scan across multiple directions to model inter-frame
dynamics in **linear** O(N) time. This gives a global receptive field
(needed for synoptic-scale weather structure) at a fraction of the
sphere-conv DC-AE cost.

WeatherBridge identity retained:
  * bilinear scaffold + tanh(scale)·residual output
  * τ conditioning via AdaLN-Zero
  * (spectral loss applied in the trainer, not here)

Mamba backend:
  * If ``mamba_ssm`` is importable, uses the fast CUDA ``Mamba`` block.
  * Otherwise falls back to a pure-PyTorch selective-scan (``PTScan``)
    so the model runs anywhere (slower, but correct).

forward(x0, xT, tau, cond, static=None) -> (x_hat, aux_dict)
so it drops into the existing eval / bare-blob loader.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _MambaCUDA
    _HAS_MAMBA = True
except Exception:  # pragma: no cover
    _MambaCUDA = None
    _HAS_MAMBA = False


# ---------------------------------------------------------------- τ embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, base_period: float = 16.0):
        super().__init__()
        self.dim = dim
        self.base_period = base_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, -math.log(self.base_period), half, device=t.device))
        args = t.view(-1, 1).float() * freqs.view(1, -1)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, dim: int = 256, freq_dim: int = 128, base_period: float = 16.0):
        super().__init__()
        self.emb = SinusoidalPosEmb(freq_dim, base_period)
        self.net = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(),
                                  nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(self.emb(t))


class AdaLNZero(nn.Module):
    """Zero-init AdaLN modulation on (B, C, H, W) conditioned on τ embedding."""

    def __init__(self, dim: int, time_dim: int = 256):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim * 3))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        g, b, gate = self.mlp(temb).chunk(3, dim=-1)
        g = g[:, :, None, None]
        b = b[:, :, None, None]
        gate = gate[:, :, None, None]
        return h + gate * ((1 + g) * self.norm(h) + b)


# ---------------------------------------------------------------- SSM block
class PTScan(nn.Module):
    """Pure-PyTorch fallback selective-scan (diagonal SSM), sequential.

    A lightweight S6-style scan: input-dependent gate + decay. Not as fast
    as the CUDA kernel but correct and dependency-free. Operates on a
    (B, L, D) token sequence.
    """

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dt_proj = nn.Linear(dim, dim)
        self.A_log = nn.Parameter(torch.log(torch.rand(dim, d_state) + 0.5))
        self.B_proj = nn.Linear(dim, d_state)
        self.C_proj = nn.Linear(dim, d_state)
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(x_in))               # (B, L, D)
        A = -torch.exp(self.A_log)                         # (D, S)
        Bm = self.B_proj(x_in)                             # (B, L, S)
        Cm = self.C_proj(x_in)                             # (B, L, S)
        # discretize + scan
        dA = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, D, self.d_state))  # (B,L,D,S)
        dB = dt.unsqueeze(-1) * Bm.unsqueeze(2)                            # (B,L,D,S)
        h = x_in.new_zeros(B, D, self.d_state)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_in[:, t].unsqueeze(-1)
            y = torch.einsum("bds,bs->bd", h, Cm[:, t])
            ys.append(y)
        y = torch.stack(ys, dim=1) + x_in * self.D.view(1, 1, D)
        y = y * F.silu(z)
        return self.out_proj(y)


class MixedSSMBlock(nn.Module):
    """VFIMamba Mixed-SSM: interleave tokens of two frames, multi-dir scan.

    Given f0, fT (B, C, H, W), flatten and interleave along the sequence,
    run a forward + reverse SSM scan (bi-directional), de-interleave, and
    return the fused per-frame features. Linear in H·W.
    """

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if _HAS_MAMBA:
            self.ssm_f = _MambaCUDA(d_model=dim, d_state=d_state, d_conv=4, expand=2)
            self.ssm_b = _MambaCUDA(d_model=dim, d_state=d_state, d_conv=4, expand=2)
            self._cuda = True
        else:
            self.ssm_f = PTScan(dim, d_state)
            self.ssm_b = PTScan(dim, d_state)
            self._cuda = False

    def forward(self, f0: torch.Tensor, fT: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f0.shape
        t0 = f0.flatten(2).transpose(1, 2)   # (B, HW, C)
        tT = fT.flatten(2).transpose(1, 2)
        # interleave: [t0_0, tT_0, t0_1, tT_1, ...]
        inter = torch.stack([t0, tT], dim=2).reshape(B, 2 * H * W, C)
        inter = self.norm(inter)
        y = self.ssm_f(inter) + torch.flip(self.ssm_b(torch.flip(inter, [1])), [1])
        inter = inter + y
        # de-interleave and average the two frames
        de = inter.reshape(B, H * W, 2, C)
        fused = de.mean(dim=2)               # (B, HW, C)
        return fused.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------- conv blocks
def conv_block(ci, co, stride=1):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, stride, 1, padding_mode="circular"),
        nn.GroupNorm(min(8, co), co), nn.SiLU(),
        nn.Conv2d(co, co, 3, 1, 1, padding_mode="circular"),
        nn.GroupNorm(min(8, co), co), nn.SiLU(),
    )


class WeatherBridgeMambaModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 24,
        n_static_features: int = 3,
        hidden: int = 64,
        n_levels: int = 3,
        d_state: int = 16,
        time_emb_dim: int = 256,
        residual_scale_init: float = 0.10,
        lat_crop: int = 0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_static_features = int(n_static_features)
        self.hidden = int(hidden)
        self.n_levels = int(n_levels)
        self.d_state = int(d_state)
        self.time_emb_dim = int(time_emb_dim)
        self.lat_crop = int(lat_crop)

        self.time_mlp = TimeMLP(time_emb_dim)

        ch = [hidden * (2 ** i) for i in range(n_levels + 1)]
        # Shared frame encoder (plain circular conv — fast, sphere-aware in lon)
        enc_in = in_channels + n_static_features
        self.frame_encoder = nn.ModuleList()
        self.frame_encoder.append(conv_block(enc_in, ch[0], stride=1))
        for i in range(n_levels):
            self.frame_encoder.append(conv_block(ch[i], ch[i + 1], stride=2))

        # Mixed-SSM inter-frame block at bottleneck + AdaLN τ-conditioning
        self.mixed_ssm = MixedSSMBlock(ch[-1], d_state=d_state)
        self.adaln = AdaLNZero(ch[-1], time_dim=time_emb_dim)

        # Decoder with frame-averaged skips (UNet-style synthesis)
        self.decoder = nn.ModuleList()
        for i in range(n_levels):
            self.decoder.append(
                nn.ConvTranspose2d(ch[-(i + 1)], ch[-(i + 2)], 4, 2, 1))
            self.decoder.append(conv_block(ch[-(i + 2)] * 2, ch[-(i + 2)]))
        self.out_conv = nn.Conv2d(ch[0], out_channels, 3, 1, 1, padding_mode="circular")
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.scale = nn.Parameter(torch.full((out_channels,), float(residual_scale_init)))

    def _encode(self, x):
        feats = []
        h = x
        for blk in self.frame_encoder:
            h = blk(h)
            feats.append(h)
        return feats

    def _prep_static(self, static, B, device):
        if self.n_static_features <= 0:
            return None
        if static.dim() == 3:
            static = static.unsqueeze(0)
        if static.size(0) != B:
            static = (static.repeat_interleave(B // static.size(0), 0)
                      if B % static.size(0) == 0 else static.expand(B, -1, -1, -1))
        return static.to(device)

    def forward(self, x0, xT, tau, cond=None, static=None):
        if tau.dim() == 1:
            tau_b = tau.view(-1, 1, 1, 1)
        elif tau.dim() == 2:
            tau_b = tau.view(tau.size(0), 1, 1, 1)
        else:
            tau_b = tau
        B = x0.size(0)
        temb = self.time_mlp(tau.view(-1))
        if temb.size(0) != B and B % temb.size(0) == 0:
            temb = temb.repeat_interleave(B // temb.size(0), 0)

        x_bilinear = (1.0 - tau_b) * x0 + tau_b * xT

        st = self._prep_static(static, B, x0.device)
        def _in(x):
            return torch.cat([x, st], dim=1) if st is not None else x
        f0 = self._encode(_in(x0))
        fT = self._encode(_in(xT))

        # Bottleneck: Mixed-SSM inter-frame fusion + τ AdaLN
        h = self.mixed_ssm(f0[-1], fT[-1])
        h = self.adaln(h, temb)

        for i in range(self.n_levels):
            up = self.decoder[2 * i]
            blk = self.decoder[2 * i + 1]
            h = up(h)
            skip = (f0[-(i + 2)] + fT[-(i + 2)]) / 2
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear",
                                   align_corners=False)
            h = blk(torch.cat([h, skip], dim=1))
        if h.shape[-2:] != x0.shape[-2:]:
            h = F.interpolate(h, size=x0.shape[-2:], mode="bilinear",
                               align_corners=False)
        delta = self.out_conv(h)
        s = torch.tanh(self.scale).view(1, -1, 1, 1)
        x_hat = x_bilinear + s * delta
        return x_hat, {"x_bilinear": x_bilinear, "delta": delta,
                        "mamba_backend": "cuda" if _HAS_MAMBA else "pytorch"}
