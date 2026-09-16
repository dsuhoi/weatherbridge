"""Hierarchical DC-AE 1.5-style spatial compression — v3.1 (4-level latent split).

Difference vs v3:
  * Latent 4-level SPLIT: z₂ (128 ch) channel-partitioned as 24/40/40/24:
      [:,  0: 24]  → planetary scale (avg_pool 8×, > 1700 km)
      [:, 24: 64]  → synoptic scale  (avg_pool 4× − pool 8×, ~880 km bandpass)
      [:, 64:104]  → mesoscale       (avg_pool 2× − pool 4×, ~440 km bandpass)
      [:,104:128]  → local-residual  (z₁ − pool 2×, sub-220 km HF)
    Aux heads (1×1 conv, zero-init) decode each group to its scale band.
    Telescoping decomposition: z₁ = planet + synop + meso + local (lossless).
    Forces inner_down to disentangle scales; structure usable downstream by
    Latent CorrDiff (e.g. only diffuse local-residual, bilinear-interp others).

Phase 2 v3.1 (~10 ep, z₂ (128, h/2, w/2) ≈ 47× compression):
  inner_down: 4×ResBlk(48) → PixelUnshuffle → 1×1 to 128 → 2×ResBlk → 6×(SelfAttn+FFN) → 2×ResBlk
  inner_up:   2×ResBlk(128) → 6×(SelfAttn+FFN) → 2×ResBlk → 1×1 → PixelShuffle → 4×ResBlk(48)
  loss = pixel + λ_lat·MSE(z₁_rec, z₁) + λ_spec·SH_hf
       + λ_split·Σₖ MSE(split_headₖ(z₂[group_k]), z₁_band_k)
"""
import os, sys, argparse, math, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')

_wti = os.environ.get("WTI_ROOT")
if not _wti:
    for c in ["/workspace/code/wti", "/home/jovyan/dsuhoi/weather_time_interpolation"]:
        if os.path.exists(os.path.join(c, "weather_time_interp")): _wti = c; break
sys.path.insert(0, _wti); os.chdir(_wti)

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import xarray as xr
import torch_harmonics as th_harm

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from diffusers.models.autoencoders.autoencoder_dc import AutoencoderDC

CH_24 = ["T1000","T925","T850","T700","U1000","U925","U850","U700",
         "V1000","V925","V850","V700","Q1000","Q925","Q850","Q700",
         "Z1000","Z925","Z850","Z700","t2m","u10","v10","mslp"]

CH_WEIGHTS = {
    'T1000': 1.0, 'T925': 1.0, 'T850': 1.0, 'T700': 1.0,
    'U1000': 0.5, 'U925': 0.7, 'U850': 0.9, 'U700': 1.2,
    'V1000': 0.5, 'V925': 0.7, 'V850': 0.9, 'V700': 1.2,
    'Q1000': 1.5, 'Q925': 1.3, 'Q850': 1.0, 'Q700': 0.6,
    'Z1000': 1.0, 'Z925': 0.8, 'Z850': 0.5, 'Z700': 0.3,
    't2m': 3.0, 'u10': 0.77, 'v10': 0.66, 'mslp': 1.5,
}


class TISRLoader:
    """fp16-scaled (×1e-5) memmap reader."""
    def __init__(self, cache_dir, years):
        self.cache_dir = Path(cache_dir)
        self.maps, self.start_times, self.scales = {}, {}, {}
        self.dt_ns = int(3600e9)
        for y in years:
            meta = json.load(open(self.cache_dir / f"tisr_{y}.json"))
            self.maps[y] = np.memmap(self.cache_dir / f"tisr_{y}.bin",
                                      dtype='float16', mode='r',
                                      shape=tuple(meta['shape']))
            self.start_times[y] = np.datetime64(meta['start_time'], 'ns')
            self.scales[y] = float(meta.get('scale', 1.0))
        self.mean = 1071964.4; self.std = 1437247.4

    def get(self, year, t_unix_ns):
        ofs = t_unix_ns - self.start_times[year].astype('int64')
        idx = max(0, min(self.maps[year].shape[0] - 1, int(ofs // self.dt_ns)))
        return self.maps[year][idx].astype(np.float32) / self.scales[year]


class SingleFrameWithTISR(Dataset):
    def __init__(self, base, tisr_loader, years, max_tau=6):
        self.base = base; self.tisr = tisr_loader
        self.year_t0_unix = []
        for (y, t0, tau_h, tau_n) in base.index:
            t0_abs = np.datetime64(f'{y}-01-01T00:00:00', 'ns') + \
                     np.timedelta64(int(t0), 'h')
            self.year_t0_unix.append((y, t0_abs.astype('int64'), tau_h))
    def __len__(self): return len(self.base) * 2
    def __getitem__(self, idx):
        b, c = divmod(idx, 2)
        out = self.base[b]
        key = 'x0' if c == 0 else ('xT' if 'xT' in out else 'x1')
        x = out[key][..., :24, :, :].clone()
        static = out['static'] if 'static' in out else None
        y, t0_unix, _ = self.year_t0_unix[b]
        t_unix = t0_unix + (0 if c == 0 else int(self.base.max_tau_hours) * int(3600e9))
        tisr = self.tisr.get(y, t_unix)
        tisr_norm = (tisr - self.tisr.mean) / self.tisr.std
        tisr_t = torch.from_numpy(tisr_norm).float().unsqueeze(0)
        item = {'x': x, 'tisr': tisr_t}
        if static is not None: item['static'] = static
        return item


def _resblock(ch):
    return nn.Sequential(
        nn.GroupNorm(8, ch), nn.SiLU(),
        nn.Conv2d(ch, ch, 3, padding=1),
        nn.GroupNorm(8, ch), nn.SiLU(),
        nn.Conv2d(ch, ch, 3, padding=1),
    )


class _ResBlk(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = _resblock(ch)
    def forward(self, x): return x + self.body(x)


class _ChannelFFN(nn.Module):
    """Per-pixel MLP: 1×1 expand → SiLU → 1×1 contract. Pre-norm + residual."""
    def __init__(self, ch, ratio=2):
        super().__init__()
        h = ch * ratio
        self.body = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, h, 1), nn.SiLU(),
            nn.Conv2d(h, ch, 1),
        )
    def forward(self, x): return x + self.body(x)


class _SelfAttn2D(nn.Module):
    """Cheap multi-head self-attention on (B, C, H, W) — pre-norm + residual.

    For z₂=(23, 46): 1058 tokens × ch=128 → ~0.14 GFLOPs per call, trivial.
    """
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.heads = heads
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]   # (B, heads, c_h, HW)
        scale = (C // self.heads) ** -0.5
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class _AttnFFNBlock(nn.Module):
    """Transformer-style block: SelfAttn2D → ChannelFFN, both residual."""
    def __init__(self, ch, heads, ffn_ratio):
        super().__init__()
        self.attn = _SelfAttn2D(ch, heads=heads)
        self.ffn = _ChannelFFN(ch, ratio=ffn_ratio)
    def forward(self, x): return self.ffn(self.attn(x))


class InnerDown(nn.Module):
    """Capacious 4th stage: 4×ResBlk @ in_ch → PixelUnshuffle → 1×1 to out_ch
    → 2×ResBlk → 2× (SelfAttn + ChannelFFN) → 2×ResBlk.
    (48, h, w) → (out_ch, h/2, w/2)."""
    def __init__(self, in_ch=48, out_ch=128, attn_heads=4, ffn_ratio=4,
                 n_attn_blocks=2, n_outer_res=4, n_inner_res=2):
        super().__init__()
        layers = [_ResBlk(in_ch) for _ in range(n_outer_res)]
        layers += [nn.PixelUnshuffle(2), nn.Conv2d(in_ch * 4, out_ch, 1)]
        layers += [_ResBlk(out_ch) for _ in range(n_inner_res)]
        layers += [_AttnFFNBlock(out_ch, attn_heads, ffn_ratio)
                   for _ in range(n_attn_blocks)]
        layers += [_ResBlk(out_ch) for _ in range(n_inner_res)]
        self.body = nn.Sequential(*layers)
    def forward(self, x): return self.body(x)


class InnerUp(nn.Module):
    """Inverse: 2×ResBlk → 2× (SelfAttn + ChannelFFN) → 2×ResBlk → 1×1
    → PixelShuffle → 4×ResBlk @ out_ch.
    (in_ch, h, w) → (out_ch, h*2, w*2)."""
    def __init__(self, in_ch=128, out_ch=48, attn_heads=4, ffn_ratio=4,
                 n_attn_blocks=2, n_outer_res=4, n_inner_res=2):
        super().__init__()
        layers = [_ResBlk(in_ch) for _ in range(n_inner_res)]
        layers += [_AttnFFNBlock(in_ch, attn_heads, ffn_ratio)
                   for _ in range(n_attn_blocks)]
        layers += [_ResBlk(in_ch) for _ in range(n_inner_res)]
        layers += [nn.Conv2d(in_ch, out_ch * 4, 1), nn.PixelShuffle(2)]
        layers += [_ResBlk(out_ch) for _ in range(n_outer_res)]
        self.body = nn.Sequential(*layers)
    def forward(self, x): return self.body(x)


class SHHighFreqLoss(nn.Module):
    """Spherical Harmonics high-freq loss (l > threshold).

    Wavelength at degree l ≈ 2πR / l. For l > 30 on 0.5° grid → wavelengths
    < ~1300 km (sub-synoptic detail).
    """
    def __init__(self, nlat=360, nlon=720, lmax=180, l_threshold=30):
        super().__init__()
        self.sht = th_harm.RealSHT(nlat=nlat, nlon=nlon, lmax=lmax, mmax=lmax,
                                    grid='equiangular').float()
        l_idx = torch.arange(lmax)
        hf_mask = (l_idx > l_threshold).float()
        self.register_buffer('hf_mask', hf_mask.view(1, 1, -1, 1))
        self.norm_factor = lmax / max(1.0, float(lmax - l_threshold))

    def forward(self, x_pred, x_target):
        """x: (B, C, nlat, nlon). Returns scalar L1 over HF magnitudes."""
        with torch.amp.autocast('cuda', enabled=False):
            A_p = self.sht(x_pred.float())   # (B, C, lmax, mmax) complex
            A_t = self.sht(x_target.float())
        diff = (A_p.abs() - A_t.abs()).abs() * self.hf_mask
        return diff.mean() * self.norm_factor


def _smooth_z1(z1, factor):
    """Average-pool z₁ by `factor` then bilinearly upsample back to z₁ size.
    Result is a low-pass version of z₁ at scale `factor`."""
    z_small = F.avg_pool2d(z1, kernel_size=factor, stride=factor)
    return F.interpolate(z_small, size=z1.shape[-2:], mode='bilinear',
                         align_corners=False)


def decompose_z1_4lvl(z1):
    """4-level lossless scale decomposition of z₁.

    z₁ = z₁_planetary + z₁_synoptic + z₁_meso + z₁_local
       = pool8↑       + (pool4↑ − pool8↑) + (pool2↑ − pool4↑) + (z₁ − pool2↑)

    Each band represents a half-octave wider scale than the next. Telescoping
    construction guarantees exact reconstruction (sum identity).
    """
    s2 = _smooth_z1(z1, 2)
    s4 = _smooth_z1(z1, 4)
    s8 = _smooth_z1(z1, 8)
    z_planet = s8
    z_synop  = s4 - s8
    z_meso   = s2 - s4
    z_local  = z1 - s2
    return z_planet, z_synop, z_meso, z_local


# -------------------- Main model --------------------
class HierCompressAEv2(pl.LightningModule):
    """Spatial-hierarchical DC-AE compression.

    Phase 1: 3-stage encoder (no polar crop), latent (48, 46, 92), ~32× compression.
    Phase 2: Phase 1 frozen + inner 4th stage, latent (96, 23, 46), ~64× compression.
    """
    LAT_PAD = 4   # 360 → 368, divisible by 16 (4 stages) and 8 (3 stages)
    LON_PAD = 8   # 720 → 736, circular

    def __init__(self, phase=1, phase1_ckpt=None,
                 in_channels=24,
                 latent_ch_phase1=48,
                 latent_ch_phase2=128,
                 phase1_block_out=(96, 96, 192),
                 phase1_layers=(2, 2, 2),
                 n_static=3,
                 lr=2e-4, weight_decay=1e-4,
                 lambda_spec=0.05, lambda_latent=0.5,
                 sh_l_threshold=30,
                 sh_lmax=180,
                 unfreeze_phase1=True,
                 lr_ratio_p1=0.1,
                 attn_heads=4,
                 ffn_ratio=8,
                 n_attn_blocks=6,
                 n_outer_res=4,
                 n_inner_res=2,
                 lambda_split=0.3,
                 split_groups=(24, 40, 40, 24),
                 stds=None):
        super().__init__()
        self.save_hyperparameters(ignore=['stds'])
        self.phase = phase

        # Phase 1: 3-stage AutoencoderDC
        self.ae = AutoencoderDC(
            in_channels=in_channels + 1,   # +1 TISR
            latent_channels=latent_ch_phase1,
            encoder_block_types=("ResBlock",) * len(phase1_block_out),
            decoder_block_types=("ResBlock",) * len(phase1_block_out),
            encoder_block_out_channels=tuple(phase1_block_out),
            decoder_block_out_channels=tuple(phase1_block_out),
            encoder_layers_per_block=tuple(phase1_layers),
            decoder_layers_per_block=tuple(phase1_layers),
            scaling_factor=1.0,
        )
        # Static adapter for Phase 1 decoder (3 → latent_ch_phase1, zero-init)
        self.static_adapter_p1 = nn.Conv2d(n_static, latent_ch_phase1, 1)
        nn.init.zeros_(self.static_adapter_p1.weight)
        nn.init.zeros_(self.static_adapter_p1.bias)

        if phase == 2:
            assert phase1_ckpt is not None, "phase=2 requires phase1_ckpt"
            sd = torch.load(phase1_ckpt, map_location='cpu',
                            weights_only=False)['state_dict']
            sd_p1 = {k: v for k, v in sd.items()
                     if not k.startswith('inner_down.') and not k.startswith('inner_up.')
                     and not k.startswith('static_adapter_p2.')}
            miss, unexp = self.load_state_dict(sd_p1, strict=False)
            print(f'[phase2] phase1 loaded missing={len(miss)} unexpected={len(unexp)}')
            if unfreeze_phase1:
                for p in self.ae.parameters(): p.requires_grad = True
                for p in self.static_adapter_p1.parameters(): p.requires_grad = True
                print(f'[phase2] phase1 UNFROZEN '
                      f'({sum(p.numel() for p in self.ae.parameters())/1e6:.2f}M @ lr_ratio={lr_ratio_p1})')
            else:
                for p in self.ae.parameters(): p.requires_grad = False
                for p in self.static_adapter_p1.parameters(): p.requires_grad = False
                print(f'[phase2] phase1 frozen ({sum(p.numel() for p in self.ae.parameters())/1e6:.2f}M)')

            self.inner_down = InnerDown(latent_ch_phase1, latent_ch_phase2,
                                         attn_heads=attn_heads, ffn_ratio=ffn_ratio,
                                         n_attn_blocks=n_attn_blocks,
                                         n_outer_res=n_outer_res,
                                         n_inner_res=n_inner_res)
            self.inner_up = InnerUp(latent_ch_phase2, latent_ch_phase1,
                                     attn_heads=attn_heads, ffn_ratio=ffn_ratio,
                                     n_attn_blocks=n_attn_blocks,
                                     n_outer_res=n_outer_res,
                                     n_inner_res=n_inner_res)
            # Static adapter for inner decoder (latent_ch_phase2 size, zero-init)
            self.static_adapter_p2 = nn.Conv2d(n_static, latent_ch_phase2, 1)
            nn.init.zeros_(self.static_adapter_p2.weight)
            nn.init.zeros_(self.static_adapter_p2.bias)

            n_inner = sum(p.numel() for p in self.inner_down.parameters()) + \
                      sum(p.numel() for p in self.inner_up.parameters())
            print(f'[phase2] inner cascade trainable: {n_inner/1e6:.2f}M (z₂ ch={latent_ch_phase2})')

            # SH loss (high-freq)
            if lambda_spec > 0:
                self.sh_loss = SHHighFreqLoss(nlat=360, nlon=720,
                                                lmax=sh_lmax,
                                                l_threshold=sh_l_threshold)

            # 4-level latent split heads — each maps a z₂ channel group
            # to its scale band of z₁ (planetary/synoptic/meso/local).
            # Zero-init: heads start as no-op, do not disturb pixel loss in ep0.
            if lambda_split > 0:
                groups = tuple(split_groups)
                assert sum(groups) == latent_ch_phase2, (
                    f'split_groups {groups} sum to {sum(groups)}, '
                    f'expected latent_ch_phase2={latent_ch_phase2}')
                assert len(groups) == 4, f'split_groups must have 4 entries, got {len(groups)}'
                self.split_groups = groups
                heads = []
                for g in groups:
                    h = nn.Conv2d(g, latent_ch_phase1, kernel_size=1)
                    nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
                    heads.append(h)
                self.split_heads = nn.ModuleList(heads)
                band_labels = ('planetary', 'synoptic', 'meso', 'local')
                print(f'[phase2] 4-level latent split heads: '
                      f'z₂ groups {groups} → {band_labels} bands of z₁')

        # Lat weights (sum = H, mean = 1)
        H = 360
        lat = torch.linspace(89.75, -89.75, H, dtype=torch.float32)
        lw = torch.cos(torch.deg2rad(lat))
        lw = lw / lw.sum() * H
        self.register_buffer('lat_w', lw.view(1, 1, -1, 1), persistent=False)

        # Per-channel weights
        ch_w = torch.tensor([CH_WEIGHTS[c] for c in CH_24], dtype=torch.float32)
        self.register_buffer('ch_w', ch_w.view(1, -1, 1, 1), persistent=False)

        if stds is None: stds = [1.0] * 24
        self.register_buffer('stds', torch.tensor(stds, dtype=torch.float32),
                             persistent=False)

        try:
            stt = torch.load(f'{_wti}/data/static_features_0p5.pt',
                             weights_only=False).float()
            self.register_buffer('default_static', stt, persistent=False)
        except Exception:
            self.default_static = None

        # Per-epoch val accumulators
        self._val_sse = None; self._val_cnt = 0

    def _pad(self, x):
        """REPLICATE lat (preserve poles), CIRCULAR lon (sphere wrap)."""
        # First circular lon, then replicate lat
        if self.LON_PAD > 0:
            x = F.pad(x, (self.LON_PAD, self.LON_PAD, 0, 0), mode='circular')
        if self.LAT_PAD > 0:
            x = F.pad(x, (0, 0, self.LAT_PAD, self.LAT_PAD), mode='replicate')
        return x

    def _unpad(self, x, target_hw=(360, 720)):
        if self.LAT_PAD > 0:
            x = x[..., self.LAT_PAD:-self.LAT_PAD, :]
        if self.LON_PAD > 0:
            x = x[..., :, self.LON_PAD:-self.LON_PAD]
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return x

    def forward(self, x, tisr, static, return_z=False, return_splits=False):
        x_in = torch.cat([x, tisr], dim=1)          # (B, 25, 360, 720)
        x_p = self._pad(x_in)                        # (B, 25, 368, 736)
        z1 = self.ae.encoder(x_p)                    # (B, 48, h_z1, w_z1)

        split_preds = None
        if self.phase == 2:
            z2 = self.inner_down(z1)                 # (B, latent_ch_phase2, h_z2, w_z2)
            # phase-2 static injection at z2 level (decoder-only static)
            if static is not None:
                static_p = self._pad(static.float())
                static_low_z2 = F.adaptive_avg_pool2d(static_p, z2.shape[-2:])
                z2 = z2 + self.static_adapter_p2(static_low_z2)

            # 4-level split heads — predict scale bands of z₁ from z₂ groups.
            # Each head reads its channel slice of z₂, upsamples to z₁ res.
            if return_splits and hasattr(self, 'split_heads'):
                split_preds = []
                ch_off = 0
                for g, head in zip(self.split_groups, self.split_heads):
                    z2_g = z2[:, ch_off:ch_off + g]
                    pred = head(z2_g)
                    pred = F.interpolate(pred, size=z1.shape[-2:],
                                          mode='bilinear', align_corners=False)
                    split_preds.append(pred)
                    ch_off += g

            z1_rec = self.inner_up(z2)               # (B, 48, h_z1, w_z1)
        else:
            z1_rec = z1

        # Static injection at z1 level — frozen Phase 1 decoder was trained
        # to expect this, so we re-apply it in BOTH phase=1 and phase=2 paths.
        z1_for_decoder = z1_rec
        if static is not None:
            static_p = self._pad(static.float())
            static_low_z1 = F.adaptive_avg_pool2d(static_p, z1_for_decoder.shape[-2:])
            z1_for_decoder = z1_for_decoder + self.static_adapter_p1(static_low_z1)

        d_out = self.ae.decoder(z1_for_decoder)
        d_out = self._unpad(d_out, target_hw=(360, 720))
        x_rec = d_out[:, :24]
        if return_z and return_splits:
            return x_rec, z1, z1_rec, split_preds
        if return_z:
            return x_rec, z1, z1_rec
        return x_rec

    def _step(self, batch, prefix):
        x = batch['x']
        tisr = batch['tisr']
        static = batch.get('static', None)
        if static is None and self.default_static is not None:
            B = x.size(0)
            static = self.default_static.unsqueeze(0).expand(B, -1, -1, -1)

        # In Phase 2 we also need z1, z1_rec for latent supervision
        need_z = (self.phase == 2) and (float(self.hparams.lambda_latent) > 0)
        need_split = (self.phase == 2 and float(self.hparams.lambda_split) > 0
                      and hasattr(self, 'split_heads'))
        if need_z or need_split:
            if need_split:
                x_rec, z1, z1_rec, split_preds = self.forward(
                    x, tisr, static, return_z=True, return_splits=True)
            else:
                x_rec, z1, z1_rec = self.forward(x, tisr, static, return_z=True)
                split_preds = None
        else:
            x_rec = self.forward(x, tisr, static)
            z1 = z1_rec = None; split_preds = None

        diff = (x_rec.float() - x.float()).pow(2)
        weighted = diff * self.lat_w * self.ch_w
        pixel_loss = weighted.mean()

        loss = pixel_loss
        self.log(f'{prefix}/pixel_loss', pixel_loss, sync_dist=True,
                 prog_bar=(prefix == 'train'))

        # Latent reconstruction loss (Hier v2 P2 supervision: z1_rec ≈ z1)
        if need_z:
            latent_loss = F.mse_loss(z1_rec.float(), z1.float().detach())
            loss = loss + self.hparams.lambda_latent * latent_loss
            self.log(f'{prefix}/latent_loss', latent_loss, sync_dist=True)

        if self.phase == 2 and self.hparams.lambda_spec > 0:
            spec = self.sh_loss(x_rec, x)
            loss = loss + self.hparams.lambda_spec * spec
            self.log(f'{prefix}/sh_hf', spec, sync_dist=True)

        # 4-level latent split auxiliary loss
        if need_split and split_preds is not None and z1 is not None:
            z_planet, z_synop, z_meso, z_local = decompose_z1_4lvl(z1.float().detach())
            targets = (z_planet, z_synop, z_meso, z_local)
            split_losses = [F.mse_loss(p.float(), t) for p, t in zip(split_preds, targets)]
            split_total = sum(split_losses) / len(split_losses)
            loss = loss + self.hparams.lambda_split * split_total
            self.log(f'{prefix}/split_total', split_total, sync_dist=True)
            for band, l in zip(('planet', 'synop', 'meso', 'local'), split_losses):
                self.log(f'{prefix}/split_{band}', l, sync_dist=True)

        if prefix == 'val':
            with torch.no_grad():
                per_ch = (diff * self.lat_w).mean(dim=(2, 3))  # (B, 24)
                if self._val_sse is None:
                    self._val_sse = torch.zeros(24, device=per_ch.device,
                                                 dtype=torch.float64)
                self._val_sse += per_ch.sum(dim=0).double()
                self._val_cnt += per_ch.shape[0]

        self.log(f'{prefix}/loss', loss, sync_dist=True, prog_bar=True)
        return loss

    def training_step(self, b, i): return self._step(b, 'train')
    def validation_step(self, b, i): return self._step(b, 'val')

    def on_validation_epoch_end(self):
        if self._val_sse is None: return
        rmse_norm = (self._val_sse / max(1, self._val_cnt)).sqrt().cpu().numpy()
        rmse_phys = rmse_norm * self.stds.cpu().numpy()
        print(f'\n=== Epoch {self.current_epoch} per-channel physical RMSE ===')
        print(f'{"Channel":<8} {"norm":>9} {"phys":>10} unit')
        for i, ch in enumerate(CH_24):
            r_n = rmse_norm[i]; r_p = rmse_phys[i]
            unit = 'K'; disp = r_p
            if ch.startswith('Q'): disp = r_p * 1000; unit = 'g/kg'
            elif ch == 'mslp':     disp = r_p / 100;  unit = 'hPa'
            elif ch.startswith('Z'): disp = r_p / 9.81; unit = 'm(gph)'
            elif ch.startswith('U') or ch.startswith('V') or ch in ('u10','v10'):
                unit = 'm/s'
            print(f'{ch:<8} {r_n:>9.4f} {disp:>10.4f} {unit}')
            self.log(f'val_phys/{ch}', float(disp), sync_dist=True)
            self.log(f'val_norm/{ch}', float(r_n), sync_dist=True)
        mn = rmse_norm.mean()
        print(f'Mean norm = {mn:.4f}\n')
        self.log('val/mean_rmse_norm', float(mn), sync_dist=True, prog_bar=True)
        self._val_sse = None; self._val_cnt = 0

    def configure_optimizers(self):
        # Split-LR: pretrained Phase 1 params (encoder/decoder/static_adapter_p1)
        # at lr * lr_ratio_p1, everything else (inner cascade + static_adapter_p2)
        # at full lr. Protects pretrained Phase 1 from drift during P2 fine-tune.
        p1_prefixes = ('ae.', 'static_adapter_p1.')
        p1_params, inner_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            if name.startswith(p1_prefixes):
                p1_params.append(p)
            else:
                inner_params.append(p)
        ratio = float(self.hparams.lr_ratio_p1) if self.phase == 2 else 1.0
        groups = []
        if p1_params:
            groups.append({'params': p1_params, 'lr': self.hparams.lr * ratio,
                           'name': 'phase1'})
        if inner_params:
            groups.append({'params': inner_params, 'lr': self.hparams.lr,
                           'name': 'inner'})
        n_p1 = sum(p.numel() for p in p1_params) / 1e6
        n_in = sum(p.numel() for p in inner_params) / 1e6
        print(f'[opt] P1 group: {n_p1:.2f}M @ lr×{ratio:.2f} | inner group: {n_in:.2f}M @ lr×1.0')
        opt = torch.optim.AdamW(groups, weight_decay=self.hparams.weight_decay,
                                 betas=(0.9, 0.95))
        total = self.trainer.estimated_stepping_batches if self.trainer else 100000
        warmup = min(500, total // 30)
        def lr_lambda(step):
            if step < warmup: return step / max(1, warmup)
            p = min(1.0, (step - warmup) / max(1, total - warmup))
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', type=int, choices=[1, 2], required=True)
    ap.add_argument('--phase1_ckpt', default=None)
    ap.add_argument('--years', nargs='+', type=int, default=[2017, 2018, 2019])
    ap.add_argument('--max_epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--latent_ch_phase1', type=int, default=48)
    ap.add_argument('--latent_ch_phase2', type=int, default=128)
    ap.add_argument('--lambda_spec', type=float, default=0.05)
    ap.add_argument('--lambda_latent', type=float, default=0.5,
                    help='Phase 2 latent reconstruction loss weight (z1_rec vs z1).')
    ap.add_argument('--sh_l_threshold', type=int, default=30)
    ap.add_argument('--unfreeze_phase1', action='store_true', default=True,
                    help='Phase 2: unfreeze Phase 1 encoder/decoder for full end-to-end fine-tune.')
    ap.add_argument('--freeze_phase1', dest='unfreeze_phase1', action='store_false',
                    help='Phase 2: keep Phase 1 frozen (legacy v2 behaviour).')
    ap.add_argument('--lr_ratio_p1', type=float, default=0.1,
                    help='Phase 1 LR multiplier when unfrozen.')
    ap.add_argument('--attn_heads', type=int, default=4)
    ap.add_argument('--ffn_ratio', type=int, default=8)
    ap.add_argument('--n_attn_blocks', type=int, default=6,
                    help='Number of transformer-style (SelfAttn + FFN) blocks in z₂.')
    ap.add_argument('--n_outer_res', type=int, default=4,
                    help='Number of ResBlks at z₁ side of inner cascade.')
    ap.add_argument('--n_inner_res', type=int, default=2,
                    help='Number of ResBlks at z₂ side, before/after the attn stack.')
    ap.add_argument('--lambda_split', type=float, default=0.3,
                    help='Weight for 4-level latent split aux loss (set 0 to disable).')
    ap.add_argument('--split_groups', nargs=4, type=int, default=[24, 40, 40, 24],
                    help='Channel groups (planetary, synoptic, meso, local); must sum to latent_ch_phase2.')
    ap.add_argument('--gpus', nargs='+', type=int, default=[0])
    ap.add_argument('--exp_name', required=True)
    ap.add_argument('--memmap_dir', default='/tmp/wb2_0p5_cache')
    ap.add_argument('--log_root', default=None)
    ap.add_argument('--save_top_k', type=int, default=2)
    ap.add_argument('--precision', default='32-true')
    args = ap.parse_args()

    dsx = xr.open_dataset(f'{_wti}/data/json_stats_0p5.nc')
    nc = list(dsx.params.values)
    sfc = json.loads(Path(f'{_wti}/data/surface_stats_0p5.json').read_text())
    stds = []
    for ch in CH_24:
        if ch in nc:
            stds.append(float(dsx.climate_statistics.sel(stats='std', params=ch).values))
        elif ch in sfc:
            stds.append(float(sfc[ch]['std']))
        else:
            stds.append(1.0)

    all_years = sorted(set(args.years + [2020]))
    tisr_loader = TISRLoader(args.memmap_dir, all_years)

    base_train = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=args.years, max_tau_hours=6,
        samples_per_date=4, train=True, eval_hours=[1, 2, 3, 4, 5],
        static_path=f'{_wti}/data/static_features_0p5.pt',
        stats_path=f'{_wti}/data/json_stats_0p5.nc',
        surface_stats_path=f'{_wti}/data/surface_stats_0p5.json')
    base_val = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=[2020], max_tau_hours=6,
        samples_per_date=4, train=False, eval_hours=[1, 2, 3, 4, 5],
        static_path=f'{_wti}/data/static_features_0p5.pt',
        stats_path=f'{_wti}/data/json_stats_0p5.nc',
        surface_stats_path=f'{_wti}/data/surface_stats_0p5.json')

    train_ds = SingleFrameWithTISR(base_train, tisr_loader, args.years)
    val_ds = SingleFrameWithTISR(base_val, tisr_loader, [2020])
    print(f'train={len(train_ds)} val={len(val_ds)} phase={args.phase}')

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                               num_workers=args.workers, pin_memory=True,
                               persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    model = HierCompressAEv2(
        phase=args.phase, phase1_ckpt=args.phase1_ckpt,
        in_channels=24,
        latent_ch_phase1=args.latent_ch_phase1,
        latent_ch_phase2=args.latent_ch_phase2,
        lr=args.lr,
        lambda_spec=args.lambda_spec if args.phase == 2 else 0.0,
        lambda_latent=args.lambda_latent if args.phase == 2 else 0.0,
        sh_l_threshold=args.sh_l_threshold,
        unfreeze_phase1=args.unfreeze_phase1,
        lr_ratio_p1=args.lr_ratio_p1,
        attn_heads=args.attn_heads,
        ffn_ratio=args.ffn_ratio,
        n_attn_blocks=args.n_attn_blocks,
        n_outer_res=args.n_outer_res,
        n_inner_res=args.n_inner_res,
        lambda_split=args.lambda_split if args.phase == 2 else 0.0,
        split_groups=tuple(args.split_groups),
        stds=stds,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_trainable/1e6:.2f}M')

    log_root = args.log_root or f'{_wti}/logs'
    ckpt_dir = Path(log_root) / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename='{epoch}-{step}-r{val/mean_rmse_norm:.4f}',
        monitor='val/mean_rmse_norm', mode='min', save_top_k=args.save_top_k,
        save_last=True, auto_insert_metric_name=False)
    logger = TensorBoardLogger(save_dir=str(log_root), name=args.exp_name)
    strategy = "auto"
    if len(args.gpus) > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(process_group_backend="nccl",
                                find_unused_parameters=True)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpus, accelerator='gpu',
        precision=args.precision,
        strategy=strategy,
        log_every_n_steps=50,
        callbacks=[ckpt_cb], logger=logger,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
