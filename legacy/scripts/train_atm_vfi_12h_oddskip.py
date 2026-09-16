#!/usr/bin/env python3
"""ATM-VFI (pixel-space cross-frame attention VFI) — 12h holdout protocol.

Adapted from train_pixel_attn_vfi.py.

Window: 12h between anchor frames x(t_0), x(t_0+12h).
Train tau subset: {1,2,3,5,7,9,10,11} ("seen" — 8 points, all odd + boundary evens).
Held-out tau (unseen): {4,6,8}  (3 middle evens — never seen during training).
24-channel set (drops sst/tcc/tcwv).
Train: 2017-2019 (3 years). Val: 2020.

Val logs:
  val/seen/rmse_mean  — RMSE on train hours {1,2,3,5,7,9,10,11}
  val/unseen/rmse_mean — RMSE on held-out hours {4,6,8}
  val_rmse/h{1..11}/<chan> — per-hour per-channel breakdown
"""
import os, sys, argparse, math, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')

# Make code importable
sys.path.insert(0, os.environ.get('WTI_ROOT', '/workspace/code/wti'))
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from weather_time_interp.memmap_dataset import ERA5MemmapDataset

# 24-channel set (sst/tcc/tcwv dropped)
CH_24 = ["T1000","T925","T850","T700","U1000","U925","U850","U700",
         "V1000","V925","V850","V700","Q1000","Q925","Q850","Q700",
         "Z1000","Z925","Z850","Z700","t2m","u10","v10","mslp"]

DELTA_T = 12.0  # 12h window


class TauRescaleAnd24chWrapper(Dataset):
    """Drop sst/tcc/tcwv (last 3 of 27) AND rescale tau = tau_hour / delta_t."""
    def __init__(self, base, delta_t=12.0, n_keep=24):
        self.base = base
        self.delta_t = float(delta_t)
        self.n_keep = n_keep
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        out = self.base[idx]
        # Rescale tau: memmap_dataset divides by HOURS_PER_TAU_UNIT=6 hardcoded.
        # We want tau ∈ [0,1] for a 12h window, so tau = tau_hour / 12.
        if 'tau_hour' in out:
            th = out['tau_hour'].float()
            out['tau'] = (th / self.delta_t).view_as(out['tau']) if out['tau'].shape == th.shape else (th / self.delta_t)
        # Truncate channels
        for k in ('x0', 'x1', 'target'):
            if k in out and isinstance(out[k], torch.Tensor) and out[k].dim() >= 3:
                out[k] = out[k][..., :self.n_keep, :, :].contiguous()
        return out


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


class PixelAttentionVFI(pl.LightningModule):
    def __init__(self, in_channels=24, hidden=64, n_levels=3,
                 static_features_path='data/static_features_0p5.pt',
                 lr=2e-4, weight_decay=1e-4,
                 train_taus=(1,2,3,5,7,9,10,11),
                 eval_taus=(1,2,3,4,5,6,7,8,9,10,11),
                 delta_t=12.0):
        super().__init__()
        self.save_hyperparameters()
        self.net = PixelAttentionVFINet(in_ch=in_channels, hidden=hidden, n_levels=n_levels)
        static = torch.load(static_features_path, weights_only=False).float()
        lsm = static[0]
        self.register_buffer('land_mask', (lsm > 0.5).float(), persistent=False)
        H = lsm.shape[0]
        lat = torch.linspace(89.75, -89.75, H, dtype=torch.float32)
        lw = torch.cos(torch.deg2rad(lat)); lw = lw / lw.sum() * H
        self.register_buffer('lat_w', lw.view(1, 1, -1, 1), persistent=False)
        self.train_taus = set(int(t) for t in train_taus)
        self.eval_taus = list(eval_taus)
        self.delta_t = float(delta_t)
        # Per-tau val accumulators (sq sum + count)
        for h in self.eval_taus:
            self.register_buffer(f'val_sq_h{h}', torch.zeros(in_channels), persistent=False)
            self.register_buffer(f'val_cnt_h{h}', torch.tensor(0.0), persistent=False)

    def _step(self, batch, prefix):
        x_0, x_T = batch['x0'], batch['x1']
        x_tgt = batch['target']
        tau = batch['tau']
        if tau.dim() > 1: tau = tau.view(-1)
        x_pred = self.net(x_0, x_T, tau)
        err = ((x_pred - x_tgt).abs() * self.lat_w).mean()
        sq = (x_pred.float() - x_tgt.float()) ** 2
        mse = (sq * self.lat_w).mean()
        self.log(f'{prefix}/recon_l1', err, sync_dist=True, prog_bar=True)
        self.log(f'{prefix}/mse', mse.detach(), sync_dist=True)
        if prefix == 'val':
            # Per-sample tau_hour for split
            tau_hour = batch['tau_hour'].view(-1).long()
            # Per-sample lat-weighted sum-of-sq (spatial)
            sq_lw = (sq * self.lat_w).sum(dim=(-2, -1))  # (B, C)
            for h in self.eval_taus:
                mask = (tau_hour == h)
                if mask.any():
                    buf_sq = getattr(self, f'val_sq_h{h}')
                    buf_ct = getattr(self, f'val_cnt_h{h}')
                    buf_sq += sq_lw[mask].sum(dim=0).detach()
                    buf_ct += mask.float().sum()
        return err

    def training_step(self, b, i): return self._step(b, 'train')
    def validation_step(self, b, i): return self._step(b, 'val')

    def on_validation_epoch_end(self):
        if self.trainer is not None and self.trainer.world_size > 1:
            for h in self.eval_taus:
                torch.distributed.all_reduce(getattr(self, f'val_sq_h{h}'), op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(getattr(self, f'val_cnt_h{h}'), op=torch.distributed.ReduceOp.SUM)
        n_px = self.land_mask.numel()
        seen_rmse, unseen_rmse = [], []
        for h in self.eval_taus:
            sq = getattr(self, f'val_sq_h{h}')
            ct = getattr(self, f'val_cnt_h{h}')
            if ct.item() > 0:
                mse_pc = sq / (ct * n_px)
                rmse = mse_pc.sqrt()
                mean_rmse = rmse.mean().item()
                self.log(f'val_rmse/h{h}/mean', mean_rmse, rank_zero_only=True)
                for i, c in enumerate(CH_24):
                    self.log(f'val_rmse/h{h}/{c}', rmse[i].item(), rank_zero_only=True)
                if h in self.train_taus:
                    seen_rmse.append(mean_rmse)
                else:
                    unseen_rmse.append(mean_rmse)
        if seen_rmse:
            self.log('val/seen/rmse_mean', sum(seen_rmse)/len(seen_rmse), rank_zero_only=True, prog_bar=True)
        if unseen_rmse:
            self.log('val/unseen/rmse_mean', sum(unseen_rmse)/len(unseen_rmse), rank_zero_only=True, prog_bar=True)
        # reset
        for h in self.eval_taus:
            getattr(self, f'val_sq_h{h}').zero_()
            getattr(self, f'val_cnt_h{h}').zero_()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay, betas=(0.9, 0.95))
        total = self.trainer.estimated_stepping_batches if self.trainer else 100000
        warmup = min(500, total // 30)
        def lr_lambda(step):
            if step < warmup: return step / max(1, warmup)
            p = min(1.0, (step - warmup) / max(1, total - warmup))
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def parse_int_list(s):
    return [int(x) for x in s.split()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--years', nargs='+', type=int, default=[2017, 2018, 2019])
    p.add_argument('--val_years', nargs='+', type=int, default=[2020])
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--val_bs', type=int, default=4)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--val_workers', type=int, default=4)
    p.add_argument('--max_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--n_levels', type=int, default=3)
    p.add_argument('--exp_name', default='exp_atmvfi_12h_oddskip')
    p.add_argument('--gpus', nargs='+', type=int, default=[0])
    p.add_argument('--memmap_dir', default='/tmp/wb2_0p5_cache')
    p.add_argument('--static_path', default='data/static_features_0p5.pt')
    p.add_argument('--stats_path', default='data/json_stats_0p5.nc')
    p.add_argument('--surface_stats_path', default='data/surface_stats_0p5.json')
    p.add_argument('--window_hours', type=int, default=12)
    p.add_argument('--train_tau_subset', nargs='+', type=int, default=[1,2,3,5,7,9,10,11])
    p.add_argument('--eval_tau', nargs='+', type=int, default=[1,2,3,4,5,6,7,8,9,10,11])
    p.add_argument('--samples_per_date_train', type=int, default=2)
    p.add_argument('--samples_per_date_val', type=int, default=1)
    p.add_argument('--limit_val_batches', type=float, default=1.0)
    p.add_argument('--limit_train_batches', type=float, default=1.0)
    p.add_argument('--precision', default='bf16-mixed')
    p.add_argument('--ckpt_path', default=None)
    p.add_argument('--val_every_n_epochs', type=int, default=1)
    p.add_argument('--ckpt_every_n_epochs', type=int, default=2)
    args = p.parse_args()

    print(f'=== TRAIN ATM-VFI 12h ODDSKIP ===')
    print(f'  years={args.years} val_years={args.val_years} epochs={args.max_epochs} bs={args.bs}')
    print(f'  train_tau (SEEN) = {args.train_tau_subset}')
    print(f'  eval_tau         = {args.eval_tau} → unseen = {sorted(set(args.eval_tau) - set(args.train_tau_subset))}')
    print(f'  window={args.window_hours}h, hidden={args.hidden}, levels={args.n_levels}, gpus={args.gpus}')

    train_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=args.years, max_tau_hours=args.window_hours,
        samples_per_date=args.samples_per_date_train, train=True,
        train_hours=args.train_tau_subset,
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path)
    train_ds = TauRescaleAnd24chWrapper(train_base, delta_t=float(args.window_hours), n_keep=24)
    val_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=args.val_years, max_tau_hours=args.window_hours,
        samples_per_date=args.samples_per_date_val, train=False,
        eval_hours=args.eval_tau,
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path)
    val_ds = TauRescaleAnd24chWrapper(val_base, delta_t=float(args.window_hours), n_keep=24)

    print(f'  train samples: {len(train_ds)}  val samples: {len(val_ds)}', flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_bs, shuffle=False,
                            num_workers=args.val_workers, pin_memory=True,
                            persistent_workers=args.val_workers > 0)

    model = PixelAttentionVFI(
        in_channels=24, hidden=args.hidden, n_levels=args.n_levels,
        static_features_path=args.static_path,
        lr=args.lr,
        train_taus=tuple(args.train_tau_subset),
        eval_taus=tuple(args.eval_tau),
        delta_t=float(args.window_hours),
    )
    n_p = sum(q.numel() for q in model.parameters()) / 1e6
    print(f'params: {n_p:.2f}M', flush=True)

    out_dir = Path(os.environ.get('OUT_DIR', f'logs/{args.exp_name}'))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=out_dir, save_last=True, save_top_k=-1,
                              every_n_epochs=args.ckpt_every_n_epochs,
                              filename='{epoch}-{step}')
    logger = CSVLogger(save_dir=str(out_dir / 'lightning_logs'), name='', version=0)

    if len(args.gpus) > 1:
        from pytorch_lightning.strategies import DDPStrategy
        backend = os.environ.get('DDP_BACKEND', 'nccl')
        strategy = DDPStrategy(process_group_backend=backend, find_unused_parameters=False)
    else:
        strategy = 'auto'
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator='gpu', devices=args.gpus,
        strategy=strategy, precision=args.precision,
        callbacks=[ckpt_cb], logger=logger,
        log_every_n_steps=20, gradient_clip_val=1.0,
        check_val_every_n_epoch=args.val_every_n_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)
    print('=== DONE ===')


if __name__ == '__main__':
    main()
