#!/usr/bin/env python3
"""Phase 2: Decoder residual fine-tune on τ corrections to bilinear baseline.

Encoder FROZEN from Phase 1. Decoder + per-channel scale learns:
    x_τ = (1−τ)·x_0 + τ·x_T  +  scale · decoder(encoder(x_τ))
                                └────── δ residual ──────┘

Logs per-channel val RMSE (model vs bilinear baseline) so you can see exactly
which channels benefit from the residual. LadCast SST sentinel masking applied.

Usage:
    python train_phase2_residual_decoder.py \
        --phase1_ckpt /path/to/phase1/last.ckpt \
        --gpus 0 1 \
        --max_epochs 8
"""
import os, sys, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
torch.set_float32_matmul_precision('high')

# Default Phase 1 trainer location; override via env if needed
_PHASE1_TRAINER = os.environ.get('PHASE1_TRAINER', '/tmp/train_dcae_autoencoder_6yr.py')
_ns = {}
exec(open(_PHASE1_TRAINER).read().replace("if __name__ == '__main__':", "if False:"), _ns)
WeatherDCAEAutoencoder = _ns['WeatherDCAEAutoencoder']
CH_27 = _ns['CH_27']

# Repo root for ERA5MemmapDataset import
sys.path.insert(0, '/home/jovyan/dsuhoi/weather_time_interpolation')
sys.path.insert(0, '/workspace/code/wti')                                       # fibo container
from weather_time_interp.memmap_dataset import ERA5MemmapDataset


class Phase2ResidualVFI(pl.LightningModule):
    """Frozen encoder + trainable decoder + per-channel learnable scale.
    Trains residual δ to bilinear baseline."""

    def __init__(self, phase1_ckpt, in_channels=27,
                 lr=1e-4, weight_decay=1e-4,
                 residual_scale_init=0.10,
                 use_per_channel_scale=True,
                 finetune_static_proj=True):
        super().__init__()
        self.save_hyperparameters()
        self.ae = WeatherDCAEAutoencoder.load_from_checkpoint(phase1_ckpt, map_location='cpu')
        # Freeze encoder
        for p in self.ae.encoder.parameters():
            p.requires_grad = False
        self.ae.encoder.eval()
        # Decoder + static_proj remain trainable
        if not finetune_static_proj:
            for p in self.ae.static_proj.parameters():
                p.requires_grad = False

        if use_per_channel_scale:
            init = torch.full((in_channels,), float(residual_scale_init))
            self.scale_raw = nn.Parameter(init)
        else:
            self.scale_raw = nn.Parameter(torch.tensor(float(residual_scale_init)))

    @property
    def scale(self):
        return torch.tanh(self.scale_raw)

    def _encode(self, x):
        with torch.no_grad():
            x_c = self.ae._crop_lat(x)
            x_c = self.ae._pad_lon(x_c)
            return self.ae.encoder(x_c)

    def _decode(self, z, target_h, target_w):
        static_lat = F.adaptive_avg_pool2d(self.ae.static_full, z.shape[-2:])
        static_lat = static_lat.unsqueeze(0).expand(z.size(0), -1, -1, -1)
        z_aug = torch.cat([z, static_lat], dim=1)
        z_proj = self.ae.static_proj(z_aug)
        out = self.ae.decoder(z_proj)
        out = self.ae._crop_lon_back(out)
        if out.shape[-2:] != (target_h, target_w):
            out = F.interpolate(out, size=(target_h, target_w),
                                mode='bilinear', align_corners=False)
        return out

    def forward(self, x_target):
        z = self._encode(x_target)
        delta = self._decode(z, x_target.size(-2), x_target.size(-1))
        return delta, z

    def _broadcast_scale(self, x):
        s = self.scale
        if s.dim() == 1:
            return s.view(1, -1, 1, 1)
        return s.view(1, 1, 1, 1)

    def _step(self, batch, prefix):
        x_0 = batch['x0']
        x_T = batch['x1']
        x_tgt = batch['target']
        tau = batch['tau']
        if tau.dim() == 1: tau = tau.view(-1, 1, 1, 1)
        elif tau.dim() == 2: tau = tau.view(-1, 1, 1, 1)

        x_bilinear = (1.0 - tau) * x_0 + tau * x_T
        delta, z = self(x_tgt)
        s = self._broadcast_scale(delta)
        x_pred = x_bilinear + s * delta

        # LadCast SST mask: zero out land contribution for SST
        x_pred_m = self.ae._apply_sst_mask(x_pred)
        x_tgt_m = self.ae._apply_sst_mask(x_tgt)
        x_bil_m = self.ae._apply_sst_mask(x_bilinear)

        # Lat-weighted L1 loss (training objective)
        err = (x_pred_m - x_tgt_m).abs()
        recon = (err * self.ae.lat_w).mean()

        # Diagnostics: MSE for model vs bilinear-only
        sq_model = (x_pred_m.float() - x_tgt_m.float()) ** 2
        sq_bil = (x_bil_m.float() - x_tgt_m.float()) ** 2
        mse_model = (sq_model * self.ae.lat_w).mean()
        mse_bil = (sq_bil * self.ae.lat_w).mean()

        self.log(f'{prefix}/recon_l1', recon, sync_dist=True, prog_bar=True)
        self.log(f'{prefix}/mse_model', mse_model.detach(), sync_dist=True)
        self.log(f'{prefix}/mse_bilinear', mse_bil.detach(), sync_dist=True)
        self.log(f'{prefix}/improve_vs_bil', (mse_bil - mse_model).detach(), sync_dist=True)
        self.log(f'{prefix}/scale_mean', self.scale.mean().detach(), sync_dist=True)
        self.log(f'{prefix}/delta_abs_mean', delta.abs().mean().detach(), sync_dist=True)

        if prefix == 'val':
            sq_model_lw = (sq_model * self.ae.lat_w).sum(dim=(-2, -1))           # (B, 27)
            sq_bil_lw = (sq_bil * self.ae.lat_w).sum(dim=(-2, -1))
            self.ae.val_sq_sum += sq_model_lw.sum(dim=0).detach()
            self.ae.val_count += sq_model_lw.size(0)
            if not hasattr(self, '_val_bil_sum'):
                self.register_buffer('_val_bil_sum',
                                     torch.zeros_like(self.ae.val_sq_sum),
                                     persistent=False)
            self._val_bil_sum += sq_bil_lw.sum(dim=0).detach()
        return recon

    def training_step(self, batch, idx): return self._step(batch, 'train')
    def validation_step(self, batch, idx): return self._step(batch, 'val')

    def on_validation_epoch_end(self):
        if self.trainer is not None and self.trainer.world_size > 1:
            torch.distributed.all_reduce(self.ae.val_sq_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.ae.val_count, op=torch.distributed.ReduceOp.SUM)
            if hasattr(self, '_val_bil_sum'):
                torch.distributed.all_reduce(self._val_bil_sum, op=torch.distributed.ReduceOp.SUM)
        n_px = self.ae.land_mask.numel()
        if self.ae.val_count.item() > 0:
            mse_per_ch = self.ae.val_sq_sum / (self.ae.val_count * n_px)
            rmse = mse_per_ch.sqrt()
            for i, ch in enumerate(CH_27):
                self.log(f'val_rmse/{ch}', rmse[i].item(), rank_zero_only=True)
            sfc = ['t2m', 'u10', 'v10', 'mslp', 'sst', 'tcc', 'tcwv']
            self.log('val_rmse/surface_mean',
                     torch.stack([rmse[CH_27.index(c)] for c in sfc]).mean().item(),
                     rank_zero_only=True)
            if hasattr(self, '_val_bil_sum'):
                mse_bil_per_ch = self._val_bil_sum / (self.ae.val_count * n_px)
                rmse_bil = mse_bil_per_ch.sqrt()
                for i, ch in enumerate(CH_27):
                    self.log(f'val_bil_rmse/{ch}', rmse_bil[i].item(), rank_zero_only=True)
                improve = (1.0 - rmse / rmse_bil.clamp(min=1e-8)).mean()
                self.log('val_rmse/relative_improvement', improve.item(), rank_zero_only=True)
                self._val_bil_sum.zero_()
        self.ae.val_sq_sum.zero_()
        self.ae.val_count.zero_()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay,
                                 betas=(0.9, 0.95))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase1_ckpt', required=True)
    p.add_argument('--years', nargs='+', type=int,
                   default=[2014, 2015, 2016, 2017, 2018, 2019])
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--max_epochs', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--exp_name', default='exp_dcae_phase2_residual_6yr')
    p.add_argument('--gpus', nargs='+', type=int, default=[0])
    p.add_argument('--memmap_dir', default='/tmp/wb2_0p5_cache')
    p.add_argument('--log_root', default='/home/jovyan/dsuhoi/weather_time_interpolation/logs')
    p.add_argument('--per_channel_scale', action='store_true', default=True)
    p.add_argument('--no_finetune_static_proj', action='store_true')
    args = p.parse_args()

    train_ds = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=args.years,
        max_tau_hours=6, samples_per_date=4, train=True,
        eval_hours=[1, 2, 3, 4, 5],
        static_path='data/static_features_0p5.pt',
        stats_path='data/json_stats_0p5.nc',
        surface_stats_path='data/surface_stats_0p5.json',
    )
    val_ds = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=[2020],
        max_tau_hours=6, samples_per_date=4, train=False,
        eval_hours=[1, 2, 3, 4, 5],
        static_path='data/static_features_0p5.pt',
        stats_path='data/json_stats_0p5.nc',
        surface_stats_path='data/surface_stats_0p5.json',
    )
    print(f'train: {len(train_ds)}  val: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = Phase2ResidualVFI(
        phase1_ckpt=args.phase1_ckpt,
        in_channels=27,
        lr=args.lr,
        use_per_channel_scale=args.per_channel_scale,
        finetune_static_proj=not args.no_finetune_static_proj)
    trainable = sum(q.numel() for q in model.parameters() if q.requires_grad)
    total = sum(q.numel() for q in model.parameters())
    print(f'params: total={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M  '
          f'(per_channel_scale={args.per_channel_scale})')

    out_dir = Path(args.log_root) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=out_dir, save_top_k=-1, every_n_epochs=1,
                              filename='{epoch}-{step}', save_last=True)
    logger = TensorBoardLogger(save_dir=str(out_dir / 'lightning_logs'), name='', version=0)

    strategy = 'ddp' if len(args.gpus) > 1 else 'auto'
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu', devices=args.gpus,
        strategy=strategy,
        precision='bf16-mixed',
        callbacks=[ckpt_cb], logger=logger,
        log_every_n_steps=20,
        gradient_clip_val=1.0,
        val_check_interval=0.5,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
