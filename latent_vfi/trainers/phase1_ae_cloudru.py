#!/usr/bin/env python3
"""DC-AE autoencoder on weather fields (27ch, 360x720, 6yr ERA5).

v2.1: LadCast-style SST masking + lat-weighted L1 + per-channel val RMSE +
static features (LSM + orography + lat_cos) injected into decoder via
1x1 projection at latent bottleneck.

(SphereConv2d patch experiment failed due to DC-AE's pixel_unshuffle stride
interaction — kept I/O circular lon-pad only, zero-pad inside DC-AE default.)

Architecture: 5 stages, f=16 spatial, deeper layers. ~123M params + static proj.
Latent: 64 x 22 x 46 ≈ 65k floats → ~108× compression.
"""
import os, sys, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')

sys.path.insert(0, '/home/jovyan/dsuhoi/weather_time_interpolation')
from weather_time_interp.model.dcae import Encoder as DCAEEncoder, Decoder as DCAEDecoder

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from weather_time_interp.memmap_dataset import ERA5MemmapDataset

CH_27 = ["T1000","T925","T850","T700","U1000","U925","U850","U700","V1000","V925","V850","V700",
         "Q1000","Q925","Q850","Q700","Z1000","Z925","Z850","Z700",
         "t2m","u10","v10","mslp","sst","tcc","tcwv"]
SST_IDX = 24


class WeatherDCAEAutoencoder(pl.LightningModule):
    def __init__(self,
                 in_channels=27,
                 latent_channels=64,
                 block_out_channels=(192, 192, 384, 384, 768),
                 layers_per_block=(3, 3, 3, 3, 3),
                 block_type=('ResBlock', 'ResBlock', 'ResBlock',
                             'EfficientViTBlock', 'EfficientViTBlock'),
                 qkv_multiscales=((), (), (), (5,), (5,)),
                 attention_head_dim=32,
                 lat_crop=8,
                 lon_pad=16,
                 sst_idx=SST_IDX,
                 sst_sentinel=0.0,
                 static_features_path='data/static_features_0p5.pt',
                 n_static=3,
                 lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.95)):
        super().__init__()
        self.save_hyperparameters()
        self.lat_crop = lat_crop
        self.lon_pad = lon_pad
        self.sst_idx = sst_idx
        self.sst_sentinel = sst_sentinel
        self.n_static = n_static

        self.encoder = DCAEEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_type=block_type,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            qkv_multiscales=qkv_multiscales,
            attention_head_dim=attention_head_dim,
            downsample_block_type='pixel_unshuffle',
            out_shortcut=True,
        )
        self.decoder = DCAEDecoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            block_type=block_type,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            qkv_multiscales=qkv_multiscales,
            attention_head_dim=attention_head_dim,
            upsample_block_type='pixel_shuffle',
            in_shortcut=True,
        )
        # Static injection: project (latent_ch + n_static) → latent_ch via 1x1
        self.static_proj = nn.Conv2d(latent_channels + n_static,
                                     latent_channels, kernel_size=1)
        # Init: identity on latent part, zero on static part — start as no-op
        with torch.no_grad():
            self.static_proj.weight.zero_()
            self.static_proj.bias.zero_()
            for c in range(latent_channels):
                self.static_proj.weight[c, c, 0, 0] = 1.0

        static = torch.load(static_features_path, weights_only=False).float()  # (3, H, W)
        lsm = static[0]
        self.register_buffer('static_full', static, persistent=False)
        self.register_buffer('land_mask', (lsm > 0.5).float(), persistent=False)

        H = lsm.shape[0]
        lat = torch.linspace(89.75, -89.75, H, dtype=torch.float32)
        lw = torch.cos(torch.deg2rad(lat))
        lw = lw / lw.sum() * H                                                   # mean(lw)=1
        self.register_buffer('lat_w', lw.view(1, 1, -1, 1), persistent=False)

        self.register_buffer('val_sq_sum', torch.zeros(in_channels), persistent=False)
        self.register_buffer('val_count', torch.tensor(0.0), persistent=False)

    def _crop_lat(self, x):
        if self.lat_crop <= 0: return x
        top = self.lat_crop // 2
        bot = self.lat_crop - top
        return x[:, :, top:-bot] if bot > 0 else x[:, :, top:]

    def _pad_lon(self, x):
        if self.lon_pad <= 0: return x
        l = self.lon_pad // 2
        r = self.lon_pad - l
        return F.pad(x, (l, r, 0, 0), mode='circular')

    def _crop_lon_back(self, x):
        if self.lon_pad <= 0: return x
        l = self.lon_pad // 2
        r = self.lon_pad - l
        return x[:, :, :, l:-r] if r > 0 else x[:, :, :, l:]

    def forward(self, x):
        x_c = self._crop_lat(x)
        x_c = self._pad_lon(x_c)
        z = self.encoder(x_c)                                                    # (B, lat_ch, h, w)
        static_lat = F.adaptive_avg_pool2d(self.static_full, z.shape[-2:])       # (3, h, w)
        static_lat = static_lat.unsqueeze(0).expand(z.size(0), -1, -1, -1)
        z_aug = torch.cat([z, static_lat], dim=1)
        z_proj = self.static_proj(z_aug)
        x_rec = self.decoder(z_proj)
        x_rec = self._crop_lon_back(x_rec)
        if x_rec.shape[-2:] != x.shape[-2:]:
            x_rec = F.interpolate(x_rec, size=x.shape[-2:], mode='bilinear',
                                  align_corners=False)
        return x_rec, z

    def _apply_sst_mask(self, t):
        idx = self.sst_idx
        sst = t[:, idx]
        sst = torch.where(self.land_mask.bool(),
                          torch.full_like(sst, self.sst_sentinel), sst)
        out = t.clone()
        out[:, idx] = sst
        return out

    def _step(self, batch, prefix):
        x = batch['x0']
        x_rec, z = self(x)
        x_rec_m = self._apply_sst_mask(x_rec)
        x_tgt_m = self._apply_sst_mask(x)
        err = (x_rec_m - x_tgt_m).abs()
        recon = (err * self.lat_w).mean()
        sq = (x_rec_m.float() - x_tgt_m.float()) ** 2
        mse = (sq * self.lat_w).mean()
        self.log(f'{prefix}/recon_l1', recon, sync_dist=True, prog_bar=True)
        self.log(f'{prefix}/mse', mse.detach(), sync_dist=True)
        self.log(f'{prefix}/lat_std', z.std().detach(), sync_dist=True)
        self.log(f'{prefix}/lat_abs_mean', z.abs().mean().detach(), sync_dist=True)
        if prefix == 'val':
            sq_lw = (sq * self.lat_w).sum(dim=(-2, -1))
            self.val_sq_sum += sq_lw.sum(dim=0).detach()
            self.val_count += sq_lw.size(0)
        return recon

    def training_step(self, batch, idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, idx):
        return self._step(batch, 'val')

    def on_validation_epoch_end(self):
        if self.trainer is not None and self.trainer.world_size > 1:
            torch.distributed.all_reduce(self.val_sq_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.val_count, op=torch.distributed.ReduceOp.SUM)
        n_px = self.land_mask.numel()
        if self.val_count.item() > 0:
            mse_per_ch = self.val_sq_sum / (self.val_count * n_px)
            rmse = mse_per_ch.sqrt()
            for i, ch in enumerate(CH_27):
                self.log(f'val_rmse/{ch}', rmse[i].item(), rank_zero_only=True)
            sfc = ['t2m', 'u10', 'v10', 'mslp', 'sst', 'tcc', 'tcwv']
            self.log('val_rmse/surface_mean',
                     torch.stack([rmse[CH_27.index(c)] for c in sfc]).mean().item(),
                     rank_zero_only=True)
        self.val_sq_sum.zero_()
        self.val_count.zero_()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay,
                                 betas=self.hparams.betas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--years', nargs='+', type=int,
                   default=[2014, 2015, 2016, 2017, 2018, 2019])
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--max_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--latent_channels', type=int, default=64)
    p.add_argument('--exp_name', default='exp_dcae_ae_static_6yr')
    p.add_argument('--gpus', nargs='+', type=int, default=[0])
    p.add_argument('--memmap_dir', default='/tmp/wb2_0p5_cache')
    p.add_argument('--resume', default=None)
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

    model = WeatherDCAEAutoencoder(
        in_channels=27, latent_channels=args.latent_channels, lr=args.lr)
    n = sum(q.numel() for q in model.parameters())
    print(f'params: {n / 1e6:.2f}M  latent_ch: {args.latent_channels}')

    out_dir = Path('/home/jovyan/dsuhoi/weather_time_interpolation/logs') / args.exp_name
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
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
