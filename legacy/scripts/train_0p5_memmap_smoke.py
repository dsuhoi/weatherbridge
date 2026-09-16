#!/usr/bin/env python3
"""SMOKE training using memmap Dataset on local SSD.

Verifies: end-to-end training works with FuXi SwinV2 0.5° at fast SSD speed.
Target: GPU >50% utilization within first 20 batches.
"""
import sys, time
sys.path.insert(0, "/home/jovyan/dsuhoi/weather_time_interpolation")
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import WeatherHermiteLightningModule, ERA5WeatherHermiteDataset


print("=== TRAIN 0.5° SMOKE (memmap) ===", flush=True)
t0 = time.time()

ds_train = ERA5MemmapDataset(
    memmap_dir="/tmp/wb2_0p5_cache",
    years=[2018],
    max_tau_hours=6,
    samples_per_date=4,
    train=True,
    train_hours=[1, 3, 5],
    static_path="data/static_features_0p5.pt",
    stats_path="data/json_stats_0p5.nc",
    surface_stats_path="data/surface_stats_0p5.json",
)
print(f"train samples: {len(ds_train)}", flush=True)
print(f"channels: {ds_train.in_channels} PL + {len(ds_train.surface_variables)} surf", flush=True)
print(f"channel_groups: {ds_train.channel_groups}", flush=True)

channel_groups = ds_train.channel_groups
ds_train_wrapped = ERA5WeatherHermiteDataset(ds_train, delta_t_hours=6.0)

# Sample one item to verify
batch_sample = ds_train_wrapped[0]
print(f"sample x0 shape: {tuple(batch_sample['x0'].shape)}, xT={tuple(batch_sample['xT'].shape)}, target={tuple(batch_sample['target'].shape)}", flush=True)
ds_train = ds_train_wrapped

loader = DataLoader(ds_train, batch_size=1, shuffle=True, num_workers=4,
                    pin_memory=True, persistent_workers=True)
print(f"loader ready. dataloader iter init...", flush=True)

# Lightning module
model = WeatherHermiteLightningModule(
    model_type="fuxi_swinv2_residual_linear",
    channel_groups=channel_groups,
    n_pl_channels=20,
    n_surface_channels=7,
    n_static_features=3,
    latent_channels=256,
    fuxi_depth=8,
    fuxi_num_heads=8,
    fuxi_window_size_h=5,
    fuxi_window_size_w=9,
    fuxi_patch_size=4,
    fuxi_drop_path=0.1,
    lat_weighted_loss=True,
    lat_crop=1,
    lambda_residual=1.0,
    residual_scale_floor=0.05,
    residual_scale_init=0.30,
    lambda_anchor=0.5,
    anchor_every_n_batches=4,
    lr=1e-4,
    weight_decay=1e-5,
    use_aurora_weights=True,
    use_physical_scales_loss=False,
    max_tau_hours=6,
)
print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

import os
max_steps = int(os.environ.get("MAX_STEPS", 30))
max_epochs = int(os.environ.get("MAX_EPOCHS", 1))
log_every = int(os.environ.get("LOG_EVERY", 25))
out_dir = os.environ.get("OUT_DIR", "logs/exp_fuxi_0p5_2018_smoke")
print(f"max_epochs={max_epochs}, max_steps={max_steps}, out_dir={out_dir}", flush=True)
from lightning.pytorch.callbacks import ModelCheckpoint
ckpt_cb = ModelCheckpoint(dirpath=out_dir, save_last=True, save_top_k=0, every_n_epochs=1)
trainer = pl.Trainer(
    max_epochs=max_epochs,
    max_steps=max_steps if max_steps > 0 else -1,
    log_every_n_steps=log_every, devices=1, accelerator="gpu",
    precision="16-mixed", limit_val_batches=0, num_sanity_val_steps=0,
    enable_checkpointing=True, enable_progress_bar=False,
    callbacks=[ckpt_cb],
    default_root_dir=out_dir,
)
print(f"Starting trainer.fit(), startup={time.time()-t0:.1f}s", flush=True)
trainer.fit(model, train_dataloaders=loader)
print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)
