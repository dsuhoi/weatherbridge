#!/usr/bin/env python3
"""WeatherDCAE-Hybrid training: DC-AE backbone + cross-frame attention +
adaptive skip + per-channel residual scale.

Trains on τ ∈ {1, 3, 5}; τ ∈ {2, 4} are honest held-out (same protocol as
the other paper models).

Env knobs:
  YEARS           default "2014 2015 2016 2017 2018 2019"
  MAX_EPOCHS      default 8
  BATCH_SIZE      default 2
  LR              default 1e-4
  NUM_WORKERS     default 8
  OUT_DIR         default logs/exp_wbhybrid_6yr
  BLOCK_OUT       default "96 192 384"  (space-separated)
  LATENT          default 192
  LAYERS_PER_BLOCK default "2 2 2"
"""
import os
import sys
import time

sys.path.insert(0, "/home/jovyan/dsuhoi/weather_time_interpolation")
import torch

torch.set_float32_matmul_precision("high")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import WeatherHermiteLightningModule, ERA5WeatherHermiteDataset


N_KEEP = 24  # 20 PL + 4 surface


class TruncateChannelsWrapper(Dataset):
    def __init__(self, base, n_keep=N_KEEP):
        self.base = base; self.n_keep = n_keep
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        out = self.base[idx]
        out["x0"] = out["x0"][..., :self.n_keep, :, :].clone()
        out["xT"] = out["xT"][..., :self.n_keep, :, :].clone()
        out["target"] = out["target"][..., :self.n_keep, :, :].clone()
        return out


def main():
    years = list(map(int, os.environ.get("YEARS",
                                          "2014 2015 2016 2017 2018 2019").split()))
    max_epochs = int(os.environ.get("MAX_EPOCHS", "8"))
    batch_size = int(os.environ.get("BATCH_SIZE", "2"))
    lr = float(os.environ.get("LR", "1e-4"))
    num_workers = int(os.environ.get("NUM_WORKERS", "8"))
    out_dir = os.environ.get("OUT_DIR", "logs/exp_wbhybrid_6yr")
    block_out = tuple(int(x) for x in os.environ.get("BLOCK_OUT", "96 192 384").split())
    latent = int(os.environ.get("LATENT", "192"))
    layers = tuple(int(x) for x in os.environ.get("LAYERS_PER_BLOCK", "2 2 2").split())

    print("=== WeatherDCAE-Hybrid training ===")
    print(f"  years={years}, epochs={max_epochs}, batch={batch_size}, lr={lr}")
    print(f"  block_out={block_out}, layers={layers}, latent={latent}")
    print(f"  out={out_dir}", flush=True)
    t0 = time.time()

    ds_base = ERA5MemmapDataset(
        memmap_dir="/tmp/wb2_0p5_cache",
        years=years, max_tau_hours=6, samples_per_date=4, train=True,
        train_hours=[1, 3, 5],
        static_path="data/static_features_0p5.pt",
        stats_path="data/json_stats_0p5.nc",
        surface_stats_path="data/surface_stats_0p5.json",
    )
    cg_full = ds_base.channel_groups
    cg_24 = {k: v for k, v in cg_full.items() if k not in ("sst", "tcc", "tcwv")}
    print(f"  channel_groups (24-ch): {list(cg_24.keys())}")

    ds_train = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=6.0)
    ds_train = TruncateChannelsWrapper(ds_train, n_keep=N_KEEP)
    print(f"train samples: {len(ds_train)}", flush=True)

    loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0)

    model = WeatherHermiteLightningModule(
        model_type="weatherbridge_hybrid",
        channel_groups=cg_24,
        n_pl_channels=20, n_surface_channels=4, n_static_features=3,
        latent_channels=latent,
        block_out_channels=block_out,
        layers_per_block=layers,
        lat_weighted_loss=True, lat_crop=-8,
        lambda_residual=1.0, residual_scale_floor=0.05, residual_scale_init=0.30,
        lambda_anchor=0.5, anchor_every_n_batches=16,
        lr=lr, weight_decay=1e-5,
        use_aurora_weights=True, use_physical_scales_loss=False,
        max_tau_hours=6,
    )
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params: {n:.2f}M", flush=True)

    ckpt_cb = ModelCheckpoint(dirpath=out_dir, save_last=True,
                              save_top_k=-1, every_n_epochs=2)
    trainer = pl.Trainer(
        max_epochs=max_epochs, log_every_n_steps=20,
        devices=1, accelerator="gpu", precision="bf16-mixed",
        limit_val_batches=0, num_sanity_val_steps=0,
        enable_checkpointing=True, enable_progress_bar=False,
        callbacks=[ckpt_cb], default_root_dir=out_dir,
    )
    trainer.fit(model, train_dataloaders=loader)
    print(f"=== DONE in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
