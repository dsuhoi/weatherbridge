#!/usr/bin/env python3
"""One-shot per-channel reconstruction RMSE on val set for AE ckpt.

Usage: CKPT=path GPU=1 python3 eval_ae_per_channel.py
"""
import os, sys, json
sys.path.insert(0, '/home/jovyan/dsuhoi/weather_time_interpolation')
sys.path.insert(0, '/tmp')
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('high')

from weather_time_interp.memmap_dataset import ERA5MemmapDataset

# Import the model class from the trainer script
_trainer_src = open('/tmp/train_dcae_autoencoder_6yr.py').read()
_ns = {}
exec(_trainer_src.replace("if __name__ == '__main__':", "if False:"), _ns)
WeatherDCAEAutoencoder = _ns['WeatherDCAEAutoencoder']

CH_27 = ["T1000","T925","T850","T700","U1000","U925","U850","U700","V1000","V925","V850","V700",
         "Q1000","Q925","Q850","Q700","Z1000","Z925","Z850","Z700",
         "t2m","u10","v10","mslp","sst","tcc","tcwv"]


def lat_weights(H, device):
    lat = np.linspace(89.75, -89.75, H, dtype=np.float32)
    w = np.cos(np.deg2rad(lat)); w = w / w.sum()
    return torch.from_numpy(w).to(device=device, dtype=torch.float32).view(1, 1, -1, 1)


def main():
    ckpt = os.environ['CKPT']
    gpu = int(os.environ.get('GPU', 1))
    device = torch.device(f'cuda:{gpu}')

    ds = ERA5MemmapDataset(
        memmap_dir='/tmp/wb2_0p5_cache', years=[2020],
        max_tau_hours=6, samples_per_date=4, train=False,
        eval_hours=[1, 2, 3, 4, 5],
        static_path='data/static_features_0p5.pt',
        stats_path='data/json_stats_0p5.nc',
        surface_stats_path='data/surface_stats_0p5.json',
    )
    print(f'val items: {len(ds)}')

    model = WeatherDCAEAutoencoder.load_from_checkpoint(ckpt, map_location='cpu')
    model.to(device).eval()
    print(f'loaded {ckpt}')

    lw = lat_weights(360, device).squeeze()
    sq_sum = torch.zeros(27, device=device)
    n = 0

    loader = DataLoader(ds, batch_size=4, num_workers=4, shuffle=False)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for i, batch in enumerate(loader):
            x = batch['x0'].to(device)
            x_rec, _ = model(x)
            err = (x_rec.float() - x.float()) ** 2          # (B, 27, H, W)
            err_lw = (err * lw.view(1, 1, -1, 1)).sum(dim=(2, 3))  # (B, 27)
            sq_sum += err_lw.sum(dim=0)
            n += err_lw.size(0)
            if i % 100 == 0:
                print(f'  batch {i}/{len(loader)}', flush=True)

    rmse = (sq_sum / n).sqrt().cpu().numpy()
    res = {'ckpt': ckpt, 'n_samples': n,
           'channels': CH_27, 'rmse_normalized': rmse.tolist()}

    print()
    print(f'{"channel":<10}{"RMSE (norm.)":>16}')
    print('-' * 26)
    for ch, r in zip(CH_27, rmse):
        print(f'{ch:<10}{r:>16.5f}')
    print()

    out = Path(ckpt).parent / f'eval_per_channel_{Path(ckpt).stem}.json'
    json.dump(res, open(out, 'w'), indent=2)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
