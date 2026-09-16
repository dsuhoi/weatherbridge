#!/usr/bin/env python3
"""Encode all weather fields to DC-AE latents and save to disk.

Output: latents_<year>.npz with keys 'z' (N, 64, 22, 46) and 'time_idx' (N,).
Run once after AE converges → downstream latent VFI training reads these directly.

Usage:
    CKPT=path/to/last.ckpt YEARS=2014,2015,... python encode_dataset_to_latents.py
"""
import os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')

sys.path.insert(0, '/home/jovyan/dsuhoi/weather_time_interpolation')
from weather_time_interp.memmap_dataset import ERA5MemmapDataset

# Load model class from trainer file
_ns = {}
exec(open('/tmp/train_dcae_autoencoder_6yr.py').read()
     .replace("if __name__ == '__main__':", "if False:"), _ns)
WeatherDCAEAutoencoder = _ns['WeatherDCAEAutoencoder']


def main():
    ckpt = os.environ.get('CKPT', '/home/jovyan/dsuhoi/weather_time_interpolation/logs/exp_dcae_ae_static_6yr/last.ckpt')
    years = [int(y) for y in os.environ.get('YEARS', '2014,2015,2016,2017,2018,2019,2020').split(',')]
    out_dir = Path(os.environ.get('OUT_DIR', '/home/jovyan/dsuhoi/weather_time_interpolation/latents_dcae_f16'))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')

    print(f'loading AE from {ckpt}')
    model = WeatherDCAEAutoencoder.load_from_checkpoint(ckpt, map_location='cpu')
    model.eval().to(device)
    print(f'ready, latent_ch={model.hparams.latent_channels}')

    for year in years:
        ds = ERA5MemmapDataset(
            memmap_dir='/tmp/wb2_0p5_cache', years=[year],
            max_tau_hours=6, samples_per_date=4, train=False,
            eval_hours=[1, 2, 3, 4, 5],
            static_path='data/static_features_0p5.pt',
            stats_path='data/json_stats_0p5.nc',
            surface_stats_path='data/surface_stats_0p5.json',
        )
        # We want one latent per timestamp; the dataset gives pair-samples
        # so iterate over the underlying memmap directly:
        arr = ds._arrs[0]                                  # (T, 27, 360, 720)
        T = arr.shape[0]
        print(f'year {year}: T={T} timestamps')

        latents = np.empty((T, 64, 22, 46), dtype=np.float16)  # fp16 for storage
        t0 = time.time()
        bs = 16
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, T, bs):
                j = min(i + bs, T)
                x = torch.from_numpy(np.ascontiguousarray(arr[i:j])).float().to(device)
                # Apply same crop+pad as model.forward
                x_c = model._crop_lat(x)
                x_c = model._pad_lon(x_c)
                z = model.encoder(x_c)                     # (B, 64, 22, 46)
                latents[i:j] = z.cpu().float().numpy().astype(np.float16)
                if i % 1024 == 0:
                    el = time.time() - t0
                    print(f'  {i}/{T}  elapsed={el:.1f}s  est_total={el*T/(i+1):.1f}s', flush=True)

        out_path = out_dir / f'latents_{year}.npz'
        np.savez_compressed(out_path, z=latents,
                            time_idx=np.arange(T, dtype=np.int32))
        size_mb = out_path.stat().st_size / 1e6
        print(f'saved {out_path}  size={size_mb:.1f} MB  ({time.time()-t0:.1f}s)')


if __name__ == '__main__':
    main()
