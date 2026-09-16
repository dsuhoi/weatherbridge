#!/usr/bin/env python3
"""Offline eval for Phase 2 checkpoints: per-tau × per-channel RMSE breakdown.

Loads ckpt → runs val_loader → accumulates lat-weighted sq_err separately for each
(tau_hour ∈ {1,2,3,4,5}, channel ∈ 27). Computes:
  • RMSE per (tau, channel) for the model (bilinear + scale·δ)
  • RMSE per (tau, channel) for pure bilinear baseline
  • Per-channel relative improvement vs bilinear (averaged across τ)
  • Per-tau aggregate (avg over channels)

Output: JSON ready for paper plots (per-hour curves like paper/per_hour mosaic).

Usage:
    CKPT=path/to/last.ckpt PHASE1_TRAINER=/tmp/train_dcae_autoencoder_6yr.py \\
    python eval_phase2_per_tau.py
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('high')

PHASE1 = os.environ.get('PHASE1_TRAINER', '/tmp/train_dcae_autoencoder_6yr.py')
PHASE2_TRAINER = os.environ.get('PHASE2_TRAINER', '/tmp/train_phase2_adaln_zero.py')

_ns = {}
exec(open(PHASE1).read().replace("if __name__ == '__main__':", "if False:"), _ns)
WeatherDCAEAutoencoder = _ns['WeatherDCAEAutoencoder']
CH_27 = _ns['CH_27']

# Phase 2 model class
_ns2 = {}
exec(open(PHASE2_TRAINER).read().replace("if __name__ == '__main__':", "if False:"), _ns2)
Phase2AdaLNZero = _ns2['Phase2AdaLNZero']

sys.path.insert(0, '/home/jovyan/dsuhoi/weather_time_interpolation')
sys.path.insert(0, '/workspace/code/wti')
from weather_time_interp.memmap_dataset import ERA5MemmapDataset


def main():
    ckpt_path = os.environ['CKPT']
    out_dir = Path(os.environ.get('OUT_DIR',
        '/home/jovyan/dsuhoi/weather_time_interpolation/metrics/eval_phase2_per_tau'))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = os.environ.get('OUT_NAME', Path(ckpt_path).stem)
    memmap_dir = os.environ.get('MEMMAP_DIR', '/tmp/wb2_0p5_cache')
    device = torch.device('cuda')

    print(f'loading Phase 2 ckpt: {ckpt_path}')
    model = Phase2AdaLNZero.load_from_checkpoint(ckpt_path, map_location='cpu')
    model.to(device).eval()

    val_ds = ERA5MemmapDataset(
        memmap_dir=memmap_dir, years=[2020],
        max_tau_hours=6, samples_per_date=4, train=False,
        eval_hours=[1, 2, 3, 4, 5],
        static_path='data/static_features_0p5.pt',
        stats_path='data/json_stats_0p5.nc',
        surface_stats_path='data/surface_stats_0p5.json',
    )
    print(f'val items: {len(val_ds)}')

    loader = DataLoader(val_ds, batch_size=4, num_workers=4, shuffle=False)
    # Accumulators per (h, ch)
    H_LIST = [1, 2, 3, 4, 5]
    sums_model = {h: torch.zeros(27, device=device) for h in H_LIST}
    sums_bil = {h: torch.zeros(27, device=device) for h in H_LIST}
    counts = {h: 0 for h in H_LIST}

    lat_w = model.ae.lat_w                                                      # (1,1,H,1)
    n_px = model.ae.land_mask.numel()

    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for i, batch in enumerate(loader):
            x_0 = batch['x0'].to(device)
            x_T = batch['x1'].to(device)
            x_tgt = batch['target'].to(device)
            tau = batch['tau'].to(device)
            tau_h = batch['tau_hour'].to(device).squeeze(-1)
            if tau.dim() == 1: tau_b = tau.view(-1, 1, 1, 1)
            elif tau.dim() == 2: tau_b = tau.view(-1, 1, 1, 1)
            else: tau_b = tau
            tau_scalar = tau.view(-1) if tau.dim() <= 2 else tau.flatten()[:x_0.size(0)]

            x_bilinear = (1.0 - tau_b) * x_0 + tau_b * x_T
            delta, _ = model(x_tgt, tau_scalar)
            s = model.scale.view(1, -1, 1, 1)
            x_pred = x_bilinear + s * delta

            # SST mask
            x_pred_m = model.ae._apply_sst_mask(x_pred)
            x_tgt_m = model.ae._apply_sst_mask(x_tgt)
            x_bil_m = model.ae._apply_sst_mask(x_bilinear)

            sq_model = (x_pred_m.float() - x_tgt_m.float()) ** 2
            sq_bil = (x_bil_m.float() - x_tgt_m.float()) ** 2
            sq_model_lw = (sq_model * lat_w).sum(dim=(-2, -1))                  # (B, 27)
            sq_bil_lw = (sq_bil * lat_w).sum(dim=(-2, -1))

            for k, h_val in enumerate(tau_h.tolist()):
                h = int(h_val)
                if h in sums_model:
                    sums_model[h] += sq_model_lw[k]
                    sums_bil[h] += sq_bil_lw[k]
                    counts[h] += 1

            if i % 100 == 0:
                elapsed = time.time() - t0
                est_total = elapsed * len(loader) / max(1, i + 1)
                print(f'  batch {i}/{len(loader)}  elapsed={elapsed:.0f}s  est_total={est_total:.0f}s', flush=True)

    # Compute RMSE per (h, channel)
    result = {
        'ckpt': str(ckpt_path),
        'n_samples_per_h': {str(h): counts[h] for h in H_LIST},
        'channel_names': CH_27,
        'per_hour_model': {},
        'per_hour_bilinear': {},
        'per_hour_relative_improvement': {},
    }
    for h in H_LIST:
        if counts[h] > 0:
            mse_model = sums_model[h] / (counts[h] * n_px)
            mse_bil = sums_bil[h] / (counts[h] * n_px)
            rmse_model = mse_model.sqrt()
            rmse_bil = mse_bil.sqrt()
            improve = (1.0 - rmse_model / rmse_bil.clamp(min=1e-8))
            result['per_hour_model'][str(h)] = rmse_model.cpu().tolist()
            result['per_hour_bilinear'][str(h)] = rmse_bil.cpu().tolist()
            result['per_hour_relative_improvement'][str(h)] = improve.cpu().tolist()

    # Aggregate
    all_h_model_mse = sum([sums_model[h] for h in H_LIST]) / (sum(counts.values()) * n_px)
    all_h_bil_mse = sum([sums_bil[h] for h in H_LIST]) / (sum(counts.values()) * n_px)
    result['aggregate_rmse_model'] = all_h_model_mse.sqrt().cpu().tolist()
    result['aggregate_rmse_bilinear'] = all_h_bil_mse.sqrt().cpu().tolist()

    out_path = out_dir / f'{out_name}.json'
    json.dump(result, open(out_path, 'w'), indent=2)
    print(f'\nsaved {out_path}  ({(time.time()-t0)/60:.1f} min)')

    # Print summary
    print('\n=== Per-hour summary (avg over channels of relative improvement) ===')
    for h in H_LIST:
        if str(h) in result['per_hour_relative_improvement']:
            ri = np.mean(result['per_hour_relative_improvement'][str(h)])
            print(f'  h={h}: relative_improvement = {ri*100:.1f}%')

    print('\n=== Per-channel relative improvement (averaged over τ) ===')
    avg_ri_per_ch = np.zeros(27)
    n_h = 0
    for h in H_LIST:
        if str(h) in result['per_hour_relative_improvement']:
            avg_ri_per_ch += np.array(result['per_hour_relative_improvement'][str(h)])
            n_h += 1
    avg_ri_per_ch /= max(1, n_h)
    for i, ch in enumerate(CH_27):
        print(f'  {ch:<8}: {avg_ri_per_ch[i]*100:+.1f}%')


if __name__ == '__main__':
    main()
