"""Pull per-channel val RMSE for both AE trainings + denormalize to physical units.

Uses json_stats_0p5.nc and surface_stats_0p5.json to recover channel-wise std.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, '/home/jovyan/dsuhoi/weather_time_interpolation')
sys.path.insert(0, '/workspace/code/wti')

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CH_27 = ["T1000","T925","T850","T700","U1000","U925","U850","U700","V1000","V925","V850","V700",
         "Q1000","Q925","Q850","Q700","Z1000","Z925","Z850","Z700",
         "t2m","u10","v10","mslp","sst","tcc","tcwv"]

# Load stats
try:
    import xarray as xr
    stats_path = 'data/json_stats_0p5.nc' if os.path.exists('data/json_stats_0p5.nc') \
                 else '/workspace/code/wti/data/json_stats_0p5.nc'
    ds = xr.open_dataset(stats_path)
    print('PL stats vars:', list(ds.data_vars)[:6])
    # The pl stats typically have shape (n_var, n_lev). Need to figure out exact structure.
except Exception as e:
    print(f'xarray fail: {e}')
    ds = None

try:
    sfc_path = 'data/surface_stats_0p5.json' if os.path.exists('data/surface_stats_0p5.json') \
               else '/workspace/code/wti/data/surface_stats_0p5.json'
    sfc_stats = json.load(open(sfc_path))
    print('Surface stats keys:', list(sfc_stats.keys())[:5])
    # Typically {'t2m': {'mean': ..., 'std': ...}, ...}
except Exception as e:
    print(f'sfc stats fail: {e}')

# Hardcoded approximate stds from WB-2 / ERA5 typical values
# (will use these as fallback if file parsing fails)
APPROX_STDS = {
    'T1000': 13.5, 'T925': 14.2, 'T850': 14.0, 'T700': 13.0,
    'U1000': 5.8,  'U925': 6.5,  'U850': 7.5,  'U700': 9.5,
    'V1000': 4.8,  'V925': 5.5,  'V850': 6.5,  'V700': 7.5,
    'Q1000': 0.0058, 'Q925': 0.0055, 'Q850': 0.0048, 'Q700': 0.0028,
    'Z1000': 87.0, 'Z925': 105.0, 'Z850': 130.0, 'Z700': 195.0,  # m²/s² actually but typical units
    't2m': 15.5, 'u10': 5.6, 'v10': 4.8, 'mslp': 1140.0,
    'sst': 11.2, 'tcc': 0.43, 'tcwv': 15.5,
}
UNITS = {
    'T1000': 'K', 'T925': 'K', 'T850': 'K', 'T700': 'K',
    'U1000': 'm/s', 'U925': 'm/s', 'U850': 'm/s', 'U700': 'm/s',
    'V1000': 'm/s', 'V925': 'm/s', 'V850': 'm/s', 'V700': 'm/s',
    'Q1000': 'kg/kg', 'Q925': 'kg/kg', 'Q850': 'kg/kg', 'Q700': 'kg/kg',
    'Z1000': 'm²/s²', 'Z925': 'm²/s²', 'Z850': 'm²/s²', 'Z700': 'm²/s²',
    't2m': 'K', 'u10': 'm/s', 'v10': 'm/s', 'mslp': 'Pa',
    'sst': 'K', 'tcc': '0-1', 'tcwv': 'kg/m²',
}

# Try to get exact stds from stats files
ch_std = dict(APPROX_STDS)
if ds is not None:
    try:
        # PL: shape (5, 4) for T/U/V/Q/Z × 4 levels (1000, 925, 850, 700)
        # var names like 'temperature', 'u_component_of_wind', etc.
        var_to_short = {'temperature': 'T', 'u_component_of_wind': 'U',
                        'v_component_of_wind': 'V', 'specific_humidity': 'Q',
                        'geopotential': 'Z'}
        levels = [1000, 925, 850, 700]
        for full, short in var_to_short.items():
            if full + '_std' in ds.data_vars or 'std_' + full in ds.data_vars or full in ds.data_vars:
                arr = ds.get(full + '_std') if full + '_std' in ds.data_vars else ds.get(full)
                vals = np.asarray(arr.values).flatten()
                for i, lev in enumerate(levels):
                    if i < len(vals):
                        ch_std[f'{short}{lev}'] = float(vals[i])
    except Exception as e:
        print(f'pl parse fail: {e}')

if 'sfc_stats' in dir():
    for ch in ['t2m', 'u10', 'v10', 'mslp', 'sst', 'tcc', 'tcwv']:
        if ch in sfc_stats:
            d = sfc_stats[ch]
            if isinstance(d, dict) and 'std' in d:
                ch_std[ch] = float(d['std'])

# Get latest val_rmse from each training
def get_val_rmse(log_dir):
    if not os.path.exists(log_dir):
        return None
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    rmse = {}
    for ch in CH_27:
        tag = f'val_rmse/{ch}'
        if tag in tags:
            ev = ea.Scalars(tag)
            if ev: rmse[ch] = ev[-1].value
    return rmse

cr_path = '/home/jovyan/dsuhoi/weather_time_interpolation/logs/exp_dcae_ae_static_6yr/lightning_logs/version_0'
fb_path = '/workspace/code/wti/logs/exp_dcae_ae_f16_lat128_3yr_fibo_ddp/lightning_logs/version_0'

cr_rmse = get_val_rmse(cr_path)
fb_rmse = get_val_rmse(fb_path)

print('\n' + '=' * 80)
print(f'{"channel":<8} {"unit":<8} {"std":>10} '
      f'{"cloud.ru norm":>14} {"cloud.ru phys":>14} '
      f'{"fibo norm":>10} {"fibo phys":>12}')
print('-' * 80)
for ch in CH_27:
    std = ch_std.get(ch, 1.0)
    unit = UNITS.get(ch, '-')
    cr = cr_rmse.get(ch, float('nan')) if cr_rmse else float('nan')
    fb = fb_rmse.get(ch, float('nan')) if fb_rmse else float('nan')
    cr_phys = cr * std
    fb_phys = fb * std
    print(f'{ch:<8} {unit:<8} {std:>10.4g} '
          f'{cr:>14.4f} {cr_phys:>14.4g} '
          f'{fb:>10.4f} {fb_phys:>12.4g}')
