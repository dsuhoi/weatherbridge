# Detailed Weather Interpolation Benchmark

Lower RMSE, spectral energy/shape error, and balance-ratio distance from one are better; higher ACC and coherence are better.

## Aggregate RMSE and ACC

| Horizon | Year | Model | RMSE norm | ACC | Field-hour cells |
|---|---:|---|---:|---:|---:|
| 6h | 2020 | Flow-Spectral | 0.062628 | 0.995411 | 120 |
| 6h | 2020 | Linear interpolation | 0.126800 | 0.985966 | 120 |
| 6h | 2020 | Refine | 0.062428 | 0.995356 | 120 |
| 6h | 2020 | WeatherBridge | 0.062454 | 0.995421 | 120 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | 0.070749 | 0.993334 | 120 |
| 6h | 2021 | Flow-Spectral | 0.062375 | 0.995431 | 120 |
| 6h | 2021 | Linear interpolation | 0.126303 | 0.986034 | 120 |
| 6h | 2021 | Refine | 0.062185 | 0.995374 | 120 |
| 6h | 2021 | WeatherBridge | 0.062211 | 0.995442 | 120 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | 0.070396 | 0.993363 | 120 |

## WeatherBridge detail-path ablation by hour

| Horizon | Year | Baseline | Hour | Delta RMSE | Wins |
|---|---:|---|---:|---:|---:|
| 6h | 2020 | Flow-Spectral | 1 | +0.35% | 14/24 |
| 6h | 2020 | Flow-Spectral | 2 | -1.72% | 19/24 |
| 6h | 2020 | Flow-Spectral | 3 | +0.25% | 16/24 |
| 6h | 2020 | Flow-Spectral | 4 | +1.21% | 5/24 |
| 6h | 2020 | Flow-Spectral | 5 | +0.23% | 13/24 |
| 6h | 2020 | Linear interpolation | 1 | -47.32% | 24/24 |
| 6h | 2020 | Linear interpolation | 2 | -46.69% | 24/24 |
| 6h | 2020 | Linear interpolation | 3 | -49.96% | 24/24 |
| 6h | 2020 | Linear interpolation | 4 | -42.82% | 24/24 |
| 6h | 2020 | Linear interpolation | 5 | -45.69% | 24/24 |
| 6h | 2020 | Refine | 1 | -1.41% | 22/24 |
| 6h | 2020 | Refine | 2 | +2.11% | 8/24 |
| 6h | 2020 | Refine | 3 | -1.79% | 24/24 |
| 6h | 2020 | Refine | 4 | +2.97% | 3/24 |
| 6h | 2020 | Refine | 5 | -1.57% | 23/24 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | 1 | -13.71% | 24/24 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | 2 | -9.35% | 19/24 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | 3 | -12.45% | 24/24 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | 4 | -7.70% | 18/24 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | 5 | -12.93% | 24/24 |
| 6h | 2021 | Flow-Spectral | 1 | +0.40% | 14/24 |
| 6h | 2021 | Flow-Spectral | 2 | -1.74% | 19/24 |
| 6h | 2021 | Flow-Spectral | 3 | +0.25% | 16/24 |
| 6h | 2021 | Flow-Spectral | 4 | +1.23% | 5/24 |
| 6h | 2021 | Flow-Spectral | 5 | +0.24% | 13/24 |
| 6h | 2021 | Linear interpolation | 1 | -47.32% | 24/24 |
| 6h | 2021 | Linear interpolation | 2 | -46.78% | 24/24 |
| 6h | 2021 | Linear interpolation | 3 | -49.97% | 24/24 |
| 6h | 2021 | Linear interpolation | 4 | -42.81% | 24/24 |
| 6h | 2021 | Linear interpolation | 5 | -45.69% | 24/24 |
| 6h | 2021 | Refine | 1 | -1.44% | 22/24 |
| 6h | 2021 | Refine | 2 | +2.04% | 8/24 |
| 6h | 2021 | Refine | 3 | -1.81% | 24/24 |
| 6h | 2021 | Refine | 4 | +3.04% | 3/24 |
| 6h | 2021 | Refine | 5 | -1.59% | 23/24 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | 1 | -13.55% | 24/24 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | 2 | -9.45% | 19/24 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | 3 | -12.38% | 24/24 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | 4 | -7.54% | 18/24 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | 5 | -12.79% | 24/24 |

## RMSE exceptions

Showing the 30 largest regressions out of 208; the CSV contains every field-hour cell.

- 6h 2021 h=4 Z700 vs Refine: +11.513%
- 6h 2020 h=4 Z700 vs Refine: +10.912%
- 6h 2020 h=2 Z700 vs Refine: +9.824%
- 6h 2021 h=2 Z700 vs Refine: +9.366%
- 6h 2021 h=4 Z850 vs Refine: +9.146%
- 6h 2020 h=2 Z850 vs Refine: +8.748%
- 6h 2020 h=4 Z850 vs Refine: +8.687%
- 6h 2021 h=4 Z700 vs WeatherDCAE-14M (6-year): +8.588%
- 6h 2021 h=2 Z850 vs Refine: +8.283%
- 6h 2020 h=2 Z925 vs Refine: +8.246%
- 6h 2021 h=4 Z925 vs Refine: +8.202%
- 6h 2021 h=4 Z850 vs WeatherDCAE-14M (6-year): +7.948%
- 6h 2021 h=2 Z925 vs Refine: +7.805%
- 6h 2020 h=4 Z925 vs Refine: +7.787%
- 6h 2020 h=2 Z925 vs WeatherDCAE-14M (6-year): +7.787%
- 6h 2020 h=4 Z700 vs WeatherDCAE-14M (6-year): +7.726%
- 6h 2021 h=4 Z1000 vs Refine: +7.680%
- 6h 2020 h=2 Z850 vs WeatherDCAE-14M (6-year): +7.653%
- 6h 2020 h=2 Z1000 vs Refine: +7.643%
- 6h 2021 h=4 mslp vs Refine: +7.592%
- 6h 2020 h=2 mslp vs Refine: +7.441%
- 6h 2020 h=4 Z1000 vs Refine: +7.338%
- 6h 2020 h=4 Z850 vs WeatherDCAE-14M (6-year): +7.326%
- 6h 2020 h=4 mslp vs Refine: +7.265%
- 6h 2021 h=2 Z1000 vs Refine: +7.243%
- 6h 2021 h=4 Z925 vs WeatherDCAE-14M (6-year): +7.122%
- 6h 2021 h=2 mslp vs Refine: +7.062%
- 6h 2020 h=2 Z1000 vs WeatherDCAE-14M (6-year): +7.010%
- 6h 2021 h=2 Z925 vs WeatherDCAE-14M (6-year): +6.943%
- 6h 2020 h=2 Z700 vs WeatherDCAE-14M (6-year): +6.848%

## Spherical spectrum

| Horizon | Year | Model | Energy ratio | Energy error | Shape error | Coherence |
|---|---:|---|---:|---:|---:|---:|

## Diagnostic balance

| Horizon | Year | Model | Ageo 850 | Ageo 700 | Hydrostatic |
|---|---:|---|---:|---:|---:|
| 6h | 2020 | Flow-Spectral | 0.9492 | 0.9402 | 0.9923 |
| 6h | 2020 | Linear interpolation | 0.9364 | 0.9032 | 0.9928 |
| 6h | 2020 | Refine | 0.9870 | 0.9766 | 0.9948 |
| 6h | 2020 | WeatherBridge | 0.9891 | 0.9781 | 0.9946 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | 0.9620 | 0.9624 | 0.9942 |
| 6h | 2021 | Flow-Spectral | 0.9491 | 0.9398 | 0.9926 |
| 6h | 2021 | Linear interpolation | 0.9356 | 0.9043 | 0.9931 |
| 6h | 2021 | Refine | 0.9875 | 0.9772 | 0.9951 |
| 6h | 2021 | WeatherBridge | 0.9892 | 0.9785 | 0.9948 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | 0.9610 | 0.9617 | 0.9944 |

## Regional diurnal cycle

| Horizon | Year | Model | Amplitude error | Peak error (h) | Strata |
|---|---:|---|---:|---:|---:|
| 6h | 2020 | Flow-Spectral | 0.0432 | 0.507 | 144 |
| 6h | 2020 | Linear interpolation | 0.1704 | 1.569 | 144 |
| 6h | 2020 | Refine | 0.0381 | 0.431 | 144 |
| 6h | 2020 | WeatherBridge | 0.0464 | 0.604 | 144 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | 0.0470 | 0.674 | 144 |
| 6h | 2021 | Flow-Spectral | 0.0470 | 0.458 | 144 |
| 6h | 2021 | Linear interpolation | 0.1794 | 1.778 | 144 |
| 6h | 2021 | Refine | 0.0438 | 0.479 | 144 |
| 6h | 2021 | WeatherBridge | 0.0473 | 0.472 | 144 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | 0.0533 | 0.604 | 144 |

## Paired weekly-block tests

Holm correction is applied within each horizon-year-baseline-metric family.

| Horizon | Year | Baseline | Metric | Significant | Tests |
|---|---:|---|---|---:|---:|
| 6h | 2020 | Flow-Spectral | acc | 6 | 6 |
| 6h | 2020 | Flow-Spectral | physical_global_mslp_bias | 1 | 1 |
| 6h | 2020 | Flow-Spectral | physical_hydrostatic_balance_mse | 1 | 1 |
| 6h | 2020 | Flow-Spectral | physical_kinetic_energy_nmse | 1 | 1 |
| 6h | 2020 | Flow-Spectral | physical_lower_tropospheric_moisture_bias | 1 | 1 |
| 6h | 2020 | Flow-Spectral | physical_q_negative_fraction | 1 | 1 |
| 6h | 2020 | Flow-Spectral | physical_wind_divergence_nmse | 1 | 1 |
| 6h | 2020 | Flow-Spectral | physical_wind_vorticity_nmse | 1 | 1 |
| 6h | 2020 | Flow-Spectral | rmse_norm | 6 | 6 |
| 6h | 2020 | Flow-Spectral | temporal_curvature_rmse | 1 | 1 |
| 6h | 2020 | Refine | acc | 6 | 6 |
| 6h | 2020 | Refine | physical_global_mslp_bias | 1 | 1 |
| 6h | 2020 | Refine | physical_hydrostatic_balance_mse | 1 | 1 |
| 6h | 2020 | Refine | physical_kinetic_energy_nmse | 1 | 1 |
| 6h | 2020 | Refine | physical_lower_tropospheric_moisture_bias | 1 | 1 |
| 6h | 2020 | Refine | physical_q_negative_fraction | 1 | 1 |
| 6h | 2020 | Refine | physical_wind_divergence_nmse | 1 | 1 |
| 6h | 2020 | Refine | physical_wind_vorticity_nmse | 1 | 1 |
| 6h | 2020 | Refine | rmse_norm | 6 | 6 |
| 6h | 2020 | Refine | temporal_curvature_rmse | 1 | 1 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | acc | 6 | 6 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | physical_global_mslp_bias | 1 | 1 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | physical_hydrostatic_balance_mse | 1 | 1 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | physical_kinetic_energy_nmse | 1 | 1 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | physical_lower_tropospheric_moisture_bias | 1 | 1 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | physical_q_negative_fraction | 1 | 1 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | physical_wind_divergence_nmse | 1 | 1 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | physical_wind_vorticity_nmse | 1 | 1 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | rmse_norm | 6 | 6 |
| 6h | 2020 | WeatherDCAE-14M (6-year) | temporal_curvature_rmse | 1 | 1 |
| 6h | 2021 | Flow-Spectral | acc | 6 | 6 |
| 6h | 2021 | Flow-Spectral | physical_global_mslp_bias | 1 | 1 |
| 6h | 2021 | Flow-Spectral | physical_hydrostatic_balance_mse | 1 | 1 |
| 6h | 2021 | Flow-Spectral | physical_kinetic_energy_nmse | 1 | 1 |
| 6h | 2021 | Flow-Spectral | physical_lower_tropospheric_moisture_bias | 1 | 1 |
| 6h | 2021 | Flow-Spectral | physical_q_negative_fraction | 1 | 1 |
| 6h | 2021 | Flow-Spectral | physical_wind_divergence_nmse | 1 | 1 |
| 6h | 2021 | Flow-Spectral | physical_wind_vorticity_nmse | 1 | 1 |
| 6h | 2021 | Flow-Spectral | rmse_norm | 5 | 6 |
| 6h | 2021 | Flow-Spectral | temporal_curvature_rmse | 1 | 1 |
| 6h | 2021 | Refine | acc | 6 | 6 |
| 6h | 2021 | Refine | physical_global_mslp_bias | 1 | 1 |
| 6h | 2021 | Refine | physical_hydrostatic_balance_mse | 1 | 1 |
| 6h | 2021 | Refine | physical_kinetic_energy_nmse | 1 | 1 |
| 6h | 2021 | Refine | physical_lower_tropospheric_moisture_bias | 1 | 1 |
| 6h | 2021 | Refine | physical_q_negative_fraction | 1 | 1 |
| 6h | 2021 | Refine | physical_wind_divergence_nmse | 1 | 1 |
| 6h | 2021 | Refine | physical_wind_vorticity_nmse | 1 | 1 |
| 6h | 2021 | Refine | rmse_norm | 6 | 6 |
| 6h | 2021 | Refine | temporal_curvature_rmse | 1 | 1 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | acc | 6 | 6 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | physical_global_mslp_bias | 1 | 1 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | physical_hydrostatic_balance_mse | 1 | 1 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | physical_kinetic_energy_nmse | 1 | 1 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | physical_lower_tropospheric_moisture_bias | 1 | 1 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | physical_q_negative_fraction | 1 | 1 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | physical_wind_divergence_nmse | 1 | 1 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | physical_wind_vorticity_nmse | 1 | 1 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | rmse_norm | 6 | 6 |
| 6h | 2021 | WeatherDCAE-14M (6-year) | temporal_curvature_rmse | 1 | 1 |
