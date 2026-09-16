# Per-Region Surface Variable Eval Spec (6h)

Supplementary data for AAAI 2027 paper. Per-region × per-τ × per-surface-channel × per-model RMSE and ACC, for 6h temporal interpolation on 2020 ERA5 test year.

Status: spec only — not run yet. Estimated runtime on fibo: ~6h sequential, ~2h with 3-way parallel (GPU 1/2/3).

## Regions (8)

Geographic bounding boxes, area-weighted by cos(lat). Order: continents first, oceans second.

| Region | Lat range | Lon range | Notes |
|---|---|---|---|
| `australia` | (−45, −10) | (110, 160) | Mainland + Tasmania |
| `north_america` | (15, 72) | (−170, −50) | CONUS + Canada + Alaska + Central Am. |
| `europe` | (35, 72) | (−15, 40) | Iberian Peninsula east to Urals |
| `africa` | (−35, 38) | (−20, 55) | Whole continent + Madagascar |
| `asia` | (5, 55) | (60, 150) | South + East + SE Asia, excl. Middle East |
| `north_atlantic` | (15, 65) | (−70, 0) | Bermuda Triangle → North Sea |
| `north_pacific` | (15, 65) | (135, 235) | Honshu → Vancouver Is. (lon wrap-around as 135–360+E ≤ 360 or 135E to −125E) |
| `southern_ocean` | (−75, −40) | (−180, 180) | Antarctic Circumpolar |

Implementation: in `tools/eval/region_per_channel_eval.py` add a `GEO_REGIONS` dict next to the existing latitude-band `REGIONS`. Mask = `(lat ∈ [lat_min, lat_max]) ∧ (lon ∈ [lon_min, lon_max])` with cos(lat) weighting on the lat axis. For `north_pacific` and `southern_ocean` handle dateline wrap-around (lon mask as `((lon ≥ 135) | (lon ≤ -125))` for N. Pacific).

## Channels (4 surface)

Mapped to ERA5 / WB-2 names:

| Symbol | ERA5 name | Unit |
|---|---|---|
| `t2m` | `2m_temperature` | K |
| `u10` | `10m_u_component_of_wind` | m/s |
| `v10` | `10m_v_component_of_wind` | m/s |
| `mslp` | `mean_sea_level_pressure` | Pa |

Per-channel reporting in normalised units (divide by per-channel std from 1979–2019 climatology), to match the headline leaderboard.

## Models (7)

Same as headline 6h leaderboard:

| Model | Checkpoint |
|---|---|
| Bilinear | closed form (no ckpt) |
| FuXi 24ch | `logs/exp_fuxi_swinv2_0p5_6yr/last.ckpt` |
| ModAFNO 24ch | `logs/exp_modafno_v9_0p5_6yr/last.ckpt` |
| S-DYff 24ch | `logs/exp_sdyff_0p5_6yr/last.ckpt` |
| ATM-VFI v2 | `logs/exp_atm_vfi_v2_0p5_6yr/last.ckpt` |
| WeatherDCAE 6yr (ours) | `logs/exp_dcae_noskip_0p5_6yr/last.ckpt` |
| CorrDiff/WeatherDCAE | `logs/exp_corrdiff_fm_weatherdcae_6h_3yr/last.ckpt` |

(Paths are fibo-local — verify on fibo with `ls /home/d.sukhorukov/weather_time_interpolation/logs/`.)

## Metrics

For each `(model, region, channel, τ ∈ {1,2,3,4,5})`:

1. **RMSE**: lat-weighted RMSE on the region mask, normalised by channel std. Formula:
   ```
   rmse_norm = sqrt( sum( w_lat * mask_geo * (pred - truth)^2 ) / sum( w_lat * mask_geo ) ) / std_ch
   ```
2. **ACC**: anomaly correlation against WB-2 hourly climatology (1990–2019), region-masked.
   - Anomaly: `a = field - clim(hour, lat, lon)`
   - ACC: weighted Pearson correlation of `a_pred` and `a_truth` over the region.
3. **Bootstrap CI** (optional, B=2000): paired bootstrap on the N=1463 test windows.

Total numbers per model: 8 regions × 4 channels × 5 τ × 2 metrics = **320**. Across 7 models: **2240 numbers**.

## Output schema

`metrics/region_per_channel_2020/{model}.json`:

```json
{
  "model": "WeatherDCAE_NoSkip_6yr",
  "year": 2020,
  "regions": ["australia", "north_america", ...],
  "channels": ["t2m", "u10", "v10", "mslp"],
  "taus": [1, 2, 3, 4, 5],
  "rmse_norm": {
    "australia": {"t2m": {"1": 0.034, "2": 0.052, ...}, "u10": {...}, ...},
    ...
  },
  "acc": { "...same structure as rmse_norm" },
  "n_samples_per_tau": 1463,
  "checkpoint": "...",
  "config": "..."
}
```

## How to run

Once the script `tools/eval/region_per_channel_eval.py` is written:

```bash
# On fibo, in container exw:cu128
docker exec -it exw bash -lc "
  cd /home/d.sukhorukov/weather_time_interpolation
  for ckpt_alias in fuxi modafno sdyff atm_vfi_v2 weatherdcae corrdiff_weatherdcae bilinear; do
    python tools/eval/region_per_channel_eval.py \
      --model-alias $ckpt_alias \
      --year 2020 \
      --output metrics/region_per_channel_2020/$ckpt_alias.json
  done
"
```

Or in parallel on 3 GPUs (1, 2, 3):

```bash
# Group A on GPU 1
python tools/eval/region_per_channel_eval.py --model-alias bilinear,fuxi --gpu 1 &
# Group B on GPU 2
python tools/eval/region_per_channel_eval.py --model-alias modafno,sdyff --gpu 2 &
# Group C on GPU 3
python tools/eval/region_per_channel_eval.py --model-alias atm_vfi_v2,weatherdcae,corrdiff_weatherdcae --gpu 3 &
wait
```

## Paper integration

Add as **Appendix A9** (`paper/A9_geo_region_eval.tex`):

- Tables: one per region, columns = 7 models, rows = 4 channels × 5 τ (or flatten). Or one heatmap fig per model showing 8 regions × 4 channels at τ=3.
- Section text: 2-3 paragraphs interpreting:
  - Which regions favour Skip vs NoSkip?
  - How does ATM-VFI's perceptual loss bias play out on convective tropics (Asia) vs stable mid-lats (Europe)?
  - Do diffusion (CorrDiff/WeatherDCAE) gains concentrate in any specific region?

## Estimated cost

- Per model: 1463 windows × 5 τ × 4 channels × 8 regions ≈ 234k float operations, ~5 min/model on B300 with batch=64.
- 7 models sequential: ~35 min. Realistic with checkpoint-load overhead: **~2 hours total**.

This is a small task compared to a retrain — can be run alongside the cloud.ru WeatherDCAE 6yr 12h DDP without interference.
