# Data

The models use a 360 x 720 grid and 24 normalized fields, in this order:

```text
T1000 T925 T850 T700
U1000 U925 U850 U700
V1000 V925 V850 V700
Q1000 Q925 Q850 Q700
Z1000 Z925 Z850 Z700
t2m u10 v10 mslp
```

`json_stats_0p5.nc` and `surface_stats_0p5.json` contain normalization
statistics. `normalization_provenance.json` records their source.
`static_features_0p5.pt` contains sea-land mask, orography and latitude features.

Obtain hourly ERA5 and forecasts from
[WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)
or the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).

The paper uses block-averaged 0.5-degree WeatherBench 2 fields. Preserve the
recorded grid orientation, channel order and normalization when preparing
memmaps with `tools/data/`. Training scripts expect `wb2_YEAR.bin` and
`wb2_YEAR.json` pairs. The loader selects 24 model fields from the stored channels.

CSV files here are historical UI exports. Use `metrics/journal_unified/`
for the manuscript's current RMSE/ACC summaries. The 2020 and 2021 comparisons
cover full years; the 2022 extension uses the dates in
`repro/postselection_holdout_2022.json`.
