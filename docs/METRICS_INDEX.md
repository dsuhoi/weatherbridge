# Paper metrics

| Directory under `metrics/` | Use |
|---|---|
| `journal_unified/` | RMSE and ACC by model, field, hour and year |
| `journal_spectra_v6/` | Scalar and vector spherical-harmonic diagnostics |
| `detailed_benchmark_v2/` | Physical, diurnal and paired comparisons |
| `full_vs_heldout_6h_s202707/` | Matched-update full-query controls |
| `weatherbridge_component_ablation_v1/` | Inference-time component removals |
| `sdyff_ensemble_n21/` | Paired N=1 and ensemble evaluations |
| `case_studies_weatherbridge/` | Regional WeatherBridge examples |
| `case_studies_weatherdcae_14m/` | Matching WeatherDCAE examples |
| `case_studies_pixelattn_vfi/` | Matching PixelAttn-VFI examples |
| `nwp_blend_6h_2022_v4_endpoint_guard/` | Forecast-anchor coefficient adaptation |
| `nwp_blend_spectra_6h_2022_v2_endpoint_guard/` | Its spectral comparisons |
| `postselection_2022_v1/` | Sampled 2022 assessment |

Artifact names retain the original experiment identifiers. Plotting scripts
assign the paper's display names. Before comparing two files, check the
checkpoint, query hours, channel order, normalization and sampled dates in
their metadata.

The six-hour and twelve-hour cohorts in 2020 and 2021 are full-year
evaluations. The 2022 extension uses a fixed subset of dates. Aggregate
summaries are sufficient to redraw curves; paired statistical tests also
require the underlying per-window arrays.

Absolute paths in historical metadata record where results were produced.
They do not identify downloadable data or valid paths on a new machine.
