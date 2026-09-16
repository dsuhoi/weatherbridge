# Repository map

| Directory | Contents |
|---|---|
| `weather_time_interp/` | Models, datasets, normalization and metrics |
| `tools/train/` | Matched trainers and training protocol helpers |
| `tools/eval/` | RMSE, ACC, spectral and statistical evaluation |
| `scripts/` | Plotting and recorded experiment launchers |
| `repro/` | Experiment manifests and reproduction commands |
| `metrics/` | Stored results used by figures and tables |
| `paper/` | Manuscript and figure sources |
| `tests/` | Unit, protocol and manuscript checks |
| `weatherbridge-release/` | Standalone checkpoint loader and examples |
| `legacy/` | Implementations required by older checkpoints |

WeatherBridge is `WeatherBridgeModel` in
`weather_time_interp/model/weatherbridge_flow_model.py`.
Historical artifacts use `flow_pp3` for this model. WeatherDCAE-14M is
a separate autoencoder, implemented by `WeatherDCAEAdaLNModel`.

Artifact filenames retain their original names and hashes.
`scripts/paper_plot_style.py` maps them to display names.
