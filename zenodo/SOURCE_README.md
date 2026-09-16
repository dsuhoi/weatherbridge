# WeatherBridge

Code for *WeatherBridge and the limits of global weather interpolation at
unseen query times*. WeatherBridge reconstructs hourly weather fields between
two supplied anchor states. The experiments compare six learned architectures
and linear interpolation on ERA5, and evaluate adaptation to IFS HRES forecasts.

The six-hour task withholds hours 2 and 4 during training. The twelve-hour
experiment withholds hours 4, 6 and 8. Full-query controls compare the same
architectures at matched update counts.

## Start here

- [Reproduce figures and tables](repro/README.md)
- [Models and inference](weatherbridge-release/README.md)
- [Training presets](repro/scripts/train_paper_matched.sh)
- [Model implementations](weather_time_interp/model/README.md)
- [Data and channel order](data/README.md)
- [Release checks](zenodo/REPRODUCIBILITY.md)

## CPU checks and plots

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install numpy matplotlib scipy pandas xarray netcdf4 pyyaml pytest pymupdf
python -m pytest -q tests/repro tests/test_sdyff_ensemble.py \
  tests/test_align_window_artifact.py tests/test_make_fig_specific_channels_phys.py \
  tests/test_export_component_lesions_tex.py tests/test_export_selector_diagnostics_tex.py
python -m scripts.make_fig_main_scores
python -m scripts.make_fig_specific_channels_phys --mode all
```

These commands redraw figures from stored scores. They do not run the models.
Tensor-level tests are skipped when PyTorch is absent. Training and full-year
evaluation need CUDA, ERA5/HRES data and the dependencies recorded in
`requirements-publication.txt`. That file records the original environment;
it is not a portable installation list.

## Training

```bash
bash repro/scripts/train_paper_matched.sh --dry-run weatherbridge 6
bash repro/scripts/train_paper_matched.sh weatherbridge 6 \
  --memmap_dir /data/era5 --static_path data/static_features_0p5.pt
```

The same script accepts `weatherdcae` or `pixelattn_vfi`, and intervals
`6` or `12`. It sets the architecture, loss, query hours and update budget.
SwinV2, ModAFNO and S-DYff use the retained trainers under `legacy/scripts/`;
their configurations and checkpoint identities are recorded in the manuscript
training tables and metric manifests.

The generic Hydra recipes in `conf/` include earlier experiments. Their
PixelAttn-VFI and WeatherDCAE recipes differ from the paper's matched presets.

## Data and weights

[Release v1.0.0](https://github.com/dsuhoi/weatherbridge/releases/tag/v1.0.0)
provides the standalone model archive. It contains seven checkpoints:
WeatherBridge at 6 h and 12 h, and five 6 h comparators, with normalization,
static fields and one ERA5 test window. Other 12 h comparators, full-query
controls and adapted HRES checkpoints are not included in that archive.

The repository contains figure inputs, metric summaries and available
per-window results. Obtain ERA5 and IFS HRES archives separately from
WeatherBench 2 or the Copernicus Climate Data Store; see
[data/README.md](data/README.md).
A full rerun requires the matching checkpoints and data.

## Citation and license

[CITATION.cff](CITATION.cff) records the software authors and version.
Use the release tag or commit hash when citing code.

Training and evaluation code is Apache-2.0. The standalone model package has
its own MIT license. Vendored components and ERA5-derived data retain their
original terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
