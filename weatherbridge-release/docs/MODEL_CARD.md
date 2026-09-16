# WeatherBridge model card

WeatherBridge reconstructs weather fields between two known atmospheric
states. It requires the later anchor and cannot forecast beyond it.

## Model and data

The 14.26 M-parameter model predicts bidirectional displacement, blends warped
anchors with linear interpolation and adds local and spherical-spectral
corrections. A temperature-conditioned head adjusts geopotential and pressure.
It is a learned correction, not an exact hydrostatic constraint.

The paper checkpoint uses circular convolution padding on both tensor axes.
Its geometry differs from a pole-aware spherical convolution. Spherical
harmonic transforms are used in the spectral branch.

The six-hour model was trained on ERA5 from 2014--2019, using query hours
1, 3 and 5. The twelve-hour model used 2017--2019 and withheld query hours
4, 6 and 8. Both use 24 fields on a 360 x 720 grid.

## Scope and limitations

The intended use is temporal interpolation of archived weather fields.
The model has not been validated for safety-critical warnings, aviation
routing, other spatial resolutions or precipitation, which is not an output.

ERA5 interpolation can reproduce assimilation increments as well as physical
evolution. Performance varies by field and query hour. The model is
deterministic and does not provide calibrated uncertainty.

IFS HRES adaptation is evaluated retrospectively and combines interpolation
error with forecast-to-ERA5 mismatch. The archive here contains ERA5-trained
weights, not the HRES-adapted checkpoints.

## Evaluation and provenance

The paper reports full-year 2020/2021 evaluations and a sampled 2022 extension.
The supplied ERA5 window is only a loader and numerical regression test.

`weights/catalogue.json` records the source checkpoints.
The model archive's `MANIFEST.sha256` identifies the distributed files.
