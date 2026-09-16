# Checkpoints and inference

This package loads WeatherBridge and the five learned six-hour comparators.
The release archive includes weights, normalization, static fields and one
ERA5 test window.

## Download and install

Download `weatherbridge-models-v1.0.0.tar.gz` and `SHA256SUMS` from
[release v1.0.0](https://github.com/dsuhoi/weatherbridge/releases/tag/v1.0.0).
Verify the checksum before extracting. From the extracted directory:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
```

Use the packaged `torch-harmonics==0.7.3`. The S-DYff spectral weights need
its handling of `lmax=16, mmax=32`; changing the dependency can change the
operator shape. Other dependency upgrades have not been validated here.

```python
import weatherbridge as wb

model = wb.load_model("weatherbridge-6h", device="cuda")
x_hat = wb.predict(model, x0, xT, tau_hours=3)
```

## Checkpoints

| Name | Parameters | Anchor interval |
|---|---:|---:|
| `weatherbridge-6h` | 14.26 M | 6 h |
| `weatherbridge-12h` | 14.26 M | 12 h |
| `weatherdcae-14m-6h` | 14.37 M | 6 h |
| `fuxi-6h` (SwinV2) | 7.84 M | 6 h |
| `modafno-6h` | 36.93 M | 6 h |
| `sdyff-6h` | 101.31 M | 6 h |
| `pixelattn-vfi-6h` | 11.70 M | 6 h |

`weights/catalogue.json` records the architecture and source checkpoint.
The archive does not include every twelve-hour comparator, full-query
control or HRES-adapted checkpoint.

## Inputs

Anchors are normalized tensors shaped `(B, 24, 360, 720)`:

```text
T1000 T925 T850 T700
U1000 U925 U850 U700
V1000 V925 V850 V700
Q1000 Q925 Q850 Q700
Z1000 Z925 Z850 Z700
t2m u10 v10 mslp
```

Use the pressure-level and surface statistics in `data/`.
`wb.predict` attaches static fields and converts query hours to model time.
S-DYff is stochastic; set the random seed when comparing runs.

## Tests

```bash
pytest
```

The tests check file hashes, load the checkpoints and compare predictions on
the supplied 2020-07-01 00--06 UTC window with `data/reference_scores.json`.
This one-window test does not reproduce the paper's full-year evaluation.
The twelve-hour checkpoint is excluded from the six-hour sample comparison.

## License

Original inference code and weights are MIT-licensed. Vendored code and
ERA5-derived data retain their own terms; see `THIRD_PARTY_NOTICES.md`.
`MANIFEST.sha256` in the download records the archived files.
