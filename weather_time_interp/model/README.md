# Models

| Model | Source |
|---|---|
| WeatherBridge | `weatherbridge_flow_model.py` |
| WeatherDCAE-14M | `dcae_adaln_model.py`, blocks in `dcae.py` |
| SwinV2 | `fuxi_swinv2_model.py` |
| PixelAttn-VFI | `../../legacy/scripts/train_atm_vfi_12h_oddskip.py` |
| ModAFNO | `modafno_baseline_model.py`, `physicsnemo_vendor/` |
| S-DYff | `sdyff_dyffusion_model.py` |

WeatherBridge combines anchor warping and blending with fine-grid,
coarse-grid and spherical-spectral corrections. Older checkpoints use the
`flow_pp3` name. The matched trainer selects the paper configuration;
other options in the class belong to separate experiments.

WeatherDCAE-14M uses time-conditioned residual blocks and an EfficientViT
bottleneck, without encoder-to-decoder skips. SwinV2 has four pairs of windowed
and shifted-window blocks. PixelAttn-VFI encodes the endpoints separately
with shared weights before cross-frame attention.

S-DYff and ModAFNO are adapted to this study's fields, grid and interpolation
task. Their results do not reproduce the original forecasting benchmarks.

Other retained files provide shared blocks, checkpoint compatibility and the
paper's ablations. The matched trainer exposes the paper architectures and
transport controls; separate exploratory model families are not distributed.
