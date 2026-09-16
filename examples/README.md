# Inference examples

Use the loader in [weatherbridge-release](../weatherbridge-release/README.md)
for the packaged checkpoints. Its `predict` function accepts query time in
hours and attaches static fields.

The scripts here expect bare checkpoints under `weights/`:

```python
from examples._bare_loader import load_bare
model = load_bare("weights/weatherbridge_14m_6h_bare.pt", device="cuda")
```

Copy the checkpoint from the model archive or change the path in the example.
Some examples use historical ablation checkpoints that are not in the release.

WeatherBridge and WeatherDCAE accept normalized `(B, 24, 360, 720)` tensors,
query time `t = tau_hours / interval_hours`, the interval in hours, and three
static fields. See [channel order](../data/README.md).

S-DYff samples noise during inference. The paper reports one draw (S-DYff)
and the mean of 21 draws (S-DYff-ENS). Ensemble evaluation is implemented in
`tools/eval/eval_sdyff_ensemble.py`.
