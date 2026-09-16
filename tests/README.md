# Tests

The CPU command in the root README checks experiment manifests, paper/model
identities, ensemble summaries and table/figure generation.

Model and dataset tests require PyTorch and the relevant architecture
dependencies. In the training environment, run:

```bash
python -m pytest tests/
```

`conftest.py` builds synthetic memmaps when PyTorch is available.
CUDA tests are skipped without a GPU. Real-data tests require 360 x 720
memmaps and an explicit `python -m pytest -m prod_grid` command.

The standalone archive has separate checkpoint and single-window inference
tests in `weatherbridge-release/tests/`. The CPU protocol suite does not
run those tests.
