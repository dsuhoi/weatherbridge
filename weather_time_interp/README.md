# Core package

`memmap_dataset.py` reads ERA5 memmaps. `normalization.py` loads channel
statistics and checks their provenance. `eval_runner.py` provides shared
evaluation setup and accumulation. Model implementations are in
[model/](model/README.md).

The model input has 24 fields on a 360 x 720 grid, plus three static fields.
See [data/README.md](../data/README.md) for the order.

The legacy dataset represents query time in six-hour units. The twelve-hour
training and evaluation wrappers convert physical query hours to
`t = tau_hours / interval_hours` before calling the model. Retain that
conversion when using the dataset outside the supplied runners.

Training and evaluation have separate hour lists. The six-hour held-out
protocol trains on 1, 3 and 5 h; the twelve-hour protocol excludes 4, 6 and 8 h.
