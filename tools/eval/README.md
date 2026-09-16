# Evaluation

`batch_eval_12h_memmap.py` supports six- and twelve-hour intervals. It
computes latitude-weighted RMSE and ACC by field and query hour.
`eval_sdyff_ensemble.py` extends it with repeated stochastic predictions.

In the training environment, inspect the arguments with:

```bash
python -m tools.eval.batch_eval_12h_memmap --help
python -m tools.eval.eval_sdyff_ensemble --help
```

For a full-year run, pass `--full-year`; `--eval-days-per-month` selects
a subset of dates. Supply the recorded memmap, climatology, normalization,
static fields and compatible training checkpoint. The default filesystem
paths refer to the original experiments and may need overrides.

Six-hour evaluation uses query hours 1 through 5, with trained hours
1, 3 and 5. Twelve-hour evaluation uses hours 1 through 11, with hours
4, 6 and 8 withheld from training. Evaluators rescale the model time input
by the anchor interval.

`export_*_tex.py` scripts build tables from stored results.
`summarize_*.py` scripts perform paired comparisons and check cohort
alignment. Do not combine artifacts with different dates, field orders,
normalization or checkpoint identities.

See [the reproduction manifest](../../repro/paper_experiments.json) for
the commands associated with the current manuscript.
