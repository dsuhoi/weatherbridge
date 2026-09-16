# Reproducing the results

`paper_experiments.json` lists commands, inputs and outputs:

```bash
python -m tools.repro.run --list
python -m tools.repro.run comparison_figures --dry-run
python -m tools.repro.run comparison_figures
```

The runner records commands and file hashes under `artifacts/repro_runs/`.
The comparison target redraws RMSE, ACC and spectral plots from stored results.
It does not rerun inference.

Other local targets build spectral and case figures, downstream tables and
the manuscript. Case replay checks the hash of the original checkpoint;
set `WEATHERBRIDGE_CHECKPOINT` to that training checkpoint.
Bare inference weights have different file hashes and cannot replace it
in this provenance check.

Targets beginning with `cloud_` are records of the original cluster runs.
They need external data and checkpoints, and their launchers contain the
original filesystem paths. They are not portable CPU reproduction commands.

## Evaluation scope

`metrics/journal_unified/` contains the RMSE/ACC summaries for all 24 fields
and the 6 h/12 h cohorts in 2020, 2021 and 2022. The first two years are
full-year evaluations; the 2022 extension is sampled. S-DYff-ENS is the
21-member mean. Its paired N=1 baseline uses the first draw from that run;
it should not be substituted for an older independent single draw.

`metrics/full_vs_heldout_6h_s202707/` contains the matched-update comparison.
`metrics/weatherbridge_component_ablation_v1/` contains inference-time
component removals. These are different experimental designs.

## What a full rerun needs

Download ERA5/HRES, prepare the recorded grid and climatology, and supply the
checkpoint named by each evaluation manifest. The model release contains
WeatherBridge at both intervals and five 6 h comparators. It does not contain
every ablation, adapted or 12 h comparator checkpoint.

The plotting and CPU tests cannot establish full numerical reproducibility.
See [release checks](../zenodo/REPRODUCIBILITY.md) for the checks performed here.
