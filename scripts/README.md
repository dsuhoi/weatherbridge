# Figure and experiment scripts

Paper entry points are listed in
[repro/paper_experiments.json](../repro/paper_experiments.json).

```bash
python -m scripts.make_fig_main_scores
python -m scripts.make_fig_specific_channels_phys --mode all
python -m tools.repro.run --list
```

Plotting scripts read stored metrics and write to `paper/images/`.
Training presets are in `repro/scripts/train_paper_matched.sh`.

Scripts ending in `_cloudru.sh` record the original cluster launches.
Adapt their paths, environments and GPU locks before using them elsewhere.
The underlying Python entry points are in `tools/train/` and `tools/eval/`.
