# Experiments Layout (archived)

**Why archived.** This was the project-local run registry used by the
pre-Hydra workflow (`exp_register_run.py`, `exp_eval_matrix.py`). Run
tracking has moved to native Hydra `outputs/` + `multirun/` plus the
canonical `metrics/` JSON tree, so this registry is no longer updated.
Kept for historical lookup of older runs and to seed the deterministic
baseline cache (`baselines/cache/`).

This directory standardizes experiment tracking, cached deterministic baselines,
and comparable metrics across runs.

## Structure

- `configs/`:
  - Optional canonical config files (`*.yaml`, `*.json`) for reusable experiments.
- `registry/`:
  - `runs.jsonl`: append-only run registry.
  - `latest_by_experiment.json`: helper index with latest run per experiment.
- `runs/<run_id>/`:
  - `manifest.json`: resolved metadata and pointers to artifacts.
  - `metrics_long.csv`: long-form metrics table for plotting/comparisons.
  - `comparison_vs_bilinear.csv`: per-hour/per-variable model vs bilinear deltas.
- `baselines/cache/<fingerprint>/`:
  - Cached deterministic baseline metrics (bilinear/bicubic) for a fixed dataset
    fingerprint. Reused across model runs.
- `reports/`:
  - Generated comparison summaries (`*.md`/`*.csv`).

## Fingerprint

Baseline cache must be keyed by a deterministic dataset fingerprint that includes:

- years/split
- `samples_per_date`
- `stats_path`
- variables and pressure levels
- eval hours

Use `scripts/exp_register_run.py` and `scripts/exp_eval_matrix.py` to keep this
consistent.
