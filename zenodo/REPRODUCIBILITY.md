# Release checks

Checked on 18 September 2026 with Python 3.14, PyTorch 2.14.0+cpu and
torch-harmonics 0.8.0. These are verification dependencies, not a replacement
for the recorded training environment.

Code and seven checkpoint files are distributed through Git and Git LFS.
GitHub Release attachment upload requires separate API permissions; the LFS
files can be downloaded without a Release attachment.

## Completed

- The full `tests/` suite passed: 670 passed, 11 skipped. The skips require
  CUDA or the external production grid/data. Tensor-level S-DYff tests ran.
- All 17 retained matched-trainer configurations produced identical state
  dictionaries before and after cleanup when initialized with the same seed.
- Removed 292 files belonging to separate exploratory models, search queues,
  the web application and obsolete exports. Published checkpoints and frozen
  source snapshots are unchanged.
- Restored `metrics.weather`, needed by legacy baseline loaders, and the
  missing extreme-event score file used by the manuscript figure.
- The comparison-figure manifest ran successfully from the public snapshot,
  including RMSE, ACC, physical-field, spectral and HRES plots.
- Main text and Supplementary Information compiled with latexmk. The build
  checked unresolved citations, text overflow and embedded fonts.
- The code-availability section and its repository/version links were checked
  in the rendered PDF.
- Staged Python files were parsed for syntax. Local README links and common
  credential patterns were checked before publication.
- The seven weight files and their LFS identifiers are unchanged from the
  verified `weights-v1.0.0` snapshot. Source version 1.0.1 uses model-package
  version 1.0.0.

The reproduction runner now uses its current Python interpreter for Python
commands and adds the repository to child-process import paths. A regression
test covers that behavior.

## Not rerun

Training, full-year inference and the standalone model inference tests were
not rerun for this publication. Earlier inference checks are not evidence
that a new dependency environment will produce identical results.

ERA5/HRES archives and full training checkpoints remain external inputs.
The model download contains WeatherBridge at both intervals and five
six-hour comparators; it omits the other twelve-hour, full-query and
HRES-adapted checkpoints. Historical cluster launchers require path changes
before use outside their original environment.

## Commands

```bash
python -m pytest -q tests/repro tests/test_sdyff_ensemble.py \
  tests/test_align_window_artifact.py tests/test_make_fig_specific_channels_phys.py \
  tests/test_export_component_lesions_tex.py tests/test_export_selector_diagnostics_tex.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q tests
python -m tools.repro.run comparison_figures
bash paper/build_npj_submission.sh
SOURCE_ONLY=1 bash zenodo/build_release.sh
python zenodo/validate_release.py --source-only
```

Source archives record the commit and dirty-tree state in
`RELEASE_PROVENANCE.txt`; `SOURCE_MANIFEST.sha256` covers their contents.
Use the tagged release for a fixed code version.
