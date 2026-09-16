# Release checks

Checked on 16 September 2026 with Python 3.14 in a CPU-only environment.

## Completed

- 93 tests passed; two tensor-level S-DYff tests were skipped because PyTorch
  was not installed. The test command is in the root README.
- The comparison-figure manifest ran successfully from the public snapshot,
  including RMSE, ACC, physical-field, spectral and HRES plots.
- Main text and Supplementary Information compiled with latexmk. The build
  checked unresolved citations, text overflow and embedded fonts.
- The code-availability section and its repository/version links were checked
  in the rendered PDF.
- Staged Python files were parsed for syntax. Local README links and common
  credential patterns were checked before publication.
- The model archive's file manifest and checksums were verified. The seven
  weight files are unchanged; documentation was corrected.

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
python -m tools.repro.run comparison_figures
bash paper/build_npj_submission.sh
SOURCE_ONLY=1 bash zenodo/build_release.sh
python zenodo/validate_release.py --source-only
```

Source archives record the commit and dirty-tree state in
`RELEASE_PROVENANCE.txt`; `SOURCE_MANIFEST.sha256` covers their contents.
Use the tagged release for a fixed code version.
