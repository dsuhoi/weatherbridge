#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

uv run --with numpy python scripts/export_weatherbridge_spectral_latex_data.py
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -cd paper/figs/fig_weatherbridge_spectra_tau23.tex
cp paper/figs/fig_weatherbridge_spectra_tau23.pdf \
  paper/images/fig_weatherbridge_spectra_tau23.pdf
