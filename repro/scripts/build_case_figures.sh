#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
: "${WEATHERBRIDGE_CHECKPOINT:?set WEATHERBRIDGE_CHECKPOINT to the WeatherBridge checkpoint}"
PYTHONPATH=.:scripts uv run --with matplotlib --with numpy \
  python scripts/make_fig_case_ida_wind.py
PYTHONPATH=.:scripts uv run --with matplotlib --with numpy \
  python scripts/make_fig5_haishen_local.py
uv run python tools/eval/export_case_metrics_tex.py \
  --root . --weatherbridge-checkpoint "$WEATHERBRIDGE_CHECKPOINT" \
  --out-tex paper/case_metrics_v2.tex
