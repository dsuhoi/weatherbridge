#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/paper/graphical_overview"
PNG="$ROOT/paper/images/fig_graphical_overview.png"
PDF="$ROOT/paper/images/fig_graphical_overview.pdf"
URI="file://$HERE/overview.html"

chromium \
  --headless \
  --disable-gpu \
  --no-sandbox \
  --hide-scrollbars \
  --force-device-scale-factor=1 \
  --window-size=3600,2210 \
  --screenshot="$PNG" \
  "$URI"

python - "$PNG" "$PDF" <<'PY'
from pathlib import Path
import sys

from PIL import Image

png = Path(sys.argv[1])
pdf = Path(sys.argv[2])
with Image.open(png) as image:
    image.convert("RGB").save(pdf, "PDF", resolution=300.0, quality=95)
PY

echo "rendered $PNG"
echo "rendered $PDF"
