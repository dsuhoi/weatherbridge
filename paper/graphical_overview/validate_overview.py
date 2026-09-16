#!/usr/bin/env python3
"""Validate the rendered WeatherBridge graphical overview."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HTML = HERE / "overview.html"
PNG = ROOT / "paper" / "images" / "fig_graphical_overview.png"
PDF = ROOT / "paper" / "images" / "fig_graphical_overview.pdf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    dump = subprocess.run(
        [
            "chromium",
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--virtual-time-budget=3000",
            "--dump-dom",
            HTML.as_uri(),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if 'data-render-ready="true"' not in dump:
        raise RuntimeError("browser did not complete the graphical overview")
    if 'data-layout-overflow-count="0"' not in dump:
        raise RuntimeError("browser layout audit detected an overflow")
    if 'data-region-overlap-count="0"' not in dump:
        raise RuntimeError("browser layout audit detected a panel overlap")
    if 'data-small-text-count="0"' not in dump:
        raise RuntimeError("browser legibility audit detected undersized text")
    if 'data-content-overflow-count="0"' not in dump:
        raise RuntimeError("browser layout audit detected overflowing card content")
    if 'data-geometry-mismatch-count="0"' not in dump:
        raise RuntimeError("browser geometry audit detected misaligned plots")

    with Image.open(PNG) as image:
        if image.size != (3600, 2210):
            raise ValueError(f"unexpected PNG size: {image.size}")
        rgb = image.convert("RGB")
        extrema = rgb.getextrema()
        if any(high - low < 120 for low, high in extrema):
            raise ValueError(f"render has insufficient colour range: {extrema}")
        if max(ImageStat.Stat(rgb).var) < 400.0:
            raise ValueError("render appears blank or nearly uniform")

    required = (
        "Six-hour weather interpolation",
        "independent query-conditioned predictions",
        "Training",
        "difference",
        "query t",
        "static fields",
        "TRAINABLE NEURAL INTERPOLATOR",
        "supervision only",
        "back-propagation updates",
        "WeatherBridge",
    )
    normalised_text = " ".join(dump.split()).lower()
    missing = [item for item in required if item.lower() not in normalised_text]
    if missing:
        raise ValueError(f"rendered DOM is missing expected text: {missing}")

    fonts = subprocess.run(
        ["pdffonts", str(PDF)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[2:]
    if any(line.strip() for line in fonts):
        raise ValueError("flattened overview PDF unexpectedly contains fonts")

    images = subprocess.run(
        ["pdfimages", "-list", str(PDF)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[2:]
    rows = [line.split() for line in images if line.strip()]
    if len(rows) != 1 or rows[0][5].lower() != "rgb":
        raise ValueError("overview PDF must contain one RGB raster image")

    report = {
        "schema_version": 21,
        "html": str(HTML.relative_to(ROOT)),
        "png": {
            "path": str(PNG.relative_to(ROOT)),
            "pixel_size": [3600, 2210],
            "sha256": sha256_file(PNG),
        },
        "pdf": {
            "path": str(PDF.relative_to(ROOT)),
            "sha256": sha256_file(PDF),
        },
        "layout_overflow_count": 0,
        "region_overlap_count": 0,
        "small_text_count": 0,
        "content_overflow_count": 0,
        "geometry_mismatch_count": 0,
        "required_text_checked": list(required),
    }
    (HERE / "render_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("graphical overview validation passed")


if __name__ == "__main__":
    main()
