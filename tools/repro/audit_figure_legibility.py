#!/usr/bin/env python3
"""Measure the effective point size of text in every figure the paper includes.

Springer Nature requires figure text to stay legible at final size, with 5 pt as
the hard floor. A figure drawn much wider than the manuscript column is placed
at a scale well below one, and its nominal point sizes shrink by that factor, so
the size chosen in the plotting script is not the size a reader sees. This
measures what actually reaches the PDF:

    effective pt = (smallest point size in the figure) x (placement scale)

Mathtext exponents and subscripts use dedicated Computer Modern script fonts.
They are conventional typography rather than labels, so the floor is measured
on non-script runs of three characters or more.

Usage:
    python tools/repro/audit_figure_legibility.py [--paper-dir paper] [--strict]

With --strict the script exits non-zero if any included figure falls below the
floor, which is how the publication gate consumes it.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Text block of the sn-jnl referee layout, measured from the class itself:
#   \the\columnwidth == \the\textwidth == 372.0pt
COLUMN_WIDTH_PT = 372.0
TEXT_HEIGHT_PT = 552.7

FLOOR_PT = 5.0
COMFORTABLE_PT = 6.0

INCLUDE = re.compile(r"\\includegraphics(?:\[(?P<opt>[^\]]*)\])?\{(?P<path>[^}]+)\}")


def included_figures(paper_dir: Path) -> dict[Path, tuple[str, bool]]:
    """Map each images/ figure to its options and page orientation."""
    uses: dict[Path, tuple[str, bool]] = {}
    for name in ("main.tex", "supplementary.tex"):
        source = (paper_dir / name).read_text()
        for match in INCLUDE.finditer(source):
            path = match.group("path")
            if not path.startswith("images/"):
                continue
            before = source[: match.start()]
            landscape_start = max(
                before.rfind(r"\begin{landscape}"),
                before.rfind(r"\begin{supplementlandscape}"),
            )
            landscape_end = max(
                before.rfind(r"\end{landscape}"),
                before.rfind(r"\end{supplementlandscape}"),
            )
            landscape = landscape_start > landscape_end
            uses.setdefault(
                paper_dir / path,
                (match.group("opt") or "", landscape),
            )
    return uses


def natural_size(pdf: Path) -> tuple[float, float]:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    match = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", out)
    if not match:
        raise RuntimeError(f"cannot read the page size of {pdf}")
    return float(match.group(1)), float(match.group(2))


def smallest_text_pt(pdf: Path) -> float | None:
    # -zoom 1 makes pdftohtml report sizes in points rather than its default 1.5x
    result = subprocess.run(
        ["pdftohtml", "-xml", "-stdout", "-i", "-q", "-zoom", "1", str(pdf)],
        capture_output=True,
        text=True,
    )
    xml = result.stdout
    specs = {}
    for match in re.finditer(
        r'<fontspec id="(\d+)"[^>]*size="([\d.]+)"[^>]*family="([^"]+)"',
        xml,
    ):
        specs[match.group(1)] = (float(match.group(2)), match.group(3))
    sizes = []
    for match in re.finditer(r'<text[^>]*font="(\d+)"[^>]*>(.*?)</text>', xml):
        text = re.sub("<[^>]+>", "", match.group(2)).strip()
        if len(text) < 3:
            continue
        spec = specs.get(match.group(1))
        if spec:
            size, family = spec
            if re.search(r"(?:CMR|CMMI|CMSY|CMEX)5$", family):
                continue
            sizes.append(size)
    return min(sizes) if sizes else None


def placement_scale(
    options: str,
    width: float,
    height: float,
    *,
    landscape: bool = False,
) -> float | None:
    scale = None
    width_basis = TEXT_HEIGHT_PT if landscape else COLUMN_WIDTH_PT
    height_basis = COLUMN_WIDTH_PT if landscape else TEXT_HEIGHT_PT
    match = re.search(r"width=([\d.]*)\\(?:columnwidth|textwidth|linewidth)", options)
    if match:
        fraction = float(match.group(1)) if match.group(1) else 1.0
        scale = fraction * width_basis / width
    limit = re.search(r"height=([\d.]+)\\textheight", options)
    if limit and "keepaspectratio" in options:
        capped = float(limit.group(1)) * height_basis / height
        scale = capped if scale is None else min(scale, capped)
    return scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if not (args.paper_dir / "main.tex").is_file():
        print(f"no main.tex under {args.paper_dir}", file=sys.stderr)
        return 2

    print(f"{'figure':<42}{'natW':>8}{'scale':>7}{'minpt':>7}{'effpt':>7}  flag")
    failures = 0
    for pdf, (options, landscape) in sorted(included_figures(args.paper_dir).items()):
        if not pdf.is_file():
            print(f"{pdf.name:<42}{'-':>8}{'-':>7}{'-':>7}{'-':>7}  MISSING")
            failures += 1
            continue
        width, height = natural_size(pdf)
        scale = placement_scale(options, width, height, landscape=landscape)
        smallest = smallest_text_pt(pdf)
        effective = smallest * scale if (smallest and scale) else None
        flag = ""
        if effective is not None and effective < FLOOR_PT:
            flag = f"FAIL <{FLOOR_PT:.0f}pt"
            failures += 1
        elif effective is not None and effective < COMFORTABLE_PT:
            flag = "tight"
        print(
            f"{pdf.name:<42}"
            f"{width:8.0f}"
            f"{(f'{scale:.3f}' if scale else '-'):>7}"
            f"{(f'{smallest:.1f}' if smallest else '-'):>7}"
            f"{(f'{effective:.2f}' if effective else '-'):>7}  {flag}"
        )

    if failures:
        print(
            f"\n{failures} figure(s) fall below the {FLOOR_PT:.0f} pt legibility floor "
            "at their placed size.",
            file=sys.stderr,
        )
    return 1 if (failures and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
