#!/usr/bin/env python3
"""Plot the six-hour sparse-query versus full-query training control."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from paper_plot_style import (
    COLUMN_WIDTH_IN,
    MODEL_COLORS,
    MODEL_MARKERS,
    use_paper_rc,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = (
    ROOT / "metrics" / "full_vs_heldout_6h_s202707" / "summary_2020_2021"
)
DEFAULT_OUTPUT = ROOT / "paper" / "images"
ARCHITECTURES = (
    "WeatherBridge",
    "WeatherDCAE-14M",
    "PixelAttn-VFI",
    "SwinV2",
    "S-DYff",
)
YEARS = (2020, 2021)
HOURS = (1, 2, 3, 4, 5)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_summary(rows: list[dict[str, str]]) -> None:
    keys = {
        (row["architecture"], int(row["year"]), row["hours"]) for row in rows
    }
    expected = {
        (architecture, year, split)
        for architecture in ARCHITECTURES
        for year in YEARS
        for split in ("trained", "held_out", "all")
    }
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"unexpected summary rows; missing={missing}, extra={extra}")


def summary_figure(rows: list[dict[str, str]], output: Path) -> None:
    use_paper_rc(7.2)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(COLUMN_WIDTH_IN, 4.8),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.28, "wspace": 0.18},
    )
    y_positions = np.arange(len(ARCHITECTURES))[::-1]
    split_specs = (
        ("trained", "trained hours 1, 3 and 5"),
        ("held_out", "previously held-out hours 2 and 4"),
    )

    for row_index, year in enumerate(YEARS):
        for column_index, (split, title) in enumerate(split_specs):
            ax = axes[row_index, column_index]
            for y, architecture in zip(y_positions, ARCHITECTURES, strict=True):
                item = next(
                    row
                    for row in rows
                    if row["architecture"] == architecture
                    and int(row["year"]) == year
                    and row["hours"] == split
                )
                sparse = float(item["heldout_nrmse"])
                full = float(item["full_nrmse"])
                gain = float(item["full_gain_pct"])
                color = MODEL_COLORS[architecture]
                ax.annotate(
                    "",
                    xy=(full, y),
                    xytext=(sparse, y),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "lw": 1.0,
                        "mutation_scale": 7,
                        "shrinkA": 4,
                        "shrinkB": 5,
                    },
                    zorder=1,
                )
                ax.scatter(
                    sparse,
                    y,
                    s=27,
                    marker="o",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.1,
                    zorder=3,
                )
                ax.scatter(
                    full,
                    y,
                    s=31,
                    marker=MODEL_MARKERS[architecture],
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=4,
                )
                ax.text(
                    max(sparse, full) + 0.0022,
                    y,
                    f"{gain:+.1f}%",
                    va="center",
                    ha="left",
                    fontsize=7.2,
                    color="0.25",
                )

            panel = chr(ord("a") + row_index * 2 + column_index)
            ax.set_title(f"{panel})  {year}\n{title}", loc="left", fontweight="bold")
            ax.set_yticks(y_positions, ARCHITECTURES)
            ax.set_xlim(0.047, 0.131)
            ax.set_ylim(-0.6, len(ARCHITECTURES) - 0.4)
            ax.grid(axis="x", color="0.84", linewidth=0.45)
            ax.set_axisbelow(True)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.tick_params(axis="y", length=0)
            if row_index == 1:
                ax.set_xlabel("Mean normalised RMSE")

    legend_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="0.25",
            label="Sparse-query checkpoint",
        ),
        plt.Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markerfacecolor="0.25",
            markeredgecolor="white",
            label="Full-query checkpoint",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.20, right=0.99)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)


def field_hour_figure(rows: list[dict[str, str]], output: Path) -> None:
    use_paper_rc(7.2)
    fields = [
        row["field"]
        for row in rows
        if row["architecture"] == "WeatherBridge"
        and int(row["year"]) == 2020
        and int(row["hour"]) == 1
    ]
    if len(fields) != 24 or len(set(fields)) != 24:
        raise ValueError(f"expected 24 ordered fields, got {fields}")

    gains = np.asarray([float(row["full_gain_pct"]) for row in rows])
    limit = max(5.0, 5.0 * np.ceil(np.max(np.abs(gains)) / 5.0))
    fig, axes = plt.subplots(
        1,
        len(ARCHITECTURES),
        figsize=(9.35, 5.25),
        sharey=True,
    )
    image = None
    for column, (ax, architecture) in enumerate(zip(axes, ARCHITECTURES, strict=True)):
        matrix = np.empty((len(fields), len(YEARS) * len(HOURS)), dtype=float)
        for field_index, field in enumerate(fields):
            for year_index, year in enumerate(YEARS):
                for hour_index, hour in enumerate(HOURS):
                    item = next(
                        row
                        for row in rows
                        if row["architecture"] == architecture
                        and int(row["year"]) == year
                        and row["field"] == field
                        and int(row["hour"]) == hour
                    )
                    matrix[field_index, year_index * len(HOURS) + hour_index] = float(
                        item["full_gain_pct"]
                    )
        image = ax.pcolormesh(
            np.arange(matrix.shape[1] + 1),
            np.arange(matrix.shape[0] + 1),
            matrix,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="flat",
            edgecolors=(1.0, 1.0, 1.0, 0.65),
            linewidth=0.22,
        )
        ax.set_xlim(0, matrix.shape[1])
        ax.set_ylim(matrix.shape[0], 0)
        evidence = "matched control" if architecture == "WeatherBridge" else "diagnostic"
        ax.set_title(f"{architecture}\n[{evidence}]", fontsize=7.4, fontweight="bold")
        ax.set_xticks(
            np.arange(10) + 0.5,
            ["1", "2*", "3", "4*", "5", "1", "2*", "3", "4*", "5"],
            fontsize=6.5,
        )
        ax.axvline(5.0, color="0.15", linewidth=0.8)
        for x_position, year in ((0.25, "2020"), (0.75, "2021")):
            ax.text(
                x_position,
                -0.085,
                year,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=6.7,
                fontweight="bold",
                clip_on=False,
            )
        ax.set_yticks(np.arange(len(fields)) + 0.5)
        if column == 0:
            ax.set_yticklabels(fields, fontsize=6.5)
        ax.tick_params(axis="both", length=2.0, pad=1.5)

    fig.text(0.495, 0.025, "Query hour (* absent from sparse-query training)", ha="center")
    assert image is not None
    colorbar_axis = fig.add_axes((0.94, 0.12, 0.014, 0.72))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Full-query RMSE gain (%)", fontsize=6.5)
    fig.subplots_adjust(left=0.07, right=0.91, top=0.86, bottom=0.145, wspace=0.09)
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.02)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_rows(args.metrics_dir / "full_vs_heldout_summary.csv")
    field_hour = read_rows(args.metrics_dir / "full_vs_heldout_field_hour.csv")
    validate_summary(summary)
    if len(field_hour) != len(ARCHITECTURES) * len(YEARS) * 24 * len(HOURS):
        raise ValueError(f"expected 1200 field-hour rows, got {len(field_hour)}")

    summary_figure(summary, args.output_dir / "fig_full_vs_heldout_summary.pdf")
    field_hour_figure(
        field_hour,
        args.output_dir / "fig_full_vs_heldout_field_hour.pdf",
    )
    print("wrote full-query supervision figures")


if __name__ == "__main__":
    main()
