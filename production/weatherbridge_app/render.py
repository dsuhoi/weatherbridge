"""Dependency-light rendering of global forecast fields to WebP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

PALETTES = {
    "thermal": (
        (0.0, "#173F7A"),
        (0.2, "#3B82C4"),
        (0.45, "#D7EDF2"),
        (0.6, "#F4E7A1"),
        (0.8, "#E68345"),
        (1.0, "#9E2A2B"),
    ),
    "wind": (
        (0.0, "#251A5B"),
        (0.25, "#2457A6"),
        (0.5, "#37B6A1"),
        (0.75, "#E2D95B"),
        (1.0, "#C53D35"),
    ),
    "pressure": (
        (0.0, "#32205F"),
        (0.25, "#315FA5"),
        (0.5, "#75C4A5"),
        (0.75, "#E3D875"),
        (1.0, "#B83C38"),
    ),
    "humidity": (
        (0.0, "#F2EFE7"),
        (0.25, "#9FD3C7"),
        (0.5, "#3FAE91"),
        (0.75, "#21706E"),
        (1.0, "#173B57"),
    ),
    "geopotential": (
        (0.0, "#253064"),
        (0.25, "#3977A8"),
        (0.5, "#62B19C"),
        (0.75, "#D6C766"),
        (1.0, "#9C3F37"),
    ),
}


def palette_for(field: str) -> str:
    if field.startswith("T") or field == "t2m":
        return "thermal"
    if field.startswith(("U", "V")) or field in {"u10", "v10"}:
        return "wind"
    if field.startswith("Q"):
        return "humidity"
    if field.startswith("Z"):
        return "geopotential"
    return "pressure"


def _rgb(hex_color: str) -> np.ndarray:
    value = hex_color.lstrip("#")
    return np.asarray(
        [int(value[index : index + 2], 16) for index in (0, 2, 4)], dtype=np.float32
    )


def colorize(
    values: np.ndarray,
    vmin: float,
    vmax: float,
    palette: str,
    land_mask: np.ndarray | None = None,
) -> Image.Image:
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("render input must be a finite two-dimensional field")
    scale = max(vmax - vmin, 1.0e-12)
    normalized = np.clip((values.astype(np.float32) - vmin) / scale, 0.0, 1.0)
    stops = PALETTES[palette]
    positions = np.asarray([position for position, _ in stops], dtype=np.float32)
    colors = np.stack([_rgb(color) for _, color in stops])
    flat = normalized.ravel()
    channels = [np.interp(flat, positions, colors[:, channel]) for channel in range(3)]
    rgb = np.stack(channels, axis=-1).reshape((*values.shape, 3)).astype(np.uint8)
    if land_mask is not None:
        mask = land_mask > 0.5
        edge = np.zeros_like(mask)
        edge[1:] |= mask[1:] != mask[:-1]
        edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        rgb[edge] = np.asarray([20, 24, 31], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((1440, 720), Image.Resampling.BILINEAR)


def save_webp(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="WEBP", quality=90, method=6)
