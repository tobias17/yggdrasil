"""Headless 3D visualization for structures.

Uses matplotlib with the Agg backend so screenshots can be rendered
without a display (WSL, CI, servers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless; must be set before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402

# Default block colors as RGB in 0..1. Unknown blocks get the fallback color.
DEFAULT_PALETTE: dict[str, tuple[float, float, float]] = {
    "stone": (0.58, 0.58, 0.62),
    "dirt": (0.55, 0.37, 0.22),
    "grass": (0.32, 0.62, 0.28),
}
FALLBACK_COLOR = (0.85, 0.25, 0.60)


def build_color_table(
    atlas,
    palette: Mapping[str, tuple[float, float, float]] | None = None,
) -> np.ndarray:
    """Build an (n_index, 3) float RGB lookup table for every atlas index."""
    palette = {**DEFAULT_PALETTE, **(palette or {})}
    index_to_name = {i: name for name, i in atlas.to_dict().items()}
    max_index = max(index_to_name, default=0)
    table = np.full((max_index + 1, 3), FALLBACK_COLOR, dtype=np.float32)
    for index, name in index_to_name.items():
        table[index] = palette.get(name, FALLBACK_COLOR)
    return table


def render_screenshot(
    structure,
    out_path,
    title: str | None = None,
    palette: Mapping[str, tuple[float, float, float]] | None = None,
    elev: float = 20.0,
    azim: float = -60.0,
    dpi: int = 150,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> Path:
    """Render a structure as voxels and save a single PNG screenshot.

    The structure's Y axis (up) is mapped to the plot's vertical axis so the
    model stands upright in the image. Returns the path of the PNG.
    """
    data = structure.data
    table = build_color_table(structure.atlas, palette)

    # Our structure is (X, Y, Z) with Y up; matplotlib voxels render array
    # axis 2 as vertical. Transpose to (X, Z, Y) so up stays up.
    filled = np.transpose(data != 0, (0, 2, 1))
    colors = np.transpose(table[data.astype(np.intp)], (0, 2, 1, 3))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection="3d")
    ax.voxels(filled, facecolors=colors, edgecolor="black", linewidth=0.15)

    # Proportional axes (display axis sizes: X right, Z depth, Y up).
    shape = np.array(filled.shape, dtype=float)
    ax.set_box_aspect(shape / shape.max())

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (up)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=elev, azim=azim)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
