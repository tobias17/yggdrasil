"""Headless 3D visualization for structures.

Uses matplotlib with the Agg backend so screenshots can be rendered
without a display (WSL, CI, servers).

Visible faces are extracted with vectorized numpy and drawn as a single
:class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection` per panel, instead of
using ``ax.voxels`` — which loops in Python and builds one collection per
voxel — so large structures render in well under a second.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless; must be set before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

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


# Corner tables for the six possible face directions. Each table gives the
# four corners of the unit face as (x, y, z) with the face's own axis
# coordinate at 0 (the grid offset is added when emitting). Windings are
# chosen so matplotlib's shading normal, (v0-v1) x (v1-v2), points AWAY from
# the block (outward) on every face.
_FACE_TABLES = {
    (-1, 0): [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]],  # -x
    (+1, 0): [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],  # +x
    (-1, 1): [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # -y
    (+1, 1): [[0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]],  # +y
    (-1, 2): [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]],  # -z
    (+1, 2): [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],  # +z
}


def structure_faces(
    structure,
    palette: Mapping[str, tuple[float, float, float]] | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract every visible face of a structure, vectorized.

    A face is emitted wherever a solid voxel touches air, including the
    array boundary (treated as air). Interior faces are never built, so the
    work scales with the surface rather than the volume.

    Returns ``(faces, colors)``: ``faces`` is a list of (4, 3) vertex arrays
    in *display* coordinates (X right, Z depth, Y up) ready for
    ``Poly3DCollection``, and ``colors`` an (n_faces, 3) float RGB array
    holding the block color of the voxel each face belongs to.
    """
    data = structure.data
    table = build_color_table(structure.atlas, palette)
    cell_color = table[data.astype(np.intp)]  # (X, Y, Z, 3)

    # Out-of-bounds neighbours count as air. (dir, axis): dir is -1/1 along
    # axis; the face belongs to the voxel on the solid side of the pair.
    p = np.pad(data != 0, 1)
    planes = [
        (~p[:-2, 1:-1, 1:-1], -1, 0, 0),  # -x
        (~p[2:, 1:-1, 1:-1], +1, 0, 1),   # +x
        (~p[1:-1, :-2, 1:-1], -1, 1, 0),  # -y
        (~p[1:-1, 2:, 1:-1], +1, 1, 1),   # +y
        (~p[1:-1, 1:-1, :-2], -1, 2, 0),  # -z
        (~p[1:-1, 1:-1, 2:], +1, 2, 1),   # +z
    ]

    faces: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for air_plane, direction, axis, offset in planes:
        mask = (data != 0) & air_plane
        coords = np.argwhere(mask).astype(np.float64)  # (N, 3)
        if coords.size == 0:
            continue
        table = np.array(_FACE_TABLES[(direction, axis)], dtype=np.float64)
        plane = coords[:, axis] + offset
        verts = coords[:, None, :] + table          # (N, 4, 3) in data coords
        verts[:, :, axis] = plane[:, None]
        # Data (X, Y, Z) -> display (X, Z, Y) so Y is vertical on screen.
        # The axis swap is a reflection, so the vertex order is reversed to
        # keep the shading normal pointing outward.
        verts = verts[:, :, [0, 2, 1]]
        verts = verts[:, ::-1]
        faces.append(verts)                         # one (N, 4, 3) block
        colors.append(cell_color[mask])

    if not faces:
        return [], np.empty((0, 3))
    # Poly3DCollection wants a list of polygons, each an (N, 3) array.
    all_faces = np.concatenate(faces, axis=0)
    return [face for face in all_faces.reshape(-1, 4, 3)], np.vstack(colors)


def _draw_structure(
    ax,
    structure,
    palette: Mapping[str, tuple[float, float, float]] | None = None,
) -> None:
    """Add the structure's faces to a 3D axes as one shaded collection."""
    faces, colors = structure_faces(structure, palette)
    if not faces:
        return
    ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors=colors,
            edgecolors="black",
            linewidths=0.15,
            shade=True,
        )
    )


def _setup_axes(ax, structure) -> None:
    """Proportional, upright view for a structure: X right, Z depth, Y up."""
    sx, sy, sz = structure.shape
    ax.set_box_aspect((sx, sz, sy))
    ax.set_xlim(0, sx)
    ax.set_ylim(0, sz)
    ax.set_zlim(0, sy)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (up)")
    ax.grid(True, alpha=0.25)


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
    """Render a structure and save a single PNG screenshot.

    The structure's Y axis (up) is mapped to the plot's vertical axis so the
    model stands upright in the image. Returns the path of the PNG.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection="3d")
    _draw_structure(ax, structure, palette)
    _setup_axes(ax, structure)
    if title:
        ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_sheet(
    structures,
    out_path,
    labels=None,
    title: str | None = None,
    palette: Mapping[str, tuple[float, float, float]] | None = None,
    elev: float = 20.0,
    azim: float = -60.0,
    dpi: int = 120,
    panel: float = 6.0,
) -> Path:
    """Render several structures side by side, one 3D panel each.

    Each structure fills its own panel so panels can be compared for shape
    rather than absolute size. ``labels`` is an optional per-panel caption.
    Returns the PNG path.
    """
    structures = list(structures)
    n = len(structures)
    if n == 0:
        raise ValueError("need at least one structure")
    labels = list(labels or [""] * n)

    fig = plt.figure(figsize=(panel * n, panel), dpi=dpi)
    for i, (structure, label) in enumerate(zip(structures, labels)):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        _draw_structure(ax, structure, palette)
        _setup_axes(ax, structure)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(label or f"panel {i + 1}")

    if title:
        fig.suptitle(title)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
