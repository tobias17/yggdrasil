"""Generate a (hard-coded) island structure and render it.

Infra scaffold for the island generator: no real generation yet. This
builds a fixed stone/dirt/grass island, saves it in the project model
format (3D int array + atlas), and renders a single screenshot for
inspection.

Run from the repo root:

    .venv/bin/python islands/generate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so `utils` resolves no matter where we run from.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from utils import Atlas, Structure, render_screenshot  # noqa: E402

# --- Hard-coded island parameters (replace with real generation later) ----
DIAMETER = 18    # flat top diameter, in blocks
HEIGHT = 12      # approximate total height, in blocks
BULGE = 3        # extra radius the bottom bulges out by, in blocks
DIRT_LAYERS = 3  # layers of dirt beneath the grass cap
MARGIN = 4       # empty space around the island in the saved array


def build_island(atlas: Atlas) -> Structure:
    """Build the hard-coded demo island: flat top, bulging bottom."""
    r_top = DIAMETER // 2
    footprint = DIAMETER + 2 * BULGE + 2 * MARGIN

    # Structure axes are (X, Y, Z), Y up.
    shape = (footprint, HEIGHT + 2 * MARGIN, footprint)
    structure = Structure(shape, atlas=atlas)

    cx, cz = shape[0] // 2, shape[2] // 2
    y_bottom = MARGIN
    y_top = y_bottom + HEIGHT - 1
    y_grass = y_top
    y_dirt = y_top - (DIRT_LAYERS - 1)

    # Squared distance from the island center for every (X, Z) column.
    xs, zs = np.meshgrid(np.arange(shape[0]), np.arange(shape[2]), indexing="ij")
    dist2 = (xs - cx) ** 2 + (zs - cz) ** 2

    for y in range(y_bottom, y_top + 1):
        if y >= y_dirt:
            # Flat top section (grass cap + dirt): constant radius = top diameter.
            r = r_top
        else:
            # Stone: bulges out toward the bottom. The ramp reaches 0 at the
            # top stone layer (directly under the dirt) so no stone sticks out
            # past the cap, and grows monotonically downward (no overhangs).
            t = (y_dirt - 1 - y) / (y_dirt - 1 - y_bottom)
            r = r_top + BULGE * t

        mask_xz = dist2 <= r * r

        if y == y_grass:
            block = "grass"  # dirt with no other block on top
        elif y >= y_dirt:
            block = "dirt"
        else:
            block = "stone"
        structure.set_layer(block, y, mask_xz)

    return structure


def main() -> None:
    atlas = Atlas()
    for name in ("stone", "dirt", "grass"):
        atlas.add(name)

    structure = build_island(atlas)

    out_dir = Path(__file__).resolve().parent / "out"
    model_path = structure.save(out_dir / "island.npz")
    shot_path = render_screenshot(
        structure, out_dir / "island.png", title="Yggdrasil - hard-coded island"
    )

    print(f"model     : {model_path}")
    print(f"screenshot: {shot_path}")
    print(f"shape     : {structure.shape}  (X, Y, Z)")
    print(f"atlas     : {structure.atlas.to_dict()}")
    print(f"blocks    : {structure.stats()}")
    print(f"bounds    : {structure.bounds()}")


if __name__ == "__main__":
    main()
