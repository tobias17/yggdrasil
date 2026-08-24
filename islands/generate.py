"""
Floating Island Generator for Minecraft
=========================================

Procedurally generates floating rock islands (irregular grassy top,
tapering rocky underside, hanging "root" stalactites, moss/vine
decoration) in the style of concept-art floating islands.

Outputs:
  1. A .npz model file in the project's canonical format: a 3D numpy
     array of int16 block indices (X, Y, Z with Y up, 0 = air) plus a
     JSON atlas legend naming each index - the same format as
     islands_old/generate.py.
  2. A 3D preview image rendered as full shaded blocks so you can check
     the shape BEFORE building anything in-game.

No external world/game connection is needed to preview - this is pure
geometry + a plotting library.

Usage:
    python floating_islands.py --diameter 40           # single island, flat 40-block-wide top
    python floating_islands.py --diameter 40 --seed 7  # different random variation, same size
    python floating_islands.py --scene                 # old multi-island demo composition

By default the island has a perfectly FLAT top (single Y level) so it's
easy to build on, but the outline is irregular (not a perfect circle) and
the underside tapers down into rock with hanging root/stalactite drips.
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import Atlas, Structure, render_screenshot  # noqa: E402


# ---------------------------------------------------------------------------
# Noise helpers
# ---------------------------------------------------------------------------

def value_noise_2d(res, grid_size, seed):
    """Smooth 2D value noise on an res x res grid, roughly in [-1, 1].
    Pure numpy (no external noise library needed)."""
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(-1, 1, (grid_size + 1, grid_size + 1))

    coords = np.linspace(0, grid_size - 1e-9, res)
    xi = np.floor(coords).astype(int)
    xf = coords - xi

    def smoothstep(t):
        return t * t * (3 - 2 * t)

    X, Y = np.meshgrid(xi, xi, indexing="ij")
    XF, YF = np.meshgrid(xf, xf, indexing="ij")
    sx, sy = smoothstep(XF), smoothstep(YF)

    top = coarse[X, Y] * (1 - sx) + coarse[X + 1, Y] * sx
    bot = coarse[X, Y + 1] * (1 - sx) + coarse[X + 1, Y + 1] * sx
    return top * (1 - sy) + bot * sy


# ---------------------------------------------------------------------------
# Island generation
# ---------------------------------------------------------------------------

STONE_VARIANTS = [
    ("minecraft:stone", 0.85),
    ("minecraft:andesite", 0.15),
]


def pick_stone(rng):
    roll = rng.random()
    acc = 0
    for block, p in STONE_VARIANTS:
        acc += p
        if roll <= acc:
            return block
    return "minecraft:stone"


def generate_island(seed=0, diameter=40, top_thickness=3, max_depth=14,
                     num_drips=10, flat_top=True, decorate_top=False,
                     decorate_underside=True, offset=(0, 0, 0)):
    """Returns dict {(x, y, z): "minecraft:block_id"} for one island,
    positioned with its center at `offset`.

    diameter        - island top diameter in blocks (radius = diameter / 2)
    flat_top        - if True (default), the top surface is a single flat
                       Y level, so you get a clean buildable circle. The
                       OUTLINE is still irregular (not a perfect circle) so
                       it reads as natural rock rather than a disc.
    decorate_top     - if True, scatters grass/flowers/trees on top. Off by
                       default so the surface stays clear to build on.
    decorate_underside - hanging root drips + vines + moss on the rock
                       underside. On by default for the floating-island look;
                       doesn't affect the top surface at all.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    radius = diameter / 2.0

    size = int(radius * 2 + 6)
    half = size // 2

    # Irregular radial silhouette (a few sine harmonics -> lumpy circle,
    # not a perfect disc)
    thetas = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    R = np.full(360, float(radius))
    for k in range(2, 6):
        amp = radius * np_rng.uniform(0.015, 0.045)
        phase = np_rng.uniform(0, 2 * np.pi)
        R += amp * np.sin(k * thetas + phase)
    R = np.clip(R, radius * 0.55, None)

    edge_noise = value_noise_2d(size, 8, seed + 1) * (radius * 0.05)
    hill_noise = value_noise_2d(size, 6, seed + 2)
    # per-column jitter on the taper radius so the underside isn't a
    # perfectly smooth cone
    taper_noise = value_noise_2d(size, 10, seed + 3) * (radius * 0.12)

    # how much the body's radius shrinks from just-under-the-dirt (t=0) down
    # to max_depth (t=1). >0 means it starts narrowing immediately rather
    # than staying full-width for the first several layers.
    TAPER_STRENGTH = 0.55

    blocks = {}
    col_bottom = {}
    columns = []  # (x, z, topY, depth, r, localR) - depth is the carved depth
    for xi in range(size):
        for zi in range(size):
            x, z = xi - half, zi - half
            r = math.hypot(x, z)
            theta = math.atan2(z, x) % (2 * math.pi)
            idx = int(theta / (2 * math.pi) * 360) % 360
            localR = R[idx] + edge_noise[xi, zi]
            if r > localR:
                continue
            topY = 0 if flat_top else int(round(hill_noise[xi, zi] * 2.2))

            blocks[(x, topY, z)] = "minecraft:grass_block"
            for dy in range(1, top_thickness):
                blocks[(x, topY - dy, z)] = "minecraft:dirt"

            # Carve the rock body: at each step down, the allowed radius
            # shrinks (taper), so columns near the current edge drop out
            # first while the center keeps going - giving continuous
            # narrowing instead of a flat plateau then a sudden cliff.
            body_top = topY - top_thickness
            jitter = taper_noise[xi, zi]
            bottomY = body_top
            for y_offset in range(0, max_depth + 1):
                t = y_offset / max_depth
                allowed_r = localR * (1 - TAPER_STRENGTH * t) + jitter
                if y_offset > 0 and r > max(allowed_r, 0):
                    break
                y = body_top - y_offset
                blocks[(x, y, z)] = pick_stone(rng)
                bottomY = y

            depth = body_top - bottomY
            col_bottom[(x, z)] = (bottomY, topY, r, localR)
            columns.append((x, z, topY, depth, r, localR))

            # occasional moss cap on the very bottom face near the edge
            if r / localR > 0.65 and rng.random() < 0.35:
                blocks[(x, bottomY, z)] = "minecraft:mossy_cobblestone"

    if decorate_underside:
        # thin hanging root/stalactite drips, mostly toward the edges
        edge_cols = [c for c in columns if c[4] / c[5] > 0.35]
        rng.shuffle(edge_cols)
        for (x, z, topY, depth, r, localR) in edge_cols[:num_drips]:
            bottomY, _, _, _ = col_bottom[(x, z)]
            drip_len = rng.randint(5, 16)
            drip_r = rng.choice([0, 0, 1])
            for dl in range(drip_len):
                y = bottomY - dl
                th = max(0, drip_r - dl // 5)
                for dx in range(-th, th + 1):
                    for dz in range(-th, th + 1):
                        if dx * dx + dz * dz <= th * th + 1:
                            block = "minecraft:mossy_cobblestone" if dl > drip_len - 3 else "minecraft:stone"
                            blocks[(x + dx, y, z + dz)] = block
            if rng.random() < 0.6:
                blocks[(x, bottomY - drip_len, z)] = "minecraft:vine"

        # vines draped down from the underside near the outer rim
        for (x, z, topY, depth, r, localR) in columns:
            if r / localR > 0.55 and rng.random() < 0.18:
                bottomY, _, _, _ = col_bottom[(x, z)]
                for vy in range(rng.randint(2, 6)):
                    blocks.setdefault((x, bottomY - vy, z), "minecraft:vine")

    if decorate_top:
        # sparse grass/flower decoration on the top surface
        for (x, z, topY, depth, r, localR) in columns:
            if r / localR < 0.9 and rng.random() < 0.12:
                block = rng.choice(
                    ["minecraft:short_grass", "minecraft:short_grass",
                     "minecraft:fern", "minecraft:poppy", "minecraft:dandelion"]
                )
                blocks.setdefault((x, topY + 1, z), block)

        # a couple of small trees
        tree_spots = [c for c in columns if c[4] / c[5] < 0.55]
        rng.shuffle(tree_spots)
        for (x, z, topY, depth, r, localR) in tree_spots[: rng.randint(0, 2)]:
            trunk_h = rng.randint(3, 5)
            for dy in range(trunk_h):
                blocks[(x, topY + 1 + dy, z)] = "minecraft:oak_log"
            leaf_y = topY + 1 + trunk_h
            for dx in range(-2, 3):
                for dz in range(-2, 3):
                    for dy in range(0, 3):
                        if dx * dx + dz * dz + dy * dy <= 5 and rng.random() < 0.85:
                            blocks.setdefault((x + dx, leaf_y + dy, z + dz), "minecraft:oak_leaves")

    # apply world offset
    ox, oy, oz = offset
    return {(x + ox, y + oy, z + oz): b for (x, y, z), b in blocks.items()}


def generate_scene(seed=0):
    """Builds one big island plus several smaller satellite islands and
    floating debris chunks, echoing the reference composition."""
    rng = random.Random(seed)
    blocks = {}

    # main island
    blocks.update(generate_island(seed=seed, diameter=40, max_depth=16,
                                   num_drips=14, offset=(0, 90, 0)))

    # a couple of medium islands
    satellite_specs = [
        dict(diameter=20, max_depth=9, num_drips=6, offset=(-42, 110, -10)),
        dict(diameter=16, max_depth=8, num_drips=5, offset=(30, 118, -28)),
    ]
    for i, spec in enumerate(satellite_specs):
        blocks.update(generate_island(seed=seed + 10 + i, **spec))

    # tiny floating debris (just a few blocks each), like the specks in the sky
    for i in range(5):
        cx = rng.randint(-55, 55)
        cy = rng.randint(95, 130)
        cz = rng.randint(-40, 20)
        n = rng.randint(1, 4)
        for _ in range(n):
            dx, dy, dz = rng.randint(-1, 1), rng.randint(-1, 1), rng.randint(-1, 1)
            blocks[(cx + dx, cy + dy, cz + dz)] = "minecraft:stone"

    return blocks


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def blocks_to_structure(blocks):
    """Converts a {(x, y, z): "minecraft:block_id"} dict to the project's
    canonical Structure: a 3D numpy array of int16 block indices
    (X, Y, Z with Y up, 0 = air) plus an Atlas naming each index.
    Coordinates are shifted so the bounding box starts at the origin."""
    items = list(blocks.items())
    atlas = Atlas()
    indices = {name: atlas.add(name) for name in sorted({b for _, b in items})}
    xs = np.array([k[0] for k, _ in items])
    ys = np.array([k[1] for k, _ in items])
    zs = np.array([k[2] for k, _ in items])
    origin = (int(xs.min()), int(ys.min()), int(zs.min()))
    shape = (int(xs.max()) - origin[0] + 1,
             int(ys.max()) - origin[1] + 1,
             int(zs.max()) - origin[2] + 1)
    data = np.zeros(shape, dtype=np.int16)
    data[xs - origin[0], ys - origin[1], zs - origin[2]] = (
        np.array([indices[b] for _, b in items], dtype=np.int16)
    )
    return Structure.from_data(data, atlas)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

BLOCK_COLORS = {
    "minecraft:grass_block": "#5b8a3a",
    "minecraft:dirt": "#6b4a2b",
    "minecraft:stone": "#8a8a8a",
    "minecraft:andesite": "#a3a3a0",
    "minecraft:mossy_cobblestone": "#5e6b4a",
    "minecraft:vine": "#3f6b2a",
    "minecraft:oak_log": "#5a3d1f",
    "minecraft:oak_leaves": "#3f7a2f",
    "minecraft:short_grass": "#6fae3f",
    "minecraft:fern": "#4f8f3f",
    "minecraft:poppy": "#c0392b",
    "minecraft:dandelion": "#e8c93a",
}


def preview(structure, out_path="preview.png", title=None):
    """Renders the island as full shaded blocks (one 3D screenshot in the
    same style as islands_old's screenshots) and saves it to out_path."""
    palette = {
        name: (
            int(color.lstrip("#")[0:2], 16) / 255.0,
            int(color.lstrip("#")[2:4], 16) / 255.0,
            int(color.lstrip("#")[4:6], 16) / 255.0,
        )
        for name, color in BLOCK_COLORS.items()
    }
    return render_screenshot(structure, out_path, title=title, palette=palette)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate a floating island for Minecraft.")
    ap.add_argument("--seed", type=int, default=1, help="random seed")
    ap.add_argument("--diameter", type=int, default=32,
                     help="top diameter of the island in blocks (default: 32)")
    ap.add_argument("--max-depth", type=int, default=None,
                     help="how far the rock tapers down below the top (default: scales with diameter)")
    ap.add_argument("--num-drips", type=int, default=None,
                     help="number of hanging root/stalactite drips (default: scales with diameter)")
    ap.add_argument("--decorate-top", action="store_true",
                     help="scatter grass/flowers/trees on top (off by default so it stays buildable)")
    ap.add_argument("--no-underside-decor", action="store_true",
                     help="disable drips/vines/moss on the underside")
    ap.add_argument("--out", type=str, default="island", help="output file prefix")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "out",
                     dest="out_dir",
                     help="directory for outputs (default: islands/out)")
    ap.add_argument("--scene", action="store_true",
                     help="generate the old multi-island demo scene instead of a single island")
    args = ap.parse_args()

    if args.scene:
        blocks = generate_scene(seed=args.seed)
    else:
        max_depth = args.max_depth if args.max_depth is not None else max(6, args.diameter // 2)
        num_drips = args.num_drips if args.num_drips is not None else max(4, args.diameter // 3)
        blocks = generate_island(
            seed=args.seed,
            diameter=args.diameter,
            max_depth=max_depth,
            num_drips=num_drips,
            decorate_top=args.decorate_top,
            decorate_underside=not args.no_underside_decor,
        )

    structure = blocks_to_structure(blocks)
    npz_path = structure.save(args.out_dir / f"{args.out}.npz")
    print(f"Wrote {len(blocks)} blocks to {npz_path}")

    title = (f"multi-island demo scene (seed={args.seed})" if args.scene
             else f"floating island (d={args.diameter}, seed={args.seed})")
    png_path = preview(structure, out_path=args.out_dir / f"{args.out}.png", title=title)
    print(f"Saved preview image to {png_path}")


if __name__ == "__main__":
    main()
