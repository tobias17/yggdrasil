"""
Floating Island Generator for Minecraft
=========================================

Procedurally generates floating rock islands (irregular grassy top,
tapering rocky underside, hanging "root" stalactites, moss/vine
decoration) in the style of concept-art floating islands.

Outputs:
  1. A plain block-list (CSV + JSON): universal (x, y, z, block_id) format
     you can feed into any importer/mod/plugin you write yourself.
  2. A .schem file (Sponge schematic) you can drop straight into a
     Minecraft world with WorldEdit: //schem load <name>  then  //paste
  3. A 3D voxel preview image (and interactive matplotlib window) so you
     can check the shape BEFORE generating/pasting anything in-game.

No external world/game connection is needed to preview - this is pure
geometry + a plotting library.

Usage:
    python floating_islands.py --diameter 40           # single island, flat 40-block-wide top
    python floating_islands.py --diameter 40 --seed 7  # different random variation, same size
    python floating_islands.py --diameter 40 --show    # also pop up an interactive 3D window
    python floating_islands.py --scene                 # old multi-island demo composition

By default the island has a perfectly FLAT top (single Y level) so it's
easy to build on, but the outline is irregular (not a perfect circle) and
the underside tapers down into rock with hanging root/stalactite drips.
"""

import argparse
import json
import csv
import math
import random
import numpy as np


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
    ("minecraft:stone", 0.80),
    ("minecraft:andesite", 0.08),
    ("minecraft:deepslate", 0.06),
    ("minecraft:mossy_cobblestone", 0.06),
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
        amp = radius * np_rng.uniform(0.03, 0.09)
        phase = np_rng.uniform(0, 2 * np.pi)
        R += amp * np.sin(k * thetas + phase)
    R = np.clip(R, radius * 0.55, None)

    edge_noise = value_noise_2d(size, 8, seed + 1) * (radius * 0.10)
    hill_noise = value_noise_2d(size, 6, seed + 2)
    depth_noise = value_noise_2d(size, 10, seed + 3)

    columns = []  # (x, z, topY, depth, r, localR)
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
            falloff = max(0.0, 1 - (r / localR) ** 2)
            depth = int(max_depth * (falloff ** 0.6) + max(0, depth_noise[xi, zi]) * 4)
            depth = max(depth, 2)
            columns.append((x, z, topY, depth, r, localR))

    blocks = {}
    col_bottom = {}
    for (x, z, topY, depth, r, localR) in columns:
        blocks[(x, topY, z)] = "minecraft:grass_block"
        for dy in range(1, top_thickness):
            blocks[(x, topY - dy, z)] = "minecraft:dirt"
        bottomY = topY - top_thickness - depth
        col_bottom[(x, z)] = (bottomY, topY, r, localR)
        for y in range(bottomY, topY - top_thickness):
            blocks[(x, y, z)] = pick_stone(rng)
        # occasional moss cap on the very bottom face near the edge
        if r / localR > 0.65 and rng.random() < 0.35:
            blocks[(x, bottomY, z)] = "minecraft:moss_block"

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
                blocks[(x, bottomY - drip_len - 1, z)] = "minecraft:vine"

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

def export_block_list(blocks, csv_path, json_path):
    rows = [{"x": x, "y": y, "z": z, "block": b} for (x, y, z), b in blocks.items()]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "z", "block"])
        w.writeheader()
        w.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f)
    return len(rows)


def export_schem(blocks, path):
    """Writes a Sponge .schem WorldEdit can load directly:
    //schem load <name>   then   //paste
    """
    import mcschematic
    schem = mcschematic.MCSchematic()
    for (x, y, z), block in blocks.items():
        schem.setBlock((x, y, z), block)
    schem.save("", path, mcschematic.Version.JE_1_20_1)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

BLOCK_COLORS = {
    "minecraft:grass_block": "#5b8a3a",
    "minecraft:dirt": "#6b4a2b",
    "minecraft:stone": "#8a8a8a",
    "minecraft:andesite": "#a3a3a0",
    "minecraft:deepslate": "#3f3f45",
    "minecraft:mossy_cobblestone": "#5e6b4a",
    "minecraft:moss_block": "#4a7a2a",
    "minecraft:vine": "#3f6b2a",
    "minecraft:oak_log": "#5a3d1f",
    "minecraft:oak_leaves": "#3f7a2f",
    "minecraft:short_grass": "#6fae3f",
    "minecraft:fern": "#4f8f3f",
    "minecraft:poppy": "#c0392b",
    "minecraft:dandelion": "#e8c93a",
}


def preview(blocks, out_path="preview.png", show=False, max_blocks=45000):
    import matplotlib.pyplot as plt

    items = list(blocks.items())
    if len(items) > max_blocks:
        # thin out for a manageable preview render (keeps the shape, drops density)
        step = math.ceil(len(items) / max_blocks)
        items = items[::step]

    xs = np.array([p[0][0] for p in items])
    ys = np.array([p[0][1] for p in items])
    zs = np.array([p[0][2] for p in items])
    colors = [BLOCK_COLORS.get(p[1], "#999999") for p in items]

    fig = plt.figure(figsize=(15, 7))

    # --- 3D angled view ---
    ax = fig.add_subplot(121, projection="3d")
    # Minecraft Y is "up" -> plot (x, z, y) so the vertical axis reads naturally
    ax.scatter(xs, zs, ys, c=colors, marker="s", s=6, depthshade=True, linewidths=0)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (height)")
    ax.set_title("3D view")
    ax.view_init(elev=18, azim=-60)
    max_range = max(xs.max() - xs.min(), zs.max() - zs.min(), ys.max() - ys.min()) / 2.0
    mid_x, mid_z, mid_y = (xs.max() + xs.min()) / 2, (zs.max() + zs.min()) / 2, (ys.max() + ys.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_z - max_range, mid_z + max_range)
    ax.set_zlim(mid_y - max_range, mid_y + max_range)

    # --- top-down footprint view (the buildable surface) ---
    top_y = ys.max()
    mask = ys >= top_y - 0.5  # just the topmost layer (grass surface)
    ax2 = fig.add_subplot(122)
    ax2.scatter(xs[mask], zs[mask], c=[c for c, m in zip(colors, mask) if m],
                marker="s", s=14, linewidths=0)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_title(f"Top-down footprint (buildable surface, Y={int(top_y)})")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


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
    ap.add_argument("--show", action="store_true", help="open an interactive 3D window")
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

    n_csv = export_block_list(blocks, f"{args.out}.csv", f"{args.out}.json")
    print(f"Wrote {n_csv} blocks to {args.out}.csv / {args.out}.json")

    try:
        export_schem(blocks, f"{args.out}")
        print(f"Wrote {args.out}.schem (WorldEdit: //schem load {args.out}  then  //paste)")
    except Exception as e:
        print(f"Skipped .schem export ({e}). CSV/JSON output is still complete.")

    preview(blocks, out_path=f"{args.out}_preview.png", show=args.show)
    print(f"Saved preview image to {args.out}_preview.png")


if __name__ == "__main__":
    main()
