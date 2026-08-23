"""Generate a simple floating island: an inverted teardrop with a flat top.

Shape (noise-free base silhouette for now):
  * A water drop turned point-down: a small blunt point at the bottom that
    grows in an exponential/asymptotic taper into a rounded bulge in the
    upper part, then tapers slightly.
  * Both ends of the drop profile are sliced off: the top a little above
    the bulge (flat top that still bulges out at the rim), the bottom where
    the taper would otherwise drag out into a long 1-block tail.

Block layout:
  * Every solid voxel is created as dirt.
  * After that, any dirt block with no block directly above it becomes
    grass (the Minecraft surface rule). This gives a grass cap over the
    entire top surface and will keep working correctly when more surface
    detail (spikes, noise) is added later.

Output:
  * per size   ``islands/out/island_d<D>.npz`` — the model data
  * combined   ``islands/out/island.png`` — one side-by-side sheet with
               all sizes, re-rendered on every run so progress can be
               watched in a single image.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import Atlas, Structure, render_sheet  # noqa: E402


@dataclass(frozen=True)
class IslandParams:
    """Parameters for the simple teardrop island.

    diameter        max width of the island in blocks (width at the bulge)
    height          total height in blocks, from the tip to the flat top
    shape_exponent  tip character; higher = spikier (default 2.0)
    cut_frac        where the flat slice falls on the full drop, 0..1.
                    The full drop's bulge sits at ~0.5**(1/shape_exponent);
                    the slice must be above that for the rim to taper in
                    (default 0.80 -> rim ~90% of bulge width)
    tip_frac        where the drop is sliced off at the tip, in the same
                    units as cut_frac (position on the full drop, 0..1).
                    The visible profile runs from tip_frac (bottom) to
                    cut_frac (top), so the bottom ends on a small blunt
                    point instead of the drop's long 1-block tail
                    (default 0.12)
    margin          air padding around the model in blocks
    """

    diameter: int = 40
    height: int = 48
    shape_exponent: float = 2.0
    cut_frac: float = 0.80
    tip_frac: float = 0.12
    margin: int = 2


def island_radii(p: IslandParams) -> np.ndarray:
    """Radius in blocks of the island at each height row, tip (0) to top (-1).

    A single water-drop profile ``sin(pi * u**g)`` (u in 0..1, point at both
    ends, bulge at u = 0.5**(1/g)) sliced at both ends and stretched over the
    island height:

        u(t) = tip_frac + (cut_frac - tip_frac) * t
        r(t) = R * sin(pi * u(t) ** shape_exponent)

    The drop's taper is the "asymptotic" look: quadratic at the point,
    widening fast through the bulge, then slowly rounding off towards the
    flat top. Left uncut, the tip end of the profile stays under one block
    wide for many rows (the long 1-block tail spike), so it is sliced off at
    ``tip_frac``: the bottom row starts on a small blunt point (radius
    R*sin(pi*tip_frac**g)) and the visible curve is otherwise untouched.
    """
    R = 0.5 * p.diameter
    t = np.arange(p.height) / (p.height - 1)
    u = p.tip_frac + (p.cut_frac - p.tip_frac) * t
    return R * np.sin(np.pi * np.power(u, p.shape_exponent))


def generate_island(p: IslandParams) -> Structure:
    """Build the teardrop island as a Structure of dirt + grass."""
    atlas = Atlas()
    dirt = atlas.add("dirt")
    grass = atlas.add("grass")

    R = 0.5 * p.diameter
    # odd grid size so the center column is unambiguous
    size = int(np.ceil(2 * R)) + 2 * p.margin
    if size % 2 == 0:
        size += 1
    c = size // 2

    data = np.zeros((size, p.height, size), dtype=np.int16)

    # per-column distance from the center column, squared (X, Z)
    d = np.arange(size) - c
    dist2 = d[:, None] ** 2 + d[None, :] ** 2

    radii = island_radii(p)
    for y, r in enumerate(radii):
        mask = dist2 <= r * r
        data[:, y, :][mask] = dirt

    # Minecraft rule: dirt with no block directly above becomes grass.
    # has_above[y] is True where the block one row up is solid.
    # NOTE: must be a real bool array — an int mask would be interpreted as
    # integer indexing, not a boolean mask.
    has_above = np.zeros(data.shape, dtype=bool)
    has_above[:, : p.height - 1, :] = data[:, 1:, :] != 0
    data[(data == dirt) & ~has_above] = grass

    return Structure.from_data(data, atlas)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--diameter", type=int, nargs="+", default=[16, 40, 80],
                    metavar="D",
                    help="max width(s) in blocks at the bulge; one island per "
                         "value (default: 16 40 80)")
    ap.add_argument("--height", type=int, nargs="+", default=None,
                    help="total height(s) in blocks, tip to flat top. "
                         "One value applies to every size; a list is "
                         "paired with --diameter. If omitted, each island "
                         "uses height ~= 1.2 * diameter so proportions are "
                         "kept across sizes.")
    ap.add_argument("--shape-exponent", type=float, default=2.0, dest="shape_exponent",
                    help="tip character; higher = spikier (default: 2.0)")
    ap.add_argument("--cut-frac", type=float, default=0.80, dest="cut_frac",
                    help="flat-slice position on the full drop, 0..1 "
                         "(default: 0.80)")
    ap.add_argument("--tip-frac", type=float, default=0.12, dest="tip_frac",
                    help="tip-slice position on the full drop, 0..cut_frac; "
                         "where the exponential taper is cut off so the "
                         "bottom ends on a small blunt point (default: 0.12)")
    ap.add_argument("--margin", type=int, default=2,
                    help="air padding around the model (default: 2)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "out", dest="out_dir",
                    help="directory for outputs (default: islands/out)")
    ap.add_argument("--screenshot", type=Path, default=None,
                    help="combined screenshot path (default: <out-dir>/island.png)")
    ap.add_argument("--no-screenshot", action="store_true",
                    help="skip rendering all screenshots")
    ap.add_argument("--title", type=str, default=None, help="sheet title")
    args = ap.parse_args()

    if args.height is not None and len(args.height) not in (1, len(args.diameter)):
        ap.error("--height needs one value or one per --diameter")

    structures, labels = [], []
    for i, diameter in enumerate(args.diameter):
        if args.height is None:
            height = int(round(diameter * 1.2))
        elif len(args.height) == 1:
            height = args.height[0]
        else:
            height = args.height[i]
        params = IslandParams(
            diameter=diameter,
            height=height,
            shape_exponent=args.shape_exponent,
            cut_frac=args.cut_frac,
            tip_frac=args.tip_frac,
            margin=args.margin,
        )

        structure = generate_island(params)
        npz_path = args.out_dir / f"island_d{diameter}.npz"
        structure.save(npz_path)

        stats = structure.stats()
        radii = island_radii(params)
        bulge_y = int(np.argmax(radii))
        rim = radii[-1] / (0.5 * diameter) * 100
        print(f"d={diameter:<3} h={height:<3} -> {npz_path}   "
              f"blocks={sum(stats.values())} "
              f"(grass={stats['grass']}, dirt={stats['dirt']})   "
              f"bulge@y={bulge_y}/{height - 1}   rim={rim:.0f}% of bulge")

        structures.append(structure)
        labels.append(f"d={diameter} h={height}")

    if not args.no_screenshot:
        sheet = args.screenshot or args.out_dir / "island.png"
        title = args.title or "teardrop island — sizes"
        render_sheet(structures, sheet, labels=labels, title=title)
        print(f"wrote {sheet}")


if __name__ == "__main__":
    main()
