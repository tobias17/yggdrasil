"""Generate a simple floating island: an inverted teardrop with a flat top.

Shape (noise-free base silhouette for now):
  * A water drop turned point-down: a small blunt point at the bottom that
    grows in an exponential/asymptotic taper into a rounded bulge in the
    upper part, then tapers slightly.
  * Both ends of the drop profile are sliced off: the top a little above
    the bulge (flat top that still bulges out at the rim), the bottom where
    the taper would otherwise drag out into a long 1-block tail.
  * The spike is not necessarily centered: a target point is picked at up
    to 60% of the island radius (default 0.6, random direction). The disk
    center stays on the island center down to where the profile starts to
    narrow, then drifts towards the target so the tip row lands exactly on
    it — a centered target reproduces the classic centered spike, a far
    target skews the spike to one side. The bulge and the flat top are
    unaffected.

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
    spike_center_frac where the spike's tip lands, as a fraction of the
                    island radius from center, 0..0.6 (0 = perfectly
                    centered, 0.6 = as far out as allowed). Each island
                    picks a target point at this distance in a random
                    direction; as the profile narrows into the spike, the
                    disk center drifts from the island center to that
                    point, so the spike skews to one side
                    (default: random per island; use ``spike_target`` to
                    fix it explicitly)
    spike_target    optional explicit (dx, dz) offset of the spike tip in
                    blocks relative to the island center; overrides
                    spike_center_frac (default: None)
    """

    diameter: int = 40
    height: int = 48
    shape_exponent: float = 2.0
    cut_frac: float = 0.80
    tip_frac: float = 0.12
    margin: int = 2
    spike_center_frac: float | None = None
    spike_target: tuple[float, float] | None = None


def island_rows(p: IslandParams) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (radius, center offset) for every height row, tip (0) to top (-1).

    The radius is a single water-drop profile ``sin(pi * u**g)`` (u in 0..1,
    point at both ends, bulge at u = 0.5**(1/g)) sliced at both ends and
    stretched over the island height:

        u(t) = tip_frac + (cut_frac - tip_frac) * t
        r(t) = R * sin(pi * u(t) ** shape_exponent)

    The drop's taper is the "asymptotic" look: quadratic at the point,
    widening fast through the bulge, then slowly rounding off towards the
    flat top. Left uncut, the tip end of the profile stays under one block
    wide for many rows (the long 1-block tail spike), so it is sliced off at
    ``tip_frac``: the bottom row starts on a small blunt point (radius
    R*sin(pi*tip_frac**g)) and the visible curve is otherwise untouched.

    The center offset stays at (0, 0) from the flat top down to the bulge
    (where the profile starts to narrow), then drifts linearly towards the
    spike target so the tip row's disk is centered exactly on it. The
    target sits at ``spike_center_frac`` of the island radius in a random
    direction (or at ``spike_target`` if given).
    Returns ``(radii, centers)`` with shapes ``(height,)`` and
    ``(height, 2)`` (dx, dz in blocks relative to the island center).
    """
    R = 0.5 * p.diameter
    t = np.arange(p.height) / (p.height - 1)
    u = p.tip_frac + (p.cut_frac - p.tip_frac) * t
    radii = R * np.sin(np.pi * np.power(u, p.shape_exponent))

    # spike target: an explicit (dx, dz), or a random point at
    # spike_center_frac of the radius (default 0.6) in a random direction
    if p.spike_target is not None:
        tx, tz = p.spike_target
    else:
        frac = 0.6 if p.spike_center_frac is None else p.spike_center_frac
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        tx, tz = R * frac * np.cos(angle), R * frac * np.sin(angle)

    # The spike is rows 0..bulge_row: the radius is smallest at the tip
    # row (y=0) and grows to its maximum at the bulge, so the profile
    # "starts to narrow" (going down) at the bulge. The disk center stays
    # on the island center down to the bulge, then drifts linearly
    # towards the target so the tip row's disk is centered exactly on it.
    # Rows above the bulge (the flat top) stay perfectly centered.
    bulge_row = int(np.argmax(radii))
    centers = np.zeros((p.height, 2), dtype=float)
    if bulge_row > 0:
        w = np.clip(1.0 - np.arange(p.height) / bulge_row, 0.0, 1.0)
        centers[:, 0] = tx * w
        centers[:, 1] = tz * w
    else:
        # degenerate: a single row, land the tip on the target
        centers[0] = (tx, tz)

    return radii, centers


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

    # squared column coordinate (X, Z); each row's disk is tested against
    # its own center, which drifts from (0, 0) to the spike target.
    # data[:, y, :] is (X, Z): axis 0 -> X (uses dx), axis 1 -> Z (uses dz)
    coords = np.arange(size)
    radii, centers = island_rows(p)

    def row_mask(cx, cz, r):
        d2 = (coords[:, None] - cx) ** 2 + (coords[None, :] - cz) ** 2
        return d2 <= r * r

    prev_mask, prev_cx, prev_cz = None, None, None
    for y, r in enumerate(radii):
        cx, cz = c + centers[y, 0], c + centers[y, 1]
        mask = row_mask(cx, cz, r)
        if y > 0 and not np.any(mask & prev_mask):
            # the per-row center drift outran the (tiny) tip radius and
            # the two rows would not touch: nudge this row's center
            # towards the row below until they share a voxel. The tip row
            # (y=0) is never moved, so the tip still lands on target.
            for _ in range(64):
                step = np.hypot(prev_cx - cx, prev_cz - cz)
                if step < 1e-9:
                    break
                cx += 0.25 * (prev_cx - cx) / step
                cz += 0.25 * (prev_cz - cz) / step
                mask = row_mask(cx, cz, r)
                if np.any(mask & prev_mask):
                    break
        data[:, y, :][mask] = dirt
        prev_mask, prev_cx, prev_cz = mask, cx, cz

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
    ap.add_argument("--spike-center-frac", type=float, default=None,
                    dest="spike_center_frac",
                    help="spike tip distance from center as a fraction of "
                         "the island radius, 0..0.6; the target point is "
                         "picked at this distance in a random direction "
                         "(default: 0.6)")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for spike target placement; makes "
                         "runs reproducible (default: unseeded)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "out", dest="out_dir",
                    help="directory for outputs (default: islands/out)")
    ap.add_argument("--screenshot", type=Path, default=None,
                    help="combined screenshot path (default: <out-dir>/island.png)")
    ap.add_argument("--no-screenshot", action="store_true",
                    help="skip rendering all screenshots")
    ap.add_argument("--title", type=str, default=None, help="sheet title")
    args = ap.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
    if args.spike_center_frac is not None and not 0.0 <= args.spike_center_frac <= 0.6:
        ap.error("--spike-center-frac must be in 0..0.6")

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
        # pick this island's spike target once so the saved model, the
        # printed tip and the screenshot all agree
        frac = 0.6 if args.spike_center_frac is None else args.spike_center_frac
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        spike_target = (0.5 * diameter * frac * np.cos(angle),
                        0.5 * diameter * frac * np.sin(angle))
        params = IslandParams(
            diameter=diameter,
            height=height,
            shape_exponent=args.shape_exponent,
            cut_frac=args.cut_frac,
            tip_frac=args.tip_frac,
            margin=args.margin,
            spike_target=spike_target,
        )

        structure = generate_island(params)
        npz_path = args.out_dir / f"island_d{diameter}.npz"
        structure.save(npz_path)

        stats = structure.stats()
        radii, centers = island_rows(params)
        bulge_y = int(np.argmax(radii))
        rim = radii[-1] / (0.5 * diameter) * 100
        c = structure.shape[0] // 2
        tip_x, tip_z = c + centers[0, 0], c + centers[0, 1]
        print(f"d={diameter:<3} h={height:<3} -> {npz_path}   "
              f"blocks={sum(stats.values())} "
              f"(grass={stats['grass']}, dirt={stats['dirt']})   "
              f"bulge@y={bulge_y}/{height - 1}   rim={rim:.0f}% of bulge   "
              f"tip@({tip_x:.1f}, {tip_z:.1f})")

        structures.append(structure)
        labels.append(f"d={diameter} h={height}")

    if not args.no_screenshot:
        sheet = args.screenshot or args.out_dir / "island.png"
        title = args.title or "teardrop island — sizes"
        render_sheet(structures, sheet, labels=labels, title=title)
        print(f"wrote {sheet}")


if __name__ == "__main__":
    main()
