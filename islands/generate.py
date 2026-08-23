"""Generate a simple floating island: an inverted teardrop with a flat top.

Shape (noise-free base silhouette for now):
  * A water drop turned point-down: a single-block point at the bottom that
    grows into a rounded bulge in the upper part, then tapers slightly.
  * The top is sliced off flat a little above the bulge, so the island is
    flat on top but still bulges out a bit before tapering inward at the rim.

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
    tip_width_frac  minimum radius of the island, as a fraction of the
                    bulge radius. 0 = pure drop tapering to a single-block
                    point; >0 clips the needle so the bottom ends on a
                    small blunt point of that width (default 0.06)
    tip_linear      how much straight conical line to mix into the tip,
                    0..1. 0 = pure spiky drop (long needle tip); 1 = the
                    full tangent cone (tip is a straight line from a blunt
                    base, ends abruptly). The cone is tangent to the drop
                    so the spiky body is unchanged (default 0.7)
    margin          air padding around the model in blocks
    """

    diameter: int = 40
    height: int = 48
    shape_exponent: float = 2.0
    cut_frac: float = 0.80
    tip_width_frac: float = 0.06
    tip_linear: float = 0.7
    margin: int = 2


def _tangent_cone(g: float, cut: float) -> tuple[float, float]:
    """Tangent from the tip to the smooth drop profile ``sin(pi*(t*cut)**g)``.

    Returns ``(slope, t_star)`` of the straight line through the origin that
    touches the profile once, at ``t = t_star``:  slope = s(t*)/t* = s'(t*).
    Near the tip the profile is quadratic (zero slope), so this line rises
    above it for t < t_star and stays below it for t > t_star — combining the
    two with ``max`` turns the asymptoting needle tip into a straight cone
    that hands off to the spiky drop exactly at the tangent point.
    Returns ``(0.0, 0.0)`` if no clean tangent exists.
    """
    def s(t):
        return np.sin(np.pi * (t * cut) ** g)

    def sp(t):
        return (np.pi * cut**g * g * (t * cut) ** (g - 1)
                * np.cos(np.pi * (t * cut) ** g))

    # tangent condition: s(t)/t == s'(t)  (secant slope == tangent slope)
    f = lambda t: s(t) / t - sp(t)
    ts = np.linspace(0.05, 0.999, 400)
    vals = np.array([f(t) for t in ts])
    sign_change = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(sign_change) == 0:
        return 0.0, 0.0
    a, b = ts[sign_change[0]], ts[sign_change[0] + 1]
    for _ in range(60):
        m = 0.5 * (a + b)
        if f(a) * f(m) <= 0:
            b = m
        else:
            a = m
    t_star = float(0.5 * (a + b))
    return float(s(t_star) / t_star), t_star


def island_radii(p: IslandParams) -> np.ndarray:
    """Radius in blocks of the island at each height row, tip (0) to top (-1).

    The body is a water-drop profile ``sin(pi * u**g)`` (point at both ends,
    bulge at u = 0.5**(1/g)), sliced at ``cut_frac`` and rescaled so the flat
    top is t = 1:

        drop(t) = sin(pi * (t * cut_frac) ** shape_exponent)

    The tip is where this profile asymptotes (zero slope at the point), so
    on its own the island ends in a long thin needle. To make it end more
    abruptly we combine the drop with a straight conical line from the tip:

        r(t) = R * max( drop(t), tip_linear * cone_slope * t ),  t <= t_star
        r(t) = R * drop(t),                                       t > t_star

    ``tip_linear`` in [0, 1] scales the cone: 0 = pure drop (needle tip, the
    old look), 1 = the full tangent cone (tip is a straight line up to the
    tangent point, where it meets the drop kink-free). The drop is kept
    above the tangent point so the spiky body and the tapered rim are
    unchanged. ``tip_width_frac`` then clips the very point so the bottom is
    a small blunt disc instead of a single block.
    """
    R = 0.5 * p.diameter
    t = np.arange(p.height) / (p.height - 1)
    drop = np.sin(np.pi * np.power(t * p.cut_frac, p.shape_exponent))

    profile = drop
    if p.tip_linear > 0:
        slope, t_star = _tangent_cone(p.shape_exponent, p.cut_frac)
        if slope > 0:
            cone = p.tip_linear * slope * t
            profile = np.where((cone > drop) & (t <= t_star), cone, drop)

    radii = R * profile
    if p.tip_width_frac > 0:
        radii = np.maximum(radii, R * p.tip_width_frac)
    return radii


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
    ap.add_argument("--height", type=int, default=None,
                    help="total height in blocks, tip to flat top. "
                         "If omitted, each island uses height ~= 1.2 * diameter "
                         "so proportions are kept across sizes.")
    ap.add_argument("--shape-exponent", type=float, default=2.0, dest="shape_exponent",
                    help="tip character; higher = spikier (default: 2.0)")
    ap.add_argument("--cut-frac", type=float, default=0.80, dest="cut_frac",
                    help="flat-slice position on the full drop, 0..1 "
                         "(default: 0.80)")
    ap.add_argument("--tip-width-frac", type=float, default=0.06, dest="tip_width_frac",
                    help="tip width as a fraction of the bulge radius; "
                         "0 = pure drop to a point (default: 0.06)")
    ap.add_argument("--tip-linear", type=float, default=0.7, dest="tip_linear",
                    help="straight conical line mixed into the tip, 0..1; "
                         "0 = pure spiky drop, 1 = full tangent cone "
                         "(default: 0.7)")
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

    structures, labels = [], []
    for diameter in args.diameter:
        height = args.height if args.height is not None else int(round(diameter * 1.2))
        params = IslandParams(
            diameter=diameter,
            height=height,
            shape_exponent=args.shape_exponent,
            cut_frac=args.cut_frac,
            tip_width_frac=args.tip_width_frac,
            tip_linear=args.tip_linear,
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
