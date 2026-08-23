"""Generate a simple floating island: a flat top tapering down to a point.

Shape (base silhouette):
  * No mid-body bulge: the island is widest at the flat top and tapers
    smoothly down to a single-voxel point at the bottom — a simple
    "flat down to a point" silhouette. The default taper (shape_exponent
    1.5) leans the sides in early so the body reads slim rather than
    bloated; raise the exponent for a longer, thinner point or lower it
    for a squatter body.
  * The flat top is a disk with a rough edge, not a perfect circle: the
    boundary noise below wobbles the circumference at every height,
    including the top rows (default --noise-amp 0.08 = a slight roughness;
    0 turns it back into a perfect circle).
  * The point is not necessarily centered: the tip targets a random point
    within 50% of the island radius (default 0.5, uniformly in that disk).
    The disk center stays on the island center at the flat top, then
    drifts towards the target down the body so the tip lands exactly on
    it — a centered target gives a symmetric cone, a far target leans the
    body to one side.
  * The circumference wobble is Perlin-like value noise, layered (fBm) so
    the silhouette varies at the macro level instead of being a perfect
    solid of revolution. Its amplitude is a fraction of the row radius,
    and the same angular profile is scaled to every row, so the surface
    stays a smooth wavy solid of revolution with no horizontal layering.

Test sizes: the default run renders d=16 h=16, d=40 h=32, d=80 h=60.

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

from utils import Atlas, Structure, render_sheet, value_noise  # noqa: E402


@dataclass(frozen=True)
class IslandParams:
    """Parameters for the simple teardrop island.

    diameter        max width of the island in blocks (width at the flat top)
    height          total height in blocks, from the point to the flat top
    shape_exponent  taper character (default 1.5). Higher leans the
                    sides in early (slimmer body, longer thinner point);
                    1.0 is a smooth, balanced half-sine (wider, rounder);
                    lower keeps the sides wide (blunter, squat bottom)
    margin          air padding around the model in blocks
    spike_center_frac max distance of the tip's target point from the
                    center, as a fraction of the island radius, 0..1
                    (0 = perfectly centered; default 0.5 = within 50% of
                    the radius). Each island picks the target as a random
                    point uniformly inside that disk; as the profile
                    narrows into the spike, the disk center drifts from
                    the island center to that point, so the spike skews
                    to one side
                    (default: random per island; use ``spike_target`` to
                    fix it explicitly)
    spike_target    optional explicit (dx, dz) offset of the spike tip in
                    blocks relative to the island center; overrides
                    spike_center_frac (default: None)
    noise_amp       how much the Perlin-like boundary noise perturbs each
                    row, as a fraction of that row's radius (default
                    0.08 = a slight roughness around the circumference;
                    0 = off, a perfect circle; 0.22 = a heavier texture).
                    It is applied at full strength to every row,
                    including the flat top rows, so the disc's edge is
                    never a perfect circle
    noise_cells     number of large lobes of the boundary noise around the
                    circumference (default 5); fewer = bigger, smoother
                    macro lumps, more = finer roughness
    noise_octaves   how many octaves of the boundary noise are layered
                    (fBm). 1 = just the smooth macro lobes; 3 = macro lumps
                    plus two finer, smoother detail octaves so the surface
                    doesn't look like perfectly smooth ovals (default 3)
    noise_seed      seed for the boundary noise (default 0); combine with
                    --seed to get a stable layout + stable texture
    """

    diameter: int = 40
    height: int = 32
    shape_exponent: float = 1.5
    margin: int = 2
    spike_center_frac: float | None = None
    spike_target: tuple[float, float] | None = None
    noise_amp: float = 0.08
    noise_cells: int = 5
    noise_octaves: int = 3
    noise_seed: int = 0


def island_rows(p: IslandParams) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (radius, center offset) for every height row, tip (0) to top (-1).

    The silhouette tapers from a flat top down to a point — no mid-body
    bulge. The radius is the full island radius at the top row and decreases
    monotonically to zero at the bottom tip. It is the top half of a smooth
    drop stretched over the island height:

        r(t) = R * sin(pi/2 * t**shape_exponent)      (t in 0..1, t=0 is the
                                                       bottom tip)

    so r(0) = 0 (a single-voxel point) and r(1) = R (the flat top). Because
    sin is flat at pi/2, the radius stays very close to R over the top few
    rows (the "flat" top); because sin is linear at 0, the bottom comes to a
    clean point. ``shape_exponent`` shapes the taper: 1.0 is a balanced cone;
    higher keeps the sides narrow for longer (a longer, thinner point); lower
    widens the body sooner (fatter, rounder).

    The center offset stays at (0, 0) on the flat top (the widest row), then
    drifts linearly down towards the spike target so the tip row's disk is
    centered exactly on it. The target is a random point uniformly inside
    a disk of radius ``spike_center_frac`` of the island radius (default
    0.5 = within 50% of the radius), or at ``spike_target`` if given.
    Returns ``(radii, centers)`` with shapes ``(height,)`` and
    ``(height, 2)`` (dx, dz in blocks relative to the island center).
    """
    R = 0.5 * p.diameter
    t = np.arange(p.height) / (p.height - 1)
    radii = R * np.sin(0.5 * np.pi * np.power(t, p.shape_exponent))

    # spike target: an explicit (dx, dz), or a random point uniformly
    # inside a disk of radius spike_center_frac of the island radius
    # (default 0.5 = within 50% of the radius)
    if p.spike_target is not None:
        tx, tz = p.spike_target
    else:
        frac = 0.5 if p.spike_center_frac is None else p.spike_center_frac
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        dist = R * frac * float(np.sqrt(np.random.uniform(0.0, 1.0)))
        tx, tz = dist * np.cos(angle), dist * np.sin(angle)

    # The radius is smallest at the tip row (y=0) and grows to its maximum
    # at the flat top (the widest row), so the profile "starts to narrow"
    # (going down) at the top. The disk center stays on the island center
    # at the flat top, then drifts linearly down towards the target so the
    # tip row's disk is centered exactly on it.
    widest_row = int(np.argmax(radii))
    centers = np.zeros((p.height, 2), dtype=float)
    if widest_row > 0:
        w = np.clip(1.0 - np.arange(p.height) / widest_row, 0.0, 1.0)
        centers[:, 0] = tx * w
        centers[:, 1] = tz * w
    else:
        # degenerate: a single row, land the tip on the target
        centers[0] = (tx, tz)

    return radii, centers


def generate_island(p: IslandParams) -> Structure:
    """Build the island (flat top tapering to a point) as a Structure of
    dirt + grass."""
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

    # Per-row (X, Z) disk. The radius profile tapers from the flat top to
    # the point; the disk center drifts from (0, 0) to the spike target
    # down the body; and the boundary is perturbed by Perlin-like noise so
    # the solid stops looking like a perfect solid of revolution.
    # data[:, y, :] is (X, Z): axis 0 -> X, axis 1 -> Z
    coords = np.arange(size)
    radii, centers = island_rows(p)

    # angle of every grid column around the island center (radians). The
    # The boundary noise is a few layered, periodic 1D profiles of angle
    # (Perlin-like value noise, one octave each), summed with decreasing
    # weight — fBm style. The base octave is the coarse macro lumps; the
    # higher octaves add smoother fine detail so the surface doesn't read as
    # a few perfectly smooth ovals. Each octave is applied identically to
    # every height row, so each row is a scaled copy of the ones above it:
    # the surface stays a smooth, wavy solid of revolution with no
    # horizontal layering (a row is never locally wider than the row above
    # it, so the grass rule never paints interior shelves).
    ang = np.arctan2(coords[None, :] - c, coords[:, None] - c)
    u_ang = np.mod(ang, 2.0 * np.pi) / (2.0 * np.pi)

    def _octave(cells, seed):
        prof = value_noise(u_ang, np.full_like(u_ang, 0.5),
                           cells, 1, seed, periodic_u=True)
        prof = prof - prof.mean()
        peak = float(np.abs(prof).max())
        return prof / peak if peak > 0.0 else prof

    # zero-mean + peak-normalize each octave, then sum them with halved
    # weight per octave (macro dominates, detail is a minor variance) and
    # renormalize the sum so noise_amp still directly controls the max
    # wobble: the edge moves at most noise_amp * radius in/out. Each
    # octave's lobe count is capped so a lobe spans a few blocks — on a
    # small island the finest octaves fall out, so the surface stays
    # restrained instead of jutting out block by block.
    max_cells = max(3, int(R) // 2)
    profile = np.zeros_like(u_ang)
    weight = 1.0
    for k in range(max(1, p.noise_octaves)):
        cells_k = min(max(2, p.noise_cells * (2 ** k)), max_cells)
        profile += weight * _octave(cells_k, p.noise_seed + 7919 * k)
        weight *= 0.5
    peak = float(np.abs(profile).max())
    if peak > 0.0:
        profile = profile / peak

    # The wobble is applied at full strength to every row, including the
    # flat top rows, so the disc's circumference is rough rather than a
    # perfect circle. Because the wobble is the same profile scaled to
    # each row's radius, the surface stays a smooth wavy solid of
    # revolution and never creates interior grass shelves.

    def row_mask(cx, cz, r, amp_row):
        d2 = (coords[:, None] - cx) ** 2 + (coords[None, :] - cz) ** 2
        rr = r * (1.0 + amp_row * profile)
        return d2 <= rr * rr

    def disk(cx, cz, r, amp_row):
        # row disk; if the radius is too small for the (possibly
        # non-integer) center to contain any cell — the tip row has r = 0 —
        # fall back to a single voxel at the rounded center so the tip is
        # always a real, on-target voxel.
        m = row_mask(cx, cz, r, amp_row)
        if not np.any(m):
            m = np.zeros_like(m)
            m[int(round(cx)), int(round(cz))] = True
        return m

    prev_mask, prev_cx, prev_cz = None, None, None
    for y, r in enumerate(radii):
        cx, cz = c + centers[y, 0], c + centers[y, 1]
        mask = disk(cx, cz, r, p.noise_amp)
        if y > 0 and not np.any(mask & prev_mask):
            # the per-row center drift (or the wobble) outran the (tiny)
            # tip radius and the two rows would not touch: nudge this
            # row's center towards the row below until they share a
            # voxel. The tip row (y=0) is never moved, so the tip still
            # lands on target.
            for _ in range(64):
                step = np.hypot(prev_cx - cx, prev_cz - cz)
                if step < 1e-9:
                    break
                cx += 0.25 * (prev_cx - cx) / step
                cz += 0.25 * (prev_cz - cz) / step
                mask = disk(cx, cz, r, p.noise_amp)
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
    ap.add_argument("--diameter", type=int, nargs="+", default=None,
                    metavar="D",
                    help="max width(s) in blocks at the flat top; one island "
                         "per value (default: 16 40 80, paired with the "
                         "default heights 16 32 60)")
    ap.add_argument("--height", type=int, nargs="+", default=None,
                    help="total height(s) in blocks, point to flat top. "
                         "One value applies to every size; a list is "
                         "paired with --diameter. If omitted together with "
                         "--diameter, the default pairs (16,16) (40,32) "
                         "(80,60) are used; otherwise each island uses "
                         "height ~= 1.2 * diameter so proportions are kept "
                         "across sizes.")
    ap.add_argument("--shape-exponent", type=float, default=1.5, dest="shape_exponent",
                    help="taper character; 1.5 = default (leans in early, "
                         "slim body), higher = longer/thinner point, "
                         "lower = wider/squatter (1.0 = balanced half-sine)")
    ap.add_argument("--margin", type=int, default=2,
                    help="air padding around the model (default: 2)")
    ap.add_argument("--spike-center-frac", type=float, default=None,
                    dest="spike_center_frac",
                    help="max distance of the tip's target point from the "
                         "center, as a fraction of the island radius, "
                         "0..1; the target is a random point uniformly "
                         "inside that disk (default: 0.5 = within 50% of "
                         "the radius, 0 = perfectly centered)")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for spike target placement and the "
                         "boundary noise; makes runs reproducible "
                         "(default: unseeded)")
    ap.add_argument("--noise-amp", type=float, default=0.08, dest="noise_amp",
                    help="boundary noise amplitude as a fraction of the row "
                         "radius; roughness around the circumference, "
                         "applied to every row including the flat top "
                         "(default: 0.08 = slight roughness, 0 = perfect "
                         "circle, 0.22 = heavier texture)")
    ap.add_argument("--noise-cells", type=int, default=5, dest="noise_cells",
                    help="number of large lobes of boundary noise around "
                         "the circumference; fewer = bigger, smoother "
                         "macro lumps, more = finer roughness (default: 5)")
    ap.add_argument("--noise-octaves", type=int, default=3, dest="noise_octaves",
                    help="octaves of layered boundary noise (fBm); more = "
                         "finer detail on top of the macro lumps so the "
                         "surface isn't perfectly smooth (default: 3)")
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
    if args.spike_center_frac is not None and not 0.0 <= args.spike_center_frac <= 1.0:
        ap.error("--spike-center-frac must be in 0..1")
    if args.noise_amp < 0.0:
        ap.error("--noise-amp must be >= 0")
    if args.noise_cells < 1:
        ap.error("--noise-cells must be >= 1")
    if args.noise_octaves < 1:
        ap.error("--noise-octaves must be >= 1")

    # default test sizes: (diameter, height) pairs
    default_sizes = ((16, 16), (40, 32), (80, 60))
    if args.diameter is None:
        diameters = [d for d, _ in default_sizes]
    else:
        diameters = args.diameter
    if args.height is not None and len(args.height) not in (1, len(diameters)):
        ap.error("--height needs one value or one per --diameter")
    if args.height is None:
        if args.diameter is None:
            heights = [h for _, h in default_sizes]
        else:
            heights = [int(round(d * 1.2)) for d in diameters]
    else:
        heights = args.height

    structures, labels = [], []
    for i, diameter in enumerate(diameters):
        height = heights[0] if len(heights) == 1 else heights[i]
        # pick this island's spike target once so the saved model, the
        # printed tip and the screenshot all agree: a random point
        # uniformly inside a disk of spike_center_frac of the island
        # radius (default 0.5 = within 50% of the radius)
        frac = 0.5 if args.spike_center_frac is None else args.spike_center_frac
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        dist = 0.5 * diameter * frac * float(np.sqrt(np.random.uniform(0.0, 1.0)))
        spike_target = (dist * np.cos(angle), dist * np.sin(angle))
        params = IslandParams(
            diameter=diameter,
            height=height,
            shape_exponent=args.shape_exponent,
            margin=args.margin,
            spike_target=spike_target,
            noise_amp=args.noise_amp,
            noise_cells=args.noise_cells,
            noise_octaves=args.noise_octaves,
            noise_seed=(args.seed if args.seed is not None else 0),
        )

        structure = generate_island(params)
        npz_path = args.out_dir / f"island_d{diameter}.npz"
        structure.save(npz_path)

        stats = structure.stats()
        radii, centers = island_rows(params)
        c = structure.shape[0] // 2
        top_r = radii[-1] / (0.5 * diameter) * 100
        tip_x, tip_z = c + centers[0, 0], c + centers[0, 1]
        tip_off = float(np.hypot(centers[0, 0], centers[0, 1]))
        print(f"d={diameter:<3} h={height:<3} -> {npz_path}   "
              f"blocks={sum(stats.values())} "
              f"(grass={stats['grass']}, dirt={stats['dirt']})   "
              f"top={top_r:.0f}% of R   "
              f"tip@({tip_x:.1f}, {tip_z:.1f})  (off-center {tip_off:.1f})")

        structures.append(structure)
        labels.append(f"d={diameter} h={height}")

    if not args.no_screenshot:
        sheet = args.screenshot or args.out_dir / "island.png"
        title = args.title or "floating island — flat top to a point"
        render_sheet(structures, sheet, labels=labels, title=title)
        print(f"wrote {sheet}")


if __name__ == "__main__":
    main()
