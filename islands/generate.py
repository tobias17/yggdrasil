"""Generate a simple floating island: a flat top tapering down to a point.

Shape (base silhouette):
  * No mid-body bulge: the island is widest at the flat top and tapers
    smoothly down to a single-voxel point at the bottom — a simple
    "flat down to a point" silhouette. The default taper (shape_exponent
    1.7) leans the sides in early so the body reads slim rather than
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
  * A handful of dirt spikes hang from the underside (default on,
    auto-scaled by disc area: ~1 on d=16, ~5 on d=40, ~20 on d=80).
    Each anchors just below the local underside surface and hangs
    straight down, poking out beyond the body like an icicle. Spike
    size scales with island radius (unit size at d=16), so the big
    islands get proportionally bigger spikes. They are pure dirt, so
    the grass rule never touches them. Use --spike-count 0 to turn
    them off.

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

# how far an underhanging spike may poke out beyond the local body
# surface (blocks) at the reference size; also used to size the grid so
# nothing gets clipped. Scales with island radius (see SPIKE_REF_RADIUS)
# so spikes grow proportionally on larger islands.
MAX_SPIKE_PROTRUSION = 5.0
# island radius (blocks) at which spikes are "unit" size (d=16); spike
# width/length/poke are multiplied by R / SPIKE_REF_RADIUS so the big
# islands get proportionally bigger spikes instead of the same tiny ones
SPIKE_REF_RADIUS = 8.0


@dataclass(frozen=True)
class IslandParams:
    """Parameters for the simple teardrop island.

    diameter        max width of the island in blocks (width at the flat top)
    height          total height in blocks, from the point to the flat top
    shape_exponent  taper character (default 1.7). Higher leans the
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
    spike_count     how many underhanging spikes hang from the underside
                    (default None = auto-scale by disc area, see
                    auto_spike_count; 0 = none)
    spike_len_min   shortest acceptable spike length in rows, at the
                    reference size; scaled up by island radius (see
                    SPIKE_REF_RADIUS) and spikes shorter than that are
                    skipped (default 3)
    spike_len_max   longest spike length in rows, at the reference size;
                    scaled up by island radius (default 7)
    spike_width     typical spike base radius in blocks, at the reference
                    size; each spike's base is spike_width * uniform(0.6,
                    1.4), and everything is scaled up by island radius,
                    tapering towards a point at the tip (default 1.2)
    """

    diameter: int = 40
    height: int = 32
    shape_exponent: float = 1.7
    margin: int = 2
    spike_center_frac: float | None = None
    spike_target: tuple[float, float] | None = None
    noise_amp: float = 0.08
    noise_cells: int = 5
    noise_octaves: int = 3
    noise_seed: int = 0
    spike_count: int | None = None
    spike_len_min: int = 3
    spike_len_max: int = 7
    spike_width: float = 1.2


def auto_spike_count(diameter: int) -> int:
    """Default number of underhanging spikes for a given top diameter.

    Scales with the disc area (d**2 / 320) so the spike density stays
    roughly constant across sizes: d=16 -> 1, d=40 -> 5, d=80 -> 20.
    """
    return int(round(diameter * diameter / 320))


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


def add_underhanging_spikes(
    data: np.ndarray,
    p: IslandParams,
    radii: np.ndarray,
    centers: np.ndarray,
    c: int,
    coords: np.ndarray,
    size: int,
    profile: np.ndarray,
    dirt: int,
) -> int:
    """Hang small dirt spikes from the underside of the island body.

    Each spike anchors at a random point on the underside, its base
    buried below the local (wobbled) surface, and tapers straight down
    to a point. Width, length and how far it may hang out all scale
    with the island radius (R / SPIKE_REF_RADIUS), so the big islands
    get proportionally bigger spikes. A spike is extended row by row
    only while its outer edge pokes out at most its poke budget beyond
    the local body surface: as the cone narrows the surface retreats
    beneath the spike and ends it. All spike voxels are dirt and every
    spike row is narrower than the one above it, so the grass rule
    never paints them. Returns the number of spikes actually placed.
    """
    if p.spike_count is None:
        n = auto_spike_count(p.diameter)
    else:
        n = p.spike_count
    if n <= 0:
        return 0

    # spikes scale with island size: unit size at SPIKE_REF_RADIUS
    R = 0.5 * p.diameter
    scale = R / SPIKE_REF_RADIUS
    len_max = max(1, int(round(p.spike_len_max * scale)))
    len_min = max(1, int(round(p.spike_len_min * scale)))

    def prof_at(x, z):
        ix = min(max(int(round(x)), 0), size - 1)
        iz = min(max(int(round(z)), 0), size - 1)
        return float(profile[ix, iz])

    def boundary_r(cx, cz, r, ang):
        # distance from the row's center to its wobbled boundary in
        # direction ang. The boundary radius r * (1 + amp * profile) is
        # constant along a ray from the center (the profile depends only
        # on the column's angle), so it is monotonic and a binary
        # search finds the crossing.
        ca, sa = np.cos(ang), np.sin(ang)
        lo, hi = 0.0, r * (1.0 + p.noise_amp) + 0.5
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            x, z = cx + mid * ca, cz + mid * sa
            if mid <= r * (1.0 + p.noise_amp * prof_at(x, z)):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # anchor rows: wide enough to bury a base, not the tip rows, and not
    # the topmost rows (the grass cap stays spike-free). Wider rows are
    # more likely anchors: their gentler taper lets spikes hang longer,
    # so the retry budget is spent where spikes actually survive.
    attach_rows = [y for y in range(2, p.height - 2) if radii[y] >= 2.5]
    row_w = np.array([radii[y] for y in attach_rows], dtype=float)
    row_w = row_w / row_w.sum()
    placed = 0
    for _ in range(n * 40):  # generous tries: short candidates are skipped
        if placed >= n or not attach_rows:
            break
        y = attach_rows[int(np.random.choice(len(attach_rows), p=row_w))]
        base_r = p.spike_width * scale * float(np.random.uniform(0.6, 1.4))
        base_r = min(base_r, 0.45 * radii[y])
        ang = float(np.random.uniform(0.0, 2.0 * np.pi))
        ca, sa = np.cos(ang), np.sin(ang)
        cx0, cz0 = c + centers[y, 0], c + centers[y, 1]
        s0 = boundary_r(cx0, cz0, radii[y], ang)
        # base edge sits 0.75 inside the surface: buried even with the
        # center drift between rows, so the grass rule never sees it
        d0 = s0 - base_r - 0.75
        if d0 < 0.25:
            continue
        # the whole base disk must be inside the row: the wobble varies
        # around the circumference, so check the boundary at the disk's
        # two extreme points (along the tangent at the anchor) too
        base_ok = True
        for sgn in (-1.0, 1.0):
            px = cx0 + d0 * ca - sgn * base_r * sa
            pz = cz0 + d0 * sa + sgn * base_r * ca
            d_pt = float(np.hypot(px - cx0, pz - cz0))
            th = float(np.arctan2(pz - cz0, px - cx0))
            if d_pt - boundary_r(cx0, cz0, radii[y], th) > -0.25:
                base_ok = False
                break
        if not base_ok:
            continue
        # anchor the spike on a fixed grid column (the body's center
        # drift is negligible at this scale): every spike row then sits
        # exactly on the one above it, so the spike is a clean nested
        # cone with no grass specks and no detached voxels
        ax = c + centers[y, 0] + d0 * ca
        az = c + centers[y, 1] + d0 * sa
        # the spike may hang out 2..MAX_SPIKE_PROTRUSION blocks (at the
        # reference size) beyond the local body surface before the
        # retreating surface swallows it and the spike ends; scaled up
        # with island size
        p_max = float(np.random.uniform(2.0, MAX_SPIKE_PROTRUSION)) * scale

        # grow the spike row by row while it stays within its poke
        # budget; the lowest row written is row 1, keeping the main tip
        # row clear
        length = 0
        while length < len_max and y - length - 1 >= 1:
            j = length + 1
            cjx, cjz = c + centers[y - j, 0], c + centers[y - j, 1]
            sj = boundary_r(cjx, cjz, radii[y - j], ang)
            wr_j = max(0.5, base_r * (1.0 - 0.7 * j / (len_max + 1.0)))
            poke = float(np.hypot(ax - cjx, az - cjz)) + wr_j - sj
            if poke > p_max:
                break
            length = j
        if length < len_min:
            continue
        for j in range(length):
            wr = max(0.5, base_r * (1.0 - 0.7 * j / (len_max + 1.0)))
            d2 = (coords[:, None] - ax) ** 2 + (coords[None, :] - az) ** 2
            data[:, y - j, :][d2 <= wr * wr] = dirt
        placed += 1
    return placed


def generate_island(p: IslandParams) -> tuple[Structure, int]:
    """Build the island (flat top tapering to a point, with small dirt
    spikes hanging from the underside) as a Structure of dirt + grass.
    Returns the structure and the number of spikes actually placed."""
    atlas = Atlas()
    dirt = atlas.add("dirt")
    grass = atlas.add("grass")

    R = 0.5 * p.diameter
    # odd grid size so the center column is unambiguous; leave room for
    # the wobbled edge (up to noise_amp of the radius) and for a spike's
    # maximum poke-out beyond the body surface (spikes scale with R, so
    # so does the worst-case poke)
    scale = R / SPIKE_REF_RADIUS
    size = int(np.ceil(2 * R * (1.0 + p.noise_amp)
                       + 2 * MAX_SPIKE_PROTRUSION * scale))
    size += 2 * p.margin
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

    # small dirt spikes hanging from the underside (pure dirt; the grass
    # pass below leaves them alone because every spike block has a solid
    # block above it)
    n_spikes = add_underhanging_spikes(data, p, radii, centers, c, coords,
                                       size, profile, dirt)

    # Minecraft rule: dirt with no block directly above becomes grass.
    # has_above[y] is True where the block one row up is solid.
    # NOTE: must be a real bool array — an int mask would be interpreted as
    # integer indexing, not a boolean mask.
    has_above = np.zeros(data.shape, dtype=bool)
    has_above[:, : p.height - 1, :] = data[:, 1:, :] != 0
    data[(data == dirt) & ~has_above] = grass

    return Structure.from_data(data, atlas), n_spikes


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
    ap.add_argument("--shape-exponent", type=float, default=1.7, dest="shape_exponent",
                    help="taper character; 1.7 = default (leans in early, "
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
    ap.add_argument("--spike-count", type=int, default=None,
                    dest="spike_count",
                    help="number of small dirt spikes hanging from the "
                         "underside; default = auto-scale by disc area "
                         "(d=16 -> 1, d=40 -> 5, d=80 -> 20); 0 = none")
    ap.add_argument("--spike-len-min", type=int, default=3,
                    dest="spike_len_min",
                    help="shortest acceptable spike length in rows at the "
                         "reference size (d=16); scaled up by island "
                         "radius, candidates shorter than that are "
                         "skipped (default: 3)")
    ap.add_argument("--spike-len-max", type=int, default=7,
                    dest="spike_len_max",
                    help="longest spike length in rows at the reference "
                         "size (d=16); scaled up by island radius "
                         "(default: 7)")
    ap.add_argument("--spike-width", type=float, default=1.2,
                    dest="spike_width",
                    help="typical spike base radius in blocks at the "
                         "reference size (d=16); each spike varies by "
                         "+/-30%% and everything scales up with island "
                         "radius (default: 1.2)")
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
    if args.spike_count is not None and args.spike_count < 0:
        ap.error("--spike-count must be >= 0")
    if args.spike_len_min < 2:
        ap.error("--spike-len-min must be >= 2")
    if args.spike_len_max < args.spike_len_min:
        ap.error("--spike-len-max must be >= --spike-len-min")
    if args.spike_width <= 0.0:
        ap.error("--spike-width must be > 0")

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
            spike_count=args.spike_count,
            spike_len_min=args.spike_len_min,
            spike_len_max=args.spike_len_max,
            spike_width=args.spike_width,
        )

        structure, n_spikes = generate_island(params)
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
              f"spikes={n_spikes}   "
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
