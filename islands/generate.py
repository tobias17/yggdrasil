"""Generate a floating island: a lobed grass cap over a lumpy, asymmetric
rock body that frays into a fringe of hanging icicles.

Shape (target silhouette: islands/out/reference.jpg):
  * The cap -- the top is a roughly circular plateau whose rim is lobed and
    notched (angular value noise) instead of a perfect circle.
  * The body -- below the cap the rock tapers smoothly down to a narrow
    neck. The radius is neck + (R - neck) * sin(pi/2 * t**taper), with its
    own angular noise (so the cone is lumpy rather than a solid of
    revolution) and a directional asymmetry term (one side narrows faster
    than the other).
  * The neck -- the solid mass ends in a small disk (about 10-20% of the
    radius) roughly 30-50% of the way down; below it the underside is a
    fringe, not a point.
  * The fringe -- a field of thin vertical icicles hangs from the
    underside: 1-3 blocks wide at the base, tapering to a single-voxel
    tip, slightly bent and eroded. Each icicle starts a couple of rows
    inside the body at its column's underside and hangs down. Central
    columns (near the tip target) reach into the body lowest, so they hang
    longest; outer columns start higher and are shorter. The silhouette
    thus frays out of the cone instead of having spikes stuck on it.
  * The tip -- the lowest point of the fringe is a random point within
    `tip_frac` of the island radius (default range 0.2..0.55), i.e. the tip
    target the icicle cluster hangs around; a centered target gives a
    symmetric fringe, a far target swings the fringe to one side.
  * Per-island variety -- unless a flag pins a value, every island
    samples its own parameter set (taper, noise, asymmetry, neck, tip,
    icicle length and spread), so a batch of islands varies like the
    reference: spindly ones, stubby ones, notched ones.

Block layout:
  * Every solid voxel is created as dirt.
  * After that, any dirt block with no block directly above it becomes
    grass (the Minecraft surface rule): a green cap and nothing else,
    because every column (body or icicle) is solid top-to-bottom and
    never widens going down.

Output:
  * per size   ``islands/out/island_d<D>.npz`` -- the model data
  * combined   ``islands/out/island.png`` -- one side-by-side sheet with
               all sizes, re-rendered on every run so progress can be
               watched in a single image.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import Atlas, Structure, render_sheet, value_noise  # noqa: E402

# Per-island random ranges (low, high) sampled when the corresponding flag
# is not given explicitly. Values are what a batch of islands "rolls" so
# the shapes vary like the reference image.
RANGE_TAPER = (1.3, 2.0)        # taper character of the body
RANGE_CAP_AMP = (0.15, 0.28)    # cap rim lobing, fraction of the radius
RANGE_BODY_AMP = (0.10, 0.25)   # body lumpiness, fraction of the radius
RANGE_ASYM = (0.10, 0.45)       # directional narrowing, fraction of radius
RANGE_TIP_FRAC = (0.2, 0.55)    # tip target within this fraction of R
RANGE_NECK_FRAC = (0.3, 0.5)    # fringe zone: bottom fraction of the height
RANGE_NECK_WIDTH = (0.10, 0.20) # neck radius, fraction of R
RANGE_ICICLE_LEN = (0.45, 0.65) # max icicle length, fraction of the height
RANGE_ICICLE_FOOT = (0.35, 0.6) # fringe spread around the tip, fraction of R
LOBES_CAP = (3, 8)              # cap lobe count range (inclusive low/high)
LOBES_BODY = (4, 10)


@dataclass(frozen=True)
class IslandParams:
    """A fully resolved parameter set for one island."""

    diameter: int          # cap width in blocks (the island's max width)
    height: int            # total height in blocks, cap to lowest icicle
    taper: float           # body taper character (see island docstring)
    cap_amp: float         # cap rim lobing amplitude (fraction of radius)
    cap_lobes: int         # number of lobes in the cap rim noise
    body_amp: float        # body lumpiness amplitude (fraction of radius)
    body_lobes: int        # number of lobes in the body noise
    asym: float            # directional asymmetry (fraction of radius)
    asym_dir: float        # direction of the wider side (radians)
    tip_target: tuple      # (dx, dz) of the tip/fringe center, blocks
    neck_frac: float       # bottom fraction of the height that is fringe
    neck_width: float      # neck radius, fraction of R
    icicle_len_frac: float # max icicle length, fraction of the height
    icicle_width: float    # typical icicle base radius in blocks
    icicle_foot: float     # fringe spread radius, fraction of R
    icicle_count: int      # number of icicles (0 = none)
    margin: int            # air padding around the model in blocks
    noise_seed: int        # seed for all of this island's noise


def auto_icicle_count(diameter: int) -> int:
    """Default fringe density: scales with the disc area so small islands
    get a handful of icicles and big ones a dense fringe
    (d=16 -> 8, d=40 -> 19, d=80 -> 77)."""
    return max(8, int(round(0.012 * diameter * diameter)))


def sample_params(rng: np.random.Generator, diameter: int, height: int,
                  o: dict) -> IslandParams:
    """Resolve an island's parameter set.

    Every entry of ``o`` that is not None is used verbatim (a pinned flag);
    everything else is sampled from its range with the island's rng, so
    different islands roll different shapes from the same code.
    """
    R = 0.5 * diameter
    taper = o["taper"] if o["taper"] is not None else float(rng.uniform(*RANGE_TAPER))
    cap_amp = o["cap_amp"] if o["cap_amp"] is not None else float(rng.uniform(*RANGE_CAP_AMP))
    body_amp = o["body_amp"] if o["body_amp"] is not None else float(rng.uniform(*RANGE_BODY_AMP))
    asym = o["asym"] if o["asym"] is not None else float(rng.uniform(*RANGE_ASYM))
    asym_dir = float(rng.uniform(0.0, 2.0 * np.pi))
    tip_frac = o["tip_frac"] if o["tip_frac"] is not None else float(rng.uniform(*RANGE_TIP_FRAC))
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    dist = R * tip_frac * float(np.sqrt(rng.uniform(0.0, 1.0)))
    tip_target = (dist * np.cos(ang), dist * np.sin(ang))
    neck_frac = o["neck_frac"] if o["neck_frac"] is not None else float(rng.uniform(*RANGE_NECK_FRAC))
    neck_width = o["neck_width"] if o["neck_width"] is not None else float(rng.uniform(*RANGE_NECK_WIDTH))
    icicle_len_frac = o["icicle_len"] if o["icicle_len"] is not None else float(rng.uniform(*RANGE_ICICLE_LEN))
    icicle_foot = o["icicle_foot"] if o["icicle_foot"] is not None else float(rng.uniform(*RANGE_ICICLE_FOOT))
    icicle_width = o["icicle_width"] if o["icicle_width"] is not None else 1.0
    icicle_count = o["icicle_count"] if o["icicle_count"] is not None else auto_icicle_count(diameter)
    cap_lobes = int(o["cap_lobes"]) if o["cap_lobes"] is not None else int(rng.integers(*LOBES_CAP))
    body_lobes = int(o["body_lobes"]) if o["body_lobes"] is not None else int(rng.integers(*LOBES_BODY))
    noise_seed = int(rng.integers(0, 2**31 - 1))
    return IslandParams(
        diameter=diameter, height=height,
        taper=taper,
        cap_amp=cap_amp, cap_lobes=cap_lobes,
        body_amp=body_amp, body_lobes=body_lobes,
        asym=asym, asym_dir=asym_dir,
        tip_target=tip_target,
        neck_frac=neck_frac, neck_width=neck_width,
        icicle_len_frac=icicle_len_frac, icicle_width=icicle_width,
        icicle_foot=icicle_foot, icicle_count=icicle_count,
        margin=o["margin"], noise_seed=noise_seed,
    )


def angular_profile(size: int, c: int, lobes: int, octaves: int,
                    seed: int) -> np.ndarray:
    """A zero-mean, peak-normalized 2D angular noise field around the grid
    center (c, c).

    Layered (fBm) periodic value noise of the angle around (c, c): the base
    octave is the coarse lobes, higher octaves add finer detail. Because
    every height row scales the same profile, the surface stays a smooth
    wavy solid of revolution -- wobble in, wobble out, no horizontal
    banding.
    """
    coords = np.arange(size)
    ang = np.arctan2(coords[None, :] - c, coords[:, None] - c)
    u = np.mod(ang, 2.0 * np.pi) / (2.0 * np.pi)
    v = np.full_like(u, 0.5)
    prof = np.zeros_like(u)
    weight = 1.0
    for k in range(max(1, octaves)):
        cells = max(2, int(round(lobes * (2 ** k))))
        # cap the finest frequency so a lobe spans a few blocks on small
        # grids; on tiny islands the fine octaves fall out entirely
        cells = min(cells, max(3, size // 4))
        p = value_noise(u, v, cells, 1, seed + 7919 * k, periodic_u=True)
        p = p - p.mean()
        peak = float(np.abs(p).max())
        if peak > 0.0:
            p = p / peak
        prof += weight * p
        weight *= 0.5
    peak = float(np.abs(prof).max())
    return prof / peak if peak > 0.0 else prof


def _cap_blend(r: float, R: float) -> float:
    """1.0 in the flat cap, 0.0 deep in the body, smooth in between."""
    t = (r - 0.8 * R) / (0.2 * R)
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _shoulder_fits(rr: np.ndarray, center: np.ndarray, c: int,
                   xi: int, zi: int, base_w: float) -> bool:
    """Whether a disc of radius base_w around grid point (xi, zi) sits fully
    inside the row boundary rr (measured around the row's center) with a
    0.5-block margin.

    Checked at the point itself and at the disc's two tangent extreme
    points (the boundary wobbles around the circumference, so a single
    direction is not enough).
    """
    cx, cz = c + center[0], c + center[1]
    d0 = float(np.hypot(xi - cx, zi - cz))
    if d0 + base_w + 0.5 > float(rr[xi, zi]):
        return False
    th = float(np.arctan2(zi - cz, xi - cx))
    tx_, tz_ = -np.sin(th), np.cos(th)
    for sgn in (-1.0, 1.0):
        qx = int(round(xi + sgn * base_w * tx_))
        qz = int(round(zi + sgn * base_w * tz_))
        qx = min(max(qx, 0), rr.shape[0] - 1)
        qz = min(max(qz, 0), rr.shape[1] - 1)
        d = float(np.hypot(qx - cx, qz - cz))
        if d + 0.25 > float(rr[qx, qz]):
            return False
    return True


def generate_island(p: IslandParams) -> tuple[Structure, int]:
    """Build the island (lobed cap, lumpy asymmetric body, icicle fringe)
    as a Structure of dirt + grass. Returns the structure and the number
    of icicles actually placed."""
    atlas = Atlas()
    dirt = atlas.add("dirt")
    grass = atlas.add("grass")

    R = 0.5 * p.diameter
    H = p.height
    # body bottom row: below it the underside is the icicle fringe
    N = int(round(H * p.neck_frac))
    N = max(1, min(N, H - 3))

    # neck: a small disk, but wide enough to shoulder the central icicles
    neck_r = max(R * p.neck_width, p.icicle_width * 0.9 + 0.6)

    # grid: room for the wobbled edge, the off-center tip/fringe and the
    # icicle bases
    amp_max = max(p.cap_amp, p.body_amp)
    size = int(np.ceil(2 * R * (1.5 + amp_max + p.asym) + 4)) + 2 * p.margin
    if size % 2 == 0:
        size += 1
    c = size // 2

    data = np.zeros((size, H, size), dtype=np.int16)
    coords = np.arange(size)

    # two independent angular noise fields: one for the lobed cap rim, one
    # for the lumpy body
    prof_cap = angular_profile(size, c, p.cap_lobes, 3, p.noise_seed)
    prof_body = angular_profile(size, c, p.body_lobes, 3, p.noise_seed + 999)
    ang = np.arctan2(coords[None, :] - c, coords[:, None] - c)
    cos_asym = np.cos(ang - p.asym_dir)

    # -- body radius profile: neck (row N) up to the cap (row H-1) --------
    # sin is flat at pi/2, so the top stays at full radius over the cap
    # rows; at the neck it levels off onto a small disk (2 rows wide).
    t = np.clip((np.arange(H) - N) / (H - 1 - N), 0.0, 1.0)
    radii = neck_r + (R - neck_r) * np.sin(0.5 * np.pi * np.power(t, p.taper))
    radii[N + 1] = radii[N]

    # disk center: island center at the cap, drifting to the tip target at
    # the neck. Smoothstep (zero slope at the top) so the cap rows do not
    # shear relative to each other.
    tx, tz = p.tip_target
    s = np.clip((H - 1 - np.arange(H)) / (H - 1 - N), 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)
    centers = np.stack([tx * s, tz * s], axis=1)

    rr_cache: dict[int, np.ndarray] = {}

    def rr(y: int) -> np.ndarray:
        """Per-cell boundary radius of body row y (2D, in blocks)."""
        if y not in rr_cache:
            w = _cap_blend(radii[y], R)
            amp = p.body_amp + (p.cap_amp - p.body_amp) * w
            prof = prof_body + w * (prof_cap - prof_body)
            rr_cache[y] = radii[y] * (1.0 + amp * prof + p.asym * cos_asym)
        return rr_cache[y]

    # -- body rows ----------------------------------------------------------
    # The wobble and the center drift can make a lower row locally wider
    # than the row above it (the row center can move faster than the
    # radius grows). Such a bulge paints a stray grass shelf, so every
    # row's boundary is clamped to sit inside the row above (the triangle
    # inequality makes S'_y subset of S'_{y+1} for every y):
    #
    #     b'[y](p) = min( b[y](p),  b'[y+1](p) - |c[y+1] - c[y]| )
    #
    # The raw boundary is kept where it already fits inside the row above;
    # only cells where the drift outruns the taper get pulled in, by at
    # most the per-row center shift.
    cent = centers + c
    clamped: dict[int, np.ndarray] = {H - 1: rr(H - 1)}
    for y in range(H - 2, N - 1, -1):
        dc = float(np.hypot(cent[y + 1, 0] - cent[y, 0],
                            cent[y + 1, 1] - cent[y, 1]))
        clamped[y] = np.minimum(rr(y), clamped[y + 1] - dc)

    for y in range(N, H):
        cx, cz = cent[y, 0], cent[y, 1]
        d2 = (coords[:, None] - cx) ** 2 + (coords[None, :] - cz) ** 2
        mask = d2 <= clamped[y] ** 2
        if not np.any(mask):
            # the (tiny) neck row can fit no cell at its (possibly
            # non-integer) center: borrow the closest cell of the row
            # above so the neck stays connected and inside it
            mask = np.zeros((size, size), dtype=bool)
            if y > N:
                d_prev = ((coords[:, None] - (c + centers[y + 1, 0])) ** 2
                          + (coords[None, :] - (c + centers[y + 1, 1])) ** 2)
                d_prev = np.where(clamped[y + 1] ** 2 > d_prev,
                                  d_prev, np.inf)
                mask[np.unravel_index(np.argmin(d_prev), d_prev.shape)] = True
            else:
                mask[int(round(cx)), int(round(cz))] = True
        data[:, y, :][mask] = dirt

    # Minecraft rule: dirt with no block directly above becomes grass.
    # The body is monotone (every row is inside the row above), so its
    # only exposed surface is the top row -- this paints the grass cap.
    # It runs BEFORE the fringe is drawn: icicles are pure dirt, and an
    # eroded tip or a bent column could otherwise expose a cell that
    # would get grass-painted.
    # NOTE: must be a real bool array -- an int mask would be interpreted as
    # integer indexing, not a boolean mask.
    has_above = np.zeros((size, H, size), dtype=bool)
    has_above[:, : H - 1, :] = data[:, 1:, :] != 0
    data[(data == dirt) & ~has_above] = grass

    # -- icicle fringe ------------------------------------------------------
    placed = 0
    if p.icicle_count > 0:
        irng = np.random.default_rng(p.noise_seed + 777)
        foot = p.icicle_foot * R
        Lmax = p.icicle_len_frac * H
        # icicles grow gently with island size (reference look: thin even
        # on big islands, chunkier on small ones)
        scale_w = 1.0 + 0.3 * (R / 8.0 - 1.0)
        txg, tzg = c + tx, c + tz

        tries = 0
        while placed < p.icicle_count and tries < p.icicle_count * 25:
            tries += 1
            # position: uniform in a disc of radius `foot` around the tip
            # target -- the fringe clusters around where the body points
            pa = float(irng.uniform(0.0, 2.0 * np.pi))
            pd = foot * float(np.sqrt(irng.uniform(0.0, 1.0)))
            px = txg + pd * np.cos(pa)
            pz = tzg + pd * np.sin(pa)
            xi, zi = int(round(px)), int(round(pz))
            if not (0 <= xi < size and 0 <= zi < size):
                continue
            base_w = max(0.6, min(3.0,
                                  p.icicle_width * scale_w
                                  * float(irng.uniform(0.7, 1.3))))
            # the underside of this column: the lowest body row whose
            # ACTUAL (clamped) wobbled boundary still contains the column
            # with room to bury the icicle shoulder -- testing the real
            # body guarantees the buried shoulder cells are solid dirt.
            # Central columns (near the tip target) reach into the body
            # lowest, so they hang longest; outer columns meet the
            # body's side higher up, so they are shorter.
            y_min = None
            for y in range(N, H):
                if _shoulder_fits(clamped[y], centers[y], c, xi, zi, base_w):
                    y_min = y
                    break
            if y_min is None:
                continue
            y_top = y_min + 2  # shoulder rows buried 2 rows into the body
            if y_top + 1 >= H - 1:
                continue
            if not _shoulder_fits(clamped[y_top + 1], centers[y_top + 1], c,
                                  xi, zi, base_w):
                continue
            # length: the tip row is how far down the fringe this icicle
            # hangs -- central columns (near the tip target) reach the
            # lowest row, outer ones end higher up, so the fringe reads
            # as a shallow bowl with the lowest point near the tip. The
            # result is clamped to the max icicle length.
            rho = float(np.hypot(px - txg, pz - tzg))
            tip_row = int(round(min(N * (rho / foot) ** 1.3
                                    * float(irng.uniform(0.4, 0.9)),
                                   max(0.0, y_top - 3))))
            L = min(y_top - tip_row, int(round(Lmax)))
            if L < 3:
                continue
            # slight bend: the column drifts up to ~1.2 blocks over 10 rows
            dm = float(irng.uniform(0.0, 1.2)) * L / 10.0
            da = float(irng.uniform(0.0, 2.0 * np.pi))
            ddx, ddz = dm * np.cos(da), dm * np.sin(da)
            for k in range(L + 2):
                row = y_top + 1 - k
                f = (k / (L + 2)) ** 1.5
                wx = px + f * ddx
                wz = pz + f * ddz
                # strictly narrowing towards a single-voxel tip (keeps the
                # grass rule off the icicles); per-row jitter = erosion
                w = base_w * (1.0 - k / (L + 2)) ** 1.5
                w = max(0.5, w * float(irng.uniform(0.92, 1.0)))
                d2 = (coords[:, None] - wx) ** 2 + (coords[None, :] - wz) ** 2
                m = d2 <= w * w
                # always include the row's center cell: at the tip the disc
                # is a single voxel, and a sub-voxel center can otherwise
                # cover no cell at all (a hole in the column -> a floating
                # speck). Consecutive center cells are at most 1 apart (the
                # bend drifts < 0.2 blocks per row), so the column is
                # connected top to bottom.
                m[int(round(wx)), int(round(wz))] = True
                data[:, row, :][m] = dirt
            placed += 1

    return Structure.from_data(data, atlas), placed


def count_floating(data: np.ndarray) -> int:
    """Number of solid voxels not 26-connected to the top row.

    A healthy island is one connected mass; anything that fails this is a
    floating speck (a disconnected voxel or cluster).
    """
    xs, ys, zs = np.nonzero(data != 0)
    if xs.size == 0:
        return 0
    cells = set(zip(xs.tolist(), ys.tolist(), zs.tolist()))
    top = int(ys.max())
    seed = set((x, top, z) for (x, y, z) in cells if y == top)
    if not seed:
        return int(xs.size)
    seen = set(seed)
    q = deque(seed)
    while q:
        x, y, z = q.popleft()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    n = (x + dx, y + dy, z + dz)
                    if n in cells and n not in seen:
                        seen.add(n)
                        q.append(n)
    return len(cells) - len(seen)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--diameter", type=int, nargs="+", default=None,
                    metavar="D",
                    help="cap width(s) in blocks; one island per value "
                         "(default: 16 40 80, paired with the default "
                         "heights 16 32 60)")
    ap.add_argument("--height", type=int, nargs="+", default=None,
                    help="total height(s) in blocks, cap to lowest icicle. "
                         "One value applies to every size; a list is paired "
                         "with --diameter. If omitted together with "
                         "--diameter, the default pairs (16,16) (40,32) "
                         "(80,60) are used; otherwise each island uses "
                         "height ~= 1.2 * diameter so proportions are kept "
                         "across sizes.")
    ap.add_argument("--taper", type=float, default=None,
                    help="body taper character; higher = slimmer body and "
                         "a longer, thinner neck (default: random per "
                         "island in 1.3..2.0)")
    ap.add_argument("--cap-noise", type=float, default=None, dest="cap_noise",
                    help="cap rim lobing amplitude as a fraction of the "
                         "radius (default: random per island in 0.15..0.28; "
                         "0 = a circular rim)")
    ap.add_argument("--cap-lobes", type=int, default=None, dest="cap_lobes",
                    help="number of lobes in the cap rim noise (default: "
                         "random per island in 3..7)")
    ap.add_argument("--body-noise", type=float, default=None, dest="body_noise",
                    help="body lumpiness amplitude as a fraction of the "
                         "radius (default: random per island in 0.10..0.25)")
    ap.add_argument("--body-lobes", type=int, default=None, dest="body_lobes",
                    help="number of lobes in the body noise (default: "
                         "random per island in 4..9)")
    ap.add_argument("--asym", type=float, default=None,
                    help="directional asymmetry: one side narrows faster, "
                         "as a fraction of the radius (default: random per "
                         "island in 0.10..0.45; 0 = symmetric)")
    ap.add_argument("--tip-frac", type=float, default=None, dest="tip_frac",
                    help="max distance of the tip target from the center, "
                         "as a fraction of the island radius, 0..1 "
                         "(default: random per island in 0.2..0.55; 0 = "
                         "perfectly centered)")
    ap.add_argument("--neck-frac", type=float, default=None, dest="neck_frac",
                    help="bottom fraction of the height that is the icicle "
                         "fringe, 0.1..0.7 (default: random per island in "
                         "0.3..0.5)")
    ap.add_argument("--neck-width", type=float, default=None, dest="neck_width",
                    help="neck radius as a fraction of the island radius, "
                         "0.05..0.35 (default: random per island in "
                         "0.10..0.20)")
    ap.add_argument("--icicle-count", type=int, default=None, dest="icicle_count",
                    help="number of icicles in the fringe; default = "
                         "auto-scale by disc area (d=16 -> 6, d=40 -> 19, "
                         "d=80 -> 77); 0 = none")
    ap.add_argument("--icicle-length", type=float, default=None, dest="icicle_len",
                    help="max icicle length as a fraction of the island "
                         "height (default: random per island in 0.45..0.65)")
    ap.add_argument("--icicle-width", type=float, default=None, dest="icicle_width",
                    help="typical icicle base radius in blocks; each icicle "
                         "varies by -30%%/+30%% and grows gently with the "
                         "island size (default: 1.0)")
    ap.add_argument("--icicle-footprint", type=float, default=None, dest="icicle_foot",
                    help="fringe spread radius around the tip target as a "
                         "fraction of the island radius (default: random "
                         "per island in 0.35..0.6)")
    ap.add_argument("--margin", type=int, default=2,
                    help="air padding around the model (default: 2)")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed; makes runs reproducible and each "
                         "island rolls its own shape from it "
                         "(default: unseeded)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "out", dest="out_dir",
                    help="directory for outputs (default: islands/out)")
    ap.add_argument("--screenshot", type=Path, default=None,
                    help="combined screenshot path (default: <out-dir>/island.png)")
    ap.add_argument("--no-screenshot", action="store_true",
                    help="skip rendering the screenshot")
    ap.add_argument("--title", type=str, default=None, help="sheet title")
    args = ap.parse_args()

    if args.tip_frac is not None and not 0.0 <= args.tip_frac <= 1.0:
        ap.error("--tip-frac must be in 0..1")
    if args.asym is not None and args.asym < 0.0:
        ap.error("--asym must be >= 0")
    if args.neck_frac is not None and not 0.1 <= args.neck_frac <= 0.7:
        ap.error("--neck-frac must be in 0.1..0.7")
    if args.neck_width is not None and not 0.05 <= args.neck_width <= 0.35:
        ap.error("--neck-width must be in 0.05..0.35")
    if args.icicle_count is not None and args.icicle_count < 0:
        ap.error("--icicle-count must be >= 0")
    if args.icicle_width is not None and args.icicle_width <= 0.0:
        ap.error("--icicle-width must be > 0")

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

    o = dict(
        taper=args.taper,
        cap_amp=args.cap_noise,
        cap_lobes=args.cap_lobes,
        body_amp=args.body_noise,
        body_lobes=args.body_lobes,
        asym=args.asym,
        tip_frac=args.tip_frac,
        neck_frac=args.neck_frac,
        neck_width=args.neck_width,
        icicle_count=args.icicle_count,
        icicle_len=args.icicle_len,
        icicle_width=args.icicle_width,
        icicle_foot=args.icicle_foot,
        margin=args.margin,
    )

    structures, labels = [], []
    for i, diameter in enumerate(diameters):
        height = heights[0] if len(heights) == 1 else heights[i]
        # each island rolls its own parameter set (and its own noise) from
        # a per-island rng, so the batch varies like the reference image
        rng = np.random.default_rng(
            (args.seed * 1000003 + i) if args.seed is not None else None)
        params = sample_params(rng, diameter, height, o)
        structure, n_icicles = generate_island(params)
        floating = count_floating(structure.data)
        npz_path = args.out_dir / f"island_d{diameter}.npz"
        structure.save(npz_path)

        stats = structure.stats()
        # where is the lowest point? (the frayed tip of the fringe)
        xs, ys, zs = np.nonzero(structure.data)
        y_low = int(ys.min())
        sel = ys == y_low
        c = structure.shape[0] // 2
        tip = (float(xs[sel].mean()) - c, float(zs[sel].mean()) - c)
        tip_off = float(np.hypot(*params.tip_target))
        print(f"d={diameter:<3} h={height:<3} -> {npz_path}   "
              f"blocks={sum(stats.values())} "
              f"(grass={stats['grass']}, dirt={stats['dirt']})   "
              f"taper={params.taper:.2f}  asym={params.asym:.2f}   "
              f"icicles={n_icicles}/{params.icicle_count}   "
              f"floating={floating}   "
              f"tip target off-center {tip_off:.1f}  "
              f"lowest@({tip[0]:.1f}, {tip[1]:.1f}) row {y_low}")

        structures.append(structure)
        labels.append(f"d={diameter} h={height}")

    if not args.no_screenshot:
        sheet = args.screenshot or args.out_dir / "island.png"
        title = args.title or "floating island -- lobed cap, lumpy body, icicle fringe"
        render_sheet(structures, sheet, labels=labels, title=title)
        print(f"wrote {sheet}")


if __name__ == "__main__":
    main()
