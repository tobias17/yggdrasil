"""Generate a floating island: a lobed grass cap over a thin, bowl-shaped
rock shell that droops at the bottom, hung with a few large stalactite
spikes whose tips trace the bowl.

Shape (target silhouette: islands/out/reference.jpg):
  * The cap -- the top is a roughly circular, flat plateau whose rim is
    lobed and notched (angular value noise) instead of a perfect circle.
  * The shell -- below the cap the rock is a fairly thin slab whose
    underside is a shallow concave dish: the bottom is a little deeper
    toward the (off-center) droop point and rises to a short skirt at
    the rim. The sides lean gently inward as they descend.
  * The droop -- the island's lowest point is a single off-center spot
    (within about half the radius). The spike tips form a bowl-shaped
    envelope around it: the longest spike hangs from the droop point and
    reaches the lowest row, and the outer spikes are progressively
    shorter, so the underside silhouette is a broad, shallow, lopsided
    concave bowl.
  * The spikes -- a handful of large stalactites hang from the shell
    underside: fewer and more spaced apart than a fringe of icicles.
    Each is a fat cone, several blocks across at the base, tapering to a
    single-voxel tip, with a slight bend. The central spike is the
    fattest and longest; the others sit well apart and follow the bowl
    with their tips.
  * Per-island variety -- unless a flag pins a value, every island
    samples its own parameter set (shell depth, lumpiness, asymmetry,
    droop depth, spike count, width, spacing), so a batch of islands
    varies like the reference: deeper droops, shallower ones, lopsided
    ones, stumpy-spike and long-spike ones.

Block layout:
  * Every solid voxel is created as dirt.
  * The cap is painted before the spikes are drawn: any dirt block with
    no block directly above it becomes grass (the Minecraft surface
    rule). Every body column is solid all the way up to the top row, so
    this paints exactly the cap -- the spikes stay pure dirt.

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
RANGE_CAP_AMP = (0.12, 0.20)      # cap rim lobing, fraction of the radius
LOBES_CAP = (3, 7)                # cap lobe count range (inclusive)
RANGE_BODY_AMP = (0.05, 0.12)     # bowl lumpiness, fraction of R (low freq)
BODY_CELLS = (3, 6)               # lumpiness noise frequency (cells across)
RANGE_ASYM = (0.05, 0.15)         # directional asymmetry, fraction of R
RANGE_SHELL_CENTER = (0.20, 0.30)  # shell slab depth at the bowl center, frac of H
RANGE_SHELL_SKIRT = (0.08, 0.14)   # shell slab depth at the rim, frac of H
RANGE_SHELL_EXP = (1.5, 2.5)       # flatness of the shell dish bottom
RANGE_TAPER_AMT = (0.05, 0.15)     # inward lean of the sides, fraction of R
RANGE_DROOP = (0.55, 0.72)         # longest spike depth (lowest point), frac of H
RANGE_SKIRT = (0.25, 0.40)         # outer spike-tip depth, frac of H
RANGE_BOWL_EXP = (1.5, 2.5)        # flatness of the spike-tip bowl envelope
RANGE_TIP_FRAC = (0.25, 0.50)      # droop point (lowest point) within this frac of R
RANGE_SPIKE_WIDTH = (0.080, 0.120) # typical spike base radius, fraction of R
RANGE_SPIKE_SPACING = (0.30, 0.45)  # min spike spacing, fraction of R
RANGE_SPIKE_BEND = (0.1, 0.5)     # max spike bend drift, blocks


def auto_spike_count(diameter: int, rng: np.random.Generator) -> int:
    """Default spike count: a handful, growing gently with size
    (d=16 -> 3, d=40 -> ~4-5, d=80 -> ~8-9)."""
    return max(3, int(round(0.11 * diameter * float(rng.uniform(0.9, 1.1)))))


@dataclass(frozen=True)
class IslandParams:
    """A fully resolved parameter set for one island."""

    diameter: int          # cap width in blocks (the island's max width)
    height: int            # total height in blocks, cap to lowest spike tip
    cap_amp: float         # cap rim lobing amplitude (fraction of R)
    cap_lobes: int         # number of lobes in the cap rim noise
    body_amp: float        # bowl lumpiness (fraction of R)
    body_cells: int        # lumpiness noise frequency (cells across)
    asym: float            # directional asymmetry (fraction of R)
    asym_dir: float        # direction of the deeper/wider side (radians)
    shell_center: float    # shell slab depth at the bowl center (fraction of H)
    shell_skirt: float     # shell slab depth at the rim (fraction of H)
    shell_exp: float       # flatness of the shell dish bottom
    taper_amt: float       # inward lean of the sides (fraction of R)
    droop: float           # longest spike depth (fraction of H)
    skirt: float           # outer spike-tip depth (fraction of H)
    bowl_exp: float        # flatness of the spike-tip bowl envelope
    tip: tuple             # (dx, dz) of the droop point, blocks
    spike_count: int       # number of spikes (0 = none)
    spike_w: float         # typical spike base radius, blocks
    min_spacing: float     # minimum spike spacing, blocks
    spike_bend: float      # max spike bend drift, blocks
    margin: int            # air padding around the model in blocks
    noise_seed: int        # seed for all of this island's noise


def sample_params(rng: np.random.Generator, diameter: int, height: int,
                  o: dict) -> IslandParams:
    """Resolve an island's parameter set.

    Every entry of ``o`` that is not None is used verbatim (a pinned flag);
    everything else is sampled from its range with the island's rng, so
    different islands roll different shapes from the same code.
    """
    R = 0.5 * diameter
    cap_amp = o["cap_amp"] if o["cap_amp"] is not None else float(rng.uniform(*RANGE_CAP_AMP))
    cap_lobes = int(o["cap_lobes"]) if o["cap_lobes"] is not None else int(rng.integers(*LOBES_CAP))
    body_amp = o["body_amp"] if o["body_amp"] is not None else float(rng.uniform(*RANGE_BODY_AMP))
    body_cells = int(o["body_cells"]) if o["body_cells"] is not None else int(rng.integers(*BODY_CELLS))
    asym = o["asym"] if o["asym"] is not None else float(rng.uniform(*RANGE_ASYM))
    asym_dir = float(rng.uniform(0.0, 2.0 * np.pi))
    shell_center = o["shell_center"] if o["shell_center"] is not None else float(rng.uniform(*RANGE_SHELL_CENTER))
    shell_skirt = o["shell_skirt"] if o["shell_skirt"] is not None else float(rng.uniform(*RANGE_SHELL_SKIRT))
    shell_exp = o["shell_exp"] if o["shell_exp"] is not None else float(rng.uniform(*RANGE_SHELL_EXP))
    taper_amt = o["taper_amt"] if o["taper_amt"] is not None else float(rng.uniform(*RANGE_TAPER_AMT))
    droop = o["droop"] if o["droop"] is not None else float(rng.uniform(*RANGE_DROOP))
    skirt = o["skirt"] if o["skirt"] is not None else float(rng.uniform(*RANGE_SKIRT))
    bowl_exp = o["bowl_exp"] if o["bowl_exp"] is not None else float(rng.uniform(*RANGE_BOWL_EXP))
    tip_frac = o["tip_frac"] if o["tip_frac"] is not None else float(rng.uniform(*RANGE_TIP_FRAC))
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    # land in the upper half of the allowed offset: the droop point is
    # deliberately visibly off-center, not "mostly centered with a little
    # jitter"
    tip_dist = R * tip_frac * float(rng.uniform(0.55, 1.0))
    tip = (tip_dist * np.cos(ang), tip_dist * np.sin(ang))
    spike_count = o["spike_count"] if o["spike_count"] is not None else auto_spike_count(diameter, rng)
    spike_w = o["spike_width"] if o["spike_width"] is not None \
        else R * float(rng.uniform(*RANGE_SPIKE_WIDTH)) * float(rng.uniform(0.85, 1.15))
    spike_w = max(1.2, spike_w)
    min_spacing = o["spacing"] if o["spacing"] is not None else R * float(rng.uniform(*RANGE_SPIKE_SPACING))
    spike_bend = float(rng.uniform(*RANGE_SPIKE_BEND))
    noise_seed = int(rng.integers(0, 2**31 - 1))
    return IslandParams(
        diameter=diameter, height=height,
        cap_amp=cap_amp, cap_lobes=cap_lobes,
        body_amp=body_amp, body_cells=body_cells,
        asym=asym, asym_dir=asym_dir,
        shell_center=shell_center, shell_skirt=shell_skirt, shell_exp=shell_exp,
        taper_amt=taper_amt,
        droop=droop, skirt=skirt, bowl_exp=bowl_exp,
        tip=tip,
        spike_count=spike_count, spike_w=spike_w,
        min_spacing=min_spacing, spike_bend=spike_bend,
        margin=o["margin"], noise_seed=noise_seed,
    )


def angular_profile(size: int, c: int, lobes: int, octaves: int,
                    seed: int) -> np.ndarray:
    """A zero-mean, peak-normalized 2D angular noise field around the grid
    center (c, c).

    Layered (fBm) periodic value noise of the angle around (c, c): the base
    octave is the coarse lobes, higher octaves add finer detail. Used for
    the cap rim so the circumference is lobed, not a perfect circle.
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


def _bump(v, lo: float, hi: float):
    """Clamp to [lo, hi]; works on scalars and arrays alike."""
    return np.clip(v, lo, hi)


def generate_island(p: IslandParams) -> tuple[Structure, int]:
    """Build the island (lobed cap, bowl shell, large spaced spikes) as a
    Structure of dirt + grass. Returns the structure and the number of
    spikes actually placed."""
    atlas = Atlas()
    dirt = atlas.add("dirt")
    grass = atlas.add("grass")

    R = 0.5 * p.diameter
    H = p.height

    # grid: room for the wobbled edge, the off-center tip and the spike
    # bases
    size = int(np.ceil(2 * R * (1.0 + p.cap_amp + p.asym) + 4)) + 2 * p.margin
    if size % 2 == 0:
        size += 1
    c = size // 2

    data = np.zeros((size, H, size), dtype=np.int16)
    coords = np.arange(size)
    X, Z = np.meshgrid(coords, coords, indexing="ij")
    ang = np.arctan2(Z - c, X - c)

    # -- cap rim: lobed, notched circumference ----------------------------
    prof_cap = angular_profile(size, c, p.cap_lobes, 3, p.noise_seed)
    edge = R * (1.0 + p.cap_amp * prof_cap + p.asym * np.cos(ang - p.asym_dir))
    d2c = (X - c) ** 2 + (Z - c) ** 2
    re = np.sqrt(d2c) / edge  # radius over local edge (1 = at the rim)

    # -- the bowl: one radial field around the droop point ----------------
    # s: 0 at the droop point, ~1 at the far rim (normalized by the max
    # droop-to-rim distance so the rim always reaches its shallow end).
    tx, tz = p.tip
    tip_off = float(np.hypot(tx, tz))
    txg, tzg = c + tx, c + tz
    dT = np.hypot(X - txg, Z - tzg)
    s = _bump(dT / (R + tip_off), 0.0, 1.0) if (R + tip_off) > 0 else np.zeros_like(dT)

    # Low-frequency 2D noise (one octave, no fine detail: small bumps
    # would read as a stub fringe) + a directional term make the bowl
    # uneven rather than a solid of revolution. The SAME field feeds both
    # surfaces below, so the shell and the spike tips stay in the same
    # dish and the spike lengths vary smoothly around the bowl.
    un = _bump((X - c) / (2.0 * R) + 0.5, 0.0, 0.999)
    vn = _bump((Z - c) / (2.0 * R) + 0.5, 0.0, 0.999)
    lump = value_noise(un, vn, p.body_cells, p.body_cells, p.noise_seed + 991)
    lp = float(np.abs(lump).max())
    if lp > 0.0:
        lump = lump / lp
    angT = np.arctan2(Z - tzg, X - txg)
    lumps = p.body_amp * R * lump + p.asym * R * np.cos(angT - p.asym_dir) * (0.3 + 0.7 * s)

    # -- shell underside: a shallow concave slab --------------------------
    # Shell depth = how far the SOLID body hangs below the cap at each
    # column: a thin slab, a little deeper toward the droop point,
    # rising to a short skirt at the rim.
    shell_depth = (p.shell_skirt * H
                   + (p.shell_center * H - p.shell_skirt * H)
                   * (1.0 - np.power(s, p.shell_exp))
                   + lumps)
    # bottom[x,z] = the lowest solid row of the column (before the side
    # lean is applied); keep a 3-row skirt at the rim and headroom below
    bottom = (H - 1 - np.round(shell_depth)).astype(int)
    bottom = np.clip(bottom, 2, H - 3)

    # -- spike-tip bowl envelope -------------------------------------------
    # The spikes hang below the shell and their tips trace a deeper bowl:
    # deepest at the droop point (p.droop * H below the cap -- the
    # island's lowest point), rising to p.skirt * H at the rim.
    env_depth = (p.skirt * H
                 + (p.droop * H - p.skirt * H) * (1.0 - np.power(s, p.bowl_exp))
                 + lumps)

    # -- side lean: rows narrow gently as they descend --------------------
    # radius at row y = edge * scale(y); scale = 1 at the cap, taper_amt
    # smaller at the bottom row. A column at radius r is radially allowed
    # only where edge*scale >= r, i.e. above a row y_c; below y_c the side
    # has tapered past it.
    ta = min(p.taper_amt, 0.30)  # keep the side lean invertible
    u_rows = (H - 1 - np.arange(H)) / (H - 1)  # 0 at the cap, 1 at the bottom
    scale = 1.0 - ta * np.power(u_rows, 1.5)
    uc = np.power(np.clip((1.0 - re) / ta, 0.0, 1.0), 2.0 / 3.0)
    y_rad = H - 1 - uc * (H - 1)  # deepest radially allowed row
    underside = np.maximum(bottom, np.ceil(y_rad).astype(int))

    # the droop point must stay inside the body even after the lean:
    # pull the tip in so the central spike's base disc fits under the
    # expected anchor row
    u_a = float(np.clip(p.shell_center, 0.0, 1.0))
    edge_min = R * (1.0 - p.cap_amp - p.asym) * (1.0 - ta * u_a ** 1.5)
    max_tip = max(0.0, edge_min - 1.8 * p.spike_w - 1.0)
    if tip_off > max_tip and tip_off > 0.0:
        k = max_tip / tip_off
        tx, tz = tx * k, tz * k
        tip_off = max_tip
        txg, tzg = c + tx, c + tz
        dT = np.hypot(X - txg, Z - tzg)
        s = _bump(dT / (R + tip_off), 0.0, 1.0) if (R + tip_off) > 0 else np.zeros_like(dT)
        angT = np.arctan2(Z - tzg, X - txg)
        lumps = (p.body_amp * R * lump
                 + p.asym * R * np.cos(angT - p.asym_dir) * (0.3 + 0.7 * s))
        shell_depth = (p.shell_skirt * H
                       + (p.shell_center * H - p.shell_skirt * H)
                       * (1.0 - np.power(s, p.shell_exp)) + lumps)
        env_depth = (p.skirt * H
                     + (p.droop * H - p.skirt * H)
                     * (1.0 - np.power(s, p.bowl_exp)) + lumps)
        bottom = (H - 1 - np.round(shell_depth)).astype(int)
        bottom = np.clip(bottom, 2, H - 3)
        underside = np.maximum(bottom, np.ceil(y_rad).astype(int))

    # -- body rows ---------------------------------------------------------
    # A column is solid at row y iff it is inside the (leant) edge at that
    # row and at or above its shell bottom. Rows are strictly nested (the
    # scale only shrinks going down), so the mass is one connected blob
    # and the only exposed top surface is the cap row.
    for y in range(H):
        m = (d2c <= np.power(edge * scale[y], 2.0)) & (bottom <= y)
        data[:, y, :][m] = dirt

    # Minecraft rule: dirt with no block directly above becomes grass.
    # Every column is solid up to the top row, so this paints exactly the
    # cap. It runs BEFORE the spikes: they are pure dirt, and an eroded
    # tip could otherwise expose a cell that would get grass-painted.
    # NOTE: must be a real bool array -- an int mask would be interpreted
    # as integer indexing, not a boolean mask.
    has_above = np.zeros((size, H, size), dtype=bool)
    has_above[:, : H - 1, :] = data[:, 1:, :] != 0
    data[(data == dirt) & ~has_above] = grass

    # -- spikes: large, few, spaced apart ----------------------------------
    # Each spike's tip sits on the bowl envelope at its anchor position,
    # so the tips trace the drooping bowl: longest at the droop point,
    # shorter toward the rim.
    placed = 0
    if p.spike_count > 0:
        srng = np.random.default_rng(p.noise_seed + 777)
        anchors: list[tuple[int, int, bool]] = []

        # central spike: grows from the droop point and reaches the
        # lowest row of the island (the island's single lowest point)
        ix, iz = int(round(txg)), int(round(tzg))
        if not (0 <= ix < size and 0 <= iz < size) or re[ix, iz] >= 1.0:
            # (defensive: the tip clamp above should make this unreachable)
            bm = np.where(re < 1.0, underside, 10 ** 9)
            iy, iz = np.unravel_index(int(np.argmin(bm)), bm.shape)
            ix, iz = int(iy), int(iz)
        anchors.append((ix, iz, True))

        # the rest: rejection-sampled with a minimum spacing so the spikes
        # are clearly apart, and a base-fit check so each base sits fully
        # on the (leant) body at its anchor row
        tries = 0
        while len(anchors) < p.spike_count and tries < p.spike_count * 60:
            tries += 1
            pa = float(srng.uniform(0.0, 2.0 * np.pi))
            pd = 0.82 * R * float(np.sqrt(srng.uniform(0.0, 1.0)))
            x = int(round(c + pd * np.cos(pa)))
            z = int(round(c + pd * np.sin(pa)))
            if not (0 <= x < size and 0 <= z < size):
                continue
            b0 = int(underside[x, z])
            if re[x, z] >= 0.92 or b0 < 4:
                continue
            if np.hypot(x - c, z - c) + p.spike_w * 1.2 + 0.5 > \
                    edge[x, z] * scale[b0]:
                continue
            if any(np.hypot(x - ax, z - az) < p.min_spacing
                   for ax, az, _ in anchors):
                continue
            anchors.append((x, z, False))

        for x0, z0, central in anchors:
            b0 = int(underside[x0, z0])
            if central:
                # the droop spike hangs to the very bottom row: it IS the
                # island's lowest point
                tip_row = 0
            else:
                # tip row: the bowl envelope at this position (a per-spike
                # jitter keeps the tips from sitting on a perfectly smooth
                # curve), clamped so the spike hangs at least 2 rows below
                # the shell and never above row 0
                env = float(env_depth[x0, z0])
                env += float(srng.uniform(-0.03, 0.03)) * H
                tip_row = H - 1 - int(round(env))
                tip_row = max(0, min(tip_row, b0 - 2))
            L = (b0 + 1) - tip_row
            if L < 2:
                continue
            w0 = (p.spike_w * float(srng.uniform(1.25, 1.6)) if central
                  else p.spike_w * float(srng.uniform(0.85, 1.15)))
            # slight bend: the column drifts up to `spike_bend` blocks
            dm = float(srng.uniform(0.0, p.spike_bend))
            da = float(srng.uniform(0.0, 2.0 * np.pi))
            ddx, ddz = dm * np.cos(da), dm * np.sin(da)
            px, pz = float(x0), float(z0)
            for k in range(L + 2):
                row = b0 + 1 - k
                if row < 0:
                    break
                if k == 0:
                    w = w0 * 0.85  # tuck the top row into the body
                else:
                    f = (k - 1) / (L + 1)  # 0 at the anchor, 1 at the tip
                    w = 0.5 + (w0 - 0.5) * (1.0 - f) ** 1.15
                w = max(0.5, w * float(srng.uniform(0.95, 1.0)))
                fb = k / (L + 1)
                wx = px + fb * ddx
                wz = pz + fb * ddz
                d2 = (coords[:, None] - wx) ** 2 + (coords[None, :] - wz) ** 2
                m = d2 <= w * w
                # always include the row's center cell: at the tip the disc
                # is a single voxel, and a sub-voxel center can otherwise
                # cover no cell at all (a hole in the column). Consecutive
                # centers drift by far less than a block per row, so the
                # column is connected top to bottom.
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
                    help="total height(s) in blocks, cap to lowest spike "
                         "tip. One value applies to every size; a list is "
                         "paired with --diameter. If omitted together with "
                         "--diameter, the default pairs (16,16) (40,32) "
                         "(80,60) are used; otherwise each island uses "
                         "height ~= 1.2 * diameter so proportions are kept "
                         "across sizes.")
    ap.add_argument("--cap-noise", type=float, default=None, dest="cap_amp",
                    help="cap rim lobing amplitude as a fraction of the "
                         "radius (default: random per island in "
                         "0.12..0.20; 0 = a circular rim)")
    ap.add_argument("--cap-lobes", type=int, default=None, dest="cap_lobes",
                    help="number of lobes in the cap rim noise (default: "
                         "random per island in 3..7)")
    ap.add_argument("--body-noise", type=float, default=None, dest="body_amp",
                    help="bowl lumpiness amplitude as a fraction of the "
                         "radius (default: random per island in "
                         "0.05..0.12)")
    ap.add_argument("--body-lobes", type=int, default=None, dest="body_cells",
                    help="lumpiness noise frequency in cells across the "
                         "island (default: random per island in 3..6; "
                         "lower = lumpier)")
    ap.add_argument("--asym", type=float, default=None,
                    help="directional asymmetry: one side of the bowl is "
                         "deeper, as a fraction of the radius (default: "
                         "random per island in 0.05..0.15; 0 = symmetric)")
    ap.add_argument("--shell-center", type=float, default=None, dest="shell_center",
                    help="shell slab depth at the bowl center as a "
                         "fraction of the island height (default: random "
                         "per island in 0.20..0.30)")
    ap.add_argument("--shell-skirt", type=float, default=None, dest="shell_skirt",
                    help="shell slab depth at the rim (the short skirt "
                         "under the cap edge) as a fraction of the height "
                         "(default: random per island in 0.08..0.14)")
    ap.add_argument("--shell-exp", type=float, default=None, dest="shell_exp",
                    help="flatness of the shell dish bottom, >1 flatter "
                         "(default: random per island in 1.5..2.5)")
    ap.add_argument("--taper", type=float, default=None, dest="taper_amt",
                    help="inward lean of the body sides as a fraction of "
                         "the radius (default: random per island in "
                         "0.05..0.15; 0 = vertical walls)")
    ap.add_argument("--droop", type=float, default=None,
                    help="depth of the longest (central) spike -- the "
                         "island's lowest point -- as a fraction of the "
                         "height (default: random per island in "
                         "0.55..0.72)")
    ap.add_argument("--skirt", type=float, default=None,
                    help="depth of the outer spike tips as a fraction of "
                         "the height (default: random per island in "
                         "0.25..0.40)")
    ap.add_argument("--bowl-exp", type=float, default=None, dest="bowl_exp",
                    help="flatness of the spike-tip bowl envelope, >1 "
                         "flatter (default: random per island in "
                         "1.5..2.5)")
    ap.add_argument("--tip-frac", type=float, default=None, dest="tip_frac",
                    help="max distance of the droop point (the island's "
                         "lowest point) from the cap center, as a fraction "
                         "of the island radius, 0.12..1 (default: random "
                         "per island in 0.25..0.50; lower = more centered)")
    ap.add_argument("--spike-count", type=int, default=None, dest="spike_count",
                    help="number of spikes; default = auto-scale by "
                         "diameter (d=16 -> ~3, d=40 -> ~4-5, d=80 -> "
                         "~8-9); 0 = none")
    ap.add_argument("--spike-width", type=float, default=None, dest="spike_width",
                    help="typical spike base radius in blocks; each spike "
                         "varies -15%%/+20%% (default: 0.055..0.085*R "
                         "sampled per island; the central spike gets an "
                         "extra +25..+60%%)")
    ap.add_argument("--spacing", type=float, default=None,
                    help="minimum distance between spike bases as a "
                         "fraction of the island radius (default: random "
                         "per island in 0.30..0.45)")
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

    if args.tip_frac is not None and not 0.12 <= args.tip_frac <= 1.0:
        ap.error("--tip-frac must be in 0.12..1")
    if args.asym is not None and args.asym < 0.0:
        ap.error("--asym must be >= 0")
    if args.droop is not None and not 0.3 <= args.droop <= 0.9:
        ap.error("--droop must be in 0.3..0.9")
    if args.skirt is not None and not 0.15 <= args.skirt <= 0.6:
        ap.error("--skirt must be in 0.15..0.6")
    if args.shell_center is not None and not 0.1 <= args.shell_center <= 0.5:
        ap.error("--shell-center must be in 0.1..0.5")
    if args.shell_skirt is not None and not 0.03 <= args.shell_skirt <= 0.3:
        ap.error("--shell-skirt must be in 0.03..0.3")
    if args.shell_exp is not None and args.shell_exp < 0.5:
        ap.error("--shell-exp must be >= 0.5")
    if args.bowl_exp is not None and args.bowl_exp < 0.5:
        ap.error("--bowl-exp must be >= 0.5")
    if args.taper_amt is not None and not 0.0 <= args.taper_amt <= 0.3:
        ap.error("--taper must be in 0..0.3")
    if args.spike_count is not None and args.spike_count < 0:
        ap.error("--spike-count must be >= 0")
    if args.spike_width is not None and args.spike_width <= 0.0:
        ap.error("--spike-width must be > 0")
    if args.spacing is not None and args.spacing < 0.05:
        ap.error("--spacing must be >= 0.05")

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
        cap_amp=args.cap_amp,
        cap_lobes=args.cap_lobes,
        body_amp=args.body_amp,
        body_cells=args.body_cells,
        asym=args.asym,
        shell_center=args.shell_center,
        shell_skirt=args.shell_skirt,
        shell_exp=args.shell_exp,
        taper_amt=args.taper_amt,
        droop=args.droop,
        skirt=args.skirt,
        bowl_exp=args.bowl_exp,
        tip_frac=args.tip_frac,
        spike_count=args.spike_count,
        spike_width=args.spike_width,
        spacing=args.spacing,
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
        structure, n_spikes = generate_island(params)
        floating = count_floating(structure.data)
        npz_path = args.out_dir / f"island_d{diameter}.npz"
        structure.save(npz_path)

        stats = structure.stats()
        # where is the lowest point? (the central spike's tip)
        xs, ys, zs = np.nonzero(structure.data)
        y_low = int(ys.min())
        sel = ys == y_low
        c = structure.shape[0] // 2
        tip = (float(xs[sel].mean()) - c, float(zs[sel].mean()) - c)
        print(f"d={diameter:<3} h={height:<3} -> {npz_path}   "
              f"blocks={sum(stats.values())} "
              f"(grass={stats['grass']}, dirt={stats['dirt']})   "
              f"shell={params.shell_center:.2f}/{params.shell_skirt:.2f}  "
              f"droop={params.droop:.2f}  skirt={params.skirt:.2f}   "
              f"spikes={n_spikes}/{params.spike_count}   "
              f"floating={floating}   "
              f"tip off-center {float(np.hypot(*params.tip)):.1f}  "
              f"lowest@({tip[0]:.1f}, {tip[1]:.1f}) row {y_low}")

        structures.append(structure)
        labels.append(f"d={diameter} h={height}")

    if not args.no_screenshot:
        sheet = args.screenshot or args.out_dir / "island.png"
        title = args.title or "floating island -- bowl shell, drooping bottom, large spaced spikes"
        # low side angle (like the reference's camera): the drooping spike
        # envelope and the off-center low point read against the silhouette
        render_sheet(structures, sheet, labels=labels, title=title,
                     elev=10.0, azim=-60.0)
        print(f"wrote {sheet}")


if __name__ == "__main__":
    main()
