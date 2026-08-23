"""Seeded value noise for procedural generation.

Noise is defined on a small integer lattice of random values and sampled
with a smoothstep-interpolated kernel. Callers sample it in *normalized*
coordinates (0..1) relative to the object being generated, so feature sizes
stay proportional to the object: the same noise call gives the same relative
jaggedness whether the island is 16 or 160 blocks across.
"""

from __future__ import annotations

import numpy as np


def value_noise(
    u: np.ndarray,
    v: np.ndarray,
    cells_u: int,
    cells_v: int,
    seed: int,
    periodic_u: bool = False,
) -> np.ndarray:
    """Sample 2D value noise at points (u, v), both in [0, 1).

    u, v: numeric arrays of identical shape.
    cells_u, cells_v: number of noise cells along each axis. This is the
        feature frequency *in normalized space*: with cells_u=7, a noise
        feature spans 1/7 of the u range regardless of object size.
    seed: RNG seed. Use a different seed per field so fields are independent.
    periodic_u: wrap around in u (use for angles so there is no seam at 0/2pi).

    Returns: array shaped like u with values in [-1, 1].
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")
    if cells_u < 1 or cells_v < 1:
        raise ValueError("cell counts must be >= 1")

    rng = np.random.default_rng(seed)
    if periodic_u:
        cu = cells_u
        lat = rng.random((cu, cells_v + 1))
    else:
        cu = cells_u
        lat = rng.random((cu + 1, cells_v + 1))

    gu = u * cu
    i0 = np.floor(gu).astype(np.int64)
    fu = gu - i0
    if periodic_u:
        i0 = i0 % cu
        i1 = (i0 + 1) % cu
    else:
        i0 = np.clip(i0, 0, cu - 1)
        i1 = i0 + 1

    gv = v * cells_v
    j0 = np.clip(np.floor(gv).astype(np.int64), 0, cells_v - 1)
    fv = gv - np.floor(gv)
    j1 = j0 + 1

    su = fu * fu * (3.0 - 2.0 * fu)
    sv = fv * fv * (3.0 - 2.0 * fv)

    a = lat[i0, j0]
    b = lat[i0, j1]
    c = lat[i1, j0]
    d = lat[i1, j1]
    top = a + (b - a) * su
    bot = c + (d - c) * su
    out = top + (bot - top) * sv
    return 2.0 * out - 1.0


def angular_noise(angle: np.ndarray, cells: int, seed: int, phase: float = 0.5) -> np.ndarray:
    """1D periodic value noise around a full circle (angle in radians).

    Seams-free: the field wraps at 0/2pi.
    """
    u = np.mod(angle, 2.0 * np.pi) / (2.0 * np.pi)
    return value_noise(u, np.full_like(u, phase), cells, 1, seed, periodic_u=True)
