"""Structure: the project's canonical 3D model.

A structure is a 3D numpy array of integer block indices (0 = air) plus
the Atlas that names each index. Array axes are (X, Y, Z) with Y pointing
up.

On disk, a structure is a single ``.npz`` file containing:
  - ``data``  : the integer voxel array (int16)
  - ``atlas`` : JSON legend mapping block name -> index
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .atlas import Atlas


class Structure:
    """A 3D voxel grid of block indices and its block atlas."""

    def __init__(self, shape, atlas: Atlas | None = None, dtype=np.int16) -> None:
        self.data = np.zeros(tuple(shape), dtype=dtype)
        self.atlas = atlas if atlas is not None else Atlas()

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_data(cls, data: np.ndarray, atlas: Atlas) -> "Structure":
        """Wrap an existing (X, Y, Z) integer voxel array."""
        structure = cls(data.shape, atlas=atlas)
        structure.data = np.ascontiguousarray(data, dtype=structure.data.dtype)
        return structure

    # -- block access -----------------------------------------------------

    def set(self, x: int, y: int, z: int, block: str | int) -> None:
        """Set a single voxel."""
        self.data[x, y, z] = self.atlas.resolve(block)

    def get(self, x: int, y: int, z: int) -> int:
        """Return the block index at (x, y, z)."""
        return int(self.data[x, y, z])

    def block_at(self, x: int, y: int, z: int) -> str:
        """Return the block name at (x, y, z)."""
        return self.atlas.index_to_name(int(self.data[x, y, z]))

    def set_layer(self, block: str | int, y: int, mask_xz: np.ndarray) -> None:
        """Set voxels of one horizontal layer (height ``y``) from a 2D (X, Z) boolean mask."""
        self.data[:, y, :][mask_xz] = self.atlas.resolve(block)

    def set_where(self, mask: np.ndarray, block: str | int) -> None:
        """Set every voxel where the boolean 3D mask (X, Y, Z) is True."""
        self.data[mask] = self.atlas.resolve(block)

    # -- inspection ---------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def stats(self) -> dict[str, int]:
        """Voxel count per block name (air excluded)."""
        return {name: int((self.data == self.atlas[name]).sum()) for name in self.atlas.names()}

    def bounds(self) -> dict[str, tuple[int, int, int]] | None:
        """Min/max (x, y, z) of non-air voxels, or None if empty."""
        coords = np.argwhere(self.data != 0)
        if coords.size == 0:
            return None
        lo = coords.min(axis=0).astype(int)
        hi = coords.max(axis=0).astype(int)
        return {"min": tuple(lo), "max": tuple(hi)}

    # -- persistence ----------------------------------------------------------

    def save(self, path) -> Path:
        """Save as a single .npz file (data + JSON atlas). Returns the path."""
        path = Path(path)
        if not path.name.endswith(".npz"):
            path = path.with_name(path.name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            data=self.data,
            atlas=np.array(json.dumps(self.atlas.to_dict()), dtype=np.str_),
        )
        return path

    @classmethod
    def load(cls, path) -> "Structure":
        """Load a structure saved by :meth:`save`."""
        with np.load(path) as npz:
            data = npz["data"]
            atlas = Atlas.from_dict(json.loads(str(npz["atlas"])))
        structure = cls(data.shape, atlas=atlas)
        structure.data = data.astype(structure.data.dtype, copy=False)
        return structure
