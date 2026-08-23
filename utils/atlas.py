"""Block atlas: shared legend between integer voxel indices and block names.

A structure stores its voxels as a 3D numpy array of integers; the atlas
tells you what each integer means. Index 0 is reserved for air.
"""

from __future__ import annotations

from typing import Iterator


class Atlas:
    """Bidirectional mapping between block names and integer indices."""

    AIR_INDEX = 0

    def __init__(self) -> None:
        self._name_to_index: dict[str, int] = {}
        self._index_to_name: dict[int, str] = {}

    def add(self, name: str, index: int | None = None) -> int:
        """Register a block name and return its integer index.

        Existing names are returned as-is. If no index is given, the next
        free index above air is used.
        """
        if name in self._name_to_index:
            return self._name_to_index[name]
        if index is None:
            index = max(self._index_to_name, default=self.AIR_INDEX) + 1
        if index == self.AIR_INDEX:
            raise ValueError("index 0 is reserved for air")
        if index in self._index_to_name:
            raise ValueError(f"index {index} is already taken by {self._index_to_name[index]!r}")
        self._name_to_index[name] = index
        self._index_to_name[index] = name
        return index

    def resolve(self, block: str | int) -> int:
        """Return the integer index for a block name or integer index."""
        if isinstance(block, str):
            if block not in self._name_to_index:
                raise KeyError(f"unknown block {block!r}; known blocks: {sorted(self._name_to_index)}")
            return self._name_to_index[block]
        index = int(block)
        if index == self.AIR_INDEX or index in self._index_to_name:
            return index
        raise ValueError(f"block index {index} not in atlas")

    def index_to_name(self, index: int) -> str:
        if index == self.AIR_INDEX:
            return "air"
        return self._index_to_name[index]

    def __getitem__(self, name: str) -> int:
        return self._name_to_index[name]

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_index

    def __len__(self) -> int:
        return len(self._name_to_index)

    def names(self) -> Iterator[str]:
        """Yield all registered block names in index order."""
        return (self._index_to_name[i] for i in sorted(self._index_to_name))

    def to_dict(self) -> dict[str, int]:
        """Serialize as {block_name: index}."""
        return {name: self._name_to_index[name] for name in self.names()}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "Atlas":
        atlas = cls()
        for name, index in data.items():
            atlas.add(name, index=index)
        return atlas
