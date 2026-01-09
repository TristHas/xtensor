from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping as MappingABC, Iterator
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

import numpy as np
import torch

from .indexes import BaseIndex, CoordArray


def _clone_index(index: BaseIndex) -> BaseIndex:
    return index.clone()


def _clone_extra(values: CoordArray) -> CoordArray:
    if isinstance(values, torch.Tensor):
        return values.clone()
    if pd is not None and isinstance(values, pd.Index):
        return values.copy()
    return tuple(values)


def _normalize_extra(values: Any) -> CoordArray:
    if isinstance(values, torch.Tensor):
        return values.clone()
    if pd is not None and isinstance(values, pd.Index):
        return values.copy()
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return (values.item(),)
        return tuple(values.tolist())
    if isinstance(values, (list, tuple)):
        return tuple(values)
    return (values,)


class Coordinates:
    """Container tracking dimension indexes and auxiliary coordinates."""

    def __init__(
        self,
        dim_indexes: Mapping[str, BaseIndex],
        *,
        extra_coords: Optional[Mapping[str, CoordArray]] = None,
        copy: bool = True,
    ) -> None:
        if copy:
            self._dim_indexes: MutableMapping[str, BaseIndex] = OrderedDict(
                (dim, _clone_index(index)) for dim, index in dim_indexes.items()
            )
            self._extra_coords: MutableMapping[str, CoordArray] = OrderedDict(
                (name, _clone_extra(values)) for name, values in (extra_coords or {}).items()
            )
        else:
            self._dim_indexes = OrderedDict(dim_indexes)
            self._extra_coords = OrderedDict(extra_coords or {})

    def copy(self) -> "Coordinates":
        return Coordinates(self._dim_indexes, extra_coords=self._extra_coords, copy=True)

    def dim_names(self) -> Tuple[str, ...]:
        return tuple(self._dim_indexes.keys())

    def dim_indexes(self) -> Mapping[str, BaseIndex]:
        return OrderedDict((dim, _clone_index(index)) for dim, index in self._dim_indexes.items())

    def dim_index(self, name: str) -> BaseIndex:
        return _clone_index(self._dim_indexes[name])

    def dim_sizes(self) -> Mapping[str, int]:
        return {dim: len(index) for dim, index in self._dim_indexes.items()}

    def extra_items(self) -> Mapping[str, CoordArray]:
        return OrderedDict((name, _clone_extra(values)) for name, values in self._extra_coords.items())

    def coord_values(self, name: str) -> CoordArray:
        if name in self._dim_indexes:
            return self._dim_indexes[name].coord_array()
        if name in self._extra_coords:
            return _clone_extra(self._extra_coords[name])
        raise KeyError(name)

    def has_dim(self, name: str) -> bool:
        return name in self._dim_indexes

    def has_coord(self, name: str) -> bool:
        return name in self._dim_indexes or name in self._extra_coords

    def to_dict(self) -> Dict[str, CoordArray]:
        data: Dict[str, CoordArray] = {}
        for dim, index in self._dim_indexes.items():
            data[dim] = index.coord_array()
        for name, values in self._extra_coords.items():
            data[name] = _clone_extra(values)
        return data

    def replace(
        self,
        *,
        dim_indexes: Optional[Mapping[str, BaseIndex]] = None,
        extra_coords: Optional[Mapping[str, Any]] = None,
        drop_dims: Optional[Iterable[str]] = None,
        drop_extra: Optional[Iterable[str]] = None,
    ) -> "Coordinates":
        dim_updates = OrderedDict()
        drop_dims = set(drop_dims or ())
        for dim, index in self._dim_indexes.items():
            if dim in drop_dims:
                continue
            if dim_indexes and dim in dim_indexes:
                dim_updates[dim] = _clone_index(dim_indexes[dim])
            else:
                dim_updates[dim] = _clone_index(index)
        if dim_indexes:
            for dim, index in dim_indexes.items():
                if dim in dim_updates:
                    continue
                dim_updates[dim] = _clone_index(index)

        extras = OrderedDict()
        drop_extra = set(drop_extra or ())
        for name, values in self._extra_coords.items():
            if name in drop_extra:
                continue
            extras[name] = _clone_extra(values)
        if extra_coords:
            for name, values in extra_coords.items():
                extras[name] = _normalize_extra(values)
        return Coordinates(dim_updates, extra_coords=extras, copy=False)

    def take(self, dim: str, indexer: Any) -> "Coordinates":
        if dim not in self._dim_indexes:
            raise KeyError(dim)
        taken = self._dim_indexes[dim].take(indexer)
        updated = OrderedDict(
            (name, (taken if name == dim else _clone_index(index)))
            for name, index in self._dim_indexes.items()
        )
        return Coordinates(updated, extra_coords=self._extra_coords, copy=False)

    def drop_dims(self, dims: Iterable[str]) -> "Coordinates":
        drop = set(dims)
        remaining = OrderedDict(
            (dim, _clone_index(index)) for dim, index in self._dim_indexes.items() if dim not in drop
        )
        extras = OrderedDict(self._extra_coords)
        for dim in drop:
            if dim in self._dim_indexes:
                extras.setdefault(dim, self._dim_indexes[dim].coord_array())
        return Coordinates(remaining, extra_coords=extras, copy=False)

    def rename(self, mapping: Mapping[str, str]) -> "Coordinates":
        renamed = OrderedDict()
        for dim, index in self._dim_indexes.items():
            renamed[mapping.get(dim, dim)] = _clone_index(index)
        extra = OrderedDict()
        for name, values in self._extra_coords.items():
            extra[mapping.get(name, name)] = _clone_extra(values)
        return Coordinates(renamed, extra_coords=extra, copy=False)

    def to(self, *args: Any, **kwargs: Any) -> "Coordinates":
        moved = OrderedDict(
            (dim, index.to(*args, **kwargs)) for dim, index in self._dim_indexes.items()
        )
        extras = OrderedDict()
        for name, values in self._extra_coords.items():
            if isinstance(values, torch.Tensor):
                extras[name] = values.to(*args, **kwargs)
            else:
                extras[name] = _clone_extra(values)
        return Coordinates(moved, extra_coords=extras, copy=False)


class CoordinatesView(MappingABC):
    """Mapping-style view that mirrors xarray's coords accessor."""

    def __init__(self, coordinates: Coordinates) -> None:
        self._coordinates = coordinates

    def __getitem__(self, key: str) -> CoordArray:
        return self._coordinates.coord_values(key)

    def __iter__(self) -> Iterator[str]:
        for dim in self._coordinates._dim_indexes.keys():
            yield dim
        for name in self._coordinates._extra_coords.keys():
            if name not in self._coordinates._dim_indexes:
                yield name

    def __len__(self) -> int:
        return len(self._coordinates._dim_indexes) + len(self._coordinates._extra_coords)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self._coordinates.has_coord(key)

    def to_dict(self) -> Dict[str, CoordArray]:
        return self._coordinates.to_dict()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        data = ", ".join(self.keys())
        return f"CoordinatesView([{data}])"


class IndexesView(MappingABC):
    """Mapping exposing BaseIndex objects for each dimension."""

    def __init__(self, coordinates: Coordinates) -> None:
        self._coordinates = coordinates

    def __getitem__(self, key: str) -> BaseIndex:
        if not self._coordinates.has_dim(key):
            raise KeyError(key)
        return self._coordinates.dim_index(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._coordinates._dim_indexes.keys())

    def __len__(self) -> int:
        return len(self._coordinates._dim_indexes)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self._coordinates.has_dim(key)

    def to_dict(self) -> Dict[str, BaseIndex]:
        return self._coordinates.dim_indexes()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        data = ", ".join(self.keys())
        return f"IndexesView([{data}])"
