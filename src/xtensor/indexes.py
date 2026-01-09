from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

import numpy as np
import torch

CoordValue = Union[Sequence[Any], np.ndarray, torch.Tensor, "pd.Index"]
CoordArray = Union[torch.Tensor, Tuple[Any, ...], "pd.Index"]


class BaseIndex:
    def __len__(self) -> int:
        raise NotImplementedError

    def clone(self) -> "BaseIndex":
        raise NotImplementedError

    def coord_array(self) -> CoordArray:
        raise NotImplementedError

    def take(self, indexer: Any) -> "BaseIndex":
        raise NotImplementedError

    def get_loc(self, value: Any) -> int:
        raise NotImplementedError

    def coord_value(self, position: int) -> Any:
        raise NotImplementedError

    def equals(self, other: "BaseIndex") -> bool:
        raise NotImplementedError

    def to_xarray(self):
        raise NotImplementedError

    def to(self, *args: Any, **kwargs: Any) -> "BaseIndex":
        return self

    def slice_indexer(self, start: Any, stop: Any, step: Optional[int]) -> slice:
        start_pos = self.get_loc(start) if start is not None else 0
        stop_pos = self.get_loc(stop) if stop is not None else len(self) - 1
        stop_pos = min(stop_pos, len(self) - 1)
        return slice(start_pos, stop_pos + 1, step)


class TorchIndex(BaseIndex):
    def __init__(self, values: torch.Tensor):
        if values.ndim != 1:
            raise ValueError("TorchIndex expects a 1D tensor.")
        self._values = values

    def __len__(self) -> int:
        return self._values.shape[0]

    def clone(self) -> "TorchIndex":
        return TorchIndex(self._values.clone())

    def coord_array(self) -> CoordArray:
        return self._values.clone()

    def take(self, indexer: Any) -> "TorchIndex":
        if isinstance(indexer, slice):
            taken = self._values[indexer]
        else:
            indices = _as_long_tensor(indexer, device=self._values.device)
            taken = self._values.index_select(0, indices)
        return TorchIndex(taken.reshape(-1))

    def get_loc(self, value: Any) -> int:
        target = torch.as_tensor(value, dtype=self._values.dtype, device=self._values.device)
        matches = torch.nonzero(self._values == target, as_tuple=False)
        if matches.numel() == 0:
            raise KeyError(f"Coordinate value '{value}' not found.")
        return int(matches[0].item())

    def coord_value(self, position: int) -> torch.Tensor:
        return self._values[position].clone()

    def equals(self, other: "BaseIndex") -> bool:
        if not isinstance(other, TorchIndex):
            return False
        if self._values.dtype.is_floating_point:
            return torch.allclose(self._values, other._values)
        return torch.equal(self._values, other._values)

    def to_xarray(self):
        return self._values.detach().cpu().numpy()

    def to(self, *args: Any, **kwargs: Any) -> "TorchIndex":
        return TorchIndex(self._values.to(*args, **kwargs))


class ArrayIndex(BaseIndex):
    def __init__(self, values: Tuple[Any, ...]):
        self._values = tuple(values)

    def __len__(self) -> int:
        return len(self._values)

    def clone(self) -> "ArrayIndex":
        return ArrayIndex(self._values)

    def coord_array(self) -> CoordArray:
        return tuple(self._values)

    def take(self, indexer: Any) -> "ArrayIndex":
        if isinstance(indexer, slice):
            return ArrayIndex(self._values[indexer])
        indices = _as_index_list(indexer)
        return ArrayIndex(tuple(self._values[i] for i in indices))

    def get_loc(self, value: Any) -> int:
        try:
            return self._values.index(value)
        except ValueError as error:
            raise KeyError(f"Coordinate value '{value}' not found.") from error

    def coord_value(self, position: int) -> Any:
        return self._values[position]

    def equals(self, other: "BaseIndex") -> bool:
        if not isinstance(other, ArrayIndex):
            return False
        return self._values == other._values

    def to_xarray(self):
        return np.asarray(self._values)


class PandasIndex(BaseIndex):
    def __init__(self, index: "pd.Index"):
        if pd is None:  # pragma: no cover - defensive
            raise RuntimeError("pandas is required for PandasIndex.")
        self._index = index.copy()

    def __len__(self) -> int:
        return len(self._index)

    def clone(self) -> "PandasIndex":
        return PandasIndex(self._index.copy())

    def coord_array(self) -> CoordArray:
        return self._index.copy()

    def take(self, indexer: Any) -> "PandasIndex":
        if isinstance(indexer, slice):
            taken = self._index[indexer]
        else:
            indices = _as_index_list(indexer)
            taken = self._index.take(indices)
        return PandasIndex(taken)

    def get_loc(self, value: Any) -> int:
        loc = self._index.get_loc(value)
        if isinstance(loc, slice):
            start = loc.start if loc.start is not None else 0
            return start
        if isinstance(loc, (np.ndarray, list)):
            if len(loc) == 0:
                raise KeyError(f"Coordinate value '{value}' not found.")
            return int(loc[0])
        return int(loc)

    def coord_value(self, position: int) -> Any:
        return self._index[position]

    def equals(self, other: "BaseIndex") -> bool:
        if not isinstance(other, PandasIndex):
            return False
        return self._index.equals(other._index)

    def to_xarray(self):
        return self._index.copy()

    def slice_indexer(self, start: Any, stop: Any, step: Optional[int]) -> slice:
        start_loc, stop_loc = self._index.slice_locs(start, stop)
        return slice(start_loc, stop_loc, step)


def _as_long_tensor(indexer: Any, device: torch.device) -> torch.Tensor:
    if isinstance(indexer, torch.Tensor):
        tensor = indexer.to(device=device, dtype=torch.long)
    elif isinstance(indexer, np.ndarray):
        tensor = torch.as_tensor(indexer, dtype=torch.long, device=device)
    elif isinstance(indexer, slice):
        raise TypeError("slice should be handled separately")
    else:
        tensor = torch.as_tensor(list(indexer), dtype=torch.long, device=device)
    return tensor


def _as_index_list(indexer: Any) -> Sequence[int]:
    if isinstance(indexer, torch.Tensor):
        return [int(value) for value in indexer.cpu().tolist()]
    if isinstance(indexer, np.ndarray):
        return [int(value) for value in indexer.tolist()]
    if isinstance(indexer, slice):
        raise TypeError("slice should be handled separately")
    if isinstance(indexer, Sequence):
        return [int(value) for value in indexer]
    return [int(indexer)]


def build_index(values: Optional[CoordValue], size: int, dim: str, *, device: torch.device) -> BaseIndex:
    if values is None:
        data = torch.arange(size, device=device, dtype=torch.float64)
        return TorchIndex(data)

    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.shape[0] != size:
            raise ValueError(f"Coordinate length mismatch on dim '{dim}'. Expected {size}, got {values.shape[0]}")
        return TorchIndex(values.to(device=device))

    pandas_index = _try_pandas_index(values)
    if pandas_index is not None:
        if len(pandas_index) != size:
            raise ValueError(f"Coordinate length mismatch on dim '{dim}'. Expected {size}, got {len(pandas_index)}")
        return PandasIndex(pandas_index)

    if isinstance(values, np.ndarray):
        array = values
    elif hasattr(values, "to_numpy"):
        array = values.to_numpy()
    else:
        array = np.asarray(list(values))

    if array.ndim != 1 or array.shape[0] != size:
        raise ValueError(f"Coordinate length mismatch on dim '{dim}'. Expected {size}, got {array.shape[0]}")

    kind = array.dtype.kind
    if kind in ("f", "i", "u", "b"):
        tensor = torch.as_tensor(array, device=device)
        return TorchIndex(tensor)
    if kind == "M":
        if pd is not None:
            converted = pd.to_datetime(array)
            return PandasIndex(pd.Index(converted))
        converted = tuple(np.asarray(array, dtype="datetime64[ns]").tolist())
        return ArrayIndex(converted)
    if kind == "m":
        if pd is not None:
            converted = pd.to_timedelta(array)
            return PandasIndex(pd.Index(converted))
        converted = tuple(np.asarray(array, dtype="timedelta64[ns]").tolist())
        return ArrayIndex(converted)
    return ArrayIndex(tuple(array.tolist()))


def _try_pandas_index(values: Any) -> Optional["pd.Index"]:
    if pd is None:
        return None
    if isinstance(values, pd.Index):
        return values
    return None
