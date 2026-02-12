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

    def get_indexer(self, values: Any, *, device: torch.device) -> torch.Tensor:
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
        self._init_lookup()

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
        indexer = self.get_indexer([value], device=self._values.device)
        if indexer.numel() == 0:
            raise KeyError(f"Coordinate value '{value}' not found.")
        return int(indexer.item())

    def get_indexer(self, values: Any, *, device: torch.device) -> torch.Tensor:
        queries = _as_lookup_tensor(values, dtype=self._lookup_dtype, device=self._values.device)
        if queries.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if self._unique_lookup_values.numel() == 0:
            raise KeyError("Cannot index into an empty coordinate.")
        positions = torch.searchsorted(self._unique_lookup_values, queries)
        max_pos = self._unique_lookup_values.numel() - 1
        positions = positions.clamp(max=max_pos)
        matches = self._unique_lookup_values[positions] == queries
        if not torch.all(matches):
            missing_tensor = _recover_missing_queries(queries[~matches], self._values.dtype)
            raise KeyError(f"Coordinate value(s) {missing_tensor.tolist()} not found.")
        result = self._unique_positions[positions]
        return result.to(device=device, dtype=torch.long)

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

    def _init_lookup(self) -> None:
        lookup_values = self._values
        if lookup_values.dtype == torch.bool:
            lookup_values = lookup_values.to(dtype=torch.uint8)
        self._lookup_dtype = lookup_values.dtype
        if lookup_values.numel() == 0:
            self._unique_lookup_values = lookup_values
            self._unique_positions = torch.empty(0, dtype=torch.long, device=lookup_values.device)
            return
        presorted = _presorted_unique_lookup(lookup_values)
        if presorted is not None:
            self._unique_lookup_values, self._unique_positions = presorted
            return
        sort_idx = torch.argsort(lookup_values)
        sorted_values = lookup_values[sort_idx]
        numel = sorted_values.numel()
        if numel == 1:
            change = torch.ones(1, dtype=torch.bool, device=sorted_values.device)
        else:
            change = torch.ones(numel, dtype=torch.bool, device=sorted_values.device)
            change[1:] = sorted_values[1:] != sorted_values[:-1]
        self._unique_lookup_values = sorted_values[change]
        self._unique_positions = sort_idx[change]


class RangeIndex(BaseIndex):
    def __init__(
        self,
        start: Union[int, float],
        step: Union[int, float],
        size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if size < 0:
            raise ValueError("size must be non-negative.")
        if step == 0:
            raise ValueError("step must be non-zero.")
        device = device or torch.device("cpu")
        dtype = dtype or torch.float64
        self._size = int(size)
        self._device = device
        self._dtype = dtype
        self._start = torch.as_tensor(start, dtype=dtype, device=device).reshape(())
        self._step = torch.as_tensor(step, dtype=dtype, device=device).reshape(())

    def __len__(self) -> int:
        return self._size

    def clone(self) -> "RangeIndex":
        return RangeIndex(
            self._start.item(),
            self._step.item(),
            self._size,
            device=self._device,
            dtype=self._dtype,
        )

    def coord_array(self) -> CoordArray:
        if self._size == 0:
            return torch.empty(0, dtype=self._dtype, device=self._device)
        if float(self._start.item()) == 0.0 and float(self._step.item()) == 1.0:
            return torch.arange(self._size, dtype=self._dtype, device=self._device)
        arange = torch.arange(self._size, dtype=self._dtype, device=self._device)
        return self._start + arange * self._step

    def take(self, indexer: Any) -> "BaseIndex":
        if isinstance(indexer, slice):
            start, stop, stride = indexer.indices(self._size)
            if stride <= 0:
                values = self.coord_array()[indexer]
                return TorchIndex(values)
            new_size = max((stop - start + (stride - 1)) // stride, 0)
            new_start = (self._start + self._step * start).item()
            new_step = (self._step * stride).item()
            return RangeIndex(new_start, new_step, new_size, device=self._device, dtype=self._dtype)
        indices = _as_long_tensor(indexer, device=self._device)
        values = self.coord_array().index_select(0, indices)
        return TorchIndex(values)

    def get_loc(self, value: Any) -> int:
        result = self.get_indexer([value], device=self._device)
        if result.numel() == 0:
            raise KeyError(value)
        return int(result.item())

    def get_indexer(self, values: Any, *, device: torch.device) -> torch.Tensor:
        queries = _as_lookup_tensor(values, dtype=self._dtype, device=self._device)
        if queries.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        rel = (queries - self._start) / self._step
        positions = torch.round(rel)
        matches = torch.isclose(rel, positions, atol=1e-8, rtol=0.0)
        positions = positions.to(dtype=torch.long)
        valid = (positions >= 0) & (positions < self._size)
        matches &= valid
        if not bool(torch.all(matches)):
            missing = queries[~matches]
            raise KeyError(f"Coordinate value(s) {missing.tolist()} not found.")
        return positions.to(device=device)

    def coord_value(self, position: int) -> Any:
        if position < 0 or position >= self._size:
            raise IndexError(position)
        return (self._start + self._step * position).clone()

    def equals(self, other: "BaseIndex") -> bool:
        if not isinstance(other, RangeIndex):
            return False
        return (
            self._size == other._size
            and self._dtype == other._dtype
            and self._device == other._device
            and torch.allclose(self._start, other._start)
            and torch.allclose(self._step, other._step)
        )

    def to_xarray(self):
        return self.coord_array().detach().cpu().numpy()

    def to(self, *args: Any, **kwargs: Any) -> "RangeIndex":
        start = self._start.to(*args, **kwargs)
        step = self._step.to(*args, **kwargs)
        return RangeIndex(start.item(), step.item(), self._size, device=start.device, dtype=start.dtype)

    def slice_indexer(self, start: Any, stop: Any, step: Optional[int]) -> slice:
        start_loc = self.get_loc(start) if start is not None else 0
        stop_loc = self.get_loc(stop) if stop is not None else self._size - 1
        stop_loc = min(stop_loc, self._size - 1)
        return slice(start_loc, stop_loc + 1, step)


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
        return int(self.get_indexer([value], device=torch.device("cpu")).item())

    def get_indexer(self, values: Any, *, device: torch.device) -> torch.Tensor:
        queries = _as_pandas_lookup(values)
        if len(queries) == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        result = self._index.get_indexer(queries)
        if isinstance(result, slice):
            start = result.start or 0
            stop = result.stop or 0
            result = np.arange(start, stop, result.step or 1, dtype=np.int64)
        missing_mask = result < 0
        if np.any(missing_mask):
            missing = [queries[idx] for idx in np.flatnonzero(missing_mask)]
            raise KeyError(f"Coordinate value(s) {missing} not found.")
        return torch.as_tensor(result, dtype=torch.long, device=device)

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


class SortedUniqueIndex(BaseIndex):
    def __init__(self, values: torch.Tensor, *, safety_check: bool = True):
        if values.ndim != 1:
            raise ValueError("SortedUniqueIndex expects a 1D tensor.")
        if safety_check and values.numel() > 1:
            diffs = values[1:] - values[:-1]
            if not bool(torch.all(diffs > 0)):
                raise ValueError("Values must be strictly increasing.")
        self._values = values.clone()

    def __len__(self) -> int:
        return self._values.shape[0]

    def clone(self) -> "SortedUniqueIndex":
        return SortedUniqueIndex(self._values, safety_check=False)

    def coord_array(self) -> CoordArray:
        return self._values.clone()

    def take(self, indexer: Any) -> "BaseIndex":
        if isinstance(indexer, slice):
            return SortedUniqueIndex(self._values[indexer], safety_check=False)
        indices = _as_long_tensor(indexer, device=self._values.device)
        taken = self._values.index_select(0, indices)
        return TorchIndex(taken)

    def get_loc(self, value: Any) -> int:
        return int(self.get_indexer([value], device=self._values.device).item())

    def get_indexer(self, values: Any, *, device: torch.device) -> torch.Tensor:
        queries = _as_lookup_tensor(values, dtype=self._values.dtype, device=self._values.device)
        if queries.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        positions = torch.searchsorted(self._values, queries)
        within = positions < self._values.shape[0]
        gathered = torch.zeros_like(queries)
        if positions.numel() > 0:
            clamped = positions.clamp(max=self._values.shape[0] - 1)
            gathered = self._values[clamped]
        matches = within & (gathered == queries)
        if not bool(torch.all(matches)):
            missing = queries[~matches]
            raise KeyError(f"Coordinate value(s) {missing.tolist()} not found.")
        return positions.to(dtype=torch.long, device=device)

    def coord_value(self, position: int) -> Any:
        return self._values[position].clone()

    def equals(self, other: "BaseIndex") -> bool:
        if not isinstance(other, SortedUniqueIndex):
            return False
        return torch.equal(self._values, other._values)

    def to_xarray(self):
        return self._values.detach().cpu().numpy()

    def to(self, *args: Any, **kwargs: Any) -> "SortedUniqueIndex":
        return SortedUniqueIndex(self._values.to(*args, **kwargs), safety_check=False)


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
    if isinstance(values, BaseIndex):
        return values

    if values is None:
        data = torch.arange(size, device=device, dtype=torch.float64)
        return TorchIndex(data)

    tensor_values = _try_build_tensor_index(values, size, dim, device=device)
    if tensor_values is not None:
        return TorchIndex(tensor_values)

    pandas_index = _build_pandas_index(values, size, dim)
    return PandasIndex(pandas_index)


def _try_build_tensor_index(values: CoordValue, size: int, dim: str, *, device: torch.device) -> Optional[torch.Tensor]:
    if isinstance(values, torch.Tensor):
        tensor = values
        if tensor.ndim != 1 or tensor.shape[0] != size:
            raise ValueError(f"Coordinate length mismatch on dim '{dim}'. Expected {size}, got {tensor.shape[0]}")
        return tensor.to(device=device)

    array = _coordinate_array(values)
    if array is None:
        return None
    if array.ndim != 1 or array.shape[0] != size:
        raise ValueError(f"Coordinate length mismatch on dim '{dim}'. Expected {size}, got {array.shape[0]}")
    if array.dtype.kind not in ("f", "i", "u", "b"):
        return None
    return torch.as_tensor(array, device=device)


def _build_pandas_index(values: CoordValue, size: int, dim: str):
    if pd is None:
        raise RuntimeError("pandas is required for non-numeric coordinate values.")
    if isinstance(values, pd.Index):
        index = values.copy()
    else:
        array = _coordinate_array(values)
        if array is None:
            array = np.asarray([values])
        if array.ndim != 1 or array.shape[0] != size:
            raise ValueError(f"Coordinate length mismatch on dim '{dim}'. Expected {size}, got {array.shape[0]}")
        index = pd.Index(array)
    if len(index) != size:
        raise ValueError(f"Coordinate length mismatch on dim '{dim}'. Expected {size}, got {len(index)}")
    return index


def _coordinate_array(values: CoordValue) -> Optional[np.ndarray]:
    if isinstance(values, np.ndarray):
        return values
    if pd is not None and isinstance(values, pd.Index):
        return values.to_numpy()
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    if hasattr(values, "to_numpy"):
        array = values.to_numpy()
        return np.asarray(array)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return np.asarray(values)
    try:
        return np.asarray(list(values))
    except TypeError:
        return None


def _presorted_unique_lookup(values: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return (unique_values, positions) if values are already monotonic nondecreasing."""
    if values.numel() <= 1:
        arange = torch.arange(values.numel(), dtype=torch.long, device=values.device)
        return values, arange
    # Fast path for already sorted coordinates (common for range-style indexes).
    diffs = values[1:] >= values[:-1]
    if not bool(torch.all(diffs)):
        return None
    change = torch.ones(values.numel(), dtype=torch.bool, device=values.device)
    change[1:] = values[1:] != values[:-1]
    positions = torch.arange(values.numel(), dtype=torch.long, device=values.device)
    unique_values = values[change]
    unique_positions = positions[change]
    return unique_values, unique_positions


def _as_lookup_tensor(values: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.to(device=device, dtype=dtype)
    elif isinstance(values, np.ndarray):
        tensor = torch.as_tensor(values, dtype=dtype, device=device)
    elif pd is not None and isinstance(values, pd.Index):
        tensor = torch.as_tensor(values.to_numpy(), dtype=dtype, device=device)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        tensor = torch.as_tensor(values, dtype=dtype, device=device)
    else:
        tensor = torch.as_tensor([values], dtype=dtype, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    return tensor.reshape(-1)


def _recover_missing_queries(queries: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.bool:
        return queries.to(dtype=torch.bool)
    return queries.to(dtype=dtype)


def _as_pandas_lookup(values: Any) -> Sequence[Any]:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    if pd is not None and isinstance(values, pd.Index):
        return values.tolist()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return [values]


def arange_index(
    size: int,
    start: Union[int, float]=0,
    step: Union[int, float]=1,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> RangeIndex:
    return RangeIndex(start, step, size, device=device, dtype=dtype)


def sorted_unique_index(values: Union[torch.Tensor, Sequence[Any], np.ndarray], *, safety_check: bool = True) -> SortedUniqueIndex:
    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    return SortedUniqueIndex(tensor.reshape(-1), safety_check=safety_check)
