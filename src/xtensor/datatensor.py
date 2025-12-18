from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

CoordValue = Union[Sequence[Any], np.ndarray, torch.Tensor]

def _to_tensor(data: Union[np.ndarray, torch.Tensor, Sequence[Any]]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.clone()
    return torch.as_tensor(data)

def _to_coord_tuple(values: Optional[CoordValue], size: int, dim: str) -> Tuple[Any, ...]:
    if values is None:
        return tuple(range(size))

    array = None
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    elif isinstance(values, np.ndarray):
        array = values
    elif hasattr(values, "to_numpy"):
        array = values.to_numpy()
    else:
        array = np.asarray(list(values))

    if array.ndim != 1 or array.shape[0] != size:
        raise ValueError(f"Coordinate length mismatch on dim '{dim}'. Expected {size}, got {array.shape[0]}")

    if array.dtype == object:
        return tuple(array.tolist())
    return tuple(array)

def _as_list(value: Any) -> Sequence[Any]:
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

class DataTensor:
    """Minimal xarray.DataArray inspired wrapper around torch.Tensor."""

    __array_priority__ = 1000

    def __init__(self, data: Union[np.ndarray, torch.Tensor, Sequence[Any]], coords: Mapping[str, CoordValue], dims: Sequence[str]):
        tensor = _to_tensor(data)
        dims = tuple(dims)
        if tensor.ndim != len(dims):
            raise ValueError(f"Expected dims of length {tensor.ndim}, received {len(dims)}")

        normalized_coords: Dict[str, Tuple[Any, ...]] = {}
        for dim, size in zip(dims, tensor.shape):
            coord_values = coords.get(dim)
            normalized_coords[dim] = _to_coord_tuple(coord_values, size, dim)

        self._data = tensor
        self._dims = dims
        self._coords = normalized_coords

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def values(self) -> torch.Tensor:
        return self._data

    @property
    def dims(self) -> Tuple[str, ...]:
        return self._dims

    @property
    def coords(self) -> Dict[str, Tuple[Any, ...]]:
        return dict(self._coords)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def sizes(self) -> Dict[str, int]:
        return {dim: self._data.shape[idx] for idx, dim in enumerate(self._dims)}

    @staticmethod
    def from_pandas(obj: Any, dims: Optional[Sequence[str]] = None) -> "DataTensor":
        import pandas as pd

        if isinstance(obj, pd.Series):
            dim = (dims[0] if dims else obj.index.name) or "index"
            data = torch.as_tensor(obj.to_numpy())
            coords = {dim: obj.index}
            return DataTensor(data, coords, (dim,))

        if isinstance(obj, pd.DataFrame):
            dims = dims or (obj.columns.name or "columns", obj.index.name or "index")
            if len(dims) != 2:
                raise ValueError("DataFrame conversion expects exactly two dims.")
            data = torch.as_tensor(obj.to_numpy().T)
            coords = {
                dims[0]: obj.columns,
                dims[1]: obj.index,
            }
            return DataTensor(data, coords, tuple(dims))

        raise TypeError("from_pandas expects a pandas Series or DataFrame.")

    @staticmethod
    def from_dataarray(array: Any) -> "DataTensor":
        try:
            import xarray as xr  # noqa: F401
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("xarray must be installed to build from a DataArray.") from error

        dims = tuple(array.dims)
        coords = {dim: array.coords[dim].to_numpy() for dim in dims}
        return DataTensor(array.data, coords, dims)

    def sel(self, **indexers: Any) -> "DataTensor":
        return self._select(indexers, use_coords=True)

    def isel(self, **indexers: Any) -> "DataTensor":
        return self._select(indexers, use_coords=False)

    def mean(self, dim: Optional[Union[str, Sequence[str]]] = None, keepdims: bool = False) -> "DataTensor":
        return self._reduce(torch.mean, dim=dim, keepdims=keepdims)

    def std(self, dim: Optional[Union[str, Sequence[str]]] = None, keepdims: bool = False, unbiased: bool = False) -> "DataTensor":
        def _std(data: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
            if dim is None:
                return torch.std(data.view(-1), unbiased=unbiased)
            return torch.std(data, dim=dim, keepdim=keepdim, unbiased=unbiased)

        return self._reduce(_std, dim=dim, keepdims=keepdims, allow_all_reduce=True)

    def sum(self, dim: Optional[Union[str, Sequence[str]]] = None, keepdims: bool = False) -> "DataTensor":
        return self._reduce(torch.sum, dim=dim, keepdims=keepdims)

    def min(self, dim: Optional[Union[str, Sequence[str]]] = None, keepdims: bool = False) -> "DataTensor":
        def _amin(data: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
            if dim is None:
                return torch.amin(data)
            return torch.amin(data, dim=dim, keepdim=keepdim)

        return self._reduce(_amin, dim=dim, keepdims=keepdims, allow_all_reduce=True)

    def max(self, dim: Optional[Union[str, Sequence[str]]] = None, keepdims: bool = False) -> "DataTensor":
        def _amax(data: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
            if dim is None:
                return torch.amax(data)
            return torch.amax(data, dim=dim, keepdim=keepdim)

        return self._reduce(_amax, dim=dim, keepdims=keepdims, allow_all_reduce=True)

    def prod(self, dim: Optional[Union[str, Sequence[str]]] = None, keepdims: bool = False) -> "DataTensor":
        return self._reduce(torch.prod, dim=dim, keepdims=keepdims)

    def to(self, *args: Any, **kwargs: Any) -> "DataTensor":
        moved = self._data.to(*args, **kwargs)
        return DataTensor(moved, self._coords, self._dims)

    def transpose(self, *dims: str) -> "DataTensor":
        if not dims:
            dims = tuple(reversed(self._dims))
        if set(dims) != set(self._dims) or len(dims) != len(self._dims):
            raise ValueError(f"transpose requires a permutation of {self._dims}, received {dims}")
        perm = [self._dims.index(dim) for dim in dims]
        data = self._data.permute(*perm)
        coords = {dim: self._coords[dim] for dim in dims}
        return DataTensor(data, coords, dims)

    def expand_dims(
        self,
        dims: Union[str, Sequence[str], Mapping[str, CoordValue]],
        axis: Optional[int] = 0,
    ) -> "DataTensor":
        if isinstance(dims, str):
            items = [(dims, None)]
        elif isinstance(dims, Mapping):
            items = list(dims.items())
        else:
            items = [(name, None) for name in dims]

        target_axis = axis if axis is not None else 0
        if target_axis < 0:
            target_axis += len(self._dims) + 1
        target_axis = max(0, min(target_axis, len(self._dims)))

        data = self._data
        new_dims = list(self._dims)
        new_coords = dict(self._coords)

        for offset, (dim, coord_values) in enumerate(items):
            insert_at = target_axis + offset
            data = data.unsqueeze(insert_at)
            new_dims.insert(insert_at, dim)
            coord_tuple = _to_coord_tuple(coord_values, 1, dim) if coord_values is not None else (0,)
            new_coords[dim] = coord_tuple

        return DataTensor(data, new_coords, tuple(new_dims))

    def squeeze(self, dims: Optional[Union[str, Sequence[str]]] = None) -> "DataTensor":
        if dims is None:
            target_dims = [dim for dim, size in zip(self._dims, self._data.shape) if size == 1]
        else:
            target_dims = [dims] if isinstance(dims, str) else list(dims)
        if not target_dims:
            return self

        axes = []
        for dim in target_dims:
            if dim not in self._dims:
                raise ValueError(f"Unknown dimension '{dim}'.")
            axis = self._dims.index(dim)
            if self._data.shape[axis] != 1:
                raise ValueError(f"Cannot squeeze dimension '{dim}' with size {self._data.shape[axis]}.")
            axes.append(axis)

        data = self._data
        for axis in sorted(axes, reverse=True):
            data = data.squeeze(axis)

        new_dims = tuple(dim for dim in self._dims if dim not in target_dims)
        new_coords = {dim: self._coords[dim] for dim in new_dims}
        return DataTensor(data, new_coords, new_dims)

    def to_dataarray(self):
        try:
            import xarray as xr
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("xarray must be installed to export to DataArray.") from error

        coords = {dim: np.asarray(values) for dim, values in self._coords.items()}
        data = self._data.detach().cpu().numpy()
        return xr.DataArray(data, dims=self._dims, coords=coords)

    def to_pandas(self):
        import pandas as pd

        if len(self._dims) == 1:
            dim = self._dims[0]
            index = pd.Index(self._coords[dim], name=dim)
            data = self._data.detach().cpu().numpy()
            return pd.Series(data, index=index)

        if len(self._dims) == 2:
            row_dim, col_dim = self._dims
            index = pd.Index(self._coords[row_dim], name=row_dim)
            columns = pd.Index(self._coords[col_dim], name=col_dim)
            data = self._data.detach().cpu().numpy()
            return pd.DataFrame(data, index=index, columns=columns)

        raise ValueError("to_pandas only supports tensors with one or two dimensions.")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        coord_str = ", ".join(f"{dim}: {vals[:3]}..." if len(vals) > 3 else f"{dim}: {vals}" for dim, vals in self._coords.items())
        return f"DataTensor(shape={self.shape}, dims={self._dims}, coords={{ {coord_str} }})"

    # Elementwise math -------------------------------------------------
    def _binary_op(self, other: Any, op: Callable[[torch.Tensor, Any], torch.Tensor], op_name: str) -> "DataTensor":
        if isinstance(other, DataTensor):
            dims = self._dims
            if dims != other.dims:
                raise ValueError(f"{op_name} requires matching dims. {self._dims} vs {other.dims}")
            coords = self._merge_coords(other, op_name)
            result = op(self._data, other.data)
        else:
            result = op(self._data, other)
            coords = self._coords
            dims = self._dims
        return DataTensor(result, coords, dims)

    def _merge_coords(self, other: "DataTensor", op_name: str) -> Dict[str, Tuple[Any, ...]]:
        merged: Dict[str, Tuple[Any, ...]] = {}
        for dim in self._dims:
            coords_a = self._coords[dim]
            coords_b = other.coords[dim]
            len_a = len(coords_a)
            len_b = len(coords_b)
            if len_a == len_b:
                if coords_a != coords_b:
                    raise ValueError(f"{op_name} requires matching coordinates on dim '{dim}'.")
                merged[dim] = coords_a
            elif len_a == 1:
                merged[dim] = coords_b
            elif len_b == 1:
                merged[dim] = coords_a
            else:
                raise ValueError(f"{op_name} cannot broadcast dimension '{dim}' (sizes {len_a} vs {len_b}).")
        return merged

    def __add__(self, other: Any) -> "DataTensor":
        return self._binary_op(other, torch.add, "add")

    def __radd__(self, other: Any) -> "DataTensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "DataTensor":
        return self._binary_op(other, torch.sub, "sub")

    def __rsub__(self, other: Any) -> "DataTensor":
        return self._binary_op(other, lambda lhs, rhs: torch.sub(rhs, lhs), "rsub")

    def __mul__(self, other: Any) -> "DataTensor":
        return self._binary_op(other, torch.mul, "mul")

    def __rmul__(self, other: Any) -> "DataTensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "DataTensor":
        return self._binary_op(other, torch.true_divide, "truediv")

    def __rtruediv__(self, other: Any) -> "DataTensor":
        return self._binary_op(other, lambda lhs, rhs: torch.true_divide(rhs, lhs), "rtruediv")

    # Helpers ----------------------------------------------------------
    def _reduce(
        self,
        op: Callable[..., torch.Tensor],
        dim: Optional[Union[str, Sequence[str]]] = None,
        keepdims: bool = False,
        allow_all_reduce: bool = False,
    ) -> "DataTensor":
        axes = self._dims_to_axes(dim)
        axes_set = set(axes) if axes is not None else None
        reduced_dims = set(self._dims if axes is None else (self._dims[idx] for idx in axes))
        if axes is None:
            reduced = op(self._data, dim=None) if allow_all_reduce else op(self._data)
            if keepdims:
                reduced = reduced.reshape([1] * self._data.ndim)
                new_dims = self._dims
            else:
                new_dims = ()
        else:
            reduced = self._data
            for axis in sorted(axes, reverse=True):
                reduced = op(reduced, dim=axis, keepdim=keepdims)
            if keepdims:
                new_dims = self._dims
            else:
                new_dims = tuple(dim for idx, dim in enumerate(self._dims) if idx not in axes_set)

        if not new_dims:
            return DataTensor(reduced, {}, ())
        if keepdims:
            new_coords = dict(self._coords)
            for dim in reduced_dims:
                new_coords.pop(dim, None)
        else:
            new_coords = {dim: self._coords[dim] for dim in new_dims}
        return DataTensor(reduced, new_coords, new_dims)

    def _dims_to_axes(self, dim: Optional[Union[str, Sequence[str]]]) -> Optional[Sequence[int]]:
        if dim is None:
            return None
        dims = (dim,) if isinstance(dim, str) else tuple(dim)
        axes = []
        for d in dims:
            if d not in self._dims:
                raise ValueError(f"Unknown dimension '{d}'. Known dims: {self._dims}")
            axes.append(self._dims.index(d))
        return axes

    def _select(self, indexers: Mapping[str, Any], use_coords: bool) -> "DataTensor":
        if not indexers:
            return self

        index_tuple: list[Any] = []
        new_dims: list[str] = []
        new_coords: Dict[str, Tuple[Any, ...]] = {}

        for axis, dim in enumerate(self._dims):
            axis_coords = self._coords[dim]
            if dim in indexers:
                indexer = indexers[dim]
                normalized, coord_values, drop_dim = self._normalize_indexer(axis_coords, indexer, use_coords)
                index_tuple.append(normalized)
                if not drop_dim:
                    new_dims.append(dim)
                    new_coords[dim] = coord_values
            else:
                index_tuple.append(slice(None))
                new_dims.append(dim)
                new_coords[dim] = axis_coords

        data = self._data[tuple(index_tuple)]
        return DataTensor(data, new_coords, tuple(new_dims))

    def _normalize_indexer(self, axis_coords: Tuple[Any, ...], selector: Any, use_coords: bool):
        if isinstance(selector, slice):
            if use_coords:
                start = self._coord_to_index(axis_coords, selector.start) if selector.start is not None else 0
                stop = self._coord_to_index(axis_coords, selector.stop) if selector.stop is not None else len(axis_coords) - 1
                stop = min(stop, len(axis_coords) - 1)
                idx = slice(start, stop + 1, selector.step)
            else:
                idx = selector
            selected_coords = axis_coords[idx]
            return idx, tuple(selected_coords), False

        values = _as_list(selector)

        if use_coords:
            indices = [self._coord_to_index(axis_coords, val) for val in values]
        else:
            indices = [int(val) for val in values]

        if len(indices) == 1 and not isinstance(selector, (list, tuple, np.ndarray, torch.Tensor)):
            idx_value = indices[0]
            coord_value = axis_coords[idx_value]
            return idx_value, tuple([coord_value]), True

        tensor_index = torch.as_tensor(indices, dtype=torch.long, device=self._data.device)
        coord_values = tuple(axis_coords[i] for i in indices)
        return tensor_index, coord_values, False

    @staticmethod
    def _coord_to_index(axis_coords: Tuple[Any, ...], value: Any) -> int:
        try:
            return axis_coords.index(value)
        except ValueError as error:
            raise KeyError(f"Coordinate value '{value}' not found.") from error
