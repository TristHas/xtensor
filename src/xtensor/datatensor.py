from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .alignment import align_binary_operands
from .coordinates import Coordinates, CoordinatesView, IndexesView
from .indexes import BaseIndex, CoordArray, CoordValue, build_index
from .variable import Variable

_DTYPE_MAP = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "float": torch.float32,
    "double": torch.float64,
    "half": torch.float16,
    "bfloat16": torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float32,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _to_tensor(data: Union[np.ndarray, torch.Tensor, Sequence[Any]]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data  # .clone()
    return torch.as_tensor(data)


def _as_list(value: Any) -> Sequence[Any]:
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_dtype(value: Union[str, np.dtype, torch.dtype, type, None]) -> Optional[torch.dtype]:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, np.dtype):
        key = value.name
    elif isinstance(value, type):
        try:
            key = np.dtype(value).name
        except TypeError:
            key = value.__name__
    else:
        key = str(value)
    key = key.lower()
    if key.startswith("torch."):
        key = key.split(".", 1)[1]
    return _DTYPE_MAP.get(key)


_TORCH_HANDLERS: Dict[Any, Callable[..., Any]] = {}


def _implements(*torch_funcs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        for torch_func in torch_funcs:
            _TORCH_HANDLERS[torch_func] = func
        return func
    return decorator


def _disable_torch_function_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    with torch._C.DisableTorchFunction():
        return func(*args, **kwargs)


def _ensure_out_argument_supported(out: Optional[Any]) -> None:
    if out is not None:
        raise NotImplementedError("The 'out' argument is not supported for DataTensor torch integrations.")


def _expanded_indexer(key: Any, ndim: int) -> Tuple[Any, ...]:
    if not isinstance(key, tuple):
        key = (key,)
    new_key: list[Any] = []
    found_ellipsis = False
    for item in key:
        if item is Ellipsis:
            if not found_ellipsis:
                new_key.extend((ndim + 1 - len(key)) * [slice(None)])
                found_ellipsis = True
            else:
                new_key.append(slice(None))
        else:
            new_key.append(item)
    if len(new_key) > ndim:
        raise IndexError("too many indices")
    new_key.extend((ndim - len(new_key)) * [slice(None)])
    return tuple(new_key)

class DataTensor:
    """Minimal xarray.DataArray inspired wrapper around torch.Tensor."""

    __array_priority__ = 1000

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _TORCH_HANDLERS:
            return NotImplemented
        if not any(issubclass(t, cls) for t in types):
            return NotImplemented
        handler = _TORCH_HANDLERS[func]
        return handler(*args, **kwargs)

    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor, Sequence[Any]],
        coords: Mapping[str, CoordValue],
        dims: Sequence[str],
        *,
        attrs: Optional[Mapping[str, Any]] = None,
    ):
        tensor = _to_tensor(data)
        dims = tuple(dims)
        if tensor.ndim != len(dims):
            raise ValueError(f"Expected dims of length {tensor.ndim}, received {len(dims)}")

        self._variable = Variable(tensor, dims)
        self._dims = self._variable.dims
        coord_map = dict(coords)
        dim_indexes: "OrderedDict[str, BaseIndex]" = OrderedDict()
        for dim, size in zip(self._dims, tensor.shape):
            coord_values = coord_map.get(dim)
            dim_indexes[dim] = build_index(coord_values, size, dim, device=tensor.device)
        extra_coords = {name: value for name, value in coord_map.items() if name not in dim_indexes}

        self._coords = Coordinates(dim_indexes, extra_coords=extra_coords)
        self._attrs: Dict[str, Any] = dict(attrs or {})

    @property
    def data(self) -> torch.Tensor:
        return self._variable.data

    @property
    def values(self) -> torch.Tensor:
        return self._variable.data

    @property
    def device(self) -> torch.device:
        return self._variable.data.device

    @property
    def grad(self) -> Optional["DataTensor"]:
        grad = self._variable.data.grad
        if grad is None:
            return None
        return DataTensor(grad, self.coords, self._dims)

    @property
    def dims(self) -> Tuple[str, ...]:
        return self._variable.dims

    @property
    def coords(self) -> CoordinatesView:
        return CoordinatesView(self._coords)

    @property
    def indexes(self) -> IndexesView:
        return IndexesView(self._coords)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._variable.shape

    @property
    def sizes(self) -> Dict[str, int]:
        return self._variable.sizes()

    @property
    def attrs(self) -> Dict[str, Any]:
        return dict(self._attrs)

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
        moved = self.data.to(*args, **kwargs)
        variable = self._variable.with_data(moved)
        moved_coords = self._coords.to(*args, **kwargs)
        return self._new(variable=variable, coords=moved_coords)

    def transpose(self, *dims: str) -> "DataTensor":
        if not dims:
            dims = tuple(reversed(self._dims))
        if set(dims) != set(self._dims) or len(dims) != len(self._dims):
            raise ValueError(f"transpose requires a permutation of {self._dims}, received {dims}")
        perm = [self._dims.index(dim) for dim in dims]
        data = self.data.permute(*perm)
        variable = self._variable.with_data(data, dims)
        return self._new(variable=variable, dims=dims)

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

        data = self.data
        new_dims = list(self._dims)
        base_indexes = self._coords.dim_indexes()
        insert_indexes: Dict[str, BaseIndex] = {}

        for offset, (dim, coord_values) in enumerate(items):
            insert_at = target_axis + offset
            data = data.unsqueeze(insert_at)
            new_dims.insert(insert_at, dim)
            values = coord_values if coord_values is not None else (0,)
            insert_indexes[dim] = build_index(values, 1, dim, device=self.device)

        new_dims_tuple = tuple(new_dims)
        variable = self._variable.with_data(data, new_dims_tuple)
        ordered_indexes: "OrderedDict[str, BaseIndex]" = OrderedDict()
        for dim in new_dims_tuple:
            if dim in insert_indexes:
                ordered_indexes[dim] = insert_indexes[dim]
            else:
                ordered_indexes[dim] = base_indexes[dim]
        new_coords = Coordinates(ordered_indexes, extra_coords=self._coords.extra_items())
        return self._new(variable=variable, dims=new_dims_tuple, coords=new_coords)

    def squeeze(self, dims: Optional[Union[str, Sequence[str]]] = None) -> "DataTensor":
        if dims is None:
            target_dims = [dim for dim, size in zip(self._dims, self.shape) if size == 1]
        else:
            target_dims = [dims] if isinstance(dims, str) else list(dims)
        if not target_dims:
            return self

        axes = []
        for dim in target_dims:
            if dim not in self._dims:
                raise ValueError(f"Unknown dimension '{dim}'.")
            axis = self._dims.index(dim)
            if self.shape[axis] != 1:
                raise ValueError(f"Cannot squeeze dimension '{dim}' with size {self.shape[axis]}.")
            axes.append(axis)

        data = self.data
        for axis in sorted(axes, reverse=True):
            data = data.squeeze(axis)

        new_dims = tuple(dim for dim in self._dims if dim not in target_dims)
        variable = self._variable.with_data(data, new_dims)
        new_coords = self._coords.drop_dims(target_dims)
        return self._new(variable=variable, coords=new_coords, dims=new_dims)

    def assign_coords(self, **coords: CoordValue) -> "DataTensor":
        if not coords:
            return self
        dim_updates: Dict[str, BaseIndex] = {}
        extra_updates: Dict[str, CoordValue] = {}
        for dim, values in coords.items():
            if dim in self._dims:
                dim_updates[dim] = build_index(values, self.sizes[dim], dim, device=self.device)
            else:
                extra_updates[dim] = values
        new_coords = self._coords.replace(dim_indexes=dim_updates or None, extra_coords=extra_updates or None)
        return self._new(coords=new_coords)

    def rename(self, dims: Optional[Mapping[str, str]] = None, **names: str) -> "DataTensor":
        mapping = dict(dims or {})
        mapping.update(names)
        if not mapping:
            return self
        new_dims = []
        seen: set[str] = set()
        for dim in self._dims:
            new_dim = mapping.get(dim, dim)
            if new_dim in seen:
                raise ValueError(f"Duplicate dimension '{new_dim}' after rename.")
            seen.add(new_dim)
            new_dims.append(new_dim)
        renamed_coords = self._coords.rename(mapping)
        return self._new(dims=tuple(new_dims), coords=renamed_coords)

    def astype(self, dtype: Union[str, np.dtype, torch.dtype]) -> "DataTensor":
        resolved = _resolve_dtype(dtype)
        if resolved is None:
            raise TypeError(f"Unsupported dtype {dtype!r}")
        converted = self.data.to(dtype=resolved)
        variable = self._variable.with_data(converted)
        return self._new(variable=variable)

    def reset_coords(self, drop: bool = False) -> "DataTensor":
        if not drop:
            return self._new()
        cleared = Coordinates(self._coords.dim_indexes(), extra_coords=None)
        return self._new(coords=cleared)

    def to_dataarray(self):
        try:
            import xarray as xr
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("xarray must be installed to export to DataArray.") from error

        def _coord_to_numpy(values):
            if isinstance(values, torch.Tensor):
                return values.detach().cpu().numpy()
            arr = np.asarray(values)
            if arr.size:
                first = arr.reshape(-1)[0]
                if isinstance(first, np.datetime64):
                    return pd.DatetimeIndex(np.asarray(values, dtype="datetime64[ns]"))
                if isinstance(first, np.timedelta64):
                    return pd.TimedeltaIndex(np.asarray(values, dtype="timedelta64[ns]"))
            return arr

        coords = {dim: _coord_to_numpy(self._coords.coord_values(dim)) for dim in self._dims}
        for name, values in self._coords.extra_items().items():
            extra = _coord_to_numpy(values)
            arr = np.asarray(extra)
            if arr.ndim <= 1 and arr.size == 1:
                coords[name] = arr.reshape(-1)[0]
            else:
                coords[name] = extra
        data = self.data.detach().cpu().numpy()
        return xr.DataArray(data, dims=self._dims, coords=coords)

    def to_xarray(self):
        return self.to_dataarray()

    def to_pandas(self):
        import pandas as pd

        def _index_from_coords(values, name):
            if isinstance(values, torch.Tensor):
                data = values.detach().cpu().numpy()
                return pd.Index(data, name=name)
            try:
                return pd.DatetimeIndex(values, name=name)
            except (TypeError, ValueError):
                pass
            try:
                return pd.TimedeltaIndex(values, name=name)
            except (TypeError, ValueError):
                pass
            return pd.Index(np.asarray(values), name=name)

        if len(self._dims) == 1:
            dim = self._dims[0]
            index = _index_from_coords(self._coords.coord_values(dim), dim)
            data = self.data.detach().cpu().numpy()
            return pd.Series(data, index=index)

        if len(self._dims) == 2:
            row_dim, col_dim = self._dims
            index = _index_from_coords(self._coords.coord_values(row_dim), row_dim)
            columns = _index_from_coords(self._coords.coord_values(col_dim), col_dim)
            data = self.data.detach().cpu().numpy()
            return pd.DataFrame(data, index=index, columns=columns)

        raise ValueError("to_pandas only supports tensors with one or two dimensions.")

    def __getitem__(self, key: Any) -> "DataTensor":
        if isinstance(key, str):
            return self._coord_as_datatensor(key)

        if isinstance(key, Mapping):
            indexers = dict(key)
        else:
            expanded = _expanded_indexer(key, self.data.ndim)
            indexers = {dim: sel for dim, sel in zip(self._dims, expanded)}
        return self.isel(**indexers)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        try:
            return self.to_xarray().__repr__()
        except Exception:  # fallback to a lightweight summary
            coord_summary = ", ".join(f"{dim}: {len(self._coords.dim_index(dim))}" for dim in self._dims)
            return f"DataTensor(shape={self.shape}, dims={self._dims}, coords=[{coord_summary}])"

    def _repr_html_(self):
        try:
            html = self.to_xarray()._repr_html_()
        except Exception:
            return None
        if html is None:
            return None
        return html.replace("xarray.DataArray", "xtensor.DataTensor")

    # Elementwise math -------------------------------------------------
    def _binary_op(self, other: Any, op: Callable[[torch.Tensor, Any], torch.Tensor], op_name: str) -> "DataTensor":
        if isinstance(other, DataTensor):
            lhs, rhs, indexes = align_binary_operands(self, other, op_name)
            result = op(lhs.data, rhs.data)
            variable = lhs._variable.with_data(result, lhs.dims)
            coords = Coordinates(indexes, extra_coords=lhs._coords.extra_items())
            return lhs._new(variable=variable, coords=coords, dims=lhs.dims)
        else:
            result = op(self.data, other)
            variable = self._variable.with_data(result, self._dims)
            return self._new(variable=variable)

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
            reduced = op(self.data, dim=None) if allow_all_reduce else op(self.data)
            if keepdims:
                reduced = reduced.reshape([1] * self.data.ndim)
                new_dims = self._dims
            else:
                new_dims = ()
        else:
            reduced = self.data
            for axis in sorted(axes, reverse=True):
                reduced = op(reduced, dim=axis, keepdim=keepdims)
            if keepdims:
                new_dims = self._dims
            else:
                new_dims = tuple(dim for idx, dim in enumerate(self._dims) if idx not in axes_set)

        if not new_dims:
            variable = self._variable.with_data(reduced, ())
            scalar_coords = Coordinates({}, extra_coords=self._coords.extra_items())
            return self._new(variable=variable, coords=scalar_coords, dims=())
        if keepdims:
            dim_updates = {}
            for dim in reduced_dims:
                axis_index = self._coords.dim_index(dim)
                if len(axis_index) == 0:
                    dim_updates[dim] = axis_index
                    continue
                dim_updates[dim] = axis_index.take(slice(0, 1))
            new_coords = self._coords.replace(dim_indexes=dim_updates)
        else:
            retained = OrderedDict((dim, self._coords.dim_index(dim)) for dim in new_dims)
            new_coords = Coordinates(retained, extra_coords=self._coords.extra_items())
        variable = self._variable.with_data(reduced, new_dims)
        return self._new(variable=variable, coords=new_coords, dims=new_dims)

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
        new_indexes: "OrderedDict[str, BaseIndex]" = OrderedDict()

        for axis, dim in enumerate(self._dims):
            axis_index = self._coords.dim_index(dim)
            if dim in indexers:
                indexer = indexers[dim]
                normalized, subset_index, drop_dim = self._normalize_indexer(axis_index, indexer, use_coords)
                index_tuple.append(normalized)
                if not drop_dim:
                    new_dims.append(dim)
                    if subset_index is None:
                        new_indexes[dim] = axis_index
                    else:
                        new_indexes[dim] = subset_index
            else:
                index_tuple.append(slice(None))
                new_dims.append(dim)
                new_indexes[dim] = axis_index.clone()

        data = self.data[tuple(index_tuple)]
        new_dims_tuple = tuple(new_dims)
        variable = self._variable.with_data(data, new_dims_tuple)
        new_coords = Coordinates(new_indexes, extra_coords=self._coords.extra_items())
        return self._new(variable=variable, coords=new_coords, dims=new_dims_tuple)

    def _normalize_indexer(self, axis_index: BaseIndex, selector: Any, use_coords: bool):
        if isinstance(selector, slice):
            if use_coords:
                idx = axis_index.slice_indexer(selector.start, selector.stop, selector.step)
            else:
                idx = selector
            subset_index = axis_index.take(idx)
            return idx, subset_index, False

        values = _as_list(selector)

        if use_coords:
            indices = [axis_index.get_loc(val) for val in values]
        else:
            indices = [int(val) for val in values]

        if len(indices) == 1 and not isinstance(selector, (list, tuple, np.ndarray, torch.Tensor)):
            idx_value = indices[0]
            return idx_value, None, True

        tensor_index = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        subset_index = axis_index.take(tensor_index)
        return tensor_index, subset_index, False

    def item(self) -> Any:
        if self.data.numel() != 1:
            raise ValueError("Only scalar DataTensor instances support .item().")
        return self.data.item()

    def _new(
        self,
        *,
        variable: Optional[Variable] = None,
        coords: Optional[Coordinates] = None,
        dims: Optional[Sequence[str]] = None,
        attrs: Optional[Mapping[str, Any]] = None,
    ) -> "DataTensor":
        obj = self.__class__.__new__(self.__class__)
        base_variable = variable if variable is not None else self._variable
        if dims is not None and variable is None:
            base_variable = base_variable.with_dims(dims)
        obj._variable = base_variable
        obj._dims = obj._variable.dims
        if coords is not None:
            obj._coords = coords.copy()
        else:
            obj._coords = self._coords.copy()
        obj._attrs = dict(attrs) if attrs is not None else dict(self._attrs)
        return obj

    def _dim_index_map(self) -> Dict[str, BaseIndex]:
        return self._coords.dim_indexes()

    def _get_index(self, dim: str) -> BaseIndex:
        return self._coords.dim_index(dim)

    def _indexes_copy(self) -> Dict[str, BaseIndex]:
        return self._coords.dim_indexes()

    def _coordinates_copy(self) -> Coordinates:
        return self._coords.copy()

    def _extra_coords(self) -> Mapping[str, CoordArray]:
        return self._coords.extra_items()

    def _coord_as_datatensor(self, name: str) -> "DataTensor":
        if not self._coords.has_coord(name):
            raise KeyError(name)
        values = self._coords.coord_values(name)
        if isinstance(values, torch.Tensor):
            data = values.clone()
        else:
            try:
                data = torch.as_tensor(list(values))
            except (TypeError, ValueError, RuntimeError):
                return values
        return DataTensor(data, {name: values}, (name,))


def _binary_elementwise(name: str, op: Callable[[torch.Tensor, Any], torch.Tensor], reverse_op: Callable[[torch.Tensor, Any], torch.Tensor], a: Any, b: Any) -> "DataTensor":
    if isinstance(a, DataTensor):
        return a._binary_op(b, op, name)
    if isinstance(b, DataTensor):
        return b._binary_op(a, reverse_op, name)
    return NotImplemented


def _unary_elementwise(name: str, op: Callable[[torch.Tensor], torch.Tensor], operand: Any) -> "DataTensor":
    if not isinstance(operand, DataTensor):
        return NotImplemented
    result = _disable_torch_function_call(op, operand.data)
    variable = operand._variable.with_data(result)
    return operand._new(variable=variable)


def _normalize_torch_dims(dim_arg: Optional[Union[int, str, Sequence[Union[int, str]]]], dims: Tuple[str, ...]) -> Optional[Union[str, Tuple[str, ...]]]:
    if dim_arg is None:
        return None

    def _convert(single: Union[int, str]) -> str:
        if isinstance(single, int):
            if not dims:
                raise ValueError("Cannot apply dimension-based reduction on scalar DataTensor.")
            index = single % len(dims)
            return dims[index]
        return single

    if isinstance(dim_arg, (list, tuple)):
        converted = tuple(_convert(item) for item in dim_arg)
        # collapse single-entry tuples into str for compatibility
        if len(converted) == 1:
            return converted[0]
        return converted
    return _convert(dim_arg)


def _cast_dtype_if_needed(tensor: DataTensor, dtype: Optional[Union[str, np.dtype, torch.dtype, type]]) -> DataTensor:
    if dtype is None:
        return tensor
    resolved = _resolve_dtype(dtype)
    if resolved is None or tensor.data.dtype == resolved:
        return tensor
    return tensor.astype(resolved)


@_implements(torch.add, torch.Tensor.add)
def _torch_add(input: Any, other: Any, *, alpha: Any = 1, out: Optional[Any] = None):
    _ensure_out_argument_supported(out)

    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.add, lhs, rhs, alpha=alpha)

    return _binary_elementwise("add", op, op, input, other)


@_implements(torch.sub, torch.Tensor.sub)
def _torch_sub(input: Any, other: Any, *, alpha: Any = 1, out: Optional[Any] = None):
    _ensure_out_argument_supported(out)

    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.sub, lhs, rhs, alpha=alpha)

    def reverse(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.sub, rhs, lhs, alpha=alpha)

    return _binary_elementwise("sub", op, reverse, input, other)


@_implements(torch.mul, torch.Tensor.mul)
def _torch_mul(input: Any, other: Any, *, out: Optional[Any] = None):
    _ensure_out_argument_supported(out)

    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.mul, lhs, rhs)

    return _binary_elementwise("mul", op, op, input, other)


@_implements(torch.div, torch.Tensor.div, torch.divide, torch.Tensor.divide)
def _torch_div(input: Any, other: Any, *, rounding_mode: Optional[str] = None, out: Optional[Any] = None):
    _ensure_out_argument_supported(out)

    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.div, lhs, rhs, rounding_mode=rounding_mode)

    def reverse(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.div, rhs, lhs, rounding_mode=rounding_mode)

    return _binary_elementwise("div", op, reverse, input, other)


@_implements(torch.true_divide, torch.Tensor.true_divide)
def _torch_true_divide(input: Any, other: Any, *, out: Optional[Any] = None):
    _ensure_out_argument_supported(out)

    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.true_divide, lhs, rhs)

    def reverse(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.true_divide, rhs, lhs)

    return _binary_elementwise("truediv", op, reverse, input, other)


@_implements(torch.pow, torch.Tensor.pow)
def _torch_pow(input: Any, exponent: Any, *, out: Optional[Any] = None):
    _ensure_out_argument_supported(out)

    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.pow, lhs, rhs)

    def reverse(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.pow, rhs, lhs)

    return _binary_elementwise("pow", op, reverse, input, exponent)


@_implements(torch.remainder, torch.Tensor.remainder)
def _torch_remainder(input: Any, other: Any, *, out: Optional[Any] = None):
    _ensure_out_argument_supported(out)

    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.remainder, lhs, rhs)

    def reverse(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.remainder, rhs, lhs)

    return _binary_elementwise("remainder", op, reverse, input, other)


@_implements(torch.minimum, torch.Tensor.minimum)
def _torch_minimum(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.minimum, lhs, rhs)

    return _binary_elementwise("minimum", op, op, input, other)


@_implements(torch.maximum, torch.Tensor.maximum)
def _torch_maximum(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.maximum, lhs, rhs)

    return _binary_elementwise("maximum", op, op, input, other)


@_implements(torch.neg, torch.Tensor.neg)
def _torch_neg(input: Any):
    return _unary_elementwise("neg", torch.neg, input)


@_implements(torch.abs, torch.Tensor.abs)
def _torch_abs(input: Any):
    return _unary_elementwise("abs", torch.abs, input)


@_implements(torch.sum, torch.Tensor.sum)
def _torch_sum(input: Any, dim: Optional[Any] = None, keepdim: bool = False, dtype: Optional[Any] = None, out: Optional[Any] = None):
    if not isinstance(input, DataTensor):
        return NotImplemented
    _ensure_out_argument_supported(out)
    tensor = _cast_dtype_if_needed(input, dtype)
    dims = _normalize_torch_dims(dim, tensor.dims)
    return tensor.sum(dim=dims, keepdims=keepdim)


@_implements(torch.mean, torch.Tensor.mean)
def _torch_mean(input: Any, dim: Optional[Any] = None, keepdim: bool = False, dtype: Optional[Any] = None, out: Optional[Any] = None):
    if not isinstance(input, DataTensor):
        return NotImplemented
    _ensure_out_argument_supported(out)
    tensor = _cast_dtype_if_needed(input, dtype)
    dims = _normalize_torch_dims(dim, tensor.dims)
    return tensor.mean(dim=dims, keepdims=keepdim)


@_implements(torch.prod, torch.Tensor.prod)
def _torch_prod(input: Any, dim: Optional[Any] = None, keepdim: bool = False, dtype: Optional[Any] = None, out: Optional[Any] = None):
    if not isinstance(input, DataTensor):
        return NotImplemented
    _ensure_out_argument_supported(out)
    tensor = _cast_dtype_if_needed(input, dtype)
    dims = _normalize_torch_dims(dim, tensor.dims)
    return tensor.prod(dim=dims, keepdims=keepdim)


@_implements(torch.std, torch.Tensor.std)
def _torch_std(input: Any, dim: Optional[Any] = None, unbiased: bool = True, keepdim: bool = False, out: Optional[Any] = None):
    if not isinstance(input, DataTensor):
        return NotImplemented
    _ensure_out_argument_supported(out)
    dims = _normalize_torch_dims(dim, input.dims)
    return input.std(dim=dims, keepdims=keepdim, unbiased=unbiased)


@_implements(torch.amin, torch.Tensor.amin)
def _torch_amin(input: Any, dim: Optional[Any] = None, keepdim: bool = False):
    if not isinstance(input, DataTensor):
        return NotImplemented
    dims = _normalize_torch_dims(dim, input.dims)
    return input.min(dim=dims, keepdims=keepdim)


@_implements(torch.amax, torch.Tensor.amax)
def _torch_amax(input: Any, dim: Optional[Any] = None, keepdim: bool = False):
    if not isinstance(input, DataTensor):
        return NotImplemented
    dims = _normalize_torch_dims(dim, input.dims)
    return input.max(dim=dims, keepdims=keepdim)
