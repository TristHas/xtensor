from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

import numpy as np
import torch
import torch.nn.functional as F

from .alignment import _broadcast_tensor, _merge_dim_indexes, align_binary_operands
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
    if isinstance(value, DataTensor):
        data = value.data.detach().cpu()
        if data.ndim != 1:
            data = data.reshape(-1)
        return data.tolist()
    if isinstance(value, torch.Tensor):
        tensor = value.cpu()
        if tensor.ndim != 1:
            tensor = tensor.reshape(-1)
        return tensor.tolist()
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim != 1:
            array = array.reshape(-1)
        return array.tolist()
    if pd is not None and isinstance(value, pd.Index):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_coord_values(values: Any, size: int) -> CoordValue:
    if isinstance(values, torch.Tensor):
        tensor = values.reshape(-1)
        if tensor.shape[0] != size:
            raise ValueError(f"Coordinate length mismatch. Expected {size}, received {tensor.shape[0]}")
        return tensor.clone()
    if pd is not None and isinstance(values, pd.Index):
        if len(values) != size:
            raise ValueError(f"Coordinate length mismatch. Expected {size}, received {len(values)}")
        if _pandas_index_is_numeric(values):
            return torch.as_tensor(values.to_numpy())
        return values.copy()
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != size:
        raise ValueError(f"Coordinate length mismatch. Expected {size}, received {array.shape[0]}")
    if array.dtype.kind in ("b", "i", "u", "f"):
        return torch.as_tensor(array)
    if pd is None:
        raise RuntimeError("pandas is required for non-numeric coordinate values.")
    return pd.Index(array)


def _pandas_index_is_numeric(index: "pd.Index") -> bool:
    dtype = index.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return True
    return pd.api.types.is_numeric_dtype(dtype)


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


def _supports_nan(dtype: torch.dtype) -> bool:
    return dtype.is_floating_point or dtype.is_complex()


def _nanmean_op(data: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
    if not _supports_nan(data.dtype):
        return torch.mean(data, dim=dim, keepdim=keepdim)
    return torch.nanmean(data, dim=dim, keepdim=keepdim)


def _nanvar_impl(data: torch.Tensor, dim: Optional[int], keepdim: bool, unbiased: bool) -> torch.Tensor:
    if not _supports_nan(data.dtype):
        return torch.var(data, dim=dim, keepdim=keepdim, unbiased=unbiased)
    mask = ~torch.isnan(data)
    safe = torch.where(mask, data, torch.zeros_like(data))
    min_count = 2 if unbiased else 1
    if dim is None:
        valid = int(mask.sum().item())
        if valid < min_count:
            return data.new_full((), float("nan"))
        count = torch.as_tensor(valid, dtype=data.dtype, device=data.device)
        total = safe.sum()
        mean = total / count
        centered = torch.where(mask, data - mean, torch.zeros_like(data))
        sumsq = (centered.conj() * centered).sum()
        denom = count - (1 if unbiased else 0)
        return sumsq / denom
    count = mask.sum(dim=dim, keepdim=True).to(data.dtype)
    valid = count >= min_count
    safe_count = torch.where(valid, count, torch.ones_like(count))
    total = safe.sum(dim=dim, keepdim=True)
    mean = total / safe_count
    centered = torch.where(mask, data - mean, torch.zeros_like(data))
    sumsq = (centered.conj() * centered).sum(dim=dim, keepdim=True)
    denom = safe_count - (1 if unbiased else 0)
    denom = torch.clamp(denom, min=1)
    var = sumsq / denom
    nan_fill = var.new_full(var.shape, float("nan"))
    var = torch.where(valid, var, nan_fill)
    if not keepdim:
        var = var.squeeze(dim)
    return var

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
        name: Optional[str] = None,
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
        self._name: Optional[str] = name

    @property
    def dtype(self) -> torch.Tensor:
        return self._variable.dtype

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
        return DataTensor(grad, self.coords, self._dims, name=self._name)

    def retain_grad(self) -> "DataTensor":
        """Retain gradient information on the underlying tensor."""
        self._variable.data.retain_grad()
        return self

    def require_grad(self, requires_grad: bool = True) -> "DataTensor":
        """In-place requires_grad flag that mirrors torch.Tensor.requires_grad_."""
        self._variable.data.requires_grad_(requires_grad)
        return self

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

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        self._name = value

    @staticmethod
    def from_pandas(obj: Any, dims: Optional[Sequence[str]] = None) -> "DataTensor":
        if pd is None:  # pragma: no cover - defensive
            raise RuntimeError("pandas must be installed to construct a DataTensor from pandas objects.")

        if isinstance(obj, pd.Series):
            dim = (dims[0] if dims else obj.index.name) or "index"
            data = torch.as_tensor(obj.to_numpy())
            coords = {dim: _normalize_coord_values(obj.index, data.shape[0])}
            return DataTensor(data, coords, (dim,), name=obj.name)

        if isinstance(obj, pd.DataFrame):
            dims = dims or (obj.index.name or "index", obj.columns.name or "columns")
            if len(dims) != 2:
                raise ValueError("DataFrame conversion expects exactly two dims.")
            data = torch.as_tensor(obj.to_numpy())
            coords = {
                dims[0]: _normalize_coord_values(obj.index, data.shape[0]),
                dims[1]: _normalize_coord_values(obj.columns, data.shape[1]),
            }
            return DataTensor(data, coords, tuple(dims), name=getattr(obj, "name", None))

        raise TypeError("from_pandas expects a pandas Series or DataFrame.")

    @staticmethod
    def from_dataarray(array: Any) -> "DataTensor":
        try:
            import xarray as xr  # noqa: F401
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("xarray must be installed to build from a DataArray.") from error

        dims = tuple(array.dims)
        coords = {}
        for axis, dim in enumerate(dims):
            coord_var = array.coords[dim]
            size = array.shape[axis]
            if pd is not None and hasattr(coord_var, "to_index"):
                source = coord_var.to_index()
            else:
                source = coord_var.to_numpy()
            coords[dim] = _normalize_coord_values(source, size)
        return DataTensor(array.data, coords, dims, attrs=getattr(array, "attrs", None), name=getattr(array, "name", None))

    def sel(self, **indexers: Any) -> "DataTensor":
        return self._select(indexers, use_coords=True)

    def isel(self, **indexers: Any) -> "DataTensor":
        return self._select(indexers, use_coords=False)

    def mean(self, dim: Optional[Union[str, Sequence[str]]] = None, keepdims: bool = False) -> "DataTensor":
        return self._reduce(_nanmean_op, dim=dim, keepdims=keepdims, allow_all_reduce=True)

    def std(self, dim: Optional[Union[str, Sequence[str]]] = None, keepdims: bool = False, unbiased: bool = False) -> "DataTensor":
        def _nanstd(data: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
            variance = _nanvar_impl(data, dim, True, unbiased)
            std = torch.sqrt(variance)
            if dim is not None and not keepdim:
                std = std.squeeze(dim)
            return std

        return self._reduce(_nanstd, dim=dim, keepdims=keepdims, allow_all_reduce=True)

    def var(self, dim: Optional[Union[str, Sequence[str]]] = None, keepdims: bool = False, unbiased: bool = False) -> "DataTensor":
        def _nanvar(data: torch.Tensor, axis: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
            def _apply(values: torch.Tensor, reduction_dim: Optional[int], keep: bool) -> torch.Tensor:
                mask = torch.isfinite(values)
                mean = torch.nanmean(values, dim=reduction_dim, keepdim=True)
                centered = torch.where(mask, values - mean, torch.zeros_like(values))
                sumsq = torch.nansum(centered ** 2, dim=reduction_dim, keepdim=True)
                counts = mask.sum(dim=reduction_dim, keepdim=True).to(values.dtype)
                if unbiased:
                    counts = torch.clamp(counts - 1, min=1)
                counts = torch.clamp(counts, min=1)
                variance = sumsq / counts
                if not keep:
                    variance = variance.squeeze(dim=reduction_dim)
                return variance

            if axis is None:
                return _apply(data.reshape(-1), None, keepdim)
            return _apply(data, axis, keepdim)

        return self._reduce(_nanvar, dim=dim, keepdims=keepdims, allow_all_reduce=True)

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

    def any(self, dim: Optional[Union[str, Sequence[str]]] = None, keepdims: bool = False) -> "DataTensor":
        return self._reduce(torch.any, dim=dim, keepdims=keepdims)

    def var(self, dim: Optional[Union[str, Sequence[str]]] = None, keepdims: bool = False, unbiased: bool = False) -> "DataTensor":
        def _nanvar(data: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
            return _nanvar_impl(data, dim, keepdim, unbiased)

        return self._reduce(_nanvar, dim=dim, keepdims=keepdims, allow_all_reduce=True)

    def to(self, device=None, dtype=None, **kwargs: Any) -> "DataTensor":
        moved = self.data.to(device=device, dtype=dtype, **kwargs)
        variable = self._variable.with_data(moved)
        moved_coords = self._coords.to(device=device, **kwargs)
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

    def squeeze(self, dims: Optional[Union[str, Sequence[str]]] = None, drop: bool = False) -> "DataTensor":
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
        if drop and target_dims:
            new_coords = new_coords.replace(drop_extra=target_dims)
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
        attrs = dict(self._attrs)
        return xr.DataArray(data, dims=self._dims, coords=coords, name=self._name, attrs=attrs)

    def to_xarray(self):
        return self.to_dataarray()

    @property
    def plot(self):
        try:
            data_array = self.to_dataarray()
        except Exception as error:  # pragma: no cover - defensive
            raise RuntimeError("Plotting requires xarray to be installed.") from error
        return data_array.plot

    @property
    def hvplot(self):
        try:
            import hvplot.xarray  # noqa: F401
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("hvplot must be installed to use DataTensor.hvplot") from error
        if self.name is None:
            name = "DataTensor"
        else:
            name = self.name
        return self.to_dataarray(name=name).hvplot

    def to_dataset(
        self,
        dim: Optional[str] = None,
        *,
        name: Optional[str] = None,
        promote_attrs: bool = False,
    ) -> "Dataset":
        from .dataset import Dataset  # local import to avoid circular dependency

        if dim is not None and name is not None:
            raise TypeError("cannot supply both dim and name arguments")

        attrs = self.attrs if promote_attrs else None

        if dim is None:
            var_name = name if name is not None else self.name
            if var_name is None:
                raise ValueError("unable to convert unnamed DataTensor to a Dataset without providing an explicit name.")
            return Dataset({var_name: self}, attrs=attrs)

        if dim not in self._dims:
            raise ValueError(f"Dimension '{dim}' not present in DataTensor.")

        coord_values = self.coords[dim]
        labels = list(_as_list(coord_values))
        data_vars: "OrderedDict[Any, DataTensor]" = OrderedDict()
        for idx, label in enumerate(labels):
            data_vars[label] = self.isel(**{dim: idx})
        return Dataset(data_vars, attrs=attrs)

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
        #try:
        #    return self.to_xarray().__repr__()
        #except Exception:  # fallback to a lightweight summary
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

    def __pow__(self, other: Any) -> "DataTensor":
        return self._binary_op(other, torch.pow, "pow")

    def __rpow__(self, other: Any) -> "DataTensor":
        return self._binary_op(other, lambda lhs, rhs: torch.pow(rhs, lhs), "rpow")

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
            tensor_index = axis_index.get_indexer(values, device=self.device)
        else:
            tensor_index = torch.as_tensor(values, dtype=torch.long, device=self.device)

        if tensor_index.numel() == 1 and not isinstance(selector, (list, tuple, np.ndarray, torch.Tensor)):
            idx_value = int(tensor_index.item())
            return idx_value, None, True

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
        name: Optional[str] = None,
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
        obj._name = self._name if name is None else name
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


def concat(
    objects: Sequence[Any],
    dim: Optional[Union[str, Tuple[str, CoordValue], DataTensor, CoordValue]] = None,
    *,
    coords: str = "different",
) -> Any:
    """Concatenate DataTensors or Datasets along a dimension, similar to ``xarray.concat``."""
    items = list(objects)
    if not items:
        raise ValueError("concat expects at least one object.")
    if coords not in {"different", "minimal"}:
        raise NotImplementedError("concat currently supports coords='different' or coords='minimal'.")

    first = items[0]
    if isinstance(first, DataTensor):
        if not all(isinstance(obj, DataTensor) for obj in items):
            raise TypeError("concat requires inputs to be all DataTensor or all Dataset instances.")
        return _concat_data_tensors(items, dim, coords)

    try:  # Lazy import to avoid circular dependency during module loading.
        from .dataset import Dataset  # type: ignore
    except ImportError:  # pragma: no cover - defensive
        Dataset = None

    if Dataset is not None and isinstance(first, Dataset):
        if not all(isinstance(obj, Dataset) for obj in items):
            raise TypeError("concat requires inputs to be all DataTensor or all Dataset instances.")
        return _concat_datasets(items, dim, coords)

    raise TypeError("concat expects DataTensor or Dataset sequences.")


def _concat_data_tensors(
    tensors: Sequence[DataTensor],
    dim: Optional[Union[str, Tuple[str, CoordValue], DataTensor, CoordValue]],
    coords: str,
) -> DataTensor:
    tensors = list(tensors)
    target_dim, coord_provider = _parse_concat_dim_argument(dim)
    has_dim = all(target_dim in tensor.dims for tensor in tensors)
    has_any_dim = any(target_dim in tensor.dims for tensor in tensors)

    if has_dim:
        aligned = _align_concat_tensors(tensors, exclude_dim=target_dim)
        return _concat_existing_dim(aligned, target_dim, coord_provider, coords)
    if has_any_dim:
        raise ValueError(f"Dimension '{target_dim}' is present in only a subset of tensors.")
    aligned = _align_concat_tensors(tensors)
    return _concat_new_dim(aligned, target_dim, coord_provider, coords)


def _concat_datasets(
    datasets: Sequence["Dataset"],
    dim: Optional[Union[str, Tuple[str, CoordValue], DataTensor, CoordValue]],
    coords: str,
) -> "Dataset":
    from .dataset import Dataset  # Local import to avoid cyclic initialization issues.

    mapped = [ds.data_vars for ds in datasets]
    if not mapped:
        raise ValueError("concat expects at least one Dataset.")  # pragma: no cover - defensive
    base_names = list(mapped[0].keys())
    base_set = set(base_names)
    for current in mapped[1:]:
        names = set(current.keys())
        if names != base_set:
            raise ValueError("All Dataset inputs must define the same data variables.")
    new_vars: "OrderedDict[str, DataTensor]" = OrderedDict()
    for name in base_names:
        pieces = [current[name] for current in mapped]
        new_vars[name] = _concat_data_tensors(pieces, dim, coords)
    shared_coords = _shared_dataset_extra_coords(datasets, coords)
    attrs = datasets[0].attrs
    return Dataset(new_vars, coords=shared_coords or None, attrs=attrs)


def _parse_concat_dim_argument(
    dim: Optional[Union[str, Tuple[str, CoordValue], DataTensor, CoordValue]],
) -> Tuple[str, Optional[Callable[[int], CoordValue]]]:
    if dim is None:
        return "concat_dim", None

    if isinstance(dim, str):
        return dim, None

    if isinstance(dim, DataTensor):
        if dim.data.ndim != 1:
            raise ValueError("The dimension specification DataTensor must be 1D.")
        name = dim.dims[0] if dim.dims else "concat_dim"

        def _provider(size: int) -> CoordValue:
            if dim.data.shape[0] != size:
                raise ValueError(f"Expected concat dim of length {size}, received {dim.data.shape[0]}.")
            return _normalize_coord_values(dim.data, size)

        return name, _provider

    if isinstance(dim, tuple) and len(dim) == 2 and isinstance(dim[0], str):
        name = dim[0]
        raw_values = dim[1]

        def _provider(size: int) -> CoordValue:
            return _normalize_coord_values(raw_values, size)

        return name, _provider

    if pd is not None and isinstance(dim, pd.Index):
        name = dim.name or "concat_dim"

        def _provider(size: int) -> CoordValue:
            return _normalize_coord_values(dim, size)

        return name, _provider

    # Treat remaining inputs as coordinate values with implicit dimension.
    if isinstance(dim, (list, tuple, np.ndarray, torch.Tensor)) and not isinstance(dim, str):
        raw_values = dim

        def _provider(size: int) -> CoordValue:
            return _normalize_coord_values(raw_values, size)

        return "concat_dim", _provider

    raise TypeError("dim must be a string, (name, values) tuple, DataTensor, or coordinate values.")


def _concat_existing_dim(
    tensors: Sequence[DataTensor],
    dim: str,
    coord_provider: Optional[Callable[[int], CoordValue]],
    coords: str,
) -> DataTensor:
    base = tensors[0]

    axis = base.dims.index(dim)
    pieces = [tensor.data for tensor in tensors]
    concatenated = torch.cat(pieces, dim=axis)

    total_size = concatenated.shape[axis]
    if coord_provider is not None:
        coord_values = coord_provider(total_size)
    else:
        indexes = [tensor._coords.dim_index(dim) for tensor in tensors]
        coord_values = _concat_index_values(indexes)
    dim_index = build_index(coord_values, total_size, dim, device=base.device)

    dim_indexes = OrderedDict(base._coords.dim_indexes())
    dim_indexes[dim] = dim_index
    extra_coords = _common_extra_coords(tensors, coords)
    coordinates = Coordinates(dim_indexes, extra_coords=extra_coords or None)
    variable = base._variable.with_data(concatenated)
    return base._new(variable=variable, coords=coordinates)


def _concat_new_dim(
    tensors: Sequence[DataTensor],
    dim: str,
    coord_provider: Optional[Callable[[int], CoordValue]],
    coords: str,
) -> DataTensor:
    base = tensors[0]
    stacked = torch.stack([tensor.data for tensor in tensors], dim=0)
    result_dims = (dim,) + base.dims

    total_size = len(tensors)
    if coord_provider is not None:
        coord_values = coord_provider(total_size)
    else:
        coord_values = torch.arange(total_size, dtype=torch.float64, device=base.device)
    concat_index = build_index(coord_values, total_size, dim, device=base.device)

    base_indexes = base._coords.dim_indexes()
    dim_indexes = OrderedDict([(dim, concat_index)])
    dim_indexes.update(base_indexes)
    extra_coords = _common_extra_coords(tensors, coords)
    coordinates = Coordinates(dim_indexes, extra_coords=extra_coords or None)
    variable = base._variable.with_data(stacked, result_dims)
    return base._new(variable=variable, coords=coordinates, dims=result_dims)


def _align_concat_tensors(
    tensors: Sequence[DataTensor],
    *,
    exclude_dim: Optional[str] = None,
) -> Sequence[DataTensor]:
    if len(tensors) <= 1:
        return tensors

    target_dims: list[str] = list(tensors[0].dims)
    merged_indexes: "OrderedDict[str, BaseIndex]" = OrderedDict(tensors[0]._dim_index_map())

    for tensor in tensors[1:]:
        for dim in tensor.dims:
            if dim not in target_dims:
                target_dims.append(dim)

    merge_dims = tuple(dim for dim in target_dims if dim != exclude_dim)
    merged_indexes = OrderedDict(
        (dim, index) for dim, index in merged_indexes.items() if dim != exclude_dim
    )
    for tensor in tensors[1:]:
        subset = OrderedDict(
            (dim, index) for dim, index in tensor._dim_index_map().items() if dim != exclude_dim
        )
        merged_indexes = OrderedDict(
            _merge_dim_indexes(merged_indexes, subset, merge_dims, "concat")
        )

    dims_tuple = tuple(target_dims)
    aligned = []
    for tensor in tensors:
        tensor_indexes = tensor._dim_index_map()
        target_indexes: "OrderedDict[str, BaseIndex]" = OrderedDict()
        for dim in dims_tuple:
            if dim == exclude_dim:
                target_indexes[dim] = tensor_indexes[dim]
            else:
                target_indexes[dim] = merged_indexes[dim]
        aligned.append(_broadcast_tensor(tensor, dims_tuple, target_indexes))
    return aligned


def _concat_index_values(indexes: Sequence[BaseIndex]) -> CoordValue:
    first = indexes[0].coord_array()
    if isinstance(first, torch.Tensor):
        arrays = [index.coord_array() for index in indexes]
        return torch.cat(arrays, dim=0)
    if pd is not None and isinstance(first, pd.Index):
        combined = first
        for index in indexes[1:]:
            combined = combined.append(index.coord_array())  # type: ignore[call-arg]
        return combined
    values: list[Any] = []
    for index in indexes:
        values.extend(index.coord_array())
    return values


def _common_extra_coords(tensors: Sequence[DataTensor], coords: str) -> Mapping[str, CoordArray]:
    base_extras = tensors[0]._coords.extra_items()
    if coords == "minimal" or not base_extras:
        return {}
    retained: "OrderedDict[str, CoordArray]" = OrderedDict()
    for name, values in base_extras.items():
        keep = True
        for tensor in tensors[1:]:
            extras = tensor._coords.extra_items()
            other = extras.get(name)
            if not _coord_values_equal(values, other):
                keep = False
                break
        if keep:
            retained[name] = values
    return retained


def _coord_values_equal(lhs: Optional[CoordArray], rhs: Optional[CoordArray]) -> bool:
    if lhs is None or rhs is None:
        return False
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        if lhs.dtype.is_floating_point or rhs.dtype.is_floating_point:
            return torch.allclose(lhs, rhs)
        return torch.equal(lhs, rhs)
    if pd is not None and isinstance(lhs, pd.Index) and isinstance(rhs, pd.Index):
        return lhs.equals(rhs)
    return tuple(lhs) == tuple(rhs)


def _shared_dataset_extra_coords(datasets: Sequence["Dataset"], coords: str) -> Mapping[str, CoordValue]:
    if coords == "minimal":
        return {}
    from .dataset import Dataset  # Local import; typing aid.

    base = datasets[0]
    base_coords = base.coords
    dim_names = set(base.dims.keys())
    retained: "OrderedDict[str, CoordValue]" = OrderedDict()
    for name in base_coords:
        if name in dim_names:
            continue
        base_value = base_coords[name]
        keep = True
        for other in datasets[1:]:
            if name in other.dims or name not in other.coords:
                keep = False
                break
            other_value = other.coords[name]
            if not _coord_values_equal(base_value, other_value):
                keep = False
                break
        if keep:
            retained[name] = base_value
    return retained


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


@_implements(torch.sin, torch.Tensor.sin)
def _torch_sin(input: Any):
    return _unary_elementwise("sin", torch.sin, input)


@_implements(torch.cos, torch.Tensor.cos)
def _torch_cos(input: Any):
    return _unary_elementwise("cos", torch.cos, input)


@_implements(torch.tan, torch.Tensor.tan)
def _torch_tan(input: Any):
    return _unary_elementwise("tan", torch.tan, input)


@_implements(torch.asin, torch.Tensor.asin)
def _torch_asin(input: Any):
    return _unary_elementwise("asin", torch.asin, input)


@_implements(torch.acos, torch.Tensor.acos)
def _torch_acos(input: Any):
    return _unary_elementwise("acos", torch.acos, input)


@_implements(torch.atan, torch.Tensor.atan)
def _torch_atan(input: Any):
    return _unary_elementwise("atan", torch.atan, input)


@_implements(torch.sinh, torch.Tensor.sinh)
def _torch_sinh(input: Any):
    return _unary_elementwise("sinh", torch.sinh, input)


@_implements(torch.cosh, torch.Tensor.cosh)
def _torch_cosh(input: Any):
    return _unary_elementwise("cosh", torch.cosh, input)


@_implements(torch.tanh, torch.Tensor.tanh)
def _torch_tanh(input: Any):
    return _unary_elementwise("tanh", torch.tanh, input)


@_implements(torch.asinh, torch.Tensor.asinh)
def _torch_asinh(input: Any):
    return _unary_elementwise("asinh", torch.asinh, input)


@_implements(torch.acosh, torch.Tensor.acosh)
def _torch_acosh(input: Any):
    return _unary_elementwise("acosh", torch.acosh, input)


@_implements(torch.atanh, torch.Tensor.atanh)
def _torch_atanh(input: Any):
    return _unary_elementwise("atanh", torch.atanh, input)


@_implements(torch.exp, torch.Tensor.exp)
def _torch_exp(input: Any):
    return _unary_elementwise("exp", torch.exp, input)


@_implements(torch.expm1, torch.Tensor.expm1)
def _torch_expm1(input: Any):
    return _unary_elementwise("expm1", torch.expm1, input)


@_implements(torch.log, torch.Tensor.log)
def _torch_log(input: Any):
    return _unary_elementwise("log", torch.log, input)


@_implements(torch.log10, torch.Tensor.log10)
def _torch_log10(input: Any):
    return _unary_elementwise("log10", torch.log10, input)


@_implements(torch.log1p, torch.Tensor.log1p)
def _torch_log1p(input: Any):
    return _unary_elementwise("log1p", torch.log1p, input)


@_implements(torch.sqrt, torch.Tensor.sqrt)
def _torch_sqrt(input: Any):
    return _unary_elementwise("sqrt", torch.sqrt, input)


@_implements(torch.rsqrt, torch.Tensor.rsqrt)
def _torch_rsqrt(input: Any):
    return _unary_elementwise("rsqrt", torch.rsqrt, input)


@_implements(torch.square, torch.Tensor.square)
def _torch_square(input: Any):
    return _unary_elementwise("square", torch.square, input)


@_implements(torch.reciprocal, torch.Tensor.reciprocal)
def _torch_reciprocal(input: Any):
    return _unary_elementwise("reciprocal", torch.reciprocal, input)


@_implements(torch.floor, torch.Tensor.floor)
def _torch_floor(input: Any):
    return _unary_elementwise("floor", torch.floor, input)


@_implements(torch.ceil, torch.Tensor.ceil)
def _torch_ceil(input: Any):
    return _unary_elementwise("ceil", torch.ceil, input)


@_implements(torch.trunc, torch.Tensor.trunc)
def _torch_trunc(input: Any):
    return _unary_elementwise("trunc", torch.trunc, input)


@_implements(torch.round, torch.Tensor.round)
def _torch_round(input: Any):
    return _unary_elementwise("round", torch.round, input)


@_implements(torch.frac, torch.Tensor.frac)
def _torch_frac(input: Any):
    return _unary_elementwise("frac", torch.frac, input)


@_implements(torch.sigmoid, torch.Tensor.sigmoid)
def _torch_sigmoid(input: Any):
    return _unary_elementwise("sigmoid", torch.sigmoid, input)


@_implements(torch.relu, torch.Tensor.relu)
def _torch_relu(input: Any):
    return _unary_elementwise("relu", torch.relu, input)


@_implements(torch.sign, torch.Tensor.sign)
def _torch_sign(input: Any):
    return _unary_elementwise("sign", torch.sign, input)


@_implements(torch.signbit, torch.Tensor.signbit)
def _torch_signbit(input: Any):
    return _unary_elementwise("signbit", torch.signbit, input)


@_implements(torch.logical_not, torch.Tensor.logical_not)
def _torch_logical_not(input: Any):
    return _unary_elementwise("logical_not", torch.logical_not, input)


@_implements(torch.bitwise_not, torch.Tensor.bitwise_not)
def _torch_bitwise_not(input: Any):
    return _unary_elementwise("bitwise_not", torch.bitwise_not, input)


@_implements(torch.isfinite, torch.Tensor.isfinite)
def _torch_isfinite(input: Any):
    return _unary_elementwise("isfinite", torch.isfinite, input)


@_implements(torch.isinf, torch.Tensor.isinf)
def _torch_isinf(input: Any):
    return _unary_elementwise("isinf", torch.isinf, input)


@_implements(torch.isreal, torch.Tensor.isreal)
def _torch_isreal(input: Any):
    return _unary_elementwise("isreal", torch.isreal, input)


@_implements(F.softplus)
def _torch_softplus(input: Any, beta: float = 1.0, threshold: float = 20.0):
    if not isinstance(input, DataTensor):
        return NotImplemented
    result = _disable_torch_function_call(F.softplus, input.data, beta=beta, threshold=threshold)
    variable = input._variable.with_data(result)
    return input._new(variable=variable)


@_implements(F.softsign)
def _torch_softsign(input: Any):
    if not isinstance(input, DataTensor):
        return NotImplemented
    result = _disable_torch_function_call(F.softsign, input.data)
    variable = input._variable.with_data(result)
    return input._new(variable=variable)


@_implements(torch.atan2, torch.Tensor.atan2)
def _torch_atan2(input: Any, other: Any, *, out: Optional[Any] = None):
    _ensure_out_argument_supported(out)

    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.atan2, lhs, rhs)

    def reverse(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.atan2, rhs, lhs)

    return _binary_elementwise("atan2", op, reverse, input, other)


@_implements(torch.isnan, torch.Tensor.isnan)
def _torch_isnan(input: Any):
    return _unary_elementwise("isnan", torch.isnan, input)


@_implements(torch.nan_to_num, torch.Tensor.nan_to_num)
def _torch_nan_to_num(input: Any, nan: float = 0.0, posinf: Optional[float] = None, neginf: Optional[float] = None, *, out: Optional[Any] = None):
    if not isinstance(input, DataTensor):
        return NotImplemented
    _ensure_out_argument_supported(out)

    def op(data: torch.Tensor) -> torch.Tensor:
        return _disable_torch_function_call(torch.nan_to_num, data, nan=nan, posinf=posinf, neginf=neginf)

    result = op(input.data)
    variable = input._variable.with_data(result)
    return input._new(variable=variable)


@_implements(torch.logical_and, torch.Tensor.logical_and)
def _torch_logical_and(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.logical_and, lhs, rhs)

    return _binary_elementwise("logical_and", op, op, input, other)


@_implements(torch.logical_or, torch.Tensor.logical_or)
def _torch_logical_or(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.logical_or, lhs, rhs)

    return _binary_elementwise("logical_or", op, op, input, other)


@_implements(torch.logical_xor, torch.Tensor.logical_xor)
def _torch_logical_xor(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.logical_xor, lhs, rhs)

    return _binary_elementwise("logical_xor", op, op, input, other)


@_implements(torch.bitwise_and, torch.Tensor.bitwise_and)
def _torch_bitwise_and(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.bitwise_and, lhs, rhs)

    return _binary_elementwise("bitwise_and", op, op, input, other)


@_implements(torch.bitwise_or, torch.Tensor.bitwise_or)
def _torch_bitwise_or(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.bitwise_or, lhs, rhs)

    return _binary_elementwise("bitwise_or", op, op, input, other)


@_implements(torch.bitwise_xor, torch.Tensor.bitwise_xor)
def _torch_bitwise_xor(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.bitwise_xor, lhs, rhs)

    return _binary_elementwise("bitwise_xor", op, op, input, other)


@_implements(torch.eq, torch.Tensor.eq)
def _torch_eq(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.eq, lhs, rhs)

    def reverse(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.eq, rhs, lhs)

    return _binary_elementwise("eq", op, reverse, input, other)


@_implements(torch.ne, torch.Tensor.ne)
def _torch_ne(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.ne, lhs, rhs)

    def reverse(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.ne, rhs, lhs)

    return _binary_elementwise("ne", op, reverse, input, other)


@_implements(torch.lt, torch.Tensor.lt)
def _torch_lt(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.lt, lhs, rhs)

    def reverse(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.lt, rhs, lhs)

    return _binary_elementwise("lt", op, reverse, input, other)


@_implements(torch.le, torch.Tensor.le)
def _torch_le(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.le, lhs, rhs)

    def reverse(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.le, rhs, lhs)

    return _binary_elementwise("le", op, reverse, input, other)


@_implements(torch.gt, torch.Tensor.gt)
def _torch_gt(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.gt, lhs, rhs)

    def reverse(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.gt, rhs, lhs)

    return _binary_elementwise("gt", op, reverse, input, other)


@_implements(torch.ge, torch.Tensor.ge)
def _torch_ge(input: Any, other: Any):
    def op(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.ge, lhs, rhs)

    def reverse(lhs: torch.Tensor, rhs: Any) -> torch.Tensor:
        return _disable_torch_function_call(torch.ge, rhs, lhs)

    return _binary_elementwise("ge", op, reverse, input, other)


@_implements(torch.clamp, torch.Tensor.clamp)
def _torch_clamp(input: Any, min: Optional[Any] = None, max: Optional[Any] = None, *, out: Optional[Any] = None):
    _ensure_out_argument_supported(out)
    if not isinstance(input, DataTensor):
        return NotImplemented
    result = _disable_torch_function_call(torch.clamp, input.data, min=min, max=max)
    variable = input._variable.with_data(result)
    return input._new(variable=variable)


if hasattr(torch, "clip"):
    tensor_clip = getattr(torch.Tensor, "clip", None)
    if tensor_clip is not None:
        _implements(torch.clip, tensor_clip)(_torch_clamp)
    else:
        _implements(torch.clip)(_torch_clamp)


@_implements(torch.where, torch.Tensor.where)
def _torch_where(condition: Any, input: Any, other: Any):
    if not isinstance(condition, DataTensor):
        return NotImplemented

    def _prepare_operand(value: Any, reference: DataTensor) -> Any:
        if isinstance(value, DataTensor):
            if value.dims != reference.dims:
                raise ValueError("torch.where requires matching dimensions when using DataTensor operands.")
            return value.data
        return value

    lhs = _prepare_operand(input, condition)
    rhs = _prepare_operand(other, condition)
    result = _disable_torch_function_call(torch.where, condition.data, lhs, rhs)
    variable = condition._variable.with_data(result)
    return condition._new(variable=variable)


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


@_implements(torch.var, torch.Tensor.var)
def _torch_var(input: Any, dim: Optional[Any] = None, unbiased: bool = True, keepdim: bool = False, out: Optional[Any] = None):
    if not isinstance(input, DataTensor):
        return NotImplemented
    _ensure_out_argument_supported(out)
    dims = _normalize_torch_dims(dim, input.dims)
    return input.var(dim=dims, keepdims=keepdim, unbiased=unbiased)


@_implements(torch.any, torch.Tensor.any)
def _torch_any(input: Any, dim: Optional[Any] = None, keepdim: bool = False, out: Optional[Any] = None):
    if not isinstance(input, DataTensor):
        return NotImplemented
    _ensure_out_argument_supported(out)
    dims = _normalize_torch_dims(dim, input.dims)
    return input.any(dim=dims, keepdims=keepdim)


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
