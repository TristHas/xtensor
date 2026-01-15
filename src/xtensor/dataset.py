from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .coordinates import Coordinates, CoordinatesView, IndexesView
from .datatensor import DataTensor
from .indexes import BaseIndex, CoordArray, CoordValue, build_index

DataVarInput = Union[
    DataTensor,
    Tuple[Sequence[str], Any, Mapping[str, CoordValue]],
    Tuple[Sequence[str], Any],
]

_TORCH = None


def _try_import_torch():  # pragma: no cover - helper
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    try:
        import torch  # type: ignore

        _TORCH = torch
    except ImportError:
        _TORCH = None
    return _TORCH


def _to_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    torch = _try_import_torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.item()
    return value


def _is_scalar_selector(value: Any) -> bool:
    if isinstance(value, slice):
        return False
    if isinstance(value, (list, tuple)):
        return False
    if isinstance(value, np.generic):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    torch = _try_import_torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.ndim == 0
    return True


class Dataset:
    """Lightweight Dataset analogue built from DataTensor variables."""

    def __init__(
        self,
        data_vars: Optional[Mapping[str, DataVarInput]] = None,
        *,
        coords: Optional[Mapping[str, CoordValue]] = None,
        attrs: Optional[Mapping[str, Any]] = None,
    ):
        self._data_vars: MutableMapping[str, DataTensor] = OrderedDict()
        data_vars = data_vars or {}
        explicit_coords = dict(coords or {})
        for name, value in data_vars.items():
            tensor = self._convert_to_datatensor(name, value, explicit_coords)
            self._validate_variable_dims(name, tensor)
            self._data_vars[name] = tensor
        base_coords = self._coords_from_data_vars(self._data_vars)
        self._coords = self._apply_explicit_coords(base_coords, explicit_coords)
        self._dim_order = self._compute_dim_order(self._data_vars)
        self._attrs = dict(attrs or {})

    def __getitem__(self, key: str):
        if self._coords.has_coord(key):
            return self._coord_as_datatensor(key)
        if key in self._data_vars:
            return self._data_vars[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: DataVarInput) -> None:
        if self._coords.has_coord(key) and key not in self._data_vars:
            coord_tensor = self._convert_to_datatensor(key, value, self.coords)
            self._assign_coord(key, coord_tensor)
            return
        current_coords = self._coords
        tensor = self._convert_to_datatensor(key, value, self.coords)
        self._validate_variable_dims(key, tensor)
        self._data_vars[key] = tensor
        present_dims = set(self._collect_dims(self._data_vars))
        extra_coords = OrderedDict(
            (dim, current_coords.coord_values(dim)) for dim in current_coords.dim_names() if dim not in present_dims
        )
        self._coords = self._coords_from_data_vars(
            self._data_vars,
            base_coords=current_coords,
            extra_coords=extra_coords or None,
        )
        self._promote_coordinate_if_needed(key, tensor)

    def __contains__(self, key: str) -> bool:
        return key in self._data_vars

    @property
    def data_vars(self) -> Mapping[str, DataTensor]:
        return dict(self._data_vars)

    @property
    def coords(self) -> CoordinatesView:
        return CoordinatesView(self._coords)

    @property
    def indexes(self) -> IndexesView:
        return IndexesView(self._coords)

    @property
    def attrs(self) -> Mapping[str, Any]:
        return dict(self._attrs)

    @property
    def sizes(self) -> Mapping[str, int]:
        dim_sizes = self._coords.dim_sizes()
        return {dim: dim_sizes[dim] for dim in self._iter_dims()}

    @property
    def dims(self) -> Mapping[str, int]:
        return self.sizes

    def sel(self, **indexers: Any) -> "Dataset":
        return self._apply_indexers("sel", indexers)

    def isel(self, **indexers: Any) -> "Dataset":
        return self._apply_indexers("isel", indexers)

    def assign(self, **kwargs: DataVarInput) -> "Dataset":
        if not kwargs:
            return self
        updated = OrderedDict(self._data_vars)
        for name, value in kwargs.items():
            updated[name] = self._convert_to_datatensor(name, value, self.coords)
        return self._replace(data_vars=updated, recompute_coords=True)

    def assign_coords(self, **coords: CoordValue) -> "Dataset":
        if not coords:
            return self
        new_vars = OrderedDict()
        dim_updates: Dict[str, BaseIndex] = {}
        extra_updates: Dict[str, CoordValue] = {}
        for dim, values in coords.items():
            if dim in self.dims:
                size = self.sizes[dim]
            else:
                try:
                    size = len(values)  # type: ignore[arg-type]
                except TypeError as error:
                    extra_updates[dim] = values
                    continue
            device = self._device_for_dim(dim)
            dim_updates[dim] = build_index(values, size, dim, device=device)
        for name, var in self._data_vars.items():
            updates = {dim: dim_updates[dim].coord_array() for dim in var.dims if dim in dim_updates}
            if updates:
                new_vars[name] = var.assign_coords(**updates)
            else:
                new_vars[name] = var
        updated_coords = self._coords.replace(
            dim_indexes=dim_updates or None,
            extra_coords=extra_updates or None,
        )
        return self._replace(data_vars=new_vars, coords=updated_coords)

    def rename(self, dims: Optional[Mapping[str, str]] = None, **names: str) -> "Dataset":
        mapping = dict(dims or {})
        mapping.update(names)
        if not mapping:
            return self
        var_mapping = {k: v for k, v in mapping.items() if k in self._data_vars}
        dim_mapping = {k: v for k, v in mapping.items() if self._coords.has_coord(k)}
        invalid = set(mapping) - (set(var_mapping) | set(dim_mapping))
        if invalid:
            raise ValueError(f"Unknown names in rename: {sorted(invalid)}")
        new_vars = OrderedDict()
        for name, var in self._data_vars.items():
            new_name = var_mapping.get(name, name)
            if name in var_mapping and new_name in new_vars:
                raise ValueError(f"Duplicate variable '{new_name}' after rename.")
            mapped = var.rename(dim_mapping) if dim_mapping else var
            new_vars[new_name] = mapped
        new_coords = self._coords.rename(mapping)
        return self._replace(data_vars=new_vars, coords=new_coords)

    def transpose(self, *dims: str) -> "Dataset":
        if not dims:
            dims = tuple(reversed(tuple(self.dims.keys())))
        new_vars = OrderedDict()
        for name, var in self._data_vars.items():
            requested = [dim for dim in dims if dim in var.dims]
            remaining = [dim for dim in var.dims if dim not in requested]
            order = tuple(requested + remaining)
            new_vars[name] = var.transpose(*order) if order != var.dims else var
        return self._replace(data_vars=new_vars, recompute_coords=True)

    def squeeze(self, dims: Optional[Union[str, Sequence[str]]] = None) -> "Dataset":
        if dims is None:
            target_dims = [dim for dim, size in self.sizes.items() if size == 1]
        elif isinstance(dims, str):
            target_dims = [dims]
        else:
            target_dims = list(dims)
        if not target_dims:
            return self
        new_vars = OrderedDict()
        for name, var in self._data_vars.items():
            applicable = [dim for dim in target_dims if dim in var.dims]
            if not applicable:
                new_vars[name] = var
            else:
                arg = applicable if len(applicable) > 1 else applicable[0]
                new_vars[name] = var.squeeze(arg)
        present_after = set(self._collect_dims(new_vars))
        preserved = {
            dim: self._coords.coord_values(dim)
            for dim in target_dims
            if dim not in present_after and self._coords.has_dim(dim)
        }
        return self._replace(data_vars=new_vars, recompute_coords=True, extra_coords=preserved or None)

    def to_datatensor(self, dim: str = "variable", name: Optional[str] = None) -> DataTensor:
        if not self._data_vars:
            raise ValueError("Cannot convert an empty Dataset to a DataTensor.")

        dataset_dims = self._iter_dims()
        dim_positions = {d: idx for idx, d in enumerate(dataset_dims)}
        target_shape = tuple(self.sizes[d] for d in dataset_dims)
        resolved_dtype: Optional[torch.dtype] = None
        target_device: Optional[torch.device] = None
        aligned_tensors = []

        for var_name, var in self._data_vars.items():
            data = var.data
            ordered_dims = var.dims
            if ordered_dims:
                perm = sorted(range(len(ordered_dims)), key=lambda axis: dim_positions[ordered_dims[axis]])
                if perm != list(range(len(ordered_dims))):
                    data = data.permute(*perm)
                ordered_dims = tuple(ordered_dims[idx] for idx in perm)
            data_dims = list(ordered_dims)
            aligned = data
            for axis, dataset_dim in enumerate(dataset_dims):
                if axis < len(data_dims) and data_dims[axis] == dataset_dim:
                    continue
                aligned = aligned.unsqueeze(axis)
                data_dims.insert(axis, dataset_dim)
            if dataset_dims:
                for axis, target in enumerate(target_shape):
                    current = aligned.shape[axis]
                    if current not in (1, target):
                        raise ValueError(
                            f"Variable '{var_name}' dimension '{dataset_dims[axis]}' has incompatible size {current}."
                        )
                aligned = aligned.expand(*target_shape)

            aligned_tensors.append(aligned)
            if resolved_dtype is None:
                resolved_dtype = aligned.dtype
            else:
                resolved_dtype = torch.promote_types(resolved_dtype, aligned.dtype)
            if target_device is None:
                target_device = aligned.device

        assert resolved_dtype is not None
        assert target_device is not None

        stacked_inputs = []
        for tensor in aligned_tensors:
            if tensor.dtype != resolved_dtype or tensor.device != target_device:
                stacked_inputs.append(tensor.to(device=target_device, dtype=resolved_dtype))
            else:
                stacked_inputs.append(tensor)
        stacked = torch.stack(stacked_inputs, dim=0)

        coords: Dict[str, CoordValue] = {}
        coords[dim] = tuple(self._data_vars.keys())
        for dataset_dim in dataset_dims:
            coords[dataset_dim] = self._coords.coord_values(dataset_dim)
        for name_key, values in self._coords.extra_items().items():
            coords[name_key] = values

        result_dims = (dim,) + dataset_dims
        return DataTensor(stacked, coords, result_dims, attrs=self.attrs, name=name)

    def to(self, *args: Any, **kwargs: Any) -> "Dataset":
        if not self._data_vars:
            return self
        moved = OrderedDict()
        for name, var in self._data_vars.items():
            moved[name] = var.to(*args, **kwargs)
        return self._replace(data_vars=moved, recompute_coords=True)

    def to_xarray(self):
        try:
            import xarray as xr
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("xarray must be installed to convert Dataset.") from error
        data_vars = {}
        for name, var in self._data_vars.items():
            data_vars[name] = var.to_dataarray()
        ds = xr.Dataset(data_vars)
        dim_names = set(self.dims.keys())
        for dim in self._coords.dim_names():
            values = self._coords.coord_values(dim)
            if isinstance(values, torch.Tensor):
                array = values.cpu().detach().numpy()
            else:
                array = np.asarray(values)
            if dim in dim_names:
                ds = ds.assign_coords({dim: array})
            else:
                scalar = array.item() if array.ndim <= 1 and array.size == 1 else array
                ds = ds.assign_coords({dim: scalar})
        for name, values in self._coords.extra_items().items():
            if isinstance(values, torch.Tensor):
                array = values.cpu().detach().numpy()
            else:
                array = np.asarray(values)
            scalar = array.reshape(-1)[0] if array.ndim <= 1 and array.size == 1 else array
            ds = ds.assign_coords({name: scalar})
        ds.attrs.update(self._attrs)
        return ds

    @staticmethod
    def from_xarray(dataset) -> "Dataset":
        data_vars = {name: DataTensor.from_dataarray(var) for name, var in dataset.data_vars.items()}
        coords = {name: dataset.coords[name].to_numpy() for name in dataset.coords}
        return Dataset(data_vars, coords=coords, attrs=dict(dataset.attrs))

    def _apply_indexers(self, method: str, indexers: Mapping[str, Any]) -> "Dataset":
        if not indexers:
            return self
        new_vars = OrderedDict()
        for name, var in self._data_vars.items():
            applicable = {dim: sel for dim, sel in indexers.items() if dim in var.dims}
            new_vars[name] = getattr(var, method)(**applicable) if applicable else var
        present_after = set(self._collect_dims(new_vars))
        extra_coords: Dict[str, CoordValue] = {}
        for dim, selector in indexers.items():
            if dim in present_after:
                continue
            value = self._scalar_selection_value(dim, selector, method)
            if value is not None:
                extra_coords[dim] = (value,)
        return self._replace(
            data_vars=new_vars,
            recompute_coords=True,
            extra_coords=extra_coords or None,
        )

    def _replace(
        self,
        *,
        data_vars: Optional[Mapping[str, DataTensor]] = None,
        coords: Optional[Coordinates] = None,
        attrs: Optional[Mapping[str, Any]] = None,
        recompute_coords: bool = False,
        extra_coords: Optional[Mapping[str, CoordValue]] = None,
    ) -> "Dataset":
        obj = self.__class__.__new__(self.__class__)
        obj._data_vars = OrderedDict(data_vars if data_vars is not None else self._data_vars)
        if coords is not None:
            obj._coords = coords.copy()
        elif recompute_coords:
            base_coords = getattr(self, "_coords", None)
            obj._coords = self._coords_from_data_vars(
                obj._data_vars,
                base_coords=base_coords,
                extra_coords=extra_coords,
            )
        else:
            obj._coords = self._coords.copy()
        obj._dim_order = self._compute_dim_order(obj._data_vars)
        obj._attrs = dict(attrs if attrs is not None else self._attrs)
        return obj

    def _coords_from_data_vars(
        self,
        data_vars: Mapping[str, DataTensor],
        base_coords: Optional[Coordinates] = None,
        extra_coords: Optional[Mapping[str, CoordValue]] = None,
    ) -> Coordinates:
        dim_indexes: "OrderedDict[str, BaseIndex]" = OrderedDict()
        extras: "OrderedDict[str, CoordArray]" = OrderedDict()
        present_dims = self._collect_dims(data_vars)
        if base_coords is not None:
            for dim in base_coords.dim_names():
                if dim in present_dims:
                    dim_indexes[dim] = base_coords.dim_index(dim)
            for name, values in base_coords.extra_items().items():
                extras.setdefault(name, values)
        for var in data_vars.values():
            for dim, index in var._dim_index_map().items():
                dim_indexes[dim] = index
            for name, values in var._extra_coords().items():
                extras.setdefault(name, values)
        if extra_coords:
            for name, values in extra_coords.items():
                extras[name] = values
        return Coordinates(dim_indexes, extra_coords=extras)

    def _apply_explicit_coords(self, coords: Coordinates, explicit: Mapping[str, CoordValue]) -> Coordinates:
        if not explicit:
            return coords
        dim_updates: Dict[str, BaseIndex] = {}
        extra_updates: Dict[str, CoordValue] = {}
        for name, values in explicit.items():
            if coords.has_dim(name):
                size = len(coords.dim_index(name))
            else:
                try:
                    size = len(values)  # type: ignore[arg-type]
                except TypeError:
                    extra_updates[name] = values
                    continue
            device = self._device_for_dim(name)
            dim_updates[name] = build_index(values, size, name, device=device)
        return coords.replace(
            dim_indexes=dim_updates or None,
            extra_coords=extra_updates or None,
        )

    def __repr__(self) -> str:
        try:
            return self.to_xarray().__repr__()
        except Exception:
            vars_summary = ", ".join(self._data_vars.keys())
            coords_summary = ", ".join(self._coords.dim_names())
            return f"Dataset(data_vars=[{vars_summary}], coords=[{coords_summary}])"

    # Elementwise math -------------------------------------------------
    def __pow__(self, other: Any) -> "Dataset":
        if isinstance(other, Dataset):
            return self._dataset_binary_op(other, lambda lhs, rhs: lhs**rhs, "pow")
        new_vars = OrderedDict()
        for name, var in self._data_vars.items():
            new_vars[name] = var**other
        return self._replace(data_vars=new_vars, recompute_coords=True)

    def __rpow__(self, other: Any) -> "Dataset":
        new_vars = OrderedDict()
        for name, var in self._data_vars.items():
            new_vars[name] = var.__rpow__(other)
        return self._replace(data_vars=new_vars, recompute_coords=True)

    def _repr_html_(self):
        try:
            html = self.to_xarray()._repr_html_()
        except Exception:
            return None
        if html is None:
            return None
        return html.replace("xarray.Dataset", "xtensor.Dataset")

    def _promote_coordinate_if_needed(self, name: str, tensor: DataTensor) -> None:
        if tensor.dims != (name,):
            return
        self._assign_coord(name, tensor)

    def _assign_coord(self, name: str, tensor: DataTensor) -> None:
        coord_values = tensor.data.detach().clone()
        coord_index = build_index(coord_values, coord_values.shape[0], name, device=coord_values.device)
        updated_vars = OrderedDict()
        for var_name, var in self._data_vars.items():
            if name in var.dims:
                updated_vars[var_name] = var.assign_coords(**{name: coord_index.coord_array()})
            else:
                updated_vars[var_name] = var
        self._data_vars = updated_vars
        self._coords = self._coords.replace(dim_indexes={name: coord_index})

    def _coord_as_datatensor(self, name: str):
        values = self._coords.coord_values(name)
        if isinstance(values, torch.Tensor):
            data = values.clone()
        else:
            try:
                data = torch.as_tensor(list(values))
            except (TypeError, ValueError):
                return values
        return DataTensor(data, {name: values}, (name,))

    def _collect_dims(self, data_vars: Optional[Mapping[str, DataTensor]] = None) -> Tuple[str, ...]:
        dims = OrderedDict()
        vars_map = data_vars if data_vars is not None else self._data_vars
        for var in vars_map.values():
            for dim in var.dims:
                dims.setdefault(dim, None)
        return tuple(dims.keys())

    def _dataset_binary_op(
        self,
        other: "Dataset",
        func: Callable[[DataTensor, DataTensor], DataTensor],
        op_name: str,
    ) -> "Dataset":
        left = set(self._data_vars)
        right = set(other._data_vars)
        if left != right:
            missing = sorted(left ^ right)
            raise ValueError(f"Dataset {op_name} requires identical data variables. Differing variables: {missing}")
        new_vars: "OrderedDict[str, DataTensor]" = OrderedDict()
        for name in self._data_vars:
            new_vars[name] = func(self._data_vars[name], other._data_vars[name])
        return self._replace(data_vars=new_vars, recompute_coords=True)

    def _compute_dim_order(self, data_vars: Mapping[str, DataTensor]) -> Tuple[str, ...]:
        present = self._collect_dims(data_vars)
        ordered: list[str] = []
        seen: set[str] = set()
        prior = getattr(self, "_dim_order", tuple())
        for dim in prior:
            if dim in present and dim not in seen:
                ordered.append(dim)
                seen.add(dim)
        for dim in present:
            if dim not in seen:
                ordered.append(dim)
                seen.add(dim)
        return tuple(ordered)

    def _iter_dims(self, data_vars: Optional[Mapping[str, DataTensor]] = None) -> Tuple[str, ...]:
        vars_map = data_vars if data_vars is not None else self._data_vars
        present = self._collect_dims(vars_map)
        ordered: list[str] = []
        seen: set[str] = set()
        for dim in getattr(self, "_dim_order", ()):
            if dim in present and dim not in seen:
                ordered.append(dim)
                seen.add(dim)
        for dim in present:
            if dim not in seen:
                ordered.append(dim)
                seen.add(dim)
        return tuple(ordered)

    def _convert_to_datatensor(
        self,
        name: str,
        value: DataVarInput,
        coords: Mapping[str, CoordValue],
    ) -> DataTensor:
        if isinstance(value, DataTensor):
            return value
        if not isinstance(value, tuple) or len(value) < 2:
            raise TypeError(f"Invalid specification for data variable '{name}'.")
        dims = tuple(value[0])
        data = value[1]
        coord_overrides = value[2] if len(value) > 2 else {}
        coord_map: Dict[str, CoordValue] = {}
        for dim in dims:
            if dim in coord_overrides:
                coord_map[dim] = coord_overrides[dim]
            elif dim in coords:
                coord_map[dim] = coords[dim]
        return DataTensor(data, coord_map, dims)

    def _infer_dim_size(self, dim: str) -> int:
        for var in self._data_vars.values():
            if dim in var.dims:
                return var.sizes[dim]
        if self._coords.has_dim(dim):
            return len(self._coords.dim_index(dim))
        raise ValueError(f"Dimension '{dim}' not present in Dataset.")

    def _default_device(self) -> torch.device:
        for var in self._data_vars.values():
            return var.data.device
        return torch.device("cpu")

    def _device_for_dim(self, dim: str) -> torch.device:
        for var in self._data_vars.values():
            if dim in var.dims:
                return var.data.device
        coords = getattr(self, "_coords", None)
        if coords is not None and coords.has_dim(dim):
            values = coords.coord_values(dim)
            if isinstance(values, torch.Tensor):
                return values.device
        return self._default_device()

    def _scalar_selection_value(self, dim: str, selector: Any, method: str) -> Optional[Any]:
        if not _is_scalar_selector(selector):
            return None
        scalar = _to_scalar(selector)
        if method == "sel":
            return scalar
        if not self._coords.has_dim(dim):
            return None
        index = int(scalar)
        values = self._coords.coord_values(dim)
        length = len(values) if not isinstance(values, torch.Tensor) else values.shape[0]
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(f"Index {index} out of bounds for dimension '{dim}'.")
        if isinstance(values, torch.Tensor):
            return _to_scalar(values[index])
        return _to_scalar(values[index])

    def _validate_variable_dims(self, name: str, tensor: DataTensor) -> None:
        if not self._data_vars:
            return
        for dim, size in tensor.sizes.items():
            for existing in self._data_vars.values():
                if dim in existing.dims and existing.sizes[dim] != size:
                    raise ValueError(
                        f"Variable '{name}' dimension '{dim}' of size {size} "
                        f"differs from existing size {existing.sizes[dim]}."
                    )
