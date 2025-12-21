from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .datatensor import CoordValue, DataTensor

DataVarInput = Union[
    DataTensor,
    Tuple[Sequence[str], Any, Mapping[str, CoordValue]],
    Tuple[Sequence[str], Any],
]

def _to_coord_tuple(values: CoordValue, size: int, dim: str) -> Tuple[Any, ...]:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != size:
        raise ValueError(f"Coordinate length mismatch on dim '{dim}'. Expected {size}, got {array.shape[0]}")
    return tuple(array.tolist())

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
            self._data_vars[name] = self._convert_to_datatensor(name, value, explicit_coords)
        self._coords = self._coords_from_data_vars(self._data_vars)
        for dim, values in explicit_coords.items():
            if dim in self._coords:
                size = len(self._coords[dim])
            else:
                try:
                    size = self._infer_dim_size(dim)
                except ValueError:
                    try:
                        size = len(values)  # type: ignore[arg-type]
                    except TypeError as error:
                        raise ValueError(f"Cannot determine length for coordinate '{dim}'.") from error
            self._coords[dim] = _to_coord_tuple(values, size, dim)
        self._dim_order = self._compute_dim_order(self._data_vars)
        self._attrs = dict(attrs or {})

    def __getitem__(self, key: str) -> DataTensor:
        if key in self._coords:
            return self._coord_as_datatensor(key)
        if key in self._data_vars:
            return self._data_vars[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: DataVarInput) -> None:
        if key in self._coords:
            coord_tensor = self._convert_to_datatensor(key, value, self._coords)
            self._assign_coord(key, coord_tensor)
            return
        current_coords = OrderedDict(self._coords)
        tensor = self._convert_to_datatensor(key, value, self._coords)
        self._data_vars[key] = tensor
        present_dims = set(self._collect_dims(self._data_vars))
        extra_coords = OrderedDict(
            (dim, values) for dim, values in current_coords.items() if dim not in present_dims
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
    def coords(self) -> Mapping[str, Tuple[Any, ...]]:
        return dict(self._coords)

    @property
    def attrs(self) -> Mapping[str, Any]:
        return dict(self._attrs)

    @property
    def sizes(self) -> Mapping[str, int]:
        return {dim: self._infer_dim_size(dim) for dim in self._iter_dims()}

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
            updated[name] = self._convert_to_datatensor(name, value, self._coords)
        return self._replace(data_vars=updated, recompute_coords=True)

    def assign_coords(self, **coords: CoordValue) -> "Dataset":
        if not coords:
            return self
        normalized = OrderedDict(self._coords)
        new_vars = OrderedDict()
        for dim, values in coords.items():
            try:
                size = self._infer_dim_size(dim)
            except ValueError:
                try:
                    size = len(values)  # type: ignore[arg-type]
                except TypeError as error:
                    raise ValueError(f"Cannot determine length for coordinate '{dim}'.") from error
            normalized[dim] = _to_coord_tuple(values, size, dim)
        for name, var in self._data_vars.items():
            updates = {dim: normalized[dim] for dim in var.dims if dim in coords}
            if updates:
                new_vars[name] = var.assign_coords(**updates)
            else:
                new_vars[name] = var
        return self._replace(data_vars=new_vars, coords=normalized)

    def rename(self, dims: Optional[Mapping[str, str]] = None, **names: str) -> "Dataset":
        mapping = dict(dims or {})
        mapping.update(names)
        if not mapping:
            return self
        var_mapping = {k: v for k, v in mapping.items() if k in self._data_vars}
        dim_mapping = {k: v for k, v in mapping.items() if k in self._coords}
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
        new_coords = OrderedDict()
        for dim, values in self._coords.items():
            new_coords[dim_mapping.get(dim, dim)] = values
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
            dim: self._coords[dim]
            for dim in target_dims
            if dim not in present_after and dim in self._coords
        }
        return self._replace(data_vars=new_vars, recompute_coords=True, extra_coords=preserved or None)

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
        for dim, values in self._coords.items():
            if isinstance(values, torch.Tensor):
                array = values.cpu().detach().numpy()
            else:
                array = np.asarray(values)
            if dim in dim_names:
                ds = ds.assign_coords({dim: array})
            else:
                scalar = array.item() if array.ndim <= 1 and array.size == 1 else array
                ds = ds.assign_coords({dim: scalar})
        ds.attrs.update(self._attrs)
        return ds

    @staticmethod
    def from_xarray(dataset) -> "Dataset":
        data_vars = {name: DataTensor.from_dataarray(var) for name, var in dataset.data_vars.items()}
        coords = {dim: dataset.coords[dim].to_numpy() for dim in dataset.dims if dim in dataset.coords}
        return Dataset(data_vars, coords=coords, attrs=dict(dataset.attrs))

    def _apply_indexers(self, method: str, indexers: Mapping[str, Any]) -> "Dataset":
        if not indexers:
            return self
        new_vars = OrderedDict()
        for name, var in self._data_vars.items():
            applicable = {dim: sel for dim, sel in indexers.items() if dim in var.dims}
            new_vars[name] = getattr(var, method)(**applicable) if applicable else var
        present_after = set(self._collect_dims(new_vars))
        extra_coords: Dict[str, Tuple[Any, ...]] = {}
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
        coords: Optional[Mapping[str, Any]] = None,
        attrs: Optional[Mapping[str, Any]] = None,
        recompute_coords: bool = False,
        extra_coords: Optional[Mapping[str, Any]] = None,
    ) -> "Dataset":
        obj = self.__class__.__new__(self.__class__)
        obj._data_vars = OrderedDict(data_vars if data_vars is not None else self._data_vars)
        if coords is not None:
            obj._coords = OrderedDict(coords)
        elif recompute_coords:
            base_coords = getattr(self, "_coords", None)
            merged_extra: OrderedDict[str, Any] = OrderedDict(extra_coords or {})
            if base_coords:
                present_dims = set(self._collect_dims(obj._data_vars))
                for dim, values in base_coords.items():
                    if dim not in present_dims and dim not in merged_extra:
                        merged_extra[dim] = values
            obj._coords = self._coords_from_data_vars(
                obj._data_vars,
                base_coords=base_coords,
                extra_coords=merged_extra or None,
            )
        else:
            obj._coords = OrderedDict(self._coords)
        obj._dim_order = self._compute_dim_order(obj._data_vars)
        obj._attrs = dict(attrs if attrs is not None else self._attrs)
        return obj

    def _coords_from_data_vars(
        self,
        data_vars: Mapping[str, DataTensor],
        base_coords: Optional[Mapping[str, Any]] = None,
        extra_coords: Optional[Mapping[str, Any]] = None,
    ) -> OrderedDict:
        coords = OrderedDict()
        base_coords = base_coords or OrderedDict()
        present_dims = self._collect_dims(data_vars)
        for dim, values in base_coords.items():
            if dim in present_dims:
                coords[dim] = values
        for var in data_vars.values():
            for dim in var.dims:
                coords[dim] = var.coords[dim]
        if extra_coords:
            for dim, values in extra_coords.items():
                coords.setdefault(dim, values)
        return coords

    def __repr__(self) -> str:
        try:
            return self.to_xarray().__repr__()
        except Exception:
            vars_summary = ", ".join(self._data_vars.keys())
            coords_summary = ", ".join(self._coords.keys())
            return f"Dataset(data_vars=[{vars_summary}], coords=[{coords_summary}])"

    def _repr_html_(self):
        try:
            return self.to_xarray()._repr_html_()
        except Exception:
            return None

    def _promote_coordinate_if_needed(self, name: str, tensor: DataTensor) -> None:
        if tensor.dims != (name,):
            return
        coord_values = tensor.data.detach().clone()
        self._assign_coord(name, tensor)

    def _assign_coord(self, name: str, tensor: DataTensor) -> None:
        coord_values = tensor.data.detach().clone()
        updated_vars = OrderedDict()
        for var_name, var in self._data_vars.items():
            if name in var.dims:
                updated_vars[var_name] = var.assign_coords(**{name: coord_values})
            else:
                updated_vars[var_name] = var
        self._data_vars = updated_vars
        self._coords[name] = coord_values

    def _coord_as_datatensor(self, name: str) -> DataTensor:
        values = self._coords[name]
        if isinstance(values, torch.Tensor):
            data = values.clone()
        else:
            data = torch.as_tensor(list(values))
        return DataTensor(data, {name: values}, (name,))

    def _collect_dims(self, data_vars: Optional[Mapping[str, DataTensor]] = None) -> Tuple[str, ...]:
        dims = OrderedDict()
        vars_map = data_vars if data_vars is not None else self._data_vars
        for var in vars_map.values():
            for dim in var.dims:
                dims.setdefault(dim, None)
        return tuple(dims.keys())

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
        if dim in self._coords:
            return len(self._coords[dim])
        raise ValueError(f"Dimension '{dim}' not present in Dataset.")

    def _scalar_selection_value(self, dim: str, selector: Any, method: str) -> Optional[Any]:
        if not _is_scalar_selector(selector):
            return None
        scalar = _to_scalar(selector)
        if method == "sel":
            return scalar
        coord_values = self._coords.get(dim)
        if coord_values is None:
            return None
        index = int(scalar)
        if index < 0:
            index += len(coord_values)
        if index < 0 or index >= len(coord_values):
            raise IndexError(f"Index {index} out of bounds for dimension '{dim}'.")
        return coord_values[index]
