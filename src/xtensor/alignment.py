from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Mapping, Sequence, Tuple, TYPE_CHECKING

from .coordinates import Coordinates
from .indexes import BaseIndex

if TYPE_CHECKING:  # pragma: no cover
    from .datatensor import DataTensor


def align_binary_operands(lhs: "DataTensor", rhs: "DataTensor", op_name: str) -> Tuple["DataTensor", "DataTensor", Dict[str, BaseIndex]]:
    target_dims: list[str] = list(lhs.dims)
    for dim in rhs.dims:
        if dim not in target_dims:
            target_dims.append(dim)

    merged_indexes = _merge_dim_indexes(lhs._dim_index_map(), rhs._dim_index_map(), tuple(target_dims), op_name)
    lhs_aligned = _broadcast_tensor(lhs, target_dims, merged_indexes)
    rhs_aligned = _broadcast_tensor(rhs, target_dims, merged_indexes)
    return lhs_aligned, rhs_aligned, merged_indexes


def _merge_dim_indexes(a: Mapping[str, BaseIndex], b: Mapping[str, BaseIndex], dims: Tuple[str, ...], op_name: str) -> Dict[str, BaseIndex]:
    merged: Dict[str, BaseIndex] = {}
    for dim in dims:
        index_a = a.get(dim)
        index_b = b.get(dim)

        if index_a is None and index_b is None:
            raise ValueError(f"{op_name} cannot determine coordinates for dim '{dim}'.")
        if index_a is None:
            merged[dim] = index_b.clone()  # type: ignore[union-attr]
            continue
        if index_b is None:
            merged[dim] = index_a.clone()
            continue

        len_a = len(index_a)
        len_b = len(index_b)
        if len_a == len_b:
            if not index_a.equals(index_b):
                raise ValueError(f"{op_name} requires matching coordinates on dim '{dim}'.")
            merged[dim] = index_a.clone()
        elif len_a == 1:
            merged[dim] = index_b.clone()
        elif len_b == 1:
            merged[dim] = index_a.clone()
        else:
            raise ValueError(f"{op_name} cannot broadcast dimension '{dim}' (sizes {len_a} vs {len_b}).")
    return merged


def _broadcast_tensor(tensor: "DataTensor", target_dims: Sequence[str], merged_indexes: Mapping[str, BaseIndex]) -> "DataTensor":
    target_dims_tuple = tuple(target_dims)
    data = tensor.data
    current_dims = tensor.dims

    # Ensure axes order matches the subset of target dims already present.
    present_order = [dim for dim in target_dims_tuple if dim in current_dims]
    if len(present_order) != len(current_dims):
        missing = [dim for dim in current_dims if dim not in present_order]
        raise ValueError(f"Cannot align dimension(s) {missing} not present in target dims {target_dims_tuple}.")
    if present_order:
        if tuple(present_order) != current_dims:
            perm = [current_dims.index(dim) for dim in present_order]
            data = data.permute(*perm)
    elif current_dims:
        # No overlap between dims and target dims.
        raise ValueError(f"Cannot align tensor with dims {current_dims} to target dims {target_dims_tuple}.")

    reshape_sizes: list[int] = []
    shape_map = {dim: data.shape[idx] for idx, dim in enumerate(present_order)}
    for dim in target_dims_tuple:
        reshape_sizes.append(shape_map.get(dim, 1))
    if target_dims_tuple:
        data = data.reshape(tuple(reshape_sizes))

    target_sizes = [len(merged_indexes[dim]) for dim in target_dims_tuple]
    if target_dims_tuple and any(target != current for target, current in zip(target_sizes, reshape_sizes)):
        data = data.expand(*target_sizes)

    ordered_indexes = OrderedDict((dim, merged_indexes[dim].clone()) for dim in target_dims_tuple)
    coords = Coordinates(ordered_indexes, extra_coords=tensor._coords.extra_items())
    variable = tensor._variable.with_data(data, target_dims_tuple)
    return tensor._new(variable=variable, coords=coords, dims=target_dims_tuple)
