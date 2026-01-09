from __future__ import annotations

from typing import Dict, Mapping, Tuple, TYPE_CHECKING

from .indexes import BaseIndex

if TYPE_CHECKING:  # pragma: no cover
    from .datatensor import DataTensor


def align_binary_operands(lhs: "DataTensor", rhs: "DataTensor", op_name: str) -> Tuple["DataTensor", "DataTensor", Dict[str, BaseIndex]]:
    if set(lhs.dims) != set(rhs.dims):
        raise ValueError(f"{op_name} requires operands to share the same dimension set.")
    if lhs.dims != rhs.dims:
        rhs = rhs.transpose(*lhs.dims)
    merged_indexes = _merge_dim_indexes(lhs._dim_index_map(), rhs._dim_index_map(), lhs.dims, op_name)
    return lhs, rhs, merged_indexes


def _merge_dim_indexes(a: Mapping[str, BaseIndex], b: Mapping[str, BaseIndex], dims: Tuple[str, ...], op_name: str) -> Dict[str, BaseIndex]:
    merged: Dict[str, BaseIndex] = {}
    for dim in dims:
        index_a = a[dim]
        index_b = b[dim]
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
