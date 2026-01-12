from __future__ import annotations

from typing import Dict, Sequence, Tuple, Optional

import torch


class Variable:
    """Torch-backed variable that tracks data and named dimensions."""

    def __init__(self, data: torch.Tensor, dims: Sequence[str]):
        self._data = data
        self._dims = tuple(dims)

    @property
    def dtype(self) -> torch.Tensor:
        return self._data.dtype
        
    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def dims(self) -> Tuple[str, ...]:
        return self._dims

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    def sizes(self) -> Dict[str, int]:
        return {dim: self._data.shape[idx] for idx, dim in enumerate(self._dims)}

    def with_data(self, data: torch.Tensor, dims: Optional[Sequence[str]] = None) -> "Variable":
        target_dims = tuple(dims) if dims is not None else self._dims
        return Variable(data, target_dims)

    def with_dims(self, dims: Sequence[str]) -> "Variable":
        return Variable(self._data, dims)
