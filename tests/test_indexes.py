import pandas as pd
import pytest
import torch

from xtensor.indexes import PandasIndex, TorchIndex


def test_torch_index_vectorized_lookup_returns_tensor():
    values = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    index = TorchIndex(values)
    result = index.get_indexer([20, 30, 40], device=values.device)
    expected = torch.tensor([1, 2, 3], dtype=torch.long, device=values.device)
    torch.testing.assert_close(result, expected)


def test_torch_index_vectorized_lookup_handles_boolean_dtype():
    values = torch.tensor([True, False], dtype=torch.bool)
    index = TorchIndex(values)
    result = index.get_indexer([False, False, True], device=values.device)
    expected = torch.tensor([1, 1, 0], dtype=torch.long, device=values.device)
    torch.testing.assert_close(result, expected)


def test_torch_index_raises_for_missing_value():
    values = torch.tensor([1.0, 2.0, 3.0])
    index = TorchIndex(values)
    with pytest.raises(KeyError):
        index.get_indexer([1.0, 4.0], device=values.device)


def test_pandas_index_vectorized_lookup_returns_tensor():
    raw_index = pd.Index(["north", "east", "south", "west"])
    index = PandasIndex(raw_index)
    result = index.get_indexer(["east", "west"], device=torch.device("cpu"))
    expected = torch.tensor([1, 3], dtype=torch.long)
    torch.testing.assert_close(result, expected)


def test_pandas_index_missing_value_raises():
    raw_index = pd.Index(["alpha", "beta"])
    index = PandasIndex(raw_index)
    with pytest.raises(KeyError):
        index.get_indexer(["gamma"], device=torch.device("cpu"))
