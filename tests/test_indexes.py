import pandas as pd
import pytest
import torch

import xtensor as xt
from xtensor.indexes import PandasIndex, RangeIndex, SortedUniqueIndex, TorchIndex


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


def test_torch_index_sorted_duplicate_prefers_first_position():
    values = torch.tensor([1.0, 1.0, 2.0, 4.0])
    index = TorchIndex(values)
    assert index.get_loc(1.0) == 0
    result = index.get_indexer([1.0, 2.0, 4.0], device=values.device)
    expected = torch.tensor([0, 2, 3], dtype=torch.long, device=values.device)
    torch.testing.assert_close(result, expected)


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


def test_range_index_lookup_and_take():
    index = RangeIndex(10, 2, 5)
    coords = index.coord_array()
    torch.testing.assert_close(coords, torch.arange(5, dtype=torch.float64) * 2 + 10)
    locs = index.get_indexer([12.0, 18.0], device=torch.device("cpu"))
    torch.testing.assert_close(locs, torch.tensor([1, 4], dtype=torch.long))
    taken = index.take(slice(1, 4))
    assert isinstance(taken, RangeIndex)
    torch.testing.assert_close(taken.coord_array(), torch.tensor([12.0, 14.0, 16.0], dtype=torch.float64))


def test_sorted_unique_index_binary_search():
    values = torch.tensor([0.5, 1.5, 3.5, 7.5], dtype=torch.float64)
    index = SortedUniqueIndex(values)
    locs = index.get_indexer([0.5, 7.5], device=values.device)
    torch.testing.assert_close(locs, torch.tensor([0, 3], dtype=torch.long))
    with pytest.raises(KeyError):
        index.get_indexer([2.0], device=values.device)


def test_index_helpers_exposed_via_module():
    rng = xt.arange_index(0, 1, 3)
    assert isinstance(rng, RangeIndex)
    torch.testing.assert_close(rng.coord_array(), torch.arange(3, dtype=torch.float64))
    sorted_idx = xt.sorted_unique_index(torch.tensor([1.0, 2.0, 3.0]))
    assert isinstance(sorted_idx, SortedUniqueIndex)
