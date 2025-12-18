import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from xtensor import DataTensor


def test_to_dataarray_roundtrip(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    xr_roundtrip = tensor.to_dataarray()
    np.testing.assert_allclose(xr_roundtrip.data, base_array.data)
    assert tuple(xr_roundtrip.dims) == base_array.dims


def test_to_pandas_series_and_dataframe():
    tensor = DataTensor([[1, 2], [3, 4]], {"row": ["a", "b"], "col": ["x", "y"]}, ("row", "col"))
    df = tensor.to_pandas()
    assert isinstance(df, pd.DataFrame)
    np.testing.assert_allclose(df.to_numpy(), tensor.data.numpy())

    series = DataTensor([5, 6, 7], {"axis": [10, 20, 30]}, ("axis",)).to_pandas()
    assert isinstance(series, pd.Series)
    np.testing.assert_allclose(series.to_numpy(), [5, 6, 7])

    tensor_3d = DataTensor(torch.ones((2, 2, 2)), {"a": [0, 1], "b": [0, 1], "c": [0, 1]}, ("a", "b", "c"))
    with pytest.raises(ValueError):
        tensor_3d.to_pandas()


def test_datetime_coords_roundtrip():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    values = np.arange(8).reshape(4, 2)
    tensor = DataTensor(values, {"time": dates, "feature": ["x", "y"]}, ("time", "feature"))
    df = tensor.to_pandas()
    assert isinstance(df.index, pd.DatetimeIndex)
    xr_round = tensor.to_dataarray()
    assert isinstance(xr_round.indexes["time"], pd.DatetimeIndex)


def test_dataarray_datetime_roundtrip():
    dates = pd.date_range("2021-01-01", periods=3, freq="h")
    spaces = ["a", "b"]
    arr = xr.DataArray(
        np.arange(6).reshape(3, 2),
        dims=("time", "space"),
        coords={"time": dates, "space": spaces},
    )
    tensor = DataTensor.from_dataarray(arr)
    xr_round = tensor.to_dataarray()
    assert isinstance(xr_round.indexes["time"], pd.DatetimeIndex)
    assert xr_round.indexes["time"].equals(arr.indexes["time"])
