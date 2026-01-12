import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from xtensor import DataTensor


def test_from_dataarray_matches_shape(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    np.testing.assert_allclose(tensor.data.numpy(), base_array.data)
    assert tensor.dims == base_array.dims
    coord = tensor.coords["x"]
    if isinstance(coord, torch.Tensor):
        np.testing.assert_allclose(coord.cpu().numpy(), base_array.coords["x"].values)
    else:
        assert coord == tuple(base_array.coords["x"].values.tolist())


def test_constructor_validates_coords():
    data = np.ones((2, 2))
    coords = {"x": [0, 1], "y": [0]}  # mismatch
    with pytest.raises(ValueError):
        DataTensor(data, coords, ("x", "y"))


def test_from_pandas_series_and_dataframe():
    series = pd.Series([1, 3, 5], index=pd.Index([0, 1, 2], name="time"))
    tensor = DataTensor.from_pandas(series)
    assert tensor.shape == (3,)
    coord = tensor.coords["time"]
    if isinstance(coord, torch.Tensor):
        np.testing.assert_array_equal(coord.cpu().numpy(), np.array([0, 1, 2]))
    elif hasattr(coord, "equals"):
        assert coord.equals(series.index)
    else:
        assert coord == (0, 1, 2)

    df = pd.DataFrame([[1, 2], [3, 4]], index=pd.Index(["a", "b"], name="row"), columns=pd.Index(["x", "y"], name="col"))
    tensor_df = DataTensor.from_pandas(df)
    assert tensor_df.dims == ("row", "col")
    np.testing.assert_allclose(tensor_df.data.numpy(), df.to_numpy())


def test_datatensor_device_property():
    data = torch.ones((2, 2), device=torch.device("cpu"))
    tensor = DataTensor(data, {"x": [0, 1], "y": [0, 1]}, ("x", "y"))
    assert tensor.device == data.device


def test_datatensor_to_updates_coord_tensors():
    data = torch.arange(5, dtype=torch.float64)
    coords = {"x": torch.arange(5, dtype=torch.float64)}
    tensor = DataTensor(data, coords, ("x",))
    converted = tensor.to(dtype=torch.float32)

    assert converted.data.dtype == torch.float32
    coord = converted.coords["x"]
    assert isinstance(coord, torch.Tensor)
    assert coord.dtype == torch.float32


def test_from_pandas_and_xarray_produce_matching_indexes():
    index = pd.Index([0, 1, 2], name="row")
    columns = pd.Index(["a", "b"], name="col")
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=index, columns=columns)
    xr_array = xr.DataArray(df, dims=("row", "col"))

    xt_from_pd = DataTensor.from_pandas(df)
    xt_from_xr = DataTensor.from_dataarray(xr_array)

    assert xt_from_pd.dims == xt_from_xr.dims
    torch.testing.assert_close(xt_from_pd.data, xt_from_xr.data)
    for dim in xt_from_pd.dims:
        pd_coord = xt_from_pd.coords[dim]
        xr_coord = xt_from_xr.coords[dim]
        if isinstance(pd_coord, torch.Tensor):
            assert isinstance(xr_coord, torch.Tensor)
            torch.testing.assert_close(pd_coord, xr_coord)
        else:
            assert hasattr(pd_coord, "equals")
            assert pd_coord.equals(xr_coord)
