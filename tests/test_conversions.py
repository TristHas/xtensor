import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from xtensor import DataTensor, Dataset, open_datatensor, open_dataset, read_pickle


def _as_numpy(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    if hasattr(values, "to_numpy"):
        return np.asarray(values.to_numpy())
    return np.asarray(values)


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


def test_open_dataarray(tmp_path, base_array):
    path = tmp_path / "array.nc"
    base_array.to_netcdf(path)
    tensor = open_datatensor(path)
    np.testing.assert_allclose(tensor.data.numpy(), base_array.data)
    assert tensor.dims == base_array.dims


def test_open_dataset(tmp_path, base_dataset):
    path = tmp_path / "dataset.nc"
    base_dataset.to_netcdf(path)
    dataset = open_dataset(path)
    temp = dataset["temp"]
    wind = dataset["wind"]
    np.testing.assert_allclose(temp.data.numpy(), base_dataset["temp"].data)
    np.testing.assert_allclose(wind.data.numpy(), base_dataset["wind"].data)
    assert set(dataset.dims) == set(base_dataset.dims)


def test_read_pickle_series(tmp_path):
    path = tmp_path / "series.pkl"
    index = pd.date_range("2000-01-01", periods=4, freq="D", name="time")
    series = pd.Series([10, 20, 30, 40], index=index)
    series.to_pickle(path)

    tensor = read_pickle(path)

    assert tensor.dims == ("time",)
    coord = tensor["time"]
    assert isinstance(coord, pd.DatetimeIndex)
    assert coord.equals(index)

    limited = tensor.sel(time=slice(None, "2000-01-02"))
    expected = series.loc[: "2000-01-02"].to_numpy()
    np.testing.assert_allclose(limited.data.cpu().numpy(), expected)


def test_dataset_to_datatensor_matches_xarray(base_dataset):
    xr_ds = base_dataset.assign_coords(scalar=10).assign_attrs(source="xt")
    ds = Dataset.from_xarray(xr_ds)
    tensor = ds.to_datatensor(name="combined")
    xr_array = xr_ds.to_dataarray(name="combined")

    np.testing.assert_allclose(tensor.data.cpu().numpy(), xr_array.data)
    assert tensor.dims == xr_array.dims
    assert tensor.name == "combined"
    assert tensor.attrs == xr_array.attrs

    for dim in tensor.dims:
        xt_coord = _as_numpy(tensor.coords[dim])
        xr_coord = _as_numpy(xr_array.coords[dim].values)
        np.testing.assert_array_equal(xt_coord, xr_coord)
    np.testing.assert_array_equal(_as_numpy(tensor.coords["scalar"]), _as_numpy(xr_array.coords["scalar"].values))


def test_datatensor_to_dataset_dim_roundtrip(base_dataset):
    xr_ds = base_dataset.assign_coords(scalar=10)
    ds = Dataset.from_xarray(xr_ds)
    tensor = ds.to_datatensor()
    xt_roundtrip = tensor.to_dataset(dim="variable")
    xr_roundtrip = xr_ds.to_dataarray().to_dataset(dim="variable")
    xr.testing.assert_identical(xt_roundtrip.to_xarray(), xr_roundtrip)


def test_datatensor_to_dataset_requires_name(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    with pytest.raises(ValueError):
        tensor.to_dataset()

    tensor.name = "values"
    expected = base_array.rename("values").to_dataset()
    result = tensor.to_dataset()
    xr.testing.assert_identical(result.to_xarray(), expected)


def test_datatensor_to_dataset_promotes_attrs(base_array):
    named = base_array.rename("signal")
    named = named.assign_attrs(units="m/s")
    tensor = DataTensor.from_dataarray(named)

    ds_no_attrs = tensor.to_dataset()
    assert ds_no_attrs.attrs == {}

    ds_attrs = tensor.to_dataset(promote_attrs=True)
    assert ds_attrs.attrs == {"units": "m/s"}


def test_datatensor_to_dataset_disallows_dim_and_name(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    with pytest.raises(TypeError):
        tensor.to_dataset(dim="x", name="bad")


def test_dataset_to_datatensor_requires_variables():
    ds = Dataset({})
    with pytest.raises(ValueError):
        ds.to_datatensor()
