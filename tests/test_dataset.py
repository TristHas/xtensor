import numpy as np
import xarray as xr
import torch

from xtensor import DataTensor, Dataset


def _assert_identical(xt_ds: Dataset, xr_ds: xr.Dataset) -> None:
    xr.testing.assert_identical(xt_ds.to_xarray(), xr_ds)


def test_dataset_roundtrip_matches_xarray(base_dataset):
    ds = Dataset.from_xarray(base_dataset)
    _assert_identical(ds, base_dataset)


def test_dataset_selection_matches_xarray(base_dataset):
    ds = Dataset.from_xarray(base_dataset)
    xt_sel = ds.sel(time=base_dataset.time.values[2])
    xr_sel = base_dataset.sel(time=base_dataset.time.values[2])
    _assert_identical(xt_sel, xr_sel)

    xt_isel = ds.isel(time=[0, 3], level=slice(1, None))
    xr_isel = base_dataset.isel(time=[0, 3], level=slice(1, None))
    _assert_identical(xt_isel, xr_isel)


def test_dataset_assign_coords_matches_xarray(base_dataset):
    ds = Dataset.from_xarray(base_dataset)
    shifted = base_dataset.assign_coords(time=base_dataset.time + 10.0)
    xt_shifted = ds.assign_coords(time=(base_dataset.time.values + 10.0))
    _assert_identical(xt_shifted, shifted)


def test_dataset_assign_and_rename(base_dataset):
    ds = Dataset.from_xarray(base_dataset)
    new_var = np.linspace(0.0, 1.0, base_dataset.sizes["time"])
    xr_assigned = base_dataset.assign(speed=("time", new_var)).rename({"temp": "temperature"})
    xt_assigned = ds.assign(speed=(("time",), new_var)).rename({"temp": "temperature"})
    _assert_identical(xt_assigned, xr_assigned)


def test_dataset_transpose_and_squeeze(base_dataset):
    ds = Dataset.from_xarray(base_dataset)
    xr_transposed = base_dataset.transpose("level", "time")
    xt_transposed = ds.transpose("level", "time")
    _assert_identical(xt_transposed, xr_transposed)

    xr_squeezed = xr_transposed.expand_dims(batch=[0]).squeeze()
    xt_squeezed = Dataset.from_xarray(xr_transposed.expand_dims(batch=[0])).squeeze()
    _assert_identical(xt_squeezed, xr_squeezed)


def test_dataset_coordinate_precedence(base_dataset):
    ds = Dataset.from_xarray(base_dataset)
    # Add a data variable with the same name as an existing coordinate
    time_coord = ds["time"]
    coord_values = time_coord.data
    ds = ds.assign(new_time=DataTensor(coord_values + 1.0, {"time": coord_values}, ("time",)))
    ds["time"] = ds["time"] + 1.0
    coord = ds["time"]
    torch.testing.assert_close(coord.data, coord_values + 1.0)
    torch.testing.assert_close(ds.data_vars["new_time"].data, coord_values + 1.0)
