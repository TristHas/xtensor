import numpy as np
import pandas as pd
import torch

from xtensor import DataTensor


def test_label_and_index_selection(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    label = tensor.sel(x=base_array.x.values[1], y=["north", "east"])
    xp = base_array.sel(x=base_array.x.values[1], y=["north", "east"])
    np.testing.assert_allclose(label.data.numpy(), xp.data)

    indexed = tensor.isel(x=slice(1, 3), y=[0, 2])
    xp_indexed = base_array.isel(x=slice(1, 3), y=[0, 2])
    np.testing.assert_allclose(indexed.data.numpy(), xp_indexed.data)


def test_selector_is_differentiable():
    data = torch.arange(0.0, 12.0).reshape(3, 4)
    data = data.clone().detach().requires_grad_(True)
    tensor = DataTensor(data, {"x": [0, 1, 2], "y": [0, 1, 2, 3]}, ("x", "y"))
    sliced = tensor.sel(x=1).data.sum()
    sliced.backward()
    expected = torch.zeros_like(data)
    expected[1, :] = 1.0
    torch.testing.assert_close(data.grad, expected)


def test_getitem_matches_xarray(base_array):
    tensor = DataTensor.from_dataarray(base_array)

    xp_first = base_array[1]
    xt_first = tensor[1]
    np.testing.assert_allclose(xt_first.data.numpy(), xp_first.data)
    assert xt_first.dims == xp_first.dims

    xp_slice = base_array[:, 1]
    xt_slice = tensor[:, 1]
    np.testing.assert_allclose(xt_slice.data.numpy(), xp_slice.data)
    assert xt_slice.dims == xp_slice.dims

    xp_scalar = base_array[1, 2]
    xt_scalar = tensor[1, 2]
    assert xt_scalar.data.item() == xp_scalar.data.item()
    assert xt_scalar.dims == xp_scalar.dims

    xp_dict = base_array[{"x": slice(1, None), "y": [0, 2]}]
    xt_dict = tensor[{"x": slice(1, None), "y": [0, 2]}]
    np.testing.assert_allclose(xt_dict.data.numpy(), xp_dict.data)
    assert xt_dict.dims == xp_dict.dims

    xp_ellipsis = base_array[..., 1]
    xt_ellipsis = tensor[..., 1]
    np.testing.assert_allclose(xt_ellipsis.data.numpy(), xp_ellipsis.data)
    assert xt_ellipsis.dims == xp_ellipsis.dims


def test_isel_supports_negative_indexes(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    result = tensor.isel(x=-1)
    xp = base_array.isel(x=-1)
    np.testing.assert_allclose(result.data.numpy(), xp.data)


def test_string_coordinate_lookup_returns_raw(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    coords = tensor["y"]
    assert isinstance(coords, pd.Index)
    expected = pd.Index(base_array.coords["y"].values)
    assert coords.equals(expected)


def test_sel_accepts_coordinate_datatensor(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    selected_x = tensor.sel(x=tensor["x"])
    torch.testing.assert_close(selected_x.data, tensor.data)
    assert selected_x.dims == tensor.dims

    selected_y = tensor.sel(y=tensor["y"])
    torch.testing.assert_close(selected_y.data, tensor.data)
    assert selected_y.dims == tensor.dims
