import numpy as np
import torch
import xarray as xr

from xtensor import DataTensor


def test_transpose_and_expand_dims():
    tensor = DataTensor(
        np.arange(6).reshape(2, 3),
        {"x": ["a", "b"], "y": [0, 1, 2]},
        ("x", "y"),
    )

    transposed = tensor.transpose("y", "x")
    assert transposed.dims == ("y", "x")
    np.testing.assert_allclose(transposed.data.numpy(), tensor.data.numpy().T)

    expanded = tensor.expand_dims("batch")
    assert expanded.dims == ("batch", "x", "y")
    assert expanded.shape[0] == 1


def test_squeeze():
    tensor = DataTensor(
        np.arange(6).reshape(1, 2, 3),
        {"batch": [0], "x": ["a", "b"], "y": [0, 1, 2]},
        ("batch", "x", "y"),
    )
    squeezed = tensor.squeeze("batch")
    assert squeezed.dims == ("x", "y")
    np.testing.assert_allclose(squeezed.data.numpy(), tensor.data.squeeze(0).numpy())


def test_assign_coords_matches_xarray(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    new_x = np.linspace(-1.0, 1.0, base_array.sizes["x"])
    reassigned = tensor.assign_coords(x=new_x)
    xr_expected = base_array.assign_coords(x=new_x)
    np.testing.assert_allclose(reassigned.data.numpy(), xr_expected.data)
    coord = reassigned.coords["x"]
    if isinstance(coord, torch.Tensor):
        np.testing.assert_allclose(coord.cpu().numpy(), new_x)
    else:
        assert coord == tuple(new_x)


def test_rename_and_astype_align_with_xarray(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    renamed = tensor.rename({"x": "lon", "y": "lat"}).astype(np.float64)
    xr_expected = base_array.rename({"x": "lon", "y": "lat"}).astype(np.float64)
    assert renamed.dims == xr_expected.dims
    np.testing.assert_allclose(renamed.data.numpy(), xr_expected.data)
