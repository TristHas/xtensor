import numpy as np
import torch
import xarray as xr

from xtensor import DataTensor


def test_reductions_align_with_xarray(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    xr_mean = base_array.mean(dim="x")
    xr_std = base_array.std(dim="y")
    xr_sum_all = base_array.sum()

    np.testing.assert_allclose(tensor.mean(dim="x").data.numpy(), xr_mean.data)
    np.testing.assert_allclose(tensor.std(dim="y").data.numpy(), xr_std.data, atol=1e-6)
    np.testing.assert_allclose(tensor.sum().data.numpy(), xr_sum_all.data, atol=1e-6)


def test_keepdims_flag(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    kept = tensor.mean(dim="x", keepdims=True)
    assert kept.dims == tensor.dims
    assert kept.shape[0] == 1
    expected = base_array.coords["x"].values[:1]
    coord = kept.coords["x"]
    if isinstance(coord, torch.Tensor):
        np.testing.assert_allclose(coord.cpu().numpy(), expected)
    else:
        np.testing.assert_allclose(np.asarray(coord), expected)


def test_torch_reduction_dispatch(base_array):
    tensor = DataTensor.from_dataarray(base_array)

    summed = torch.sum(tensor, dim=1)
    expected = tensor.sum(dim="y")
    torch.testing.assert_close(summed.data, expected.data)
    assert summed.dims == expected.dims

    kept = torch.mean(tensor, dim=-1, keepdim=True)
    expected_keep = tensor.mean(dim="y", keepdims=True)
    torch.testing.assert_close(kept.data, expected_keep.data)
    assert kept.dims == expected_keep.dims

    prod = torch.prod(tensor, dim=0)
    expected_prod = tensor.prod(dim="x")
    torch.testing.assert_close(prod.data, expected_prod.data)

    std = torch.std(tensor, dim=0, unbiased=False, keepdim=True)
    expected_std = tensor.std(dim="x", keepdims=True, unbiased=False)
    torch.testing.assert_close(std.data, expected_std.data)

    dtype_sum = torch.sum(tensor, dtype=torch.float64)
    assert dtype_sum.data.dtype == torch.float64


def test_nan_aware_mean_var_std():
    data = np.array(
        [
            [np.nan, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, np.nan],
        ],
        dtype=np.float64,
    )
    array = xr.DataArray(data, dims=("x", "y"))
    tensor = DataTensor.from_dataarray(array)

    np.testing.assert_allclose(tensor.mean().data.numpy(), array.mean().data, equal_nan=True)
    np.testing.assert_allclose(
        tensor.mean(dim="x").data.numpy(),
        array.mean(dim="x").data,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        tensor.std(dim="y", unbiased=False).data.numpy(),
        array.std(dim="y").data,
        rtol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        tensor.var(dim="y", unbiased=False).data.numpy(),
        array.var(dim="y").data,
        rtol=1e-6,
        equal_nan=True,
    )


def test_nan_reductions_return_nan_when_no_valid_values():
    data = np.array([[np.nan, np.nan], [1.0, 2.0]], dtype=np.float32)
    array = xr.DataArray(data, dims=("x", "y"))
    tensor = DataTensor.from_dataarray(array)

    xr_mean = array.mean(dim="x")
    xt_mean = tensor.mean(dim="x")
    np.testing.assert_allclose(xt_mean.data.numpy(), xr_mean.data, equal_nan=True)

    xr_var = array.var(dim="x")
    xt_var = tensor.var(dim="x")
    np.testing.assert_allclose(xt_var.data.numpy(), xr_var.data, equal_nan=True)


def test_any_reduction_matches_torch():
    data = torch.tensor([[False, False, True], [False, False, False]], dtype=torch.bool)
    tensor = DataTensor(data, {"x": [0, 1], "y": [0, 1, 2]}, ("x", "y"))

    reduced = tensor.any(dim="y")
    torch.testing.assert_close(reduced.data, data.any(dim=1))

    kept = tensor.any(dim="y", keepdims=True)
    torch.testing.assert_close(kept.data, data.any(dim=1, keepdim=True))
