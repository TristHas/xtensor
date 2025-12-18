import numpy as np

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
