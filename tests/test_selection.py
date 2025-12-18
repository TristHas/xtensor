import numpy as np
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
