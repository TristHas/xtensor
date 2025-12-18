import numpy as np
import torch

from xtensor import DataTensor


def test_elementwise_operations_align_with_xarray(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    result = tensor * 2 + 1
    xp = base_array * 2 + 1
    np.testing.assert_allclose(result.data.numpy(), xp.data)

    other = DataTensor.from_dataarray(base_array)
    combined = tensor + other
    np.testing.assert_allclose(combined.data.numpy(), (base_array + base_array).data)


def test_operations_support_broadcasting():
    data = torch.arange(0, 6, dtype=torch.float32).reshape(2, 3)
    other = torch.tensor([[1.0], [2.0]])
    tensor = DataTensor(data, {"x": ["a", "b"], "y": [0, 1, 2]}, ("x", "y"))
    broadcast = DataTensor(other, {"x": ["a", "b"], "y": [0]}, ("x", "y"))
    combined = tensor + broadcast
    expected = data + other
    np.testing.assert_allclose(combined.data.numpy(), expected.numpy())


def test_operations_are_differentiable():
    data_a = torch.randn(2, 3, requires_grad=True)
    data_b = torch.randn(2, 3, requires_grad=True)
    tensor_a = DataTensor(data_a, {"x": [0, 1], "y": [0, 1, 2]}, ("x", "y"))
    tensor_b = DataTensor(data_b, {"x": [0, 1], "y": [0, 1, 2]}, ("x", "y"))
    loss = (tensor_a * tensor_b + 2).data.sum()
    loss.backward()
    torch.testing.assert_close(data_a.grad, data_b.detach())
    torch.testing.assert_close(data_b.grad, data_a.detach())
