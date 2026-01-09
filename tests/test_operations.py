import numpy as np
import pytest
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


def test_grad_returns_datatensor():
    data = torch.arange(0.0, 6.0).reshape(2, 3).clone().detach().requires_grad_(True)
    coords = {"x": [10.0, 20.0], "y": [0, 1, 2]}
    tensor = DataTensor(data, coords, ("x", "y"))
    loss = (tensor.data ** 2).sum()
    loss.backward()
    grad_tensor = tensor.grad
    assert grad_tensor is not None
    torch.testing.assert_close(grad_tensor.data, data.grad)
    assert grad_tensor.dims == tensor.dims
    for dim in tensor.dims:
        coord = tensor.coords[dim]
        grad_coord = grad_tensor.coords[dim]
        if isinstance(coord, torch.Tensor):
            torch.testing.assert_close(grad_coord, coord)
        else:
            assert grad_coord == coord


def test_torch_elementwise_dispatch(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    other = tensor + 1.0

    added = torch.add(tensor, other)
    torch.testing.assert_close(added.data, tensor.data + other.data)
    assert added.dims == tensor.dims

    scaled = torch.mul(tensor, 3.0)
    torch.testing.assert_close(scaled.data, tensor.data * 3.0)

    subtracted = torch.sub(other.data, tensor)
    torch.testing.assert_close(subtracted.data, other.data - tensor.data)

    divided = torch.true_divide(other, tensor + 2.0)
    torch.testing.assert_close(divided.data, other.data / (tensor.data + 2.0))

    pw = torch.pow(tensor, 2.0)
    torch.testing.assert_close(pw.data, tensor.data ** 2)

    minimum = torch.minimum(tensor, tensor + 5.0)
    torch.testing.assert_close(minimum.data, tensor.data)


def test_elementwise_aligns_dimension_order(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    other = DataTensor.from_dataarray(base_array.transpose("y", "x"))
    summed = tensor + other
    np.testing.assert_allclose(summed.data.numpy(), (base_array + base_array).data)


def test_elementwise_coordinate_mismatch_raises():
    left = DataTensor(
        torch.arange(4.0).reshape(2, 2),
        {"x": [0, 1], "y": [10, 20]},
        ("x", "y"),
    )
    right = DataTensor(
        torch.arange(4.0).reshape(2, 2),
        {"x": [0, 2], "y": [10, 20]},
        ("x", "y"),
    )
    with pytest.raises(ValueError, match="requires matching coordinates"):
        _ = left + right


def test_elementwise_missing_dimension_raises():
    base = DataTensor(
        torch.arange(4.0).reshape(2, 2),
        {"x": [0, 1], "y": [0, 1]},
        ("x", "y"),
    )
    extra = DataTensor(
        torch.arange(2.0),
        {"x": [0, 1]},
        ("x",),
    )
    with pytest.raises(ValueError, match="dimension set"):
        _ = base + extra
