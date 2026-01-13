import numpy as np
import pytest
import torch
import torch.nn.functional as F

from xtensor import DataTensor


def _simple_tensor(values, *, dtype=torch.float32):
    data = torch.as_tensor(values, dtype=dtype).reshape(2, 3)
    coords = {"x": torch.arange(2), "y": torch.arange(3)}
    return DataTensor(data, coords, ("x", "y"))


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


def test_elementwise_missing_dimension_broadcasts():
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
    summed = base + extra
    expected = torch.arange(4.0).reshape(2, 2) + torch.arange(2.0).reshape(2, 1)
    torch.testing.assert_close(summed.data, expected)
    assert summed.dims == ("x", "y")


def test_torch_isnan_dispatch_preserves_coordinates():
    data = torch.tensor([[float("nan"), 1.0], [2.0, float("nan")]], dtype=torch.float32)
    tensor = DataTensor(data, {"x": [0, 1], "y": ["a", "b"]}, ("x", "y"))

    mask = torch.isnan(tensor)
    torch.testing.assert_close(mask.data, torch.isnan(data))
    assert mask.data.dtype == torch.bool
    assert mask.dims == tensor.dims
    assert tuple(mask.coords["y"]) == tuple(tensor.coords["y"])

def test_torch_nan_to_num_dispatch():
    data = torch.tensor([[float("nan"), float("inf")], [-float("inf"), 1.0]], dtype=torch.float32)
    tensor = DataTensor(data, {"dim": torch.arange(2), "col": ["a", "b"]}, ("dim", "col"))

    cleaned = torch.nan_to_num(tensor, nan=0.0, posinf=5.0, neginf=-5.0)
    expected = torch.nan_to_num(data, nan=0.0, posinf=5.0, neginf=-5.0)
    torch.testing.assert_close(cleaned.data, expected)
    assert cleaned.dims == tensor.dims
    torch.testing.assert_close(cleaned.coords["dim"], tensor.coords["dim"])


@pytest.mark.parametrize(
    "func_name",
    ["sin", "cos", "tan", "sinh", "cosh", "tanh", "asinh", "atan", "exp", "expm1", "square"],
)
def test_unary_math_dispatch(func_name):
    values = torch.linspace(-0.5, 0.5, steps=6)
    tensor = _simple_tensor(values)
    torch_func = getattr(torch, func_name)
    result = torch_func(tensor)
    expected = torch_func(tensor.data)
    torch.testing.assert_close(result.data, expected)
    assert result.dims == tensor.dims


@pytest.mark.parametrize("func_name", ["asin", "acos", "atanh"])
def test_inverse_trig_dispatch(func_name):
    values = torch.linspace(-0.8, 0.8, steps=6)
    tensor = _simple_tensor(values)
    torch_func = getattr(torch, func_name)
    result = torch_func(tensor)
    expected = torch_func(tensor.data)
    torch.testing.assert_close(result.data, expected)


@pytest.mark.parametrize("func_name", ["log", "log10", "log1p", "sqrt", "rsqrt", "reciprocal"])
def test_positive_unary_dispatch(func_name):
    values = torch.linspace(1.5, 3.0, steps=6)
    tensor = _simple_tensor(values)
    torch_func = getattr(torch, func_name)
    result = torch_func(tensor)
    expected = torch_func(tensor.data)
    torch.testing.assert_close(result.data, expected)


@pytest.mark.parametrize("func_name", ["floor", "ceil", "trunc", "round", "frac"])
def test_rounding_dispatch(func_name):
    values = torch.tensor([[-1.75, -0.25, 0.25], [0.5, 1.25, 2.5]])
    tensor = _simple_tensor(values.reshape(-1))
    torch_func = getattr(torch, func_name)
    result = torch_func(tensor)
    expected = torch_func(tensor.data)
    torch.testing.assert_close(result.data, expected)


def test_sigmoid_relu_soft_dispatch():
    values = torch.linspace(-2.0, 2.0, steps=6)
    tensor = _simple_tensor(values)

    sigmoid = torch.sigmoid(tensor)
    torch.testing.assert_close(sigmoid.data, torch.sigmoid(tensor.data))

    relu = torch.relu(tensor)
    torch.testing.assert_close(relu.data, torch.relu(tensor.data))

    softplus = F.softplus(tensor)
    torch.testing.assert_close(softplus.data, F.softplus(tensor.data))

    softsign = F.softsign(tensor)
    torch.testing.assert_close(softsign.data, F.softsign(tensor.data))


def test_sign_and_signbit_dispatch():
    values = torch.tensor([[-2.0, -0.0, 0.0], [1.5, -3.5, 4.0]])
    tensor = _simple_tensor(values.reshape(-1))
    signed = torch.sign(tensor)
    torch.testing.assert_close(signed.data, torch.sign(tensor.data))

    signbit = torch.signbit(tensor)
    torch.testing.assert_close(signbit.data, torch.signbit(tensor.data))
    assert signbit.data.dtype == torch.bool


def test_isfinite_isinf_isreal_dispatch():
    values = torch.tensor([[float("nan"), float("inf"), -float("inf")], [1.0, -2.0, 3.0]])
    tensor = _simple_tensor(values.reshape(-1))
    finite = torch.isfinite(tensor)
    torch.testing.assert_close(finite.data, torch.isfinite(tensor.data))
    inf = torch.isinf(tensor)
    torch.testing.assert_close(inf.data, torch.isinf(tensor.data))
    real = torch.isreal(tensor)
    torch.testing.assert_close(real.data, torch.isreal(tensor.data))


def test_logical_unary_dispatch():
    values = torch.tensor([[True, False, True], [False, True, False]])
    tensor = _simple_tensor(values.reshape(-1), dtype=torch.bool)
    inverted = torch.logical_not(tensor)
    torch.testing.assert_close(inverted.data, torch.logical_not(tensor.data))


def test_bitwise_not_dispatch():
    values = torch.arange(6, dtype=torch.int64)
    tensor = _simple_tensor(values, dtype=torch.int64)
    flipped = torch.bitwise_not(tensor)
    torch.testing.assert_close(flipped.data, torch.bitwise_not(tensor.data))


def test_logical_binary_dispatch():
    a_values = torch.tensor([[True, False, True], [False, True, False]])
    b_values = torch.tensor([[False, False, True], [True, True, False]])
    lhs = _simple_tensor(a_values.reshape(-1), dtype=torch.bool)
    rhs = _simple_tensor(b_values.reshape(-1), dtype=torch.bool)
    torch.testing.assert_close(torch.logical_and(lhs, rhs).data, torch.logical_and(lhs.data, rhs.data))
    torch.testing.assert_close(torch.logical_or(lhs, rhs).data, torch.logical_or(lhs.data, rhs.data))
    torch.testing.assert_close(torch.logical_xor(lhs, rhs).data, torch.logical_xor(lhs.data, rhs.data))


def test_bitwise_binary_dispatch():
    left = _simple_tensor(torch.arange(6, dtype=torch.int32), dtype=torch.int32)
    reversed_values = torch.flip(torch.arange(6, dtype=torch.int32), dims=[0])
    right = _simple_tensor(reversed_values, dtype=torch.int32)
    torch.testing.assert_close(torch.bitwise_and(left, right).data, torch.bitwise_and(left.data, right.data))
    torch.testing.assert_close(torch.bitwise_or(left, right).data, torch.bitwise_or(left.data, right.data))
    torch.testing.assert_close(torch.bitwise_xor(left, right).data, torch.bitwise_xor(left.data, right.data))


def test_comparison_dispatch():
    base = _simple_tensor(torch.linspace(-1.0, 1.0, steps=6))
    other = _simple_tensor(torch.linspace(-0.5, 1.5, steps=6))
    torch.testing.assert_close(torch.eq(base, other).data, torch.eq(base.data, other.data))
    torch.testing.assert_close(torch.ne(base, other).data, torch.ne(base.data, other.data))
    torch.testing.assert_close(torch.lt(base, other).data, torch.lt(base.data, other.data))
    torch.testing.assert_close(torch.le(base, other).data, torch.le(base.data, other.data))
    torch.testing.assert_close(torch.gt(base, other).data, torch.gt(base.data, other.data))
    torch.testing.assert_close(torch.ge(base, other).data, torch.ge(base.data, other.data))


def test_atan2_dispatch():
    y = _simple_tensor(torch.linspace(-2.0, 2.0, steps=6))
    x = _simple_tensor(torch.linspace(1.0, 3.0, steps=6))
    result = torch.atan2(y, x)
    torch.testing.assert_close(result.data, torch.atan2(y.data, x.data))


def test_clamp_and_clip_dispatch():
    tensor = _simple_tensor(torch.linspace(-2.0, 2.0, steps=6))
    clamped = torch.clamp(tensor, min=-0.5, max=0.5)
    torch.testing.assert_close(clamped.data, torch.clamp(tensor.data, min=-0.5, max=0.5))
    clipped = torch.clip(tensor, min=-1.0, max=1.0)
    torch.testing.assert_close(clipped.data, torch.clip(tensor.data, min=-1.0, max=1.0))


def test_where_dispatch():
    condition = _simple_tensor(torch.tensor([[True, False, True], [False, True, False]]).reshape(-1), dtype=torch.bool)
    x = _simple_tensor(torch.arange(6, dtype=torch.float32))
    y = _simple_tensor(torch.ones(6, dtype=torch.float32))
    result = torch.where(condition, x, y)
    expected = torch.where(condition.data, x.data, y.data)
    torch.testing.assert_close(result.data, expected)
    assert result.dims == condition.dims
