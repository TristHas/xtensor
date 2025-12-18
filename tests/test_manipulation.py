import numpy as np

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
