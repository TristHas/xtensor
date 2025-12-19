import numpy as np
import pandas as pd
import pytest
import torch

from xtensor import DataTensor


def test_from_dataarray_matches_shape(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    np.testing.assert_allclose(tensor.data.numpy(), base_array.data)
    assert tensor.dims == base_array.dims
    coord = tensor.coords["x"]
    if isinstance(coord, torch.Tensor):
        np.testing.assert_allclose(coord.cpu().numpy(), base_array.coords["x"].values)
    else:
        assert coord == tuple(base_array.coords["x"].values.tolist())


def test_constructor_validates_coords():
    data = np.ones((2, 2))
    coords = {"x": [0, 1], "y": [0]}  # mismatch
    with pytest.raises(ValueError):
        DataTensor(data, coords, ("x", "y"))


def test_from_pandas_series_and_dataframe():
    series = pd.Series([1, 3, 5], index=pd.Index([0, 1, 2], name="time"))
    tensor = DataTensor.from_pandas(series)
    assert tensor.shape == (3,)
    coord = tensor.coords["time"]
    if isinstance(coord, torch.Tensor):
        np.testing.assert_array_equal(coord.cpu().numpy(), np.array([0, 1, 2]))
    else:
        assert coord == (0, 1, 2)

    df = pd.DataFrame([[1, 2], [3, 4]], index=pd.Index(["a", "b"], name="row"), columns=pd.Index(["x", "y"], name="col"))
    tensor_df = DataTensor.from_pandas(df)
    assert tensor_df.dims == ("col", "row")
    np.testing.assert_allclose(tensor_df.data.numpy(), df.to_numpy().T)
