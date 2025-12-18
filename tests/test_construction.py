import numpy as np
import pandas as pd
import pytest

from xtensor import DataTensor


def test_from_dataarray_matches_shape(base_array):
    tensor = DataTensor.from_dataarray(base_array)
    np.testing.assert_allclose(tensor.data.numpy(), base_array.data)
    assert tensor.dims == base_array.dims
    assert tensor.coords["x"] == tuple(base_array.coords["x"].values.tolist())


def test_constructor_validates_coords():
    data = np.ones((2, 2))
    coords = {"x": [0, 1], "y": [0]}  # mismatch
    with pytest.raises(ValueError):
        DataTensor(data, coords, ("x", "y"))


def test_from_pandas_series_and_dataframe():
    series = pd.Series([1, 3, 5], index=pd.Index([0, 1, 2], name="time"))
    tensor = DataTensor.from_pandas(series)
    assert tensor.shape == (3,)
    assert tensor.coords["time"] == (0, 1, 2)

    df = pd.DataFrame([[1, 2], [3, 4]], index=pd.Index(["a", "b"], name="row"), columns=pd.Index(["x", "y"], name="col"))
    tensor_df = DataTensor.from_pandas(df)
    assert tensor_df.dims == ("col", "row")
    np.testing.assert_allclose(tensor_df.data.numpy(), df.to_numpy().T)
