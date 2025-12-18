import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def base_array():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(4, 3)).astype(np.float32)
    xs = np.linspace(0.0, 1.0, 4)
    ys = ["north", "south", "east"]
    return xr.DataArray(data, dims=("x", "y"), coords={"x": xs, "y": ys})
