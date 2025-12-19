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


@pytest.fixture
def base_dataset():
    rng = np.random.default_rng(1)
    temp = rng.normal(size=(5, 3)).astype(np.float64)
    wind = rng.normal(size=5).astype(np.float64)
    time = np.linspace(0.0, 4.0, 5)
    level = np.array([1000.0, 850.0, 700.0])
    ds = xr.Dataset(
        {
            "temp": (("time", "level"), temp),
            "wind": (("time",), wind),
        },
        coords={"time": time, "level": level},
    )
    return ds
