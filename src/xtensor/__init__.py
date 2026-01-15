from typing import Any

from .datatensor import DataTensor, concat
from .dataset import Dataset

def open_datatensor(*args: Any, **kwargs: Any) -> DataTensor:
    """Open a DataTensor from inputs accepted by xarray.open_dataarray."""
    try:
        import xarray as xr
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("xarray must be installed to open a DataArray.") from error

    data_array = xr.open_dataarray(*args, **kwargs)
    try:
        return DataTensor.from_dataarray(data_array)
    finally:
        data_array.close()

def open_dataset(*args: Any, **kwargs: Any) -> Dataset:
    """Open a Dataset from inputs accepted by xarray.open_dataset."""
    try:
        import xarray as xr
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("xarray must be installed to open a Dataset.") from error

    ds = xr.open_dataset(*args, **kwargs)
    try:
        return Dataset.from_xarray(ds)
    finally:
        ds.close()

def read_pickle(*args: Any, dims=None, **kwargs: Any) -> DataTensor:
    """Load a pickled pandas object and convert it into a DataTensor."""
    import pandas as pd
    obj = pd.read_pickle(*args, **kwargs)
    return DataTensor.from_pandas(obj, dims=dims)

def read_feather(*args: Any, dims=None, **kwargs: Any) -> DataTensor:
    """Load a pickled pandas object and convert it into a DataTensor."""
    import pandas as pd
    obj = pd.read_feather(*args, **kwargs)
    return DataTensor.from_pandas(obj, dims=dims)

from_pandas = DataTensor.from_pandas
from_dataarray = DataTensor.from_dataarray

__all__ = ["DataTensor", "Dataset", "concat", "open_datatensor", "open_dataset", "read_pickle"]
