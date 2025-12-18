# xtensor

`xtensor` provides a `DataTensor` class that wraps a `torch.Tensor` with labels and coordinates inspired by `xarray.DataArray`. The goal is to enjoy lightweight labeled computations without leaving eager PyTorch workflows.

## Features

- Construction from torch / numpy data with dimension names and coordinates.
- Factory helpers from `pandas` objects or existing `xarray.DataArray`.
- Label (`sel`) and positional (`isel`) slicing.
- Mean, standard deviation, sum, min, max style reductions.
- Elementwise math with scalars or other `DataTensor` objects that share the same coordinates.
- Device-agnostic `.to()` mirror of `torch.Tensor.to`.

## Development

Install dependencies in editable mode.

```bash
pip install -e ".[dev]"
```

Run the tests.

```bash
pytest
```
