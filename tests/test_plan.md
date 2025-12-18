# Test Plan

This document outlines the intended breadth of automated coverage for `DataTensor`. The current test module only covers the "happy path" subset listed below; the sections describe how we will expand the suite.

1. **Construction**
   - Validate tensor conversion from numpy / torch (preserving dtype and device semantics).
   - Ensure coordinate validation errors surface when sizes mismatch or dims are missing.
   - Round-trip tests for `from_pandas` (Series & DataFrame), `from_dataarray`, and plain constructor.
2. **Selection & Manipulation APIs**
   - Label selections (`sel`) with scalars, lists, chained dims, and slicing by labels with inclusive semantics.
   - Positional selections (`isel`) for negative indexes, boolean masks, fancy indexing, and broadcasting multiple dims.
   - Ensure the same semantics as `xarray.DataArray` for dimension dropping vs. retaining via explicit list inputs.
   - Layout utilities: `transpose`, `expand_dims`, and `to_pandas` conversions for validation of dimension metadata.
3. **Reduction Ops**
   - Compare `mean`, `std`, `sum`, `prod`, `min`, `max` against `xarray` results across dim combinations and `keepdims` toggles.
   - Stress-tests for empty dims, singleton dims, and dtype promotion (float16, float64, ints).
4. **Elementwise Math / Alignment**
   - Operations between tensors with aligned coords, mismatched dims (should raise), and partially overlapping coords (future support).
   - Broadcasting against scalars, numpy arrays, and raw `torch.Tensor` inputs.
5. **Device / Autograd**
   - `.to` migrations across CPU/GPU and dtype conversions.
   - Gradient propagation through the wrapper using `torch.autograd.grad` to ensure differentiable reductions and selections.
6. **Interoperability**
   - `.isel` and `.sel` round-trip conversions back to `xarray.DataArray`.
   - Serialization tests (pickling) for distributed workloads.

Progress will follow this order so that every regression is caught early and each feature increment ships with guard-rails.
