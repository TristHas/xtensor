# Test Plan

The suite now focuses on guarding the fragile spots identified in the structural review.

1. **Coordinate & Index Integrity**
   - `DataTensor.coords`/`.indexes` behave like mappings (membership checks, cloning semantics).
   - keepdims reductions retain coordinate labels instead of synthesising default indexes.
   - Dataset construction from raw `torch`/`numpy` inputs preserves shared coordinate objects.
2. **Alignment Semantics**
   - Elementwise math raises when operands disagree on coordinates or dimension sets.
   - Broadcasting with singleton dimensions continues to succeed (existing coverage).
3. **Dataset APIs**
   - Direct `Dataset` construction without xarray ensures `__setitem__`, `assign_coords`, and device propagation stay consistent.
4. **Interoperability / Round-Trips**
   - xarray ↔ xtensor conversions and pandas exports remain in sync with the new coordinate plumbing.

Future additions will extend this plan with boolean/fancy indexing, dtype promotion stress tests, and serialization hooks once those APIs land.
