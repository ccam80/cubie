# CPU vs GPU Interpolation Precision Analysis

This document analyzes the precision differences between the CPU and GPU
evaluation functions for the array interpolator, specifically in the
`test_wrap_vs_clamp` test case.

## Overview

The test compares values computed by:
- **GPU**: `ArrayInterpolator.evaluate_all()` in 
  `src/cubie/integrators/array_interpolator.py`
- **CPU**: `_cpu_evaluate()` in 
  `tests/integrators/test_array_interpolator.py`

Both functions evaluate polynomial splines using Horner's method, but differ
in how intermediate values are typed and cast.

Note: The `DriverEvaluator` class in
`tests/integrators/cpu_reference/cpu_utils.py` is also a CPU reference, and
its behavior matches the test's `_cpu_evaluate()` function in terms of
explicit precision casting.

## Concrete Differences Found

### 1. `scaled` Computation - Missing Explicit Cast on GPU

**GPU code** (line 382):
```python
scaled = (time - evaluation_start) * inv_resolution
```

**CPU code** (line 109):
```python
scaled = precision((time - evaluation_start) * inv_res)
```

**Difference**: The GPU code does not explicitly cast the result to
`precision`. When `time` is a float32 and all other operands are float32,
CUDA may still perform intermediate computations at higher precision or with
FMA (Fused Multiply-Add) operations that have different rounding behavior.

The CPU code explicitly casts the entire expression result to `precision`,
ensuring the value is truncated/rounded to float32 before subsequent
operations.

### 2. `tau` Computation (Non-Wrap Case) - Missing Outer Cast on GPU

**GPU code** (line 395):
```python
tau = scaled - precision(seg)
```

**CPU code** (line 127):
```python
tau = precision(scaled - precision(float(segment)))
```

**Difference**: The GPU code casts `seg` (int32) to `precision` and subtracts
from `scaled`, but does not cast the final result. The CPU code:
1. Converts `segment` to `float`
2. Casts to `precision`
3. Subtracts from `scaled`
4. Casts the result to `precision`

The additional outer cast on CPU ensures the `tau` value is explicitly
represented in the target precision before Horner evaluation.

### 3. `in_range` Comparison - Different Type for `num_segments`

**GPU code** (line 391):
```python
in_range = (scaled >= precision(0.0)) and (scaled <= num_segments)
# Note: num_segments is int32(self.num_segments)
```

**CPU code** (lines 120-122):
```python
in_range = (
    scaled >= precision(0.0)
    and scaled <= precision(float(num_segments))
)
```

**Difference**: The GPU compares `scaled` (float32) directly with
`num_segments` (int32). The CPU explicitly converts `num_segments` to float
then to `precision`. While this is unlikely to cause precision errors in the
output values (it only affects the `in_range` boolean), it represents an
asymmetry in type handling.

## Impact

These differences can cause discrepancies of approximately 2-3 ULPs (Units in
Last Place) for float32 values, which corresponds to about 3e-7 absolute
error for values near 1.0. This is at the edge of or slightly beyond the
`tolerance.rel_tight = 1e-7` test threshold.

## Recommended Fix

To ensure mathematical equivalence between CPU and GPU implementations, the
GPU code should add explicit precision casts:

```python
# Line 382: Add cast to precision
scaled = precision((time - evaluation_start) * inv_resolution)

# Line 391: Cast num_segments for comparison
in_range = (scaled >= precision(0.0)) and (scaled <= precision(num_segments))

# Line 395: Add outer cast to result
tau = precision(scaled - precision(seg))
```

These changes would make the GPU code explicitly match the precision handling
of the CPU reference implementation.
