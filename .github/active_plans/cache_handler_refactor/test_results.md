# Test Results Summary

## Overview
- **Tests Run**: 97
- **Passed**: 97
- **Failed**: 0
- **Errors**: 0
- **Skipped**: 0

## Test Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short tests/test_cubie_cache.py tests/batchsolving/test_solver.py
```

## Fixes Applied

Two bugs were discovered and fixed during this test run:

### 1. TypeError: 'str' object is not callable (BatchSolverKernel.py:554)

**File**: `src/cubie/batchsolving/BatchSolverKernel.py`
**Line**: 554
**Issue**: `config_hash` was being called as a method with `self.config_hash()`, but it's a property on the `CUDAFactory` base class that returns a string.
**Fix**: Changed `self.config_hash()` to `self.config_hash` (removed parentheses).

### 2. AssertionError: cache_mode not set correctly (cubie_cache.py:706)

**File**: `src/cubie/cubie_cache.py`
**Lines**: 706-712
**Issue**: `kwargs.update(config_params)` was overwriting user-provided kwargs (like `cache_mode="flush_on_change"`) with defaults from `params_from_user_kwarg`.
**Fix**: Changed merge order to `config_params.update(kwargs)` so user-provided kwargs override defaults, then use `config_params` in `build_config`.

## Result

All 97 tests now pass successfully.

## Warnings

The test run produced 3014 warnings, primarily:
- NumbaDeprecationWarning about `nopython=False` (55 warnings)
- UserWarning about unrecognized parameters (27 warnings)
- DeprecationWarning about bitwise inversion on bool (2928 warnings) - these are in the integrator loop code and unrelated to the cache handler refactor
