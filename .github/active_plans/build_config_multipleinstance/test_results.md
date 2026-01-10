# Test Results Summary

## Overview
- **Tests Run**: 117
- **Passed**: 51
- **Failed**: 60
- **Errors**: 6
- **Skipped**: 0

## Critical Issues

### 1. TypeError: unhashable type: 'set' (Most Common - 45 failures)

**Location**: `src/cubie/CUDAFactory.py:647`

**Root Cause**: In `MultipleInstanceCUDAFactoryConfig.update()`, the `super().update()` call returns a tuple `(recognized, changed)` but the code is iterating over it as if it were a set.

**Code Issue**:
```python
recognized_prefixed = super().update(all_updates)  # Returns tuple (set, set)

recognized = set()
for key in recognized_prefixed:  # Iterating over tuple, not set
    if key in self.prefixed_attributes:
        recognized.add(f"{self.prefix}_{key}")
    else:
        recognized.add(key)  # key is a set, not a string - causes TypeError
```

**Affected Tests**:
- All LinearSolver tests (22 failures + 6 errors)
- All NewtonKrylov tests (23 failures)

### 2. AttributeError: 'TestConfig' object has no attribute 'atol'

**Location**: `tests/test_CUDAFactory.py`

**Root Cause**: Test fixtures use `_atol` as an internal attribute with underscore, but tests access `atol` (without underscore). The TestConfig class needs a property or the tests need to use the underscore version.

**Affected Tests**:
- `test_build_config_with_instance_label`
- `test_build_config_instance_label_non_prefixed_class`
- `test_build_config_instance_label_prefixed_takes_precedence`
- `test_build_config_backward_compatible_no_instance_label`

### 3. AttributeError: 'LinearSolver' object has no attribute 'instance_label'

**Location**: `tests/test_CUDAFactory.py::test_multiple_instance_factory_instance_label_stored`

**Root Cause**: The `LinearSolver` class does not expose `instance_label` as a public attribute. Either the attribute is missing or it's stored with a different name.

**Affected Tests**:
- `test_multiple_instance_factory_instance_label_stored`
- `test_linear_solver_forwards_kwargs_to_norm`

### 4. ValueError: instance_label cannot be empty or None

**Location**: `tests/integrators/test_norms.py`

**Root Cause**: `ScaledNorm` now requires a non-empty `instance_label`, but tests are instantiating it without providing one. Tests need to either:
1. Provide a valid instance_label, OR
2. ScaledNorm should allow empty instance_label for standalone use

**Affected Tests**:
- `test_scaled_norm_converged_when_under_tolerance`
- `test_scaled_norm_exceeds_when_over_tolerance`
- `test_scaled_norm_update_invalidates_cache`
- `test_scaled_norm_factory_builds_device_function`
- `test_scaled_norm_instance_label_empty_no_prefix` (ironically testing empty case)

### 5. KeyError: "'test_value' is not a valid compile setting"

**Location**: `tests/test_CUDAFactory.py::test_multiple_instance_factory_mixed_keys`

**Root Cause**: The test is trying to update with a key `test_value` that isn't recognized by the config.

### 6. AttributeError: 'LinearSolverConfig' object has no attribute 'kyrlov_max_iters'

**Location**: Multiple tests

**Root Cause**: Typo in test code - `kyrlov` instead of `krylov`.

**Affected Tests**:
- `test_linear_solver_config_settings_dict_excludes_tolerance_arrays`
- `test_linear_solver_settings_dict_includes_tolerance_arrays`
- `test_linear_solver_no_manual_cache_invalidation`
- `test_newton_krylov_settings_dict_includes_tolerance_arrays`

### 7. AttributeError: 'NewtonKrylovConfig' object has no attribute 'newton_max_iters'

**Location**: `tests/integrators/matrix_free_solvers/test_newton_krylov.py`

**Root Cause**: The config doesn't have `newton_max_iters`, it has `max_iters`.

**Affected Tests**:
- `test_newton_krylov_config_settings_dict_excludes_tolerance_arrays`
- `test_newton_krylov_init_with_newton_prefixed_kwargs`

### 8. AssertionError: assert None is not None

**Location**: `test_newton_krylov_no_manual_cache_invalidation`

**Root Cause**: `NewtonKrylovConfig.norm_device_function` is None when the test expects it to be populated.

## Recommendations

### Priority 1: Fix MultipleInstanceCUDAFactoryConfig.update() (45+ failures)
The `update()` method in `CUDAFactory.py` line 647 needs to properly unpack the tuple returned by `super().update()`:

```python
# Change from:
recognized_prefixed = super().update(all_updates)

# Change to:
recognized_prefixed, changed_prefixed = super().update(all_updates)
```

Then update the return statement to return `(recognized, changed)` tuple.

### Priority 2: Fix ScaledNorm instance_label requirement (5 failures)
Either:
1. Make `instance_label` optional with a default value for standalone use, OR
2. Update all test fixtures to provide a valid `instance_label`

### Priority 3: Fix attribute name typos in tests (4 failures)
- Change `kyrlov_max_iters` to `krylov_max_iters` in test files

### Priority 4: Fix test fixture attribute access (4 failures)
- Update `TestConfig` to expose `atol` property, OR
- Update tests to access `_atol`

### Priority 5: Expose instance_label on LinearSolver/ScaledNorm (2 failures)
Add public `instance_label` property to classes that use it.

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -vs tests/test_CUDAFactory.py tests/integrators/test_norms.py tests/integrators/matrix_free_solvers/test_base_solver.py tests/integrators/matrix_free_solvers/test_linear_solver.py tests/integrators/matrix_free_solvers/test_newton_krylov.py -m "not nocudasim and not cupy"
```
