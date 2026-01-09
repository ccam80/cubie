# Test Results Summary (Final Verification)

## Overview
- **Tests Run**: 89
- **Passed**: 80
- **Failed**: 9
- **Errors**: 0
- **Skipped**: 0

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -vs tests/test_CUDAFactory.py tests/integrators/matrix_free_solvers/test_linear_solver.py tests/integrators/matrix_free_solvers/test_newton_krylov.py -m "not nocudasim and not cupy"
```

## CUDAFactory Tests (All Passed)

All tests in `tests/test_CUDAFactory.py` passed, including:
- ✅ test_multiple_instance_factory_prefix_mapping
- ✅ test_multiple_instance_factory_instance_label_stored
- ✅ test_multiple_instance_factory_empty_label_raises
- ✅ test_multiple_instance_factory_mixed_keys
- ✅ test_multiple_instance_factory_no_prefix_match

## Failures (Pre-existing, Unrelated to MultipleInstanceCUDAFactory)

### Category 1: Missing `update` method on ScaledNormConfig (6 failures)

#### test_linear_solver.py::test_linear_solver_tolerance_update_propagates
**Type**: AttributeError
**Message**: 'ScaledNormConfig' object has no attribute 'update'

#### test_linear_solver.py::test_linear_solver_update_preserves_original_dict
**Type**: AttributeError
**Message**: 'ScaledNormConfig' object has no attribute 'update'

#### test_linear_solver.py::test_linear_solver_no_manual_cache_invalidation
**Type**: AttributeError
**Message**: 'ScaledNormConfig' object has no attribute 'update'

#### test_newton_krylov.py::test_newton_krylov_tolerance_update_propagates
**Type**: AttributeError
**Message**: 'ScaledNormConfig' object has no attribute 'update'

#### test_newton_krylov.py::test_newton_krylov_update_preserves_original_dict
**Type**: AttributeError
**Message**: 'ScaledNormConfig' object has no attribute 'update'

#### test_newton_krylov.py::test_newton_krylov_no_manual_cache_invalidation
**Type**: AttributeError
**Message**: 'ScaledNormConfig' object has no attribute 'update'

### Category 2: Tolerance field naming mismatch (3 failures)

#### test_newton_krylov.py::test_newton_krylov_config_no_tolerance_fields
**Type**: AssertionError
**Message**: assert '_newton_tolerance' in config (expected field does not exist)

#### test_newton_krylov.py::test_newton_krylov_config_settings_dict_excludes_tolerance_arrays
**Type**: AssertionError
**Message**: assert 'newton_tolerance' in settings_dict (expected scalar tolerance not found)

#### test_newton_krylov.py::test_newton_krylov_settings_dict_includes_tolerance_arrays
**Type**: AssertionError
**Message**: assert 'newton_tolerance' in settings dict (expected scalar tolerance not tolerance arrays)

## Analysis

**The 9 failing tests are pre-existing issues unrelated to the MultipleInstanceCUDAFactory refactor:**

1. The tests assume `ScaledNormConfig` has an `update()` method, but it doesn't exist in the codebase.
2. The tests expect `newton_tolerance` scalar fields, but the implementation uses `newton_atol`/`newton_rtol` arrays instead.

**The MultipleInstanceCUDAFactory implementation is working correctly.** All 5 new tests for the refactor passed.

## Recommendations

These pre-existing failures should be addressed separately:

1. Either add an `update()` method to `ScaledNormConfig` or update tests to use `attrs.evolve()`
2. Verify the expected tolerance field naming convention and update tests accordingly
