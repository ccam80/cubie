# Test Results Summary

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" tests/batchsolving/test_cache_config.py tests/test_cubie_cache.py -v --tb=short
```

## Overview
- **Tests Run**: 71
- **Passed**: 49
- **Failed**: 1
- **Errors**: 21
- **Skipped**: 0

## Errors (21 tests)

### All errors in tests/batchsolving/test_cache_config.py

**Type**: TypeError  
**Message**: `create_ODE_system() got an unexpected keyword argument 'equations'`

**Root Cause**: The `simple_system` fixture at line 586-598 in `test_cache_config.py` uses `equations=equations` but the actual function signature of `create_ODE_system` uses `dxdt=` as the first parameter.

**Affected Tests**:
- `TestParseCacheParam::test_parse_cache_param_true`
- `TestParseCacheParam::test_parse_cache_param_false`
- `TestParseCacheParam::test_parse_cache_param_flush_on_change`
- `TestParseCacheParam::test_parse_cache_param_path`
- `TestParseCacheParam::test_parse_cache_param_string_path`
- `TestKernelCacheConfigProperty::test_kernel_cache_config_property`
- `TestKernelCacheConfigProperty::test_kernel_cache_config_matches_compile_settings`
- `TestSetCacheDir::test_set_cache_dir_updates_config`
- `TestSetCacheDir::test_set_cache_dir_invalidates_cache`
- `TestSetCacheDir::test_set_cache_dir_accepts_string`
- `TestSetCacheDir::test_set_cache_dir_accepts_path`
- `TestSolverCacheParam::test_solver_cache_param_passed_to_kernel`
- `TestSolverCacheParam::test_solver_cache_true_default`
- `TestSolverCacheParam::test_solver_cache_false`
- `TestSolverCacheParam::test_solver_cache_flush_on_change`
- `TestSolverCacheProperties::test_solver_cache_enabled_property`
- `TestSolverCacheProperties::test_solver_cache_mode_property`
- `TestSolverCacheProperties::test_solver_cache_dir_property`
- `TestSolverSetCacheDir::test_solver_set_cache_dir_delegates`
- `TestSolverSetCacheDir::test_solver_set_cache_dir_string`
- `TestSolverSetCacheDir::test_solver_set_cache_dir_path`

**Fix Required**: In `test_cache_config.py`, line 594, change `equations=equations` to `dxdt=equations`.

## Failures (1 test)

### tests/test_cubie_cache.py::test_batch_solver_kernel_cache_disabled

**Type**: KeyError  
**Message**: `"'caching_enabled' is not a valid compile setting for this object, and so was not updated."`

**Root Cause**: The test calls `solver.update_compile_settings(caching_enabled=False)` at line 322, but `caching_enabled` is a read-only property on `BatchSolverConfig` (derived from `cache_config.enabled`). It is not a settable attribute.

The `update_compile_settings` method checks attributes on `compile_settings` and nested attrs classes. The `caching_enabled` property is not an attribute, so it cannot be updated this way.

**Fix Required**: In `test_cubie_cache.py`, line 322, change:
```python
solver.update_compile_settings(caching_enabled=False)
```
to:
```python
solver.update_compile_settings(enabled=False)
```
This will find `enabled` in the nested `cache_config` attrs class and update it.

Also update line 325 assertion to match:
```python
assert solver.compile_settings.cache_config.enabled is False
```

## Recommendations

1. **Fix `simple_system` fixture**: Change `equations=equations` to `dxdt=equations` in `test_cache_config.py` at line 594.

2. **Fix `test_batch_solver_kernel_cache_disabled`**: Use `enabled=False` instead of `caching_enabled=False` when calling `update_compile_settings`, and update the assertion accordingly.

3. All other tests (49) are passing successfully.
