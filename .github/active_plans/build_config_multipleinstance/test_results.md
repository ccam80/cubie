# Test Results Summary

## Overview
- **Tests Run**: 117
- **Passed**: 93
- **Failed**: 24
- **Errors**: 0
- **Skipped**: 0

## Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -vs tests/test_CUDAFactory.py tests/integrators/test_norms.py tests/integrators/matrix_free_solvers/test_base_solver.py tests/integrators/matrix_free_solvers/test_linear_solver.py tests/integrators/matrix_free_solvers/test_newton_krylov.py -m "not nocudasim and not cupy"
```

## Failures

### tests/test_CUDAFactory.py::test_multiple_instance_factory_prefix_mapping
**Type**: AssertionError
**Message**: assert 100 == 50 (max_iters on LinearSolverConfig was 100 instead of expected 50)

### tests/test_CUDAFactory.py::test_multiple_instance_factory_mixed_keys
**Type**: KeyError
**Message**: "'test_value' is not a valid compile setting for this object, and so was not updated."

### tests/test_CUDAFactory.py::test_build_config_with_instance_label
**Type**: AssertionError
**Message**: assert 1e-06 == 1e-10 (atol on TestConfig was 1e-06 instead of expected 1e-10)

### tests/test_CUDAFactory.py::test_build_config_instance_label_prefixed_takes_precedence
**Type**: AssertionError
**Message**: assert 1e-08 == 1e-12 (atol on TestConfig was 1e-08 instead of expected 1e-12)

### tests/integrators/matrix_free_solvers/test_newton_krylov.py (Multiple Tests)
**Type**: TypeError
**Message**: "'NoneType' object is not callable" - linear_solver_fn is None when called in newton_krylov.py:396

Affected tests:
- test_newton_krylov_placeholder
- test_newton_krylov_symbolic (all parameter combinations: 12 tests)
- test_newton_krylov_failure
- test_newton_krylov_scaled_tolerance_converges
- test_newton_krylov_linear_solver_failure_propagates
- test_newton_krylov_newton_max_iters_exceeded
- test_newton_krylov_scalar_tolerance_backward_compatible

### tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_no_manual_cache_invalidation
**Type**: AssertionError
**Message**: assert None is not None - norm_device_function is None on NewtonKrylovConfig

### tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_init_with_newton_prefixed_kwargs
**Type**: AssertionError
**Message**: assert 100 == 50 - newton_max_iters was 100 instead of expected 50

## Analysis

### Category 1: build_config instance_label not being applied (4 failures)
The `build_config` method is not properly applying instance_label-prefixed settings to factory objects. Tests show that:
- `test_build_config_with_instance_label`: Instance-labeled atol value (1e-10) not applied, got default 1e-06
- `test_build_config_instance_label_prefixed_takes_precedence`: Prefixed value (1e-12) not taking precedence over non-prefixed
- `test_multiple_instance_factory_prefix_mapping`: linear_max_iters (50) not applied to LinearSolver
- `test_multiple_instance_factory_mixed_keys`: KeyError indicates settings dictionary handling issue

### Category 2: Newton-Krylov linear_solver_function is None (17 failures)
The `linear_solver_fn` captured in the Newton-Krylov kernel is `None`. This suggests that:
- The LinearSolver is not being properly built/wired into NewtonKrylov
- The `linear_solver_function` setting is not being properly propagated through build_config

### Category 3: norm_device_function is None (1 failure)
The `test_newton_krylov_no_manual_cache_invalidation` test shows that `norm_device_function` is None on NewtonKrylovConfig, indicating norm is not being properly built.

## Recommendations

1. **Fix build_config instance_label handling**: The `build_config` method needs to properly extract instance_label-prefixed settings and apply them to the factory. Currently, settings like `norm_atol` or `linear_max_iters` are not being matched with their instance label prefix.

2. **Fix MultipleInstanceCUDAFactoryConfig setting propagation**: When building nested factories (like NewtonKrylov with LinearSolver and ScaledNorm), the settings dictionary needs to properly route prefixed settings to child factories.

3. **Verify norm and linear_solver_function wiring**: The NewtonKrylov factory needs to ensure that its norm and linear_solver child factories are built and their device functions are captured.

## Passing Tests (93)

All tests in:
- tests/integrators/test_norms.py (all passed)
- tests/integrators/matrix_free_solvers/test_base_solver.py (all passed)
- tests/integrators/matrix_free_solvers/test_linear_solver.py (all passed)
- Most tests in tests/test_CUDAFactory.py (basic factory tests passed)
