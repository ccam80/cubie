# Test Results Summary

## Overview
- **Tests Run**: 99
- **Passed**: 99
- **Failed**: 0
- **Errors**: 0
- **Skipped**: 0

## Test Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short tests/integrators/matrix_free_solvers/test_base_solver.py tests/integrators/matrix_free_solvers/test_linear_solver.py tests/integrators/matrix_free_solvers/test_newton_krylov.py tests/integrators/test_norms.py tests/integrators/step_control/
```

## Failures

None - all tests passed!

## Errors

None - all tests passed!

## Tests by File

### tests/integrators/matrix_free_solvers/test_base_solver.py
- ✅ test_matrix_free_solver_config_precision_property
- ✅ test_matrix_free_solver_extract_prefixed_tolerance
- ✅ test_matrix_free_solver_config_validation
- ✅ test_matrix_free_solver_norm_update_propagates_to_config
- ✅ test_matrix_free_solver_config_max_iters_default
- ✅ test_matrix_free_solver_config_max_iters_validation
- ✅ test_matrix_free_solver_config_norm_device_function_field
- ✅ test_matrix_free_solver_creates_norm

### tests/integrators/matrix_free_solvers/test_linear_solver.py
- ✅ All symbolic tests (steepest_descent, minimal_residual variants)
- ✅ test_neumann_preconditioner (linear-1, linear-2)
- ✅ test_linear_solver_placeholder (steepest_descent, minimal_residual)
- ✅ test_linear_solver_uses_scaled_norm
- ✅ test_linear_solver_tolerance_update_propagates
- ✅ test_linear_solver_config_no_tolerance_fields
- ✅ test_linear_solver_config_settings_dict_excludes_tolerance_arrays
- ✅ test_linear_solver_inherits_from_matrix_free_solver
- ✅ test_linear_solver_update_preserves_original_dict
- ✅ test_linear_solver_no_manual_cache_invalidation
- ✅ test_linear_solver_settings_dict_includes_tolerance_arrays
- ✅ test_linear_solver_max_iters_exceeded
- ✅ test_linear_solver_config_scalar_tolerance_broadcast
- ✅ test_linear_solver_config_array_tolerance_accepted
- ✅ test_linear_solver_config_wrong_length_raises
- ✅ test_linear_solver_scaled_tolerance_converges (steepest_descent, minimal_residual)
- ✅ test_linear_solver_scalar_tolerance_backward_compatible (steepest_descent, minimal_residual)

### tests/integrators/matrix_free_solvers/test_newton_krylov.py
- ✅ test_newton_krylov_placeholder
- ✅ All symbolic tests (linear, nonlinear, stiff, coupled variants)
- ✅ test_newton_krylov_failure
- ✅ test_newton_krylov_max_newton_iters_exceeded
- ✅ test_newton_krylov_linear_solver_failure_propagates
- ✅ test_newton_krylov_scaled_tolerance_converges
- ✅ test_newton_krylov_scalar_tolerance_backward_compatible
- ✅ test_newton_krylov_uses_scaled_norm
- ✅ test_newton_krylov_tolerance_update_propagates
- ✅ test_newton_krylov_config_no_tolerance_fields
- ✅ test_newton_krylov_config_settings_dict_excludes_tolerance_arrays
- ✅ test_newton_krylov_inherits_from_matrix_free_solver
- ✅ test_newton_krylov_update_preserves_original_dict
- ✅ test_newton_krylov_no_manual_cache_invalidation
- ✅ test_newton_krylov_settings_dict_includes_tolerance_arrays
- ✅ test_newton_krylov_config_scalar_tolerance_broadcast
- ✅ test_newton_krylov_config_array_tolerance_accepted
- ✅ test_newton_krylov_config_wrong_length_raises

### tests/integrators/test_norms.py
- ✅ test_scaled_norm_config_default_tolerance
- ✅ test_scaled_norm_config_custom_tolerance
- ✅ test_scaled_norm_factory_builds_device_function
- ✅ test_scaled_norm_converged_when_under_tolerance
- ✅ test_scaled_norm_exceeds_when_over_tolerance
- ✅ test_scaled_norm_update_invalidates_cache

### tests/integrators/step_control/
- ✅ test_pi_controller_uses_tableau_order
- ✅ TestControllerEquivalence tests (i, pi, pid, gustafsson)
- ✅ TestControllers::test_controller_builds (i, pi, pid, gustafsson)
- ✅ TestControllers::test_dt_clamps (min_limit, max_limit for all controllers)

## Warnings

56 warnings total:
- 55 NumbaDeprecationWarning about `nopython=False` keyword argument
- 1 RuntimeWarning about overflow in gustafsson_controller.py (pre-existing, not related to changes)

## Recommendations

All tests passed! The scaled tolerance refactoring improvements are working correctly:
- Base solver tests confirm infrastructure is in place
- LinearSolver tests confirm tolerance propagation and scaling
- NewtonKrylov tests confirm end-to-end solver behavior
- Norms tests confirm ScaledNorm functionality
- Step control tests confirm no regressions in that subsystem
