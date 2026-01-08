# Test Results Summary

## Overview
- **Tests Run**: 51
- **Passed**: 26
- **Failed**: 25
- **Errors**: 0
- **Skipped**: 0

## Test Command
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short tests/integrators/matrix_free_solvers/test_linear_solver.py tests/integrators/matrix_free_solvers/test_newton_krylov.py tests/integrators/algorithms/instrumented/test_instrumented.py tests/integrators/algorithms/test_ode_implicitstep.py
```

Note: `tests/integrators/algorithms/instrumented/test_instrumented.py` contained no matching tests.

## Critical Failures (Missing Attributes)

### test_implicit_step_accepts_tolerance_arrays
**Type**: AttributeError
**Message**: 'NewtonKrylov' object has no attribute 'krylov_atol'
**Location**: `src/cubie/integrators/algorithms/ode_implicitstep.py:366`

### test_implicit_step_exposes_tolerance_properties
**Type**: AttributeError
**Message**: 'NewtonKrylov' object has no attribute 'krylov_atol'
**Location**: `src/cubie/integrators/algorithms/ode_implicitstep.py:366`

## Numerical Accuracy Failures (Linear Solver)

These tests fail due to numerical precision issues (rtol=1e-07, atol=1e-07 tolerance):

### test_linear_solver_placeholder[minimal_residual]
**Type**: AssertionError
**Message**: Result `[0.090908974, 0.6363635, 1.4999996]` not close to expected `[0.09090909, 0.6363636, 1.5]`

### test_linear_solver_placeholder[steepest_descent]
**Type**: AssertionError  
**Message**: Result `[0.09090893, 0.6363635, 1.499999]` not close to expected `[0.09090909, 0.6363636, 1.5]`

### test_linear_solver_symbolic[0-steepest_descent-coupled_linear]
**Type**: AssertionError
**Message**: Result `[1.0060363, 1.0050277, 1.0070448]` not close to expected `[1.0060352, 1.0050272, 1.0070443]`

### test_linear_solver_symbolic[0-minimal_residual-coupled_linear]
**Type**: AssertionError
**Message**: Same as above - coupled_linear test case fails for both solver methods

### test_linear_solver_scaled_tolerance_converges[minimal_residual]
**Type**: AssertionError
**Message**: Result `[0.090908974, 0.6363635, 1.4999996]` not close to expected `[0.09090909, 0.6363636, 1.5]`

### test_linear_solver_scaled_tolerance_converges[steepest_descent]
**Type**: AssertionError
**Message**: Result `[0.09090893, 0.6363635, 1.499999]` not close to expected `[0.09090909, 0.6363636, 1.5]`

### test_linear_solver_scalar_tolerance_backward_compatible[minimal_residual]
**Type**: AssertionError
**Message**: Same numerical precision issue

### test_linear_solver_scalar_tolerance_backward_compatible[steepest_descent]
**Type**: AssertionError
**Message**: Same numerical precision issue

## Numerical Accuracy Failures (Newton-Krylov)

### test_newton_krylov_placeholder
**Type**: AssertionError
**Message**: Result `[0.009999951]` not close to expected `[0.010100961]`
**Difference**: ~1% relative difference

### test_newton_krylov_symbolic[*-linear] (3 failures)
**Type**: AssertionError
**Message**: Max absolute difference: 0.0013492424, Max relative difference: 4.5%
**Expected**: `[-0.01005, -0.020101, -0.030151]`
**Got**: `[-0.0105, -0.021, -0.0315]`

### test_newton_krylov_symbolic[*-coupled_linear] (3 failures)
**Type**: AssertionError
**Message**: Max absolute difference: 0.0004497273, Max relative difference: 4.5%

### test_newton_krylov_symbolic[*-coupled_nonlinear] (3 failures)
**Type**: AssertionError
**Message**: Max absolute difference: 0.0005010143, Max relative difference: 5%

### test_newton_krylov_symbolic[*-nonlinear] (3 failures)
**Type**: AssertionError
**Message**: Small differences (~1e-7) but exceeds tight tolerance

### test_newton_krylov_scalar_tolerance_backward_compatible
**Type**: AssertionError
**Message**: Result `[0.009999951]` not close to expected `[0.010100961]`

## Logic/Assertion Failures

### test_newton_krylov_failure
**Type**: AssertionError
**Message**: `assert 2 == (MAX_NEWTON_ITERATIONS_EXCEEDED | NEWTON_BACKTRACKING_NO_SUITABLE_STEP)`
**Note**: The test expects the return code to be a combination of both failure codes, but only `MAX_NEWTON_ITERATIONS_EXCEEDED (2)` is returned.

## Recommendations

1. **Missing Attribute - Critical Fix**: The `NewtonKrylov` class is missing the `krylov_atol` property. Add this property to expose the Krylov solver tolerance:
   - Location: `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`
   - The property should return the linear solver's `atol` value

2. **Numerical Tolerance Issues**: Many tests use very tight tolerances (rtol=1e-07, atol=1e-07) which may be too strict for float32 precision. Consider:
   - Using `tolerances.rel_loose` / `tolerances.abs_loose` (1e-05) instead of tight tolerances
   - Or investigating if the solver is not converging properly

3. **Newton-Krylov Accuracy**: The Newton-Krylov solver results show ~4-5% relative differences on linear problems, suggesting either:
   - Not enough iterations to converge
   - A bug in the convergence check or solver logic

4. **Test Logic Fix**: `test_newton_krylov_failure` expects bitwise OR of return codes but receives only one code. Either the test expectation or the solver behavior needs review.

## Passing Tests (26 tests)

- `test_neumann_preconditioner[linear-1-]`
- `test_neumann_preconditioner[linear-2-]`
- `test_linear_solver_symbolic[0-minimal_residual-linear]`
- `test_linear_solver_symbolic[0-steepest_descent-linear]`
- `test_linear_solver_symbolic[1-steepest_descent-coupled_linear]`
- `test_linear_solver_symbolic[1-minimal_residual-linear]`
- `test_linear_solver_symbolic[1-steepest_descent-linear]`
- `test_linear_solver_symbolic[1-minimal_residual-coupled_linear]`
- `test_linear_solver_symbolic[2-minimal_residual-coupled_linear]`
- `test_linear_solver_symbolic[2-steepest_descent-linear]`
- `test_linear_solver_symbolic[2-steepest_descent-coupled_linear]`
- `test_linear_solver_symbolic[2-minimal_residual-linear]`
- `test_linear_solver_config_array_tolerance_accepted`
- `test_linear_solver_config_wrong_length_raises`
- `test_linear_solver_config_scalar_tolerance_broadcast`
- `test_linear_solver_max_iters_exceeded`
- `test_newton_krylov_symbolic[0-stiff]`
- `test_newton_krylov_symbolic[1-stiff]`
- `test_newton_krylov_symbolic[2-stiff]`
- `test_newton_krylov_max_newton_iters_exceeded`
- `test_newton_krylov_linear_solver_failure_propagates`
- `test_newton_krylov_scaled_tolerance_converges`
- `test_newton_krylov_config_wrong_length_raises`
- `test_newton_krylov_config_array_tolerance_accepted`
- `test_newton_krylov_config_scalar_tolerance_broadcast`
- `test_implicit_step_linear_solver_newton_atol_returns_none`
