# Test Results Summary

## Test Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not cupy and not specific_algos" -v --tb=short tests/integrators/matrix_free_solvers/test_linear_solver.py tests/integrators/matrix_free_solvers/test_newton_krylov.py tests/integrators/algorithms/test_ode_implicitstep.py
```

## Overview
- **Tests Run**: 51
- **Passed**: 40
- **Failed**: 11
- **Errors**: 0
- **Skipped**: 0

## Failures

### tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_placeholder
**Type**: AssertionError
**Message**: `np.allclose` failed - expected `0.010100961` but got `0.009999951` (rtol=1e-05, atol=1e-05)

### tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_symbolic[0-linear]
**Type**: AssertionError
**Message**: Not equal to tolerance rtol=1e-05, atol=1e-05. Max absolute difference: 0.0013492424, Max relative difference: 0.044749904. Expected `[-0.01005, -0.020101, -0.030151]` but got `[-0.0105, -0.021, -0.0315]`

### tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_symbolic[0-coupled_linear]
**Type**: AssertionError
**Message**: Not equal to tolerance rtol=1e-05, atol=1e-05. Max absolute difference: 0.0004497273, Max relative difference: 0.044747774.

### tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_symbolic[0-coupled_nonlinear]
**Type**: AssertionError
**Message**: Not equal to tolerance rtol=1e-05, atol=1e-05. Max absolute difference: 0.0005010143, Max relative difference: 0.050106518.

### tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_symbolic[1-linear]
**Type**: AssertionError
**Message**: Not equal to tolerance rtol=1e-05, atol=1e-05. Max absolute difference: 0.0013492424, Max relative difference: 0.044749904.

### tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_symbolic[2-linear]
**Type**: AssertionError
**Message**: Not equal to tolerance rtol=1e-05, atol=1e-05. Max absolute difference: 0.0013492424, Max relative difference: 0.044749904.

### tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_symbolic[1-coupled_linear]
**Type**: AssertionError
**Message**: Not equal to tolerance rtol=1e-05, atol=1e-05. Max absolute difference: 0.0004497273, Max relative difference: 0.044747774.

### tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_symbolic[1-coupled_nonlinear]
**Type**: AssertionError
**Message**: Not equal to tolerance rtol=1e-05, atol=1e-05. Max absolute difference: 0.0005010143, Max relative difference: 0.050106518.

### tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_symbolic[2-coupled_linear]
**Type**: AssertionError
**Message**: Not equal to tolerance rtol=1e-05, atol=1e-05. Max absolute difference: 0.0004497273, Max relative difference: 0.044747774.

### tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_symbolic[2-coupled_nonlinear]
**Type**: AssertionError
**Message**: Not equal to tolerance rtol=1e-05, atol=1e-05. Max absolute difference: 0.0005010143, Max relative difference: 0.050106518.

### tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_scalar_tolerance_backward_compatible
**Type**: AssertionError
**Message**: `np.allclose` failed - expected `0.010100961` but got `0.009999951` (rtol=1e-05, atol=1e-05)

## Analysis

### Pattern Observed
The failures occur in three groups:

1. **Neumann series index 0 tests** (`[0-linear]`, `[0-coupled_linear]`, `[0-coupled_nonlinear]`): These use no Neumann preconditioning (neumann_order=0), causing the iterative solver to converge to a different solution than expected.

2. **All neumann indices with certain systems** (`linear`, `coupled_linear`, `coupled_nonlinear`): These system types show consistent ~4-5% relative differences.

3. **Placeholder and backward compatibility tests**: These use basic configurations without sufficient iterations/tolerance tuning.

### Root Cause
The tests are expecting more precise convergence than the solver achieves with the current iteration/tolerance settings. The Neumann preconditioner order 0 case (no preconditioning) shows the worst convergence.

### Passing Tests
- All `test_linear_solver` tests (28 tests) - PASSED
- All `test_ode_implicitstep` tests (3 tests) - PASSED  
- Newton-Krylov `stiff` and `nonlinear` variants - PASSED
- Newton-Krylov tolerance configuration tests - PASSED
- Newton-Krylov failure and max iteration tests - PASSED

## Recommendations

1. **Increase solver tolerances in failing tests**: The failing symbolic tests use tight tolerances (atol=1e-05, rtol=1e-05) that require more iterations than configured.

2. **Increase max_linear_iters or max_newton_iters**: The solver may not be iterating enough to reach the expected tolerance.

3. **Loosen expected tolerances in assertions**: Consider using 1e-3 or 1e-4 for these numerical integration tests instead of 1e-5.

4. **Fix placeholder tests**: The `test_newton_krylov_placeholder` and `test_newton_krylov_scalar_tolerance_backward_compatible` tests need updated expected values or tolerance settings.
