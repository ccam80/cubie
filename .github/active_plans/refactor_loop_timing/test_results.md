# Test Results Summary

## Test Command Executed
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not specific_algos" -v --tb=short tests/integrators/loops/ tests/batchsolving/ tests/outputhandling/
```

## Overview
- **Tests Run**: 483 (353 passed + 18 failed + 112 errors)
- **Passed**: 353
- **Failed**: 18
- **Errors**: 112
- **Skipped**: 0

## Critical Issue: Missing `dt_save` Attribute

The primary failure affecting 112+ tests is a missing attribute `dt_save` on `SingleIntegratorRun`. This is a refactor regression where the attribute name was changed but references were not updated.

**Error Location**: `src/cubie/batchsolving/BatchSolverKernel.py:950`
```
AttributeError: 'SingleIntegratorRun' object has no attribute 'dt_save'
```

**Root Cause**: The `BatchSolverKernel.output_length` property references `self.single_integrator.dt_save` but the `SingleIntegratorRun` class no longer exposes a `dt_save` attribute after the refactor.

## Errors (112 tests)

All 112 errors share the same root cause - AttributeError for missing `dt_save` attribute. These tests cannot initialize solvers properly.

### Affected Test Files:
- `tests/batchsolving/test_solver.py` - ~50+ tests
- `tests/batchsolving/test_SolverKernel.py` - ~6 tests
- `tests/batchsolving/arrays/test_batchinputarrays.py` - ~8 tests
- `tests/batchsolving/arrays/test_batchoutputarrays.py` - ~2 tests
- `tests/batchsolving/test_config_plumbing.py` - ~6 tests
- `tests/outputhandling/test_output_sizes.py` - ~2 tests
- `tests/integrators/loops/test_ode_loop.py` - ~1 test

### Sample Error Tests:
| Test | Type | Message |
|------|------|---------|
| test_solver_initialization | AttributeError | 'SingleIntegratorRun' object has no attribute 'dt_save' |
| test_solver_properties | AttributeError | 'SingleIntegratorRun' object has no attribute 'dt_save' |
| test_solve_basic | AttributeError | 'SingleIntegratorRun' object has no attribute 'dt_save' |
| test_kernel_builds | AttributeError | 'SingleIntegratorRun' object has no attribute 'dt_save' |
| test_dtype | AttributeError | 'SingleIntegratorRun' object has no attribute 'dt_save' |

## Failures (18 tests)

### tests/batchsolving/test_solver.py - 5 Failures

| Test | Type | Message |
|------|------|---------|
| test_solve_ivp_function | AttributeError | 'SingleIntegratorRun' object has no attribute 'dt_save' |
| test_solver_with_different_algorithms | AttributeError | 'SingleIntegratorRun' object has no attribute 'dt_save' |
| test_solver_output_types | AttributeError | 'SingleIntegratorRun' object has no attribute 'dt_save' |
| test_solve_ivp_save_every_param | AttributeError | 'SingleIntegratorRun' object has no attribute 'dt_save' |
| test_solve_ivp_no_dt_save | AttributeError | 'SingleIntegratorRun' object has no attribute 'dt_save' |

### tests/integrators/loops/test_ode_loop.py - 13 Failures

These tests are passing the kernel execution but failing on numerical assertions - the device results don't match reference values. This may be a secondary issue related to timing parameter changes.

| Test | Type | Message |
|------|------|---------|
| test_loop[backwards_euler_pc] | AssertionError | state summaries mismatch. Max diff: 0.1276 |
| test_loop[backwards_euler] | AssertionError | state summaries mismatch. Max diff: 0.1276 |
| test_loop[crank_nicolson] | AssertionError | state summaries mismatch. Max diff: 0.1276 |
| test_loop[rosenbrock] | AssertionError | state summaries mismatch. Max diff: 0.1276 |
| test_loop[erk] | AssertionError | state summaries mismatch. Max diff: 0.1276 |
| test_loop[dirk] | AssertionError | state summaries mismatch. Max diff: 0.1276 |
| test_loop[firk] | AssertionError | state summaries mismatch. Max diff: 0.1276 |
| test_loop[euler] | AssertionError | state summaries mismatch. Max diff: 0.1276 |
| test_all_summary_metrics_numerical_check[1st generation metrics] | AssertionError | state summaries mismatch |
| test_all_summary_metrics_numerical_check[combined metrics] | AssertionError | state summaries mismatch. Max diff: 15.4 |
| test_all_summary_metrics_numerical_check[no combos] | AssertionError | state summaries mismatch. Max diff: 15.4 |

## Recommendations

### Priority 1: Fix `dt_save` Attribute Reference (Critical)
The `BatchSolverKernel.output_length` property at line 950 references `self.single_integrator.dt_save`. This attribute no longer exists after the loop timing refactor. Possible fixes:
1. Add `dt_save` property to `SingleIntegratorRun` that exposes the save interval
2. Update `BatchSolverKernel.output_length` to use the new timing parameter structure

### Priority 2: Investigate Numerical Mismatches
After fixing the `dt_save` issue, investigate why the ODE loop tests show ~12.7% relative difference in state summaries. This could be:
1. Changed timing calculation affecting integration steps
2. Reference values need to be regenerated with new timing parameters
3. Bug in the refactored timing logic

### Priority 3: Verify `dt_save` Usage Across Codebase
Search for all usages of `dt_save` to ensure consistency:
```bash
grep -r "dt_save" src/cubie/
```
