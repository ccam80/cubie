# Test Results Summary

## Test Run Information
- **Date**: 2026-01-05
- **Branch**: copilot/complete-loop-timing-refactor
- **Environment**: NUMBA_ENABLE_CUDASIM=1 (CUDA simulation mode)
- **Markers Excluded**: nocudasim, cupy

## Overview

| Test File | Passed | Failed | Errors | Skipped | Total |
|-----------|--------|--------|--------|---------|-------|
| test_ode_loop.py | 7 | 34 | 4 | 0 | 45 |
| test_dt_update_summaries_validation.py | 20 | 2 | 0 | 0 | 22 |
| test_config_plumbing.py | 6 | 0 | 0 | 0 | 6 |
| test_solveresult.py | 31 | 0 | 2 | 0 | 33 |
| test_cpu_utils.py | 2 | 0 | 0 | 0 | 2 |
| **TOTAL** | **66** | **36** | **6** | **0** | **108** |

## Failures

### test_ode_loop.py

#### test_loop[euler] (and 28 other algorithm variants)
**Type**: AssertionError  
**Message**: state summaries mismatch - shapes (2, 2) vs (5, 2) mismatch. The device output has fewer summary rows than the reference.

This failure affects all algorithm variants:
- euler, backwards_euler, crank_nicolson, backwards_euler_pc
- dirk, dirk-implicit-midpoint, dirk-lobatto-iiic-3, dirk-l-stable-3, dirk-l-stable-4, dirk-sdirk-2-2
- erk, erk-cash-karp-54, erk-dormand-prince-54, erk-bogacki-shampine-32, erk-ralston-33, erk-classical-rk4, erk-tsit5, erk-dop853, erk-vern7, erk-heun-21
- firk, firk-gauss-legendre-2, firk-radau
- rosenbrock, rosenbrock-ros3p, rosenbrock-rodas3p, rosenbrock-ode23s

**Root Cause Analysis**: The test expects 5 summary rows but receives only 2. This suggests the summary collection interval (`summarise_every`) is not being applied correctly, or the loop is collecting summaries at the wrong intervals.

#### test_save_last_with_save_every[solver_settings_override0]
**Type**: FAILED (captured in output)

#### test_save_at_settling_time_boundary[solver_settings_override0]
**Type**: FAILED (captured in output)

#### test_all_summary_metrics_numerical_check[combined metrics]
**Type**: FAILED
**Message**: Shape mismatch in summary output

#### test_all_summary_metrics_numerical_check[1st generation metrics]
**Type**: FAILED
**Message**: Shape mismatch in summary output

#### test_all_summary_metrics_numerical_check[no combos]
**Type**: FAILED
**Message**: Shape mismatch in summary output

### test_dt_update_summaries_validation.py

#### test_tolerant_validation_auto_adjusts
**Type**: ValueError  
**Message**: summarise_every (1.005) must be an integer multiple of sample_summaries_every (0.1). The ratio 10.0500 is not close to any integer.

**Root Cause Analysis**: The tolerant validation logic is failing to auto-adjust the value when there's a small floating-point difference. The tolerance check is too strict.

#### test_tolerant_validation_warns_on_adjustment
**Type**: ValueError  
**Message**: summarise_every (1.008) must be an integer multiple of sample_summaries_every (0.1). The ratio 10.0800 is not close to any integer.

**Root Cause Analysis**: Same as above - the tolerant validation is not allowing adjustments for small deviations.

## Errors

### test_ode_loop.py

#### test_save_last_flag_from_config[solver_settings_override0]
**Type**: ValueError  
**Message**: cannot convert float NaN to integer

**Root Cause Analysis**: A NaN value is being passed where an integer is expected. This likely occurs in output size calculations or array indexing.

#### test_summarise_last_flag_from_config[solver_settings_override0]
**Type**: ValueError  
**Message**: cannot convert float NaN to integer

**Root Cause Analysis**: Same as above.

#### test_summarise_last_collects_final_summary[solver_settings_override0]
**Type**: ValueError (ERROR during test setup/execution)

#### test_summarise_last_with_summarise_every_combined[solver_settings_override0]
**Type**: ERROR

#### test_loop[erk-fehlberg-45]
**Type**: ERROR (likely same NaN conversion issue)

### test_solveresult.py

#### test_from_solver_numpy_instantiation
**Type**: AttributeError  
**Message**: tid=[0, 7, 0] ctaid=[0, 0, 0]: module 'numba.cuda' has no attribute 'local'

**Root Cause Analysis**: The `cuda.local` attribute is not available in CUDA simulation mode. This is a known limitation - code using `cuda.local.array` will fail in simulation mode.

#### test_from_solver_pandas_instantiation
**Type**: AttributeError  
**Message**: tid=[0, 7, 0] ctaid=[0, 0, 0]: module 'numba.cuda' has no attribute 'local'

**Root Cause Analysis**: Same as above.

## Passing Tests

### test_config_plumbing.py (6/6 passed)
All configuration plumbing tests passed successfully.

### test_cpu_utils.py (2/2 passed)
All CPU utility tests passed successfully.

### test_ode_loop.py (7/45 passed)
- test_getters
- test_initial_observable_seed_matches_reference
- test_summarise_last_with_summarise_every[solver_settings_override0]
- test_adaptive_controller_with_float32[]
- test_float32_small_timestep_accumulation[]
- test_large_t0_with_small_steps[float64]
- test_large_t0_with_small_steps[float32]

### test_solveresult.py (31/33 passed)
Most SolveResult tests pass; only the instantiation tests using `cuda.local` fail in simulation mode.

### test_dt_update_summaries_validation.py (20/22 passed)
Most validation tests pass; only the tolerant adjustment tests fail.

## Recommendations

### High Priority

1. **Fix summary collection interval logic**: The test_loop failures indicate that summaries are being collected at incorrect intervals. Review the loop timing logic in `ode_loop.py` to ensure `summarise_every` is properly honored.

2. **Fix NaN handling in output size calculations**: Several tests error with "cannot convert float NaN to integer". Check for proper initialization of timing values and guard against NaN propagation.

3. **Fix tolerant validation tolerance**: The `test_tolerant_validation_auto_adjusts` and `test_tolerant_validation_warns_on_adjustment` tests expect the validation to be more lenient. Increase the tolerance or fix the adjustment logic.

### Medium Priority

4. **Mark cuda.local tests as nocudasim**: The `test_from_solver_numpy_instantiation` and `test_from_solver_pandas_instantiation` tests use `cuda.local.array` which is not available in simulation mode. Consider marking these with `@pytest.mark.nocudasim`.

### Investigation Needed

5. **Debug the captured stdout**: Many tests print `[[0 0 0 0]...]` matrices which may indicate the loop timing counters are not incrementing correctly. This could be related to the summary collection issues.

## Summary

- **Critical Issues**: 36 test failures + 6 errors
- **Main Problem Areas**:
  1. Loop timing/summary collection (causes most failures in test_ode_loop.py)
  2. NaN handling in calculations (causes errors in save_last and summarise_last tests)
  3. Tolerant validation tolerance too strict (causes 2 failures in validation tests)
  4. CUDA simulation mode incompatibility with cuda.local (causes 2 errors in solveresult tests)
