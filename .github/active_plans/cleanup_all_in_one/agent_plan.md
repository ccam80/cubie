# Agent Plan: Cleanup all_in_one.py Debug Script

## Overview

This plan guides the detailed_implementer and taskmaster agents through cleaning up the `tests/all_in_one.py` debug script to align factory-scope variables with production implementations.

## Problem Statement

The all_in_one.py debug script has accumulated clutter from incremental modifications when adding FIRK and Rosenbrock step functions. Factory-scope variables may:
- Include unnecessary or unused variables
- Have incorrect typing (e.g., `int32` where Python `int` is required)
- Differ from production factory implementations

## Architectural Context

### Production Factory Patterns

Production factories follow a consistent pattern:

1. **Array size variables**: Use native Python `int` for `cuda.local.array` sizes
2. **Loop bound variables**: Convert to `int32(n)` only when needed in device code
3. **Precision typing**: Use `numba_precision = numba_from_dtype(precision)` for typed constants
4. **Factory closure**: Only capture variables actually used by the device function

### Key Typing Rules

```python
# CORRECT - Python int for array sizes
n_arraysize = n  # native int
stage_rhs = cuda.local.array(n_arraysize, precision)

# CORRECT - int32 for loop bounds inside device code
n = int32(n)
for i in range(n):
    ...

# INCORRECT - int32 for array sizes causes issues
n_arraysize = int32(n)  # WRONG - Numba may reject this
```

## Components to Clean

### 1. FIRK Step Factory (`firk_step_inline_factory`)

**Production reference**: `generic_firk.py::FIRKStep.build_step` (lines 573-872)

**Factory-scope variables in production**:
- `config`, `precision`, `tableau`
- `nonlinear_solver`
- `n = int32(n)`, `n_drivers = int32(n_drivers)`, `stage_count = int32(self.stage_count)`
- `all_stages_n = int32(config.all_stages_n)`
- `has_driver_function`, `has_error`
- `stage_rhs_coeffs`, `solution_weights`, `typed_zero`, `error_weights`
- `stage_time_fractions`
- `accumulates_output`, `accumulates_error`, `b_row`, `b_hat_row`, `ends_at_one`
- Buffer settings variables for selective shared/local allocation

**Expected cleanup**:
- Remove any variables not in the production list
- Ensure array size variables are Python `int`
- Match production variable naming

### 2. Rosenbrock Step Factory (`rosenbrock_step_inline_factory`)

**Production reference**: `generic_rosenbrock_w.py::GenericRosenbrockWStep.build_step` (lines 534-946)

**Factory-scope variables in production**:
- `config`, `precision`, `tableau`
- `linear_solver`, `prepare_jacobian`, `time_derivative_rhs`
- `n = int32(n)`, `stage_count = int32(self.stage_count)`
- `stages_except_first = stage_count - int32(1)`
- `has_driver_function`, `has_error`, `typed_zero`
- `a_coeffs`, `C_coeffs`, `gamma_stages`, `gamma`
- `solution_weights`, `error_weights`, `stage_time_fractions`
- `accumulates_output`, `accumulates_error`, `b_row`, `b_hat_row`
- Buffer settings variables

**Expected cleanup**:
- Match production variable set
- Ensure `cached_auxiliary_count` related sizes are Python `int` for array allocation

### 3. ERK Step Factory (`erk_step_inline_factory`)

**Production reference**: `generic_erk.py::ERKStep.build_step` (lines 454-795)

**Factory-scope variables in production**:
- `config`, `precision`, `tableau`
- `typed_zero`, `n_arraysize = n`, `n = int32(n)`
- `stage_count = int32(tableau.stage_count)`
- `stages_except_first = stage_count - int32(1)`
- `accumulator_length = (tableau.stage_count - 1) * n_arraysize` (Python int!)
- `has_driver_function`, `first_same_as_last`, `multistage`, `has_error`
- `stage_rhs_coeffs`, `solution_weights`, `stage_nodes`
- `error_weights`, `accumulates_output`, `accumulates_error`
- `b_row`, `b_hat_row`
- Buffer settings variables

**Expected cleanup**:
- Ensure `accumulator_length` uses Python `int` multiplication, not `int32`
- Match production variable set

### 4. DIRK Step Factory (`dirk_step_inline_factory`)

**Production reference**: `generic_dirk.py::DIRKStep.build_step` (lines 592-1040)

**Factory-scope variables in production**:
- `config`, `precision`, `tableau`, `nonlinear_solver`
- `n = int32(n)`, `stage_count = int32(tableau.stage_count)`
- `stages_except_first = stage_count - int32(1)`
- `has_driver_function`, `has_error`, `multistage`
- `first_same_as_last`, `can_reuse_accepted_start`
- `explicit_a_coeffs`, `solution_weights`, `typed_zero`
- `error_weights`, `stage_time_fractions`, `diagonal_coeffs`
- `accumulates_output`, `accumulates_error`, `b_row`, `b_hat_row`
- `stage_implicit`, `accumulator_length = int32(max(stage_count - 1, 0) * n)`
- Buffer settings variables

**Expected cleanup**:
- Match production pattern for `accumulator_length`
- Ensure solver scratch sizing is correct

### 5. Linear Solver Factory (`linear_solver_inline_factory`)

**Production reference**: `linear_solver.py::linear_solver_factory` (lines 154-404)

**Factory-scope variables in production**:
- `sd_flag`, `mr_flag` (plain Python int 0/1)
- `preconditioned` (plain Python int 0/1)
- `n_val = int32(n)`
- `max_iters = int32(max_iters)`
- `precision = from_dtype(precision)`
- `typed_zero`, `tol_squared`
- Buffer settings variables for selective allocation

**Expected cleanup**:
- Ensure `n_arraysize` for local arrays is Python `int`
- Match production variable naming

### 6. Linear Solver Cached Factory (`linear_solver_cached_inline_factory`)

**Production reference**: `linear_solver.py::linear_solver_cached_factory` (lines 407-596)

**Similar pattern to regular linear solver**

### 7. Newton-Krylov Factory (`newton_krylov_inline_factory`)

**Production reference**: `newton_krylov.py::newton_krylov_solver_factory` (lines 17-297)

**Factory-scope variables in production**:
- `precision = from_dtype(precision_dtype)`
- `tol_squared = precision(tolerance * tolerance)`
- `typed_zero`, `typed_one`, `typed_damping`
- `n = int32(n)`
- `max_iters = int32(max_iters)`
- `max_backtracks = int32(max_backtracks)`

**Expected cleanup**:
- Remove any `one_int64` or similar unnecessary variables
- Match production pattern exactly

## Integration Points

### Interaction with Module-Level Config

The all_in_one.py script has module-level configuration that feeds into factories:
- `n_states`, `n_parameters`, `n_drivers`, etc.
- These should remain as native Python `int` 
- Conversion to `int32` should happen inside factory scope when needed

### Buffer Size Calculations

Module-level buffer size calculations should use Python `int`:
```python
# These are correct as Python int
accumulator_size = int32((stage_count - 1) * n_states)  # WRONG - int32 at module level
accumulator_size = (stage_count - 1) * n_states  # CORRECT - Python int at module level
```

## Expected Changes Summary

1. **Remove unnecessary factory-scope variables** not present in production
2. **Convert array size variables to Python int** where currently using `int32`
3. **Match production naming conventions** for factory-scope variables
4. **Ensure type consistency** between all_in_one.py and production factories

## Validation

After cleanup:
1. The script should compile without Numba type errors
2. The script should run with each algorithm type (erk, dirk, firk, rosenbrock)
3. Factory-scope variable sets should match production

## Edge Cases

1. **`int32` in slices**: `int32` is acceptable inside device code for slices and loop bounds
2. **Zero-size arrays**: Use `max(size, 1)` pattern where size could be zero
3. **Cached auxiliary count**: This may be `int32(1)` minimum in Rosenbrock
