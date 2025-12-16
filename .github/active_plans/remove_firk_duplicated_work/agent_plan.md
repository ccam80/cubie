# Agent Plan: Remove FIRK Duplicated Work

## Overview

This plan eliminates redundant computations in FIRK step device functions by using the mathematical property that stage increments at Newton convergence contain `u_i = h * k_i`.

## Component Descriptions

### 1. Stage Increment Buffer (`stage_increment`)

**Current Behavior:**
- After Newton solve, contains flattened stage increments: `stage_increment[stage_idx * n + component_idx]`
- Values are `u_i = h * k_i` where `k_i = f(Y_i)` and `Y_i = y_n + Σ a_ij * u_j`
- This relationship holds because the residual function solves `M*u - h*f(Y_i) = 0`

**Post-Optimization:**
- No change to buffer contents
- Buffer is used directly for accumulation without intermediate reconstructions

### 2. Post-Solve Stage Loop (lines 764-821)

**Current Behavior:**
- Iterates through all stages after Newton solve
- For each stage:
  1. Computes `stage_time`
  2. Extracts driver values from `stage_driver_stack`
  3. Reconstructs `stage_state[idx] = state[idx] + Σ a_ij * stage_increment[j*n + idx]`
  4. Checks for stiffly-accurate shortcuts (`b_matches_a_row`, `b_hat_matches_a_row`)
  5. If `do_more_work` (needs accumulation), calls `observables_function` and `dxdt_fn`

**Expected Post-Optimization Behavior:**
- The loop is split into two concerns:
  1. **Stiffly-accurate shortcut path**: Only reconstruct stage states when `b_matches_a_row` or `b_hat_matches_a_row` matches. No dxdt_fn needed.
  2. **Accumulation path**: Eliminated from this loop entirely - accumulation happens directly from `stage_increment`

### 3. Output Accumulation (lines 824-837)

**Current Behavior:**
```python
if accumulates_output:
    for idx in range(n):
        for stage_idx in range(stage_count):
            rhs_value = stage_rhs_flat[stage_idx * n + idx]
            solution_acc += solution_weights[stage_idx] * rhs_value
        proposed_state[idx] = state[idx] + solution_acc * dt_scalar
```

**Expected Post-Optimization Behavior:**
```python
if accumulates_output:
    for idx in range(n):
        for stage_idx in range(stage_count):
            inc_value = stage_increment[stage_idx * n + idx]
            solution_acc += solution_weights[stage_idx] * inc_value
        proposed_state[idx] = state[idx] + solution_acc  # No dt_scalar
```

**Key Change:** Uses `stage_increment` instead of `stage_rhs_flat`, removes `dt_scalar` multiplication since stage_increment already contains `h * k_i`.

### 4. Error Accumulation (lines 839-846)

**Current Behavior:**
```python
if has_error and accumulates_error:
    for idx in range(n):
        for stage_idx in range(stage_count):
            rhs_value = stage_rhs_flat[stage_idx * n + idx]
            error_acc += error_weights[stage_idx] * rhs_value
        error[idx] = dt_scalar * error_acc
```

**Expected Post-Optimization Behavior:**
```python
if has_error and accumulates_error:
    for idx in range(n):
        for stage_idx in range(stage_count):
            inc_value = stage_increment[stage_idx * n + idx]
            error_acc += error_weights[stage_idx] * inc_value
        error[idx] = error_acc  # No dt_scalar multiplication
```

**Key Change:** Uses `stage_increment` directly, removes `dt_scalar` multiplication.

### 5. Stiffly-Accurate Shortcut Path

**Current Behavior:**
- When `b_matches_a_row` is not None, `proposed_state` is captured directly from a reconstructed stage state
- When `b_hat_matches_a_row` is not None, `error` (as embedded solution) is captured from a stage state

**Expected Post-Optimization Behavior:**
- This path is PRESERVED - stage state reconstruction is still needed for these shortcuts
- Only the stage that matches needs reconstruction, not all stages
- The condition `do_more_work` (evaluating `dxdt_fn`) is no longer needed for these shortcuts

## Architectural Changes

### Stage Loop Restructuring

The post-solve loop currently handles three concerns:
1. Stiffly-accurate shortcuts (capture proposed_state or error from specific stage)
2. Accumulation preparation (compute stage_rhs_flat via dxdt_fn)
3. Driver restoration for observables

After optimization:
1. **Stiffly-accurate shortcuts**: Only reconstruct the specific stage(s) needed
2. **Accumulation**: Direct from stage_increment, no loop needed for this
3. **Driver restoration**: May still be needed for final observables if `ends_at_one` is False

### Condition Logic Changes

**Current:**
```python
do_more_work = ((has_error and accumulates_error) or accumulates_output)
if do_more_work:
    observables_function(...)
    dxdt_fn(...)
```

**Optimized:**
- The `do_more_work` logic is removed entirely from the stage loop
- Accumulation happens directly from `stage_increment` after the loop

### Buffer Elimination

**Current:**
- `stage_rhs_flat` is a slice of `solver_scratch` used to store `f(Y_i)` values

**Optimized:**
- `stage_rhs_flat` is NO LONGER NEEDED for accumulation when `accumulates_output` or `accumulates_error` is True
- The slice definition (`stage_rhs_flat = solver_scratch[:all_stages_n]`) can remain but is unused in the accumulation path

## Integration Points

### FIRKStep Class
- `build()` method compiles the step device function
- Closure captures `accumulates_output`, `accumulates_error`, `b_row`, `b_hat_row`
- These compile-time constants enable branch elimination

### Newton-Krylov Solver
- No changes needed
- Solver continues to update `stage_increment` in place
- Final values represent converged `u_i = h * k_i`

### Tableau Properties
- `FIRKTableau.accumulates_output` returns `self.b_matches_a_row is None`
- `FIRKTableau.accumulates_error` returns `self.b_hat_matches_a_row is None`
- These properties remain unchanged

### End-Time Observables
- The `ends_at_one` path for final observables computation is PRESERVED
- Only affects whether `observables_function(proposed_state, ...)` is called at end_time

## Data Structures

### stage_increment Buffer Layout
```
[u_0_comp0, u_0_comp1, ..., u_0_compN-1,  // Stage 0 increments
 u_1_comp0, u_1_comp1, ..., u_1_compN-1,  // Stage 1 increments
 ...
 u_S-1_comp0, ..., u_S-1_compN-1]          // Stage S-1 increments
```
Where `u_i_compj = h * k_i[j]` = `h * f(Y_i)[j]`

### Accumulation Indexing
For component `idx` and stage `stage_idx`:
- Index: `stage_idx * n + idx`
- Value: `stage_increment[stage_idx * n + idx]` = `h * k_{stage_idx}[idx]`

## Edge Cases

### 1. Single-Stage FIRK (e.g., Backward Euler as FIRK)
- `stage_count = 1`
- Loop still works correctly with single iteration
- Stiffly-accurate shortcut likely applies

### 2. No Embedded Error Estimate
- `has_error = False`
- Error accumulation path is skipped via compile-time branching
- No impact on optimization

### 3. All-Explicit Tableau (c[0] = 0, first row of a is zero)
- Still works; first stage increment represents explicit Euler-like contribution
- Accumulation formula remains valid

### 4. Non-Identity Mass Matrix
- When M ≠ I, the residual is `M*u - h*f(Y_i) = 0`
- At convergence: `u = M^{-1} * h * f(Y_i)`
- The accumulation formula may need adjustment (TBD: verify with test)
- **Note**: Current FIRK implementation assumes M = I

## Dependencies

### Required Imports (unchanged)
- `numba.cuda`, `numba.int32`
- Precision handling via `from_dtype`

### Required Closures (unchanged)
- `accumulates_output: bool`
- `accumulates_error: bool`
- `b_row: int` (from `b_matches_a_row` or -1)
- `b_hat_row: int` (from `b_hat_matches_a_row` or -1)
- `solution_weights: precision[:]` (tableau.b)
- `error_weights: precision[:]` (tableau.d = b - b_hat)
- `stage_rhs_coeffs: precision[:]` (flattened tableau.a)

## Files to Modify

1. **src/cubie/integrators/algorithms/generic_firk.py**
   - Lines 764-846: Restructure post-solve loop and accumulation

2. **tests/integrators/algorithms/instrumented/generic_firk.py**
   - Lines 457-550: Mirror changes, preserve instrumentation logging

3. **tests/all_in_one.py**
   - Lines 2885-3059: Update `firk_step_inline_factory` consistently

## Verification Strategy

1. Run existing FIRK algorithm tests to ensure numerical equivalence
2. Test with Radau IIA (stiffly-accurate, `b_matches_a_row = 2`)
3. Test with Gauss-Legendre (not stiffly-accurate, `accumulates_output = True`)
4. Verify error estimates match between old and new implementations
