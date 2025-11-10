# Agent Implementation Plan: Reduce Nonlinear Solver Memory Footprint

## Objective

Eliminate the `eval_state` buffer from Newton-Krylov solver to reduce shared memory usage from 3n to 2n elements, achieving a 33% reduction in memory footprint for implicit solvers.

## Component Overview

### 1. Newton-Krylov Solver (`newton_krylov.py`)

**Current Behavior:**
- Allocates 3 n-sized buffers: `delta`, `residual`, `eval_state`
- Computes `eval_state[i] = base_state[i % n_base] + a_ij * stage_increment[i]`
- Passes `eval_state` as first argument to `linear_solver()`
- Residual function receives `stage_increment` and computes state internally

**Expected New Behavior:**
- Allocates 2 n-sized buffers: `delta`, `residual` only
- Does NOT compute eval_state
- Passes `stage_increment` as first argument to `linear_solver()` instead of `eval_state`
- Linear solver and residual functions compute `base_state + a_ij * stage_increment` inline

**Key Changes:**
- Remove line 143: `eval_state = shared_scratch[2 * n: 3 * n]`
- Remove lines 176-177: eval_state population loop
- Update line 178: Change `linear_solver(eval_state, ...)` to `linear_solver(stage_increment, ...)`
- Update docstring to reflect 2-buffer usage (currently says 3 buffers in lines 115-137)
- Update scratch space requirement comment (line 130: "three vectors" → "two vectors")

### 2. Nonlinear Residual Codegen (`nonlinear_residuals.py`)

**Current Behavior:**
The single-stage residual (`_build_residual_lines`) already does inline computation correctly:
```python
state_subs[state_sym] = base[i] + aij_sym * u[i]
```
This creates substitution where state symbols are replaced with the evaluation point.

The n-stage residual (`_build_n_stage_residual_lines`) also does this correctly:
```python
stage_state_subs[state_sym] = expr  # where expr = base_state + sum(coeffs * u[...])
```

**Expected Behavior:**
No changes needed! The residual codegen already computes state inline via symbolic substitution.

**Verification Needed:**
- Confirm generated code doesn't reference a separate `state` or `eval_state` parameter
- Ensure `u` (stage_increment) and `base_state` are accessed directly

### 3. Linear Operator Codegen (`linear_operators.py`)

**Current Behavior:**
Templates define `operator_apply` with signature:
```python
def operator_apply(state, parameters, drivers, base_state, t, h, a_ij, v, out)
```

The body building functions receive `state` parameter but need to use it to compute the evaluation point.

Looking at `_build_operator_body` and `_build_n_stage_operator_lines`:
- They build expressions that reference `state` symbols
- State symbols are mapped via `index_map.all_arrayrefs`
- Need to update substitution to compute `base_state + a_ij * state` inline

**Expected New Behavior:**
- Signature remains `operator_apply(state, parameters, drivers, base_state, t, h, a_ij, v, out)`
  where `state` is now the stage_increment
- State symbol substitutions compute: `state_sym → base_state[i] + a_ij * state[i]`
- This matches the pattern in residual codegen

**Key Changes:**

For single-stage operator (`_build_operator_body`):
- Currently doesn't explicitly build state substitution (relies on passed state)
- Need to add state_subs similar to residual codegen:
  ```python
  state_subs = {}
  for i, state_sym in enumerate(state_symbols):
      state_subs[state_sym] = base_state[i] + a_ij_sym * state[i]
  ```
- Apply this substitution to JVP terms and auxiliary assignments

For n-stage operator (`_build_n_stage_operator_lines`):
- Lines 537-548 already build `stage_state_subs` with the pattern:
  ```python
  expr = base_state[state_idx]
  for contrib_idx in range(stage_count):
      expr += coeff_sym * state_vec[...]
  ```
- This is correct! It computes the evaluation state inline
- Verify this is applied to all expressions

**Template Signature Updates:**
The templates already have correct signatures:
- `OPERATOR_APPLY_TEMPLATE` line 83: `state, parameters, drivers, base_state, t, h, a_ij, v, out`
- `CACHED_OPERATOR_APPLY_TEMPLATE` line 52: Same signature
- `N_STAGE_OPERATOR_TEMPLATE` line 707: Same signature

### 4. Linear Solver (`linear_solver.py`)

**Current Behavior:**
Signature line 79-89:
```python
def linear_solver(state, parameters, drivers, base_state, t, h, a_ij, rhs, x):
```
Calls `operator_apply(state, parameters, drivers, base_state, t, h, a_ij, x, temp)`

**Expected Behavior:**
No changes needed to the linear solver itself! It already passes `state` (which will now be `stage_increment`) to the operator. The operator will handle inline computation.

**Verification:**
- Confirm parameter forwarding is correct
- Ensure no assumptions about `state` being pre-evaluated

### 5. Generic FIRK Algorithm (`generic_firk.py`)

**Current Behavior:**
Creates residual and linear operator via codegen, then passes to `newton_krylov_solver_factory`.

**Expected Behavior:**
No changes needed! The algorithm just provides the building blocks. The Newton-Krylov solver and codegen changes handle everything.

**Verification:**
- Test that FIRK integration still works
- Verify convergence behavior unchanged

### 6. Generic DIRK Algorithm (`generic_dirk.py`)

**Current Behavior:**
Similar to FIRK but for diagonal implicit methods. Creates residual/operator, passes to Newton-Krylov.

**Expected Behavior:**
No changes needed! Same reasoning as FIRK.

**Verification:**
- Test that DIRK integration still works
- Verify convergence behavior unchanged

## Architectural Integration

### Data Flow Changes

**Before:**
```
Newton-Krylov (line 176-177):
  for i in range(n):
      eval_state[i] = base_state[i % n_base] + a_ij * stage_increment[i]

Linear Solver Call (line 178):
  linear_solver(eval_state, params, drivers, base_state, t, h, a_ij, ...)
  
Operator Apply (in codegen):
  Uses state parameter directly (pre-computed eval_state)
```

**After:**
```
Newton-Krylov:
  # No eval_state computation
  
Linear Solver Call:
  linear_solver(stage_increment, params, drivers, base_state, t, h, a_ij, ...)
  
Operator Apply (in codegen):
  state_eval[i] = base_state[i] + a_ij * stage_increment[i]  # inline
  # Use state_eval in Jacobian computations
```

### Expected Interactions

1. **Newton-Krylov ↔ Linear Solver:**
   - Newton-Krylov passes `stage_increment` instead of `eval_state`
   - Linear solver forwards to operator unchanged
   
2. **Linear Solver ↔ Operator:**
   - Linear solver passes `stage_increment` as `state` parameter
   - Operator computes evaluation point inline

3. **Newton-Krylov ↔ Residual:**
   - Newton-Krylov passes `stage_increment` (no change here)
   - Residual computes evaluation point inline (already does this)

## Edge Cases

### 1. Multi-stage (FIRK) vs Single-stage (DIRK)

**Challenge:** FIRK has `s*n` stage increments, DIRK has `n`.

**Solution:** 
- FIRK n-stage operator/residual already handles this with stage-specific indexing
- The inline computation respects stage structure via coefficient matrix
- `base_state[i % n_base]` pattern in Newton-Krylov works for both

### 2. Cached vs Non-cached Operators

**Challenge:** Some operators use cached auxiliaries, others don't.

**Solution:**
- State substitution happens before auxiliary computation
- Cached auxiliaries depend on state, so they'll use the inline-computed state
- No special handling needed

### 3. Preconditioners

**Challenge:** Preconditioners may also receive state parameter.

**Current Investigation:**
- Need to check if preconditioners use `state` parameter
- If yes, they need same inline computation pattern

**Expected:** Preconditioners in `preconditioners.py` likely need review, but they're optional components. Primary focus is operator/residual.

### 4. Zero `a_ij` coefficient

**Challenge:** When `a_ij = 0`, state evaluation becomes just `base_state[i]`.

**Solution:**
- Expression `base_state[i] + 0 * stage_increment[i]` is fine
- Compiler will optimize away the zero multiplication
- SymPy may simplify during expression building

## Dependencies

### Required Imports
No new imports needed. All changes are within existing modules.

### Component Dependencies
1. `newton_krylov.py` depends on:
   - `residual_function` (callback)
   - `linear_solver` (callback)
   
2. `linear_operators.py` depends on:
   - `JVPEquations` for Jacobian expressions
   - `IndexedBases` for symbol mapping
   - SymPy for symbolic manipulation

3. `nonlinear_residuals.py` depends on:
   - `ParsedEquations` for ODE system
   - `IndexedBases` for symbol mapping
   - SymPy for symbolic manipulation

### Integration Points
- ODE system creation → generates operators/residuals with new inline pattern
- Algorithm step creation → passes generated functions to Newton-Krylov
- Newton-Krylov → calls linear solver and residual with new signatures

## Memory Layout Changes

### Before (3 Buffers)
```
shared_scratch array (size 3n):
  [0 : n)       → delta (Newton direction)
  [n : 2n)      → residual
  [2n : 3n)     → eval_state
```

**Total:** 3n elements

### After (2 Buffers)
```
shared_scratch array (size 2n):
  [0 : n)       → delta (Newton direction)
  [n : 2n)      → residual
```

**Total:** 2n elements

**Savings:** n elements = 33% reduction

## Testing Strategy

### Unit Tests Affected

1. **`test_newton_krylov.py`:**
   - Tests for solver convergence
   - Verify same iteration counts
   - Verify same final residuals
   - May need fixture updates if they mock eval_state

2. **`test_linear_solver.py`:**
   - Tests for linear solver iterations
   - Should be unaffected (just parameter forwarding)

3. **`test_solver_helpers.py`:**
   - Tests for codegen output
   - Verify generated functions have correct signatures
   - Verify inline state computation in generated code

4. **Algorithm tests (`generic_dirk.py`, `generic_firk.py`):**
   - Integration tests for full step functions
   - Verify convergence behavior unchanged
   - Compare against reference solutions

### Expected Test Changes

**Breaking tests:**
- Tests that explicitly check for 3-buffer allocation
- Tests that verify eval_state buffer contents
- Tests that mock the eval_state computation

**Unchanged tests:**
- Convergence criteria tests
- Error tolerance tests
- Iteration count tests (should be identical)

### Validation Approach

1. Run existing test suite to establish baseline
2. Make changes incrementally:
   - Step 1: Update Newton-Krylov (will break tests)
   - Step 2: Update operator codegen
   - Step 3: Update residual codegen (if needed)
   - Step 4: Fix test fixtures
3. Compare numerical results against baseline
4. Verify memory usage with profiling

## Performance Considerations

### Computational Overhead

**Added:** Inline computation of `base_state[i] + a_ij * stage_increment[i]`
- Simple addition and multiplication
- Happens during Jacobian evaluation (already expensive)
- Negligible compared to ODE function evaluation

**Removed:** Buffer copying in Newton-Krylov
- Eliminates loop that populates eval_state
- Saves n iterations of memory writes

**Net:** Likely slightly faster due to better memory locality

### Memory Access Pattern

**Before:**
1. Write eval_state (sequential)
2. Read eval_state in operator (scattered based on JVP)

**After:**
1. Read base_state + stage_increment directly (scattered)

**Impact:** Fewer memory transactions, better cache behavior

### GPU Occupancy

**Critical Benefit:** 33% reduction in shared memory per thread block
- More blocks can run concurrently
- Higher GPU utilization
- This is the PRIMARY motivation for the change

## Behavior Descriptions

### Newton-Krylov Solver

**Responsibility:** Orchestrate damped Newton iteration with line search

**New Behavior:**
1. Compute initial residual by calling `residual_function(stage_increment, ...)`
2. Enter Newton loop (max_iters iterations):
   - Call `linear_solver(stage_increment, ..., delta)` to solve for Newton direction
   - Backtracking line search:
     - Update `stage_increment[i] += scale * delta[i]`
     - Evaluate residual at new point
     - Accept/reject based on norm reduction
3. Return status and iteration count

**Key Change:** Never materializes `eval_state` buffer. Relies on operators/residuals to compute evaluation state on-demand.

### Linear Operator (Single-Stage)

**Responsibility:** Compute `beta * M * v - gamma * a_ij * h * J * v`

**New Behavior:**
1. Receive `state` (actually stage_increment), `base_state`, `a_ij`, and `v` (search direction)
2. For each state variable in ODE:
   - Compute evaluation point: `y[i] = base_state[i] + a_ij * state[i]`
   - Substitute into symbolic expressions for Jacobian
3. Evaluate mass matrix contribution: `M * v`
4. Evaluate Jacobian-vector product: `J * v` at evaluation point
5. Combine: `out = beta * (M * v) - gamma * a_ij * h * (J * v)`

### Linear Operator (N-Stage)

**Responsibility:** Compute operator for coupled FIRK stages

**New Behavior:**
1. For each stage s = 0..S-1:
   - Compute stage evaluation point:
     `y_s[i] = base_state[i] + sum_j(A[s,j] * state[j*n + i])`
   - Substitute into symbolic expressions for stage s
   - Evaluate stage contribution to operator
2. Output is flattened `s*n` vector

### Nonlinear Residual (Single-Stage)

**Responsibility:** Compute `beta * M * u - gamma * h * f(t, base_state + a_ij * u)`

**Current Behavior (CORRECT):**
Already does inline computation via symbolic substitution.

**Verification:** No changes needed, just confirm it works with updated Newton-Krylov.

### Nonlinear Residual (N-Stage)

**Responsibility:** Compute residual for coupled FIRK stages

**Current Behavior (CORRECT):**
Already does inline computation via symbolic substitution.

**Verification:** No changes needed, just confirm it works with updated Newton-Krylov.

## Expected Component Behavior Summary

| Component | Current Behavior | New Behavior | Change Required |
|-----------|------------------|--------------|-----------------|
| Newton-Krylov | Allocates eval_state, populates it, passes to linear solver | Skips eval_state, passes stage_increment to linear solver | YES - Remove buffer |
| Linear Solver | Forwards state parameter to operator | Forwards state (now stage_increment) to operator | NO - Just parameter pass-through |
| Operator (single) | Uses state parameter (pre-computed) | Computes `base + a_ij * state` inline | YES - Add inline computation |
| Operator (n-stage) | Uses state with stage indexing | Computes stage state inline | VERIFY - May already do this |
| Residual (single) | Computes `base + a_ij * u` inline | Same | NO - Already correct |
| Residual (n-stage) | Computes stage state inline | Same | NO - Already correct |
| DIRK Algorithm | Creates residual/operator, calls NK | Same | NO - No changes |
| FIRK Algorithm | Creates residual/operator, calls NK | Same | NO - No changes |

## Notes for Downstream Agents

### For detailed_implementer:
- Focus on the 3 components requiring changes: `newton_krylov.py`, `linear_operators.py` (single-stage operator)
- Verify n-stage operator already has inline computation
- Pay careful attention to symbolic substitution in codegen
- Test each change incrementally

### For reviewer:
- Verify memory usage is actually 2n (check shared_scratch allocation)
- Confirm numerical results match baseline tests
- Check that state evaluation happens exactly once per operator/residual call
- Validate no memory access patterns cause race conditions

### Critical Implementation Details:
1. In `linear_operators.py`, add state substitution BEFORE auxiliary computation
2. The substitution must respect the `index_map.states` ordering
3. Use SymPy `.subs()` method to replace state symbols with evaluation expressions
4. Ensure `a_ij` symbol is available in scope when building expressions
