# Implementation Plan: Reduce Nonlinear Solver Memory Footprint

## Component Overview

This plan eliminates the third shared memory buffer (`eval_state`) from the Newton-Krylov solver by moving the evaluation state computation inline into the linear operator and nonlinear residual code generators.

## Architecture Changes

### Newton-Krylov Solver (`newton_krylov.py`)

**Current Behavior:**
- Allocates `shared_scratch` with 3n elements (or 3*s*n for FIRK)
- Partitions into `delta`, `residual`, `eval_state`
- Computes `eval_state[i] = base_state[i % n_base] + a_ij * stage_increment[i]` before calling linear solver
- Passes `eval_state` to linear solver as the `state` parameter

**New Behavior:**
- Allocates `shared_scratch` with 2n elements (or 2*s*n for FIRK)
- Partitions into `delta`, `residual` only
- Removes eval_state computation loop (lines 176-177)
- Passes `stage_increment` directly to linear solver instead of `eval_state`
- Linear solver signature changes from `(state, ...)` to `(stage_increment, ...)`

**Key Implementation Details:**
- Buffer slicing changes:
  - `delta = shared_scratch[:n]` (unchanged)
  - `residual = shared_scratch[n:2*n]` (unchanged)
  - Remove: `eval_state = shared_scratch[2*n:3*n]`
- Update linear solver call (line 178) to pass `stage_increment` instead of `eval_state`
- Residual function already receives `stage_increment` (as parameter `u`), no change needed

### Linear Operator Codegen (`linear_operators.py`)

**Current Behavior:**
- Generated operator functions receive `state` parameter
- `state` is pre-computed eval_state passed from Newton solver
- Uses `state` directly in Jacobian evaluation

**New Behavior:**
- Generated operator functions receive `stage_increment` parameter instead of `state`
- Compute eval_state inline at the point of use:
  ```python
  eval_point = base_state[i % n_base] + a_ij * stage_increment[i]
  ```
- Use `eval_point` in Jacobian evaluation where `state` was previously used

**Templates to Modify:**

1. **`OPERATOR_APPLY_TEMPLATE`** (lines 60-86)
   - Change signature line 83: `(state, parameters, ...)` → `(stage_increment, parameters, ...)`
   - Add computation in body: compute eval_state inline before state substitutions

2. **`CACHED_OPERATOR_APPLY_TEMPLATE`** (lines 25-57)
   - Change signature line 52: `(state, parameters, ...)` → `(stage_increment, parameters, ...)`
   - Add computation in body: compute eval_state inline before state substitutions

3. **`N_STAGE_OPERATOR_TEMPLATE`** (lines 686-710)
   - Change signature line 707: `(state, parameters, ...)` → `(stage_increment, parameters, ...)`
   - Already computes stage states inline (lines 539-548), but uses `state_vec` (which was flattened stage increments)
   - Rename `state_vec` to `stage_increment` for clarity
   - Update `state` parameter references to `stage_increment`

**Code Generation Changes:**

Function `_build_operator_body`:
- Modify state substitution logic to compute eval inline
- Insert eval computation before using state in expressions
- Handle both single-stage (n elements) and multi-stage (s*n elements) cases

Function `_build_n_stage_operator_lines`:
- Update parameter names: `state` → `stage_increment`
- Computation already inline (lines 539-548), just rename for consistency

### Nonlinear Residual Codegen (`nonlinear_residuals.py`)

**Current Behavior:**
- Generated residual functions receive `u` parameter (stage increment)
- Compute eval_state inline: `eval_point = base[i] + aij_sym * u[i]` (line 116)
- This is already the desired pattern!

**New Behavior:**
- No changes needed - already computes eval inline
- Verify compatibility with updated operator signatures
- Template signatures already correct

**Templates (No Modification Needed):**

1. **`RESIDUAL_TEMPLATE`** (lines 23-45)
   - Signature already correct: `(u, parameters, drivers, t, h, a_ij, base_state, out)`
   - Already computes inline: `base[i] + aij_sym * u[i]` (lines 114-116)

2. **`N_STAGE_RESIDUAL_TEMPLATE`** (lines 48-71)
   - Signature already correct
   - Already computes stage states inline (lines 227-235)

### Linear Solver (`linear_solver.py`)

**Current Behavior:**
- Function signature: `linear_solver(state, parameters, drivers, base_state, t, h, a_ij, rhs, x)`
- Calls `operator_apply(state, parameters, drivers, base_state, t, h, a_ij, x, temp)`
- The `state` parameter was pre-computed eval_state

**New Behavior:**
- Function signature: `linear_solver(stage_increment, parameters, drivers, base_state, t, h, a_ij, rhs, x)`
- Calls `operator_apply(stage_increment, parameters, drivers, base_state, t, h, a_ij, x, temp)`
- Operator will compute eval_state inline from `stage_increment`, `base_state`, and `a_ij`

**Functions to Modify:**

1. **`linear_solver_factory`** (lines 18-210)
   - Update signature line 79: `state` → `stage_increment`
   - Update operator call line 133: first parameter `state` → `stage_increment`
   - Update operator call line 162: first parameter `state` → `stage_increment`
   - Update docstring to reflect parameter name change

2. **`linear_solver_cached_factory`** (lines 213-340)
   - Update signature line 239: `state` → `stage_increment`
   - Update operator call line 257: first parameter `state` → `stage_increment`
   - Update operator call line 290: first parameter `state` → `stage_increment`
   - Update docstring to reflect parameter name change

### FIRK Algorithm (`generic_firk.py`)

**Current Behavior:**
- Property `solver_shared_elements` returns `3 * self.compile_settings.all_stages_n` (line 625)
- This allocates space for delta, residual, and eval_state

**New Behavior:**
- Property `solver_shared_elements` returns `2 * self.compile_settings.all_stages_n`
- Allocates only delta and residual

**Change:**
- Line 625: Change `3 * self.compile_settings.all_stages_n` to `2 * self.compile_settings.all_stages_n`

### DIRK Algorithm (`generic_dirk.py`)

**Current Behavior:**
- Property `solver_shared_elements` returns `3 * n` (implicit from base class or local calculation)
- Need to locate exact line - may be computed elsewhere

**New Behavior:**
- Property `solver_shared_elements` returns `2 * n`

**Investigation Needed:**
- Find where `solver_shared_elements` is defined for DIRK
- May be in `ODEImplicitStep` base class
- Update to `2 * n` instead of `3 * n`

## Data Structures

### Shared Memory Layout Changes

**Before:**
```
shared_scratch[0:n]       -> delta
shared_scratch[n:2*n]     -> residual  
shared_scratch[2*n:3*n]   -> eval_state
```

**After:**
```
shared_scratch[0:n]       -> delta
shared_scratch[n:2*n]     -> residual
```

### Multi-Stage Layout Changes (FIRK)

**Before:**
```
shared_scratch[0:s*n]           -> delta (all stages flattened)
shared_scratch[s*n:2*s*n]       -> residual (all stages flattened)
shared_scratch[2*s*n:3*s*n]     -> eval_state (all stages flattened)
```

**After:**
```
shared_scratch[0:s*n]           -> delta (all stages flattened)
shared_scratch[s*n:2*s*n]       -> residual (all stages flattened)
```

## Expected Component Interactions

### Newton-Krylov → Linear Solver → Operator

**Call Chain:**
1. Newton-Krylov calls `linear_solver(stage_increment, parameters, drivers, base_state, ...)`
2. Linear solver calls `operator_apply(stage_increment, parameters, drivers, base_state, ...)`
3. Operator computes `eval[i] = base_state[i % n_base] + a_ij * stage_increment[i]`
4. Operator evaluates Jacobian at `eval` point
5. Returns result to linear solver

### Newton-Krylov → Residual Function

**Call Chain:**
1. Newton-Krylov calls `residual_function(stage_increment, parameters, drivers, ...)`
2. Residual function computes `eval[i] = base_state[i] + a_ij * stage_increment[i]` (already inline)
3. Residual evaluates `f(t, eval)` at `eval` point
4. Returns residual to Newton-Krylov

## Dependencies and Imports

No new dependencies required. All changes are within existing modules.

**Modules Affected:**
- `cubie.integrators.matrix_free_solvers.newton_krylov`
- `cubie.integrators.matrix_free_solvers.linear_solver`
- `cubie.odesystems.symbolic.codegen.linear_operators`
- `cubie.odesystems.symbolic.codegen.nonlinear_residuals` (verify only, no changes)
- `cubie.integrators.algorithms.generic_firk`
- `cubie.integrators.algorithms.generic_dirk`

**Import Dependencies:**
- All existing imports remain unchanged
- No new imports needed

## Edge Cases

### Single-Stage Methods (DIRK)
- `n_base = n` (base_state and stage_increment have same size)
- Modulo operation `i % n_base` is no-op when `n_base == n`
- Inline computation is simply: `eval[i] = base_state[i] + a_ij * stage_increment[i]`

### Multi-Stage Methods (FIRK)
- `n_base = n` (base_state size), `stage_increment.size = s * n`
- Modulo operation `i % n_base` wraps correctly for all stages
- Stage 0: `i in [0, n)` → `i % n = i`
- Stage 1: `i in [n, 2*n)` → `i % n = i - n`
- Stage k: `i in [k*n, (k+1)*n)` → `i % n = i - k*n`

### Explicit Stages (a_ij = 0)
- When `a_ij = 0`, `eval[i] = base_state[i % n_base] + 0 * stage_increment[i] = base_state[i % n_base]`
- Compiler should optimize multiplication by zero
- No special handling needed

### First Newton Iteration
- `stage_increment` may contain initial guess
- Initial guess typically zero or from predictor
- Inline computation still valid: `eval[i] = base_state[i % n_base] + a_ij * 0`

### Cached Operators
- Cached operators already have separate signature including `cached_aux`
- Need to update to use `stage_increment` instead of `state`
- Inline computation happens before accessing cached auxiliaries

### Preconditioners
- Preconditioners may use operator-like signatures
- Check if preconditioners are generated with similar templates
- Update if needed to maintain consistency

## Performance Considerations

### Memory Access Patterns
- Inline computation adds one memory load (base_state) per element
- base_state is read-only, likely cached in L1/L2
- Stage_increment already accessed for operator evaluation
- Additional cost is negligible compared to Jacobian evaluation

### Register Pressure
- Inline computation uses temporary register for eval_point
- No increase in long-lived register usage
- Compiler can reuse registers between iterations

### Shared Memory Savings
- 33% reduction in shared memory usage per solver
- Improves occupancy on memory-constrained GPUs
- Enables larger system sizes

### Warp Divergence
- No new conditional branches introduced
- Modulo operation may cause minor arithmetic overhead
- Overall impact negligible

## Testing Strategy

### Existing Tests to Pass
- All tests in `tests/integrators/matrix_free_solvers/`
- All tests in `tests/integrators/algorithms/test_generic_firk.py`
- All tests in `tests/integrators/algorithms/test_generic_dirk.py`
- All integration tests using implicit methods

### Instrumented Tests
- `tests/integrators/algorithms/instrumented/matrix_free_solvers.py` has instrumented versions
- Instrumented Newton-Krylov already computes eval_state inline (line 409)
- Instrumented tests verify buffer usage and sizes
- These tests validate the change doesn't break solver behavior

### No New Tests Needed
- Behavior is identical to current implementation
- Existing tests fully cover the functionality
- Buffer size reduction is validated by existing shared memory checks

## Rollout Considerations

### Code Regeneration
- All ODE systems need regeneration after codegen template changes
- Users must regenerate their systems on next import/compilation
- Cached compiled functions will be invalidated automatically

### Performance Validation
- Monitor CI performance benchmarks
- Shared memory usage should decrease measurably
- Throughput may increase on memory-bound workloads

### Documentation Updates
- No user-facing API changes
- Internal documentation may need updates
- Code comments should reflect new parameter names
