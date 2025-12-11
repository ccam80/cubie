# Krylov Solver and Newton-Krylov Loop Rework - Agent Plan

## Overview

This plan details the architectural changes needed to reduce dynamic indexing and branching in the Krylov and Newton-Krylov solvers. The goal is to improve warp efficiency on CUDA by using predicated commits and single return points.

---

## Component: Linear Solver (`linear_solver.py`)

### Current Behavior

The `linear_solver_factory` and `linear_solver_cached_factory` create CUDA device functions that:
1. Run a loop up to `max_iters`
2. Check convergence after each iteration using `all_sync(mask, converged)`
3. Return immediately from inside the loop on convergence with `return_status = 0 | (iters << 16)`
4. Return `4 | (iters << 16)` after loop exhaustion

**Problem**: Multiple return points create early exits that may cause warp divergence. Status encodes iteration count in upper 16 bits, requiring bit manipulation by callers.

### Expected Behavior After Changes

1. **Single boolean for convergence tracking**: `converged` flag (already present, but used for early return)
2. **Opt-out pattern for converged threads**: Use `selp` to zero out `alpha_effective` (already done), but also skip other operations via predicated values
3. **Single return point**: Break from loop when `all_sync` succeeds, then fall through to single return
4. **Iteration count separation**: Accept an optional `krylov_iters_out` single-element array to write iteration count

### Architectural Changes

#### New Parameter
Add `krylov_iters_out: int32[::1]` parameter to store iteration count. When Newton-Krylov calls this solver, it passes a local array.

#### Control Flow
```
converged = acc <= tol_squared  # Before loop
for iteration in range(max_iters):
    # ... iteration work with predicated commits ...
    converged = converged or (acc <= tol_squared)
    if all_sync(mask, converged):
        break

# Single exit - determine status based on converged flag
final_status = selp(converged, int32(0), int32(4))
krylov_iters_out[0] = iter_count
return final_status
```

#### Edge Cases
- All threads converged on entry (initial residual below tolerance): Loop body never executes, return 0
- Some threads converge mid-loop: Continue with predicated zero alpha for converged threads
- No threads converge by max_iters: Return status 4

### Integration Points

- Called by `newton_krylov_solver` in `newton_krylov.py`
- Called by Rosenbrock step factories via `linear_solver_cached_factory`
- Buffer settings control shared vs local memory for scratch arrays

---

## Component: Newton-Krylov Solver (`newton_krylov.py`)

### Current Behavior

The `newton_krylov_solver_factory` creates a CUDA device function that:
1. Uses `status = status_active` (-1) as the "still working" sentinel
2. Uses `status >= 0` to detect completion
3. Sets status codes (0, 1, 2, 4) based on various error conditions
4. Uses `if all_sync(mask, status >= 0): break` for early loop exit
5. Returns `status | (iters << 16)` encoding both outcome and iteration count

**Problems**:
1. `status` controls both loop and outcome - dynamic branching
2. `if status < 0:` guards work blocks - conditional execution
3. Multiple potential exit paths via status checks
4. Iteration count encoded in status requires caller to bit-shift

### Expected Behavior After Changes

1. **Boolean flags for control**: 
   - `converged`: Thread has reached tolerance
   - `has_error`: Thread encountered unrecoverable error (backtrack failure or linear solver failure)
2. **Status accumulation**: Errors OR'd into `final_status`, not used for control
3. **Predicated work**: `active = not converged and not has_error` gates all work
4. **Single return**: After loop, compute final status from flags and counters

### Architectural Changes

#### Control Variables
```python
converged = norm2_prev <= tol_squared  # Initial check
has_error = False  # Error flag
krylov_iters_local = cuda.local.array(1, int32)  # For linear solver
iters_count = int32(0)
total_krylov_iters = int32(0)
```

#### Main Loop Structure
```python
for _ in range(max_iters):
    active = not converged and not has_error
    if all_sync(mask, not active):
        break
    
    iters_count = selp(active, iters_count + int32(1), iters_count)
    
    # Linear solver call (predicated)
    if active:
        lin_status = linear_solver(..., krylov_iters_local)
        total_krylov_iters += krylov_iters_local[0]
        has_error = has_error or (lin_status != 0)
    
    # Backtracking (predicated)
    # ... (all updates use selp)
    
    converged = converged or (norm2_new <= tol_squared)
```

#### Status Computation at End
```python
# Compute final status from flags
final_status = int32(0)
if has_error:
    # Linear solver error (4) or backtrack failure (1) already set
    final_status |= error_code
elif not converged:
    # Max iterations exceeded
    final_status = int32(2)
# else: success (0)

counters[0] = iters_count
counters[1] = total_krylov_iters
return final_status
```

### Edge Cases
- Converged on entry (initial residual below tolerance): Skip all work, return 0
- Linear solver fails: Set `has_error`, propagate linear solver code
- Backtracking fails to find acceptable step: Set `has_error` with code 1
- Max Newton iterations exceeded: `converged` still False, return 2

### Integration Points
- Called by implicit algorithm step functions (BackwardsEuler, DIRK, FIRK)
- Receives `counters` array from step functions
- Calls linear solver with new iteration count array

---

## Component: Instrumented Solvers (`tests/integrators/algorithms/instrumented/matrix_free_solvers.py`)

### Expected Behavior

Mirror all changes from source files with these additions:
1. **Logging arrays**: Continue to write to logging arrays at each iteration
2. **Slot index**: Continue to use `slot_index` and stage-based slot calculations
3. **Snapshot recording**: Continue to record snapshots for debugging

### Architectural Changes

Same control flow changes as source, plus:
- Write to `linear_initial_guesses`, `linear_iteration_guesses`, etc. at appropriate points
- Maintain `log_slot` tracking for multi-stage algorithms
- Ensure logging happens regardless of convergence state (for debugging)

---

## Component: All-in-One Debug Script (`tests/all_in_one.py`)

### Expected Behavior

The inline factories `linear_solver_inline_factory` and `newton_krylov_inline_factory` must match the source implementations exactly.

### Architectural Changes

Apply identical changes:
1. Add `krylov_iters_out` parameter to linear solver signature
2. Change Newton-Krylov to use boolean flags
3. Single return point for both solvers
4. Predicated commits throughout

---

## Dependencies and Imports

### New Imports Required
None - all required utilities (`selp`, `all_sync`, `activemask`) already imported.

### Data Structures

#### Krylov Iterations Output (new)
- Type: `int32[::1]` (single-element contiguous array)
- Purpose: Store Krylov iteration count for Newton-Krylov to aggregate
- Lifetime: Allocated by Newton-Krylov as local array, passed to linear solver

### Expected Interactions Between Components

```
Step Function (DIRK, etc.)
    └── newton_krylov_solver(counters=[2])
            ├── krylov_iters_local = local.array(1, int32)
            └── linear_solver(..., krylov_iters_local)
                    └── krylov_iters_local[0] = iteration_count
            └── counters[1] += krylov_iters_local[0]
```

---

## Implementation Order

1. **linear_solver.py**: Add `krylov_iters_out` parameter, restructure to single return
2. **newton_krylov.py**: Change to boolean flags, create local array for krylov iters, use predicated commits
3. **matrix_free_solvers.py (instrumented)**: Mirror source changes with logging
4. **all_in_one.py**: Mirror source changes in inline factories

---

## Validation Considerations

### Functional Equivalence
- All status codes should have identical meaning
- Convergence behavior should be identical
- Iteration counts should be identical

### Behavioral Changes
- Upper 16 bits of returned status will be 0 (iteration count no longer encoded)
- This is a breaking change for callers extracting iterations from status

### Test Expectations
- Existing tests that check status codes should continue passing
- Tests that extract iterations via `(status >> 16) & 0xFFFF` will get 0
- These tests should be updated to read from `counters` array instead
