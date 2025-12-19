# Agent Implementation Plan: Instrumented Matrix-Free Solvers

## Overview

This plan details the conversion of instrumented matrix-free solvers from factory functions to CUDAFactory subclasses. The implementation must preserve all existing logging functionality while adopting the production architecture pattern.

## Component 1: InstrumentedLinearSolverCache

**Type:** attrs class (CUDAFunctionCache subclass)

**Location:** `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`

**Purpose:** Container for compiled instrumented linear solver device function

**Expected Behavior:**
- Holds single device function reference
- Used as return value from InstrumentedLinearSolver.build()
- Validates device function with is_device_validator

**Attributes:**
- `linear_solver`: Callable (device function with logging signature)

**Dependencies:**
- Import from cubie.CUDAFactory: CUDAFunctionCache
- Import from cubie._utils: is_device_validator

## Component 2: InstrumentedLinearSolver

**Type:** Python class (LinearSolver subclass)

**Location:** `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`

**Purpose:** Factory for instrumented linear solver device functions with iteration logging

**Expected Behavior:**
- Inherits from LinearSolver
- Reuses parent's __init__, compile_settings, buffer_registry interaction
- Overrides build() to compile instrumented device function
- Returns InstrumentedLinearSolverCache from build()
- Supports both use_cached_auxiliaries=True and False variants

**Inheritance:**
- Inherits: `cubie.integrators.matrix_free_solvers.linear_solver.LinearSolver`
- Inherits __init__ unchanged (no override needed)
- Inherits buffer registration from parent (production buffers still registered)

**Overridden Methods:**

### build() -> InstrumentedLinearSolverCache

**Access compile_settings via:** `self.compile_settings`

**Extract from config:**
- operator_apply (validate not None)
- preconditioner (optional)
- n (vector length)
- correction_type ('steepest_descent' or 'minimal_residual')
- tolerance (squared internally)
- max_iters
- precision
- use_cached_auxiliaries (determines signature variant)

**Compute flags:**
- sd_flag = correction_type == "steepest_descent"
- mr_flag = correction_type == "minimal_residual"
- preconditioned = preconditioner is not None

**Type conversions:**
- n_val = int32(n)
- max_iters_val = int32(max_iters)
- precision_numba = from_dtype(np.dtype(precision))
- typed_zero = precision_numba(0.0)
- tol_squared = precision_numba(tolerance * tolerance)

**Get allocators from buffer_registry:**
- Use parent's registered buffers for preconditioned_vec and temp
- Call buffer_registry.get_allocator() with appropriate names based on use_cached_auxiliaries

**Compile two device function variants:**

#### Variant 1: use_cached_auxiliaries=False

**Signature:**
```python
@cuda.jit(device=True, inline=True, **compile_kwargs)
def linear_solver(
    state, parameters, drivers, base_state,
    t, h, a_ij, rhs, x, shared, krylov_iters_out,
    # Logging parameters:
    slot_index,
    linear_initial_guesses,
    linear_iteration_guesses,
    linear_residuals,
    linear_squared_norms,
    linear_preconditioned_vectors,
):
```

**Logic flow:**
1. Allocate preconditioned_vec and temp from buffer_registry allocators
2. Initial operator application: operator_apply(state, parameters, drivers, base_state, t, h, a_ij, x, temp)
3. Compute initial residual in rhs[i] = rhs[i] - temp[i]
4. Compute initial squared norm accumulator
5. Check convergence: acc <= tol_squared
6. **Logging:** Record x[i] to linear_initial_guesses[slot_index, i]
7. Initialize iteration = int32(0)
8. Iteration loop (max_iters_val):
   - Break if all_sync(mask, converged)
   - Increment iteration counter
   - Apply preconditioner or copy rhs to preconditioned_vec
   - Apply operator to preconditioned_vec -> temp
   - Compute numerator/denominator based on sd_flag or mr_flag
   - Compute alpha with selp for zero denominator
   - Apply alpha_effective = selp(converged, precision_numba(0.0), alpha)
   - Update x[i] and rhs[i]
   - Recompute squared norm accumulator
   - Update converged flag
   - **Logging:** log_iter = iteration - int32(1)
     - Record x[i] to linear_iteration_guesses[slot_index, log_iter, i]
     - Record rhs[i] to linear_residuals[slot_index, log_iter, i]
     - Record preconditioned_vec[i] to linear_preconditioned_vectors[slot_index, log_iter, i]
     - Record acc to linear_squared_norms[slot_index, log_iter]
9. Compute final_status = selp(converged, int32(0), int32(4))
10. Store iteration to krylov_iters_out[0]
11. Return final_status

#### Variant 2: use_cached_auxiliaries=True

**Signature:**
```python
@cuda.jit(device=True, inline=True, **compile_kwargs)
def linear_solver_cached(
    state, parameters, drivers, base_state, cached_aux,
    t, h, a_ij, rhs, x, shared, krylov_iters_out,
    # Logging parameters:
    slot_index,
    linear_initial_guesses,
    linear_iteration_guesses,
    linear_residuals,
    linear_squared_norms,
    linear_preconditioned_vectors,
):
```

**Logic flow:**
- Identical to variant 1 except operator_apply and preconditioner calls include cached_aux parameter

**Return value:**
- InstrumentedLinearSolverCache(linear_solver=<compiled function>)

## Component 3: InstrumentedNewtonKrylovCache

**Type:** attrs class (CUDAFunctionCache subclass)

**Location:** `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`

**Purpose:** Container for compiled instrumented Newton-Krylov solver device function

**Expected Behavior:**
- Holds single device function reference
- Used as return value from InstrumentedNewtonKrylov.build()
- Validates device function with is_device_validator

**Attributes:**
- `newton_krylov_solver`: Callable (device function with logging signature)

**Dependencies:**
- Import from cubie.CUDAFactory: CUDAFunctionCache
- Import from cubie._utils: is_device_validator

## Component 4: InstrumentedNewtonKrylov

**Type:** Python class (NewtonKrylov subclass)

**Location:** `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`

**Purpose:** Factory for instrumented Newton-Krylov solver device functions with iteration logging

**Expected Behavior:**
- Inherits from NewtonKrylov
- Reuses parent's __init__, compile_settings, buffer_registry interaction
- Overrides build() to compile instrumented device function
- Returns InstrumentedNewtonKrylovCache from build()
- Embeds InstrumentedLinearSolver device function

**Inheritance:**
- Inherits: `cubie.integrators.matrix_free_solvers.newton_krylov.NewtonKrylov`
- Inherits __init__ unchanged (no override needed)
- Inherits buffer registration from parent (delta, residual, residual_temp, stage_base_bt)

**Overridden Methods:**

### build() -> InstrumentedNewtonKrylovCache

**Access compile_settings via:** `self.compile_settings`

**Extract from config:**
- residual_function (validate not None)
- linear_solver (validate not None, must be InstrumentedLinearSolver instance)
- n (vector length)
- tolerance (squared internally)
- max_iters
- damping
- max_backtracks
- precision

**Validate linear_solver type:**
- Must be InstrumentedLinearSolver instance
- Access device function via linear_solver.device_function (triggers linear solver build if needed)

**Type conversions:**
- precision_dtype = np.dtype(precision)
- numba_precision = from_dtype(precision_dtype)
- tol_squared = numba_precision(tolerance * tolerance)
- typed_zero = numba_precision(0.0)
- typed_one = numba_precision(1.0)
- typed_damping = numba_precision(damping)
- n_val = int32(n)
- max_iters_val = int32(max_iters)
- max_backtracks_val = int32(max_backtracks + 1)

**Get allocators from buffer_registry:**
- alloc_delta = buffer_registry.get_allocator('newton_delta', self)
- alloc_residual = buffer_registry.get_allocator('newton_residual', self)
- alloc_residual_temp = buffer_registry.get_allocator('newton_residual_temp', self)
- alloc_stage_base_bt = buffer_registry.get_allocator('newton_stage_base_bt', self)

**Compute linear solver shared offset:**
- lin_shared_offset = buffer_registry.shared_buffer_size(self)

**Compile device function:**

**Signature:**
```python
@cuda.jit(device=True, inline=True, **compile_kwargs)
def newton_krylov_solver(
    stage_increment, parameters, drivers,
    t, h, a_ij, base_state, shared_scratch, counters,
    # Logging parameters:
    stage_index,
    newton_initial_guesses,
    newton_iteration_guesses,
    newton_residuals,
    newton_squared_norms,
    newton_iteration_scale,
    linear_initial_guesses,
    linear_iteration_guesses,
    linear_residuals,
    linear_squared_norms,
    linear_preconditioned_vectors,
):
```

**Logic flow:**
1. Allocate buffers from registry:
   - delta = alloc_delta(shared_scratch, shared_scratch)
   - residual = alloc_residual(shared_scratch, shared_scratch)
   - residual_temp = alloc_residual_temp(shared_scratch, shared_scratch)
   - stage_base_bt = alloc_stage_base_bt(shared_scratch, shared_scratch)

2. Initialize delta and residual to zero

3. Evaluate initial residual:
   - residual_function(stage_increment, parameters, drivers, t, h, a_ij, base_state, residual)
   - Compute norm2_prev from residual
   - Negate residual values: residual[i] = -residual[i]
   - **Logging:** Record stage_increment[i] to newton_initial_guesses[stage_index, i]

4. Create local arrays:
   - residual_copy = cuda.local.array(n, numba_precision)
   - Copy residual to residual_copy
   - **Logging:** Record to newton_iteration_guesses[stage_index, 0, i]
   - **Logging:** Record to newton_residuals[stage_index, 0, i]
   - **Logging:** Record norm2_prev to newton_squared_norms[stage_index, 0]
   - Initialize log_index = int32(1)

5. Check initial convergence: converged = norm2_prev <= tol_squared

6. Initialize state variables:
   - has_error = False
   - final_status = int32(0)
   - krylov_iters_local = cuda.local.array(1, int32)
   - iters_count = int32(0)
   - total_krylov_iters = int32(0)
   - mask = activemask()

7. Newton iteration loop (max_iters_val):
   - Break if all_sync(mask, converged or has_error)
   - Predicated iteration count: iters_count = selp(active, iters_count + 1, iters_count)
   
   - Compute linear solver slot: linear_slot_base = int32(stage_index * max_iters)
   - Compute iter_slot = int(iters_count) - 1
   - Reset krylov_iters_local[0] = int32(0)
   
   - Call linear solver:
     ```python
     lin_shared = shared_scratch[lin_shared_offset:]
     lin_status = linear_solver_fn(
         stage_increment, parameters, drivers, base_state,
         t, h, a_ij, residual, delta, lin_shared, krylov_iters_local,
         # Logging parameters:
         linear_slot_base + iter_slot,
         linear_initial_guesses,
         linear_iteration_guesses,
         linear_residuals,
         linear_squared_norms,
         linear_preconditioned_vectors,
     )
     ```
   
   - Check linear solver status and update has_error, final_status
   - Accumulate total_krylov_iters
   
   - Save stage_base_bt = stage_increment for backtracking
   
   - Backtracking loop (max_backtracks_val):
     - Compute trial: stage_increment = stage_base_bt + alpha * delta
     - Evaluate residual_function -> residual_temp
     - Compute norm2_new
     - **Snapshot:** stage_increment_snapshot and residual_snapshot for logging
     - Check convergence: norm2_new <= tol_squared
     - Check improvement: norm2_new < norm2_prev
     - If found_step, copy residual_temp to residual (negated)
     - Damp alpha *= typed_damping
   
   - Handle backtrack failure
   - Revert stage_increment if backtracked failed
   
   - **Logging (if snapshot_ready):**
     - Record stage_increment_snapshot[i] to newton_iteration_guesses[stage_index, log_index, i]
     - Record residual_snapshot[i] to newton_residuals[stage_index, log_index, i]
     - Record norm2_new to newton_squared_norms[stage_index, log_index]
     - Increment log_index
   - **Logging:** Record alpha to newton_iteration_scale[stage_index, iter_slot]

8. Handle max iterations exceeded

9. Write counters:
   - counters[0] = iters_count
   - counters[1] = total_krylov_iters

10. Return int32(final_status)

**Return value:**
- InstrumentedNewtonKrylovCache(newton_krylov_solver=<compiled function>)

## Component 5: Test Infrastructure Updates

**File:** `tests/integrators/algorithms/instrumented/conftest.py`

**Changes Required:**

### Import Updates
- Add: `from .matrix_free_solvers import InstrumentedLinearSolver, InstrumentedNewtonKrylov`
- Add: `from cubie.integrators.matrix_free_solvers.linear_solver import LinearSolverConfig`
- Add: `from cubie.integrators.matrix_free_solvers.newton_krylov import NewtonKrylovConfig`

### Fixture Updates

**Replace factory function calls with class instantiation:**

**Current pattern:**
```python
linear_solver_fn = inst_linear_solver_factory(
    operator_apply=...,
    n=...,
    precision=...,
    ...
)
```

**Target pattern:**
```python
linear_solver_config = LinearSolverConfig(
    precision=precision,
    n=n,
    operator_apply=operator_apply,
    preconditioner=preconditioner,
    correction_type='minimal_residual',
    tolerance=tolerance,
    max_iters=max_iters,
    use_cached_auxiliaries=use_cached,
)
linear_solver_instance = InstrumentedLinearSolver(linear_solver_config)
linear_solver_fn = linear_solver_instance.device_function
```

**For Newton-Krylov:**
```python
newton_config = NewtonKrylovConfig(
    precision=precision,
    n=n,
    residual_function=residual_function,
    linear_solver=linear_solver_instance,  # Pass instance, not function
    tolerance=tolerance,
    max_iters=max_iters,
    damping=damping,
    max_backtracks=max_backtracks,
)
newton_instance = InstrumentedNewtonKrylov(newton_config)
newton_solver_fn = newton_instance.device_function
```

**Expected fixtures needing updates:**
- Any fixture creating linear solvers for implicit algorithms
- Any fixture creating Newton-Krylov solvers for implicit algorithms
- Fixtures in instrumented algorithm modules (backwards_euler.py, etc.)

## Component 6: Instrumented Algorithm Updates

**Files to update:**
- `backwards_euler.py`
- `backwards_euler_predict_correct.py`
- `crank_nicolson.py`
- `generic_dirk.py`
- `generic_firk.py`
- `generic_rosenbrock_w.py`

**Expected changes:**

### Constructor Updates
- Accept solver instances instead of device functions
- Extract device functions via `.device_function` property

### Device Function Call Updates
- Add logging array parameters to all solver calls
- Ensure slot_index calculations match expected format
- For linear solver: slot_index = stage_index or linear_slot_base + iter_slot
- For Newton-Krylov: stage_index passed directly

**Example transformation:**

**Current:**
```python
status = newton_solver(
    stage_increment, parameters, drivers,
    t, h, a_ij, base_state, shared_scratch, counters,
    newton_initial_guesses, newton_iteration_guesses, ...
)
```

**Target:**
```python
status = newton_solver_fn(
    stage_increment, parameters, drivers,
    t, h, a_ij, base_state, shared_scratch, counters,
    stage_index,  # Added
    newton_initial_guesses, newton_iteration_guesses, ...
)
```

## Expected Interactions Between Components

### Compile-Time
1. Test creates LinearSolverConfig
2. Test creates InstrumentedLinearSolver(config)
3. Test creates NewtonKrylovConfig with linear_solver=instrumented_instance
4. Test creates InstrumentedNewtonKrylov(config)
5. Test accesses .device_function on Newton-Krylov
6. Newton-Krylov.build() accesses linear_solver.device_function
7. Linear solver.build() compiles and caches device function
8. Newton-Krylov.build() compiles with embedded linear solver function
9. Both cache their compiled functions

### Runtime
1. Algorithm calls Newton-Krylov device function with logging arrays
2. Newton-Krylov records initial state
3. Newton-Krylov enters iteration loop
4. Newton-Krylov calls embedded linear solver with logging arrays
5. Linear solver records its iterations in provided arrays
6. Newton-Krylov continues with backtracking
7. Newton-Krylov records final iteration state
8. Newton-Krylov returns status
9. Host code validates logged data

## Data Structures

### Logging Array Shapes

**Linear Solver:**
- linear_initial_guesses: (num_slots, n) - dtype: precision
- linear_iteration_guesses: (num_slots, max_linear_iters, n) - dtype: precision
- linear_residuals: (num_slots, max_linear_iters, n) - dtype: precision
- linear_squared_norms: (num_slots, max_linear_iters) - dtype: precision
- linear_preconditioned_vectors: (num_slots, max_linear_iters, n) - dtype: precision

**Newton-Krylov:**
- newton_initial_guesses: (num_stages, n) - dtype: precision
- newton_iteration_guesses: (num_stages, max_newton_iters, n) - dtype: precision
- newton_residuals: (num_stages, max_newton_iters, n) - dtype: precision
- newton_squared_norms: (num_stages, max_newton_iters) - dtype: precision
- newton_iteration_scale: (num_stages, max_newton_iters) - dtype: precision

**Slot Index Calculation:**
- Linear solver in standalone use: slot_index = stage_index
- Linear solver in Newton-Krylov: slot_index = stage_index * max_newton_iters + newton_iter_index

## Edge Cases

### Edge Case 1: Linear Solver Fails Immediately
**Behavior:** Should still record initial guess, return status 4

### Edge Case 2: Newton-Krylov Fails in First Linear Solve
**Behavior:** Should record Newton initial state, propagate linear solver error status

### Edge Case 3: Backtracking Fails Without Finding Step
**Behavior:** Should revert to stage_base_bt, set backtrack_failed flag, still record iteration scale

### Edge Case 4: Convergence on First Iteration
**Behavior:** Should still record initial guess and first iteration state with converged status

### Edge Case 5: Multiple Algorithms Sharing Solver Instances
**Behavior:** Cache prevents recompilation, device functions safe to share across algorithms

## Dependencies

### Internal Dependencies
- cubie.CUDAFactory (base class)
- cubie.integrators.matrix_free_solvers.linear_solver (LinearSolver, LinearSolverConfig)
- cubie.integrators.matrix_free_solvers.newton_krylov (NewtonKrylov, NewtonKrylovConfig)
- cubie.buffer_registry (buffer allocation)
- cubie._utils (validators, types)
- cubie.cuda_simsafe (CUDA helpers)

### Test Dependencies
- tests/integrators/algorithms/instrumented/conftest.py (fixture definitions)
- tests/integrators/algorithms/instrumented/<algorithm>.py (instrumented algorithms)
- tests/integrators/algorithms/instrumented/_utils.py (instrumentation helpers)

### External Dependencies
- numba.cuda (device function compilation)
- numpy (array operations, dtypes)
- attrs (class definitions)

## Validation Criteria

### Functional Validation
- All instrumented tests pass with new implementation
- Logged data matches existing factory function output
- Device function signatures accepted by calling code
- Cache invalidation works correctly with config changes

### Structural Validation
- InstrumentedLinearSolver isinstance(LinearSolver) → True
- InstrumentedNewtonKrylov isinstance(NewtonKrylov) → True
- Cache classes inherit from CUDAFunctionCache
- Device functions pass is_device_validator

### Integration Validation
- Test fixtures create solver instances without errors
- Instrumented algorithms compile and execute
- Logging arrays populated with expected data
- No changes to logging array shapes or indexing patterns

## Notes for Implementer

### Copy-Paste Sources
- Production device function logic in:
  - `src/cubie/integrators/matrix_free_solvers/linear_solver.py` (lines 250-520)
  - `src/cubie/integrators/matrix_free_solvers/newton_krylov.py` (lines 290-480)
- Current instrumented logic in:
  - `tests/integrators/algorithms/instrumented/matrix_free_solvers.py` (entire file)

**Strategy:**
1. Copy production build() method structure
2. Inject logging statements from current instrumented factories
3. Preserve all production logic (convergence checks, warp votes, etc.)
4. Add logging parameters to signature
5. Ensure cuda.local.array() used for production buffers (preconditioned_vec, temp) via allocators
6. Keep logging arrays as parameters (not allocated internally)

### Common Pitfalls
- Don't call build() directly, always use .device_function property
- Don't forget to pass cached_aux in cached variant calls
- Maintain exact parameter order from production signatures
- Use selp() for predicated updates to avoid divergence
- Ensure linear solver slot indexing matches Newton-Krylov expectations
- Remember to validate linear_solver is InstrumentedLinearSolver type in Newton-Krylov

### Testing Strategy
1. First implement and test InstrumentedLinearSolver standalone
2. Then implement InstrumentedNewtonKrylov with embedded linear solver
3. Update conftest.py fixtures
4. Update one instrumented algorithm (e.g., backwards_euler)
5. Run tests incrementally
6. Update remaining instrumented algorithms
7. Validate all tests pass
