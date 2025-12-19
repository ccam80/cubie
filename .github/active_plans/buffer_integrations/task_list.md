# Implementation Task List
# Feature: Buffer Settings Integrations (Task 3 of 3)
# Plan Reference: .github/active_plans/buffer_integrations/agent_plan.md

## Overview

This task list covers the integration of buffer settings across step algorithms, matrix-free solvers, and the integration loop. The primary goals are:

1. **Remove factory parameter from solver factories** - Newton-Krylov and linear solver factories should use `cuda.local.array` for buffers instead of registering with the buffer registry
2. **Fix solver_scratch hard-coding** - Remove incorrect hard-coded solver_scratch slices from step functions  
3. **Clean up instrumented test files** - Mirror source changes in instrumented versions

### Scope Boundaries

**DO NOT MODIFY (handled by Tasks 1 and 2):**
- `src/cubie/buffer_registry.py` - Task 1
- `src/cubie/batchsolving/solver.py` - Task 2
- `src/cubie/integrators/loops/ode_loop_config.py` - Task 2

---

## Task Group 1: Matrix-Free Solvers - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (entire file)

**Input Validation Required**:
- None - no changes to validation logic

**Tasks**:

### 1.1 Update linear_solver.py - Remove factory parameter and buffer registration

**File**: `src/cubie/integrators/matrix_free_solvers/linear_solver.py`
**Action**: Modify

**Changes to `linear_solver_factory` (lines 19-30):**

Remove `factory` parameter and buffer location parameters from signature:

```python
def linear_solver_factory(
    operator_apply: Callable,
    n: int,
    precision: PrecisionDType,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
) -> Callable:
```

Remove these lines (86-100):
```python
    # Register buffers with central registry
    buffer_registry.register(
        'lin_preconditioned_vec', factory, n, preconditioned_vec_location,
        precision=precision
    )
    buffer_registry.register(
        'lin_temp', factory, n, temp_location, precision=precision
    )

    # Get allocators from registry
    alloc_precond = buffer_registry.get_allocator(
        'lin_preconditioned_vec', factory
    )
    alloc_temp = buffer_registry.get_allocator('lin_temp', factory)
```

Remove the import line:
```python
from cubie.buffer_registry import buffer_registry
```

Modify buffer allocation inside `linear_solver` device function (lines 165-167):
Replace:
```python
        # Allocate buffers from registry
        preconditioned_vec = alloc_precond(shared, shared)
        temp = alloc_temp(shared, shared)
```
With:
```python
        # Allocate buffers in thread-local memory
        preconditioned_vec = cuda.local.array(n_val, precision_numba)
        temp = cuda.local.array(n_val, precision_numba)
```

Remove the `shared` parameter from the device function signature (line 119):
Change:
```python
        shared,
```
To: (remove the line entirely since shared is no longer used)

Wait - actually the shared parameter must stay for signature compatibility with Newton caller. Instead, just don't use it internally.

**Changes to `linear_solver_cached_factory` (lines 255-266):**

Remove `factory` parameter from signature:
```python
def linear_solver_cached_factory(
    operator_apply: Callable,
    n: int,
    precision: PrecisionDType,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
) -> Callable:
```

Remove buffer registration and allocator retrieval (lines 315-328):
```python
    # Register buffers with central registry
    buffer_registry.register(
        'lin_cached_preconditioned_vec', factory, n, preconditioned_vec_location,
        precision=precision
    )
    buffer_registry.register(
        'lin_cached_temp', factory, n, temp_location, precision=precision
    )

    # Get allocators from registry
    alloc_precond = buffer_registry.get_allocator(
        'lin_cached_preconditioned_vec', factory
    )
    alloc_temp = buffer_registry.get_allocator('lin_cached_temp', factory)
```

Modify buffer allocation inside `linear_solver_cached` device function (lines 351-352):
Replace:
```python
        # Allocate buffers from registry
        preconditioned_vec = alloc_precond(shared, shared)
        temp = alloc_temp(shared, shared)
```
With:
```python
        # Allocate buffers in thread-local memory
        preconditioned_vec = cuda.local.array(n_val, precision_scalar)
        temp = cuda.local.array(n_val, precision_scalar)
```

**Edge cases**: 
- Ensure n_val is used as the array size (already defined as `n_val = int32(n)`)
- The `shared` parameter in the device function remains for signature compatibility but is unused

**Integration**: 
- Newton-Krylov solver calls linear_solver; it passes shared_scratch but linear solver ignores it

---

### 1.2 Update newton_krylov.py - Remove factory parameter and buffer registration

**File**: `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`
**Action**: Modify

**Changes to `newton_krylov_solver_factory` signature (lines 18-32):**

Remove `factory` parameter and all location parameters:
```python
def newton_krylov_solver_factory(
    residual_function: Callable,
    linear_solver: Callable,
    n: int,
    tolerance: float,
    max_iters: int,
    precision: PrecisionDType,
    damping: float = 0.5,
    max_backtracks: int = 8,
) -> Callable:
```

Remove the import:
```python
from cubie.buffer_registry import buffer_registry
```

Remove buffer registration (lines 91-104):
```python
    # Register buffers with central registry
    buffer_registry.register(
        'newton_delta', factory, n, delta_location, precision=precision
    )
    buffer_registry.register(
        'newton_residual', factory, n, residual_location, precision=precision
    )
    buffer_registry.register(
        'newton_residual_temp', factory, n, residual_temp_location,
        precision=precision
    )
    buffer_registry.register(
        'newton_stage_base_bt', factory, n, stage_base_bt_location,
        precision=precision
    )
```

Remove allocator retrieval (lines 106-114):
```python
    # Get allocators from registry
    alloc_delta = buffer_registry.get_allocator('newton_delta', factory)
    alloc_residual = buffer_registry.get_allocator('newton_residual', factory)
    alloc_residual_temp = buffer_registry.get_allocator(
        'newton_residual_temp', factory
    )
    alloc_stage_base_bt = buffer_registry.get_allocator(
        'newton_stage_base_bt', factory
    )
```

Modify buffer allocation inside `newton_krylov_solver` device function (lines 181-184):
Replace:
```python
        # Allocate buffers from registry
        delta = alloc_delta(shared_scratch, shared_scratch)
        residual = alloc_residual(shared_scratch, shared_scratch)
        residual_temp = alloc_residual_temp(shared_scratch, shared_scratch)
        stage_base_bt = alloc_stage_base_bt(shared_scratch, shared_scratch)
```
With:
```python
        # Allocate buffers in thread-local memory
        delta = cuda.local.array(n_val, numba_precision)
        residual = cuda.local.array(n_val, numba_precision)
        residual_temp = cuda.local.array(n_val, numba_precision)
        stage_base_bt = cuda.local.array(n_val, numba_precision)
```

Remove the linear solver slice calculation (lines 228-230):
```python
            # TODO: AI Error
            # Linear solver uses remaining shared space after Newton buffers
            lin_start = buffer_registry.shared_buffer_size(factory)
            lin_shared = shared_scratch[lin_start:]
```
Replace with:
```python
            # Linear solver uses its own local buffers; pass shared_scratch for compatibility
            lin_shared = shared_scratch
```

**Edge cases**:
- The `shared_scratch` parameter remains in the device function signature for compatibility with callers
- Linear solver now uses its own `cuda.local.array` buffers, so the slice passed doesn't matter

**Integration**:
- Step algorithms call newton_krylov_solver_factory; they no longer pass `factory=self`
- The shared_scratch passed to Newton is passed through to linear solver but neither uses it for their internal buffers

---

**Outcomes**: 
[ ] linear_solver_factory no longer requires factory parameter
[ ] linear_solver_cached_factory no longer requires factory parameter
[ ] newton_krylov_solver_factory no longer requires factory parameter
[ ] All solver buffers use cuda.local.array instead of buffer_registry

---

## Task Group 2: Base Algorithm Step and Implicit Step Updates - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 226-293)

**Input Validation Required**:
- None - no changes to validation logic

**Tasks**:

### 2.1 Update ode_implicitstep.py - Remove factory parameter from solver factory calls

**File**: `src/cubie/integrators/algorithms/ode_implicitstep.py`
**Action**: Modify

**Changes to `build_implicit_helpers` method (lines 226-293):**

Modify the `linear_solver_factory` call (lines 270-276):
Replace:
```python
        linear_solver = linear_solver_factory(operator,
                                              n=n,
                                              precision=self.precision,
                                              preconditioner=preconditioner,
                                              correction_type=correction_type,
                                              tolerance=krylov_tolerance,
                                              max_iters=max_linear_iters)
```
With:
```python
        linear_solver = linear_solver_factory(
            operator,
            n=n,
            precision=self.precision,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
        )
```
(No actual change needed - factory parameter was never passed here)

Modify the `newton_krylov_solver_factory` call (lines 283-292):
Replace:
```python
        nonlinear_solver = newton_krylov_solver_factory(
            residual_function=residual,
            linear_solver=linear_solver,
            n=n,
            tolerance=newton_tolerance,
            max_iters=max_newton_iters,
            damping=newton_damping,
            max_backtracks=newton_max_backtracks,
            precision=self.precision,
        )
```
With: (no change needed - factory parameter was never passed here)

**Edge cases**: None
**Integration**: This is the base class; subclasses override this method

---

**Outcomes**:
[ ] ODEImplicitStep.build_implicit_helpers confirmed to not pass factory parameter

---

## Task Group 3: Simple Implicit Step Algorithms - PARALLEL
**Status**: [ ]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler.py (lines 102-260)
- File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py (lines 14-183)
- File: src/cubie/integrators/algorithms/crank_nicolson.py (lines 115-314)

**Input Validation Required**:
- None - no changes to validation logic

**Tasks**:

### 3.1 Update backwards_euler.py - No changes required

**File**: `src/cubie/integrators/algorithms/backwards_euler.py`
**Action**: Review (no changes needed)

The current implementation at line 221:
```python
            solver_scratch = shared[: solver_shared_elements]
```

This is correct behavior - the solver_shared_elements comes from `self.solver_shared_elements` which returns `n * 2` (from ODEImplicitStep.solver_shared_elements property at line 296-299).

The algorithm correctly:
1. Uses `solver_shared_elements = self.solver_shared_elements` at line 140
2. Slices shared memory for solver: `solver_scratch = shared[: solver_shared_elements]`
3. Passes the slice to the Newton solver

Since Newton now uses `cuda.local.array` for its internal buffers, the shared_scratch passed to it is effectively unused for buffer allocation. The slice is still valid and the code works correctly.

**No changes required for backwards_euler.py**

---

### 3.2 Update backwards_euler_predict_correct.py - No changes required

**File**: `src/cubie/integrators/algorithms/backwards_euler_predict_correct.py`
**Action**: Review (no changes needed)

The current implementation at line 155:
```python
            solver_scratch = shared[: solver_shared_elements]
```

Same pattern as backwards_euler.py - the implementation is correct.

**No changes required for backwards_euler_predict_correct.py**

---

### 3.3 Update crank_nicolson.py - No changes required

**File**: `src/cubie/integrators/algorithms/crank_nicolson.py`
**Action**: Review (no changes needed)

The current implementation at line 240:
```python
            solver_scratch = shared[:solver_shared_elements]
```

Same pattern as backwards_euler.py - the implementation is correct.

**No changes required for crank_nicolson.py**

---

**Outcomes**:
[ ] backwards_euler.py reviewed - no changes needed
[ ] backwards_euler_predict_correct.py reviewed - no changes needed
[ ] crank_nicolson.py reviewed - no changes needed

---

## Task Group 4: Complex Implicit Step Algorithms - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 299-370, 424-443)
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 291-368, 416-428)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 276-344, 424-436)

**Input Validation Required**:
- None - no changes to validation logic

**Tasks**:

### 4.1 Update generic_dirk.py - Remove factory parameter from solver calls

**File**: `src/cubie/integrators/algorithms/generic_dirk.py`
**Action**: Modify

**Changes to `build_implicit_helpers` method (lines 299-370):**

Modify the `linear_solver_factory` call (lines 342-351):
Replace:
```python
        linear_solver = linear_solver_factory(
            operator,
            n=n,
            precision=precision,
            factory=self,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
        )
```
With:
```python
        linear_solver = linear_solver_factory(
            operator,
            n=n,
            precision=precision,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
        )
```

Modify the `newton_krylov_solver_factory` call (lines 358-368):
Replace:
```python
        nonlinear_solver = newton_krylov_solver_factory(
            residual_function=residual,
            linear_solver=linear_solver,
            n=n,
            factory=self,
            tolerance=newton_tolerance,
            max_iters=max_newton_iters,
            damping=newton_damping,
            max_backtracks=newton_max_backtracks,
            precision=precision,
        )
```
With:
```python
        nonlinear_solver = newton_krylov_solver_factory(
            residual_function=residual,
            linear_solver=linear_solver,
            n=n,
            tolerance=newton_tolerance,
            max_iters=max_newton_iters,
            damping=newton_damping,
            max_backtracks=newton_max_backtracks,
            precision=precision,
        )
```

**Edge cases**: None
**Integration**: DIRK builds its own solver chain; factory parameter removal is straightforward

---

### 4.2 Update generic_firk.py - Remove factory parameter from solver calls

**File**: `src/cubie/integrators/algorithms/generic_firk.py`
**Action**: Modify

**Changes to `build_implicit_helpers` method (lines 291-368):**

Modify the `linear_solver_factory` call (lines 341-350):
Replace:
```python
        linear_solver = linear_solver_factory(
            operator,
            n=all_stages_n,
            precision=precision,
            factory=self,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
        )
```
With:
```python
        linear_solver = linear_solver_factory(
            operator,
            n=all_stages_n,
            precision=precision,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
        )
```

Modify the `newton_krylov_solver_factory` call (lines 357-367):
Replace:
```python
        nonlinear_solver = newton_krylov_solver_factory(
            residual_function=residual,
            linear_solver=linear_solver,
            n=all_stages_n,
            factory=self,
            tolerance=newton_tolerance,
            max_iters=max_newton_iters,
            damping=newton_damping,
            max_backtracks=newton_max_backtracks,
            precision=precision,
        )
```
With:
```python
        nonlinear_solver = newton_krylov_solver_factory(
            residual_function=residual,
            linear_solver=linear_solver,
            n=all_stages_n,
            tolerance=newton_tolerance,
            max_iters=max_newton_iters,
            damping=newton_damping,
            max_backtracks=newton_max_backtracks,
            precision=precision,
        )
```

**Edge cases**: Note that FIRK uses `all_stages_n` (flattened dimension) instead of `n`
**Integration**: FIRK builds its own solver chain; factory parameter removal is straightforward

---

### 4.3 Update generic_rosenbrock_w.py - Remove factory parameter from solver call

**File**: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
**Action**: Modify

**Changes to `build_implicit_helpers` method (lines 276-344):**

Modify the `linear_solver_cached_factory` call (lines 327-336):
Replace:
```python
        linear_solver = linear_solver_cached_factory(
            linear_operator,
            precision=precision,
            n=n,
            factory=self,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
        )
```
With:
```python
        linear_solver = linear_solver_cached_factory(
            linear_operator,
            n=n,
            precision=precision,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
        )
```

Note: Rosenbrock uses `linear_solver_cached_factory` (not `linear_solver_factory`) and does not use Newton-Krylov.

**Edge cases**: Rosenbrock only uses the linear solver, not Newton-Krylov
**Integration**: Rosenbrock returns a tuple from build_implicit_helpers, not a single solver

---

**Outcomes**:
[ ] generic_dirk.py - factory parameter removed from solver calls
[ ] generic_firk.py - factory parameter removed from solver calls  
[ ] generic_rosenbrock_w.py - factory parameter removed from solver call

---

## Task Group 5: Instrumented Test Files - PARALLEL
**Status**: [ ]
**Dependencies**: Task Groups 1-4

**Required Context**:
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (entire file)
- File: tests/integrators/algorithms/instrumented/backwards_euler.py (lines 109-176)
- File: tests/integrators/algorithms/instrumented/backwards_euler_predict_correct.py (entire file)
- File: tests/integrators/algorithms/instrumented/crank_nicolson.py (entire file)
- File: tests/integrators/algorithms/instrumented/generic_dirk.py (entire file)
- File: tests/integrators/algorithms/instrumented/generic_firk.py (entire file)
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py (entire file)

**Input Validation Required**:
- None - instrumented files mirror source functionality

**Tasks**:

### 5.1 Update instrumented/matrix_free_solvers.py - Already correct

**File**: `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`
**Action**: Review (no changes needed)

The instrumented solvers already:
1. Do NOT have factory parameter in their signatures (line 11-19, 311-320)
2. Use `cuda.local.array` for buffers (lines 57-58, 201-202, 384-386)
3. Do NOT use buffer_registry

The instrumented versions are already aligned with the target architecture.

**No changes required for instrumented/matrix_free_solvers.py**

---

### 5.2 Update instrumented/backwards_euler.py - Already correct

**File**: `tests/integrators/algorithms/instrumented/backwards_euler.py`
**Action**: Review (no changes needed)

The instrumented version already:
1. Uses `inst_linear_solver_factory` and `inst_newton_krylov_solver_factory` without factory parameter (lines 150-175)
2. These instrumented factories already use local arrays

**No changes required for instrumented/backwards_euler.py**

---

### 5.3 Confirm instrumented/generic_dirk.py - Already correct

**File**: `tests/integrators/algorithms/instrumented/generic_dirk.py`
**Action**: Review (no changes needed)

The instrumented version at lines 223-238 already:
1. Uses `inst_linear_solver_factory` and `inst_newton_krylov_solver_factory` from instrumented module
2. Does NOT pass factory parameter

**No changes required for instrumented/generic_dirk.py**

---

### 5.4 Confirm instrumented/generic_firk.py - Already correct

**File**: `tests/integrators/algorithms/instrumented/generic_firk.py`
**Action**: Review (no changes needed)

The instrumented version at lines 221-245 already:
1. Uses `inst_linear_solver_factory` and `inst_newton_krylov_solver_factory` from instrumented module
2. Does NOT pass factory parameter

**No changes required for instrumented/generic_firk.py**

---

### 5.5 Confirm instrumented/generic_rosenbrock_w.py - Already correct

**File**: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
**Action**: Review (no changes needed)

The instrumented version at lines 212-220 already:
1. Uses `inst_linear_solver_cached_factory` from instrumented module
2. Does NOT pass factory parameter

**No changes required for instrumented/generic_rosenbrock_w.py**

---

**Outcomes**:
[ ] instrumented/matrix_free_solvers.py - confirmed no changes needed
[ ] instrumented/backwards_euler.py - confirmed no changes needed
[ ] instrumented/generic_dirk.py - confirmed no changes needed
[ ] instrumented/generic_firk.py - confirmed no changes needed
[ ] instrumented/generic_rosenbrock_w.py - confirmed no changes needed

---

## Task Group 6: Verification - SEQUENTIAL
**Status**: [ ]
**Dependencies**: All previous groups

**Required Context**:
- All modified files

**Tasks**:

### 6.1 Verify ode_loop.py - No changes needed

**File**: `src/cubie/integrators/loops/ode_loop.py`
**Action**: Review (no changes needed)

The loop correctly:
1. Computes `remaining_scratch_start = buffer_registry.shared_buffer_size(self)` (line 343)
2. Passes `remaining_shared_scratch = shared_scratch[remaining_scratch_start:]` to step function (line 457)

Since Newton and Linear solvers now use `cuda.local.array` for their internal buffers, the shared_scratch slice passed to them is effectively not used for those buffers. The algorithm's own buffers (if any are shared) still work correctly.

**No changes required for ode_loop.py**

---

### 6.2 Verify SingleIntegratorRunCore.py - No changes needed

**File**: `src/cubie/integrators/SingleIntegratorRunCore.py`
**Action**: Review (no changes needed)

The core correctly:
1. Passes `algorithm_local_elements=self._algo_step.persistent_local_required` (line 189-190)
2. The `persistent_local_required` property in BaseAlgorithmStep returns `solver_local_elements + algorithm_local_elements` (line 624-626)

Since solver buffers are now `cuda.local.array` (not persistent local), the `solver_local_elements` property in ODEImplicitStep returns 0 (line 302-305), which is correct.

**No changes required for SingleIntegratorRunCore.py**

---

### 6.3 Verify BatchSolverKernel.py - No changes needed

**File**: `src/cubie/batchsolving/BatchSolverKernel.py`
**Action**: Review (no changes needed)

The kernel queries memory sizes from SingleIntegratorRun which correctly aggregates from its children. No changes needed.

**No changes required for BatchSolverKernel.py**

---

**Outcomes**:
[ ] ode_loop.py - verified no changes needed
[ ] SingleIntegratorRunCore.py - verified no changes needed
[ ] BatchSolverKernel.py - verified no changes needed

---

## Implementation Order Summary

1. **Task Group 1**: Matrix-Free Solvers (linear_solver.py, newton_krylov.py)
   - Remove factory parameter and location parameters
   - Remove buffer_registry usage
   - Use cuda.local.array for buffers

2. **Task Group 2**: Base Implicit Step (ode_implicitstep.py)
   - Verify no factory parameter passed (already correct)

3. **Task Group 3**: Simple Implicit Algorithms (backwards_euler.py, etc.)
   - Review only - no changes needed

4. **Task Group 4**: Complex Implicit Algorithms (generic_dirk.py, generic_firk.py, generic_rosenbrock_w.py)
   - Remove factory parameter from solver factory calls

5. **Task Group 5**: Instrumented Test Files
   - Review and update if needed to match source changes

6. **Task Group 6**: Verification
   - Confirm ode_loop.py, SingleIntegratorRunCore.py, BatchSolverKernel.py need no changes

---

## Test Requirements

### Existing Tests Should Pass
After changes, all existing tests should pass since behavior is preserved.

### Key Test Files to Run
- `tests/integrators/algorithms/test_step_algorithms.py`
- `tests/integrators/algorithms/test_buffer_settings.py`
- `tests/integrators/algorithms/instrumented/test_instrumented.py`

---

## Estimated Complexity

- **Task Group 1**: Medium - Core solver changes
- **Task Group 2**: Low - Verification only
- **Task Group 3**: Low - No changes needed
- **Task Group 4**: Low - Parameter removal only
- **Task Group 5**: Low - Verification and possible small updates
- **Task Group 6**: Low - Verification only

**Total Estimated Effort**: 4-6 hours for implementation and testing
