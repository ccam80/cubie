# Buffer Registry Migration: Agent Implementation Plan

This document provides the technical specification for the detailed_implementer and reviewer agents to complete Tasks 3-9 of the Buffer Registry Refactor.

---

## Current State Summary

### BufferRegistry Infrastructure (COMPLETE)
Located at: `src/cubie/buffer_registry.py`

Key components:
- `BufferEntry` - Immutable record with name, factory, size, location, persistent, aliases, precision
- `BufferContext` - Groups entries per factory with cached layouts
- `BufferRegistry` - Singleton with register(), get_allocator(), size properties
- `buffer_registry` - Module-level singleton instance

### Files Requiring Migration

#### Matrix-Free Solvers
1. `src/cubie/integrators/matrix_free_solvers/linear_solver.py`
   - Contains: LinearSolverBufferSettings, LinearSolverLocalSizes, LinearSolverSliceIndices
   - Functions: linear_solver_factory, linear_solver_cached_factory

2. `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`
   - Contains: NewtonBufferSettings, NewtonLocalSizes, NewtonSliceIndices
   - Functions: newton_krylov_solver_factory

#### Algorithm Files
3. `src/cubie/integrators/algorithms/generic_erk.py`
   - Contains: ERKBufferSettings, ERKLocalSizes, ERKSliceIndices
   - Class: ERKStep

4. `src/cubie/integrators/algorithms/generic_dirk.py`
   - Contains: DIRKBufferSettings, DIRKLocalSizes, DIRKSliceIndices
   - Class: DIRKStep
   - Special: FSAL aliasing for increment_cache/rhs_cache

5. `src/cubie/integrators/algorithms/generic_firk.py`
   - Contains: FIRKBufferSettings, FIRKLocalSizes, FIRKSliceIndices
   - Class: FIRKStep

6. `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
   - Contains: RosenbrockBufferSettings, RosenbrockLocalSizes, RosenbrockSliceIndices
   - Class: RosenbrockWStep

#### Loop Files
7. `src/cubie/integrators/loops/ode_loop.py`
   - Contains: LoopBufferSettings, LoopLocalSizes, LoopSliceIndices
   - Class: IVPLoop

#### Files to Delete
8. `src/cubie/BufferSettings.py` - Delete entirely
9. `tests/test_buffer_settings.py` - Delete or merge relevant tests

---

## Task 3: Migrate Matrix-Free Solvers

### 3.1 linear_solver.py Migration

**Remove:**
```python
# Lines 21-123: LocalSizes, SliceIndices base classes and LinearSolver* classes
class LocalSizes: ...
class SliceIndices: ...
class LinearSolverLocalSizes(LocalSizes): ...
class LinearSolverSliceIndices(SliceIndices): ...
class LinearSolverBufferSettings: ...
```

**Add import:**
```python
from cubie.buffer_registry import buffer_registry
```

**Modify linear_solver_factory:**

Before:
```python
def linear_solver_factory(
    operator_apply: Callable,
    n: int,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
    precision: PrecisionDType = np.float64,
    buffer_settings: Optional[LinearSolverBufferSettings] = None,
) -> Callable:
    # ...
    if buffer_settings is None:
        buffer_settings = LinearSolverBufferSettings(n=n)
    # Then use buffer_settings.shared_indices, buffer_settings.local_sizes
```

After:
```python
def linear_solver_factory(
    operator_apply: Callable,
    n: int,
    factory: object,  # NEW: owning factory for registry
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
    precision: PrecisionDType = np.float64,
    preconditioned_vec_location: str = 'local',  # NEW: location param
    temp_location: str = 'local',  # NEW: location param
) -> Callable:
    # Register buffers
    buffer_registry.register(
        'lin_preconditioned_vec', factory, n, preconditioned_vec_location,
        precision=precision
    )
    buffer_registry.register(
        'lin_temp', factory, n, temp_location, precision=precision
    )
    
    # Get allocators
    alloc_precond = buffer_registry.get_allocator('lin_preconditioned_vec', factory)
    alloc_temp = buffer_registry.get_allocator('lin_temp', factory)
    
    # In device function:
    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def linear_solver(..., shared, persistent_local, ...):
        preconditioned_vec = alloc_precond(shared, persistent_local)
        temp = alloc_temp(shared, persistent_local)
        # ... rest of solver
```

**Important:** The `factory` parameter must be passed by the caller (newton_krylov or algorithm).

### 3.2 newton_krylov.py Migration

**Remove:**
```python
# Lines 21-165: NewtonLocalSizes, NewtonSliceIndices, NewtonBufferSettings
```

**Modify newton_krylov_solver_factory:**

Before:
```python
def newton_krylov_solver_factory(
    ...,
    buffer_settings: Optional[NewtonBufferSettings] = None,
) -> Callable:
    if buffer_settings is None:
        buffer_settings = NewtonBufferSettings(n=n)
```

After:
```python
def newton_krylov_solver_factory(
    residual_function: Callable,
    linear_solver: Callable,
    n: int,
    factory: object,  # NEW
    tolerance: float,
    max_iters: int,
    damping: float = 0.5,
    max_backtracks: int = 8,
    precision: PrecisionDType = np.float32,
    delta_location: str = 'shared',
    residual_location: str = 'shared',
    residual_temp_location: str = 'local',
    stage_base_bt_location: str = 'local',
) -> Callable:
    # Register Newton buffers
    buffer_registry.register('newton_delta', factory, n, delta_location, precision=precision)
    buffer_registry.register('newton_residual', factory, n, residual_location, precision=precision)
    buffer_registry.register('newton_residual_temp', factory, n, residual_temp_location, precision=precision)
    buffer_registry.register('newton_stage_base_bt', factory, n, stage_base_bt_location, precision=precision)
    
    # Get allocators
    alloc_delta = buffer_registry.get_allocator('newton_delta', factory)
    alloc_residual = buffer_registry.get_allocator('newton_residual', factory)
    alloc_residual_temp = buffer_registry.get_allocator('newton_residual_temp', factory)
    alloc_stage_base_bt = buffer_registry.get_allocator('newton_stage_base_bt', factory)
```

---

## Task 4: Migrate Algorithm Files

### 4.1 generic_erk.py Migration

**Remove:**
```python
# Lines 52-279: LocalSizes, SliceIndices, BufferSettings base classes
# ERKLocalSizes, ERKSliceIndices, ERKBufferSettings classes
# ALL_ERK_BUFFER_LOCATION_PARAMETERS constant
```

**Modify ERKStep.__init__:**

Before:
```python
def __init__(self, ..., stage_rhs_location=None, stage_accumulator_location=None):
    buffer_settings = ERKBufferSettings(n=n, stage_count=stage_count, ...)
    config = ERKStepConfig(..., buffer_settings=buffer_settings)
```

After:
```python
def __init__(self, ..., stage_rhs_location='local', stage_accumulator_location='local'):
    # Register buffers with registry
    accumulator_length = max(tableau.stage_count - 1, 0) * n
    
    buffer_registry.register(
        'erk_stage_rhs', self, n, stage_rhs_location, precision=precision
    )
    buffer_registry.register(
        'erk_stage_accumulator', self, accumulator_length, stage_accumulator_location,
        precision=precision
    )
    
    # stage_cache registration depends on aliasing
    if stage_rhs_location == 'shared':
        buffer_registry.register(
            'erk_stage_cache', self, n, 'shared',
            aliases='erk_stage_rhs', precision=precision
        )
    elif stage_accumulator_location == 'shared':
        buffer_registry.register(
            'erk_stage_cache', self, n, 'shared',
            aliases='erk_stage_accumulator', precision=precision
        )
    else:
        # persistent_local when no shared buffers
        buffer_registry.register(
            'erk_stage_cache', self, n, 'local',
            persistent=True, precision=precision
        )
    
    # Store locations for build()
    self._stage_rhs_location = stage_rhs_location
    self._stage_accumulator_location = stage_accumulator_location
```

**Modify ERKStep.build_step:**
```python
def build_step(self, ...):
    # Get allocators
    alloc_stage_rhs = buffer_registry.get_allocator('erk_stage_rhs', self)
    alloc_stage_accumulator = buffer_registry.get_allocator('erk_stage_accumulator', self)
    alloc_stage_cache = buffer_registry.get_allocator('erk_stage_cache', self)
    
    @cuda.jit(device=True, inline=True)
    def step(..., shared, persistent_local, ...):
        stage_rhs = alloc_stage_rhs(shared, persistent_local)
        stage_accumulator = alloc_stage_accumulator(shared, persistent_local)
        stage_cache = alloc_stage_cache(shared, persistent_local)
        # ... rest of step
```

**Update size properties:**
```python
@property
def shared_memory_required(self) -> int:
    return buffer_registry.shared_buffer_size(self)

@property
def persistent_local_required(self) -> int:
    return buffer_registry.persistent_local_buffer_size(self)
```

### 4.2 generic_dirk.py Migration (with Aliasing)

**Key difference:** DIRK uses aliasing for FSAL optimization.

**Buffer registration pattern:**
```python
def __init__(self, ...):
    # Calculate solver requirements
    solver_shared_size = self._calculate_solver_shared_size(...)
    
    # Register solver_scratch as parent for aliasing
    buffer_registry.register(
        'dirk_solver_scratch', self, solver_shared_size, 'shared',
        precision=precision
    )
    
    # Register FSAL caches as aliases of solver_scratch
    buffer_registry.register(
        'dirk_increment_cache', self, n, 'shared',
        aliases='dirk_solver_scratch', persistent=True, precision=precision
    )
    buffer_registry.register(
        'dirk_rhs_cache', self, n, 'shared',
        aliases='dirk_solver_scratch', persistent=True, precision=precision
    )
    
    # Register algorithm buffers
    buffer_registry.register(
        'dirk_stage_increment', self, n, stage_increment_location,
        precision=precision
    )
    buffer_registry.register(
        'dirk_stage_base', self, n, stage_base_location,
        precision=precision
    )
    buffer_registry.register(
        'dirk_accumulator', self, accumulator_length, accumulator_location,
        precision=precision
    )
```

---

## Task 5: Migrate Loop Files

### 5.1 ode_loop.py Migration

**Remove:**
```python
# Lines 27-43: LocalSizes, SliceIndices, BufferSettings base classes
# LoopLocalSizes, LoopSliceIndices, LoopBufferSettings classes
# ALL_BUFFER_LOCATION_PARAMETERS constant
```

**Modify IVPLoop.__init__:**
```python
def __init__(self, ...):
    # Register all loop buffers
    buffer_registry.register('loop_state', self, n_states, state_location, precision=precision)
    buffer_registry.register('loop_proposed_state', self, n_states, state_proposal_location, precision=precision)
    buffer_registry.register('loop_parameters', self, n_parameters, parameters_location, precision=precision)
    buffer_registry.register('loop_drivers', self, n_drivers, drivers_location, precision=precision)
    buffer_registry.register('loop_proposed_drivers', self, n_drivers, drivers_proposal_location, precision=precision)
    buffer_registry.register('loop_observables', self, n_observables, observables_location, precision=precision)
    buffer_registry.register('loop_proposed_observables', self, n_observables, observables_proposal_location, precision=precision)
    buffer_registry.register('loop_error', self, n_error, error_location, precision=precision)
    buffer_registry.register('loop_counters', self, n_counters, counters_location, precision=precision)
    buffer_registry.register('loop_state_summary', self, state_summary_height, state_summary_location, precision=precision)
    buffer_registry.register('loop_observable_summary', self, observable_summary_height, observable_summary_location, precision=precision)
```

---

## Task 6: Update Batch Solving

### 6.1 SingleIntegratorRun.py

This file coordinates algorithm and loop. Update memory size properties:

```python
@property
def shared_memory_bytes(self) -> int:
    # Query loop and algorithm from registry
    loop_shared = buffer_registry.shared_buffer_size(self.loop)
    algo_shared = buffer_registry.shared_buffer_size(self.algorithm)
    return (loop_shared + algo_shared) * self.precision_size
```

---

## Task 7: Instrumented Tests

The instrumented test files import from source modules:
```python
from cubie.integrators.algorithms.generic_dirk import DIRKStep, DIRKBufferSettings
```

After migration, these imports will fail because DIRKBufferSettings no longer exists.

**Action:** Update instrumented tests to not import *BufferSettings classes. The instrumented tests only use the Step classes and device functions.

---

## Task 8: Delete Old Files

### 8.1 Delete BufferSettings.py
```bash
rm src/cubie/BufferSettings.py
```

### 8.2 Delete test_buffer_settings.py
```bash
rm tests/test_buffer_settings.py
```

Or merge useful tests into test_buffer_registry.py.

---

## Task 9: Integration Testing

Run full test suite:
```bash
NUMBA_ENABLE_CUDASIM=1 pytest -m "not nocudasim and not cupy"
```

Expected failures to fix:
- Any test importing *BufferSettings classes
- Any test using old buffer_settings parameter

---

## Edge Cases

### 1. Factory Cleanup
When a factory is garbage collected, its buffers remain in registry. Consider:
```python
def __del__(self):
    buffer_registry.clear_factory(self)
```

### 2. Multiple Solves
If a factory is used for multiple solves, buffers are already registered. Options:
- Clear and re-register
- Skip registration if already exists
- Use `update_buffer()` for changes

**Recommended:** Clear factory in `__init__` before registration:
```python
buffer_registry.clear_factory(self)
```

### 3. Thread Safety
The registry is not thread-safe. For concurrent access, consider adding locks.

### 4. Precision Mismatch
Ensure precision parameter matches across all buffer registrations for a factory.

---

## Dependency Order

```
Task 3 (Matrix-Free Solvers)
    ↓ (newton depends on linear)
Task 4 (Algorithm Files)
    ↓ (DIRK uses newton, FIRK uses newton)
Task 5 (Loop Files)
    ↓
Task 6 (Batch Solving - update size queries)
    ↓
Task 7 (Instrumented Tests - update imports)
    ↓
Task 8 (Delete Old Files)
    ↓
Task 9 (Integration Testing)
```

---

## Validation Criteria

### For Each Migrated File
1. No import from `cubie.BufferSettings`
2. No local *BufferSettings, *LocalSizes, *SliceIndices classes
3. Uses `buffer_registry.register()` in `__init__`
4. Uses `buffer_registry.get_allocator()` in `build()`
5. Size properties query `buffer_registry.*_buffer_size(self)`
6. All existing tests pass

### For Complete Migration
1. `src/cubie/BufferSettings.py` does not exist
2. No file in `src/cubie/` imports BufferSettings
3. All tests in `tests/` pass
4. buffer_registry tests remain green
