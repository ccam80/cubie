# Agent Implementation Plan: Fix Instrumented Algorithms

## Task Group 1: Production backwards_euler.py Enhancement

### Task 1.1: Create BackwardsEulerStepConfig class
**File**: `src/cubie/integrators/algorithms/backwards_euler.py`

**Behavior**:
- Create new `BackwardsEulerStepConfig` attrs class extending `ImplicitStepConfig`
- Add `increment_cache_location` field with default 'local' and validator for ['local', 'shared']
- Pattern follows DIRKStepConfig which has stage_increment_location, stage_base_location, etc.

**Integration Points**:
- Import `attrs` at top of file
- Class defined after `ImplicitStepConfig` import
- Used by BackwardsEulerStep.__init__()

### Task 1.2: Update BackwardsEulerStep.__init__ to use config class
**File**: `src/cubie/integrators/algorithms/backwards_euler.py`

**Behavior**:
- Add `increment_cache_location: Optional[str] = None` parameter to __init__
- Build `BackwardsEulerStepConfig` instead of `ImplicitStepConfig`
- Conditionally include increment_cache_location if not None

**Expected Changes**:
```python
def __init__(
    self,
    precision: PrecisionDType,
    n: int,
    ...
    increment_cache_location: Optional[str] = None,  # NEW
) -> None:
```

### Task 1.3: Add register_buffers() method
**File**: `src/cubie/integrators/algorithms/backwards_euler.py`

**Behavior**:
- Add `register_buffers()` method following DIRK pattern
- Clear existing registrations with buffer_registry.clear_parent(self)
- Call get_child_allocators for solver
- Register 'increment_cache' buffer using config.increment_cache_location

**Pattern from DIRK**:
```python
def register_buffers(self) -> None:
    config = self.compile_settings
    buffer_registry.clear_parent(self)
    _ = buffer_registry.get_child_allocators(self, self.solver, name='solver')
    buffer_registry.register(
        'increment_cache', self, config.n, 
        config.increment_cache_location,
        aliases='solver_shared', persistent=True,
        precision=config.precision
    )
```

### Task 1.4: Call register_buffers() in __init__
**File**: `src/cubie/integrators/algorithms/backwards_euler.py`

**Behavior**:
- After super().__init__ call, call self.register_buffers()

### Task 1.5: Update build_step to use buffer_registry allocator
**File**: `src/cubie/integrators/algorithms/backwards_euler.py`

**Behavior**:
- Already calls get_child_allocators - this is correct
- Verify allocator usage matches the registered buffers

---

## Task Group 2: Revert Production generic_dirk.py

### Task 2.1: Verify production generic_dirk.py
**File**: `src/cubie/integrators/algorithms/generic_dirk.py`

**Behavior**:
- Only change from original: `self._newton_solver` → `self.solver` (line ~448)
- This change is CORRECT - should keep `self.solver`
- No other changes needed

---

## Task Group 3: Revert Production crank_nicolson.py

### Task 3.1: Revert alias name in register_buffers
**File**: `src/cubie/integrators/algorithms/crank_nicolson.py`

**Behavior**:
- In register_buffers(), the alias should be 'solver_shared' (current is correct)
- Verify current state matches expected

### Task 3.2: Verify get_child_allocators name
**File**: `src/cubie/integrators/algorithms/crank_nicolson.py`

**Behavior**:
- Name should be 'solver' (current is correct)
- No changes needed if already correct

---

## Task Group 4: Revert Production generic_rosenbrock_w.py

### Task 4.1: Verify build_step signature
**File**: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

**Behavior**:
- `driver_del_t` should come from `config.driver_del_t`, not function parameter
- Current implementation is correct (accessing from config)
- Verify build_step signature matches parent class

---

## Task Group 5: Fix Instrumented backwards_euler.py

### Task 5.1: Fix __init__ - remove duplicate super() call
**File**: `tests/integrators/algorithms/instrumented/backwards_euler.py`

**Behavior**:
- Current file has two super().__init__ calls - remove duplicate
- Keep only ONE call to super().__init__

### Task 5.2: Remove nested build_implicit_helpers definition
**File**: `tests/integrators/algorithms/instrumented/backwards_euler.py`

**Behavior**:
- Current file has `def build_implicit_helpers` INSIDE __init__ - this is wrong
- Move to class level as instance method
- Fix indentation

### Task 5.3: Fix build_implicit_helpers to create instrumented solver
**File**: `tests/integrators/algorithms/instrumented/backwards_euler.py`

**Behavior**:
- Create InstrumentedLinearSolver instance
- Create InstrumentedNewtonKrylov instance 
- Set self.solver = newton_instance
- Update compile_settings with solver_function

**Pattern**:
```python
def build_implicit_helpers(self) -> Callable:
    # Get base class helper functions
    config = self.compile_settings
    beta, gamma, mass = config.beta, config.gamma, config.M
    preconditioner_order = config.preconditioner_order
    n = config.n
    precision = config.precision
    
    get_fn = config.get_solver_helper_fn
    
    preconditioner = get_fn('neumann_preconditioner', ...)
    residual = get_fn('stage_residual', ...)
    operator = get_fn('linear_operator', ...)
    
    # Create instrumented solvers
    linear_solver = InstrumentedLinearSolver(...)
    linear_solver.update(
        operator_apply=operator,
        preconditioner=preconditioner,
    )
    
    newton = InstrumentedNewtonKrylov(...)
    newton.update(residual_function=residual, ...)
    
    # Replace parent solver
    self.solver = newton
    
    self.update_compile_settings(solver_function=self.solver.device_function)
```

### Task 5.4: Fix build_step to pass logging arrays
**File**: `tests/integrators/algorithms/instrumented/backwards_euler.py`

**Behavior**:
- Solver call must include logging array parameters
- Use stage_index (int32(0) for single-stage methods)
- Pass all newton_* and linear_* logging arrays

---

## Task Group 6: Fix Instrumented crank_nicolson.py

### Task 6.1: Add build_implicit_helpers method
**File**: `tests/integrators/algorithms/instrumented/crank_nicolson.py`

**Behavior**:
- Override build_implicit_helpers to create instrumented Newton solver
- Pattern same as backwards_euler

### Task 6.2: Fix solver calls in step to pass logging arrays
**File**: `tests/integrators/algorithms/instrumented/crank_nicolson.py`

**Behavior**:
- Both solver calls (CN and BE steps) need logging arrays
- Use different stage_index for each call (0 and 1)

---

## Task Group 7: Fix Instrumented generic_dirk.py

### Task 7.1: Add build_implicit_helpers method
**File**: `tests/integrators/algorithms/instrumented/generic_dirk.py`

**Behavior**:
- Override to create InstrumentedNewtonKrylov
- Pattern follows backwards_euler

### Task 7.2: Fix solver calls in step to pass logging arrays
**File**: `tests/integrators/algorithms/instrumented/generic_dirk.py`

**Behavior**:
- Each stage's nonlinear solver call needs logging arrays
- Pass stage_idx as slot index

---

## Task Group 8: Fix Instrumented generic_firk.py

### Task 8.1: Add build_implicit_helpers method
**File**: `tests/integrators/algorithms/instrumented/generic_firk.py`

**Behavior**:
- Override to create InstrumentedNewtonKrylov
- Pattern follows backwards_euler

### Task 8.2: Fix solver call in step to pass logging arrays
**File**: `tests/integrators/algorithms/instrumented/generic_firk.py`

**Behavior**:
- Single solver call for all stages
- Use stage_index 0

---

## Task Group 9: Fix Instrumented generic_rosenbrock_w.py

### Task 9.1: Add build_implicit_helpers method
**File**: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`

**Behavior**:
- Override to create InstrumentedLinearSolver (not Newton - Rosenbrock uses linear only)
- Pattern adapted for linear-only solver

### Task 9.2: Fix linear solver calls in step to pass logging arrays
**File**: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`

**Behavior**:
- Each stage's linear_solver call needs logging arrays
- Pass stage_idx as slot index

---

## Dependency Graph

```
Task Group 1 (backwards_euler production) → Task Group 5 (backwards_euler instrumented)
Task Group 2 (dirk production - no changes) → Task Group 7 (dirk instrumented)
Task Group 3 (crank_nicolson production) → Task Group 6 (crank_nicolson instrumented)
Task Group 4 (rosenbrock production) → Task Group 9 (rosenbrock instrumented)
Task Group 8 (firk instrumented) can be done independently
```

## Edge Cases

1. **Single-stage vs multi-stage algorithms**: Single-stage (BE, CN) always use stage_index=0. Multi-stage (DIRK, FIRK, Rosenbrock) use loop index.

2. **Newton vs Linear solvers**: Rosenbrock uses LinearSolver only (no Newton iteration). Others use NewtonKrylov which contains LinearSolver.

3. **Logging array dimensions**: Arrays sized for max_stages * max_iterations. Unused slots remain zero-initialized.
