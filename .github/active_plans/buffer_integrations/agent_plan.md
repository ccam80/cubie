# Buffer Integrations - Agent Plan

## Scope

This plan covers Task 3 of 3 - integrations for buffer settings across step algorithms, matrix-free solvers, and the integration loop.

**Files to Modify:**
- `src/cubie/integrators/algorithms/base_algorithm_step.py`
- `src/cubie/integrators/algorithms/ode_implicitstep.py`
- `src/cubie/integrators/algorithms/backwards_euler.py`
- `src/cubie/integrators/algorithms/backwards_euler_predict_correct.py`
- `src/cubie/integrators/algorithms/crank_nicolson.py`
- `src/cubie/integrators/algorithms/generic_dirk.py`
- `src/cubie/integrators/algorithms/generic_firk.py`
- `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
- `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`
- `src/cubie/integrators/matrix_free_solvers/linear_solver.py`
- `src/cubie/integrators/loops/ode_loop.py`
- `src/cubie/integrators/SingleIntegratorRunCore.py`
- `src/cubie/batchsolving/BatchSolverKernel.py`
- All corresponding instrumented test files

**Do NOT Modify:**
- `src/cubie/buffer_registry.py` - Task 1
- `src/cubie/batchsolving/solver.py` - Task 2
- `src/cubie/integrators/loops/ode_loop_config.py` - Task 2

---

## User Stories Reference

See `human_overview.md` for:
- US-1: Hierarchical solver memory management
- US-2: Remove factory parameter from solver factories
- US-3: Base step compile settings memory query methods
- US-4: Streamlined IVPLoop parameter management

---

## Component Specifications by File

### File 1: base_algorithm_step.py

**Purpose:** Add query methods for shared and persistent memory needs to base classes.

**Changes to BaseStepConfig:**

Add new methods that query the buffer registry for memory requirements:

```python
def get_solver_shared_elements(self, step_instance) -> int:
    """Query registry for shared elements required by solver chain.
    
    Parameters
    ----------
    step_instance
        The step algorithm instance owning the solver.
        
    Returns
    -------
    int
        Total shared memory elements required by solver hierarchy.
    """
    # Base config has no solver; subclasses override
    return 0

def get_solver_persistent_elements(self, step_instance) -> int:
    """Query registry for persistent local elements required by solver.
    
    Parameters
    ----------
    step_instance
        The step algorithm instance owning the solver.
        
    Returns
    -------
    int
        Total persistent local elements required by solver hierarchy.
    """
    # Base config has no solver; subclasses override
    return 0
```

**Changes to BaseAlgorithmStep:**

Add pass-through properties:

```python
def get_solver_shared_elements(self) -> int:
    """Return shared elements needed by solver infrastructure.
    
    Queries the compile settings for solver memory requirements.
    This method is called by parent components to determine
    how much shared memory to allocate for the solver chain.
    
    Returns
    -------
    int
        Shared memory elements for solver hierarchy.
    """
    return self.compile_settings.get_solver_shared_elements(self)

def get_solver_persistent_elements(self) -> int:
    """Return persistent local elements needed by solver infrastructure.
    
    Queries the compile settings for solver memory requirements.
    This method is called by parent components to determine
    how much persistent local memory to allocate.
    
    Returns
    -------
    int
        Persistent local elements for solver hierarchy.
    """
    return self.compile_settings.get_solver_persistent_elements(self)
```

**Update existing properties:**

Modify `solver_shared_elements` and `solver_local_elements` to use the new query methods:

```python
@property
def solver_shared_elements(self) -> int:
    """Return shared-memory elements consumed by solver infrastructure."""
    return self.get_solver_shared_elements()

@property
def solver_local_elements(self) -> int:
    """Return persistent local elements consumed by solver infrastructure."""
    return self.get_solver_persistent_elements()
```

---

### File 2: ode_implicitstep.py

**Purpose:** Update implicit step base to use hierarchical buffer pattern.

**Changes to ImplicitStepConfig:**

Override the query methods:

```python
def get_solver_shared_elements(self, step_instance) -> int:
    """Return shared elements for Newton-Krylov solver chain.
    
    The implicit solver chain requires:
    - Newton: 4 buffers of size n (delta, residual, residual_temp, stage_base_bt)
    - Linear: 2 buffers of size n (preconditioned_vec, temp)
    
    Total: 6n elements in shared memory (when shared location requested).
    """
    n = self.n
    # Newton buffers: delta, residual, residual_temp, stage_base_bt = 4n
    # Linear buffers: preconditioned_vec, temp = 2n
    # Default is local for all, so shared elements = 0 unless locations changed
    return 0  # Buffers default to local

def get_solver_persistent_elements(self, step_instance) -> int:
    """Return persistent local elements for Newton-Krylov solver chain."""
    return 0  # No persistent local by default
```

**Changes to ODEImplicitStep:**

1. **Remove `factory` parameter from solver factory calls**:

In `build_implicit_helpers`:

```python
def build_implicit_helpers(self) -> Callable:
    """Construct the nonlinear solver chain used by implicit methods."""
    
    config = self.compile_settings
    # ... existing helper function retrieval ...
    
    # Linear solver no longer receives factory
    linear_solver = linear_solver_factory(
        operator,
        n=n,
        precision=self.precision,
        preconditioner=preconditioner,
        correction_type=correction_type,
        tolerance=krylov_tolerance,
        max_iters=max_linear_iters,
        # NO factory parameter
    )
    
    # Newton solver no longer receives factory
    nonlinear_solver = newton_krylov_solver_factory(
        residual_function=residual,
        linear_solver=linear_solver,
        n=n,
        tolerance=newton_tolerance,
        max_iters=max_newton_iters,
        damping=newton_damping,
        max_backtracks=newton_max_backtracks,
        precision=self.precision,
        # NO factory parameter
    )
    return nonlinear_solver
```

2. **Update solver_shared_elements property**:

```python
@property
def solver_shared_elements(self) -> int:
    """Return shared scratch dedicated to the Newton--Krylov solver.
    
    Newton-Krylov requires 2n elements for delta and residual arrays.
    The linear solver uses additional 2n for preconditioned_vec and temp.
    Total: 4n when all buffers are shared.
    
    Note: Buffer locations are determined by compile settings. This
    property returns the size when shared; actual allocation may differ.
    """
    return self.compile_settings.n * 4  # Newton(2n) + Linear(2n)
```

---

### File 3: backwards_euler.py

**Purpose:** Use registry pattern and fix solver_scratch hard-coding.

**Remove buffer_registry imports if present** (buffers registered in parent).

**Changes to build_step:**

```python
def build_step(
    self,
    solver_fn: Callable,
    dxdt_fn: Callable,
    observables_function: Callable,
    driver_function: Optional[Callable],
    numba_precision: type,
    n: int,
    n_drivers: int,
) -> StepCache:
    """Build the device function for a backward Euler step."""
    
    a_ij = numba_precision(1.0)
    has_driver_function = driver_function is not None
    n = int32(n)
    typed_zero = numba_precision(0.0)
    
    # Get solver memory requirements for slicing
    solver_shared_start = self.algorithm_shared_elements
    solver_shared_size = self.solver_shared_elements

    @cuda.jit(device=True, inline=True)
    def step(
        state,
        proposed_state,
        parameters,
        driver_coefficients,
        drivers_buffer,
        proposed_drivers,
        observables,
        proposed_observables,
        error,
        dt_scalar,
        time_scalar,
        first_step_flag,
        accepted_flag,
        shared,
        persistent_local,
        counters,
    ):
        """Perform one backward Euler update."""
        
        # Algorithm has no shared buffers; entire shared goes to solver
        solver_shared = shared[solver_shared_start:solver_shared_start + solver_shared_size]
        
        # Initialize stage increment from solver shared (first n elements)
        for i in range(n):
            proposed_state[i] = solver_shared[i]

        next_time = time_scalar + dt_scalar
        if has_driver_function:
            driver_function(
                next_time,
                driver_coefficients,
                proposed_drivers,
            )

        status = solver_fn(
            proposed_state,
            parameters,
            proposed_drivers,
            next_time,
            dt_scalar,
            a_ij,
            state,
            solver_shared,
            counters,
        )

        for i in range(n):
            solver_shared[i] = proposed_state[i]
            proposed_state[i] += state[i]

        observables_function(
            proposed_state,
            parameters,
            proposed_drivers,
            proposed_observables,
            next_time,
        )

        return status

    return StepCache(step=step, nonlinear_solver=solver_fn)
```

**Key Changes:**
- Remove hard-coded `solver_scratch = shared[: solver_shared_elements]`
- Use `solver_shared_start` offset (which is 0 for backwards euler since `algorithm_shared_elements = 0`)
- Compute slice bounds using queried sizes

---

### File 4: backwards_euler_predict_correct.py

**Purpose:** Apply same pattern as backwards_euler.

**Changes:**
- Same solver_shared slicing pattern
- Remove any hard-coded solver_scratch slices
- Use `algorithm_shared_elements` offset for solver slice start

---

### File 5: crank_nicolson.py

**Purpose:** Apply same pattern as backwards_euler.

**Changes:**
- Same solver_shared slicing pattern  
- Remove any hard-coded solver_scratch slices
- Use `algorithm_shared_elements` offset for solver slice start

---

### File 6: generic_dirk.py

**Purpose:** Update to hierarchical pattern while preserving complex tableau logic.

**Current State:**
- Already uses buffer_registry for algorithm buffers
- Has `solver_scratch` buffer registered
- Calls solver factories with `factory=self` parameter

**Required Changes:**

1. **Remove factory parameter from solver factory calls** in `build_implicit_helpers`:

```python
linear_solver = linear_solver_factory(
    operator,
    n=n,
    precision=precision,
    # REMOVE: factory=self,
    preconditioner=preconditioner,
    correction_type=correction_type,
    tolerance=krylov_tolerance,
    max_iters=max_linear_iters,
)

nonlinear_solver = newton_krylov_solver_factory(
    residual_function=residual,
    linear_solver=linear_solver,
    n=n,
    # REMOVE: factory=self,
    tolerance=newton_tolerance,
    max_iters=max_newton_iters,
    damping=newton_damping,
    max_backtracks=newton_max_backtracks,
    precision=precision,
)
```

2. **Update solver_scratch registration** in `__init__`:

The solver_scratch buffer should be sized to include both Newton and Linear solver needs:

```python
# solver_scratch provides space for Newton + Linear chains
# Newton needs: delta(n) + residual(n) + residual_temp(n) + stage_base_bt(n) = 4n
# Linear needs: preconditioned_vec(n) + temp(n) = 2n
# Total: 6n (but Newton reuses Linear's space, so 4n minimum)
solver_shared_size = 4 * n  # Newton buffers; linear slices within
buffer_registry.register(
    'dirk_solver_scratch', self, solver_shared_size, 'shared',
    precision=precision
)
```

3. **Update build_step** to pass solver slice correctly:

The solver_scratch allocator already provides the buffer. Ensure the slice passed to Newton covers the full solver hierarchy.

---

### File 7: generic_firk.py

**Purpose:** Apply same pattern as generic_dirk.

**Changes:**
- Remove factory parameter from solver factory calls
- Update solver buffer sizing to cover full hierarchy
- Ensure solver receives properly-sized slice

---

### File 8: generic_rosenbrock_w.py

**Purpose:** Apply same pattern as generic_dirk.

**Changes:**
- Remove factory parameter from solver factory calls
- Update solver buffer sizing
- Ensure proper slice passing

---

### File 9: newton_krylov.py

**Purpose:** Remove factory parameter and buffer registration.

**Current State:**
- Receives `factory` parameter
- Registers buffers: newton_delta, newton_residual, newton_residual_temp, newton_stage_base_bt
- Gets allocators from registry
- Has hard-coded slice for linear solver: `lin_shared = shared_scratch[lin_start:]`

**Required Changes:**

1. **Remove factory parameter from signature:**

```python
def newton_krylov_solver_factory(
    residual_function: Callable,
    linear_solver: Callable,
    n: int,
    # REMOVE: factory: object,
    tolerance: float,
    max_iters: int,
    precision: PrecisionDType,
    damping: float = 0.5,
    max_backtracks: int = 8,
    # REMOVE all location parameters - buffers are local by default
) -> Callable:
```

2. **Remove buffer registration:**

Remove all `buffer_registry.register(...)` calls.

3. **Remove allocator retrieval:**

Remove all `buffer_registry.get_allocator(...)` calls.

4. **Use cuda.local.array for buffers:**

```python
# Inside newton_krylov_solver device function:
delta = cuda.local.array(n_val, precision)
residual = cuda.local.array(n_val, precision)
residual_temp = cuda.local.array(n_val, precision)
stage_base_bt = cuda.local.array(n_val, precision)
```

5. **Remove linear solver slice calculation:**

The line `lin_start = buffer_registry.shared_buffer_size(factory)` must be removed.
Linear solver receives the same shared_scratch; it manages its own local buffers.

```python
# Replace:
# lin_start = buffer_registry.shared_buffer_size(factory)
# lin_shared = shared_scratch[lin_start:]

# With:
# Linear solver uses its own cuda.local.array buffers
# Pass shared_scratch directly (for any shared buffers from algorithm)
```

6. **Update imports:**

Remove `from cubie.buffer_registry import buffer_registry`

---

### File 10: linear_solver.py

**Purpose:** Remove factory parameter and buffer registration.

**Changes to linear_solver_factory:**

1. **Remove factory parameter:**

```python
def linear_solver_factory(
    operator_apply: Callable,
    n: int,
    # REMOVE: factory: object,
    precision: PrecisionDType,
    preconditioner: Optional[Callable] = None,
    correction_type: str = "minimal_residual",
    tolerance: float = 1e-6,
    max_iters: int = 100,
    # REMOVE location parameters
) -> Callable:
```

2. **Remove buffer registration and allocator retrieval.**

3. **Use cuda.local.array:**

```python
# Inside linear_solver device function:
preconditioned_vec = cuda.local.array(n_val, precision)
temp = cuda.local.array(n_val, precision)
```

4. **Remove shared parameter handling:**

The function no longer allocates from shared; it uses local arrays.

**Changes to linear_solver_cached_factory:**

Apply same pattern - remove factory, use local arrays.

---

### File 11: ode_loop.py

**Purpose:** Ensure loop queries step for total memory needs.

**Current State:**
- IVPLoop receives `controller_local_len` and `algorithm_local_len` parameters
- These are passed from SingleIntegratorRunCore

**Changes:**

The loop already receives memory sizes from the parent. Verify that:

1. The `remaining_scratch_start` calculation is correct:

```python
# In build():
remaining_scratch_start = buffer_registry.shared_buffer_size(self)
```

This should correctly compute the offset after loop's own buffers.

2. Step function receives the remaining shared scratch:

```python
# In loop_fn:
remaining_shared_scratch = shared_scratch[remaining_scratch_start:]

# This slice goes to step function
step_status = int32(
    step_function(
        # ... other args ...
        remaining_shared_scratch,  # Shared scratch for algorithm + solver
        algo_local,  # Persistent local for algorithm
        proposed_counters,
    )
)
```

**No major changes needed** - the loop already delegates memory management to its children. Verify the slice calculation is correct.

---

### File 12: SingleIntegratorRunCore.py

**Purpose:** Manage buffer size queries as part of dependency injection.

**Current State:**
- Passes `controller_local_elements` and `algorithm_local_elements` to IVPLoop
- These come from step controller and algorithm step properties

**Verify Current Flow:**

```python
# In instantiate_loop:
loop_kwargs.update(
    # ...
    controller_local_len=controller_local_elements,
    algorithm_local_len=algorithm_local_elements,
    # ...
)
```

**Ensure properties are correct:**

The sizes should include solver memory. Verify:

```python
# algorithm_local_elements should include solver persistent local
algorithm_local_elements=self._algo_step.persistent_local_required
```

Where `persistent_local_required` is:

```python
@property
def persistent_local_required(self) -> int:
    """Return the persistent local precision-entry requirement."""
    return self.solver_local_elements + self.algorithm_local_elements
```

**No changes needed** if the property chain is correct. The solver_local_elements and algorithm_local_elements sum to the total.

---

### File 13: BatchSolverKernel.py

**Purpose:** Ensure memory sizes flow through correctly.

**Current State:**
- Queries `single_integrator.local_memory_elements` and `shared_memory_elements`
- These values are used for kernel configuration

**Verify:**

```python
initial_config = BatchSolverConfig(
    # ...
    local_memory_elements=(
        self.single_integrator.local_memory_elements
    ),
    shared_memory_elements=(
        self.single_integrator.shared_memory_elements
    ),
    # ...
)
```

**No changes needed** if SingleIntegratorRun correctly computes totals.

---

## Instrumented Files

### Pattern for All Instrumented Files

Each instrumented file in `tests/integrators/algorithms/instrumented/` must be updated to match the corresponding source file changes.

**Key Differences in Instrumented Versions:**
- Additional logging arrays in step function signatures
- Recording of intermediate values for debugging
- Same buffer management pattern as source files

### Instrumented Files to Update:

1. **instrumented/backwards_euler.py**
   - Update build_step to use new slicing pattern
   - Keep all logging array parameters
   - Mirror source solver_shared slice calculation

2. **instrumented/backwards_euler_predict_correct.py**
   - Same updates as backwards_euler

3. **instrumented/crank_nicolson.py**
   - Same updates as backwards_euler

4. **instrumented/generic_dirk.py**
   - Update solver factory calls (remove factory param)
   - Keep logging functionality

5. **instrumented/generic_firk.py**
   - Same as generic_dirk

6. **instrumented/generic_rosenbrock_w.py**
   - Same as generic_dirk

7. **instrumented/matrix_free_solvers.py**
   - Update `inst_newton_krylov_solver_factory` to remove factory parameter
   - Update `inst_linear_solver_factory` to remove factory parameter
   - Use cuda.local.array for buffers
   - Keep all logging functionality

---

## Edge Cases

### 1. Zero-Size Solver Buffers
Explicit algorithms (like ExplicitEulerStep) have no solver:
- `solver_shared_elements` returns 0
- `solver_local_elements` returns 0
- No slice calculation errors because slice is `shared[0:0]` which is valid

### 2. Mixed Buffer Locations
When algorithm buffers are shared but solver buffers are local:
- Algorithm registers shared buffers
- Solver uses cuda.local.array
- No conflict; different memory spaces

### 3. Cache Invalidation
When child factory (solver) is invalidated:
- Parent (algorithm) calls solver's build again
- New solver may have different buffer needs
- Parent re-queries sizes before next allocation
- This is automatic through lazy evaluation

### 4. Multi-Stage Implicit Methods (DIRK, FIRK)
Each stage may call the solver:
- Solver buffers are reused across stages
- Same shared slice passed to each stage's solver call
- No special handling needed

---

## Implementation Order

1. **newton_krylov.py** - Remove factory, use local arrays
2. **linear_solver.py** - Remove factory, use local arrays
3. **base_algorithm_step.py** - Add query methods
4. **ode_implicitstep.py** - Update base implicit step
5. **backwards_euler.py** - Fix solver_scratch pattern
6. **backwards_euler_predict_correct.py** - Same pattern
7. **crank_nicolson.py** - Same pattern
8. **generic_dirk.py** - Update solver calls, buffer sizing
9. **generic_firk.py** - Same pattern
10. **generic_rosenbrock_w.py** - Same pattern
11. **Instrumented files** - Mirror all changes

---

## Test Requirements

### Existing Tests Should Pass
The refactoring preserves behavior while fixing the architecture. All existing algorithm tests should pass after changes.

### Verify Buffer Sizing
Add tests that verify:
- `solver_shared_elements` returns correct size for implicit algorithms
- Explicit algorithms return 0 for solver elements
- Total shared memory computed correctly at loop level

### Verify Solver Execution
Run integration tests with implicit algorithms (backwards_euler, DIRK) to ensure:
- Newton iterations complete successfully
- Linear solver converges
- No memory access errors

---

## Implementation Notes

### No Backwards Compatibility
Per project guidelines: Breaking changes are expected. Remove all obsolete parameters.

### No Optional Parameters
Buffer management is compulsory. Do not add fallbacks for missing sizes.

### Guarantee-by-Design
If a slice calculation would fail, that indicates a bug in the calling code, not something to guard against with defensive checks.

### Comment Style
Comments describe current behavior, not changes:
- Bad: "now uses local arrays instead of registry"
- Good: "allocates buffers in thread-local memory"
