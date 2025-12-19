# Buffer Integrations - Agent Plan

## Scope

This plan covers hierarchical buffer integration across step algorithms,
matrix-free solvers, and the integration loop. The core architecture is:

1. **Solvers register their own buffers** with themselves as factory
2. **Parents query children's sizes** via `buffer_registry.shared_buffer_size(child)`
3. **Parents register aggregation buffers** to reserve space for children
4. **Memory sizes flow UP through properties, allocations flow DOWN at runtime**

**Files to Modify:**
- `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`
- `src/cubie/integrators/matrix_free_solvers/linear_solver.py`
- `src/cubie/integrators/algorithms/ode_implicitstep.py`
- `src/cubie/integrators/algorithms/backwards_euler.py`
- `src/cubie/integrators/algorithms/backwards_euler_predict_correct.py`
- `src/cubie/integrators/algorithms/crank_nicolson.py`
- `src/cubie/integrators/algorithms/generic_dirk.py`
- `src/cubie/integrators/algorithms/generic_firk.py`
- `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
- `src/cubie/integrators/loops/ode_loop.py`
- All corresponding instrumented test files

---

## User Stories Reference

See `human_overview.md` for:
- US-1: Solver Buffer Registration
- US-2: Hierarchical Memory Size Queries
- US-3: Loop Memory Allocation
- US-4: Property-Based Memory Propagation

---

## Critical Design Principles

### 1. Solvers Register Their Own Buffers
Each solver registers its buffers with the buffer_registry using ITSELF as
the factory. This is the existing pattern and must be preserved.

### 2. Location Parameters Must Be User-Settable
Buffer locations (shared, local, persistent) are specified via factory
parameters. Do NOT hard-code buffer locations.

### 3. Registry is Source of Truth
Use `buffer_registry.shared_buffer_size(factory)` and
`buffer_registry.persistent_local_buffer_size(factory)` to query sizes.
Never compute sizes inline or duplicate size logic.

### 4. Parents Register Aggregation Buffers
Step algorithms register `solver_shared` and `solver_persistent` buffers
with sizes obtained by querying the solver. At runtime, allocators provide
slices from the parent's arrays to child solvers.

---

## Component Specifications by File

### File 1: linear_solver.py

**Purpose:** Linear solver registers its own buffers and provides size queries.

**Current State (CORRECT):**
- Has `factory` parameter in signature
- Registers `lin_preconditioned_vec` and `lin_temp` buffers
- Gets allocators from registry
- Has location parameters (`preconditioned_vec_location`, `temp_location`)

**Changes Required:**
The current implementation is largely correct. Verify:
- Factory parameter is kept
- Location parameters are kept
- Buffer registration uses `factory` parameter correctly

**No structural changes needed for linear_solver.py** - only verification that
the existing pattern is correct. If any changes were made to remove factory,
they must be reverted.

---

### File 2: newton_krylov.py

**Purpose:** Newton-Krylov solver registers its own buffers and queries
linear solver sizes.

**Current State (CORRECT):**
- Has `factory` parameter in signature
- Registers `newton_delta`, `newton_residual`, `newton_residual_temp`,
  `newton_stage_base_bt` buffers
- Gets allocators from registry
- Has location parameters for each buffer

**Required Changes:**

1. **Fix the linear solver slice calculation** (current bug marked with
   TODO: AI Error):

Current incorrect code:
```python
# TODO: AI Error
lin_start = buffer_registry.shared_buffer_size(factory)
lin_shared = shared_scratch[lin_start:]
```

The problem: `factory` here refers to the Newton solver's parent (the step),
not the linear solver. We need to query the LINEAR solver's shared needs.

**However**, the linear solver is passed in as a compiled device function,
not as a factory object. We cannot query the registry for its sizes.

**Solution:** The linear solver should be passed the shared buffer that
Newton uses. Since linear solver buffers default to local, the slice passed
doesn't affect allocation. But if the user sets linear solver buffers to
shared, they need space.

**Fix approach:** Newton solver needs a reference to the linear solver's
factory (not just the compiled function) to query its sizes. Alternatively,
the step algorithm should register a single `solver_shared` buffer sized to
hold both Newton and Linear needs, and Newton should receive the full buffer.

**Implementation:** Add a `linear_solver_factory` parameter to newton_krylov
that holds the factory reference, or have step compute total solver size
including linear.

2. **Alternative simpler approach:** Since Newton calls linear_solver and
passes a slice, Newton needs to know where linear's shared buffers start.
But if linear's buffers are all local (default), the slice doesn't matter.
The fix is to compute the slice based on Newton's own shared size:

```python
# Newton's shared buffers end at this offset within shared_scratch
newton_shared_size = buffer_registry.shared_buffer_size(self_factory_ref)
# Linear gets the remaining slice (if any shared buffers)
lin_shared = shared_scratch[newton_shared_size:]
```

But `self_factory_ref` is the factory that registered Newton's buffers -
which is Newton's parent (step). This is circular.

**Best solution:** The step algorithm computes total solver shared needs,
registers one aggregation buffer, and passes it to Newton. Newton doesn't
need to slice for linear - it passes the same buffer. Linear's allocators
handle getting the right slice.

---

### File 3: ode_implicitstep.py

**Purpose:** Step algorithm queries solver sizes and registers aggregation
buffers.

**Current State:**
- Calls `linear_solver_factory` without factory parameter
- Calls `newton_krylov_solver_factory` without factory parameter
- Has `solver_shared_elements` property returning `n * 2`

**Required Changes:**

1. **Pass `factory=self` to solver factories:**

```python
linear_solver = linear_solver_factory(
    operator,
    n=n,
    factory=self,  # ADD THIS
    precision=self.precision,
    preconditioner=preconditioner,
    correction_type=correction_type,
    tolerance=krylov_tolerance,
    max_iters=max_linear_iters,
)

nonlinear_solver = newton_krylov_solver_factory(
    residual_function=residual,
    linear_solver=linear_solver,
    n=n,
    factory=self,  # ADD THIS
    tolerance=newton_tolerance,
    max_iters=max_newton_iters,
    damping=newton_damping,
    max_backtracks=newton_max_backtracks,
    precision=self.precision,
)
```

2. **Update `solver_shared_elements` to query registry:**

```python
@property
def solver_shared_elements(self) -> int:
    """Return shared elements for Newton-Krylov solver chain.
    
    Queries the buffer_registry for total shared memory registered
    by this step's solver chain (Newton + Linear buffers).
    """
    from cubie.buffer_registry import buffer_registry
    return buffer_registry.shared_buffer_size(self)
```

Wait - this won't work because Newton and Linear register their buffers
with `self` (the step) as factory. So querying `buffer_registry.shared_buffer_size(self)`
would return the step's registered buffers, which now include the solver buffers.

**This is actually the correct pattern!** When Newton registers with
`factory=step_instance`, querying `buffer_registry.shared_buffer_size(step_instance)`
returns all buffers registered by Newton and Linear (because they use step
as factory).

3. **Alternative: Separate Solver Factory Objects**

If we want Newton to register with itself as factory (not the step), then:
- Newton needs to be a CUDAFactory subclass (which it isn't - it's a function)
- Step would query `buffer_registry.shared_buffer_size(newton_instance)`

**Current limitation:** The solver factories are functions, not CUDAFactory
classes. They cannot serve as registry factories.

**Recommended approach:** Keep the current pattern where solvers register
with the parent step as factory. The step's `solver_shared_elements` property
returns the total solver needs by querying the registry for buffers it owns.

---

### File 4: backwards_euler.py

**Purpose:** Use registry-based buffer allocation.

**Current State:**
- Uses `solver_scratch = shared[: solver_shared_elements]` slicing
- Passes `solver_scratch` to Newton solver

**Required Changes:**

1. **Ensure `solver_shared_elements` is correctly computed:**

The property should return total solver shared needs. If Newton and Linear
register shared buffers with `factory=self`, then:

```python
@property
def solver_shared_elements(self) -> int:
    from cubie.buffer_registry import buffer_registry
    return buffer_registry.shared_buffer_size(self)
```

2. **Keep the current slicing pattern:**

```python
solver_scratch = shared[: solver_shared_elements]
```

This is correct if `solver_shared_elements` returns the right value. The
slice starts at 0 because backwards_euler has no algorithm-specific shared
buffers.

3. **Alternatively, use registry allocator:**

Register a `solver_shared` buffer in the step's __init__:

```python
from cubie.buffer_registry import buffer_registry

# In __init__ or during build setup:
solver_size = buffer_registry.shared_buffer_size(self)
buffer_registry.register(
    'be_solver_shared', self, solver_size, 'shared',
    precision=self.precision
)
alloc_solver = buffer_registry.get_allocator('be_solver_shared', self)
```

Then in device function:
```python
solver_scratch = alloc_solver(shared, persistent_local)
```

---

### File 5-8: Other Implicit Algorithms

Apply the same pattern as backwards_euler:
- `backwards_euler_predict_correct.py`
- `crank_nicolson.py`
- `generic_dirk.py`
- `generic_firk.py`
- `generic_rosenbrock_w.py`

Each must:
1. Pass `factory=self` to solver factory calls (if not already)
2. Query registry for solver shared/persistent sizes
3. Use registry allocators for solver buffer slices

---

### File 9: ode_loop.py

**Purpose:** Query step for total memory requirements.

**Current State:**
- Receives memory size parameters from SingleIntegratorRunCore
- Allocates shared arrays based on parameters

**Required Changes:**

Verify the loop correctly:
1. Queries `step.shared_memory_elements` for total step needs
2. Allocates `algorithm_shared` array of that size
3. Passes full array to step function

The step's `shared_memory_elements` property should return:
```python
@property
def shared_memory_elements(self) -> int:
    return self.algorithm_shared_elements + self.solver_shared_elements
```

Where `solver_shared_elements` comes from registry query.

---

## Key Implementation Pattern

### Registration Phase (build time)
```python
# In linear_solver_factory:
buffer_registry.register('lin_precond_vec', factory, n, location, ...)
buffer_registry.register('lin_temp', factory, n, location, ...)

# In newton_krylov_solver_factory:
buffer_registry.register('newton_delta', factory, n, location, ...)
buffer_registry.register('newton_residual', factory, n, location, ...)
# ... etc

# Factory is the step algorithm instance
```

### Query Phase (property access)
```python
# In step algorithm:
@property
def solver_shared_elements(self) -> int:
    return buffer_registry.shared_buffer_size(self)
```

### Runtime Phase (device execution)
```python
# In step's device function:
solver_scratch = alloc_solver_shared(shared, persistent_local)
# Or directly: solver_scratch = shared[: solver_shared_elements]

# Pass to Newton:
status = newton_solver(state, params, ..., solver_scratch, counters)

# Newton allocates its buffers from solver_scratch:
delta = alloc_delta(solver_scratch, solver_scratch)
residual = alloc_residual(solver_scratch, solver_scratch)
# ...

# Newton passes remaining slice to Linear:
# (Linear's allocators get appropriate slices)
```

---

## Edge Cases

### 1. All Buffers Local (Default)
When all solver buffers use local location:
- `buffer_registry.shared_buffer_size(step)` returns 0 for solver buffers
- `solver_shared_elements` property returns 0 (or only algorithm shared)
- No shared memory wasted on unused solver buffers

### 2. Mixed Buffer Locations
When some buffers are shared, others local:
- Registry correctly computes shared size (only shared buffers counted)
- Local buffers use `cuda.local.array` via allocators
- Shared buffers slice from parent's shared array

### 3. Explicit Algorithms
Explicit algorithms (ExplicitEulerStep) have no solvers:
- `solver_shared_elements` returns 0
- No solver buffer registration
- Works correctly with existing pattern

---

## Implementation Order

1. **Verify linear_solver.py** - Confirm factory parameter present
2. **Verify newton_krylov.py** - Confirm factory parameter present
3. **Fix newton_krylov.py** - Fix the lin_shared slice calculation
4. **Update ode_implicitstep.py** - Pass factory=self to solver factories
5. **Update step algorithms** - Ensure correct registry queries
6. **Verify ode_loop.py** - Confirm memory propagation works
7. **Update instrumented files** - Mirror source changes

---

## Test Requirements

### Verify Buffer Registration
- Solver buffers registered with step factory
- Registry returns correct shared/persistent sizes

### Verify Memory Propagation
- Step's `solver_shared_elements` matches registry query
- Loop receives correct total from step
- Kernel allocates sufficient shared memory

### Verify Location Parameters
- Changing buffer location affects registry size queries
- Shared buffers get shared memory allocation
- Local buffers get local array allocation

---

## Instrumented Files

Each instrumented file must mirror the source changes while preserving
logging functionality. Key files:

1. **instrumented/matrix_free_solvers.py** - Keep factory parameter,
   keep buffer registration
2. **instrumented/backwards_euler.py** - Use same allocation pattern
3. **instrumented/generic_dirk.py** - Pass factory to solvers

---

## Implementation Notes

### Factory Parameter is REQUIRED
The `factory` parameter in solver factories is essential for buffer
registration. Do NOT remove it.

### Location Parameters Stay
Buffer location parameters allow user configuration. Do NOT hard-code
locations or remove location parameters.

### Registry Queries for Sizes
Always use `buffer_registry.shared_buffer_size(factory)` to get sizes.
Do NOT compute sizes inline.

### Comment Style
Comments describe current behavior:
- Good: "registers delta buffer with parent step factory"
- Bad: "now registers with factory instead of using local arrays"
