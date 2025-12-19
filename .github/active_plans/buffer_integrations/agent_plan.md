# Buffer Integrations - Agent Plan

## Scope

This plan covers hierarchical buffer integration across step algorithms,
matrix-free solvers, and the integration loop. The core architecture is:

1. **Solvers register buffers with step as factory** - Newton and Linear
   register their buffers using the step algorithm (passed as `factory`
   parameter) as the owning parent
2. **Manual size calculation for now** - Step algorithms compute solver memory
   needs manually because solvers are not CUDAFactory objects
3. **Memory sizes flow UP through properties, allocations flow DOWN at runtime**
4. **Build-time offset computation** - Newton computes slice offset after
   registration and captures it in the closure

**Key Constraint**: Solvers will NOT be converted to CUDAFactory objects in
this task. That is a separate, parallel work stream.

**Files to Modify:**
- `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`
- `src/cubie/integrators/algorithms/ode_implicitstep.py`

**Files Already Correct (No Changes):**
- `src/cubie/integrators/matrix_free_solvers/linear_solver.py`
- `src/cubie/integrators/algorithms/backwards_euler.py`
- `src/cubie/integrators/algorithms/backwards_euler_predict_correct.py`
- `src/cubie/integrators/algorithms/crank_nicolson.py`
- `src/cubie/integrators/algorithms/generic_dirk.py`
- `src/cubie/integrators/algorithms/generic_firk.py`
- `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
- `src/cubie/integrators/loops/ode_loop.py`
- All instrumented test files (independent implementations)

---

## User Stories Reference

See `human_overview.md` for:
- US-1: Hierarchical Memory Size Propagation
- US-2: Runtime Slice Allocation
- US-3: Buffer Location Configurability
- US-4: Fix Newton-Krylov Slice Bug

---

## Critical Design Principles

### 1. Solvers Register with Step's Factory Context
Solver factories receive `factory=self` from the step algorithm. All solver
buffers are registered under the step's factory context, not the solver's.

### 2. Manual Solver Size Calculation (Temporary)
Since solvers are factory functions (not CUDAFactory objects), step algorithms
cannot query the registry for solver sizes. Calculate manually:
- Newton buffers: `n * 4` (delta, residual, residual_temp, stage_base_bt)
- Linear buffers: `n * 2` (preconditioned_vec, temp)
- Total: `n * 6` when all buffers are shared (default: all local, so 0)

### 3. Build-Time Offset Computation for Newton
After registering Newton's buffers, compute `newton_shared_size` by querying
`buffer_registry.shared_buffer_size(factory)`. Capture this as a compile-time
constant. Do NOT call registry methods inside device functions.

### 4. Location Parameters Preserved
All buffer location parameters remain user-settable. Default is 'local' for
all solver buffers.

---

## Component Specifications by File

### File 1: newton_krylov.py

**Purpose:** Fix runtime registry call bug - compute slice offset at build time.

**Current Bug (lines 227-230):**
```python
            # TODO: AI Error
            # Linear solver uses remaining shared space after Newton buffers
            lin_start = buffer_registry.shared_buffer_size(factory)
            lin_shared = shared_scratch[lin_start:]
```

**Problem:** `buffer_registry.shared_buffer_size(factory)` is called inside
the CUDA device function. Python object methods cannot be called from CUDA
device code - this causes compilation/runtime failures.

**Solution:** Compute the offset at factory build time, after Newton's buffer
registration, and capture it in the closure.

**Required Changes:**

1. After the allocator retrieval section (~line 114), add:
```python
    # Compute Newton's shared buffer size for linear solver slicing
    # This value is captured in the closure at compile time
    newton_shared_size = int32(buffer_registry.shared_buffer_size(factory))
```

2. Replace lines 227-230 with:
```python
            # Linear solver uses shared space after Newton's buffers
            lin_shared = shared_scratch[newton_shared_size:]
```

**Behavior:**
- If all Newton buffers are local (default), `newton_shared_size` is 0
- Linear solver receives the full `shared_scratch` array
- If some Newton buffers are shared, linear solver receives remaining slice
- This matches the existing default behavior while fixing the bug

---

### File 2: ode_implicitstep.py

**Purpose:** Pass `factory=self` to solver factories in base class.

**Current State (lines 270-292):**
The base `build_implicit_helpers` method calls `linear_solver_factory` and
`newton_krylov_solver_factory` without passing `factory=self`.

**Problem:** Newton and Linear solver buffers are not registered under the
step's factory context. This breaks the hierarchical buffer ownership model
and means `solver_shared_elements` cannot use registry queries.

**Required Changes:**

1. Update `linear_solver_factory` call (lines 270-276):
```python
        linear_solver = linear_solver_factory(
            operator,
            n=n,
            factory=self,
            precision=self.precision,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
        )
```

2. Update `newton_krylov_solver_factory` call (lines 283-292):
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
            precision=self.precision,
        )
```

**Note about `solver_shared_elements` property:**

The current implementation (line 299) returns `n * 2`:
```python
    @property
    def solver_shared_elements(self) -> int:
        """Return shared scratch dedicated to the Newton--Krylov solver."""
        return self.compile_settings.n * 2
```

This is technically incorrect but acceptable for now because:
- Default buffer locations are all 'local', so no shared memory is used
- When all buffers are local, the slice size doesn't matter
- Fixing this requires either registry queries (after `factory=self` is
  passed) or manual calculation of `n * 6` (when all shared)

**Future improvement:** Update to query registry after `build_implicit_helpers`
is called, or calculate based on actual location settings.

---

### Files Already Correct (Verification Only)

#### linear_solver.py
- Already has `factory` parameter in signature
- Already registers buffers with `factory` parameter
- No changes needed

#### generic_dirk.py, generic_firk.py
- Already pass `factory=self` to both solver factories
- Override `build_implicit_helpers` with correct implementation
- No changes needed

#### generic_rosenbrock_w.py
- Already passes `factory=self` to `linear_solver_cached_factory`
- No changes needed

#### backwards_euler.py, backwards_euler_predict_correct.py, crank_nicolson.py
- Inherit `build_implicit_helpers` from ODEImplicitStep
- Will automatically get `factory=self` after base class is fixed
- No changes needed

#### ode_loop.py
- Already queries `buffer_registry.shared_buffer_size(self)` at build time
- Already passes remaining scratch to step function
- No changes needed

#### Instrumented test files
- Use independent implementations with `cuda.local.array`
- Do not use buffer_registry
- No changes needed

---

## Edge Cases

### 1. All Buffers Local (Default)
- `newton_shared_size` is 0
- Linear solver receives full `shared_scratch`
- Step's `solver_shared_elements` returns 0 (via registry query) or `n * 2`
- No shared memory allocated for solvers

### 2. Some Buffers Shared
- Newton computes actual shared size after registration
- Slice offset correctly computed at build time
- Linear solver receives remaining slice
- Shared memory correctly allocated

### 3. Explicit Algorithms
- No solver factories called
- `solver_shared_elements` returns 0 (from BaseAlgorithmStep)
- Works correctly

---

## Implementation Order

1. **Fix newton_krylov.py** - Compute `newton_shared_size` at build time
2. **Update ode_implicitstep.py** - Pass `factory=self` to solver factories
3. **Run tests** - Verify changes don't break existing behavior

---

## Test Requirements

### Verify Newton-Krylov Fix
- Newton solver compiles without registry call in device function
- Linear solver receives correct slice offset
- Implicit algorithms work with both local and shared buffers

### Verify Factory Parameter
- Solver buffers registered under step's factory context
- `buffer_registry.shared_buffer_size(step)` returns correct value
- Location updates propagate through registry

---

## Comment Style Reminder

Comments describe current behavior:
- Good: "Linear solver uses shared space after Newton's buffers"
- Bad: "Now uses build-time computed offset instead of runtime registry call"
