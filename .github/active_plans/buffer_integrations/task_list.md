# Implementation Task List
# Feature: Buffer Integrations - Registry-Based Hierarchical Memory
# Plan Reference: .github/active_plans/buffer_integrations/agent_plan.md

## Overview

This task list implements hierarchical buffer management using the buffer_registry
pattern. The core architecture is:

1. **Solvers register their own buffers** with the buffer_registry using
   the step algorithm (factory parameter) as the owning factory
2. **Parents query children's sizes** via `buffer_registry.shared_buffer_size(self)`
   after solvers have registered
3. **Memory sizes flow UP through properties, allocations flow DOWN at runtime**
4. **Location parameters remain user-settable** for all buffers

### Key Architecture Principles

- Solvers KEEP the `factory` parameter (solvers register buffers under the
  step's factory context)
- Location parameters MUST be preserved (user-configurable)
- Use `buffer_registry.shared_buffer_size(factory)` for size queries at build
  time (NOT inside device functions)
- Step's `solver_shared_elements` property returns registry query result
- Linear solver slice offset must be computed at factory build time, not at
  device runtime

---

## Task Group 1: Fix Newton-Krylov Linear Solver Slice Bug - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (entire file)

**Input Validation Required**:
- None - this is a bug fix

**Tasks**:

### 1.1 Fix runtime registry call in newton_krylov.py

**File**: `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`
**Action**: Modify

**Problem**: Lines 227-230 contain a runtime call to
`buffer_registry.shared_buffer_size(factory)` inside the CUDA device function.
This won't work because buffer_registry is a Python object that cannot be
called from CUDA code.

**Current incorrect code (lines 227-230):**
```python
            # TODO: AI Error
            # Linear solver uses remaining shared space after Newton buffers
            lin_start = buffer_registry.shared_buffer_size(factory)
            lin_shared = shared_scratch[lin_start:]
```

**Solution**: Compute the slice offset at factory build time and capture it in
the closure. The offset is the size of Newton's own shared buffers.

**Fix approach**:
1. After registering Newton's buffers, compute the offset using:
   ```python
   # Compute Newton's shared buffer size for linear solver slicing
   newton_shared_size = buffer_registry.shared_buffer_size(factory)
   ```
2. Capture `newton_shared_size` in the closure as an int32 constant
3. Use the captured constant inside the device function:
   ```python
   lin_shared = shared_scratch[newton_shared_size:]
   ```

**Detailed changes to `newton_krylov_solver_factory`:**

After the allocator retrieval section (after line 114), add:
```python
    # Compute Newton's shared buffer size for linear solver slicing
    # This value is captured in the closure at compile time
    newton_shared_size = int32(buffer_registry.shared_buffer_size(factory))
```

Replace lines 227-230:
```python
            # TODO: AI Error
            # Linear solver uses remaining shared space after Newton buffers
            lin_start = buffer_registry.shared_buffer_size(factory)
            lin_shared = shared_scratch[lin_start:]
```

With:
```python
            # Linear solver uses shared space after Newton's buffers
            lin_shared = shared_scratch[newton_shared_size:]
```

**Edge cases**:
- If all Newton buffers are local, `newton_shared_size` will be 0, and
  `lin_shared` will be the entire `shared_scratch` - this is correct
- If linear solver buffers are also local, the slice doesn't matter since
  linear solver's allocators will use `cuda.local.array`

**Integration**:
- No signature changes required
- The fix is entirely within the factory function closure

---

**Outcomes**:
- [ ] newton_krylov.py - Runtime registry call replaced with compile-time
      captured constant

---

## Task Group 2: Add factory Parameter to Base Implicit Step - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 226-293)

**Input Validation Required**:
- None - this adds the factory parameter

**Tasks**:

### 2.1 Update ode_implicitstep.py - Pass factory=self to solver factories

**File**: `src/cubie/integrators/algorithms/ode_implicitstep.py`
**Action**: Modify

**Problem**: The base `build_implicit_helpers` method does not pass
`factory=self` to the solver factories. This means Newton and Linear solver
buffers are not registered under the step's factory context, so
`solver_shared_elements` cannot query the registry correctly.

**Changes to `build_implicit_helpers` method (lines 270-276):**

Current code:
```python
        linear_solver = linear_solver_factory(operator,
                                              n=n,
                                              precision=self.precision,
                                              preconditioner=preconditioner,
                                              correction_type=correction_type,
                                              tolerance=krylov_tolerance,
                                              max_iters=max_linear_iters)
```

Replace with:
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

**Changes to newton_krylov_solver_factory call (lines 283-292):**

Current code:
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

Replace with:
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

**Edge cases**: None
**Integration**: This is the base class; subclasses that override this method
already pass `factory=self` (e.g., generic_dirk.py, generic_firk.py)

---

### 2.2 Update solver_shared_elements property to use registry

**File**: `src/cubie/integrators/algorithms/ode_implicitstep.py`
**Action**: Modify

**Problem**: The `solver_shared_elements` property currently returns a
hard-coded value `n * 2`. It should query the buffer_registry for the actual
shared buffer size.

**Current code (lines 295-299):**
```python
    @property
    def solver_shared_elements(self) -> int:
        """Return shared scratch dedicated to the Newton--Krylov solver."""

        return self.compile_settings.n * 2
```

**Replace with:**
```python
    @property
    def solver_shared_elements(self) -> int:
        """Return shared scratch dedicated to the Newton--Krylov solver.
        
        Queries the buffer_registry for total shared memory registered
        by this step's solver chain (Newton + Linear buffers).
        """
        from cubie.buffer_registry import buffer_registry
        return buffer_registry.shared_buffer_size(self)
```

**Edge cases**:
- If no buffers are registered yet (before `build_implicit_helpers` is called),
  the registry returns 0 - this may cause issues if the property is accessed
  too early
- All implicit step algorithms call `build_implicit_helpers` before accessing
  `solver_shared_elements`, so this ordering is maintained

**Integration**:
- The property now dynamically returns the correct size based on buffer
  location settings
- If users set buffer locations to 'shared', the returned size increases
  accordingly

---

**Outcomes**:
- [ ] ode_implicitstep.py - factory=self passed to both solver factories
- [ ] ode_implicitstep.py - solver_shared_elements queries buffer_registry

---

## Task Group 3: Verify Algorithm Implementations - PARALLEL
**Status**: [ ]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler.py
- File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py
- File: src/cubie/integrators/algorithms/crank_nicolson.py
- File: src/cubie/integrators/algorithms/generic_dirk.py
- File: src/cubie/integrators/algorithms/generic_firk.py
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py

**Input Validation Required**:
- None - verification only

**Tasks**:

### 3.1 Verify backwards_euler.py - Uses base class method

**File**: `src/cubie/integrators/algorithms/backwards_euler.py`
**Action**: Review (no changes needed)

BackwardsEulerStep does not override `build_implicit_helpers`, so it inherits
from ODEImplicitStep. After Task Group 2, it will correctly pass `factory=self`
to solver factories.

The device function at line 221:
```python
            solver_scratch = shared[: solver_shared_elements]
```

This is correct - `solver_shared_elements` is captured from
`self.solver_shared_elements` at line 140, which will now return the registry
query result.

**No changes required**

---

### 3.2 Verify backwards_euler_predict_correct.py - Uses base class method

**File**: `src/cubie/integrators/algorithms/backwards_euler_predict_correct.py`
**Action**: Review (no changes needed)

Same pattern as backwards_euler.py - inherits `build_implicit_helpers` from
ODEImplicitStep.

**No changes required**

---

### 3.3 Verify crank_nicolson.py - Uses base class method

**File**: `src/cubie/integrators/algorithms/crank_nicolson.py`
**Action**: Review (no changes needed)

Same pattern as backwards_euler.py - inherits `build_implicit_helpers` from
ODEImplicitStep.

**No changes required**

---

### 3.4 Verify generic_dirk.py - Already passes factory=self

**File**: `src/cubie/integrators/algorithms/generic_dirk.py`
**Action**: Review (no changes needed)

DIRKStep overrides `build_implicit_helpers` and already passes `factory=self`
to both solver factories (lines 342-368). The implementation is correct.

**No changes required**

---

### 3.5 Verify generic_firk.py - Already passes factory=self

**File**: `src/cubie/integrators/algorithms/generic_firk.py`
**Action**: Review (no changes needed)

FIRKStep overrides `build_implicit_helpers` and already passes `factory=self`
to both solver factories (lines 341-367). The implementation is correct.

**No changes required**

---

### 3.6 Verify generic_rosenbrock_w.py - Already passes factory=self

**File**: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
**Action**: Review (no changes needed)

RosenbrockWStep overrides `build_implicit_helpers` and already passes
`factory=self` to `linear_solver_cached_factory` (lines 327-336). The
implementation is correct.

**No changes required**

---

**Outcomes**:
- [ ] backwards_euler.py - confirmed uses base class (no changes)
- [ ] backwards_euler_predict_correct.py - confirmed uses base class (no changes)
- [ ] crank_nicolson.py - confirmed uses base class (no changes)
- [ ] generic_dirk.py - confirmed already correct (no changes)
- [ ] generic_firk.py - confirmed already correct (no changes)
- [ ] generic_rosenbrock_w.py - confirmed already correct (no changes)

---

## Task Group 4: Verify Instrumented Test Files - PARALLEL
**Status**: [ ]
**Dependencies**: Task Groups 1-3

**Required Context**:
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
- File: tests/integrators/algorithms/instrumented/backwards_euler.py
- File: tests/integrators/algorithms/instrumented/generic_dirk.py
- File: tests/integrators/algorithms/instrumented/generic_firk.py
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py

**Input Validation Required**:
- None - instrumented files use local arrays for simplicity

**Tasks**:

### 4.1 Verify instrumented files are independent

**Action**: Review (no changes needed)

The instrumented test files:
1. Use their own `inst_*_factory` functions that do NOT use buffer_registry
2. Allocate buffers with `cuda.local.array` directly
3. Have additional logging parameters for capturing iteration data

This is intentional - instrumented versions are simplified for testing and
logging, not for production buffer management. They should NOT be updated to
mirror the source changes because they serve a different purpose.

**No changes required for any instrumented files**

---

**Outcomes**:
- [ ] Instrumented files confirmed independent (no changes needed)

---

## Task Group 5: Verification and Integration Testing - SEQUENTIAL
**Status**: [ ]
**Dependencies**: All previous groups

**Required Context**:
- All modified files

**Tasks**:

### 5.1 Verify ode_loop.py - No changes needed

**File**: `src/cubie/integrators/loops/ode_loop.py`
**Action**: Review (no changes needed)

The loop queries `buffer_registry.shared_buffer_size(self)` at build time
(line 343) and passes the appropriate slice to the step function. This is
correct.

**No changes required**

---

### 5.2 Verify memory flow is correct

**Action**: Verify architecture

Memory size flow (UP through properties):
1. Linear solver registers `lin_preconditioned_vec`, `lin_temp` with factory
2. Newton solver registers `newton_delta`, `newton_residual`, etc. with factory
3. Step's `solver_shared_elements` queries `buffer_registry.shared_buffer_size(self)`
4. Loop queries `step.shared_memory_elements` to get total requirements

Memory allocation flow (DOWN at runtime):
1. Loop allocates `algorithm_shared` of size `step.shared_memory_elements`
2. Step slices `shared[: solver_shared_elements]` for solver_scratch
3. Newton's allocators get buffers from solver_scratch
4. Newton slices `solver_scratch[newton_shared_size:]` for linear solver
5. Linear solver's allocators get buffers from remaining slice

**No code changes required - verification only**

---

### 5.3 Run tests to verify changes

**Action**: Execute tests

Run the following test commands to verify the implementation:
```bash
pytest tests/integrators/algorithms/test_step_algorithms.py -v
pytest tests/integrators/algorithms/test_buffer_settings.py -v
pytest tests/integrators/matrix_free_solvers/ -v
```

---

**Outcomes**:
- [ ] ode_loop.py - verified no changes needed
- [ ] Memory flow architecture verified
- [ ] Tests pass after changes

---

## Implementation Summary

### Files to Modify

1. **newton_krylov.py** (Task 1.1)
   - Compute `newton_shared_size` at factory build time
   - Replace runtime registry call with captured constant

2. **ode_implicitstep.py** (Tasks 2.1, 2.2)
   - Add `factory=self` to `linear_solver_factory` call
   - Add `factory=self` to `newton_krylov_solver_factory` call
   - Update `solver_shared_elements` property to query registry

### Files to Verify (No Changes)

- backwards_euler.py
- backwards_euler_predict_correct.py
- crank_nicolson.py
- generic_dirk.py
- generic_firk.py
- generic_rosenbrock_w.py
- linear_solver.py
- ode_loop.py
- All instrumented test files

---

## Dependency Chain

```
Task Group 1 (newton_krylov.py fix)
       |
       v
Task Group 2 (ode_implicitstep.py updates)
       |
       +---> Task Group 3 (verify algorithms) [PARALLEL]
       |
       +---> Task Group 4 (verify instrumented) [PARALLEL]
       |
       v
Task Group 5 (integration testing)
```

---

## Estimated Complexity

- **Task Group 1**: Low - Single bug fix in newton_krylov.py
- **Task Group 2**: Low - Two small edits to ode_implicitstep.py
- **Task Group 3**: Low - Verification only
- **Task Group 4**: Low - Verification only
- **Task Group 5**: Low - Testing only

**Total Estimated Effort**: 2-3 hours for implementation and testing
