# Instrumented Test Files Update - Agent Plan

## Overview

This plan describes the architectural changes needed to update the instrumented test files in `tests/integrators/algorithms/instrumented/` and the debug file `tests/all_in_one.py` to match the refactored production code structure.

The key change is migrating from fixed shared memory slicing to the new `BufferSettings` pattern that enables selective allocation between shared and local memory.

---

## Component Descriptions

### BufferSettings Pattern (Production Reference)

The production code now uses a hierarchy of buffer settings classes:

1. **BufferSettings** (abstract base in `src/cubie/BufferSettings.py`)
   - Defines interface for shared_memory_elements, local_memory_elements
   - Provides local_sizes and shared_indices properties

2. **LocalSizes** (abstract base)
   - Holds buffer size values for local array allocation
   - Provides `nonzero(attr_name)` method returning `max(size, 1)`

3. **SliceIndices** (abstract base)
   - Holds slice objects for shared memory regions
   - Empty slices (slice(0, 0)) for local buffers

### ERK Buffer Settings (Production Example)

From `src/cubie/integrators/algorithms/generic_erk.py`:

- `ERKLocalSizes`: stage_rhs, stage_accumulator, stage_cache sizes
- `ERKSliceIndices`: stage_rhs, stage_accumulator, stage_cache slices
- `ERKBufferSettings`: n, stage_count, location configurations
  - `stage_rhs_location`: 'local' or 'shared'
  - `stage_accumulator_location`: 'local' or 'shared'
  - Derived: `use_shared_stage_cache`, `stage_cache_aliases_rhs`, etc.

### DIRK Buffer Settings (Production Example)

From `src/cubie/integrators/algorithms/generic_dirk.py`:

- `DIRKLocalSizes`: stage_increment, stage_base, accumulator, solver_scratch, increment_cache, rhs_cache
- `DIRKSliceIndices`: corresponding slices
- `DIRKBufferSettings`: location configurations for each buffer

### Linear Solver Buffer Settings

From `src/cubie/integrators/matrix_free_solvers/linear_solver.py`:

- `LinearSolverLocalSizes`: preconditioned_vec, temp sizes
- `LinearSolverSliceIndices`: corresponding slices
- `LinearSolverBufferSettings`: location configurations

---

## Expected Behavior Changes

### 1. Memory Allocation in Step Functions

**Current (Instrumented):**
```python
stage_rhs = cuda.local.array(n, numba_precision)
stage_accumulator = shared[:accumulator_length]
```

**Target (Match Production):**
```python
if stage_rhs_shared:
    stage_rhs = shared[stage_rhs_slice]
else:
    stage_rhs = cuda.local.array(stage_rhs_local_size, precision)
    for _i in range(stage_rhs_local_size):
        stage_rhs[_i] = typed_zero
```

### 2. Device Function Signatures

**Current (Instrumented):**
```python
numba_precision[:], numba_precision[:], ...
```

**Target (Match Production):**
```python
numba_precision[::1], numba_precision[::1], ...
numba_precision[:, :, ::1], ...  # for 3D arrays
```

### 3. Persistent Local Storage for FSAL

**Current (Instrumented):**
- FSAL caches often alias shared memory directly

**Target (Match Production):**
- When solver_scratch is local, increment_cache and rhs_cache use persistent_local
- persistent_local passed into step function for cross-step persistence

---

## Architectural Changes Required

### File: generic_erk.py (Instrumented)

1. **Add BufferSettings integration:**
   - Add `buffer_settings` parameter to `__init__`
   - Create ERKBufferSettings instance if not provided
   - Store as `config.buffer_settings`

2. **Update ERKStepConfig:**
   - Add `buffer_settings: Optional[ERKBufferSettings]` field

3. **Update build_step method:**
   - Extract buffer settings from compile_settings
   - Unpack boolean flags as compile-time constants
   - Unpack slice indices and local sizes
   - Implement selective allocation pattern

4. **Update step device function:**
   - Change signature arrays to use contiguity specifiers
   - Implement compile-time branching for allocation
   - Add zero-initialization loops for local arrays
   - Update stage_cache aliasing logic

5. **Add property implementations:**
   - `persistent_local_required`: return buffer_settings.persistent_local_elements

### File: generic_dirk.py (Instrumented)

1. **Add BufferSettings integration:**
   - Add `buffer_settings` parameter to `__init__`
   - Create DIRKBufferSettings instance if not provided

2. **Update DIRKStepConfig:**
   - Add `buffer_settings: Optional[DIRKBufferSettings]` field

3. **Update build_step method:**
   - Extract stage_increment_shared, stage_base_shared, etc. flags
   - Unpack slice indices from buffer_settings.shared_indices
   - Unpack local sizes from buffer_settings.local_sizes

4. **Update step device function:**
   - Implement selective allocation for stage_increment, accumulator, solver_scratch
   - Handle stage_base aliasing (aliases accumulator when multistage)
   - Implement persistent_local for increment_cache and rhs_cache when solver_scratch is local

5. **Add persistent_local_required property:**
   - Return buffer_settings.persistent_local_elements

### File: matrix_free_solvers.py (Instrumented)

1. **Add buffer_settings parameter to inst_linear_solver_factory:**
   - Default to LinearSolverBufferSettings(n=n)
   - Extract boolean flags and slice indices

2. **Update linear_solver device function:**
   - Add `shared` parameter for selective allocation
   - Implement compile-time branching for preconditioned_vec and temp

3. **Update inst_newton_krylov_solver_factory:**
   - No buffer_settings needed (uses delta/residual from solver_scratch passed in)
   - Ensure compatible with new linear_solver signature

### File: backwards_euler.py (Instrumented)

Already partially fixed. Verify:
- Signature uses contiguity specifiers
- Solver scratch handling matches production

### File: backwards_euler_predict_correct.py (Instrumented)

Similar updates to backwards_euler.py.

### File: crank_nicolson.py (Instrumented)

Similar updates to backwards_euler.py. May use beta/gamma differently.

### File: explicit_euler.py (Instrumented)

Simpler updates - no implicit solver. Update:
- Signature contiguity specifiers
- Local array allocation patterns

### File: generic_firk.py (Instrumented)

Similar to DIRK but fully implicit stages. Update:
- BufferSettings integration
- Selective allocation pattern
- Signature updates

### File: generic_rosenbrock_w.py (Instrumented)

Rosenbrock-W specific handling. Update:
- BufferSettings integration (if applicable)
- Selective allocation pattern
- Signature updates

### File: all_in_one.py

The debug file needs comprehensive updates:
1. Update buffer allocation patterns in factories
2. Update step function implementations
3. Align configuration options
4. Ensure memory location flags work correctly

---

## Integration Points

### With conftest.py

The instrumented conftest.py needs to:
- Import BufferSettings classes from production if used
- Pass buffer_settings to instrumented step constructors
- Handle persistent_local sizing correctly

### With test_instrumented.py

No changes expected if:
- Device function signatures compatible
- Shared memory sizing correct
- Instrumentation output arrays unchanged

### With CPU Reference (cpu_reference.py)

No changes - CPU reference is independent implementation.

---

## Data Structures

### ERKBufferSettings Fields

| Field | Type | Purpose |
|-------|------|---------|
| n | int | Number of state variables |
| stage_count | int | Number of RK stages |
| stage_rhs_location | str | 'local' or 'shared' |
| stage_accumulator_location | str | 'local' or 'shared' |

### DIRKBufferSettings Fields

| Field | Type | Purpose |
|-------|------|---------|
| n | int | Number of state variables |
| stage_count | int | Number of stages |
| stage_increment_location | str | 'local' or 'shared' |
| stage_base_location | str | 'local' or 'shared' |
| accumulator_location | str | 'local' or 'shared' |
| solver_scratch_location | str | 'local' or 'shared' |

---

## Dependencies

### Required Imports (Production)

```python
from cubie.BufferSettings import BufferSettings, LocalSizes, SliceIndices
from cubie.integrators.algorithms.generic_erk import ERKBufferSettings
from cubie.integrators.algorithms.generic_dirk import DIRKBufferSettings
from cubie.integrators.matrix_free_solvers.linear_solver import LinearSolverBufferSettings
```

### For Instrumented (Option A - Import Production)

Import production BufferSettings classes and use them directly.

### For Instrumented (Option B - Simplified Local)

Define simplified local versions that don't need full production integration. This may be preferred to keep instrumented files self-contained.

---

## Edge Cases

### 1. Single-Stage Methods

- For single-stage methods (Euler, BE), accumulator is zero-length
- stage_base doesn't alias accumulator in single-stage case
- Handle with conditional allocation

### 2. Zero-Length Driver Arrays

- When n_drivers=0, driver arrays may be zero-length
- Production uses `max(n_drivers, 1)` for local array allocation
- Instrumented should match

### 3. FSAL with Local Solver Scratch

- When solver_scratch is local, increment_cache and rhs_cache need persistent_local
- persistent_local slice used instead of solver_scratch slice
- Cross-step caching preserved in persistent_local

### 4. Observables Zero-Length

- When n_observables=0, observable arrays may be zero-length
- Use `max(n_observables, 1)` for allocation

---

## Implementation Order

1. **matrix_free_solvers.py** - Foundation for implicit methods
2. **generic_erk.py** - Most straightforward explicit method
3. **generic_dirk.py** - Complex implicit with multiple buffers
4. **backwards_euler.py** - Verify/complete partial fix
5. **backwards_euler_predict_correct.py** - Similar to BE
6. **crank_nicolson.py** - Similar to BE
7. **explicit_euler.py** - Simplest explicit
8. **generic_firk.py** - Complex fully implicit
9. **generic_rosenbrock_w.py** - Rosenbrock-specific
10. **all_in_one.py** - Debug file last

---

## Validation Approach

After each file update:
1. Run `pytest tests/integrators/algorithms/instrumented/test_instrumented.py -v`
2. Check for signature compatibility errors
3. Check for array dimension mismatches
4. Verify instrumentation output unchanged

Full validation:
1. Run full instrumented test suite
2. Compare CPU vs GPU outputs for accuracy
3. Verify shared memory usage unchanged (or improved)
