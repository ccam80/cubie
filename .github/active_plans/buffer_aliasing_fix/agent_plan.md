# Buffer Aliasing Fix - Agent Plan

## Technical Context for Agents

This plan addresses incorrect shared memory aliasing in BufferSettings objects. The core issue is that child arrays (slices of parent arrays) are being aliased based on incomplete conditions - ignoring whether the parent array is actually in shared memory.

---

## Component Descriptions

### 1. BufferSettings Base Class (`src/cubie/BufferSettings.py`)

The abstract base class for all buffer configuration. Provides:
- `shared_memory_elements` property: Total shared memory count
- `local_memory_elements` property: Total local memory count  
- `local_sizes` property: Returns LocalSizes instance
- `shared_indices` property: Returns SliceIndices instance

**No changes required** to the base class itself.

### 2. DIRKBufferSettings (`src/cubie/integrators/algorithms/generic_dirk.py`)

Currently manages these arrays:
- `stage_accumulator` - Stores accumulated stage contributions (configurable location)
- `increment_cache` - Cache for increment values (currently aliased from `solver_scratch`)
- `rhs_cache` - Cache for RHS values (currently aliased from `solver_scratch`)
- `stage_base` - Current stage base state (currently aliased from `stage_accumulator`)
- `proposed_increment` - Proposed increment buffer (configurable location)
- `solver_scratch` - Marked "always shared" but actually depends on Newton settings

**Issues Identified:**
1. Line ~410: `stage_base` aliasing only checks `multistage`, not `stage_accumulator` location
2. Lines ~350-360: `increment_cache` and `rhs_cache` alias from `solver_scratch`, but if Newton solver uses all local, there's no shared memory to slice
3. Memory accounting doesn't handle the aliasing conditions correctly

### 3. ERKBufferSettings (`src/cubie/integrators/algorithms/generic_erk.py`)

Manages explicit Runge-Kutta buffers. To be analyzed for similar patterns:
- Stage storage arrays
- Accumulator arrays

### 4. FIRKBufferSettings (`src/cubie/integrators/algorithms/generic_firk.py`)

Manages fully implicit RK buffers. To be analyzed for similar patterns.

### 5. RosenbrockBufferSettings (`src/cubie/integrators/algorithms/generic_rosenbrock_w.py`)

Currently manages:
- `stage_rhs` - Stage RHS buffer (configurable location)
- `stage_store` - Stage store buffer (configurable location)
- `cached_auxiliaries` - Cached auxiliaries buffer (configurable location)
- Contains `linear_solver_buffer_settings`

**Note**: Rosenbrock uses a linear solver directly (not Newton), so its aliasing pattern differs.

### 6. NewtonBufferSettings (`src/cubie/integrators/matrix_free_solvers/newton_krylov.py`)

Currently manages:
- `delta` - Newton direction buffer (default: shared)
- `residual` - Residual buffer (default: shared)
- `residual_temp` - Temporary residual buffer (default: local)
- Contains `linear_solver_buffer_settings`

The Newton solver receives a `shared_scratch` parameter from its parent. The parent needs to:
1. Check if Newton has any shared memory requirements
2. If yes, slice from parent's shared pool
3. If no, pass an empty/dummy reference

### 7. LinearSolverBufferSettings (`src/cubie/integrators/matrix_free_solvers/linear_solver.py`)

Currently manages:
- `preconditioned_vec` - Preconditioned vector (default: local)
- `temp` - Temporary vector (default: local)

---

## Expected Behavior Specification

### Aliasing Logic Matrix

| Parent Location | Child Configured | Child Actual Behavior |
|-----------------|------------------|----------------------|
| shared | shared | Alias from parent slice |
| shared | local | Separate local allocation |
| local | shared | Separate shared allocation |
| local | local | Separate local allocation |

### Memory Accounting Rules

1. **Aliased arrays**: Only count parent's contribution to shared/local totals
2. **Separate allocations**: Count both parent and child independently
3. **Child buffer settings**: Always add their totals (they manage their own internal allocations)

---

## Architectural Changes Required

### A. DIRKBufferSettings Modifications

#### A1. Add Explicit Location Settings for Child Arrays

New attributes needed:
- `stage_base_location: str` - 'local' or 'shared' (default: 'shared')
- `increment_cache_location: str` - 'local' or 'shared' (default: 'shared')  
- `rhs_cache_location: str` - 'local' or 'shared' (default: 'shared')

Add corresponding `use_shared_*` boolean properties.

#### A2. Update shared_memory_elements Property

Current logic counts arrays based on their individual `use_shared_*` flags. This needs to:
1. Check aliasing conditions before counting
2. Avoid double-counting when aliased

Pseudo-logic:
```
total = 0
# stage_accumulator
if use_shared_stage_accumulator:
    total += stage_accumulator_elements
# stage_base - may alias stage_accumulator
if use_shared_stage_base:
    if NOT (use_shared_stage_accumulator AND can_alias_stage_base):
        total += stage_base_elements  # Separate allocation needed
# increment_cache and rhs_cache - may alias solver_scratch
if newton_has_shared_memory:
    # solver_scratch exists, can potentially alias
    if use_shared_increment_cache AND increment_cache_aliases_solver_scratch:
        pass  # Already counted in Newton's contribution
    elif use_shared_increment_cache:
        total += increment_cache_elements
# ... similar for rhs_cache
total += newton_buffer_settings.shared_memory_elements
return total
```

#### A3. Update local_memory_elements Property

Similar logic but for local arrays.

#### A4. Update shared_indices Property

The SliceIndices object needs to:
1. Include slices for all arrays that ARE in shared memory
2. Handle the case where child arrays are NOT aliased but still shared (need separate slice allocation)

#### A5. Update local_sizes Property

The LocalSizes object needs to correctly report sizes for all arrays that ARE local, including child arrays that couldn't be aliased.

#### A6. Update Device Function (`build_step`)

The compile-time branching in the device function needs to:
1. Check both parent location AND aliasing eligibility
2. Have fallback allocation paths for all combinations

Current pattern:
```python
if multistage:
    stage_base = stage_accumulator[base_offset:base_end]
```

Required pattern:
```python
if use_shared_stage_accumulator and use_shared_stage_base and multistage:
    # Alias case
    stage_base = shared[stage_base_slice_in_accumulator]
elif use_shared_stage_base:
    # Separate shared case
    stage_base = shared[stage_base_separate_slice]
else:
    # Local case
    stage_base = cuda.local.array(stage_base_size, precision)
```

### B. Similar Patterns in Other Algorithm Files

#### B1. generic_erk.py

Analyze and document any aliasing patterns. Expected arrays:
- Stage vectors
- Accumulator arrays

#### B2. generic_firk.py

Analyze and document any aliasing patterns.

#### B3. generic_rosenbrock_w.py

Current implementation appears simpler (no Newton solver, uses linear solver directly). Verify that:
- `linear_solver_buffer_settings` shared memory is correctly accounted
- No internal aliasing issues

### C. Newton/Linear Solver Integration

#### C1. Parent-to-Newton Interface

The parent (DIRK) passes `shared_scratch` to Newton. This should be:
1. A valid slice when Newton has shared memory requirements
2. An empty/zero-length reference when Newton uses all local

The Newton solver already handles this via `lin_solver_start` in its slice indices.

#### C2. Newton-to-LinearSolver Interface

Similar pattern: Newton passes `shared[lin_solver_start:]` to linear solver.

---

## Integration Points

### Existing Patterns to Follow

1. **Selective allocation pattern**: Already implemented throughout - `if X_shared: ... else: cuda.local.array(...)`

2. **BufferSettings attrs class pattern**: Use `@attrs.define`, validators from `cubie._utils`

3. **SliceIndices pattern**: Return empty slices `slice(0, 0)` for arrays not in shared memory

4. **LocalSizes.nonzero() pattern**: Used for local array sizing (returns max(1, size))

### Dependencies

- Changes to BufferSettings must maintain compatibility with how they're instantiated in algorithm `__init__` methods
- Device function changes must maintain the same external interface (parameters)
- Memory accounting must remain consistent with what loops expect

---

## Data Structures

### DIRKSliceIndices (existing, to be extended)

```
stage_accumulator: slice  # Existing
increment_cache: slice    # Existing but may need separate slice
rhs_cache: slice          # Existing but may need separate slice
stage_base: slice         # NEW - if separate shared allocation needed
solver_scratch: slice     # Existing (for Newton's portion)
local_end: int            # Existing
```

### DIRKLocalSizes (existing, to be extended)

```
stage_accumulator: int    # Existing
increment_cache: int      # NEW - if not aliased and local
rhs_cache: int           # NEW - if not aliased and local
stage_base: int          # NEW - if not aliased and local
proposed_increment: int   # Existing
```

---

## Edge Cases to Consider

1. **Single-stage methods**: `multistage=False` means no stage_base aliasing possible anyway
2. **All-local Newton**: No solver_scratch to slice from; parent must handle gracefully
3. **Zero-sized arrays**: Some arrays may be zero-sized in certain configurations
4. **Cache invalidation**: Changing buffer location settings should invalidate CUDAFactory caches

---

## Testing Considerations

Tests should cover:
1. All combinations of parent/child shared/local settings
2. Memory accounting correctness for each combination
3. Actual device function execution with different settings
4. Edge cases: single-stage, all-local solver, etc.

Existing test infrastructure in `tests/integrators/algorithms/` should be extended.

---

*This document is for use by detailed_implementer and reviewer agents.*
