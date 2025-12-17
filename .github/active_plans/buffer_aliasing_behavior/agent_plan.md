# Agent Plan: Buffer Aliasing Behavior for Togglable Shared/Local Arrays

## Overview

This document provides detailed architectural specifications for implementing correct buffer aliasing behavior across all BufferSettings subclasses. The implementation must enforce three key scenarios:

1. **Parent shared → Child aliases parent slice**
2. **Parent local + Child shared → Child gets own shared slice**  
3. **Parent local + Child local → Child gets own local array**

---

## Current State Analysis

### BufferSettings Hierarchy

The hierarchy consists of:

```
BufferSettings (abstract base)
├── LoopBufferSettings (ode_loop.py)
├── ERKBufferSettings (generic_erk.py)
├── DIRKBufferSettings (generic_dirk.py)
│   └── stores: newton_buffer_settings: NewtonBufferSettings
├── FIRKBufferSettings (generic_firk.py)
│   └── stores: newton_buffer_settings: NewtonBufferSettings
├── RosenbrockBufferSettings (generic_rosenbrock_w.py)
│   └── stores: linear_solver_buffer_settings: LinearSolverBufferSettings
├── NewtonBufferSettings (newton_krylov.py)
│   └── stores: linear_solver_buffer_settings: LinearSolverBufferSettings
└── LinearSolverBufferSettings (linear_solver.py)
```

### Current Issues

1. **DIRKBufferSettings.solver_scratch_elements**: Currently returns `newton_buffer_settings.shared_memory_elements`. This is wrong when Newton uses local buffers - we would double-count.

2. **DIRKBufferSettings.shared_memory_elements**: Includes `solver_scratch_elements` unconditionally, assuming solver scratch is always provided from shared memory.

3. **FIRKBufferSettings.solver_scratch_elements**: Same issue as DIRK.

4. **NewtonBufferSettings.shared_memory_elements**: Includes linear solver's shared memory. When Newton's delta/residual are local, the linear solver's shared memory starts at offset 0, not offset based on Newton's buffers.

5. **RosenbrockBufferSettings**: Stores `linear_solver_buffer_settings` but doesn't account for it in shared_memory_elements.

---

## Component Specifications

### 1. LinearSolverBufferSettings (No Changes Needed)

This is a leaf node with no nested buffer settings. It correctly computes its own shared and local memory elements based on `preconditioned_vec_location` and `temp_location`.

**Verification**: 
- `shared_memory_elements`: Correct (sums shared buffers only)
- `local_memory_elements`: Correct (sums local buffers only)
- `shared_indices`: Correct (starts at 0, slices for shared buffers)

### 2. NewtonBufferSettings

**Current Behavior:**
- `shared_memory_elements` includes `linear_solver_buffer_settings.shared_memory_elements`
- This is correct when Newton provides shared scratch to the linear solver

**Required Behavior:**
The linear solver shared memory should be included in Newton's shared memory count because Newton allocates the scratch region and passes it to the linear solver. The key insight is:

- Newton's `solver_scratch` (provided by parent DIRK/FIRK) contains Newton's buffers
- Newton then slices `lin_shared = shared_scratch[lin_solver_start:]` for the linear solver
- Linear solver's shared elements should be counted in Newton's total

**Changes Required:**
- **NONE** - Current behavior is correct. Newton includes linear solver shared elements because Newton passes its scratch to the linear solver.

**Slice Computation:**
- `lin_solver_start` = Newton's own shared elements (delta + residual + residual_temp if shared)
- Linear solver slices from Newton's scratch starting at `lin_solver_start`

### 3. DIRKBufferSettings

**Current Behavior:**
- `solver_scratch_elements` = `newton_buffer_settings.shared_memory_elements`
- `shared_memory_elements` always includes `solver_scratch_elements`
- Assumes parent (loop) always provides solver_scratch in shared memory

**Required Behavior:**

The parent-child relationship here is:
- **Parent**: DIRK algorithm's `solver_scratch` (provided from loop's remaining shared scratch)
- **Child**: Newton's buffers (delta, residual, etc.)

Scenario analysis:
1. **solver_scratch shared (normal case)**: Newton slices delta/residual from it
2. **solver_scratch local**: Not meaningful - solver_scratch is always from parent

Actually, looking at the code more carefully: `solver_scratch` is always a slice of the parent's shared memory (`shared[solver_scratch_slice]`). It's not togglable at the DIRK level - it's provided by the loop.

The issue is: **when should DIRKBufferSettings include Newton's memory in its accounting?**

The answer: DIRK's `shared_memory_elements` should include Newton's shared requirements because DIRK carves out the solver_scratch region from the parent loop's shared memory.

**Changes Required:**
- The current implementation is actually correct for the design intent
- `solver_scratch_elements` returning Newton's shared memory is correct because DIRK must reserve that space in its shared memory layout

**Validation:**
- Ensure `solver_scratch_slice` in `shared_indices` has correct size equal to `newton_buffer_settings.shared_memory_elements`

### 4. FIRKBufferSettings

**Same analysis as DIRK** - FIRK carves out solver_scratch for Newton, so it must include Newton's shared memory requirements.

**Changes Required:**
- Current implementation appears correct
- Verify `solver_scratch_elements` matches Newton's total shared memory needs

### 5. RosenbrockBufferSettings

**Current Behavior:**
- Stores `linear_solver_buffer_settings`
- Does NOT include linear solver memory in `shared_memory_elements`
- Does NOT include linear solver memory in `local_memory_elements`

**Required Behavior:**
Rosenbrock uses a linear solver directly (not through Newton). The linear solver needs its own scratch space. Currently, Rosenbrock passes `shared` directly to the linear solver (line 800 in generic_rosenbrock_w.py: `linear_solver(..., shared, ...)`).

This means:
- Rosenbrock should include linear solver's shared memory in its total
- Currently it does NOT - this is a **bug**

**Changes Required:**
Add linear solver's shared memory to Rosenbrock's `shared_memory_elements`:

```python
@property
def shared_memory_elements(self) -> int:
    total = 0
    if self.use_shared_stage_rhs:
        total += self.n
    if self.use_shared_stage_store:
        total += self.stage_store_elements
    if self.use_shared_cached_auxiliaries:
        total += self.cached_auxiliary_count
    # ADD: Include linear solver's shared memory
    if self.linear_solver_buffer_settings is not None:
        total += self.linear_solver_buffer_settings.shared_memory_elements
    return total
```

Similarly for `local_memory_elements` if the linear solver uses local memory.

**Slice Computation:**
Need to add `linear_solver_slice` to `RosenbrockSliceIndices` that comes after the other Rosenbrock buffers.

---

## Integration Points

### Loop → Algorithm Integration

The loop's `LoopBufferSettings` computes `shared_memory_elements` for its own buffers. The `scratch` slice (`shared_indices.scratch`) provides remaining shared memory to the algorithm.

The algorithm (ERK/DIRK/FIRK/Rosenbrock) then carves out its own buffers from the scratch space.

**Key Insight**: The algorithm's `shared_memory_elements` tells the loop how much scratch space is needed. The loop doesn't need to know the internal structure - just the total.

### Algorithm → Solver Integration

For implicit algorithms:
- DIRK/FIRK provide `solver_scratch` to Newton
- Rosenbrock provides `shared` directly to linear solver

The solver then uses its buffer settings to determine where its buffers are within the provided scratch.

---

## Memory Accounting Verification

For each BufferSettings subclass, verify:

1. **No double-counting**: Each memory element is counted exactly once
2. **All elements counted**: Every buffer that's allocated has its memory accounted for
3. **Location-aware**: Shared elements only counted when location='shared'

### Verification Matrix

| BufferSettings | Nested Settings | Includes Nested in shared_memory_elements? | Includes Nested in local_memory_elements? |
|---------------|-----------------|---------------------------------------------|-------------------------------------------|
| LinearSolver | None | N/A | N/A |
| Newton | LinearSolver | Yes (correct) | Yes (correct) |
| DIRK | Newton | Yes (correct) | No (Newton local not counted - Newton's local is separate) |
| FIRK | Newton | Yes (correct) | No |
| Rosenbrock | LinearSolver | **No (BUG)** | **No (BUG)** |

---

## Implementation Steps

### Step 1: Fix RosenbrockBufferSettings Memory Accounting

**File**: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

**Changes**:
1. Update `shared_memory_elements` property to include linear solver's shared memory
2. Update `local_memory_elements` property to include linear solver's local memory
3. Add `linear_solver_slice` to `RosenbrockSliceIndices`
4. Update `shared_indices` property to compute linear solver slice

### Step 2: Add RosenbrockSliceIndices.linear_solver Attribute

**File**: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

The slice indices class needs a new attribute for the linear solver's shared memory region.

### Step 3: Update Tests

**File**: `tests/integrators/algorithms/test_buffer_settings.py`

Add tests verifying:
1. Rosenbrock includes linear solver shared memory
2. Rosenbrock includes linear solver local memory
3. Rosenbrock slice indices include linear solver slice

### Step 4: Verify Existing Behavior

Ensure DIRK, FIRK, Newton implementations correctly handle:
1. All-shared configuration
2. All-local configuration  
3. Mixed configuration

---

## Edge Cases

### 1. Single-Stage Methods
- DIRK/FIRK with `stage_count=1` have empty accumulators
- Solver scratch is still needed for the single implicit solve

### 2. Zero-Size Buffers
- When `n_drivers=0`, driver stack has zero elements
- `LocalSizes.nonzero()` ensures cuda.local.array gets size >= 1

### 3. Default Buffer Settings
- When `newton_buffer_settings` or `linear_solver_buffer_settings` is None
- `__attrs_post_init__` creates default settings

---

## Dependencies and Imports

No new dependencies required. All changes are within existing module structure.

Imports used:
- `attrs` for attrs.define decorators
- `validators` from attrs for location validation
- Existing `BufferSettings`, `LocalSizes`, `SliceIndices` base classes

---

## Expected Behavior After Implementation

1. **RosenbrockBufferSettings.shared_memory_elements**: Includes linear solver's shared memory
2. **RosenbrockBufferSettings.local_memory_elements**: Includes linear solver's local memory
3. **RosenbrockSliceIndices**: Has `linear_solver` slice attribute
4. All BufferSettings: Correct memory accounting without double-counting
5. All algorithms: Work correctly with any combination of shared/local locations

---

## Test Strategy

### Unit Tests (per BufferSettings subclass)
1. Default locations produce expected boolean flags
2. Shared memory elements correct for all-shared config
3. Shared memory elements correct for all-local config
4. Local memory elements correct for all-shared config
5. Local memory elements correct for all-local config
6. Nested buffer settings included in accounting
7. Slice indices have correct start/stop offsets

### Integration Tests
1. DIRK step compiles with all-shared buffers
2. DIRK step compiles with all-local buffers
3. FIRK step compiles with all-shared buffers
4. FIRK step compiles with all-local buffers
5. Rosenbrock step compiles with all-shared buffers
6. Rosenbrock step compiles with all-local buffers
