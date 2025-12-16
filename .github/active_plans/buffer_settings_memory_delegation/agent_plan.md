# Agent Plan: Buffer Settings Memory Delegation

## Problem Summary

When instantiating a Rosenbrock solver, BatchSolverKernel throws an error because memory getters in DIRK, FIRK, and Rosenbrock algorithms calculate memory manually instead of delegating to BufferSettings. This creates:

1. Initialization failures when cached_auxiliary_count isn't yet available
2. Inconsistency with ERK which correctly delegates to BufferSettings
3. Potential memory calculation mismatches between properties and BufferSettings

## Component Descriptions

### BufferSettings Base Class
**Location**: `src/cubie/BufferSettings.py`

The abstract base class defines:
- `shared_memory_elements`: Total shared memory elements (abstract)
- `local_memory_elements`: Total local memory elements (abstract)
- `local_sizes`: LocalSizes instance (abstract)
- `shared_indices`: SliceIndices instance (abstract)

**Required Change**: Add `persistent_local_elements` as an abstract property. This property tracks memory that must persist between step invocations (e.g., FSAL stage caches).

### ERKBufferSettings (Reference Pattern)
**Location**: `src/cubie/integrators/algorithms/generic_erk.py` lines 95-205

Correctly implements:
- `shared_memory_elements`: Sums contributions based on location flags
- `persistent_local_elements`: Returns n if stage_cache cannot alias shared buffers

The ERK algorithm's memory properties correctly delegate:
- `shared_memory_required` → `buffer_settings.shared_memory_elements`
- `persistent_local_required` → `buffer_settings.persistent_local_elements`

### DIRKBufferSettings
**Location**: `src/cubie/integrators/algorithms/generic_dirk.py` lines 111-280

**Current State**:
- Has `persistent_local_elements` property
- Does NOT include cached_auxiliary_count in shared_memory_elements

**Required Change**: 
- Add `cached_auxiliary_count` attribute (default 0)
- Add `cached_auxiliaries_location` attribute (default 'shared' to match current behavior)
- Update `shared_memory_elements` to include cached auxiliaries when shared

### FIRKBufferSettings  
**Location**: `src/cubie/integrators/algorithms/generic_firk.py` lines 103-250

**Current State**:
- Has `shared_memory_elements` and `local_memory_elements`
- Does NOT have `persistent_local_elements`

**Required Change**:
- Add `persistent_local_elements` property (returns 0 as FIRK doesn't use FSAL caching)

### RosenbrockBufferSettings
**Location**: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py` lines 100-235

**Current State**:
- Already has `cached_auxiliary_count` attribute
- Already includes it in `shared_memory_elements`
- Does NOT have `persistent_local_elements`

**Required Change**:
- Add `persistent_local_elements` property (returns 0 for Rosenbrock)

### GenericDIRKStep
**Location**: `src/cubie/integrators/algorithms/generic_dirk.py` lines 1062-1088

**Current State** (INCORRECT):
```python
@property
def shared_memory_required(self) -> int:
    tableau = self.tableau
    stage_count = tableau.stage_count
    accumulator_span = max(stage_count - 1, 0) * self.compile_settings.n
    return (accumulator_span
        + self.solver_shared_elements
        + self.cached_auxiliary_count
    )
```

**Required Change**:
```python
@property
def shared_memory_required(self) -> int:
    return self.compile_settings.buffer_settings.shared_memory_elements
```

Note: The `build_implicit_helpers` method must update `buffer_settings.cached_auxiliary_count`.

### GenericFIRKStep
**Location**: `src/cubie/integrators/algorithms/generic_firk.py` lines 886-908

**Current State** (INCORRECT):
```python
@property
def shared_memory_required(self) -> int:
    config = self.compile_settings
    stage_driver_total = self.stage_count * config.n_drivers
    return (
        self.solver_shared_elements
        + stage_driver_total
        + config.all_stages_n
    )
```

**Required Change**:
```python
@property
def shared_memory_required(self) -> int:
    return self.compile_settings.buffer_settings.shared_memory_elements
```

Also update `persistent_local_required` to delegate.

### GenericRosenbrockWStep
**Location**: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py` lines 967-984

**Current State** (INCORRECT):
```python
@property
def shared_memory_required(self) -> int:
    accumulator_span = self.stage_count * self.n
    cached_auxiliary_count = self.cached_auxiliary_count
    shared_buffers = self.n
    return accumulator_span + cached_auxiliary_count + shared_buffers
```

**Required Change**:
```python
@property
def shared_memory_required(self) -> int:
    return self.compile_settings.buffer_settings.shared_memory_elements
```

Also update `persistent_local_required` to delegate.

## Integration Points

### Algorithm → BufferSettings Update
When `build_implicit_helpers()` determines the actual cached_auxiliary_count, it must:
1. Store to `self._cached_auxiliary_count`
2. Update `self.compile_settings.buffer_settings.cached_auxiliary_count`

**Already correct in Rosenbrock** (lines 478-481):
```python
self._cached_auxiliary_count = get_fn("cached_aux_count")
self.compile_settings.buffer_settings.cached_auxiliary_count = (
    self._cached_auxiliary_count
)
```

**DIRK needs similar logic** if it uses cached auxiliaries. Currently DIRK initializes `_cached_auxiliary_count = 0` and doesn't update buffer_settings.

### BatchSolverKernel → Algorithm Queries
BatchSolverKernel queries `shared_memory_required` at init time before algorithm is built. This works correctly as long as:
1. BufferSettings defaults to 0 for optional attributes (cached_auxiliary_count)
2. Memory properties return valid (possibly minimal) values at init time
3. Actual memory is accurate when kernel is compiled

## Dependencies and Imports

No new imports required. All changes are within existing module structures.

## Edge Cases

### Zero Cached Auxiliaries
When cached_auxiliary_count is 0 (at init or for systems without caching):
- BufferSettings correctly returns memory without cached aux contribution
- Algorithm properties correctly delegate

### Non-Default Buffer Locations
When buffers are configured for local instead of shared:
- BufferSettings already handles this via location flags
- Delegation pattern preserves this behavior

### FIRK Without FSAL
FIRK doesn't use first-same-as-last optimization:
- `persistent_local_elements` returns 0
- This is correct behavior

### Instrumented Algorithms
Test instrumented versions in `tests/integrators/algorithms/instrumented/` must be updated to match source changes. These files add logging but must have identical memory property implementations.

## Verification Steps

After implementation:
1. Instantiate Rosenbrock solver without error: `cubie.solver(system, algorithm='ode23s')`
2. Verify memory calculations match between manual and BufferSettings
3. Run CUDA kernels to verify no "invalid address" errors
4. All existing tests pass
5. Instrumented test algorithms remain in sync

## Files to Modify

1. `src/cubie/BufferSettings.py` - Add persistent_local_elements abstract property
2. `src/cubie/integrators/algorithms/generic_dirk.py`:
   - DIRKBufferSettings: Add cached_auxiliary_count, cached_auxiliaries_location
   - GenericDIRKStep: Delegate memory properties
3. `src/cubie/integrators/algorithms/generic_firk.py`:
   - FIRKBufferSettings: Add persistent_local_elements
   - GenericFIRKStep: Delegate memory properties
4. `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`:
   - RosenbrockBufferSettings: Add persistent_local_elements
   - GenericRosenbrockWStep: Delegate memory properties
5. `tests/integrators/algorithms/instrumented/generic_dirk.py` - Match source changes
6. `tests/integrators/algorithms/instrumented/generic_firk.py` - Match source changes  
7. `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py` - Match source changes
8. `tests/all_in_one.py` - Verify scratch buffer calculations remain correct

## Critical Verification Before Implementation

Before implementing delegation, verify that BufferSettings calculations EXACTLY match the current manual calculations. The issue mentions that using buffer_settings caused "invalid address" CUDA errors, suggesting a possible mismatch.

**For each algorithm, verify**:
- Manual calculation formula
- BufferSettings formula
- Difference (if any)

This verification must happen before changing the delegation to ensure we're not introducing memory shortages.
