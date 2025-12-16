# Agent Plan: Fix Rosenbrock BatchSolver Instantiation Circular Dependency

## Problem Overview

The circular dependency in Rosenbrock solver instantiation occurs because:

1. `BatchSolverKernel.__init__` accesses `single_integrator.shared_memory_elements`
2. This calls `GenericRosenbrockWStep.shared_memory_required`
3. This property accesses `self.cached_auxiliary_count`
4. `cached_auxiliary_count` triggers `build_implicit_helpers()` if value is `None`
5. This violates CuBIE's lazy compilation design

## Component Descriptions

### GenericRosenbrockWStep (src/cubie/integrators/algorithms/generic_rosenbrock_w.py)

The Rosenbrock-W step function implementing implicit Runge-Kutta integration with matrix-free Newton solvers. Inherits from `ODEImplicitStep` and `CUDAFactory`.

**Current problematic property:**
```python
@property
def shared_memory_required(self) -> int:
    """Return the number of precision entries required in shared memory."""
    accumulator_span = self.stage_count * self.n
    cached_auxiliary_count = self.cached_auxiliary_count  # <-- triggers build!
    shared_buffers = self.n
    return accumulator_span + cached_auxiliary_count + shared_buffers
```

### RosenbrockBufferSettings (same file, lines 100-191)

Configuration class for Rosenbrock step buffer sizes. Has:
- `cached_auxiliary_count: int` field with default value of 0
- `shared_memory_elements` property that calculates total shared memory correctly

The `shared_memory_elements` property already handles the calculation properly, including the cached_auxiliary_count when configured for shared memory.

### GenericERKStep (src/cubie/integrators/algorithms/generic_erk.py)

Reference implementation showing the correct pattern:
```python
@property
def shared_memory_required(self) -> int:
    """Return the number of precision entries required in shared memory."""
    return self.compile_settings.buffer_settings.shared_memory_elements
```

## Expected Behavior After Fix

### Before Build (Init Time)
- `GenericRosenbrockWStep.shared_memory_required` returns `compile_settings.buffer_settings.shared_memory_elements`
- `buffer_settings.cached_auxiliary_count` is 0 (default)
- `buffer_settings.shared_memory_elements` calculates based on memory location settings with 0 cached auxiliaries
- No `build_implicit_helpers()` call occurs

### After Build
- `build_implicit_helpers()` is called as part of the normal build process
- `cached_auxiliary_count` is updated in `buffer_settings` (line 479)
- Subsequent access to `shared_memory_required` → `buffer_settings.shared_memory_elements` returns the correct value with actual `cached_auxiliary_count`

## Architectural Change Required

### Change GenericRosenbrockWStep.shared_memory_required Property

**From (lines 967-973):**
```python
@property
def shared_memory_required(self) -> int:
    """Return the number of precision entries required in shared memory."""
    accumulator_span = self.stage_count * self.n
    cached_auxiliary_count = self.cached_auxiliary_count
    shared_buffers = self.n
    return accumulator_span + cached_auxiliary_count + shared_buffers
```

**To:**
```python
@property
def shared_memory_required(self) -> int:
    """Return the number of precision entries required in shared memory."""
    return self.compile_settings.buffer_settings.shared_memory_elements
```

## Integration Points

1. **BatchSolverKernel.__init__** (lines 156-161): No change needed. Will receive correct value from the delegated calculation.

2. **SingleIntegratorRun.shared_memory_elements**: No change needed. Already calls `self._algo_step.shared_memory_required`.

3. **RosenbrockBufferSettings.shared_memory_elements**: Already implements the calculation correctly. No change needed.

4. **build_implicit_helpers()** (line 479): Already updates `buffer_settings.cached_auxiliary_count`. No change needed.

## Expected Interactions

```
BatchSolverKernel.__init__
    └── SingleIntegratorRun.shared_memory_elements
        └── GenericRosenbrockWStep.shared_memory_required
            └── compile_settings.buffer_settings.shared_memory_elements  # FIXED
                └── Returns calculated value (0 for cached_aux at init)

# Later, during build:
GenericRosenbrockWStep.build()
    └── build_implicit_helpers()
        └── updates buffer_settings.cached_auxiliary_count
            └── shared_memory_elements now returns correct value
```

## Data Structures

### RosenbrockBufferSettings Fields Used
- `n: int` - Number of state variables
- `stage_count: int` - Number of RK stages
- `cached_auxiliary_count: int` - Default 0, updated during build
- `stage_rhs_location: str` - 'local' or 'shared'
- `stage_store_location: str` - 'local' or 'shared'
- `cached_auxiliaries_location: str` - 'local' or 'shared'

### RosenbrockBufferSettings.shared_memory_elements Calculation
```python
@property
def shared_memory_elements(self) -> int:
    total = 0
    if self.use_shared_stage_rhs:
        total += self.n
    if self.use_shared_stage_store:
        total += self.stage_store_elements  # stage_count * n
    if self.use_shared_cached_auxiliaries:
        total += self.cached_auxiliary_count
    return total
```

## Dependencies

### Required Imports
No new imports required. The change uses existing `self.compile_settings` access pattern.

### Existing Patterns Referenced
- `GenericERKStep.shared_memory_required` (line 808-810)
- `GenericERKStep.persistent_local_required` (line 824-825)

## Edge Cases

1. **No Cached Auxiliaries**: If the ODE system has no cached auxiliaries, `cached_auxiliary_count` remains 0. The calculation works correctly.

2. **All Buffers in Local Memory**: With default locations ('local'), `shared_memory_elements` returns 0. This matches expected behavior.

3. **All Buffers in Shared Memory**: With all locations set to 'shared', the calculation correctly sums all components.

4. **Solver Never Built**: If solver is never built (unlikely but possible), `shared_memory_elements` returns value based on default `cached_auxiliary_count` of 0.

## Verification Approach

1. Test that `cubie.solver(system, algorithm='ode23s')` succeeds without error
2. Confirm no `build_implicit_helpers()` call during init (can verify via debugger or log)
3. Confirm solver works correctly after calling `.solve()` (build occurs, then integration works)
4. Existing tests for Rosenbrock algorithms should continue to pass
