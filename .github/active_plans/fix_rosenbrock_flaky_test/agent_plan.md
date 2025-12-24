# Agent Plan: Fix Rosenbrock Solver Flaky Test Errors

## Problem Statement

The Rosenbrock-W solver in `generic_rosenbrock_w.py` experiences flaky test failures because the `stage_increment` buffer is used as an initial guess for the linear solver before it has been initialized on the first integration step.

## Root Cause Analysis

### Buffer Registration (Current)

In `register_buffers()` at lines 268-274:

```python
# Stage increment should persist between steps for initial guess
buffer_registry.register(
    'stage_increment', self, n,
    config.stage_store_location,
    aliases='stage_store',
    precision=precision
)
```

The buffer:
1. Uses `stage_store_location` (typically `'local'`)
2. Has `persistent=False` (default)
3. Aliases `stage_store` for memory reuse

### Buffer Usage in Step Function

At line 526-540 in `build_step()`:
```python
# Use stored copy as the initial guess for the first stage.
status_code |= linear_solver(
    state,
    parameters,
    drivers_buffer,
    base_state_placeholder,
    cached_auxiliaries,
    stage_time,
    dt_scalar,
    numba_precision(1.0),
    stage_rhs,
    stage_increment,  # <-- Initial guess comes from here
    solver_shared,
    solver_persistent,
    krylov_iters_out,
)
```

### Why This Causes Flaky Failures

1. On first step, `stage_increment` contains uninitialized memory (garbage values)
2. The linear solver uses `stage_increment` as its initial guess (`x` parameter)
3. Bad initial guess may cause:
   - Slow convergence → max iterations exceeded → error code
   - Divergence → solver failure → error code
   - Sometimes luck → garbage happens to be close to solution → passes

4. The flakiness depends on:
   - What garbage values happen to be in memory
   - Which may vary between runs, GPU states, or CI environments

## Solution Architecture

### Component: register_buffers() in GenericRosenbrockWStep

**Location**: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

**Change**: Add `persistent=True` to the `stage_increment` buffer registration.

**Expected Behavior**:
- The buffer will be allocated from `persistent_local` instead of `local`
- `persistent_local` is zeroed at loop entry (see `ode_loop.py` line 449)
- First step will see zeros as initial guess (valid starting point)
- Subsequent steps will see previous solution (warm-start preserved)

### Why persistent=True Works

The `persistent` flag in buffer registration causes:

1. **BufferGroup.register()** stores `persistent=True` in the CUDABuffer entry

2. **CUDABuffer.is_persistent_local** returns `True` when `location='local'` and `persistent=True`

3. **BufferGroup.build_persistent_layout()** includes this buffer in persistent slice computation

4. **CUDABuffer.build_allocator()** generates an allocator that returns a slice from `persistent_local` instead of creating a fresh `cuda.local.array`

5. **ODE Loop** zeros `persistent_local` at entry:
   ```python
   persistent_local[:] = precision(0.0)
   ```

### Integration Points

1. **Buffer Registry** (`buffer_registry.py`):
   - No changes needed
   - Existing infrastructure handles `persistent=True` correctly

2. **ODE Loop** (`ode_loop.py`):
   - No changes needed
   - Already zeros `persistent_local` at line 449

3. **Linear Solver** (`linear_solver.py`):
   - No changes needed
   - Already receives initial guess in `x` parameter

4. **Rosenbrock Step** (`generic_rosenbrock_w.py`):
   - Change buffer registration to include `persistent=True`

### Memory Layout Impact

Before:
- `stage_increment`: allocated via `cuda.local.array(n, precision)` (uninitialized)

After:
- `stage_increment`: slice from `persistent_local[offset:offset+n]` (zeroed at loop entry)

The persistent_local array size for the Rosenbrock step will increase by `n` elements (one per state variable).

## Detailed Implementation

### File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py

**Method**: `register_buffers()`

**Current code** (lines 268-274):
```python
# Stage increment should persist between steps for initial guess
buffer_registry.register(
    'stage_increment', self, n,
    config.stage_store_location,
    aliases='stage_store',
    precision=precision
)
```

**Required change**:
Add `persistent=True` parameter to the `buffer_registry.register()` call.

**Rationale**:
- The comment already explains the intent: buffer should persist between steps
- Adding `persistent=True` implements this intent correctly
- The aliasing with `stage_store` is preserved for stages after stage 0

### Dependencies

- `buffer_registry` import already exists at line 53
- `CUDABuffer` class in buffer_registry.py already supports `persistent` parameter
- No new imports needed

### Edge Cases

1. **Aliasing with stage_store**: The alias relationship is preserved. If `stage_store` is shared memory, the aliasing behavior continues. The `persistent=True` only affects the fallback when the buffer needs its own allocation.

2. **stage_store_location='shared'**: If the user configures shared memory for stage_store, the stage_increment will alias into shared (which is also zeroed at loop entry via `shared_scratch[:] = 0.0`). The fix still works.

3. **Subsequent stages**: Stages 1+ use the previous stage's solution from `stage_store`, not `stage_increment`. This is unaffected by the change.

## Validation Approach

1. **Unit test**: The existing Rosenbrock tests should pass consistently after this fix

2. **Behavioral check**: First step should no longer fail due to uninitialized buffer

3. **Regression check**: Warm-start behavior for subsequent steps should be preserved (buffer contains previous solution, not zeros)

## Summary

| Aspect | Details |
|--------|---------|
| **File** | `src/cubie/integrators/algorithms/generic_rosenbrock_w.py` |
| **Method** | `register_buffers()` |
| **Change** | Add `persistent=True` to stage_increment registration |
| **Lines** | 268-274 |
| **Impact** | First-step initial guess will be zeros instead of garbage |
| **Risk** | Low - uses existing infrastructure correctly |
