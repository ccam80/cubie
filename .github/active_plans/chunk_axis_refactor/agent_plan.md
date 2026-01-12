# Chunk Axis Refactor: Agent Plan

## Problem Statement

The chunk_axis parameter controls whether large batch integrations are
chunked along the "run" or "time" dimension when GPU memory is
insufficient. Currently, chunk_axis has multiple sources of truth
leading to inconsistent behavior in array managers.

## Current Architecture Analysis

### Chunk Axis Flow (Current - Broken)

1. **BatchSolverKernel.__init__()** sets `self.chunk_axis = "run"`
2. **Solver.solve()** calls `kernel.run(..., chunk_axis=chunk_axis)`
3. **BatchSolverKernel.run()** does:
   - Calls `self.input_arrays.update(self, ...)` which internally calls
     `update_from_solver(solver)` reading `solver.chunk_axis` (still "run")
   - Calls `self.output_arrays.update(self)` similarly
   - THEN calls `self.memory_manager.allocate_queue(self, chunk_axis=chunk_axis)`
   - Memory manager computes chunks and returns via callback

The problem: InputArrays.update_from_solver() reads chunk_axis BEFORE
allocate_queue() runs. The callback _on_allocation_complete() does set
_chunk_axis correctly, but this happens AFTER update_from_solver().

### Key Files and Their Roles

#### BatchSolverKernel.py (line 343-556)

The run() method orchestrates execution:
```python
def run(self, ..., chunk_axis: str = "run") -> None:
    # Line ~429: Arrays updated BEFORE allocate_queue
    self.input_arrays.update(self, inits, params, driver_coefficients)
    self.output_arrays.update(self)

    # Line ~433: Memory manager determines actual chunking
    self.memory_manager.allocate_queue(self, chunk_axis=chunk_axis)

    # Line ~436: chunk_run uses chunk_axis and self.chunks
    chunk_params = self.chunk_run(chunk_axis, ...)
```

Note: `self.chunk_axis` is set at init (line 150) to "run" but is NOT
updated when run() receives a different chunk_axis parameter!

#### BatchInputArrays.py (line 277-301)

update_from_solver() reads chunk_axis:
```python
def update_from_solver(self, solver_instance: "BatchSolverKernel") -> None:
    self._sizes = BatchInputSizes.from_solver(solver_instance).nonzero
    self._precision = solver_instance.precision
    self._chunk_axis = solver_instance.chunk_axis  # LINE 292 - Problem!
```

This reads solver.chunk_axis which may not yet reflect the chunk_axis
passed to run().

#### BaseArrayManager.py (line 272-316)

_on_allocation_complete() callback correctly receives chunk_axis:
```python
def _on_allocation_complete(self, response: ArrayResponse) -> None:
    # ...
    self._chunks = response.chunks
    self._chunk_axis = response.chunk_axis  # LINE 315 - Correct!
```

But this runs AFTER update_from_solver(), and the order matters for
how initialise() and finalise() slice arrays.

#### BatchOutputArrays.py

OutputArrays.update_from_solver() does NOT read chunk_axis:
```python
def update_from_solver(self, solver_instance) -> Dict[str, NDArray]:
    self._sizes = BatchOutputSizes.from_solver(solver_instance).nonzero
    self._precision = solver_instance.precision
    # Note: Does NOT set _chunk_axis here!
```

This is correct - OutputArrays relies on _on_allocation_complete().

## Required Changes

### Change 1: Set chunk_axis on BatchSolverKernel at run() start

**File:** `src/cubie/batchsolving/BatchSolverKernel.py`
**Location:** run() method, early in the method

**Current behavior:** chunk_axis is only set at __init__ to "run"

**Required behavior:** Set self.chunk_axis from the run() parameter
BEFORE calling input_arrays.update() and output_arrays.update()

**Rationale:** This ensures that any code reading solver.chunk_axis
during the update phase gets the correct value.

**Implementation notes:**
- Add `self.chunk_axis = chunk_axis` early in run(), around line 400
- This should come BEFORE the calls to input_arrays.update() and
  output_arrays.update()

### Change 2: Remove chunk_axis read from InputArrays.update_from_solver()

**File:** `src/cubie/batchsolving/arrays/BatchInputArrays.py`
**Location:** update_from_solver() method, line 292

**Current behavior:** 
```python
self._chunk_axis = solver_instance.chunk_axis
```

**Required behavior:** Remove this line. Let _on_allocation_complete()
be the sole setter of _chunk_axis.

**Rationale:** _on_allocation_complete() is called AFTER the memory
manager determines the actual chunking. Reading chunk_axis earlier
creates inconsistency.

**Risk mitigation:** The _chunk_axis attribute has a default value of
"run" from the attrs field definition, so removal is safe.

## Verification Approach

### Test Cases

The existing tests in `tests/batchsolving/test_chunked_solver.py`
should pass after these changes:

1. `test_chunked_solve_produces_valid_output[run]` - Tests "run" axis
2. `test_chunked_solve_produces_valid_output[time]` - Tests "time" axis

### Expected Behavior After Fix

1. When Solver.solve(chunk_axis="time") is called:
   - BatchSolverKernel.run() sets self.chunk_axis = "time"
   - Memory manager allocate_queue() uses chunk_axis="time"
   - ArrayResponse carries chunk_axis="time"
   - _on_allocation_complete() sets _chunk_axis="time"
   - initialise() and finalise() slice along time axis

2. The chunk_axis flows consistently from user input through to
   array slicing operations.

## Integration Points

### Memory Manager Interface

The MemoryManager.allocate_queue() method accepts chunk_axis parameter
and includes it in ArrayResponse. No changes needed to memory manager.

### ArrayResponse

ArrayResponse already carries chunk_axis from memory manager to array
managers. No changes needed.

### BaseArrayManager

The _on_allocation_complete() callback already correctly sets
_chunk_axis from the response. No changes needed to base class.

## Data Structures Involved

### ArrayResponse (array_requests.py)
- `chunks: int` - Number of chunks
- `chunk_axis: str` - "run", "time", or "variable"
- `arr: dict` - Allocated device arrays

### BaseArrayManager attributes
- `_chunks: int` - Set by _on_allocation_complete()
- `_chunk_axis: str` - Set by _on_allocation_complete()

### BatchSolverKernel attributes
- `chunk_axis: str` - Currently only set at init, needs to be set in run()
- `chunks: int` - Set by _on_allocation callback

## Edge Cases

1. **Single chunk (no chunking needed):** chunks=1, chunk_axis still
   flows through but slicing is no-op

2. **Time axis chunking:** Requires special handling in kernel loop
   (already implemented in run())

3. **Default chunk_axis:** "run" is the default, should work as before

## Dependencies

- Change 1 must be applied before Change 2 is tested
- Both changes are independent of each other in terms of code
- Tests require both changes for "time" axis to work correctly

## Files Summary

| File | Change Type | Description |
|------|-------------|-------------|
| BatchSolverKernel.py | Add line | Set self.chunk_axis early in run() |
| BatchInputArrays.py | Remove line | Remove chunk_axis read from update_from_solver() |
