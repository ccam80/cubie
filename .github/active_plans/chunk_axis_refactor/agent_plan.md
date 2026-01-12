# chunk_axis Refactoring: Agent Implementation Plan

## Overview

This document provides detailed technical specifications for the chunk_axis refactoring. The goal is to establish `BaseArrayManager._chunk_axis` as the single source of truth, with `BatchSolverKernel` providing coordinated access through a property/setter pattern.

---

## Component 1: BatchSolverKernel Property and Setter

### Current State
- `BatchSolverKernel.__init__()` sets `self.chunk_axis = "run"` as a public attribute (line 150)
- `Solver.chunk_axis` property returns `self.kernel.chunk_axis` (line 874)
- The attribute is never updated after initial assignment, even when user passes different value to `run()`

### Expected Behavior

**Property (getter):**
1. Access `self.input_arrays._chunk_axis`
2. Access `self.output_arrays._chunk_axis`
3. If values match, return the value
4. If values don't match, raise `ValueError` with descriptive message

**Property (setter):**
1. Accept new chunk_axis value (must be "run", "variable", or "time")
2. Validate the value (optional - attrs already validates on array managers)
3. Set `self.input_arrays._chunk_axis = value`
4. Set `self.output_arrays._chunk_axis = value`

### Architectural Changes

**Remove:**
- Line 150: `self.chunk_axis = "run"` - remove this line

**Add:**
```python
@property
def chunk_axis(self) -> str:
    """Current chunking axis.
    
    Returns the chunk_axis value from the array managers, validating
    that input and output arrays have consistent values.
    
    Returns
    -------
    str
        The axis along which arrays are chunked ("run", "variable", or "time").
    
    Raises
    ------
    ValueError
        If input_arrays and output_arrays have different chunk_axis values.
    """
    input_axis = self.input_arrays._chunk_axis
    output_axis = self.output_arrays._chunk_axis
    if input_axis != output_axis:
        raise ValueError(
            f"Inconsistent chunk_axis: input_arrays has '{input_axis}', "
            f"output_arrays has '{output_axis}'"
        )
    return input_axis

@chunk_axis.setter
def chunk_axis(self, value: str) -> None:
    """Set chunk_axis on both input and output array managers."""
    self.input_arrays._chunk_axis = value
    self.output_arrays._chunk_axis = value
```

### Integration Points

- `Solver.chunk_axis` property (line 874 in solver.py) - **No changes needed**, will work with new property
- `BatchSolverKernel.run()` method - Add setter call before array operations
- `BatchSolverKernel.chunk_run()` method - Continue passing chunk_axis as parameter (no change)

---

## Component 2: BatchSolverKernel.run() Update

### Current State
- `run()` receives `chunk_axis` as parameter (line 353)
- Parameter is passed directly to `memory_manager.allocate_queue()` (line 433)
- Parameter is passed to `chunk_run()` (line 437)
- The value is never stored on the kernel or propagated to array managers until callback

### Expected Behavior

Early in `run()`, before calling `update()` on array managers:
1. Set `self.chunk_axis = chunk_axis` using the new setter
2. This ensures both array managers have the correct value before any operations

### Location for Update

The setter call should be placed after timing parameter storage (around line 403) and before array update calls (line 429):

```python
# After time parameters are stored
self._duration = duration
self._warmup = warmup
self._t0 = t0

# Set chunk_axis on both array managers
self.chunk_axis = chunk_axis

# Queue allocations (these now have correct chunk_axis)
self.input_arrays.update(self, inits, params, driver_coefficients)
self.output_arrays.update(self)
```

---

## Component 3: BatchInputArrays.update_from_solver() Cleanup

### Current State
- Line 292: `self._chunk_axis = solver_instance.chunk_axis`
- This reads chunk_axis from the solver/kernel and stores on the array manager

### Expected Behavior

**Remove this line.** The chunk_axis is now set directly by the kernel's setter before `update_from_solver()` is called. Having this line would be redundant (though harmless since it reads the same value back).

### Reasoning

With the new flow:
1. `kernel.run()` sets `chunk_axis` on both array managers via setter
2. `kernel.run()` calls `input_arrays.update(self, ...)` 
3. `update()` calls `update_from_solver(solver_instance)`
4. At this point, `self._chunk_axis` is already correct

Removing the redundant assignment simplifies the code and removes the dependency on `solver_instance.chunk_axis`.

---

## Component 4: BaseArrayManager._on_allocation_complete() Behavior

### Current State
- Line 315: `self._chunk_axis = response.chunk_axis`
- This updates chunk_axis from the ArrayResponse after allocation

### Expected Behavior

**Keep this line.** The allocation response may have a different chunk_axis if the memory manager decided to chunk differently. The response should still update the array manager's value.

However, in the new design, the input and output arrays should already have matching values before allocation. The allocation response should also have matching values since both arrays are allocated with the same chunk_axis parameter.

### Consistency Validation

After allocation, the following should all be equal:
- `input_arrays._chunk_axis`
- `output_arrays._chunk_axis`
- `response.chunk_axis` (from MemoryManager)

The `_on_allocation_complete` callback updates each array manager with `response.chunk_axis`, which should match what was already set. This provides a double-check but doesn't change the value in normal operation.

---

## Component 5: Integration with MemoryManager

### Current State
- `allocate_queue(triggering_instance, chunk_axis="run")` receives chunk_axis as parameter
- `single_request(instance, requests, chunk_axis="run")` receives chunk_axis as parameter
- Both pass chunk_axis to `chunk_arrays()` and include in `ArrayResponse`

### Expected Behavior

**No changes to MemoryManager.** The memory manager receives chunk_axis as a parameter from `BatchSolverKernel.run()` and this flow remains unchanged. The manager doesn't need to access the kernel or array managers to get chunk_axis.

---

## Expected Interactions Between Components

### Initialization Flow
1. `BatchSolverKernel.__init__()` creates `InputArrays` and `OutputArrays`
2. Both array managers initialize with `_chunk_axis = "run"` (attrs default)
3. No public attribute on kernel (removed)

### Runtime Flow (kernel.run())
1. User calls `solver.solve(chunk_axis="time")`
2. Solver calls `kernel.run(chunk_axis="time")`
3. `run()` stores timing parameters (duration, warmup, t0)
4. `run()` calls `self.chunk_axis = "time"` (setter)
   - Sets `input_arrays._chunk_axis = "time"`
   - Sets `output_arrays._chunk_axis = "time"`
5. `run()` calls `input_arrays.update(self, ...)` and `output_arrays.update(self)`
   - `update_from_solver()` no longer touches chunk_axis
6. `run()` calls `memory_manager.allocate_queue(self, chunk_axis="time")`
7. MemoryManager creates ArrayResponse with chunk_axis="time"
8. Callbacks update array managers: `_on_allocation_complete(response)`
   - `_chunk_axis` set from response (should match)
9. `run()` calls `chunk_run("time", ...)`
10. Loop iterates, calling `initialise()` and `finalise()` which use `self._chunk_axis`

### Property Access Flow
1. User calls `solver.chunk_axis`
2. `Solver.chunk_axis` returns `self.kernel.chunk_axis`
3. `BatchSolverKernel.chunk_axis` property:
   - Reads `input_arrays._chunk_axis` → "time"
   - Reads `output_arrays._chunk_axis` → "time"
   - Values match → returns "time"

---

## Data Structures

### BaseArrayManager._chunk_axis
- Type: `str`
- Validator: `attrsval_in(["run", "variable", "time"])`
- Default: `"run"`
- Storage: Private attribute with underscore prefix
- Access: Direct attribute access (`self._chunk_axis`)

### ArrayResponse.chunk_axis
- Type: `str`
- Validator: `val.in_(["run", "variable", "time"])`
- Default: `"run"`
- Storage: Frozen attrs class field
- Access: Attribute access (`response.chunk_axis`)

---

## Dependencies and Imports

### BatchSolverKernel.py
No new imports required. The property/setter use existing types.

### BatchInputArrays.py
No changes to imports. Remove line 292.

### BatchOutputArrays.py
No changes required.

---

## Edge Cases

### Edge Case 1: Array Managers Not Yet Created
**Scenario:** Accessing `kernel.chunk_axis` before array managers are initialized.
**Handling:** Not possible - array managers are created in `__init__` before any external access.

### Edge Case 2: Inconsistent Values After Manual Modification
**Scenario:** Someone directly modifies `input_arrays._chunk_axis` without using the setter.
**Handling:** Property getter raises `ValueError` with clear message explaining the inconsistency.

### Edge Case 3: Multiple Solve Calls with Different chunk_axis
**Scenario:** User calls `solve(chunk_axis="run")` then `solve(chunk_axis="time")`.
**Handling:** Each `run()` call sets chunk_axis via setter, ensuring correct value for that run.

### Edge Case 4: chunk_axis Not in Array's stride_order
**Scenario:** Array has stride_order `("variable", "run")` but chunk_axis is `"time"`.
**Handling:** Existing behavior - `initialise()`/`finalise()` skip chunking for that array. No change needed.

### Edge Case 5: Allocation Callback with Different Value
**Scenario:** MemoryManager returns ArrayResponse with different chunk_axis than what was set.
**Handling:** This shouldn't happen (MemoryManager passes through the value). If it did, `_on_allocation_complete` would update with the response value, potentially causing property inconsistency on next access.

---

## Validation Points

### Unit Test: Property Returns Consistent Value
- Set chunk_axis via setter
- Verify property returns same value
- Verify both array managers have same value

### Unit Test: Property Raises on Inconsistency
- Manually set different values on input/output arrays
- Access property
- Verify ValueError is raised

### Unit Test: Setter Updates Both Managers
- Access initial values (should be "run")
- Set via setter to "time"
- Verify both managers updated

### Integration Test: End-to-End Flow
- Create solver
- Call solve with chunk_axis="time"
- Verify chunk_axis property returns "time" after solve
- Verify array slicing uses correct axis

---

## Files to Modify

1. **src/cubie/batchsolving/BatchSolverKernel.py**
   - Remove line 150 (`self.chunk_axis = "run"`)
   - Add property getter after other properties
   - Add property setter after getter
   - Add setter call in `run()` method

2. **src/cubie/batchsolving/arrays/BatchInputArrays.py**
   - Remove line 292 (`self._chunk_axis = solver_instance.chunk_axis`)

3. **tests/** - Add tests for new property behavior
   - Property consistency validation
   - Setter synchronization
   - Error on inconsistency

---

## Migration Notes

- No changes to public API (`solver.solve()`, `solver.chunk_axis`)
- Property access pattern unchanged for external users
- Internal direct attribute access (`kernel.chunk_axis = x`) continues to work via setter
