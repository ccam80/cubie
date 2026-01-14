# Array Manager Test Refactor - Agent Plan

## Overview

This plan addresses 7 failing tests caused by incorrect chunked array manager setup patterns in tests. The tests bypass the allocation API and manually set chunking state, causing `needs_chunked_transfer` to return `False` when `True` is expected.

## Root Cause Analysis

### Problem 1: `needs_chunked_transfer` Returns False
**Affected tests:**
- `test_basearraymanager.py::test_chunked_shape_propagates_through_allocation`
- `test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_convert_host_to_numpy_uses_needs_chunked_transfer`
- `test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_finalise_uses_needs_chunked_transfer`

**Cause:** Tests create ArrayManager instances and manually set `_chunks` and `_chunk_axis` without calling `allocate_queue()`. The `chunked_shape` on `ManagedArray` is only populated by `_on_allocation_complete()`, which is called by the memory manager after `allocate_queue()`.

**Fix:** Tests must call `allocate_queue()` with `chunk_axis` parameter to trigger proper allocation flow.

### Problem 2: Buffer Pool Tests Have Empty `_active_buffers`
**Affected tests:**
- `test_batchinputarrays.py::TestBufferPoolIntegration::test_initialise_uses_buffer_pool_when_chunked`
- `test_batchinputarrays.py::TestBufferPoolIntegration::test_release_buffers_returns_to_pool`
- `test_batchinputarrays.py::TestBufferPoolIntegration::test_reset_clears_buffer_pool_and_active_buffers`

**Cause:** Same as Problem 1 - tests manually set chunking state but `chunked_shape` is never set, so `needs_chunked_transfer` returns `False` and the buffer pool path is not taken.

**Fix:** Use proper allocation flow to set `chunked_shape`.

### Problem 3: Index Out of Bounds in test_all_lower_plumbing
**Affected test:**
- `test_SolverKernel.py::test_all_lower_plumbing`

**Cause:** The test specifies `saved_observable_indices: [0, 1, 2]` but the default test system (nonlinear) may have fewer observables.

**Fix:** Ensure indices match the system's observable count, or use `None` to save all.

## Component Descriptions

### Component 1: Chunked Array Manager Test Fixture

**Purpose:** Provide a reusable fixture that creates an array manager with proper chunked allocation.

**Location:** Tests that need chunked array managers should use the solver fixture and call `allocate_queue()` with appropriate parameters.

**Behavior:**
1. Accept parameters for chunk count and chunk axis
2. Create input/output arrays through solver pattern
3. Call `allocate_queue()` to trigger `_on_allocation_complete()` callback
4. Verify `chunked_shape` is set on managed arrays

**Expected interactions:**
- Tests request the fixture with parameters
- Fixture uses solver fixture to get properly configured manager
- Manager's `_chunks` and `_chunk_axis` are set by allocation response
- `ManagedArray.chunked_shape` is populated

### Component 2: Buffer Pool Integration Test Pattern

**Purpose:** Correctly test buffer pool acquisition and release in chunked mode.

**Behavior:**
1. Use solver fixture to get InputArrays instance
2. Call `update()` with test data
3. Call `allocate_queue()` with chunk settings that force chunking
4. Call `initialise()` with chunk index
5. Verify `_active_buffers` populated

**Key integration point:** The `initialise()` method checks `host_obj.needs_chunked_transfer` to decide whether to use buffer pool. This only returns `True` if `chunked_shape` differs from `shape`.

### Component 3: SolverKernel Test Fixture Usage

**Purpose:** Ensure tests use fixtures instead of direct instantiation.

**Current pattern (incorrect):**
```python
kernel = BatchSolverKernel(system, ...)
```

**Correct pattern:**
```python
# Use fixture
def test_something(solverkernel_mutable):
    kernel = solverkernel_mutable
    kernel.update({...})
```

## Test File Changes

### test_basearraymanager.py

**test_chunked_shape_propagates_through_allocation:**
- Currently creates ConcreteArrayManager and simulates allocation response manually
- Issue: The test needs to trigger proper allocation flow
- Fix: Create ArrayResponse with chunked_shapes and call `_on_allocation_complete()` correctly, OR use actual MemoryManager allocation

### test_batchinputarrays.py (TestBufferPoolIntegration)

**All three failing tests:**
- Currently call `from_solver()` then manually set `_chunks` and `_chunk_axis`
- Issue: `chunked_shape` is not set on ManagedArrays
- Fix: After `update()`, call `allocate_queue()` with `chunk_axis` parameter to get proper chunked allocation

**Required pattern:**
```python
input_arrays = InputArrays.from_solver(solver)
input_arrays.update(solver, inits, params, driver_coefficients)
# Key: use allocate_queue to set up chunked allocation
default_memmgr.allocate_queue(input_arrays, chunk_axis="run")
# Now chunked_shape is set and needs_chunked_transfer will work
```

### test_batchoutputarrays.py (TestNeedsChunkedTransferBranching)

**test_convert_host_to_numpy_uses_needs_chunked_transfer:**
- Creates ArrayResponse manually but doesn't properly simulate allocation
- Fix: Call `_on_allocation_complete()` with proper chunked_shapes dict

**test_finalise_uses_needs_chunked_transfer:**
- Same issue as above
- Fix: Ensure allocation response sets `chunked_shape` on device arrays

### test_SolverKernel.py

**test_all_lower_plumbing:**
- Uses `saved_observable_indices: [0, 1, 2]` for any system
- Issue: Some systems may have fewer than 3 observables
- Fix: Check system sizes before specifying indices, or use `None`

### test_chunk_axis_property.py

**Commented tests in TestChunkAxisInRun and TestUpdateFromSolverChunkAxis:**
- Currently commented because they instantiate BatchSolverKernel directly
- Fix: Use solver fixture pattern instead

**Reinstated pattern:**
```python
def test_run_sets_chunk_axis_on_arrays(self, solver_mutable, precision):
    solver = solver_mutable
    inits = np.ones((solver.system_sizes.states, 1), dtype=precision)
    params = np.ones((solver.system_sizes.parameters, 1), dtype=precision)
    
    solver.solve(inits, params, duration=0.1, chunk_axis="run")
    
    assert solver.kernel.input_arrays._chunk_axis == "run"
```

## Dependencies and Imports

### Required imports for test files:
- `from cubie.memory import default_memmgr`
- `from cubie.memory.mem_manager import ArrayResponse`

### Fixture dependencies:
- `solver` or `solver_mutable` from conftest.py
- `precision` from conftest.py
- `sample_input_arrays` from test file local fixtures

## Edge Cases

1. **System with no observables:** Some tests may fail if the system has no observables and tests specify observable indices.

2. **Single chunk scenario:** When `chunks == 1`, `needs_chunked_transfer` should return `False` because `chunked_shape == shape`.

3. **Unchunkable arrays:** Arrays with `is_chunked=False` (like driver_coefficients) should never use buffer pool.

## Unused Properties/Attributes to Remove

After implementing fixes, scan for:
1. Any `_chunk_axis` setter that is unused
2. Any manual `_chunks` assignment patterns
3. Dead code in allocation paths

## Verification Steps

1. Run each failing test individually after fix
2. Verify `needs_chunked_transfer` returns expected value
3. Verify `_active_buffers` is populated in buffer pool tests
4. Ensure reinstated tests in test_chunk_axis_property.py pass
5. Run full test suite to catch regressions
