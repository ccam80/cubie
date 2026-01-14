# Implementation Task List
# Feature: Array Manager Test Refactor
# Plan Reference: .github/active_plans/array_manager_test_refactor/agent_plan.md

## Task Group 1: Fix test_basearraymanager.py::test_chunked_shape_propagates_through_allocation
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: tests/batchsolving/arrays/test_basearraymanager.py (lines 1674-1768 - the test)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 344-390 - _on_allocation_complete method)
- File: src/cubie/memory/array_requests.py (entire file - ArrayResponse class)

**Input Validation Required**:
- None - this is a test fix, not implementation code

**Tasks**:
1. **Fix test_chunked_shape_propagates_through_allocation test**
   - File: tests/batchsolving/arrays/test_basearraymanager.py
   - Action: Modify
   - Details:
     The test at line 1674 creates an ArrayResponse and calls `_on_allocation_complete` 
     but the host arrays' `needs_chunked_transfer` returns False because the test does not
     properly set up the host array shapes to differ from the chunked shapes.
     
     The issue is that `needs_chunked_transfer` compares `shape` with `chunked_shape`.
     The host array's `shape` property returns `_array.shape` if `_array` is not None.
     The test sets host arrays with shape `host_shape = (10, 3, 5)` and expects 
     `needs_chunked_transfer` to return True when `chunked_shape = (10, 3, 2)`.
     
     However, the test is checking `manager.host.arr1.needs_chunked_transfer` but the
     host arrays are created with `default_shape=host_shape` and then assigned arrays
     with `host_shape`. After `_on_allocation_complete`, the `chunked_shape` is set
     but the `shape` property returns the actual array shape which equals the full shape.
     
     The fix is to ensure the test is checking the correct condition:
     - After allocation, device arrays have `chunked_shape` set
     - `needs_chunked_transfer` on host arrays compares host array's shape (full) 
       vs chunked_shape (smaller) -> should return True
     
     Current problem: The test assigns arrays with shape `host_shape = (10, 3, 5)` to 
     host.arr1.array, so `host.arr1.shape` returns `(10, 3, 5)`. When `chunked_shape` 
     is set to `(10, 3, 2)`, `needs_chunked_transfer` compares `(10, 3, 5) != (10, 3, 2)`
     which should return True.
     
     On closer inspection, the test is correct but the `_on_allocation_complete` method
     sets chunked_shape only for arrays in `_needs_reallocation`. The test does set
     `manager._needs_reallocation = ["arr1", "arr2"]` before calling the method.
     
     The issue is likely that the assertions at lines 1763-1764 check host arrays but
     the `_on_allocation_complete` method sets chunked_shape on both host AND device
     containers (line 376-378 in BaseArrayManager.py).
     
     Wait - re-reading the code at lines 374-379:
     ```python
     if array_label in response.chunked_shapes:
         for container in (self.device, self.host):
             array = container.get_managed_array(array_label)
             array.chunked_shape = chunked_shapes[array_label]
             array.chunked_slice_fn = chunked_slices[array_label]
     ```
     
     This sets chunked_shape on BOTH containers. So the test should work...
     
     Let me trace through: The test creates host arrays with shape (10, 3, 5),
     then calls `_on_allocation_complete` with chunked_shapes = {arr1: (10,3,2), arr2: (10,3,2)}.
     After this call, both host.arr1.chunked_shape and device.arr1.chunked_shape should be (10,3,2).
     Then `host.arr1.needs_chunked_transfer` compares shape (10,3,5) != chunked_shape (10,3,2) -> True.
     
     The test assertion at line 1763 is `assert manager.host.arr1.needs_chunked_transfer is True`
     
     This should pass IF the chunked_shapes dict in the response includes "arr1" and "arr2".
     Looking at the test more carefully at lines 1738-1743:
     ```python
     chunked_shapes = {
         "arr1": expected_chunked_shape,
         "arr2": expected_chunked_shape,
     }
     ```
     
     And the response is created with these chunked_shapes. This should work.
     
     The actual issue might be that `response.chunked_slices` is not provided in the test,
     and `_on_allocation_complete` at line 369 does:
     ```python
     chunked_slices = response.chunked_slices
     ```
     If `chunked_slices` is None or empty, line 379 might fail.
     
     Looking at ArrayResponse, `chunked_slices` has a factory that returns an empty dict.
     So accessing `response.chunked_slices` returns `{}`, and then line 379 tries to get
     `chunked_slices[array_label]` which would raise KeyError.
     
     But wait, line 379 is inside the `if array_label in response.chunked_shapes:` block,
     so it only runs if array_label is in chunked_shapes. But it unconditionally accesses
     chunked_slices[array_label] which may not exist.
     
     This is a BUG in the source code or the test needs to provide chunked_slices.
     
     Let me check if ArrayResponse.chunked_slices defaults properly...
     Actually, looking at the test failure message: "needs_chunked_transfer returns False".
     This means the assertion at line 1763 fails.
     
     The test creates response with chunked_shapes but NOT chunked_slices. When
     `_on_allocation_complete` runs, at line 379 it does:
     `array.chunked_slice_fn = chunked_slices[array_label]`
     
     If chunked_slices is `{}`, this raises KeyError for "arr1". But the error message
     says "needs_chunked_transfer returns False", not a KeyError.
     
     Perhaps the source code was updated to handle missing chunked_slices, or the test
     is catching the exception silently.
     
     Looking at the actual ArrayResponse class to understand chunked_slices default...
     The test should provide chunked_slices in the response, or the source should handle
     missing entries gracefully.
     
     **FIX**: Update the test to provide `chunked_slices` in the ArrayResponse.
     The chunked_slices should be callable functions that return slices for each array.
     For testing purposes, we can provide simple lambda functions that return a slice.
     
   - Edge cases: Empty chunked_slices dict, None chunked_shapes
   - Integration: Test validates _on_allocation_complete callback

**Tests to Create**:
- None - this is fixing an existing test

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py::test_chunked_shape_propagates_through_allocation

**Outcomes**: 
- Files Modified:
  * tests/batchsolving/arrays/test_basearraymanager.py (15 lines changed)
- Functions/Methods Added/Modified:
  * Added `make_slice_fn` helper within test function
  * Added `chunked_slices` dict to ArrayResponse creation
- Implementation Summary:
  The test was missing `chunked_slices` in the ArrayResponse. When `_on_allocation_complete`
  tried to access `chunked_slices[array_label]`, it raised KeyError which was caught,
  but this skipped setting `chunked_shape` on the host container. Added slice functions
  to the response so the allocation callback completes successfully.
- Issues Flagged: None

---

## Task Group 2: Fix test_batchinputarrays.py Buffer Pool Integration Tests
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/batchsolving/arrays/test_batchinputarrays.py (lines 466-668 - TestBufferPoolIntegration class)
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (entire file - InputArrays class)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 344-390 - _on_allocation_complete)
- File: src/cubie/memory/mem_manager.py (lines 254-450 - MemoryManager.allocate_queue)
- File: tests/conftest.py (lines 759-782 - solver and solver_mutable fixtures)

**Input Validation Required**:
- None - these are test fixes

**Tasks**:
1. **Fix test_initialise_uses_buffer_pool_when_chunked test**
   - File: tests/batchsolving/arrays/test_batchinputarrays.py
   - Action: Modify
   - Details:
     The test at line 479 manually sets `_chunks` and `_chunk_axis` and then manually
     sets `chunked_shape` on device arrays. However, the test doesn't allocate device
     arrays through `allocate_queue()`, so the device arrays are still the default
     1x1 size-1 arrays from `__attrs_post_init__`.
     
     The issue is that `input_arrays.device.initial_values.shape` returns the actual
     array shape (which is the default small array), not the expected full shape.
     So when `chunked_shape` is set to a smaller value, `needs_chunked_transfer`
     compares the tiny default shape with chunked_shape and may return False.
     
     **FIX**: The test must call `default_memmgr.allocate_queue(input_arrays, chunk_axis="run")`
     BEFORE attempting to use the arrays. However, since this is testing the buffer pool
     behavior during `initialise()`, we need a different approach:
     
     1. Call `input_arrays.update(...)` to populate host arrays
     2. Call `default_memmgr.allocate_queue(input_arrays, chunk_axis="run")` to allocate
        device arrays with proper chunked shapes computed by MemoryManager
     3. The `_on_allocation_complete` callback will set `chunked_shape` on all ManagedArrays
     4. Now `needs_chunked_transfer` will return True for chunked arrays
     5. Call `initialise(host_indices)` and verify `_active_buffers` is populated
     
     However, the test creates a new InputArrays via `InputArrays.from_solver(solver)`
     which doesn't use the session-scoped solver's memory manager registration.
     
     The pattern from working tests in this file (e.g., test_allocation_and_getters_not_none)
     shows the correct approach:
     ```python
     input_arrays_manager.update(solver, ...)
     default_memmgr.allocate_queue(input_arrays_manager, chunk_axis="run")
     ```
     
     The failing test creates its own InputArrays and then manually configures chunking
     without going through the allocation API.
     
     **SOLUTION**: Modify the test to:
     1. Use `InputArrays.from_solver(solver)` to create the instance
     2. Call `update(solver, inits, params, drivers)` with test data
     3. Force chunking by setting a small VRAM cap OR by using a memory manager in
        active mode with a small proportion
     4. Call `default_memmgr.allocate_queue(input_arrays, chunk_axis="run")` which
        will compute chunked_shapes and call `_on_allocation_complete`
     5. Now `needs_chunked_transfer` returns True for chunked arrays
     6. Call `initialise(host_indices)` and verify buffer pool behavior
     
     The challenge is forcing chunking. MemoryManager computes chunks based on:
     - Available memory
     - Requested array sizes
     - Instance's memory cap
     
     For the test, we can either:
     a) Use a mock MemoryManager with controlled chunking
     b) Create an ArrayResponse manually and call `_on_allocation_complete` directly
     c) Configure the memory manager to force chunking
     
     Option (b) is cleanest for unit testing - create the response with the desired
     chunked_shapes and call `_on_allocation_complete` directly, similar to how
     `test_chunked_shape_propagates_through_allocation` works.
     
     ```python
     def test_initialise_uses_buffer_pool_when_chunked(
         self, solver, sample_input_arrays, precision
     ):
         input_arrays = InputArrays.from_solver(solver)
         input_arrays.update(
             solver,
             sample_input_arrays["initial_values"],
             sample_input_arrays["parameters"],
             sample_input_arrays["driver_coefficients"],
         )
         
         # Allocate device arrays first
         default_memmgr.allocate_queue(input_arrays, chunk_axis="run")
         
         # Now create a chunked scenario by calling _on_allocation_complete
         # with chunked_shapes that differ from full shapes
         num_runs = sample_input_arrays["initial_values"].shape[1]
         chunk_size = max(1, num_runs // 3)
         
         chunked_shapes = {}
         chunked_slices = {}
         for name, device_slot in input_arrays.device.iter_managed_arrays():
             full_shape = device_slot.shape
             if "run" in device_slot.stride_order:
                 run_idx = device_slot.stride_order.index("run")
                 chunked = list(full_shape)
                 chunked[run_idx] = chunk_size
                 chunked_shapes[name] = tuple(chunked)
                 chunked_slices[name] = lambda idx, c=chunk_size, r=run_idx: (
                     slice(None) if i != r else slice(idx*c, (idx+1)*c)
                     for i in range(len(full_shape))
                 )
             else:
                 chunked_shapes[name] = full_shape
                 chunked_slices[name] = lambda idx: (slice(None),) * len(full_shape)
         
         # Set chunks > 1 to trigger chunked path
         input_arrays._chunks = 3
         
         # Manually set chunked_shape on device and host arrays
         for name, shape in chunked_shapes.items():
             input_arrays.device.get_managed_array(name).chunked_shape = shape
             input_arrays.host.get_managed_array(name).chunked_shape = shape
         
         # Clear existing active buffers
         input_arrays._active_buffers.clear()
         
         # Call initialise with chunk indices
         host_indices = slice(0, chunk_size)
         input_arrays.initialise(host_indices)
         
         # Verify buffers were acquired
         assert len(input_arrays._active_buffers) > 0
     ```
     
   - Edge cases: num_runs == 1 (no chunking possible), empty sample arrays
   - Integration: Tests buffer pool acquisition during chunked transfers

2. **Fix test_release_buffers_returns_to_pool test**
   - File: tests/batchsolving/arrays/test_batchinputarrays.py
   - Action: Modify
   - Details:
     Same issue as above - manually setting `_chunks` and `_chunk_axis` doesn't
     populate `chunked_shape` on ManagedArrays. Apply the same fix pattern:
     allocate through the proper API, then set chunked_shapes to force
     `needs_chunked_transfer` to return True.
   - Edge cases: Releasing empty buffer list
   - Integration: Tests buffer pool release mechanism

3. **Fix test_reset_clears_buffer_pool_and_active_buffers test**
   - File: tests/batchsolving/arrays/test_batchinputarrays.py
   - Action: Modify
   - Details:
     Same pattern - ensure chunked_shape is set so initialise actually
     acquires buffers that can then be verified as cleared by reset.
   - Edge cases: Reset with no active buffers
   - Integration: Tests buffer pool and active buffers cleanup

**Tests to Create**:
- None - these are fixing existing tests

**Tests to Run**:
- tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_initialise_uses_buffer_pool_when_chunked
- tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_release_buffers_returns_to_pool
- tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_reset_clears_buffer_pool_and_active_buffers

**Outcomes**: 
- Files Modified:
  * tests/batchsolving/arrays/test_batchinputarrays.py (80 lines changed)
- Functions/Methods Added/Modified:
  * Fixed test_initialise_uses_buffer_pool_when_chunked
  * Fixed test_release_buffers_returns_to_pool
  * Fixed test_reset_clears_buffer_pool_and_active_buffers
  * Fixed test_buffers_reused_across_chunks
  * Fixed test_non_chunked_uses_direct_pinned
- Implementation Summary:
  Tests were setting chunked_shape only on device arrays but needs_chunked_transfer
  is checked on host arrays. Added chunked_shape and chunked_slice_fn on both
  host and device ManagedArrays. Changed initialise() calls to use chunk_index
  (int) instead of slice. Added make_slice_fn helper functions in each test.
- Issues Flagged: None

---

## Task Group 3: Fix test_batchoutputarrays.py TestNeedsChunkedTransferBranching Tests
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/batchsolving/arrays/test_batchoutputarrays.py (lines 657-801 - TestNeedsChunkedTransferBranching class)
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (entire file)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 344-390 - _on_allocation_complete)
- File: src/cubie/memory/array_requests.py (ArrayResponse class)

**Input Validation Required**:
- None - these are test fixes

**Tasks**:
1. **Fix test_convert_host_to_numpy_uses_needs_chunked_transfer test**
   - File: tests/batchsolving/arrays/test_batchoutputarrays.py
   - Action: Modify
   - Details:
     The test at line 660 creates an ArrayResponse with chunked_shapes and calls
     `_on_allocation_complete`, but the response's `arr` dict contains the host
     arrays (`output_arrays_manager.host.get_array(name)`), not device arrays.
     
     The `_on_allocation_complete` method expects device arrays in `response.arr`
     and attaches them to `self.device` container. Passing host arrays causes
     the device container to reference host arrays, which is incorrect.
     
     Additionally, the response needs to include `chunked_slices` (callables that
     return slices) to avoid KeyError when `_on_allocation_complete` tries to
     access them.
     
     **FIX**:
     1. Create the response with device arrays in the `arr` dict
     2. Include `chunked_slices` dict with callable slice functions
     3. Ensure `_needs_reallocation` is populated before calling `_on_allocation_complete`
     
     ```python
     def test_convert_host_to_numpy_uses_needs_chunked_transfer(
         self, output_arrays_manager, solver, test_memory_manager
     ):
         from cubie.memory.mem_manager import ArrayResponse
         
         # Allocate first to set up arrays
         output_arrays_manager.update(solver)
         test_memory_manager.allocate_queue(
             output_arrays_manager, chunk_axis="run"
         )
         
         # Get num_runs from allocated array
         num_runs = output_arrays_manager.state.shape[2]
         chunk_size = max(1, num_runs // 2)
         
         # Build chunked_shapes and chunked_slices for all arrays
         chunked_shapes = {}
         chunked_slices = {}
         for name, slot in output_arrays_manager.device.iter_managed_arrays():
             if not slot.is_chunked:
                 chunked_shapes[name] = slot.shape
                 chunked_slices[name] = lambda idx, s=slot.shape: tuple(
                     slice(None) for _ in s
                 )
             else:
                 if "run" in slot.stride_order:
                     axis_idx = slot.stride_order.index("run")
                     chunked_shape = tuple(
                         chunk_size if i == axis_idx else dim
                         for i, dim in enumerate(slot.shape)
                     )
                     chunked_shapes[name] = chunked_shape
                     # Create slice function for this array
                     def make_slice_fn(axis, chunk_sz):
                         def fn(idx):
                             slices = [slice(None)] * len(slot.shape)
                             slices[axis] = slice(idx * chunk_sz, (idx + 1) * chunk_sz)
                             return tuple(slices)
                         return fn
                     chunked_slices[name] = make_slice_fn(axis_idx, chunk_size)
                 else:
                     chunked_shapes[name] = slot.shape
                     chunked_slices[name] = lambda idx, s=slot.shape: tuple(
                         slice(None) for _ in s
                     )
         
         # Create response with DEVICE arrays
         response = ArrayResponse(
             arr={
                 name: output_arrays_manager.device.get_array(name)
                 for name in output_arrays_manager.device.array_names()
             },
             chunks=2,
             chunk_axis="run",
             chunked_shapes=chunked_shapes,
             chunked_slices=chunked_slices,
         )
         
         # Populate _needs_reallocation so _on_allocation_complete processes arrays
         output_arrays_manager._needs_reallocation = list(
             output_arrays_manager.device.array_names()
         )
         
         # Trigger allocation complete to store chunked_shapes
         output_arrays_manager._on_allocation_complete(response)
         
         # Now verify needs_chunked_transfer works correctly
         host_state = output_arrays_manager.host.get_managed_array("state")
         assert host_state.needs_chunked_transfer is True
         
         host_status = output_arrays_manager.host.get_managed_array("status_codes")
         assert host_status.needs_chunked_transfer is False
     ```
     
   - Edge cases: Arrays with is_chunked=False (status_codes)
   - Integration: Tests memory type conversion based on chunking

2. **Fix test_finalise_uses_needs_chunked_transfer test**
   - File: tests/batchsolving/arrays/test_batchoutputarrays.py
   - Action: Modify
   - Details:
     Same pattern as above. The test creates ArrayResponse with host arrays
     and without chunked_slices. Need to:
     1. Use device arrays in response.arr
     2. Include chunked_slices dict
     3. Populate _needs_reallocation before calling _on_allocation_complete
     
   - Edge cases: Empty _pending_buffers after finalise
   - Integration: Tests buffer acquisition during chunked finalise

**Tests to Create**:
- None - these are fixing existing tests

**Tests to Run**:
- tests/batchsolving/arrays/test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_convert_host_to_numpy_uses_needs_chunked_transfer
- tests/batchsolving/arrays/test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_finalise_uses_needs_chunked_transfer

**Outcomes**: 
- Files Modified:
  * tests/batchsolving/arrays/test_batchoutputarrays.py (60 lines changed)
- Functions/Methods Added/Modified:
  * Fixed test_convert_host_to_numpy_uses_needs_chunked_transfer
  * Fixed test_finalise_uses_needs_chunked_transfer
- Implementation Summary:
  Tests were missing chunked_slices in ArrayResponse and not setting
  _needs_reallocation before calling _on_allocation_complete. Added
  chunked_slices dict with callable slice functions and set
  _needs_reallocation. Changed ArrayResponse import to use
  cubie.memory.array_requests module. Fixed finalise() call to use
  chunk_index (int) instead of slice.
- Issues Flagged: None

---

## Task Group 4: Fix test_SolverKernel.py::test_all_lower_plumbing
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: tests/batchsolving/test_SolverKernel.py (lines 149-265 - test_all_lower_plumbing)
- File: tests/system_fixtures.py (system builder functions)
- File: tests/conftest.py (lines 334-366 - system fixture)

**Input Validation Required**:
- None - this is a test fix

**Tasks**:
1. **Fix test_all_lower_plumbing IndexError**
   - File: tests/batchsolving/test_SolverKernel.py
   - Action: Modify
   - Details:
     The test at line 149 sets `saved_observable_indices: [0, 1, 2]` in `new_settings`.
     This assumes the system has at least 3 observables, but the default "nonlinear"
     system may have fewer.
     
     Looking at the test, it uses `solverkernel_mutable` which is built from the
     `system` fixture. The default system is "nonlinear" (from `build_three_state_nonlinear_system`).
     
     The fix is to either:
     a) Check the system's observable count and limit indices accordingly
     b) Use `None` for indices (saves all)
     c) Parametrize the test with a system that has enough observables
     
     Option (a) is safest as it makes the test robust:
     
     ```python
     def test_all_lower_plumbing(
         system,
         solverkernel_mutable,
         step_controller_settings,
         algorithm_settings,
         precision,
         driver_array,
     ):
         solverkernel = solverkernel_mutable
         
         # Limit indices to actual system sizes
         n_states = system.sizes.states
         n_obs = system.sizes.observables
         
         new_settings = {
             "dt_min": 0.0001,
             "dt_max": 0.01,
             "save_every": 0.01,
             "summarise_every": 0.1,
             "sample_summaries_every": 0.05,
             "atol": 1e-2,
             "rtol": 1e-1,
             "saved_state_indices": list(range(min(3, n_states))),
             "saved_observable_indices": list(range(min(3, n_obs))),
             "summarised_state_indices": [0] if n_states > 0 else [],
             "summarised_observable_indices": [0] if n_obs > 0 else [],
             "output_types": [
                 "state",
                 "observables", 
                 "mean",
                 "max",
                 "rms",
                 "peaks[3]",
             ],
         }
         # ... rest of test
     ```
     
     Also need to update `output_settings` dict to match:
     ```python
     output_settings = {
         "saved_state_indices": np.asarray(list(range(min(3, n_states)))),
         "saved_observable_indices": np.asarray(list(range(min(3, n_obs)))),
         "summarised_state_indices": np.asarray([0] if n_states > 0 else []),
         "summarised_observable_indices": np.asarray([0] if n_obs > 0 else []),
         "output_types": [...],
     }
     ```
     
   - Edge cases: System with 0 observables, system with 0 states
   - Integration: Tests configuration propagation through solver hierarchy

**Tests to Create**:
- None - this is fixing an existing test

**Tests to Run**:
- tests/batchsolving/test_SolverKernel.py::test_all_lower_plumbing

**Outcomes**: 
- Files Modified:
  * tests/batchsolving/test_SolverKernel.py (25 lines changed)
- Functions/Methods Added/Modified:
  * Modified test_all_lower_plumbing
- Implementation Summary:
  Test was hardcoded to use indices [0, 1, 2] for states and observables,
  and array shape (3, 1) for inits/params. Changed to use dynamic indices
  based on system.sizes.states and system.sizes.observables. Also fixed
  inits/params array shapes to use actual system sizes.
- Issues Flagged: None

---

## Task Group 5: Reinstate Commented Tests in test_chunk_axis_property.py
**Status**: [x]
**Dependencies**: Task Groups 1-4

**Required Context**:
- File: tests/batchsolving/test_chunk_axis_property.py (lines 55-137 - commented tests)
- File: tests/conftest.py (lines 759-782 - solver and solver_mutable fixtures)
- File: src/cubie/batchsolving/solver.py (Solver class)

**Input Validation Required**:
- None - these are test reinstatements

**Tasks**:
1. **Reinstate TestChunkAxisInRun class tests**
   - File: tests/batchsolving/test_chunk_axis_property.py
   - Action: Modify
   - Details:
     The commented tests at lines 56-112 directly instantiate BatchSolverKernel,
     which causes issues. The fix is to use the solver fixture instead.
     
     Uncomment and refactor the tests to use `solver_mutable` fixture:
     
     ```python
     class TestChunkAxisInRun:
         """Tests for chunk_axis handling in solver.solve()."""
     
         def test_run_sets_chunk_axis_on_arrays(
             self, solver_mutable, precision, driver_array
         ):
             """Verify solve() sets chunk_axis on kernel arrays."""
             import numpy as np
             solver = solver_mutable
             
             inits = np.ones(
                 (solver.system_sizes.states, 1), dtype=precision
             )
             params = np.ones(
                 (solver.system_sizes.parameters, 1), dtype=precision
             )
             
             coefficients = (
                 driver_array.coefficients if driver_array is not None else None
             )
             
             solver.solve(
                 inits=inits,
                 params=params,
                 driver_coefficients=coefficients,
                 duration=0.1,
                 chunk_axis="time",
             )
             
             # After solve, kernel arrays should have the chunk_axis value
             assert solver.kernel.input_arrays._chunk_axis == "time"
             assert solver.kernel.output_arrays._chunk_axis == "time"
     
         def test_chunk_axis_property_after_run(
             self, solver_mutable, precision, driver_array
         ):
             """Verify chunk_axis property returns correct value after solve."""
             import numpy as np
             solver = solver_mutable
             
             inits = np.ones(
                 (solver.system_sizes.states, 1), dtype=precision
             )
             params = np.ones(
                 (solver.system_sizes.parameters, 1), dtype=precision
             )
             
             coefficients = (
                 driver_array.coefficients if driver_array is not None else None
             )
             
             solver.solve(
                 inits=inits,
                 params=params,
                 driver_coefficients=coefficients,
                 duration=0.1,
                 chunk_axis="time",
             )
             
             assert solver.kernel.chunk_axis == "time"
     ```
     
   - Edge cases: chunk_axis="run" (default), chunk_axis="variable"
   - Integration: Tests chunk_axis propagation through solve()

2. **Reinstate TestUpdateFromSolverChunkAxis class test**
   - File: tests/batchsolving/test_chunk_axis_property.py
   - Action: Modify
   - Details:
     Uncomment and refactor to use solver fixture:
     
     ```python
     class TestUpdateFromSolverChunkAxis:
         """Tests for update_from_solver chunk_axis behavior."""
     
         def test_update_from_solver_does_not_change_chunk_axis(
             self, solver_mutable
         ):
             """Verify update_from_solver preserves existing chunk_axis."""
             solver = solver_mutable
             kernel = solver.kernel
             
             # Set chunk_axis to non-default value via setter
             kernel.chunk_axis = "time"
             
             # Call update_from_solver (simulating what run() does)
             kernel.input_arrays.update_from_solver(kernel)
             
             # chunk_axis should be preserved
             assert kernel.input_arrays._chunk_axis == "time"
     ```
     
   - Edge cases: Updating with different solver configurations
   - Integration: Tests chunk_axis preservation across updates

**Tests to Create**:
- None - these are reinstating commented tests

**Tests to Run**:
- tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisInRun::test_run_sets_chunk_axis_on_arrays
- tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisInRun::test_chunk_axis_property_after_run
- tests/batchsolving/test_chunk_axis_property.py::TestUpdateFromSolverChunkAxis::test_update_from_solver_does_not_change_chunk_axis

**Outcomes**: 
- Files Modified:
  * tests/batchsolving/test_chunk_axis_property.py (70 lines changed)
- Functions/Methods Added/Modified:
  * Reinstated TestChunkAxisInRun class with two tests
  * Reinstated TestUpdateFromSolverChunkAxis class with one test
- Implementation Summary:
  Uncommented and refactored the previously-commented test classes.
  Changed from directly instantiating BatchSolverKernel to using
  solver_mutable fixture and accessing .kernel property. Tests now
  use solver.solve() instead of kernel.run() for integration testing.
  Added numpy import at module level.
- Issues Flagged: None

---

## Task Group 6: Cleanup - Remove Unused Properties/Patterns (if any)
**Status**: [x]
**Dependencies**: Task Groups 1-5

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (entire file)
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (entire file)
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (entire file)

**Input Validation Required**:
- None - this is cleanup

**Tasks**:
1. **Scan for unused _chunk_axis setters or manual _chunks assignments**
   - File: All array manager source files
   - Action: Review and potentially remove
   - Details:
     After the test fixes, scan the source files for any code patterns that:
     - Directly set `_chunks` outside of `_on_allocation_complete`
     - Directly set `_chunk_axis` outside of `_on_allocation_complete`
     - Have unused setters or properties related to chunking
     
     These patterns indicate legacy code that should be removed since chunking
     is now controlled through `allocate_queue()` with `chunk_axis` parameter.
     
     If no such patterns exist, mark this task as "No action needed".
     
   - Edge cases: Code that legitimately sets these for testing purposes
   - Integration: Ensures clean codebase after refactor

**Tests to Create**:
- None - this is cleanup/review

**Tests to Run**:
- Run full test suite for affected files after any changes:
  - tests/batchsolving/arrays/test_basearraymanager.py
  - tests/batchsolving/arrays/test_batchinputarrays.py
  - tests/batchsolving/arrays/test_batchoutputarrays.py
  - tests/batchsolving/test_SolverKernel.py
  - tests/batchsolving/test_chunk_axis_property.py

**Outcomes**: 
- Files Modified: None
- Implementation Summary:
  Reviewed BaseArrayManager.py, BatchInputArrays.py, and BatchOutputArrays.py
  for unused properties or patterns related to chunking. No action needed:
  - `_chunks` and `_chunk_axis` are plain attrs fields set only in
    `_on_allocation_complete`
  - No unused setters or properties exist
  - The manual assignments in tests are acceptable for testing purposes
- Issues Flagged: None

---

# Summary

## Total Task Groups: 6

## Dependency Chain:
```
Task Group 1 (BaseArrayManager test fix)
    ↓
Task Group 2 (BatchInputArrays buffer pool tests) ──────┐
Task Group 3 (BatchOutputArrays chunked transfer tests) ─┼── All depend on Task Group 1
Task Group 4 (SolverKernel IndexError fix) ─────────────┘   (can run in parallel)
    ↓
Task Group 5 (Reinstate chunk_axis_property tests) ── depends on Groups 1-4
    ↓
Task Group 6 (Cleanup) ── depends on all previous groups
```

## Tests to be Created: None (all tasks fix existing tests)

## Tests to be Run (after all fixes):
1. tests/batchsolving/arrays/test_basearraymanager.py::test_chunked_shape_propagates_through_allocation
2. tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_initialise_uses_buffer_pool_when_chunked
3. tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_release_buffers_returns_to_pool
4. tests/batchsolving/arrays/test_batchinputarrays.py::TestBufferPoolIntegration::test_reset_clears_buffer_pool_and_active_buffers
5. tests/batchsolving/arrays/test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_convert_host_to_numpy_uses_needs_chunked_transfer
6. tests/batchsolving/arrays/test_batchoutputarrays.py::TestNeedsChunkedTransferBranching::test_finalise_uses_needs_chunked_transfer
7. tests/batchsolving/test_SolverKernel.py::test_all_lower_plumbing
8. tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisInRun::test_run_sets_chunk_axis_on_arrays
9. tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisInRun::test_chunk_axis_property_after_run
10. tests/batchsolving/test_chunk_axis_property.py::TestUpdateFromSolverChunkAxis::test_update_from_solver_does_not_change_chunk_axis

## Estimated Complexity: Medium
- Most fixes follow a consistent pattern (proper allocation flow vs manual chunking setup)
- No new features being added, just fixing test setup patterns
- Risk: Some tests may have subtle dependencies on the incorrect patterns
