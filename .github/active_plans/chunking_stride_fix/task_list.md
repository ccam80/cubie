# Implementation Task List
# Feature: Chunking and Stride Fix
# Plan Reference: .github/active_plans/chunking_stride_fix/agent_plan.md

## Task Group 1: Fix chunk_arrays() to Handle Missing Axes
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/memory/mem_manager.py (lines 1278-1321) - chunk_arrays method
- File: src/cubie/memory/array_requests.py (entire file)
- File: .github/context/cubie_internal_structure.md (for architecture context)

**Input Validation Required**:
- No new validation needed; defensive check for missing axis

**Tasks**:
1. **Fix chunk_arrays() to skip arrays without the chunk axis**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     ```python
     def chunk_arrays(
         self,
         requests: dict[str, ArrayRequest],
         numchunks: int,
         axis: str = "run",
     ) -> dict[str, ArrayRequest]:
         """..."""
         chunked_requests = deepcopy(requests)
         for key, request in chunked_requests.items():
             # Skip chunking for explicitly unchunkable requests
             if getattr(request, "unchunkable", False):
                 continue
             # Skip chunking if the axis is not in this array's stride_order
             if axis not in request.stride_order:
                 continue
             # Divide all indices along selected axis by chunks
             run_index = request.stride_order.index(axis)
             newshape = tuple(
                 int(np_ceil(value / numchunks)) if i == run_index else value
                 for i, value in enumerate(request.shape)
             )
             request.shape = newshape
             chunked_requests[key] = request
         return chunked_requests
     ```
   - Edge cases: 2D arrays chunked on "time" axis (they don't have time axis)
   - Integration: Called by single_request() and allocate_queue()

**Tests to Create**:
- Test file: tests/memory/test_memmgmt.py
- Test function: test_chunk_arrays_skips_missing_axis
- Description: Verify chunk_arrays skips arrays whose stride_order does not contain the chunk axis

**Tests to Run**:
- tests/memory/test_memmgmt.py::test_chunk_arrays_skips_missing_axis
- tests/memory/test_memmgmt.py::test_chunk_arrays

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: Fix Host Slice Compatibility in from_device Transfers
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (lines 329-366) - finalise method
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (lines 338-378) - initialise method
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 926-963) - to_device/from_device methods
- File: src/cubie/memory/mem_manager.py (lines 1491-1517) - from_device method
- File: src/cubie/_utils.py (find slice_variable_dimension function)
- File: .github/context/cubie_internal_structure.md

**Input Validation Required**:
- None; this is a contiguity fix for copy operations

**Tasks**:
1. **Make host slices contiguous in OutputArrays.finalise()**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     ```python
     def finalise(self, host_indices: ChunkIndices) -> None:
         """
         Copy device arrays to host array slices.

         Parameters
         ----------
         host_indices
             Indices for the chunk being finalized.

         Returns
         -------
         None
             This method mutates host buffers in place.

         Notes
         -----
         Host slices are made contiguous before transfer to ensure
         compatible strides with device arrays.
         """
         from_ = []
         to_ = []
         host_slices = []  # Track original slices for post-copy writeback

         for array_name, slot in self.host.iter_managed_arrays():
             device_array = self.device.get_array(array_name)
             host_array = slot.array
             stride_order = slot.stride_order

             if self._chunk_axis in stride_order:
                 chunk_index = stride_order.index(self._chunk_axis)
                 slice_tuple = slice_variable_dimension(
                     host_indices, chunk_index, len(stride_order)
                 )
                 host_slice = host_array[slice_tuple]
                 # Make contiguous copy for device transfer
                 contiguous_slice = np_ascontiguousarray(host_slice)
                 host_slices.append((host_array, slice_tuple, contiguous_slice))
                 to_.append(contiguous_slice)
             else:
                 host_slices.append(None)
                 to_.append(host_array)
             from_.append(device_array)

         self.from_device(from_, to_)
         # Sync stream before writeback
         self._memory_manager.sync_stream(self)

         # Copy contiguous buffers back to original host slices
         for item in host_slices:
             if item is not None:
                 host_array, slice_tuple, contiguous_slice = item
                 host_array[slice_tuple] = contiguous_slice
     ```
   - Edge cases: Arrays without the chunk axis in stride_order
   - Integration: Called by BatchSolverKernel.run() after each chunk
   - Notes: Add import `from numpy import ascontiguousarray as np_ascontiguousarray`

2. **Make host slices contiguous in InputArrays.initialise()**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Details:
     ```python
     def initialise(self, host_indices: Union[slice, NDArray]) -> None:
         """Copy a batch chunk of host data to device buffers.

         Parameters
         ----------
         host_indices
             Indices for the chunk being initialized.

         Returns
         -------
         None
             Host slices are staged into device arrays in place.

         Notes
         -----
         Host slices are made contiguous before transfer to ensure
         compatible strides with device arrays.
         """
         from_ = []
         to_ = []

         if self._chunks <= 1:
             arrays_to_copy = [array for array in self._needs_overwrite]
             self._needs_overwrite = []
         else:
             arrays_to_copy = list(self.device.array_names())

         for array_name in arrays_to_copy:
             device_obj = self.device.get_managed_array(array_name)
             to_.append(device_obj.array)
             host_obj = self.host.get_managed_array(array_name)
             if self._chunks <= 1 or not device_obj.is_chunked:
                 from_.append(host_obj.array)
             else:
                 stride_order = host_obj.stride_order
                 # Skip chunking if axis not in stride_order
                 if self._chunk_axis not in stride_order:
                     from_.append(host_obj.array)
                     continue
                 chunk_index = stride_order.index(self._chunk_axis)
                 slice_tuple = [slice(None)] * len(stride_order)
                 slice_tuple[chunk_index] = host_indices
                 host_slice = host_obj.array[tuple(slice_tuple)]
                 # Make contiguous for device transfer
                 from_.append(np_ascontiguousarray(host_slice))

         self.to_device(from_, to_)
     ```
   - Edge cases: Arrays without chunk axis (e.g., 2D input arrays when chunking on time)
   - Integration: Called by BatchSolverKernel.run() before each chunk
   - Notes: Add import `from numpy import ascontiguousarray as np_ascontiguousarray`

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_chunked_transfers.py
- Test function: test_finalise_with_chunked_host_slices
- Description: Verify finalise correctly transfers device data to non-contiguous host slices
- Test function: test_initialise_with_chunked_host_slices
- Description: Verify initialise correctly transfers chunked host slices to device
- Test function: test_chunking_skips_arrays_without_chunk_axis
- Description: Verify arrays without the chunk axis are not sliced

**Tests to Run**:
- tests/batchsolving/arrays/test_chunked_transfers.py::test_finalise_with_chunked_host_slices
- tests/batchsolving/arrays/test_chunked_transfers.py::test_initialise_with_chunked_host_slices
- tests/batchsolving/arrays/test_chunked_transfers.py::test_chunking_skips_arrays_without_chunk_axis

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: Remove Custom Striding from MemoryManager
**Status**: [ ]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/memory/mem_manager.py (entire file)
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 1281-1290) - set_stride_order method
- File: tests/memory/test_memmgmt.py (lines 436-467, 583-612) - stride-related tests
- File: .github/context/cubie_internal_structure.md

**Input Validation Required**:
- None; this is removal of unused functionality

**Tasks**:
1. **Remove _stride_order attribute from MemoryManager**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details: Remove the `_stride_order` field from the MemoryManager class definition (line ~302-305)
     ```python
     # REMOVE THIS:
     # _stride_order: tuple[str, str, str] = field(
     #     default=("time", "variable", "run"), validator=attrsval_instance_of(
     #                 tuple)
     # )
     ```

2. **Remove set_global_stride_ordering() method**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details: Remove the entire `set_global_stride_ordering()` method (lines 784-818)

3. **Remove get_strides() method**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details: Remove the entire `get_strides()` method (lines 876-921)

4. **Simplify create_host_array() to always create C-contiguous arrays**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     ```python
     def create_host_array(
         self,
         shape: tuple[int, ...],
         dtype: type,
         stride_order: Optional[tuple[str, ...]] = None,
         memory_type: str = "pinned",
     ) -> ndarray:
         """
         Create a C-contiguous host array.

         Parameters
         ----------
         shape
             Shape of the array to create.
         dtype
             Data type for the array elements.
         stride_order
             Ignored; kept for API compatibility. Arrays are always
             C-contiguous.
         memory_type
             Memory type for the host array. Must be ``"pinned"`` or
             ``"host"``. Defaults to ``"pinned"``.

         Returns
         -------
         numpy.ndarray
             C-contiguous host array.
         """
         _ensure_cuda_context()
         if memory_type not in ("pinned", "host"):
             raise ValueError(
                 f"memory_type must be 'pinned' or 'host', got '{memory_type}'"
             )
         use_pinned = memory_type == "pinned"

         if use_pinned:
             arr = cuda.pinned_array(shape, dtype=dtype)
             arr.fill(0)
         else:
             arr = np_zeros(shape, dtype=dtype)
         return arr
     ```

5. **Simplify allocate_all() to not use custom strides**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details: Remove the `strides = self.get_strides(request)` call and `strides=strides` argument (lines 1182-1196)
     ```python
     def allocate_all(
         self,
         requests: dict[str, ArrayRequest],
         instance_id: int,
         stream: "cuda.cudadrv.driver.Stream",
     ) -> dict[str, object]:
         """..."""
         responses = {}
         instance_settings = self.registry[instance_id]
         for key, request in requests.items():
             arr = self.allocate(
                 shape=request.shape,
                 dtype=request.dtype,
                 memory_type=request.memory,
                 stream=stream,
             )
             instance_settings.add_allocation(key, arr)
             responses[key] = arr
         return responses
     ```

6. **Simplify allocate() to not accept strides parameter**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details: Remove the `strides` parameter and its usage (lines 1198-1246)
     ```python
     def allocate(
         self,
         shape: tuple[int, ...],
         dtype: Callable,
         memory_type: str,
         stream: "cuda.cudadrv.driver.Stream" = 0,
     ) -> object:
         """
         Allocate a single C-contiguous array with specified parameters.

         Parameters
         ----------
         shape
             Shape of the array to allocate.
         dtype
             Constructor returning the precision object for the array elements.
         memory_type
             Type of memory: "device", "mapped", "pinned", or "managed".
         stream
             CUDA stream for the allocation. Defaults to 0.

         Returns
         -------
         object
             Allocated GPU array.

         Raises
         ------
         ValueError
             If memory_type is not recognized.
         NotImplementedError
             If memory_type is "managed" (not yet supported).
         """
         _ensure_cuda_context()
         cp_ = self._allocator == CuPyAsyncNumbaManager
         with current_cupy_stream(stream) if cp_ else contextlib.nullcontext():
             if memory_type == "device":
                 return cuda.device_array(shape, dtype)
             elif memory_type == "mapped":
                 return cuda.mapped_array(shape, dtype)
             elif memory_type == "pinned":
                 return cuda.pinned_array(shape, dtype)
             elif memory_type == "managed":
                 raise NotImplementedError("Managed memory not implemented")
             else:
                 raise ValueError(f"Invalid memory type: {memory_type}")
     ```

7. **Remove set_stride_order() method from BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details: Remove the `set_stride_order()` method (lines 1281-1290)

**Tests to Create**:
- None; existing tests will be updated in Task 8

**Tests to Run**:
- tests/memory/test_memmgmt.py (after updating tests in Task 8)

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: Remove Custom Striding from BaseArrayManager
**Status**: [ ]
**Dependencies**: Task Group 3

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 685-812)
- File: .github/context/cubie_internal_structure.md

**Input Validation Required**:
- None; this is removal of unused functionality

**Tasks**:
1. **Remove _convert_to_device_strides() method**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details: Remove the entire `_convert_to_device_strides()` method (lines 685-751)

2. **Simplify _update_host_array() to not call stride conversion**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details: Remove the call to `self._convert_to_device_strides()` (lines 753-813)
     ```python
     def _update_host_array(
         self, new_array: NDArray, current_array: Optional[NDArray], label: str,
         shape_only: bool = False
     ) -> None:
         """
         Mark host arrays for overwrite or reallocation based on updates.

         Parameters
         ----------
         new_array
             Updated array that should replace the stored host array.
         current_array
             Previously stored host array or ``None``.
         label
             Array name used to index tracking lists.
         shape_only
             Only check shape equality when comparing arrays. Faster for
             output arrays that will be overwritten. Defaults to ``False``.

         Raises
         ------
         ValueError
             If ``new_array`` is ``None``.

         Returns
         -------
         None
             Nothing is returned.
         """
         if new_array is None:
             raise ValueError("New array is None")
         managed = self.host.get_managed_array(label)
         # Fast path: if current exists and arrays have matching shape/dtype
         # (and optionally content when shape_only=False), skip update
         if current_array is not None and self._arrays_equal(
             new_array, current_array, shape_only=shape_only
         ):
             return None
         # Handle new array (current is None)
         if current_array is None:
             self._needs_reallocation.append(label)
             self._needs_overwrite.append(label)
             self.host.attach(label, new_array)
             return None
         # Arrays differ; determine if shape changed or just values
         if current_array.shape != new_array.shape:
             if label not in self._needs_reallocation:
                 self._needs_reallocation.append(label)
             if label not in self._needs_overwrite:
                 self._needs_overwrite.append(label)
             if 0 in new_array.shape:
                 newshape = (1,) * len(current_array.shape)
                 new_array = np_zeros(newshape, dtype=managed.dtype)
         else:
             self._needs_overwrite.append(label)
         self.host.attach(label, new_array)
         return None
     ```

**Tests to Create**:
- None; existing tests will verify behavior

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py
- tests/batchsolving/arrays/test_batchoutputarrays.py
- tests/batchsolving/arrays/test_batchinputarrays.py

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: Update Tests for Removed Striding APIs
**Status**: [ ]
**Dependencies**: Task Group 3, Task Group 4

**Required Context**:
- File: tests/memory/test_memmgmt.py (entire file)
- File: src/cubie/memory/mem_manager.py (for reference to removed APIs)
- File: .github/context/cubie_internal_structure.md

**Input Validation Required**:
- None; this is test cleanup

**Tasks**:
1. **Remove or update stride-related tests in test_memmgmt.py**
   - File: tests/memory/test_memmgmt.py
   - Action: Modify
   - Details:
     - Remove `test_set_strides` (lines 436-458) - tests removed get_strides method
     - Remove `test_set_global_stride_ordering` (lines 460-467) - tests removed method
     - Remove `test_get_strides` (lines 583-612) - tests removed get_strides method
     - Update `test_create_host_array_3d_custom_stride` (lines 646-656) - simplify to verify C-contiguous behavior
     - Remove `stride_order` from `mem_manager_settings` fixture defaults (line 117)

2. **Update test_create_host_array tests to expect C-contiguous arrays**
   - File: tests/memory/test_memmgmt.py
   - Action: Modify
   - Details: Update test assertions to verify arrays are C-contiguous:
     ```python
     def test_create_host_array_3d_custom_stride(self, mgr):
         """Test create_host_array returns C-contiguous 3D array."""
         arr = mgr.create_host_array(
             shape=(2, 3, 4),
             dtype=np.float32,
             stride_order=("run", "variable", "time"),  # ignored
         )
         assert arr.shape == (2, 3, 4)
         assert arr.dtype == np.float32
         assert arr.flags['C_CONTIGUOUS']
         np.testing.assert_array_equal(
             arr, np.zeros((2, 3, 4), dtype=np.float32)
         )
     ```

**Tests to Create**:
- None; updating existing tests

**Tests to Run**:
- tests/memory/test_memmgmt.py

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 6: Create Chunking Regression Tests
**Status**: [ ]
**Dependencies**: Task Group 1, Task Group 2

**Required Context**:
- File: tests/batchsolving/conftest.py (for solver fixtures)
- File: tests/conftest.py (for general fixtures)
- File: tests/system_fixtures.py (for ODE system fixtures)
- File: src/cubie/batchsolving/BatchSolverKernel.py (for understanding run() behavior)
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py
- File: .github/context/cubie_internal_structure.md

**Input Validation Required**:
- None; these are regression tests

**Tasks**:
1. **Create test file for chunked transfers**
   - File: tests/batchsolving/arrays/test_chunked_transfers.py
   - Action: Create
   - Details:
     ```python
     """Tests for chunked array transfers between host and device."""

     import pytest
     import numpy as np
     from numpy import float32 as np_float32, int32 as np_int32

     from cubie.memory.mem_manager import MemoryManager, ArrayRequest
     from cubie.batchsolving.arrays.BatchOutputArrays import (
         OutputArrays,
         OutputArrayContainer,
     )
     from cubie.batchsolving.arrays.BatchInputArrays import (
         InputArrays,
         InputArrayContainer,
     )
     from cubie.batchsolving.arrays.BaseArrayManager import ManagedArray


     class MockSolverInstance:
         """Mock solver instance for testing array managers."""

         def __init__(self, precision=np_float32, n_runs=10, n_states=3):
             self.precision = precision
             self.num_runs = n_runs
             self._memory_manager = None
             self.stream_group = "test"
             self.chunk_axis = "run"


     @pytest.fixture
     def test_memory_manager():
         """Create a memory manager for testing."""
         class TestMemoryManager(MemoryManager):
             def get_memory_info(self):
                 return 1 * 1024**3, 8 * 1024**3

         return TestMemoryManager()


     class TestChunkArraysSkipsMissingAxis:
         """Test chunk_arrays handles arrays without the chunk axis."""

         def test_chunk_arrays_skips_2d_array_when_chunking_time(
             self, test_memory_manager
         ):
             """2D arrays with (variable, run) should not be chunked on time axis."""
             mgr = test_memory_manager
             requests = {
                 "input_2d": ArrayRequest(
                     shape=(10, 50),
                     dtype=np_float32,
                     memory="device",
                     stride_order=("variable", "run"),
                 ),
                 "output_3d": ArrayRequest(
                     shape=(100, 10, 50),
                     dtype=np_float32,
                     memory="device",
                     stride_order=("time", "variable", "run"),
                 ),
             }

             chunked = mgr.chunk_arrays(requests, numchunks=4, axis="time")

             # 2D array should be unchanged (no time axis)
             assert chunked["input_2d"].shape == (10, 50)
             # 3D array should be chunked on time axis
             assert chunked["output_3d"].shape == (25, 10, 50)

         def test_chunk_arrays_skips_1d_status_codes(self, test_memory_manager):
             """1D status code arrays should not be chunked."""
             mgr = test_memory_manager
             requests = {
                 "status_codes": ArrayRequest(
                     shape=(100,),
                     dtype=np_int32,
                     memory="device",
                     stride_order=("run",),
                     unchunkable=True,
                 ),
             }

             chunked = mgr.chunk_arrays(requests, numchunks=4, axis="time")

             # Unchunkable array should be unchanged
             assert chunked["status_codes"].shape == (100,)

         def test_chunk_arrays_handles_run_axis_correctly(
             self, test_memory_manager
         ):
             """Arrays should be correctly chunked on run axis."""
             mgr = test_memory_manager
             requests = {
                 "input_2d": ArrayRequest(
                     shape=(10, 50),
                     dtype=np_float32,
                     memory="device",
                     stride_order=("variable", "run"),
                 ),
                 "output_3d": ArrayRequest(
                     shape=(100, 10, 50),
                     dtype=np_float32,
                     memory="device",
                     stride_order=("time", "variable", "run"),
                 ),
             }

             chunked = mgr.chunk_arrays(requests, numchunks=5, axis="run")

             # Both arrays have run axis, both should be chunked
             assert chunked["input_2d"].shape == (10, 10)  # ceil(50/5)
             assert chunked["output_3d"].shape == (100, 10, 10)


     class TestChunkedHostSliceTransfers:
         """Test contiguous slice handling for chunked transfers."""

         def test_noncontiguous_host_slice_detected(self):
             """Verify that slicing a host array creates non-contiguous views."""
             host_array = np.zeros((100, 10, 50), dtype=np_float32)
             # Slice on run axis (last axis)
             host_slice = host_array[:, :, 0:10]
             # This slice is NOT contiguous because the parent has
             # strides based on shape (100, 10, 50), not (100, 10, 10)
             assert not host_slice.flags['C_CONTIGUOUS']

         def test_contiguous_copy_matches_shape(self):
             """Verify ascontiguousarray creates matching-shape contiguous array."""
             host_array = np.arange(100 * 10 * 50, dtype=np_float32).reshape(
                 100, 10, 50
             )
             host_slice = host_array[:, :, 0:10]
             contiguous = np.ascontiguousarray(host_slice)

             assert contiguous.flags['C_CONTIGUOUS']
             assert contiguous.shape == (100, 10, 10)
             np.testing.assert_array_equal(contiguous, host_slice)
     ```

2. **Create integration test for chunked solver execution**
   - File: tests/batchsolving/test_chunked_solver.py
   - Action: Create
   - Details:
     ```python
     """Integration tests for chunked solver execution."""

     import pytest
     import numpy as np

     from cubie import Solver


     @pytest.fixture
     def simple_ode_system(three_state_linear):
         """Provide a simple ODE system for integration tests."""
         return three_state_linear


     class TestChunkedSolverExecution:
         """Test solver execution with forced chunking."""

         @pytest.mark.parametrize("chunk_axis", ["run", "time"])
         def test_chunked_solve_produces_valid_output(
             self, simple_ode_system, precision, chunk_axis
         ):
             """Verify chunked solver produces valid output arrays."""
             solver = Solver(simple_ode_system, algorithm="explicit_euler")
             n_runs = 10
             n_states = simple_ode_system.n_states

             inits = np.ones((n_states, n_runs), dtype=precision)
             params = np.ones(
                 (simple_ode_system.n_parameters, n_runs), dtype=precision
             )

             result = solver.solve(
                 inits,
                 params,
                 duration=0.1,
                 save_every=0.01,
                 chunk_axis=chunk_axis,
             )

             # Verify output shape and that values are not all zeros/NaN
             assert result.state is not None
             assert result.state.shape[2] == n_runs
             assert not np.all(result.state == 0)
             assert not np.any(np.isnan(result.state))
     ```

**Tests to Create**:
- tests/batchsolving/arrays/test_chunked_transfers.py (entire file as shown above)
- tests/batchsolving/test_chunked_solver.py (entire file as shown above)

**Tests to Run**:
- tests/batchsolving/arrays/test_chunked_transfers.py
- tests/batchsolving/test_chunked_solver.py

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

| Group | Name | Dependencies | Tasks | Tests |
|-------|------|--------------|-------|-------|
| 1 | Fix chunk_arrays() | None | 1 | 1 |
| 2 | Fix Host Slice Compatibility | 1 | 2 | 3 |
| 3 | Remove MemoryManager Striding | 2 | 7 | 0 |
| 4 | Remove BaseArrayManager Striding | 3 | 2 | 0 |
| 5 | Update Tests | 3, 4 | 2 | 0 |
| 6 | Create Regression Tests | 1, 2 | 2 | 2 files |

**Total Task Groups**: 6
**Dependency Chain**: 1 → 2 → 3 → 4 → 5 (parallel: 6)
**Estimated Complexity**: Medium - mostly removal of code with targeted fixes
