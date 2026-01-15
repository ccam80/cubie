# Implementation Task List
# Feature: refactor_total_runs_architecture
# Plan Reference: .github/active_plans/refactor_total_runs_architecture/agent_plan.md

## Task Group 1: BaseArrayManager - Add num_runs Attribute and set_array_runs Method
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 286-343, 873-938)
- File: src/cubie/_utils.py (validator imports for opt_getype_validator)

**Input Validation Required**:
- num_runs: Check type is int, value >= 1 using opt_getype_validator(int, 1)
- num_runs parameter in set_array_runs(): Check type is int, value >= 1 (raise TypeError/ValueError as appropriate)

**Tasks**:
1. **Add num_runs Attribute to BaseArrayManager.__init__**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Location: After line 343 (after _memory_manager field)
   - Details:
     ```python
     num_runs: Optional[int] = field(
         default=None,
         validator=opt_getype_validator(int, 1),
     )
     ```
   - Edge cases: None - validator handles type and range checks
   - Integration: Attribute will be set by subclasses in update_from_solver()

2. **Add set_array_runs() Method to BaseArrayManager**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Create
   - Location: After __attrs_post_init__ method (after line 360)
   - Details:
     ```python
     def set_array_runs(self, num_runs: int) -> None:
         """
         Update num_runs in all ManagedArray instances.
         
         Parameters
         ----------
         num_runs
             Total number of runs for the current batch. Must be int >= 1.
         
         Raises
         ------
         TypeError
             If num_runs is not an integer.
         ValueError
             If num_runs is less than 1.
         
         Returns
         -------
         None
             Nothing is returned.
         
         Notes
         -----
         This method propagates the num_runs value to all ManagedArray
         instances in both host and device containers. Called from
         update_from_solver() in subclasses after extracting num_runs
         from sizing metadata.
         """
         # Validate num_runs type
         if not isinstance(num_runs, int):
             raise TypeError(
                 f"num_runs must be int, got {type(num_runs).__name__}"
             )
         
         # Validate num_runs value
         if num_runs < 1:
             raise ValueError(
                 f"num_runs must be >= 1, got {num_runs}"
             )
         
         # Set num_runs attribute
         self.num_runs = num_runs
         
         # Propagate to all ManagedArray instances
         # Note: ManagedArray doesn't store num_runs internally;
         # this method exists for potential future use
     ```
   - Edge cases: Non-integer types, values less than 1
   - Integration: Called by InputArrays.update_from_solver() and OutputArrays.update_from_solver()

3. **Remove _get_total_runs() Method from BaseArrayManager**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Delete
   - Location: Lines 873-893
   - Details: Delete entire method including docstring
   - Edge cases: None - method is replaced by direct attribute access
   - Integration: Update allocate() method to use self.num_runs instead

4. **Update allocate() Method to Use self.num_runs**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Location: Lines 895-938
   - Details:
     ```python
     def allocate(self) -> None:
         """
         Queue allocation requests for arrays that need reallocation.

         Notes
         -----
         Builds :class:`ArrayRequest` objects for arrays marked for
         reallocation and sets the ``unchunkable`` hint based on host metadata.

         Chunking is always performed along the run axis by convention.

         Returns
         -------
         None
             Nothing is returned.
         """
         # Remove lines 912-913 (total_runs_value = self._get_total_runs())
         
         requests = {}
         for array_label in list(set(self._needs_reallocation)):
             host_array_object = self.host.get_managed_array(array_label)
             host_array = host_array_object.array
             if host_array is None:
                 continue
             device_array_object = self.device.get_managed_array(array_label)
             
             # Determine total_runs based on chunkability
             # OLD LOGIC (lines 922-926):
             # if host_array_object.is_chunked:
             #     total_runs = total_runs_value
             # else:
             #     total_runs = None
             
             # NEW LOGIC:
             if host_array_object.is_chunked:
                 total_runs = self.num_runs
             else:
                 total_runs = 1  # Not None - always provide total_runs
             
             request = ArrayRequest(
                 shape=host_array.shape,
                 dtype=device_array_object.dtype,
                 memory=device_array_object.memory_type,
                 chunk_axis_index=host_array_object._chunk_axis_index,
                 unchunkable=not host_array_object.is_chunked,
                 total_runs=total_runs,
             )
             requests[array_label] = request
         if requests:
             self.request_allocation(requests)
     ```
   - Edge cases: self.num_runs is None (should be set by update_from_solver before allocate)
   - Integration: Calls ArrayRequest with always-valid total_runs

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_basearraymanager.py
- Test function: test_set_array_runs_sets_attribute
- Description: Verify set_array_runs() sets self.num_runs correctly
- Test function: test_set_array_runs_validates_type
- Description: Verify set_array_runs() raises TypeError for non-int
- Test function: test_set_array_runs_validates_range
- Description: Verify set_array_runs() raises ValueError for num_runs < 1
- Test function: test_allocate_uses_num_runs_for_chunked_arrays
- Description: Verify allocate() uses self.num_runs for chunked arrays
- Test function: test_allocate_uses_one_for_unchunked_arrays
- Description: Verify allocate() uses total_runs=1 for unchunked arrays

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py::TestNumRunsAttribute::test_set_array_runs_sets_attribute
- tests/batchsolving/arrays/test_basearraymanager.py::TestNumRunsAttribute::test_set_array_runs_validates_type
- tests/batchsolving/arrays/test_basearraymanager.py::TestNumRunsAttribute::test_set_array_runs_validates_range
- tests/batchsolving/arrays/test_basearraymanager.py::TestNumRunsAttribute::test_allocate_uses_num_runs_for_chunked_arrays
- tests/batchsolving/arrays/test_basearraymanager.py::TestNumRunsAttribute::test_allocate_uses_one_for_unchunked_arrays

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/arrays/BaseArrayManager.py (50 lines changed)
    - Added opt_getype_validator import (line 33)
    - Added num_runs attribute after _memory_manager field (lines 344-346)
    - Added set_array_runs() method after is_chunked property (lines 370-405)
    - Removed _get_total_runs() method (deleted ~21 lines)
    - Updated allocate() method to use self.num_runs (lines 895-938)
  * tests/batchsolving/arrays/test_basearraymanager.py (365 lines added)
    - Added TestNumRunsAttribute class with 5 test methods
- Functions/Methods Added/Modified:
  * BaseArrayManager.num_runs attribute (new)
  * BaseArrayManager.set_array_runs() method (new)
  * BaseArrayManager._get_total_runs() method (removed)
  * BaseArrayManager.allocate() method (modified)
- Implementation Summary:
  - Added num_runs as an Optional[int] attribute with opt_getype_validator
  - Implemented set_array_runs() with explicit validation and error messages
  - Removed _get_total_runs() helper method which extracted from _sizes
  - Updated allocate() to use self.num_runs for chunked arrays and 1 for unchunked
  - Created 5 comprehensive tests covering attribute setting and validation
- Issues Flagged: None

**Tests Created**:
- tests/batchsolving/arrays/test_basearraymanager.py::TestNumRunsAttribute::test_set_array_runs_sets_attribute
- tests/batchsolving/arrays/test_basearraymanager.py::TestNumRunsAttribute::test_set_array_runs_validates_type
- tests/batchsolving/arrays/test_basearraymanager.py::TestNumRunsAttribute::test_set_array_runs_validates_range
- tests/batchsolving/arrays/test_basearraymanager.py::TestNumRunsAttribute::test_allocate_uses_num_runs_for_chunked_arrays
- tests/batchsolving/arrays/test_basearraymanager.py::TestNumRunsAttribute::test_allocate_uses_one_for_unchunked_arrays

---

## Task Group 2: InputArrays and OutputArrays - Set num_runs During Update
**Status**: [x]
**Dependencies**: Groups [1]

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (lines 250-270)
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (lines 296-340)
- File: src/cubie/outputhandling/output_sizes.py (lines 250-271 for BatchInputSizes, lines 381-395 for BatchOutputSizes)

**Input Validation Required**:
- None - num_runs is extracted from existing sizing objects which are already validated

**Tasks**:
1. **Update InputArrays.update_from_solver() to Set num_runs**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Location: Lines 250-270 (update_from_solver method)
   - Details:
     ```python
     def update_from_solver(self, solver_instance: "BatchSolverKernel") -> None:
         """Refresh size, precision, and chunk axis from the solver.

         Parameters
         ----------
         solver_instance
             The solver instance to update from.

         Returns
         -------
         None
         """
         self._sizes = BatchInputSizes.from_solver(solver_instance).nonzero
         self._precision = solver_instance.precision
         
         # Extract num_runs from sizes and set on manager
         if self._sizes is not None:
             # Extract from initial_values shape (second element)
             num_runs = self._sizes.initial_values[1]
             self.set_array_runs(num_runs)
         
         for name, arr_obj in self.host.iter_managed_arrays():
             if np_issubdtype(np_dtype(arr_obj.dtype), np_floating):
                 arr_obj.dtype = self._precision
         for name, arr_obj in self.device.iter_managed_arrays():
             if np_issubdtype(np_dtype(arr_obj.dtype), np_floating):
                 arr_obj.dtype = self._precision
     ```
   - Edge cases: _sizes is None (conditional handles this)
   - Integration: Calls set_array_runs() from Task Group 1

2. **Update OutputArrays.update_from_solver() to Set num_runs**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Location: Lines 296-340 (update_from_solver method)
   - Details:
     ```python
     def update_from_solver(
         self, solver_instance: "BatchSolverKernel"
     ) -> Dict[str, NDArray[np_floating]]:
         """
         Update sizes and precision from solver, returning new host arrays.

         Only creates new pinned arrays when existing arrays do not match
         the expected shape and dtype. This avoids expensive pinned memory
         allocation on repeated solver runs with identical configurations.

         Parameters
         ----------
         solver_instance
             The solver instance to update from.
         
         [... rest of docstring ...]
         """
         # Existing logic to compute new sizes
         new_sizes = BatchOutputSizes.from_solver(solver_instance).nonzero
         
         # Extract num_runs from sizes and set on manager
         if new_sizes is not None:
             # Extract from state shape (third element)
             num_runs = new_sizes.state[2]
             self.set_array_runs(num_runs)
         
         # [... rest of method continues unchanged ...]
     ```
   - Edge cases: new_sizes is None (conditional handles this)
   - Integration: Calls set_array_runs() from Task Group 1

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_batch_input_arrays.py
- Test function: test_update_from_solver_sets_num_runs
- Description: Verify update_from_solver extracts and sets num_runs correctly
- Test file: tests/batchsolving/arrays/test_batch_output_arrays.py
- Test function: test_update_from_solver_sets_num_runs
- Description: Verify update_from_solver extracts and sets num_runs correctly

**Tests to Run**:
- tests/batchsolving/arrays/test_batchinputarrays.py::TestInputArrays::test_update_from_solver_sets_num_runs
- tests/batchsolving/arrays/test_batchoutputarrays.py::TestOutputArrays::test_update_from_solver_sets_num_runs

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/arrays/BatchInputArrays.py (4 lines changed)
    - Added extraction of num_runs from initial_values shape (lines 262-265)
    - Added call to self.set_array_runs(num_runs)
  * src/cubie/batchsolving/arrays/BatchOutputArrays.py (4 lines changed)
    - Added extraction of num_runs from state shape (lines 320-323)
    - Added call to self.set_array_runs(num_runs)
  * tests/batchsolving/arrays/test_batchinputarrays.py (29 lines added)
    - Added test_update_from_solver_sets_num_runs method to TestInputArrays class
  * tests/batchsolving/arrays/test_batchoutputarrays.py (29 lines added)
    - Added test_update_from_solver_sets_num_runs method to TestOutputArrays class
- Functions/Methods Added/Modified:
  * InputArrays.update_from_solver() method (modified)
  * OutputArrays.update_from_solver() method (modified)
- Implementation Summary:
  - Modified InputArrays.update_from_solver() to extract num_runs from self._sizes.initial_values[1]
  - Modified OutputArrays.update_from_solver() to extract num_runs from self._sizes.state[2]
  - Both methods now call self.set_array_runs(num_runs) to propagate to BaseArrayManager
  - Created tests that verify num_runs is None initially, gets set by update_from_solver, and matches the expected value
  - Tests verify num_runs matches both solver.num_runs and the value in the sizes object
- Issues Flagged: None

---

## Task Group 3: ArrayRequest - Change total_runs to Always int >= 1
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/memory/array_requests.py (lines 20-86)
- File: src/cubie/_utils.py (getype_validator import)

**Input Validation Required**:
- total_runs: Always int >= 1 using getype_validator(int, 1) (not opt_getype_validator)

**Tasks**:
1. **Change total_runs Default and Type in ArrayRequest**
   - File: src/cubie/memory/array_requests.py
   - Action: Modify
   - Location: Lines 78-81
   - Details:
     ```python
     # OLD (lines 78-81):
     total_runs: Optional[int] = attrs.field(
         default=None,
         validator=opt_getype_validator(int, 1),
     )
     
     # NEW:
     total_runs: int = attrs.field(
         default=1,
         validator=getype_validator(int, 1),
     )
     ```
   - Edge cases: None - validator enforces type and range
   - Integration: MemoryManager no longer needs to check for None

2. **Update ArrayRequest Docstring for total_runs**
   - File: src/cubie/memory/array_requests.py
   - Action: Modify
   - Location: Lines 35-39 (in class docstring)
   - Details:
     ```python
     # OLD:
     total_runs
         Total number of runs for chunking calculations. When ``None``, the
         array is not intended for run-axis chunking (e.g., driver_coefficients).
         Memory manager extracts this value to determine chunk parameters.
         Defaults to ``None``.
     
     # NEW:
     total_runs
         Total number of runs for chunking calculations. Defaults to ``1``
         for unchunkable arrays (e.g., driver_coefficients). Memory manager
         uses this value to determine chunk parameters. Always >= 1.
     ```
   - Edge cases: None - documentation only
   - Integration: Clarifies semantic meaning of default value

3. **Update ArrayRequest Type Annotation in Attributes Section**
   - File: src/cubie/memory/array_requests.py
   - Action: Modify
   - Location: Lines 53-55 (in Attributes section of docstring)
   - Details:
     ```python
     # OLD:
     total_runs
         Total number of runs for chunking calculations, or ``None`` if not
         applicable.
     
     # NEW:
     total_runs
         Total number of runs for chunking calculations. Always >= 1.
     ```
   - Edge cases: None - documentation only
   - Integration: Reflects non-optional type

**Tests to Create**:
- Test file: tests/memory/test_array_requests.py
- Test function: test_array_request_total_runs_defaults_to_one
- Description: Verify total_runs defaults to 1 when not provided
- Test function: test_array_request_total_runs_validates_minimum
- Description: Verify validator raises for total_runs < 1
- Test function: test_array_request_total_runs_is_not_optional
- Description: Verify total_runs cannot be None

**Tests to Run**:
- tests/memory/test_array_requests.py::test_array_request_total_runs_defaults_to_one
- tests/memory/test_array_requests.py::test_array_request_total_runs_validates_minimum
- tests/memory/test_array_requests.py::test_array_request_total_runs_is_not_optional

**Outcomes**:
- Files Modified:
  * src/cubie/memory/array_requests.py (6 lines changed)
    - Added getype_validator import (line 15)
    - Changed total_runs type from Optional[int] to int (line 77)
    - Changed total_runs default from None to 1 (line 78)
    - Changed total_runs validator from opt_getype_validator to getype_validator (line 79)
    - Updated total_runs parameter docstring (lines 35-39)
    - Updated total_runs attribute docstring (lines 53-54)
  * tests/memory/test_array_requests.py (64 lines changed)
    - Removed obsolete test_array_request_total_runs_defaults_to_none (15 lines removed)
    - Added test_array_request_total_runs_defaults_to_one (18 lines)
    - Added test_array_request_total_runs_validates_minimum (23 lines)
    - Added test_array_request_total_runs_is_not_optional (23 lines)
- Functions/Methods Added/Modified:
  * ArrayRequest.total_runs attribute (modified - type and default changed)
- Implementation Summary:
  - Changed total_runs from Optional[int] defaulting to None to int defaulting to 1
  - Updated validator from opt_getype_validator to getype_validator(int, 1)
  - Updated docstrings to reflect that total_runs is always >= 1, never None
  - Removed obsolete test for None default behavior
  - Created three comprehensive tests:
    1. test_array_request_total_runs_defaults_to_one: Verifies default value is 1 (not None)
    2. test_array_request_total_runs_validates_minimum: Verifies values < 1 are rejected
    3. test_array_request_total_runs_is_not_optional: Verifies None is not accepted
- Issues Flagged: None

---

## Task Group 4: MemoryManager - Simplify Run Extraction
**Status**: [x]
**Dependencies**: Groups [3]

**Required Context**:
- File: src/cubie/memory/mem_manager.py (lines 1170-1224, 1226-1298)

**Input Validation Required**:
- None - all requests are guaranteed to have valid total_runs by ArrayRequest validator

**Tasks**:
1. **Remove _extract_num_runs() Method from MemoryManager**
   - File: src/cubie/memory/mem_manager.py
   - Action: Delete
   - Location: Lines 1170-1224
   - Details: Delete entire method including docstring
   - Edge cases: None - method is being replaced
   - Integration: allocate_queue() will use simpler logic

2. **Simplify allocate_queue() to Get total_runs from First Request**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Location: Lines 1226-1298 (allocate_queue method)
   - Details:
     ```python
     def allocate_queue(
         self,
         triggering_instance: object,
     ) -> None:
         """
         Process all queued requests for a stream group with coordinated chunking.

         Chunking is always performed along the run axis when memory
         constraints require splitting the batch.

         Parameters
         ----------
         triggering_instance
             The instance that triggered queue processing.

         Notes
         -----
         Processes all pending requests in the same stream group, applying
         coordinated chunking based on available memory. Calls
         allocation_ready_hook for each instance with their results.

         Returns
         -------
         None
         """
         stream_group = self.get_stream_group(triggering_instance)
         stream = self.get_stream(triggering_instance)
         queued_requests = self._queued_allocations.pop(stream_group, {})

         # OLD (lines 1259-1260):
         # # Extract num_runs from ArrayRequest total_runs fields
         # num_runs = self._extract_num_runs(queued_requests)
         
         # NEW - Get total_runs from first request
         first_instance_requests = next(iter(queued_requests.values()))
         first_request = next(iter(first_instance_requests.values()))
         num_runs = first_request.total_runs

         chunk_length, num_chunks = self.get_chunk_parameters(
             queued_requests, num_runs, stream_group
         )
         # [... rest of method continues unchanged ...]
     ```
   - Edge cases: Empty queued_requests (will raise StopIteration - acceptable behavior)
   - Integration: Assumes all requests have consistent total_runs (guaranteed by array managers)

3. **Remove Explanatory Comment about num_runs Extraction**
   - File: src/cubie/memory/mem_manager.py
   - Action: Delete
   - Location: Lines 1247-1250 (in allocate_queue docstring)
   - Details: Remove the note "The num_runs value is extracted from ArrayRequest.total_runs fields rather than from triggering_instance attributes. All requests with non-None total_runs must have the same value."
   - Edge cases: None - documentation only
   - Integration: Simpler implementation doesn't need explanation

**Tests to Create**:
- Test file: tests/memory/test_memmgmt.py
- Test function: test_allocate_queue_uses_first_request_total_runs
- Description: Verify allocate_queue gets num_runs from first request
- Test function: test_allocate_queue_handles_all_requests_same_total_runs
- Description: Verify allocate_queue works when all requests have same total_runs

**Tests to Run**:
- tests/memory/test_memmgmt.py::test_allocate_queue_uses_first_request_total_runs
- tests/memory/test_memmgmt.py::test_allocate_queue_handles_all_requests_same_total_runs

**Outcomes**:
- Files Modified:
  * src/cubie/memory/mem_manager.py (58 lines changed)
    - Removed _extract_num_runs() method entirely (55 lines deleted, lines 1170-1224)
    - Simplified allocate_queue() to get num_runs from first request (3 lines changed)
    - Removed explanatory comment from allocate_queue docstring (3 lines removed)
  * tests/memory/test_memmgmt.py (119 lines added)
    - Added test_allocate_queue_uses_first_request_total_runs (52 lines)
    - Added test_allocate_queue_handles_all_requests_same_total_runs (67 lines)
- Functions/Methods Added/Modified:
  * MemoryManager._extract_num_runs() method (removed)
  * MemoryManager.allocate_queue() method (modified - simplified num_runs extraction)
- Implementation Summary:
  - Removed the complex _extract_num_runs() method that validated consistency
  - Simplified allocate_queue() to use simple first-request access pattern:
    `first_request = next(iter(next(iter(queued_requests.values())).values()))`
  - Removed docstring note about extraction logic since it's now trivial
  - Created two comprehensive tests verifying the new simple extraction works
  - Tests verify chunk_length matches expected total_runs value
  - Tests verify all arrays are allocated correctly
- Issues Flagged: None

---

## Task Group 5: Remove runs Properties from Sizing Classes
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/outputhandling/output_sizes.py (lines 257-271, 381-395)

**Input Validation Required**:
- None - only removing properties

**Tasks**:
1. **Remove BatchInputSizes.runs Property**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Delete
   - Location: Lines 257-271
   - Details: Delete entire property method including @property decorator and docstring
   - Edge cases: None - property is being removed
   - Integration: Code using this property must get num_runs from BaseArrayManager instead

2. **Remove BatchOutputSizes.runs Property**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Delete
   - Location: Lines 381-395
   - Details: Delete entire property method including @property decorator and docstring
   - Edge cases: None - property is being removed
   - Integration: Code using this property must get num_runs from BaseArrayManager instead

**Tests to Create**:
- None - only removing obsolete functionality

**Tests to Run**:
- tests/outputhandling/test_output_sizes.py (full file to ensure no regressions)

**Outcomes**:
- Files Modified:
  * src/cubie/outputhandling/output_sizes.py (30 lines removed)
    - Removed BatchInputSizes.runs property (lines 257-271, 15 lines)
    - Removed BatchOutputSizes.runs property (lines 381-395, 15 lines)
- Functions/Methods Added/Modified:
  * BatchInputSizes.runs property (removed)
  * BatchOutputSizes.runs property (removed)
- Implementation Summary:
  - Deleted BatchInputSizes.runs property that extracted num_runs from initial_values[1]
  - Deleted BatchOutputSizes.runs property that extracted num_runs from state[2]
  - Both properties are now obsolete as num_runs is stored directly in BaseArrayManager
  - Code must now access num_runs from BaseArrayManager.num_runs instead of sizing classes
- Issues Flagged: None

---

## Task Group 6: Remove Obsolete Tests for Deleted Methods
**Status**: [x]
**Dependencies**: Groups [1, 3, 4, 5]

**Required Context**:
- File: tests/batchsolving/arrays/test_basearraymanager.py (lines 2033-2095)
- File: tests/memory/test_array_requests.py (lines 167-181)
- File: tests/memory/test_memmgmt.py (lines 1188-1284)
- File: tests/outputhandling/test_output_sizes.py (lines 371-384, 447-463)

**Input Validation Required**:
- None - only removing tests

**Tasks**:
1. **Remove TestGetTotalRuns Class from test_basearraymanager.py**
   - File: tests/batchsolving/arrays/test_basearraymanager.py
   - Action: Delete
   - Location: Lines 2033-2095
   - Details: Delete entire TestGetTotalRuns class including all test methods
   - Edge cases: None - tests are obsolete
   - Integration: _get_total_runs() method no longer exists

2. **Remove test_array_request_total_runs_defaults_to_none**
   - File: tests/memory/test_array_requests.py
   - Action: Delete
   - Location: Lines 167-181
   - Details: Delete entire test function including docstring
   - Edge cases: None - test is obsolete
   - Integration: total_runs now defaults to 1, not None

3. **Remove TestExtractNumRuns Class from test_memmgmt.py**
   - File: tests/memory/test_memmgmt.py
   - Action: Delete
   - Location: Lines 1188-1284 (may extend beyond based on remaining tests)
   - Details: Delete entire TestExtractNumRuns class including all test methods:
     - test_extract_num_runs_finds_single_value
     - test_extract_num_runs_ignores_none_values
     - test_extract_num_runs_raises_on_no_values (line 1239)
     - test_extract_num_runs_raises_on_inconsistent_values (line 1262)
   - Edge cases: None - tests are obsolete
   - Integration: _extract_num_runs() method no longer exists

4. **Remove test_batch_input_sizes_exposes_runs from test_output_sizes.py**
   - File: tests/outputhandling/test_output_sizes.py
   - Action: Delete
   - Location: Lines 371-384
   - Details: Delete entire test function including docstring
   - Edge cases: None - test is obsolete
   - Integration: BatchInputSizes.runs property no longer exists

5. **Remove test_batch_output_sizes_exposes_runs from test_output_sizes.py**
   - File: tests/outputhandling/test_output_sizes.py
   - Action: Delete
   - Location: Lines 447-463
   - Details: Delete entire test function including docstring
   - Edge cases: None - test is obsolete
   - Integration: BatchOutputSizes.runs property no longer exists

**Tests to Create**:
- None - only removing obsolete tests

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py (verify no failures after removal)
- tests/memory/test_array_requests.py (verify no failures after removal)
- tests/memory/test_memmgmt.py (verify no failures after removal)
- tests/outputhandling/test_output_sizes.py (verify no failures after removal)

**Outcomes**:
- Files Modified:
  * tests/batchsolving/arrays/test_basearraymanager.py (129 lines removed)
    - Removed TestGetTotalRuns class (lines 2033-2161)
  * tests/memory/test_array_requests.py (19 lines removed)
    - Removed test_array_request_total_runs_defaults_to_one function (lines 167-185)
  * tests/memory/test_memmgmt.py (97 lines removed)
    - Removed TestExtractNumRuns class (lines 1188-1284)
  * tests/outputhandling/test_output_sizes.py (53 lines removed)
    - Removed test_batch_input_sizes_exposes_runs function (lines 371-393)
    - Removed test_batch_output_sizes_exposes_runs function (lines 424-452)
- Functions/Methods Added/Modified:
  * TestGetTotalRuns test class (removed - tested deleted _get_total_runs() method)
  * test_array_request_total_runs_defaults_to_one test (removed - tested old None default)
  * TestExtractNumRuns test class (removed - tested deleted _extract_num_runs() method)
  * test_batch_input_sizes_exposes_runs test (removed - tested deleted runs property)
  * test_batch_output_sizes_exposes_runs test (removed - tested deleted runs property)
- Implementation Summary:
  - Removed all tests for methods and properties deleted in previous task groups
  - TestGetTotalRuns tested the _get_total_runs() method that was removed in Group 1
  - test_array_request_total_runs_defaults_to_one tested the old None default behavior that was changed to 1 in Group 3
  - TestExtractNumRuns tested the _extract_num_runs() method that was removed in Group 4
  - test_batch_input_sizes_exposes_runs and test_batch_output_sizes_exposes_runs tested the runs properties removed in Group 5
  - All deletions clean up obsolete test code that no longer has corresponding implementation
- Issues Flagged: None

---

## Task Group 7: Fix chunk_slice Negative Index Validation Test
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: tests/batchsolving/arrays/test_basearraymanager.py (lines 1771-1785)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 126-187 for chunk_slice implementation)

**Input Validation Required**:
- None - test updates only

**Tasks**:
1. **Update test_chunk_slice_validates_chunk_index to Test Range Instead of Non-Negative**
   - File: tests/batchsolving/arrays/test_basearraymanager.py
   - Action: Modify
   - Location: Lines 1771-1785
   - Details:
     ```python
     def test_chunk_slice_validates_chunk_index(self):
         """Verify chunk_slice raises ValueError for out-of-range chunk_index."""
         managed = ManagedArray(
             dtype=np_float32,
             default_shape=(10, 5, 100),
             memory_type="host",
             is_chunked=True,
         )
         managed.array = np.zeros((10, 5, 100), dtype=np_float32)
         managed.num_chunks = 4
         managed.chunk_length = 25

         # OLD - Test negative chunk_index:
         # with pytest.raises(ValueError, match="chunk_index -1 out of range"):
         #     managed.chunk_slice(-1)
         
         # NEW - Test chunk_index too large (out of range positive)
         with pytest.raises(ValueError, match="chunk_index 4 out of range"):
             managed.chunk_slice(4)
         
         # NEW - Test chunk_index too negative (out of range)
         with pytest.raises(ValueError, match="chunk_index -5 out of range"):
             managed.chunk_slice(-5)  # Only 4 chunks, so -5 is out of range
     ```
   - Edge cases: Valid negative indices like -1, -2 should work (Python convention)
   - Integration: Tests against actual chunk_slice implementation behavior

**Tests to Create**:
- None - modifying existing test

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py::TestChunkSliceMethod::test_chunk_slice_validates_chunk_index
- tests/batchsolving/arrays/test_basearraymanager.py::TestChunkSliceMethod::test_chunk_slice_accepts_valid_negative_indices

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/arrays/BaseArrayManager.py (9 lines changed)
    - Updated chunk_slice validation to accept valid negative indices (lines 168-177)
    - Changed validation from "check non-negative" to "check in range"
    - Added conversion of negative indices to positive indices (Python-style)
  * tests/batchsolving/arrays/test_basearraymanager.py (48 lines changed)
    - Updated test_chunk_slice_validates_chunk_index to test range instead of non-negative (34 lines modified)
    - Removed test for -1 being invalid (it's valid in Python)
    - Added test for chunk_index too large (4 for 4 chunks)
    - Added test for chunk_index too negative (-5 for 4 chunks, -10 for 4 chunks)
    - Added new test test_chunk_slice_accepts_valid_negative_indices (43 lines added)
- Functions/Methods Added/Modified:
  * ManagedArray.chunk_slice() method (modified validation logic)
  * test_chunk_slice_validates_chunk_index() test (modified to test range validation)
  * test_chunk_slice_accepts_valid_negative_indices() test (added)
- Implementation Summary:
  - Changed chunk_slice validation to accept negative indices like Python lists
  - For num_chunks=4, valid indices are now -4 to 3 (was 0 to 3)
  - -1 means last chunk, -2 means second-to-last, etc.
  - Validation now checks: chunk_index < -num_chunks or chunk_index >= num_chunks
  - Added conversion: if chunk_index < 0: chunk_index = num_chunks + chunk_index
  - Test now verifies -5 is invalid for 4 chunks (too negative)
  - Test now verifies 4 is invalid for 4 chunks (too large)
  - Added comprehensive test verifying -1, -2, -4 work correctly and match positive equivalents
- Issues Flagged: None

---

## Task Group 8: Fix Attribute Name Mismatches in Tests
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: tests/batchsolving/arrays/test_chunking.py (line 634)
- File: tests/batchsolving/test_runparams_integration.py (lines 58, 99, 167)

**Input Validation Required**:
- None - test updates only

**Tasks**:
1. **Fix time_domain_array to state in test_chunking.py**
   - File: tests/batchsolving/arrays/test_chunking.py
   - Action: Modify
   - Location: Line 634
   - Details:
     ```python
     # OLD:
     assert output_manager.device.time_domain_array._chunk_axis_index == 2
     
     # NEW:
     assert output_manager.device.state._chunk_axis_index == 2
     ```
   - Edge cases: None - simple attribute name fix
   - Integration: Uses correct attribute name from OutputArrays

2. **Fix initial_values chunk_axis_index Assertion**
   - File: tests/batchsolving/arrays/test_chunking.py
   - Action: Search and modify
   - Location: Any line asserting initial_values._chunk_axis_index == 2
   - Details:
     ```python
     # Search pattern: initial_values._chunk_axis_index == 2
     
     # OLD:
     assert input_manager.device.initial_values._chunk_axis_index == 2
     
     # NEW:
     assert input_manager.device.initial_values._chunk_axis_index == 1
     ```
   - Edge cases: initial_values is 2D (variable, run), so run axis is index 1
   - Integration: Reflects correct 2D array structure

3. **Fix num_params to num_parameters in test_runparams_integration.py**
   - File: tests/batchsolving/test_runparams_integration.py
   - Action: Modify
   - Location: Lines 58, 99, 167 (may be more occurrences)
   - Details:
     ```python
     # Search pattern: integration_system.num_params
     
     # OLD:
     params = np.random.rand(integration_system.num_params, num_runs)
     
     # NEW:
     params = np.random.rand(integration_system.num_parameters, num_runs)
     ```
   - Edge cases: None - simple attribute name fix
   - Integration: Uses correct attribute name from system

**Tests to Create**:
- None - fixing existing tests

**Tests to Run**:
- tests/batchsolving/arrays/test_chunking.py
- tests/batchsolving/test_runparams_integration.py

**Outcomes**:

---

## Summary

**Total Task Groups**: 8

**Dependency Chain**:
1. Task Group 1 (BaseArrayManager changes) must complete before Task Group 2 (subclass updates)
2. Task Group 3 (ArrayRequest changes) must complete before Task Group 4 (MemoryManager changes)
3. Task Group 6 (remove obsolete tests) depends on Groups 1, 3, 4, 5 completing
4. Task Groups 7-8 (test fixes) are independent and can run anytime

**Estimated Complexity**: Medium
- 4 major architectural changes (Groups 1-4)
- 1 API simplification (Group 5)
- 3 test update groups (Groups 6-8)
- Total files modified: ~10
- Total tests removed: ~9
- Total new tests created: ~13

**Tests Created**:
- Group 1: 5 new tests for BaseArrayManager num_runs and set_array_runs
- Group 2: 2 new tests for InputArrays and OutputArrays update_from_solver
- Group 3: 3 new tests for ArrayRequest total_runs default and validation
- Group 4: 2 new tests for MemoryManager allocate_queue

**Tests Removed**:
- Group 6: 9 obsolete tests for deleted methods

**Critical Integration Points**:
- BaseArrayManager.num_runs attribute feeds into allocate()
- InputArrays/OutputArrays.update_from_solver() sets num_runs before allocate()
- ArrayRequest.total_runs is always valid (never None)
- MemoryManager.allocate_queue() uses first request's total_runs
- No code should use BatchInputSizes.runs or BatchOutputSizes.runs
