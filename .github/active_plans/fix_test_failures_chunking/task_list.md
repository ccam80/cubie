# Implementation Task List
# Feature: Fix Test Failures in Chunking and Memory Management
# Plan Reference: .github/active_plans/fix_test_failures_chunking/agent_plan.md

## Task Group 1: ArrayRequest Enhancement (Data Carrier)
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/memory/array_requests.py (entire file)
- File: .github/context/cubie_internal_structure.md (lines 289-303 for memory subsystem)

**Input Validation Required**:
- total_runs: When provided (not None), validate it is an integer ≥ 1 using opt_getype_validator(int, 1)

**Tasks**:
1. **Add total_runs field to ArrayRequest**
   - File: src/cubie/memory/array_requests.py
   - Action: Modify
   - Details:
     ```python
     # Add new field after unchunkable field (around line 69)
     total_runs: Optional[int] = attrs.field(
         default=None,
         validator=opt_getype_validator(int, 1),
     )
     ```
   - Edge cases: None value is valid (indicates array not used for chunking), zero or negative rejected by validator
   - Integration: This field will be extracted by MemoryManager._extract_num_runs() in next task group

2. **Update ArrayRequest docstring**
   - File: src/cubie/memory/array_requests.py
   - Action: Modify
   - Details:
     ```python
     # In class docstring Parameters section (after unchunkable, around line 34):
     total_runs
         Total number of runs for chunking calculations. When ``None``, the
         array is not intended for run-axis chunking (e.g., driver_coefficients).
         Memory manager extracts this value to determine chunk parameters.
         Defaults to ``None``.
     
     # In Attributes section (after unchunkable, around line 47):
     total_runs
         Total number of runs for chunking calculations, or ``None`` if not
         applicable.
     ```
   - Edge cases: None
   - Integration: Documentation clarifies when total_runs should be provided

**Tests to Create**:
- Test file: tests/memory/test_array_requests.py
- Test function: test_array_request_accepts_total_runs
- Description: Verify ArrayRequest can be created with total_runs=100
- Test function: test_array_request_validates_total_runs_positive
- Description: Verify ArrayRequest raises ValueError for total_runs=0 or negative
- Test function: test_array_request_total_runs_defaults_to_none
- Description: Verify total_runs defaults to None when not provided

**Tests to Run**:
- tests/memory/test_array_requests.py::test_array_request_accepts_total_runs
- tests/memory/test_array_requests.py::test_array_request_validates_total_runs_positive
- tests/memory/test_array_requests.py::test_array_request_total_runs_defaults_to_none

**Outcomes**:
- Files Modified:
  * src/cubie/memory/array_requests.py (7 lines changed)
  * tests/memory/test_array_requests.py (54 lines added)
- Functions/Methods Added/Modified:
  * ArrayRequest class in array_requests.py - added total_runs field
  * ArrayRequest class docstring in array_requests.py - documented total_runs parameter and attribute
  * test_array_request_accepts_total_runs() in test_array_requests.py
  * test_array_request_validates_total_runs_positive() in test_array_requests.py
  * test_array_request_total_runs_defaults_to_none() in test_array_requests.py
- Implementation Summary:
  Added total_runs field to ArrayRequest class with Optional[int] type, default None, and validation that requires values >= 1 when provided. Updated docstring to document the new field in both Parameters and Attributes sections. Created three comprehensive test functions to verify the field accepts positive integers, rejects zero/negative values, and defaults to None.
- Issues Flagged: None

---

## Task Group 2: Memory Manager Decoupling (Extract num_runs from Requests)
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/memory/mem_manager.py (lines 1190-1230 for allocate_queue method)
- File: src/cubie/memory/array_requests.py (entire file, for ArrayRequest structure)

**Input Validation Required**:
- queued_requests: Validate it is a dict with structure instance_id -> {array_label -> ArrayRequest}
- Validate at least one ArrayRequest has total_runs that is not None
- Validate all non-None total_runs values are identical

**Tasks**:
1. **Add _extract_num_runs helper method to MemoryManager**
   - File: src/cubie/memory/mem_manager.py
   - Action: Create
   - Details:
     ```python
     # Add method before allocate_queue (around line 1180)
     def _extract_num_runs(
         self,
         queued_requests: Dict[str, Dict[str, ArrayRequest]],
     ) -> int:
         """Extract total_runs from queued allocation requests.
         
         Iterates through all ArrayRequest objects in queued_requests and returns
         the first non-None total_runs value found. Validates that all requests
         with total_runs set have the same value.
         
         Parameters
         ----------
         queued_requests
             Nested dict: instance_id -> {array_label -> ArrayRequest}
         
         Returns
         -------
         int
             The total number of runs for chunking calculations
         
         Raises
         ------
         ValueError
             If no requests contain total_runs, or if inconsistent values found
         
         Notes
         -----
         Requests with total_runs=None are ignored (e.g., driver_coefficients).
         At least one request must provide total_runs for chunking to work.
         """
         total_runs_values = set()
         
         # Iterate through nested dict structure
         for instance_id, requests_dict in queued_requests.items():
             for array_label, request in requests_dict.items():
                 if request.total_runs is not None:
                     total_runs_values.add(request.total_runs)
         
         # Validate we found at least one total_runs
         if len(total_runs_values) == 0:
             raise ValueError(
                 "No total_runs found in allocation requests. At least one "
                 "request must specify total_runs for chunking calculations."
             )
         
         # Validate all total_runs are consistent
         if len(total_runs_values) > 1:
             raise ValueError(
                 f"Inconsistent total_runs in requests: found {total_runs_values}. "
                 "All requests with total_runs must have the same value."
             )
         
         # Return the single value
         return total_runs_values.pop()
     ```
   - Edge cases: No total_runs found → ValueError, Multiple different values → ValueError, Single value → return it
   - Integration: Called by allocate_queue to get num_runs without accessing instance attributes

2. **Replace line 1203 in allocate_queue**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     ```python
     # Replace this line (around line 1203):
     # OLD:
     num_runs = triggering_instance.run_params.runs
     
     # NEW:
     num_runs = self._extract_num_runs(queued_requests)
     ```
   - Edge cases: None - _extract_num_runs handles validation
   - Integration: Decouples memory manager from triggering_instance.run_params

3. **Update allocate_queue docstring**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     ```python
     # Update docstring (around lines 1188-1196):
     # OLD text says "The num_runs value is extracted from triggering_instance.run_params.runs"
     # NEW text:
     """
     Process all queued allocation requests for a stream group.
     
     Allocates GPU memory for all queued requests in the same stream group as
     the triggering instance. Computes chunking parameters from total_runs
     values in ArrayRequest objects, then allocates and distributes arrays.
     
     The num_runs value is extracted from ArrayRequest.total_runs fields
     for determining chunk parameters. All requests must have consistent
     total_runs values.
     
     Parameters
     ----------
     triggering_instance
         Instance triggering allocation; used to identify stream group.
     
     Returns
     -------
     None
     
     Raises
     ------
     ValueError
         If no requests contain total_runs or values are inconsistent.
     """
     ```
   - Edge cases: None
   - Integration: Documentation reflects new extraction mechanism

**Tests to Create**:
- Test file: tests/memory/test_memmgmt.py
- Test function: test_extract_num_runs_finds_single_value
- Description: Verify _extract_num_runs returns correct value when one request has total_runs
- Test function: test_extract_num_runs_ignores_none_values
- Description: Verify _extract_num_runs works when some requests have total_runs=None
- Test function: test_extract_num_runs_raises_on_no_values
- Description: Verify _extract_num_runs raises ValueError when all total_runs are None
- Test function: test_extract_num_runs_raises_on_inconsistent_values
- Description: Verify _extract_num_runs raises ValueError when requests have different total_runs

**Tests to Run**:
- tests/memory/test_memmgmt.py::test_extract_num_runs_finds_single_value
- tests/memory/test_memmgmt.py::test_extract_num_runs_ignores_none_values
- tests/memory/test_memmgmt.py::test_extract_num_runs_raises_on_no_values
- tests/memory/test_memmgmt.py::test_extract_num_runs_raises_on_inconsistent_values

**Outcomes**:
- Files Modified:
  * src/cubie/memory/mem_manager.py (60 lines changed: 56 added for _extract_num_runs method, 1 changed for num_runs extraction, 3 changed in docstring)
  * tests/memory/test_memmgmt.py (107 lines added for test class and 4 test methods)
- Functions/Methods Added/Modified:
  * _extract_num_runs() method added to MemoryManager class in mem_manager.py
  * allocate_queue() method modified in mem_manager.py (line 1203 changed from triggering_instance.run_params.runs to self._extract_num_runs(queued_requests))
  * allocate_queue() docstring updated in mem_manager.py to reflect new extraction mechanism
  * TestExtractNumRuns class added in test_memmgmt.py
  * test_extract_num_runs_finds_single_value() in test_memmgmt.py
  * test_extract_num_runs_ignores_none_values() in test_memmgmt.py
  * test_extract_num_runs_raises_on_no_values() in test_memmgmt.py
  * test_extract_num_runs_raises_on_inconsistent_values() in test_memmgmt.py
- Implementation Summary:
  Added _extract_num_runs helper method to MemoryManager that iterates through queued ArrayRequest objects, collects non-None total_runs values, validates consistency, and returns the single value. Modified allocate_queue to call this helper instead of accessing triggering_instance.run_params.runs, thereby decoupling the memory manager from instance-specific attributes. Updated allocate_queue docstring to document the new extraction mechanism. Created comprehensive test class with 4 test methods covering single value extraction, None value handling, error on no values, and error on inconsistent values.
- Issues Flagged: None

---

## Task Group 3: Sizing Classes Enhancement (Expose runs Attribute)
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/outputhandling/output_sizes.py (lines 202-256 for BatchInputSizes, lines 258-363 for BatchOutputSizes)

**Input Validation Required**:
- None (read-only properties exposing existing data)

**Tasks**:
1. **Add runs property to BatchInputSizes**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Add property after from_solver method (around line 255)
     @property
     def runs(self) -> int:
         """Extract number of runs from initial_values shape.
         
         Returns
         -------
         int
             Number of runs (second element of initial_values shape).
         
         Notes
         -----
         The run count is stored in the second position of the
         initial_values and parameters tuples: (n_variables, n_runs).
         """
         return self.initial_values[1]
     ```
   - Edge cases: None - initial_values always has 2 elements
   - Integration: BaseArrayManager._get_total_runs() will access this property

2. **Add runs property to BatchOutputSizes**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Add property after from_solver method (around line 363)
     @property
     def runs(self) -> int:
         """Extract number of runs from state shape.
         
         Returns
         -------
         int
             Number of runs (third element of state shape).
         
         Notes
         -----
         The run count is stored in the third position of the
         state, observables, and summary tuples: (time, variable, n_runs).
         """
         return self.state[2]
     ```
   - Edge cases: None - state always has 3 elements
   - Integration: BaseArrayManager._get_total_runs() will access this property

**Tests to Create**:
- Test file: tests/outputhandling/test_output_sizes.py
- Test function: test_batch_input_sizes_exposes_runs
- Description: Verify BatchInputSizes.runs returns correct value from initial_values
- Test function: test_batch_output_sizes_exposes_runs
- Description: Verify BatchOutputSizes.runs returns correct value from state shape

**Tests to Run**:
- tests/outputhandling/test_output_sizes.py::test_batch_input_sizes_exposes_runs
- tests/outputhandling/test_output_sizes.py::test_batch_output_sizes_exposes_runs

**Outcomes**:
- Files Modified:
  * src/cubie/outputhandling/output_sizes.py (30 lines changed: 15 added for BatchInputSizes.runs property, 15 added for BatchOutputSizes.runs property)
  * tests/outputhandling/test_output_sizes.py (64 lines added for 2 test methods)
- Functions/Methods Added/Modified:
  * runs property added to BatchInputSizes class in output_sizes.py (lines 257-271)
  * runs property added to BatchOutputSizes class in output_sizes.py (lines 381-395)
  * test_batch_input_sizes_exposes_runs() in test_output_sizes.py
  * test_batch_output_sizes_exposes_runs() in test_output_sizes.py
- Implementation Summary:
  Added read-only runs property to BatchInputSizes that extracts the run count from the second element of initial_values tuple (n_variables, n_runs). Added read-only runs property to BatchOutputSizes that extracts the run count from the third element of state tuple (time, variable, n_runs). Both properties include comprehensive docstrings with Returns and Notes sections explaining the tuple structure. Created two test functions that verify the properties return correct values for different run counts (42, 100 for BatchInputSizes; 37, 200 for BatchOutputSizes).
- Issues Flagged: None

---

## Task Group 4: Array Manager Integration (Pass total_runs to ArrayRequest)
**Status**: [x]
**Dependencies**: Groups 1, 3

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 873-906 for allocate method)
- File: src/cubie/outputhandling/output_sizes.py (lines 202-363 for sizing classes with runs property)
- File: src/cubie/memory/array_requests.py (lines 19-75 for ArrayRequest with total_runs field)

**Input Validation Required**:
- None (validation happens in ArrayRequest validator)

**Tasks**:
1. **Add _get_total_runs helper to BaseArrayManager**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Create
   - Details:
     ```python
     # Add method before allocate() (around line 870)
     def _get_total_runs(self) -> Optional[int]:
         """Extract total runs from sizing metadata.
         
         Returns
         -------
         Optional[int]
             Number of runs if sizing metadata available and has runs
             attribute, otherwise None.
         
         Notes
         -----
         Returns None when _sizes is None or doesn't have a runs attribute.
         This allows graceful handling of edge cases during initialization.
         """
         if self._sizes is None:
             return None
         
         if not hasattr(self._sizes, 'runs'):
             return None
         
         return self._sizes.runs
     ```
   - Edge cases: _sizes is None → return None, _sizes has no runs → return None
   - Integration: Called by allocate() to get total_runs for ArrayRequest

2. **Modify allocate() to pass total_runs to ArrayRequest**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     # In allocate() method, modify ArrayRequest creation (around line 896):
     # OLD:
     request = ArrayRequest(
         shape=host_array.shape,
         dtype=device_array_object.dtype,
         memory=device_array_object.memory_type,
         chunk_axis_index=host_array_object._chunk_axis_index,
         unchunkable=not host_array_object.is_chunked,
     )
     
     # NEW:
     request = ArrayRequest(
         shape=host_array.shape,
         dtype=device_array_object.dtype,
         memory=device_array_object.memory_type,
         chunk_axis_index=host_array_object._chunk_axis_index,
         unchunkable=not host_array_object.is_chunked,
         total_runs=self._get_total_runs(),
     )
     ```
   - Edge cases: _get_total_runs() returns None → ArrayRequest accepts None
   - Integration: Requests now carry total_runs for memory manager to extract

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_basearraymanager.py
- Test function: test_get_total_runs_returns_none_when_sizes_none
- Description: Verify _get_total_runs returns None when _sizes is None
- Test function: test_get_total_runs_returns_runs_from_sizes
- Description: Verify _get_total_runs returns correct value when _sizes has runs
- Test function: test_allocate_passes_total_runs_to_request
- Description: Verify allocate() creates ArrayRequest with correct total_runs

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py::test_get_total_runs_returns_none_when_sizes_none
- tests/batchsolving/arrays/test_basearraymanager.py::test_get_total_runs_returns_runs_from_sizes
- tests/batchsolving/arrays/test_basearraymanager.py::test_allocate_passes_total_runs_to_request

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/arrays/BaseArrayManager.py (29 lines changed: 28 lines added for _get_total_runs method, 1 line modified in allocate to pass total_runs)
  * tests/batchsolving/arrays/test_basearraymanager.py (120 lines added for TestGetTotalRuns class with 3 test methods)
- Functions/Methods Added/Modified:
  * _get_total_runs() method added to BaseArrayManager class in BaseArrayManager.py (lines 873-899)
  * allocate() method modified in BaseArrayManager class to pass total_runs to ArrayRequest (line 929)
  * test_get_total_runs_returns_none_when_sizes_none() in test_basearraymanager.py
  * test_get_total_runs_returns_runs_from_sizes() in test_basearraymanager.py
  * test_allocate_passes_total_runs_to_request() in test_basearraymanager.py
- Implementation Summary:
  Added _get_total_runs() helper method that safely extracts the run count from sizing metadata by checking if _sizes is None or lacks a runs attribute before accessing it. Modified allocate() to call _get_total_runs() and pass the result to ArrayRequest constructor as the total_runs parameter. Created comprehensive test class TestGetTotalRuns with three test methods: one verifying None is returned when _sizes is None, one verifying the correct run count is returned when sizes are available, and one verifying that allocate() creates ArrayRequests with the correct total_runs value extracted from sizing metadata.
- Issues Flagged: None

---

## Task Group 5: Chunk Metadata Propagation (Fix chunk_slice and Allocation Hook)
**Status**: [x]
**Dependencies**: Groups 1, 2, 4

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 393-451 for _on_allocation_complete, lines 126-187 for chunk_slice)
- File: src/cubie/memory/array_requests.py (lines 77-117 for ArrayResponse structure)

**Input Validation Required**:
- None (ArrayResponse provides validated chunk_length and chunks)

**Tasks**:
1. **Verify _on_allocation_complete propagates chunk metadata**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Verify (no changes needed)
   - Details:
     ```python
     # Lines 423-436 already implement this correctly:
     # Extract chunk parameters from response
     chunks = response.chunks
     chunk_length = response.chunk_length
     
     for array_label in self._needs_reallocation:
         try:
             self.device.attach(array_label, arrays[array_label])
             # Store chunked_shape and chunk parameters in ManagedArray
             if array_label in response.chunked_shapes:
                 for container in (self.device, self.host):
                     array = container.get_managed_array(array_label)
                     array.chunked_shape = chunked_shapes[array_label]
                     array.chunk_length = chunk_length  # ✓ Already present
                     array.num_chunks = chunks          # ✓ Already present
     ```
   - Edge cases: Already handled - only sets for arrays in chunked_shapes
   - Integration: ManagedArray objects receive chunk_length and num_chunks from response

2. **Verify chunk_slice uses correct chunk metadata**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Verify (no changes needed)
   - Details:
     ```python
     # Lines 126-187 already implement slice logic correctly:
     # The method uses self.chunk_length and self.num_chunks which are set
     # by _on_allocation_complete. Logic is:
     # 1. Return full array if no chunking active (lines 161-166)
     # 2. Validate chunk_index in range (lines 169-173)
     # 3. Compute start = chunk_index * chunk_length (line 175)
     # 4. For final chunk, use end=None; otherwise end=start+chunk_length (177-180)
     # 5. Build slice tuple and return sliced array (lines 183-186)
     ```
   - Edge cases: Already handled - final chunk uses None endpoint, validates range
   - Integration: Correct - uses ManagedArray.chunk_length and num_chunks set by allocation hook

3. **Add integration test for chunk metadata flow**
   - File: tests/batchsolving/arrays/test_basearraymanager.py
   - Action: Create test
   - Details:
     ```python
     # Test that verifies complete flow:
     # 1. Create manager with _sizes containing runs
     # 2. Call allocate() to create requests with total_runs
     # 3. Simulate allocation response with chunk_length and chunks
     # 4. Verify ManagedArray has correct chunk_length and num_chunks
     # 5. Verify chunk_slice returns correct shapes
     ```
   - Edge cases: Test with num_chunks=1 (no chunking), num_chunks=4 (chunking), dangling chunk
   - Integration: End-to-end test of allocation → propagation → slicing

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_basearraymanager.py
- Test function: test_allocation_complete_sets_chunk_metadata
- Description: Verify _on_allocation_complete sets chunk_length and num_chunks on ManagedArray
- Test function: test_chunk_slice_uses_chunk_metadata_from_response
- Description: Verify chunk_slice returns correct shapes when chunk metadata is set
- Test function: test_chunk_metadata_flow_integration
- Description: End-to-end test of allocate → response → chunk_slice

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py::test_allocation_complete_sets_chunk_metadata
- tests/batchsolving/arrays/test_basearraymanager.py::test_chunk_slice_uses_chunk_metadata_from_response
- tests/batchsolving/arrays/test_basearraymanager.py::test_chunk_metadata_flow_integration

**Outcomes**:
- Files Modified:
  * tests/batchsolving/arrays/test_basearraymanager.py (397 lines added)
- Functions/Methods Added/Modified:
  * TestChunkMetadataFlow class added in test_basearraymanager.py
  * test_allocation_complete_sets_chunk_metadata() in test_basearraymanager.py
  * test_chunk_slice_uses_chunk_metadata_from_response() in test_basearraymanager.py
  * test_chunk_metadata_flow_integration() in test_basearraymanager.py
- Implementation Summary:
  Verified that _on_allocation_complete (lines 423-436) correctly extracts chunk_length and chunks from ArrayResponse and sets them on both host and device ManagedArray objects for arrays in chunked_shapes. Verified that chunk_slice method (lines 126-187) correctly uses self.chunk_length and self.num_chunks to compute chunk slices, with proper handling of the final chunk via None endpoint and validation of chunk_index range. Created comprehensive integration test class TestChunkMetadataFlow with three test methods: (1) test_allocation_complete_sets_chunk_metadata verifies chunk metadata propagation from ArrayResponse to both host and device ManagedArray objects; (2) test_chunk_slice_uses_chunk_metadata_from_response verifies chunk_slice computes correct slices using chunk metadata with detailed data pattern verification; (3) test_chunk_metadata_flow_integration provides end-to-end testing of the complete flow with three scenarios: num_chunks=1 (no chunking), num_chunks=4 (even chunking), and num_chunks=5 (dangling chunk with smaller final chunk).
- Issues Flagged: None - existing implementation is correct, verification confirmed

---

## Task Group 6: Validation and Regression Testing
**Status**: [x]
**Dependencies**: Groups 1, 2, 3, 4, 5

**Required Context**:
- File: .github/active_plans/fix_test_failures_chunking/human_overview.md (entire file for context)
- No specific source files needed - running existing tests

**Input Validation Required**:
- None (validation tests only)

**Tasks**:
1. **Run all previously failing tests**
   - File: N/A (test execution only)
   - Action: Run tests
   - Details:
     ```bash
     # Run all tests in memory subsystem
     pytest tests/memory/ -v
     
     # Run all tests in batchsolving arrays subsystem
     pytest tests/batchsolving/arrays/ -v
     
     # Run specific runparams tests
     pytest tests/batchsolving/test_runparams.py -v
     pytest tests/batchsolving/test_runparams_integration.py -v
     
     # Run full test suite to check for regressions
     pytest tests/ -x --tb=short
     ```
   - Edge cases: Tests may reveal additional integration issues
   - Integration: Verifies all 35+ failing tests now pass

2. **Document any remaining failures**
   - File: .github/active_plans/fix_test_failures_chunking/test_results.md
   - Action: Create (if failures remain)
   - Details:
     ```markdown
     # Test Results After Implementation
     
     ## Date: [Auto-filled by run_tests agent]
     
     ## Summary
     - Total tests run: [count]
     - Passed: [count]
     - Failed: [count]
     - Skipped: [count]
     
     ## Failing Tests
     [List any remaining failures with error messages]
     
     ## Next Steps
     [Recommendations for fixing remaining failures]
     ```
   - Edge cases: None
   - Integration: Provides feedback for any additional work needed

**Tests to Create**:
- None (running existing tests)

**Tests to Run**:
- tests/memory/test_array_requests.py
- tests/memory/test_memmgmt.py
- tests/batchsolving/arrays/test_basearraymanager.py
- tests/batchsolving/arrays/test_batchinputarrays.py
- tests/batchsolving/arrays/test_batchoutputarrays.py
- tests/batchsolving/arrays/test_chunking.py
- tests/batchsolving/test_runparams.py
- tests/batchsolving/test_runparams_integration.py
- tests/batchsolving/test_solver.py

**Outcomes**:
- Files Modified: None (test execution task group)
- Implementation Summary:
  Task Group 6 contains no code implementation tasks - it is purely a test execution and validation group. All implementation work from Groups 1-5 has been completed. The run_tests agent should execute the comprehensive test suite listed in "Tests to Run" to verify that all 35+ previously failing tests now pass. This includes tests across the memory subsystem (ArrayRequest, MemoryManager extraction), batchsolving arrays subsystem (BaseArrayManager, InputArrays, OutputArrays, chunking), and runparams subsystem. The test execution will validate the complete data flow: ArrayManager creates ArrayRequest with total_runs → MemoryManager extracts num_runs from requests → MemoryManager computes chunk_length and creates ArrayResponse → RunParams updates from ArrayResponse → chunk_slice uses correct metadata.
- Issues Flagged: None - awaiting test execution by run_tests agent

---

# Summary

## Total Task Groups: 6

## Dependency Chain:
```
Group 1 (ArrayRequest) ─┬─> Group 2 (MemoryManager)
                        │
Group 3 (Sizing)        ├─> Group 4 (ArrayManager) ─> Group 5 (Propagation) ─> Group 6 (Validation)
```

## Tests Created: 17
- Group 1: 3 tests (ArrayRequest validation)
- Group 2: 4 tests (MemoryManager extraction)
- Group 3: 2 tests (Sizing classes)
- Group 4: 3 tests (ArrayManager integration)
- Group 5: 3 tests (Chunk metadata flow)
- Group 6: 2+ test runs (validation suite)

## Estimated Complexity: Medium
- Architectural changes are surgical and well-defined
- No breaking changes to external API
- Changes follow existing patterns in codebase
- Test coverage is comprehensive
- All implementations are straightforward data flow modifications
