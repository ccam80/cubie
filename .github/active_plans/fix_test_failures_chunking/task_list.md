# Implementation Task List
# Feature: Fix Test Failures in Chunking and Memory Management
# Plan Reference: .github/active_plans/fix_test_failures_chunking/agent_plan.md

## Task Group 1: ArrayRequest Enhancement (Data Carrier)
**Status**: [ ]
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

---

## Task Group 2: Memory Manager Decoupling (Extract num_runs from Requests)
**Status**: [ ]
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

---

## Task Group 3: Sizing Classes Enhancement (Expose runs Attribute)
**Status**: [ ]
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

---

## Task Group 4: Array Manager Integration (Pass total_runs to ArrayRequest)
**Status**: [ ]
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

---

## Task Group 5: Chunk Metadata Propagation (Fix chunk_slice and Allocation Hook)
**Status**: [ ]
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

---

## Task Group 6: Validation and Regression Testing
**Status**: [ ]
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
