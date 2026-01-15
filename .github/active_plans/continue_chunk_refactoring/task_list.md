# Implementation Task List
# Feature: Continue Chunk Removal Refactoring
# Plan Reference: .github/active_plans/continue_chunk_refactoring/agent_plan.md

## Task Group 1: Create RunParams Class
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 1-100, lines 60-100)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- index: Check 0 <= index < num_chunks (raise IndexError if out of bounds)
- num_chunks: Check num_chunks >= 1
- chunk_length: Check chunk_length > 0 when num_chunks > 1
- runs: Check runs > 0
- duration: Check duration >= 0.0
- warmup: Check warmup >= 0.0

**Tasks**:
1. **Create RunParams attrs class**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Create
   - Details:
     ```python
     from attrs import define, field, evolve
     from attrs.validators import instance_of as attrsval_instance_of
     from cubie._utils import getype_validator
     
     @define(frozen=True)
     class RunParams:
         """Run parameters with optional chunking metadata.
         
         Chunking always occurs along the run axis.
         
         Parameters
         ----------
         duration : float
             Full duration of the simulation window.
         warmup : float
             Full warmup time before the main simulation.
         t0 : float
             Initial integration time.
         runs : int
             Total number of runs in the batch.
         num_chunks : int, default=1
             Number of chunks the batch is divided into.
         chunk_length : int, default=0
             Number of runs per chunk (except possibly the last).
         
         Notes
         -----
         When num_chunks=1, no chunking has occurred.
         When num_chunks>1, chunk_length represents the standard chunk size.
         """
         duration: float = field(validator=getype_validator(float, 0.0))
         warmup: float = field(validator=getype_validator(float, 0.0))
         t0: float = field(validator=getype_validator(float, 0.0))
         runs: int = field(validator=getype_validator(int, 1))
         num_chunks: int = field(default=1, repr=False, validator=getype_validator(int, 1))
         chunk_length: int = field(default=0, repr=False, validator=getype_validator(int, 0))
         
         def __getitem__(self, index: int) -> "RunParams":
             """Return RunParams for a specific chunk.
             
             Parameters
             ----------
             index : int
                 Chunk index (0-based).
             
             Returns
             -------
             RunParams
                 New RunParams instance with runs set to chunk size.
             
             Raises
             ------
             IndexError
                 If index is out of range [0, num_chunks).
             
             Notes
             -----
             For the last chunk (index == num_chunks - 1), the number of runs
             is calculated as runs - (num_chunks - 1) * chunk_length to handle
             the "dangling" chunk case.
             """
             # Validation
             if index < 0 or index >= self.num_chunks:
                 raise IndexError(
                     f"Chunk index {index} out of range [0, {self.num_chunks})"
                 )
             
             # Compute runs for this chunk
             if index == self.num_chunks - 1:
                 # Last chunk: calculate remaining runs
                 chunk_runs = self.runs - (self.num_chunks - 1) * self.chunk_length
             else:
                 # Standard chunk
                 chunk_runs = self.chunk_length
             
             return evolve(self, runs=chunk_runs)
         
         def update_from_allocation(self, response: "ArrayResponse") -> "RunParams":
             """Update with chunking metadata from allocation response.
             
             Parameters
             ----------
             response : ArrayResponse
                 Allocation response containing chunking information.
             
             Returns
             -------
             RunParams
                 New RunParams instance with updated chunking metadata.
             
             Notes
             -----
             Extracts num_chunks and chunk_length from the response. When
             num_chunks=1, chunk_length is set equal to runs (no chunking).
             """
             num_chunks = response.chunks
             
             # Calculate chunk_length from total runs and num_chunks
             if num_chunks == 1:
                 chunk_length = self.runs
             else:
                 from numpy import ceil as np_ceil
                 chunk_length = int(np_ceil(self.runs / num_chunks))
             
             return evolve(
                 self,
                 num_chunks=num_chunks,
                 chunk_length=chunk_length
             )
     ```
   - Edge cases:
     - Single chunk (num_chunks=1): __getitem__(0) returns runs unchanged
     - Exact division: All chunks have same size
     - Dangling chunk: Last chunk smaller than chunk_length
     - Invalid index: Raise IndexError with clear message
   - Integration: Place definition before ChunkParams class definition (will be removed later)

2. **Add type annotation import for ArrayResponse**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # In the TYPE_CHECKING block, add:
     from cubie.memory.array_requests import ArrayResponse
     ```
   - Edge cases: None
   - Integration: Needed for type hints in RunParams methods

**Tests to Create**:
- Test file: tests/batchsolving/test_runparams.py
- Test function: test_runparams_creation
- Description: Verify RunParams can be created with valid parameters
- Test function: test_runparams_getitem_single_chunk
- Description: Verify __getitem__(0) returns full runs when num_chunks=1
- Test function: test_runparams_getitem_multiple_chunks
- Description: Verify __getitem__ returns correct runs for each chunk
- Test function: test_runparams_getitem_dangling_chunk
- Description: Verify last chunk gets correct remaining runs
- Test function: test_runparams_getitem_exact_division
- Description: Verify all chunks equal when runs % chunk_length == 0
- Test function: test_runparams_getitem_out_of_bounds
- Description: Verify IndexError raised for invalid indices
- Test function: test_runparams_update_from_allocation
- Description: Verify update_from_allocation sets num_chunks and chunk_length

**Tests to Run**:
- tests/batchsolving/test_runparams.py

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (137 lines added)
- Functions/Methods Added/Modified:
  * RunParams class created in BatchSolverKernel.py with:
    - __getitem__(index) method for per-chunk parameter access
    - update_from_allocation(response) method for chunking metadata updates
  * Added getype_validator to imports from cubie._utils
- Implementation Summary:
  * Created RunParams attrs class with frozen=True for immutability
  * Implemented chunk indexing with dangling chunk support
  * Implemented allocation response processing to extract chunking metadata
  * Used getype_validator for input validation as specified
  * Imported numpy.ceil as np_ceil following project conventions
  * Placed RunParams class definition before FullRunParams class
- Tests Created:
  * tests/batchsolving/test_runparams.py with 16 test functions:
    - test_runparams_creation
    - test_runparams_creation_validates_duration
    - test_runparams_creation_validates_warmup
    - test_runparams_creation_validates_runs
    - test_runparams_creation_validates_num_chunks
    - test_runparams_getitem_single_chunk
    - test_runparams_getitem_multiple_chunks
    - test_runparams_getitem_dangling_chunk
    - test_runparams_getitem_exact_division
    - test_runparams_getitem_out_of_bounds
    - test_runparams_update_from_allocation
    - test_runparams_update_from_allocation_single_chunk
    - test_runparams_update_from_allocation_dangling_chunk
    - test_runparams_immutability
    - test_runparams_evolve_pattern
- Issues Flagged: None

---

## Task Group 2: Remove Fields from ArrayResponse
**Status**: [x]
**Dependencies**: Groups []

**Required Context**:
- File: src/cubie/memory/array_requests.py (entire file)
- File: .github/context/cubie_internal_structure.md (lines 289-304)

**Input Validation Required**:
- None (this task only removes fields, no new validation)

**Tasks**:
1. **Remove axis_length and dangling_chunk_length from ArrayResponse**
   - File: src/cubie/memory/array_requests.py
   - Action: Modify
   - Details:
     The current ArrayResponse class has only `arr`, `chunks`, and `chunk_axis` fields.
     No fields need to be removed - the plan assumed fields that don't exist.
     This task is a NO-OP verification task.
     
     Verify the current ArrayResponse definition:
     ```python
     @attrs.define
     class ArrayResponse:
         arr: dict[str, DeviceNDArrayBase] = attrs.field(...)
         chunks: int = attrs.field(default=1)
         chunk_axis: str = attrs.field(default="run", ...)
     ```
     
     Confirm that there are no `axis_length` or `dangling_chunk_length` fields
     in the current implementation.
   - Edge cases: None
   - Integration: ArrayResponse is already in target state

**Tests to Create**:
- Test file: tests/memory/test_array_requests.py
- Test function: test_array_response_has_no_axis_length
- Description: Verify ArrayResponse does not have axis_length field
- Test function: test_array_response_has_no_dangling_chunk_length  
- Description: Verify ArrayResponse does not have dangling_chunk_length field

**Tests to Run**:
- tests/memory/test_array_requests.py::test_array_response_has_no_axis_length
- tests/memory/test_array_requests.py::test_array_response_has_no_dangling_chunk_length

**Outcomes**:
- Files Modified:
  * src/cubie/memory/array_requests.py (6 lines removed)
  * tests/memory/test_array_requests.py (28 lines added)
- Functions/Methods Added/Modified:
  * Removed axis_length field from ArrayResponse class (line 121)
  * Removed dangling_chunk_length field from ArrayResponse class (line 133-135)
  * Added test_array_response_has_no_axis_length() in test_array_requests.py
  * Added test_array_response_has_no_dangling_chunk_length() in test_array_requests.py
- Implementation Summary:
  * Task description stated fields didn't exist, but they did exist in the source code
  * Removed axis_length field (was at line 121-123)
  * Removed dangling_chunk_length field (was at line 133-135)
  * Created two test functions to verify these fields don't exist
  * Tests use hasattr() to confirm fields are not present on ArrayResponse instances
  * ArrayResponse now has: arr, chunks, chunk_length, chunked_shapes, chunked_slices
- Issues Flagged:
  * Task description was inaccurate - claimed fields didn't exist when they did
  * Note: chunk_axis field mentioned in task description does NOT exist in current code
  * Remaining fields (chunk_length, chunked_shapes, chunked_slices) are not mentioned in task but remain in code

---

## Task Group 3: Update BatchSolverKernel to Use RunParams
**Status**: [x]
**Dependencies**: Groups [1]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file)
- File: .github/context/cubie_internal_structure.md (lines 318-397)

**Input Validation Required**:
- None (validation delegated to RunParams class)

**Tasks**:
1. **Add run_params field to BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     In `BatchSolverKernel.__init__`, after the line:
     ```python
     self.chunk_axis = "run"
     ```
     
     Add:
     ```python
     self.run_params = RunParams(
         duration=precision(0.0),
         warmup=precision(0.0),
         t0=precision(0.0),
         runs=1
     )
     ```
   - Edge cases: Default values should match existing defaults
   - Integration: Initializes alongside existing time parameters

2. **Update _on_allocation callback to update run_params**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     Replace the current `_on_allocation` method:
     ```python
     def _on_allocation(self, response: Any) -> None:
         """Record the number of chunks required by the memory manager."""
         self.chunks = response.chunks
     ```
     
     With:
     ```python
     def _on_allocation(self, response: Any) -> None:
         """Update run parameters with chunking metadata from allocation."""
         self.chunks = response.chunks
         self.run_params = self.run_params.update_from_allocation(response)
     ```
   - Edge cases: Should preserve backward compatibility with existing chunks attribute
   - Integration: Called by memory manager after allocation

3. **Update run() method to create and update run_params**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     In the `run()` method, after the lines:
     ```python
     self._duration = duration
     self._warmup = warmup
     self._t0 = t0
     ```
     
     Add:
     ```python
     # Update run params with actual values
     self.run_params = RunParams(
         duration=duration,
         warmup=warmup,
         t0=t0,
         runs=numruns
     )
     ```
     
     This replaces the old pattern of storing these separately.
   - Edge cases: Values should use the validated duration/warmup/t0 (np_float64 casts)
   - Integration: Executed before memory allocation

4. **Update chunk iteration to use run_params**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     In the `run()` method, replace the chunk iteration logic.
     
     Find the loop:
     ```python
     for i in range(self.chunks):
         indices = slice(i * chunk_params.size, (i + 1) * chunk_params.size)
     ```
     
     Before this loop, add logic to get per-chunk params:
     ```python
     for i in range(self.chunks):
         # Get parameters for this specific chunk
         chunk_run_params = self.run_params[i]
         chunk_runs = chunk_run_params.runs
         
         indices = slice(i * chunk_params.size, (i + 1) * chunk_params.size)
     ```
     
     Then update the kernel_runs calculation to use chunk_runs:
     ```python
     # Use the chunk-local run count
     kernel_runs = int(chunk_runs)
     ```
     
     This replaces the conditional logic using chunk_axis.
   - Edge cases: Ensure chunk_runs is correctly computed for dangling chunks
   - Integration: Changes how kernel_runs is determined for each chunk iteration

5. **Add properties to access run_params fields**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     Update existing properties to delegate to run_params:
     
     The `duration`, `warmup`, `t0` properties already exist and use underscored
     variables. These should continue to work as-is since we're still updating
     the underscored variables.
     
     Add a new property for accessing num_runs from run_params:
     ```python
     @property
     def total_runs(self) -> int:
         """Total number of runs in the full batch."""
         return self.run_params.runs
     ```
     
     The existing `num_runs` property (line ~473) should continue to work
     as it stores a separate value during run() execution.
   - Edge cases: Properties should remain read-only
   - Integration: Provides consistent access to run parameters

**Tests to Create**:
- Test file: tests/batchsolving/test_batchsolverkernel.py
- Test function: test_runparams_initialized_on_construction
- Description: Verify BatchSolverKernel initializes run_params with defaults
- Test function: test_runparams_updated_on_run
- Description: Verify run_params gets updated when run() is called
- Test function: test_on_allocation_updates_runparams
- Description: Verify _on_allocation updates run_params chunking metadata
- Test function: test_chunk_iteration_uses_runparams
- Description: Verify chunk iteration correctly uses run_params[i]

**Tests to Run**:
- tests/batchsolving/test_SolverKernel.py::TestRunParamsIntegration

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (21 lines changed)
  * tests/batchsolving/test_SolverKernel.py (126 lines added)
- Functions/Methods Added/Modified:
  * BatchSolverKernel.__init__() - Added run_params initialization
  * BatchSolverKernel._on_allocation() - Added run_params.update_from_allocation() call
  * BatchSolverKernel.run() - Added run_params creation before allocation
  * BatchSolverKernel.run() chunk iteration - Modified to use run_params[i]
  * BatchSolverKernel.total_runs property - Added new property returning run_params.runs
- Implementation Summary:
  * Initialized run_params in __init__ with default values using precision() function
  * Updated _on_allocation to call run_params.update_from_allocation(response)
  * Modified run() to create RunParams before memory allocation with actual timing values
  * Updated chunk iteration loop to extract chunk_run_params = self.run_params[i]
  * Added total_runs property to access run_params.runs
  * Created comprehensive test class TestRunParamsIntegration with 5 test methods
  * Tests verify initialization, updates during run(), allocation updates, chunk iteration, and property access
- Issues Flagged: None

---

## Task Group 4: Remove ChunkParams Class
**Status**: [x]
**Dependencies**: Groups [1, 3]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 60-100, lines 661-700)

**Input Validation Required**:
- None (removing deprecated class)

**Tasks**:
1. **Remove ChunkParams attrs class definition**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Delete
   - Details:
     Remove the entire ChunkParams class definition (lines ~60-76):
     ```python
     @define(frozen=True)
     class ChunkParams:
         """Chunked execution parameters calculated for a batch run.
         ...
         """
         duration: float
         warmup: float
         t0: float
         size: int
         runs: int
     ```
   - Edge cases: Ensure no imports reference ChunkParams
   - Integration: Class is now fully replaced by RunParams

2. **Remove chunk_run() method**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Delete
   - Details:
     Remove the `chunk_run()` method (lines ~661-700) that creates ChunkParams:
     ```python
     def chunk_run(
         self,
         chunk_axis: str,
         duration: float,
         warmup: float,
         t0: float,
         numruns: int,
         chunks: int,
     ) -> ChunkParams:
         # ... method body ...
     ```
     
     This functionality is now handled by RunParams.__getitem__.
   - Edge cases: Ensure no callers remain in the codebase
   - Integration: Method is no longer needed

3. **Update run() method to remove chunk_run() call**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     In the `run()` method, remove the call to chunk_run:
     ```python
     # ------------ from here on dimensions are "chunked" -----------------
     chunk_params = self.chunk_run(
         chunk_axis,
         duration,
         warmup,
         t0,
         numruns,
         self.chunks,
     )
     chunk_warmup = chunk_params.warmup
     chunk_t0 = chunk_params.t0
     ```
     
     Replace with simpler initialization using run_params:
     ```python
     # ------------ from here on dimensions are "chunked" -----------------
     # Get initial chunk parameters (warmup and t0 may be modified per chunk)
     chunk_warmup = self.run_params.warmup
     chunk_t0 = self.run_params.t0
     ```
     
     The chunk_params.duration usage should be replaced with accessing
     from run_params directly, and chunk_params.size should be calculated
     inline based on chunk index logic.
   - Edge cases: Maintain time-chunking behavior if chunk_axis=="time"
   - Integration: Simplifies chunk parameter handling

4. **Update chunk iteration to compute chunk_params.size inline**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     In the chunk iteration loop, compute the size value inline:
     ```python
     for i in range(self.chunks):
         chunk_run_params = self.run_params[i]
         chunk_runs = chunk_run_params.runs
         
         # Calculate chunk size based on output length or runs
         if chunk_axis == "run":
             chunk_size = chunk_runs
         else:  # chunk_axis == "time"
             chunk_size = int(np_ceil(self.output_length / self.chunks))
         
         indices = slice(i * chunk_size, (i + 1) * chunk_size)
     ```
   - Edge cases: Handle both run and time chunking axes
   - Integration: Preserves chunking behavior for both axes

**Tests to Create**:
- Test file: tests/batchsolving/test_batchsolverkernel.py
- Test function: test_chunkparams_not_defined
- Description: Verify ChunkParams class no longer exists
- Test function: test_chunk_run_not_defined
- Description: Verify chunk_run method no longer exists

**Tests to Run**:
- tests/batchsolving/test_SolverKernel.py::TestChunkParamsRemoval::test_chunkparams_not_defined
- tests/batchsolving/test_SolverKernel.py::TestChunkParamsRemoval::test_chunk_run_not_defined

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (104 lines removed, 8 lines added)
  * tests/batchsolving/test_SolverKernel.py (14 lines added)
- Functions/Methods Added/Modified:
  * Removed ChunkParams class definition (lines 208-303)
  * Removed self.chunk_params initialization in BatchSolverKernel.__init__
  * Modified BatchSolverKernel._on_allocation() - removed ChunkParams.from_allocation_response call
  * Modified BatchSolverKernel.run() - replaced chunk_params usage with run_params
  * Modified chunk iteration loop - removed chunkparams variable reference
  * Added TestChunkParamsRemoval.test_chunkparams_not_defined() test
  * Added TestChunkParamsRemoval.test_chunk_run_not_defined() test
- Implementation Summary:
  * Removed entire ChunkParams class (96 lines including docstrings and methods)
  * Removed chunk_params instance variable initialization in __init__
  * Updated _on_allocation to only use run_params.update_from_allocation()
  * Replaced chunk_params[0] with run_params[0] and extracted values directly
  * Removed redundant chunkparams variable in chunk iteration loop
  * Created two verification tests to ensure ChunkParams cannot be imported and chunk_run method doesn't exist
  * All ChunkParams functionality now handled by RunParams.__getitem__
- Issues Flagged:
  * Note: The chunk_run() method mentioned in task did not exist in current code
  * ChunkParams was initialized in __init__ but never used directly after allocation
  * FullRunParams class still exists and may be candidate for future removal

---

## Task Group 5: Remove Memory Manager Helper Methods
**Status**: [x]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: src/cubie/memory/mem_manager.py (entire file)
- File: .github/context/cubie_internal_structure.md (lines 289-304)

**Input Validation Required**:
- num_runs: Check num_runs > 0 in allocate_queue

**Tasks**:
1. **Update allocate_queue to accept num_runs from triggering instance**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     The current `allocate_queue` method needs to extract num_runs from
     the triggering instance instead of computing axis_length from requests.
     
     Modify the method signature and implementation:
     ```python
     def allocate_queue(
         self,
         triggering_instance: object,
         limit_type: str = "group",
         chunk_axis: str = "run",
     ) -> None:
         """Process all queued requests for a stream group with coordinated chunking.
         
         Parameters
         ----------
         triggering_instance
             The instance that triggered queue processing.
         limit_type
             Limiting strategy: "group" for aggregate limits or "instance" for
             individual instance limits. Defaults to "group".
         chunk_axis
             Axis along which to chunk arrays if needed. Defaults to "run".
         
         Notes
         -----
         Processes all pending requests in the same stream group, applying
         coordinated chunking based on the specified limit type. Calls
         allocation_ready_hook for each instance with their results.
         
         The num_runs value is extracted from triggering_instance.run_params.runs
         for determining chunk parameters.
         """
         stream_group = self.get_stream_group(triggering_instance)
         peers = self.stream_groups.get_instances_in_group(stream_group)
         stream = self.get_stream(triggering_instance)
         queued_requests = self._queued_allocations.get(stream_group, {})
         n_queued = len(queued_requests)
         if not queued_requests:
             return None
         elif n_queued == 1:
             for instance_id, requests_dict in queued_requests.items():
                 self.single_request(
                     instance=instance_id,
                     requests=requests_dict,
                     chunk_axis=chunk_axis,
                 )
         else:
             # Extract num_runs from triggering instance
             num_runs = triggering_instance.run_params.runs
             
             numchunks = 1  # safe default
             if limit_type == "group":
                 available_memory = self.get_available_group(stream_group)
                 request_size = sum(
                     [
                         get_total_request_size(request)
                         for request in queued_requests.values()
                     ]
                 )
                 numchunks = self.get_chunks(request_size, available_memory)
             elif limit_type == "instance":
                 numchunks = 0
                 for instance_id, requests_dict in queued_requests.items():
                     available_memory = self.get_available_single(instance_id)
                     request_size = get_total_request_size(requests_dict)
                     chunks = self.get_chunks(request_size, available_memory)
                     numchunks = chunks if chunks > numchunks else numchunks
             
             notaries = set(peers) - set(queued_requests.keys())
             for instance_id, requests_dict in queued_requests.items():
                 chunked_request = self.chunk_arrays(
                     requests_dict, numchunks, chunk_axis
                 )
                 arrays = self.allocate_all(
                     chunked_request, instance_id, stream=stream
                 )
                 response = ArrayResponse(
                     arr=arrays, chunks=numchunks, chunk_axis=chunk_axis
                 )
                 self.registry[instance_id].allocation_ready_hook(response)
             
             for peer in notaries:
                 self.registry[peer].allocation_ready_hook(
                     ArrayResponse(
                         arr={}, chunks=numchunks, chunk_axis=chunk_axis
                     )
                 )
         return None
     ```
     
     The key change is extracting num_runs from `triggering_instance.run_params.runs`
     instead of computing axis_length from array requests.
   - Edge cases: Handle instances without run_params attribute gracefully
   - Integration: Called by BatchSolverKernel after queueing requests

2. **Verify no references to removed helper methods**
   - File: src/cubie/memory/mem_manager.py
   - Action: Verify
   - Details:
     Search the codebase to confirm that the following methods/functions
     do not exist or are not called:
     - `get_chunk_axis_length()` - should not exist
     - `replace_with_chunked_size()` - check if exists and is used
     - `compute_per_chunk_slice()` - check if exists and is used
     
     If `replace_with_chunked_size()` exists, examine whether it's used
     only in `compute_chunked_shapes()`. If so, inline the logic.
     
     If `compute_per_chunk_slice()` exists, examine usage in BaseArrayManager.
     Determine if it's needed for chunked transfers or can be simplified.
     
     Based on the current source, these methods do NOT appear to exist in
     mem_manager.py, so this is a verification task only.
   - Edge cases: None
   - Integration: Ensures clean codebase without unused methods

**Tests to Create**:
- Test file: tests/memory/test_memmgmt.py
- Test function: test_allocate_queue_extracts_num_runs
- Description: Verify allocate_queue correctly extracts num_runs from triggering instance
- Test function: test_allocate_queue_chunks_correctly
- Description: Verify chunking calculations work with num_runs from instance

**Tests to Run**:
- tests/memory/test_memmgmt.py::TestAllocateQueueExtractsNumRuns::test_allocate_queue_extracts_num_runs
- tests/memory/test_memmgmt.py::TestAllocateQueueExtractsNumRuns::test_allocate_queue_chunks_correctly
- tests/memory/test_memmgmt.py::TestAllocateQueueExtractsNumRuns::test_allocate_queue_fallback_without_runparams

**Outcomes**:
- Files Modified:
  * src/cubie/memory/mem_manager.py (26 lines changed)
  * tests/memory/test_memmgmt.py (196 lines added)
- Functions/Methods Added/Modified:
  * allocate_queue() in mem_manager.py - Modified to extract num_runs from triggering_instance.run_params.runs
  * Updated docstring to document num_runs extraction from run_params
  * Added hasattr() check for run_params with fallback to get_chunk_axis_length()
  * Removed references to removed ArrayResponse fields (axis_length, dangling_chunk_length)
  * Updated ArrayResponse construction to only include existing fields
  * TestAllocateQueueExtractsNumRuns.test_allocate_queue_extracts_num_runs() in test_memmgmt.py
  * TestAllocateQueueExtractsNumRuns.test_allocate_queue_chunks_correctly() in test_memmgmt.py
  * TestAllocateQueueExtractsNumRuns.test_allocate_queue_fallback_without_runparams() in test_memmgmt.py
- Implementation Summary:
  * Modified allocate_queue to extract num_runs from triggering_instance.run_params.runs with graceful fallback
  * Updated allocate_queue to use num_runs variable consistently instead of axis_length
  * Removed creation of dangling_chunk_length in allocate_queue (no longer needed)
  * Updated ArrayResponse instantiation to remove axis_length and dangling_chunk_length fields
  * Created comprehensive test class with 3 test methods covering normal case, chunking case, and fallback
  * Verified helper methods still exist and are in use:
    - get_chunk_axis_length() - used as fallback when run_params not available
    - replace_with_chunked_size() - used in compute_chunked_shapes()
    - compute_per_chunk_slice() - used in allocate_queue for creating per-chunk slicing functions
  * All helper methods remain in codebase as they are still needed for chunking operations
- Issues Flagged:
  * ArrayResponse in Task Group 2 outcomes showed that axis_length and dangling_chunk_length were removed, but they were still being referenced in allocate_queue - this has been fixed
  * Helper methods are still needed and should not be removed contrary to what task description suggested

---

## Task Group 6: Update Array Managers for New Pattern
**Status**: [x]
**Dependencies**: Groups [1, 2, 3, 5]

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (entire file)
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (entire file)
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (entire file)

**Input Validation Required**:
- None (array managers receive validated data from solver)

**Tasks**:
1. **Update BaseArrayManager._on_allocation_complete to not access removed fields**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     The current implementation already doesn't access axis_length or
     dangling_chunk_length from ArrayResponse. Verify this:
     
     ```python
     def _on_allocation_complete(self, response: ArrayResponse) -> None:
         # ... existing implementation ...
         self._chunks = response.chunks
         self._chunk_axis = response.chunk_axis
         # ... rest of method ...
     ```
     
     This is already correct - no changes needed.
   - Edge cases: None
   - Integration: Callback remains compatible with simplified ArrayResponse

2. **Verify array managers don't use removed ArrayResponse fields**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Verify
   - Details:
     Search for any references to `response.axis_length` or
     `response.dangling_chunk_length` in:
     - BatchInputArrays.py
     - BatchOutputArrays.py
     - BaseArrayManager.py
     
     Confirm these fields are not accessed anywhere. Based on the current
     source code review, they are not present.
   - Edge cases: None
   - Integration: No changes needed if fields aren't used

3. **Document that num_runs comes from solver context**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     Add a docstring note to BaseArrayManager class documentation:
     
     ```python
     class BaseArrayManager(ABC):
         """Coordinate allocation and transfer for batch host and device arrays.
         
         Parameters
         ----------
         ... existing parameters ...
         
         Notes
         -----
         The number of runs in the batch is tracked by the solver (via
         RunParams) and does not need to be stored separately in the
         array manager. Chunking metadata (num_chunks, chunk_axis) is
         received via ArrayResponse after allocation.
         
         ... existing notes ...
         """
     ```
   - Edge cases: None
   - Integration: Documentation clarity for future developers

**Tests to Create**:
- Test file: tests/batchsolving/arrays/test_base_array_manager.py
- Test function: test_allocation_callback_works_with_simplified_response
- Description: Verify array managers handle ArrayResponse without removed fields

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py::TestAllocationCallbackSimplifiedResponse::test_allocation_callback_works_with_simplified_response

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/arrays/BaseArrayManager.py (5 lines added to docstring)
  * tests/batchsolving/arrays/test_basearraymanager.py (117 lines added)
- Functions/Methods Added/Modified:
  * BaseArrayManager class docstring - Added Notes section documenting num_runs tracking
  * TestAllocationCallbackSimplifiedResponse.test_allocation_callback_works_with_simplified_response() in test_basearraymanager.py
- Implementation Summary:
  * Verified _on_allocation_complete does NOT access removed fields (axis_length, dangling_chunk_length)
  * _on_allocation_complete only accesses: response.arr, response.chunks, response.chunked_shapes, response.chunked_slices
  * Verified BatchInputArrays.py and BatchOutputArrays.py do NOT reference removed fields
  * Added documentation to BaseArrayManager class explaining num_runs is tracked by solver via RunParams
  * Created comprehensive test verifying array managers work with simplified ArrayResponse
  * Test confirms removed fields (axis_length, dangling_chunk_length) are not present on ArrayResponse
  * Test verifies allocation callback completes successfully without accessing removed fields
  * Test validates chunked transfer detection still works correctly
- Issues Flagged: None

---

## Task Group 7: Integration Tests and Cleanup
**Status**: [x]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file)
- File: src/cubie/memory/mem_manager.py (entire file)
- File: tests/batchsolving/test_solver.py (if exists)

**Input Validation Required**:
- None (integration tests verify end-to-end behavior)

**Tasks**:
1. **Create end-to-end integration test for RunParams flow**
   - File: tests/batchsolving/test_runparams_integration.py
   - Action: Create
   - Details:
     ```python
     """Integration tests for RunParams refactoring."""
     
     import pytest
     import numpy as np
     from cubie import Solver
     from cubie.odesystems.symbolic import SymbolicODE
     
     
     def test_runparams_single_chunk():
         """Verify RunParams works correctly when no chunking occurs."""
         # Create simple ODE system
         system = create_test_system()
         solver = Solver(system, algorithm='explicit_euler')
         
         # Run with parameters that fit in memory (no chunking)
         inits = np.random.rand(system.n_states, 10)
         params = np.random.rand(system.n_params, 10)
         
         result = solver.solve(
             inits, params,
             duration=1.0,
             warmup=0.1,
             t0=0.0
         )
         
         # Verify run_params was created and used
         assert solver.kernel.run_params.runs == 10
         assert solver.kernel.run_params.num_chunks == 1
         assert solver.kernel.run_params.duration == 1.0
         assert solver.kernel.run_params.warmup == 0.1
         assert solver.kernel.run_params.t0 == 0.0
     
     
     def test_runparams_multiple_chunks():
         """Verify RunParams works correctly with chunking."""
         # This test may require large arrays to force chunking
         # or manual memory proportion adjustment
         system = create_test_system()
         solver = Solver(system, algorithm='explicit_euler')
         
         # Force chunking by setting a small memory proportion
         solver.kernel.memory_manager.set_manual_proportion(
             solver.kernel, 0.01  # Very small to force chunking
         )
         
         # Run with enough data to require chunking
         num_runs = 1000
         inits = np.random.rand(system.n_states, num_runs)
         params = np.random.rand(system.n_params, num_runs)
         
         result = solver.solve(
             inits, params,
             duration=1.0,
         )
         
         # Verify chunking occurred and run_params was updated
         assert solver.kernel.run_params.runs == num_runs
         assert solver.kernel.run_params.num_chunks > 1
         assert solver.kernel.run_params.chunk_length > 0
         
         # Verify dangling chunk calculation
         last_chunk_params = solver.kernel.run_params[
             solver.kernel.run_params.num_chunks - 1
         ]
         expected_last_chunk_runs = (
             num_runs 
             - (solver.kernel.run_params.num_chunks - 1) 
             * solver.kernel.run_params.chunk_length
         )
         assert last_chunk_params.runs == expected_last_chunk_runs
     
     
     def test_runparams_exact_division():
         """Verify RunParams when runs divide evenly into chunks."""
         system = create_test_system()
         solver = Solver(system, algorithm='explicit_euler')
         
         # Set up scenario where runs divide evenly
         # This may require manipulating memory settings
         # to get specific chunk_length
         
         # Example: 100 runs, 4 chunks = 25 runs per chunk exactly
         # Implementation depends on ability to control chunking
         pass  # Implement based on memory manager API
     
     
     def create_test_system():
         """Create a simple test ODE system."""
         # Implementation of simple system creation
         # This should match test fixtures used elsewhere
         pass
     ```
   - Edge cases: Test single chunk, multiple chunks, exact division, dangling chunk
   - Integration: End-to-end validation of refactoring

2. **Update existing tests that reference ChunkParams**
   - File: Search all test files
   - Action: Modify
   - Details:
     Find all tests that import or use ChunkParams:
     ```bash
     grep -r "ChunkParams" tests/
     grep -r "chunk_params" tests/
     ```
     
     For each occurrence:
     - Replace ChunkParams with RunParams
     - Update test assertions to use RunParams fields
     - Update mocking/fixtures to use RunParams
     
     Example transformation:
     ```python
     # Old:
     from cubie.batchsolving.BatchSolverKernel import ChunkParams
     chunk_params = ChunkParams(duration=1.0, warmup=0.0, t0=0.0, size=10, runs=100)
     
     # New:
     from cubie.batchsolving.BatchSolverKernel import RunParams
     run_params = RunParams(duration=1.0, warmup=0.0, t0=0.0, runs=100)
     run_params = run_params.update_from_allocation(mock_response)
     chunk_params = run_params[0]
     ```
   - Edge cases: Tests may use indirect parameterization patterns
   - Integration: Maintain test coverage across refactoring

3. **Remove or update docstring references to removed concepts**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     Search for docstring references to:
     - "FullRunParams"
     - "ChunkParams"
     - "axis_length"
     - "dangling_chunk_length"
     
     Update any docstrings that mention these concepts to reflect
     the new RunParams-based architecture.
     
     Example:
     ```python
     # Old docstring:
     """
     ... uses ChunkParams to determine per-chunk execution ...
     """
     
     # New docstring:
     """
     ... uses RunParams.__getitem__ to determine per-chunk execution ...
     """
     ```
   - Edge cases: May appear in Notes sections
   - Integration: Keep documentation current

4. **Verify no imports reference removed classes**
   - File: All files
   - Action: Verify
   - Details:
     Search the entire codebase for imports of removed classes:
     ```bash
     grep -r "from.*import.*ChunkParams" src/
     grep -r "from.*import.*FullRunParams" src/
     ```
     
     Remove any remaining imports of these classes.
   - Edge cases: May appear in __init__.py files
   - Integration: Clean module namespace

**Tests to Create**:
- Test file: tests/batchsolving/test_runparams_integration.py
- Test function: test_runparams_single_chunk
- Description: End-to-end test with no chunking
- Test function: test_runparams_multiple_chunks
- Description: End-to-end test with chunking
- Test function: test_runparams_exact_division
- Description: End-to-end test with even division into chunks

**Tests to Run**:
- tests/batchsolving/test_runparams_integration.py
- tests/batchsolving/test_batchsolverkernel.py
- tests/memory/test_memmgmt.py
- tests/batchsolving/arrays/test_base_array_manager.py

**Outcomes**:
- Files Modified:
  * tests/batchsolving/test_runparams_integration.py (305 lines added - new file)
  * src/cubie/batchsolving/BatchSolverKernel.py (19 lines changed)
- Functions/Methods Added/Modified:
  * Created test_runparams_integration.py with 6 test functions:
    - test_runparams_single_chunk() - verifies no chunking case
    - test_runparams_multiple_chunks() - verifies chunking with dangling chunk
    - test_runparams_exact_division() - verifies exact division into chunks
    - test_runparams_indexing_edge_cases() - verifies __getitem__ edge cases
    - test_runparams_immutability() - verifies frozen attrs behavior
    - integration_system() and integration_solver() fixtures
  * Fixed BatchSolverKernel.chunks property to use run_params instead of chunk_params
  * Updated FullRunParams docstring to clarify its purpose and relationship to RunParams
- Implementation Summary:
  * Created comprehensive integration tests covering all RunParams scenarios
  * Tests verify single chunk (no chunking), multiple chunks (with dangling chunk), and exact division
  * Tests verify RunParams.__getitem__ correctly calculates per-chunk runs
  * Tests verify immutability of RunParams instances
  * Fixed incorrect reference to self.chunk_params.num_chunks (changed to self.run_params.num_chunks)
  * Updated FullRunParams docstring to clarify it's for property getters/setters, distinct from RunParams
  * Verified no test files import or reference ChunkParams (ChunkParams was already removed in Task Group 4)
  * Verified no docstrings reference removed concepts (axis_length, dangling_chunk_length)
  * Verified FullRunParams is intentionally kept for backward compatibility with property setters
- Issues Flagged:
  * No ChunkParams imports found in test files (already cleaned up in previous task groups)
  * FullRunParams is intentionally retained for property getters/setters - this is correct design
  * All removed field references (axis_length, dangling_chunk_length) were already cleaned in Task Group 5

---

## Summary

### Total Task Groups: 7

### Dependency Chain:
```
Group 1 (RunParams) → Group 3 (BatchSolverKernel)
                   → Group 4 (Remove ChunkParams)
                   → Group 5 (MemoryManager)

Group 2 (ArrayResponse) → Group 5 (MemoryManager)
                        → Group 6 (Array Managers)

Group 1,2,3,5 → Group 6 (Array Managers)

All → Group 7 (Integration Tests)
```

### Tests Created: 21 test functions across 5 test files

### Tests to Run:
- All new test files for each task group
- Existing test suite to verify no regressions

### Estimated Complexity: **Medium-High**

**Reasoning:**
- Creates new core abstraction (RunParams) that replaces two existing classes
- Removes redundant fields from data structures
- Updates memory manager allocation flow
- Maintains backward compatibility during transition
- Requires careful handling of chunking logic edge cases
- Integration tests needed to verify end-to-end behavior

**Key Risks:**
- Chunking calculation errors in RunParams.__getitem__ (dangling chunk)
- Missing references to removed ChunkParams in tests
- Memory manager may access removed fields indirectly
- Time-based chunking path needs verification (chunk_axis="time")

**Mitigation:**
- Comprehensive unit tests for RunParams edge cases
- Systematic grep/search for ChunkParams references
- Integration tests covering both run and time chunking
- Phased implementation allows validation at each step

---

*This task list provides function-level implementation details for the chunk removal refactoring. Each task group can be executed by a separate taskmaster agent invocation.*
