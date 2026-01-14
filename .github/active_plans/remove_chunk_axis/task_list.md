# Implementation Task List
# Feature: Remove Chunk Axis
# Plan Reference: .github/active_plans/remove_chunk_axis/agent_plan.md

## Task Group 1: Remove chunk_axis from Data Structures
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/memory/array_requests.py (lines 130-152)
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 63-86, 88-126)
- File: .github/context/cubie_internal_structure.md (lines 1-50)

**Input Validation Required**:
- None (removing fields, not adding)

**Tasks**:

1. **Remove chunk_axis from ArrayResponse**
   - File: src/cubie/memory/array_requests.py
   - Action: Modify
   - Details:
     ```python
     # Remove chunk_axis field from ArrayResponse attrs class (line ~141-143)
     # Before:
     @attrs.define
     class ArrayResponse:
         arr: dict[str, DeviceNDArrayBase]
         chunks: int
         chunk_axis: str = attrs.field(
             default="run", validator=val.in_(["run", "variable", "time"])
         )
         # ... other fields
     
     # After:
     @attrs.define
     class ArrayResponse:
         arr: dict[str, DeviceNDArrayBase]
         chunks: int
         # chunk_axis field removed - chunking is hardcoded to "run" axis
         # ... other fields
     ```
   - Edge cases: Ensure no other code references response.chunk_axis
   - Integration: ArrayResponse used by MemoryManager and BaseArrayManager

2. **Remove chunk_axis from FullRunParams**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Remove chunk_axis field from FullRunParams (line ~85)
     # Before:
     @define(frozen=True)
     class FullRunParams:
         duration: float
         warmup: float
         t0: float
         runs: int
         chunk_axis: str
     
     # After:
     @define(frozen=True)
     class FullRunParams:
         """Full batch run parameters before chunking.
         
         Chunking always occurs along the run axis.
         
         Attributes
         ----------
         duration
             Full duration of the simulation window.
         warmup
             Full warmup time before the main simulation.
         t0
             Initial integration time.
         runs
             Total number of runs in the batch.
         """
         duration: float
         warmup: float
         t0: float
         runs: int
     ```
   - Edge cases: Remove docstring reference to chunk_axis (line ~77-78)
   - Integration: FullRunParams used by ChunkParams.from_allocation_response()

3. **Remove _chunk_axis from ChunkParams**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Remove _chunk_axis field from ChunkParams (line ~125)
     # Before:
     @define(frozen=True)
     class ChunkParams:
         duration: float
         warmup: float
         t0: float
         runs: int
         num_chunks: int = field(default=1, repr=False)
         _full_params: FullRunParams = field(default=None, repr=False)
         _axis_length: int = field(default=0, repr=False)
         _chunk_length: int = field(default=0, repr=False)
         _dangling_chunk_length: int = field(default=0, repr=False)
         _chunk_axis: str = field(default="run", repr=False)
     
     # After:
     @define(frozen=True)
     class ChunkParams:
         """Chunked execution parameters calculated for a batch run.
         
         Chunking always occurs along the run axis.
         
         Attributes
         ----------
         duration
             Duration assigned to each chunk.
         warmup
             Warmup duration applied to the current chunk.
         t0
             Start time of the chunk.
         runs
             Number of runs scheduled within a chunk.
         num_chunks
             Number of chunks the full run has been divided into.
         _full_params
             Internal copy of the full run parameters for chunk calculations.
         _axis_length
             Length of the run axis in the full run.
         _chunk_length
             Length of the chunk along the run axis.
         _dangling_chunk_length
             Length of the final chunk along the run axis.
         """
         duration: float
         warmup: float
         t0: float
         runs: int
         num_chunks: int = field(default=1, repr=False)
         _full_params: FullRunParams = field(default=None, repr=False)
         _axis_length: int = field(default=0, repr=False)
         _chunk_length: int = field(default=0, repr=False)
         _dangling_chunk_length: int = field(default=0, repr=False)
     ```
   - Edge cases: Update docstring to remove chunk_axis references (line ~112-113)
   - Integration: ChunkParams used in BatchSolverKernel.run() method

**Tests to Create**:
- None (data structure changes verified by compilation and dependent tests)

**Tests to Run**:
- tests/memory/test_array_requests.py (verify ArrayResponse instantiation)
- tests/batchsolving/test_solver.py (verify kernel initialization)

**Outcomes**: 

---

## Task Group 2: Simplify ChunkParams Logic and Remove Time-Axis Branching
**Status**: [ ]
**Dependencies**: Groups [1]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 127-204)
- File: .github/context/cubie_internal_structure.md (lines 1-50)

**Input Validation Required**:
- None (removing logic, not adding)

**Tasks**:

1. **Update ChunkParams.from_allocation_response() to remove chunk_axis parameter**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Update from_allocation_response classmethod (lines ~127-160)
     # Remove chunk_axis parameter extraction from allocation_response
     # Before:
     @classmethod
     def from_allocation_response(
         cls,
         full_params: FullRunParams,
         allocation_response: "ArrayResponse",
     ) -> "ChunkParams":
         # ... existing code ...
         # Uses allocation_response.chunk_axis
     
     # After:
     @classmethod
     def from_allocation_response(
         cls,
         full_params: FullRunParams,
         allocation_response: "ArrayResponse",
     ) -> "ChunkParams":
         """Construct ChunkParams from memory allocation response.
         
         Chunking is always performed along the run axis.
         
         Parameters
         ----------
         full_params
             Full batch run parameters.
         allocation_response
             Memory allocation response containing chunk metadata.
         
         Returns
         -------
         ChunkParams instance with chunking parameters.
         """
         return cls(
             duration=full_params.duration,
             warmup=full_params.warmup,
             t0=full_params.t0,
             runs=full_params.runs,
             num_chunks=allocation_response.chunks,
             _full_params=full_params,
             _axis_length=allocation_response.axis_length,
             _chunk_length=allocation_response.chunk_length,
             _dangling_chunk_length=allocation_response.dangling_chunk_length,
             # _chunk_axis removed - hardcoded to "run"
         )
     ```
   - Edge cases: Ensure runs parameter correctly extracted from full_params
   - Integration: Called by BatchSolverKernel after allocate_queue()

2. **Simplify ChunkParams.__getitem__() to remove time-axis logic**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Simplify __getitem__ method (lines ~162-203)
     # Remove time-axis branching entirely
     # Before:
     def __getitem__(self, index: int) -> "ChunkParams":
         length = self._chunk_length
         _duration = self._full_params.duration
         _warmup = self._full_params.warmup
         _t0 = self._full_params.t0
         _runs = self._full_params.runs
         
         if index == self.num_chunks - 1:
             length = self._dangling_chunk_length
         
         if self._chunk_axis == "run":
             _runs = length
         elif self._chunk_axis == "time":
             # time-axis logic here (lines 187-196)
         
         return evolve(self, ...)
     
     # After:
     def __getitem__(self, index: int) -> "ChunkParams":
         """Get chunk parameters for a specific chunk index.
         
         Chunking is performed along the run axis, so each chunk
         contains a subset of the total runs.
         
         Parameters
         ----------
         index
             Chunk index (0-based)
         
         Returns
         -------
         A copy of self with updated runs for the selected chunk.
         All other parameters (duration, warmup, t0) remain unchanged.
         """
         # Determine chunk length (handle dangling chunk)
         length = self._chunk_length
         if index == self.num_chunks - 1:
             length = self._dangling_chunk_length
         
         # For run-axis chunking, only runs changes per chunk
         return evolve(
             self,
             runs=length,
         )
     ```
   - Edge cases: Dangling chunk (last chunk with fewer runs)
   - Integration: Called during chunk iteration in BatchSolverKernel.run()

**Tests to Create**:
- None (logic changes verified by existing chunking tests)

**Tests to Run**:
- tests/batchsolving/arrays/test_chunking.py (verify run-axis chunking still works)
- tests/batchsolving/test_solver.py (verify multi-chunk execution)

**Outcomes**: 

---

## Task Group 3: Remove chunk_axis from MemoryManager
**Status**: [ ]
**Dependencies**: Groups [1, 2]

**Required Context**:
- File: src/cubie/memory/mem_manager.py (lines 1157-1300)
- File: src/cubie/memory/mem_manager.py (lines 700-900) - helper functions
- File: .github/context/cubie_internal_structure.md (lines 285-305)

**Input Validation Required**:
- None (removing parameter, not adding)

**Tasks**:

1. **Remove chunk_axis parameter from allocate_queue()**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     ```python
     # Update allocate_queue method (line ~1157-1180)
     # Before:
     def allocate_queue(
         self,
         triggering_instance: object,
         chunk_axis: str = "run",
     ) -> None:
         """Process all queued requests..."""
     
     # After:
     def allocate_queue(
         self,
         triggering_instance: object,
     ) -> None:
         """Process all queued requests for a stream group with coordinated
         chunking.
         
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
         # Add hardcoded chunk_axis at start of method body
         chunk_axis = "run"
         # ... rest of existing logic unchanged ...
     ```
   - Edge cases: All internal calls to chunk_axis use local variable
   - Integration: Called by BatchSolverKernel without chunk_axis argument

2. **Remove chunk_axis from get_chunk_parameters()**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     ```python
     # Find get_chunk_parameters method (search for def get_chunk_parameters)
     # Remove chunk_axis parameter, hardcode to "run"
     # Before:
     def get_chunk_parameters(
         self,
         requests: List[ArrayRequest],
         chunk_axis: str,
     ) -> dict:
         """Calculate chunking parameters..."""
     
     # After:
     def get_chunk_parameters(
         self,
         requests: List[ArrayRequest],
     ) -> dict:
         """Calculate chunking parameters for run-axis chunking.
         
         Parameters
         ----------
         requests
             List of array allocation requests.
         
         Returns
         -------
         dict
             Chunking parameters including chunks, axis_length, etc.
         """
         chunk_axis = "run"  # Hardcoded
         # ... rest of method unchanged ...
     ```
   - Edge cases: Update all callers to not pass chunk_axis
   - Integration: Called internally by allocate_queue()

3. **Remove chunk_axis from helper functions**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     ```python
     # Update these functions (search for each):
     # - get_portioned_request_size(request, chunk_axis, chunk_length)
     # - get_chunk_axis_length(requests, chunk_axis)
     # - is_request_chunkable(request, chunk_axis)
     # - compute_per_chunk_slice(request, chunk_axis, ...)
     
     # Pattern for each:
     # Before:
     def function_name(..., chunk_axis: str, ...):
         """..."""
         if chunk_axis not in request.stride_order:
             # ...
     
     # After:
     def function_name(...):  # Remove chunk_axis parameter
         """...
         
         Chunking is performed along the run axis only.
         """
         chunk_axis = "run"  # Hardcode at start of function
         # ... rest unchanged ...
         if "run" not in request.stride_order:
             # Or: if chunk_axis not in request.stride_order:
             # ...
     ```
   - Edge cases: Ensure stride_order validation uses "run" literal or local var
   - Integration: These are internal helper functions

4. **Remove chunk_axis from ArrayResponse construction**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     ```python
     # Find where ArrayResponse is constructed (in allocate_queue)
     # Before:
     response = ArrayResponse(
         arr=device_arrays,
         chunks=chunks,
         chunk_axis=chunk_axis,
         # ... other fields ...
     )
     
     # After:
     response = ArrayResponse(
         arr=device_arrays,
         chunks=chunks,
         # chunk_axis field removed from ArrayResponse
         # ... other fields ...
     )
     ```
   - Edge cases: Verify no other ArrayResponse construction sites
   - Integration: ArrayResponse consumed by BaseArrayManager

**Tests to Create**:
- None (changes verified by existing memory management tests)

**Tests to Run**:
- tests/memory/test_memmgmt.py (verify allocate_queue works without chunk_axis)
- tests/batchsolving/arrays/test_chunking.py (verify chunking still works)

**Outcomes**: 

---

## Task Group 4: Remove _chunk_axis from BaseArrayManager
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 220-280, 355-374)
- File: .github/context/cubie_internal_structure.md (lines 336-398)

**Input Validation Required**:
- None (removing attribute)

**Tasks**:

1. **Remove _chunk_axis attribute and validator**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     # Remove _chunk_axis field from BaseArrayManager (line ~261-263)
     # Before:
     @define
     class BaseArrayManager:
         # ... other fields ...
         _chunks: int = field(default=0, validator=attrsval_instance_of(int))
         _chunk_axis: str = field(
             default="run", validator=attrsval_in(["run", "variable", "time"])
         )
         _stream_group: str = field(
             default="default", validator=attrsval_instance_of(str)
         )
         # ... rest ...
     
     # After:
     @define
     class BaseArrayManager:
         # ... other fields ...
         _chunks: int = field(default=0, validator=attrsval_instance_of(int))
         # _chunk_axis removed - chunking is hardcoded to "run" axis
         _stream_group: str = field(
             default="default", validator=attrsval_instance_of(str)
         )
         # ... rest ...
     ```
   - Edge cases: None
   - Integration: BaseArrayManager base class for InputArrays and OutputArrays

2. **Remove _chunk_axis from class docstring**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     # Update class docstring (lines ~220-228)
     # Before:
     """Base array manager for batch operations.
     
     Attributes
     ----------
     # ... other attributes ...
     _chunks
         Number of chunks for memory management.
     _chunk_axis
         Axis along which to perform chunking. Must be one of "run",
         "variable", or "time".
     _stream_group
         Stream group identifier for CUDA operations.
     # ... rest ...
     
     # After:
     """Base array manager for batch operations.
     
     Attributes
     ----------
     # ... other attributes ...
     _chunks
         Number of chunks for memory management. Chunking is always
         performed along the run axis.
     _stream_group
         Stream group identifier for CUDA operations.
     # ... rest ...
     """
     ```
   - Edge cases: None
   - Integration: Documentation update only

3. **Remove _chunk_axis assignment from _on_allocation_complete()**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     # Update _on_allocation_complete method (line ~368)
     # Before:
     def _on_allocation_complete(self, response: ArrayResponse) -> None:
         # ... existing code ...
         self._chunks = response.chunks
         self._chunk_axis = response.chunk_axis  # REMOVE THIS LINE
         if self.is_chunked:
             self._convert_host_to_numpy()
         # ... rest ...
     
     # After:
     def _on_allocation_complete(self, response: ArrayResponse) -> None:
         # ... existing code ...
         self._chunks = response.chunks
         # _chunk_axis assignment removed - ArrayResponse no longer has field
         if self.is_chunked:
             self._convert_host_to_numpy()
         # ... rest ...
     ```
   - Edge cases: Verify no other code in method references chunk_axis
   - Integration: Hook called by MemoryManager after allocation

4. **Replace self._chunk_axis references with "run" literal**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     # Search entire file for self._chunk_axis references
     # Replace each with literal "run" string
     # Example pattern:
     # Before:
     if self._chunk_axis == "run":
         # do something
     
     # After:
     # Chunking is always on run axis, so condition is always true
     # Either remove conditional or use literal:
     if "run" == "run":  # Simplifies to: if True:
         # do something
     # Or just remove the if statement if always true
     ```
   - Edge cases: Some conditions may become always-true and can be simplified
   - Integration: Check InputArrays and OutputArrays subclasses for usage

**Tests to Create**:
- None (verified by existing tests)

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py
- tests/batchsolving/arrays/test_batchinputarrays.py
- tests/batchsolving/arrays/test_batchoutputarrays.py

**Outcomes**: 

---

## Task Group 5: Remove chunk_axis from BatchSolverKernel
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 700-900, 1145-1180)
- File: .github/context/cubie_internal_structure.md (lines 323-398)

**Input Validation Required**:
- None (removing parameter and property)

**Tasks**:

1. **Remove chunk_axis parameter from run() method**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Update run() method signature (search for "def run(")
     # Before:
     def run(
         self,
         inits: NDArray,
         params: NDArray,
         driver_coefficients: Optional[NDArray],
         duration: float,
         warmup: float,
         t0: float,
         blocksize: int,
         stream: Any = None,
         chunk_axis: str = "run",
     ) -> None:
         """Execute batch integration..."""
     
     # After:
     def run(
         self,
         inits: NDArray,
         params: NDArray,
         driver_coefficients: Optional[NDArray],
         duration: float,
         warmup: float,
         t0: float,
         blocksize: int,
         stream: Any = None,
     ) -> None:
         """Execute batch integration.
         
         Chunking is performed along the run axis when memory constraints
         require splitting the batch.
         
         Parameters
         ----------
         inits
             Initial state values array.
         params
             Parameter values array.
         driver_coefficients
             Driver interpolation coefficients.
         duration
             Integration time window.
         warmup
             Warm-up period before recording outputs.
         t0
             Initial integration time.
         blocksize
             CUDA block size for kernel launch.
         stream
             CUDA stream for execution.
         
         Returns
         -------
         None
         """
         # Remove chunk_axis from docstring Parameters section
     ```
   - Edge cases: Update docstring to remove chunk_axis documentation
   - Integration: Called by Solver.solve() without chunk_axis

2. **Remove chunk_axis from FullRunParams construction**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Find FullRunParams construction in run() method
     # Before:
     full_params = FullRunParams(
         duration=duration,
         warmup=warmup,
         t0=t0,
         runs=n_runs,
         chunk_axis=chunk_axis,
     )
     
     # After:
     full_params = FullRunParams(
         duration=duration,
         warmup=warmup,
         t0=t0,
         runs=n_runs,
         # chunk_axis field removed from FullRunParams
     )
     ```
   - Edge cases: None
   - Integration: FullRunParams passed to ChunkParams.from_allocation_response()

3. **Remove chunk_axis from allocate_queue() call**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Find allocate_queue call in run() method
     # Before:
     self.memory_manager.allocate_queue(
         triggering_instance=self,
         chunk_axis=chunk_axis,
     )
     
     # After:
     self.memory_manager.allocate_queue(
         triggering_instance=self,
         # chunk_axis parameter removed - defaults to "run"
     )
     ```
   - Edge cases: None
   - Integration: Memory manager handles chunking internally

4. **Remove chunk_axis property**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Delete
   - Details:
     ```python
     # Delete entire chunk_axis property (lines ~1152-1177)
     # Before:
     @property
     def chunk_axis(self) -> str:
         """Current chunking axis.
         
         Returns the chunk_axis value from the array managers, validating
         that input and output arrays have consistent values.
         
         Returns
         -------
         str
             The axis along which arrays are chunked.
         
         Raises
         ------
         ValueError
             If input_arrays and output_arrays have different chunk_axis
             values.
         """
         input_axis = self.input_arrays._chunk_axis
         output_axis = self.output_arrays._chunk_axis
         if input_axis != output_axis:
             raise ValueError(
                 f"Inconsistent chunk_axis: input_arrays has '{input_axis}', "
                 f"output_arrays has '{output_axis}'"
             )
         return input_axis
     
     # After:
     # Property completely removed - no replacement needed
     ```
   - Edge cases: Ensure no code references self.chunk_axis or kernel.chunk_axis
   - Integration: Property accessed by Solver.chunk_axis property (remove that too)

**Tests to Create**:
- None (verified by integration tests)

**Tests to Run**:
- tests/batchsolving/test_solver.py
- tests/batchsolving/arrays/test_chunking.py

**Outcomes**: 

---

## Task Group 6: Remove chunk_axis from Solver Public API
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5]

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 48-150, 341-460, 885-900)
- File: .github/context/cubie_internal_structure.md (lines 1-50)

**Input Validation Required**:
- None (removing parameter)

**Tasks**:

1. **Remove chunk_axis parameter from Solver.solve()**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # Update solve() method signature (line ~341-356)
     # Before:
     def solve(
         self,
         initial_values: Union[ndarray, Dict[str, Union[float, ndarray]]],
         parameters: Union[ndarray, Dict[str, Union[float, ndarray]]],
         drivers: Optional[Dict[str, Any]] = None,
         duration: float = 1.0,
         settling_time: float = 0.0,
         t0: float = 0.0,
         blocksize: int = 256,
         stream: Any = None,
         chunk_axis: str = "run",
         grid_type: str = "verbatim",
         results_type: str = "full",
         nan_error_trajectories: bool = True,
         **kwargs: Any,
     ) -> SolveResult:
     
     # After:
     def solve(
         self,
         initial_values: Union[ndarray, Dict[str, Union[float, ndarray]]],
         parameters: Union[ndarray, Dict[str, Union[float, ndarray]]],
         drivers: Optional[Dict[str, Any]] = None,
         duration: float = 1.0,
         settling_time: float = 0.0,
         t0: float = 0.0,
         blocksize: int = 256,
         stream: Any = None,
         grid_type: str = "verbatim",
         results_type: str = "full",
         nan_error_trajectories: bool = True,
         **kwargs: Any,
     ) -> SolveResult:
     ```
   - Edge cases: None
   - Integration: Public API method called by users

2. **Remove chunk_axis from solve() docstring**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # Update docstring Parameters section (lines ~359-399)
     # Before:
     """Solve a batch initial value problem.
     
     Parameters
     ----------
     # ... other parameters ...
     stream
         Stream on which to execute the kernel. ``None`` uses the solver's
         default stream.
     chunk_axis
         Dimension along which to chunk when memory is limited. Default is
         ``"run"``.
     grid_type
         Strategy for constructing the integration grid from inputs.
         Only used when dict inputs trigger grid construction.
     # ... rest ...
     
     # After:
     """Solve a batch initial value problem.
     
     Parameters
     ----------
     # ... other parameters ...
     stream
         Stream on which to execute the kernel. ``None`` uses the solver's
         default stream.
     grid_type
         Strategy for constructing the integration grid from inputs.
         Only used when dict inputs trigger grid construction.
     # ... rest ...
     
     Notes
     -----
     When GPU memory is insufficient for the full batch, arrays are
     automatically chunked along the run axis.
     ```
   - Edge cases: Consider adding Note about automatic run-axis chunking
   - Integration: User-facing documentation

3. **Remove chunk_axis from kernel.run() call**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # Update kernel.run() call (line ~441-451)
     # Before:
     self.kernel.run(
         inits=inits,
         params=params,
         driver_coefficients=self.driver_interpolator.coefficients,
         duration=duration,
         warmup=settling_time,
         t0=t0,
         blocksize=blocksize,
         stream=stream,
         chunk_axis=chunk_axis,
     )
     
     # After:
     self.kernel.run(
         inits=inits,
         params=params,
         driver_coefficients=self.driver_interpolator.coefficients,
         duration=duration,
         warmup=settling_time,
         t0=t0,
         blocksize=blocksize,
         stream=stream,
         # chunk_axis parameter removed
     )
     ```
   - Edge cases: None
   - Integration: Calls BatchSolverKernel.run()

4. **Remove chunk_axis property from Solver**
   - File: src/cubie/batchsolving/solver.py
   - Action: Delete
   - Details:
     ```python
     # Delete entire chunk_axis property (lines ~891-894)
     # Before:
     @property
     def chunk_axis(self) -> str:
         """Return the axis used for chunking large runs."""
         return self.kernel.chunk_axis
     
     # After:
     # Property completely removed
     ```
   - Edge cases: Check if any user code references solver.chunk_axis
   - Integration: Public property - breaking change

5. **Remove chunk_axis parameter from solve_ivp()**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # Search for solve_ivp function (line ~48)
     # Check if chunk_axis is in parameters or **kwargs
     # If explicit parameter: remove it
     # If in **kwargs: document that it's no longer accepted
     # Update docstring if chunk_axis is documented
     
     # Typical pattern:
     # Before:
     def solve_ivp(
         system: BaseODE,
         y0: Union[ndarray, Dict[str, ndarray]],
         # ... other params ...
         chunk_axis: str = "run",  # If present, remove
         **kwargs: Any,
     ) -> SolveResult:
     
     # After:
     def solve_ivp(
         system: BaseODE,
         y0: Union[ndarray, Dict[str, ndarray]],
         # ... other params ...
         # chunk_axis parameter removed
         **kwargs: Any,
     ) -> SolveResult:
     ```
   - Edge cases: solve_ivp forwards kwargs to Solver.solve()
   - Integration: Convenience wrapper function

**Tests to Create**:
- None (verified by integration tests)

**Tests to Run**:
- tests/batchsolving/test_solver.py (verify Solver.solve works)
- tests/batchsolving/arrays/test_chunking.py (verify chunking behavior)

**Outcomes**: 

---

## Task Group 7: Remove chunk_axis from Test Fixtures
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6]

**Required Context**:
- File: tests/batchsolving/arrays/conftest.py (lines 55-145)
- File: tests/conftest.py (check for chunk_axis fixtures)
- File: .github/copilot-instructions.md (lines 120-140)

**Input Validation Required**:
- None (removing test fixtures)

**Tasks**:

1. **Remove chunk_axis fixture from conftest.py**
   - File: tests/batchsolving/arrays/conftest.py
   - Action: Delete
   - Details:
     ```python
     # Delete chunk_axis fixture (lines ~59-63)
     # Before:
     @pytest.fixture(scope="session")
     def chunk_axis(request):
         if hasattr(request, "param"):
             return request.param
         return "run"
     
     # After:
     # Fixture completely removed
     ```
   - Edge cases: Check if fixture used in other conftest files
   - Integration: Used by indirect parametrization in tests

2. **Remove chunk_axis from chunked_solved_solver fixture**
   - File: tests/batchsolving/arrays/conftest.py
   - Action: Modify
   - Details:
     ```python
     # Update fixture signature and solve call (lines ~66-105)
     # Before:
     @pytest.fixture(scope="session")
     def chunked_solved_solver(
         system, precision, low_mem_solver, driver_settings, chunk_axis
     ):
         solver = low_mem_solver
         # ... setup code ...
         result = solver.solve(
             inits,
             params,
             drivers=driver_settings,
             duration=0.05,
             summarise_every=None,
             save_every=0.01,
             dt=0.01,
             chunk_axis=chunk_axis,
         )
         return solver, result
     
     # After:
     @pytest.fixture(scope="session")
     def chunked_solved_solver(
         system, precision, low_mem_solver, driver_settings
     ):
         """Fixture providing a solver with chunked execution.
         
         Chunking occurs along the run axis due to memory constraints.
         """
         solver = low_mem_solver
         # ... setup code unchanged ...
         result = solver.solve(
             inits,
             params,
             drivers=driver_settings,
             duration=0.05,
             summarise_every=None,
             save_every=0.01,
             dt=0.01,
             # chunk_axis parameter removed
         )
         return solver, result
     ```
   - Edge cases: Update comments about chunking behavior (lines ~79-94)
   - Integration: Used by chunking tests

3. **Remove chunk_axis from unchunked_solved_solver fixture**
   - File: tests/batchsolving/arrays/conftest.py
   - Action: Modify
   - Details:
     ```python
     # Update fixture (lines ~108-135)
     # Before:
     @pytest.fixture(scope="session")
     def unchunked_solved_solver(
         system,
         precision,
         driver_settings,
         unchunking_solver,
         chunk_axis,
     ):
         solver = unchunking_solver
         # ... setup ...
         result = solver.solve(
             inits,
             params,
             drivers=driver_settings,
             duration=0.05,
             summarise_every=None,
             save_every=0.01,
             dt=0.01,
             chunk_axis=chunk_axis,
         )
         return solver, result
     
     # After:
     @pytest.fixture(scope="session")
     def unchunked_solved_solver(
         system,
         precision,
         driver_settings,
         unchunking_solver,
     ):
         """Fixture providing a solver with unchunked execution."""
         solver = unchunking_solver
         # ... setup unchanged ...
         result = solver.solve(
             inits,
             params,
             drivers=driver_settings,
             duration=0.05,
             summarise_every=None,
             save_every=0.01,
             dt=0.01,
             # chunk_axis parameter removed
         )
         return solver, result
     ```
   - Edge cases: None
   - Integration: Used by chunking tests

**Tests to Create**:
- None (fixture updates only)

**Tests to Run**:
- tests/batchsolving/arrays/test_chunking.py (verify fixtures still work)

**Outcomes**: 

---

## Task Group 8: Remove chunk_axis Parametrization from Tests
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6, 7]

**Required Context**:
- File: tests/batchsolving/arrays/test_chunking.py (entire file)
- File: tests/batchsolving/arrays/test_basearraymanager.py (entire file)
- File: tests/batchsolving/arrays/test_batchinputarrays.py (entire file)
- File: tests/batchsolving/arrays/test_batchoutputarrays.py (entire file)
- File: .github/copilot-instructions.md (lines 120-140)

**Input Validation Required**:
- None (removing test parametrization)

**Tasks**:

1. **Remove TestChunkAxisProperty class from test_chunking.py**
   - File: tests/batchsolving/arrays/test_chunking.py
   - Action: Delete
   - Details:
     ```python
     # Delete entire class (lines ~26-50)
     # Before:
     class TestChunkAxisProperty:
         """Tests for chunk_axis property getter behavior."""
         
         def test_chunk_axis_property_returns_consistent_value(
             self, solverkernel_mutable
         ):
             """Verify property returns value when arrays are consistent."""
             # ... test code ...
         
         def test_chunk_axis_property_raises_on_inconsistency(
             self, solverkernel_mutable
         ):
             """Verify property raises ValueError for mismatched arrays."""
             # ... test code ...
     
     # After:
     # Class completely removed - property no longer exists
     ```
   - Edge cases: None
   - Integration: Tests removed property

2. **Remove chunk_axis parametrization from test_chunking.py tests**
   - File: tests/batchsolving/arrays/test_chunking.py
   - Action: Modify
   - Details:
     ```python
     # Find all @pytest.mark.parametrize("chunk_axis", ...) decorators
     # Remove decorator and chunk_axis parameter from function
     
     # Before:
     @pytest.mark.parametrize("chunk_axis", ["run", "time"], indirect=True)
     def test_run_sets_chunk_axis_on_arrays(
         chunked_solved_solver, system, driver_settings, chunk_axis
     ):
         """Verify solve() sets chunk_axis before array operations."""
         solver, result = chunked_solved_solver
         assert solver.kernel.input_arrays._chunk_axis == chunk_axis
         assert solver.kernel.output_arrays._chunk_axis == chunk_axis
     
     # After:
     def test_run_sets_chunk_axis_on_arrays(
         chunked_solved_solver, system, driver_settings
     ):
         """Verify solve() executes with run-axis chunking."""
         solver, result = chunked_solved_solver
         # Assertions removed - _chunk_axis attribute no longer exists
         # Could add assertion that chunking occurred:
         assert solver.kernel.chunks > 1  # Verify chunking happened
     ```
   - Edge cases: Some tests may become redundant after removing parametrization
   - Integration: Tests verify chunking behavior

3. **Simplify parametrized tests with both chunk_axis and other params**
   - File: tests/batchsolving/arrays/test_chunking.py
   - Action: Modify
   - Details:
     ```python
     # Find tests with multiple parametrize decorators including chunk_axis
     # Example (lines ~78-100):
     # Before:
     @pytest.mark.parametrize(
         "chunk_axis, forced_free_mem",
         [
             ["run", 860],
             ["run", 1024],
             ["time", 630],
             ["time", 890],
         ],
         indirect=True,
     )
     @pytest.mark.parametrize("forced_free_mem", [...], indirect=True)
     def test_chunking_with_memory_constraint(
         chunk_axis, forced_free_mem, ...
     ):
         # ... test code ...
     
     # After:
     @pytest.mark.parametrize(
         "forced_free_mem",
         [860, 1024, 1240, 1460, 2048],  # Run-axis values only
         indirect=True,
     )
     def test_chunking_with_memory_constraint(
         forced_free_mem, ...
     ):
         """Test run-axis chunking with various memory constraints."""
         # ... test code updated to remove chunk_axis references ...
     ```
   - Edge cases: Remove time-axis test values, keep only run-axis values
   - Integration: Tests still verify chunking behavior

4. **Remove chunk_axis from test_basearraymanager.py**
   - File: tests/batchsolving/arrays/test_basearraymanager.py
   - Action: Modify
   - Details:
     ```python
     # Search for all chunk_axis references
     # Remove from:
     # - allocate_queue() calls: remove chunk_axis="run" argument
     # - manager._chunk_axis assignments: remove lines
     # - assertions on _chunk_axis: remove assertions
     
     # Pattern:
     # Before:
     manager._memory_manager.allocate_queue(
         triggering_instance=manager,
         chunk_axis="run",
     )
     assert manager._chunk_axis == "run"
     
     # After:
     manager._memory_manager.allocate_queue(
         triggering_instance=manager,
     )
     # _chunk_axis assertion removed
     ```
   - Edge cases: Some tests may need new assertions to replace _chunk_axis checks
   - Integration: Tests verify BaseArrayManager behavior

5. **Remove chunk_axis from test_batchinputarrays.py**
   - File: tests/batchsolving/arrays/test_batchinputarrays.py
   - Action: Modify
   - Details:
     ```python
     # Search for chunk_axis references
     # Remove from allocate_queue calls and assertions
     # Pattern same as test_basearraymanager.py
     ```
   - Edge cases: None
   - Integration: Tests verify InputArrays behavior

6. **Remove chunk_axis from test_batchoutputarrays.py**
   - File: tests/batchsolving/arrays/test_batchoutputarrays.py
   - Action: Modify
   - Details:
     ```python
     # Search for chunk_axis references
     # Many allocate_queue calls will have chunk_axis="run"
     # Remove all such arguments
     # Remove _chunk_axis assignments and assertions
     ```
   - Edge cases: File likely has many occurrences
   - Integration: Tests verify OutputArrays behavior

**Tests to Create**:
- None (test modifications only)

**Tests to Run**:
- tests/batchsolving/arrays/test_chunking.py
- tests/batchsolving/arrays/test_basearraymanager.py
- tests/batchsolving/arrays/test_batchinputarrays.py
- tests/batchsolving/arrays/test_batchoutputarrays.py

**Outcomes**: 

---

## Task Group 9: Remove chunk_axis from Remaining Tests
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6, 7, 8]

**Required Context**:
- File: tests/batchsolving/test_solver.py (entire file)
- File: tests/batchsolving/test_config_plumbing.py (entire file)
- File: tests/memory/test_memmgmt.py (entire file)
- File: tests/memory/test_array_requests.py (entire file)

**Input Validation Required**:
- None (removing test code)

**Tasks**:

1. **Remove chunk_axis from test_solver.py**
   - File: tests/batchsolving/test_solver.py
   - Action: Modify
   - Details:
     ```python
     # Search for chunk_axis references
     # Remove from:
     # - solve() calls: remove chunk_axis="run" or chunk_axis=... arguments
     # - solver.chunk_axis property access: remove assertions
     # - Any tests specifically testing chunk_axis property
     
     # Pattern:
     # Before:
     result = solver.solve(
         initial_values,
         parameters,
         duration=1.0,
         chunk_axis="run",
     )
     assert solver.chunk_axis == "run"
     
     # After:
     result = solver.solve(
         initial_values,
         parameters,
         duration=1.0,
     )
     # Property access removed - property no longer exists
     ```
   - Edge cases: Remove any tests dedicated to chunk_axis property
   - Integration: Tests verify Solver behavior

2. **Remove chunk_axis from test_config_plumbing.py**
   - File: tests/batchsolving/test_config_plumbing.py
   - Action: Modify
   - Details:
     ```python
     # Search for chunk_axis references
     # If file tests configuration parameter plumbing:
     # - Remove chunk_axis from config dictionaries
     # - Remove assertions that chunk_axis propagates
     # - Update test docstrings
     
     # If no chunk_axis references, no changes needed
     ```
   - Edge cases: File may not reference chunk_axis
   - Integration: Tests verify configuration system

3. **Remove chunk_axis from test_memmgmt.py**
   - File: tests/memory/test_memmgmt.py
   - Action: Modify
   - Details:
     ```python
     # Search for allocate_queue calls with chunk_axis parameter
     # Remove chunk_axis="run" from all calls
     
     # Before:
     memmgr.allocate_queue(
         triggering_instance=instance,
         chunk_axis="run",
     )
     
     # After:
     memmgr.allocate_queue(
         triggering_instance=instance,
     )
     
     # Update test names and docstrings referencing chunk_axis
     # Example: "test_compute_per_chunk_slice_handles_arrays_without_chunk_axis"
     # May need to be renamed or updated to reflect run-axis-only behavior
     ```
   - Edge cases: Tests about chunk_axis in stride_order may need updates
   - Integration: Tests verify MemoryManager behavior

4. **Remove chunk_axis from test_array_requests.py**
   - File: tests/memory/test_array_requests.py
   - Action: Modify
   - Details:
     ```python
     # Search for ArrayResponse instantiations
     # Remove chunk_axis from constructor calls
     
     # Before:
     response = ArrayResponse(
         arr=device_arrays,
         chunks=2,
         chunk_axis="run",
         axis_length=100,
         # ... other fields ...
     )
     assert response.chunk_axis == "run"
     
     # After:
     response = ArrayResponse(
         arr=device_arrays,
         chunks=2,
         # chunk_axis field removed
         axis_length=100,
         # ... other fields ...
     )
     # chunk_axis assertion removed
     ```
   - Edge cases: Tests may verify ArrayResponse structure
   - Integration: Tests verify ArrayResponse dataclass

**Tests to Create**:
- None (test modifications only)

**Tests to Run**:
- tests/batchsolving/test_solver.py
- tests/batchsolving/test_config_plumbing.py
- tests/memory/test_memmgmt.py
- tests/memory/test_array_requests.py

**Outcomes**: 

---

## Task Group 10: Final Verification and Documentation
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6, 7, 8, 9]

**Required Context**:
- File: src/cubie/batchsolving/solver.py (entire file)
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file)
- File: src/cubie/memory/mem_manager.py (entire file)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (entire file)

**Input Validation Required**:
- None (verification only)

**Tasks**:

1. **Verify no chunk_axis references remain in source code**
   - File: N/A (command-line verification)
   - Action: Execute
   - Details:
     ```bash
     # Run grep to find any remaining chunk_axis references
     # Command to run:
     grep -r "chunk_axis" src/cubie/
     
     # Expected output: Empty (or only in comments explaining removal)
     # If matches found:
     # - Review each match
     # - Remove if it's code
     # - Update if it's a comment to describe run-axis-only behavior
     ```
   - Edge cases: Comments explaining the removal are acceptable
   - Integration: Final verification step

2. **Update module docstrings to reflect run-axis-only chunking**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # Update module docstring if it mentions chunk_axis
     # Add note about automatic run-axis chunking
     # Example:
     """High level batch-solver interface.
     
     This module exposes the user-facing :class:`Solver` class and a convenience
     wrapper :func:`solve_ivp` for solving batches of initial value problems on the
     GPU.
     
     Notes
     -----
     When GPU memory is insufficient for the full batch, arrays are automatically
     chunked along the run axis. Chunking is transparent to the user.
     """
     ```
   - Edge cases: Only update if module docstring exists
   - Integration: User-facing documentation

3. **Update function docstrings in MemoryManager**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     ```python
     # Review docstrings for functions modified in previous groups
     # Ensure they describe run-axis-only behavior
     # Remove any lingering references to selectable chunk_axis
     # Add "Chunking is performed along the run axis" where appropriate
     ```
   - Edge cases: Focus on public/documented methods
   - Integration: Developer documentation

4. **Update function docstrings in BaseArrayManager**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Modify
   - Details:
     ```python
     # Review class and method docstrings
     # Ensure chunking described as run-axis-only
     # Remove references to configurable chunk_axis
     ```
   - Edge cases: None
   - Integration: Developer documentation

5. **Run full test suite**
   - File: N/A (command-line verification)
   - Action: Execute
   - Details:
     ```bash
     # Run CUDASIM test suite (no GPU required)
     pytest -m "not nocudasim and not cupy"
     
     # If available, run full test suite with CUDA
     pytest
     
     # Verify all tests pass
     # Pay special attention to:
     # - tests/batchsolving/arrays/test_chunking.py
     # - tests/memory/test_memmgmt.py
     # - tests/batchsolving/test_solver.py
     ```
   - Edge cases: Some tests may need GPU
   - Integration: Final validation

**Tests to Create**:
- None (verification only)

**Tests to Run**:
- pytest -m "not nocudasim and not cupy" (full CUDASIM suite)
- tests/batchsolving/arrays/test_chunking.py
- tests/memory/test_memmgmt.py
- tests/batchsolving/test_solver.py

**Outcomes**: 

---

## Summary

**Total Task Groups**: 10

**Dependency Chain**:
1. Group 1: Data Structures (independent)
2. Group 2: ChunkParams Logic (depends on Group 1)
3. Group 3: MemoryManager (depends on Groups 1, 2)
4. Group 4: BaseArrayManager (depends on Groups 1, 2, 3)
5. Group 5: BatchSolverKernel (depends on Groups 1, 2, 3, 4)
6. Group 6: Solver Public API (depends on Groups 1-5)
7. Group 7: Test Fixtures (depends on Groups 1-6)
8. Group 8: Test Parametrization (depends on Groups 1-7)
9. Group 9: Remaining Tests (depends on Groups 1-8)
10. Group 10: Final Verification (depends on Groups 1-9)

**Tests Created**: 0 (removal task - existing tests verify behavior)

**Tests to Run**: 
- tests/memory/test_array_requests.py
- tests/memory/test_memmgmt.py
- tests/batchsolving/test_solver.py
- tests/batchsolving/arrays/test_chunking.py
- tests/batchsolving/arrays/test_basearraymanager.py
- tests/batchsolving/arrays/test_batchinputarrays.py
- tests/batchsolving/arrays/test_batchoutputarrays.py
- tests/batchsolving/test_config_plumbing.py
- Full CUDASIM suite: `pytest -m "not nocudasim and not cupy"`

**Estimated Complexity**: Medium-High
- **Breaking change**: Public API modified (Solver.solve, solve_ivp, Solver.chunk_axis property)
- **Files modified**: 5 source files, 9+ test files
- **Lines changed**: ~150-200 lines removed
- **Risk**: Low - chunking logic preserved, only configuration removed
- **Validation**: Existing tests ensure run-axis chunking still works correctly
