# Implementation Task List
# Feature: Run-Contiguous Array Memory Layout
# Plan Reference: .github/active_plans/run_contiguous_arrays/agent_plan.md

## Task Group 1: Memory Layer Defaults - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/memory/array_requests.py (lines 81-95)
- File: src/cubie/memory/mem_manager.py (lines 300-302)

**Input Validation Required**:
- None - these are default value changes only

**Tasks**:
1. **Update ArrayRequest 3D default stride_order**
   - File: src/cubie/memory/array_requests.py
   - Action: Modify
   - Details:
     ```python
     # Line 92: Change the default for 3D arrays
     # Current:
     self.stride_order = ("time", "run", "variable")
     # Target:
     self.stride_order = ("time", "variable", "run")
     ```
   - Edge cases: None - 2D default is already `("variable", "run")`
   - Integration: This is the fallback when stride_order is not explicitly provided

2. **Verify MemoryManager default stride_order**
   - File: src/cubie/memory/mem_manager.py
   - Action: Verify (no change needed)
   - Details:
     ```python
     # Lines 300-302: Already correct
     _stride_order: tuple[str, str, str] = attrs.field(
         default=("time", "variable", "run"), validator=val.instance_of(tuple)
     )
     ```
   - Edge cases: None
   - Integration: MemoryManager already defaults to target layout

**Outcomes**: 
- [x] ArrayRequest defaults to run-contiguous 3D layout
- [x] MemoryManager default confirmed correct
- Files Modified:
  * src/cubie/memory/array_requests.py (2 lines changed - default value and docstring)

---

## Task Group 2: Input Array Container Stride Orders - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (lines 26-47)

**Input Validation Required**:
- None - these are default value changes only

**Tasks**:
1. **Update initial_values stride_order**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Details:
     ```python
     # Lines 26-31: Change stride_order
     # Current:
     initial_values: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("run", "variable"),
             shape=(1, 1),
         )
     )
     # Target:
     initial_values: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("variable", "run"),
             shape=(1, 1),
         )
     )
     ```
   - Edge cases: Shape stays (1, 1) - semantic meaning changes with stride_order
   - Integration: Used by InputArrays manager for host/device transfers

2. **Update parameters stride_order**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Details:
     ```python
     # Lines 33-38: Change stride_order
     # Current:
     parameters: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("run", "variable"),
             shape=(1, 1),
         )
     )
     # Target:
     parameters: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("variable", "run"),
             shape=(1, 1),
         )
     )
     ```
   - Edge cases: None
   - Integration: Used by InputArrays manager for host/device transfers

3. **Update driver_coefficients stride_order**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Details:
     ```python
     # Lines 40-46: Change stride_order
     # Current:
     driver_coefficients: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("time", "run", "variable"),
             shape=(1, 1, 1),
             is_chunked=False,
         )
     )
     # Target:
     driver_coefficients: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("time", "variable", "run"),
             shape=(1, 1, 1),
             is_chunked=False,
         )
     )
     ```
   - Edge cases: is_chunked=False remains unchanged
   - Integration: Used for interpolated forcing terms

**Outcomes**: 
- [x] All InputArrayContainer fields use run-contiguous stride orders
- Files Modified:
  * src/cubie/batchsolving/arrays/BatchInputArrays.py (3 stride_order changes)

---

## Task Group 3: Output Array Container Stride Orders - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (lines 29-71)

**Input Validation Required**:
- None - these are default value changes only

**Tasks**:
1. **Update state stride_order**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     ```python
     # Lines 29-35: Change stride_order
     # Current:
     state: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("time", "run", "variable"),
             shape=(1, 1, 1),
         )
     )
     # Target:
     state: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("time", "variable", "run"),
             shape=(1, 1, 1),
         )
     )
     ```
   - Edge cases: None
   - Integration: Used by OutputArrays manager for host/device transfers

2. **Update observables stride_order**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     ```python
     # Lines 36-42: Change stride_order
     # Current:
     observables: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("time", "run", "variable"),
             shape=(1, 1, 1),
         )
     )
     # Target:
     observables: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("time", "variable", "run"),
             shape=(1, 1, 1),
         )
     )
     ```
   - Edge cases: None
   - Integration: Used by OutputArrays manager for host/device transfers

3. **Update state_summaries stride_order**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     ```python
     # Lines 43-49: Change stride_order
     # Current:
     state_summaries: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("time", "run", "variable"),
             shape=(1, 1, 1),
         )
     )
     # Target:
     state_summaries: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("time", "variable", "run"),
             shape=(1, 1, 1),
         )
     )
     ```
   - Edge cases: None
   - Integration: Used by OutputArrays manager for host/device transfers

4. **Update observable_summaries stride_order**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     ```python
     # Lines 50-56: Change stride_order
     # Current:
     observable_summaries: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("time", "run", "variable"),
             shape=(1, 1, 1),
         )
     )
     # Target:
     observable_summaries: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.float32,
             stride_order=("time", "variable", "run"),
             shape=(1, 1, 1),
         )
     )
     ```
   - Edge cases: None
   - Integration: Used by OutputArrays manager for host/device transfers

5. **Update iteration_counters stride_order**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Modify
   - Details:
     ```python
     # Lines 65-71: Change stride_order
     # Current:
     iteration_counters: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.int32,
             stride_order=("run", "time", "variable"),
             shape=(1, 1, 4),
         )
     )
     # Target:
     iteration_counters: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.int32,
             stride_order=("time", "variable", "run"),
             shape=(1, 4, 1),
         )
     )
     ```
   - Edge cases: Shape must change from (1, 1, 4) to (1, 4, 1) to match new stride_order semantics
   - Integration: Used by OutputArrays manager; stores Newton/Krylov iteration counts

6. **status_codes stride_order (No change needed)**
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Action: Verify (no change needed)
   - Details:
     ```python
     # Lines 57-63: Already correct (1D array)
     status_codes: ManagedArray = attrs.field(
         factory=lambda: ManagedArray(
             dtype=np.int32,
             stride_order=("run",),
             shape=(1,),
             is_chunked=False,
         )
     )
     ```
   - Edge cases: 1D array has no reordering needed
   - Integration: Used for per-run status codes

**Outcomes**: 
- [x] All OutputArrayContainer fields use run-contiguous stride orders
- [x] iteration_counters shape updated to match new stride semantics
- Files Modified:
  * src/cubie/batchsolving/arrays/BatchOutputArrays.py (5 stride_order changes + shape change)

---

## Task Group 4: Output Sizing Classes - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 2, 3

**Required Context**:
- File: src/cubie/outputhandling/output_sizes.py (lines 374-552)

**Input Validation Required**:
- None - these are default value changes only

**Tasks**:
1. **Update BatchInputSizes.stride_order default**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Lines 405-410: Change default stride_order
     # Current:
     stride_order: Tuple[str, ...] = attrs.field(
         default=("run", "variable"),
         validator=attrs.validators.deep_iterable(
             attrs.validators.in_(["run", "variable"])
         ),
     )
     # Target:
     stride_order: Tuple[str, ...] = attrs.field(
         default=("variable", "run"),
         validator=attrs.validators.deep_iterable(
             attrs.validators.in_(["run", "variable"])
         ),
     )
     ```
   - Edge cases: Validator allows both orderings
   - Integration: Used by InputArrays for sizing calculations

2. **Update BatchOutputSizes.stride_order default**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Lines 483-488: Change default stride_order
     # Current:
     stride_order: Tuple[str, ...] = attrs.field(
         default=("time", "run", "variable"),
         validator=attrs.validators.deep_iterable(
             attrs.validators.in_(["time", "run", "variable"])
         ),
     )
     # Target:
     stride_order: Tuple[str, ...] = attrs.field(
         default=("time", "variable", "run"),
         validator=attrs.validators.deep_iterable(
             attrs.validators.in_(["time", "run", "variable"])
         ),
     )
     ```
   - Edge cases: Validator allows any ordering of the three elements
   - Integration: Used by OutputArrays for sizing calculations

3. **Update BatchOutputSizes.iteration_counters default**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Lines 480-481: Change default shape
     # Current:
     iteration_counters: Tuple[int, int, int] = attrs.field(
         default=(1, 1, 4), validator=attrs.validators.instance_of(Tuple)
     )
     # Target:
     iteration_counters: Tuple[int, int, int] = attrs.field(
         default=(1, 4, 1), validator=attrs.validators.instance_of(Tuple)
     )
     ```
   - Edge cases: Shape must match new (time, variable, run) semantics
   - Integration: Used for iteration counter sizing

4. **Update BatchOutputSizes.from_solver iteration_counters calculation**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Lines 536-542: Change iteration_counters shape calculation
     # Current:
     # Iteration counters have shape (n_runs, n_saves, 4)
     # where 4 is for [Newton, Krylov, steps, rejections]
     iteration_counters = (
         num_runs,
         single_run_sizes.state[0],  # n_saves
         4,
     )
     # Target:
     # Iteration counters have shape (n_saves, 4, n_runs)
     # where 4 is for [Newton, Krylov, steps, rejections]
     iteration_counters = (
         single_run_sizes.state[0],  # n_saves
         4,
         num_runs,
     )
     ```
   - Edge cases: n_saves can be 0 if no saves configured
   - Integration: Must match kernel indexing

5. **Update BatchOutputSizes docstring**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Lines 438-463: Update docstring to reflect new layout
     # Current docstring mentions:
     # "3D array sizes (time × run × variable)"
     # "Shape of state output array as (time_samples, n_runs, n_variables)"
     
     # Target docstring should say:
     # "3D array sizes (time × variable × run)"
     # "Shape of state output array as (time_samples, n_variables, n_runs)"
     ```
   - Edge cases: None
   - Integration: Documentation accuracy

**Outcomes**: 
- [x] BatchInputSizes uses run-contiguous stride order
- [x] BatchOutputSizes uses run-contiguous stride order
- [x] iteration_counters sizing matches new layout
- [x] Docstrings updated to reflect new layout
- Files Modified:
  * src/cubie/outputhandling/output_sizes.py (stride_order defaults, from_solver shapes, docstrings)

---

## Task Group 5: BatchSolverKernel Array Indexing - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 2, 3, 4

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 497-620)

**Input Validation Required**:
- None - these are indexing changes only

**Tasks**:
1. **Update kernel signature contiguity annotations**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Verify (already correct)
   - Details:
     ```python
     # Lines 497-513: Signature already uses correct contiguity
     @cuda.jit(
         (
             precision[:, ::1],    # inits: (variable, run) - run-contiguous
             precision[:, ::1],    # params: (variable, run) - run-contiguous
             precision[:, :, ::1], # d_coefficients: (time, variable, run)
             precision[:, :, :],   # state_output (view, non-contiguous)
             precision[:, :, :],   # observables_output
             precision[:, :, :],   # state_summaries_output
             precision[:, :, :],   # observables_summaries_output
             int32[:, :, :],       # iteration_counters_output
             int32[::1],           # status_codes_output
             ...
         ),
         **compile_kwargs,
     )
     ```
   - Edge cases: Output arrays use non-contiguous views after slicing
   - Integration: Numba JIT compiler uses these for optimization

2. **Update inits array indexing**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Line 587: Change indexing
     # Current:
     rx_inits = inits[run_index, :]
     # Target:
     rx_inits = inits[:, run_index]
     ```
   - Edge cases: None
   - Integration: Passed to loopfunction

3. **Update params array indexing**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Line 588: Change indexing
     # Current:
     rx_params = params[run_index, :]
     # Target:
     rx_params = params[:, run_index]
     ```
   - Edge cases: None
   - Integration: Passed to loopfunction

4. **Update state_output array indexing**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Line 589: Change indexing
     # Current:
     rx_state = state_output[:, run_index * save_state, :]
     # Target:
     rx_state = state_output[:, :, run_index * save_state]
     ```
   - Edge cases: save_state is 0 or 1 boolean flag
   - Integration: Passed to loopfunction for state saves

5. **Update observables_output array indexing**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Lines 590-592: Change indexing
     # Current:
     rx_observables = observables_output[
         :, run_index * save_observables, :
     ]
     # Target:
     rx_observables = observables_output[
         :, :, run_index * save_observables
     ]
     ```
   - Edge cases: save_observables is 0 or 1 boolean flag
   - Integration: Passed to loopfunction for observable saves

6. **Update state_summaries_output array indexing**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Lines 593-595: Change indexing
     # Current:
     rx_state_summaries = state_summaries_output[
         :, run_index * save_state_summaries, :
     ]
     # Target:
     rx_state_summaries = state_summaries_output[
         :, :, run_index * save_state_summaries
     ]
     ```
   - Edge cases: save_state_summaries is 0 or 1 boolean flag
   - Integration: Passed to loopfunction for state summary saves

7. **Update observables_summaries_output array indexing**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Lines 596-598: Change indexing
     # Current:
     rx_observables_summaries = observables_summaries_output[
         :, run_index * save_observable_summaries, :
     ]
     # Target:
     rx_observables_summaries = observables_summaries_output[
         :, :, run_index * save_observable_summaries
     ]
     ```
   - Edge cases: save_observable_summaries is 0 or 1 boolean flag
   - Integration: Passed to loopfunction for observable summary saves

8. **Update iteration_counters_output array indexing**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Lines 599-601: Change indexing
     # Current:
     rx_iteration_counters = iteration_counters_output[
         run_index, :, :
     ]
     # Target:
     rx_iteration_counters = iteration_counters_output[
         :, :, run_index
     ]
     ```
   - Edge cases: None
   - Integration: Passed to loopfunction for iteration counter saves

9. **Update critical_shapes in build_kernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Lines 630-644: Update critical_shapes to reflect new layout
     # Current:
     integration_kernel.critical_shapes = (
         (1, n_states),  # inits - [n_runs, n_states]
         (1, n_parameters),  # params - [n_runs, n_parameters]
         (100,n_states,6),  # d_coefficients - complex driver array
         (100, 1, n_states), # state_output
         (100, 1, n_observables),  # observables_output
         (100, 1, n_observables), # state_summaries_output
         (100, 1, n_observables), # observables_summaries_output
         (100, 1, 4), # iteration_counters_output
         (1,),  # status_codes_output
         ...
     )
     # Target:
     integration_kernel.critical_shapes = (
         (n_states, 1),  # inits - [n_states, n_runs]
         (n_parameters, 1),  # params - [n_parameters, n_runs]
         (100, n_states, 6),  # d_coefficients - (time, variable, run)
         (100, n_states, 1), # state_output - (time, variable, run)
         (100, n_observables, 1),  # observables_output
         (100, n_observables, 1), # state_summaries_output
         (100, n_observables, 1), # observables_summaries_output
         (100, 4, 1), # iteration_counters_output - (time, 4, run)
         (1,),  # status_codes_output
         ...
     )
     ```
   - Edge cases: None
   - Integration: Used for dummy compilation during kernel build

**Outcomes**: 
- [x] Kernel uses run-contiguous indexing for all arrays
- [x] critical_shapes reflect new layout
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (8 indexing changes + critical_shapes)

---

## Task Group 6: SolveResult Stride Order - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 4, 5

**Required Context**:
- File: src/cubie/batchsolving/solveresult.py (lines 186-188, 469-470)

**Input Validation Required**:
- None - these are default value changes only

**Tasks**:
1. **Update SolveResult._stride_order default**
   - File: src/cubie/batchsolving/solveresult.py
   - Action: Modify
   - Details:
     ```python
     # Lines 186-188: Change default stride_order
     # Current:
     _stride_order: Union[tuple[str, ...], list[str]] = attrs.field(
         default=("time", "run", "variable")
     )
     # Target:
     _stride_order: Union[tuple[str, ...], list[str]] = attrs.field(
         default=("time", "variable", "run")
     )
     ```
   - Edge cases: None
   - Integration: Used by as_pandas and per_summary_arrays for index lookup

2. **Update cleave_time default stride_order**
   - File: src/cubie/batchsolving/solveresult.py
   - Action: Modify
   - Details:
     ```python
     # Lines 469-470: Change default stride_order in cleave_time
     # Current:
     if stride_order is None:
         stride_order = ["time", "run", "variable"]
     # Target:
     if stride_order is None:
         stride_order = ["time", "variable", "run"]
     ```
   - Edge cases: Called with explicit stride_order from solver
   - Integration: Used for extracting time column from state array

**Outcomes**: 
- [x] SolveResult uses run-contiguous stride order by default
- [x] cleave_time uses correct default
- Files Modified:
  * src/cubie/batchsolving/solveresult.py (2 default value changes)

---

## Task Group 7: Test File Updates - Memory Tests - PARALLEL
**Status**: [x]
**Dependencies**: Groups 1-6

**Required Context**:
- File: tests/memory/test_memmgmt.py (lines 117, 373-403, 519-548, 708-728)
- File: tests/memory/test_array_requests.py (lines 38, 48, 53)
- File: tests/memory/conftest.py (line 19)

**Input Validation Required**:
- None - test fixture updates only

**Tasks**:
1. **Update mem_manager_settings fixture default stride_order**
   - File: tests/memory/test_memmgmt.py
   - Action: Modify
   - Details:
     ```python
     # Line 117: Change default in fixture
     # Current:
     defaults = {"mode": "passive", "stride_order": ("time", "run", "variable")}
     # Target:
     defaults = {"mode": "passive", "stride_order": ("time", "variable", "run")}
     ```
   - Edge cases: None
   - Integration: Used by mgr fixture

2. **Update test_set_strides stride_order values**
   - File: tests/memory/test_memmgmt.py
   - Action: Modify
   - Details:
     ```python
     # Lines 373-394: Update test assertions
     # Current test creates ArrayRequest with stride_order=("time", "run", "variable")
     # and expects mgr.get_strides(req) to return None
     
     # The test logic should change:
     # - Request with stride_order=("time", "variable", "run") should return None (matches default)
     # - Request with stride_order=("time", "run", "variable") should return non-None strides
     ```
   - Edge cases: Manually computed strides need recalculation
   - Integration: Tests stride calculation logic

3. **Update test_set_global_stride_ordering assertions**
   - File: tests/memory/test_memmgmt.py
   - Action: Modify
   - Details:
     ```python
     # Lines 396-403: Test uses ("run", "variable", "time") which is still valid
     # No change needed - tests arbitrary ordering, not default
     ```
   - Edge cases: None
   - Integration: Tests set_global_stride_ordering method

4. **Update test_get_strides assertions**
   - File: tests/memory/test_memmgmt.py
   - Action: Modify
   - Details:
     ```python
     # Lines 519-548: Update test assertions
     # Current:
     # - req_default with ("time", "run", "variable") expects None
     # - req_custom with ("run", "variable", "time") expects strides
     
     # Target:
     # - req_default with ("time", "variable", "run") expects None
     # - req_custom with ("time", "run", "variable") expects strides (non-default now)
     ```
   - Edge cases: None
   - Integration: Tests stride calculation

5. **Update test_default_stride_ordering in test_array_requests.py**
   - File: tests/memory/test_array_requests.py
   - Action: Modify
   - Details:
     ```python
     # Lines 41-58: Update expected defaults
     # Current test expects 3D default to be ("time", "run", "variable")
     # Target: 3D default should be ("time", "variable", "run")
     
     # Current:
     assert req3.stride_order == ("time", "run", "variable")
     # Target:
     assert req3.stride_order == ("time", "variable", "run")
     ```
   - Edge cases: 2D default is already ("variable", "run") - no change
   - Integration: Tests ArrayRequest defaults

6. **Update conftest.py array_request_settings fixture**
   - File: tests/memory/conftest.py
   - Action: Modify
   - Details:
     ```python
     # Line 19: Update default stride_order
     # Current:
     defaults = {
         ...
         "stride_order": ("time", "run", "variable"),
     }
     # Target:
     defaults = {
         ...
         "stride_order": ("time", "variable", "run"),
     }
     ```
   - Edge cases: None
   - Integration: Used by array_request fixture

7. **Update test_instantiation assertion in test_array_requests.py**
   - File: tests/memory/test_array_requests.py
   - Action: Modify
   - Details:
     ```python
     # Line 38: Update expected stride_order
     # Current:
     assert array_request.stride_order == ("time", "run", "variable")
     # Target:
     assert array_request.stride_order == ("time", "variable", "run")
     ```
   - Edge cases: None
   - Integration: Tests ArrayRequest instantiation

**Outcomes**: 
- [x] Memory tests use new stride order defaults
- [x] Test assertions match new expected values
- Files Modified:
  * tests/memory/test_memmgmt.py (fixture defaults, test assertions, stride calculations)
  * tests/memory/test_array_requests.py (expected stride_order assertions)
  * tests/memory/conftest.py (fixture stride_order default)

---

## Task Group 8: Test File Updates - Batch Array Tests - PARALLEL
**Status**: [x]
**Dependencies**: Groups 1-6

**Required Context**:
- File: tests/batchsolving/arrays/test_basearraymanager.py (lines 44-96, 109-131, 172-218, 659-667)
- File: tests/batchsolving/arrays/test_batchinputarrays.py (line 96)
- File: tests/batchsolving/arrays/test_batchoutputarrays.py (line 113)

**Input Validation Required**:
- None - test fixture updates only

**Tasks**:
1. **Update TestArrays stride_order in test_basearraymanager.py**
   - File: tests/batchsolving/arrays/test_basearraymanager.py
   - Action: Modify
   - Details:
     ```python
     # Lines 44-76: Update stride_order in TestArrays class
     # Current: stride_order=("time", "run", "variable") for all fields
     # Target: stride_order=("time", "variable", "run") for all fields
     ```
   - Edge cases: None
   - Integration: Test fixture for BaseArrayManager tests

2. **Update TestArraysSimple stride_order**
   - File: tests/batchsolving/arrays/test_basearraymanager.py
   - Action: Modify
   - Details:
     ```python
     # Lines 79-96: Update stride_order in TestArraysSimple class
     # Current: stride_order=("time", "run", "variable")
     # Target: stride_order=("time", "variable", "run")
     ```
   - Edge cases: None
   - Integration: Test fixture for simple array tests

3. **Update arraytest_settings fixture stride_template**
   - File: tests/batchsolving/arrays/test_basearraymanager.py
   - Action: Modify
   - Details:
     ```python
     # Lines 109-131: Update stride_template
     # Current:
     stride_template = ("time", "run", "variable")
     # Target:
     stride_template = ("time", "variable", "run")
     ```
   - Edge cases: None
   - Integration: Used by hostarrays, devarrays fixtures

4. **Update batch_output_sizes fixture stride_order**
   - File: tests/batchsolving/arrays/test_basearraymanager.py
   - Action: Modify
   - Details:
     ```python
     # Lines 659-667: Update stride_order in fixture
     # Current:
     return BatchOutputSizes(
         ...
         stride_order=arraytest_settings["stride_tuple"],
     )
     # The stride_tuple is set from stride_template which is updated above
     ```
   - Edge cases: None
   - Integration: Used by test_manager_with_sizing

5. **Update test_container_stride_order assertion in test_batchinputarrays.py**
   - File: tests/batchsolving/arrays/test_batchinputarrays.py
   - Action: Modify
   - Details:
     ```python
     # Line 96: Update expected stride_order
     # Current:
     assert stride_order == ("run", "variable")
     # Target:
     assert stride_order == ("variable", "run")
     ```
   - Edge cases: None
   - Integration: Tests InputArrayContainer stride order

6. **Update test_container_stride_order assertion in test_batchoutputarrays.py**
   - File: tests/batchsolving/arrays/test_batchoutputarrays.py
   - Action: Modify
   - Details:
     ```python
     # Line 113: Update expected stride_order
     # Current:
     assert stride_order == ("time", "run", "variable")
     # Target:
     assert stride_order == ("time", "variable", "run")
     ```
   - Edge cases: None
   - Integration: Tests OutputArrayContainer stride order

7. **Update sample_input_arrays fixture shape**
   - File: tests/batchsolving/arrays/test_batchinputarrays.py
   - Action: Modify
   - Details:
     ```python
     # Lines 55-73: Update array shapes to match new layout
     # Current shapes use (run, variable) ordering
     # Target shapes should use (variable, run) ordering
     
     # Current:
     "initial_values": np.random.rand(variables_count, num_runs).astype(dtype)
     # This is actually already (variables, runs) which matches (variable, run)!
     # No change needed - the current code creates arrays in the target layout
     ```
   - Edge cases: Arrays are already in the correct shape
   - Integration: Used by input array tests

8. **Update sample_output_arrays fixture shape**
   - File: tests/batchsolving/arrays/test_batchoutputarrays.py
   - Action: Modify
   - Details:
     ```python
     # Lines 59-84: Update array shapes to match new layout
     # Current:
     "state": np.random.rand(time_points, num_runs, variables_count).astype(dtype)
     # Target:
     "state": np.random.rand(time_points, variables_count, num_runs).astype(dtype)
     ```
   - Edge cases: All 3D arrays need shape swap
   - Integration: Used by output array tests

**Outcomes**: 
- [x] Batch array test fixtures use new stride orders
- [x] Test assertions match new expected values
- [x] Sample array shapes match new layout
- Files Modified:
  * tests/batchsolving/arrays/test_basearraymanager.py (TestArrays, TestArraysSimple, fixture stride_template)
  * tests/batchsolving/arrays/test_batchinputarrays.py (stride_order assertion)
  * tests/batchsolving/arrays/test_batchoutputarrays.py (stride_order assertion, sample array shapes)
  * tests/batchsolving/test_solveresult.py (stride_order assertion, cleave_time test arrays)
  * tests/outputhandling/test_output_sizes.py (batch shape assertions)

---

## Task Group 9: Test File Updates - Output Sizes Tests - PARALLEL
**Status**: [x]
**Dependencies**: Groups 4

**Required Context**:
- File: tests/outputhandling/test_output_sizes.py (lines 562-565)

**Input Validation Required**:
- None - test assertion updates only

**Tasks**:
1. **Update test_stride_order_default assertion for BatchInputSizes**
   - File: tests/outputhandling/test_output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Lines 562-565: Update expected stride_order
     # Current:
     def test_stride_order_default(self):
         """Test that _stride_order has correct default value"""
         sizes = BatchInputSizes()
         assert sizes.stride_order == ("run", "variable")
     # Target:
     def test_stride_order_default(self):
         """Test that stride_order has correct default value"""
         sizes = BatchInputSizes()
         assert sizes.stride_order == ("variable", "run")
     ```
   - Edge cases: None
   - Integration: Tests BatchInputSizes default

**Outcomes**: 
- [x] Output sizes tests use new stride order defaults
- Files Modified:
  * tests/outputhandling/test_output_sizes.py (stride_order and batch shape assertions)

---

## Summary

### Total Task Groups: 9
### Dependency Chain:
1. Group 1 (Memory Layer Defaults) - Foundation
2. Groups 2, 3 (Array Containers) - Depend on Group 1
3. Group 4 (Output Sizing) - Depends on Groups 2, 3
4. Group 5 (Kernel Indexing) - Depends on Groups 2, 3, 4
5. Group 6 (SolveResult) - Depends on Groups 4, 5
6. Groups 7, 8, 9 (Tests) - Can run in parallel after Groups 1-6

### Parallel Execution Opportunities:
- Groups 2 and 3 can run in parallel (both depend only on Group 1)
- Groups 7, 8, and 9 can run in parallel (all test updates)

### Estimated Complexity:
- Group 1: Low (2 tasks, simple defaults)
- Group 2: Low (3 tasks, simple defaults)
- Group 3: Medium (6 tasks, includes shape change)
- Group 4: Medium (5 tasks, includes docstring and calculation changes)
- Group 5: High (9 tasks, kernel indexing is critical)
- Group 6: Low (2 tasks, simple defaults)
- Group 7: Medium (7 tasks, test updates)
- Group 8: Medium (8 tasks, test updates with shape changes)
- Group 9: Low (1 task, single assertion)

### Critical Path:
Group 1 → Groups 2+3 → Group 4 → Group 5 → Group 6 → Tests

---

# Implementation Complete - Ready for Review

## Execution Summary
- Total Task Groups: 9
- Completed: 9
- Failed: 0
- Total Files Modified: 14

## Task Group Completion
- Group 1: [x] Memory Layer Defaults - Complete
- Group 2: [x] Input Array Container Stride Orders - Complete
- Group 3: [x] Output Array Container Stride Orders - Complete
- Group 4: [x] Output Sizing Classes - Complete
- Group 5: [x] BatchSolverKernel Array Indexing - Complete
- Group 6: [x] SolveResult Stride Order - Complete
- Group 7: [x] Memory Tests - Complete
- Group 8: [x] Batch Array Tests - Complete
- Group 9: [x] Output Sizes Tests - Complete

## All Modified Files
### Source Files:
1. src/cubie/memory/array_requests.py (default stride_order, docstring)
2. src/cubie/batchsolving/arrays/BatchInputArrays.py (stride_order defaults)
3. src/cubie/batchsolving/arrays/BatchOutputArrays.py (stride_order defaults, shape)
4. src/cubie/outputhandling/output_sizes.py (defaults, from_solver, docstrings)
5. src/cubie/batchsolving/BatchSolverKernel.py (array indexing, critical_shapes)
6. src/cubie/batchsolving/solveresult.py (stride_order defaults)

### Test Files:
7. tests/memory/test_memmgmt.py (fixture, test assertions)
8. tests/memory/test_array_requests.py (expected assertions)
9. tests/memory/conftest.py (fixture default)
10. tests/batchsolving/arrays/test_basearraymanager.py (fixtures, stride_template)
11. tests/batchsolving/arrays/test_batchinputarrays.py (stride_order assertion)
12. tests/batchsolving/arrays/test_batchoutputarrays.py (stride_order, sample shapes)
13. tests/batchsolving/test_solveresult.py (stride_order, cleave_time test)
14. tests/outputhandling/test_output_sizes.py (stride_order, batch shapes)

## Flagged Issues
None - all changes follow the specification exactly.

## Handoff to Reviewer
All implementation tasks complete. Task list updated with outcomes.
Ready for reviewer agent to validate against user stories and goals.
