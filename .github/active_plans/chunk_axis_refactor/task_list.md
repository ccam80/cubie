# Implementation Task List
# Feature: Chunk Axis Refactor
# Plan Reference: .github/active_plans/chunk_axis_refactor/agent_plan.md

## Task Group 1: Set chunk_axis Early in BatchSolverKernel.run()
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 140-160, 380-440)

**Input Validation Required**:
- None - chunk_axis parameter is already validated by the run() method signature

**Tasks**:
1. **Add early chunk_axis assignment in run()**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     Add `self.chunk_axis = chunk_axis` in the run() method BEFORE the calls
     to `self.input_arrays.update()` and `self.output_arrays.update()`.

     Current code block (around lines 414-430):
     ```python
     # Refresh compile-critical settings before array updates
     self.update_compile_settings(
         {
             "loop_fn": self.single_integrator.compiled_loop_function,
             "precision": self.single_integrator.precision,
             "local_memory_elements": (
                 self.single_integrator.local_memory_elements
             ),
             "shared_memory_elements": (
                 self.single_integrator.shared_memory_elements
             ),
         }
     )

     # Queue allocations
     self.input_arrays.update(self, inits, params, driver_coefficients)
     ```

     Modified code:
     ```python
     # Refresh compile-critical settings before array updates
     self.update_compile_settings(
         {
             "loop_fn": self.single_integrator.compiled_loop_function,
             "precision": self.single_integrator.precision,
             "local_memory_elements": (
                 self.single_integrator.local_memory_elements
             ),
             "shared_memory_elements": (
                 self.single_integrator.shared_memory_elements
             ),
         }
     )

     # Set chunk_axis before array updates so any code reading
     # solver.chunk_axis during updates gets the correct value
     self.chunk_axis = chunk_axis

     # Queue allocations
     self.input_arrays.update(self, inits, params, driver_coefficients)
     ```
   - Edge cases: None - this is a straightforward assignment
   - Integration: This ensures solver.chunk_axis is set before arrays are updated

**Tests to Create**:
- None - existing tests cover this behavior

**Tests to Run**:
- tests/batchsolving/test_chunked_solver.py::TestChunkedSolverExecution::test_chunked_solve_produces_valid_output

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (4 lines added)
- Functions/Methods Added/Modified:
  * run() method in BatchSolverKernel.py
- Implementation Summary:
  Added `self.chunk_axis = chunk_axis` assignment after update_compile_settings()
  and before input_arrays.update() call. This ensures solver.chunk_axis is set
  before any array update code reads the chunk_axis attribute.
- Issues Flagged: None

---

## Task Group 2: Remove Early chunk_axis Read from BatchInputArrays
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (lines 277-301)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 272-316)

**Input Validation Required**:
- None - this is a removal of a line, no new input validation needed

**Tasks**:
1. **Remove chunk_axis assignment from update_from_solver()**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Details:
     Remove line 292 which sets `self._chunk_axis = solver_instance.chunk_axis`.
     The `_chunk_axis` attribute is correctly set by `_on_allocation_complete()`
     callback in the base class (BaseArrayManager, line 315).

     Current code block (lines 289-293):
     ```python
         """
         self._sizes = BatchInputSizes.from_solver(solver_instance).nonzero
         self._precision = solver_instance.precision
         self._chunk_axis = solver_instance.chunk_axis
         for name, arr_obj in self.host.iter_managed_arrays():
     ```

     Modified code:
     ```python
         """
         self._sizes = BatchInputSizes.from_solver(solver_instance).nonzero
         self._precision = solver_instance.precision
         for name, arr_obj in self.host.iter_managed_arrays():
     ```
   - Edge cases:
     - The `_chunk_axis` attribute has a default value of "run" from the attrs
       field definition in BaseArrayManager, so removal is safe
     - The `_on_allocation_complete()` callback will set `_chunk_axis` correctly
       from the ArrayResponse after the memory manager determines chunking
   - Integration: This ensures `_chunk_axis` is only set once, by the allocation
     callback, which is the authoritative source of truth

2. **Update docstring to remove mention of chunk_axis**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Details:
     Update the docstring for `update_from_solver()` to remove mention of
     "chunk axis" since we're no longer setting it.

     Current docstring (lines 277-279):
     ```python
     def update_from_solver(self, solver_instance: "BatchSolverKernel") -> None:
         """Refresh size, precision, and chunk axis from the solver.
     ```

     Modified docstring:
     ```python
     def update_from_solver(self, solver_instance: "BatchSolverKernel") -> None:
         """Refresh size and precision from the solver.
     ```
   - Edge cases: None
   - Integration: Keeps documentation accurate

**Tests to Create**:
- None - existing tests cover this behavior

**Tests to Run**:
- tests/batchsolving/test_chunked_solver.py::TestChunkedSolverExecution::test_chunked_solve_produces_valid_output

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/arrays/BatchInputArrays.py (2 lines changed)
- Functions/Methods Added/Modified:
  * update_from_solver() in BatchInputArrays.py
- Implementation Summary:
  Removed the early `self._chunk_axis = solver_instance.chunk_axis` assignment
  from update_from_solver() method. The chunk_axis is now set only by the
  _on_allocation_complete() callback in BaseArrayManager (line 315), which is
  the authoritative source of truth. Also updated the docstring to reflect
  that the method now only refreshes "size and precision" (removed "chunk axis").
- Issues Flagged: None

---

## Task Group 3: Verify All Tests Pass
**Status**: [x]
**Dependencies**: Task Groups 1 and 2

**Required Context**:
- File: tests/batchsolving/test_chunked_solver.py (entire file)

**Input Validation Required**:
- None - verification only

**Tasks**:
1. **Run chunked solver tests**
   - File: N/A
   - Action: Verify
   - Details:
     Run the chunked solver tests to verify both "run" and "time" chunk axes
     work correctly with the changes.

     ```bash
     pytest tests/batchsolving/test_chunked_solver.py -v
     ```

     Expected results:
     - test_chunked_solve_produces_valid_output[run] passes
     - test_chunked_solve_produces_valid_output[time] passes

   - Edge cases: None
   - Integration: This validates the complete fix

**Tests to Create**:
- None

**Tests to Run**:
- tests/batchsolving/test_chunked_solver.py::TestChunkedSolverExecution::test_chunked_solve_produces_valid_output

**Outcomes**:
- Files Modified:
  * None (verification task only)
- Functions/Methods Added/Modified:
  * None (verification task only)
- Implementation Summary:
  This task group is a verification step only. The run_tests agent should
  execute the chunked solver tests to verify both "run" and "time" chunk
  axes work correctly with the changes made in Task Groups 1 and 2.
  Tests to run:
  - test_chunked_solve_produces_valid_output[run]
  - test_chunked_solve_produces_valid_output[time]
- Issues Flagged: None
