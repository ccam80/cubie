# Implementation Task List
# Feature: chunk_axis Refactoring - BaseArrayManager as Single Source of Truth
# Plan Reference: .github/active_plans/chunk_axis_refactor/agent_plan.md

## Task Group 1: BatchSolverKernel Property/Setter Implementation
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 1-200, focus on __init__ and imports)
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 886-995, existing properties pattern)
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 166-227, BaseArrayManager class definition with _chunk_axis)
- File: .github/context/cubie_internal_structure.md (entire file for architectural patterns)

**Input Validation Required**:
- Property getter: No validation needed (attrs validates on array managers)
- Property setter: No validation needed (attrs on BaseArrayManager already validates against `["run", "variable", "time"]`)

**Tasks**:
1. **Remove public chunk_axis attribute from __init__**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Remove line 150: self.chunk_axis = "run"
     # This line is no longer needed as chunk_axis is now managed
     # by the array managers. The property/setter pattern provides
     # coordinated access.
     ```
   - Edge cases: None - the attribute is never used before array managers are created
   - Integration: Array managers are created on lines 183-184, they initialize with `_chunk_axis="run"` by default

2. **Add chunk_axis property getter**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify (add property after existing properties, around line 1043)
   - Details:
     ```python
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
     ```
   - Edge cases: 
     - Array managers not yet created: Not possible - they are created in __init__
     - Inconsistent values: Raises ValueError with descriptive message
   - Integration: Property reads directly from array manager private attributes

3. **Add chunk_axis property setter**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify (add setter immediately after getter)
   - Details:
     ```python
     @chunk_axis.setter
     def chunk_axis(self, value: str) -> None:
         """Set chunk_axis on both input and output array managers.

         Parameters
         ----------
         value
             The chunking axis to set.
         """
         self.input_arrays._chunk_axis = value
         self.output_arrays._chunk_axis = value
     ```
   - Edge cases: Invalid values will be caught by attrs validator on next allocation callback
   - Integration: Updates both array managers atomically

**Tests to Create**:
- Test file: tests/batchsolving/test_chunk_axis_property.py
- Test function: test_chunk_axis_property_returns_default_run
- Description: Verify that chunk_axis property returns "run" by default after kernel creation
- Test function: test_chunk_axis_setter_updates_both_arrays
- Description: Verify that setting chunk_axis updates both input_arrays and output_arrays
- Test function: test_chunk_axis_property_raises_on_inconsistency
- Description: Verify that property raises ValueError when arrays have mismatched chunk_axis

**Tests to Run**:
- tests/batchsolving/test_chunk_axis_property.py::test_chunk_axis_property_returns_default_run
- tests/batchsolving/test_chunk_axis_property.py::test_chunk_axis_setter_updates_both_arrays
- tests/batchsolving/test_chunk_axis_property.py::test_chunk_axis_property_raises_on_inconsistency
- tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisProperty::test_chunk_axis_property_returns_consistent_value
- tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisSetter::test_chunk_axis_setter_allows_valid_values

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (39 lines changed - removed 1 line, added 38 lines)
  * tests/batchsolving/test_chunk_axis_property.py (55 lines - new file)
- Functions/Methods Added/Modified:
  * chunk_axis property getter in BatchSolverKernel
  * chunk_axis property setter in BatchSolverKernel
  * Removed self.chunk_axis = "run" assignment from __init__
- Implementation Summary:
  Removed the public chunk_axis attribute from BatchSolverKernel.__init__ and replaced
  it with a property/setter pair. The getter reads from both input_arrays and output_arrays
  _chunk_axis attributes, validates they match, and returns the value. The setter updates
  both array managers atomically. Created initial test file with 5 tests covering default
  value, consistency, inconsistency error, setter behavior, and valid values.
- Issues Flagged: None

---

## Task Group 2: BatchSolverKernel.run() Setter Call
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 343-445, run() method)
- File: .github/active_plans/chunk_axis_refactor/agent_plan.md (Component 2 section)

**Input Validation Required**:
- chunk_axis parameter: No explicit validation needed (setter propagates to attrs-validated fields)

**Tasks**:
1. **Add chunk_axis setter call in run() method**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # After the timing parameters are stored (around line 402), add:
     # Current code:
     #     self._duration = duration
     #     self._warmup = warmup
     #     self._t0 = t0
     #
     # Add this line immediately after:
     self.chunk_axis = chunk_axis
     #
     # This ensures both array managers have the correct chunk_axis
     # value BEFORE update() is called on them (lines 429-430).
     ```
   - Edge cases: 
     - Invalid chunk_axis: Will be caught by attrs validator when allocation callback fires
   - Integration: This sets the value before `input_arrays.update()` and `output_arrays.update()` are called

**Tests to Create**:
- Test file: tests/batchsolving/test_chunk_axis_property.py
- Test function: test_run_sets_chunk_axis_on_arrays
- Description: Verify that calling kernel.run() with chunk_axis="time" sets the value on both array managers
- Test function: test_chunk_axis_property_after_run
- Description: Verify that chunk_axis property returns correct value after run() completes

**Tests to Run**:
- tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisInRun::test_run_sets_chunk_axis_on_arrays
- tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisInRun::test_chunk_axis_property_after_run

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (3 lines changed - added setter call in run())
  * tests/batchsolving/test_chunk_axis_property.py (58 lines changed - added TestChunkAxisInRun class with 2 tests)
- Functions/Methods Added/Modified:
  * run() method in BatchSolverKernel - added chunk_axis setter call after timing parameters
- Implementation Summary:
  Added `self.chunk_axis = chunk_axis` after the timing parameters (duration, warmup, t0) are stored
  in the run() method. This ensures both input_arrays and output_arrays have the correct chunk_axis
  value before update() is called on them. Created TestChunkAxisInRun test class with two tests:
  test_run_sets_chunk_axis_on_arrays verifies both array managers are updated, and
  test_chunk_axis_property_after_run verifies the property returns the correct value after run.
- Issues Flagged: None

---

## Task Group 3: BatchInputArrays Cleanup
**Status**: [x]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (lines 277-301, update_from_solver method)
- File: .github/active_plans/chunk_axis_refactor/agent_plan.md (Component 3 section)

**Input Validation Required**:
- None - this is a removal task

**Tasks**:
1. **Remove redundant chunk_axis assignment from update_from_solver()**
   - File: src/cubie/batchsolving/arrays/BatchInputArrays.py
   - Action: Modify
   - Details:
     ```python
     # Remove line 292: self._chunk_axis = solver_instance.chunk_axis
     #
     # Current code (lines 289-292):
     #     def update_from_solver(self, solver_instance: "BatchSolverKernel") -> None:
     #         ...
     #         self._sizes = BatchInputSizes.from_solver(solver_instance).nonzero
     #         self._precision = solver_instance.precision
     #         self._chunk_axis = solver_instance.chunk_axis  # REMOVE THIS LINE
     #         for name, arr_obj in self.host.iter_managed_arrays():
     #
     # After removal, the method should look like:
     #     def update_from_solver(self, solver_instance: "BatchSolverKernel") -> None:
     #         ...
     #         self._sizes = BatchInputSizes.from_solver(solver_instance).nonzero
     #         self._precision = solver_instance.precision
     #         for name, arr_obj in self.host.iter_managed_arrays():
     ```
   - Edge cases: None - chunk_axis is now set by the kernel's setter before this method is called
   - Integration: The kernel.run() method sets chunk_axis on both array managers BEFORE calling update(), so this assignment is redundant

**Tests to Create**:
- Test file: tests/batchsolving/test_chunk_axis_property.py
- Test function: test_update_from_solver_does_not_change_chunk_axis
- Description: Verify that update_from_solver() preserves an existing chunk_axis value on the array manager

**Tests to Run**:
- tests/batchsolving/test_chunk_axis_property.py::TestUpdateFromSolverChunkAxis::test_update_from_solver_does_not_change_chunk_axis

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/arrays/BatchInputArrays.py (1 line removed)
  * tests/batchsolving/test_chunk_axis_property.py (24 lines added)
- Functions/Methods Added/Modified:
  * update_from_solver() in BatchInputArrays.py - removed redundant chunk_axis assignment
  * TestUpdateFromSolverChunkAxis class with test_update_from_solver_does_not_change_chunk_axis() added to test file
- Implementation Summary:
  Removed the redundant `self._chunk_axis = solver_instance.chunk_axis` line from the
  update_from_solver() method in BatchInputArrays.py. This assignment was redundant because
  the kernel.run() method now sets chunk_axis on both array managers via the setter BEFORE
  calling update(). Added test class TestUpdateFromSolverChunkAxis with test function
  test_update_from_solver_does_not_change_chunk_axis to verify that update_from_solver()
  preserves an existing chunk_axis value on the array manager.
- Issues Flagged: None

---

## Task Group 4: Test Suite for chunk_axis Property
**Status**: [x]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: tests/batchsolving/test_SolverKernel.py (entire file for testing patterns)
- File: tests/batchsolving/conftest.py (entire file for fixtures)
- File: tests/conftest.py (entire file for root fixtures - system, precision, driver_array fixtures)
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 84-198, class and __init__)
- File: src/cubie/batchsolving/arrays/BatchInputArrays.py (lines 128-180, InputArrays class)
- File: src/cubie/batchsolving/arrays/BatchOutputArrays.py (lines 1-100, OutputArrays class structure)

**Input Validation Required**:
- Tests should verify error messages match expected patterns

**Tasks**:
1. **Create test file for chunk_axis property tests**
   - File: tests/batchsolving/test_chunk_axis_property.py
   - Action: Create
   - Details:
     ```python
     """Tests for BatchSolverKernel.chunk_axis property and setter."""

     import numpy as np
     import pytest


     class TestChunkAxisProperty:
         """Tests for chunk_axis property getter behavior."""

         def test_chunk_axis_property_returns_default_run(
             self, solverkernel
         ):
             """Verify chunk_axis returns 'run' by default."""
             assert solverkernel.chunk_axis == "run"

         def test_chunk_axis_property_returns_consistent_value(
             self, solverkernel_mutable
         ):
             """Verify property returns value when arrays are consistent."""
             kernel = solverkernel_mutable
             # Both arrays should have same default value
             assert kernel.input_arrays._chunk_axis == "run"
             assert kernel.output_arrays._chunk_axis == "run"
             assert kernel.chunk_axis == "run"

         def test_chunk_axis_property_raises_on_inconsistency(
             self, solverkernel_mutable
         ):
             """Verify property raises ValueError for mismatched arrays."""
             kernel = solverkernel_mutable
             # Manually create inconsistent state
             kernel.input_arrays._chunk_axis = "run"
             kernel.output_arrays._chunk_axis = "time"

             with pytest.raises(ValueError, match=r"Inconsistent chunk_axis"):
                 _ = kernel.chunk_axis


     class TestChunkAxisSetter:
         """Tests for chunk_axis property setter behavior."""

         def test_chunk_axis_setter_updates_both_arrays(
             self, solverkernel_mutable
         ):
             """Verify setter updates both input and output arrays."""
             kernel = solverkernel_mutable
             kernel.chunk_axis = "time"

             assert kernel.input_arrays._chunk_axis == "time"
             assert kernel.output_arrays._chunk_axis == "time"

         def test_chunk_axis_setter_allows_valid_values(
             self, solverkernel_mutable
         ):
             """Verify setter accepts all valid chunk_axis values."""
             kernel = solverkernel_mutable
             for value in ["run", "variable", "time"]:
                 kernel.chunk_axis = value
                 assert kernel.chunk_axis == value


     class TestChunkAxisInRun:
         """Tests for chunk_axis handling in kernel.run()."""

         def test_run_sets_chunk_axis_on_arrays(
             self, system, precision, driver_array
         ):
             """Verify run() sets chunk_axis before array operations."""
             from cubie.batchsolving.BatchSolverKernel import (
                 BatchSolverKernel,
             )

             kernel = BatchSolverKernel(
                 system,
                 algorithm_settings={"algorithm": "euler"},
             )
             inits = np.ones((3, 1), dtype=precision)
             params = np.ones((3, 1), dtype=precision)

             kernel.run(
                 inits=inits,
                 params=params,
                 driver_coefficients=driver_array.coefficients,
                 duration=0.1,
                 chunk_axis="time",
             )

             # After run, both arrays should have the chunk_axis value
             assert kernel.input_arrays._chunk_axis == "time"
             assert kernel.output_arrays._chunk_axis == "time"

         def test_chunk_axis_property_after_run(
             self, system, precision, driver_array
         ):
             """Verify chunk_axis property returns correct value after run."""
             from cubie.batchsolving.BatchSolverKernel import (
                 BatchSolverKernel,
             )

             kernel = BatchSolverKernel(
                 system,
                 algorithm_settings={"algorithm": "euler"},
             )
             inits = np.ones((3, 1), dtype=precision)
             params = np.ones((3, 1), dtype=precision)

             kernel.run(
                 inits=inits,
                 params=params,
                 driver_coefficients=driver_array.coefficients,
                 duration=0.1,
                 chunk_axis="variable",
             )

             assert kernel.chunk_axis == "variable"


     class TestUpdateFromSolverChunkAxis:
         """Tests for update_from_solver chunk_axis behavior."""

         def test_update_from_solver_does_not_change_chunk_axis(
             self, system, precision, driver_array
         ):
             """Verify update_from_solver preserves existing chunk_axis."""
             from cubie.batchsolving.BatchSolverKernel import (
                 BatchSolverKernel,
             )

             kernel = BatchSolverKernel(
                 system,
                 algorithm_settings={"algorithm": "euler"},
             )

             # Set chunk_axis to non-default value via setter
             kernel.chunk_axis = "time"

             # Call update_from_solver (simulating what run() does)
             kernel.input_arrays.update_from_solver(kernel)

             # chunk_axis should be preserved
             assert kernel.input_arrays._chunk_axis == "time"
     ```
   - Edge cases: 
     - All valid values tested
     - Inconsistency error tested
     - Value preservation tested
   - Integration: Uses existing fixtures from conftest.py

**Tests to Create**:
- All tests listed in the code above

**Tests to Run**:
- tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisProperty::test_chunk_axis_property_returns_default_run
- tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisProperty::test_chunk_axis_property_returns_consistent_value
- tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisProperty::test_chunk_axis_property_raises_on_inconsistency
- tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisSetter::test_chunk_axis_setter_updates_both_arrays
- tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisSetter::test_chunk_axis_setter_allows_valid_values
- tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisInRun::test_run_sets_chunk_axis_on_arrays
- tests/batchsolving/test_chunk_axis_property.py::TestChunkAxisInRun::test_chunk_axis_property_after_run
- tests/batchsolving/test_chunk_axis_property.py::TestUpdateFromSolverChunkAxis::test_update_from_solver_does_not_change_chunk_axis

**Outcomes**:
- Files Modified:
  * tests/batchsolving/test_chunk_axis_property.py (142 lines - reviewed, verified complete)
- Functions/Methods Added/Modified:
  * No modifications made - test file was already complete from prior task groups
- Implementation Summary:
  Reviewed and verified the test file created by Task Groups 1, 2, and 3. All 8 expected
  tests are present and correctly implemented:
  1. test_chunk_axis_property_returns_default_run - verifies default "run" value
  2. test_chunk_axis_property_returns_consistent_value - verifies consistent array values
  3. test_chunk_axis_property_raises_on_inconsistency - tests ValueError on mismatch
  4. test_chunk_axis_setter_updates_both_arrays - verifies setter synchronization
  5. test_chunk_axis_setter_allows_valid_values - tests all valid values
  6. test_run_sets_chunk_axis_on_arrays - integration with run() method
  7. test_chunk_axis_property_after_run - property access after run completes
  8. test_update_from_solver_does_not_change_chunk_axis - cleanup verification
  Tests use correct fixtures (solverkernel, solverkernel_mutable, system, precision,
  driver_array) and follow project conventions. No type hints in tests.
- Issues Flagged: None

---

## Summary

### Total Task Groups: 4

### Dependency Chain:
```
Group 1 (Property/Setter) 
    ↓
Group 2 (run() setter call) 
    ↓
Group 3 (Cleanup redundant assignment)
    ↓
Group 4 (Test Suite)
```

### Tests to Create and Run:
1. `test_chunk_axis_property_returns_default_run` - Default value check
2. `test_chunk_axis_property_returns_consistent_value` - Consistency check
3. `test_chunk_axis_property_raises_on_inconsistency` - Error handling
4. `test_chunk_axis_setter_updates_both_arrays` - Setter synchronization
5. `test_chunk_axis_setter_allows_valid_values` - Valid values acceptance
6. `test_run_sets_chunk_axis_on_arrays` - Integration with run()
7. `test_chunk_axis_property_after_run` - Property access after run
8. `test_update_from_solver_does_not_change_chunk_axis` - Cleanup verification

### Estimated Complexity: Low-Medium
- Groups 1-3: Simple code modifications (~10 lines total)
- Group 4: Comprehensive test suite (~100 lines)
- No architectural changes beyond what's planned
- No new dependencies
