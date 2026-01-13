# Implementation Task List
# Feature: Memory Refactor Cleanup
# Plan Reference: .github/active_plans/memory_refactor_cleanup/agent_plan.md

## Task Group 1: Fix ensure_nonzero_size Bug
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/_utils.py (lines 607-644, specifically the ensure_nonzero_size function)

**Input Validation Required**:
- value: Already validated by type checking (int or tuple)
- No additional validation needed

**Tasks**:
1. **Fix ensure_nonzero_size to preserve non-zero dimensions**
   - File: src/cubie/_utils.py
   - Action: Modify
   - Details:
     ```python
     def ensure_nonzero_size(
         value: Union[int, Tuple[int, ...]],
     ) -> Union[int, Tuple[int, ...]]:
         """
         Replace zero-size shape with a one-size shape to ensure non-zero sizes.

         Parameters
         ----------
         value : Union[int, Tuple[int, ...]]
             Input value or tuple of values to process.

         Returns
         -------
         Union[int, Tuple[int, ...]]
             The input value with any zeros replaced by ones. For integers,
             returns max(1, value). For tuples, replaces each zero element
             with 1 while preserving non-zero elements.

         Examples
         --------
         >>> ensure_nonzero_size(0)
         1
         >>> ensure_nonzero_size(5)
         5
         >>> ensure_nonzero_size((0, 2, 0))
         (1, 2, 1)
         >>> ensure_nonzero_size((2, 3, 4))
         (2, 3, 4)
         >>> ensure_nonzero_size((2, 0, 2))
         (2, 1, 2)
         """
         if isinstance(value, int):
             return max(1, value)
         elif isinstance(value, tuple):
             # Replace only zero elements with 1, preserving non-zero elements
             return tuple(max(1, v) for v in value)
         else:
             return value
     ```
   - Edge cases:
     - Single zero in tuple: `(2, 0, 2)` → `(2, 1, 2)`
     - All zeros: `(0, 0, 0)` → `(1, 1, 1)`
     - No zeros: `(2, 3, 4)` → `(2, 3, 4)` (unchanged)
     - Single integer zero: `0` → `1`
   - Integration: Used by BatchOutputSizes.nonzero property for array sizing

**Tests to Create**:
- Test file: tests/test_utils.py
- Test function: test_ensure_nonzero_size_preserves_nonzero_dimensions
- Description: Verify that ensure_nonzero_size((2, 0, 2)) returns (2, 1, 2), not (1, 1, 1)

**Tests to Run**:
- tests/test_utils.py::TestEnsureNonzeroSize

**Outcomes**: 
- Files Modified:
  * src/cubie/_utils.py (4 lines changed - simplified tuple branch logic)
  * tests/test_utils.py (57 lines added - import + 8 test methods)
- Functions/Methods Added/Modified:
  * ensure_nonzero_size() in _utils.py - fixed to preserve non-zero dimensions
  * TestEnsureNonzeroSize class added to test_utils.py with 8 test methods
- Implementation Summary:
  Fixed the tuple branch of ensure_nonzero_size to use `tuple(max(1, v) for v in value)` 
  instead of checking `any(v == 0 for v in value)` and returning all ones. The fix 
  replaces only zero elements while preserving non-zero elements. Updated docstring 
  examples to reflect correct behavior.
- Issues Flagged: None

---

## Task Group 2: Remove Tests for Deleted MemoryManager Methods
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/memory/test_memmgmt.py (entire file - lines 1-1878)
- File: src/cubie/memory/mem_manager.py (lines 1-1537, for reference on what methods exist)

**Input Validation Required**:
- None - this is test cleanup only

**Tasks**:
1. **Remove test_get_chunks test**
   - File: tests/memory/test_memmgmt.py
   - Action: Delete
   - Details: Remove the `test_get_chunks` method (lines 515-530) from `TestMemoryManager` class
   - Edge cases: None
   - Integration: Method `get_chunks` no longer exists on MemoryManager

2. **Remove test_get_available_single test**
   - File: tests/memory/test_memmgmt.py
   - Action: Delete
   - Details: Remove the `test_get_available_single` method (lines 646-663) from `TestMemoryManager` class
   - Edge cases: None
   - Integration: Method `get_available_single` no longer exists

3. **Remove test_get_available_group test**
   - File: tests/memory/test_memmgmt.py
   - Action: Delete
   - Details: Remove the `test_get_available_group` method (lines 665-688) from `TestMemoryManager` class
   - Edge cases: None
   - Integration: Method renamed to `get_available_memory`

4. **Remove test_chunk_arrays test**
   - File: tests/memory/test_memmgmt.py
   - Action: Delete
   - Details: Remove the `test_chunk_arrays` method (lines 755-783) from `TestMemoryManager` class
   - Edge cases: None
   - Integration: Method `chunk_arrays` no longer exists

5. **Remove test_chunk_arrays_skips_missing_axis test**
   - File: tests/memory/test_memmgmt.py
   - Action: Delete
   - Details: Remove the `test_chunk_arrays_skips_missing_axis` method (lines 785-809) from `TestMemoryManager` class
   - Edge cases: None
   - Integration: Method `chunk_arrays` no longer exists

6. **Remove test_single_request test**
   - File: tests/memory/test_memmgmt.py
   - Action: Delete
   - Details: Remove the `test_single_request` method (lines 810-845) from `TestMemoryManager` class
   - Edge cases: None
   - Integration: Method `single_request` replaced by `queue_request` + `allocate_queue`

7. **Remove test_get_available_single_low_memory_warning test**
   - File: tests/memory/test_memmgmt.py
   - Action: Delete
   - Details: Remove the `test_get_available_single_low_memory_warning` method (lines 968-998) from class
   - Edge cases: None
   - Integration: Method `get_available_single` no longer exists

8. **Remove test_get_available_group_low_memory_warning test**
   - File: tests/memory/test_memmgmt.py
   - Action: Delete
   - Details: Remove the `test_get_available_group_low_memory_warning` method (lines 1000-1026) from class
   - Edge cases: None
   - Integration: Method `get_available_group` replaced by `get_available_memory`

9. **Remove test_get_total_request_size standalone test**
   - File: tests/memory/test_memmgmt.py
   - Action: Delete
   - Details: Remove the `test_get_total_request_size` function (lines 1115-1130)
   - Edge cases: None
   - Integration: Function `get_total_request_size` no longer exists

10. **Remove test_get_chunkable_request_size_* tests**
    - File: tests/memory/test_memmgmt.py
    - Action: Delete
    - Details: Remove these standalone tests:
      - `test_get_chunkable_request_size_excludes_unchunkable` (lines 1156-1180)
      - `test_get_chunkable_request_size_excludes_wrong_axis` (lines 1183-1205)
      - `test_get_chunkable_request_size_returns_zero_all_unchunkable` (lines 1208-1229)
    - Edge cases: None
    - Integration: Function `get_chunkable_request_size` replaced by `get_portioned_request_size`

11. **Remove test_compute_chunked_shapes_* tests with old signature**
    - File: tests/memory/test_memmgmt.py
    - Action: Delete
    - Details: Remove these tests that use old `num_chunks` parameter:
      - `test_compute_chunked_shapes_uses_floor_division` (lines 1233-1248)
      - `test_compute_chunked_shapes_preserves_unchunkable` (lines 1251-1277)
      - `test_compute_chunked_shapes_empty_requests` (lines 1279-1285)
      - `test_compute_chunked_shapes_single_chunk` (lines 1288-1302)
    - Edge cases: None
    - Integration: Method signature changed from `num_chunks` to `chunk_size`

12. **Remove test_chunk_arrays_uses_floor_division test**
    - File: tests/memory/test_memmgmt.py
    - Action: Delete
    - Details: Remove the `test_chunk_arrays_uses_floor_division` test (lines 1305-1318)
    - Edge cases: None
    - Integration: Method `chunk_arrays` no longer exists

13. **Remove test_chunk_arrays_min_one_element test**
    - File: tests/memory/test_memmgmt.py
    - Action: Delete
    - Details: Remove the `test_chunk_arrays_min_one_element` test (lines 1321-1334)
    - Edge cases: None
    - Integration: Method `chunk_arrays` no longer exists

14. **Remove test_single_request_populates_chunked_shapes test**
    - File: tests/memory/test_memmgmt.py
    - Action: Delete
    - Details: Remove the `test_single_request_populates_chunked_shapes` test (lines 1337-1371)
    - Edge cases: None
    - Integration: Method `single_request` no longer exists

15. **Remove test_allocate_queue_populates_chunked_shapes test**
    - File: tests/memory/test_memmgmt.py
    - Action: Delete
    - Details: Remove the `test_allocate_queue_populates_chunked_shapes` test (lines 1373-1428)
    - Edge cases: None
    - Integration: Test uses old `limit_type` parameter that no longer exists

16. **Remove test_chunk_calculation_5_runs_4_chunks test**
    - File: tests/memory/test_memmgmt.py
    - Action: Delete
    - Details: Remove the `test_chunk_calculation_5_runs_4_chunks` test (lines 1431-1469)
    - Edge cases: None
    - Integration: Uses removed `compute_chunked_shapes` with old signature

17. **Remove test_all_arrays_unchunkable_produces_one_chunk test**
    - File: tests/memory/test_memmgmt.py
    - Action: Delete
    - Details: Remove the `test_all_arrays_unchunkable_produces_one_chunk` test (lines 1472-1499)
    - Edge cases: None
    - Integration: Uses removed `get_chunkable_request_size` function

18. **Remove test_final_chunk_has_correct_indices test**
    - File: tests/memory/test_memmgmt.py
    - Action: Delete
    - Details: Remove the `test_final_chunk_has_correct_indices` test (lines 1502-1557)
    - Edge cases: None
    - Integration: Uses removed `compute_chunked_shapes` with old signature

19. **Remove test_uneven_chunk_division_7_runs_3_chunks test**
    - File: tests/memory/test_memmgmt.py
    - Action: Delete
    - Details: Remove the `test_uneven_chunk_division_7_runs_3_chunks` test (lines 1560-1589)
    - Edge cases: None
    - Integration: Uses removed `compute_chunked_shapes` with old signature

20. **Remove test_chunk_size_minimum_one_when_runs_less_than_chunks test**
    - File: tests/memory/test_memmgmt.py
    - Action: Delete
    - Details: Remove the `test_chunk_size_minimum_one_when_runs_less_than_chunks` test (lines 1591-1608)
    - Edge cases: None
    - Integration: Uses removed `compute_chunked_shapes` with old signature

21. **Fix test_allocate_queue_empty_queue parametrization**
    - File: tests/memory/test_memmgmt.py
    - Action: Modify
    - Details: Remove the `limit_type` and `chunk_axis` parameters from the parametrize decorator (lines 914-923). The test currently has:
      ```python
      @pytest.mark.parametrize(
          "fixed_mem_override, limit_type, chunk_axis",
          [
              [{"total": 512 * 1024**2}, "instance", "run"],
              [{"total": 512 * 1024**2}, "group", "run"],
              [{"total": 512 * 1024**2}, "instance", "time"],
              [{"total": 512 * 1024**2}, "group", "time"],
          ],
          indirect=["fixed_mem_override"],
      )
      def test_allocate_queue_empty_queue(
          self, registered_mgr, registered_instance
      ):
      ```
      Change to:
      ```python
      def test_allocate_queue_empty_queue(
          self, registered_mgr, registered_instance
      ):
      ```
    - Edge cases: None
    - Integration: `limit_type` parameter no longer exists in `allocate_queue`

**Tests to Create**:
- None - this task group removes obsolete tests

**Tests to Run**:
- tests/memory/test_memmgmt.py (run full test file to ensure no collection errors)

**Outcomes**: 
- Files Modified:
  * tests/memory/test_memmgmt.py (~680 lines removed, file reduced from 1608 to 930 lines)
- Functions/Methods Removed:
  * test_get_chunks (TestMemoryManager class method)
  * test_get_available_single (TestMemoryManager class method)
  * test_get_available_group (TestMemoryManager class method)
  * test_chunk_arrays (TestMemoryManager class method)
  * test_chunk_arrays_skips_missing_axis (TestMemoryManager class method)
  * test_single_request (TestMemoryManager class method)
  * test_get_available_single_low_memory_warning (TestMemoryManager class method)
  * test_get_available_group_low_memory_warning (TestMemoryManager class method)
  * test_get_total_request_size (standalone function)
  * test_get_chunkable_request_size_excludes_unchunkable (standalone function)
  * test_get_chunkable_request_size_excludes_wrong_axis (standalone function)
  * test_get_chunkable_request_size_returns_zero_all_unchunkable (standalone function)
  * test_compute_chunked_shapes_uses_floor_division (standalone function)
  * test_compute_chunked_shapes_preserves_unchunkable (standalone function)
  * test_compute_chunked_shapes_empty_requests (standalone function)
  * test_compute_chunked_shapes_single_chunk (standalone function)
  * test_chunk_arrays_uses_floor_division (standalone function)
  * test_chunk_arrays_min_one_element (standalone function)
  * test_single_request_populates_chunked_shapes (standalone function)
  * test_allocate_queue_populates_chunked_shapes (standalone function)
  * test_chunk_calculation_5_runs_4_chunks (standalone function)
  * test_all_arrays_unchunkable_produces_one_chunk (standalone function)
  * test_final_chunk_has_correct_indices (standalone function)
  * test_uneven_chunk_division_7_runs_3_chunks (standalone function)
  * test_chunk_size_minimum_one_when_runs_less_than_chunks (standalone function)
- Functions/Methods Modified:
  * test_allocate_queue_empty_queue - removed @pytest.mark.parametrize decorator with limit_type and chunk_axis parameters
- Implementation Summary:
  Removed 20+ tests that referenced deleted MemoryManager methods (get_chunks, get_available_single, get_available_group, chunk_arrays, single_request) and removed functions (get_total_request_size, get_chunkable_request_size, compute_chunked_shapes with num_chunks parameter). Fixed test_allocate_queue_empty_queue by removing the parametrize decorator that used the removed limit_type parameter. Also removed unused warnings import.
- Issues Flagged: None

---

## Task Group 3: Update BaseArrayManager Tests
**Status**: [x]
**Dependencies**: Task Group 1, Task Group 2

**Required Context**:
- File: tests/batchsolving/arrays/test_basearraymanager.py (entire file)
- File: src/cubie/memory/mem_manager.py (lines 1157-1237 for allocate_queue reference)

**Input Validation Required**:
- None - this is test cleanup only

**Tasks**:
1. **Remove TestChunkArraysSkipsMissingAxis class**
   - File: tests/batchsolving/arrays/test_basearraymanager.py
   - Action: Delete
   - Details: Remove the entire `TestChunkArraysSkipsMissingAxis` class (lines 1414-1485) which tests the removed `chunk_arrays` method
   - Edge cases: None
   - Integration: Method `chunk_arrays` no longer exists on MemoryManager

2. **Remove test_unchunkable_array_chunked_shape_unchanged test**
   - File: tests/batchsolving/arrays/test_basearraymanager.py
   - Action: Delete
   - Details: Remove the `test_unchunkable_array_chunked_shape_unchanged` test (lines 1842-1877) which uses removed `compute_chunked_shapes` with old signature
   - Edge cases: None
   - Integration: Test uses `num_chunks` parameter which was replaced by `chunk_size`

**Tests to Create**:
- None - this task group removes obsolete tests

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py (run full test file to ensure no collection errors)

**Outcomes**: 
- Files Modified:
  * tests/batchsolving/arrays/test_basearraymanager.py (~108 lines removed)
- Functions/Methods Removed:
  * TestChunkArraysSkipsMissingAxis class (with 3 test methods: test_chunk_arrays_skips_2d_array_when_chunking_time, test_chunk_arrays_skips_1d_status_codes, test_chunk_arrays_handles_run_axis_correctly)
  * test_unchunkable_array_chunked_shape_unchanged function
  * MockMemoryManager class (only used by removed tests)
  * low_memory_manager fixture (only used by removed tests)
- Implementation Summary:
  Removed TestChunkArraysSkipsMissingAxis class which tested the removed `chunk_arrays` method, and removed test_unchunkable_array_chunked_shape_unchanged which used the old `compute_chunked_shapes` signature with `num_chunks` parameter (now replaced by `chunk_size`). Also removed the MockMemoryManager class and low_memory_manager fixture which were only used by the removed test class.
- Issues Flagged: None

---

## Task Group 4: Add New Tests for Updated ensure_nonzero_size
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/test_utils.py (if exists, or create new)
- File: src/cubie/_utils.py (lines 607-644, the ensure_nonzero_size function)

**Input Validation Required**:
- None - this is test creation

**Tasks**:
1. **Create test for ensure_nonzero_size preserving non-zero dimensions**
   - File: tests/test_utils.py
   - Action: Create or Modify
   - Details:
     ```python
     import pytest
     from cubie._utils import ensure_nonzero_size


     class TestEnsureNonzeroSize:
         """Tests for ensure_nonzero_size utility function."""

         def test_single_zero_replaced(self):
             """Test that single zero in middle is replaced."""
             result = ensure_nonzero_size((2, 0, 2))
             assert result == (2, 1, 2)

         def test_multiple_zeros_replaced(self):
             """Test that multiple zeros are each replaced."""
             result = ensure_nonzero_size((0, 2, 0))
             assert result == (1, 2, 1)

         def test_all_zeros_replaced(self):
             """Test that all zeros are replaced."""
             result = ensure_nonzero_size((0, 0, 0))
             assert result == (1, 1, 1)

         def test_no_zeros_unchanged(self):
             """Test that tuple with no zeros is unchanged."""
             result = ensure_nonzero_size((2, 3, 4))
             assert result == (2, 3, 4)

         def test_integer_zero(self):
             """Test that integer zero becomes 1."""
             result = ensure_nonzero_size(0)
             assert result == 1

         def test_integer_nonzero(self):
             """Test that nonzero integer is unchanged."""
             result = ensure_nonzero_size(5)
             assert result == 5

         def test_first_element_zero(self):
             """Test zero in first position is replaced."""
             result = ensure_nonzero_size((0, 3, 4))
             assert result == (1, 3, 4)

         def test_last_element_zero(self):
             """Test zero in last position is replaced."""
             result = ensure_nonzero_size((2, 3, 0))
             assert result == (2, 3, 1)
     ```
   - Edge cases: All edge cases covered by individual test methods
   - Integration: Validates fix for shape mismatch bug in solver

**Tests to Create**:
- Test file: tests/test_utils.py
- Test class: TestEnsureNonzeroSize
- Test functions:
  - test_single_zero_replaced
  - test_multiple_zeros_replaced
  - test_all_zeros_replaced
  - test_no_zeros_unchanged
  - test_integer_zero
  - test_integer_nonzero
  - test_first_element_zero
  - test_last_element_zero

**Tests to Run**:
- tests/test_utils.py::TestEnsureNonzeroSize

**Outcomes**: 
- Files Modified:
  * None - tests were already created as part of Task Group 1
- Functions/Methods Added/Modified:
  * TestEnsureNonzeroSize class already exists in tests/test_utils.py
- Implementation Summary:
  This task group was completed as part of Task Group 1. The TestEnsureNonzeroSize
  class with 8 test methods was added to tests/test_utils.py during the bug fix
  implementation. All specified tests (test_single_zero_replaced, test_multiple_zeros_replaced,
  test_all_zeros_replaced, test_no_zeros_unchanged, test_integer_zero, test_integer_nonzero,
  test_first_element_zero, test_last_element_zero) are present and match the specification.
- Issues Flagged: None

---

## Task Group 5: Verify Solver Integration
**Status**: [x]
**Dependencies**: Task Groups 1-4

**Required Context**:
- File: tests/batchsolving/test_solver.py (lines 981-996 for test_solver_solve_with_save_variables)
- File: src/cubie/outputhandling/output_sizes.py (for BatchOutputSizes.nonzero property)

**Input Validation Required**:
- None - verification only

**Tasks**:
1. **Verify test_solver_solve_with_save_variables passes**
   - File: tests/batchsolving/test_solver.py
   - Action: Verify (no modification needed)
   - Details: After fixing `ensure_nonzero_size`, this test should pass without shape mismatch errors. The test:
     - Saves only a subset of states (`save_variables=state_names[:1]`)
     - This causes observable dimensions to be 0
     - The `ensure_nonzero_size` fix ensures host and device arrays have matching shapes
   - Edge cases: None
   - Integration: End-to-end validation of the bug fix

**Tests to Create**:
- None - verification uses existing test

**Tests to Run**:
- tests/batchsolving/test_solver.py::test_solver_solve_with_save_variables

**Outcomes**: 
- Files Modified: None (verification only)
- Verification Summary:
  * Confirmed ensure_nonzero_size fix in src/cubie/_utils.py correctly preserves non-zero 
    tuple elements: `(2, 0, 2)` → `(2, 1, 2)` instead of `(1, 1, 1)`
  * Verified ArraySizingClass.nonzero property in output_sizes.py uses ensure_nonzero_size
    to process each tuple field (line 56)
  * Verified BatchOutputArrays.from_solver (line 357) and update_from_solver (line 386) 
    both call `.nonzero` on BatchOutputSizes
  * Integration flow: BatchOutputSizes.from_solver().nonzero → ensure_nonzero_size() → 
    correct array shapes for host/device allocation
  * All key files syntactically correct:
    - src/cubie/_utils.py (ensure_nonzero_size function)
    - src/cubie/outputhandling/output_sizes.py (ArraySizingClass.nonzero property)
    - src/cubie/batchsolving/arrays/BatchOutputArrays.py (OutputArrays class)
    - tests/memory/test_memmgmt.py (obsolete tests removed)
    - tests/batchsolving/arrays/test_basearraymanager.py (obsolete tests removed)
    - tests/test_utils.py (new TestEnsureNonzeroSize tests added)
  * No remaining code smells or integration issues identified
  * Chunking logic flows correctly: MemoryManager.allocate_queue() → 
    compute_chunked_shapes() → BaseArrayManager._on_allocation_complete()
- Issues Flagged: None

---

# Summary

## Total Task Groups: 5

## Dependency Chain Overview:
```
Task Group 1 (Fix Bug) ─┬─> Task Group 4 (Add Tests for Bug Fix)
                        │
                        └─> Task Group 2 (Remove Obsolete Memory Tests)
                                    │
                                    └─> Task Group 3 (Remove Obsolete Array Manager Tests)
                                                │
                                                └─> Task Group 5 (Verify Solver Integration)
```

## Tests to be Created:
- tests/test_utils.py::TestEnsureNonzeroSize (8 test methods)

## Tests to be Run:
- tests/test_utils.py::TestEnsureNonzeroSize
- tests/memory/test_memmgmt.py (full file for collection verification)
- tests/batchsolving/arrays/test_basearraymanager.py (full file for collection verification)
- tests/batchsolving/test_solver.py::test_solver_solve_with_save_variables

## Estimated Complexity:
- **Task Group 1**: Low - 1 line change in function logic
- **Task Group 2**: Medium - Multiple test deletions, careful not to break other tests
- **Task Group 3**: Low - 2 test class/function deletions
- **Task Group 4**: Low - Create simple unit tests
- **Task Group 5**: Low - Run existing test to verify integration
