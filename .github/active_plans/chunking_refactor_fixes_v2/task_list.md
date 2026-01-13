# Implementation Task List
# Feature: Chunking Refactor Test Fixes
# Plan Reference: .github/active_plans/chunking_refactor_fixes_v2/agent_plan.md

## Task Group 1: Fix ensure_nonzero_size Function
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/_utils.py (lines 607-648)
- File: tests/test_utils.py (lines 785-837)
- File: tests/outputhandling/test_output_sizes.py (lines 31-45, 352-363)

**Input Validation Required**:
- No additional input validation needed - function handles int, tuple, and other types

**Tasks**:
1. **Fix ensure_nonzero_size function logic**
   - File: src/cubie/_utils.py
   - Action: Modify
   - Details:
     Replace the function implementation at lines 607-648 with the following:
     ```python
     def ensure_nonzero_size(
         value: Union[int, Tuple[int, ...]],
     ) -> Union[int, Tuple[int, ...]]:
         """
         Replace zero-size shapes with minimal placeholder shapes for safe allocation.

         When creating CUDA local arrays, zero-sized dimensions cause errors. This
         function converts shapes containing any zero (or None) to minimal size-1
         placeholder shapes. If ANY dimension is zero, the entire shape becomes
         all 1s, creating a minimal memory footprint for inactive arrays.

         Parameters
         ----------
         value : Union[int, Tuple[int, ...]]
             Input value or tuple of values to process.

         Returns
         -------
         Union[int, Tuple[int, ...]]
             For integers: max(1, value).
             For tuples: if ANY element is 0 or None, returns tuple of all 1s
             with the same length. If no zeros/Nones, returns original tuple.
             Non-numeric values in tuples are treated as valid (non-zero).
             Other types are passed through unchanged.

         Examples
         --------
         >>> ensure_nonzero_size(0)
         1
         >>> ensure_nonzero_size(5)
         5
         >>> ensure_nonzero_size((0, 5))
         (1, 1)
         >>> ensure_nonzero_size((0, 2, 0))
         (1, 1, 1)
         >>> ensure_nonzero_size((2, 3, 4))
         (2, 3, 4)
         >>> ensure_nonzero_size((0, None))
         (1, 1)
         """
         if isinstance(value, int):
             return max(1, value)
         elif isinstance(value, tuple):
             # If ANY element is 0 or None, return all-ones tuple
             has_zero = any(
                 (isinstance(v, (int, float)) and v == 0) or v is None
                 for v in value
             )
             if has_zero:
                 return tuple(1 for _ in value)
             return value
         else:
             return value
     ```
   - Edge cases:
     - Integer 0 → 1
     - Integer non-zero → unchanged
     - Tuple with any 0 → all 1s tuple of same length
     - Tuple with None → all 1s tuple of same length
     - Tuple with no zeros → unchanged
     - Non-tuple/non-int → pass through unchanged
   - Integration: Used by ArraySizingClass.nonzero property in output_sizes.py

2. **Update test_utils.py tests for new ensure_nonzero_size behavior**
   - File: tests/test_utils.py
   - Action: Modify
   - Details:
     Update the TestEnsureNonzeroSize class to reflect the new behavior where ANY zero means ALL dimensions become 1. Replace lines 785-837 with:
     ```python
     class TestEnsureNonzeroSize:
         """Tests for ensure_nonzero_size utility function."""

         def test_single_zero_means_all_ones(self):
             """Test that single zero causes entire tuple to become all 1s."""
             result = ensure_nonzero_size((2, 0, 2))
             assert result == (1, 1, 1)

         def test_multiple_zeros_all_ones(self):
             """Test that multiple zeros cause entire tuple to become all 1s."""
             result = ensure_nonzero_size((0, 2, 0))
             assert result == (1, 1, 1)

         def test_all_zeros_replaced(self):
             """Test that all zeros are replaced with all 1s."""
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

         def test_first_element_zero_all_ones(self):
             """Test zero in first position causes all 1s."""
             result = ensure_nonzero_size((0, 3, 4))
             assert result == (1, 1, 1)

         def test_last_element_zero_all_ones(self):
             """Test zero in last position causes all 1s."""
             result = ensure_nonzero_size((2, 3, 0))
             assert result == (1, 1, 1)

         def test_string_tuple_passthrough(self):
             """Test that tuple of strings is passed through unchanged."""
             result = ensure_nonzero_size(("time", "variable", "run"))
             assert result == ("time", "variable", "run")

         def test_mixed_type_tuple_with_zero(self):
             """Test tuple with mixed numeric and non-numeric values with zero."""
             result = ensure_nonzero_size((0, "label", 2))
             assert result == (1, 1, 1)

         def test_mixed_type_tuple_no_zero(self):
             """Test tuple with mixed numeric and non-numeric values, no zero."""
             result = ensure_nonzero_size((3, "label", 2))
             assert result == (3, "label", 2)

         def test_none_treated_as_zero(self):
             """Test that None values in tuple cause all 1s."""
             result = ensure_nonzero_size((5, None, 3))
             assert result == (1, 1, 1)

         def test_two_element_tuple_with_zero(self):
             """Test two-element tuple with zero becomes (1, 1)."""
             result = ensure_nonzero_size((0, 5))
             assert result == (1, 1)

         def test_two_element_tuple_no_zero(self):
             """Test two-element tuple without zero is unchanged."""
             result = ensure_nonzero_size((3, 5))
             assert result == (3, 5)
     ```
   - Edge cases: All edge cases from the function are covered
   - Integration: These tests validate the core function behavior

**Tests to Create**:
- None (tests already exist, just need modification)

**Tests to Run**:
- tests/test_utils.py::TestEnsureNonzeroSize
- tests/outputhandling/test_output_sizes.py::TestNonzeroProperty::test_nonzero_property_tuple_values
- tests/outputhandling/test_output_sizes.py::TestBatchInputSizes::test_nonzero_functionality

**Outcomes**:
- Files Modified: 
  * src/cubie/_utils.py (53 lines changed - replaced lines 607-648)
  * tests/test_utils.py (72 lines changed - replaced lines 785-837)
- Functions/Methods Added/Modified:
  * ensure_nonzero_size() in _utils.py - changed to return all-1s tuple when ANY element is 0 or None
  * TestEnsureNonzeroSize class in test_utils.py - updated all tests to expect new behavior
- Implementation Summary:
  Changed the function logic so that when any element in a tuple is 0 or None, the entire tuple becomes all 1s (e.g., (0, 5) → (1, 1)). This creates minimal placeholder shapes for inactive CUDA local arrays. Updated the docstring, examples, and all related tests.
- Issues Flagged: None

---

## Task Group 2: Fix Memory Estimation in get_chunk_parameters
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/memory/mem_manager.py (lines 1240-1325)
- File: src/cubie/memory/mem_manager.py (lines 1456-1497) - get_portioned_request_size function
- File: tests/memory/test_memmgmt.py (entire file for context on test fixtures)

**Input Validation Required**:
- No additional input validation needed - parameters are already validated upstream

**Tasks**:
1. **Fix memory estimation error message formatting**
   - File: src/cubie/memory/mem_manager.py
   - Action: Modify
   - Details:
     The error message at line 1316-1320 is missing a space after the period. Fix formatting:
     ```python
     # Current (line 1314-1321):
     if max_chunk_size == 0:
         raise ValueError(
             "Can't fit a single run in GPU VRAM."
             f"Available memory: {available_memory}."
             f"Request size: {request_size}. Request "
             f"size that is 'chunkable': "
             f"{chunkable_size}."
         )
     
     # Fixed:
     if max_chunk_size == 0:
         raise ValueError(
             "Can't fit a single run in GPU VRAM. "
             f"Available memory: {available_memory}. "
             f"Request size: {request_size}. "
             f"Chunkable request size: {chunkable_size}."
         )
     ```
   - Edge cases: None - this is a formatting fix
   - Integration: Error message now properly formatted with spaces between sentences

2. **Verify memory calculation logic is correct**
   - File: src/cubie/memory/mem_manager.py
   - Action: Verify (no changes needed based on code review)
   - Details:
     After reviewing the code at lines 1274-1325, the memory calculation logic is correct:
     - `chunkable_size` and `unchunkable_size` are computed by `get_portioned_request_size`
     - `available_memory` is computed by `get_available_memory(stream_group)`
     - The chunk ratio calculation `chunkable_size / available_to_chunk` is correct
     - The `max_chunk_size` calculation `int(np_floor(axis_length / chunk_ratio))` is correct
     
     The "Can't fit a single run" error occurs when `max_chunk_size == 0`, which means
     even a single element on the chunk axis won't fit. This is correct behavior.
     
     **No code changes required for the calculation logic itself.**
   - Edge cases: N/A
   - Integration: N/A

**Tests to Create**:
- None

**Tests to Run**:
- tests/memory/test_memmgmt.py

**Outcomes**:
- Files Modified: 
  * src/cubie/memory/mem_manager.py (6 lines changed - fixed error message at lines 1314-1320)
- Functions/Methods Added/Modified:
  * get_chunk_parameters() in mem_manager.py - fixed error message formatting
- Implementation Summary:
  Fixed the ValueError message formatting when max_chunk_size == 0. Added proper spaces after periods and simplified the "Chunkable request size" phrasing for clarity.
- Issues Flagged: None

---

## Task Group 3: Fix Test Assertions for Updated ensure_nonzero_size
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/outputhandling/test_output_sizes.py (entire file)
- File: src/cubie/outputhandling/output_sizes.py (lines 1-60)

**Input Validation Required**:
- No additional input validation needed

**Tasks**:
1. **Update test_nonzero_property_tuple_values test**
   - File: tests/outputhandling/test_output_sizes.py
   - Action: Modify
   - Details:
     The test at lines 31-45 creates sizes with zeros and expects partial replacement.
     With the new behavior, any zero means ALL dimensions become 1.
     
     Current test (lines 31-45):
     ```python
     def test_nonzero_property_tuple_values(self):
         """Test that nonzero property converts zero tuple values to 1"""
         sizes = SingleRunOutputSizes(
             state=(0, 5),
             observables=(3, 0),
             state_summaries=(0, 0),
             observable_summaries=(2, 4),
         )
         nonzero_sizes = sizes.nonzero

         assert nonzero_sizes.state == (1, 1)
         assert nonzero_sizes.observables == (1, 1)
         assert nonzero_sizes.state_summaries == (1, 1)
         assert nonzero_sizes.observable_summaries == (2, 4)
     ```
     
     This test already has the correct expected values! The assertions expect:
     - `(0, 5)` → `(1, 1)` ✓
     - `(3, 0)` → `(1, 1)` ✓
     - `(0, 0)` → `(1, 1)` ✓
     - `(2, 4)` → `(2, 4)` (no zeros) ✓
     
     **No changes needed to this test - it already expects the correct behavior.**
   - Edge cases: N/A
   - Integration: Test validates the ArraySizingClass.nonzero property

2. **Update test_nonzero_functionality test in TestBatchInputSizes**
   - File: tests/outputhandling/test_output_sizes.py
   - Action: Modify
   - Details:
     The test at lines 352-363 tests BatchInputSizes.nonzero with a None value.
     With the new behavior, the None causes all dimensions to become 1.
     
     Current test (lines 352-363):
     ```python
     def test_nonzero_functionality(self):
         """Test that nonzero property works correctly"""
         sizes = BatchInputSizes(
             initial_values=(0, 0), parameters=(0, 0), driver_coefficients=(0, None)
         )
         nonzero_sizes = sizes.nonzero

         # All tuple values should have elements >= 1
         assert all(v >= 1 for v in nonzero_sizes.initial_values)
         assert all(v >= 1 for v in nonzero_sizes.parameters)
         assert nonzero_sizes.driver_coefficients[0] == 1
         assert nonzero_sizes.driver_coefficients[1] == 1
     ```
     
     With the new behavior:
     - `(0, 0)` → `(1, 1)` ✓ (all assertions pass)
     - `(0, None)` → `(1, 1)` ✓ (assertions check [0]==1 and [1]==1, which passes)
     
     **No changes needed to this test - it already expects the correct behavior.**
   - Edge cases: N/A
   - Integration: Test validates BatchInputSizes.nonzero behavior

**Tests to Create**:
- None

**Tests to Run**:
- tests/outputhandling/test_output_sizes.py::TestNonzeroProperty
- tests/outputhandling/test_output_sizes.py::TestBatchInputSizes::test_nonzero_functionality

**Outcomes**:

---

## Task Group 4: Verify and Document Allocation Flow
**Status**: [ ]
**Dependencies**: Task Groups 1-3

**Required Context**:
- File: src/cubie/batchsolving/arrays/BaseArrayManager.py (lines 85-100, 338-384)
- File: src/cubie/memory/mem_manager.py (lines 1053-1081, 1158-1238)
- File: tests/batchsolving/arrays/test_basearraymanager.py (entire file)

**Input Validation Required**:
- No additional input validation needed

**Tasks**:
1. **Review and verify allocation callback flow**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Verify (document findings)
   - Details:
     After code review, the allocation flow is correct:
     
     1. `BaseArrayManager.allocate()` (not shown in context) creates ArrayRequest and calls `queue_request()`
     2. `MemoryManager.queue_request()` (lines 1053-1081) adds request to `_queued_allocations`
     3. `MemoryManager.allocate_queue()` (lines 1158-1238) processes queue:
        - Computes `chunk_length, num_chunks` via `get_chunk_parameters()`
        - Computes `chunked_shapes` via `compute_chunked_shapes()`
        - Creates `chunked_requests` with modified shapes (line 1213-1215)
        - Calls `allocate_all(chunked_requests, ...)` which allocates with correct shapes
        - Calls `allocation_ready_hook` (which is `_on_allocation_complete`) with ArrayResponse
     4. `_on_allocation_complete()` (lines 338-384) stores arrays and chunked_shape
     
     The flow is correct. The issue described in the plan (device arrays with size 1)
     would only occur if:
     - `allocate_queue()` is not called after `queue_request()`
     - OR the default shape from `ManagedArray.__attrs_post_init__` is used
     
     The `ManagedArray.__attrs_post_init__` (lines 96-100) creates a default numpy
     array with shape from `self.shape` or `(1,) * len(stride_order)`. This is
     a HOST array (numpy), not a device array. The device array is attached via
     `_on_allocation_complete()`.
     
     **If tests are failing due to device arrays having size 1, the issue is likely
     that tests are not calling the full allocation flow (queue_request → allocate_queue).**
     
     **No production code changes needed - test fixtures may need updating.**
   - Edge cases: N/A
   - Integration: Documents the expected allocation flow

2. **Review needs_chunked_transfer property**
   - File: src/cubie/batchsolving/arrays/BaseArrayManager.py
   - Action: Verify (document findings)
   - Details:
     The `needs_chunked_transfer` property (lines 85-94):
     ```python
     @property
     def needs_chunked_transfer(self) -> bool:
         if self.chunked_shape is None:
             return False
         return self.shape != self.chunked_shape
     ```
     
     This returns `False` when `chunked_shape is None`, which happens when:
     - `allocate_queue()` was never called (so `_on_allocation_complete` never ran)
     - OR the array wasn't included in the allocation response
     
     This behavior is correct! If `chunked_shape` is None, it means chunking
     wasn't configured, so chunked transfers are not needed.
     
     **The issue is that tests may not be calling `allocate_queue()` to set up
     `chunked_shape`. This is a test fixture issue, not a production code issue.**
     
     **No production code changes needed.**
   - Edge cases: N/A
   - Integration: Documents the expected behavior

**Tests to Create**:
- None

**Tests to Run**:
- tests/batchsolving/arrays/test_basearraymanager.py

**Outcomes**:

---

## Task Group 5: Run Full Test Suite and Validate Fixes
**Status**: [ ]
**Dependencies**: Task Groups 1-4

**Required Context**:
- File: tests/outputhandling/test_output_sizes.py (entire file)
- File: tests/test_utils.py (lines 785-837)
- File: tests/memory/test_memmgmt.py (entire file)
- File: tests/batchsolving/arrays/test_basearraymanager.py (entire file)

**Input Validation Required**:
- No additional input validation needed

**Tasks**:
1. **Run Phase 1 tests (ensure_nonzero_size)**
   - Action: Run tests
   - Details: Run tests to verify the ensure_nonzero_size fix works correctly
   - Tests: 
     - `pytest tests/test_utils.py::TestEnsureNonzeroSize -v`
     - `pytest tests/outputhandling/test_output_sizes.py::TestNonzeroProperty -v`
     - `pytest tests/outputhandling/test_output_sizes.py::TestBatchInputSizes::test_nonzero_functionality -v`

2. **Run memory management tests**
   - Action: Run tests
   - Details: Verify memory estimation and allocation work correctly
   - Tests:
     - `pytest tests/memory/test_memmgmt.py -v`

3. **Run batch array manager tests**
   - Action: Run tests
   - Details: Verify allocation callbacks work correctly
   - Tests:
     - `pytest tests/batchsolving/arrays/test_basearraymanager.py -v`

**Tests to Create**:
- None

**Tests to Run**:
- tests/test_utils.py::TestEnsureNonzeroSize
- tests/outputhandling/test_output_sizes.py::TestNonzeroProperty
- tests/outputhandling/test_output_sizes.py::TestBatchInputSizes::test_nonzero_functionality
- tests/memory/test_memmgmt.py
- tests/batchsolving/arrays/test_basearraymanager.py

**Outcomes**:

---

## Summary

### Total Task Groups: 5

### Dependency Chain:
```
Task Group 1 (ensure_nonzero_size fix)
    ↓
Task Group 2 (memory estimation message fix)
    ↓
Task Group 3 (verify test assertions)
    ↓
Task Group 4 (verify allocation flow)
    ↓
Task Group 5 (run tests)
```

### Key Changes:
1. **src/cubie/_utils.py**: Fix `ensure_nonzero_size` to return all-1s tuple when ANY element is zero or None
2. **src/cubie/memory/mem_manager.py**: Fix error message formatting (add spaces between sentences)
3. **tests/test_utils.py**: Update `TestEnsureNonzeroSize` tests to match new behavior

### Tests Modified:
- `tests/test_utils.py::TestEnsureNonzeroSize` - Updated to expect all-1s behavior

### Tests to Run (all):
- `tests/test_utils.py::TestEnsureNonzeroSize`
- `tests/outputhandling/test_output_sizes.py::TestNonzeroProperty`
- `tests/outputhandling/test_output_sizes.py::TestBatchInputSizes::test_nonzero_functionality`
- `tests/memory/test_memmgmt.py`
- `tests/batchsolving/arrays/test_basearraymanager.py`

### Estimated Complexity: Low-Medium
- Most failures are due to the `ensure_nonzero_size` function behavior
- Test assertions in `test_output_sizes.py` already expect the correct behavior
- Memory estimation code is correct; only message formatting needs fixing
- Allocation flow is correct; failures may be due to test fixtures not calling full flow

### Notes on Other Phases in agent_plan.md:

Based on thorough code review, Phases 2-6 in the agent_plan.md may not require production code changes:

- **Phase 2 (Device Array Allocation)**: The allocation code is correct. Device arrays are allocated with chunked shapes when `allocate_queue()` is called. If tests fail, it's because test fixtures skip `allocate_queue()`.

- **Phase 3 (Memory Estimation)**: The calculation logic is correct. Only the error message formatting is fixed.

- **Phase 4 (needs_chunked_transfer)**: The property behavior is correct. Returns False when `chunked_shape` is None (not configured), which is expected.

- **Phase 5 (Test Fixtures)**: May require test-specific fixes if tests bypass the proper allocation flow.

- **Phase 6 (Individual Tests)**: Individual test issues would need specific investigation.

The primary fix is **Task Group 1** - fixing `ensure_nonzero_size` to implement the "any zero means all 1s" behavior.
