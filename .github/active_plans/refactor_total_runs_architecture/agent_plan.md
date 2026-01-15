# Refactor total_runs Architecture - Agent Implementation Plan

## Overview

This plan refactors the num_runs/total_runs tracking architecture to simplify the codebase based on code review feedback. The refactoring establishes BaseArrayManager.num_runs as an internal attribute that is set during update() and propagated to ManagedArray instances, replacing the previous pattern of extracting run counts from sizing objects on-demand.

## Component Changes

### 1. BaseArrayManager - Add num_runs Attribute and set_array_runs Method

**File:** `src/cubie/batchsolving/arrays/BaseArrayManager.py`

**Changes:**

1. Add `num_runs` attribute to `__init__`:
   - Type: `Optional[int]` 
   - Default: `None`
   - Validator: `opt_getype_validator(int, 1)`
   - This tracks the total number of runs for the current batch

2. Remove `_get_total_runs()` method (lines 873-893):
   - This helper is replaced by direct attribute access
   - No longer needed with num_runs as an attribute

3. Add `set_array_runs(num_runs: int)` method:
   - Purpose: Update num_runs in all ManagedArray instances
   - Behavior: Iterate through both host and device containers
   - For each ManagedArray, set an internal attribute tracking its num_runs
   - Called from `update()` or `update_from_solver()` in subclasses
   - Validation: num_runs must be int >= 1 (minimal validation per review feedback)

4. Modify `allocate()` method (lines 895-938):
   - Remove call to `_get_total_runs()` (line 912)
   - Change logic at lines 922-926:
     ```python
     # OLD:
     if host_array_object.is_chunked:
         total_runs = total_runs_value
     else:
         total_runs = None
     
     # NEW:
     if host_array_object.is_chunked:
         total_runs = self.num_runs
     else:
         total_runs = 1  # Not None - always provide total_runs
     ```
   - Update ArrayRequest creation to always provide total_runs

**Expected Behavior:**
- BaseArrayManager tracks num_runs internally
- set_array_runs() propagates num_runs to all managed arrays
- allocate() uses self.num_runs instead of extracting from sizes
- total_runs is always provided (never None)

**Integration:**
- Subclasses (InputArrays, OutputArrays) must call set_array_runs() in their update methods
- ManagedArray must have a way to store the num_runs value (may need attribute added)

### 2. InputArrays and OutputArrays - Set num_runs During Update

**File:** `src/cubie/batchsolving/arrays/BatchInputArrays.py`

**Changes in update_from_solver() method (around line 250):**

1. Extract num_runs from sizes:
   ```python
   # After setting self._sizes
   if self._sizes is not None:
       # Extract num_runs from initial_values shape (second element)
       num_runs = self._sizes.initial_values[1]
       self.num_runs = num_runs
       self.set_array_runs(num_runs)
   ```

**File:** `src/cubie/batchsolving/arrays/BatchOutputArrays.py`

**Changes in update_from_solver() method (around line 296):**

1. Extract num_runs from sizes:
   ```python
   # After setting self._sizes or computing new sizes
   if self._sizes is not None:
       # Extract num_runs from state shape (third element)
       num_runs = self._sizes.state[2]
       self.num_runs = num_runs
       self.set_array_runs(num_runs)
   ```

**Expected Behavior:**
- update_from_solver() extracts num_runs from sizing metadata
- Sets self.num_runs attribute
- Calls set_array_runs() to propagate to all ManagedArray instances
- This happens before allocate() is called

### 3. ArrayRequest - Change total_runs to Always int >= 1

**File:** `src/cubie/memory/array_requests.py`

**Changes:**

1. Line 38: Change default from None to 1:
   ```python
   # OLD:
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

2. Update docstring (lines 36-39):
   - Remove language about None meaning "not intended for chunking"
   - State that default is 1 for unchunkable arrays
   - Clarify that total_runs represents the total run dimension

3. Update type annotation (line 78):
   ```python
   # OLD: total_runs: Optional[int]
   # NEW: total_runs: int
   ```

**Expected Behavior:**
- ArrayRequest always has a valid total_runs value
- Default of 1 is used for unchunkable arrays (like driver_coefficients)
- Validator enforces >= 1 constraint
- No None checks needed in memory manager

### 4. MemoryManager - Simplify Run Extraction

**File:** `src/cubie/memory/mem_manager.py`

**Changes:**

1. Remove `_extract_num_runs()` method entirely (lines 1170-1224):
   - Delete the method and all its validation logic
   - This complex extraction is replaced by simple first-request access

2. Modify `allocate_queue()` method (around line 1260):
   ```python
   # OLD:
   # Extract num_runs from ArrayRequest total_runs fields
   num_runs = self._extract_num_runs(queued_requests)
   
   # NEW:
   # Get total_runs from first request (all are the same by construction)
   first_instance_requests = next(iter(queued_requests.values()))
   first_request = next(iter(first_instance_requests.values()))
   num_runs = first_request.total_runs
   ```

3. Remove explanatory comment (lines 1247-1250):
   - Delete the "note" about where num_runs comes from
   - Just use the value directly without explanation

**Expected Behavior:**
- allocate_queue() gets num_runs from first request
- No validation of consistency (guaranteed by array managers)
- Simpler, more direct code

### 5. Remove runs Properties from Sizing Classes

**File:** `src/cubie/outputhandling/output_sizes.py`

**Changes:**

1. Remove `BatchInputSizes.runs` property (lines 257-270):
   - Delete entire property method
   - Remove associated docstring

2. Remove `BatchOutputSizes.runs` property (lines 381-395):
   - Delete entire property method
   - Remove associated docstring

**Expected Behavior:**
- Sizing classes only describe shapes, don't provide convenience accessors
- Code that needs num_runs gets it from array managers instead
- No fragile index-based access to shape tuples

### 6. ManagedArray - Store num_runs (if needed)

**File:** `src/cubie/batchsolving/arrays/BaseArrayManager.py`

**Investigation Required:**
- Check if ManagedArray needs an attribute to store num_runs
- If set_array_runs() needs to store the value, add optional attribute
- May not be needed if only used for ArrayRequest creation

**Potential Change:**
```python
@define(slots=False)
class ManagedArray:
    # ... existing fields ...
    
    _num_runs: Optional[int] = field(
        default=None,
        validator=opt_getype_validator(int, 1),
    )
```

## Test Changes

### Tests to Remove

1. **tests/batchsolving/arrays/test_basearraymanager.py**
   - Lines 2033-2090: `TestGetTotalRuns` class
     - test_get_total_runs_returns_none_when_sizes_none
     - test_get_total_runs_returns_runs_from_sizes
   - Reason: _get_total_runs() method removed

2. **tests/memory/test_array_requests.py**
   - Lines 167-181: test_array_request_total_runs_defaults_to_none
   - Reason: total_runs now defaults to 1, not None

3. **tests/memory/test_memmgmt.py**
   - Lines 1188-1300: `TestExtractNumRuns` class (all tests)
     - test_extract_num_runs_finds_single_value
     - test_extract_num_runs_ignores_none_values
     - test_extract_num_runs_validates_consistency
     - test_extract_num_runs_raises_when_no_total_runs
   - Reason: _extract_num_runs() method removed

4. **tests/outputhandling/test_output_sizes.py**
   - Lines 371-384: test_batch_input_sizes_exposes_runs
   - Lines 447-463: test_batch_output_sizes_exposes_runs
   - Reason: runs properties removed from sizing classes

### Tests to Fix

#### 1. chunk_slice API Change
**Test:** Multiple tests in test_basearraymanager.py that call chunk_slice
**Issue:** API changed to pass host slice instead of full array
**Fix:** 
- Identify tests calling chunk_slice
- Update to pass the slice object returned from ManagedArray
- Don't test with full host array

#### 2. Negative chunk_index Validation
**Test:** test_chunk_slice_validates_chunk_index (line 1771)
**Current:** Tests that negative indices raise ValueError
**Issue:** Negative indices are valid in Python
**Fix:**
```python
# OLD:
with pytest.raises(ValueError, match="chunk_index -1 out of range"):
    managed.chunk_slice(-1)

# NEW:
# Test out of range (too large)
with pytest.raises(ValueError, match="chunk_index 4 out of range"):
    managed.chunk_slice(4)

# Test out of range (too negative)
with pytest.raises(ValueError, match="chunk_index -5 out of range"):
    managed.chunk_slice(-5)  # Only 4 chunks, so -5 is out of range
```

#### 3. Remove Axis 0 Chunking Tests
**Test:** test_chunk_slice_different_axis_indices (line 1868)
**Issue:** This tests chunking on axis 0, which is incorrect
**Fix:** 
- Remove the entire test
- Verify no other tests assume axis 0 chunking
- Remove any documentation suggesting axis 0 chunking

#### 4. Read-Only queue_request
**Test:** Any test trying to modify mgr.queue_request
**Issue:** queue_request is read-only
**Fix:**
- Find tests that assign to mgr.queue_request
- Use mocking pattern instead
- Or test through public API that doesn't modify read-only attributes

#### 5. Remove Skip Conditions
**Tests:** Search for pytest.skip, skipif, xfail in test files
**Issue:** Test parameters are deterministic, skip conditions are useless
**Fix:**
- Remove conditional skip logic
- If test must force chunking, move to test_chunking.py
- Use chunked_solved_solver fixture for chunking tests

#### 6. Fix chunk_axis_index Assertion
**Test:** test in test_chunking.py line 634
**Current:** `assert output_manager.device.time_domain_array._chunk_axis_index == 2`
**Issue:** time_domain_array doesn't exist, should test 'state' instead
**Fix:**
```python
# OLD:
assert output_manager.device.time_domain_array._chunk_axis_index == 2

# NEW:
assert output_manager.device.state._chunk_axis_index == 2
```

#### 7. Fix initial_values Assertion
**Test:** Any test checking initial_values._chunk_axis_index == 2
**Issue:** initial_values is 2D (variable, run), so run axis is index 1, not 2
**Fix:**
```python
# OLD:
assert input_manager.device.initial_values._chunk_axis_index == 2

# NEW:
assert input_manager.device.initial_values._chunk_axis_index == 1
```

#### 8. Use Correct Attribute Names
**Tests:** Multiple tests using wrong attribute names
**Fixes:**
- Replace `time_domain_array` with `state`
- Replace `num_params` with `num_parameters`

**File:** tests/batchsolving/test_runparams_integration.py
```python
# Lines 58, 99, 167
# OLD:
params = np.random.rand(integration_system.num_params, num_runs)

# NEW:
params = np.random.rand(integration_system.num_parameters, num_runs)
```

## Implementation Order

1. **Phase 1 - Core Attribute Changes:**
   - Add num_runs attribute to BaseArrayManager
   - Add set_array_runs() method to BaseArrayManager
   - Update InputArrays.update_from_solver() to set num_runs
   - Update OutputArrays.update_from_solver() to set num_runs

2. **Phase 2 - ArrayRequest Changes:**
   - Change ArrayRequest.total_runs type to int with default=1
   - Update docstrings and type annotations

3. **Phase 3 - Remove Old Methods:**
   - Remove BaseArrayManager._get_total_runs()
   - Update BaseArrayManager.allocate() to use self.num_runs
   - Remove MemoryManager._extract_num_runs()
   - Simplify MemoryManager.allocate_queue()

4. **Phase 4 - Remove Sizing Properties:**
   - Remove BatchInputSizes.runs property
   - Remove BatchOutputSizes.runs property

5. **Phase 5 - Test Cleanup:**
   - Remove tests for deleted methods
   - Fix tests for API changes
   - Update attribute names in tests
   - Remove skip conditions

## Edge Cases to Consider

1. **num_runs is None:**
   - When is num_runs not set?
   - Should set_array_runs() handle None?
   - Default behavior if update_from_solver() not called?

2. **Driver coefficients:**
   - is_chunked=False for driver_coefficients
   - Should still get total_runs=1 in ArrayRequest
   - Verify chunking logic handles this correctly

3. **Empty batches:**
   - What if num_runs=1?
   - Should still work correctly with no chunking

4. **Update called multiple times:**
   - num_runs might change between solver runs
   - Ensure set_array_runs() updates all arrays correctly

## Validation Strategy

1. **Unit tests** for new methods:
   - Test set_array_runs() propagates correctly
   - Test num_runs attribute is set during update

2. **Integration tests** for chunking:
   - Verify chunked and non-chunked allocation works
   - Verify total_runs=1 for unchunkable arrays
   - Verify total_runs=N for chunkable arrays

3. **End-to-end tests:**
   - Run existing solver tests
   - Verify chunking still works correctly
   - Verify multi-run batches work

## Success Criteria

1. All unit tests pass with new architecture
2. All integration tests pass
3. No references to _get_total_runs() remain
4. No references to _extract_num_runs() remain
5. ArrayRequest.total_runs is never None
6. BaseArrayManager.num_runs is set correctly
7. ManagedArray instances have num_runs when needed
8. Code is simpler and easier to understand

## Dependencies and Assumptions

**Dependencies:**
- BaseArrayManager must be updated before subclasses
- ArrayRequest changes must happen before MemoryManager changes
- Tests should be fixed after implementation changes

**Assumptions:**
- All requests from same manager have same total_runs
- update_from_solver() is always called before allocate()
- Sizing metadata always contains run dimension information
- num_runs >= 1 always (no empty batches)

**Critical Integration Points:**
- BatchSolverKernel.run() → InputArrays.update() → BaseArrayManager.update_from_solver()
- BatchSolverKernel.run() → OutputArrays.update() → BaseArrayManager.update_from_solver()
- BaseArrayManager.allocate() → MemoryManager.queue_request()
- MemoryManager.allocate_queue() → ArrayRequest.total_runs
