# Complete Array Coupling Refactor - Agent Plan

## Context

This task completes a partial refactoring in PR #511 (or the current branch copilot/refactor-array-coupling) that aimed to reduce coupling between array management and memory management in CuBIE's GPU memory subsystem.

### Changes Already Completed

The following changes to the codebase have already been made:

1. **ArrayRequest Changes** (`src/cubie/memory/array_requests.py`):
   - Removed `stride_order` field
   - Added `chunk_axis_index` field (Optional[int], default=2)
   - Field allows explicit specification of which array axis to chunk

2. **ArrayResponse Changes** (`src/cubie/memory/array_requests.py`):
   - Removed `axis_length` field
   - Removed `dangling_chunk_index` field (if it existed)
   - Now contains only: arr, chunks, chunk_length, chunked_shapes

3. **Memory Manager Updates** (`src/cubie/memory/mem_manager.py`):
   - Uses `request.chunk_axis_index` directly to determine chunking dimension
   - Function `get_chunk_axis_length()` extracts axis length from `request.shape[request.chunk_axis_index]`

4. **ManagedArray Updates** (`src/cubie/batchsolving/arrays/BaseArrayManager.py`):
   - Still has `stride_order` attribute (used for other purposes)
   - Computes `_chunk_axis_index` from `stride_order.index("run")` in `__attrs_post_init__`
   - Uses `_chunk_axis_index` for chunk slicing operations

### Outstanding Issues

The implementation appears logically complete, but cleanup is needed:

1. Test files that verify "field was removed" exist (meaningless post-refactor)
2. Comments reference the refactoring process rather than current behavior
3. Some test assertions are incorrect (checking for removed fields)
4. Need to verify no manual solver instantiation in tests (pattern violation)

## Implementation Tasks

### Task Group 1: Verify Logical Correctness

**Objective**: Ensure the refactored code behaves correctly

**Components to Check**:

1. **ArrayRequest (`src/cubie/memory/array_requests.py`)**:
   - Verify chunk_axis_index is used correctly throughout
   - Check default value (2) is appropriate for typical use cases
   - Ensure validator allows None and non-negative integers

2. **Memory Manager (`src/cubie/memory/mem_manager.py`)**:
   - Verify `get_chunk_axis_length()` correctly extracts from chunk_axis_index
   - Check `calculate_chunked_shapes()` uses chunk_axis_index properly
   - Ensure `is_request_chunkable()` checks chunk_axis_index appropriately
   - Verify no references to removed fields

3. **BaseArrayManager (`src/cubie/batchsolving/arrays/BaseArrayManager.py`)**:
   - Check `ManagedArray._chunk_axis_index` computation is correct
   - Verify `chunk_slice()` method uses _chunk_axis_index properly
   - Ensure `needs_chunked_transfer` property works correctly
   - Check interaction between stride_order and _chunk_axis_index

4. **Integration Points**:
   - Verify InputArrays and OutputArrays correctly pass chunk_axis_index
   - Check that chunked execution produces same results as unchunked
   - Ensure chunk boundaries are computed correctly

**Expected Behavior**:
- chunk_axis_index explicitly specifies which dimension to chunk
- Memory manager chunks arrays along specified axis
- ManagedArray derives _chunk_axis_index from stride_order for backward compatibility
- Chunked and unchunked runs produce identical results

**Red Flags**:
- Hardcoded axis indices (should use chunk_axis_index)
- References to stride_order in chunking logic
- Assumptions about "axis 0" convention
- Off-by-one errors in chunk boundary calculations

### Task Group 2: Clean Up Comments

**Objective**: Remove all references to the refactoring process from source code

**Files to Clean**:

Search patterns to find and remove/reword:
- Comments containing "new method" or "old method"
- Comments saying "was removed" or "has been removed"  
- Comments saying "no longer" or "eliminated"
- Comments saying "now computed" or "changed from"
- References to "refactoring" or "refactor"
- Comments explaining what code "used to do"

**Rewriting Guidelines**:
- Bad: "The stride_order field was removed; now using chunk_axis_index"
- Good: "Chunking axis specified by chunk_axis_index"

- Bad: "Now computed inline; eliminating the need for a buffer"
- Good: "Computed inline; no dedicated buffer required"

- Bad: "Memory manager now uses axis 0 by convention"
- Good: "Memory manager chunks along axis specified by chunk_axis_index"

**Files Likely to Need Updates**:
- `tests/memory/test_array_request_no_stride_order.py`
- `tests/memory/test_array_response_no_chunked_slices.py`
- Any files in `src/cubie/batchsolving/arrays/`
- Any files in `src/cubie/memory/`

### Task Group 3: Remove Change-Verification Tests

**Objective**: Delete tests that verify "field X was removed" or "method Y no longer exists"

**Tests to Remove Completely**:

1. **`tests/memory/test_array_request_no_stride_order.py`**:
   - Entire file verifies stride_order removal
   - File name itself indicates it's testing the change, not functionality
   - Tests like `test_array_request_no_stride_order_field` are meaningless post-refactor
   - **Action**: Delete entire file

2. **`tests/memory/test_array_response_no_chunked_slices.py`**:
   - Tests verify removed fields don't exist
   - Has INCORRECT assertions (checks for axis_length, dangling_chunk_length which should be removed)
   - **Action**: Delete entire file OR fix to test actual ArrayResponse functionality

**Criteria for Removal**:
- Test name mentions "no X" where X is something that was removed
- Test docstring says "verify X was removed" or "ensure X doesn't exist"
- Test only checks `not hasattr()` or `with pytest.raises(TypeError)` for removed fields
- Test doesn't verify any positive behavior

### Task Group 4: Fix Incorrect Test File

**File**: `tests/memory/test_array_response_no_chunked_slices.py`

**Issue**: Test checks for fields that should have been removed:
```python
assert hasattr(response, "axis_length")  # WRONG - should be removed
assert hasattr(response, "dangling_chunk_length")  # WRONG - should be removed
```

**Options**:
1. Delete the file (if it only verifies removal)
2. Fix to test actual ArrayResponse behavior:
   - Verify correct fields exist: arr, chunks, chunk_length, chunked_shapes
   - Test creation with various parameters
   - Test default values
   - Test that chunked_shapes is populated correctly

**Recommendation**: Delete the file. ArrayResponse behavior is tested elsewhere.

### Task Group 5: Review Test Patterns

**Objective**: Ensure no tests manually instantiate solvers or call solve/run

**Pattern to Check**:
- Direct instantiation: `Solver(system, ...)`, `BatchSolverKernel(...)`
- Direct execution: `solver.solve(...)`, `kernel.run(...)`

**Allowed Patterns**:
- Using fixtures: `def test_something(solved_solver):` where solved_solver is from conftest
- Testing runparams: `solver.kernel.runparams.dt`
- Testing results: `result.time_domain_array`

**Files to Review**:

Check these test files for manual instantiation:
- `tests/batchsolving/test_solver.py`
- `tests/batchsolving/test_SolverKernel.py`
- `tests/batchsolving/test_config_plumbing.py`
- `tests/batchsolving/test_runparams_integration.py`
- `tests/batchsolving/test_solveresult.py`
- `tests/batchsolving/arrays/test_basearraymanager.py`
- `tests/batchsolving/arrays/test_chunking.py`

**Exception**: `tests/_utils.py` and `tests/batchsolving/arrays/conftest.py` are allowed to instantiate solvers since they create fixtures.

**For Chunk-Related Tests**:
- Should be in `tests/batchsolving/arrays/test_chunking.py`
- Should use `chunked_solved_solver` and `unchunked_solved_solver` fixtures
- Should follow pattern established in existing tests in that file

**Actions**:
- If test manually instantiates AND tests chunking: Move to test_chunking.py, use fixtures
- If test manually instantiates AND doesn't test chunking: Determine if it should use fixtures or if it has valid reason for manual setup
- If test is in test_chunking.py but doesn't use fixtures: Refactor to use fixtures

### Task Group 6: Add Functional Tests (If Needed)

**Objective**: Ensure chunk_axis_index behavior is properly tested

**Check Existing Coverage**:
1. Does test_chunking.py verify chunked vs unchunked results match?
2. Are different chunk_axis_index values tested?
3. Is edge case behavior tested (chunk_axis_index=None, last chunk smaller, etc.)?

**If Coverage Gaps Exist, Add Tests**:

Location: `tests/batchsolving/arrays/test_chunking.py`

Use fixtures: `chunked_solved_solver`, `unchunked_solved_solver` from conftest

Example test structure:
```python
def test_chunk_axis_index_controls_chunking_dimension(
    chunked_solved_solver, unchunked_solved_solver
):
    """Verify chunking occurs along chunk_axis_index dimension."""
    chunked_solver, result_chunked = chunked_solved_solver
    unchunked_solver, result_normal = unchunked_solved_solver
    
    # Verify chunking occurred
    assert chunked_solver.chunks > 1
    assert unchunked_solver.chunks == 1
    
    # Verify results match
    np.testing.assert_allclose(
        result_chunked.time_domain_array,
        result_normal.time_domain_array,
        rtol=1e-5, atol=1e-7
    )
```

**Do Not Add**:
- Tests that verify removed fields don't exist
- Tests that check implementation details
- Tests that manually instantiate solvers

## Expected Outcomes

After completing these tasks:

1. **Source Code**:
   - No comments reference the refactoring process
   - All comments describe current behavior
   - Code uses chunk_axis_index consistently

2. **Tests**:
   - No tests verify "field was removed"
   - All chunk tests in test_chunking.py use proper fixtures
   - No manual solver instantiation (except in fixture definitions)
   - Tests verify functional behavior, not implementation details

3. **Behavior**:
   - Chunking works correctly along specified axis
   - Chunked execution produces same results as unchunked
   - Memory manager correctly sizes chunks based on chunk_axis_index

## Validation Criteria

Implementation is complete when:

1. ✅ All "change verification" tests removed
2. ✅ No comments reference refactoring or removed fields
3. ✅ All chunk tests use proper fixtures
4. ✅ Test suite passes
5. ✅ Chunked vs unchunked results match in tests
6. ✅ No manual solver instantiation in test files (except fixtures)

## Known Dependencies

- `src/cubie/memory/array_requests.py` - Core data structures
- `src/cubie/memory/mem_manager.py` - Chunking logic
- `src/cubie/batchsolving/arrays/BaseArrayManager.py` - ManagedArray and chunking
- `tests/batchsolving/arrays/conftest.py` - Test fixtures
- `tests/batchsolving/arrays/test_chunking.py` - Pattern to follow

## Edge Cases to Consider

1. **chunk_axis_index = None**: Should disable chunking
2. **Last chunk smaller**: Should handle remainder correctly
3. **Unchunkable arrays**: Should retain original shape
4. **Mixed chunkable/unchunkable**: Should handle both in same batch
5. **Different stride_orders**: ManagedArray should compute correct _chunk_axis_index
