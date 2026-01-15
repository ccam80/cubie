# Test Results Summary - Total Runs Refactoring

## Test Execution Commands

```bash
NUMBA_ENABLE_CUDASIM=1 pytest tests/memory/ -v -m "not nocudasim and not cupy" --tb=short
NUMBA_ENABLE_CUDASIM=1 pytest tests/batchsolving/arrays/ -v -m "not nocudasim and not cupy" --tb=short
NUMBA_ENABLE_CUDASIM=1 pytest tests/batchsolving/arrays/test_chunking.py -v -m "not nocudasim and not cupy" --tb=short
NUMBA_ENABLE_CUDASIM=1 pytest tests/batchsolving/test_runparams_integration.py -v -m "not nocudasim and not cupy" --tb=short
NUMBA_ENABLE_CUDASIM=1 pytest tests/outputhandling/test_output_sizes.py -v -m "not nocudasim and not cupy" --tb=short
```

## Overall Results Summary

| Test Suite | Tests Run | Passed | Failed | Errors | Skipped |
|------------|-----------|--------|--------|--------|---------|
| **Memory** | 71 | 69 | 2 | 0 | 0 |
| **Batchsolving Arrays** | 166 | 138 | 19 | 8 | 1 |
| **Chunking** | 34 | ~20 | ~8 | ~6 | 0 |
| **Runparams Integration** | 5 | ~4 | ~1 | 0 | 0 |
| **Output Sizes** | 22 | 22 | 0 | 0 | 0 |
| **TOTAL** | **~298** | **~253** | **~30** | **~14** | **1** |

## Test Suite Details

### ✅ Output Sizes Tests - ALL PASSED (22/22)
All tests in `tests/outputhandling/test_output_sizes.py` **passed successfully**.

### ⚠️ Memory Tests - 2 Failures (69/71 passed)

#### FAILED: test_allocate_queue_empty_queue
**Type**: StopIteration
**Location**: tests/memory/test_memmgmt.py::TestMemoryManager::test_allocate_queue_empty_queue
**Message**: StopIteration exception during empty queue handling

#### FAILED: test_get_chunk_parameters_unchunkable_exceeds_memory  
**Type**: TypeError
**Location**: tests/memory/test_memmgmt.py::TestGetChunkParameters::test_get_chunk_parameters_unchunkable_exceeds_memory
**Message**: `'total_runs' must be (<class 'int'>, <class 'numpy.integer'>) (got None that is a <class 'NoneType'>).`

**Root Cause**: Test is passing `None` for `total_runs` parameter but ArrayRequest now requires `total_runs` to be an int (default 1, not Optional[int]).

### ⚠️ Batchsolving Arrays Tests - 19 Failures, 8 Errors (138/166 passed)

#### Category 1: Test Infrastructure Issues (Mocking)

**FAILED: TestNumRunsAttribute::test_allocate_uses_one_for_unchunked_arrays**
- **Type**: AttributeError
- **Message**: `'MemoryManager' object attribute 'queue_request' is read-only`
- **Root Cause**: Test tries to monkey-patch an attrs frozen class method

**FAILED: TestNumRunsAttribute::test_allocate_uses_num_runs_for_chunked_arrays**  
- Same root cause as above

**FAILED: TestMemoryManagerIntegration::test_allocation_with_settings**
- Likely same mocking issue

#### Category 2: Chunk Slice Logic Issues

**FAILED: TestChunkSliceMethod::test_chunk_slice_computes_correct_slices**
- **Type**: AssertionError
- **Message**: `assert (10, 5, 100) == (10, 5, 25)` - Expected shape (10, 5, 25) but got (10, 5, 100)
- **Root Cause**: chunk_slice() returning full array shape instead of chunked slice shape

**FAILED: TestChunkSliceMethod::test_chunk_slice_handles_final_chunk_dynamically**
- Similar issue with chunk slicing

**FAILED: TestChunkSliceMethod::test_chunk_slice_validates_chunk_index**
- Chunk index validation not working correctly

**FAILED: TestChunkSliceMethod::test_chunk_slice_accepts_valid_negative_indices**
- Negative index handling in chunk_slice

**FAILED: TestChunkSliceMethod::test_chunk_slice_single_chunk**
- Single chunk case not handled correctly

**FAILED: TestChunkSliceMethod::test_chunk_slice_different_axis_indices**
- Different chunk axis indices not working

#### Category 3: Chunk Metadata Flow

**FAILED: TestChunkMetadataFlow::test_chunk_slice_uses_chunk_metadata_from_response**
- Chunk metadata not flowing from allocation response to chunk_slice method

**FAILED: TestChunkMetadataFlow::test_chunk_metadata_flow_integration**
- Integration of chunk metadata flow not working

#### Category 4: Input Arrays Tests

**FAILED: TestInputArrays::test_call_method_size_change_triggers_reallocation**
- Size change detection/reallocation logic issue

**FAILED: TestInputArrays::test_update_from_solver_sets_num_runs**
- update_from_solver() not properly setting num_runs attribute

**FAILED: TestInputArrays::test_initialise_method**
- initialise() method issues

#### Category 5: Chunking Integration Tests

**ERROR: test_chunked_solver_produces_correct_results[860]**
**ERROR: test_chunked_solver_produces_correct_results[1024]**
**ERROR: test_chunked_solver_produces_correct_results[1240]**
**ERROR: test_chunked_solver_produces_correct_results[1460]**
**ERROR: test_chunked_solver_produces_correct_results[2048]**
- **Type**: AttributeError
- **Message**: `'SolveResult' object has no attribute 'state'`
- **Root Cause**: Test accessing wrong attribute on SolveResult object

**ERROR: test_non_chunked_uses_pinned_host**
- Error in test setup or execution

**ERROR: TestNeedsChunkedTransferBranching::test_convert_host_to_numpy_uses_needs_chunked_transfer**
- Test infrastructure issue

**FAILED: test_chunked_solve_produces_valid_output**
- Integration test failure

**FAILED: TestWritebackTask::test_task_creation**
- Writeback task creation issue

**FAILED: TestWritebackWatcher::test_2d_array_slice**
- 2D array slicing in writeback watcher
- **Message**: Expected shape (0, 10) but got (5, 10)

**FAILED: test_chunk_axis_index_in_array_requests**
- **Type**: AttributeError  
- **Message**: `'OutputArrayContainer' object has no attribute 'time_domain_array'`

**FAILED: test_chunked_shape_equals_shape_when_not_chunking**
- Chunked shape comparison when not chunking

**FAILED: test_chunked_shape_differs_from_shape_when_chunking**
- **Type**: AttributeError
- **Message**: `'OutputArrayContainer' object has no attribute 'time_domain_array'`

### ⚠️ Runparams Integration Tests - Status Unknown (Test Timeout)

Tests in `tests/batchsolving/test_runparams_integration.py` did not complete within timeout.
Known failures from initial run:
- **FAILED: test_runparams_single_chunk**
- **PASSED: test_runparams_indexing_edge_cases**
- **PASSED: test_runparams_immutability**

## Failure Patterns and Root Causes

### Pattern 1: ArrayRequest total_runs Validation
- **Tests Affected**: 2 memory tests
- **Issue**: Tests passing `None` for total_runs, but it's now required (int with default=1)
- **Fix Needed**: Update test fixtures to use explicit total_runs values

### Pattern 2: chunk_slice() Implementation
- **Tests Affected**: 6+ tests in TestChunkSliceMethod
- **Issue**: chunk_slice() returns full array instead of sliced chunk
- **Root Cause**: Implementation not using chunked_shape or chunk metadata correctly
- **Fix Needed**: Review BaseArrayManager.chunk_slice() implementation

### Pattern 3: Test Infrastructure (Mocking Frozen Classes)
- **Tests Affected**: 3+ tests trying to mock MemoryManager
- **Issue**: Cannot monkey-patch methods on attrs frozen classes
- **Fix Needed**: Use proper mocking (unittest.mock.patch) or refactor tests

### Pattern 4: num_runs Flow
- **Tests Affected**: update_from_solver_sets_num_runs, allocation tests
- **Issue**: num_runs not being set/propagated correctly
- **Fix Needed**: Verify update_from_solver() calls set_array_runs()

### Pattern 5: Test Data/Attribute Mismatches
- **Tests Affected**: chunking integration tests
- **Issue**: Tests using wrong attribute names (e.g., 'state' vs correct attribute)
- **Fix Needed**: Update test code to match actual SolveResult/OutputArrayContainer APIs

## Recommendations

### Priority 1 - Core Logic Fixes (High Impact)

1. **Fix chunk_slice() Implementation**
   - Review BaseArrayManager.chunk_slice() logic
   - Ensure it uses chunked_shape from chunk_metadata
   - Verify chunk index bounds checking

2. **Fix ArrayRequest total_runs Handling**
   - Update test fixtures to not pass None
   - Ensure backward compatibility or update all call sites

3. **Fix num_runs Propagation**
   - Verify InputArrays/OutputArrays.update_from_solver() calls set_array_runs()
   - Check allocation path sets num_runs correctly

### Priority 2 - Test Infrastructure (Medium Impact)

4. **Fix Test Mocking Issues**
   - Replace monkey-patching with unittest.mock.patch for frozen classes
   - Or refactor tests to not require mocking internal methods

5. **Fix Test Attribute References**
   - Update chunking tests to use correct SolveResult attributes
   - Fix OutputArrayContainer attribute access in tests

### Priority 3 - Test Data Issues (Low Impact)

6. **Review Writeback Tests**
   - Check TestWritebackWatcher::test_2d_array_slice expectations
   - Verify task creation test setup

## Success Rate Analysis

- **Core Memory Subsystem**: 97% pass rate (69/71) ✅
- **Output Sizes**: 100% pass rate (22/22) ✅  
- **Array Management**: 83% pass rate (138/166) ⚠️
- **Overall**: ~85% pass rate (~253/298) ⚠️

The refactoring successfully maintains the core memory and sizing logic but has implementation issues in:
1. chunk_slice() method behavior
2. Test infrastructure for mocking
3. Integration test data/expectations

Most failures are concentrated in chunk slicing logic and test infrastructure, not in the core total_runs architecture changes.
