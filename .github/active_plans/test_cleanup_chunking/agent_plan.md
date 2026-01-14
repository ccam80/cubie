# Test Cleanup for Chunking System: Agent Plan

## Overview

This plan guides the detailed_implementer agent through cleaning up the chunking-related tests in the CuBIE repository. The goal is to ensure tests accurately test what they claim and remove redundant tests.

## Context

The chunking system was recently refactored and many tests were added. Some tests:
1. Claim to test chunking but actually test non-chunked execution
2. Duplicate functionality already tested in `test_solver.py`
3. Use workarounds for cudasim limitations that mask the intended test behavior

## Files to Modify

### Primary Targets

1. **`tests/batchsolving/test_pinned_memory_refactor.py`**
   - Location of duplicate test to remove

2. **`tests/batchsolving/test_chunked_solver.py`**
   - Contains commented-out time axis test that needs investigation

## Detailed Component Analysis

### 1. Duplicate Test Removal

**Test to Remove:** `TestChunkedVsNonChunkedResults.test_chunked_results_match_non_chunked`

**Location:** `tests/batchsolving/test_pinned_memory_refactor.py` (lines 164-203)

**Reason:** This test is a near-exact duplicate of `TestSyncStreamRemoval.test_chunked_solver_produces_correct_results` in `tests/batchsolving/test_chunked_solver.py` (lines 52-93).

**Comparison of the two tests:**

| Aspect | test_chunked_solver.py | test_pinned_memory_refactor.py |
|--------|------------------------|--------------------------------|
| n_runs | 3 | 3 |
| duration | 0.1 | 0.1 |
| save_every | 0.01 | 0.01 |
| drivers | Yes (driver_settings) | No |
| dt override | No | Yes (dt=solver.dt) |
| Comparison | np.testing.assert_allclose | np.testing.assert_allclose |
| rtol/atol | 1e-5, 1e-7 | 1e-5, 1e-7 |

The test in `test_chunked_solver.py` is more complete because it includes driver settings. The version in `test_pinned_memory_refactor.py` is redundant.

**Action:** Remove the `TestChunkedVsNonChunkedResults` class entirely from `test_pinned_memory_refactor.py` (lines 164-203, including the blank line at 204 if present).

### 2. Test Documentation Improvement

**Test to Document:** `TestChunkedSolverExecution.test_chunked_solve_produces_valid_output`

**Location:** `tests/batchsolving/test_chunked_solver.py` (lines 17-47)

**Issue:** The test is parameterized with `chunk_axis="run"` only, with `"time"` commented out:
```python
@pytest.mark.parametrize("chunk_axis", ["run"])  # , "time"])
```

**Action:** Add a comment explaining why the time axis test is disabled, OR enable it if the underlying issue has been fixed.

The docstring already mentions testing both axes:
> "These tests verify that the chunking functionality works correctly, testing both the 'run' and 'time' chunk axes"

But only "run" is actually tested. This should be reconciled.

### 3. Test Intent Verification

The following tests were reviewed and found to be correctly testing what they claim:

**`test_pinned_memory_refactor.py`:**
- `test_non_chunked_uses_pinned_host` - Tests memory type, not solver correctness
- `test_chunked_uses_numpy_host` - Tests memory type in chunked mode
- `test_total_pinned_memory_bounded` - Tests buffer pool behavior

**`test_chunked_solver.py`:**
- `test_chunked_solve_produces_valid_output` - Tests chunked execution produces valid output
- `test_chunked_solver_produces_correct_results` - Compares chunked vs non-chunked results
- `test_input_buffers_released_after_kernel` - Tests cleanup behavior

**`test_writeback_watcher.py`:**
All tests are unit tests for the WritebackWatcher class, no integration overlap.

**`test_chunk_buffer_pool.py`:**
All tests are unit tests for ChunkBufferPool class, no integration overlap.

**`test_conditional_memory.py`:**
All tests are unit tests for memory type selection, no integration overlap.

**`test_chunk_axis_property.py`:**
All tests are unit tests for chunk_axis property, no integration overlap.

## Integration Points

### Fixture Dependencies

The tests use these key fixtures from `tests/conftest.py`:

- `solver` - Session-scoped solver with default memory
- `low_mem_solver` - Session-scoped solver with MockMemoryManager (32KB limit)
- `system` - The ODE system
- `precision` - float32 or float64
- `driver_settings` - Driver configuration

These fixtures do not need modification.

### Expected Interactions

After the cleanup:
1. `test_chunked_solver.py` will be the authoritative location for chunked vs non-chunked comparison tests
2. `test_pinned_memory_refactor.py` will focus solely on memory strategy tests
3. No test will be ambiguous about what it's testing

## Data Structures

No data structures need modification. This is purely a test cleanup.

## Edge Cases

1. **Cudasim Mode:** Tests should work in both CUDA and cudasim modes. The removed test was not marked `@pytest.mark.nocudasim`, and neither is the test being kept, so behavior is consistent.

2. **Time Axis Chunking:** The commented-out time axis test suggests there may be issues with time-axis chunking. The implementer should:
   - Check if time-axis chunking is supported
   - If not supported, update docstring to clarify only run-axis is tested
   - If supported but broken, add a comment explaining the issue

## Dependencies and Imports

No import changes needed. The removal is of an entire test class.

## Summary of Changes

| File | Change Type | Description |
|------|-------------|-------------|
| `tests/batchsolving/test_pinned_memory_refactor.py` | Remove | Delete `TestChunkedVsNonChunkedResults` class (lines 164-203) |
| `tests/batchsolving/test_chunked_solver.py` | Document | Update docstring or comment to explain time axis status |

## Validation

After changes:
1. Run `pytest tests/batchsolving/test_pinned_memory_refactor.py -v` - Should pass with one fewer test
2. Run `pytest tests/batchsolving/test_chunked_solver.py -v` - Should pass unchanged
3. Run `pytest tests/batchsolving/ -v` - Full batchsolving test suite should pass

## Notes for Reviewer

The reviewer agent should verify:
1. The removed test is truly a duplicate (compare implementations)
2. No coverage gap was introduced
3. The remaining test in `test_chunked_solver.py` adequately tests chunked vs non-chunked comparison
4. Test docstrings accurately describe what is being tested
