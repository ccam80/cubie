# Implementation Task List
# Feature: test_cleanup_chunking
# Plan Reference: .github/active_plans/test_cleanup_chunking/agent_plan.md

## Task Group 1: Remove Duplicate Test Class
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: tests/batchsolving/test_pinned_memory_refactor.py (entire file, 204 lines)
- File: tests/batchsolving/test_chunked_solver.py (lines 49-93 for comparison)

**Input Validation Required**:
- None (this is a test removal task)

**Tasks**:
1. **Remove TestChunkedVsNonChunkedResults class**
   - File: tests/batchsolving/test_pinned_memory_refactor.py
   - Action: Delete
   - Details:
     Remove the entire `TestChunkedVsNonChunkedResults` class (lines 164-204).
     This class contains only one test method `test_chunked_results_match_non_chunked`
     which is a near-duplicate of `TestSyncStreamRemoval.test_chunked_solver_produces_correct_results`
     in `tests/batchsolving/test_chunked_solver.py`.
     
     The test in `test_chunked_solver.py` is more complete as it includes driver settings.
     
     Lines to remove (inclusive):
     ```python
     class TestChunkedVsNonChunkedResults:
         def test_chunked_results_match_non_chunked(
             self, system, solver, precision, low_mem_solver
         ):
             """Chunked execution produces same results as non-chunked."""
             # Create two solvers - one normal, one with low memory forcing chunks
             solver_normal = solver
             solver_low = low_mem_solver

             n_runs = 3
             n_states = system.sizes.states
             n_params = system.sizes.parameters

             inits = np.ones((n_states, n_runs), dtype=precision)
             params = np.ones((n_params, n_runs), dtype=precision)

             # Run with normal (non-chunked) solver
             result_normal = solver_normal.solve(
                 inits.copy(),
                 params.copy(),
                 duration=0.1,
                 save_every=0.01,
             )

             # Run with low memory (chunked) solver
             result_chunked = solver_low.solve(
                 inits.copy(),
                 params.copy(),
                 duration=0.1,
                 save_every=0.01,
                 dt=solver.dt,
             )

             # Results should match (within floating point tolerance)
             np.testing.assert_allclose(
                 result_chunked.time_domain_array,
                 result_normal.time_domain_array,
                 rtol=1e-5,
                 atol=1e-7,
             )
     ```
   - Edge cases: None - straightforward deletion
   - Integration: The remaining tests in `test_pinned_memory_refactor.py` still cover 
     memory strategy behavior. The duplicate comparison test remains in `test_chunked_solver.py`.

**Tests to Create**:
- None (removing a test, not adding)

**Tests to Run**:
- tests/batchsolving/test_pinned_memory_refactor.py (verify remaining tests still pass)
- tests/batchsolving/test_chunked_solver.py::TestSyncStreamRemoval::test_chunked_solver_produces_correct_results (verify the kept test passes)

**Outcomes**: 
- Files Modified:
  * tests/batchsolving/test_pinned_memory_refactor.py (42 lines removed)
- Functions/Methods Added/Modified:
  * Removed TestChunkedVsNonChunkedResults class entirely
  * Removed unused `Solver` import
- Implementation Summary:
  Deleted the duplicate test class `TestChunkedVsNonChunkedResults` which contained
  `test_chunked_results_match_non_chunked`. This test was a near-duplicate of
  `TestSyncStreamRemoval.test_chunked_solver_produces_correct_results` in
  `test_chunked_solver.py`. Also removed the orphaned `Solver` import that was
  no longer used by any remaining tests. Cleaned up trailing blank lines.
- Issues Flagged: None

---

## Task Group 2: Update Test Documentation
**Status**: [x]
**Dependencies**: None (can run in parallel with Task Group 1)

**Required Context**:
- File: tests/batchsolving/test_chunked_solver.py (lines 1-47)

**Input Validation Required**:
- None (this is a documentation update task)

**Tasks**:
1. **Update module docstring to reflect actual test coverage**
   - File: tests/batchsolving/test_chunked_solver.py
   - Action: Modify
   - Details:
     The current module docstring (lines 1-7) states:
     ```python
     """Integration tests for chunked solver execution.

     These tests verify that the chunking functionality works correctly,
     testing both the "run" and "time" chunk axes to ensure the fixes for:
     1. Stride incompatibility when chunking on the run axis
     2. Missing axis error when chunking on the time axis
     """
     ```
     
     However, line 16 shows that only "run" axis is tested:
     ```python
     @pytest.mark.parametrize("chunk_axis", ["run"])  # , "time"])
     ```
     
     Update the docstring to accurately describe what is tested.
     The new docstring should clarify that only the "run" chunk axis is
     currently tested, while noting that "time" axis support exists but
     is not covered by this test.
     
     New docstring:
     ```python
     """Integration tests for chunked solver execution.

     These tests verify that the chunking functionality works correctly
     for the "run" chunk axis. Time-axis chunking is supported but not
     covered by these tests.
     """
     ```
   - Edge cases: None
   - Integration: Documentation only, no functional changes

2. **Add comment explaining disabled time axis parameterization**
   - File: tests/batchsolving/test_chunked_solver.py
   - Action: Modify
   - Details:
     Add a comment above line 16 explaining why time axis is commented out.
     The comment should be brief and explain the status without making
     claims about history or changes.
     
     Current line:
     ```python
     @pytest.mark.parametrize("chunk_axis", ["run"])  # , "time"])
     ```
     
     Updated with comment:
     ```python
     # Time-axis chunking is supported but not tested here; run-axis
     # chunking is the primary use case.
     @pytest.mark.parametrize("chunk_axis", ["run"])
     ```
     
     Note: Remove the inline comment `# , "time"]` since it adds confusion
     without context. The new comment above the line provides clarity.
   - Edge cases: None
   - Integration: Documentation only, no functional changes

**Tests to Create**:
- None (documentation only)

**Tests to Run**:
- tests/batchsolving/test_chunked_solver.py (verify tests still pass after docstring changes)

**Outcomes**:
- Files Modified:
  * tests/batchsolving/test_chunked_solver.py (4 lines changed)
- Functions/Methods Added/Modified:
  * None (documentation only)
- Implementation Summary:
  Updated module docstring to accurately describe that only "run" chunk axis
  is tested, while noting that time-axis chunking is supported but not covered.
  Added explanatory comment above the @pytest.mark.parametrize decorator and
  removed the confusing inline comment `# , "time"]`.
- Issues Flagged: None

---

## Summary

| Task Group | Description | Files Modified | Dependencies |
|------------|-------------|----------------|--------------|
| 1 | Remove duplicate test class | test_pinned_memory_refactor.py | None |
| 2 | Update test documentation | test_chunked_solver.py | None |

### Dependency Chain Overview

```
Task Group 1 ─┐
              ├──> Both can run in parallel (no dependencies)
Task Group 2 ─┘
```

### Tests Summary

**Tests to Run (via run_tests agent):**
- `tests/batchsolving/test_pinned_memory_refactor.py`
- `tests/batchsolving/test_chunked_solver.py`

### Estimated Complexity

- **Task Group 1**: Low - Single class deletion
- **Task Group 2**: Low - Docstring and comment updates

**Total estimated time**: ~10 minutes for implementation, ~5 minutes for test validation
