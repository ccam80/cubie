# Implementation Task List
# Feature: Fix Rosenbrock Solver Flaky Test Errors
# Plan Reference: .github/active_plans/fix_rosenbrock_flaky_test/agent_plan.md

## Task Group 1: Fix stage_increment Buffer Registration
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 243-275)

**Input Validation Required**:
- None - this is a parameter addition to an existing function call

**Tasks**:
1. **Add persistent=True to stage_increment buffer registration**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     # Current code (lines 268-274):
     # Stage increment should persist between steps for initial guess
     buffer_registry.register(
         'stage_increment', self, n,
         config.stage_store_location,
         aliases='stage_store',
         precision=precision
     )
     
     # Change to:
     # Stage increment should persist between steps for initial guess
     buffer_registry.register(
         'stage_increment', self, n,
         config.stage_store_location,
         aliases='stage_store',
         persistent=True,
         precision=precision
     )
     ```
   - Edge cases:
     - If `stage_store_location='shared'`: The aliasing still works correctly. Shared memory is also zeroed at loop entry.
     - Subsequent stages use previous stage's solution from `stage_store`, not `stage_increment` - unaffected.
   - Integration: The buffer registry already supports the `persistent` parameter. No changes needed to buffer_registry.py or ode_loop.py.

**Tests to Create**:
- No new tests needed - existing tests validate this behavior

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py::test_loop[rosenbrock]
- tests/integrators/loops/test_ode_loop.py::test_loop[rosenbrock-ros3p]
- tests/integrators/loops/test_ode_loop.py::test_loop[rosenbrock-ode23s]
- tests/integrators/loops/test_ode_loop.py::test_loop[rosenbrock-rodas3p]

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: Verify Fix with Multiple Test Runs
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: tests/integrators/loops/test_ode_loop.py (entire file)
- File: tests/_utils.py (lines 40-81, STEP_CASES with rosenbrock algorithms)

**Input Validation Required**:
- None - this is a verification task

**Tasks**:
1. **Run Rosenbrock tests 3 times to verify no flaky failures**
   - File: N/A (test execution)
   - Action: Run tests
   - Details:
     Run the rosenbrock algorithm tests multiple times to verify the fix eliminates flaky failures.
     The tests should pass consistently on all runs.
     
     Test commands to execute (3 iterations):
     ```bash
     pytest tests/integrators/loops/test_ode_loop.py -k "rosenbrock" -v
     pytest tests/integrators/loops/test_ode_loop.py -k "rosenbrock" -v
     pytest tests/integrators/loops/test_ode_loop.py -k "rosenbrock" -v
     ```
   - Edge cases:
     - If running without CUDA (NUMBA_ENABLE_CUDASIM=1), tests should still pass
     - Tests marked with `specific_algos` (ros3p, ode23s, rodas3p) may be skipped unless `-m "specific_algos"` is included
   - Integration: Uses existing test infrastructure

**Tests to Create**:
- No new tests needed

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py (with -k "rosenbrock" filter)
- Run 3 times to verify no flaky failures

**Outcomes**: 
- Files Modified: None (verification task only)
- Functions/Methods Added/Modified: None
- Implementation Summary:
  Task Group 2 is a verification task requiring test execution.
  The run_tests agent should execute the rosenbrock tests 3 times to verify
  the fix from Task Group 1 (adding persistent=True to stage_increment buffer
  registration) eliminates flaky failures.
  
  Test command: `pytest tests/integrators/loops/test_ode_loop.py -k "rosenbrock" -v`
  Required iterations: 3
  Environment: Set NUMBA_ENABLE_CUDASIM=1 if no CUDA GPU available
  
- Issues Flagged: None - this is a verification task for run_tests agent

---

## Summary

| Task Group | Description | Files Modified | Estimated Complexity |
|------------|-------------|----------------|---------------------|
| 1 | Fix buffer registration | generic_rosenbrock_w.py | Low (1 line change) |
| 2 | Verify fix with tests | None (test execution) | Low |

### Dependency Chain

```
Task Group 1 (Fix) ──► Task Group 2 (Verify)
```

### Total Changes

- **Source files modified**: 1
- **Lines changed**: 1 (add `persistent=True,`)
- **New tests**: 0
- **Existing tests to run**: 4+ (rosenbrock variants in test_ode_loop.py)
