# Implementation Task List
# Feature: FSAL Caching Warp Divergence Fix
# Plan Reference: .github/active_plans/fsal_caching_divergence/agent_plan.md

## Task Group 1: Import Warp Synchronization Primitives - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_erk.py (lines 1-21)
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 1-26)
- File: src/cubie/cuda_simsafe.py (lines 185-209) - for reference on available functions
- File: src/cubie/integrators/loops/ode_loop.py (line 16) - for existing import pattern
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (line 14) - for existing import pattern

**Input Validation Required**:
- None (imports do not require validation)

**Tasks**:
1. **Add warp synchronization imports to generic_erk.py**
   - File: src/cubie/integrators/algorithms/generic_erk.py
   - Action: Modify
   - Details:
     Add import after existing imports (around line 6-7, after numba imports):
     ```python
     from cubie.cuda_simsafe import activemask, all_sync
     ```
   - Edge cases: None - these functions already exist and are used elsewhere
   - Integration: Follows existing pattern from ode_loop.py and newton_krylov.py

2. **Add warp synchronization imports to generic_dirk.py**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Add import after existing imports (around line 7-8, after numba imports):
     ```python
     from cubie.cuda_simsafe import activemask, all_sync
     ```
   - Edge cases: None - these functions already exist and are used elsewhere
   - Integration: Follows existing pattern from ode_loop.py and newton_krylov.py

**Outcomes**: 
[Empty - to be filled by do_task agent]

---

## Task Group 2: Implement Warp-Synchronized FSAL Caching in Generic ERK - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_erk.py (lines 200-228) - current FSAL caching implementation
- File: src/cubie/integrators/algorithms/generic_erk.py (lines 186-195) - shared memory layout
- File: src/cubie/integrators/algorithms/generic_erk.py (lines 38-48) - ERKStepConfig with first_same_as_last property
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 224-226) - example usage of all_sync
- File: src/cubie/integrators/loops/ode_loop.py (line 16) - activemask and all_sync import

**Input Validation Required**:
- None (all inputs are compile-time constants or per-thread state from loop)

**Tasks**:
1. **Modify FSAL cache decision to use warp-synchronized vote in ERK**
   - File: src/cubie/integrators/algorithms/generic_erk.py
   - Action: Modify
   - Details:
     Replace lines 205-206:
     ```python
     # CURRENT CODE:
     use_cached_rhs = ((not first_step_flag) and accepted_flag and
                       first_same_as_last)
     
     # NEW CODE:
     mask = activemask()
     all_threads_accepted = all_sync(mask, accepted_flag != int16(0))
     use_cached_rhs = ((not first_step_flag) and all_threads_accepted and
                       first_same_as_last)
     ```
     Implementation logic:
     1. Call activemask() to get the active thread mask in the warp
     2. Use all_sync() to check if ALL threads in the warp have accepted_flag != 0
     3. Use the warp-level acceptance in the cache decision instead of per-thread acceptance
     4. Keep all other conditions (first_step_flag, first_same_as_last) unchanged
   - Edge cases:
     - First step: first_step_flag is uniform across threads, no divergence
     - Non-FSAL tableaus: first_same_as_last is False at compile time, short-circuits
     - Single-stage methods: multistage is False, cache path not executed
     - CUDASIM mode: all_sync returns True trivially (single thread)
   - Integration: The loop (ode_loop.py) continues to pass per-thread accepted_flag unchanged; this change happens purely inside the step function

**Outcomes**: 
[Empty - to be filled by do_task agent]

---

## Task Group 3: Implement Warp-Synchronized FSAL Caching in Generic DIRK - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 315-336) - current FSAL caching implementation
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 298-301) - shared memory layout
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 493-495) - cache commit via increment_cache aliasing
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 224-226) - example usage of all_sync
- File: src/cubie/integrators/loops/ode_loop.py (line 16) - activemask and all_sync import

**Input Validation Required**:
- None (all inputs are compile-time constants or per-thread state from loop)

**Tasks**:
1. **Modify FSAL cache decision to use warp-synchronized vote in DIRK**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Replace lines 319-323:
     ```python
     # CURRENT CODE:
     first_step = first_step_flag != int16(0)
     prev_state_accepted = accepted_flag != int16(0)
     use_cached_rhs = (
         first_same_as_last and not first_step and prev_state_accepted
     )
     
     # NEW CODE:
     first_step = first_step_flag != int16(0)
     mask = activemask()
     all_threads_accepted = all_sync(mask, accepted_flag != int16(0))
     use_cached_rhs = (
         first_same_as_last and not first_step and all_threads_accepted
     )
     ```
     Implementation logic:
     1. Keep the first_step conversion unchanged
     2. Call activemask() to get the active thread mask in the warp
     3. Use all_sync() to check if ALL threads in the warp have accepted_flag != 0
     4. Replace prev_state_accepted with all_threads_accepted in the cache decision
     5. Keep all other conditions unchanged
   - Edge cases:
     - First step: first_step is uniform across threads, no divergence
     - Non-FSAL tableaus: first_same_as_last is False at compile time, short-circuits
     - CUDASIM mode: all_sync returns True trivially (single thread)
     - Cache aliasing: increment_cache aliasing at line 495 remains unchanged
   - Integration: The loop (ode_loop.py) continues to pass per-thread accepted_flag unchanged; cache mechanism via aliasing to solver_scratch remains identical

**Outcomes**: 
[Empty - to be filled by do_task agent]

---

## Task Group 4: Validation Testing - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 2, 3

**Required Context**:
- File: tests/integrators/algorithms/test_generic_erk_tableaus.py (entire file) - existing ERK tests
- File: tests/integrators/algorithms/test_dirk_tableaus.py (entire file) - existing DIRK tests
- File: tests/integrators/algorithms/test_step_algorithms.py (entire file) - generic algorithm tests
- File: tests/conftest.py (entire file) - test fixtures and markers
- File: tests/system_fixtures.py (entire file) - ODE system fixtures

**Input Validation Required**:
- None (validation task, not implementation)

**Tasks**:
1. **Run existing ERK test suite**
   - File: N/A (testing task)
   - Action: Test
   - Details:
     Execute the existing test suite for generic ERK algorithms:
     ```bash
     pytest tests/integrators/algorithms/test_generic_erk_tableaus.py -v
     ```
     Validation:
     1. All existing tests must pass with identical numerical results
     2. No new failures introduced
     3. Tests cover adaptive stepping scenarios where accepted_flag varies
   - Edge cases: 
     - CUDASIM mode: Run with `pytest -m "not nocudasim"` if GPU unavailable
     - Fixed-step mode: Tests should show 100% cache hit rate
     - Non-FSAL tableaus: Tests should show no performance change
   - Integration: Tests validate correctness across multiple tableau types and stepping scenarios

2. **Run existing DIRK test suite**
   - File: N/A (testing task)
   - Action: Test
   - Details:
     Execute the existing test suite for generic DIRK algorithms:
     ```bash
     pytest tests/integrators/algorithms/test_dirk_tableaus.py -v
     ```
     Validation:
     1. All existing tests must pass with identical numerical results
     2. No new failures introduced
     3. Tests cover implicit solving with Newton-Krylov convergence variations
   - Edge cases:
     - CUDASIM mode: Run with `pytest -m "not nocudasim"` if GPU unavailable
     - Fixed-step mode: Tests should show 100% cache hit rate
     - Non-FSAL tableaus: Tests should show no performance change
   - Integration: Tests validate correctness with implicit solvers and varying convergence patterns

3. **Run comprehensive algorithm test suite**
   - File: N/A (testing task)
   - Action: Test
   - Details:
     Execute the broader algorithm test suite:
     ```bash
     pytest tests/integrators/algorithms/test_step_algorithms.py -v
     ```
     Validation:
     1. All tests pass without regression
     2. Tests cover multiple algorithms, precisions, and system sizes
     3. Adaptive stepping scenarios validated
   - Edge cases:
     - Mixed acceptance patterns: Tests with heterogeneous error growth
     - Large batch sizes: Multiple warps with different acceptance patterns
     - Small systems: Single-stage or non-FSAL methods
   - Integration: Validates that changes don't affect other algorithm implementations

**Outcomes**: 
[Empty - to be filled by do_task agent]

---

## Summary

**Total Task Groups**: 4

**Dependency Chain**:
```
Group 1 (Imports) 
    ↓
Group 2 (ERK Implementation) ←—┐
    ↓                          |
Group 3 (DIRK Implementation) —┘
    ↓
Group 4 (Validation Testing)
```

**Parallel Execution Opportunities**:
- Groups 2 and 3 can be done in any order after Group 1 completes
- All tasks within Group 4 can run in parallel

**Estimated Complexity**:
- **Simple**: Groups 1, 2, 3 (minimal code changes, well-defined pattern)
- **Moderate**: Group 4 (validation requires running full test suite)

**Critical Success Factors**:
1. Imports must match existing pattern from ode_loop.py and newton_krylov.py
2. Warp vote must use `accepted_flag != int16(0)` for correct boolean conversion
3. All other cache decision conditions (first_step_flag, first_same_as_last) remain unchanged
4. Cache storage and aliasing mechanisms are NOT modified
5. All existing tests must pass with identical numerical results
6. No API changes or backward compatibility breaks

**Files Modified**:
- src/cubie/integrators/algorithms/generic_erk.py (2 locations: import + cache decision)
- src/cubie/integrators/algorithms/generic_dirk.py (2 locations: import + cache decision)

**Files Tested**:
- tests/integrators/algorithms/test_generic_erk_tableaus.py
- tests/integrators/algorithms/test_dirk_tableaus.py
- tests/integrators/algorithms/test_step_algorithms.py

**Total Lines Changed**: ~10 lines across 2 files

**Performance Impact**:
- Uniform acceptance (all threads accept): Retains FSAL benefit, minimal warp-sync overhead
- Mixed acceptance (divergent): Eliminates warp divergence penalty
- Net improvement in realistic adaptive scenarios
- No regression in fixed-step or non-FSAL scenarios
