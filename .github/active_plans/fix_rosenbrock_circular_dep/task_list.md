# Implementation Task List
# Feature: Fix Rosenbrock Circular Dependency
# Plan Reference: .github/active_plans/fix_rosenbrock_circular_dep/agent_plan.md

## Task Group 1: Fix shared_memory_required Property - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 967-973)
- File: src/cubie/integrators/algorithms/generic_erk.py (lines 808-810) - Reference pattern

**Input Validation Required**:
- None - no parameters to validate; this is a property modification

**Tasks**:
1. **Modify GenericRosenbrockWStep.shared_memory_required Property**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     Replace the current implementation that directly calculates shared memory
     using `self.cached_auxiliary_count` (which triggers a build) with delegation
     to `buffer_settings.shared_memory_elements` following the GenericERKStep pattern.
     
     **Current code (lines 967-973):**
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""
         accumulator_span = self.stage_count * self.n
         cached_auxiliary_count = self.cached_auxiliary_count
         shared_buffers = self.n
         return accumulator_span + cached_auxiliary_count + shared_buffers
     ```
     
     **Replace with:**
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""
         return self.compile_settings.buffer_settings.shared_memory_elements
     ```
     
   - Edge cases:
     - At init time: `cached_auxiliary_count` is 0 in buffer_settings (default),
       so `shared_memory_elements` returns correct pre-build value
     - After build: `cached_auxiliary_count` is updated in buffer_settings (line 479),
       so subsequent calls return the correct value with actual auxiliary count
     - With all buffers configured as 'local' (default): Returns 0
     - With all buffers configured as 'shared': Returns full calculation
     
   - Integration: 
     - No interface changes required
     - BatchSolverKernel.__init__ will receive correct values
     - SingleIntegratorRun.shared_memory_elements already calls this property
     - RosenbrockBufferSettings.shared_memory_elements already implements 
       the calculation correctly

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: Verify No Regressions - PARALLEL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/integrators/algorithms/ - Rosenbrock algorithm tests
- File: tests/batchsolving/ - Solver instantiation tests (if any)

**Input Validation Required**:
- None - test execution only

**Tasks**:
1. **Run Existing Rosenbrock Tests**
   - File: N/A (test execution)
   - Action: Execute
   - Details:
     Run existing Rosenbrock algorithm tests to verify no regressions:
     ```bash
     pytest tests/integrators/algorithms/ -k rosenbrock -v
     ```
     
     If no tests exist with that pattern, run broader tests:
     ```bash
     pytest tests/integrators/algorithms/ -v
     ```
     
   - Edge cases:
     - Tests may fail if CUDASIM is not enabled and no GPU available
     - Mark tests as needed for the environment
     
   - Integration: Existing test infrastructure

2. **Run Solver Instantiation Tests**
   - File: N/A (test execution)
   - Action: Execute
   - Details:
     Run solver/batchsolving tests to ensure instantiation works:
     ```bash
     pytest tests/batchsolving/ -v
     ```
     
   - Edge cases:
     - Same CUDA environment considerations
     
   - Integration: Existing test infrastructure

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

### Total Task Groups: 2
### Dependency Chain:
1. Task Group 1 (Fix property) → No dependencies
2. Task Group 2 (Verify) → Depends on Task Group 1

### Parallel Execution Opportunities:
- Task Group 1 is a single task (cannot parallelize)
- Task Group 2 tasks (tests) can run in parallel

### Estimated Complexity: Low
- Single property modification (6 lines → 3 lines)
- No new logic, just delegation to existing infrastructure
- Pattern already established in GenericERKStep
- No interface changes
