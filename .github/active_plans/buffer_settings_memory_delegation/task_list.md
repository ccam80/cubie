# Implementation Task List
# Feature: Buffer Settings Memory Delegation
# Plan Reference: .github/active_plans/buffer_settings_memory_delegation/agent_plan.md

## Task Group 1: BufferSettings Base Class Enhancement - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/BufferSettings.py (entire file, 115 lines)

**Input Validation Required**:
- None (abstract property only)

**Tasks**:
1. **Add persistent_local_elements abstract property to BufferSettings**
   - File: src/cubie/BufferSettings.py
   - Action: Modify
   - Details:
     ```python
     @property
     @abstractmethod
     def persistent_local_elements(self) -> int:
         """Return persistent local memory elements required.
         
         Persistent local memory survives between step invocations,
         used for FSAL (First Same As Last) caching optimization.
         """
         pass
     ```
   - Location: Add after `shared_indices` abstract property (after line 114)
   - Edge cases: None - abstract property definition only
   - Integration: All BufferSettings subclasses must implement this property

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: RosenbrockBufferSettings Enhancement - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 100-236)
  - RosenbrockBufferSettings class definition

**Input Validation Required**:
- None (property only returns int)

**Tasks**:
1. **Add persistent_local_elements property to RosenbrockBufferSettings**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     @property
     def persistent_local_elements(self) -> int:
         """Return persistent local memory elements required.
         
         Rosenbrock methods do not use FSAL caching, so no persistent
         local memory is required.
         """
         return 0
     ```
   - Location: Add after `shared_indices` property (after line 235)
   - Edge cases: None - always returns 0 for Rosenbrock
   - Integration: Enables RosenbrockBufferSettings to satisfy BufferSettings ABC

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: FIRKBufferSettings Enhancement - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 103-270)
  - FIRKBufferSettings class definition

**Input Validation Required**:
- None (property only returns int)

**Tasks**:
1. **Add persistent_local_elements property to FIRKBufferSettings**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     @property
     def persistent_local_elements(self) -> int:
         """Return persistent local memory elements required.
         
         FIRK methods do not use FSAL caching, so no persistent
         local memory is required.
         """
         return 0
     ```
   - Location: Add after `local_memory_elements` property (around line 212)
   - Edge cases: None - always returns 0 for FIRK
   - Integration: Enables FIRKBufferSettings to satisfy BufferSettings ABC

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: DIRKBufferSettings Enhancement - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 111-313)
  - DIRKBufferSettings class definition
  - DIRKBufferSettings already has `persistent_local_elements` property (lines 179-189)

**Input Validation Required**:
- None (no changes needed)

**Tasks**:
1. **Verify DIRKBufferSettings already has persistent_local_elements**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Verify (no modification needed)
   - Details: DIRKBufferSettings already has persistent_local_elements property at lines 179-189. Verify it exists and satisfies the ABC.
   - Edge cases: None
   - Integration: DIRKBufferSettings already satisfies BufferSettings ABC for this property

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: GenericRosenbrockWStep Memory Delegation - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 967-984)
  - Current shared_memory_required property (lines 967-973)
  - Current local_scratch_required property (lines 975-979)
  - Current persistent_local_required property (lines 981-984)

**Input Validation Required**:
- None (property delegation only)

**Tasks**:
1. **Update shared_memory_required to delegate to BufferSettings**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     Replace lines 967-973:
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""
         accumulator_span = self.stage_count * self.n
         cached_auxiliary_count = self.cached_auxiliary_count
         shared_buffers = self.n
         return accumulator_span + cached_auxiliary_count + shared_buffers
     ```
     With:
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""
         return self.compile_settings.buffer_settings.shared_memory_elements
     ```
   - Edge cases: 
     - At init time, buffer_settings.cached_auxiliary_count = 0, giving minimal memory
     - After build, cached_auxiliary_count is updated, giving accurate memory
   - Integration: Matches ERK delegation pattern

2. **Update persistent_local_required to delegate to BufferSettings**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     Replace lines 981-984:
     ```python
     @property
     def persistent_local_required(self) -> int:
         """Return the number of persistent local entries required."""
         return 0
     ```
     With:
     ```python
     @property
     def persistent_local_required(self) -> int:
         """Return the number of persistent local entries required."""
         buffer_settings = self.compile_settings.buffer_settings
         return buffer_settings.persistent_local_elements
     ```
   - Edge cases: Always returns 0 for Rosenbrock (matches current behavior)
   - Integration: Matches ERK delegation pattern

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 6: GenericDIRKStep Memory Delegation - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 4

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 1062-1088)
  - Current shared_memory_required property (lines 1062-1072)
  - Current local_scratch_required property (lines 1074-1077)
  - Current persistent_local_required property (lines 1079-1088, ALREADY DELEGATES)

**Input Validation Required**:
- None (property delegation only)

**Tasks**:
1. **Update shared_memory_required to delegate to BufferSettings**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Replace lines 1062-1072:
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""

         tableau = self.tableau
         stage_count = tableau.stage_count
         accumulator_span = max(stage_count - 1, 0) * self.compile_settings.n
         return (accumulator_span
             + self.solver_shared_elements
             + self.cached_auxiliary_count
         )
     ```
     With:
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""
         return self.compile_settings.buffer_settings.shared_memory_elements
     ```
   - Edge cases:
     - Must verify DIRKBufferSettings includes solver_scratch and cached_auxiliaries in shared_memory_elements
     - Current DIRKBufferSettings.shared_memory_elements (lines 204-223) includes accumulator, solver_scratch, stage_increment, and stage_base but NOT cached_auxiliary_count
   - **CRITICAL**: DIRKBufferSettings.shared_memory_elements may need enhancement to include cached_auxiliary_count. Verify calculation match before delegating.
   - Integration: Matches ERK delegation pattern

2. **Verify persistent_local_required already delegates**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Verify (no modification needed)
   - Details: Lines 1079-1088 already delegate to buffer_settings.persistent_local_elements
   - Edge cases: None
   - Integration: Already follows correct pattern

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 7: GenericFIRKStep Memory Delegation - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 3

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 885-908)
  - Current shared_memory_required property (lines 885-895)
  - Current local_scratch_required property (lines 897-902)
  - Current persistent_local_required property (lines 904-908)

**Input Validation Required**:
- None (property delegation only)

**Tasks**:
1. **Update shared_memory_required to delegate to BufferSettings**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Replace lines 885-895:
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""
         #TODO: When using the shared buffer settings, invalid address
         config = self.compile_settings
         stage_driver_total = self.stage_count * config.n_drivers
         return (
             self.solver_shared_elements
             + stage_driver_total
             + config.all_stages_n
         )
     ```
     With:
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""
         return self.compile_settings.buffer_settings.shared_memory_elements
     ```
   - Edge cases:
     - Note the TODO comment mentions "invalid address" - verify FIRKBufferSettings.shared_memory_elements matches the current manual calculation
     - Current FIRKBufferSettings.shared_memory_elements (lines 180-195) includes solver_scratch, stage_increment, stage_driver_stack, and stage_state
   - **CRITICAL**: Verify calculation match. Current manual = solver_shared (2*all_stages_n) + stage_driver_total + all_stages_n. BufferSettings calculates based on location flags.
   - Integration: Matches ERK delegation pattern

2. **Update persistent_local_required to delegate to BufferSettings**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Replace lines 904-908:
     ```python
     @property
     def persistent_local_required(self) -> int:
         """Return the number of persistent local entries required."""

         return 0
     ```
     With:
     ```python
     @property
     def persistent_local_required(self) -> int:
         """Return the number of persistent local entries required."""
         buffer_settings = self.compile_settings.buffer_settings
         return buffer_settings.persistent_local_elements
     ```
   - Edge cases: Always returns 0 for FIRK (matches current behavior)
   - Integration: Matches ERK delegation pattern

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 8: Instrumented Rosenbrock Memory Properties - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 5

**Required Context**:
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py (lines 723-743)
  - Current shared_memory_required property (lines 723-732)
  - Current local_scratch_required property (lines 734-738)
  - Current persistent_local_required property (lines 740-743)

**Input Validation Required**:
- None (property delegation only)

**Tasks**:
1. **Update shared_memory_required to delegate to BufferSettings**
   - File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     Replace lines 723-732:
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""

         stage_count = max(self.tableau.stage_count, 1)
         state_len = self.compile_settings.n
         stage_buffer = stage_count * state_len
         shared_vectors = 3 * state_len
         cached_auxiliary_count = self.cached_auxiliary_count
         return stage_buffer + shared_vectors + cached_auxiliary_count
     ```
     With:
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""
         return self.compile_settings.buffer_settings.shared_memory_elements
     ```
   - Edge cases: Instrumented version has slightly different calculation formula - verify match
   - Integration: Must match source algorithm implementation

2. **Update persistent_local_required to delegate to BufferSettings**
   - File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     Replace lines 740-743:
     ```python
     @property
     def persistent_local_required(self) -> int:
         """Return the number of persistent local entries required."""
         return 0
     ```
     With:
     ```python
     @property
     def persistent_local_required(self) -> int:
         """Return the number of persistent local entries required."""
         buffer_settings = self.compile_settings.buffer_settings
         return buffer_settings.persistent_local_elements
     ```
   - Edge cases: None
   - Integration: Must match source algorithm implementation

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 9: Instrumented DIRK Memory Properties - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 6

**Required Context**:
- File: tests/integrators/algorithms/instrumented/generic_dirk.py (lines 760-786)
  - Current shared_memory_required property (lines 760-770)
  - Current local_scratch_required property (lines 772-775)
  - Current persistent_local_required property (lines 777-786, ALREADY DELEGATES)

**Input Validation Required**:
- None (property delegation only)

**Tasks**:
1. **Update shared_memory_required to delegate to BufferSettings**
   - File: tests/integrators/algorithms/instrumented/generic_dirk.py
   - Action: Modify
   - Details:
     Replace lines 760-770:
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""

         tableau = self.tableau
         stage_count = tableau.stage_count
         accumulator_span = max(stage_count - 1, 0) * self.compile_settings.n
         return (accumulator_span
             + self.solver_shared_elements
             + self.cached_auxiliary_count
         )
     ```
     With:
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""
         return self.compile_settings.buffer_settings.shared_memory_elements
     ```
   - Edge cases: Same as source algorithm
   - Integration: Must match source algorithm implementation

2. **Verify persistent_local_required already delegates**
   - File: tests/integrators/algorithms/instrumented/generic_dirk.py
   - Action: Verify (no modification needed)
   - Details: Lines 777-786 already delegate to buffer_settings.persistent_local_elements
   - Edge cases: None
   - Integration: Already follows correct pattern

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 10: Instrumented FIRK Memory Properties - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 7

**Required Context**:
- File: tests/integrators/algorithms/instrumented/generic_firk.py (lines 589-610)
  - Current shared_memory_required property (lines 589-599)
  - Current local_scratch_required property (lines 601-604)
  - Current persistent_local_required property (lines 606-610)

**Input Validation Required**:
- None (property delegation only)

**Tasks**:
1. **Update shared_memory_required to delegate to BufferSettings**
   - File: tests/integrators/algorithms/instrumented/generic_firk.py
   - Action: Modify
   - Details:
     Replace lines 589-599:
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""

         config = self.compile_settings
         stage_driver_total = config.stage_count * config.n_drivers
         return (
             self.solver_shared_elements
             + stage_driver_total
             + config.all_stages_n
         )
     ```
     With:
     ```python
     @property
     def shared_memory_required(self) -> int:
         """Return the number of precision entries required in shared memory."""
         return self.compile_settings.buffer_settings.shared_memory_elements
     ```
   - Edge cases: Same as source algorithm
   - Integration: Must match source algorithm implementation

2. **Update persistent_local_required to delegate to BufferSettings**
   - File: tests/integrators/algorithms/instrumented/generic_firk.py
   - Action: Modify
   - Details:
     Replace lines 606-610:
     ```python
     @property
     def persistent_local_required(self) -> int:
         """Return the number of persistent local entries required."""

         return 0
     ```
     With:
     ```python
     @property
     def persistent_local_required(self) -> int:
         """Return the number of persistent local entries required."""
         buffer_settings = self.compile_settings.buffer_settings
         return buffer_settings.persistent_local_elements
     ```
   - Edge cases: None
   - Integration: Must match source algorithm implementation

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 11: Verification - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 5, 6, 7, 8, 9, 10

**Required Context**:
- All modified files from previous groups

**Input Validation Required**:
- None

**Tasks**:
1. **Run linter to verify no syntax errors**
   - Action: Execute `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
   - Details: Verify all modified files pass basic syntax checks
   - Edge cases: None
   - Integration: Pre-test validation

2. **Run existing tests to verify no regressions**
   - Action: Execute `pytest -m "not nocudasim and not cupy" -x`
   - Details: Run CPU-compatible tests to verify changes don't break existing functionality
   - Edge cases: Some tests may fail due to CUDA requirements - focus on algorithm-related tests
   - Integration: Regression testing

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Critical Notes for Implementation

### Calculation Mismatch Risk
The agent_plan.md mentions that using buffer_settings previously caused "invalid address" CUDA errors. Before implementing delegation, verify that BufferSettings calculations EXACTLY match the current manual calculations:

**Rosenbrock Manual**: `stage_count * n + cached_auxiliary_count + n`
**Rosenbrock BufferSettings**: Sums based on location flags (stage_rhs, stage_store, cached_auxiliaries)

**DIRK Manual**: `(stage_count - 1) * n + solver_shared_elements + cached_auxiliary_count`
**DIRK BufferSettings**: Sums based on location flags (NO cached_auxiliary_count currently)
- **ISSUE**: DIRKBufferSettings does NOT include cached_auxiliary_count in shared_memory_elements

**FIRK Manual**: `solver_shared_elements + stage_count * n_drivers + all_stages_n`
**FIRK BufferSettings**: Sums based on location flags (matches when all shared)

### DIRKBufferSettings Enhancement Required
Before Group 6 can be implemented, DIRKBufferSettings.shared_memory_elements must be enhanced to include cached_auxiliary_count. This may require:
1. Adding cached_auxiliary_count attribute to DIRKBufferSettings
2. Adding cached_auxiliaries_location attribute
3. Updating shared_memory_elements to include cached auxiliaries

**This is flagged as a potential blocker** - taskmaster should verify or implement this enhancement before Group 6.

---

## Summary

| Group | Description | Status | Parallelizable |
|-------|-------------|--------|----------------|
| 1 | BufferSettings ABC enhancement | [ ] | No |
| 2 | RosenbrockBufferSettings persistent_local | [ ] | After 1 |
| 3 | FIRKBufferSettings persistent_local | [ ] | After 1 |
| 4 | DIRKBufferSettings verification | [ ] | After 1 |
| 5 | GenericRosenbrockWStep delegation | [ ] | After 2 |
| 6 | GenericDIRKStep delegation | [ ] | After 4 |
| 7 | GenericFIRKStep delegation | [ ] | After 3 |
| 8 | Instrumented Rosenbrock | [ ] | After 5 |
| 9 | Instrumented DIRK | [ ] | After 6 |
| 10 | Instrumented FIRK | [ ] | After 7 |
| 11 | Verification | [ ] | After 8, 9, 10 |

**Total Task Groups**: 11
**Dependency Chain**: 1 → (2, 3, 4) → (5, 6, 7) → (8, 9, 10) → 11
**Parallel Execution Opportunities**: Groups 2, 3, 4 can run in parallel after Group 1. Groups 5, 6, 7 can run in parallel after their respective dependencies. Groups 8, 9, 10 can run in parallel.
**Estimated Complexity**: Medium - primarily property modifications with one potential BufferSettings enhancement for DIRK.
