# Implementation Task List
# Feature: Consolidate Memory Sizing Properties to Base CUDAFactory
# Plan Reference: .github/active_plans/consolidate_memory_sizing/agent_plan.md

## Task Group 1: Add Core Memory Sizing Properties to CUDAFactory Base Class
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/CUDAFactory.py (entire file - 737 lines)
- File: src/cubie/buffer_registry.py (lines 746-795 - buffer_registry sizing methods)

**Input Validation Required**:
- None - properties delegate to buffer_registry which handles all edge cases

**Tasks**:
1. **Add buffer_registry import to CUDAFactory.py**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     Add import statement at top of file after existing imports:
     ```python
     from cubie.buffer_registry import buffer_registry
     ```
   - Edge cases: None - buffer_registry is a singleton module
   - Integration: Import goes with other cubie imports

2. **Add shared_buffer_size property to CUDAFactory class**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     Add property after the `simsafe_precision` property (around line 572):
     ```python
     @property
     def shared_buffer_size(self) -> int:
         """Return total shared memory elements registered for this factory.

         Returns
         -------
         int
             Total shared memory elements from buffer_registry.
         """
         return buffer_registry.shared_buffer_size(self)
     ```
   - Edge cases: Returns 0 if no buffers registered (handled by buffer_registry)
   - Integration: All CUDAFactory subclasses inherit this property automatically

3. **Add local_buffer_size property to CUDAFactory class**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     Add property after shared_buffer_size:
     ```python
     @property
     def local_buffer_size(self) -> int:
         """Return total local memory elements registered for this factory.

         Returns
         -------
         int
             Total local memory elements from buffer_registry.
         """
         return buffer_registry.local_buffer_size(self)
     ```
   - Edge cases: Returns 0 if no buffers registered
   - Integration: All CUDAFactory subclasses inherit this property automatically

4. **Add persistent_local_buffer_size property to CUDAFactory class**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     Add property after local_buffer_size:
     ```python
     @property
     def persistent_local_buffer_size(self) -> int:
         """Return total persistent local elements registered for this factory.

         Returns
         -------
         int
             Total persistent local elements from buffer_registry.
         """
         return buffer_registry.persistent_local_buffer_size(self)
     ```
   - Edge cases: Returns 0 if no buffers registered
   - Integration: All CUDAFactory subclasses inherit this property automatically

**Tests to Create**:
- None for this task group - base class changes tested via subclass tests

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py
- tests/integrators/algorithms/test_step_algorithms.py

**Outcomes**: 
- Files Modified: 
  * src/cubie/CUDAFactory.py (34 lines added)
- Functions/Methods Added/Modified:
  * shared_buffer_size property in CUDAFactory class
  * local_buffer_size property in CUDAFactory class
  * persistent_local_buffer_size property in CUDAFactory class
- Implementation Summary:
  Added import for buffer_registry singleton and three new properties to the
  CUDAFactory base class that delegate to buffer_registry methods. These
  properties provide a unified interface for accessing memory sizing
  information for all CUDAFactory subclasses.
- Issues Flagged: None

---

## Task Group 2: Remove Redundant Properties from IVPLoop
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 870-895 - properties to remove)

**Input Validation Required**:
- None - only removing code

**Tasks**:
1. **Remove shared_memory_elements property from IVPLoop**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 874-877):
     ```python
     @property
     def shared_memory_elements(self) -> int:
         """Return the loop's shared-memory requirement."""
         return buffer_registry.shared_buffer_size(self)
     ```
   - Edge cases: None
   - Integration: IVPLoop now inherits shared_buffer_size from CUDAFactory base class

2. **Remove local_memory_elements property from IVPLoop**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 879-882):
     ```python
     @property
     def local_memory_elements(self) -> int:
         """Return the loop's persistent local-memory requirement."""
         return buffer_registry.local_buffer_size(self)
     ```
   - Edge cases: None
   - Integration: IVPLoop now inherits local_buffer_size from CUDAFactory base class

3. **Remove persistent_local_elements property from IVPLoop**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 884-887):
     ```python
     @property
     def persistent_local_elements(self) -> int:
         """Return the loop's persistent local-memory requirement."""
         return buffer_registry.persistent_local_buffer_size(self)
     ```
   - Edge cases: None
   - Integration: IVPLoop now inherits persistent_local_buffer_size from CUDAFactory base class

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/loops/ode_loop.py (15 lines removed)
- Functions/Methods Added/Modified:
  * Removed shared_memory_elements property from IVPLoop class
  * Removed local_memory_elements property from IVPLoop class
  * Removed persistent_local_elements property from IVPLoop class
- Implementation Summary:
  Removed three redundant properties that were delegating to buffer_registry.
  IVPLoop now inherits shared_buffer_size, local_buffer_size, and
  persistent_local_buffer_size from the CUDAFactory base class instead.
- Issues Flagged: None

---

## Task Group 3: Remove Redundant Properties from BaseAlgorithmStep
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (lines 635-665 - properties to remove)

**Input Validation Required**:
- None - only removing code

**Tasks**:
1. **Remove shared_memory_elements property from BaseAlgorithmStep**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 640-643):
     ```python
     @property
     def shared_memory_elements(self) -> int:
         """Return the precision-entry count of shared memory required."""
         return buffer_registry.shared_buffer_size(self)
     ```
   - Edge cases: None
   - Integration: BaseAlgorithmStep now inherits shared_buffer_size from CUDAFactory

2. **Remove local_scratch_elements property from BaseAlgorithmStep**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 645-648):
     ```python
     @property
     def local_scratch_elements(self) -> int:
         """Return the precision-entry count of local scratch required."""
         return buffer_registry.local_buffer_size(self)
     ```
   - Edge cases: None
   - Integration: BaseAlgorithmStep now inherits local_buffer_size from CUDAFactory

3. **Remove persistent_local_elements property from BaseAlgorithmStep**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 650-653):
     ```python
     @property
     def persistent_local_elements(self) -> int:
         """Return the persistent local precision-entry requirement."""
         return buffer_registry.persistent_local_buffer_size(self)
     ```
   - Edge cases: None
   - Integration: BaseAlgorithmStep now inherits persistent_local_buffer_size from CUDAFactory

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/test_step_algorithms.py

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/algorithms/base_algorithm_step.py (15 lines removed)
- Functions/Methods Added/Modified:
  * Removed shared_memory_elements property from BaseAlgorithmStep class
  * Removed local_scratch_elements property from BaseAlgorithmStep class
  * Removed persistent_local_elements property from BaseAlgorithmStep class
- Implementation Summary:
  Removed three redundant properties that were delegating to buffer_registry.
  BaseAlgorithmStep now inherits shared_buffer_size, local_buffer_size, and
  persistent_local_buffer_size from the CUDAFactory base class instead.
- Issues Flagged: None

---

## Task Group 4: Remove Redundant Properties from LinearSolver
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 570-595 - properties to remove)

**Input Validation Required**:
- None - only removing code

**Tasks**:
1. **Remove shared_buffer_size property from LinearSolver**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 570-573):
     ```python
     @property
     def shared_buffer_size(self) -> int:
         """Return total shared memory elements required."""
         return buffer_registry.shared_buffer_size(self)
     ```
   - Edge cases: None
   - Integration: LinearSolver now inherits shared_buffer_size from CUDAFactory

2. **Remove local_buffer_size property from LinearSolver**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 575-578):
     ```python
     @property
     def local_buffer_size(self) -> int:
         """Return total local memory elements required."""
         return buffer_registry.local_buffer_size(self)
     ```
   - Edge cases: None
   - Integration: LinearSolver now inherits local_buffer_size from CUDAFactory

3. **Remove persistent_local_buffer_size property from LinearSolver**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 580-583):
     ```python
     @property
     def persistent_local_buffer_size(self) -> int:
         """Return total persistent local memory elements required."""
         return buffer_registry.persistent_local_buffer_size(self)
     ```
   - Edge cases: None
   - Integration: LinearSolver now inherits persistent_local_buffer_size from CUDAFactory

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_linear_solver.py (if exists)

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (15 lines removed)
- Functions/Methods Added/Modified:
  * Removed shared_buffer_size property from LinearSolver class
  * Removed local_buffer_size property from LinearSolver class
  * Removed persistent_local_buffer_size property from LinearSolver class
- Implementation Summary:
  Removed three redundant properties that were delegating to buffer_registry.
  LinearSolver now inherits shared_buffer_size, local_buffer_size, and
  persistent_local_buffer_size from the CUDAFactory base class instead.
- Issues Flagged: None

---

## Task Group 5: Remove Redundant Properties from NewtonKrylovSolver
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 590-610 - properties to remove)

**Input Validation Required**:
- None - only removing code

**Tasks**:
1. **Remove local_buffer_size property from NewtonKrylovSolver**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 591-597):
     ```python
     @property
     def local_buffer_size(self) -> int:
         """Return total local memory elements required.

         Includes both Newton buffers and nested LinearSolver buffers.
         """
         return buffer_registry.local_buffer_size(self)
     ```
   - Edge cases: None
   - Integration: NewtonKrylovSolver now inherits local_buffer_size from CUDAFactory

2. **Remove persistent_local_buffer_size property from NewtonKrylovSolver**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 599-605):
     ```python
     @property
     def persistent_local_buffer_size(self) -> int:
         """Return total persistent local memory elements required.

         Includes both Newton buffers and nested LinearSolver buffers.
         """
         return buffer_registry.persistent_local_buffer_size(self)
     ```
   - Edge cases: None
   - Integration: NewtonKrylovSolver now inherits persistent_local_buffer_size from CUDAFactory

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_newton_krylov.py (if exists)

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (14 lines removed)
- Functions/Methods Added/Modified:
  * Removed local_buffer_size property from NewtonKrylovSolver class
  * Removed persistent_local_buffer_size property from NewtonKrylovSolver class
- Implementation Summary:
  Removed two redundant properties that were delegating to buffer_registry.
  NewtonKrylovSolver now inherits local_buffer_size and
  persistent_local_buffer_size from the CUDAFactory base class instead.
- Issues Flagged: None

---

## Task Group 6: Remove Legacy Child-Specific Properties from SingleIntegratorRun
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/integrators/SingleIntegratorRun.py (lines 55-100, 200-280)

**Input Validation Required**:
- None - only removing code

**Tasks**:
1. **Remove shared_memory_elements_loop property from SingleIntegratorRun**
   - File: src/cubie/integrators/SingleIntegratorRun.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 214-218):
     ```python
     @property
     def shared_memory_elements_loop(self) -> int:
         """Return the loop contribution to shared memory."""

         return self._loop.shared_memory_elements
     ```
   - Edge cases: None
   - Integration: This was a legacy property; callers should use shared_memory_elements instead

2. **Remove local_memory_elements_loop property from SingleIntegratorRun**
   - File: src/cubie/integrators/SingleIntegratorRun.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 220-224):
     ```python
     @property
     def local_memory_elements_loop(self) -> int:
         """Return the loop contribution to local memory."""

         return self._loop.local_memory_elements
     ```
   - Edge cases: None
   - Integration: This was a legacy property; callers should use local_memory_elements instead

3. **Remove local_memory_elements_controller property from SingleIntegratorRun**
   - File: src/cubie/integrators/SingleIntegratorRun.py
   - Action: Modify
   - Details:
     Remove the following property (approximately lines 265-269):
     ```python
     @property
     def local_memory_elements_controller(self) -> int:
         """Return the controller contribution to local memory."""

         return self._step_controller.local_memory_elements
     ```
   - Edge cases: None
   - Integration: This was a legacy child-specific property

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/test_SingleIntegratorRun.py (if exists)

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/SingleIntegratorRun.py (17 lines removed)
- Functions/Methods Added/Modified:
  * Removed shared_memory_elements_loop property from SingleIntegratorRun class
  * Removed local_memory_elements_loop property from SingleIntegratorRun class
  * Removed local_memory_elements_controller property from SingleIntegratorRun class
- Implementation Summary:
  Removed three legacy child-specific properties that exposed individual component
  memory contributions. These were redundant with the main aggregation properties
  (shared_memory_elements, local_memory_elements, persistent_local_elements) which
  remain available for callers to use.
- Issues Flagged: None

---

## Task Group 7: Update SingleIntegratorRun Aggregation Properties
**Status**: [x]
**Dependencies**: Task Group 2, Task Group 6

**Required Context**:
- File: src/cubie/integrators/SingleIntegratorRun.py (lines 55-100)

**Input Validation Required**:
- None - only updating property names for consistency

**Tasks**:
1. **Update shared_memory_elements to use new property name**
   - File: src/cubie/integrators/SingleIntegratorRun.py
   - Action: Modify
   - Details:
     Update the property to delegate to `_loop.shared_buffer_size` instead of `_loop.shared_memory_elements`:
     ```python
     @property
     def shared_memory_elements(self) -> int:
         """Return total shared-memory elements required by the loop."""
         return self._loop.shared_buffer_size
     ```
   - Edge cases: None
   - Integration: IVPLoop now uses shared_buffer_size inherited from CUDAFactory

2. **Update local_memory_elements to use new property name**
   - File: src/cubie/integrators/SingleIntegratorRun.py
   - Action: Modify
   - Details:
     Update the property to delegate to `_loop.local_buffer_size` instead of `_loop.local_memory_elements`:
     ```python
     @property
     def local_memory_elements(self) -> int:
         """Return total persistent local-memory requirement."""
         return self._loop.local_buffer_size
     ```
   - Edge cases: None
   - Integration: IVPLoop now uses local_buffer_size inherited from CUDAFactory

3. **Update persistent_local_elements to use new property name**
   - File: src/cubie/integrators/SingleIntegratorRun.py
   - Action: Modify
   - Details:
     Update the property to delegate to `_loop.persistent_local_buffer_size` instead of `_loop.persistent_local_elements`:
     ```python
     @property
     def persistent_local_elements(self) -> int:
         """Return total persistent local-memory elements required by the loop."""
         return self._loop.persistent_local_buffer_size
     ```
   - Edge cases: None
   - Integration: IVPLoop now uses persistent_local_buffer_size inherited from CUDAFactory

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/test_SingleIntegratorRun.py (if exists)
- tests/_utils.py references singleintegratorrun.persistent_local_elements - verify usage

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/SingleIntegratorRun.py (3 lines changed)
- Functions/Methods Added/Modified:
  * shared_memory_elements property - updated to delegate to _loop.shared_buffer_size
  * local_memory_elements property - updated to delegate to _loop.local_buffer_size
  * persistent_local_elements property - updated to delegate to _loop.persistent_local_buffer_size
- Implementation Summary:
  Updated three aggregation properties in SingleIntegratorRun to reference the new
  property names inherited by IVPLoop from CUDAFactory base class. The properties
  now delegate to shared_buffer_size, local_buffer_size, and persistent_local_buffer_size
  instead of the removed shared_memory_elements, local_memory_elements, and
  persistent_local_elements.
- Issues Flagged: None

---

## Task Group 8: Update Test Files for Renamed Properties
**Status**: [x]
**Dependencies**: Task Groups 1-7

**Required Context**:
- File: tests/integrators/algorithms/test_step_algorithms.py (lines 505-560, 600-660, 740-780)
- File: tests/_utils.py (lines 770-790)
- File: tests/integrators/algorithms/instrumented/conftest.py (lines 245-270)

**Input Validation Required**:
- None - only updating property name references

**Tasks**:
1. **Update test_step_algorithms.py expected properties dict**
   - File: tests/integrators/algorithms/test_step_algorithms.py
   - Action: Modify
   - Details:
     In the `generate_step_props` function (lines 505-557), update all occurrences of `"persistent_local_elements"` to `"persistent_local_buffer_size"`:
     ```python
     # Change all instances of:
     "persistent_local_elements": 0,
     # To:
     "persistent_local_buffer_size": 0,
     ```
   - Edge cases: None
   - Integration: Tests will now check the new property name

2. **Update test_step_algorithms.py device_step_results test**
   - File: tests/integrators/algorithms/test_step_algorithms.py
   - Action: Modify
   - Details:
     Update line ~612 from:
     ```python
     persistent_len = max(1, step_object.persistent_local_elements)
     ```
     To:
     ```python
     persistent_len = max(1, step_object.persistent_local_buffer_size)
     ```
   - Edge cases: None
   - Integration: Uses the new base class property

3. **Update test_step_algorithms.py test_algorithm_determinism test**
   - File: tests/integrators/algorithms/test_step_algorithms.py
   - Action: Modify
   - Details:
     Update line ~753 from:
     ```python
     persistent_len = max(1, step_object.persistent_local_elements)
     ```
     To:
     ```python
     persistent_len = max(1, step_object.persistent_local_buffer_size)
     ```
   - Edge cases: None
   - Integration: Uses the new base class property

4. **Update tests/_utils.py run_loop_on_device function**
   - File: tests/_utils.py
   - Action: Modify
   - Details:
     Update line ~776 from:
     ```python
     persistent_required = max(1,singleintegratorrun.persistent_local_elements)
     ```
     To:
     ```python
     persistent_required = max(1,singleintegratorrun.persistent_local_elements)
     ```
     Note: This stays the same because SingleIntegratorRun keeps `persistent_local_elements` property (it delegates to _loop.persistent_local_buffer_size)
   - Edge cases: None
   - Integration: SingleIntegratorRun still exposes persistent_local_elements for API compatibility

5. **Update instrumented conftest.py**
   - File: tests/integrators/algorithms/instrumented/conftest.py
   - Action: Modify
   - Details:
     Update line ~252 from:
     ```python
     persistent_len = max(1, instrumented_step_object.persistent_local_elements)
     ```
     To:
     ```python
     persistent_len = max(1, instrumented_step_object.persistent_local_buffer_size)
     ```
   - Edge cases: None
   - Integration: Uses the new base class property

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/test_step_algorithms.py
- tests/integrators/algorithms/instrumented/test_instrumented.py

**Outcomes**: 
- Files Modified: 
  * tests/integrators/algorithms/test_step_algorithms.py (10 lines changed)
  * tests/integrators/algorithms/instrumented/conftest.py (2 lines changed)
- Functions/Methods Added/Modified:
  * generate_step_props() in test_step_algorithms.py - updated 7 property dict entries
  * device_step_results() in test_step_algorithms.py - updated property references
  * _execute_step_twice() in test_step_algorithms.py - updated property references
  * device_step_results fixture in instrumented/conftest.py - updated property references
- Implementation Summary:
  Updated all test files to reference the new property names from CUDAFactory base
  class. Changed `persistent_local_elements` to `persistent_local_buffer_size` and
  `shared_memory_elements` to `shared_buffer_size` in test_step_algorithms.py and
  instrumented/conftest.py. The tests/_utils.py file was left unchanged because
  SingleIntegratorRun still exposes `persistent_local_elements` property for API
  compatibility (it delegates to _loop.persistent_local_buffer_size internally).
- Issues Flagged: None

---

## Summary

### Total Task Groups: 8

### Dependency Chain:
```
Task Group 1 (CUDAFactory base class)
    ├── Task Group 2 (IVPLoop cleanup)
    │       └── Task Group 6 (SingleIntegratorRun legacy removal)
    │               └── Task Group 7 (SingleIntegratorRun updates)
    ├── Task Group 3 (BaseAlgorithmStep cleanup)
    ├── Task Group 4 (LinearSolver cleanup)
    └── Task Group 5 (NewtonKrylovSolver cleanup)
            └── Task Group 8 (Test updates) - depends on all above
```

### Tests to be Created: None

### Tests to be Run:
- tests/integrators/loops/test_ode_loop.py
- tests/integrators/algorithms/test_step_algorithms.py
- tests/integrators/algorithms/instrumented/test_instrumented.py

### Estimated Complexity:
- Task Group 1: Low (add 4 items: 1 import + 3 properties)
- Task Group 2: Low (remove 3 properties)
- Task Group 3: Low (remove 3 properties)
- Task Group 4: Low (remove 3 properties)
- Task Group 5: Low (remove 2 properties)
- Task Group 6: Low (remove 3 properties)
- Task Group 7: Low (update 3 property delegations)
- Task Group 8: Low (update ~5 property references in tests)

Total: ~25 small surgical changes across 10 files
