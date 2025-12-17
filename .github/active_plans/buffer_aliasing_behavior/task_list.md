# Implementation Task List
# Feature: Buffer Aliasing Behavior Fix for RosenbrockBufferSettings
# Plan Reference: .github/active_plans/buffer_aliasing_behavior/agent_plan.md

## Task Group 1: Update RosenbrockSliceIndices Class - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 80-100)

**Input Validation Required**:
- None (attrs defines validator automatically for slice type)

**Tasks**:
1. **Add linear_solver slice attribute to RosenbrockSliceIndices**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Location: Lines 80-100 (RosenbrockSliceIndices class)
   - Details:
     ```python
     @attrs.define
     class RosenbrockSliceIndices(SliceIndices):
         """Slice container for Rosenbrock shared memory buffer layouts.

         Attributes
         ----------
         stage_rhs : slice
             Slice covering the stage RHS buffer (empty if local).
         stage_store : slice
             Slice covering the stage store buffer.
         cached_auxiliaries : slice
             Slice covering the cached auxiliaries buffer.
         linear_solver : slice
             Slice covering the linear solver's shared memory region.
         local_end : int
             Offset of the end of algorithm-managed shared memory.
         """

         stage_rhs: slice = attrs.field()
         stage_store: slice = attrs.field()
         cached_auxiliaries: slice = attrs.field()
         linear_solver: slice = attrs.field()
         local_end: int = attrs.field()
     ```
   - Edge cases: Empty slice (slice(0, 0)) when linear solver uses local memory
   - Integration: This attribute is used by the RosenbrockBufferSettings.shared_indices property

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 80-102)
- Functions/Methods Added/Modified:
  * RosenbrockSliceIndices class - added `linear_solver: slice = attrs.field()` attribute
- Implementation Summary:
  Added linear_solver slice attribute after cached_auxiliaries in RosenbrockSliceIndices class. Updated docstring to document the new attribute.
- Issues Flagged: None

---

## Task Group 2: Update Memory Accounting Properties - PARALLEL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 180-210)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 59-113 for LinearSolverBufferSettings)

**Input Validation Required**:
- None (linear_solver_buffer_settings already validated by attrs in __attrs_post_init__)

**Tasks**:
1. **Update shared_memory_elements property to include linear solver's shared memory**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Location: Lines 180-194 (shared_memory_elements property)
   - Details:
     ```python
     @property
     def shared_memory_elements(self) -> int:
         """Return total shared memory elements required.

         Includes stage_rhs, stage_store, cached_auxiliaries,
         and linear solver buffers if configured for shared memory.
         """
         total = 0
         if self.use_shared_stage_rhs:
             total += self.n
         if self.use_shared_stage_store:
             total += self.stage_store_elements
         if self.use_shared_cached_auxiliaries:
             total += self.cached_auxiliary_count
         if self.linear_solver_buffer_settings is not None:
             total += self.linear_solver_buffer_settings.shared_memory_elements
         return total
     ```
   - Edge cases: 
     - When linear_solver_buffer_settings is None (handled by explicit check)
     - When linear solver has 0 shared elements (returns 0, correct)
   - Integration: Used by GenericRosenbrockWStep.shared_memory_required property

2. **Update local_memory_elements property to include linear solver's local memory**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Location: Lines 196-210 (local_memory_elements property)
   - Details:
     ```python
     @property
     def local_memory_elements(self) -> int:
         """Return total local memory elements required.

         Includes buffers configured with location='local' and
         linear solver local buffers.
         """
         total = 0
         if not self.use_shared_stage_rhs:
             total += self.n
         if not self.use_shared_stage_store:
             total += self.stage_store_elements
         if not self.use_shared_cached_auxiliaries:
             total += self.cached_auxiliary_count
         if self.linear_solver_buffer_settings is not None:
             total += self.linear_solver_buffer_settings.local_memory_elements
         return total
     ```
   - Edge cases:
     - When linear_solver_buffer_settings is None (handled by explicit check)
     - When linear solver has 0 local elements (returns 0, correct)
   - Integration: Used by GenericRosenbrockWStep.local_scratch_required property

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 183-217)
- Functions/Methods Added/Modified:
  * shared_memory_elements property - added linear solver shared memory accounting
  * local_memory_elements property - added linear solver local memory accounting
- Implementation Summary:
  Added conditional checks to both shared_memory_elements and local_memory_elements properties to include linear_solver_buffer_settings memory elements when not None. Updated docstrings to document linear solver inclusion.
- Issues Flagged: None

---

## Task Group 3: Update shared_indices Property - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 1 (requires linear_solver attribute in RosenbrockSliceIndices)

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 224-259)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 95-102 for LinearSolverBufferSettings.shared_memory_elements)

**Input Validation Required**:
- None (linear_solver_buffer_settings already validated)

**Tasks**:
1. **Add linear_solver slice computation to shared_indices property**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Location: Lines 224-259 (shared_indices property)
   - Details:
     ```python
     @property
     def shared_indices(self) -> RosenbrockSliceIndices:
         """Return RosenbrockSliceIndices instance with shared memory layout.

         The returned object contains slices for each buffer's region
         in shared memory. Local buffers receive empty slices.
         """
         ptr = 0

         if self.use_shared_stage_rhs:
             stage_rhs_slice = slice(ptr, ptr + self.n)
             ptr += self.n
         else:
             stage_rhs_slice = slice(0, 0)

         if self.use_shared_stage_store:
             stage_store_slice = slice(ptr, ptr + self.stage_store_elements)
             ptr += self.stage_store_elements
         else:
             stage_store_slice = slice(0, 0)

         if self.use_shared_cached_auxiliaries:
             cached_auxiliaries_slice = slice(
                 ptr, ptr + self.cached_auxiliary_count
             )
             ptr += self.cached_auxiliary_count
         else:
             cached_auxiliaries_slice = slice(0, 0)

         if (self.linear_solver_buffer_settings is not None and
                 self.linear_solver_buffer_settings.shared_memory_elements > 0):
             lin_solver_shared = (
                 self.linear_solver_buffer_settings.shared_memory_elements
             )
             linear_solver_slice = slice(ptr, ptr + lin_solver_shared)
             ptr += lin_solver_shared
         else:
             linear_solver_slice = slice(0, 0)

         return RosenbrockSliceIndices(
             stage_rhs=stage_rhs_slice,
             stage_store=stage_store_slice,
             cached_auxiliaries=cached_auxiliaries_slice,
             linear_solver=linear_solver_slice,
             local_end=ptr,
         )
     ```
   - Edge cases:
     - linear_solver_buffer_settings is None: returns empty slice
     - linear_solver has 0 shared elements: returns empty slice
     - All Rosenbrock buffers local but linear solver shared: linear_solver starts at 0
   - Integration: Used in GenericRosenbrockWStep.build_step() to access shared memory layout

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 232-277)
- Functions/Methods Added/Modified:
  * shared_indices property - added linear_solver slice computation and constructor argument
- Implementation Summary:
  Added conditional block after cached_auxiliaries_slice computation to calculate linear_solver_slice. When linear_solver_buffer_settings is not None and has shared memory elements > 0, computes the slice starting at current pointer. Otherwise returns empty slice(0, 0). Added linear_solver=linear_solver_slice to RosenbrockSliceIndices constructor call.
- Issues Flagged: None

---

## Task Group 4: Add Tests for RosenbrockBufferSettings Linear Solver Accounting - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Groups 1, 2, 3

**Required Context**:
- File: tests/integrators/algorithms/test_buffer_settings.py (entire file, especially lines 382-475 TestRosenbrockBufferSettings)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (RosenbrockBufferSettings, RosenbrockSliceIndices)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (LinearSolverBufferSettings)

**Input Validation Required**:
- None (tests use valid fixture values)

**Tasks**:
1. **Add test for Rosenbrock shared_memory_elements including linear solver**
   - File: tests/integrators/algorithms/test_buffer_settings.py
   - Action: Modify
   - Location: After line 420 (after test_shared_memory_elements test)
   - Details:
     ```python
     def test_shared_memory_elements_includes_linear_solver(self):
         """Shared memory should include linear solver's shared elements."""
         linear_settings = LinearSolverBufferSettings(
             n=3,
             preconditioned_vec_location='shared',
             temp_location='shared',
         )
         settings = RosenbrockBufferSettings(
             n=3,
             stage_count=4,
             cached_auxiliary_count=10,
             stage_rhs_location='shared',
             stage_store_location='shared',
             cached_auxiliaries_location='shared',
             linear_solver_buffer_settings=linear_settings,
         )
         # rhs (3) + store (12) + aux (10) + linear_solver (3+3=6) = 31
         assert settings.shared_memory_elements == 31
     ```
   - Edge cases: Tested in a separate test below
   - Integration: Validates fix from Task Group 2

2. **Add test for Rosenbrock local_memory_elements including linear solver**
   - File: tests/integrators/algorithms/test_buffer_settings.py
   - Action: Modify
   - Location: After line 433 (after test_local_memory_elements test)
   - Details:
     ```python
     def test_local_memory_elements_includes_linear_solver(self):
         """Local memory should include linear solver's local elements."""
         linear_settings = LinearSolverBufferSettings(
             n=3,
             preconditioned_vec_location='local',
             temp_location='local',
         )
         settings = RosenbrockBufferSettings(
             n=3,
             stage_count=4,
             cached_auxiliary_count=10,
             stage_rhs_location='local',
             stage_store_location='local',
             cached_auxiliaries_location='local',
             linear_solver_buffer_settings=linear_settings,
         )
         # rhs (3) + store (12) + aux (10) + linear_solver (3+3=6) = 31
         assert settings.local_memory_elements == 31
     ```
   - Edge cases: Tested in a separate test below
   - Integration: Validates fix from Task Group 2

3. **Add test for Rosenbrock shared_indices including linear_solver slice**
   - File: tests/integrators/algorithms/test_buffer_settings.py
   - Action: Modify
   - Location: After line 465 (after test_shared_indices_property test)
   - Details:
     ```python
     def test_shared_indices_includes_linear_solver_slice(self):
         """shared_indices should include linear_solver slice."""
         linear_settings = LinearSolverBufferSettings(
             n=3,
             preconditioned_vec_location='shared',
             temp_location='shared',
         )
         settings = RosenbrockBufferSettings(
             n=3,
             stage_count=4,
             cached_auxiliary_count=10,
             stage_rhs_location='shared',
             stage_store_location='shared',
             cached_auxiliaries_location='shared',
             linear_solver_buffer_settings=linear_settings,
         )
         indices = settings.shared_indices

         assert indices.stage_rhs == slice(0, 3)
         assert indices.stage_store == slice(3, 15)
         assert indices.cached_auxiliaries == slice(15, 25)
         # linear_solver: 6 elements (3 precond + 3 temp)
         assert indices.linear_solver == slice(25, 31)
         assert indices.local_end == 31
     ```
   - Edge cases: Tested in a separate test below
   - Integration: Validates fix from Task Group 3

4. **Add test for linear_solver slice when linear solver uses local memory**
   - File: tests/integrators/algorithms/test_buffer_settings.py
   - Action: Modify
   - Location: After the test_shared_indices_includes_linear_solver_slice test
   - Details:
     ```python
     def test_shared_indices_linear_solver_slice_empty_when_local(self):
         """linear_solver slice should be empty when solver uses local."""
         linear_settings = LinearSolverBufferSettings(
             n=3,
             preconditioned_vec_location='local',
             temp_location='local',
         )
         settings = RosenbrockBufferSettings(
             n=3,
             stage_count=4,
             cached_auxiliary_count=10,
             stage_rhs_location='shared',
             stage_store_location='shared',
             cached_auxiliaries_location='shared',
             linear_solver_buffer_settings=linear_settings,
         )
         indices = settings.shared_indices

         # linear_solver uses local, so slice is empty
         assert indices.linear_solver == slice(0, 0)
         # local_end is 25 (rhs=3 + store=12 + aux=10)
         assert indices.local_end == 25
     ```
   - Edge cases: Covers case where linear solver is local but Rosenbrock is shared
   - Integration: Validates correct slice behavior for mixed configurations

5. **Add test verifying no double-counting in mixed configuration**
   - File: tests/integrators/algorithms/test_buffer_settings.py
   - Action: Modify
   - Location: After the test_shared_indices_linear_solver_slice_empty_when_local test
   - Details:
     ```python
     def test_memory_accounting_no_double_counting(self):
         """Memory accounting should not double-count between shared/local."""
         linear_settings = LinearSolverBufferSettings(
             n=3,
             preconditioned_vec_location='shared',
             temp_location='local',
         )
         settings = RosenbrockBufferSettings(
             n=3,
             stage_count=4,
             cached_auxiliary_count=10,
             stage_rhs_location='shared',
             stage_store_location='local',
             cached_auxiliaries_location='shared',
             linear_solver_buffer_settings=linear_settings,
         )
         # shared: rhs (3) + aux (10) + linear precond (3) = 16
         # local: store (12) + linear temp (3) = 15
         assert settings.shared_memory_elements == 16
         assert settings.local_memory_elements == 15
         # Total should equal all elements counted once
         total_elements = 3 + 12 + 10 + 3 + 3  # rhs + store + aux + precond + temp
         assert (settings.shared_memory_elements + 
                 settings.local_memory_elements) == total_elements
     ```
   - Edge cases: Mixed configuration with some buffers shared, some local
   - Integration: Validates user story US-4 (non-duplicative memory accounting)

**Outcomes**: 
- Files Modified: 
  * tests/integrators/algorithms/test_buffer_settings.py (lines 476-585)
- Functions/Methods Added/Modified:
  * test_shared_memory_elements_includes_linear_solver() - new test method
  * test_local_memory_elements_includes_linear_solver() - new test method
  * test_shared_indices_includes_linear_solver_slice() - new test method
  * test_shared_indices_linear_solver_slice_empty_when_local() - new test method
  * test_memory_accounting_no_double_counting() - new test method
- Implementation Summary:
  Added 5 new test methods to TestRosenbrockBufferSettings class. All tests verify the linear solver memory accounting fixes: shared memory inclusion, local memory inclusion, shared_indices slice generation, empty slice when local, and no double-counting.
- Issues Flagged: None

---

## Summary

### Total Task Groups: 4
### Dependency Chain:
1. Task Group 1 (RosenbrockSliceIndices) - **No dependencies**
2. Task Group 2 (Memory accounting properties) - **No dependencies** (can run in parallel with Group 1)
3. Task Group 3 (shared_indices property) - **Depends on Group 1**
4. Task Group 4 (Tests) - **Depends on Groups 1, 2, 3**

### Parallel Execution Opportunities:
- Task Groups 1 and 2 can be executed in parallel
- Task Group 3 must wait for Group 1
- Task Group 4 must wait for all prior groups

### Estimated Complexity:
- Task Group 1: Low (add one attribute)
- Task Group 2: Low (add two conditionals to existing properties)
- Task Group 3: Medium (add slice computation with proper offset tracking)
- Task Group 4: Medium (5 new tests with various configurations)

### Files Modified:
1. `src/cubie/integrators/algorithms/generic_rosenbrock_w.py` - 3 changes
2. `tests/integrators/algorithms/test_buffer_settings.py` - 5 new tests

### Key Implementation Notes:
- All changes follow existing patterns in DIRKBufferSettings, FIRKBufferSettings, and NewtonBufferSettings
- The linear solver slice comes AFTER all Rosenbrock-specific buffers in shared memory layout
- Empty slices (slice(0, 0)) are used when a buffer uses local memory, consistent with existing convention
- No changes to the step function (`build_step`) are needed; it already passes `shared` to the linear solver
