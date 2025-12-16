# Implementation Task List
# Feature: BufferSettings Review Fixes
# Plan Reference: .github/active_plans/buffer_settings_review_fixes/agent_plan.md

## Task Group 1: NewtonBufferSettings - Add residual_temp Toggleability - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 23-168)

**Input Validation Required**:
- residual_temp_location: Validate in ["local", "shared"] (use attrs validators.in_)

**Tasks**:

1. **Add residual_temp slice to NewtonSliceIndices**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     @attrs.define
     class NewtonSliceIndices(SliceIndices):
         """Slice container for Newton solver shared memory layouts.
     
         Attributes
         ----------
         delta : slice
             Slice covering the delta buffer (empty if local).
         residual : slice
             Slice covering the residual buffer (empty if local).
         residual_temp : slice
             Slice covering the residual_temp buffer (empty if local).
         local_end : int
             Offset of the end of Newton-managed shared memory.
         lin_solver_start : int
             Start offset for linear solver shared memory.
         """
     
         delta: slice = attrs.field()
         residual: slice = attrs.field()
         residual_temp: slice = attrs.field()  # ADD THIS
         local_end: int = attrs.field()
         lin_solver_start: int = attrs.field()
     ```
   - Edge cases: None - simple addition of attribute
   - Integration: Used by newton_krylov_solver_factory for memory allocation

2. **Add residual_temp_location attribute to NewtonBufferSettings**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     Add after line 92 (residual_location):
     ```python
     residual_temp_location: str = attrs.field(
         default='local', validator=validators.in_(["local", "shared"])
     )
     ```
   - Edge cases: Default 'local' maintains current behavior
   - Integration: Must add before linear_solver_buffer_settings attribute

3. **Add use_shared_residual_temp property to NewtonBufferSettings**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     Add after use_shared_residual property (after line 105):
     ```python
     @property
     def use_shared_residual_temp(self) -> bool:
         """Return True if residual_temp buffer uses shared memory."""
         return self.residual_temp_location == 'shared'
     ```
   - Edge cases: None
   - Integration: Used by shared_memory_elements and local_memory_elements

4. **Update shared_memory_elements to include residual_temp when shared**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     Modify shared_memory_elements property (lines 107-118) to add:
     ```python
     @property
     def shared_memory_elements(self) -> int:
         """Return total shared memory elements required."""
         total = 0
         if self.use_shared_delta:
             total += self.n
         if self.use_shared_residual:
             total += self.n
         if self.use_shared_residual_temp:
             total += self.n
         # Add linear solver shared memory
         if self.linear_solver_buffer_settings is not None:
             total += self.linear_solver_buffer_settings.shared_memory_elements
         return total
     ```
   - Edge cases: When residual_temp is local (default), no change in behavior
   - Integration: Affects memory accounting upstream

5. **Update local_memory_elements to use conditional for residual_temp**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     Modify local_memory_elements property (lines 120-134):
     ```python
     @property
     def local_memory_elements(self) -> int:
         """Return total local memory elements required."""
         total = 0
         if not self.use_shared_delta:
             total += self.n
         if not self.use_shared_residual:
             total += self.n
         # residual_temp conditional on location
         if not self.use_shared_residual_temp:
             total += self.n
         total += 1       # krylov_iters (int32, but counted as 1 element)
         # Add linear solver local memory
         if self.linear_solver_buffer_settings is not None:
             total += self.linear_solver_buffer_settings.local_memory_elements
         return total
     ```
   - Edge cases: Old hardcoded `total += self.n` replaced with conditional
   - Integration: Affects memory accounting upstream

6. **Update shared_indices to return residual_temp slice**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     Modify shared_indices property (lines 146-167):
     ```python
     @property
     def shared_indices(self) -> NewtonSliceIndices:
         """Return NewtonSliceIndices instance with shared memory layout."""
         ptr = 0
         if self.use_shared_delta:
             delta_slice = slice(ptr, ptr + self.n)
             ptr += self.n
         else:
             delta_slice = slice(0, 0)
     
         if self.use_shared_residual:
             residual_slice = slice(ptr, ptr + self.n)
             ptr += self.n
         else:
             residual_slice = slice(0, 0)
     
         if self.use_shared_residual_temp:
             residual_temp_slice = slice(ptr, ptr + self.n)
             ptr += self.n
         else:
             residual_temp_slice = slice(0, 0)
     
         return NewtonSliceIndices(
             delta=delta_slice,
             residual=residual_slice,
             residual_temp=residual_temp_slice,
             local_end=ptr,
             lin_solver_start=ptr,
         )
     ```
   - Edge cases: When local, returns slice(0, 0)
   - Integration: Slice must be passed to NewtonSliceIndices constructor

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (6 changes)
- Functions/Methods Added/Modified:
  * NewtonSliceIndices: added residual_temp slice attribute
  * NewtonBufferSettings: added residual_temp_location attribute
  * NewtonBufferSettings: added use_shared_residual_temp property
  * shared_memory_elements: updated to include residual_temp conditional
  * local_memory_elements: updated residual_temp to be conditional
  * shared_indices: updated to return residual_temp slice
- Implementation Summary: Added residual_temp_location toggleability with attrs validator
- Issues Flagged: None

---

## Task Group 2: DIRKBufferSettings - Remove solver_scratch_location - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 110-335)

**Input Validation Required**:
- None - removing attribute, not adding

**Tasks**:

1. **Remove solver_scratch_location attribute from DIRKBufferSettings**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Remove lines 146-148:
     ```python
     # DELETE THESE LINES:
     solver_scratch_location: str = attrs.field(
         default='local', validator=validators.in_(["local", "shared"])
     )
     ```
   - Edge cases: None
   - Integration: Must also update properties and calculations that use this

2. **Remove use_shared_solver_scratch property**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Remove lines 169-171:
     ```python
     # DELETE THESE LINES:
     @property
     def use_shared_solver_scratch(self) -> bool:
         """Return True if solver_scratch buffer uses shared memory."""
         return self.solver_scratch_location == 'shared'
     ```
   - Edge cases: None
   - Integration: Properties that use this need to be updated

3. **Update solver_scratch_elements property to remove fallback**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Modify lines 179-189:
     ```python
     @property
     def solver_scratch_elements(self) -> int:
         """Return the number of solver scratch elements.

         Returns newton_buffer_settings.shared_memory_elements.
         """
         return self.newton_buffer_settings.shared_memory_elements
     ```
   - Edge cases: newton_buffer_settings is now required, so no None check needed
   - Integration: Depends on newton_buffer_settings being required

4. **Update persistent_local_elements property**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Remove solver_scratch_location check (lines 191-203):
     ```python
     @property
     def persistent_local_elements(self) -> int:
         """Return persistent local elements for increment_cache and rhs_cache.

         Returns 2 * n since solver_scratch is always shared from parent.
         increment_cache and rhs_cache share the solver_scratch region.
         """
         return 0
     ```
     
     NOTE: Actually reviewing the code more carefully - increment_cache and
     rhs_cache are used for FSAL optimization. When solver_scratch is shared,
     they alias it. Since solver_scratch_location is removed (always implicitly
     shared from parent), persistent_local_elements should always be 0.
   - Edge cases: None
   - Integration: Affects DIRKStep.persistent_local_required

5. **Update shared_memory_elements property**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Remove solver_scratch conditional (lines 218-236):
     ```python
     @property
     def shared_memory_elements(self) -> int:
         """Return total shared memory elements required.

         Includes accumulator, solver_scratch, and stage_increment if shared.
         solver_scratch is always included as it is passed from shared memory.
         stage_base aliases accumulator when multistage, so not counted
         separately.
         """
         total = 0
         if self.use_shared_accumulator:
             total += self.accumulator_length
         total += self.solver_scratch_elements  # Always included
         if self.use_shared_stage_increment:
             total += self.n
         # stage_base aliases accumulator when multistage; only add if
         # single-stage and shared
         if not self.multistage and self.use_shared_stage_base:
             total += self.n
         return total
     ```
   - Edge cases: Single-stage case still handled
   - Integration: Affects memory reporting upstream

6. **Update local_memory_elements property**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Remove solver_scratch conditional (lines 238-255):
     ```python
     @property
     def local_memory_elements(self) -> int:
         """Return total local memory elements required.

         Includes buffers configured with location='local'.
         solver_scratch is not included as it is always shared from parent.
         """
         total = 0
         if not self.use_shared_accumulator:
             total += self.accumulator_length
         # solver_scratch removed - always shared from parent
         if not self.use_shared_stage_increment:
             total += self.n
         # stage_base needs local storage when single-stage and local
         if not self.multistage and not self.use_shared_stage_base:
             total += self.n
         return total
     ```
   - Edge cases: Single-stage case still handled
   - Integration: Affects memory reporting upstream

7. **Update local_sizes property**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Update solver_scratch size in local_sizes (lines 256-279):
     ```python
     @property
     def local_sizes(self) -> DIRKLocalSizes:
         """Return DIRKLocalSizes instance with buffer sizes.

         The returned object provides nonzero sizes suitable for
         cuda.local.array allocation.
         """
         # stage_base size depends on whether it aliases accumulator
         if self.multistage:
             stage_base_size = 0  # Aliases accumulator when multistage
         else:
             stage_base_size = self.n
         return DIRKLocalSizes(
             stage_increment=self.n,
             stage_base=stage_base_size,
             accumulator=self.accumulator_length,
             solver_scratch=self.solver_scratch_elements,
             increment_cache=0,  # Always 0 since solver_scratch shared
             rhs_cache=0,        # Always 0 since solver_scratch shared
         )
     ```
   - Edge cases: increment_cache and rhs_cache are 0 now
   - Integration: Used for local array allocation

8. **Update shared_indices property**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Always include solver_scratch slice (lines 280-325):
     ```python
     @property
     def shared_indices(self) -> DIRKSliceIndices:
         """Return DIRKSliceIndices instance with shared memory layout.

         The returned object contains slices for each buffer's region
         in shared memory. Local buffers receive empty slices.
         """
         ptr = 0

         if self.use_shared_accumulator:
             accumulator_slice = slice(ptr, ptr + self.accumulator_length)
             ptr += self.accumulator_length
         else:
             accumulator_slice = slice(0, 0)

         # solver_scratch always included in shared memory layout
         solver_scratch_slice = slice(ptr, ptr + self.solver_scratch_elements)
         ptr += self.solver_scratch_elements

         if self.use_shared_stage_increment:
             stage_increment_slice = slice(ptr, ptr + self.n)
             ptr += self.n
         else:
             stage_increment_slice = slice(0, 0)

         # stage_base aliases accumulator when multistage
         if self.stage_base_aliases_accumulator:
             stage_base_slice = slice(
                 accumulator_slice.start,
                 accumulator_slice.start + self.n
             )
         elif self.use_shared_stage_base and not self.multistage:
             stage_base_slice = slice(ptr, ptr + self.n)
             ptr += self.n
         else:
             stage_base_slice = slice(0, 0)

         return DIRKSliceIndices(
             stage_increment=stage_increment_slice,
             stage_base=stage_base_slice,
             accumulator=accumulator_slice,
             solver_scratch=solver_scratch_slice,
             local_end=ptr,
         )
     ```
   - Edge cases: Always provides a valid slice for solver_scratch
   - Integration: Used by DIRKStep.build_step

9. **Remove solver_scratch_location from ALL_DIRK_BUFFER_LOCATION_PARAMETERS**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Modify lines 329-334:
     ```python
     # Buffer location parameters for DIRK algorithms
     ALL_DIRK_BUFFER_LOCATION_PARAMETERS = {
         "stage_increment_location",
         "stage_base_location",
         "accumulator_location",
         # solver_scratch_location removed
     }
     ```
   - Edge cases: None
   - Integration: Used for parameter validation elsewhere

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_dirk.py (9 changes)
- Functions/Methods Added/Modified:
  * Removed solver_scratch_location attribute from DIRKBufferSettings
  * Removed use_shared_solver_scratch property
  * Updated solver_scratch_elements to remove fallback
  * Updated persistent_local_elements to always return 0
  * Updated shared_memory_elements to always include solver_scratch
  * Updated local_memory_elements to remove solver_scratch
  * Updated local_sizes to set increment_cache and rhs_cache to 0
  * Updated shared_indices to always include solver_scratch slice
  * Removed solver_scratch_location from ALL_DIRK_BUFFER_LOCATION_PARAMETERS
- Implementation Summary: Removed solver_scratch_location toggleability - always shared from parent
- Issues Flagged: None

---

## Task Group 3: DIRKBufferSettings - Make newton_buffer_settings Required - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 110-160, 485-520)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (NewtonBufferSettings)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (LinearSolverBufferSettings)

**Input Validation Required**:
- None - using factory default, not user input

**Tasks**:

1. **Add imports for NewtonBufferSettings and LinearSolverBufferSettings at module header**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Add after existing imports (around line 55):
     ```python
     from cubie.integrators.matrix_free_solvers.linear_solver import (
         LinearSolverBufferSettings,
         linear_solver_factory,
     )
     from cubie.integrators.matrix_free_solvers.newton_krylov import (
         NewtonBufferSettings,
         newton_krylov_solver_factory,
     )
     ```
     Then update the existing import at line 51-54:
     ```python
     # CHANGE FROM:
     from cubie.integrators.matrix_free_solvers import (
         linear_solver_factory,
         newton_krylov_solver_factory,
     )
     # TO:
     # (imports now at individual module level above)
     ```
   - Edge cases: Must handle both LinearSolverBufferSettings and NewtonBufferSettings
   - Integration: Removes need for inline imports in __init__

2. **Change newton_buffer_settings attribute to required with factory default**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Change lines 149-151:
     ```python
     # CHANGE FROM:
     newton_buffer_settings: Optional["NewtonBufferSettings"] = attrs.field(
         default=None,
     )
     
     # TO:
     newton_buffer_settings: NewtonBufferSettings = attrs.field()
     ```
     
     Then add __attrs_post_init__ to create default if needed:
     ```python
     def __attrs_post_init__(self):
         """Set default newton_buffer_settings if not provided."""
         if self.newton_buffer_settings is None:
             linear_settings = LinearSolverBufferSettings(n=self.n)
             newton_settings = NewtonBufferSettings(
                 n=self.n,
                 linear_solver_buffer_settings=linear_settings,
             )
             object.__setattr__(self, 'newton_buffer_settings', newton_settings)
     ```
     
     BUT WAIT - attrs won't allow None if there's no default. Need a different approach.
     
     Use attrs converter instead:
     ```python
     def _convert_newton_buffer_settings(value):
         """Placeholder converter - actual default set in __attrs_post_init__."""
         return value
     
     newton_buffer_settings: NewtonBufferSettings = attrs.field(
         default=None,
         converter=_convert_newton_buffer_settings,
     )
     ```
     
     Actually, the simplest approach: keep Optional but add __attrs_post_init__
     to replace None with proper default, then never do None checks in code.
     
     FINAL APPROACH:
     Keep the Optional type hint temporarily but use __attrs_post_init__ to 
     guarantee it's always set. Update type to remove Optional in signature.
     ```python
     _newton_buffer_settings: Optional[NewtonBufferSettings] = attrs.field(
         default=None,
         alias='newton_buffer_settings',
     )
     
     def __attrs_post_init__(self):
         """Set default newton_buffer_settings if not provided."""
         if self._newton_buffer_settings is None:
             linear_settings = LinearSolverBufferSettings(n=self.n)
             newton_settings = NewtonBufferSettings(
                 n=self.n,
                 linear_solver_buffer_settings=linear_settings,
             )
             object.__setattr__(self, '_newton_buffer_settings', newton_settings)
     
     @property
     def newton_buffer_settings(self) -> NewtonBufferSettings:
         """Return newton_buffer_settings (guaranteed non-None after init)."""
         return self._newton_buffer_settings
     ```
     
     NO - the copilot instructions say "Never add aliases to underscored variables".
     
     REVISED FINAL APPROACH: Keep as is with Optional, add __attrs_post_init__:
     ```python
     newton_buffer_settings: Optional[NewtonBufferSettings] = attrs.field(
         default=None,
     )
     
     def __attrs_post_init__(self):
         """Set default newton_buffer_settings if not provided."""
         if self.newton_buffer_settings is None:
             linear_settings = LinearSolverBufferSettings(n=self.n)
             newton_settings = NewtonBufferSettings(
                 n=self.n,
                 linear_solver_buffer_settings=linear_settings,
             )
             object.__setattr__(self, 'newton_buffer_settings', newton_settings)
     ```
     
     This maintains Optional in the type hint but guarantees it's set after init.
     All None checks can be safely removed since __attrs_post_init__ ensures value.
   - Edge cases: __attrs_post_init__ runs after __init__, accessing self.n is safe
   - Integration: Enables removal of all None checks for newton_buffer_settings

3. **Remove inline imports from DIRKStep.__init__**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Remove lines 491-496:
     ```python
     # DELETE THESE LINES:
     from cubie.integrators.matrix_free_solvers.linear_solver import (
         LinearSolverBufferSettings
     )
     from cubie.integrators.matrix_free_solvers.newton_krylov import (
         NewtonBufferSettings
     )
     ```
   - Edge cases: None - imports are now at module level
   - Integration: Already added at module header in Task 1

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_dirk.py (3 changes)
- Functions/Methods Added/Modified:
  * Updated imports at module header to include NewtonBufferSettings and LinearSolverBufferSettings
  * Added __attrs_post_init__ to DIRKBufferSettings to create default newton_buffer_settings
  * Removed inline imports from DIRKStep.__init__
- Implementation Summary: Module-level imports and __attrs_post_init__ for default newton_buffer_settings
- Issues Flagged: None

---

## Task Group 4: DIRKStepConfig Properties and build_implicit_helpers - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 3

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 387-400, 551-629)

**Input Validation Required**:
- None - properties, not user input

**Tasks**:

1. **Add newton_buffer_settings property to DIRKStepConfig**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Add after the buffer_settings attribute (around line 399):
     ```python
     @property
     def newton_buffer_settings(self) -> NewtonBufferSettings:
         """Return newton_buffer_settings from buffer_settings."""
         return self.buffer_settings.newton_buffer_settings
     
     @property
     def linear_solver_buffer_settings(self) -> LinearSolverBufferSettings:
         """Return linear_solver_buffer_settings from newton_buffer_settings."""
         return self.buffer_settings.newton_buffer_settings.linear_solver_buffer_settings
     ```
     
     NOTE: Need to handle Optional[DIRKBufferSettings] on buffer_settings.
     Actually checking the code - buffer_settings is Optional[DIRKBufferSettings]
     so we need to handle that case. But if buffer_settings is None, we can't
     access newton_buffer_settings. Review shows DIRKStep.__init__ always creates
     buffer_settings, so this is safe in practice. Add guard for safety:
     
     ```python
     @property
     def newton_buffer_settings(self) -> Optional[NewtonBufferSettings]:
         """Return newton_buffer_settings from buffer_settings."""
         if self.buffer_settings is None:
             return None
         return self.buffer_settings.newton_buffer_settings
     
     @property
     def linear_solver_buffer_settings(self) -> Optional[LinearSolverBufferSettings]:
         """Return linear_solver_buffer_settings from newton_buffer_settings."""
         newton_settings = self.newton_buffer_settings
         if newton_settings is None:
             return None
         return newton_settings.linear_solver_buffer_settings
     ```
   - Edge cases: buffer_settings could be None (Optional)
   - Integration: Used by build_implicit_helpers

2. **Simplify build_implicit_helpers to use direct properties**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Replace lines 594-600:
     ```python
     # CHANGE FROM:
     # Extract newton_buffer_settings from buffer_settings for memory chain
     newton_buffer_settings = config.buffer_settings.newton_buffer_settings
     linear_buffer_settings = None
     if newton_buffer_settings is not None:
         linear_buffer_settings = (
             newton_buffer_settings.linear_solver_buffer_settings
         )
     
     # TO:
     newton_buffer_settings = config.newton_buffer_settings
     linear_buffer_settings = config.linear_solver_buffer_settings
     ```
   - Edge cases: Now using properties that handle None internally
   - Integration: Cleaner code, same behavior

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_dirk.py (2 changes)
- Functions/Methods Added/Modified:
  * DIRKStepConfig: added newton_buffer_settings property
  * DIRKStepConfig: added linear_solver_buffer_settings property
  * build_implicit_helpers: simplified to use direct config properties
- Implementation Summary: Config properties provide direct access without drilling through attributes
- Issues Flagged: None

---

## Task Group 5: FIRKBufferSettings - Same Changes as DIRK - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 102-290, 371-610)

**Input Validation Required**:
- None - removing attribute, not adding

**Tasks**:

1. **Add imports for NewtonBufferSettings and LinearSolverBufferSettings at module header**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Update imports around lines 49-52:
     ```python
     # CHANGE FROM:
     from cubie.integrators.matrix_free_solvers import (
         linear_solver_factory,
         newton_krylov_solver_factory,
     )
     
     # TO:
     from cubie.integrators.matrix_free_solvers.linear_solver import (
         LinearSolverBufferSettings,
         linear_solver_factory,
     )
     from cubie.integrators.matrix_free_solvers.newton_krylov import (
         NewtonBufferSettings,
         newton_krylov_solver_factory,
     )
     ```
   - Edge cases: None
   - Integration: Enables removal of inline imports in __init__

2. **Remove solver_scratch_location attribute from FIRKBufferSettings**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Remove lines 133-135:
     ```python
     # DELETE THESE LINES:
     solver_scratch_location: str = attrs.field(
         default='local', validator=validators.in_(["local", "shared"])
     )
     ```
   - Edge cases: None
   - Integration: Must update dependent properties

3. **Remove use_shared_solver_scratch property**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Remove lines 149-152:
     ```python
     # DELETE THESE LINES:
     @property
     def use_shared_solver_scratch(self) -> bool:
         """Return True if solver_scratch buffer uses shared memory."""
         return self.solver_scratch_location == 'shared'
     ```
   - Edge cases: None
   - Integration: Dependent code needs updating

4. **Update solver_scratch_elements to remove fallback**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Modify lines 175-183:
     ```python
     @property
     def solver_scratch_elements(self) -> int:
         """Return solver scratch elements (includes linear solver).

         Returns newton_buffer_settings.shared_memory_elements.
         """
         return self.newton_buffer_settings.shared_memory_elements
     ```
   - Edge cases: None - newton_buffer_settings guaranteed after __attrs_post_init__
   - Integration: Depends on newton_buffer_settings being required

5. **Update shared_memory_elements to always include solver_scratch**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Modify lines 191-206:
     ```python
     @property
     def shared_memory_elements(self) -> int:
         """Return total shared memory elements required.

         Always includes solver_scratch as it is passed from shared memory.
         Includes stage_increment and stage_driver_stack if configured for
         shared memory.
         """
         total = self.solver_scratch_elements  # Always included
         if self.use_shared_stage_increment:
             total += self.all_stages_n
         if self.use_shared_stage_driver_stack:
             total += self.stage_driver_stack_elements
         if self.use_shared_stage_state:
             total += self.n
         return total
     ```
   - Edge cases: None
   - Integration: Used for memory accounting

6. **Update local_memory_elements to exclude solver_scratch**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Modify lines 208-224:
     ```python
     @property
     def local_memory_elements(self) -> int:
         """Return total local memory elements required.

         Includes buffers configured with location='local'.
         solver_scratch is not included as it is always shared from parent.
         """
         total = 0
         # solver_scratch always shared from parent, not counted
         if not self.use_shared_stage_increment:
             total += self.all_stages_n
         if not self.use_shared_stage_driver_stack:
             total += self.stage_driver_stack_elements
         if not self.use_shared_stage_state:
             total += self.n
         return total
     ```
   - Edge cases: None
   - Integration: Used for memory accounting

7. **Update shared_indices to always include solver_scratch slice**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Modify lines 239-280:
     ```python
     @property
     def shared_indices(self) -> FIRKSliceIndices:
         """Return FIRKSliceIndices instance with shared memory layout.

         The returned object contains slices for each buffer's region
         in shared memory. Local buffers receive empty slices.
         """
         ptr = 0

         # solver_scratch always included in shared memory layout
         solver_scratch_slice = slice(ptr, ptr + self.solver_scratch_elements)
         ptr += self.solver_scratch_elements

         if self.use_shared_stage_increment:
             stage_increment_slice = slice(ptr, ptr + self.all_stages_n)
             ptr += self.all_stages_n
         else:
             stage_increment_slice = slice(0, 0)

         if self.use_shared_stage_driver_stack:
             stage_driver_stack_slice = slice(
                 ptr, ptr + self.stage_driver_stack_elements
             )
             ptr += self.stage_driver_stack_elements
         else:
             stage_driver_stack_slice = slice(0, 0)

         if self.use_shared_stage_state:
             stage_state_slice = slice(ptr, ptr + self.n)
             ptr += self.n
         else:
             stage_state_slice = slice(0, 0)

         return FIRKSliceIndices(
             solver_scratch=solver_scratch_slice,
             stage_increment=stage_increment_slice,
             stage_driver_stack=stage_driver_stack_slice,
             stage_state=stage_state_slice,
             local_end=ptr,
         )
     ```
   - Edge cases: None
   - Integration: Used by FIRKStep.build_step

8. **Remove solver_scratch_location from ALL_FIRK_BUFFER_LOCATION_PARAMETERS**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Modify lines 283-289:
     ```python
     # Buffer location parameters for FIRK algorithms
     ALL_FIRK_BUFFER_LOCATION_PARAMETERS = {
         # solver_scratch_location removed
         "stage_increment_location",
         "stage_driver_stack_location",
         "stage_state_location",
     }
     ```
   - Edge cases: None
   - Integration: Used for parameter validation

9. **Add __attrs_post_init__ to FIRKBufferSettings for newton_buffer_settings default**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Add after newton_buffer_settings attribute:
     ```python
     def __attrs_post_init__(self):
         """Set default newton_buffer_settings if not provided."""
         if self.newton_buffer_settings is None:
             all_stages_n = self.stage_count * self.n
             linear_settings = LinearSolverBufferSettings(n=all_stages_n)
             newton_settings = NewtonBufferSettings(
                 n=all_stages_n,
                 linear_solver_buffer_settings=linear_settings,
             )
             object.__setattr__(self, 'newton_buffer_settings', newton_settings)
     ```
     NOTE: FIRK uses all_stages_n (stage_count * n) unlike DIRK which uses n.
   - Edge cases: Uses all_stages_n for coupled system dimension
   - Integration: Guarantees newton_buffer_settings is always set

10. **Remove inline imports from FIRKStep.__init__**
    - File: src/cubie/integrators/algorithms/generic_firk.py
    - Action: Modify
    - Details:
      Remove lines 465-470:
      ```python
      # DELETE THESE LINES:
      from cubie.integrators.matrix_free_solvers.linear_solver import (
          LinearSolverBufferSettings
      )
      from cubie.integrators.matrix_free_solvers.newton_krylov import (
          NewtonBufferSettings
      )
      ```
    - Edge cases: None - imports now at module level
    - Integration: Already added at module header in Task 1

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_firk.py (10 changes)
- Functions/Methods Added/Modified:
  * Updated imports at module header
  * Removed solver_scratch_location attribute
  * Removed use_shared_solver_scratch property
  * Updated solver_scratch_elements to remove fallback
  * Updated shared_memory_elements to always include solver_scratch
  * Updated local_memory_elements to exclude solver_scratch
  * Updated shared_indices to always include solver_scratch
  * Removed solver_scratch_location from ALL_FIRK_BUFFER_LOCATION_PARAMETERS
  * Added __attrs_post_init__ with all_stages_n default
  * Removed inline imports from FIRKStep.__init__
- Implementation Summary: Same changes as DIRK, using all_stages_n for FIRK
- Issues Flagged: None

---

## Task Group 6: FIRKStepConfig Properties and build_implicit_helpers - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 5

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 344-370, 526-610)

**Input Validation Required**:
- None - properties, not user input

**Tasks**:

1. **Add newton_buffer_settings and linear_solver_buffer_settings properties to FIRKStepConfig**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Add after the all_stages_n property (around line 369):
     ```python
     @property
     def newton_buffer_settings(self) -> Optional[NewtonBufferSettings]:
         """Return newton_buffer_settings from buffer_settings."""
         if self.buffer_settings is None:
             return None
         return self.buffer_settings.newton_buffer_settings
     
     @property
     def linear_solver_buffer_settings(self) -> Optional[LinearSolverBufferSettings]:
         """Return linear_solver_buffer_settings from newton_buffer_settings."""
         newton_settings = self.newton_buffer_settings
         if newton_settings is None:
             return None
         return newton_settings.linear_solver_buffer_settings
     ```
   - Edge cases: buffer_settings could be None (Optional)
   - Integration: Used by build_implicit_helpers

2. **Simplify build_implicit_helpers to use direct properties**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Replace lines 576-582:
     ```python
     # CHANGE FROM:
     # Extract newton_buffer_settings from buffer_settings for memory chain
     newton_buffer_settings = config.buffer_settings.newton_buffer_settings
     linear_buffer_settings = None
     if newton_buffer_settings is not None:
         linear_buffer_settings = (
             newton_buffer_settings.linear_solver_buffer_settings
         )
     
     # TO:
     newton_buffer_settings = config.newton_buffer_settings
     linear_buffer_settings = config.linear_solver_buffer_settings
     ```
   - Edge cases: Properties handle None internally
   - Integration: Cleaner code, same behavior

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_firk.py (2 changes)
- Functions/Methods Added/Modified:
  * FIRKStepConfig: added newton_buffer_settings property
  * FIRKStepConfig: added linear_solver_buffer_settings property
  * build_implicit_helpers: simplified to use direct config properties
- Implementation Summary: Config properties provide direct access without drilling
- Issues Flagged: None

---

## Task Group 7: DIRKStep - Remove solver_scratch_location Parameter - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 3

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 402-550)

**Input Validation Required**:
- None - removing parameter

**Tasks**:

1. **Remove solver_scratch_location parameter from DIRKStep.__init__**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Remove line 426 from the __init__ signature:
     ```python
     # DELETE THIS PARAMETER:
     solver_scratch_location: Optional[str] = None,
     ```
     
     And remove lines 515-516 from the buffer_kwargs construction:
     ```python
     # DELETE THESE LINES:
     if solver_scratch_location is not None:
         buffer_kwargs['solver_scratch_location'] = solver_scratch_location
     ```
   - Edge cases: None
   - Integration: solver_scratch_location no longer configurable

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_dirk.py (2 changes)
- Functions/Methods Added/Modified:
  * DIRKStep.__init__: removed solver_scratch_location parameter
  * buffer_kwargs: removed solver_scratch_location handling
  * build_step: removed solver_scratch_shared and local size handling
- Implementation Summary: Removed solver_scratch_location parameter from __init__
- Issues Flagged: None

---

## Task Group 8: FIRKStep - Remove solver_scratch_location Parameter - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 5

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 371-525)

**Input Validation Required**:
- None - removing parameter

**Tasks**:

1. **Remove solver_scratch_location parameter from FIRKStep.__init__**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     Remove line 392 from the __init__ signature:
     ```python
     # DELETE THIS PARAMETER:
     solver_scratch_location: Optional[str] = None,
     ```
     
     And remove lines 485-486 from the buffer_kwargs construction:
     ```python
     # DELETE THESE LINES:
     if solver_scratch_location is not None:
         buffer_kwargs['solver_scratch_location'] = solver_scratch_location
     ```
   - Edge cases: None
   - Integration: solver_scratch_location no longer configurable

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_firk.py (2 changes)
- Functions/Methods Added/Modified:
  * FIRKStep.__init__: removed solver_scratch_location parameter
  * buffer_kwargs: removed solver_scratch_location handling
  * build_step: removed solver_scratch_shared and local size handling
- Implementation Summary: Removed solver_scratch_location parameter from __init__
- Issues Flagged: None

---

## Task Group 9: Test Updates - test_newton_buffer_settings.py - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py (entire file)

**Input Validation Required**:
- None - test file

**Tasks**:

1. **Remove test_shared_memory_elements_default (tests default values)**
   - File: tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py
   - Action: Modify
   - Details:
     Remove lines 16-19:
     ```python
     # DELETE THIS TEST:
     def test_shared_memory_elements_default(self):
         """Default: delta and residual shared gives 2*n."""
         settings = NewtonBufferSettings(n=10)
         assert settings.shared_memory_elements == 20  # 2 * n
     ```
   - Edge cases: None
   - Integration: Tests for defaults may break when defaults change

2. **Remove test_local_memory_elements_default (tests default values)**
   - File: tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py
   - Action: Modify
   - Details:
     Remove lines 36-40:
     ```python
     # DELETE THIS TEST:
     def test_local_memory_elements_default(self):
         """Default shared buffers give residual_temp + krylov_iters."""
         settings = NewtonBufferSettings(n=10)
         # residual_temp (n) + krylov_iters (1) = 11
         assert settings.local_memory_elements == 11
     ```
   - Edge cases: None
   - Integration: None

3. **Remove test_shared_indices_contiguous (tests implementation detail)**
   - File: tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py
   - Action: Modify
   - Details:
     Remove lines 52-57:
     ```python
     # DELETE THIS TEST:
     def test_shared_indices_contiguous(self):
         """Shared memory slices should be contiguous."""
         settings = NewtonBufferSettings(n=10)
         indices = settings.shared_indices
         assert indices.delta.stop == indices.residual.start
         assert indices.local_end == indices.residual.stop
     ```
   - Edge cases: None
   - Integration: None

4. **Remove test_lin_solver_start_matches_local_end (tests implementation detail)**
   - File: tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py
   - Action: Modify
   - Details:
     Remove lines 81-85:
     ```python
     # DELETE THIS TEST:
     def test_lin_solver_start_matches_local_end(self):
         """lin_solver_start should equal local_end."""
         settings = NewtonBufferSettings(n=10)
         indices = settings.shared_indices
         assert indices.lin_solver_start == indices.local_end
     ```
   - Edge cases: None
   - Integration: None

5. **Update test_shared_indices_all_local_gives_empty to test shared_memory_elements**
   - File: tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py
   - Action: Modify
   - Details:
     Update lines 59-69:
     ```python
     # CHANGE FROM:
     def test_shared_indices_all_local_gives_empty(self):
         """Local buffers get empty slices."""
         settings = NewtonBufferSettings(
             n=10,
             delta_location='local',
             residual_location='local',
         )
         indices = settings.shared_indices
         assert indices.delta == slice(0, 0)
         assert indices.residual == slice(0, 0)
         assert indices.local_end == 0
     
     # TO:
     def test_shared_memory_elements_all_local(self):
         """All local buffers should give zero shared memory elements."""
         settings = NewtonBufferSettings(
             n=10,
             delta_location='local',
             residual_location='local',
             residual_temp_location='local',
         )
         assert settings.shared_memory_elements == 0
     ```
   - Edge cases: Need to include residual_temp_location='local' for new attribute
   - Integration: Tests the property instead of internal indices

6. **Add test for residual_temp toggleability**
   - File: tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py
   - Action: Modify
   - Details:
     Add after boolean_flags test (around line 80):
     ```python
     def test_residual_temp_toggleability(self):
         """residual_temp_location should affect memory calculations."""
         # Default local
         settings_local = NewtonBufferSettings(n=10)
         assert settings_local.use_shared_residual_temp is False
         
         # Explicit shared
         settings_shared = NewtonBufferSettings(
             n=10,
             residual_temp_location='shared',
         )
         assert settings_shared.use_shared_residual_temp is True
         # Shared should have n more shared elements
         diff = (settings_shared.shared_memory_elements - 
                 settings_local.shared_memory_elements)
         assert diff == 10  # n elements for residual_temp
     ```
   - Edge cases: None
   - Integration: Tests new residual_temp_location feature

**Outcomes**: 
- Files Modified:
  * tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py (6 changes)
- Functions/Methods Added/Modified:
  * Removed test_shared_memory_elements_default
  * Removed test_local_memory_elements_default
  * Removed test_shared_indices_contiguous
  * Removed test_lin_solver_start_matches_local_end
  * Updated test_shared_indices_all_local_gives_empty to test_shared_memory_elements_all_local
  * Added test_residual_temp_toggleability
  * Updated NewtonSliceIndices test to include residual_temp
- Implementation Summary: Updated tests to match new residual_temp toggleability
- Issues Flagged: None

---

## Task Group 10: Test Updates - test_buffer_settings.py (algorithms) - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Groups 2, 5

**Required Context**:
- File: tests/integrators/algorithms/test_buffer_settings.py (entire file)

**Input Validation Required**:
- None - test file

**Tasks**:

1. **Update test_default_locations for DIRK to not include solver_scratch_location**
   - File: tests/integrators/algorithms/test_buffer_settings.py
   - Action: Modify
   - Details:
     Update lines 140-147:
     ```python
     def test_default_locations(self):
         """Default locations should match all_in_one.py."""
         settings = DIRKBufferSettings(n=3, stage_count=4)
 
         assert settings.stage_increment_location == 'local'
         assert settings.stage_base_location == 'local'
         assert settings.accumulator_location == 'local'
         # solver_scratch_location removed - no longer testable
     ```
   - Edge cases: None
   - Integration: solver_scratch_location attribute removed

2. **Update test_solver_scratch_elements for DIRK to require newton_buffer_settings**
   - File: tests/integrators/algorithms/test_buffer_settings.py
   - Action: Modify
   - Details:
     Update lines 182-185:
     ```python
     # CHANGE FROM:
     def test_solver_scratch_elements(self):
         """Solver scratch should be 2 * n without newton_buffer_settings."""
         settings = DIRKBufferSettings(n=5, stage_count=3)
         assert settings.solver_scratch_elements == 10
     
     # TO:
     def test_solver_scratch_elements(self):
         """Solver scratch should use newton_buffer_settings."""
         from cubie.integrators.matrix_free_solvers.newton_krylov import (
             NewtonBufferSettings
         )
         from cubie.integrators.matrix_free_solvers.linear_solver import (
             LinearSolverBufferSettings
         )
         linear_settings = LinearSolverBufferSettings(n=5)
         newton_settings = NewtonBufferSettings(
             n=5,
             linear_solver_buffer_settings=linear_settings,
         )
         settings = DIRKBufferSettings(
             n=5,
             stage_count=3,
             newton_buffer_settings=newton_settings,
         )
         assert settings.solver_scratch_elements == (
             newton_settings.shared_memory_elements
         )
     ```
   - Edge cases: None
   - Integration: Tests with explicit newton_buffer_settings

3. **Update test_shared_memory_elements_multistage for DIRK**
   - File: tests/integrators/algorithms/test_buffer_settings.py
   - Action: Modify
   - Details:
     Update lines 210-220:
     ```python
     # CHANGE FROM:
     def test_shared_memory_elements_multistage(self):
         """Shared memory should sum sizes of shared buffers."""
         settings = DIRKBufferSettings(
             n=3,
             stage_count=4,
             accumulator_location='shared',
             solver_scratch_location='shared',
             stage_increment_location='shared',
         )
         # accumulator ((4-1)*3=9) + solver (2*3=6) + increment (3) = 18
         assert settings.shared_memory_elements == 18
     
     # TO:
     def test_shared_memory_elements_multistage(self):
         """Shared memory should sum sizes of shared buffers."""
         from cubie.integrators.matrix_free_solvers.newton_krylov import (
             NewtonBufferSettings
         )
         from cubie.integrators.matrix_free_solvers.linear_solver import (
             LinearSolverBufferSettings
         )
         linear_settings = LinearSolverBufferSettings(n=3)
         newton_settings = NewtonBufferSettings(
             n=3,
             linear_solver_buffer_settings=linear_settings,
         )
         settings = DIRKBufferSettings(
             n=3,
             stage_count=4,
             accumulator_location='shared',
             stage_increment_location='shared',
             newton_buffer_settings=newton_settings,
         )
         # accumulator ((4-1)*3=9) + solver (newton shared) + increment (3)
         expected = 9 + newton_settings.shared_memory_elements + 3
         assert settings.shared_memory_elements == expected
     ```
   - Edge cases: None
   - Integration: solver_scratch always included in shared

4. **Update test_default_locations for FIRK to not include solver_scratch_location**
   - File: tests/integrators/algorithms/test_buffer_settings.py
   - Action: Modify
   - Details:
     Update lines 253-260:
     ```python
     def test_default_locations(self):
         """Default locations should match expected values."""
         settings = FIRKBufferSettings(n=3, stage_count=4)
 
         # solver_scratch_location removed - no longer testable
         assert settings.stage_increment_location == 'local'
         assert settings.stage_driver_stack_location == 'local'
         assert settings.stage_state_location == 'local'
     ```
   - Edge cases: None
   - Integration: solver_scratch_location attribute removed

5. **Update test_solver_scratch_elements for FIRK to require newton_buffer_settings**
   - File: tests/integrators/algorithms/test_buffer_settings.py
   - Action: Modify
   - Details:
     Update lines 278-282:
     ```python
     # CHANGE FROM:
     def test_solver_scratch_elements(self):
         """Solver scratch should be 2 * all_stages_n without buffer settings."""
         settings = FIRKBufferSettings(n=3, stage_count=4)
         # 2 * (4 * 3) = 24
         assert settings.solver_scratch_elements == 24
     
     # TO:
     def test_solver_scratch_elements(self):
         """Solver scratch should use newton_buffer_settings."""
         from cubie.integrators.matrix_free_solvers.newton_krylov import (
             NewtonBufferSettings
         )
         from cubie.integrators.matrix_free_solvers.linear_solver import (
             LinearSolverBufferSettings
         )
         all_stages_n = 4 * 3  # stage_count * n
         linear_settings = LinearSolverBufferSettings(n=all_stages_n)
         newton_settings = NewtonBufferSettings(
             n=all_stages_n,
             linear_solver_buffer_settings=linear_settings,
         )
         settings = FIRKBufferSettings(
             n=3,
             stage_count=4,
             newton_buffer_settings=newton_settings,
         )
         assert settings.solver_scratch_elements == (
             newton_settings.shared_memory_elements
         )
     ```
   - Edge cases: Uses all_stages_n for FIRK
   - Integration: Tests with explicit newton_buffer_settings

6. **Update test_shared_memory_elements for FIRK**
   - File: tests/integrators/algorithms/test_buffer_settings.py
   - Action: Modify
   - Details:
     Update lines 314-326:
     ```python
     # CHANGE FROM:
     def test_shared_memory_elements(self):
         """Shared memory should sum sizes of shared buffers."""
         settings = FIRKBufferSettings(
             n=3,
             stage_count=4,
             n_drivers=2,
             solver_scratch_location='shared',
             stage_increment_location='shared',
             stage_driver_stack_location='shared',
             stage_state_location='shared',
         )
         # solver (2*12=24) + increment (12) + drivers (8) + state (3) = 47
         assert settings.shared_memory_elements == 47
     
     # TO:
     def test_shared_memory_elements(self):
         """Shared memory should sum sizes of shared buffers."""
         from cubie.integrators.matrix_free_solvers.newton_krylov import (
             NewtonBufferSettings
         )
         from cubie.integrators.matrix_free_solvers.linear_solver import (
             LinearSolverBufferSettings
         )
         all_stages_n = 4 * 3
         linear_settings = LinearSolverBufferSettings(n=all_stages_n)
         newton_settings = NewtonBufferSettings(
             n=all_stages_n,
             linear_solver_buffer_settings=linear_settings,
         )
         settings = FIRKBufferSettings(
             n=3,
             stage_count=4,
             n_drivers=2,
             stage_increment_location='shared',
             stage_driver_stack_location='shared',
             stage_state_location='shared',
             newton_buffer_settings=newton_settings,
         )
         # solver (newton) + increment (12) + drivers (8) + state (3)
         expected = newton_settings.shared_memory_elements + 12 + 8 + 3
         assert settings.shared_memory_elements == expected
     ```
   - Edge cases: solver_scratch always included in shared now
   - Integration: Tests with explicit newton_buffer_settings

7. **Update test_local_memory_elements for FIRK**
   - File: tests/integrators/algorithms/test_buffer_settings.py
   - Action: Modify
   - Details:
     Update lines 328-340:
     ```python
     # CHANGE FROM:
     def test_local_memory_elements(self):
         """Local memory should sum sizes of local buffers."""
         settings = FIRKBufferSettings(
             n=3,
             stage_count=4,
             n_drivers=2,
             solver_scratch_location='local',
             stage_increment_location='local',
             stage_driver_stack_location='local',
             stage_state_location='local',
         )
         # solver (24) + increment (12) + drivers (8) + state (3) = 47
         assert settings.local_memory_elements == 47
     
     # TO:
     def test_local_memory_elements(self):
         """Local memory should sum sizes of local buffers."""
         from cubie.integrators.matrix_free_solvers.newton_krylov import (
             NewtonBufferSettings
         )
         from cubie.integrators.matrix_free_solvers.linear_solver import (
             LinearSolverBufferSettings
         )
         all_stages_n = 4 * 3
         linear_settings = LinearSolverBufferSettings(n=all_stages_n)
         newton_settings = NewtonBufferSettings(
             n=all_stages_n,
             linear_solver_buffer_settings=linear_settings,
         )
         settings = FIRKBufferSettings(
             n=3,
             stage_count=4,
             n_drivers=2,
             stage_increment_location='local',
             stage_driver_stack_location='local',
             stage_state_location='local',
             newton_buffer_settings=newton_settings,
         )
         # solver NOT included (always shared from parent)
         # increment (12) + drivers (8) + state (3) = 23
         assert settings.local_memory_elements == 23
     ```
   - Edge cases: solver_scratch not counted in local anymore
   - Integration: Tests new local memory calculation

8. **Update test_shared_indices_property for FIRK**
   - File: tests/integrators/algorithms/test_buffer_settings.py
   - Action: Modify
   - Details:
     Update lines 353-371:
     ```python
     # CHANGE FROM:
     def test_shared_indices_property(self):
         """shared_indices property should return FIRKSliceIndices instance."""
         settings = FIRKBufferSettings(
             n=3,
             stage_count=4,
             n_drivers=2,
             solver_scratch_location='shared',
             stage_increment_location='shared',
             stage_driver_stack_location='shared',
             stage_state_location='shared',
         )
         indices = settings.shared_indices
 
         assert isinstance(indices, FIRKSliceIndices)
         assert indices.solver_scratch == slice(0, 24)
         assert indices.stage_increment == slice(24, 36)
         assert indices.stage_driver_stack == slice(36, 44)
         assert indices.stage_state == slice(44, 47)
         assert indices.local_end == 47
     
     # TO:
     def test_shared_indices_property(self):
         """shared_indices property should return FIRKSliceIndices instance."""
         from cubie.integrators.matrix_free_solvers.newton_krylov import (
             NewtonBufferSettings
         )
         from cubie.integrators.matrix_free_solvers.linear_solver import (
             LinearSolverBufferSettings
         )
         all_stages_n = 4 * 3
         linear_settings = LinearSolverBufferSettings(n=all_stages_n)
         newton_settings = NewtonBufferSettings(
             n=all_stages_n,
             linear_solver_buffer_settings=linear_settings,
         )
         settings = FIRKBufferSettings(
             n=3,
             stage_count=4,
             n_drivers=2,
             stage_increment_location='shared',
             stage_driver_stack_location='shared',
             stage_state_location='shared',
             newton_buffer_settings=newton_settings,
         )
         indices = settings.shared_indices
         solver_size = newton_settings.shared_memory_elements

         assert isinstance(indices, FIRKSliceIndices)
         assert indices.solver_scratch == slice(0, solver_size)
         assert indices.stage_increment == slice(solver_size, solver_size + 12)
         assert indices.stage_driver_stack == slice(solver_size + 12, solver_size + 20)
         assert indices.stage_state == slice(solver_size + 20, solver_size + 23)
         assert indices.local_end == solver_size + 23
     ```
   - Edge cases: Indices are now relative to newton shared size
   - Integration: Tests correct shared memory layout

**Outcomes**: 
- Files Modified:
  * tests/integrators/algorithms/test_buffer_settings.py (8 changes)
- Functions/Methods Added/Modified:
  * TestDIRKBufferSettings.test_default_locations: removed solver_scratch_location assertion
  * TestDIRKBufferSettings.test_solver_scratch_elements: updated to use newton_buffer_settings
  * TestDIRKBufferSettings.test_shared_memory_elements_multistage: updated for newton_buffer_settings
  * TestDIRKBufferSettings.test_local_sizes_property: removed solver_scratch assertion
  * TestDIRKBufferSettings.test_shared_indices_property: updated for dynamic solver_size
  * TestFIRKBufferSettings.test_default_locations: removed solver_scratch_location assertion
  * TestFIRKBufferSettings: Updated multiple tests to use newton_buffer_settings
- Implementation Summary: Updated DIRK and FIRK tests for removed solver_scratch_location
- Issues Flagged: None

---

## Task Group 11: Test Updates - test_buffer_settings.py (loops) - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Groups 2, 5

**Required Context**:
- File: tests/integrators/loops/test_buffer_settings.py (entire file)

**Input Validation Required**:
- None - test file

**Tasks**:

1. **Remove test_proposed_counters_in_local_memory (easily invalidated)**
   - File: tests/integrators/loops/test_buffer_settings.py
   - Action: Modify
   - Details:
     Remove lines 130-140:
     ```python
     # DELETE THIS TEST:
     def test_proposed_counters_in_local_memory(self):
         """Proposed_counters should be counted in local_memory when local."""
         settings = LoopBufferSettings(
             n_states=3,
             n_counters=4,
             counters_location='local',
         )
         # counters (4) + proposed_counters (2) when counters local
         local = settings.local_memory_elements
         # local should include at least counters_size (max(1,4)=4) + 2
         assert local >= 6
     ```
   - Edge cases: None
   - Integration: Test assumes specific array default locations

**Outcomes**: 
- Files Modified:
  * tests/integrators/loops/test_buffer_settings.py (1 change)
- Functions/Methods Added/Modified:
  * Removed test_proposed_counters_in_local_memory
- Implementation Summary: Removed easily invalidated test
- Issues Flagged: None

---

## Summary

### Implementation Complete - Ready for Review

## Execution Summary
- Total Task Groups: 11
- Completed: 11
- Failed: 0
- Total Files Modified: 5

## Task Group Completion
- Group 1: [x] NewtonBufferSettings - Add residual_temp Toggleability - Complete
- Group 2: [x] DIRKBufferSettings - Remove solver_scratch_location - Complete
- Group 3: [x] DIRKBufferSettings - Make newton_buffer_settings Required - Complete
- Group 4: [x] DIRKStepConfig Properties and build_implicit_helpers - Complete
- Group 5: [x] FIRKBufferSettings - Same Changes as DIRK - Complete
- Group 6: [x] FIRKStepConfig Properties and build_implicit_helpers - Complete
- Group 7: [x] DIRKStep - Remove solver_scratch_location Parameter - Complete
- Group 8: [x] FIRKStep - Remove solver_scratch_location Parameter - Complete
- Group 9: [x] Test Updates - test_newton_buffer_settings.py - Complete
- Group 10: [x] Test Updates - test_buffer_settings.py (algorithms) - Complete
- Group 11: [x] Test Updates - test_buffer_settings.py (loops) - Complete

## All Modified Files
1. src/cubie/integrators/matrix_free_solvers/newton_krylov.py
2. src/cubie/integrators/algorithms/generic_dirk.py
3. src/cubie/integrators/algorithms/generic_firk.py
4. tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py
5. tests/integrators/algorithms/test_buffer_settings.py
6. tests/integrators/loops/test_buffer_settings.py

## Flagged Issues
None

## Handoff to Reviewer
All implementation tasks complete. Task list updated with outcomes.
Ready for reviewer agent to validate against user stories and goals.
