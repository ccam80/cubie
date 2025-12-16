# Implementation Task List
# Feature: BufferSettings Stocktake
# Plan Reference: .github/active_plans/buffer_settings_stocktake/agent_plan.md

## Overview

This task list details the implementation work required to audit and fix
BufferSettings classes to ensure accurate memory calculations. The primary
issues are:

1. **DIRK solver_scratch missing linear solver space** - Currently only 2*n
   but needs 2*n + linear_solver.shared_memory_elements
2. **Newton solver has no BufferSettings** - Uses implicit 2*n carving
3. **LinearSolverBufferSettings not wired** - Exists but not passed through
4. **Rosenbrock lazy cached_auxiliary_count** - May cause stale buffer_settings

---

## Task Group 1: NewtonBufferSettings Creation - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 1-303)
- File: src/cubie/BufferSettings.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 25-152)

**Input Validation Required**:
- n: Must be int >= 1
- Location parameters: Must be 'local' or 'shared'

**Tasks**:

1. **Create NewtonLocalSizes class**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify (add new class)
   - Details:
     ```python
     @attrs.define
     class NewtonLocalSizes(LocalSizes):
         """Local array sizes for Newton solver buffers.
         
         Attributes
         ----------
         delta : int
             Newton direction buffer size (n elements).
         residual : int
             Residual buffer size (n elements).
         residual_temp : int
             Temporary residual buffer size (n elements, always local).
         krylov_iters : int
             Krylov iteration counter (1 element, always local).
         """
         delta: int = attrs.field(validator=getype_validator(int, 0))
         residual: int = attrs.field(validator=getype_validator(int, 0))
         residual_temp: int = attrs.field(validator=getype_validator(int, 0))
         krylov_iters: int = attrs.field(validator=getype_validator(int, 0))
     ```
   - Edge cases: n=0 should return 0-sized buffers, nonzero() returns 1
   - Integration: Import LocalSizes from cubie.BufferSettings

2. **Create NewtonSliceIndices class**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify (add new class)
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
         local_end : int
             Offset of the end of Newton-managed shared memory.
         lin_solver_start : int
             Start offset for linear solver shared memory.
         """
         delta: slice = attrs.field()
         residual: slice = attrs.field()
         local_end: int = attrs.field()
         lin_solver_start: int = attrs.field()
     ```
   - Edge cases: Empty slices when buffers are local
   - Integration: Import SliceIndices from cubie.BufferSettings

3. **Create NewtonBufferSettings class**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify (add new class)
   - Details:
     ```python
     @attrs.define
     class NewtonBufferSettings(BufferSettings):
         """Configuration for Newton solver buffer sizes and locations.
         
         Controls memory locations for delta and residual buffers used
         during Newton-Krylov iteration. residual_temp and krylov_iters
         are always local.
         
         Attributes
         ----------
         n : int
             Number of state variables.
         delta_location : str
             Memory location for delta buffer: 'local' or 'shared'.
         residual_location : str
             Memory location for residual buffer: 'local' or 'shared'.
         linear_solver_buffer_settings : LinearSolverBufferSettings
             Buffer settings for the nested linear solver.
         """
         n: int = attrs.field(validator=getype_validator(int, 1))
         delta_location: str = attrs.field(
             default='shared', validator=validators.in_(["local", "shared"])
         )
         residual_location: str = attrs.field(
             default='shared', validator=validators.in_(["local", "shared"])
         )
         linear_solver_buffer_settings: Optional[LinearSolverBufferSettings] = (
             attrs.field(default=None)
         )
         
         @property
         def use_shared_delta(self) -> bool:
             return self.delta_location == 'shared'
         
         @property
         def use_shared_residual(self) -> bool:
             return self.residual_location == 'shared'
         
         @property
         def shared_memory_elements(self) -> int:
             total = 0
             if self.use_shared_delta:
                 total += self.n
             if self.use_shared_residual:
                 total += self.n
             # Add linear solver shared memory
             if self.linear_solver_buffer_settings is not None:
                 total += self.linear_solver_buffer_settings.shared_memory_elements
             return total
         
         @property
         def local_memory_elements(self) -> int:
             total = 0
             if not self.use_shared_delta:
                 total += self.n
             if not self.use_shared_residual:
                 total += self.n
             # residual_temp and krylov_iters always local
             total += self.n  # residual_temp
             total += 1       # krylov_iters (int32, but counted as 1 element)
             # Add linear solver local memory
             if self.linear_solver_buffer_settings is not None:
                 total += self.linear_solver_buffer_settings.local_memory_elements
             return total
         
         @property
         def local_sizes(self) -> NewtonLocalSizes:
             return NewtonLocalSizes(
                 delta=self.n,
                 residual=self.n,
                 residual_temp=self.n,
                 krylov_iters=1,
             )
         
         @property
         def shared_indices(self) -> NewtonSliceIndices:
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
             
             return NewtonSliceIndices(
                 delta=delta_slice,
                 residual=residual_slice,
                 local_end=ptr,
                 lin_solver_start=ptr,
             )
     ```
   - Edge cases: linear_solver_buffer_settings can be None (for backwards compat)
   - Integration: Used by newton_krylov_solver_factory

4. **Add required imports to newton_krylov.py**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details: Add imports for attrs, validators, BufferSettings base classes,
     LinearSolverBufferSettings, and getype_validator
     ```python
     import attrs
     from attrs import validators
     from typing import Optional
     from cubie._utils import getype_validator
     from cubie.BufferSettings import BufferSettings, LocalSizes, SliceIndices
     from cubie.integrators.matrix_free_solvers.linear_solver import (
         LinearSolverBufferSettings
     )
     ```
   - Edge cases: None
   - Integration: Needed for new classes

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (150 lines added)
- Functions/Methods Added/Modified:
  * NewtonLocalSizes class
  * NewtonSliceIndices class  
  * NewtonBufferSettings class
- Implementation Summary:
  Added NewtonBufferSettings infrastructure with delta/residual location
  toggles, shared_memory_elements and local_memory_elements properties,
  and support for nested linear_solver_buffer_settings.
- Issues Flagged: None

---

## Task Group 2: Update newton_krylov_solver_factory - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 16-302)

**Input Validation Required**:
- buffer_settings: Optional[NewtonBufferSettings], if None create default

**Tasks**:

1. **Add buffer_settings parameter to newton_krylov_solver_factory**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     - Add `buffer_settings: Optional[NewtonBufferSettings] = None` parameter
     - Default to all-shared buffers when None (current implicit behavior)
     - Extract compile-time constants from buffer_settings
     - Update device function to use selective allocation
     ```python
     def newton_krylov_solver_factory(
         residual_function: Callable,
         linear_solver: Callable,
         n: int,
         tolerance: float,
         max_iters: int,
         damping: float = 0.5,
         max_backtracks: int = 8,
         precision: PrecisionDType = np.float32,
         buffer_settings: Optional[NewtonBufferSettings] = None,
     ) -> Callable:
         # ... docstring update ...
         
         # Default buffer settings - shared delta/residual (current behavior)
         if buffer_settings is None:
             buffer_settings = NewtonBufferSettings(n=n)
         
         # Extract compile-time flags
         delta_shared = buffer_settings.use_shared_delta
         residual_shared = buffer_settings.use_shared_residual
         shared_indices = buffer_settings.shared_indices
         delta_slice = shared_indices.delta
         residual_slice = shared_indices.residual
         lin_solver_start = shared_indices.lin_solver_start
         local_sizes = buffer_settings.local_sizes
         delta_local_size = local_sizes.nonzero('delta')
         residual_local_size = local_sizes.nonzero('residual')
     ```
   - Edge cases: buffer_settings=None maintains backwards compatibility
   - Integration: Called from DIRK/FIRK build_implicit_helpers

2. **Update device function for selective allocation**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details: Replace hardcoded slicing with compile-time conditional allocation
     ```python
     # Inside newton_krylov_solver device function:
     # Current (lines 154-155):
     #   delta = shared_scratch[:n]
     #   residual = shared_scratch[n: 2 * n]
     # 
     # Replace with:
     if delta_shared:
         delta = shared_scratch[delta_slice]
     else:
         delta = cuda.local.array(delta_local_size, precision)
         for _i in range(delta_local_size):
             delta[_i] = typed_zero
     
     if residual_shared:
         residual = shared_scratch[residual_slice]
     else:
         residual = cuda.local.array(residual_local_size, precision)
         for _i in range(residual_local_size):
             residual[_i] = typed_zero
     
     # Update lin_shared to use compile-time offset:
     # Current (line 197):
     #   lin_shared = shared_scratch[2 * n:]
     # Replace with:
     lin_shared = shared_scratch[lin_solver_start:]
     ```
   - Edge cases: Zero-sized arrays when n=0
   - Integration: Must maintain existing function signature

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (30 lines changed)
- Functions/Methods Added/Modified:
  * newton_krylov_solver_factory() - added buffer_settings parameter
  * newton_krylov_solver device function - updated for selective allocation
- Implementation Summary:
  Updated factory to accept optional buffer_settings, extract compile-time
  flags, and use selective allocation for delta/residual buffers.
  Linear solver start offset now uses lin_solver_start from shared_indices.
- Issues Flagged: None

---

## Task Group 3: Wire LinearSolverBufferSettings Through Algorithms - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 522-590)
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 497-571)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 154-405)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (entire file)

**Input Validation Required**:
- buffer_settings passed to linear_solver_factory: Optional[LinearSolverBufferSettings]
- buffer_settings passed to newton_krylov_solver_factory: Optional[NewtonBufferSettings]

**Tasks**:

1. **Update linear_solver_factory to create buffer_settings if None**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Verify (already done at line 221-222)
   - Details: Verify buffer_settings defaulting is present:
     ```python
     if buffer_settings is None:
         buffer_settings = LinearSolverBufferSettings(n=n)
     ```
   - Edge cases: None
   - Integration: Called from DIRK/FIRK build_implicit_helpers

2. **Update DIRKStep.build_implicit_helpers to pass buffer_settings**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details: Create LinearSolverBufferSettings and NewtonBufferSettings,
     pass through solver factory chain
     ```python
     def build_implicit_helpers(self) -> Callable:
         # ... existing code ...
         
         # Create buffer settings for solver chain
         from cubie.integrators.matrix_free_solvers.linear_solver import (
             LinearSolverBufferSettings
         )
         from cubie.integrators.matrix_free_solvers.newton_krylov import (
             NewtonBufferSettings
         )
         
         linear_buffer_settings = LinearSolverBufferSettings(n=n)
         newton_buffer_settings = NewtonBufferSettings(
             n=n,
             linear_solver_buffer_settings=linear_buffer_settings,
         )
         
         linear_solver = linear_solver_factory(
             operator,
             n=n,
             preconditioner=preconditioner,
             correction_type=correction_type,
             tolerance=krylov_tolerance,
             max_iters=max_linear_iters,
             buffer_settings=linear_buffer_settings,
         )
         
         nonlinear_solver = newton_krylov_solver_factory(
             residual_function=residual,
             linear_solver=linear_solver,
             n=n,
             tolerance=newton_tolerance,
             max_iters=max_newton_iters,
             damping=newton_damping,
             max_backtracks=newton_max_backtracks,
             precision=precision,
             buffer_settings=newton_buffer_settings,
         )
         
         return nonlinear_solver
     ```
   - Edge cases: None
   - Integration: Updates solver chain construction

3. **Update FIRKStep.build_implicit_helpers similarly**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details: Same pattern as DIRK, but use all_stages_n instead of n
     ```python
     def build_implicit_helpers(self) -> Callable:
         # ... existing code ...
         
         from cubie.integrators.matrix_free_solvers.linear_solver import (
             LinearSolverBufferSettings
         )
         from cubie.integrators.matrix_free_solvers.newton_krylov import (
             NewtonBufferSettings
         )
         
         # FIRK uses all_stages_n for the coupled system
         linear_buffer_settings = LinearSolverBufferSettings(n=all_stages_n)
         newton_buffer_settings = NewtonBufferSettings(
             n=all_stages_n,
             linear_solver_buffer_settings=linear_buffer_settings,
         )
         
         linear_solver = linear_solver_factory(
             operator,
             n=all_stages_n,
             preconditioner=preconditioner,
             correction_type=correction_type,
             tolerance=krylov_tolerance,
             max_iters=max_linear_iters,
             buffer_settings=linear_buffer_settings,
         )
         
         nonlinear_solver = newton_krylov_solver_factory(
             residual_function=residual,
             linear_solver=linear_solver,
             n=all_stages_n,
             tolerance=newton_tolerance,
             max_iters=max_newton_iters,
             damping=newton_damping,
             max_backtracks=newton_max_backtracks,
             precision=precision,
             buffer_settings=newton_buffer_settings,
         )
         
         return nonlinear_solver
     ```
   - Edge cases: all_stages_n = stage_count * n
   - Integration: Updates solver chain construction

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_dirk.py (20 lines changed)
  * src/cubie/integrators/algorithms/generic_firk.py (20 lines changed)
- Functions/Methods Added/Modified:
  * DIRKStep.build_implicit_helpers() - passes buffer_settings through chain
  * FIRKStep.build_implicit_helpers() - passes buffer_settings through chain
- Implementation Summary:
  Updated both algorithm files to extract newton_buffer_settings from
  config.buffer_settings and pass linear_buffer_settings to linear_solver_factory
  and newton_buffer_settings to newton_krylov_solver_factory.
- Issues Flagged: None

---

## Task Group 4: Fix DIRKBufferSettings solver_scratch Calculation - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 3

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 110-313)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (NewtonBufferSettings)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (LinearSolverBufferSettings)

**Input Validation Required**:
- None (calculations based on existing validated inputs)

**Tasks**:

1. **Add newton_buffer_settings to DIRKBufferSettings**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details: Add Optional[NewtonBufferSettings] attribute for solver memory
     ```python
     @attrs.define
     class DIRKBufferSettings(BufferSettings):
         # ... existing attributes ...
         
         # NEW: Newton solver buffer settings for accurate memory accounting
         newton_buffer_settings: Optional[NewtonBufferSettings] = attrs.field(
             default=None,
         )
         
         @property
         def solver_scratch_elements(self) -> int:
             """Return the number of solver scratch elements.
             
             When newton_buffer_settings is provided, returns its shared
             memory requirement (which includes 2*n + linear solver).
             Otherwise falls back to 2*n for backwards compatibility.
             """
             if self.newton_buffer_settings is not None:
                 return self.newton_buffer_settings.shared_memory_elements
             # Fallback for backwards compatibility
             return 2 * self.n
     ```
   - Edge cases: newton_buffer_settings=None maintains backwards compat
   - Integration: Used by shared_memory_elements property

2. **Update DIRKBufferSettings.shared_memory_elements**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Verify
   - Details: Verify it uses solver_scratch_elements property (which it does):
     ```python
     @property
     def shared_memory_elements(self) -> int:
         total = 0
         if self.use_shared_accumulator:
             total += self.accumulator_length
         if self.use_shared_solver_scratch:
             total += self.solver_scratch_elements  # Now includes linear solver
         # ... rest unchanged ...
     ```
   - Edge cases: None
   - Integration: Flows through to SingleIntegratorRun

3. **Update DIRKStep.__init__ to create newton_buffer_settings**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details: Create and wire NewtonBufferSettings when creating DIRKBufferSettings
     ```python
     def __init__(self, ...):
         # ... existing setup ...
         
         # Import at top of file:
         # from cubie.integrators.matrix_free_solvers.linear_solver import (
         #     LinearSolverBufferSettings
         # )
         # from cubie.integrators.matrix_free_solvers.newton_krylov import (
         #     NewtonBufferSettings
         # )
         
         # Create solver buffer settings for memory accounting
         linear_buffer_settings = LinearSolverBufferSettings(n=n)
         newton_buffer_settings = NewtonBufferSettings(
             n=n,
             linear_solver_buffer_settings=linear_buffer_settings,
         )
         
         buffer_kwargs = {
             'n': n,
             'stage_count': tableau.stage_count,
             'newton_buffer_settings': newton_buffer_settings,
         }
         # ... rest of location settings ...
     ```
   - Edge cases: None
   - Integration: Ensures correct memory accounting at init time

4. **Add required imports to generic_dirk.py**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details: Add imports at top of file:
     ```python
     from typing import Callable, Optional
     # Add after existing imports:
     from cubie.integrators.matrix_free_solvers.linear_solver import (
         LinearSolverBufferSettings
     )
     from cubie.integrators.matrix_free_solvers.newton_krylov import (
         NewtonBufferSettings
     )
     ```
   - Edge cases: Circular import prevention - use local imports if needed
   - Integration: May need to defer to local imports in methods

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_dirk.py (30 lines changed)
- Functions/Methods Added/Modified:
  * DIRKBufferSettings - added newton_buffer_settings attribute
  * DIRKBufferSettings.solver_scratch_elements - updated to use newton settings
  * DIRKStep.__init__() - creates newton_buffer_settings for memory accounting
- Implementation Summary:
  Added newton_buffer_settings attribute to DIRKBufferSettings. Updated
  solver_scratch_elements to return newton.shared_memory_elements when
  available. Updated __init__ to create nested buffer settings chain.
  Used local imports to avoid circular dependencies.
- Issues Flagged: None

---

## Task Group 5: Fix FIRKBufferSettings solver_scratch Calculation - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 3

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 102-269)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (NewtonBufferSettings)

**Input Validation Required**:
- None (calculations based on existing validated inputs)

**Tasks**:

1. **Add newton_buffer_settings to FIRKBufferSettings**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details: Same pattern as DIRK but using all_stages_n
     ```python
     @attrs.define
     class FIRKBufferSettings(BufferSettings):
         # ... existing attributes ...
         
         # NEW: Newton solver buffer settings for accurate memory accounting
         newton_buffer_settings: Optional[NewtonBufferSettings] = attrs.field(
             default=None,
         )
         
         @property
         def solver_scratch_elements(self) -> int:
             """Return solver scratch elements (includes linear solver).
             
             When newton_buffer_settings is provided, returns its shared
             memory requirement. Otherwise falls back to 2*all_stages_n.
             """
             if self.newton_buffer_settings is not None:
                 return self.newton_buffer_settings.shared_memory_elements
             return 2 * self.all_stages_n
     ```
   - Edge cases: newton_buffer_settings=None maintains backwards compat
   - Integration: Used by shared_memory_elements property

2. **Update FIRKStep.__init__ to create newton_buffer_settings**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details: Create and wire NewtonBufferSettings
     ```python
     def __init__(self, ...):
         # ... existing setup ...
         
         # Create solver buffer settings for memory accounting
         from cubie.integrators.matrix_free_solvers.linear_solver import (
             LinearSolverBufferSettings
         )
         from cubie.integrators.matrix_free_solvers.newton_krylov import (
             NewtonBufferSettings
         )
         
         all_stages_n_val = tableau.stage_count * n
         linear_buffer_settings = LinearSolverBufferSettings(n=all_stages_n_val)
         newton_buffer_settings = NewtonBufferSettings(
             n=all_stages_n_val,
             linear_solver_buffer_settings=linear_buffer_settings,
         )
         
         buffer_kwargs = {
             'n': n,
             'stage_count': tableau.stage_count,
             'n_drivers': n_drivers,
             'newton_buffer_settings': newton_buffer_settings,
         }
         # ... rest of location settings ...
     ```
   - Edge cases: all_stages_n = stage_count * n
   - Integration: Ensures correct memory accounting at init time

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_firk.py (30 lines changed)
- Functions/Methods Added/Modified:
  * FIRKBufferSettings - added newton_buffer_settings attribute
  * FIRKBufferSettings.solver_scratch_elements - updated to use newton settings
  * FIRKStep.__init__() - creates newton_buffer_settings for memory accounting
- Implementation Summary:
  Same pattern as DIRK but using all_stages_n for the coupled system.
  Added newton_buffer_settings attribute and updated solver_scratch_elements.
  Created nested buffer settings in __init__ with local imports.
- Issues Flagged: None

---

## Task Group 6: Fix RosenbrockBufferSettings Lazy Initialization - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 100-236, 437-501)

**Input Validation Required**:
- None (fix lazy initialization issue)

**Tasks**:

1. **Document cached_auxiliary_count lazy initialization behavior**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify (add documentation)
   - Details: Add clear docstring explaining the lazy pattern and how users
     should access buffer_settings only after build_implicit_helpers is called:
     ```python
     @attrs.define
     class RosenbrockBufferSettings(BufferSettings):
         """Configuration for Rosenbrock step buffer sizes and memory locations.
         
         ... existing docstring ...
         
         Notes
         -----
         The cached_auxiliary_count is initially 0 and is updated after
         build_implicit_helpers() is called on GenericRosenbrockWStep.
         Memory calculations requiring the final count should access
         buffer_settings after the step has been built.
         """
     ```
   - Edge cases: cached_auxiliary_count=0 before build
   - Integration: Documentation only, no code change

2. **Verify buffer_settings update in build_implicit_helpers**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Verify (already done at lines 478-481)
   - Details: Verify the cached_auxiliary_count update is present:
     ```python
     self._cached_auxiliary_count = get_fn("cached_aux_count")
     self.compile_settings.buffer_settings.cached_auxiliary_count = (
         self._cached_auxiliary_count
     )
     ```
   - Edge cases: None
   - Integration: Existing code is correct

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (7 lines added)
- Functions/Methods Added/Modified:
  * RosenbrockBufferSettings docstring - added Notes section
- Implementation Summary:
  Added documentation explaining the lazy initialization pattern for
  cached_auxiliary_count. Verified existing update in build_implicit_helpers.
- Issues Flagged: None

---

## Task Group 7: Add LoopBufferSettings proposed_counters Fix - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 159-626)

**Input Validation Required**:
- None (audit existing implementation)

**Tasks**:

1. **Audit LoopBufferSettings proposed_counters handling**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Verify
   - Details: Verify that proposed_counters is correctly handled:
     - shared_memory_elements (lines 419-422): Adds 2 when n_counters > 0 and shared
     - local_memory_elements: Does NOT add proposed_counters for local (bug?)
     - calculate_shared_indices (lines 567-573): Correct handling
     - local_sizes (lines 613-614): Correct handling (2 if n_counters > 0 else 0)
   - Edge cases: n_counters=0 means no proposed_counters
   - Integration: Need to verify local_memory_elements includes proposed_counters

2. **Fix local_memory_elements to include proposed_counters**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details: Add proposed_counters to local_memory_elements when counters
     are local:
     ```python
     @property
     def local_memory_elements(self) -> int:
         """Return total local memory elements required by loop buffers."""
         total = 0
         # ... existing code ...
         if not self.use_shared_counters:
             total += self.counters_size
             # Add proposed_counters (2 elements when active)
             if self.n_counters > 0:
                 total += 2
         # ... rest unchanged ...
     ```
   - Edge cases: n_counters=0 means proposed_counters is not used
   - Integration: Flows through to SingleIntegratorRun

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/loops/ode_loop.py (3 lines changed)
- Functions/Methods Added/Modified:
  * LoopBufferSettings.local_memory_elements - added proposed_counters
- Implementation Summary:
  Fixed local_memory_elements to include proposed_counters (2 elements)
  when counters are local and n_counters > 0. This mirrors the shared
  memory accounting in shared_memory_elements.
- Issues Flagged: None

---

## Task Group 8: Unit Tests for BufferSettings - PARALLEL
**Status**: [x]
**Dependencies**: Task Groups 1-7

**Required Context**:
- File: tests/integrators/ (test structure)
- File: tests/conftest.py (fixtures)

**Input Validation Required**:
- None (test code)

**Tasks**:

1. **Create test_buffer_settings.py for base classes**
   - File: tests/test_buffer_settings.py
   - Action: Create
   - Details: Test base class behavior:
     ```python
     """Tests for base BufferSettings infrastructure."""
     import pytest
     from cubie.BufferSettings import BufferSettings, LocalSizes, SliceIndices
     
     
     class TestLocalSizes:
         def test_nonzero_returns_value_when_positive(self):
             # Implementation uses nonzero() to get minimum 1 for cuda.local.array
             pass
         
         def test_nonzero_returns_one_when_zero(self):
             pass
     
     
     class TestSliceIndices:
         # SliceIndices is abstract, test via subclasses
         pass
     ```
   - Edge cases: Zero-sized buffers must return 1 from nonzero()
   - Integration: Foundation tests

2. **Create test_newton_buffer_settings.py**
   - File: tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py
   - Action: Create
   - Details: Test NewtonBufferSettings:
     ```python
     """Tests for Newton solver BufferSettings."""
     import pytest
     from cubie.integrators.matrix_free_solvers.newton_krylov import (
         NewtonBufferSettings,
         NewtonLocalSizes,
         NewtonSliceIndices,
     )
     from cubie.integrators.matrix_free_solvers.linear_solver import (
         LinearSolverBufferSettings,
     )
     
     
     class TestNewtonBufferSettings:
         def test_shared_memory_elements_default(self):
             # Default: delta and residual shared
             settings = NewtonBufferSettings(n=10)
             assert settings.shared_memory_elements == 20  # 2 * n
         
         def test_shared_memory_elements_with_linear_solver(self):
             lin_settings = LinearSolverBufferSettings(n=10)
             settings = NewtonBufferSettings(
                 n=10,
                 linear_solver_buffer_settings=lin_settings,
             )
             # 2*n (newton) + lin_solver shared
             expected = 20 + lin_settings.shared_memory_elements
             assert settings.shared_memory_elements == expected
         
         def test_local_memory_elements(self):
             settings = NewtonBufferSettings(n=10)
             # residual_temp (n) + krylov_iters (1)
             assert settings.local_memory_elements == 11
         
         def test_shared_indices_contiguous(self):
             settings = NewtonBufferSettings(n=10)
             indices = settings.shared_indices
             assert indices.delta.stop == indices.residual.start
             assert indices.local_end == indices.residual.stop
     ```
   - Edge cases: All local, all shared, mixed configurations
   - Integration: Verifies NewtonBufferSettings works correctly

3. **Create test_algorithm_buffer_settings.py**
   - File: tests/integrators/algorithms/test_algorithm_buffer_settings.py
   - Action: Create
   - Details: Test algorithm BufferSettings integration:
     ```python
     """Tests for algorithm BufferSettings integration."""
     import pytest
     import numpy as np
     from cubie.integrators.algorithms.generic_dirk import (
         DIRKBufferSettings,
         DIRKStep,
     )
     from cubie.integrators.algorithms.generic_firk import (
         FIRKBufferSettings,
         FIRKStep,
     )
     
     
     class TestDIRKBufferSettings:
         def test_solver_scratch_includes_linear_solver(self):
             # With newton_buffer_settings, solver_scratch should include
             # linear solver shared memory
             pass
         
         def test_shared_memory_elements_accurate(self):
             pass
     
     
     class TestFIRKBufferSettings:
         def test_solver_scratch_uses_all_stages_n(self):
             pass
     ```
   - Edge cases: Single-stage vs multi-stage algorithms
   - Integration: Verifies algorithm memory accounting

4. **Create test_loop_buffer_settings.py**
   - File: tests/integrators/loops/test_loop_buffer_settings.py
   - Action: Create
   - Details: Test LoopBufferSettings:
     ```python
     """Tests for loop BufferSettings."""
     import pytest
     from cubie.integrators.loops.ode_loop import (
         LoopBufferSettings,
         LoopLocalSizes,
         LoopSliceIndices,
     )
     
     
     class TestLoopBufferSettings:
         def test_proposed_counters_in_local_memory(self):
             # When counters are local, proposed_counters should be counted
             settings = LoopBufferSettings(
                 n_states=10,
                 n_counters=4,
                 counters_location='local',
             )
             # Should include counters (4) + proposed_counters (2)
             local = settings.local_memory_elements
             assert local >= 6  # At least counters + proposed_counters
         
         def test_shared_indices_contiguous(self):
             settings = LoopBufferSettings(
                 n_states=10,
                 n_parameters=5,
                 state_buffer_location='shared',
                 parameters_location='shared',
             )
             indices = settings.shared_indices
             # Verify no gaps between slices
             assert indices.state.stop == indices.proposed_state.start or \
                    indices.proposed_state == slice(0, 0)
     ```
   - Edge cases: Zero counters, zero drivers, etc.
   - Integration: Verifies loop memory accounting

**Outcomes**: 
- Files Created:
  * tests/test_buffer_settings.py (67 lines)
  * tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py (125 lines)
- Files Modified:
  * tests/integrators/algorithms/test_buffer_settings.py (40 lines added)
  * tests/integrators/loops/test_buffer_settings.py (12 lines added)
- Tests Added:
  * TestLocalSizes.test_nonzero_* - base class nonzero behavior
  * TestNewtonBufferSettings - full coverage of Newton buffer settings
  * TestDIRKBufferSettings.test_solver_scratch_with_newton_buffer_settings
  * TestFIRKBufferSettings.test_solver_scratch_with_newton_buffer_settings
  * TestLoopBufferSettings.test_proposed_counters_in_local_memory
- Implementation Summary:
  Created comprehensive test coverage for new NewtonBufferSettings class
  and updated existing algorithm/loop test files with new test cases.
- Issues Flagged: None

---

## Summary

### Total Task Groups: 8
### Dependency Chain:
1. Task Group 1 (NewtonBufferSettings) - Foundation
2. Task Group 2 (Update factory) - Depends on 1
3. Task Group 3 (Wire through algorithms) - Depends on 2
4. Task Group 4 (DIRK fix) - Depends on 3
5. Task Group 5 (FIRK fix) - Depends on 3
6. Task Group 6 (Rosenbrock docs) - Independent
7. Task Group 7 (Loop proposed_counters) - Independent
8. Task Group 8 (Tests) - Depends on all others

### Parallel Execution Opportunities:
- Task Groups 4 and 5 can run in parallel (both depend on 3)
- Task Groups 6 and 7 can run in parallel with Groups 1-5
- Task Group 8 should run last

### Estimated Complexity:
- Task Group 1: Medium (new classes, follows existing patterns)
- Task Group 2: Medium (factory update, device function changes)
- Task Group 3: Medium (wiring changes in two files)
- Task Group 4: Low (property updates, imports)
- Task Group 5: Low (same pattern as Group 4)
- Task Group 6: Low (documentation only)
- Task Group 7: Low (small fix)
- Task Group 8: Medium (comprehensive tests)

### Critical Path:
1 → 2 → 3 → 4/5 → 8

Total estimated files to modify: 6 source files, 4 test files

---

# Implementation Complete - Ready for Review

## Execution Summary
- Total Task Groups: 8
- Completed: 8
- Failed: 0
- Total Files Modified: 10

## Task Group Completion
- Group 1: [x] NewtonBufferSettings Creation - Complete
- Group 2: [x] Update newton_krylov_solver_factory - Complete
- Group 3: [x] Wire LinearSolverBufferSettings Through Algorithms - Complete
- Group 4: [x] Fix DIRKBufferSettings solver_scratch Calculation - Complete
- Group 5: [x] Fix FIRKBufferSettings solver_scratch Calculation - Complete
- Group 6: [x] Fix RosenbrockBufferSettings Lazy Initialization - Complete
- Group 7: [x] Add LoopBufferSettings proposed_counters Fix - Complete
- Group 8: [x] Unit Tests for BufferSettings - Complete

## All Modified Files
1. src/cubie/integrators/matrix_free_solvers/newton_krylov.py (150+ lines)
2. src/cubie/integrators/matrix_free_solvers/__init__.py (8 lines)
3. src/cubie/integrators/algorithms/generic_dirk.py (45 lines)
4. src/cubie/integrators/algorithms/generic_firk.py (45 lines)
5. src/cubie/integrators/algorithms/generic_rosenbrock_w.py (7 lines)
6. src/cubie/integrators/loops/ode_loop.py (3 lines)
7. tests/test_buffer_settings.py (67 lines - new)
8. tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py (125 lines - new)
9. tests/integrators/algorithms/test_buffer_settings.py (40 lines added)
10. tests/integrators/loops/test_buffer_settings.py (12 lines added)

## Flagged Issues
None

## Handoff to Reviewer
All implementation tasks complete. Task list updated with outcomes.
Ready for reviewer agent to validate against user stories and goals.
