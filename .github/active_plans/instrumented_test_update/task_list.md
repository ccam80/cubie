# Implementation Task List
# Feature: Instrumented Test Files Update
# Plan Reference: .github/active_plans/instrumented_test_update/agent_plan.md

## Overview

This task list provides detailed, function-level implementation tasks for updating the instrumented test files to match the refactored production code. The key change is migrating from fixed shared memory slicing to the new `BufferSettings` pattern that enables selective allocation between shared and local memory.

**Note**: The instrumented files have additional logging arrays that must be preserved. Only the memory allocation patterns, device function signatures, and computational logic need updating.

---

## Task Group 1: matrix_free_solvers.py - [SEQUENTIAL]
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (entire file)
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (entire file)

**Input Validation Required**:
- None - factory functions validate parameters via production code patterns

**Tasks**:

### 1.1 Update `inst_linear_solver_factory` Function
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
- Action: Modify
- Details:
  ```python
  def inst_linear_solver_factory(
      operator_apply: Callable,
      n: int,
      preconditioner: Optional[Callable] = None,
      correction_type: str = "minimal_residual",
      tolerance: float = 1e-6,
      max_iters: int = 100,
      precision: PrecisionDType = np.float64,
  ) -> Callable:
      """Create an instrumented steepest-descent or minimal-residual solver."""
      
      # Add these compile-time constants matching production:
      n_val = int32(n)
      max_iters = int32(max_iters)
      precision_dtype = np.dtype(precision)
      precision_scalar = from_dtype(precision_dtype)
      typed_zero = precision_scalar(0.0)
      tol_squared = tolerance * tolerance
      
      # Update iteration counter pattern to match production:
      # - Use iter_count = int32(0) 
      # - Increment at start of loop with iter_count += int32(1)
      # - Return: return_status |= (iter_count + int32(1)) << 16
  ```
- Edge cases: Loop termination with early break on convergence
- Integration: Used by all instrumented implicit algorithm step classes

### 1.2 Update Device Function Signature to Match Production
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
- Action: Modify the `@cuda.jit` decorator and function parameters for `linear_solver`
- Details:
  The instrumented version does NOT need the `shared` parameter from production since it uses local arrays only for simplicity. However, ensure these patterns match:
  ```python
  # Iteration counter pattern (match production):
  iter_count = int32(0)
  for _ in range(max_iters):
      iter_count += int32(1)
      # ... iteration logic
      if all_sync(mask, converged):
          return_status = int32(0)
          return_status |= (iter_count + int32(1)) << 16
          return return_status
  return_status = int32(4)
  return_status |= (iter_count + int32(1)) << 16
  return return_status
  ```
- Edge cases: Warp-level synchronization for convergence check
- Integration: Called by `inst_newton_krylov_solver_factory`

### 1.3 Update `inst_linear_solver_cached_factory` Function
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
- Action: Modify to match production iteration pattern
- Details:
  Same iteration counter pattern as 1.2. The cached version already accepts `cached_aux` parameter which matches production.

### 1.4 Update `inst_newton_krylov_solver_factory` Signature
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
- Action: Modify device function signature
- Details:
  Update the `@cuda.jit` decorator to use contiguous array specifiers:
  ```python
  @cuda.jit(
      [(numba_precision[::1],      # stage_increment (was [:])
        numba_precision[::1],      # parameters (was [:])
        numba_precision[::1],      # drivers (was [:])
        numba_precision,           # t
        numba_precision,           # h
        numba_precision,           # a_ij
        numba_precision[::1],      # base_state (was [:])
        numba_precision[::1],      # shared_scratch (was [:])
        int32[::1],                # counters (was [:])
        int32,                     # stage_index
        # ... remaining instrumentation arrays unchanged
       )],
      device=True,
      inline=True)
  ```
- Edge cases: Arrays used as scratch may have non-contiguous slices internally
- Integration: Called by instrumented implicit step classes

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: backwards_euler.py - [SEQUENTIAL]
**Status**: [ ] (Partially complete - dt param already removed)
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler.py (lines 102-260)
- File: tests/integrators/algorithms/instrumented/backwards_euler.py (entire file)

**Input Validation Required**:
- None - validation handled by base classes

**Tasks**:

### 2.1 Update Device Function Signature with Contiguous Specifiers
- File: tests/integrators/algorithms/instrumented/backwards_euler.py
- Action: Modify `@cuda.jit` decorator in `build_step`
- Details:
  ```python
  @cuda.jit(
      (
          numba_precision[::1],      # state (was [:])
          numba_precision[::1],      # proposed_state (was [:])
          numba_precision[::1],      # parameters (was [:])
          numba_precision[:, :, ::1], # driver_coefficients (was [:,:,:])
          numba_precision[::1],      # drivers_buffer (was [:])
          numba_precision[::1],      # proposed_drivers (was [:])
          numba_precision[::1],      # observables (was [:])
          numba_precision[::1],      # proposed_observables (was [:])
          numba_precision[::1],      # error (was [:])
          numba_precision[:, ::1],   # residuals (was [:,:])
          numba_precision[:, ::1],   # jacobian_updates (was [:,:])
          # ... continue for all 2D arrays with [:, ::1]
          # ... continue for all 3D arrays with [:, :, ::1]
          numba_precision,           # dt_scalar
          numba_precision,           # time_scalar
          int16,                     # first_step_flag
          int16,                     # accepted_flag
          numba_precision[::1],      # shared (was [:])
          numba_precision[::1],      # persistent_local (was [:])
          int32[::1],                # counters (was [:])
      ),
      device=True,
      inline=True,
  )
  ```
- Edge cases: 3D arrays like driver_coefficients need `[:, :, ::1]`
- Integration: Compatible with production step signature

### 2.2 Add n as int32 for Consistency
- File: tests/integrators/algorithms/instrumented/backwards_euler.py
- Action: Modify `build_step` method
- Details:
  At the start of `build_step`, add:
  ```python
  n = int32(n)  # Cast to int32 for Numba loop bounds
  ```
  This matches the production pattern where n is explicitly typed.
- Edge cases: None
- Integration: Used in all loop bounds

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: backwards_euler_predict_correct.py - [SEQUENTIAL]
**Status**: [ ]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py (if exists)
- File: tests/integrators/algorithms/instrumented/backwards_euler_predict_correct.py (entire file)

**Input Validation Required**:
- None - inherits from BackwardsEulerStep

**Tasks**:

### 3.1 Update Device Function Signature with Contiguous Specifiers
- File: tests/integrators/algorithms/instrumented/backwards_euler_predict_correct.py
- Action: Modify `@cuda.jit` decorator
- Details:
  Same pattern as Task 2.1 - update all 1D arrays to `[::1]`, 2D to `[:, ::1]`, 3D to `[:, :, ::1]`
- Edge cases: None
- Integration: Extends BackwardsEulerStep

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: crank_nicolson.py - [SEQUENTIAL]
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/crank_nicolson.py (if exists, for reference)
- File: tests/integrators/algorithms/instrumented/crank_nicolson.py (entire file)

**Input Validation Required**:
- None - validation handled by base classes

**Tasks**:

### 4.1 Update Device Function Signature with Contiguous Specifiers
- File: tests/integrators/algorithms/instrumented/crank_nicolson.py
- Action: Modify `@cuda.jit` decorator in `build_step`
- Details:
  Same pattern as Task 2.1 - update all array specifiers:
  - 1D arrays: `numba_precision[:]` → `numba_precision[::1]`
  - 2D arrays: `numba_precision[:, :]` → `numba_precision[:, ::1]`
  - 3D arrays: `numba_precision[:, :, :]` → `numba_precision[:, :, ::1]`
- Edge cases: None
- Integration: Compatible with production step signature

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: explicit_euler.py - [SEQUENTIAL]
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/explicit_euler.py (for reference)
- File: tests/integrators/algorithms/instrumented/explicit_euler.py (entire file)

**Input Validation Required**:
- None - simple explicit method

**Tasks**:

### 5.1 Update Device Function Signature with Contiguous Specifiers
- File: tests/integrators/algorithms/instrumented/explicit_euler.py
- Action: Modify `@cuda.jit` decorator in `build_step`
- Details:
  Same pattern as previous groups - update all array type specifiers to use contiguous markers.
- Edge cases: None
- Integration: Simplest explicit method, no solver dependencies

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 6: generic_erk.py - [SEQUENTIAL]
**Status**: [ ]
**Dependencies**: None (explicit method)

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_erk.py (lines 50-260 for BufferSettings, lines 450-800 for step)
- File: tests/integrators/algorithms/instrumented/generic_erk.py (entire file)

**Input Validation Required**:
- None - tableau validation handled by production

**Tasks**:

### 6.1 Add ERKBufferSettings Import and Integration
- File: tests/integrators/algorithms/instrumented/generic_erk.py
- Action: Modify imports and ERKStepConfig
- Details:
  The instrumented version can use a simplified approach that defaults to all-shared memory (matching current behavior). Add to imports:
  ```python
  from cubie.integrators.algorithms.generic_erk import (
      ERKBufferSettings,
      ERKLocalSizes,
      ERKSliceIndices,
  )
  ```
  
  Update ERKStepConfig:
  ```python
  @attrs.define
  class ERKStepConfig(ExplicitStepConfig):
      """Configuration describing an explicit Runge--Kutta integrator."""
      
      tableau: ERKTableau = attrs.field(default=DEFAULT_ERK_TABLEAU)
      buffer_settings: Optional[ERKBufferSettings] = attrs.field(
          default=None,
          validator=validators.optional(
              validators.instance_of(ERKBufferSettings)
          ),
      )
      
      @property
      def first_same_as_last(self) -> bool:
          """Return ``True`` when the tableau shares the first and last stage."""
          return self.tableau.first_same_as_last
  ```
- Edge cases: Validators import needed
- Integration: Matches production ERKStepConfig

### 6.2 Update ERKStep.__init__ to Create Buffer Settings
- File: tests/integrators/algorithms/instrumented/generic_erk.py
- Action: Modify `__init__` method
- Details:
  ```python
  def __init__(
      self,
      precision: PrecisionDType,
      n: int,
      dxdt_function: Optional[Callable] = None,
      observables_function: Optional[Callable] = None,
      driver_function: Optional[Callable] = None,
      get_solver_helper_fn: Optional[Callable] = None,
      tableau: ERKTableau = DEFAULT_ERK_TABLEAU,
      n_drivers: int = 0,
      stage_rhs_location: Optional[str] = None,
      stage_accumulator_location: Optional[str] = None,
  ) -> None:
      # Create buffer_settings with default shared memory locations
      buffer_kwargs = {
          'n': n,
          'stage_count': tableau.stage_count,
      }
      if stage_rhs_location is not None:
          buffer_kwargs['stage_rhs_location'] = stage_rhs_location
      if stage_accumulator_location is not None:
          buffer_kwargs['stage_accumulator_location'] = stage_accumulator_location
      
      # Default to shared memory for instrumented (matches old behavior)
      if 'stage_rhs_location' not in buffer_kwargs:
          buffer_kwargs['stage_rhs_location'] = 'shared'
      if 'stage_accumulator_location' not in buffer_kwargs:
          buffer_kwargs['stage_accumulator_location'] = 'shared'
      
      buffer_settings = ERKBufferSettings(**buffer_kwargs)
      
      config_kwargs = {
          "precision": precision,
          "n": n,
          "n_drivers": n_drivers,
          "dxdt_function": dxdt_function,
          "observables_function": observables_function,
          "driver_function": driver_function,
          "get_solver_helper_fn": get_solver_helper_fn,
          "tableau": tableau,
          "buffer_settings": buffer_settings,
      }
      config = ERKStepConfig(**config_kwargs)
      # ... rest unchanged
  ```
- Edge cases: Default to shared for backward compatibility with tests
- Integration: Allows tests to optionally specify memory locations

### 6.3 Update Device Function Signature with Contiguous Specifiers
- File: tests/integrators/algorithms/instrumented/generic_erk.py
- Action: Modify `@cuda.jit` decorator in `build_step`
- Details:
  Same pattern as previous groups - all arrays get contiguous specifiers.
- Edge cases: None
- Integration: Compatible with production

### 6.4 Update build_step Buffer Allocation to Use BufferSettings
- File: tests/integrators/algorithms/instrumented/generic_erk.py
- Action: Modify `build_step` method
- Details:
  The instrumented version can keep the simpler fixed-shared approach for now since tests pass. However, ensure these lines are updated:
  ```python
  # Extract buffer settings for slicing
  buffer_settings = config.buffer_settings
  stage_rhs_shared = buffer_settings.use_shared_stage_rhs
  stage_accumulator_shared = buffer_settings.use_shared_stage_accumulator
  stage_cache_shared = buffer_settings.use_shared_stage_cache
  
  # Use slice indices from buffer_settings
  shared_indices = buffer_settings.shared_indices
  stage_rhs_slice = shared_indices.stage_rhs
  stage_accumulator_slice = shared_indices.stage_accumulator
  stage_cache_slice = shared_indices.stage_cache
  ```
  
  In the step function, use selective allocation:
  ```python
  # Simplified: Keep using shared only for instrumented (matches current tests)
  stage_accumulator = shared[stage_accumulator_slice]
  stage_rhs = cuda.local.array(n, numba_precision)  # Keep local for safety
  if multistage:
      stage_cache = stage_accumulator[:n]
  ```
- Edge cases: stage_cache aliasing when both rhs and accumulator are shared
- Integration: Maintains test compatibility

### 6.5 Update shared_memory_required Property
- File: tests/integrators/algorithms/instrumented/generic_erk.py  
- Action: Modify property
- Details:
  ```python
  @property
  def shared_memory_required(self) -> int:
      """Return the number of precision entries required in shared memory."""
      return self.compile_settings.buffer_settings.shared_memory_elements
  ```
- Edge cases: None
- Integration: Matches production pattern

### 6.6 Add persistent_local_required Property
- File: tests/integrators/algorithms/instrumented/generic_erk.py
- Action: Add new property
- Details:
  ```python
  @property
  def persistent_local_required(self) -> int:
      """Return the number of persistent local entries required."""
      buffer_settings = self.compile_settings.buffer_settings
      return buffer_settings.persistent_local_elements
  ```
- Edge cases: Returns 0 when stage_cache aliases shared memory
- Integration: Matches production interface

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 7: generic_dirk.py - [SEQUENTIAL]
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 50-320 for BufferSettings, lines 590-1040 for step)
- File: tests/integrators/algorithms/instrumented/generic_dirk.py (entire file)

**Input Validation Required**:
- None - tableau and solver validation handled by production

**Tasks**:

### 7.1 Add DIRKBufferSettings Import and Integration
- File: tests/integrators/algorithms/instrumented/generic_dirk.py
- Action: Modify imports and DIRKStepConfig
- Details:
  Add imports:
  ```python
  from attrs import validators
  from cubie.integrators.algorithms.generic_dirk import (
      DIRKBufferSettings,
      DIRKLocalSizes,
      DIRKSliceIndices,
  )
  ```
  
  Update DIRKStepConfig:
  ```python
  @attrs.define
  class DIRKStepConfig(ImplicitStepConfig):
      """Configuration describing the DIRK integrator."""
      
      tableau: DIRKTableau = attrs.field(
          default=DEFAULT_DIRK_TABLEAU,
      )
      buffer_settings: Optional[DIRKBufferSettings] = attrs.field(
          default=None,
          validator=validators.optional(
              validators.instance_of(DIRKBufferSettings)
          ),
      )
  ```
- Edge cases: Ensure attrs validators import
- Integration: Matches production DIRKStepConfig

### 7.2 Update DIRKStep.__init__ to Create Buffer Settings
- File: tests/integrators/algorithms/instrumented/generic_dirk.py
- Action: Modify `__init__` method
- Details:
  ```python
  def __init__(
      self,
      # ... existing params ...
      stage_increment_location: Optional[str] = None,
      stage_base_location: Optional[str] = None,
      accumulator_location: Optional[str] = None,
      solver_scratch_location: Optional[str] = None,
  ) -> None:
      mass = np.eye(n, dtype=precision)
      
      # Create buffer_settings with default shared memory locations
      buffer_kwargs = {
          'n': n,
          'stage_count': tableau.stage_count,
      }
      if stage_increment_location is not None:
          buffer_kwargs['stage_increment_location'] = stage_increment_location
      if stage_base_location is not None:
          buffer_kwargs['stage_base_location'] = stage_base_location
      if accumulator_location is not None:
          buffer_kwargs['accumulator_location'] = accumulator_location
      if solver_scratch_location is not None:
          buffer_kwargs['solver_scratch_location'] = solver_scratch_location
      
      # Default to shared memory (matches current instrumented behavior)
      if 'accumulator_location' not in buffer_kwargs:
          buffer_kwargs['accumulator_location'] = 'shared'
      if 'solver_scratch_location' not in buffer_kwargs:
          buffer_kwargs['solver_scratch_location'] = 'shared'
      
      buffer_settings = DIRKBufferSettings(**buffer_kwargs)
      
      config_kwargs = {
          # ... existing config kwargs ...
          "buffer_settings": buffer_settings,
      }
      # ... rest unchanged
  ```
- Edge cases: Default to shared for backward compatibility
- Integration: Allows optional memory location configuration

### 7.3 Update Device Function Signature with Contiguous Specifiers
- File: tests/integrators/algorithms/instrumented/generic_dirk.py
- Action: Modify `@cuda.jit` decorator in `build_step`
- Details:
  Same pattern - all arrays get contiguous specifiers.
- Edge cases: None
- Integration: Compatible with production

### 7.4 Update build_step Buffer Allocation  
- File: tests/integrators/algorithms/instrumented/generic_dirk.py
- Action: Modify `build_step` method
- Details:
  The instrumented version keeps simple fixed-shared slicing for test stability:
  ```python
  # Extract buffer settings
  buffer_settings = config.buffer_settings
  
  # Simplified: Continue using shared memory indices directly
  # (Instrumented version doesn't need full selective allocation)
  accumulator_length = max(stage_count - 1, 0) * n
  solver_shared_elements = self.solver_shared_elements
  
  # Shared memory layout (unchanged from current)
  acc_start = 0
  acc_end = accumulator_length
  solver_start = acc_end
  solver_end = acc_end + solver_shared_elements
  ```
  
  The step function body can remain mostly unchanged since it already uses shared memory properly.
- Edge cases: solver_scratch slicing for increment_cache/rhs_cache
- Integration: Maintains test compatibility

### 7.5 Update persistent_local_required Property
- File: tests/integrators/algorithms/instrumented/generic_dirk.py
- Action: Modify property (currently returns 0)
- Details:
  ```python
  @property
  def persistent_local_required(self) -> int:
      """Return the number of persistent local entries required."""
      buffer_settings = self.compile_settings.buffer_settings
      return buffer_settings.persistent_local_elements
  ```
- Edge cases: Returns 0 when solver_scratch is shared
- Integration: Matches production interface

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 8: generic_firk.py - [SEQUENTIAL]
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (for reference, if exists)
- File: tests/integrators/algorithms/instrumented/generic_firk.py (entire file)

**Input Validation Required**:
- None - tableau validation handled by production

**Tasks**:

### 8.1 Update Device Function Signature with Contiguous Specifiers
- File: tests/integrators/algorithms/instrumented/generic_firk.py
- Action: Modify `@cuda.jit` decorator in `build_step`
- Details:
  Same pattern - all arrays get contiguous specifiers:
  - 1D: `[:]` → `[::1]`
  - 2D: `[:, :]` → `[:, ::1]`
  - 3D: `[:, :, :]` → `[:, :, ::1]`
- Edge cases: None
- Integration: Compatible with production

### 8.2 Cast n and Indices to int32
- File: tests/integrators/algorithms/instrumented/generic_firk.py
- Action: Modify `build_step` to cast indices
- Details:
  Ensure loop indices are int32:
  ```python
  n = int32(n)
  stage_count = int32(tableau.stage_count)
  all_stages_n = int32(config.all_stages_n)
  ```
- Edge cases: None
- Integration: Matches production int32 usage

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 9: generic_rosenbrock_w.py - [SEQUENTIAL]
**Status**: [ ]
**Dependencies**: Task Group 1 (uses inst_linear_solver_cached_factory)

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (for reference)
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py (entire file)

**Input Validation Required**:
- None - tableau validation handled by production

**Tasks**:

### 9.1 Update Device Function Signature with Contiguous Specifiers
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
- Action: Modify `@cuda.jit` decorator in `build_step`
- Details:
  Same pattern - all arrays get contiguous specifiers.
- Edge cases: None
- Integration: Compatible with production

### 9.2 Cast Loop Indices to int32
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
- Action: Modify `build_step`
- Details:
  ```python
  stage_count = int32(self.stage_count)
  stages_except_first = stage_count - int32(1)
  n = int32(n)
  ```
- Edge cases: None
- Integration: Matches production

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 10: Run Tests and Validate - [SEQUENTIAL]
**Status**: [ ]
**Dependencies**: Task Groups 1-9

**Required Context**:
- File: tests/integrators/algorithms/instrumented/test_instrumented.py (for test verification)
- File: tests/integrators/algorithms/instrumented/conftest.py (for fixture understanding)

**Input Validation Required**:
- None - tests validate functionality

**Tasks**:

### 10.1 Run Instrumented Tests
- File: N/A (command execution)
- Action: Execute
- Details:
  ```bash
  pytest tests/integrators/algorithms/instrumented/test_instrumented.py -v
  ```
  Verify all 31 tests pass.
- Edge cases: CUDASIM mode may be needed without GPU
- Integration: Final validation

### 10.2 Fix Any Test Failures
- File: Varies based on failures
- Action: Debug and fix
- Details:
  If tests fail, analyze the error messages:
  - Type mismatches: Check array specifiers
  - Iteration count: Check return value encoding
  - Memory access: Check slice indices
- Edge cases: Simulator vs GPU differences
- Integration: Iterative fixes until tests pass

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

### Total Task Groups: 10
### Dependency Chain Overview:
1. **Foundation**: Task Group 1 (matrix_free_solvers.py) - Used by all implicit methods
2. **Simple Implicit**: Task Groups 2-4 (backwards_euler, BEPC, crank_nicolson)
3. **Simple Explicit**: Task Group 5 (explicit_euler)
4. **Complex Explicit**: Task Group 6 (generic_erk)
5. **Complex Implicit**: Task Groups 7-9 (generic_dirk, generic_firk, generic_rosenbrock_w)
6. **Validation**: Task Group 10 (testing)

### Parallel Execution Opportunities:
- Task Groups 2, 3, 4, 5 can run in parallel (all depend only on Group 1)
- Task Group 6 can run in parallel with Task Groups 2-5
- Task Groups 7, 8, 9 can run in parallel after Group 1

### Estimated Complexity:
- **Low**: Groups 2, 3, 4, 5 (simple signature updates)
- **Medium**: Groups 1, 8, 9 (signature updates + some logic changes)
- **High**: Groups 6, 7 (BufferSettings integration + signature updates)

### Key Constraints:
1. **Preserve Instrumentation**: All logging arrays must remain unchanged
2. **Maintain Test Compatibility**: 31 tests must continue passing
3. **Match Production Logic**: Computational behavior must match production
4. **Default to Shared Memory**: For backward compatibility with existing tests
