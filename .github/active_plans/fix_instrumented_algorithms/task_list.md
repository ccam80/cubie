# Implementation Task List
# Feature: Fix Instrumented Algorithms
# Plan Reference: .github/active_plans/fix_instrumented_algorithms/agent_plan.md

## Task Group 1: Production backwards_euler.py Enhancement - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler.py (entire file)
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 25-83, 156-159)
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 105-127, 289-345) - pattern reference

**Input Validation Required**:
- No additional validation needed; uses existing ImplicitStepConfig validators

**Tasks**:

### Task 1.1: Create BackwardsEulerStepConfig class
- File: src/cubie/integrators/algorithms/backwards_euler.py
- Action: Create
- Details:
  ```python
  @attrs.define
  class BackwardsEulerStepConfig(ImplicitStepConfig):
      """Configuration for Backward Euler step with buffer location control."""
      
      increment_cache_location: str = attrs.field(
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])
      )
  ```
- Location: After imports, before ALGO_CONSTANTS
- Integration: Import attrs at top of file (already imported via attrs.validators usage in parent)

### Task 1.2: Add import for attrs module
- File: src/cubie/integrators/algorithms/backwards_euler.py
- Action: Modify imports
- Details:
  ```python
  import attrs
  ```
- Location: Add after existing imports, before `from numba import cuda, int32`

### Task 1.3: Update __init__ to accept increment_cache_location parameter
- File: src/cubie/integrators/algorithms/backwards_euler.py
- Action: Modify
- Details:
  - Add parameter `increment_cache_location: Optional[str] = None` to __init__ signature
  - Build BackwardsEulerStepConfig instead of ImplicitStepConfig
  - Conditionally include increment_cache_location in config_kwargs if not None
  ```python
  def __init__(
      self,
      precision: PrecisionDType,
      n: int,
      ...
      newton_max_backtracks: Optional[int] = None,
      increment_cache_location: Optional[str] = None,  # NEW
  ) -> None:
      ...
      # Build config kwargs conditionally
      config_kwargs = {
          'precision': precision,
          'n': n,
          'get_solver_helper_fn': get_solver_helper_fn,
          'beta': beta,
          'gamma': gamma,
          'M': M,
          'dxdt_function': dxdt_function,
          'observables_function': observables_function,
          'driver_function': driver_function,
      }
      if preconditioner_order is not None:
          config_kwargs['preconditioner_order'] = preconditioner_order
      if increment_cache_location is not None:
          config_kwargs['increment_cache_location'] = increment_cache_location
      
      config = BackwardsEulerStepConfig(**config_kwargs)
      ...
  ```

### Task 1.4: Add register_buffers() method
- File: src/cubie/integrators/algorithms/backwards_euler.py
- Action: Create method
- Details:
  ```python
  def register_buffers(self) -> None:
      """Register buffers with buffer_registry."""
      config = self.compile_settings
      buffer_registry.clear_parent(self)
      
      # Register solver child allocators
      _ = buffer_registry.get_child_allocators(
          self, self.solver, name='solver'
      )
      
      # Register increment cache buffer
      buffer_registry.register(
          'increment_cache',
          self,
          config.n,
          config.increment_cache_location,
          aliases='solver_shared',
          persistent=True,
          precision=config.precision
      )
  ```
- Location: After __init__, before build_step
- Integration: Called at end of __init__ after super().__init__

### Task 1.5: Call register_buffers() in __init__
- File: src/cubie/integrators/algorithms/backwards_euler.py
- Action: Modify
- Details:
  - Add `self.register_buffers()` call after `super().__init__` call
  ```python
  super().__init__(config, BE_DEFAULTS.copy(), **solver_kwargs)
  self.register_buffers()
  ```

### Task 1.6: Update build_step to use buffer_registry allocator for increment_cache
- File: src/cubie/integrators/algorithms/backwards_euler.py
- Action: Modify
- Details:
  - Get allocator from buffer_registry instead of using solver_scratch directly for increment_cache
  ```python
  def build_step(...):
      ...
      # Get allocators from buffer registry
      alloc_solver_shared, alloc_solver_persistent = (
          buffer_registry.get_child_allocators(self, self.solver, name='solver')
      )
      alloc_increment_cache = buffer_registry.get_allocator('increment_cache', self)
      
      # Inside step function:
      solver_scratch = alloc_solver_shared(shared, persistent_local)
      solver_persistent = alloc_solver_persistent(shared, persistent_local)
      increment_cache = alloc_increment_cache(shared, persistent_local)
      
      # Use increment_cache instead of solver_scratch for stage increment storage
      for i in range(n):
          proposed_state[i] = increment_cache[i]  # Changed from solver_scratch
      ...
      for i in range(n):
          increment_cache[i] = proposed_state[i]  # Changed from solver_scratch
          proposed_state[i] += state[i]
  ```

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: Verify Production generic_dirk.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (entire file)

**Input Validation Required**:
- None - verification only

**Tasks**:

### Task 2.1: Verify production generic_dirk.py uses self.solver
- File: src/cubie/integrators/algorithms/generic_dirk.py
- Action: Verify (no changes needed if correct)
- Details:
  - Confirm that the file uses `self.solver` (not `self._newton_solver`)
  - Current state appears correct - uses `self.solver` throughout
  - No changes required

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: Verify Production crank_nicolson.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/crank_nicolson.py (entire file)

**Input Validation Required**:
- None - verification only

**Tasks**:

### Task 3.1: Verify production crank_nicolson.py register_buffers()
- File: src/cubie/integrators/algorithms/crank_nicolson.py
- Action: Verify (no changes needed if correct)
- Details:
  - Confirm register_buffers() uses correct alias 'solver_shared'
  - Current state appears correct
  - No changes required

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: Verify Production generic_rosenbrock_w.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (entire file)

**Input Validation Required**:
- None - verification only

**Tasks**:

### Task 4.1: Verify production generic_rosenbrock_w.py build_step
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
- Action: Verify (no changes needed if correct)
- Details:
  - Confirm driver_del_t comes from config (not function parameter)
  - Current state appears correct: `driver_del_t = config.driver_del_t`
  - No changes required

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: Fix Instrumented backwards_euler.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/integrators/algorithms/instrumented/backwards_euler.py (entire file)
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (lines 38-260, 409-725)
- File: src/cubie/integrators/algorithms/backwards_euler.py (entire file - after Task Group 1)
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 271-320)

**Input Validation Required**:
- None - uses parent class validation

**Tasks**:

### Task 5.1: Fix __init__ - remove duplicate super().__init__ and nested function
- File: tests/integrators/algorithms/instrumented/backwards_euler.py
- Action: Modify
- Details:
  - Current file has TWO `super().__init__` calls at lines 124 and 196 - keep ONLY the first one
  - Current file has `def build_implicit_helpers(self)` INSIDE __init__ - this is wrong
  - Remove the nested function definition and the second super().__init__ call
  - The __init__ should end after line 124's super().__init__ call
  ```python
  def __init__(
      self,
      ...
  ) -> None:
      ...
      super().__init__(config, BE_DEFAULTS.copy(), **solver_kwargs)
      # END of __init__ - remove everything after this until build_step
  ```

### Task 5.2: Add build_implicit_helpers method at class level
- File: tests/integrators/algorithms/instrumented/backwards_euler.py
- Action: Create method
- Details:
  ```python
  def build_implicit_helpers(self) -> None:
      """Construct instrumented nonlinear solver chain."""
      config = self.compile_settings
      beta = config.beta
      gamma = config.gamma
      mass = config.M
      preconditioner_order = config.preconditioner_order
      n = config.n
      precision = config.precision

      get_fn = config.get_solver_helper_fn

      preconditioner = get_fn(
          'neumann_preconditioner',
          beta=beta,
          gamma=gamma,
          mass=mass,
          preconditioner_order=preconditioner_order
      )
      residual = get_fn(
          'stage_residual',
          beta=beta,
          gamma=gamma,
          mass=mass,
          preconditioner_order=preconditioner_order
      )
      operator = get_fn(
          'linear_operator',
          beta=beta,
          gamma=gamma,
          mass=mass,
          preconditioner_order=preconditioner_order
      )

      # Create instrumented linear solver
      linear_solver = InstrumentedLinearSolver(
          precision=precision,
          n=n,
          correction_type=self.linear_correction_type,
          krylov_tolerance=self.krylov_tolerance,
          max_linear_iters=self.max_linear_iters,
      )
      linear_solver.update(
          operator_apply=operator,
          preconditioner=preconditioner,
      )

      # Create instrumented Newton solver
      newton_solver = InstrumentedNewtonKrylov(
          precision=precision,
          n=n,
          linear_solver=linear_solver,
          newton_tolerance=self.newton_tolerance,
          max_newton_iters=self.max_newton_iters,
          newton_damping=self.newton_damping,
          newton_max_backtracks=self.newton_max_backtracks,
      )
      newton_solver.update(residual_function=residual)

      # Replace parent solver with instrumented version
      self.solver = newton_solver

      self.update_compile_settings(
          solver_function=self.solver.device_function
      )
  ```
- Location: After __init__, before build_step

### Task 5.3: Fix build_step - remove anti-pattern buffer registration
- File: tests/integrators/algorithms/instrumented/backwards_euler.py
- Action: Modify
- Details:
  - Remove the inline buffer_registry.register call (lines 241-246)
  - This should be handled by a register_buffers method or inherited from production
  ```python
  def build_step(
      self,
      ...
  ) -> StepCache:
      a_ij = numba_precision(1.0)
      has_driver_function = driver_function is not None
      driver_function = driver_function
      n = int32(n)
      
      # Get child allocators for Newton solver
      alloc_solver_shared, alloc_solver_persistent = (
          buffer_registry.get_child_allocators(self, self.solver,
                                               name='solver')
      )
      # REMOVE the buffer_registry.register call that was here
      
      solver_fn = solver_function
      ...
  ```

### Task 5.4: Fix solver call in step function to pass logging arrays with stage_index
- File: tests/integrators/algorithms/instrumented/backwards_euler.py
- Action: Verify
- Details:
  - The current solver call already includes logging arrays (lines 395-417)
  - Verify stage_index is passed as int32(0) for single-stage method
  - Current implementation appears correct

### Task 5.5: Remove unused NewtonKrylovConfig import
- File: tests/integrators/algorithms/instrumented/backwards_euler.py
- Action: Modify
- Details:
  - Remove `from cubie import NewtonKrylovConfig` import (line 8) - not used

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 6: Fix Instrumented crank_nicolson.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 3

**Required Context**:
- File: tests/integrators/algorithms/instrumented/crank_nicolson.py (entire file)
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (lines 38-260, 409-725)
- File: src/cubie/integrators/algorithms/crank_nicolson.py (entire file)

**Input Validation Required**:
- None - uses parent class validation

**Tasks**:

### Task 6.1: Add build_implicit_helpers method
- File: tests/integrators/algorithms/instrumented/crank_nicolson.py
- Action: Create method
- Details:
  ```python
  def build_implicit_helpers(self) -> None:
      """Construct instrumented nonlinear solver chain."""
      config = self.compile_settings
      beta = config.beta
      gamma = config.gamma
      mass = config.M
      preconditioner_order = config.preconditioner_order
      n = config.n
      precision = config.precision

      get_fn = config.get_solver_helper_fn

      preconditioner = get_fn(
          'neumann_preconditioner',
          beta=beta,
          gamma=gamma,
          mass=mass,
          preconditioner_order=preconditioner_order
      )
      residual = get_fn(
          'stage_residual',
          beta=beta,
          gamma=gamma,
          mass=mass,
          preconditioner_order=preconditioner_order
      )
      operator = get_fn(
          'linear_operator',
          beta=beta,
          gamma=gamma,
          mass=mass,
          preconditioner_order=preconditioner_order
      )

      # Create instrumented linear solver
      linear_solver = InstrumentedLinearSolver(
          precision=precision,
          n=n,
          correction_type=self.linear_correction_type,
          krylov_tolerance=self.krylov_tolerance,
          max_linear_iters=self.max_linear_iters,
      )
      linear_solver.update(
          operator_apply=operator,
          preconditioner=preconditioner,
      )

      # Create instrumented Newton solver
      newton_solver = InstrumentedNewtonKrylov(
          precision=precision,
          n=n,
          linear_solver=linear_solver,
          newton_tolerance=self.newton_tolerance,
          max_newton_iters=self.max_newton_iters,
          newton_damping=self.newton_damping,
          newton_max_backtracks=self.newton_max_backtracks,
      )
      newton_solver.update(residual_function=residual)

      # Replace parent solver with instrumented version
      self.solver = newton_solver

      self.update_compile_settings(
          solver_function=self.solver.device_function
      )
  ```
- Location: After register_buffers(), before build_step

### Task 6.2: Add imports for instrumented solvers
- File: tests/integrators/algorithms/instrumented/crank_nicolson.py
- Action: Modify
- Details:
  ```python
  from integrators.algorithms.instrumented.matrix_free_solvers import (
      InstrumentedLinearSolver,
      InstrumentedNewtonKrylov,
  )
  ```
- Location: After existing imports

### Task 6.3: Fix solver calls in step function to pass logging arrays
- File: tests/integrators/algorithms/instrumented/crank_nicolson.py
- Action: Modify
- Details:
  - The current solver calls (lines 352-363, 380-391) do NOT include logging arrays
  - Add logging array parameters to both solver calls
  - First solver call uses stage_index=int32(0), second uses int32(1)
  ```python
  # First solver call (CN step)
  status = solver_fn(
      proposed_state,
      parameters,
      proposed_drivers,
      end_time,
      dt_scalar,
      stage_coefficient,
      base_state,
      solver_scratch,
      solver_persistent,
      counters,
      int32(0),  # stage_index
      newton_initial_guesses,
      newton_iteration_guesses,
      newton_residuals,
      newton_squared_norms,
      newton_iteration_scale,
      linear_initial_guesses,
      linear_iteration_guesses,
      linear_residuals,
      linear_squared_norms,
      linear_preconditioned_vectors,
  )
  
  # Second solver call (BE step)
  be_status = solver_fn(
      base_state,
      parameters,
      proposed_drivers,
      end_time,
      dt_scalar,
      be_coefficient,
      state,
      solver_scratch,
      solver_persistent,
      counters,
      int32(1),  # stage_index
      newton_initial_guesses,
      newton_iteration_guesses,
      newton_residuals,
      newton_squared_norms,
      newton_iteration_scale,
      linear_initial_guesses,
      linear_iteration_guesses,
      linear_residuals,
      linear_squared_norms,
      linear_preconditioned_vectors,
  )
  ```

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 7: Fix Instrumented generic_dirk.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 2

**Required Context**:
- File: tests/integrators/algorithms/instrumented/generic_dirk.py (entire file)
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (lines 38-260, 409-725)
- File: src/cubie/integrators/algorithms/generic_dirk.py (entire file)

**Input Validation Required**:
- None - uses parent class validation

**Tasks**:

### Task 7.1: Add build_implicit_helpers method
- File: tests/integrators/algorithms/instrumented/generic_dirk.py
- Action: Create method
- Details:
  ```python
  def build_implicit_helpers(self) -> None:
      """Construct instrumented nonlinear solver chain."""
      config = self.compile_settings
      beta = config.beta
      gamma = config.gamma
      mass = config.M
      preconditioner_order = config.preconditioner_order
      n = config.n
      precision = config.precision

      get_fn = config.get_solver_helper_fn

      preconditioner = get_fn(
          'neumann_preconditioner',
          beta=beta,
          gamma=gamma,
          mass=mass,
          preconditioner_order=preconditioner_order
      )
      residual = get_fn(
          'stage_residual',
          beta=beta,
          gamma=gamma,
          mass=mass,
          preconditioner_order=preconditioner_order
      )
      operator = get_fn(
          'linear_operator',
          beta=beta,
          gamma=gamma,
          mass=mass,
          preconditioner_order=preconditioner_order
      )

      # Create instrumented linear solver
      linear_solver = InstrumentedLinearSolver(
          precision=precision,
          n=n,
          correction_type=self.linear_correction_type,
          krylov_tolerance=self.krylov_tolerance,
          max_linear_iters=self.max_linear_iters,
      )
      linear_solver.update(
          operator_apply=operator,
          preconditioner=preconditioner,
      )

      # Create instrumented Newton solver
      newton_solver = InstrumentedNewtonKrylov(
          precision=precision,
          n=n,
          linear_solver=linear_solver,
          newton_tolerance=self.newton_tolerance,
          max_newton_iters=self.max_newton_iters,
          newton_damping=self.newton_damping,
          newton_max_backtracks=self.newton_max_backtracks,
      )
      newton_solver.update(residual_function=residual)

      # Replace parent solver with instrumented version
      self.solver = newton_solver

      self.update_compile_settings(
          solver_function=self.solver.device_function
      )
  ```
- Location: After register_buffers(), before build_step

### Task 7.2: Add imports for instrumented solvers
- File: tests/integrators/algorithms/instrumented/generic_dirk.py
- Action: Modify
- Details:
  ```python
  from integrators.algorithms.instrumented.matrix_free_solvers import (
      InstrumentedLinearSolver,
      InstrumentedNewtonKrylov,
  )
  ```
- Location: After existing imports

### Task 7.3: Fix solver calls in step function to pass logging arrays
- File: tests/integrators/algorithms/instrumented/generic_dirk.py
- Action: Modify
- Details:
  - The current solver calls (lines 510-521, 637-648) do NOT include logging arrays
  - Add logging array parameters to solver calls
  - Pass stage_idx as slot index for multi-stage method
  ```python
  # Stage 0 solver call
  if stage_implicit[0]:
      solver_status = nonlinear_solver(
          stage_increment,
          parameters,
          proposed_drivers,
          stage_time,
          dt_scalar,
          diagonal_coeffs[0],
          stage_base,
          solver_shared,
          solver_persistent,
          counters,
          int32(0),  # stage_index
          newton_initial_guesses,
          newton_iteration_guesses,
          newton_residuals,
          newton_squared_norms,
          newton_iteration_scale,
          linear_initial_guesses,
          linear_iteration_guesses,
          linear_residuals,
          linear_squared_norms,
          linear_preconditioned_vectors,
      )
      ...
  
  # Stages 1-s solver call (inside loop)
  if stage_implicit[stage_idx]:
      solver_status = nonlinear_solver(
          stage_increment,
          parameters,
          proposed_drivers,
          stage_time,
          dt_scalar,
          diagonal_coeffs[stage_idx],
          stage_base,
          solver_shared,
          solver_persistent,
          counters,
          stage_idx,  # stage_index
          newton_initial_guesses,
          newton_iteration_guesses,
          newton_residuals,
          newton_squared_norms,
          newton_iteration_scale,
          linear_initial_guesses,
          linear_iteration_guesses,
          linear_residuals,
          linear_squared_norms,
          linear_preconditioned_vectors,
      )
  ```

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 8: Fix Instrumented generic_firk.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: tests/integrators/algorithms/instrumented/generic_firk.py (entire file)
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (lines 38-260, 409-725)
- File: src/cubie/integrators/algorithms/generic_firk.py (entire file)

**Input Validation Required**:
- None - uses parent class validation

**Tasks**:

### Task 8.1: Add build_implicit_helpers method
- File: tests/integrators/algorithms/instrumented/generic_firk.py
- Action: Create method
- Details:
  ```python
  def build_implicit_helpers(self) -> None:
      """Construct instrumented nonlinear solver chain."""
      config = self.compile_settings
      tableau = config.tableau
      beta = config.beta
      gamma = config.gamma
      mass = config.M
      n = config.n
      precision = config.precision

      get_fn = config.get_solver_helper_fn

      stage_coefficients = [list(row) for row in tableau.a]
      stage_nodes = list(tableau.c)

      residual = get_fn(
          'n_stage_residual',
          beta=beta,
          gamma=gamma,
          mass=mass,
          stage_coefficients=stage_coefficients,
          stage_nodes=stage_nodes,
      )
      operator = get_fn(
          'n_stage_linear_operator',
          beta=beta,
          gamma=gamma,
          mass=mass,
          stage_coefficients=stage_coefficients,
          stage_nodes=stage_nodes,
      )
      preconditioner = get_fn(
          'n_stage_neumann_preconditioner',
          beta=beta,
          gamma=gamma,
          preconditioner_order=config.preconditioner_order,
          stage_coefficients=stage_coefficients,
          stage_nodes=stage_nodes,
      )

      # Create instrumented linear solver (uses n*stage_count for FIRK)
      all_stages_n = tableau.stage_count * n
      linear_solver = InstrumentedLinearSolver(
          precision=precision,
          n=all_stages_n,
          correction_type=self.linear_correction_type,
          krylov_tolerance=self.krylov_tolerance,
          max_linear_iters=self.max_linear_iters,
      )
      linear_solver.update(
          operator_apply=operator,
          preconditioner=preconditioner,
      )

      # Create instrumented Newton solver
      newton_solver = InstrumentedNewtonKrylov(
          precision=precision,
          n=all_stages_n,
          linear_solver=linear_solver,
          newton_tolerance=self.newton_tolerance,
          max_newton_iters=self.max_newton_iters,
          newton_damping=self.newton_damping,
          newton_max_backtracks=self.newton_max_backtracks,
      )
      newton_solver.update(residual_function=residual)

      # Replace parent solver with instrumented version
      self.solver = newton_solver

      self.update_compile_settings(
          solver_function=self.solver.device_function
      )
  ```
- Location: After register_buffers(), before build_step

### Task 8.2: Add imports for instrumented solvers
- File: tests/integrators/algorithms/instrumented/generic_firk.py
- Action: Modify
- Details:
  ```python
  from integrators.algorithms.instrumented.matrix_free_solvers import (
      InstrumentedLinearSolver,
      InstrumentedNewtonKrylov,
  )
  ```
- Location: After existing imports

### Task 8.3: Fix solver call in step function to pass logging arrays
- File: tests/integrators/algorithms/instrumented/generic_firk.py
- Action: Modify
- Details:
  - The current solver call (lines 549-560) does NOT include logging arrays
  - Add logging array parameters
  - FIRK solves all stages simultaneously with single call, use stage_index=int32(0)
  ```python
  solver_status = nonlinear_solver(
      stage_increment,
      parameters,
      stage_driver_stack,
      current_time,
      dt_scalar,
      typed_zero,
      state,
      solver_shared,
      solver_persistent,
      counters,
      int32(0),  # stage_index
      newton_initial_guesses,
      newton_iteration_guesses,
      newton_residuals,
      newton_squared_norms,
      newton_iteration_scale,
      linear_initial_guesses,
      linear_iteration_guesses,
      linear_residuals,
      linear_squared_norms,
      linear_preconditioned_vectors,
  )
  ```

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 9: Fix Instrumented generic_rosenbrock_w.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 4

**Required Context**:
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py (entire file)
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (lines 38-260)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (entire file)

**Input Validation Required**:
- None - uses parent class validation

**Tasks**:

### Task 9.1: Add build_implicit_helpers method
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
- Action: Create method
- Details:
  - Rosenbrock uses LINEAR solver only (no Newton iteration)
  ```python
  def build_implicit_helpers(self) -> None:
      """Construct instrumented linear solver for Rosenbrock methods."""
      config = self.compile_settings
      beta = config.beta
      gamma = config.gamma
      mass = config.M
      preconditioner_order = config.preconditioner_order
      n = config.n
      precision = config.precision

      get_fn = config.get_solver_helper_fn

      preconditioner = get_fn(
          'neumann_preconditioner',
          beta=beta,
          gamma=gamma,
          mass=mass,
          preconditioner_order=preconditioner_order
      )
      operator = get_fn(
          'linear_operator',
          beta=beta,
          gamma=gamma,
          mass=mass,
          preconditioner_order=preconditioner_order
      )

      prepare_jacobian = get_fn(
          'prepare_jac',
          preconditioner_order=preconditioner_order,
      )
      self._cached_auxiliary_count = get_fn('cached_aux_count')

      # Update buffer registry with the actual cached_auxiliary_count
      buffer_registry.update_buffer(
          'cached_auxiliaries', self,
          size=self._cached_auxiliary_count
      )

      time_derivative_function = get_fn('time_derivative_rhs')

      # Create instrumented linear solver
      linear_solver = InstrumentedLinearSolver(
          precision=precision,
          n=n,
          correction_type=self.linear_correction_type,
          krylov_tolerance=self.krylov_tolerance,
          max_linear_iters=self.max_linear_iters,
      )
      linear_solver.update(
          operator_apply=operator,
          preconditioner=preconditioner,
          use_cached_auxiliaries=True,
      )

      # Replace parent solver with instrumented version
      self.solver = linear_solver

      self.update_compile_settings(
          solver_function=self.solver.device_function,
          time_derivative_function=time_derivative_function,
          prepare_jacobian_function=prepare_jacobian
      )
  ```
- Location: After register_buffers(), before build_step

### Task 9.2: Add imports for instrumented solvers
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
- Action: Modify
- Details:
  ```python
  from integrators.algorithms.instrumented.matrix_free_solvers import (
      InstrumentedLinearSolver,
  )
  ```
- Location: After existing imports

### Task 9.3: Fix linear solver calls in step function to pass logging arrays
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
- Action: Modify
- Details:
  - The current linear solver calls (lines 498-512, 656-670) do NOT include logging arrays
  - Add logging array parameters
  - Pass stage_idx as slot index for multi-stage method
  - Note: Rosenbrock uses the CACHED variant of linear solver
  ```python
  # Stage 0 linear solver call
  status_code |= linear_solver(
      state,
      parameters,
      drivers_buffer,
      base_state_placeholder,
      cached_auxiliaries,
      stage_time,
      dt_scalar,
      numba_precision(1.0),
      stage_rhs,
      stage_increment,
      shared,
      krylov_iters_out,
      int32(0),  # slot_index
      linear_initial_guesses,
      linear_iteration_guesses,
      linear_residuals,
      linear_squared_norms,
      linear_preconditioned_vectors,
  )
  
  # Stages 1-s linear solver call (inside loop)
  status_code |= linear_solver(
      state,
      parameters,
      drivers_buffer,
      base_state_placeholder,
      cached_auxiliaries,
      stage_time,
      dt_scalar,
      numba_precision(1.0),
      stage_rhs,
      stage_increment,
      shared,
      krylov_iters_out,
      stage_idx,  # slot_index
      linear_initial_guesses,
      linear_iteration_guesses,
      linear_residuals,
      linear_squared_norms,
      linear_preconditioned_vectors,
  )
  ```

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Dependency Graph

```
Task Group 1 (backwards_euler production) ──┐
                                            ├──> Task Group 5 (backwards_euler instrumented)
                                            │
Task Group 2 (dirk production - verify) ────┼──> Task Group 7 (dirk instrumented)
                                            │
Task Group 3 (crank_nicolson production) ───┼──> Task Group 6 (crank_nicolson instrumented)
                                            │
Task Group 4 (rosenbrock production) ───────┼──> Task Group 9 (rosenbrock instrumented)
                                            │
                                            └──> Task Group 8 (firk instrumented) - independent
```

## Execution Order

1. **First Wave (PARALLEL)**: Task Groups 1, 2, 3, 4 (production file changes/verification)
2. **Second Wave (PARALLEL after First)**: Task Groups 5, 6, 7, 8, 9 (instrumented file fixes)

## Edge Cases

1. **Single-stage vs multi-stage algorithms**: 
   - Single-stage (BE, CN) always use stage_index=0 or 0/1 for dual solves
   - Multi-stage (DIRK, FIRK, Rosenbrock) use loop stage_idx

2. **Newton vs Linear solvers**: 
   - Rosenbrock uses LinearSolver only (no Newton iteration)
   - Others use NewtonKrylov which contains LinearSolver

3. **Logging array dimensions**: 
   - Arrays sized for max_stages * max_iterations
   - Unused slots remain zero-initialized

4. **FIRK n dimension**: 
   - FIRK solves all stages simultaneously
   - Linear solver n = n * stage_count (flattened)

## Estimated Complexity

- Task Group 1: Medium (new config class + register_buffers + build_step updates)
- Task Groups 2-4: Low (verification only)
- Task Groups 5-9: Medium-High (method additions + solver call modifications)

Total: ~9 task groups, ~30 individual tasks
