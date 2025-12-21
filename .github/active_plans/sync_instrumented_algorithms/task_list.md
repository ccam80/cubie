# Implementation Task List
# Feature: Synchronize Instrumented Algorithm Tests
# Plan Reference: .github/active_plans/sync_instrumented_algorithms/agent_plan.md

## Overview

This task list details the synchronization of 6 instrumented algorithm files to match production code exactly, with added logging capabilities. Each task group is self-contained, corresponding to one file, and can be executed independently.

**Goal**: Make each instrumented file a VERBATIM copy of the production file with logging additions.

---

## Task Group 1: backwards_euler.py - SEQUENTIAL

**Status**: [ ]
**Dependencies**: None

**Required Context**:
- Production: `src/cubie/integrators/algorithms/backwards_euler.py` (entire file - 327 lines)
- Instrumented: `tests/integrators/algorithms/instrumented/backwards_euler.py` (entire file)
- Reference: `tests/integrators/algorithms/instrumented/matrix_free_solvers.py` (InstrumentedLinearSolver, InstrumentedNewtonKrylov)

**Input Validation Required**: None - these are verbatim copies

**Tasks**:

### 1.1 Update Imports
- File: `tests/integrators/algorithms/instrumented/backwards_euler.py`
- Action: Modify
- Details:
  ```python
  # FROM (current instrumented):
  from cubie._utils import PrecisionDType
  from cubie.integrators.algorithms import ImplicitStepConfig
  from cubie.integrators.algorithms.base_algorithm_step import (
      StepCache,
      StepControlDefaults,
  )
  from cubie.integrators.algorithms.ode_implicitstep import ODEImplicitStep

  from .matrix_free_solvers import (
      InstrumentedLinearSolver,
      InstrumentedNewtonKrylov,
  )
  from cubie.integrators.matrix_free_solvers.linear_solver import (
      LinearSolverConfig
  )
  from cubie.integrators.matrix_free_solvers.newton_krylov import (
      NewtonKrylovConfig
  )
  
  # TO (match production + instrumented solver imports):
  from cubie._utils import PrecisionDType
  from cubie.buffer_registry import buffer_registry
  from cubie.integrators.algorithms import ImplicitStepConfig
  from cubie.integrators.algorithms.base_algorithm_step import StepCache, \
      StepControlDefaults
  from cubie.integrators.algorithms.ode_implicitstep import ODEImplicitStep

  from .matrix_free_solvers import (
      InstrumentedLinearSolver,
      InstrumentedNewtonKrylov,
  )
  from cubie.integrators.matrix_free_solvers.linear_solver import (
      LinearSolverConfig
  )
  from cubie.integrators.matrix_free_solvers.newton_krylov import (
      NewtonKrylovConfig
  )
  ```
- Edge cases: Must add `buffer_registry` import

### 1.2 Update `__init__` Method Docstring
- File: `tests/integrators/algorithms/instrumented/backwards_euler.py`
- Action: Modify
- Details: Copy VERBATIM docstring from production (lines 47-86)
  - Production has extensive parameter documentation with "If None" clauses
  - Current instrumented has shorter docstrings

### 1.3 Update `__init__` Method Logic
- File: `tests/integrators/algorithms/instrumented/backwards_euler.py`
- Action: Modify
- Details: Match production line 87-119 EXACTLY:
  ```python
  config = ImplicitStepConfig(
      get_solver_helper_fn=get_solver_helper_fn,
      beta=beta,
      gamma=gamma,
      M=M,
      n=n,
      preconditioner_order=preconditioner_order,  # REMOVE "if not None else 1"
      dxdt_function=dxdt_function,
      observables_function=observables_function,
      driver_function=driver_function,
      precision=precision,
  )
  ```
- Current instrumented has: `preconditioner_order=preconditioner_order if preconditioner_order is not None else 1`
- Production REMOVES this conditional

### 1.4 Keep `build_implicit_helpers()` Method
- File: `tests/integrators/algorithms/instrumented/backwards_euler.py`
- Action: Keep as-is (instrumented version)
- Details: The `build_implicit_helpers()` method creates InstrumentedLinearSolver and InstrumentedNewtonKrylov. Production does NOT override this method (parent handles it), but instrumented MUST override to use instrumented solvers. KEEP current instrumented implementation.

### 1.5 Update `build_step()` Method
- File: `tests/integrators/algorithms/instrumented/backwards_euler.py`
- Action: Modify
- Details: Match production `build_step()` signature and internal logic (lines 121-287), with logging additions:
  
  **Production signature** (use same):
  ```python
  def build_step(
      self,
      dxdt_fn: Callable,
      observables_function: Callable,
      driver_function: Optional[Callable],
      solver_function: Callable,
      numba_precision: type,
      n: int,
      n_drivers: int,
  ) -> StepCache:
  ```
  
  **Production buffer allocation pattern** (add to instrumented):
  ```python
  # Get child allocators for Newton solver
  alloc_solver_shared, alloc_solver_persistent = (
      buffer_registry.get_child_allocators(self, self.solver,
                                           name='solver_scratch')
  )
  ```
  
  **Production step function signature** (base, NOT instrumented additions):
  - state, proposed_state, parameters, driver_coefficients, drivers_buffer
  - proposed_drivers, observables, proposed_observables, error
  - dt_scalar, time_scalar, first_step_flag, accepted_flag
  - shared, persistent_local, counters

  **Instrumented additions** (keep from current):
  - residuals, jacobian_updates, stage_states, stage_derivatives
  - stage_observables, stage_drivers, stage_increments
  - newton_initial_guesses, newton_iteration_guesses, newton_residuals
  - newton_squared_norms, newton_iteration_scale
  - linear_initial_guesses, linear_iteration_guesses, linear_residuals
  - linear_squared_norms, linear_preconditioned_vectors

  **Step function body changes**:
  1. Replace `solver_scratch = shared[: solver_shared_elements]` with:
     ```python
     solver_scratch = alloc_solver_shared(shared, persistent_local)
     solver_persistent = alloc_solver_persistent(shared, persistent_local)
     ```
  2. Update solver call to match production signature + logging params:
     ```python
     status = solver_fn(
         proposed_state,
         parameters,
         proposed_drivers,
         next_time,
         dt_scalar,
         a_ij,
         state,
         solver_scratch,
         solver_persistent,  # ADD THIS - production uses it
         counters,
         int32(0),  # stage index for logging
         newton_initial_guesses,
         # ... rest of logging params
     )
     ```
  3. Match production loop structure exactly, adding logging code blocks

### 1.6 Update Properties
- File: `tests/integrators/algorithms/instrumented/backwards_euler.py`
- Action: Verify match with production
- Details: Properties should match production (lines 289-326):
  - `is_multistage`, `is_adaptive`, `threads_per_step`, `settings_dict`
  - `order`, `dxdt_function`, `identifier`

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: crank_nicolson.py - SEQUENTIAL

**Status**: [ ]
**Dependencies**: None

**Required Context**:
- Production: `src/cubie/integrators/algorithms/crank_nicolson.py` (entire file - 401 lines)
- Instrumented: `tests/integrators/algorithms/instrumented/crank_nicolson.py` (entire file)

**Input Validation Required**: None - these are verbatim copies

**Tasks**:

### 2.1 Update Imports
- File: `tests/integrators/algorithms/instrumented/crank_nicolson.py`
- Action: Modify
- Details:
  ```python
  # ADD missing import:
  import attrs
  from cubie.buffer_registry import buffer_registry
  ```

### 2.2 Add `CrankNicolsonStepConfig` Class
- File: `tests/integrators/algorithms/instrumented/crank_nicolson.py`
- Action: Add (MISSING from instrumented)
- Details: Copy VERBATIM from production (lines 35-43):
  ```python
  @attrs.define
  class CrankNicolsonStepConfig(ImplicitStepConfig):
      """Configuration for Crank-Nicolson step."""
      
      dxdt_location: str = attrs.field(
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])
      )
  ```

### 2.3 Update `__init__` Method Signature
- File: `tests/integrators/algorithms/instrumented/crank_nicolson.py`
- Action: Modify
- Details: Add `dxdt_location` parameter (production line 64):
  ```python
  def __init__(
      self,
      precision: PrecisionDType,
      n: int,
      # ... existing params ...
      newton_max_backtracks: Optional[int] = None,
      dxdt_location: Optional[str] = None,  # ADD THIS
  ) -> None:
  ```

### 2.4 Update `__init__` Method Logic
- File: `tests/integrators/algorithms/instrumented/crank_nicolson.py`
- Action: Modify
- Details: Match production lines 116-158:
  1. Build config_kwargs dict conditionally
  2. Use `CrankNicolsonStepConfig` instead of `ImplicitStepConfig`
  3. Add `dxdt_location` to config_kwargs if not None
  4. Call `self.register_buffers()` after parent init

  ```python
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
  if dxdt_location is not None:
      config_kwargs['dxdt_location'] = dxdt_location
  
  config = CrankNicolsonStepConfig(**config_kwargs)
  
  # ... solver_kwargs building ...
  
  super().__init__(config, CN_DEFAULTS.copy(), **solver_kwargs)
  
  self.register_buffers()  # ADD THIS
  ```

### 2.5 Add `register_buffers()` Method
- File: `tests/integrators/algorithms/instrumented/crank_nicolson.py`
- Action: Add (MISSING from instrumented)
- Details: Copy VERBATIM from production (lines 160-170):
  ```python
  def register_buffers(self) -> None:
      """Register buffers with buffer_registry."""
      config = self.compile_settings
      buffer_registry.register(
          'cn_dxdt',
          self,
          config.n,
          config.dxdt_location,
          aliases='solver_scratch_shared',
          precision=config.precision
      )
  ```

### 2.6 Keep `build_implicit_helpers()` Method
- File: `tests/integrators/algorithms/instrumented/crank_nicolson.py`
- Action: Keep as-is (instrumented version)
- Details: Creates InstrumentedLinearSolver and InstrumentedNewtonKrylov

### 2.7 Update `build_step()` Method
- File: `tests/integrators/algorithms/instrumented/crank_nicolson.py`
- Action: Modify
- Details: Match production structure (lines 172-376) with logging:
  
  1. Add allocator setup:
     ```python
     # Get child allocators for Newton solver
     alloc_solver_shared, alloc_solver_persistent = (
         buffer_registry.get_child_allocators(self, self.solver,
                                              name='solver_scratch')
     )
     alloc_dxdt = buffer_registry.get_allocator('cn_dxdt', self)
     ```
  
  2. In step function, replace direct slicing with allocators:
     ```python
     solver_scratch = alloc_solver_shared(shared, persistent_local)
     solver_persistent = alloc_solver_persistent(shared, persistent_local)
     dxdt = alloc_dxdt(shared, persistent_local)
     ```
  
  3. Update solver calls to match production signature with logging

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: generic_dirk.py - SEQUENTIAL

**Status**: [ ]
**Dependencies**: None

**Required Context**:
- Production: `src/cubie/integrators/algorithms/generic_dirk.py` (entire file - 842 lines)
- Instrumented: `tests/integrators/algorithms/instrumented/generic_dirk.py` (entire file - 856 lines)

**Input Validation Required**: None - these are verbatim copies

**Tasks**:

### 3.1 Update Imports
- File: `tests/integrators/algorithms/instrumented/generic_dirk.py`
- Action: Verify/Modify
- Details: Ensure imports match production + instrumented solver imports:
  ```python
  from cubie.buffer_registry import buffer_registry  # Already present
  ```

### 3.2 Update `DIRKStepConfig` Class
- File: `tests/integrators/algorithms/instrumented/generic_dirk.py`
- Action: Modify
- Details: Add missing `stage_rhs_location` field and validators (production lines 105-127):
  ```python
  @attrs.define
  class DIRKStepConfig(ImplicitStepConfig):
      """Configuration describing the DIRK integrator."""

      tableau: DIRKTableau = attrs.field(
          default=DEFAULT_DIRK_TABLEAU,
      )
      stage_increment_location: str = attrs.field(
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])  # ADD validator
      )
      stage_base_location: str = attrs.field(
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])  # ADD validator
      )
      accumulator_location: str = attrs.field(
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])  # ADD validator
      )
      stage_rhs_location: str = attrs.field(  # ADD entire field
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])
      )
  ```

### 3.3 Update `__init__` Method Signature
- File: `tests/integrators/algorithms/instrumented/generic_dirk.py`
- Action: Modify
- Details: 
  1. Add `stage_rhs_location` parameter
  2. Change solver params from required to Optional with None defaults (match production):
  ```python
  def __init__(
      self,
      precision: PrecisionDType,
      n: int,
      dxdt_function: Optional[Callable] = None,
      observables_function: Optional[Callable] = None,
      driver_function: Optional[Callable] = None,
      get_solver_helper_fn: Optional[Callable] = None,
      preconditioner_order: Optional[int] = None,  # CHANGE from int = 2
      krylov_tolerance: Optional[float] = None,  # CHANGE from float = 1e-6
      max_linear_iters: Optional[int] = None,  # CHANGE from int = 200
      linear_correction_type: Optional[str] = None,  # CHANGE from str = "minimal_residual"
      newton_tolerance: Optional[float] = None,  # CHANGE from float = 1e-6
      max_newton_iters: Optional[int] = None,  # CHANGE from int = 100
      newton_damping: Optional[float] = None,  # CHANGE from float = 0.5
      newton_max_backtracks: Optional[int] = None,  # CHANGE from int = 8
      tableau: DIRKTableau = DEFAULT_DIRK_TABLEAU,
      n_drivers: int = 0,
      stage_increment_location: Optional[str] = None,
      stage_base_location: Optional[str] = None,
      accumulator_location: Optional[str] = None,
      stage_rhs_location: Optional[str] = None,  # ADD THIS
  ) -> None:
  ```

### 3.4 Update `__init__` Method Logic
- File: `tests/integrators/algorithms/instrumented/generic_dirk.py`
- Action: Modify
- Details: Match production pattern (lines 232-287):
  1. Remove solver kwargs from config_kwargs (production doesn't include them)
  2. Add stage_rhs_location handling
  3. Remove inline buffer registration (move to `register_buffers()`)
  4. Build solver_kwargs conditionally
  5. Call `self.register_buffers()` after parent init

### 3.5 Add/Update `register_buffers()` Method
- File: `tests/integrators/algorithms/instrumented/generic_dirk.py`
- Action: Add (MISSING - buffer registration is in __init__ currently)
- Details: Copy VERBATIM from production (lines 289-344):
  ```python
  def register_buffers(self) -> None:
      """Register buffers according to locations in compile settings."""
      config = self.compile_settings
      precision = config.precision
      n = config.n
      tableau = config.tableau

      # Clear any existing buffer registrations
      buffer_registry.clear_parent(self)

      # Calculate buffer sizes
      accumulator_length = max(tableau.stage_count - 1, 0) * n

      # Register solver scratch and solver persistent buffers
      _ = buffer_registry.get_child_allocators(
              self,
              self.solver,
              name='solver'
      )

      # Register buffers
      buffer_registry.register(
          'stage_increment',
          self,
          n,
          config.stage_increment_location,
          persistent=True,
          precision=precision
      )
      buffer_registry.register(
          'accumulator',
          self,
          accumulator_length,
          config.accumulator_location,
          precision=precision
      )
      buffer_registry.register(
          'stage_base',
          self,
          n,
          config.stage_base_location,
          aliases='accumulator',
          precision=precision
      )
      buffer_registry.register(
          'stage_rhs',
          self,
          n,
          config.stage_rhs_location,
          persistent=True,
          precision=precision
      )
  ```

### 3.6 Update `build_implicit_helpers()` Method
- File: `tests/integrators/algorithms/instrumented/generic_dirk.py`
- Action: Modify
- Details: Match production structure but use instrumented solvers. Current instrumented creates LinearSolverConfig/NewtonKrylovConfig manually - keep that pattern but ensure signature matches what production expects.

### 3.7 Update `build_step()` Method Signature
- File: `tests/integrators/algorithms/instrumented/generic_dirk.py`
- Action: Modify
- Details: Add `solver_function` parameter (production has it):
  ```python
  def build_step(
      self,
      dxdt_fn: Callable,
      observables_function: Callable,
      driver_function: Optional[Callable],
      solver_function: Callable,  # ADD THIS
      numba_precision: type,
      n: int,
      n_drivers: int,
  ) -> StepCache:
  ```

### 3.8 Update `build_step()` Buffer Allocation
- File: `tests/integrators/algorithms/instrumented/generic_dirk.py`
- Action: Modify
- Details: Use production buffer names and allocators:
  ```python
  # Get child allocators for Newton solver (production pattern)
  alloc_solver_shared, alloc_solver_persistent = (
      buffer_registry.get_child_allocators(self, self._newton_solver,
                                           name='solver')
  )

  # Get allocators from buffer registry
  getalloc = buffer_registry.get_allocator
  alloc_stage_increment = getalloc('stage_increment', self)  # NOT 'dirk_stage_increment'
  alloc_accumulator = getalloc('accumulator', self)  # NOT 'dirk_accumulator'
  alloc_stage_base = getalloc('stage_base', self)  # NOT 'dirk_stage_base'
  alloc_stage_rhs = getalloc('stage_rhs', self)  # ADD THIS
  ```

### 3.9 Update Step Function Body
- File: `tests/integrators/algorithms/instrumented/generic_dirk.py`
- Action: Modify
- Details: Match production step function body exactly, preserving logging additions:
  1. Use production allocator calls
  2. Remove FSAL cache-related code if not in production
  3. Add `alloc_stage_rhs` usage
  4. Keep logging code blocks

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: generic_erk.py - SEQUENTIAL

**Status**: [ ]
**Dependencies**: None

**Required Context**:
- Production: `src/cubie/integrators/algorithms/generic_erk.py` (entire file - 595 lines)
- Instrumented: `tests/integrators/algorithms/instrumented/generic_erk.py` (entire file - 566 lines)

**Input Validation Required**: None - these are verbatim copies

**Tasks**:

### 4.1 Update Imports
- File: `tests/integrators/algorithms/instrumented/generic_erk.py`
- Action: Verify
- Details: Imports should match - already has `buffer_registry`

### 4.2 Update `ERKStepConfig` Class
- File: `tests/integrators/algorithms/instrumented/generic_erk.py`
- Action: Modify
- Details: Add validators to fields (production lines 106-118):
  ```python
  @attrs.define
  class ERKStepConfig(ExplicitStepConfig):
      """Configuration describing an explicit Runge--Kutta integrator."""

      tableau: ERKTableau = attrs.field(default=DEFAULT_ERK_TABLEAU)
      stage_rhs_location: str = attrs.field(
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])  # ADD validator
      )
      stage_accumulator_location: str = attrs.field(
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])  # ADD validator
      )
  ```

### 4.3 Update `__init__` Method
- File: `tests/integrators/algorithms/instrumented/generic_erk.py`
- Action: Modify
- Details: Match production pattern (lines 129-241):
  1. Remove inline buffer registration
  2. Call `self.register_buffers()` after parent init

  Current instrumented does buffer registration in `__init__` - move to `register_buffers()`:
  ```python
  # END of __init__, after super().__init__():
  self.register_buffers()
  ```

### 4.4 Add/Update `register_buffers()` Method
- File: `tests/integrators/algorithms/instrumented/generic_erk.py`
- Action: Add (MISSING - buffer registration is in __init__ currently)
- Details: Copy VERBATIM from production (lines 243-271):
  ```python
  def register_buffers(self) -> None:
      """Register buffers with buffer_registry."""
      config = self.compile_settings
      precision = config.precision
      n = config.n
      tableau = config.tableau

      # Clear any existing buffer registrations
      buffer_registry.clear_parent(self)

      # Calculate buffer sizes
      accumulator_length = max(tableau.stage_count - 1, 0) * n

      # Register algorithm buffers using config values
      buffer_registry.register(
          'stage_rhs',
          self,
          n,
          config.stage_rhs_location,
          persistent=True,
          precision=precision
      )
      buffer_registry.register(
          'stage_accumulator',
          self,
          accumulator_length,
          config.stage_accumulator_location,
          precision=precision
      )
  ```

### 4.5 Update `build_step()` Buffer Allocation
- File: `tests/integrators/algorithms/instrumented/generic_erk.py`
- Action: Modify
- Details: Use production buffer names:
  ```python
  # Get allocators from buffer registry
  getalloc = buffer_registry.get_allocator
  alloc_stage_rhs = getalloc('stage_rhs', self)  # NOT 'erk_stage_rhs'
  alloc_stage_accumulator = getalloc('stage_accumulator', self)  # NOT 'erk_stage_accumulator'
  ```
  Remove `alloc_stage_cache` since production doesn't have it in same form.

### 4.6 Update Step Function Body
- File: `tests/integrators/algorithms/instrumented/generic_erk.py`
- Action: Modify
- Details: Match production step function structure with logging additions. Production removes `stage_cache` allocation in step function.

### 4.7 Add Missing Properties
- File: `tests/integrators/algorithms/instrumented/generic_erk.py`
- Action: Verify
- Details: Properties should match production:
  - `shared_memory_required`, `local_scratch_required`, `persistent_local_required` using buffer_registry

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: generic_firk.py - SEQUENTIAL

**Status**: [ ]
**Dependencies**: None

**Required Context**:
- Production: `src/cubie/integrators/algorithms/generic_firk.py` (entire file - 654 lines)
- Instrumented: `tests/integrators/algorithms/instrumented/generic_firk.py` (entire file - 604 lines)

**Input Validation Required**: None - these are verbatim copies

**Tasks**:

### 5.1 Update Imports
- File: `tests/integrators/algorithms/instrumented/generic_firk.py`
- Action: Modify
- Details: Remove unused import, verify others:
  ```python
  # REMOVE unused:
  from cubie.cuda_simsafe import compile_kwargs  # Remove if not needed in step()
  ```

### 5.2 Update `FIRKStepConfig` Class
- File: `tests/integrators/algorithms/instrumented/generic_firk.py`
- Action: Modify
- Details: Add validators to fields (production lines 105-136):
  ```python
  @attrs.define
  class FIRKStepConfig(ImplicitStepConfig):
      """Configuration describing the FIRK integrator."""

      tableau: FIRKTableau = attrs.field(
          default=DEFAULT_FIRK_TABLEAU,
      )
      stage_increment_location: str = attrs.field(
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])  # ADD validator
      )
      stage_driver_stack_location: str = attrs.field(
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])  # ADD validator
      )
      stage_state_location: str = attrs.field(
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])  # ADD validator
      )
  ```

### 5.3 Update `__init__` Method Signature
- File: `tests/integrators/algorithms/instrumented/generic_firk.py`
- Action: Modify
- Details: Change solver params to Optional (match production):
  ```python
  preconditioner_order: Optional[int] = None,  # CHANGE from int = 2
  krylov_tolerance: Optional[float] = None,  # CHANGE from float = 1e-6
  # ... etc for all solver params
  ```

### 5.4 Update `__init__` Method Logic
- File: `tests/integrators/algorithms/instrumented/generic_firk.py`
- Action: Modify
- Details: Match production pattern (lines 241-294):
  1. Remove solver kwargs from config_kwargs
  2. Remove inline buffer registration (move to `register_buffers()`)
  3. Build solver_kwargs conditionally
  4. Call `self.register_buffers()` after parent init

### 5.5 Add/Update `register_buffers()` Method
- File: `tests/integrators/algorithms/instrumented/generic_firk.py`
- Action: Add (MISSING - buffer registration is in __init__ currently)
- Details: Copy VERBATIM from production (lines 296-326):
  ```python
  def register_buffers(self) -> None:
      """Register buffers according to locations in compile settings."""
      config = self.compile_settings
      precision = config.precision
      n = config.n
      tableau = config.tableau
      
      # Clear any existing buffer registrations
      buffer_registry.clear_parent(self)

      # Calculate buffer sizes
      all_stages_n = tableau.stage_count * n
      stage_driver_stack_elements = tableau.stage_count * config.n_drivers

      # Register algorithm buffers using config values
      buffer_registry.register(
          'stage_increment',
          self,
          all_stages_n,
          config.stage_increment_location,
          persistent=True,
          precision=precision
      )
      buffer_registry.register(
          'stage_driver_stack', self, stage_driver_stack_elements,
          config.stage_driver_stack_location, precision=precision
      )
      buffer_registry.register(
          'stage_state', self, n, config.stage_state_location,
          precision=precision
      )
  ```

### 5.6 Update `build_implicit_helpers()` Method
- File: `tests/integrators/algorithms/instrumented/generic_firk.py`
- Action: Modify
- Details: Match production structure, use `all_stages_n` dimension for solver

### 5.7 Update `build_step()` Method Signature
- File: `tests/integrators/algorithms/instrumented/generic_firk.py`
- Action: Modify
- Details: Add `solver_function` parameter:
  ```python
  def build_step(
      self,
      dxdt_fn: Callable,
      observables_function: Callable,
      driver_function: Optional[Callable],
      solver_function: Callable,  # ADD THIS
      numba_precision: type,
      n: int,
      n_drivers: int,  # REMOVE default = 0
  ) -> StepCache:
  ```

### 5.8 Update `build_step()` Buffer Allocation
- File: `tests/integrators/algorithms/instrumented/generic_firk.py`
- Action: Modify
- Details: Use production buffer names and allocators:
  ```python
  # Get allocators from buffer registry
  getalloc = buffer_registry.get_allocator
  alloc_stage_increment = getalloc('stage_increment', self)  # NOT 'firk_stage_increment'
  alloc_stage_driver_stack = getalloc('stage_driver_stack', self)  # NOT 'firk_stage_driver_stack'
  alloc_stage_state = getalloc('stage_state', self)  # NOT 'firk_stage_state'
  
  # Get child allocators for Newton solver
  alloc_solver_shared, alloc_solver_persistent = (
      buffer_registry.get_child_allocators(self, nonlinear_solver)
  )
  ```

### 5.9 Update Step Function Body
- File: `tests/integrators/algorithms/instrumented/generic_firk.py`
- Action: Modify
- Details: Remove `**compile_kwargs` from `@cuda.jit` decorator, match production step body with logging

### 5.10 Add Missing Properties
- File: `tests/integrators/algorithms/instrumented/generic_firk.py`
- Action: Add
- Details: Add `persistent_local_required` property from production (line 627-629):
  ```python
  @property
  def persistent_local_required(self) -> int:
      """Return the number of persistent local entries required."""
      return buffer_registry.persistent_local_buffer_size(self)
  ```

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 6: generic_rosenbrock_w.py - SEQUENTIAL

**Status**: [ ]
**Dependencies**: None

**Required Context**:
- Production: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py` (entire file - 829 lines)
- Instrumented: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py` (entire file - 835 lines)

**Input Validation Required**: None - these are verbatim copies

**Tasks**:

### 6.1 Update Imports
- File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
- Action: Modify
- Details: Add missing import:
  ```python
  from cubie._utils import PrecisionDType, is_device_validator  # ADD is_device_validator
  ```

### 6.2 Update `RosenbrockWStepConfig` Class
- File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
- Action: Modify
- Details: Match production exactly (lines 112-140):
  1. Rename `time_derivative_fn` to `time_derivative_function`
  2. Add `prepare_jacobian_function` field
  3. Add validators to all device function fields:
  ```python
  @attrs.define
  class RosenbrockWStepConfig(ImplicitStepConfig):
      """Configuration describing the Rosenbrock-W integrator."""

      tableau: RosenbrockTableau = attrs.field(default=DEFAULT_ROSENBROCK_TABLEAU)
      time_derivative_function: Optional[Callable] = attrs.field(
              default=None,
              validator=attrs.validators.optional(is_device_validator)
      )
      prepare_jacobian_function: Optional[Callable] = attrs.field(
              default=None,
              validator=attrs.validators.optional(is_device_validator)
      )
      driver_del_t: Optional[Callable] = attrs.field(
              default=None,
              validator=attrs.validators.optional(is_device_validator)
      )
      stage_rhs_location: str = attrs.field(
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])
      )
      stage_store_location: str = attrs.field(
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])
      )
      cached_auxiliaries_location: str = attrs.field(
          default='local',
          validator=attrs.validators.in_(['local', 'shared'])
      )
  ```

### 6.3 Update `__init__` Method Signature
- File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
- Action: Modify
- Details: Change solver params to Optional (match production):
  ```python
  preconditioner_order: Optional[int] = None,  # CHANGE from int = 2
  krylov_tolerance: Optional[float] = None,  # CHANGE from float = 1e-6
  # ... etc
  ```

### 6.4 Update `__init__` Method Logic
- File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
- Action: Modify
- Details: Match production pattern (lines 232-279):
  1. Remove solver kwargs from config_kwargs
  2. Remove inline buffer registration (move to `register_buffers()`)
  3. Build solver_kwargs conditionally
  4. Call `self.register_buffers()` after parent init
  5. Remove `self.solver.update(use_cached_auxiliaries=True)` line - handled differently in production

### 6.5 Add/Update `register_buffers()` Method
- File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
- Action: Add (MISSING - buffer registration is in __init__ currently)
- Details: Copy VERBATIM from production (lines 281-316):
  ```python
  def register_buffers(self) -> None:
      """Register buffers according to locations in compile settings."""
      config = self.compile_settings
      precision = config.precision
      n = config.n
      tableau = config.tableau

      # Clear any existing buffer registrations
      buffer_registry.clear_parent(self)

      # Calculate buffer sizes
      stage_store_elements = tableau.stage_count * n

      # Register algorithm buffers using config values
      buffer_registry.register(
          'stage_rhs', self, n, config.stage_rhs_location,
          precision=precision
      )
      buffer_registry.register(
          'stage_store', self, stage_store_elements,
          config.stage_store_location, precision=precision
      )
      # cached_auxiliaries registered with 0 size; updated in build_implicit_helpers
      buffer_registry.register(
          'cached_auxiliaries', self, 0,
          config.cached_auxiliaries_location, precision=precision
      )

      # Stage increment should persist between steps for initial guess
      buffer_registry.register(
          'stage_increment', self, n,
          config.stage_store_location,
          aliases='stage_store',
          persistent=True,
          precision=precision
      )
  ```

### 6.6 Update `build_implicit_helpers()` Method
- File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
- Action: Modify
- Details: Match production pattern (lines 318-378):
  1. Use `buffer_registry.update_buffer()` for cached_auxiliaries size
  2. Update compile_settings with solver_function, time_derivative_function, prepare_jacobian_function
  3. Keep InstrumentedLinearSolver usage

### 6.7 Remove `build()` Method Override
- File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
- Action: Modify
- Details: Production does NOT override `build()`. The current instrumented override should be removed - parent handles it.

### 6.8 Update `build_step()` Method Signature
- File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
- Action: Modify
- Details: Add `solver_function` parameter (production has it):
  ```python
  def build_step(
      self,
      dxdt_fn: Callable,
      observables_function: Callable,
      driver_function: Optional[Callable],
      solver_function: Callable,  # ADD THIS
      driver_del_t: Optional[Callable],
      numba_precision: type,
      n: int,
      n_drivers: int,
  ) -> StepCache:
  ```

### 6.9 Update `build_step()` Buffer Allocation
- File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
- Action: Modify
- Details: Use production buffer names:
  ```python
  # Get allocators from buffer registry
  getalloc = buffer_registry.get_allocator
  alloc_stage_rhs = getalloc('stage_rhs', self)  # NOT 'rosenbrock_stage_rhs'
  alloc_stage_store = getalloc('stage_store', self)  # NOT 'rosenbrock_stage_store'
  alloc_cached_auxiliaries = getalloc('cached_auxiliaries', self)  # NOT 'rosenbrock_cached_auxiliaries'
  alloc_stage_increment = getalloc('stage_increment', self)  # ADD THIS
  ```

### 6.10 Update Step Function Body
- File: `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
- Action: Modify
- Details: Match production step function structure with logging additions. Production accesses helpers from compile_settings:
  ```python
  linear_solver = solver_function
  prepare_jacobian = config.prepare_jacobian_function
  time_derivative_rhs = config.time_derivative_function
  ```

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

| Task Group | File | Key Changes |
|------------|------|-------------|
| 1 | backwards_euler.py | Add buffer_registry, update allocators, fix preconditioner_order handling |
| 2 | crank_nicolson.py | Add CrankNicolsonStepConfig, register_buffers(), dxdt_location param |
| 3 | generic_dirk.py | Add stage_rhs_location, validators, register_buffers(), fix buffer names |
| 4 | generic_erk.py | Add validators, register_buffers(), fix buffer names |
| 5 | generic_firk.py | Add validators, register_buffers(), fix buffer names, add solver_function param |
| 6 | generic_rosenbrock_w.py | Fix config fields, register_buffers(), remove build() override, fix buffer names |

**Total Task Groups**: 6
**Dependency Chain**: All groups are independent
**Parallel Execution**: All 6 groups can be executed in parallel (no dependencies between files)
**Estimated Complexity**: Medium - primarily mechanical verbatim copying with logging additions

---

## Logging Pattern Reference

All instrumented `step()` functions add these logging parameters:
```python
# After standard step parameters, before dt_scalar:
residuals,
jacobian_updates,
stage_states,
stage_derivatives,
stage_observables,
stage_drivers,  # or proposed_drivers_out
stage_increments,
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
```

Logging buffers use `cuda.local.array()` for local allocation where needed.
