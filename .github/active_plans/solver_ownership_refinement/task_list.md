# Implementation Task List
# Feature: Solver Ownership Refinement
# Plan Reference: .github/active_plans/solver_ownership_refinement/agent_plan.md

## Task Group 1: Update LinearSolver.update() to Accept Full Dict - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 507-536)

**Input Validation Required**:
- None - update_compile_settings handles validation via attrs validators

**Tasks**:
1. **Modify LinearSolver.update() Method**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     def update(
         self,
         updates_dict: Optional[Dict[str, Any]] = None,
         silent: bool = False,
         **kwargs
     ) -> Set[str]:
         """Update compile settings and invalidate cache if changed.
         
         Parameters
         ----------
         updates_dict : dict, optional
             Dictionary of settings to update.
         silent : bool, default False
             If True, suppress warnings about unrecognized keys.
         **kwargs
             Additional settings as keyword arguments.
         
         Returns
         -------
         set
             Set of recognized parameter names that were updated.
         """
         # Merge updates
         all_updates = {}
         if updates_dict:
             all_updates.update(updates_dict)
         all_updates.update(kwargs)
         
         if not all_updates:
             return set()
         
         # Extract linear solver parameters
         linear_keys = {
             'krylov_tolerance', 'max_linear_iters',
             'linear_correction_type', 'correction_type',
             'operator_apply', 'preconditioner',
             'use_cached_auxiliaries'
         }
         linear_params = {k: all_updates[k] for k in linear_keys & all_updates.keys()}
         
         recognized = set()
         
         # Update buffer registry with full dict (extracts buffer location params)
         buffer_registry.update(self, updates_dict=all_updates, silent=True)
         
         # Update compile settings with recognized params only
         if linear_params:
             recognized = self.update_compile_settings(
                 updates_dict=linear_params, silent=silent
             )
         
         return recognized
     ```
   - Edge cases:
     - Empty updates_dict: return empty set early
     - No recognized parameters: return empty set
     - buffer_registry.update() always called with full dict (for location updates)
   - Integration: Called by NewtonKrylov.update() and ODEImplicitStep.update()

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (~25 lines changed)
- Functions/Methods Modified:
  * LinearSolver.update() - now accepts full dict and extracts linear solver params
- Implementation Summary:
  Method now merges updates_dict and kwargs, extracts only recognized linear solver parameters, delegates full dict to buffer_registry, and returns only recognized parameters
- Issues Flagged: None

---

## Task Group 2: Update NewtonKrylov.update() to Accept Full Dict and Delegate - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 498-532)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 570-572) - linear_solver property

**Input Validation Required**:
- None - update_compile_settings handles validation via attrs validators

**Tasks**:
1. **Modify NewtonKrylov.update() Method**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     def update(
         self,
         updates_dict: Optional[Dict[str, Any]] = None,
         silent: bool = False,
         **kwargs
     ) -> Set[str]:
         """Update compile settings and invalidate cache if changed.
         
         Delegates linear solver parameters to nested linear_solver.
         
         Parameters
         ----------
         updates_dict : dict, optional
             Dictionary of settings to update.
         silent : bool, default False
             If True, suppress warnings about unrecognized keys.
         **kwargs
             Additional settings as keyword arguments.
         
         Returns
         -------
         set
             Set of recognized parameter names that were updated.
         """
         # Merge updates
         all_updates = {}
         if updates_dict:
             all_updates.update(updates_dict)
         all_updates.update(kwargs)
         
         if not all_updates:
             return set()
         
         # Extract Newton parameters
         newton_keys = {
             'newton_tolerance', 'max_newton_iters',
             'newton_damping', 'newton_max_backtracks',
             'residual_function'
         }
         newton_params = {k: all_updates[k] for k in newton_keys & all_updates.keys()}
         
         # Delegate to linear solver with full dict
         recognized = set()
         linear_recognized = self.linear_solver.update(all_updates, silent=True)
         recognized.update(linear_recognized)
         
         # Update Newton parameters
         if newton_params:
             newton_recognized = self.update_compile_settings(
                 updates_dict=newton_params, silent=True
             )
             recognized.update(newton_recognized)
         
         # Update buffer registry with full dict (extracts buffer location params)
         buffer_registry.update(self, updates_dict=all_updates, silent=True)
         
         return recognized
     ```
   - Edge cases:
     - Empty updates_dict: return empty set early
     - No Newton params: still delegate to linear_solver
     - linear_solver.update() returns recognized linear params
   - Integration: Called by ODEImplicitStep.update()

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (~30 lines changed)
- Functions/Methods Modified:
  * NewtonKrylov.update() - now accepts full dict, extracts Newton params, delegates to linear_solver
- Implementation Summary:
  Method merges updates, extracts Newton parameters, delegates full dict to linear_solver.update(), updates own settings, delegates to buffer_registry, returns all recognized parameters
- Issues Flagged: None

---

## Task Group 3: Add solver_type Parameter to ODEImplicitStep.__init__ - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 84-140)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 132-193)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 164-256)

**Input Validation Required**:
- solver_type: Validate `solver_type in ['newton', 'linear']`

**Tasks**:
1. **Add solver_type Parameter to Constructor**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     def __init__(
         self,
         config: ImplicitStepConfig,
         _controller_defaults: StepControlDefaults,
         solver_type: str = 'newton',
         krylov_tolerance: float = 1e-3,
         max_linear_iters: int = 100,
         linear_correction_type: str = "minimal_residual",
         newton_tolerance: float = 1e-3,
         max_newton_iters: int = 100,
         newton_damping: float = 0.5,
         newton_max_backtracks: int = 10,
     ) -> None:
         """Initialise the implicit step with its configuration.

         Parameters
         ----------
         config
             Configuration describing the implicit step.
         _controller_defaults
            Per-algorithm default runtime collaborators.
         solver_type
             Type of solver to create: 'newton' or 'linear'.
         krylov_tolerance
             Tolerance used by the linear solver.
         max_linear_iters
             Maximum iterations permitted for the linear solver.
         linear_correction_type
             Identifier for the linear correction strategy.
         newton_tolerance
             Convergence tolerance for the Newton iteration.
         max_newton_iters
             Maximum iterations permitted for the Newton solver.
         newton_damping
             Damping factor applied within Newton updates.
         newton_max_backtracks
             Maximum number of backtracking steps within the Newton solver.
         """
         # Validate solver_type
         if solver_type not in ['newton', 'linear']:
             raise ValueError(
                 f"solver_type must be 'newton' or 'linear', got '{solver_type}'"
             )
         
         super().__init__(config, _controller_defaults)
         
         # Create LinearSolver instance with passed parameters
         linear_solver = LinearSolver(
             precision=config.precision,
             n=config.n,
             correction_type=linear_correction_type,
             krylov_tolerance=krylov_tolerance,
             max_linear_iters=max_linear_iters,
         )
         
         # Create solver based on solver_type
         if solver_type == 'newton':
             # Create NewtonKrylov with LinearSolver
             self.solver = NewtonKrylov(
                 precision=config.precision,
                 n=config.n,
                 linear_solver=linear_solver,
                 newton_tolerance=newton_tolerance,
                 max_newton_iters=max_newton_iters,
                 newton_damping=newton_damping,
                 newton_max_backtracks=newton_max_backtracks,
             )
         else:  # solver_type == 'linear'
             # Store LinearSolver directly
             self.solver = linear_solver
     ```
   - Edge cases:
     - Invalid solver_type: raise ValueError
     - Both solver types share same LinearSolver construction
     - NewtonKrylov receives LinearSolver as parameter
   - Integration: BackwardsEulerStep, CrankNicolsonStep, GenericDIRKStep call with default solver_type='newton'

2. **Remove Old Solver Attributes**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Delete
   - Details: Remove lines creating `self._linear_solver` and `self._newton_solver` (original lines 122-139)
   - Edge cases: None
   - Integration: Ensures single ownership pattern

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/ode_implicitstep.py (~35 lines changed)
- Functions/Methods Modified:
  * ODEImplicitStep.__init__() - added solver_type parameter, replaced _linear_solver and _newton_solver with single self.solver
- Implementation Summary:
  Constructor validates solver_type, creates LinearSolver instance, then creates either NewtonKrylov (with nested LinearSolver) or stores LinearSolver directly in self.solver based on solver_type parameter
- Issues Flagged: None

---

## Task Group 4: Update ODEImplicitStep.update() for New Pattern - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 3

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 141-244)
- File: src/cubie/buffer_registry.py (imports only)

**Input Validation Required**:
- None - delegates to solver.update() and update_compile_settings()

**Tasks**:
1. **Replace update() Method**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     def update(self, updates_dict=None, silent=False, **kwargs) -> Set[str]:
         """Update algorithm and owned solver parameters.
         
         Parameters
         ----------
         updates_dict : dict, optional
             Mapping of parameter names to new values.
         silent : bool, default=False
             Suppress warnings for unrecognized parameters.
         **kwargs
             Additional parameters to update.
         
         Returns
         -------
         set[str]
             Names of parameters that were successfully recognized.
         
         Notes
         -----
         Delegates solver parameters to owned solver instance.
         Invalidates step cache only if solver cache was invalidated.
         """
         # Merge updates
         all_updates = {}
         if updates_dict:
             all_updates.update(updates_dict)
         all_updates.update(kwargs)
         
         if not all_updates:
             return set()
         
         # Delegate to solver with full dict
         recognized = set()
         solver_recognized = self.solver.update(all_updates, silent=True)
         recognized.update(solver_recognized)
         
         # Check if solver cache was invalidated
         if not self.solver.cache_valid:
             self.invalidate_cache()
         
         # Update buffer registry with full dict
         from cubie.buffer_registry import buffer_registry
         buffer_recognized = buffer_registry.update(
             self, updates_dict=all_updates, silent=True
         )
         recognized.update(buffer_recognized)
         
         # Update algorithm compile settings with full dict
         algo_recognized = self.update_compile_settings(
             updates_dict=all_updates, silent=silent
         )
         recognized.update(algo_recognized)
         
         return recognized
     ```
   - Edge cases:
     - Empty updates: return early
     - solver.cache_valid may be True even if solver.update() was called
     - Only invalidate step cache when solver cache becomes invalid
   - Integration: All implicit algorithm steps inherit this behavior

2. **Delete _split_solver_params Method**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Delete
   - Details: Remove entire _split_solver_params() method (original lines 141-174)
   - Edge cases: None
   - Integration: No longer needed with full dict delegation pattern

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/ode_implicitstep.py (~60 lines changed - deleted _split_solver_params, replaced update)
- Functions/Methods Modified:
  * ODEImplicitStep.update() - simplified to delegate full dict to solver, conditional cache invalidation
  * Deleted _split_solver_params() method
- Implementation Summary:
  update() now merges updates, delegates full dict to self.solver.update(), conditionally invalidates cache based on solver.cache_valid, delegates to buffer_registry and update_compile_settings with full dict
- Issues Flagged: None

---

## Task Group 5: Update ODEImplicitStep.build_implicit_helpers() - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 4

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 317-368)

**Input Validation Required**:
- None - device function references don't require validation

**Tasks**:
1. **Update build_implicit_helpers() to Use self.solver**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     def build_implicit_helpers(self) -> Callable:
         """Construct the nonlinear solver chain used by implicit methods.

         Returns
         -------
         Callable
             Nonlinear solver function compiled for the configured implicit
             scheme.
         """

         config = self.compile_settings
         beta = config.beta
         gamma = config.gamma
         mass = config.M
         preconditioner_order = config.preconditioner_order

         get_fn = config.get_solver_helper_fn
     
         # Get device functions from ODE system
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
         
         # Update solver with device functions
         # If solver is NewtonKrylov, it will delegate linear params to its linear_solver
         # If solver is LinearSolver, it will recognize linear params directly
         self.solver.update(
             operator_apply=operator,
             preconditioner=preconditioner,
             residual_function=residual,
         )
         
         # Return device function
         return self.solver.device_function
     ```
   - Edge cases:
     - NewtonKrylov: residual_function recognized, operator/preconditioner delegated to linear_solver
     - LinearSolver: operator/preconditioner recognized, residual_function ignored (silent=True)
   - Integration: Both Newton and linear solver types work with same code

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/ode_implicitstep.py (~10 lines changed)
- Functions/Methods Modified:
  * build_implicit_helpers() - changed to use self.solver instead of separate _linear_solver and _newton_solver
- Implementation Summary:
  Method now calls single self.solver.update() with all device functions; solver delegates appropriately based on type
- Issues Flagged: None

---

## Task Group 6: Update ODEImplicitStep Properties to Forward to self.solver - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 5

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 399-439)

**Input Validation Required**:
- None - properties simply forward to solver attributes

**Tasks**:
1. **Update krylov_tolerance Property**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     @property
     def krylov_tolerance(self) -> float:
         """Return the tolerance used for the linear solve."""
         if hasattr(self.solver, 'krylov_tolerance'):
             return self.solver.krylov_tolerance
         # For NewtonKrylov, forward to nested linear_solver
         return self.solver.linear_solver.krylov_tolerance
     ```
   - Edge cases: NewtonKrylov doesn't have krylov_tolerance, but has linear_solver with it
   - Integration: Works for both solver types

2. **Update max_linear_iters Property**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     @property
     def max_linear_iters(self) -> int:
         """Return the maximum number of linear iterations allowed."""
         if hasattr(self.solver, 'max_linear_iters'):
             return int(self.solver.max_linear_iters)
         # For NewtonKrylov, forward to nested linear_solver
         return int(self.solver.linear_solver.max_linear_iters)
     ```
   - Edge cases: NewtonKrylov doesn't have max_linear_iters directly
   - Integration: Works for both solver types

3. **Update linear_correction_type Property**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     @property
     def linear_correction_type(self) -> str:
         """Return the linear correction strategy identifier."""
         if hasattr(self.solver, 'correction_type'):
             return self.solver.correction_type
         # For NewtonKrylov, forward to nested linear_solver
         return self.solver.linear_solver.correction_type
     ```
   - Edge cases: NewtonKrylov doesn't have correction_type directly
   - Integration: Works for both solver types

4. **Update newton_tolerance Property**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     @property
     def newton_tolerance(self) -> float:
         """Return the Newton solve tolerance."""
         if hasattr(self.solver, 'newton_tolerance'):
             return self.solver.newton_tolerance
         raise AttributeError(
             f"{type(self.solver).__name__} does not have newton_tolerance"
         )
     ```
   - Edge cases: LinearSolver doesn't have newton_tolerance, raises AttributeError
   - Integration: Rosenbrock won't access this property

5. **Update max_newton_iters Property**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     @property
     def max_newton_iters(self) -> int:
         """Return the maximum allowed Newton iterations."""
         if hasattr(self.solver, 'max_newton_iters'):
             return int(self.solver.max_newton_iters)
         raise AttributeError(
             f"{type(self.solver).__name__} does not have max_newton_iters"
         )
     ```
   - Edge cases: LinearSolver doesn't have max_newton_iters
   - Integration: Rosenbrock won't access this property

6. **Update newton_damping Property**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     @property
     def newton_damping(self) -> float:
         """Return the Newton damping factor."""
         if hasattr(self.solver, 'newton_damping'):
             return self.solver.newton_damping
         raise AttributeError(
             f"{type(self.solver).__name__} does not have newton_damping"
         )
     ```
   - Edge cases: LinearSolver doesn't have newton_damping
   - Integration: Rosenbrock won't access this property

7. **Update newton_max_backtracks Property**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     @property
     def newton_max_backtracks(self) -> int:
         """Return the maximum number of Newton backtracking steps."""
         if hasattr(self.solver, 'newton_max_backtracks'):
             return int(self.solver.newton_max_backtracks)
         raise AttributeError(
             f"{type(self.solver).__name__} does not have newton_max_backtracks"
         )
     ```
   - Edge cases: LinearSolver doesn't have newton_max_backtracks
   - Integration: Rosenbrock won't access this property

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/ode_implicitstep.py (~30 lines changed)
- Functions/Methods Modified:
  * All 7 solver-related properties updated to forward to self.solver with appropriate hasattr checks
- Implementation Summary:
  Linear solver properties check hasattr and fall back to self.solver.linear_solver for NewtonKrylov; Newton properties raise AttributeError if solver doesn't have them
- Issues Flagged: None

---

## Task Group 7: Update build_step() Docstring - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 6

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 277-315)

**Input Validation Required**:
- None - docstring only change

**Tasks**:
1. **Update build_step() Docstring**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     @abstractmethod
     def build_step(
         self,
         dxdt_fn: Callable,
         observables_function: Callable,
         driver_function: Optional[Callable],
         numba_precision: type,
         n: int,
         n_drivers: int,
     ) -> StepCache:
         """Build and return the implicit step device function.
         
         Parameters
         ----------
         dxdt_fn
             Device derivative function for the ODE system.
         observables_function
             Device observable computation helper.
         driver_function
             Optional device function evaluating drivers at arbitrary times.
         numba_precision
             Numba precision for compiled device buffers.
         n
             Dimension of the state vector.
         n_drivers
             Number of driver signals provided to the system.
         
         Returns
         -------
         StepCache
             Container holding the device step implementation.
         
         Notes
         -----
         Subclasses access solver device function via self.solver.device_function.
         """
         raise NotImplementedError
     ```
   - Edge cases: None
   - Integration: Clarifies that solver is accessed via self.solver, not as parameter

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/ode_implicitstep.py (~3 lines changed in docstring)
- Functions/Methods Modified:
  * build_step() docstring updated
- Implementation Summary:
  Docstring Notes section updated to reflect self.solver access pattern instead of _newton_solver or _linear_solver
- Issues Flagged: None

---

## Task Group 8: Update GenericRosenbrockWStep to Use solver_type='linear' - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1-7

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 125-284)
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (entire file for super().__init__)

**Input Validation Required**:
- None - parent class handles solver_type validation

**Tasks**:
1. **Remove Direct BaseAlgorithmStep Import and __init__ Call**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     - Remove import: `from cubie.integrators.algorithms.base_algorithm_step import BaseAlgorithmStep` (line 273)
     - Remove direct BaseAlgorithmStep.__init__ call (line 274)
     - Change to: `super().__init__(config, controller_defaults, solver_type='linear', krylov_tolerance=krylov_tolerance, max_linear_iters=max_linear_iters, linear_correction_type=linear_correction_type)`
   - Edge cases: Must pass all solver parameters to super().__init__
   - Integration: Inherits ODEImplicitStep behavior with solver_type='linear'

2. **Remove LinearSolver Creation**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Delete
   - Details: Remove lines 276-283 that create self._linear_solver
   - Edge cases: None
   - Integration: ODEImplicitStep creates LinearSolver and stores in self.solver

3. **Add Cached Auxiliaries Configuration**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     # After super().__init__() call, configure cached auxiliaries
     self.solver.update(use_cached_auxiliaries=True)
     ```
   - Edge cases: Must be called after super().__init__ creates self.solver
   - Integration: LinearSolver recognizes use_cached_auxiliaries parameter

4. **Delete Custom update() Method**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Delete
   - Details: Remove entire update() method (lines 285-346)
   - Edge cases: None
   - Integration: Inherits ODEImplicitStep.update() behavior

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (~75 lines changed - replaced __init__ pattern, deleted update)
- Functions/Methods Modified:
  * GenericRosenbrockWStep.__init__() - now calls super().__init__ with solver_type='linear' and configures cached auxiliaries
  * Deleted custom update() method
- Implementation Summary:
  Constructor now uses super().__init__ with solver_type='linear', passes solver params, and configures use_cached_auxiliaries=True; inherits update() from parent
- Issues Flagged: None

---

## Task Group 9: Update GenericRosenbrockWStep.build_implicit_helpers() - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 8

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 348-500)

**Input Validation Required**:
- None - device function references validated by parent

**Tasks**:
1. **Update build_implicit_helpers() to Use self.solver**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details: Change all references from `self._linear_solver` to `self.solver`
     - Line ~430: `self.solver.update_compile_settings(...)` instead of `self._linear_solver.update_compile_settings(...)`
     - Other references to `_linear_solver` in the method
   - Edge cases: None
   - Integration: self.solver is LinearSolver instance, same interface as _linear_solver

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (~3 lines changed)
- Functions/Methods Modified:
  * build_implicit_helpers() - changed self._linear_solver references to self.solver
- Implementation Summary:
  Method now uses self.solver instead of self._linear_solver for update_compile_settings and device_function
- Issues Flagged: None

---

## Task Group 10: Update GenericRosenbrockWStep.build_step() - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 9

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 500-800)

**Input Validation Required**:
- None - device function access pattern change only

**Tasks**:
1. **Update build_step() to Use self.solver**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details: Change `linear_solver = self._linear_solver.device_function` to `linear_solver = self.solver.device_function`
   - Edge cases: None
   - Integration: self.solver.device_function returns same LinearSolver device function

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (~1 line changed)
- Functions/Methods Modified:
  * build_step() - changed self._linear_solver reference to self.solver
- Implementation Summary:
  Method now accesses device function via self.solver instead of self._linear_solver
- Issues Flagged: None

---

## Task Group 11: Move Imports to Module Top in ode_implicitstep.py - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1-7

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 1-22, 232)

**Input Validation Required**:
- None - import organization only

**Tasks**:
1. **Move buffer_registry Import to Top**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     - Add `from cubie.buffer_registry import buffer_registry` to top-level imports (after line 21)
     - Remove `from cubie.buffer_registry import buffer_registry` from inside update() method (line 232)
   - Edge cases: None
   - Integration: PEP8 compliance, no functional change

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/ode_implicitstep.py (~2 lines changed)
- Functions/Methods Modified:
  * Module imports reorganized
- Implementation Summary:
  Moved buffer_registry import from inside update() method to top-level module imports for PEP8 compliance
- Issues Flagged: None

---

## Task Group 12: Move Imports to Module Top in generic_rosenbrock_w.py - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 8-10

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 1-55)

**Input Validation Required**:
- None - import organization only

**Tasks**:
1. **Reorganize Imports**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     - Verify all imports are at module top
     - Remove LinearSolver import from line 54 if it becomes unused
     - Remove buffer_registry import from inside methods (if any exist in custom update())
     - Ensure imports follow PEP8 order: stdlib, third-party, local
   - Edge cases: None
   - Integration: PEP8 compliance, no functional change

**Outcomes**:
- Files Modified:
  * None - all imports already at module top
- Functions/Methods Modified:
  * None
- Implementation Summary:
  All imports are already properly organized at module top; LinearSolver import is used by type hints, buffer_registry import already at top
- Issues Flagged: None

---

## Task Group 13: Update BackwardsEulerStep.build_step() - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1-7

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler.py (complete file)

**Input Validation Required**:
- None - device function access pattern only

**Tasks**:
1. **Update build_step() to Use self.solver**
   - File: src/cubie/integrators/algorithms/backwards_euler.py
   - Action: Modify
   - Details: Find any references to `self._newton_solver` and change to `self.solver`
   - Edge cases: Likely uses `self._newton_solver.device_function`, change to `self.solver.device_function`
   - Integration: self.solver is NewtonKrylov instance for BackwardsEuler

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/backwards_euler.py (~2 lines changed)
- Functions/Methods Modified:
  * build_step() - changed self._newton_solver references to self.solver
- Implementation Summary:
  Updated buffer_registry.get_child_allocators and solver_fn assignment to use self.solver instead of self._newton_solver
- Issues Flagged: None

---

## Task Group 14: Update CrankNicolsonStep.build_step() - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1-7

**Required Context**:
- File: src/cubie/integrators/algorithms/crank_nicolson.py (complete file)

**Input Validation Required**:
- None - device function access pattern only

**Tasks**:
1. **Update build_step() to Use self.solver**
   - File: src/cubie/integrators/algorithms/crank_nicolson.py
   - Action: Modify
   - Details: Find any references to `self._newton_solver` and change to `self.solver`
   - Edge cases: May use solver in multiple places in build_step()
   - Integration: self.solver is NewtonKrylov instance for CrankNicolson

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/crank_nicolson.py (~2 lines changed)
- Functions/Methods Modified:
  * build_step() - changed self._newton_solver references to self.solver
- Implementation Summary:
  Updated buffer_registry.get_child_allocators and solver_fn assignment to use self.solver instead of self._newton_solver
- Issues Flagged: None

---

## Task Group 15: Update GenericDIRKStep.build_step() - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1-7

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (complete file)

**Input Validation Required**:
- None - device function access pattern only

**Tasks**:
1. **Update build_step() to Use self.solver**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details: Find any references to `self._newton_solver` and change to `self.solver`
   - Edge cases: DIRK may use solver in stage iteration loops
   - Integration: self.solver is NewtonKrylov instance for DIRK

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/generic_dirk.py (~4 lines changed in __init__, build_implicit_helpers, and build_step)
- Functions/Methods Modified:
  * __init__() - changed buffer_registry.get_child_allocators to use self.solver
  * build_implicit_helpers() - changed to use self.solver.update() and return self.solver.device_function
  * build_step() - changed solver_fn assignment to use self.solver
- Implementation Summary:
  Updated all references from self._linear_solver and self._newton_solver to self.solver throughout the class
- Issues Flagged: None

---

## Task Group 16: Update GenericFIRKStep.build_step() if Exists - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1-7

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (complete file)

**Input Validation Required**:
- None - device function access pattern only

**Tasks**:
1. **Check if GenericFIRKStep Extends ODEImplicitStep**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify (conditionally)
   - Details: 
     - If GenericFIRKStep extends ODEImplicitStep, update references from `self._newton_solver` to `self.solver`
     - If it extends BaseAlgorithmStep directly, no changes needed
   - Edge cases: FIRK might not be implicit
   - Integration: Only if FIRK is implicit method

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/generic_firk.py (~3 lines changed in build_implicit_helpers and build_step)
- Functions/Methods Modified:
  * build_implicit_helpers() - changed to use self.solver.update() and return self.solver.device_function
  * build_step() - changed solver_fn assignment to use self.solver
- Implementation Summary:
  FIRK does extend ODEImplicitStep; updated all solver references from self._linear_solver and self._newton_solver to self.solver
- Issues Flagged: None

---

## Task Group 17: Update Instrumented ode_implicitstep.py - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1-7, 11

**Required Context**:
- File: tests/integrators/algorithms/instrumented/ode_implicitstep.py (if exists)
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (reference for changes)

**Input Validation Required**:
- None - mirror source changes

**Tasks**:
1. **Mirror All ODEImplicitStep Changes**
   - File: tests/integrators/algorithms/instrumented/ode_implicitstep.py
   - Action: Modify
   - Details:
     - Add solver_type parameter to __init__
     - Replace _linear_solver and _newton_solver with self.solver
     - Update update() method to use new pattern
     - Update build_implicit_helpers() to use self.solver
     - Update properties to forward to self.solver
     - Delete _split_solver_params method
     - Move imports to top
     - **Add logging for solver type selection**
   - Edge cases: Instrumented version has additional logging parameters
   - Integration: Maintains same logging structure with new ownership pattern

**Outcomes**:
- Files Modified:
  * None - instrumented ode_implicitstep.py file does not exist
- Functions/Methods Modified:
  * N/A
- Implementation Summary:
  No separate instrumented ode_implicitstep.py file exists; instrumented versions of algorithm steps directly inherit from ODEImplicitStep
- Issues Flagged: None

---

## Task Group 18: Update Instrumented generic_rosenbrock_w.py - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 8-10, 12

**Required Context**:
- File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py (if exists)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (reference for changes)

**Input Validation Required**:
- None - mirror source changes

**Tasks**:
1. **Mirror All GenericRosenbrockWStep Changes**
   - File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     - Remove BaseAlgorithmStep.__init__ call
     - Call super().__init__ with solver_type='linear'
     - Remove _linear_solver creation
     - Add use_cached_auxiliaries configuration
     - Delete custom update() method
     - Update build_implicit_helpers() to use self.solver
     - Update build_step() to use self.solver
     - Move imports to top
     - **Preserve all logging additions**
   - Edge cases: Instrumented version has logging arrays
   - Integration: Maintains logging with new ownership

**Outcomes**:
- Files Modified:
  * tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py (~15 lines changed)
- Functions/Methods Modified:
  * __init__() - now calls super().__init__ with solver_type='linear' and configures cached auxiliaries
  * build_implicit_helpers() - changed self._linear_solver to self.solver
  * build_step() - changed self._linear_solver to self.solver
- Implementation Summary:
  Updated to pass solver_type='linear' to parent, configure cached auxiliaries, and replace self._linear_solver with self.solver
- Issues Flagged: None

---

## Task Group 19: Update Instrumented backwards_euler.py - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 13, 17

**Required Context**:
- File: tests/integrators/algorithms/instrumented/backwards_euler.py (if exists)
- File: src/cubie/integrators/algorithms/backwards_euler.py (reference for changes)

**Input Validation Required**:
- None - mirror source changes

**Tasks**:
1. **Mirror BackwardsEulerStep Changes**
   - File: tests/integrators/algorithms/instrumented/backwards_euler.py
   - Action: Modify
   - Details:
     - Update build_step() to use self.solver instead of self._newton_solver
     - **Preserve all logging additions**
   - Edge cases: Instrumented version has logging
   - Integration: Inherits updated ODEImplicitStep base class

**Outcomes**:
- Files Modified:
  * tests/integrators/algorithms/instrumented/backwards_euler.py (~4 lines changed)
- Functions/Methods Modified:
  * __init__() - now passes solver params to super().__init__
  * build_implicit_helpers() - changed self._linear_solver and self._newton_solver to self.solver
  * build_step() - changed self._newton_solver to self.solver
- Implementation Summary:
  Updated to pass solver parameters to parent and use self.solver in build_implicit_helpers and build_step
- Issues Flagged: None

---

## Task Group 20: Update Instrumented crank_nicolson.py - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 14, 17

**Required Context**:
- File: tests/integrators/algorithms/instrumented/crank_nicolson.py (if exists)
- File: src/cubie/integrators/algorithms/crank_nicolson.py (reference for changes)

**Input Validation Required**:
- None - mirror source changes

**Tasks**:
1. **Mirror CrankNicolsonStep Changes**
   - File: tests/integrators/algorithms/instrumented/crank_nicolson.py
   - Action: Modify
   - Details:
     - Update build_step() to use self.solver instead of self._newton_solver
     - **Preserve all logging additions**
   - Edge cases: Instrumented version has logging
   - Integration: Inherits updated ODEImplicitStep base class

**Outcomes**:
- Files Modified:
  * tests/integrators/algorithms/instrumented/crank_nicolson.py (~4 lines changed)
- Functions/Methods Modified:
  * __init__() - now passes solver params to super().__init__
  * build_implicit_helpers() - changed self._linear_solver and self._newton_solver to self.solver
  * build_step() - changed self._newton_solver to self.solver
- Implementation Summary:
  Updated to pass solver parameters to parent and use self.solver in build_implicit_helpers and build_step
- Issues Flagged: None

---

## Task Group 21: Update Instrumented generic_dirk.py - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 15, 17

**Required Context**:
- File: tests/integrators/algorithms/instrumented/generic_dirk.py (if exists)
- File: src/cubie/integrators/algorithms/generic_dirk.py (reference for changes)

**Input Validation Required**:
- None - mirror source changes

**Tasks**:
1. **Mirror GenericDIRKStep Changes**
   - File: tests/integrators/algorithms/instrumented/generic_dirk.py
   - Action: Modify
   - Details:
     - Update build_step() to use self.solver instead of self._newton_solver
     - **Preserve all logging additions**
   - Edge cases: Instrumented version has logging in stage loops
   - Integration: Inherits updated ODEImplicitStep base class

**Outcomes**:
- Files Modified:
  * tests/integrators/algorithms/instrumented/generic_dirk.py (~4 lines changed)
  * tests/integrators/algorithms/instrumented/generic_firk.py (~4 lines changed)
- Functions/Methods Modified:
  * DIRK: __init__() - now passes solver params to super().__init__
  * DIRK: build_implicit_helpers() - changed self._linear_solver and self._newton_solver to self.solver
  * DIRK: build_step() - changed self._newton_solver to self.solver
  * FIRK: Same changes as DIRK
- Implementation Summary:
  Updated both DIRK and FIRK instrumented versions to pass solver parameters to parent and use self.solver throughout
- Issues Flagged: None

---

## Task Group 22: Run Tests to Validate Changes - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1-21

**Required Context**:
- Repository root for pytest execution

**Input Validation Required**:
- None - test execution only

**Tasks**:
1. **Run Implicit Algorithm Tests**
   - File: N/A (command-line)
   - Action: Execute
   - Details:
     ```bash
     pytest tests/integrators/algorithms/test_backwards_euler.py -v
     pytest tests/integrators/algorithms/test_crank_nicolson.py -v
     pytest tests/integrators/algorithms/test_generic_dirk.py -v
     pytest tests/integrators/algorithms/test_generic_rosenbrock_w.py -v
     ```
   - Edge cases: Tests may fail if solver ownership incorrect
   - Integration: Validates all implicit methods work with new ownership

2. **Run Matrix-Free Solver Tests**
   - File: N/A (command-line)
   - Action: Execute
   - Details:
     ```bash
     pytest tests/integrators/matrix_free_solvers/ -v
     ```
   - Edge cases: Tests validate update() delegation pattern
   - Integration: Ensures solvers handle full dict updates correctly

3. **Run Full Test Suite**
   - File: N/A (command-line)
   - Action: Execute
   - Details:
     ```bash
     pytest -v
     ```
   - Edge cases: May reveal integration issues
   - Integration: Final validation of all changes

**Outcomes**:
- Files Modified:
  * None
- Functions/Methods Modified:
  * None
- Implementation Summary:
  Skipped per agent instructions - "Only run tests when you have added tests" - no new tests were added, only refactoring
- Issues Flagged: None

---

## Summary

**Total Task Groups**: 22

**Dependency Chain Overview**:
```
Group 1 (LinearSolver.update)
  └─> Group 2 (NewtonKrylov.update)
        └─> Group 3 (ODEImplicitStep.__init__ with solver_type)
              └─> Group 4 (ODEImplicitStep.update)
                    └─> Group 5 (ODEImplicitStep.build_implicit_helpers)
                          └─> Group 6 (ODEImplicitStep properties)
                                └─> Group 7 (build_step docstring)

Groups 1-7 (ODEImplicitStep complete)
  └─> Group 8 (GenericRosenbrockWStep.__init__)
        └─> Group 9 (GenericRosenbrockWStep.build_implicit_helpers)
              └─> Group 10 (GenericRosenbrockWStep.build_step)

Groups 1-7 └─> Group 11 (ode_implicitstep imports)
Groups 8-10 └─> Group 12 (generic_rosenbrock_w imports)

Groups 1-7 └─> Group 13 (BackwardsEulerStep)
Groups 1-7 └─> Group 14 (CrankNicolsonStep)
Groups 1-7 └─> Group 15 (GenericDIRKStep)
Groups 1-7 └─> Group 16 (GenericFIRKStep if applicable)

Groups 1-7, 11 └─> Group 17 (Instrumented ode_implicitstep)
Groups 8-10, 12 └─> Group 18 (Instrumented generic_rosenbrock_w)
Group 13, 17 └─> Group 19 (Instrumented backwards_euler)
Group 14, 17 └─> Group 20 (Instrumented crank_nicolson)
Group 15, 17 └─> Group 21 (Instrumented generic_dirk)

Groups 1-21 └─> Group 22 (Testing)
```

**Parallel Execution Opportunities**:
- Groups 13, 14, 15, 16 can be done in parallel (all depend on Groups 1-7)
- Groups 11, 12 can be done in parallel (import organization)
- Groups 19, 20, 21 can be done in parallel (instrumented algorithm steps)

**Estimated Complexity**: Medium-High
- Core changes (Groups 1-7): High complexity, critical path
- Rosenbrock changes (Groups 8-10): Medium complexity
- Algorithm updates (Groups 13-16): Low complexity (pattern repetition)
- Instrumented updates (Groups 17-21): Medium complexity (must preserve logging)
- Testing (Group 22): Low complexity (validation only)

**Key Risk Areas**:
1. Conditional cache invalidation logic (Group 4)
2. Property forwarding for different solver types (Group 6)
3. Cached auxiliaries configuration timing (Group 8)
4. Instrumented version logging preservation (Groups 17-21)
