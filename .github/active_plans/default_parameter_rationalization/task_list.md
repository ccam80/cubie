# Implementation Task List
# Feature: Default Parameter Rationalization
# Plan Reference: .github/active_plans/default_parameter_rationalization/agent_plan.md

## Overview

This task list implements **Option B: Helper Function with Required/Optional Split** to eliminate verbose `if param is not None:` patterns throughout cubie's configuration cascade. The implementation follows the plan's 8-phase approach.

### Key Design Decisions

1. **Single Source of Truth**: All optional parameter defaults remain in attrs config classes
2. **Helper Function**: `build_config()` merges defaults, required params, and optional overrides
3. **None Filtering**: `None` in kwargs means "use default from config class"
4. **Required Params**: Stay in function signatures for clarity and IDE support
5. **Optional Params**: Captured via `**kwargs` and passed to `build_config()`

---

## Task Group 1: Add `build_config` Helper Function - [SEQUENTIAL]
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/_utils.py (entire file - add function at end)
- Reference: attrs library `fields()` function, `attrs.NOTHING` sentinel

**Input Validation Required**:
- config_class: Must be an attrs class (use `attrs.has()` to verify)
- required: Must be a dict
- Verify all required fields of config_class are present in merged result
- Filter out keys that aren't valid attrs fields (ignore extra kwargs)

**Tasks**:
1. **Add `build_config` function to `_utils.py`**
   - File: src/cubie/_utils.py
   - Action: Modify (append function)
   - Details:
     ```python
     def build_config(
         config_class: type,
         required: dict,
         **optional
     ) -> Any:
         """Build attrs config instance from required and optional parameters.
         
         Starts with config class defaults for all optional fields, then applies
         provided required and optional values. This eliminates the verbose pattern
         of checking `if param is not None` for every optional parameter.
         
         Parameters
         ----------
         config_class : type
             Attrs class to instantiate (e.g., DIRKStepConfig).
         required : dict
             Required parameters that must be provided. These are typically
             function parameters like precision, n, dxdt_function.
         **optional
             Optional parameter overrides. Only non-None values override defaults
             from the config class.
         
         Returns
         -------
         config_class instance
             Configured attrs object with defaults + required + optional overrides.
         
         Raises
         ------
         TypeError
             If config_class is not an attrs class.
         ValueError
             If required fields of config_class are missing after merge.
         
         Examples
         --------
         >>> config = build_config(
         ...     DIRKStepConfig,
         ...     required={'precision': np.float32, 'n': 3},
         ...     krylov_tolerance=1e-8
         ... )
         
         Notes
         -----
         The helper automatically:
         - Extracts defaults from config_class attrs fields
         - Merges: defaults <- required <- optional (non-None)
         - Filters out None values from optional kwargs
         - Validates all required config fields are present
         - Ignores extra keys not in config class fields
         """
         import attrs
         
         if not attrs.has(config_class):
             raise TypeError(
                 f"{config_class.__name__} is not an attrs class"
             )
         
         # Extract field info from config class
         defaults = {}
         required_fields = set()
         valid_fields = set()
         
         for field in attrs.fields(config_class):
             valid_fields.add(field.name)
             if field.default is not attrs.NOTHING:
                 # Has default value - extract it
                 if isinstance(field.default, attrs.Factory):
                     defaults[field.name] = field.default.factory()
                 else:
                     defaults[field.name] = field.default
             else:
                 # No default - this is a required field
                 required_fields.add(field.name)
         
         # Filter optional kwargs to remove None values
         # (None means "use default", not "set to None")
         filtered_optional = {
             k: v for k, v in optional.items() if v is not None
         }
         
         # Merge: defaults <- required <- filtered_optional
         merged = {**defaults, **required, **filtered_optional}
         
         # Validate all required config fields are present
         missing = required_fields - set(merged.keys())
         if missing:
             raise ValueError(
                 f"{config_class.__name__} missing required fields: {missing}"
             )
         
         # Filter to only valid fields (ignore extra keys)
         final = {k: v for k, v in merged.items() if k in valid_fields}
         
         return config_class(**final)
     ```
   - Edge cases:
     - Handle attrs.Factory defaults (e.g., `factory=dict`)
     - Handle inheritance (fields from parent classes)
     - Ignore underscore-prefixed field aliases (attrs handles internally)
   - Integration: Import `attrs` at module level (already done) or locally in function

**Outcomes**: 
[x] `build_config` function added to `_utils.py`
[x] Function handles attrs.Factory defaults correctly
[x] Function validates required fields
[x] Function ignores extra kwargs not in config class

**Implementation Summary**:
- Files Modified:
  * src/cubie/_utils.py (97 lines changed)
- Functions/Methods Added:
  * build_config() in _utils.py
- Implementation Summary:
  Added build_config helper function that:
  - Validates config_class is an attrs class using attrs.has()
  - Extracts defaults from attrs fields, handling attrs.Factory
  - Filters None values from optional kwargs
  - Merges defaults <- required <- filtered_optional
  - Validates all required fields are present
  - Filters to only valid fields for the config class
  - Returns instantiated config object
- Issues Flagged: None

---

## Task Group 2: Refactor BackwardsEulerStep - [SEQUENTIAL]
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler.py (entire file)
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 83-206)
- File: src/cubie/_utils.py (build_config function)

**Input Validation Required**:
- precision: Validated by config class converter/validator
- n: Validated by config class validator (getype_validator)
- All other params: Validated by respective config class validators

**Tasks**:
1. **Update BackwardsEulerStep.__init__ to use build_config**
   - File: src/cubie/integrators/algorithms/backwards_euler.py
   - Action: Modify
   - Details:
     Replace verbose pattern (lines 42-180) with:
     ```python
     def __init__(
         self,
         precision: PrecisionDType,
         n: int,
         dxdt_function: Optional[Callable] = None,
         observables_function: Optional[Callable] = None,
         driver_function: Optional[Callable] = None,
         get_solver_helper_fn: Optional[Callable] = None,
         **kwargs,
     ) -> None:
         """Initialise the backward Euler step configuration.

         Parameters
         ----------
         precision
             Precision applied to device buffers.
         n
             Number of state entries advanced per step.
         dxdt_function
             Device derivative function evaluating ``dx/dt``.
         observables_function
             Device function computing system observables.
         driver_function
             Optional device function evaluating drivers at arbitrary times.
         get_solver_helper_fn
             Callable returning device helpers used by the nonlinear solver.
         **kwargs
             Optional parameters passed to config classes. See
             BackwardsEulerStepConfig, ImplicitStepConfig, and solver config
             classes for available parameters. None values are ignored.
         """
         from cubie._utils import build_config
         
         beta = ALGO_CONSTANTS['beta']
         gamma = ALGO_CONSTANTS['gamma']
         M = ALGO_CONSTANTS['M'](n, dtype=precision)
         
         config = build_config(
             BackwardsEulerStepConfig,
             required={
                 'precision': precision,
                 'n': n,
                 'dxdt_function': dxdt_function,
                 'observables_function': observables_function,
                 'driver_function': driver_function,
                 'get_solver_helper_fn': get_solver_helper_fn,
                 'beta': beta,
                 'gamma': gamma,
                 'M': M,
             },
             **kwargs
         )
         
         # Extract solver-specific kwargs and pass to parent
         super().__init__(config, BE_DEFAULTS.copy(), **kwargs)
         
         self.register_buffers()
     ```
   - Edge cases:
     - Solver kwargs must flow through to ODEImplicitStep
     - ALGO_CONSTANTS values computed before build_config
   - Integration: Parent __init__ also receives **kwargs for solver configuration

2. **Add import for build_config**
   - File: src/cubie/integrators/algorithms/backwards_euler.py
   - Action: Modify
   - Details: Add to imports section:
     ```python
     from cubie._utils import PrecisionDType, build_config
     ```

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/backwards_euler.py (89 lines reduced to 55 lines in __init__)
- Functions/Methods Modified:
  * BackwardsEulerStep.__init__() - refactored to use build_config pattern
- Implementation Summary:
  * Added `build_config` import to the file imports
  * Replaced verbose 15+ parameter signature with 6 parameters + **kwargs
  * Replaced manual config_kwargs building with `build_config()` helper
  * Replaced manual solver_kwargs building with direct **kwargs pass-through
  * All optional parameters (preconditioner_order, krylov_tolerance, max_linear_iters,
    linear_correction_type, newton_tolerance, max_newton_iters, newton_damping,
    newton_max_backtracks, preconditioned_vec_location, temp_location, delta_location,
    residual_location, residual_temp_location, stage_base_bt_location,
    increment_cache_location) now captured via **kwargs
  * Docstring updated to reference **kwargs pattern
- Parameter Information Collected for Documentation:
  * precision: PrecisionDType - Precision applied to device buffers (required)
  * n: int - Number of state entries advanced per step (required)
  * dxdt_function: Optional[Callable] - Device derivative function (optional)
  * observables_function: Optional[Callable] - Device observables function (optional)
  * driver_function: Optional[Callable] - Device driver function (optional)
  * get_solver_helper_fn: Optional[Callable] - Solver helper callable (optional)
  * preconditioner_order: int - Order of truncated Neumann preconditioner (default: 1)
  * increment_cache_location: str - 'local' or 'shared' (default: 'local')
  * krylov_tolerance: float - Linear solver tolerance
  * max_linear_iters: int - Max linear solver iterations
  * linear_correction_type: str - Linear correction strategy
  * newton_tolerance: float - Newton convergence tolerance
  * max_newton_iters: int - Max Newton iterations
  * newton_damping: float - Newton damping factor
  * newton_max_backtracks: int - Max Newton backtracks
  * preconditioned_vec_location: str - Buffer location
  * temp_location: str - Buffer location
  * delta_location: str - Buffer location
  * residual_location: str - Buffer location
  * residual_temp_location: str - Buffer location
  * stage_base_bt_location: str - Buffer location
- Issues Flagged: None

---

## Task Group 3: Refactor ODEImplicitStep Base Class - [SEQUENTIAL]
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 147-250)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 160-280)

**Input Validation Required**:
- config: Must be ImplicitStepConfig or subclass
- solver_type: Must be 'newton' or 'linear'
- Solver kwargs validated by respective config classes

**Tasks**:
1. **Update ODEImplicitStep.__init__ to accept **kwargs**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     Replace verbose pattern (lines 86-206) with:
     ```python
     def __init__(
         self,
         config: ImplicitStepConfig,
         _controller_defaults: StepControlDefaults,
         solver_type: str = "newton",
         **kwargs,
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
         **kwargs
             Optional solver parameters (krylov_tolerance, max_linear_iters,
             newton_tolerance, etc.). None values are ignored and defaults
             from solver config classes are used.
         """
         super().__init__(config, _controller_defaults)
         
         if solver_type not in ['newton', 'linear']:
             raise ValueError(
                 f"solver_type must be 'newton' or 'linear', got '{solver_type}'"
             )
         
         # Pass kwargs to solvers - they handle None filtering internally
         linear_solver = LinearSolver(
             precision=config.precision,
             n=config.n,
             **kwargs,
         )
         
         if solver_type == 'newton':
             self.solver = NewtonKrylov(
                 precision=config.precision,
                 n=config.n,
                 linear_solver=linear_solver,
                 **kwargs,
             )
         else:
             self.solver = linear_solver
     ```
   - Edge cases:
     - Both LinearSolver and NewtonKrylov receive same kwargs (they filter internally)
     - Solver classes must handle extra kwargs gracefully
   - Integration: Child algorithm classes (BackwardsEuler, DIRK, etc.) pass **kwargs up

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/ode_implicitstep.py (lines 86-206 replaced with lines 86-160)
- Functions/Methods Modified:
  * ODEImplicitStep.__init__() - refactored to use **kwargs pattern
- Implementation Summary:
  * Added class-level frozenset constants _LINEAR_SOLVER_PARAMS and _NEWTON_KRYLOV_PARAMS
    to define valid parameters for each solver type
  * Replaced 13 explicit optional parameters (krylov_tolerance, max_linear_iters,
    linear_correction_type, newton_tolerance, max_newton_iters, newton_damping,
    newton_max_backtracks, preconditioned_vec_location, temp_location, delta_location,
    residual_location, residual_temp_location, stage_base_bt_location) with **kwargs
  * Removed verbose 13-line if-not-None pattern for linear_solver_kwargs
  * Removed verbose 8-line if-not-None pattern for newton_kwargs
  * Added dict comprehension filtering that extracts only valid params for each solver
    and filters None values in one step
  * Docstring updated to reference **kwargs pattern
- [x] ODEImplicitStep.__init__ uses **kwargs pattern
- [x] Verbose parameter forwarding removed
- [x] Solver classes receive kwargs directly (filtered to valid params)
- Issues Flagged: None

---

## Task Group 4: Refactor Other Algorithm Init Functions - [PARALLEL]
**Status**: [x]
**Dependencies**: Task Group 3

**Required Context**:
- File: src/cubie/integrators/algorithms/crank_nicolson.py (entire file)
- File: src/cubie/integrators/algorithms/generic_dirk.py (entire file)
- File: src/cubie/integrators/algorithms/generic_firk.py (entire file)
- File: src/cubie/integrators/algorithms/generic_erk.py (entire file)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (entire file)
- File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py (entire file)
- File: src/cubie/_utils.py (build_config function)

**Input Validation Required**:
- Each algorithm validates through its config class
- Required params: precision, n (all algorithms)
- tableau: Required for ERK, DIRK, FIRK, Rosenbrock-W
- Buffer locations: Validated by config class validators

**Tasks**:
1. **Refactor DIRKStep.__init__**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     Replace lines 132-326 with build_config pattern:
     ```python
     def __init__(
         self,
         precision: PrecisionDType,
         n: int,
         dxdt_function: Optional[Callable] = None,
         observables_function: Optional[Callable] = None,
         driver_function: Optional[Callable] = None,
         get_solver_helper_fn: Optional[Callable] = None,
         tableau: DIRKTableau = DEFAULT_DIRK_TABLEAU,
         n_drivers: int = 0,
         **kwargs,
     ) -> None:
         """Initialise the DIRK step configuration.
         
         ... (keep existing docstring, update to mention **kwargs)
         """
         from cubie._utils import build_config
         
         mass = np.eye(n, dtype=precision)
         
         config = build_config(
             DIRKStepConfig,
             required={
                 'precision': precision,
                 'n': n,
                 'n_drivers': n_drivers,
                 'dxdt_function': dxdt_function,
                 'observables_function': observables_function,
                 'driver_function': driver_function,
                 'get_solver_helper_fn': get_solver_helper_fn,
                 'tableau': tableau,
                 'beta': 1.0,
                 'gamma': 1.0,
                 'M': mass,
             },
             **kwargs
         )
         
         # Select defaults based on error estimate
         if tableau.has_error_estimate:
             controller_defaults = DIRK_ADAPTIVE_DEFAULTS
         else:
             controller_defaults = DIRK_FIXED_DEFAULTS
         
         super().__init__(config, controller_defaults, **kwargs)
         
         self.register_buffers()
     ```
   - Edge cases: Controller defaults selection must happen before super().__init__

2. **Refactor CrankNicolsonStep.__init__**
   - File: src/cubie/integrators/algorithms/crank_nicolson.py
   - Action: Modify
   - Details: Apply same pattern as BackwardsEulerStep

3. **Refactor GenericERKStep.__init__**
   - File: src/cubie/integrators/algorithms/generic_erk.py
   - Action: Modify
   - Details: Apply build_config pattern (explicit algorithm - simpler case)

4. **Refactor FIRKStep.__init__**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details: Apply build_config pattern with tableau handling

5. **Refactor RosenbrockWStep.__init__**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details: Apply build_config pattern with tableau handling

6. **Refactor BackwardsEulerPCStep.__init__**
   - File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py
   - Action: Modify
   - Details: BackwardsEulerPCStep inherits from BackwardsEulerStep and does not
     override __init__, so no changes needed (already refactored via parent).

7. **Refactor ExplicitEulerStep.__init__**
   - File: src/cubie/integrators/algorithms/explicit_euler.py
   - Action: Modify
   - Details: Apply build_config pattern for consistency

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/generic_dirk.py (reduced from ~195 to ~88 lines in __init__)
  * src/cubie/integrators/algorithms/crank_nicolson.py (reduced from ~150 to ~55 lines in __init__)
  * src/cubie/integrators/algorithms/generic_erk.py (reduced from ~110 to ~95 lines in __init__)
  * src/cubie/integrators/algorithms/generic_firk.py (reduced from ~195 to ~88 lines in __init__)
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (reduced from ~175 to ~92 lines in __init__)
  * src/cubie/integrators/algorithms/explicit_euler.py (updated to use build_config pattern)
- Functions/Methods Modified:
  * DIRKStep.__init__() - refactored to use build_config and **kwargs pattern
  * CrankNicolsonStep.__init__() - refactored to use build_config and **kwargs pattern
  * ERKStep.__init__() - refactored to use build_config and **kwargs pattern
  * FIRKStep.__init__() - refactored to use build_config and **kwargs pattern
  * GenericRosenbrockWStep.__init__() - refactored to use build_config and **kwargs pattern
  * ExplicitEulerStep.__init__() - refactored to use build_config and **kwargs pattern
- Implementation Summary:
  * Added `build_config` import to all algorithm files
  * Replaced verbose explicit optional parameters with `**kwargs`
  * Replaced manual config_kwargs building with `build_config()` helper
  * Replaced manual solver_kwargs building with direct **kwargs pass-through
  * Docstrings updated to reference **kwargs pattern
  * BackwardsEulerPCStep already uses parent's refactored __init__
- [x] All algorithm __init__ methods use build_config pattern
- [x] Verbose `if param is not None:` checks removed from all algorithms
- [x] Each algorithm file updated with correct imports
- [x] Parameter information collected for documentation
- Issues Flagged: None

---

## Task Group 5: Refactor LinearSolver and NewtonKrylov - [SEQUENTIAL]
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 147-250)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 160-280)

**Input Validation Required**:
- precision: Validated by config class
- n: Validated by config class
- Buffer locations: Validated by config class validators

**Tasks**:
1. **Update LinearSolver.__init__ to use build_config**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details: Apply build_config pattern, filtering unused kwargs

2. **Update NewtonKrylov.__init__ to use build_config**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details: Apply build_config pattern, filtering unused kwargs

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (reduced from 55 to 32 lines in __init__)
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (reduced from 76 to 38 lines in __init__)
- Functions/Methods Modified:
  * LinearSolver.__init__() - refactored to use build_config and **kwargs pattern
  * NewtonKrylov.__init__() - refactored to use build_config and **kwargs pattern
- Implementation Summary:
  * Added `build_config` import to both solver files
  * Replaced verbose 5+ explicit optional parameters with `**kwargs`
  * Replaced manual if-not-None pattern building with `build_config()` helper
  * LinearSolver: Removed linear_correction_type, krylov_tolerance, max_linear_iters,
    preconditioned_vec_location, temp_location explicit params
  * NewtonKrylov: Removed newton_tolerance, max_newton_iters, newton_damping,
    newton_max_backtracks, delta_location, residual_location, residual_temp_location,
    stage_base_bt_location explicit params
  * Docstrings updated to reference **kwargs pattern and config classes
- [x] Solver classes use build_config pattern
- [x] Unused kwargs are silently ignored (no warnings)
- [x] Parameter validation happens in config classes
- Issues Flagged: None

---

## Task Group 6: Refactor Step Controllers - [PARALLEL]
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/step_control/adaptive_PID_controller.py (entire file)
- File: src/cubie/integrators/step_control/adaptive_PI_controller.py (entire file)
- File: src/cubie/integrators/step_control/adaptive_I_controller.py (entire file)
- File: src/cubie/integrators/step_control/fixed_step_controller.py (entire file)
- File: src/cubie/integrators/step_control/gustafsson_controller.py (entire file)

**Input Validation Required**:
- precision: Validated by config class
- Controller-specific params validated by respective config classes

**Tasks**:
1. **Refactor AdaptivePIDController.__init__**
   - File: src/cubie/integrators/step_control/adaptive_PID_controller.py
   - Action: Modify (if verbose pattern exists)
   - Details: Current pattern is already clean - verify and enhance if needed

2. **Refactor AdaptivePIController.__init__**
   - File: src/cubie/integrators/step_control/adaptive_PI_controller.py
   - Action: Modify (if verbose pattern exists)

3. **Refactor AdaptiveIController.__init__**
   - File: src/cubie/integrators/step_control/adaptive_I_controller.py
   - Action: Modify (if verbose pattern exists)

4. **Refactor FixedStepController.__init__**
   - File: src/cubie/integrators/step_control/fixed_step_controller.py
   - Action: Modify (if verbose pattern exists)

5. **Refactor GustafssonController.__init__**
   - File: src/cubie/integrators/step_control/gustafsson_controller.py
   - Action: Modify (if verbose pattern exists)

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/step_control/adaptive_PID_controller.py (reduced from ~69 to ~27 lines in __init__)
  * src/cubie/integrators/step_control/adaptive_PI_controller.py (reduced from ~64 to ~27 lines in __init__)
  * src/cubie/integrators/step_control/adaptive_I_controller.py (reduced from ~56 to ~27 lines in __init__)
  * src/cubie/integrators/step_control/fixed_step_controller.py (updated to use build_config pattern)
  * src/cubie/integrators/step_control/gustafsson_controller.py (reduced from ~65 to ~28 lines in __init__)
- Functions/Methods Modified:
  * AdaptivePIDController.__init__() - refactored to use build_config and **kwargs pattern
  * AdaptivePIController.__init__() - refactored to use build_config and **kwargs pattern
  * AdaptiveIController.__init__() - refactored to use build_config and **kwargs pattern
  * FixedStepController.__init__() - refactored to use build_config and **kwargs pattern
  * GustafssonController.__init__() - refactored to use build_config and **kwargs pattern
- Implementation Summary:
  * Added `build_config` import to all controller files
  * Replaced verbose explicit optional parameters with `**kwargs`
  * Replaced direct config class instantiation with `build_config()` helper
  * Docstrings updated to reference **kwargs pattern and config classes
  * All 12+ explicit parameters per controller reduced to 2-3 required params + **kwargs
- Parameter Information Collected for Documentation:
  * AdaptivePIDController: precision (req), n (opt), dt_min, dt_max, atol, rtol, 
    algorithm_order, kp, ki, kd, min_gain, max_gain, deadband_min, deadband_max
  * AdaptivePIController: precision (req), n (opt), dt_min, dt_max, atol, rtol,
    algorithm_order, kp, ki, min_gain, max_gain, deadband_min, deadband_max
  * AdaptiveIController: precision (req), n (opt), dt_min, dt_max, atol, rtol,
    algorithm_order, min_gain, max_gain, deadband_min, deadband_max
  * FixedStepController: precision (req), dt (req), n (opt)
  * GustafssonController: precision (req), n (opt), dt_min, dt_max, atol, rtol,
    algorithm_order, min_gain, max_gain, gamma, max_newton_iters, deadband_min, deadband_max
- [x] Controller classes use consistent init pattern
- [x] Parameter information collected for documentation
- Issues Flagged: None

---

## Task Group 7: Refactor Loop and Output Functions - [PARALLEL]
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 60-300)
- File: src/cubie/outputhandling/output_functions.py (lines 66-200)

**Input Validation Required**:
- Validated by respective config classes

**Tasks**:
1. **Refactor IVPLoop.__init__ if needed**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Review and modify if verbose pattern exists

2. **Refactor OutputFunctions.__init__ if needed**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Review and modify if verbose pattern exists

**Outcomes**: 
[ ] Loop and output function classes use consistent pattern
[ ] Parameter information collected for documentation

---

## Task Group 8: Update Tests - [SEQUENTIAL]
**Status**: [ ]
**Dependencies**: Task Groups 2-7

**Required Context**:
- File: tests/test_utils.py (add tests for build_config)
- File: tests/integrators/algorithms/test_step_algorithms.py (update if needed)
- File: tests/conftest.py (fixture patterns)

**Input Validation Required**:
- Test build_config with valid attrs classes
- Test build_config with missing required fields
- Test build_config with None values in optional
- Test build_config with extra kwargs (should be ignored)
- Test build_config with attrs.Factory defaults

**Tasks**:
1. **Add unit tests for build_config helper**
   - File: tests/test_utils.py
   - Action: Modify (add new tests)
   - Details:
     ```python
     class TestBuildConfig:
         """Tests for build_config helper function."""
         
         def test_build_config_basic(self):
             """Verify basic config construction."""
             # Create test with actual config class from cubie
             from cubie.integrators.algorithms.backwards_euler import (
                 BackwardsEulerStepConfig
             )
             import numpy as np
             
             config = build_config(
                 BackwardsEulerStepConfig,
                 required={'precision': np.float32, 'n': 3},
             )
             assert config.precision == np.float32
             assert config.n == 3
         
         def test_build_config_optional_override(self):
             """Verify optional parameters override defaults."""
             ...
         
         def test_build_config_none_ignored(self):
             """Verify None values don't override defaults."""
             ...
         
         def test_build_config_missing_required(self):
             """Verify error on missing required fields."""
             ...
         
         def test_build_config_extra_kwargs_ignored(self):
             """Verify extra kwargs are silently ignored."""
             ...
         
         def test_build_config_non_attrs_raises(self):
             """Verify TypeError for non-attrs class."""
             ...
     ```

2. **Update algorithm tests if needed**
   - File: tests/integrators/algorithms/test_step_algorithms.py
   - Action: Review and update if fixtures use old patterns

**Outcomes**: 
[ ] build_config has comprehensive test coverage
[ ] Existing algorithm tests pass with refactored code
[ ] No test failures related to new pattern

---

## Task Group 9: Remove Obsolete Code and Validate - [SEQUENTIAL]
**Status**: [ ]
**Dependencies**: Task Group 8

**Required Context**:
- All modified files from previous task groups

**Input Validation Required**:
- All tests pass
- No unused imports
- Linting passes

**Tasks**:
1. **Run full test suite**
   - Action: Execute `pytest` and verify all tests pass

2. **Run linters**
   - Action: Execute `flake8` and `ruff` checks

3. **Remove any remaining obsolete patterns**
   - Action: Search for remaining `if param is not None:` patterns that should use build_config

**Outcomes**: 
[ ] All tests pass
[ ] Linting passes
[ ] No verbose patterns remain in scope of refactoring

---

## Task Group 10: Create Optional Arguments Documentation - [SEQUENTIAL]
**Status**: [ ]
**Dependencies**: Task Groups 2-7 (parameter information collected)

**Required Context**:
- File: docs/source/user_guide/index.rst (toctree to update)
- File: docs/source/user_guide/algorithms.rst (reference for style)
- Parameter information collected from Task Groups 2-7

**Input Validation Required**:
- All parameter names match source code
- Default values match config class definitions
- Applicability tables are accurate

**Tasks**:
1. **Create optional_arguments.rst**
   - File: docs/source/user_guide/optional_arguments.rst
   - Action: Create
   - Details: Create comprehensive RST documentation with:
     - Introduction explaining kwargs flow
     - Algorithm Options section with parameter descriptions and applicability table
     - Controller Options section with parameter descriptions and applicability table
     - Loop Options section
     - Output Options section
   - Style: Follow existing docs/source/user_guide/*.rst style

2. **Update user_guide/index.rst toctree**
   - File: docs/source/user_guide/index.rst
   - Action: Modify
   - Details: Add `optional_arguments` after `algorithms` in toctree

**Outcomes**: 
[ ] optional_arguments.rst created with all parameters documented
[ ] Sphinx builds successfully
[ ] Documentation is user-friendly and technically accurate

---

## Summary

| Task Group | Description | Execution | Dependencies |
|------------|-------------|-----------|--------------|
| 1 | Add build_config helper | SEQUENTIAL | None |
| 2 | Refactor BackwardsEulerStep | SEQUENTIAL | 1 |
| 3 | Refactor ODEImplicitStep | SEQUENTIAL | 2 |
| 4 | Refactor other algorithms | PARALLEL | 3 |
| 5 | Refactor solvers | SEQUENTIAL | 1 |
| 6 | Refactor controllers | PARALLEL | 1 |
| 7 | Refactor loop/output | PARALLEL | 1 |
| 8 | Update tests | SEQUENTIAL | 2-7 |
| 9 | Validate and cleanup | SEQUENTIAL | 8 |
| 10 | Create documentation | SEQUENTIAL | 2-7 |

### Parallel Execution Opportunities

- Task Groups 4, 5, 6, 7 can execute in parallel after their dependencies are met
- Task Group 10 can start drafting during Task Groups 2-7 (collecting parameter info)

### Estimated Complexity

- **Task Group 1**: Low - single function addition
- **Task Groups 2-7**: Medium - repetitive refactoring with consistent pattern
- **Task Group 8**: Medium - test updates and new test creation
- **Task Group 9**: Low - validation and cleanup
- **Task Group 10**: Medium - documentation writing

### Critical Path

1 → 2 → 3 → 4 → 8 → 9

Parallel paths: 1 → 5, 1 → 6, 1 → 7, and 2-7 → 10
