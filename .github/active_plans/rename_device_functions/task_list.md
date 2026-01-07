# Implementation Task List
# Feature: Device Function Renaming
# Plan Reference: .github/active_plans/rename_device_functions/agent_plan.md

## Task Group 1: Update Parameter Name Constants
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (lines 23-50)

**Input Validation Required**:
- None (string set modification only)

**Tasks**:
1. **Update ALL_ALGORITHM_STEP_PARAMETERS constant**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     ```python
     # Line 23-50: Update the set definition
     ALL_ALGORITHM_STEP_PARAMETERS = {
         'algorithm',
         'precision', 'n', 'evaluate_f', 'evaluate_observables',
         'evaluate_driver_at_t', 'get_solver_helper_fn', "driver_del_t",
         # ... rest remains unchanged
     }
     ```
   - Edge cases: None (simple set constant)
   - Integration: Referenced by algorithm initialization and validation logic

**Tests to Create**:
- None (tests will validate indirectly through algorithm initialization)

**Tests to Run**:
- tests/integrators/algorithms/test_explicit_euler.py::test_explicit_euler_step_init
- tests/integrators/algorithms/test_backwards_euler.py::test_backwards_euler_init

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/algorithms/base_algorithm_step.py (3 lines changed on line 25-26)
- Functions/Methods Added/Modified:
  * ALL_ALGORITHM_STEP_PARAMETERS constant updated
- Implementation Summary:
  Updated the ALL_ALGORITHM_STEP_PARAMETERS set to use the new parameter names: 'evaluate_f', 'evaluate_observables', and 'evaluate_driver_at_t' instead of 'dxdt_function', 'observables_function', and 'driver_function'. This constant is used for algorithm initialization and validation logic throughout the codebase.
- Issues Flagged: None

---

## Task Group 2: Update Base ODE Properties
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/odesystems/baseODE.py (lines 1-50, 364-380)

**Input Validation Required**:
- None (property renaming only, no logic changes)

**Tasks**:
1. **Rename dxdt_function property to evaluate_f**
   - File: src/cubie/odesystems/baseODE.py
   - Action: Modify
   - Details:
     ```python
     # Line 365-367: Rename property
     @property
     def evaluate_f(self):
         """Compiled CUDA device function for evaluating f(t, y)."""
         return self.get_cached_output("dxdt")
     ```
   - Edge cases: None (cache key "dxdt" remains unchanged for backward compatibility with ODECache)
   - Integration: Called by algorithm steps to retrieve device function

2. **Rename observables_function property to evaluate_observables**
   - File: src/cubie/odesystems/baseODE.py
   - Action: Modify
   - Details:
     ```python
     # Line 370-379: Rename property and update docstring
     @property
     def evaluate_observables(self) -> Callable:
         """Return the compiled observables device function.

         Returns
         -------
         Callable
             CUDA device function that computes observables without
             updating the derivative buffer.
         """
         return self.get_cached_output("observables")
     ```
   - Edge cases: None (cache key "observables" remains unchanged)
   - Integration: Called by algorithm steps and output handlers

**Tests to Create**:
- None (existing tests will be updated in later task groups)

**Tests to Run**:
- tests/odesystems/test_baseODE.py::test_base_ode_properties
- tests/odesystems/symbolic/test_symbolicODE.py::test_symbolic_ode_compilation

**Outcomes**:
- Files Modified: 
  * src/cubie/odesystems/baseODE.py (8 lines changed: lines 365-367, 370-379)
- Functions/Methods Added/Modified:
  * evaluate_f property (renamed from dxdt_function)
  * evaluate_observables property (renamed from observables_function)
- Implementation Summary:
  Renamed two BaseODE properties to use more descriptive names:
  - dxdt_function → evaluate_f (evaluates f(t, y))
  - observables_function → evaluate_observables
  The cache keys ("dxdt" and "observables") remain unchanged for backward compatibility with ODECache. Updated docstrings to reflect the new property names and improved clarity.
- Issues Flagged: None

---

## Task Group 3: Update BaseStepConfig Class
**Status**: [x]
**Dependencies**: Groups [1, 2]

**Required Context**:
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (lines 338-410)

**Input Validation Required**:
- None (field renaming only)

**Tasks**:
1. **Rename dxdt_function field to evaluate_f**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     ```python
     # Lines 369-373: Rename field and update docstring
     evaluate_f: Optional[Callable] = field(
         default=None,
         validator=validators.optional(is_device_validator),
         eq=False
     )
     ```
   - Edge cases: None
   - Integration: Used by all algorithm step implementations

2. **Rename observables_function field to evaluate_observables**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     ```python
     # Lines 374-378: Rename field
     evaluate_observables: Optional[Callable] = field(
         default=None,
         validator=validators.optional(is_device_validator),
         eq=False
     )
     ```
   - Edge cases: None
   - Integration: Used by all algorithm step implementations

3. **Rename driver_function field to evaluate_driver_at_t**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     ```python
     # Lines 379-383: Rename field
     evaluate_driver_at_t: Optional[Callable] = field(
         default=None,
         validator=validators.optional(is_device_validator),
         eq=False
     )
     ```
   - Edge cases: None
   - Integration: Used by algorithms that support driver functions

4. **Update docstring parameter descriptions**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     ```python
     # Lines 351-359: Update parameter descriptions in class docstring
     """Configuration shared by explicit and implicit integration steps.

     Parameters
     ----------
     precision
         Numerical precision to apply to device buffers. Supported values are
         ``float16``, ``float32``, and ``float64``.
     n
         Number of state entries advanced by each step call.
     n_drivers
         Number of external driver signals consumed by the step (>= 0).
     evaluate_f
         Device function that evaluates the system right-hand side f(t, y).
     evaluate_observables
         Device function that evaluates the system observables.
     evaluate_driver_at_t
         Device function that evaluates driver arrays for a given time t.
     get_solver_helper_fn
         Optional callable that returns device helpers required by the
         nonlinear solver construction.
     """
     ```
   - Edge cases: None
   - Integration: Documentation for config class

**Tests to Create**:
- None (config class is tested through algorithm initialization)

**Tests to Run**:
- tests/integrators/algorithms/test_explicit_euler.py::test_explicit_euler_step_init
- tests/integrators/algorithms/test_backwards_euler.py::test_backwards_euler_init

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/algorithms/base_algorithm_step.py (16 lines changed: lines 340-360 docstring, lines 369-383 field definitions)
- Functions/Methods Added/Modified:
  * evaluate_f field (renamed from dxdt_function)
  * evaluate_observables field (renamed from observables_function)
  * evaluate_driver_at_t field (renamed from driver_function)
- Implementation Summary:
  Renamed three field names in the BaseStepConfig class to use more descriptive names:
  - dxdt_function → evaluate_f (evaluates the system right-hand side f(t, y))
  - observables_function → evaluate_observables
  - driver_function → evaluate_driver_at_t (evaluates driver arrays for a given time t)
  Updated the class docstring to reflect the new field names with improved parameter descriptions. All field definitions, validators, and default values remain unchanged. These fields are used by all explicit and implicit algorithm step implementations throughout the codebase.
- Issues Flagged: None

---

## Task Group 4: Update ODEExplicitStep Base Class
**Status**: [x]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_explicitstep.py (entire file)

**Input Validation Required**:
- None (parameter renaming only)

**Tasks**:
1. **Update build method variable names**
   - File: src/cubie/integrators/algorithms/ode_explicitstep.py
   - Action: Modify
   - Details:
     ```python
     # Lines 24-47: Update variable names in build method
     def build(self) -> StepCache:
         """Create and cache the device function for the explicit algorithm.

         Returns
         -------
         StepCache
             Container with the compiled step device function.
         """

         config = self.compile_settings
         evaluate_f = config.evaluate_f
         numba_precision = config.numba_precision
         n = config.n
         evaluate_observables = config.evaluate_observables
         evaluate_driver_at_t = config.evaluate_driver_at_t
         n_drivers = config.n_drivers
         return self.build_step(
             evaluate_f,
             evaluate_observables,
             evaluate_driver_at_t,
             numba_precision,
             n,
             n_drivers,
         )
     ```
   - Edge cases: None
   - Integration: Called by all explicit algorithm implementations

2. **Update build_step abstract method signature**
   - File: src/cubie/integrators/algorithms/ode_explicitstep.py
   - Action: Modify
   - Details:
     ```python
     # Lines 50-81: Update method signature and docstring
     @abstractmethod
     def build_step(
         self,
         evaluate_f: Callable,
         evaluate_observables: Callable,
         evaluate_driver_at_t: Optional[Callable],
         numba_precision: type,
         n: int,
         n_drivers: int,
     ) -> StepCache:
         """Build and return the explicit step device function.

         Parameters
         ----------
         evaluate_f
             Device function for evaluating the ODE right-hand side f(t, y).
         evaluate_observables
             Device helper that computes observables for the system.
         evaluate_driver_at_t
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
         """
         raise NotImplementedError
     ```
   - Edge cases: None
   - Integration: Abstract method implemented by all explicit algorithms

**Tests to Create**:
- None (base class tested through concrete implementations)

**Tests to Run**:
- tests/integrators/algorithms/test_explicit_euler.py::test_explicit_euler_step
- tests/integrators/algorithms/test_generic_erk.py::test_generic_erk_step

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/algorithms/ode_explicitstep.py (15 lines changed: lines 34-38 variable names, lines 40-46 parameter passing, lines 52-54 method signature, lines 63-68 docstring)
- Functions/Methods Added/Modified:
  * build() method - updated variable names to use evaluate_f, evaluate_observables, evaluate_driver_at_t
  * build_step() abstract method - updated parameter names and docstring
- Implementation Summary:
  Updated the ODEExplicitStep base class to use the new parameter naming convention. In the build() method, renamed local variables from dxdt_function → evaluate_f, observables_function → evaluate_observables, and driver_function → evaluate_driver_at_t. Updated the abstract build_step() method signature to use the same new parameter names and improved the docstring to clarify that evaluate_f evaluates "the ODE right-hand side f(t, y)" rather than the less descriptive "derivative function". This base class is inherited by all explicit algorithm implementations (ExplicitEulerStep, GenericERKStep, etc.).
- Issues Flagged: None

---

## Task Group 5: Update ODEImplicitStep Base Class
**Status**: [x]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (entire file)

**Input Validation Required**:
- None (parameter renaming only)

**Tasks**:
1. **Update build method variable names**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     # Find the build method (similar pattern to explicit step)
     # Update all references from dxdt_function → evaluate_f
     # Update all references from observables_function → evaluate_observables
     # Update all references from driver_function → evaluate_driver_at_t
     ```
   - Edge cases: None
   - Integration: Called by all implicit algorithm implementations

2. **Update build_step abstract method signature**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     # Update abstract method signature and docstring
     # Parameter names: evaluate_f, evaluate_observables, evaluate_driver_at_t
     # Update parameter descriptions in docstring
     ```
   - Edge cases: None
   - Integration: Abstract method implemented by all implicit algorithms

**Tests to Create**:
- None (base class tested through concrete implementations)

**Tests to Run**:
- tests/integrators/algorithms/test_backwards_euler.py::test_backwards_euler_step
- tests/integrators/algorithms/test_crank_nicolson.py::test_crank_nicolson_step

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/algorithms/ode_implicitstep.py (14 lines changed: lines 218-223 variable names, lines 227-230 parameter passing, lines 239-241 method signature, lines 251-257 docstring)
- Functions/Methods Added/Modified:
  * build() method - updated variable names to use evaluate_f, evaluate_observables, evaluate_driver_at_t
  * build_step() abstract method - updated parameter names and docstring
- Implementation Summary:
  Updated the ODEImplicitStep base class to use the new parameter naming convention, mirroring the changes made to ODEExplicitStep in Task Group 4. In the build() method, renamed local variables from dxdt_fn → evaluate_f, observables_function → evaluate_observables, and driver_function → evaluate_driver_at_t. Updated the abstract build_step() method signature to use the same new parameter names and improved the docstring to clarify that evaluate_f evaluates "the ODE right-hand side f(t, y)" rather than "device derivative function for the ODE system". The solver_function parameter remains unchanged. This base class is inherited by all implicit algorithm implementations (BackwardsEulerStep, CrankNicolsonStep, etc.).
- Issues Flagged: None

---

## Task Group 6: Update Explicit Algorithm Implementations (Part 1)
**Status**: [x]
**Dependencies**: Groups [1, 2, 3, 4]

**Required Context**:
- File: src/cubie/integrators/algorithms/explicit_euler.py (entire file)
- File: src/cubie/integrators/algorithms/generic_erk.py (entire file)

**Input Validation Required**:
- None (parameter renaming only)

**Tasks**:
1. **Update ExplicitEulerStep.__init__ signature**
   - File: src/cubie/integrators/algorithms/explicit_euler.py
   - Action: Modify
   - Details:
     ```python
     # Lines 25-56: Update __init__ signature and docstring
     def __init__(
         self,
         precision: PrecisionDType,
         n: int,
         evaluate_f: Optional[Callable] = None,
         evaluate_observables: Optional[Callable] = None,
         evaluate_driver_at_t: Optional[Callable] = None,
         get_solver_helper_fn: Optional[Callable] = None,
         **kwargs,
     ) -> None:
         """Initialise the explicit Euler step configuration.

         Parameters
         ----------
         precision
             Precision applied to device buffers.
         n
             Number of state entries advanced per step.
         evaluate_f
             Device function for evaluating f(t, y) right-hand side.
         evaluate_observables
             Device function computing system observables.
         evaluate_driver_at_t
             Optional device function evaluating drivers at arbitrary times.
         get_solver_helper_fn
             Present for interface parity with implicit steps and ignored here.
         **kwargs
             Optional parameters passed to config classes. See
             ExplicitStepConfig for available parameters. None values are
             ignored.
         """
         config = build_config(
             ExplicitStepConfig,
             required={
                 'precision': precision,
                 'n': n,
                 'evaluate_f': evaluate_f,
                 'evaluate_observables': evaluate_observables,
                 'evaluate_driver_at_t': evaluate_driver_at_t,
             },
             **kwargs
         )

         super().__init__(config, EE_DEFAULTS.copy())
     ```
   - Edge cases: None
   - Integration: Constructor called by SingleIntegratorRunCore

2. **Update ExplicitEulerStep.build_step signature and implementation**
   - File: src/cubie/integrators/algorithms/explicit_euler.py
   - Action: Modify
   - Details:
     ```python
     # Lines 71-end: Update build_step signature, parameters, and all references
     def build_step(
         self,
         evaluate_f: Callable,
         evaluate_observables: Callable,
         evaluate_driver_at_t: Optional[Callable],
         numba_precision: type,
         n: int,
         n_drivers: int,
     ) -> StepCache:
         """Build the device function for an explicit Euler step.
         
         Parameters
         ----------
         evaluate_f
             Device function for evaluating f(t, y).
         evaluate_observables
             Device function for computing observables.
         evaluate_driver_at_t
             Optional device function for evaluating drivers at time t.
         numba_precision
             Numba type for device buffers.
         n
             State vector dimension.
         n_drivers
             Number of driver signals.
             
         Returns
         -------
         StepCache
             Compiled step function.
         """
         # Update all internal references in device function compilation
         # Replace dxdt_function(...) calls with evaluate_f(...)
         # Replace observables_function(...) with evaluate_observables(...)
         # Replace driver_function(...) with evaluate_driver_at_t(...)
     ```
   - Edge cases: Check conditional branches for driver_function presence
   - Integration: CUDA device code compiled and cached

3. **Update GenericERKStep with same pattern**
   - File: src/cubie/integrators/algorithms/generic_erk.py
   - Action: Modify
   - Details: Apply same renaming pattern as ExplicitEulerStep to __init__ and build_step
   - Edge cases: ERK has multiple stage evaluations, ensure all calls updated
   - Integration: Tableau-based algorithm uses evaluate_f multiple times per step

**Tests to Create**:
- None (existing tests cover functionality)

**Tests to Run**:
- tests/integrators/algorithms/test_explicit_euler.py
- tests/integrators/algorithms/test_generic_erk.py

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/algorithms/explicit_euler.py (47 lines changed)
  * src/cubie/integrators/algorithms/generic_erk.py (52 lines changed)
- Functions/Methods Added/Modified:
  * ExplicitEulerStep.__init__() - renamed parameters (evaluate_f, evaluate_observables, evaluate_driver_at_t)
  * ExplicitEulerStep.build_step() - renamed parameters and all device function calls
  * ERKStep.__init__() - renamed parameters (evaluate_f, evaluate_observables, evaluate_driver_at_t)
  * ERKStep.build_step() - renamed parameters and all device function calls (3 locations in device code)
- Implementation Summary:
  Updated ExplicitEulerStep and GenericERKStep to use the new parameter naming convention.
  For both classes:
  - Updated __init__ signature to use evaluate_f, evaluate_observables, and evaluate_driver_at_t
  - Updated parameter docstrings to improve clarity
  - Updated build_step signature to match new parameter names
  - Updated all device function calls within CUDA kernels to use new names:
    * dxdt_function → evaluate_f
    * observables_function → evaluate_observables  
    * driver_function → evaluate_driver_at_t
  ExplicitEuler has 2 function calls updated (evaluate_f once, evaluate_observables once).
  GenericERK has 5 function calls updated (evaluate_f twice for stage 0 and stages 1-s, evaluate_observables twice, evaluate_driver_at_t twice for stage evaluations and final time).
- Issues Flagged: None

---

## Task Group 7: Update Implicit Algorithm Implementations (Part 1)
**Status**: [x]
**Dependencies**: Groups [1, 2, 3, 5]

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler.py (entire file)
- File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py (entire file)
- File: src/cubie/integrators/algorithms/crank_nicolson.py (entire file)

**Input Validation Required**:
- None (parameter renaming only)

**Tasks**:
1. **Update BackwardsEulerStep**
   - File: src/cubie/integrators/algorithms/backwards_euler.py
   - Action: Modify
   - Details: Apply renaming pattern to __init__ signature, build method, and all device code references
   - Edge cases: Newton solver calls evaluate_f in residual computation
   - Integration: Implicit solver with Newton-Krylov iteration

2. **Update BackwardsEulerPCStep**
   - File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py
   - Action: Modify
   - Details: Apply renaming pattern to __init__ signature, build method, and device code
   - Edge cases: Predictor-corrector has explicit prediction step using evaluate_f
   - Integration: Predictor-corrector variant of backward Euler

3. **Update CrankNicolsonStep**
   - File: src/cubie/integrators/algorithms/crank_nicolson.py
   - Action: Modify
   - Details: Apply renaming pattern to __init__ signature, build method, and device code
   - Edge cases: Crank-Nicolson evaluates f at both t and t+dt
   - Integration: Second-order implicit method

**Tests to Create**:
- None (existing tests cover functionality)

**Tests to Run**:
- tests/integrators/algorithms/test_backwards_euler.py
- tests/integrators/algorithms/test_backwards_euler_predict_correct.py
- tests/integrators/algorithms/test_crank_nicolson.py

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/algorithms/backwards_euler.py (43 lines changed)
  * src/cubie/integrators/algorithms/backwards_euler_predict_correct.py (23 lines changed)
  * src/cubie/integrators/algorithms/crank_nicolson.py (37 lines changed)
- Functions/Methods Added/Modified:
  * BackwardsEulerStep.__init__() - renamed parameters (evaluate_f, evaluate_observables, evaluate_driver_at_t)
  * BackwardsEulerStep.build_step() - renamed parameters and all device function calls
  * BackwardsEulerStep.evaluate_f property - renamed from dxdt_function
  * BackwardsEulerPCStep.build_step() - renamed parameters and all device function calls (evaluate_f for predictor, evaluate_observables for output, evaluate_driver_at_t for driver interpolation)
  * CrankNicolsonStep.__init__() - renamed parameters (evaluate_f, evaluate_observables, evaluate_driver_at_t)
  * CrankNicolsonStep.build_step() - renamed parameters and all device function calls (evaluate_f for initial state derivative, evaluate_observables for output, evaluate_driver_at_t for driver interpolation)
- Implementation Summary:
  Updated three implicit algorithm implementations to use the new parameter naming convention:
  - BackwardsEulerStep: Updated __init__ signature, config dictionary, build_step signature and docstrings, device function calls (evaluate_driver_at_t once, evaluate_observables once), and property name (dxdt_function → evaluate_f)
  - BackwardsEulerPCStep: Updated build_step signature and docstrings, device function calls (evaluate_f once for predictor, evaluate_driver_at_t once, evaluate_observables once)
  - CrankNicolsonStep: Updated __init__ signature, config dictionary, build_step signature and docstrings, device function calls (evaluate_f once for initial derivative, evaluate_driver_at_t once, evaluate_observables once)
  All three classes now use evaluate_f instead of dxdt_fn/dxdt_function, evaluate_observables instead of observables_function, and evaluate_driver_at_t instead of driver_function throughout their implementations.
- Issues Flagged: None

---

## Task Group 8: Update Tableau-Based Algorithm Implementations
**Status**: [x]
**Dependencies**: Groups [1, 2, 3, 4, 5]

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (entire file)
- File: src/cubie/integrators/algorithms/generic_firk.py (entire file)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (entire file)

**Input Validation Required**:
- None (parameter renaming only)

**Tasks**:
1. **Update GenericDIRKStep**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details: Apply renaming pattern to __init__, build methods, and all stage evaluations
   - Edge cases: DIRK has implicit stages requiring Newton solver with evaluate_f
   - Integration: Diagonally implicit Runge-Kutta methods

2. **Update GenericFIRKStep**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details: Apply renaming pattern to __init__, build methods, and coupled stage system
   - Edge cases: FIRK solves all stages simultaneously using evaluate_f
   - Integration: Fully implicit Runge-Kutta methods

3. **Update GenericRosenbrockWStep**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details: Apply renaming pattern to __init__, build methods, and Jacobian-dependent code
   - Edge cases: Rosenbrock-W uses Jacobian from solver helpers
   - Integration: Linearly implicit methods with Jacobian action

**Tests to Create**:
- None (existing tests cover functionality)

**Tests to Run**:
- tests/integrators/algorithms/test_generic_dirk.py
- tests/integrators/algorithms/test_generic_firk.py
- tests/integrators/algorithms/test_generic_rosenbrock_w.py

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/algorithms/generic_dirk.py (59 lines changed)
  * src/cubie/integrators/algorithms/generic_firk.py (47 lines changed)
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (51 lines changed)
- Functions/Methods Added/Modified:
  * DIRKStep.__init__() - renamed parameters (evaluate_f, evaluate_observables, evaluate_driver_at_t)
  * DIRKStep.build_step() - renamed parameters and all device function calls (5 locations in device code)
  * FIRKStep.__init__() - renamed parameters (evaluate_f, evaluate_observables, evaluate_driver_at_t)
  * FIRKStep.build_step() - renamed parameters and all device function calls (2 locations in device code)
  * GenericRosenbrockWStep.__init__() - renamed parameters (evaluate_f, evaluate_observables, evaluate_driver_at_t)
  * GenericRosenbrockWStep.build_step() - renamed parameters and all device function calls (4 locations in device code)
- Implementation Summary:
  Updated three tableau-based implicit algorithm implementations to use the new parameter naming convention:
  - DIRKStep: Updated __init__ signature, config dictionary, build_step signature and docstrings. Updated device function calls at 5 locations: evaluate_driver_at_t (2x for stage 0 and stages 1-s), evaluate_observables (3x for stage 0, stages 1-s, and final time), evaluate_f (2x for stage 0 and stages 1-s).
  - FIRKStep: Updated __init__ signature, config dictionary, build_step signature and docstrings. Updated device function calls at 2 locations: evaluate_driver_at_t (1x for filling stage driver stack and 1x for end time), evaluate_observables (1x for final output).
  - GenericRosenbrockWStep: Updated __init__ signature, config dictionary, build_step signature and docstrings. Updated device function calls at 4 locations: evaluate_f (2x for stage 0 and stages 1-s), evaluate_observables (2x for stage intermediates and final time), evaluate_driver_at_t (2x for stage time and end time).
  All three classes now consistently use evaluate_f instead of dxdt_fn/dxdt_function, evaluate_observables instead of observables_function, and evaluate_driver_at_t instead of driver_function throughout their implementations. The parameter renaming maintains consistency with the base classes and other algorithm implementations.
- Issues Flagged: None

---

## Task Group 9: Update SingleIntegratorRunCore
**Status**: [x]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6, 7, 8]

**Required Context**:
- File: src/cubie/integrators/SingleIntegratorRunCore.py (entire file)
- File: src/cubie/integrators/SingleIntegratorRun.py (entire file)

**Input Validation Required**:
- None (parameter renaming only)

**Tasks**:
1. **Update SingleIntegratorRunCore.__init__ parameter**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     # Line 91: Rename parameter
     evaluate_driver_at_t: Optional[Callable] = None,
     # Update docstring on line 63
     evaluate_driver_at_t
         Optional device function which interpolates arbitrary driver inputs
         for use by step algorithms.
     ```
   - Edge cases: None
   - Integration: Constructor parameter passed to algorithm initialization

2. **Update internal variable names in __init__**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details: Update all references to driver_function → evaluate_driver_at_t in constructor body
   - Edge cases: None
   - Integration: Passed to algorithm step factory

3. **Update property exposure in SingleIntegratorRun**
   - File: src/cubie/integrators/SingleIntegratorRun.py
   - Action: Modify
   - Details: Update any property names that expose device functions using old names
   - Edge cases: None
   - Integration: Public wrapper properties

**Tests to Create**:
- None (existing integration tests cover functionality)

**Tests to Run**:
- tests/integrators/test_SingleIntegratorRun.py
- tests/integrators/loops/test_ode_loop.py

**Outcomes**:
- Files Modified:
  * src/cubie/integrators/SingleIntegratorRunCore.py (18 lines changed)
  * src/cubie/integrators/SingleIntegratorRun.py (4 lines changed)
- Functions/Methods Added/Modified:
  * SingleIntegratorRunCore.__init__() - renamed parameter evaluate_driver_at_t (line 91), updated docstring (line 63-65)
  * SingleIntegratorRunCore.__init__() body - renamed variable references: algorithm_settings["evaluate_driver_at_t"] (line 127), instantiate_loop evaluate_driver_at_t parameter (line 174)
  * SingleIntegratorRunCore.instantiate_loop() - renamed parameter evaluate_driver_at_t (line 410), updated docstring (line 436-437), updated loop_kwargs assignment (line 461)
  * SingleIntegratorRunCore.build() - renamed variables evaluate_f, evaluate_observables (lines 634-635), updated config dictionary keys (lines 638-640), updated compiled_functions dictionary (line 661)
  * SingleIntegratorRun.evaluate_f property - renamed from dxdt_function (lines 149-152)
- Implementation Summary:
  Updated SingleIntegratorRunCore and SingleIntegratorRun classes to use the new parameter naming convention. In SingleIntegratorRunCore.__init__, renamed the driver_function parameter to evaluate_driver_at_t and updated the docstring. Updated all internal references in the constructor body, including algorithm_settings dictionary assignment and the instantiate_loop call. In the instantiate_loop method, renamed the parameter and updated both the docstring and the loop_kwargs dictionary assignment. In the build method, renamed variables from dxdt_fn/observables_fn to evaluate_f/evaluate_observables and updated the compiled_fns_dict keys to use the new names (evaluate_f, evaluate_observables). In SingleIntegratorRun, renamed the dxdt_function property to evaluate_f. All changes maintain consistency with the renamed properties from BaseODE (Task Group 2) and the updated algorithm step classes (Task Groups 4-8).
- Issues Flagged: None

---

## Task Group 10: Update Batch Solver Components
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 9]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file)
- File: src/cubie/batchsolving/solver.py (entire file)

**Input Validation Required**:
- None (parameter renaming only)

**Tasks**:
1. **Update BatchSolverKernel references**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details: Search for any direct references to old function names and update
   - Edge cases: Likely minimal direct references (uses algorithm properties)
   - Integration: Top-level kernel compilation

2. **Update Solver class references**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details: Check docstrings and parameter passing for old names
   - Edge cases: None
   - Integration: Public API entry point

**Tests to Create**:
- None (existing solver tests cover functionality)

**Tests to Run**:
- tests/batchsolving/test_solver.py
- tests/batchsolving/test_BatchSolverKernel.py

**Outcomes**:

---

## Task Group 11: Update Test Fixtures and Conftest
**Status**: [ ]
**Dependencies**: Groups [1, 2]

**Required Context**:
- File: tests/conftest.py (entire file)
- File: tests/system_fixtures.py (entire file)

**Input Validation Required**:
- None (test fixture renaming only)

**Tasks**:
1. **Update conftest.py fixtures**
   - File: tests/conftest.py
   - Action: Modify
   - Details: Update any fixture parameter names or assertions referencing old names
   - Edge cases: Check indirect parameterization patterns
   - Integration: Global test fixtures used across all tests

2. **Update system_fixtures.py**
   - File: tests/system_fixtures.py
   - Action: Modify
   - Details: Update ODE system fixture creation and property access
   - Edge cases: None
   - Integration: System fixtures used by integration tests

**Tests to Create**:
- None (fixtures themselves are not tested)

**Tests to Run**:
- tests/test_conftest.py (if exists)
- tests/integrators/algorithms/test_explicit_euler.py (to verify fixtures work)

**Outcomes**:

---

## Task Group 12: Update ODE System Tests
**Status**: [ ]
**Dependencies**: Groups [1, 2, 11]

**Required Context**:
- File: tests/odesystems/test_baseODE.py (entire file)
- File: tests/odesystems/symbolic/test_symbolicODE.py (entire file)

**Input Validation Required**:
- None (test assertion renaming only)

**Tasks**:
1. **Update test_baseODE.py**
   - File: tests/odesystems/test_baseODE.py
   - Action: Modify
   - Details: Update all assertions checking property names (system.dxdt_function → system.evaluate_f)
   - Edge cases: None
   - Integration: Tests for BaseODE properties

2. **Update test_symbolicODE.py**
   - File: tests/odesystems/symbolic/test_symbolicODE.py
   - Action: Modify
   - Details: Update property access and assertions for symbolic ODE compilation
   - Edge cases: None
   - Integration: Tests for SymbolicODE device function generation

**Tests to Create**:
- None (updating existing tests)

**Tests to Run**:
- tests/odesystems/test_baseODE.py
- tests/odesystems/symbolic/test_symbolicODE.py

**Outcomes**:

---

## Task Group 13: Update Algorithm Tests
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3, 4, 5, 6, 7, 8, 11]

**Required Context**:
- File: tests/integrators/algorithms/test_explicit_euler.py (entire file)
- File: tests/integrators/algorithms/test_backwards_euler.py (entire file)
- File: tests/integrators/algorithms/test_backwards_euler_predict_correct.py (entire file)
- File: tests/integrators/algorithms/test_crank_nicolson.py (entire file)
- File: tests/integrators/algorithms/test_generic_erk.py (entire file)
- File: tests/integrators/algorithms/test_generic_dirk.py (entire file)
- File: tests/integrators/algorithms/test_generic_firk.py (entire file)
- File: tests/integrators/algorithms/test_generic_rosenbrock_w.py (entire file)

**Input Validation Required**:
- None (test renaming only)

**Tasks**:
1. **Update test_explicit_euler.py**
   - File: tests/integrators/algorithms/test_explicit_euler.py
   - Action: Modify
   - Details: Update all parameter passing in test fixtures (dxdt_function → evaluate_f, etc.)
   - Edge cases: None
   - Integration: Tests for explicit Euler algorithm

2. **Update test_backwards_euler.py**
   - File: tests/integrators/algorithms/test_backwards_euler.py
   - Action: Modify
   - Details: Apply same renaming pattern
   - Edge cases: None
   - Integration: Tests for backward Euler algorithm

3. **Update test_backwards_euler_predict_correct.py**
   - File: tests/integrators/algorithms/test_backwards_euler_predict_correct.py
   - Action: Modify
   - Details: Apply same renaming pattern
   - Edge cases: None
   - Integration: Tests for predictor-corrector variant

4. **Update test_crank_nicolson.py**
   - File: tests/integrators/algorithms/test_crank_nicolson.py
   - Action: Modify
   - Details: Apply same renaming pattern
   - Edge cases: None
   - Integration: Tests for Crank-Nicolson algorithm

5. **Update test_generic_erk.py**
   - File: tests/integrators/algorithms/test_generic_erk.py
   - Action: Modify
   - Details: Apply same renaming pattern
   - Edge cases: Multiple ERK tableaus tested
   - Integration: Tests for explicit Runge-Kutta methods

6. **Update test_generic_dirk.py**
   - File: tests/integrators/algorithms/test_generic_dirk.py
   - Action: Modify
   - Details: Apply same renaming pattern
   - Edge cases: Multiple DIRK tableaus tested
   - Integration: Tests for diagonally implicit Runge-Kutta methods

7. **Update test_generic_firk.py**
   - File: tests/integrators/algorithms/test_generic_firk.py
   - Action: Modify
   - Details: Apply same renaming pattern
   - Edge cases: Multiple FIRK tableaus tested
   - Integration: Tests for fully implicit Runge-Kutta methods

8. **Update test_generic_rosenbrock_w.py**
   - File: tests/integrators/algorithms/test_generic_rosenbrock_w.py
   - Action: Modify
   - Details: Apply same renaming pattern
   - Edge cases: Rosenbrock tableaus tested
   - Integration: Tests for Rosenbrock-W methods

**Tests to Create**:
- None (updating existing tests)

**Tests to Run**:
- tests/integrators/algorithms/

**Outcomes**:

---

## Task Group 14: Update Instrumented Algorithm Copies
**Status**: [ ]
**Dependencies**: Groups [6, 7, 8]

**Required Context**:
- File: tests/integrators/algorithms/instrumented/ (all files)
- File: src/cubie/integrators/algorithms/explicit_euler.py (reference)
- File: src/cubie/integrators/algorithms/backwards_euler.py (reference)

**Input Validation Required**:
- None (mirror source changes with logging preserved)

**Tasks**:
1. **Update instrumented explicit_euler.py**
   - File: tests/integrators/algorithms/instrumented/explicit_euler.py
   - Action: Modify
   - Details: Mirror all changes from src version, preserving logging additions
   - Edge cases: Ensure logging parameters unchanged, only function names updated
   - Integration: Instrumented version used for detailed algorithm debugging

2. **Update instrumented backwards_euler.py**
   - File: tests/integrators/algorithms/instrumented/backwards_euler.py
   - Action: Modify
   - Details: Mirror all changes from src version, preserving logging additions
   - Edge cases: Ensure logging parameters unchanged
   - Integration: Instrumented implicit solver

3. **Update all other instrumented files**
   - File: tests/integrators/algorithms/instrumented/*.py
   - Action: Modify
   - Details: For each instrumented algorithm, mirror source changes while preserving logging
   - Edge cases: None
   - Integration: Complete instrumented algorithm suite

**Tests to Create**:
- None (instrumented versions tested through specific_algos marker)

**Tests to Run**:
- pytest -m specific_algos tests/integrators/algorithms/

**Outcomes**:

---

## Task Group 15: Update Loop and Integration Tests
**Status**: [ ]
**Dependencies**: Groups [1, 2, 9, 11]

**Required Context**:
- File: tests/integrators/loops/test_ode_loop.py (entire file)
- File: tests/integrators/test_SingleIntegratorRun.py (entire file)
- File: tests/integrators/loops/test_interp_vs_symbolic.py (entire file)

**Input Validation Required**:
- None (test renaming only)

**Tasks**:
1. **Update test_ode_loop.py**
   - File: tests/integrators/loops/test_ode_loop.py
   - Action: Modify
   - Details: Update parameter passing and assertions
   - Edge cases: None
   - Integration: Tests for IVPLoop compilation

2. **Update test_SingleIntegratorRun.py**
   - File: tests/integrators/test_SingleIntegratorRun.py
   - Action: Modify
   - Details: Update parameter names in SingleIntegratorRun initialization
   - Edge cases: None
   - Integration: Tests for single integrator coordination

3. **Update test_interp_vs_symbolic.py**
   - File: tests/integrators/loops/test_interp_vs_symbolic.py
   - Action: Modify
   - Details: Update driver_function references to evaluate_driver_at_t
   - Edge cases: Test compares interpolated vs symbolic drivers
   - Integration: Driver function comparison tests

**Tests to Create**:
- None (updating existing tests)

**Tests to Run**:
- tests/integrators/loops/
- tests/integrators/test_SingleIntegratorRun.py

**Outcomes**:

---

## Task Group 16: Update Batch Solver Tests
**Status**: [ ]
**Dependencies**: Groups [1, 2, 10, 11]

**Required Context**:
- File: tests/batchsolving/test_solver.py (entire file)
- File: tests/batchsolving/test_BatchSolverKernel.py (entire file)

**Input Validation Required**:
- None (test renaming only)

**Tasks**:
1. **Update test_solver.py**
   - File: tests/batchsolving/test_solver.py
   - Action: Modify
   - Details: Update any property access or parameter passing using old names
   - Edge cases: None
   - Integration: Tests for Solver public API

2. **Update test_BatchSolverKernel.py**
   - File: tests/batchsolving/test_BatchSolverKernel.py
   - Action: Modify
   - Details: Update kernel compilation tests
   - Edge cases: None
   - Integration: Tests for batch kernel factory

**Tests to Create**:
- None (updating existing tests)

**Tests to Run**:
- tests/batchsolving/

**Outcomes**:

---

## Task Group 17: Update CPU Reference Implementations
**Status**: [ ]
**Dependencies**: Groups [1, 2]

**Required Context**:
- File: tests/integrators/cpu_reference.py (entire file)

**Input Validation Required**:
- None (parameter renaming in reference code)

**Tasks**:
1. **Update cpu_reference.py function signatures**
   - File: tests/integrators/cpu_reference.py
   - Action: Modify
   - Details: Update parameter names in CPUODESystem and reference loop functions
   - Edge cases: None
   - Integration: Reference implementations for validation against GPU results

**Tests to Create**:
- None (reference code used by other tests)

**Tests to Run**:
- tests/integrators/test_cpu_reference.py (if exists)

**Outcomes**:

---

## Task Group 18: Update Documentation Files
**Status**: [ ]
**Dependencies**: Groups [1, 2]

**Required Context**:
- File: .github/context/cubie_internal_structure.md (entire file)
- File: docs/source/API_reference/ (relevant RST files)

**Input Validation Required**:
- None (documentation updates only)

**Tasks**:
1. **Update cubie_internal_structure.md**
   - File: .github/context/cubie_internal_structure.md
   - Action: Modify
   - Details: Update all references to device function names in architectural documentation
   - Edge cases: None
   - Integration: Internal architecture reference for agents

2. **Update API reference RST files**
   - File: docs/source/API_reference/*.rst
   - Action: Modify
   - Details: Update function name references in API documentation
   - Edge cases: Check code examples and cross-references
   - Integration: User-facing API documentation

3. **Update README.md if needed**
   - File: README.md
   - Action: Modify (if references exist)
   - Details: Update any code examples showing old function names
   - Edge cases: None
   - Integration: Repository overview

**Tests to Create**:
- None (documentation changes)

**Tests to Run**:
- None (documentation not tested)

**Outcomes**:

---

## Task Group 19: Update CHANGELOG
**Status**: [ ]
**Dependencies**: Groups [1-18]

**Required Context**:
- File: CHANGELOG.md (top of file)

**Input Validation Required**:
- None (changelog entry only)

**Tasks**:
1. **Add breaking change entry to CHANGELOG**
   - File: CHANGELOG.md
   - Action: Modify
   - Details:
     ```markdown
     ## [Unreleased] - Breaking Changes
     
     ### Changed
     - **BREAKING**: Renamed device function references for improved clarity
       - `dxdt_function` → `evaluate_f`
       - `observables_function` → `evaluate_observables`
       - `driver_function` → `evaluate_driver_at_t`
       - Updated all algorithm steps, ODE systems, and integration loops
       - This is an internal API change affecting BaseODE properties and 
         algorithm step parameters
     ```
   - Edge cases: None
   - Integration: Changelog tracks breaking changes

**Tests to Create**:
- None (changelog entry)

**Tests to Run**:
- None

**Outcomes**:

---

## Summary

**Total Task Groups**: 19

**Dependency Chain Overview**:
1. Constants (Group 1) → Base classes (Groups 2, 3) → Algorithm bases (Groups 4, 5)
2. Algorithm implementations (Groups 6, 7, 8) → Integration (Group 9) → Batch (Group 10)
3. Test fixtures (Group 11) → All test groups (Groups 12-17)
4. Instrumented copies (Group 14) depend on source algorithm changes (Groups 6, 7, 8)
5. Documentation (Group 18) can run in parallel after Groups 1-2
6. CHANGELOG (Group 19) waits for all implementation groups

**Parallel Work Opportunities**:
- After Groups 1-3: Groups 4 and 5 can run in parallel (explicit vs implicit bases)
- After Groups 4-5: Groups 6, 7, 8 can run in parallel (different algorithm families)
- After Group 11: Test groups 12-17 can run in parallel
- Group 18 can start after Groups 1-2

**Tests to Create**: 0 (all tests already exist, will be updated)

**Estimated Scope**:
- Source files: ~20 files
- Test files: ~25 files
- Documentation files: ~5 files
- Total files: ~50 files
- Total references: 750+ (as identified in plan)

**Complexity**: Medium-Low
- Pure renaming operation with no logic changes
- Systematic pattern applied consistently across codebase
- Main risk is missing edge cases in string literals or comments
- All changes are breaking but acceptable per project guidelines
