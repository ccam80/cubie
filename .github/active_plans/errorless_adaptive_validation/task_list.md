# Implementation Task List
# Feature: Errorless Adaptive Validation
# Plan Reference: .github/active_plans/errorless_adaptive_validation/agent_plan.md

## Task Group 1: Enhanced Compatibility Error Messages - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/SingleIntegratorRunCore.py (lines 169-184)
- File: src/cubie/integrators/IntegratorRunSettings.py (entire file)

**Input Validation Required**:
- None (this task only modifies error messages, no new validation logic)

**Tasks**:

1. **Enhance check_compatibility error message**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     def check_compatibility(self) -> None:
         """Validate that algorithm and controller step modes are aligned.

         Raises
         ------
         ValueError
             Raised when an adaptive controller is paired with a fixed-step
             algorithm.
         """
         
         if (not self._algo_step.is_adaptive and
                 self._step_controller.is_adaptive):
             # Extract algorithm and controller names from settings
             algo_name = self.compile_settings.algorithm
             controller_name = self.compile_settings.step_controller
             
             raise ValueError(
                 f"Adaptive step controller '{controller_name}' cannot be "
                 f"used with fixed-step algorithm '{algo_name}'. "
                 f"The algorithm does not provide an error estimate "
                 f"required for adaptive stepping. "
                 f"Use step_controller='fixed' or choose an adaptive "
                 f"algorithm with error estimation."
             )
     ```
   - Edge cases:
     - Algorithm/controller names must exist in compile_settings
     - Error message must be formatted properly (79 char limit)
   - Integration:
     - Uses existing `self.compile_settings` (IntegratorRunSettings)
     - Accesses `algorithm` and `step_controller` string fields
     - No changes to conditional logic, only error message content

**Outcomes**:
- Enhanced error message in check_compatibility() method to include algorithm name and controller name extracted from compile_settings
- Error message now provides specific details about the incompatibility and suggests solutions (use step_controller='fixed' or choose adaptive algorithm)
- Error message fits within 79 character line limit per PEP8

---

## Task Group 2: Enable Compatibility Validation - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/integrators/SingleIntegratorRunCore.py (lines 85-160)

**Input Validation Required**:
- None (validation call itself, not input validation)

**Tasks**:

1. **Add check_compatibility call in __init__**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     # After line 128 where controller is instantiated:
     self._step_controller = get_controller(
         precision=precision,
         settings=controller_settings,
     )
     
     # Add validation call here (NEW):
     self.check_compatibility()
     
     # Before line 130 (loop_settings updates):
     loop_settings["dt0"] = self._step_controller.dt0
     ```
   - Implementation logic:
     1. Locate line 128 (end of controller instantiation)
     2. Insert blank line
     3. Insert `self.check_compatibility()`
     4. Continue with existing loop_settings updates
   - Edge cases:
     - Must be called after both `_algo_step` and `_step_controller` exist
     - Must be called before `_loop` instantiation
     - If validation fails, __init__ should not complete
   - Integration:
     - Validation occurs before CUDA compilation (loop instantiation)
     - Error propagates to caller (Solver or solve_ivp)
     - No try/except needed; let ValueError propagate

**Outcomes**:
- Added self.check_compatibility() call in __init__ method after controller instantiation (line 130)
- Validation occurs after both _algo_step and _step_controller are created
- Validation occurs before _loop instantiation to prevent unnecessary CUDA compilation
- ValueError propagates to caller as expected

---

## Task Group 3: Dynamic ERK Controller Defaults - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_erk.py (lines 1-86)
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (lines 96-101, 166-175)
- File: src/cubie/integrators/algorithms/generic_erk_tableaus.py (for DEFAULT_ERK_TABLEAU)

**Input Validation Required**:
- None (uses existing tableau validation)

**Tasks**:

1. **Define separate adaptive and fixed defaults**
   - File: src/cubie/integrators/algorithms/generic_erk.py
   - Action: Modify
   - Details:
     ```python
     # Replace ERK_DEFAULTS (line 24-36) with two separate constants:
     
     ERK_ADAPTIVE_DEFAULTS = StepControlDefaults(
         step_controller={
             "step_controller": "pi",
             "dt_min": 1e-6,
             "dt_max": 1e-1,
             "kp": 0.6,
             "kd": 0.4,
             "deadband_min": 1.0,
             "deadband_max": 1.1,
             "min_gain": 0.5,
             "max_gain": 2.0,
         }
     )
     
     ERK_FIXED_DEFAULTS = StepControlDefaults(
         step_controller={
             "step_controller": "fixed",
             "dt": 1e-3,
         }
     )
     ```
   - Edge cases:
     - Adaptive defaults identical to current ERK_DEFAULTS
     - Fixed defaults match pattern from EE_DEFAULTS
   - Integration:
     - Uses existing StepControlDefaults class
     - No import changes needed

2. **Add dynamic defaults selection in ERKStep.__init__**
   - File: src/cubie/integrators/algorithms/generic_erk.py
   - Action: Modify
   - Details:
     ```python
     def __init__(
         self,
         precision: PrecisionDType,
         n: int,
         dt: Optional[float],
         dxdt_function: Optional[Callable] = None,
         observables_function: Optional[Callable] = None,
         driver_function: Optional[Callable] = None,
         get_solver_helper_fn: Optional[Callable] = None,
         tableau: ERKTableau = DEFAULT_ERK_TABLEAU,
         n_drivers: int = 0,
     ) -> None:
         """Initialise the Runge--Kutta step configuration.

         Parameters
         ----------
         tableau
             Explicit Runge--Kutta tableau describing the coefficients used
             by the integrator. Defaults to :data:`DEFAULT_ERK_TABLEAU`.
         """
         
         config = ERKStepConfig(
             precision=precision,
             n=n,
             n_drivers=n_drivers,
             dt=dt,
             dxdt_function=dxdt_function,
             observables_function=observables_function,
             driver_function=driver_function,
             get_solver_helper_fn=get_solver_helper_fn,
             tableau=tableau,
         )
         
         # Select defaults based on tableau capability (NEW):
         if tableau.has_error_estimate:
             defaults = ERK_ADAPTIVE_DEFAULTS
         else:
             defaults = ERK_FIXED_DEFAULTS
         
         super().__init__(config, defaults)
     ```
   - Implementation logic:
     1. Create config object (unchanged)
     2. Check `tableau.has_error_estimate` property
     3. Select appropriate defaults
     4. Pass to super().__init__
   - Edge cases:
     - DEFAULT_ERK_TABLEAU (Dormand-Prince) has error estimate → adaptive
     - CLASSICAL_RK4_TABLEAU has no error estimate → fixed
     - Custom tableaus handled by has_error_estimate property
   - Integration:
     - Uses existing `tableau.has_error_estimate` property
     - No changes to ERKStepConfig
     - No changes to build_step method

**Outcomes**:
- Replaced ERK_DEFAULTS with two separate constants: ERK_ADAPTIVE_DEFAULTS and ERK_FIXED_DEFAULTS
- Added dynamic defaults selection in ERKStep.__init__ based on tableau.has_error_estimate property
- Adaptive tableaus (like Dormand-Prince) now default to PI controller
- Errorless tableaus (like Classical RK4) now default to fixed controller
- No changes to ERKStepConfig or build_step method

---

## Task Group 4: Dynamic DIRK Controller Defaults - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 3

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 1-100)
- File: src/cubie/integrators/algorithms/generic_dirk_tableaus.py (for DEFAULT_DIRK_TABLEAU)

**Input Validation Required**:
- None (uses existing tableau validation)

**Tasks**:

1. **Define separate adaptive and fixed defaults for DIRK**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     ```python
     # Replace DIRK_DEFAULTS (line 29-41) with two separate constants:
     
     DIRK_ADAPTIVE_DEFAULTS = StepControlDefaults(
         step_controller={
             "step_controller": "pi",
             "dt_min": 1e-6,
             "dt_max": 1e-1,
             "kp": 0.6,
             "kd": 0.4,
             "deadband_min": 1.0,
             "deadband_max": 1.1,
             "min_gain": 0.5,
             "max_gain": 2.0,
         }
     )
     
     DIRK_FIXED_DEFAULTS = StepControlDefaults(
         step_controller={
             "step_controller": "fixed",
             "dt": 1e-3,
         }
     )
     ```
   - Edge cases:
     - Adaptive defaults identical to current DIRK_DEFAULTS
     - Fixed defaults consistent with other fixed algorithms
   - Integration:
     - Uses existing StepControlDefaults class

2. **Add dynamic defaults selection in DIRKStep.__init__**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details:
     ```python
     # In DIRKStep.__init__, after config creation (line ~98):
     config = DIRKStepConfig(
         # ... all existing parameters ...
     )
     self._cached_auxiliary_count = 0
     
     # Select defaults based on tableau capability (NEW):
     if tableau.has_error_estimate:
         defaults = DIRK_ADAPTIVE_DEFAULTS
     else:
         defaults = DIRK_FIXED_DEFAULTS
     
     super().__init__(config, defaults)
     ```
   - Implementation logic:
     1. Create config object (unchanged)
     2. Set _cached_auxiliary_count (unchanged)
     3. Check `tableau.has_error_estimate` property
     4. Select appropriate defaults
     5. Pass to super().__init__
   - Edge cases:
     - DEFAULT_DIRK_TABLEAU should have error estimate (check tableaus file)
     - Custom tableaus handled by has_error_estimate property
   - Integration:
     - Uses inherited `has_error_estimate` from ButcherTableau
     - No changes to DIRKStepConfig

**Outcomes**:
- Replaced DIRK_DEFAULTS with two separate constants: DIRK_ADAPTIVE_DEFAULTS and DIRK_FIXED_DEFAULTS
- Added dynamic defaults selection in DIRKStep.__init__ based on tableau.has_error_estimate property
- Adaptive tableaus now default to PI controller
- Errorless tableaus now default to fixed controller
- No changes to DIRKStepConfig

---

## Task Group 5: Dynamic FIRK Controller Defaults - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 4

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (lines 1-100)
- File: src/cubie/integrators/algorithms/generic_firk_tableaus.py (for DEFAULT_FIRK_TABLEAU)

**Input Validation Required**:
- None (uses existing tableau validation)

**Tasks**:

1. **Define separate adaptive and fixed defaults for FIRK**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     # Replace FIRK_DEFAULTS (line 28-40) with two separate constants:
     
     FIRK_ADAPTIVE_DEFAULTS = StepControlDefaults(
         step_controller={
             "step_controller": "pi",
             "dt_min": 1e-6,
             "dt_max": 1e-1,
             "kp": 0.6,
             "kd": 0.4,
             "deadband_min": 1.0,
             "deadband_max": 1.1,
             "min_gain": 0.5,
             "max_gain": 2.0,
         }
     )
     
     FIRK_FIXED_DEFAULTS = StepControlDefaults(
         step_controller={
             "step_controller": "fixed",
             "dt": 1e-3,
         }
     )
     ```
   - Edge cases:
     - Adaptive defaults identical to current FIRK_DEFAULTS
     - Fixed defaults consistent with other fixed algorithms
   - Integration:
     - Uses existing StepControlDefaults class

2. **Add dynamic defaults selection in FIRKStep.__init__**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details:
     ```python
     # In FIRKStep.__init__, after config creation:
     config = FIRKStepConfig(
         # ... all existing parameters ...
     )
     self._cached_auxiliary_count = None
     
     # Select defaults based on tableau capability (NEW):
     if tableau.has_error_estimate:
         defaults = FIRK_ADAPTIVE_DEFAULTS
     else:
         defaults = FIRK_FIXED_DEFAULTS
     
     super().__init__(config, defaults)
     ```
   - Implementation logic:
     1. Create config object (unchanged)
     2. Set _cached_auxiliary_count (unchanged)
     3. Check `tableau.has_error_estimate` property
     4. Select appropriate defaults
     5. Pass to super().__init__
   - Edge cases:
     - DEFAULT_FIRK_TABLEAU should have error estimate
     - Custom tableaus handled by has_error_estimate property
   - Integration:
     - Uses inherited `has_error_estimate` from ButcherTableau

**Outcomes**:
- Replaced FIRK_DEFAULTS with two separate constants: FIRK_ADAPTIVE_DEFAULTS and FIRK_FIXED_DEFAULTS
- Added dynamic defaults selection in FIRKStep.__init__ based on tableau.has_error_estimate property
- Adaptive tableaus now default to PI controller
- Errorless tableaus now default to fixed controller
- No changes to FIRKStepConfig

---

## Task Group 6: Dynamic Rosenbrock Controller Defaults - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 5

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (lines 1-95)
- File: src/cubie/integrators/algorithms/generic_rosenbrockw_tableaus.py (for DEFAULT_ROSENBROCK_TABLEAU)

**Input Validation Required**:
- None (uses existing tableau validation)

**Tasks**:

1. **Define separate adaptive and fixed defaults for Rosenbrock**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     # Replace ROSENBROCK_DEFAULTS (line 29-41) with:
     
     ROSENBROCK_ADAPTIVE_DEFAULTS = StepControlDefaults(
         step_controller={
             "step_controller": "pi",
             "dt_min": 1e-6,
             "dt_max": 1e-1,
             "kp": 0.6,
             "kd": 0.4,
             "deadband_min": 1.0,
             "deadband_max": 1.1,
             "min_gain": 0.5,
             "max_gain": 2.0,
         }
     )
     
     ROSENBROCK_FIXED_DEFAULTS = StepControlDefaults(
         step_controller={
             "step_controller": "fixed",
             "dt": 1e-3,
         }
     )
     ```
   - Edge cases:
     - Adaptive defaults identical to current ROSENBROCK_DEFAULTS
     - Fixed defaults consistent with other fixed algorithms
   - Integration:
     - Uses existing StepControlDefaults class

2. **Add dynamic defaults selection in GenericRosenbrockWStep.__init__**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details:
     ```python
     # In GenericRosenbrockWStep.__init__, after config creation (line ~93):
     config = RosenbrockWStepConfig(
         # ... all existing parameters ...
     )
     self._cached_auxiliary_count = None
     
     # Select defaults based on tableau capability (NEW):
     if tableau.has_error_estimate:
         defaults = ROSENBROCK_ADAPTIVE_DEFAULTS
     else:
         defaults = ROSENBROCK_FIXED_DEFAULTS
     
     super().__init__(config, defaults)
     ```
   - Implementation logic:
     1. Create config object (unchanged)
     2. Set _cached_auxiliary_count (unchanged)
     3. Check `tableau.has_error_estimate` property
     4. Select appropriate defaults
     5. Pass to super().__init__
   - Edge cases:
     - DEFAULT_ROSENBROCK_TABLEAU should have error estimate
     - RosenbrockTableau inherits from ButcherTableau
   - Integration:
     - Uses inherited `has_error_estimate` from ButcherTableau

**Outcomes**:
- Replaced ROSENBROCK_DEFAULTS with two separate constants: ROSENBROCK_ADAPTIVE_DEFAULTS and ROSENBROCK_FIXED_DEFAULTS
- Added dynamic defaults selection in GenericRosenbrockWStep.__init__ based on tableau.has_error_estimate property
- Adaptive tableaus now default to PI controller
- Errorless tableaus now default to fixed controller
- No changes to RosenbrockWStepConfig

---

## Task Group 7: Test Coverage for Incompatibility Detection - PARALLEL
**Status**: [x]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: tests/conftest.py (entire file - for fixture patterns)
- File: tests/system_fixtures.py (entire file - for system fixtures)
- File: src/cubie/integrators/SingleIntegratorRunCore.py (entire file)
- File: src/cubie/integrators/algorithms/explicit_euler.py (entire file)
- File: src/cubie/integrators/algorithms/generic_erk.py (lines 1-86)
- File: src/cubie/integrators/algorithms/generic_erk_tableaus.py (for CLASSICAL_RK4_TABLEAU)

**Input Validation Required**:
- Test assertions validate error messages contain expected strings
- No production code input validation

**Tasks**:

1. **Create test file for algorithm-controller compatibility**
   - File: tests/integrators/test_algorithm_controller_compatibility.py
   - Action: Create
   - Details:
     ```python
     """Test algorithm-controller compatibility validation."""
     
     import pytest
     import numpy as np
     
     from cubie.integrators.SingleIntegratorRunCore import (
         SingleIntegratorRunCore
     )
     from cubie.integrators.algorithms.generic_erk_tableaus import (
         CLASSICAL_RK4_TABLEAU,
         DORMAND_PRINCE_54_TABLEAU,
     )
     
     
     def test_errorless_euler_with_adaptive_raises(three_state_linear):
         """Errorless explicit Euler with adaptive PI raises ValueError."""
         
         algorithm_settings = {
             "algorithm": "explicit_euler",
         }
         step_control_settings = {
             "step_controller": "pi",
             "dt_min": 1e-6,
             "dt_max": 1e-1,
         }
         
         with pytest.raises(ValueError) as exc_info:
             SingleIntegratorRunCore(
                 system=three_state_linear,
                 algorithm_settings=algorithm_settings,
                 step_control_settings=step_control_settings,
             )
         
         error_msg = str(exc_info.value).lower()
         assert "explicit_euler" in error_msg
         assert "pi" in error_msg
         assert "adaptive" in error_msg
         assert "fixed" in error_msg or "error estimate" in error_msg
     
     
     def test_errorless_rk4_tableau_with_adaptive_raises(
         three_state_linear
     ):
         """Errorless RK4 tableau with adaptive PI raises ValueError."""
         
         algorithm_settings = {
             "algorithm": "erk",
             "tableau": CLASSICAL_RK4_TABLEAU,
         }
         step_control_settings = {
             "step_controller": "pi",
             "dt_min": 1e-6,
             "dt_max": 1e-1,
         }
         
         with pytest.raises(ValueError) as exc_info:
             SingleIntegratorRunCore(
                 system=three_state_linear,
                 algorithm_settings=algorithm_settings,
                 step_control_settings=step_control_settings,
             )
         
         error_msg = str(exc_info.value).lower()
         assert "erk" in error_msg
         assert "pi" in error_msg
     
     
     def test_adaptive_tableau_with_adaptive_succeeds(
         three_state_linear
     ):
         """Adaptive Dormand-Prince with PI controller succeeds."""
         
         algorithm_settings = {
             "algorithm": "erk",
             "tableau": DORMAND_PRINCE_54_TABLEAU,
         }
         step_control_settings = {
             "step_controller": "pi",
             "dt_min": 1e-6,
             "dt_max": 1e-1,
         }
         
         # Should not raise
         core = SingleIntegratorRunCore(
             system=three_state_linear,
             algorithm_settings=algorithm_settings,
             step_control_settings=step_control_settings,
         )
         
         assert core._algo_step.is_adaptive
         assert core._step_controller.is_adaptive
     
     
     def test_errorless_euler_with_fixed_succeeds(three_state_linear):
         """Errorless explicit Euler with fixed controller succeeds."""
         
         algorithm_settings = {
             "algorithm": "explicit_euler",
         }
         step_control_settings = {
             "step_controller": "fixed",
             "dt": 1e-3,
         }
         
         # Should not raise
         core = SingleIntegratorRunCore(
             system=three_state_linear,
             algorithm_settings=algorithm_settings,
             step_control_settings=step_control_settings,
         )
         
         assert not core._algo_step.is_adaptive
         assert not core._step_controller.is_adaptive
     
     
     def test_error_message_contains_algorithm_and_controller(
         three_state_linear
     ):
         """Error message includes algorithm and controller names."""
         
         algorithm_settings = {
             "algorithm": "explicit_euler",
         }
         step_control_settings = {
             "step_controller": "pid",
             "dt_min": 1e-6,
             "dt_max": 1e-1,
         }
         
         with pytest.raises(ValueError) as exc_info:
             SingleIntegratorRunCore(
                 system=three_state_linear,
                 algorithm_settings=algorithm_settings,
                 step_control_settings=step_control_settings,
             )
         
         error_msg = str(exc_info.value)
         # Check for both algorithm and controller names
         assert "explicit_euler" in error_msg
         assert "pid" in error_msg
         # Check for explanation
         assert "error estimate" in error_msg.lower()
         # Check for suggestion
         assert "fixed" in error_msg.lower()
     ```
   - Implementation logic:
     1. Import required modules
     2. Use three_state_linear fixture from conftest
     3. Create settings dictionaries
     4. Assert ValueError raised with correct content
   - Edge cases:
     - Multiple adaptive controllers tested (pi, pid)
     - Both direct algorithms and tableau-based tested
     - Success cases validate no exception raised
   - Integration:
     - Uses existing fixtures from conftest.py
     - Uses existing algorithm and controller infrastructure
     - Tests actual integration, not mocks

**Outcomes**:
- Created tests/integrators/test_algorithm_controller_compatibility.py with 5 test cases
- test_errorless_euler_with_adaptive_raises: validates explicit_euler + PI raises error
- test_errorless_rk4_tableau_with_adaptive_raises: validates RK4 + PI raises error
- test_adaptive_tableau_with_adaptive_succeeds: validates Dormand-Prince + PI succeeds
- test_errorless_euler_with_fixed_succeeds: validates explicit_euler + fixed succeeds
- test_error_message_contains_algorithm_and_controller: validates enhanced error message content
- All tests use real fixtures and actual integration code, no mocks

---

## Task Group 8: Test Coverage for Dynamic Defaults - PARALLEL
**Status**: [x]
**Dependencies**: Groups 3, 4, 5, 6

**Required Context**:
- File: tests/conftest.py (entire file)
- File: tests/system_fixtures.py (entire file)
- File: src/cubie/integrators/algorithms/generic_erk.py (modified version)
- File: src/cubie/integrators/algorithms/generic_erk_tableaus.py (for tableaus)

**Input Validation Required**:
- Test assertions validate defaults are correctly selected
- No production code input validation

**Tasks**:

1. **Add tests for ERK dynamic defaults**
   - File: tests/integrators/algorithms/test_generic_erk.py
   - Action: Modify (if exists) or Create
   - Details:
     ```python
     # Add to existing test file or create new one:
     
     import pytest
     import numpy as np
     
     from cubie.integrators.algorithms.generic_erk import ERKStep
     from cubie.integrators.algorithms.generic_erk_tableaus import (
         CLASSICAL_RK4_TABLEAU,
         DORMAND_PRINCE_54_TABLEAU,
         HEUN_21_TABLEAU,
     )
     
     
     def test_erk_errorless_tableau_defaults_to_fixed():
         """ERK with errorless tableau defaults to fixed controller."""
         
         step = ERKStep(
             precision=np.float32,
             n=3,
             dt=None,
             tableau=CLASSICAL_RK4_TABLEAU,
         )
         
         defaults = step.controller_defaults.step_controller
         assert defaults["step_controller"] == "fixed"
         assert "dt" in defaults
     
     
     def test_erk_adaptive_tableau_defaults_to_adaptive():
         """ERK with adaptive tableau defaults to adaptive controller."""
         
         step = ERKStep(
             precision=np.float32,
             n=3,
             dt=None,
             tableau=DORMAND_PRINCE_54_TABLEAU,
         )
         
         defaults = step.controller_defaults.step_controller
         assert defaults["step_controller"] == "pi"
         assert "dt_min" in defaults
         assert "dt_max" in defaults
     
     
     def test_erk_heun_tableau_defaults_to_fixed():
         """ERK with Heun tableau (errorless) defaults to fixed."""
         
         step = ERKStep(
             precision=np.float32,
             n=3,
             dt=None,
             tableau=HEUN_21_TABLEAU,
         )
         
         defaults = step.controller_defaults.step_controller
         assert defaults["step_controller"] == "fixed"
     
     
     def test_erk_default_tableau_defaults_to_adaptive():
         """ERK with default tableau defaults to adaptive controller."""
         
         # DEFAULT_ERK_TABLEAU is Dormand-Prince which has error estimate
         step = ERKStep(
             precision=np.float32,
             n=3,
             dt=None,
         )
         
         defaults = step.controller_defaults.step_controller
         assert defaults["step_controller"] == "pi"
     ```
   - Implementation logic:
     1. Instantiate ERKStep with various tableaus
     2. Access controller_defaults property
     3. Verify step_controller matches expected value
   - Edge cases:
     - Default tableau (Dormand-Prince) → adaptive
     - Classical RK4 → fixed
     - Heun's method → fixed (if errorless)
     - Custom tableaus follow has_error_estimate
   - Integration:
     - Tests actual ERKStep behavior
     - Uses real tableau objects
     - No mocking

2. **Add tests for DIRK dynamic defaults**
   - File: tests/integrators/algorithms/test_generic_dirk.py
   - Action: Modify (if exists) or Create
   - Details:
     ```python
     # Similar pattern to ERK tests:
     
     import pytest
     import numpy as np
     
     from cubie.integrators.algorithms.generic_dirk import DIRKStep
     from cubie.integrators.algorithms.generic_dirk_tableaus import (
         DEFAULT_DIRK_TABLEAU,
     )
     
     
     def test_dirk_default_tableau_has_appropriate_defaults(
         three_state_linear
     ):
         """DIRK default tableau selects appropriate controller defaults."""
         
         # Check if default tableau has error estimate
         has_error = DEFAULT_DIRK_TABLEAU.has_error_estimate
         
         step = DIRKStep(
             precision=np.float32,
             n=3,
             dt=None,
             get_solver_helper_fn=three_state_linear.solver_helper,
         )
         
         defaults = step.controller_defaults.step_controller
         
         if has_error:
             assert defaults["step_controller"] == "pi"
         else:
             assert defaults["step_controller"] == "fixed"
     ```
   - Implementation logic:
     1. Check default tableau capability
     2. Instantiate DIRKStep
     3. Verify defaults match tableau capability
   - Edge cases:
     - Default DIRK tableau may or may not have error estimate
     - Test adapts based on actual tableau properties
   - Integration:
     - Uses solver_helper from system fixture
     - Tests actual DIRK behavior

3. **Add tests for FIRK dynamic defaults**
   - File: tests/integrators/algorithms/test_generic_firk.py
   - Action: Modify (if exists) or Create
   - Details:
     ```python
     # Similar pattern to DIRK:
     
     import numpy as np
     
     from cubie.integrators.algorithms.generic_firk import FIRKStep
     from cubie.integrators.algorithms.generic_firk_tableaus import (
         DEFAULT_FIRK_TABLEAU,
     )
     
     
     def test_firk_default_tableau_has_appropriate_defaults(
         three_state_linear
     ):
         """FIRK default tableau selects appropriate controller defaults."""
         
         has_error = DEFAULT_FIRK_TABLEAU.has_error_estimate
         
         step = FIRKStep(
             precision=np.float32,
             n=3,
             dt=None,
             get_solver_helper_fn=three_state_linear.solver_helper,
         )
         
         defaults = step.controller_defaults.step_controller
         
         if has_error:
             assert defaults["step_controller"] == "pi"
         else:
             assert defaults["step_controller"] == "fixed"
     ```
   - Implementation logic:
     1. Check default tableau capability
     2. Instantiate FIRKStep
     3. Verify defaults match tableau capability
   - Edge cases:
     - Similar to DIRK pattern
   - Integration:
     - Uses solver_helper from system fixture

4. **Add tests for Rosenbrock dynamic defaults**
   - File: tests/integrators/algorithms/test_generic_rosenbrock_w.py
   - Action: Modify (if exists) or Create
   - Details:
     ```python
     # Similar pattern:
     
     import numpy as np
     
     from cubie.integrators.algorithms.generic_rosenbrock_w import (
         GenericRosenbrockWStep
     )
     from cubie.integrators.algorithms.generic_rosenbrockw_tableaus import (
         DEFAULT_ROSENBROCK_TABLEAU,
     )
     
     
     def test_rosenbrock_default_tableau_has_appropriate_defaults(
         three_state_linear
     ):
         """Rosenbrock default tableau selects appropriate defaults."""
         
         has_error = DEFAULT_ROSENBROCK_TABLEAU.has_error_estimate
         
         step = GenericRosenbrockWStep(
             precision=np.float32,
             n=3,
             dt=None,
             get_solver_helper_fn=three_state_linear.solver_helper,
         )
         
         defaults = step.controller_defaults.step_controller
         
         if has_error:
             assert defaults["step_controller"] == "pi"
         else:
             assert defaults["step_controller"] == "fixed"
     ```
   - Implementation logic:
     1. Check default tableau capability
     2. Instantiate GenericRosenbrockWStep
     3. Verify defaults match tableau capability
   - Edge cases:
     - Similar to other tableau-based methods
   - Integration:
     - Uses solver_helper from system fixture

**Outcomes**:
- Created tests/integrators/algorithms/test_generic_erk.py with 4 test cases for ERK defaults
- Created tests/integrators/algorithms/test_generic_dirk.py with 1 test case for DIRK defaults
- Created tests/integrators/algorithms/test_generic_firk.py with 1 test case for FIRK defaults
- Created tests/integrators/algorithms/test_generic_rosenbrock_w.py with 1 test case for Rosenbrock defaults
- ERK tests validate: CLASSICAL_RK4 → fixed, DORMAND_PRINCE → adaptive, HEUN_21 → fixed, default → adaptive
- DIRK/FIRK/Rosenbrock tests validate defaults based on actual tableau capabilities
- All tests use real algorithm objects and actual tableaus, no mocks

---

## Task Group 9: Integration Test with Solver API - PARALLEL
**Status**: [x]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: tests/conftest.py (entire file)
- File: tests/system_fixtures.py (entire file)
- File: src/cubie/batchsolving/solver.py (Solver class, solve_ivp function)

**Input Validation Required**:
- Test assertions validate errors propagate through API
- No production code input validation

**Tasks**:

1. **Create integration tests for Solver API**
   - File: tests/batchsolving/test_solver_validation.py
   - Action: Create
   - Details:
     ```python
     """Integration tests for solver-level validation."""
     
     import pytest
     import numpy as np
     
     from cubie import Solver, solve_ivp
     from cubie.integrators.algorithms.generic_erk_tableaus import (
         CLASSICAL_RK4_TABLEAU,
     )
     
     
     def test_solver_with_incompatible_config_raises(three_state_linear):
         """Solver raises ValueError for incompatible configuration."""
         
         with pytest.raises(ValueError) as exc_info:
             Solver(
                 system=three_state_linear,
                 algorithm="explicit_euler",
                 step_controller="pi",
                 dt_min=1e-6,
                 dt_max=1e-1,
             )
         
         error_msg = str(exc_info.value).lower()
         assert "explicit_euler" in error_msg
         assert "pi" in error_msg
     
     
     def test_solve_ivp_with_incompatible_config_raises(
         three_state_linear
     ):
         """solve_ivp raises ValueError for incompatible configuration."""
         
         initial_values = np.array([1.0, 0.0, 0.0], dtype=np.float32)
         parameters = np.array([1.0], dtype=np.float32)
         
         with pytest.raises(ValueError) as exc_info:
             solve_ivp(
                 system=three_state_linear,
                 initial_values=initial_values,
                 parameters=parameters,
                 t_span=(0.0, 10.0),
                 algorithm="erk",
                 tableau=CLASSICAL_RK4_TABLEAU,
                 step_controller="pi",
                 dt_min=1e-6,
                 dt_max=1e-1,
             )
         
         error_msg = str(exc_info.value).lower()
         assert "erk" in error_msg
         assert "pi" in error_msg
     
     
     def test_solver_with_compatible_config_succeeds(
         three_state_linear
     ):
         """Solver succeeds with compatible configuration."""
         
         # Should not raise
         solver = Solver(
             system=three_state_linear,
             algorithm="explicit_euler",
             step_controller="fixed",
             dt=1e-3,
         )
         
         assert solver is not None
     ```
   - Implementation logic:
     1. Import Solver and solve_ivp
     2. Test incompatible configurations raise ValueError
     3. Test compatible configurations succeed
     4. Verify error messages propagate correctly
   - Edge cases:
     - Both Solver and solve_ivp tested
     - Multiple incompatible combinations tested
     - Success cases verify no exception
   - Integration:
     - Tests full API surface
     - Uses real system fixtures
     - No mocking

**Outcomes**:
- Created tests/batchsolving/test_solver_validation.py with 3 test cases
- test_solver_with_incompatible_config_raises: validates Solver API raises error for incompatible config
- test_solve_ivp_with_incompatible_config_raises: validates solve_ivp raises error for incompatible config
- test_solver_with_compatible_config_succeeds: validates Solver API succeeds with compatible config
- All tests use real system fixtures and test full API integration
- No mocking of internal components

---

## Summary

**Total Task Groups**: 9
- Sequential groups: 6 (Groups 1-6)
- Parallel groups: 3 (Groups 7-9)

**Dependency Chain**:
```
Group 1 (Error Messages)
    ↓
Group 2 (Enable Validation) ────────────┬─→ Group 7 (Compat Tests)
    ↓                                    └─→ Group 9 (Integration Tests)
Group 3 (ERK Defaults)
    ↓
Group 4 (DIRK Defaults)
    ↓
Group 5 (FIRK Defaults)
    ↓
Group 6 (Rosenbrock Defaults)
    ↓
Group 8 (Defaults Tests)
```

**Parallel Execution Opportunities**:
- Groups 7, 8, 9 can run in parallel after their dependencies complete
- Groups 3-6 must run sequentially (same pattern repeated)

**Estimated Complexity**:
- **Low complexity**: Groups 1, 2 (simple modifications)
- **Medium complexity**: Groups 3-6 (repetitive pattern)
- **Medium complexity**: Groups 7-9 (test creation following patterns)

**Files Modified**: 7
- src/cubie/integrators/SingleIntegratorRunCore.py
- src/cubie/integrators/algorithms/generic_erk.py
- src/cubie/integrators/algorithms/generic_dirk.py
- src/cubie/integrators/algorithms/generic_firk.py
- src/cubie/integrators/algorithms/generic_rosenbrock_w.py
- tests/integrators/test_algorithm_controller_compatibility.py (new)
- tests/integrators/algorithms/test_generic_erk.py (new or modified)
- tests/integrators/algorithms/test_generic_dirk.py (new or modified)
- tests/integrators/algorithms/test_generic_firk.py (new or modified)
- tests/integrators/algorithms/test_generic_rosenbrock_w.py (new or modified)
- tests/batchsolving/test_solver_validation.py (new)

**No Changes Required**:
- CUDA kernel generation
- Loop compilation
- Memory management
- Output handling
- Public API signatures (Solver, solve_ivp)
- ODE system classes
- Base algorithm or controller classes
