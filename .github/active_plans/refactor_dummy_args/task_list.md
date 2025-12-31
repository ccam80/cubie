# Implementation Task List
# Feature: Refactor Dummy Arguments for Compile-Time Logging
# Plan Reference: .github/active_plans/refactor_dummy_args/agent_plan.md

## Task Group 1: CUDAFactory Base Class Changes
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/CUDAFactory.py (entire file - lines 1-844)

**Input Validation Required**:
- None - abstract method signature only, no validation logic

**Tasks**:
1. **Add abstract `_generate_dummy_args` method to CUDAFactory**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     ```python
     # Add import for Dict at top of file
     from typing import Set, Any, Tuple, Dict
     
     # Add abstract method after build() (around line 520)
     @abstractmethod
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for compile-time measurement.
         
         Returns
         -------
         Dict[str, Tuple]
             Mapping of cached output names to their dummy argument tuples.
             Each tuple contains NumPy arrays and scalars matching the
             device function's signature with appropriate shapes.
         
         Notes
         -----
         Called by _build() to trigger CUDA compilation with correctly-shaped
         arguments. Subclasses implement this to return arguments that avoid
         illegal memory access or infinite loops during dummy execution.
         """
     ```
   - Edge cases: None - abstract method
   - Integration: All CUDAFactory subclasses must implement

2. **Modify `_build` to use `_generate_dummy_args`**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     ```python
     # In _build() method (lines 738-760), after build() is called,
     # replace the loop that calls specialize_and_compile:
     
     # Trigger compilation by running a placeholder kernel
     if default_timelogger.verbosity is not None:
         dummy_args = self._generate_dummy_args()
         for field in attrs.fields(type(self._cache)):
             device_func = getattr(self._cache, field.name)
             if device_func is None or device_func == -1:
                 continue
             if hasattr(device_func, 'py_func'):
                 event_name = f"compile_{field.name}"
                 func_args = dummy_args.get(field.name, ())
                 self.specialize_and_compile(
                     device_func, event_name, func_args
                 )
     ```
   - Edge cases: 
     - Device function not in dummy_args dict: use empty tuple
     - Verbosity not enabled: skip timing entirely
   - Integration: Relies on _generate_dummy_args returning proper dict

3. **Modify `specialize_and_compile` signature**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     ```python
     # Change signature (line 798-800) from:
     def specialize_and_compile(
         self, device_function: Any, event_name: str
     ) -> None:
     
     # To:
     def specialize_and_compile(
         self, device_function: Any, event_name: str, 
         dummy_args: Tuple = ()
     ) -> None:
         """Trigger compilation of device function and record timing.
         
         Parameters
         ----------
         device_function
             Numba CUDA device function to compile
         event_name
             Name of timing event to record (must be pre-registered)
         dummy_args
             Tuple of arguments matching the device function signature.
             NumPy arrays are transferred to device before kernel launch.
         
         Notes
         -----
         ... (keep existing notes)
         """
     ```
   - Edge cases: Empty tuple for functions with no parameters
   - Integration: Called by _build() with args from _generate_dummy_args

4. **Update `specialize_and_compile` to use provided `dummy_args`**
   - File: src/cubie/CUDAFactory.py  
   - Action: Modify
   - Details:
     ```python
     # Replace lines 824-843 to use dummy_args instead of 
     # _create_placeholder_args:
     
     if CUDA_SIMULATION:
         return
     
     # Start timing
     self._timing_start(event_name)
     
     # Transfer arrays to device and prepare arguments
     device_args = []
     for arg in dummy_args:
         if isinstance(arg, np.ndarray):
             device_args.append(cuda.to_device(arg))
         else:
             device_args.append(arg)
     placeholder_args = (tuple(device_args),)
     
     if is_devfunc(device_function):
         # Create and launch placeholder kernel
         _run_placeholder_kernel(device_function, placeholder_args)
     else:
         # If function is a kernel, just run it directly
         for signature in placeholder_args:
             device_function[1, 1, 0, 32768](*signature)
     
     cuda.synchronize()
     
     # Stop timing
     self._timing_stop(event_name)
     ```
   - Edge cases: 
     - Empty dummy_args: _run_placeholder_kernel handles 0 params
     - Scalar args: pass through without device transfer
   - Integration: Works with _run_placeholder_kernel unchanged

5. **Remove `_create_placeholder_args` function (optional - can deprecate)**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     ```python
     # The function at lines 50-211 can be either:
     # Option A: Delete entirely (breaking change)
     # Option B: Add deprecation warning and keep for backwards compat
     # Recommendation: Keep function but it's no longer called internally
     ```
   - Edge cases: External code may still call it
   - Integration: No longer called by specialize_and_compile

**Tests to Create**:
- Test file: tests/test_CUDAFactory.py
- Test function: test_generate_dummy_args_called_on_build
- Description: Verify _generate_dummy_args is called when verbosity enabled
- Test function: test_specialize_and_compile_uses_provided_args  
- Description: Verify provided dummy_args are used instead of generated ones

**Tests to Run**:
- tests/test_CUDAFactory.py::test_generate_dummy_args_called_on_build
- tests/test_CUDAFactory.py::test_specialize_and_compile_uses_provided_args

**Outcomes**: 
- Files Modified:
  * src/cubie/CUDAFactory.py (31 lines changed)
- Functions/Methods Added/Modified:
  * Added abstract method `_generate_dummy_args()` to CUDAFactory class
  * Modified `_build()` to call `_generate_dummy_args()` and pass args
  * Modified `specialize_and_compile()` signature to accept `dummy_args` param
  * Updated `specialize_and_compile()` to use provided args instead of calling `_create_placeholder_args`
- Implementation Summary:
  Refactored CUDAFactory to use a new abstract `_generate_dummy_args()` method
  that subclasses implement to provide correctly-shaped arguments for compile-
  time measurement. The `_build()` method now calls this to get dummy args and
  passes them to `specialize_and_compile()`. The `specialize_and_compile()`
  method now accepts a `dummy_args` tuple parameter and transfers NumPy arrays
  to device before kernel launch. The `_create_placeholder_args` function is
  kept for backwards compatibility but is no longer called internally.
- Issues Flagged: Tests in test_CUDAFactory.py will fail until Task Group 10
  updates test fixtures to implement `_generate_dummy_args`.

---

## Task Group 2: OutputFunctions Implementation
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/outputhandling/output_functions.py (entire file - lines 1-357)
- File: src/cubie/outputhandling/output_config.py (for OutputConfig structure)
- File: src/cubie/CUDAFactory.py (for abstract method signature)

**Input Validation Required**:
- None - method generates correctly-shaped arrays from compile_settings

**Tasks**:
1. **Implement `_generate_dummy_args` in OutputFunctions**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     ```python
     # Add after build() method (around line 232)
     
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for compile-time measurement.
         
         Returns
         -------
         Dict[str, Tuple]
             Mapping of function names to argument tuples for:
             - save_state_function
             - update_summaries_function  
             - save_summaries_function
         """
         config = self.compile_settings
         precision = config.precision
         n_saved_states = len(config.saved_state_indices)
         n_saved_obs = len(config.saved_observable_indices)
         n_counters = 4  # Fixed counter count
         buf_height = config.summaries_buffer_height_per_var
         n_summ_states = len(config.summarised_state_indices)
         n_summ_obs = len(config.summarised_observable_indices)
         
         # save_state_function(state, observables, counters, t,
         #                     state_out, obs_out, counters_out)
         save_state_args = (
             np.ones((n_saved_states,), dtype=precision),  # state
             np.ones((n_saved_obs,), dtype=precision),     # observables
             np.ones((n_counters,), dtype=np.int32),       # counters
             precision(0.0),                                # t
             np.ones((n_saved_states,), dtype=precision),  # state_out
             np.ones((n_saved_obs,), dtype=precision),     # obs_out
             np.ones((n_counters,), dtype=np.int32),       # counters_out
         )
         
         # update_summaries_function(state, observables, 
         #                           state_buffer, obs_buffer, idx)
         update_args = (
             np.ones((n_summ_states,), dtype=precision),
             np.ones((n_summ_obs,), dtype=precision),
             np.ones((buf_height * n_summ_states,), dtype=precision),
             np.ones((buf_height * n_summ_obs,), dtype=precision),
             np.int32(1),
         )
         
         # save_summaries_function(state_buffer, obs_buffer,
         #                         state_out, obs_out, summarise_every)
         save_summ_args = (
             np.ones((buf_height * n_summ_states,), dtype=precision),
             np.ones((buf_height * n_summ_obs,), dtype=precision),
             np.ones((n_summ_states,), dtype=precision),
             np.ones((n_summ_obs,), dtype=precision),
             np.int32(10),
         )
         
         return {
             'save_state_function': save_state_args,
             'update_summaries_function': update_args,
             'save_summaries_function': save_summ_args,
         }
     ```
   - Edge cases: 
     - Zero saved states/observables: arrays with shape (0,) are valid
     - Zero summarised indices: empty arrays still work
   - Integration: Called by CUDAFactory._build()

2. **Add Dict import if missing**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     ```python
     # Add Dict to imports if not present (around line 6)
     from typing import Callable, Dict, Sequence, Tuple, Union, Optional
     ```
   - Edge cases: None
   - Integration: Required for type hint

**Tests to Create**:
- Test file: tests/outputhandling/test_output_functions.py
- Test function: test_generate_dummy_args_returns_correct_keys
- Description: Verify dict contains expected function names as keys
- Test function: test_generate_dummy_args_shapes_match_config
- Description: Verify array shapes match compile_settings dimensions

**Tests to Run**:
- tests/outputhandling/test_output_functions.py::test_generate_dummy_args_returns_correct_keys
- tests/outputhandling/test_output_functions.py::test_generate_dummy_args_shapes_match_config

**Outcomes**: 
- Files Modified:
  * src/cubie/outputhandling/output_functions.py (59 lines changed)
  * tests/outputhandling/test_output_functions.py (67 lines added)
- Functions/Methods Added/Modified:
  * Added `_generate_dummy_args()` method to OutputFunctions class
  * Updated imports to include Dict and Tuple
- Implementation Summary:
  Implemented `_generate_dummy_args()` method in OutputFunctions that returns
  a dictionary mapping function names to argument tuples for save_state_function,
  update_summaries_function, and save_summaries_function. Arrays are created with
  shapes derived from compile_settings dimensions (saved/summarised indices,
  buffer heights). Added two test functions to verify correct keys and shapes.
- Issues Flagged: None

---

## Task Group 3: Step Controller Implementations  
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/step_control/base_step_controller.py (entire file - lines 1-272)
- File: src/cubie/integrators/step_control/fixed_step_controller.py (entire file)
- File: src/cubie/integrators/step_control/adaptive_I_controller.py (entire file)
- File: src/cubie/integrators/step_control/adaptive_PI_controller.py (entire file)
- File: src/cubie/integrators/step_control/adaptive_PID_controller.py (entire file)

**Input Validation Required**:
- None - method generates correctly-shaped arrays from compile_settings

**Tasks**:
1. **Add abstract `_generate_dummy_args` to BaseStepController**
   - File: src/cubie/integrators/step_control/base_step_controller.py
   - Action: Modify
   - Details:
     ```python
     # Add Dict, Tuple imports at top
     from typing import Callable, Dict, Optional, Tuple, Union
     
     # Add after local_memory_elements property (around line 198)
     @abstractmethod
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for compile-time measurement.
         
         Returns
         -------
         Dict[str, Tuple]
             Mapping of 'device_function' to argument tuple matching
             the controller signature.
         """
     ```
   - Edge cases: None - abstract method
   - Integration: All controller subclasses must implement

2. **Implement `_generate_dummy_args` in FixedStepController**
   - File: src/cubie/integrators/step_control/fixed_step_controller.py
   - Action: Modify
   - Details:
     ```python
     # Add imports
     from typing import Dict, Tuple
     import numpy as np
     
     # Add method in FixedStepController class
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for fixed step controller."""
         config = self.compile_settings
         precision = config.precision
         n = config.n
         
         # Controller signature: (dt, proposed_state, current_state,
         #                        error, niters, accept_step,
         #                        shared_scratch, persistent_local)
         return {
             'device_function': (
                 np.ones((1,), dtype=precision),      # dt buffer
                 np.ones((n,), dtype=precision),      # proposed_state
                 np.ones((n,), dtype=precision),      # current_state  
                 np.ones((n,), dtype=precision),      # error
                 np.int32(1),                         # niters
                 np.ones((1,), dtype=np.int32),       # accept_step
                 np.ones((8,), dtype=precision),      # shared_scratch
                 np.ones((8,), dtype=precision),      # persistent_local
             ),
         }
     ```
   - Edge cases: n=0 gives empty state arrays (still valid)
   - Integration: Called by CUDAFactory._build()

3. **Implement `_generate_dummy_args` in AdaptiveIController**
   - File: src/cubie/integrators/step_control/adaptive_I_controller.py
   - Action: Modify
   - Details:
     ```python
     # Same pattern as FixedStepController
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for adaptive I controller."""
         config = self.compile_settings
         precision = config.precision
         n = config.n
         
         return {
             'device_function': (
                 np.ones((1,), dtype=precision),
                 np.ones((n,), dtype=precision),
                 np.ones((n,), dtype=precision),
                 np.ones((n,), dtype=precision),
                 np.int32(1),
                 np.ones((1,), dtype=np.int32),
                 np.ones((8,), dtype=precision),
                 np.ones((8,), dtype=precision),
             ),
         }
     ```
   - Edge cases: Same as fixed
   - Integration: Called by CUDAFactory._build()

4. **Implement `_generate_dummy_args` in AdaptivePIController**
   - File: src/cubie/integrators/step_control/adaptive_PI_controller.py
   - Action: Modify
   - Details: Same pattern as AdaptiveIController
   - Edge cases: Same as fixed
   - Integration: Called by CUDAFactory._build()

5. **Implement `_generate_dummy_args` in AdaptivePIDController**
   - File: src/cubie/integrators/step_control/adaptive_PID_controller.py
   - Action: Modify
   - Details: Same pattern as AdaptiveIController
   - Edge cases: Same as fixed
   - Integration: Called by CUDAFactory._build()

**Tests to Create**:
- Test file: tests/integrators/step_control/test_step_controllers.py
- Test function: test_fixed_controller_generate_dummy_args
- Description: Verify FixedStepController returns properly shaped args
- Test function: test_adaptive_controller_generate_dummy_args
- Description: Verify adaptive controllers return properly shaped args

**Tests to Run**:
- tests/integrators/step_control/test_step_controllers.py::TestFixedControllerGenerateDummyArgs
- tests/integrators/step_control/test_step_controllers.py::TestAdaptiveControllerGenerateDummyArgs

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/step_control/base_step_controller.py (13 lines changed)
  * src/cubie/integrators/step_control/fixed_step_controller.py (33 lines changed)
  * src/cubie/integrators/step_control/adaptive_step_controller.py (30 lines changed)
  * tests/integrators/step_control/test_step_controllers.py (147 lines added)
- Functions/Methods Added/Modified:
  * Added abstract method `_generate_dummy_args()` to BaseStepController class
  * Added `_generate_dummy_args()` to FixedStepController class
  * Added `_generate_dummy_args()` to BaseAdaptiveStepController class (inherited
    by AdaptiveIController, AdaptivePIController, AdaptivePIDController)
- Implementation Summary:
  Added abstract `_generate_dummy_args()` method to BaseStepController that
  subclasses must implement. Implemented the method in FixedStepController with
  controller-specific dummy arguments. Implemented the method once in
  BaseAdaptiveStepController (rather than in each adaptive controller) since all
  adaptive controllers share the same signature. This approach is more
  maintainable than duplicating code across AdaptiveIController, 
  AdaptivePIController, and AdaptivePIDController. Created comprehensive tests
  covering both fixed and all adaptive controller implementations.
- Issues Flagged: None

---

## Task Group 4: Algorithm Step Implementations
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file - lines 1-708)
- File: src/cubie/integrators/algorithms/explicit_euler.py (entire file)
- File: src/cubie/integrators/algorithms/backwards_euler.py (entire file)
- File: src/cubie/integrators/algorithms/crank_nicolson.py (entire file)
- File: src/cubie/integrators/algorithms/generic_erk.py (entire file)
- File: src/cubie/integrators/algorithms/generic_dirk.py (entire file)

**Input Validation Required**:
- None - method generates correctly-shaped arrays from compile_settings

**Tasks**:
1. **Add abstract `_generate_dummy_args` to BaseAlgorithmStep**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     ```python
     # Add after stage_count property (around line 708)
     @abstractmethod
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for compile-time measurement.
         
         Returns
         -------
         Dict[str, Tuple]
             Mapping of 'step' to argument tuple matching the step
             function signature.
         """
     ```
   - Edge cases: None - abstract method
   - Integration: All algorithm step subclasses must implement

2. **Implement `_generate_dummy_args` in ExplicitEulerStep**
   - File: src/cubie/integrators/algorithms/explicit_euler.py
   - Action: Modify
   - Details:
     ```python
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for explicit euler step."""
         config = self.compile_settings
         precision = config.precision
         n = config.n
         n_drivers = config.n_drivers
         
         # Step function signature:
         # (state, proposed_state, parameters, driver_coeffs,
         #  drivers, proposed_drivers, observables, proposed_obs,
         #  error, dt, t, first_step, prev_accepted,
         #  shared_scratch, persistent_local, proposed_counters)
         return {
             'step': (
                 np.ones((n,), dtype=precision),           # state
                 np.ones((n,), dtype=precision),           # proposed_state
                 np.ones((n,), dtype=precision),           # parameters
                 np.ones((100, n, 6), dtype=precision),    # driver_coeffs
                 np.ones((n_drivers,), dtype=precision),   # drivers
                 np.ones((n_drivers,), dtype=precision),   # proposed_drivers
                 np.ones((n,), dtype=precision),           # observables
                 np.ones((n,), dtype=precision),           # proposed_obs
                 np.ones((n,), dtype=precision),           # error
                 precision(0.001),                         # dt
                 precision(0.0),                           # t
                 np.int32(1),                              # first_step
                 np.int32(1),                              # prev_accepted
                 np.ones((64,), dtype=precision),          # shared_scratch
                 np.ones((64,), dtype=precision),          # persistent_local
                 np.ones((2,), dtype=np.int32),            # proposed_counters
             ),
         }
     ```
   - Edge cases: 
     - n_drivers=0: empty driver arrays
     - n=1: minimal state arrays
   - Integration: Called by CUDAFactory._build()

3. **Implement `_generate_dummy_args` in BackwardsEulerStep**
   - File: src/cubie/integrators/algorithms/backwards_euler.py
   - Action: Modify
   - Details: Same pattern as ExplicitEulerStep with matching signature
   - Edge cases: Same as explicit euler
   - Integration: Called by CUDAFactory._build()

4. **Implement `_generate_dummy_args` in CrankNicolsonStep**
   - File: src/cubie/integrators/algorithms/crank_nicolson.py
   - Action: Modify
   - Details: Same pattern as ExplicitEulerStep with matching signature
   - Edge cases: Same as explicit euler
   - Integration: Called by CUDAFactory._build()

5. **Implement `_generate_dummy_args` in GenericERKStep**
   - File: src/cubie/integrators/algorithms/generic_erk.py
   - Action: Modify
   - Details: Same pattern as ExplicitEulerStep with matching signature
   - Edge cases: Same as explicit euler
   - Integration: Called by CUDAFactory._build()

6. **Implement `_generate_dummy_args` in GenericDIRKStep**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify
   - Details: Same pattern as ExplicitEulerStep with matching signature
   - Edge cases: Same as explicit euler
   - Integration: Called by CUDAFactory._build()

**Tests to Create**:
- Test file: tests/integrators/algorithms/test_algorithm_steps.py
- Test function: test_explicit_euler_generate_dummy_args
- Description: Verify ExplicitEulerStep returns properly shaped args
- Test function: test_implicit_algorithm_generate_dummy_args
- Description: Verify implicit algorithms return properly shaped args

**Tests to Run**:
- tests/integrators/algorithms/test_generate_dummy_args.py::TestExplicitEulerGenerateDummyArgs
- tests/integrators/algorithms/test_generate_dummy_args.py::TestImplicitAlgorithmGenerateDummyArgs
- tests/integrators/algorithms/test_generate_dummy_args.py::TestERKStepGenerateDummyArgs

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/explicit_euler.py (40 lines changed)
  * src/cubie/integrators/algorithms/backwards_euler.py (38 lines changed)
  * src/cubie/integrators/algorithms/crank_nicolson.py (38 lines changed)
  * src/cubie/integrators/algorithms/generic_erk.py (40 lines changed)
  * src/cubie/integrators/algorithms/generic_dirk.py (38 lines changed)
  * tests/integrators/algorithms/test_generate_dummy_args.py (170 lines added)
- Functions/Methods Added/Modified:
  * Added `_generate_dummy_args()` to ExplicitEulerStep class
  * Added `_generate_dummy_args()` to BackwardsEulerStep class
  * Added `_generate_dummy_args()` to CrankNicolsonStep class
  * Added `_generate_dummy_args()` to ERKStep class
  * Added `_generate_dummy_args()` to DIRKStep class
- Implementation Summary:
  Implemented `_generate_dummy_args()` method in each algorithm step class to
  satisfy the abstract method requirement from CUDAFactory. Each implementation
  returns a dictionary with 'step' key mapping to a 16-element tuple of dummy
  arguments matching the step function signature. Arrays are created with shapes
  derived from compile_settings (n, n_drivers) and shared/persistent memory
  requirements. Added comprehensive tests covering explicit Euler, implicit
  algorithms (BackwardsEuler, CrankNicolson, DIRK), and ERK step implementations.
- Issues Flagged: BaseAlgorithmStep already inherits the abstract method from
  CUDAFactory, so no additional abstract declaration was needed in the base
  class.

---

## Task Group 5: IVPLoop Implementation and Critical Shape Removal
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (entire file - lines 1-930)
- File: src/cubie/integrators/loops/ode_loop_config.py (for ODELoopConfig structure)

**Input Validation Required**:
- None - method generates correctly-shaped arrays from compile_settings

**Tasks**:
1. **Implement `_generate_dummy_args` in IVPLoop**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Add Dict, Tuple imports at top if needed
     from typing import Callable, Dict, Optional, Set, Tuple
     
     # Add method after build() (around line 767)
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for compile-time measurement.
         
         Returns
         -------
         Dict[str, Tuple]
             Mapping of 'loop_function' to argument tuple.
         """
         config = self.compile_settings
         precision = config.precision
         n_states = config.n_states
         n_parameters = config.n_parameters
         n_observables = config.n_observables
         n_counters = config.n_counters
         dt_save = config.dt_save
         
         # loop_fn signature: (initial_states, parameters, driver_coeffs,
         #                     shared_scratch, persistent_local,
         #                     state_output, observables_output,
         #                     state_summaries_output, observable_summaries_output,
         #                     iteration_counters_output, duration, settling_time, t0)
         return {
             'loop_function': (
                 np.ones((n_states,), dtype=precision),
                 np.ones((n_parameters,), dtype=precision),
                 np.ones((100, n_states, 6), dtype=precision),
                 np.ones((4096,), dtype=np.float32),  # shared_scratch
                 np.ones((4096,), dtype=precision),   # persistent_local
                 np.ones((100, n_states), dtype=precision),
                 np.ones((100, n_observables), dtype=precision),
                 np.ones((100, n_states), dtype=precision),
                 np.ones((100, n_observables), dtype=precision),
                 np.ones((1, n_counters), dtype=np.int32),
                 np.float64(dt_save + 0.01),  # duration
                 np.float64(0.0),             # settling_time
                 np.float64(0.0),             # t0
             ),
         }
     ```
   - Edge cases:
     - n_observables=0: shape (100, 0) arrays
     - n_counters=0: shape (1, 0) array
   - Integration: Called by CUDAFactory._build()

2. **Remove `critical_shapes` and `critical_values` attributes**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Delete lines 732-766 that attach critical_shapes and critical_values
     # to loop_fn after it's defined. The block starts with:
     #   # Attach critical shapes for dummy execution
     # and ends with:
     #   loop_fn.critical_values = (...)
     ```
   - Edge cases: None - removing dead code
   - Integration: These attributes are no longer used

**Tests to Create**:
- Test file: tests/integrators/loops/test_ode_loop.py
- Test function: test_ivploop_generate_dummy_args
- Description: Verify IVPLoop returns properly shaped args
- Test function: test_ivploop_no_critical_shapes_attribute
- Description: Verify loop_fn no longer has critical_shapes attribute

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py::test_ivploop_generate_dummy_args
- tests/integrators/loops/test_ode_loop.py::test_ivploop_no_critical_shapes_attribute

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/loops/ode_loop.py (74 lines changed - added method, removed critical_shapes/values)
  * tests/integrators/loops/test_ode_loop.py (74 lines added)
- Functions/Methods Added/Modified:
  * Added `_generate_dummy_args()` to IVPLoop class
  * Removed `critical_shapes` attribute assignment from `build()` method
  * Removed `critical_values` attribute assignment from `build()` method
- Implementation Summary:
  Implemented `_generate_dummy_args()` method in IVPLoop that returns a 
  dictionary mapping 'loop_function' to a 13-element tuple of dummy arguments
  matching the loop device function signature. Arrays are created with shapes
  derived from compile_settings (n_states, n_parameters, n_observables,
  n_counters). Removed the critical_shapes and critical_values attribute
  assignments from the build() method as they are no longer needed with the
  new _generate_dummy_args approach. Created comprehensive tests to verify
  correct keys, shapes, dtypes, and absence of deprecated attributes.
- Issues Flagged: None

---

## Task Group 6: SingleIntegratorRunCore Implementation
**Status**: [x]
**Dependencies**: Task Group 5

**Required Context**:
- File: src/cubie/integrators/SingleIntegratorRunCore.py (entire file - lines 1-556)
- File: src/cubie/integrators/loops/ode_loop.py (for IVPLoop._generate_dummy_args)

**Input Validation Required**:
- None - delegates to IVPLoop

**Tasks**:
1. **Implement `_generate_dummy_args` in SingleIntegratorRunCore**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     # Add Dict, Tuple imports at top
     from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple
     
     # Add method after build() (around line 555)
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for compile-time measurement.
         
         Returns
         -------
         Dict[str, Tuple]
             Mapping of 'single_integrator_function' to argument tuple.
             Delegates to the underlying IVPLoop for actual generation.
         """
         loop_args = self._loop._generate_dummy_args()
         return {
             'single_integrator_function': loop_args.get('loop_function', ()),
         }
     ```
   - Edge cases: 
     - IVPLoop not yet built: should work since compile_settings available
   - Integration: Delegates to IVPLoop._generate_dummy_args()

**Tests to Create**:
- Test file: tests/integrators/test_single_integrator_run.py
- Test function: test_single_integrator_generate_dummy_args
- Description: Verify SingleIntegratorRunCore returns properly shaped args

**Tests to Run**:
- tests/integrators/test_SingleIntegratorRun.py::test_single_integrator_generate_dummy_args

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/SingleIntegratorRunCore.py (14 lines changed)
  * tests/integrators/test_SingleIntegratorRun.py (31 lines added)
- Functions/Methods Added/Modified:
  * Added `_generate_dummy_args()` to SingleIntegratorRunCore class
  * Updated imports to include Tuple
- Implementation Summary:
  Implemented `_generate_dummy_args()` method in SingleIntegratorRunCore that
  delegates to the underlying IVPLoop's `_generate_dummy_args()` method and
  remaps the 'loop_function' key to 'single_integrator_function' to match the
  SingleIntegratorRunCache field name. The method returns a dictionary with
  the same 13-element tuple that IVPLoop generates. Created comprehensive test
  to verify correct key mapping, argument count, and shape matching.
- Issues Flagged: None

---

## Task Group 7: BatchSolverKernel Implementation and Critical Shape Removal
**Status**: [x]
**Dependencies**: Task Group 6

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file - lines 1-1252)
- File: src/cubie/integrators/SingleIntegratorRunCore.py (for system_sizes)

**Input Validation Required**:
- None - method generates correctly-shaped arrays from system_sizes

**Tasks**:
1. **Implement `_generate_dummy_args` in BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Add Dict, Tuple imports if needed at top
     from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
     
     # Add method after build() (around line 903)
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for compile-time measurement.
         
         Returns
         -------
         Dict[str, Tuple]
             Mapping of 'solver_kernel' to argument tuple.
         """
         precision = self.precision
         system_sizes = self.system_sizes
         n_states = int(system_sizes.states)
         n_params = int(system_sizes.parameters)
         n_obs = int(system_sizes.observables)
         n_drivers = int(system_sizes.drivers)
         
         # integration_kernel signature matches kernel definition
         return {
             'solver_kernel': (
                 np.ones((n_states, 1), dtype=precision),        # inits
                 np.ones((n_params, 1), dtype=precision),        # params
                 np.ones((100, n_drivers, 6), dtype=precision),  # d_coefficients
                 np.ones((100, n_states, 1), dtype=precision),   # state_output
                 np.ones((100, n_obs, 1), dtype=precision),      # observables_output
                 np.ones((100, n_states, 1), dtype=precision),   # state_summaries
                 np.ones((100, n_obs, 1), dtype=precision),      # obs_summaries
                 np.ones((100, 4, 1), dtype=np.int32),           # iteration_counters
                 np.ones((1,), dtype=np.int32),                  # status_codes
                 np.float64(0.001),                              # duration
                 np.float64(0.0),                                # warmup
                 np.float64(0.0),                                # t0
                 np.int32(1),                                    # n_runs
             ),
         }
     ```
   - Edge cases:
     - n_obs=0: shape (100, 0, 1) arrays
     - n_drivers=0: shape (100, 0, 6) array
   - Integration: Called by CUDAFactory._build()

2. **Remove `critical_shapes` and `critical_values` attributes from build_kernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Delete lines 705-745 that attach critical_shapes and critical_values
     # to integration_kernel. The block starts with:
     #   # Attach critical shapes for dummy execution
     # and ends with:
     #   integration_kernel.critical_values = (...)
     ```
   - Edge cases: None - removing dead code
   - Integration: These attributes are no longer used

**Tests to Create**:
- Test file: tests/batchsolving/test_batch_solver_kernel.py
- Test function: test_batch_solver_generate_dummy_args
- Description: Verify BatchSolverKernel returns properly shaped args
- Test function: test_batch_solver_no_critical_shapes_attribute
- Description: Verify kernel no longer has critical_shapes attribute

**Tests to Run**:
- tests/batchsolving/test_batch_solver_kernel.py::TestBatchSolverGenerateDummyArgs::test_batch_solver_generate_dummy_args
- tests/batchsolving/test_batch_solver_kernel.py::TestBatchSolverGenerateDummyArgs::test_batch_solver_no_critical_shapes_attribute
- tests/batchsolving/test_batch_solver_kernel.py::TestBatchSolverGenerateDummyArgs::test_batch_solver_no_critical_values_attribute

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (76 lines changed - added method, removed critical_shapes/values)
  * tests/batchsolving/test_batch_solver_kernel.py (89 lines added - new test file)
- Functions/Methods Added/Modified:
  * Added `_generate_dummy_args()` method to BatchSolverKernel class
  * Removed `critical_shapes` attribute assignment from `build_kernel()` method
  * Removed `critical_values` attribute assignment from `build_kernel()` method
- Implementation Summary:
  Implemented `_generate_dummy_args()` method in BatchSolverKernel that returns a
  dictionary mapping 'solver_kernel' to a 13-element tuple of dummy arguments
  matching the integration_kernel signature. Arrays are created with shapes derived
  from system_sizes (n_states, n_params, n_observables, n_drivers). Removed the
  critical_shapes and critical_values attribute assignments from build_kernel() as
  they are no longer needed with the new _generate_dummy_args approach. Created
  comprehensive tests to verify correct keys, shapes, dtypes, and absence of
  deprecated attributes.
- Issues Flagged: None

---

## Task Group 8: BaseODE and SummaryMetrics Implementation
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/odesystems/baseODE.py (entire file - lines 1-415)
- File: src/cubie/odesystems/symbolic/symbolicODE.py (for SymbolicODE implementation)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (for SummaryMetric class)

**Input Validation Required**:
- None - method generates correctly-shaped arrays from compile_settings

**Tasks**:
1. **Add abstract `_generate_dummy_args` to BaseODE**
   - File: src/cubie/odesystems/baseODE.py
   - Action: Modify
   - Details:
     ```python
     # Add Dict, Tuple imports at top
     from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
     
     # Add abstract method after get_solver_helper (around line 414)
     @abstractmethod
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for compile-time measurement.
         
         Returns
         -------
         Dict[str, Tuple]
             Mapping of cached output names (e.g., 'dxdt', 'observables')
             to their argument tuples.
         """
     ```
   - Edge cases: None - abstract method
   - Integration: SymbolicODE must implement

2. **Implement `_generate_dummy_args` in SymbolicODE**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Modify
   - Details:
     ```python
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for compile-time measurement."""
         precision = self.precision
         sizes = self.sizes
         n_states = int(sizes.states)
         n_params = int(sizes.parameters)
         n_drivers = int(sizes.drivers)
         n_obs = int(sizes.observables)
         
         # dxdt signature: (state, dxdt_out, parameters, drivers, t)
         dxdt_args = (
             np.ones((n_states,), dtype=precision),
             np.ones((n_states,), dtype=precision),
             np.ones((n_params,), dtype=precision),
             np.ones((n_drivers,), dtype=precision),
             precision(0.0),
         )
         
         # observables signature: (state, params, drivers, obs_out, t)
         obs_args = (
             np.ones((n_states,), dtype=precision),
             np.ones((n_params,), dtype=precision),
             np.ones((n_drivers,), dtype=precision),
             np.ones((n_obs,), dtype=precision),
             precision(0.0),
         )
         
         result = {
             'dxdt': dxdt_args,
             'observables': obs_args,
         }
         
         # Add solver helper functions if they exist
         # These are optional and may return -1 if not implemented
         return result
     ```
   - Edge cases:
     - n_drivers=0: empty driver arrays
     - n_obs=0: empty observable array
   - Integration: Called by CUDAFactory._build()

3. **Implement `_generate_dummy_args` in SummaryMetric**
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     ```python
     # Add to SummaryMetric class after build() method
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for compile-time measurement."""
         config = self.compile_settings
         precision = config.precision
         
         # update signature: (value, buffer, idx, custom_var)
         update_args = (
             precision(1.0),
             np.ones((10,), dtype=precision),
             np.int32(0),
             precision(1.0),
         )
         
         # save signature: (buffer, output, summarise_every, custom_var)
         save_args = (
             np.ones((10,), dtype=precision),
             np.ones((1,), dtype=precision),
             np.int32(10),
             precision(1.0),
         )
         
         return {
             'update': update_args,
             'save': save_args,
         }
     ```
   - Edge cases: None - simple scalar and buffer args
   - Integration: Called by CUDAFactory._build()

**Tests to Create**:
- Test file: tests/odesystems/test_base_ode.py
- Test function: test_symbolic_ode_generate_dummy_args
- Description: Verify SymbolicODE returns properly shaped args for dxdt
- Test file: tests/outputhandling/summarymetrics/test_metrics.py
- Test function: test_summary_metric_generate_dummy_args
- Description: Verify SummaryMetric returns properly shaped args

**Tests to Run**:
- tests/odesystems/test_base_ode.py::test_symbolic_ode_generate_dummy_args
- tests/odesystems/test_base_ode.py::test_symbolic_ode_generate_dummy_args_observables
- tests/odesystems/test_base_ode.py::test_symbolic_ode_generate_dummy_args_no_drivers
- tests/outputhandling/summarymetrics/test_summary_metrics.py::test_summary_metric_generate_dummy_args

**Outcomes**: 
- Files Modified:
  * src/cubie/odesystems/symbolic/symbolicODE.py (40 lines changed)
  * src/cubie/outputhandling/summarymetrics/metrics.py (32 lines changed)
  * tests/odesystems/test_base_ode.py (97 lines added - new file)
  * tests/outputhandling/summarymetrics/test_summary_metrics.py (36 lines added)
- Functions/Methods Added/Modified:
  * Added `_generate_dummy_args()` to SymbolicODE class
  * Added `_generate_dummy_args()` to SummaryMetric class
  * Updated imports in both files (Dict, Tuple)
- Implementation Summary:
  Implemented `_generate_dummy_args()` method in SymbolicODE that returns a
  dictionary mapping 'dxdt' and 'observables' to argument tuples with shapes
  derived from system sizes (n_states, n_params, n_drivers, n_observables).
  Implemented the method in SummaryMetric with 'update' and 'save' keys for
  metric buffer operations. Note: BaseODE already inherits the abstract method
  from CUDAFactory, so no additional declaration was needed. Created
  comprehensive tests for both implementations covering shape validation and
  edge cases like zero drivers.
- Issues Flagged: None

---

## Task Group 9: Additional Algorithm Implementations
**Status**: [x]
**Dependencies**: Task Group 4

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py (entire file)
- File: src/cubie/integrators/algorithms/generic_firk.py (entire file)
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (entire file)

**Input Validation Required**:
- None - method generates correctly-shaped arrays from compile_settings

**Tasks**:
1. **Implement `_generate_dummy_args` in BackwardsEulerPCStep**
   - File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py
   - Action: Modify
   - Details: Same pattern as ExplicitEulerStep with matching signature
   - Edge cases: Same as explicit euler
   - Integration: Called by CUDAFactory._build()

2. **Implement `_generate_dummy_args` in GenericFIRKStep**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify
   - Details: Same pattern as ExplicitEulerStep with matching signature
   - Edge cases: Same as explicit euler
   - Integration: Called by CUDAFactory._build()

3. **Implement `_generate_dummy_args` in GenericRosenbrockWStep**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify
   - Details: Same pattern as ExplicitEulerStep with matching signature
   - Edge cases: Same as explicit euler
   - Integration: Called by CUDAFactory._build()

**Tests to Create**:
- Test file: tests/integrators/algorithms/test_algorithm_steps.py
- Test function: test_firk_generate_dummy_args
- Description: Verify GenericFIRKStep returns properly shaped args
- Test function: test_rosenbrock_generate_dummy_args
- Description: Verify GenericRosenbrockWStep returns properly shaped args

**Tests to Run**:
- tests/integrators/algorithms/test_generate_dummy_args.py::TestFIRKStepGenerateDummyArgs
- tests/integrators/algorithms/test_generate_dummy_args.py::TestRosenbrockWStepGenerateDummyArgs
- tests/integrators/algorithms/test_generate_dummy_args.py::TestBackwardsEulerPCStepGenerateDummyArgs

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_firk.py (37 lines added)
  * src/cubie/integrators/algorithms/generic_rosenbrock_w.py (37 lines added)
  * tests/integrators/algorithms/test_generate_dummy_args.py (118 lines added)
- Functions/Methods Added/Modified:
  * Added `_generate_dummy_args()` to FIRKStep class
  * Added `_generate_dummy_args()` to GenericRosenbrockWStep class
  * Updated imports to include Dict and Tuple in both files
- Implementation Summary:
  Implemented `_generate_dummy_args()` method in FIRKStep and GenericRosenbrockWStep
  classes following the same pattern as existing algorithm step implementations. Each
  method returns a dictionary mapping 'step' to a 16-element tuple of dummy arguments
  matching the step function signature. BackwardsEulerPCStep inherits its
  `_generate_dummy_args()` from BackwardsEulerStep since both have the same step
  signature. Created comprehensive tests for all three classes covering dict keys,
  tuple length, and array shapes matching compile_settings dimensions.
- Issues Flagged: None

---

## Task Group 10: Test Updates and Cleanup
**Status**: [x]
**Dependencies**: Task Groups 1-9

**Required Context**:
- File: tests/test_CUDAFactory.py (entire file)
- File: src/cubie/CUDAFactory.py (for updated signatures)

**Input Validation Required**:
- None - test code only

**Tasks**:
1. **Update test fixtures to implement `_generate_dummy_args`**
   - File: tests/test_CUDAFactory.py
   - Action: Modify
   - Details:
     ```python
     # Update the ConcreteFactory class in factory fixture (around line 36)
     class ConcreteFactory(CUDAFactory):
         def __init__(self):
             super().__init__()
     
         def build(self):
             return testCache(device_function=lambda: 20.0)
         
         def _generate_dummy_args(self):
             """Return empty dummy args for test factory."""
             return {'device_function': ()}
     ```
   - Edge cases: None - test factory only
   - Integration: Required for tests to pass after abstract method added

2. **Add test for `_generate_dummy_args` being called**
   - File: tests/test_CUDAFactory.py
   - Action: Modify
   - Details:
     ```python
     @pytest.mark.nocudasim
     def test_generate_dummy_args_called_on_build_with_verbosity(
         factory_with_settings
     ):
         """Test that _generate_dummy_args is called when verbosity enabled."""
         from cubie.time_logger import TimeLogger
         
         call_count = [0]
         original_method = factory_with_settings._generate_dummy_args
         
         def tracking_method():
             call_count[0] += 1
             return original_method()
         
         factory_with_settings._generate_dummy_args = tracking_method
         
         # Enable verbosity
         timelogger = TimeLogger(verbosity='verbose')
         
         # Trigger build
         _ = factory_with_settings.device_function
         
         # Verify method was called
         assert call_count[0] >= 1
     ```
   - Edge cases: CUDASIM mode skips compilation
   - Integration: Tests new abstract method behavior

3. **Update tests that use `_create_placeholder_args` directly**
   - File: tests/test_CUDAFactory.py
   - Action: Modify
   - Details:
     ```python
     # Tests at lines 259-335 that test _create_placeholder_args
     # should still work as the function remains available
     # No changes needed unless function is removed
     ```
   - Edge cases: None - existing tests still valid
   - Integration: Backward compatible

**Tests to Create**:
- Test file: tests/test_CUDAFactory.py
- Test function: test_generate_dummy_args_called_on_build_with_verbosity
- Description: Verify _generate_dummy_args called when verbosity enabled

**Tests to Run**:
- tests/test_CUDAFactory.py (entire file)

**Outcomes**: 
- Files Modified:
  * tests/test_CUDAFactory.py (10 lines changed)
- Functions/Methods Added/Modified:
  * Added `_generate_dummy_args()` method to ConcreteFactory test fixture
  * Added `test_generate_dummy_args_called_on_build_with_verbosity()` test
- Implementation Summary:
  Updated the ConcreteFactory class in the factory fixture to implement the
  abstract `_generate_dummy_args()` method, returning a minimal dictionary
  with empty tuple for 'device_function'. Added a new test marked with
  @pytest.mark.nocudasim that verifies `_generate_dummy_args` is called
  when TimeLogger verbosity is enabled. Existing tests for
  `_create_placeholder_args` remain unchanged and functional.
- Issues Flagged: None

---

# Summary

## Total Task Groups: 10
## Dependency Chain Overview:
1. **Task Group 1** (CUDAFactory base) → Foundation for all others
2. **Task Groups 2-4, 8** (OutputFunctions, Controllers, Algorithms, ODE) → Can run in parallel after Group 1
3. **Task Group 5** (IVPLoop) → Depends on Group 1
4. **Task Group 6** (SingleIntegratorRunCore) → Depends on Group 5
5. **Task Group 7** (BatchSolverKernel) → Depends on Group 6
6. **Task Group 9** (Additional algorithms) → Depends on Group 4
7. **Task Group 10** (Tests) → Depends on Groups 1-9

## Tests Summary:
- **Tests to Create**: 20+ new test functions across multiple test files
- **Tests to Run**: Full test suite after all groups complete
- **Key test files**:
  - tests/test_CUDAFactory.py
  - tests/outputhandling/test_output_functions.py
  - tests/integrators/step_control/test_step_controllers.py
  - tests/integrators/algorithms/test_algorithm_steps.py
  - tests/integrators/loops/test_ode_loop.py
  - tests/batchsolving/test_batch_solver_kernel.py
  - tests/odesystems/test_base_ode.py

## Estimated Complexity:
- **High complexity**: Task Groups 1, 7 (core changes)
- **Medium complexity**: Task Groups 4, 5, 8 (multiple files)
- **Low complexity**: Task Groups 2, 3, 6, 9, 10 (straightforward pattern)

## Critical Removals:
- `critical_shapes` attribute in BatchSolverKernel (lines 713-727)
- `critical_values` attribute in BatchSolverKernel (lines 729-745)
- `critical_shapes` attribute in IVPLoop (lines 737-751)
- `critical_values` attribute in IVPLoop (lines 752-766)
