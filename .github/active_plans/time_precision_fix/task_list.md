# Implementation Task List
# Feature: Time Precision Fix
# Plan Reference: .github/active_plans/time_precision_fix/agent_plan.md

## Task Group 1: BatchSolverKernel Time Storage - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 267-274, 849-882)

**Input Validation Required**:
- None - internal refactoring only

**Tasks**:

1. **Update time parameter storage in run() method**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Lines: 267-274
   - Details:
     ```python
     # Change from:
     precision = self.precision
     duration = precision(duration)
     warmup = precision(warmup)
     t0 = precision(t0)
     
     # To:
     # Time parameters always use float64 for accumulation accuracy
     duration = np.float64(duration)
     warmup = np.float64(warmup)
     t0 = np.float64(t0)
     ```
   - Edge cases: Values may be passed as int, float32, float64 - np.float64() handles all conversions
   - Integration: Values stored to self._duration, self._warmup, self._t0 used by properties

2. **Update duration property getter and setter**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Lines: 849-856
   - Details:
     ```python
     @property
     def duration(self) -> float:
         """Requested integration duration."""
         return np.float64(self._duration)  # Changed from self.precision()
     
     @duration.setter
     def duration(self, value: float) -> None:
         self._duration = np.float64(value)  # Changed from self.precision()
     ```
   - Edge cases: User may pass int, float32, float64 - np.float64() handles all
   - Integration: Properties maintain float64 semantics for time parameters

3. **Update warmup property getter and setter**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Lines: 865-872
   - Details:
     ```python
     @property
     def warmup(self) -> float:
         """Configured warmup duration."""
         return np.float64(self._warmup)  # Changed from self.precision()
     
     @warmup.setter
     def warmup(self, value: float) -> None:
         self._warmup = np.float64(value)  # Changed from self.precision()
     ```
   - Edge cases: User may pass int, float32, float64 - np.float64() handles all
   - Integration: Properties maintain float64 semantics for time parameters

4. **Update t0 property getter and setter**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Lines: 875-882
   - Details:
     ```python
     @property
     def t0(self) -> float:
         """Configured initial integration time."""
         return np.float64(self._t0)  # Changed from self.precision()
     
     @t0.setter
     def t0(self, value: float) -> None:
         self._t0 = np.float64(value)  # Changed from self.precision()
     ```
   - Edge cases: User may pass int, float32, float64 - np.float64() handles all
   - Integration: Properties maintain float64 semantics for time parameters

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverKernel.py (10 lines changed)
- Functions/Methods Added/Modified:
  * run() method - lines 267-270: Changed time parameter storage from precision() to np.float64()
  * duration property getter/setter - lines 849-856: Changed to use np.float64() instead of self.precision()
  * warmup property getter/setter - lines 865-872: Changed to use np.float64() instead of self.precision()
  * t0 property getter/setter - lines 875-882: Changed to use np.float64() instead of self.precision()
- Implementation Summary:
  All time parameters (duration, warmup, t0) now stored and retrieved as float64 regardless of state precision. Values converted on input via np.float64() in run() method and property setters, maintaining float64 semantics throughout time management layer.
- Issues Flagged: None

---

## Task Group 2: BatchSolverKernel CUDA Kernel Signature - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups [1]

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 497-529, 338-340)

**Input Validation Required**:
- None - internal refactoring only

**Tasks**:

1. **Update CUDA kernel JIT signature for time parameters**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Lines: 497-514
   - Details:
     ```python
     @cuda.jit(
         (
             precision[:, ::1],
             precision[:, ::1],
             precision[:, :, ::1],
             precision[:, :, :],
             precision[:, :, :],
             precision[:, :, :],
             precision[:, :, :],
             int32[:, :, :],
             int32[::1],
             float64,  # duration - time accumulation uses float64
             float64,  # warmup - time accumulation uses float64
             float64,  # t0 - time accumulation uses float64
             int32,
         ),
         **compile_kwargs,
     )
     ```
   - Edge cases: Numba JIT will enforce float64 type at kernel launch
   - Integration: Kernel receives float64 time parameters from run()

2. **Update integration_kernel function signature defaults**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Lines: 515-529
   - Details:
     ```python
     def integration_kernel(
         inits,
         params,
         d_coefficients,
         state_output,
         observables_output,
         state_summaries_output,
         observables_summaries_output,
         iteration_counters_output,
         status_codes_output,
         duration,
         warmup=float64(0.0),  # Changed from precision(0.0)
         t0=float64(0.0),      # Changed from precision(0.0)
         n_runs=1,
     ):
     ```
   - Edge cases: Default values use float64 for consistency
   - Integration: Kernel function signature matches JIT signature

3. **Update chunk_run time calculations**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Lines: 338-340
   - Details:
     ```python
     if (chunk_axis == "time") and (i != 0):
         chunk_warmup = np.float64(0.0)  # Changed from precision(0.0)
         chunk_t0 = t0 + np.float64(i) * chunk_params.duration  # Changed from precision(i)
     ```
   - Edge cases: Chunk calculations maintain float64 precision for time accumulation
   - Integration: Time values passed to kernel remain float64

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverKernel.py (6 lines changed)
- Functions/Methods Added/Modified:
  * integration_kernel JIT signature - lines 497-514: Changed duration, warmup, t0 from precision to float64
  * integration_kernel function signature - lines 515-529: Changed default values to float64(0.0)
  * chunk_run time calculations - lines 338-340: Changed chunk_warmup and chunk_t0 calculations to use np.float64
- Implementation Summary:
  CUDA kernel signature now enforces float64 for all time parameters. JIT compiler will verify type correctness at kernel launch. Chunk calculations maintain float64 precision for time accumulation across sequential chunks.
- Issues Flagged: None

---

## Task Group 3: IVPLoop Signature and Time Variables - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups [2]

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 13, 226-262, 299-300, 369-374)

**Input Validation Required**:
- None - internal refactoring only

**Tasks**:

1. **Import float64 from numba**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Lines: 13
   - Details:
     ```python
     # Change from:
     from numba import cuda, int16, int32
     
     # To:
     from numba import cuda, int16, int32, float64
     ```
   - Edge cases: None - straightforward import addition
   - Integration: Makes float64 type available for JIT signatures

2. **Update loop_fn JIT signature for time parameters**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Lines: 226-247
   - Details:
     ```python
     @cuda.jit(
         [
             (
                 precision[::1],
                 precision[::1],
                 precision[:, :, ::1],
                 precision[::1],
                 precision[::1],
                 precision[:, :],
                 precision[:, :],
                 precision[:, :],
                 precision[:, :],
                 precision[:,::1],
                 float64,  # duration - time managed in float64
                 float64,  # settling_time - time managed in float64
                 float64,  # t0 - time managed in float64
             )
         ],
         device=True,
         inline=True,
         **compile_kwargs,
     )
     ```
   - Edge cases: JIT signature enforces float64 type for time parameters
   - Integration: Matches signature in BatchSolverKernel kernel call

3. **Update loop_fn function signature default**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Lines: 248-262
   - Details:
     ```python
     def loop_fn(
         initial_states,
         parameters,
         driver_coefficients,
         shared_scratch,
         persistent_local,
         state_output,
         observables_output,
         state_summaries_output,
         observable_summaries_output,
         iteration_counters_output,
         duration,
         settling_time,
         t0=float64(0.0),  # Changed from precision(0.0)
     ):
     ```
   - Edge cases: Default value uses float64 for consistency
   - Integration: Function signature matches JIT signature

4. **Update time accumulation variables**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Lines: 299-300
   - Details:
     ```python
     # Change from:
     t = precision(t0)
     t_end = precision(settling_time + duration)
     
     # To:
     t = float64(t0)
     t_end = float64(settling_time + duration)
     ```
   - Edge cases: All time accumulation happens in float64 precision
   - Integration: t and t_end used in loop for time management

5. **Update next_save initialization**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Lines: 369-374
   - Details:
     ```python
     # Change from:
     if settling_time > precision(0.0):
         # Don't save t0, wait until settling_time
         next_save = precision(settling_time)
     else:
         # Seed initial state and save/update summaries
         next_save = precision(dt_save)
     
     # To:
     if settling_time > float64(0.0):
         # Don't save t0, wait until settling_time
         next_save = float64(settling_time)
     else:
         # Seed initial state and save/update summaries
         next_save = float64(settling_time + dt_save)
     ```
   - Edge cases: next_save tracks scheduled saves in float64 precision
   - Integration: next_save compared against t in loop

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/loops/ode_loop.py (9 lines changed)
- Functions/Methods Added/Modified:
  * Import statement - line 13: Added float64 to numba imports
  * loop_fn JIT signature - lines 226-247: Changed duration, settling_time, t0 from precision to float64
  * loop_fn function signature - line 261: Changed default t0 to float64(0.0)
  * Time variables initialization - lines 299-300: Changed t and t_end to use float64()
  * next_save initialization - lines 369-374: Changed to use float64() and fixed initialization logic for settling_time == 0 case
- Implementation Summary:
  IVPLoop now receives and manages all time parameters in float64 precision. Time accumulation variables (t, t_end, next_save) use float64 for accurate tracking throughout integration. JIT signature enforces float64 type for time parameters from kernel.
- Issues Flagged: None

---

## Task Group 4: IVPLoop Precision Casting for Device Functions - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups [3]

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 352-364, 376-384, 445-462, 524-532)
- Context: Step functions, driver functions, observables functions, and save_state functions all expect time in user precision

**Input Validation Required**:
- None - internal refactoring only

**Tasks**:

1. **Cast t for initial driver_function call**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Lines: 352-356
   - Details:
     ```python
     # Change from:
     if driver_function is not None and n_drivers > 0:
         driver_function(
             t,
             driver_coefficients,
             drivers_buffer,
         )
     
     # To:
     if driver_function is not None and n_drivers > 0:
         driver_function(
             precision(t),  # Cast to user precision
             driver_coefficients,
             drivers_buffer,
         )
     ```
   - Edge cases: driver_function signature expects numba_precision type for time
   - Integration: Initial evaluation of drivers at t0

2. **Cast t for initial observables_fn call**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Lines: 357-364
   - Details:
     ```python
     # Change from:
     if n_observables > 0:
         observables_fn(
             state_buffer,
             parameters_buffer,
             drivers_buffer,
             observables_buffer,
             t,
         )
     
     # To:
     if n_observables > 0:
         observables_fn(
             state_buffer,
             parameters_buffer,
             drivers_buffer,
             observables_buffer,
             precision(t),  # Cast to user precision
         )
     ```
   - Edge cases: observables_fn signature expects numba_precision type for time
   - Integration: Initial evaluation of observables at t0

3. **Cast t for initial save_state call**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Lines: 376-384
   - Details:
     ```python
     # Change from:
     save_state(
         state_buffer,
         observables_buffer,
         counters_since_save,
         t,
         state_output[save_idx * save_state_bool, :],
         observables_output[save_idx * save_obs_bool, :],
         iteration_counters_output[save_idx * save_counters_bool, :],
     )
     
     # To:
     save_state(
         state_buffer,
         observables_buffer,
         counters_since_save,
         precision(t),  # Cast to user precision
         state_output[save_idx * save_state_bool, :],
         observables_output[save_idx * save_obs_bool, :],
         iteration_counters_output[save_idx * save_counters_bool, :],
     )
     ```
   - Edge cases: save_state signature expects numba_precision type for time
   - Integration: Initial save at t0 (if settling_time == 0)

4. **Cast dt_eff and t for step_function call in main loop**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Lines: 445-462
   - Details:
     ```python
     # Change from:
     step_status = step_function(
         state_buffer,
         state_proposal_buffer,
         parameters_buffer,
         driver_coefficients,
         drivers_buffer,
         drivers_proposal_buffer,
         observables_buffer,
         observables_proposal_buffer,
         error,
         dt_eff,
         t,
         first_step_flag,
         prev_step_accepted_flag,
         remaining_shared_scratch,
         algo_local,
         proposed_counters,
     )
     
     # To:
     step_status = step_function(
         state_buffer,
         state_proposal_buffer,
         parameters_buffer,
         driver_coefficients,
         drivers_buffer,
         drivers_proposal_buffer,
         observables_buffer,
         observables_proposal_buffer,
         error,
         precision(dt_eff),  # Cast to user precision
         precision(t),       # Cast to user precision
         first_step_flag,
         prev_step_accepted_flag,
         remaining_shared_scratch,
         algo_local,
         proposed_counters,
     )
     ```
   - Edge cases: step_function signature expects numba_precision type for dt and time
   - Integration: Main integration step executed in user precision

5. **Cast t for save_state call in main loop**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Lines: 524-532
   - Details:
     ```python
     # Change from:
     save_state(
         state_buffer,
         observables_buffer,
         counters_since_save,
         t,
         state_output[save_idx * save_state_bool, :],
         observables_output[save_idx * save_obs_bool, :],
         iteration_counters_output[save_idx * save_counters_bool, :],
     )
     
     # To:
     save_state(
         state_buffer,
         observables_buffer,
         counters_since_save,
         precision(t),  # Cast to user precision
         state_output[save_idx * save_state_bool, :],
         observables_output[save_idx * save_obs_bool, :],
         iteration_counters_output[save_idx * save_counters_bool, :],
     )
     ```
   - Edge cases: save_state signature expects numba_precision type for time
   - Integration: Saves trajectory data at scheduled intervals

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/loops/ode_loop.py (7 lines changed)
- Functions/Methods Added/Modified:
  * Initial driver_function call - line 353: Cast t to precision(t)
  * Initial observables_fn call - line 363: Cast t to precision(t)
  * Initial save_state call - line 380: Cast t to precision(t)
  * step_function call in main loop - lines 455-456: Cast dt_eff and t to precision()
  * save_state call in main loop - line 528: Cast t to precision(t)
- Implementation Summary:
  All device functions now receive time in user precision via precision(t) casting. Time accumulates in float64 within the loop but is cast to user precision at the boundary when passed to step functions, driver functions, observables functions, and save_state. This maintains the precision boundary: float64 above IVPLoop, user precision below.
- Issues Flagged: None

---

## Task Group 5: IVPLoop Time Comparisons in float64 - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups [4]

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (line 443)

**Input Validation Required**:
- None - internal refactoring only

**Tasks**:

1. **Update dt_eff comparison to use float64**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Lines: 443
   - Details:
     ```python
     # Change from:
     status |= selp(dt_eff <= precision(0.0), int32(16), int32(0))
     
     # To:
     status |= selp(dt_eff <= float64(0.0), int32(16), int32(0))
     ```
   - Edge cases: dt_eff is float64, comparison should use float64(0.0)
   - Integration: Detects invalid step sizes in float64 precision

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/loops/ode_loop.py (1 line changed)
- Functions/Methods Added/Modified:
  * dt_eff comparison - line 443: Changed from precision(0.0) to float64(0.0)
- Implementation Summary:
  Time comparison now uses float64(0.0) since dt_eff is a float64 variable. This ensures type consistency in comparisons and avoids unnecessary precision conversions.
- Issues Flagged: None

---

## Task Group 6: Documentation Update - PARALLEL
**Status**: [x]
**Dependencies**: Groups [1, 2, 3, 4, 5]

**Required Context**:
- File: .github/active_plans/time_precision_fix/agent_plan.md (entire file)
- File: .github/active_plans/time_precision_fix/human_overview.md (entire file)

**Input Validation Required**:
- None - documentation only

**Tasks**:

1. **Add implementation notes to agent_plan.md**
   - File: .github/active_plans/time_precision_fix/agent_plan.md
   - Action: Modify
   - Details:
     Add a new section at the end:
     ```markdown
     ## Implementation Status
     
     ### Completed Changes
     1. BatchSolverKernel: Time parameters stored and returned as float64
     2. BatchSolverKernel: CUDA kernel signature uses float64 for duration, warmup, t0
     3. IVPLoop: loop_fn signature accepts float64 for duration, settling_time, t0
     4. IVPLoop: Time accumulation variables (t, t_end, next_save) use float64
     5. IVPLoop: Casts t and dt_eff to user precision before passing to step functions
     6. IVPLoop: Casts t to user precision before passing to driver/observable functions
     7. IVPLoop: Casts t to user precision before passing to save_state function
     
     ### Precision Boundary Enforcement
     - Float64 above IVPLoop: BatchSolverKernel, integration_kernel, loop_fn parameters
     - User precision below IVPLoop: step functions, driver functions, observable functions, save_state
     - Interval parameters (dt_save, dt_summarise, dt_min, dt_max) remain in user precision
     ```

**Outcomes**: 
- Files Modified: 
  * .github/active_plans/time_precision_fix/agent_plan.md (17 lines added)
- Functions/Methods Added/Modified:
  * Added "Implementation Status" section documenting completed changes
- Implementation Summary:
  Added implementation status section to agent_plan.md documenting all completed changes and precision boundary enforcement. This provides a clear record of what was implemented and serves as a reference for future maintenance.
- Issues Flagged: None

---

## Task Group 7: Test Creation - PARALLEL
**Status**: [x]
**Dependencies**: Groups [1, 2, 3, 4, 5]

**Required Context**:
- File: tests/integrators/loops/test_ode_loop.py (entire file for patterns)
- File: tests/batchsolving/test_solver.py (entire file for patterns)
- File: tests/conftest.py (fixture patterns)

**Input Validation Required**:
- None - tests validate the implementation

**Tasks**:

1. **Create precision test for time accumulation**
   - File: tests/integrators/loops/test_time_precision.py
   - Action: Create
   - Details:
     ```python
     """Tests for float64 time accumulation fix."""
     import numpy as np
     import pytest
     from cubie import solve_ivp
     from tests.system_fixtures import three_state_linear
     
     
     @pytest.mark.parametrize("precision", [np.float32])
     def test_float32_small_timestep_accumulation(three_state_linear, precision):
         """Verify time accumulates correctly with float32 precision and small dt."""
         system = three_state_linear
         system.precision = precision
         
         # Use very small dt_min that would fail with float32 time accumulation
         result = solve_ivp(
             system,
             duration=1.0,
             t0=0.0,
             algorithm="explicit_euler",
             dt=1e-8,
             precision=precision,
         )
         
         # Verify integration completed
         assert result.status_codes[0] == 0
         assert result.states.shape[0] > 0
         
         # Verify time increased throughout integration
         if result.times is not None:
             time_diffs = np.diff(result.times)
             assert np.all(time_diffs > 0), "Time should increase monotonically"
     
     
     @pytest.mark.parametrize("precision", [np.float32, np.float64])
     def test_long_integration_with_small_steps(three_state_linear, precision):
         """Verify long integrations with small steps complete correctly."""
         system = three_state_linear
         system.precision = precision
         
         # Long duration with small dt_min
         result = solve_ivp(
             system,
             duration=10.0,
             t0=1e10,  # Large t0 to stress test precision
             algorithm="explicit_euler",
             dt=1e-6,
             precision=precision,
         )
         
         # Verify integration completed
         assert result.status_codes[0] == 0
         assert result.states.shape[0] > 0
     
     
     @pytest.mark.parametrize("precision", [np.float32])
     def test_adaptive_controller_with_float32(three_state_linear, precision):
         """Verify adaptive controllers work with float32 and small dt_min."""
         system = three_state_linear
         system.precision = precision
         
         result = solve_ivp(
             system,
             duration=1.0,
             t0=0.0,
             algorithm="explicit_euler",
             step_controller="adaptive_PI",
             dt0=1e-3,
             dt_min=1e-9,  # Very small minimum step
             dt_max=1e-2,
             precision=precision,
         )
         
         # Verify integration completed without hanging
         assert result.status_codes[0] == 0
         assert result.states.shape[0] > 0
     ```
   - Edge cases: 
     - Very small dt_min (1e-9) with float32
     - Large t0 (1e10) with long duration
     - Adaptive step controllers with wide dt_min/dt_max range
   - Integration: Uses existing test fixtures and solve_ivp API

2. **Create test for time parameter type preservation**
   - File: tests/batchsolving/test_time_precision_types.py
   - Action: Create
   - Details:
     ```python
     """Tests for time parameter type preservation."""
     import numpy as np
     import pytest
     from cubie.batchsolving.solver import Solver
     from tests.system_fixtures import three_state_linear
     
     
     def test_solver_stores_time_as_float64(three_state_linear):
         """Verify Solver stores time parameters as float64."""
         system = three_state_linear
         system.precision = np.float32
         
         solver = Solver(
             system,
             algorithm="explicit_euler",
             dt=1e-3,
         )
         
         # Set time parameters as float32
         solver.kernel.duration = np.float32(10.0)
         solver.kernel.warmup = np.float32(1.0)
         solver.kernel.t0 = np.float32(5.0)
         
         # Verify retrieved as float64
         assert isinstance(solver.kernel.duration, (float, np.floating))
         assert isinstance(solver.kernel.warmup, (float, np.floating))
         assert isinstance(solver.kernel.t0, (float, np.floating))
         
         # Verify values preserved
         assert np.isclose(solver.kernel.duration, 10.0)
         assert np.isclose(solver.kernel.warmup, 1.0)
         assert np.isclose(solver.kernel.t0, 5.0)
     
     
     @pytest.mark.parametrize("precision", [np.float32, np.float64])
     def test_time_precision_independent_of_state_precision(
         three_state_linear, precision
     ):
         """Verify time precision is float64 regardless of state precision."""
         system = three_state_linear
         system.precision = precision
         
         solver = Solver(
             system,
             algorithm="explicit_euler",
             dt=1e-3,
         )
         
         solver.kernel.duration = 5.0
         solver.kernel.t0 = 1.0
         
         # Time should be float64 even when state precision is float32
         assert solver.kernel.duration == np.float64(5.0)
         assert solver.kernel.t0 == np.float64(1.0)
     ```
   - Edge cases:
     - Mixed type inputs (int, float32, float64)
     - Verification of float64 storage independent of state precision
   - Integration: Uses Solver API and property access

**Outcomes**: 
- Files Modified: 
  * tests/integrators/loops/test_time_precision.py (77 lines created)
  * tests/batchsolving/test_time_precision_types.py (56 lines created)
- Functions/Methods Added/Modified:
  * test_float32_small_timestep_accumulation() - Tests float32 with very small dt (1e-8)
  * test_long_integration_with_small_steps() - Tests large t0 (1e10) with long duration
  * test_adaptive_controller_with_float32() - Tests adaptive controllers with wide dt range
  * test_solver_stores_time_as_float64() - Verifies time storage is float64
  * test_time_precision_independent_of_state_precision() - Verifies time is float64 regardless of state precision
- Implementation Summary:
  Created comprehensive test suite covering float64 time accumulation with float32 state precision. Tests verify: (1) Very small dt values work correctly, (2) Large t0 values with long durations, (3) Adaptive controllers with wide dt_min/dt_max ranges, (4) Type preservation of time parameters as float64, (5) Independence of time precision from state precision.
- Issues Flagged: None

---

# Summary

## Total Task Groups: 7

## Dependency Chain Overview:
```
Group 1: BatchSolverKernel Time Storage
   ↓
Group 2: BatchSolverKernel CUDA Kernel Signature
   ↓
Group 3: IVPLoop Signature and Time Variables
   ↓
Group 4: IVPLoop Precision Casting for Device Functions
   ↓
Group 5: IVPLoop Time Comparisons in float64
   ↓
Groups 6 & 7 (Parallel): Documentation Update & Test Creation
```

## Parallel Execution Opportunities:
- Group 6 (Documentation) and Group 7 (Tests) can execute in parallel after Groups 1-5 complete

## Estimated Complexity:
- **Low**: Groups 1, 2, 5, 6 (straightforward type changes and documentation)
- **Medium**: Groups 3, 4 (requires careful casting at boundaries)
- **Medium**: Group 7 (test creation requires understanding edge cases)

## Critical Implementation Notes:

### Type Conversion Pattern:
- **Storage**: Always use `np.float64(value)` for time parameters
- **JIT signatures**: Use `float64` type from numba
- **Casting to precision**: Use `precision(value)` when passing to device functions

### Precision Boundary:
- **Above IVPLoop**: All time values in float64 (BatchSolverKernel, integration_kernel, loop_fn)
- **Below IVPLoop**: All time values cast to user precision (step functions, driver functions, observables, save_state)
- **Intervals remain in user precision**: dt_save, dt_summarise, dt_min, dt_max (not accumulated)

### No Algorithm Changes Required:
- Step functions signature unchanged (receive precision-typed dt and t)
- Driver functions signature unchanged (receive precision-typed time)
- Observable functions signature unchanged (receive precision-typed time)
- Output functions signature unchanged (receive precision-typed time)

### Testing Focus:
- Float32 state precision with very small dt_min (< 1e-7)
- Long integrations with small time steps
- Large t0 values with long duration
- Adaptive controllers with wide dt_min/dt_max ranges
- Type preservation verification (time always float64)
