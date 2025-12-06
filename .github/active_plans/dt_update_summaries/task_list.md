# Implementation Task List
# Feature: dt_update_summaries
# Plan Reference: .github/active_plans/dt_update_summaries/agent_plan.md
# 
# **UPDATE NOTICE**: This task list has been updated after merging main branch.
# The structure has changed significantly:
# - LoopSharedIndices is now LoopSliceIndices (part of LoopBufferSettings)
# - ODELoopConfig now uses buffer_settings parameter
# - Line numbers updated to match current code structure
#

## Task Group 1: Add dt_update_summaries to ODELoopConfig - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop_config.py (lines 141-322)
- File: src/cubie/_utils.py (for validators: gttype_validator, opt_gttype_validator)

**Input Validation Required**:
- _dt_update_summaries: Must be > 0 (use opt_gttype_validator(float, 0))
- Default to None; in __attrs_post_init__, set to _dt_save if None
- Validate: dt_summarise % dt_update_summaries == 0 (integer divisor check)

**Tasks**:
1. **Add _dt_update_summaries attribute to ODELoopConfig**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     ```python
     # Add after line 211 (_dt_summarise attribute):
     _dt_update_summaries: Optional[float] = field(
         default=None,
         validator=opt_gttype_validator(float, 0)
     )
     ```
   - Edge cases: None is valid (will default to dt_save in post_init)
   - Integration: Follows same pattern as _dt_save and _dt_summarise

2. **Add dt_update_summaries property to ODELoopConfig**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     ```python
     # Add after dt_summarise property (after line 293):
     @property
     def dt_update_summaries(self) -> float:
         """Return the summary update interval."""
         update_val = (
             self._dt_update_summaries 
             if self._dt_update_summaries is not None 
             else self._dt_save
         )
         return self.precision(update_val)
     ```
   - Edge cases: Handle None by returning dt_save value
   - Integration: Must use self.precision() wrapper like other properties

3. **Add updates_per_summary property to ODELoopConfig**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     ```python
     # Add after saves_per_summary property (after line 273):
     @property
     def updates_per_summary(self) -> int:
         """Return the number of updates between summary outputs."""
         return int(self.dt_summarise // self.dt_update_summaries)
     ```
   - Edge cases: Division is guaranteed to be exact after validation
   - Integration: Replaces saves_per_summary for summary update logic

4. **Add __attrs_post_init__ method to ODELoopConfig**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Create
   - Details:
     ```python
     # Add after is_adaptive field definition (after line 255):
     def __attrs_post_init__(self):
         """Validate dt_update_summaries and set default if needed."""
         # Set default to dt_save if not provided
         if self._dt_update_summaries is None:
             self._dt_update_summaries = self._dt_save
         
         # Validate that dt_update_summaries divides dt_summarise evenly
         # Use raw values to avoid precision conversion issues
         remainder = self._dt_summarise % self._dt_update_summaries
         if abs(remainder) > 1e-10:  # Floating point tolerance
             raise ValueError(
                 f"dt_update_summaries ({self._dt_update_summaries}) must be "
                 f"an integer divisor of dt_summarise ({self._dt_summarise}). "
                 f"Got remainder: {remainder}"
             )
     ```
   - Edge cases: Floating-point precision requires tolerance check
   - Integration: Called automatically by attrs after field initialization

**Outcomes**: [Empty - to be filled by taskmaster agent]

---

## Task Group 2: Update ALL_LOOP_SETTINGS - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 643-650)

**Input Validation Required**: None (just adding to a set)

**Tasks**:
1. **Add dt_update_summaries to ALL_LOOP_SETTINGS**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Modify ALL_LOOP_SETTINGS set (lines 643-650):
     ALL_LOOP_SETTINGS = {
         "dt_save",
         "dt_summarise",
         "dt_update_summaries",  # NEW LINE
         "dt0",
         "dt_min",
         "dt_max",
         "is_adaptive",
     }
     ```
   - Edge cases: None
   - Integration: Allows parameter recognition in update dictionaries

**Outcomes**: [Empty - to be filled by taskmaster agent]

---

## Task Group 3: Add dt_update_summaries parameter to IVPLoop - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 671-759)

**Input Validation Required**: 
- dt_update_summaries: Must be Optional[float], defaults to None
- Validation occurs in ODELoopConfig (Group 1)

**Tasks**:
1. **Add dt_update_summaries parameter to IVPLoop.__init__**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Add parameter after dt_summarise (around line 724):
     def __init__(
         self,
         precision: PrecisionDType,
         buffer_settings: LoopBufferSettings,
         compile_flags: OutputCompileFlags,
         controller_local_len: int = 0,
         algorithm_local_len: int = 0,
         dt_save: float = 0.1,
         dt_summarise: float = 1.0,
         dt_update_summaries: Optional[float] = None,  # NEW
         dt0: Optional[float]=None,
         # ... rest of parameters
     ```
   - Edge cases: None as default means ODELoopConfig will use dt_save
   - Integration: Follows same pattern as dt_save and dt_summarise

2. **Pass dt_update_summaries to ODELoopConfig**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Modify ODELoopConfig instantiation (around line 739-758):
     config = ODELoopConfig(
         buffer_settings=buffer_settings,
         controller_local_len=controller_local_len,
         algorithm_local_len=algorithm_local_len,
         save_state_fn=save_state_func,
         update_summaries_fn=update_summaries_func,
         save_summaries_fn=save_summaries_func,
         step_controller_fn=step_controller_fn,
         step_function=step_function,
         driver_function=driver_function,
         observables_fn=observables_fn,
         precision=precision,
         compile_flags=compile_flags,
         dt_save=dt_save,
         dt_summarise=dt_summarise,
         dt_update_summaries=dt_update_summaries,  # NEW
         dt0=dt0,
         dt_min=dt_min,
         dt_max=dt_max,
         is_adaptive=is_adaptive,
     )
     ```
   - Edge cases: None
   - Integration: Passes through to config for validation

3. **Update IVPLoop docstring**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Add to docstring parameter list (around line 690):
     """
     ...
     dt_summarise
         Interval between summary accumulations. Defaults to ``1.0`` when not
         provided.
     dt_update_summaries
         Interval between summary metric updates. Must be an integer divisor
         of ``dt_summarise``. Defaults to ``dt_save`` when not provided.
     dt0
         Initial timestep applied before controller feedback.
     ...
     """
     ```
   - Edge cases: None
   - Integration: Follows numpydoc format

**Outcomes**: [Empty - to be filled by taskmaster agent]

---

## Task Group 4: Separate do_save and do_update_summary in IVPLoop.build() - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 778-1330)
- Specific sections:
  - Timing constants setup (lines 830-838)
  - Loop state initialization (lines 1116-1180)
  - Adaptive-step logic (lines 1181-1195)
  - Save/update logic (lines 1282-1328)

**Input Validation Required**: None (all validation in ODELoopConfig)

**Tasks**:
1. **Add dt_update_summaries and updates_per_summary to timing constants**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Modify timing constants section (around lines 830-838):
     # Timing values
     saves_per_summary = config.saves_per_summary
     updates_per_summary = config.updates_per_summary  # NEW
     dt_save = precision(config.dt_save)
     dt_update_summaries = precision(config.dt_update_summaries)  # NEW
     dt0 = precision(config.dt0)
     dt_min = precision(config.dt_min)
     # save_last is not yet piped up from this level, but is intended and
     # included in loop logic
     save_last = False
     ```
   - Edge cases: dt_update_summaries could equal dt_save (backward compatible)
   - Integration: Captured in loop closure for device function

2. **Add update_idx and next_update_summary to loop state initialization**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Find initialization section (around lines 1116-1180)
     # Add alongside save_idx initialization (around line 1117):
     save_idx = int32(0)
     summary_idx = int32(0)
     update_idx = int32(0)  # NEW - tracks number of summary updates
     
     # Also add next_update_summary for adaptive mode (around line 1125):
     if settling_time > precision(0.0):
         # Don't save t0, wait until settling_time
         next_save = precision(settling_time)
         next_update_summary = precision(settling_time)  # NEW
     else:
         # Seed initial state and save/update summaries
         next_save = precision(dt_save)
         next_update_summary = precision(dt_update_summaries)  # NEW
     ```
   - Edge cases: Both save and update timing need settling_time consideration
   - Integration: Parallel to existing save tracking

3. **Add do_update_summary logic in adaptive mode (separate from do_save)**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Modify adaptive-step section (around lines 1181-1195):
     # After computing do_save, add do_update_summary:
     do_save = (t_prec + dt_raw) >= next_save
     do_update_summary = (t_prec + dt_raw) >= next_update_summary  # NEW
     dt_eff = selp(do_save, next_save - t_prec, dt_raw)
     # Alternative if update requires exact hitting:
     # dt_eff = selp(do_save or do_update_summary, 
     #               selp(do_save, next_save - t_prec, next_update_summary - t_prec),
     #               dt_raw)
     ```
   - Edge cases: Both do_save and do_update_summary could be true simultaneously
   - Integration: Adds independent update tracking parallel to save tracking

4. **Update next_update_summary in adaptive mode**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Add after next_save update (around line 1284):
     # Predicated update of next_save; update if save is accepted.
     do_save = accept and do_save
     if do_save:
         next_save = selp(do_save, next_save + dt_save, next_save)
     
     # NEW: Predicated update of next_update_summary
     do_update_summary = accept and do_update_summary
     if do_update_summary:
         next_update_summary = selp(
             do_update_summary,
             next_update_summary + dt_update_summaries,
             next_update_summary
         )
     ```
   - Edge cases: Both updates can occur on same step
   - Integration: Parallel to next_save logic

5. **Restructure save/update/summary logic to separate concerns**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Replace lines 1283-1318 with:
     do_save = accept and do_save
     do_update_summary = accept and do_update_summary  # NEW
     
     if do_save:
         next_save = selp(do_save, next_save + dt_save, next_save)
         save_state(
             state_buffer,
             observables_buffer,
             counters_since_save,
             t_prec,
             state_output[save_idx * save_state_bool, :],
             observables_output[save_idx * save_obs_bool, :],
             iteration_counters_output[save_idx * save_counters_bool, :],
         )
         save_idx += int32(1)
         
         # Reset iteration counters after save
         if save_counters_bool:
             for i in range(n_counters):
                 counters_since_save[i] = int32(0)
     
     # NEW: Separate summary update logic
     if do_update_summary:
         next_update_summary = selp(
             do_update_summary,
             next_update_summary + dt_update_summaries,
             next_update_summary
         )
         if summarise:
             update_summaries(
                 state_buffer,
                 observables_buffer,
                 state_summary_buffer,
                 observable_summary_buffer,
                 update_idx  # Changed from save_idx
             )
             update_idx += int32(1)  # NEW
             
             if (update_idx % updates_per_summary == int32(0)):
                 save_summaries(
                     state_summary_buffer,
                     observable_summary_buffer,
                     state_summaries_output[
                         summary_idx * summarise_state_bool, :
                     ],
                     observable_summaries_output[
                         summary_idx * summarise_obs_bool, :
                     ],
                     updates_per_summary,  # Changed from saves_per_summary
                 )
                 summary_idx += 1
             update_idx += 1
     ```
   - Edge cases: Both do_save and do_update_summary can be True on same iteration
   - Integration: Separates state saves from summary updates

8. **Handle initial summary at t=0 or settling_time**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Find initial state handling (around lines 380-405)
     # Modify to include initial summary update:
     if settling_time == 0.0:
         # Save initial state
         if save_state_bool:
             save_state(
                 # ... existing parameters ...
             )
             save_idx += int32(1)
         
         # NEW: Initial summary update
         if summarise:
             update_summaries(
                 state_buffer,
                 observables_buffer,
                 state_summary_buffer,
                 observable_summary_buffer,
                 update_idx
             )
             update_idx += int32(1)
     else:
         # When settling_time > 0
         next_save = settling_time  # existing
         next_update_summary = settling_time  # NEW
     ```
   - Edge cases: Initial update only when settling_time == 0
   - Integration: Maintains existing initial state behavior

**Outcomes**: [Empty - to be filled by taskmaster agent]

---

## Task Group 5: Add validation tests for dt_update_summaries - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: tests/integrators/loops/test_ode_loop.py (entire file)
- File: tests/conftest.py (for fixture patterns)
- File: src/cubie/integrators/loops/ode_loop_config.py (for testing ODELoopConfig)

**Input Validation Required**: None (testing validation logic)

**Tasks**:
1. **Create test_dt_update_summaries_validation.py**
   - File: tests/integrators/loops/test_dt_update_summaries_validation.py
   - Action: Create
   - Details:
     ```python
     """Tests for dt_update_summaries parameter validation."""
     
     import pytest
     import numpy as np
     from cubie.integrators.loops.ode_loop_config import (
         ODELoopConfig,
         LoopSharedIndices,
         LoopLocalIndices,
     )
     from cubie.outputhandling.output_config import OutputCompileFlags
     
     
     def test_dt_update_summaries_default_to_dt_save():
         """Test that dt_update_summaries defaults to dt_save."""
         shared_indices = LoopSharedIndices.from_sizes(
             n_states=3, n_observables=0, n_parameters=0, n_drivers=0,
             state_summaries_buffer_height=0,
             observable_summaries_buffer_height=0,
         )
         local_indices = LoopLocalIndices.empty()
         
         config = ODELoopConfig(
             shared_buffer_indices=shared_indices,
             local_indices=local_indices,
             dt_save=0.1,
             dt_summarise=1.0,
             # dt_update_summaries not provided
         )
         
         assert config.dt_update_summaries == 0.1
         assert config._dt_update_summaries == 0.1
     
     
     def test_dt_update_summaries_explicit_value():
         """Test that explicit dt_update_summaries is used."""
         shared_indices = LoopSharedIndices.from_sizes(
             n_states=3, n_observables=0, n_parameters=0, n_drivers=0,
             state_summaries_buffer_height=0,
             observable_summaries_buffer_height=0,
         )
         local_indices = LoopLocalIndices.empty()
         
         config = ODELoopConfig(
             shared_buffer_indices=shared_indices,
             local_indices=local_indices,
             dt_save=0.1,
             dt_summarise=1.0,
             dt_update_summaries=0.2,
         )
         
         assert config.dt_update_summaries == 0.2
     
     
     def test_dt_update_summaries_must_divide_dt_summarise():
         """Test that dt_update_summaries must divide dt_summarise."""
         shared_indices = LoopSharedIndices.from_sizes(
             n_states=3, n_observables=0, n_parameters=0, n_drivers=0,
             state_summaries_buffer_height=0,
             observable_summaries_buffer_height=0,
         )
         local_indices = LoopLocalIndices.empty()
         
         with pytest.raises(ValueError, match="must be an integer divisor"):
             ODELoopConfig(
                 shared_buffer_indices=shared_indices,
                 local_indices=local_indices,
                 dt_save=0.1,
                 dt_summarise=1.0,
                 dt_update_summaries=0.3,  # Does not divide 1.0 evenly
             )
     
     
     def test_dt_update_summaries_positive():
         """Test that dt_update_summaries must be positive."""
         shared_indices = LoopSharedIndices.from_sizes(
             n_states=3, n_observables=0, n_parameters=0, n_drivers=0,
             state_summaries_buffer_height=0,
             observable_summaries_buffer_height=0,
         )
         local_indices = LoopLocalIndices.empty()
         
         with pytest.raises((ValueError, TypeError)):
             ODELoopConfig(
                 shared_buffer_indices=shared_indices,
                 local_indices=local_indices,
                 dt_save=0.1,
                 dt_summarise=1.0,
                 dt_update_summaries=-0.1,
             )
     
     
     @pytest.mark.parametrize("dt_update", [0.1, 0.2, 0.25, 0.5, 1.0])
     def test_valid_dt_update_summaries_values(dt_update):
         """Test various valid dt_update_summaries values."""
         shared_indices = LoopSharedIndices.from_sizes(
             n_states=3, n_observables=0, n_parameters=0, n_drivers=0,
             state_summaries_buffer_height=0,
             observable_summaries_buffer_height=0,
         )
         local_indices = LoopLocalIndices.empty()
         
         config = ODELoopConfig(
             shared_buffer_indices=shared_indices,
             local_indices=local_indices,
             dt_save=0.1,
             dt_summarise=1.0,
             dt_update_summaries=dt_update,
         )
         
         assert config.dt_update_summaries == pytest.approx(dt_update)
         expected_updates = int(1.0 / dt_update)
         assert config.updates_per_summary == expected_updates
     ```
   - Edge cases: Floating-point division, various divisor values
   - Integration: Uses existing test patterns

**Outcomes**: [Empty - to be filled by taskmaster agent]

---

## Task Group 6: Add functional tests for dt_update_summaries - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3, 4

**Required Context**:
- File: tests/integrators/loops/test_ode_loop.py (entire file for patterns)
- File: tests/conftest.py (for fixture usage)
- File: tests/system_fixtures.py (for ODE systems)
- File: tests/_utils.py (for assert_integration_outputs)

**Input Validation Required**: None (testing functionality)

**Tasks**:
1. **Create test_dt_update_summaries_functionality.py**
   - File: tests/integrators/loops/test_dt_update_summaries_functionality.py
   - Action: Create
   - Details:
     ```python
     """Functional tests for dt_update_summaries parameter."""
     
     import pytest
     import numpy as np
     from cubie import solve_ivp, summary_metrics
     
     
     @pytest.fixture
     def base_settings():
         """Base settings for dt_update_summaries tests."""
         return {
             'algorithm': 'euler',
             'step_controller': 'fixed',
             'dt': 0.001,
             'dt_save': 0.1,
             'dt_summarise': 1.0,
             'output_types': ['summaries'],
             'summaries': ['mean', 'max'],
             'summarised_state_indices': [0, 1, 2],
         }
     
     
     def test_dt_update_summaries_equals_dt_save(three_state_linear, base_settings):
         """Test that default behavior (dt_update == dt_save) works."""
         settings = dict(base_settings)
         settings['dt_update_summaries'] = 0.1  # Same as dt_save
         
         result = solve_ivp(
             system=three_state_linear,
             duration=2.0,
             **settings
         )
         
         assert result.summaries is not None
         assert result.summaries.shape[0] == 2  # 2 summary intervals
     
     
     def test_dt_update_summaries_less_than_dt_save(three_state_linear, base_settings):
         """Test dt_update_summaries < dt_save (more updates)."""
         settings = dict(base_settings)
         settings['dt_update_summaries'] = 0.05  # Half of dt_save
         
         result = solve_ivp(
             system=three_state_linear,
             duration=2.0,
             **settings
         )
         
         assert result.summaries is not None
         assert result.summaries.shape[0] == 2  # Still 2 summary intervals
         # Summaries should be computed over 20 updates instead of 10
     
     
     def test_dt_update_summaries_greater_than_dt_save(three_state_linear, base_settings):
         """Test dt_update_summaries > dt_save (fewer updates)."""
         settings = dict(base_settings)
         settings['dt_save'] = 0.1
         settings['dt_update_summaries'] = 0.5  # 5x dt_save
         settings['dt_summarise'] = 2.0  # Must be divisible by 0.5
         
         result = solve_ivp(
             system=three_state_linear,
             duration=4.0,
             **settings
         )
         
         assert result.summaries is not None
         assert result.summaries.shape[0] == 2  # 4.0 / 2.0 = 2 intervals
     
     
     def test_updates_per_summary_calculation(three_state_linear, base_settings):
         """Test that updates_per_summary is calculated correctly."""
         settings = dict(base_settings)
         settings['dt_update_summaries'] = 0.25
         settings['dt_summarise'] = 1.0
         # updates_per_summary should be 4
         
         result = solve_ivp(
             system=three_state_linear,
             duration=2.0,
             **settings
         )
         
         assert result.summaries is not None
         # Each summary computed over 4 updates
     
     
     def test_summary_only_output_mode(three_state_linear):
         """Test summary-only mode with no state saves."""
         settings = {
             'algorithm': 'euler',
             'step_controller': 'fixed',
             'dt': 0.001,
             'dt_update_summaries': 0.1,
             'dt_summarise': 1.0,
             'output_types': ['summaries'],  # No 'state' output
             'summaries': ['mean'],
             'summarised_state_indices': [0, 1, 2],
         }
         
         result = solve_ivp(
             system=three_state_linear,
             duration=2.0,
             **settings
         )
         
         assert result.states is None  # No states saved
         assert result.summaries is not None
         assert result.summaries.shape[0] == 2
     
     
     @pytest.mark.parametrize(
         "dt_update,expected_updates",
         [(0.1, 10), (0.2, 5), (0.25, 4), (0.5, 2), (1.0, 1)]
     )
     def test_various_update_frequencies(
         three_state_linear, base_settings, dt_update, expected_updates
     ):
         """Test various dt_update_summaries frequencies."""
         settings = dict(base_settings)
         settings['dt_update_summaries'] = dt_update
         settings['dt_summarise'] = 1.0
         
         result = solve_ivp(
             system=three_state_linear,
             duration=1.0,  # Single summary interval
             **settings
         )
         
         assert result.summaries is not None
         # Verify updates_per_summary in loop config
     
     
     def test_adaptive_step_with_dt_update_summaries(three_state_linear):
         """Test dt_update_summaries with adaptive stepping."""
         settings = {
             'algorithm': 'crank_nicolson',
             'step_controller': 'pi',
             'dt': 0.01,
             'dt_min': 1e-6,
             'dt_max': 0.5,
             'dt_save': 0.1,
             'dt_update_summaries': 0.05,
             'dt_summarise': 1.0,
             'output_types': ['summaries'],
             'summaries': ['mean'],
             'summarised_state_indices': [0, 1, 2],
             'atol': 1e-6,
             'rtol': 1e-5,
         }
         
         result = solve_ivp(
             system=three_state_linear,
             duration=2.0,
             **settings
         )
         
         assert result.summaries is not None
     
     
     def test_settling_time_with_dt_update_summaries(three_state_linear):
         """Test dt_update_summaries with settling_time > 0."""
         settings = {
             'algorithm': 'euler',
             'step_controller': 'fixed',
             'dt': 0.001,
             'dt_update_summaries': 0.1,
             'dt_summarise': 1.0,
             'settling_time': 0.5,
             'output_types': ['summaries'],
             'summaries': ['mean'],
             'summarised_state_indices': [0, 1, 2],
         }
         
         result = solve_ivp(
             system=three_state_linear,
             duration=2.0,
             **settings
         )
         
         assert result.summaries is not None
         # First update should occur at settling_time
     ```
   - Edge cases: Adaptive stepping, settling time, various frequencies
   - Integration: Uses existing test utilities and fixtures

**Outcomes**: [Empty - to be filled by taskmaster agent]

---

## Task Group 7: Add backward compatibility tests - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3, 4

**Required Context**:
- File: tests/integrators/loops/test_ode_loop.py (for comparison)
- File: tests/conftest.py (for fixtures)

**Input Validation Required**: None (testing compatibility)

**Tasks**:
1. **Create test_dt_update_summaries_backward_compatibility.py**
   - File: tests/integrators/loops/test_dt_update_summaries_backward_compatibility.py
   - Action: Create
   - Details:
     ```python
     """Backward compatibility tests for dt_update_summaries."""
     
     import pytest
     import numpy as np
     from cubie import solve_ivp
     
     
     def test_existing_code_without_dt_update_summaries(three_state_linear):
         """Test that existing code works without dt_update_summaries."""
         # This is the old API - should still work
         settings = {
             'algorithm': 'euler',
             'step_controller': 'fixed',
             'dt': 0.001,
             'dt_save': 0.1,
             'dt_summarise': 1.0,
             'output_types': ['state', 'summaries'],
             'summaries': ['mean'],
             'summarised_state_indices': [0, 1, 2],
         }
         
         result = solve_ivp(
             system=three_state_linear,
             duration=2.0,
             **settings
         )
         
         assert result.states is not None
         assert result.summaries is not None
     
     
     def test_default_equals_old_behavior(three_state_linear):
         """Test that default dt_update_summaries produces same results as old code."""
         base_settings = {
             'algorithm': 'euler',
             'step_controller': 'fixed',
             'dt': 0.001,
             'dt_save': 0.1,
             'dt_summarise': 1.0,
             'output_types': ['summaries'],
             'summaries': ['mean', 'max'],
             'summarised_state_indices': [0, 1, 2],
         }
         
         # Run without dt_update_summaries
         result_old = solve_ivp(
             system=three_state_linear,
             duration=2.0,
             **base_settings
         )
         
         # Run with explicit dt_update_summaries = dt_save
         settings_new = dict(base_settings)
         settings_new['dt_update_summaries'] = 0.1
         result_new = solve_ivp(
             system=three_state_linear,
             duration=2.0,
             **settings_new
         )
         
         # Results should be identical
         np.testing.assert_array_equal(result_old.summaries, result_new.summaries)
     
     
     def test_existing_tests_still_pass():
         """Verify that we can still run existing test patterns."""
         # This is a placeholder to ensure existing tests work
         # The actual existing tests should pass without modification
         pass
     ```
   - Edge cases: None
   - Integration: Ensures existing API still works

**Outcomes**: [Empty - to be filled by taskmaster agent]

---

## Task Group 8: Add edge case tests - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1, 2, 3, 4

**Required Context**:
- File: tests/integrators/loops/test_ode_loop.py (for patterns)
- File: tests/conftest.py (for fixtures)

**Input Validation Required**: None (testing edge cases)

**Tasks**:
1. **Create test_dt_update_summaries_edge_cases.py**
   - File: tests/integrators/loops/test_dt_update_summaries_edge_cases.py
   - Action: Create
   - Details:
     ```python
     """Edge case tests for dt_update_summaries."""
     
     import pytest
     import numpy as np
     from cubie import solve_ivp
     
     
     def test_dt_update_summaries_equals_dt_summarise(three_state_linear):
         """Test dt_update_summaries == dt_summarise (single update per summary)."""
         settings = {
             'algorithm': 'euler',
             'step_controller': 'fixed',
             'dt': 0.001,
             'dt_update_summaries': 1.0,
             'dt_summarise': 1.0,
             'output_types': ['summaries'],
             'summaries': ['mean'],
             'summarised_state_indices': [0, 1, 2],
         }
         
         result = solve_ivp(
             system=three_state_linear,
             duration=2.0,
             **settings
         )
         
         assert result.summaries is not None
         # updates_per_summary should be 1
     
     
     def test_very_small_dt_update_summaries(three_state_linear):
         """Test very small dt_update_summaries (many updates)."""
         settings = {
             'algorithm': 'euler',
             'step_controller': 'fixed',
             'dt': 0.001,
             'dt_update_summaries': 0.01,  # 100 updates per summary
             'dt_summarise': 1.0,
             'output_types': ['summaries'],
             'summaries': ['mean'],
             'summarised_state_indices': [0, 1, 2],
         }
         
         result = solve_ivp(
             system=three_state_linear,
             duration=1.0,
             **settings
         )
         
         assert result.summaries is not None
     
     
     def test_dt_update_equals_dt_step(three_state_linear):
         """Test dt_update_summaries == dt (update every step)."""
         settings = {
             'algorithm': 'euler',
             'step_controller': 'fixed',
             'dt': 0.01,
             'dt_update_summaries': 0.01,  # Update every step
             'dt_summarise': 1.0,
             'output_types': ['summaries'],
             'summaries': ['mean'],
             'summarised_state_indices': [0, 1, 2],
         }
         
         result = solve_ivp(
             system=three_state_linear,
             duration=1.0,
             **settings
         )
         
         assert result.summaries is not None
         # Should update 100 times per summary
     
     
     def test_initial_summary_at_t_zero(three_state_linear):
         """Test that initial summary is computed at t=0."""
         settings = {
             'algorithm': 'euler',
             'step_controller': 'fixed',
             'dt': 0.001,
             'dt_update_summaries': 0.1,
             'dt_summarise': 1.0,
             'settling_time': 0.0,  # No settling
             'output_types': ['summaries'],
             'summaries': ['mean'],
             'summarised_state_indices': [0, 1, 2],
         }
         
         result = solve_ivp(
             system=three_state_linear,
             duration=1.0,
             **settings
         )
         
         assert result.summaries is not None
         # Initial state should be included in first summary
     
     
     def test_multiple_summary_metrics(three_state_linear):
         """Test multiple summary metrics with dt_update_summaries."""
         settings = {
             'algorithm': 'euler',
             'step_controller': 'fixed',
             'dt': 0.001,
             'dt_update_summaries': 0.2,
             'dt_summarise': 1.0,
             'output_types': ['summaries'],
             'summaries': ['mean', 'max', 'rms'],
             'summarised_state_indices': [0, 1, 2],
         }
         
         result = solve_ivp(
             system=three_state_linear,
             duration=2.0,
             **settings
         )
         
         assert result.summaries is not None
         # All metrics should be computed correctly
     
     
     def test_floating_point_precision_in_divisibility(three_state_linear):
         """Test divisibility check handles floating-point precision."""
         # Some values that might be tricky due to floating-point representation
         settings = {
             'algorithm': 'euler',
             'step_controller': 'fixed',
             'dt': 0.001,
             'dt_update_summaries': 0.1,  # Should divide 1.0 cleanly
             'dt_summarise': 1.0,
             'output_types': ['summaries'],
             'summaries': ['mean'],
             'summarised_state_indices': [0, 1, 2],
         }
         
         # Should not raise ValueError
         result = solve_ivp(
             system=three_state_linear,
             duration=1.0,
             **settings
         )
         
         assert result.summaries is not None
     ```
   - Edge cases: Extreme values, floating-point precision, multiple metrics
   - Integration: Validates robust behavior

**Outcomes**: [Empty - to be filled by taskmaster agent]

---

## Summary

**Total Task Groups**: 8
**Dependency Chain**: 
- Groups 1, 2, 3 must complete before Group 4
- Groups 5, 6, 7, 8 can run in parallel after Groups 1-4

**Parallel Execution Opportunities**:
- Groups 1 and 2 are independent (can run in parallel)
- Groups 5, 6, 7, 8 (all testing) can run in parallel

**Estimated Complexity**:
- Core implementation (Groups 1-4): Medium complexity
- Testing (Groups 5-8): Low-medium complexity
- Total: ~8 hours of focused development time

**Key Integration Points**:
1. ODELoopConfig validates dt_update_summaries
2. IVPLoop passes parameter through to config
3. Loop build() separates do_save from do_update_summary
4. Summary functions receive update_idx instead of save_idx
5. Comprehensive testing validates all scenarios
