# Implementation Task List
# Feature: Refactor Integrator Loop Timing Parameters
# Plan Reference: .github/active_plans/refactor_loop_timing/agent_plan.md

## Task Group 1: ODELoopConfig Enhancement - Duration Field and samples_per_summary
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop_config.py (entire file)
- File: src/cubie/_utils.py (lines 1-50, for opt_gttype_validator)

**Input Validation Required**:
- _duration: Optional[float], validate > 0 when not None (use opt_gttype_validator)

**Tasks**:
1. **Add _duration field to ODELoopConfig**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     ```python
     # Add after _sample_summaries_every field (around line 192)
     _duration: Optional[float] = field(
         default=None,
         validator=opt_gttype_validator(float, 0)
     )
     ```
   - Edge cases: None is valid (duration not yet known)
   - Integration: Used by samples_per_summary property when summarise_last=True

2. **Add duration property to ODELoopConfig**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     ```python
     # Add after sample_summaries_every property (around line 375)
     @property
     def duration(self) -> Optional[float]:
         """Return the integration duration, or None if not configured."""
         if self._duration is None:
             return None
         return self.precision(self._duration)
     ```
   - Edge cases: Returns None when not set
   - Integration: Accessed by samples_per_summary calculation

3. **Enhance samples_per_summary property for duration fallback**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     ```python
     @property
     def samples_per_summary(self) -> Optional[int]:
         """Return the number of updates between summary outputs.
         
         When summarise_every is None but summarise_last is True,
         defaults to duration/100 samples if duration is available.
         """
         if self._summarise_every is None:
             if self._duration is not None and self.summarise_last:
                 # Default to 100 samples when using summarise_last
                 return max(1, int(self._duration / 100))
             return None
         return round(self.summarise_every / self.sample_summaries_every)
     ```
   - Edge cases: 
     - duration=None returns None
     - Very short duration returns at least 1
   - Integration: Used by IVPLoop to determine summary aggregation frequency

4. **Add 'duration' to ALL_LOOP_SETTINGS**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # Add 'duration' to the ALL_LOOP_SETTINGS set (around line 36-61)
     ALL_LOOP_SETTINGS = {
         "save_every",
         "summarise_every",
         "sample_summaries_every",
         "duration",  # Add this line
         "dt0",
         ...
     }
     ```
   - Edge cases: None
   - Integration: Allows duration to be passed through update chain

**Tests to Create**:
- Test file: tests/integrators/loops/test_ode_loop_config_timing.py
- Test function: test_samples_per_summary_with_summarise_last_and_duration
- Description: Verify that samples_per_summary returns duration/100 when summarise_last=True and duration is set
- Test function: test_samples_per_summary_without_duration_returns_none
- Description: Verify that samples_per_summary returns None when summarise_last=True but duration is not set
- Test function: test_duration_property
- Description: Verify duration property returns precision-cast value or None

**Tests to Run**:
- tests/integrators/loops/test_ode_loop_config_timing.py::test_samples_per_summary_with_summarise_last_and_duration
- tests/integrators/loops/test_ode_loop_config_timing.py::test_samples_per_summary_without_duration_returns_none
- tests/integrators/loops/test_ode_loop_config_timing.py::test_duration_property

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: BatchSolverKernel None-Safe Output Length Properties
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 942-975, output_length and summaries_length properties)
- File: src/cubie/integrators/SingleIntegratorRun.py (entire file, for understanding property chain)
- File: src/cubie/integrators/loops/ode_loop.py (lines 837-850, save_every/summarise_every properties)
- File: src/cubie/integrators/loops/ode_loop_config.py (lines 195-202, save_last/summarise_last fields)

**Input Validation Required**:
- None (properties only read existing validated values)

**Tasks**:
1. **Make output_length property None-safe**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     @property
     def output_length(self) -> int:
         """Number of saved trajectory samples in the main run.
         
         Includes both initial state (at t=t0 or t=settling_time) and final
         state (at t=t_end) for complete trajectory coverage.
         
         When save_every is None (save_last mode), returns 2 for initial
         and final state capture only.
         """
         save_every = self.single_integrator.save_every
         if save_every is None:
             # save_last mode: initial + final = 2
             return 2
         return (int(
                 np_floor(self.precision(self.duration) /
                         self.precision(save_every)))
                 + 1)
     ```
   - Edge cases: 
     - save_every=None with save_last=True returns 2
     - save_every=0 should not occur (validated elsewhere)
   - Integration: Used by SingleRunOutputSizes.from_solver() and BatchOutputSizes.from_solver()

2. **Make summaries_length property None-safe**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     @property
     def summaries_length(self) -> int:
         """Number of complete summary intervals across the integration window.
         
         When summarise_every is None (summarise_last mode), returns 2 for
         initial and final summary capture only.
         
         For periodic summaries, counts only complete summarise_every periods
         using floor division. No summary is recorded for t=0 and partial
         intervals at the tail of integration are excluded.
         """
         summarise_every = self.single_integrator.summarise_every
         if summarise_every is None:
             # summarise_last mode: initial + final = 2
             return 2
         precision = self.precision
         return int(precision(self._duration) / precision(summarise_every))
     ```
   - Edge cases:
     - summarise_every=None with summarise_last=True returns 2
     - summarise_every=0 should not occur (validated elsewhere)
   - Integration: Used by SingleRunOutputSizes.from_solver() for summary array allocation

3. **Make warmup_length property None-safe**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     @property
     def warmup_length(self) -> int:
         """Number of warmup save intervals completed before capturing output.
         
         Returns 0 when save_every is None (save_last mode only).
         
         Note: Warmup uses ceil(warmup/save_every) WITHOUT the +1 because warmup
         saves are transient and discarded after settling. The final warmup
         state becomes the initial state of the main run, so there is no need
         to save both endpoints in the warmup phase.
         """
         save_every = self.single_integrator.save_every
         if save_every is None:
             return 0
         return int(np_ceil(self._warmup / save_every))
     ```
   - Edge cases: warmup=0 returns 0
   - Integration: Used for warmup buffer sizing

4. **Make save_every property None-safe (return Optional)**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     @property
     def save_every(self) -> Optional[float]:
         """Interval between saved samples from the loop, or None if save_last only."""
         return self.single_integrator.save_every
     ```
   - Edge cases: None when save_last mode only
   - Integration: Exposed to solver and upstream components

5. **Make summarise_every property None-safe (return Optional)**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     @property
     def summarise_every(self) -> Optional[float]:
         """Interval between summary reductions from the loop, or None if summarise_last only."""
         return self.single_integrator.summarise_every
     ```
   - Edge cases: None when summarise_last mode only
   - Integration: Exposed to solver and upstream components

**Tests to Create**:
- Test file: tests/batchsolving/test_kernel_output_lengths.py
- Test function: test_output_length_with_save_every_none
- Description: Verify output_length returns 2 when save_every is None
- Test function: test_output_length_with_periodic_save
- Description: Verify output_length calculation with explicit save_every
- Test function: test_summaries_length_with_summarise_every_none
- Description: Verify summaries_length returns 2 when summarise_every is None
- Test function: test_summaries_length_with_periodic_summarise
- Description: Verify summaries_length calculation with explicit summarise_every
- Test function: test_warmup_length_with_save_every_none
- Description: Verify warmup_length returns 0 when save_every is None

**Tests to Run**:
- tests/batchsolving/test_kernel_output_lengths.py::test_output_length_with_save_every_none
- tests/batchsolving/test_kernel_output_lengths.py::test_output_length_with_periodic_save
- tests/batchsolving/test_kernel_output_lengths.py::test_summaries_length_with_summarise_every_none
- tests/batchsolving/test_kernel_output_lengths.py::test_summaries_length_with_periodic_summarise
- tests/batchsolving/test_kernel_output_lengths.py::test_warmup_length_with_save_every_none

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: Parameter Reset Mechanism for Timing Parameters
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop_config.py (entire file, focus on __attrs_post_init__)
- File: src/cubie/integrators/loops/ode_loop.py (lines 950-1001, update method)
- File: src/cubie/CUDAFactory.py (entire file, for update_compile_settings behavior)

**Input Validation Required**:
- None (mechanism change, not new validation)

**Tasks**:
1. **Add reset_timing_inference method to ODELoopConfig**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     ```python
     # Add after __attrs_post_init__ method
     def reset_timing_inference(self) -> None:
         """Reset inferred timing values and re-run inference logic.
         
         Call this method after updating timing parameters to None to
         ensure derived values are recalculated rather than preserved.
         This enables proper parameter reset on subsequent solves.
         """
         # Reset flags to defaults before re-inferring
         self.save_last = False
         self.summarise_last = False
         # Re-run inference logic
         self.__attrs_post_init__()
     ```
   - Edge cases: Should handle all combinations of None/value timing params
   - Integration: Called by IVPLoop.update when timing params change

2. **Modify IVPLoop.update to handle None timing parameter reset**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # In the update method, after update_compile_settings call (around line 987)
     # Check if any timing parameter was explicitly set to None
     timing_params = {'save_every', 'summarise_every', 'sample_summaries_every'}
     timing_reset_needed = any(
         key in updates_dict and updates_dict[key] is None
         for key in timing_params
     )
     
     recognised = self.update_compile_settings(updates_dict, silent=True)
     
     # If timing params were reset to None, re-run inference
     if timing_reset_needed:
         self.compile_settings.reset_timing_inference()
         # Cache invalidated by timing changes
         self._invalidate_cache()
     ```
   - Edge cases: 
     - Only reset when None is explicitly passed, not when key is absent
     - Preserve non-timing updates that occurred in same call
   - Integration: Ensures fresh calculation on each solve when timing params are None

**Tests to Create**:
- Test file: tests/integrators/loops/test_ode_loop_config_timing.py
- Test function: test_timing_parameter_reset_on_none
- Description: Verify that setting sample_summaries_every to None after a previous value recalculates it
- Test function: test_reset_timing_inference_clears_flags
- Description: Verify reset_timing_inference resets save_last and summarise_last before re-inferring
- Test function: test_update_with_none_timing_triggers_reset
- Description: Verify IVPLoop.update with timing param=None triggers reset_timing_inference

**Tests to Run**:
- tests/integrators/loops/test_ode_loop_config_timing.py::test_timing_parameter_reset_on_none
- tests/integrators/loops/test_ode_loop_config_timing.py::test_reset_timing_inference_clears_flags
- tests/integrators/loops/test_ode_loop_config_timing.py::test_update_with_none_timing_triggers_reset

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: Duration Propagation Through Update Chain
**Status**: [ ]
**Dependencies**: Task Group 1, Task Group 3

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 264-358, run method; lines 707-781, update method)
- File: src/cubie/integrators/SingleIntegratorRunCore.py (lines 357-464, update method)
- File: src/cubie/integrators/loops/ode_loop.py (lines 950-1001, update method)

**Input Validation Required**:
- duration: float, > 0 (already validated in kernel.run)

**Tasks**:
1. **Propagate duration through kernel.run to loop config**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # In run method, after setting self._duration (around line 321-324)
     # Add duration to the update call that refreshes compile settings
     self.update_compile_settings(
         {
             "loop_fn": self.single_integrator.compiled_loop_function,
             "precision": self.single_integrator.precision,
             "local_memory_elements": (
                 self.single_integrator.local_memory_elements
             ),
             "shared_memory_elements": (
                 self.single_integrator.shared_memory_elements
             ),
         }
     )
     
     # Propagate duration to single_integrator for loop config
     self.single_integrator.update({"duration": duration}, silent=True)
     ```
   - Edge cases: duration=0 should be caught by earlier validation
   - Integration: Flows duration to ODELoopConfig for samples_per_summary calculation

2. **Thread duration through SingleIntegratorRunCore.update**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     # In update method, ensure duration is passed to loop
     # The existing loop update call already passes updates_dict which
     # will include duration if provided. No code change needed if
     # updates_dict already contains duration - verify flow.
     
     # Verify that loop_recognized includes 'duration' in recognized params
     loop_recognized = self._loop.update(updates_dict, silent=True)
     ```
   - Edge cases: Duration should flow through even without other timing changes
   - Integration: Ensures ODELoopConfig receives duration for samples_per_summary

**Tests to Create**:
- Test file: tests/batchsolving/test_duration_propagation.py
- Test function: test_duration_propagates_to_loop_config
- Description: Verify that duration set in kernel.run reaches ODELoopConfig._duration
- Test function: test_samples_per_summary_uses_propagated_duration
- Description: Verify that samples_per_summary calculates correctly when summarise_last=True

**Tests to Run**:
- tests/batchsolving/test_duration_propagation.py::test_duration_propagates_to_loop_config
- tests/batchsolving/test_duration_propagation.py::test_samples_per_summary_uses_propagated_duration

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: Duration Dependency Warning at Solver Level
**Status**: [ ]
**Dependencies**: Task Group 1, Task Group 4

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 429-559, solve method)
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 264-358, run method)
- File: src/cubie/integrators/loops/ode_loop_config.py (lines 195-202, save_last/summarise_last)

**Input Validation Required**:
- None (warning generation only)

**Tasks**:
1. **Add duration dependency warning in Solver.solve**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     # Add import at top of file (around line 7)
     import warnings
     
     # In solve method, before kernel.run call (around line 537)
     # Check for duration dependency condition
     loop_config = self.kernel.single_integrator._loop.compile_settings
     if (loop_config.summarise_last and 
         loop_config._summarise_every is None):
         warnings.warn(
             "sample_summaries_every has been calculated from duration. "
             "Changing duration will trigger kernel recompilation. "
             "Set summarise_every explicitly to avoid recompilation.",
             UserWarning,
             stacklevel=2
         )
     ```
   - Edge cases:
     - Warning should only fire when summarise_last=True AND summarise_every=None
     - Warning should not fire when user explicitly set summarise_every
   - Integration: Alerts users to potential performance issue before kernel.run

**Tests to Create**:
- Test file: tests/batchsolving/test_solver_warnings.py
- Test function: test_duration_dependency_warning_emitted
- Description: Verify warning is raised when summarise_last=True and summarise_every=None
- Test function: test_no_warning_with_explicit_summarise_every
- Description: Verify no warning when summarise_every is explicitly set
- Test function: test_no_warning_with_summarise_last_false
- Description: Verify no warning when summarise_last=False

**Tests to Run**:
- tests/batchsolving/test_solver_warnings.py::test_duration_dependency_warning_emitted
- tests/batchsolving/test_solver_warnings.py::test_no_warning_with_explicit_summarise_every
- tests/batchsolving/test_solver_warnings.py::test_no_warning_with_summarise_last_false

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 6: Solver Property Type Annotations and None-Safety
**Status**: [ ]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 1026-1038, save_every/summarise_every properties)
- File: src/cubie/batchsolving/solveresult.py (lines 1-100, SolveSpec class)

**Input Validation Required**:
- None (type annotation updates only)

**Tasks**:
1. **Update Solver.save_every property return type**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     @property
     def save_every(self) -> Optional[float]:
         """Return the interval between saved outputs, or None if save_last only."""
         return self.kernel.save_every
     ```
   - Edge cases: None when save_last mode only
   - Integration: Matches kernel property signature

2. **Update Solver.summarise_every property return type**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     @property
     def summarise_every(self) -> Optional[float]:
         """Return the interval between summary computations, or None if summarise_last only."""
         return self.kernel.summarise_every
     ```
   - Edge cases: None when summarise_last mode only
   - Integration: Matches kernel property signature

3. **Update Solver.sample_summaries_every property return type**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     ```python
     @property
     def sample_summaries_every(self) -> Optional[float]:
         """Return the interval between summary metric samples, or None if not set."""
         return self.kernel.sample_summaries_every
     ```
   - Edge cases: None when not configured
   - Integration: Matches kernel property signature

4. **Ensure SolveSpec handles Optional timing values**
   - File: src/cubie/batchsolving/solveresult.py
   - Action: Verify/Modify
   - Details:
     ```python
     # Verify SolveSpec class fields are already Optional for timing params
     # If not, update the type annotations:
     save_every: Optional[float] = None
     summarise_every: Optional[float] = None
     sample_summaries_every: Optional[float] = None
     ```
   - Edge cases: SolveSpec should store None values without issue
   - Integration: Used by Solver.solve_info property

**Tests to Create**:
- Test file: tests/batchsolving/test_solver_timing_properties.py
- Test function: test_solver_save_every_returns_none_in_save_last_mode
- Description: Verify Solver.save_every returns None when save_every is not set
- Test function: test_solver_summarise_every_returns_none_in_summarise_last_mode
- Description: Verify Solver.summarise_every returns None when summarise_every is not set
- Test function: test_solve_info_handles_none_timing_values
- Description: Verify SolveSpec created from solve_info handles None timing values

**Tests to Run**:
- tests/batchsolving/test_solver_timing_properties.py::test_solver_save_every_returns_none_in_save_last_mode
- tests/batchsolving/test_solver_timing_properties.py::test_solver_summarise_every_returns_none_in_summarise_last_mode
- tests/batchsolving/test_solver_timing_properties.py::test_solve_info_handles_none_timing_values

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 7: Integration Tests for All Timing Modes
**Status**: [ ]
**Dependencies**: Task Groups 1-6

**Required Context**:
- File: tests/batchsolving/conftest.py (solver fixtures)
- File: tests/conftest.py (system fixtures)
- File: src/cubie/batchsolving/solver.py (solve_ivp function and Solver class)

**Input Validation Required**:
- None (integration tests only)

**Tasks**:
1. **Create comprehensive timing mode integration tests**
   - File: tests/batchsolving/test_timing_modes.py
   - Action: Create
   - Details:
     ```python
     """Integration tests for all timing parameter modes.
     
     Tests verify the complete flow from solve_ivp through to output
     array sizing for each timing parameter combination:
     
     1. save_every=None, summarise_every=None (save_last + summarise_last)
     2. save_every=X, summarise_every=None (periodic save, summarise_last)
     3. save_every=None, summarise_every=X (save_last, periodic summarise)
     4. save_every=X, summarise_every=Y (both periodic)
     5. Parameter reset between solves
     """
     import pytest
     import warnings
     import numpy as np
     from numpy import float32
     
     from cubie import solve_ivp
     
     
     class TestTimingModeOutputLengths:
         """Test output array lengths for each timing mode."""
         
         def test_save_last_only_output_length(self, system):
             """save_every=None produces output_length=2."""
             # Implementation: call solve_ivp with save_every=None
             # Assert result state shape[0] == 2
             pass
         
         def test_summarise_last_only_summaries_length(self, system):
             """summarise_every=None produces summaries_length=2."""
             # Implementation: call solve_ivp with summarise_every=None
             # Assert result summaries shape[0] == 2
             pass
         
         def test_periodic_save_output_length(self, system):
             """save_every=X produces floor(duration/X)+1 outputs."""
             pass
         
         def test_periodic_summarise_length(self, system):
             """summarise_every=X produces floor(duration/X) summaries."""
             pass
     
     
     class TestParameterReset:
         """Test parameter reset behavior between solves."""
         
         def test_sample_summaries_every_recalculates_on_none(self, system):
             """sample_summaries_every recalculates when reset to None."""
             # First solve with explicit value
             # Second solve with None - should recalculate, not preserve
             pass
     
     
     class TestDurationDependencyWarning:
         """Test duration dependency warning behavior."""
         
         def test_warning_on_summarise_last_without_summarise_every(
             self, system
         ):
             """Warning raised when duration affects samples_per_summary."""
             with warnings.catch_warnings(record=True) as w:
                 warnings.simplefilter("always")
                 # call solve_ivp with conditions that trigger warning
                 # Assert warning was raised with expected message
             pass
     ```
   - Edge cases: All permutations of timing parameter combinations
   - Integration: End-to-end validation of timing parameter flow

**Tests to Create**:
- Test file: tests/batchsolving/test_timing_modes.py
- Test function: test_save_last_only_output_length
- Description: Verify output_length=2 when save_every=None
- Test function: test_summarise_last_only_summaries_length
- Description: Verify summaries_length=2 when summarise_every=None
- Test function: test_periodic_save_output_length
- Description: Verify output_length calculation with explicit save_every
- Test function: test_periodic_summarise_length
- Description: Verify summaries_length calculation with explicit summarise_every
- Test function: test_sample_summaries_every_recalculates_on_none
- Description: Verify parameter reset behavior across multiple solves
- Test function: test_warning_on_summarise_last_without_summarise_every
- Description: Verify duration dependency warning is emitted

**Tests to Run**:
- tests/batchsolving/test_timing_modes.py::test_save_last_only_output_length
- tests/batchsolving/test_timing_modes.py::test_summarise_last_only_summaries_length
- tests/batchsolving/test_timing_modes.py::test_periodic_save_output_length
- tests/batchsolving/test_timing_modes.py::test_periodic_summarise_length
- tests/batchsolving/test_timing_modes.py::test_sample_summaries_every_recalculates_on_none
- tests/batchsolving/test_timing_modes.py::test_warning_on_summarise_last_without_summarise_every

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

# Summary

## Total Task Groups: 7

## Dependency Chain Overview:
```
Task Group 1 (ODELoopConfig Enhancement)
    ├── Task Group 2 (BatchSolverKernel None-Safe Properties)
    │       └── Task Group 6 (Solver Property Type Annotations)
    ├── Task Group 3 (Parameter Reset Mechanism)
    │       └── Task Group 4 (Duration Propagation)
    │               └── Task Group 5 (Duration Dependency Warning)
    └── Task Group 7 (Integration Tests) [depends on all above]
```

## Tests to be Created:
1. `tests/integrators/loops/test_ode_loop_config_timing.py` - Config inference and reset tests
2. `tests/batchsolving/test_kernel_output_lengths.py` - None-safe output length tests
3. `tests/batchsolving/test_duration_propagation.py` - Duration propagation tests
4. `tests/batchsolving/test_solver_warnings.py` - Warning emission tests
5. `tests/batchsolving/test_solver_timing_properties.py` - Property type tests
6. `tests/batchsolving/test_timing_modes.py` - Integration tests

## Estimated Complexity:
- Task Group 1: Medium (core config changes, property modifications)
- Task Group 2: Medium (multiple property updates with None handling)
- Task Group 3: Medium (new method + update logic modification)
- Task Group 4: Low-Medium (threading parameter through existing chain)
- Task Group 5: Low (single warning addition)
- Task Group 6: Low (type annotation updates)
- Task Group 7: Medium (comprehensive integration test suite)

## Key Files Modified:
1. `src/cubie/integrators/loops/ode_loop_config.py`
2. `src/cubie/integrators/loops/ode_loop.py`
3. `src/cubie/batchsolving/BatchSolverKernel.py`
4. `src/cubie/batchsolving/solver.py`
5. `src/cubie/integrators/SingleIntegratorRunCore.py` (verification only)
6. `src/cubie/batchsolving/solveresult.py` (verification only)

## Constraint Compliance:
- ✓ Does not modify summary metrics code
- ✓ Focuses on timing parameter handling and array sizing
- ✓ Follows CuBIE conventions for attrs classes and properties
- ✓ Maintains backwards compatibility for existing timing patterns
