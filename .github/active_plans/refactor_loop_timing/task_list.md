# Implementation Task List
# Feature: Refactor Loop Timing Parameters
# Plan Reference: .github/active_plans/refactor_loop_timing/agent_plan.md

## Task Group 1: ODELoopConfig - Remove Deprecated Fields and Add New Flags
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop_config.py (entire file)
- File: .github/context/cubie_internal_structure.md (lines 196-206 for Attrs Classes Pattern)

**Input Validation Required**:
- `save_last`: Must be bool, validator already handles via `validators.instance_of(bool)`
- `summarise_last`: Must be bool, validator already handles via `validators.instance_of(bool)`
- No additional validation beyond attrs validators

**Tasks**:
1. **Remove deprecated attrs fields**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     Delete the following field definitions (lines 196-207):
     ```python
     # DELETE these lines:
     _dt_save: Optional[float] = field(
         default=None,
         validator=opt_gttype_validator(float, 0)
     )
     _dt_summarise: Optional[float] = field(
         default=None,
         validator=opt_gttype_validator(float, 0)
     )
     _dt_update_summaries: Optional[float] = field(
         default=None,
         validator=opt_gttype_validator(float, 0)
     )
     ```
   - Edge cases: None
   - Integration: This removes backward compatibility - any code using these fields will break

2. **Add save_last and summarise_last flag fields**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     Add after `_sample_summaries_every` field definition (around line 193):
     ```python
     # Flags for end-of-run-only behavior
     save_last: bool = field(
         default=False,
         validator=validators.instance_of(bool)
     )
     summarise_last: bool = field(
         default=False,
         validator=validators.instance_of(bool)
     )
     ```
   - Edge cases: Flags should default to False for backward compatibility with explicit timing values
   - Integration: These flags will be read by IVPLoop.build()

3. **Rewrite __attrs_post_init__ with new None-handling logic**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     Replace entire `__attrs_post_init__` method with new logic:
     ```python
     def __attrs_post_init__(self):
         """Validate timing parameters and set flags for None handling.
         
         When all timing parameters are None, sets save_last and 
         summarise_last flags to True for end-of-run-only behavior.
         Otherwise applies inference logic to fill missing values.
         """
         # Case 1: All three None - set flags for end-of-run-only behavior
         if (self._save_every is None and self._summarise_every is None and 
                 self._sample_summaries_every is None):
             object.__setattr__(self, 'save_last', True)
             object.__setattr__(self, 'summarise_last', True)
             # Set sentinel values for loop timing (will be overridden)
             object.__setattr__(self, '_save_every', 0.1)
             object.__setattr__(self, '_summarise_every', 1.0)
             object.__setattr__(self, '_sample_summaries_every', 0.1)
             return  # Skip validation when using save_last/summarise_last
         
         # Case 2: Only save_every specified
         elif (self._save_every is not None and self._summarise_every is None and
                 self._sample_summaries_every is None):
             object.__setattr__(self, 'summarise_last', True)
             object.__setattr__(self, '_summarise_every', 10.0 * self._save_every)
             object.__setattr__(self, '_sample_summaries_every', self._save_every)
         
         # Case 3: Only summarise_every specified
         elif (self._save_every is None and self._summarise_every is not None and
                 self._sample_summaries_every is None):
             object.__setattr__(self, '_save_every', self._summarise_every / 10.0)
             object.__setattr__(
                 self, '_sample_summaries_every', self._summarise_every / 10.0
             )
         
         # Case 4: save_every and summarise_every specified
         elif (self._save_every is not None and self._summarise_every is not None and
                 self._sample_summaries_every is None):
             object.__setattr__(self, '_sample_summaries_every', self._save_every)
         
         # Case 5: save_every and sample_summaries_every specified
         elif (self._save_every is not None and self._summarise_every is None and
                 self._sample_summaries_every is not None):
             object.__setattr__(self, '_summarise_every', 10.0 * self._save_every)
         
         # Case 6: summarise_every and sample_summaries_every specified
         elif (self._save_every is None and self._summarise_every is not None and
                 self._sample_summaries_every is not None):
             object.__setattr__(self, '_save_every', self._sample_summaries_every)
         
         # Case 7: All three specified - no defaults needed
         
         # Validate that sample_summaries_every divides summarise_every evenly
         # Skip validation when summarise_last is True
         if not self.summarise_last:
             tolerance = 1e-6 if self.precision == float32 else 1e-9
             ratio = self._summarise_every / self._sample_summaries_every
             if abs(ratio - round(ratio)) > tolerance:
                 raise ValueError(
                     f"sample_summaries_every ({self._sample_summaries_every}) must "
                     f"be an integer divisor of summarise_every "
                     f"({self._summarise_every}). Ratio: {ratio}"
                 )
     ```
   - Edge cases: Validation skipped when summarise_last=True
   - Integration: Logic determines behavior of loop kernel at runtime

4. **Remove deprecated backward compatibility properties**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     Delete the following property definitions (lines 407-421):
     ```python
     # DELETE these properties:
     @property
     def dt_save(self) -> float:
         """Return the output save interval (deprecated, use save_every)."""
         return self.save_every

     @property
     def dt_summarise(self) -> float:
         """Return the summary interval (deprecated, use summarise_every)."""
         return self.summarise_every

     @property
     def dt_update_summaries(self) -> float:
         """Return the summary update interval (deprecated, use sample_summaries_every)."""
         return self.sample_summaries_every
     ```
   - Edge cases: Any code accessing these properties will break
   - Integration: Forces users to migrate to new property names

**Tests to Create**:
- Test file: tests/integrators/loops/test_dt_update_summaries_validation.py
- Test function: test_all_none_sets_save_last_flag
- Description: Verify that when all timing params are None, save_last becomes True
- Test function: test_all_none_sets_summarise_last_flag
- Description: Verify that when all timing params are None, summarise_last becomes True
- Test function: test_only_save_every_sets_summarise_last
- Description: Verify that specifying only save_every sets summarise_last=True
- Test function: test_deprecated_params_removed
- Description: Verify that dt_save, dt_summarise, dt_update_summaries are not valid params

**Tests to Run**:
- tests/integrators/loops/test_dt_update_summaries_validation.py::test_all_none_sets_save_last_flag
- tests/integrators/loops/test_dt_update_summaries_validation.py::test_all_none_sets_summarise_last_flag
- tests/integrators/loops/test_dt_update_summaries_validation.py::test_only_save_every_sets_summarise_last
- tests/integrators/loops/test_dt_update_summaries_validation.py::test_deprecated_params_removed

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/loops/ode_loop_config.py (67 lines changed)
  * tests/integrators/loops/test_dt_update_summaries_validation.py (63 lines added)
- Functions/Methods Added/Modified:
  * `save_last` field added in ODELoopConfig
  * `summarise_last` field added in ODELoopConfig
  * `__attrs_post_init__()` rewritten with new None-handling logic
- Implementation Summary:
  Removed deprecated `_dt_save`, `_dt_summarise`, `_dt_update_summaries` fields
  and their corresponding backward compatibility properties `dt_save`, 
  `dt_summarise`, `dt_update_summaries`. Added new `save_last` and 
  `summarise_last` bool fields. Rewrote `__attrs_post_init__` to set flags
  when all timing parameters are None or when only save_every is specified.
  Removed unused `warn` import.
- Issues Flagged: The backward compatibility tests in the test file (lines 145-218)
  will now fail since deprecated parameters were removed. These should be removed
  in Task Group 6.

---

## Task Group 2: IVPLoop - Remove Deprecated Parameters and Wire Flags
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (entire file)
- File: src/cubie/integrators/loops/ode_loop_config.py (for ODELoopConfig with new flags)

**Input Validation Required**:
- None - parameters validated by ODELoopConfig

**Tasks**:
1. **Update ALL_LOOP_SETTINGS set**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     Update the ALL_LOOP_SETTINGS set (lines 36-62) to remove deprecated entries:
     ```python
     ALL_LOOP_SETTINGS = {
         "save_every",
         "summarise_every",
         "sample_summaries_every",
         # REMOVE: "dt_save",  # Deprecated
         # REMOVE: "dt_summarise",  # Deprecated
         # REMOVE: "dt_update_summaries",  # Deprecated
         "dt0",
         "dt_min",
         "dt_max",
         "is_adaptive",
         "save_last",      # ADD new flag
         "summarise_last", # ADD new flag
         # Buffer location parameters...
     }
     ```
   - Edge cases: Any code passing deprecated keys will fail silently
   - Integration: Solver uses this set to filter kwargs

2. **Remove deprecated parameters from __init__**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     Remove from __init__ signature (lines 138-141):
     ```python
     # REMOVE these parameters:
     dt_save: Optional[float] = None,
     dt_summarise: Optional[float] = None,
     dt_update_summaries: Optional[float] = None,
     ```
     Also remove from build_config call (lines 228-230):
     ```python
     # REMOVE from build_config required dict:
     'dt_save': dt_save,
     'dt_summarise': dt_summarise,
     'dt_update_summaries': dt_update_summaries,
     ```
     Update docstrings to remove references to deprecated parameters.
   - Edge cases: Existing code passing these parameters will fail
   - Integration: Forces migration to new parameter names

3. **Wire save_last and summarise_last flags in build()**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     Update build() method (around lines 385-388) to read flags from config:
     ```python
     # REPLACE hardcoded flags:
     # OLD:
     # save_last = False
     # summarise_last = False
     
     # NEW:
     save_last = config.save_last
     summarise_last = config.summarise_last
     ```
   - Edge cases: Flags affect loop termination and save logic
   - Integration: Loop device function uses these values

4. **Remove deprecated property methods**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     Delete the dt_save and dt_summarise properties (lines 812-825):
     ```python
     # DELETE these properties:
     @property
     def dt_save(self) -> float:
         """Return the save interval (deprecated, use save_every)."""
         return self.compile_settings.save_every

     @property
     def dt_summarise(self) -> float:
         """Return the summary interval (deprecated, use summarise_every)."""
         return self.compile_settings.summarise_every
     ```
   - Edge cases: Code accessing these properties will break
   - Integration: Users must access via save_every and summarise_every

**Tests to Create**:
- Test file: tests/integrators/loops/test_ode_loop.py
- Test function: test_save_last_flag_from_config
- Description: Verify IVPLoop reads save_last from ODELoopConfig
- Test function: test_summarise_last_flag_from_config
- Description: Verify IVPLoop reads summarise_last from ODELoopConfig

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py::test_save_last_flag_from_config
- tests/integrators/loops/test_ode_loop.py::test_summarise_last_flag_from_config

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/loops/ode_loop.py (45 lines changed)
  * tests/integrators/loops/test_ode_loop.py (48 lines added)
- Functions/Methods Added/Modified:
  * ALL_LOOP_SETTINGS set updated (removed deprecated, added new flags)
  * __init__() signature simplified (removed 3 deprecated params)
  * build() method updated (reads save_last/summarise_last from config)
  * dt_save property deleted
  * dt_summarise property deleted
- Implementation Summary:
  Removed deprecated `dt_save`, `dt_summarise`, `dt_update_summaries` parameters
  from __init__ and build_config call. Removed these from ALL_LOOP_SETTINGS set
  and added `save_last` and `summarise_last`. Updated build() to read flags from
  config instead of hardcoded False. Removed deprecated dt_save and dt_summarise
  properties, keeping only save_every and summarise_every. Added two tests to
  verify flag wiring from ODELoopConfig.
- Issues Flagged: Existing test `test_getters` in test_ode_loop.py uses deprecated
  `dt_save` and `dt_summarise` properties which will now fail. This should be
  addressed in Task Group 6 (Test Updates).

---

## Task Group 3: OutputConfig and OutputFunctions - Rename dt_save to save_every
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/outputhandling/output_config.py (entire file)
- File: src/cubie/outputhandling/output_functions.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (for MetricConfig)

**Input Validation Required**:
- `save_every`: Must be > 0.0 (validated by opt_gttype_validator)

**Tasks**:
1. **Rename _dt_save to _save_every in OutputConfig**
   - File: src/cubie/outputhandling/output_config.py
   - Action: Modify
   - Details:
     Rename field (around line 182-185):
     ```python
     # OLD:
     _dt_save: float = field(
         default=0.01,
         validator=opt_gttype_validator(float, 0.0)
     )
     
     # NEW:
     _save_every: float = field(
         default=0.01,
         validator=opt_gttype_validator(float, 0.0)
     )
     ```
     Rename property (around line 664-667):
     ```python
     # OLD:
     @property
     def dt_save(self) -> float:
         """Time interval between saved states."""
         return self._dt_save
     
     # NEW:
     @property
     def save_every(self) -> float:
         """Time interval between saved states."""
         return self._save_every
     ```
   - Edge cases: None
   - Integration: OutputFunctions and other consumers must use new name

2. **Update from_loop_settings classmethod parameter**
   - File: src/cubie/outputhandling/output_config.py
   - Action: Modify
   - Details:
     Update parameter name in from_loop_settings (around lines 905-906):
     ```python
     # OLD:
     dt_save: Optional[float] = 0.01,
     
     # NEW:
     save_every: Optional[float] = 0.01,
     ```
     Update return statement (around line 968):
     ```python
     # OLD:
     dt_save=dt_save,
     
     # NEW:
     save_every=save_every,
     ```
     Update docstring to reference save_every instead of dt_save.
   - Edge cases: None
   - Integration: Callers must use new parameter name

3. **Update ALL_OUTPUT_FUNCTION_PARAMETERS set**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     Update the set (around lines 28-38):
     ```python
     ALL_OUTPUT_FUNCTION_PARAMETERS = {
         "output_types",
         "saved_states", "saved_observables",
         "summarised_states", "summarised_observables",
         "saved_state_indices",
         "saved_observable_indices",
         "summarised_state_indices",
         "summarised_observable_indices",
         "save_every",  # RENAMED from "dt_save"
         "precision",
     }
     ```
   - Edge cases: Code passing dt_save will fail
   - Integration: Solver uses this set to filter kwargs

4. **Update OutputFunctions to use save_every**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     Update __init__ parameter (around line 108):
     ```python
     # OLD:
     dt_save: Optional[float] = None,
     
     # NEW:
     save_every: Optional[float] = None,
     ```
     Update from_loop_settings call (around line 124):
     ```python
     # OLD:
     dt_save=dt_save,
     
     # NEW:
     save_every=save_every,
     ```
     Update build() method (around line 202):
     ```python
     # OLD:
     summary_metrics.update(dt_save=config.dt_save, precision=config.precision)
     
     # NEW:
     summary_metrics.update(dt_save=config.save_every, precision=config.precision)
     ```
     Note: summary_metrics.update still uses dt_save internally as that's the MetricConfig field name.
   - Edge cases: None
   - Integration: Maintains consistency with rest of API

**Tests to Create**:
- Test file: tests/outputhandling/test_output_config.py
- Test function: test_save_every_property
- Description: Verify OutputConfig has save_every property that returns correct value
- Test function: test_from_loop_settings_save_every
- Description: Verify from_loop_settings accepts save_every parameter

**Tests to Run**:
- tests/outputhandling/test_output_config.py::TestSaveEveryProperty::test_save_every_property
- tests/outputhandling/test_output_config.py::TestSaveEveryProperty::test_from_loop_settings_save_every

**Outcomes**: 
- Files Modified: 
  * src/cubie/outputhandling/output_config.py (8 lines changed)
  * src/cubie/outputhandling/output_functions.py (5 lines changed)
  * tests/outputhandling/test_output_config.py (42 lines added)
- Functions/Methods Added/Modified:
  * `_save_every` field renamed from `_dt_save` in OutputConfig
  * `save_every` property renamed from `dt_save` in OutputConfig
  * `from_loop_settings()` parameter renamed from `dt_save` to `save_every`
  * `ALL_OUTPUT_FUNCTION_PARAMETERS` set updated with "save_every"
  * `OutputFunctions.__init__()` parameter renamed from `dt_save` to `save_every`
  * `OutputFunctions.build()` updated to reference `config.save_every`
- Implementation Summary:
  Renamed `_dt_save` field to `_save_every` and `dt_save` property to 
  `save_every` in OutputConfig. Updated `from_loop_settings` classmethod 
  parameter and docstring. Updated `ALL_OUTPUT_FUNCTION_PARAMETERS` set 
  to use "save_every". Updated `OutputFunctions.__init__` and its 
  `from_loop_settings` call. Updated `build()` to reference 
  `config.save_every` while preserving the internal `dt_save` keyword 
  for `summary_metrics.update()` (as MetricConfig uses that name internally).
  Added test class `TestSaveEveryProperty` with four test functions.
- Issues Flagged: None

---

## Task Group 4: Solver - Remove Deprecated API
**Status**: [x]
**Dependencies**: Task Group 2, Task Group 3

**Required Context**:
- File: src/cubie/batchsolving/solver.py (entire file)
- File: src/cubie/batchsolving/solveresult.py (for SolveSpec updates)

**Input Validation Required**:
- None - parameters validated by downstream components

**Tasks**:
1. **Update solve_ivp function**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     Remove dt_save parameter and backward compat logic (around lines 55-126):
     ```python
     # REMOVE dt_save parameter:
     # OLD:
     def solve_ivp(
         ...
         save_every: Optional[float] = None,
         dt_save: Optional[float] = None,  # REMOVE this line
         ...
     ):
     
     # REMOVE backward compatibility block (lines 115-122):
     # DELETE:
     # Handle backward compatibility for dt_save vs save_every
     if dt_save is not None and save_every is not None:
         raise ValueError(...)
     if dt_save is not None:
         save_every = dt_save
     ```
     Update docstring to remove dt_save references.
   - Edge cases: Code passing dt_save will fail with unexpected keyword argument
   - Integration: Users must use save_every

2. **Remove deprecated Solver properties**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     Delete deprecated property definitions (around lines 1038-1049):
     ```python
     # DELETE these properties:
     @property
     def dt_save(self):
         """Return the interval between saved outputs (deprecated, use save_every)."""
         return self.kernel.save_every
     
     @property
     def dt_summarise(self):
         """Return the interval between summary computations (deprecated, use summarise_every)."""
         return self.kernel.summarise_every
     ```
   - Edge cases: Code accessing these properties will break
   - Integration: Users must use save_every and summarise_every properties

3. **Update solve_info property**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     Update SolveSpec construction (around lines 1110-1130):
     ```python
     # OLD:
     return SolveSpec(
         ...
         dt_save=self.save_every,
         dt_summarise=self.summarise_every,
         ...
     )
     
     # NEW:
     return SolveSpec(
         ...
         save_every=self.save_every,
         summarise_every=self.summarise_every,
         ...
     )
     ```
   - Edge cases: None
   - Integration: SolveSpec must also be updated (Task Group 5)

**Tests to Create**:
- Test file: tests/batchsolving/test_solver.py
- Test function: test_solve_ivp_save_every_param
- Description: Verify solve_ivp accepts save_every parameter
- Test function: test_solve_ivp_no_dt_save
- Description: Verify solve_ivp rejects dt_save parameter

**Tests to Run**:
- tests/batchsolving/test_solver.py::test_solve_ivp_save_every_param
- tests/batchsolving/test_solver.py::test_solve_ivp_no_dt_save

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/solver.py (23 lines changed)
  * tests/batchsolving/test_solver.py (42 lines added)
- Functions/Methods Added/Modified:
  * `solve_ivp()` - removed `dt_save` parameter and backward compatibility logic
  * `dt_save` property deleted from Solver class
  * `dt_summarise` property deleted from Solver class
  * `solve_info` property updated to use `save_every`/`summarise_every` kwargs
  * Solver class docstring updated to reference new parameter names
- Implementation Summary:
  Removed the deprecated `dt_save` parameter from `solve_ivp()` function along
  with all backward compatibility logic. Removed `dt_save` and `dt_summarise` 
  deprecated property aliases from Solver class. Updated `solve_info` property
  to construct SolveSpec with `save_every` and `summarise_every` keyword 
  arguments (requires Task Group 5 to update SolveSpec fields). Updated Solver
  class docstring to reference new parameter names. Added two test functions
  to verify solve_ivp accepts save_every and rejects dt_save.
- Issues Flagged: The `solve_info` property now uses `save_every` and 
  `summarise_every` kwargs which will cause an error until Task Group 5
  updates SolveSpec to use those field names instead of `dt_save`/`dt_summarise`.
  Existing tests using `dt_save` and `dt_summarise` will fail until Task Group 6
  updates them.

---

## Task Group 5: SolveSpec - Rename Fields
**Status**: [x]
**Dependencies**: Task Group 4

**Required Context**:
- File: src/cubie/batchsolving/solveresult.py (lines 52-120 for SolveSpec)

**Input Validation Required**:
- `save_every`: Must be > 0.0 (gttype_validator)
- `summarise_every`: Must be >= 0.0 (getype_validator)

**Tasks**:
1. **Rename dt_save and dt_summarise fields in SolveSpec**
   - File: src/cubie/batchsolving/solveresult.py
   - Action: Modify
   - Details:
     Update field definitions (around lines 91-96):
     ```python
     # OLD:
     dt_save: float = attrs.field(validator=gttype_validator(float, 0.0))
     dt_summarise: float = attrs.field(validator=getype_validator(float, 0.0))
     
     # NEW:
     save_every: float = attrs.field(validator=gttype_validator(float, 0.0))
     summarise_every: float = attrs.field(validator=getype_validator(float, 0.0))
     ```
     Update docstring references (around lines 58-66):
     ```python
     # OLD:
     dt_save
         Interval at which state values are stored.
     dt_summarise
         Interval for computing summary outputs.
     
     # NEW:
     save_every
         Interval at which state values are stored.
     summarise_every
         Interval for computing summary outputs.
     ```
   - Edge cases: None
   - Integration: All code constructing SolveSpec must use new names

**Tests to Create**:
- Test file: tests/batchsolving/test_solveresult.py
- Test function: test_solvespec_save_every_field
- Description: Verify SolveSpec has save_every field
- Test function: test_solvespec_summarise_every_field
- Description: Verify SolveSpec has summarise_every field

**Tests to Run**:
- tests/batchsolving/test_solveresult.py::TestSolveSpecFields::test_solvespec_save_every_field
- tests/batchsolving/test_solveresult.py::TestSolveSpecFields::test_solvespec_summarise_every_field

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/solveresult.py (4 lines changed)
  * tests/batchsolving/test_solveresult.py (57 lines added)
- Functions/Methods Added/Modified:
  * `save_every` field renamed from `dt_save` in SolveSpec
  * `summarise_every` field renamed from `dt_summarise` in SolveSpec
- Implementation Summary:
  Renamed `dt_save` field to `save_every` and `dt_summarise` field to 
  `summarise_every` in SolveSpec attrs class. Updated docstring to 
  reference the new field names. Added TestSolveSpecFields class with 
  two test functions to verify the new field names exist and work correctly.
- Issues Flagged: None

---

## Task Group 6: Test Updates - Remove Backward Compatibility Tests
**Status**: [ ]
**Dependencies**: Task Groups 1-5

**Required Context**:
- File: tests/integrators/loops/test_dt_update_summaries_validation.py (entire file)
- File: tests/_utils.py (lines 24-39 for MID_RUN_PARAMS, LONG_RUN_PARAMS)

**Input Validation Required**:
- None - test modifications only

**Tasks**:
1. **Update test parameter dictionaries in tests/_utils.py**
   - File: tests/_utils.py
   - Action: Modify
   - Details:
     Update MID_RUN_PARAMS and LONG_RUN_PARAMS (lines 24-38):
     ```python
     # OLD:
     MID_RUN_PARAMS = {
         'dt': 0.001,
         'dt_save': 0.02,
         'dt_summarise': 0.1,
         'dt_max': 0.5,
         'output_types': ['state', 'time', 'observables', 'mean'],
     }
     
     LONG_RUN_PARAMS = {
         'duration': 0.3,
         'dt': 0.0005,
         'dt_save': 0.05,
         'dt_summarise': 0.15,
         'output_types': ['state', 'observables', 'time', 'mean', 'rms'],
     }
     
     # NEW:
     MID_RUN_PARAMS = {
         'dt': 0.001,
         'save_every': 0.02,
         'summarise_every': 0.1,
         'dt_max': 0.5,
         'output_types': ['state', 'time', 'observables', 'mean'],
     }
     
     LONG_RUN_PARAMS = {
         'duration': 0.3,
         'dt': 0.0005,
         'save_every': 0.05,
         'summarise_every': 0.15,
         'output_types': ['state', 'observables', 'time', 'mean', 'rms'],
     }
     ```
   - Edge cases: None
   - Integration: All tests using these dicts will use new names

2. **Remove backward compatibility tests**
   - File: tests/integrators/loops/test_dt_update_summaries_validation.py
   - Action: Modify
   - Details:
     Delete the following test functions (lines 145-218):
     - `test_backward_compat_dt_save`
     - `test_backward_compat_dt_summarise`
     - `test_backward_compat_dt_update_summaries`
     - `test_cannot_specify_both_dt_save_and_save_every`
     
     These tests verify deprecated behavior that no longer exists.
   - Edge cases: None
   - Integration: Tests now only verify new API

3. **Add new tests for save_last and summarise_last flags**
   - File: tests/integrators/loops/test_dt_update_summaries_validation.py
   - Action: Modify
   - Details:
     Add new test functions after existing tests:
     ```python
     def test_all_none_sets_save_last_flag():
         """Test that all None timing params sets save_last=True."""
         config = ODELoopConfig(
             n_states=3,
             n_parameters=0,
             n_drivers=0,
             n_observables=0,
             n_error=0,
             n_counters=0,
             state_summaries_buffer_height=0,
             observable_summaries_buffer_height=0,
         )
         assert config.save_last is True


     def test_all_none_sets_summarise_last_flag():
         """Test that all None timing params sets summarise_last=True."""
         config = ODELoopConfig(
             n_states=3,
             n_parameters=0,
             n_drivers=0,
             n_observables=0,
             n_error=0,
             n_counters=0,
             state_summaries_buffer_height=0,
             observable_summaries_buffer_height=0,
         )
         assert config.summarise_last is True


     def test_only_save_every_sets_summarise_last():
         """Test that specifying only save_every sets summarise_last=True."""
         config = ODELoopConfig(
             n_states=3,
             n_parameters=0,
             n_drivers=0,
             n_observables=0,
             n_error=0,
             n_counters=0,
             state_summaries_buffer_height=0,
             observable_summaries_buffer_height=0,
             save_every=0.2,
         )
         assert config.summarise_last is True
         assert config.save_last is False


     def test_explicit_values_dont_set_flags():
         """Test that explicit timing values keep flags False."""
         config = ODELoopConfig(
             n_states=3,
             n_parameters=0,
             n_drivers=0,
             n_observables=0,
             n_error=0,
             n_counters=0,
             state_summaries_buffer_height=0,
             observable_summaries_buffer_height=0,
             save_every=0.1,
             summarise_every=1.0,
             sample_summaries_every=0.1,
         )
         assert config.save_last is False
         assert config.summarise_last is False
     ```
   - Edge cases: None
   - Integration: Validates new flag behavior

4. **Update test_all_none_uses_defaults to reflect new behavior**
   - File: tests/integrators/loops/test_dt_update_summaries_validation.py
   - Action: Modify
   - Details:
     Update test_all_none_uses_defaults (around lines 9-24) to verify flags:
     ```python
     def test_all_none_uses_defaults():
         """Test that all None sets save_last and summarise_last flags."""
         config = ODELoopConfig(
             n_states=3,
             n_parameters=0,
             n_drivers=0,
             n_observables=0,
             n_error=0,
             n_counters=0,
             state_summaries_buffer_height=0,
             observable_summaries_buffer_height=0,
         )
         
         # Sentinel values still set for loop timing calculations
         assert config.save_every == pytest.approx(0.1)
         assert config.summarise_every == pytest.approx(1.0)
         assert config.sample_summaries_every == pytest.approx(0.1)
         # But flags indicate end-of-run-only behavior
         assert config.save_last is True
         assert config.summarise_last is True
     ```
   - Edge cases: None
   - Integration: Test reflects new behavior

**Tests to Create**:
- None - this task group modifies existing tests

**Tests to Run**:
- tests/integrators/loops/test_dt_update_summaries_validation.py

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

**Total Task Groups**: 6
**Dependency Chain**: 
- Task Group 1 (ODELoopConfig) → Task Groups 2, 3
- Task Groups 2, 3 → Task Group 4 (Solver)
- Task Group 4 → Task Group 5 (SolveSpec)
- Task Groups 1-5 → Task Group 6 (Tests)

**Tests to Create Overall**:
1. tests/integrators/loops/test_dt_update_summaries_validation.py - New flag tests
2. tests/integrators/loops/test_ode_loop.py - Flag wiring tests
3. tests/outputhandling/test_output_config.py - save_every tests
4. tests/batchsolving/test_solver.py - API tests
5. tests/batchsolving/test_solveresult.py - SolveSpec tests

**Estimated Complexity**: Medium
- Primary changes are field/parameter renames
- New logic is mostly in __attrs_post_init__ for flag inference
- Test updates are straightforward search-and-replace plus new flag tests
