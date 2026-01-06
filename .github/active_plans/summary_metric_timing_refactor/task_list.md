# Implementation Task List
# Feature: Summary Metric Timing Parameters Refactor
# Plan Reference: .github/active_plans/summary_metric_timing_refactor/agent_plan.md

## Task Group 1: Core Metrics Infrastructure (MetricConfig and SummaryMetric)
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/metrics.py (entire file)
- File: .github/copilot-instructions.md (attrs classes pattern, comment style)

**Input Validation Required**:
- No additional validation needed; existing validators on `_dt_save` already validate `> 0.0`

**Tasks**:
1. **Rename MetricConfig._dt_save to _sample_summaries_every**
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     - Line 57: Rename `_dt_save` → `_sample_summaries_every`
     - Preserve the same validator (`val_optional(gttype_validator(float, 0.0))`)
     - Preserve the same default value (`0.01`)
   - Integration: All SummaryMetric subclasses access this via compile_settings

2. **Rename MetricConfig.dt_save property to sample_summaries_every**
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     - Lines 64-66: Rename property `dt_save` → `sample_summaries_every`
     - Update return statement: `return self._sample_summaries_every`
     - Update docstring: "Time interval between summary metric samples."

3. **Update MetricConfig class docstring**
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     - Lines 41-52: Update docstring to reference `sample_summaries_every` instead of `dt_save`
     - Change description from "Time interval between saved states" to "Time interval between summary metric samples"
     - Update attribute name in docstring from `dt_save` to `sample_summaries_every`

4. **Update SummaryMetric.__init__ parameter and docstring**
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     - Line 141: Rename parameter `dt_save` → `sample_summaries_every`
     - Lines 119-120: Update class docstring attribute `dt_save` → `sample_summaries_every`
     - Lines 157-158: Update parameter docstring from `dt_save` to `sample_summaries_every`
     - Line 172: Update MetricConfig instantiation:
       ```python
       MetricConfig(sample_summaries_every=sample_summaries_every, precision=precision)
       ```
   - Integration: All metric subclasses inherit this behavior

5. **Update SummaryMetrics.update() docstring**
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     - Line 307: Update docstring example from `dt_save=0.02` to `sample_summaries_every=0.02`
   - Integration: OutputFunctions.build() calls this method

**Tests to Create**:
- No new tests needed; existing tests will validate the rename

**Tests to Run**:
- tests/outputhandling/test_output_functions.py

**Outcomes**: 
- Files Modified: 
  * src/cubie/outputhandling/summarymetrics/metrics.py (25 lines changed)
- Functions/Methods Added/Modified:
  * MetricConfig class: renamed `_dt_save` to `_sample_summaries_every`, renamed `dt_save` property to `sample_summaries_every`
  * SummaryMetric.__init__(): renamed parameter `dt_save` to `sample_summaries_every`
  * SummaryMetric.update(): updated docstring example
  * SummaryMetrics.update(): updated docstring example
- Implementation Summary:
  Renamed all occurrences of `dt_save` to `sample_summaries_every` in the core metrics infrastructure classes (MetricConfig and SummaryMetric). Updated class docstrings, property names, parameter names, and docstring examples. MetricConfig instantiation now uses `sample_summaries_every` keyword argument.
- Issues Flagged: None 


---

## Task Group 2: Derivative Metric Files Update (dxdt_* and d2xdt2_*)
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/dxdt_max.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/dxdt_min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/dxdt_extrema.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_max.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_min.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py (entire file)

**Input Validation Required**:
- None; these files only read from compile_settings

**Tasks**:
1. **Update dxdt_max.py build() method**
   - File: src/cubie/outputhandling/summarymetrics/dxdt_max.py
   - Action: Modify
   - Details:
     - Line 57: Change `dt_save = self.compile_settings.dt_save` → `sample_summaries_every = self.compile_settings.sample_summaries_every`
     - Line 131: Update save function to use `sample_summaries_every`:
       ```python
       output_array[0] = buffer[1] / precision(sample_summaries_every)
       ```
     - Line 29-30: Update class docstring comment "scaled by dt_save" → "scaled by sample_summaries_every"
     - Line 54-55: Update build() docstring "scales by dt_save" → "scales by sample_summaries_every"
     - Lines 126-130: Update save() docstring references from "dt_save" to "sample_summaries_every"
   - Integration: Compiled device function uses this for derivative scaling

2. **Update dxdt_min.py build() method**
   - File: src/cubie/outputhandling/summarymetrics/dxdt_min.py
   - Action: Modify
   - Details:
     - Line 57: Change `dt_save = self.compile_settings.dt_save` → `sample_summaries_every = self.compile_settings.sample_summaries_every`
     - Line 131: Update save function to use `sample_summaries_every`:
       ```python
       output_array[0] = buffer[1] / precision(sample_summaries_every)
       ```
     - Line 29-30: Update class docstring comment "scaled by dt_save" → "scaled by sample_summaries_every"
     - Line 54-55: Update build() docstring "scales by dt_save" → "scales by sample_summaries_every"
     - Lines 126-130: Update save() docstring references from "dt_save" to "sample_summaries_every"
   - Integration: Compiled device function uses this for derivative scaling

3. **Update dxdt_extrema.py build() method**
   - File: src/cubie/outputhandling/summarymetrics/dxdt_extrema.py
   - Action: Modify
   - Details:
     - Line 57: Change `dt_save = self.compile_settings.dt_save` → `sample_summaries_every = self.compile_settings.sample_summaries_every`
     - Lines 133-134: Update save function to use `sample_summaries_every`:
       ```python
       output_array[0] = buffer[1] / precision(sample_summaries_every)
       output_array[1] = buffer[2] / precision(sample_summaries_every)
       ```
     - Line 28-29: Update class docstring comment "scaled by dt_save" → "scaled by sample_summaries_every"
     - Line 53-54: Update build() docstring "scales by dt_save" → "scales by sample_summaries_every"
     - Lines 128-132: Update save() docstring references from "dt_save" to "sample_summaries_every"
   - Integration: Compiled device function uses this for derivative scaling

4. **Update d2xdt2_max.py build() method**
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_max.py
   - Action: Modify
   - Details:
     - Line 58: Change `dt_save = self.compile_settings.dt_save` → `sample_summaries_every = self.compile_settings.sample_summaries_every`
     - Line 134: Update save function to use `sample_summaries_every`:
       ```python
       output_array[0] = buffer[2] / (precision(sample_summaries_every) * precision(sample_summaries_every))
       ```
     - Line 30-31: Update class docstring comment "scaled by dt_save²" → "scaled by sample_summaries_every²"
     - Line 55-56: Update build() docstring "scales by dt_save²" → "scales by sample_summaries_every²"
     - Lines 129-133: Update save() docstring references from "dt_save²" to "sample_summaries_every²"
   - Integration: Compiled device function uses this for second derivative scaling

5. **Update d2xdt2_min.py build() method**
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_min.py
   - Action: Modify
   - Details:
     - Line 58: Change `dt_save = self.compile_settings.dt_save` → `sample_summaries_every = self.compile_settings.sample_summaries_every`
     - Line 134: Update save function to use `sample_summaries_every`:
       ```python
       output_array[0] = buffer[2] / (precision(sample_summaries_every) * precision(sample_summaries_every))
       ```
     - Line 30-31: Update class docstring comment "scaled by dt_save²" → "scaled by sample_summaries_every²"
     - Line 55-56: Update build() docstring "scales by dt_save²" → "scales by sample_summaries_every²"
     - Lines 129-133: Update save() docstring references from "dt_save²" to "sample_summaries_every²"
   - Integration: Compiled device function uses this for second derivative scaling

6. **Update d2xdt2_extrema.py build() method**
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py
   - Action: Modify
   - Details:
     - Line 58: Change `dt_save = self.compile_settings.dt_save` → `sample_summaries_every = self.compile_settings.sample_summaries_every`
     - Lines 140-142: Update save function to use `sample_summaries_every`:
       ```python
       dt_save_sq = precision(sample_summaries_every) * precision(sample_summaries_every)
       ```
       Note: Keep variable name `dt_save_sq` as local variable name or rename to `sample_interval_sq` for consistency
     - Line 29-30: Update class docstring comment "scaled by dt_save²" → "scaled by sample_summaries_every²"
     - Line 54-55: Update build() docstring "scales by dt_save²" → "scales by sample_summaries_every²"
     - Lines 135-139: Update save() docstring references from "dt_save²" to "sample_summaries_every²"
   - Integration: Compiled device function uses this for second derivative scaling

**Tests to Create**:
- No new tests needed; existing tests validate derivative metrics

**Tests to Run**:
- tests/outputhandling/test_output_functions.py
- tests/outputhandling/test_summarymetrics.py (if exists)

**Outcomes**: 
- Files Modified: 
  * src/cubie/outputhandling/summarymetrics/dxdt_max.py (8 lines changed)
  * src/cubie/outputhandling/summarymetrics/dxdt_min.py (8 lines changed)
  * src/cubie/outputhandling/summarymetrics/dxdt_extrema.py (8 lines changed)
  * src/cubie/outputhandling/summarymetrics/d2xdt2_max.py (10 lines changed)
  * src/cubie/outputhandling/summarymetrics/d2xdt2_min.py (10 lines changed)
  * src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py (10 lines changed)
- Functions/Methods Added/Modified:
  * DxdtMax.build(): renamed `dt_save` to `sample_summaries_every`, updated save() scaling
  * DxdtMin.build(): renamed `dt_save` to `sample_summaries_every`, updated save() scaling
  * DxdtExtrema.build(): renamed `dt_save` to `sample_summaries_every`, updated save() scaling
  * D2xdt2Max.build(): renamed `dt_save` to `sample_summaries_every`, added `sample_interval_sq` local variable
  * D2xdt2Min.build(): renamed `dt_save` to `sample_summaries_every`, added `sample_interval_sq` local variable
  * D2xdt2Extrema.build(): renamed `dt_save` to `sample_summaries_every`, renamed `dt_save_sq` to `sample_interval_sq`
- Implementation Summary:
  Renamed all occurrences of `dt_save` to `sample_summaries_every` in all 6 derivative metric files. Updated class docstrings, build() method docstrings, and save() function docstrings. For second derivative metrics, updated the squared scaling variable from `dt_save_sq` to `sample_interval_sq` for consistency.
- Issues Flagged: None

---

## Task Group 3: OutputConfig Class Update
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/outputhandling/output_config.py (entire file)
- File: .github/copilot-instructions.md (attrs classes pattern)

**Input Validation Required**:
- `_sample_summaries_every`: Use `opt_gttype_validator(float, 0.0)` (same as `_save_every`)

**Tasks**:
1. **Add _sample_summaries_every attribute to OutputConfig**
   - File: src/cubie/outputhandling/output_config.py
   - Action: Modify
   - Details:
     - After line 185 (after `_save_every` definition), add new attribute:
       ```python
       _sample_summaries_every: Optional[float] = attrs.field(
           default=None,
           validator=opt_gttype_validator(float, 0.0)
       )
       ```
   - Integration: OutputFunctions.build() will access this property

2. **Add sample_summaries_every property to OutputConfig**
   - File: src/cubie/outputhandling/output_config.py
   - Action: Modify
   - Details:
     - After the `save_every` property (around line 668), add new property:
       ```python
       @property
       def sample_summaries_every(self) -> float:
           """Time interval between summary metric samples.
           
           Returns the configured sample_summaries_every value, or defaults
           to save_every if not explicitly set.
           """
           if self._sample_summaries_every is None:
               return self._save_every
           return self._sample_summaries_every
       ```
   - Edge cases: When `_sample_summaries_every` is None, defaults to `save_every`
   - Integration: OutputFunctions.build() reads this for metric compilation

3. **Update OutputConfig.from_loop_settings() classmethod**
   - File: src/cubie/outputhandling/output_config.py
   - Action: Modify
   - Details:
     - Line 906: Add parameter `sample_summaries_every: Optional[float] = None,` after `save_every`
     - Lines 960-970: Add `sample_summaries_every=sample_summaries_every,` to the cls() call
     - Update docstring to document the new parameter
   - Integration: IVPLoop uses this to create OutputConfig

4. **Update OutputConfig class docstring**
   - File: src/cubie/outputhandling/output_config.py
   - Action: Modify
   - Details:
     - Add `sample_summaries_every` to the Parameters section in the class docstring (around line 130)
     - Description: "Time between summary metric samples. Defaults to save_every."

**Tests to Create**:
- No new tests needed; integration tests will validate

**Tests to Run**:
- tests/outputhandling/test_output_config.py (if exists)
- tests/outputhandling/test_output_functions.py

**Outcomes**: 
- Files Modified: 
  * src/cubie/outputhandling/output_config.py (18 lines changed)
- Functions/Methods Added/Modified:
  * OutputConfig class: added `_sample_summaries_every` attribute with validation
  * OutputConfig.sample_summaries_every property: new property that defaults to save_every when not set
  * OutputConfig.from_loop_settings(): added `sample_summaries_every` parameter and passed to cls()
  * OutputConfig class docstring: added `sample_summaries_every` parameter documentation
- Implementation Summary:
  Added the new `_sample_summaries_every` attribute to OutputConfig with the same validator as `_save_every`. Created a property that returns the configured value or falls back to `save_every` when None. Updated the `from_loop_settings()` classmethod to accept and pass through the new parameter. Updated class and method docstrings.
- Issues Flagged: None 


---

## Task Group 4: OutputFunctions Class Update
**Status**: [x]
**Dependencies**: Task Group 1, Task Group 3

**Required Context**:
- File: src/cubie/outputhandling/output_functions.py (entire file)
- File: src/cubie/outputhandling/output_config.py (lines 664-680, sample_summaries_every property)

**Input Validation Required**:
- None; validation happens in OutputConfig

**Tasks**:
1. **Add sample_summaries_every to ALL_OUTPUT_FUNCTION_PARAMETERS**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     - Lines 28-38: Add `"sample_summaries_every"` to the set:
       ```python
       ALL_OUTPUT_FUNCTION_PARAMETERS = {
           "output_types",
           "saved_states", "saved_observables",
           "summarised_states", "summarised_observables",
           "saved_state_indices",
           "saved_observable_indices",
           "summarised_state_indices",
           "summarised_observable_indices",
           "save_every",
           "sample_summaries_every",  # Add this line
           "precision",
       }
       ```
   - Integration: Allows filtering kwargs for OutputFunctions

2. **Update OutputFunctions.build() to use sample_summaries_every**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     - Line 202: Change from:
       ```python
       summary_metrics.update(dt_save=config.save_every, precision=config.precision)
       ```
       To:
       ```python
       summary_metrics.update(sample_summaries_every=config.sample_summaries_every, precision=config.precision)
       ```
   - Integration: Propagates sample_summaries_every to all registered metrics

**Tests to Create**:
- No new tests needed; existing tests validate the integration

**Tests to Run**:
- tests/outputhandling/test_output_functions.py

**Outcomes**:
- Files Modified: 
  * src/cubie/outputhandling/output_functions.py (5 lines changed)
- Functions/Methods Added/Modified:
  * ALL_OUTPUT_FUNCTION_PARAMETERS: added `"sample_summaries_every"` entry
  * OutputFunctions.build(): updated summary_metrics.update() call to use `sample_summaries_every=config.sample_summaries_every`
- Implementation Summary:
  Added `"sample_summaries_every"` to the ALL_OUTPUT_FUNCTION_PARAMETERS set to allow filtering kwargs. Updated OutputFunctions.build() to call summary_metrics.update() with the new `sample_summaries_every` parameter name instead of `dt_save`, using `config.sample_summaries_every` property from OutputConfig (added in Task Group 3).
- Issues Flagged: None 


---

## Task Group 5: Test Utility Functions Update
**Status**: [x]
**Dependencies**: Task Groups 1-4

**Required Context**:
- File: tests/_utils.py (lines 145-220, calculate_expected_summaries function)
- File: tests/_utils.py (lines 221-542, calculate_single_summary_array function)

**Input Validation Required**:
- None; these are test utilities

**Tasks**:
1. **Rename calculate_expected_summaries parameters**
   - File: tests/_utils.py
   - Action: Modify
   - Details:
     - Line 150: Rename parameter `summarise_every` → `samples_per_summary`
     - Line 154: Rename parameter `dt_save` → `sample_summaries_every`
     - Line 163-166: Update docstring:
       - `summarise_every` → `samples_per_summary: Number of samples to summarise over (batch size)`
       - `dt_save` → `sample_summaries_every: Time between summary samples (for derivative calculations)`
     - Line 186: Update usage `int(saved_samples / summarise_every)` → `int(saved_samples / samples_per_summary)`
     - Line 216: Update call to calculate_single_summary_array:
       ```python
       calculate_single_summary_array(_input_array, samples_per_summary,
                                      summary_height_per_variable,
                                      output_types,
                                      output_array=_output_array,
                                      sample_summaries_every=sample_summaries_every,
                                      peak_index_offset=peak_index_offset)
       ```
   - Integration: Called by run_reference_loop

2. **Rename calculate_single_summary_array parameters**
   - File: tests/_utils.py
   - Action: Modify
   - Details:
     - Line 223: Rename parameter `summarise_every` → `samples_per_summary`
     - Line 227: Rename parameter `dt_save` → `sample_summaries_every`
     - Lines 232-239: Update docstring:
       - `summarise_every` → `samples_per_summary: Number of samples to summarise over`
       - `dt_save` → `sample_summaries_every: Time between summary samples (for derivative calculations)`
     - Line 247: Update usage `int(input_array.shape[0] / summarise_every)` → `int(input_array.shape[0] / samples_per_summary)`
     - Line 259: Update `i * summarise_every` → `i * samples_per_summary`
     - Line 260: Update `(i + 1) * summarise_every` → `(i + 1) * samples_per_summary`
     - Line 273: Update `i * summarise_every` → `i * samples_per_summary`
     - Line 353: Update `i * summarise_every` → `i * samples_per_summary`
     - All derivative calculation usages (lines 434, 435, 445, 449, 459, 463, 478, 484, 498, 504, 518, 524):
       Update `dt_save` → `sample_summaries_every`
   - Integration: Called by calculate_expected_summaries

**Tests to Create**:
- No new tests needed; existing integration tests validate

**Tests to Run**:
- tests/integrators/test_loops.py (uses these functions)

**Outcomes**: 
- Files Modified: 
  * tests/_utils.py (20+ lines changed)
- Functions/Methods Added/Modified:
  * calculate_expected_summaries(): renamed parameter `summarise_every` to `samples_per_summary`, renamed parameter `dt_save` to `sample_summaries_every`
  * calculate_single_summary_array(): renamed parameter `summarise_every` to `samples_per_summary`, renamed parameter `dt_save` to `sample_summaries_every`
- Implementation Summary:
  Renamed all occurrences of `summarise_every` to `samples_per_summary` and `dt_save` to `sample_summaries_every` in both test utility functions. Updated docstrings to reflect new parameter names. Updated all internal usages of these parameters throughout the function bodies.
- Issues Flagged: None

---

## Task Group 6: CPU Reference Loop Update
**Status**: [x]
**Dependencies**: Task Group 5

**Required Context**:
- File: tests/integrators/cpu_reference/loops.py (entire file)

**Input Validation Required**:
- None; test utility code

**Tasks**:
1. **Update run_reference_loop call to calculate_expected_summaries**
   - File: tests/integrators/cpu_reference/loops.py
   - Action: Modify
   - Details:
     - Lines 217-228: Update the call to calculate_expected_summaries with new parameter names:
       ```python
       state_summary, observable_summary = calculate_expected_summaries(
           state_output,
           observables_output,
           summarised_state_indices,
           summarised_observable_indices,
           samples_per_summary,  # Already correct variable name
           output_functions.compile_settings.output_types,
           output_functions.summaries_output_height_per_var,
           precision,
           sample_summaries_every=sample_summaries_every,  # Renamed from dt_save=save_every
           exclude_first=True,
       )
       ```
     - Note: The `sample_summaries_every` variable is already defined on line 93-96
     - Note: The `samples_per_summary` calculation on line 215 is already correct
   - Integration: Used by integration tests to generate reference outputs

**Tests to Create**:
- No new tests needed

**Tests to Run**:
- tests/integrators/test_loops.py

**Outcomes**:
- Files Modified: 
  * tests/integrators/cpu_reference/loops.py (1 line changed)
- Functions/Methods Added/Modified:
  * run_reference_loop(): updated calculate_expected_summaries call parameter
- Implementation Summary:
  Changed the keyword argument `dt_save=save_every` to `sample_summaries_every=sample_summaries_every` in the call to calculate_expected_summaries within run_reference_loop(). The variable `sample_summaries_every` was already defined earlier in the function.
- Issues Flagged: None 


---

## Summary

### Total Task Groups: 6
### Dependency Chain:
```
Task Group 1 (Core Metrics) 
    ↓
Task Group 2 (Derivative Metrics) ←──┐
Task Group 3 (OutputConfig)      ←──┤── All depend on Group 1
    ↓                               │
Task Group 4 (OutputFunctions) ─────┘
    ↓
Task Group 5 (Test Utilities)
    ↓
Task Group 6 (CPU Reference Loop)
```

### Tests to Create: None (existing tests cover the refactor)

### Tests to Run (in order):
1. `tests/outputhandling/test_output_functions.py`
2. `tests/outputhandling/test_output_config.py` (if exists)
3. `tests/outputhandling/test_summarymetrics.py` (if exists)
4. `tests/integrators/test_loops.py`

### Estimated Complexity: Low-Medium
- Primarily parameter and property renames
- No new functionality
- No architectural changes
- Clear one-to-one mappings throughout
