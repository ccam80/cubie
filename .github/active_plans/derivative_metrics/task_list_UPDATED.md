# Implementation Task List
# Feature: Derivative-Based Output Metrics  
# Plan Reference: .github/active_plans/derivative_metrics/agent_plan.md

## Overview

This task list implements six new summary metrics (dxdt_max, dxdt_min, dxdt_extrema, d2xdt2_max, d2xdt2_min, d2xdt2_extrema) that compute derivatives via finite differences. The implementation uses the CUDAFactory compile_settings pattern to capture dt_save in closure, avoiding signature changes to existing metrics.

**Key Architecture Decisions:**
- dt_save captured in closure via CUDAFactory compile_settings pattern  
- NO changes to existing metric signatures (build() stays `build(self)`)
- NO changes to device function signatures (customisable_variable stays int32)
- dt_save NOT stored in buffers (smaller memory footprint)
- Predicated commit pattern (selp) used instead of if/else for warp efficiency

**Dependencies:**
Task groups can be executed with some parallelism where noted.

---

## Task Group 1: Add dt_save to OutputConfig and Infrastructure - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/outputhandling/output_config.py (lines 94-200, OutputConfig class definition)
- File: src/cubie/outputhandling/output_functions.py (lines 29-37, ALL_OUTPUT_FUNCTION_PARAMETERS)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 203-216, __attrs_post_init__)

**Input Validation Required**:
- dt_save: NO validation required (assume validated at higher level)

**Tasks**:

### 1.1 Add dt_save Field to OutputConfig
   - File: src/cubie/outputhandling/output_config.py
   - Action: Modify
   - Details:
     Add dt_save field to OutputConfig attrs class after max_observables field.
     Store as `_dt_save` with float validator, expose via property.
     Default value 0.01 for backwards compatibility.
     No validation beyond type check (assume validated at higher level).
   - Edge cases: Default value ensures backwards compatibility
   - Integration: Available via self.compile_settings.dt_save in OutputFunctions

### 1.2 Add dt_save to ALL_OUTPUT_FUNCTION_PARAMETERS
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     Add "dt_save" string to the ALL_OUTPUT_FUNCTION_PARAMETERS set at line 29.
     This allows dt_save to flow through to OutputConfig.
   - Edge cases: None
   - Integration: Enables parameter passing through configuration

### 1.3 Update Combined Metrics Registry
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     In __attrs_post_init__ method (line 203), add two new entries to _combined_metrics dict:
     - `frozenset(["dxdt_max", "dxdt_min"]): "dxdt_extrema"`
     - `frozenset(["d2xdt2_max", "d2xdt2_min"]): "d2xdt2_extrema"`
   - Edge cases: Both constituents must be present for substitution
   - Integration: Auto-substitution logic already handles this

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 2: Update OutputFunctions to Pass dt_save to Metric Build - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/outputhandling/output_functions.py (lines 176-222, build() method)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (lines 120-151, build() and properties)

**Input Validation Required**:
- None (dt_save accessed from compile_settings)

**Tasks**:

### 2.1 Modify SummaryMetric.build() Abstract Method Signature
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     Change abstract method signature from `build(self) -> MetricFuncCache` 
     to `build(self, dt_save: float) -> MetricFuncCache`.
     Update docstring to document dt_save parameter.
     Note that dt_save can be captured in closure when compiling device functions.
   - Edge cases: All metrics must update to match new signature
   - Integration: All subclasses will be updated in later groups

### 2.2 Update SummaryMetric Properties to Pass dt_save
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     The update_device_func and save_device_func properties call build().
     These need access to dt_save, which should come from wherever the
     metric is being compiled. This requires investigation of call chain.
     
     CRITICAL DESIGN DECISION NEEDED: How does dt_save reach metric.build()?
     - Option A: Store dt_save in metric instance when compiling
     - Option B: Pass through registry methods (update_functions, save_functions)
     - Option C: Access from nonlocal scope in factory functions
   - Edge cases: Thread-safety if dt_save stored per-instance
   - Integration: Affects how metrics are compiled in factories

### 2.3 Update Metric Compilation in OutputFunctions
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     In build() method (line 176), extract dt_save from config.
     Pass dt_save to update_summary_factory and save_summary_factory.
     These factories will pass dt_save when calling metric.build().
   - Edge cases: dt_save must be available in compile_settings
   - Integration: Requires updating factory signatures

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 3: Update ALL Existing Metrics to Accept dt_save Parameter - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/max.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/min.py (entire file)
- All other existing metric files

**Input Validation Required**:
- dt_save: No validation (parameter unused by existing metrics)

**Tasks**:

### 3.1 Update max.py build() Signature
   - File: src/cubie/outputhandling/summarymetrics/max.py
   - Action: Modify
   - Details:
     Change `def build(self) -> MetricFuncCache:` to 
     `def build(self, dt_save: float) -> MetricFuncCache:`
     Add parameter to docstring noting it's unused for max metric.
     NO changes to device function signatures (stay int32 for customisable_variable).
   - Edge cases: dt_save unused but must accept parameter
   - Integration: Matches new base class signature

### 3.2 Update Remaining Existing Metrics (11 files)
   - Files: min.py, mean.py, rms.py, std.py, peaks.py, negative_peaks.py, 
     max_magnitude.py, extrema.py, mean_std.py, std_rms.py, mean_std_rms.py
   - Action: Modify each
   - Details: Same pattern as max.py - add dt_save parameter to build(), 
     document as unused, no other changes
   - Edge cases: None
   - Integration: All metrics now consistent with base class

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 4: Update Summary Factories to Pass dt_save - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 2, 3

**Required Context**:
- File: src/cubie/outputhandling/update_summaries.py (lines 60-150)
- File: src/cubie/outputhandling/save_summaries.py (entire file)
- File: src/cubie/outputhandling/summarymetrics/metrics.py (update_functions, save_functions methods)

**Input Validation Required**:
- dt_save: Type float (passed from OutputFunctions)

**Tasks**:

### 4.1 Update update_summary_factory Signature and Implementation
   - File: src/cubie/outputhandling/update_summaries.py
   - Action: Modify
   - Details:
     Add dt_save parameter to factory signature.
     Pass dt_save when calling summary_metrics.update_functions().
     The registry should pass dt_save to each metric's build() method.
     dt_save will be captured in closure when metrics compile device functions.
   - Edge cases: dt_save must be in scope for all metrics
   - Integration: Requires updating SummaryMetrics.update_functions()

### 4.2 Update save_summary_factory Signature and Implementation
   - File: src/cubie/outputhandling/save_summaries.py
   - Action: Modify
   - Details: Same pattern as update_summary_factory
   - Edge cases: Same as update factory
   - Integration: Consistent with update factory

### 4.3 Update SummaryMetrics.update_functions() to Accept dt_save
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details:
     Add dt_save parameter to update_functions() method.
     Pass dt_save to each metric's build() when retrieving update functions.
     Ensure dt_save is captured in closure scope for device functions.
   - Edge cases: Must handle metrics compiled on-demand
   - Integration: Core pattern for dt_save propagation

### 4.4 Update SummaryMetrics.save_functions() to Accept dt_save
   - File: src/cubie/outputhandling/summarymetrics/metrics.py
   - Action: Modify
   - Details: Same pattern as update_functions()
   - Edge cases: Same as update_functions()
   - Integration: Consistent with update_functions()

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 5: Implement First Derivative Metrics with Predicated Commit - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1-4

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/max.py (lines 1-122) - Reference pattern
- File: src/cubie/outputhandling/summarymetrics/extrema.py (lines 1-128) - Combined metric pattern
- File: .github/copilot-instructions.md (lines 43-58) - Predicated commit pattern

**Input Validation Required**:
- None (dt_save from parameter, no validation)

**Tasks**:

### 5.1 Create dxdt_max.py
   - File: src/cubie/outputhandling/summarymetrics/dxdt_max.py
   - Action: Create
   - Details:
     Implement DxdtMax class following max.py pattern.
     Buffer size: 2 (prev_value, max_unscaled).
     Output size: 1.
     build(self, dt_save) captures dt_save in closure.
     update() uses predicated commit: `update_max = (buffer[0] != 0.0) and (unscaled_dxdt > buffer[1])`
     then `buffer[1] = selp(update_max, unscaled_dxdt, buffer[1])`.
     save() scales by dt_save from closure: `output_array[0] = buffer[1] / dt_save`.
     Import selp from numba.cuda.
   - Edge cases: First save outputs 0.0, buffer[0]==0.0 guard for initialization
   - Integration: Auto-registers via decorator

### 5.2 Create dxdt_min.py
   - File: src/cubie/outputhandling/summarymetrics/dxdt_min.py
   - Action: Create
   - Details:
     Same as dxdt_max.py but track minimum.
     update_min = (buffer[0] != 0.0) and (unscaled_dxdt < buffer[1]).
     Buffer size: 2 (prev_value, min_unscaled).
   - Edge cases: Same as dxdt_max
   - Integration: Auto-registers via decorator

### 5.3 Create dxdt_extrema.py
   - File: src/cubie/outputhandling/summarymetrics/dxdt_extrema.py
   - Action: Create
   - Details:
     Combined metric for max and min.
     Buffer size: 3 (prev_value, max_unscaled, min_unscaled).
     Output size: 2.
     update() uses two predicated commits for max and min.
     save() scales both by dt_save.
   - Edge cases: Same as individual metrics
   - Integration: Auto-substituted via _combined_metrics registry

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 6: Implement Second Derivative Metrics with Predicated Commit - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 1-4

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/dxdt_max.py (from Group 5)
- File: .github/copilot-instructions.md (lines 43-58) - Predicated commit pattern

**Input Validation Required**:
- None

**Tasks**:

### 6.1 Create d2xdt2_max.py
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_max.py
   - Action: Create
   - Details:
     Implement D2xdt2Max class.
     Buffer size: 3 (prev_value, prev_prev_value, max_unscaled).
     Output size: 1.
     update() computes unscaled_d2xdt2 = value - 2.0 * buffer[0] + buffer[1].
     Three-state guard: have_prev = buffer[0] != 0.0, have_prev_prev = buffer[1] != 0.0.
     update_max = have_prev and have_prev_prev and (unscaled_d2xdt2 > buffer[2]).
     Predicated commits for max and history shift.
     save() scales by dt_save²: `output_array[0] = buffer[2] / (dt_save * dt_save)`.
   - Edge cases: First two saves output 0.0, three-state initialization
   - Integration: Auto-registers via decorator

### 6.2 Create d2xdt2_min.py
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_min.py
   - Action: Create
   - Details:
     Same as d2xdt2_max.py but track minimum.
     update_min = have_prev and have_prev_prev and (unscaled_d2xdt2 < buffer[2]).
     Buffer size: 3 (prev_value, prev_prev_value, min_unscaled).
   - Edge cases: Same as d2xdt2_max
   - Integration: Auto-registers via decorator

### 6.3 Create d2xdt2_extrema.py
   - File: src/cubie/outputhandling/summarymetrics/d2xdt2_extrema.py
   - Action: Create
   - Details:
     Combined metric for max and min second derivatives.
     Buffer size: 4 (prev_value, prev_prev_value, max_unscaled, min_unscaled).
     Output size: 2.
     update() uses two predicated commits for max and min.
     save() scales both by dt_save².
   - Edge cases: Same as individual metrics
   - Integration: Auto-substituted via _combined_metrics registry

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 7: Update Module Imports - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 5, 6

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/__init__.py (entire file)

**Input Validation Required**:
- None

**Tasks**:

### 7.1 Add New Metric Imports to __init__.py
   - File: src/cubie/outputhandling/summarymetrics/__init__.py
   - Action: Modify
   - Details:
     Add imports in alphabetical order:
     - `from cubie.outputhandling.summarymetrics.d2xdt2_extrema import D2xdt2Extrema`
     - `from cubie.outputhandling.summarymetrics.d2xdt2_max import D2xdt2Max`
     - `from cubie.outputhandling.summarymetrics.d2xdt2_min import D2xdt2Min`
     - `from cubie.outputhandling.summarymetrics.dxdt_extrema import DxdtExtrema`
     - `from cubie.outputhandling.summarymetrics.dxdt_max import DxdtMax`
     - `from cubie.outputhandling.summarymetrics.dxdt_min import DxdtMin`
   - Edge cases: None
   - Integration: Triggers auto-registration

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 8: Add Tests for New Metrics - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Groups 1-7

**Required Context**:
- File: tests/outputhandling/summarymetrics/test_summary_metrics.py (entire file)
- File: tests/outputhandling/test_output_functions.py (search for long_run tests)

**Input Validation Required**:
- None (test fixtures handle setup)

**Tasks**:

### 8.1 Add Derivative Metric Test Cases to Existing Tests
   - File: tests/outputhandling/summarymetrics/test_summary_metrics.py
   - Action: Modify
   - Details:
     Locate existing parametrized tests that validate all metrics.
     Add new metrics to test parameters: dxdt_max, dxdt_min, dxdt_extrema, 
     d2xdt2_max, d2xdt2_min, d2xdt2_extrema.
     The existing test framework should auto-discover and test new metrics.
     NO new test files or functions needed.
   - Edge cases: First derivative outputs may be 0.0 for first save
   - Integration: Uses existing test infrastructure

### 8.2 Verify Combined Metric Substitution
   - File: tests/outputhandling/summarymetrics/test_summary_metrics.py
   - Action: Modify
   - Details:
     Add test cases to verify:
     - ["dxdt_max", "dxdt_min"] substitutes to ["dxdt_extrema"]
     - ["d2xdt2_max", "d2xdt2_min"] substitutes to ["d2xdt2_extrema"]
     Follow existing pattern for extrema substitution test.
   - Edge cases: Order independence
   - Integration: Tests registry substitution logic

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Task Group 9: Refactor Existing Metrics to Use Predicated Commit (OPTIONAL) - PARALLEL
**Status**: [ ]
**Dependencies**: None (independent improvement)

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/max.py (lines 59-83)
- File: src/cubie/outputhandling/summarymetrics/min.py (update function)
- File: src/cubie/outputhandling/summarymetrics/extrema.py (update function)
- File: .github/copilot-instructions.md (lines 43-58) - Predicated commit pattern

**Input Validation Required**:
- None

**Tasks**:

### 9.1 Refactor max.py to Use Predicated Commit
   - File: src/cubie/outputhandling/summarymetrics/max.py
   - Action: Modify
   - Details:
     Import selp from numba.cuda.
     Replace `if value > buffer[0]: buffer[0] = value` 
     with `update_max = value > buffer[0]; buffer[0] = selp(update_max, value, buffer[0])`.
   - Edge cases: Behavior unchanged, performance improvement only
   - Integration: No API changes

### 9.2 Refactor min.py to Use Predicated Commit
   - File: src/cubie/outputhandling/summarymetrics/min.py
   - Action: Modify
   - Details:
     Same pattern: `update_min = value < buffer[0]; buffer[0] = selp(update_min, value, buffer[0])`.
   - Edge cases: Behavior unchanged
   - Integration: No API changes

### 9.3 Refactor extrema.py to Use Predicated Commit
   - File: src/cubie/outputhandling/summarymetrics/extrema.py
   - Action: Modify
   - Details:
     Replace if/else for max and min with two predicated commits.
   - Edge cases: Behavior unchanged
   - Integration: No API changes

### 9.4 Review Other Metrics for if/else Patterns
   - Files: peaks.py, negative_peaks.py, max_magnitude.py
   - Action: Review and refactor if applicable
   - Details: Identify conditional assignments and convert to predicated commits
   - Edge cases: Preserve exact behavior
   - Integration: Performance optimization only

**Outcomes**:
[Empty - to be filled by do_task agent]

---

## Summary

**Total Task Groups**: 9 (8 required, 1 optional)
**Execution Strategy**: Groups 5-6 and 9 can run in parallel; others sequential
**Estimated Complexity**: Medium

**Dependency Chain:**
1. Group 1: Add dt_save to OutputConfig and registry
2. Group 2: Update metric build() signature and OutputFunctions
3. Group 3: Update all existing metrics to accept dt_save
4. Group 4: Update factories to pass dt_save
5. Groups 5-6: Implement new metrics (can run in parallel)
6. Group 7: Update imports
7. Group 8: Add tests
8. Group 9: (Optional) Refactor existing metrics for performance

**Critical Changes from Original Plan:**
- NO breaking changes to existing metric signatures at device level
- dt_save captured in closure, NOT passed as device function parameter
- Smaller buffer sizes (dt_save not stored)
- Predicated commit pattern for warp efficiency
- Simpler testing (use existing infrastructure)

**Files to Create**: 6
- dxdt_max.py, dxdt_min.py, dxdt_extrema.py
- d2xdt2_max.py, d2xdt2_min.py, d2xdt2_extrema.py

**Files to Modify**:
- Infrastructure: output_config.py, output_functions.py, metrics.py (3 files)
- Factories: update_summaries.py, save_summaries.py (2 files)
- Existing metrics: 12 files (add dt_save parameter to build())
- Imports: __init__.py (1 file)
- Tests: test_summary_metrics.py (1 file)
- Optional: max.py, min.py, extrema.py, others (performance)
- **Total: 19-22 files**

**Breaking Changes**: None at device level, minimal at Python API level (build() signature)

**Rollback Strategy**:
1. Revert Group 8 (tests)
2. Revert Group 7 (imports)
3. Revert Groups 5-6 (new metrics)
4. Revert Group 4 (factories)
5. Revert Group 3 (existing metrics)
6. Revert Groups 1-2 (infrastructure)

**Parallel Execution Opportunities**:
- Groups 5 and 6 can run in parallel (independent metric implementations)
- Group 9 can run anytime (independent refactoring)
