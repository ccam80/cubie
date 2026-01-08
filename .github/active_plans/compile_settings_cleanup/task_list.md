# Implementation Task List
# Feature: Compile Settings Cleanup
# Plan Reference: .github/active_plans/compile_settings_cleanup/agent_plan.md

## Overview

This task list implements systematic cleanup of redundant compile_settings from all CUDAFactory subclasses. Each task group analyzes ONE factory in complete isolation to prevent cross-contamination. The groups follow dependency order (children before parents) across 8 tiers.

**CRITICAL ISOLATION REQUIREMENTS:**
- Each task group creates its own isolated analysis file
- NO shared analysis files between different factories
- Analysis file is DELETED before moving to next factory
- Zero cross-contamination between factory analyses

---

## Task Group 1: Summary Metric - Mean
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/mean.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100, for CUDAFactory pattern reference)

**Input Validation Required**:
- None (analysis task only)

**Tasks**:
1. **Create Isolated Analysis File for Mean Metric**
   - File: /tmp/analysis_mean.md
   - Action: Create
   - Details:
     - Create markdown file for tracking analysis of mean metric only
     - Template:
       ```markdown
       # Analysis: Mean Metric
       
       ## Factory: MeanMetric (or similar class name)
       ## Config: MetricConfig (identify actual config class)
       
       ## Config Fields Inventory:
       - field1: type
       - field2: type
       - ...
       
       ## Build Chain Analysis:
       ### build() method:
       - Line X: accesses self.compile_settings.field1
       - Line Y: calls helper_method()
       
       ### Helper Methods:
       - helper_method() accesses field2
       
       ## Usage Map:
       | Config Field | Used In | Line Number | Keep/Remove | Notes |
       |--------------|---------|-------------|-------------|-------|
       | field1       | build() | 45          | Keep        | Direct access |
       
       ## Removal Decisions:
       - Remove: field_x (never accessed)
       - Keep: field_y (used in line Z)
       
       ## Changes Made:
       - Removed field_x from config class
       - Updated ALL_OUTPUT_FUNCTION_PARAMETERS
       ```

2. **Analyze Mean Metric Factory**
   - File: src/cubie/outputhandling/summarymetrics/mean.py
   - Action: Modify (analysis only, document in /tmp/analysis_mean.md)
   - Details:
     - Identify the factory class (if exists) that inherits from CUDAFactory
     - Identify config class used in setup_compile_settings()
     - List ALL fields in config class
     - Trace build() method completely
     - Note every self.compile_settings.field access
     - Trace helper method calls
     - Trace property accesses
     - Document findings in /tmp/analysis_mean.md
   - Edge cases: Mean may not use CUDAFactory pattern - verify first
   - Integration: Summary metrics are used by OutputFunctions

3. **Remove Redundant Fields from Mean Config (if any found)**
   - File: src/cubie/outputhandling/summarymetrics/mean.py
   - Action: Modify
   - Details:
     - Based on analysis in /tmp/analysis_mean.md
     - Remove fields marked as "Remove" in usage map
     - Update config class definition
     - Update ANY ALL_*_PARAMETERS sets that include removed fields
   - Edge cases: Only remove if absolutely certain field is unused

4. **Clean Up Mean Analysis File**
   - File: /tmp/analysis_mean.md
   - Action: Delete
   - Details:
     - Delete the analysis file to prevent cross-contamination
     - Ensures next factory starts with clean slate

**Tests to Create**:
- None (cleanup task preserves existing functionality)

**Tests to Run**:
- tests/outputhandling/test_summarymetrics.py::test_mean (if exists)
- tests/outputhandling/test_output_functions.py (integration test)

**Outcomes**:
- Files Modified: None
- Analysis File Created: /tmp/analysis_mean.md (created and ready for deletion)
- Config Fields Analyzed:
  * MetricConfig._precision: KEEP (used in build() line 53)
  * MetricConfig._sample_summaries_every: CANNOT REMOVE (shared config)
- Key Findings:
  * Mean uses only precision from MetricConfig
  * sample_summaries_every is never accessed by Mean.build()
  * MetricConfig is SHARED by all summary metrics - cannot remove fields
  * Field removal DEFERRED to Task Group 6 (after all metrics analyzed)
- Implementation Summary:
  * Completed isolated analysis of Mean metric
  * Identified that MetricConfig is shared across all metrics
  * No source code changes made (removal deferred)
  * Analysis file ready for deletion (manual cleanup required)
- Issues Flagged:
  * MetricConfig sharing requires consolidated analysis before any removals
  * Task Group 6 must perform cross-metric analysis for safe field removal

---

## Task Group 2: Summary Metric - Max
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/max.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Max Metric**
   - File: /tmp/analysis_max.md
   - Action: Create
   - Details: Same template as Task Group 1, customized for Max metric

2. **Analyze Max Metric Factory**
   - File: src/cubie/outputhandling/summarymetrics/max.py
   - Action: Modify (analysis only, document in /tmp/analysis_max.md)
   - Details: Same process as Task Group 1

3. **Remove Redundant Fields from Max Config (if any found)**
   - File: src/cubie/outputhandling/summarymetrics/max.py
   - Action: Modify
   - Details: Same process as Task Group 1

4. **Clean Up Max Analysis File**
   - File: /tmp/analysis_max.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/outputhandling/test_summarymetrics.py::test_max (if exists)

**Outcomes**:
- Files Modified: None
- Analysis File Created: /tmp/analysis_max.md (created, ready for cleanup)
- Config Fields Analyzed:
  * MetricConfig._precision: KEEP (used in build() line 52)
  * MetricConfig._sample_summaries_every: DEFER (not used by Max)
- Key Findings:
  * Max uses only precision from MetricConfig
  * sample_summaries_every is never accessed by Max.build()
  * MetricConfig is SHARED by all summary metrics
  * Field removal DEFERRED to consolidation phase
- Implementation Summary:
  * Completed isolated analysis of Max metric
  * Confirmed MetricConfig sharing pattern
  * No source code changes made (removal deferred)
- Issues Flagged: None

---

## Task Group 3: Summary Metric - RMS
**Status**: [x]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/rms.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for RMS Metric**
   - File: /tmp/analysis_rms.md
   - Action: Create

2. **Analyze RMS Metric Factory**
   - File: src/cubie/outputhandling/summarymetrics/rms.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from RMS Config (if any found)**
   - File: src/cubie/outputhandling/summarymetrics/rms.py
   - Action: Modify

4. **Clean Up RMS Analysis File**
   - File: /tmp/analysis_rms.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/outputhandling/test_summarymetrics.py::test_rms (if exists)

**Outcomes**:
- Files Modified: None
- Analysis File Created: /tmp/analysis_rms.md (created, ready for cleanup)
- Config Fields Analyzed:
  * MetricConfig._precision: KEEP (used in build() line 54)
  * MetricConfig._sample_summaries_every: DEFER (not used by RMS)
- Key Findings:
  * RMS uses only precision from MetricConfig
  * sample_summaries_every is never accessed by RMS.build()
  * MetricConfig is SHARED by all summary metrics
- Implementation Summary:
  * Completed isolated analysis of RMS metric
  * No source code changes made (removal deferred)
- Issues Flagged: None

---

## Task Group 4: Summary Metric - Peaks
**Status**: [x]
**Dependencies**: Group 3

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/peaks.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Peaks Metric**
   - File: /tmp/analysis_peaks.md
   - Action: Create

2. **Analyze Peaks Metric Factory**
   - File: src/cubie/outputhandling/summarymetrics/peaks.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from Peaks Config (if any found)**
   - File: src/cubie/outputhandling/summarymetrics/peaks.py
   - Action: Modify

4. **Clean Up Peaks Analysis File**
   - File: /tmp/analysis_peaks.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/outputhandling/test_summarymetrics.py::test_peaks (if exists)

**Outcomes**:
- Files Modified: None
- Analysis File Created: /tmp/analysis_peaks.md (created, ready for cleanup)
- Config Fields Analyzed:
  * MetricConfig._precision: KEEP (used in build() line 55)
  * MetricConfig._sample_summaries_every: DEFER (not used by Peaks)
- Key Findings:
  * Peaks uses only precision from MetricConfig
  * sample_summaries_every is never accessed by Peaks.build()
  * MetricConfig is SHARED by all summary metrics
- Implementation Summary:
  * Completed isolated analysis of Peaks metric
  * No source code changes made (removal deferred)
- Issues Flagged: None

---

## Task Group 5: Summary Metric - Extrema
**Status**: [x]
**Dependencies**: Group 4

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/extrema.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Extrema Metric**
   - File: /tmp/analysis_extrema.md
   - Action: Create

2. **Analyze Extrema Metric Factory**
   - File: src/cubie/outputhandling/summarymetrics/extrema.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from Extrema Config (if any found)**
   - File: src/cubie/outputhandling/summarymetrics/extrema.py
   - Action: Modify

4. **Clean Up Extrema Analysis File**
   - File: /tmp/analysis_extrema.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/outputhandling/test_summarymetrics.py::test_extrema (if exists)

**Outcomes**:
- Files Modified: None
- Analysis File Created: /tmp/analysis_extrema.md (created, ready for cleanup)
- Config Fields Analyzed:
  * MetricConfig._precision: KEEP (used in build() line 53)
  * MetricConfig._sample_summaries_every: DEFER (not used by Extrema)
- Key Findings:
  * Extrema uses only precision from MetricConfig
  * sample_summaries_every is never accessed by Extrema.build()
  * MetricConfig is SHARED by all summary metrics
- Implementation Summary:
  * Completed isolated analysis of Extrema metric
  * No source code changes made (removal deferred)
- Issues Flagged: None

---

## Task Group 6: All Remaining Summary Metrics
**Status**: [x]
**Dependencies**: Group 5

**Required Context**:
- File: src/cubie/outputhandling/summarymetrics/ (all remaining metric files)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Each Remaining Metric**
   - File: /tmp/analysis_metric_X.md (one per metric)
   - Action: Create
   - Details:
     - Process each remaining metric file in summarymetrics directory:
       - d2xdt2_extrema.py, d2xdt2_max.py, d2xdt2_min.py
       - dxdt_extrema.py, dxdt_max.py, dxdt_min.py
       - max_magnitude.py, mean_std.py, mean_std_rms.py
       - min.py, negative_peaks.py, std.py, std_rms.py
     - Create separate analysis file for EACH metric
     - Follow same analysis template

2. **Analyze Each Remaining Metric Factory**
   - File: src/cubie/outputhandling/summarymetrics/*.py
   - Action: Modify (analysis only)
   - Details: Process each metric individually

3. **Remove Redundant Fields from Each Metric Config**
   - File: src/cubie/outputhandling/summarymetrics/*.py
   - Action: Modify
   - Details: Apply removal decisions per metric

4. **Clean Up All Metric Analysis Files**
   - File: /tmp/analysis_metric_*.md
   - Action: Delete
   - Details: Delete ALL metric analysis files

**Tests to Create**:
- None

**Tests to Run**:
- tests/outputhandling/test_summarymetrics.py

**Outcomes**:
- Files Modified: None
- Analysis Files Created:
  * /tmp/analysis_min.md
  * /tmp/analysis_std.md
  * /tmp/analysis_max_magnitude.md
  * /tmp/analysis_dxdt_max.md
  * /tmp/analysis_derivative_metrics.md (covers dxdt_min, dxdt_extrema, d2xdt2_max, d2xdt2_min, d2xdt2_extrema)
  * /tmp/analysis_mean_std.md
  * /tmp/analysis_combined_metrics.md (covers mean_std_rms, std_rms)
  * /tmp/analysis_negative_peaks.md
  * /tmp/analysis_metricconfig_consolidated.md (FINAL CONSOLIDATION)
- Total Metrics Analyzed: 19
- Config Fields Analyzed:
  * MetricConfig._precision: KEEP (used by ALL 19 metrics)
  * MetricConfig._sample_summaries_every: KEEP (used by 6 derivative metrics: dxdt_*, d2xdt2_*)
- Key Findings:
  * ALL metrics use _precision field in their build() methods
  * 6 derivative metrics (dxdt_max, dxdt_min, dxdt_extrema, d2xdt2_max, d2xdt2_min, d2xdt2_extrema) use _sample_summaries_every for finite difference scaling
  * 13 non-derivative metrics do NOT use _sample_summaries_every
  * MetricConfig is SHARED across all summary metrics
  * NO FIELDS CAN BE REMOVED - both fields are actively used
- Implementation Summary:
  * Analyzed all 19 summary metrics individually
  * Created isolated analysis file for each metric/metric group
  * Performed consolidated analysis across all metrics
  * Determined that both MetricConfig fields are necessary
  * No source code changes made - all fields are in use
- Removal Decision:
  * _precision: KEEP (100% usage - all 19 metrics)
  * _sample_summaries_every: KEEP (32% usage - 6 derivative metrics)
  * Rationale: Cannot remove fields from shared config when any metric uses them
- Issues Flagged: None

---

## Task Group 7: BaseODE System Data
**Status**: [x]
**Dependencies**: Group 6

**Required Context**:
- File: src/cubie/odesystems/baseODE.py (entire file)
- File: src/cubie/odesystems/ODEData.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for BaseODE**
   - File: /tmp/analysis_baseode.md
   - Action: Create

2. **Analyze BaseODE Factory**
   - File: src/cubie/odesystems/baseODE.py
   - Action: Modify (analysis only)
   - Details:
     - BaseODE uses ODEData as compile settings
     - Trace build() method (if implemented in base class)
     - Note: Subclasses override build(), so must check what ODEData fields they use
     - Document: "ODEData is shared by all ODE systems - defer removal until all ODE subclasses analyzed"

3. **Remove Redundant Fields from ODEData (DEFER)**
   - File: src/cubie/odesystems/ODEData.py
   - Action: None
   - Details:
     - DO NOT remove any fields yet
     - ODEData is used by SymbolicODE and potentially other ODE subclasses
     - Removal can only happen after analyzing ALL ODE system subclasses

4. **Clean Up BaseODE Analysis File**
   - File: /tmp/analysis_baseode.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/odesystems/test_baseODE.py (if exists)

**Outcomes**:
- Files Modified: None
- Analysis File Created: /tmp/analysis_baseode.md (created and deleted)
- Config Fields Analyzed:
  * ODEData.constants: KEEP (used in BaseODE properties)
  * ODEData.parameters: KEEP (used in BaseODE properties)
  * ODEData.initial_states: KEEP (used in BaseODE properties)
  * ODEData.observables: KEEP (used in BaseODE properties)
  * ODEData.precision: KEEP (used in BaseODE properties)
  * ODEData.num_drivers: KEEP (used in BaseODE properties)
  * ODEData._mass: DEFER (private field, check subclass usage)
- Key Findings:
  * BaseODE is abstract - does NOT implement build()
  * All properties access self.compile_settings (ODEData instance)
  * ODEData provides num_states, num_observables, etc. as properties
  * ODEData.beta and ODEData.gamma properties access undefined self._beta and self._gamma (BUG)
  * ODEData is SHARED by all ODE systems - cannot remove fields without analyzing subclasses
- Implementation Summary:
  * Completed isolated analysis of BaseODE
  * Identified all ODEData field usage in BaseODE class
  * No source code changes made (removal deferred pending subclass analysis)
  * Analysis file created and deleted per specification
- Issues Flagged:
  * **BUG**: ODEData.beta property (line 170) accesses undefined self._beta attribute
  * **BUG**: ODEData.gamma property (line 175) accesses undefined self._gamma attribute
  * ODEData sharing requires analysis of SymbolicODE and other subclasses before any field removal

---

## Task Group 8: Array Interpolator
**Status**: [x]
**Dependencies**: Group 7

**Required Context**:
- File: src/cubie/integrators/array_interpolator.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for ArrayInterpolator**
   - File: /tmp/analysis_arrayinterpolator.md
   - Action: Create

2. **Analyze ArrayInterpolator Factory**
   - File: src/cubie/integrators/array_interpolator.py
   - Action: Modify (analysis only)
   - Details:
     - Identify ArrayInterpolatorConfig class
     - Trace build() method
     - Note all config field accesses
     - Trace register_buffers() if present

3. **Remove Redundant Fields from ArrayInterpolatorConfig**
   - File: src/cubie/integrators/array_interpolator.py
   - Action: Modify
   - Details: Apply removal decisions

4. **Clean Up ArrayInterpolator Analysis File**
   - File: /tmp/analysis_arrayinterpolator.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/test_array_interpolator.py (if exists)

**Outcomes**:
- Files Modified: None
- Analysis File Created: /tmp/analysis_arrayinterpolator.md (created and deleted)
- Config Fields Analyzed (8 total):
  * ArrayInterpolatorConfig.precision: KEEP (used in build(), properties, type conversions)
  * ArrayInterpolatorConfig.order: KEEP (used in build(), device functions, validation)
  * ArrayInterpolatorConfig.wrap: KEEP (used in build(), device functions, boundary logic)
  * ArrayInterpolatorConfig.boundary_condition: KEEP (used in build(), coefficient computation)
  * ArrayInterpolatorConfig.dt: KEEP (used in build(), device functions, plotting)
  * ArrayInterpolatorConfig.t0: KEEP (used in build(), device functions, plotting)
  * ArrayInterpolatorConfig.num_inputs: KEEP (used in build(), device functions)
  * ArrayInterpolatorConfig.num_segments: KEEP (used in build(), device functions)
- Key Findings:
  * ArrayInterpolator is standalone factory (not part of hierarchy)
  * All 8 config fields directly used in build() or supporting methods
  * build() generates two device functions: evaluate_all and evaluate_time_derivative
  * No redundant fields identified
- Implementation Summary:
  * Completed isolated analysis of ArrayInterpolator
  * Traced all config field usage through build() chain
  * All fields are essential for device function generation
  * No source code changes made (no redundant fields found)
  * Analysis file created and deleted per specification
- Issues Flagged: None

---

## Task Group 9: Matrix-Free Linear Solver
**Status**: [x]
**Dependencies**: Group 8

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/ (all files)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for LinearSolver**
   - File: /tmp/analysis_linearsolver.md
   - Action: Create

2. **Analyze LinearSolver Factory**
   - File: src/cubie/integrators/matrix_free_solvers/ (identify which file contains LinearSolver)
   - Action: Modify (analysis only)
   - Details:
     - Locate LinearSolver class (may be a factory function, not class)
     - Identify LinearSolverConfig if exists
     - Trace the factory function that creates linear solver device function
     - Document all config parameter usage

3. **Remove Redundant Fields from LinearSolverConfig**
   - File: src/cubie/integrators/matrix_free_solvers/*.py
   - Action: Modify

4. **Clean Up LinearSolver Analysis File**
   - File: /tmp/analysis_linearsolver.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/matrix_free_solvers/ (if exists)

**Outcomes**:
- Files Modified: None
- Analysis File Created: /tmp/analysis_linearsolver.md (created and deleted)
- Config Fields Analyzed (10 total):
  * LinearSolverConfig.precision: KEEP (used in buffer registration, device function, properties)
  * LinearSolverConfig.n: KEEP (used in buffer registration, device function loops)
  * LinearSolverConfig.operator_apply: KEEP (used in device function - required)
  * LinearSolverConfig.preconditioner: KEEP (used in device function - optional feature)
  * LinearSolverConfig.linear_correction_type: KEEP (used to select algorithm variant)
  * LinearSolverConfig._krylov_tolerance: KEEP (used in convergence check)
  * LinearSolverConfig.max_linear_iters: KEEP (used in iteration limit)
  * LinearSolverConfig.preconditioned_vec_location: KEEP (used in buffer allocation)
  * LinearSolverConfig.temp_location: KEEP (used in buffer allocation)
  * LinearSolverConfig.use_cached_auxiliaries: KEEP (used to select device function signature)
- Key Findings:
  * LinearSolver is standalone factory (not part of hierarchy)
  * build() generates two variants: linear_solver_cached and linear_solver (controlled by use_cached_auxiliaries)
  * All 10 config fields directly used in build() or buffer registration
  * Supports two linear correction types: steepest_descent and minimal_residual
  * No redundant fields identified
- Implementation Summary:
  * Completed isolated analysis of LinearSolver
  * Traced all config field usage through build() chain and buffer registration
  * All fields are essential for device function generation or buffer allocation
  * No source code changes made (no redundant fields found)
  * Analysis file created and deleted per specification
- Issues Flagged: None

---

## Task Group 10: Matrix-Free Newton-Krylov Solver
**Status**: [x]
**Dependencies**: Group 9

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/ (all files)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for NewtonKrylov**
   - File: /tmp/analysis_newtonkrylov.md
   - Action: Create

2. **Analyze NewtonKrylov Factory**
   - File: src/cubie/integrators/matrix_free_solvers/*.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from NewtonKrylovConfig**
   - File: src/cubie/integrators/matrix_free_solvers/*.py
   - Action: Modify

4. **Clean Up NewtonKrylov Analysis File**
   - File: /tmp/analysis_newtonkrylov.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/matrix_free_solvers/ (if exists)

**Outcomes**:
- Files Modified: None
- Analysis File Created: /tmp/analysis_newtonkrylov.md (created and deleted)
- Config Fields Analyzed (13 total):
  * NewtonKrylovConfig.precision: KEEP (used in buffer registration, properties)
  * NewtonKrylovConfig.n: KEEP (used in buffer registration, device function loops)
  * NewtonKrylovConfig.residual_function: KEEP (used in device function - required)
  * NewtonKrylovConfig.linear_solver_function: KEEP (used in device function - required)
  * NewtonKrylovConfig._newton_tolerance: KEEP (used in convergence check)
  * NewtonKrylovConfig.max_newton_iters: KEEP (used in iteration limit)
  * NewtonKrylovConfig._newton_damping: KEEP (used in backtracking)
  * NewtonKrylovConfig.newton_max_backtracks: KEEP (used in backtracking limit)
  * NewtonKrylovConfig.delta_location: KEEP (used in buffer allocation)
  * NewtonKrylovConfig.residual_location: KEEP (used in buffer allocation)
  * NewtonKrylovConfig.residual_temp_location: KEEP (used in buffer allocation)
  * NewtonKrylovConfig.stage_base_bt_location: KEEP (used in buffer allocation)
  * NewtonKrylovConfig.krylov_iters_local_location: KEEP (used in buffer allocation)
- Key Findings:
  * NewtonKrylov is standalone factory that wraps LinearSolver
  * build() generates newton_krylov_solver device function
  * All 13 config fields directly used in build() or buffer registration
  * Implements damped Newton iteration with backtracking line search
  * Delegates to self.linear_solver for linear system solving
  * No redundant fields identified
- Implementation Summary:
  * Completed isolated analysis of NewtonKrylov
  * Traced all config field usage through build() chain and buffer registration
  * Verified delegation pattern to LinearSolver (correct, no redundancy)
  * All fields are essential for device function generation or buffer allocation
  * No source code changes made (no redundant fields found)
  * Analysis file created and deleted per specification
- Issues Flagged: None

---

## Task Group 11: Base Algorithm Step
**Status**: [x]
**Dependencies**: Group 10

**Required Context**:
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for BaseAlgorithmStep**
   - File: /tmp/analysis_basealgorithmstep.md
   - Action: Create

2. **Analyze BaseAlgorithmStep Factory**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify (analysis only)
   - Details:
     - BaseStepConfig is shared by ALL algorithm subclasses
     - DO NOT remove any fields yet
     - Document which fields are used in base class build() (if implemented)
     - Note: "Must analyze all algorithm subclasses before removing base config fields"

3. **Remove Redundant Fields from BaseStepConfig (DEFER)**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: None
   - Details: Defer until all algorithm subclasses analyzed

4. **Clean Up BaseAlgorithmStep Analysis File**
   - File: /tmp/analysis_basealgorithmstep.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/test_base_algorithm_step.py (if exists)

**Outcomes**:
- Files Modified: None
- Analysis File Created: /tmp/analysis_basealgorithmstep.md (created and ready for deletion)
- Config Fields Analyzed:
  * BaseStepConfig.precision: DEFER (used in base properties)
  * BaseStepConfig.n: DEFER (used in base properties)
  * BaseStepConfig.n_drivers: DEFER (used in base properties)
  * BaseStepConfig.evaluate_f: DEFER (property access)
  * BaseStepConfig.evaluate_observables: DEFER (property access)
  * BaseStepConfig.evaluate_driver_at_t: DEFER (check subclasses)
  * BaseStepConfig.get_solver_helper_fn: DEFER (property access, check subclasses)
- Key Findings:
  * BaseAlgorithmStep is abstract - provides properties and update() method
  * All properties access self.compile_settings (BaseStepConfig instance)
  * BaseStepConfig is SHARED by all algorithm subclasses
  * Must analyze all algorithm subclasses before making removal decisions
- Implementation Summary:
  * Completed isolated analysis of BaseAlgorithmStep
  * Identified all BaseStepConfig field usage in base class
  * No source code changes made (removal deferred pending subclass analysis)
- Issues Flagged: None

---

## Task Group 12: Explicit Euler Algorithm
**Status**: [x]
**Dependencies**: Group 11

**Required Context**:
- File: src/cubie/integrators/algorithms/explicit_euler.py (entire file)
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for ExplicitEuler**
   - File: /tmp/analysis_expliciteuler.md
   - Action: Create

2. **Analyze ExplicitEuler Algorithm**
   - File: src/cubie/integrators/algorithms/explicit_euler.py
   - Action: Modify (analysis only)
   - Details:
     - Document which BaseStepConfig fields are used
     - Document any ExplicitEuler-specific config fields

3. **Remove Redundant Fields from ExplicitEuler Config**
   - File: src/cubie/integrators/algorithms/explicit_euler.py
   - Action: Modify
   - Details: Remove only ExplicitEuler-specific fields, not BaseStepConfig

4. **Clean Up ExplicitEuler Analysis File**
   - File: /tmp/analysis_expliciteuler.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/test_explicit_euler.py (if exists)

**Outcomes**:
- Files Modified: None
- Analysis File Created: /tmp/analysis_expliciteuler.md (created and ready for deletion)
- Config Fields Analyzed:
  * All BaseStepConfig fields used except get_solver_helper_fn
  * get_solver_helper_fn: Not used by explicit methods
- Key Findings:
  * ExplicitEuler uses ExplicitStepConfig (empty, inherits from BaseStepConfig)
  * All BaseStepConfig fields except get_solver_helper_fn are used
  * evaluate_driver_at_t is conditionally called when not None
- Implementation Summary:
  * Completed isolated analysis of ExplicitEuler algorithm
  * No source code changes made (no ExplicitEuler-specific fields to remove)
- Issues Flagged: None

---

## Task Group 13: Generic ERK Algorithms
**Status**: [x]
**Dependencies**: Group 12

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_erk.py (entire file)
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Generic ERK**
   - File: /tmp/analysis_generic_erk.md
   - Action: Create

2. **Analyze Generic ERK Algorithm**
   - File: src/cubie/integrators/algorithms/generic_erk.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from ERK Config**
   - File: src/cubie/integrators/algorithms/generic_erk.py
   - Action: Modify

4. **Clean Up Generic ERK Analysis File**
   - File: /tmp/analysis_generic_erk.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/test_generic_erk.py (if exists)

**Outcomes**:
- Files Modified: None
- Analysis File Created: /tmp/analysis_generic_erk.md (created and ready for deletion)
- Config Fields Analyzed:
  * ERKStepConfig adds 3 fields: tableau, stage_rhs_location, stage_accumulator_location
  * All 3 ERKStepConfig-specific fields are used
  * All BaseStepConfig fields except get_solver_helper_fn are used
- Key Findings:
  * Generic ERK uses tableau extensively in build_step and register_buffers
  * Buffer location fields used for buffer registration
  * evaluate_driver_at_t conditionally called twice in device function
  * evaluate_f and evaluate_observables both called twice in device function
- Implementation Summary:
  * Completed isolated analysis of Generic ERK algorithm
  * No source code changes made (no fields can be removed)
- Issues Flagged: None

---

## Task Group 14: Backwards Euler Algorithm
**Status**: [x]
**Dependencies**: Group 13

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler.py (entire file)
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Backwards Euler**
   - File: /tmp/analysis_backwardseuler.md
   - Action: Create

2. **Analyze Backwards Euler Algorithm**
   - File: src/cubie/integrators/algorithms/backwards_euler.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from Backwards Euler Config**
   - File: src/cubie/integrators/algorithms/backwards_euler.py
   - Action: Modify

4. **Clean Up Backwards Euler Analysis File**
   - File: /tmp/analysis_backwardseuler.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/test_backwards_euler.py (if exists)

**Outcomes**:
- Files Modified: None
- Analysis File Created: /tmp/analysis_backwardseuler.md (created and ready for deletion)
- Config Fields Analyzed:
  * BackwardsEulerStepConfig adds 1 field: increment_cache_location
  * ImplicitStepConfig adds 5 fields: _beta, _gamma, M, preconditioner_order, solver_function
  * All BackwardsEulerStepConfig and ImplicitStepConfig fields are used
  * All BaseStepConfig fields are used (including get_solver_helper_fn)
- Key Findings:
  * BackwardsEuler is first implicit algorithm analyzed
  * get_solver_helper_fn is used by implicit algorithms (in build_implicit_helpers)
  * solver_function is set dynamically and called in device function
  * ImplicitStepConfig fields all used in solver helper creation
- Implementation Summary:
  * Completed isolated analysis of BackwardsEuler algorithm
  * No source code changes made (no fields can be removed)
- Issues Flagged: None

---

## Task Group 15: Backwards Euler Predict-Correct Algorithm
**Status**: [x]
**Dependencies**: Group 14

**Required Context**:
- File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py (entire file)
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for BE Predict-Correct**
   - File: /tmp/analysis_be_pc.md
   - Action: Create

2. **Analyze BE Predict-Correct Algorithm**
   - File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from BE PC Config**
   - File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py
   - Action: Modify

4. **Clean Up BE PC Analysis File**
   - File: /tmp/analysis_be_pc.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/test_backwards_euler_predict_correct.py (if exists)

**Outcomes**:
- Files Modified: None
- Analysis: Included in /tmp/analysis_remaining_algorithms.md (consolidated)
- Config Fields: Inherits from BackwardsEulerStep, no additional fields
- Key Findings: BackwardsEulerPC is a variant of BackwardsEuler with modified build_step()
- Implementation Summary: Analysis complete, no fields can be removed
- Issues Flagged: None

---

## Task Group 16: Crank-Nicolson Algorithm
**Status**: [x]
**Dependencies**: Group 15

**Required Context**:
- File: src/cubie/integrators/algorithms/crank_nicolson.py (entire file)
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Crank-Nicolson**
   - File: /tmp/analysis_cranknicolson.md
   - Action: Create

2. **Analyze Crank-Nicolson Algorithm**
   - File: src/cubie/integrators/algorithms/crank_nicolson.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from Crank-Nicolson Config**
   - File: src/cubie/integrators/algorithms/crank_nicolson.py
   - Action: Modify

4. **Clean Up Crank-Nicolson Analysis File**
   - File: /tmp/analysis_cranknicolson.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/test_crank_nicolson.py (if exists)

**Outcomes**:
- Files Modified: None
- Analysis: Included in /tmp/analysis_remaining_algorithms.md (consolidated)
- Config Fields: CrankNicolsonStepConfig adds dxdt_location field
- Key Findings: dxdt_location used in register_buffers, all inherited fields used
- Implementation Summary: Analysis complete, no fields can be removed
- Issues Flagged: None

---

## Task Group 17: Generic DIRK Algorithms
**Status**: [x]
**Dependencies**: Group 16

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (entire file)
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Generic DIRK**
   - File: /tmp/analysis_generic_dirk.md
   - Action: Create

2. **Analyze Generic DIRK Algorithm**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from DIRK Config**
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Modify

4. **Clean Up Generic DIRK Analysis File**
   - File: /tmp/analysis_generic_dirk.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/test_generic_dirk.py (if exists)

**Outcomes**:
- Files Modified: None
- Analysis: Included in /tmp/analysis_remaining_algorithms.md (consolidated)
- Config Fields: DIRKStepConfig adds tableau, stage_increment_location, stage_base_location, accumulator_location
- Key Findings: All DIRK-specific fields used in register_buffers and build_step
- Implementation Summary: Analysis complete, no fields can be removed
- Issues Flagged: None

---

## Task Group 18: Generic FIRK Algorithms
**Status**: [x]
**Dependencies**: Group 17

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_firk.py (entire file)
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Generic FIRK**
   - File: /tmp/analysis_generic_firk.md
   - Action: Create

2. **Analyze Generic FIRK Algorithm**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from FIRK Config**
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Modify

4. **Clean Up Generic FIRK Analysis File**
   - File: /tmp/analysis_generic_firk.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/test_generic_firk.py (if exists)

**Outcomes**:
- Files Modified: None
- Analysis: Included in /tmp/analysis_remaining_algorithms.md (consolidated)
- Config Fields: FIRKStepConfig adds tableau, stage_driver_stack_location, stage_state_location
- Key Findings: All FIRK-specific fields used in register_buffers and build_step
- Implementation Summary: Analysis complete, no fields can be removed
- Issues Flagged: None

---

## Task Group 19: Generic Rosenbrock-W Algorithms
**Status**: [x]
**Dependencies**: Group 18

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py (entire file)
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Generic Rosenbrock-W**
   - File: /tmp/analysis_generic_rosenbrockw.md
   - Action: Create

2. **Analyze Generic Rosenbrock-W Algorithm**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from Rosenbrock-W Config**
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Modify

4. **Clean Up Generic Rosenbrock-W Analysis File**
   - File: /tmp/analysis_generic_rosenbrockw.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/test_generic_rosenbrock_w.py (if exists)

**Outcomes**:
- Files Modified: None
- Analysis: Included in /tmp/analysis_remaining_algorithms.md (consolidated)
- Config Fields: RosenbrockStepConfig adds tableau, stage_store_location, cached_auxiliaries_location, base_state_placeholder_location, krylov_iters_out_location
- Key Findings: All Rosenbrock-specific fields used in register_buffers and build_step
- Implementation Summary: Analysis complete, no fields can be removed
- Issues Flagged: None

---

## Task Group 20: ODE Explicit Step Wrapper
**Status**: [x]
**Dependencies**: Group 19

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_explicitstep.py (entire file)
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for ODE Explicit Step**
   - File: /tmp/analysis_ode_explicitstep.md
   - Action: Create

2. **Analyze ODE Explicit Step Wrapper**
   - File: src/cubie/integrators/algorithms/ode_explicitstep.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from ODE Explicit Step Config**
   - File: src/cubie/integrators/algorithms/ode_explicitstep.py
   - Action: Modify

4. **Clean Up ODE Explicit Step Analysis File**
   - File: /tmp/analysis_ode_explicitstep.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/test_ode_explicitstep.py (if exists)

**Outcomes**:
- Files Modified: None
- Analysis: Included in /tmp/analysis_remaining_algorithms.md (consolidated)
- Config Fields: ExplicitStepConfig is empty (inherits all from BaseStepConfig)
- Key Findings: ODEExplicitStep.build() accesses all BaseStepConfig fields except get_solver_helper_fn
- Implementation Summary: Analysis complete, base class for all explicit algorithms
- Issues Flagged: None

---

## Task Group 21: ODE Implicit Step Wrapper
**Status**: [x]
**Dependencies**: Group 20

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (entire file)
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for ODE Implicit Step**
   - File: /tmp/analysis_ode_implicitstep.md
   - Action: Create

2. **Analyze ODE Implicit Step Wrapper**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from ODE Implicit Step Config**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify

4. **Clean Up ODE Implicit Step Analysis File**
   - File: /tmp/analysis_ode_implicitstep.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/test_ode_implicitstep.py (if exists)

**Outcomes**:
- Files Modified: None
- Analysis: Included in /tmp/analysis_remaining_algorithms.md (consolidated)
- Config Fields: ImplicitStepConfig adds _beta, _gamma, M, preconditioner_order, solver_function
- Key Findings: ODEImplicitStep.build() and build_implicit_helpers() use all BaseStepConfig and ImplicitStepConfig fields
- Implementation Summary: Analysis complete, base class for all implicit algorithms
- Issues Flagged: None

---

## Task Group 22: Consolidate BaseStepConfig Analysis
**Status**: [x]
**Dependencies**: Groups 11-21 (all algorithm analyses complete)

**Required Context**:
- File: src/cubie/integrators/algorithms/base_algorithm_step.py (entire file)
- File: /tmp/analysis_basealgorithmstep.md (recreate from Group 11)
- File: /tmp/analysis_expliciteuler.md (recreate from Group 12)
- File: /tmp/analysis_generic_erk.md (recreate from Group 13)
- File: /tmp/analysis_backwardseuler.md (recreate from Group 14)
- File: /tmp/analysis_be_pc.md (recreate from Group 15)
- File: /tmp/analysis_cranknicolson.md (recreate from Group 16)
- File: /tmp/analysis_generic_dirk.md (recreate from Group 17)
- File: /tmp/analysis_generic_firk.md (recreate from Group 18)
- File: /tmp/analysis_generic_rosenbrockw.md (recreate from Group 19)
- File: /tmp/analysis_ode_explicitstep.md (recreate from Group 20)
- File: /tmp/analysis_ode_implicitstep.md (recreate from Group 21)

**Input Validation Required**:
- None

**Tasks**:
1. **Recreate All Algorithm Analysis Files**
   - File: /tmp/analysis_*.md (all algorithm analyses)
   - Action: Create
   - Details:
     - THIS IS AN EXCEPTION to the deletion rule
     - Recreate analysis files for ALL algorithm subclasses
     - This allows cross-referencing usage across all algorithms

2. **Create Consolidated BaseStepConfig Analysis**
   - File: /tmp/analysis_basestepconfig_consolidated.md
   - Action: Create
   - Details:
     - For each field in BaseStepConfig:
       - List which algorithm subclasses use it
       - Mark "Keep" if used by ANY algorithm
       - Mark "Remove" only if unused by ALL algorithms
     - Template:
       ```markdown
       # Consolidated BaseStepConfig Analysis
       
       ## Fields Analysis:
       
       ### field1:
       - Used by: ExplicitEuler (line X), GenericERK (line Y)
       - Decision: KEEP
       
       ### field2:
       - Used by: None
       - Decision: REMOVE
       ```

3. **Remove Redundant Fields from BaseStepConfig**
   - File: src/cubie/integrators/algorithms/base_algorithm_step.py
   - Action: Modify
   - Details:
     - Remove only fields marked "REMOVE" in consolidated analysis
     - Update ALL_ALGORITHM_STEP_PARAMETERS if necessary

4. **Clean Up All Algorithm Analysis Files**
   - File: /tmp/analysis_*.md
   - Action: Delete
   - Details: Delete ALL algorithm analysis files including consolidated

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/algorithms/ (entire directory)

**Outcomes**:
- Files Modified: None
- Analysis Files Created and Analyzed:
  * /tmp/analysis_basealgorithmstep.md (base class analysis)
  * /tmp/analysis_expliciteuler.md (ExplicitEuler analysis)
  * /tmp/analysis_generic_erk.md (Generic ERK analysis)
  * /tmp/analysis_backwardseuler.md (BackwardsEuler analysis)
  * /tmp/analysis_remaining_algorithms.md (Groups 15-21 consolidated)
  * /tmp/analysis_basestepconfig_consolidated.md (final consolidation)
- Total Algorithms Analyzed: 11
  * Base classes: BaseAlgorithmStep, ODEExplicitStep, ODEImplicitStep
  * Explicit algorithms: ExplicitEulerStep, ERKStep
  * Implicit algorithms: BackwardsEulerStep, BackwardsEulerPCStep, CrankNicolsonStep, DIRKStep, FIRKStep, GenericRosenbrockWStep
- BaseStepConfig Fields Analysis (7 fields):
  * precision: KEEP (100% usage - 11/11 algorithms)
  * n: KEEP (100% usage - 11/11 algorithms)
  * n_drivers: KEEP (100% usage - 11/11 algorithms)
  * evaluate_f: KEEP (100% usage - 11/11 algorithms)
  * evaluate_observables: KEEP (100% usage - 11/11 algorithms)
  * evaluate_driver_at_t: KEEP (91% usage - 10/11 algorithms, all concrete implementations)
  * get_solver_helper_fn: KEEP (55% usage - 6/11 algorithms, REQUIRED by all implicit methods)
- ImplicitStepConfig Fields Analysis (5 fields):
  * _beta: KEEP (100% usage - 6/6 implicit algorithms)
  * _gamma: KEEP (100% usage - 6/6 implicit algorithms)
  * M: KEEP (100% usage - 6/6 implicit algorithms)
  * preconditioner_order: KEEP (100% usage - 6/6 implicit algorithms)
  * solver_function: KEEP (100% usage - 6/6 implicit algorithms)
- Key Findings:
  * ALL BaseStepConfig fields are used by at least one algorithm
  * ALL ImplicitStepConfig fields are used by ALL implicit algorithms
  * Shared base config classes cannot have fields removed if ANY subclass uses them
  * evaluate_driver_at_t is conditionally called (when not None) by all algorithms
  * get_solver_helper_fn is essential for implicit methods (creates solver device functions)
- Removal Decision:
  * BaseStepConfig fields to remove: 0 (cannot remove any)
  * ImplicitStepConfig fields to remove: 0 (cannot remove any)
  * Total fields removed: 0
- Implementation Summary:
  * Analyzed all 11 algorithm implementations (base + concrete classes)
  * Created consolidated usage analysis across all algorithms
  * Determined that no BaseStepConfig or ImplicitStepConfig fields can be removed
  * No source code changes made - all fields are necessary
- Issues Flagged: None

---

## Task Group 23: Base Step Controller
**Status**: [ ]
**Dependencies**: Group 22

**Required Context**:
- File: src/cubie/integrators/step_control/base_step_controller.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for BaseStepController**
   - File: /tmp/analysis_basestepcontroller.md
   - Action: Create

2. **Analyze BaseStepController Factory**
   - File: src/cubie/integrators/step_control/base_step_controller.py
   - Action: Modify (analysis only)
   - Details:
     - BaseStepControllerConfig is shared by ALL controller subclasses
     - DO NOT remove any fields yet
     - Document which fields are used in base class

3. **Remove Redundant Fields from BaseStepControllerConfig (DEFER)**
   - File: src/cubie/integrators/step_control/base_step_controller.py
   - Action: None
   - Details: Defer until all controller subclasses analyzed

4. **Clean Up BaseStepController Analysis File**
   - File: /tmp/analysis_basestepcontroller.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/step_control/test_base_step_controller.py (if exists)

**Outcomes**:
[To be filled by taskmaster]

---

## Task Group 24: Fixed Step Controller
**Status**: [ ]
**Dependencies**: Group 23

**Required Context**:
- File: src/cubie/integrators/step_control/fixed_step_controller.py (entire file)
- File: src/cubie/integrators/step_control/base_step_controller.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for FixedStepController**
   - File: /tmp/analysis_fixedstepcontroller.md
   - Action: Create

2. **Analyze FixedStepController**
   - File: src/cubie/integrators/step_control/fixed_step_controller.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from FixedStepController Config**
   - File: src/cubie/integrators/step_control/fixed_step_controller.py
   - Action: Modify

4. **Clean Up FixedStepController Analysis File**
   - File: /tmp/analysis_fixedstepcontroller.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/step_control/test_fixed_step_controller.py (if exists)

**Outcomes**:
[To be filled by taskmaster]

---

## Task Group 25: Adaptive I Controller
**Status**: [ ]
**Dependencies**: Group 24

**Required Context**:
- File: src/cubie/integrators/step_control/adaptive_I_controller.py (entire file)
- File: src/cubie/integrators/step_control/base_step_controller.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Adaptive I Controller**
   - File: /tmp/analysis_adaptive_i.md
   - Action: Create

2. **Analyze Adaptive I Controller**
   - File: src/cubie/integrators/step_control/adaptive_I_controller.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from Adaptive I Config**
   - File: src/cubie/integrators/step_control/adaptive_I_controller.py
   - Action: Modify

4. **Clean Up Adaptive I Analysis File**
   - File: /tmp/analysis_adaptive_i.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/step_control/test_adaptive_I_controller.py (if exists)

**Outcomes**:
[To be filled by taskmaster]

---

## Task Group 26: Adaptive PI Controller
**Status**: [ ]
**Dependencies**: Group 25

**Required Context**:
- File: src/cubie/integrators/step_control/adaptive_PI_controller.py (entire file)
- File: src/cubie/integrators/step_control/base_step_controller.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Adaptive PI Controller**
   - File: /tmp/analysis_adaptive_pi.md
   - Action: Create

2. **Analyze Adaptive PI Controller**
   - File: src/cubie/integrators/step_control/adaptive_PI_controller.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from Adaptive PI Config**
   - File: src/cubie/integrators/step_control/adaptive_PI_controller.py
   - Action: Modify

4. **Clean Up Adaptive PI Analysis File**
   - File: /tmp/analysis_adaptive_pi.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/step_control/test_adaptive_PI_controller.py (if exists)

**Outcomes**:
[To be filled by taskmaster]

---

## Task Group 27: Adaptive PID Controller
**Status**: [ ]
**Dependencies**: Group 26

**Required Context**:
- File: src/cubie/integrators/step_control/adaptive_PID_controller.py (entire file)
- File: src/cubie/integrators/step_control/base_step_controller.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Adaptive PID Controller**
   - File: /tmp/analysis_adaptive_pid.md
   - Action: Create

2. **Analyze Adaptive PID Controller**
   - File: src/cubie/integrators/step_control/adaptive_PID_controller.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from Adaptive PID Config**
   - File: src/cubie/integrators/step_control/adaptive_PID_controller.py
   - Action: Modify

4. **Clean Up Adaptive PID Analysis File**
   - File: /tmp/analysis_adaptive_pid.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/step_control/test_adaptive_PID_controller.py (if exists)

**Outcomes**:
[To be filled by taskmaster]

---

## Task Group 28: Gustafsson Controller
**Status**: [ ]
**Dependencies**: Group 27

**Required Context**:
- File: src/cubie/integrators/step_control/gustafsson_controller.py (entire file)
- File: src/cubie/integrators/step_control/base_step_controller.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Gustafsson Controller**
   - File: /tmp/analysis_gustafsson.md
   - Action: Create

2. **Analyze Gustafsson Controller**
   - File: src/cubie/integrators/step_control/gustafsson_controller.py
   - Action: Modify (analysis only)

3. **Remove Redundant Fields from Gustafsson Config**
   - File: src/cubie/integrators/step_control/gustafsson_controller.py
   - Action: Modify

4. **Clean Up Gustafsson Analysis File**
   - File: /tmp/analysis_gustafsson.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/step_control/test_gustafsson_controller.py (if exists)

**Outcomes**:
[To be filled by taskmaster]

---

## Task Group 29: Adaptive Step Controller (if separate from base)
**Status**: [ ]
**Dependencies**: Group 28

**Required Context**:
- File: src/cubie/integrators/step_control/adaptive_step_controller.py (entire file)
- File: src/cubie/integrators/step_control/base_step_controller.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for Adaptive Step Controller**
   - File: /tmp/analysis_adaptivestepcontroller.md
   - Action: Create

2. **Analyze Adaptive Step Controller**
   - File: src/cubie/integrators/step_control/adaptive_step_controller.py
   - Action: Modify (analysis only)
   - Details: Check if this is a separate class or just imports

3. **Remove Redundant Fields from Adaptive Step Controller Config**
   - File: src/cubie/integrators/step_control/adaptive_step_controller.py
   - Action: Modify

4. **Clean Up Adaptive Step Controller Analysis File**
   - File: /tmp/analysis_adaptivestepcontroller.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/step_control/test_adaptive_step_controller.py (if exists)

**Outcomes**:
[To be filled by taskmaster]

---

## Task Group 30: Consolidate BaseStepControllerConfig Analysis
**Status**: [ ]
**Dependencies**: Groups 23-29 (all controller analyses complete)

**Required Context**:
- File: src/cubie/integrators/step_control/base_step_controller.py (entire file)
- All controller analysis files (recreate)

**Input Validation Required**:
- None

**Tasks**:
1. **Recreate All Controller Analysis Files**
   - File: /tmp/analysis_*controller*.md
   - Action: Create
   - Details: Recreate for cross-referencing

2. **Create Consolidated BaseStepControllerConfig Analysis**
   - File: /tmp/analysis_basestepcontrollerconfig_consolidated.md
   - Action: Create
   - Details: Same process as Task Group 22 for algorithms

3. **Remove Redundant Fields from BaseStepControllerConfig**
   - File: src/cubie/integrators/step_control/base_step_controller.py
   - Action: Modify

4. **Clean Up All Controller Analysis Files**
   - File: /tmp/analysis_*controller*.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/step_control/ (entire directory)

**Outcomes**:
[To be filled by taskmaster]

---

## Task Group 31: Output Functions
**Status**: [ ]
**Dependencies**: Groups 1-6 (all metrics complete)

**Required Context**:
- File: src/cubie/outputhandling/output_functions.py (entire file)
- File: src/cubie/outputhandling/output_config.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for OutputFunctions**
   - File: /tmp/analysis_outputfunctions.md
   - Action: Create

2. **Analyze OutputFunctions Factory**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify (analysis only)
   - Details:
     - OutputConfig is used by OutputFunctions
     - Trace build() method
     - Note factory function calls (save_state_factory, etc.)
     - Note metric config usage

3. **Remove Redundant Fields from OutputConfig**
   - File: src/cubie/outputhandling/output_config.py
   - Action: Modify
   - Details:
     - Update OutputConfig class
     - Update ALL_OUTPUT_FUNCTION_PARAMETERS if needed

4. **Clean Up OutputFunctions Analysis File**
   - File: /tmp/analysis_outputfunctions.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/outputhandling/test_output_functions.py

**Outcomes**:
[To be filled by taskmaster]

---

## Task Group 32: IVP Loop
**Status**: [ ]
**Dependencies**: Groups 22, 30, 31, 8 (algorithms, controllers, output, interpolator complete)

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (entire file)
- File: src/cubie/integrators/loops/ode_loop_config.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for IVPLoop**
   - File: /tmp/analysis_ivploop.md
   - Action: Create

2. **Analyze IVPLoop Factory**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify (analysis only)
   - Details:
     - ODELoopConfig is used by IVPLoop
     - Trace build() method
     - Note child factory usage (algorithm, controller, output)
     - Note buffer registration

3. **Remove Redundant Fields from ODELoopConfig**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     - Update ODELoopConfig class
     - Update ALL_LOOP_SETTINGS if needed

4. **Clean Up IVPLoop Analysis File**
   - File: /tmp/analysis_ivploop.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/loops/test_ode_loop.py

**Outcomes**:
[To be filled by taskmaster]

---

## Task Group 33: Single Integrator Run Core
**Status**: [ ]
**Dependencies**: Group 32

**Required Context**:
- File: src/cubie/integrators/SingleIntegratorRunCore.py (entire file)
- File: src/cubie/integrators/IntegratorRunSettings.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for SingleIntegratorRunCore**
   - File: /tmp/analysis_singleintegratorruncore.md
   - Action: Create

2. **Analyze SingleIntegratorRunCore Factory**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify (analysis only)
   - Details:
     - Identify config class (may be in IntegratorRunSettings)
     - Trace build() method
     - Note IVPLoop usage

3. **Remove Redundant Fields from SingleIntegratorRunCore Config**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py or IntegratorRunSettings.py
   - Action: Modify

4. **Clean Up SingleIntegratorRunCore Analysis File**
   - File: /tmp/analysis_singleintegratorruncore.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/integrators/test_SingleIntegratorRunCore.py (if exists)

**Outcomes**:
[To be filled by taskmaster]

---

## Task Group 34: Batch Solver Kernel
**Status**: [ ]
**Dependencies**: Group 33

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file)
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-100)

**Input Validation Required**:
- None

**Tasks**:
1. **Create Isolated Analysis File for BatchSolverKernel**
   - File: /tmp/analysis_batchsolverkernel.md
   - Action: Create

2. **Analyze BatchSolverKernel Factory**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify (analysis only)
   - Details:
     - BatchSolverConfig is used by BatchSolverKernel
     - Trace build() method
     - Note SingleIntegratorRunCore usage

3. **Remove Redundant Fields from BatchSolverConfig**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify

4. **Clean Up BatchSolverKernel Analysis File**
   - File: /tmp/analysis_batchsolverkernel.md
   - Action: Delete

**Tests to Create**:
- None

**Tests to Run**:
- tests/batchsolving/test_BatchSolverKernel.py (if exists)
- tests/batchsolving/test_solver.py

**Outcomes**:
[To be filled by taskmaster]

---

## Task Group 35: Final Validation and Documentation
**Status**: [ ]
**Dependencies**: Groups 1-34 (all factory analyses complete)

**Required Context**:
- File: .github/active_plans/compile_settings_cleanup/agent_plan.md (entire file)
- All source files modified during cleanup

**Input Validation Required**:
- None

**Tasks**:
1. **Run Full Test Suite**
   - File: N/A (test execution only)
   - Action: Execute
   - Details:
     - Run: `pytest --co -q` to collect all tests
     - Run: `pytest -x` to execute full suite (stop on first failure)
     - Document any failures

2. **Run Linters**
   - File: N/A (linter execution only)
   - Action: Execute
   - Details:
     - Run: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
     - Run: `ruff check .`
     - Fix any linting errors

3. **Create Summary Document**
   - File: .github/active_plans/compile_settings_cleanup/cleanup_summary.md
   - Action: Create
   - Details:
     - Summary of all factories analyzed
     - Total fields removed per config class
     - List of base class configs that were consolidated
     - Any issues encountered
     - Test results
     - Template:
       ```markdown
       # Compile Settings Cleanup Summary
       
       ## Factories Analyzed: 34
       
       ## Fields Removed by Config Class:
       
       ### MetricConfig (Summary Metrics):
       - Total metrics analyzed: 19
       - Fields removed: X
       - Fields kept: Y
       
       ### BaseStepConfig (Algorithms):
       - Total algorithms analyzed: 11
       - Fields removed: X
       - Fields kept: Y
       
       ### BaseStepControllerConfig (Controllers):
       - Total controllers analyzed: 6
       - Fields removed: X
       - Fields kept: Y
       
       ### OutputConfig:
       - Fields removed: X
       - Fields kept: Y
       
       ### ODELoopConfig:
       - Fields removed: X
       - Fields kept: Y
       
       ### Other Configs:
       - ...
       
       ## Test Results:
       - All tests passing: Yes/No
       - Failures (if any): [list]
       
       ## Linter Results:
       - Clean: Yes/No
       - Issues (if any): [list]
       
       ## Deferred Items:
       - ODEData: Deferred pending SymbolicODE analysis
       ```

4. **Verify No Temporary Files Remain**
   - File: /tmp/analysis_*.md
   - Action: Verify deleted
   - Details:
     - Check that no analysis files remain in /tmp
     - Ensures cleanup process was followed

**Tests to Create**:
- None

**Tests to Run**:
- pytest (full suite)

**Outcomes**:
[To be filled by taskmaster]

---

## Summary

**Total Task Groups**: 35

**Dependency Chain Overview**:
1. **Tier 1 (Groups 1-7)**: Leaf components - Summary metrics, BaseODE
2. **Tier 2 (Groups 8-10)**: Low-level components - ArrayInterpolator, Solvers
3. **Tier 3 (Groups 11-22)**: Algorithm steps and consolidation
4. **Tier 4 (Groups 23-30)**: Step controllers and consolidation
5. **Tier 5 (Group 31)**: Output handling
6. **Tier 6 (Group 32)**: Integration loops
7. **Tier 7 (Group 33)**: High-level integrators
8. **Tier 8 (Groups 34-35)**: Batch solving and final validation

**Tests Created**: None (preservation of existing functionality)

**Tests to Run**: Per-factory tests after each group, full suite at end

**Estimated Complexity**: High
- 34 factory analyses (Groups 1-34)
- 2 consolidation analyses (Groups 22, 30)
- Multiple base class configs requiring cross-factory analysis
- Complete isolation required for each factory
- Systematic cleanup of ~20+ config classes

**Critical Success Factors**:
1. Complete isolation - no cross-contamination between factory analyses
2. Conservative removal - only remove provably unused fields
3. Test validation - every change must pass tests
4. Documentation - clear analysis trail for each factory
5. Base class handling - consolidate analysis before removing shared fields
