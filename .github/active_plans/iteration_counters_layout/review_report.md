# Implementation Review Report
# Feature: iteration_counters_layout
# Review Date: 2025-12-17
# Reviewer: Harsh Critic Agent

## Executive Summary
Iteration counter buffers now allocate with a time-major shape and are passed
through the kernel in that order, which matches the intended natural layout.
However, the implementation does not yet prove the new ordering to users or
tests, and short runs can still be reported as if counters are inactive. Common
result formats also drop the counters entirely, so consumers cannot rely on the
new layout.

## User Story Validation
**User Stories** (from human_overview.md):
- Iteration counters stored and indexed in time, counter, run order: Partial –
  device/kernel plumbing follows that order, but no test asserts it and helper
  utilities still omit the data.

**Acceptance Criteria Assessment**: Missing explicit verification of the new
axis order; callers using convenience result paths cannot observe counters, and
active flagging fails for single-save runs.

## Goal Alignment
**Original Goals** (from human_overview.md):
- Store iteration counters in natural (time, counter, run) order: Partial –
  array shapes follow the order, but exposure and validation are incomplete.
- Remove evidence of the old ordering: Missing – no tests guard against a
  regression and helper outputs still hide counters.

**Assessment**: The structural change is in place, but lack of assertions and
consumer visibility leaves the goal unverified and easy to regress.

## Code Quality Analysis

### Strengths
- `BatchOutputSizes.from_solver` defines iteration counters as `(n_saves, 4,
  n_runs)` with stride `("time", "variable", "run")`, aligning the buffers with
  the natural layout (src/cubie/outputhandling/output_sizes.py:370-385).
- Kernel slices counters as `[:, :, run_index]`, matching the time, counter,
  run expectation on device (src/cubie/batchsolving/BatchSolverKernel.py:592-614).

### Areas of Concern

#### Unnecessary Additions
- **Location**: tests/_utils.py, lines 893-909  
  **Issue**: `assert_integration_outputs` prints counters but never checks
  shape or ordering. The user story hinges on axis order, yet no test asserts
  it, leaving old ordering undetected.  
  **Impact**: Acceptance criterion is unverified; regressions will slip
  through.

#### Unnecessary Complexity
- **Location**: src/cubie/batchsolving/arrays/BatchOutputArrays.py,
  lines 167-190  
  **Issue**: Iteration counters are considered active only when `size > 4`,
  which flags single-save runs (shape `(1, 4, 1)`) as inactive even when
  counters are enabled. The heuristic couples activation to element count
  instead of configuration.  
  **Impact**: ActiveOutputs reports counters as disabled for short runs, so
  downstream consumers cannot rely on the flag when interpreting layout.

#### Convention Violations
- **PEP8 / API completeness**: SolveResult convenience outputs omit iteration
  counters entirely (src/cubie/batchsolving/solveresult.py:416-451), so users
  requesting `results_type="numpy"` or `"numpy_per_summary"` cannot access the
  counters or their axis order. This undermines the stated requirement and
  silently discards data.

## Performance Analysis
- No performance regressions identified; concerns above are correctness and
  visibility rather than throughput.

## Architecture Assessment
- Kernel-side ordering matches the intended layout, but the absence of tests
  and missing exposure in result helpers leaves the feature partially
  integrated and hard to validate.

## Suggested Edits

### High Priority (Correctness/Critical)
1. **Add explicit assertions for iteration counter ordering**
   - Task Group: iteration_counters_layout
   - File: tests/_utils.py (or dedicated iteration counter test)
   - Issue: Counters are never checked for `(time, counter, run)` ordering or
     value consistency.
   - Fix: Assert `solver.iteration_counters.shape == (n_saves, 4, n_runs)` and
     validate per-axis indexing in integration and solver kernel tests.
   - Rationale: Directly verifies the acceptance criterion and prevents
     reintroducing the old layout.

### Medium Priority (Quality/Simplification)
2. **Activate counters based on configuration, not element count**
   - Task Group: iteration_counters_layout
   - File: src/cubie/batchsolving/arrays/BatchOutputArrays.py
   - Issue: `size > 4` marks single-save runs as inactive even when counters
     are enabled.
   - Fix: Tie activation to the configured save_counters flag or to array
     dimensionality (non-None with second axis length 4) instead of element
     count.
   - Rationale: Accurate ActiveOutputs is needed for consumers to trust counter
     presence and ordering.

### Medium Priority (Quality/Simplification)
3. **Surface counters in convenience outputs with documented axis order**
   - Task Group: iteration_counters_layout
   - File: src/cubie/batchsolving/solveresult.py
   - Issue: `as_numpy` and `as_numpy_per_summary` drop iteration counters and
     give no axis metadata.
   - Fix: Include `iteration_counters` (with axis order noted as
     time, counter, run) in these paths and mirror the stride info so users
     can consume the new layout outside the full SolveResult.
   - Rationale: Makes the new layout observable and prevents silent data loss.

## Recommendations
- **Immediate Actions**: Add axis-order assertions in the integration and
  solver kernel tests; adjust ActiveOutputs detection; expose counters in the
  convenience result formats.
- **Future Refactoring**: Document counter axis order alongside legends so
  consumers do not infer layout implicitly.
- **Testing Additions**: Add regression tests that fail if counters are
  permuted or omitted in any result type.
- **Documentation Needs**: Briefly note counter axis order in user-facing
  result docs once code exposure is fixed.

## Overall Rating
**Implementation Quality**: Fair  
**User Story Achievement**: Partial  
**Goal Achievement**: Partial  
**Recommended Action**: Revise

## Task Outcomes
- [x] Added explicit iteration counter ordering assertions in tests.
- [x] Activated counter flags based on configuration rather than element
      counts.
- [x] Exposed iteration counters and stride labels in convenience numpy
      outputs.
