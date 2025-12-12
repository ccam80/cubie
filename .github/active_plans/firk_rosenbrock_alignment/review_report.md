# Implementation Review Report
# Feature: firk_rosenbrock_alignment
# Review Date: 2025-12-12
# Reviewer: Harsh Critic Agent

## Executive Summary
The FIRK path now flattens stage buffers and mirrors much of the generic
solver wiring, but key control-path defaults are still taken from the
global script rather than the tableau capabilities. Adaptive PID remains
selectable even when the tableau lacks an error estimate, so the FIRK
branch can still be configured into invalid states that production code
guards against.

The Rosenbrock branch diverges more sharply from `generic_rosenbrock_w`.
Cached Jacobian helpers are reduced to a hard-coded single-slot buffer,
the linear solver wrapper drops cached-operator integration, and driver
derivative/time-derivative handling does not match the module staging.
Placeholders extend beyond the allowed system-helper codegen stubs,
leaving buffer sizing and solver signatures out of parity.

## User Story Validation
**User Stories** (from human_overview.md):
- FIRK parity in tests/all_in_one.py: **Partial** - Stage-flattened
  buffers and Kahan accumulation are present, but controller defaults are
  not derived from tableau error capability (controller_type stays
  script-driven even when tableau has no error estimate).
- Rosenbrock parity in tests/all_in_one.py: **Not Met** - Cached
  auxiliary sizing is hard-coded to one, cached Jacobian helpers and
  linear solver wiring omit the cached-operator path, and driver/time
  derivative hooks deviate from module staging; placeholders exceed the
  permitted system-helper codegen stubs.

**Acceptance Criteria Assessment**: FIRK accepts any tableau and uses
flattened buffers, but fails the controller-default requirement.
Rosenbrock does not mirror cached helper sizing or solver signatures and
keeps broader placeholders than allowed, so acceptance criteria are not
met.

## Goal Alignment
**Original Goals** (from human_overview.md):
- FIRK factory parity with `generic_firk`: **Partial** - Buffer layout is
  aligned; controller selection and defaults still diverge.
- Rosenbrock factory parity with `generic_rosenbrock_w`: **Missing** -
  Cached auxiliaries, cached operator usage, and helper integration are
  not aligned; placeholders persist beyond system helper codegen.
- Loop/controller/output parity across algorithms: **Partial** - Shared
  sizing includes new buffers, but controller forcing for errorless
  tableaus is absent, allowing inconsistent loop configuration.

**Assessment**: The implementation improves buffer parity but leaves
control-path and helper integration mismatches that keep the debug
kernel from mirroring production flows.

## Code Quality Analysis

### Strengths
- FIRK step uses flattened `stage_count * n` buffers with stage driver
  stack prefill and Kahan accumulation for solution/error
  (tests/all_in_one.py, lines 1853-2140).
- Loop counter handling now pulls iterations from the step counters
  buffer, aligning with implicit solver outputs (tests/all_in_one.py,
  lines 3579-3635).

### Areas of Concern

#### Unnecessary Additions / Missing Parity
- **Location**: tests/all_in_one.py, lines 2994-3085  
  **Issue**: Rosenbrock build hard-codes `cached_auxiliary_count = 1`,
  uses a linear solver wrapper that ignores cached auxiliaries/shared
  scratch, and retains placeholder helpers beyond the allowed
  system-codegen stubs. Cached operator preparation is not integrated.  
  **Impact**: Buffer sizing and solver signatures diverge from
  `generic_rosenbrock_w`; debug runs do not exercise cached-Jacobian
  flow, breaking the parity goal.

#### Unnecessary Complexity / Incorrect Defaults
- **Location**: tests/all_in_one.py, lines 3090-3113  
  **Issue**: Controller selection remains purely script-driven. PID is
  allowed even when the selected FIRK/Rosenbrock tableau lacks an error
  estimate; module defaults would force fixed-step in that case.  
  **Impact**: Debug kernel can be configured into unsupported adaptive
  modes, diverging from production controller defaults and violating
  acceptance criteria.

#### Incorrect Behaviour
- **Location**: tests/all_in_one.py, lines 2358-2418 and 2489-2515  
  **Issue**: Rosenbrock stage 0 and final-stage derivative handling omit
  driver/time-derivative staging parity. Stage 0 never refreshes driver
  values at `c0`, and the driver derivative for the last stage is taken
  at `current_time` rather than the stage time. Initial stage increment
  guess copies uninitialised `time_derivative` buffer before it is
  populated.  
  **Impact**: Driver-dependent systems and stage timing differ from
  module logic, reducing fidelity of debug runs and risking incorrect
  cached-derivative contributions.

### Convention Violations
- Controller defaults not conditioned on tableau error availability
  (tests/all_in_one.py, lines 3090-3113) violate repository guidance to
  mirror module defaults for implicit schemes.

## Performance Analysis
- Cached Jacobian path is effectively disabled (cached auxiliaries fixed
  to length one, cached operator unused), so memory allocations do not
  reflect real helper outputs and cached data is never reused.
- Stage driver/time-derivative staging mismatches can increase solver
  iterations by feeding inconsistent RHS/derivative estimates.

## Architecture Assessment
- FIRK buffer layout integrates cleanly with the loop, but controller
  defaults and acceptance flow can diverge from production due to
  script-level overrides.
- Rosenbrock integration omits cached-operator wiring, leaving the debug
  kernel architecturally different from `generic_rosenbrock_w` and
  reducing maintainability when helpers change.

## Suggested Edits

### High Priority (Correctness/Critical)
1. **Enforce controller defaults per tableau**  
   - Task Group: 4  
   - File: tests/all_in_one.py (lines 3090-3113)  
   - Issue: Controller choice ignores `tableau.has_error_estimate`; PID
     remains selectable for errorless FIRK/Rosenbrock tableaus.  
   - Fix: Derive controller_type from tableau: force fixed-step when no
     error estimate, otherwise allow PID defaults mirroring module
     settings before dt0/fixed_mode are computed.  
   - Rationale: Matches module safeguards and acceptance criteria for
     parity and prevents invalid adaptive configurations.

2. **Restore cached Jacobian helper parity**  
   - Task Group: 3  
   - File: tests/all_in_one.py (lines 2994-3085, 2234-2436)  
   - Issue: `cached_auxiliary_count` is hard-coded to one and the linear
     solver wrapper ignores cached auxiliaries/shared scratch, leaving
     placeholders beyond system-helper codegen.  
   - Fix: Size cached auxiliaries from helper metadata, pass cached
     auxiliaries/shared pointer through the linear solver, and align the
     wrapper signature/usage with `linear_solver_cached_factory`
     semantics. Keep placeholders only for Jacobian/time-derivative
     bodies.  
   - Rationale: Required for buffer/layout parity with
     `generic_rosenbrock_w`.

### Medium Priority (Quality/Simplification)
3. **Align driver/time-derivative staging**  
   - Task Group: 3  
   - File: tests/all_in_one.py (lines 2358-2515)  
   - Issue: Stage 0 never refreshes drivers at `c0`; driver derivatives
     for the last stage are taken at `current_time` instead of stage
     time; initial stage increment copies an uninitialised buffer.  
   - Fix: Call `driver_function` for stage 0, evaluate driver derivatives
     at the stage time for the final stage, and initialise
     `time_derivative` before copying into the stage increment guess.  
   - Rationale: Keeps stage assembly consistent with module timing and
     avoids garbage initial guesses.

## Recommendations
- **Immediate Actions**: Enforce controller defaults based on tableau
  error capability; rework Rosenbrock cached helper sizing/wiring to
  match `linear_solver_cached_factory`; fix driver/time-derivative
  staging.
- **Future Refactoring**: Consider sourcing cached helper metadata from a
  small shim mirroring solver_helpers outputs to keep debug script
  aligned when helper interfaces evolve.
- **Testing Additions**: Add focused CUDASIM assertions that the debug
  build selects fixed controller when tableau lacks error estimate and
  that Rosenbrock cached_auxiliary_count matches helper metadata.
- **Documentation Needs**: Update in-file comments to clarify which
  placeholders remain intentionally limited to system-helper codegen.

## Overall Rating
**Implementation Quality**: Fair  
**User Story Achievement**: FIRK Partial, Rosenbrock Not Met  
**Goal Achievement**: ~50%  
**Recommended Action**: Revise
