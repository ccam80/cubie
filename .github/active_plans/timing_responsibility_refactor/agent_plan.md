# Agent Plan: Timing Responsibility Refactor

## Overview

This plan refactors timing parameter responsibility to SingleIntegratorRun. The goal is to create a single source of truth for timing configuration while keeping ODELoopConfig as a passive data container.

---

## Component: SingleIntegratorRun / SingleIntegratorRunCore

### Expected Behavior Changes

**New Properties on SingleIntegratorRun:**

1. **`any_time_domain_outputs`** (property, bool)
   - Returns True if time-domain outputs are requested
   - Logic: `(save_every is not None or save_last) AND any of ["state", "observables", "time"] in output_functions.output_types`
   - Used by BatchSolverKernel to determine if time-domain output arrays are needed

2. **`any_summary_outputs`** (property, bool)
   - Returns True if summary outputs are requested
   - Logic: `(summarise_every is not None or summarise_last) AND any summary metric in output_functions.output_types`

3. **`save_every`** (property, Optional[float])
   - Returns the consolidated save interval or None if save_last mode

4. **`summarise_every`** (property, Optional[float])
   - Returns the consolidated summary interval or None if summarise_last mode

5. **`sample_summaries_every`** (property, Optional[float])
   - Returns the summary sampling interval
   - May be auto-calculated from chunk_duration when summarise_last=True

6. **`save_last`** (property, bool)
   - True if save_every is None AND time-domain output_types are requested

7. **`summarise_last`** (property, bool)
   - True if summarise_every is None AND summary output_types are requested

**New Methods on SingleIntegratorRun:**

1. **`output_length(duration: float) -> int`**
   - Calculates number of time-domain output samples
   - Formula: `1 + floor(duration / save_every) + (1 if save_last else 0)`
   - When save_every is None (pure save_last): returns 2 (initial + final)
   - Called by BatchSolverKernel with chunk_duration

2. **`summaries_length(duration: float) -> int`**
   - Calculates number of summary output samples
   - Formula depends on summarise_last vs periodic mode
   - Called by BatchSolverKernel with chunk_duration

### Architectural Changes to SingleIntegratorRunCore

**New Internal State:**
- `_save_every: Optional[float]` - consolidated save interval
- `_summarise_every: Optional[float]` - consolidated summary interval
- `_sample_summaries_every: Optional[float]` - summary sampling interval
- `_save_last: bool` - derived flag
- `_summarise_last: bool` - derived flag
- `_timing_warning_emitted: bool` - tracks if warning was already shown

**Modified `update()` Method:**

The update method must:
1. Recognize `chunk_duration` in updates_dict and intercept it
2. Call internal timing consolidation helper
3. Pass final timing values to loop

**New Internal Helper: `_consolidate_timing(updates_dict)`**

This method:
1. Reads current timing parameters from updates_dict or existing state
2. Reads output_types from self._output_functions
3. Determines save_last based on:
   - save_every is None AND
   - Any of ["state", "observables", "time"] in output_types
4. Determines summarise_last based on:
   - summarise_every is None AND
   - Any summary metrics in output_types
5. If summarise_last and sample_summaries_every is None:
   - If chunk_duration available: calculate sample_summaries_every = chunk_duration / 100
   - Emit warning if not already emitted
6. Sets internal timing state
7. Returns dict of timing values to pass to loop

**Warning Emission:**

When sample_summaries_every is auto-calculated from duration:
```
"Summary metrics were requested with no summarise_every or sample_summaries_every 
timing. Sample_summaries_every was set to duration / 100 by default. If duration 
changes, the kernel will need to recompile, which will cause a slow integration 
(once). Set timing parameters explicitly to avoid this."
```

---

## Component: ODELoopConfig

### Expected Behavior Changes

**Simplify to Passive Storage:**

Remove the following:
- `_duration` field and `duration` property
- `reset_timing_inference()` method
- `_infer_save_every()` method (if exists)
- `_infer_summarise_every()` method (if exists)
- Mode-setting logic in `__attrs_post_init__` related to timing inference
- `save_last` and `summarise_last` inference logic (if present)

**Keep:**
- `_dt_save` field and `dt_save` property (receives final value from SingleIntegratorRun)
- `_dt_summarise` field and `dt_summarise` property (receives final value)
- `saves_per_summary` property (calculated from dt_save and dt_summarise)
- Basic validation (positive values, etc.)

**The config becomes a simple attrs data class that:**
1. Stores timing values it receives
2. Returns them with precision casting
3. Performs basic validation
4. Does NOT infer or modify timing values

---

## Component: IVPLoop (ode_loop.py)

### Expected Behavior Changes

Minimal changes:
- Remove any timing inference calls
- Ensure update() passes timing params to config without modification
- The loop should remain largely unchanged as it already consumes dt_save and dt_summarise from config

---

## Component: BatchSolverKernel

### Expected Behavior Changes

**Modified Properties:**

1. **`output_length`** (property, int)
   - Current: calculates locally using `floor(duration / dt_save) + 1`
   - New: delegates to `self.single_integrator.output_length(self._duration)`
   - Or when chunking: uses chunk_duration

2. **`summaries_length`** (property, int)
   - Current: calculates locally
   - New: delegates to `self.single_integrator.summaries_length(self._duration)`

3. **`warmup_length`** (property, int)
   - Check if this is used anywhere
   - If not used: delete
   - If used: delegate or keep calculation

**Modified `run()` Method:**

- Pass `chunk_duration` in updates_dict when calling single_integrator.update()
- This allows SingleIntegratorRun to calculate sample_summaries_every when needed

---

## Component: Solver

### Expected Behavior Changes

**Remove Warning Logic:**

- Remove any code that emits warnings about sample_summaries_every
- Warning responsibility moves to SingleIntegratorRun
- The warning is emitted during timing consolidation, not during solve()

---

## Integration Points

### Parameter Flow

```
User Parameters (solve_ivp/Solver)
    ↓
Solver.solve() - passes duration to kernel.run()
    ↓
BatchSolverKernel.run() - calculates chunk_duration, calls single_integrator.update()
    ↓
SingleIntegratorRun.update() - intercepts chunk_duration, consolidates timing
    ↓
_consolidate_timing() - determines final timing values, emits warnings
    ↓
ODELoopConfig - receives and stores final timing values
```

### Output Sizing Flow

```
BatchSolverKernel needs output_length
    ↓
Calls self.single_integrator.output_length(self._duration)
    ↓
SingleIntegratorRun calculates: 1 + floor(duration / save_every) + save_last
    ↓
Returns int
```

---

## Data Structures

### Timing Parameters Set (add to recognized parameters)

```python
TIMING_PARAMETERS = {
    "save_every",
    "summarise_every", 
    "sample_summaries_every",
    "chunk_duration",  # New - passed from BatchSolverKernel
}
```

---

## Dependencies and Imports

**SingleIntegratorRun/Core:**
- Needs access to output_functions.output_types
- warnings module for warning emission
- numpy for floor calculation

**ODELoopConfig:**
- Reduced dependencies - no longer needs duration-based calculations

---

## Edge Cases

1. **All timing parameters None**: Results in save_last=True, summarise_last=True based on output_types
2. **No time-domain outputs requested**: save_last=False, save_every irrelevant
3. **No summary outputs requested**: summarise_last=False, summarise_every irrelevant
4. **Duration changes between solves**: Warning already emitted on first auto-calculation; recompilation happens silently
5. **Explicit sample_summaries_every with None summarise_every**: No warning, uses provided value

---

## Test Consolidation

### Tests to Delete
- `tests/batchsolving/test_duration_propagation.py` - Tests removed duration storage
- `tests/batchsolving/test_timing_modes.py` - Manual construction violates conventions

### Tests to Move/Consolidate

From `test_kernel_output_lengths.py`:
- Keep small test verifying None handling
- Move to `tests/batchsolving/test_SolverKernel.py`

From `test_solver_timing_properties.py`:
- Consolidate into `tests/batchsolving/test_solver.py`

From `test_solver_warnings.py`:
- Move warning tests to `tests/integrators/test_SingleIntegratorRun.py`
- Rewrite using fixtures

From `test_ode_loop_config_timing.py`:
- Delete tests for removed inference logic
- Keep tests for basic dt_save/dt_summarise storage if any

### New Tests Needed

In `tests/integrators/test_SingleIntegratorRun.py`:
1. Test `any_time_domain_outputs` property with various output_types
2. Test `output_length(duration)` method calculations
3. Test `summaries_length(duration)` method calculations
4. Test warning emission when sample_summaries_every auto-calculated
5. Test timing consolidation with fixture overrides

---

## Implementation Notes for detailed_implementer

1. **Start with ODELoopConfig simplification** - Remove mode-inference logic first to establish the "passive storage" pattern
2. **Then add SingleIntegratorRun timing properties and methods** - Add the new timing API
3. **Update SingleIntegratorRunCore.update()** - Add chunk_duration interception and consolidation
4. **Update BatchSolverKernel** - Delegate output_length calls
5. **Remove Solver warning logic** - Clean up deprecated code
6. **Delete test files** - Remove files that test removed behavior
7. **Consolidate/rewrite tests** - Using fixtures per project convention

---

## Validation for Reviewer

The reviewer should verify:
1. Timing parameters flow correctly from user through to loop
2. output_length/summaries_length calculations match expected behavior
3. Warning is emitted by SingleIntegratorRun, not Solver
4. ODELoopConfig no longer contains mode-inference logic
5. All tests use fixtures (no manual Solver construction)
6. No duration storage below BatchSolverKernel
