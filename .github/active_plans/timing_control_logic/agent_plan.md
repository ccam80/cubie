# Agent Plan: Timing Control Logic Implementation

## Overview

This plan implements the timing control logic in SingleIntegratorRunCore's `__init__` and `update` methods. The architectural foundation (properties, methods, simplified ODELoopConfig) is already in place. This implementation adds the active logic that detects output types, sets timing flags, and computes sample_summaries_every when needed.

---

## Component: SingleIntegratorRunCore

### Expected Behavior Changes in `__init__`

**Location**: After OutputFunctions is created, before loop instantiation

**New Logic**:
1. Extract output_types from the newly created OutputFunctions
2. Determine if time-domain outputs are requested (state, observables, time)
3. Determine if summary outputs are requested (any other output_types)
4. If timing parameters are None but outputs are requested, set appropriate flags

**Constants to Add** (at module level in SingleIntegratorRun.py):
```python
# Output types that represent time-domain samples
TIME_DOMAIN_OUTPUT_TYPES = frozenset({"state", "observables", "time"})
```

**Logic to Add in __init__**:
```python
# After OutputFunctions creation, before loop instantiation:

# Determine if timing flags need to be set based on output_types
output_types = set(self._output_functions.output_types)
has_time_domain_outputs = bool(TIME_DOMAIN_OUTPUT_TYPES & output_types)
has_summary_outputs = bool(output_types - TIME_DOMAIN_OUTPUT_TYPES)

# Get timing parameters from loop_settings (may be None)
save_every = loop_settings.get("save_every", None)
summarise_every = loop_settings.get("summarise_every", None)
sample_summaries_every = loop_settings.get("sample_summaries_every", None)

# Set save_last if time-domain outputs requested but no save_every
if save_every is None and has_time_domain_outputs:
    loop_settings["save_last"] = True

# Set summarise_last if summary outputs requested but no summarise timing
if summarise_every is None and sample_summaries_every is None and has_summary_outputs:
    loop_settings["summarise_last"] = True
```

### Expected Behavior Changes in `update`

**Location**: Beginning of update method, after updates_dict is prepared

**New Logic**:
1. Check for "chunk_duration" in updates_dict
2. Intercept chunk_duration (remove from dict before passing to loop)
3. If summarise_last is True and sample_summaries_every is None:
   - Calculate sample_summaries_every = chunk_duration / 100
   - Emit warning (via SingleIntegratorRun)
   - Add computed value to updates_dict for loop

**Logic to Add in update**:
```python
# Intercept chunk_duration - not passed to lower layers
chunk_duration = updates_dict.pop("chunk_duration", None)

# After output_functions update, before loop update:
# If we need to compute sample_summaries_every from duration
if chunk_duration is not None:
    # Check current timing state
    loop_config = self._loop.compile_settings
    summarise_last = loop_config.summarise_last
    current_sample_summaries_every = loop_config._sample_summaries_every
    
    # If summarise_last mode with no explicit sample_summaries_every
    if summarise_last and current_sample_summaries_every is None:
        # Compute from duration
        computed_sample_summaries_every = chunk_duration / 100.0
        updates_dict["sample_summaries_every"] = computed_sample_summaries_every
        # Warning emitted by SingleIntegratorRun wrapper
```

---

## Component: SingleIntegratorRun

### Expected Behavior Changes

**New Internal State**:
- `_timing_warning_emitted: bool` - tracks if duration-dependency warning was shown

**Warning Emission Logic**:

The warning should be emitted when sample_summaries_every is computed from duration. This happens in the update flow.

**Option A**: Add warning check in SingleIntegratorRun.update() override (if exists)
**Option B**: Add warning emission as a method called from SingleIntegratorRunCore.update()

Since SingleIntegratorRun is a subclass that adds properties, and update() is in SingleIntegratorRunCore, we should:
1. Add a flag to track warning emission
2. Check and emit warning in SingleIntegratorRunCore.update() when computing sample_summaries_every

**Warning Text**:
```python
warnings.warn(
    "Summary metrics were requested with no summarise_every or "
    "sample_summaries_every timing. Sample_summaries_every was set to "
    "duration / 100 by default. If duration changes, the kernel will need "
    "to recompile, which will cause a slow integration (once). Set timing "
    "parameters explicitly to avoid this.",
    UserWarning,
    stacklevel=3
)
```

---

## Component: BatchSolverKernel

### Expected Behavior Changes in `run()`

**Location**: After chunk_run() call, when updating single_integrator

**Current Code** (around line 344):
```python
# Propagate duration to single_integrator for loop config
self.single_integrator.update({"duration": duration}, silent=True)
```

**New Code**:
```python
# Propagate chunk_duration to single_integrator for timing calculations
# (duration itself is NOT passed lower - only used for timing computation)
self.single_integrator.update({"chunk_duration": chunk_duration}, silent=True)
```

Where `chunk_duration` is obtained from `chunk_params.duration` (already calculated).

**Note**: The existing code passes `duration` which is the full simulation duration. We need to pass `chunk_duration` which is the per-chunk duration for proper timing calculations.

---

## Integration Points

### Parameter Recognition

Add "chunk_duration" to recognized parameters in SingleIntegratorRunCore:
- It's intercepted in update() and not passed to loop
- Should be marked as recognized to avoid KeyError

### Data Flow

```
BatchSolverKernel.run()
    |
    +-> chunk_run() calculates chunk_params.duration
    |
    +-> single_integrator.update({"chunk_duration": chunk_duration})
            |
            +-> SingleIntegratorRunCore.update()
                    |
                    +-> Pop chunk_duration (don't pass to loop)
                    |
                    +-> If summarise_last and no sample_summaries_every:
                    |       |
                    |       +-> Compute sample_summaries_every = duration/100
                    |       |
                    |       +-> Emit warning (once)
                    |       |
                    |       +-> Add to updates_dict for loop
                    |
                    +-> Continue with normal update flow
```

---

## Edge Cases

1. **chunk_duration not provided**: No timing computation, use existing values
2. **sample_summaries_every already set**: Don't override, no warning
3. **summarise_last is False**: No computation needed
4. **Multiple update calls**: Warning only emitted once (tracked by flag)
5. **No summary outputs requested**: summarise_last should be False, no computation

---

## Dependencies and Imports

**SingleIntegratorRunCore**:
- Already imports `warnings`
- No new imports needed

**SingleIntegratorRun**:
- No new imports needed

**BatchSolverKernel**:
- No new imports needed
- Uses existing chunk_params structure

---

## Tests to Create

### Test: Timing Flag Auto-Detection
- Verify save_last=True when state output requested with no save_every
- Verify summarise_last=True when mean output requested with no summarise timing

### Test: Sample Summaries Calculation
- Verify sample_summaries_every computed from chunk_duration when summarise_last
- Verify computed value = chunk_duration / 100

### Test: Warning Emission
- Verify warning emitted when sample_summaries_every computed from duration
- Verify warning only emitted once across multiple updates

### Test: Chunk Duration Interception
- Verify chunk_duration not passed to loop
- Verify loop doesn't receive duration parameter

---

## Implementation Order

1. **SingleIntegratorRunCore.__init__**: Add timing flag detection logic
2. **SingleIntegratorRunCore.update**: Add chunk_duration interception and sample_summaries_every computation
3. **SingleIntegratorRunCore**: Add warning flag and emission logic
4. **BatchSolverKernel.run**: Update to pass chunk_duration instead of duration
5. **Tests**: Create tests for new behavior

---

## Validation for Reviewer

The reviewer should verify:
1. Timing flags (save_last, summarise_last) set correctly based on output_types
2. sample_summaries_every computed from chunk_duration when appropriate
3. Warning emitted once when duration-dependent computation occurs
4. chunk_duration intercepted and not passed to loop
5. No duration storage below BatchSolverKernel level
6. All tests use fixtures per project convention
