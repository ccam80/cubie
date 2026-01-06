# Fix Timing Issues - Agent Plan

## Summary

This plan addresses four surgical fixes to the timing parameter handling in CuBIE:
1. Correct `OutputFunctions` API usage (remove `save_every` from output_settings)
2. Add `SingleIntegratorRun.is_duration_dependent` property
3. Apply sentinel duration early to prevent NaN errors
4. Remove sentinel values from loop properties

---

## Component 1: OutputFunctions API Correction

### Current Behavior
The `OutputFunctions` class accepts `sample_summaries_every` in its constructor (defined in `ALL_OUTPUT_FUNCTION_PARAMETERS`). However, the test `test_all_lower_plumbing` incorrectly passes `save_every` in `output_settings`, which is not a valid parameter for `OutputFunctions`.

### Expected Behavior
- `save_every` should only appear in `loop_settings`
- `output_settings` should only contain parameters from `ALL_OUTPUT_FUNCTION_PARAMETERS`
- The test should pass without API errors

### Integration Points
- File: `tests/batchsolving/test_SolverKernel.py`
- Function: `test_all_lower_plumbing`
- Line ~187: Remove `"save_every": 0.01,` from `output_settings` dict

### Data Flow
```
output_settings dict → BatchSolverKernel → SingleIntegratorRunCore → OutputFunctions(**output_settings)
```

The `save_every` is redundantly in both `output_settings` AND `loop_settings` in the test. Since it's already in `loop_settings`, removing it from `output_settings` is the correct fix.

---

## Component 2: is_duration_dependent Property

### Current Behavior
There is no property to check if the loop function depends on duration for compilation. The `chunk_duration` update logic in `SingleIntegratorRunCore.update()` checks conditions inline.

### Expected Behavior
Add `SingleIntegratorRun.is_duration_dependent` property that:
- Checks if `summarise_last` is True in the loop config
- Checks if `_sample_summaries_every` is None (no explicit value set)
- Returns True only when both conditions are met (duration is needed to compute `sample_summaries_every`)

### Integration Points
- File: `src/cubie/integrators/SingleIntegratorRun.py`
- Location: Add new property in the "Loop properties" section (after line ~210)

### Property Logic
```python
@property
def is_duration_dependent(self) -> bool:
    """Return True when the loop is compile-dependent on duration."""
    loop_config = self._loop.compile_settings
    return loop_config.summarise_last and loop_config._sample_summaries_every is None
```

---

## Component 3: Sentinel Duration for NaN Prevention

### Current Behavior
When `is_duration_dependent` is True but no duration has been provided yet, calculations that depend on `sample_summaries_every` may produce NaN values because the value hasn't been computed from `chunk_duration`.

### Expected Behavior
- During the `update()` method, update the loop with the updates dict BEFORE checking `is_duration_dependent`
- This ensures that if the user provided timing parameters in the update, they are in place before the duration-dependent check
- The existing logic in `update()` already computes `sample_summaries_every` from `chunk_duration` when needed

### Integration Points
- File: `src/cubie/integrators/SingleIntegratorRunCore.py`
- Method: `update()`
- The update already happens in the correct order - the issue is that the condition check should use the new property

### Modification
The chunk_duration update block should use the new `is_duration_dependent` property (accessed via self since SingleIntegratorRun inherits from SingleIntegratorRunCore) or check the loop config directly. Since `is_duration_dependent` is on SingleIntegratorRun (the subclass), the core should check the conditions directly.

```python
# Gate the chunk_duration processing with duration dependency check
loop_config = self._loop.compile_settings
is_duration_dep = loop_config.summarise_last and loop_config._sample_summaries_every is None

if chunk_duration is not None and is_duration_dep:
    # Compute sample_summaries_every from chunk_duration
    ...
```

The existing code already does this check inline - but it should be gated properly.

---

## Component 4: Remove samples_per_summary Sentinel

### Current Behavior
In `ODELoopConfig.samples_per_summary`, when in `summarise_last` mode with no timing set, the property returns `2**30` as a sentinel value:

```python
@property
def samples_per_summary(self) -> Optional[int]:
    """Return the number of updates between summary outputs."""
    if self._summarise_every is None or self._sample_summaries_every is None:
        # In summarise_last mode, return large sentinel so modulo never triggers
        if self.summarise_last:
            return 2**30
        return None
    return round(self.summarise_every / self.sample_summaries_every)
```

### Expected Behavior
- Return `None` consistently when timing is not configured
- Calling code should handle `None` appropriately
- No magic sentinel values in the property

### Integration Points
- File: `src/cubie/integrators/loops/ode_loop_config.py`
- Property: `samples_per_summary` (line ~317)

- File: `src/cubie/integrators/loops/ode_loop.py`
- Location: Where `samples_per_summary` is used (line ~373, ~558, ~805, ~815)

### Modification Strategy
1. Change `samples_per_summary` to return `None` instead of `2**30` in summarise_last mode
2. In `ode_loop.py`, where `samples_per_summary` is used in the compiled loop:
   - If `samples_per_summary` is `None` and `summarise_last` is True, use `2**30` as a compile-time constant directly in the loop compilation (not from the property)
   - This keeps the property clean while maintaining the same runtime behavior

### Code in ode_loop.py build()
```python
samples_per_summary = config.samples_per_summary
# For summarise_last mode, use large value so modulo never triggers regular saves
if samples_per_summary is None and config.summarise_last:
    samples_per_summary = 2**30
```

This moves the sentinel logic from the config property to the loop compilation where it's actually needed.

---

## Dependencies Between Components

```
Component 1 (API fix) ─── Independent, can be done first
        │
        ▼
Component 4 (sentinel removal) ─── Independent of 2 & 3
        │
        ▼
Component 2 (is_duration_dependent) ─── Creates property needed by logic
        │
        ▼  
Component 3 (NaN prevention) ─── Uses is_duration_dependent check
```

All components are relatively independent but should be done in this order for logical coherence.

---

## Edge Cases

### Edge Case 1: No timing parameters provided
- `save_every`, `summarise_every`, `sample_summaries_every` all None
- Loop should use `save_last` and `summarise_last` modes
- `samples_per_summary` returns None, loop compilation handles it

### Edge Case 2: Only summarise_every provided
- `sample_summaries_every` inferred from `summarise_every`
- `is_duration_dependent` returns False (timing is explicit)

### Edge Case 3: summarise_last mode with chunk_duration
- `is_duration_dependent` returns True initially
- When `chunk_duration` is provided, `sample_summaries_every` is computed
- After update, `is_duration_dependent` returns False

### Edge Case 4: chunk_duration changes multiple times
- Warning should only be emitted once (existing `_timing_warning_emitted` flag)
- Each new chunk_duration recomputes `sample_summaries_every`

---

## Validation Criteria

1. Test `test_all_lower_plumbing` passes without API errors
2. `is_duration_dependent` property returns correct values in all scenarios
3. No NaN errors during loop initialization
4. `samples_per_summary` property returns `None` in summarise_last mode
5. Loop compilation still works correctly with the sentinel moved to build()
