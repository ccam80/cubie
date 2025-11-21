# Technical Implementation Plan: dt_update_summaries

## Component Overview

This implementation separates the concerns of state saving (`dt_save`) and summary metric updates (`dt_update_summaries`) in CuBIE's integration loop architecture.

### Core Components to Modify

1. **ODELoopConfig** (`src/cubie/integrators/loops/ode_loop_config.py`)
   - Configuration container for loop compile settings
   - Currently stores `_dt_save` and `_dt_summarise`
   - Needs to store and validate `_dt_update_summaries`

2. **IVPLoop** (`src/cubie/integrators/loops/ode_loop.py`)
   - Main integration loop factory
   - Currently couples `do_save` with `update_summaries` calls
   - Needs separate `do_update_summary` logic

3. **OutputConfig** (`src/cubie/outputhandling/output_config.py`) [Optional]
   - May need `dt_update_summaries` if it should be exposed at output configuration level
   - Evaluate whether this belongs in OutputConfig or only in ODELoopConfig

4. **Test Suite** (`tests/integrators/loops/test_ode_loop.py`)
   - Add tests for new parameter validation
   - Add tests for separate update timing
   - Verify backward compatibility

### Expected Behavior

#### ODELoopConfig Behavior

**New Attribute:**
- `_dt_update_summaries`: Private float attribute with leading underscore
- Property `dt_update_summaries` returns `self.precision(self._dt_update_summaries)`
- Default value: `None` (will be set to `dt_save` if not provided)
- Validator: Must be `> 0` and must divide evenly into `dt_summarise`

**New Property:**
- `updates_per_summary`: Returns `int(self.dt_summarise // self.dt_update_summaries)`
- Replaces current usage of `saves_per_summary` for summary logic

**Validation Logic:**
- In `__attrs_post_init__` or setter: Check `dt_summarise % dt_update_summaries == 0`
- Raise `ValueError` with clear message if not an integer divisor
- If `dt_update_summaries` is `None`, default to `dt_save` value

#### IVPLoop Behavior

**Timing Variables (in `build()` method):**
```python
dt_update_summaries = precision(config.dt_update_summaries)
updates_per_summary = config.updates_per_summary
steps_per_update = int32(ceil(precision(dt_update_summaries) / precision(dt0)))
```

**Fixed-Step Mode:**
- Add `update_counter` variable (initialized to 0)
- Check `do_update_summary = (update_counter % steps_per_update) == 0`
- Increment `update_counter` after each accepted step
- Reset `update_counter` to 0 after each update

**Adaptive-Step Mode:**
- Add `next_update_summary` variable (initialized like `next_save`)
- Check `do_update_summary = (t + dt[0] + equality_breaker) >= next_update_summary`
- Update `next_update_summary += dt_update_summaries` when update occurs

**Loop Structure Changes:**
```python
# Current (lines 518-554):
if do_save:
    save_state(...)
    if summarise:
        update_summaries(...)
        if (save_idx + 1) % saves_per_summary == 0:
            save_summaries(...)

# New structure:
if do_save:
    save_state(...)

if do_update_summary:
    if summarise:
        update_summaries(...)
        if (update_idx + 1) % updates_per_summary == 0:
            save_summaries(...)
        update_idx += 1
```

**Index Management:**
- Add `update_idx` variable to track summary updates (separate from `save_idx`)
- Initialize `update_idx = int32(0)` alongside `save_idx`
- Increment `update_idx` after each `update_summaries` call
- Use `update_idx` (not `save_idx`) in `update_summaries` calls

**Initial Summary Handling:**
- At t=0 or after settling time, if summaries enabled:
  - Call `update_summaries` with `update_idx = 0`
  - Initialize `next_update_summary` appropriately
  - May need to initialize with empty summary or first update

### Architectural Changes

#### ODELoopConfig Structure

**Current attributes:**
```python
_dt_save: float = 0.1
_dt_summarise: float = 1.0
```

**New attributes:**
```python
_dt_save: float = 0.1
_dt_summarise: float = 1.0
_dt_update_summaries: Optional[float] = None  # Defaults to dt_save
```

**New properties:**
```python
@property
def dt_update_summaries(self) -> float:
    """Return the summary update interval."""
    update_val = self._dt_update_summaries if self._dt_update_summaries is not None else self._dt_save
    return self.precision(update_val)

@property
def updates_per_summary(self) -> int:
    """Return the number of updates between summary outputs."""
    return int(self.dt_summarise // self.dt_update_summaries)
```

**Validation method:**
```python
def __attrs_post_init__(self):
    """Validate dt_update_summaries divides dt_summarise evenly."""
    if self._dt_update_summaries is None:
        self._dt_update_summaries = self._dt_save
    
    # Validate divisibility
    if self.dt_summarise % self.dt_update_summaries != 0:
        raise ValueError(
            f"dt_update_summaries ({self.dt_update_summaries}) must be an "
            f"integer divisor of dt_summarise ({self.dt_summarise})"
        )
```

#### IVPLoop Integration Points

**Constructor:**
```python
def __init__(
    self,
    precision: PrecisionDType,
    shared_indices: LoopSharedIndices,
    local_indices: LoopLocalIndices,
    compile_flags: OutputCompileFlags,
    dt_save: float = 0.1,
    dt_summarise: float = 1.0,
    dt_update_summaries: Optional[float] = None,  # NEW
    # ... other parameters
):
```

**Pass to ODELoopConfig:**
```python
config = ODELoopConfig(
    # ... existing parameters
    dt_save=dt_save,
    dt_summarise=dt_summarise,
    dt_update_summaries=dt_update_summaries,  # NEW
    # ... remaining parameters
)
```

**Property to expose:**
```python
@property
def dt_update_summaries(self) -> float:
    """Return the summary update interval."""
    return self.compile_settings.dt_update_summaries
```

#### Integration with OutputConfig

**Decision Point:** Should `dt_update_summaries` be part of OutputConfig?

**Analysis:**
- OutputConfig already has `dt_save` as it's used for output array sizing
- `dt_update_summaries` affects loop behavior but not output buffer sizes
- Summary buffer sizes depend on metric types, not update frequency
- `updates_per_summary` replaces `saves_per_summary` but both are computed values

**Recommendation:** 
- Keep `dt_update_summaries` in ODELoopConfig only
- Do NOT add to OutputConfig (it's a loop timing parameter, not an output configuration)
- ODELoopConfig can calculate `updates_per_summary` independently

### Data Structures

#### Loop State Variables

**Current state tracking:**
```python
save_idx = int32(0)        # Tracks number of saves
summary_idx = int32(0)     # Tracks number of summary writes
next_save = precision(dt_save)  # Next save time
```

**New state tracking:**
```python
save_idx = int32(0)        # Tracks number of saves (unchanged)
summary_idx = int32(0)     # Tracks number of summary writes (unchanged)
update_idx = int32(0)      # NEW: Tracks number of summary updates
next_save = precision(dt_save)              # Next save time (unchanged)
next_update_summary = precision(dt_update_summaries)  # NEW: Next update time
```

**Fixed-step mode counters:**
```python
step_counter = int32(0)    # Existing: counts steps for saves
update_counter = int32(0)  # NEW: counts steps for summary updates
```

#### Timing Constants

**Captured in loop closure:**
```python
dt_save = precision(config.dt_save)
dt_summarise = precision(config.dt_summarise)
dt_update_summaries = precision(config.dt_update_summaries)  # NEW
saves_per_summary = config.saves_per_summary  # Remove or keep for compatibility
updates_per_summary = config.updates_per_summary  # NEW
steps_per_save = int32(ceil(dt_save / dt0))
steps_per_update = int32(ceil(dt_update_summaries / dt0))  # NEW
```

### Dependencies and Imports

**No new imports required** - all necessary types and utilities already available:
- `attrs` for validation
- `gttype_validator`, `opt_gttype_validator` from `cubie._utils`
- `int32` from numba for loop counters
- `ceil` from math for step calculations

**Existing imports to utilize:**
```python
from cubie._utils import gttype_validator, opt_gttype_validator
from numba import int32
from math import ceil
```

### Edge Cases to Consider

1. **dt_update_summaries == dt_save (default)**
   - Should behave identically to current implementation
   - Update and save happen simultaneously
   - No additional overhead

2. **dt_update_summaries > dt_save**
   - Updates occur less frequently than saves
   - Valid use case: save states frequently but update summaries occasionally
   - Summaries computed over fewer samples

3. **dt_update_summaries < dt_save**
   - Updates occur more frequently than saves
   - Primary use case: summary-only output with no state saves (dt_save → ∞)
   - Summaries computed over many samples

4. **dt_update_summaries doesn't divide dt_summarise**
   - Must be caught in validation
   - Clear error message with actual values
   - Suggest nearest valid values in error message?

5. **Settling time interactions**
   - If settling_time > 0, first update should occur at settling_time
   - Similar logic to current first save handling
   - Initialize `next_update_summary` appropriately

6. **Initial summary (t=0) handling**
   - Current code calls `update_summaries` at t=0 if settling_time == 0
   - New code should maintain this behavior
   - First update_idx should be 0 at initial call

7. **Fixed-step mode with very small dt_update_summaries**
   - `steps_per_update` could be 1 (update every step)
   - Should work correctly but may impact performance
   - No special handling needed - natural consequence

8. **Adaptive mode with dt approaching dt_update_summaries**
   - Similar to current dt_save logic
   - Use equality_breaker to handle floating-point precision
   - `do_update_summary = (t + dt[0] + equality_breaker) >= next_update_summary`

9. **Summary-only mode (no saves, only summaries)**
   - User sets output_types to only include summary metrics
   - save_state_bool = False, summarise = True
   - `do_save` always False, but `do_update_summary` can be True
   - This is the primary motivation for the feature

10. **Memory and iteration counters**
    - iteration_counters (if enabled) saved with saves, not updates
    - No interaction with dt_update_summaries
    - Keep counter logic tied to `do_save`

### Interactions Between Components

#### ODELoopConfig → IVPLoop
- ODELoopConfig stores and validates `dt_update_summaries`
- IVPLoop reads `config.dt_update_summaries` and `config.updates_per_summary`
- Validation errors surface during ODELoopConfig initialization, before loop compilation

#### update_summaries → save_summaries
- `update_summaries` accumulates metrics into buffers
- Called when `do_update_summary` is True
- Increments `update_idx` after call
- `save_summaries` writes buffers to output when `update_idx % updates_per_summary == 0`
- Receives `updates_per_summary` parameter (not `saves_per_summary`)

#### Fixed-step vs Adaptive logic
- Both modes need separate `do_update_summary` calculation
- Fixed: counter-based (`update_counter % steps_per_update`)
- Adaptive: time-based (`t >= next_update_summary`)
- Logic should be symmetric to existing `do_save` logic

#### Initial state (t=0 or settling_time)
- When settling_time == 0:
  - Save initial state
  - Call `update_summaries` with `update_idx = 0`
  - Set `next_update_summary = dt_update_summaries`
  - Increment `update_idx` to 1
- When settling_time > 0:
  - Skip initial save/update
  - Set `next_update_summary = settling_time` (first update after settling)
  - First update occurs when t reaches settling_time

### ALL_LOOP_SETTINGS Update

**Current set:**
```python
ALL_LOOP_SETTINGS = {
    "dt_save",
    "dt_summarise",
    "dt0",
    "dt_min",
    "dt_max",
    "is_adaptive",
}
```

**Updated set:**
```python
ALL_LOOP_SETTINGS = {
    "dt_save",
    "dt_summarise",
    "dt_update_summaries",  # NEW
    "dt0",
    "dt_min",
    "dt_max",
    "is_adaptive",
}
```

This allows the parameter to be recognized in update dictionaries and parameter filtering.

### Testing Strategy

**Test Categories:**

1. **Validation Tests**
   - Test that `dt_update_summaries > 0` is enforced
   - Test that `dt_summarise % dt_update_summaries == 0` is enforced
   - Test error messages are clear and helpful
   - Test default value (`None` → `dt_save`)

2. **Backward Compatibility Tests**
   - Verify existing tests pass without modification
   - Verify default behavior matches current implementation
   - Compare results with/without explicit `dt_update_summaries = dt_save`

3. **Functional Tests**
   - Test `dt_update_summaries < dt_save` (more updates than saves)
   - Test `dt_update_summaries > dt_save` (fewer updates than saves)
   - Test `dt_update_summaries == dt_save` (default case)
   - Test with various `updates_per_summary` values (2, 5, 10, 100)

4. **Integration Tests**
   - Test with different algorithms (fixed-step, adaptive)
   - Test with different controllers
   - Test with settling_time > 0
   - Test summary-only output mode (no state saves)

5. **Edge Case Tests**
   - Test `dt_update_summaries` very small (many updates)
   - Test `dt_update_summaries` very large (few updates)
   - Test initial summary at t=0
   - Test summary metrics correctness with different update cadences

**Test Implementation Approach:**
- Use pytest fixtures from `tests/conftest.py`
- Parameterize tests with different `dt_update_summaries` values
- Use `indirect` fixture overrides for settings
- Verify summary metric values are mathematically correct
- Compare against CPU reference implementations where applicable

