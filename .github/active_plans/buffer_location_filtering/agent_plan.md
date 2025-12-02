# Buffer Location Argument Filtering - Agent Plan

## Overview

This plan implements buffer location argument filtering to enable users to specify buffer memory locations (local/shared) at the Solver level, with proper propagation through the hierarchy and nested update support in CUDAFactory.

## Component 1: ALL_BUFFER_LOCATION_PARAMETERS Constant

### Location
`src/cubie/integrators/loops/ode_loop.py`

### Expected Behavior
A `frozenset` or `set` constant containing all location attribute names from `LoopBufferSettings`:
- `state_buffer_location`
- `state_proposal_location`  
- `parameters_location`
- `drivers_location`
- `drivers_proposal_location`
- `observables_location`
- `observables_proposal_location`
- `error_location`
- `counters_location`
- `state_summary_location`
- `observable_summary_location`
- `scratch_location`

### Integration Points
- Imported by `solver.py` for argument filtering
- Used with `merge_kwargs_into_settings()` utility

---

## Component 2: Solver-Level Argument Filtering

### Location
`src/cubie/batchsolving/solver.py`

### Expected Behavior

#### At Initialization (`__init__`)
1. Import `ALL_BUFFER_LOCATION_PARAMETERS` from `ode_loop`
2. After existing `merge_kwargs_into_settings` calls, add filtering for buffer locations:
   ```
   buffer_settings, buffer_recognized = merge_kwargs_into_settings(
       kwargs=kwargs, 
       valid_keys=ALL_BUFFER_LOCATION_PARAMETERS,
       user_settings=loop_settings
   )
   ```
3. The returned `buffer_settings` replaces `loop_settings` (merged into it)
4. Add `buffer_recognized` to `recognized_kwargs`
5. Pass merged `loop_settings` to `BatchSolverKernel` (unchanged interface)

#### At Update (`update`)
1. Buffer location parameters flow through `self.kernel.update()`
2. The nested update mechanism in CUDAFactory handles recognition

### Data Structures
- No new data structures; buffer kwargs merge into existing `loop_settings` dict

---

## Component 3: Nested Update Support in CUDAFactory

### Location
`src/cubie/CUDAFactory.py`

### Expected Behavior

Enhance `update_compile_settings()` to check one level of nesting:

1. After checking top-level attributes, if key not found:
2. Iterate through `_compile_settings` attributes
3. For each attribute that is a dict or attrs class:
   - Check if the update key exists in that nested structure
   - If found, update the nested value and track recognition/change
4. If any nested value changed, mark for cache invalidation
5. Return union of recognized keys (top-level + nested)

### Algorithm

```
for key, value in updates_dict.items():
    # First: try top-level (existing logic)
    recognized, updated = _check_and_update(f"_{key}", value)
    if not recognized:
        recognized, updated = _check_and_update(key, value)
    
    # Second: try nested (new logic)
    if not recognized:
        for attr_name in compile_settings_fields:
            nested = getattr(_compile_settings, attr_name)
            if is_dict(nested) and key in nested:
                # Update dict value
                if nested[key] != value:
                    nested[key] = value
                    updated = True
                recognized = True
                break
            elif is_attrs_class(nested) and in_attr(key, nested):
                # Use existing _check_and_update pattern on nested
                r, u = _check_and_update_on(nested, key, value)
                recognized |= r
                updated |= u
                break
```

### Dependencies
- `attrs.has()` - check if object is attrs class
- `attrs.fields()` - iterate attrs class fields
- Existing `in_attr()` utility from `_utils.py`

### Edge Cases
- Nested structure is `None` - skip
- Key appears in multiple nested structures - first match wins (document behavior)
- Nested attrs class has underscore-prefixed attribute - handle via existing `in_attr`

---

## Component 4: SingleIntegratorRunCore Buffer Settings Propagation

### Location
`src/cubie/integrators/SingleIntegratorRunCore.py`

### Expected Behavior

#### At `instantiate_loop()`
1. Extract buffer location kwargs from `loop_settings` dict
2. Pass location kwargs to `LoopBufferSettings` constructor
3. Current code already creates `LoopBufferSettings` but hardcodes all locations to 'shared'
4. Modify to use values from `loop_settings` if provided, else use defaults

#### At `update()`
1. Buffer location kwargs flow through `self._loop.update()`
2. The nested update mechanism handles them via `ODELoopConfig.buffer_settings`

### Changes Required
In `instantiate_loop()`:
- Extract location keys from `loop_settings`
- Build `LoopBufferSettings` with extracted locations
- Remove hardcoded 'shared' defaults where user provides alternative

In `update()`:
- When rebuilding `buffer_settings`, extract location kwargs from `updates_dict`
- Preserve existing location values when not explicitly updated

---

## Component 5: IVPLoop Configuration Integration

### Location
`src/cubie/integrators/loops/ode_loop.py`

### Expected Behavior
- `IVPLoop.__init__` already accepts `buffer_settings` parameter
- `ODELoopConfig` already has `buffer_settings` field
- No changes required to IVPLoop itself

### Verification
Ensure `ODELoopConfig.buffer_settings` is included in compile settings and accessible for nested updates.

---

## Integration Points Summary

| Component | Imports From | Exports To |
|-----------|--------------|------------|
| `ode_loop.py` | - | `ALL_BUFFER_LOCATION_PARAMETERS` |
| `solver.py` | `ALL_BUFFER_LOCATION_PARAMETERS` | (filtered kwargs to BatchSolverKernel) |
| `CUDAFactory.py` | `in_attr` from `_utils` | (enhanced update_compile_settings) |
| `SingleIntegratorRunCore.py` | - | (buffer_settings to IVPLoop) |

---

## Dependencies and Imports

### New Imports Required

**solver.py:**
```python
from cubie.integrators.loops.ode_loop import ALL_BUFFER_LOCATION_PARAMETERS
```

**CUDAFactory.py:**
```python
from attrs import fields, has
# (may already be imported)
```

---

## Expected Interactions

### Instantiation Flow
1. User: `Solver(system, state_buffer_location='shared')`
2. Solver filters kwarg via `merge_kwargs_into_settings`
3. Filtered dict passed to `BatchSolverKernel(loop_settings=...)`
4. BatchSolverKernel passes to `SingleIntegratorRun(loop_settings=...)`
5. SingleIntegratorRunCore calls `instantiate_loop()` with loop_settings
6. `instantiate_loop` extracts location kwargs, creates `LoopBufferSettings`
7. `LoopBufferSettings` passed to `IVPLoop(buffer_settings=...)`
8. `IVPLoop` sets up `ODELoopConfig` with buffer_settings
9. Kernel compilation uses buffer locations from settings

### Update Flow
1. User: `solver.update(state_buffer_location='local')`
2. Solver calls `self.kernel.update(updates_dict)`
3. BatchSolverKernel calls `self.single_integrator.update(updates_dict)`
4. SingleIntegratorRunCore calls `self._loop.update(updates_dict)`
5. IVPLoop calls `self.update_compile_settings(updates_dict)`
6. CUDAFactory nested check finds `state_buffer_location` in `buffer_settings`
7. Value updated in `ODELoopConfig.buffer_settings`
8. Cache invalidated, loop will recompile on next access

---

## Data Structures

### ALL_BUFFER_LOCATION_PARAMETERS
```python
ALL_BUFFER_LOCATION_PARAMETERS = {
    "state_buffer_location",
    "state_proposal_location",
    "parameters_location",
    "drivers_location",
    "drivers_proposal_location",
    "observables_location",
    "observables_proposal_location",
    "error_location",
    "counters_location",
    "state_summary_location",
    "observable_summary_location",
    "scratch_location",
}
```

### No New Classes
All changes use existing data structures:
- `LoopBufferSettings` (attrs class)
- `ODELoopConfig` (attrs class with `buffer_settings` field)
- Standard Python dicts for settings

---

## Edge Cases

1. **User provides invalid location**: `LoopBufferSettings` validator raises `ValueError`
2. **User provides both flat kwargs and explicit loop_settings with same key**: Existing merge logic handles (kwargs win with warning)
3. **Update with unrecognized nested key**: Returns empty set, no error if `silent=True`
4. **None buffer_settings in ODELoopConfig**: Nested check skips None attributes
5. **User updates size-related setting that affects buffer_settings**: `SingleIntegratorRunCore.update()` rebuilds `LoopBufferSettings` preserving locations

---

## Testing Considerations

Tests should verify:
1. Solver accepts buffer location kwargs at instantiation
2. `solver.update()` recognizes buffer location parameters
3. Buffer locations propagate to `LoopBufferSettings`
4. Changed locations trigger cache invalidation
5. Nested update in CUDAFactory handles both dict and attrs attributes
6. Invalid locations raise appropriate errors
