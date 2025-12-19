# Buffer Settings Plumbing - Agent Plan

## Scope

This plan covers Task 2 of 3 - init and plumbing changes for buffer settings. This task modifies:
- `src/cubie/batchsolving/solver.py`
- `src/cubie/integrators/SingleIntegratorRunCore.py`
- `src/cubie/integrators/loops/ode_loop.py`
- `src/cubie/integrators/loops/ode_loop_config.py`

Do NOT modify:
- `src/cubie/buffer_registry.py` - Task 1
- Algorithm files (`src/cubie/integrators/algorithms/`) - Task 3
- Matrix-free solvers (`src/cubie/integrators/matrix_free_solvers/`) - Task 3
- Loop template/builder files beyond ode_loop.py and ode_loop_config.py

## User Stories Reference

See `human_overview.md` for:
- US-1: User-specified buffer locations via solve_ivp
- US-2: Integration with argument filtering system
- US-3: Compile settings as source of truth
- US-4: Removal of aliasing logic from CUDAFactory classes

---

## Component Specifications

### solver.py Changes

**Purpose:** Enable buffer location keywords to be passed through solve_ivp and Solver.

**Current State:**
- `merge_kwargs_into_settings` is called with `ALL_LOOP_SETTINGS` to filter loop-related kwargs
- Loop settings are passed to BatchSolverKernel

**Required Changes:**

1. **Import ALL_BUFFER_LOCATION_PARAMETERS**

Add import from SingleIntegratorRunCore:
```python
from cubie.integrators.SingleIntegratorRunCore import (
    ALL_BUFFER_LOCATION_PARAMETERS,
)
```

2. **Union buffer location parameters with loop settings in merge**

In `Solver.__init__`, modify the loop_settings merge to include buffer locations:
```python
loop_settings, loop_recognized = merge_kwargs_into_settings(
    kwargs=kwargs, 
    valid_keys=ALL_LOOP_SETTINGS | ALL_BUFFER_LOCATION_PARAMETERS,
    user_settings=loop_settings
)
```

3. **No changes needed to solve_ivp**

solve_ivp already passes **kwargs through to Solver, so buffer location kwargs will flow automatically.

**Expected Behavior:**
- `solve_ivp(system, y0, params, state_location='shared')` works
- `Solver(system, state_location='shared')` works
- `Solver(system, loop_settings={'state_location': 'shared'})` works
- All three approaches produce identical results

---

### SingleIntegratorRunCore.py Changes

**Purpose:** Pass buffer location settings through to IVPLoop and handle updates.

**Current State:**
- `ALL_BUFFER_LOCATION_PARAMETERS` is already defined
- `instantiate_loop` extracts buffer_location_kwargs from loop_settings
- update() passes updates to loop

**Required Changes:**

1. **Ensure buffer locations flow to IVPLoop.update()**

In the `update()` method, buffer location updates must reach the loop. Currently the code extracts buffer_location_kwargs only during instantiate_loop. The update path must also handle these.

Review the current update() implementation - it already calls `self._loop.update(updates_dict, silent=True)`. Since IVPLoop.update() will be modified to recognize these parameters (via compile settings), no additional changes are needed here UNLESS the loop rejects them.

The key is that buffer location params should be in updates_dict when passed to loop.update().

2. **No removal of aliasing logic**

SingleIntegratorRunCore does not contain aliasing logic - it passes locations to IVPLoop. The aliasing decision happens in buffer_registry (Task 1).

**Expected Behavior:**
- Buffer location kwargs passed to Solver reach IVPLoop.update() unchanged
- `solver.update(state_location='shared')` causes loop rebuild with new location

---

### ode_loop_config.py Changes

**Purpose:** Store buffer locations in compile settings for cache invalidation.

**Current State:**
- `ODELoopConfig` is an attrs class with various compile-critical settings
- Uses `@define` from attrs and `field()` for attributes
- Already has validation patterns using `validators.in_()` and custom validators
- Does NOT currently include buffer location fields

**Required Changes:**

1. **Add buffer location fields to ODELoopConfig**

Add the following fields to ODELoopConfig after the existing size/precision fields and before the device function fields:

```python
# Buffer location settings
state_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
state_proposal_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
parameters_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
drivers_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
drivers_proposal_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
observables_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
observables_proposal_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
error_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
counters_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
state_summary_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
observable_summary_location: str = field(
    default='local',
    validator=validators.in_(['shared', 'local'])
)
```

**Note on Defaults:**
Per requirements, defaults should only exist in compile_settings. The ODELoopConfig attrs class IS the compile settings for IVPLoop, so having defaults here is correct. The requirement "Do NOT have default arguments anywhere but compile_settings" means we should NOT have defaults in IVPLoop.__init__ signature.

**Expected Behavior:**
- ODELoopConfig stores all buffer locations
- Changing a location field on the config triggers cache invalidation
- All 11 location fields have 'local' as default

---

### ode_loop.py Changes

**Purpose:** Use compile settings for buffer locations during registration; remove any hardcoded defaults.

**Current State:**
- IVPLoop.__init__ has default values for all location parameters: `state_location: str = 'local'`
- Locations are passed directly to buffer_registry.register()
- No aliasing logic in IVPLoop (this is good - nothing to remove)

**Required Changes:**

1. **Modify IVPLoop.__init__ signature to use Optional[str] = None**

Change signature from:
```python
state_location: str = 'local',
state_proposal_location: str = 'local',
# ... etc
```

To:
```python
state_location: Optional[str] = None,
state_proposal_location: Optional[str] = None,
# ... etc
```

This allows the compile settings defaults to be the source of truth.

2. **Update buffer registration to use provided or default locations**

In __init__, after creating config, use the config values for buffer registration. The registration happens BEFORE compile_settings is set, so we need to handle the None case:

```python
# Determine effective location (use provided or default)
effective_state_location = state_location if state_location is not None else 'local'

# Register buffers with effective locations
buffer_registry.register(
    'loop_state', self, n_states, effective_state_location,
    precision=precision
)
```

The pattern repeats for all 11 buffers.

**Alternative approach (cleaner):** Create the ODELoopConfig FIRST with all location kwargs, then use config values for registration:

```python
# Build config kwargs
config_kwargs = dict(
    n_states=n_states,
    # ... other fields
)
# Only include location fields if provided
if state_location is not None:
    config_kwargs['state_location'] = state_location
# ... repeat for other locations

config = ODELoopConfig(**config_kwargs)

# Now register using config values (which have defaults)
buffer_registry.register(
    'loop_state', self, n_states, config.state_location,
    precision=precision
)
```

This is the preferred approach because:
- ODELoopConfig holds the single source of truth for defaults
- Registration uses validated config values
- No duplication of default values

3. **Reorder __init__ to create config before buffer registration**

Current order:
1. Clear factory buffers
2. Register buffers with passed locations
3. Create ODELoopConfig
4. setup_compile_settings(config)

New order:
1. Clear factory buffers
2. Create ODELoopConfig (with location kwargs if provided)
3. Register buffers using config.* location values
4. setup_compile_settings(config)

4. **Update IVPLoop.update() to handle location changes**

When a location changes via update(), we need to:
1. Update compile_settings (already handled by update_compile_settings)
2. Update the affected buffer's location in the registry

Add after the update_compile_settings call:
```python
# Update buffer locations in registry if any location changed
for param in ALL_BUFFER_LOCATION_PARAMETERS:
    if param in updates_dict:
        buffer_name = self._param_to_buffer_name(param)
        new_location = getattr(self.compile_settings, param)
        buffer_registry.update_buffer(buffer_name, self, location=new_location)
```

Where `_param_to_buffer_name` maps e.g. 'state_location' -> 'loop_state'.

Alternatively, import the mapping directly or define it as a module-level dict:
```python
LOCATION_PARAM_TO_BUFFER = {
    'state_location': 'loop_state',
    'state_proposal_location': 'loop_proposed_state',
    'parameters_location': 'loop_parameters',
    'drivers_location': 'loop_drivers',
    'drivers_proposal_location': 'loop_proposed_drivers',
    'observables_location': 'loop_observables',
    'observables_proposal_location': 'loop_proposed_observables',
    'error_location': 'loop_error',
    'counters_location': 'loop_counters',
    'state_summary_location': 'loop_state_summary',
    'observable_summary_location': 'loop_observable_summary',
}
```

5. **Import ALL_BUFFER_LOCATION_PARAMETERS**

Import from SingleIntegratorRunCore (or define locally if circular import issues):
```python
from cubie.integrators.SingleIntegratorRunCore import ALL_BUFFER_LOCATION_PARAMETERS
```

If circular import is an issue, define the set locally in ode_loop.py and ensure SingleIntegratorRunCore imports from there instead.

**Expected Behavior:**
- IVPLoop accepts Optional[str] = None for all location params
- If None, the ODELoopConfig default ('local') is used
- Compile settings store all location values
- update() with location changes triggers buffer update and cache invalidation

---

## Integration Points

### Existing System Dependencies

**buffer_registry (Task 1 dependency):**
- `buffer_registry.register(name, parent, size, location, ...)` signature unchanged
- `buffer_registry.update_buffer(name, parent, **kwargs)` used for location updates
- `buffer_registry.get_allocator(name, parent)` used in build() unchanged

**CUDAFactory base class:**
- `setup_compile_settings()` called with ODELoopConfig
- `update_compile_settings()` handles location field updates
- Cache invalidation triggered automatically on field changes

### Data Flow Summary

```
User: solve_ivp(system, y0, params, state_location='shared')
                          │
                          ▼
Solver.__init__: merge_kwargs_into_settings(kwargs, ALL_LOOP_SETTINGS | ALL_BUFFER_LOCATION_PARAMETERS)
                          │
                          ▼
loop_settings = {'state_location': 'shared', ...}
                          │
                          ▼
BatchSolverKernel.__init__: passes loop_settings to SingleIntegratorRun
                          │
                          ▼
SingleIntegratorRunCore.__init__: passes loop_settings to instantiate_loop()
                          │
                          ▼
instantiate_loop: extracts buffer_location_kwargs, passes to IVPLoop
                          │
                          ▼
IVPLoop.__init__: 
  1. Registers buffers with passed locations
  2. Creates ODELoopConfig with locations
  3. setup_compile_settings(config)
```

---

## Edge Cases

### 1. Mixed Explicit and Default Locations
User provides `state_location='shared'` but not other locations:
- Only state_location is 'shared'
- All others use ODELoopConfig defaults ('local')

### 2. Invalid Location Value
User provides `state_location='gpu'` (invalid):
- ODELoopConfig validator raises ValueError at config creation time
- Clear error message about valid values

### 3. Location Update Without Other Changes
User calls `solver.update(state_location='shared')`:
- update_compile_settings recognizes state_location
- Field is updated on ODELoopConfig
- Cache is invalidated
- buffer_registry.update_buffer is called
- Next device_function access triggers rebuild

### 4. Conflicting Settings Sources
User provides both `state_location='shared'` and `loop_settings={'state_location': 'local'}`:
- Per merge_kwargs_into_settings behavior, keyword arguments take precedence
- A warning is issued about duplicate settings
- Final value is 'shared'

---

## Implementation Order

1. **ode_loop_config.py** - Add location fields to ODELoopConfig
2. **ode_loop.py** - Modify __init__ to use Optional[str] = None, store in config, handle update()
3. **solver.py** - Add import and union of ALL_BUFFER_LOCATION_PARAMETERS
4. **Tests** - Add tests for buffer location flow

---

## Test Requirements

### New Tests Needed

**test_buffer_location_kwargs_recognized:**
```python
def test_buffer_location_kwargs_recognized():
    """Buffer location kwargs pass through Solver to loop."""
    solver = Solver(system, state_location='shared')
    assert solver.kernel.single_integrator._loop.compile_settings.state_location == 'shared'
```

**test_buffer_location_update:**
```python
def test_buffer_location_update():
    """Buffer location can be updated via solver.update()."""
    solver = Solver(system)
    solver.update(state_location='shared')
    assert solver.kernel.single_integrator._loop.compile_settings.state_location == 'shared'
```

**test_buffer_location_default:**
```python
def test_buffer_location_default():
    """Buffer locations default to 'local' when not specified."""
    solver = Solver(system)
    config = solver.kernel.single_integrator._loop.compile_settings
    assert config.state_location == 'local'
    assert config.parameters_location == 'local'
    # ... all 11 locations
```

**test_invalid_buffer_location_raises:**
```python
def test_invalid_buffer_location_raises():
    """Invalid buffer location values raise ValueError."""
    with pytest.raises(ValueError):
        Solver(system, state_location='gpu')
```

**test_buffer_location_in_loop_settings:**
```python
def test_buffer_location_in_loop_settings():
    """Buffer locations can be specified via loop_settings dict."""
    solver = Solver(system, loop_settings={'state_location': 'shared'})
    assert solver.kernel.single_integrator._loop.compile_settings.state_location == 'shared'
```

### Existing Tests to Verify

Check that existing IVPLoop tests pass with the new Optional parameters.

---

## Implementation Notes

### No Backwards Compatibility
Per project guidelines: "Never retain an obsolete feature or argument for API compatibility."

### No Optional Buffer Management
Per requirements: "Do NOT make any objects optional or provide fallbacks - buffer management is compulsory."

### Guarantee-by-Design
Per requirements: "Prefer guarantee-by-design instead of defensive guards for conditions that shouldn't be met."
- If a location value reaches buffer_registry.register(), it's already validated by ODELoopConfig
- No need to re-validate in register()

### Comment Style
Per project guidelines: Comments explain current behavior, not changes made.
- Bad: "now stores locations in compile settings instead of __init__ defaults"
- Good: "buffer locations stored in compile settings for cache invalidation"
