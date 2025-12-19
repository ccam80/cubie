# Buffer Settings Plumbing - Agent Plan

## Scope

This plan covers buffer settings plumbing for **ALL CUDAFactory subclasses**:

### Source Files to Modify

**Core Infrastructure:**
- `src/cubie/buffer_registry.py` - Add `update()` method
- `src/cubie/integrators/loops/ode_loop.py` - Location fields in compile settings
- `src/cubie/integrators/loops/ode_loop_config.py` - Add location fields to ODELoopConfig

**Algorithm Files:**
- `src/cubie/integrators/algorithms/backwards_euler.py`
- `src/cubie/integrators/algorithms/backwards_euler_predict_correct.py`
- `src/cubie/integrators/algorithms/crank_nicolson.py`
- `src/cubie/integrators/algorithms/explicit_euler.py`
- `src/cubie/integrators/algorithms/generic_dirk.py`
- `src/cubie/integrators/algorithms/generic_erk.py`
- `src/cubie/integrators/algorithms/generic_firk.py`
- `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
- `src/cubie/integrators/algorithms/ode_explicitstep.py`
- `src/cubie/integrators/algorithms/ode_implicitstep.py`

**Matrix-Free Solvers:**
- `src/cubie/integrators/matrix_free_solvers/linear_solver.py`
- `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`

**Batch Solving:**
- `src/cubie/batchsolving/BatchSolverKernel.py`

**Instrumented Test Files (must mirror source changes):**
- `tests/integrators/algorithms/instrumented/backwards_euler.py`
- `tests/integrators/algorithms/instrumented/backwards_euler_predict_correct.py`
- `tests/integrators/algorithms/instrumented/crank_nicolson.py`
- `tests/integrators/algorithms/instrumented/explicit_euler.py`
- `tests/integrators/algorithms/instrumented/generic_dirk.py`
- `tests/integrators/algorithms/instrumented/generic_erk.py`
- `tests/integrators/algorithms/instrumented/generic_firk.py`
- `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
- `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`

## User Stories Reference

See `human_overview.md` for:
- US-1: User-specified buffer locations via solve_ivp
- US-2: Integration with argument filtering system
- US-3: Each CUDAFactory owns its buffers
- US-4: Unified update pattern via BufferRegistry.update()

---

## CRITICAL Design Requirements

**From user feedback - these are non-negotiable:**

1. **Each CUDAFactory owns its own buffers and their locations**
   - The registry holds per-CUDAFactory information
   - Only loop-assigned buffers have keywords at IVPLoop level
   - Algorithm buffers have keywords at algorithm level

2. **Do NOT create separate buffer-location keyword arg dicts**
   - Join buffer location kwargs to each factory's existing arguments
   - Use the same filtering used for other arguments

3. **Do NOT clutter __init__ with "if location is not None:" noise**
   - Use existing `split_applicable_settings()` and `merge_kwargs_into_settings()` from `cubie/_utils.py`

4. **Do NOT have a separate location-param-to-buffer mapping dict**
   - Instead, add an `update()` method to BufferRegistry that:
     - Finds CUDABuffer objects where keyword is `[buffer.name]_location`
     - Updates the buffer's location
     - Invalidates that context's cache
     - Returns the keyword as recognized
   - Copy existing `update()` and `update_compile_settings()` logic exactly

5. **OVERALL PRINCIPLE**: Buffer location parameters are NOT separate from other compile settings
   - They are the same as `dt_save`, `tolerance`, etc.
   - Ownership, init, and update should match exactly

---

## Component Specifications

### 1. buffer_registry.py - Add update() Method

**Purpose:** Provide unified location update mechanism matching `update_compile_settings()` pattern

**Required Changes:**

Add the following method to `BufferRegistry` class:

```python
def update(
    self,
    factory: object,
    updates_dict: Optional[Dict[str, Any]] = None,
    silent: bool = False,
    **kwargs: Any,
) -> Set[str]:
    """Update buffer locations from keyword arguments.
    
    For each key of the form '[buffer_name]_location', finds the
    corresponding buffer and updates its location. Mirrors the pattern
    of CUDAFactory.update_compile_settings().
    
    Parameters
    ----------
    factory
        Factory instance that owns the buffers to update.
    updates_dict
        Mapping of parameter names to new values.
    silent
        Suppress errors for unrecognized parameters.
    **kwargs
        Additional parameters merged into updates_dict.
    
    Returns
    -------
    Set[str]
        Names of parameters that were successfully recognized and updated.
    
    Raises
    ------
    KeyError
        If unrecognized parameters are supplied and silent is False.
    
    Notes
    -----
    A parameter is recognized if it matches the pattern '[buffer_name]_location'
    where buffer_name is a registered buffer for the factory. When a location
    is updated, the factory's context cache is invalidated.
    """
    if updates_dict is None:
        updates_dict = {}
    updates_dict = updates_dict.copy()
    if kwargs:
        updates_dict.update(kwargs)
    if updates_dict == {}:
        return set()
    
    if factory not in self._contexts:
        if not silent:
            raise KeyError(f"Factory {factory} has no registered buffers.")
        return set()
    
    context = self._contexts[factory]
    recognized = set()
    updated = False
    
    for key, value in updates_dict.items():
        # Check if key matches pattern [buffer_name]_location
        if not key.endswith('_location'):
            continue
        
        buffer_name = key[:-9]  # Remove '_location' suffix
        if buffer_name not in context.entries:
            continue
        
        # Validate location value
        if value not in ('shared', 'local'):
            raise ValueError(
                f"Invalid location '{value}' for buffer '{buffer_name}'. "
                "Must be 'shared' or 'local'."
            )
        
        entry = context.entries[buffer_name]
        if entry.location != value:
            # Update the buffer entry with new location
            self.update_buffer(buffer_name, factory, location=value)
            updated = True
        
        recognized.add(key)
    
    if updated:
        context.invalidate_layouts()
    
    unrecognized = set(updates_dict.keys()) - recognized
    if unrecognized and not silent:
        raise KeyError(
            f"Unrecognized buffer location parameters: {unrecognized}"
        )
    
    return recognized
```

**Key Points:**
- Pattern matches `update_compile_settings()` signature and behavior
- Automatically derives buffer name from parameter name
- Validates location values
- Invalidates cache when locations change
- Returns recognized parameters like other update methods

---

### 2. newton_krylov.py - Location Parameters in Factory

**Current State:**
Location parameters are accepted in function signature:
```python
def newton_krylov_solver_factory(
    ...
    delta_location: str = 'local',
    residual_location: str = 'local',
    residual_temp_location: str = 'local',
    stage_base_bt_location: str = 'local',
) -> Callable:
```

**Required Changes:**

1. The factory already accepts location parameters - this is correct.

2. Ensure the owning factory (the algorithm that calls this) stores locations in compile settings:
   - Add location fields to `ImplicitStepConfig` in `ode_implicitstep.py`
   - Pass locations from compile settings to newton_krylov_solver_factory

3. When the algorithm's `update()` is called with location parameters:
   - `update_compile_settings()` recognizes and updates location fields
   - Call `buffer_registry.update(self, updates_dict)` to update buffer locations

---

### 3. linear_solver.py - Location Parameters in Factory

**Current State:**
Similar to newton_krylov.py, location parameters in function signature.

**Required Changes:**

1. Add location fields to linear solver config if not present
2. Ensure locations flow from algorithm compile settings to solver factory
3. Location updates follow same pattern as newton_krylov.py

---

### 4. ode_implicitstep.py - Base Implicit Step

**Purpose:** Base class for all implicit algorithm steps

**Required Changes:**

1. Add location fields to `ImplicitStepConfig` attrs class:
```python
@attrs.define
class ImplicitStepConfig:
    # ... existing fields ...
    
    # Newton-Krylov buffer locations
    newton_delta_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    newton_residual_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    newton_residual_temp_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    newton_stage_base_bt_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    
    # Linear solver buffer locations
    krylov_delta_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    # ... other linear solver buffers
```

2. Update `build()` method to pass locations from compile settings to solver factories

3. Override `update()` method to call `buffer_registry.update()`:
```python
def update(
    self,
    updates_dict: Optional[dict] = None,
    silent: bool = False,
    **kwargs
) -> Set[str]:
    if updates_dict is None:
        updates_dict = {}
    updates_dict = updates_dict.copy()
    if kwargs:
        updates_dict.update(kwargs)
    
    # Update compile settings (handles location fields)
    recognized = self.update_compile_settings(updates_dict, silent=True)
    
    # Update buffer locations in registry
    recognized |= buffer_registry.update(self, updates_dict, silent=True)
    
    unrecognized = set(updates_dict.keys()) - recognized
    if unrecognized and not silent:
        raise KeyError(f"Unrecognized parameters: {unrecognized}")
    
    return recognized
```

---

### 5. ode_explicitstep.py - Base Explicit Step

**Purpose:** Base class for all explicit algorithm steps

**Required Changes:**

1. Add any explicit-step-specific buffer location fields to `ExplicitStepConfig`
   (may be minimal - explicit steps typically don't allocate buffers)

2. If buffers exist, follow same pattern as implicit step for update()

---

### 6. Algorithm Step Classes (10 files)

For each algorithm step class:

**Files:**
- `backwards_euler.py`
- `backwards_euler_predict_correct.py`
- `crank_nicolson.py`
- `explicit_euler.py`
- `generic_dirk.py`
- `generic_erk.py`
- `generic_firk.py`
- `generic_rosenbrock_w.py`

**Required Changes:**

1. If algorithm has specific buffers beyond base class:
   - Add location fields to the algorithm's config section
   - Register buffers with locations from config

2. Ensure `build()` method passes locations from compile settings

3. Inherit update() from base class (ode_implicitstep or ode_explicitstep)

---

### 7. ode_loop_config.py - Add Location Fields

**Purpose:** Store loop buffer locations in compile settings for cache invalidation

**Required Changes:**

Add buffer location fields to `ODELoopConfig`:

```python
@attrs.define
class ODELoopConfig:
    # ... existing fields ...
    
    # Buffer location settings
    state_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    state_proposal_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    parameters_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    drivers_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    drivers_proposal_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    observables_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    observables_proposal_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    error_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    counters_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    state_summary_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
    observable_summary_location: str = attrs.field(
        default='local',
        validator=validators.in_(['shared', 'local'])
    )
```

---

### 8. ode_loop.py - Use Compile Settings Locations

**Purpose:** Use location values from compile settings during buffer registration

**Required Changes:**

1. In `__init__`, create config with location kwargs if provided, then use config values for registration:

```python
def __init__(
    self,
    precision: PrecisionDType,
    n_states: int,
    ...,
    state_location: str = 'local',
    state_proposal_location: str = 'local',
    # ... other location params with defaults matching ODELoopConfig
):
    super().__init__()
    
    # Build config with all settings including locations
    config = ODELoopConfig(
        n_states=n_states,
        ...,
        state_location=state_location,
        state_proposal_location=state_proposal_location,
        # ... other locations
    )
    
    # Register buffers using config values
    buffer_registry.clear_factory(self)
    buffer_registry.register(
        'loop_state', self, n_states, config.state_location,
        precision=precision
    )
    # ... other registrations using config.* locations
    
    self.setup_compile_settings(config)
```

2. Update `update()` method to call `buffer_registry.update()`:

```python
def update(
    self,
    updates_dict: Optional[dict[str, object]] = None,
    silent: bool = False,
    **kwargs: object,
) -> Set[str]:
    if updates_dict is None:
        updates_dict = {}
    updates_dict = updates_dict.copy()
    if kwargs:
        updates_dict.update(kwargs)
    if updates_dict == {}:
        return set()

    updates_dict, unpacked_keys = unpack_dict_values(updates_dict)

    # Update compile settings
    recognised = self.update_compile_settings(updates_dict, silent=True)
    
    # Update buffer locations in registry
    recognised |= buffer_registry.update(self, updates_dict, silent=True)
    
    unrecognised = set(updates_dict.keys()) - recognised
    if not silent and unrecognised:
        raise KeyError(
            f"Unrecognized parameters in update: {unrecognised}."
        )
    
    return recognised | unpacked_keys
```

---

### 9. Instrumented Test Files (9 files)

**CRITICAL**: Instrumented test files must mirror ALL changes made to source files.

For each instrumented file:
1. Apply identical signature changes to location parameters
2. Apply identical config/compile settings changes
3. Apply identical update() method changes if any

The only difference is instrumented files add logging arrays.

---

## Integration Points

### Existing System Dependencies

**buffer_registry:**
- `register(name, factory, size, location, ...)` - unchanged
- `update_buffer(name, factory, **kwargs)` - already exists
- `clear_factory(factory)` - unchanged
- NEW: `update(factory, updates_dict, silent, **kwargs)` - to be added

**CUDAFactory base class:**
- `setup_compile_settings()` - unchanged
- `update_compile_settings()` - unchanged
- Pattern: location fields in compile_settings trigger cache invalidation

**Utility functions (`cubie/_utils.py`):**
- `split_applicable_settings()` - use for filtering kwargs
- `merge_kwargs_into_settings()` - use for merging user settings
- `unpack_dict_values()` - use for nested settings dicts

---

## Edge Cases

### 1. Mixed Factory Location Updates
User calls `solver.update(state_location='shared', newton_delta_location='local')`:
- `state_location` is recognized by IVPLoop
- `newton_delta_location` is recognized by implicit algorithm
- Both factories update their respective compile settings and registry entries

### 2. Invalid Location Value
User provides `state_location='gpu'`:
- ODELoopConfig validator raises ValueError at config creation
- Clear error: "Invalid value 'gpu' for state_location"

### 3. Unregistered Buffer Name
User provides `nonexistent_buffer_location='shared'`:
- `buffer_registry.update()` returns empty set
- `update_compile_settings()` returns empty set
- If not silent, KeyError raised for unrecognized parameter

### 4. Location Update on Uninitialized Factory
Factory created but buffers not yet registered:
- `buffer_registry.update()` silently returns empty set
- No error unless user requests non-silent mode

---

## Implementation Order

1. **buffer_registry.py** - Add `update()` method first (foundation)
2. **ode_loop_config.py** - Add location fields to ODELoopConfig
3. **ode_loop.py** - Update to use config locations and call registry.update()
4. **ode_implicitstep.py** - Add location fields and update pattern
5. **ode_explicitstep.py** - Add location fields if needed
6. **matrix_free_solvers/** - Ensure locations flow from caller
7. **Algorithm files** - Verify locations passed correctly
8. **Instrumented files** - Mirror all changes
9. **Tests** - Add tests for location parameter flow

---

## Test Requirements

### New Tests Needed

**test_buffer_location_kwargs_recognized:**
```python
def test_buffer_location_kwargs_recognized():
    """Buffer location kwargs flow through to compile settings."""
    solver = Solver(system, state_location='shared')
    loop = solver.kernel.single_integrator._loop
    assert loop.compile_settings.state_location == 'shared'
```

**test_buffer_registry_update:**
```python
def test_buffer_registry_update():
    """BufferRegistry.update() recognizes location parameters."""
    # Setup factory with registered buffers
    recognized = buffer_registry.update(
        factory, 
        {'loop_state_location': 'shared'}
    )
    assert 'loop_state_location' in recognized
```

**test_algorithm_location_update:**
```python
def test_algorithm_location_update():
    """Algorithm buffer locations update via solver.update()."""
    solver = Solver(system, algorithm='backwards_euler')
    solver.update(newton_delta_location='shared')
    # Verify buffer was updated
```

**test_invalid_location_raises:**
```python
def test_invalid_location_raises():
    """Invalid location values raise ValueError."""
    with pytest.raises(ValueError):
        Solver(system, state_location='gpu')
```

---

## Comment Style Reminder

Per project guidelines:
- Describe functionality and current behavior, NOT implementation changes
- Bad: "now uses compile settings instead of hardcoded defaults"
- Good: "buffer locations stored in compile settings for cache invalidation"
