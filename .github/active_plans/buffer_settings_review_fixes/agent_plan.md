# BufferSettings Review Fixes - Agent Plan

## Component Changes

### 1. NewtonBufferSettings (newton_krylov.py)

#### Current State
- `residual_temp` is always local (hardcoded in local_memory_elements)
- `NewtonSliceIndices` does not include residual_temp slice
- `NewtonLocalSizes` includes residual_temp but no toggle

#### Required Changes

**NewtonSliceIndices** - Add residual_temp slice:
```
Attributes to add:
- residual_temp: slice - Slice covering residual_temp buffer (empty if local)
```

**NewtonBufferSettings** - Add residual_temp_location:
```
New attribute:
- residual_temp_location: str - 'local' or 'shared', default 'local'

New property:
- use_shared_residual_temp: bool - Return True if residual_temp uses shared

Update shared_memory_elements:
- If use_shared_residual_temp: add self.n to total

Update local_memory_elements:
- Replace hardcoded `total += self.n` with conditional based on location

Update shared_indices:
- Add residual_temp_slice calculation based on location
- Return NewtonSliceIndices with residual_temp parameter
```

### 2. DIRKBufferSettings (generic_dirk.py)

#### Current State
- `newton_buffer_settings: Optional[NewtonBufferSettings] = None`
- `solver_scratch_location` toggle exists
- `solver_scratch_elements` has fallback to `2 * self.n`
- Guards check if newton_buffer_settings is None

#### Required Changes

**Remove solver_scratch_location**:
```
Remove attribute:
- solver_scratch_location: str

Remove from ALL_DIRK_BUFFER_LOCATION_PARAMETERS:
- "solver_scratch_location"

Remove property:
- use_shared_solver_scratch

Remove from shared_memory_elements calculation:
- The conditional for solver_scratch - children handle this

Remove from local_memory_elements calculation:
- The conditional for solver_scratch

Remove from shared_indices:
- solver_scratch_slice - children handle their own indices

Update local_sizes:
- solver_scratch uses newton_buffer_settings values
```

**Make newton_buffer_settings required with factory default**:
```
Change attribute:
- newton_buffer_settings: NewtonBufferSettings = attrs.field(factory=...)

Factory creates NewtonBufferSettings with n from self.n
Note: attrs factories are tricky - may need post-init hook or property

Remove Optional typing and validator

Remove None checks in:
- solver_scratch_elements property
- Any other locations
```

**Update solver_scratch_elements**:
```
Remove fallback logic:
- Remove: if self.newton_buffer_settings is not None:
- Return self.newton_buffer_settings.shared_memory_elements directly
```

### 3. FIRKBufferSettings (generic_firk.py)

#### Current State
- Same issues as DIRKBufferSettings
- `newton_buffer_settings: Optional[NewtonBufferSettings] = None`
- `solver_scratch_location` toggle exists
- Fallback to `2 * self.all_stages_n`

#### Required Changes
Same pattern as DIRKBufferSettings:
- Remove solver_scratch_location
- Make newton_buffer_settings required with factory default
- Remove None checks and fallbacks

### 4. DIRKStep.__init__ (generic_dirk.py)

#### Current State
- Imports LinearSolverBufferSettings and NewtonBufferSettings inside __init__
- Creates buffer settings inline

#### Required Changes
```
Move imports to module header:
- from cubie.integrators.matrix_free_solvers.linear_solver import LinearSolverBufferSettings
- from cubie.integrators.matrix_free_solvers.newton_krylov import NewtonBufferSettings

Remove from __init__ body:
- The import statements on lines 491-496
```

### 5. FIRKStep.__init__ (generic_firk.py)

#### Current State
- Same issue - imports inside __init__

#### Required Changes
Same pattern:
- Move imports to module header
- Remove from __init__ body

### 6. DIRKStepConfig (generic_dirk.py)

#### Required New Properties
```
@property
def newton_buffer_settings(self) -> NewtonBufferSettings:
    """Return newton_buffer_settings from buffer_settings."""
    return self.buffer_settings.newton_buffer_settings

@property
def linear_solver_buffer_settings(self) -> LinearSolverBufferSettings:
    """Return linear_solver_buffer_settings from newton_buffer_settings."""
    return self.buffer_settings.newton_buffer_settings.linear_solver_buffer_settings
```

### 7. FIRKStepConfig (generic_firk.py)

#### Required New Properties
Same as DIRKStepConfig:
- newton_buffer_settings property
- linear_solver_buffer_settings property

### 8. DIRKStep.build_implicit_helpers (generic_dirk.py)

#### Current State
Lines 594-600 have convoluted logic:
```python
newton_buffer_settings = config.buffer_settings.newton_buffer_settings
linear_buffer_settings = None
if newton_buffer_settings is not None:
    linear_buffer_settings = (
        newton_buffer_settings.linear_solver_buffer_settings
    )
```

#### Required Changes
```python
newton_buffer_settings = config.newton_buffer_settings
linear_buffer_settings = config.linear_solver_buffer_settings
```

### 9. FIRKStep.build_implicit_helpers (generic_firk.py)

#### Current State
Lines 576-582 have same pattern with None check

#### Required Changes
Same simplification using direct properties

---

## Test Changes

### test_newton_buffer_settings.py

**Remove tests for defaults that may change**:
- Line 16-19: test_shared_memory_elements_default
- Line 36-40: test_local_memory_elements_default

**Remove pointless tests**:
- Lines 52-57: test_shared_indices_contiguous (tests implementation detail)
- Lines 81-85: test_lin_solver_start_matches_local_end

**Fix test at line 60-68**:
- Test shared_memory_elements property instead of end index

### test_buffer_settings.py (algorithms)

**Remove test for optional behavior**:
- Lines 278+ that test fallback behavior

### test_buffer_settings.py (loops)

**Remove easily invalidated test**:
- Line 130: test that assumes specific array default locations

### All Test Files

**Move imports to module header**:
- Any imports inside test function bodies should be at file top

---

## Integration Points

### Affected Files Summary

Source files:
1. `src/cubie/integrators/matrix_free_solvers/newton_krylov.py`
   - NewtonSliceIndices: add residual_temp slice
   - NewtonBufferSettings: add residual_temp_location, update calculations
   
2. `src/cubie/integrators/algorithms/generic_dirk.py`
   - Move imports to header
   - DIRKBufferSettings: remove solver_scratch_location, make newton_buffer_settings required
   - DIRKStepConfig: add properties
   - DIRKStep.build_implicit_helpers: simplify using properties
   
3. `src/cubie/integrators/algorithms/generic_firk.py`
   - Same pattern as generic_dirk.py

Test files:
1. `tests/integrators/matrix_free_solvers/test_newton_buffer_settings.py`
2. `tests/integrators/algorithms/test_buffer_settings.py`
3. `tests/integrators/loops/test_buffer_settings.py`
4. `tests/test_buffer_settings.py`

---

## Dependencies and Order

### Implementation Order

1. **First**: Update NewtonBufferSettings with residual_temp toggleability
   - This is a foundational change that other components depend on
   
2. **Second**: Update DIRKBufferSettings
   - Remove solver_scratch_location
   - Make newton_buffer_settings required (using factory)
   - This depends on NewtonBufferSettings being complete
   
3. **Third**: Update FIRKBufferSettings
   - Same changes as DIRKBufferSettings
   
4. **Fourth**: Move imports to module headers
   - DIRKStep and FIRKStep __init__ methods
   
5. **Fifth**: Add properties to config classes
   - DIRKStepConfig and FIRKStepConfig
   
6. **Sixth**: Simplify build_implicit_helpers methods
   - Use new properties instead of drilling through attributes
   
7. **Last**: Update tests
   - Remove/modify tests as specified
   - Move imports to module headers

---

## Edge Cases

### Factory Default for newton_buffer_settings

The factory needs access to `self.n` which isn't available at attrs definition time. Solutions:

1. **Post-init hook**: Use `__attrs_post_init__` to set newton_buffer_settings if not provided
2. **Converter function**: Convert None to default NewtonBufferSettings
3. **Property with lazy init**: Property creates on first access (not recommended for attrs)

**Recommended**: Use converter function:
```python
def _default_newton_settings(value, n):
    if value is None:
        from ... import LinearSolverBufferSettings, NewtonBufferSettings
        linear = LinearSolverBufferSettings(n=n)
        return NewtonBufferSettings(n=n, linear_solver_buffer_settings=linear)
    return value
```

Or use `__attrs_post_init__`:
```python
def __attrs_post_init__(self):
    if self.newton_buffer_settings is None:
        object.__setattr__(self, 'newton_buffer_settings', 
                          NewtonBufferSettings(n=self.n, ...))
```

### DIRK vs FIRK n value

DIRK uses `n` for Newton solver (solves each stage independently).
FIRK uses `all_stages_n = stage_count * n` (solves coupled system).

The factory default must use the correct n value for each algorithm type.

### Shared Memory Calculation Without solver_scratch_location

When solver_scratch_location is removed, the shared_memory_elements calculation for DIRK/FIRK should:
- NOT include solver_scratch directly
- The solver_scratch region is implicitly shared (passed from loop)
- Child newton_buffer_settings.shared_memory_elements provides the requirement

This means the "scratch" slice in shared memory comes from the loop level, and Newton/linear solver use their portion of it.
