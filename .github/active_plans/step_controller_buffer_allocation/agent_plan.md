# Agent Plan: Step Controller Buffer Allocation Refactor

## Purpose
This document provides detailed architectural specifications for refactoring step controllers to use the centralized buffer registry pattern. The detailed_implementer will use this to create function-level implementation tasks.

---

## Component Descriptions

### 1. BaseStepControllerConfig (Enhanced)
**Location**: `src/cubie/integrators/step_control/base_step_controller.py`

**Current State**: 
- Contains `precision` and `n` fields
- Defines abstract properties for dt_min, dt_max, dt0, is_adaptive

**Target State**:
- Add `timestep_memory` field with default='local'
- Add validator ensuring value is 'local' or 'shared'
- Add to `settings_dict` property

**Expected Behavior**:
- When `timestep_memory='shared'`, controller buffers are allocated from shared memory
- When `timestep_memory='local'`, controller buffers use local memory (cuda.local.array or persistent_local)
- Changing the value triggers cache invalidation (handled by CUDAFactory)

### 2. BaseStepController (Enhanced)
**Location**: `src/cubie/integrators/step_control/base_step_controller.py`

**Current State**:
- Abstract base class for controllers
- Has `local_memory_elements` abstract property
- Has `update()` method for compile settings

**Target State**:
- Add `register_buffers()` method that registers with buffer_registry
- Method uses `self.local_memory_elements` for size
- Method uses `self.compile_settings.timestep_memory` for location
- Buffer named 'timestep_buffer'
- Call `register_buffers()` after `setup_compile_settings()` in subclass __init__

**Expected Behavior**:
- After init, the controller's buffer is registered with the global buffer_registry
- `buffer_registry.get_allocator('timestep_buffer', self)` returns a valid allocator
- Size correctly reflects controller type (0 for I, 1 for PI, 2 for PID/Gustafsson)

### 3. AdaptiveStepControlConfig (Enhanced)
**Location**: `src/cubie/integrators/step_control/adaptive_step_controller.py`

**Current State**:
- Extends BaseStepControllerConfig with tolerance fields
- Has settings_dict property

**Target State**:
- No additional changes needed if base class has timestep_memory
- Verify `timestep_memory` propagates through inheritance

### 4. FixedStepControlConfig (Enhanced)
**Location**: `src/cubie/integrators/step_control/fixed_step_controller.py`

**Current State**:
- Simple config with just `dt` field
- Extends BaseStepControllerConfig

**Target State**:
- Inherits `timestep_memory` from base class
- No additional changes needed

### 5. Controller Device Functions (All Controllers)
**Locations**: 
- `adaptive_I_controller.py`
- `adaptive_PI_controller.py` 
- `adaptive_PID_controller.py`
- `gustafsson_controller.py`
- `fixed_step_controller.py`

**Current Signature**:
```python
def controller_fn(dt, state, state_prev, error, niters, accept_out, local_temp):
```

**Target Signature**:
```python
def controller_fn(dt, state, state_prev, error, niters, accept_out, 
                  shared_scratch, persistent_local):
```

**Expected Behavior**:
- Device function receives shared and persistent arrays from caller
- Uses `alloc_timestep_buffer(shared_scratch, persistent_local)` to get buffer
- For controllers with `local_memory_elements == 0`, allocator still works but returns minimal buffer
- Internal logic remains unchanged after allocator call

### 6. Controller build() Methods (All Controllers)
**Current State**:
- Compile device function with `local_temp` parameter
- Return ControllerCache

**Target State**:
- Get allocator from buffer_registry: `buffer_registry.get_allocator('timestep_buffer', self)`
- Compile device function with allocator captured in closure
- Device function uses allocator to get buffer from passed arrays

### 7. IVPLoop (Modified)
**Location**: `src/cubie/integrators/loops/ode_loop.py`

**Current State**:
- Uses `alloc_controller_persistent` from registry
- Calls controller with: `step_controller(dt, state_proposal_buffer, ..., controller_temp)`

**Target State**:
- Uses `alloc_controller_shared` and `alloc_controller_persistent` from registry (already exists)
- Calls controller with: `step_controller(dt, state_proposal_buffer, ..., ctrl_shared, ctrl_persistent)`
- The two arrays replace the single `controller_temp` argument

**Expected Behavior**:
- Controller receives both shared and persistent slices
- Controller's internal allocator selects correct memory region
- No change to overall loop logic

### 8. ALL_STEP_CONTROLLER_PARAMETERS (Enhanced)
**Location**: `src/cubie/integrators/step_control/base_step_controller.py`

**Current State**:
```python
ALL_STEP_CONTROLLER_PARAMETERS = {
    'precision', 'n', 'step_controller', 'dt',
    'dt_min', 'dt_max', 'atol', 'rtol', 'algorithm_order',
    'min_gain', 'max_gain', 'safety',
    'kp', 'ki', 'kd', 'deadband_min', 'deadband_max',
    'gamma', 'max_newton_iters'
}
```

**Target State**:
```python
ALL_STEP_CONTROLLER_PARAMETERS = {
    'precision', 'n', 'step_controller', 'dt',
    'dt_min', 'dt_max', 'atol', 'rtol', 'algorithm_order',
    'min_gain', 'max_gain', 'safety',
    'kp', 'ki', 'kd', 'deadband_min', 'deadband_max',
    'gamma', 'max_newton_iters',
    'timestep_memory'  # Added
}
```

---

## Integration Points

### 1. SingleIntegratorRunCore → StepController
**Current**: Creates controller via `get_controller()`, calls `get_child_allocators()`
**Target**: No change needed - already calls `get_child_allocators(loop, controller, 'controller')`

The key is that `get_child_allocators` computes sizes from:
- `buffer_registry.shared_buffer_size(child)` 
- `buffer_registry.persistent_local_buffer_size(child)`

These work if the controller has called `register_buffers()`.

### 2. StepController → buffer_registry
**Registration**: Controller calls `buffer_registry.register()` in `register_buffers()`
**Allocation**: Controller's `build()` gets allocator via `buffer_registry.get_allocator()`

### 3. IVPLoop → StepController Device Function
**Current Call**: `step_controller(..., controller_temp)`
**Target Call**: `step_controller(..., ctrl_shared, ctrl_persistent)`

The loop already allocates these via `get_child_allocators` pattern.

---

## Data Structures

### Buffer Registration Entry
```python
# In buffer_registry after controller.register_buffers():
CUDABuffer(
    name='timestep_buffer',
    size=2,  # varies by controller type
    location='local',  # or 'shared' based on config
    persistent=False,  # local memory for scratch
    aliases=None,
    precision=np.float64  # from controller precision
)
```

### Config Field Addition
```python
# In BaseStepControllerConfig:
timestep_memory: str = field(
    default='local',
    validator=validators.in_(["local", "shared"])
)
```

---

## Dependencies

### Import Requirements
Controllers will need:
```python
from cubie.buffer_registry import buffer_registry
```

### Existing Dependencies (No Changes)
- `cubie.CUDAFactory` - base class pattern
- `numba.cuda` - device function compilation
- `attrs` - config classes

---

## Edge Cases

### 1. Zero-Size Buffers (I Controller)
The I controller has `local_memory_elements = 0`. The buffer registry handles this:
- Register with size=0
- `cuda.local.array(max(size, 1), ...)` ensures minimum size 1
- Controller doesn't use the buffer, so no functional impact

### 2. Controller Switching at Runtime
When algorithm changes and controller is swapped:
1. New controller created via `get_controller()`
2. New controller calls `register_buffers()` in __init__
3. `SingleIntegratorRunCore` calls `get_child_allocators()` with new controller
4. Loop rebuilds with new allocators

### 3. Shared Memory Pressure
If `timestep_memory='shared'` with large systems:
- 2 elements * n_controllers is negligible (max 2 floats)
- No special handling needed

### 4. CUDASIM Mode
The allocator pattern works identically in CUDASIM:
- `cuda.local.array()` returns numpy array
- Shared slices work on numpy arrays
- No special handling needed

---

## Call Signature Changes Summary

### Before (Current)
```python
# IVPLoop.build() - line ~642
controller_status = step_controller(
    dt,
    state_proposal_buffer,
    state_buffer,
    error,
    niters,
    accept_step,
    controller_temp,
)
```

### After (Target)
```python
# IVPLoop.build() - modified
controller_status = step_controller(
    dt,
    state_proposal_buffer,
    state_buffer,
    error,
    niters,
    accept_step,
    ctrl_shared,
    ctrl_persistent,
)
```

### Controller Device Function
```python
# Before
def controller_I(dt, state, state_prev, error, niters, accept_out, local_temp):
    # Uses local_temp directly
    
# After
def controller_I(dt, state, state_prev, error, niters, accept_out, 
                 shared_scratch, persistent_local):
    timestep_buffer = alloc_timestep_buffer(shared_scratch, persistent_local)
    # Uses timestep_buffer (same as before, just allocated differently)
```

---

## Testing Considerations

### Existing Tests to Verify
- Controller instantiation tests should still pass
- Integration tests with IVPLoop should still pass
- Buffer size calculations should match current behavior

### New Test Cases Needed
- Verify `timestep_memory='shared'` allocates from shared memory
- Verify `timestep_memory` appears in `settings_dict`
- Verify buffer registration happens during __init__
- Verify allocator returns correct size buffer

---

## Implementation Order

The detailed_implementer should create tasks in this order:

1. **Add timestep_memory to config classes** (base → children)
2. **Add register_buffers to BaseStepController**
3. **Update ALL_STEP_CONTROLLER_PARAMETERS**
4. **Update each controller's build() method** (add allocator, change signature)
5. **Update IVPLoop controller call** (pass two arrays instead of one)
6. **Verify existing tests pass**
7. **Add new tests for the feature**
