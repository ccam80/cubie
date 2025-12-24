# Fix Refactor Test Failures: Agent Plan

## Component Overview

This plan addresses three distinct bugs introduced by the buffer-settings-system refactor.

---

## Bug 1: Buffer Field Name Mismatch

### Affected Components

**Primary File:** `src/cubie/integrators/loops/ode_loop.py`

**Affected Class:** `IVPLoop`

### Expected Behavior

When `IVPLoop.__init__()` is called with `state_summaries_buffer_height` and `observable_summaries_buffer_height` parameters, these values should be correctly captured in `ODELoopConfig` and subsequently used by `register_buffers()` to register properly-sized buffers.

### Current Behavior

The `build_config()` call in `IVPLoop.__init__()` uses incorrect parameter names:
- Uses `state_summary_buffer_height` (singular 'summary')
- Should use `state_summaries_buffer_height` (plural 'summaries')

`build_config()` silently drops unrecognized parameters, so:
1. The passed values are ignored
2. `ODELoopConfig` uses default values (0)
3. `register_buffers()` registers buffers with size=0
4. Summary metrics fail with IndexError when accessing empty arrays

### Fix Location

`src/cubie/integrators/loops/ode_loop.py`, lines 197-199, in `IVPLoop.__init__()`

### Changes Required

Replace in the `required` dict passed to `build_config()`:
- `'state_summary_buffer_height'` → `'state_summaries_buffer_height'`
- `'observable_summary_buffer_height'` → `'observable_summaries_buffer_height'`

### Integration Points

- `ODELoopConfig` fields: `state_summaries_buffer_height`, `observable_summaries_buffer_height`
- `register_buffers()` method reads these from `config.state_summaries_buffer_height`
- `buffer_registry.register()` receives these sizes for 'state_summary' and 'observable_summary' buffers

---

## Bug 2: Tolerance Array Shape Mismatch During Controller Switching

### Affected Components

**Primary File:** `src/cubie/integrators/SingleIntegratorRunCore.py`

**Affected Method:** `_switch_controllers()`

**Secondary File:** `src/cubie/integrators/step_control/adaptive_step_controller.py`

**Affected Function:** `tol_converter()`

### Expected Behavior

When switching from a fixed-step controller to an adaptive controller:
1. The new controller should receive the correct `n` value from `updates_dict`
2. Tolerance arrays should be initialized with shape `(n,)` matching the system size
3. If scalar tolerances are provided, they should be broadcast to `(n,)` shape

### Current Behavior

In `_switch_controllers()`:
1. `old_settings` comes from `FixedStepController.settings_dict` which only contains `{n: 1, dt: ...}`
2. `old_settings["step_controller"]` is set to the new controller name
3. `get_controller()` creates new adaptive controller with `old_settings`
4. `AdaptiveStepControlConfig` has default `atol = np.asarray([1e-6])` shape (1,)
5. But `n` from `updates_dict` might be 3
6. `tol_converter` is called with `self_.n = 1` (from old_settings) during config creation
7. After config creation, if `n` is updated, the tolerance arrays are stale

**Alternative root cause:** The `old_settings` dict doesn't include the updated `n` from `updates_dict`, so the new controller is created with `n=1` (the old controller's n).

### Fix Location

`src/cubie/integrators/SingleIntegratorRunCore.py`, method `_switch_controllers()`

### Changes Required

Before creating the new controller, ensure `old_settings` includes the updated `n` value:

```python
old_settings = self._step_controller.settings_dict
old_settings["step_controller"] = new_controller
old_settings["n"] = updates_dict.get("n", self._system.sizes.states)
old_settings["algorithm_order"] = updates_dict.get(
    "algorithm_order", self._algo_step.order)
```

Additionally, ensure that any tolerance values from `updates_dict` override defaults:
- If `updates_dict` contains `atol`, it should be passed to the new controller
- If `updates_dict` contains `rtol`, it should be passed to the new controller

### Integration Points

- `get_controller()` in `step_control/__init__.py` creates controller instances
- `AdaptiveIController`, `AdaptivePIController`, `AdaptivePIDController`, `GustafssonController` all inherit from `BaseAdaptiveStepController`
- `AdaptiveStepControlConfig.tol_converter()` validates tolerance shapes
- `updates_dict` flows from `SingleIntegratorRunCore.update()` through `_switch_controllers()`

### Additional Consideration

The `_switch_controllers` method should also merge relevant tolerance parameters from `updates_dict` into `old_settings` before creating the new controller:

```python
# Merge tolerance updates from updates_dict
for key in ['atol', 'rtol', 'dt_min', 'dt_max']:
    if key in updates_dict:
        old_settings[key] = updates_dict[key]
```

---

## Bug 3: CUDA Simulation Mode Compatibility

### Investigation Required

The error message indicates:
```
AttributeError: tid=[0, 8, 0] ctaid=[0, 0, 0]: module 'numba.cuda' has no attribute 'local'
```

### Affected Tests

Tests in `test_solveresult.py`

### Expected Behavior

CUDA device code should work in both real CUDA and CUDASIM mode without AttributeError.

### Analysis

Looking at `buffer_registry.py` line 136:
```python
array = cuda.local.array(_local_size, _precision)
```

This should work in simulation mode as `numba.cuda.local.array` is supported in the simulator. The error may be:
1. A different code path in solveresult.py
2. A Numba version issue
3. An import issue

### Diagnosis Steps

1. Check if `test_solveresult.py` has explicit `cuda.local` usage
2. Verify cuda_simsafe wrappers are being used consistently
3. Check if the error originates from a different location

### Potential Fix Location

If found in `src/cubie/batchsolving/solveresult.py`, ensure any `cuda.local` usage goes through the `cuda_simsafe` module.

---

## Dependencies and Order

1. **Bug 1** is independent - can be fixed first
2. **Bug 2** is independent - can be fixed in parallel with Bug 1
3. **Bug 3** requires investigation to identify the exact location

## Validation Strategy

1. After fixing Bug 1, run: `pytest tests/integrators/loops/test_ode_loop.py::test_loop -v`
2. After fixing Bug 2, run: `pytest tests/batchsolving/test_config_plumbing.py -v`
3. After fixing Bug 3, run: `pytest tests/batchsolving/test_solveresult.py -v`

---

## Edge Cases

### Bug 1

- What if `state_summaries_buffer_height = 0` legitimately (no summaries requested)?
  - Answer: This is valid; buffer is registered with size 0 and summary metrics are not invoked

### Bug 2

- What if switching from adaptive to fixed controller?
  - Answer: Fixed controller doesn't use atol/rtol, so extra fields are ignored
- What if scalar tolerance provided in updates?
  - Answer: `tol_converter` handles scalars by broadcasting to shape (n,)
- What if `n` changes during the same update?
  - Answer: `n` is captured from `updates_dict` before controller creation

### Bug 3

- Simulator vs real CUDA code paths
  - Answer: Must test in CUDASIM mode to validate

---

## Data Structures

### ODELoopConfig Fields (relevant)

```python
state_summaries_buffer_height: int  # NOT state_summary_buffer_height
observable_summaries_buffer_height: int  # NOT observable_summary_buffer_height
```

### AdaptiveStepControlConfig Fields (relevant)

```python
n: int  # Number of state variables
atol: np.ndarray  # Shape (n,) - converted from scalar or array
rtol: np.ndarray  # Shape (n,) - converted from scalar or array
```

### FixedStepControlConfig.settings_dict

```python
{
    'n': self.n,  # From BaseStepControllerConfig
    'dt': self.dt,
}
```

Note: Does NOT include atol, rtol, dt_min, dt_max as those are adaptive-only
