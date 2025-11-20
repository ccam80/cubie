# Time Precision Fix - Agent Plan

## Overview

This plan details the implementation of float64 time management while maintaining user-specified precision for state calculations. The fix targets three key files: `solver.py`, `BatchSolverKernel.py`, and `ode_loop.py`.

## Component 1: Solver Class (solver.py)

### Current Behavior
- Stores `_duration`, `_warmup`, `_t0` with `self.precision()` wrapper
- Properties return these values wrapped in `self.precision()`

### Required Changes

#### Time Parameter Storage
**Location:** `Solver.__init__()` is not present, but `BatchSolverKernel.__init__()`
**Current:** 
```python
self._duration = precision(0.0)
self._warmup = precision(0.0)
self._t0 = precision(0.0)
```
**Required:**
```python
self._duration = np.float64(0.0)
self._warmup = np.float64(0.0)
self._t0 = np.float64(0.0)
```

#### Property Getters (in BatchSolverKernel)
**Location:** Lines 849-882 in BatchSolverKernel.py
**Current:**
```python
@property
def duration(self) -> float:
    return self.precision(self._duration)
```
**Required:**
```python
@property
def duration(self) -> float:
    return np.float64(self._duration)
```
Apply to: `duration`, `warmup`, `t0` properties

#### Property Setters (in BatchSolverKernel)
**Location:** Lines 855-882 in BatchSolverKernel.py
**Current:**
```python
@duration.setter
def duration(self, value: float) -> None:
    self._duration = self.precision(value)
```
**Required:**
```python
@duration.setter
def duration(self, value: float) -> None:
    self._duration = np.float64(value)
```
Apply to: `duration`, `warmup`, `t0` setter methods

## Component 2: BatchSolverKernel Class

### Current Behavior
- `run()` method casts time parameters to `precision` before passing to kernel
- Kernel signature uses `precision` for duration, warmup, t0
- `chunk_run()` returns time values in user precision

### Required Changes

#### Import float64 from numba
**Location:** Top of BatchSolverKernel.py (line 8)
**Current:**
```python
from numba import cuda, float64, float32
```
**Status:** Already imported ✓

#### run() Method - Remove Precision Casts
**Location:** Lines 267-274 in BatchSolverKernel.py
**Current:**
```python
precision = self.precision
duration = precision(duration)
warmup = precision(warmup)
t0 = precision(t0)
```
**Required:**
```python
# Time parameters always use float64 for accumulation accuracy
duration = np.float64(duration)
warmup = np.float64(warmup)
t0 = np.float64(t0)
```

#### Kernel Signature - Change Time Parameter Types
**Location:** Lines 496-514 in BatchSolverKernel.py
**Current:**
```python
@cuda.jit(
    (
        precision[:, ::1],
        precision[:, ::1],
        precision[:, :, ::1],
        precision[:, :, :],
        precision[:, :, :],
        precision[:, :, :],
        precision[:, :, :],
        int32[:, :, :],
        int32[::1],
        precision,  # duration
        precision,  # warmup
        precision,  # t0
        int32,
    ),
    **compile_kwargs,
)
```
**Required:**
```python
@cuda.jit(
    (
        precision[:, ::1],
        precision[:, ::1],
        precision[:, :, ::1],
        precision[:, :, :],
        precision[:, :, :],
        precision[:, :, :],
        precision[:, :, :],
        int32[:, :, :],
        int32[::1],
        float64,  # duration - time accumulation uses float64
        float64,  # warmup - time accumulation uses float64
        float64,  # t0 - time accumulation uses float64
        int32,
    ),
    **compile_kwargs,
)
```

#### Kernel Function Signature
**Location:** Lines 515-529 in BatchSolverKernel.py
**Current:**
```python
def integration_kernel(
    inits,
    params,
    d_coefficients,
    state_output,
    observables_output,
    state_summaries_output,
    observables_summaries_output,
    iteration_counters_output,
    status_codes_output,
    duration,
    warmup=precision(0.0),
    t0=precision(0.0),
    n_runs=1,
):
```
**Required:**
```python
def integration_kernel(
    inits,
    params,
    d_coefficients,
    state_output,
    observables_output,
    state_summaries_output,
    observables_summaries_output,
    iteration_counters_output,
    status_codes_output,
    duration,
    warmup=float64(0.0),
    t0=float64(0.0),
    n_runs=1,
):
```

#### chunk_run() - Maintain Return Types
**Location:** Lines 418-469 in BatchSolverKernel.py
**Current behavior:** Returns ChunkParams with time values
**Required:** No changes needed - ChunkParams attributes are typed as float, which accepts float64

#### chunk_run() Call Site - Remove Precision Cast
**Location:** Lines 338-340 in BatchSolverKernel.py
**Current:**
```python
if (chunk_axis == "time") and (i != 0):
    chunk_warmup = precision(0.0)
    chunk_t0 = t0 + precision(i) * chunk_params.duration
```
**Required:**
```python
if (chunk_axis == "time") and (i != 0):
    chunk_warmup = np.float64(0.0)
    chunk_t0 = t0 + np.float64(i) * chunk_params.duration
```

## Component 3: IVPLoop Class (ode_loop.py)

### Current Behavior
- Accepts `duration`, `settling_time`, `t0` as parameters (typed to precision)
- Internal time variable `t` uses precision
- All time arithmetic uses precision

### Required Changes

#### Import float64
**Location:** Top of ode_loop.py (line 13)
**Current:**
```python
from numba import cuda, int16, int32
```
**Required:**
```python
from numba import cuda, int16, int32, float64
```

#### Loop Function Signature - Change Time Parameter Types
**Location:** Lines 226-247 in ode_loop.py
**Current:**
```python
@cuda.jit(
    [
        (
            precision[::1],
            precision[::1],
            precision[:, :, ::1],
            precision[::1],
            precision[::1],
            precision[:, :],
            precision[:, :],
            precision[:, :],
            precision[:, :],
            precision[:,::1],
            precision,  # duration
            precision,  # settling_time
            precision,  # t0
        )
    ],
    device=True,
    inline=True,
    **compile_kwargs,
)
```
**Required:**
```python
@cuda.jit(
    [
        (
            precision[::1],
            precision[::1],
            precision[:, :, ::1],
            precision[::1],
            precision[::1],
            precision[:, :],
            precision[:, :],
            precision[:, :],
            precision[:, :],
            precision[:,::1],
            float64,  # duration - time managed in float64
            float64,  # settling_time - time managed in float64
            float64,  # t0 - time managed in float64
        )
    ],
    device=True,
    inline=True,
    **compile_kwargs,
)
```

#### Loop Function Signature
**Location:** Lines 248-262 in ode_loop.py
**Current:**
```python
def loop_fn(
    initial_states,
    parameters,
    driver_coefficients,
    shared_scratch,
    persistent_local,
    state_output,
    observables_output,
    state_summaries_output,
    observable_summaries_output,
    iteration_counters_output,
    duration,
    settling_time,
    t0=precision(0.0),
):
```
**Required:**
```python
def loop_fn(
    initial_states,
    parameters,
    driver_coefficients,
    shared_scratch,
    persistent_local,
    state_output,
    observables_output,
    state_summaries_output,
    observable_summaries_output,
    iteration_counters_output,
    duration,
    settling_time,
    t0=float64(0.0),
):
```

#### Time Variables - Change to float64
**Location:** Lines 299-300 in ode_loop.py
**Current:**
```python
t = precision(t0)
t_end = precision(settling_time + duration)
```
**Required:**
```python
t = float64(t0)
t_end = float64(settling_time + duration)
```

#### Time Comparison Variables - Change to float64
**Location:** Lines 371-374 in ode_loop.py
**Current:**
```python
if settling_time > precision(0.0):
    # Don't save t0, wait until settling_time
    next_save = precision(settling_time)
else:
    # Seed initial state and save/update summaries
    next_save = precision(dt_save)
```
**Required:**
```python
if settling_time > float64(0.0):
    # Don't save t0, wait until settling_time
    next_save = float64(settling_time)
else:
    # Seed initial state and save/update summaries
    next_save = float64(settling_time + dt_save)
```

Note: `next_save` must use float64, but dt_save remains in precision (it's added to float64 values)

#### Cast dt and t to Precision for Step Function
**Location:** Lines 445-461 in ode_loop.py (step_function call)
**Current:**
```python
step_status = step_function(
    state_buffer,
    state_proposal_buffer,
    parameters_buffer,
    driver_coefficients,
    drivers_buffer,
    drivers_proposal_buffer,
    observables_buffer,
    observables_proposal_buffer,
    error,
    dt_eff,
    t,
    first_step_flag,
    prev_step_accepted_flag,
    remaining_shared_scratch,
    algo_local,
    proposed_counters,
)
```
**Required:**
```python
step_status = step_function(
    state_buffer,
    state_proposal_buffer,
    parameters_buffer,
    driver_coefficients,
    drivers_buffer,
    drivers_proposal_buffer,
    observables_buffer,
    observables_proposal_buffer,
    error,
    precision(dt_eff),  # Cast to user precision
    precision(t),       # Cast to user precision
    first_step_flag,
    prev_step_accepted_flag,
    remaining_shared_scratch,
    algo_local,
    proposed_counters,
)
```

#### Cast t for Driver Function
**Location:** Lines 352-356 in ode_loop.py (initial driver call)
**Current:**
```python
if driver_function is not None and n_drivers > 0:
    driver_function(
        t,
        driver_coefficients,
        drivers_buffer,
    )
```
**Required:**
```python
if driver_function is not None and n_drivers > 0:
    driver_function(
        precision(t),  # Cast to user precision
        driver_coefficients,
        drivers_buffer,
    )
```

#### Cast t for Observables Function (initial)
**Location:** Lines 357-364 in ode_loop.py
**Current:**
```python
if n_observables > 0:
    observables_fn(
        state_buffer,
        parameters_buffer,
        drivers_buffer,
        observables_buffer,
        t,
    )
```
**Required:**
```python
if n_observables > 0:
    observables_fn(
        state_buffer,
        parameters_buffer,
        drivers_buffer,
        observables_buffer,
        precision(t),  # Cast to user precision
    )
```

#### Update save_state calls with cast t
**Location:** Lines 376-384, 524-532 in ode_loop.py
**Current:**
```python
save_state(
    state_buffer,
    observables_buffer,
    counters_since_save,
    t,
    state_output[save_idx * save_state_bool, :],
    observables_output[save_idx * save_obs_bool, :],
    iteration_counters_output[save_idx * save_counters_bool, :],
)
```
**Required:**
```python
save_state(
    state_buffer,
    observables_buffer,
    counters_since_save,
    precision(t),  # Cast to user precision
    state_output[save_idx * save_state_bool, :],
    observables_output[save_idx * save_obs_bool, :],
    iteration_counters_output[save_idx * save_counters_bool, :],
)
```

#### Keep dt_eff Comparison in float64
**Location:** Line 443 in ode_loop.py
**Current:**
```python
status |= selp(dt_eff <= precision(0.0), int32(16), int32(0))
```
**Required:**
```python
status |= selp(dt_eff <= float64(0.0), int32(16), int32(0))
```

#### Remove precision casts for dt_min in max_steps calculation
**Location:** Line 304 in ode_loop.py
**Current:**
```python
max_steps = (int32(ceil(t_end / dt_min)) + int32(2))
```
**Status:** dt_min is stored as precision in config, no change needed. The division will automatically promote to float64 when t_end is float64.

## Component 4: Related Configuration Classes

### ODELoopConfig
**Location:** ode_loop_config.py
**Expected:** No changes needed - config stores dt_save, dt_summarise, dt_min in precision as they are interval specifications, not accumulated values. When used in float64 arithmetic, they will be automatically promoted.

### ChunkParams
**Location:** BatchSolverKernel.py, lines 44-65
**Expected:** No changes needed - attrs class with float-typed fields accepts float64 values

## Data Structures and Their Purposes

### Time Variables (float64)
- `t`: Accumulated simulation time - must be float64 to avoid precision loss
- `t_end`: Target end time computed from duration and settling_time
- `next_save`: Next scheduled save point - computed from accumulation
- `t_proposal`: Proposed next time value after step
- `duration`, `warmup`, `t0`: High-level time parameters from user

### Interval Variables (remain in user precision)
- `dt_save`: Save interval specification
- `dt_summarise`: Summary interval specification  
- `dt`, `dt_eff`: Current/effective step size
- `dt0`, `dt_min`, `dt_max`: Step size constraints

### Rationale
Time accumulation suffers from catastrophic precision loss in float32. Intervals and step sizes are applied once per step and don't accumulate, so user precision is sufficient.

## Integration Points with Current Codebase

### Step Functions (algorithms/)
**Signature:** Receive `dt_scalar` and `time_scalar` as `numba_precision` type
**Change Required:** None - IVPLoop casts to precision before passing
**Files:** All files in `src/cubie/integrators/algorithms/`

### Driver Functions
**Signature:** Receive time as first parameter in precision
**Change Required:** None - IVPLoop casts to precision before passing
**Files:** `src/cubie/integrators/array_interpolator.py`

### Output Functions
**Signature:** save_state receives time as parameter
**Change Required:** None - IVPLoop casts to precision before passing
**Files:** `src/cubie/outputhandling/output_functions.py`

## Edge Cases to Consider

### Case 1: Very Long Integrations
**Scenario:** t0=1e10, duration=1e6, dt_min=1e-8
**Handling:** Float64 provides ~15 significant digits, sufficient for this range
**Test:** Verify t increases correctly throughout integration

### Case 2: Very Small Time Steps
**Scenario:** precision=float32, dt_min=1e-9
**Handling:** Time accumulation in float64 prevents precision loss
**Test:** Run 1e6 steps of dt=1e-9, verify t reaches 1e-3

### Case 3: Comparison Near Save Points
**Scenario:** next_save = 1.0, t = 0.999999999, dt = 0.000000002
**Handling:** Float64 precision allows accurate comparison
**Test:** Verify saves occur at correct intervals

### Case 4: Mixed-Precision Arithmetic
**Scenario:** next_save (float64) + dt_save (precision)
**Handling:** NumPy/Numba automatically promotes to float64
**Test:** Verify no warnings and correct results

## Dependencies and Imports Required

### New Imports
- `from numba import float64` in ode_loop.py ✓ (already present via existing import)
- No new numpy imports needed (np.float64 is a built-in type)

### Modified Imports
- None - all necessary imports already present

## Expected Interactions Between Components

```
User Input (float)
    ↓
Solver (stores as np.float64)
    ↓
BatchSolverKernel.run (casts to np.float64)
    ↓
integration_kernel (receives float64)
    ↓
IVPLoop.loop_fn (receives float64, manages time in float64)
    ↓ (casts to precision for:)
    ├→ step_function(precision(dt_eff), precision(t))
    ├→ driver_function(precision(t), ...)
    ├→ observables_fn(..., precision(t))
    └→ save_state(..., precision(t), ...)
```

## Validation Points

1. **Type Checking:** Verify CUDA compilation succeeds with new signatures
2. **Precision Preservation:** Verify time variables maintain float64 throughout loop
3. **Boundary Casting:** Verify dt and t are cast to precision before step functions
4. **Accumulation Accuracy:** Verify t increases correctly with small dt values
5. **Comparison Accuracy:** Verify next_save comparisons work correctly

## Summary of Changes by File

### solver.py
- No changes (time parameters handled by BatchSolverKernel)

### BatchSolverKernel.py
- Change `_duration`, `_warmup`, `_t0` initialization to np.float64
- Change duration, warmup, t0 property getters to return np.float64
- Change duration, warmup, t0 property setters to cast to np.float64
- Remove precision casts in run() method for time parameters
- Change CUDA kernel signature time parameters from precision to float64
- Change kernel function signature time parameters from precision to float64
- Update chunk_run warmup and t0 to use np.float64

### ode_loop.py  
- Import float64 from numba
- Change loop_fn signature time parameters from precision to float64
- Change t, t_end, next_save to use float64
- Cast dt_eff and t to precision when calling step_function
- Cast t to precision when calling driver_function
- Cast t to precision when calling observables_fn  
- Cast t to precision when calling save_state
- Change dt_eff comparison to use float64(0.0)
- Keep dt_save and dt_summarise in precision (intervals)

## Implementation Status

### Completed Changes
1. BatchSolverKernel: Time parameters stored and returned as float64
2. BatchSolverKernel: CUDA kernel signature uses float64 for duration, warmup, t0
3. IVPLoop: loop_fn signature accepts float64 for duration, settling_time, t0
4. IVPLoop: Time accumulation variables (t, t_end, next_save) use float64
5. IVPLoop: Casts t and dt_eff to user precision before passing to step functions
6. IVPLoop: Casts t to user precision before passing to driver/observable functions
7. IVPLoop: Casts t to user precision before passing to save_state function

### Precision Boundary Enforcement
- Float64 above IVPLoop: BatchSolverKernel, integration_kernel, loop_fn parameters
- User precision below IVPLoop: step functions, driver functions, observable functions, save_state
- Interval parameters (dt_save, dt_summarise, dt_min, dt_max) remain in user precision
