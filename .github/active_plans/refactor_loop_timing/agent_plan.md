# Agent Plan: Refactor Integrator Loop Timing Parameters

## Overview

This document provides detailed technical specifications for the detailed_implementer and reviewer agents. The refactor completes loop timing parameter handling by updating initialization logic, array sizing, and parameter propagation from the solver level down to ODELoopConfig.

---

## Component 1: ODELoopConfig Enhancement

### Current State
`ODELoopConfig` in `src/cubie/integrators/loops/ode_loop_config.py` has:
- `_save_every`, `_summarise_every`, `_sample_summaries_every` as Optional[float]
- `save_last`, `summarise_last` as bool flags
- `__attrs_post_init__()` with some inference logic

### Required Behavior

#### Case Matrix for Timing Parameters

| save_every | summarise_every | sample_summaries_every | Result |
|------------|-----------------|------------------------|--------|
| None | None | None | save_last=True, summarise_last=True |
| X | None | None | summarise_last=True, summarise_every=10*X, sample_summaries_every=X |
| None | X | None | save_every=X/10, sample_summaries_every=X/10 |
| X | Y | None | sample_summaries_every=X |
| X | None | Z | summarise_every=10*X |
| None | X | Z | save_every=Z |
| X | Y | Z | No defaults needed |

This matrix is already implemented. The changes needed are:

1. **samples_per_summary with duration dependency**: When `summarise_last=True` but `summarise_every=None`, the config needs access to `duration` to calculate `samples_per_summary`. This requires:
   - A new optional `_duration` field
   - Modified `samples_per_summary` property to use duration when summarise_every is None

2. **Reset capability**: Add method or mechanism to reset derived values when None is passed on update, ensuring fresh calculation.

### Integration Points
- Used by `IVPLoop` during initialization and update
- Properties accessed by `BatchSolverKernel` for sizing calculations

---

## Component 2: IVPLoop Update Method Enhancement

### Current State
`IVPLoop.update()` in `src/cubie/integrators/loops/ode_loop.py` passes updates to `update_compile_settings()` and re-registers buffers.

### Required Behavior

1. **None value handling**: When timing parameters are `None`, they should trigger recalculation in ODELoopConfig rather than being ignored.

2. **Duration propagation**: Accept `duration` in updates when `summarise_last` mode needs it.

### Integration Points
- Called by `SingleIntegratorRunCore.update()`
- Calls `ODELoopConfig` methods via `self.update_compile_settings()`

---

## Component 3: BatchSolverKernel Output Length Properties

### Current State
`BatchSolverKernel` in `src/cubie/batchsolving/BatchSolverKernel.py` has:
- `output_length` property: `floor(duration / save_every) + 1`
- `summaries_length` property: `floor(duration / summarise_every)`

### Required Behavior

1. **None-safe output_length**:
   ```python
   @property
   def output_length(self) -> int:
       save_every = self.single_integrator.save_every
       if save_every is None:
           # save_last mode: initial + final = 2
           return 2
       return floor(duration / save_every) + 1
   ```

2. **None-safe summaries_length**:
   ```python
   @property
   def summaries_length(self) -> int:
       summarise_every = self.single_integrator.summarise_every
       if summarise_every is None:
           # summarise_last mode: initial + final = 2
           return 2
       return floor(duration / summarise_every)
   ```

3. **Increment for save_last/summarise_last**: When these flags are True AND periodic saving is also active, ensure the length accounts for both.

### Integration Points
- Used by `SingleRunOutputSizes.from_solver()` and `BatchOutputSizes.from_solver()`
- Drives array allocation in `OutputArrays`

---

## Component 4: Duration Warning at Solver Level

### Current State
`Solver.solve()` in `src/cubie/batchsolving/solver.py` passes duration to `kernel.run()` but doesn't warn about timing dependencies.

### Required Behavior

1. **Detect duration dependency**: After updating timing params, check if `summarise_last=True` and `summarise_every=None`.

2. **Emit warning**:
   ```python
   if (self.kernel.single_integrator._loop.compile_settings.summarise_last and
       self.kernel.single_integrator._loop.compile_settings.summarise_every is None):
       warnings.warn(
           "sample_summaries_every calculated from duration. Changing duration "
           "will trigger recompilation. Set summarise_every explicitly to avoid.",
           UserWarning
       )
   ```

3. **Propagate duration**: Pass duration to update chain when needed for samples_per_summary calculation.

### Integration Points
- Called before `kernel.run()`
- Reads state from `kernel.single_integrator._loop.compile_settings`

---

## Component 5: Parameter Reset Mechanism

### Current State
`update()` methods throughout the chain (Solver, BatchSolverKernel, SingleIntegratorRunCore, IVPLoop) process updates by passing them to sub-components.

### Required Behavior

1. **Explicit None handling**: When a timing parameter is explicitly `None` in an update, it should reset to "unset" state rather than being ignored.

2. **Implementation approach**: 
   - In `ODELoopConfig`, detect when a previously-set value is now None
   - Re-run `__attrs_post_init__()` after updates to re-derive values
   - This already happens via `update_compile_settings()` which should trigger cache invalidation and fresh calculation

3. **Test verification**: Confirm that:
   - Set sample_summaries_every=0.01 on run 1
   - Set sample_summaries_every=None on run 2
   - Value is recalculated, not left at 0.01

### Integration Points
- All `update()` methods in the chain
- `ODELoopConfig.__attrs_post_init__()` for inference

---

## Component 6: Array Sizing Safety

### Current State
`SingleRunOutputSizes.from_solver()` and `BatchOutputSizes.from_solver()` in `src/cubie/outputhandling/output_sizes.py` use solver properties directly.

### Required Behavior

1. **Minimum sizes**: Ensure minimum output_length=1 (for initial value) when no outputs active.

2. **Handle None gracefully**: Properties should never return None; return sensible defaults instead.

3. **Existing safety**: The `ArraySizingClass.nonzero` property coerces zeros to ones, but we should ensure we don't pass None.

### Integration Points
- Called by `OutputArrays.update()` 
- Drives CUDA memory allocation

---

## Expected Interactions Between Components

```
Solver.solve()
    ├── Check for duration dependency → emit warning if detected
    ├── kernel.update(timing_params)
    │       ├── single_integrator.update(timing_params)
    │       │       ├── _output_functions.update()
    │       │       ├── _algo_step.update()
    │       │       ├── _step_controller.update()
    │       │       └── _loop.update(timing_params)
    │       │               ├── update_compile_settings() → ODELoopConfig
    │       │               │       └── __attrs_post_init__() → infer timing
    │       │               └── register_buffers()
    │       └── update_compile_settings() → BatchSolverConfig
    └── kernel.run(duration)
            ├── output_length (None-safe)
            ├── summaries_length (None-safe)
            └── Allocate arrays
```

---

## Data Structures

### ODELoopConfig Changes

```python
# Add optional duration field for samples_per_summary calculation
_duration: Optional[float] = field(
    default=None,
    validator=opt_gttype_validator(float, 0)
)

# Enhanced samples_per_summary property
@property
def samples_per_summary(self) -> Optional[int]:
    if self._summarise_every is None:
        if self._duration is not None:
            # Default to 100 samples when using summarise_last
            return max(1, int(self._duration / 100))
        return None
    return round(self.summarise_every / self.sample_summaries_every)
```

### BatchSolverKernel Changes

```python
@property
def output_length(self) -> int:
    save_every = self.single_integrator.save_every
    save_last = self._get_save_last_flag()
    
    if save_every is None:
        # save_last mode only: initial + final
        return 2 if save_last else 1
    
    base = int(floor(self._duration / save_every)) + 1
    # If save_last is also True and final save doesn't coincide
    # with periodic save, this is handled by loop logic
    return base
```

---

## Edge Cases to Consider

1. **duration=0**: Should not cause division by zero
2. **save_every > duration**: Should produce output_length >= 2 when save_last=True
3. **summarise_every not divisible by sample_summaries_every**: Already has adjustment logic with warning
4. **All timing params None, no outputs requested**: Should fail validation in OutputConfig
5. **Switching from periodic to last-only mode**: Should recalculate all derived values

---

## Dependencies and Imports

No new external dependencies required. Internal imports may need:
- `warnings` module for duration dependency warning
- No changes to existing import structure

---

## Files to Modify

1. `src/cubie/integrators/loops/ode_loop_config.py` - ODELoopConfig enhancements
2. `src/cubie/integrators/loops/ode_loop.py` - IVPLoop update handling
3. `src/cubie/batchsolving/BatchSolverKernel.py` - None-safe length properties
4. `src/cubie/batchsolving/solver.py` - Duration warning
5. `src/cubie/integrators/SingleIntegratorRunCore.py` - Parameter propagation

## Files to Add Tests

1. `tests/integrators/loops/test_ode_loop_config.py` - Config inference tests (may exist)
2. `tests/batchsolving/test_solver.py` - Integration tests for timing modes
