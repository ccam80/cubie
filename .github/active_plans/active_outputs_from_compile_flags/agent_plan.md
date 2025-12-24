# Agent Plan: ActiveOutputs Derivation from OutputCompileFlags

## Overview

This plan addresses the circular dependency in output flag propagation where `ActiveOutputs` incorrectly uses array sizes to determine activation status. The fix establishes `OutputCompileFlags` as the single source of truth and propagates flags through a factory method.

---

## Component Descriptions

### ActiveOutputs (Modified)
**Location:** `src/cubie/batchsolving/arrays/BatchOutputArrays.py`

**Current Behavior:**
- `update_from_outputarrays()` method checks `array.size > 1` for each output type
- Creates circular dependency: arrays sized → active checked → kernel compiled → arrays used

**Expected Behavior After Fix:**
- New class method `from_compile_flags(compile_flags: OutputCompileFlags) -> ActiveOutputs`
- Creates instance with boolean flags copied from compile flags
- Mapping: save_state→state, save_observables→observables, summarise_state→state_summaries, summarise_observables→observable_summaries, save_counters→iteration_counters
- `status_codes` always `True` (always written during kernel execution)
- `update_from_outputarrays()` deprecated or removed; replaced by factory method

### OutputArrays (Modified)
**Location:** `src/cubie/batchsolving/arrays/BatchOutputArrays.py`

**Current Behavior:**
- `active_outputs` property calls `update_from_outputarrays(self)` which uses size checks
- Used by BatchSolverKernel to get ActiveOutputs after allocation

**Expected Behavior After Fix:**
- `active_outputs` property accepts an optional `compile_flags` parameter OR
- New method to set `_active_outputs` from external source
- When accessing, returns the stored `_active_outputs` without recalculating

### OutputCompileFlags (Unchanged)
**Location:** `src/cubie/outputhandling/output_config.py`

**Role:**
- Authoritative source of boolean compile-time flags
- Computed from `OutputConfig` which knows user-requested output_types and indices
- Already exists and works correctly

### BatchSolverKernel (Modified)
**Location:** `src/cubie/batchsolving/BatchSolverKernel.py`

**Current Behavior:**
- `__init__` calls `output_arrays.update(self)` then gets `output_arrays.active_outputs`
- `update()` method calls `output_arrays.active_outputs` which triggers size-based check
- `run()` method also accesses `output_arrays.active_outputs`

**Expected Behavior After Fix:**
- `__init__` creates `ActiveOutputs` from `single_integrator.output_compile_flags`
- `update()` refreshes `ActiveOutputs` from `single_integrator.output_compile_flags` after updating integrator
- Passes `ActiveOutputs` to compile settings before accessing kernel

### SingleIntegratorRun (Unchanged)
**Location:** `src/cubie/integrators/SingleIntegratorRun.py`

**Role:**
- Already exposes `output_compile_flags` property returning `_output_functions.compile_flags`
- No changes needed; serves as the source for flags

---

## Architectural Changes

### Change 1: Add Factory Method to ActiveOutputs

Add `ActiveOutputs.from_compile_flags()` class method that:
1. Accepts `OutputCompileFlags` instance
2. Maps compile flags to ActiveOutputs attributes
3. Returns new `ActiveOutputs` instance with correct boolean values

### Change 2: Modify ActiveOutputs.update_from_outputarrays

Either:
- **Option A:** Remove the method entirely (breaking change)
- **Option B (Preferred):** Keep method but have it accept `OutputCompileFlags` instead of `OutputArrays`, or
- **Option C:** Keep method signature but ignore size checks, delegating to stored flags

Recommend Option B with deprecation path: keep old signature but warn if called.

### Change 3: Update BatchSolverKernel.__init__

After creating `single_integrator` and before `setup_compile_settings`:
1. Get `compile_flags = single_integrator.output_compile_flags`
2. Create `active_outputs = ActiveOutputs.from_compile_flags(compile_flags)`
3. Pass to `BatchSolverConfig` constructor

### Change 4: Update BatchSolverKernel.update

After `single_integrator.update()` returns:
1. Get fresh `compile_flags = single_integrator.output_compile_flags`
2. Create fresh `active_outputs = ActiveOutputs.from_compile_flags(compile_flags)`
3. Include in `update_compile_settings()` call

### Change 5: Update OutputArrays.active_outputs Property

Modify to:
1. Return stored `_active_outputs` if it has been set externally
2. Or accept `ActiveOutputs` via a setter method
3. Remove dynamic recalculation from size checks

---

## Integration Points

### BatchSolverKernel ↔ SingleIntegratorRun
- BSK accesses `single_integrator.output_compile_flags` (existing property)
- No new coupling; uses existing interface

### BatchSolverKernel ↔ OutputArrays
- BSK sets `output_arrays._active_outputs` after computing from compile flags
- Or BSK bypasses `output_arrays.active_outputs` property entirely

### BatchSolverKernel ↔ BatchSolverConfig
- `ActiveOutputs` already part of `BatchSolverConfig`
- No schema change; just different source of values

### OutputArrays ↔ ActiveOutputs
- Loose coupling: OutputArrays stores ActiveOutputs but doesn't compute it
- Owner relationship becomes: OutputArrays contains, BatchSolverKernel computes

---

## Data Structures

### ActiveOutputs (No schema change)
```python
@attrs.define
class ActiveOutputs:
    state: bool = False
    observables: bool = False
    state_summaries: bool = False
    observable_summaries: bool = False
    status_codes: bool = False
    iteration_counters: bool = False
    
    @classmethod
    def from_compile_flags(cls, flags: OutputCompileFlags) -> "ActiveOutputs":
        return cls(
            state=flags.save_state,
            observables=flags.save_observables,
            state_summaries=flags.summarise_state,
            observable_summaries=flags.summarise_observables,
            status_codes=True,  # Always active
            iteration_counters=flags.save_counters,
        )
```

### OutputCompileFlags (Unchanged)
Already has the boolean flags needed:
- `save_state`, `save_observables`, `summarise`, `summarise_observables`, `summarise_state`, `save_counters`

---

## Dependencies and Imports

### BatchOutputArrays.py
New import needed:
```python
from cubie.outputhandling.output_config import OutputCompileFlags
```

### No other import changes required
All other files already have necessary imports.

---

## Edge Cases

### Edge Case 1: Empty Output Types
When user requests no outputs (validation should catch this, but defensively):
- All ActiveOutputs flags `False` except `status_codes` (always True)

### Edge Case 2: Single-Run Batch
- `status_codes` array has size 1
- Should still be marked active (status_codes=True)
- Fix directly addresses this

### Edge Case 3: Single Variable Summary
- Summary array for one variable with one metric has size 1
- Should still be marked active based on compile flags
- Fix directly addresses this

### Edge Case 4: Update Without Changing Outputs
- Calling `update({"dt": 0.001})` shouldn't change ActiveOutputs
- Compile flags unchanged → ActiveOutputs unchanged
- Natural behavior with proposed approach

### Edge Case 5: Update Enabling Summaries
- Calling `update({"output_types": ["state", "mean"]})` should enable state_summaries
- Compile flags updated → ActiveOutputs.state_summaries=True
- This is the `test_all_lower_plumbing` scenario

---

## Interactions Between Components

### Initialization Flow
1. `BatchSolverKernel.__init__` creates `SingleIntegratorRun`
2. `SingleIntegratorRun.__init__` creates `OutputFunctions` with `compile_flags`
3. BSK gets `compile_flags = single_integrator.output_compile_flags`
4. BSK creates `ActiveOutputs.from_compile_flags(compile_flags)`
5. BSK passes to `BatchSolverConfig`
6. BSK creates `OutputArrays` (sizes determined by solver, not flags)
7. BSK stores `ActiveOutputs` in `output_arrays._active_outputs`

### Update Flow
1. User calls `solverkernel.update({"output_types": [...]})`
2. BSK calls `single_integrator.update(updates_dict)`
3. SingleIntegrator updates `_output_functions` (compile_flags changes)
4. BSK gets fresh `compile_flags = single_integrator.output_compile_flags`
5. BSK creates fresh `ActiveOutputs.from_compile_flags(compile_flags)`
6. BSK calls `update_compile_settings({"ActiveOutputs": active_outputs})`
7. BSK calls `output_arrays.update(self)` to resize arrays
8. Kernel recompilation triggered by changed compile settings

### Kernel Build Flow
1. BSK.build_kernel() accesses `config.ActiveOutputs`
2. Flags used to gate array indexing: `run_index * save_state`
3. Correct slicing based on configuration, not array size
