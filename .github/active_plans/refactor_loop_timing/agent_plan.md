# Agent Plan: Refactor Loop Timing Parameters

## Overview

This plan details the architectural changes required to refactor CuBIE's loop timing parameters. The refactoring involves:

1. Complete removal of deprecated parameter names (no backward compatibility)
2. New `save_last` and `summarise_last` flags for end-of-run-only behavior
3. Smart inference logic when timing parameters are `None`
4. Duration-dependent recompile warnings

---

## Component 1: ODELoopConfig (Primary Change Target)

**File:** `src/cubie/integrators/loops/ode_loop_config.py`

### Current Behavior
- Has both new names (`save_every`, `summarise_every`, `sample_summaries_every`) and deprecated names (`dt_save`, `dt_summarise`, `dt_update_summaries`)
- `__attrs_post_init__` handles backward compat translation and None inference
- Defaults all timing parameters to `None`
- Applies hardcoded defaults when all are `None`: `save_every=0.1`, `summarise_every=1.0`

### Required Changes

#### Remove Deprecated Fields
Delete the following attrs fields:
- `_dt_save`
- `_dt_summarise`
- `_dt_update_summaries`

Delete the following backward compatibility properties:
- `dt_save` property (keep only `save_every`)
- `dt_summarise` property (keep only `summarise_every`)
- `dt_update_summaries` property (keep only `sample_summaries_every`)

#### Add New Flag Fields
Add two new attrs fields:
- `save_last: bool = field(default=False)` - Controls save-only-at-end behavior
- `summarise_last: bool = field(default=False)` - Controls summarise-only-at-end behavior

#### New None-Handling Logic in `__attrs_post_init__`

**Case 1: All Three None**
- Set `save_last = True`
- Set `summarise_last = True`
- Set `save_every` to a sentinel value or `None` (loop handles specially)
- Set `summarise_every` to a sentinel value or `None`
- No validation of sample dividing summarise (not applicable)

**Case 2: Only `save_every` Set**
- `summarise_last = True` (summarise only at end)
- `summarise_every` will be set to `duration` at kernel compile time
- `sample_summaries_every = save_every` (or `summarise_every / 10` after duration known)

**Case 3: `sample_summaries_every` None, `summarise_every` Set**
- Set `sample_summaries_every = summarise_every / 10`
- Keep existing behavior

**Case 4: `summarise_every` None, `sample_summaries_every` Set**
- `summarise_every` will be set to `duration` at kernel compile time
- Issue recompile warning

**Case 5: Both Summary Params None, `save_every` Set**
- `summarise_last = True`
- `summarise_every` will be `duration / 10` at compile time
- Issue recompile warning

#### Duration Integration
The ODELoopConfig does not receive `duration` at construction time. The duration-dependent logic must:
- Store a flag indicating "duration-dependent" state
- Defer actual value assignment to when duration is known (at kernel start)
- Warning can be issued at config time or kernel compile time

---

## Component 2: IVPLoop

**File:** `src/cubie/integrators/loops/ode_loop.py`

### Current Behavior
- Accepts deprecated parameter names in `__init__`
- Has hardcoded `save_last = False` and `summarise_last = False` in `build()`
- Uses timing values from compile_settings directly

### Required Changes

#### Update `__init__` Signature
Remove deprecated parameter names:
- Remove `dt_save` parameter
- Remove `dt_summarise` parameter
- Remove `dt_update_summaries` parameter

#### Update `ALL_LOOP_SETTINGS` Set
Remove deprecated entries:
- Remove `"dt_save"`
- Remove `"dt_summarise"`
- Remove `"dt_update_summaries"`

#### Update `build()` Method
Replace hardcoded flags with config values:
```python
# Current:
save_last = False
summarise_last = False

# New:
save_last = config.save_last
summarise_last = config.summarise_last
```

#### Update Deprecated Properties
Remove deprecated property methods:
- Remove `dt_save` property (keep `save_every`)
- Remove `dt_summarise` property (keep `summarise_every`)

---

## Component 3: OutputConfig and OutputFunctions

**File:** `src/cubie/outputhandling/output_config.py`

### Required Changes
- Rename `_dt_save` to `_save_every` and `dt_save` property to `save_every`
- Update docstrings to reference new parameter name
- Update `from_loop_settings` classmethod parameter name

**File:** `src/cubie/outputhandling/output_functions.py`

### Required Changes
- Update `ALL_OUTPUT_FUNCTION_PARAMETERS` set:
  - Replace `"dt_save"` with `"save_every"`
- Update `__init__` parameter from `dt_save` to `save_every`
- Update `build()` reference from `config.dt_save` to `config.save_every`

---

## Component 4: Solver and solve_ivp

**File:** `src/cubie/batchsolving/solver.py`

### Required Changes

#### Update `solve_ivp` Function
- Remove `dt_save` parameter
- Keep only `save_every` parameter
- Remove backward compatibility logic that maps dt_save to save_every

#### Update `Solver` Class
Remove deprecated properties:
- Remove `dt_save` property (keep `save_every`)
- Remove `dt_summarise` property (keep `summarise_every`)

#### Update `solve_info` Property
In `SolveSpec` construction:
- Change `dt_save=self.save_every` to `save_every=self.save_every`
- Change `dt_summarise=self.summarise_every` to `summarise_every=self.summarise_every`

---

## Component 5: SolveResult/SolveSpec

**File:** `src/cubie/batchsolving/solveresult.py`

### Required Changes
Update `SolveSpec` attrs class:
- Rename `dt_save` field to `save_every`
- Rename `dt_summarise` field to `summarise_every`
- Add `sample_summaries_every` field if not present

---

## Component 6: SingleIntegratorRunCore

**File:** `src/cubie/integrators/SingleIntegratorRunCore.py`

### Required Changes
- Remove any references to deprecated parameter names in docstrings
- Ensure loop_settings forwarding doesn't reference deprecated names

---

## Component 7: Summary Metrics

**File:** `src/cubie/outputhandling/summarymetrics/metrics.py`

### Required Changes
- Update any references from `dt_save` to `save_every`
- Ensure metric update methods use new parameter name

---

## Component 8: Duration-Dependent Warning System

### Expected Behavior
When `summarise_every` must be derived from `duration`:

1. At config creation time:
   - Set a flag `_summarise_from_duration = True`

2. At kernel compile time (when duration is known):
   - If `_summarise_from_duration` is True:
     - Set `summarise_every = duration` (or appropriate fraction)
     - Issue `UserWarning`:
       ```
       "Summarising only at the end of the run forces the CUDA kernel to 
       recompile whenever duration changes. Set an explicit summarise_every 
       value to avoid this overhead."
       ```

### Implementation Location
The warning should be issued from:
- `BatchSolverKernel.run()` when duration is first known, OR
- `IVPLoop.build()` if duration can be passed there

---

## Integration Points

### Flow: User → Solver → Loop → Kernel

1. **User calls** `solve_ivp(system, y0, params, duration=10.0, save_every=None)`

2. **Solver** creates `BatchSolverKernel` with empty loop_settings

3. **BatchSolverKernel** creates `SingleIntegratorRunCore`

4. **SingleIntegratorRunCore** creates `IVPLoop` with timing=None

5. **IVPLoop** creates `ODELoopConfig`:
   - `__attrs_post_init__` detects all-None condition
   - Sets `save_last=True`, `summarise_last=True`

6. **At kernel execution** (when duration known):
   - If summarise needs duration, compute and warn
   - Pass flags to loop device function

7. **Loop device function**:
   - If `save_last=True`: only save when `t >= t_end`
   - If `summarise_last=True`: only summarise when `t >= t_end`

---

## Edge Cases

### Edge Case 1: Zero Duration
When `duration=0.0`:
- `save_last` should still save initial state
- `summarise_last` should compute summary of initial state only

### Edge Case 2: Very Small Timing Values
When timing values approach precision limits:
- Validation should use precision-aware tolerance (already implemented)

### Edge Case 3: summarise_every Set But No Summaries Requested
When `output_types` doesn't include summary metrics:
- `summarise_every` value is irrelevant
- Don't issue warnings about recompilation

---

## Test Updates Required

### Files to Update
1. `tests/integrators/loops/test_dt_update_summaries_validation.py`
   - Remove backward compatibility tests
   - Add tests for `save_last` and `summarise_last` flags
   - Add tests for new None-handling logic

2. `tests/_utils.py`
   - Update settings dictionaries to use new parameter names

3. `tests/all_in_one.py`
   - Update all timing parameter references

4. All test files using `dt_save`, `dt_summarise`, `dt_update_summaries`:
   - Search and replace with new names
   - Remove deprecation warning suppression

---

## Dependencies and Imports

### Files Importing Timing Constants
Search for files importing or referencing:
- `ALL_LOOP_SETTINGS`
- `ALL_OUTPUT_FUNCTION_PARAMETERS`

Ensure these files are updated when the parameter names change in the sets.

---

## Validation Requirements

### Timing Relationship Validation
When not in `summarise_last` mode:
- `sample_summaries_every` must divide `summarise_every` evenly
- Tolerance based on precision (float32 uses 1e-6, float64 uses 1e-9)

### Flag Consistency Validation
- If `save_last=True`, `save_every` should be None or ignored
- If `summarise_last=True`, `summarise_every` and `sample_summaries_every` can be None

---

## Data Structures

### ODELoopConfig Modified Fields
```python
@define
class ODELoopConfig:
    # ... existing fields ...
    
    # New timing parameters (renamed from dt_* versions)
    _save_every: Optional[float] = field(default=None, ...)
    _summarise_every: Optional[float] = field(default=None, ...)
    _sample_summaries_every: Optional[float] = field(default=None, ...)
    
    # New flags for end-of-run-only behavior
    save_last: bool = field(default=False, ...)
    summarise_last: bool = field(default=False, ...)
    
    # Flag indicating duration-dependent compilation
    _summarise_from_duration: bool = field(default=False, init=False)
```

### OutputConfig Modified Fields
```python
@define
class OutputConfig:
    # ... existing fields ...
    
    # Renamed from _dt_save
    _save_every: float = field(default=0.01, ...)
```

---

## Notes for detailed_implementer

1. **Order of Implementation**: Start with ODELoopConfig, then propagate changes outward to IVPLoop, OutputConfig/Functions, and finally Solver

2. **Search Strategy**: Use repository-wide search for `dt_save`, `dt_summarise`, `dt_update_summaries` to find all occurrences

3. **Test Strategy**: Run existing timing tests first to understand current behavior, then modify tests alongside implementation

4. **Warning Implementation**: Use Python's `warnings.warn()` with `UserWarning` category for recompile warnings

5. **Compile Settings**: Remember that timing values captured in `build()` closures become compile-time constants; changes require recompilation
