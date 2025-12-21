# Buffer Allocation Refactoring - Agent Plan

## Context

The main algorithm files in `src/cubie/integrators/algorithms/` have been refactored with a new buffer allocation system using `BufferSettings`, `LocalSizes`, and `SliceIndices` classes. The instrumented counterparts in `tests/integrators/algorithms/instrumented/` need to be updated to match.

## Key Patterns to Replicate

### 1. __init__ Methods
Main files now:
- Accept optional buffer location parameters (e.g., `stage_rhs_location`)
- Create buffer settings objects with these parameters
- Pass buffer_settings to config

### 2. build_step Methods
Main files now:
- Unpack buffer_settings from compile_settings
- Unpack boolean flags as compile-time constants (e.g., `stage_rhs_shared`)
- Unpack slice indices for shared memory layout
- Unpack local sizes for local array allocation
- Use selective allocation pattern: `if X_shared: X = shared[slice] else: cuda.local.array(...)`

## Files to Update

1. **explicit_euler.py** - Already synchronized (no changes needed)
2. **backwards_euler.py** - Update build_step buffer allocation
3. **crank_nicolson.py** - Update build_step buffer allocation  
4. **generic_erk.py** - Update __init__ and build_step
5. **generic_dirk.py** - Update __init__ and build_step
6. **generic_firk.py** - Update __init__ and build_step
7. **generic_rosenbrock_w.py** - Update __init__ and build_step

## Preservation Requirements

- All `# LOGGING:` comment sections must be preserved
- Extra parameters in step functions (residuals, jacobian_updates, etc.) must be preserved
- Instrumented solver factory calls must be preserved

## Expected Integration Points

- Buffer settings classes are imported from main algorithm modules
- The instrumented files use `inst_*` factory functions from local `matrix_free_solvers.py`
