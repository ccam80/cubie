# CUDA Kernel Caching Analysis - Comprehensive Review

## Executive Summary

Comprehensive analysis of all CUDAFactory classes reveals **systematic issues with callable device function tracking**. Multiple classes have callable fields marked `eq=False`, which excludes them from configuration hashing. When these functions are updated via `update()` or `update_compile_settings()`, the config hash doesn't change, leading to stale cached kernels being reused.

## Critical Issue: Loop Function Chain

### Problem Description

The most critical issue is in the **SingleIntegratorRun → IVPLoop** chain:

1. **SingleIntegratorRun.build()** (lines 655-664):
   - Updates IVPLoop with callable functions via `self._loop.update(compiled_functions)`
   - Functions: `save_state_fn`, `update_summaries_fn`, `save_summaries_fn`, `step_controller_fn`, `step_function`, `evaluate_observables`

2. **ODELoopConfig** has ALL these fields marked `eq=False`:
   - `save_state_fn` (line 168)
   - `update_summaries_fn` (line 173)
   - `save_summaries_fn` (line 178)
   - `step_controller_fn` (line 183)
   - `step_function` (line 188)
   - `evaluate_driver_at_t` (line 193)
   - `evaluate_observables` (line 198)

3. **IVPLoop.build()** (lines 346-352):
   - Reads these functions from `config` and captures them in closure
   - Functions are baked into compiled kernel as compile-time constants

### Impact

When any of these device functions change:
- `self._loop.update(compiled_functions)` is called
- Config fields are updated, but hash doesn't change (due to `eq=False`)
- `build()` is NOT called (cache hit with stale hash)
- OLD kernel with OLD function pointers continues to be used

This is **exactly the behavior described in the chunked tests issue** - on CUDA hardware, stale kernels persist across configuration changes.

## All Affected CUDAFactory Classes

### 1. BatchSolverKernel
**File**: `src/cubie/batchsolving/BatchSolverKernel.py`

**Config**: `BatchSolverConfig`
- `loop_fn` (line 117) - marked `eq=False`, appears unused

**Analysis**: 
- `build_kernel()` uses `self.single_integrator.device_function` directly
- `loop_fn` is set in `update()` but never read - dead code
- However, `single_integrator` is child factory, so changes propagate via `config_hash`

### 2. SingleIntegratorRunCore
**File**: `src/cubie/integrators/SingleIntegratorRunCore.py`

**No config fields with eq=False** - uses child factories properly

**Issue**: Updates child factories' callable fields which have `eq=False`:
- Lines 655-663: Updates `_loop` with device functions
- Lines 638-646: Updates `_algo_step` with device functions

### 3. IVPLoop ⚠️ CRITICAL
**File**: `src/cubie/integrators/loops/ode_loop.py`

**Config**: `ODELoopConfig`
- `save_state_fn` - eq=False
- `update_summaries_fn` - eq=False  
- `save_summaries_fn` - eq=False
- `step_controller_fn` - eq=False
- `step_function` - eq=False
- `evaluate_driver_at_t` - eq=False
- `evaluate_observables` - eq=False

**build() method** (lines 346-352): Reads all these from config and captures in closure

**Impact**: Most critical - sits at center of integration chain

### 4. BaseAlgorithmStep (and subclasses)
**File**: `src/cubie/integrators/algorithms/base_algorithm_step.py`

**Config**: `AlgorithmConfig`
- `evaluate_f` (line 150) - eq=False
- `evaluate_observables` (line 155) - eq=False
- `evaluate_driver_at_t` (line 160) - eq=False
- `get_solver_helper_fn` (line 165) - eq=False

**Subclasses**: FIRKStep, DIRKStep, ExplicitEuler, BackwardsEuler, etc.

### 5. ImplicitStep (Rosenbrock, FIRK, DIRK, etc.)
**File**: `src/cubie/integrators/algorithms/ode_implicitstep.py`

**Config**: `ImplicitStepConfig`
- `solver_function` (line 78) - eq=False

**Additional in RosenbrockStep**:
- `time_derivative_function` - eq=False
- `prepare_jacobian_function` - eq=False
- `driver_del_t` - eq=False

**build() method** (line 225-231): Reads these from config

### 6. MatrixFreeSolver (Newton-Krylov, Linear Solvers)
**File**: `src/cubie/integrators/matrix_free_solvers/base_solver.py`

**Config**: `BaseSolverConfig`
- `norm_device_function` (line 58) - eq=False

**LinearSolver**:
- `operator_apply` - eq=False
- `preconditioner` - eq=False

**NewtonKrylov**:
- `residual_function` - eq=False
- `linear_solver_function` - eq=False

## Root Cause Analysis

### Why eq=False Was Used

Callable device functions are marked `eq=False` because:
1. Function objects aren't easily hashable in a stable way
2. Direct comparison with `==` doesn't work for compiled functions
3. Developers wanted to avoid including function identity in hash

### Why This Causes Bugs

1. **Child factory pattern assumed**: Developers expected all callables to come from child factories whose hashes are included
2. **Direct updates bypass hashing**: When `update()` or `update_compile_settings()` changes callable fields, the config object is modified but hash stays same
3. **Cache keyed on stale hash**: Kernel cache uses config_hash as key, so stale kernels persist

### Specific Failure Mode in Chunking

For chunked tests infinite loop:
1. Initial run compiles kernel with certain output flags
2. Test changes output configuration (e.g., different saved arrays)
3. `SingleIntegratorRun.build()` updates loop functions via `self._loop.update(...)`
4. ODELoopConfig updated with new functions, but hash unchanged
5. IVPLoop sees cache hit, returns old kernel
6. Old kernel has wrong array indexing logic baked in
7. On CUDA hardware: memory corruption, infinite loops, or crashes

## Recommendations

### Option 1: Remove eq=False from All Non-Function Fields ✓ SAFEST

Only keep `eq=False` on actual callable fields, ensure all other config fields participate in hashing.

**Pros**: 
- Preserves current architecture
- Child factory pattern continues to work
- Non-functional changes trigger rebuilds as expected

**Cons**:
- Doesn't fix the callable update issue
- Requires ensuring all child factories properly registered

### Option 2: Track Function Hashes Instead of Identity

Store hash/signature of each callable instead of marking eq=False:
```python
_step_function_hash: str = field(default="", init=False, repr=False)

@property 
def step_function(self):
    return self._step_function

@step_function.setter
def step_function(self, value):
    self._step_function = value
    self._step_function_hash = hash_callable(value)
```

**Pros**:
- Properly tracks when functions actually change
- Compatible with update() pattern

**Cons**:
- Requires implementing stable function hashing
- More complex implementation

### Option 3: Force Rebuild on Callable Updates ✓ RECOMMENDED

When `update_compile_settings()` changes any callable field, explicitly invalidate cache:

```python
def update_compile_settings(self, updates_dict=None, **kwargs):
    recognized, changed = super().update_compile_settings(updates_dict, **kwargs)
    
    # Check if any changed fields are callables with eq=False
    for field_name in changed:
        field_obj = self._field_map.get(field_name)
        if field_obj and field_obj.eq is False and callable(getattr(self, field_obj.name)):
            self._invalidate_cache()
            break
    
    return recognized
```

**Pros**:
- Fixes the immediate issue
- Minimal code changes
- Works with existing architecture

**Cons**:
- Rebuilds even when function identity unchanged
- Requires careful implementation in each factory

## Immediate Action Required

1. **Verify child factory relationships**: Ensure all CUDAFactory instances are properly tracked as children
2. **Add cache invalidation logic**: Force rebuild when callable fields updated
3. **Test on CUDA hardware**: Reproduce chunking infinite loop and verify fix

## Files Requiring Changes

1. `src/cubie/integrators/loops/ode_loop_config.py` - ODELoopConfig callable fields
2. `src/cubie/integrators/algorithms/base_algorithm_step.py` - AlgorithmConfig callable fields  
3. `src/cubie/integrators/algorithms/ode_implicitstep.py` - ImplicitStepConfig callable fields
4. `src/cubie/CUDAFactory.py` - Add invalidation logic to update_compile_settings
5. `src/cubie/batchsolving/BatchSolverConfig.py` - Remove unused loop_fn or fix tracking
