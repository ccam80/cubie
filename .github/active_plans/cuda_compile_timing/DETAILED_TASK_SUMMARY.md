# Detailed Implementation Task Summary
# CUDA Compilation Timing Feature

## Overview

This document provides a comprehensive summary of the implementation tasks required for the CUDA compilation timing feature based on the updated architectural plan.

## Critical Architecture Changes

### 1. Remove 'build' Category from TimeLogger
- **Breaking Change**: The 'build' category is being removed
- **New Categories**: Only 'codegen', 'runtime', and 'compile' are supported
- **Impact**: All existing code using 'build' must migrate to 'compile'

### 2. CUDAFunctionCache Base Class
- **Purpose**: Auto-register timing events via attrs introspection
- **Mechanism**: `__attrs_post_init__` iterates through attrs fields
- **Registration**: Automatically creates `compile_{field_name}` events
- **Requirement**: All factory caches must subclass CUDAFunctionCache

### 3. Remove _device_function Mechanism
- **Change**: No more `_device_function` attribute in CUDAFactory
- **Enforcement**: `_build()` requires attrs class return from `build()`
- **Migration**: Single-function factories must create cache with `device_function` field
- **Property Update**: `device_function` property calls `get_cached_output('device_function')`

## Implementation Task Groups

### Group 1: TimeLogger Category Update (SEQUENTIAL)
**File**: `src/cubie/time_logger.py`
**Dependencies**: None

**Changes**:
- Line 377: Change category validation from `{'codegen', 'build', 'runtime'}` to `{'codegen', 'runtime', 'compile'}`
- Update error message to reflect three categories only
- **Breaking**: Existing 'build' events will raise ValueError

**Validation**: category ∈ {'codegen', 'runtime', 'compile'}

---

### Group 2: CUDAFunctionCache Base Class (SEQUENTIAL)
**File**: `src/cubie/CUDAFactory.py`
**Dependencies**: Group 1

**Changes**:
1. **Create CUDAFunctionCache attrs class** (before CUDAFactory class):
   - Implement `__attrs_post_init__(self, factory=None)`
   - Iterate through attrs.fields(self.__class__)
   - For each field with device function (has 'py_func'):
     - Register event: `compile_{field.name}`
     - Description: `"Compilation time for {field.name}"`
     - Category: `"compile"`
   - Skip None, -1, and non-device-functions

**Validation**: 
- factory: Check not None before registration
- device_func: Verify has 'py_func' attribute

---

### Group 3: Signature Introspection Utilities (SEQUENTIAL)
**File**: `src/cubie/CUDAFactory.py`
**Dependencies**: None (standalone)

**New Functions** (module-level, before CUDAFunctionCache):

1. **`get_device_function_params(device_function) -> list[str]`**:
   - Uses `inspect.signature(device_function.py_func)`
   - Returns list of parameter names
   - Falls back to empty list on error

2. **`create_dummy_args(param_names, precision) -> tuple`**:
   - For each param name, create 1-element array or scalar
   - Heuristic: 'dt', 't', 'time', 'duration', 'step' → scalar
   - Default: array (safer for GPU functions)
   - Returns tuple of arguments

3. **`create_dummy_kernel(device_func, num_params) -> callable`**:
   - Closure-based approach for 0-10 parameters
   - Each count has explicit kernel signature
   - `if cuda.grid(1) == 0: device_func(a1, a2, ...)`
   - Fallback to 10-param for num_params > 10

**Required Imports**:
```python
import inspect
import warnings
from numba import cuda
```

---

### Group 4: specialize_and_compile Method (SEQUENTIAL)
**File**: `src/cubie/CUDAFactory.py`
**Dependencies**: Groups 1, 2, 3

**New Method** (in CUDAFactory class):

```python
def specialize_and_compile(self, device_function, event_name):
    """Trigger compilation of device function and record timing."""
    # Skip if None, -1, or not device function
    if device_function is None or device_function == -1:
        return
    if not hasattr(device_function, 'py_func'):
        return
    
    # Skip in CUDA simulator mode
    from cubie.cuda_simsafe import is_cudasim_enabled
    if is_cudasim_enabled():
        return
    
    try:
        # Introspect signature
        params = get_device_function_params(device_function)
        num_params = len(params)
        
        # Get precision from compile_settings or default
        precision = getattr(self._compile_settings, 'precision', np.float64)
        
        # Create dummy arguments
        dummy_args = create_dummy_args(params, precision)
        
        # Create and launch kernel
        self._timing_start(event_name)
        kernel = create_dummy_kernel(device_function, num_params)
        kernel[1, 1](*dummy_args)
        cuda.synchronize()
        self._timing_stop(event_name)
        
    except Exception as e:
        warnings.warn(
            f"Failed to time compilation of {event_name}: {e}",
            RuntimeWarning
        )
```

**Edge Cases**:
- device_function is None: return silently
- CUDA simulator mode: return silently
- Compilation fails: catch exception, warn, continue
- No precision in settings: use default np.float64

---

### Group 5: CUDAFactory _build() Refactoring (SEQUENTIAL)
**File**: `src/cubie/CUDAFactory.py`
**Dependencies**: Groups 2, 4

**Changes**:

1. **Remove _device_function from `__init__`** (line 79):
   - Delete: `self._device_function = None`

2. **Update _build() method** (lines 256-269):
```python
def _build(self):
    """Rebuild cached outputs if they are invalid."""
    build_result = self.build()

    # build() must always return attrs class cache
    if not is_attrs_class(build_result):
        raise TypeError(
            "build() must return an attrs class "
            "(CUDAFunctionCache subclass). "
            "Single-function factories should create a cache with "
            "a 'device_function' field."
        )
    
    # Store cache and call post-init for event registration
    self._cache = build_result
    if hasattr(build_result, '__attrs_post_init__'):
        build_result.__attrs_post_init__(factory=self)
    
    self._cache_valid = True
    
    # Trigger compilation timing for all device functions in cache
    for field in attrs.fields(type(self._cache)):
        device_func = getattr(self._cache, field.name)
        if device_func is None or device_func == -1:
            continue
        if hasattr(device_func, 'py_func'):
            event_name = f"compile_{field.name}"
            self.specialize_and_compile(device_func, event_name)
```

3. **Update device_function property** (lines 127-137):
```python
@property
def device_function(self):
    """Return the compiled CUDA device function."""
    return self.get_cached_output('device_function')
```

**Validation**: build_result must be attrs class

---

### Group 6: Update BaseODE Cache (SEQUENTIAL)
**File**: `src/cubie/odesystems/baseODE.py`
**Dependencies**: Group 2

**Changes**:
1. Add import: `from cubie.CUDAFactory import CUDAFunctionCache`
2. Change line 15-16:
   ```python
   @attrs.define
   class ODECache(CUDAFunctionCache):
   ```

**Result**: Auto-registers events for all ODECache fields:
- compile_dxdt
- compile_linear_operator
- compile_linear_operator_cached
- compile_neumann_preconditioner
- compile_neumann_preconditioner_cached
- compile_stage_residual
- compile_n_stage_residual
- compile_n_stage_linear_operator
- compile_n_stage_neumann_preconditioner
- compile_observables
- compile_prepare_jac
- compile_calculate_cached_jvp
- compile_time_derivative_rhs

---

### Group 7: Update IVPLoop Cache (SEQUENTIAL)
**File**: `src/cubie/integrators/loops/ode_loop.py`
**Dependencies**: Group 2

**Changes**:
1. Create cache class (after imports, before IVPLoop class):
```python
from cubie.CUDAFactory import CUDAFunctionCache
import attrs

@attrs.define
class IVPLoopCache(CUDAFunctionCache):
    """Cache for IVP loop device function."""
    device_function: Callable = attrs.field()
```

2. Update IVPLoop.build() return statement:
```python
# Change from:
return loop_function

# Change to:
return IVPLoopCache(device_function=loop_function)
```

**Result**: Auto-registers compile_device_function event

---

### Group 8: Update OutputFunctions Cache (SEQUENTIAL)
**File**: `src/cubie/outputhandling/output_functions.py`
**Dependencies**: Group 2

**Changes**:
1. Add import: `from cubie.CUDAFactory import CUDAFunctionCache`
2. Change line 43-44:
```python
@attrs.define
class OutputFunctionCache(CUDAFunctionCache):
```

**Result**: Auto-registers events:
- compile_save_state_function
- compile_update_summaries_function
- compile_save_summaries_function

---

### Group 9: Update SingleIntegratorRunCore Cache (SEQUENTIAL)
**File**: `src/cubie/integrators/SingleIntegratorRunCore.py`
**Dependencies**: Group 2

**Changes**:
1. Create cache class (after imports, before SingleIntegratorRunCore):
```python
from cubie.CUDAFactory import CUDAFunctionCache
import attrs

@attrs.define
class SingleIntegratorCache(CUDAFunctionCache):
    """Cache for single integrator loop function."""
    device_function: Callable = attrs.field()
```

2. Update SingleIntegratorRunCore.build() return:
```python
# Change from:
return integrator_loop_function

# Change to:
return SingleIntegratorCache(device_function=integrator_loop_function)
```

**Result**: Auto-registers compile_device_function event

---

### Group 10: Update TimeLogger Tests (SEQUENTIAL)
**File**: `tests/test_time_logger.py`
**Dependencies**: Group 1

**Changes**:
1. **Update test_none_verbosity_no_op** (line 73):
   - Change `"build"` to `"compile"`

2. **Add test_compile_category_accepted**:
```python
def test_compile_category_accepted(self):
    """Test that 'compile' category is accepted."""
    logger = TimeLogger()
    logger._register_event("test_compile", "compile", "Test compilation")
    assert "test_compile" in logger._event_registry
    assert logger._event_registry["test_compile"]["category"] == "compile"
```

3. **Add test_build_category_rejected**:
```python
def test_build_category_rejected(self):
    """Test that 'build' category is no longer accepted."""
    logger = TimeLogger()
    with pytest.raises(ValueError, match="category must be"):
        logger._register_event("test", "build", "Test")
```

4. **Update test_start_event** (line 90):
   - Change `"build"` to `"compile"`

---

### Group 11: Update CUDAFactory Tests (SEQUENTIAL)
**File**: `tests/test_CUDAFactory.py`
**Dependencies**: Group 5

**Changes**:
1. **Update ConcreteFactory fixture** (lines 22-30):
```python
from cubie.CUDAFactory import CUDAFunctionCache

@attrs.define
class ConcreteCache(CUDAFunctionCache):
    device_function: object = attrs.field(default=None)

class ConcreteFactory(CUDAFactory):
    def __init__(self):
        super().__init__()
    
    def build(self):
        return ConcreteCache(device_function=None)
```

2. **Add test_build_must_return_attrs_class**:
```python
def test_build_must_return_attrs_class():
    """Test that build() returning non-attrs class raises TypeError."""
    class BadFactory(CUDAFactory):
        def __init__(self):
            super().__init__()
        
        def build(self):
            return lambda x: x  # Return function instead of cache
    
    factory = BadFactory()
    with pytest.raises(TypeError, match="must return an attrs class"):
        _ = factory.device_function
```

3. **Add test_device_function_uses_get_cached_output**:
```python
def test_device_function_uses_get_cached_output(factory_with_settings):
    """Test that device_function property uses get_cached_output."""
    func = factory_with_settings.device_function
    assert func is factory_with_settings.get_cached_output('device_function')
```

---

### Group 12: Add Integration Tests for Compilation Timing (PARALLEL)
**File**: `tests/test_compilation_timing.py` (NEW FILE)
**Dependencies**: All previous groups

**Complete Test File**:
```python
"""Tests for CUDA compilation timing infrastructure."""

import pytest
import attrs
from numba import cuda

from cubie.CUDAFactory import CUDAFactory, CUDAFunctionCache
from cubie.time_logger import TimeLogger


@attrs.define
class MockCache(CUDAFunctionCache):
    """Mock cache for testing."""
    device_function: object = attrs.field()


class MockFactory(CUDAFactory):
    """Mock factory for testing compilation timing."""
    
    def __init__(self, precision):
        super().__init__()
        self.precision = precision
    
    def build(self):
        @cuda.jit(device=True)
        def mock_device_func(x, y):
            return x + y
        
        return MockCache(device_function=mock_device_func)


@pytest.mark.nocudasim
def test_compilation_timing_records_events():
    """Test that compilation timing records events."""
    import numpy as np
    
    logger = TimeLogger(verbosity='verbose')
    factory = MockFactory(precision=np.float64)
    
    # Override timing callbacks to use test logger
    factory._register_event = logger._register_event
    factory._timing_start = logger.start_event
    factory._timing_stop = logger.stop_event
    
    # Trigger build and compilation timing
    _ = factory.device_function
    
    # Check that compile event was recorded
    assert "compile_device_function" in logger._event_registry
    duration = logger.get_event_duration("compile_device_function")
    assert duration is not None
    assert duration > 0


def test_compilation_timing_skips_none_functions():
    """Test that None device functions are skipped."""
    import numpy as np
    
    @attrs.define
    class CacheWithNone(CUDAFunctionCache):
        device_function: object = attrs.field(default=None)
    
    class FactoryWithNone(CUDAFactory):
        def __init__(self):
            super().__init__()
            self.precision = np.float64
        
        def build(self):
            return CacheWithNone(device_function=None)
    
    logger = TimeLogger(verbosity='verbose')
    factory = FactoryWithNone()
    factory._register_event = logger._register_event
    factory._timing_start = logger.start_event
    factory._timing_stop = logger.stop_event
    
    # Should not raise, should not record events
    _ = factory.device_function
    
    # No compile events should be registered
    compile_events = [k for k in logger._event_registry.keys() 
                     if k.startswith('compile_')]
    assert len(compile_events) == 0


def test_compilation_timing_skips_not_implemented():
    """Test that -1 (not implemented) functions are skipped."""
    import numpy as np
    
    @attrs.define
    class CacheWithNotImpl(CUDAFunctionCache):
        device_function: object = attrs.field(default=-1)
    
    class FactoryWithNotImpl(CUDAFactory):
        def __init__(self):
            super().__init__()
            self.precision = np.float64
        
        def build(self):
            return CacheWithNotImpl(device_function=-1)
    
    logger = TimeLogger(verbosity='verbose')
    factory = FactoryWithNotImpl()
    factory._register_event = logger._register_event
    factory._timing_start = logger.start_event
    factory._timing_stop = logger.stop_event
    
    # Should not raise, should not record events
    with pytest.raises(NotImplementedError):
        _ = factory.device_function


@pytest.mark.sim_only
def test_compilation_timing_skips_in_simulator():
    """Test that compilation timing is skipped in CUDA simulator mode."""
    import numpy as np
    
    logger = TimeLogger(verbosity='verbose')
    factory = MockFactory(precision=np.float64)
    factory._register_event = logger._register_event
    factory._timing_start = logger.start_event
    factory._timing_stop = logger.stop_event
    
    # Trigger build
    _ = factory.device_function
    
    # Events may be registered but no timing should occur
    if "compile_device_function" in logger._event_registry:
        duration = logger.get_event_duration("compile_device_function")
        # In simulator mode, timing may be None or not recorded
        assert duration is None or duration == 0
```

---

## Dependency Chain

```
Group 1 (TimeLogger)
  └─> Group 2 (CUDAFunctionCache)
       └─> Groups 6, 7, 8, 9 (Subclass cache updates)

Group 3 (Utilities) → Group 4 (specialize_and_compile) → Group 5 (_build())

Groups 2, 4 → Group 5

Group 1 → Group 10 (TimeLogger tests)
Group 5 → Group 11 (CUDAFactory tests)
Groups 6-9 → Group 12 (Integration tests)
```

## Parallel Execution Opportunities

- Groups 6, 7, 8, 9 (subclass cache updates) - PARALLEL
- Groups 10, 11, 12 (all tests) - PARALLEL

## Estimated Complexity

**Total Task Groups**: 12
**Total LOC**: ~600-700 lines including tests and documentation

**Complexity by Group**:
- Low: Groups 1, 6, 7, 8, 9, 10 (simple modifications)
- Medium: Groups 2, 3, 5, 11 (attrs classes, utilities, refactoring)
- High: Groups 4, 12 (core compilation timing, integration tests)

## Critical Implementation Notes

1. **Breaking Change**: 'build' category removed - all references must change to 'compile'
2. **Auto-Registration**: CUDAFunctionCache.__attrs_post_init__ handles event registration
3. **Mandatory Caches**: All factories MUST return attrs cache from build()
4. **Single-Function Pattern**: Create cache with single 'device_function' field
5. **Graceful Degradation**: Timing skipped in simulator mode and on errors
6. **Field Name Convention**: Event names are `compile_{field_name}`
7. **Inheritance Required**: All cache classes inherit from CUDAFunctionCache
8. **No Manual Registration**: Subclasses no longer call _register_event manually
