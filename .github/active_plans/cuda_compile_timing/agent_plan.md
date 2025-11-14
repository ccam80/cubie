# CUDA Compilation Timing - Agent Plan

## Overview

This plan adds CUDA compilation timing to CuBIE's existing time logging infrastructure. The implementation enables tracking of just-in-time (JIT) compilation time for Numba CUDA device functions, providing visibility into compilation overhead that currently contributes to the "click, wait, hope" user experience.

### Key Architectural Changes

1. **Remove 'build' category from TimeLogger** - Simplify to 3 categories: 'codegen', 'runtime', 'compile'
2. **Create CUDAFunctionCache base class** - Auto-registers timing events via attrs introspection
3. **Remove _device_function attribute** - All factories must return attrs cache from build()
4. **Update device_function property** - Calls get_cached_output('device_function') for consistency
5. **Automatic event registration** - CUDAFunctionCache.__attrs_post_init__ handles registration

## Behavioral Requirements

### TimeLogger Category Extension

**Current behavior:**
- TimeLogger validates categories against `{'codegen', 'build', 'runtime'}`
- Registration of events outside these categories raises ValueError

**Required behavior:**
- TimeLogger validates categories against `{'codegen', 'runtime', 'compile'}`
- Remove 'build' category (was effectively an alias for compile)
- 'compile' category events aggregate separately from other categories
- Error messages reflect the three-category system
- All existing 'build' event registrations must be migrated to 'compile'

### CUDAFactory Compilation Specialization

**Current behavior:**
- CUDAFactory.build() creates device functions but doesn't trigger compilation
- Device functions compile on first call with specific types (Numba JIT behavior)
- No timing or visibility into compilation phase

**Required behavior:**
- CUDAFactory provides `specialize_and_compile(device_function, event_name)` method
- Method creates minimal dummy kernel that calls device function
- Method launches kernel with appropriately typed arguments to trigger compilation
- Method times the entire process: kernel creation → launch → synchronization
- Timing events use category='compile'
- Results are not returned (method called for timing side effect only)

### Automatic Compilation Timing Invocation

**Current behavior:**
- CUDAFactory._build() calls self.build() and caches result
- No post-build operations

**Required behavior:**
- After calling self.build(), _build() triggers compilation timing for device functions
- Detection of device functions in build result (check for attrs class vs single function)
- Automatic invocation of specialize_and_compile() for each device function
- Timing only occurs once per cache invalidation (same as build())
- If timing fails (no GPU, etc.), log warning but don't raise exception

### CUDAFunctionCache Base Class

**Current behavior:**
- No base class for cache attrs classes
- Manual event registration in each CUDAFactory subclass `__init__`

**Required behavior:**
- Create `CUDAFunctionCache` base attrs class in CUDAFactory.py
- Implement `__attrs_post_init__` method that:
  - Accepts `factory` parameter (reference to parent CUDAFactory)
  - Iterates through all attrs fields in the cache instance
  - For each field containing a CUDA device function:
    - Registers timing event with name `compile_{field_name}`
    - Uses description `"Compilation time for {field_name}"`
    - Uses category 'compile'
- All CUDAFactory subclass caches must inherit from CUDAFunctionCache
- Auto-registration eliminates need for manual `_register_event()` calls

### CUDAFactory device_function Property

**Current behavior:**
- `device_function` property returns `self._device_function` directly
- `_device_function` is populated in `_build()` for both single and multi-output cases

**Required behavior:**
- Remove `_device_function` attribute entirely
- Change `device_function` property to call `self.get_cached_output('device_function')`
- All caches must include a `device_function` field
- Backward compatible: existing code using `factory.device_function` continues to work

## Architectural Changes

### TimeLogger Module (`src/cubie/time_logger.py`)

**Change:** Remove 'build' category and keep only 'codegen', 'runtime', 'compile'

**Location:** `_register_event()` method, line ~377

**Modification:**
```python
# Change from:
if category not in {'codegen', 'build', 'runtime'}:
    raise ValueError(
        f"category must be 'codegen', 'build', or 'runtime', "
        f"got '{category}'"
    )

# Change to:
if category not in {'codegen', 'runtime', 'compile'}:
    raise ValueError(
        f"category must be 'codegen', 'runtime', or 'compile', "
        f"got '{category}'"
    )
```

**Impact:**
- Enables registration of 'compile' category events
- Removes 'build' category (breaking change)
- Existing code using 'build' must be updated to 'compile'
- No changes to data structures or aggregation logic

### CUDAFunctionCache Base Class (`src/cubie/CUDAFactory.py`)

**Change:** Create base attrs class for auto-registering device function compilation events

**Location:** New class before CUDAFactory class definition

**Implementation:**
```python
@attrs.define
class CUDAFunctionCache:
    """Base class for CUDAFactory cache containers.
    
    Automatically registers compilation timing events for device functions
    by introspecting attrs fields during initialization.
    """
    
    def __attrs_post_init__(self, factory=None):
        """Register compilation events for all device function fields."""
        if factory is None:
            return
        
        for field in attrs.fields(self.__class__):
            device_func = getattr(self, field.name)
            if device_func is None or device_func == -1:
                continue
            if not hasattr(device_func, 'py_func'):
                continue
            
            event_name = f"compile_{field.name}"
            description = f"Compilation time for {field.name}"
            factory._register_event(event_name, "compile", description)
```

**Impact:**
- Eliminates manual event registration in subclass `__init__` methods
- Ensures consistency between cache field names and event names

### CUDAFactory Base Class (`src/cubie/CUDAFactory.py`)

#### Change 1: Add `specialize_and_compile()` method

**Location:** New method in CUDAFactory class

**Behavior:**
1. Accept device function and event name as parameters
2. Start timing event
3. Introspect device function signature to determine parameters
4. Create minimal dummy arguments (arrays and scalars) based on signature
5. Define CUDA kernel that calls device function with dummy arguments
6. Launch kernel with grid [1, 1] and block size 1
7. Call `cuda.synchronize()` to wait for compilation
8. Stop timing event

**Signature:**
```python
def specialize_and_compile(
    self, device_function: Callable, event_name: str
) -> None:
    """Trigger compilation of device function and record timing.
    
    Parameters
    ----------
    device_function
        Numba CUDA device function to compile
    event_name
        Name of timing event to record (must be pre-registered)
    
    Notes
    -----
    This method creates a minimal CUDA kernel that calls the device
    function with appropriately typed arguments. The kernel launch
    triggers Numba's JIT compilation, which is timed and recorded.
    
    The method is called automatically by _build() for all device
    functions returned from build(). Manual invocation is not needed.
    """
```

**Edge cases:**
- Device function is None: skip timing (no-op)
- Device function is not a CUDA dispatcher: skip timing with warning
- CUDA not available (simulator mode): skip timing silently
- Event not registered: TimeLogger will raise ValueError (existing behavior)
- Signature introspection fails: fall back to minimal default signature

**Dependencies:**
- `inspect` module for signature introspection
- `cuda` from `numba` for kernel creation and synchronization
- `cuda_simsafe.is_devfunc()` to verify device function

#### Change 2: Remove _device_function and update _build()

**Location:** `_build()` method and `device_function` property

**Current code (~line 256-269):**
```python
def _build(self):
    """Rebuild cached outputs if they are invalid."""
    build_result = self.build()

    # Multi-output case
    if is_attrs_class(build_result):
        self._cache = build_result
        # If 'device_function' is in the dict, make it an attribute
        if in_attr("device_function", build_result):
            self._device_function = build_result.device_function
    else:
        self._device_function = build_result

    self._cache_valid = True
```

**Required modifications:**

1. **Remove _device_function from __init__:**
```python
def __init__(self):
    self._compile_settings = None
    self._cache_valid = True
    # Remove: self._device_function = None
    self._cache = None
    # ... timing callbacks
```

2. **Update _build() to always expect attrs cache:**
```python
def _build(self):
    """Rebuild cached outputs if they are invalid."""
    build_result = self.build()

    # build() must always return attrs class cache
    if not is_attrs_class(build_result):
        raise TypeError(
            "build() must return an attrs class (CUDAFunctionCache subclass)"
        )
    
    # Pass factory reference for auto-registration
    self._cache = build_result
    # Call __attrs_post_init__ if not already called
    if hasattr(build_result, '__attrs_post_init__'):
        build_result.__attrs_post_init__(factory=self)
    
    self._cache_valid = True
    
    # Trigger compilation timing for all device functions
    for field in attrs.fields(type(self._cache)):
        device_func = getattr(self._cache, field.name)
        if device_func is None or device_func == -1:
            continue
        if hasattr(device_func, 'py_func'):
            event_name = f"compile_{field.name}"
            self.specialize_and_compile(device_func, event_name)
```

3. **Update device_function property:**
```python
@property
def device_function(self):
    """Return the compiled CUDA device function."""
    return self.get_cached_output('device_function')
```

**Impact:**
- Simplifies caching logic (single code path)
- All factories must create cache, even for single function
- Backward compatible: `factory.device_function` still works

### CUDAFactory Subclasses

Each subclass's cache attrs class must inherit from `CUDAFunctionCache`. Event registration happens automatically via `__attrs_post_init__`.

#### BaseODE (`src/cubie/odesystems/baseODE.py`)

**Location:** `ODECache` class definition (lines 15-50)

**Change:** Make ODECache inherit from CUDAFunctionCache

**Current:**
```python
@attrs.define
class ODECache:
    """Cache compiled CUDA device and support functions for an ODE system."""
    dxdt: Optional[Callable] = attrs.field()
    linear_operator: Optional[Union[Callable, int]] = attrs.field(default=-1)
    # ... other fields
```

**Updated:**
```python
from cubie.CUDAFactory import CUDAFunctionCache

@attrs.define
class ODECache(CUDAFunctionCache):
    """Cache compiled CUDA device and support functions for an ODE system."""
    dxdt: Optional[Callable] = attrs.field()
    linear_operator: Optional[Union[Callable, int]] = attrs.field(default=-1)
    # ... other fields (unchanged)
```

**Impact:**
- Automatic registration of compile events for all ODECache fields
- Event names: compile_dxdt, compile_linear_operator, compile_neumann_preconditioner, etc.
- Remove any manual `_register_event()` calls from BaseODE.__init__

#### IVPLoop (`src/cubie/integrators/loops/ode_loop.py`)

**Location:** `build()` method return value

**Change:** Create cache attrs class inheriting from CUDAFunctionCache

**Current:**
```python
def build(self):
    # ... build loop function
    return loop_function  # returns device function directly
```

**Updated:**
```python
from cubie.CUDAFactory import CUDAFunctionCache

@attrs.define
class IVPLoopCache(CUDAFunctionCache):
    """Cache for IVP loop device function."""
    device_function: Callable = attrs.field()

# In IVPLoop class:
def build(self):
    # ... build loop function
    return IVPLoopCache(device_function=loop_function)
```

**Impact:**
- Creates cache with single `device_function` field
- Automatic registration of compile_device_function event
- Maintains backward compatibility via `factory.device_function` property

#### OutputFunctions (`src/cubie/outputhandling/output_functions.py`)

**Location:** `OutputFunctionCache` class definition

**Change:** Make OutputFunctionCache inherit from CUDAFunctionCache

**Current:**
```python
@attrs.define
class OutputFunctionCache:
    """Cache for output device functions."""
    save_state_function: Callable = attrs.field()
    update_summaries_function: Callable = attrs.field()
    save_summaries_function: Callable = attrs.field()
```

**Updated:**
```python
from cubie.CUDAFactory import CUDAFunctionCache

@attrs.define
class OutputFunctionCache(CUDAFunctionCache):
    """Cache for output device functions."""
    save_state_function: Callable = attrs.field()
    update_summaries_function: Callable = attrs.field()
    save_summaries_function: Callable = attrs.field()
```

**Impact:**
- Automatic registration of compile_save_state_function, compile_update_summaries_function, compile_save_summaries_function
- Remove any manual `_register_event()` calls from OutputFunctions.__init__

#### SingleIntegratorRunCore (`src/cubie/integrators/SingleIntegratorRunCore.py`)

**Location:** `build()` method return value

**Change:** Create cache attrs class inheriting from CUDAFunctionCache

**Current:**
```python
def build(self):
    # ... build integrator loop
    return compiled_loop_function
```

**Updated:**
```python
from cubie.CUDAFactory import CUDAFunctionCache

@attrs.define
class SingleIntegratorCache(CUDAFunctionCache):
    """Cache for single integrator loop function."""
    device_function: Callable = attrs.field()

# In SingleIntegratorRunCore class:
def build(self):
    # ... build integrator loop
    return SingleIntegratorCache(device_function=compiled_loop_function)
```

**Impact:**
- Creates cache with single `device_function` field
- Automatic registration of compile_device_function event

#### Algorithm Steps, Controllers, Metrics

**Recommendation:** Defer to implementation phase

**Rationale:**
- Focus on high-level components (BaseODE, IVPLoop, OutputFunctions, SingleIntegratorRunCore)
- These may already return attrs caches or single functions
- Can be updated incrementally to inherit from CUDAFunctionCache
- Implementation will determine best approach per component

#### Algorithm Steps, Controllers, Metrics

**Recommendation:** Start with BaseODE, IVPLoop, OutputFunctions, SingleIntegratorRunCore

**Defer:** Algorithm steps, controllers, and metrics to future work if they add complexity

**Rationale:**
- Focus on high-level components that users directly interact with
- Algorithm/controller compilation may be fast relative to ODE system
- Can add incrementally in follow-up PRs

## Integration Points

### Signature Introspection Utility

**Location:** New utility function in `src/cubie/CUDAFactory.py` or `src/cubie/_utils.py`

**Purpose:** Extract parameter information from device function

**Signature:**
```python
def get_device_function_params(device_function: Callable) -> List[str]:
    """Extract parameter names from CUDA device function.
    
    Parameters
    ----------
    device_function
        Numba CUDA device function (CUDADispatcher)
    
    Returns
    -------
    list[str]
        List of parameter names in order
    
    Notes
    -----
    Accesses py_func attribute for introspection. Falls back to
    empty list if introspection unavailable.
    """
```

**Implementation approach:**
```python
import inspect

def get_device_function_params(device_function):
    try:
        if hasattr(device_function, 'py_func'):
            sig = inspect.signature(device_function.py_func)
            return list(sig.parameters.keys())
    except Exception:
        pass
    return []
```

### Dummy Argument Creation

**Location:** Helper method in CUDAFactory or utility function

**Purpose:** Create minimal typed arguments for device function call

**Signature:**
```python
def create_dummy_args(
    param_names: List[str], precision: PrecisionDType
) -> Tuple[Any, ...]:
    """Create minimal dummy arguments for device function.
    
    Parameters
    ----------
    param_names
        List of parameter names from device function
    precision
        Numerical precision for scalar and array arguments
    
    Returns
    -------
    tuple
        Tuple of dummy arguments (arrays and scalars)
    
    Notes
    -----
    Creates 1-element arrays for array-like parameters and zero
    scalars for scalar parameters. Uses heuristics based on
    parameter names to infer types.
    """
```

**Implementation heuristics:**
- Parameters named like arrays (state, params, buffer, etc.): 1-element array
- Parameters named like scalars (dt, t, duration, etc.): scalar zero
- Default: assume array (safer for CUDA kernels)
- Use precision from factory settings for dtype

### Kernel Template

**Location:** Defined within `specialize_and_compile()` method

**Purpose:** Minimal kernel that calls device function

**Pattern:**
```python
from numba import cuda

# Inside specialize_and_compile()
@cuda.jit
def dummy_compilation_kernel(arg1, arg2, arg3, ...):
    """Minimal kernel to trigger device function compilation."""
    idx = cuda.grid(1)
    if idx == 0:
        # Call device function with passed arguments
        device_function(arg1, arg2, arg3, ...)
```

**Challenges:**
- Cannot use *args in kernel call (Numba limitation)
- Must generate kernel with exact signature matching param count
- Two options:
  1. Use exec() to generate kernel string dynamically
  2. Create fixed templates for common param counts (1-10 params)
  3. Use closure to capture device function and fixed args

**Recommendation:** Use option 3 (closure) for simplicity and safety

**Example implementation:**
```python
def create_dummy_kernel(device_func, num_params):
    """Create dummy kernel with appropriate signature."""
    
    # Create closure that captures device_func
    # Use conditional logic for different param counts
    if num_params == 1:
        @cuda.jit
        def kernel(a1):
            if cuda.grid(1) == 0:
                device_func(a1)
    elif num_params == 2:
        @cuda.jit
        def kernel(a1, a2):
            if cuda.grid(1) == 0:
                device_func(a1, a2)
    # ... continue for common param counts
    else:
        # Fall back to maximum expected params
        @cuda.jit
        def kernel(a1, a2, a3, a4, a5, a6, a7, a8):
            if cuda.grid(1) == 0:
                device_func(a1, a2, a3, a4, a5, a6, a7, a8)
    
    return kernel
```

## Data Structures

### No New Data Structures Required

**Rationale:**
- TimeLogger event registry already supports arbitrary event names
- ODECache, OutputFunctionCache already use attrs classes
- No new cache containers needed
- Timing data stored in existing TimeLogger.events list

### Modified Data Structures

**TimeLogger._event_registry**
- No structural changes
- New entries with category='compile'
- Example: `{'compile_dxdt': {'category': 'compile', 'description': '...'}}`

## Dependencies and Imports

### New imports in `src/cubie/CUDAFactory.py`
```python
import inspect  # For signature introspection
from numba import cuda  # For kernel creation and synchronization
from cubie.cuda_simsafe import is_devfunc  # For device function detection
```

### Existing imports (already available)
- `attrs` - for attrs class detection
- `cubie._utils` - for `in_attr()` and `is_attrs_class()`
- `cubie.time_logger` - for timing callbacks

## Edge Cases

### Device Function is None or -1
**Scenario:** ODECache field is -1 or None (function not implemented)
**Handling:** Skip timing (is_devfunc() returns False)
**Impact:** No error, no timing event recorded

### CUDA Not Available (Simulator Mode)
**Scenario:** Running with NUMBA_ENABLE_CUDASIM=1 or no GPU
**Handling:** 
- `is_cudasim_enabled()` check before kernel launch
- Skip compilation timing in simulator mode
- Log debug message if verbosity >= debug
**Impact:** No timing data in simulator mode (acceptable)

### Signature Introspection Fails
**Scenario:** Device function lacks py_func attribute or signature fails
**Handling:** Fall back to generic signature (e.g., 3 array params, 2 scalars)
**Impact:** May fail at kernel launch, caught and logged as warning

### Event Not Registered
**Scenario:** Subclass forgot to register event in __init__
**Handling:** Registry check in _build() before calling specialize_and_compile()
**Alternative:** Allow TimeLogger to raise ValueError (existing behavior)
**Recommendation:** Use registry check to avoid errors, enable gradual rollout

### Kernel Launch Fails
**Scenario:** Kernel launch raises exception (type mismatch, memory error, etc.)
**Handling:** Wrap in try/except, log warning, continue without timing
**Impact:** Missing compilation timing for that function, but doesn't break build

### Multiple Compilations per Function
**Scenario:** Device function called with different type signatures
**Handling:** Current design only times first compilation (in _build())
**Impact:** Subsequent specializations not timed (acceptable for MVP)
**Future:** Could add tracking of all specializations if needed

## Expected Interactions Between Components

### Solver → BaseODE → TimeLogger
1. Solver creates system (e.g., SymbolicODE which inherits from BaseODE)
2. BaseODE.__init__() registers compile events
3. First access to system.device_function triggers _build()
4. _build() calls build(), caches result, then triggers compilation timing
5. specialize_and_compile() launches dummy kernel for each device function
6. TimeLogger records compilation durations
7. Subsequent accesses use cached compiled functions (no re-timing)

### SingleIntegratorRun → IVPLoop → TimeLogger
1. SingleIntegratorRun creates IVPLoop
2. IVPLoop.__init__() registers compile event
3. Access to compiled_loop_function triggers _build()
4. Compilation timing triggered for loop function
5. Loop function used by integrator kernel (already compiled)

### BatchSolverKernel → Integrator → ODE System
1. BatchSolverKernel creates SingleIntegratorRun
2. SingleIntegratorRun creates IVPLoop, OutputFunctions, Algorithm, Controller
3. Each factory registers compile events and times on first build
4. All compilation timing happens before first solve() call
5. Actual solve() uses pre-compiled functions (fast path)

## Implementation Sequence

### Phase 1: Core Infrastructure
1. Modify TimeLogger to accept 'compile' category
2. Implement `specialize_and_compile()` in CUDAFactory
3. Add helper functions (signature introspection, dummy args, kernel creation)
4. Modify `_build()` to invoke compilation timing
5. Add tests for specialize_and_compile()

### Phase 2: Subclass Integration
1. Add event registration to BaseODE
2. Add event registration to IVPLoop
3. Add event registration to OutputFunctions
4. Add event registration to SingleIntegratorRunCore
5. Update tests to verify compilation events

### Phase 3: Testing and Validation
1. Test with GPU available (compilation actually happens)
2. Test with CUDASIM (compilation timing skipped gracefully)
3. Test timing data appears in reports
4. Test aggregate durations by category
5. Validate performance overhead is acceptable

### Phase 4: Documentation
1. Update docstrings in TimeLogger
2. Update docstrings in CUDAFactory
3. Add examples to user guide (if exists)
4. Document 'compile' category in timing documentation

## Testing Strategy

### Unit Tests

**Test TimeLogger category validation:**
```python
def test_compile_category_accepted():
    logger = TimeLogger()
    logger._register_event("test", "compile", "Test compilation")
    assert "test" in logger._event_registry

def test_invalid_category_rejected():
    logger = TimeLogger()
    with pytest.raises(ValueError, match="category must be"):
        logger._register_event("test", "invalid", "Test")
```

**Test specialize_and_compile():**
```python
@pytest.mark.nocudasim
def test_specialize_and_compile_times_device_function():
    from numba import cuda
    
    @cuda.jit(device=True)
    def dummy_device(x, y):
        return x + y
    
    factory = MockCUDAFactory()
    factory._register_event("compile_dummy", "compile", "Test")
    
    factory.specialize_and_compile(dummy_device, "compile_dummy")
    
    duration = factory._timing_logger.get_event_duration("compile_dummy")
    assert duration is not None
    assert duration > 0
```

**Test signature introspection:**
```python
def test_get_device_function_params():
    @cuda.jit(device=True)
    def func(state, params, dt):
        pass
    
    params = get_device_function_params(func)
    assert params == ['state', 'params', 'dt']
```

### Integration Tests

**Test BaseODE compilation timing:**
```python
@pytest.mark.nocudasim
def test_baseode_records_compilation_timing(three_state_linear):
    logger = TimeLogger(verbosity='verbose')
    system = three_state_linear
    # Configure system to use logger
    
    # Trigger compilation
    _ = system.device_function
    
    # Check compilation event recorded
    assert logger.get_event_duration("compile_dxdt") is not None
```

**Test end-to-end with Solver:**
```python
@pytest.mark.nocudasim
def test_solver_shows_compilation_timing(three_state_linear):
    result = solve_ivp(
        three_state_linear,
        algorithm='euler',
        duration=1.0,
        time_logging_level='verbose'
    )
    
    # Compilation timing should be in output
    # (Verify via captured stdout or logger inspection)
```

### Edge Case Tests

**Test with simulator mode:**
```python
@pytest.mark.sim_only
def test_compilation_timing_skipped_in_simulator():
    logger = TimeLogger(verbosity='verbose')
    # Verify no compile events recorded (or recorded as zero/skipped)
```

**Test with None device function:**
```python
def test_skip_timing_for_none_function():
    factory = MockCUDAFactory()
    factory._register_event("compile_test", "compile", "Test")
    factory.specialize_and_compile(None, "compile_test")
    # Should not raise, should not record timing
```

## Performance Considerations

### Overhead Analysis

**One-time costs (per cache invalidation):**
- Signature introspection: microseconds
- Dummy argument creation: microseconds  
- Kernel compilation: 10-1000ms (what we're measuring)
- Kernel launch: microseconds
- Synchronization: microseconds (after compilation)

**Total overhead:** ~0.1-1% of compilation time (negligible)

**Recurring costs:** None (cached after first build)

### Optimization Opportunities

**Skip in production:**
- Set time_logging_level=None to disable all timing
- No overhead when logger is None (existing no-op callbacks)

**Lazy compilation option:**
- Could add flag to disable auto-compilation in _build()
- Allow manual compilation triggering for advanced users
- Defer to future enhancement if requested

## Security Considerations

### Code Injection Risk
**Risk:** Using exec() to generate kernels could enable code injection
**Mitigation:** Use closure-based approach instead of exec()
**Status:** Mitigated by design choice

### Resource Exhaustion
**Risk:** Creating many kernels could exhaust GPU memory
**Mitigation:** One kernel per device function, cached after first use
**Status:** Low risk (bounded by number of device functions)

### Information Disclosure
**Risk:** Timing information could leak sensitive model details
**Mitigation:** Timing is opt-in, user controls verbosity
**Status:** Low risk (user explicitly enables timing)
