# CUDA Local Array CUDASIM Fix - Agent Plan

## Problem Statement

Tests in `test_solver.py` and `test_solveresult.py` fail intermittently in CUDASIM mode with:

```
AttributeError: tid=[0, 13, 0] ctaid=[0, 0, 0]: module 'numba.cuda' has no attribute 'local'
```

The error occurs in `buffer_registry.py:136` and other locations where `cuda.local.array()` is called.

## Technical Context

### How Numba CUDASIM Works

1. When `NUMBA_ENABLE_CUDASIM=1`, Numba uses a simulator instead of real CUDA
2. The simulator provides fake implementations of CUDA intrinsics
3. At kernel execution time, Numba's `swapped_cuda_module` context manager temporarily replaces the `cuda` module in function globals with a fake version containing `local`, `shared`, `const`, etc.
4. After kernel execution, the original module is restored

### Why the Failure is Intermittent

The `swapped_cuda_module` function works by:
```python
@contextmanager
def swapped_cuda_module(fn, fake_cuda_module):
    fn_globs = fn.__globals__
    orig = dict((k, v) for k, v in fn_globs.items() if v is cuda)
    repl = dict((k, fake_cuda_module) for k, v in orig.items())
    fn_globs.update(repl)
    try:
        yield
    finally:
        fn_globs.update(orig)
```

The race condition occurs when:
1. The device function's `__globals__` dictionary lookup happens before the swap is complete
2. Or when thread timing causes the function to execute with an incomplete swap
3. Or when Python's reference caching returns the pre-swap module

This explains why:
- Same test passes in run 1, fails in run 2, passes in run 3
- The failure is tied to specific test execution order and timing
- The error includes thread ID information (`tid=[0, 13, 0]`)

### Files Using `cuda.local.array`

1. **`src/cubie/buffer_registry.py:136`** - Inside `CUDABuffer.build_allocator()`:
   ```python
   array = cuda.local.array(_local_size, _precision)
   ```

2. **`src/cubie/batchsolving/BatchSolverKernel.py:578`** - Inside kernel:
   ```python
   local_scratch = cuda.local.array(local_elements_per_run, dtype=float32)
   ```

3. **`src/cubie/integrators/loops/ode_loop.py:487`** - Inside loop device function:
   ```python
   proposed_counters = cuda.local.array(2, dtype=simsafe_int32)
   ```

4. **`src/cubie/integrators/matrix_free_solvers/newton_krylov.py:383`** - Inside solver:
   ```python
   krylov_iters_local = cuda.local.array(1, int32)
   ```

5. **`src/cubie/integrators/algorithms/generic_rosenbrock_w.py:448-449`** - Inside step:
   ```python
   base_state_placeholder = cuda.local.array(1, int32)
   krylov_iters_out = cuda.local.array(1, int32)
   ```

## Solution Architecture

### Component: `cuda_simsafe.local_array`

A wrapper function that provides consistent behavior across both CUDASIM and real CUDA modes.

**Behavior:**
- In CUDASIM mode (`CUDA_SIMULATION=True`): Returns `numpy.zeros(size, dtype=dtype)`
- In real CUDA mode (`CUDA_SIMULATION=False`): Calls `cuda.local.array(size, dtype)`

**Why This Works:**
- The `CUDA_SIMULATION` constant is evaluated at module import time
- The function is captured in closures at compile time (during `build()` methods)
- At runtime, the correct implementation is already selected
- No dependency on the `swapped_cuda_module` mechanism

### Implementation Pattern

The wrapper must be compatible with Numba's JIT compilation. Since the wrapper is called inside `@cuda.jit` decorated functions, it needs to either:

1. **Option A:** Be a `@cuda.jit(device=True)` function that internally uses `cuda.local.array`
2. **Option B:** Be selected at compile-time so only the appropriate code path is compiled

**Chosen: Option B** - Use compile-time selection via `if CUDA_SIMULATION` branch at build time, not inside the device function itself.

This means the solution modifies how build methods construct their closures, selecting either:
- A device function using `numpy.zeros` for CUDASIM
- A device function using `cuda.local.array` for real CUDA

## Detailed Component Descriptions

### 1. `cuda_simsafe.py` - New `local_array` Function

Add a factory function that creates the appropriate local array allocator based on CUDASIM status:

```python
def local_array_factory(size, dtype):
    """
    Factory returning appropriate local array allocator.
    
    In CUDASIM mode, returns numpy.zeros.
    In real CUDA mode, returns cuda.local.array result.
    
    Parameters
    ----------
    size : int
        Size of the local array
    dtype : numpy.dtype
        Data type for the array
        
    Returns
    -------
    function
        Allocator function suitable for use in device code
    """
```

However, this won't work directly because device functions can't call regular Python functions at runtime.

**Revised Approach:** Use a conditional import/function definition pattern at module level:

```python
if CUDA_SIMULATION:
    def _local_array_impl(size, dtype):
        return np.zeros(size, dtype=dtype)
else:
    def _local_array_impl(size, dtype):
        return cuda.local.array(size, dtype)

# Export for use in other modules
local_array = _local_array_impl
```

But this still won't work because `cuda.local.array` is a special intrinsic that only works inside device function scope.

**Final Approach:** The fix must be at the call sites where `cuda.local.array` is used. Each call site should:

1. Check `CUDA_SIMULATION` at compile time (in the `build()` method)
2. Generate different device function code based on the result

### 2. `buffer_registry.py` - `CUDABuffer.build_allocator()`

Current problematic code:
```python
@cuda.jit(device=True, inline=True, **compile_kwargs)
def allocate_buffer(shared, persistent):
    """Allocate buffer from appropriate memory region."""
    if _use_shared:
        array = shared[_shared_slice]
    elif _use_persistent:
        array = persistent[_persistent_slice]
    else:
        array = cuda.local.array(_local_size, _precision)  # PROBLEM
```

**Fix:** Import the CUDA_SIMULATION flag and create conditional device functions:

```python
from cubie.cuda_simsafe import CUDA_SIMULATION

# In build_allocator:
if CUDA_SIMULATION:
    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def allocate_buffer(shared, persistent):
        if _use_shared:
            array = shared[_shared_slice]
        elif _use_persistent:
            array = persistent[_persistent_slice]
        else:
            # CUDASIM: use numpy zeros instead
            array = np.zeros(_local_size, dtype=_precision)
        # ... rest
else:
    @cuda.jit(device=True, inline=True, **compile_kwargs)
    def allocate_buffer(shared, persistent):
        if _use_shared:
            array = shared[_shared_slice]
        elif _use_persistent:
            array = persistent[_persistent_slice]
        else:
            array = cuda.local.array(_local_size, _precision)
        # ... rest
```

This pattern duplicates code but guarantees correct behavior in both modes.

### 3. Other Call Sites

Apply the same conditional compilation pattern to:
- `BatchSolverKernel.build_kernel()`
- `IVPLoop.build()`
- `NewtonKrylov.build()`
- `GenericRosenbrockWStep.build_step()`

## Expected Behavior

### In CUDASIM Mode
- Device functions use `np.zeros()` for local array allocation
- No dependency on Numba's module swapping mechanism
- Tests pass deterministically

### In Real CUDA Mode
- Device functions use `cuda.local.array()` as before
- No performance regression
- No behavioral change

## Integration Points

### With Existing Code
- All changes are internal to build methods
- No API changes
- No changes to how device functions are called

### With Tests
- No test changes required
- Existing tests should pass in both modes

## Edge Cases

### 1. Zero-size Local Arrays
The `buffer_registry.py` already handles this:
```python
sizes[name] = max(entry.size, 1)  # cuda.local.array requires size >= 1
```
The numpy fallback should do the same.

### 2. Non-numeric Dtypes
Local arrays in the codebase use `float32`, `float64`, `int32` only. The numpy fallback handles these correctly.

### 3. Nested Device Function Calls
The wrapper pattern works correctly because:
- Each call site is handled independently
- The selection happens at compile time, not runtime

## Dependencies

### Imports Required
Each affected file needs:
```python
from cubie.cuda_simsafe import CUDA_SIMULATION
```

### Module Load Order
- `cuda_simsafe.py` must be importable before other modules
- `CUDA_SIMULATION` is set at import time from environment
- No circular import issues expected

## Validation Criteria

1. Run `NUMBA_ENABLE_CUDASIM=1 pytest tests/batchsolving/test_solver.py tests/batchsolving/test_solveresult.py` 10 times with 0 failures
2. Run `pytest tests/batchsolving/test_solver.py tests/batchsolving/test_solveresult.py` with real CUDA (if available) with 0 failures
3. No new linting errors from `flake8` or `ruff`
4. Code coverage maintained or improved
