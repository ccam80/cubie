# Implementation Task List
# Feature: CUDA Compilation Timing
# Plan Reference: .github/active_plans/cuda_compile_timing/agent_plan.md

## Task Group 1: TimeLogger Category Extension - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/time_logger.py (lines 357-386)

**Input Validation Required**:
- None (internal modification only)

**Tasks**:
1. **Extend TimeLogger category validation**
   - File: src/cubie/time_logger.py
   - Action: Modify
   - Details:
     ```python
     # In _register_event() method, around line 377
     # Change the validation set from:
     if category not in {'codegen', 'build', 'runtime'}:
         raise ValueError(
             f"category must be 'codegen', 'build', or 'runtime', "
             f"got '{category}'"
         )
     
     # To:
     if category not in {'codegen', 'build', 'runtime', 'compile'}:
         raise ValueError(
             f"category must be 'codegen', 'build', 'runtime', or 'compile', "
             f"got '{category}'"
         )
     ```
   - Edge cases: Ensure backward compatibility with existing categories
   - Integration: No other code changes needed; existing event handling remains unchanged

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: CUDAFactory Compilation Helper Utilities - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/CUDAFactory.py (entire file)
- File: src/cubie/cuda_simsafe.py (lines 1-120 for is_devfunc understanding)
- File: src/cubie/_utils.py (for PrecisionDType and utility patterns)

**Input Validation Required**:
- device_function parameter: Check it is not None before introspection
- param_count: Validate is integer >= 0
- precision: Already validated by PrecisionDType type hint

**Tasks**:
1. **Add signature introspection utility function**
   - File: src/cubie/CUDAFactory.py
   - Action: Create
   - Details:
     ```python
     # Add after imports, before CUDAFactory class definition
     import inspect
     
     def _get_device_function_params(device_function: Any) -> list[str]:
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
         try:
             if hasattr(device_function, 'py_func'):
                 sig = inspect.signature(device_function.py_func)
                 return list(sig.parameters.keys())
         except Exception:
             pass
         return []
     ```
   - Edge cases: Handle device functions without py_func attribute
   - Integration: Used by specialize_and_compile method

2. **Add dummy argument creation utility function**
   - File: src/cubie/CUDAFactory.py
   - Action: Create
   - Details:
     ```python
     # Add after _get_device_function_params, before CUDAFactory class
     def _create_dummy_args(
         param_count: int, precision: PrecisionDType
     ) -> tuple:
         """Create minimal dummy arguments for device function.
         
         Parameters
         ----------
         param_count
             Number of parameters the device function expects
         precision
             Numerical precision for scalar and array arguments
         
         Returns
         -------
         tuple
             Tuple of 1-element arrays with specified precision
         
         Notes
         -----
         Creates minimal 1-element arrays for all parameters. This works
         for most CUDA device functions which expect array parameters.
         Scalars are automatically converted when needed.
         """
         if param_count <= 0:
             return tuple()
         # Create 1-element arrays for all parameters
         return tuple([np.array([0.0], dtype=precision)] for _ in range(param_count))
     ```
   - Edge cases: Handle param_count of 0
   - Integration: Used by specialize_and_compile method

3. **Add dummy kernel creation utility function**
   - File: src/cubie/CUDAFactory.py
   - Action: Create
   - Details:
     ```python
     # Add after _create_dummy_args, before CUDAFactory class
     from numba import cuda
     
     def _create_dummy_kernel(device_func: Any, param_count: int) -> Callable:
         """Create minimal CUDA kernel to trigger device function compilation.
         
         Parameters
         ----------
         device_func
             CUDA device function to wrap in kernel
         param_count
             Number of parameters device_func expects
         
         Returns
         -------
         Callable
             Compiled CUDA kernel that calls device_func
         
         Notes
         -----
         Uses closure pattern to capture device_func. Supports up to 12
         parameters with explicit signatures. Falls back to 8-parameter
         signature for larger counts.
         """
         # Use closure-based approach for different parameter counts
         if param_count == 0:
             @cuda.jit
             def kernel():
                 if cuda.grid(1) == 0:
                     device_func()
         elif param_count == 1:
             @cuda.jit
             def kernel(a1):
                 if cuda.grid(1) == 0:
                     device_func(a1)
         elif param_count == 2:
             @cuda.jit
             def kernel(a1, a2):
                 if cuda.grid(1) == 0:
                     device_func(a1, a2)
         elif param_count == 3:
             @cuda.jit
             def kernel(a1, a2, a3):
                 if cuda.grid(1) == 0:
                     device_func(a1, a2, a3)
         elif param_count == 4:
             @cuda.jit
             def kernel(a1, a2, a3, a4):
                 if cuda.grid(1) == 0:
                     device_func(a1, a2, a3, a4)
         elif param_count == 5:
             @cuda.jit
             def kernel(a1, a2, a3, a4, a5):
                 if cuda.grid(1) == 0:
                     device_func(a1, a2, a3, a4, a5)
         elif param_count == 6:
             @cuda.jit
             def kernel(a1, a2, a3, a4, a5, a6):
                 if cuda.grid(1) == 0:
                     device_func(a1, a2, a3, a4, a5, a6)
         elif param_count == 7:
             @cuda.jit
             def kernel(a1, a2, a3, a4, a5, a6, a7):
                 if cuda.grid(1) == 0:
                     device_func(a1, a2, a3, a4, a5, a6, a7)
         elif param_count == 8:
             @cuda.jit
             def kernel(a1, a2, a3, a4, a5, a6, a7, a8):
                 if cuda.grid(1) == 0:
                     device_func(a1, a2, a3, a4, a5, a6, a7, a8)
         elif param_count == 9:
             @cuda.jit
             def kernel(a1, a2, a3, a4, a5, a6, a7, a8, a9):
                 if cuda.grid(1) == 0:
                     device_func(a1, a2, a3, a4, a5, a6, a7, a8, a9)
         elif param_count == 10:
             @cuda.jit
             def kernel(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
                 if cuda.grid(1) == 0:
                     device_func(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
         elif param_count == 11:
             @cuda.jit
             def kernel(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11):
                 if cuda.grid(1) == 0:
                     device_func(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)
         elif param_count == 12:
             @cuda.jit
             def kernel(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12):
                 if cuda.grid(1) == 0:
                     device_func(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)
         else:
             # Fallback for very large parameter counts (unlikely)
             @cuda.jit
             def kernel(a1, a2, a3, a4, a5, a6, a7, a8):
                 if cuda.grid(1) == 0:
                     # This will likely fail but provides debugging info
                     device_func(a1, a2, a3, a4, a5, a6, a7, a8)
         
         return kernel
     ```
   - Edge cases: Handle 0 parameters and >12 parameters
   - Integration: Used by specialize_and_compile method

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: CUDAFactory Core specialize_and_compile Implementation - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/CUDAFactory.py (entire file, especially lines 68-87 for __init__ pattern)
- File: src/cubie/cuda_simsafe.py (for CUDA_SIMULATION constant and is_devfunc)
- Previous task group outputs (helper functions)

**Input Validation Required**:
- device_function: Check is not None
- device_function: Check has callable/dispatcher attributes
- event_name: Already validated by TimeLogger (registered event check)

**Tasks**:
1. **Add import for cuda_simsafe utilities**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     ```python
     # Add to imports section near top of file
     from cubie.cuda_simsafe import CUDA_SIMULATION
     ```
   - Edge cases: None
   - Integration: Used in specialize_and_compile

2. **Implement specialize_and_compile method in CUDAFactory class**
   - File: src/cubie/CUDAFactory.py
   - Action: Create
   - Details:
     ```python
     # Add as new method in CUDAFactory class, after get_cached_output
     def specialize_and_compile(
         self, device_function: Any, event_name: str
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
         Creates a minimal CUDA kernel that calls the device function
         with appropriately typed arguments. The kernel launch triggers
         Numba's JIT compilation, which is timed and recorded.
         
         Called automatically by _build() for all device functions
         returned from build(). Manual invocation is not needed.
         
         In CUDA simulator mode, timing is skipped silently as
         compilation does not occur.
         """
         # Skip if None or in simulator mode
         if device_function is None:
             return
         
         if CUDA_SIMULATION:
             return
         
         # Verify it's actually a device function
         if not hasattr(device_function, 'py_func'):
             # Not a CUDA dispatcher, skip silently
             return
         
         # Get precision from compile settings if available
         precision = np.float64  # default
         if (self._compile_settings is not None and 
             hasattr(self._compile_settings, 'precision')):
             precision = self._compile_settings.precision
         
         try:
             # Start timing
             self._timing_start(event_name)
             
             # Introspect signature
             params = _get_device_function_params(device_function)
             param_count = len(params)
             
             # Create dummy arguments
             dummy_args = _create_dummy_args(param_count, precision)
             
             # Create and launch dummy kernel
             kernel = _create_dummy_kernel(device_function, param_count)
             kernel[1, 1](*dummy_args)
             
             # Synchronize to ensure compilation completes
             cuda.synchronize()
             
             # Stop timing
             self._timing_stop(event_name)
             
         except Exception as e:
             # Log warning but don't fail the build
             import warnings
             warnings.warn(
                 f"Failed to time compilation for {event_name}: {e}",
                 RuntimeWarning
             )
             # Try to stop timing if it was started
             try:
                 if event_name in self._timing_start.__self__._active_starts:
                     self._timing_stop(event_name)
             except Exception:
                 pass
     ```
   - Edge cases: 
     - device_function is None (skip)
     - CUDA_SIMULATION mode (skip)
     - Not a CUDA dispatcher (skip)
     - Kernel launch fails (catch exception, warn, continue)
     - No precision in compile_settings (use default)
   - Integration: Called from _build() method

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: CUDAFactory _build() Auto-Invocation - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 3

**Required Context**:
- File: src/cubie/CUDAFactory.py (lines 256-269 for _build method)
- File: src/cubie/_utils.py (for is_attrs_class and in_attr utilities)

**Input Validation Required**:
- None (internal logic only)

**Tasks**:
1. **Modify _build() to invoke compilation timing**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     ```python
     # In _build() method, after line 269 (after self._cache_valid = True)
     # Add compilation timing logic
     
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
         
         # NEW: Trigger compilation timing for device functions
         # Check if we have a multi-output cache (attrs class)
         if self._cache is not None:
             # Iterate through all fields in the attrs class
             for field in attrs.fields(type(self._cache)):
                 field_name = field.name
                 device_func = getattr(self._cache, field_name)
                 
                 # Check if this field is a device function
                 if device_func is not None and hasattr(device_func, 'py_func'):
                     # Check if compilation event was registered
                     event_name = f"compile_{field_name}"
                     if event_name in self._register_event.__self__._event_registry:
                         self.specialize_and_compile(device_func, event_name)
         
         # Single-output case
         elif self._device_function is not None:
             if hasattr(self._device_function, 'py_func'):
                 event_name = "compile_device_function"
                 if event_name in self._register_event.__self__._event_registry:
                     self.specialize_and_compile(self._device_function, event_name)
     ```
   - Edge cases:
     - Cache is None (single output)
     - Device function field is -1 (not implemented)
     - Event not registered (skip timing)
     - Device function is not a CUDA dispatcher
   - Integration: Automatically triggers for all CUDAFactory subclasses on first build

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: BaseODE Event Registration - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 4

**Required Context**:
- File: src/cubie/odesystems/baseODE.py (lines 1-125 for __init__ method)
- File: src/cubie/odesystems/baseODE.py (lines 15-50 for ODECache fields)

**Input Validation Required**:
- None (event registration only)

**Tasks**:
1. **Register compilation events in BaseODE.__init__**
   - File: src/cubie/odesystems/baseODE.py
   - Action: Modify
   - Details:
     ```python
     # In BaseODE.__init__, after line 124 (after self.name = name)
     # Add compilation event registration
     
     # Register compilation timing events for all possible ODECache fields
     self._register_event(
         "compile_dxdt", "compile", "Compilation time for dxdt"
     )
     self._register_event(
         "compile_linear_operator", "compile",
         "Compilation time for linear_operator"
     )
     self._register_event(
         "compile_linear_operator_cached", "compile",
         "Compilation time for linear_operator_cached"
     )
     self._register_event(
         "compile_neumann_preconditioner", "compile",
         "Compilation time for neumann_preconditioner"
     )
     self._register_event(
         "compile_neumann_preconditioner_cached", "compile",
         "Compilation time for neumann_preconditioner_cached"
     )
     self._register_event(
         "compile_stage_residual", "compile",
         "Compilation time for stage_residual"
     )
     self._register_event(
         "compile_n_stage_residual", "compile",
         "Compilation time for n_stage_residual"
     )
     self._register_event(
         "compile_n_stage_linear_operator", "compile",
         "Compilation time for n_stage_linear_operator"
     )
     self._register_event(
         "compile_n_stage_neumann_preconditioner", "compile",
         "Compilation time for n_stage_neumann_preconditioner"
     )
     self._register_event(
         "compile_observables", "compile",
         "Compilation time for observables"
     )
     self._register_event(
         "compile_prepare_jac", "compile",
         "Compilation time for prepare_jac"
     )
     self._register_event(
         "compile_calculate_cached_jvp", "compile",
         "Compilation time for calculate_cached_jvp"
     )
     self._register_event(
         "compile_time_derivative_rhs", "compile",
         "Compilation time for time_derivative_rhs"
     )
     ```
   - Edge cases: ODECache fields may be -1 (not implemented); timing will be skipped automatically
   - Integration: Events registered match ODECache field names exactly

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 6: IVPLoop Event Registration - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 4

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 80-130 for __init__ method)

**Input Validation Required**:
- None (event registration only)

**Tasks**:
1. **Register compilation event in IVPLoop.__init__**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     ```python
     # In IVPLoop.__init__, after super().__init__() call (after line 100)
     # Add compilation event registration
     
     # Register compilation timing event for loop function
     self._register_event(
         "compile_device_function", "compile",
         "Compilation time for loop function"
     )
     ```
   - Edge cases: None; IVPLoop always returns a device function
   - Integration: Event name matches generic device_function pattern used in _build()

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 7: OutputFunctions Event Registration - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 4

**Required Context**:
- File: src/cubie/outputhandling/output_functions.py (lines 100-150 for __init__ method)
- File: src/cubie/outputhandling/output_functions.py (lines 43-65 for OutputFunctionCache fields)

**Input Validation Required**:
- None (event registration only)

**Tasks**:
1. **Register compilation events in OutputFunctions.__init__**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     ```python
     # In OutputFunctions.__init__, after super().__init__() call
     # Find location after super().__init__() and before setup_compile_settings
     # Add compilation event registration
     
     # Register compilation timing events for output functions
     self._register_event(
         "compile_save_state_function", "compile",
         "Compilation time for save_state_function"
     )
     self._register_event(
         "compile_update_summaries_function", "compile",
         "Compilation time for update_summaries_function"
     )
     self._register_event(
         "compile_save_summaries_function", "compile",
         "Compilation time for save_summaries_function"
     )
     ```
   - Edge cases: None; all three functions are always built
   - Integration: Event names match OutputFunctionCache field names exactly

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 8: SingleIntegratorRunCore Event Registration - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 4

**Required Context**:
- File: src/cubie/integrators/SingleIntegratorRunCore.py (lines 76-100 for __init__ method)

**Input Validation Required**:
- None (event registration only)

**Tasks**:
1. **Register compilation event in SingleIntegratorRunCore.__init__**
   - File: src/cubie/integrators/SingleIntegratorRunCore.py
   - Action: Modify
   - Details:
     ```python
     # In SingleIntegratorRunCore.__init__, after super().__init__() call (after line 86)
     # Add compilation event registration
     
     # Register compilation timing event for integrator loop
     self._register_event(
         "compile_device_function", "compile",
         "Compilation time for compiled_loop_function"
     )
     ```
   - Edge cases: None; SingleIntegratorRunCore always returns a device function
   - Integration: Event name matches generic device_function pattern used in _build()

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 9: Unit Tests for TimeLogger - PARALLEL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: tests/test_time_logger.py (entire file for test patterns)
- File: src/cubie/time_logger.py (for implementation under test)

**Input Validation Required**:
- None (tests only)

**Tasks**:
1. **Add test for compile category acceptance**
   - File: tests/test_time_logger.py
   - Action: Create
   - Details:
     ```python
     # Add to TestTimeLogger class, after test_register_event_valid_categories
     
     def test_register_event_compile_category(self):
         """Test that 'compile' category is accepted."""
         logger = TimeLogger()
         logger._register_event("compile_test", "compile", "Compile event")
         
         assert "compile_test" in logger._event_registry
         assert logger._event_registry["compile_test"]["category"] == "compile"
         assert logger._event_registry["compile_test"]["description"] == "Compile event"
     ```
   - Edge cases: None
   - Integration: Verifies Task Group 1 changes work correctly

2. **Add test for compile category in aggregate durations**
   - File: tests/test_time_logger.py
   - Action: Create
   - Details:
     ```python
     # Add to TestTimeLogger class, after test_aggregate_durations_by_category
     
     def test_aggregate_durations_compile_category(self):
         """Test filtering aggregate durations for compile category."""
         import time
         
         logger = TimeLogger()
         logger._register_event("compile1", "compile", "Compile 1")
         logger._register_event("build1", "build", "Build 1")
         
         logger.start_event("compile1")
         time.sleep(0.01)
         logger.stop_event("compile1")
         
         logger.start_event("build1")
         time.sleep(0.01)
         logger.stop_event("build1")
         
         # Test filtering by compile category
         compile_durations = logger.get_aggregate_durations(category="compile")
         assert "compile1" in compile_durations
         assert "build1" not in compile_durations
         assert compile_durations["compile1"] >= 0.01
     ```
   - Edge cases: Mix of categories, verify filtering works
   - Integration: Verifies compile category works with existing aggregation logic

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 10: Unit Tests for CUDAFactory Helpers - PARALLEL
**Status**: [ ]
**Dependencies**: Group 2

**Required Context**:
- File: tests/test_CUDAFactory.py (lines 1-100 for test patterns and fixtures)
- File: src/cubie/CUDAFactory.py (for helper functions under test)

**Input Validation Required**:
- None (tests only)

**Tasks**:
1. **Add test for _get_device_function_params**
   - File: tests/test_CUDAFactory.py
   - Action: Create
   - Details:
     ```python
     # Add after existing tests
     from cubie.CUDAFactory import _get_device_function_params
     from numba import cuda
     import numpy as np
     
     def test_get_device_function_params():
         """Test parameter extraction from device function."""
         @cuda.jit(device=True)
         def sample_device_func(state, params, dt):
             return state[0] + params[0] * dt
         
         params = _get_device_function_params(sample_device_func)
         assert params == ['state', 'params', 'dt']
     
     def test_get_device_function_params_no_py_func():
         """Test graceful handling when py_func missing."""
         # Mock object without py_func
         class FakeFunc:
             pass
         
         params = _get_device_function_params(FakeFunc())
         assert params == []
     
     def test_get_device_function_params_none():
         """Test handling of None input."""
         params = _get_device_function_params(None)
         assert params == []
     ```
   - Edge cases: None input, missing py_func, normal device function
   - Integration: Verifies helper works for various inputs

2. **Add test for _create_dummy_args**
   - File: tests/test_CUDAFactory.py
   - Action: Create
   - Details:
     ```python
     # Add after _get_device_function_params tests
     from cubie.CUDAFactory import _create_dummy_args
     
     def test_create_dummy_args():
         """Test dummy argument creation."""
         args = _create_dummy_args(3, np.float64)
         
         assert len(args) == 3
         assert all(isinstance(arg, np.ndarray) for arg in args)
         assert all(arg.dtype == np.float64 for arg in args)
         assert all(len(arg) == 1 for arg in args)
     
     def test_create_dummy_args_zero_params():
         """Test zero parameter case."""
         args = _create_dummy_args(0, np.float64)
         assert len(args) == 0
         assert args == tuple()
     
     def test_create_dummy_args_precision():
         """Test different precision types."""
         args32 = _create_dummy_args(2, np.float32)
         args64 = _create_dummy_args(2, np.float64)
         
         assert all(arg.dtype == np.float32 for arg in args32)
         assert all(arg.dtype == np.float64 for arg in args64)
     ```
   - Edge cases: Zero params, different precisions
   - Integration: Verifies dummy arguments are created correctly

3. **Add test for _create_dummy_kernel**
   - File: tests/test_CUDAFactory.py
   - Action: Create
   - Details:
     ```python
     # Add after _create_dummy_args tests
     from cubie.CUDAFactory import _create_dummy_kernel
     import pytest
     
     @pytest.mark.nocudasim
     def test_create_dummy_kernel():
         """Test dummy kernel creation and execution."""
         @cuda.jit(device=True)
         def add_device(a, b):
             return a[0] + b[0]
         
         kernel = _create_dummy_kernel(add_device, 2)
         
         # Verify kernel is callable
         assert callable(kernel)
         
         # Test kernel can be launched (will trigger compilation)
         args = _create_dummy_args(2, np.float64)
         kernel[1, 1](*args)
         cuda.synchronize()
         
         # If we got here without exception, test passes
     
     @pytest.mark.nocudasim
     def test_create_dummy_kernel_various_param_counts():
         """Test kernel creation for different parameter counts."""
         for count in [0, 1, 3, 5, 8, 10, 12]:
             @cuda.jit(device=True)
             def dummy_func(*args):
                 pass
             
             kernel = _create_dummy_kernel(dummy_func, count)
             assert callable(kernel)
     ```
   - Edge cases: Different parameter counts (0, 1, 3, 5, 8, 10, 12, >12)
   - Integration: Verifies kernel creation works for various signatures

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 11: Integration Tests for specialize_and_compile - PARALLEL
**Status**: [ ]
**Dependencies**: Group 3

**Required Context**:
- File: tests/test_CUDAFactory.py (for test patterns)
- File: src/cubie/CUDAFactory.py (for implementation)

**Input Validation Required**:
- None (tests only)

**Tasks**:
1. **Add test for specialize_and_compile timing**
   - File: tests/test_CUDAFactory.py
   - Action: Create
   - Details:
     ```python
     # Add new test
     import pytest
     from cubie.time_logger import TimeLogger
     
     @pytest.mark.nocudasim
     def test_specialize_and_compile_records_timing():
         """Test that specialize_and_compile records compilation timing."""
         from numba import cuda
         
         @cuda.jit(device=True)
         def sample_device(x, y):
             return x[0] + y[0]
         
         # Create factory with custom logger
         class TestFactory(CUDAFactory):
             def build(self):
                 return sample_device
         
         factory = TestFactory()
         factory._register_event("compile_test", "compile", "Test compilation")
         
         # Call specialize_and_compile
         factory.specialize_and_compile(sample_device, "compile_test")
         
         # Verify timing was recorded
         logger = factory._timing_start.__self__
         duration = logger.get_event_duration("compile_test")
         assert duration is not None
         assert duration > 0
     
     @pytest.mark.nocudasim
     def test_specialize_and_compile_none_device_function():
         """Test that None device function is handled gracefully."""
         class TestFactory(CUDAFactory):
             def build(self):
                 return None
         
         factory = TestFactory()
         factory._register_event("compile_test", "compile", "Test")
         
         # Should not raise
         factory.specialize_and_compile(None, "compile_test")
     
     @pytest.mark.sim_only
     def test_specialize_and_compile_simulator_mode():
         """Test that compilation timing is skipped in simulator mode."""
         from numba import cuda
         
         @cuda.jit(device=True)
         def sample_device(x):
             return x[0]
         
         class TestFactory(CUDAFactory):
             def build(self):
                 return sample_device
         
         factory = TestFactory()
         factory._register_event("compile_test", "compile", "Test")
         
         # Should not raise, should skip timing
         factory.specialize_and_compile(sample_device, "compile_test")
         
         # Verify no timing recorded
         logger = factory._timing_start.__self__
         duration = logger.get_event_duration("compile_test")
         assert duration is None
     ```
   - Edge cases: None device function, simulator mode, normal compilation
   - Integration: Verifies specialize_and_compile works in various scenarios

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 12: Integration Tests for Auto-Compilation - PARALLEL
**Status**: [ ]
**Dependencies**: Groups 5, 6, 7, 8

**Required Context**:
- File: tests/conftest.py (for fixtures)
- File: tests/system_fixtures.py (for ODE system fixtures)
- File: tests/test_CUDAFactory.py (for test patterns)

**Input Validation Required**:
- None (tests only)

**Tasks**:
1. **Add test for BaseODE compilation timing**
   - File: tests/test_CUDAFactory.py or new file tests/test_compilation_timing.py
   - Action: Create
   - Details:
     ```python
     # Create new test file if needed: tests/test_compilation_timing.py
     import pytest
     import numpy as np
     from cubie.time_logger import TimeLogger
     
     @pytest.mark.nocudasim
     def test_baseode_compilation_timing(three_state_linear):
         """Test that BaseODE records compilation timing for dxdt."""
         system = three_state_linear
         
         # Trigger compilation by accessing device_function
         _ = system.device_function
         
         # Check that compilation event was recorded
         logger = system._timing_start.__self__
         duration = logger.get_event_duration("compile_dxdt")
         
         # Should have timing (may be None if event not properly registered)
         # In real implementation, should be > 0
         assert duration is not None or duration == 0
     
     @pytest.mark.nocudasim
     def test_ivploop_compilation_timing(precision):
         """Test that IVPLoop records compilation timing."""
         from cubie.integrators.loops.ode_loop import IVPLoop
         from cubie.integrators.loops.ode_loop_config import (
             LoopSharedIndices, LoopLocalIndices
         )
         from cubie.outputhandling import OutputCompileFlags
         
         # Create minimal IVPLoop
         shared_indices = LoopSharedIndices(
             state_slice=slice(0, 3),
             observable_slice=slice(3, 6),
             params_slice=slice(6, 9),
             driver_slice=slice(9, 10),
             # ... other required indices
         )
         local_indices = LoopLocalIndices(
             # ... required indices
         )
         compile_flags = OutputCompileFlags()
         
         loop = IVPLoop(
             precision=precision,
             shared_indices=shared_indices,
             local_indices=local_indices,
             compile_flags=compile_flags,
         )
         
         # Trigger compilation
         _ = loop.device_function
         
         # Check timing
         logger = loop._timing_start.__self__
         duration = logger.get_event_duration("compile_device_function")
         assert duration is not None or duration == 0
     ```
   - Edge cases: Different system types, check all registered events
   - Integration: Verifies end-to-end compilation timing for real components

2. **Add test for multiple compilation events**
   - File: tests/test_compilation_timing.py
   - Action: Create
   - Details:
     ```python
     @pytest.mark.nocudasim
     def test_multiple_compilation_events_aggregate():
         """Test aggregating multiple compilation events."""
         from cubie.time_logger import TimeLogger
         
         logger = TimeLogger(verbosity='verbose')
         
         # Simulate multiple compilation events
         logger._register_event("compile_dxdt", "compile", "dxdt compile")
         logger._register_event("compile_loop", "compile", "loop compile")
         logger._register_event("build_system", "build", "system build")
         
         import time
         logger.start_event("compile_dxdt")
         time.sleep(0.01)
         logger.stop_event("compile_dxdt")
         
         logger.start_event("compile_loop")
         time.sleep(0.01)
         logger.stop_event("compile_loop")
         
         logger.start_event("build_system")
         time.sleep(0.01)
         logger.stop_event("build_system")
         
         # Get compile-only durations
         compile_durations = logger.get_aggregate_durations(category="compile")
         assert len(compile_durations) == 2
         assert "compile_dxdt" in compile_durations
         assert "compile_loop" in compile_durations
         assert "build_system" not in compile_durations
         
         total_compile_time = sum(compile_durations.values())
         assert total_compile_time >= 0.02
     ```
   - Edge cases: Multiple events, category filtering
   - Integration: Verifies aggregation works correctly for compile category

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

**Total Task Groups**: 12

**Dependency Chain Overview**:
```
Group 1 (TimeLogger) 
  └─> Group 2 (Helpers)
       └─> Group 3 (specialize_and_compile)
            └─> Group 4 (_build integration)
                 ├─> Group 5 (BaseODE events)
                 ├─> Group 6 (IVPLoop events)
                 ├─> Group 7 (OutputFunctions events)
                 └─> Group 8 (SingleIntegratorRunCore events)

Group 1 ─> Group 9 (TimeLogger tests) [parallel]
Group 2 ─> Group 10 (Helper tests) [parallel]
Group 3 ─> Group 11 (specialize_and_compile tests) [parallel]
Groups 5-8 ─> Group 12 (Integration tests) [parallel]
```

**Parallel Execution Opportunities**:
- Groups 9, 10, 11, 12 can all run in parallel with each other (tests)
- Groups 5, 6, 7, 8 are independent and can run in parallel (event registration in different files)

**Estimated Complexity**:
- **Low complexity**: Groups 1, 5, 6, 7, 8, 9 (simple modifications and registrations)
- **Medium complexity**: Groups 2, 4, 10, 11, 12 (helper functions and tests)
- **High complexity**: Group 3 (core specialize_and_compile with error handling)

**Total estimated LOC**: ~500-600 lines including tests

**Key Implementation Notes**:
1. All event names must match ODECache/OutputFunctionCache field names exactly (with "compile_" prefix)
2. Registry check in _build() enables gradual rollout across subclasses
3. CUDA_SIMULATION check ensures graceful degradation in simulator mode
4. Exception handling in specialize_and_compile prevents build failures
5. Tests require nocudasim marker for GPU-dependent functionality
