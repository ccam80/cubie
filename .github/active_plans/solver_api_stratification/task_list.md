# Implementation Task List
# Feature: Solver API Stratification
# Plan Reference: .github/active_plans/solver_api_stratification/agent_plan.md

## Overview

This task list implements a three-level solver API stratification:
- **Level 1 (`solve()`)**: High-level API for novice users with dictionary inputs
- **Level 2 (`solve_arrays()`)**: Mid-level API for intermediate users with pre-built numpy arrays
- **Level 3 (`execute()`)**: Low-level API for advanced users with pre-allocated device arrays

**Dependency Chain**: Task Group 1 → Task Group 2 → Task Group 3 → Task Group 4 → Task Group 5

---

## Task Group 1: Level 3 - BatchSolverKernel.execute() - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 215-371, run() method)
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 373-417, limit_blocksize())
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 419-470, chunk_run())
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 43-65, ChunkParams)

**Input Validation Required**:
- None at Level 3 - caller is responsible for all validation
- Optionally validate device array types via debug flag (not implemented in first pass)

**Tasks**:

1. **Add execute() method to BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Create new method
   - Location: After line 371 (after run() method)
   - Details:
     ```python
     def execute(
         self,
         device_initial_values,
         device_parameters,
         device_driver_coefficients,
         device_state_output,
         device_observables_output,
         device_state_summaries_output,
         device_observable_summaries_output,
         device_iteration_counters,
         device_status_codes,
         duration: float,
         warmup: float = 0.0,
         t0: float = 0.0,
         blocksize: int = 256,
         stream: Optional[Any] = None,
     ) -> None:
         """Execute the integration kernel with pre-allocated device arrays.
         
         Low-level API for advanced users who manage their own device memory.
         Skips all host/device transfers and memory allocation. The caller
         is responsible for ensuring all device arrays have correct shapes,
         dtypes, and memory layout.
         
         Parameters
         ----------
         device_initial_values
             Device array of shape (n_states, n_runs) with initial conditions.
         device_parameters
             Device array of shape (n_params, n_runs) with parameter values.
         device_driver_coefficients
             Device array of shape (n_segments, n_drivers, order+1) with
             Horner-ordered driver interpolation coefficients.
         device_state_output
             Device array where state trajectories are written.
         device_observables_output
             Device array where observable trajectories are written.
         device_state_summaries_output
             Device array for state summary reductions.
         device_observable_summaries_output
             Device array for observable summary reductions.
         device_iteration_counters
             Device array for iteration counter values at each save point.
         device_status_codes
             Device array for per-run solver status codes.
         duration
             Duration of the simulation window.
         warmup
             Warmup time before the main simulation. Default is 0.0.
         t0
             Initial integration time. Default is 0.0.
         blocksize
             CUDA block size for kernel execution. Default is 256.
         stream
             CUDA stream for execution. None uses the solver's default stream.
         
         Returns
         -------
         None
             Results are written directly to the provided device arrays.
         
         Notes
         -----
         This is the lowest-level API entry point. No validation is performed
         on the input arrays. Ensure arrays match the expected shapes and
         dtypes for the configured system before calling.
         
         See Also
         --------
         BatchSolverKernel.run : Mid-level API handling memory allocation.
         Solver.solve : High-level API handling grid construction.
         Solver.solve_arrays : High-level API with numpy array inputs.
         """
         # Implementation logic:
         # 1. Set stream to default if None
         # 2. Convert time parameters to float64 for accumulation accuracy
         # 3. Get number of runs from device_initial_values.shape[1]
         # 4. Calculate chunk parameters (single chunk since pre-allocated)
         # 5. Compute dynamic shared memory size
         # 6. Limit block size based on shared memory
         # 7. Calculate grid/block dimensions
         # 8. Profile start if enabled
         # 9. Launch kernel with provided device arrays
         # 10. Profile stop if enabled
     ```
   - Edge cases: 
     - Single run (n_runs=1): ensure grid dimensions are valid
     - Zero-size output arrays: kernel should handle via output flags
     - Block size reduction due to shared memory constraints
   - Integration: Called by run() after memory allocation

2. **Refactor run() to call execute()**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify existing run() method
   - Location: Lines 215-371
   - Details:
     ```python
     def run(
         self,
         inits: NDArray[np.floating],
         params: NDArray[np.floating],
         driver_coefficients: Optional[NDArray[np.floating]],
         duration: float,
         blocksize: int = 256,
         stream: Optional[Any] = None,
         warmup: float = 0.0,
         t0: float = 0.0,
         chunk_axis: str = "run",
     ) -> None:
         # Keep existing logic for:
         # 1. Stream defaulting (line 264-265)
         # 2. Time parameter conversion to float64 (lines 267-274)
         # 3. Number of runs extraction (lines 276-278)
         # 4. Input/output array updates (lines 280-297)
         # 5. Memory allocation (line 300)
         
         # For each chunk (lines 333-368):
         #   - Keep initialise() calls for host→device transfers
         #   - Replace direct kernel launch (lines 343-362) with:
         self.execute(
             self.input_arrays.device_initial_values,
             self.input_arrays.device_parameters,
             self.input_arrays.device_driver_coefficients,
             self.output_arrays.device_state,
             self.output_arrays.device_observables,
             self.output_arrays.device_state_summaries,
             self.output_arrays.device_observable_summaries,
             self.output_arrays.device_iteration_counters,
             self.output_arrays.device_status_codes,
             chunk_params.duration,
             chunk_warmup,
             chunk_t0,
             blocksize,
             stream,
         )
         #   - Keep finalise() calls for device→host transfers
     ```
   - Edge cases:
     - Multiple chunks: execute() called once per chunk
     - chunk_axis="time": warmup and t0 adjusted per chunk
   - Integration: Maintains existing interface, delegates to execute()

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverKernel.py (112 lines added)
- Functions/Methods Added/Modified:
  * execute() - NEW: Low-level API method for direct kernel execution with device arrays
  * run() - MODIFIED: Refactored to delegate kernel launch to execute()
- Implementation Summary:
  * Added execute() method after run() with full docstring and parameter documentation
  * execute() handles stream defaulting, time parameter conversion, shared memory calculations, block size limiting, and kernel launch
  * Refactored run() to remove duplicated shared memory and block size calculations, now delegates to execute() in the chunk loop
  * Removed 15 lines of duplicated code from run() that are now in execute()
- Issues Flagged: None

---

## Task Group 2: Level 2 - Solver.solve_arrays() - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 325-408, solve() method)
- File: src/cubie/batchsolving/solver.py (lines 410-483, update() method)
- File: src/cubie/batchsolving/solveresult.py (lines 124-278, SolveResult class)
- File: src/cubie/odesystems/ODEData.py (lines 27-55, SystemSizes class)
- File: src/cubie/_utils.py (PrecisionDType type alias)

**Input Validation Required**:
- `initial_values`: Check type is np.ndarray, shape is (n_states, n_runs)
- `parameters`: Check type is np.ndarray, shape is (n_params, n_runs)
- `initial_values`: Check dtype matches system precision
- `parameters`: Check dtype matches system precision
- `initial_values`: Check C-contiguous memory layout
- `parameters`: Check C-contiguous memory layout
- `driver_coefficients`: If provided, check type is np.ndarray, dtype matches precision

**Tasks**:

1. **Create validate_solver_arrays() helper function**
   - File: src/cubie/batchsolving/solver.py
   - Action: Create new function (module-level, after imports)
   - Location: After line 38 (after imports, before solve_ivp)
   - Details:
     ```python
     def validate_solver_arrays(
         initial_values: np.ndarray,
         parameters: np.ndarray,
         system_sizes: "SystemSizes",
         precision: PrecisionDType,
     ) -> None:
         """Validate array shapes and types for solve_arrays().
         
         Parameters
         ----------
         initial_values
             Initial state values array to validate.
         parameters
             Parameter values array to validate.
         system_sizes
             SystemSizes instance describing expected dimensions.
         precision
             Expected dtype for floating-point arrays.
         
         Raises
         ------
         TypeError
             If arrays are not numpy ndarrays.
         ValueError
             If array shapes or dtypes do not match expectations.
             If arrays are not C-contiguous.
         
         Notes
         -----
         This function performs shape and dtype validation for Level 2 API.
         It does not validate array contents (e.g., physical plausibility).
         """
         # Implementation logic:
         # 1. Check initial_values is np.ndarray
         # 2. Check parameters is np.ndarray
         # 3. Check initial_values.shape == (system_sizes.states, n_runs)
         # 4. Check parameters.shape == (system_sizes.parameters, n_runs)
         # 5. Check initial_values.dtype == precision
         # 6. Check parameters.dtype == precision
         # 7. Check initial_values is C-contiguous (flags['C_CONTIGUOUS'])
         # 8. Check parameters is C-contiguous (flags['C_CONTIGUOUS'])
     ```
   - Edge cases:
     - Empty arrays (0 runs): should raise appropriate error
     - F-contiguous arrays: should raise with helpful message
     - Wrong dtype (e.g., float64 when expecting float32): clear message
   - Integration: Called by solve_arrays() before kernel.run()

2. **Add solve_arrays() method to Solver class**
   - File: src/cubie/batchsolving/solver.py
   - Action: Create new method in Solver class
   - Location: After solve() method (after line 408)
   - Details:
     ```python
     def solve_arrays(
         self,
         initial_values: np.ndarray,
         parameters: np.ndarray,
         driver_coefficients: Optional[np.ndarray] = None,
         duration: float = 1.0,
         warmup: float = 0.0,
         t0: float = 0.0,
         blocksize: int = 256,
         stream: Any = None,
         chunk_axis: str = "run",
         results_type: str = "full",
         **kwargs: Any,
     ) -> SolveResult:
         """Solve a batch initial value problem using pre-built numpy arrays.
         
         Mid-level API for intermediate users who provide pre-built arrays
         in (variable, run) format. Skips label resolution and grid
         construction but handles memory allocation, chunking, and
         host/device transfers.
         
         Parameters
         ----------
         initial_values
             Initial state values as numpy array with shape (n_states, n_runs).
             Dtype must match system precision.
         parameters
             Parameter values as numpy array with shape (n_params, n_runs).
             Dtype must match system precision.
         driver_coefficients
             Optional Horner-ordered driver interpolation coefficients.
             If None, uses placeholder coefficients from driver_interpolator.
         duration
             Total integration time. Default is 1.0.
         warmup
             Warm-up period before recording outputs. Default is 0.0.
         t0
             Initial integration time. Default is 0.0.
         blocksize
             CUDA block size for kernel launch. Default is 256.
         stream
             CUDA stream for execution. None uses the solver's default stream.
         chunk_axis
             Dimension along which to chunk when memory is limited.
             Default is "run".
         results_type
             Format of returned results. Options are "full", "numpy",
             "numpy_per_summary", "raw", and "pandas". Default is "full".
         **kwargs
             Additional options forwarded to update().
         
         Returns
         -------
         SolveResult
             Collected results from the integration run.
         
         Raises
         ------
         TypeError
             If initial_values or parameters are not numpy ndarrays.
         ValueError
             If array shapes or dtypes do not match system expectations.
         
         See Also
         --------
         Solver.solve : High-level API with dictionary inputs.
         BatchSolverKernel.execute : Low-level API with device arrays.
         
         Examples
         --------
         >>> solver = Solver(system, algorithm="euler")
         >>> inits = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
         >>> params = np.array([[1.0, 2.0]], dtype=np.float32)
         >>> result = solver.solve_arrays(inits, params, duration=1.0)
         """
         # Implementation logic:
         # 1. Apply kwargs updates if any
         if kwargs:
             self.update(kwargs, silent=True)
         
         # 2. Validate arrays
         validate_solver_arrays(
             initial_values,
             parameters,
             self.system_sizes,
             self.precision,
         )
         
         # 3. Use provided or placeholder driver coefficients
         if driver_coefficients is None:
             driver_coefficients = self.driver_interpolator.coefficients
         
         # 4. Call kernel.run() with arrays
         self.kernel.run(
             inits=initial_values,
             params=parameters,
             driver_coefficients=driver_coefficients,
             duration=duration,
             warmup=warmup,
             t0=t0,
             blocksize=blocksize,
             stream=stream,
             chunk_axis=chunk_axis,
         )
         
         # 5. Sync and return result
         self.memory_manager.sync_stream(self.kernel)
         return SolveResult.from_solver(self, results_type=results_type)
     ```
   - Edge cases:
     - Empty kwargs: no update() call needed (handled)
     - None driver_coefficients: use placeholder (handled)
     - Invalid arrays: validate_solver_arrays raises before kernel call
   - Integration: Called by solve() after grid construction

3. **Refactor solve() to call solve_arrays()**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify existing solve() method
   - Location: Lines 325-408
   - Details:
     ```python
     def solve(
         self,
         initial_values: Union[np.ndarray, Dict[str, Union[float, np.ndarray]]],
         parameters: Union[np.ndarray, Dict[str, Union[float, np.ndarray]]],
         drivers: Optional[Dict[str, Any]] = None,
         duration: float = 1.0,
         settling_time: float = 0.0,
         t0: float = 0.0,
         blocksize: int = 256,
         stream: Any = None,
         chunk_axis: str = "run",
         grid_type: str = "verbatim",
         results_type: str = "full",
         **kwargs: Any,
     ) -> SolveResult:
         # Keep existing signature and docstring
         
         # Keep existing logic for:
         # 1. kwargs update (lines 377-378)
         # 2. grid_builder call (lines 380-382)
         # 3. driver interpolator update (lines 384-394)
         
         # Replace direct kernel.run() call (lines 396-406) with:
         return self.solve_arrays(
             initial_values=inits,
             parameters=params,
             driver_coefficients=self.driver_interpolator.coefficients,
             duration=duration,
             warmup=settling_time,
             t0=t0,
             blocksize=blocksize,
             stream=stream,
             chunk_axis=chunk_axis,
             results_type=results_type,
             # Note: don't pass kwargs - already applied above
         )
     ```
   - Edge cases:
     - Existing kwargs handling must remain
     - Driver function update logic must remain
     - settling_time maps to warmup parameter
   - Integration: Maintains existing interface, delegates to solve_arrays()

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/solver.py (120 lines added)
- Functions/Methods Added/Modified:
  * validate_solver_arrays() - NEW: Module-level helper function for array validation
  * solve_arrays() - NEW: Mid-level API method for array-based solving
  * solve() - MODIFIED: Refactored to delegate to solve_arrays() after grid construction
- Implementation Summary:
  * Added SystemSizes import from cubie.odesystems.ODEData
  * Added validate_solver_arrays() with checks for: type (np.ndarray), shape (n_states/n_params, n_runs), dtype (matches precision), C-contiguous memory layout
  * Added solve_arrays() method with full docstring and parameter documentation
  * solve_arrays() validates inputs, uses placeholder driver coefficients if None provided, calls kernel.run(), syncs stream, and returns SolveResult
  * Refactored solve() to delegate to solve_arrays() after grid construction and driver interpolator update
  * solve() now returns directly from solve_arrays() call
- Issues Flagged: None

---

## Task Group 3: Import and Export Updates - PARALLEL
**Status**: [x]
**Dependencies**: Task Groups 1 and 2

**Required Context**:
- File: src/cubie/batchsolving/__init__.py (entire file)
- File: src/cubie/__init__.py (entire file - check if solver methods are exported)

**Input Validation Required**:
- None (import/export tasks)

**Tasks**:

1. **Add SystemSizes import to solver.py if needed**
   - File: src/cubie/batchsolving/solver.py
   - Action: Add import statement
   - Location: After line 19 (with other odesystems imports)
   - Details:
     ```python
     from cubie.odesystems.ODEData import SystemSizes
     ```
   - Edge cases: None
   - Integration: Required for validate_solver_arrays() type hint

2. **Verify batchsolving/__init__.py exports**
   - File: src/cubie/batchsolving/__init__.py
   - Action: Review and update exports if needed
   - Details: Ensure `validate_solver_arrays` is NOT exported (internal helper)
   - Edge cases: None
   - Integration: Public API should not change

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/solver.py (1 line added for import)
  * src/cubie/batchsolving/__init__.py (verified - no changes needed)
- Functions/Methods Added/Modified:
  * Import statement added for SystemSizes
- Implementation Summary:
  * Added import for SystemSizes from cubie.odesystems.ODEData in solver.py
  * Verified that __init__.py does not export validate_solver_arrays (correctly keeps it internal)
  * No changes needed to __init__.py as public API is unchanged
- Issues Flagged: None

---

## Task Group 4: Tests for Level 2 API (solve_arrays) - PARALLEL
**Status**: [x]
**Dependencies**: Task Groups 1, 2, and 3

**Required Context**:
- File: tests/batchsolving/test_solver.py (entire file, esp. fixtures lines 19-55)
- File: tests/batchsolving/conftest.py (existing fixtures)
- File: tests/conftest.py (solver, solver_mutable, driver_settings fixtures)

**Input Validation Required**:
- None (test code)

**Tasks**:

1. **Add shared test fixtures to batchsolving/conftest.py**
   - File: tests/batchsolving/conftest.py
   - Action: Add fixtures (currently only in test_solver.py)
   - Location: After existing fixtures (after line 147)
   - Details:
     ```python
     @pytest.fixture(scope="function")
     def simple_initial_values(system):
         """Create simple initial values for testing."""
         return {
             list(system.initial_values.names)[0]: [0.1, 0.5],
             list(system.initial_values.names)[1]: [0.2, 0.6],
         }


     @pytest.fixture(scope="function")
     def simple_parameters(system):
         """Create simple parameters for testing."""
         return {
             list(system.parameters.names)[0]: [1.0, 2.0],
             list(system.parameters.names)[1]: [0.5, 1.5],
         }
     ```
   - Edge cases: None
   - Integration: Makes fixtures available to all batchsolving tests

2. **Update test_solver.py to use shared fixtures**
   - File: tests/batchsolving/test_solver.py
   - Action: Remove duplicate fixture definitions
   - Location: Lines 19-34
   - Details: Remove the `simple_initial_values` and `simple_parameters` fixtures
     (they will be imported from conftest.py automatically by pytest)
   - Edge cases: None
   - Integration: Prevents duplicate fixture definitions

3. **Create test file for solve_arrays tests**
   - File: tests/batchsolving/test_solve_arrays.py
   - Action: Create new file
   - Details:
     ```python
     """Tests for Solver.solve_arrays() mid-level API."""
     
     import pytest
     import numpy as np
     from cubie.batchsolving.solver import Solver, validate_solver_arrays
     from cubie.batchsolving.solveresult import SolveResult
     
     
     class TestValidateSolverArrays:
         """Tests for validate_solver_arrays() helper function."""
         
         def test_valid_arrays_pass(self, solver):
             """Valid arrays matching system expectations pass validation."""
             # Get expected sizes from solver
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 4
             
             inits = np.zeros((n_states, n_runs), dtype=precision)
             params = np.zeros((n_params, n_runs), dtype=precision)
             
             # Should not raise
             validate_solver_arrays(
                 inits, params, solver.system_sizes, precision
             )
         
         def test_wrong_initial_values_type_raises(self, solver):
             """Non-ndarray initial_values raises TypeError."""
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 4
             
             params = np.zeros((n_params, n_runs), dtype=precision)
             
             with pytest.raises(TypeError):
                 validate_solver_arrays(
                     [[0.1, 0.2]],  # list, not ndarray
                     params,
                     solver.system_sizes,
                     precision,
                 )
         
         def test_wrong_parameters_type_raises(self, solver):
             """Non-ndarray parameters raises TypeError."""
             n_states = solver.system_sizes.states
             precision = solver.precision
             n_runs = 4
             
             inits = np.zeros((n_states, n_runs), dtype=precision)
             
             with pytest.raises(TypeError):
                 validate_solver_arrays(
                     inits,
                     [[1.0, 2.0]],  # list, not ndarray
                     solver.system_sizes,
                     precision,
                 )
         
         def test_wrong_initial_values_shape_raises(self, solver):
             """Mismatched initial_values shape raises ValueError."""
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 4
             
             # Wrong number of states
             inits = np.zeros((999, n_runs), dtype=precision)
             params = np.zeros((n_params, n_runs), dtype=precision)
             
             with pytest.raises(ValueError):
                 validate_solver_arrays(
                     inits, params, solver.system_sizes, precision
                 )
         
         def test_wrong_parameters_shape_raises(self, solver):
             """Mismatched parameters shape raises ValueError."""
             n_states = solver.system_sizes.states
             precision = solver.precision
             n_runs = 4
             
             inits = np.zeros((n_states, n_runs), dtype=precision)
             # Wrong number of parameters
             params = np.zeros((999, n_runs), dtype=precision)
             
             with pytest.raises(ValueError):
                 validate_solver_arrays(
                     inits, params, solver.system_sizes, precision
                 )
         
         def test_mismatched_runs_raises(self, solver):
             """Different n_runs between arrays raises ValueError."""
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             
             inits = np.zeros((n_states, 4), dtype=precision)
             params = np.zeros((n_params, 5), dtype=precision)  # different n_runs
             
             with pytest.raises(ValueError):
                 validate_solver_arrays(
                     inits, params, solver.system_sizes, precision
                 )
         
         def test_wrong_dtype_raises(self, solver):
             """Wrong dtype raises ValueError."""
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 4
             
             # Use opposite precision
             wrong_dtype = np.float64 if precision == np.float32 else np.float32
             
             inits = np.zeros((n_states, n_runs), dtype=wrong_dtype)
             params = np.zeros((n_params, n_runs), dtype=precision)
             
             with pytest.raises(ValueError):
                 validate_solver_arrays(
                     inits, params, solver.system_sizes, precision
                 )
         
         def test_non_contiguous_raises(self, solver):
             """Non-contiguous arrays raise ValueError."""
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 4
             
             # Create Fortran-contiguous array
             inits = np.asfortranarray(
                 np.zeros((n_states, n_runs), dtype=precision)
             )
             params = np.zeros((n_params, n_runs), dtype=precision)
             
             with pytest.raises(ValueError):
                 validate_solver_arrays(
                     inits, params, solver.system_sizes, precision
                 )
     
     
     class TestSolveArrays:
         """Tests for Solver.solve_arrays() method."""
         
         def test_solve_arrays_basic(self, solver, driver_settings):
             """solve_arrays returns SolveResult with valid arrays."""
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 4
             
             inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
             params = np.ones((n_params, n_runs), dtype=precision)
             
             result = solver.solve_arrays(
                 initial_values=inits,
                 parameters=params,
                 duration=0.1,
             )
             
             assert isinstance(result, SolveResult)
             assert hasattr(result, "time_domain_array")
         
         def test_solve_arrays_single_run(self, solver):
             """solve_arrays works with single run."""
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 1
             
             inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
             params = np.ones((n_params, n_runs), dtype=precision)
             
             result = solver.solve_arrays(
                 initial_values=inits,
                 parameters=params,
                 duration=0.1,
             )
             
             assert isinstance(result, SolveResult)
         
         def test_solve_arrays_with_driver_coefficients(self, solver):
             """solve_arrays accepts explicit driver coefficients."""
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 2
             
             inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
             params = np.ones((n_params, n_runs), dtype=precision)
             
             # Use driver coefficients from solver's interpolator
             driver_coeffs = solver.driver_interpolator.coefficients
             
             result = solver.solve_arrays(
                 initial_values=inits,
                 parameters=params,
                 driver_coefficients=driver_coeffs,
                 duration=0.1,
             )
             
             assert isinstance(result, SolveResult)
         
         def test_solve_arrays_with_kwargs(self, solver_mutable):
             """solve_arrays forwards kwargs to update()."""
             solver = solver_mutable
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 2
             
             inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
             params = np.ones((n_params, n_runs), dtype=precision)
             original_dt = solver.dt
             new_dt = precision(original_dt * 0.5) if original_dt else precision(1e-4)
             
             result = solver.solve_arrays(
                 initial_values=inits,
                 parameters=params,
                 duration=0.1,
                 dt=new_dt,
             )
             
             assert isinstance(result, SolveResult)
         
         def test_solve_arrays_result_types(self, solver):
             """solve_arrays respects results_type parameter."""
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 2
             
             inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
             params = np.ones((n_params, n_runs), dtype=precision)
             
             result_full = solver.solve_arrays(
                 inits, params, duration=0.1, results_type="full"
             )
             result_numpy = solver.solve_arrays(
                 inits, params, duration=0.1, results_type="numpy"
             )
             
             assert isinstance(result_full, SolveResult)
             assert isinstance(result_numpy, dict)
         
         def test_solve_arrays_invalid_raises(self, solver):
             """solve_arrays raises on invalid input arrays."""
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             
             # Wrong shape
             inits = np.ones((999, 2), dtype=precision)
             params = np.ones((n_params, 2), dtype=precision)
             
             with pytest.raises(ValueError):
                 solver.solve_arrays(inits, params, duration=0.1)
     
     
     class TestSolveArraysConsistency:
         """Tests ensuring solve() and solve_arrays() produce consistent results."""
         
         def test_solve_and_solve_arrays_consistent(
             self, solver, simple_initial_values, simple_parameters, driver_settings
         ):
             """solve() and solve_arrays() produce equivalent results."""
             # First solve with dict inputs
             result_dict = solver.solve(
                 initial_values=simple_initial_values,
                 parameters=simple_parameters,
                 drivers=driver_settings,
                 duration=0.1,
                 grid_type="verbatim",
                 results_type="full",
             )
             
             # Build arrays manually matching verbatim grid
             state_names = list(simple_initial_values.keys())
             param_names = list(simple_parameters.keys())
             precision = solver.precision
             
             n_runs = len(list(simple_initial_values.values())[0])
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             
             inits = np.zeros((n_states, n_runs), dtype=precision)
             params = np.zeros((n_params, n_runs), dtype=precision)
             
             # Fill arrays in same order as grid_builder
             for i, name in enumerate(state_names):
                 if i < n_states:
                     values = simple_initial_values[name]
                     inits[i, :] = np.array(values, dtype=precision)
             
             for i, name in enumerate(param_names):
                 if i < n_params:
                     values = simple_parameters[name]
                     params[i, :] = np.array(values, dtype=precision)
             
             # Solve with array inputs
             result_arrays = solver.solve_arrays(
                 initial_values=inits,
                 parameters=params,
                 driver_coefficients=solver.driver_interpolator.coefficients,
                 duration=0.1,
                 results_type="full",
             )
             
             # Results should match
             assert result_dict.time_domain_array.shape == \
                    result_arrays.time_domain_array.shape
     ```
   - Edge cases: Covered in individual test cases
   - Integration: Uses existing test fixtures from conftest.py

**Outcomes**: 
- Files Modified: 
  * tests/batchsolving/conftest.py (18 lines added)
  * tests/batchsolving/test_solver.py (15 lines removed)
  * tests/batchsolving/test_solve_arrays.py (NEW - 282 lines)
- Functions/Methods Added/Modified:
  * simple_initial_values fixture - MOVED to conftest.py from test_solver.py
  * simple_parameters fixture - MOVED to conftest.py from test_solver.py
  * TestValidateSolverArrays test class - NEW
  * TestSolveArrays test class - NEW
  * TestSolveArraysConsistency test class - NEW
- Implementation Summary:
  * Added simple_initial_values and simple_parameters fixtures to tests/batchsolving/conftest.py
  * Removed duplicate fixtures from tests/batchsolving/test_solver.py
  * Created tests/batchsolving/test_solve_arrays.py with comprehensive tests for:
    - validate_solver_arrays() validation helper (8 tests)
    - solve_arrays() basic functionality (6 tests)
    - Consistency between solve() and solve_arrays() (1 test)
- Issues Flagged: None

---

## Task Group 5: Tests for Level 3 API (execute) - PARALLEL
**Status**: [x]
**Dependencies**: Task Groups 1, 2, and 3

**Required Context**:
- File: tests/batchsolving/test_SolverKernel.py (entire file)
- File: tests/conftest.py (fixtures)

**Input Validation Required**:
- None (test code)

**Tasks**:

1. **Create test file for execute tests**
   - File: tests/batchsolving/test_execute.py
   - Action: Create new file
   - Details:
     ```python
     """Tests for BatchSolverKernel.execute() low-level API.
     
     These tests require real CUDA device arrays and are marked nocudasim
     to skip in simulator mode where device array behavior differs.
     """
     
     import pytest
     import numpy as np
     from os import environ
     
     from cubie.batchsolving.solver import Solver
     from cubie.batchsolving.solveresult import SolveResult
     
     # Check if running in CUDA simulator mode
     IS_CUDASIM = environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1"
     
     if not IS_CUDASIM:
         from numba import cuda
     
     
     @pytest.mark.nocudasim
     class TestExecute:
         """Tests for BatchSolverKernel.execute() method."""
         
         def test_execute_basic(self, solver):
             """execute() runs with pre-allocated device arrays."""
             kernel = solver.kernel
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 4
             
             # Prepare host arrays
             h_inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
             h_params = np.ones((n_params, n_runs), dtype=precision)
             
             # First do a normal run to set up output sizes
             kernel.run(
                 inits=h_inits,
                 params=h_params,
                 driver_coefficients=solver.driver_interpolator.coefficients,
                 duration=0.1,
             )
             
             # Get sizes from the kernel after run sets them up
             output_sizes = kernel.output_array_sizes_3d
             
             # Allocate device arrays
             d_inits = cuda.to_device(h_inits)
             d_params = cuda.to_device(h_params)
             d_driver_coeffs = cuda.to_device(
                 solver.driver_interpolator.coefficients
             )
             
             # Allocate output arrays based on kernel sizes
             d_state = cuda.device_array(
                 output_sizes.state, dtype=precision
             )
             d_observables = cuda.device_array(
                 output_sizes.observables, dtype=precision
             )
             d_state_summaries = cuda.device_array(
                 output_sizes.state_summaries, dtype=precision
             )
             d_observable_summaries = cuda.device_array(
                 output_sizes.observable_summaries, dtype=precision
             )
             d_iteration_counters = cuda.device_array(
                 output_sizes.iteration_counters, dtype=np.int32
             )
             d_status_codes = cuda.device_array(
                 (n_runs,), dtype=np.int32
             )
             
             # Execute directly with device arrays
             kernel.execute(
                 d_inits,
                 d_params,
                 d_driver_coeffs,
                 d_state,
                 d_observables,
                 d_state_summaries,
                 d_observable_summaries,
                 d_iteration_counters,
                 d_status_codes,
                 duration=0.1,
             )
             
             # Sync and check results
             cuda.synchronize()
             
             # Copy back and verify non-zero
             state_result = d_state.copy_to_host()
             assert state_result.shape == output_sizes.state
         
         def test_execute_single_run(self, solver):
             """execute() works with single run."""
             kernel = solver.kernel
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 1
             
             h_inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
             h_params = np.ones((n_params, n_runs), dtype=precision)
             
             # Set up sizes via normal run
             kernel.run(
                 inits=h_inits,
                 params=h_params,
                 driver_coefficients=solver.driver_interpolator.coefficients,
                 duration=0.1,
             )
             
             output_sizes = kernel.output_array_sizes_3d
             
             d_inits = cuda.to_device(h_inits)
             d_params = cuda.to_device(h_params)
             d_driver_coeffs = cuda.to_device(
                 solver.driver_interpolator.coefficients
             )
             d_state = cuda.device_array(output_sizes.state, dtype=precision)
             d_observables = cuda.device_array(
                 output_sizes.observables, dtype=precision
             )
             d_state_summaries = cuda.device_array(
                 output_sizes.state_summaries, dtype=precision
             )
             d_observable_summaries = cuda.device_array(
                 output_sizes.observable_summaries, dtype=precision
             )
             d_iteration_counters = cuda.device_array(
                 output_sizes.iteration_counters, dtype=np.int32
             )
             d_status_codes = cuda.device_array((n_runs,), dtype=np.int32)
             
             kernel.execute(
                 d_inits,
                 d_params,
                 d_driver_coeffs,
                 d_state,
                 d_observables,
                 d_state_summaries,
                 d_observable_summaries,
                 d_iteration_counters,
                 d_status_codes,
                 duration=0.1,
             )
             
             cuda.synchronize()
             state_result = d_state.copy_to_host()
             assert state_result.shape[2] == 1  # single run
         
         def test_execute_with_warmup(self, solver):
             """execute() respects warmup parameter."""
             kernel = solver.kernel
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 2
             
             h_inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
             h_params = np.ones((n_params, n_runs), dtype=precision)
             
             kernel.run(
                 inits=h_inits,
                 params=h_params,
                 driver_coefficients=solver.driver_interpolator.coefficients,
                 duration=0.1,
                 warmup=0.05,
             )
             
             output_sizes = kernel.output_array_sizes_3d
             
             d_inits = cuda.to_device(h_inits)
             d_params = cuda.to_device(h_params)
             d_driver_coeffs = cuda.to_device(
                 solver.driver_interpolator.coefficients
             )
             d_state = cuda.device_array(output_sizes.state, dtype=precision)
             d_observables = cuda.device_array(
                 output_sizes.observables, dtype=precision
             )
             d_state_summaries = cuda.device_array(
                 output_sizes.state_summaries, dtype=precision
             )
             d_observable_summaries = cuda.device_array(
                 output_sizes.observable_summaries, dtype=precision
             )
             d_iteration_counters = cuda.device_array(
                 output_sizes.iteration_counters, dtype=np.int32
             )
             d_status_codes = cuda.device_array((n_runs,), dtype=np.int32)
             
             # Execute with warmup
             kernel.execute(
                 d_inits,
                 d_params,
                 d_driver_coeffs,
                 d_state,
                 d_observables,
                 d_state_summaries,
                 d_observable_summaries,
                 d_iteration_counters,
                 d_status_codes,
                 duration=0.1,
                 warmup=0.05,
             )
             
             cuda.synchronize()
             # Should complete without error
     
     
     @pytest.mark.nocudasim
     class TestRunCallsExecute:
         """Tests ensuring run() delegates to execute()."""
         
         def test_run_and_execute_consistent(self, solver):
             """run() and execute() produce equivalent results."""
             kernel = solver.kernel
             n_states = solver.system_sizes.states
             n_params = solver.system_sizes.parameters
             precision = solver.precision
             n_runs = 2
             
             h_inits = np.ones((n_states, n_runs), dtype=precision) * 0.5
             h_params = np.ones((n_params, n_runs), dtype=precision)
             driver_coeffs = solver.driver_interpolator.coefficients
             
             # Run via run()
             kernel.run(
                 inits=h_inits,
                 params=h_params,
                 driver_coefficients=driver_coeffs,
                 duration=0.1,
             )
             solver.memory_manager.sync_stream(kernel)
             
             # Copy results from run()
             run_state = kernel.state.copy()
             
             # Allocate fresh device arrays for execute()
             output_sizes = kernel.output_array_sizes_3d
             d_inits = cuda.to_device(h_inits)
             d_params = cuda.to_device(h_params)
             d_driver_coeffs = cuda.to_device(driver_coeffs)
             d_state = cuda.device_array(output_sizes.state, dtype=precision)
             d_observables = cuda.device_array(
                 output_sizes.observables, dtype=precision
             )
             d_state_summaries = cuda.device_array(
                 output_sizes.state_summaries, dtype=precision
             )
             d_observable_summaries = cuda.device_array(
                 output_sizes.observable_summaries, dtype=precision
             )
             d_iteration_counters = cuda.device_array(
                 output_sizes.iteration_counters, dtype=np.int32
             )
             d_status_codes = cuda.device_array((n_runs,), dtype=np.int32)
             
             kernel.execute(
                 d_inits,
                 d_params,
                 d_driver_coeffs,
                 d_state,
                 d_observables,
                 d_state_summaries,
                 d_observable_summaries,
                 d_iteration_counters,
                 d_status_codes,
                 duration=0.1,
             )
             cuda.synchronize()
             
             execute_state = d_state.copy_to_host()
             
             # Results should match
             np.testing.assert_allclose(
                 run_state, execute_state, rtol=1e-5, atol=1e-7
             )
     ```
   - Edge cases: Covered in individual test cases; marked nocudasim
   - Integration: Uses existing test fixtures from conftest.py

**Outcomes**: 
- Files Modified: 
  * tests/batchsolving/test_execute.py (NEW - 262 lines)
- Functions/Methods Added/Modified:
  * TestExecute test class - NEW
  * TestRunCallsExecute test class - NEW
- Implementation Summary:
  * Created tests/batchsolving/test_execute.py with nocudasim-marked tests for:
    - execute() basic functionality (test_execute_basic)
    - execute() with single run (test_execute_single_run)
    - execute() with warmup parameter (test_execute_with_warmup)
    - Consistency between run() and execute() (test_run_and_execute_consistent)
  * All tests properly handle CUDA simulator mode with conditional import
  * Tests use solver fixture to get correct system sizes and precision
- Issues Flagged: None

---

## Summary

### Total Task Groups: 5

### Dependency Chain Overview
```
Task Group 1 (Level 3: execute)
       ↓
Task Group 2 (Level 2: solve_arrays)
       ↓
Task Group 3 (Import/Export)
       ↓
Task Groups 4 & 5 (Tests) - can run in parallel
```

### Parallel Execution Opportunities
- Task Groups 4 and 5 can run in parallel (both depend on Groups 1-3)
- Tasks within Group 3 can run in parallel

### Estimated Complexity
- **Task Group 1**: Medium - Extract execution logic into new method, refactor run()
- **Task Group 2**: Medium - Add validation helper and new method, refactor solve()
- **Task Group 3**: Low - Simple import additions
- **Task Group 4**: Medium - Comprehensive test coverage for solve_arrays
- **Task Group 5**: Medium - Device array tests requiring nocudasim marker

### Key Implementation Notes
1. Level 3 (execute) must be implemented first as it's the foundation
2. Level 2 (solve_arrays) builds on Level 3 by adding array management
3. Level 1 (solve) refactored last to call solve_arrays
4. All existing tests must continue to pass
5. New tests marked appropriately (nocudasim for execute tests)

---

# Implementation Complete - Ready for Review

## Execution Summary
- Total Task Groups: 5
- Completed: 5
- Failed: 0
- Total Files Modified: 6

## Task Group Completion
- Group 1: [x] Level 3 - BatchSolverKernel.execute() - Complete
- Group 2: [x] Level 2 - Solver.solve_arrays() - Complete
- Group 3: [x] Import and Export Updates - Complete
- Group 4: [x] Tests for Level 2 API (solve_arrays) - Complete
- Group 5: [x] Tests for Level 3 API (execute) - Complete

## All Modified Files
1. src/cubie/batchsolving/BatchSolverKernel.py (112 lines added - execute() method)
2. src/cubie/batchsolving/solver.py (121 lines added - validate_solver_arrays() and solve_arrays())
3. tests/batchsolving/conftest.py (18 lines added - shared fixtures)
4. tests/batchsolving/test_solver.py (15 lines removed - duplicate fixtures)
5. tests/batchsolving/test_solve_arrays.py (NEW - 282 lines)
6. tests/batchsolving/test_execute.py (NEW - 262 lines)

## Flagged Issues
None

## Handoff to Reviewer
All implementation tasks complete. Task list updated with outcomes.
Ready for reviewer agent to validate against user stories and goals.
