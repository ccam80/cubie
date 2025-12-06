# Implementation Task List
# Feature: Solver API Redesign
# Plan Reference: .github/active_plans/solver_api_redesign/agent_plan.md

## Overview

This task list implements the Solver API redesign to provide:
1. Single `solve()` entry point with automatic fast paths based on input detection
2. New `build_grid()` helper for external grid creation
3. BatchSolverKernel stays internal - not part of user API

Total Task Groups: 4
Dependency Chain: Group 1 → Group 2 → Group 3 → Group 4
Parallel Opportunities: Tasks within Group 3 can be parallelized

---

## Task Group 1: Input Classification Helper - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 1-50, 117-200, 325-410)
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 563-582) - reference for fast-path patterns

**Input Validation Required**:
- initial_values: Check if dict, np.ndarray, or has `__cuda_array_interface__`
- parameters: Check if dict, np.ndarray, or has `__cuda_array_interface__`
- For arrays: verify shape is 2D with format (n_vars, n_runs)
- For arrays: verify both have matching run counts (axis 1)

**Tasks**:
1. **Add _classify_inputs() private method**
   - File: src/cubie/batchsolving/solver.py
   - Action: Create new method after line 323 (before solve method)
   - Details:
     ```python
     def _classify_inputs(
         self,
         initial_values: Union[np.ndarray, Dict[str, Union[float, np.ndarray]]],
         parameters: Union[np.ndarray, Dict[str, Union[float, np.ndarray]]],
     ) -> str:
         """Classify input types to determine optimal processing path.

         Parameters
         ----------
         initial_values
             Initial state values as dict or array.
         parameters
             Parameter values as dict or array.

         Returns
         -------
         str
             Classification: 'dict', 'array', or 'device'.

         Notes
         -----
         Returns 'dict' when either input is a dictionary, triggering
         full grid construction. Returns 'array' when both inputs are
         numpy arrays with matching run counts in (n_vars, n_runs) format.
         Returns 'device' when both have __cuda_array_interface__.
         """
         # If either input is a dict, use grid builder path
         if isinstance(initial_values, dict) or isinstance(parameters, dict):
             return 'dict'

         # Check for device arrays (CUDA arrays with interface)
         init_is_device = hasattr(initial_values, '__cuda_array_interface__')
         param_is_device = hasattr(parameters, '__cuda_array_interface__')
         if init_is_device and param_is_device:
             return 'device'

         # Check for numpy arrays with correct shapes
         if isinstance(initial_values, np.ndarray) and isinstance(
             parameters, np.ndarray
         ):
             # Must be 2D arrays in (n_vars, n_runs) format
             if initial_values.ndim == 2 and parameters.ndim == 2:
                 n_states = self.system_sizes.n_states
                 n_params = self.system_sizes.n_parameters
                 # Verify variable counts match system expectations
                 if (initial_values.shape[0] == n_states and
                         parameters.shape[0] == n_params):
                     # Verify run counts match
                     if initial_values.shape[1] == parameters.shape[1]:
                         return 'array'

         # Default to dict path (grid builder handles conversion)
         return 'dict'
     ```
   - Edge cases: 
     - 1D arrays should return 'dict' (grid builder handles them)
     - Mixed array types should return 'dict'
     - Mismatched run counts should return 'dict'
   - Integration: Called at start of solve() method

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/solver.py (53 lines added)
- Functions/Methods Added/Modified:
  * _classify_inputs() method added to Solver class (lines 325-377)
- Implementation Summary:
  Added private method to classify input types as 'dict', 'array', or 'device'. Method checks for dictionary inputs first, then CUDA device arrays via __cuda_array_interface__, then validates numpy arrays have correct 2D shape with matching run counts against system_sizes. Falls back to 'dict' for edge cases.
- Issues Flagged: None

---

## Task Group 2: Array Validation Helper - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 608-612 - precision property)
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 714-770 - _sanitise_arraylike method)
- File: src/cubie/_utils.py (PrecisionDType type)

**Input Validation Required**:
- dtype: Cast to self.precision if mismatched
- contiguity: Make C-contiguous copy if not contiguous
- shape: Verified by _classify_inputs, no additional check needed

**Tasks**:
1. **Add _validate_arrays() private method**
   - File: src/cubie/batchsolving/solver.py
   - Action: Create new method after _classify_inputs() method
   - Details:
     ```python
     def _validate_arrays(
         self,
         initial_values: np.ndarray,
         parameters: np.ndarray,
     ) -> Tuple[np.ndarray, np.ndarray]:
         """Validate and prepare pre-built arrays for kernel execution.

         Parameters
         ----------
         initial_values
             Initial state array in (n_states, n_runs) format.
         parameters
             Parameter array in (n_params, n_runs) format.

         Returns
         -------
         Tuple[np.ndarray, np.ndarray]
             Validated arrays cast to system precision and made
             C-contiguous if necessary.

         Notes
         -----
         Arrays are cast to the system precision dtype and copied
         to ensure C-contiguous memory layout when needed.
         """
         precision = self.precision

         # Cast to correct dtype if needed
         if initial_values.dtype != precision:
             initial_values = initial_values.astype(precision)
         if parameters.dtype != precision:
             parameters = parameters.astype(precision)

         # Ensure C-contiguous layout
         if not initial_values.flags['C_CONTIGUOUS']:
             initial_values = np.ascontiguousarray(initial_values)
         if not parameters.flags['C_CONTIGUOUS']:
             parameters = np.ascontiguousarray(parameters)

         return initial_values, parameters
     ```
   - Edge cases:
     - F-contiguous arrays need conversion
     - Sliced arrays may not be contiguous
   - Integration: Called by solve() for array fast path

2. **Add Tuple import to typing imports**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify line 7 import statement
   - Details:
     ```python
     # Current:
     from typing import Any, Dict, List, Optional, Set, Tuple, Union
     # Already includes Tuple, no change needed
     ```
   - Note: Verify Tuple is already imported (it is on line 7)

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/solver.py (40 lines added)
- Functions/Methods Added/Modified:
  * _validate_arrays() method added to Solver class (lines 379-418)
- Implementation Summary:
  Added private method to validate and prepare pre-built numpy arrays for kernel execution. Method casts arrays to system precision dtype if needed, and ensures C-contiguous memory layout by copying if necessary. Tuple import already present in file.
- Issues Flagged: None

---

## Task Group 3: Modified solve() and New build_grid() - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 325-408 - current solve method)
- File: src/cubie/batchsolving/solver.py (lines 199 - grid_builder initialization)
- File: src/cubie/batchsolving/BatchGridBuilder.py (lines 477-540 - __call__ method signature)

**Input Validation Required**:
- grid_type parameter in build_grid: Must be "combinatorial" or "verbatim" (passed through to grid_builder)

**Tasks**:
1. **Modify solve() method to add input classification and branching**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify solve() method (lines 325-408)
   - Details:
     ```python
     def solve(
         self,
         initial_values: Union[np.ndarray, Dict[str, Union[float,np.ndarray]]],
         parameters: Union[np.ndarray, Dict[str, Union[float,np.ndarray]]],
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
         """Solve a batch initial value problem.

         Parameters
         ----------
         initial_values
             Initial state values for each integration run. Accepts
             dictionaries mapping state names to values for grid
             construction, or pre-built arrays in (n_states, n_runs)
             format for fast-path execution.
         parameters
             Parameter values for each run. Accepts dictionaries
             mapping parameter names to values, or pre-built arrays
             in (n_params, n_runs) format.
         drivers
             Driver samples or configuration matching
             :class:`cubie.integrators.array_interpolator.ArrayInterpolator`.
         duration
             Total integration time. Default is ``1.0``.
         settling_time
             Warm-up period before recording outputs. Default ``0.0``.
         t0
             Initial integration time. Default ``0.0``.
         blocksize
             CUDA block size used for kernel launch. Default ``256``.
         stream
             Stream on which to execute the kernel. ``None`` uses the solver's
             default stream.
         chunk_axis
             Dimension along which to chunk when memory is limited. Default is
             ``"run"``.
         grid_type
             Strategy for constructing the integration grid from inputs.
             Only used when dict inputs trigger grid construction.
         results_type
             Format of returned results, for example ``"full"`` or ``"numpy"``.
         **kwargs
             Additional options forwarded to :meth:`update`.

         Returns
         -------
         SolveResult
             Collected results from the integration run.

         Notes
         -----
         Input type detection determines the processing path:

         - Dictionary inputs trigger grid construction via
           :class:`BatchGridBuilder`
         - Pre-built numpy arrays with correct shapes skip grid
           construction for improved performance
         - Device arrays receive minimal processing before kernel
           execution
         """
         if kwargs:
             self.update(kwargs, silent=True)

         # Classify inputs to determine processing path
         input_type = self._classify_inputs(initial_values, parameters)

         if input_type == 'dict':
             # Dictionary inputs: use grid builder (existing behavior)
             inits, params = self.grid_builder(
                 states=initial_values, params=parameters, kind=grid_type
             )
         elif input_type == 'array':
             # Pre-built arrays: validate and use directly (fast path)
             inits, params = self._validate_arrays(initial_values, parameters)
         else:
             # Device arrays: use directly with minimal processing
             inits, params = initial_values, parameters

         fn_changed = False  # ensure defined if drivers is None
         if drivers is not None:
             ArrayInterpolator.check_against_system_drivers(
                 drivers, self.system
             )
             fn_changed = self.driver_interpolator.update_from_dict(drivers)
         if fn_changed:
             self.update(
                 {"driver_function": self.driver_interpolator.evaluation_function,
                  "driver_del_t": self.driver_interpolator.driver_del_t}
             )

         self.kernel.run(
             inits=inits,
             params=params,
             driver_coefficients=self.driver_interpolator.coefficients,
             duration=duration,
             warmup=settling_time,
             t0=t0,
             blocksize=blocksize,
             stream=stream,
             chunk_axis=chunk_axis,
         )
         self.memory_manager.sync_stream(self.kernel)
         return SolveResult.from_solver(self, results_type=results_type)
     ```
   - Edge cases:
     - Drivers update independently of input classification
     - kwargs update happens before classification
   - Integration: Uses _classify_inputs and _validate_arrays from Groups 1-2

2. **Add build_grid() public method**
   - File: src/cubie/batchsolving/solver.py
   - Action: Create new public method after solve() method (around line 410)
   - Details:
     ```python
     def build_grid(
         self,
         initial_values: Union[np.ndarray, Dict[str, Union[float, np.ndarray]]],
         parameters: Union[np.ndarray, Dict[str, Union[float, np.ndarray]]],
         grid_type: str = "verbatim",
     ) -> Tuple[np.ndarray, np.ndarray]:
         """Build parameter and state grids for external use.

         Parameters
         ----------
         initial_values
             Initial state values as dictionaries mapping state names
             to value sequences, or arrays in (n_states, n_runs) format.
         parameters
             Parameter values as dictionaries mapping parameter names
             to value sequences, or arrays in (n_params, n_runs) format.
         grid_type
             Strategy for constructing the grid. ``"combinatorial"``
             produces all combinations while ``"verbatim"`` preserves
             column-wise pairings. Default is ``"verbatim"``.

         Returns
         -------
         Tuple[np.ndarray, np.ndarray]
             Tuple of (initial_values, parameters) arrays in
             (n_vars, n_runs) format with system precision dtype.
             These arrays can be passed directly to :meth:`solve`
             for fast-path execution.

         Examples
         --------
         >>> inits, params = solver.build_grid(
         ...     {"x": [1, 2, 3]}, {"p": [0.1, 0.2]},
         ...     grid_type="combinatorial"
         ... )
         >>> result = solver.solve(inits, params)  # Uses fast path
         """
         return self.grid_builder(
             states=initial_values, params=parameters, kind=grid_type
         )
     ```
   - Edge cases:
     - Arrays passed in are still processed by grid_builder
     - Returns arrays suitable for fast-path solve
   - Integration: Wraps existing grid_builder call

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/solver.py (updated docstring ~30 lines, added branching logic ~20 lines, added build_grid method ~40 lines)
- Functions/Methods Added/Modified:
  * solve() method updated with input classification branching (lines 420-531)
  * build_grid() public method added (lines 533-572)
- Implementation Summary:
  Modified solve() to classify inputs and branch to appropriate path. Dict inputs use grid_builder, array inputs go through _validate_arrays(), device arrays pass through directly. Added comprehensive docstring explaining the three processing paths. Added build_grid() public method that wraps grid_builder for users who want to pre-build grids for reuse.
- Issues Flagged: None

---

## Task Group 4: Tests - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: tests/batchsolving/test_solver.py (entire file - test patterns)
- File: tests/conftest.py (lines 136-146 - solver fixture builders)
- File: tests/conftest.py (lines 316-400 - solver_settings fixture)

**Input Validation Required**:
- None (tests validate implementation behavior)

**Tasks**:
1. **Add test for _classify_inputs() method**
   - File: tests/batchsolving/test_solver.py
   - Action: Add new test after existing tests (around line 628)
   - Details:
     ```python
     def test_classify_inputs_dict(solver, simple_initial_values, simple_parameters):
         """Test that dict inputs are classified as 'dict'."""
         result = solver._classify_inputs(simple_initial_values, simple_parameters)
         assert result == 'dict'


     def test_classify_inputs_mixed(solver, system):
         """Test that mixed inputs (dict + array) are classified as 'dict'."""
         n_states = solver.system_sizes.n_states
         inits_array = np.ones((n_states, 2), dtype=solver.precision)
         params_dict = {list(system.parameters.names)[0]: [1.0, 2.0]}
         
         result = solver._classify_inputs(inits_array, params_dict)
         assert result == 'dict'
         
         # Test the reverse case
         inits_dict = {list(system.initial_values.names)[0]: [0.1, 0.2]}
         n_params = solver.system_sizes.n_parameters
         params_array = np.ones((n_params, 2), dtype=solver.precision)
         
         result = solver._classify_inputs(inits_dict, params_array)
         assert result == 'dict'


     def test_classify_inputs_array(solver):
         """Test that matching numpy arrays are classified as 'array'."""
         n_states = solver.system_sizes.n_states
         n_params = solver.system_sizes.n_parameters
         n_runs = 4
         
         inits = np.ones((n_states, n_runs), dtype=solver.precision)
         params = np.ones((n_params, n_runs), dtype=solver.precision)
         
         result = solver._classify_inputs(inits, params)
         assert result == 'array'


     def test_classify_inputs_mismatched_runs(solver):
         """Test that mismatched run counts fall back to 'dict'."""
         n_states = solver.system_sizes.n_states
         n_params = solver.system_sizes.n_parameters
         
         inits = np.ones((n_states, 3), dtype=solver.precision)
         params = np.ones((n_params, 5), dtype=solver.precision)
         
         result = solver._classify_inputs(inits, params)
         assert result == 'dict'


     def test_classify_inputs_wrong_var_count(solver):
         """Test that wrong variable counts fall back to 'dict'."""
         n_params = solver.system_sizes.n_parameters
         n_runs = 4
         
         # Wrong number of states
         inits = np.ones((999, n_runs), dtype=solver.precision)
         params = np.ones((n_params, n_runs), dtype=solver.precision)
         
         result = solver._classify_inputs(inits, params)
         assert result == 'dict'


     def test_classify_inputs_1d_arrays(solver):
         """Test that 1D arrays fall back to 'dict'."""
         n_states = solver.system_sizes.n_states
         n_params = solver.system_sizes.n_parameters
         
         inits = np.ones(n_states, dtype=solver.precision)
         params = np.ones(n_params, dtype=solver.precision)
         
         result = solver._classify_inputs(inits, params)
         assert result == 'dict'
     ```
   - Integration: Tests internal classification logic

2. **Add test for _validate_arrays() method**
   - File: tests/batchsolving/test_solver.py
   - Action: Add new test after classification tests
   - Details:
     ```python
     def test_validate_arrays_dtype_cast(solver):
         """Test that arrays are cast to system precision."""
         n_states = solver.system_sizes.n_states
         n_params = solver.system_sizes.n_parameters
         n_runs = 2
         
         # Create arrays with wrong dtype
         wrong_dtype = np.float64 if solver.precision == np.float32 else np.float32
         inits = np.ones((n_states, n_runs), dtype=wrong_dtype)
         params = np.ones((n_params, n_runs), dtype=wrong_dtype)
         
         validated_inits, validated_params = solver._validate_arrays(inits, params)
         
         assert validated_inits.dtype == solver.precision
         assert validated_params.dtype == solver.precision


     def test_validate_arrays_contiguity(solver):
         """Test that non-contiguous arrays are made contiguous."""
         n_states = solver.system_sizes.n_states
         n_params = solver.system_sizes.n_parameters
         n_runs = 4
         
         # Create F-contiguous arrays
         inits = np.asfortranarray(np.ones((n_states, n_runs), dtype=solver.precision))
         params = np.asfortranarray(np.ones((n_params, n_runs), dtype=solver.precision))
         
         assert not inits.flags['C_CONTIGUOUS']
         assert not params.flags['C_CONTIGUOUS']
         
         validated_inits, validated_params = solver._validate_arrays(inits, params)
         
         assert validated_inits.flags['C_CONTIGUOUS']
         assert validated_params.flags['C_CONTIGUOUS']
     ```
   - Integration: Tests validation helper

3. **Add test for build_grid() method**
   - File: tests/batchsolving/test_solver.py
   - Action: Add new test after validation tests
   - Details:
     ```python
     def test_build_grid_returns_correct_shape(
         solver, simple_initial_values, simple_parameters
     ):
         """Test that build_grid returns arrays with correct shapes."""
         inits, params = solver.build_grid(
             simple_initial_values, simple_parameters, grid_type="verbatim"
         )
         
         assert isinstance(inits, np.ndarray)
         assert isinstance(params, np.ndarray)
         assert inits.ndim == 2
         assert params.ndim == 2
         assert inits.shape[0] == solver.system_sizes.n_states
         assert params.shape[0] == solver.system_sizes.n_parameters
         # Verbatim: run count matches input length
         assert inits.shape[1] == params.shape[1]


     def test_build_grid_combinatorial(
         solver, simple_initial_values, simple_parameters
     ):
         """Test that build_grid with combinatorial creates product grid."""
         inits, params = solver.build_grid(
             simple_initial_values, simple_parameters, grid_type="combinatorial"
         )
         
         # Combinatorial produces more runs than verbatim
         n_init_values = len(list(simple_initial_values.values())[0])
         n_param_values = len(list(simple_parameters.values())[0])
         # Number of runs is product of all value counts
         assert inits.shape[1] >= n_init_values
         assert params.shape[1] >= n_param_values


     def test_build_grid_precision(solver, simple_initial_values, simple_parameters):
         """Test that build_grid returns arrays with correct precision."""
         inits, params = solver.build_grid(
             simple_initial_values, simple_parameters
         )
         
         assert inits.dtype == solver.precision
         assert params.dtype == solver.precision
     ```
   - Integration: Tests new public method

4. **Add test for solve() fast path with pre-built arrays**
   - File: tests/batchsolving/test_solver.py
   - Action: Add new test after build_grid tests
   - Details:
     ```python
     @pytest.mark.parametrize("solver_settings_override",
                              [{
                                 "duration": 0.05,
                                 "dt_save": 0.02,
                                 "dt_summarise": 0.04,
                                 "output_types": ["state", "time"]
                              }],
                              indirect=True
     )
     def test_solve_with_prebuilt_arrays(
         solver, simple_initial_values, simple_parameters, driver_settings
     ):
         """Test that solve() works with pre-built arrays (fast path)."""
         # First build the grid
         inits, params = solver.build_grid(
             simple_initial_values, simple_parameters, grid_type="verbatim"
         )
         
         # Now solve with the pre-built arrays
         result = solver.solve(
             initial_values=inits,
             parameters=params,
             drivers=driver_settings,
             duration=0.05,
         )
         
         assert isinstance(result, SolveResult)


     @pytest.mark.parametrize("solver_settings_override",
                              [{
                                 "duration": 0.05,
                                 "dt_save": 0.02,
                                 "dt_summarise": 0.04,
                                 "output_types": ["state", "time"]
                              }],
                              indirect=True
     )
     def test_solve_array_path_matches_dict_path(
         solver, simple_initial_values, simple_parameters, driver_settings
     ):
         """Test that array fast path produces same results as dict path."""
         # Solve with dict inputs
         result_dict = solver.solve(
             initial_values=simple_initial_values,
             parameters=simple_parameters,
             drivers=driver_settings,
             duration=0.05,
             grid_type="verbatim",
         )
         
         # Build grid and solve with arrays
         inits, params = solver.build_grid(
             simple_initial_values, simple_parameters, grid_type="verbatim"
         )
         result_array = solver.solve(
             initial_values=inits,
             parameters=params,
             drivers=driver_settings,
             duration=0.05,
         )
         
         # Results should match
         assert result_dict.time_domain_array.shape == result_array.time_domain_array.shape
         np.testing.assert_allclose(
             result_dict.time_domain_array,
             result_array.time_domain_array,
             rtol=1e-5,
             atol=1e-7,
         )
     ```
   - Integration: Tests full solve path with arrays

5. **Add test for backward compatibility with dict inputs**
   - File: tests/batchsolving/test_solver.py
   - Action: Add new test to verify existing behavior unchanged
   - Details:
     ```python
     @pytest.mark.parametrize("solver_settings_override",
                              [{
                                 "duration": 0.05,
                                 "dt_save": 0.02,
                                 "dt_summarise": 0.04,
                                 "output_types": ["state"]
                              }],
                              indirect=True
     )
     def test_solve_dict_path_backward_compatible(
         solver, simple_initial_values, simple_parameters, driver_settings
     ):
         """Test that dict inputs still work exactly as before."""
         result = solver.solve(
             initial_values=simple_initial_values,
             parameters=simple_parameters,
             drivers=driver_settings,
             duration=0.05,
             settling_time=0.0,
             blocksize=32,
             grid_type="combinatorial",
             results_type="full",
         )
         
         assert isinstance(result, SolveResult)
         assert hasattr(result, "time_domain_array")
         assert hasattr(result, "summaries_array")
     ```
   - Integration: Validates US-4 backward compatibility

**Outcomes**: 
- Files Modified:
  * tests/batchsolving/test_solver.py (~260 lines added)
- Functions/Methods Added/Modified:
  * test_classify_inputs_dict()
  * test_classify_inputs_mixed()
  * test_classify_inputs_array()
  * test_classify_inputs_mismatched_runs()
  * test_classify_inputs_wrong_var_count()
  * test_classify_inputs_1d_arrays()
  * test_validate_arrays_dtype_cast()
  * test_validate_arrays_contiguity()
  * test_build_grid_returns_correct_shape()
  * test_build_grid_combinatorial()
  * test_build_grid_precision()
  * test_solve_with_prebuilt_arrays()
  * test_solve_array_path_matches_dict_path()
  * test_solve_dict_path_backward_compatible()
- Implementation Summary:
  Added comprehensive tests for all new functionality. Tests cover input classification (dict, array, mixed, mismatched, wrong shapes), array validation (dtype casting, contiguity), build_grid (shapes, combinatorial, precision), and solve fast paths (prebuilt arrays, path equivalence, backward compatibility).
- Issues Flagged: None

---

## Summary

| Group | Name | Type | Tasks | Estimated Complexity |
|-------|------|------|-------|---------------------|
| 1 | Input Classification Helper | SEQUENTIAL | 1 | Low |
| 2 | Array Validation Helper | SEQUENTIAL | 2 | Low |
| 3 | Modified solve() and build_grid() | SEQUENTIAL | 2 | Medium |
| 4 | Tests | SEQUENTIAL | 5 | Medium |

### Dependency Chain
```
Group 1: _classify_inputs()
    ↓
Group 2: _validate_arrays()
    ↓
Group 3: solve() modification + build_grid()
    ↓
Group 4: All tests
```

### User Story Coverage
- **US-1**: Satisfied by Group 3 Task 1 (single solve() with input detection)
- **US-2**: Satisfied by Groups 1-3 (automatic fast path detection)
- **US-3**: Satisfied by Group 3 Task 2 (build_grid() helper)
- **US-4**: Satisfied by Group 3 Task 1 + Group 4 Task 5 (backward compatibility)

### Key Implementation Notes

1. **No changes to BatchGridBuilder.py** - The fast path in lines 563-582 already handles array inputs correctly. The input classification in Solver determines whether to call grid_builder at all.

2. **No changes to BatchSolverKernel.py** - The `run()` method already accepts arrays. It remains internal.

3. **Minimal solve() changes** - Only add classification at entry and branching. All other logic unchanged.

4. **Type hints** - All new methods have full type hints in PEP484 format.

5. **Docstrings** - All new methods have numpydoc-style docstrings.

6. **PEP8 compliance** - Max 79 chars per line enforced.
