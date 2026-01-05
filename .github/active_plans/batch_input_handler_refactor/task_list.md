# Implementation Task List
# Feature: BatchInputHandler Refactoring
# Plan Reference: .github/active_plans/batch_input_handler_refactor/agent_plan.md

## Task Group 1: Rename BatchGridBuilder File and Class
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/BatchGridBuilder.py (entire file - 978 lines)
- File: .github/copilot-instructions.md (lines 1-50 for code style)

**Input Validation Required**:
- No additional validation beyond existing logic

**Tasks**:
1. **Rename file BatchGridBuilder.py to BatchInputHandler.py**
   - File: src/cubie/batchsolving/BatchGridBuilder.py
   - Action: Rename to src/cubie/batchsolving/BatchInputHandler.py
   - Details: Use `git mv` to preserve history

2. **Rename class BatchGridBuilder to BatchInputHandler**
   - File: src/cubie/batchsolving/BatchInputHandler.py (after rename)
   - Action: Modify
   - Details:
     ```python
     # Line 446: Change class definition
     class BatchInputHandler:
         """Process and validate solver inputs for batch runs.
     
         The handler converts dictionaries or arrays into solver-ready
         two-dimensional arrays, classifies input types for optimal
         processing paths, and validates array shapes and dtypes.
     
         Parameters
         ----------
         interface
             System interface containing parameter and state metadata.
     
         Attributes
         ----------
         parameters
             Parameter metadata sourced from ``interface``.
         states
             State metadata sourced from ``interface``.
         precision
             Floating-point precision for returned arrays.
         """
     ```
   - Edge cases: None
   - Integration: All other class references will be updated in subsequent task groups

3. **Update from_system classmethod return type annotation**
   - File: src/cubie/batchsolving/BatchInputHandler.py
   - Action: Modify
   - Details:
     ```python
     # Line 473-488: Update classmethod
     @classmethod
     def from_system(cls, system: BaseODE) -> "BatchInputHandler":
         """Create a handler from a system model.
     
         Parameters
         ----------
         system
             System model providing parameter and state metadata.
     
         Returns
         -------
         BatchInputHandler
             Handler configured for ``system``.
         """
         interface = SystemInterface.from_system(system)
         return cls(interface)
     ```

4. **Update __call__ method argument order to (states, params)**
   - File: src/cubie/batchsolving/BatchInputHandler.py
   - Action: Modify
   - Details:
     ```python
     # Line 490-519: Change __call__ signature
     def __call__(
         self,
         states: Optional[Union[Dict, ArrayLike]] = None,
         params: Optional[Union[Dict, ArrayLike]] = None,
         kind: str = "combinatorial",
     ) -> tuple[np.ndarray, np.ndarray]:
         """Process user input to generate state and parameter arrays.
     
         Parameters
         ----------
         states
             Optional dictionary or array describing initial state sweeps.
         params
             Optional dictionary or array describing parameter sweeps.
         kind
             Strategy for grid assembly. ``"combinatorial"`` expands
             all combinations while ``"verbatim"`` preserves pairings.
     
         Returns
         -------
         tuple[np.ndarray, np.ndarray]
             Initial state and parameter arrays aligned for batch execution.
     
         Notes
         -----
         Passing ``states`` and ``params`` as arrays treats each as a
         complete grid. ``kind="combinatorial"`` computes the Cartesian
         product of both grids. When arrays already describe paired runs,
         set ``kind`` to ``"verbatim"`` to keep them aligned.
         """
     ```
   - Edge cases: Existing tests use params=..., states=... keyword order which works

5. **Rename _are_right_sized_arrays parameter from inits to states**
   - File: src/cubie/batchsolving/BatchInputHandler.py
   - Action: Modify
   - Details:
     ```python
     # Line 774-828: Rename parameter
     def _are_right_sized_arrays(
         self,
         states: Optional[Union[ArrayLike, Dict]],
         params: Optional[Union[ArrayLike, Dict]],
     ) -> bool:
         """Check if both inputs are pre-formatted arrays ready for the solver.
     
         This method only returns True when both inputs are 2D numpy arrays
         with matching run counts and correct variable counts for their
         respective SystemValues objects. Returns False for None, dicts,
         or arrays that need further processing.
     
         Also handles the special case where a SystemValues object is empty
         (no variables), in which case None or an empty 2D array is acceptable.
     
         Parameters
         ----------
         states
             Initial states as array or dict.
         params
             Parameters as array, dict, or None.
     
         Returns
         -------
         bool
             True if both inputs are correctly sized 2D arrays with matching
             run counts.
         """
         # Handle empty parameters case: states must be right-sized array,
         # params can be None
         if self.parameters.empty:
             if isinstance(states, np.ndarray) and states.ndim == 2:
                 if states.shape[0] == self.states.n:
                     if params is None:
                         return True
                     if isinstance(params, np.ndarray) and params.ndim == 2:
                         return (params.shape[0] == 0
                                 and params.shape[1] == states.shape[1])
             return False
     
         # Normal case: both must be 2D arrays
         if isinstance(states, np.ndarray) and isinstance(params, np.ndarray):
             # Both arrays: check run counts match and arrays are system-sized
             if states.ndim != 2 or params.ndim != 2:
                 return False
             states_runs = states.shape[1]
             states_variables = states.shape[0]
             params_runs = params.shape[1]
             params_variables = params.shape[0]
             if states_runs == params_runs:
                 if (states_variables == self.states.n
                         and params_variables == self.parameters.n):
                     return True
         return False
     ```
   - Edge cases: Uses states consistently instead of mixing inits/states

6. **Update module docstring to reflect new class name and responsibilities**
   - File: src/cubie/batchsolving/BatchInputHandler.py
   - Action: Modify
   - Details:
     ```python
     """Batch input handling for state and parameter processing.
     
     This module processes user-supplied dictionaries or arrays into the 2D NumPy
     arrays expected by the batch solver. :class:`BatchInputHandler` is the primary
     entry point and is usually accessed through :class:`cubie.batchsolving.solver.Solver`.
     
     The handler provides:
     - Input classification to determine optimal processing paths
     - Array validation and precision casting
     - Grid construction for combinatorial or verbatim runs
     
     Notes
     -----
     ``BatchInputHandler.__call__`` accepts three arguments:
     
     ``states``
         Mapping or array containing state values only. One-dimensional
         inputs override defaults for every run, while two-dimensional inputs
         are treated as pre-built grids in (variable, run) format.
     ``params``
         Mapping or array containing parameter values only. Interpretation matches
         ``states``.
     ``kind``
         Controls how inputs are combined. ``"combinatorial"`` builds the
         Cartesian product, while ``"verbatim"`` preserves column-wise groupings.
     
     [... rest of docstring with examples updated to use states, params order ...]
     """
     ```

**Tests to Create**:
- None for this group (tests updated in Task Group 4)

**Tests to Run**:
- None for this group (deferred to after Task Group 4)

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: Add classify_inputs and validate_arrays Methods
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/batchsolving/BatchInputHandler.py (entire file after Task Group 1)
- File: src/cubie/batchsolving/solver.py (lines 340-425 for _classify_inputs and _validate_arrays)
- File: .github/copilot-instructions.md (lines 1-50 for code style)

**Input Validation Required**:
- classify_inputs: Check if inputs are dict, have __cuda_array_interface__, or are correctly-shaped numpy arrays
- validate_arrays: Cast arrays to system precision dtype

**Tasks**:
1. **Add classify_inputs method to BatchInputHandler**
   - File: src/cubie/batchsolving/BatchInputHandler.py
   - Action: Add method after __call__ (around line 547)
   - Details:
     ```python
     def classify_inputs(
         self,
         states: Union[np.ndarray, Dict[str, Union[float, np.ndarray]], None],
         params: Union[np.ndarray, Dict[str, Union[float, np.ndarray]], None],
     ) -> str:
         """Classify input types to determine optimal processing path.
     
         Parameters
         ----------
         states
             Initial state values as dict, array, or None.
         params
             Parameter values as dict, array, or None.
     
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
         if isinstance(states, dict) or isinstance(params, dict):
             return 'dict'
     
         # Check for device arrays (CUDA arrays with interface)
         states_is_device = hasattr(states, '__cuda_array_interface__')
         params_is_device = hasattr(params, '__cuda_array_interface__')
         if states_is_device and params_is_device:
             return 'device'
     
         # Check for numpy arrays with correct shapes
         if isinstance(states, np.ndarray) and isinstance(params, np.ndarray):
             # Must be 2D arrays in (n_vars, n_runs) format
             if states.ndim == 2 and params.ndim == 2:
                 n_states = self.states.n
                 n_params = self.parameters.n
                 # Verify variable counts match system expectations
                 if (states.shape[0] == n_states and
                         params.shape[0] == n_params):
                     # Verify run counts match
                     if states.shape[1] == params.shape[1]:
                         return 'array'
     
         # Default to dict path (grid builder handles conversion)
         return 'dict'
     ```
   - Edge cases: Handle None inputs (fall through to 'dict')
   - Integration: Used by Solver.solve() to determine processing path

2. **Add validate_arrays method to BatchInputHandler**
   - File: src/cubie/batchsolving/BatchInputHandler.py
   - Action: Add method after classify_inputs
   - Details:
     ```python
     def validate_arrays(
         self,
         states: np.ndarray,
         params: np.ndarray,
     ) -> Tuple[np.ndarray, np.ndarray]:
         """Validate and prepare pre-built arrays for kernel execution.
     
         Parameters
         ----------
         states
             Initial state array in (n_states, n_runs) format.
         params
             Parameter array in (n_params, n_runs) format.
     
         Returns
         -------
         Tuple[np.ndarray, np.ndarray]
             Validated arrays cast to system precision in (states, params) order.
     
         Notes
         -----
         Arrays are cast to the system precision dtype when needed.
         Returned as contiguous arrays for optimal kernel performance.
         """
         # Update precision from current system state
         self.precision = self.states.precision
     
         # Cast to correct dtype if needed
         if states.dtype != self.precision:
             states = np.ascontiguousarray(
                 states.astype(self.precision, copy=False)
             )
         if params.dtype != self.precision:
             params = np.ascontiguousarray(
                 params.astype(self.precision, copy=False)
             )
     
         return states, params
     ```
   - Edge cases: Handle arrays already at correct precision (no copy needed)
   - Integration: Used by Solver.solve() for array fast path

3. **Add Tuple to imports**
   - File: src/cubie/batchsolving/BatchInputHandler.py
   - Action: Modify imports (line 111)
   - Details:
     ```python
     from typing import Dict, List, Optional, Tuple, Union
     ```

**Tests to Create**:
- Test file: tests/batchsolving/test_batch_input_handler.py
- Test function: test_classify_inputs_dict
- Description: Verify dict inputs return 'dict'
- Test function: test_classify_inputs_array
- Description: Verify correctly-shaped arrays return 'array'
- Test function: test_classify_inputs_device
- Description: Verify device arrays return 'device'
- Test function: test_validate_arrays_dtype_cast
- Description: Verify arrays are cast to system precision

**Tests to Run**:
- tests/batchsolving/test_batch_input_handler.py::test_classify_inputs_dict
- tests/batchsolving/test_batch_input_handler.py::test_classify_inputs_array
- tests/batchsolving/test_batch_input_handler.py::test_validate_arrays_dtype_cast

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: Update Solver and Package Exports
**Status**: [ ]
**Dependencies**: Task Groups 1 and 2

**Required Context**:
- File: src/cubie/batchsolving/solver.py (entire file - 1102 lines)
- File: src/cubie/batchsolving/__init__.py (entire file - 75 lines)
- File: src/cubie/batchsolving/BatchInputHandler.py (lines 1-50, 446-600 for new methods)

**Input Validation Required**:
- No additional validation; delegate to BatchInputHandler

**Tasks**:
1. **Update solver.py import statement**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify line 15
   - Details:
     ```python
     from cubie.batchsolving.BatchInputHandler import BatchInputHandler
     ```

2. **Rename grid_builder attribute to input_handler in Solver.__init__**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify line 224
   - Details:
     ```python
     self.input_handler = BatchInputHandler(interface)
     ```

3. **Update Solver._classify_inputs to delegate to input_handler**
   - File: src/cubie/batchsolving/solver.py
   - Action: Delete method (lines 340-392)
   - Details: Remove the entire _classify_inputs method; Solver.solve() will call self.input_handler.classify_inputs() instead

4. **Update Solver._validate_arrays to delegate to input_handler**
   - File: src/cubie/batchsolving/solver.py
   - Action: Delete method (lines 394-425)
   - Details: Remove the entire _validate_arrays method; Solver.solve() will call self.input_handler.validate_arrays() instead

5. **Update Solver.solve() to use input_handler**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify lines 508-521
   - Details:
     ```python
     # Classify inputs to determine processing path
     input_type = self.input_handler.classify_inputs(
         states=initial_values,
         params=parameters
     )
     
     if input_type == 'dict':
         # Dictionary inputs: use grid builder (existing behavior)
         inits, params = self.input_handler(
             states=initial_values,
             params=parameters,
             kind=grid_type
         )
     elif input_type == 'array':
         # Pre-built arrays: validate and use directly (fast path)
         inits, params = self.input_handler.validate_arrays(
             states=initial_values,
             params=parameters
         )
     else:
         # Device arrays: use directly with minimal processing
         inits, params = initial_values, parameters
     ```
   - Edge cases: Preserve device array fast path

6. **Update Solver.build_grid() to use input_handler**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify lines 596-598
   - Details:
     ```python
     return self.input_handler(
         states=initial_values,
         params=parameters,
         kind=grid_type
     )
     ```

7. **Update batchsolving/__init__.py imports and exports**
   - File: src/cubie/batchsolving/__init__.py
   - Action: Modify lines 18 and 49-54
   - Details:
     ```python
     # Line 18: Update import
     from cubie.batchsolving.BatchInputHandler import BatchInputHandler  # noqa: E402
     # Backward compatibility alias
     BatchGridBuilder = BatchInputHandler
     
     # In __all__ list, update and add entries:
     __all__ = [
         "ActiveOutputs",
         "ArrayContainer",
         "ArrayTypes",
         "BatchGridBuilder",  # Deprecated alias for backward compatibility
         "BatchInputHandler",
         # ... rest of exports
     ]
     ```

**Tests to Create**:
- None for this group (tests updated in Task Group 4)

**Tests to Run**:
- tests/batchsolving/test_solver.py::test_solver_initialization

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: Update Tests and Fixtures
**Status**: [ ]
**Dependencies**: Task Groups 1, 2, and 3

**Required Context**:
- File: tests/batchsolving/test_batch_grid_builder.py (entire file - 994 lines)
- File: tests/batchsolving/conftest.py (entire file - 157 lines)
- File: tests/batchsolving/test_solver.py (entire file - 803 lines)
- File: src/cubie/batchsolving/BatchInputHandler.py (lines 446-650 for class and new methods)

**Input Validation Required**:
- None; tests verify existing validation

**Tasks**:
1. **Rename test file**
   - File: tests/batchsolving/test_batch_grid_builder.py
   - Action: Rename to tests/batchsolving/test_batch_input_handler.py
   - Details: Use `git mv` to preserve history

2. **Update imports in test_batch_input_handler.py**
   - File: tests/batchsolving/test_batch_input_handler.py
   - Action: Modify lines 5-14
   - Details:
     ```python
     from cubie.batchsolving.BatchInputHandler import (
         BatchInputHandler,
         combinatorial_grid,
         combine_grids,
         extend_grid_to_array,
         generate_array,
         generate_grid,
         unique_cartesian_product,
         verbatim_grid,
     )
     ```

3. **Update grid_builder fixture to input_handler**
   - File: tests/batchsolving/test_batch_input_handler.py
   - Action: Modify lines 19-21
   - Details:
     ```python
     @pytest.fixture(scope="session")
     def input_handler(system):
         return BatchInputHandler.from_system(system)
     
     # Keep backward compatibility alias
     @pytest.fixture(scope="session")
     def grid_builder(input_handler):
         return input_handler
     ```

4. **Update test_docstring_examples to use new class name**
   - File: tests/batchsolving/test_batch_input_handler.py
   - Action: Modify lines 433-437
   - Details:
     ```python
     def test_docstring_examples(input_handler, system, tolerance):
         # Example 1: combinatorial dict
     
         handler = BatchInputHandler.from_system(system)
         params = {"p0": [0.1, 0.2], "p1": [10, 20]}
         states = {"x0": [1.0, 2.0], "x1": [0.5, 1.5]}
         initial_states, parameters = handler(
             states=states, params=params, kind="combinatorial"
         )
     ```

5. **Update conftest.py imports and fixture name**
   - File: tests/batchsolving/conftest.py
   - Action: Modify lines 10 and 16-20
   - Details:
     ```python
     # Line 10: Update import
     from cubie.batchsolving.BatchInputHandler import BatchInputHandler
     
     # Lines 16-20: Rename fixture
     @pytest.fixture(scope="session")
     def input_handler(system) -> BatchInputHandler:
         """Return a batch input handler for the configured system."""
         return BatchInputHandler.from_system(system)
     
     # Add backward compatibility alias
     @pytest.fixture(scope="session")
     def batchconfig_instance(input_handler) -> BatchInputHandler:
         """Backward compatibility alias for input_handler."""
         return input_handler
     ```

6. **Update batch_input_arrays fixture**
   - File: tests/batchsolving/conftest.py
   - Action: Modify lines 72-91
   - Details:
     ```python
     @pytest.fixture(scope="session")
     def batch_input_arrays(
         batch_request,
         batch_settings,
         input_handler,
         system,
     ) -> tuple[Array, Array]:
         """Return the initial state and parameter arrays for the batch run."""
         # Separate batch_request into states and params based on system names
         state_names = set(system.initial_values.names)
         param_names = set(system.parameters.names)
     
         states_dict = {k: v for k, v in batch_request.items() if k in state_names}
         params_dict = {k: v for k, v in batch_request.items() if k in param_names}
     
         return input_handler(
             states=states_dict,
             params=params_dict,
             kind=batch_settings["kind"]
         )
     ```

7. **Update test_solver.py imports**
   - File: tests/batchsolving/test_solver.py
   - Action: Modify line 7
   - Details:
     ```python
     from cubie.batchsolving.BatchInputHandler import BatchInputHandler
     ```

8. **Update test_solver_initialization to check input_handler**
   - File: tests/batchsolving/test_solver.py
   - Action: Modify lines 53-61
   - Details:
     ```python
     def test_solver_initialization(solver, system):
         """Test that the solver initializes correctly."""
         assert solver is not None
         assert solver.system_interface is not None
         assert solver.input_handler is not None
         assert solver.kernel is not None
         assert isinstance(solver.system_interface, SystemInterface)
         assert isinstance(solver.input_handler, BatchInputHandler)
     ```

9. **Update test_classify_inputs tests to use input_handler**
   - File: tests/batchsolving/test_solver.py
   - Action: Modify lines 577-649
   - Details:
     ```python
     def test_classify_inputs_dict(solver, simple_initial_values, simple_parameters):
         """Test that dict inputs are classified as 'dict'."""
         result = solver.input_handler.classify_inputs(
             states=simple_initial_values,
             params=simple_parameters
         )
         assert result == 'dict'
     
     
     def test_classify_inputs_mixed(solver, system):
         """Test that mixed inputs (dict + array) are classified as 'dict'."""
         n_states = solver.system_sizes.states
         inits_array = np.ones((n_states, 2), dtype=solver.precision)
         params_dict = {list(system.parameters.names)[0]: [1.0, 2.0]}
     
         result = solver.input_handler.classify_inputs(
             states=inits_array,
             params=params_dict
         )
         assert result == 'dict'
     
         # Test the reverse case
         inits_dict = {list(system.initial_values.names)[0]: [0.1, 0.2]}
         n_params = solver.system_sizes.parameters
         params_array = np.ones((n_params, 2), dtype=solver.precision)
     
         result = solver.input_handler.classify_inputs(
             states=inits_dict,
             params=params_array
         )
         assert result == 'dict'
     
     
     def test_classify_inputs_array(solver):
         """Test that matching numpy arrays are classified as 'array'."""
         n_states = solver.system_sizes.states
         n_params = solver.system_sizes.parameters
         n_runs = 4
     
         inits = np.ones((n_states, n_runs), dtype=solver.precision)
         params = np.ones((n_params, n_runs), dtype=solver.precision)
     
         result = solver.input_handler.classify_inputs(
             states=inits,
             params=params
         )
         assert result == 'array'
     
     
     def test_classify_inputs_mismatched_runs(solver):
         """Test that mismatched run counts fall back to 'dict'."""
         n_states = solver.system_sizes.states
         n_params = solver.system_sizes.parameters
     
         inits = np.ones((n_states, 3), dtype=solver.precision)
         params = np.ones((n_params, 5), dtype=solver.precision)
     
         result = solver.input_handler.classify_inputs(
             states=inits,
             params=params
         )
         assert result == 'dict'
     
     
     def test_classify_inputs_wrong_var_count(solver):
         """Test that wrong variable counts fall back to 'dict'."""
         n_params = solver.system_sizes.parameters
         n_runs = 4
     
         # Wrong number of states
         inits = np.ones((999, n_runs), dtype=solver.precision)
         params = np.ones((n_params, n_runs), dtype=solver.precision)
     
         result = solver.input_handler.classify_inputs(
             states=inits,
             params=params
         )
         assert result == 'dict'
     
     
     def test_classify_inputs_1d_arrays(solver):
         """Test that 1D arrays fall back to 'dict'."""
         n_states = solver.system_sizes.states
         n_params = solver.system_sizes.parameters
     
         inits = np.ones(n_states, dtype=solver.precision)
         params = np.ones(n_params, dtype=solver.precision)
     
         result = solver.input_handler.classify_inputs(
             states=inits,
             params=params
         )
         assert result == 'dict'
     ```

10. **Update test_validate_arrays_dtype_cast to use input_handler**
    - File: tests/batchsolving/test_solver.py
    - Action: Modify lines 656-671
    - Details:
      ```python
      def test_validate_arrays_dtype_cast(solver):
          """Test that arrays are cast to system precision."""
          n_states = solver.system_sizes.states
          n_params = solver.system_sizes.parameters
          n_runs = 2
      
          # Create arrays with wrong dtype
          wrong_dtype = np.float64 if solver.precision == np.float32 else np.float32
          inits = np.ones((n_states, n_runs), dtype=wrong_dtype)
          params = np.ones((n_params, n_runs), dtype=wrong_dtype)
      
          validated_inits, validated_params = solver.input_handler.validate_arrays(
              states=inits,
              params=params
          )
      
          assert validated_inits.dtype == solver.precision
          assert validated_params.dtype == solver.precision
      ```

**Tests to Create**:
- Test file: tests/batchsolving/test_batch_input_handler.py
- Test function: test_handler_classify_inputs_dict
- Description: Verify classify_inputs returns 'dict' for dict inputs
- Test function: test_handler_classify_inputs_array
- Description: Verify classify_inputs returns 'array' for correct arrays
- Test function: test_handler_validate_arrays
- Description: Verify validate_arrays casts to precision

**Tests to Run**:
- tests/batchsolving/test_batch_input_handler.py (entire file)
- tests/batchsolving/test_solver.py (entire file)

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: Add Positional Argument Regression Test
**Status**: [ ]
**Dependencies**: Task Groups 1, 2, 3, and 4

**Required Context**:
- File: tests/batchsolving/test_batch_input_handler.py (lines 1-50 for imports/fixtures)
- File: src/cubie/batchsolving/solver.py (lines 51-139 for solve_ivp signature)
- File: src/cubie/batchsolving/BatchInputHandler.py (lines 490-547 for __call__ signature)

**Input Validation Required**:
- None; test verifies existing validation

**Tasks**:
1. **Add positional argument regression test**
   - File: tests/batchsolving/test_batch_input_handler.py
   - Action: Add test at end of file
   - Details:
     ```python
     def test_call_positional_argument_order(input_handler, system):
         """Verify positional args to __call__ route correctly.
     
         Regression test: states must go to states bucket,
         params must go to params bucket, even with positional args.
         """
         n_states = system.sizes.states
         n_params = system.sizes.parameters
     
         # Use distinctive values to verify routing
         # States get value 1.5, params get value 99.0
         states = np.full((n_states, 2), 1.5, dtype=system.precision)
         params = np.full((n_params, 2), 99.0, dtype=system.precision)
     
         # Call with positional arguments (states first, params second)
         result_states, result_params = input_handler(states, params, "verbatim")
     
         # Verify states went to states bucket
         assert result_states[0, 0] == 1.5, "States should have value 1.5"
         # Verify params went to params bucket
         assert result_params[0, 0] == 99.0, "Params should have value 99.0"
     
         # Verify shapes are correct
         assert result_states.shape == (n_states, 2)
         assert result_params.shape == (n_params, 2)
     ```
   - Edge cases: Test would fail if arguments were accidentally swapped
   - Integration: Prevents regression of argument order

2. **Add solve_ivp positional argument regression test**
   - File: tests/batchsolving/test_solver.py
   - Action: Add test at end of file
   - Details:
     ```python
     def test_solve_ivp_positional_argument_order(system, solver_settings):
         """Verify positional args to solve_ivp route correctly.
     
         Regression test: y0 (states) must go to states bucket,
         parameters must go to params bucket, even without keywords.
         """
         n_states = system.sizes.states
         n_params = system.sizes.parameters
     
         # Use distinctive values to verify routing
         states = np.full((n_states, 2), 1.5, dtype=system.precision)
         params = np.full((n_params, 2), 99.0, dtype=system.precision)
     
         result = solve_ivp(
             system,
             states,      # positional: y0
             params,      # positional: parameters
             duration=0.01,
             dt=0.001,
             dt_save=0.01,
         )
     
         # Verify states went to states bucket (initial_values)
         assert result.initial_values[0, 0] == 1.5, \
             "States should have value 1.5"
         # Verify params went to params bucket (parameters)
         assert result.parameters[0, 0] == 99.0, \
             "Params should have value 99.0"
     ```
   - Edge cases: Test fails if arguments are swapped
   - Integration: Prevents regression in solve_ivp API

**Tests to Create**:
- Test file: tests/batchsolving/test_batch_input_handler.py
- Test function: test_call_positional_argument_order
- Description: Verify positional args route correctly to states/params buckets
- Test file: tests/batchsolving/test_solver.py
- Test function: test_solve_ivp_positional_argument_order
- Description: Verify solve_ivp positional args route correctly

**Tests to Run**:
- tests/batchsolving/test_batch_input_handler.py::test_call_positional_argument_order
- tests/batchsolving/test_solver.py::test_solve_ivp_positional_argument_order

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

### Total Task Groups: 5

### Dependency Chain:
```
Task Group 1 (File/Class Rename)
       ↓
Task Group 2 (Add Methods)
       ↓
Task Group 3 (Update Solver/Exports)
       ↓
Task Group 4 (Update Tests)
       ↓
Task Group 5 (Regression Tests)
```

### Tests to Create:
1. `test_classify_inputs_dict` - Verify dict classification
2. `test_classify_inputs_array` - Verify array classification
3. `test_classify_inputs_device` - Verify device array classification
4. `test_validate_arrays_dtype_cast` - Verify dtype casting
5. `test_handler_classify_inputs_dict` - Handler-level dict test
6. `test_handler_classify_inputs_array` - Handler-level array test
7. `test_handler_validate_arrays` - Handler-level validation test
8. `test_call_positional_argument_order` - Positional arg regression
9. `test_solve_ivp_positional_argument_order` - solve_ivp positional arg regression

### Tests to Run (Final):
- `tests/batchsolving/test_batch_input_handler.py`
- `tests/batchsolving/test_solver.py`

### Estimated Complexity:
- **Task Group 1**: Medium - File rename and class refactoring
- **Task Group 2**: Medium - New methods with clear specifications
- **Task Group 3**: Medium - Solver updates with method removal
- **Task Group 4**: High - Many test file updates
- **Task Group 5**: Low - Simple regression tests

### Files Changed:
| File | Change Type |
|------|-------------|
| src/cubie/batchsolving/BatchGridBuilder.py | Renamed to BatchInputHandler.py |
| src/cubie/batchsolving/BatchInputHandler.py | Created (renamed + modified) |
| src/cubie/batchsolving/solver.py | Modified |
| src/cubie/batchsolving/__init__.py | Modified |
| tests/batchsolving/test_batch_grid_builder.py | Renamed to test_batch_input_handler.py |
| tests/batchsolving/test_batch_input_handler.py | Created (renamed + modified) |
| tests/batchsolving/conftest.py | Modified |
| tests/batchsolving/test_solver.py | Modified |
