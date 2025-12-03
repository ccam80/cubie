# Implementation Task List
# Feature: Array Sizing Consolidation (Option A)
# Plan Reference: .github/active_plans/array_sizing_consolidation/agent_plan.md

## Task Group 1: Update Summary Factory Signatures - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/outputhandling/save_summaries.py (lines 1-30, 184-231)
- File: src/cubie/outputhandling/update_summaries.py (lines 1-30, 165-220)
- File: src/cubie/outputhandling/output_sizes.py (lines 60-108) - `SummariesBufferSizes` class definition

**Input Validation Required**:
- `summaries_buffer_height_per_var`: No explicit validation needed (integer from OutputConfig which already validates)

**Tasks**:

1. **Update `save_summary_factory` signature in save_summaries.py**
   - File: src/cubie/outputhandling/save_summaries.py
   - Action: Modify
   - Details:
     ```python
     # Current signature (line 184-189):
     def save_summary_factory(
         buffer_sizes: SummariesBufferSizes,
         summarised_state_indices: Union[Sequence[int], ArrayLike],
         summarised_observable_indices: Union[Sequence[int], ArrayLike],
         summaries_list: Sequence[str],
     ) -> Callable:
     
     # Target signature:
     def save_summary_factory(
         summaries_buffer_height_per_var: int,
         summarised_state_indices: Union[Sequence[int], ArrayLike],
         summarised_observable_indices: Union[Sequence[int], ArrayLike],
         summaries_list: Sequence[str],
     ) -> Callable:
     ```
   - Additional changes in the same file:
     - Remove import: `from .output_sizes import SummariesBufferSizes` (line 28)
     - Update line 226: change `int32(buffer_sizes.per_variable)` to `int32(summaries_buffer_height_per_var)`
     - Update docstring (lines 196-225): change parameter description from `buffer_sizes` to `summaries_buffer_height_per_var`
   - Edge cases: When `summaries_buffer_height_per_var=0`, existing chain_metrics logic handles this by returning `do_nothing`

2. **Update `update_summary_factory` signature in update_summaries.py**
   - File: src/cubie/outputhandling/update_summaries.py
   - Action: Modify
   - Details:
     ```python
     # Current signature (line 165-170):
     def update_summary_factory(
         buffer_sizes: SummariesBufferSizes,
         summarised_state_indices: Union[Sequence[int], ArrayLike],
         summarised_observable_indices: Union[Sequence[int], ArrayLike],
         summaries_list: Sequence[str],
     ) -> Callable:
     
     # Target signature:
     def update_summary_factory(
         summaries_buffer_height_per_var: int,
         summarised_state_indices: Union[Sequence[int], ArrayLike],
         summarised_observable_indices: Union[Sequence[int], ArrayLike],
         summaries_list: Sequence[str],
     ) -> Callable:
     ```
   - Additional changes in the same file:
     - Remove import: `from .output_sizes import SummariesBufferSizes` (line 27)
     - Update line 205: change `int32(buffer_sizes.per_variable)` to `int32(summaries_buffer_height_per_var)`
     - Update docstring (lines 173-201): change parameter description from `buffer_sizes` to `summaries_buffer_height_per_var`
   - Edge cases: Same as above - handles zero via chain_metrics logic

**Outcomes**:

---

## Task Group 2: Update OutputFunctions.build() Method - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/outputhandling/output_functions.py (lines 1-30, 190-240, 345-354)
- File: src/cubie/outputhandling/output_config.py (lines 677-693) - `summaries_buffer_height_per_var` property

**Input Validation Required**:
- None - using already-validated properties from OutputConfig

**Tasks**:

1. **Update `build()` method in OutputFunctions**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Modify
   - Details:
     ```python
     # Current implementation (lines 205-233):
     def build(self) -> OutputFunctionCache:
         config = self.compile_settings
         summary_metrics.update(dt_save=config.dt_save, precision=config.precision)
         
         buffer_sizes = self.summaries_buffer_sizes  # Remove this line
         
         # ... save_state_factory call unchanged ...
         
         update_summary_metrics_func = update_summary_factory(
             buffer_sizes,  # Change to: config.summaries_buffer_height_per_var
             config.summarised_state_indices,
             config.summarised_observable_indices,
             config.summary_types,
         )
         
         save_summary_metrics_func = save_summary_factory(
             buffer_sizes,  # Change to: config.summaries_buffer_height_per_var
             config.summarised_state_indices,
             config.summarised_observable_indices,
             config.summary_types,
         )
     
     # Target implementation:
     def build(self) -> OutputFunctionCache:
         config = self.compile_settings
         summary_metrics.update(dt_save=config.dt_save, precision=config.precision)
         
         # ... save_state_factory call unchanged ...
         
         update_summary_metrics_func = update_summary_factory(
             config.summaries_buffer_height_per_var,
             config.summarised_state_indices,
             config.summarised_observable_indices,
             config.summary_types,
         )
         
         save_summary_metrics_func = save_summary_factory(
             config.summaries_buffer_height_per_var,
             config.summarised_state_indices,
             config.summarised_observable_indices,
             config.summary_types,
         )
     ```
   - Edge cases: `config.summaries_buffer_height_per_var` returns 0 when no summaries configured

2. **Remove `summaries_buffer_sizes` property from OutputFunctions**
   - File: src/cubie/outputhandling/output_functions.py
   - Action: Delete
   - Details:
     ```python
     # Remove this property (lines 347-349):
     @property
     def summaries_buffer_sizes(self) -> SummariesBufferSizes:
         """Summary buffer size helper built from the active configuration."""
         return SummariesBufferSizes.from_output_fns(self)
     ```
   - Also remove the import for `SummariesBufferSizes` from line 18-20:
     ```python
     # Current (line 18-20):
     from cubie.outputhandling.output_sizes import (
         SummariesBufferSizes,
         OutputArrayHeights,
     )
     # Target:
     from cubie.outputhandling.output_sizes import OutputArrayHeights
     ```

**Outcomes**:

---

## Task Group 3: Update BatchInputSizes.from_solver() - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None (can run in parallel with Groups 1-2)

**Required Context**:
- File: src/cubie/outputhandling/output_sizes.py (lines 410-436) - `BatchInputSizes.from_solver`
- File: src/cubie/outputhandling/output_sizes.py (lines 160-222) - `LoopBufferSizes` class and `from_solver` method
- Note: `solver_instance.system_sizes` is a `SystemSizes` attrs class with properties: `states`, `observables`, `parameters`, `drivers`

**Input Validation Required**:
- None - using already-validated properties from solver_instance

**Tasks**:

1. **Update `BatchInputSizes.from_solver()` to use system_sizes directly**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Current implementation (lines 412-435):
     @classmethod
     def from_solver(
         cls, solver_instance: "BatchSolverKernel"
     ) -> "BatchInputSizes":
         """Create batch input shapes based on solver metadata.
         ...
         """
         loopBufferSizes = LoopBufferSizes.from_solver(solver_instance)
         num_runs = solver_instance.num_runs
         initial_values = (num_runs, loopBufferSizes.state)
         parameters = (num_runs, loopBufferSizes.parameters)
         driver_coefficients = (None, loopBufferSizes.drivers, None)

         obj = cls(initial_values, parameters, driver_coefficients)
         return obj
     
     # Target implementation:
     @classmethod
     def from_solver(
         cls, solver_instance: "BatchSolverKernel"
     ) -> "BatchInputSizes":
         """Create batch input shapes based on solver metadata.
         ...
         """
         system_sizes = solver_instance.system_sizes
         num_runs = solver_instance.num_runs
         initial_values = (num_runs, system_sizes.states)
         parameters = (num_runs, system_sizes.parameters)
         driver_coefficients = (None, system_sizes.drivers, None)

         obj = cls(initial_values, parameters, driver_coefficients)
         return obj
     ```
   - Edge cases: When system has zero drivers, `system_sizes.drivers` is 0; tuple handles this

**Outcomes**:

---

## Task Group 4: Update Test Fixtures in conftest.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None (can run in parallel with Groups 1-3)

**Required Context**:
- File: tests/conftest.py (lines 38, 151-198, 721-766, 875-888)
- File: src/cubie/integrators/loops/ode_loop.py - `LoopBufferSettings` class definition

**Input Validation Required**:
- None - using already-validated properties from system and output_functions

**Tasks**:

1. **Remove import of `LoopBufferSizes` from conftest.py**
   - File: tests/conftest.py
   - Action: Modify
   - Details:
     ```python
     # Current import (line 38):
     from cubie.outputhandling.output_sizes import LoopBufferSizes
     
     # Action: Delete this line entirely
     ```

2. **Update `_build_loop_instance()` function signature and body**
   - File: tests/conftest.py
   - Action: Modify
   - Details:
     ```python
     # Current signature (lines 151-160):
     def _build_loop_instance(
         precision: np.dtype,
         system: SymbolicODE,
         step_object: Any,
         loop_buffer_sizes: LoopBufferSizes,
         output_functions: OutputFunctions,
         step_controller: Any,
         solver_settings: Dict[str, Any],
         driver_array: Optional[ArrayInterpolator],
     ) -> IVPLoop:
     
     # Target signature:
     def _build_loop_instance(
         precision: np.dtype,
         system: SymbolicODE,
         step_object: Any,
         output_functions: OutputFunctions,
         step_controller: Any,
         solver_settings: Dict[str, Any],
         driver_array: Optional[ArrayInterpolator],
     ) -> IVPLoop:
     ```
   - Update function body (lines 161-197):
     ```python
     # Current body uses loop_buffer_sizes parameter:
     n_error = loop_buffer_sizes.state if step_object.is_adaptive else 0
     buffer_settings = LoopBufferSettings(
         n_states=loop_buffer_sizes.state,
         n_parameters=loop_buffer_sizes.parameters,
         n_drivers=loop_buffer_sizes.drivers,
         n_observables=loop_buffer_sizes.observables,
         state_summary_buffer_height=loop_buffer_sizes.state_summaries,
         observable_summary_buffer_height=loop_buffer_sizes.observable_summaries,
         n_error=n_error,
         n_counters=0
     )
     
     # Target body uses system.sizes and output_functions directly:
     n_error = system.sizes.states if step_object.is_adaptive else 0
     buffer_settings = LoopBufferSettings(
         n_states=system.sizes.states,
         n_parameters=system.sizes.parameters,
         n_drivers=system.sizes.drivers,
         n_observables=system.sizes.observables,
         state_summary_buffer_height=output_functions.state_summaries_buffer_height,
         observable_summary_buffer_height=output_functions.observable_summaries_buffer_height,
         n_error=n_error,
         n_counters=0
     )
     ```
   - Edge cases: When system has zero observables/drivers, the sizes are 0 which LoopBufferSettings handles

3. **Update `loop` fixture to remove loop_buffer_sizes parameter**
   - File: tests/conftest.py
   - Action: Modify
   - Details:
     ```python
     # Current fixture (lines 721-742):
     @pytest.fixture(scope="session")
     def loop(
         precision,
         system,
         step_object,
         loop_buffer_sizes,  # Remove this parameter
         output_functions,
         step_controller,
         solver_settings,
         driver_array,
         loop_settings,
     ):
         return _build_loop_instance(
             precision=precision,
             system=system,
             step_object=step_object,
             loop_buffer_sizes=loop_buffer_sizes,  # Remove this argument
             output_functions=output_functions,
             step_controller=step_controller,
             solver_settings=solver_settings,
             driver_array=driver_array,
         )
     
     # Target fixture:
     @pytest.fixture(scope="session")
     def loop(
         precision,
         system,
         step_object,
         output_functions,
         step_controller,
         solver_settings,
         driver_array,
         loop_settings,
     ):
         return _build_loop_instance(
             precision=precision,
             system=system,
             step_object=step_object,
             output_functions=output_functions,
             step_controller=step_controller,
             solver_settings=solver_settings,
             driver_array=driver_array,
         )
     ```

4. **Update `loop_mutable` fixture similarly**
   - File: tests/conftest.py
   - Action: Modify
   - Details:
     ```python
     # Current fixture (lines 745-766):
     @pytest.fixture(scope="function")
     def loop_mutable(
         precision,
         system,
         step_object_mutable,
         loop_buffer_sizes_mutable,  # Remove this parameter
         output_functions_mutable,
         step_controller_mutable,
         solver_settings,
         driver_array,
         loop_settings,
     ):
         return _build_loop_instance(
             precision=precision,
             system=system,
             step_object=step_object_mutable,
             loop_buffer_sizes=loop_buffer_sizes_mutable,  # Remove this argument
             output_functions=output_functions_mutable,
             step_controller=step_controller_mutable,
             solver_settings=solver_settings,
             driver_array=driver_array,
         )
     
     # Target fixture (remove loop_buffer_sizes_mutable parameter and argument):
     @pytest.fixture(scope="function")
     def loop_mutable(
         precision,
         system,
         step_object_mutable,
         output_functions_mutable,
         step_controller_mutable,
         solver_settings,
         driver_array,
         loop_settings,
     ):
         return _build_loop_instance(
             precision=precision,
             system=system,
             step_object=step_object_mutable,
             output_functions=output_functions_mutable,
             step_controller=step_controller_mutable,
             solver_settings=solver_settings,
             driver_array=driver_array,
         )
     ```

5. **Remove `loop_buffer_sizes` fixture**
   - File: tests/conftest.py
   - Action: Delete
   - Details:
     ```python
     # Remove this fixture entirely (lines 875-879):
     @pytest.fixture(scope="session")
     def loop_buffer_sizes(system, output_functions):
         """Loop buffer sizes derived from the system and output configuration."""
         return LoopBufferSizes.from_system_and_output_fns(system, output_functions)
     ```

6. **Remove `loop_buffer_sizes_mutable` fixture**
   - File: tests/conftest.py
   - Action: Delete
   - Details:
     ```python
     # Remove this fixture entirely (lines 882-888):
     @pytest.fixture(scope="function")
     def loop_buffer_sizes_mutable(system, output_functions_mutable):
         """Function-scoped buffer sizes derived from the mutable outputs."""
         return LoopBufferSizes.from_system_and_output_fns(
             system, output_functions_mutable
         )
     ```

**Outcomes**:

---

## Task Group 5: Remove Classes from output_sizes.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1, 2, 3, 4 (all consumers must be updated first)

**Required Context**:
- File: src/cubie/outputhandling/output_sizes.py (lines 59-222)
- Ensure no remaining references to `SummariesBufferSizes` or `LoopBufferSizes` in codebase

**Input Validation Required**:
- None

**Tasks**:

1. **Remove `SummariesBufferSizes` class from output_sizes.py**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Delete
   - Details:
     ```python
     # Remove this entire class (lines 59-108):
     @attrs.define
     class SummariesBufferSizes(ArraySizingClass):
         """Buffer heights for summary metric staging buffers.
         ...
         """
         state: int = attrs.field(...)
         observables: int = attrs.field(...)
         per_variable: int = attrs.field(...)

         @classmethod
         def from_output_fns(cls, output_fns: "OutputFunctions") -> "SummariesBufferSizes":
             ...
     ```

2. **Remove `LoopBufferSizes` class from output_sizes.py**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Delete
   - Details:
     ```python
     # Remove this entire class (lines 111-222):
     @attrs.define
     class LoopBufferSizes(ArraySizingClass):
         """Staging buffer sizes consumed inside the integrator loop.
         ...
         """
         state_summaries: int = attrs.field(...)
         observable_summaries: int = attrs.field(...)
         state: int = attrs.field(...)
         observables: int = attrs.field(...)
         dxdt: int = attrs.field(...)
         parameters: int = attrs.field(...)
         drivers: int = attrs.field(...)

         @classmethod
         def from_system_and_output_fns(cls, system, output_fns) -> "LoopBufferSizes":
             ...

         @classmethod
         def from_solver(cls, solver_instance) -> "LoopBufferSizes":
             ...
     ```

3. **Update module docstring in output_sizes.py**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Current docstring (lines 1-8):
     """Sizing helpers for output buffers and arrays.
     
     The classes in this module compute buffer and array shapes needed for CUDA
     batch solving, covering temporary loop storage as well as host-visible output
     layouts. Each class inherits from :class:`ArraySizingClass`, which offers a
     utility for coercing zero-sized buffers to a minimum of one element for safe
     allocation.
     """
     
     # Target docstring:
     """Output array shape helpers for CUDA batch solving.
     
     This module computes host-visible output array dimensions for time-series
     and summary results. Classes here determine array shapes used by
     :mod:`cubie.batchsolving.arrays` for memory allocation.
     
     Internal buffer sizing for CUDA loops is handled by
     :class:`cubie.integrators.loops.ode_loop.LoopBufferSettings`.
     """
     ```

4. **Remove TYPE_CHECKING imports that are no longer needed**
   - File: src/cubie/outputhandling/output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Current TYPE_CHECKING block (lines 10-16):
     if TYPE_CHECKING:
         from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
         from cubie.outputhandling.output_functions import OutputFunctions
         from cubie.odesystems.baseODE import BaseODE
     
     # Target TYPE_CHECKING block (remove OutputFunctions and BaseODE):
     if TYPE_CHECKING:
         from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
     ```
   - Note: `OutputFunctions` was only used by `SummariesBufferSizes.from_output_fns()` and `BaseODE` was only used by `LoopBufferSizes.from_system_and_output_fns()`

**Outcomes**:

---

## Task Group 6: Update Package Exports - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 5

**Required Context**:
- File: src/cubie/outputhandling/__init__.py (entire file)

**Input Validation Required**:
- None

**Tasks**:

1. **Update imports and exports in __init__.py**
   - File: src/cubie/outputhandling/__init__.py
   - Action: Modify
   - Details:
     ```python
     # Current imports (lines 14-21):
     from cubie.outputhandling.output_sizes import (
         BatchInputSizes,
         BatchOutputSizes,
         LoopBufferSizes,
         OutputArrayHeights,
         SingleRunOutputSizes,
         SummariesBufferSizes,
     )
     
     # Target imports:
     from cubie.outputhandling.output_sizes import (
         BatchInputSizes,
         BatchOutputSizes,
         OutputArrayHeights,
         SingleRunOutputSizes,
     )
     
     # Current __all__ (lines 25-38):
     __all__ = [
         "OutputCompileFlags",
         "OutputConfig",
         "OutputFunctionCache",
         "OutputFunctions",
         "OutputArrayHeights",
         "SummariesBufferSizes",
         "LoopBufferSizes",
         "SingleRunOutputSizes",
         "BatchInputSizes",
         "BatchOutputSizes",
         "summary_metrics",
         "register_metric",
     ]
     
     # Target __all__:
     __all__ = [
         "OutputCompileFlags",
         "OutputConfig",
         "OutputFunctionCache",
         "OutputFunctions",
         "OutputArrayHeights",
         "SingleRunOutputSizes",
         "BatchInputSizes",
         "BatchOutputSizes",
         "summary_metrics",
         "register_metric",
     ]
     ```

**Outcomes**:

---

## Task Group 7: Update Tests - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 5, 6

**Required Context**:
- File: tests/outputhandling/test_output_sizes.py (entire file)

**Input Validation Required**:
- None

**Tasks**:

1. **Remove `SummariesBufferSizes` import from test_output_sizes.py**
   - File: tests/outputhandling/test_output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Current imports (lines 9-16):
     from cubie.outputhandling.output_sizes import (
         SummariesBufferSizes,
         LoopBufferSizes,
         OutputArrayHeights,
         SingleRunOutputSizes,
         BatchOutputSizes,
         BatchInputSizes,
     )
     
     # Target imports:
     from cubie.outputhandling.output_sizes import (
         OutputArrayHeights,
         SingleRunOutputSizes,
         BatchOutputSizes,
         BatchInputSizes,
     )
     ```

2. **Update `TestNonzeroProperty` class to use remaining classes**
   - File: tests/outputhandling/test_output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Current test_nonzero_property_int_values (lines 22-29) uses SummariesBufferSizes
     # Replace with OutputArrayHeights which also has int values:
     def test_nonzero_property_int_values(self):
         """Test that nonzero property converts zero int values to 1"""
         heights = OutputArrayHeights(
             state=0, observables=0, state_summaries=5, observable_summaries=3
         )
         nonzero_heights = heights.nonzero
         
         assert nonzero_heights.state == 1
         assert nonzero_heights.observables == 1
         assert nonzero_heights.state_summaries == 5
         assert nonzero_heights.observable_summaries == 3
     
     # test_nonzero_property_tuple_values (lines 31-44) uses SingleRunOutputSizes
     # This test is fine as-is, no changes needed
     
     # test_nonzero_property_preserves_original (lines 46-59) uses SummariesBufferSizes
     # Replace with OutputArrayHeights:
     def test_nonzero_property_preserves_original(self):
         """Test that nonzero property doesn't modify the original object"""
         original = OutputArrayHeights(
             state=0, observables=3, state_summaries=0, observable_summaries=2
         )
         nonzero_copy = original.nonzero
         
         # Original should be unchanged
         assert original.state == 0
         assert original.observables == 3
         assert original.state_summaries == 0
         
         # Copy should have zeros converted to ones
         assert nonzero_copy.state == 1
         assert nonzero_copy.observables == 3
         assert nonzero_copy.state_summaries == 1
     ```

3. **Remove `TestSummariesBufferSizes` test class**
   - File: tests/outputhandling/test_output_sizes.py
   - Action: Delete
   - Details:
     ```python
     # Remove this entire class (lines 92-148):
     class TestSummariesBufferSizes:
         """Test SummariesBufferSizes class"""
         ...
     ```

4. **Remove `TestLoopBufferSizes` test class**
   - File: tests/outputhandling/test_output_sizes.py
   - Action: Delete
   - Details:
     ```python
     # Remove this entire class (lines 150-271):
     class TestLoopBufferSizes:
         """Test LoopBufferSizes class"""
         ...
     ```

5. **Update integration test `test_edge_case_all_zeros_with_nonzero`**
   - File: tests/outputhandling/test_output_sizes.py
   - Action: Modify
   - Details:
     ```python
     # Current test (lines 670-706) uses LoopBufferSizes:
     def test_edge_case_all_zeros_with_nonzero(
         self, system, solverkernel, output_functions
     ):
         """Test edge case where everything is zero but using nonzero property"""
         ...
         loop_buffer = LoopBufferSizes.from_system_and_output_fns(
             system, output_functions
         )
         ...
         nonzero_loop_buffer = loop_buffer.nonzero
         ...
         assert nonzero_loop_buffer.state >= 1
         assert nonzero_loop_buffer.observables >= 1
         assert nonzero_loop_buffer.parameters >= 1
     
     # Target test - remove LoopBufferSizes usage entirely:
     def test_edge_case_all_zeros_with_nonzero(
         self, system, solverkernel, output_functions
     ):
         """Test edge case where everything is zero but using nonzero property"""
         numruns = 0

         # Test with nonzero property - everything should become at least 1
         single_run = SingleRunOutputSizes.from_solver(solverkernel)
         batch = BatchOutputSizes.from_solver(solverkernel)

         # Use nonzero property to get nonzero versions
         nonzero_single_run = single_run.nonzero
         nonzero_batch = batch.nonzero

         assert all(v >= 1 for v in nonzero_single_run.state)
         assert all(v >= 1 for v in nonzero_single_run.observables)

         assert all(v >= 1 for v in nonzero_batch.state)
         assert all(v >= 1 for v in nonzero_batch.observables)
     ```

**Outcomes**:

---

## Summary

### Total Task Groups: 7
### Dependency Chain Overview:
```
Group 1 (Factory Signatures) ─┐
                              ├──> Group 2 (OutputFunctions.build) ─┐
                              │                                      │
Group 3 (BatchInputSizes)  ───┼──────────────────────────────────────┼──> Group 5 (Remove Classes)
                              │                                      │            │
Group 4 (Test Fixtures) ──────┘                                      │            v
                                                                     │    Group 6 (Package Exports)
                                                                     │            │
                                                                     └────────────┴──> Group 7 (Update Tests)
```

### Parallel Execution Opportunities:
- Groups 1, 3, 4 can run in parallel (no dependencies on each other)
- Group 2 requires Group 1
- Groups 5+ are sequential after all consumers are updated

### Estimated Complexity:
- **Group 1**: Medium - Two factory signature changes with docstring updates
- **Group 2**: Low - Simple method refactor
- **Group 3**: Low - Single classmethod update
- **Group 4**: Medium - Multiple fixture updates, need to verify test compatibility
- **Group 5**: Low - Delete classes, update docstrings
- **Group 6**: Low - Update exports
- **Group 7**: Medium - Carefully update tests to maintain coverage
