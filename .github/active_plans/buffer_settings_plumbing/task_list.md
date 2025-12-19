# Implementation Task List
# Feature: Buffer Settings Plumbing (Expanded Scope - ALL CUDAFactory Subclasses)
# Plan Reference: .github/active_plans/buffer_settings_plumbing/agent_plan.md

## Overview

This task list implements buffer settings plumbing for **ALL CUDAFactory subclasses** that register buffers. The scope includes:

1. Adding `update()` method to BufferRegistry
2. Adding buffer location fields to compile settings across all factories
3. Ensuring location parameters flow through existing argument filtering utilities
4. Updating instrumented test files to mirror source changes

**CRITICAL Design Requirements (from user feedback):**
- Do NOT create separate buffer-location keyword arg dicts
- Do NOT clutter __init__ with "if location is not None:" noise
- Do NOT have a separate location-param-to-buffer mapping dict
- Buffer location parameters are NOT separate from other compile settings
- Use existing `split_applicable_settings()` and `merge_kwargs_into_settings()` utilities from `cubie/_utils.py`
- Each CUDAFactory owns its own buffers and their locations

**Buffer Naming Convention:**
- IVPLoop buffers use prefix `loop_` (e.g., `loop_state`, `loop_proposed_state`)
- Newton-Krylov buffers use prefix `newton_` (e.g., `newton_delta`, `newton_residual`)
- Linear solver buffers use prefix `lin_` (e.g., `lin_preconditioned_vec`, `lin_temp`)
- Location parameter naming: `[buffer_name]_location` (e.g., `loop_state_location`, `newton_delta_location`)

---

## Task Group 1: Add update() Method to BufferRegistry - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/buffer_registry.py (entire file, especially lines 241-294 for update_buffer pattern)
- File: src/cubie/CUDAFactory.py (update_compile_settings pattern, lines 565-636)

**Input Validation Required**:
- `factory`: Must be an object that exists in `_contexts` (silent mode returns empty set if not)
- `updates_dict`: Optional dict, merged with kwargs
- `value` for location: Must be 'shared' or 'local' - raise ValueError otherwise

**Tasks**:

1. **Add required imports to buffer_registry.py**
   - File: src/cubie/buffer_registry.py
   - Action: Modify line 3
   - Details:
     Update imports at top of file to include Set and Any:
     ```python
     from typing import Callable, Dict, Optional, Any, Set
     ```

2. **Add update() method to BufferRegistry class**
   - File: src/cubie/buffer_registry.py
   - Action: Add method after `clear_factory()` method (around line 294)
   - Details:
     ```python
     def update(
         self,
         factory: object,
         updates_dict: Optional[Dict[str, Any]] = None,
         silent: bool = False,
         **kwargs: Any,
     ) -> Set[str]:
         """Update buffer locations from keyword arguments.

         For each key of the form '[buffer_name]_location', finds the
         corresponding buffer and updates its location. Mirrors the pattern
         of CUDAFactory.update_compile_settings().

         Parameters
         ----------
         factory
             Factory instance that owns the buffers to update.
         updates_dict
             Mapping of parameter names to new values.
         silent
             Suppress errors for unrecognized parameters.
         **kwargs
             Additional parameters merged into updates_dict.

         Returns
         -------
         Set[str]
             Names of parameters that were successfully recognized.

         Raises
         ------
         ValueError
             If a location value is not 'shared' or 'local'.

         Notes
         -----
         A parameter is recognized if it matches '[buffer_name]_location'
         where buffer_name is registered for the factory. The method
         silently ignores unrecognized parameters when silent=True.
         """
         if updates_dict is None:
             updates_dict = {}
         updates_dict = updates_dict.copy()
         if kwargs:
             updates_dict.update(kwargs)
         if not updates_dict:
             return set()

         if factory not in self._contexts:
             return set()

         context = self._contexts[factory]
         recognized = set()
         updated = False

         for key, value in updates_dict.items():
             if not key.endswith('_location'):
                 continue

             buffer_name = key[:-9]  # Remove '_location' suffix
             if buffer_name not in context.entries:
                 continue

             if value not in ('shared', 'local'):
                 raise ValueError(
                     f"Invalid location '{value}' for buffer "
                     f"'{buffer_name}'. Must be 'shared' or 'local'."
                 )

             entry = context.entries[buffer_name]
             if entry.location != value:
                 self.update_buffer(buffer_name, factory, location=value)
                 updated = True

             recognized.add(key)

         if updated:
             context.invalidate_layouts()

         return recognized
     ```
   - Edge cases:
     - Empty updates_dict: Returns empty set
     - Factory not registered: Returns empty set silently
     - Invalid location value: Raises ValueError
     - Buffer name not found: Silently skipped, key not in recognized set
   - Integration:
     - Mirrors CUDAFactory.update_compile_settings() pattern
     - Works with existing update_buffer() method
     - Invalidates layouts when locations change

**Outcomes**: 
- [ ] BufferRegistry.update() method added at correct location
- [ ] Method follows same pattern as CUDAFactory.update_compile_settings()
- [ ] Required imports added (Set, Any)
- [ ] Method handles edge cases correctly

---

## Task Group 2: Add Location Fields to ODELoopConfig - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop_config.py (entire ODELoopConfig class, lines 134-330)
- File: src/cubie/integrators/loops/ode_loop.py (buffer registrations, lines 172-217)

**Input Validation Required**:
- All location fields: Must be 'shared' or 'local' using `validators.in_(['shared', 'local'])`

**Tasks**:

1. **Add buffer location fields to ODELoopConfig attrs class**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Add 11 location fields after `algorithm_local_len` field (around line 212) and before `precision` field
   - Details:
     ```python
     # Buffer location settings (match buffer names in IVPLoop.__init__)
     loop_state_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     loop_proposed_state_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     loop_parameters_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     loop_drivers_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     loop_proposed_drivers_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     loop_observables_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     loop_proposed_observables_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     loop_error_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     loop_counters_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     loop_state_summary_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     loop_observable_summary_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     ```
   - Note: Field names match buffer names registered in IVPLoop (e.g., `loop_state` buffer → `loop_state_location` field)

**Outcomes**: 
- [ ] ODELoopConfig has 11 buffer location fields with `loop_` prefix
- [ ] All fields validated to be 'shared' or 'local'
- [ ] All fields default to 'local'
- [ ] Field names match buffer names exactly

---

## Task Group 3: Update IVPLoop for Location Config Integration - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (entire file)
  - __init__ method: lines 130-246
  - update() method: lines 855-900
  - Buffer registration: lines 172-217

**Input Validation Required**:
- Location params in __init__: Validated by ODELoopConfig attrs validators
- update() method: Delegate validation to buffer_registry.update()

**Tasks**:

1. **Update IVPLoop.__init__ to pass location params to ODELoopConfig**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify ODELoopConfig instantiation (around line 219-245)
   - Details:
     Add location fields to config creation:
     ```python
     config = ODELoopConfig(
         n_states=n_states,
         n_parameters=n_parameters,
         n_drivers=n_drivers,
         n_observables=n_observables,
         n_error=n_error,
         n_counters=n_counters,
         state_summary_buffer_height=state_summary_buffer_height,
         observable_summary_buffer_height=observable_summary_buffer_height,
         controller_local_len=controller_local_len,
         algorithm_local_len=algorithm_local_len,
         # Add location fields
         loop_state_location=state_location,
         loop_proposed_state_location=state_proposal_location,
         loop_parameters_location=parameters_location,
         loop_drivers_location=drivers_location,
         loop_proposed_drivers_location=drivers_proposal_location,
         loop_observables_location=observables_location,
         loop_proposed_observables_location=observables_proposal_location,
         loop_error_location=error_location,
         loop_counters_location=counters_location,
         loop_state_summary_location=state_summary_location,
         loop_observable_summary_location=observable_summary_location,
         # Existing fields continue...
         save_state_fn=save_state_func,
         # ...
     )
     ```

2. **Update IVPLoop.update() to call buffer_registry.update()**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify update() method (lines 855-900)
   - Details:
     Add call to buffer_registry.update() after update_compile_settings():
     ```python
     def update(
         self,
         updates_dict: Optional[dict[str, object]] = None,
         silent: bool = False,
         **kwargs: object,
     ) -> Set[str]:
         """Update compile settings through the CUDAFactory interface.
         ...
         """
         if updates_dict is None:
             updates_dict = {}
         updates_dict = updates_dict.copy()
         if kwargs:
             updates_dict.update(kwargs)
         if updates_dict == {}:
             return set()

         updates_dict, unpacked_keys = unpack_dict_values(updates_dict)

         recognised = self.update_compile_settings(updates_dict, silent=True)

         # Update buffer locations in registry
         recognised |= buffer_registry.update(self, updates_dict, silent=True)

         unrecognised = set(updates_dict.keys()) - recognised
         if not silent and unrecognised:
             raise KeyError(
                 f"Unrecognized parameters in update: {unrecognised}. "
                 "These parameters were not updated.",
             )
         return recognised | unpacked_keys
     ```

**Outcomes**: 
- [ ] IVPLoop.__init__ passes location params to ODELoopConfig
- [ ] IVPLoop.update() calls buffer_registry.update()
- [ ] Location updates propagate to both compile_settings and buffer_registry
- [ ] Cache invalidation occurs when locations change

---

## Task Group 4: Add Location Fields to ImplicitStepConfig - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (entire file)
  - ImplicitStepConfig class: lines 22-138
  - ODEImplicitStep class: lines 141-389
  - build_implicit_helpers() method: lines 226-293
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
  - newton_krylov_solver_factory: lines 18-316 (buffer registrations at lines 91-104)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
  - linear_solver_factory: lines 19-252 (buffer registrations at lines 88-94)

**Input Validation Required**:
- All location fields: Must be 'shared' or 'local' using `validators.in_(['shared', 'local'])`

**Tasks**:

1. **Add Newton-Krylov buffer location fields to ImplicitStepConfig**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Add location fields after `newton_max_backtracks` field (around line 90)
   - Details:
     ```python
     # Newton-Krylov buffer locations (match buffer names in newton_krylov.py)
     newton_delta_location: str = attrs.field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     newton_residual_location: str = attrs.field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     newton_residual_temp_location: str = attrs.field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     newton_stage_base_bt_location: str = attrs.field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     # Linear solver buffer locations (match buffer names in linear_solver.py)
     lin_preconditioned_vec_location: str = attrs.field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     lin_temp_location: str = attrs.field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     ```
   - Note: Add `from attrs import validators` import if not already present

2. **Update build_implicit_helpers() to pass location values to solver factories**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify linear_solver_factory and newton_krylov_solver_factory calls (lines 270-293)
   - Details:
     ```python
     linear_solver = linear_solver_factory(
         operator,
         n=n,
         factory=self,  # Pass self as factory for buffer registration
         precision=self.precision,
         preconditioner=preconditioner,
         correction_type=correction_type,
         tolerance=krylov_tolerance,
         max_iters=max_linear_iters,
         preconditioned_vec_location=config.lin_preconditioned_vec_location,
         temp_location=config.lin_temp_location,
     )

     nonlinear_solver = newton_krylov_solver_factory(
         residual_function=residual,
         linear_solver=linear_solver,
         n=n,
         factory=self,  # Pass self as factory for buffer registration
         tolerance=newton_tolerance,
         max_iters=max_newton_iters,
         damping=newton_damping,
         max_backtracks=newton_max_backtracks,
         precision=self.precision,
         delta_location=config.newton_delta_location,
         residual_location=config.newton_residual_location,
         residual_temp_location=config.newton_residual_temp_location,
         stage_base_bt_location=config.newton_stage_base_bt_location,
     )
     ```

3. **Add buffer_registry import to ode_implicitstep.py**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Add import at top of file (around line 6)
   - Details:
     ```python
     from cubie.buffer_registry import buffer_registry
     ```

**Outcomes**: 
- [ ] ImplicitStepConfig has 6 buffer location fields (4 Newton + 2 linear solver)
- [ ] build_implicit_helpers() passes locations to solver factories
- [ ] Solver factories register buffers with locations from config
- [ ] Factory reference (`self`) passed to solver factories for buffer ownership

---

## Task Group 5: Update Algorithm Files for Location Inheritance - PARALLEL
**Status**: [ ]
**Dependencies**: Task Groups 1, 4

**Required Context**:
- All files in src/cubie/integrators/algorithms/
- Each algorithm inherits from ODEImplicitStep or ODEExplicitStep

**Input Validation Required**:
- Explicit algorithms (no buffers): No location fields needed
- Implicit algorithms: Location fields inherited from ImplicitStepConfig
- DIRK/FIRK with custom buffers: Add stage-specific location fields

**Tasks (can run in parallel)**:

1. **backwards_euler.py** - Verify inherits from ODEImplicitStep
   - File: src/cubie/integrators/algorithms/backwards_euler.py
   - Action: Verify only - uses ImplicitStepConfig, no changes needed
   - Details: Inherits location fields from ImplicitStepConfig

2. **backwards_euler_predict_correct.py** - Verify inherits correctly
   - File: src/cubie/integrators/algorithms/backwards_euler_predict_correct.py
   - Action: Verify only - no changes needed

3. **crank_nicolson.py** - Verify inherits correctly
   - File: src/cubie/integrators/algorithms/crank_nicolson.py
   - Action: Verify only - no changes needed

4. **explicit_euler.py** - No location fields needed
   - File: src/cubie/integrators/algorithms/explicit_euler.py
   - Action: Verify only - explicit methods don't register buffers

5. **generic_dirk.py** - Check for DIRK-specific buffers
   - File: src/cubie/integrators/algorithms/generic_dirk.py
   - Action: Review and add location fields if DIRK registers custom buffers
   - Details: Check if DIRKStep registers buffers beyond base implicit step

6. **generic_erk.py** - No location fields needed
   - File: src/cubie/integrators/algorithms/generic_erk.py
   - Action: Verify only - explicit methods don't register buffers

7. **generic_firk.py** - Check for FIRK-specific buffers
   - File: src/cubie/integrators/algorithms/generic_firk.py
   - Action: Review and add location fields if FIRK registers custom buffers

8. **generic_rosenbrock_w.py** - Check for Rosenbrock-specific buffers
   - File: src/cubie/integrators/algorithms/generic_rosenbrock_w.py
   - Action: Review and add location fields if Rosenbrock registers custom buffers

9. **ode_explicitstep.py** - Base explicit step
   - File: src/cubie/integrators/algorithms/ode_explicitstep.py
   - Action: Verify only - explicit base class has no buffers

10. **base_algorithm_step.py** - Add location params to ALL_ALGORITHM_STEP_PARAMETERS
    - File: src/cubie/integrators/algorithms/base_algorithm_step.py
    - Action: Modify ALL_ALGORITHM_STEP_PARAMETERS set (around line 23)
    - Details:
      Add all location parameter names to the set:
      ```python
      ALL_ALGORITHM_STEP_PARAMETERS = {
          'algorithm',
          'precision', 'n', 'dxdt_function', 'observables_function',
          'driver_function', 'get_solver_helper_fn', "driver_del_t",
          'beta', 'gamma', 'M', 'preconditioner_order', 'krylov_tolerance',
          'max_linear_iters', 'linear_correction_type', 'newton_tolerance',
          'max_newton_iters', 'newton_damping', 'newton_max_backtracks',
          'n_drivers',
          # Buffer location parameters
          'newton_delta_location', 'newton_residual_location',
          'newton_residual_temp_location', 'newton_stage_base_bt_location',
          'lin_preconditioned_vec_location', 'lin_temp_location',
      }
      ```

**Outcomes**: 
- [ ] All algorithm files properly inherit location handling
- [ ] Algorithm-specific buffers have location fields in their configs
- [ ] ALL_ALGORITHM_STEP_PARAMETERS updated with location param names

---

## Task Group 6: Update Instrumented Test Files - PARALLEL
**Status**: [ ]
**Dependencies**: Task Groups 4, 5

**Required Context**:
- All files in tests/integrators/algorithms/instrumented/
- Must mirror changes made to source files exactly

**Tasks (can run in parallel)**:

1. **backwards_euler.py** - Mirror source changes
   - File: tests/integrators/algorithms/instrumented/backwards_euler.py
   - Action: Add location params to __init__ signature if source file changes

2. **backwards_euler_predict_correct.py** - Mirror source changes
   - File: tests/integrators/algorithms/instrumented/backwards_euler_predict_correct.py
   - Action: Mirror any changes from source

3. **crank_nicolson.py** - Mirror source changes
   - File: tests/integrators/algorithms/instrumented/crank_nicolson.py
   - Action: Mirror any changes from source

4. **explicit_euler.py** - Mirror source changes
   - File: tests/integrators/algorithms/instrumented/explicit_euler.py
   - Action: Mirror any changes from source

5. **generic_dirk.py** - Mirror source changes
   - File: tests/integrators/algorithms/instrumented/generic_dirk.py
   - Action: Mirror any changes from source

6. **generic_erk.py** - Mirror source changes
   - File: tests/integrators/algorithms/instrumented/generic_erk.py
   - Action: Mirror any changes from source

7. **generic_firk.py** - Mirror source changes
   - File: tests/integrators/algorithms/instrumented/generic_firk.py
   - Action: Mirror any changes from source

8. **generic_rosenbrock_w.py** - Mirror source changes
   - File: tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py
   - Action: Mirror any changes from source

9. **matrix_free_solvers.py** - Mirror newton_krylov and linear_solver changes
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Add location params to factory signatures

**CRITICAL**: Instrumented files must have IDENTICAL changes to source files except for logging additions.

**Outcomes**: 
- [ ] All instrumented files mirror source changes
- [ ] Only difference is logging/instrumentation code

---

## Task Group 7: Integration Tests - SEQUENTIAL
**Status**: [ ]
**Dependencies**: All previous task groups

**Required Context**:
- File: tests/test_buffer_registry.py (existing tests)
- File: tests/integrators/loops/test_ode_loop.py (existing tests)

**Tasks**:

1. **Test buffer_registry.update() method**
   - File: tests/test_buffer_registry.py
   - Action: Add new test functions
   - Details:
     ```python
     def test_buffer_registry_update_recognizes_location_params():
         """BufferRegistry.update() recognizes [buffer_name]_location params."""
         # Setup factory with registered buffers
         # Call update with location params
         # Assert recognized set contains the param names

     def test_buffer_registry_update_changes_buffer_location():
         """BufferRegistry.update() changes buffer location in entry."""
         # Setup factory with buffer at 'local'
         # Call update with 'shared' location
         # Verify buffer entry location changed

     def test_buffer_registry_update_invalid_location_raises():
         """BufferRegistry.update() raises ValueError for invalid location."""
         # Setup factory with buffer
         # Call update with invalid location value
         # Assert ValueError raised

     def test_buffer_registry_update_unregistered_factory_silent():
         """BufferRegistry.update() returns empty set for unregistered factory."""
         # Call update with factory that has no registered buffers
         # Assert returns empty set
     ```

2. **Test location parameter flow through IVPLoop.update()**
   - File: tests/integrators/loops/test_ode_loop.py
   - Action: Add new test functions
   - Details:
     ```python
     def test_ivploop_update_with_location_params():
         """IVPLoop.update() recognizes buffer location parameters."""
         # Create IVPLoop with default locations
         # Call update with location params
         # Assert params recognized in returned set

     def test_ivploop_update_propagates_to_registry():
         """IVPLoop.update() propagates location changes to buffer_registry."""
         # Create IVPLoop
         # Call update with location change
         # Verify buffer_registry entry updated
     ```

3. **Test algorithm location update flow**
   - File: tests/integrators/algorithms/test_algorithm_buffer_locations.py
   - Action: Create new test file
   - Details:
     ```python
     def test_implicit_step_location_params_in_config():
         """ImplicitStepConfig has buffer location fields."""
         # Create ImplicitStepConfig
         # Assert location fields exist and have defaults

     def test_implicit_step_passes_locations_to_solvers():
         """ODEImplicitStep passes location values to solver factories."""
         # Create implicit step with custom locations
         # Verify solvers received location values
     ```

**Outcomes**: 
- [ ] buffer_registry.update() tested for all cases
- [ ] IVPLoop location update flow tested
- [ ] Algorithm location handling tested

---

## Summary

### Total Task Groups: 7
### Dependency Chain:
```
Task Group 1 (BufferRegistry.update)
      ↓
      ├──────────────────────────────────────┐
      ↓                                      ↓
Task Group 2 (ODELoopConfig)         Task Group 4 (ImplicitStepConfig)
      ↓                                      ↓
Task Group 3 (IVPLoop)               Task Group 5 (Algorithm files)
      ↓                                      ↓
      └─────────────────┬────────────────────┘
                        ↓
               Task Group 6 (Instrumented files)
                        ↓
               Task Group 7 (Tests)
```

### Parallel Execution Opportunities:
- Task Groups 2, 4 can run in parallel (both depend only on 1)
- Task Group 5 subtasks can run in parallel
- Task Group 6 subtasks can run in parallel
- Task Group 7 can run after all implementation complete

### Files Modified (Total: 20+)

**Core Infrastructure (1 file)**:
- src/cubie/buffer_registry.py - Add update() method

**Loop Files (2 files)**:
- src/cubie/integrators/loops/ode_loop_config.py - Add 11 location fields
- src/cubie/integrators/loops/ode_loop.py - Pass locations to config, update() integration

**Algorithm Files (3 files modified)**:
- src/cubie/integrators/algorithms/ode_implicitstep.py - Add 6 location fields, pass to solvers
- src/cubie/integrators/algorithms/base_algorithm_step.py - Update ALL_ALGORITHM_STEP_PARAMETERS
- (Other algorithm files: verify inheritance only, no changes unless they register custom buffers)

**Instrumented Test Files (9 files)**:
- tests/integrators/algorithms/instrumented/*.py - Mirror source changes

**New/Modified Test Files (3 files)**:
- tests/test_buffer_registry.py - Add tests for update() method
- tests/integrators/loops/test_ode_loop.py - Add location update tests
- tests/integrators/algorithms/test_algorithm_buffer_locations.py - New test file

### Key Design Decisions

1. **Buffer naming convention**: Buffer names use prefixes (`loop_`, `newton_`, `lin_`) and location params use `[buffer_name]_location` format

2. **Ownership model**: Each CUDAFactory owns its buffers. Solver factories receive `factory=self` to register buffers under the owning algorithm

3. **Update flow**: 
   - `update_compile_settings()` handles location fields in config
   - `buffer_registry.update()` handles location changes in registry
   - Both paths invalidate cache

4. **No separate handling**: Location params use same filtering utilities as other compile settings
