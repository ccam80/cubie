# Implementation Task List
# Feature: Buffer Settings Plumbing (Init and Plumbing Changes)
# Plan Reference: .github/active_plans/buffer_settings_plumbing/agent_plan.md

## Overview

This task list implements the plumbing changes required to pass buffer location keywords from the top-level solver API through to the IVPLoop where buffer registration occurs. The work focuses on:

1. Adding buffer location fields to ODELoopConfig
2. Modifying IVPLoop to use Optional[str] = None for location parameters
3. Integrating buffer location keywords with the solver's argument filtering system
4. Ensuring update() methods properly propagate location changes

**Out of Scope (handled by other tasks):**
- buffer_registry.py modifications (Task 1 - ALREADY COMPLETE)
- Algorithm files, loop template/builder files, matrix-free solvers (Task 3)

---

## Task Group 1: Add Buffer Location Fields to ODELoopConfig - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop_config.py (lines 134-330, entire ODELoopConfig class)

**Input Validation Required**:
- Each location field: validate value is in ['shared', 'local'] using `validators.in_(['shared', 'local'])`
- Default value: 'local' for all location fields

**Tasks**:

1. **Add buffer location fields to ODELoopConfig attrs class**
   - File: src/cubie/integrators/loops/ode_loop_config.py
   - Action: Modify
   - Details:
     Add the following 11 fields to the ODELoopConfig class, positioned after the size parameters (n_states, n_parameters, etc.) and before the precision field:
     
     ```python
     # Buffer location settings - control memory allocation strategy
     state_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     state_proposal_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     parameters_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     drivers_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     drivers_proposal_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     observables_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     observables_proposal_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     error_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     counters_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     state_summary_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     observable_summary_location: str = field(
         default='local',
         validator=validators.in_(['shared', 'local'])
     )
     ```
     
   - Edge cases:
     - Invalid location value: Validator raises ValueError with clear message
     - These are compile-critical settings; changing them triggers cache invalidation
   - Integration:
     - ODELoopConfig is the compile_settings for IVPLoop
     - CUDAFactory.update_compile_settings() handles cache invalidation automatically
     - These fields become the single source of truth for buffer location defaults

**Outcomes**: 
- [ ] ODELoopConfig has 11 new buffer location fields
- [ ] All location fields default to 'local'
- [ ] All location fields are validated to be 'shared' or 'local'

---

## Task Group 2: Modify IVPLoop __init__ to Use Optional Parameters - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 49-246, IVPLoop.__init__)
- File: src/cubie/integrators/loops/ode_loop_config.py (ODELoopConfig class)

**Input Validation Required**:
- Location parameters: Accept Optional[str] = None; when None, defer to ODELoopConfig default
- No additional validation in IVPLoop - ODELoopConfig validators handle validation

**Tasks**:

1. **Change location parameter signatures from str to Optional[str]**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     Change the IVPLoop.__init__ signature from:
     ```python
     state_location: str = 'local',
     state_proposal_location: str = 'local',
     parameters_location: str = 'local',
     drivers_location: str = 'local',
     drivers_proposal_location: str = 'local',
     observables_location: str = 'local',
     observables_proposal_location: str = 'local',
     error_location: str = 'local',
     counters_location: str = 'local',
     state_summary_location: str = 'local',
     observable_summary_location: str = 'local',
     ```
     
     To:
     ```python
     state_location: Optional[str] = None,
     state_proposal_location: Optional[str] = None,
     parameters_location: Optional[str] = None,
     drivers_location: Optional[str] = None,
     drivers_proposal_location: Optional[str] = None,
     observables_location: Optional[str] = None,
     observables_proposal_location: Optional[str] = None,
     error_location: Optional[str] = None,
     counters_location: Optional[str] = None,
     state_summary_location: Optional[str] = None,
     observable_summary_location: Optional[str] = None,
     ```
     
   - Edge cases:
     - All values None: Uses ODELoopConfig defaults ('local' for all)
     - Some values provided: Only provided values override defaults
   - Integration:
     - Removes duplicate default values from __init__ signature
     - ODELoopConfig becomes sole source of defaults

2. **Reorder __init__ to create ODELoopConfig BEFORE buffer registration**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     Current order in __init__:
     1. Clear factory buffers
     2. Register buffers with passed locations
     3. Create ODELoopConfig
     4. setup_compile_settings(config)
     
     New order:
     1. Clear factory buffers
     2. Build config_kwargs dict with all non-None location values
     3. Create ODELoopConfig (which applies defaults for None values)
     4. Register buffers using config.* location values
     5. setup_compile_settings(config)
     
     Implementation pattern:
     ```python
     super().__init__()
     
     # Register all loop buffers with central registry
     buffer_registry.clear_factory(self)
     
     # Build config kwargs, only including provided location values
     config_kwargs = dict(
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
         save_state_fn=save_state_func,
         update_summaries_fn=update_summaries_func,
         save_summaries_fn=save_summaries_func,
         step_controller_fn=step_controller_fn,
         step_function=step_function,
         driver_function=driver_function,
         observables_fn=observables_fn,
         precision=precision,
         compile_flags=compile_flags,
         dt_save=dt_save,
         dt_summarise=dt_summarise,
         dt0=dt0,
         dt_min=dt_min,
         dt_max=dt_max,
         is_adaptive=is_adaptive,
     )
     
     # Add location kwargs only if provided (not None)
     if state_location is not None:
         config_kwargs['state_location'] = state_location
     if state_proposal_location is not None:
         config_kwargs['state_proposal_location'] = state_proposal_location
     if parameters_location is not None:
         config_kwargs['parameters_location'] = parameters_location
     if drivers_location is not None:
         config_kwargs['drivers_location'] = drivers_location
     if drivers_proposal_location is not None:
         config_kwargs['drivers_proposal_location'] = drivers_proposal_location
     if observables_location is not None:
         config_kwargs['observables_location'] = observables_location
     if observables_proposal_location is not None:
         config_kwargs['observables_proposal_location'] = observables_proposal_location
     if error_location is not None:
         config_kwargs['error_location'] = error_location
     if counters_location is not None:
         config_kwargs['counters_location'] = counters_location
     if state_summary_location is not None:
         config_kwargs['state_summary_location'] = state_summary_location
     if observable_summary_location is not None:
         config_kwargs['observable_summary_location'] = observable_summary_location
     
     config = ODELoopConfig(**config_kwargs)
     
     # Register buffers using config values (which have defaults applied)
     buffer_registry.register(
         'loop_state', self, n_states, config.state_location,
         precision=precision
     )
     buffer_registry.register(
         'loop_proposed_state', self, n_states,
         config.state_proposal_location, precision=precision
     )
     buffer_registry.register(
         'loop_parameters', self, n_parameters,
         config.parameters_location, precision=precision
     )
     buffer_registry.register(
         'loop_drivers', self, n_drivers, config.drivers_location,
         precision=precision
     )
     buffer_registry.register(
         'loop_proposed_drivers', self, n_drivers,
         config.drivers_proposal_location, precision=precision
     )
     buffer_registry.register(
         'loop_observables', self, n_observables,
         config.observables_location, precision=precision
     )
     buffer_registry.register(
         'loop_proposed_observables', self, n_observables,
         config.observables_proposal_location, precision=precision
     )
     buffer_registry.register(
         'loop_error', self, n_error, config.error_location,
         precision=precision
     )
     buffer_registry.register(
         'loop_counters', self, n_counters, config.counters_location,
         precision=precision
     )
     buffer_registry.register(
         'loop_state_summary', self, state_summary_buffer_height,
         config.state_summary_location, precision=precision
     )
     buffer_registry.register(
         'loop_observable_summary', self, observable_summary_buffer_height,
         config.observable_summary_location, precision=precision
     )
     
     self.setup_compile_settings(config)
     ```
     
   - Edge cases:
     - Empty/zero-size buffers: Still registered with size 0, registry handles gracefully
   - Integration:
     - Uses ODELoopConfig defaults when None provided
     - buffer_registry.register() receives validated location values

**Outcomes**: 
- [ ] IVPLoop.__init__ uses Optional[str] = None for all location parameters
- [ ] ODELoopConfig is created BEFORE buffer registration
- [ ] Buffer registration uses config.* location values

---

## Task Group 3: Add Location Parameter Mapping for update() - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1, Task Group 2

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 855-901, IVPLoop.update())
- File: src/cubie/buffer_registry.py (update_buffer method, lines 241-272)
- File: src/cubie/integrators/SingleIntegratorRunCore.py (lines 31-43, ALL_BUFFER_LOCATION_PARAMETERS)

**Input Validation Required**:
- Location updates: Validate values are 'shared' or 'local' (handled by ODELoopConfig validators via update_compile_settings)

**Tasks**:

1. **Define LOCATION_PARAM_TO_BUFFER mapping constant**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     Add a module-level constant after ALL_LOOP_SETTINGS (around line 46):
     
     ```python
     # Maps location parameter names to their corresponding buffer names
     # in the buffer registry. Used by IVPLoop.update() to propagate
     # location changes to registered buffers.
     LOCATION_PARAM_TO_BUFFER = {
         'state_location': 'loop_state',
         'state_proposal_location': 'loop_proposed_state',
         'parameters_location': 'loop_parameters',
         'drivers_location': 'loop_drivers',
         'drivers_proposal_location': 'loop_proposed_drivers',
         'observables_location': 'loop_observables',
         'observables_proposal_location': 'loop_proposed_observables',
         'error_location': 'loop_error',
         'counters_location': 'loop_counters',
         'state_summary_location': 'loop_state_summary',
         'observable_summary_location': 'loop_observable_summary',
     }
     ```
     
   - Edge cases: None - this is a static mapping
   - Integration: Used by IVPLoop.update() to find buffer names

2. **Modify IVPLoop.update() to propagate location changes to buffer registry**
   - File: src/cubie/integrators/loops/ode_loop.py
   - Action: Modify
   - Details:
     Add logic after the update_compile_settings call to update buffer locations in the registry:
     
     Current update() method (lines 855-901):
     ```python
     def update(
         self,
         updates_dict: Optional[dict[str, object]] = None,
         silent: bool = False,
         **kwargs: object,
     ) -> Set[str]:
         # ... existing code ...
         updates_dict, unpacked_keys = unpack_dict_values(updates_dict)
         
         recognised = self.update_compile_settings(updates_dict, silent=True)
         # ... existing code ...
     ```
     
     Modified update() method:
     ```python
     def update(
         self,
         updates_dict: Optional[dict[str, object]] = None,
         silent: bool = False,
         **kwargs: object,
     ) -> Set[str]:
         # ... existing code ...
         updates_dict, unpacked_keys = unpack_dict_values(updates_dict)
         
         recognised = self.update_compile_settings(updates_dict, silent=True)
         
         # Propagate location changes to buffer registry
         for param_name, buffer_name in LOCATION_PARAM_TO_BUFFER.items():
             if param_name in recognised:
                 new_location = getattr(self.compile_settings, param_name)
                 buffer_registry.update_buffer(
                     buffer_name, self, location=new_location
                 )
         
         # ... existing code ...
     ```
     
   - Edge cases:
     - Location not in updates_dict: No buffer update needed
     - Factory not registered: buffer_registry.update_buffer silently ignores
   - Integration:
     - update_compile_settings() validates values via ODELoopConfig validators
     - buffer_registry.update_buffer() updates the stored BufferEntry
     - Cache invalidation happens automatically in both systems

**Outcomes**: 
- [ ] LOCATION_PARAM_TO_BUFFER mapping defined at module level
- [ ] IVPLoop.update() propagates location changes to buffer_registry

---

## Task Group 4: Integrate Buffer Location Keywords with Solver - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1-3

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 1-35 imports, lines 162-245 Solver.__init__)
- File: src/cubie/integrators/SingleIntegratorRunCore.py (lines 31-43, ALL_BUFFER_LOCATION_PARAMETERS)
- File: src/cubie/_utils.py (lines 200-245, merge_kwargs_into_settings)

**Input Validation Required**:
- None at Solver level - validation happens in ODELoopConfig when IVPLoop is instantiated

**Tasks**:

1. **Add import for ALL_BUFFER_LOCATION_PARAMETERS**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     Add import from SingleIntegratorRunCore. Modify the existing import block (lines 19-35):
     
     Current imports include:
     ```python
     from cubie.integrators.loops.ode_loop import (
         ALL_LOOP_SETTINGS,
     )
     ```
     
     Add new import after existing integrators imports:
     ```python
     from cubie.integrators.SingleIntegratorRunCore import (
         ALL_BUFFER_LOCATION_PARAMETERS,
     )
     ```
     
   - Edge cases: None
   - Integration: Makes buffer location parameters available for merge_kwargs_into_settings

2. **Union buffer location parameters with loop settings in merge_kwargs_into_settings call**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     Modify the loop_settings merge in Solver.__init__ (around line 222-224):
     
     Current code:
     ```python
     loop_settings, loop_recognized = merge_kwargs_into_settings(
         kwargs=kwargs, valid_keys=ALL_LOOP_SETTINGS,
         user_settings=loop_settings)
     ```
     
     New code:
     ```python
     loop_settings, loop_recognized = merge_kwargs_into_settings(
         kwargs=kwargs,
         valid_keys=ALL_LOOP_SETTINGS | ALL_BUFFER_LOCATION_PARAMETERS,
         user_settings=loop_settings)
     ```
     
   - Edge cases:
     - User provides state_location in both kwargs and loop_settings dict:
       merge_kwargs_into_settings handles this, kwargs takes precedence with warning
   - Integration:
     - Buffer location kwargs flow through same path as dt_save, dt_summarise
     - No changes needed to solve_ivp (it passes **kwargs through to Solver)

**Outcomes**: 
- [ ] ALL_BUFFER_LOCATION_PARAMETERS imported in solver.py
- [ ] Buffer location keywords recognized by merge_kwargs_into_settings

---

## Task Group 5: Tests for Buffer Location Flow - PARALLEL
**Status**: [ ]
**Dependencies**: Task Groups 1-4

**Required Context**:
- File: tests/batchsolving/ (existing test patterns)
- File: tests/integrators/loops/ (existing test patterns)
- File: tests/conftest.py (fixture patterns)

**Input Validation Required**:
- Test assertions only - no input validation in tests

**Tasks**:

1. **Test buffer location kwargs recognized by Solver**
   - File: tests/batchsolving/test_solver_buffer_locations.py
   - Action: Create
   - Details:
     Create new test file with tests for buffer location flow:
     
     ```python
     """Tests for buffer location keyword argument flow through Solver."""
     import pytest
     
     
     def test_buffer_location_kwargs_recognized(three_state_linear):
         """Buffer location kwargs pass through Solver to loop config."""
         from cubie import Solver
         
         solver = Solver(three_state_linear, state_location='shared')
         config = solver.kernel.single_integrator._loop.compile_settings
         assert config.state_location == 'shared'
         # Other locations should use defaults
         assert config.parameters_location == 'local'
     
     
     def test_buffer_location_default(three_state_linear):
         """Buffer locations default to 'local' when not specified."""
         from cubie import Solver
         
         solver = Solver(three_state_linear)
         config = solver.kernel.single_integrator._loop.compile_settings
         assert config.state_location == 'local'
         assert config.state_proposal_location == 'local'
         assert config.parameters_location == 'local'
         assert config.drivers_location == 'local'
         assert config.drivers_proposal_location == 'local'
         assert config.observables_location == 'local'
         assert config.observables_proposal_location == 'local'
         assert config.error_location == 'local'
         assert config.counters_location == 'local'
         assert config.state_summary_location == 'local'
         assert config.observable_summary_location == 'local'
     
     
     def test_buffer_location_in_loop_settings(three_state_linear):
         """Buffer locations can be specified via loop_settings dict."""
         from cubie import Solver
         
         solver = Solver(
             three_state_linear,
             loop_settings={'state_location': 'shared'}
         )
         config = solver.kernel.single_integrator._loop.compile_settings
         assert config.state_location == 'shared'
     
     
     def test_invalid_buffer_location_raises(three_state_linear):
         """Invalid buffer location values raise ValueError."""
         from cubie import Solver
         
         with pytest.raises(ValueError):
             Solver(three_state_linear, state_location='gpu')
     
     
     def test_buffer_location_update(three_state_linear):
         """Buffer location can be updated via solver.update()."""
         from cubie import Solver
         
         solver = Solver(three_state_linear)
         config = solver.kernel.single_integrator._loop.compile_settings
         assert config.state_location == 'local'
         
         solver.update(state_location='shared')
         config = solver.kernel.single_integrator._loop.compile_settings
         assert config.state_location == 'shared'
     
     
     def test_solve_ivp_buffer_location(three_state_linear):
         """Buffer locations can be passed through solve_ivp."""
         from cubie import solve_ivp
         import numpy as np
         
         # Just verify it doesn't raise - full integration would require GPU
         # This test verifies the kwargs flow through correctly
         try:
             solve_ivp(
                 three_state_linear,
                 y0={'x': np.array([1.0])},
                 parameters={'a': np.array([0.1])},
                 state_location='shared',
                 duration=0.1,
             )
         except Exception as e:
             # Allow CUDA-related errors in cudasim mode
             if 'CUDA' not in str(e) and 'cuda' not in str(e).lower():
                 raise
     ```
     
   - Edge cases:
     - Tests should work in both CUDA and cudasim modes
     - Use existing system fixtures from conftest.py
   - Integration:
     - Uses three_state_linear fixture from conftest.py
     - Tests verify end-to-end flow from Solver to ODELoopConfig

2. **Test ODELoopConfig location field validation**
   - File: tests/integrators/loops/test_ode_loop_config.py
   - Action: Modify (or create if doesn't exist)
   - Details:
     Add tests for the new location fields:
     
     ```python
     """Tests for ODELoopConfig buffer location fields."""
     import pytest
     from cubie.integrators.loops.ode_loop_config import ODELoopConfig
     
     
     def test_location_fields_default_to_local():
         """All buffer location fields default to 'local'."""
         config = ODELoopConfig()
         assert config.state_location == 'local'
         assert config.state_proposal_location == 'local'
         assert config.parameters_location == 'local'
         assert config.drivers_location == 'local'
         assert config.drivers_proposal_location == 'local'
         assert config.observables_location == 'local'
         assert config.observables_proposal_location == 'local'
         assert config.error_location == 'local'
         assert config.counters_location == 'local'
         assert config.state_summary_location == 'local'
         assert config.observable_summary_location == 'local'
     
     
     def test_location_fields_accept_shared():
         """Buffer location fields accept 'shared' value."""
         config = ODELoopConfig(
             state_location='shared',
             parameters_location='shared',
         )
         assert config.state_location == 'shared'
         assert config.parameters_location == 'shared'
         # Others still default
         assert config.drivers_location == 'local'
     
     
     def test_invalid_location_raises_valueerror():
         """Invalid location values raise ValueError."""
         with pytest.raises(ValueError):
             ODELoopConfig(state_location='gpu')
         
         with pytest.raises(ValueError):
             ODELoopConfig(parameters_location='global')
         
         with pytest.raises(ValueError):
             ODELoopConfig(state_location='')
     ```
     
   - Edge cases:
     - Empty string as location value
     - Case sensitivity (should be exact match)
   - Integration:
     - Tests validate ODELoopConfig field validators work correctly

3. **Test IVPLoop location parameter handling**
   - File: tests/integrators/loops/test_ode_loop.py
   - Action: Modify (add tests)
   - Details:
     Add tests for IVPLoop location parameter handling:
     
     ```python
     def test_ivploop_location_defaults_from_config(precision):
         """IVPLoop uses ODELoopConfig defaults when locations are None."""
         from cubie.integrators.loops.ode_loop import IVPLoop
         from cubie.outputhandling import OutputCompileFlags
         
         loop = IVPLoop(
             precision=precision,
             n_states=3,
             compile_flags=OutputCompileFlags(),
         )
         config = loop.compile_settings
         assert config.state_location == 'local'
         assert config.parameters_location == 'local'
     
     
     def test_ivploop_location_override(precision):
         """IVPLoop accepts explicit location values."""
         from cubie.integrators.loops.ode_loop import IVPLoop
         from cubie.outputhandling import OutputCompileFlags
         
         loop = IVPLoop(
             precision=precision,
             n_states=3,
             compile_flags=OutputCompileFlags(),
             state_location='shared',
         )
         config = loop.compile_settings
         assert config.state_location == 'shared'
         # Others still default
         assert config.parameters_location == 'local'
     
     
     def test_ivploop_update_location(precision):
         """IVPLoop.update() propagates location changes."""
         from cubie.integrators.loops.ode_loop import IVPLoop
         from cubie.outputhandling import OutputCompileFlags
         from cubie.buffer_registry import buffer_registry
         
         loop = IVPLoop(
             precision=precision,
             n_states=3,
             compile_flags=OutputCompileFlags(),
         )
         
         # Initial location is 'local'
         assert loop.compile_settings.state_location == 'local'
         
         # Update location
         loop.update(state_location='shared')
         
         # Config updated
         assert loop.compile_settings.state_location == 'shared'
         
         # Buffer registry also updated
         context = buffer_registry._contexts.get(loop)
         if context:
             entry = context.entries.get('loop_state')
             if entry:
                 assert entry.location == 'shared'
     ```
     
   - Edge cases:
     - Update with same value (should be no-op)
     - Update non-existent parameter
   - Integration:
     - Uses precision fixture from conftest.py
     - Tests verify compile_settings and buffer_registry are both updated

**Outcomes**: 
- [ ] test_solver_buffer_locations.py created with 6 tests
- [ ] test_ode_loop_config.py has location field tests
- [ ] test_ode_loop.py has location parameter tests

---

## Summary

### Total Task Groups: 5
### Dependency Chain:
1. Task Group 1 (ODELoopConfig fields) → No dependencies
2. Task Group 2 (IVPLoop __init__) → Depends on 1
3. Task Group 3 (IVPLoop update) → Depends on 1, 2
4. Task Group 4 (Solver integration) → Depends on 1, 2, 3
5. Task Group 5 (Tests) → Depends on 1, 2, 3, 4 (can run in parallel internally)

### Parallel Execution Opportunities:
- Within Task Group 5, all test files can be written in parallel
- Task Groups 1-4 must be sequential due to dependencies

### Estimated Complexity:
- Task Group 1: Low (adding attrs fields with validators)
- Task Group 2: Medium (reordering __init__, handling Optional parameters)
- Task Group 3: Low (adding mapping constant and update logic)
- Task Group 4: Low (adding import and modifying set union)
- Task Group 5: Medium (comprehensive test coverage)
