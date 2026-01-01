# Implementation Task List
# Feature: Solver Helper Dummy Arguments
# Plan Reference: .github/active_plans/solver_helper_dummy_args/agent_plan.md

## Task Group 1: Add Dummy Argument Generation for All Solver Helpers
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/odesystems/symbolic/symbolicODE.py (lines 666-705, _generate_dummy_args method)
- File: src/cubie/odesystems/baseODE.py (lines 16-49, ODECache attrs class for all field names)
- File: src/cubie/odesystems/symbolic/codegen/linear_operators.py (lines 46-161, template signatures)
- File: src/cubie/odesystems/symbolic/codegen/preconditioners.py (lines 42-182, template signatures)
- File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py (lines 35-89, template signatures)
- File: src/cubie/odesystems/symbolic/codegen/time_derivative.py (lines 29-51, template signature)

**Input Validation Required**:
- None - this method generates dummy args for compile-time measurement only

**Tasks**:
1. **Extend _generate_dummy_args method to cover all solver helpers**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Modify
   - Details:
     ```python
     def _generate_dummy_args(self) -> Dict[str, Tuple]:
         """Generate dummy arguments for compile-time measurement.

         Returns
         -------
         Dict[str, Tuple]
             Mapping of cached output names to their argument tuples.
         """
         precision = self.precision
         sizes = self.sizes
         n_states = int(sizes.states)
         n_params = int(sizes.parameters)
         n_drivers = max(1, int(sizes.drivers))
         n_obs = max(1, int(sizes.observables))

         # n-stage helpers use a default of 2 stages
         n_stages = 2
         n_flat_states = n_stages * n_states
         n_flat_drivers = n_stages * n_drivers

         # Cached auxiliary buffer size (reasonable default)
         n_aux = max(1, n_states * 2)

         # Common arrays
         state_arr = np.ones((n_states,), dtype=precision)
         params_arr = np.ones((n_params,), dtype=precision)
         drivers_arr = np.ones((n_drivers,), dtype=precision)
         obs_arr = np.ones((n_obs,), dtype=precision)
         out_arr = np.ones((n_states,), dtype=precision)
         cached_aux_arr = np.ones((n_aux,), dtype=precision)

         # Scalars
         t = precision(0.0)
         h = precision(0.01)
         a_ij = precision(1.0)

         # n-stage arrays
         flat_state_arr = np.ones((n_flat_states,), dtype=precision)
         flat_drivers_arr = np.ones((n_flat_drivers,), dtype=precision)
         flat_out_arr = np.ones((n_flat_states,), dtype=precision)

         # dxdt(state, parameters, drivers, observables, out, t)
         dxdt_args = (
             state_arr.copy(),
             params_arr.copy(),
             drivers_arr.copy(),
             obs_arr.copy(),
             out_arr.copy(),
             t,
         )

         # get_observables(state, parameters, drivers, observables, t)
         obs_args = (
             state_arr.copy(),
             params_arr.copy(),
             drivers_arr.copy(),
             obs_arr.copy(),
             t,
         )

         # operator_apply(state, parameters, drivers, base_state,
         #                t, h, a_ij, v, out)
         linear_operator_args = (
             state_arr.copy(),
             params_arr.copy(),
             drivers_arr.copy(),
             state_arr.copy(),  # base_state
             t,
             h,
             a_ij,
             state_arr.copy(),  # v
             out_arr.copy(),
         )

         # operator_apply(state, parameters, drivers, cached_aux, base_state,
         #                t, h, a_ij, v, out)
         linear_operator_cached_args = (
             state_arr.copy(),
             params_arr.copy(),
             drivers_arr.copy(),
             cached_aux_arr.copy(),
             state_arr.copy(),  # base_state
             t,
             h,
             a_ij,
             state_arr.copy(),  # v
             out_arr.copy(),
         )

         # prepare_jac(state, parameters, drivers, t, cached_aux)
         prepare_jac_args = (
             state_arr.copy(),
             params_arr.copy(),
             drivers_arr.copy(),
             t,
             cached_aux_arr.copy(),
         )

         # calculate_cached_jvp(state, parameters, drivers, cached_aux,
         #                      t, v, out)
         calculate_cached_jvp_args = (
             state_arr.copy(),
             params_arr.copy(),
             drivers_arr.copy(),
             cached_aux_arr.copy(),
             t,
             state_arr.copy(),  # v
             out_arr.copy(),
         )

         # preconditioner(state, parameters, drivers, base_state,
         #                t, h, a_ij, v, out, jvp)
         neumann_preconditioner_args = (
             state_arr.copy(),
             params_arr.copy(),
             drivers_arr.copy(),
             state_arr.copy(),  # base_state
             t,
             h,
             a_ij,
             state_arr.copy(),  # v
             out_arr.copy(),
             state_arr.copy(),  # jvp scratch buffer
         )

         # preconditioner(state, parameters, drivers, cached_aux, base_state,
         #                t, h, a_ij, v, out, jvp)
         neumann_preconditioner_cached_args = (
             state_arr.copy(),
             params_arr.copy(),
             drivers_arr.copy(),
             cached_aux_arr.copy(),
             state_arr.copy(),  # base_state
             t,
             h,
             a_ij,
             state_arr.copy(),  # v
             out_arr.copy(),
             state_arr.copy(),  # jvp scratch buffer
         )

         # residual(u, parameters, drivers, t, h, a_ij, base_state, out)
         stage_residual_args = (
             state_arr.copy(),  # u
             params_arr.copy(),
             drivers_arr.copy(),
             t,
             h,
             a_ij,
             state_arr.copy(),  # base_state
             out_arr.copy(),
         )

         # n_stage residual(u, parameters, drivers, t, h, a_ij,
         #                  base_state, out)
         n_stage_residual_args = (
             flat_state_arr.copy(),  # u (n_stages * n_states)
             params_arr.copy(),
             flat_drivers_arr.copy(),  # (n_stages * n_drivers)
             t,
             h,
             a_ij,
             state_arr.copy(),  # base_state (n_states)
             flat_out_arr.copy(),  # out (n_stages * n_states)
         )

         # n_stage operator_apply(state, parameters, drivers, base_state,
         #                        t, h, a_ij, v, out)
         n_stage_linear_operator_args = (
             flat_state_arr.copy(),  # state (n_stages * n_states)
             params_arr.copy(),
             flat_drivers_arr.copy(),  # (n_stages * n_drivers)
             state_arr.copy(),  # base_state (n_states)
             t,
             h,
             a_ij,
             flat_state_arr.copy(),  # v (n_stages * n_states)
             flat_out_arr.copy(),  # out (n_stages * n_states)
         )

         # n_stage preconditioner(state, parameters, drivers, base_state,
         #                        t, h, a_ij, v, out, jvp)
         n_stage_neumann_preconditioner_args = (
             flat_state_arr.copy(),  # state (n_stages * n_states)
             params_arr.copy(),
             flat_drivers_arr.copy(),  # (n_stages * n_drivers)
             state_arr.copy(),  # base_state (n_states)
             t,
             h,
             a_ij,
             flat_state_arr.copy(),  # v (n_stages * n_states)
             flat_out_arr.copy(),  # out (n_stages * n_states)
             flat_state_arr.copy(),  # jvp scratch (n_stages * n_states)
         )

         # time_derivative_rhs(state, parameters, drivers, driver_dt,
         #                     observables, out, t)
         time_derivative_rhs_args = (
             state_arr.copy(),
             params_arr.copy(),
             drivers_arr.copy(),
             drivers_arr.copy(),  # driver_dt
             obs_arr.copy(),
             out_arr.copy(),
             t,
         )

         return {
             'dxdt': dxdt_args,
             'observables': obs_args,
             'linear_operator': linear_operator_args,
             'linear_operator_cached': linear_operator_cached_args,
             'prepare_jac': prepare_jac_args,
             'calculate_cached_jvp': calculate_cached_jvp_args,
             'neumann_preconditioner': neumann_preconditioner_args,
             'neumann_preconditioner_cached': neumann_preconditioner_cached_args,
             'stage_residual': stage_residual_args,
             'n_stage_residual': n_stage_residual_args,
             'n_stage_linear_operator': n_stage_linear_operator_args,
             'n_stage_neumann_preconditioner': n_stage_neumann_preconditioner_args,
             'time_derivative_rhs': time_derivative_rhs_args,
         }
     ```
   - Edge cases:
     - Zero drivers: Use `max(1, n_drivers)` to avoid empty arrays
     - Zero observables: Use `max(1, n_obs)` to avoid empty arrays
     - Zero parameters: Could use `max(1, n_params)` but parameters typically exist
   - Integration: Method is called by compile-time measurement tools via `_generate_dummy_args()`

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_symbolicode.py
- Test function: test_generate_dummy_args_returns_all_keys
  - Description: Verify that `_generate_dummy_args()` returns entries for all expected solver helper keys
- Test function: test_generate_dummy_args_correct_arities
  - Description: Verify each argument tuple has the correct number of elements matching the function signature
- Test function: test_generate_dummy_args_array_shapes
  - Description: Verify array shapes are consistent with system sizes
- Test function: test_generate_dummy_args_precision
  - Description: Verify all arrays and scalars use the correct precision dtype

**Tests to Run**:
- tests/odesystems/symbolic/test_symbolicode.py::test_generate_dummy_args_returns_all_keys
- tests/odesystems/symbolic/test_symbolicode.py::test_generate_dummy_args_correct_arities
- tests/odesystems/symbolic/test_symbolicode.py::test_generate_dummy_args_array_shapes
- tests/odesystems/symbolic/test_symbolicode.py::test_generate_dummy_args_precision

**Outcomes**: 
- Files Modified:
  * src/cubie/odesystems/symbolic/symbolicODE.py (~180 lines changed)
- Functions/Methods Modified:
  * `_generate_dummy_args()` in symbolicODE.py - Extended from 40 lines to 220 lines
- Implementation Summary:
  Extended `_generate_dummy_args` method to generate dummy argument tuples for all 13 solver helper functions. Added max(1, n_drivers) and max(1, n_obs) to avoid empty arrays, n_stages = 2 default stage count, cached_aux buffer with shape max(1, n_states * 2), and flattened arrays for n_stage functions.
- Issues Flagged: None

---

## Task Group 2: Add Tests for Dummy Argument Generation
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/odesystems/symbolic/test_symbolicode.py (entire file)
- File: tests/odesystems/symbolic/conftest.py (entire file for fixtures)
- File: src/cubie/odesystems/symbolic/symbolicODE.py (lines 666-705+, updated _generate_dummy_args method)
- File: src/cubie/odesystems/baseODE.py (lines 16-49, ODECache attrs class)

**Input Validation Required**:
- None - tests validate implementation correctness

**Tasks**:
1. **Add test for complete key coverage**
   - File: tests/odesystems/symbolic/test_symbolicode.py
   - Action: Modify (add new test function)
   - Details:
     ```python
     def test_generate_dummy_args_returns_all_keys(simple_ode_strict):
         """Verify _generate_dummy_args returns all expected solver helper keys."""
         dummy_args = simple_ode_strict._generate_dummy_args()
         
         expected_keys = {
             'dxdt',
             'observables',
             'linear_operator',
             'linear_operator_cached',
             'prepare_jac',
             'calculate_cached_jvp',
             'neumann_preconditioner',
             'neumann_preconditioner_cached',
             'stage_residual',
             'n_stage_residual',
             'n_stage_linear_operator',
             'n_stage_neumann_preconditioner',
             'time_derivative_rhs',
         }
         
         assert set(dummy_args.keys()) == expected_keys
     ```
   - Edge cases: None - this tests complete coverage
   - Integration: Uses existing `simple_ode_strict` fixture

2. **Add test for argument arities**
   - File: tests/odesystems/symbolic/test_symbolicode.py
   - Action: Modify (add new test function)
   - Details:
     ```python
     def test_generate_dummy_args_correct_arities(simple_ode_strict):
         """Verify each argument tuple has the correct arity."""
         dummy_args = simple_ode_strict._generate_dummy_args()
         
         # Expected arities based on codegen template signatures
         expected_arities = {
             'dxdt': 6,  # state, params, drivers, obs, out, t
             'observables': 5,  # state, params, drivers, obs, t
             'linear_operator': 9,  # state, params, drivers, base, t, h, a, v, out
             'linear_operator_cached': 10,  # +cached_aux
             'prepare_jac': 5,  # state, params, drivers, t, cached_aux
             'calculate_cached_jvp': 7,  # state, params, drivers, aux, t, v, out
             'neumann_preconditioner': 10,  # state, params, drivers, base, t, h, a, v, out, jvp
             'neumann_preconditioner_cached': 11,  # +cached_aux
             'stage_residual': 8,  # u, params, drivers, t, h, a, base, out
             'n_stage_residual': 8,  # u, params, drivers, t, h, a, base, out
             'n_stage_linear_operator': 9,  # state, params, drivers, base, t, h, a, v, out
             'n_stage_neumann_preconditioner': 10,  # state, params, drivers, base, t, h, a, v, out, jvp
             'time_derivative_rhs': 7,  # state, params, drivers, driver_dt, obs, out, t
         }
         
         for key, expected_arity in expected_arities.items():
             assert len(dummy_args[key]) == expected_arity, \
                 f"{key} has arity {len(dummy_args[key])}, expected {expected_arity}"
     ```
   - Edge cases: None
   - Integration: Uses existing fixture

3. **Add test for array shapes**
   - File: tests/odesystems/symbolic/test_symbolicode.py
   - Action: Modify (add new test function)
   - Details:
     ```python
     def test_generate_dummy_args_array_shapes(simple_ode_strict):
         """Verify array arguments have shapes consistent with system sizes."""
         import numpy as np
         
         dummy_args = simple_ode_strict._generate_dummy_args()
         sizes = simple_ode_strict.sizes
         n_states = int(sizes.states)
         n_params = int(sizes.parameters)
         n_drivers = max(1, int(sizes.drivers))
         n_obs = max(1, int(sizes.observables))
         n_stages = 2  # Default stage count
         
         # Check dxdt arrays
         dxdt = dummy_args['dxdt']
         assert dxdt[0].shape == (n_states,)  # state
         assert dxdt[1].shape == (n_params,)  # parameters
         assert dxdt[2].shape == (n_drivers,)  # drivers
         assert dxdt[3].shape == (n_obs,)  # observables
         assert dxdt[4].shape == (n_states,)  # out
         assert np.isscalar(dxdt[5]) or dxdt[5].shape == ()  # t
         
         # Check n_stage arrays use flattened sizes
         n_stage_res = dummy_args['n_stage_residual']
         assert n_stage_res[0].shape == (n_stages * n_states,)  # u
         assert n_stage_res[2].shape == (n_stages * n_drivers,)  # drivers
         assert n_stage_res[6].shape == (n_states,)  # base_state
         assert n_stage_res[7].shape == (n_stages * n_states,)  # out
     ```
   - Edge cases: Handles max(1, ...) for drivers/observables
   - Integration: Uses existing fixture

4. **Add test for precision consistency**
   - File: tests/odesystems/symbolic/test_symbolicode.py
   - Action: Modify (add new test function)
   - Details:
     ```python
     def test_generate_dummy_args_precision(simple_ode_strict):
         """Verify all arrays and scalars use the correct precision dtype."""
         import numpy as np
         
         dummy_args = simple_ode_strict._generate_dummy_args()
         precision = simple_ode_strict.precision
         
         for key, args in dummy_args.items():
             for i, arg in enumerate(args):
                 if isinstance(arg, np.ndarray):
                     assert arg.dtype == precision, \
                         f"{key}[{i}] has dtype {arg.dtype}, expected {precision}"
                 elif np.isscalar(arg):
                     # Scalar should be numpy scalar with correct dtype
                     assert type(arg) == precision or \
                         (hasattr(arg, 'dtype') and arg.dtype == precision), \
                         f"{key}[{i}] scalar has wrong precision"
     ```
   - Edge cases: Handles both numpy scalars and arrays
   - Integration: Uses existing fixture

**Tests to Create**:
(Already defined in Tasks above)

**Tests to Run**:
- tests/odesystems/symbolic/test_symbolicode.py::test_generate_dummy_args_returns_all_keys
- tests/odesystems/symbolic/test_symbolicode.py::test_generate_dummy_args_correct_arities
- tests/odesystems/symbolic/test_symbolicode.py::test_generate_dummy_args_array_shapes
- tests/odesystems/symbolic/test_symbolicode.py::test_generate_dummy_args_precision

**Outcomes**: 
- Files Modified:
  * tests/odesystems/symbolic/test_symbolicode.py (~97 lines added)
- Functions/Methods Added:
  * test_generate_dummy_args_returns_all_keys() - Tests all 13 solver helper keys
  * test_generate_dummy_args_correct_arities() - Tests arity of each argument tuple
  * test_generate_dummy_args_array_shapes() - Tests array shapes match system sizes
  * test_generate_dummy_args_precision() - Tests dtype consistency
- Implementation Summary:
  Added four test functions to validate the _generate_dummy_args method. Tests verify complete key coverage, correct arities matching codegen template signatures, proper array shapes for both regular and n_stage helpers, and precision dtype consistency across all arrays and scalars.
- Issues Flagged: None

---

## Summary

### Total Task Groups: 2
### Dependency Chain: Task Group 1 → Task Group 2

### Tests to Create:
1. `test_generate_dummy_args_returns_all_keys` - Verify all 13 solver helper keys are returned
2. `test_generate_dummy_args_correct_arities` - Verify each tuple has correct element count
3. `test_generate_dummy_args_array_shapes` - Verify array shapes match system sizes
4. `test_generate_dummy_args_precision` - Verify dtype consistency

### Tests to Run:
- tests/odesystems/symbolic/test_symbolicode.py (all new test functions)

### Estimated Complexity: Low
- Single method modification with well-defined signatures
- All signatures documented in codegen templates
- No architectural changes required
