# Implementation Task List
# Feature: Test Parameterization Refactoring to Reduce CUDA Compilation Sessions
# Plan Reference: .github/active_plans/test_refactor_reduce_sessions/agent_plan.md

## Task Group 1: Add RUN_DEFAULTS Constant - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: tests/_utils.py (lines 19-44 for existing parameter sets)

**Input Validation Required**:
- None (constant definition only)

**Tasks**:
1. **Add RUN_DEFAULTS constant to tests/_utils.py**
   - File: tests/_utils.py
   - Action: Modify
   - Details:
     ```python
     # Add after line 44 (after LONG_RUN_PARAMS definition)
     RUN_DEFAULTS = {
         'duration': 0.1,    # Default simulation duration
         't0': 0.0,          # Default start time
         'warmup': 0.0,      # Default warmup period
     }
     ```
   - Edge cases: None
   - Integration: Tests import and unpack this constant directly in test functions

2. **Remove duration from SHORT_RUN_PARAMS, MID_RUN_PARAMS, LONG_RUN_PARAMS**
   - File: tests/_utils.py
   - Action: Modify
   - Details:
     - Remove 'duration' key from SHORT_RUN_PARAMS (line 24)
     - Keep 'duration' in LONG_RUN_PARAMS for backward compatibility (some tests override this)
     - Note: MID_RUN_PARAMS already lacks 'duration'
     ```python
     SHORT_RUN_PARAMS = {
         'dt_save': 0.05,
         'dt_summarise': 0.05,
         'output_types': ['state', 'time', 'observables', 'mean'],
     }
     ```
   - Edge cases: Tests that explicitly need different durations will pass them as solver_settings_override
   - Integration: Tests that use SHORT_RUN_PARAMS will get duration from RUN_DEFAULTS

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: Add precision and system_type to solver_settings Defaults - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/conftest.py (lines 450-535 for solver_settings fixture)

**Input Validation Required**:
- None (fixture modification only)

**Tasks**:
1. **Add 'system_type' key to solver_settings defaults**
   - File: tests/conftest.py
   - Action: Modify
   - Details:
     Add 'system_type' key to the defaults dict in solver_settings fixture (around line 454):
     ```python
     defaults = {
         "algorithm": "euler",
         "system_type": "nonlinear",  # Add this line
         "duration": np.float64(0.2),
         # ... rest of defaults
     }
     ```
   - Edge cases: None
   - Integration: The `system` fixture will read this value

2. **Ensure 'precision' key exists in solver_settings defaults**
   - File: tests/conftest.py
   - Action: Verify (already exists at line 478)
   - Details:
     The solver_settings fixture already includes `"precision": precision` at line 478.
     Verify this is present - no change needed if already there.
   - Edge cases: None

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: Refactor precision Fixture to Read from solver_settings - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 2

**Required Context**:
- File: tests/conftest.py (lines 342-361 for precision fixtures)
- File: tests/conftest.py (lines 450-535 for solver_settings fixture)

**Input Validation Required**:
- solver_settings must contain 'precision' key (already validated by Task Group 2)

**Tasks**:
1. **Remove precision_override fixture**
   - File: tests/conftest.py
   - Action: Delete
   - Details:
     Delete lines 342-347:
     ```python
     @pytest.fixture(scope="session")
     def precision_override(request):
         if hasattr(request, "param"):
             if request.param is np.float64:
                 return np.float64
     ```
   - Edge cases: Tests using precision_override will need to use solver_settings_override instead
   - Integration: Update precision fixture

2. **Simplify precision fixture to read from solver_settings**
   - File: tests/conftest.py
   - Action: Modify
   - Details:
     Replace lines 349-361 with:
     ```python
     @pytest.fixture(scope="session")
     def precision(solver_settings_override):
         """
         Return precision from solver_settings_override, defaulting to float32.

         Usage:
         @pytest.mark.parametrize("solver_settings_override", 
             [{"precision": np.float64}], indirect=True)
         def test_something(precision):
             # precision will be np.float64 here
         """
         override = solver_settings_override if solver_settings_override else {}
         return override.get('precision', np.float32)
     ```
   - Edge cases: Tests that parameterize precision_override must be updated
   - Integration: solver_settings fixture will use this precision

3. **Update solver_settings fixture to use precision fixture**
   - File: tests/conftest.py  
   - Action: Modify
   - Details:
     The solver_settings fixture signature (line 450-451) currently:
     ```python
     def solver_settings(solver_settings_override, solver_settings_override2,
         system, precision):
     ```
     Keep precision parameter - it's derived from solver_settings_override.
     The order ensures precision is resolved before solver_settings uses it.
   - Edge cases: Circular dependency - precision comes from override, solver_settings uses precision
   - Integration: The fixture chain becomes solver_settings_override → precision → solver_settings

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: Refactor system Fixture to Read from solver_settings - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 3

**Required Context**:
- File: tests/conftest.py (lines 395-436 for system fixtures)
- File: tests/conftest.py (lines 450-535 for solver_settings fixture)

**Input Validation Required**:
- solver_settings_override may contain 'system_type' key (string value)

**Tasks**:
1. **Remove system_override fixture**
   - File: tests/conftest.py
   - Action: Delete
   - Details:
     Delete lines 395-401:
     ```python
     @pytest.fixture(scope="session")
     def system_override(request):
         """Override for system model type, if provided."""
         if hasattr(request, "param"):
             if request.param:
                 return request.param
         return "nonlinear"
     ```
   - Edge cases: Tests using system_override will need solver_settings_override
   - Integration: Update system fixture

2. **Simplify system fixture to read from solver_settings_override**
   - File: tests/conftest.py
   - Action: Modify
   - Details:
     Replace lines 404-436 with:
     ```python
     @pytest.fixture(scope="session")
     def system(request, solver_settings_override, precision):
         """
         Return the appropriate symbolic system, defaulting to nonlinear.

         Usage
         -----
         @pytest.mark.parametrize(
             "solver_settings_override",
             [{"system_type": "three_chamber"}],
             indirect=True,
         )
         def test_something(system):
             # system will be the cardiovascular symbolic model here
         """
         override = solver_settings_override if solver_settings_override else {}
         model_type = override.get('system_type', 'nonlinear')

         if model_type == "linear":
             return build_three_state_linear_system(precision)
         if model_type == "nonlinear":
             return build_three_state_nonlinear_system(precision)
         if model_type in ["three_chamber", "threecm"]:
             return build_three_chamber_system(precision)
         if model_type == "stiff":
             return build_three_state_very_stiff_system(precision)
         if model_type == "large":
             return build_large_nonlinear_system(precision)
         if model_type == "constant_deriv":
             return build_three_state_constant_deriv_system(precision)
         if isinstance(model_type, object):
             return model_type

         raise ValueError(f"Unknown model type: {model_type}")
     ```
   - Edge cases: Tests passing a SymbolicODE instance directly as system_type
   - Integration: Works with solver_settings_override pattern

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: Remove solver_settings_override2 Fixture - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 4

**Required Context**:
- File: tests/conftest.py (lines 444-448 for solver_settings_override2)
- File: tests/conftest.py (lines 450-535 for solver_settings fixture)

**Input Validation Required**:
- None

**Tasks**:
1. **Remove solver_settings_override2 fixture definition**
   - File: tests/conftest.py
   - Action: Delete
   - Details:
     Delete lines 444-448:
     ```python
     @pytest.fixture(scope="session")
     def solver_settings_override2(request):
         """Override for solver settings, if provided. A second one, so that we
         can do a class-level and function-level override without conflicts."""
         return request.param if hasattr(request, "param") else {}
     ```
   - Edge cases: All tests using solver_settings_override2 must be updated
   - Integration: solver_settings fixture must be updated

2. **Update solver_settings fixture to remove solver_settings_override2**
   - File: tests/conftest.py
   - Action: Modify
   - Details:
     Change solver_settings signature from:
     ```python
     def solver_settings(solver_settings_override, solver_settings_override2,
         system, precision):
     ```
     To:
     ```python
     def solver_settings(solver_settings_override, system, precision):
     ```
     And update the loop (around lines 519-526) from:
     ```python
     for override in [solver_settings_override, solver_settings_override2]:
         if override:
             # Update defaults with any overrides provided
             for key, value in override.items():
                 if key in float_keys:
                     defaults[key] = precision(value)
                 else:
                     defaults[key] = value
     ```
     To:
     ```python
     if solver_settings_override:
         for key, value in solver_settings_override.items():
             if key in float_keys:
                 defaults[key] = precision(value)
             else:
                 defaults[key] = value
     ```
   - Edge cases: None
   - Integration: Tests must combine overrides into single dict

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 6: Update test_ode_loop.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 5

**Required Context**:
- File: tests/integrators/loops/test_ode_loop.py (entire file)

**Input Validation Required**:
- None

**Tasks**:
1. **Update imports in test_ode_loop.py**
   - File: tests/integrators/loops/test_ode_loop.py
   - Action: Modify
   - Details:
     Update line 13 to import RUN_DEFAULTS:
     ```python
     from tests._utils import assert_integration_outputs, MID_RUN_PARAMS, RUN_DEFAULTS
     ```
   - Edge cases: None
   - Integration: RUN_DEFAULTS will be used in tests

2. **Remove system_override parameterization from test_all_summary_metrics_numerical_check**
   - File: tests/integrators/loops/test_ode_loop.py
   - Action: Modify
   - Details:
     Lines 335-336 currently:
     ```python
     @pytest.mark.parametrize("system_override", ["linear"], ids=[""],
                              indirect=True)
     ```
     Replace with combined solver_settings_override. Merge into solver_settings_override2:
     ```python
     # Remove the system_override parameterization line entirely
     # Update solver_settings_override2 on lines 337-346 to include system_type:
     @pytest.mark.parametrize("solver_settings_override",
          [{
             "system_type": "linear",
             "algorithm": "euler",
             "duration": 0.2,
             "dt": 0.0025,
             "dt_save": 0.01,
             "dt_summarise": 0.1,
         }],
         indirect=True,
         ids = [""]
     )
     # Then add another parameterization for output_types
     ```
     
     Actually simpler approach - combine all into one override:
     The test needs both the algorithm/timing settings AND the output_types.
     Use nested parameterization where the outer sets base config and inner varies output_types.
     
     For now, merge system_type into the first parameterization.
   - Edge cases: Maintaining test functionality while simplifying
   - Integration: Works with new fixture structure

3. **Remove precision_override and system_override from test_float32_small_timestep_accumulation**
   - File: tests/integrators/loops/test_ode_loop.py
   - Action: Modify
   - Details:
     Lines 379-393 currently have:
     ```python
     @pytest.mark.parametrize("precision_override",
                              [np.float32],
                              indirect=True,
                              ids=[""])
     @pytest.mark.parametrize("solver_settings_override",
                              [{...}],
                              indirect=True,
                              ids=[""])
     ```
     Combine into single solver_settings_override with precision:
     ```python
     @pytest.mark.parametrize("solver_settings_override",
                              [{
                                  'precision': np.float32,
                                  'output_types': ['state', 'time'],
                                  'duration': 1e-4,
                                  'dt_save': 2e-5,
                                  't0': 1.0,
                                  'algorithm': "euler",
                                  'dt': 1e-7,
                              }],
                              indirect=True,
                              ids=[""])
     ```
   - Edge cases: None
   - Integration: Works with new fixture structure

4. **Update test_large_t0_with_small_steps to use solver_settings_override for precision**
   - File: tests/integrators/loops/test_ode_loop.py
   - Action: Modify
   - Details:
     Lines 399-418 currently:
     ```python
     @pytest.mark.parametrize("precision_override", [np.float32, np.float64],
                              indirect=True)
     @pytest.mark.parametrize("solver_settings_override",
                              [{...}],
                              indirect=True,
                              ids=[""])
     ```
     Expand to two separate test cases in solver_settings_override:
     ```python
     @pytest.mark.parametrize("solver_settings_override",
                              [
                                  {
                                      'precision': np.float32,
                                      'output_types': ['state', 'time'],
                                      'duration': 1e-3,
                                      'dt_save': 2e-4,
                                      't0': 1e2,
                                      'algorithm': 'euler',
                                      'dt': 1e-6,
                                  },
                                  {
                                      'precision': np.float64,
                                      'output_types': ['state', 'time'],
                                      'duration': 1e-3,
                                      'dt_save': 2e-4,
                                      't0': 1e2,
                                      'algorithm': 'euler',
                                      'dt': 1e-6,
                                  },
                              ],
                              indirect=True,
                              ids=["float32", "float64"])
     ```
   - Edge cases: None
   - Integration: Works with new fixture structure

5. **Update test_adaptive_controller_with_float32**
   - File: tests/integrators/loops/test_ode_loop.py
   - Action: Modify
   - Details:
     Lines 422-437 - combine precision_override into solver_settings_override:
     ```python
     @pytest.mark.parametrize("solver_settings_override",
                              [{
                                  'precision': np.float32,
                                  'duration': 1e-4,
                                  'dt_save': 2e-5,
                                  't0': 1.0,
                                  'algorithm': 'crank_nicolson',
                                  'step_controller': 'PI',
                                  'output_types': ['state', 'time'],
                                  'dt_min': 1e-7,
                                  'dt_max': 1e-6,
                              }],
                              indirect=True)
     ```
   - Edge cases: None
   - Integration: Works with new fixture structure

6. **Update test_save_at_settling_time_boundary**
   - File: tests/integrators/loops/test_ode_loop.py
   - Action: Modify
   - Details:
     Lines 445-460 - combine precision_override into solver_settings_override:
     ```python
     @pytest.mark.parametrize("solver_settings_override",
         [{
             "precision": np.float32,
             "duration": 0.2000,
             "settling_time": 0.1,
             "t0": 1.0,
             "output_types": ["state", "time"],
             "algorithm": "euler",
             "dt": 1e-2,
             "dt_save": 0.1,
         }],
         indirect=True,
     )
     ```
   - Edge cases: None
   - Integration: Works with new fixture structure

7. **Update test_loop and test_initial_observable_seed_matches_reference**
   - File: tests/integrators/loops/test_ode_loop.py  
   - Action: Modify
   - Details:
     For test_loop (lines 245-272), remove solver_settings_override2 and combine:
     ```python
     # Lines 245-255 currently use solver_settings_override2 for MID_RUN_PARAMS
     # and solver_settings_override for LOOP_CASES
     # Combine by merging MID_RUN_PARAMS into each LOOP_CASE
     ```
     Create a helper to merge:
     ```python
     # At module level after imports (around line 18):
     def _merge_with_mid_run(case_settings):
         """Merge LOOP_CASE settings with MID_RUN_PARAMS."""
         merged = MID_RUN_PARAMS.copy()
         merged.update(case_settings)
         return merged
     
     # Update LOOP_CASES to include MID_RUN_PARAMS values
     LOOP_CASES_WITH_DEFAULTS = [
         pytest.param(
             _merge_with_mid_run({"algorithm": "euler", "step_controller": "fixed"}),
             id="euler",
         ),
         # ... etc for all cases
     ]
     ```
     
     Actually, simpler: since MID_RUN_PARAMS is a shared baseline, modify DEFAULT_OVERRIDES usage:
     The test already uses `DEFAULT_OVERRIDES = MID_RUN_PARAMS` on line 18.
     The parameterization uses solver_settings_override2 for this.
     
     Instead, merge at the test call site or have LOOP_CASES include all needed settings.
   - Edge cases: Maintaining exact same test behavior
   - Integration: Works with single override pattern

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 7: Update test_step_algorithms.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 5

**Required Context**:
- File: tests/integrators/algorithms/test_step_algorithms.py (entire file)

**Input Validation Required**:
- None

**Tasks**:
1. **Update imports in test_step_algorithms.py**
   - File: tests/integrators/algorithms/test_step_algorithms.py
   - Action: Modify
   - Details:
     Line 51 - add RUN_DEFAULTS import:
     ```python
     from tests._utils import MID_RUN_PARAMS, RUN_DEFAULTS
     ```
   - Edge cases: None
   - Integration: RUN_DEFAULTS will be available

2. **Remove solver_settings_override2 from test_stage_cache_reuse**
   - File: tests/integrators/algorithms/test_step_algorithms.py
   - Action: Modify
   - Details:
     Lines 1123-1133 currently:
     ```python
     @pytest.mark.parametrize(
         "solver_settings_override2",
         [STEP_OVERRIDES],
         ids=[""],
         indirect=True,
     )
     @pytest.mark.parametrize(
         "solver_settings_override",
         CACHE_REUSE_CASES,
         indirect=True,
     )
     ```
     Merge STEP_OVERRIDES (which is MID_RUN_PARAMS) into each CACHE_REUSE_CASE:
     ```python
     # Update CACHE_REUSE_CASES to include MID_RUN_PARAMS values
     # Or create merged version at module level
     ```
   - Edge cases: Maintaining test behavior
   - Integration: Works with single override

3. **Remove system_override from test_against_euler**
   - File: tests/integrators/algorithms/test_step_algorithms.py
   - Action: Modify
   - Details:
     Lines 1205-1270 - merge system_override and solver_settings_override2:
     ```python
     # Lines 1205-1217 currently:
     @pytest.mark.parametrize(
         "system_override",
         ["constant_deriv"],
         ids=[""],
         indirect=True,
     )
     @pytest.mark.parametrize(
         "solver_settings_override2",
         [STEP_OVERRIDES],
         ids=[""],
         indirect=True,
     )
     @pytest.mark.parametrize(
         "solver_settings_override",
         STEP_CASES,
         indirect=True,
     )
     ```
     Simplify to single parameterization by merging system_type and STEP_OVERRIDES into STEP_CASES:
     Create merged version or update parameterization.
   - Edge cases: Maintaining test behavior
   - Integration: Works with single override

4. **Remove solver_settings_override2 from test_algorithm**
   - File: tests/integrators/algorithms/test_step_algorithms.py
   - Action: Modify
   - Details:
     Lines 1304-1314 currently:
     ```python
     @pytest.mark.parametrize(
             "solver_settings_override2",
             [STEP_OVERRIDES],
             ids=[""],
             indirect=True
     )
     @pytest.mark.parametrize(
         "solver_settings_override",
         STEP_CASES,
         indirect=True,
     )
     ```
     Merge STEP_OVERRIDES into STEP_CASES or create combined version.
   - Edge cases: Maintaining test behavior
   - Integration: Works with single override

5. **Create merged test case lists at module level**
   - File: tests/integrators/algorithms/test_step_algorithms.py
   - Action: Modify
   - Details:
     Add helper function and merged lists after line 459 (after STEP_OVERRIDES):
     ```python
     def _merge_settings(*dicts):
         """Merge multiple setting dicts, later dicts override earlier."""
         result = {}
         for d in dicts:
             result.update(d)
         return result
     
     # Merged versions for tests that used solver_settings_override2
     STEP_CASES_MERGED = [
         pytest.param(
             _merge_settings(STEP_OVERRIDES, case.values[0]),
             id=case.id,
             marks=case.marks if case.marks else (),
         )
         for case in STEP_CASES
     ]
     
     CACHE_REUSE_CASES_MERGED = [
         pytest.param(
             _merge_settings(STEP_OVERRIDES, case.values[0]),
             id=case.id,
             marks=case.marks if case.marks else (),
         )
         for case in CACHE_REUSE_CASES
     ]
     ```
   - Edge cases: pytest.param structure handling
   - Integration: Tests use merged lists

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 8: Update test_controllers.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 5

**Required Context**:
- File: tests/integrators/step_control/test_controllers.py (entire file)

**Input Validation Required**:
- None

**Tasks**:
1. **Combine solver_settings_override and solver_settings_override2 in TestControllers class**
   - File: tests/integrators/step_control/test_controllers.py
   - Action: Modify
   - Details:
     Lines 117-127 currently use solver_settings_override2 at class level:
     ```python
     @pytest.mark.parametrize(
         "solver_settings_override2",
         [
             ({"step_controller": "i", 'atol':1e-3,'rtol':0.0}),
             ...
         ],
         ids=("i", "pi", "pid", "gustafsson"),
         indirect=True
     )
     class TestControllers:
     ```
     
     And test_dt_clamps (lines 132-159) uses solver_settings_override at function level.
     
     Merge by expanding class-level parameterization to include both:
     ```python
     @pytest.mark.parametrize(
         "solver_settings_override",
         [
             {"step_controller": "i", 'atol':1e-3,'rtol':0.0},
             {"step_controller": "pi", 'atol':1e-3,'rtol':0.0},
             {"step_controller": "pid", 'atol':1e-3,'rtol':0.0},
             {"step_controller": "gustafsson", 'atol':1e-3,'rtol':0.0},
         ],
         ids=("i", "pi", "pid", "gustafsson"),
         indirect=True
     )
     class TestControllers:
     ```
     
     For test_dt_clamps, merge the dt_min/dt_max into step_setup since it's function-level:
     Or use fixture combination.
   - Edge cases: Nested parameterization handling
   - Integration: Works with single override

2. **Update test_pi_controller_uses_tableau_order**
   - File: tests/integrators/step_control/test_controllers.py
   - Action: Modify
   - Details:
     Lines 162-174 use both solver_settings_override and solver_settings_override2:
     ```python
     @pytest.mark.parametrize(
         (
             "solver_settings_override",
             "solver_settings_override2",
         ),
         [
             (
                 {"algorithm": "rosenbrock"},
                 {"step_controller": "pi", "atol": 1e-3, "rtol": 0.0},
             ),
         ],
         indirect=True,
     )
     ```
     Combine into single dict:
     ```python
     @pytest.mark.parametrize(
         "solver_settings_override",
         [
             {
                 "algorithm": "rosenbrock",
                 "step_controller": "pi",
                 "atol": 1e-3,
                 "rtol": 0.0,
             },
         ],
         indirect=True,
     )
     ```
   - Edge cases: None
   - Integration: Works with single override

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 9: Verify and Test Changes - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1-8

**Required Context**:
- All modified files

**Input Validation Required**:
- None

**Tasks**:
1. **Run linting on modified files**
   - Files: tests/_utils.py, tests/conftest.py, tests/integrators/loops/test_ode_loop.py, tests/integrators/algorithms/test_step_algorithms.py, tests/integrators/step_control/test_controllers.py
   - Action: Verify
   - Details:
     Run flake8 on modified files to ensure no syntax errors:
     ```bash
     flake8 tests/_utils.py tests/conftest.py tests/integrators/loops/test_ode_loop.py tests/integrators/algorithms/test_step_algorithms.py tests/integrators/step_control/test_controllers.py --count --select=E9,F63,F7,F82 --show-source --statistics
     ```
   - Edge cases: None
   - Integration: Ensures code quality

2. **Run pytest collection to verify fixture resolution**
   - Action: Verify
   - Details:
     Run pytest --collect-only to verify fixtures resolve correctly:
     ```bash
     pytest --collect-only tests/integrators/loops/test_ode_loop.py -q
     pytest --collect-only tests/integrators/algorithms/test_step_algorithms.py -q
     pytest --collect-only tests/integrators/step_control/test_controllers.py -q
     ```
   - Edge cases: Fixture dependency resolution
   - Integration: Verifies refactoring works

3. **Run selected tests to verify functionality**
   - Action: Verify
   - Details:
     Run a subset of tests to verify they still work:
     ```bash
     pytest tests/integrators/loops/test_ode_loop.py::test_getters -v
     pytest tests/integrators/algorithms/test_step_algorithms.py::test_algorithm_factory_resolves_tableau_alias -v
     ```
   - Edge cases: None
   - Integration: Verifies tests still pass

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

### Total Task Groups: 9
### Dependency Chain:
1. Task Group 1 (RUN_DEFAULTS) → Foundation
2. Task Groups 2-5 (Fixture Refactoring) → Sequential, builds on each other
3. Task Groups 6-8 (Test File Updates) → Can run in parallel after Group 5
4. Task Group 9 (Verification) → Final validation

### Parallel Execution Opportunities:
- Task Groups 6, 7, 8 can be executed in parallel after Task Group 5 completes

### Estimated Complexity:
- Task Group 1: Low (simple constant addition)
- Task Groups 2-5: Medium (fixture refactoring requires careful dependency handling)
- Task Groups 6-8: Medium-High (many test parameterization updates)
- Task Group 9: Low (verification only)
