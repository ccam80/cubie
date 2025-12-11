# Test Parameterization Restructure - Agent Plan

## Overview

This plan restructures test suite parameterization to reduce CUDA kernel compilation overhead from ~80 unique parameter combinations to ~13, achieving 60-75% reduction in compilation time while preserving all test coverage and numerical validation.

## Component Descriptions

### Component 1: Standard Parameter Set Definitions (tests/conftest.py)

**Purpose:** Define three standard parameter dictionaries that capture common test scenarios

**Expected Behavior:**
- Three module-level constant dictionaries: SHORT_RUN_PARAMS, MID_RUN_PARAMS, LONG_RUN_PARAMS
- Each dictionary contains subset of solver_settings keys
- Values selected to serve distinct validation purposes
- Can be imported and referenced in test files

**Location:** `tests/conftest.py` at module level (after imports, before fixtures)

**Structure:**
```python
SHORT_RUN_PARAMS = {
    'duration': 0.05,
    'dt_save': 0.05,
    'dt_summarise': 0.05,
    'output_types': ['state', 'time'],
}

MID_RUN_PARAMS = {
    'dt': 0.001,
    'dt_save': 0.02,
    'dt_summarise': 0.1,
    'dt_max': 0.5,
    'output_types': ['state', 'time', 'mean'],
}

LONG_RUN_PARAMS = {
    'duration': 0.3,
    'dt': 0.0005,
    'dt_save': 0.05,
    'dt_summarise': 0.15,
    'output_types': ['state', 'observables', 'time', 'mean', 'rms'],
}
```

**Key Properties:**
- SHORT_RUN: Minimal duration (0.05s), single save point, minimal outputs
- MID_RUN: Default duration (0.2s), frequent saves (10 points), includes summary metric
- LONG_RUN: Extended duration (0.3s), moderate saves, full output types

**Design Rationale:**
- Only override parameters that differ from conftest.py defaults
- All other parameters use existing fixture defaults (dt=0.01, atol=1e-6, etc.)
- Values chosen based on analysis in test_parameterization_report.md

### Component 2: Test File Parameter Updates

**Purpose:** Replace unique parameter dictionaries with references to standard sets

#### File 2a: test_solveresult.py

**Current State:** 
- 13 tests use 3 different parameter variations
- All variations are "short run" category (0.05-0.06s duration)
- Differences: dt_save values (0.02 vs 0.05), output_types variations

**Changes Required:**
- Replace all `solver_settings_override` parametrization with `SHORT_RUN_PARAMS`
- Current parameterizations at lines with decorator patterns like:
  ```python
  @pytest.mark.parametrize('solver_settings_override', [
      {'duration': 0.05, 'dt_save': 0.02, ...}
  ], indirect=True)
  ```
- Replace with:
  ```python
  @pytest.mark.parametrize('solver_settings_override', [
      SHORT_RUN_PARAMS
  ], indirect=True)
  ```

**Expected Behavior After Change:**
- All SolveResult tests run with consistent SHORT_RUN parameters
- Tests validate result object structure, data access, formatting
- No numerical accuracy assertions affected (tests check structure, not values)

#### File 2b: test_step_algorithms.py

**Current State:**
- Module-level constant STEP_OVERRIDES used by ~30 algorithm test cases
- Current value: `{'dt': 0.001953125, 'dt_min': 1e-6, 'atol': 1e-6, 'rtol': 1e-6, 'newton_tolerance': 1e-6, 'krylov_tolerance': 1e-6, 'output_types': ['state']}`

**Changes Required:**
- Locate STEP_OVERRIDES constant definition (likely near top of file)
- Replace entire dictionary with reference to MID_RUN_PARAMS
- Change from:
  ```python
  STEP_OVERRIDES = {'dt': 0.001953125, ...}
  ```
- To:
  ```python
  from tests.conftest import MID_RUN_PARAMS
  STEP_OVERRIDES = MID_RUN_PARAMS
  ```

**Expected Behavior After Change:**
- Algorithm step tests run with MID_RUN parameters (dt=0.001, duration=0.2s default)
- Multiple timesteps allow numerical error accumulation testing
- CPU/GPU comparison tests maintain validity (dt change is minor)

**Integration Points:**
- STEP_OVERRIDES passed to `solver_settings_override` or `solver_settings_override2` fixtures
- Fixtures merge overrides into default solver_settings
- Changes propagate to algorithm_settings, step_controller_settings

#### File 2c: test_ode_loop.py

**Current State:**
- Module-level constant DEFAULT_OVERRIDES at lines 17-28
- LOOP_CASES list uses DEFAULT_OVERRIDES for ~30 algorithm/controller combinations
- Additional edge case overrides: loop_float32_small, loop_large_t0, loop_adaptive

**Changes Required:**
1. Replace DEFAULT_OVERRIDES dictionary with MID_RUN_PARAMS reference
2. Keep edge case parameter sets unchanged:
   - loop_float32_small: Tests float32 accumulation with tiny duration (1e-4s)
   - loop_large_t0: Tests large t0 stability (t0=1e6, duration=0.001s)  
   - loop_adaptive: Tests adaptive controller with short duration

**Specific Change:**
```python
# Replace lines 17-28
from tests.conftest import MID_RUN_PARAMS
DEFAULT_OVERRIDES = MID_RUN_PARAMS

# Keep existing edge case overrides unchanged
```

**Expected Behavior After Change:**
- Main loop tests use MID_RUN parameters (frequent saves, moderate duration)
- Edge cases remain separately tested with specific parameter values
- Numerical validation against CPU reference loops maintained

**Integration Points:**
- DEFAULT_OVERRIDES merged into solver_settings via fixture chain
- Loop fixture (via single_integrator_run) receives merged settings
- Edge cases use separate pytest.param instances with custom overrides

#### File 2d: test_solver.py

**Current State:**
- Multiple test functions with inline parameter dictionaries
- All use short durations (0.05s) for API/structural testing
- Focus: grid building, solver initialization, method calls

**Changes Required:**
- Locate parameterization decorators with solver_settings_override
- Replace inline dicts with SHORT_RUN_PARAMS reference
- Tests likely at: test_solver_properties, test_manual_proportion, test_variable_indices_methods

**Expected Behavior After Change:**
- Solver construction and method tests use SHORT_RUN parameters
- No numerical accuracy assertions (tests verify API behavior)
- Grid construction tests remain valid (parameter values don't affect grid logic)

#### File 2e: test_output_sizes.py

**Current State:**
- Tests calculate output buffer sizes based on configuration
- Two parameter variations: default (0.0s) and custom settings
- Purely structural tests (no kernel execution)

**Changes Required:**
- Replace default parameter override with SHORT_RUN_PARAMS
- Preserve any edge cases that test specific size calculations

**Expected Behavior After Change:**
- Size calculation tests use SHORT_RUN parameters
- Tests verify math of buffer size computations
- Duration value doesn't affect correctness (size = function of samples, not time)

#### File 2f: test_controllers.py

**Current State:**
- Tests adaptive controller device functions
- Parameterized by controller type (I, PI, PID, Gustafsson)
- Edge cases test dt clamping and gain clamping with specific min/max values

**Changes Required:**
1. Update base controller tests to use MID_RUN_PARAMS
2. Preserve edge case parameterizations:
   - test_dt_clamps: Uses dt_min=0.1, dt_max=0.2 to verify clamping
   - test_gain_clamps: Uses min_gain and max_gain specific values

**Specific Changes:**
- Base test class (TestControllers) uses MID_RUN_PARAMS
- test_dt_clamps keeps custom solver_settings_override with dt_min/dt_max
- test_gain_clamps keeps custom settings with gain limits

**Expected Behavior After Change:**
- Controller proposal tests use MID_RUN parameters
- Edge case tests continue verifying boundary behaviors
- Device/CPU step controller agreement maintained

**Integration Points:**
- solver_settings_override2 fixture allows class-level and function-level overrides
- step_controller_settings derived from merged solver_settings
- step_controller fixture builds controller from settings

#### File 2g: test_SolverKernel.py

**Current State:**
- Full integration test with two test cases
- Already uses long duration (0.3s) and comprehensive outputs
- Tests CPU/GPU agreement for batch solver kernel

**Changes Required:**
- Verify existing parameterization matches or exceeds LONG_RUN_PARAMS
- Update to reference LONG_RUN_PARAMS for consistency
- Parameterization at lines 15-45 (smoke_test and fire_test)

**Expected Behavior After Change:**
- Kernel integration tests use LONG_RUN parameters
- Full numerical validation over extended integration
- Tests remain most comprehensive and expensive in suite

**Integration Points:**
- solverkernel fixture builds BatchSolverKernel with all settings
- cpu_batch_results fixture runs CPU reference integration
- assert_integration_outputs compares device vs CPU results

### Component 3: Edge Case Parameter Sets

**Purpose:** Preserve validation of specific failure modes and boundary conditions

**Edge Cases to Maintain:**

1. **loop_float32_small** (test_ode_loop.py)
   - Purpose: Detect float32 accumulation errors in short integrations
   - Parameters: algorithm='euler', duration=1e-4, dt=1e-7, dt_save=2e-5
   - Why separate: Extremely short duration and timestep specifically trigger float32 issues

2. **loop_large_t0** (test_ode_loop.py)
   - Purpose: Verify numerical stability when t0 is large
   - Parameters: algorithm='euler', t0=1e6, duration=0.001, dt=1e-6, dt_save=2e-4
   - Why separate: Large t0 value critical to triggering precision issues

3. **loop_adaptive** (test_ode_loop.py)
   - Purpose: Test adaptive controller with very short integration
   - Parameters: algorithm='crank_nicolson', step_controller='PI', duration=1e-4, dt_min=1e-7, dt_max=1e-6
   - Why separate: Short duration with adaptive control tests specific failure mode

4. **controller_dt_clamps** (test_controllers.py)
   - Purpose: Verify dt clamping at boundaries
   - Parameters: dt_min=0.1, dt_max=0.2 (with extreme error values)
   - Why separate: Specific dt bounds required to test clamping logic

5. **controller_gain_clamps** (test_controllers.py)
   - Purpose: Verify gain clamping in adaptive controllers
   - Parameters: min_gain, max_gain with specific test values
   - Why separate: Specific gain bounds required to test limiting behavior

**Expected Behavior:**
- Edge cases remain as separate pytest.param entries
- Each retains unique parameter dictionary with critical values
- Standard tests do NOT attempt to cover these scenarios

## Architectural Changes Required

### Changes to tests/conftest.py

**Add at module level (after imports, before fixtures):**
```python
# Standard parameter sets for test categories
SHORT_RUN_PARAMS = {...}
MID_RUN_PARAMS = {...}
LONG_RUN_PARAMS = {...}
```

**No changes to:**
- Existing fixture definitions
- Fixture dependency chains
- Override fixture patterns
- Default parameter values in solver_settings fixture

### Changes to Test Files

**Pattern for all test file updates:**
1. Add import at top: `from tests.conftest import SHORT_RUN_PARAMS` (or MID/LONG)
2. Replace parameter dictionary literals with constant reference
3. Keep pytest.mark.parametrize structure unchanged
4. Keep indirect=True flag in parameterization

**Files requiring updates:**
- tests/batchsolving/test_solveresult.py
- tests/integrators/algorithms/test_step_algorithms.py
- tests/integrators/loops/test_ode_loop.py
- tests/batchsolving/test_solver.py
- tests/outputhandling/test_output_sizes.py
- tests/integrators/step_control/test_controllers.py
- tests/batchsolving/test_SolverKernel.py

**No changes to:**
- Test function signatures
- Test assertion logic
- Fixture requests in test signatures
- pytest marker usage (except parameterize values)

## Data Structures

### Standard Parameter Set Schema

Each standard parameter set is a Python dictionary with keys that are valid solver_settings override keys:

**Required Keys for SHORT_RUN:**
- duration: float (simulation time span)
- dt_save: float (save interval)
- dt_summarise: float (summary interval)
- output_types: list[str] (output categories to enable)

**Required Keys for MID_RUN:**
- dt: float (integration timestep)
- dt_save: float (save interval)
- dt_summarise: float (summary interval)
- dt_max: float (maximum allowed timestep for adaptive)
- output_types: list[str]

**Required Keys for LONG_RUN:**
- duration: float
- dt: float
- dt_save: float
- dt_summarise: float
- output_types: list[str]

**Optional Keys (use conftest defaults if not specified):**
- algorithm: str
- dt_min: float
- atol: float
- rtol: float
- newton_tolerance: float
- krylov_tolerance: float
- step_controller: str
- saved_state_indices: list[int]
- saved_observable_indices: list[int]
- summarised_state_indices: list[int]
- summarised_observable_indices: list[int]

## Dependencies and Imports

### New Import Requirements

**tests/conftest.py:**
- No new imports required (uses existing numpy, pytest, cubie imports)

**Test files requiring imports:**
```python
from tests.conftest import SHORT_RUN_PARAMS  # or MID_RUN_PARAMS, LONG_RUN_PARAMS
```

**Import location:** Add after existing pytest and cubie imports, before test class/function definitions

### Dependency Chain

```
Standard params defined in conftest.py
    ↓
Imported in test file
    ↓
Used in pytest.mark.parametrize decorator
    ↓
Passed to solver_settings_override fixture
    ↓
Merged into solver_settings fixture
    ↓
Distributed to component fixtures (algorithm_settings, loop_settings, etc.)
    ↓
Used to build test objects (SingleIntegratorRun, Solver, etc.)
    ↓
Objects cached by session scope (compilation happens once per unique params)
```

## Edge Cases to Consider

### Edge Case 1: Floating-Point Precision in Parameter Values

**Issue:** Parameter values must be castable to test precision (float32 or float64)

**Handling:** 
- Define parameter values as Python floats in standard sets
- solver_settings fixture handles precision casting via `precision()` calls
- No special handling needed in standard param definitions

### Edge Case 2: Parameter Conflicts Between Overrides

**Issue:** solver_settings_override and solver_settings_override2 both exist for layered overrides

**Handling:**
- Understand fixture precedence: override2 applied after override
- Test classes may use override for class-level, override2 for method-level
- Standard params typically used in override, not override2
- Edge cases can use override2 for method-specific variations

### Edge Case 3: Algorithm-Specific Parameter Requirements

**Issue:** Some algorithms require specific parameters (e.g., Rosenbrock needs tableau)

**Handling:**
- Standard params do not specify algorithm (use fixture default='euler')
- Tests override algorithm in separate dict key: `{'algorithm': 'rosenbrock', **MID_RUN_PARAMS}`
- algorithm_settings fixture and get_algorithm_step handle algorithm-specific requirements

### Edge Case 4: Output Type Dependencies

**Issue:** Some output types depend on others (e.g., 'peaks[2]' requires state or observables)

**Handling:**
- Standard params always include base output types: 'state', 'time'
- LONG_RUN includes 'observables' for comprehensive validation
- Tests requiring specific metrics can override: `{**LONG_RUN_PARAMS, 'output_types': ['state', 'peaks[2]']}`

### Edge Case 5: System-Specific Parameter Needs

**Issue:** Some test systems (three_chamber, stiff) may need tighter tolerances

**Handling:**
- Standard params use conftest defaults (atol=1e-6, rtol=1e-6)
- Tests can override system and tolerances separately:
  ```python
  @pytest.mark.parametrize('system_override', ['stiff'], indirect=True)
  @pytest.mark.parametrize('solver_settings_override', [
      {**MID_RUN_PARAMS, 'atol': 1e-8, 'rtol': 1e-8}
  ], indirect=True)
  ```

### Edge Case 6: Driver-Dependent Tests

**Issue:** Tests with drivers need driver_settings configured

**Handling:**
- driver_settings fixture derives from solver_settings automatically
- dt_save affects driver sample spacing
- Standard params' dt_save values work for default driver configurations
- No special handling needed

### Edge Case 7: Memory-Limited Environments

**Issue:** LONG_RUN params may stress memory in CI

**Handling:**
- LONG_RUN duration (0.3s) still modest enough for CI
- Memory manager chunks large batches automatically
- No changes to chunking logic required
- Tests monitor memory only when marked with specific markers

## Expected Interactions Between Components

### Interaction 1: conftest.py ↔ Test Files

**Flow:**
1. Test file imports standard params from conftest
2. Uses in pytest.mark.parametrize decorator
3. Pytest passes dict to fixture as indirect parameter
4. Fixture receives dict, merges with defaults

**Data exchanged:** Python dictionary with string keys, heterogeneous values (float, int, str, list)

### Interaction 2: Parameterization ↔ Session-Scoped Fixtures

**Flow:**
1. pytest collects all test items with parameterizations
2. Builds unique combinations of indirect parameters
3. Creates session-scoped fixtures for each unique combination
4. Caches fixture results (including compiled CUDA kernels)
5. Reuses cached fixtures for tests with identical params

**Caching behavior:**
- Before: ~80 unique combinations → 80 compilations
- After: ~13 unique combinations → 13 compilations
- Compilation savings: ~67 avoided compilations

### Interaction 3: Standard Params ↔ Edge Case Params

**Flow:**
1. Edge case tests use separate pytest.param entries
2. Each edge case has unique parameter dict
3. Edge case params merged into solver_settings independently
4. Session fixtures cache edge case compilations separately

**Isolation:** 
- Edge cases do not interfere with standard params
- Standard tests do not attempt edge case scenarios
- Clear separation of concerns

### Interaction 4: Test Assertions ↔ Parameter Changes

**Flow:**
1. Test runs with new parameters
2. Device loop executes with new dt, duration, save intervals
3. Outputs have different shapes (different number of save points)
4. Assertions validate shapes dynamically (using output_functions fixture)

**Assertion changes needed:**
- None - tests use fixtures to derive expected shapes
- output_functions.num_state_samples calculates expected from dt_save
- Assertions check shapes match expectations, not hardcoded values

## Validation Strategy

### Validation Phase 1: Compilation Time Measurement

**Approach:**
1. Run full test suite with `--durations=0` flag before changes
2. Record total session time and fixture setup times
3. Apply changes
4. Run full test suite again with same flags
5. Compare session and setup times

**Success Criteria:**
- Total session time reduced by 30-50% (includes compilation + test execution)
- Fixture setup time (visible in durations report) reduced by 60-75%

### Validation Phase 2: Test Passage

**Approach:**
1. Run full test suite: `pytest`
2. Run CUDA-free subset: `pytest -m "not nocudasim and not cupy"`
3. Check for any failures or errors

**Success Criteria:**
- All tests pass that passed before
- No new failures introduced
- No test skips added (except pre-existing)

### Validation Phase 3: Numerical Result Comparison

**Approach:**
1. Select representative numerical tests (test_ode_loop, test_SolverKernel)
2. Capture outputs before changes (save arrays to file)
3. Run same tests after changes
4. Compare outputs element-wise

**Success Criteria:**
- Results within existing tolerance bounds
- Relative differences < 1% for most values
- Some variation acceptable due to different dt/duration, but patterns match

### Validation Phase 4: Edge Case Coverage

**Approach:**
1. List all edge cases before changes
2. Verify each edge case still tested after changes
3. Check edge case parameters unchanged

**Success Criteria:**
- float32 accumulation test still uses tiny duration/dt
- Large t0 test still uses large t0 value
- Adaptive controller test still uses adaptive parameters
- Clamping tests still use specific bounds

## Implementation Sequence

### Step 1: Define Standard Parameters in conftest.py

**Action:** Add three constant dictionaries after imports

**Verification:** Import constants in Python REPL, print values

### Step 2: Update test_solveresult.py

**Action:** Replace parameterization with SHORT_RUN_PARAMS

**Verification:** Run `pytest tests/batchsolving/test_solveresult.py -v`

### Step 3: Update test_step_algorithms.py

**Action:** Replace STEP_OVERRIDES with MID_RUN_PARAMS

**Verification:** Run `pytest tests/integrators/algorithms/test_step_algorithms.py -v -k "test_euler"`

### Step 4: Update test_ode_loop.py

**Action:** Replace DEFAULT_OVERRIDES with MID_RUN_PARAMS, preserve edge cases

**Verification:** Run `pytest tests/integrators/loops/test_ode_loop.py -v -k "euler"`

### Step 5: Update test_solver.py

**Action:** Replace inline dicts with SHORT_RUN_PARAMS

**Verification:** Run `pytest tests/batchsolving/test_solver.py -v`

### Step 6: Update test_output_sizes.py

**Action:** Replace default params with SHORT_RUN_PARAMS

**Verification:** Run `pytest tests/outputhandling/test_output_sizes.py -v`

### Step 7: Update test_controllers.py

**Action:** Use MID_RUN_PARAMS for base tests, preserve edge cases

**Verification:** Run `pytest tests/integrators/step_control/test_controllers.py -v`

### Step 8: Update test_SolverKernel.py

**Action:** Verify/update to LONG_RUN_PARAMS

**Verification:** Run `pytest tests/batchsolving/test_SolverKernel.py -v`

### Step 9: Run Full Test Suite

**Action:** `pytest -v --durations=20`

**Verification:** All tests pass, note compilation time reduction

### Step 10: Document Changes

**Action:** Update test suite documentation if it exists

**Verification:** Review CHANGELOG.md or test README

## Notes for Implementation Team

### Performance Expectations

- First test run after changes: Full recompilation (~13 parameter sets)
- Subsequent runs: No compilation (unless fixtures change)
- CI environments: Fresh compilation each run, but 60-75% faster
- Development: Dramatic improvement when iterating on tests

### Debugging Parameterization Issues

**If tests fail after changes:**

1. Check parameter dictionary structure:
   - Print received parameters in solver_settings fixture
   - Verify keys match expected override keys
   - Check for typos in dictionary keys

2. Check merge behavior:
   - Verify override values actually override defaults
   - Check for conflicts between override and override2
   - Print merged solver_settings in fixture

3. Check fixture caching:
   - Clear pytest cache: `pytest --cache-clear`
   - Check for stale compiled kernels
   - Verify session scope working correctly

### Common Pitfalls to Avoid

1. **Don't modify test assertions:** Only change parameterization, not test logic
2. **Don't remove edge cases:** They test critical failure modes
3. **Don't change conftest defaults:** Standard params override defaults, not replace them
4. **Don't hardcode new values in tests:** Use fixture-derived expectations
5. **Don't skip validation phases:** Each phase catches different issues

### Testing Strategy

**Incremental testing:**
- Update one file at a time
- Run that file's tests before moving to next
- Catch issues early before compound effects

**Regression prevention:**
- Keep git history clean with small commits
- Easy to bisect if issues arise
- Can revert individual file changes if needed
