# Test Parameterization Refactoring: Agent Plan

## Architectural Overview

This refactoring restructures test parameterization to reduce CUDA compilation sessions by:
1. Separating runtime values from compile-time settings
2. Consolidating override fixtures (precision_override, system_override into solver_settings)
3. Rationalizing solver_settings_override2 usage (keep for class-level, merge for function-level)
4. Rationalizing system and precision parameterization
5. Eliminating redundant test variations

## Component Changes

### Component 1: RUN_DEFAULTS Constant

**Location**: `tests/_utils.py`

**Purpose**: Centralize runtime parameters that do not affect compilation.

**Expected Behavior**:
- Contains `duration`, `t0`, `warmup` with sensible defaults
- Tests unpack these values directly in test function bodies
- Does NOT participate in fixture parameterization

**Integration Points**:
- Test functions import and unpack `RUN_DEFAULTS`
- `run_device_loop` and `run_reference_loop` accept these as runtime kwargs
- No fixture changes needed for these values

**Structure**:
```python
RUN_DEFAULTS = {
    'duration': 0.1,    # Default simulation duration
    't0': 0.0,          # Default start time
    'warmup': 0.0,      # Default warmup period
}
```

### Component 2: Fixture Consolidation in conftest.py

**Location**: `tests/conftest.py`

**Purpose**: Replace layered override fixtures with simplified structure.

**Changes Required**:

#### 2a. Remove `precision_override` Fixture
- Current: `precision_override` is a separate session-scoped fixture
- Target: Precision is extracted from `solver_settings['precision']` 
- The `precision` fixture becomes a simple accessor to `solver_settings`

#### 2b. Remove `system_override` Fixture
- Current: `system_override` controls system selection independently
- Target: System type is specified via `solver_settings['system_type']` key
- Default to 'nonlinear' when not specified

#### 2c. Rationalize `solver_settings_override2` Pattern
- Current: Two override fixtures (`solver_settings_override`, `solver_settings_override2`)
- Target: **KEEP `solver_settings_override2` for class-level parameterization**
- Pattern: Classes use `override2` (parameterized at class level), methods use `override` (parameterized at method level)
- **Merge function-level dual overrides** into single `solver_settings_override`

**Valid Pattern (KEEP):**
```python
@pytest.mark.parametrize("solver_settings_override2", 
    [{"step_controller": "i"}, {"step_controller": "pi"}],
    indirect=True)
class TestControllers:
    # Class parameterized by controller type
    
    @pytest.mark.parametrize("solver_settings_override",
        [{"dt_min": 0.1, "dt_max": 0.2}], indirect=True)
    def test_dt_clamps(self, ...):
        # Method has its own parameterization
```

**Invalid Pattern (MERGE):**
```python
# Before - function-level dual override (BAD)
@pytest.mark.parametrize("solver_settings_override2", [MID_RUN_PARAMS], indirect=True)
@pytest.mark.parametrize("solver_settings_override", STEP_CASES, indirect=True)
def test_algorithm(...):

# After - merged into single override (GOOD)
@pytest.mark.parametrize("solver_settings_override", STEP_CASES_MERGED, indirect=True)
def test_algorithm(...):
```

#### 2d. Remove Runtime Values from solver_settings
- Current: `duration`, `t0`, `warmup` in `solver_settings` defaults
- Target: These are NOT in `solver_settings` (runtime values)
- Tests pass these directly to execution functions

**Expected Fixture Structure**:
```python
@pytest.fixture(scope="session")
def solver_settings_override(request):
    """Override for solver settings at method/function level."""
    return request.param if hasattr(request, "param") else {}

@pytest.fixture(scope="session")
def solver_settings_override2(request):
    """Override for solver settings at class level.
    
    Used when a class is parameterized by one setting (e.g., controller type)
    and individual methods need their own overrides. Class uses override2,
    methods use override.
    """
    return request.param if hasattr(request, "param") else {}

@pytest.fixture(scope="session")
def solver_settings(solver_settings_override, solver_settings_override2, 
                    precision, request):
    """Consolidated solver settings with defaults."""
    # No duration/t0/warmup here - those are runtime
    defaults = {
        "algorithm": "euler",
        "precision": np.float32,  # Default precision
        "system_type": "nonlinear",  # Default system
        "dt": precision(0.01),
        # ... other compile-time settings
    }
    # Apply class-level overrides first, then method-level
    for override in [solver_settings_override2, solver_settings_override]:
        if override:
            for key, value in override.items():
                defaults[key] = value
    return defaults

@pytest.fixture(scope="session")
def precision(solver_settings_override, solver_settings_override2):
    """Extract precision from overrides, defaulting to float32."""
    for override in [solver_settings_override2, solver_settings_override]:
        if override and 'precision' in override:
            return override['precision']
    return np.float32

@pytest.fixture(scope="session")
def system(solver_settings_override, solver_settings_override2, precision):
    """Build system based on system_type from overrides."""
    system_type = 'nonlinear'  # default
    for override in [solver_settings_override2, solver_settings_override]:
        if override and 'system_type' in override:
            system_type = override['system_type']
            break
    # ... build appropriate system
```

### Component 3: Test File Updates

#### 3a. test_ode_loop.py Changes

**Purpose**: Consolidate loop tests, merge function-level dual overrides.

**Changes**:
- Remove `system_override` parameterization - use `solver_settings_override` with `system_type` key
- Remove `precision_override` parameterization - use `solver_settings_override` with `precision` key
- **Merge function-level `solver_settings_override2` into `solver_settings_override`**
- Unpack `RUN_DEFAULTS` in test bodies for duration/t0/warmup

**Before Pattern (function-level dual override - ELIMINATE):**
```python
@pytest.mark.parametrize("system_override", ["linear"], indirect=True)
@pytest.mark.parametrize("solver_settings_override2", [DEFAULT_OVERRIDES], indirect=True)
@pytest.mark.parametrize("solver_settings_override", LOOP_CASES, indirect=True)
def test_loop(...):
```

**After Pattern (merged single override):**
```python
@pytest.mark.parametrize("solver_settings_override", 
    [merge_dicts(DEFAULT_OVERRIDES, case, {"system_type": "linear"}) 
     for case in LOOP_CASES], 
    indirect=True)
def test_loop(solver_settings, ...):
    duration = RUN_DEFAULTS['duration']
    t0 = RUN_DEFAULTS['t0']
```

#### 3b. test_step_algorithms.py Changes

**Purpose**: Merge function-level dual overrides, eliminate single-step tests.

**Changes**:
- **Merge `solver_settings_override2` (STEP_OVERRIDES/MID_RUN_PARAMS) into STEP_CASES**
- Remove separate single-step fixture from parameterization
- Modify tests to always use dual-step execution via `_execute_step_twice`
- First step of dual execution validates single-step behavior
- Second step validates cache reuse

**Before Pattern (function-level dual override - ELIMINATE):**
```python
@pytest.mark.parametrize("solver_settings_override2", [STEP_OVERRIDES], indirect=True)
@pytest.mark.parametrize("solver_settings_override", STEP_CASES, indirect=True)
def test_algorithm(...):
```

**After Pattern (merged):**
```python
STEP_CASES_MERGED = [merge_dicts(STEP_OVERRIDES, case) for case in STEP_CASES]

@pytest.mark.parametrize("solver_settings_override", STEP_CASES_MERGED, indirect=True)
def test_algorithm(...):
```

#### 3c. test_controllers.py Changes

**Purpose**: KEEP class-level parameterization, merge function-level where dual used incorrectly.

**Changes**:
- **KEEP** `solver_settings_override2` at class level for `TestControllers`
- **KEEP** `solver_settings_override` at method level for specific test settings
- **MERGE** in `test_pi_controller_uses_tableau_order` (currently uses both at function level)
- Remove redundant system parameterization

**Keep This Pattern (class-level - VALID):**
```python
@pytest.mark.parametrize("solver_settings_override2",
    [{"step_controller": "i"}, {"step_controller": "pi"}, ...],
    indirect=True)
class TestControllers:
    @pytest.mark.parametrize("solver_settings_override",
        [{"dt_min": 0.1, "dt_max": 0.2}], indirect=True)
    def test_dt_clamps(self, ...):
```

**Fix This Pattern (function-level dual - MERGE):**
```python
# Before
@pytest.mark.parametrize(("solver_settings_override", "solver_settings_override2"),
    [({"algorithm": "rosenbrock"}, {"step_controller": "pi", "atol": 1e-3})],
    indirect=True)
def test_pi_controller_uses_tableau_order(...):

# After
@pytest.mark.parametrize("solver_settings_override",
    [{"algorithm": "rosenbrock", "step_controller": "pi", "atol": 1e-3}],
    indirect=True)
def test_pi_controller_uses_tableau_order(...):
```

#### 3d. test_controller_equivalence_sequences.py Changes

**Purpose**: KEEP class-level parameterization pattern.

**Changes**:
- **KEEP** current class-level `solver_settings_override2` for controller types
- **KEEP** current method-level `solver_settings_override` for timing/tolerance settings
- Remove `system_override` - use `solver_settings_override` with `system_type` key

#### 3e. test_solver.py Changes

**Purpose**: Reduce unnecessary system variations.

**Changes**:
- Use default system for most tests
- Only parameterize system for tests specifically checking system compatibility
- Combine overrides into single parameterization

### Component 4: Test Helper Updates

#### 4a. run_device_loop Updates

**Location**: `tests/_utils.py`

**Current**: Receives config via `solver_config` dict that includes duration/t0/warmup

**Target**: Continue receiving these via dict parameter (no change needed - the change is in how tests provide them)

#### 4b. _build_cpu_step_controller Updates

**Location**: `tests/conftest.py`

**Purpose**: Ensure CPU controller uses settings from unified solver_settings.

### Component 5: Precision Testing Strategy

**Purpose**: Establish clear precision testing guidelines.

**Strategy**:
1. Default precision is `np.float32` (most common use case)
2. A small set of designated tests verify `np.float64` compatibility
3. Tests that specifically verify precision handling include both
4. Other tests use default precision only

**Designated Float64 Tests**:
- `test_large_t0_with_small_steps` - validates time accumulation
- `test_time_precision_independent_of_state_precision` - validates time handling
- One representative test per algorithm family for float64

### Component 6: System Testing Strategy

**Purpose**: Reduce unnecessary system variations.

**Default System**: 3-state nonlinear (sufficient for most algorithm testing)

**When to Use Other Systems**:
- `three_chamber`: Testing with drivers, larger system validation
- `stiff`: Testing implicit solver convergence
- `linear`: When testing superposition or linearity assumptions
- `large`: Memory scaling tests
- `constant_deriv`: Algorithm order verification (all should match Euler)

## Integration Points

### Fixture Dependencies (After Refactoring)

```
solver_settings_override (method/function level)
solver_settings_override2 (class level)
    │
    └── solver_settings (merges both overrides)
        ├── precision (accessor, checks overrides)
        ├── tolerance (depends on precision)
        ├── system (depends on system_type from overrides)
        ├── driver_settings
        │   └── driver_array
        ├── algorithm_settings
        ├── step_controller_settings
        ├── output_settings
        ├── loop_settings
        └── memory_settings
            └── [object fixtures: single_integrator_run, solver, etc.]
```

### Override Merge Order

When both overrides are present:
1. `solver_settings_override2` applied first (class-level baseline)
2. `solver_settings_override` applied second (method-level specifics)
3. Method-level values take precedence over class-level

### Session Boundaries

**Session 1**: Default configuration (nonlinear, float32, euler)
**Session 2**: Different algorithm requiring recompile
**Session 3**: Different system type requiring recompile
**Session 4**: Different precision requiring recompile

Within a session, runtime parameters (duration, t0, warmup) can vary freely without triggering recompilation.

## Edge Cases

1. **Test-specific tolerances**: Some tests need custom tolerance settings. These should be handled via test-local variables, not fixture parameterization.

2. **Driver-dependent systems**: Systems with drivers need driver_array. The driver configuration should come from solver_settings, not separate overrides.

3. **Precision-sensitive assertions**: Tests checking precision-specific behavior (ULP accuracy) need explicit precision but shouldn't create extra sessions for other tests.

4. **Algorithm tableau variations**: Different tableaus for same algorithm type can share sessions if precision/system match.

5. **Class-level vs function-level parameterization**: 
   - Classes that group tests by a shared attribute (e.g., controller type) should use `solver_settings_override2` at class level
   - Methods within those classes use `solver_settings_override` for test-specific settings
   - Standalone functions should NEVER use both overrides - merge into single `solver_settings_override`

6. **Converting class to standalone functions**: If a class's methods don't benefit from shared parameterization, consider converting to standalone functions with merged overrides instead of using dual overrides.

## Dependencies

### Required Imports in Test Files

```python
from tests._utils import RUN_DEFAULTS, run_device_loop
```

### Configuration Constants

Retain existing parameter sets but clarify their purpose:
- `SHORT_RUN_PARAMS`: Compile-time settings for quick tests
- `MID_RUN_PARAMS`: Compile-time settings for standard tests  
- `LONG_RUN_PARAMS`: Compile-time settings for thorough tests

These should NOT include duration/t0/warmup after refactoring.

## Migration Notes

### Backward Compatibility

Not required - this is internal test infrastructure. All changes can be made atomically.

### Test Discovery

After refactoring, pytest collection should show fewer session boundaries. Verify by running `pytest --collect-only` and counting distinct session scopes.

### Validation

1. Run full test suite - all tests should pass
2. Compare test run time before/after
3. Count compilation messages in test output before/after
4. Verify same test coverage via coverage report
