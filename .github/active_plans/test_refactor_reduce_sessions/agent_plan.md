# Test Parameterization Refactoring: Agent Plan

## Architectural Overview

This refactoring restructures test parameterization to reduce CUDA compilation sessions by:
1. Separating runtime values from compile-time settings
2. Consolidating override fixtures
3. Rationalizing system and precision parameterization
4. Eliminating redundant test variations

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

#### 2c. Simplify `solver_settings_override` Pattern
- Current: Two override fixtures (`solver_settings_override`, `solver_settings_override2`)
- Target: Single `solver_settings_override` fixture
- Remove `solver_settings_override2` entirely
- Tests needing multiple overrides combine them in a single dict

#### 2d. Remove Runtime Values from solver_settings
- Current: `duration`, `t0`, `warmup` in `solver_settings` defaults
- Target: These are NOT in `solver_settings` (runtime values)
- Tests pass these directly to execution functions

**Expected Fixture Structure**:
```python
@pytest.fixture(scope="session")
def solver_settings_override(request):
    """Single override point for all solver settings."""
    return request.param if hasattr(request, "param") else {}

@pytest.fixture(scope="session")
def solver_settings(solver_settings_override, precision, request):
    """Consolidated solver settings with defaults."""
    # No duration/t0/warmup here - those are runtime
    defaults = {
        "algorithm": "euler",
        "precision": np.float32,  # Default precision
        "system_type": "nonlinear",  # Default system
        "dt": precision(0.01),
        # ... other compile-time settings
    }
    # Apply overrides
    for key, value in solver_settings_override.items():
        defaults[key] = value
    return defaults

@pytest.fixture(scope="session")
def precision(solver_settings):
    """Extract precision from solver_settings."""
    return solver_settings['precision']

@pytest.fixture(scope="session")
def system(solver_settings, precision):
    """Build system based on solver_settings['system_type']."""
    system_type = solver_settings.get('system_type', 'nonlinear')
    # ... build appropriate system
```

### Component 3: Test File Updates

#### 3a. test_ode_loop.py Changes

**Purpose**: Consolidate loop tests, remove redundant parameterization.

**Changes**:
- Remove `system_override` parameterization where not needed
- Remove `precision_override` parameterization where not needed
- Use single `solver_settings_override` combining all settings
- Unpack `RUN_DEFAULTS` in test bodies for duration/t0/warmup

**Before Pattern**:
```python
@pytest.mark.parametrize("precision_override", [np.float32], indirect=True)
@pytest.mark.parametrize("system_override", ["linear"], indirect=True)
@pytest.mark.parametrize("solver_settings_override2", [{...}], indirect=True)
@pytest.mark.parametrize("solver_settings_override", [{...}], indirect=True)
def test_something(...):
```

**After Pattern**:
```python
@pytest.mark.parametrize("solver_settings_override", 
    [{"precision": np.float32, "system_type": "linear", ...}], 
    indirect=True)
def test_something(solver_settings, ...):
    duration = RUN_DEFAULTS['duration']
    t0 = RUN_DEFAULTS['t0']
```

#### 3b. test_step_algorithms.py Changes

**Purpose**: Eliminate single-step tests in favor of dual-step tests.

**Changes**:
- Remove separate single-step fixture (`device_step_results`) from test parameterization
- Modify tests to always use dual-step execution via `_execute_step_twice`
- First step of dual execution validates single-step behavior
- Second step validates cache reuse

**Test Structure**:
```python
def test_algorithm(...):
    # Execute two steps
    result = _execute_step_twice(...)
    
    # First step validates single-step behavior
    assert result.first_state matches expected
    
    # Second step validates cache reuse
    assert result.second_state matches expected
```

#### 3c. test_controllers.py Changes

**Purpose**: Simplify controller test parameterization.

**Changes**:
- Combine `solver_settings_override` and `solver_settings_override2` into single override
- Use consolidated `solver_settings` for controller settings
- Remove redundant system parameterization

#### 3d. test_solver.py Changes

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
solver_settings_override
    └── solver_settings
        ├── precision (accessor)
        ├── tolerance (depends on precision)
        ├── system (depends on solver_settings.system_type)
        ├── driver_settings
        │   └── driver_array
        ├── algorithm_settings
        ├── step_controller_settings
        ├── output_settings
        ├── loop_settings
        └── memory_settings
            └── [object fixtures: single_integrator_run, solver, etc.]
```

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
