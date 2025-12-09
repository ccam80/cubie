# Test Fixture Optimization - Agent Plan

## Problem Statement

Test fixtures in `tests/conftest.py` and subdirectories violate the principle that each fixture should use at most one CUDAFactory-based fixture. This causes:

1. **Redundant object construction**: Multiple fixtures build the same CUDAFactory objects
2. **Inefficient caching**: Session-scoped fixtures rebuild unnecessarily
3. **Slow test execution**: Each build triggers CUDA compilation and GPU memory allocation

## Current Architecture Issues

### Fixture Categories

**CUDAFactory-based fixtures** (build compiled CUDA objects):
- `system` - SymbolicODE instance
- `driver_array` - ArrayInterpolator instance
- `step_object` / `step_object_mutable` - Algorithm step instances
- `output_functions` / `output_functions_mutable` - OutputFunctions instances
- `step_controller` / `step_controller_mutable` - Controller instances
- `loop` / `loop_mutable` - IVPLoop instances
- `single_integrator_run` / `single_integrator_run_mutable` - SingleIntegratorRun instances
- `solverkernel` / `solverkernel_mutable` - BatchSolverKernel instances
- `solver` / `solver_mutable` - Solver instances

**Settings fixtures** (return configuration dicts):
- `solver_settings` - Master configuration dict
- `algorithm_settings` - Algorithm configuration
- `step_controller_settings` - Controller configuration
- `output_settings` - Output configuration
- `loop_settings` - Loop configuration
- `memory_settings` - Memory manager configuration
- `driver_settings` - Driver array configuration

### Violations Found

#### 1. algorithm_settings (tests/conftest.py:521-543)

**Current dependencies:**
```python
def algorithm_settings(system, solver_settings, driver_array):
```

**Violations:**
- Requests 2 CUDAFactory fixtures: `system`, `driver_array`
- Accesses: `system.num_drivers`, `system.dxdt_function`, `system.observables_function`, `system.get_solver_helper`
- Accesses: `driver_array.evaluation_function`, `driver_array.driver_del_t`

**Required changes:**
- Extract `num_drivers` from system at a higher level and pass via solver_settings
- DO NOT include `dxdt_function`, `observables_function`, `get_solver_helper` in algorithm_settings
- These functions should be accessed only when building step_object, not when creating settings
- driver_function and driver_del_t should be extracted from driver_array only when building objects

**Corrected approach:**
```python
def algorithm_settings(solver_settings):
    # Filter settings for algorithm
    # Do NOT access system or driver_array
    # n_drivers should come from solver_settings
```

#### 2. step_controller_settings (tests/conftest.py:556-565)

**Current dependencies:**
```python
def step_controller_settings(solver_settings, system, step_object):
```

**Violations:**
- Requests 2 CUDAFactory fixtures: `system`, `step_object`
- Only uses `step_object.order` (algorithm order)
- Unnecessarily builds step_object just to get order value

**Required changes:**
- Remove `system` and `step_object` dependencies
- `algorithm_order` should be derived from algorithm name and stored in solver_settings
- Use algorithm name lookup to determine order without building step_object

**Corrected approach:**
```python
def step_controller_settings(solver_settings):
    # Get algorithm_order from solver_settings
    # Do NOT build step_object
```

#### 3. buffer_settings (tests/conftest.py:859-871)

**Current dependencies:**
```python
def buffer_settings(system, output_functions, step_object):
```

**Violations:**
- Requests 3 CUDAFactory fixtures: `system`, `output_functions`, `step_object`
- Needs: `system.sizes.states/parameters/drivers/observables`
- Needs: `output_functions.state_summaries_buffer_height`, `output_functions.observable_summaries_buffer_height`
- Needs: `step_object.is_adaptive` to determine n_error

**Required changes:**
- Request `single_integrator_run` instead (contains all three)
- Access sizes and settings from single_integrator_run properties
- Alternative: Extract buffer_settings calculation into SingleIntegratorRun and expose as property

**Corrected approach:**
```python
def buffer_settings(single_integrator_run):
    return single_integrator_run.loop_buffer_settings
```

#### 4. loop (tests/conftest.py:704-724)

**Current dependencies:**
```python
def loop(precision, system, step_object, buffer_settings, 
         output_functions, step_controller, solver_settings, 
         driver_array, loop_settings):
```

**Violations:**
- Requests 6 CUDAFactory fixtures: `system`, `step_object`, `output_functions`, `step_controller`, `driver_array`, `buffer_settings`
- Creates massive dependency fan-out

**Required changes:**
- Request `single_integrator_run` instead
- Access loop property: `single_integrator_run.loop`
- This is the correct pattern: request the lowest-level fixture containing all needed objects

**Corrected approach:**
```python
def loop(single_integrator_run):
    return single_integrator_run.loop
```

#### 5. cpu_loop_runner (tests/conftest.py:894-956)

**Current dependencies:**
```python
def cpu_loop_runner(system, cpu_system, precision, solver_settings,
                    step_controller_settings, output_functions,
                    cpu_driver_evaluator, step_object):
```

**Violations:**
- Requests 3 CUDAFactory fixtures: `system`, `output_functions`, `step_object`
- Used for reference CPU execution

**Required changes:**
- Request `single_integrator_run` to get system, output_functions, step_object
- OR keep as-is since it needs specific objects for CPU reference (acceptable exception)

**Decision:** Keep as-is - CPU reference fixtures may need direct access to components

## Refactoring Strategy

### Phase 1: Enrich solver_settings

Add derived configuration to solver_settings that fixtures currently extract from built objects:

**Step 1.1:** Add algorithm_order to solver_settings
- Lookup algorithm name from solver_settings['algorithm']
- Resolve to algorithm order without building step_object
- Store as solver_settings['algorithm_order']

**Step 1.2:** Add system size metadata to solver_settings
- Extract from system fixture: n_states, n_parameters, n_drivers, n_observables
- Store in solver_settings under 'n_states', 'n_parameters', etc.
- This metadata is available early (from system fixture which is session-scoped)

**Step 1.3:** Document solver_settings contract
- List all keys that fixtures depend on
- Ensure overrides work correctly

### Phase 2: Refactor Settings Fixtures

**Step 2.1:** Update algorithm_settings
```python
@pytest.fixture(scope="session")
def algorithm_settings(solver_settings):
    settings, _ = merge_kwargs_into_settings(
        kwargs=solver_settings,
        valid_keys=ALL_ALGORITHM_STEP_PARAMETERS,
    )
    # n_drivers comes from solver_settings
    # dxdt_function, observables_function, get_solver_helper_fn
    # are NOT part of settings - they're passed when building step_object
    return settings
```

**Step 2.2:** Update step_controller_settings
```python
@pytest.fixture(scope="session")
def step_controller_settings(solver_settings):
    settings, _ = merge_kwargs_into_settings(
        kwargs=solver_settings,
        valid_keys=ALL_STEP_CONTROLLER_PARAMETERS,
    )
    # algorithm_order comes from solver_settings
    settings.update(algorithm_order=solver_settings['algorithm_order'])
    return settings
```

### Phase 3: Refactor Object Fixtures

**Step 3.1:** Update buffer_settings

Option A (access from single_integrator_run):
```python
@pytest.fixture(scope="session")
def buffer_settings(single_integrator_run):
    return single_integrator_run.loop_buffer_settings
```

Option B (build from solver_settings if buffer_settings is needed before single_integrator_run):
```python
@pytest.fixture(scope="session")
def buffer_settings(solver_settings):
    # Calculate from settings data
    return LoopBufferSettings(
        n_states=solver_settings['n_states'],
        n_parameters=solver_settings['n_parameters'],
        n_drivers=solver_settings['n_drivers'],
        n_observables=solver_settings['n_observables'],
        state_summary_buffer_height=...,  # from solver_settings
        observable_summary_buffer_height=...,  # from solver_settings
        n_error=...,  # based on is_adaptive flag from algorithm
        n_counters=0,
    )
```

**Step 3.2:** Update loop fixture
```python
@pytest.fixture(scope="session")
def loop(single_integrator_run):
    return single_integrator_run.loop
```

**Step 3.3:** Update loop_mutable fixture
```python
@pytest.fixture(scope="function")
def loop_mutable(single_integrator_run_mutable):
    return single_integrator_run_mutable.loop
```

### Phase 4: Update Specialized conftest Files

Each subdirectory conftest.py must be checked:

**tests/batchsolving/conftest.py:**
- Check cpu_batch_results fixture
- Verify it doesn't violate the rule

**tests/integrators/algorithms/instrumented/conftest.py:**
- Very complex fixture: instrumented_step_results
- Currently depends on: instrumented_step_object, step_inputs, solver_settings, system, precision, dts, num_steps, driver_array
- Requests 2 CUDAFactory fixtures: system, driver_array
- May need special handling for instrumentation

**tests/integrators/matrix_free_solvers/conftest.py:**
- system_setup fixture creates systems on the fly (acceptable)
- No violations detected

**tests/memory/conftest.py:**
- array_request fixtures - no CUDAFactory dependencies
- No violations detected

**tests/odesystems/symbolic/conftest.py:**
- Simple system builders
- No violations detected

## Implementation Details

### Algorithm Order Lookup

Create helper function to resolve algorithm order without building step_object:

```python
from cubie.integrators.algorithms import resolve_alias, resolve_supplied_tableau

def get_algorithm_order(algorithm_name_or_tableau):
    """Get algorithm order without building step object."""
    if isinstance(algorithm_name_or_tableau, str):
        algorithm_type, tableau = resolve_alias(algorithm_name_or_tableau)
    else:
        algorithm_type, tableau = resolve_supplied_tableau(algorithm_name_or_tableau)
    
    # Extract order from tableau or algorithm type
    if tableau is not None and hasattr(tableau, 'order'):
        return tableau.order
    
    # Default orders for algorithms without tableaus
    defaults = {
        'euler': 1,
        'backwards_euler': 1,
        'backwards_euler_pc': 1,
        'crank_nicolson': 2,
    }
    
    algorithm_name = algorithm_name_or_tableau.lower()
    return defaults.get(algorithm_name, 1)
```

### Extracting Functions from System

The algorithm_settings fixture currently extracts functions from system:
- `dxdt_function`
- `observables_function`
- `get_solver_helper`

**Decision:** These should NOT be in algorithm_settings at all. They should be passed directly when constructing the step object, not stored in settings dict.

**Rationale:** These are callable functions specific to a system instance, not configuration. Settings dicts should contain serializable configuration, not runtime objects.

### Driver Array Functions

Similar issue with driver_array:
- `evaluation_function`
- `driver_del_t`

**Decision:** Extract these when building objects that need them, not in settings fixtures.

### Buffer Settings Calculation

LoopBufferSettings depends on:
- System sizes (n_states, n_parameters, n_drivers, n_observables)
- Output buffer heights (from output_functions)
- Error array size (based on is_adaptive)

**Two options:**

1. **Make buffer_settings a property of single_integrator_run**
   - Add `loop_buffer_settings` property to SingleIntegratorRun
   - buffer_settings fixture requests single_integrator_run
   - Simple and clean

2. **Calculate from solver_settings**
   - Requires all buffer sizing info in solver_settings
   - More complex but allows buffer_settings to be independent

**Recommendation:** Option 1 (property of single_integrator_run)

## Edge Cases and Considerations

### CPU Reference Fixtures

Fixtures like cpu_loop_runner, cpu_system, cpu_driver_evaluator may need direct access to CUDAFactory objects for reference execution. These are acceptable exceptions since they're creating CPU-side equivalents.

### Instrumented Fixtures

tests/integrators/algorithms/instrumented/conftest.py has complex instrumentation fixtures that may need special handling. The instrumented_step_results fixture depends on system and driver_array for creating test kernels.

**Decision:** This may be an acceptable exception due to instrumentation needs, but should be reviewed.

### Mutable vs Session-Scoped

Some fixtures have both session-scoped and function-scoped (mutable) versions:
- output_functions / output_functions_mutable
- step_object / step_object_mutable
- step_controller / step_controller_mutable
- loop / loop_mutable
- single_integrator_run / single_integrator_run_mutable

After refactoring, mutable versions should follow the same pattern as session-scoped versions.

### Backwards Compatibility

This refactoring changes the internal structure of fixtures but should NOT change test behavior. All tests should pass after refactoring.

## Validation Strategy

1. **Fixture dependency audit:**
   - Run `pytest --collect-only` and verify fixture initialization order
   - Check that each fixture requests at most one CUDAFactory fixture

2. **Performance measurement:**
   - Before: `pytest --durations=20`
   - After: `pytest --durations=20`
   - Compare total time and slowest tests

3. **Test correctness:**
   - All existing tests must pass
   - No changes to test logic or assertions
   - Only fixture dependencies change

## Files to Modify

1. `tests/conftest.py` (primary changes)
   - solver_settings fixture
   - algorithm_settings fixture
   - step_controller_settings fixture
   - buffer_settings fixture
   - buffer_settings_mutable fixture
   - loop fixture
   - loop_mutable fixture

2. `tests/batchsolving/conftest.py` (review only)
   - Verify cpu_batch_results doesn't violate rules

3. `tests/integrators/algorithms/instrumented/conftest.py` (review and possibly modify)
   - instrumented_step_results fixture
   - May need special exception for instrumentation

4. Helper function (new):
   - Add `get_algorithm_order()` helper to tests/_utils.py or tests/conftest.py

## Success Criteria

1. Each fixture requests at most one CUDAFactory-based fixture
2. Settings fixtures only access solver_settings (no built objects)
3. All tests pass unchanged
4. Test execution time reduced measurably
5. Fixture dependency graph is simplified
