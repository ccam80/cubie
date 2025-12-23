# Agent Plan: Test Fixture Refactor for Buffer Allocation

## Problem Statement

The buffer allocation refactor centralized memory management in `buffer_registry.py`, but test fixtures weren't updated to work with the new architecture. This causes:

1. **Zero-sized array errors**: Implicit algorithm tests fail with "index X out of bounds for axis 0 with size 0"
2. **Non-zero status codes**: Implicit algorithms (backwards_euler, rosenbrock) return failure status
3. **Summary metric mismatches**: State summaries don't match CPU reference
4. **NaN processing issues**: SolveResult tests see unexpected NaN values

## Component Analysis

### 1. Test Kernel Buffer Allocation (`tests/_utils.py`)

The `run_device_loop` function creates test kernels that execute `SingleIntegratorRun.device_function`. Current issues:

- **Current**: Retrieves `shared_bytes` from `singleintegratorrun.shared_memory_bytes`
- **Issue**: For implicit algorithms, the algorithm's Newton solver workspace buffers must be included
- **Expected Behavior**: `shared_memory_bytes` should return the total of all registered buffers (loop buffers + algorithm buffers)

**Investigation Required**:
- Check if `SingleIntegratorRun.shared_memory_bytes` correctly aggregates algorithm buffer requirements
- Verify buffer registration happens before size queries

### 2. Step Algorithm Test Kernel (`tests/integrators/algorithms/test_step_algorithms.py`)

The `device_step_results` and `_execute_step_twice` fixtures create kernels to test individual step functions.

**Buffer Access Pattern**:
```python
shared_elems = step_object.shared_memory_required
shared_bytes = precision(0).itemsize * shared_elems
```

**Issue**: `step_object.shared_memory_required` may not include all buffers if:
- Buffer registration uses different context than query
- Algorithm's solver helper buffers aren't registered

**Investigation Required**:
- Verify `step_object.shared_memory_required` returns correct size for implicit algorithms
- Check if BackwardsEulerStep, RosenbrockWStep register their Newton solver workspace

### 3. CPU Reference Configuration Parity (`tests/conftest.py`)

CPU reference implementations must use identical settings to GPU implementations.

**Key Settings to Verify**:
- `newton_tolerance` → affects convergence
- `max_newton_iters` → affects whether iterations succeed
- `krylov_tolerance` → affects linear solver convergence
- `max_linear_iters` → affects linear solver success
- `preconditioner_order` → affects convergence rate

**Investigation Required**:
- Verify `cpu_step_results` fixture passes same settings as GPU
- Verify `_execute_cpu_step_twice` uses same `solver_settings` values

### 4. Summary Metrics Calculation (`tests/integrators/loops/test_ode_loop.py`)

The `test_all_summary_metrics_numerical_check` test compares GPU summary outputs to CPU reference.

**Potential Issues**:
- Buffer sizing for summary output arrays
- Timing of summary calculations (settling_time boundary handling)
- Precision enforcement in CPU reference

**Investigation Required**:
- Verify `state_summary_width` and `observable_summary_width` are non-zero
- Check if summary calculation uses correct dt_save/dt_summarise ratio

### 5. SolveResult NaN Processing (`tests/batchsolving/test_solveresult.py`)

Tests manipulate status codes to verify NaN processing behavior.

**Current Issue**:
- `test_successful_runs_unchanged_with_nan_enabled` - expects no NaN, but sees NaN
- `test_multiple_errors_all_set_to_nan` - expects specific NaN pattern

**Root Cause Hypothesis**:
- Status codes may be non-zero due to integration failures (not just fixture manipulation)
- Session-scoped fixture `solved_batch_solver_errorcode` may inherit errors from previous tests

**Investigation Required**:
- Verify baseline integration succeeds before status code manipulation
- Check if session scope causes test pollution

## Architectural Context

### Buffer Registry Pattern

The `buffer_registry` singleton manages buffer metadata:

```
buffer_registry.register(name, owner, size, location, precision)
    → Creates CUDABuffer record
    → Computes slice layout when build() is called
    → Returns allocator device function
```

### SingleIntegratorRun Assembly

```
SingleIntegratorRun.__init__():
    1. Creates algorithm step object (e.g., BackwardsEulerStep)
    2. Algorithm registers its buffers with buffer_registry
    3. Creates IVPLoop
    4. IVPLoop registers its buffers with buffer_registry
    5. Creates step controller
    6. Controller registers its buffers (if any)
    7. Properties query buffer_registry for total sizes
```

### Expected Buffer Sizes

For a 3-state implicit system (backwards_euler):
- State: 3 elements
- Proposed state: 3 elements
- Parameters: 3 elements (example)
- Drivers: 1+ elements
- Newton solver workspace: ~6-12 elements (2-4 × n_states)
- **Total shared**: Should be non-zero

## Integration Points

### Algorithm → Buffer Registry

Algorithms register workspace buffers during initialization:

```python
# Example from BackwardsEulerStep
buffer_registry.register(
    'newton_scratch', self, 2 * n_states, 'shared', precision=precision
)
```

### Loop → Algorithm

IVPLoop passes shared/persistent arrays to step_function:

```python
status = step_function(
    state, proposed_state, params, driver_coeffs,
    drivers, proposed_drivers, observables, proposed_observables,
    error, dt, time, first_step, accepted, shared, persistent, counters
)
```

### Test → SingleIntegratorRun

Tests access composite sizes via properties:

```python
shared_bytes = singleintegratorrun.shared_memory_bytes
local_elements = singleintegratorrun.local_memory_elements
```

## Dependencies and Imports

### Test File Dependencies

```
tests/_utils.py:
    from cubie import SingleIntegratorRun
    from cubie.outputhandling import OutputFunctions
    from cubie.integrators.array_interpolator import ArrayInterpolator

tests/conftest.py:
    from cubie.integrators.SingleIntegratorRun import SingleIntegratorRun
    from cubie.integrators.algorithms import get_algorithm_step
    from cubie.integrators.step_control import get_controller
    from cubie.outputhandling.output_functions import OutputFunctions

tests/integrators/algorithms/test_step_algorithms.py:
    from cubie.integrators.algorithms import get_algorithm_step
    from tests.integrators.cpu_reference import get_ref_stepper
```

## Edge Cases

### Zero Drivers
When `n_drivers == 0`:
- Driver buffer should have size 0 or minimum 1
- Driver function may be None
- CPU reference must handle this consistently

### Non-Adaptive Algorithms
When algorithm has no error estimate:
- Error buffer may have size 0
- Controller should be FixedStepController
- dt remains unchanged throughout integration

### Session-Scoped Fixture Pollution
Tests modifying shared state (like status codes) must:
- Save original values before modification
- Restore after test completion
- Not rely on test execution order

## Data Structures

### LoopRunResult (tests/_utils.py)
```python
@attrs.define
class LoopRunResult:
    state: Array           # (save_samples, state_width)
    observables: Array     # (save_samples, observable_width)
    state_summaries: Array # (summary_samples, state_summary_width)
    observable_summaries: Array
    status: int
    counters: Array = None
```

### StepResult (tests/integrators/algorithms/test_step_algorithms.py)
```python
@attrs.define
class StepResult:
    state: Array
    observables: Array
    error: Array
    status: int
    niters: int
    counters: Optional[Array] = None
```

## Expected Interactions

### Test Kernel Launch

```python
@cuda.jit
def kernel(init_vec, params_vec, ...):
    shared = cuda.shared.array(0, dtype=numba_precision)  # dynamically sized
    persistent = cuda.local.array(local_len, dtype=numba_precision)
    # ...
    status = loop_fn(init_vec, params_vec, ..., shared, persistent, ...)

# Launch with correct shared memory
kernel[1, 1, 0, shared_bytes](...)
```

The `shared_bytes` parameter in the launch configuration must equal or exceed the total shared memory registered by all components.

### CPU Reference Loop

```python
run_reference_loop(
    evaluator=cpu_system,
    inputs=inputs,
    driver_evaluator=cpu_driver_evaluator,
    solver_settings=solver_settings,  # Must contain all tolerance values
    output_functions=output_functions,
    controller=controller,
    tableau=tableau,
)
```

The `solver_settings` dict must contain identical values used to configure the GPU algorithm.
