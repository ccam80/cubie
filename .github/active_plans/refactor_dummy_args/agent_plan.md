# Agent Plan: Refactor Dummy Arguments for Compile-Time Logging

## Problem Statement

The compile-time logging system has two bugs:
1. `_create_placeholder_args()` checks `device_function.signatures` which is an empty list `[]` when device signatures haven't been populated, not `None`. The check `if device_function.signatures:` is always `False` for lazily-compiled functions.
2. Once fixed, running with default scalar arguments (1.0) fails for device functions that expect arrays with specific shapes for indexing operations.

## Solution Architecture

### New Abstract Method: `_generate_dummy_args()`

Add an abstract method to `CUDAFactory`:

```python
@abstractmethod
def _generate_dummy_args(self) -> Dict[str, Tuple]:
    """Generate dummy arguments for compile-time measurement.
    
    Returns
    -------
    Dict[str, Tuple]
        Mapping of cached output names to their dummy argument tuples.
        Each tuple contains NumPy arrays and scalars matching the
        device function's signature with appropriate shapes.
    
    Notes
    -----
    This method is called by specialize_and_compile() to trigger
    CUDA compilation with correctly-shaped arguments. Subclasses
    must implement this to return arguments that won't cause
    illegal memory access or infinite loops during dummy execution.
    """
```

### Expected Behavior

Each CUDAFactory subclass implements `_generate_dummy_args()` to:
1. Access `self.compile_settings` for sizing information
2. Create minimal NumPy arrays with correct shapes
3. Create scalar values that produce finite execution (e.g., small duration)
4. Return a dict mapping device function names to argument tuples

### Component Changes

#### 1. CUDAFactory (src/cubie/CUDAFactory.py)

**Modifications:**
- Add `_generate_dummy_args()` as abstract method with return type `Dict[str, Tuple]`
- Modify `specialize_and_compile()` to:
  - Accept `dummy_args: Tuple` parameter instead of calling `_create_placeholder_args()`
  - Use provided dummy_args directly
- Modify `_build()` to:
  - Call `self._generate_dummy_args()` once after build()
  - Pass appropriate dummy args to `specialize_and_compile()` for each device function
- Remove or simplify `_create_placeholder_args()` function
- Keep `_run_placeholder_kernel()` as-is (it handles array transfer and kernel invocation)

**New method signature:**
```python
def specialize_and_compile(
    self, device_function: Any, event_name: str, dummy_args: Tuple
) -> None:
```

#### 2. BatchSolverKernel (src/cubie/batchsolving/BatchSolverKernel.py)

**Implement `_generate_dummy_args()`:**
- Access system_sizes from compile_settings or single_integrator
- Generate arrays matching integration_kernel signature:
  - inits: (n_states, 1)
  - params: (n_parameters, 1)
  - d_coefficients: (100, n_drivers, 6) or minimal
  - state_output: (100, n_states, 1)
  - observables_output: (100, n_observables, 1)
  - state_summaries_output: (100, n_states, 1)
  - observable_summaries_output: (100, n_observables, 1)
  - iteration_counters_output: (100, 4, 1)
  - status_codes_output: (1,)
  - duration: 0.001 (small value)
  - warmup: 0.0
  - t0: 0.0
  - n_runs: 1

**Remove:**
- `integration_kernel.critical_shapes = (...)` assignment
- `integration_kernel.critical_values = (...)` assignment

#### 3. IVPLoop (src/cubie/integrators/loops/ode_loop.py)

**Implement `_generate_dummy_args()`:**
- Access config.n_states, n_parameters, n_observables, n_counters
- Generate arrays matching loop_fn signature:
  - initial_states: (n_states,)
  - parameters: (n_parameters,)
  - driver_coefficients: (100, n_states, 6)
  - shared_scratch: (4096,) # arbitrary shared memory
  - persistent_local: (4096,) # arbitrary persistent local
  - state_output: (100, n_states)
  - observables_output: (100, n_observables)
  - state_summaries_output: (100, n_states)
  - observable_summaries_output: (100, n_observables)
  - iteration_counters_output: (1, n_counters)
  - duration: dt_save + 0.01 (small value ensuring one save)
  - settling_time: 0.0
  - t0: 0.0

**Remove:**
- `loop_fn.critical_shapes = (...)` assignment
- `loop_fn.critical_values = (...)` assignment

#### 4. SingleIntegratorRunCore (src/cubie/integrators/SingleIntegratorRunCore.py)

**Implement `_generate_dummy_args()`:**
- This factory returns a loop function from IVPLoop
- Delegate to `self._loop._generate_dummy_args()`
- Return dict mapping 'single_integrator_function' to the loop's dummy args

#### 5. BaseAlgorithmStep (src/cubie/integrators/algorithms/base_algorithm_step.py)

**Add abstract `_generate_dummy_args()`:**
- Each algorithm step subclass implements based on its signature
- Base implementation can provide guidance in docstring

**Concrete implementations needed in:**
- ExplicitEulerStep
- BackwardsEulerStep
- BackwardsEulerPCStep
- CrankNicolsonStep
- GenericERKStep
- GenericDIRKStep
- GenericFIRKStep
- GenericRosenbrockWStep

Each must return dummy args matching their step function signature.

#### 6. BaseStepController (src/cubie/integrators/step_control/base_step_controller.py)

**Add abstract `_generate_dummy_args()`:**
- Controllers have simpler signatures
- Concrete implementations in:
  - FixedStepController
  - AdaptiveIController
  - AdaptivePIController
  - AdaptivePIDController

#### 7. OutputFunctions (src/cubie/outputhandling/output_functions.py)

**Implement `_generate_dummy_args()`:**
- Returns dict with entries for:
  - save_state_function
  - update_summaries_function
  - save_summaries_function
- Each with appropriate array shapes from config

#### 8. BaseODE (src/cubie/odesystems/baseODE.py)

**Add abstract `_generate_dummy_args()`:**
- Returns dict with dummy args for dxdt and any solver helpers
- SymbolicODE implements with proper shapes from system_sizes

#### 9. BufferRegistry (src/cubie/buffer_registry.py)

**No changes required** - BufferRegistry manages memory layout, not dummy arguments.

### Integration Points

1. **Build-time flow:**
   - `CUDAFactory._build()` calls `self.build()` to get cache
   - If verbosity enabled, calls `self._generate_dummy_args()`
   - For each device function in cache, calls `specialize_and_compile()` with appropriate dummy args

2. **Dummy args structure:**
   ```python
   {
       'device_function_name': (arg1, arg2, arg3, ...),
       'another_function': (arg1, arg2, ...),
   }
   ```

3. **Timing happens after build:**
   - Build produces device functions (not yet specialized)
   - `_generate_dummy_args()` produces test data
   - `specialize_and_compile()` triggers compilation with test data
   - Timing recorded for each function

### Edge Cases

1. **Device functions with no parameters:**
   - Return empty tuple in dummy_args dict
   - `_run_placeholder_kernel` handles param_count == 0

2. **Factories with multiple cached outputs:**
   - Dict maps each output name to its dummy args
   - Only device functions (has `py_func`) get compiled

3. **Optional solver helpers (returns -1):**
   - Skip timing for outputs == -1
   - Already handled in current _build()

4. **CUDA simulator mode:**
   - `specialize_and_compile()` already returns early
   - No dummy args needed in CUDASIM

### Dependencies

The implementation order should be:
1. CUDAFactory base changes (abstract method, specialize_and_compile signature)
2. Simple factories: OutputFunctions, BaseStepController implementations
3. Algorithm steps: BaseAlgorithmStep and concrete implementations
4. Loop: IVPLoop
5. Higher-level: SingleIntegratorRunCore, BatchSolverKernel
6. ODE systems: BaseODE and SymbolicODE

### Data Structures

**Dummy args return type:**
```python
Dict[str, Tuple[Union[np.ndarray, float, int], ...]]
```

Where:
- Key: Cache attribute name (e.g., 'device_function', 'step', 'loop_function')
- Value: Tuple of arguments matching device function signature

**Example for BatchSolverKernel:**
```python
def _generate_dummy_args(self) -> Dict[str, Tuple]:
    precision = self.compile_settings.precision
    system_sizes = self.system_sizes
    n_states = int(system_sizes.states)
    n_params = int(system_sizes.parameters)
    n_obs = int(system_sizes.observables)
    
    return {
        'solver_kernel': (
            np.ones((n_states, 1), dtype=precision),  # inits
            np.ones((n_params, 1), dtype=precision),  # params
            np.ones((100, n_states, 6), dtype=precision),  # d_coefficients
            np.ones((100, n_states, 1), dtype=precision),  # state_output
            np.ones((100, n_obs, 1), dtype=precision),  # observables_output
            np.ones((100, n_states, 1), dtype=precision),  # state_summaries
            np.ones((100, n_obs, 1), dtype=precision),  # obs_summaries
            np.ones((100, 4, 1), dtype=np.int32),  # iteration_counters
            np.ones((1,), dtype=np.int32),  # status_codes
            np.float64(0.001),  # duration
            np.float64(0.0),  # warmup
            np.float64(0.0),  # t0
            np.int32(1),  # n_runs
        ),
    }
```

### Validation

After implementation:
1. Run tests with `NUMBA_ENABLE_CUDASIM=1` to verify basic functionality
2. Run with actual CUDA and verbosity enabled to verify timing works
3. Check that no `critical_shapes` or `critical_values` attributes remain
4. Verify compile times are recorded in time logger
