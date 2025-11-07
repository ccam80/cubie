# CuBIE Internal Structure

CuBIE (CUDA Batch Integration Engine) is a Python library for GPU-accelerated batch integration of ODEs/SDEs using Numba CUDA. This document provides architectural context for agents working on the codebase.

## Core Architecture

### CUDAFactory Pattern
Base class in `src/cubie/CUDAFactory.py` that implements cached compilation of Numba CUDA device functions. All CUDA-generating components inherit from this:
- Subclasses override `build()` to return compiled CUDA device function or attrs class containing multiple cached outputs
- Compile settings stored as attrs classes; any change invalidates cache
- Never call `build()` directly; access via properties (automatically cache or build as needed)
- Properties: `device_function`, `compile_settings`, `cache_valid`
- Methods: `setup_compile_settings()`, `update_compile_settings()`, `get_cached_output()`
- Cache invalidation is automatic when settings change

#### Compile Settings and Cache Invalidation Pattern
CUDAFactory subclasses use attrs classes as compile settings to enable automatic cache invalidation:

**Pattern Overview:**
1. Factory defines attrs class for compile-time configuration
2. Factory calls `setup_compile_settings(attrs_instance)` during initialization
3. CUDAFactory stores settings and marks cache as valid on first `build()`
4. When settings change via `update_compile_settings()`, cache is automatically invalidated
5. Next property access triggers rebuild with new settings

**Example: OutputFunctions with dt_save**
```python
# OutputConfig is an attrs class with dt_save field
config = OutputConfig(max_states=3, dt_save=0.01, ...)

# OutputFunctions inherits from CUDAFactory
output_funcs = OutputFunctions(...)
output_funcs.setup_compile_settings(config)

# First access builds and caches
funcs = output_funcs.device_function  # build() called

# Update dt_save invalidates cache
config.dt_save = 0.02
output_funcs.update_compile_settings(config)  # Cache invalidated

# Next access rebuilds with new dt_save
funcs = output_funcs.device_function  # build() called again with dt_save=0.02
```

**Implementation Details:**
- Compile settings must be attrs classes (supports comparison via `__eq__`)
- Array fields use `eq=attrs.cmp_using(eq=array_equal)` for numpy array comparison
- Settings accessible in `build()` via `self.compile_settings`
- dt_save and other compile-time constants can be captured in closure within `build()`
- No need to pass constants as device function parameters when in closure

**Adding New Compile-Time Parameters:**
1. Add field to attrs compile_settings class (e.g., `dt_save` to `OutputConfig`)
2. Add parameter name to corresponding `ALL_*_PARAMETERS` set (e.g., `ALL_OUTPUT_FUNCTION_PARAMETERS`)
3. Access in `build()` method via `self.compile_settings.dt_save`
4. Use in closure when compiling device functions (captured at compile time)
5. Cache invalidation happens automatically when parameter changes

### Attrs Classes Pattern
Used throughout for data containers and compile settings:
- All configuration classes use `@attrs.define` decorator
- Floating-point attributes stored with leading underscore, exposed via property returning `self.precision(self._attribute)`
- Never include underscore in `__init__` calls (attrs handles internally)
- Never add aliases to underscored variables
- Compile settings containers must be attrs classes when passed to CUDAFactory

#### Validators
- Use validators from `cubie._utils` (imported from full path) rather than `instance_of(float)`
- `cubie._utils` validators are tolerant of NumPy dtypes (e.g., `np.floating`, `np.integer`)
- Available validators: `getype_validator(dtype, min)`, `gttype_validator(dtype, min)`, `letype_validator(dtype, max)`, `lttype_validator(dtype, max)`, `inrangetype_validator(dtype, min, max)`
- For optional fields, use `validators.optional(...)` wrapper around cubie._utils validators

### Precision System
Centralized in `src/cubie/_utils.py`:
- `PrecisionDType` type alias for np.float16/32/64 types
- `precision_converter()` and `precision_validator()` enforce allowed precisions
- `ALLOWED_PRECISIONS` set contains np.float16, np.float32, np.float64
- Numba types obtained via `from_dtype()` from cuda_simsafe module

## Module Hierarchy

### Root Package (`src/cubie/`)
- `__init__.py` - Exposes public API: Solver, solve_ivp, summary_metrics, SymbolicODE, create_ODE_system, ArrayTypes
- `CUDAFactory.py` - Base class for all CUDA-generating components
- `_utils.py` - Shared utilities: precision handling, array slicing, validators, timing decorators
- `cuda_simsafe.py` - Compatibility layer for CUDA simulator mode (`NUMBA_ENABLE_CUDASIM=1`)

### ODE Systems (`src/cubie/odesystems/`)
Defines and compiles CUDA-ready ODE system representations:
- `baseODE.py` - `BaseODE(CUDAFactory)`: Abstract base for ODE systems; manages values, precision, caching
  - `ODECache` attrs class: Holds compiled dxdt and optional solver helpers (linear_operator, preconditioner, etc.)
  - Subclasses override `build()` to compile dxdt device function
- `ODEData.py` - `ODEData` attrs class: Bundles numerical values and metadata; `SystemSizes` for component counts
- `SystemValues.py` - `SystemValues`: Name-value mapping for states, parameters, constants, observables

#### Symbolic Subsystem (`odesystems/symbolic/`)
SymPy-driven CUDA code generation:
- `symbolicODE.py` - `SymbolicODE(BaseODE)`: Generates CUDA kernels from symbolic expressions
- `odefile.py` - Manages generated code directory (`GENERATED_DIR`)
- `parsing/` - Parsers for symbolic definitions (CellML, JVP equations, auxiliary caching)
- `codegen/` - CUDA code generators:
  - `dxdt.py` - Time derivative kernels
  - `jacobian.py` - Jacobian matrix computations
  - `linear_operators.py` - Matrix-free operators for implicit methods
  - `preconditioners.py` - Neumann series preconditioners
  - `nonlinear_residuals.py` - Residual functions for Newton solvers
  - `numba_cuda_printer.py` - Custom SymPy printer for Numba CUDA
- `solver_helpers.py` - Generates helper functions for Newton-Krylov solvers
- `indexedbasemaps.py` - Maps symbolic variables to device memory indices

### Integrators (`src/cubie/integrators/`)
Numerical integration components for IVP solving:
- `SingleIntegratorRun.py` - Main entry point wrapping SingleIntegratorRunCore; exposes read-only properties
- `SingleIntegratorRunCore.py` - Core logic for assembling loop, controller, algorithm into compiled device function
- `IntegratorRunSettings.py` - Settings container for integrator configuration
- `IntegratorReturnCodes` - Enum for algorithm-level status codes (SUCCESS, NEWTON_BACKTRACKING_NO_SUITABLE_STEP, etc.)
- `array_interpolator.py` - Device-side interpolation for forcing functions

#### Algorithms (`integrators/algorithms/`)
Step function factories sharing common signature and returning status codes:
- `base_algorithm_step.py` - `BaseStepConfig` for precision, `StepCache` for outputs
- `explicit_euler.py` - `ExplicitEulerStep`
- `backwards_euler.py` - `BackwardsEulerStep` (Newton-Krylov)
- `backwards_euler_predict_correct.py` - `BackwardsEulerPCStep` (predictor-corrector)
- `crank_nicolson.py` - `CrankNicolsonStep`
- `generic_erk.py`, `generic_dirk.py`, `generic_firk.py`, `generic_rosenbrock_w.py` - Tableau-based Runge-Kutta methods
- `ode_explicitstep.py`, `ode_implicitstep.py` - Explicit/implicit step wrappers
- `get_algorithm_step()` - Factory function to retrieve algorithm by name
- `ALL_ALGORITHM_STEP_PARAMETERS` - Set of recognized configuration keys

#### Matrix-Free Solvers (`integrators/matrix_free_solvers/`)
CUDA device solvers for implicit methods:
- `linear_solver.py` - `linear_solver_factory()`: GMRES-like iterative solver
- `newton_krylov.py` - `newton_krylov_solver_factory()`: Newton-Krylov iteration
- `SolverRetCodes` - Enum for solver status (SUCCESS, MAX_ITERATIONS_EXCEEDED, etc.)
- Upper 16 bits encode Newton iteration count when returned from implicit algorithms
- Relies on warp-vote helpers for convergence checks
- Expects caller-supplied operator/residual callbacks and preallocated buffers

#### Step Control (`integrators/step_control/`)
Adaptive and fixed step-size controllers:
- `base_step_controller.py` - Base controller interface, `ALL_STEP_CONTROLLER_PARAMETERS`
- `fixed_step_controller.py` - `FixedStepController` (no-op)
- `adaptive_I_controller.py` - `AdaptiveIController` (integral only)
- `adaptive_PI_controller.py` - `AdaptivePIController` (proportional-integral)
- `adaptive_PID_controller.py` - `AdaptivePIDController` (proportional-integral-derivative)
- Controllers return CUDA device functions that propose new step sizes

#### Loops (`integrators/loops/`)
CUDA loop construction:
- `ode_loop.py` - `IVPLoop(CUDAFactory)`: Compiles main integration loop
- `ode_loop_config.py` - `LoopSharedIndices`, `LoopLocalIndices`, `ODELoopConfig` for memory layouts
- `ALL_LOOP_SETTINGS` - Set of recognized loop configuration keys

### Memory Management (`src/cubie/memory/`)
GPU memory subsystem with optional CuPy integration:
- `mem_manager.py` - `MemoryManager` singleton orchestrating allocations:
  - Instance registry with proportion-based VRAM caps
  - Chunked allocation support for large arrays
  - Invalidation hooks for cache management
  - Stream grouping integration
  - `ALL_MEMORY_MANAGER_PARAMETERS` - Set of recognized configuration keys
- `array_requests.py` - `ArrayRequest`/`ArrayResponse` containers for allocation metadata
- `stream_groups.py` - `StreamGroups` manages CUDA streams for async operations
- `cupy_emm.py` - CuPy memory pool integration via Numba EMM interface:
  - `CuPyAsyncNumbaManager`, `CuPySyncNumbaManager`
  - `current_cupy_stream()` context manager for stream adoption
  - Falls back to Numba allocator if CuPy unavailable
- `default_memmgr` - Global singleton instance exported from package

### Output Handling (`src/cubie/outputhandling/`)
CUDA output and summary metric system:
- `output_functions.py` - `OutputFunctions(CUDAFactory)`: Compiles save/summary callbacks
  - `OutputFunctionCache` attrs class for cached outputs
  - `ALL_OUTPUT_FUNCTION_PARAMETERS` - Set of recognized configuration keys
- `output_config.py` - `OutputConfig`, `OutputCompileFlags` for validated settings
- `output_sizes.py` - Sizing helpers for buffer planning:
  - `BatchOutputSizes`, `SingleRunOutputSizes`, `LoopBufferSizes`
  - `SummariesBufferSizes`, `BatchInputSizes`, `OutputArrayHeights`

#### Summary Metrics (`outputhandling/summarymetrics/`)
Extensible metric registry:
- `metrics.py` - `SummaryMetrics` registry with decorator `register_metric()`
- Built-in metrics auto-register on import: mean, max, rms, peaks
- Each metric module returns (update_callable, save_callable) pair
- Metrics compile CUDA device functions via CUDAFactory
- `summary_metrics` singleton instance exported

### Batch Solving (`src/cubie/batchsolving/`)
High-level batch integration API:
- `solver.py` - User-facing interface:
  - `Solver` class: Configures and executes batch runs
  - `solve_ivp()` convenience wrapper
- `BatchSolverKernel.py` - `BatchSolverKernel(CUDAFactory)`: Compiles batch kernel
  - Delegates to SingleIntegratorRun for per-system logic
  - Handles chunking for memory constraints
  - `ChunkParams` attrs class for chunk metadata
- `BatchSolverConfig.py` - `BatchSolverConfig` attrs class for solver settings
- `BatchGridBuilder.py` - `BatchGridBuilder`: Constructs parameter grids (combinatorial or verbatim)
- `SystemInterface.py` - `SystemInterface`: Adapts BaseODE instances for kernels
- `solveresult.py` - `SolveResult`, `SolveSpec` containers for results and metadata
- `_utils.py` - Validators for CUDA arrays (1D, 2D, 3D, optional variants)

#### Array Management (`batchsolving/arrays/`)
Host/device buffer coordination:
- `BaseArrayManager.py` - Base utilities:
  - `BaseArrayManager`: Registers containers with memory manager, chunk-aware transfers
  - `ArrayContainer`, `ManagedArray` attrs classes
- `BatchInputArrays.py`:
  - `InputArrayContainer`: Host-visible input arrays (initial_values, parameters, drivers)
  - `InputArrays`: Manager sizing from solver metadata, device buffer sync
- `BatchOutputArrays.py`:
  - `OutputArrayContainer`: Host-visible output arrays (state, observables, summaries)
  - `ActiveOutputs`: Flags for enabled outputs
  - `OutputArrays`: Manager collecting device trajectories and summaries
- Modules use `cubie.outputhandling.output_sizes` for stride metadata

## Data Flow Patterns

### ODE System Creation
1. User provides symbolic expressions or BaseODE subclass
2. `create_ODE_system()` or direct instantiation → `SymbolicODE` instance
3. SymbolicODE compiles dxdt and solver helpers via codegen pipeline
4. Results cached in `ODECache` attrs class
5. System metadata stored in `ODEData` (states, parameters, constants, precision)

### Solver Initialization
1. User calls `Solver(system, algorithm=...)` or `solve_ivp(system, ...)`
2. Solver creates `BatchSolverKernel(CUDAFactory)`:
   - Instantiates `SingleIntegratorRun` with algorithm, controller, loop settings
   - SingleIntegratorRun composes: IVPLoop + algorithm step + controller
   - All compiled to CUDA device functions via CUDAFactory pattern
3. Memory manager allocates VRAM based on instance proportions
4. `BatchGridBuilder` constructs parameter grid from user inputs

### Batch Execution
1. `Solver.solve()` creates `InputArrays` and `OutputArrays` managers
2. Arrays registered with memory manager in appropriate stream group
3. Input data transferred host→device
4. Kernel launched with configured grid/block dimensions
5. Each thread runs SingleIntegratorRun device loop:
   - Reads parameters from device input arrays
   - Iterates timesteps calling algorithm step function
   - Controller adjusts step size if adaptive
   - Output functions save states/observables at dt_save intervals
   - Summary metrics accumulated continuously
6. Device→host transfer after kernel completion
7. Results packaged in `SolveResult` container

### Compilation Caching
1. CUDAFactory subclass receives compile settings (attrs class)
2. First property access triggers `build()` if cache invalid
3. `build()` returns device function or attrs class with multiple outputs
4. Cache marked valid; subsequent accesses return cached result
5. `update_compile_settings()` invalidates cache if values change
6. Array comparisons use `array_equal()` for numpy arrays

### Memory Chunking
1. BatchSolverKernel calculates required VRAM for full batch
2. If exceeds instance cap, splits into chunks
3. `ChunkParams` computed for each chunk (size, runs, duration, t0, warmup)
4. Chunks executed sequentially, results concatenated
5. Memory manager tracks allocations per registered instance

## Testing Infrastructure

### Pytest Configuration
- Root conftest.py in `tests/`
- Markers: nocudasim, cupy, slow, specific_algos, sim_only
- Coverage configured for src/cubie with exclusions for pragma comments
- Test without CUDA: `pytest -m "not nocudasim and not cupy"`
- Fixtures use indirect parameterization (settings dicts)

### Fixture Patterns
Defined in `tests/conftest.py` and `tests/system_fixtures.py`:
- System fixtures: three_state_linear, three_state_nonlinear, three_chamber, large_nonlinear
- Component fixtures: precision, algorithm, controller, loop settings
- Use indirect overrides via "override" fixtures pattern
- Prefer instantiating real cubie objects over mocks
- Never shortcut `is_device` checks or patch CUDA availability

### CPU Reference Implementations
Located in `tests/integrators/cpu_reference.py`:
- `CPUODESystem` - Pure Python ODE evaluation
- `CPUAdaptiveController` - Reference step control
- `run_reference_loop()` - Reference integration loop
- Used for validation against GPU results

### Test Organization
- Mirror src/ structure: tests/integrators/, tests/odesystems/, etc.
- Each module has corresponding test_<module>.py
- Parametrized tests for different algorithms, precisions, system sizes
- Separate test files for algorithms, loops, controllers, memory management

## Common Gotchas

### CUDAFactory Usage
- Never call `build()` directly on subclass instances
- Always use property access: `obj.device_function`, not `obj.build()`
- Properties auto-cache; manual `build()` defeats cache invalidation
- Storing `device_function` reference then updating settings creates stale reference
- Always re-access property after settings updates

### Attrs Classes
- Floating-point attributes need leading underscore + property wrapper
- Don't pass underscored names to `__init__` (attrs handles mapping)
- Compile settings must be attrs classes to work with CUDAFactory
- Use `in_attr()` utility to check field existence (handles underscore variants)

### Memory Management
- Registered instances have VRAM proportions summing to ≤1.0
- Chunking is automatic when allocations exceed cap
- Invalidation hooks must be set before allocations
- Stream groups coordinate async operations; mismatched streams cause issues
- CuPy integration requires explicit EMM setup via set_cuda_memory_manager()

### CUDA Simulation Mode
- Environment variable `NUMBA_ENABLE_CUDASIM="1"` for CPU-only development
- Never set in source code; only via external environment
- cuda_simsafe module provides compatibility layer
- Some tests marked nocudasim will fail in simulator mode
- Simulator mode bypasses real GPU memory constraints

### Algorithm Return Codes
- Implicit algorithms embed Newton iteration count in upper 16 bits
- Extract via bitwise operations: `iterations = (code >> 16) & 0xFFFF`
- Status in lower 16 bits: `status = code & 0xFFFF`
- Compare against IntegratorReturnCodes enum values
- Solver codes distinct from integrator codes despite overlap

### Precision Handling
- All arithmetic uses precision specified in ODEData
- Numba types obtained via `from_dtype(precision)` from cuda_simsafe
- Mixed precision not supported; entire system uses single precision
- Properties returning floats must wrap with `self.precision(value)`

### Loop and Buffer Sizing
- Shared memory limited to ~48KB per block on most GPUs
- Local memory (persistent registers) limited per thread
- SingleIntegratorRun tracks `shared_memory_bytes`, `local_memory_elements`
- Output buffer strides computed by output_sizes helpers
- Summaries stored flat with stride calculations for multi-variable arrays

## Build and Test Commands

### Installation
```bash
pip install -e .[dev]  # Development install with test dependencies
```

### Linting
```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
ruff check .  # Line length 79, max-doc-length 72
```

### Testing
```bash
pytest                                    # Full test suite with CUDA
pytest -m "not nocudasim and not cupy"    # CPU-only tests
pytest tests/integrators/                 # Specific module
pytest -k test_solver                     # Specific test pattern
pytest --durations=10                     # Show slowest tests
```

### Coverage
```bash
pytest --cov=cubie --cov-report=html      # HTML coverage report
pytest --cov=cubie --cov-report=term      # Terminal coverage
```

## Dependencies

### Core
- numpy==1.26.4 (pinned for stability)
- numba (JIT compiler)
- numba-cuda[cu12] (CUDA support)
- attrs (data classes)
- sympy (symbolic math for codegen)

### Development
- pytest, pytest-cov, pytest-durations, pytest-json-report
- flake8, ruff (linting)
- scipy (reference implementations)

### Optional
- cupy-cuda12x (GPU array operations, memory pool)
- pandas (data manipulation)
- matplotlib (plotting)

## File Organization

### Source Tree
```
src/cubie/
├── __init__.py                    # Public API exports
├── CUDAFactory.py                 # Base cached compilation class
├── _utils.py                      # Shared utilities
├── cuda_simsafe.py                # CUDA simulator compatibility
├── batchsolving/                  # Batch solver API
│   ├── solver.py                  # Solver, solve_ivp
│   ├── BatchSolverKernel.py       # Kernel factory
│   ├── BatchGridBuilder.py        # Parameter grid construction
│   ├── SystemInterface.py         # ODE system adapter
│   ├── solveresult.py             # Result containers
│   └── arrays/                    # Array managers
│       ├── BaseArrayManager.py
│       ├── BatchInputArrays.py
│       └── BatchOutputArrays.py
├── integrators/                   # Integration components
│   ├── SingleIntegratorRun.py     # Main integrator wrapper
│   ├── IntegratorRunSettings.py
│   ├── algorithms/                # Step functions
│   ├── loops/                     # CUDA loop builders
│   ├── matrix_free_solvers/       # Newton-Krylov solvers
│   └── step_control/              # Adaptive controllers
├── memory/                        # Memory management
│   ├── mem_manager.py             # MemoryManager singleton
│   ├── array_requests.py          # Request/response containers
│   ├── stream_groups.py           # CUDA stream coordination
│   └── cupy_emm.py                # CuPy integration
├── odesystems/                    # ODE system definitions
│   ├── baseODE.py                 # BaseODE abstract class
│   ├── ODEData.py                 # Data containers
│   ├── SystemValues.py            # Value mappings
│   └── symbolic/                  # SymPy codegen
│       ├── symbolicODE.py
│       ├── odefile.py
│       ├── codegen/               # CUDA generators
│       └── parsing/               # Expression parsers
└── outputhandling/                # Output and metrics
    ├── output_functions.py        # OutputFunctions factory
    ├── output_config.py           # Configuration
    ├── output_sizes.py            # Buffer sizing
    └── summarymetrics/            # Metric registry
        ├── metrics.py
        ├── mean.py
        ├── max.py
        ├── rms.py
        └── peaks.py
```

### Test Tree
```
tests/
├── conftest.py                    # Global fixtures
├── system_fixtures.py             # ODE system builders
├── _utils.py                      # Test utilities
├── batchsolving/                  # Batch solver tests
├── integrators/                   # Integrator tests
│   ├── cpu_reference.py           # Reference implementations
│   ├── algorithms/
│   ├── loops/
│   └── step_control/
├── memory/                        # Memory tests
├── odesystems/                    # ODE system tests
└── outputhandling/                # Output tests
```

---
*This file is maintained by agents to provide architectural context for future development work.*
