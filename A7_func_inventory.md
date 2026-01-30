# A7 Functionality Inventory

Covers all `__init__.py` files, `IntegratorRunSettings.py`, and
`vendored/numba_cuda_cache.py`.

---

## 1. `src/cubie/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Sets `NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS` env var to `"0"` at import time |
| 2 | Re-exports from `batchsolving`, `integrators`, `outputhandling`, `memory`, `odesystems`, `_utils` via star imports |
| 3 | Imports `TimeLogger` and `default_timelogger` from `cubie.time_logger` |
| 4 | `__all__` contains exactly: `summary_metrics`, `default_memmgr`, `ArrayTypes`, `Solver`, `solve_ivp`, `SymbolicODE`, `create_ODE_system`, `TimeLogger`, `default_timelogger`, `load_cellml_model` |
| 5 | `__version__` set from `importlib.metadata.version("cubie")` when installed |
| 6 | `__version__` falls back to `"unknown"` when `ImportError` raised (package not installed) |

---

## 2. `src/cubie/integrators/__init__.py`

### Module-Level Logic & Exports

| # | Functionality |
|---|--------------|
| 1 | Re-exports `SingleIntegratorRun` from `.SingleIntegratorRun` |
| 2 | Re-exports algorithm classes: `BackwardsEulerPCStep`, `BackwardsEulerStep`, `CrankNicolsonStep`, `ExplicitEulerStep`, `ExplicitStepConfig`, `ImplicitStepConfig`, `get_algorithm_step` |
| 3 | Re-exports `IVPLoop` from `.loops` |
| 4 | Re-exports solver classes: `LinearSolver`, `LinearSolverConfig`, `LinearSolverCache`, `NewtonKrylov`, `NewtonKrylovConfig`, `NewtonKrylovCache` |
| 5 | Re-exports controller classes: `AdaptiveIController`, `AdaptivePIController`, `AdaptivePIDController`, `FixedStepController`, `GustafssonController`, `get_controller` |

### `IntegratorReturnCodes` (IntEnum)

| # | Functionality |
|---|--------------|
| 6 | `SUCCESS = 0` |
| 7 | `NEWTON_BACKTRACKING_NO_SUITABLE_STEP = 1` |
| 8 | `MAX_NEWTON_ITERATIONS_EXCEEDED = 2` |
| 9 | `MAX_LINEAR_ITERATIONS_EXCEEDED = 4` |
| 10 | `STEP_TOO_SMALL = 8` |
| 11 | `DT_EFF_EFFECTIVELY_ZERO = 16` |
| 12 | `MAX_LOOP_ITERS_EXCEEDED = 32` |

---

## 3. `src/cubie/integrators/algorithms/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | `_ALGORITHM_REGISTRY` maps 8 string keys to step classes: `euler`, `backwards_euler`, `backwards_euler_pc`, `crank_nicolson`, `dirk`, `firk`, `erk`, `rosenbrock` |
| 2 | `_TABLEAU_REGISTRY_BY_ALGORITHM` initialised from `_ALGORITHM_REGISTRY` with `None` tableaus |
| 3 | ERK tableau aliases merged into `_TABLEAU_REGISTRY_BY_ALGORITHM` with `ERKStep` constructor |
| 4 | DIRK tableau aliases merged into `_TABLEAU_REGISTRY_BY_ALGORITHM` with `DIRKStep` constructor |
| 5 | FIRK tableau aliases merged into `_TABLEAU_REGISTRY_BY_ALGORITHM` with `FIRKStep` constructor |
| 6 | Rosenbrock tableau aliases merged into `_TABLEAU_REGISTRY_BY_ALGORITHM` with `GenericRosenbrockWStep` constructor |

### `resolve_alias(alias)`

| # | Functionality |
|---|--------------|
| 7 | Lowercases alias before lookup |
| 8 | Returns `(step_class, tableau)` tuple for known alias |
| 9 | Raises `KeyError` for unknown alias |

### `resolve_supplied_tableau(tableau)`

| # | Functionality |
|---|--------------|
| 10 | Returns `(ERKStep, tableau)` when `isinstance(tableau, ERKTableau)` |
| 11 | Returns `(DIRKStep, tableau)` when `isinstance(tableau, DIRKTableau)` |
| 12 | Returns `(FIRKStep, tableau)` when `isinstance(tableau, FIRKTableau)` |
| 13 | Returns `(GenericRosenbrockWStep, tableau)` when `isinstance(tableau, RosenbrockTableau)` |
| 14 | Raises `TypeError` for unrecognised tableau type |

### `get_algorithm_step(precision, settings, warn_on_unused, **kwargs)`

| # | Functionality |
|---|--------------|
| 15 | Merges `settings` dict and `kwargs` (kwargs override) |
| 16 | Pops `"algorithm"` key from merged settings |
| 17 | Raises `ValueError` when `algorithm` key is missing (`None`) |
| 18 | Branch: `algorithm` is `str` -> calls `resolve_alias` |
| 19 | `resolve_alias` `KeyError` wrapped as `ValueError("Unknown algorithm ...")` |
| 20 | Branch: `algorithm` is `ButcherTableau` -> calls `resolve_supplied_tableau` |
| 21 | Branch: `algorithm` is neither str nor tableau -> raises `TypeError` |
| 22 | Injects `precision` into settings |
| 23 | Injects `tableau` into settings when `resolved_tableau is not None` |
| 24 | Does NOT inject `tableau` when `resolved_tableau is None` |
| 25 | Instantiates and returns algorithm via `algorithm_type(**algorithm_settings)` |

---

## 4. `src/cubie/integrators/step_control/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | `_CONTROLLER_REGISTRY` maps 5 keys: `fixed`, `i`, `pi`, `pid`, `gustafsson` |
| 2 | Re-exports all 5 controller classes plus `get_controller` |

### `get_controller(precision, settings, warn_on_unused, **kwargs)`

| # | Functionality |
|---|--------------|
| 3 | Merges `settings` dict and `kwargs` (kwargs override) |
| 4 | Pops `"step_controller"` from merged settings |
| 5 | Raises `ValueError` when `step_controller` is `None` (missing) |
| 6 | Lowercases `step_controller` value |
| 7 | Looks up controller class in `_CONTROLLER_REGISTRY` |
| 8 | Raises `ValueError` wrapping `KeyError` for unknown controller type |
| 9 | Injects `precision` into settings |
| 10 | Instantiates and returns controller via `controller_type(**controller_settings)` |

---

## 5. `src/cubie/integrators/loops/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Re-exports `IVPLoop` from `.ode_loop` |
| 2 | `__all__` contains exactly `["IVPLoop"]` |

---

## 6. `src/cubie/integrators/matrix_free_solvers/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Re-exports `MatrixFreeSolverConfig` from `.base_solver` |
| 2 | Re-exports `LinearSolver`, `LinearSolverConfig`, `LinearSolverCache` from `.linear_solver` |
| 3 | Re-exports `NewtonKrylov`, `NewtonKrylovConfig`, `NewtonKrylovCache` from `.newton_krylov` |

### `SolverRetCodes` (IntEnum)

| # | Functionality |
|---|--------------|
| 4 | `SUCCESS = 0` |
| 5 | `NEWTON_BACKTRACKING_NO_SUITABLE_STEP = 1` |
| 6 | `MAX_NEWTON_ITERATIONS_EXCEEDED = 2` |
| 7 | `MAX_LINEAR_ITERATIONS_EXCEEDED = 4` |

---

## 7. `src/cubie/odesystems/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Re-exports `ODEData`, `SystemSizes` from `.ODEData` |
| 2 | Re-exports `SystemValues` from `.SystemValues` |
| 3 | Re-exports `BaseODE`, `ODECache` from `.baseODE` |
| 4 | Re-exports `SymbolicODE`, `create_ODE_system`, `load_cellml_model` from `.symbolic` |
| 5 | `__all__` contains 8 names |

---

## 8. `src/cubie/odesystems/symbolic/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Star-imports from `codegen`, `codegen.dxdt`, `indexedbasemaps`, `codegen.jacobian`, `odefile`, `parsing`, `symbolicODE`, `sym_utils`, `codegen.time_derivative` |
| 2 | `__all__` contains exactly `["SymbolicODE", "create_ODE_system", "load_cellml_model"]` |

---

## 9. `src/cubie/odesystems/symbolic/parsing/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Star-imports from `auxiliary_caching`, `cellml`, `jvp_equations`, `parser` |
| 2 | `__all__` contains exactly `["load_cellml_model"]` |

---

## 10. `src/cubie/odesystems/symbolic/codegen/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Star-imports from `linear_operators`, `nonlinear_residuals`, `preconditioners`, `numba_cuda_printer` |
| 2 | `__all__` is empty list |

---

## 11. `src/cubie/outputhandling/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Re-exports `OutputCompileFlags`, `OutputConfig` from `.output_config` |
| 2 | Re-exports `OutputFunctionCache`, `OutputFunctions` from `.output_functions` |
| 3 | Re-exports `BatchInputSizes`, `BatchOutputSizes`, `OutputArrayHeights`, `SingleRunOutputSizes` from `.output_sizes` |
| 4 | Re-exports `register_metric`, `summary_metrics` from `.summarymetrics` |
| 5 | `__all__` contains 10 names |

---

## 12. `src/cubie/outputhandling/summarymetrics/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Imports `SummaryMetrics` class and `register_metric` from `.metrics` |
| 2 | Creates module-level `summary_metrics` singleton as `SummaryMetrics(precision=float32)` |
| 3 | Default precision is `float32` (comment warns: only default dtype in project) |
| 4 | Imports 18 metric modules to trigger self-registration: `mean`, `max`, `rms`, `peaks`, `std`, `min`, `max_magnitude`, `extrema`, `negative_peaks`, `mean_std_rms`, `mean_std`, `std_rms`, `dxdt_max`, `dxdt_min`, `dxdt_extrema`, `d2xdt2_max`, `d2xdt2_min`, `d2xdt2_extrema` |
| 5 | `__all__` contains `["summary_metrics", "register_metric"]` |

---

## 13. `src/cubie/memory/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Re-exports `current_cupy_stream`, `CuPySyncNumbaManager`, `CuPyAsyncNumbaManager` from `.cupy_emm` |
| 2 | Re-exports `MemoryManager` from `.mem_manager` |
| 3 | Creates module-level singleton `default_memmgr = MemoryManager()` |
| 4 | `__all__` contains 5 names |

---

## 14. `src/cubie/batchsolving/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Defines `ArrayTypes` type alias: `Optional[Union[NDArray, DeviceNDArrayBase, MappedNDArray]]` |
| 2 | Re-exports `BatchInputHandler` from `.BatchInputHandler` |
| 3 | Re-exports `BatchSolverConfig`, `ActiveOutputs` from `.BatchSolverConfig` |
| 4 | Re-exports `BatchSolverKernel` from `.BatchSolverKernel` |
| 5 | Re-exports `SystemInterface` from `.SystemInterface` |
| 6 | Re-exports `ArrayContainer`, `BaseArrayManager`, `ManagedArray` from `.arrays.BaseArrayManager` |
| 7 | Re-exports `InputArrayContainer`, `InputArrays` from `.arrays.BatchInputArrays` |
| 8 | Re-exports `OutputArrayContainer`, `OutputArrays` from `.arrays.BatchOutputArrays` |
| 9 | Re-exports `Solver`, `solve_ivp` from `.solver` |
| 10 | Re-exports `SolveResult`, `SolveSpec` from `.solveresult` |
| 11 | Re-exports `summary_metrics` from `cubie.outputhandling` |
| 12 | `__all__` contains 17 names |

---

## 15. `src/cubie/batchsolving/arrays/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | File is empty (single blank line) — no exports, no logic |

---

## 16. `src/cubie/vendored/__init__.py`

### Module-Level Logic

| # | Functionality |
|---|--------------|
| 1 | Docstring only — no exports, no logic |

---

## 17. `src/cubie/integrators/IntegratorRunSettings.py`

### `IntegratorRunSettings` (attrs class, extends `CUDAFactoryConfig`)

| # | Functionality |
|---|--------------|
| 1 | `algorithm` attribute defaults to `"euler"`, validated as `str` |
| 2 | `step_controller` attribute defaults to `"fixed"`, validated as `str` |
| 3 | `__attrs_post_init__` calls `super().__attrs_post_init__()` (inherits precision handling from `CUDAFactoryConfig`) |
| 4 | Construction with explicit `algorithm` and `step_controller` |
| 5 | Construction with defaults only (no args beyond `precision`) |
| 6 | Validator rejects non-string `algorithm` |
| 7 | Validator rejects non-string `step_controller` |

---

## 18. `src/cubie/vendored/numba_cuda_cache.py`

### `_Cache` (abstract base class)

| # | Functionality |
|---|--------------|
| 1 | `cache_path` — abstract property |
| 2 | `load_overload(sig, target_context)` — abstract method |
| 3 | `save_overload(sig, data)` — abstract method |
| 4 | `enable()` — abstract method |
| 5 | `disable()` — abstract method |
| 6 | `flush()` — abstract method |

### `Cache(_Cache)` — concrete implementation

#### `__init__(self, py_func)`

| # | Functionality |
|---|--------------|
| 7 | Stores `repr(py_func)` as `_name` |
| 8 | Stores `py_func` as `_py_func` |
| 9 | Creates `_impl` from `_impl_class(py_func)` |
| 10 | Retrieves `_cache_path` from `_impl.locator.get_cache_path()` |
| 11 | Creates `IndexDataCacheFile` with `cache_path`, `filename_base`, `source_stamp` |
| 12 | Calls `self.enable()` at end of init |

#### `__repr__`

| # | Functionality |
|---|--------------|
| 13 | Returns `"<ClassName py_func=name>"` format string |

#### `cache_path` (property)

| # | Functionality |
|---|--------------|
| 14 | Returns `self._cache_path` — table-driven forwarding check against `_impl.locator.get_cache_path()` |

#### `enable` / `disable`

| # | Functionality |
|---|--------------|
| 15 | `enable()` sets `_enabled = True` |
| 16 | `disable()` sets `_enabled = False` |

#### `flush`

| # | Functionality |
|---|--------------|
| 17 | Delegates to `_cache_file.flush()` |

#### `load_overload(self, sig, target_context)`

| # | Functionality |
|---|--------------|
| 18 | Calls `target_context.refresh()` before loading |
| 19 | Wraps `_load_overload` call in `_guard_against_spurious_io_errors` context manager |

#### `_load_overload(self, sig, target_context)`

| # | Functionality |
|---|--------------|
| 20 | Returns `None` early when `_enabled` is `False` |
| 21 | Computes index key via `_index_key(sig, target_context.codegen())` |
| 22 | Loads data from `_cache_file.load(key)` |
| 23 | Branch: `data is not None` -> calls `_impl.rebuild(target_context, data)` |
| 24 | Branch: `data is None` -> returns `None` |

#### `save_overload(self, sig, data)`

| # | Functionality |
|---|--------------|
| 25 | Wraps `_save_overload` in `_guard_against_spurious_io_errors` context manager |

#### `_save_overload(self, sig, data)`

| # | Functionality |
|---|--------------|
| 26 | Returns early when `_enabled` is `False` |
| 27 | Returns early when `_impl.check_cachable(data)` is `False` |
| 28 | Calls `_impl.locator.ensure_cache_path()` |
| 29 | Computes index key via `_index_key(sig, data.codegen)` |
| 30 | Reduces data via `_impl.reduce(data)` |
| 31 | Saves via `_cache_file.save(key, data)` |

#### `_guard_against_spurious_io_errors`

| # | Functionality |
|---|--------------|
| 32 | On Windows (`os.name == "nt"`): catches `OSError` and re-raises only if `errno != EACCES` |
| 33 | On Windows: silently swallows `EACCES` errors |
| 34 | On non-Windows: yields without any error handling |

#### `_index_key(self, sig, codegen)`

| # | Functionality |
|---|--------------|
| 35 | Extracts `co_code` from `_py_func.__code__` |
| 36 | Branch: `__closure__ is not None` -> serialises closure cell contents via `dumps` |
| 37 | Branch: `__closure__ is None` -> uses empty bytes `b""` |
| 38 | Hashes code bytes and closure bytes with SHA-256 |
| 39 | Returns tuple of `(sig, codegen.magic_tuple(), (code_hash, closure_hash))` |

### `CUDACache(Cache)` — subclass

#### `load_overload(self, sig, target_context)`

| # | Functionality |
|---|--------------|
| 40 | Wraps parent `load_overload` in `utils.numba_target_override()` context manager |

---
