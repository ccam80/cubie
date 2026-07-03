<!-- Parent: ../AGENTS.md -->

# batchsolving

## Purpose
The high-level batch-integration layer, split in two roles. **`Solver`** (and the `solve_ivp`
wrapper) is the API / instantiation / user-facing-data layer: it takes a `BaseODE`/`SymbolicODE`
system plus user inputs (dicts or arrays of initial values, parameters, drivers), routes loose
kwargs into typed settings groups, builds the input grid, launches, and returns a `SolveResult`.
**`BatchSolverKernel`** is the CUDA layer: the **only** kernel with a batch-wide view — one launch
maps each run to a `SingleIntegratorRun` device function that integrates one system in isolation —
and it drives all the lower-level machinery (the integrator, the array managers, memory). Host/
device buffer coordination lives in `arrays/`.

See `CUDAFactory` (root) for build/cache/`update`, config, and attrs conventions.

## Key Files
| File | Description |
|------|-------------|
| `solver.py` | `Solver` + `solve_ivp()` — the public API. `Solver` owns `system_interface`, `input_handler` (`BatchInputHandler`), `driver_interpolator` (`ArrayInterpolator`), and `kernel` (`BatchSolverKernel`); most getters are thin pass-throughs to `kernel`. |
| `BatchSolverKernel.py` | `BatchSolverKernel(CUDAFactory)` — the batch `@cuda.jit` kernel; maps each run to the `SingleIntegratorRun` device loop. Defines `RunParams` (frozen: duration/warmup/t0/runs + chunk metadata) and `BatchSolverCache`; owns the `InputArrays`/`OutputArrays` managers and memory-manager registration. |
| `BatchSolverConfig.py` | `BatchSolverConfig(CUDAFactoryConfig)` — holds `precision`, `loop_fn`, `compile_flags`. `ActiveOutputs(_CubieConfigBase)` — booleans for which output arrays are produced, built via `ActiveOutputs.from_compile_flags(...)`. |
| `BatchInputHandler.py` | `BatchInputHandler` (plain class) + module-level grid builders (`unique_cartesian_product`, `combinatorial_grid`, `verbatim_grid`, `generate_grid`, `combine_grids`, `extend_grid_to_array`). Converts user dicts/arrays into `(variable, run)` 2D arrays. |
| `SystemInterface.py` | `SystemInterface` — wraps the system's `SystemValues`; resolves labels↔indices, and `merge_variable_labels_and_idxs` merges `save_variables`/`summarise_variables` labels + index kwargs into final index arrays. |
| `solveresult.py` | `SolveSpec` (attrs config snapshot) and `SolveResult` — `SolveResult.from_solver(results_type=…)` assembles host arrays, legends, and metadata, applying NaN-on-error masking; exposes `as_numpy`/`as_numpy_per_summary`/`as_pandas`. |
| `writeback_watcher.py` | `WritebackWatcher` (daemon thread) + `WritebackTask`/`PendingBuffer` — polls CUDA events via `event.query()` and copies completed pinned-buffer data into host arrays for chunked async D2H writeback. |
| `_utils.py` | Docstring only — no exports (dead validators removed). |
| `__init__.py` | Defines the `ArrayTypes` alias (`Optional[Union[NDArray, DeviceNDArrayBase, MappedNDArray]]`) and re-exports the public surface. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `arrays/` | Host/device array managers (`InputArrays`/`OutputArrays`/`BaseArrayManager`/`ManagedArray`) — allocation, chunked transfers, writeback. See `arrays/AGENTS.md`. |

## For AI Agents

### Data flow
`Solver.solve()` → `input_handler(...)` builds `(n_vars, n_runs)` `inits`/`params` →
`kernel.run()` sets `RunParams`, refreshes compile settings (`loop_fn` from
`SingleIntegratorRun.device_function`), queues allocations via `InputArrays.update`/
`OutputArrays.update`, calls `memory_manager.allocate_queue(self)` (which may split into
chunks), then loops chunks launching the compiled kernel. Results flow back through
`OutputArrays` → `SolveResult.from_solver`.

### Solver: settings routing
`Solver.__init__` uses `merge_kwargs_into_settings` to split loose kwargs against the
subcomponents' `ALL_*_PARAMETERS` sets (`ALL_OUTPUT_FUNCTION_PARAMETERS`,
`ALL_MEMORY_MANAGER_PARAMETERS`, `ALL_STEP_CONTROLLER_PARAMETERS`,
`ALL_ALGORITHM_STEP_PARAMETERS`, `ALL_LOOP_SETTINGS`, `ALL_CACHE_PARAMETERS`); `strict=True`
raises `KeyError` on an unrecognised kwarg. Add a new result accessor on `kernel` and expose
it as a `Solver` property.

### Grids
`BatchInputHandler` converts user dicts/arrays into `(variable, run)` arrays via the
module-level grid builders. Two grid types: `combinatorial` (cartesian product across inputs)
and `verbatim` (inputs zipped run-for-run). **The default differs by entry point:** `solve_ivp`
defaults `grid_type="combinatorial"`; `Solver.solve`/`Solver.build_grid` default `"verbatim"`.

### Variable selection (states/observables)
`None` = use all, `[]` = explicitly none, labels + index kwargs = union.
`SystemInterface.merge_variable_labels_and_idxs` pops `save_variables`/`summarise_variables`
and writes `saved_*_indices`/`summarised_*_indices` into the settings dict in place;
summarised defaults to saved when all summarise inputs are `None`.

### The batch kernel
- One launch integrates many runs; each thread runs a `SingleIntegratorRun` device function
  over a single system in isolation. `BatchSolverKernel` is the only place with a batch-wide view.
- **`RunParams` is a frozen value object** (duration/warmup/t0/runs + chunk metadata). Frozen so
  per-chunk views and allocation updates produce independent copies via `attrs.evolve` instead of
  mutating shared state: `run_params[i]` returns a copy with that chunk's run count (last chunk
  gets the dangling remainder); `update_from_allocation` returns a copy carrying
  `num_chunks`/`chunk_length`.
- **Time is float64:** `duration`/`warmup`/`t0` are coerced to `float64` in `run()` for
  accumulation accuracy, then cast to `precision` per chunk at launch.
- **Chunking is driven by memory availability:** when the batch's arrays exceed available GPU
  memory, the memory manager splits it along the run axis; `num_chunks`/`chunk_length` come back
  on the allocation response. The run loop iterates chunks, calling `input_arrays.initialise(i)`
  (H2D) and `output_arrays.finalise(i)` (D2H/writeback).
- **Shared-memory sizing:** `limit_blocksize` halves the block size until dynamic shared memory
  fits under a 32 KiB ceiling; `shared_memory_needs_padding` adds a 4-byte skew only for single
  precision with an even element count (float64 never pads — it would misalign).

### Results
`SolveResult.from_solver(results_type=…)` builds the requested representation: `"raw"` returns a
plain dict, `"full"` the `SolveResult` itself; the instance exposes `as_numpy`,
`as_numpy_per_summary`, and `as_pandas` (lazy `pandas` import). Trajectories for runs that errored
(nonzero `status_codes`) are NaN-masked. `SolveResult.status_messages` (and the `Solver`
pass-through) decodes the per-run status word into named `CUBIE_RESULT_CODES` flags via
`cubie.result_codes.decode_status_codes`. `SolveSpec` is an attrs snapshot of the solve
configuration.

### Testing
`tests/batchsolving/` (`test_solver.py`, `test_BatchSolverKernel.py`, input-handler/result tests).
Prefer real system fixtures (`tests/system_fixtures.py`) over mocks.

## Dependencies
### Internal
- `cubie.CUDAFactory`; `cubie.integrators` (`SingleIntegratorRun`, `ArrayInterpolator`);
  `cubie.memory` (`default_memmgr`, `MemoryManager`, `ArrayRequest`/`ArrayResponse`,
  `chunk_buffer_pool`) + `cubie.buffer_registry`; `cubie.outputhandling` (`OutputCompileFlags`,
  `output_sizes`, `summary_metrics`); `cubie.odesystems` (`BaseODE`, `SymbolicODE`,
  `SystemValues`); `cubie.cubie_cache` (`CacheConfig`, `CubieCacheHandler`,
  `ALL_CACHE_PARAMETERS`); `cubie.cuda_simsafe`; `cubie._utils`.
### External
- `numba`/`numba.cuda`; `numpy`; `attrs`; optional `pandas` (lazy in `as_pandas`).
