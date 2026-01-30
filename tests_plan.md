# CuBIE Test Sweep — Master Inventory

## Rules

1. **Phase 1 inventories functionality from source only.** No reference
   to existing tests. Existing tests are irrelevant until Phase 3.
2. Simple pass-through properties → table-driven forwarding checks
   against child object properties directly.
3. Calculated properties → dedicated tests.
4. Every conditional branch in source → inventory item.

## Legend

- [ ] = inventory not started
- [~] = inventory in progress
- [x] = inventory complete

---

## Completed Inventories

### `integrators/SingleIntegratorRunCore.py` [x]

#### `__init__` — Construction

| # | Functionality |
|---|--------------|
| 1 | Construction succeeds with all-`None` optional args |
| 2 | Construction succeeds with explicit values (no fallback) |
| 3 | Algorithm controller defaults flow through when no user `step_control_settings` |
| 4 | User `step_control_settings` override algorithm defaults |
| 5 | Precision popped from `output_settings`; system's precision used instead |
| 6 | `dt` from `step_control_settings` threaded into `algorithm_settings` |

#### `_process_loop_timing` — Timing Derivation

| # | Functionality |
|---|--------------|
| 7 | `_user_timing` dict updated from incoming settings |
| 8 | `save_last = True` when time-domain outputs requested + no `save_every` |
| 9 | `is_duration_dependent = True` when summary outputs requested + no `summarise_every` |
| 10 | `sample_summaries_every` auto-derived as `summarise_every / 10` when not provided |
| 11 | `summarise_every` and `sample_summaries_every` forced to `None` when no summary outputs |
| 12 | `save_regularly` / `summarise_regularly` booleans computed correctly |
| 13 | Derived values forwarded to loop and output_functions |
| 14 | Warning emitted (from Solver level) when `is_duration_dependent` is True |

#### `set_summary_timing_from_duration`

| # | Functionality |
|---|--------------|
| 15 | Duration-dependent path: `summarise_every = duration` propagated to loop |
| 16 | Duration-dependent path: `sample_summaries_every = duration / 100` propagated to loop and output_functions |
| 17 | Non-dependent path: existing values unchanged |

#### `n_error` Property

| # | Functionality |
|---|--------------|
| 18 | Returns `system.sizes.states` for adaptive algorithm (parametrized) |
| 19 | Returns 0 for non-adaptive algorithm (parametrized) |

#### `check_compatibility`

| # | Functionality |
|---|--------------|
| 20 | Incompatible pair (non-adaptive algo + adaptive controller) triggers replacement with fixed controller |
| 21 | Replacement uses `dt0` from original adaptive controller |
| 22 | Warning emitted naming algorithm, controller, mentioning "error estimate" and "fixed" |
| 23 | `None` fallback paths for `algorithm_name`, `controller_name`, `precision` (called from `update()`) |
| 24 | Replacement controller created with `warn_on_unused=False` |
| 25 | `is_controller_fixed` set on `_algo_step` based on final controller state |
| 26 | Compatible: adaptive algo + adaptive controller passes without warning |
| 27 | Compatible: non-adaptive algo + fixed controller passes without warning |
| 28 | Compatible: adaptive algo + fixed controller passes without warning |

#### `instantiate_loop`

| # | Functionality |
|---|--------------|
| 29 | All named parameters forwarded to IVPLoop constructor |
| 30 | `n_counters = 4` when `compile_flags.save_counters` is True |
| 31 | `n_counters = 0` when `compile_flags.save_counters` is False |
| 32 | `evaluate_driver_at_t` not overwritten if already in loop_settings |
| 33 | `evaluate_driver_at_t` set from argument if absent from loop_settings |
| 34 | `n_error` forwarded from `self.n_error` |

#### `update` — Parameter Routing

| # | Functionality |
|---|--------------|
| 35 | `updates_dict` and `kwargs` merged |
| 36 | Empty dict → early return with empty set |
| 37 | Nested dicts flattened via `unpack_dict_values` |
| 38 | Forwarded to `_system.update(silent=True)`, recognised keys captured |
| 39 | `n` injected from `_system.sizes.states` |
| 40 | Forwarded to `_output_functions.update(silent=True)`, `buffer_sizes_dict` captured if recognised |
| 41 | `_switch_algos` called, may swap algorithm |
| 42 | Forwarded to `_algo_step.update(silent=True)`, `threads_per_step` captured |
| 43 | `algorithm_order` injected |
| 44 | `_switch_controllers` called, may swap controller |
| 45 | Forwarded to `_step_controller.update(silent=True)`, `is_adaptive`/`dt_min`/`dt_max`/`dt0` captured |
| 46 | Algo and controller buffers re-registered with loop |
| 47 | Forwarded to `_loop.update(silent=True)` |
| 48 | `_process_loop_timing` called with updates |
| 49 | `update_compile_settings` called |
| 50 | Unrecognised params raise `KeyError` when `silent=False` |
| 51 | No `KeyError` when `silent=True` |
| 52 | Cache invalidated if any keys recognised |
| 53 | `check_compatibility` called after updates |
| 54 | Unpacked dict keys included in returned recognised set |
| 55 | Downstream components see upstream changes (system → algo → controller → loop ordering) |

#### `_switch_algos`

| # | Functionality |
|---|--------------|
| 56 | No-op when `"algorithm"` not in updates_dict (returns empty set) |
| 57 | Buffer registry reset on any algorithm change attempt |
| 58 | New algorithm step created when name differs from current |
| 59 | Old settings carried forward to new algorithm |
| 60 | `compile_settings.algorithm` updated |
| 61 | Algorithm name normalised to lowercase |
| 62 | Algorithm controller defaults merged into updates_dict (only for missing keys) |
| 63 | `algorithm_order` injected |

#### `_switch_controllers`

| # | Functionality |
|---|--------------|
| 64 | No-op when `"step_controller"` not in updates_dict (returns empty set) |
| 65 | Buffer registry reset |
| 66 | Controller name normalised to lowercase |
| 67 | New controller created when name differs, old settings carried forward |
| 68 | `algorithm_order` injected from updates_dict if present |
| 69 | `algorithm_order` falls back to `self._algo_step.order` if not in updates_dict |
| 70 | `compile_settings.step_controller` updated |

#### `build`

| # | Functionality |
|---|--------------|
| 71 | Fetches `evaluate_f`, `evaluate_observables`, `get_solver_helper` from system |
| 72 | Conditionally updates algo_step only when functions differ from current |
| 73 | Does NOT update algo_step when functions unchanged |
| 74 | Changed `evaluate_f` propagates through to algo_step |
| 75 | Algo and controller buffers re-registered with loop |
| 76 | Compiled functions dict assembled from output_functions, step_controller, algo_step |
| 77 | Loop updated with compiled functions dict |
| 78 | Returns `SingleIntegratorRunCache` wrapping loop's `device_function` |

#### Computed Properties

| # | Functionality |
|---|--------------|
| 79 | `time_domain_outputs_requested` forwards to `_output_functions.has_time_domain_outputs` |
| 80 | `summary_outputs_requested` forwards to `_output_functions.has_summary_outputs` |
| 81 | `has_time_domain_outputs`: True when types requested + `save_every` set |
| 82 | `has_time_domain_outputs`: True when types requested + `save_last` True |
| 83 | `has_time_domain_outputs`: False when types requested + neither timing set |
| 84 | `has_time_domain_outputs`: False when no types requested + timing set |
| 85 | `has_summary_outputs`: True when types requested + `summarise_every` set |
| 86 | `has_summary_outputs`: False when types requested + no `summarise_every` |
| 87 | `has_summary_outputs`: False when no types requested + `summarise_every` set |
| 88 | `has_summary_outputs`: False when no types + no `summarise_every` |

---

### `integrators/SingleIntegratorRun.py` [x]

#### Forwarding Properties (table-driven)

All compared against child object property directly.

| # | Property | Delegates to |
|---|----------|-------------|
| 1 | `algorithm` | `compile_settings.algorithm` |
| 2 | `step_controller` | `compile_settings.step_controller` |
| 3 | `shared_memory_elements` | `_loop.shared_buffer_size` |
| 4 | `shared_memory_bytes` | **dedicated test** (see below) |
| 5 | `local_memory_elements` | `_loop.local_buffer_size` |
| 6 | `persistent_local_elements` | `_loop.persistent_local_buffer_size` |
| 7 | `dt0` | `_step_controller.dt0` |
| 8 | `dt_min` | `_step_controller.dt_min` |
| 9 | `dt_max` | `_step_controller.dt_max` |
| 10 | `is_adaptive` | `_step_controller.is_adaptive` |
| 11 | `system` | `_system` |
| 12 | `system_sizes` | `_system.sizes` |
| 13 | `save_summaries_func` | `save_summary_metrics_func` (chain) |
| 14 | `evaluate_f` | `_algo_step.evaluate_f` |
| 15 | `save_every` | `_loop.save_every` |
| 16 | `summarise_every` | `_loop.summarise_every` |
| 17 | `sample_summaries_every` | `_loop.sample_summaries_every` |
| 18 | `save_last` | `_loop.compile_settings.save_last` |
| 19 | `compile_flags` | `_loop.compile_flags` |
| 20 | `save_state_fn` | `_loop.save_state_fn` |
| 21 | `update_summaries_fn` | `_loop.update_summaries_fn` |
| 22 | `save_summaries_fn` | `_loop.save_summaries_fn` |
| 23 | `atol` | `_step_controller.atol` (with hasattr guard) |
| 24 | `rtol` | `_step_controller.rtol` (with hasattr guard) |
| 25 | `dt` | `_step_controller.dt` (with hasattr guard) |
| 26 | `threads_per_step` | `_algo_step.threads_per_step` |
| 27 | `save_state_func` | `_output_functions.save_state_func` |
| 28 | `update_summaries_func` | `_output_functions.update_summaries_func` |
| 29 | `save_summary_metrics_func` | `_output_functions.save_summary_metrics_func` |
| 30 | `output_types` | `_output_functions.output_types` |
| 31 | `output_compile_flags` | `_output_functions.compile_flags` |
| 32 | `save_time` | `_output_functions.save_time` |
| 33 | `saved_state_indices` | `_output_functions.saved_state_indices` |
| 34 | `saved_observable_indices` | `_output_functions.saved_observable_indices` |
| 35 | `summarised_state_indices` | `_output_functions.summarised_state_indices` |
| 36 | `summarised_observable_indices` | `_output_functions.summarised_observable_indices` |
| 37 | `output_array_heights` | `_output_functions.output_array_heights` |
| 38 | `summary_legend_per_variable` | `_output_functions.summary_legend_per_variable` |
| 39 | `summary_unit_modifications` | `_output_functions.summary_unit_modifications` |

#### `shared_memory_bytes` — Dedicated Test

| # | Functionality |
|---|--------------|
| 40 | Returns `elements * 4` for float32 precision |
| 41 | Returns `elements * 8` for float64 precision |

#### `output_length` — Calculation

| # | Functionality |
|---|--------------|
| 42 | Always includes 1 initial sample |
| 43 | Adds `floor(duration / save_every)` regular samples when `save_every` set |
| 44 | Adds 1 final sample when `save_last` is True |
| 45 | Returns 1 when `save_every` is None and `save_last` is False |
| 46 | Returns 2 when `save_every` is None and `save_last` is True |

#### `summaries_length` — Calculation

| # | Functionality |
|---|--------------|
| 47 | Returns `int(duration / summarise_every)` when set |
| 48 | Returns 0 when `summarise_every` is None |

---

## Remaining Files — Inventory Not Started

### Core Infrastructure

| # | Source File | Inventory |
|---|------------|-----------|
| 1 | `_utils.py` | [ ] |
| 2 | `cuda_simsafe.py` | [ ] |
| 3 | `buffer_registry.py` | [ ] |
| 4 | `CUDAFactory.py` | [ ] |
| 5 | `time_logger.py` | [ ] |
| 6 | `cubie_cache.py` | [ ] |
| 7 | `vendored/numba_cuda_cache.py` | [ ] |
| 8 | `__init__.py` | [ ] |

### ODE Systems

| # | Source File | Inventory |
|---|------------|-----------|
| 9 | `odesystems/ODEData.py` | [ ] |
| 10 | `odesystems/SystemValues.py` | [ ] |
| 11 | `odesystems/baseODE.py` | [ ] |
| 12 | `odesystems/__init__.py` | [ ] |

### Symbolic ODE & Parsing

| # | Source File | Inventory |
|---|------------|-----------|
| 13 | `odesystems/symbolic/symbolicODE.py` | [ ] |
| 14 | `odesystems/symbolic/sym_utils.py` | [ ] |
| 15 | `odesystems/symbolic/indexedbasemaps.py` | [ ] |
| 16 | `odesystems/symbolic/odefile.py` | [ ] |
| 17 | `odesystems/symbolic/__init__.py` | [ ] |
| 18 | `odesystems/symbolic/parsing/parser.py` | [ ] |
| 19 | `odesystems/symbolic/parsing/auxiliary_caching.py` | [ ] |
| 20 | `odesystems/symbolic/parsing/jvp_equations.py` | [ ] |
| 21 | `odesystems/symbolic/parsing/cellml.py` | [ ] |
| 22 | `odesystems/symbolic/parsing/cellml_cache.py` | [ ] |
| 23 | `odesystems/symbolic/parsing/__init__.py` | [ ] |

### Codegen

| # | Source File | Inventory |
|---|------------|-----------|
| 24 | `codegen/numba_cuda_printer.py` | [ ] |
| 25 | `codegen/dxdt.py` | [ ] |
| 26 | `codegen/time_derivative.py` | [ ] |
| 27 | `codegen/jacobian.py` | [ ] |
| 28 | `codegen/linear_operators.py` | [ ] |
| 29 | `codegen/preconditioners.py` | [ ] |
| 30 | `codegen/nonlinear_residuals.py` | [ ] |
| 31 | `codegen/_stage_utils.py` | [ ] |
| 32 | `codegen/__init__.py` | [ ] |

### Integrators — Algorithms

| # | Source File | Inventory |
|---|------------|-----------|
| 33 | `algorithms/base_algorithm_step.py` | [ ] |
| 34 | `algorithms/ode_explicitstep.py` | [ ] |
| 35 | `algorithms/ode_implicitstep.py` | [ ] |
| 36 | `algorithms/explicit_euler.py` | [ ] |
| 37 | `algorithms/generic_erk.py` | [ ] |
| 38 | `algorithms/generic_erk_tableaus.py` | [ ] |
| 39 | `algorithms/generic_dirk.py` | [ ] |
| 40 | `algorithms/generic_dirk_tableaus.py` | [ ] |
| 41 | `algorithms/generic_firk.py` | [ ] |
| 42 | `algorithms/generic_firk_tableaus.py` | [ ] |
| 43 | `algorithms/generic_rosenbrock_w.py` | [ ] |
| 44 | `algorithms/generic_rosenbrockw_tableaus.py` | [ ] |
| 45 | `algorithms/backwards_euler.py` | [ ] |
| 46 | `algorithms/backwards_euler_predict_correct.py` | [ ] |
| 47 | `algorithms/crank_nicolson.py` | [ ] |
| 48 | `algorithms/__init__.py` | [ ] |

### Integrators — Step Controllers

| # | Source File | Inventory |
|---|------------|-----------|
| 49 | `step_control/base_step_controller.py` | [ ] |
| 50 | `step_control/fixed_step_controller.py` | [ ] |
| 51 | `step_control/adaptive_step_controller.py` | [ ] |
| 52 | `step_control/adaptive_I_controller.py` | [ ] |
| 53 | `step_control/adaptive_PI_controller.py` | [ ] |
| 54 | `step_control/adaptive_PID_controller.py` | [ ] |
| 55 | `step_control/gustafsson_controller.py` | [ ] |
| 56 | `step_control/__init__.py` | [ ] |

### Integrators — Solvers & Loops

| # | Source File | Inventory |
|---|------------|-----------|
| 57 | `matrix_free_solvers/base_solver.py` | [ ] |
| 58 | `matrix_free_solvers/linear_solver.py` | [ ] |
| 59 | `matrix_free_solvers/newton_krylov.py` | [ ] |
| 60 | `matrix_free_solvers/__init__.py` | [ ] |
| 61 | `norms.py` | [ ] |
| 62 | `array_interpolator.py` | [ ] |
| 63 | `loops/ode_loop_config.py` | [ ] |
| 64 | `loops/ode_loop.py` | [ ] |
| 65 | `loops/__init__.py` | [ ] |
| 66 | `IntegratorRunSettings.py` | [ ] |
| 67 | `integrators/__init__.py` | [ ] |

### Output Handling

| # | Source File | Inventory |
|---|------------|-----------|
| 68 | `outputhandling/output_config.py` | [ ] |
| 69 | `outputhandling/output_sizes.py` | [ ] |
| 70 | `outputhandling/output_functions.py` | [ ] |
| 71 | `outputhandling/save_state.py` | [ ] |
| 72 | `outputhandling/save_summaries.py` | [ ] |
| 73 | `outputhandling/update_summaries.py` | [ ] |
| 74 | `outputhandling/__init__.py` | [ ] |

### Summary Metrics

| # | Source File | Inventory |
|---|------------|-----------|
| 75 | `summarymetrics/metrics.py` | [ ] |
| 76 | `summarymetrics/__init__.py` | [ ] |
| 77–96 | `summarymetrics/{mean,max,min,std,rms,peaks,negative_peaks,extrema,max_magnitude,mean_std,mean_std_rms,std_rms,dxdt_max,dxdt_min,dxdt_extrema,d2xdt2_max,d2xdt2_min,d2xdt2_extrema,final_state,dxdt_final_state}.py` | [ ] |
| 97 | `summarymetrics/d2xdt2_final_state.py` | [ ] |

### Memory Management

| # | Source File | Inventory |
|---|------------|-----------|
| 98 | `memory/mem_manager.py` | [ ] |
| 99 | `memory/stream_groups.py` | [ ] |
| 100 | `memory/cupy_emm.py` | [ ] |
| 101 | `memory/chunk_buffer_pool.py` | [ ] |
| 102 | `memory/array_requests.py` | [ ] |
| 103 | `memory/__init__.py` | [ ] |

### Batch Solving

| # | Source File | Inventory |
|---|------------|-----------|
| 104 | `batchsolving/_utils.py` | [ ] |
| 105 | `batchsolving/SystemInterface.py` | [ ] |
| 106 | `batchsolving/BatchInputHandler.py` | [ ] |
| 107 | `batchsolving/BatchSolverConfig.py` | [ ] |
| 108 | `batchsolving/BatchSolverKernel.py` | [ ] |
| 109 | `batchsolving/solveresult.py` | [ ] |
| 110 | `batchsolving/solver.py` | [ ] |
| 111 | `batchsolving/writeback_watcher.py` | [ ] |
| 112 | `batchsolving/arrays/BaseArrayManager.py` | [ ] |
| 113 | `batchsolving/arrays/BatchInputArrays.py` | [ ] |
| 114 | `batchsolving/arrays/BatchOutputArrays.py` | [ ] |
| 115 | `batchsolving/arrays/__init__.py` | [ ] |
| 116 | `batchsolving/__init__.py` | [ ] |

### GUI (excluded — Qt dependency)

| # | Source File | Inventory |
|---|------------|-----------|
| — | `gui/__init__.py` | skip |
| — | `gui/constants_editor.py` | skip |
| — | `gui/states_editor.py` | skip |

---

## Removed Vestiges (during walkthrough)

- `SingleIntegratorRun.algorithm_key` — identical to `algorithm`, removed.
  Consumer `BatchSolverKernel.algorithm` updated to use `.algorithm`.
- `SingleIntegratorRun.compiled_loop_function` — alias for `device_function`, removed.
  Consumer `BatchSolverKernel` updated to use `.device_function`.
- `SingleIntegratorRun.threads_per_loop` — alias for `threads_per_step`, removed.
  Consumer `BatchSolverKernel` updated to use `.threads_per_step`.

## Bugs Fixed (during walkthrough)

- `SingleIntegratorRunCore._switch_algos`: `return set("algorithm")` →
  `return {"algorithm"}` (was returning set of characters).
- `SingleIntegratorRunCore._switch_controllers`: `return set("step_controller")` →
  `return {"step_controller"}` (same bug).
