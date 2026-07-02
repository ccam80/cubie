<!-- Parent: ../AGENTS.md -->

# integrators

## Purpose
The integration layer: it contains the **complete integrator for a single thread /
system — nothing here is batched** (batching lives in `batchsolving/`).
`SingleIntegratorRunCore` is the **composition/glue layer** (a composition root): it
instantiates the algorithm step, step controller, output functions, and CUDA loop, wires
their compiled device functions and data together, and injects the composed objects into
the children that need them; `SingleIntegratorRun` adds read-only properties over it. The
directory also provides array-driven forcing-term interpolation (`ArrayInterpolator`) and
a general scaled error-norm utility (`ScaledNorm`). `IntegratorReturnCodes` — the
kernel-level status vocabulary — is defined here.

See `CUDAFactory` (repo root) for the build/cache/`update`, buffer-registry, and
attrs-config mechanics. Subsystems (algorithms, loops, matrix_free_solvers, step_control)
each have their own `AGENTS.md`.

## Key Files
| File | Description |
|------|-------------|
| `SingleIntegratorRun.py` | `SingleIntegratorRun(SingleIntegratorRunCore)`: read-only properties exposing compiled loop artifacts, memory sizing, controller bounds, and output metadata to `BatchSolverKernel`. No `build()` override. |
| `SingleIntegratorRunCore.py` | `SingleIntegratorRunCore(CUDAFactory)`: owns `_output_functions`, `_algo_step`, `_step_controller`, `_loop`; wires them and delegates compilation to `IVPLoop` in `build()`. Defines `SingleIntegratorRunCache` (holds `single_integrator_function`). |
| `IntegratorRunSettings.py` | `IntegratorRunSettings(CUDAFactoryConfig)`: thin compile-settings holding only `algorithm` and `step_controller` names (plus inherited `precision`) — the core's own cache key. |
| `array_interpolator.py` | `ArrayInterpolator(CUDAFactory)`: builds piecewise-polynomial (spline) coefficients from sampled driver arrays and compiles `evaluate_all` (Horner evaluation of all drivers at `t`) and `evaluate_time_derivative`. Defines `ArrayInterpolatorConfig`, `InterpolatorCache`. |
| `norms.py` | `ScaledNorm(MultipleInstanceCUDAFactory)`: a general scaled error-norm utility — compiles `sum((|v_i|/tol_i)^2)/n` (mean-squared, tolerance-scaled). Currently used by the matrix-free solvers for convergence testing. Defines `ScaledNormConfig`, `ScaledNormCache`. |
| `__init__.py` | Package API re-exports (`SingleIntegratorRun`, `IVPLoop`, algorithm/solver/controller classes, `get_algorithm_step`, `get_controller`); defines `IntegratorReturnCodes(IntEnum)`. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `algorithms/` | Step-function factories + `get_algorithm_step()` (see `algorithms/AGENTS.md`). |
| `loops/` | `IVPLoop` and `ODELoopConfig` (see `loops/AGENTS.md`). |
| `matrix_free_solvers/` | Matrix-free linear (steepest-descent / minimal-residual) and Newton-Krylov solvers (see `matrix_free_solvers/AGENTS.md`). |
| `step_control/` | Fixed/adaptive step-size controllers + `get_controller()` (see `step_control/AGENTS.md`). |

## For AI Agents

### IntegratorReturnCodes — kernel status-bit meanings
`__init__.py` defines `IntegratorReturnCodes(IntEnum)`, the integer status codes
OR-combined into the returned status word: `SUCCESS=0`,
`NEWTON_BACKTRACKING_NO_SUITABLE_STEP=1`, `MAX_NEWTON_ITERATIONS_EXCEEDED=2`,
`MAX_LINEAR_ITERATIONS_EXCEEDED=4`, `STEP_TOO_SMALL=8`, `DT_EFF_EFFECTIVELY_ZERO=16`,
`MAX_LOOP_ITERS_EXCEEDED=32` (1/2/4 mirror `SolverRetCodes`; `STEP_TOO_SMALL=8` is the
controllers' reject-at-min).

### Component assembly (`SingleIntegratorRunCore.__init__`)
Order matters — each component seeds the next:
1. `OutputFunctions` first (its compile flags + summary buffer heights feed `IVPLoop`).
2. `_algo_step = get_algorithm_step(precision, settings)` — supplies
   `controller_defaults.step_controller`, seeding the controller settings before user
   overrides merge in.
3. `_step_controller = get_controller(precision, controller_settings)`.
4. `check_compatibility()` — if the algorithm is errorless but the controller is
   adaptive, the controller is **silently replaced with `FixedStepController`** and a
   `UserWarning` is issued (an errorless algorithm gives no error signal to adapt on).
   Happens before the loop is created.
5. `instantiate_loop()` — creates `IVPLoop` from the finalised sizes/flags/timing.
6. `get_child_allocators(self._loop, self._algo_step, name='algorithm')` (and the
   controller equivalent) — registers algo/controller buffers as children of the loop's
   group. **Must be re-run after every algo/controller swap** (in `update()`,
   `_switch_algos()`, `_switch_controllers()`, `build()`).

### build() delegates to IVPLoop
`SingleIntegratorRunCore.build()` defines no device function of its own. It (1) updates
`_algo_step` if the system's `evaluate_f`/`evaluate_observables`/`get_solver_helper_fn`
changed; (2) re-registers child allocators; (3) calls `self._loop.update(...)` with the
latest compiled device-function references; (4) accesses `self._loop.device_function`
(triggering the loop's build if invalid); (5) returns
`SingleIntegratorRunCache(single_integrator_function=loop_fn)` — the same object as the
loop's `loop_function`.

### Two-phase timing
`_process_loop_timing()` derives `save_every`, `summarise_every`,
`sample_summaries_every`, and the `save_*`/`summarise_regularly` flags from user intent.
If `summarise_every` is omitted, `is_duration_dependent=True` and
`set_summary_timing_from_duration()` must be called later (done by `BatchSolverKernel`
before each solve); this triggers a recompile on first use (warned).

### Hot-swap
A new `"algorithm"`/`"step_controller"` in `update()` routes through
`_switch_algos()`/`_switch_controllers()`, which call `buffer_registry.reset()`, rebuild
the sub-component from the old settings as a base, and propagate defaults into
`updates_dict`. Never call these directly — go through `update()`. Because the swap calls
`buffer_registry.reset()`, any cached allocator references become stale.

### Testing
Top-level files are exercised via `tests/integrators/` integration tests and
`tests/batchsolving/` end-to-end tests. `ArrayInterpolator` has dedicated tests
(`tests/integrators/test_array_interpolator.py`); `ScaledNorm` is tested with the
matrix-free solvers.

## Dependencies
### Internal
- `cubie.CUDAFactory`; `cubie.buffer_registry`; `cubie._utils` (`PrecisionDType`,
  `unpack_dict_values`, `build_config`, tolerance validators, `tol_converter`);
  `cubie.cuda_simsafe`; `cubie.odesystems.ODEData` (`SystemSizes`), `baseODE`
  (TYPE_CHECKING); `cubie.outputhandling` (`OutputFunctions`, `OutputCompileFlags`); the
  four integrators subpackages.
### External
- `numba` (`cuda.jit`, `int32`, `from_dtype`); `numpy`; `attrs`; `math` (in
  `ArrayInterpolator`).
