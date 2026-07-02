<!-- Parent: ../AGENTS.md -->

# matrix_free_solvers

## Purpose
CUDA device-function factories for the inner solvers of implicit methods: a
Jacobian-free preconditioned **linear** solver (steepest-descent /
minimal-residual) and a damped-backtracking **Newton–Krylov** nonlinear solver
that calls the linear solver for each correction. The implicit algorithm steps
(`generic_dirk`, `generic_firk`, `backwards_euler`, `crank_nicolson`,
Rosenbrock-W) invoke these once per implicit stage. No Jacobian is materialised —
the caller passes device callbacks that apply the operator / preconditioner /
residual, and the solver iterates using only those plus preallocated scratch.

Both are `MultipleInstanceCUDAFactory` subclasses; the build/cache/`update`,
buffer-registry, attrs-config, and cache-invalidation mechanics are common to all
factories and live with `CUDAFactory` (repo root). This file documents only what
is specific to the solvers.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Re-exports factories/configs/caches; defines `SolverRetCodes(IntEnum)`. |
| `base_solver.py` | `MatrixFreeSolver` / `MatrixFreeSolverConfig` base — holds the norm device function and the shared `n` / `max_iters` / tolerance plumbing. |
| `linear_solver.py` | `LinearSolver` — matrix-free preconditioned steepest-descent / minimal-residual linear solve (cached and non-cached variants). |
| `newton_krylov.py` | `NewtonKrylov` — damped backtracking Newton iteration that calls a `LinearSolver` for each correction. |

## For AI Agents

**Class factories, not free functions.** The public surface is the classes
`LinearSolver` and `NewtonKrylov`; there are no `linear_solver_factory` /
`newton_krylov_solver_factory` functions. Get the compiled callable from
`.device_function`.

### Compiled device-function signatures (the caller contract)
- `LinearSolver` (non-cached): `linear_solver(state, parameters, drivers,
  base_state, t, h, a_ij, rhs, x, shared, persistent_local, krylov_iters_out) ->
  int32`. The cached variant inserts `cached_aux` after `base_state`. `rhs` enters
  as the RHS and is overwritten with the residual; `x` enters as the initial guess
  and is overwritten with the solution; `krylov_iters_out` is a length-1 int32 array.
- `NewtonKrylov`: `newton_krylov_solver(stage_increment, parameters, drivers, t, h,
  a_ij, base_state, shared_scratch, persistent_scratch, counters) -> int32`.
  `stage_increment` is updated in place. `counters` is a length-2 int32 array:
  `[0]` = Newton iterations, `[1]` = total Krylov iterations.

### Caller-supplied callbacks (set via config/`update`)
- `operator_apply` — applies `F @ v`; sig `(state, parameters, drivers, base_state,
  t, h, a_ij, v, out)` (cached variant inserts `cached_aux` after `drivers`).
- `preconditioner` (optional; `None` → search direction is `rhs`); sig
  `(state, parameters, drivers, base_state, t, h, a_ij, rhs, preconditioned_vec, temp)`.
- `residual_function` (Newton); sig `(stage_increment, parameters, drivers, t, h,
  a_ij, base_state, residual_out)`.
- `linear_solver_function` (Newton) — the inner `LinearSolver.device_function`.
  `NewtonKrylov` owns a child `LinearSolver`: its `update` forwards `krylov_`-prefixed
  params to the child and re-injects the recompiled device function.

### Registered buffers (length `n` unless noted)
- `LinearSolver`: `preconditioned_vec`, `temp`.
- `NewtonKrylov`: `delta`, `residual`, `residual_temp`, `stage_base_bt`, and
  `krylov_iters_local` (length 1, int32). It carves child buffers for its inner
  `LinearSolver` via `buffer_registry.get_child_allocators(self, self.linear_solver)`.

### Status codes & convergence
- `SolverRetCodes`: `SUCCESS=0`, `NEWTON_BACKTRACKING_NO_SUITABLE_STEP=1`,
  `MAX_NEWTON_ITERATIONS_EXCEEDED=2`, `MAX_LINEAR_ITERATIONS_EXCEEDED=4`.
  `newton_krylov_solver` OR-combines these into a **low-bits** status word — it does
  NOT pack the iteration count into high bits (counts go to `counters`). Callers OR
  this word into their own step status.
- Convergence is tested with the solver's **norm device function** (a `Norm`
  CUDAFactory; see `cubie.integrators.norms`). The per-solver tolerances are the
  prefixed params `krylov_atol`/`krylov_rtol` (linear) and `newton_atol`/`newton_rtol`
  (Newton), read via the `*_atol`/`*_rtol` properties (`NewtonKrylov.krylov_*`
  delegate to the inner `LinearSolver`).

### Solver-specific gotchas
- **Warp-coherent loops.** Iterative loops exit on warp votes (`all_sync`/`any_sync`
  from `cuda_simsafe`) so every active lane agrees before breaking; `selp` gives
  branchless commits. Don't add un-voted data-dependent `break`/early-return — it
  breaks lane lockstep.
- **`newton_max_backtracks` is `+1` in `build()`** ("off by 1"), so the configured
  value is the number of *additional* damping attempts.

### Testing
Solver behaviour is exercised through the implicit algorithm steps under
`tests/integrators/algorithms/`. **Critical:** these device functions are mirrored
(with added logging) in `tests/integrators/algorithms/instrumented/matrix_free_solvers.py`
(`InstrumentedLinearSolver` / `InstrumentedNewtonKrylov`, overriding only `build()`).
Any change to a device function's algorithm, signature, buffers, or status logic
must be mirrored there. Run e.g.
`pytest tests/integrators/algorithms -k "newton or krylov or implicit"`.

## Dependencies
### Internal
- `cubie.CUDAFactory` — `MultipleInstanceCUDAFactory` + config/cache bases.
- `cubie.integrators.norms` — convergence norm device function.
- `cubie.buffer_registry` — scratch buffer allocators.
- `cubie.cuda_simsafe` — `activemask`, `all_sync`, `any_sync`, `selp`.
- `cubie._utils` — `build_config`, device/precision validators, `PrecisionDType`.
- Consumed by `cubie.integrators.algorithms.*` (implicit steps).
### External
- `numba.cuda`, `attrs`, `numpy`.
