<!-- Parent: ../AGENTS.md -->

# symbolic

## Purpose
SymPy-driven CUDA codegen pipeline that turns symbolic ODE definitions into JIT-compiled
Numba-CUDA device functions. This top level holds the user-facing system class (`SymbolicODE`),
the disk-backed source cache (`ODEFile`/`GENERATED_DIR`), the symbol-to-device-index maps
(`IndexedBaseMap`/`IndexedBases`), and shared SymPy utilities (CSE, topological sort, hashing,
assignment pruning). Equation parsing lives in `parsing/`; the CUDA source emitters live in
`codegen/`. `SymbolicODE` orchestrates both: parse via `parsing.parse_input`, generate
`dxdt`/`observables`/solver-helper factories via `codegen`, write them to a per-system module on
disk, and reload the compiled factories. As the sole concrete `BaseODE` subclass it is the main
entry point for defining systems — users construct one via `create_ODE_system()` (string / SymPy
/ callable) or `load_cellml_model()`.

See `CUDAFactory` (root) for the build/cache/`update` contract, closure capture, config, and
attrs conventions; `BaseODE` (parent, `../AGENTS.md`) for `ODECache`/`config_hash`/`set_constants`.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Star-imports `codegen`, `parsing`, `indexedbasemaps`, `odefile`, `symbolicODE`, `sym_utils`; declares `__all__ = ["SymbolicODE", "create_ODE_system", "load_cellml_model"]`. |
| `symbolicODE.py` | `SymbolicODE(BaseODE)` plus `create_ODE_system()`. Owns parsing, codegen caching, constant/parameter conversion, units, optional Qt GUIs, and `get_solver_helper()` which dispatches to every `codegen` emitter. |
| `odefile.py` | `ODEFile` disk cache and `GENERATED_DIR` (`./generated/`). Writes generated factory source to `generated/<name>/<name>.py`, hash-guards staleness, checks per-function caching, and imports factories via `importlib`. |
| `indexedbasemaps.py` | `IndexedBaseMap` (named scalar symbols → fixed-size `sympy.IndexedBase`) and `IndexedBases` (bundle of state/parameter/constant/observable/driver/dxdt maps). Provides `from_user_inputs`, constant↔parameter conversion, units, ref/index/symbol maps. |
| `sym_utils.py` | Stateless SymPy helpers: `topological_sort` (Kahn), `cse_and_stack`, `hash_system_definition` (SHA-256, order-independent), `render_constant_assignments`, `prune_unused_assignments`. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `codegen/` | CUDA source emitters for dxdt, observables, Jacobian/JVP, linear operators, preconditioners, residuals, time derivatives, and the Numba-CUDA SymPy printer (see `codegen/AGENTS.md`). |
| `parsing/` | Converts string / SymPy / callable / CellML input into `ParsedEquations` + `IndexedBases`, plus `JVPEquations` and auxiliary-caching heuristics; one normalised front end classifies input as explicit or DAE and routes the latter through `structural/` (see `parsing/AGENTS.md`). |
| `structural/` | MTK-style structural simplification and tearing (alias elimination, Pantelides index reduction, dummy derivatives, Carpanzano/Modia tearing); enabled automatically for DAE-shaped input or forced via `create_ODE_system(..., simplify=True)` (see `structural/AGENTS.md`). |

## For AI Agents

### get_solver_helper — the single codegen dispatch point
`build()` compiles only `dxdt` and `observables`; **every other device function comes from
`get_solver_helper(func_type, ...)`**. Supported `func_type`: `linear_operator`,
`linear_operator_cached`, `neumann_preconditioner`, `neumann_preconditioner_cached`,
`stage_residual`, `n_stage_residual`, `n_stage_linear_operator`,
`n_stage_neumann_preconditioner`, `prepare_jac`, `calculate_cached_jvp`, `time_derivative_rhs`,
and the non-codegen `cached_aux_count`. Adding a helper means a branch here **and** a generator
in `codegen/`. The `_HELPERS_NEEDING_PRECONDITIONER_KWARGS` frozenset selects which helpers get
`beta`/`gamma`/`order` factory kwargs. `n_stage_*` helpers suffix the factory name with the
stage count (`f"{func_type}_{len(stage_nodes)}"`), so each stage count caches separately. The
cached helpers (`linear_operator_cached`, `neumann_preconditioner_cached`, `prepare_jac`,
`calculate_cached_jvp`, `cached_aux_count`) are requested by `GenericRosenbrockWStep` and run
every step; how many auxiliaries actually get precomputed is set by the caching planner's
thresholds (see `parsing/` and `codegen/AGENTS.md`).

### build() and system identity
`build()` compiles `dxdt`+`observables` into the `ODECache`, first recomputing the system hash —
swapping `self.gen_file` to a fresh `ODEFile` if constants↔parameters changed since construction.
The identity is `fn_hash` from `hash_system_definition`: **equations + constant *labels* +
observable labels — NOT parameter labels and NOT constant *values*** (constants and parameters
together cover all non-state LHS symbols, so a constant↔parameter flip changes the hash and
forces re-codegen; a constant *value* change only forces a rebuild). The hash is order-independent
(sorted by LHS name), so string- and SymPy-input paths hit the same cache.

### Constant/parameter conversion
`make_parameter`/`make_constant` update both `self.indices`
(`constant_to_parameter`/`parameter_to_constant`) and `compile_settings` (`remove_entry`/
`add_entry`), then call `_invalidate_cache()` — keep both sides in sync. `SymbolicODE` overrides
`set_constants()` to update the index map (`self.indices.update_constants(...)`) before delegating
to `BaseODE.set_constants`; it does not override `update`.

### Codegen cache gotchas (`ODEFile`)
- `function_is_cached` parses the generated file textually: it needs a top-level `def <name>(`
  with a `return` one indent level in. A generator that emits a factory without a `return` is
  treated as uncached forever.
- `GENERATED_DIR` is `Path(getcwd())/"generated"` computed at import, so output lands relative to
  the process CWD, not the package.

### IndexedBaseMap rebuilds on structural change
`push`/`pop` rebuild the `sympy.IndexedBase` (shape change), so any `ref_map` array reference
captured before a conversion goes stale — re-read it after `make_parameter`/`make_constant`.

### Qt GUIs are lazily imported
`constants_gui`/`states_gui` import the `cubie.gui` editors *inside* the method (Qt is optional).
Never import Qt or `cubie.gui` at module top level.

### Testing
`tests/odesystems/symbolic/` (`test_symbolicode.py`, `test_odefile.py`, `test_indexedbasemaps.py`,
`test_sym_utils.py`, `test_solver_helpers.py`); codegen tests under `.../codegen/`. Prefer real
`SymbolicODE` fixtures (`conftest.py`). See root for CUDASIM/real-CUDA commands.

## Dependencies
### Internal
- `cubie.odesystems.baseODE` (`BaseODE`, `ODECache`); `cubie.odesystems.symbolic.codegen` (all
  source emitters); `cubie.odesystems.symbolic.parsing` (`parse_input`, `IndexedBases`,
  `ParsedEquations`, `JVPEquations`); `cubie.array_interpolator.ArrayInterpolator`
  (driver-array setup); `cubie._utils` (`PrecisionDType`), `cubie.time_logger.default_timelogger`,
  `cubie.cuda_simsafe` (in the generated module header), `cubie.gui.*` (lazy, optional).
### External
- `sympy`; `numpy` (`float32`, `ndarray`); `numba`/`numba.cuda` (generated header + precision types).
