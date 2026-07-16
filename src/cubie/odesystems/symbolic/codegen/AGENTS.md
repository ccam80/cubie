<!-- Parent: ../AGENTS.md -->

# codegen

## Purpose
IR-to-CUDA **source generators**. Each public `generate_*` function takes parsed equations
(`ParsedEquations`/`JVPEquations`) plus an index map (`IndexedBases`), converts them once to
the `engine/` IR through `engine.adapter.system_ir`, and returns a
**Python source string** defining a `*_factory(constants, precision, ...)` function. Nothing here
compiles CUDA: `symbolicODE.get_solver_helper()` writes the string to disk via `ODEFile`, imports
it, and calls the factory to JIT a Numba CUDA device function. This module is the source of the
explicit `dxdt`/observables RHS, the analytic Jacobian and Jacobian-vector products (JVP), and the
three matrix-free device callbacks the implicit solvers need:

- **Linear operator** — `out = β·M·v − γ·a_ij·h·(J·v)`
- **Nonlinear residual** — `out = β·M·u − γ·h·f(base_state + a_ij·u)`
- **Neumann preconditioner** — truncated series approximating `(β·I − γ·a_ij·h·J)⁻¹·v` (Horner form)

These three are algebraically locked together: the operator is the exact Jacobian of the residual
with respect to the stage increment `u`. The `a_ij` placement differs between them *by design*
(see the sign-convention note below) and must not be "symmetrised".

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Star-imports `linear_operators`, `nonlinear_residuals`, `preconditioners` and re-exports the engine printer (`print_cuda`, `print_cuda_multiple`, `CUDA_FUNCTIONS`). `dxdt`, `time_derivative`, `jacobian`, `_stage_utils` are imported by full path, not star-imported. |
| `_matrix_utils.py` | `mass_matrix_ir` — normalises a mass matrix (`None`/SymPy/NumPy/nested sequences) to row-major IR entries; integer entries become floats so emitted mass terms carry explicit float literals. |
| `dxdt.py` | `generate_dxdt_fac_code` (emits `dxdt(state, parameters, drivers, observables, out, t)`) and `generate_observables_fac_code` (emits `get_observables(...)`). The explicit RHS used by all algorithms. |
| `time_derivative.py` | `generate_time_derivative_fac_code`: emits `time_derivative_rhs(state, parameters, drivers, driver_dt, observables, out, t)` computing ∂RHS/∂t = direct ∂t + driver chain (via `driver_dt`) + chain rule through intermediates. Used by Rosenbrock-W. |
| `jacobian.py` | Pure-symbolic (emits no CUDA): `generate_jacobian` (full analytic Jacobian via chain rule over auxiliary assignments; returns row-major lists of IR expressions) and `generate_analytical_jvp` (returns `JVPEquations` with `j_ij` entry symbols and `Arr("jvp", i)` product terms). Memoised in a module-level `_cache` keyed by `get_cache_key` over interned IR nodes. |
| `linear_operators.py` | Emits the matrix-free linear operator and JVP cache helpers: `generate_operator_apply_code`, `generate_cached_operator_apply_code`, `generate_prepare_jac_code` (populates `cached_aux`, returns `(code, aux_count)`), `generate_cached_jvp_code`, `generate_n_stage_linear_operator_code` (flattened FIRK). `*_from_jvp` variants take a prebuilt `JVPEquations`. |
| `preconditioners.py` | Emits truncated Neumann-series preconditioners: `generate_neumann_preconditioner_code` (inline state eval), `generate_neumann_preconditioner_cached_code`, `generate_n_stage_neumann_preconditioner_code` (FIRK, `A⊗J`). Device fn takes a caller-supplied `jvp` scratch buffer. The **only** generator whose `order` argument is live (Neumann truncation degree, factory default 1). |
| `nonlinear_residuals.py` | Emits the Newton residual: `generate_residual_code` / `generate_stage_residual_code` (single stage, SDIRK/ESDIRK) and `generate_n_stage_residual_code` (flattened FIRK). |
| `_stage_utils.py` | Shared FIRK helpers: `prepare_stage_data` (Butcher `A`/`c` → IR rows, nodes, stage count) and `build_stage_metadata` (emit `c_i`, `a_i_j` symbol assignments). Used by every `n_stage_*` generator. |

## Generator variants
Most callbacks come in up to three forms, differing only in how state/auxiliaries are supplied —
this is a property of the emitted code, not separate subsystems:

- **single-stage** (`generate_operator_apply_code`, `generate_residual_code`,
  `generate_neumann_preconditioner_code`): the `state` argument is the stage increment; the
  generator substitutes `state_sym → base_state[i] + a_ij*state[i]` inline.
- **`n_stage_*`** (FIRK): one flattened system of `s·n` unknowns; stage coupling (`A⊗J`) and
  per-stage time nodes are baked in via `_stage_utils`.
- **`*_cached` / `prepare_jac`** (Rosenbrock-W): `state` is the actual state (no substitution);
  selected auxiliaries are precomputed once per step into `cached_aux` by `prepare_jac` and read
  back by the operator/JVP/preconditioner. `GenericRosenbrockWStep` requests these (`prepare_jac`,
  `linear_operator_cached`, `neumann_preconditioner_cached`, `cached_aux_count`) and runs
  `prepare_jacobian` once per step. *Which* auxiliaries get cached is chosen by the planner
  (`parsing/auxiliary_caching.plan_auxiliary_cache`), gated by `JVPEquations.min_ops_threshold`
  (default 10 ops saved) and `cache_slot_limit` (default `2*len(jvp_terms)`, overridable via
  `max_cached_terms`). The threshold is deliberately conservative, so on many systems few terms
  are cached — but the path is live, not disabled.

**Consumers:** emitted device functions are wired by `symbolicODE.get_solver_helper(func_type)`
and consumed downstream by `cubie/integrators/matrix_free_solvers/` (`LinearSolver`,
`NewtonKrylov`) and the implicit algorithms in `cubie/integrators/algorithms/`. Treat the
device-function signatures in each template docstring as the contract.

## For AI Agents

### Generators emit strings, not functions
Every public `generate_*` builds Python source from a module-level `*_TEMPLATE`. To wire in a new
one, add a branch in `symbolicODE.get_solver_helper` (and the import/type list near the top of
`symbolicODE.py`) — generating the string alone does nothing. **Templates are
indentation-sensitive:** bodies come from `print_cuda_multiple(...)` then joined with explicit
leading spaces (8 inside a factory body, 12 inside the preconditioner's `for _ in range(order)`
loop). Preserve the exact counts or the generated source won't parse.

### Sign/coefficient convention (load-bearing)
Operator `β·M·v − γ·a_ij·h·(J·v)` (explicit `a_ij`); residual `β·M·u − γ·h·f(base + a_ij·u)`
(`a_ij` only inside the eval point, not multiplying `f`); preconditioner approximates
`(β·I − γ·a_ij·h·J)⁻¹` with `h_eff = (γ·a_ij/β)·h`. The operator equals `∂residual/∂u` —
differentiating the residual's eval point pulls the `a_ij` down via the chain rule, which is why
it appears explicitly in the operator but implicitly in the residual. **The `a_ij` asymmetry is
deliberate; changing one form without the others breaks Newton convergence.** State evaluation
also differs by variant (see Generator variants): non-cached paths substitute
`state → base_state + a_ij*state`; cached paths read `cached_aux` and apply no substitution
(`_build_operator_body`'s `use_cached_aux` flag gates this).

### beta/gamma are renamed (#373)
Emitted as `_cubie_codegen_beta`/`_cubie_codegen_gamma` so they can't collide with user
state/parameter symbols named `beta`/`gamma`; factory signatures still expose plain
`beta=1.0, gamma=1.0`.

### Mass matrix & order
`M` defaults to identity (`sp.eye(n)`); pass an explicit matrix for DAEs (integer entries are cast
to `sp.Float`). `order` is in every factory signature but only the preconditioner uses it (Neumann
truncation degree, factory default `1`); operators/residuals accept and ignore it.

### Reuse jvp_equations
`generate_analytical_jvp` is the expensive step; operator/preconditioner/cached-JVP generators all
accept a prebuilt `JVPEquations` via `jvp_equations=` so one differentiation feeds the whole helper
set.

### Printer (engine)
The printer lives in `engine/printer.py` (see `engine/AGENTS.md`); the emission rules are
unchanged from the SymPy era: `precision(...)` wrapping except in indices and exponents,
`x**2`/`x**3` multiplication chains (now structural Pow rules, not regex), Neumann/constant
integer-exponent aliases (`sym_utils.EXPONENT_ALIAS_PREFIX`), `CUDA_FUNCTIONS` → alias →
`d_` passthrough → plain-call function resolution, Piecewise as nested ternaries, and
scalar-to-array remapping via a name-keyed symbol map (generators pass `sysir.arrayrefs`).

### Codegen hygiene
- `engine.prune_unused(..., output_name=...)` runs last in every `_build_*`, dropping
  intermediates that don't feed the named output array (`'out'`, `'jvp'`, `'cached_aux'`, …).
  Omitting it bloats output and leaves dangling symbols.
- Stage builders construct **one combined substitution map per stage** and apply it with a
  single `engine.xreplace` pass; all non-dx equation left-hand sides are stage-renamed by
  `build_stage_substitutions` so repeated assignment targets never collapse across stages.
- `cse` toggles the engine's `cse_and_stack` vs `topological_sort` (either emits in dependency
  order); the Jacobian/JVP cache normalises the CSE flag into its key.
- Jacobian `_cache` is process-global and unbounded, keyed by equation tuple + input/output orders
  + CSE flag; the Jacobian matrix and JVP share one entry (`"jac"`/`"jvp"`).
- Each generator module registers `default_timelogger` events at import (the functions return
  strings, not cacheable objects) and brackets work with `start_event`/`stop_event`.

### Testing
`tests/odesystems/symbolic/` (`test_dxdt`, `test_time_derivative`, `test_jacobian`,
`test_cuda_printer`, `test_solver_helpers`) + `tests/odesystems/symbolic/codegen/`
(`test_stage_utils`). These are numerically critical — validate emitted operators against
finite-difference Jacobians and the trio's algebraic consistency, not just that the source imports.
See root for CUDASIM/real-CUDA commands.

## Dependencies
### Internal
- `cubie.odesystems.symbolic.parsing` (`ParsedEquations`, `IndexedBases`, `JVPEquations`,
  `TIME_SYMBOL`); `cubie.odesystems.symbolic.sym_utils` (`cse_and_stack`, `topological_sort`,
  `prune_unused_assignments`, `render_constant_assignments`); `cubie.time_logger`
  (`default_timelogger` codegen timing). Consumed by `symbolicODE` and, downstream,
  `cubie.integrators.matrix_free_solvers` and the implicit algorithms.
### External
- `sympy` only at the conversion boundary (via `engine.from_sympy`). `numba` (CUDA) is the
  target of the emitted source, invoked downstream, not here.
