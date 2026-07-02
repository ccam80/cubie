<!-- Parent: ../AGENTS.md -->

# odesystems

## Purpose
Defines the abstract and concrete representations of CUDA-backed ODE systems.
`BaseODE(CUDAFactory)` is the abstract root — **never instantiated directly** — and owns the
machinery every system needs: the `CUDAFactory` cache interaction, the `ODEData`
compile-settings container (states, parameters, constants, observables, precision), and the
`SystemValues` name↔value/array mappings. `SymbolicODE` (in `symbolic/`) is the sole concrete
subclass, adding programmatic generation of the `dxdt`, Jacobian, and matrix-free solver-helper
device functions. `ODECache` is the attrs cache `build()` returns (compiled `dxdt` + optional
solver helpers); `ODEData`/`SystemSizes` bundle the system metadata integrator factories read.

See `CUDAFactory` (root) for the build/cache/`update` contract, closure capture, config, and
attrs conventions.

## Key Files
| File | Description |
|------|-------------|
| `baseODE.py` | `BaseODE(CUDAFactory)` abstract base and `ODECache(CUDADispatcherCache)` — the cache `build()` returns. |
| `ODEData.py` | `ODEData(CUDAFactoryConfig)` compile-settings bundle + `SystemSizes` (frozen per-category counts passed to kernels). |
| `SystemValues.py` | `SystemValues` — name↔value mapping with dict/array access, precision coercion, and sympy-key conversion. |
| `__init__.py` | Re-exports `BaseODE`, `ODECache`, `ODEData`, `SystemSizes`, `SystemValues`, and (from `symbolic/`) `SymbolicODE`, `create_ODE_system`, `load_cellml_model`. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `symbolic/` | SymPy-driven CUDA codegen for `SymbolicODE(BaseODE)` (see `symbolic/AGENTS.md`). |

## For AI Agents

### ODECache & the `-1` sentinel
`build()` (implemented by `SymbolicODE`, not `BaseODE`) returns an `ODECache`. All optional
fields default to the **integer `-1`**, not `None` — guard with `cache.field != -1` before
using a helper. Full field set: `dxdt` (required), `linear_operator`, `linear_operator_cached`,
`neumann_preconditioner`, `neumann_preconditioner_cached`, `stage_residual`, `n_stage_residual`,
`n_stage_linear_operator`, `n_stage_neumann_preconditioner`, `observables`, `prepare_jac`,
`calculate_cached_jvp`, `time_derivative_rhs`, `cached_aux_count`. `get_cached_output` raises
`NotImplementedError` for a `-1` field.

### BaseODE.update() — additions over the base contract
On top of the standard `CUDAFactory` update (non-underscored keys, `KeyError` on unknown unless
`silent`, returns the recognised `set`), `BaseODE.update()` also routes constant-*value* changes
through `set_constants()` (updates the `constants` `SystemValues` and re-applies it) and
propagates a `precision` change to all four embedded `SystemValues` via
`compile_settings.update_precisions()`, unioning the recognised labels.

### config_hash folds constant values
`BaseODE.config_hash` extends the parent hash with a SHA-256 over the sorted constant items,
because constants are captured into CUDA closures and so affect compiled output even though
mutating a value doesn't trip the attrs equality check.

### `_mass` is outside the cache key
`ODEData._mass` is `eq=False`, so mass-matrix changes are invisible to the `CUDAFactory` cache;
a subclass that uses mass must invalidate itself. (Tracked in `_CLEANUP.md`.)

### get_solver_helper at the base level
`BaseODE.get_solver_helper(func_name, beta, gamma, mass, preconditioner_order)` ignores every
argument except `func_name` — it just delegates to `get_cached_output(func_name)`. Only
`SymbolicODE` overrides it with parameterised codegen.

### SystemValues
- **A plain Python class, not attrs** — don't use `attrs.fields()`/`has()` on it.
- Accepts `sympy.Symbol` keys (auto-converted to strings by `_convert_symbol_keys`) and
  lists/tuples of names (expanded to `{name: 0.0}` — used to declare variables before values).
- **Precision is fixed at construction:** `__init__` calls `update_param_array_and_indices()`,
  materialising `values_array` at `precision`. Reassigning `.precision` later does *not* recast
  existing values — call `update_param_array_and_indices()` again to rebuild.
- `update_from_dict()` returns the **recognised** keys as `set[str]` (not the unrecognised);
  `add_entry()`/`remove_entry()` mutate in place and rebuild the index maps, so call them only
  before compilation (in-flight device functions hold stale sizes).

### `initial_values` is an alias for `states`
`BaseODE.initial_values` and `.states` both return `compile_settings.initial_states`. Don't
confuse the property with the `initial_values` constructor argument, which feeds
`ODEData.from_BaseODE_initargs` — the canonical `ODEData` builder (never construct `ODEData`
directly).

### Adding a component category
A fifth slot (beyond states/parameters/constants/observables) touches two files in several
places: in `ODEData.py`, add the field, the `SystemSizes` count, and extend
`update_precisions()` and `from_BaseODE_initargs()`; in `baseODE.py`, add a property.

### Testing
`tests/odesystems/test_ODEData.py`, `test_SystemValues.py`; `BaseODE` is covered indirectly via
`SymbolicODE` fixtures. See root for the CUDASIM/real-CUDA commands.

## Dependencies
### Internal
- `cubie.CUDAFactory` (`CUDAFactory`, `CUDAFactoryConfig`, `CUDADispatcherCache`, `hash_tuple`);
  `cubie._utils` (`PrecisionDType`); `cubie.odesystems.symbolic` (re-exported).
### External
- `attrs`; `numpy`; `sympy` (`Symbol`); `numba`/`numba-cuda` (compilation in subclasses).
