<!-- Parent: ../AGENTS.md -->

# parsing

## Purpose
Front end of the symbolic codegen pipeline. Converts every supported ODE input form —
newline/iterable equation strings, raw SymPy equations, a Python callable, or a CellML file —
into the common triple `(equation_map, funcs, new_params)` and ultimately a frozen
`ParsedEquations` container plus an `IndexedBases` symbol map and a system hash. `parse_input` is
the single entry point used by `SymbolicODE.create`; CellML loading (`load_cellml_model`) and the
Jacobian-vector-product structures (`JVPEquations`, `plan_auxiliary_cache`) used later by
`codegen` also live here.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Star-imports `auxiliary_caching`, `cellml`, `jvp_equations`, `parser`; declares `__all__ = ["load_cellml_model"]` (the rest is re-exported via star imports). |
| `parser.py` | Core parser. `parse_input` dispatches on input type; `ParsedEquations` (frozen attrs) partitions equations into state-derivatives/observables/auxiliaries; `EquationWarning`; constants `PARSE_TRANSFORMS`, `KNOWN_FUNCTIONS`, `TIME_SYMBOL`, `DRIVER_SETTING_KEYS`. Holds the string and SymPy LHS/RHS validation passes. |
| `cellml.py` | `load_cellml_model` — wraps optional `cellmlmanip`, sanitises symbol names, splits differential vs algebraic equations, classifies constants/parameters/observables, then calls `parse_input`. Cache-aware (early + post-GUI checks). |
| `cellml_cache.py` | `CellMLCache` — disk LRU cache (≤5 configs per model) of pickled parse results under `generated/<model>/`, keyed by file-content SHA-256 + serialised args, tracked in `cellml_cache_manifest.json`. |
| `jvp_equations.py` | `JVPEquations` (mutable attrs) — holds ordered JVP/auxiliary assignments and derives dependency graphs, op-cost, JVP usage/closure, dependency levels, and slot limits; lazily computes/stores a `CacheSelection`; `cached_partition()` splits into cached/runtime/prepare. |
| `auxiliary_caching.py` | Heuristic cache planner. `CacheGroup`/`CacheSelection` (frozen attrs) and `plan_auxiliary_cache` — enumerate seed-rooted leaf groups, simulate runtime-op savings, and search group combinations under the slot limit to pick what to precompute. |
| `function_inspector.py` | AST analysis of a callable ODE. `inspect_ode_function` → `FunctionInspection`; `_OdeAstVisitor` collects state/constant accesses, assignments (incl. annotated), calls, unrolls `for` (also inside if-branches), synthesises `IfExp` from if/elif/else, rejects unsupported constructs (`while`/`with`/`try`/`match`/nested `def`/comprehensions; branch bodies raise on statements other than assignments and nested `if`/`for`); `AstToSympyConverter` maps AST nodes to SymPy — resolves user-function calls before `KNOWN_FUNCTIONS` (inlining non-device callables), inlines dxdt-named locals, and (in `strict_names` mode) raises on unknown bare names, suggesting the container access when the name is declared. Extra args used only by bare name are `scalar_params` (SciPy `args=` convention), bound to the like-named declared symbol. |
| `function_parser.py` | `parse_function_input` — bridges `FunctionInspection` to the parser's `(equation_map, funcs, new_params)` triple: builds the symbol map (container accesses search parameters → constants → drivers; undeclared attribute/string accesses infer parameters in non-strict mode with `EquationWarning`), emits auxiliary/observable/dxdt equations, inlines `dx = expr; return [dx]` aliases. `infer_function_states` derives state names from dict-return keys or synthesises them for pure positional access when `states` is omitted. |

## For AI Agents

### parse_input — the entry point
Returns `(index_map, all_symbols, funcs, parsed_equations, fn_hash)` — a 5-tuple consumed directly
by `SymbolicODE.create` and `cellml.load_cellml_model`. `_detect_input_type` dispatches to
`"string"`, `"sympy"`, or `"function"` (the function branch imports `function_parser` lazily). All
three branches must produce the same `equation_map` shape (list of `(sp.Symbol, sp.Expr)`).
`strict=False` is the default: undeclared RHS symbols are inferred as parameters and pushed onto
`index_map.parameters`; anonymous `dX`/aux LHS symbols become auxiliaries. `strict=True` requires
every symbol declared and refuses a stateless system.

### Two parallel validation-pass pairs
`_lhs_pass`/`_rhs_pass` (string) and `_lhs_pass_sympy`/`_rhs_pass_sympy` (SymPy) must stay
behaviourally aligned: same state-aware d-prefix detection (`dX` is a derivative only if `X` is a
declared state; `d(x, t)` function notation honoured) and same conversion of underived states →
observables. Symbols are created `real=True` throughout (`TIME_SYMBOL = sp.Symbol("t",
real=True)`); the SymPy branch substitutes user symbols to canonical index-map symbols twice
(before and after the LHS pass) so identity matches.

### ParsedEquations & JVPEquations
`ParsedEquations` is frozen — build a new one via `from_equations`, don't mutate; its
`_state/_observable/_auxiliary_symbols` fields are exposed through same-named properties.
`JVPEquations` is produced by `codegen.jacobian.generate_analytical_jvp`, not here — this module
only defines the container and its derived metadata; treat its `_*` fields as `init=False` computed
state set in `__attrs_post_init__` (never set them directly).

### Driver settings
Keys in `DRIVER_SETTING_KEYS` (`time`, `dt`, `wrap`, `order`) are configuration, not driver
symbols; they're stripped before building driver names and reattached via
`drivers.set_passthrough_defaults`.

### CellML (optional)
`cellmlmanip` is imported in a `try/except` and may be `None`; `load_cellml_model` raises
`ImportError` at call time when it's absent (never import it at top level unguarded). Numeric Dummy
atoms (e.g. `_0.5`) are converted to `sp.Float`/`sp.Integer`; algebraic equations with a numeric
RHS become constants (or parameters if named), non-numeric ones become observables/auxiliaries.
`CellMLCache` is a disk LRU (≤5 configs per model) under `generated/<model>/`, keyed by
file-content SHA-256 + serialised args in `cellml_cache_manifest.json`; both it and `cellml.py`
compute `generated/` from `Path.cwd()` (matching `odefile.GENERATED_DIR`) and invalidate on any
content change (whitespace included).

### User functions
Renamed with a trailing underscore during string parsing to dodge SymPy name clashes; device
functions / functions with derivative helpers are wrapped in dynamic `sp.Function` subclasses whose
`fdiff` emits derivative placeholders (`d_<name>` or the provided derivative's `__name__`).
Non-device callables are inlined when they accept SymPy args. `function_parser` intentionally does
**not** map `dx`/`dv` dxdt symbols into the symbol map, so `dx = expr; return [dx]` inlines `expr`
instead of creating a circular reference to the output.

### function_inspector — supported AST subset
A `for` loop is supported only if it can be **fully unrolled** at parse time: the iterable must be
a literal `range(...)` (integer-literal args), a literal list, or a literal tuple, and the target a
simple name. Every other `for` (over a variable, a non-`range` call, non-constant elements, or with
tuple-unpacking) raises `NotImplementedError`, as do `while`, comprehensions, generators, `with`,
`del`, `assert`, `raise`, `global`, `nonlocal`.

### auxiliary_caching
`plan_auxiliary_cache` skips `_cse`-prefixed symbols when simulating removals and rejects a cache
group if any removed node still has an external dependent (`_simulate_cached_leaves` returns
`None`).

### Testing
`tests/odesystems/symbolic/` (`test_parser`, `test_cellml`, `test_cellml_cache`,
`test_function_inspector`, `test_function_parser`; `JVPEquations` via `test_jacobian`).
Pure-Python parsing — runs without a GPU; CellML tests need optional `cellmlmanip`. See root for
CUDASIM/real-CUDA commands.

## Dependencies
### Internal
- `cubie.odesystems.symbolic.indexedbasemaps` (`IndexedBases`); `cubie.odesystems.symbolic.sym_utils`
  (`hash_system_definition`); `cubie.odesystems.symbolic.symbolicODE` (`SymbolicODE`, lazy in
  `cellml.py`); `cubie.odesystems.symbolic.codegen.jacobian` (produces `JVPEquations`; imported by
  callers, not here); `cubie._utils` (`is_devfunc`, `PrecisionDType`),
  `cubie.time_logger.default_timelogger`, `cubie.gui.constants_editor` (lazy).
### External
- `sympy` (symbols, parsing, `cse`, `Function`, `Piecewise`); `attrs` (`ParsedEquations`,
  `JVPEquations`, `CacheGroup`, `CacheSelection`); `cellmlmanip` (optional); `numpy` (precision dtype
  in the CellML loader). Stdlib `ast`, `inspect`, `pickle`, `json`, `hashlib`, `re`, `itertools`.
