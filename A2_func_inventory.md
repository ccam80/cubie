# A2 Functionality Inventory — Parsing + Codegen + CellML

---

## `odesystems/symbolic/parsing/parser.py`

### `_detect_input_type`

| # | Functionality |
|---|--------------|
| 1 | Returns `"function"` when dxdt is callable and not str/list/tuple |
| 2 | Returns `"string"` when dxdt is a str |
| 3 | Returns `"string"` when first element of iterable is a str |
| 4 | Returns `"sympy"` when first element is `sp.Expr` or `sp.Equality` |
| 5 | Returns `"sympy"` when first element is a 2-tuple of (Symbol/Derivative, Expr) |
| 6 | Raises `TypeError` when dxdt is None |
| 7 | Raises `TypeError` when dxdt is non-iterable non-string non-callable |
| 8 | Raises `ValueError` when dxdt is an empty iterable |
| 9 | Raises `TypeError` when first element is unrecognised type |

### `_normalize_sympy_equations`

| # | Functionality |
|---|--------------|
| 10 | Converts `sp.Equality` with Symbol LHS to (Symbol, Expr) tuple |
| 11 | Converts `sp.Equality` with Derivative LHS to (dX Symbol, Expr) tuple |
| 12 | Raises `ValueError` when Derivative has no arguments |
| 13 | Raises `ValueError` when Derivative arg is not Symbol |
| 14 | Raises `ValueError` when Equality LHS is neither Symbol nor Derivative |
| 15 | Converts tuple with Symbol LHS and Expr RHS |
| 16 | Converts tuple with Derivative LHS to (dX Symbol, Expr) |
| 17 | Raises `TypeError` when tuple has wrong length |
| 18 | Raises `TypeError` when tuple LHS is not Symbol/Derivative |
| 19 | Raises `TypeError` when tuple RHS is not sp.Expr |
| 20 | Raises `TypeError` for bare sp.Expr (no LHS) |
| 21 | Raises `TypeError` for invalid element type |
| 22 | Raises `TypeError` when equations not iterable |

### `ParsedEquations` (frozen attrs class)

| # | Functionality |
|---|--------------|
| 23 | `__iter__` returns iterator over `ordered` |
| 24 | `__len__` returns length of `ordered` |
| 25 | `__getitem__` returns equation at index from `ordered` |
| 26 | `copy` returns dict mapping lhs->rhs |
| 27 | `to_equation_list` returns mutable list of ordered equations |
| 28 | `state_symbols` property forwards `_state_symbols` |
| 29 | `observable_symbols` property forwards `_observable_symbols` |
| 30 | `auxiliary_symbols` property forwards `_auxiliary_symbols` |
| 31 | `non_observable_equations` filters out equations with observable LHS |
| 32 | `dxdt_equations` returns tuple of non-observable equations |
| 33 | `observable_system` returns all ordered equations |
| 34 | `from_equations` classmethod: partitions equations into state/observable/auxiliary by index_map |
| 35 | `from_equations` handles dict input (converts to items) |
| 36 | `from_equations` handles iterable input |

### `_sanitise_input_math`

| # | Functionality |
|---|--------------|
| 37 | Delegates to `_replace_if` for ternary-to-Piecewise conversion |

### `_replace_if`

| # | Functionality |
|---|--------------|
| 38 | Converts `X if COND else Y` to `Piecewise((X, COND), (Y, True))` |
| 39 | Recursively handles nested ternaries |
| 40 | Returns unchanged string when no ternary found |

### `_normalise_indexed_tokens`

| # | Functionality |
|---|--------------|
| 41 | Rewrites `name[index]` to `nameindex` for integer literals |
| 42 | Leaves non-matching tokens unchanged |

### `_rename_user_calls`

| # | Functionality |
|---|--------------|
| 43 | Returns original lines and empty dict when no user_functions |
| 44 | Appends `_` suffix to function call tokens in lines |
| 45 | Returns mapping from original names to suffixed names |

### `_build_sympy_user_functions`

| # | Functionality |
|---|--------------|
| 46 | Creates dynamic Function subclass with `fdiff` for device functions |
| 47 | Creates dynamic Function subclass with `fdiff` when derivative helper provided |
| 48 | `fdiff` uses derivative print name when available |
| 49 | `fdiff` falls back to `d_<orig_name>` when no derivative helper |
| 50 | Creates plain `sp.Function` for non-device functions without derivatives |
| 51 | Returns parse_locals, alias_map, and is_device_map |
| 52 | Handles None user_functions (empty iteration) |

### `_inline_nondevice_calls`

| # | Functionality |
|---|--------------|
| 53 | Returns expr unchanged when no user_functions |
| 54 | Inlines non-device user function calls that return SymPy expressions |
| 55 | Skips device functions (leaves symbolic) |
| 56 | Skips functions not found in user_functions |
| 57 | Keeps symbolic call when inline evaluation fails |
| 58 | Keeps symbolic call when return value is not SymPy type |

### `_process_calls`

| # | Functionality |
|---|--------------|
| 59 | Detects function call tokens via regex in equation lines |
| 60 | Resolves user function names to callables |
| 61 | Resolves known SymPy function names |
| 62 | Raises `ValueError` for unknown function names |

### `_process_parameters`

| # | Functionality |
|---|--------------|
| 63 | Delegates to `IndexedBases.from_user_inputs` with all arguments |

### `_lhs_pass`

| # | Functionality |
|---|--------------|
| 64 | Detects `d(name, t)` function notation as derivative of known state |
| 65 | `d(name, t)` where name is observable: converts observable to state with warning |
| 66 | `d(name, t)` where name is unknown + strict: raises ValueError |
| 67 | `d(name, t)` where name is unknown + non-strict: infers new state |
| 68 | Detects `dX` prefix as derivative when X is a known state |
| 69 | `dX` prefix where X is observable: converts to state with warning |
| 70 | `dX` prefix where X is unknown + non-strict + no initial states: infers new state |
| 71 | `dX` prefix where X is unknown + strict or had initial states: treated as auxiliary |
| 72 | `dX` prefix where X is unknown observable name: tracked as assigned observable |
| 73 | Direct state assignment raises ValueError |
| 74 | Assignment to parameter/constant/driver raises ValueError |
| 75 | Unrecognised LHS not in observables: treated as anonymous auxiliary |
| 76 | LHS matching observable name: tracked as assigned observable |
| 77 | Missing observable assignments raise ValueError |
| 78 | States with no derivative: converted to observables with warning |
| 79 | State-to-observable conversion raises ValueError if already observable |

### `_lhs_pass_sympy`

| # | Functionality |
|---|--------------|
| 80 | Mirrors `_lhs_pass` logic for SymPy equation objects |
| 81 | `d`-prefix detection using SymPy Symbol names |
| 82 | Observable-to-state conversion with warning |
| 83 | Non-strict state inference (no initial states) |
| 84 | Strict mode: d-prefix unknown treated as auxiliary |
| 85 | Direct state assignment raises ValueError |
| 86 | Immutable input assignment raises ValueError |
| 87 | Anonymous auxiliary detection |
| 88 | Missing observable validation |
| 89 | Underived state-to-observable conversion with warning |

### `_process_user_functions_for_rhs`

| # | Functionality |
|---|--------------|
| 90 | Builds SymPy wrappers via `_build_sympy_user_functions` |
| 91 | Returns callable mapping with original user function names |
| 92 | Returns empty dict when user_funcs is None |

### `_rhs_pass_sympy`

| # | Functionality |
|---|--------------|
| 93 | Validates all RHS free_symbols are declared (strict mode) |
| 94 | Raises ValueError for undeclared symbols in strict mode |
| 95 | Infers undeclared symbols as parameters in non-strict mode |
| 96 | Returns validated equations, funcs, and new_symbols |

### `_rhs_pass`

| # | Functionality |
|---|--------------|
| 97 | Calls `_process_calls` to validate function references |
| 98 | Renames user function calls via `_rename_user_calls` |
| 99 | Builds SymPy function wrappers via `_build_sympy_user_functions` |
| 100 | Parses RHS with transforms in strict mode |
| 101 | Raises ValueError (from NameError/TypeError) for undefined symbols in strict mode |
| 102 | Parses RHS without transforms in non-strict mode, infers new symbols |
| 103 | Inlines non-device function calls via `_inline_nondevice_calls` |
| 104 | Uses `raw_lines` in error messages when provided |
| 105 | Falls back to `lines` for error messages when raw_lines is None |
| 106 | Raises ValueError for unresolved symbols after all passes |

### `parse_input`

| # | Functionality |
|---|--------------|
| 107 | Defaults states to `{}` when None |
| 108 | Raises ValueError when states=None and strict=True |
| 109 | Defaults observables/parameters/constants/drivers to empty |
| 110 | Extracts driver names from dict, filtering out setting keys |
| 111 | Raises ValueError when driver dict has no driver symbols |
| 112 | Calls `_process_parameters` to build IndexedBases |
| 113 | Detects input type via `_detect_input_type` |
| 114 | String path: splits multiline string into lines |
| 115 | String path: accepts list/tuple of strings |
| 116 | String path: raises ValueError for other string-like types |
| 117 | String path: normalises indexed tokens |
| 118 | String path: runs `_lhs_pass` and `_rhs_pass` |
| 119 | SymPy path: normalises equations, substitutes canonical symbols |
| 120 | SymPy path: runs `_lhs_pass_sympy` and `_rhs_pass_sympy` |
| 121 | SymPy path: second substitution pass after LHS changes |
| 122 | Function path: delegates to `parse_function_input` |
| 123 | Raises RuntimeError for invalid input_type |
| 124 | Pushes inferred parameters into index_map |
| 125 | Sets driver passthrough defaults when driver_dict provided |
| 126 | Exposes user_functions in all_symbols dict |
| 127 | Exposes user_function_derivatives in all_symbols |
| 128 | Builds `__function_aliases__` for string pathway with renaming |
| 129 | Constructs `ParsedEquations` via `from_equations` |
| 130 | Computes system hash via `hash_system_definition` |
| 131 | Returns 5-tuple: (index_map, all_symbols, funcs, parsed_equations, fn_hash) |

---

## `odesystems/symbolic/parsing/function_inspector.py`

### `FunctionInspection.__init__`

| # | Functionality |
|---|--------------|
| 132 | Stores all 9 parameters as instance attributes |

### `_OdeAstVisitor.__init__`

| # | Functionality |
|---|--------------|
| 133 | Initialises state_param, constant_params, and empty collection attributes |

### `_OdeAstVisitor.visit_Subscript`

| # | Functionality |
|---|--------------|
| 134 | Records int subscript access on state param to state_accesses |
| 135 | Records string subscript access on state param to state_accesses |
| 136 | Records int subscript access on constant param to constant_accesses |
| 137 | Records string subscript access on constant param to constant_accesses |
| 138 | Handles `ast.Index` wrapper for Python 3.8 compat |
| 139 | Records `ast.Name` slice as "name" pattern type |
| 140 | Records complex slice expression as "expr" pattern type |
| 141 | Ignores subscripts on non-state/non-constant bases |

### `_OdeAstVisitor.visit_Attribute`

| # | Functionality |
|---|--------------|
| 142 | Records attribute access on state param to state_accesses |
| 143 | Records attribute access on constant param to constant_accesses |
| 144 | Ignores attributes on non-state/non-constant bases |

### `_OdeAstVisitor.visit_Assign`

| # | Functionality |
|---|--------------|
| 145 | Records single Name target assignment |
| 146 | Records Tuple target assignments (unpacking) |

### `_OdeAstVisitor.visit_Return`

| # | Functionality |
|---|--------------|
| 147 | Appends Return node to return_nodes list |

### `_OdeAstVisitor.visit_Call`

| # | Functionality |
|---|--------------|
| 148 | Extracts function name via `_call_name` and adds to function_calls |
| 149 | Skips when `_call_name` returns None |

### `_call_name`

| # | Functionality |
|---|--------------|
| 150 | Returns function name for `ast.Name` func node |
| 151 | Returns `module.attr` for `ast.Attribute` func node |
| 152 | Returns None for unsupported func node types |

### `_resolve_func_name`

| # | Functionality |
|---|--------------|
| 153 | Strips known module prefix (math, np, numpy, cmath) |
| 154 | Returns name unchanged when no module prefix |

### `AstToSympyConverter.__init__`

| # | Functionality |
|---|--------------|
| 155 | Stores symbol_map |

### `AstToSympyConverter.convert`

| # | Functionality |
|---|--------------|
| 156 | Dispatches to `_convert_constant` for ast.Constant |
| 157 | Dispatches to `_convert_name` for ast.Name |
| 158 | Dispatches to `_convert_binop` for ast.BinOp |
| 159 | Dispatches to `_convert_unaryop` for ast.UnaryOp |
| 160 | Dispatches to `_convert_call` for ast.Call |
| 161 | Dispatches to `_convert_subscript` for ast.Subscript |
| 162 | Dispatches to `_convert_attribute` for ast.Attribute |
| 163 | Dispatches to `_convert_compare` for ast.Compare |
| 164 | Dispatches to `_convert_ifexp` for ast.IfExp |
| 165 | Dispatches to `_convert_boolop` for ast.BoolOp |
| 166 | Raises NotImplementedError for ast.Tuple |
| 167 | Raises NotImplementedError for ast.List |
| 168 | Raises NotImplementedError for unsupported node types |

### `AstToSympyConverter._convert_constant`

| # | Functionality |
|---|--------------|
| 169 | Converts int to sp.Integer |
| 170 | Converts float to sp.Float |
| 171 | Converts bool to sp.true/sp.false |
| 172 | Raises NotImplementedError for unsupported constant types |

### `AstToSympyConverter._convert_name`

| # | Functionality |
|---|--------------|
| 173 | Returns symbol from symbol_map when present |
| 174 | Creates new real Symbol and caches in symbol_map when absent |

### `AstToSympyConverter._convert_binop`

| # | Functionality |
|---|--------------|
| 175 | Handles Add, Sub, Mult, Div, FloorDiv, Pow, Mod operations |
| 176 | Raises NotImplementedError for unsupported binary ops |

### `AstToSympyConverter._convert_unaryop`

| # | Functionality |
|---|--------------|
| 177 | Handles USub (negation), UAdd (identity), Not |
| 178 | Raises NotImplementedError for unsupported unary ops |

### `AstToSympyConverter._convert_call`

| # | Functionality |
|---|--------------|
| 179 | Raises NotImplementedError for unnamed function calls |
| 180 | Resolves module-qualified names via `_resolve_func_name` |
| 181 | Raises NotImplementedError for unknown function names |
| 182 | Converts known function call with SymPy equivalent |

### `AstToSympyConverter._convert_subscript`

| # | Functionality |
|---|--------------|
| 183 | Looks up `base[key]` or `base['key']` in symbol_map |
| 184 | Handles ast.Index wrapper for Python 3.8 compat |
| 185 | Raises NotImplementedError for non-constant subscripts |
| 186 | Raises NotImplementedError when lookup not in symbol_map |
| 187 | Raises NotImplementedError for complex subscript targets |

### `AstToSympyConverter._convert_attribute`

| # | Functionality |
|---|--------------|
| 188 | Looks up `base.attr` in symbol_map |
| 189 | Raises NotImplementedError when lookup not in symbol_map |
| 190 | Raises NotImplementedError for complex attribute targets |

### `AstToSympyConverter._convert_compare`

| # | Functionality |
|---|--------------|
| 191 | Handles chained comparisons with And |
| 192 | Single comparison returns relation directly |

### `AstToSympyConverter._comparison_op` (static)

| # | Functionality |
|---|--------------|
| 193 | Maps Gt, GtE, Lt, LtE, Eq, NotEq to SymPy relational operators |
| 194 | Raises NotImplementedError for unsupported comparison ops |

### `AstToSympyConverter._convert_ifexp`

| # | Functionality |
|---|--------------|
| 195 | Converts ternary to `sp.Piecewise((body, test), (orelse, True))` |

### `AstToSympyConverter._convert_boolop`

| # | Functionality |
|---|--------------|
| 196 | Converts `and` to `sp.And` |
| 197 | Converts `or` to `sp.Or` |
| 198 | Raises NotImplementedError for unsupported bool ops |

### `inspect_ode_function`

| # | Functionality |
|---|--------------|
| 199 | Raises TypeError when func is not callable |
| 200 | Raises TypeError for lambda functions |
| 201 | Raises TypeError for builtins without inspectable source |
| 202 | Parses source and finds FunctionDef via ast.walk |
| 203 | Raises ValueError when no FunctionDef found |
| 204 | Raises ValueError when fewer than 2 parameters |
| 205 | Warns when first param is not 't' |
| 206 | Warns when second param is not in conventional names |
| 207 | Visits function body with `_OdeAstVisitor` |
| 208 | Raises ValueError when no return statement found |
| 209 | Raises ValueError when multiple return statements found |
| 210 | Validates access consistency per base variable |
| 211 | Returns populated FunctionInspection |

### `_validate_access_consistency`

| # | Functionality |
|---|--------------|
| 212 | Raises ValueError when both int and string patterns on same base |
| 213 | Passes when single pattern type (discarding expr and name) |

---

## `odesystems/symbolic/parsing/function_parser.py`

### `parse_function_input`

| # | Functionality |
|---|--------------|
| 214 | Defaults observables to empty list when None |
| 215 | Inspects function via `inspect_ode_function` |
| 216 | Builds symbol map via `_build_symbol_map` |
| 217 | Unpacks return value via `_unpack_return` |
| 218 | Raises ValueError when return element count != state count |
| 219 | Skips auxiliary assignments for observables, dxdt, states, constant params, state param |
| 220 | Skips auxiliary assignments that are direct access aliases |
| 221 | Converts remaining local assignments to auxiliary equations |
| 222 | Converts observable assignments using index_map symbol lookup |
| 223 | Dict return: maps keys to state derivative equations |
| 224 | Dict return: raises ValueError for non-string-literal keys |
| 225 | Dict return: raises ValueError for non-state keys |
| 226 | List/tuple return: positional mapping to dxdt equations |
| 227 | Returns (equation_map, empty funcs, empty new_params) |

### `_build_symbol_map`

| # | Functionality |
|---|--------------|
| 228 | Maps time parameter to TIME_SYMBOL |
| 229 | Maps integer-indexed state accesses to state symbols |
| 230 | Maps string-indexed state accesses to state symbols |
| 231 | Maps attribute state accesses to state symbols |
| 232 | Maps integer-indexed constant/parameter accesses |
| 233 | Maps string-indexed constant/parameter accesses |
| 234 | Maps attribute constant/parameter accesses |
| 235 | Resolves assignment aliases to state/constant symbols |
| 236 | Does NOT add dxdt symbols to symbol map (avoids circular refs) |
| 237 | Maps observable symbols from index_map |

### `_resolve_alias`

| # | Functionality |
|---|--------------|
| 238 | Returns symbol for subscript access on known base |
| 239 | Returns symbol for attribute access on known base |
| 240 | Returns None for non-matching nodes |

### `_is_access_alias`

| # | Functionality |
|---|--------------|
| 241 | Returns True for subscript on state_param or constant_params |
| 242 | Returns True for attribute on state_param or constant_params |
| 243 | Returns False otherwise |

### `_unpack_return`

| # | Functionality |
|---|--------------|
| 244 | Unpacks ast.List/ast.Tuple into list of expressions |
| 245 | Unpacks ast.Dict values into list of expressions |
| 246 | Wraps single expression as single-element list |
| 247 | Inlines local assignment when return element is Name matching an assignment |
| 248 | Defaults assignments to empty dict when None |

---

## `odesystems/symbolic/codegen/numba_cuda_printer.py`

### `CUDAPrinter.__init__`

| # | Functionality |
|---|--------------|
| 249 | Initialises symbol_map, cuda_functions, func_aliases |
| 250 | Extracts `__function_aliases__` from symbol_map when present |
| 251 | Initialises `_in_index` and `_in_pow` context flags to False |

### `CUDAPrinter.doprint`

| # | Functionality |
|---|--------------|
| 252 | Forces outer assignment for Piecewise when assign_to provided |
| 253 | Delegates to super().doprint for non-Piecewise |
| 254 | Applies `_replace_powers_with_multiplication` post-processing |

### `CUDAPrinter._print_Symbol`

| # | Functionality |
|---|--------------|
| 255 | Returns array-substituted print when symbol in symbol_map |
| 256 | Falls back to default Symbol printing |

### `CUDAPrinter._print_Indexed`

| # | Functionality |
|---|--------------|
| 257 | Sets `_in_index=True` while printing indices |
| 258 | Prints `base[indices]` format |

### `CUDAPrinter._print_Integer`

| # | Functionality |
|---|--------------|
| 259 | Returns unwrapped integer when `_in_index` or `_in_pow` is True |
| 260 | Returns `precision(N)` wrapped integer otherwise |

### `CUDAPrinter._print_Pow`

| # | Functionality |
|---|--------------|
| 261 | Parenthesizes base for compound expressions |
| 262 | Sets `_in_pow=True` while printing exponent |
| 263 | Returns `base**exponent` format |

### `CUDAPrinter._print_Piecewise`

| # | Functionality |
|---|--------------|
| 264 | Builds nested ternary from Piecewise pieces in reverse |
| 265 | Last piece used as fallback expression |

### `CUDAPrinter._replace_powers_with_multiplication`

| # | Functionality |
|---|--------------|
| 266 | Delegates to `_replace_square_powers` then `_replace_cube_powers` |

### `CUDAPrinter._replace_square_powers`

| # | Functionality |
|---|--------------|
| 267 | Replaces `x**2` with `x*x` for simple identifiers |
| 268 | Replaces `(expr)**2` with `(expr)*(expr)` for parenthesized expressions |
| 269 | Handles `x**2.0` variant |

### `CUDAPrinter._replace_cube_powers`

| # | Functionality |
|---|--------------|
| 270 | Replaces `x**3` with `x*x*x` for simple identifiers |
| 271 | Replaces `(expr)**3` with `(expr)*(expr)*(expr)` for parenthesized expressions |
| 272 | Handles `x**3.0` variant |

### `CUDAPrinter._print_Function`

| # | Functionality |
|---|--------------|
| 273 | Maps CUDA-known functions to `math.*` equivalents |
| 274 | Maps aliased user functions to original names |
| 275 | Prints derivative functions (`d_*`) as-is |
| 276 | Falls back to plain function name for unknown functions |

### `CUDAPrinter._print_Float`

| # | Functionality |
|---|--------------|
| 277 | Returns unwrapped float for 2.0/3.0 when in pow context |
| 278 | Returns `precision(value)` wrapped float otherwise |

### `CUDAPrinter._print_Rational`

| # | Functionality |
|---|--------------|
| 279 | Returns `precision(p/q)` wrapped rational |

### `print_cuda`

| # | Functionality |
|---|--------------|
| 280 | Creates CUDAPrinter and prints single expression |

### `print_cuda_multiple`

| # | Functionality |
|---|--------------|
| 281 | Creates CUDAPrinter and prints each (assign_to, expr) pair |

---

## `odesystems/symbolic/codegen/dxdt.py`

### `generate_dxdt_lines`

| # | Functionality |
|---|--------------|
| 282 | Extracts non-observable equations from ParsedEquations |
| 283 | Applies CSE when cse=True |
| 284 | Applies topological sort when cse=False |
| 285 | Filters out observable symbols when index_map provided |
| 286 | Prunes unused assignments when index_map provided |
| 287 | Uses index_map.all_arrayrefs as symbol_map |
| 288 | Returns `["pass"]` when no lines generated |

### `generate_observables_lines`

| # | Functionality |
|---|--------------|
| 289 | Returns `["pass"]` early when no observables in index_map |
| 290 | Applies CSE or topological sort |
| 291 | Substitutes dxdt symbols with numbered `dxout_` symbols |
| 292 | Substitutes arrayrefs into equations |
| 293 | Prunes unused assignments for observables |
| 294 | Returns `["pass"]` when no lines generated |

### `generate_dxdt_fac_code`

| # | Functionality |
|---|--------------|
| 295 | Generates dxdt lines via `generate_dxdt_lines` |
| 296 | Renders constant assignments block |
| 297 | Formats DXDT_TEMPLATE with func_name, const_lines, body |
| 298 | Logs timing via default_timelogger |

### `generate_observables_fac_code`

| # | Functionality |
|---|--------------|
| 299 | Generates observable lines via `generate_observables_lines` |
| 300 | Renders constant assignments block |
| 301 | Formats OBSERVABLES_TEMPLATE with func_name, const_lines, body |
| 302 | Logs timing via default_timelogger |

---

## `odesystems/symbolic/codegen/time_derivative.py`

### `_build_time_derivative_assignments`

| # | Functionality |
|---|--------------|
| 303 | Topologically sorts non-observable equations |
| 304 | Creates driver_dt IndexedBase when drivers present |
| 305 | Computes direct time derivative via `sp.diff(rhs, TIME_SYMBOL)` per equation |
| 306 | Computes driver partial contribution when driver in free_symbols |
| 307 | Computes chain rule term from previously processed auxiliaries |
| 308 | Creates `time_<lhs>` derivative symbols |
| 309 | Appends output mapping `time_rhs[i]` for each dxdt symbol |
| 310 | Returns (assignments, final_symbol_map) |

### `generate_time_derivative_lines`

| # | Functionality |
|---|--------------|
| 311 | Builds assignments via `_build_time_derivative_assignments` |
| 312 | Applies CSE or topological sort |
| 313 | Prunes unused assignments for `time_rhs` outputs |
| 314 | Prints CUDA lines with combined symbol maps |
| 315 | Returns `["pass"]` when no lines |

### `generate_time_derivative_fac_code`

| # | Functionality |
|---|--------------|
| 316 | Generates time derivative lines |
| 317 | Renders constant assignments block |
| 318 | Formats TIME_DERIVATIVE_TEMPLATE |
| 319 | Logs timing |

---

## `odesystems/symbolic/codegen/jacobian.py`

### `get_cache_key`

| # | Functionality |
|---|--------------|
| 320 | Converts dict equations to tuple of items |
| 321 | Converts iterable equations to tuple of tuples |
| 322 | Returns 4-tuple of (eq_tuple, input_tuple, output_tuple, cse) |

### `generate_jacobian`

| # | Functionality |
|---|--------------|
| 323 | Returns cached Jacobian when available and use_cache=True |
| 324 | Topologically sorts equations |
| 325 | Separates auxiliary and output equations |
| 326 | Computes chain-rule gradients for auxiliary equations |
| 327 | Raises ValueError for topological order violation |
| 328 | Computes Jacobian rows for output equations with chain rule |
| 329 | Caches result: adds to existing cache entry or creates new |
| 330 | Skips cache when use_cache=False |

### `generate_analytical_jvp`

| # | Functionality |
|---|--------------|
| 331 | Substitutes observable symbols with numbered auxiliaries |
| 332 | Returns cached JVP when available |
| 333 | Constructs ParsedEquations for substituted system |
| 334 | Generates Jacobian via `generate_jacobian` |
| 335 | Flattens Jacobian, dropping zero entries, to (j_ij, expr) pairs |
| 336 | Builds JVP sum `sum(j_ij * v[j])` per output |
| 337 | Removes output equations (not needed for JVP) |
| 338 | Applies CSE or topological sort |
| 339 | Prunes unused assignments |
| 340 | Caches and returns JVPEquations |

---

## `odesystems/symbolic/codegen/linear_operators.py`

### `_partition_cached_assignments`

| # | Functionality |
|---|--------------|
| 341 | Delegates to `equations.cached_partition()` |

### `_inline_aux_assignments`

| # | Functionality |
|---|--------------|
| 342 | Returns auxiliary expressions in non_jvp_order |

### `_build_operator_body`

| # | Functionality |
|---|--------------|
| 343 | Computes mass matrix-vector product `M @ v` terms |
| 344 | Converts integer mass matrix entries to Float |
| 345 | Builds `beta*M*v - gamma*a_ij*h*J*v` output updates |
| 346 | Non-cached path: applies state substitution `base_state + a_ij*state` |
| 347 | Cached path: reads auxiliaries from `cached_aux[idx]` |
| 348 | Non-cached path: deduplicates combined assignments |
| 349 | Prunes unused assignments for `out` outputs |
| 350 | Returns `"        pass"` when no lines |

### `_build_cached_jvp_body`

| # | Functionality |
|---|--------------|
| 351 | Reads cached auxiliaries from indexed base |
| 352 | Builds `J*v` output updates from jvp_terms |
| 353 | Prunes unused assignments |
| 354 | Returns `"        pass"` when no lines |

### `_build_prepare_body`

| # | Functionality |
|---|--------------|
| 355 | Assigns preparation expressions |
| 356 | Writes cached values to `cached_aux[idx]` |
| 357 | Prunes unused assignments for `cached_aux` |
| 358 | Returns `"        pass"` when no lines |

### `generate_operator_apply_code_from_jvp`

| # | Functionality |
|---|--------------|
| 359 | Gets inline aux assignments and builds operator body |
| 360 | Renders constants and formats OPERATOR_APPLY_TEMPLATE |

### `generate_cached_operator_apply_code_from_jvp`

| # | Functionality |
|---|--------------|
| 361 | Partitions cached/runtime assignments and builds operator body |
| 362 | Renders constants and formats CACHED_OPERATOR_APPLY_TEMPLATE |

### `generate_prepare_jac_code_from_jvp`

| # | Functionality |
|---|--------------|
| 363 | Partitions assignments, builds prepare body |
| 364 | Returns (code, aux_count) tuple |

### `generate_cached_jvp_code_from_jvp`

| # | Functionality |
|---|--------------|
| 365 | Partitions assignments, builds cached JVP body |
| 366 | Formats CACHED_JVP_TEMPLATE |

### `generate_operator_apply_code`

| # | Functionality |
|---|--------------|
| 367 | Defaults M to identity matrix when None |
| 368 | Generates JVP equations when not provided |
| 369 | Delegates to `generate_operator_apply_code_from_jvp` |
| 370 | Logs timing |

### `generate_cached_operator_apply_code`

| # | Functionality |
|---|--------------|
| 371 | Defaults M to identity when None |
| 372 | Generates JVP equations when not provided |
| 373 | Delegates to `generate_cached_operator_apply_code_from_jvp` |
| 374 | Logs timing |

### `generate_prepare_jac_code`

| # | Functionality |
|---|--------------|
| 375 | Generates JVP equations when not provided |
| 376 | Delegates to `generate_prepare_jac_code_from_jvp` |
| 377 | Logs timing |

### `generate_cached_jvp_code`

| # | Functionality |
|---|--------------|
| 378 | Generates JVP equations when not provided |
| 379 | Delegates to `generate_cached_jvp_code_from_jvp` |
| 380 | Logs timing |

### `_build_n_stage_operator_lines`

| # | Functionality |
|---|--------------|
| 381 | Builds stage metadata via `build_stage_metadata` |
| 382 | Per stage: substitutes dx/observable/time/driver symbols |
| 383 | Per stage: computes state evaluation points with coefficient sums |
| 384 | Per stage: builds direction vector combinations for v substitution |
| 385 | Per stage: builds auxiliary assignments with stage-indexed symbols |
| 386 | Per stage: builds JVP terms with substitutions |
| 387 | Per stage: builds output `beta*M*v - gamma*h*jvp` updates |
| 388 | Applies CSE or topological sort to combined expressions |
| 389 | Prunes unused assignments for `out` |
| 390 | Returns `"        pass"` when no lines |

### `generate_n_stage_linear_operator_code`

| # | Functionality |
|---|--------------|
| 391 | Prepares stage data via `prepare_stage_data` |
| 392 | Defaults M to identity when None |
| 393 | Generates JVP equations when not provided |
| 394 | Builds body via `_build_n_stage_operator_lines` |
| 395 | Formats N_STAGE_OPERATOR_TEMPLATE |
| 396 | Logs timing |

---

## `odesystems/symbolic/codegen/preconditioners.py`

### `_build_neumann_body_with_state_subs`

| # | Functionality |
|---|--------------|
| 397 | Builds state substitution `base_state[i] + a_ij * state[i]` |
| 398 | Applies state substitution to JVP assignments |
| 399 | Prints CUDA lines and replaces `v[` with `out[` |
| 400 | Prunes unused assignments for `out` |
| 401 | Returns `["pass"]` when no lines |

### `_build_cached_neumann_body`

| # | Functionality |
|---|--------------|
| 402 | Partitions cached/runtime from equations |
| 403 | Reads cached auxiliaries from `cached_aux[idx]` |
| 404 | Builds JVP output assignments |
| 405 | Prunes unused assignments for `v` |
| 406 | Replaces `v[` with `out[` in printed lines |
| 407 | Returns `"            pass"` when no lines |

### `_build_n_stage_neumann_lines`

| # | Functionality |
|---|--------------|
| 408 | Builds stage metadata |
| 409 | Per stage: substitutes symbols and computes state evaluation points |
| 410 | Per stage: builds direction combos and v-substitution |
| 411 | Per stage: builds stage aux assignments with substitutions |
| 412 | Per stage: builds JVP terms and writes to `jvp[offset+i]` |
| 413 | Applies CSE or topological sort |
| 414 | Prunes unused assignments for `jvp` |
| 415 | Returns `"            pass"` when no lines |

### `generate_neumann_preconditioner_code`

| # | Functionality |
|---|--------------|
| 416 | Generates JVP equations when not provided |
| 417 | Builds body via `_build_neumann_body_with_state_subs` |
| 418 | Formats NEUMANN_TEMPLATE with n_out, jv_body, const_lines |
| 419 | Logs timing |

### `generate_neumann_preconditioner_cached_code`

| # | Functionality |
|---|--------------|
| 420 | Generates JVP equations when not provided |
| 421 | Builds body via `_build_cached_neumann_body` |
| 422 | Formats NEUMANN_CACHED_TEMPLATE |
| 423 | Logs timing |

### `generate_n_stage_neumann_preconditioner_code`

| # | Functionality |
|---|--------------|
| 424 | Prepares stage data |
| 425 | Generates JVP equations when not provided |
| 426 | Builds body via `_build_n_stage_neumann_lines` |
| 427 | Formats N_STAGE_NEUMANN_TEMPLATE with total_states and state_count |
| 428 | Logs timing |

---

## `odesystems/symbolic/codegen/nonlinear_residuals.py`

### `_build_residual_lines`

| # | Functionality |
|---|--------------|
| 429 | Substitutes dxdt symbols with `dx_i` intermediates |
| 430 | Substitutes observable symbols with numbered `aux_` symbols |
| 431 | Applies state evaluation `base_state[i] + a_ij * u[i]` |
| 432 | Computes `beta * M * u - gamma * h * dx_i` per output |
| 433 | Applies CSE or topological sort |
| 434 | Prunes unused assignments for `out` |
| 435 | Returns `"        pass"` when no lines |

### `_build_n_stage_residual_lines`

| # | Functionality |
|---|--------------|
| 436 | Builds stage metadata |
| 437 | Per stage: substitutes dx/observable/time/driver symbols |
| 438 | Per stage: computes state evaluation points with coefficient sums |
| 439 | Per stage: builds `beta*M*u - gamma*h*dx` output updates |
| 440 | Applies CSE or topological sort |
| 441 | Prunes unused assignments for `out` |
| 442 | Returns `"        pass"` when no lines |

### `generate_residual_code`

| # | Functionality |
|---|--------------|
| 443 | Defaults M to identity when None |
| 444 | Builds residual lines via `_build_residual_lines` |
| 445 | Renders constants and formats RESIDUAL_TEMPLATE |

### `generate_stage_residual_code`

| # | Functionality |
|---|--------------|
| 446 | Delegates to `generate_residual_code` |
| 447 | Logs timing |

### `generate_n_stage_residual_code`

| # | Functionality |
|---|--------------|
| 448 | Prepares stage data |
| 449 | Defaults M to identity when None |
| 450 | Builds body via `_build_n_stage_residual_lines` |
| 451 | Formats N_STAGE_RESIDUAL_TEMPLATE |
| 452 | Logs timing |

---

## `odesystems/symbolic/codegen/_stage_utils.py`

### `prepare_stage_data`

| # | Functionality |
|---|--------------|
| 453 | Sympifies coefficient matrix via `sp.Matrix.applyfunc(sp.S)` |
| 454 | Sympifies node expressions via `sp.S` |
| 455 | Returns (coeff_matrix, node_exprs, stage_count) |

### `build_stage_metadata`

| # | Functionality |
|---|--------------|
| 456 | Creates `c_<stage>` node symbols and assigns node values |
| 457 | Creates `a_<stage>_<col>` coefficient symbols and assigns values |
| 458 | Returns (metadata_exprs, coeff_symbols, node_symbols) |

---

## `odesystems/symbolic/parsing/cellml.py`

### `_sanitize_symbol_name`

| # | Functionality |
|---|--------------|
| 459 | Replaces `$` with `_` |
| 460 | Replaces `.` with `_` |
| 461 | Prepends `var` when name starts with `_` followed by digit |
| 462 | Prepends `var_` when name starts with digit |
| 463 | Replaces remaining invalid characters with `_` |

### `load_cellml_model`

| # | Functionality |
|---|--------------|
| 464 | Raises ImportError when cellmlmanip not installed |
| 465 | Raises TypeError when path is not string |
| 466 | Raises FileNotFoundError when path does not exist |
| 467 | Raises ValueError when file lacks .cellml extension |
| 468 | Defaults name to filename stem when None |
| 469 | Non-GUI path: checks cache early, returns cached ODE on hit |
| 470 | Loads model via `cellmlmanip.load_model` |
| 471 | Converts Dummy symbols to regular Symbols with sanitized names |
| 472 | Extracts initial values from state variables |
| 473 | Extracts units from state variables |
| 474 | Identifies time variable from derivative independent variables |
| 475 | Raises ValueError for multiple independent variables |
| 476 | Maps time variable to standard `t` symbol |
| 477 | Converts numeric Dummy symbols (`_0.5`, `_1.0`) to Integer/Float |
| 478 | Separates differential and algebraic equations |
| 479 | Substitutes Dummy-to-Symbol mapping in all equations |
| 480 | Builds dxdt equations from differential equations |
| 481 | Classifies algebraic equations as constants vs parameters vs auxiliaries |
| 482 | Parameters classification: checks against parameters_set |
| 483 | Collects units for parameters and observables |
| 484 | Handles parameters as dict: merges with CellML-extracted values (CellML takes precedence) |
| 485 | GUI path: launches `edit_pre_parse_dicts` for user editing |
| 486 | Post-GUI cache check with effective parameters |
| 487 | Cache miss: calls `parse_input` with all extracted data |
| 488 | Saves parsed result to cache |
| 489 | Constructs and returns SymbolicODE |

---

## `odesystems/symbolic/parsing/cellml_cache.py`

### `CellMLCache.__init__`

| # | Functionality |
|---|--------------|
| 490 | Raises TypeError when model_name is not string |
| 491 | Raises TypeError when cellml_path is not string |
| 492 | Raises ValueError when model_name is empty |
| 493 | Raises FileNotFoundError when cellml_path does not exist |
| 494 | Sets cache_dir relative to CWD/generated/model_name |
| 495 | Sets max_entries to 5 |

### `CellMLCache.get_cellml_hash`

| # | Functionality |
|---|--------------|
| 496 | Reads file in binary mode and returns SHA256 hex digest |

### `CellMLCache._serialize_args`

| # | Functionality |
|---|--------------|
| 497 | Sorts parameter and observable lists for order-independence |
| 498 | Converts precision to string via `__name__` or `str()` |
| 499 | Handles None precision |
| 500 | Returns deterministic JSON string |

### `CellMLCache.compute_cache_key`

| # | Functionality |
|---|--------------|
| 501 | Combines file hash and serialized args hash |
| 502 | Returns first 16 characters of SHA256 |

### `CellMLCache._load_manifest`

| # | Functionality |
|---|--------------|
| 503 | Returns empty manifest when file doesn't exist |
| 504 | Returns empty manifest on JSON decode error |
| 505 | Returns parsed manifest on success |

### `CellMLCache._save_manifest`

| # | Functionality |
|---|--------------|
| 506 | Creates cache directory if needed |
| 507 | Writes manifest as indented JSON |
| 508 | Logs failure message without raising |

### `CellMLCache._update_lru_order`

| # | Functionality |
|---|--------------|
| 509 | Removes existing entry for args_hash |
| 510 | Appends new entry with updated timestamp at end |

### `CellMLCache._evict_lru`

| # | Functionality |
|---|--------------|
| 511 | Removes oldest entries when over max_entries |
| 512 | Deletes cache pickle files for evicted entries |
| 513 | Ignores FileNotFoundError during deletion |

### `CellMLCache.cache_valid`

| # | Functionality |
|---|--------------|
| 514 | Returns False when file hash has changed |
| 515 | Returns False when args_hash not in entries |
| 516 | Returns False when cache pickle file missing |
| 517 | Returns True when file hash matches and pickle exists |

### `CellMLCache.load_from_cache`

| # | Functionality |
|---|--------------|
| 518 | Returns None when cache_valid is False |
| 519 | Loads pickle file and returns cached data dict |
| 520 | Updates LRU order on successful load |
| 521 | Returns None and logs error on load failure |

### `CellMLCache.save_to_cache`

| # | Functionality |
|---|--------------|
| 522 | Creates cache directory if needed |
| 523 | Serializes cache data as pickle with HIGHEST_PROTOCOL |
| 524 | Updates manifest file_hash |
| 525 | Updates LRU order |
| 526 | Evicts oldest entries if over limit |
| 527 | Saves updated manifest |
| 528 | Logs failure message without raising |
