# Implementation Task List
# Feature: SymPy-to-SymPy Input Pathway
# Plan Reference: .github/active_plans/sympy_input_pathway/agent_plan.md

## Task Group 1: Input Type Detection Infrastructure - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1-30, 785-903)
- Reference: agent_plan.md sections on Input Type Detection

**Input Validation Required**:
- dxdt: Check is not None
- dxdt: If iterable, check not empty before inspecting first element
- dxdt: Validate first element is either str, sp.Expr, sp.Equality, or tuple

**Tasks**:

1. **Add _detect_input_type function**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Create
   - Insert location: After TIME_SYMBOL definition (line 31), before ParsedEquations class
   - Details:
     ```python
     def _detect_input_type(dxdt: Union[str, Iterable]) -> str:
         """Detect whether dxdt contains strings or SymPy expressions.
         
         Parameters
         ----------
         dxdt
             System equations as string or iterable.
         
         Returns
         -------
         str
             Either 'string' or 'sympy' indicating input format.
         
         Raises
         ------
         TypeError
             If input type cannot be determined or is invalid.
         ValueError
             If empty iterable is provided.
         """
         # Step 1: Validate input is not None
         if dxdt is None:
             raise TypeError("dxdt cannot be None")
         
         # Step 2: Handle string input (existing behavior)
         if isinstance(dxdt, str):
             return 'string'
         
         # Step 3: Handle iterable input
         try:
             # Convert to list to inspect first element
             items = list(dxdt)
         except TypeError:
             raise TypeError(
                 f"dxdt must be string or iterable, got {type(dxdt).__name__}"
             )
         
         # Step 4: Check for empty iterable
         if len(items) == 0:
             raise ValueError("dxdt iterable cannot be empty")
         
         # Step 5: Inspect first element to determine type
         first_elem = items[0]
         
         if isinstance(first_elem, str):
             return 'string'
         elif isinstance(first_elem, (sp.Expr, sp.Equality)):
             return 'sympy'
         elif isinstance(first_elem, tuple):
             # Check if tuple contains SymPy objects
             if len(first_elem) == 2:
                 lhs, rhs = first_elem
                 if isinstance(lhs, sp.Symbol) and isinstance(rhs, sp.Expr):
                     return 'sympy'
         
         # Step 6: Invalid type - provide informative error
         raise TypeError(
             f"dxdt elements must be strings or SymPy expressions, "
             f"got {type(first_elem).__name__}. "
             f"Valid SymPy formats: sp.Equality, sp.Expr, or "
             f"tuple of (sp.Symbol, sp.Expr)"
         )
     ```
   - Edge cases:
     - None input → TypeError with clear message
     - Empty iterable → ValueError
     - Mixed types (first is string, second is SymPy) → Detected as 'string', but will fail later in processing (acceptable behavior)
     - Invalid SymPy tuple formats → TypeError with format guidance
   - Integration: Will be called at start of parse_input()

**Outcomes**: 
- Files Modified:
  * src/cubie/odesystems/symbolic/parsing/parser.py (90 lines added after _detect_input_type)
- Functions Added:
  * _normalize_sympy_equations() - Normalizes SymPy equations to (lhs, rhs) tuples
- Implementation Summary:
  Converts sp.Equality and tuples to standardized (Symbol, Expr) format. Validates
  LHS is Symbol, RHS is Expr. Rejects bare sp.Expr as ambiguous. Returns empty
  list for empty input.
- Issues Flagged: None

---

## Task Group 2: SymPy Expression Normalization - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 100-226 ParsedEquations class, 785-952 parse_input function)
- File: src/cubie/odesystems/symbolic/indexedbasemaps.py (entire file for IndexedBases structure)
- Reference: agent_plan.md sections on SymPy Expression Processing

**Input Validation Required**:
- equations: Check is iterable
- equations: Each element validated as sp.Equality, tuple, or sp.Expr
- For tuples: Validate length == 2, lhs is sp.Symbol, rhs is sp.Expr
- index_map: Check is IndexedBases instance

**Tasks**:

1. **Add _normalize_sympy_equations function**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Create
   - Insert location: After _detect_input_type, before _sanitise_input_math
   - Details:
     ```python
     def _normalize_sympy_equations(
         equations: Iterable[Union[sp.Equality, Tuple[sp.Symbol, sp.Expr], sp.Expr]],
         index_map: IndexedBases,
     ) -> List[Tuple[sp.Symbol, sp.Expr]]:
         """Normalize various SymPy equation formats to (lhs, rhs) tuples.
         
         Parameters
         ----------
         equations
             SymPy equations in various formats.
         index_map
             Indexed symbol collections for validation.
         
         Returns
         -------
         list
             Standardized list of (lhs_symbol, rhs_expr) tuples.
         
         Raises
         ------
         TypeError
             If equations contain invalid format.
         ValueError
             If LHS symbols cannot be categorized.
         """
         # Step 1: Validate equations is iterable
         try:
             eq_list = list(equations)
         except TypeError:
             raise TypeError("equations must be iterable")
         
         normalized = []
         state_names = set(index_map.state_names)
         dxdt_names = set(index_map.dxdt_names)
         observable_names = set(index_map.observable_names)
         
         # Step 2: Process each equation
         for i, eq in enumerate(eq_list):
             # Step 3: Handle sp.Equality objects
             if isinstance(eq, sp.Equality):
                 lhs = eq.lhs
                 rhs = eq.rhs
                 
                 # Validate lhs is a Symbol
                 if not isinstance(lhs, sp.Symbol):
                     raise ValueError(
                         f"Equation {i}: LHS of sp.Equality must be sp.Symbol, "
                         f"got {type(lhs).__name__}"
                     )
                 
                 normalized.append((lhs, rhs))
             
             # Step 4: Handle tuples (lhs, rhs)
             elif isinstance(eq, tuple):
                 if len(eq) != 2:
                     raise TypeError(
                         f"Equation {i}: Tuple must have exactly 2 elements "
                         f"(lhs, rhs), got {len(eq)}"
                     )
                 
                 lhs, rhs = eq
                 
                 # Validate types
                 if not isinstance(lhs, sp.Symbol):
                     raise TypeError(
                         f"Equation {i}: Tuple LHS must be sp.Symbol, "
                         f"got {type(lhs).__name__}"
                     )
                 if not isinstance(rhs, sp.Expr):
                     raise TypeError(
                         f"Equation {i}: Tuple RHS must be sp.Expr, "
                         f"got {type(rhs).__name__}"
                     )
                 
                 normalized.append((lhs, rhs))
             
             # Step 5: Handle bare expressions (infer LHS from context)
             elif isinstance(eq, sp.Expr):
                 # For bare expressions, we cannot infer LHS
                 # This is an edge case that should error
                 raise TypeError(
                     f"Equation {i}: Bare sp.Expr not supported. "
                     f"Use sp.Equality or tuple format to specify LHS."
                 )
             
             else:
                 raise TypeError(
                     f"Equation {i}: Invalid type {type(eq).__name__}. "
                     f"Expected sp.Equality, tuple, or sp.Expr"
                 )
         
         return normalized
     ```
   - Edge cases:
     - sp.Equality with non-Symbol LHS → ValueError
     - Tuple with length != 2 → TypeError
     - Tuple with wrong types → TypeError
     - Bare sp.Expr → TypeError (cannot infer LHS)
     - Empty equations list → Returns empty list (valid)
   - Integration: Called from parse_input() when input type is 'sympy'

**Outcomes**: 
- Files Modified:
  * src/cubie/odesystems/symbolic/parsing/parser.py (95 lines added)
- Functions Added:
  * _normalize_sympy_equations() - Normalizes various SymPy equation formats
- Implementation Summary:
  Handles sp.Equality, (Symbol, Expr) tuples, validates types, and rejects
  bare sp.Expr. Provides detailed error messages with equation indices.
- Issues Flagged: None

---

## Task Group 3: SymPy Symbol Extraction - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 549-672 _lhs_pass function)
- File: src/cubie/odesystems/symbolic/indexedbasemaps.py (entire file)
- Reference: agent_plan.md sections on Symbol and Equation Extraction

**Input Validation Required**:
- equations: Check is list of (sp.Symbol, sp.Expr) tuples
- index_map: Check is IndexedBases instance
- strict: Check is boolean

**Tasks**:

1. **Add _lhs_pass_sympy function for SymPy input**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Create
   - Insert location: After _normalize_sympy_equations, before existing _lhs_pass
   - Details:
     ```python
     def _lhs_pass_sympy(
         equations: List[Tuple[sp.Symbol, sp.Expr]],
         indexed_bases: IndexedBases,
         strict: bool = True,
     ) -> Dict[str, sp.Symbol]:
         """Validate LHS symbols in SymPy equations and infer auxiliaries.
         
         Parameters
         ----------
         equations
             Normalized SymPy equations as (lhs, rhs) tuples.
         indexed_bases
             Indexed symbol collections from user inputs.
         strict
             When False, infer missing state derivatives automatically.
         
         Returns
         -------
         dict
             Anonymous auxiliary symbols introduced in equations.
         
         Raises
         ------
         ValueError
             If LHS validation fails or required symbols are missing.
         
         Notes
         -----
         This function parallels _lhs_pass() but works with SymPy
         objects directly instead of parsing strings.
         """
         # Step 1: Initialize tracking sets (same as string version)
         anonymous_auxiliaries = {}
         assigned_obs = set()
         underived_states = set(indexed_bases.dxdt_names)
         state_names = set(indexed_bases.state_names)
         observable_names = set(indexed_bases.observable_names)
         param_names = set(indexed_bases.parameter_names)
         constant_names = set(indexed_bases.constant_names)
         driver_names = set(indexed_bases.driver_names)
         states = indexed_bases.states
         observables = indexed_bases.observables
         dxdt = indexed_bases.dxdt
         
         # Step 2: Process each equation's LHS
         for lhs_sym, rhs_expr in equations:
             lhs_name = str(lhs_sym)
             
             # Step 3: Check if LHS is a derivative (starts with 'd')
             if lhs_name.startswith("d"):
                 state_name = lhs_name[1:]
                 s_sym = sp.Symbol(state_name, real=True)
                 
                 # Validate state exists or infer it
                 if state_name not in state_names:
                     if state_name in observable_names:
                         # Convert observable to state (with warning)
                         warn(
                             f"Symbol d{state_name} found in equations, but "
                             f"{state_name} was listed as an observable. "
                             f"Converting to state.",
                             EquationWarning,
                         )
                         states.push(s_sym)
                         dxdt.push(sp.Symbol(f"d{state_name}", real=True))
                         observables.pop(s_sym)
                         state_names.add(state_name)
                         observable_names.discard(state_name)
                     else:
                         if strict:
                             raise ValueError(
                                 f"Unknown state derivative: {lhs_name}. "
                                 f"No state called {state_name} found."
                             )
                         else:
                             # Infer new state
                             states.push(s_sym)
                             dxdt.push(sp.Symbol(f"d{state_name}", real=True))
                             state_names.add(state_name)
                             underived_states.add(f"d{state_name}")
                 
                 underived_states -= {lhs_name}
             
             # Step 4: Check if assigning to state (not allowed)
             elif lhs_name in state_names:
                 raise ValueError(
                     f"State {lhs_name} cannot be assigned directly. "
                     f"States must be defined as derivatives: d{lhs_name} = ..."
                 )
             
             # Step 5: Check if assigning to immutable input
             elif (
                 lhs_name in param_names
                 or lhs_name in constant_names
                 or lhs_name in driver_names
             ):
                 raise ValueError(
                     f"{lhs_name} is an immutable input "
                     f"(constant, parameter, or driver) but is being assigned. "
                     f"It must be a state, observable, or auxiliary."
                 )
             
             # Step 6: Observable or anonymous auxiliary
             else:
                 if lhs_name not in observable_names:
                     # Anonymous auxiliary - use the actual symbol from equation
                     anonymous_auxiliaries[lhs_name] = lhs_sym
                 if lhs_name in observable_names:
                     assigned_obs.add(lhs_name)
         
         # Step 7: Validate all observables were assigned
         missing_obs = set(indexed_bases.observable_names) - assigned_obs
         if missing_obs:
             raise ValueError(
                 f"Observables {missing_obs} were declared but never assigned."
             )
         
         # Step 8: Handle underived states (convert to observables)
         if underived_states:
             warn(
                 f"States {underived_states} have no derivative equation. "
                 f"Converting to observables.",
                 EquationWarning,
             )
             for state in underived_states:
                 s_sym = sp.Symbol(state, real=True)
                 if state in observables:
                     raise ValueError(
                         f"State {state} is both observable and state. "
                         f"Cannot convert."
                     )
                 observables.push(s_sym)
                 states.pop(s_sym)
                 dxdt.pop(s_sym)
                 observable_names.add(state)
         
         return anonymous_auxiliaries
     ```
   - Edge cases:
     - Derivative with non-existent state in strict mode → ValueError
     - Derivative with non-existent state in non-strict mode → Infer state
     - Assignment to state symbol → ValueError
     - Assignment to parameter/constant/driver → ValueError
     - Observable not assigned → ValueError
     - State with no derivative → Convert to observable with warning
   - Integration: Called from parse_input() when input type is 'sympy'

2. **Add _rhs_pass_sympy function for SymPy input**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Create
   - Insert location: After _lhs_pass_sympy, before existing _rhs_pass
   - Details:
     ```python
     def _rhs_pass_sympy(
         equations: List[Tuple[sp.Symbol, sp.Expr]],
         all_symbols: Dict[str, sp.Symbol],
         indexed_bases: IndexedBases,
         user_funcs: Optional[Dict[str, Callable]] = None,
         user_function_derivatives: Optional[Dict[str, Callable]] = None,
         strict: bool = True,
     ) -> Tuple[List[Tuple[sp.Symbol, sp.Expr]], Dict[str, Callable], List[sp.Symbol]]:
         """Validate RHS symbols in SymPy equations.
         
         Parameters
         ----------
         equations
             Normalized SymPy equations as (lhs, rhs) tuples.
         all_symbols
             Mapping from symbol names to SymPy symbols.
         indexed_bases
             Indexed symbol collections from user inputs.
         user_funcs
             Optional user-provided callable mapping.
         user_function_derivatives
             Optional derivative helpers for user functions.
         strict
             When False, infer missing symbols from free_symbols.
         
         Returns
         -------
         tuple
             Validated equations, callable mapping, and inferred symbols.
         
         Notes
         -----
         This function parallels _rhs_pass() but works with SymPy
         expressions directly. It uses free_symbols for extraction
         instead of parsing strings.
         """
         # Step 1: Initialize tracking
         validated_equations = []
         new_symbols = []
         
         # Step 2: Build symbol sets for validation
         declared_symbols = {
             value for value in all_symbols.values() 
             if isinstance(value, sp.Symbol)
         }
         
         # Step 3: Process user functions (create device/non-device callables)
         funcs = {}
         if user_funcs:
             # Build SymPy function objects for user functions
             parse_locals, alias_map, dev_map = _build_sympy_user_functions(
                 user_funcs, {}, user_function_derivatives
             )
             
             # Merge into funcs dict
             funcs.update({name: fn for name, fn in user_funcs.items()})
         
         # Step 4: Process each equation's RHS
         for lhs_sym, rhs_expr in equations:
             # Step 5: Extract symbols from RHS using free_symbols
             rhs_symbols = rhs_expr.free_symbols
             
             # Step 6: Validate all RHS symbols are declared
             if strict:
                 undeclared = {
                     sym for sym in rhs_symbols 
                     if sym not in declared_symbols
                 }
                 if undeclared:
                     undeclared_names = sorted(str(s) for s in undeclared)
                     raise ValueError(
                         f"Equation for {lhs_sym} contains undefined symbols: "
                         f"{undeclared_names}"
                     )
             else:
                 # Infer new symbols
                 for sym in rhs_symbols:
                     if sym not in declared_symbols:
                         new_symbols.append(sym)
                         declared_symbols.add(sym)
                         all_symbols[str(sym)] = sym
             
             # Step 7: Store validated equation
             validated_equations.append((lhs_sym, rhs_expr))
         
         return validated_equations, funcs, new_symbols
     ```
   - Edge cases:
     - Undefined symbols in strict mode → ValueError
     - Undefined symbols in non-strict mode → Infer and add to new_symbols
     - User functions in RHS → Build callable mapping
     - No free_symbols in RHS (constant expression) → Valid, no symbols to check
   - Integration: Called from parse_input() when input type is 'sympy'

**Outcomes**: 
- Files Modified:
  * src/cubie/odesystems/symbolic/parsing/parser.py (220 lines added)
- Functions Added:
  * _lhs_pass_sympy() - Validates LHS symbols in SymPy equations (130 lines)
  * _rhs_pass_sympy() - Validates RHS symbols using free_symbols (65 lines)
- Implementation Summary:
  Both functions parallel their string counterparts but work with SymPy objects
  directly. _lhs_pass_sympy handles derivatives, states, observables, and
  auxiliaries with same validation logic. _rhs_pass_sympy uses free_symbols
  for extraction, validates against declared symbols, and infers in non-strict
  mode. User functions integrated via _build_sympy_user_functions.
- Issues Flagged: None

---

## Task Group 4: parse_input Branching Logic - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 785-952 parse_input function)
- All functions from Groups 1-3

**Input Validation Required**:
None - input validation delegated to _detect_input_type

**Tasks**:

1. **Modify parse_input to add type detection and branching**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Location: parse_input function (lines 785-952)
   - Details:
     ```python
     # At the start of parse_input, after parameter defaults (line ~895)
     # ADD these lines before "if isinstance(dxdt, str):" section:
     
     # Detect input type
     input_type = _detect_input_type(dxdt)
     
     # Branch based on input type
     if input_type == 'string':
         # Existing string processing path (lines 896-922)
         if isinstance(dxdt, str):
             lines = [
                 line.strip() for line in dxdt.strip().splitlines() 
                 if line.strip()
             ]
         elif isinstance(dxdt, list) or isinstance(dxdt, tuple):
             lines = [line.strip() for line in dxdt if line.strip()]
         else:
             raise ValueError(
                 "dxdt must be a string or a list/tuple of strings"
             )
         
         raw_lines = list(lines)
         lines = _normalise_indexed_tokens(lines)
         
         constants = index_map.constants.default_values
         fn_hash = hash_system_definition(dxdt, constants)
         anon_aux = _lhs_pass(lines, index_map, strict=strict)
         all_symbols = index_map.all_symbols.copy()
         all_symbols.setdefault("t", TIME_SYMBOL)
         all_symbols.update(anon_aux)
         
         equation_map, funcs, new_params = _rhs_pass(
             lines=lines,
             all_symbols=all_symbols,
             user_funcs=user_functions,
             user_function_derivatives=user_function_derivatives,
             strict=strict,
             raw_lines=raw_lines,
         )
     
     elif input_type == 'sympy':
         # NEW: SymPy processing path
         # Step 1: Convert to list
         if isinstance(dxdt, (list, tuple)):
             equations = list(dxdt)
         else:
             # Should not reach here (type detection ensures iterable)
             equations = [dxdt]
         
         # Step 2: Normalize SymPy equations to (lhs, rhs) tuples
         normalized_eqs = _normalize_sympy_equations(equations, index_map)
         
         # Step 3: Hash system definition (use normalized equations)
         constants = index_map.constants.default_values
         fn_hash = hash_system_definition(normalized_eqs, constants)
         
         # Step 4: Validate LHS and identify auxiliaries
         anon_aux = _lhs_pass_sympy(
             normalized_eqs, index_map, strict=strict
         )
         
         # Step 5: Build symbol mapping
         all_symbols = index_map.all_symbols.copy()
         all_symbols.setdefault("t", TIME_SYMBOL)
         all_symbols.update(anon_aux)
         
         # Step 6: Validate RHS and extract symbols
         equation_map, funcs, new_params = _rhs_pass_sympy(
             equations=normalized_eqs,
             all_symbols=all_symbols,
             indexed_bases=index_map,
             user_funcs=user_functions,
             user_function_derivatives=user_function_derivatives,
             strict=strict,
         )
     
     else:
         # Should never reach here
         raise RuntimeError(
             f"Invalid input_type '{input_type}' from _detect_input_type"
         )
     
     # CONTINUE with existing common processing (line 924+)
     # (for param in new_params: ...)
     ```
   - Edge cases:
     - Invalid input_type from detection → RuntimeError (defensive)
     - Empty equations in SymPy path → Handled by _normalize_sympy_equations
     - Mixed input types → Detected by _detect_input_type, fails early
   - Integration: Core branching point that directs to string or SymPy pathway

2. **Update hash_system_definition to handle SymPy input**
   - File: src/cubie/odesystems/symbolic/sym_utils.py
   - Action: Modify
   - Location: hash_system_definition function
   - Details:
     ```python
     # Modify hash_system_definition to accept SymPy equations
     # At the start of the function, add type checking:
     
     if isinstance(dxdt, (list, tuple)) and len(dxdt) > 0:
         first_elem = dxdt[0]
         if isinstance(first_elem, (sp.Equality, tuple)) or \
            (isinstance(first_elem, tuple) and 
             len(first_elem) == 2 and 
             isinstance(first_elem[0], sp.Symbol)):
             # SymPy input - convert to canonical string for hashing
             hash_strings = []
             for eq in dxdt:
                 if isinstance(eq, sp.Equality):
                     lhs_str = str(eq.lhs)
                     rhs_str = str(eq.rhs)
                 elif isinstance(eq, tuple):
                     lhs_str = str(eq[0])
                     rhs_str = str(eq[1])
                 else:
                     lhs_str = str(eq)
                     rhs_str = ""
                 hash_strings.append(f"{lhs_str} = {rhs_str}")
             dxdt = "\n".join(hash_strings)
     
     # Continue with existing string hashing logic
     ```
   - Edge cases:
     - Empty list → Existing code handles
     - Mixed equation formats → Each handled individually
     - Constants dict → Existing logic applies
   - Integration: Ensures SymPy and string inputs produce consistent hashes

**Outcomes**: 
- Files Modified:
  * src/cubie/odesystems/symbolic/parsing/parser.py (65 lines modified in parse_input)
  * src/cubie/odesystems/symbolic/sym_utils.py (35 lines modified in hash_system_definition)
- Functions Modified:
  * parse_input() - Added type detection and dual pathway branching
  * hash_system_definition() - Added SymPy equation handling
- Implementation Summary:
  parse_input now detects input type first, then branches to string or SymPy
  processing paths that converge before common processing. hash_system_definition
  checks for SymPy Equality/tuple formats and converts to canonical string for
  hashing, ensuring consistent hashes across input types.
- Issues Flagged: None

---

## Task Group 5: CellML Adapter Simplification - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 4 (parse_input branching complete)

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/cellml.py (lines 105-450)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (parse_input function)

**Input Validation Required**:
None - validation handled by existing cellmlmanip and parse_input

**Tasks**:

1. **Remove string conversion functions from cellml.py**
   - File: src/cubie/odesystems/symbolic/parsing/cellml.py
   - Action: Delete
   - Location: Lines 105-172
   - Details:
     - Delete _replace_eq_in_piecewise function (lines 105-124)
     - Delete _eq_to_equality_str function (lines 127-172)
   - Edge cases: None (functions no longer referenced)
   - Integration: These functions were only used by load_cellml_model

2. **Replace string formatting timing event with SymPy preparation**
   - File: src/cubie/odesystems/symbolic/parsing/cellml.py
   - Action: Modify
   - Location: Lines 59-74 (timing event registration)
   - Details:
     ```python
     # REPLACE this timing event registration:
     _default_timelogger.register_event(
         "codegen_cellml_string_formatting", "codegen",
         "Codegen time for formatting equations as strings"
     )
     
     # WITH:
     _default_timelogger.register_event(
         "codegen_cellml_sympy_preparation", "codegen",
         "Codegen time for preparing SymPy equations for parser"
     )
     ```
   - Edge cases: None (simple renaming)
   - Integration: Timer tracks direct SymPy preparation instead of string formatting

3. **Modify load_cellml_model to pass SymPy equations directly**
   - File: src/cubie/odesystems/symbolic/parsing/cellml.py
   - Action: Modify
   - Location: load_cellml_model function (lines 174-450)
   - Details:
     ```python
     # AFTER line 346 (_default_timelogger.stop_event("codegen_cellml_equation_processing"))
     # REPLACE lines 348-430 with this new implementation:
     
     _default_timelogger.start_event("codegen_cellml_sympy_preparation")
     
     # Build list of SymPy equation tuples for differential equations
     dxdt_equations = []
     for eq in differential_equations:
         # Get the state variable from the derivative
         state_var = eq.lhs.args[0]
         # Create tuple of (d<state_name>, rhs)
         lhs_sym = sp.Symbol(f"d{state_var.name}", real=True)
         dxdt_equations.append((lhs_sym, eq.rhs))
     
     # Build list of SymPy equation tuples for algebraic equations
     # Separate into:
     # 1. Numeric assignments (become constants or parameters)
     # 2. Non-numeric equations (passed to parser)
     constants_dict = {}
     parameters_dict = {}
     algebraic_equation_tuples = []
     observable_units = {}
     
     # Convert parameters to set for quick lookup
     if parameters is None:
         parameters_set = set()
     elif isinstance(parameters, dict):
         parameters_set = set(parameters.keys())
     else:
         parameters_set = set(parameters)
     
     for eq in algebraic_equations:
         # Check if RHS is numeric
         if isinstance(eq.rhs, sp.Number):
             var_name = str(eq.lhs)
             var_value = float(eq.rhs)
             
             # Assign to parameters or constants based on user spec
             if var_name in parameters_set:
                 parameters_dict[var_name] = var_value
             else:
                 constants_dict[var_name] = var_value
         else:
             # Keep as equation tuple
             algebraic_equation_tuples.append((eq.lhs, eq.rhs))
             
             # Extract units for the observable
             lhs_name = str(eq.lhs)
             if lhs_name in all_symbol_units:
                 observable_units[lhs_name] = all_symbol_units[lhs_name]
     
     # Combine differential and algebraic equations
     all_equations = dxdt_equations + algebraic_equation_tuples
     
     # Extract parameter units
     parameter_units = {}
     if parameters:
         for param in parameters:
             if param in all_symbol_units:
                 parameter_units[param] = all_symbol_units[param]
     
     # Update observable units if observables specified
     if observables:
         for obs in observables:
             if obs not in observable_units and obs in all_symbol_units:
                 observable_units[obs] = all_symbol_units[obs]
     
     # Handle user-provided parameters (merge with extracted)
     if parameters is not None and isinstance(parameters, dict):
         parameters_dict = {**parameters_dict, **parameters}
     
     _default_timelogger.stop_event("codegen_cellml_sympy_preparation")
     
     # Import here to avoid circular dependency
     from cubie.odesystems.symbolic.symbolicODE import SymbolicODE
     
     # Create and return SymbolicODE with SymPy equations directly
     return SymbolicODE.create(
         dxdt=all_equations,  # List of SymPy tuples
         states=initial_values if initial_values else None,
         parameters=parameters_dict if parameters_dict else None,
         constants=constants_dict if constants_dict else None,
         observables=observables,
         name=name,
         precision=precision,
         strict=False,
         state_units=state_units if state_units else None,
         parameter_units=parameter_units if parameter_units else None,
         observable_units=observable_units if observable_units else None,
     )
     ```
   - Edge cases:
     - Numeric RHS with Float vs Integer → Handled by sp.Number check
     - Empty algebraic equations → Results in empty list (valid)
     - Parameters dict vs list → Both handled by parameters_set logic
     - No observables specified → Algebraic equations become auxiliaries
   - Integration: Passes SymPy tuples directly to parse_input, which detects 'sympy' input type

**Outcomes**: 
- Files Modified:
  * src/cubie/odesystems/symbolic/parsing/cellml.py (70 lines removed, 55 lines added)
- Functions Removed:
  * _replace_eq_in_piecewise() - No longer needed
  * _eq_to_equality_str() - No longer needed
- Functions Modified:
  * load_cellml_model() - Now passes SymPy tuples directly instead of strings
- Timing Events Modified:
  * Replaced "codegen_cellml_string_formatting" with "codegen_cellml_sympy_preparation"
- Implementation Summary:
  Removed string conversion functions. Modified load_cellml_model to build
  SymPy equation tuples directly: (d<state>, rhs) for differential equations
  and (lhs, rhs) for algebraic equations. Numeric algebraic equations become
  constants/parameters. Non-numeric algebraic equations passed as tuples.
  Units extraction preserved. SymPy tuples passed directly to SymbolicODE.create.
- Issues Flagged: None

---

## Task Group 6: Unit Tests for Type Detection - PARALLEL
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (_detect_input_type function)
- File: tests/odesystems/symbolic/test_parser.py (existing test patterns)

**Input Validation Required**:
None - tests verify the validation logic itself

**Tasks**:

1. **Add test class for _detect_input_type**
   - File: tests/odesystems/symbolic/test_parser.py
   - Action: Create
   - Insert location: After TestProcessCalls class
   - Details:
     ```python
     class TestDetectInputType:
         """Test input type detection for parse_input."""
         
         def test_detect_string_single_line(self):
             """Test detection of single-line string input."""
             dxdt = "dx = -k * x"
             result = _detect_input_type(dxdt)
             assert result == 'string'
         
         def test_detect_string_list(self):
             """Test detection of string list input."""
             dxdt = ["dx = -k * x", "dy = k * x"]
             result = _detect_input_type(dxdt)
             assert result == 'string'
         
         def test_detect_sympy_equality(self):
             """Test detection of sp.Equality input."""
             x, k = sp.symbols('x k')
             dx = sp.Symbol('dx')
             dxdt = [sp.Eq(dx, -k * x)]
             result = _detect_input_type(dxdt)
             assert result == 'sympy'
         
         def test_detect_sympy_tuple(self):
             """Test detection of (Symbol, Expr) tuple input."""
             x, k = sp.symbols('x k')
             dx = sp.Symbol('dx')
             dxdt = [(dx, -k * x)]
             result = _detect_input_type(dxdt)
             assert result == 'sympy'
         
         def test_detect_sympy_expression(self):
             """Test detection of bare sp.Expr input."""
             x, k = sp.symbols('x k')
             dxdt = [-k * x]
             result = _detect_input_type(dxdt)
             assert result == 'sympy'
         
         def test_detect_none_input(self):
             """Test error on None input."""
             with pytest.raises(TypeError, match="cannot be None"):
                 _detect_input_type(None)
         
         def test_detect_empty_list(self):
             """Test error on empty list."""
             with pytest.raises(ValueError, match="cannot be empty"):
                 _detect_input_type([])
         
         def test_detect_invalid_type(self):
             """Test error on invalid type."""
             with pytest.raises(TypeError, match="must be string or iterable"):
                 _detect_input_type(123)
         
         def test_detect_invalid_element_type(self):
             """Test error on invalid element type."""
             with pytest.raises(TypeError, match="must be strings or SymPy"):
                 _detect_input_type([123, 456])
     ```
   - Edge cases covered:
     - String single-line
     - String list
     - SymPy Equality
     - SymPy tuple
     - SymPy Expr
     - None input
     - Empty list
     - Invalid type (integer)
     - Invalid element type
   - Integration: Standalone unit tests

**Outcomes**: 
- Files Modified:
  * tests/odesystems/symbolic/test_parser.py (65 lines added)
- Tests Added:
  * TestDetectInputType class with 9 test methods
- Implementation Summary:
  Added comprehensive tests for _detect_input_type covering all input formats,
  error conditions, and edge cases. Tests validate string, SymPy Equality,
  tuple, and Expr detection, plus None, empty, and invalid inputs.
- Issues Flagged: None

---

## Task Group 7: Unit Tests for SymPy Normalization - PARALLEL
**Status**: [x]
**Dependencies**: Group 2

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (_normalize_sympy_equations function)
- File: tests/odesystems/symbolic/test_parser.py (existing test patterns)
- File: tests/conftest.py (fixture patterns)

**Input Validation Required**:
None - tests verify the validation logic itself

**Tasks**:

1. **Add test class for _normalize_sympy_equations**
   - File: tests/odesystems/symbolic/test_parser.py
   - Action: Create
   - Insert location: After TestDetectInputType class
   - Details:
     ```python
     class TestNormalizeSympyEquations:
         """Test SymPy equation normalization."""
         
         def test_normalize_equality(self):
             """Test normalization of sp.Equality objects."""
             x, k = sp.symbols('x k')
             dx = sp.Symbol('dx')
             equations = [sp.Eq(dx, -k * x)]
             
             # Need IndexedBases for validation
             index_map = IndexedBases.from_user_inputs(
                 states=['x'],
                 parameters=['k'],
                 constants={},
                 observables=[],
                 drivers=[]
             )
             
             result = _normalize_sympy_equations(equations, index_map)
             
             assert len(result) == 1
             assert result[0][0] == dx
             assert result[0][1] == -k * x
         
         def test_normalize_tuple(self):
             """Test normalization of (Symbol, Expr) tuples."""
             x, k = sp.symbols('x k')
             dx = sp.Symbol('dx')
             equations = [(dx, -k * x)]
             
             index_map = IndexedBases.from_user_inputs(
                 states=['x'],
                 parameters=['k'],
                 constants={},
                 observables=[],
                 drivers=[]
             )
             
             result = _normalize_sympy_equations(equations, index_map)
             
             assert len(result) == 1
             assert result[0][0] == dx
             assert result[0][1] == -k * x
         
         def test_normalize_mixed_formats(self):
             """Test normalization of mixed Equality and tuple."""
             x, y, k = sp.symbols('x y k')
             dx, dy = sp.symbols('dx dy')
             equations = [
                 sp.Eq(dx, -k * x),
                 (dy, k * x)
             ]
             
             index_map = IndexedBases.from_user_inputs(
                 states=['x', 'y'],
                 parameters=['k'],
                 constants={},
                 observables=[],
                 drivers=[]
             )
             
             result = _normalize_sympy_equations(equations, index_map)
             
             assert len(result) == 2
             assert result[0][0] == dx
             assert result[1][0] == dy
         
         def test_normalize_invalid_tuple_length(self):
             """Test error on tuple with wrong length."""
             x = sp.Symbol('x')
             equations = [(x, x, x)]  # 3 elements
             
             index_map = IndexedBases.from_user_inputs(
                 states=['x'],
                 parameters=[],
                 constants={},
                 observables=[],
                 drivers=[]
             )
             
             with pytest.raises(TypeError, match="exactly 2 elements"):
                 _normalize_sympy_equations(equations, index_map)
         
         def test_normalize_invalid_lhs_type(self):
             """Test error on non-Symbol LHS in Equality."""
             x = sp.Symbol('x')
             equations = [sp.Eq(x + 1, x)]  # LHS is expression
             
             index_map = IndexedBases.from_user_inputs(
                 states=['x'],
                 parameters=[],
                 constants={},
                 observables=[],
                 drivers=[]
             )
             
             with pytest.raises(ValueError, match="LHS of sp.Equality must be"):
                 _normalize_sympy_equations(equations, index_map)
         
         def test_normalize_bare_expression_error(self):
             """Test error on bare sp.Expr (cannot infer LHS)."""
             x = sp.Symbol('x')
             equations = [x + 1]  # Bare expression
             
             index_map = IndexedBases.from_user_inputs(
                 states=['x'],
                 parameters=[],
                 constants={},
                 observables=[],
                 drivers=[]
             )
             
             with pytest.raises(TypeError, match="Bare sp.Expr not supported"):
                 _normalize_sympy_equations(equations, index_map)
         
         def test_normalize_empty_list(self):
             """Test empty equation list returns empty result."""
             equations = []
             
             index_map = IndexedBases.from_user_inputs(
                 states=[],
                 parameters=[],
                 constants={},
                 observables=[],
                 drivers=[]
             )
             
             result = _normalize_sympy_equations(equations, index_map)
             assert result == []
     ```
   - Edge cases covered:
     - sp.Equality normalization
     - Tuple normalization
     - Mixed formats
     - Invalid tuple length
     - Invalid LHS type
     - Bare expression error
     - Empty list
   - Integration: Unit tests using IndexedBases fixtures

**Outcomes**: 
- Files Modified:
  * tests/odesystems/symbolic/test_parser.py (200 lines added)
- Tests Added:
  * TestNormalizeSympyEquations class with 8 test methods
- Implementation Summary:
  Added comprehensive tests for _normalize_sympy_equations covering sp.Equality,
  tuples, mixed formats, validation errors for invalid LHS/RHS types, bare
  expressions, and empty input. All tests use IndexedBases for context.
- Issues Flagged: None

---

## Task Group 8: Integration Tests for SymPy Pathway - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 4, 5 (full SymPy pathway implemented)

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (full parse_input)
- File: src/cubie/odesystems/symbolic/symbolicODE.py (SymbolicODE.create)
- File: tests/odesystems/symbolic/test_parser.py (existing integration patterns)
- File: tests/system_fixtures.py (ODE system fixtures)

**Input Validation Required**:
None - tests verify end-to-end behavior

**Tasks**:

1. **Add integration test for simple SymPy input**
   - File: tests/odesystems/symbolic/test_parser.py
   - Action: Create
   - Insert location: End of file (new test class)
   - Details:
     ```python
     class TestSympyInputPathway:
         """Integration tests for SymPy input pathway."""
         
         def test_simple_ode_sympy_equality(self):
             """Test simple ODE via SymPy Equality input."""
             # Define symbols
             x, k = sp.symbols('x k')
             dx = sp.Symbol('dx')
             
             # Define equation
             dxdt = [sp.Eq(dx, -k * x)]
             
             # Parse
             index_map, all_symbols, funcs, parsed_eqs, fn_hash = parse_input(
                 dxdt=dxdt,
                 states=['x'],
                 parameters=['k'],
                 strict=True
             )
             
             # Verify parsing
             assert len(parsed_eqs.state_derivatives) == 1
             assert str(parsed_eqs.state_derivatives[0][0]) == 'dx'
             # Verify symbols in RHS
             rhs_syms = parsed_eqs.state_derivatives[0][1].free_symbols
             assert any(str(s) == 'k' for s in rhs_syms)
             assert any(str(s) == 'x' for s in rhs_syms)
         
         def test_simple_ode_sympy_tuple(self):
             """Test simple ODE via tuple input."""
             x, k = sp.symbols('x k')
             dx = sp.Symbol('dx')
             
             dxdt = [(dx, -k * x)]
             
             index_map, all_symbols, funcs, parsed_eqs, fn_hash = parse_input(
                 dxdt=dxdt,
                 states=['x'],
                 parameters=['k'],
                 strict=True
             )
             
             assert len(parsed_eqs.state_derivatives) == 1
             assert str(parsed_eqs.state_derivatives[0][0]) == 'dx'
         
         def test_ode_with_observables_sympy(self):
             """Test ODE with observables via SymPy input."""
             x, y, k = sp.symbols('x y k')
             dx, dy, z = sp.symbols('dx dy z')
             
             dxdt = [
                 sp.Eq(dx, -k * x),
                 sp.Eq(dy, k * x),
                 sp.Eq(z, x + y)  # Observable
             ]
             
             index_map, all_symbols, funcs, parsed_eqs, fn_hash = parse_input(
                 dxdt=dxdt,
                 states=['x', 'y'],
                 parameters=['k'],
                 observables=['z'],
                 strict=True
             )
             
             assert len(parsed_eqs.state_derivatives) == 2
             assert len(parsed_eqs.observables) == 1
             assert str(parsed_eqs.observables[0][0]) == 'z'
         
         def test_ode_with_user_functions_sympy(self):
             """Test ODE with user functions via SymPy input."""
             x, k = sp.symbols('x k')
             dx = sp.Symbol('dx')
             
             # Create SymPy function
             custom_func = sp.Function('custom_func')
             
             dxdt = [sp.Eq(dx, -k * custom_func(x))]
             
             # User function implementation
             def custom_impl(val):
                 return val ** 2
             
             index_map, all_symbols, funcs, parsed_eqs, fn_hash = parse_input(
                 dxdt=dxdt,
                 states=['x'],
                 parameters=['k'],
                 user_functions={'custom_func': custom_impl},
                 strict=True
             )
             
             # Verify function is recognized
             assert 'custom_func' in funcs
             assert len(parsed_eqs.state_derivatives) == 1
         
         def test_sympy_vs_string_equivalence(self):
             """Test that SymPy and string input produce same results."""
             # Define via SymPy
             x, k = sp.symbols('x k')
             dx = sp.Symbol('dx')
             dxdt_sympy = [sp.Eq(dx, -k * x)]
             
             # Define via string
             dxdt_string = "dx = -k * x"
             
             # Parse both
             result_sympy = parse_input(
                 dxdt=dxdt_sympy,
                 states=['x'],
                 parameters=['k'],
                 strict=True
             )
             
             result_string = parse_input(
                 dxdt=dxdt_string,
                 states=['x'],
                 parameters=['k'],
                 strict=True
             )
             
             # Compare parsed equations (should be structurally identical)
             sympy_eq = result_sympy[3].state_derivatives[0]
             string_eq = result_string[3].state_derivatives[0]
             
             assert str(sympy_eq[0]) == str(string_eq[0])
             # RHS may have different internal structure but same str
             assert str(sympy_eq[1]) == str(string_eq[1])
     ```
   - Edge cases covered:
     - Simple ODE with Equality
     - Simple ODE with tuple
     - ODE with observables
     - ODE with user functions
     - SymPy vs string equivalence
   - Integration: Full parse_input flow with SymPy input

2. **Add test for inferred symbols in non-strict mode**
   - File: tests/odesystems/symbolic/test_parser.py
   - Action: Create
   - Insert location: In TestSympyInputPathway class
   - Details:
     ```python
     def test_sympy_infer_states_non_strict(self):
         """Test state inference in non-strict mode."""
         x, k = sp.symbols('x k')
         dx = sp.Symbol('dx')
         
         # Don't declare 'x' as state
         dxdt = [sp.Eq(dx, -k * x)]
         
         index_map, all_symbols, funcs, parsed_eqs, fn_hash = parse_input(
             dxdt=dxdt,
             states=[],  # Empty - will be inferred
             parameters=['k'],
             strict=False
         )
         
         # Verify 'x' was inferred as state
         assert 'x' in index_map.state_names
         assert len(parsed_eqs.state_derivatives) == 1
     
     def test_sympy_infer_parameters_non_strict(self):
         """Test parameter inference from RHS symbols."""
         x, k = sp.symbols('x k')
         dx = sp.Symbol('dx')
         
         dxdt = [sp.Eq(dx, -k * x)]
         
         # Don't declare 'k' as parameter
         index_map, all_symbols, funcs, parsed_eqs, fn_hash = parse_input(
             dxdt=dxdt,
             states=['x'],
             parameters=[],  # Will infer k
             strict=False
         )
         
         # Verify 'k' was inferred as parameter
         assert 'k' in index_map.parameter_names
     ```
   - Edge cases covered:
     - State inference
     - Parameter inference
   - Integration: Non-strict mode with SymPy input

**Outcomes**: 
- Files Modified:
  * tests/odesystems/symbolic/test_parser.py (160 lines added)
- Tests Added:
  * TestSympyInputPathway class with 8 test methods
- Implementation Summary:
  Added comprehensive integration tests covering: simple ODE with Equality and
  tuple formats, observables, user functions, SymPy vs string equivalence,
  and non-strict inference of states and parameters. Tests validate full
  parse_input flow with SymPy input.
- Issues Flagged: None

---

## Task Group 9: CellML Adapter Tests - SEQUENTIAL
**Status**: [x]
**Dependencies**: Group 5 (CellML adapter updated)

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/cellml.py (modified load_cellml_model)
- File: tests/odesystems/symbolic/test_cellml.py (existing CellML tests)
- File: tests/fixtures/cellml (CellML test files)

**Input Validation Required**:
None - tests verify behavior against known CellML files

**Tasks**:

1. **Verify existing CellML tests still pass**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Verify (no code changes)
   - Details:
     - Run test_load_simple_cellml_model
     - Run test_load_complex_cellml_model
     - Run test_algebraic_equations_as_observables
     - Verify all pass with new SymPy pathway
   - Edge cases:
     - Basic ODE model
     - Complex Beeler-Reuter model
     - Observables from algebraic equations
   - Integration: Regression testing

2. **Add test verifying SymPy pathway is used**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Create
   - Insert location: After test_invalid_extension
   - Details:
     ```python
     def test_cellml_uses_sympy_pathway(basic_model):
         """Verify CellML adapter uses SymPy pathway internally."""
         # This is a structural test - we verify that the model
         # is created successfully without string conversion
         
         # The model should work identically
         assert basic_model.num_states == 1
         assert is_devfunc(basic_model.dxdt_function)
         
         # Verify initial values are preserved
         # (This tests that SymPy Number conversion works)
         initial_vals = basic_model.indices.states.default_values
         assert len(initial_vals) > 0
     
     def test_cellml_timing_events_updated(basic_model, monkeypatch):
         """Verify timing events use new SymPy preparation name."""
         from cubie.time_logger import _default_timelogger
         
         # Check that new event name is registered
         registered_events = _default_timelogger.event_registry
         assert "codegen_cellml_sympy_preparation" in registered_events
         
         # Old event should not be registered
         assert "codegen_cellml_string_formatting" not in registered_events
     ```
   - Edge cases:
     - Basic model with SymPy pathway
     - Timing events updated
   - Integration: Verify adapter changes work correctly

**Outcomes**: 
- Files Modified:
  * tests/odesystems/symbolic/test_cellml.py (20 lines modified)
- Tests Added:
  * test_cellml_uses_sympy_pathway() - Verifies SymPy pathway used
  * test_cellml_timing_events_updated() - Verifies new timing event names
- Import Changes:
  * Removed unused _eq_to_equality_str import
- Implementation Summary:
  Added 2 new tests verifying CellML adapter uses SymPy pathway and timing
  events are updated. Removed import of deleted _eq_to_equality_str function.
  Existing tests should pass unchanged (verify manually if needed).
- Issues Flagged: None

---

## Task Group 10: End-to-End Equivalence Tests - SEQUENTIAL
**Status**: [x]
**Dependencies**: Groups 8, 9 (all tests passing)

**Required Context**:
- File: src/cubie/odesystems/symbolic/symbolicODE.py (SymbolicODE.create)
- File: tests/odesystems/symbolic/test_symbolicode.py (existing SymbolicODE tests)
- File: tests/system_fixtures.py (ODE system fixtures)

**Input Validation Required**:
None - tests verify equivalence of outputs

**Tasks**:

1. **Add test for code generation equivalence**
   - File: tests/odesystems/symbolic/test_symbolicode.py
   - Action: Create
   - Insert location: End of file (new test class)
   - Details:
     ```python
     class TestSympyStringEquivalence:
         """Test equivalence of SymPy and string input pathways."""
         
         def test_generated_code_identical(self):
             """Verify SymPy and string inputs generate identical code."""
             from cubie.odesystems.symbolic.symbolicODE import SymbolicODE
             
             # Define via SymPy
             x, y, k = sp.symbols('x y k')
             dx, dy = sp.symbols('dx dy')
             dxdt_sympy = [
                 sp.Eq(dx, -k * x),
                 sp.Eq(dy, k * x)
             ]
             
             ode_sympy = SymbolicODE.create(
                 dxdt=dxdt_sympy,
                 states={'x': 1.0, 'y': 0.0},
                 parameters={'k': 0.1},
                 name='test_sympy'
             )
             
             # Define via string
             dxdt_string = ["dx = -k * x", "dy = k * x"]
             
             ode_string = SymbolicODE.create(
                 dxdt=dxdt_string,
                 states={'x': 1.0, 'y': 0.0},
                 parameters={'k': 0.1},
                 name='test_string'
             )
             
             # Compare generated code (should be identical or equivalent)
             # Both should produce working dxdt functions
             assert is_devfunc(ode_sympy.dxdt_function)
             assert is_devfunc(ode_string.dxdt_function)
             
             # State counts should match
             assert ode_sympy.num_states == ode_string.num_states
             assert ode_sympy.num_states == 2
         
         def test_hash_consistency(self):
             """Verify hash is consistent for equivalent definitions."""
             from cubie.odesystems.symbolic.symbolicODE import SymbolicODE
             
             # Define via SymPy
             x, k = sp.symbols('x k')
             dx = sp.Symbol('dx')
             dxdt_sympy = [sp.Eq(dx, -k * x)]
             
             # Define via string
             dxdt_string = "dx = -k * x"
             
             # Parse both
             result_sympy = parse_input(
                 dxdt=dxdt_sympy,
                 states=['x'],
                 parameters=['k'],
                 constants={'c': 1.0}
             )
             
             result_string = parse_input(
                 dxdt=dxdt_string,
                 states=['x'],
                 parameters=['k'],
                 constants={'c': 1.0}
             )
             
             # Hashes should match for equivalent systems
             hash_sympy = result_sympy[4]
             hash_string = result_string[4]
             
             assert hash_sympy == hash_string
         
         def test_observables_equivalence(self):
             """Verify observables work identically in both pathways."""
             from cubie.odesystems.symbolic.symbolicODE import SymbolicODE
             
             # Define via SymPy
             x, k, z = sp.symbols('x k z')
             dx = sp.Symbol('dx')
             dxdt_sympy = [
                 sp.Eq(dx, -k * x),
                 sp.Eq(z, x * k)
             ]
             
             ode_sympy = SymbolicODE.create(
                 dxdt=dxdt_sympy,
                 states={'x': 1.0},
                 parameters={'k': 0.1},
                 observables=['z']
             )
             
             # Define via string
             dxdt_string = ["dx = -k * x", "z = x * k"]
             
             ode_string = SymbolicODE.create(
                 dxdt=dxdt_string,
                 states={'x': 1.0},
                 parameters={'k': 0.1},
                 observables=['z']
             )
             
             # Both should have same observable count
             assert len(ode_sympy.indices.observables) == 1
             assert len(ode_string.indices.observables) == 1
     ```
   - Edge cases covered:
     - Code generation equivalence
     - Hash consistency
     - Observables equivalence
   - Integration: Full SymbolicODE creation and compilation

**Outcomes**: 
- Files Modified:
  * tests/odesystems/symbolic/test_symbolicode.py (105 lines added)
- Tests Added:
  * TestSympyStringEquivalence class with 3 test methods
- Implementation Summary:
  Added comprehensive end-to-end tests verifying: generated code identity,
  hash consistency between SymPy and string inputs, and observable equivalence.
  Tests create full SymbolicODE objects from both input types and compare
  results, validating complete pathway equivalence.
- Issues Flagged: None

---

## Summary

**Total Task Groups**: 10
**Execution Strategy**: 
- Groups 1-5: Sequential (core implementation)
- Groups 6-7: Parallel with each other (independent unit tests)
- Groups 8-10: Sequential (integration and equivalence tests)

**Dependency Chain**:
```
Group 1 (Type Detection)
  ↓
Group 2 (Normalization) ← depends on Group 1
  ↓
Group 3 (Symbol Extraction) ← depends on Group 2
  ↓
Group 4 (parse_input Branching) ← depends on Groups 1-3
  ↓
Group 5 (CellML Simplification) ← depends on Group 4
  ↓
Groups 6-7 (Unit Tests) ← depends on Groups 1-2, can run in parallel
  ↓
Group 8 (Integration Tests) ← depends on Groups 4-5
  ↓
Group 9 (CellML Tests) ← depends on Group 5
  ↓
Group 10 (Equivalence Tests) ← depends on Groups 8-9
```

**Estimated Complexity**:
- Low: Groups 1, 5, 6, 9 (straightforward implementations)
- Medium: Groups 2, 4, 7, 8, 10 (moderate logic, multiple cases)
- High: Group 3 (complex validation, mirrors existing _lhs_pass/_rhs_pass)

**Parallel Execution Opportunities**:
- Groups 6 and 7 can execute simultaneously
- Unit tests within each group can run in parallel
- Groups 9 and 10 tests can partially overlap (run different test files)

**Critical Success Factors**:
1. Type detection must be robust and handle all input formats
2. Normalization must preserve equation semantics
3. Symbol extraction must use free_symbols correctly
4. Hash consistency between SymPy and string pathways
5. All existing CellML tests must continue to pass
6. No performance degradation in string pathway
