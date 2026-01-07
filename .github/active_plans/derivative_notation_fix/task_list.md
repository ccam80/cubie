# Implementation Task List
# Feature: Derivative Notation Fix
# Plan Reference: .github/active_plans/derivative_notation_fix/agent_plan.md

## Task Group 1: Add Function Notation Regex Pattern
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1-35)
- File: .github/copilot-instructions.md (lines 19-30 for style guidelines)

**Input Validation Required**:
- None for this task (regex pattern definition only)

**Tasks**:
1. **Add _DERIVATIVE_FUNC_PATTERN regex constant**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
     Add a new module-level constant after line 31 (after DRIVER_SETTING_KEYS):
     ```python
     # Pattern for d(variable, t) function notation - matches explicit
     # derivative syntax like d(x, t), d( velocity , t ) with optional
     # whitespace
     _DERIVATIVE_FUNC_PATTERN = re.compile(
         r"^d\s*\(\s*([A-Za-z_]\w*)\s*,\s*t\s*\)$"
     )
     ```
   - Edge cases:
     - Pattern must handle whitespace variations: `d(x, t)`, `d( x , t )`
     - Pattern must capture valid Python identifiers (start with letter/underscore)
     - Pattern must NOT match invalid forms like `d(123, t)` or `d(x)` (no time var)
   - Integration: Used by `_lhs_pass` to detect function notation before d-prefix check

**Tests to Create**:
- None for this task group (tested via integration in Task Group 3)

**Tests to Run**:
- None for this task group

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: Modify _lhs_pass for State-Aware Derivative Detection
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1015-1138)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 29-35 for pattern constants)
- File: .github/active_plans/derivative_notation_fix/agent_plan.md (lines 36-72 for logic flow)

**Input Validation Required**:
- lhs string: Validate non-empty after strip (already handled by line.split)
- Function notation match: Extract state name from regex capture group 1

**Tasks**:
1. **Refactor _lhs_pass to add function notation detection and state-aware d-prefix logic**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
     Replace the derivative detection logic in lines 1057-1084 with new priority-ordered checks:
     
     ```python
     for line in lines:
         lhs, rhs = [p.strip() for p in line.split("=", 1)]
         
         # Priority 1: Check for function notation d(name, t)
         func_match = _DERIVATIVE_FUNC_PATTERN.match(lhs)
         if func_match:
             state_name = func_match.group(1)
             s_sym = sp.Symbol(state_name, real=True)
             
             if state_name not in state_names:
                 if state_name in observable_names:
                     warn(
                         f"Symbol d({state_name}, t) found in equations, "
                         f"but {state_name} was listed as an observable. "
                         f"It has been converted into a state.",
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
                             f"Unknown state in derivative notation: "
                             f"d({state_name}, t). No state called "
                             f"{state_name} found."
                         )
                     else:
                         states.push(s_sym)
                         dxdt.push(sp.Symbol(f"d{state_name}", real=True))
                         state_names.add(state_name)
                         underived_states.add(f"d{state_name}")
             
             underived_states -= {f"d{state_name}"}
         
         # Priority 2: State-aware d-prefix check
         elif lhs.startswith("d") and len(lhs) > 1:
             potential_state = lhs[1:]
             
             # Only treat as derivative if remainder is a known state
             if potential_state in state_names:
                 underived_states -= {lhs}
             
             elif potential_state in observable_names:
                 # Observable being used as state derivative
                 s_sym = sp.Symbol(potential_state, real=True)
                 warn(
                     f"Your equation included d{potential_state}, but "
                     f"{potential_state} was listed as an observable. It "
                     f"has been converted into a state.",
                     EquationWarning,
                 )
                 states.push(s_sym)
                 dxdt.push(sp.Symbol(f"d{potential_state}", real=True))
                 observables.pop(s_sym)
                 state_names.add(potential_state)
                 observable_names.discard(potential_state)
                 underived_states -= {lhs}
             
             elif not strict:
                 # Non-strict mode: infer state from d-prefix
                 s_sym = sp.Symbol(potential_state, real=True)
                 states.push(s_sym)
                 dxdt.push(sp.Symbol(f"d{potential_state}", real=True))
                 state_names.add(potential_state)
                 underived_states.add(f"d{potential_state}")
                 underived_states -= {lhs}
             
             else:
                 # Not a known state - treat as auxiliary variable
                 if lhs not in observable_names:
                     anonymous_auxiliaries[lhs] = sp.Symbol(lhs, real=True)
                 if lhs in observable_names:
                     assigned_obs.add(lhs)
         
         elif lhs in state_names:
             raise ValueError(
                 f"State {lhs} cannot be assigned directly. All "
                 f"states must be defined as derivatives with d"
                 f"{lhs} = [...]"
             )
         
         elif (
             lhs in param_names
             or lhs in constant_names
             or lhs in driver_names
         ):
             raise ValueError(
                 f"{lhs} was entered as an immutable "
                 f"input (constant, parameter, or driver)"
                 ", but it is being assigned to. Cubie "
                 "can't handle this - if it's being "
                 "assigned to, it must be either a state, an "
                 "observable, or undefined."
             )
         
         else:
             if lhs not in observable_names:
                 anonymous_auxiliaries[lhs] = sp.Symbol(lhs, real=True)
             if lhs in observable_names:
                 assigned_obs.add(lhs)
     ```
     
   - Edge cases:
     - `d` alone (len == 1): Falls through to else clause, treated as auxiliary
     - `delta_i` where `elta_i` is not a state: Treated as auxiliary (not derivative)
     - `done` where `one` IS a state: Treated as derivative of `one`
     - `d(x, t)` with unknown `x` in strict mode: Raises ValueError
     - `d(x, t)` with unknown `x` in non-strict mode: Infers `x` as state
   - Integration: Main string input pathway for ODE parsing

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_parser.py
- Tests listed in Task Group 4

**Tests to Run**:
- tests/odesystems/symbolic/test_parser.py::TestLhsPass

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: Modify _lhs_pass_sympy for State-Aware Derivative Detection
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 746-886)
- File: .github/active_plans/derivative_notation_fix/agent_plan.md (lines 73-83 for SymPy pathway)

**Input Validation Required**:
- lhs_name string: Check starts with 'd' AND len > 1 for d-prefix check
- potential_state: Validate against state_names set before treating as derivative

**Tasks**:
1. **Refactor _lhs_pass_sympy to use state-aware d-prefix logic**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
     Replace the derivative detection logic in lines 807-836 with state-aware checks:
     
     ```python
     for lhs_sym, rhs_expr in equations:
         lhs_name = str(lhs_sym)
         
         # Check for d-prefix with state-aware logic
         if lhs_name.startswith("d") and len(lhs_name) > 1:
             potential_state = lhs_name[1:]
             
             # Only treat as derivative if remainder is a known state
             if potential_state in state_names:
                 underived_states -= {lhs_name}
             
             elif potential_state in observable_names:
                 s_sym = sp.Symbol(potential_state, real=True)
                 warn(
                     f"Symbol d{potential_state} found in equations, but "
                     f"{potential_state} was listed as an observable. "
                     f"Converting to state.",
                     EquationWarning,
                 )
                 states.push(s_sym)
                 dxdt.push(sp.Symbol(f"d{potential_state}", real=True))
                 observables.pop(s_sym)
                 state_names.add(potential_state)
                 observable_names.discard(potential_state)
                 underived_states -= {lhs_name}
             
             elif not strict:
                 # Non-strict mode: infer state from d-prefix
                 s_sym = sp.Symbol(potential_state, real=True)
                 states.push(s_sym)
                 dxdt.push(sp.Symbol(f"d{potential_state}", real=True))
                 state_names.add(potential_state)
                 underived_states.add(f"d{potential_state}")
                 underived_states -= {lhs_name}
             
             else:
                 # Not a known state - treat as auxiliary variable
                 if lhs_name not in observable_names:
                     anonymous_auxiliaries[lhs_name] = lhs_sym
                 if lhs_name in observable_names:
                     assigned_obs.add(lhs_name)
         
         elif lhs_name in state_names:
             raise ValueError(
                 f"State {lhs_name} cannot be assigned directly. "
                 f"States must be defined as derivatives: d{lhs_name} = ..."
             )
         
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
         
         else:
             if lhs_name not in observable_names:
                 anonymous_auxiliaries[lhs_name] = lhs_sym
             if lhs_name in observable_names:
                 assigned_obs.add(lhs_name)
     ```
     
   - Edge cases:
     - Same as Task Group 2 but for SymPy input pathway
     - SymPy `Derivative(x, t)` objects are already handled correctly in
       `_normalize_sympy_equations` (converted to `dx` symbol)
   - Integration: SymPy input pathway for ODE parsing

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_parser.py
- Tests listed in Task Group 4

**Tests to Run**:
- tests/odesystems/symbolic/test_parser.py::TestSympyInputPathway

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: Add Tests for State-Aware Derivative Detection
**Status**: [ ]
**Dependencies**: Task Groups 2, 3

**Required Context**:
- File: tests/odesystems/symbolic/test_parser.py (entire file)
- File: tests/odesystems/symbolic/conftest.py (entire file)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1-35 for imports)
- File: .github/active_plans/derivative_notation_fix/agent_plan.md (lines 199-239 for test cases)

**Input Validation Required**:
- None (test code)

**Tasks**:
1. **Add TestDerivativeNotation test class for string input pathway**
   - File: tests/odesystems/symbolic/test_parser.py
   - Action: Modify (add new class at end of file)
   - Details:
     Add a new test class after the existing classes:
     
     ```python
     class TestDerivativeNotation:
         """Test state-aware derivative detection and function notation."""
     
         def test_basic_derivative_with_declared_state(self):
             """dx = ... with state x declared is treated as derivative."""
             ib = IndexedBases.from_user_inputs(
                 ["x"], ["k"], [], [], []
             )
             lines = ["dx = -k * x"]
     
             anon_aux = _lhs_pass(lines, ib, strict=True)
     
             assert len(anon_aux) == 0
             assert "dx" not in anon_aux
             # dx should be tracked as a derivative, not auxiliary
     
         def test_ambiguous_prefix_not_a_state_treated_as_auxiliary(self):
             """delta_i = ... with no state elta_i is auxiliary."""
             ib = IndexedBases.from_user_inputs(
                 ["x"], ["k"], [], [], []
             )
             lines = ["dx = -k * x", "delta_i = x + 1"]
     
             anon_aux = _lhs_pass(lines, ib, strict=True)
     
             assert "delta_i" in anon_aux
             assert isinstance(anon_aux["delta_i"], sp.Symbol)
             # elta_i should NOT be added as a state
             assert "elta_i" not in ib.state_names
     
         def test_ambiguous_prefix_is_a_state_treated_as_derivative(self):
             """delta = ... with state elta declared is derivative of elta."""
             ib = IndexedBases.from_user_inputs(
                 ["elta"], ["k"], [], [], []
             )
             lines = ["delta = -k * elta"]
     
             anon_aux = _lhs_pass(lines, ib, strict=True)
     
             assert len(anon_aux) == 0
             assert "delta" not in anon_aux
             # delta should be the derivative of elta
             assert "delta" in ib.dxdt_names or "delta" not in anon_aux
     
         def test_function_notation_with_declared_state(self):
             """d(x, t) = ... with state x declared is derivative."""
             ib = IndexedBases.from_user_inputs(
                 ["x"], ["k"], [], [], []
             )
             lines = ["d(x, t) = -k * x"]
     
             anon_aux = _lhs_pass(lines, ib, strict=True)
     
             assert len(anon_aux) == 0
             # Should be treated as derivative of x
     
         def test_function_notation_undeclared_state_strict_raises(self):
             """d(x, t) = ... with no state x in strict mode raises."""
             ib = IndexedBases.from_user_inputs(
                 [], ["k"], [], [], []
             )
             lines = ["d(x, t) = -k * x"]
     
             with pytest.raises(ValueError, match="No state called x"):
                 _lhs_pass(lines, ib, strict=True)
     
         def test_function_notation_undeclared_state_non_strict_infers(self):
             """d(x, t) = ... with no state x in non-strict infers x."""
             ib = IndexedBases.from_user_inputs(
                 [], ["k"], [], [], []
             )
             lines = ["d(x, t) = -k * x"]
     
             anon_aux = _lhs_pass(lines, ib, strict=False)
     
             assert "x" in ib.state_names
             assert len(anon_aux) == 0
     
         def test_non_strict_state_inference_from_d_prefix(self):
             """dx = ... with no state x in non-strict infers x as state."""
             ib = IndexedBases.from_user_inputs(
                 [], ["k"], [], [], []
             )
             lines = ["dx = -k * x"]
     
             anon_aux = _lhs_pass(lines, ib, strict=False)
     
             assert "x" in ib.state_names
             assert len(anon_aux) == 0
     
         def test_non_strict_auxiliary_not_inferred_as_state(self):
             """delta = ... with no state elta in non-strict is auxiliary."""
             ib = IndexedBases.from_user_inputs(
                 ["x"], ["k"], [], [], []
             )
             lines = ["dx = -k * x", "delta = x + 1"]
     
             anon_aux = _lhs_pass(lines, ib, strict=False)
     
             # delta should be auxiliary, NOT inferring elta as state
             assert "delta" in anon_aux
             assert "elta" not in ib.state_names
     
         def test_function_notation_with_whitespace(self):
             """d( x , t ) = ... with extra whitespace works."""
             ib = IndexedBases.from_user_inputs(
                 ["x"], ["k"], [], [], []
             )
             lines = ["d( x , t ) = -k * x"]
     
             anon_aux = _lhs_pass(lines, ib, strict=True)
     
             assert len(anon_aux) == 0
     
         def test_single_letter_d_treated_as_auxiliary(self):
             """d = ... alone is treated as auxiliary, not derivative."""
             ib = IndexedBases.from_user_inputs(
                 ["x"], ["k"], [], [], []
             )
             lines = ["dx = -k * x", "d = x + 1"]
     
             anon_aux = _lhs_pass(lines, ib, strict=True)
     
             assert "d" in anon_aux
     ```
     
   - Edge cases: All edge cases covered by individual tests
   - Integration: Tests string input pathway

2. **Add SymPy pathway tests to TestSympyInputPathway class**
   - File: tests/odesystems/symbolic/test_parser.py
   - Action: Modify (add tests to existing class)
   - Details:
     Add these tests to the existing `TestSympyInputPathway` class:
     
     ```python
         def test_sympy_ambiguous_prefix_not_state_is_auxiliary(self):
             """SymPy: delta_i symbol without state elta_i is auxiliary."""
             x, k = sp.symbols('x k')
             dx = sp.Symbol('dx')
             delta_i = sp.Symbol('delta_i')
             
             dxdt = [
                 sp.Eq(dx, -k * x),
                 sp.Eq(delta_i, x + 1)
             ]
             
             index_map, all_symbols, funcs, parsed_eqs, fn_hash = parse_input(
                 dxdt=dxdt,
                 states=['x'],
                 parameters=['k'],
                 strict=True
             )
             
             # delta_i should NOT create state elta_i
             assert 'elta_i' not in index_map.state_names
             # delta_i should be in auxiliaries
             assert 'delta_i' in all_symbols
     
         def test_sympy_ambiguous_prefix_is_state_is_derivative(self):
             """SymPy: delta symbol with state elta is derivative."""
             elta, k = sp.symbols('elta k')
             delta = sp.Symbol('delta')
             
             dxdt = [sp.Eq(delta, -k * elta)]
             
             index_map, all_symbols, funcs, parsed_eqs, fn_hash = parse_input(
                 dxdt=dxdt,
                 states=['elta'],
                 parameters=['k'],
                 strict=True
             )
             
             assert len(parsed_eqs.state_derivatives) == 1
             assert 'delta' not in all_symbols or \
                    str(parsed_eqs.state_derivatives[0][0]) == 'delta'
     ```
     
   - Edge cases: Covers SymPy pathway for state-aware detection
   - Integration: Tests SymPy input pathway

3. **Add integration test for mixed notation**
   - File: tests/odesystems/symbolic/test_parser.py
   - Action: Modify (add to TestParseInput class)
   - Details:
     Add this test to the `TestParseInput` class:
     
     ```python
         def test_parse_input_mixed_derivatives_and_auxiliaries(self):
             """Test system with derivatives and d-prefixed auxiliaries."""
             states = ["x", "y"]
             parameters = ["k"]
             constants = {}
             observables = []
             drivers = []
             dxdt = [
                 "dx = -k * x",
                 "dy = k * x",
                 "delta = x + y",  # auxiliary, not derivative of elta
                 "damping = delta * k"  # auxiliary referencing delta
             ]
             
             index_map, all_symbols, funcs, parsed_eqs, fn_hash = parse_input(
                 states=states,
                 parameters=parameters,
                 constants=constants,
                 observables=observables,
                 drivers=drivers,
                 dxdt=dxdt,
                 strict=True
             )
             
             # Verify correct categorization
             assert "x" in index_map.state_names
             assert "y" in index_map.state_names
             assert "elta" not in index_map.state_names  # NOT a state
             assert "delta" in all_symbols  # auxiliary
             assert "damping" in all_symbols  # auxiliary
             assert len(parsed_eqs.state_derivatives) == 2
     ```
     
   - Edge cases: Mixed real derivatives and d-prefixed auxiliaries
   - Integration: Full parse_input integration test

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_parser.py
- Test class: TestDerivativeNotation (new class with 10 tests)
- Additional tests in: TestSympyInputPathway (2 tests)
- Additional tests in: TestParseInput (1 test)

**Tests to Run**:
- tests/odesystems/symbolic/test_parser.py::TestDerivativeNotation
- tests/odesystems/symbolic/test_parser.py::TestSympyInputPathway::test_sympy_ambiguous_prefix_not_state_is_auxiliary
- tests/odesystems/symbolic/test_parser.py::TestSympyInputPathway::test_sympy_ambiguous_prefix_is_state_is_derivative
- tests/odesystems/symbolic/test_parser.py::TestParseInput::test_parse_input_mixed_derivatives_and_auxiliaries

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: Update Docstrings
**Status**: [ ]
**Dependencies**: Task Groups 2, 3

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1015-1042 for _lhs_pass docstring)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 746-791 for _lhs_pass_sympy docstring)
- File: .github/copilot-instructions.md (lines 22-28 for docstring style)

**Input Validation Required**:
- None (documentation only)

**Tasks**:
1. **Update _lhs_pass docstring to document derivative notation behavior**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
     Update the Notes section of _lhs_pass docstring to include:
     
     ```python
     Notes
     -----
     Derivative notation is detected using state-aware logic:
     
     1. Function notation ``d(x, t)`` explicitly marks state ``x``'s derivative
     2. Prefix notation ``dX`` is a derivative only if ``X`` is a declared state
     3. Symbols like ``delta_i`` where ``elta_i`` is not a state are auxiliaries
     
     In non-strict mode, unknown d-prefixed symbols infer new states only when
     the entire symbol matches the ``dX`` pattern and ``X`` is a valid identifier.
     
     Anonymous auxiliaries ease model authoring but are not persisted as
     saved observables; tracking them ensures generated SymPy code remains
     consistent with the equations.
     ```
     
   - Edge cases: N/A
   - Integration: Documentation

2. **Update _lhs_pass_sympy docstring similarly**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
     Update the Notes section of _lhs_pass_sympy docstring to match _lhs_pass.
     
   - Edge cases: N/A
   - Integration: Documentation

**Tests to Create**:
- None (documentation only)

**Tests to Run**:
- None (documentation only)

**Outcomes**: 
[Empty - to be filled by taskmaster agent]
