# Implementation Task List
# Feature: Codegen Hash Unification
# Plan Reference: .github/active_plans/codegen_hash_unification/agent_plan.md

## Task Group 1: Canonical Hash Function Implementation
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/odesystems/symbolic/sym_utils.py (lines 1-60, 130-210)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 297-422) - ParsedEquations class

**Input Validation Required**:
- equations: Check if `ParsedEquations` instance (has `.ordered` attribute) or `Iterable[Tuple[Symbol, Expr]]`
- constants: Validate is `None` or `Dict[str, float]`
- Each equation tuple: Verify first element is `sp.Symbol`, second is `sp.Expr`

**Tasks**:
1. **Replace hash_system_definition() with canonical implementation**
   - File: src/cubie/odesystems/symbolic/sym_utils.py
   - Action: Modify
   - Details:
     ```python
     def hash_system_definition(
         equations: Union["ParsedEquations", Iterable[Tuple[sp.Symbol, sp.Expr]]],
         constants: Optional[Dict[str, float]] = None,
     ) -> str:
         """Generate deterministic hash for symbolic ODE definitions.
         
         Produces identical hashes for identical equation sets regardless
         of input order by sorting equations alphabetically by LHS symbol
         name before building the hash string.
         
         Parameters
         ----------
         equations
             Parsed equations object or iterable of (symbol, expression)
             tuples representing the system.
         constants
             Optional mapping of constant names to values.
         
         Returns
         -------
         str
             Deterministic hash string reflecting equations and constants.
         
         Notes
         -----
         Sorting by LHS symbol name ensures order-independence so that
         cache hits occur for identical systems regardless of input
         pathway (string vs SymPy).
         """
         # Extract equations from ParsedEquations if needed
         if hasattr(equations, 'ordered'):
             eq_list = list(equations.ordered)
         else:
             eq_list = list(equations)
         
         # Sort equations alphabetically by LHS symbol name
         sorted_eqs = sorted(eq_list, key=lambda eq: str(eq[0]))
         
         # Build canonical equation string
         eq_strings = [f"{str(lhs)}={str(rhs)}" for lhs, rhs in sorted_eqs]
         dxdt_str = "|".join(eq_strings)
         
         # Normalize by removing whitespace
         normalized_dxdt = "".join(dxdt_str.split())
         
         # Process constants (sorted by key for determinism)
         constants_str = ""
         if constants is not None:
             sorted_constants = sorted(constants.items(), key=lambda x: x[0])
             constants_str = "|".join(f"{k}:{v}" for k, v in sorted_constants)
         
         # Combine and hash
         combined = f"dxdt:{normalized_dxdt}|constants:{constants_str}"
         return str(hash(combined))
     ```
   - Edge cases:
     - Empty equations: Return hash of just constants portion
     - None/empty constants: Constants portion is empty string
     - ParsedEquations with `.ordered` attribute vs raw iterable
   - Integration: Function called from parser.py and symbolicODE.py

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_sym_utils.py
- Test function: test_hash_order_independence
- Description: Verify hash is identical when equations are provided in different orders
- Test function: test_hash_parsed_equations_input
- Description: Verify hash accepts ParsedEquations objects correctly
- Test function: test_hash_constant_sorting
- Description: Verify constants are sorted alphabetically before hashing
- Test function: test_hash_empty_equations
- Description: Verify hash handles empty equation list
- Test function: test_hash_none_constants
- Description: Verify hash handles None constants

**Tests to Run**:
- tests/odesystems/symbolic/test_sym_utils.py::TestHashSystemDefinition::test_hash_order_independence
- tests/odesystems/symbolic/test_sym_utils.py::TestHashSystemDefinition::test_hash_parsed_equations_input
- tests/odesystems/symbolic/test_sym_utils.py::TestHashSystemDefinition::test_hash_constant_sorting
- tests/odesystems/symbolic/test_sym_utils.py::TestHashSystemDefinition::test_hash_empty_equations
- tests/odesystems/symbolic/test_sym_utils.py::TestHashSystemDefinition::test_hash_none_constants

**Outcomes**:
- Files Modified: 
  * src/cubie/odesystems/symbolic/sym_utils.py (57 lines replaced with canonical implementation)
  * tests/odesystems/symbolic/test_sym_utils.py (86 lines added)
- Functions/Methods Added/Modified:
  * hash_system_definition() in sym_utils.py - rewritten to accept ParsedEquations or Iterable[Tuple[Symbol, Expr]], sorts equations alphabetically by LHS symbol name, sorts constants alphabetically by key
- Implementation Summary:
  Replaced the complex, order-dependent hash_system_definition() with a canonical implementation that produces deterministic hashes regardless of equation input order. The function now:
  1. Accepts ParsedEquations objects (detected via .ordered attribute) or iterable of (symbol, expression) tuples
  2. Sorts equations alphabetically by LHS symbol name before building hash string
  3. Sorts constants alphabetically by key before including in hash
  4. Uses consistent format: "dxdt:{sorted_equations}|constants:{sorted_constants}"
- Issues Flagged: None

---

## Task Group 2: Update parse_input() Hash Computation Location
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1-30, 1251-1490)
- File: src/cubie/odesystems/symbolic/sym_utils.py (lines 130-210) - updated hash function

**Input Validation Required**:
- None (input validation handled by existing parse_input code)

**Tasks**:
1. **Move hash computation to after ParsedEquations creation in parse_input()**
   - File: src/cubie/odesystems/symbolic/parsing/parser.py
   - Action: Modify
   - Details:
     Current code has hash computed early in both branches:
     - Line 1378 (string path): `fn_hash = hash_system_definition(dxdt, constants)`
     - Line 1420 (sympy path): `fn_hash = hash_system_definition(substituted_eqs, constants)`
     
     Remove both early hash calls and add single hash after ParsedEquations creation:
     
     **Step 1**: Remove line 1378 (string path hash)
     ```python
     # DELETE: fn_hash = hash_system_definition(dxdt, constants)
     ```
     
     **Step 2**: Remove line 1420 (sympy path hash)
     ```python
     # DELETE: fn_hash = hash_system_definition(substituted_eqs, constants)
     ```
     
     **Step 3**: Add hash computation after line 1486 (after ParsedEquations creation):
     ```python
     parsed_equations = ParsedEquations.from_equations(equation_map, index_map)
     
     # Compute hash from canonical ParsedEquations form
     fn_hash = hash_system_definition(
         parsed_equations, index_map.constants.default_values
     )
     ```
   - Edge cases: 
     - Ensure both string and sympy paths flow to same hash computation
     - Verify constants come from index_map.constants.default_values
   - Integration: Single hash point for all input pathways

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_parser.py
- Test function: test_hash_consistency_string_vs_sympy_order
- Description: Verify string input ["dx=-k*x", "dy=k*x"] and SymPy input [Eq(dy, k*x), Eq(dx, -k*x)] produce identical hashes
- Test function: test_hash_computed_after_parsing
- Description: Verify hash correctly reflects parsed structure, not raw input

**Tests to Run**:
- tests/odesystems/symbolic/test_parser.py::TestHashConsistency::test_hash_consistency_string_vs_sympy_order
- tests/odesystems/symbolic/test_parser.py::TestHashConsistency::test_hash_computed_after_parsing
- tests/odesystems/symbolic/test_parser.py::TestIntegrationWithFixtures::test_with_simple_system_defaults

**Outcomes**: 
- Files Modified: 
  * src/cubie/odesystems/symbolic/parsing/parser.py (6 lines removed, 5 lines added)
  * tests/odesystems/symbolic/test_parser.py (69 lines added)
- Functions/Methods Added/Modified:
  * parse_input() in parser.py - hash computation moved from early in both string and sympy branches to after ParsedEquations creation
- Implementation Summary:
  Removed early hash_system_definition() calls from both string path (line ~1378) and sympy path (line ~1420). Added single hash computation after ParsedEquations.from_equations() call at line 1481. The hash now uses parsed_equations and index_map.constants.default_values, ensuring consistent hashing regardless of input pathway or equation order.
- Tests Created:
  * TestHashConsistency class with 2 test methods
  * test_hash_consistency_string_vs_sympy_order - verifies string and sympy inputs with different order produce same hash
  * test_hash_computed_after_parsing - verifies equation order does not affect hash
- Issues Flagged: None

---

## Task Group 3: Update SymbolicODE Hash Handling
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/odesystems/symbolic/symbolicODE.py (lines 1-50, 165-210, 360-400)
- File: src/cubie/odesystems/symbolic/sym_utils.py (lines 130-210) - updated hash function
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 297-350) - ParsedEquations class

**Input Validation Required**:
- None (existing validation sufficient)

**Tasks**:
1. **Update SymbolicODE.__init__() fallback hash computation**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Modify
   - Details:
     Current code (lines 180-184):
     ```python
     if fn_hash is None:
         dxdt_str = [f"{lhs}={str(rhs)}" for lhs, rhs
                     in equations]
         constants = all_indexed_bases.constants.default_values
         fn_hash = hash_system_definition(dxdt_str, constants)
     ```
     
     Replace with simplified version that passes equations directly:
     ```python
     if fn_hash is None:
         constants = all_indexed_bases.constants.default_values
         fn_hash = hash_system_definition(equations, constants)
     ```
   - Edge cases: 
     - equations parameter is ParsedEquations object
     - fn_hash already provided (skip computation)
   - Integration: Works with updated hash_system_definition signature

2. **Verify SymbolicODE.build() hash recomputation works correctly**
   - File: src/cubie/odesystems/symbolic/symbolicODE.py
   - Action: Verify (no change needed if hash function properly handles ParsedEquations)
   - Details:
     Lines 371-376 already pass `self.equations` (ParsedEquations) to hash function:
     ```python
     new_hash = hash_system_definition(
         self.equations, self.indices.constants.default_values
     )
     ```
     This should work correctly with updated hash_system_definition.
   - Edge cases: None
   - Integration: Existing code should work with new function signature

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_symbolicode.py
- Test function: test_symbolic_ode_hash_determinism
- Description: Verify creating identical SymbolicODE systems produces identical fn_hash
- Test function: test_symbolic_ode_hash_fallback
- Description: Verify __init__ correctly computes hash when fn_hash=None

**Tests to Run**:
- tests/odesystems/symbolic/test_symbolicode.py::TestSymbolicODEHash::test_symbolic_ode_hash_determinism
- tests/odesystems/symbolic/test_symbolicode.py::TestSymbolicODEHash::test_symbolic_ode_hash_fallback

**Outcomes**:
- Files Modified: 
  * src/cubie/odesystems/symbolic/symbolicODE.py (3 lines changed - removed string conversion in hash fallback)
  * tests/odesystems/symbolic/test_symbolicode.py (57 lines added)
- Functions/Methods Added/Modified:
  * SymbolicODE.__init__() in symbolicODE.py - simplified hash fallback to pass ParsedEquations directly
- Implementation Summary:
  Simplified the fallback hash computation in SymbolicODE.__init__() to pass the equations (ParsedEquations object) directly to hash_system_definition() instead of first converting to strings. The build() method already passes self.equations correctly and required no changes. Added TestSymbolicODEHash test class with two test methods to verify hash determinism and fallback computation.
- Issues Flagged: None

---

## Task Group 4: Integration Tests for Cache Consistency
**Status**: [x]
**Dependencies**: Task Groups 1, 2, 3

**Required Context**:
- File: src/cubie/odesystems/symbolic/symbolicODE.py (entire file)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1251-1490)
- File: tests/odesystems/symbolic/test_parser.py (lines 1-100, 750-850)
- File: tests/odesystems/symbolic/test_symbolicode.py (entire file)
- File: tests/odesystems/symbolic/conftest.py (entire file)

**Input Validation Required**:
- None (tests only)

**Tasks**:
1. **Create integration test for cache hit across input pathways**
   - File: tests/odesystems/symbolic/test_parser.py
   - Action: Create new test class
   - Details:
     ```python
     class TestHashConsistency:
         """Test hash consistency across input pathways."""
         
         def test_string_vs_sympy_same_hash(self):
             """Verify string and SymPy inputs produce same hash."""
             # String input
             string_result = parse_input(
                 dxdt=["dx = -k*x", "dy = k*x"],
                 states=['x', 'y'],
                 parameters=['k'],
                 constants={},
                 observables=[],
                 drivers=[],
             )
             
             # SymPy input with REVERSED order
             x, y, k = sp.symbols('x y k', real=True)
             dx, dy = sp.symbols('dx dy', real=True)
             sympy_result = parse_input(
                 dxdt=[sp.Eq(dy, k*x), sp.Eq(dx, -k*x)],
                 states=['x', 'y'],
                 parameters=['k'],
                 constants={},
                 observables=[],
                 drivers=[],
             )
             
             # Hash should be identical
             assert string_result[4] == sympy_result[4]
         
         def test_equation_order_independence(self):
             """Verify equation order does not affect hash."""
             # Order A
             result_a = parse_input(
                 dxdt=["dx = -x", "dy = x"],
                 states=['x', 'y'],
                 parameters=[],
                 constants={},
                 observables=[],
                 drivers=[],
             )
             
             # Order B (reversed)
             result_b = parse_input(
                 dxdt=["dy = x", "dx = -x"],
                 states=['x', 'y'],
                 parameters=[],
                 constants={},
                 observables=[],
                 drivers=[],
             )
             
             assert result_a[4] == result_b[4]
     ```
   - Edge cases: Different orderings, different input types
   - Integration: End-to-end test of the fix

2. **Update existing hash tests if needed**
   - File: tests/odesystems/symbolic/test_parser.py
   - Action: Modify
   - Details:
     Review existing tests in TestIntegrationWithFixtures class (lines 759-833)
     that assert `fn_hash1 == fn_hash2`. These should continue to pass.
     If any tests assert specific hash values (unlikely), update them.
   - Edge cases: None
   - Integration: Ensure no regression

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_parser.py
- Test function: test_string_vs_sympy_same_hash
- Description: Verify string input and SymPy input with reversed order produce identical hash
- Test function: test_equation_order_independence
- Description: Verify equation input order does not affect hash

**Tests to Run**:
- tests/odesystems/symbolic/test_parser.py::TestHashConsistency::test_hash_consistency_string_vs_sympy_order
- tests/odesystems/symbolic/test_parser.py::TestHashConsistency::test_hash_computed_after_parsing
- tests/odesystems/symbolic/test_parser.py::TestIntegrationWithFixtures::test_with_simple_system_defaults
- tests/odesystems/symbolic/test_symbolicode.py::TestSympyStringEquivalence::test_hash_consistency
- tests/odesystems/symbolic/test_symbolicode.py::TestSymbolicODEHash::test_symbolic_ode_hash_determinism
- tests/odesystems/symbolic/test_symbolicode.py::TestSymbolicODEHash::test_symbolic_ode_hash_fallback

**Outcomes**:
- Files Modified: 
  * No new files created - tests already exist from Task Group 2
- Functions/Methods Added/Modified:
  * None - tests were already created in Task Group 2
- Implementation Summary:
  Verified that the required integration tests already exist in test_parser.py:
  1. `TestHashConsistency.test_hash_consistency_string_vs_sympy_order` (lines 1173-1205) - Tests string vs SymPy inputs with reversed order produce identical hash
  2. `TestHashConsistency.test_hash_computed_after_parsing` (lines 1207-1236) - Tests equation order independence within same input type
  
  Also verified existing tests in `TestIntegrationWithFixtures.test_with_simple_system_defaults` (line 799) still assert `fn_hash1 == fn_hash2` for equivalent string and list inputs.
  
  Additional hash tests exist in:
  - `TestSympyStringEquivalence.test_hash_consistency` in test_symbolicode.py (lines 183-211)
  - `TestSymbolicODEHash` class in test_symbolicode.py (lines 246-301)
- Issues Flagged: None

---

## Summary

### Dependency Chain
```
Task Group 1 (hash function) 
    ↓
Task Group 2 (parser.py) ──┐
    ↓                      │
Task Group 3 (symbolicODE) ├── Both depend on Group 1
                           │
                           ↓
               Task Group 4 (integration tests)
```

### Files Modified
| File | Task Group | Change Type |
|------|------------|-------------|
| src/cubie/odesystems/symbolic/sym_utils.py | 1 | Modify hash_system_definition() |
| src/cubie/odesystems/symbolic/parsing/parser.py | 2 | Move hash computation location |
| src/cubie/odesystems/symbolic/symbolicODE.py | 3 | Simplify __init__ hash fallback |
| tests/odesystems/symbolic/test_sym_utils.py | 1 | Add hash function tests |
| tests/odesystems/symbolic/test_parser.py | 4 | Add integration tests |
| tests/odesystems/symbolic/test_symbolicode.py | 3 | Add SymbolicODE hash tests |

### Tests Created (Summary)
1. **test_sym_utils.py**: 5 new tests for hash function
2. **test_parser.py**: 4 new tests for hash consistency
3. **test_symbolicode.py**: 2 new tests for SymbolicODE hash handling

### Estimated Complexity
- Task Group 1: Medium - Core function rewrite with sorting logic
- Task Group 2: Low - Moving existing code, removing redundant calls
- Task Group 3: Low - Minor simplification
- Task Group 4: Low - Test creation only
