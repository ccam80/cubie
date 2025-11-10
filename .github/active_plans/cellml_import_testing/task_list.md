# Implementation Task List
# Feature: CellML Import Testing
# Plan Reference: .github/active_plans/cellml_import_testing/agent_plan.md

## Task Group 1: Setup Test Fixtures and Directory Structure - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: tests/odesystems/symbolic/test_cellml.py (lines 1-11)
- Reference: cellmlmanip repository beeler_reuter_model_1977.cellml file
- Reference: .github/active_plans/cellml_import_testing/agent_plan.md (Component Descriptions section)

**Input Validation Required**:
- None (fixture setup)

**Tasks**:
1. **Create CellML test fixtures directory**
   - Directory: tests/fixtures/cellml/
   - Action: Create
   - Details:
     - Create the directory if it doesn't exist
     - This will hold CellML test model files

2. **Download Beeler-Reuter CellML model file**
   - File: tests/fixtures/cellml/beeler_reuter_model_1977.cellml
   - Action: Create
   - Details:
     - Download from cellmlmanip repository
     - URL: https://raw.githubusercontent.com/ModellingWebLab/cellmlmanip/main/tests/cellml_files/beeler_reuter_model_1977.cellml
     - This is a cardiac action potential model with 8 state variables
     - Representative complexity for testing
   - Integration: Primary test model for complex CellML import testing

3. **Create simple CellML test model**
   - File: tests/fixtures/cellml/basic_ode.cellml
   - Action: Create
   - Details:
     ```xml
     <?xml version="1.0" encoding="utf-8"?>
     <model name="basic_ode" xmlns="http://www.cellml.org/cellml/1.0#">
       <component name="main">
         <variable name="time" units="dimensionless" public_interface="out"/>
         <variable name="x" units="dimensionless" initial_value="1.0"/>
         <variable name="a" units="dimensionless" initial_value="0.5"/>
         <math xmlns="http://www.w3.org/1998/Math/MathML">
           <apply>
             <eq/>
             <apply>
               <diff/>
               <bvar><ci>time</ci></bvar>
               <ci>x</ci>
             </apply>
             <apply>
               <times/>
               <apply><minus/><ci>a</ci></apply>
               <ci>x</ci>
             </apply>
           </apply>
         </math>
       </component>
     </model>
     ```
   - Purpose: Fast basic tests with minimal complexity (1 state variable)
   - Integration: Used for quick loading and structure tests

**Outcomes**:


---

## Task Group 2: Add Pytest Fixtures to test_cellml.py - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 1

**Required Context**:
- File: tests/odesystems/symbolic/test_cellml.py (entire file)
- File: tests/odesystems/symbolic/conftest.py (entire file - for fixture pattern reference)
- File: tests/conftest.py (lines 1-50 - for marker patterns)
- Directory: tests/fixtures/cellml/ (created in Group 1)

**Input Validation Required**:
- None (fixtures don't need validation)

**Tasks**:
1. **Add imports to test_cellml.py**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify
   - Details:
     ```python
     import pytest
     import sympy as sp
     import numpy as np
     from pathlib import Path
     
     from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model
     from cubie import create_ODE_system, solve_ivp
     ```
   - Replace existing imports (lines 1-3) with expanded imports
   - Integration: Adds necessary modules for comprehensive testing

2. **Add cellml_fixtures_dir fixture**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify (add after imports)
   - Details:
     ```python
     @pytest.fixture
     def cellml_fixtures_dir():
         """Return path to CellML test fixtures directory."""
         test_dir = Path(__file__).parent.parent.parent
         fixtures_dir = test_dir / "fixtures" / "cellml"
         return fixtures_dir
     ```
   - Purpose: Centralize fixture directory path
   - Integration: Used by other fixtures to locate CellML files

3. **Add simple_cellml_model fixture**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify (add after cellml_fixtures_dir)
   - Details:
     ```python
     @pytest.fixture
     def simple_cellml_model(cellml_fixtures_dir):
         """Return path to basic ODE CellML model."""
         model_path = cellml_fixtures_dir / "basic_ode.cellml"
         if not model_path.exists():
             pytest.skip(f"Test fixture not found: {model_path}")
         return str(model_path)
     ```
   - Purpose: Provide path to simple test model
   - Integration: Used by basic loading tests

4. **Add complex_cellml_model fixture**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify (add after simple_cellml_model)
   - Details:
     ```python
     @pytest.fixture
     def complex_cellml_model(cellml_fixtures_dir):
         """Return path to Beeler-Reuter cardiac model."""
         model_path = cellml_fixtures_dir / "beeler_reuter_model_1977.cellml"
         if not model_path.exists():
             pytest.skip(f"Test fixture not found: {model_path}")
         return str(model_path)
     ```
   - Purpose: Provide path to complex test model
   - Integration: Used by comprehensive tests with real physiological models

**Outcomes**:


---

## Task Group 3: Implement Basic Loading Tests - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 2

**Required Context**:
- File: tests/odesystems/symbolic/test_cellml.py (current state after Group 2)
- File: src/cubie/odesystems/symbolic/parsing/cellml.py (lines 19-38)
- Reference: .github/active_plans/cellml_import_testing/agent_plan.md (Expected Behavior section)

**Input Validation Required**:
- None (tests validate outputs, don't need input validation)

**Tasks**:
1. **Modify existing test_cellml_import_error test**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify
   - Details:
     ```python
     def test_cellml_import_error(monkeypatch):
         """Missing dependency raises ImportError."""
         # Temporarily set cellmlmanip to None to simulate missing import
         import cubie.odesystems.symbolic.parsing.cellml as cellml_module
         monkeypatch.setattr(cellml_module, 'cellmlmanip', None)
         
         with pytest.raises(ImportError, match="cellmlmanip is required"):
             load_cellml_model("dummy.cellml")
     ```
   - Keep existing test but improve with monkeypatch for reliability
   - Integration: Verifies graceful handling when cellmlmanip not installed

2. **Add test_load_simple_model**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify (add new test function)
   - Details:
     ```python
     def test_load_simple_model(simple_cellml_model):
         """Load basic CellML model and verify structure."""
         pytest.importorskip("cellmlmanip")
         
         states, equations = load_cellml_model(simple_cellml_model)
         
         # Verify return types
         assert isinstance(states, list)
         assert isinstance(equations, list)
         
         # Verify non-empty
         assert len(states) > 0
         assert len(equations) > 0
         
         # Verify basic structure
         assert len(states) == len(equations), "Should have one equation per state"
     ```
   - Purpose: Basic smoke test that model loading works
   - Edge cases: Empty model (skip if fixture missing)
   - Integration: Validates load_cellml_model returns correct types

3. **Add test_load_complex_model**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify (add new test function)
   - Details:
     ```python
     def test_load_complex_model(complex_cellml_model):
         """Load Beeler-Reuter model and verify extraction."""
         pytest.importorskip("cellmlmanip")
         
         states, equations = load_cellml_model(complex_cellml_model)
         
         # Verify return types
         assert isinstance(states, list)
         assert isinstance(equations, list)
         
         # Beeler-Reuter has 8 state variables
         assert len(states) == 8, f"Expected 8 states, got {len(states)}"
         assert len(equations) == 8, f"Expected 8 equations, got {len(equations)}"
         
         # Verify all entries exist
         assert all(s is not None for s in states)
         assert all(eq is not None for eq in equations)
     ```
   - Purpose: Verify correct extraction from real physiological model
   - Edge cases: Model parsing errors (let cellmlmanip error propagate)
   - Integration: Tests with realistic model complexity

**Outcomes**:


---

## Task Group 4: Implement Type Verification Tests - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 3

**Required Context**:
- File: tests/odesystems/symbolic/test_cellml.py (current state)
- File: src/cubie/odesystems/symbolic/parsing/cellml.py (lines 19-38)
- Reference: .github/active_plans/cellml_import_testing/agent_plan.md (Edge Cases section)

**Input Validation Required**:
- None (tests validate types)

**Tasks**:
1. **Add test_states_are_symbols**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify (add new test function)
   - Details:
     ```python
     def test_states_are_symbols(complex_cellml_model):
         """Verify state variables are sympy.Symbol instances."""
         pytest.importorskip("cellmlmanip")
         
         states, equations = load_cellml_model(complex_cellml_model)
         
         # Each state should be a Symbol (not Dummy, not other types)
         for i, state in enumerate(states):
             assert isinstance(state, sp.Symbol), (
                 f"State {i} ({state}) is {type(state).__name__}, "
                 f"expected Symbol"
             )
             # Verify it's not a Dummy symbol
             assert not isinstance(state, sp.Dummy), (
                 f"State {i} ({state}) is a Dummy, should be Symbol"
             )
     ```
   - Purpose: Verify cellmlmanip returns proper Symbol types for CuBIE
   - Edge cases: cellmlmanip may return Dummy symbols (need conversion)
   - Integration: Critical for SymbolicODE compatibility

2. **Add test_equations_are_sympy_eq**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify (add new test function)
   - Details:
     ```python
     def test_equations_are_sympy_eq(complex_cellml_model):
         """Verify equations are sympy.Eq instances."""
         pytest.importorskip("cellmlmanip")
         
         states, equations = load_cellml_model(complex_cellml_model)
         
         # Each equation should be a sympy.Eq
         for i, eq in enumerate(equations):
             assert isinstance(eq, sp.Eq), (
                 f"Equation {i} is {type(eq).__name__}, expected Eq"
             )
     ```
   - Purpose: Verify equation format matches CuBIE expectations
   - Integration: Required for SymbolicODE parsing

3. **Add test_derivatives_in_equations**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify (add new test function)
   - Details:
     ```python
     def test_derivatives_in_equations(complex_cellml_model):
         """Verify equations contain derivatives on left-hand side."""
         pytest.importorskip("cellmlmanip")
         
         states, equations = load_cellml_model(complex_cellml_model)
         
         # Each equation's LHS should be a Derivative
         for i, eq in enumerate(equations):
             assert isinstance(eq.lhs, sp.Derivative), (
                 f"Equation {i} LHS is {type(eq.lhs).__name__}, "
                 f"expected Derivative"
             )
             
             # Derivative should have a function and a variable
             assert len(eq.lhs.args) == 2, (
                 f"Equation {i} derivative has {len(eq.lhs.args)} args, "
                 f"expected 2 (function, variable)"
             )
     ```
   - Purpose: Verify ODE structure (derivatives on LHS)
   - Edge cases: Algebraic equations should be filtered out
   - Integration: CuBIE expects differential equations only

4. **Add test_all_states_have_derivatives**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify (add new test function)
   - Details:
     ```python
     def test_all_states_have_derivatives(complex_cellml_model):
         """Verify every state has a corresponding derivative equation."""
         pytest.importorskip("cellmlmanip")
         
         states, equations = load_cellml_model(complex_cellml_model)
         
         # Extract state symbols from derivative equations
         derived_states = set()
         for eq in equations:
             if isinstance(eq.lhs, sp.Derivative):
                 # Get the function being differentiated
                 derived_var = eq.lhs.args[0]
                 derived_states.add(derived_var)
         
         # Convert states list to set for comparison
         state_set = set(states)
         
         # Every state should have a derivative
         missing = state_set - derived_states
         assert not missing, (
             f"States {missing} have no corresponding derivative equation"
         )
         
         # No extra derivatives
         extra = derived_states - state_set
         assert not extra, (
             f"Derivatives {extra} have no corresponding state variable"
         )
     ```
   - Purpose: Verify complete ODE system (one equation per state)
   - Edge cases: Missing derivatives, extra equations
   - Integration: Ensures complete system definition

**Outcomes**:


---

## Task Group 5: Investigate and Fix load_cellml_model Implementation - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 4

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/cellml.py (entire file)
- Test results from Group 4 (will show if Dummy symbols need conversion)
- Reference: .github/active_plans/cellml_import_testing/agent_plan.md (Potential issues identified section)

**Input Validation Required**:
- path: Check type is str, file exists, has .cellml extension
- cellmlmanip: Verify not None before use (already implemented)

**Tasks**:
1. **Run Group 4 tests and analyze failures**
   - File: N/A (analysis task)
   - Action: Execute tests
   - Details:
     - Run pytest on test_states_are_symbols and related tests
     - Document which tests fail and why
     - Identify if cellmlmanip returns Dummy symbols
     - Identify if filtering logic is correct
     - Note any other structural issues
   - Integration: Informs necessary fixes to load_cellml_model

2. **Fix Symbol vs Dummy issue (if needed)**
   - File: src/cubie/odesystems/symbolic/parsing/cellml.py
   - Action: Modify (conditionally, based on test results)
   - Details:
     ```python
     def load_cellml_model(path: str) -> tuple[list[sp.Symbol], list[sp.Eq]]:
         """Load a CellML model and extract states and derivatives.
     
         Parameters
         ----------
         path
             Filesystem path to the CellML source file.
     
         Returns
         -------
         tuple[list[sympy.Symbol], list[sympy.Eq]]
             States and differential equations defined by the model.
         """
         if cellmlmanip is None:  # pragma: no cover
             raise ImportError("cellmlmanip is required for CellML parsing")
         
         model = cellmlmanip.load_model(path)
         raw_states = list(model.get_state_variables())
         derivatives = list(model.get_derivatives())
         equations = [eq for eq in model.equations if eq.lhs in derivatives]
         
         # Convert Dummy symbols to regular Symbols if needed
         # cellmlmanip may return Dummy symbols which use hash-based equality
         # CuBIE needs regular Symbols with name-based equality
         states = []
         for state in raw_states:
             if isinstance(state, sp.Dummy):
                 # Convert Dummy to Symbol preserving the name
                 states.append(sp.Symbol(state.name, real=True))
             else:
                 states.append(state)
         
         return states, equations
     ```
   - Only implement if tests show Dummy symbols being returned
   - Edge cases: State symbols with special characters in names
   - Integration: Ensures Symbol type compatibility with CuBIE

3. **Add input validation to load_cellml_model**
   - File: src/cubie/odesystems/symbolic/parsing/cellml.py
   - Action: Modify (add validation at function start)
   - Details:
     ```python
     def load_cellml_model(path: str) -> tuple[list[sp.Symbol], list[sp.Eq]]:
         """Load a CellML model and extract states and derivatives.
     
         Parameters
         ----------
         path
             Filesystem path to the CellML source file.
     
         Returns
         -------
         tuple[list[sympy.Symbol], list[sympy.Eq]]
             States and differential equations defined by the model.
         
         Raises
         ------
         ImportError
             If cellmlmanip is not installed.
         TypeError
             If path is not a string.
         FileNotFoundError
             If the specified file does not exist.
         ValueError
             If the file is not a .cellml file.
         """
         if cellmlmanip is None:  # pragma: no cover
             raise ImportError("cellmlmanip is required for CellML parsing")
         
         # Validate path type
         if not isinstance(path, str):
             raise TypeError(
                 f"path must be a string, got {type(path).__name__}"
             )
         
         # Validate file exists
         from pathlib import Path
         path_obj = Path(path)
         if not path_obj.exists():
             raise FileNotFoundError(
                 f"CellML file not found: {path}"
             )
         
         # Validate .cellml extension
         if not path.endswith('.cellml'):
             raise ValueError(
                 f"File must have .cellml extension, got: {path}"
             )
         
         # ... rest of existing implementation
     ```
   - Edge cases: Non-string paths, missing files, wrong extension
   - Integration: Provides clear error messages for user mistakes

**Outcomes**:


---

## Task Group 6: Implement Integration Tests with SymbolicODE - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 5

**Required Context**:
- File: tests/odesystems/symbolic/test_cellml.py (current state)
- File: src/cubie/odesystems/symbolic/symbolicODE.py (lines 49-111, 212-286)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 268-278)
- Reference: .github/active_plans/cellml_import_testing/agent_plan.md (Integration Points section)

**Input Validation Required**:
- None (tests use fixtures)

**Tasks**:
1. **Add test_integration_with_symbolic_ode**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify (add new test function)
   - Details:
     ```python
     @pytest.mark.slow
     def test_integration_with_symbolic_ode(complex_cellml_model):
         """Create SymbolicODE from loaded CellML model."""
         pytest.importorskip("cellmlmanip")
         
         states, equations = load_cellml_model(complex_cellml_model)
         
         # Extract state names for initial values
         # CellML states are Symbol objects with .name attribute
         state_dict = {state.name: 0.0 for state in states}
         
         # Convert equations to string format for create_ODE_system
         # create_ODE_system expects dxdt as string equations "lhs = rhs"
         equation_strings = []
         for eq in equations:
             # eq is sp.Eq with lhs as Derivative and rhs as expression
             # Get the variable name from the derivative
             if isinstance(eq.lhs, sp.Derivative):
                 var = eq.lhs.args[0]  # The function being differentiated
                 var_name = var.name if hasattr(var, 'name') else str(var)
                 # Format as "dvar = rhs_expression"
                 equation_strings.append(f"d{var_name} = {eq.rhs}")
         
         # This should not raise
         ode = create_ODE_system(
             dxdt=equation_strings,
             states=state_dict,
             precision=np.float64
         )
         
         assert ode is not None
         assert len(ode.states) == len(states)
     ```
   - Purpose: Verify CellML models can be used with CuBIE
   - Edge cases: Equation format conversion, state naming
   - Integration: End-to-end test from CellML to SymbolicODE
   - Note: This test may reveal need for better equation handling

2. **Add alternative test using equations directly (if supported)**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify (add new test function, conditional on parser support)
   - Details:
     ```python
     @pytest.mark.slow
     def test_create_ode_from_cellml_equations_direct(complex_cellml_model):
         """Test if create_ODE_system can accept equation tuples directly."""
         pytest.importorskip("cellmlmanip")
         
         states, equations = load_cellml_model(complex_cellml_model)
         
         # Try creating ODE with equations as (lhs, rhs) tuples
         # Note: This may not be supported by current parser
         # If not supported, this test should be skipped
         
         try:
             # Convert equations to tuples format
             eq_tuples = []
             state_dict = {}
             for eq in equations:
                 if isinstance(eq.lhs, sp.Derivative):
                     var = eq.lhs.args[0]
                     var_name = var.name if hasattr(var, 'name') else str(var)
                     state_dict[var_name] = 0.0
                     # Create derivative symbol
                     dvar = sp.Symbol(f"d{var_name}", real=True)
                     eq_tuples.append((dvar, eq.rhs))
             
             # Attempt to create ODE - may not be supported
             ode = create_ODE_system(
                 dxdt=eq_tuples,
                 states=state_dict,
                 precision=np.float64
             )
             
             assert ode is not None
             assert len(ode.states) == len(states)
             
         except (TypeError, ValueError) as e:
             # If direct equation passing not supported, skip
             pytest.skip(
                 f"Direct equation passing not supported: {e}"
             )
     ```
   - Purpose: Test alternative integration path if supported
   - Edge cases: Parser may not accept equation tuples
   - Integration: Explores more direct integration option

**Outcomes**:


---

## Task Group 7: Add End-to-End solve_ivp Test (Optional) - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 6

**Required Context**:
- File: tests/odesystems/symbolic/test_cellml.py (current state)
- File: src/cubie/batchsolving/solver.py (solve_ivp function)
- File: tests/fixtures/cellml/basic_ode.cellml (simple model for faster execution)

**Input Validation Required**:
- None (test uses fixtures)

**Tasks**:
1. **Add test_solve_ivp_with_cellml**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify (add new test function)
   - Details:
     ```python
     @pytest.mark.slow
     @pytest.mark.nocudasim  # Requires real GPU
     def test_solve_ivp_with_cellml(simple_cellml_model):
         """End-to-end test: CellML to solve_ivp execution."""
         pytest.importorskip("cellmlmanip")
         
         states, equations = load_cellml_model(simple_cellml_model)
         
         # Convert to format for create_ODE_system
         state_dict = {}
         equation_strings = []
         for eq in equations:
             if isinstance(eq.lhs, sp.Derivative):
                 var = eq.lhs.args[0]
                 var_name = var.name if hasattr(var, 'name') else str(var)
                 state_dict[var_name] = 1.0  # Initial value
                 equation_strings.append(f"d{var_name} = {eq.rhs}")
         
         # Create ODE system
         ode = create_ODE_system(
             dxdt=equation_strings,
             states=state_dict,
             precision=np.float64
         )
         
         # Solve the ODE
         result = solve_ivp(
             ode,
             t_span=(0.0, 1.0),
             dt=0.01,
             dt_save=0.1,
             algorithm='explicit_euler',
             atol=1e-6,
             rtol=1e-6
         )
         
         # Basic validation
         assert result is not None
         assert result.success
         assert len(result.t) > 0
         assert result.y.shape[0] == len(states)
     ```
   - Purpose: Full integration test with GPU execution
   - Edge cases: GPU availability, CUDA errors
   - Integration: Validates complete workflow from CellML to solution
   - Note: Marked slow and nocudasim for CI considerations

**Outcomes**:


---

## Task Group 8: Add Documentation and Examples (Optional) - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Group 7

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/cellml.py (final implementation)
- File: tests/odesystems/symbolic/test_cellml.py (for usage examples)
- Reference: .github/active_plans/cellml_import_testing/agent_plan.md

**Input Validation Required**:
- None (documentation)

**Tasks**:
1. **Enhance load_cellml_model docstring**
   - File: src/cubie/odesystems/symbolic/parsing/cellml.py
   - Action: Modify
   - Details:
     ```python
     def load_cellml_model(path: str) -> tuple[list[sp.Symbol], list[sp.Eq]]:
         """Load a CellML model and extract states and derivatives.
         
         This function uses the cellmlmanip library to parse CellML files
         and extract the state variables and differential equations in a
         format compatible with CuBIE's SymbolicODE system.
         
         Parameters
         ----------
         path
             Filesystem path to the CellML source file. Must have .cellml
             extension and be a valid CellML 1.0 or 1.1 model file.
     
         Returns
         -------
         tuple[list[sympy.Symbol], list[sympy.Eq]]
             A tuple containing:
             - states: List of sympy.Symbol objects representing state variables
             - equations: List of sympy.Eq objects with derivatives on LHS
         
         Raises
         ------
         ImportError
             If cellmlmanip is not installed. Install with: pip install cellmlmanip
         TypeError
             If path is not a string.
         FileNotFoundError
             If the specified CellML file does not exist.
         ValueError
             If the file does not have .cellml extension.
         
         Examples
         --------
         Load a CellML model and create a CuBIE ODE system:
         
         >>> import numpy as np
         >>> from cubie import create_ODE_system
         >>> from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model
         >>> 
         >>> # Load CellML model
         >>> states, equations = load_cellml_model("model.cellml")
         >>> 
         >>> # Convert to create_ODE_system format
         >>> state_dict = {s.name: 0.0 for s in states}
         >>> eq_strings = [f"d{eq.lhs.args[0].name} = {eq.rhs}" for eq in equations]
         >>> 
         >>> # Create ODE system
         >>> ode = create_ODE_system(
         ...     dxdt=eq_strings,
         ...     states=state_dict,
         ...     precision=np.float64
         ... )
         
         Notes
         -----
         - Only differential equations are extracted (algebraic equations filtered)
         - State variables are converted from sympy.Dummy to sympy.Symbol if needed
         - The time variable is extracted from the derivative expressions
         - CellML models from the Physiome repository are supported
         
         See Also
         --------
         create_ODE_system : Create a SymbolicODE from equations
         SymbolicODE : Symbolic ODE system class
         """
         # ... implementation
     ```
   - Purpose: Comprehensive documentation for users
   - Integration: Helps users understand how to use CellML import

2. **Add example usage to module docstring**
   - File: src/cubie/odesystems/symbolic/parsing/cellml.py
   - Action: Modify (enhance module docstring)
   - Details:
     ```python
     """Minimal CellML parsing helpers using ``cellmlmanip``.
     
     This module provides functionality to import CellML models into CuBIE's
     symbolic ODE framework. It wraps the cellmlmanip library to extract
     state variables and differential equations in a format compatible with
     SymbolicODE.
     
     The implementation is inspired by chaste_codegen.model_with_conversions
     from the chaste-codegen project (MIT licence). Only a minimal subset
     required for basic model loading is implemented.
     
     Dependencies
     ------------
     cellmlmanip : optional
         Install with: pip install cellmlmanip
         Required for loading CellML model files
     
     Examples
     --------
     Basic usage to load and simulate a CellML model:
     
     >>> from cubie import create_ODE_system, solve_ivp
     >>> from cubie.odesystems.symbolic.parsing.cellml import load_cellml_model
     >>> import numpy as np
     >>> 
     >>> # Load the CellML model
     >>> states, equations = load_cellml_model("cardiac_model.cellml")
     >>> 
     >>> # Convert to CuBIE format
     >>> state_dict = {s.name: 0.0 for s in states}
     >>> eq_strings = []
     >>> for eq in equations:
     ...     var = eq.lhs.args[0]  # Get variable from derivative
     ...     eq_strings.append(f"d{var.name} = {eq.rhs}")
     >>> 
     >>> # Create and solve the ODE system
     >>> ode = create_ODE_system(dxdt=eq_strings, states=state_dict)
     >>> result = solve_ivp(ode, t_span=(0, 100), dt=0.01)
     
     Notes
     -----
     CellML models can be obtained from the Physiome Model Repository:
     https://models.physiomeproject.org/
     
     This module only supports CellML 1.0 and 1.1 formats. CellML 2.0
     support depends on cellmlmanip capabilities.
     """
     ```
   - Purpose: Module-level documentation and examples
   - Integration: Provides context for the module

**Outcomes**:


---

## Summary

**Total Task Groups**: 8
**Dependency Chain**: Sequential (1 → 2 → 3 → 4 → 5 → 6 → 7 → 8)
**Parallel Execution Opportunities**: None (each group depends on previous)

**Implementation Strategy**:
1. Groups 1-2: Setup infrastructure (fixtures, directory structure)
2. Groups 3-4: Implement and verify basic functionality
3. Group 5: Fix any issues found in testing
4. Group 6: Integration tests with SymbolicODE
5. Group 7: Optional end-to-end GPU test
6. Group 8: Optional documentation enhancements

**Estimated Complexity**:
- Groups 1-4: Low complexity (setup and straightforward tests)
- Group 5: Medium complexity (may need debugging/fixing)
- Group 6: Medium complexity (integration testing)
- Groups 7-8: Low complexity (optional enhancements)

**Critical Decision Points**:
1. After Group 4: Determine if load_cellml_model needs fixes
2. After Group 5: Verify all type conversion issues resolved
3. After Group 6: Decide if string conversion is acceptable or if parser should support equation tuples

**Notes**:
- Groups 7-8 are marked optional but recommended for completeness
- Tests use pytest.importorskip to handle optional cellmlmanip dependency
- Some tests marked @pytest.mark.slow for CI optimization
- GPU tests marked @pytest.mark.nocudasim to skip in CPU simulation mode
