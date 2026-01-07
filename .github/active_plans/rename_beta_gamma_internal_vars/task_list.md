# Implementation Task List
# Feature: Rename Beta/Gamma Internal Variables
# Plan Reference: .github/active_plans/rename_beta_gamma_internal_vars/agent_plan.md

## Task Group 1: Update linear_operators.py Templates and SymPy Symbols
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/linear_operators.py (lines 46-113, 200-283, 556-807)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- None (code generation only, no runtime inputs)

**Tasks**:
1. **Rename beta/gamma in CACHED_OPERATOR_APPLY_TEMPLATE**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - Line 60: Change `"    beta = precision(beta)\n"` to `"    _cubie_codegen_beta = precision(beta)\n"`
     - Line 61: Change `"    gamma = precision(gamma)\n"` to `"    _cubie_codegen_gamma = precision(gamma)\n"`
   - Edge cases: None - template string replacement only
   - Integration: Template is used by `generate_cached_operator_apply_code_from_jvp()`

2. **Rename beta/gamma in OPERATOR_APPLY_TEMPLATE**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - Line 95: Change `"    beta = precision(beta)\n"` to `"    _cubie_codegen_beta = precision(beta)\n"`
     - Line 96: Change `"    gamma = precision(gamma)\n"` to `"    _cubie_codegen_gamma = precision(gamma)\n"`
   - Edge cases: None - template string replacement only
   - Integration: Template is used by `generate_operator_apply_code_from_jvp()`

3. **Rename beta/gamma in N_STAGE_OPERATOR_TEMPLATE**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - Line 789: Change `"    gamma = precision(gamma)\n"` to `"    _cubie_codegen_gamma = precision(gamma)\n"`
     - Line 790: Change `"    beta = precision(beta)\n"` to `"    _cubie_codegen_beta = precision(beta)\n"`
   - Edge cases: None - template string replacement only
   - Integration: Template is used by `generate_n_stage_linear_operator_code()`

4. **Update SymPy symbols in _build_operator_body()**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - Line 214: Change `beta_sym = sp.Symbol("beta")` to `beta_sym = sp.Symbol("_cubie_codegen_beta")`
     - Line 215: Change `gamma_sym = sp.Symbol("gamma")` to `gamma_sym = sp.Symbol("_cubie_codegen_gamma")`
   - Edge cases: Symbol names must match template variable names exactly
   - Integration: Function builds operator body using these symbols; SymPy printer will output the new names

5. **Update SymPy symbols in _build_n_stage_operator_lines()**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - Line 578: Change `beta_sym = sp.Symbol("beta")` to `beta_sym = sp.Symbol("_cubie_codegen_beta")`
     - Line 579: Change `gamma_sym = sp.Symbol("gamma")` to `gamma_sym = sp.Symbol("_cubie_codegen_gamma")`
   - Edge cases: Symbol names must match template variable names exactly
   - Integration: Function builds n-stage operator using these symbols

6. **Update symbol_map in _build_n_stage_operator_lines()**
   - File: src/cubie/odesystems/symbolic/codegen/linear_operators.py
   - Action: Modify
   - Details:
     - Line 714: Change `"beta": beta_sym,` to `"_cubie_codegen_beta": beta_sym,`
     - Line 715: Change `"gamma": gamma_sym,` to `"_cubie_codegen_gamma": gamma_sym,`
   - Edge cases: Keys must match symbol names for SymPy printer
   - Integration: symbol_map passed to print_cuda_multiple for code generation

**Tests to Create**:
None - existing tests will validate functional correctness

**Tests to Run**:
- tests/odesystems/symbolic/test_solver_helpers.py
- tests/integrators/algorithms/test_implicit_algorithms.py

**Outcomes**:
- Files Modified:
  * src/cubie/odesystems/symbolic/codegen/linear_operators.py (6 edits across 6 locations)
- Functions/Methods Modified:
  * CACHED_OPERATOR_APPLY_TEMPLATE (lines 60-61): Renamed beta/gamma variables
  * OPERATOR_APPLY_TEMPLATE (lines 95-96): Renamed beta/gamma variables
  * N_STAGE_OPERATOR_TEMPLATE (lines 789-790): Renamed beta/gamma variables
  * _build_operator_body() (lines 214-215): Updated SymPy symbol definitions
  * _build_n_stage_operator_lines() (lines 578-579): Updated SymPy symbol definitions
  * _build_n_stage_operator_lines() (lines 715-716): Updated symbol_map keys
- Implementation Summary:
  All beta and gamma internal variables in code generation templates and SymPy symbol definitions have been successfully renamed to _cubie_codegen_beta and _cubie_codegen_gamma. This prevents naming conflicts when users define ODE systems with state variables or parameters named "beta" or "gamma". The changes maintain consistency between template variable assignments, SymPy symbol definitions, and symbol_map keys used for code printing.
- Issues Flagged: None

---

## Task Group 2: Update preconditioners.py Templates and SymPy Symbols
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/preconditioners.py (lines 42-136, 138-182, 184-260, 261-420)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- None (code generation only, no runtime inputs)

**Tasks**:
1. **Rename beta/gamma and derived variables in NEUMANN_TEMPLATE**
   - File: src/cubie/odesystems/symbolic/codegen/preconditioners.py
   - Action: Modify
   - Details:
     - Line 53: Change `"    gamma = precision(gamma)\n"` to `"    _cubie_codegen_gamma = precision(gamma)\n"`
     - Line 54: Change `"    beta = precision(beta)\n"` to `"    _cubie_codegen_beta = precision(beta)\n"`
     - Line 56: Change `"    beta_inv = precision(1.0 / beta)\n"` to `"    beta_inv = precision(1.0 / _cubie_codegen_beta)\n"`
     - Line 57: Change `"    h_eff_factor = precision(gamma * beta_inv)\n"` to `"    h_eff_factor = precision(_cubie_codegen_gamma * beta_inv)\n"`
   - Edge cases: beta_inv and h_eff_factor must reference renamed variables
   - Integration: Template is used by `generate_neumann_preconditioner_code()`

2. **Rename beta/gamma and derived variables in NEUMANN_CACHED_TEMPLATE**
   - File: src/cubie/odesystems/symbolic/codegen/preconditioners.py
   - Action: Modify
   - Details:
     - Line 103: Change `"    gamma = precision(gamma)\n"` to `"    _cubie_codegen_gamma = precision(gamma)\n"`
     - Line 104: Change `"    beta = precision(beta)\n"` to `"    _cubie_codegen_beta = precision(beta)\n"`
     - Line 105: Change `"    beta_inv = precision(1.0 / beta)\n"` to `"    beta_inv = precision(1.0 / _cubie_codegen_beta)\n"`
     - Line 106: Change `"    h_eff_factor = precision(gamma * beta_inv)\n"` to `"    h_eff_factor = precision(_cubie_codegen_gamma * beta_inv)\n"`
   - Edge cases: beta_inv and h_eff_factor must reference renamed variables
   - Integration: Template is used by `generate_neumann_preconditioner_cached_code()`

3. **Rename beta/gamma and derived variables in N_STAGE_NEUMANN_TEMPLATE**
   - File: src/cubie/odesystems/symbolic/codegen/preconditioners.py
   - Action: Modify
   - Details:
     - Line 152: Change `"    gamma = precision(gamma)\n"` to `"    _cubie_codegen_gamma = precision(gamma)\n"`
     - Line 153: Change `"    beta = precision(beta)\n"` to `"    _cubie_codegen_beta = precision(beta)\n"`
     - Line 155: Change `"    beta_inv = precision(1.0 / beta)\n"` to `"    beta_inv = precision(1.0 / _cubie_codegen_beta)\n"`
     - Line 156: Change `"    h_eff_factor = precision(gamma * beta_inv)\n"` to `"    h_eff_factor = precision(_cubie_codegen_gamma * beta_inv)\n"`
   - Edge cases: beta_inv and h_eff_factor must reference renamed variables
   - Integration: Template is used by `generate_n_stage_neumann_preconditioner_code()`

**Tests to Create**:
None - existing tests will validate functional correctness

**Tests to Run**:
- tests/odesystems/symbolic/test_solver_helpers.py
- tests/integrators/algorithms/test_implicit_algorithms.py

**Outcomes**:
- Files Modified:
  * src/cubie/odesystems/symbolic/codegen/preconditioners.py (3 edits across 3 templates)
- Functions/Methods Modified:
  * NEUMANN_TEMPLATE (lines 53-54, 56-57): Renamed beta/gamma variables and updated derived variable calculations
  * NEUMANN_CACHED_TEMPLATE (lines 103-106): Renamed beta/gamma variables and updated derived variable calculations
  * N_STAGE_NEUMANN_TEMPLATE (lines 152-153, 155-156): Renamed beta/gamma variables and updated derived variable calculations
- Implementation Summary:
  All beta and gamma internal variables in preconditioner code generation templates have been successfully renamed to _cubie_codegen_beta and _cubie_codegen_gamma. The derived variables beta_inv and h_eff_factor have been updated to reference the renamed variables, maintaining correct mathematical relationships. This prevents naming conflicts when users define ODE systems with state variables or parameters named "beta" or "gamma".
- Issues Flagged: None

---

## Task Group 3: Update nonlinear_residuals.py Templates and SymPy Symbols
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py (lines 35-90, 63-89, 92-177, 180-300)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- None (code generation only, no runtime inputs)

**Tasks**:
1. **Rename beta/gamma in RESIDUAL_TEMPLATE**
   - File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py
   - Action: Modify
   - Details:
     - Line 43: Change `"    beta = precision(beta)\n"` to `"    _cubie_codegen_beta = precision(beta)\n"`
     - Line 44: Change `"    gamma = precision(gamma)\n"` to `"    _cubie_codegen_gamma = precision(gamma)\n"`
   - Edge cases: None - template string replacement only
   - Integration: Template is used by `generate_residual_code()`

2. **Rename beta/gamma in N_STAGE_RESIDUAL_TEMPLATE**
   - File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py
   - Action: Modify
   - Details:
     - Line 71: Change `"    beta = precision(beta)\n"` to `"    _cubie_codegen_beta = precision(beta)\n"`
     - Line 72: Change `"    gamma = precision(gamma)\n"` to `"    _cubie_codegen_gamma = precision(gamma)\n"`
   - Edge cases: None - template string replacement only
   - Integration: Template is used by `generate_n_stage_residual_code()`

3. **Update SymPy symbols in _build_residual_lines()**
   - File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py
   - Action: Modify
   - Details:
     - Line 104: Change `beta_sym = sp.Symbol("beta")` to `beta_sym = sp.Symbol("_cubie_codegen_beta")`
     - Line 105: Change `gamma_sym = sp.Symbol("gamma")` to `gamma_sym = sp.Symbol("_cubie_codegen_gamma")`
   - Edge cases: Symbol names must match template variable names exactly
   - Integration: Function builds residual expression using these symbols

4. **Update symbol_map in _build_residual_lines()**
   - File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py
   - Action: Modify
   - Details:
     - Line 144: Change `"beta": beta_sym,` to `"_cubie_codegen_beta": beta_sym,`
     - Line 145: Change `"gamma": gamma_sym,` to `"_cubie_codegen_gamma": gamma_sym,`
   - Edge cases: Keys must match symbol names for SymPy printer
   - Integration: symbol_map passed to print_cuda_multiple for code generation

5. **Update SymPy symbols in _build_n_stage_residual_lines()**
   - File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py
   - Action: Modify
   - Details:
     - Line 201: Change `beta_sym = sp.Symbol("beta")` to `beta_sym = sp.Symbol("_cubie_codegen_beta")`
     - Line 202: Change `gamma_sym = sp.Symbol("gamma")` to `gamma_sym = sp.Symbol("_cubie_codegen_gamma")`
   - Edge cases: Symbol names must match template variable names exactly
   - Integration: Function builds n-stage residual using these symbols

6. **Update symbol_map in _build_n_stage_residual_lines()**
   - File: src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py
   - Action: Modify
   - Details:
     - Line 289: Change `"beta": beta_sym,` to `"_cubie_codegen_beta": beta_sym,`
     - Line 290: Change `"gamma": gamma_sym,` to `"_cubie_codegen_gamma": gamma_sym,`
   - Edge cases: Keys must match symbol names for SymPy printer
   - Integration: symbol_map passed to print_cuda_multiple for code generation

**Tests to Create**:
None - existing tests will validate functional correctness

**Tests to Run**:
- tests/odesystems/symbolic/test_solver_helpers.py
- tests/integrators/algorithms/test_implicit_algorithms.py

**Outcomes**:
- Files Modified:
  * src/cubie/odesystems/symbolic/codegen/nonlinear_residuals.py (6 edits across 6 locations)
- Functions/Methods Modified:
  * RESIDUAL_TEMPLATE (lines 43-44): Renamed beta/gamma variables
  * N_STAGE_RESIDUAL_TEMPLATE (lines 71-72): Renamed beta/gamma variables
  * _build_residual_lines() (lines 104-105): Updated SymPy symbol definitions
  * _build_residual_lines() (lines 144-145): Updated symbol_map keys
  * _build_n_stage_residual_lines() (lines 201-202): Updated SymPy symbol definitions
  * _build_n_stage_residual_lines() (lines 289-290): Updated symbol_map keys
- Implementation Summary:
  All beta and gamma internal variables in nonlinear residual code generation templates and SymPy symbol definitions have been successfully renamed to _cubie_codegen_beta and _cubie_codegen_gamma. This prevents naming conflicts when users define ODE systems with state variables or parameters named "beta" or "gamma". The changes maintain consistency between template variable assignments, SymPy symbol definitions, and symbol_map keys used for code printing.
- Issues Flagged: None

---

## Task Group 4: Comprehensive Test Validation
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: tests/odesystems/symbolic/test_solver_helpers.py (entire file)
- File: tests/integrators/algorithms/test_implicit_algorithms.py (entire file)
- File: .github/context/cubie_internal_structure.md (lines 399-427 - Testing Infrastructure)

**Input Validation Required**:
- None (test execution only)

**Tasks**:
1. **Run symbolic ODE system tests**
   - File: N/A (test execution)
   - Action: Run tests
   - Details:
     - Execute: `pytest tests/odesystems/symbolic/test_solver_helpers.py -v`
     - Verify all tests pass
     - Check for any warnings related to beta/gamma symbols
   - Edge cases: Tests may inspect generated code strings; verify they don't fail on renamed variables
   - Integration: Validates that code generation produces correct CUDA kernels

2. **Run implicit algorithm tests**
   - File: N/A (test execution)
   - Action: Run tests
   - Details:
     - Execute: `pytest tests/integrators/algorithms/test_implicit_algorithms.py -v`
     - Verify all implicit methods (Backwards Euler, Crank-Nicolson, DIRK) work correctly
     - Confirm solver helpers integrate properly with renamed variables
   - Edge cases: Tests use systems with various parameter names including potential beta/gamma
   - Integration: End-to-end validation of implicit solvers using renamed code generation

3. **Run full ODE systems test suite**
   - File: N/A (test execution)
   - Action: Run tests
   - Details:
     - Execute: `pytest tests/odesystems/ -v`
     - Verify symbolic code generation works for all test cases
     - Confirm no regression in functionality
   - Edge cases: Various ODE systems with different state/parameter configurations
   - Integration: Complete validation of ODE system functionality

4. **Create test case with beta/gamma as user variables**
   - File: tests/odesystems/symbolic/test_solver_helpers.py
   - Action: Create
   - Details:
     ```python
     def test_user_beta_gamma_variables():
         """Test that user can define beta and gamma as state variables or parameters.
         
         This test validates issue #373 fix: internal codegen variables
         are now prefixed with _cubie_codegen_ to avoid conflicts.
         """
         import sympy as sp
         from cubie import create_ODE_system
         
         # Define system with beta and gamma as state variables
         beta = sp.Symbol('beta')
         gamma = sp.Symbol('gamma')
         alpha = sp.Symbol('alpha')  # parameter
         
         equations = {
             beta: -alpha * beta + gamma,
             gamma: alpha * beta - gamma
         }
         
         parameters = {'alpha': 1.0}
         initial_values = {'beta': 1.0, 'gamma': 0.0}
         
         # This should not raise any naming conflicts
         system = create_ODE_system(
             equations=equations,
             parameters=parameters,
             initial_values=initial_values
         )
         
         # Verify system compiles successfully
         assert system.dxdt is not None
         
         # Test with implicit solver requiring solver helpers
         from cubie import solve_ivp
         result = solve_ivp(
             system,
             t_span=(0.0, 1.0),
             algorithm='BackwardsEuler',
             dt=0.01,
             precision=np.float64
         )
         
         assert result.success
         assert result.states.shape[0] > 0
     ```
   - Edge cases: User variables named exactly "beta" and "gamma" should work
   - Integration: Validates complete fix for issue #373

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_solver_helpers.py
- Test function: test_user_beta_gamma_variables
- Description: Verify that users can define ODE systems with state variables or parameters named "beta" and "gamma" without naming conflicts with internal code generation variables

**Tests to Run**:
- tests/odesystems/symbolic/test_solver_helpers.py::test_user_beta_gamma_variables
- tests/odesystems/symbolic/test_solver_helpers.py
- tests/integrators/algorithms/test_implicit_algorithms.py
- tests/odesystems/ (full suite)

**Outcomes**:
[Empty - to be filled by taskmaster agent]
