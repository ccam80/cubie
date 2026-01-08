# Implementation Task List
# Feature: Scaled Tolerance in Newton-Krylov Solver
# Plan Reference: .github/active_plans/scaled_nk_tolerance/agent_plan.md

## Task Group 1: LinearSolverConfig Tolerance Arrays
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 1-132)
- File: src/cubie/integrators/step_control/adaptive_step_controller.py (lines 1-56, 72-82)
- File: src/cubie/_utils.py (lines 471-483) - float_array_validator

**Input Validation Required**:
- krylov_atol: Must be numpy ndarray of floats with no NaN/Inf (float_array_validator handles this)
- krylov_rtol: Must be numpy ndarray of floats with no NaN/Inf (float_array_validator handles this)
- Converter handles scalar-to-array broadcasting and shape validation

**Tasks**:
1. **Add tol_converter function to linear_solver.py**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     # Add imports at top of file:
     from numpy import asarray, full, isscalar, ndarray
     from numpy.typing import ArrayLike
     from attrs import Converter
     from cubie._utils import float_array_validator
     
     # Add converter function before LinearSolverConfig class:
     def tol_converter(
         value: Union[float, ArrayLike],
         self_: "LinearSolverConfig",
     ) -> ndarray:
         """Convert tolerance input into an array with solver precision.
     
         Parameters
         ----------
         value
             Scalar or array-like tolerance specification.
         self_
             Configuration instance providing precision and dimension info.
     
         Returns
         -------
         numpy.ndarray
             Tolerance array with one value per state variable.
     
         Raises
         ------
         ValueError
             Raised when ``value`` cannot be broadcast to the expected shape.
         """
         if isscalar(value):
             tol = full(self_.n, value, dtype=self_.precision)
         else:
             tol = asarray(value, dtype=self_.precision)
             if tol.shape[0] == 1 and self_.n > 1:
                 tol = full(self_.n, tol[0], dtype=self_.precision)
             elif tol.shape[0] != self_.n:
                 raise ValueError("tol must have shape (n,).")
         return tol
     ```
   - Edge cases: 
     - Scalar input broadcasts to array of length n
     - Single-element array broadcasts to length n
     - Wrong-length array raises ValueError
   - Integration: Converter used by attrs fields in LinearSolverConfig

2. **Add krylov_atol and krylov_rtol fields to LinearSolverConfig**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     # In LinearSolverConfig class, add after _krylov_tolerance field (around line 81):
     krylov_atol: ndarray = field(
         default=asarray([1e-6]),
         validator=float_array_validator,
         converter=Converter(tol_converter, takes_self=True)
     )
     krylov_rtol: ndarray = field(
         default=asarray([1e-6]),
         validator=float_array_validator,
         converter=Converter(tol_converter, takes_self=True)
     )
     ```
   - Edge cases: Default values are single-element arrays that will broadcast
   - Integration: Fields accessible via compile_settings in build()

3. **Update LinearSolverConfig.settings_dict property**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     # In settings_dict property, add krylov_atol and krylov_rtol:
     @property
     def settings_dict(self) -> Dict[str, Any]:
         return {
             'krylov_tolerance': self.krylov_tolerance,
             'krylov_atol': self.krylov_atol,
             'krylov_rtol': self.krylov_rtol,
             'max_linear_iters': self.max_linear_iters,
             'linear_correction_type': self.linear_correction_type,
             'preconditioned_vec_location': self.preconditioned_vec_location,
             'temp_location': self.temp_location,
         }
     ```
   - Integration: Exposes new parameters for introspection

4. **Add krylov_atol and krylov_rtol properties to LinearSolver class**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     # Add after krylov_tolerance property (around line 587):
     @property
     def krylov_atol(self) -> ndarray:
         """Return absolute tolerance array."""
         return self.compile_settings.krylov_atol
     
     @property
     def krylov_rtol(self) -> ndarray:
         """Return relative tolerance array."""
         return self.compile_settings.krylov_rtol
     ```
   - Integration: Provides public access to tolerance arrays

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_linear_solver.py
- Test function: test_linear_solver_config_scalar_tolerance_broadcast
- Description: Verify scalar krylov_atol/rtol broadcasts to array of length n
- Test function: test_linear_solver_config_array_tolerance_accepted
- Description: Verify array tolerances of correct length are accepted
- Test function: test_linear_solver_config_wrong_length_raises
- Description: Verify wrong-length tolerance array raises ValueError

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_config_scalar_tolerance_broadcast
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_config_array_tolerance_accepted
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_config_wrong_length_raises

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (45 lines added)
  * tests/integrators/matrix_free_solvers/test_linear_solver.py (41 lines added)
- Functions/Methods Added/Modified:
  * tol_converter() in linear_solver.py - converts scalar/array tolerances to arrays
  * krylov_atol field added to LinearSolverConfig
  * krylov_rtol field added to LinearSolverConfig
  * settings_dict property updated in LinearSolverConfig
  * krylov_atol property added to LinearSolver
  * krylov_rtol property added to LinearSolver
- Implementation Summary:
  Added tolerance array configuration to LinearSolverConfig using the 
  tol_converter pattern from adaptive_step_controller.py. Scalar tolerances
  are broadcast to arrays of length n, single-element arrays are broadcast
  to length n, and wrong-length arrays raise ValueError.
- Issues Flagged: None

---

## Task Group 2: LinearSolver.build() Scaled Norm Implementation
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 207-522)
- File: src/cubie/integrators/step_control/adaptive_step_controller.py (lines 22-56) - tol_converter pattern reference

**Input Validation Required**:
- None - tolerance arrays validated during config creation

**Tasks**:
1. **Modify LinearSolver.build() to capture tolerance arrays in closure**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     # In build() method, after extracting krylov_tolerance (around line 228):
     krylov_atol = config.krylov_atol
     krylov_rtol = config.krylov_rtol
     
     # Add after typed_zero definition:
     typed_one = precision_numba(1.0)
     inv_n = precision_numba(1.0 / n)
     tol_floor = precision_numba(1e-16)
     ```
   - Edge cases: Division by zero protected by tol_floor minimum
   - Integration: Arrays captured in device function closure

2. **Replace L2 norm convergence check with scaled norm in linear_solver device function**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     # In linear_solver device function (non-cached variant, around lines 443-448):
     # Replace initial convergence check:
     # OLD:
     # acc = typed_zero
     # for i in range(n_val):
     #     residual_value = rhs[i] - temp[i]
     #     rhs[i] = residual_value
     #     acc += residual_value * residual_value
     # ...
     # converged = acc <= tol_squared
     
     # NEW:
     acc = typed_zero
     for i in range(n_val):
         residual_value = rhs[i] - temp[i]
         rhs[i] = residual_value
         ref_val = x[i]
         abs_ref = ref_val if ref_val >= typed_zero else -ref_val
         tol_i = krylov_atol[i] + krylov_rtol[i] * abs_ref
         tol_i = tol_i if tol_i > tol_floor else tol_floor
         abs_res = residual_value if residual_value >= typed_zero else -residual_value
         ratio = abs_res / tol_i
         acc += ratio * ratio
     acc = acc * inv_n
     mask = activemask()
     converged = acc <= typed_one
     
     # Also update the in-loop convergence check (around lines 502-514):
     # Replace:
     # acc = typed_zero
     # if not converged:
     #     for i in range(n_val):
     #         x[i] += alpha * preconditioned_vec[i]
     #         rhs[i] -= alpha * temp[i]
     #         residual_value = rhs[i]
     #         acc += residual_value * residual_value
     # else:
     #     for i in range(n_val):
     #         residual_value = rhs[i]
     #         acc += residual_value * residual_value
     # converged = converged or (acc <= tol_squared)
     
     # With:
     acc = typed_zero
     if not converged:
         for i in range(n_val):
             x[i] += alpha * preconditioned_vec[i]
             rhs[i] -= alpha * temp[i]
             residual_value = rhs[i]
             ref_val = x[i]
             abs_ref = ref_val if ref_val >= typed_zero else -ref_val
             tol_i = krylov_atol[i] + krylov_rtol[i] * abs_ref
             tol_i = tol_i if tol_i > tol_floor else tol_floor
             abs_res = residual_value if residual_value >= typed_zero else -residual_value
             ratio = abs_res / tol_i
             acc += ratio * ratio
         acc = acc * inv_n
     else:
         for i in range(n_val):
             residual_value = rhs[i]
             ref_val = x[i]
             abs_ref = ref_val if ref_val >= typed_zero else -ref_val
             tol_i = krylov_atol[i] + krylov_rtol[i] * abs_ref
             tol_i = tol_i if tol_i > tol_floor else tol_floor
             abs_res = residual_value if residual_value >= typed_zero else -residual_value
             ratio = abs_res / tol_i
             acc += ratio * ratio
         acc = acc * inv_n
     converged = converged or (acc <= typed_one)
     ```
   - Edge cases: Zero reference values handled via tol_floor; negative values handled via abs
   - Integration: Convergence check now uses scaled norm

3. **Apply same changes to linear_solver_cached device function**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details: Apply identical scaled norm changes to the cached auxiliaries variant (lines 251-364). Same pattern as above.
   - Edge cases: Same as non-cached variant
   - Integration: Both code paths use consistent convergence criterion

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_linear_solver.py
- Test function: test_linear_solver_scaled_tolerance_converges
- Description: Verify linear solver converges with per-element tolerances on a system with mixed-scale variables
- Test function: test_linear_solver_scalar_tolerance_backward_compatible
- Description: Verify scalar tolerance input produces same behavior as before

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_scaled_tolerance_converges
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_scalar_tolerance_backward_compatible
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_placeholder
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_symbolic

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (68 lines changed)
  * tests/integrators/matrix_free_solvers/test_linear_solver.py (77 lines added)
- Functions/Methods Added/Modified:
  * LinearSolver.build() - Added tolerance array extraction (krylov_atol, krylov_rtol)
  * LinearSolver.build() - Added scaled norm constants (typed_one, inv_n, tol_floor)
  * linear_solver_cached device function - Replaced L2 norm with scaled norm in both initial and in-loop checks
  * linear_solver device function - Replaced L2 norm with scaled norm in both initial and in-loop checks
- Implementation Summary:
  Replaced the L2 norm convergence criterion (residual² <= tol²) with a scaled norm
  that uses per-element tolerances: sum((residual[i] / (atol[i] + rtol[i] * |x[i]|))²) / n <= 1.0.
  The reference value for relative tolerance scaling is the current solution x[i].
  Protected against division by zero using tol_floor (1e-16).
  Both cached and non-cached device function variants updated consistently.
- Issues Flagged: None

---

## Task Group 3: NewtonKrylovConfig Tolerance Arrays
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 1-164)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 1-60) - tol_converter pattern

**Input Validation Required**:
- newton_atol: Must be numpy ndarray of floats with no NaN/Inf (float_array_validator)
- newton_rtol: Must be numpy ndarray of floats with no NaN/Inf (float_array_validator)

**Tasks**:
1. **Add tol_converter function to newton_krylov.py**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # Add imports at top of file:
     from numpy import asarray, full, isscalar, ndarray
     from numpy.typing import ArrayLike
     from attrs import Converter
     from cubie._utils import float_array_validator
     
     # Add converter function before NewtonKrylovConfig class:
     def tol_converter(
         value: Union[float, ArrayLike],
         self_: "NewtonKrylovConfig",
     ) -> ndarray:
         """Convert tolerance input into an array with solver precision.
     
         Parameters
         ----------
         value
             Scalar or array-like tolerance specification.
         self_
             Configuration instance providing precision and dimension info.
     
         Returns
         -------
         numpy.ndarray
             Tolerance array with one value per state variable.
     
         Raises
         ------
         ValueError
             Raised when ``value`` cannot be broadcast to the expected shape.
         """
         if isscalar(value):
             tol = full(self_.n, value, dtype=self_.precision)
         else:
             tol = asarray(value, dtype=self_.precision)
             if tol.shape[0] == 1 and self_.n > 1:
                 tol = full(self_.n, tol[0], dtype=self_.precision)
             elif tol.shape[0] != self_.n:
                 raise ValueError("tol must have shape (n,).")
         return tol
     ```
   - Edge cases: Same as LinearSolver tol_converter
   - Integration: Used by attrs Converter for tolerance fields

2. **Add newton_atol and newton_rtol fields to NewtonKrylovConfig**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # In NewtonKrylovConfig class, add after _newton_tolerance field (around line 83):
     newton_atol: ndarray = field(
         default=asarray([1e-3]),
         validator=float_array_validator,
         converter=Converter(tol_converter, takes_self=True)
     )
     newton_rtol: ndarray = field(
         default=asarray([1e-3]),
         validator=float_array_validator,
         converter=Converter(tol_converter, takes_self=True)
     )
     ```
   - Edge cases: Default matches existing newton_tolerance default of 1e-3
   - Integration: Fields accessible via compile_settings in build()

3. **Update NewtonKrylovConfig.settings_dict property**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # In settings_dict property, add newton_atol and newton_rtol:
     @property
     def settings_dict(self) -> Dict[str, Any]:
         return {
             'newton_tolerance': self.newton_tolerance,
             'newton_atol': self.newton_atol,
             'newton_rtol': self.newton_rtol,
             'max_newton_iters': self.max_newton_iters,
             'newton_damping': self.newton_damping,
             'newton_max_backtracks': self.newton_max_backtracks,
             'delta_location': self.delta_location,
             'residual_location': self.residual_location,
             'residual_temp_location': self.residual_temp_location,
             'stage_base_bt_location': self.stage_base_bt_location,
             'krylov_iters_local_location': self.krylov_iters_local_location,
         }
     ```
   - Integration: Exposes new parameters for introspection

4. **Add newton_atol and newton_rtol properties to NewtonKrylov class**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # Add after newton_tolerance property (around line 570):
     @property
     def newton_atol(self) -> ndarray:
         """Return absolute tolerance array."""
         return self.compile_settings.newton_atol
     
     @property
     def newton_rtol(self) -> ndarray:
         """Return relative tolerance array."""
         return self.compile_settings.newton_rtol
     ```
   - Integration: Provides public access to tolerance arrays

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_newton_krylov.py
- Test function: test_newton_krylov_config_scalar_tolerance_broadcast
- Description: Verify scalar newton_atol/rtol broadcasts to array of length n
- Test function: test_newton_krylov_config_array_tolerance_accepted
- Description: Verify array tolerances of correct length are accepted
- Test function: test_newton_krylov_config_wrong_length_raises
- Description: Verify wrong-length tolerance array raises ValueError

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_config_scalar_tolerance_broadcast
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_config_array_tolerance_accepted
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_config_wrong_length_raises

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (50 lines added)
  * tests/integrators/matrix_free_solvers/test_newton_krylov.py (46 lines added)
- Functions/Methods Added/Modified:
  * tol_converter() in newton_krylov.py - converts scalar/array tolerances to arrays
  * newton_atol field added to NewtonKrylovConfig
  * newton_rtol field added to NewtonKrylovConfig
  * settings_dict property updated in NewtonKrylovConfig
  * newton_atol property added to NewtonKrylov
  * newton_rtol property added to NewtonKrylov
- Implementation Summary:
  Added tolerance array configuration to NewtonKrylovConfig using the
  tol_converter pattern from linear_solver.py. Scalar tolerances
  are broadcast to arrays of length n, single-element arrays are broadcast
  to length n, and wrong-length arrays raise ValueError. Default tolerance
  is 1e-3 to match existing newton_tolerance default.
- Issues Flagged: None

---

## Task Group 4: NewtonKrylov.build() Scaled Norm Implementation
**Status**: [x]
**Dependencies**: Task Group 3

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 269-502)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 240-250) - scaled norm pattern

**Input Validation Required**:
- None - tolerance arrays validated during config creation

**Tasks**:
1. **Modify NewtonKrylov.build() to capture tolerance arrays in closure**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # In build() method, after extracting newton_tolerance (around line 290):
     newton_atol = config.newton_atol
     newton_rtol = config.newton_rtol
     
     # Replace tol_squared with scaled norm constants (around line 295-298):
     # OLD:
     # tol_squared = numba_precision(newton_tolerance * newton_tolerance)
     # typed_zero = numba_precision(0.0)
     # typed_one = numba_precision(1.0)
     
     # NEW:
     typed_zero = numba_precision(0.0)
     typed_one = numba_precision(1.0)
     inv_n = numba_precision(1.0 / n)
     tol_floor = numba_precision(1e-16)
     ```
   - Edge cases: Division by zero protected by tol_floor
   - Integration: Arrays captured in device function closure

2. **Replace initial residual norm check with scaled norm**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # In newton_krylov_solver device function (around lines 383-390):
     # Replace:
     # norm2_prev = typed_zero
     # for i in range(n_val):
     #     residual_value = residual[i]
     #     residual[i] = -residual_value
     #     delta[i] = typed_zero
     #     norm2_prev += residual_value * residual_value
     # converged = norm2_prev <= tol_squared
     
     # With:
     norm2_prev = typed_zero
     for i in range(n_val):
         residual_value = residual[i]
         residual[i] = -residual_value
         delta[i] = typed_zero
         ref_val = stage_increment[i]
         abs_ref = ref_val if ref_val >= typed_zero else -ref_val
         tol_i = newton_atol[i] + newton_rtol[i] * abs_ref
         tol_i = tol_i if tol_i > tol_floor else tol_floor
         abs_res = residual_value if residual_value >= typed_zero else -residual_value
         ratio = abs_res / tol_i
         norm2_prev += ratio * ratio
     norm2_prev = norm2_prev * inv_n
     converged = norm2_prev <= typed_one
     ```
   - Edge cases: Zero stage_increment values handled via tol_floor
   - Integration: Initial convergence check uses scaled norm

3. **Replace backtracking residual norm check with scaled norm**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # In backtracking loop (around lines 456-470):
     # Replace:
     # norm2_new = typed_zero
     # for i in range(n_val):
     #     residual_value = residual_temp[i]
     #     norm2_new += residual_value * residual_value
     # if norm2_new <= tol_squared:
     #     converged = True
     #     found_step = True
     # if norm2_new < norm2_prev:
     #     ...
     
     # With:
     norm2_new = typed_zero
     for i in range(n_val):
         residual_value = residual_temp[i]
         ref_val = stage_increment[i]
         abs_ref = ref_val if ref_val >= typed_zero else -ref_val
         tol_i = newton_atol[i] + newton_rtol[i] * abs_ref
         tol_i = tol_i if tol_i > tol_floor else tol_floor
         abs_res = residual_value if residual_value >= typed_zero else -residual_value
         ratio = abs_res / tol_i
         norm2_new += ratio * ratio
     norm2_new = norm2_new * inv_n
     
     if norm2_new <= typed_one:
         converged = True
         found_step = True
     
     if norm2_new < norm2_prev:
         for i in range(n_val):
             residual[i] = -residual_temp[i]
         norm2_prev = norm2_new
         found_step = True
     ```
   - Edge cases: Uses updated stage_increment for reference values
   - Integration: Backtracking uses same scaled norm as initial check

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_newton_krylov.py
- Test function: test_newton_krylov_scaled_tolerance_converges
- Description: Verify Newton solver converges with per-element tolerances on a mixed-scale system
- Test function: test_newton_krylov_scalar_tolerance_backward_compatible
- Description: Verify scalar tolerance input produces same behavior as before

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_scaled_tolerance_converges
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_scalar_tolerance_backward_compatible
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_placeholder
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_symbolic

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (30 lines changed)
  * tests/integrators/matrix_free_solvers/test_newton_krylov.py (138 lines added)
- Functions/Methods Added/Modified:
  * NewtonKrylov.build() - Added tolerance array extraction (newton_atol, newton_rtol)
  * NewtonKrylov.build() - Added scaled norm constants (typed_one, inv_n, tol_floor), removed tol_squared
  * newton_krylov_solver device function - Replaced L2 norm with scaled norm in initial check
  * newton_krylov_solver device function - Replaced L2 norm with scaled norm in backtracking check
- Implementation Summary:
  Replaced the L2 norm convergence criterion (residual² <= tol²) with a scaled norm
  that uses per-element tolerances: sum((residual[i] / (atol[i] + rtol[i] * |stage_increment[i]|))²) / n <= 1.0.
  The reference value for relative tolerance scaling is the stage_increment[i].
  Protected against division by zero using tol_floor (1e-16).
  Both initial and backtracking convergence checks use scaled norm.
- Issues Flagged: None

---

## Task Group 5: ODEImplicitStep Parameter Routing
**Status**: [x]
**Dependencies**: Task Groups 1, 3

**Required Context**:
- File: src/cubie/integrators/algorithms/ode_implicitstep.py (lines 84-106, 137-162)

**Input Validation Required**:
- None - validation handled by solver configs

**Tasks**:
1. **Add tolerance parameters to _LINEAR_SOLVER_PARAMS frozenset**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     # Update _LINEAR_SOLVER_PARAMS (around line 88):
     _LINEAR_SOLVER_PARAMS = frozenset({
         'linear_correction_type',
         'krylov_tolerance',
         'krylov_atol',
         'krylov_rtol',
         'max_linear_iters',
         'preconditioned_vec_location',
         'temp_location',
     })
     ```
   - Integration: Enables kwargs routing from algorithm to linear solver

2. **Add tolerance parameters to _NEWTON_KRYLOV_PARAMS frozenset**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     # Update _NEWTON_KRYLOV_PARAMS (around line 97):
     _NEWTON_KRYLOV_PARAMS = frozenset({
         'newton_tolerance',
         'newton_atol',
         'newton_rtol',
         'max_newton_iters',
         'newton_damping',
         'newton_max_backtracks',
         'delta_location',
         'residual_location',
         'residual_temp_location',
         'stage_base_bt_location',
     })
     ```
   - Integration: Enables kwargs routing from algorithm to Newton solver

3. **Add tolerance array properties to ODEImplicitStep**
   - File: src/cubie/integrators/algorithms/ode_implicitstep.py
   - Action: Modify
   - Details:
     ```python
     # Add after krylov_tolerance property (around line 358):
     @property
     def krylov_atol(self) -> ndarray:
         """Return the absolute tolerance array for linear solve."""
         return self.solver.krylov_atol
     
     @property
     def krylov_rtol(self) -> ndarray:
         """Return the relative tolerance array for linear solve."""
         return self.solver.krylov_rtol
     
     # Add after newton_tolerance property (around line 375):
     @property
     def newton_atol(self) -> Optional[ndarray]:
         """Return the Newton absolute tolerance array."""
         return getattr(self.solver, 'newton_atol', None)
     
     @property
     def newton_rtol(self) -> Optional[ndarray]:
         """Return the Newton relative tolerance array."""
         return getattr(self.solver, 'newton_rtol', None)
     ```
   - Edge cases: Returns None for linear-only solver (no Newton layer)
   - Integration: Exposes tolerance arrays through algorithm interface

**Tests to Create**:
- Test file: tests/integrators/algorithms/test_ode_implicitstep.py (create if needed)
- Test function: test_implicit_step_accepts_tolerance_arrays
- Description: Verify implicit step forwards tolerance arrays to nested solvers
- Test function: test_implicit_step_exposes_tolerance_properties
- Description: Verify tolerance array properties return correct values

**Tests to Run**:
- tests/integrators/algorithms/test_ode_implicitstep.py::test_implicit_step_accepts_tolerance_arrays
- tests/integrators/algorithms/test_ode_implicitstep.py::test_implicit_step_exposes_tolerance_properties
- tests/integrators/algorithms/test_ode_implicitstep.py::test_implicit_step_linear_solver_newton_atol_returns_none

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/algorithms/ode_implicitstep.py (20 lines changed)
  * tests/integrators/algorithms/test_ode_implicitstep.py (77 lines added - new file)
- Functions/Methods Added/Modified:
  * _LINEAR_SOLVER_PARAMS frozenset updated with krylov_atol, krylov_rtol
  * _NEWTON_KRYLOV_PARAMS frozenset updated with newton_atol, newton_rtol
  * krylov_atol property added to ODEImplicitStep
  * krylov_rtol property added to ODEImplicitStep
  * newton_atol property added to ODEImplicitStep
  * newton_rtol property added to ODEImplicitStep
- Implementation Summary:
  Added tolerance array parameter routing to ODEImplicitStep. The frozensets
  _LINEAR_SOLVER_PARAMS and _NEWTON_KRYLOV_PARAMS now include the new tolerance
  array parameters, enabling kwargs routing from algorithm constructors to the
  nested LinearSolver and NewtonKrylov solvers. Added four new properties to
  expose the tolerance arrays through the algorithm interface. newton_atol and
  newton_rtol use getattr with default None to handle linear-only solver cases.
- Issues Flagged: None 

---

## Task Group 6: Instrumented Test File Updates
**Status**: [ ]
**Dependencies**: Task Groups 2, 4

**Required Context**:
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 240-520) - updated device function
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 280-500) - updated device function

**Input Validation Required**:
- None - instrumented files mirror source files

**Tasks**:
1. **Update InstrumentedLinearSolver.build() with scaled norm**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify
   - Details:
     - Add tolerance array extraction from config (krylov_atol, krylov_rtol)
     - Add typed_one, inv_n, tol_floor constants
     - Replace all L2 norm convergence checks with scaled norm pattern
     - Apply to both cached and non-cached variants
     - Keep all logging code intact
   - Edge cases: Same as source file
   - Integration: Instrumented version matches source behavior

2. **Update InstrumentedNewtonKrylov.build() with scaled norm**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify
   - Details:
     - Add tolerance array extraction from config (newton_atol, newton_rtol)
     - Add typed_one, inv_n, tol_floor constants (remove tol_squared)
     - Replace initial and backtracking norm checks with scaled norm
     - Keep all logging code intact
   - Edge cases: Same as source file
   - Integration: Instrumented version matches source behavior

**Tests to Create**:
- None - existing instrumented tests will validate

**Tests to Run**:
- tests/integrators/algorithms/instrumented/test_instrumented.py

**Outcomes**: 

---

## Task Group 7: Integration Tests
**Status**: [ ]
**Dependencies**: Task Groups 1-6

**Required Context**:
- File: tests/integrators/matrix_free_solvers/test_linear_solver.py (entire file)
- File: tests/integrators/matrix_free_solvers/test_newton_krylov.py (entire file)
- File: tests/integrators/matrix_free_solvers/conftest.py (entire file)

**Input Validation Required**:
- None - tests validate implementation

**Tasks**:
1. **Add scaled tolerance tests for LinearSolver**
   - File: tests/integrators/matrix_free_solvers/test_linear_solver.py
   - Action: Modify
   - Details:
     ```python
     def test_linear_solver_config_scalar_tolerance_broadcast(precision):
         """Verify scalar krylov_atol/rtol broadcasts to array of length n."""
         n = 5
         solver = LinearSolver(
             precision=precision,
             n=n,
             krylov_atol=1e-6,
             krylov_rtol=1e-4,
         )
         assert solver.krylov_atol.shape == (n,)
         assert solver.krylov_rtol.shape == (n,)
         assert np.all(solver.krylov_atol == precision(1e-6))
         assert np.all(solver.krylov_rtol == precision(1e-4))
     
     
     def test_linear_solver_config_array_tolerance_accepted(precision):
         """Verify array tolerances of correct length are accepted."""
         n = 3
         atol = np.array([1e-6, 1e-8, 1e-4], dtype=precision)
         rtol = np.array([1e-3, 1e-5, 1e-2], dtype=precision)
         solver = LinearSolver(
             precision=precision,
             n=n,
             krylov_atol=atol,
             krylov_rtol=rtol,
         )
         assert np.allclose(solver.krylov_atol, atol)
         assert np.allclose(solver.krylov_rtol, rtol)
     
     
     def test_linear_solver_config_wrong_length_raises(precision):
         """Verify wrong-length tolerance array raises ValueError."""
         n = 3
         wrong_atol = np.array([1e-6, 1e-8], dtype=precision)  # length 2
         with pytest.raises(ValueError, match="tol must have shape"):
             LinearSolver(
                 precision=precision,
                 n=n,
                 krylov_atol=wrong_atol,
             )
     ```
   - Integration: Validates config accepts both scalars and arrays

2. **Add scaled tolerance tests for NewtonKrylov**
   - File: tests/integrators/matrix_free_solvers/test_newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     def test_newton_krylov_config_scalar_tolerance_broadcast(precision):
         """Verify scalar newton_atol/rtol broadcasts to array of length n."""
         n = 5
         linear_solver = LinearSolver(precision=precision, n=n)
         newton = NewtonKrylov(
             precision=precision,
             n=n,
             linear_solver=linear_solver,
             newton_atol=1e-6,
             newton_rtol=1e-4,
         )
         assert newton.newton_atol.shape == (n,)
         assert newton.newton_rtol.shape == (n,)
         assert np.all(newton.newton_atol == precision(1e-6))
         assert np.all(newton.newton_rtol == precision(1e-4))
     
     
     def test_newton_krylov_config_array_tolerance_accepted(precision):
         """Verify array tolerances of correct length are accepted."""
         n = 3
         atol = np.array([1e-6, 1e-8, 1e-4], dtype=precision)
         rtol = np.array([1e-3, 1e-5, 1e-2], dtype=precision)
         linear_solver = LinearSolver(precision=precision, n=n)
         newton = NewtonKrylov(
             precision=precision,
             n=n,
             linear_solver=linear_solver,
             newton_atol=atol,
             newton_rtol=rtol,
         )
         assert np.allclose(newton.newton_atol, atol)
         assert np.allclose(newton.newton_rtol, rtol)
     
     
     def test_newton_krylov_config_wrong_length_raises(precision):
         """Verify wrong-length tolerance array raises ValueError."""
         n = 3
         wrong_atol = np.array([1e-6, 1e-8], dtype=precision)  # length 2
         linear_solver = LinearSolver(precision=precision, n=n)
         with pytest.raises(ValueError, match="tol must have shape"):
             NewtonKrylov(
                 precision=precision,
                 n=n,
                 linear_solver=linear_solver,
                 newton_atol=wrong_atol,
             )
     ```
   - Integration: Validates config accepts both scalars and arrays

3. **Add backward compatibility tests**
   - File: tests/integrators/matrix_free_solvers/test_linear_solver.py
   - Action: Modify
   - Details:
     ```python
     @pytest.mark.parametrize(
         "solver_device", ["steepest_descent", "minimal_residual"], indirect=True
     )
     def test_linear_solver_scalar_tolerance_backward_compatible(
         solver_device,
         solver_kernel,
         precision,
         tolerance,
     ):
         """Verify scalar tolerance produces convergent behavior."""
         # Same test as test_linear_solver_placeholder but validates scalar tol works
         rhs = np.array([1.0, 2.0, 3.0], dtype=precision)
         matrix = np.array(
             [[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]],
             dtype=precision,
         )
         expected = np.linalg.solve(matrix, rhs)
         h = precision(0.01)
         kernel = solver_kernel(solver_device, 3, h, precision)
         base_state = np.array([1.0, -1.0, 0.5], dtype=precision)
         state = cuda.to_device(
             base_state + h * np.array([0.1, -0.2, 0.3], dtype=precision)
         )
         rhs_dev = cuda.to_device(rhs)
         x_dev = cuda.to_device(np.zeros(3, dtype=precision))
         flag = cuda.to_device(np.array([0], dtype=np.int32))
         empty_base = cuda.to_device(np.empty(0, dtype=precision))
         kernel[1, 1](state, rhs_dev, empty_base, x_dev, flag)
         code = flag.copy_to_host()[0] & 0xFF
         assert code == SolverRetCodes.SUCCESS
         assert np.allclose(
             x_dev.copy_to_host(),
             expected,
             rtol=tolerance.rel_tight,
             atol=tolerance.abs_tight,
         )
     ```
   - Integration: Ensures scalar tolerance still works after changes

**Tests to Create**:
- Listed in tasks above

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_linear_solver.py
- tests/integrators/matrix_free_solvers/test_newton_krylov.py

**Outcomes**: 

---

## Summary

**Total Task Groups**: 7

**Dependency Chain**:
```
Task Group 1 (LinearSolverConfig) ──┬──> Task Group 2 (LinearSolver.build())
                                    │
                                    ├──> Task Group 5 (ODEImplicitStep)
                                    │
Task Group 3 (NewtonKrylovConfig) ──┼──> Task Group 4 (NewtonKrylov.build())
                                    │
                                    └──> Task Group 5 (ODEImplicitStep)

Task Groups 2, 4 ──> Task Group 6 (Instrumented Tests)

Task Groups 1-6 ──> Task Group 7 (Integration Tests)
```

**Tests to Create**:
- test_linear_solver_config_scalar_tolerance_broadcast
- test_linear_solver_config_array_tolerance_accepted
- test_linear_solver_config_wrong_length_raises
- test_linear_solver_scaled_tolerance_converges
- test_linear_solver_scalar_tolerance_backward_compatible
- test_newton_krylov_config_scalar_tolerance_broadcast
- test_newton_krylov_config_array_tolerance_accepted
- test_newton_krylov_config_wrong_length_raises
- test_newton_krylov_scaled_tolerance_converges
- test_newton_krylov_scalar_tolerance_backward_compatible
- test_implicit_step_accepts_tolerance_arrays
- test_implicit_step_exposes_tolerance_properties

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_linear_solver.py
- tests/integrators/matrix_free_solvers/test_newton_krylov.py
- tests/integrators/algorithms/instrumented/test_instrumented.py
- tests/integrators/algorithms/test_ode_implicitstep.py

**Estimated Complexity**: Medium
- Config changes are straightforward (follows established pattern)
- Device function changes require careful attention to CUDA constraints
- Instrumented file updates must mirror source changes exactly
