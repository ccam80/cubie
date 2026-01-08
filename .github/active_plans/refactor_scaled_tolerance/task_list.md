# Implementation Task List
# Feature: Refactor Scaled Tolerance in Newton and Krylov Solvers
# Plan Reference: .github/active_plans/refactor_scaled_tolerance/agent_plan.md

## Task Group 1: Extract tol_converter to _utils.py
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/_utils.py (entire file - understand existing utilities and patterns)
- File: src/cubie/integrators/step_control/adaptive_step_controller.py (lines 22-56 - tol_converter definition)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 35-67 - tol_converter definition)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 33-65 - tol_converter definition)

**Input Validation Required**:
- value: Check if scalar using numpy.isscalar; if not scalar, validate shape matches expected (n,)
- self_: Must have `n` (int) and `precision` (PrecisionDType) attributes

**Tasks**:
1. **Add tol_converter function to _utils.py**
   - File: src/cubie/_utils.py
   - Action: Add function after line 528 (after getype_validator definition)
   - Details:
     ```python
     def tol_converter(
         value: Union[float, ArrayLike],
         self_: Any,
     ) -> ndarray:
         """Convert tolerance input into an array with target precision.
     
         For use as an attrs Converter with takes_self=True, converting
         scalar or array-like tolerance specifications into arrays of
         shape (n,) with dtype matching self_.precision.
     
         Parameters
         ----------
         value
             Scalar or array-like tolerance specification.
         self_
             Configuration instance providing precision and dimension
             information. Must have `n` (int) and `precision` attributes.
     
         Returns
         -------
         numpy.ndarray
             Tolerance array with one value per state variable.
     
         Raises
         ------
         ValueError
             Raised when ``value`` cannot be broadcast to shape (n,).
         """
         if isscalar(value):
             tol = full(self_.n, value, dtype=self_.precision)
         else:
             tol = asarray(value, dtype=self_.precision)
             # Broadcast single-element arrays to shape (n,)
             if tol.shape[0] == 1 and self_.n > 1:
                 tol = full(self_.n, tol[0], dtype=self_.precision)
             elif tol.shape[0] != self_.n:
                 raise ValueError("tol must have shape (n,).")
         return tol
     ```
   - Edge cases: 
     - Scalar input → broadcast to (n,) array
     - Single-element array with n > 1 → broadcast to (n,)
     - Full array (n,) → dtype conversion only
     - Wrong size array → ValueError
   - Integration: Import numpy.isscalar, numpy.full, numpy.asarray (already imported in _utils.py); add ArrayLike import from numpy.typing

2. **Update adaptive_step_controller.py to import tol_converter**
   - File: src/cubie/integrators/step_control/adaptive_step_controller.py
   - Action: Modify
   - Details:
     - Remove local tol_converter function definition (lines 22-55)
     - Add `tol_converter` to the import from `cubie._utils` (line 11)
     - Updated import line:
       ```python
       from cubie._utils import (
           PrecisionDType,
           clamp_factory,
           float_array_validator,
           getype_validator,
           inrangetype_validator,
           tol_converter,
       )
       ```
   - Edge cases: None
   - Integration: The AdaptiveStepControlConfig class already uses `Converter(tol_converter, takes_self=True)` so no other changes needed

3. **Update linear_solver.py to import tol_converter**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     - Remove local tol_converter function definition (lines 33-64)
     - Add `tol_converter` to the import from `cubie._utils` (line 16)
     - Updated import line:
       ```python
       from cubie._utils import (
           PrecisionDType,
           build_config,
           float_array_validator,
           getype_validator,
           gttype_validator,
           inrangetype_validator,
           is_device_validator,
           precision_converter,
           precision_validator,
           tol_converter,
       )
       ```
   - Edge cases: None
   - Integration: LinearSolverConfig already uses `Converter(tol_converter, takes_self=True)`

4. **Update newton_krylov.py to import tol_converter**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     - Remove local tol_converter function definition (lines 35-66)
     - Add `tol_converter` to the import from `cubie._utils` (line 16)
     - Updated import line:
       ```python
       from cubie._utils import (
           PrecisionDType,
           build_config,
           float_array_validator,
           getype_validator,
           gttype_validator,
           inrangetype_validator,
           is_device_validator,
           precision_converter,
           precision_validator,
           tol_converter,
       )
       ```
   - Edge cases: None
   - Integration: NewtonKrylovConfig already uses `Converter(tol_converter, takes_self=True)`

**Tests to Create**:
- Test file: tests/test_utils.py
- Test function: test_tol_converter_scalar_to_array
- Description: Verify scalar input is broadcast to array of shape (n,)
- Test function: test_tol_converter_single_element_broadcast
- Description: Verify single-element array is broadcast when n > 1
- Test function: test_tol_converter_full_array_passthrough
- Description: Verify full array (n,) passes through with dtype conversion
- Test function: test_tol_converter_wrong_size_raises
- Description: Verify ValueError raised for wrong size array

**Tests to Run**:
- tests/test_utils.py::test_tol_converter_scalar_to_array
- tests/test_utils.py::test_tol_converter_single_element_broadcast
- tests/test_utils.py::test_tol_converter_full_array_passthrough
- tests/test_utils.py::test_tol_converter_wrong_size_raises
- tests/integrators/step_control/test_adaptive_step_controller.py
- tests/integrators/matrix_free_solvers/test_linear_solver.py
- tests/integrators/matrix_free_solvers/test_newton_krylov.py

**Outcomes**: 
- Files Modified: 
  * src/cubie/_utils.py (43 lines added - imports and tol_converter function)
  * src/cubie/integrators/step_control/adaptive_step_controller.py (36 lines removed - local tol_converter, updated imports)
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (33 lines removed - local tol_converter, updated imports)
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (33 lines removed - local tol_converter, updated imports)
  * tests/test_utils.py (55 lines added - tests for tol_converter)
- Functions/Methods Added/Modified:
  * tol_converter() added to src/cubie/_utils.py
  * Removed tol_converter() from adaptive_step_controller.py
  * Removed tol_converter() from linear_solver.py
  * Removed tol_converter() from newton_krylov.py
- Implementation Summary:
  Extracted tol_converter function to _utils.py for centralized reuse across all files that need it. Updated imports in adaptive_step_controller.py, linear_solver.py, and newton_krylov.py to import tol_converter from cubie._utils. Added ArrayLike import from numpy.typing and added isscalar, full, asarray imports to _utils.py.
- Issues Flagged: None

---

## Task Group 2: Create MatrixFreeSolverConfig Base Class
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/_utils.py (lines 39-84 - PrecisionDType, precision_converter, precision_validator, getype_validator)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 67-182 - LinearSolverConfig class)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 69-215 - NewtonKrylovConfig class)
- File: src/cubie/integrators/step_control/base_step_controller.py (lines 35-101 - BaseStepControllerConfig pattern)
- File: src/cubie/cuda_simsafe.py (from_dtype function)
- File: src/cubie/CUDAFactory.py (entire file - CUDAFactory pattern)

**Input Validation Required**:
- precision: Use precision_converter and precision_validator from _utils
- n: Use getype_validator(int, 1) to ensure positive integer

**Tasks**:
1. **Create base_solver.py with MatrixFreeSolverConfig**
   - File: src/cubie/integrators/matrix_free_solvers/base_solver.py
   - Action: Create
   - Details:
     ```python
     """Base configuration for matrix-free solver factories.
     
     This module provides shared configuration infrastructure for the
     Newton and Krylov solvers in :mod:`cubie.integrators.matrix_free_solvers`.
     """
     
     from numpy import dtype as np_dtype
     from numba import from_dtype
     from attrs import define, field
     
     from cubie._utils import (
         PrecisionDType,
         getype_validator,
         precision_converter,
         precision_validator,
     )
     from cubie.cuda_simsafe import from_dtype as simsafe_dtype
     
     
     @define
     class MatrixFreeSolverConfig:
         """Base configuration for matrix-free solver factories.
     
         Provides common attributes shared by LinearSolverConfig and
         NewtonKrylovConfig including precision, vector size, and
         Numba/CUDA type accessors.
     
         Attributes
         ----------
         precision : PrecisionDType
             Numerical precision for computations.
         n : int
             Size of state vectors (must be >= 1).
         """
     
         precision: PrecisionDType = field(
             converter=precision_converter,
             validator=precision_validator
         )
         n: int = field(validator=getype_validator(int, 1))
     
         @property
         def numba_precision(self) -> type:
             """Return Numba type for precision."""
             return from_dtype(np_dtype(self.precision))
     
         @property
         def simsafe_precision(self) -> type:
             """Return CUDA-sim-safe type for precision."""
             return simsafe_dtype(np_dtype(self.precision))
     ```
   - Edge cases: None
   - Integration: Will be imported by LinearSolverConfig and NewtonKrylovConfig

2. **Update LinearSolverConfig to inherit from MatrixFreeSolverConfig**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     - Add import: `from cubie.integrators.matrix_free_solvers.base_solver import MatrixFreeSolverConfig`
     - Remove imports that are now in base: `from_dtype` from numba (keep for local use)
     - Change class definition:
       ```python
       @define
       class LinearSolverConfig(MatrixFreeSolverConfig):
       ```
     - Remove duplicate fields from LinearSolverConfig:
       - Remove `precision` field (inherited)
       - Remove `n` field (inherited)
     - Remove duplicate properties:
       - Remove `numba_precision` property (inherited)
       - Remove `simsafe_precision` property (inherited)
   - Edge cases: Ensure dtype imports remain for other usages in build()
   - Integration: Existing code using LinearSolverConfig unchanged

3. **Update NewtonKrylovConfig to inherit from MatrixFreeSolverConfig**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     - Add import: `from cubie.integrators.matrix_free_solvers.base_solver import MatrixFreeSolverConfig`
     - Change class definition:
       ```python
       @define
       class NewtonKrylovConfig(MatrixFreeSolverConfig):
       ```
     - Remove duplicate fields from NewtonKrylovConfig:
       - Remove `precision` field (inherited)
       - Remove `n` field (inherited)
     - Remove duplicate properties:
       - Remove `numba_precision` property (inherited)
       - Remove `simsafe_precision` property (inherited)
   - Edge cases: Ensure dtype imports remain for other usages in build()
   - Integration: Existing code using NewtonKrylovConfig unchanged

4. **Update matrix_free_solvers __init__.py to export base class**
   - File: src/cubie/integrators/matrix_free_solvers/__init__.py
   - Action: Modify
   - Details:
     - Add import: `from .base_solver import MatrixFreeSolverConfig`
     - Add to __all__: `"MatrixFreeSolverConfig"`
   - Edge cases: None
   - Integration: Makes base class available for external use

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_base_solver.py
- Test function: test_matrix_free_solver_config_precision_property
- Description: Verify numba_precision and simsafe_precision properties work correctly
- Test function: test_matrix_free_solver_config_validation
- Description: Verify precision and n validators work correctly

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_base_solver.py::test_matrix_free_solver_config_precision_property
- tests/integrators/matrix_free_solvers/test_base_solver.py::test_matrix_free_solver_config_validation
- tests/integrators/matrix_free_solvers/test_linear_solver.py
- tests/integrators/matrix_free_solvers/test_newton_krylov.py

**Outcomes**:
- Files Modified: 
  * src/cubie/integrators/matrix_free_solvers/base_solver.py (49 lines created - new MatrixFreeSolverConfig base class)
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (20 lines changed - updated imports, inherited from base, removed duplicate fields/properties)
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (20 lines changed - updated imports, inherited from base, removed duplicate fields/properties)
  * src/cubie/integrators/matrix_free_solvers/__init__.py (3 lines changed - added import and export for MatrixFreeSolverConfig)
  * tests/integrators/matrix_free_solvers/test_base_solver.py (38 lines created - tests for base config)
- Functions/Methods Added/Modified:
  * MatrixFreeSolverConfig class added to base_solver.py with precision, n fields and numba_precision, simsafe_precision properties
  * LinearSolverConfig now inherits from MatrixFreeSolverConfig
  * NewtonKrylovConfig now inherits from MatrixFreeSolverConfig
- Implementation Summary:
  Created MatrixFreeSolverConfig base class with precision/n fields and Numba type accessors. Updated LinearSolverConfig and NewtonKrylovConfig to inherit from the base class, removing duplicate field definitions and property implementations. Exported the base class from the matrix_free_solvers package __init__.py. Created tests to verify precision properties and validation work correctly.
- Issues Flagged: None

---

## Task Group 3: Create ScaledNorm CUDAFactory
**Status**: [x]
**Dependencies**: Task Group 1, Task Group 2

**Required Context**:
- File: src/cubie/CUDAFactory.py (entire file - CUDAFactory, CUDAFunctionCache pattern)
- File: src/cubie/_utils.py (tol_converter, precision utilities, validators)
- File: src/cubie/integrators/step_control/adaptive_PID_controller.py (lines 230-243 - scaled norm pattern)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 524-538 - scaled norm in linear solver)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 438-450 - scaled norm in newton)
- File: src/cubie/integrators/step_control/base_step_controller.py (lines 35-101 - CUDAFactory pattern reference)
- File: src/cubie/cuda_simsafe.py (compile_kwargs, from_dtype)

**Input Validation Required**:
- precision: Use precision_converter and precision_validator
- n: Use getype_validator(int, 1)
- atol: Use float_array_validator, Converter(tol_converter, takes_self=True)
- rtol: Use float_array_validator, Converter(tol_converter, takes_self=True)

**Tasks**:
1. **Create norms.py with ScaledNorm CUDAFactory**
   - File: src/cubie/integrators/norms.py
   - Action: Create
   - Details:
     ```python
     """Norm computation factories for tolerance-scaled convergence checks.
     
     This module provides CUDAFactory subclasses that compile device functions
     for computing scaled norms used in convergence testing across integrators
     and matrix-free solvers.
     """
     
     from typing import Callable
     
     from numpy import asarray, dtype as np_dtype, ndarray
     from numba import cuda, from_dtype
     from attrs import Converter, define, field
     
     from cubie._utils import (
         PrecisionDType,
         build_config,
         float_array_validator,
         getype_validator,
         is_device_validator,
         precision_converter,
         precision_validator,
         tol_converter,
     )
     from cubie.CUDAFactory import CUDAFactory, CUDAFunctionCache
     from cubie.cuda_simsafe import compile_kwargs
     
     
     @define
     class ScaledNormConfig:
         """Configuration for ScaledNorm factory compilation.
     
         Attributes
         ----------
         precision : PrecisionDType
             Numerical precision for computations.
         n : int
             Size of vectors to compute norm over.
         atol : ndarray
             Absolute tolerance array of shape (n,).
         rtol : ndarray
             Relative tolerance array of shape (n,).
         """
     
         precision: PrecisionDType = field(
             converter=precision_converter,
             validator=precision_validator
         )
         n: int = field(validator=getype_validator(int, 1))
         atol: ndarray = field(
             default=asarray([1e-6]),
             validator=float_array_validator,
             converter=Converter(tol_converter, takes_self=True)
         )
         rtol: ndarray = field(
             default=asarray([1e-6]),
             validator=float_array_validator,
             converter=Converter(tol_converter, takes_self=True)
         )
     
         @property
         def numba_precision(self) -> type:
             """Return Numba type for precision."""
             return from_dtype(np_dtype(self.precision))
     
         @property
         def inv_n(self) -> float:
             """Return precomputed 1/n in configured precision."""
             return self.precision(1.0 / self.n)
     
         @property
         def tol_floor(self) -> float:
             """Return minimum tolerance floor to avoid division by zero."""
             return self.precision(1e-16)
     
     
     @define
     class ScaledNormCache(CUDAFunctionCache):
         """Cache container for ScaledNorm outputs.
     
         Attributes
         ----------
         scaled_norm : Callable
             Compiled CUDA device function computing scaled norm squared.
         """
     
         scaled_norm: Callable = field(validator=is_device_validator)
     
     
     class ScaledNorm(CUDAFactory):
         """Factory for scaled norm device functions.
     
         Compiles a CUDA device function that computes the mean squared
         scaled error norm, where each element's contribution is weighted
         by a tolerance computed from absolute and relative tolerance
         arrays.
     
         The returned norm value is the mean of squared ratios:
             sum((|values[i]| / tol_i)^2) / n
         where tol_i = max(atol[i] + rtol[i] * |reference[i]|, floor).
     
         Convergence is achieved when the norm <= 1.0.
         """
     
         def __init__(
             self,
             precision: PrecisionDType,
             n: int,
             **kwargs,
         ) -> None:
             """Initialize ScaledNorm factory.
     
             Parameters
             ----------
             precision : PrecisionDType
                 Numerical precision for computations.
             n : int
                 Size of vectors to compute norm over.
             **kwargs
                 Optional parameters passed to ScaledNormConfig including
                 atol and rtol. None values are ignored.
             """
             super().__init__()
     
             config = build_config(
                 ScaledNormConfig,
                 required={
                     'precision': precision,
                     'n': n,
                 },
                 **kwargs
             )
     
             self.setup_compile_settings(config)
     
         def build(self) -> ScaledNormCache:
             """Compile scaled norm device function.
     
             Returns
             -------
             ScaledNormCache
                 Container with compiled scaled_norm device function.
             """
             config = self.compile_settings
     
             n = config.n
             atol = config.atol
             rtol = config.rtol
             numba_precision = config.numba_precision
             inv_n = config.inv_n
             tol_floor = config.tol_floor
     
             typed_zero = numba_precision(0.0)
             n_val = n
     
             # no cover: start
             @cuda.jit(
                 device=True,
                 inline=True,
                 **compile_kwargs,
             )
             def scaled_norm(values, reference):
                 """Compute mean squared scaled error norm.
     
                 Parameters
                 ----------
                 values : array
                     Error or residual values to measure.
                 reference : array
                     Reference state for relative tolerance scaling.
     
                 Returns
                 -------
                 float
                     Mean squared scaled norm. Converged when <= 1.0.
                 """
                 nrm2 = typed_zero
                 for i in range(n_val):
                     value_i = values[i]
                     ref_i = reference[i]
                     abs_ref = ref_i if ref_i >= typed_zero else -ref_i
                     tol_i = atol[i] + rtol[i] * abs_ref
                     tol_i = tol_i if tol_i > tol_floor else tol_floor
                     abs_val = value_i if value_i >= typed_zero else -value_i
                     ratio = abs_val / tol_i
                     nrm2 += ratio * ratio
                 return nrm2 * inv_n
     
             # no cover: end
             return ScaledNormCache(scaled_norm=scaled_norm)
     
         def update(
             self,
             updates_dict=None,
             silent=False,
             **kwargs
         ):
             """Update compile settings and invalidate cache if changed.
     
             Parameters
             ----------
             updates_dict : dict, optional
                 Dictionary of settings to update.
             silent : bool, default False
                 If True, suppress warnings about unrecognized keys.
             **kwargs
                 Additional settings as keyword arguments.
     
             Returns
             -------
             set
                 Set of recognized parameter names that were updated.
             """
             all_updates = {}
             if updates_dict:
                 all_updates.update(updates_dict)
             all_updates.update(kwargs)
     
             if not all_updates:
                 return set()
     
             return self.update_compile_settings(
                 updates_dict=all_updates, silent=silent
             )
     
         @property
         def device_function(self) -> Callable:
             """Return cached scaled norm device function."""
             return self.get_cached_output("scaled_norm")
     
         @property
         def precision(self) -> PrecisionDType:
             """Return configured precision."""
             return self.compile_settings.precision
     
         @property
         def n(self) -> int:
             """Return vector size."""
             return self.compile_settings.n
     
         @property
         def atol(self) -> ndarray:
             """Return absolute tolerance array."""
             return self.compile_settings.atol
     
         @property
         def rtol(self) -> ndarray:
             """Return relative tolerance array."""
             return self.compile_settings.rtol
     ```
   - Edge cases:
     - Zero reference values: tol_i = atol[i] (rtol term is zero)
     - Tolerance floor: prevents division by zero when both atol and rtol*ref are tiny
   - Integration: Will be used by LinearSolver and NewtonKrylov in future task groups

**Tests to Create**:
- Test file: tests/integrators/test_norms.py
- Test function: test_scaled_norm_config_default_tolerance
- Description: Verify ScaledNormConfig creates with default tolerances
- Test function: test_scaled_norm_config_custom_tolerance
- Description: Verify ScaledNormConfig accepts custom atol/rtol arrays
- Test function: test_scaled_norm_factory_builds_device_function
- Description: Verify ScaledNorm factory builds a valid device function
- Test function: test_scaled_norm_converged_when_under_tolerance
- Description: Verify norm returns <= 1.0 when errors within tolerance
- Test function: test_scaled_norm_exceeds_when_over_tolerance
- Description: Verify norm returns > 1.0 when errors exceed tolerance
- Test function: test_scaled_norm_update_invalidates_cache
- Description: Verify update() triggers rebuild with new tolerances

**Tests to Run**:
- tests/integrators/test_norms.py::test_scaled_norm_config_default_tolerance
- tests/integrators/test_norms.py::test_scaled_norm_config_custom_tolerance
- tests/integrators/test_norms.py::test_scaled_norm_factory_builds_device_function
- tests/integrators/test_norms.py::test_scaled_norm_converged_when_under_tolerance
- tests/integrators/test_norms.py::test_scaled_norm_exceeds_when_over_tolerance
- tests/integrators/test_norms.py::test_scaled_norm_update_invalidates_cache

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/norms.py (217 lines created - ScaledNormConfig, ScaledNormCache, ScaledNorm classes)
  * tests/integrators/test_norms.py (129 lines created - 6 test functions)
- Functions/Methods Added/Modified:
  * ScaledNormConfig class in norms.py with precision, n, atol, rtol fields and numba_precision, inv_n, tol_floor properties
  * ScaledNormCache class in norms.py to hold cached scaled_norm device function
  * ScaledNorm CUDAFactory class with __init__, build, update methods and device_function, precision, n, atol, rtol properties
- Implementation Summary:
  Created ScaledNorm CUDAFactory following the established pattern in CUDAFactory.py and base_step_controller.py. The factory builds a CUDA device function that computes mean squared scaled error norm for convergence checking. The device function uses predicated operations for abs value calculations and applies tolerance scaling per the formula: tol_i = max(atol[i] + rtol[i] * |reference[i]|, floor). Converges when norm <= 1.0.
- Issues Flagged: None

---

## Task Group 4: Integrate ScaledNorm into LinearSolver
**Status**: [x]
**Dependencies**: Task Group 3

**Required Context**:
- File: src/cubie/integrators/norms.py (entire file - ScaledNorm factory)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/base_solver.py (entire file)

**Input Validation Required**:
- None additional; tolerances validated by ScaledNorm factory

**Tasks**:
1. **Add ScaledNorm instance to LinearSolver**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     - Add import: `from cubie.integrators.norms import ScaledNorm`
     - In `__init__` after `self.setup_compile_settings(config)`:
       ```python
       # Create norm factory for convergence checks
       self.norm = ScaledNorm(
           precision=precision,
           n=n,
           atol=config.krylov_atol,
           rtol=config.krylov_rtol,
       )
       ```
     - The norm factory owns krylov_atol and krylov_rtol
   - Edge cases: None
   - Integration: LinearSolverConfig still has krylov_atol/krylov_rtol for backwards compatibility

2. **Update LinearSolver.build() to use ScaledNorm device function**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     - In `build()` method, after extracting config parameters:
       ```python
       # Get scaled norm device function from norm factory
       scaled_norm_fn = self.norm.device_function
       ```
     - Replace inline norm computation loops with calls to `scaled_norm_fn(rhs, x)`
     - In initial residual computation (both cached and non-cached variants):
       - Before: loop computing acc with scaled tolerance
       - After: `acc = scaled_norm_fn(rhs, x)`
     - In iteration loop residual computation:
       - Before: loop computing acc with scaled tolerance
       - After: `acc = scaled_norm_fn(rhs, x)`
     - Remove local tolerance arrays and constants:
       - Remove: `krylov_atol = config.krylov_atol`
       - Remove: `krylov_rtol = config.krylov_rtol`
       - Remove: `inv_n = precision_numba(1.0 / n)`
       - Remove: `tol_floor = precision_numba(1e-16)`
   - Edge cases: Maintain exact same convergence behavior
   - Integration: Device function signature unchanged

3. **Update LinearSolver.update() to propagate to norm factory**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     - In `update()` method, propagate tolerance updates to norm:
       ```python
       # Propagate tolerance updates to norm factory
       norm_updates = {}
       if 'krylov_atol' in all_updates:
           norm_updates['atol'] = all_updates['krylov_atol']
       if 'krylov_rtol' in all_updates:
           norm_updates['rtol'] = all_updates['krylov_rtol']
       if norm_updates:
           self.norm.update(norm_updates, silent=True)
           # Invalidate our cache since norm changed
           self._invalidate_cache()
       ```
   - Edge cases: None
   - Integration: krylov_atol/krylov_rtol updates flow to norm factory

4. **Update LinearSolver tolerance properties to delegate to norm**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     - Modify `krylov_atol` property:
       ```python
       @property
       def krylov_atol(self) -> ndarray:
           """Return absolute tolerance array."""
           return self.norm.atol
       ```
     - Modify `krylov_rtol` property:
       ```python
       @property
       def krylov_rtol(self) -> ndarray:
           """Return relative tolerance array."""
           return self.norm.rtol
       ```
   - Edge cases: None
   - Integration: External API unchanged

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_linear_solver.py
- Test function: test_linear_solver_uses_scaled_norm
- Description: Verify LinearSolver uses ScaledNorm for convergence checking
- Test function: test_linear_solver_tolerance_update_propagates
- Description: Verify krylov_atol/krylov_rtol updates reach norm factory

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_linear_solver.py
- tests/integrators/test_norms.py

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (60 lines changed - import, __init__, build, update, properties)
  * tests/integrators/matrix_free_solvers/test_linear_solver.py (44 lines added - 2 new tests)
- Functions/Methods Added/Modified:
  * LinearSolver.__init__() - added self.norm = ScaledNorm(...) creation
  * LinearSolver.build() - replaced inline norm loops with scaled_norm_fn calls, removed local tolerance variables
  * LinearSolver.update() - added propagation of krylov_atol/krylov_rtol updates to norm factory
  * LinearSolver.krylov_atol property - now delegates to self.norm.atol
  * LinearSolver.krylov_rtol property - now delegates to self.norm.rtol
- Implementation Summary:
  Integrated ScaledNorm factory into LinearSolver. The norm factory is created in __init__ with the tolerance values from config. In build(), the inline tolerance scaling loops are replaced with calls to the scaled_norm_fn device function from the norm factory. The update() method propagates krylov_atol/krylov_rtol changes to the norm factory and invalidates the cache. The krylov_atol and krylov_rtol properties now delegate to the norm factory instead of compile_settings.
- Issues Flagged: None

---

## Task Group 5: Integrate ScaledNorm into NewtonKrylov
**Status**: [x]
**Dependencies**: Task Group 4

**Required Context**:
- File: src/cubie/integrators/norms.py (entire file - ScaledNorm factory)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (entire file - updated version)

**Input Validation Required**:
- None additional; tolerances validated by ScaledNorm factory

**Tasks**:
1. **Add ScaledNorm instance to NewtonKrylov**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     - Add import: `from cubie.integrators.norms import ScaledNorm`
     - In `__init__` after `self.setup_compile_settings(config)`:
       ```python
       # Create norm factory for Newton convergence checks
       self.norm = ScaledNorm(
           precision=precision,
           n=n,
           atol=config.newton_atol,
           rtol=config.newton_rtol,
       )
       ```
     - The norm factory owns newton_atol and newton_rtol
   - Edge cases: None
   - Integration: NewtonKrylovConfig still has newton_atol/newton_rtol for backwards compat

2. **Update NewtonKrylov.build() to use ScaledNorm device function**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     - In `build()` method, after extracting config parameters:
       ```python
       # Get scaled norm device function from norm factory
       scaled_norm_fn = self.norm.device_function
       ```
     - Replace inline norm computation loops with calls to `scaled_norm_fn(residual, stage_increment)`
     - In initial residual norm computation:
       - Before: loop computing norm2_prev with scaled tolerance
       - After: `norm2_prev = scaled_norm_fn(residual, stage_increment)` (note: compute before negating residual)
     - In backtracking norm computation:
       - Before: loop computing norm2_new with scaled tolerance
       - After: `norm2_new = scaled_norm_fn(residual_temp, stage_increment)`
     - Remove local tolerance arrays and constants:
       - Remove: `newton_atol = config.newton_atol`
       - Remove: `newton_rtol = config.newton_rtol`
       - Remove: `inv_n = numba_precision(1.0 / n)`
       - Remove: `tol_floor = numba_precision(1e-16)`
   - Edge cases: 
     - Initial residual must be computed BEFORE negation for norm
     - The loop that negates residual[i] and zeros delta[i] runs separately from norm
   - Integration: Device function signature unchanged

3. **Update NewtonKrylov.update() to propagate to norm factory**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     - In `update()` method, propagate tolerance updates to norm:
       ```python
       # Propagate tolerance updates to norm factory
       norm_updates = {}
       if 'newton_atol' in all_updates:
           norm_updates['atol'] = all_updates['newton_atol']
       if 'newton_rtol' in all_updates:
           norm_updates['rtol'] = all_updates['newton_rtol']
       if norm_updates:
           self.norm.update(norm_updates, silent=True)
           # Invalidate our cache since norm changed
           self._invalidate_cache()
       ```
   - Edge cases: None
   - Integration: newton_atol/newton_rtol updates flow to norm factory

4. **Update NewtonKrylov tolerance properties to delegate to norm**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     - Modify `newton_atol` property:
       ```python
       @property
       def newton_atol(self) -> ndarray:
           """Return absolute tolerance array."""
           return self.norm.atol
       ```
     - Modify `newton_rtol` property:
       ```python
       @property
       def newton_rtol(self) -> ndarray:
           """Return relative tolerance array."""
           return self.norm.rtol
       ```
   - Edge cases: None
   - Integration: External API unchanged

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_newton_krylov.py
- Test function: test_newton_krylov_uses_scaled_norm
- Description: Verify NewtonKrylov uses ScaledNorm for convergence checking
- Test function: test_newton_krylov_tolerance_update_propagates
- Description: Verify newton_atol/newton_rtol updates reach norm factory

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_newton_krylov.py
- tests/integrators/matrix_free_solvers/test_linear_solver.py
- tests/integrators/test_norms.py

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (45 lines changed - import, __init__, build, update, properties)
  * tests/integrators/matrix_free_solvers/test_newton_krylov.py (52 lines added - 2 new tests)
- Functions/Methods Added/Modified:
  * NewtonKrylov.__init__() - added self.norm = ScaledNorm(...) creation
  * NewtonKrylov.build() - replaced inline norm loops with scaled_norm_fn calls, removed local tolerance variables
  * NewtonKrylov.update() - added propagation of newton_atol/newton_rtol updates to norm factory
  * NewtonKrylov.newton_atol property - now delegates to self.norm.atol
  * NewtonKrylov.newton_rtol property - now delegates to self.norm.rtol
- Implementation Summary:
  Integrated ScaledNorm factory into NewtonKrylov. The norm factory is created in __init__ with the tolerance values from config. In build(), the inline tolerance scaling loops are replaced with calls to the scaled_norm_fn device function from the norm factory. The update() method propagates newton_atol/newton_rtol changes to the norm factory and invalidates the cache. The newton_atol and newton_rtol properties now delegate to the norm factory instead of compile_settings. Critical: the initial residual norm is computed BEFORE negating the residual array.
- Issues Flagged: None

---

## Task Group 6: Update Instrumented Test Files
**Status**: [ ]
**Dependencies**: Task Group 4, Task Group 5

**Required Context**:
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (entire file - updated version)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (entire file - updated version)
- File: src/cubie/integrators/norms.py (entire file)

**Input Validation Required**:
- None; instrumented versions mirror production code

**Tasks**:
1. **Update InstrumentedLinearSolver to use ScaledNorm**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify
   - Details:
     - Add import: `from cubie.integrators.norms import ScaledNorm`
     - InstrumentedLinearSolver inherits from LinearSolver, so it gets the norm factory automatically
     - In `build()` method, get the scaled_norm_fn from self.norm.device_function
     - Replace inline norm computations with calls to scaled_norm_fn
     - Remove local tolerance variables:
       - Remove: `krylov_atol = config.krylov_atol`
       - Remove: `krylov_rtol = config.krylov_rtol`
       - Remove: `inv_n = precision_numba(1.0 / n)`
       - Remove: `tol_floor = precision_numba(1e-16)`
     - Replace norm computation loops in both cached and non-cached variants:
       - Before: manual loop computing acc
       - After: `acc = scaled_norm_fn(rhs, x)`
   - Edge cases: Logging behavior must remain identical
   - Integration: Instrumented version mirrors production behavior

2. **Update InstrumentedNewtonKrylov to use ScaledNorm**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify
   - Details:
     - InstrumentedNewtonKrylov inherits from NewtonKrylov, so it gets the norm factory automatically
     - In `build()` method, get the scaled_norm_fn from self.norm.device_function
     - Replace inline norm computations with calls to scaled_norm_fn
     - Remove local tolerance variables:
       - Remove: `newton_atol = config.newton_atol`
       - Remove: `newton_rtol = config.newton_rtol`
       - Remove: `inv_n = numba_precision(1.0 / n)`
       - Remove: `tol_floor = numba_precision(1e-16)`
     - Replace norm computation loops:
       - Initial residual norm: `norm2_prev = scaled_norm_fn(residual_copy, stage_increment)` (use copy before negation)
       - Backtracking norm: `norm2_new = scaled_norm_fn(residual_temp, stage_increment)`
   - Edge cases: Logging behavior must remain identical
   - Integration: Instrumented version mirrors production behavior

**Tests to Create**:
- None; existing instrumented tests verify behavior

**Tests to Run**:
- tests/integrators/algorithms/instrumented/
- tests/integrators/matrix_free_solvers/test_linear_solver.py
- tests/integrators/matrix_free_solvers/test_newton_krylov.py

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

### Total Task Groups: 6

### Dependency Chain:
```
Task Group 1 (tol_converter extraction)
    ↓
Task Group 2 (MatrixFreeSolverConfig base class)
    ↓
Task Group 3 (ScaledNorm CUDAFactory)
    ↓
Task Group 4 (LinearSolver integration)
    ↓
Task Group 5 (NewtonKrylov integration)
    ↓
Task Group 6 (Instrumented test files)
```

### Files Created:
- src/cubie/integrators/matrix_free_solvers/base_solver.py
- src/cubie/integrators/norms.py
- tests/test_utils.py (test additions)
- tests/integrators/matrix_free_solvers/test_base_solver.py
- tests/integrators/test_norms.py

### Files Modified:
- src/cubie/_utils.py
- src/cubie/integrators/step_control/adaptive_step_controller.py
- src/cubie/integrators/matrix_free_solvers/linear_solver.py
- src/cubie/integrators/matrix_free_solvers/newton_krylov.py
- src/cubie/integrators/matrix_free_solvers/__init__.py
- tests/integrators/algorithms/instrumented/matrix_free_solvers.py

### Estimated Complexity: Medium-High
- Task Groups 1-2: Low complexity (refactoring, imports)
- Task Group 3: Medium complexity (new CUDAFactory class)
- Task Groups 4-5: Medium-High complexity (device function refactoring)
- Task Group 6: Medium complexity (synchronizing instrumented code)

### Key Risk Areas:
1. **Device function behavior change**: Replacing inline loops with function calls must preserve exact convergence behavior
2. **Residual negation timing in NewtonKrylov**: Must compute norm BEFORE negating residual array
3. **Cache invalidation chain**: Norm factory changes must propagate to parent solver cache
