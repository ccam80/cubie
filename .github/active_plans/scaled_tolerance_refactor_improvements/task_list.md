# Implementation Task List
# Feature: Scaled Tolerance Refactoring Improvements
# Plan Reference: .github/active_plans/scaled_tolerance_refactor_improvements/agent_plan.md

## Task Group 1: Enhance MatrixFreeSolverConfig Base Class
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/base_solver.py (entire file, lines 1-51)
- File: src/cubie/_utils.py (lines 489-532 for validators)
- File: src/cubie/CUDAFactory.py (lines 37-88 for CUDAFactory pattern)

**Input Validation Required**:
- max_iters: Use `inrangetype_validator(int, 1, 32767)` - values must be integers in range [1, 32767]
- norm_device_function: Optional[Callable] with `eq=False` - no validation beyond type, eq=False prevents comparison

**Tasks**:
1. **Add max_iters field to MatrixFreeSolverConfig**
   - File: src/cubie/integrators/matrix_free_solvers/base_solver.py
   - Action: Modify
   - Details:
     ```python
     # Add import for inrangetype_validator
     from cubie._utils import (
         PrecisionDType,
         getype_validator,
         inrangetype_validator,
         precision_converter,
         precision_validator,
     )
     
     # Add field to MatrixFreeSolverConfig after 'n' field:
     max_iters: int = field(
         default=100,
         validator=inrangetype_validator(int, 1, 32767)
     )
     ```
   - Edge cases: Value must be >= 1 and <= 32767 (fits in int16 for CUDA)
   - Integration: Subclasses will inherit this field instead of defining separate max_linear_iters/max_newton_iters

2. **Add norm_device_function field to MatrixFreeSolverConfig**
   - File: src/cubie/integrators/matrix_free_solvers/base_solver.py
   - Action: Modify
   - Details:
     ```python
     # Add import for Callable and Optional
     from typing import Callable, Optional
     
     # Add field after max_iters:
     norm_device_function: Optional[Callable] = field(
         default=None,
         eq=False  # Prevents comparison of functions, cache invalidation via identity
     )
     ```
   - Edge cases: Can be None initially, gets populated when norm factory builds
   - Integration: Changes to this field trigger cache invalidation through CUDAFactory pattern

3. **Update docstring for MatrixFreeSolverConfig**
   - File: src/cubie/integrators/matrix_free_solvers/base_solver.py
   - Action: Modify
   - Details:
     ```python
     """Base configuration for matrix-free solver factories.

     Provides common attributes shared by LinearSolverConfig and
     NewtonKrylovConfig including precision, vector size, iteration
     limits, and Numba/CUDA type accessors.

     Attributes
     ----------
     precision : PrecisionDType
         Numerical precision for computations.
     n : int
         Size of state vectors (must be >= 1).
     max_iters : int
         Maximum solver iterations permitted (1 to 32767).
     norm_device_function : Optional[Callable]
         Compiled norm function for convergence checks. Updated when
         norm factory rebuilds; changes invalidate solver cache.
     """
     ```
   - Edge cases: None
   - Integration: Documentation for new fields

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_base_solver.py
- Test function: test_matrix_free_solver_config_max_iters_default
- Description: Verify max_iters defaults to 100
- Test function: test_matrix_free_solver_config_max_iters_validation
- Description: Verify max_iters rejects values < 1 and > 32767
- Test function: test_matrix_free_solver_config_norm_device_function_field
- Description: Verify norm_device_function field exists and accepts None or Callable

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_base_solver.py::test_matrix_free_solver_config_max_iters_default
- tests/integrators/matrix_free_solvers/test_base_solver.py::test_matrix_free_solver_config_max_iters_validation
- tests/integrators/matrix_free_solvers/test_base_solver.py::test_matrix_free_solver_config_norm_device_function_field

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/base_solver.py (16 lines changed)
  * tests/integrators/matrix_free_solvers/test_base_solver.py (56 lines added)
- Functions/Methods Added/Modified:
  * MatrixFreeSolverConfig class: added max_iters and norm_device_function fields
- Implementation Summary:
  Added two new fields to MatrixFreeSolverConfig: max_iters with default=100 and range validation [1, 32767], and norm_device_function as Optional[Callable] with eq=False. Updated imports to include inrangetype_validator from cubie._utils and Callable/Optional from typing. Updated docstring to document new attributes.
- Issues Flagged: None

---

## Task Group 2: Create MatrixFreeSolver CUDAFactory Base Class
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/base_solver.py (entire file after Task Group 1)
- File: src/cubie/CUDAFactory.py (entire file, lines 1-384)
- File: src/cubie/integrators/norms.py (entire file, lines 1-249)
- File: src/cubie/_utils.py (lines 714-792 for build_config)

**Input Validation Required**:
- settings_prefix: String, no validation needed beyond type (class attribute)
- atol/rtol: Delegated to ScaledNorm constructor via kwargs passthrough

**Tasks**:
1. **Create MatrixFreeSolver base class**
   - File: src/cubie/integrators/matrix_free_solvers/base_solver.py
   - Action: Modify (add new class after MatrixFreeSolverConfig)
   - Details:
     ```python
     from typing import Any, Callable, Dict, Optional, Set
     
     from cubie.CUDAFactory import CUDAFactory
     from cubie.integrators.norms import ScaledNorm
     
     
     class MatrixFreeSolver(CUDAFactory):
         """Base factory for matrix-free solver device functions.
         
         Provides shared infrastructure for tolerance parameter mapping
         and norm factory management. Subclasses set `settings_prefix`
         to enable automatic mapping of prefixed parameters (e.g.,
         "krylov_atol" -> "atol" for norm updates).
         
         Attributes
         ----------
         settings_prefix : str
             Prefix for tolerance parameters (e.g., "krylov_" or "newton_").
             Set by subclasses.
         norm : ScaledNorm
             Factory for scaled norm device function used in convergence checks.
         """
         
         settings_prefix: str = ""
         
         def __init__(
             self,
             precision: PrecisionDType,
             n: int,
             atol: Optional[Any] = None,
             rtol: Optional[Any] = None,
         ) -> None:
             """Initialize base solver with norm factory.
             
             Parameters
             ----------
             precision : PrecisionDType
                 Numerical precision for computations.
             n : int
                 Size of state vectors.
             atol : array-like, optional
                 Absolute tolerance for scaled norm.
             rtol : array-like, optional
                 Relative tolerance for scaled norm.
             """
             super().__init__()
             
             # Build norm kwargs, filtering None values
             norm_kwargs = {}
             if atol is not None:
                 norm_kwargs['atol'] = atol
             if rtol is not None:
                 norm_kwargs['rtol'] = rtol
             
             self.norm = ScaledNorm(
                 precision=precision,
                 n=n,
                 **norm_kwargs,
             )
         
         def _extract_prefixed_tolerance(
             self,
             updates: Dict[str, Any],
         ) -> Dict[str, Any]:
             """Extract and map prefixed tolerance parameters.
             
             Looks for `{prefix}atol` and `{prefix}rtol` in updates dict,
             removes them, and returns dict with unprefixed keys for norm.
             
             Parameters
             ----------
             updates : dict
                 Updates dictionary (modified in place).
             
             Returns
             -------
             dict
                 Norm updates with unprefixed tolerance keys.
             """
             prefix = self.settings_prefix
             norm_updates = {}
             
             prefixed_atol = f"{prefix}atol"
             prefixed_rtol = f"{prefix}rtol"
             
             if prefixed_atol in updates:
                 norm_updates['atol'] = updates.pop(prefixed_atol)
             if prefixed_rtol in updates:
                 norm_updates['rtol'] = updates.pop(prefixed_rtol)
             
             return norm_updates
         
         def _update_norm_and_config(
             self,
             norm_updates: Dict[str, Any],
         ) -> None:
             """Update norm factory and propagate device function to config.
             
             Parameters
             ----------
             norm_updates : dict
                 Tolerance updates for norm factory.
             """
             if norm_updates:
                 self.norm.update(norm_updates, silent=True)
             
             # Always update config with current norm device function
             # This triggers cache invalidation if the function changed
             self.update_compile_settings(
                 norm_device_function=self.norm.device_function,
                 silent=True,
             )
     ```
   - Edge cases: 
     - Empty norm_updates should still update config with current device function
     - settings_prefix can be empty string for base class
   - Integration: LinearSolver and NewtonKrylov will inherit from this class

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_base_solver.py
- Test function: test_matrix_free_solver_creates_norm
- Description: Verify MatrixFreeSolver creates ScaledNorm in constructor
- Test function: test_matrix_free_solver_extract_prefixed_tolerance
- Description: Verify _extract_prefixed_tolerance correctly maps prefixed keys
- Test function: test_matrix_free_solver_norm_update_propagates_to_config
- Description: Verify _update_norm_and_config sets norm_device_function in config

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_base_solver.py::test_matrix_free_solver_creates_norm
- tests/integrators/matrix_free_solvers/test_base_solver.py::test_matrix_free_solver_extract_prefixed_tolerance
- tests/integrators/matrix_free_solvers/test_base_solver.py::test_matrix_free_solver_norm_update_propagates_to_config

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/base_solver.py (110 lines changed - added imports and MatrixFreeSolver class)
  * tests/integrators/matrix_free_solvers/test_base_solver.py (85 lines added - 3 new tests)
- Functions/Methods Added/Modified:
  * MatrixFreeSolver class with __init__, _extract_prefixed_tolerance, _update_norm_and_config methods
- Implementation Summary:
  Created MatrixFreeSolver base class inheriting from CUDAFactory with settings_prefix class attribute (default empty string), __init__ that creates self.norm ScaledNorm factory from precision/n/atol/rtol parameters, _extract_prefixed_tolerance method that extracts and maps "{prefix}atol" -> "atol" keys from updates dict, and _update_norm_and_config method that updates norm factory and propagates device function to compile settings. Added required imports (Any, Dict, Set from typing, CUDAFactory, ScaledNorm).
- Issues Flagged: None

---

## Task Group 3: Refactor LinearSolverConfig - Remove Tolerance Fields
**Status**: [x]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 1-150)
- File: src/cubie/integrators/matrix_free_solvers/base_solver.py (after Task Groups 1-2)

**Input Validation Required**:
- No new validation - removing fields that are now validated by norm factory

**Tasks**:
1. **Remove krylov_atol and krylov_rtol fields from LinearSolverConfig**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     # Remove these field definitions from LinearSolverConfig:
     # krylov_atol: ndarray = field(...)
     # krylov_rtol: ndarray = field(...)
     
     # Keep: _krylov_tolerance for legacy scalar tolerance
     # Keep: max_linear_iters for now (will be aliased to max_iters)
     ```
   - Edge cases: settings_dict property needs to be updated to not reference removed fields
   - Integration: Tolerances now sourced from solver.norm factory

2. **Update LinearSolverConfig.settings_dict property**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     @property
     def settings_dict(self) -> Dict[str, Any]:
         """Return linear solver configuration as dictionary.
         
         Returns
         -------
         dict
             Configuration dictionary. Note: krylov_atol and krylov_rtol
             are not included here; access them via solver.krylov_atol
             and solver.krylov_rtol properties which delegate to the
             norm factory.
         """
         return {
             'krylov_tolerance': self.krylov_tolerance,
             'max_linear_iters': self.max_linear_iters,
             'linear_correction_type': self.linear_correction_type,
             'preconditioned_vec_location': self.preconditioned_vec_location,
             'temp_location': self.temp_location,
         }
     ```
   - Edge cases: None
   - Integration: Callers expecting krylov_atol/krylov_rtol in settings_dict should use solver properties

3. **Update LinearSolverConfig docstring**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     """Configuration for LinearSolver compilation.

     Attributes
     ----------
     precision : PrecisionDType
         Numerical precision for computations.
     n : int
         Length of residual and search-direction vectors.
     max_iters : int
         Maximum solver iterations permitted.
     norm_device_function : Optional[Callable]
         Compiled norm function for convergence checks.
     operator_apply : Optional[Callable]
         Device function applying operator F @ v.
     preconditioner : Optional[Callable]
         Device function for approximate inverse preconditioner.
     linear_correction_type : str
         Line-search strategy ('steepest_descent' or 'minimal_residual').
     krylov_tolerance : float
         Target on squared residual norm for convergence (legacy scalar).
     max_linear_iters : int
         Maximum iterations permitted (alias for max_iters).
     preconditioned_vec_location : str
         Memory location for preconditioned_vec buffer ('local' or 'shared').
     temp_location : str
         Memory location for temp buffer ('local' or 'shared').
     use_cached_auxiliaries : bool
         Whether to use cached auxiliary arrays (determines signature).
     
     Notes
     -----
     Tolerance arrays (krylov_atol, krylov_rtol) are managed by the solver's
     norm factory and accessed via LinearSolver.krylov_atol/krylov_rtol
     properties.
     """
     ```
   - Edge cases: None
   - Integration: Documentation update

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_linear_solver.py
- Test function: test_linear_solver_config_no_tolerance_fields
- Description: Verify LinearSolverConfig no longer has krylov_atol/krylov_rtol as direct fields
- Test function: test_linear_solver_config_settings_dict_excludes_tolerance_arrays
- Description: Verify settings_dict does not include tolerance arrays

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_config_no_tolerance_fields
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_config_settings_dict_excludes_tolerance_arrays

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (35 lines changed)
  * tests/integrators/matrix_free_solvers/test_linear_solver.py (38 lines added)
- Functions/Methods Added/Modified:
  * LinearSolverConfig class: removed krylov_atol and krylov_rtol fields
  * LinearSolverConfig.settings_dict property: updated to exclude tolerance arrays
  * LinearSolverConfig docstring: updated with Notes section
- Implementation Summary:
  Removed krylov_atol and krylov_rtol field definitions from LinearSolverConfig. Updated settings_dict property to not include tolerance arrays and updated docstring explaining that tolerance arrays are now managed by the solver's norm factory. Cleaned up unused imports (Converter, asarray, Union, ArrayLike, float_array_validator, tol_converter). Added two new tests to verify the fields are removed and settings_dict excludes tolerance arrays.
- Issues Flagged: None

---

## Task Group 4: Refactor LinearSolver - Inherit from MatrixFreeSolver
**Status**: [x]
**Dependencies**: Task Groups 1, 2, 3

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/base_solver.py (after Task Groups 1-2)
- File: src/cubie/integrators/norms.py (lines 89-249 for ScaledNorm class)

**Input Validation Required**:
- Tolerance validation delegated to MatrixFreeSolver base class and ScaledNorm

**Tasks**:
1. **Change LinearSolver inheritance and add settings_prefix**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     # Update import
     from cubie.integrators.matrix_free_solvers.base_solver import (
         MatrixFreeSolverConfig,
         MatrixFreeSolver,
     )
     
     # Change class definition:
     class LinearSolver(MatrixFreeSolver):
         """Factory for linear solver device functions.
         
         Implements steepest-descent or minimal-residual iterations
         for solving linear systems without forming Jacobian matrices.
         """
         
         settings_prefix = "krylov_"
     ```
   - Edge cases: Must ensure MatrixFreeSolver import is available
   - Integration: Inherits norm management from base class

2. **Refactor LinearSolver.__init__ to use base class**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     def __init__(
         self,
         precision: PrecisionDType,
         n: int,
         **kwargs,
     ) -> None:
         """Initialize LinearSolver with parameters.
         
         Parameters
         ----------
         precision : PrecisionDType
             Numerical precision for computations.
         n : int
             Length of residual and search-direction vectors.
         **kwargs
             Optional parameters passed to LinearSolverConfig. See
             LinearSolverConfig for available parameters. Tolerance
             parameters (krylov_atol, krylov_rtol) are passed to the
             norm factory. None values are ignored.
         """
         # Extract tolerance kwargs for base class norm factory
         atol = kwargs.pop('krylov_atol', None)
         rtol = kwargs.pop('krylov_rtol', None)
         
         # Initialize base class with norm factory
         super().__init__(
             precision=precision,
             n=n,
             atol=atol,
             rtol=rtol,
         )
         
         config = build_config(
             LinearSolverConfig,
             required={
                 'precision': precision,
                 'n': n,
             },
             **kwargs
         )
         self.setup_compile_settings(config)
         
         # Update config with norm device function
         self._update_norm_and_config({})
         
         self.register_buffers()
     ```
   - Edge cases: 
     - Need to pop tolerance kwargs before passing to build_config
     - Must call _update_norm_and_config to set initial norm_device_function
   - Integration: Removes direct self.norm creation (now in base class)

3. **Refactor LinearSolver.update to copy dict and use base class**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     def update(
         self,
         updates_dict: Optional[Dict[str, Any]] = None,
         silent: bool = False,
         **kwargs
     ) -> Set[str]:
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
         # Merge updates into a COPY to preserve original dict
         all_updates = {}
         if updates_dict:
             all_updates.update(updates_dict)
         all_updates.update(kwargs)
         
         recognized = set()
         
         if not all_updates:
             return recognized
         
         # Extract prefixed tolerance parameters (modifies all_updates in place)
         norm_updates = self._extract_prefixed_tolerance(all_updates)
         
         # Mark tolerance parameters as recognized
         if norm_updates:
             if 'atol' in norm_updates:
                 recognized.add('krylov_atol')
             if 'rtol' in norm_updates:
                 recognized.add('krylov_rtol')
         
         # Update norm and propagate to config
         self._update_norm_and_config(norm_updates)
         
         # Update compile settings with remaining parameters
         recognized |= self.update_compile_settings(
             updates_dict=all_updates, silent=True
         )
         
         # Buffer locations handled by registry
         buffer_registry.update(self, updates_dict=all_updates, silent=True)
         self.register_buffers()
         
         return recognized
     ```
   - Edge cases: 
     - Original updates_dict must not be modified (copy first)
     - Tolerance parameters must be marked as recognized
   - Integration: Removes manual _invalidate_cache() call

4. **Update LinearSolver.build to use config's norm_device_function**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     # In build() method, change:
     # scaled_norm_fn = self.norm.device_function
     # To:
     scaled_norm_fn = config.norm_device_function
     ```
   - Edge cases: Must ensure norm_device_function is set before build() is called
   - Integration: Gets norm function from config for proper cache invalidation

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_linear_solver.py
- Test function: test_linear_solver_inherits_from_matrix_free_solver
- Description: Verify LinearSolver is instance of MatrixFreeSolver
- Test function: test_linear_solver_update_preserves_original_dict
- Description: Verify update() does not modify the input updates_dict
- Test function: test_linear_solver_no_manual_cache_invalidation
- Description: Verify cache invalidation happens through config update, not manual call

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_inherits_from_matrix_free_solver
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_update_preserves_original_dict
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_no_manual_cache_invalidation
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_uses_scaled_norm
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_tolerance_update_propagates

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (45 lines changed)
  * tests/integrators/matrix_free_solvers/test_linear_solver.py (68 lines added)
- Functions/Methods Added/Modified:
  * LinearSolver class: changed inheritance to MatrixFreeSolver, added settings_prefix
  * LinearSolver.__init__: refactored to use base class norm factory
  * LinearSolver.update: refactored to copy dict and use base class methods
  * LinearSolver.build: uses config.norm_device_function instead of self.norm.device_function
- Implementation Summary:
  Refactored LinearSolver to inherit from MatrixFreeSolver instead of CUDAFactory. Added settings_prefix = "krylov_" class attribute. Updated __init__ to extract krylov_atol/krylov_rtol kwargs, pass them to base class, and call _update_norm_and_config to set initial norm_device_function in config. Updated update() method to create a copy of updates dict, use _extract_prefixed_tolerance to extract tolerance params, and use _update_norm_and_config instead of manual cache invalidation. Updated build() to use config.norm_device_function instead of self.norm.device_function. Removed unused ScaledNorm import and CUDAFactory import (now imported via base class). Added three new tests verifying inheritance, dict preservation, and cache invalidation through config.
- Issues Flagged: None

---

## Task Group 5: Refactor NewtonKrylovConfig - Remove Tolerance Fields
**Status**: [x]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 1-170)
- File: src/cubie/integrators/matrix_free_solvers/base_solver.py (after Task Groups 1-2)

**Input Validation Required**:
- No new validation - removing fields that are now validated by norm factory

**Tasks**:
1. **Remove newton_atol and newton_rtol fields from NewtonKrylovConfig**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # Remove these field definitions from NewtonKrylovConfig:
     # newton_atol: ndarray = field(...)
     # newton_rtol: ndarray = field(...)
     
     # Keep: _newton_tolerance for legacy scalar tolerance
     # Keep: max_newton_iters for now (will be aliased to max_iters)
     ```
   - Edge cases: settings_dict property needs to be updated
   - Integration: Tolerances now sourced from solver.norm factory

2. **Update NewtonKrylovConfig.settings_dict property**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     @property
     def settings_dict(self) -> Dict[str, Any]:
         """Return Newton-Krylov configuration as dictionary.
         
         Returns
         -------
         dict
             Configuration dictionary. Note: newton_atol and newton_rtol
             are not included here; access them via solver.newton_atol
             and solver.newton_rtol properties which delegate to the
             norm factory.
         """
         return {
             'newton_tolerance': self.newton_tolerance,
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
   - Edge cases: None
   - Integration: Callers expecting tolerance arrays should use solver properties

3. **Update NewtonKrylovConfig docstring**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     """Configuration for NewtonKrylov solver compilation.

     Attributes
     ----------
     precision : PrecisionDType
         Numerical precision for computations.
     n : int
         Size of state vectors.
     max_iters : int
         Maximum solver iterations permitted.
     norm_device_function : Optional[Callable]
         Compiled norm function for convergence checks.
     residual_function : Optional[Callable]
         Device function evaluating residuals.
     linear_solver_function : Optional[Callable]
         LinearSolver device function for solving linear systems.
     newton_tolerance : float
         Residual norm threshold for convergence (legacy scalar).
     max_newton_iters : int
         Maximum Newton iterations permitted (alias for max_iters).
     newton_damping : float
         Step shrink factor for backtracking.
     newton_max_backtracks : int
         Maximum damping attempts per Newton step.
     delta_location : str
         Memory location for delta buffer.
     residual_location : str
         Memory location for residual buffer.
     residual_temp_location : str
         Memory location for residual_temp buffer.
     stage_base_bt_location : str
         Memory location for stage_base_bt buffer.
     
     Notes
     -----
     Tolerance arrays (newton_atol, newton_rtol) are managed by the solver's
     norm factory and accessed via NewtonKrylov.newton_atol/newton_rtol
     properties.
     """
     ```
   - Edge cases: None
   - Integration: Documentation update

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_newton_krylov.py
- Test function: test_newton_krylov_config_no_tolerance_fields
- Description: Verify NewtonKrylovConfig no longer has newton_atol/newton_rtol as direct fields
- Test function: test_newton_krylov_config_settings_dict_excludes_tolerance_arrays
- Description: Verify settings_dict does not include tolerance arrays

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_config_no_tolerance_fields
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_config_settings_dict_excludes_tolerance_arrays

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (22 lines changed)
  * tests/integrators/matrix_free_solvers/test_newton_krylov.py (48 lines added)
- Functions/Methods Added/Modified:
  * NewtonKrylovConfig class: removed newton_atol and newton_rtol fields
  * NewtonKrylovConfig.settings_dict property: updated to exclude tolerance arrays
  * NewtonKrylovConfig docstring: updated with Notes section
- Implementation Summary:
  Removed newton_atol and newton_rtol field definitions from NewtonKrylovConfig. Updated settings_dict property to not include tolerance arrays and updated docstring explaining that tolerance arrays are now managed by the solver's norm factory. Cleaned up unused imports (Converter, asarray, Union, ArrayLike, float_array_validator, tol_converter). Added two new tests to verify the fields are removed and settings_dict excludes tolerance arrays.
- Issues Flagged: NewtonKrylov.__init__ references config.newton_atol and config.newton_rtol which will fail until Task Group 6 refactors the class to use the base class. Existing tests using NewtonKrylov will fail until Task Group 6 is complete.

---

## Task Group 6: Refactor NewtonKrylov - Inherit from MatrixFreeSolver
**Status**: [x]
**Dependencies**: Task Groups 1, 2, 4, 5

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/base_solver.py (after Task Groups 1-2)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (after Task Group 4)

**Input Validation Required**:
- Tolerance validation delegated to MatrixFreeSolver base class and ScaledNorm

**Tasks**:
1. **Change NewtonKrylov inheritance and add settings_prefix**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # Update import
     from cubie.integrators.matrix_free_solvers.base_solver import (
         MatrixFreeSolverConfig,
         MatrixFreeSolver,
     )
     
     # Change class definition:
     class NewtonKrylov(MatrixFreeSolver):
         """Factory for Newton-Krylov solver device functions.
         
         Implements damped Newton iteration using a matrix-free
         linear solver for the correction equation.
         """
         
         settings_prefix = "newton_"
     ```
   - Edge cases: Must ensure MatrixFreeSolver import is available
   - Integration: Inherits norm management from base class

2. **Refactor NewtonKrylov.__init__ to use base class**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     def __init__(
         self,
         precision: PrecisionDType,
         n: int,
         linear_solver: LinearSolver,
         **kwargs,
     ) -> None:
         """Initialize NewtonKrylov with parameters.
         
         Parameters
         ----------
         precision : PrecisionDType
             Numerical precision for computations.
         n : int
             Size of state vectors.
         linear_solver : LinearSolver
             LinearSolver instance for solving linear systems.
         **kwargs
             Optional parameters passed to NewtonKrylovConfig. See
             NewtonKrylovConfig for available parameters. Tolerance
             parameters (newton_atol, newton_rtol) are passed to the
             norm factory. None values are ignored.
         """
         # Extract tolerance kwargs for base class norm factory
         atol = kwargs.pop('newton_atol', None)
         rtol = kwargs.pop('newton_rtol', None)
         
         # Initialize base class with norm factory
         super().__init__(
             precision=precision,
             n=n,
             atol=atol,
             rtol=rtol,
         )
         
         self.linear_solver = linear_solver
         
         config = build_config(
             NewtonKrylovConfig,
             required={
                 'precision': precision,
                 'n': n,
             },
             **kwargs
         )
         
         self.setup_compile_settings(config)
         
         # Update config with norm device function
         self._update_norm_and_config({})
         
         self.register_buffers()
     ```
   - Edge cases: 
     - Need to pop tolerance kwargs before passing to build_config
     - linear_solver assignment must happen before setup_compile_settings
   - Integration: Removes direct self.norm creation (now in base class)

3. **Refactor NewtonKrylov.update to copy dict and use base class**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     def update(
         self,
         updates_dict: Optional[Dict[str, Any]] = None,
         silent: bool = False,
         **kwargs
     ) -> Set[str]:
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
         # Merge updates into a COPY to preserve original dict
         all_updates = {}
         if updates_dict:
             all_updates.update(updates_dict)
         all_updates.update(kwargs)
         
         if not all_updates:
             return set()
         
         recognized = set()
         
         # Forward krylov-prefixed params to linear solver
         recognized |= self.linear_solver.update(all_updates, silent=True)
         
         # Extract prefixed tolerance parameters (modifies all_updates in place)
         norm_updates = self._extract_prefixed_tolerance(all_updates)
         
         # Mark tolerance parameters as recognized
         if norm_updates:
             if 'atol' in norm_updates:
                 recognized.add('newton_atol')
             if 'rtol' in norm_updates:
                 recognized.add('newton_rtol')
         
         # Update norm and propagate to config
         self._update_norm_and_config(norm_updates)
         
         # Update device function reference from linear solver
         all_updates['linear_solver_function'] = self.linear_solver.device_function
         
         # Update compile settings with remaining parameters
         recognized |= self.update_compile_settings(
             updates_dict=all_updates, silent=True
         )
         
         # Buffer locations handled by registry
         buffer_registry.update(self, updates_dict=all_updates, silent=True)
         self.register_buffers()
         
         return recognized
     ```
   - Edge cases: 
     - Original updates_dict must not be modified (copy first)
     - Must update linear_solver first to get fresh device_function
     - Tolerance parameters must be marked as recognized
   - Integration: Removes manual _invalidate_cache() call

4. **Update NewtonKrylov.build to use config's norm_device_function**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     # In build() method, change:
     # scaled_norm_fn = self.norm.device_function
     # To:
     scaled_norm_fn = config.norm_device_function
     ```
   - Edge cases: Must ensure norm_device_function is set before build() is called
   - Integration: Gets norm function from config for proper cache invalidation

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_newton_krylov.py
- Test function: test_newton_krylov_inherits_from_matrix_free_solver
- Description: Verify NewtonKrylov is instance of MatrixFreeSolver
- Test function: test_newton_krylov_update_preserves_original_dict
- Description: Verify update() does not modify the input updates_dict
- Test function: test_newton_krylov_no_manual_cache_invalidation
- Description: Verify cache invalidation happens through config update, not manual call

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_inherits_from_matrix_free_solver
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_update_preserves_original_dict
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_no_manual_cache_invalidation
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_uses_scaled_norm
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_tolerance_update_propagates

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (35 lines changed)
  * tests/integrators/matrix_free_solvers/test_newton_krylov.py (75 lines added)
- Functions/Methods Added/Modified:
  * NewtonKrylov class: changed inheritance to MatrixFreeSolver, added settings_prefix
  * NewtonKrylov.__init__: refactored to use base class norm factory
  * NewtonKrylov.update: refactored to copy dict and use base class methods
  * NewtonKrylov.build: uses config.norm_device_function instead of self.norm.device_function
- Implementation Summary:
  Refactored NewtonKrylov to inherit from MatrixFreeSolver instead of CUDAFactory. Added settings_prefix = "newton_" class attribute. Updated __init__ to extract newton_atol/newton_rtol kwargs, pass them to base class, and call _update_norm_and_config to set initial norm_device_function in config. Updated update() method to create a copy of updates dict, use _extract_prefixed_tolerance to extract tolerance params, and use _update_norm_and_config instead of manual cache invalidation. Updated build() to use config.norm_device_function instead of self.norm.device_function. Removed unused ScaledNorm import and CUDAFactory import (now imported via base class). Added three new tests verifying inheritance, dict preservation, and cache invalidation through config.
- Issues Flagged: None

---

## Task Group 7: Update Instrumented Test Files
**Status**: [ ]
**Dependencies**: Task Groups 4, 6

**Required Context**:
- File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (after Task Group 4)
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (after Task Group 6)

**Input Validation Required**:
- None - instrumented files mirror source changes

**Tasks**:
1. **Update InstrumentedLinearSolver.build to use config's norm_device_function**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify
   - Details:
     ```python
     # In InstrumentedLinearSolver.build() method, change:
     # scaled_norm_fn = self.norm.device_function
     # To:
     config = self.compile_settings
     # ... existing code ...
     scaled_norm_fn = config.norm_device_function
     ```
   - Edge cases: None - mirrors source change
   - Integration: Maintains consistency with production code

2. **Update InstrumentedNewtonKrylov.build to use config's norm_device_function**
   - File: tests/integrators/algorithms/instrumented/matrix_free_solvers.py
   - Action: Modify
   - Details:
     ```python
     # In InstrumentedNewtonKrylov.build() method, change:
     # scaled_norm_fn = self.norm.device_function
     # To:
     config = self.compile_settings
     # ... existing code ...
     scaled_norm_fn = config.norm_device_function
     ```
   - Edge cases: None - mirrors source change
   - Integration: Maintains consistency with production code

**Tests to Create**:
- None - instrumented files are tested through their usage in other test modules

**Tests to Run**:
- tests/integrators/algorithms/instrumented/test_instrumented.py

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 8: Update LinearSolver settings_dict Property
**Status**: [ ]
**Dependencies**: Task Groups 3, 4

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (entire file after Task Groups 3, 4)

**Input Validation Required**:
- None

**Tasks**:
1. **Update LinearSolver.settings_dict to include tolerance arrays from norm**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     ```python
     @property
     def settings_dict(self) -> Dict[str, Any]:
         """Return linear solver configuration as dictionary.
         
         Combines config settings with tolerance arrays from norm factory.
         
         Returns
         -------
         dict
             Configuration dictionary including krylov_atol and krylov_rtol
             from the norm factory.
         """
         result = dict(self.compile_settings.settings_dict)
         result['krylov_atol'] = self.krylov_atol
         result['krylov_rtol'] = self.krylov_rtol
         return result
     ```
   - Edge cases: Must pull from norm factory, not config
   - Integration: Maintains backward-compatible settings_dict interface

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_linear_solver.py
- Test function: test_linear_solver_settings_dict_includes_tolerance_arrays
- Description: Verify settings_dict includes krylov_atol and krylov_rtol from norm factory

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_settings_dict_includes_tolerance_arrays

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 9: Update NewtonKrylov settings_dict Property
**Status**: [ ]
**Dependencies**: Task Groups 5, 6

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (entire file after Task Groups 5, 6)

**Input Validation Required**:
- None

**Tasks**:
1. **Update NewtonKrylov.settings_dict to include tolerance arrays from norm**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     ```python
     @property
     def settings_dict(self) -> Dict[str, Any]:
         """Return merged Newton and linear solver configuration.
         
         Combines Newton-level settings from compile_settings with
         linear solver settings from nested linear_solver instance,
         plus tolerance arrays from the norm factory.
         
         Returns
         -------
         dict
             Merged configuration dictionary containing both Newton
             parameters, linear solver parameters, and tolerance arrays.
         """
         combined = dict(self.linear_solver.settings_dict)
         combined.update(self.compile_settings.settings_dict)
         combined['newton_atol'] = self.newton_atol
         combined['newton_rtol'] = self.newton_rtol
         return combined
     ```
   - Edge cases: Must pull from norm factory, not config
   - Integration: Maintains backward-compatible settings_dict interface

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_newton_krylov.py
- Test function: test_newton_krylov_settings_dict_includes_tolerance_arrays
- Description: Verify settings_dict includes newton_atol and newton_rtol from norm factory

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_settings_dict_includes_tolerance_arrays

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

**Total Task Groups**: 9

**Dependency Chain**:
```
Task Group 1 (MatrixFreeSolverConfig enhancements)
    ↓
Task Group 2 (MatrixFreeSolver base class)
    ↓
    ├── Task Group 3 (LinearSolverConfig refactor)
    │       ↓
    │   Task Group 4 (LinearSolver refactor)
    │       ↓
    │   Task Group 8 (LinearSolver settings_dict)
    │
    └── Task Group 5 (NewtonKrylovConfig refactor)
            ↓
        Task Group 6 (NewtonKrylov refactor) [also depends on Task Group 4]
            ↓
        Task Group 9 (NewtonKrylov settings_dict)
            
Task Groups 4, 6 → Task Group 7 (Instrumented test files)
```

**Tests to Create**: 18 new test functions across 3 test files

**Tests to Run**: 
- All existing tests in tests/integrators/matrix_free_solvers/
- New tests created for each task group

**Estimated Complexity**: Medium-High
- Significant architectural changes to inheritance hierarchy
- Multiple files need coordinated updates
- Must maintain backward compatibility for public API
- Instrumented test files must mirror changes
