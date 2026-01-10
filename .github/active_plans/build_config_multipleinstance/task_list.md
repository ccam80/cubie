# Implementation Task List
# Feature: MultipleInstance build_config Integration
# Plan Reference: .github/active_plans/build_config_multipleinstance/agent_plan.md

## Task Group 1: Enhance build_config with instance_label Support
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/_utils.py (lines 714-792 - build_config function)
- File: src/cubie/CUDAFactory.py (lines 559-660 - MultipleInstanceCUDAFactoryConfig class)

**Input Validation Required**:
- instance_label: Check type is str (if provided), allow empty string or None
- config_class: Must be an attrs class (already validated)

**Tasks**:
1. **Add instance_label parameter to build_config**
   - File: src/cubie/_utils.py
   - Action: Modify
   - Details:
     ```python
     def build_config(
         config_class: type,
         required: dict,
         instance_label: str = "",
         **optional
     ) -> Any:
         """Build attrs config instance from required and optional parameters.
         
         ...existing docstring...
         
         Parameters
         ----------
         config_class : type
             Attrs class to instantiate (e.g., DIRKStepConfig).
         required : dict
             Required parameters that must be provided.
         instance_label : str, optional
             Instance label for MultipleInstanceCUDAFactoryConfig classes.
             When provided, prefixed keys (e.g., 'krylov_atol') are
             transformed to unprefixed keys ('atol') before field matching.
             Default is empty string (no prefix transformation).
         **optional
             Optional parameter overrides passed to the config constructor.
         
         ...rest of docstring...
         """
     ```
   - Edge cases: Empty string instance_label means no prefix transformation
   - Integration: Affects all callers that use MultipleInstance classes

2. **Implement prefix transformation logic in build_config**
   - File: src/cubie/_utils.py
   - Action: Modify
   - Details:
     ```python
     # After merging required and optional kwargs:
     merged = {**required, **optional}
     
     # Apply prefix transformation if instance_label is provided and 
     # config_class has get_prefixed_attributes
     if instance_label and hasattr(config_class, 'get_prefixed_attributes'):
         prefixed_attrs = config_class.get_prefixed_attributes()
         prefix = f"{instance_label}_"
         
         # For each prefixed attribute, check for prefixed key in merged
         for attr in prefixed_attrs:
             prefixed_key = f"{prefix}{attr}"
             if prefixed_key in merged:
                 # Prefixed key takes precedence - copy to unprefixed
                 merged[attr] = merged[prefixed_key]
         
         # Add instance_label to merged for config constructor
         merged["instance_label"] = instance_label
     ```
   - Edge cases: 
     - Both prefixed and unprefixed provided: prefixed wins
     - Attribute not in prefixed set: pass through unchanged
   - Integration: Uses get_prefixed_attributes() classmethod

3. **Update build_config docstring with examples**
   - File: src/cubie/_utils.py
   - Action: Modify
   - Details: Add example showing instance_label usage:
     ```python
     Examples
     --------
     >>> # Without instance_label (current behavior)
     >>> config = build_config(SomeConfig, required={'precision': np.float32})
     >>> 
     >>> # With instance_label (prefix transformation)
     >>> config = build_config(
     ...     ScaledNormConfig,
     ...     required={'precision': np.float32, 'n': 3},
     ...     instance_label="krylov",
     ...     krylov_atol=1e-6,  # Transformed to atol
     ... )
     ```
   - Edge cases: None
   - Integration: Documentation update

**Tests to Create**:
- Test file: tests/test_CUDAFactory.py
- Test function: test_build_config_with_instance_label
- Description: Verify prefix transformation works with instance_label parameter
- Test function: test_build_config_instance_label_prefixed_takes_precedence
- Description: Verify prefixed key wins when both prefixed and unprefixed provided
- Test function: test_build_config_backward_compatible_no_instance_label
- Description: Verify existing behavior unchanged when instance_label not provided
- Test function: test_build_config_instance_label_non_prefixed_class
- Description: Verify no transformation for classes without get_prefixed_attributes

**Tests to Run**:
- tests/test_CUDAFactory.py::test_build_config_with_instance_label
- tests/test_CUDAFactory.py::test_build_config_instance_label_prefixed_takes_precedence
- tests/test_CUDAFactory.py::test_build_config_backward_compatible_no_instance_label
- tests/test_CUDAFactory.py::test_build_config_instance_label_non_prefixed_class

**Outcomes**: 
- Files Modified:
  * src/cubie/_utils.py (38 lines changed)
  * tests/test_CUDAFactory.py (94 lines added)
- Functions/Methods Added/Modified:
  * build_config() in _utils.py - added instance_label parameter and prefix transformation logic
- Implementation Summary:
  Added instance_label parameter to build_config function with default empty string.
  When instance_label is provided and config_class has get_prefixed_attributes method,
  prefixed keys (e.g., 'krylov_atol') are transformed to unprefixed keys ('atol').
  Prefixed keys take precedence over unprefixed when both are provided.
  Added type validation for instance_label parameter.
  Updated docstring with comprehensive examples showing both usage patterns.
  Created 4 new tests verifying prefix transformation, precedence, backward compatibility,
  and behavior with non-prefixed config classes.
- Issues Flagged: None

---

## Task Group 2: Fix MultipleInstanceCUDAFactoryConfig Post-Init and Add prefix Property
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/CUDAFactory.py (lines 559-660 - MultipleInstanceCUDAFactoryConfig class)
- File: src/cubie/CUDAFactory.py (lines 80-234 - _CubieConfigBase class)

**Input Validation Required**:
- No additional validation needed (existing validators sufficient)

**Tasks**:
1. **Fix _attrs_post_init__ typo to __attrs_post_init__**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     Line 586 has `_attrs_post_init__` (missing leading underscore).
     Change to `__attrs_post_init__`:
     ```python
     def __attrs_post_init__(self):
         super().__attrs_post_init__()
         if self.instance_label != "":
             prefixed_attributes = set(
                 fld.name
                 for fld in fields(type(self))
                 if fld.metadata.get("prefixed", True)
             )
             self.prefixed_attributes = prefixed_attributes
     ```
   - Edge cases: None (bug fix)
   - Integration: Fixes prefixed_attributes population on init

2. **Add prefix property to MultipleInstanceCUDAFactoryConfig**
   - File: src/cubie/CUDAFactory.py
   - Action: Modify
   - Details:
     Add property after instance_label field definition:
     ```python
     @property
     def prefix(self) -> str:
         """Return the prefix string for this instance.
         
         Returns
         -------
         str
             The instance_label value (prefix without trailing underscore).
         """
         return self.instance_label
     ```
   - Edge cases: Empty instance_label returns empty string
   - Integration: Used by update() method, consistent with factory classes

**Tests to Create**:
- Test file: tests/test_CUDAFactory.py
- Test function: test_multiple_instance_config_prefix_property
- Description: Verify prefix property returns instance_label
- Test function: test_multiple_instance_config_post_init_populates_prefixed_attrs
- Description: Verify __attrs_post_init__ correctly populates prefixed_attributes

**Tests to Run**:
- tests/test_CUDAFactory.py::test_multiple_instance_config_prefix_property
- tests/test_CUDAFactory.py::test_multiple_instance_config_post_init_populates_prefixed_attrs

**Outcomes**: 
- Files Modified:
  * src/cubie/CUDAFactory.py (11 lines changed)
  * tests/test_CUDAFactory.py (46 lines added)
- Functions/Methods Added/Modified:
  * prefix property added to MultipleInstanceCUDAFactoryConfig
  * _attrs_post_init__ renamed to __attrs_post_init__ in MultipleInstanceCUDAFactoryConfig
- Implementation Summary:
  Fixed the typo in _attrs_post_init__ method name by adding the missing leading
  underscore, which was preventing prefixed_attributes from being populated on init.
  Added prefix property that returns instance_label for consistency with factory classes.
  Created 2 new tests verifying the prefix property and the post-init behavior.
- Issues Flagged: None

---

## Task Group 3: Refactor ScaledNorm to Use Enhanced build_config
**Status**: [x]
**Dependencies**: Task Group 1, Task Group 2

**Required Context**:
- File: src/cubie/integrators/norms.py (entire file - lines 1-245)
- File: src/cubie/_utils.py (lines 714-792 - build_config function)
- File: src/cubie/CUDAFactory.py (lines 662-708 - MultipleInstanceCUDAFactory class)

**Input Validation Required**:
- No additional validation (existing validators in ScaledNormConfig sufficient)

**Tasks**:
1. **Rename instance_type parameter to instance_label in ScaledNorm**
   - File: src/cubie/integrators/norms.py
   - Action: Modify
   - Details:
     In ScaledNorm.__init__, change:
     ```python
     def __init__(
         self,
         precision: PrecisionDType,
         n: int,
         instance_label: str = "",  # Changed from instance_type
         **kwargs,
     ) -> None:
     ```
     And update the super().__init__ call:
     ```python
     super().__init__(instance_label=instance_label)
     ```
   - Edge cases: None (parameter rename)
   - Integration: Consistent naming with MultipleInstanceCUDAFactory

2. **Update ScaledNorm to use enhanced build_config with instance_label**
   - File: src/cubie/integrators/norms.py
   - Action: Modify
   - Details:
     Replace the current build_config call with:
     ```python
     config = build_config(
         ScaledNormConfig,
         required={
             "precision": precision,
             "n": n,
         },
         instance_label=instance_label,
         **kwargs,
     )
     ```
     Remove the "instance_type" from required dict and use the new
     instance_label parameter instead.
   - Edge cases: Empty instance_label should work (no prefix transformation)
   - Integration: Prefix transformation handled by build_config

3. **Remove TODO comment about init_from_prefixed**
   - File: src/cubie/integrators/norms.py
   - Action: Modify
   - Details:
     Remove the comment on line 123-124:
     ```python
     config = build_config(  # Need to get init_from_prefixed into here
         # somehow.
     ```
     Should become:
     ```python
     config = build_config(
     ```
   - Edge cases: None
   - Integration: Code cleanup

**Tests to Create**:
- Test file: tests/integrators/test_norms.py (create if not exists)
- Test function: test_scaled_norm_instance_label_prefix_at_init
- Description: Verify ScaledNorm accepts prefixed kwargs (e.g., krylov_atol) at init
- Test function: test_scaled_norm_instance_label_empty_no_prefix
- Description: Verify empty instance_label works with unprefixed kwargs

**Tests to Run**:
- tests/integrators/test_norms.py::test_scaled_norm_instance_label_prefix_at_init
- tests/integrators/test_norms.py::test_scaled_norm_instance_label_empty_no_prefix

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/norms.py (10 lines changed)
  * tests/integrators/test_norms.py (45 lines added)
- Functions/Methods Added/Modified:
  * ScaledNorm.__init__() in norms.py - renamed instance_type to instance_label, 
    updated super().__init__ call, updated build_config call with instance_label
    parameter, removed TODO comment
- Implementation Summary:
  Renamed instance_type parameter to instance_label for consistency with
  MultipleInstanceCUDAFactory. Updated build_config call to use the new
  instance_label parameter for prefix transformation. Removed "instance_type"
  from required dict since instance_label is now passed as a separate parameter.
  Removed TODO comment about init_from_prefixed since prefix transformation
  is now handled by the enhanced build_config. Created 2 new tests verifying
  prefix transformation with instance_label and empty instance_label behavior.
- Issues Flagged: None

---

## Task Group 4: Refactor LinearSolver to Use Enhanced build_config and Forward kwargs
**Status**: [x]
**Dependencies**: Task Group 1, Task Group 3

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 123-175 - LinearSolver.__init__)
- File: src/cubie/integrators/matrix_free_solvers/base_solver.py (lines 56-146 - MatrixFreeSolver class)
- File: src/cubie/_utils.py (lines 714-792 - build_config function)

**Input Validation Required**:
- No additional validation (existing validators sufficient)

**Tasks**:
1. **Remove duplicate init_from_prefixed call in LinearSolver.__init__**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     Remove lines 150-153:
     ```python
     # Conflict here
     compile_settings = LinearSolverConfig.init_from_prefixed(
         precision=precision, n=n, **kwargs
     )
     ```
   - Edge cases: None (removing duplicate code)
   - Integration: Single path via build_config

2. **Update LinearSolver to use enhanced build_config with instance_label**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     Update the build_config call (lines 154-161) to:
     ```python
     config = build_config(
         LinearSolverConfig,
         required={
             "precision": precision,
             "n": n,
         },
         instance_label="krylov",
         **kwargs,
     )
     ```
   - Edge cases: None
   - Integration: Prefix transformation for krylov_* kwargs

3. **Forward all kwargs to parent MatrixFreeSolver**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     Update super().__init__ call (lines 162-168) to pass all kwargs:
     ```python
     super().__init__(
         precision=precision,
         solver_type="krylov",
         n=n,
         **kwargs,
     )
     ```
   - Edge cases: kwargs already consumed by build_config are harmless (ignored)
   - Integration: Enables nested parameter propagation to ScaledNorm

4. **Use config from build_config instead of compile_settings variable**
   - File: src/cubie/integrators/matrix_free_solvers/linear_solver.py
   - Action: Modify
   - Details:
     Update line 170 from:
     ```python
     self.setup_compile_settings(compile_settings)
     ```
     to:
     ```python
     self.setup_compile_settings(config)
     ```
   - Edge cases: None
   - Integration: Uses correctly constructed config

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_linear_solver.py
- Test function: test_linear_solver_init_with_krylov_prefixed_kwargs
- Description: Verify LinearSolver accepts krylov_* kwargs at init and they reach config/norm
- Test function: test_linear_solver_forwards_kwargs_to_norm
- Description: Verify kwargs passed to LinearSolver reach the nested ScaledNorm

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_init_with_krylov_prefixed_kwargs
- tests/integrators/matrix_free_solvers/test_linear_solver.py::test_linear_solver_forwards_kwargs_to_norm

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/linear_solver.py (8 lines changed)
  * tests/integrators/matrix_free_solvers/test_linear_solver.py (54 lines added)
- Functions/Methods Added/Modified:
  * LinearSolver.__init__() in linear_solver.py - removed duplicate init_from_prefixed call,
    updated build_config call with instance_label="krylov", added **kwargs to super().__init__,
    changed compile_settings to config variable
- Implementation Summary:
  Refactored LinearSolver.__init__ to use the enhanced build_config with instance_label="krylov"
  for automatic prefix transformation of krylov_* kwargs. Removed the duplicate init_from_prefixed
  call that was creating a conflict with build_config. Added **kwargs forwarding to the parent
  MatrixFreeSolver.__init__ to enable nested parameter propagation to ScaledNorm. Changed variable
  name from compile_settings to config for consistency with build_config output.
  Created 2 new tests verifying prefixed kwargs acceptance at init and kwargs forwarding to norm.
- Issues Flagged: None

---

## Task Group 5: Refactor NewtonKrylov to Use Enhanced build_config and Forward All kwargs
**Status**: [x]
**Dependencies**: Task Group 1, Task Group 4

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 155-210 - NewtonKrylov.__init__)
- File: src/cubie/integrators/matrix_free_solvers/base_solver.py (lines 56-146 - MatrixFreeSolver class)
- File: src/cubie/_utils.py (lines 714-792 - build_config function)

**Input Validation Required**:
- No additional validation (existing validators sufficient)

**Tasks**:
1. **Remove duplicate init_from_prefixed call in NewtonKrylov.__init__**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     Remove lines 186-189:
     ```python
     # Conflict here
     compile_settings = NewtonKrylovConfig.init_from_prefixed(
         precision=precision, n=n, **kwargs
     )
     ```
   - Edge cases: None (removing duplicate code)
   - Integration: Single path via build_config

2. **Update NewtonKrylov to use enhanced build_config with instance_label**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     Update the build_config call (lines 190-197) to:
     ```python
     config = build_config(
         NewtonKrylovConfig,
         required={
             "precision": precision,
             "n": n,
         },
         instance_label="newton",
         **kwargs,
     )
     ```
   - Edge cases: None
   - Integration: Prefix transformation for newton_* kwargs

3. **Forward all kwargs to parent MatrixFreeSolver**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     Update super().__init__ call (lines 198-204) to pass all kwargs:
     ```python
     super().__init__(
         precision=precision,
         solver_type="newton",
         n=n,
         **kwargs,
     )
     ```
   - Edge cases: kwargs already consumed by build_config are harmless
   - Integration: Enables nested parameter propagation to ScaledNorm

4. **Use config from build_config instead of compile_settings variable**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Modify
   - Details:
     Update line 207 from:
     ```python
     self.setup_compile_settings(compile_settings)
     ```
     to:
     ```python
     self.setup_compile_settings(config)
     ```
   - Edge cases: None
   - Integration: Uses correctly constructed config

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_newton_krylov.py
- Test function: test_newton_krylov_init_with_newton_prefixed_kwargs
- Description: Verify NewtonKrylov accepts newton_* kwargs at init and they reach config/norm
- Test function: test_newton_krylov_forwards_krylov_kwargs_to_linear_solver
- Description: Verify krylov_* kwargs passed to NewtonKrylov reach the nested LinearSolver

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_init_with_newton_prefixed_kwargs
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_newton_krylov_forwards_krylov_kwargs_to_linear_solver

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (8 lines changed)
  * tests/integrators/matrix_free_solvers/test_newton_krylov.py (54 lines added)
- Functions/Methods Added/Modified:
  * NewtonKrylov.__init__() in newton_krylov.py - removed duplicate init_from_prefixed call,
    updated build_config call with instance_label="newton", added **kwargs to super().__init__,
    changed compile_settings to config variable
- Implementation Summary:
  Refactored NewtonKrylov.__init__ to use the enhanced build_config with instance_label="newton"
  for automatic prefix transformation of newton_* kwargs. Removed the duplicate init_from_prefixed
  call that was creating a conflict with build_config. Added **kwargs forwarding to the parent
  MatrixFreeSolver.__init__ to enable nested parameter propagation to ScaledNorm. Changed variable
  name from compile_settings to config for consistency with build_config output.
  Created 2 new tests verifying prefixed kwargs acceptance at init and kwargs forwarding to norm.
- Issues Flagged: None

---

## Task Group 6: Nested Parameter Propagation Tests
**Status**: [x]
**Dependencies**: Task Group 3, Task Group 4, Task Group 5

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (entire file)
- File: src/cubie/integrators/matrix_free_solvers/base_solver.py (entire file)
- File: src/cubie/integrators/norms.py (entire file)
- File: tests/integrators/matrix_free_solvers/test_newton_krylov.py (entire file)

**Input Validation Required**:
- None (test-only task group)

**Tasks**:
1. **Create test for full nested parameter propagation at init**
   - File: tests/integrators/matrix_free_solvers/test_newton_krylov.py
   - Action: Add test
   - Details:
     ```python
     def test_nested_prefix_propagation_init(precision):
         """Verify prefixed params reach nested objects via init chain.
         
         Tests that krylov_atol passed to NewtonKrylov constructor
         reaches the nested LinearSolver's ScaledNorm at init time.
         """
         n = 3
         krylov_atol = np.array([1e-10, 1e-9, 1e-8], dtype=precision)
         krylov_rtol = np.array([1e-5, 1e-4, 1e-3], dtype=precision)
         
         linear_solver = LinearSolver(
             precision=precision,
             n=n,
             krylov_atol=krylov_atol,
             krylov_rtol=krylov_rtol,
         )
         
         # Verify tolerances reached LinearSolver's norm
         assert np.allclose(linear_solver.krylov_atol, krylov_atol)
         assert np.allclose(linear_solver.krylov_rtol, krylov_rtol)
         assert np.allclose(linear_solver.norm.atol, krylov_atol)
         assert np.allclose(linear_solver.norm.rtol, krylov_rtol)
     ```
   - Edge cases: None
   - Integration: Validates init-time propagation

2. **Create test for full nested parameter propagation via update**
   - File: tests/integrators/matrix_free_solvers/test_newton_krylov.py
   - Action: Add test
   - Details:
     ```python
     def test_nested_prefix_propagation_update(precision):
         """Verify prefixed params reach nested objects via update chain.
         
         Tests that krylov_atol passed to NewtonKrylov.update()
         reaches the nested LinearSolver's ScaledNorm.
         """
         n = 3
         linear_solver = LinearSolver(precision=precision, n=n)
         newton = NewtonKrylov(
             precision=precision,
             n=n,
             linear_solver=linear_solver,
         )
         
         new_krylov_atol = np.array([1e-12, 1e-11, 1e-10], dtype=precision)
         
         # Update via newton with krylov-prefixed key
         newton.update(krylov_atol=new_krylov_atol)
         
         # Verify update reached nested LinearSolver and its norm
         assert np.allclose(newton.krylov_atol, new_krylov_atol)
         assert np.allclose(newton.linear_solver.krylov_atol, new_krylov_atol)
         assert np.allclose(newton.linear_solver.norm.atol, new_krylov_atol)
     ```
   - Edge cases: None
   - Integration: Validates update-time propagation

3. **Create test verifying no manual key filtering in classes**
   - File: tests/test_CUDAFactory.py
   - Action: Add test
   - Details:
     ```python
     def test_no_manual_key_filtering(precision):
         """Verify factory classes don't manually filter keys.
         
         All kwargs should pass through to nested objects; each level
         extracts its own via build_config/update and ignores the rest.
         """
         from cubie.integrators.matrix_free_solvers.linear_solver import (
             LinearSolver,
         )
         
         # Pass unrelated kwargs - they should be silently ignored
         solver = LinearSolver(
             precision=precision,
             n=3,
             unrelated_param=42,  # Should not raise
             another_unknown="value",  # Should not raise
         )
         
         # Verify solver was created successfully
         assert solver.n == 3
     ```
   - Edge cases: Unknown kwargs should not raise errors
   - Integration: Validates no manual filtering

**Tests to Create**:
- All tests described above in Tasks section

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_nested_prefix_propagation_init
- tests/integrators/matrix_free_solvers/test_newton_krylov.py::test_nested_prefix_propagation_update
- tests/test_CUDAFactory.py::test_no_manual_key_filtering

**Outcomes**: 
- Files Modified:
  * tests/integrators/matrix_free_solvers/test_newton_krylov.py (48 lines added)
  * tests/test_CUDAFactory.py (21 lines added)
- Functions/Methods Added/Modified:
  * test_nested_prefix_propagation_init() in test_newton_krylov.py
  * test_nested_prefix_propagation_update() in test_newton_krylov.py
  * test_no_manual_key_filtering() in test_CUDAFactory.py
- Implementation Summary:
  Added three integration tests validating the full refactor is working correctly:
  1. test_nested_prefix_propagation_init - Verifies krylov_atol/rtol passed to 
     LinearSolver constructor reaches the nested ScaledNorm at init time
  2. test_nested_prefix_propagation_update - Verifies krylov_atol passed to 
     NewtonKrylov.update() reaches the nested LinearSolver's ScaledNorm
  3. test_no_manual_key_filtering - Verifies that unknown kwargs are silently 
     ignored by factory classes rather than raising errors
- Issues Flagged: None

---

## Task Group 7: Update Base Solver kwargs Forwarding
**Status**: [x]
**Dependencies**: Task Group 3 (completed)

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/base_solver.py (lines 73-100 - MatrixFreeSolver.__init__)
- File: src/cubie/integrators/norms.py (lines 102-134 - ScaledNorm.__init__)

**Input Validation Required**:
- No additional validation needed

**Tasks**:
1. **Update MatrixFreeSolver to pass instance_label to ScaledNorm**
   - File: src/cubie/integrators/matrix_free_solvers/base_solver.py
   - Action: Modify
   - Details:
     Update the ScaledNorm instantiation (lines 95-100) to use
     instance_label instead of instance_type:
     ```python
     self.norm = ScaledNorm(
         precision=precision,
         n=n,
         instance_label=solver_type,  # Changed from instance_type
         **kwargs,
     )
     ```
   - Edge cases: None (parameter rename)
   - Integration: Consistent with ScaledNorm changes in Task Group 3

**Tests to Create**:
- Test file: tests/integrators/matrix_free_solvers/test_base_solver.py
- Test function: test_matrix_free_solver_forwards_kwargs_to_norm
- Description: Verify kwargs passed to MatrixFreeSolver reach ScaledNorm

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_base_solver.py::test_matrix_free_solver_forwards_kwargs_to_norm

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/matrix_free_solvers/base_solver.py (1 line changed)
  * tests/integrators/matrix_free_solvers/test_base_solver.py (28 lines added)
- Functions/Methods Added/Modified:
  * MatrixFreeSolver.__init__() in base_solver.py - changed instance_type to instance_label
  * test_matrix_free_solver_forwards_kwargs_to_norm() in test_base_solver.py - new test
- Implementation Summary:
  Updated ScaledNorm instantiation in MatrixFreeSolver.__init__ to use the new
  instance_label parameter name instead of the deprecated instance_type parameter.
  This aligns with the ScaledNorm signature change made in Task Group 3.
  Created a new test that verifies prefixed kwargs (e.g., krylov_atol, krylov_rtol)
  are correctly forwarded through MatrixFreeSolver to the nested ScaledNorm factory.
- Issues Flagged: None

---

## Task Group 8: Fix Existing Test Failures
**Status**: [x]
**Dependencies**: Task Group 2, Task Group 7

**Required Context**:
- File: tests/integrators/matrix_free_solvers/test_base_solver.py (lines 118-145)

**Input Validation Required**:
- None (test-only task group)

**Tasks**:
1. **Update test_matrix_free_solver_extract_prefixed_tolerance**
   - File: tests/integrators/matrix_free_solvers/test_base_solver.py
   - Action: Modify or Remove
   - Details:
     The test `test_matrix_free_solver_extract_prefixed_tolerance` tests
     a method `_extract_prefixed_tolerance` which may no longer exist
     after refactoring. Either:
     a) Remove the test if the method is removed, or
     b) Update the test to match new implementation
     
     Based on the current base_solver.py, this method doesn't exist
     (see lines 102-145). The test needs to be removed as it tests
     non-existent functionality.
   - Edge cases: None
   - Integration: Test cleanup

2. **Update test_matrix_free_solver_norm_update_propagates_to_config**
   - File: tests/integrators/matrix_free_solvers/test_base_solver.py
   - Action: Modify or Remove
   - Details:
     The test `test_matrix_free_solver_norm_update_propagates_to_config`
     tests a method `_update_norm_and_config` which may no longer exist.
     Based on current base_solver.py (lines 102-145), this method doesn't
     exist. The test needs to be removed or updated.
   - Edge cases: None
   - Integration: Test cleanup

**Tests to Create**:
- None (this group fixes/removes broken tests)

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_base_solver.py (full file to verify no failures)

**Outcomes**: 
- Files Modified:
  * tests/integrators/matrix_free_solvers/test_base_solver.py (67 lines removed)
- Functions/Methods Added/Modified:
  * test_matrix_free_solver_extract_prefixed_tolerance() - REMOVED
  * test_matrix_free_solver_norm_update_propagates_to_config() - REMOVED
- Implementation Summary:
  Removed two obsolete tests that were testing methods no longer present in
  base_solver.py after the refactoring:
  1. test_matrix_free_solver_extract_prefixed_tolerance - tested the removed
     _extract_prefixed_tolerance method
  2. test_matrix_free_solver_norm_update_propagates_to_config - tested the
     removed _update_norm_and_config method
  The functionality these methods provided has been replaced by the enhanced
  build_config with instance_label and the refactored update() method in
  MatrixFreeSolver that now directly forwards updates to the norm factory.
- Issues Flagged: None

---

## Summary

### Total Task Groups: 8

### Dependency Chain Overview:
```
Task Group 1 (build_config enhancement)
    ↓
Task Group 2 (MultipleInstanceCUDAFactoryConfig fixes)
    ↓
Task Group 3 (ScaledNorm refactor) ← also depends on TG1
    ↓
Task Group 7 (Base Solver kwargs) ← depends on TG3
    ↓
Task Group 4 (LinearSolver refactor) ← depends on TG1, TG3
    ↓
Task Group 5 (NewtonKrylov refactor) ← depends on TG1, TG4
    ↓
Task Group 6 (Nested propagation tests) ← depends on TG3, TG4, TG5
    ↓
Task Group 8 (Fix broken tests) ← depends on TG2, TG7
```

### Tests to be Created:
1. `test_build_config_with_instance_label`
2. `test_build_config_instance_label_prefixed_takes_precedence`
3. `test_build_config_backward_compatible_no_instance_label`
4. `test_build_config_instance_label_non_prefixed_class`
5. `test_multiple_instance_config_prefix_property`
6. `test_multiple_instance_config_post_init_populates_prefixed_attrs`
7. `test_scaled_norm_instance_label_prefix_at_init`
8. `test_scaled_norm_instance_label_empty_no_prefix`
9. `test_linear_solver_init_with_krylov_prefixed_kwargs`
10. `test_linear_solver_forwards_kwargs_to_norm`
11. `test_newton_krylov_init_with_newton_prefixed_kwargs`
12. `test_newton_krylov_forwards_krylov_kwargs_to_linear_solver`
13. `test_nested_prefix_propagation_init`
14. `test_nested_prefix_propagation_update`
15. `test_no_manual_key_filtering`
16. `test_matrix_free_solver_forwards_kwargs_to_norm`

### Estimated Complexity: Medium-High
- Core logic changes in build_config function
- Multiple factory class refactoring
- Comprehensive test coverage needed
- Nested object parameter forwarding validation
