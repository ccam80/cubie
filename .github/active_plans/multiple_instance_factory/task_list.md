# Implementation Task List
# Feature: MultipleInstanceCUDAFactory
# Plan Reference: .github/active_plans/multiple_instance_factory/agent_plan.md

## Task Group 1: Add MultipleInstanceCUDAFactory Base Class
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/CUDAFactory.py (entire file, especially lines 287-557 for CUDAFactory class)

**Input Validation Required**:
- instance_label: Must be a non-empty string (use simple truthiness check)

**Tasks**:
1. **Add MultipleInstanceCUDAFactory class**
   - File: src/cubie/CUDAFactory.py
   - Action: Create new class after line 557 (after CUDAFactory class ends)
   - Details:
     ```python
     class MultipleInstanceCUDAFactory(CUDAFactory):
         """Factory for CUDA device functions with instance-specific prefixes.

         Extends CUDAFactory to automatically map prefixed configuration
         keys (e.g., ``krylov_atol``) to unprefixed internal keys (e.g.,
         ``atol``) during settings updates. Subclasses use ``instance_label``
         to differentiate configuration parameters when multiple instances
         coexist.

         Attributes
         ----------
         instance_label : str
             Prefix used to identify settings for this instance
             (e.g., "krylov", "newton"). Keys in update dicts matching
             ``{instance_label}_*`` are mapped to unprefixed equivalents.

         Notes
         -----
         The transformation occurs in ``update_compile_settings()``:
         1. Copy the updates dict to avoid side effects
         2. For each key matching ``{instance_label}_{suffix}``, add
            ``suffix`` with the same value
         3. Call parent's ``update_compile_settings()`` with transformed dict
         4. Both prefixed and unprefixed forms are recognized; prefixed
            takes precedence when both are present
         """

         def __init__(self, instance_label: str) -> None:
             """Initialize with instance label for prefix mapping.

             Parameters
             ----------
             instance_label : str
                 Prefix for external configuration keys. Should NOT
                 include trailing underscore (added automatically).
             """
             if not instance_label:
                 raise ValueError("instance_label must be a non-empty string")
             self.instance_label = instance_label
             super().__init__()

         def update_compile_settings(
             self, updates_dict=None, silent=False, **kwargs
         ) -> Set[str]:
             """Update compile settings with automatic prefix mapping.

             Intercepts update dicts to map prefixed keys (e.g.,
             ``krylov_max_iters``) to unprefixed keys (e.g., ``max_iters``)
             before forwarding to the parent class.

             Parameters
             ----------
             updates_dict : dict, optional
                 Mapping of setting names to new values. Keys matching
                 ``{instance_label}_*`` are mapped to unprefixed equivalents.
             silent : bool, default=False
                 Suppress errors for unrecognised parameters.
             **kwargs
                 Additional settings to update.

             Returns
             -------
             set[str]
                 Names of settings that were successfully updated.
             """
             if updates_dict is None:
                 updates_dict = {}
             updates_dict = updates_dict.copy()
             updates_dict.update(kwargs)

             if not updates_dict:
                 return set()

             # Transform prefixed keys to unprefixed equivalents
             prefix = f"{self.instance_label}_"
             transformed = {}
             for key, value in updates_dict.items():
                 transformed[key] = value
                 if key.startswith(prefix):
                     unprefixed = key[len(prefix):]
                     transformed[unprefixed] = value

             return super().update_compile_settings(
                 updates_dict=transformed, silent=silent
             )
     ```
   - Edge cases:
     - Empty instance_label: Raise ValueError
     - No matching prefixed keys: Dict passes through unchanged
     - Both prefixed and unprefixed present: Prefixed takes precedence
       (added after in transformed dict overwrites)
   - Integration: Subclasses (MatrixFreeSolver) will inherit from this
     instead of CUDAFactory

**Tests to Create**:
- Test file: tests/test_CUDAFactory.py
- Test function: test_multiple_instance_factory_prefix_mapping
- Description: Verify that prefixed keys are mapped to unprefixed equivalents
- Test function: test_multiple_instance_factory_mixed_keys
- Description: Verify prefixed takes precedence when both forms present
- Test function: test_multiple_instance_factory_no_prefix_match
- Description: Verify non-matching keys pass through unchanged
- Test function: test_multiple_instance_factory_instance_label_stored
- Description: Verify instance_label attribute is correctly stored
- Test function: test_multiple_instance_factory_empty_label_raises
- Description: Verify empty instance_label raises ValueError

**Tests to Run**:
- tests/test_CUDAFactory.py::test_multiple_instance_factory_prefix_mapping
- tests/test_CUDAFactory.py::test_multiple_instance_factory_mixed_keys
- tests/test_CUDAFactory.py::test_multiple_instance_factory_no_prefix_match
- tests/test_CUDAFactory.py::test_multiple_instance_factory_instance_label_stored
- tests/test_CUDAFactory.py::test_multiple_instance_factory_empty_label_raises

**Outcomes**: 
- Files Modified: 
  * src/cubie/CUDAFactory.py (86 lines added)
- Functions/Methods Added/Modified:
  * MultipleInstanceCUDAFactory class added (lines 559-643)
  * MultipleInstanceCUDAFactory.__init__() - validates and stores instance_label
  * MultipleInstanceCUDAFactory.update_compile_settings() - transforms prefixed keys
- Implementation Summary:
  Added new MultipleInstanceCUDAFactory class that extends CUDAFactory with
  automatic prefix mapping for configuration keys. The class accepts an
  instance_label parameter and maps keys matching ``{instance_label}_*`` to
  unprefixed equivalents during update_compile_settings() calls.
- Issues Flagged: None

---

## Task Group 2: Update MatrixFreeSolver to Use MultipleInstanceCUDAFactory
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/CUDAFactory.py (lines 287-350 for CUDAFactory.__init__, new MultipleInstanceCUDAFactory class)
- File: src/cubie/integrators/matrix_free_solvers/base_solver.py (entire file)

**Input Validation Required**:
- None beyond what is already validated in __init__

**Tasks**:
1. **Update import statement in base_solver.py**
   - File: src/cubie/integrators/matrix_free_solvers/base_solver.py
   - Action: Modify import on line 16
   - Details:
     Change:
     ```python
     from cubie.CUDAFactory import CUDAFactory, CUDAFactoryConfig
     ```
     To:
     ```python
     from cubie.CUDAFactory import (
         MultipleInstanceCUDAFactory,
         CUDAFactoryConfig,
     )
     ```
   - Integration: The module will use the new base class

2. **Update MatrixFreeSolver class inheritance**
   - File: src/cubie/integrators/matrix_free_solvers/base_solver.py
   - Action: Modify class declaration on line 51
   - Details:
     Change:
     ```python
     class MatrixFreeSolver(CUDAFactory):
     ```
     To:
     ```python
     class MatrixFreeSolver(MultipleInstanceCUDAFactory):
     ```
   - Integration: MatrixFreeSolver gains automatic prefix mapping

3. **Update MatrixFreeSolver.__init__ to pass instance_label**
   - File: src/cubie/integrators/matrix_free_solvers/base_solver.py
   - Action: Modify __init__ method (lines 68-105)
   - Details:
     The `__init__` signature stays the same (settings_prefix parameter).
     Update the super().__init__() call:
     
     Change lines 91-92:
     ```python
         self.settings_prefix = settings_prefix
         super().__init__()
     ```
     To:
     ```python
         self.settings_prefix = settings_prefix
         super().__init__(instance_label=settings_prefix)
     ```
   - Edge cases: None - settings_prefix is already validated by subclasses
   - Integration: The instance_label is now set in the base class, while
     settings_prefix is kept for backwards compatibility with existing code
     that references it directly (e.g., _extract_prefixed_tolerance)

**Tests to Create**:
- None (existing solver tests will validate the integration)

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_linear_solver.py
- tests/integrators/matrix_free_solvers/test_newton_krylov.py

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/matrix_free_solvers/base_solver.py (5 lines changed)
- Functions/Methods Added/Modified:
  * Import statement updated to use MultipleInstanceCUDAFactory
  * MatrixFreeSolver class inheritance changed from CUDAFactory to MultipleInstanceCUDAFactory
  * MatrixFreeSolver.__init__() updated to pass instance_label=settings_prefix to super().__init__()
- Implementation Summary:
  Updated MatrixFreeSolver to inherit from MultipleInstanceCUDAFactory instead of
  CUDAFactory. The settings_prefix parameter is now passed as instance_label to
  the parent class, enabling automatic prefix mapping for configuration keys.
  The settings_prefix attribute is retained for backwards compatibility with
  existing code (e.g., _extract_prefixed_tolerance method).
- Issues Flagged: None

---

## Task Group 3: Update NewtonKrylov to Remove Class Attribute
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py (entire file, especially lines 160-221)

**Input Validation Required**:
- None

**Tasks**:
1. **Remove class attribute settings_prefix from NewtonKrylov**
   - File: src/cubie/integrators/matrix_free_solvers/newton_krylov.py
   - Action: Delete line 167
   - Details:
     Remove this line:
     ```python
         settings_prefix = "newton_"
     ```
     The settings_prefix is already passed via `__init__` call on line 199.
     The class attribute is redundant and can cause confusion since the
     instance attribute (set by the super().__init__ call) is what's
     actually used.
   - Edge cases: None
   - Integration: The instance attribute from MatrixFreeSolver.__init__
     handles the prefix, and the class attribute is not referenced elsewhere

**Tests to Create**:
- None (existing tests validate behavior)

**Tests to Run**:
- tests/integrators/matrix_free_solvers/test_newton_krylov.py

**Outcomes**: 
- Files Modified: 
  * src/cubie/integrators/matrix_free_solvers/newton_krylov.py (2 lines changed)
- Functions/Methods Added/Modified:
  * Removed class attribute `settings_prefix = "newton_"` from NewtonKrylov class
- Implementation Summary:
  Removed the redundant class attribute `settings_prefix = "newton_"` from
  the NewtonKrylov class. The settings_prefix is already passed via the
  super().__init__() call (line 197), making the class attribute unnecessary
  and potentially confusing.
- Issues Flagged: None

---

## Task Group 4: Add Tests for MultipleInstanceCUDAFactory
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/test_CUDAFactory.py (entire file)
- File: src/cubie/CUDAFactory.py (MultipleInstanceCUDAFactory class)
- File: src/cubie/integrators/matrix_free_solvers/linear_solver.py (lines 139-193 for LinearSolver.__init__)

**Input Validation Required**:
- None (tests validate the class behavior)

**Tasks**:
1. **Add import for MultipleInstanceCUDAFactory in test file**
   - File: tests/test_CUDAFactory.py
   - Action: Modify import on lines 7-11
   - Details:
     Change:
     ```python
     from cubie.CUDAFactory import (
         CUDAFactory,
         CUDADispatcherCache,
         _CubieConfigBase,
         CUDAFactoryConfig,
     )
     ```
     To:
     ```python
     from cubie.CUDAFactory import (
         CUDAFactory,
         CUDADispatcherCache,
         _CubieConfigBase,
         CUDAFactoryConfig,
         MultipleInstanceCUDAFactory,
     )
     ```

2. **Add test_multiple_instance_factory_prefix_mapping**
   - File: tests/test_CUDAFactory.py
   - Action: Add test at end of file
   - Details:
     ```python
     def test_multiple_instance_factory_prefix_mapping(precision):
         """Test that prefixed keys are mapped to unprefixed equivalents."""
         from cubie.integrators.matrix_free_solvers.linear_solver import (
             LinearSolver,
         )

         solver = LinearSolver(precision=precision, n=3)

         # Update with prefixed key
         solver.update({"krylov_tolerance": 1e-8})

         # Verify the unprefixed setting was updated
         assert solver.compile_settings.krylov_tolerance == precision(1e-8)
     ```

3. **Add test_multiple_instance_factory_instance_label_stored**
   - File: tests/test_CUDAFactory.py
   - Action: Add test after previous test
   - Details:
     ```python
     def test_multiple_instance_factory_instance_label_stored(precision):
         """Test that instance_label attribute is correctly stored."""
         from cubie.integrators.matrix_free_solvers.linear_solver import (
             LinearSolver,
         )

         solver = LinearSolver(precision=precision, n=3)

         # Verify instance_label is set correctly
         assert solver.instance_label == "krylov_"
     ```

4. **Add test_multiple_instance_factory_empty_label_raises**
   - File: tests/test_CUDAFactory.py
   - Action: Add test after previous test
   - Details:
     ```python
     def test_multiple_instance_factory_empty_label_raises():
         """Test that empty instance_label raises ValueError."""

         class TestFactory(MultipleInstanceCUDAFactory):
             def build(self):
                 return testCache(device_function=lambda: 1.0)

         with pytest.raises(ValueError) as exc:
             TestFactory(instance_label="")

         assert "non-empty" in str(exc.value)
     ```

5. **Add test_multiple_instance_factory_mixed_keys**
   - File: tests/test_CUDAFactory.py
   - Action: Add test after previous test
   - Details:
     ```python
     def test_multiple_instance_factory_mixed_keys():
         """Test that prefixed keys take precedence over unprefixed."""

         @attrs.define
         class TestConfig(CUDAFactoryConfig):
             value: int = 10

         class TestFactory(MultipleInstanceCUDAFactory):
             def __init__(self):
                 super().__init__(instance_label="test")
                 self.setup_compile_settings(TestConfig(precision=np.float32))

             def build(self):
                 return testCache(device_function=lambda: 1.0)

         factory = TestFactory()

         # Update with both prefixed and unprefixed - prefixed should win
         factory.update_compile_settings({"value": 5, "test_value": 20})

         assert factory.compile_settings.value == 20
     ```

6. **Add test_multiple_instance_factory_no_prefix_match**
   - File: tests/test_CUDAFactory.py
   - Action: Add test after previous test
   - Details:
     ```python
     def test_multiple_instance_factory_no_prefix_match():
         """Test that non-matching keys pass through unchanged."""

         @attrs.define
         class TestConfig(CUDAFactoryConfig):
             value: int = 10

         class TestFactory(MultipleInstanceCUDAFactory):
             def __init__(self):
                 super().__init__(instance_label="test")
                 self.setup_compile_settings(TestConfig(precision=np.float32))

             def build(self):
                 return testCache(device_function=lambda: 1.0)

         factory = TestFactory()

         # Update with non-prefixed key
         factory.update_compile_settings({"value": 42})

         assert factory.compile_settings.value == 42
     ```

**Tests to Create**:
- All tests listed above

**Tests to Run**:
- tests/test_CUDAFactory.py::test_multiple_instance_factory_prefix_mapping
- tests/test_CUDAFactory.py::test_multiple_instance_factory_instance_label_stored
- tests/test_CUDAFactory.py::test_multiple_instance_factory_empty_label_raises
- tests/test_CUDAFactory.py::test_multiple_instance_factory_mixed_keys
- tests/test_CUDAFactory.py::test_multiple_instance_factory_no_prefix_match

**Outcomes**:
- Files Modified: 
  * tests/test_CUDAFactory.py (86 lines added)
- Functions/Methods Added/Modified:
  * test_multiple_instance_factory_prefix_mapping() - Tests prefixed key mapping
  * test_multiple_instance_factory_instance_label_stored() - Tests instance_label attribute
  * test_multiple_instance_factory_empty_label_raises() - Tests empty label validation
  * test_multiple_instance_factory_mixed_keys() - Tests prefixed key precedence
  * test_multiple_instance_factory_no_prefix_match() - Tests non-matching key pass-through
- Implementation Summary:
  Added 5 tests for MultipleInstanceCUDAFactory class. Tests cover prefix mapping
  via LinearSolver integration, instance_label storage, empty label validation,
  mixed key precedence (prefixed wins), and non-matching key pass-through.
  Added MultipleInstanceCUDAFactory to imports at top of file.
- Issues Flagged: None

---

## Summary

### Task Group Dependencies
```
Task Group 1 (MultipleInstanceCUDAFactory base class)
         │
         ├─────────────────────┐
         │                     │
         v                     v
Task Group 2              Task Group 4
(MatrixFreeSolver)        (Tests for base class)
         │
         v
Task Group 3
(NewtonKrylov cleanup)
```

### Files Modified
1. `src/cubie/CUDAFactory.py` - Add MultipleInstanceCUDAFactory class
2. `src/cubie/integrators/matrix_free_solvers/base_solver.py` - Update inheritance
3. `src/cubie/integrators/matrix_free_solvers/newton_krylov.py` - Remove class attribute
4. `tests/test_CUDAFactory.py` - Add tests for new class

### Estimated Complexity
- Task Group 1: Medium (~60 lines of new code)
- Task Group 2: Low (~5 line changes)
- Task Group 3: Low (1 line deletion)
- Task Group 4: Medium (~60 lines of new tests)

### Total: 4 task groups, ~125 lines changed
