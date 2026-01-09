# Implementation Task List
# Feature: PR Review Changes
# Plan Reference: .github/active_plans/pr_review_changes/agent_plan.md

---

## Task Group 1: Move CUDASIM-Conditioned Imports to cuda_simsafe.py
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/cuda_simsafe.py (entire file, focus on lines 116-164 for pattern, lines 316-347 for `__all__`)
- File: src/cubie/cubie_cache.py (lines 1-41 for imports to move)

**Input Validation Required**:
- None (this is a refactoring task, no new validation needed)

**Tasks**:

1. **Add caching class imports to cuda_simsafe.py**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     Add conditional imports for numba caching classes following the existing pattern at lines 116-164. Insert after the existing `if CUDA_SIMULATION:` block (after line 164, before `is_cuda_array` function at line 166):
     ```python
     # --- Caching infrastructure ---
     # In CUDASIM mode, provide stub classes since numba caching is unavailable
     if CUDA_SIMULATION:  # pragma: no cover - simulated
         _CacheLocator = object
         CacheImpl = object
         IndexDataCacheFile = None
         CUDACache = object
         _CACHING_AVAILABLE = False
     else:  # pragma: no cover - exercised in GPU environments
         from numba.cuda.core.caching import (
             _CacheLocator,
             CacheImpl,
             IndexDataCacheFile,
         )
         from numba.cuda.dispatcher import CUDACache
         _CACHING_AVAILABLE = True
     ```
   - Edge cases: 
     - `IndexDataCacheFile = None` because it's used as a class, not inherited from
     - `_CacheLocator = object` to allow class inheritance in stub mode
   - Integration: This block should be added after line 164, before the `is_cuda_array` function definition

2. **Update __all__ exports in cuda_simsafe.py**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     Add the new exports to the `__all__` list (currently at lines 316-347). Add these entries in alphabetical order:
     ```python
     "_CacheLocator",
     "_CACHING_AVAILABLE",
     "CacheImpl",
     "CUDACache",
     "IndexDataCacheFile",
     ```
   - Edge cases: Maintain alphabetical order in `__all__` list
   - Integration: These exports enable cubie_cache.py to import from cuda_simsafe

3. **Replace conditional imports in cubie_cache.py with imports from cuda_simsafe**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     Replace lines 22-40 (the `is_cudasim_enabled` import and conditional import block):
     
     **Current code (lines 22-40):**
     ```python
     from cubie.cuda_simsafe import is_cudasim_enabled
     from cubie.odesystems.symbolic.odefile import GENERATED_DIR

     if not is_cudasim_enabled():
         from numba.cuda.core.caching import (
             _CacheLocator,
             CacheImpl,
             IndexDataCacheFile,
         )
         from numba.cuda.dispatcher import CUDACache

         _CACHING_AVAILABLE = True
     else:
         # Stub classes for simulator mode
         _CacheLocator = object
         CacheImpl = object
         IndexDataCacheFile = None
         CUDACache = object
         _CACHING_AVAILABLE = False
     ```
     
     **New code:**
     ```python
     from cubie.cuda_simsafe import (
         _CacheLocator,
         _CACHING_AVAILABLE,
         CacheImpl,
         CUDACache,
         IndexDataCacheFile,
     )
     from cubie.odesystems.symbolic.odefile import GENERATED_DIR
     ```
   - Edge cases: Ensure import order follows PEP8 (standard library, then third-party, then local)
   - Integration: cubie_cache.py now imports from cuda_simsafe.py instead of conditionally importing

**Tests to Create**:
- None (existing tests verify the functionality)

**Tests to Run**:
- tests/test_cubie_cache.py::test_cache_locator_get_cache_path
- tests/test_cubie_cache.py::test_cache_locator_get_source_stamp
- tests/test_cubie_cache.py::test_cache_locator_get_disambiguator
- tests/test_cubie_cache.py::test_cache_locator_from_function_raises
- tests/test_cubie_cache.py::test_cache_impl_locator_property
- tests/test_cubie_cache.py::test_cache_impl_filename_base
- tests/test_cubie_cache.py::test_cache_impl_check_cachable

**Outcomes**: 
- Files Modified: 
  * src/cubie/cuda_simsafe.py (19 lines changed)
  * src/cubie/cubie_cache.py (11 lines removed, 8 lines added)
- Functions/Methods Added/Modified:
  * Added caching infrastructure conditional block in cuda_simsafe.py (lines 166-181)
  * Updated __all__ list in cuda_simsafe.py (lines 334-370, now alphabetically sorted)
  * Simplified imports in cubie_cache.py (lines 22-29)
- Implementation Summary:
  Moved CUDASIM-conditioned imports for numba caching classes (_CacheLocator, CacheImpl, IndexDataCacheFile, CUDACache, _CACHING_AVAILABLE) from cubie_cache.py to cuda_simsafe.py, following the existing pattern. The __all__ list was updated in alphabetical order. cubie_cache.py now imports these directly from cuda_simsafe.py.
- Issues Flagged: None

---

## Task Group 2: Modify ODEFile Directory Structure
**Status**: [x]
**Dependencies**: Task Group 1 (due to shared GENERATED_DIR usage)

**Required Context**:
- File: src/cubie/odesystems/symbolic/odefile.py (entire file, especially lines 24-41 for __init__ method)
- File: src/cubie/cubie_cache.py (line 111 for CUBIECacheLocator pattern reference)

**Input Validation Required**:
- None (system_name validation is already handled by existing code)

**Tasks**:

1. **Modify ODEFile.__init__ to use system-specific subdirectory**
   - File: src/cubie/odesystems/symbolic/odefile.py
   - Action: Modify
   - Details:
     Modify the `__init__` method (lines 27-40) to create files in a system-specific subdirectory:
     
     **Current code (lines 27-40):**
     ```python
     def __init__(self, system_name: str, fn_hash: int) -> None:
         """Initialise a cache file for a system definition.

         Parameters
         ----------
         system_name
             Name used when constructing the generated module filename.
         fn_hash
             Hash representing the symbolic system definition.
         """
         GENERATED_DIR.mkdir(exist_ok=True)
         self.file_path = GENERATED_DIR / f"{system_name}.py"
         self.fn_hash = fn_hash
         self._init_file(fn_hash)
     ```
     
     **New code:**
     ```python
     def __init__(self, system_name: str, fn_hash: int) -> None:
         """Initialise a cache file for a system definition.

         Parameters
         ----------
         system_name
             Name used when constructing the generated module filename.
         fn_hash
             Hash representing the symbolic system definition.
         """
         system_dir = GENERATED_DIR / system_name
         system_dir.mkdir(parents=True, exist_ok=True)
         self.file_path = system_dir / f"{system_name}.py"
         self.fn_hash = fn_hash
         self._init_file(fn_hash)
     ```
   - Edge cases:
     - Use `parents=True` to create nested directories if GENERATED_DIR doesn't exist
     - `exist_ok=True` handles cases where directory already exists
   - Integration: This aligns ODEFile with CUBIECacheLocator pattern (see cubie_cache.py line 111)

**Tests to Create**:
- None (existing tests verify ODEFile functionality and will pass with new directory structure)

**Tests to Run**:
- tests/test_cubie_cache.py (all tests - verifies cache path includes system_name)

**Outcomes**: 
- Files Modified: 
  * src/cubie/odesystems/symbolic/odefile.py (3 lines changed)
- Functions/Methods Added/Modified:
  * ODEFile.__init__() in odefile.py
- Implementation Summary:
  Modified ODEFile.__init__ to create files in system-specific subdirectory (`GENERATED_DIR / system_name / f"{system_name}.py"`) instead of directly in GENERATED_DIR. Uses `parents=True` in mkdir to create nested directories if GENERATED_DIR doesn't exist. This aligns ODEFile with CUBIECacheLocator pattern which uses `GENERATED_DIR / system_name / "CUDA_cache"`.
- Issues Flagged: None

---

## Task Group 3: Add Tests for config_hash and _iter_child_factories
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/CUDAFactory.py (lines 522-553 for config_hash and _iter_child_factories implementations)
- File: tests/test_CUDAFactory.py (entire file for existing test patterns and fixtures)

**Input Validation Required**:
- None (these are test additions)

**Tasks**:

1. **Add test_config_hash_no_children**
   - File: tests/test_CUDAFactory.py
   - Action: Modify (append test)
   - Details:
     Add test at end of file to verify config_hash equals compile_settings.values_hash when factory has no child factories:
     ```python
     def test_config_hash_no_children():
         """Test config_hash returns own values_hash when no child factories."""
         from cubie.CUDAFactory import _CubieConfigBase

         @attrs.define
         class SimpleConfig(_CubieConfigBase):
             value1: int = 10
             value2: str = "test"

         class SimpleFactory(CUDAFactory):
             def __init__(self):
                 super().__init__()

             def build(self):
                 return testCache(device_function=lambda: 1.0)

         factory = SimpleFactory()
         factory.setup_compile_settings(SimpleConfig())

         # With no children, config_hash should equal compile_settings.values_hash
         assert factory.config_hash == factory.compile_settings.values_hash
         assert len(factory.config_hash) == 64
     ```
   - Edge cases: Factory with no CUDAFactory attributes
   - Integration: Uses existing testCache and CUDAFactory from the test file

2. **Add test_config_hash_with_children**
   - File: tests/test_CUDAFactory.py
   - Action: Modify (append test)
   - Details:
     Add test to verify config_hash combines hashes from child factories:
     ```python
     def test_config_hash_with_children():
         """Test config_hash combines hashes from child factories."""
         from cubie.CUDAFactory import _CubieConfigBase

         @attrs.define
         class SimpleConfig(_CubieConfigBase):
             value1: int = 10

         class ChildFactory(CUDAFactory):
             def __init__(self):
                 super().__init__()

             def build(self):
                 return testCache(device_function=lambda: 2.0)

         class ParentFactory(CUDAFactory):
             def __init__(self):
                 super().__init__()
                 self._child = ChildFactory()
                 self._child.setup_compile_settings(SimpleConfig(value1=20))

             def build(self):
                 return testCache(device_function=lambda: 1.0)

         parent = ParentFactory()
         parent.setup_compile_settings(SimpleConfig(value1=10))

         # Hash should differ from own settings hash when children exist
         own_hash = parent.compile_settings.values_hash
         combined_hash = parent.config_hash

         assert combined_hash != own_hash
         assert len(combined_hash) == 64

         # Hash should be deterministic
         assert parent.config_hash == combined_hash
     ```
   - Edge cases: Parent with child factory attributes
   - Integration: Uses existing testCache and CUDAFactory from the test file

3. **Add test_iter_child_factories_no_children**
   - File: tests/test_CUDAFactory.py
   - Action: Modify (append test)
   - Details:
     Add test to verify _iter_child_factories yields nothing when no children:
     ```python
     def test_iter_child_factories_no_children():
         """Test _iter_child_factories yields nothing when no children."""
         from cubie.CUDAFactory import _CubieConfigBase

         @attrs.define
         class SimpleConfig(_CubieConfigBase):
             value1: int = 10

         class SimpleFactory(CUDAFactory):
             def __init__(self):
                 super().__init__()
                 self._non_factory_attr = "not a factory"
                 self._numeric_attr = 42

             def build(self):
                 return testCache(device_function=lambda: 1.0)

         factory = SimpleFactory()
         factory.setup_compile_settings(SimpleConfig())

         children = list(factory._iter_child_factories())
         assert children == []
     ```
   - Edge cases: Factory with non-CUDAFactory attributes
   - Integration: Uses existing testCache and CUDAFactory from the test file

4. **Add test_iter_child_factories_with_children**
   - File: tests/test_CUDAFactory.py
   - Action: Modify (append test)
   - Details:
     Add test to verify _iter_child_factories yields child factories in alphabetical order:
     ```python
     def test_iter_child_factories_with_children():
         """Test _iter_child_factories yields children in alphabetical order."""
         from cubie.CUDAFactory import _CubieConfigBase

         @attrs.define
         class SimpleConfig(_CubieConfigBase):
             value1: int = 10

         class ChildFactory(CUDAFactory):
             def __init__(self, name):
                 super().__init__()
                 self.name = name

             def build(self):
                 return testCache(device_function=lambda: 1.0)

         class ParentFactory(CUDAFactory):
             def __init__(self):
                 super().__init__()
                 # Attributes in non-alphabetical order
                 self._zebra_child = ChildFactory("zebra")
                 self._alpha_child = ChildFactory("alpha")
                 self._middle_child = ChildFactory("middle")
                 # Set up settings for children
                 for child in [self._zebra_child, self._alpha_child,
                               self._middle_child]:
                     child.setup_compile_settings(SimpleConfig())

             def build(self):
                 return testCache(device_function=lambda: 1.0)

         parent = ParentFactory()
         parent.setup_compile_settings(SimpleConfig())

         children = list(parent._iter_child_factories())

         # Should yield 3 children
         assert len(children) == 3

         # Should be in alphabetical order by attribute name
         names = [c.name for c in children]
         assert names == ["alpha", "middle", "zebra"]
     ```
   - Edge cases: Multiple children, alphabetical ordering
   - Integration: Uses existing testCache and CUDAFactory from the test file

5. **Add test_iter_child_factories_uniqueness**
   - File: tests/test_CUDAFactory.py
   - Action: Modify (append test)
   - Details:
     Add test to verify same child referenced by multiple attributes is yielded once:
     ```python
     def test_iter_child_factories_uniqueness():
         """Test _iter_child_factories yields each child only once."""
         from cubie.CUDAFactory import _CubieConfigBase

         @attrs.define
         class SimpleConfig(_CubieConfigBase):
             value1: int = 10

         class ChildFactory(CUDAFactory):
             def __init__(self):
                 super().__init__()

             def build(self):
                 return testCache(device_function=lambda: 1.0)

         class ParentFactory(CUDAFactory):
             def __init__(self):
                 super().__init__()
                 # Same child referenced by multiple attributes
                 shared_child = ChildFactory()
                 shared_child.setup_compile_settings(SimpleConfig())
                 self._child_a = shared_child
                 self._child_b = shared_child
                 self._child_c = shared_child

             def build(self):
                 return testCache(device_function=lambda: 1.0)

         parent = ParentFactory()
         parent.setup_compile_settings(SimpleConfig())

         children = list(parent._iter_child_factories())

         # Same child referenced 3 times should yield only once
         assert len(children) == 1
     ```
   - Edge cases: Same object referenced multiple times
   - Integration: Uses existing testCache and CUDAFactory from the test file

**Tests to Create**:
- tests/test_CUDAFactory.py::test_config_hash_no_children
- tests/test_CUDAFactory.py::test_config_hash_with_children
- tests/test_CUDAFactory.py::test_iter_child_factories_no_children
- tests/test_CUDAFactory.py::test_iter_child_factories_with_children
- tests/test_CUDAFactory.py::test_iter_child_factories_uniqueness

**Tests to Run**:
- tests/test_CUDAFactory.py::test_config_hash_no_children
- tests/test_CUDAFactory.py::test_config_hash_with_children
- tests/test_CUDAFactory.py::test_iter_child_factories_no_children
- tests/test_CUDAFactory.py::test_iter_child_factories_with_children
- tests/test_CUDAFactory.py::test_iter_child_factories_uniqueness

**Outcomes**: 
- Files Modified: 
  * tests/test_CUDAFactory.py (148 lines added)
- Functions/Methods Added/Modified:
  * test_config_hash_no_children() in test_CUDAFactory.py
  * test_config_hash_with_children() in test_CUDAFactory.py
  * test_iter_child_factories_no_children() in test_CUDAFactory.py
  * test_iter_child_factories_with_children() in test_CUDAFactory.py
  * test_iter_child_factories_uniqueness() in test_CUDAFactory.py
- Implementation Summary:
  Added 5 new tests to verify config_hash and _iter_child_factories functionality. Tests cover: config_hash without children (returns compile_settings.values_hash), config_hash with children (combines hashes), _iter_child_factories with no children (yields nothing), _iter_child_factories with multiple children (alphabetical order), and _iter_child_factories uniqueness (same child referenced multiple times yields once).
- Issues Flagged: None

---

## Task Group 4: Update test_cubie_cache.py for config_hash Parameter
**Status**: [x]
**Dependencies**: Task Group 1 (imports must work correctly first)

**Required Context**:
- File: tests/test_cubie_cache.py (entire file, focus on lines 130-170 for CUBIECache tests)
- File: src/cubie/cubie_cache.py (lines 283-325 for CUBIECache.__init__ signature)

**Input Validation Required**:
- None (these are test updates)

**Tasks**:

1. **Update test_cubie_cache_init to pass config_hash**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     Update test at lines 130-137 to pass config_hash parameter:
     
     **Current code (lines 130-137):**
     ```python
     @pytest.mark.nocudasim
     def test_cubie_cache_init():
         """Verify CUBIECache initializes with system info."""
         settings = MockCompileSettings()
         cache = CUBIECache(system_name="test_system", system_hash="abc123")
         assert cache._system_name == "test_system"
         assert cache._system_hash == "abc123"
         assert cache._name == "CUBIECache(test_system)"
     ```
     
     **New code:**
     ```python
     @pytest.mark.nocudasim
     def test_cubie_cache_init():
         """Verify CUBIECache initializes with system info."""
         cache = CUBIECache(
             system_name="test_system",
             system_hash="abc123",
             config_hash="def456789012345678901234567890123456789012345678901234567890abcd",
         )
         assert cache._system_name == "test_system"
         assert cache._system_hash == "abc123"
         assert cache._name == "CUBIECache(test_system)"
     ```
   - Edge cases: config_hash should be a 64-character hex string
   - Integration: Aligns with CUBIECache constructor signature

2. **Update test_cubie_cache_index_key to pass config_hash**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     Update test at lines 140-160 to pass config_hash parameter:
     
     **Current code (lines 140-160):**
     ```python
     @pytest.mark.nocudasim
     def test_cubie_cache_index_key():
         """Verify _index_key includes system and settings hashes."""
         settings = MockCompileSettings()
         cache = CUBIECache(system_name="test_system", system_hash="abc123")

         # Create a mock codegen object
         class MockCodegen:
             def magic_tuple(self):
                 return ("magic", "tuple")

         sig = ("float32", "float32")
         codegen = MockCodegen()

         key = cache._index_key(sig, codegen)

         # Key should be tuple of (sig, magic_tuple, system_hash, settings_hash)
         assert len(key) == 4
         assert key[0] == sig
         assert key[1] == ("magic", "tuple")
         assert key[2] == "abc123"
     ```
     
     **New code:**
     ```python
     @pytest.mark.nocudasim
     def test_cubie_cache_index_key():
         """Verify _index_key includes system and settings hashes."""
         config_hash = "def456789012345678901234567890123456789012345678901234567890abcd"
         cache = CUBIECache(
             system_name="test_system",
             system_hash="abc123",
             config_hash=config_hash,
         )

         # Create a mock codegen object
         class MockCodegen:
             def magic_tuple(self):
                 return ("magic", "tuple")

         sig = ("float32", "float32")
         codegen = MockCodegen()

         key = cache._index_key(sig, codegen)

         # Key should be tuple of (sig, magic_tuple, system_hash, config_hash)
         assert len(key) == 4
         assert key[0] == sig
         assert key[1] == ("magic", "tuple")
         assert key[2] == "abc123"
         assert key[3] == config_hash
     ```
   - Edge cases: Verify config_hash is included in the index key
   - Integration: Tests that config_hash is properly used in cache key generation

3. **Update test_cubie_cache_path to pass config_hash**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     Update test at lines 163-169 to pass config_hash parameter:
     
     **Current code (lines 163-169):**
     ```python
     @pytest.mark.nocudasim
     def test_cubie_cache_path():
         """Verify cache_path property returns expected path."""
         settings = MockCompileSettings()
         cache = CUBIECache(system_name="test_system", system_hash="abc123")
         assert "test_system" in cache.cache_path
         assert "cache" in cache.cache_path
     ```
     
     **New code:**
     ```python
     @pytest.mark.nocudasim
     def test_cubie_cache_path():
         """Verify cache_path property returns expected path."""
         cache = CUBIECache(
             system_name="test_system",
             system_hash="abc123",
             config_hash="def456789012345678901234567890123456789012345678901234567890abcd",
         )
         assert "test_system" in cache.cache_path
         assert "cache" in cache.cache_path
     ```
   - Edge cases: None (straightforward parameter addition)
   - Integration: Aligns with CUBIECache constructor signature

4. **Remove unused settings variable from tests**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     The `settings = MockCompileSettings()` lines in the three tests above are unused and can be removed. This is handled by the replacement code in tasks 1-3 above.
   - Edge cases: None
   - Integration: Cleanup of unused code

**Tests to Create**:
- None (updating existing tests)

**Tests to Run**:
- tests/test_cubie_cache.py::test_cubie_cache_init
- tests/test_cubie_cache.py::test_cubie_cache_index_key
- tests/test_cubie_cache.py::test_cubie_cache_path

**Outcomes**: 
- Files Modified: 
  * tests/test_cubie_cache.py (15 lines changed across 3 test functions)
- Functions/Methods Added/Modified:
  * test_cubie_cache_init() - removed unused settings variable, added config_hash parameter
  * test_cubie_cache_index_key() - removed unused settings variable, added config_hash parameter, added assertion to verify config_hash is included in index key
  * test_cubie_cache_path() - removed unused settings variable, added config_hash parameter
- Implementation Summary:
  Updated three CUBIECache tests to pass the config_hash parameter instead of creating unused MockCompileSettings instances. Each test now provides a 64-character hex string as the config_hash. The test_cubie_cache_index_key test also includes a new assertion to verify that the config_hash is properly included as the 4th element of the index key tuple.
- Issues Flagged: None. Note: These tests are marked @pytest.mark.nocudasim and will not run in CUDASIM mode.

---

## Summary

| Task Group | Description | Dependencies | Estimated Complexity |
|------------|-------------|--------------|---------------------|
| 1 | Move CUDASIM imports to cuda_simsafe.py | None | Low |
| 2 | Modify ODEFile directory structure | Group 1 | Low |
| 3 | Add CUDAFactory method tests | None | Medium |
| 4 | Update cache config tests | Group 1 | Low |

### Dependency Chain
```
Task Group 1 ─────┬────> Task Group 2
                  │
                  └────> Task Group 4

Task Group 3 (independent)
```

### Tests Summary
- **Tests to Create**: 5 new tests in test_CUDAFactory.py
- **Tests to Update**: 3 tests in test_cubie_cache.py
- **Total Test Files Affected**: 2
