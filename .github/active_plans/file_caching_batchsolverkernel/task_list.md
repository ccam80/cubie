# Implementation Task List
# Feature: File-Based Caching for BatchSolverKernel
# Plan Reference: .github/active_plans/file_caching_batchsolverkernel/agent_plan.md

## Task Group 1: Hash Compile Settings Utility
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/CUDAFactory.py (entire file - understanding attrs pattern)
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file - compile
  settings structure)
- File: src/cubie/odesystems/symbolic/sym_utils.py (lines 141-209 -
  hash_system_definition reference)
- File: src/cubie/outputhandling/output_config.py (entire file -
  OutputCompileFlags structure)
- File: .github/active_plans/explore-caching/core/caching.py (lines 326-350 -
  _index_key pattern)

**Input Validation Required**:
- obj: Must be an attrs class instance (check via `attrs.has(type(obj))`)
- Skip fields with `eq=False` metadata (callables are not hashable)
- Handle None values by including "None" literal in hash input
- Handle numpy arrays via `array.tobytes()` for deterministic hashing

**Tasks**:
1. **Create cubie_cache.py module**
   - File: src/cubie/cubie_cache.py
   - Action: Create
   - Details:
     ```python
     """File-based caching infrastructure for CuBIE compiled kernels.
     
     Provides cache classes that persist compiled CUDA kernels to disk,
     enabling faster startup on subsequent runs with identical settings.
     """
     
     import hashlib
     from pathlib import Path
     from typing import Any, Optional
     
     from attrs import fields, has
     from numpy import ndarray
     ```
   - Edge cases: File must not exist before creation
   - Integration: New standalone module in src/cubie/

2. **Implement hash_compile_settings function**
   - File: src/cubie/cubie_cache.py
   - Action: Modify (append)
   - Details:
     ```python
     def hash_compile_settings(obj: Any) -> str:
         """Compute a stable hash from attrs compile settings.
         
         Traverses attrs class fields and computes a deterministic hash
         of all field values suitable for cache key construction.
         
         Parameters
         ----------
         obj
             An attrs class instance containing compile settings.
             
         Returns
         -------
         str
             SHA256 hash string of the serialized settings.
             
         Notes
         -----
         Fields marked with ``eq=False`` (typically callables) are skipped.
         Numpy arrays are hashed via ``tobytes()`` for determinism.
         Nested attrs classes are recursively processed.
         """
         # Implementation logic:
         # 1. Validate obj is attrs class via has(type(obj))
         # 2. Build ordered list of (field_name, serialized_value) pairs
         # 3. For each field in attrs.fields(type(obj)):
         #    a. Skip if field.eq is False (callables, device functions)
         #    b. Get field value via getattr
         #    c. Serialize based on type:
         #       - None: "None"
         #       - ndarray: base64 of tobytes() or hex digest
         #       - nested attrs: recursive call
         #       - primitives: str(value)
         # 4. Concatenate with separator: "field_name=value|..."
         # 5. Return SHA256 hexdigest
     ```
   - Edge cases:
     - Empty attrs class (return hash of empty string)
     - Deeply nested attrs classes (recursive handling)
     - Large numpy arrays (use hash of tobytes, not raw bytes)
     - Callable fields (skip via eq=False check)
   - Integration: Used by CUBIECache._index_key()

**Tests to Create**:
- Test file: tests/test_cubie_cache.py
- Test function: test_hash_compile_settings_basic
- Description: Verify hash is produced for simple attrs class
- Test function: test_hash_compile_settings_with_arrays
- Description: Verify numpy arrays produce deterministic hashes
- Test function: test_hash_compile_settings_skips_eq_false
- Description: Verify fields with eq=False are excluded from hash
- Test function: test_hash_compile_settings_nested_attrs
- Description: Verify nested attrs classes are recursively hashed
- Test function: test_hash_compile_settings_changes_on_value_change
- Description: Verify hash changes when any field value changes

**Tests to Run**:
- tests/test_cubie_cache.py::test_hash_compile_settings_basic
- tests/test_cubie_cache.py::test_hash_compile_settings_with_arrays
- tests/test_cubie_cache.py::test_hash_compile_settings_skips_eq_false
- tests/test_cubie_cache.py::test_hash_compile_settings_nested_attrs
- tests/test_cubie_cache.py::test_hash_compile_settings_changes_on_value_change

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: CUBIECacheLocator Implementation
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/cubie_cache.py (entire file - add to existing)
- File: src/cubie/odesystems/symbolic/odefile.py (lines 1-20 - GENERATED_DIR)
- File: .github/active_plans/explore-caching/core/caching.py (lines 353-439 -
  _CacheLocator base class and mixins)

**Input Validation Required**:
- system_name: Must be a non-empty string
- system_hash: Must be a string (hash of system definition)

**Tasks**:
1. **Import _CacheLocator from numba.cuda.core.caching**
   - File: src/cubie/cubie_cache.py
   - Action: Modify (add import)
   - Details:
     ```python
     from numba.cuda.core.caching import _CacheLocator
     ```
   - Edge cases: Import may fail if numba-cuda not installed
   - Integration: Required base class

2. **Import GENERATED_DIR from odefile**
   - File: src/cubie/cubie_cache.py
   - Action: Modify (add import)
   - Details:
     ```python
     from cubie.odesystems.symbolic.odefile import GENERATED_DIR
     ```
   - Edge cases: None
   - Integration: Cache path location

3. **Implement CUBIECacheLocator class**
   - File: src/cubie/cubie_cache.py
   - Action: Modify (append)
   - Details:
     ```python
     class CUBIECacheLocator(_CacheLocator):
         """Locate cache files in CuBIE's generated directory structure.
         
         Directs cache files to ``generated/<system_name>/cache/`` instead
         of the default ``__pycache__`` location used by numba.
         
         Parameters
         ----------
         system_name
             Name of the ODE system for directory organization.
         system_hash
             Hash representing the ODE system definition for freshness.
         compile_settings_hash
             Hash of compile settings for disambiguation.
         """
         
         def __init__(
             self,
             system_name: str,
             system_hash: str,
             compile_settings_hash: str,
         ) -> None:
             # Implementation:
             # 1. Store system_name, system_hash, compile_settings_hash
             # 2. Compute cache_path as GENERATED_DIR / system_name / "cache"
             pass
         
         def get_cache_path(self) -> str:
             """Return the directory where cache files are stored.
             
             Returns
             -------
             str
                 Absolute path to the cache directory.
             """
             # Return str(self._cache_path)
             pass
         
         def get_source_stamp(self) -> str:
             """Return a stamp representing source freshness.
             
             Returns
             -------
             str
                 The system hash acts as the freshness indicator.
             """
             # Return self._system_hash
             pass
         
         def get_disambiguator(self) -> str:
             """Return a string to disambiguate similar functions.
             
             Returns
             -------
             str
                 First 16 characters of compile_settings_hash.
             """
             # Return self._compile_settings_hash[:16]
             pass
         
         @classmethod
         def from_function(cls, py_func, py_file):
             """Not used - CuBIE creates locators directly.
             
             Raises
             ------
             NotImplementedError
                 This locator does not use the from_function pattern.
             """
             raise NotImplementedError(
                 "CUBIECacheLocator requires explicit system info"
             )
     ```
   - Edge cases:
     - system_name with invalid path characters (sanitize)
     - Cache directory does not exist (ensure_cache_path handles)
   - Integration: Used by CUBIECacheImpl

**Tests to Create**:
- Test file: tests/test_cubie_cache.py
- Test function: test_cache_locator_get_cache_path
- Description: Verify cache path is in generated/<system_name>/cache/
- Test function: test_cache_locator_get_source_stamp
- Description: Verify source stamp returns system_hash
- Test function: test_cache_locator_get_disambiguator
- Description: Verify disambiguator returns truncated settings hash
- Test function: test_cache_locator_from_function_raises
- Description: Verify from_function raises NotImplementedError

**Tests to Run**:
- tests/test_cubie_cache.py::test_cache_locator_get_cache_path
- tests/test_cubie_cache.py::test_cache_locator_get_source_stamp
- tests/test_cubie_cache.py::test_cache_locator_get_disambiguator
- tests/test_cubie_cache.py::test_cache_locator_from_function_raises

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: CUBIECacheImpl Implementation
**Status**: [ ]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/cubie_cache.py (entire file - add to existing)
- File: .github/active_plans/explore-caching/core/caching.py (lines 612-690 -
  CacheImpl base class)
- File: .github/active_plans/explore-caching/dispatcher.py (lines 20-48 -
  CUDACacheImpl reference)

**Input Validation Required**:
- kernel: Must be a numba CUDA kernel with _reduce_states method
- payload: Must be a dict from _reduce_states output

**Tasks**:
1. **Import CacheImpl from numba.cuda.core.caching**
   - File: src/cubie/cubie_cache.py
   - Action: Modify (add import)
   - Details:
     ```python
     from numba.cuda.core.caching import CacheImpl
     ```
   - Edge cases: Import may fail if numba-cuda not installed
   - Integration: Required base class

2. **Implement CUBIECacheImpl class**
   - File: src/cubie/cubie_cache.py
   - Action: Modify (append)
   - Details:
     ```python
     class CUBIECacheImpl(CacheImpl):
         """Serialization logic for CuBIE compiled kernels.
         
         Delegates actual serialization to numba's built-in _Kernel methods
         while using CuBIE-specific cache locator for file paths.
         
         Parameters
         ----------
         system_name
             Name of the ODE system.
         system_hash
             Hash representing the ODE system definition.
         compile_settings_hash
             Hash of compile settings for cache key.
         """
         
         # Override locator classes to use only CuBIE locator
         _locator_classes = []  # Disable default locators
         
         def __init__(
             self,
             system_name: str,
             system_hash: str,
             compile_settings_hash: str,
         ) -> None:
             # Implementation:
             # 1. Create CUBIECacheLocator directly (not via from_function)
             # 2. Store as self._locator
             # 3. Build filename_base from system_name + disambiguator
             self._locator = CUBIECacheLocator(
                 system_name, system_hash, compile_settings_hash
             )
             disambiguator = self._locator.get_disambiguator()
             self._filename_base = f"{system_name}-{disambiguator}"
         
         @property
         def locator(self):
             """Return the cache locator instance."""
             return self._locator
         
         @property
         def filename_base(self) -> str:
             """Return base filename for cache files."""
             return self._filename_base
         
         def reduce(self, kernel) -> dict:
             """Reduce kernel to serializable form.
             
             Parameters
             ----------
             kernel
                 Compiled CUDA kernel with _reduce_states method.
                 
             Returns
             -------
             dict
                 Serializable state dictionary.
             """
             # Delegate to numba's _Kernel._reduce_states()
             return kernel._reduce_states()
         
         def rebuild(self, target_context, payload: dict):
             """Rebuild kernel from cached payload.
             
             Parameters
             ----------
             target_context
                 CUDA target context for kernel reconstruction.
             payload
                 Serialized kernel state from reduce().
                 
             Returns
             -------
             _Kernel
                 Reconstructed CUDA kernel.
             """
             # Import here to avoid circular dependency
             from numba.cuda.dispatcher import _Kernel
             return _Kernel._rebuild(**payload)
         
         def check_cachable(self, data) -> bool:
             """Check if the data is cachable.
             
             CUDA kernels are always cachable.
             
             Returns
             -------
             bool
                 Always True for CUDA kernels.
             """
             return True
     ```
   - Edge cases:
     - kernel without _reduce_states (wrong type passed)
     - _Kernel._rebuild failure (corrupted cache)
   - Integration: Used by CUBIECache

**Tests to Create**:
- Test file: tests/test_cubie_cache.py
- Test function: test_cache_impl_locator_property
- Description: Verify locator property returns CUBIECacheLocator
- Test function: test_cache_impl_filename_base
- Description: Verify filename_base format
- Test function: test_cache_impl_check_cachable
- Description: Verify check_cachable returns True

**Tests to Run**:
- tests/test_cubie_cache.py::test_cache_impl_locator_property
- tests/test_cubie_cache.py::test_cache_impl_filename_base
- tests/test_cubie_cache.py::test_cache_impl_check_cachable

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: CUBIECache Implementation
**Status**: [ ]
**Dependencies**: Task Group 3

**Required Context**:
- File: src/cubie/cubie_cache.py (entire file - add to existing)
- File: .github/active_plans/explore-caching/core/caching.py (lines 219-351 -
  Cache base class)
- File: .github/active_plans/explore-caching/dispatcher.py (lines 50-78 -
  CUDACache reference)
- File: src/cubie/odesystems/symbolic/sym_utils.py (lines 141-209 -
  hash_system_definition)

**Input Validation Required**:
- system_name: Must be non-empty string
- system_hash: Must be non-empty string
- compile_settings: Must be attrs class instance

**Tasks**:
1. **Import Cache and IndexDataCacheFile from numba.cuda.core.caching**
   - File: src/cubie/cubie_cache.py
   - Action: Modify (add import)
   - Details:
     ```python
     from numba.cuda.core.caching import Cache, IndexDataCacheFile
     ```
   - Edge cases: Import may fail if numba-cuda not installed
   - Integration: Required base classes

2. **Implement CUBIECache class**
   - File: src/cubie/cubie_cache.py
   - Action: Modify (append)
   - Details:
     ```python
     class CUBIECache(Cache):
         """File-based cache for CuBIE compiled kernels.
         
         Coordinates loading and saving of cached kernels, incorporating
         ODE system hash and compile settings hash into cache keys.
         
         Parameters
         ----------
         system_name
             Name of the ODE system.
         system_hash
             Hash representing the ODE system definition.
         compile_settings
             Attrs class instance of compile settings.
             
         Notes
         -----
         Unlike the base Cache class, this does not use py_func for
         initialization. Instead, system info is passed directly.
         """
         
         _impl_class = CUBIECacheImpl
         
         def __init__(
             self,
             system_name: str,
             system_hash: str,
             compile_settings: Any,
         ) -> None:
             # Implementation:
             # 1. Store system info
             # 2. Compute compile_settings_hash via hash_compile_settings
             # 3. Create CUBIECacheImpl directly (bypass from_function)
             # 4. Get cache_path from impl.locator
             # 5. Create IndexDataCacheFile
             # 6. Enable caching
             self._system_name = system_name
             self._system_hash = system_hash
             self._compile_settings_hash = hash_compile_settings(
                 compile_settings
             )
             self._name = f"CUBIECache({system_name})"
             
             self._impl = CUBIECacheImpl(
                 system_name,
                 system_hash,
                 self._compile_settings_hash,
             )
             self._cache_path = self._impl.locator.get_cache_path()
             
             source_stamp = self._impl.locator.get_source_stamp()
             filename_base = self._impl.filename_base
             self._cache_file = IndexDataCacheFile(
                 cache_path=self._cache_path,
                 filename_base=filename_base,
                 source_stamp=source_stamp,
             )
             self.enable()
         
         def _index_key(self, sig, codegen):
             """Compute cache key including CuBIE-specific hashes.
             
             Parameters
             ----------
             sig
                 Function signature tuple.
             codegen
                 CUDA codegen object with magic_tuple().
                 
             Returns
             -------
             tuple
                 Composite cache key.
             """
             # Override to include system_hash and compile_settings_hash
             return (
                 sig,
                 codegen.magic_tuple(),
                 self._system_hash,
                 self._compile_settings_hash,
             )
         
         def load_overload(self, sig, target_context):
             """Load cached kernel with CUDA context handling.
             
             Parameters
             ----------
             sig
                 Function signature.
             target_context
                 CUDA target context.
                 
             Returns
             -------
             Optional[_Kernel]
                 Cached kernel or None if not found.
             """
             from numba.cuda import utils
             with utils.numba_target_override():
                 return super().load_overload(sig, target_context)
     ```
   - Edge cases:
     - Empty compile_settings (valid, produces consistent hash)
     - Cache directory creation failure (handled by locator)
     - Corrupted cache files (handled by load returning None)
   - Integration: Attached to CUDADispatcher in BatchSolverKernel

**Tests to Create**:
- Test file: tests/test_cubie_cache.py
- Test function: test_cubie_cache_init
- Description: Verify CUBIECache initializes with system info
- Test function: test_cubie_cache_index_key
- Description: Verify _index_key includes system and settings hashes
- Test function: test_cubie_cache_path
- Description: Verify cache_path property returns expected path

**Tests to Run**:
- tests/test_cubie_cache.py::test_cubie_cache_init
- tests/test_cubie_cache.py::test_cubie_cache_index_key
- tests/test_cubie_cache.py::test_cubie_cache_path

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: BatchSolverKernel Integration
**Status**: [ ]
**Dependencies**: Task Group 4

**Required Context**:
- File: src/cubie/cubie_cache.py (entire file)
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 554-706 -
  build_kernel method)
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file)
- File: src/cubie/cuda_simsafe.py (entire file - is_cudasim_enabled)
- File: src/cubie/odesystems/symbolic/sym_utils.py (lines 141-209 -
  hash_system_definition)

**Input Validation Required**:
- caching_enabled: Must be bool (default True)
- System must have hash available (SymbolicODE systems do)

**Tasks**:
1. **Add caching_enabled to BatchSolverConfig**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify
   - Details:
     ```python
     # Add new field after compile_flags:
     caching_enabled: bool = attrs.field(
         default=True,
         validator=val.instance_of(bool),
     )
     ```
   - Edge cases: None
   - Integration: Controls file caching behavior

2. **Import caching utilities in BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify (add imports)
   - Details:
     ```python
     from cubie.cubie_cache import CUBIECache
     from cubie.odesystems.symbolic.sym_utils import hash_system_definition
     ```
   - Edge cases: None
   - Integration: Required for cache attachment

3. **Add cache attachment in build_kernel method**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify (after integration_kernel creation, before return)
   - Details:
     ```python
     # After line 704 (before return integration_kernel):
     # Attach file-based caching if enabled and not in simulator mode
     if (self.compile_settings.caching_enabled
             and not is_cudasim_enabled()):
         try:
             system = self.single_integrator.system
             system_name = getattr(system, 'name', 'anonymous')
             # Use system hash if available, else hash empty string
             if hasattr(system, 'system_hash'):
                 system_hash = system.system_hash
             else:
                 # Fallback for non-symbolic systems
                 system_hash = hashlib.sha256(b'').hexdigest()
             cache = CUBIECache(
                 system_name=system_name,
                 system_hash=system_hash,
                 compile_settings=self.compile_settings,
             )
             integration_kernel._cache = cache
         except Exception:
             # Caching is optional; fall back to no caching on errors
             pass
     
     return integration_kernel
     ```
   - Edge cases:
     - System without name attribute (use 'anonymous')
     - System without system_hash (use empty hash)
     - CUDASIM mode (skip caching entirely)
     - Cache initialization failure (silently continue)
   - Integration: Enables file caching for compiled kernels

4. **Add hashlib import to BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify (add import)
   - Details:
     ```python
     import hashlib
     ```
   - Edge cases: None
   - Integration: Required for fallback hash

5. **Update BatchSolverKernel.__init__ to pass caching_enabled**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify (update initial_config creation)
   - Details:
     ```python
     # In __init__, update initial_config to include caching_enabled
     # (default True is fine from BatchSolverConfig)
     # No explicit change needed if default is acceptable
     ```
   - Edge cases: None
   - Integration: Allows runtime control of caching

**Tests to Create**:
- Test file: tests/test_cubie_cache.py
- Test function: test_batch_solver_kernel_attaches_cache
- Description: Verify kernel has _cache attribute when caching enabled
- Test function: test_batch_solver_kernel_no_cache_in_cudasim
- Description: Verify no cache attached in CUDASIM mode
- Test function: test_batch_solver_kernel_cache_disabled
- Description: Verify no cache when caching_enabled=False

**Tests to Run**:
- tests/test_cubie_cache.py::test_batch_solver_kernel_attaches_cache
- tests/test_cubie_cache.py::test_batch_solver_kernel_no_cache_in_cudasim
- tests/test_cubie_cache.py::test_batch_solver_kernel_cache_disabled

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Task Group 6: Test Suite Implementation
**Status**: [ ]
**Dependencies**: Task Group 5

**Required Context**:
- File: src/cubie/cubie_cache.py (entire file)
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file)
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file)
- File: tests/conftest.py (fixture patterns)
- File: tests/batchsolving/conftest.py (solver fixtures)
- File: tests/batchsolving/test_SolverKernel.py (test patterns)

**Input Validation Required**:
- None (testing code)

**Tasks**:
1. **Create test_cubie_cache.py test file**
   - File: tests/test_cubie_cache.py
   - Action: Create
   - Details:
     ```python
     """Tests for cubie_cache module."""
     
     import pytest
     from numpy import array, float32
     
     from attrs import define, field
     
     from cubie.cubie_cache import (
         hash_compile_settings,
         CUBIECacheLocator,
         CUBIECacheImpl,
         CUBIECache,
     )
     from cubie.odesystems.symbolic.odefile import GENERATED_DIR
     
     
     # --- Fixtures ---
     
     @define
     class MockCompileSettings:
         """Simple attrs class for testing hash_compile_settings."""
         precision: type = float32
         value: int = 42
     
     
     @define
     class MockSettingsWithCallable:
         """Attrs class with eq=False field for testing."""
         precision: type = float32
         callback: object = field(default=None, eq=False)
     
     
     @define
     class MockNestedSettings:
         """Attrs class with nested attrs for testing."""
         precision: type = float32
         nested: MockCompileSettings = field(
             factory=MockCompileSettings
         )
     
     
     @define
     class MockSettingsWithArray:
         """Attrs class with numpy array for testing."""
         precision: type = float32
         data: object = field(factory=lambda: array([1.0, 2.0, 3.0]))
     
     
     # --- hash_compile_settings tests ---
     
     def test_hash_compile_settings_basic():
         """Verify hash is produced for simple attrs class."""
         settings = MockCompileSettings()
         result = hash_compile_settings(settings)
         assert isinstance(result, str)
         assert len(result) == 64  # SHA256 hex digest length
     
     
     def test_hash_compile_settings_with_arrays():
         """Verify numpy arrays produce deterministic hashes."""
         settings = MockSettingsWithArray()
         hash1 = hash_compile_settings(settings)
         hash2 = hash_compile_settings(settings)
         assert hash1 == hash2
     
     
     def test_hash_compile_settings_skips_eq_false():
         """Verify fields with eq=False are excluded from hash."""
         settings1 = MockSettingsWithCallable(callback=lambda: 1)
         settings2 = MockSettingsWithCallable(callback=lambda: 2)
         # Hashes should be identical despite different callbacks
         assert hash_compile_settings(settings1) == hash_compile_settings(
             settings2
         )
     
     
     def test_hash_compile_settings_nested_attrs():
         """Verify nested attrs classes are recursively hashed."""
         settings = MockNestedSettings()
         result = hash_compile_settings(settings)
         assert isinstance(result, str)
         assert len(result) == 64
     
     
     def test_hash_compile_settings_changes_on_value_change():
         """Verify hash changes when any field value changes."""
         settings1 = MockCompileSettings(value=1)
         settings2 = MockCompileSettings(value=2)
         assert hash_compile_settings(settings1) != hash_compile_settings(
             settings2
         )
     
     
     # --- CUBIECacheLocator tests ---
     
     def test_cache_locator_get_cache_path():
         """Verify cache path is in generated/<system_name>/cache/."""
         locator = CUBIECacheLocator(
             system_name="test_system",
             system_hash="abc123",
             compile_settings_hash="def456",
         )
         path = locator.get_cache_path()
         assert "test_system" in path
         assert "cache" in path
     
     
     def test_cache_locator_get_source_stamp():
         """Verify source stamp returns system_hash."""
         locator = CUBIECacheLocator(
             system_name="test_system",
             system_hash="abc123",
             compile_settings_hash="def456",
         )
         assert locator.get_source_stamp() == "abc123"
     
     
     def test_cache_locator_get_disambiguator():
         """Verify disambiguator returns truncated settings hash."""
         locator = CUBIECacheLocator(
             system_name="test_system",
             system_hash="abc123",
             compile_settings_hash="def456789012345678",
         )
         disambiguator = locator.get_disambiguator()
         assert len(disambiguator) == 16
         assert disambiguator == "def4567890123456"
     
     
     def test_cache_locator_from_function_raises():
         """Verify from_function raises NotImplementedError."""
         with pytest.raises(NotImplementedError):
             CUBIECacheLocator.from_function(None, None)
     
     
     # --- CUBIECacheImpl tests ---
     
     def test_cache_impl_locator_property():
         """Verify locator property returns CUBIECacheLocator."""
         impl = CUBIECacheImpl(
             system_name="test_system",
             system_hash="abc123",
             compile_settings_hash="def456",
         )
         assert isinstance(impl.locator, CUBIECacheLocator)
     
     
     def test_cache_impl_filename_base():
         """Verify filename_base format."""
         impl = CUBIECacheImpl(
             system_name="test_system",
             system_hash="abc123",
             compile_settings_hash="def456789012345678",
         )
         assert impl.filename_base.startswith("test_system-")
         assert "def4567890123456" in impl.filename_base
     
     
     def test_cache_impl_check_cachable():
         """Verify check_cachable returns True."""
         impl = CUBIECacheImpl(
             system_name="test_system",
             system_hash="abc123",
             compile_settings_hash="def456",
         )
         assert impl.check_cachable(None) is True
     
     
     # --- CUBIECache tests ---
     
     def test_cubie_cache_init():
         """Verify CUBIECache initializes with system info."""
         settings = MockCompileSettings()
         cache = CUBIECache(
             system_name="test_system",
             system_hash="abc123",
             compile_settings=settings,
         )
         assert cache._system_name == "test_system"
         assert cache._system_hash == "abc123"
     
     
     def test_cubie_cache_path():
         """Verify cache_path property returns expected path."""
         settings = MockCompileSettings()
         cache = CUBIECache(
             system_name="test_system",
             system_hash="abc123",
             compile_settings=settings,
         )
         assert "test_system" in cache.cache_path
         assert "cache" in cache.cache_path
     ```
   - Edge cases: Mock objects should match real attrs patterns
   - Integration: Validates all cache components

**Tests to Run**:
- tests/test_cubie_cache.py (all tests)

**Outcomes**: 
[Empty - to be filled by taskmaster agent]

---

## Summary

**Total Task Groups**: 6
**Dependency Chain**: 1 → 2 → 3 → 4 → 5 → 6 (linear)
**Tests to Create**: 21 test functions in tests/test_cubie_cache.py
**Estimated Complexity**: Medium

**Files to Create**:
- src/cubie/cubie_cache.py (new module with 4 classes + 1 utility function)
- tests/test_cubie_cache.py (comprehensive test suite)

**Files to Modify**:
- src/cubie/batchsolving/BatchSolverConfig.py (add caching_enabled field)
- src/cubie/batchsolving/BatchSolverKernel.py (add cache attachment)

**Key Integration Points**:
1. hash_compile_settings traverses attrs class hierarchy
2. CUBIECacheLocator uses GENERATED_DIR for cache location
3. CUBIECacheImpl delegates to numba's _Kernel methods
4. CUBIECache extends numba's Cache with custom key construction
5. BatchSolverKernel.build_kernel attaches cache to dispatcher

**Risk Areas**:
- Dependency on numba-cuda internals (may break on updates)
- CUDASIM mode compatibility (caching skipped entirely)
- Non-SymbolicODE systems may lack hash (fallback provided)
