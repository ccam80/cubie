# Implementation Task List
# Feature: Improve CuBIE Caching Implementation
# Plan Reference: .github/active_plans/improve_caching_implementation/agent_plan.md

## Overview

This task list implements improvements to CuBIE's kernel caching infrastructure:
1. Separate cache settings from compile-critical configuration (`CacheConfig`)
2. Ensure CUDASIM compatibility for cache testing
3. Align with numba-cuda caching patterns where appropriate
4. Address PR review comments

---

## Task Group 1: Create CacheConfig Class
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 80-140, 236-285)
- File: src/cubie/_utils.py (lines 1-50) - for validator patterns
- File: .github/copilot-instructions.md - for attrs class conventions

**Input Validation Required**:
- enabled: Validate is bool type
- cache_path: Validate is None, str, or Path; if provided, must be a valid directory path or convertible to one
- source_stamp: Validate is None or Tuple[float, int] (mtime, size)

**Tasks**:
1. **Create CacheConfig attrs class**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify
   - Details:
     ```python
     from pathlib import Path
     from typing import Optional, Tuple, Union
     
     @attrs.define
     class CacheConfig(_CubieConfigBase):
         """Configuration for disk-based kernel caching.
         
         This class holds cache-related settings that do NOT affect kernel
         compilation. Changes to these settings should not trigger kernel
         rebuild.
         
         Parameters
         ----------
         enabled
             Whether disk caching is enabled.
         cache_path
             Directory path for cache files. None uses default location.
         source_stamp
             Tuple of (mtime, size) for cache validation. None disables
             source stamp checking.
         """
         
         enabled: bool = attrs.field(
             default=False,
             validator=val.instance_of(bool)
         )
         _cache_path: Optional[Path] = attrs.field(
             default=None,
             alias="cache_path",
             validator=attrs.validators.optional(
                 attrs.validators.instance_of(Path)
             ),
             converter=attrs.converters.optional(Path),
         )
         source_stamp: Optional[Tuple[float, int]] = attrs.field(
             default=None,
             validator=attrs.validators.optional(
                 attrs.validators.instance_of(tuple)
             ),
         )
         
         @property
         def cache_path(self) -> Optional[Path]:
             """Resolved cache directory path."""
             return self._cache_path
         
         @property
         def cache_directory(self) -> Optional[Path]:
             """Return resolved cache directory or None if disabled."""
             if not self.enabled:
                 return None
             return self._cache_path
         
         @classmethod
         def from_cache_param(
             cls,
             cache: Union[bool, str, Path, None]
         ) -> "CacheConfig":
             """Parse cache parameter into CacheConfig.
             
             Parameters
             ----------
             cache
                 Cache configuration:
                 - True: Enable caching with default path
                 - False or None: Disable caching
                 - str or Path: Enable caching at specified path
             
             Returns
             -------
             CacheConfig
                 Configured cache settings.
             """
             if cache is None or cache is False:
                 return cls(enabled=False, cache_path=None)
             
             if cache is True:
                 return cls(enabled=True, cache_path=None)
             
             # str or Path provided
             cache_path = Path(cache) if isinstance(cache, str) else cache
             return cls(enabled=True, cache_path=cache_path)
     ```
   - Edge cases:
     - `cache=True` with no path → enabled=True, cache_path=None (use default)
     - `cache=""` (empty string) → convert to Path("") which is cwd
     - `cache=Path("/some/path")` → enabled=True, cache_path=Path("/some/path")
   - Integration: Import Path and Union from typing; add to module imports

2. **Add CacheConfig import to module __all__ or exports**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify
   - Details: Ensure CacheConfig is importable from the module (no __all__ changes needed since it's implicitly exported)

**Tests to Create**:
- Test file: tests/batchsolving/test_cache_config.py
- Test function: test_cache_config_from_false
- Description: Verify CacheConfig.from_cache_param(False) returns disabled config
- Test function: test_cache_config_from_none
- Description: Verify CacheConfig.from_cache_param(None) returns disabled config
- Test function: test_cache_config_from_true
- Description: Verify CacheConfig.from_cache_param(True) returns enabled config with None path
- Test function: test_cache_config_from_string_path
- Description: Verify CacheConfig.from_cache_param("/tmp/cache") returns enabled config with Path
- Test function: test_cache_config_from_path_object
- Description: Verify CacheConfig.from_cache_param(Path("/tmp")) returns enabled config
- Test function: test_cache_config_cache_directory_disabled
- Description: Verify cache_directory returns None when disabled
- Test function: test_cache_config_cache_directory_enabled
- Description: Verify cache_directory returns path when enabled
- Test function: test_cache_config_values_hash_stable
- Description: Verify values_hash is stable across equivalent configs
- Test function: test_cache_config_update_recognized
- Description: Verify update() recognizes valid fields

**Tests to Run**:
- tests/batchsolving/test_cache_config.py::test_cache_config_from_false
- tests/batchsolving/test_cache_config.py::test_cache_config_from_none
- tests/batchsolving/test_cache_config.py::test_cache_config_from_true
- tests/batchsolving/test_cache_config.py::test_cache_config_from_string_path
- tests/batchsolving/test_cache_config.py::test_cache_config_from_path_object
- tests/batchsolving/test_cache_config.py::test_cache_config_cache_directory_disabled
- tests/batchsolving/test_cache_config.py::test_cache_config_cache_directory_enabled
- tests/batchsolving/test_cache_config.py::test_cache_config_values_hash_stable
- tests/batchsolving/test_cache_config.py::test_cache_config_update_recognized

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverConfig.py (86 lines added)
- Functions/Methods Added/Modified:
  * CacheConfig class added with:
    - enabled, _cache_path (aliased as cache_path), source_stamp fields
    - cache_path property
    - cache_directory property
    - from_cache_param() classmethod
- Test File Created:
  * tests/batchsolving/test_cache_config.py (95 lines)
- Tests Added:
  * TestCacheConfigFromCacheParam: 5 tests for from_cache_param factory
  * TestCacheConfigProperties: 3 tests for property accessors
  * TestCacheConfigHashing: 4 tests for hashing and update behavior
- Implementation Summary:
  Created CacheConfig attrs class inheriting from _CubieConfigBase with cache settings separate from compile-critical configuration. Includes factory method for parsing cache parameter variations (bool/str/Path/None).
- Issues Flagged: None

---

## Task Group 2: Integrate CacheConfig into BatchSolverKernel
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file)
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file, including new CacheConfig)
- File: src/cubie/CUDAFactory.py (lines 287-557) - for CUDAFactory base class

**Input Validation Required**:
- cache parameter in __init__: Accept Union[bool, str, Path, None], delegate validation to CacheConfig.from_cache_param

**Tasks**:
1. **Add cache parameter to BatchSolverKernel.__init__**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Add import at top of file
     from cubie.batchsolving.BatchSolverConfig import CacheConfig
     
     # Modify __init__ signature to add cache parameter
     def __init__(
         self,
         system: "BaseODE",
         loop_settings: Optional[Dict[str, Any]] = None,
         evaluate_driver_at_t: Optional[Callable] = None,
         driver_del_t: Optional[Callable] = None,
         profileCUDA: bool = False,
         step_control_settings: Optional[Dict[str, Any]] = None,
         algorithm_settings: Optional[Dict[str, Any]] = None,
         output_settings: Optional[Dict[str, Any]] = None,
         memory_settings: Optional[Dict[str, Any]] = None,
         cache: Union[bool, str, Path, None] = None,  # NEW PARAMETER
     ) -> None:
     ```
   - Edge cases:
     - cache=None (default) → caching disabled
     - cache=True → caching enabled with default path
     - cache="/path" → caching enabled at specified path
   - Integration: Store as self._cache_config, not in BatchSolverConfig

2. **Store CacheConfig instance in BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # In __init__, after super().__init__():
     self._cache_config = CacheConfig.from_cache_param(cache)
     ```
   - Integration: Add before memory manager setup but after super().__init__()

3. **Add cache_config property to BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     @property
     def cache_config(self) -> CacheConfig:
         """Cache configuration for this kernel."""
         return self._cache_config
     
     @property
     def cache_enabled(self) -> bool:
         """Whether disk caching is enabled."""
         return self._cache_config.enabled
     ```
   - Integration: Add near other property definitions

4. **Add Union, Path imports to BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Modify existing imports
     from typing import (
         TYPE_CHECKING,
         Any,
         Callable,
         Dict,
         List,
         Optional,
         Tuple,
         Union,  # ADD THIS
     )
     from pathlib import Path  # ADD THIS
     ```
   - Integration: Extend existing typing import block

**Tests to Create**:
- Test file: tests/batchsolving/test_cache_config.py (append to existing)
- Test function: test_batchsolverkernel_cache_disabled_by_default
- Description: Verify BatchSolverKernel has cache disabled when cache param not provided
- Test function: test_batchsolverkernel_cache_enabled_with_true
- Description: Verify BatchSolverKernel has cache enabled when cache=True
- Test function: test_batchsolverkernel_cache_enabled_with_path
- Description: Verify BatchSolverKernel has cache enabled with custom path
- Test function: test_batchsolverkernel_cache_config_property
- Description: Verify cache_config property returns CacheConfig instance
- Test function: test_cache_settings_not_in_compile_hash
- Description: Verify changing cache settings does not change compile_settings.values_hash

**Tests to Run**:
- tests/batchsolving/test_cache_config.py::test_batchsolverkernel_cache_disabled_by_default
- tests/batchsolving/test_cache_config.py::test_batchsolverkernel_cache_enabled_with_true
- tests/batchsolving/test_cache_config.py::test_batchsolverkernel_cache_enabled_with_path
- tests/batchsolving/test_cache_config.py::test_batchsolverkernel_cache_config_property
- tests/batchsolving/test_cache_config.py::test_cache_settings_not_in_compile_hash

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverKernel.py (18 lines changed)
- Functions/Methods Added/Modified:
  * Added `from pathlib import Path` import
  * Added `CacheConfig` import from BatchSolverConfig
  * Added `cache: Union[bool, str, Path, None] = None` parameter to __init__
  * Added `self._cache_config = CacheConfig.from_cache_param(cache)` in __init__
  * Added `cache_config` property returning CacheConfig instance
  * Added `cache_enabled` property returning bool
  * Updated class docstring to document cache parameter
- Implementation Summary:
  Integrated CacheConfig into BatchSolverKernel by adding cache parameter to 
  __init__, storing CacheConfig instance separately from compile_settings, and
  adding cache_config/cache_enabled properties for access. Cache settings are
  stored outside of BatchSolverConfig to ensure they don't affect compile hash.
- Issues Flagged: None

---

## Task Group 3: Create Cache Test File with Initial Tests
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/batchsolving/test_SolverKernel.py (entire file) - for test patterns
- File: tests/conftest.py (lines 274-470) - for fixture patterns
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file)
- File: .github/copilot-instructions.md - for test conventions

**Input Validation Required**:
- None (test file)

**Tasks**:
1. **Create test_cache_config.py test file**
   - File: tests/batchsolving/test_cache_config.py
   - Action: Create
   - Details:
     ```python
     """Tests for CacheConfig class and cache configuration."""
     
     from pathlib import Path
     
     import pytest
     
     from cubie.batchsolving.BatchSolverConfig import CacheConfig
     
     
     class TestCacheConfigFromCacheParam:
         """Tests for CacheConfig.from_cache_param factory method."""
     
         def test_cache_config_from_false(self):
             """Verify from_cache_param(False) returns disabled config."""
             config = CacheConfig.from_cache_param(False)
             assert config.enabled is False
             assert config.cache_path is None
     
         def test_cache_config_from_none(self):
             """Verify from_cache_param(None) returns disabled config."""
             config = CacheConfig.from_cache_param(None)
             assert config.enabled is False
             assert config.cache_path is None
     
         def test_cache_config_from_true(self):
             """Verify from_cache_param(True) returns enabled with None path."""
             config = CacheConfig.from_cache_param(True)
             assert config.enabled is True
             assert config.cache_path is None
     
         def test_cache_config_from_string_path(self, tmp_path):
             """Verify from_cache_param with string returns enabled with Path."""
             path_str = str(tmp_path / "cache")
             config = CacheConfig.from_cache_param(path_str)
             assert config.enabled is True
             assert config.cache_path == Path(path_str)
     
         def test_cache_config_from_path_object(self, tmp_path):
             """Verify from_cache_param with Path returns enabled config."""
             cache_path = tmp_path / "cache_dir"
             config = CacheConfig.from_cache_param(cache_path)
             assert config.enabled is True
             assert config.cache_path == cache_path
     
     
     class TestCacheConfigProperties:
         """Tests for CacheConfig property accessors."""
     
         def test_cache_directory_returns_none_when_disabled(self):
             """Verify cache_directory is None when caching is disabled."""
             config = CacheConfig(enabled=False, cache_path=Path("/some/path"))
             assert config.cache_directory is None
     
         def test_cache_directory_returns_path_when_enabled(self, tmp_path):
             """Verify cache_directory returns path when enabled."""
             config = CacheConfig(enabled=True, cache_path=tmp_path)
             assert config.cache_directory == tmp_path
     
         def test_cache_directory_returns_none_when_enabled_no_path(self):
             """Verify cache_directory is None when enabled but no path set."""
             config = CacheConfig(enabled=True, cache_path=None)
             assert config.cache_directory is None
     
     
     class TestCacheConfigHashing:
         """Tests for CacheConfig hashing and update behavior."""
     
         def test_values_hash_stable_for_equivalent_configs(self, tmp_path):
             """Verify equivalent configs produce same hash."""
             config1 = CacheConfig(enabled=True, cache_path=tmp_path)
             config2 = CacheConfig(enabled=True, cache_path=tmp_path)
             assert config1.values_hash == config2.values_hash
     
         def test_values_hash_differs_for_different_enabled(self):
             """Verify different enabled values produce different hashes."""
             config1 = CacheConfig(enabled=True)
             config2 = CacheConfig(enabled=False)
             assert config1.values_hash != config2.values_hash
     
         def test_update_recognizes_enabled_field(self):
             """Verify update() recognizes the enabled field."""
             config = CacheConfig(enabled=False)
             recognized, changed = config.update({"enabled": True})
             assert "enabled" in recognized
             assert "enabled" in changed
             assert config.enabled is True
     
         def test_update_recognizes_cache_path_field(self, tmp_path):
             """Verify update() recognizes the cache_path field."""
             config = CacheConfig(enabled=True, cache_path=None)
             recognized, changed = config.update({"cache_path": tmp_path})
             assert "cache_path" in recognized
             assert "cache_path" in changed
             assert config.cache_path == tmp_path
     ```
   - Edge cases: Tests cover None, False, True, string paths, Path objects
   - Integration: Uses pytest fixtures (tmp_path) for path testing

**Tests to Create**:
- (Tests are defined in the task itself)

**Tests to Run**:
- tests/batchsolving/test_cache_config.py

**Outcomes**: 
- Files Modified: 
  * None (file already exists with all required tests)
- Verification Results:
  * tests/batchsolving/test_cache_config.py verified (95 lines)
  * All 12 required tests present and correct
- Test Classes Verified:
  * TestCacheConfigFromCacheParam: 5 tests for from_cache_param factory
    - test_cache_config_from_false
    - test_cache_config_from_none
    - test_cache_config_from_true
    - test_cache_config_from_string_path
    - test_cache_config_from_path_object
  * TestCacheConfigProperties: 3 tests for property accessors
    - test_cache_directory_returns_none_when_disabled
    - test_cache_directory_returns_path_when_enabled
    - test_cache_directory_returns_none_when_enabled_no_path
  * TestCacheConfigHashing: 4 tests for hashing and update behavior
    - test_values_hash_stable_for_equivalent_configs
    - test_values_hash_differs_for_different_enabled
    - test_update_recognizes_enabled_field
    - test_update_recognizes_cache_path_field
- Implementation Summary:
  Verified that Task Group 1 already created test_cache_config.py with all
  required tests. Tests use pytest fixtures (tmp_path), have no type hints,
  and follow CuBIE test patterns correctly.
- Issues Flagged: None

---

## Task Group 4: Add BatchSolverKernel Integration Tests
**Status**: [x]
**Dependencies**: Task Groups 2, 3

**Required Context**:
- File: tests/batchsolving/test_cache_config.py (entire file as created in Task Group 3)
- File: tests/conftest.py (lines 674-707) - for solverkernel fixture patterns
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file)
- File: tests/system_fixtures.py (lines 1-50) - for system building

**Input Validation Required**:
- None (test file additions)

**Tasks**:
1. **Add BatchSolverKernel cache integration tests**
   - File: tests/batchsolving/test_cache_config.py
   - Action: Modify (append)
   - Details:
     ```python
     # Add imports at top of file
     import numpy as np
     from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
     
     
     class TestBatchSolverKernelCacheConfig:
         """Tests for BatchSolverKernel cache configuration integration."""
     
         def test_batchsolverkernel_cache_disabled_by_default(self, system):
             """Verify cache is disabled when cache param not provided."""
             kernel = BatchSolverKernel(
                 system,
                 algorithm_settings={"algorithm": "euler"},
             )
             assert kernel.cache_enabled is False
             assert kernel.cache_config.enabled is False
     
         def test_batchsolverkernel_cache_enabled_with_true(self, system):
             """Verify cache is enabled when cache=True."""
             kernel = BatchSolverKernel(
                 system,
                 algorithm_settings={"algorithm": "euler"},
                 cache=True,
             )
             assert kernel.cache_enabled is True
             assert kernel.cache_config.cache_path is None
     
         def test_batchsolverkernel_cache_enabled_with_path(
             self, system, tmp_path
         ):
             """Verify cache is enabled with custom path."""
             cache_dir = tmp_path / "kernel_cache"
             kernel = BatchSolverKernel(
                 system,
                 algorithm_settings={"algorithm": "euler"},
                 cache=cache_dir,
             )
             assert kernel.cache_enabled is True
             assert kernel.cache_config.cache_path == cache_dir
     
         def test_cache_config_property_returns_cacheconfig(self, system):
             """Verify cache_config property returns CacheConfig instance."""
             kernel = BatchSolverKernel(
                 system,
                 algorithm_settings={"algorithm": "euler"},
                 cache=True,
             )
             assert isinstance(kernel.cache_config, CacheConfig)
     
         def test_cache_settings_not_in_compile_hash(self, system, tmp_path):
             """Verify cache settings don't affect compile_settings hash."""
             # Create two kernels with different cache settings
             kernel1 = BatchSolverKernel(
                 system,
                 algorithm_settings={"algorithm": "euler"},
                 cache=False,
             )
             kernel2 = BatchSolverKernel(
                 system,
                 algorithm_settings={"algorithm": "euler"},
                 cache=tmp_path,
             )
             # Compile settings hash should be the same
             # (cache settings are separate from compile settings)
             assert (
                 kernel1.compile_settings.values_hash
                 == kernel2.compile_settings.values_hash
             )
     ```
   - Edge cases: Tests with and without cache, with path variations
   - Integration: Uses existing system fixture from conftest.py

**Tests to Create**:
- (Tests are defined in the task itself)

**Tests to Run**:
- tests/batchsolving/test_cache_config.py::TestBatchSolverKernelCacheConfig

**Outcomes**: 
- Files Modified: 
  * tests/batchsolving/test_cache_config.py (67 lines added)
- Functions/Methods Added/Modified:
  * Added `import numpy as np` import
  * Added `from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel` import
  * Added TestBatchSolverKernelCacheConfig class with 5 tests:
    - test_batchsolverkernel_cache_disabled_by_default
    - test_batchsolverkernel_cache_enabled_with_true
    - test_batchsolverkernel_cache_enabled_with_path
    - test_cache_config_property_returns_cacheconfig
    - test_cache_settings_not_in_compile_hash
- Implementation Summary:
  Added BatchSolverKernel integration tests verifying cache configuration is
  properly integrated. Tests use the existing system fixture from conftest.py
  and verify cache_enabled property, cache_config property, and that cache
  settings don't affect compile_settings.values_hash (ensuring cache config
  is separate from compile-critical settings).
- Issues Flagged: None

---

## Task Group 5: CUDASIM Compatibility Review
**Status**: [x]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/cuda_simsafe.py (entire file)
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file)
- File: src/cubie/CUDAFactory.py (lines 1-60, 400-450)

**Input Validation Required**:
- None (review and minimal additions only)

**Tasks**:
1. **Review CacheConfig for CUDASIM compatibility**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Review (no changes expected)
   - Details:
     CacheConfig uses only standard Python types (bool, Path, tuple) and attrs.
     No CUDA intrinsics are used. Verify that:
     - No imports from numba.cuda directly
     - No CUDA device function calls
     - All operations are pure Python file I/O and hashing
   - Edge cases: None expected
   - Integration: CacheConfig should work identically in CUDASIM mode

2. **Verify CUDAFactory.config_hash works in CUDASIM**
   - File: src/cubie/CUDAFactory.py
   - Action: Review (no changes expected)
   - Details:
     The config_hash property uses only hashlib (pure Python). Verify:
     - _hash_tuple uses hashlib.sha256 (pure Python)
     - values_tuple uses attrs astuple (pure Python)
     - No CUDA context needed for hash generation
   - Edge cases: None expected
   - Integration: Hash generation should work identically in CUDASIM mode

3. **Document CUDASIM compatibility in CacheConfig docstring**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify
   - Details:
     Add note to CacheConfig class docstring:
     ```python
     """Configuration for disk-based kernel caching.
     
     ...existing docstring...
     
     Notes
     -----
     All cache operations (hashing, path generation, file I/O) use pure
     Python and work without CUDA intrinsics. This enables cache testing
     with NUMBA_ENABLE_CUDASIM=1.
     """
     ```
   - Integration: Documentation only, no behavioral changes

**Tests to Create**:
- Test file: tests/batchsolving/test_cache_config.py (append)
- Test function: test_cache_config_works_without_cuda
- Description: Verify CacheConfig can be created and used (this test inherently runs in CUDASIM when pytest runs with that env var)

**Tests to Run**:
- tests/batchsolving/test_cache_config.py (all tests should pass in CUDASIM mode)

**Outcomes**: 
- Files Modified: 
  * tests/batchsolving/test_cache_config.py (13 lines added)
- Review Results:
  * CacheConfig uses only pure Python types (bool, Path, Tuple) and attrs
  * No numba.cuda imports or CUDA device function calls in CacheConfig
  * CUDAFactory._hash_tuple uses hashlib.sha256 (pure Python)
  * CUDAFactory.values_hash uses attrs astuple (pure Python)
  * No CUDA context needed for hash generation
  * CUDASIM compatibility Notes already documented in CacheConfig docstring
- Tests Added:
  * TestCacheConfigProperties::test_cache_config_works_without_cuda
- Implementation Summary:
  Reviewed CacheConfig and CUDAFactory hashing for CUDASIM compatibility.
  Verified all cache operations use pure Python without CUDA intrinsics.
  CUDASIM compatibility notes already present in CacheConfig docstring.
  Added test_cache_config_works_without_cuda to validate CUDASIM mode.
- Issues Flagged: None

---

## Task Group 6: Final Integration and Documentation
**Status**: [x]
**Dependencies**: Task Groups 1-5

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file)
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file)
- File: tests/batchsolving/test_cache_config.py (entire file)
- File: .github/active_plans/improve_caching_implementation/human_overview.md

**Input Validation Required**:
- None (documentation and cleanup)

**Tasks**:
1. **Update BatchSolverKernel docstring to document cache parameter**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     Add cache parameter to class and __init__ docstrings:
     ```python
     """Factory for CUDA kernel which coordinates a batch integration.
     
     Parameters
     ----------
     system
         ODE system describing the problem to integrate.
     ...existing parameters...
     cache
         Cache configuration for disk-based kernel caching:
         - None or False: Disable caching (default)
         - True: Enable caching with default cache directory
         - str or Path: Enable caching at specified directory path
     
     ...rest of docstring...
     """
     ```
   - Integration: Documentation only

2. **Verify all tests pass**
   - File: tests/batchsolving/test_cache_config.py
   - Action: Review
   - Details: Ensure all tests in test_cache_config.py pass
   - Integration: Run full test suite for batchsolving module

3. **Update human_overview.md with completion status**
   - File: .github/active_plans/improve_caching_implementation/human_overview.md
   - Action: Modify
   - Details: Mark completed acceptance criteria in the user stories
   - Integration: Documentation update

**Tests to Create**:
- None (verification only)

**Tests to Run**:
- tests/batchsolving/test_cache_config.py
- tests/batchsolving/test_SolverKernel.py (verify no regressions)
- tests/batchsolving/test_config_plumbing.py (verify no regressions)

**Outcomes**: 
- Files Modified: 
  * .github/active_plans/improve_caching_implementation/human_overview.md (4 sections updated)
- Verification Results:
  * BatchSolverKernel docstring already includes cache parameter documentation (lines 115-119)
  * CacheConfig class properly implemented with from_cache_param() factory method
  * BatchSolverKernel properly integrates CacheConfig with cache_config and cache_enabled properties
  * All 17 tests in test_cache_config.py cover required functionality
- Acceptance Criteria Marked Complete:
  * US-1: 5/5 criteria marked complete
  * US-2: 4/4 criteria marked complete (with notes on pure Python implementation)
  * US-3: 2/4 criteria marked complete (remaining 2 are future work for actual caching)
  * US-4: 2/2 criteria marked complete
- Implementation Summary:
  Completed final integration and documentation for cache configuration feature.
  Verified all implementation files are complete and properly documented.
  Updated human_overview.md with completion status for all acceptance criteria.
- Issues Flagged: None

---

## Summary

| Task Group | Description | Status | Dependencies |
|------------|-------------|--------|--------------|
| 1 | Create CacheConfig Class | [x] | None |
| 2 | Integrate CacheConfig into BatchSolverKernel | [x] | 1 |
| 3 | Create Cache Test File | [x] | 1 |
| 4 | Add BatchSolverKernel Integration Tests | [x] | 2, 3 |
| 5 | CUDASIM Compatibility Review | [x] | 1, 2 |
| 6 | Final Integration and Documentation | [x] | 1-5 |

## Dependency Chain

```
Task Group 1 (CacheConfig class)
      │
      ├──► Task Group 2 (BatchSolverKernel integration)
      │         │
      │         └──► Task Group 4 (Integration tests)
      │                    │
      │                    └──► Task Group 6 (Final integration)
      │
      └──► Task Group 3 (Test file creation)
                │
                └──► Task Group 4 (Integration tests)

Task Group 5 (CUDASIM review) depends on Groups 1 & 2
```

## Notes on numba-cuda Patterns

The implementation follows these numba-cuda patterns where appropriate:

1. **Separation of concerns**: `CacheConfig` is separate from compile-critical `BatchSolverConfig`, similar to how numba-cuda separates `_CacheLocator` from `CacheImpl`.

2. **Pure Python cache operations**: Like numba-cuda's `IndexDataCacheFile`, CuBIE's cache configuration uses only pure Python operations (Path, hashlib) that work without CUDA.

3. **Factory method pattern**: `CacheConfig.from_cache_param()` mirrors numba-cuda's `_CacheLocator.from_function()` pattern for flexible configuration parsing.

4. **No py_func requirement**: Unlike numba-cuda's `Cache` class which requires a Python function for source stamps, CuBIE uses `config_hash` from compile settings as the cache key, which doesn't require py_func.

## Estimated Complexity

- Task Group 1: Low (attrs class creation)
- Task Group 2: Low (parameter addition and property)
- Task Group 3: Low (test file creation)
- Task Group 4: Low (test additions)
- Task Group 5: Very Low (review only)
- Task Group 6: Very Low (documentation)

Total: Low complexity implementation with clear separation of concerns.
