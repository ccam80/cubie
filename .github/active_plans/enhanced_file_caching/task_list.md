# Implementation Task List
# Feature: Enhanced File-Based Caching
# Plan Reference: .github/active_plans/enhanced_file_caching/agent_plan.md

## Task Group 1: CacheConfig and Enhanced Configuration
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file)
- File: src/cubie/_utils.py (lines 1-50 for validator imports)
- File: .github/copilot-instructions.md (attrs conventions section)

**Input Validation Required**:
- enabled: Validate is bool instance
- mode: Validate is str and in ('hash', 'flush_on_change')
- max_entries: Validate is int >= 0
- cache_dir: Validate is None, str, or Path instance

**Tasks**:
1. **Create CacheConfig attrs class**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Create new attrs class before BatchSolverConfig
   - Details:
     ```python
     @attrs.define
     class CacheConfig:
         """Configuration for file-based kernel caching.

         Parameters
         ----------
         enabled
             Whether file-based caching is enabled.
         mode
             Caching mode: 'hash' for content-addressed caching,
             'flush_on_change' to clear cache when settings change.
         max_entries
             Maximum number of cache entries before LRU eviction.
             Set to 0 to disable eviction.
         cache_dir
             Custom cache directory. None uses default location.
         """

         enabled: bool = attrs.field(
             default=True,
             validator=val.instance_of(bool),
         )
         mode: str = attrs.field(
             default='hash',
             validator=val.in_(('hash', 'flush_on_change')),
         )
         max_entries: int = attrs.field(
             default=10,
             validator=getype_validator(int, 0),
         )
         cache_dir: Optional[Path] = attrs.field(
             default=None,
             validator=val.optional(val.instance_of((str, Path))),
             converter=attrs.converters.optional(Path),
         )
     ```
   - Edge cases: cache_dir can be str or Path, converted to Path
   - Integration: Used by BatchSolverConfig and passed to CUBIECache

2. **Add cache_config field to BatchSolverConfig**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify BatchSolverConfig class
   - Details:
     ```python
     # Add after compile_flags field, remove caching_enabled field
     cache_config: CacheConfig = attrs.field(
         factory=CacheConfig,
         validator=attrs.validators.instance_of(CacheConfig),
     )
     ```
   - Edge cases: Preserve backwards compatibility property for caching_enabled
   - Integration: Replaces caching_enabled field

3. **Add backwards-compatible caching_enabled property**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify BatchSolverConfig class
   - Details:
     ```python
     @property
     def caching_enabled(self) -> bool:
         """Whether caching is enabled (backwards-compatible alias)."""
         return self.cache_config.enabled
     ```
   - Edge cases: Read-only property for backwards compatibility
   - Integration: Existing code using caching_enabled continues to work

4. **Add Path import to BatchSolverConfig.py**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify imports
   - Details:
     ```python
     from pathlib import Path
     ```
   - Edge cases: None
   - Integration: Required for CacheConfig.cache_dir type hint

**Tests to Create**:
- Test file: tests/batchsolving/test_cache_config.py
- Test function: test_cache_config_defaults
- Description: Verify CacheConfig has correct default values
- Test function: test_cache_config_mode_validation
- Description: Verify mode only accepts 'hash' or 'flush_on_change'
- Test function: test_cache_config_max_entries_validation
- Description: Verify max_entries rejects negative values
- Test function: test_cache_config_cache_dir_conversion
- Description: Verify str cache_dir converts to Path
- Test function: test_batch_solver_config_cache_config_field
- Description: Verify BatchSolverConfig has cache_config field
- Test function: test_caching_enabled_backwards_compat
- Description: Verify caching_enabled property returns cache_config.enabled

**Tests to Run**:
- tests/batchsolving/test_cache_config.py::test_cache_config_defaults
- tests/batchsolving/test_cache_config.py::test_cache_config_mode_validation
- tests/batchsolving/test_cache_config.py::test_cache_config_max_entries_validation
- tests/batchsolving/test_cache_config.py::test_cache_config_cache_dir_conversion
- tests/batchsolving/test_cache_config.py::test_batch_solver_config_cache_config_field
- tests/batchsolving/test_cache_config.py::test_caching_enabled_backwards_compat

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverConfig.py (41 lines added)
- Functions/Methods Added/Modified:
  * CacheConfig class added with enabled, mode, max_entries, cache_dir fields
  * cache_config field added to BatchSolverConfig
  * caching_enabled property added to BatchSolverConfig (backwards-compatible)
  * Path import added
- Implementation Summary:
  Created CacheConfig attrs class with validation for all fields. Replaced
  caching_enabled field in BatchSolverConfig with cache_config field that
  uses CacheConfig. Added backwards-compatible caching_enabled property
  that delegates to cache_config.enabled.
- Issues Flagged: None

---

## Task Group 2: LRU Cache Eviction
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/cubie_cache.py (entire file)
- File: src/cubie/batchsolving/BatchSolverConfig.py (CacheConfig class)
- File: src/cubie/odesystems/symbolic/odefile.py (lines 1-20 for GENERATED_DIR)

**Input Validation Required**:
- max_entries: Check is int >= 0 (validated in CacheConfig already)
- cache_path: Directory must exist or be creatable

**Tasks**:
1. **Add max_entries parameter to CUBIECache.__init__**
   - File: src/cubie/cubie_cache.py
   - Action: Modify CUBIECache.__init__
   - Details:
     ```python
     def __init__(
         self,
         system_name: str,
         system_hash: str,
         compile_settings: Any,
         max_entries: int = 10,
     ) -> None:
         # ... existing code ...
         self._max_entries = max_entries
     ```
   - Edge cases: max_entries=0 disables eviction
   - Integration: Stored for use by enforce_cache_limit

2. **Add enforce_cache_limit method to CUBIECache**
   - File: src/cubie/cubie_cache.py
   - Action: Add new method after __init__
   - Details:
     ```python
     def enforce_cache_limit(self) -> None:
         """Evict oldest cache entries if count exceeds max_entries.

         Uses filesystem mtime for LRU ordering. Evicts .nbi/.nbc
         file pairs together.
         """
         if self._max_entries == 0:
             return  # Eviction disabled

         cache_path = Path(self._cache_path)
         if not cache_path.exists():
             return

         # Find all .nbi files (index files)
         nbi_files = list(cache_path.glob("*.nbi"))
         if len(nbi_files) < self._max_entries:
             return

         # Sort by mtime (oldest first)
         nbi_files.sort(key=lambda f: f.stat().st_mtime)

         # Evict oldest until under limit (leave room for new entry)
         files_to_remove = len(nbi_files) - self._max_entries + 1
         for nbi_file in nbi_files[:files_to_remove]:
             base = nbi_file.stem
             # Remove .nbi file
             try:
                 nbi_file.unlink()
             except OSError:
                 pass
             # Remove associated .nbc files (may be multiple)
             for nbc_file in cache_path.glob(f"{base}.*.nbc"):
                 try:
                     nbc_file.unlink()
                 except OSError:
                     pass
     ```
   - Edge cases: Handle missing files, permission errors
   - Integration: Called before save_overload

3. **Add Path import to cubie_cache.py**
   - File: src/cubie/cubie_cache.py
   - Action: Modify imports
   - Details:
     ```python
     from pathlib import Path
     ```
   - Edge cases: None
   - Integration: Required for cache file path operations

4. **Override save_overload to enforce limit before save**
   - File: src/cubie/cubie_cache.py
   - Action: Add method override to CUBIECache
   - Details:
     ```python
     def save_overload(self, sig, data):
         """Save kernel to cache, enforcing entry limit first.

         Parameters
         ----------
         sig
             Function signature.
         data
             Kernel data to cache.
         """
         self.enforce_cache_limit()
         super().save_overload(sig, data)
     ```
   - Edge cases: Parent may fail silently on permission errors
   - Integration: Intercepts parent save_overload

**Tests to Create**:
- Test file: tests/batchsolving/test_cache_config.py
- Test function: test_cubie_cache_max_entries_stored
- Description: Verify max_entries is stored on CUBIECache instance
- Test function: test_enforce_cache_limit_no_eviction_under_limit
- Description: Verify no eviction when file count < max_entries
- Test function: test_enforce_cache_limit_evicts_oldest
- Description: Verify oldest files evicted when limit exceeded
- Test function: test_enforce_cache_limit_zero_disables
- Description: Verify max_entries=0 disables eviction
- Test function: test_enforce_cache_limit_pairs_nbi_nbc
- Description: Verify .nbi and .nbc files evicted together

**Tests to Run**:
- tests/batchsolving/test_cache_config.py::TestCUBIECacheMaxEntries::test_cubie_cache_max_entries_stored
- tests/batchsolving/test_cache_config.py::TestCUBIECacheMaxEntries::test_cubie_cache_max_entries_default
- tests/batchsolving/test_cache_config.py::TestEnforceCacheLimitNoEviction::test_enforce_cache_limit_no_eviction_under_limit
- tests/batchsolving/test_cache_config.py::TestEnforceCacheLimitEviction::test_enforce_cache_limit_evicts_oldest
- tests/batchsolving/test_cache_config.py::TestEnforceCacheLimitDisabled::test_enforce_cache_limit_zero_disables
- tests/batchsolving/test_cache_config.py::TestEnforceCacheLimitPairs::test_enforce_cache_limit_pairs_nbi_nbc

**Outcomes**: 
- Files Modified:
  * src/cubie/cubie_cache.py (52 lines added)
  * tests/batchsolving/test_cache_config.py (178 lines added)
- Functions/Methods Added/Modified:
  * CUBIECache.__init__() - added max_entries parameter
  * CUBIECache.enforce_cache_limit() - new method for LRU eviction
  * CUBIECache.save_overload() - override to enforce limit before save
  * Path import added
- Implementation Summary:
  Added max_entries parameter to CUBIECache with default of 10. Implemented
  enforce_cache_limit() method that evicts oldest .nbi/.nbc file pairs when
  cache exceeds limit. Overrode save_overload() to call enforce_cache_limit()
  before saving. max_entries=0 disables eviction entirely.
- Issues Flagged: None

---

## Task Group 3: Flush-on-Change Mode
**Status**: [x]
**Dependencies**: Task Group 1, Task Group 2

**Required Context**:
- File: src/cubie/cubie_cache.py (entire file)
- File: src/cubie/batchsolving/BatchSolverConfig.py (CacheConfig class)
- File: src/cubie/CUDAFactory.py (lines 330-350 for _invalidate_cache)

**Input Validation Required**:
- mode: Already validated in CacheConfig
- cache_path: Directory must exist for flush operations

**Tasks**:
1. **Add flush_cache method to CUBIECache**
   - File: src/cubie/cubie_cache.py
   - Action: Add new method to CUBIECache class
   - Details:
     ```python
     def flush_cache(self) -> None:
         """Delete all cache files in the cache directory.

         Removes all .nbi and .nbc files, then recreates an empty
         cache directory.
         """
         import shutil
         cache_path = Path(self._cache_path)
         if cache_path.exists():
             try:
                 shutil.rmtree(cache_path)
             except OSError:
                 pass
         try:
             cache_path.mkdir(parents=True, exist_ok=True)
         except OSError:
             pass
     ```
   - Edge cases: Handle permission errors, missing directory
   - Integration: Called by BatchSolverKernel._invalidate_cache

2. **Add mode parameter to CUBIECache.__init__**
   - File: src/cubie/cubie_cache.py
   - Action: Modify CUBIECache.__init__
   - Details:
     ```python
     def __init__(
         self,
         system_name: str,
         system_hash: str,
         compile_settings: Any,
         max_entries: int = 10,
         mode: str = 'hash',
     ) -> None:
         # ... existing code ...
         self._max_entries = max_entries
         self._mode = mode
     ```
   - Edge cases: None
   - Integration: Stored for mode-dependent behavior

3. **Add custom_cache_dir support to CUBIECacheLocator**
   - File: src/cubie/cubie_cache.py
   - Action: Modify CUBIECacheLocator.__init__
   - Details:
     ```python
     def __init__(
         self,
         system_name: str,
         system_hash: str,
         compile_settings_hash: str,
         custom_cache_dir: Optional[Path] = None,
     ) -> None:
         self._system_name = system_name
         self._system_hash = system_hash
         self._compile_settings_hash = compile_settings_hash
         if custom_cache_dir is not None:
             self._cache_path = Path(custom_cache_dir)
         else:
             self._cache_path = GENERATED_DIR / system_name / "cache"
     ```
   - Edge cases: custom_cache_dir may be str or Path
   - Integration: Allows cache_dir override from CacheConfig

4. **Update CUBIECacheImpl to pass custom_cache_dir**
   - File: src/cubie/cubie_cache.py
   - Action: Modify CUBIECacheImpl.__init__
   - Details:
     ```python
     def __init__(
         self,
         system_name: str,
         system_hash: str,
         compile_settings_hash: str,
         custom_cache_dir: Optional[Path] = None,
     ) -> None:
         self._locator = CUBIECacheLocator(
             system_name, system_hash, compile_settings_hash,
             custom_cache_dir=custom_cache_dir,
         )
         # ... rest unchanged ...
     ```
   - Edge cases: None
   - Integration: Passes custom_cache_dir to locator

5. **Update CUBIECache to accept and pass custom_cache_dir**
   - File: src/cubie/cubie_cache.py
   - Action: Modify CUBIECache.__init__
   - Details:
     ```python
     def __init__(
         self,
         system_name: str,
         system_hash: str,
         compile_settings: Any,
         max_entries: int = 10,
         mode: str = 'hash',
         custom_cache_dir: Optional[Path] = None,
     ) -> None:
         # ... existing setup code ...
         self._impl = CUBIECacheImpl(
             system_name,
             system_hash,
             self._compile_settings_hash,
             custom_cache_dir=custom_cache_dir,
         )
         # ... rest unchanged ...
     ```
   - Edge cases: None
   - Integration: Passes custom_cache_dir through to impl

6. **Add Optional import to cubie_cache.py**
   - File: src/cubie/cubie_cache.py
   - Action: Modify imports
   - Details:
     ```python
     from typing import Any, Optional
     ```
   - Edge cases: None
   - Integration: Required for Optional[Path] type hints

**Tests to Create**:
- Test file: tests/batchsolving/test_cache_config.py
- Test function: test_cubie_cache_mode_stored
- Description: Verify mode is stored on CUBIECache instance
- Test function: test_flush_cache_removes_files
- Description: Verify flush_cache removes all cache files
- Test function: test_flush_cache_recreates_directory
- Description: Verify flush_cache creates empty directory
- Test function: test_custom_cache_dir_used
- Description: Verify custom_cache_dir overrides default path
- Test function: test_custom_cache_dir_none_uses_default
- Description: Verify None cache_dir uses GENERATED_DIR path

**Tests to Run**:
- tests/batchsolving/test_cache_config.py::TestCUBIECacheModeStored::test_cubie_cache_mode_stored
- tests/batchsolving/test_cache_config.py::TestCUBIECacheModeStored::test_cubie_cache_mode_default
- tests/batchsolving/test_cache_config.py::TestFlushCacheRemovesFiles::test_flush_cache_removes_files
- tests/batchsolving/test_cache_config.py::TestFlushCacheRecreatesDirectory::test_flush_cache_recreates_directory
- tests/batchsolving/test_cache_config.py::TestFlushCacheRecreatesDirectory::test_flush_cache_handles_missing_directory
- tests/batchsolving/test_cache_config.py::TestCustomCacheDir::test_custom_cache_dir_used
- tests/batchsolving/test_cache_config.py::TestCustomCacheDir::test_custom_cache_dir_none_uses_default

**Outcomes**: 
- Files Modified:
  * src/cubie/cubie_cache.py (38 lines added)
  * tests/batchsolving/test_cache_config.py (117 lines added)
- Functions/Methods Added/Modified:
  * CUBIECacheLocator.__init__() - added custom_cache_dir parameter
  * CUBIECacheImpl.__init__() - added custom_cache_dir parameter
  * CUBIECache.__init__() - added mode and custom_cache_dir parameters
  * CUBIECache.flush_cache() - new method for clearing cache directory
  * Optional import added to typing imports
- Implementation Summary:
  Added mode parameter ('hash' or 'flush_on_change') to CUBIECache for 
  mode-dependent behavior. Added custom_cache_dir parameter to CUBIECacheLocator,
  CUBIECacheImpl, and CUBIECache to support custom cache directory paths.
  Implemented flush_cache() method that removes all cache files and recreates
  an empty directory. Added comprehensive tests for all new functionality.
- Issues Flagged: None

---

## Task Group 4: Enhanced Cache Keyword in BatchSolverKernel
**Status**: [x]
**Dependencies**: Task Group 1, Task Group 2, Task Group 3

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file)
- File: src/cubie/batchsolving/BatchSolverConfig.py (CacheConfig, BatchSolverConfig)
- File: src/cubie/cubie_cache.py (CUBIECache class)
- File: src/cubie/CUDAFactory.py (lines 330-350 for _invalidate_cache)

**Input Validation Required**:
- cache: Check is bool, str in ('flush_on_change'), Path, or str path
- Parse cache parameter into CacheConfig instance

**Tasks**:
1. **Add cache parameter to BatchSolverKernel.__init__**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify __init__ signature and body
   - Details:
     ```python
     def __init__(
         self,
         system: "BaseODE",
         loop_settings: Optional[Dict[str, Any]] = None,
         driver_function: Optional[Callable] = None,
         driver_del_t: Optional[Callable] = None,
         profileCUDA: bool = False,
         step_control_settings: Optional[Dict[str, Any]] = None,
         algorithm_settings: Optional[Dict[str, Any]] = None,
         output_settings: Optional[Dict[str, Any]] = None,
         memory_settings: Optional[Dict[str, Any]] = None,
         cache: Union[bool, str, Path] = True,
     ) -> None:
         # ... existing init code ...

         # Parse cache parameter into CacheConfig
         cache_config = self._parse_cache_param(cache)

         initial_config = BatchSolverConfig(
             precision=precision,
             loop_fn=None,
             local_memory_elements=(
                 self.single_integrator.local_memory_elements
             ),
             shared_memory_elements=(
                 self.single_integrator.shared_memory_elements
             ),
             compile_flags=self.single_integrator.output_compile_flags,
             cache_config=cache_config,
         )
         # ... rest unchanged ...
     ```
   - Edge cases: Handle all cache parameter variants
   - Integration: Parses cache into CacheConfig for BatchSolverConfig

2. **Add _parse_cache_param method to BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Add new method
   - Details:
     ```python
     def _parse_cache_param(
         self, cache: Union[bool, str, Path]
     ) -> "CacheConfig":
         """Parse cache parameter into CacheConfig instance.

         Parameters
         ----------
         cache
             Cache configuration: True enables hash mode, False disables
             caching, 'flush_on_change' enables flush mode, or a Path
             sets a custom cache directory.

         Returns
         -------
         CacheConfig
             Parsed cache configuration.
         """
         from cubie.batchsolving.BatchSolverConfig import CacheConfig

         if isinstance(cache, bool):
             return CacheConfig(enabled=cache)
         elif cache == 'flush_on_change':
             return CacheConfig(enabled=True, mode='flush_on_change')
         elif isinstance(cache, (str, Path)):
             return CacheConfig(enabled=True, cache_dir=Path(cache))
         else:
             raise TypeError(
                 f"cache must be bool, 'flush_on_change', or Path, "
                 f"got {type(cache).__name__}"
             )
     ```
   - Edge cases: str paths vs 'flush_on_change' mode string
   - Integration: Called by __init__ to parse cache param

3. **Add Union and Path imports to BatchSolverKernel.py**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify imports
   - Details:
     ```python
     from pathlib import Path
     # Union already imported from typing
     ```
   - Edge cases: None
   - Integration: Required for type hints

4. **Update build_kernel to use CacheConfig**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify build_kernel method
   - Details:
     ```python
     # Replace existing caching_enabled check with:
     cache_config = self.compile_settings.cache_config
     if (cache_config.enabled and not is_cudasim_enabled()):
         try:
             system = self.single_integrator.system
             system_name = getattr(system, 'name', 'anonymous')
             if hasattr(system, 'system_hash'):
                 system_hash = system.system_hash
             else:
                 system_hash = hashlib.sha256(b'').hexdigest()
             cache = CUBIECache(
                 system_name=system_name,
                 system_hash=system_hash,
                 compile_settings=self.compile_settings,
                 max_entries=cache_config.max_entries,
                 mode=cache_config.mode,
                 custom_cache_dir=cache_config.cache_dir,
             )
             integration_kernel._cache = cache
         except (OSError, TypeError, ValueError, AttributeError):
             pass
     ```
   - Edge cases: Handle all caching errors gracefully
   - Integration: Passes CacheConfig params to CUBIECache

5. **Override _invalidate_cache for flush mode**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Add method override
   - Details:
     ```python
     def _invalidate_cache(self) -> None:
         """Mark cached outputs as invalid, flushing files if in flush mode."""
         super()._invalidate_cache()
         
         cache_config = self.compile_settings.cache_config
         if (cache_config.enabled
                 and cache_config.mode == 'flush_on_change'
                 and not is_cudasim_enabled()):
             try:
                 system = self.single_integrator.system
                 system_name = getattr(system, 'name', 'anonymous')
                 if hasattr(system, 'system_hash'):
                     system_hash = system.system_hash
                 else:
                     system_hash = hashlib.sha256(b'').hexdigest()
                 cache = CUBIECache(
                     system_name=system_name,
                     system_hash=system_hash,
                     compile_settings=self.compile_settings,
                     max_entries=cache_config.max_entries,
                     mode=cache_config.mode,
                     custom_cache_dir=cache_config.cache_dir,
                 )
                 cache.flush_cache()
             except (OSError, TypeError, ValueError, AttributeError):
                 pass
     ```
   - Edge cases: Handle errors from flush operation
   - Integration: Flushes cache files when mode is 'flush_on_change'

6. **Add cache_config property to BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Add new property
   - Details:
     ```python
     @property
     def cache_config(self) -> "CacheConfig":
         """Cache configuration for the kernel."""
         return self.compile_settings.cache_config
     ```
   - Edge cases: None
   - Integration: Provides easy access to cache settings

**Tests to Create**:
- Test file: tests/batchsolving/test_cache_config.py
- Test function: test_kernel_cache_true_enables_caching
- Description: Verify cache=True creates enabled CacheConfig
- Test function: test_kernel_cache_false_disables_caching
- Description: Verify cache=False creates disabled CacheConfig
- Test function: test_kernel_cache_flush_on_change_mode
- Description: Verify cache='flush_on_change' sets mode
- Test function: test_kernel_cache_path_sets_cache_dir
- Description: Verify cache=Path sets custom cache_dir
- Test function: test_kernel_cache_config_property
- Description: Verify cache_config property returns correct object

**Tests to Run**:
- tests/batchsolving/test_cache_config.py::TestParseCacheParam::test_parse_cache_param_true
- tests/batchsolving/test_cache_config.py::TestParseCacheParam::test_parse_cache_param_false
- tests/batchsolving/test_cache_config.py::TestParseCacheParam::test_parse_cache_param_flush_on_change
- tests/batchsolving/test_cache_config.py::TestParseCacheParam::test_parse_cache_param_path
- tests/batchsolving/test_cache_config.py::TestParseCacheParam::test_parse_cache_param_string_path
- tests/batchsolving/test_cache_config.py::TestKernelCacheConfigProperty::test_kernel_cache_config_property
- tests/batchsolving/test_cache_config.py::TestKernelCacheConfigProperty::test_kernel_cache_config_matches_compile_settings

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (75 lines added)
  * tests/batchsolving/test_cache_config.py (90 lines added)
- Functions/Methods Added/Modified:
  * BatchSolverKernel.__init__() - added cache parameter with parsing
  * BatchSolverKernel._parse_cache_param() - new method for parsing cache argument
  * BatchSolverKernel.build_kernel() - updated to use CacheConfig parameters
  * BatchSolverKernel._invalidate_cache() - override for flush_on_change mode
  * BatchSolverKernel.cache_config property - provides access to cache settings
  * Path import added
  * CacheConfig import added
- Implementation Summary:
  Added cache parameter to BatchSolverKernel.__init__ accepting True, False,
  'flush_on_change', or Path/str for custom cache directory. Implemented
  _parse_cache_param method to convert these values into CacheConfig instances.
  Updated build_kernel to pass CacheConfig parameters (max_entries, mode,
  custom_cache_dir) to CUBIECache. Added _invalidate_cache override that
  flushes cache files when mode is 'flush_on_change'. Added cache_config
  property for convenient access to cache settings.
- Issues Flagged: None

---

## Task Group 5: set_cache_dir Method
**Status**: [x]
**Dependencies**: Task Group 4

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file)
- File: src/cubie/batchsolving/BatchSolverConfig.py (CacheConfig class)
- File: src/cubie/CUDAFactory.py (lines 160-230 for update_compile_settings)

**Input Validation Required**:
- path: Check is str or Path instance

**Tasks**:
1. **Add set_cache_dir method to BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Add new method
   - Details:
     ```python
     def set_cache_dir(self, path: Union[str, Path]) -> None:
         """Set a custom cache directory for compiled kernels.

         Parameters
         ----------
         path
             New cache directory path. Can be absolute or relative.

         Notes
         -----
         Invalidates the current cache, causing a rebuild on next access.
         """
         cache_config = self.compile_settings.cache_config
         cache_config.cache_dir = Path(path)
         self._invalidate_cache()
     ```
   - Edge cases: Handle both str and Path inputs
   - Integration: Updates CacheConfig and invalidates cache

**Tests to Create**:
- Test file: tests/batchsolving/test_cache_config.py
- Test function: test_set_cache_dir_updates_config
- Description: Verify set_cache_dir updates cache_config.cache_dir
- Test function: test_set_cache_dir_invalidates_cache
- Description: Verify set_cache_dir calls _invalidate_cache
- Test function: test_set_cache_dir_accepts_string
- Description: Verify set_cache_dir accepts string path
- Test function: test_set_cache_dir_accepts_path
- Description: Verify set_cache_dir accepts Path object

**Tests to Run**:
- tests/batchsolving/test_cache_config.py::TestSetCacheDir::test_set_cache_dir_updates_config
- tests/batchsolving/test_cache_config.py::TestSetCacheDir::test_set_cache_dir_invalidates_cache
- tests/batchsolving/test_cache_config.py::TestSetCacheDir::test_set_cache_dir_accepts_string
- tests/batchsolving/test_cache_config.py::TestSetCacheDir::test_set_cache_dir_accepts_path

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (17 lines added)
  * tests/batchsolving/test_cache_config.py (58 lines added)
- Functions/Methods Added/Modified:
  * BatchSolverKernel.set_cache_dir() - new method for setting custom cache directory
- Implementation Summary:
  Added set_cache_dir method to BatchSolverKernel that accepts a path (str or Path),
  updates cache_config.cache_dir, and invalidates the cached kernel to force rebuild
  with the new directory. The method converts string paths to Path objects. Created
  four tests to verify: config update, cache invalidation, string path acceptance,
  and Path object acceptance.
- Issues Flagged: None

---

## Task Group 6: Solver Pass-through Methods
**Status**: [x]
**Dependencies**: Task Group 4, Task Group 5

**Required Context**:
- File: src/cubie/batchsolving/solver.py (entire file)
- File: src/cubie/batchsolving/BatchSolverKernel.py (cache-related properties)
- File: src/cubie/batchsolving/BatchSolverConfig.py (CacheConfig class)

**Input Validation Required**:
- cache parameter in Solver.__init__: Same as BatchSolverKernel
- path in set_cache_dir: Check is str or Path instance

**Tasks**:
1. **Add cache parameter to Solver.__init__**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify __init__ signature and body
   - Details:
     ```python
     def __init__(
         self,
         system: BaseODE,
         algorithm: str = "euler",
         profileCUDA: bool = False,
         step_control_settings: Optional[Dict[str, object]] = None,
         algorithm_settings: Optional[Dict[str, object]] = None,
         output_settings: Optional[Dict[str, object]] = None,
         memory_settings: Optional[Dict[str, object]] = None,
         loop_settings: Optional[Dict[str, object]] = None,
         strict: bool = False,
         time_logging_level: Optional[str] = None,
         cache: Union[bool, str, Path] = True,
         **kwargs: Any,
     ) -> None:
         # ... existing init code ...
         
         self.kernel = BatchSolverKernel(
             system,
             loop_settings=loop_settings,
             profileCUDA=profileCUDA,
             step_control_settings=step_settings,
             algorithm_settings=algorithm_settings,
             output_settings=output_settings,
             memory_settings=memory_settings,
             cache=cache,
         )
         # ... rest unchanged ...
     ```
   - Edge cases: Pass cache through to kernel
   - Integration: Forwards cache param to BatchSolverKernel

2. **Add Path and Union imports to solver.py**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify imports
   - Details:
     ```python
     from pathlib import Path
     # Union already imported from typing
     ```
   - Edge cases: None
   - Integration: Required for type hints

3. **Add cache_enabled property to Solver**
   - File: src/cubie/batchsolving/solver.py
   - Action: Add new property
   - Details:
     ```python
     @property
     def cache_enabled(self) -> bool:
         """Whether file-based caching is enabled."""
         return self.kernel.cache_config.enabled
     ```
   - Edge cases: None
   - Integration: Delegates to kernel.cache_config

4. **Add cache_mode property to Solver**
   - File: src/cubie/batchsolving/solver.py
   - Action: Add new property
   - Details:
     ```python
     @property
     def cache_mode(self) -> str:
         """Current caching mode ('hash' or 'flush_on_change')."""
         return self.kernel.cache_config.mode
     ```
   - Edge cases: None
   - Integration: Delegates to kernel.cache_config

5. **Add cache_dir property to Solver**
   - File: src/cubie/batchsolving/solver.py
   - Action: Add new property
   - Details:
     ```python
     @property
     def cache_dir(self) -> Optional[Path]:
         """Custom cache directory, or None for default location."""
         return self.kernel.cache_config.cache_dir
     ```
   - Edge cases: None
   - Integration: Delegates to kernel.cache_config

6. **Add set_cache_dir method to Solver**
   - File: src/cubie/batchsolving/solver.py
   - Action: Add new method
   - Details:
     ```python
     def set_cache_dir(self, path: Union[str, Path]) -> None:
         """Set a custom cache directory for compiled kernels.

         Parameters
         ----------
         path
             New cache directory path. Can be absolute or relative.

         Notes
         -----
         Invalidates the current cache, causing a rebuild on next access.
         """
         self.kernel.set_cache_dir(path)
     ```
   - Edge cases: None
   - Integration: Delegates to kernel.set_cache_dir

**Tests to Create**:
- Test file: tests/batchsolving/test_cache_config.py
- Test function: test_solver_cache_param_passed_to_kernel
- Description: Verify Solver passes cache param to kernel
- Test function: test_solver_cache_enabled_property
- Description: Verify cache_enabled returns kernel value
- Test function: test_solver_cache_mode_property
- Description: Verify cache_mode returns kernel value
- Test function: test_solver_cache_dir_property
- Description: Verify cache_dir returns kernel value
- Test function: test_solver_set_cache_dir_delegates
- Description: Verify set_cache_dir calls kernel method

**Tests to Run**:
- tests/batchsolving/test_cache_config.py::TestSolverCacheParam::test_solver_cache_param_passed_to_kernel
- tests/batchsolving/test_cache_config.py::TestSolverCacheParam::test_solver_cache_true_default
- tests/batchsolving/test_cache_config.py::TestSolverCacheParam::test_solver_cache_false
- tests/batchsolving/test_cache_config.py::TestSolverCacheParam::test_solver_cache_flush_on_change
- tests/batchsolving/test_cache_config.py::TestSolverCacheProperties::test_solver_cache_enabled_property
- tests/batchsolving/test_cache_config.py::TestSolverCacheProperties::test_solver_cache_mode_property
- tests/batchsolving/test_cache_config.py::TestSolverCacheProperties::test_solver_cache_dir_property
- tests/batchsolving/test_cache_config.py::TestSolverSetCacheDir::test_solver_set_cache_dir_delegates
- tests/batchsolving/test_cache_config.py::TestSolverSetCacheDir::test_solver_set_cache_dir_string
- tests/batchsolving/test_cache_config.py::TestSolverSetCacheDir::test_solver_set_cache_dir_path

**Outcomes**: 
- Files Modified:
  * src/cubie/batchsolving/solver.py (35 lines added)
  * tests/batchsolving/test_cache_config.py (100 lines added)
- Functions/Methods Added/Modified:
  * Solver.__init__() - added cache parameter with pass-through to kernel
  * Solver.cache_enabled property - delegates to kernel.cache_config.enabled
  * Solver.cache_mode property - delegates to kernel.cache_config.mode
  * Solver.cache_dir property - delegates to kernel.cache_config.cache_dir
  * Solver.set_cache_dir() - delegates to kernel.set_cache_dir()
  * Path import added
- Implementation Summary:
  Added cache parameter to Solver.__init__ accepting True, False,
  'flush_on_change', or Path/str for custom cache directory. Parameter is
  passed through to BatchSolverKernel. Added three read-only properties
  (cache_enabled, cache_mode, cache_dir) that delegate to kernel.cache_config.
  Added set_cache_dir method that delegates to kernel.set_cache_dir.
  Created 10 tests covering parameter passing, properties, and method.
- Issues Flagged: None

---

## Task Group 7: Session Save/Load
**Status**: [x]
**Dependencies**: None (can be implemented in parallel with other groups)

**Required Context**:
- File: src/cubie/__init__.py (entire file for exports)
- File: src/cubie/odesystems/symbolic/odefile.py (lines 1-20 for GENERATED_DIR)
- File: src/cubie/batchsolving/solver.py (Solver class, compile_settings access)

**Input Validation Required**:
- name: Check is non-empty string
- solver: Check has kernel attribute with compile_settings

**Tasks**:
1. **Create src/cubie/session.py module**
   - File: src/cubie/session.py
   - Action: Create new file
   - Details:
     ```python
     """Session save/load utilities for CuBIE solver configurations.

     Provides functions to persist and restore solver compile settings
     across Python sessions.
     """

     import pickle
     import warnings
     from pathlib import Path
     from typing import Any, TYPE_CHECKING

     from cubie.odesystems.symbolic.odefile import GENERATED_DIR

     if TYPE_CHECKING:
         from cubie.batchsolving.solver import Solver

     SESSIONS_DIR = GENERATED_DIR / "sessions"


     def save_session(solver: "Solver", name: str) -> Path:
         """Save solver compile settings to a named session file.

         Parameters
         ----------
         solver
             Solver instance whose settings will be saved.
         name
             Session name used for the file (without extension).

         Returns
         -------
         Path
             Path to the saved session file.

         Notes
         -----
         Overwrites existing sessions with the same name, emitting a
         warning.
         """
         if not name:
             raise ValueError("Session name cannot be empty")

         SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
         session_path = SESSIONS_DIR / f"{name}.pkl"

         if session_path.exists():
             warnings.warn(
                 f"Overwriting existing session '{name}'",
                 UserWarning,
                 stacklevel=2,
             )

         compile_settings = solver.kernel.compile_settings
         with open(session_path, 'wb') as f:
             pickle.dump(compile_settings, f, protocol=pickle.HIGHEST_PROTOCOL)

         return session_path


     def load_from_session(name: str) -> Any:
         """Load compile settings from a named session file.

         Parameters
         ----------
         name
             Session name to load (without extension).

         Returns
         -------
         Any
             The saved compile_settings object.

         Raises
         ------
         FileNotFoundError
             If the session file does not exist.
         """
         if not name:
             raise ValueError("Session name cannot be empty")

         session_path = SESSIONS_DIR / f"{name}.pkl"

         if not session_path.exists():
             raise FileNotFoundError(
                 f"Session '{name}' not found at {session_path}"
             )

         with open(session_path, 'rb') as f:
             return pickle.load(f)
     ```
   - Edge cases: Handle empty names, missing files, overwrite warnings
   - Integration: Standalone module using GENERATED_DIR

2. **Export session functions from __init__.py**
   - File: src/cubie/__init__.py
   - Action: Add imports and exports
   - Details:
     ```python
     from cubie.session import save_session, load_from_session

     __all__ = [
         # ... existing exports ...
         "save_session",
         "load_from_session",
     ]
     ```
   - Edge cases: None
   - Integration: Makes functions available as cubie.save_session()

**Tests to Create**:
- No tests required for session module per requirements ("best-practice
  architecture, no tests required")

**Tests to Run**:
- None required

**Outcomes**: 
- Files Modified:
  * src/cubie/session.py (90 lines created)
  * src/cubie/__init__.py (3 lines added)
- Functions/Methods Added/Modified:
  * save_session() in session.py - saves solver compile_settings to pickle file
  * load_from_session() in session.py - restores compile_settings from pickle file
  * SESSIONS_DIR constant - directory path for session files
  * Added imports and exports to __init__.py
- Implementation Summary:
  Created session.py module with save_session and load_from_session functions.
  save_session pickles solver.kernel.compile_settings to generated/sessions/{name}.pkl,
  warns on overwrite. load_from_session restores compile_settings from the session file.
  Both functions validate that name is non-empty. Exported functions from cubie
  package for user access via cubie.save_session() and cubie.load_from_session().
- Issues Flagged: None

---

# Summary

## Total Task Groups: 7

## Dependency Chain Overview:
```
Task Group 1 (CacheConfig) ─┬─> Task Group 2 (LRU Eviction)
                            │
                            └─> Task Group 3 (Flush Mode) ─┬─> Task Group 4 (Kernel Cache Param)
                                                           │
                                                           └─> Task Group 5 (set_cache_dir)
                                                                           │
                                                                           └─> Task Group 6 (Solver Pass-through)

Task Group 7 (Session Save/Load) ─── (Independent, no dependencies)
```

## Tests to be Created:
- tests/batchsolving/test_cache_config.py (new file with 29 test functions)

## Estimated Complexity:
- Task Group 1: Low (new attrs class, simple field additions)
- Task Group 2: Medium (filesystem operations, LRU logic)
- Task Group 3: Medium (flush operations, parameter threading)
- Task Group 4: Medium-High (parameter parsing, cache integration)
- Task Group 5: Low (single method addition)
- Task Group 6: Low (pass-through properties and methods)
- Task Group 7: Low (standalone pickle-based persistence)
