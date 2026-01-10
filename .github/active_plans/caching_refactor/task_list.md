# Implementation Task List
# Feature: CuBIE Caching Implementation Refactor
# Plan Reference: .github/active_plans/caching_refactor/agent_plan.md

---

## Task Group 1: Fix Bugs and Clean Up Invalid Code
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/cubie_cache.py (lines 70-99) - CacheConfig.from_user_setting() method
- File: src/cubie/cubie_cache.py (lines 101-179) - CUBIECacheLocator class
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 36) - Import statement

**Input Validation Required**:
- No additional validation needed; these are bug fixes for existing code

**Tasks**:

1. **Fix CacheConfig.from_user_setting() field name bug**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     ```python
     @classmethod
     def from_user_setting(cls, user_setting: Union[bool, str, Path]):
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
         if user_setting is None or user_setting is False:
             return cls(enabled=False, cache_dir=None)  # FIX: cache_path -> cache_dir

         if user_setting is True:
             return cls(enabled=True, cache_dir=None)  # FIX: cache_path -> cache_dir

         cache_path = (
             Path(user_setting)
             if isinstance(user_setting, str)
             else user_setting
         )
         return cls(enabled=True, cache_dir=cache_path)  # FIX: cache_path -> cache_dir
     ```
   - Edge cases: None - direct field name fix
   - Integration: This method is called by BatchSolverKernel to parse user cache settings

2. **Remove invalid __attrs_post_init__ from CUBIECacheLocator**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     Remove lines 177-178:
     ```python
     def __attrs_post_init__(self):
         super().__attrs_post_init__()
     ```
     CUBIECacheLocator is NOT an attrs class (it inherits from _CacheLocator), so
     it cannot have __attrs_post_init__. This method is never called and causes
     an AttributeError if somehow invoked.
   - Edge cases: None - removal of invalid code
   - Integration: No integration changes needed

3. **Fix import of CacheConfig in BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     Change line 36 from:
     ```python
     from cubie.batchsolving.BatchSolverConfig import BatchSolverConfig, CacheConfig
     ```
     To:
     ```python
     from cubie.batchsolving.BatchSolverConfig import BatchSolverConfig
     ```
     CacheConfig is already imported from cubie.cubie_cache on line 24. The import
     from BatchSolverConfig is incorrect and will fail because CacheConfig is not
     defined in BatchSolverConfig.py.
   - Edge cases: None - import correction
   - Integration: No functionality change, just fixes the import path

**Tests to Create**:
- None for this task group - these are bug fixes covered by existing tests

**Tests to Run**:
- tests/test_cubie_cache.py::test_cache_locator_get_cache_path
- tests/test_cubie_cache.py::test_cache_locator_get_source_stamp
- tests/test_cubie_cache.py::test_cache_locator_get_disambiguator
- tests/batchsolving/test_cache_config.py::TestCacheConfigDefaults
- tests/batchsolving/test_cache_config.py::TestParseCacheParam

**Outcomes**:
- Files Modified:
  * src/cubie/cubie_cache.py (6 lines changed)
  * src/cubie/batchsolving/BatchSolverKernel.py (2 lines changed)
- Functions/Methods Added/Modified:
  * CacheConfig.from_user_setting() in cubie_cache.py - Fixed field name from cache_path to cache_dir
  * CUBIECacheLocator class in cubie_cache.py - Removed invalid __attrs_post_init__ method
- Implementation Summary:
  Fixed three bugs: (1) CacheConfig.from_user_setting() was using wrong field name cache_path instead of cache_dir, (2) CUBIECacheLocator had invalid __attrs_post_init__ since it's not an attrs class, (3) BatchSolverKernel.py imported CacheConfig from wrong module
- Issues Flagged: None

---

## Task Group 2: Add cache_config Field to BatchSolverConfig
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file)
- File: src/cubie/cubie_cache.py (lines 31-68) - CacheConfig class definition

**Input Validation Required**:
- cache_config: Must be instance of CacheConfig or None (validator handles this)

**Tasks**:

1. **Add CacheConfig import to BatchSolverConfig.py**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify
   - Details:
     Add import at the top of the file after existing imports:
     ```python
     from cubie.cubie_cache import CacheConfig
     ```
   - Edge cases: None
   - Integration: Required for the cache_config field in BatchSolverConfig

2. **Add cache_config field to BatchSolverConfig**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify
   - Details:
     Add new field to BatchSolverConfig class after compile_flags field:
     ```python
     cache_config: CacheConfig = attrs.field(
         factory=CacheConfig,
         validator=attrs.validators.instance_of(CacheConfig),
         eq=False,  # Cache config is not compile-critical
     )
     ```
     The `eq=False` ensures cache_config changes do not trigger kernel recompilation
     since caching is not a compile-time concern.
   - Edge cases: 
     - Default factory creates enabled CacheConfig with hash mode
     - Custom CacheConfig can be passed at initialization
   - Integration: BatchSolverKernel will populate this field during __init__

**Tests to Create**:
- None - existing tests in test_cache_config.py cover this

**Tests to Run**:
- tests/batchsolving/test_cache_config.py::TestBatchSolverConfigCacheConfig::test_batch_solver_config_cache_config_field
- tests/batchsolving/test_cache_config.py::TestBatchSolverConfigCacheConfig::test_batch_solver_config_cache_config_default
- tests/batchsolving/test_cache_config.py::TestBatchSolverConfigCacheConfig::test_batch_solver_config_with_custom_cache_config

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/BatchSolverConfig.py (7 lines changed)
- Functions/Methods Added/Modified:
  * Added import for CacheConfig from cubie.cubie_cache
  * Added cache_config field to BatchSolverConfig class with factory=CacheConfig, instance_of validator, and eq=False
- Implementation Summary:
  Added CacheConfig import and cache_config field to BatchSolverConfig. The field uses factory=CacheConfig for default enabled caching, instance_of validator for type safety, and eq=False to prevent cache config changes from triggering kernel recompilation since caching is a runtime concern not a compile-time concern.
- Issues Flagged: None

---

## Task Group 3: Refactor BatchSolverKernel Cache Initialization
**Status**: [x]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 124-204) - __init__ method
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 173-174) - current cache_config line
- File: src/cubie/cubie_cache.py (lines 70-99) - CacheConfig.from_user_setting()

**Input Validation Required**:
- cache parameter: Accept bool, str, Path, or "flush_on_change" string
- Validation handled by CacheConfig.from_user_setting() method

**Tasks**:

1. **Replace direct CacheConfig instantiation with from_user_setting()**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     Replace line 174:
     ```python
     self.cache_config = CacheConfig(cache)
     ```
     With:
     ```python
     # Parse user's cache parameter into CacheConfig
     if isinstance(cache, str) and cache == "flush_on_change":
         cache_config = CacheConfig(enabled=True, mode="flush_on_change")
     elif isinstance(cache, (str, Path)):
         cache_config = CacheConfig(enabled=True, cache_dir=Path(cache))
     elif cache is True:
         cache_config = CacheConfig(enabled=True)
     elif cache is False or cache is None:
         cache_config = CacheConfig(enabled=False)
     else:
         cache_config = CacheConfig(enabled=True)
     ```
     Note: This inline logic replaces the buggy from_user_setting() until that
     method is fixed in Task Group 1. The logic handles:
     - True: Enable with defaults
     - False/None: Disable caching
     - "flush_on_change": Enable with flush_on_change mode
     - str/Path: Enable with custom cache directory
   - Edge cases:
     - String "flush_on_change" should set mode, not be treated as path
     - Path objects should be converted to cache_dir
   - Integration: The parsed CacheConfig is passed to BatchSolverConfig

2. **Pass cache_config to BatchSolverConfig initialization**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     Modify the BatchSolverConfig instantiation (around line 176-187) to include
     the cache_config:
     ```python
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
         cache_config=cache_config,  # ADD THIS LINE
     )
     ```
   - Edge cases: None
   - Integration: BatchSolverConfig now stores the cache configuration

3. **Remove standalone self.cache_config assignment**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     Delete line 174 entirely (after the parsing logic is added above):
     ```python
     self.cache_config = CacheConfig(cache)  # DELETE THIS LINE
     ```
     The cache_config is now accessed via self.compile_settings.cache_config
     which is handled by the existing cache_config property.
   - Edge cases: None
   - Integration: The property at line 930-932 already delegates to compile_settings

**Tests to Create**:
- None - existing tests cover this functionality

**Tests to Run**:
- tests/batchsolving/test_cache_config.py::TestParseCacheParam::test_parse_cache_param_true
- tests/batchsolving/test_cache_config.py::TestParseCacheParam::test_parse_cache_param_false
- tests/batchsolving/test_cache_config.py::TestParseCacheParam::test_parse_cache_param_flush_on_change
- tests/batchsolving/test_cache_config.py::TestParseCacheParam::test_parse_cache_param_path
- tests/batchsolving/test_cache_config.py::TestParseCacheParam::test_parse_cache_param_string_path
- tests/batchsolving/test_cache_config.py::TestKernelCacheConfigProperty::test_kernel_cache_config_matches_compile_settings

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (12 lines changed)
- Functions/Methods Added/Modified:
  * __init__() in BatchSolverKernel - replaced direct CacheConfig(cache) with inline parsing logic and passed cache_config to BatchSolverConfig
- Implementation Summary:
  Replaced the incorrect `self.cache_config = CacheConfig(cache)` with proper inline parsing logic that handles: True (enable with defaults), False/None (disable caching), "flush_on_change" string (enable with flush_on_change mode), and str/Path (enable with custom cache directory). The parsed cache_config is now passed to BatchSolverConfig initialization. The existing cache_config property at lines 939-942 correctly delegates to self.compile_settings.cache_config.
- Issues Flagged: None

---

## Task Group 4: Add CUDASIM Mode Stub Classes
**Status**: [x]
**Dependencies**: None (can run in parallel with Task Groups 1-3)

**Required Context**:
- File: src/cubie/cuda_simsafe.py (lines 166-181) - Current caching infrastructure stubs
- File: src/cubie/cubie_cache.py (lines 101-275) - Classes that inherit from stubs

**Input Validation Required**:
- None - stub classes implement minimal interface for CUDASIM compatibility

**Tasks**:

1. **Add _StubCacheLocator class**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     Add before the CUDA_SIMULATION conditional block (around line 166):
     ```python
     class _StubCacheLocator:
         """Stub for _CacheLocator in CUDASIM mode.
         
         Provides minimal interface for CUBIECacheLocator to inherit from
         when running under CUDA simulator.
         """

         def ensure_cache_path(self):
             """Create cache directory if it does not exist."""
             import os
             path = self.get_cache_path()
             os.makedirs(path, exist_ok=True)

         def get_cache_path(self):
             """Return cache directory path. Must be overridden."""
             raise NotImplementedError

         def get_source_stamp(self):
             """Return source freshness stamp. Must be overridden."""
             raise NotImplementedError

         def get_disambiguator(self):
             """Return disambiguator string. Must be overridden."""
             raise NotImplementedError
     ```
   - Edge cases: Subclasses must override abstract methods
   - Integration: CUBIECacheLocator inherits from this in CUDASIM mode

2. **Add _StubCacheImpl class**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     Add after _StubCacheLocator:
     ```python
     class _StubCacheImpl:
         """Stub for CacheImpl in CUDASIM mode.
         
         Provides minimal interface for CUBIECacheImpl to inherit from.
         Serialization methods raise NotImplementedError since CUDA
         context is not available.
         """

         _locator_classes = []

         def reduce(self, data):
             """Reduce data for serialization. Not available in CUDASIM."""
             raise NotImplementedError("Cannot reduce in CUDASIM mode")

         def rebuild(self, target_context, payload):
             """Rebuild from cached payload. Not available in CUDASIM."""
             raise NotImplementedError("Cannot rebuild in CUDASIM mode")

         def check_cachable(self, data):
             """Check if data is cachable. Always False in CUDASIM."""
             return False
     ```
   - Edge cases: reduce/rebuild should raise NotImplementedError
   - Integration: CUBIECacheImpl inherits from this in CUDASIM mode

3. **Add _StubIndexDataCacheFile class**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     Add after _StubCacheImpl:
     ```python
     class _StubIndexDataCacheFile:
         """Stub for IndexDataCacheFile in CUDASIM mode.
         
         Provides minimal interface for cache file operations.
         Save/load operations are no-ops since CUDA caching is unavailable.
         """

         def __init__(self, cache_path, filename_base, source_stamp):
             self._cache_path = cache_path
             self._filename_base = filename_base
             self._source_stamp = source_stamp

         def flush(self):
             """Clear the index. No-op in CUDASIM mode."""
             pass

         def save(self, key, data):
             """Save data to cache. Not available in CUDASIM."""
             raise NotImplementedError("Cannot save in CUDASIM mode")

         def load(self, key):
             """Load data from cache. Always returns None in CUDASIM."""
             return None
     ```
   - Edge cases: load() returns None (cache miss), save() raises
   - Integration: CUBIECache uses this for _cache_file in CUDASIM mode

4. **Add _StubCUDACache class**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     Add after _StubIndexDataCacheFile:
     ```python
     class _StubCUDACache:
         """Stub for CUDACache in CUDASIM mode.
         
         Provides minimal interface for CUBIECache to inherit from.
         Cache operations are no-ops or return cache misses.
         """

         _impl_class = None

         def __init__(self):
             self._enabled = False

         def enable(self):
             """Enable caching. Sets internal flag."""
             self._enabled = True

         def disable(self):
             """Disable caching. Clears internal flag."""
             self._enabled = False

         def load_overload(self, sig, target_context):
             """Load cached overload. Always returns None in CUDASIM."""
             return None

         def save_overload(self, sig, data):
             """Save overload to cache. No-op in CUDASIM."""
             pass
     ```
   - Edge cases: All operations are no-ops or return None
   - Integration: CUBIECache inherits from this in CUDASIM mode

5. **Update CUDASIM conditional to use stub classes**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     Replace lines 168-173:
     ```python
     if CUDA_SIMULATION:  # pragma: no cover - simulated
         _CacheLocator = object
         CacheImpl = object
         IndexDataCacheFile = None
         CUDACache = object
         _CACHING_AVAILABLE = False
     ```
     With:
     ```python
     if CUDA_SIMULATION:  # pragma: no cover - simulated
         _CacheLocator = _StubCacheLocator
         CacheImpl = _StubCacheImpl
         IndexDataCacheFile = _StubIndexDataCacheFile
         CUDACache = _StubCUDACache
         _CACHING_AVAILABLE = False
     ```
   - Edge cases: None - direct replacement
   - Integration: All cache classes now have proper base classes in CUDASIM

**Tests to Create**:
- Test file: tests/test_cubie_cache.py
- Test function: test_cache_locator_cudasim_instantiation
- Description: Verify CUBIECacheLocator can be instantiated in CUDASIM mode

- Test function: test_cache_impl_cudasim_instantiation
- Description: Verify CUBIECacheImpl can be instantiated in CUDASIM mode

- Test function: test_cubie_cache_cudasim_operations
- Description: Verify CUBIECache path operations work in CUDASIM mode

**Tests to Run**:
- tests/test_cubie_cache.py::test_cache_locator_get_cache_path
- tests/test_cubie_cache.py::test_cache_locator_get_source_stamp
- tests/test_cubie_cache.py::test_cache_locator_get_disambiguator
- tests/test_cubie_cache.py::test_cache_impl_locator_property
- tests/test_cubie_cache.py::test_cache_impl_filename_base

**Outcomes**:
- Files Modified:
  * src/cubie/cuda_simsafe.py (108 lines added)
- Functions/Methods Added/Modified:
  * _StubCacheLocator class - stub for _CacheLocator with ensure_cache_path(), get_cache_path(), get_source_stamp(), get_disambiguator()
  * _StubCacheImpl class - stub for CacheImpl with _locator_classes, reduce(), rebuild(), check_cachable()
  * _StubIndexDataCacheFile class - stub for IndexDataCacheFile with flush(), save(), load()
  * _StubCUDACache class - stub for CUDACache with _impl_class, __init__(), enable(), disable(), load_overload(), save_overload()
- Implementation Summary:
  Added four stub classes for CUDASIM mode compatibility. These stubs provide minimal interface implementations that allow CuBIE cache classes (CUBIECacheLocator, CUBIECacheImpl, CUBIECache) to inherit from them when running under CUDA simulator mode. Updated CUDA_SIMULATION conditional block to use the new stub classes instead of bare `object`.
- Issues Flagged: None

---

## Task Group 5: Add Documentation Comments for Review Items
**Status**: [x]
**Dependencies**: Task Groups 1-4

**Required Context**:
- File: src/cubie/cubie_cache.py (lines 276-348) - CUBIECache.__init__()
- File: src/cubie/cubie_cache.py (lines 372-418) - enforce_cache_limit()
- File: src/cubie/cubie_cache.py (lines 432-451) - flush_cache()

**Input Validation Required**:
- None - documentation only

**Tasks**:

1. **Add comment explaining why super().__init__() is not called in CUBIECache**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     Replace line 348 comment:
     ```python
     # super().__init__() # Doesn't work as super().__init__ needs py_func
     ```
     With expanded documentation:
     ```python
     # Note: super().__init__() is intentionally not called.
     # Numba's Cache.__init__(py_func) requires a Python function reference,
     # but CuBIE kernels are dynamically generated without a corresponding
     # py_func. This class manually initializes the required attributes
     # (_name, _impl, _cache_file) that the parent __init__ would set.
     ```
   - Edge cases: None
   - Integration: Documentation only

2. **Add documentation for enforce_cache_limit() design decision**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     Replace the AI review comment at lines 378-381:
     ```python
     # AI review note: this method is implementing a lot of our own
     # logic. Instead, we should use exiting Numba IndexCacheFile
     # mechanics to remove certain entries by index, or resave only
     # certain indices.
     ```
     With:
     ```python
     # Implementation Note: Custom LRU eviction based on filesystem mtime.
     # Numba's IndexDataCacheFile has _load_index() and _save_index() methods
     # that could be used for entry-level removal, but these are private APIs.
     # The current approach is functionally correct and the performance impact
     # is minimal for typical cache sizes (max_entries defaults to 10).
     # TODO: Consider using IndexDataCacheFile internals if Numba exposes
     # a public API for partial index manipulation in future versions.
     ```
   - Edge cases: None
   - Integration: Documentation only

3. **Add documentation for flush_cache() design decision**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     Replace the AI review comment at line 438:
     ```python
     # AI review note: Can't we just use existing Numba cache flush logic?
     ```
     With:
     ```python
     # Implementation Note: Full directory removal via shutil.rmtree.
     # Numba's Cache.flush() only clears the index file, leaving orphaned
     # .nbc data files. For "flush_on_change" mode, complete cache removal
     # is intentional to ensure a clean slate when settings change.
     # The flush() method (index-only) is available via self._cache_file.flush()
     # if needed for lighter-weight cache invalidation.
     ```
   - Edge cases: None
   - Integration: Documentation only

**Tests to Create**:
- None - documentation changes only

**Tests to Run**:
- None - documentation changes only

**Outcomes**:
- Files Modified:
  * src/cubie/cubie_cache.py (15 lines changed)
- Functions/Methods Added/Modified:
  * CUBIECache.__init__() - Expanded comment explaining why super().__init__() is not called
  * enforce_cache_limit() - Replaced AI review note with implementation documentation
  * flush_cache() - Replaced AI review note with implementation documentation
- Implementation Summary:
  Replaced three AI review comments with proper implementation documentation explaining design decisions. The new comments explain: (1) why super().__init__() is intentionally not called due to Numba's py_func requirement, (2) why custom LRU eviction is used instead of Numba internals (private APIs), and (3) why full directory removal is used instead of Numba's flush() (orphaned .nbc files).
- Issues Flagged: None

---

## Task Group 6: Add CUDASIM Compatibility Tests
**Status**: [x]
**Dependencies**: Task Groups 1, 4

**Required Context**:
- File: tests/test_cubie_cache.py (entire file)
- File: src/cubie/cubie_cache.py (entire file)
- File: src/cubie/cuda_simsafe.py (lines 166-181)

**Input Validation Required**:
- None - test code

**Tasks**:

1. **Add CUDASIM-compatible cache instantiation tests**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     Add new test functions after the existing tests (before line 205):
     ```python
     # --- CUDASIM Mode Compatibility Tests ---


     def test_cache_locator_instantiation_works():
         """Verify CUBIECacheLocator can be instantiated regardless of mode."""
         locator = CUBIECacheLocator(
             system_name="cudasim_test",
             system_hash="abc123",
             compile_settings_hash="def456",
         )
         # Path operations should work
         assert locator.get_cache_path() is not None
         assert locator.get_source_stamp() == "abc123"
         assert locator.get_disambiguator() == "def4"


     def test_cache_impl_instantiation_works():
         """Verify CUBIECacheImpl can be instantiated regardless of mode."""
         impl = CUBIECacheImpl(
             system_name="cudasim_test",
             system_hash="abc123",
             compile_settings_hash="def456",
         )
         # Properties should be accessible
         assert impl.locator is not None
         assert impl.filename_base is not None
     ```
   - Edge cases: Tests run in both CUDASIM and real CUDA modes
   - Integration: Validates that stub classes work correctly

2. **Add import for CacheConfig to test file**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     Add CacheConfig to the import from cubie.cubie_cache:
     ```python
     from cubie.cubie_cache import (
         CacheConfig,
         CUBIECacheLocator,
         CUBIECacheImpl,
         CUBIECache,
     )
     ```
   - Edge cases: None
   - Integration: Required for test completeness

**Tests to Create**:
- Tests created inline in Task 1 above

**Tests to Run**:
- tests/test_cubie_cache.py::test_cache_locator_instantiation_works
- tests/test_cubie_cache.py::test_cache_impl_instantiation_works
- tests/test_cubie_cache.py (all tests to verify no regressions)

**Outcomes**:
- Files Modified:
  * tests/test_cubie_cache.py (28 lines added)
- Functions/Methods Added/Modified:
  * Added CacheConfig to imports from cubie.cubie_cache
  * Added test_cache_locator_instantiation_works() test function
  * Added test_cache_impl_instantiation_works() test function
- Implementation Summary:
  Added CacheConfig to the test file imports and created two new CUDASIM-compatible tests. The tests verify that CUBIECacheLocator and CUBIECacheImpl can be instantiated in both CUDASIM and real CUDA modes, and that their path/property operations work correctly regardless of mode.
- Issues Flagged: None

---

## Summary

| Task Group | Description | Dependencies | Estimated Complexity |
|------------|-------------|--------------|---------------------|
| 1 | Fix Bugs and Clean Up Invalid Code | None | Low |
| 2 | Add cache_config Field to BatchSolverConfig | 1 | Low |
| 3 | Refactor BatchSolverKernel Cache Initialization | 1, 2 | Medium |
| 4 | Add CUDASIM Mode Stub Classes | None | Medium |
| 5 | Add Documentation Comments | 1-4 | Low |
| 6 | Add CUDASIM Compatibility Tests | 1, 4 | Low |

**Total Task Groups**: 6

**Dependency Chain**:
```
Task Group 1 ─────┬──────> Task Group 2 ──────> Task Group 3 ──────┐
                  │                                                 │
Task Group 4 ─────┴─────────────────────────────────────────────────┴──> Task Group 5 ──> Task Group 6
```

**Tests to be Created**: 2 new test functions in tests/test_cubie_cache.py

**Tests to be Run**: 
- All tests in tests/test_cubie_cache.py
- All tests in tests/batchsolving/test_cache_config.py
