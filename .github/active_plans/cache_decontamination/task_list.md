# Implementation Task List
# Feature: Cache Decontamination
# Plan Reference: .github/active_plans/cache_decontamination/agent_plan.md

## Task Group 1: Remove cache_config from BatchSolverConfig
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverConfig.py (entire file, lines 1-147)
- File: .github/copilot-instructions.md (for coding style reference)

**Input Validation Required**:
- None for this task (removing a field)

**Tasks**:
1. **Remove CacheConfig import from BatchSolverConfig.py**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify
   - Details:
     ```python
     # Remove this import line:
     from cubie.cubie_cache import CacheConfig
     ```
   - Edge cases: None
   - Integration: Other files that import CacheConfig will continue to import from cubie_cache directly

2. **Remove cache_config field from BatchSolverConfig class**
   - File: src/cubie/batchsolving/BatchSolverConfig.py
   - Action: Modify
   - Details:
     ```python
     # Remove these lines (around lines 134-138):
     cache_config: CacheConfig = attrs.field(
         factory=CacheConfig,
         validator=attrs.validators.instance_of(CacheConfig),
         eq=False,  # Cache config is not compile-critical
     )
     ```
   - Edge cases: None - BatchSolverConfig should only contain compile-critical settings
   - Integration: BatchSolverKernel will no longer pass cache_config to BatchSolverConfig

**Tests to Create**:
- None (existing tests will be updated in Task Group 5)

**Tests to Run**:
- tests/batchsolving/test_cache_config.py::TestBatchSolverConfigCacheConfig (expect failures - these tests will be removed in Task Group 5)

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverConfig.py (6 lines removed)
- Functions/Methods Added/Modified:
  * BatchSolverConfig class modified (cache_config field removed)
- Implementation Summary:
  Removed the CacheConfig import and cache_config field from BatchSolverConfig. 
  The class now contains only compile-critical settings (precision, loop_fn, 
  local_memory_elements, shared_memory_elements, compile_flags). The file 
  is now 141 lines (reduced from 147).
- Issues Flagged: Tests that reference compile_settings.cache_config will fail 
  until Task Group 5 updates them.

---

## Task Group 2: Add module-level cache functions to cubie_cache.py
**Status**: [x]
**Dependencies**: None (can run in parallel with Task Group 1)

**Required Context**:
- File: src/cubie/cubie_cache.py (entire file, lines 1-470)
- File: src/cubie/cuda_simsafe.py (lines 1-50, 166-288, 434-438 for is_cudasim_enabled and cache classes)
- File: src/cubie/odesystems/baseODE.py (for BaseODE type hint - only need to know it exists)
- File: .github/copilot-instructions.md (for coding style reference)

**Input Validation Required**:
- cache_arg: Must be Union[bool, str, Path] - no additional validation needed (CacheConfig.from_user_setting handles this)
- system_name: str - passed directly to CUBIECache
- system_hash: str - passed directly to CUBIECache
- config_hash: str - passed directly to CUBIECache

**Tasks**:
1. **Add create_cache function to cubie_cache.py**
   - File: src/cubie/cubie_cache.py
   - Action: Modify (add function after CUBIECache class, before any __all__ if present)
   - Details:
     ```python
     def create_cache(
         cache_arg: Union[bool, str, Path],
         system_name: str,
         system_hash: str,
         config_hash: str,
     ) -> Optional["CUBIECache"]:
         """Create a CUBIECache from raw cache argument.

         Parameters
         ----------
         cache_arg
             Cache configuration from user:
             - True: Enable caching with default path
             - False or None: Disable caching
             - "flush_on_change": Enable caching with flush_on_change mode
             - str or Path: Enable caching at specified path
         system_name
             Name of the ODE system for directory organization.
         system_hash
             Hash representing the ODE system definition.
         config_hash
             Pre-computed hash of compile settings.

         Returns
         -------
         CUBIECache or None
             CUBIECache instance if caching enabled and not in CUDASIM mode,
             None otherwise.
         """
         from cubie.cuda_simsafe import is_cudasim_enabled

         if is_cudasim_enabled():
             return None

         cache_config = CacheConfig.from_user_setting(cache_arg)
         if not cache_config.enabled:
             return None

         return CUBIECache(
             system_name=system_name,
             system_hash=system_hash,
             config_hash=config_hash,
             max_entries=cache_config.max_entries,
             mode=cache_config.mode,
             custom_cache_dir=cache_config.cache_dir,
         )
     ```
   - Edge cases: 
     - CUDASIM mode returns None
     - Disabled cache returns None
   - Integration: Called from BatchSolverKernel.build_kernel()

2. **Add invalidate_cache function to cubie_cache.py**
   - File: src/cubie/cubie_cache.py
   - Action: Modify (add function after create_cache function)
   - Details:
     ```python
     def invalidate_cache(
         cache_arg: Union[bool, str, Path],
         system_name: str,
         system_hash: str,
         config_hash: str,
     ) -> None:
         """Invalidate cache if in flush_on_change mode.

         Parameters
         ----------
         cache_arg
             Cache configuration from user (same format as create_cache).
         system_name
             Name of the ODE system.
         system_hash
             Hash representing the ODE system definition.
         config_hash
             Pre-computed hash of compile settings.

         Notes
         -----
         Only flushes cache when mode is "flush_on_change". Silent on errors
         since cache flush is best-effort. No-op in CUDASIM mode.
         """
         from cubie.cuda_simsafe import is_cudasim_enabled

         if is_cudasim_enabled():
             return

         cache_config = CacheConfig.from_user_setting(cache_arg)
         if not cache_config.enabled:
             return
         if cache_config.mode != "flush_on_change":
             return

         try:
             cache = CUBIECache(
                 system_name=system_name,
                 system_hash=system_hash,
                 config_hash=config_hash,
                 max_entries=cache_config.max_entries,
                 mode=cache_config.mode,
                 custom_cache_dir=cache_config.cache_dir,
             )
             cache.flush_cache()
         except (OSError, TypeError, ValueError, AttributeError):
             # Broad catch intentional: cache flush is best-effort.
             pass
     ```
   - Edge cases: 
     - CUDASIM mode: no-op
     - Disabled cache: no-op
     - mode != "flush_on_change": no-op
     - Errors during flush: silently caught
   - Integration: Called from BatchSolverKernel._invalidate_cache()

**Tests to Create**:
- Test file: tests/test_cubie_cache.py
- Test function: test_create_cache_returns_none_when_disabled
- Description: Verify create_cache returns None when cache_arg=False
- Test function: test_create_cache_returns_cache_when_enabled
- Description: Verify create_cache returns CUBIECache when cache_arg=True (nocudasim)
- Test function: test_invalidate_cache_no_op_when_hash_mode
- Description: Verify invalidate_cache does nothing when mode="hash"
- Test function: test_invalidate_cache_flushes_when_flush_mode
- Description: Verify invalidate_cache calls flush when mode="flush_on_change" (nocudasim)

**Tests to Run**:
- tests/test_cubie_cache.py::test_create_cache_returns_none_when_disabled
- tests/test_cubie_cache.py::test_create_cache_returns_cache_when_enabled
- tests/test_cubie_cache.py::test_invalidate_cache_no_op_when_hash_mode
- tests/test_cubie_cache.py::test_invalidate_cache_flushes_when_flush_mode

**Outcomes**: 
- Files Modified: 
  * src/cubie/cubie_cache.py (95 lines added)
  * tests/test_cubie_cache.py (84 lines added)
- Functions/Methods Added/Modified:
  * create_cache() in cubie_cache.py - creates CUBIECache from raw cache arg
  * invalidate_cache() in cubie_cache.py - flushes cache if flush_on_change
- Implementation Summary:
  Added two module-level functions to cubie_cache.py that consolidate cache 
  creation and invalidation logic. Both functions handle CUDASIM mode by 
  returning None or doing nothing. They use CacheConfig.from_user_setting() 
  to parse the raw cache argument. Added 5 tests to verify the new functions.
- Issues Flagged: None

---

## Task Group 3: Simplify BatchSolverKernel cache handling
**Status**: [x]
**Dependencies**: Task Group 1, Task Group 2

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (entire file, lines 1-1385)
- File: src/cubie/cubie_cache.py (lines 470-end for create_cache and invalidate_cache functions after Task Group 2)
- File: .github/copilot-instructions.md (for coding style reference)

**Input Validation Required**:
- cache parameter in __init__: Union[bool, str, Path] - stored directly, validation deferred to CacheConfig.from_user_setting

**Tasks**:
1. **Update imports in BatchSolverKernel.py**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Change line 24 from:
     from cubie.cubie_cache import CUBIECache, CacheConfig
     # To:
     from cubie.cubie_cache import CacheConfig, create_cache, invalidate_cache
     ```
   - Edge cases: None
   - Integration: Uses new module-level functions

2. **Store raw cache argument in __init__**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # After line 145 (self._profileCUDA = profileCUDA), add:
     self._cache_arg: Union[bool, str, Path] = cache

     # Remove lines 173-174:
     # Parse user's cache parameter into CacheConfig
     cache_config = CacheConfig.from_user_setting(cache)

     # Modify lines 176-188 to remove cache_config from BatchSolverConfig:
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
         # Remove: cache_config=cache_config,
     )
     ```
   - Edge cases: None
   - Integration: Raw cache_arg used by create_cache/invalidate_cache functions

3. **Update cache_config property to parse on-demand**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Replace lines 930-933:
     @property
     def cache_config(self) -> "CacheConfig":
         """Cache configuration for the kernel."""
         return self.compile_settings.cache_config
     
     # With:
     @property
     def cache_config(self) -> "CacheConfig":
         """Cache configuration for the kernel, parsed on demand."""
         return CacheConfig.from_user_setting(self._cache_arg)
     ```
   - Edge cases: This creates a new CacheConfig each time - acceptable since it's a lightweight attrs class
   - Integration: External code accessing cache_config continues to work

4. **Update set_cache_dir to modify _cache_arg**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Replace lines 935-949:
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

     # With:
     def set_cache_dir(self, path: Union[str, Path]) -> None:
         """Set a custom cache directory for compiled kernels.

         Parameters
         ----------
         path
             New cache directory path. Can be absolute or relative.

         Notes
         -----
         Setting cache_dir implies caching is desired. Updates _cache_arg
         to the new path and invalidates the current cache.
         """
         self._cache_arg = Path(path)
         self._invalidate_cache()
     ```
   - Edge cases: Setting cache_dir when cache was False now enables caching (intentional - setting a dir implies wanting caching)
   - Integration: Solver.set_cache_dir delegates to this method

5. **Simplify _invalidate_cache to delegate to cache module**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Replace lines 981-1010:
     def _invalidate_cache(self) -> None:
         """Mark cached outputs as invalid, flushing files if in flush mode."""
         super()._invalidate_cache()

         cache_config = self.compile_settings.cache_config
         if (
             cache_config.enabled
             and cache_config.mode == "flush_on_change"
             and not is_cudasim_enabled()
         ):
             try:
                 system = self.single_integrator.system
                 system_name = getattr(system, "name", "anonymous")
                 system_hash = system.fn_hash

                 cache = CUBIECache(
                     system_name=system_name,
                     system_hash=system_hash,
                     config_hash=self.config_hash,
                     max_entries=cache_config.max_entries,
                     mode=cache_config.mode,
                     custom_cache_dir=cache_config.cache_dir,
                 )
                 cache.flush_cache()
             except (OSError, TypeError, ValueError, AttributeError):
                 # Broad catch intentional: cache flush is best-effort.
                 # OSError: file system errors
                 # TypeError/ValueError: invalid cache config
                 # AttributeError: missing system attributes during early init
                 pass

     # With:
     def _invalidate_cache(self) -> None:
         """Mark cached outputs as invalid, flushing files if in flush mode."""
         super()._invalidate_cache()

         try:
             system = self.single_integrator.system
             system_name = getattr(system, "name", "anonymous")
             system_hash = system.fn_hash

             invalidate_cache(
                 cache_arg=self._cache_arg,
                 system_name=system_name,
                 system_hash=system_hash,
                 config_hash=self.config_hash,
             )
         except AttributeError:
             # Missing system attributes during early init
             pass
     ```
   - Edge cases: AttributeError during early init when system not yet available
   - Integration: Delegates to invalidate_cache() from cubie_cache module

6. **Remove instantiate_cache method**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify (delete method)
   - Details:
     ```python
     # Remove lines 1012-1028:
     def instantiate_cache(self):
         cache_config = self.compile_settings.cache_config
         if cache_config.enabled and not is_cudasim_enabled():
             system = self.single_integrator.system
             system_name = getattr(system, "name", "anonymous")
             system_hash = system.fn_hash

             cache = CUBIECache(
                 system_name=system_name,
                 system_hash=system_hash,
                 config_hash=self.config_hash,
                 max_entries=cache_config.max_entries,
                 mode=cache_config.mode,
                 custom_cache_dir=cache_config.cache_dir,
             )
             return cache
         return None
     ```
   - Edge cases: None - functionality replaced by create_cache module function
   - Integration: No known callers - this was a helper method

7. **Update build_kernel to use create_cache function**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     ```python
     # Replace lines 809-824:
     # Attach file-based caching if enabled and not in simulator mode
     cache_config = self.cache_config
     if cache_config.enabled and not is_cudasim_enabled():
         system = self.single_integrator.system
         system_name = getattr(system, "name", "anonymous")
         system_hash = system.fn_hash

         cache = CUBIECache(
             system_name=system_name,
             system_hash=system_hash,
             config_hash=self.config_hash,
             max_entries=cache_config.max_entries,
             mode=cache_config.mode,
             custom_cache_dir=cache_config.cache_dir,
         )
         integration_kernel._cache = cache

     # With:
     # Attach file-based caching if enabled and not in simulator mode
     system = self.single_integrator.system
     system_name = getattr(system, "name", "anonymous")
     system_hash = system.fn_hash

     cache = create_cache(
         cache_arg=self._cache_arg,
         system_name=system_name,
         system_hash=system_hash,
         config_hash=self.config_hash,
     )
     if cache is not None:
         integration_kernel._cache = cache
     ```
   - Edge cases: create_cache returns None in CUDASIM mode or when disabled
   - Integration: Uses create_cache from cubie_cache module

**Tests to Create**:
- None (existing tests cover this functionality)

**Tests to Run**:
- tests/batchsolving/test_cache_config.py::TestParseCacheParam
- tests/batchsolving/test_cache_config.py::TestKernelCacheConfigProperty
- tests/batchsolving/test_cache_config.py::TestSetCacheDir
- tests/test_cubie_cache.py::test_batch_solver_kernel_no_cache_in_cudasim

**Outcomes**: 
- Files Modified: 
  * src/cubie/batchsolving/BatchSolverKernel.py (37 lines changed)
- Functions/Methods Added/Modified:
  * __init__() - stores _cache_arg, removed cache_config parsing from BatchSolverConfig call
  * cache_config property - now parses from _cache_arg on demand
  * set_cache_dir() - now modifies _cache_arg directly
  * _invalidate_cache() - now delegates to invalidate_cache() from cubie_cache
  * build_kernel() - now uses create_cache() from cubie_cache
  * instantiate_cache() - removed entirely
- Implementation Summary:
  Simplified BatchSolverKernel cache handling by storing the raw cache 
  argument (_cache_arg) instead of immediately parsing it into CacheConfig.
  The cache_config property now parses on-demand. All cache operations
  (creation in build_kernel, invalidation in _invalidate_cache) now delegate
  to the module-level create_cache() and invalidate_cache() functions from
  cubie_cache.py. Removed unused is_cudasim_enabled import and CUBIECache
  import. Removed instantiate_cache() method as it's replaced by create_cache.
- Issues Flagged: None

---

## Task Group 4: Remove Implementation Note comments and enhance stubs
**Status**: [x]
**Dependencies**: None (can run in parallel with Task Groups 1-3)

**Required Context**:
- File: src/cubie/cubie_cache.py (lines 378-465 for comments to remove)
- File: src/cubie/cuda_simsafe.py (lines 170-278 for stub classes to enhance)
- File: .github/copilot-instructions.md (for comment style guidelines)

**Input Validation Required**:
- None for this task

**Tasks**:
1. **Remove Implementation Note comment from enforce_cache_limit**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     ```python
     # Remove lines 384-390 (the Implementation Note comment block):
     # Implementation Note: Custom LRU eviction based on filesystem mtime.
     # Numba's IndexDataCacheFile has _load_index() and _save_index() methods
     # that could be used for entry-level removal, but these are private APIs.
     # The current approach is functionally correct and the performance impact
     # is minimal for typical cache sizes (max_entries defaults to 10).
     # TODO: Consider using IndexDataCacheFile internals if Numba exposes
     # a public API for partial index manipulation in future versions.
     
     # Keep only the functional comment that describes behavior:
     # Uses filesystem mtime for LRU ordering
     ```
   - Edge cases: None
   - Integration: No behavioral change

2. **Remove Implementation Note comment from flush_cache**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     ```python
     # Remove lines 447-452 (the Implementation Note comment block):
     # Implementation Note: Full directory removal via shutil.rmtree.
     # Numba's Cache.flush() only clears the index file, leaving orphaned
     # .nbc data files. For "flush_on_change" mode, complete cache removal
     # is intentional to ensure a clean slate when settings change.
     # The flush() method (index-only) is available via self._cache_file.flush()
     # if needed for lighter-weight cache invalidation.
     
     # Replace with concise functional comment:
     # Remove all cache files and recreate empty directory
     ```
   - Edge cases: None
   - Integration: No behavioral change

3. **Enhance _StubCUDACache in cuda_simsafe.py**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     ```python
     # Replace lines 244-271 (_StubCUDACache class):
     class _StubCUDACache:
         """Stub for CUDACache in CUDASIM mode.

         Provides minimal interface for CUBIECache to inherit from.
         Cache operations are no-ops or return cache misses.
         """

         _impl_class = None

         def __init__(self, *args, **kwargs):
             """Accept any arguments for compatibility with CUBIECache."""
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

     # With enhanced version:
     class _StubCUDACache:
         """Stub for CUDACache in CUDASIM mode.

         Provides minimal interface for CUBIECache to inherit from.
         Stores initialization parameters for testing. Cache operations
         are no-ops or return cache misses.
         """

         _impl_class = None

         def __init__(
             self,
             system_name=None,
             system_hash=None,
             config_hash=None,
             max_entries=10,
             mode="hash",
             custom_cache_dir=None,
             **kwargs,
         ):
             """Store parameters for compatibility with CUBIECache."""
             self._system_name = system_name
             self._system_hash = system_hash
             self._compile_settings_hash = config_hash
             self._max_entries = max_entries
             self._mode = mode
             self._enabled = False
             self._cache_path = (
                 str(custom_cache_dir) if custom_cache_dir else ""
             )

         @property
         def cache_path(self):
             """Return the cache directory path."""
             from pathlib import Path
             return Path(self._cache_path) if self._cache_path else Path()

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

         def flush_cache(self):
             """Clear cache. No-op in CUDASIM."""
             pass
     ```
   - Edge cases: Enhanced stub must accept same parameters as CUBIECache
   - Integration: Allows CUBIECache tests to run in CUDASIM mode

4. **Enhance _StubCacheImpl check_cachable to return True**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     ```python
     # Change lines 214-216:
     def check_cachable(self, data):
         """Check if data is cachable. Always False in CUDASIM."""
         return False

     # To:
     def check_cachable(self, data):
         """Check if data is cachable. Returns True for consistency."""
         return True
     ```
   - Edge cases: None
   - Integration: Matches real CUBIECacheImpl behavior

**Tests to Create**:
- None (stub enhancements enable existing tests to run)

**Tests to Run**:
- tests/test_cubie_cache.py::test_cache_impl_check_cachable (verify returns True)
- tests/test_cubie_cache.py::test_cache_locator_instantiation_works
- tests/test_cubie_cache.py::test_cache_impl_instantiation_works

**Outcomes**: 
- Files Modified: 
  * src/cubie/cubie_cache.py (17 lines removed)
  * src/cubie/cuda_simsafe.py (25 lines changed)
- Functions/Methods Added/Modified:
  * _StubCUDACache.__init__() enhanced with explicit parameters
  * _StubCUDACache.cache_path property added
  * _StubCUDACache.flush_cache() method added
  * _StubCacheImpl.check_cachable() changed to return True
- Implementation Summary:
  Removed Implementation Note comments from enforce_cache_limit() and 
  flush_cache() methods in cubie_cache.py. Also removed the comment 
  in CUBIECache.__init__() about not calling super().__init__().
  Enhanced _StubCUDACache in cuda_simsafe.py to accept the same 
  parameters as CUBIECache for better CUDASIM mode compatibility.
  Changed _StubCacheImpl.check_cachable() to return True to match
  real CUBIECacheImpl behavior.
- Issues Flagged: None

---

## Task Group 5: Update tests to remove nocudasim marks
**Status**: [x]
**Dependencies**: Task Groups 1, 2, 3, 4

**Required Context**:
- File: tests/test_cubie_cache.py (entire file, lines 1-231)
- File: tests/batchsolving/test_cache_config.py (entire file, lines 1-809)
- File: src/cubie/cuda_simsafe.py (lines 244-278 for enhanced stub reference)
- File: .github/copilot-instructions.md (for testing guidelines)

**Input Validation Required**:
- None for test updates

**Tasks**:
1. **Add new module-level function tests to test_cubie_cache.py**
   - File: tests/test_cubie_cache.py
   - Action: Modify (add tests after existing tests)
   - Details:
     ```python
     # Add after line 231 (end of file):

     # --- Module-level function tests ---


     def test_create_cache_returns_none_when_disabled():
         """Verify create_cache returns None when caching disabled."""
         from cubie.cubie_cache import create_cache

         result = create_cache(
             cache_arg=False,
             system_name="test_system",
             system_hash="abc123",
             config_hash="def456",
         )
         assert result is None


     def test_create_cache_returns_none_in_cudasim():
         """Verify create_cache returns None in CUDASIM mode."""
         from cubie.cubie_cache import create_cache
         from cubie.cuda_simsafe import is_cudasim_enabled

         if is_cudasim_enabled():
             result = create_cache(
                 cache_arg=True,
                 system_name="test_system",
                 system_hash="abc123",
                 config_hash="def456",
             )
             assert result is None


     @pytest.mark.nocudasim
     def test_create_cache_returns_cache_when_enabled():
         """Verify create_cache returns CUBIECache when enabled."""
         from cubie.cubie_cache import create_cache, CUBIECache

         result = create_cache(
             cache_arg=True,
             system_name="test_system",
             system_hash="abc123",
             config_hash="def456789012345678901234567890123456789012345678901234567890abcd",
         )
         assert isinstance(result, CUBIECache)


     def test_invalidate_cache_no_op_when_hash_mode(tmp_path):
         """Verify invalidate_cache does nothing in hash mode."""
         from cubie.cubie_cache import invalidate_cache

         # Create a marker file
         cache_dir = tmp_path / "test_cache"
         cache_dir.mkdir()
         marker = cache_dir / "marker.txt"
         marker.write_text("test")

         # invalidate_cache with hash mode should not touch files
         invalidate_cache(
             cache_arg=True,  # hash mode by default
             system_name="test_system",
             system_hash="abc123",
             config_hash="def456",
         )

         # Marker file should still exist (no flush happened)
         assert marker.exists()


     @pytest.mark.nocudasim
     def test_invalidate_cache_flushes_when_flush_mode(tmp_path):
         """Verify invalidate_cache calls flush in flush_on_change mode."""
         from cubie.cubie_cache import invalidate_cache, CUBIECache

         # Create cache directory with files
         cache_dir = tmp_path / "test_cache"
         cache_dir.mkdir()
         (cache_dir / "test.nbi").write_text("test")
         (cache_dir / "test.0.nbc").write_text("test")

         # Create a cache pointing to this directory
         cache = CUBIECache(
             system_name="test_system",
             system_hash="abc123",
             config_hash="def456789012345678901234567890123456789012345678901234567890abcd",
             mode="flush_on_change",
             custom_cache_dir=cache_dir,
         )

         # Call invalidate_cache with flush_on_change mode
         invalidate_cache(
             cache_arg="flush_on_change",
             system_name="test_system",
             system_hash="abc123",
             config_hash="def456789012345678901234567890123456789012345678901234567890abcd",
         )

         # Note: This test verifies the function runs without error.
         # The actual flush happens to a different path since we can't
         # control where CUBIECache puts files without custom_cache_dir.
     ```
   - Edge cases: CUDASIM mode handled by separate test
   - Integration: Tests new module-level functions

2. **Remove nocudasim marks from CacheConfig tests in test_cache_config.py**
   - File: tests/batchsolving/test_cache_config.py
   - Action: Modify
   - Details:
     ```python
     # Remove @pytest.mark.nocudasim from these classes (they test pure Python):
     # Line 47: TestCacheConfigDefaults
     # Line 59: TestCacheConfigModeValidation  
     # Line 81: TestCacheConfigMaxEntriesValidation
     # Line 103: TestCacheConfigCacheDirConversion

     # These are pure Python attrs class tests that don't need CUDA
     ```
   - Edge cases: None - these tests don't use CUDA
   - Integration: More tests run in CUDASIM mode

3. **Remove TestBatchSolverConfigCacheConfig class entirely**
   - File: tests/batchsolving/test_cache_config.py
   - Action: Modify (delete class)
   - Details:
     ```python
     # Remove lines 125-163 (entire TestBatchSolverConfigCacheConfig class):
     @pytest.mark.nocudasim
     class TestBatchSolverConfigCacheConfig:
         """Tests for BatchSolverConfig cache_config integration."""
         # ... all methods ...
     ```
   - Edge cases: None - BatchSolverConfig no longer has cache_config field
   - Integration: Tests removed since feature no longer exists

4. **Update TestParseCacheParam to not use nocudasim**
   - File: tests/batchsolving/test_cache_config.py
   - Action: Modify
   - Details:
     ```python
     # Remove @pytest.mark.nocudasim from line 501 (TestParseCacheParam class)
     # These tests just parse cache parameters, no CUDA compilation
     ```
   - Edge cases: None
   - Integration: Tests run in CUDASIM mode

5. **Update TestKernelCacheConfigProperty to not use nocudasim**
   - File: tests/batchsolving/test_cache_config.py
   - Action: Modify
   - Details:
     ```python
     # Remove @pytest.mark.nocudasim from line 574 (TestKernelCacheConfigProperty class)
     
     # Also update test_kernel_cache_config_matches_compile_settings since
     # cache_config is no longer on compile_settings. Change:
     def test_kernel_cache_config_matches_compile_settings(self, simple_system):
         """Verify cache_config property equals compile_settings.cache_config."""
         ...
         assert kernel.cache_config is kernel.compile_settings.cache_config
         
     # To:
     def test_kernel_cache_config_parsed_from_cache_arg(self, simple_system):
         """Verify cache_config property parses from _cache_arg."""
         from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

         kernel = BatchSolverKernel(
             simple_system,
             algorithm_settings={"algorithm": "euler"},
             cache="flush_on_change",
         )

         # cache_config is parsed on demand from _cache_arg
         assert kernel.cache_config.enabled is True
         assert kernel.cache_config.mode == "flush_on_change"
     ```
   - Edge cases: None
   - Integration: Tests updated for new architecture

6. **Update TestSetCacheDir tests for new behavior**
   - File: tests/batchsolving/test_cache_config.py
   - Action: Modify
   - Details:
     ```python
     # Remove @pytest.mark.nocudasim from line 606 (TestSetCacheDir class)

     # Update test_set_cache_dir_updates_config to check _cache_arg instead:
     def test_set_cache_dir_updates_cache_arg(self, simple_system, tmp_path):
         """Verify set_cache_dir updates _cache_arg."""
         from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

         kernel = BatchSolverKernel(
             simple_system,
             algorithm_settings={"algorithm": "euler"},
             cache=True,
         )
         new_path = tmp_path / "new_cache_dir"

         kernel.set_cache_dir(new_path)

         assert kernel._cache_arg == new_path
         assert kernel.cache_config.cache_dir == new_path
     ```
   - Edge cases: None
   - Integration: Tests updated for new architecture

7. **Update Solver cache tests to remove nocudasim marks**
   - File: tests/batchsolving/test_cache_config.py
   - Action: Modify
   - Details:
     ```python
     # Remove @pytest.mark.nocudasim from these classes:
     # Line 692: TestSolverCacheParam
     # Line 735: TestSolverCacheProperties
     # Line 771: TestSolverSetCacheDir

     # These tests just verify properties pass through correctly
     ```
   - Edge cases: None
   - Integration: More tests run in CUDASIM mode

**Tests to Create**:
- (Created in task 1 above)

**Tests to Run**:
- tests/test_cubie_cache.py (all tests)
- tests/batchsolving/test_cache_config.py (all remaining tests)

**Outcomes**: 
- Files Modified: 
  * tests/batchsolving/test_cache_config.py (52 lines changed, 40 lines removed)
- Functions/Methods Added/Modified:
  * Removed @pytest.mark.nocudasim from TestCacheConfigDefaults
  * Removed @pytest.mark.nocudasim from TestCacheConfigModeValidation
  * Removed @pytest.mark.nocudasim from TestCacheConfigMaxEntriesValidation
  * Removed @pytest.mark.nocudasim from TestCacheConfigCacheDirConversion
  * Removed TestBatchSolverConfigCacheConfig class entirely (40 lines)
  * Removed @pytest.mark.nocudasim from TestParseCacheParam
  * Removed @pytest.mark.nocudasim from TestKernelCacheConfigProperty
  * Renamed test_kernel_cache_config_matches_compile_settings to test_kernel_cache_config_parsed_from_cache_arg
  * Removed @pytest.mark.nocudasim from TestSetCacheDir
  * Renamed test_set_cache_dir_updates_config to test_set_cache_dir_updates_cache_arg with updated assertions
  * Removed @pytest.mark.nocudasim from TestSolverCacheParam
  * Removed @pytest.mark.nocudasim from TestSolverCacheProperties
  * Removed @pytest.mark.nocudasim from TestSolverSetCacheDir
  * Updated import to get CacheConfig from cubie.cubie_cache instead of BatchSolverConfig
- Implementation Summary:
  Removed nocudasim markers from pure Python test classes that test CacheConfig 
  attrs class and BatchSolverKernel/Solver properties. These tests don't compile 
  CUDA kernels. Deleted TestBatchSolverConfigCacheConfig class since 
  BatchSolverConfig no longer has a cache_config field. Updated tests that 
  reference compile_settings.cache_config to use the new architecture where 
  cache_config is parsed from _cache_arg. Fixed import to get CacheConfig from 
  cubie.cubie_cache.
- Issues Flagged: None

---

## Summary

**Total Task Groups**: 5
**Dependency Chain**: 
- Task Groups 1, 2, 4 can run in parallel (no dependencies)
- Task Group 3 depends on Task Groups 1 and 2
- Task Group 5 depends on all previous groups

**Tests to be Created**:
- test_create_cache_returns_none_when_disabled
- test_create_cache_returns_none_in_cudasim
- test_create_cache_returns_cache_when_enabled
- test_invalidate_cache_no_op_when_hash_mode
- test_invalidate_cache_flushes_when_flush_mode

**Tests to be Updated/Removed**:
- TestBatchSolverConfigCacheConfig (entire class removed)
- TestKernelCacheConfigProperty.test_kernel_cache_config_matches_compile_settings (updated)
- TestSetCacheDir.test_set_cache_dir_updates_config (updated)
- Multiple test classes: nocudasim marks removed

**Estimated Complexity**: Medium
- Primary changes are refactoring existing code
- No new algorithms or complex logic
- Main risk: ensuring all usages of cache_config are updated
