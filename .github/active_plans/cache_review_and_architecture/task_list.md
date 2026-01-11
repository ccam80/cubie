# Implementation Task List
# Feature: Cache Review Comments and Architecture Improvements
# Plan Reference: .github/active_plans/cache_review_and_architecture/agent_plan.md

## Task Group 1: Remove Cache Imports from cuda_simsafe.py
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/cuda_simsafe.py (lines 166-173)
- File: src/cubie/cubie_cache.py (lines 1-30)

**Input Validation Required**:
- None (import restructuring only)

**Tasks**:
1. **Remove caching imports from cuda_simsafe.py**
   - File: src/cubie/cuda_simsafe.py
   - Action: Modify
   - Details:
     ```python
     # DELETE lines 166-173:
     # --- Caching infrastructure ---
     # These classes work in both CUDA and CUDASIM modes
     from numba.cuda.core.caching import (
         _CacheLocator,
         CacheImpl,
         IndexDataCacheFile,
     )
     from cubie.vendored.numba_cuda_cache import Cache
     ```
   - Edge cases: None - straightforward removal
   - Integration: cubie_cache.py is the only consumer of these imports

2. **Update cubie_cache.py imports to use direct sources**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     Replace lines 22-27:
     ```python
     # OLD:
     from cubie.cuda_simsafe import (
         _CacheLocator,
         Cache,
         CacheImpl,
         IndexDataCacheFile,
     )
     
     # NEW:
     from numba.cuda.core.caching import (
         _CacheLocator,
         CacheImpl,
         IndexDataCacheFile,
     )
     from cubie.vendored.numba_cuda_cache import Cache
     ```
   - Edge cases: None
   - Integration: These imports are used by CUBIECacheLocator, CUBIECacheImpl, and CUBIECache classes

**Tests to Create**:
- None (no new tests needed - existing import tests cover this)

**Tests to Run**:
- tests/test_cubie_cache.py::test_cache_locator_instantiation_works
- tests/test_cubie_cache.py::test_cache_impl_instantiation_works

**Outcomes**:
- Files Modified: 
  * src/cubie/cuda_simsafe.py (8 lines removed)
  * src/cubie/cubie_cache.py (6 lines changed)
- Functions/Methods Added/Modified:
  * None (import restructuring only)
- Implementation Summary:
  Removed caching infrastructure imports from cuda_simsafe.py as they are not "simsafe" (they work in both CUDA and CUDASIM modes). Updated cubie_cache.py to import directly from numba.cuda.core.caching and cubie.vendored.numba_cuda_cache instead of through cuda_simsafe.
- Issues Flagged: None

---

## Task Group 2: Fix Docstring and Remove Commented Code
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/cubie_cache.py (lines 310-325, 470-495)

**Input Validation Required**:
- None (documentation changes only)

**Tasks**:
1. **Remove commented-out code block from CUBIECache.__init__**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     Delete lines 318-323 (the commented-out block):
     ```python
     # DELETE these lines:
         # Caching not available in CUDA simulator mode
         # if not _CACHING_AVAILABLE:
         #     raise RuntimeError(
         #         "CUBIECache is not available in CUDA simulator mode. "
         #         "File-based caching requires a real CUDA environment."
         #     )
     ```
     The line `) -> None:` on line 317 should be immediately followed by
     `self._system_name = system_name` (currently line 325).
   - Edge cases: None
   - Integration: No functional change

2. **Fix docstring in create_cache function**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     Replace lines 479-481:
     ```python
     # OLD:
     CUBIECache or None
         CUBIECache instance if caching enabled and not in CUDASIM mode,
         None otherwise.
     
     # NEW:
     CUBIECache or None
         CUBIECache instance if caching is enabled, None otherwise.
     ```
   - Edge cases: None
   - Integration: Documentation only - no functional change

**Tests to Create**:
- None (documentation changes)

**Tests to Run**:
- tests/test_cubie_cache.py::test_create_cache_returns_cache_in_cudasim

**Outcomes**:
- Files Modified: 
  * src/cubie/cubie_cache.py (7 lines removed, 1 line changed)
- Functions/Methods Added/Modified:
  * CUBIECache.__init__() - removed 6 lines of commented-out code
  * create_cache() - updated docstring Returns section
- Implementation Summary:
  Removed obsolete commented-out code that checked for CUDASIM mode in CUBIECache.__init__. Updated the create_cache docstring to remove the outdated reference to CUDASIM mode, now correctly stating that it returns a CUBIECache instance if caching is enabled.
- Issues Flagged: None

---

## Task Group 3: Simplify invalidate_cache Function
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: src/cubie/cubie_cache.py (lines 70-104, 120-148, 497-538)
- File: src/cubie/odesystems/symbolic/odefile.py (GENERATED_DIR location)

**Input Validation Required**:
- custom_cache_dir: Optional[Path] - validate with `val.optional(val.instance_of((str, Path)))` if provided

**Tasks**:
1. **Refactor invalidate_cache to compute path directly**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     Replace the entire invalidate_cache function (lines 497-538):
     ```python
     def invalidate_cache(
         cache_arg: Union[bool, str, Path],
         system_name: str,
         system_hash: str,
         config_hash: str,
         custom_cache_dir: Optional[Path] = None,
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
         custom_cache_dir
             Optional custom cache directory path for testing.

         Notes
         -----
         Only flushes cache when mode is "flush_on_change". Silent on errors
         since cache flush is best-effort.
         """
         cache_config = CacheConfig.from_user_setting(cache_arg)
         if not cache_config.enabled:
             return
         if cache_config.mode != "flush_on_change":
             return

         # Compute cache path directly without creating CUBIECache
         if custom_cache_dir is not None:
             cache_path = Path(custom_cache_dir)
         elif cache_config.cache_dir is not None:
             cache_path = Path(cache_config.cache_dir)
         else:
             cache_path = GENERATED_DIR / system_name / "CUDA_cache"

         # Best-effort flush
         try:
             if cache_path.exists():
                 import shutil
                 shutil.rmtree(cache_path)
         except OSError:
             pass
     ```
   - Edge cases:
     - cache_path doesn't exist: Check with `exists()` before rmtree
     - Permission errors: Catch OSError silently (best-effort)
     - custom_cache_dir takes priority over cache_config.cache_dir
   - Integration: The new signature adds optional custom_cache_dir param for testing

**Tests to Create**:
- None in this group (tests updated in Task Group 4)

**Tests to Run**:
- tests/test_cubie_cache.py::test_invalidate_cache_no_op_when_hash_mode

**Outcomes**:
- Files Modified: 
  * src/cubie/cubie_cache.py (48 lines changed - replaced invalidate_cache function)
- Functions/Methods Added/Modified:
  * invalidate_cache() in cubie_cache.py - refactored to compute path directly
- Implementation Summary:
  Refactored invalidate_cache function to compute the cache path directly using GENERATED_DIR / system_name / "CUDA_cache" instead of creating a full CUBIECache object just to call flush_cache(). Added optional custom_cache_dir parameter for testing. Uses shutil.rmtree directly if path exists, with silent OSError handling for best-effort flush behavior.
- Issues Flagged: None

---

## Task Group 4: Improve Test Assertions
**Status**: [x]
**Dependencies**: Task Group 3

**Required Context**:
- File: tests/test_cubie_cache.py (lines 1-60, 186-201, 308-320)
- File: src/cubie/cubie_cache.py (invalidate_cache function signature from Task Group 3)

**Input Validation Required**:
- None (test file changes only)

**Tasks**:
1. **Improve test_batch_solver_kernel_cache_in_cudasim with assertion**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     Replace lines 186-200:
     ```python
     def test_batch_solver_kernel_builds_without_error(solverkernel):
         """Verify BatchSolverKernel builds successfully in all modes.

         This smoke test confirms the kernel compilation path executes
         without raising exceptions, regardless of CUDA/CUDASIM mode.
         """
         # Build the kernel - this exercises the full compilation path
         kernel = solverkernel.kernel

         # Verify kernel was created
         assert kernel is not None
     ```
   - Edge cases: None
   - Integration: Rename test function for clarity, add meaningful assertion

2. **Improve test_invalidate_cache_flushes_when_flush_mode with assertions**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     Replace lines 308-320:
     ```python
     def test_invalidate_cache_flushes_when_flush_mode(tmp_path):
         """Verify invalidate_cache flushes cache in flush_on_change mode."""
         from cubie.cubie_cache import invalidate_cache

         # Create cache directory with marker file
         cache_dir = tmp_path / "test_cache"
         cache_dir.mkdir(parents=True, exist_ok=True)
         marker_file = cache_dir / "marker.txt"
         marker_file.write_text("test")
         assert marker_file.exists()

         # invalidate_cache should remove the cache contents
         invalidate_cache(
             cache_arg="flush_on_change",
             system_name="test_system",
             system_hash="abc123",
             config_hash="def456789012345678901234567890123456"
             "789012345678901234567890abcd",
             custom_cache_dir=cache_dir,
         )

         # Verify cache was flushed (directory removed)
         assert not cache_dir.exists()
     ```
   - Edge cases: None
   - Integration: Uses new custom_cache_dir parameter from Task Group 3

**Tests to Create**:
- None (modifying existing tests)

**Tests to Run**:
- tests/test_cubie_cache.py::test_batch_solver_kernel_builds_without_error
- tests/test_cubie_cache.py::test_invalidate_cache_flushes_when_flush_mode

**Outcomes**:
- Files Modified: 
  * tests/test_cubie_cache.py (22 lines changed)
- Functions/Methods Added/Modified:
  * test_batch_solver_kernel_builds_without_error() - renamed from test_batch_solver_kernel_cache_in_cudasim, added assertion
  * test_invalidate_cache_flushes_when_flush_mode() - added marker file creation and verification assertions
- Implementation Summary:
  Improved two tests with meaningful assertions. First test was renamed to test_batch_solver_kernel_builds_without_error and now includes an `assert kernel is not None` assertion. Second test now creates a cache directory with a marker file in tmp_path, calls invalidate_cache with the custom_cache_dir parameter, and asserts the cache directory was removed.
- Issues Flagged: None

---

## Summary

### Total Task Groups: 4

### Dependency Chain Overview:
```
Task Group 1 (Import Cleanup)
    ↓
Task Group 2 (Docstring/Comments)
    ↓
Task Group 3 (Simplify invalidate_cache)
    ↓
Task Group 4 (Test Improvements)
```

### Tests to be Created: 0
### Tests to be Modified: 2
- `test_batch_solver_kernel_cache_in_cudasim` → renamed to `test_batch_solver_kernel_builds_without_error`
- `test_invalidate_cache_flushes_when_flush_mode`

### Tests to be Run:
1. `tests/test_cubie_cache.py::test_cache_locator_instantiation_works`
2. `tests/test_cubie_cache.py::test_cache_impl_instantiation_works`
3. `tests/test_cubie_cache.py::test_create_cache_returns_cache_in_cudasim`
4. `tests/test_cubie_cache.py::test_invalidate_cache_no_op_when_hash_mode`
5. `tests/test_cubie_cache.py::test_batch_solver_kernel_builds_without_error`
6. `tests/test_cubie_cache.py::test_invalidate_cache_flushes_when_flush_mode`

### Estimated Complexity: Low
- All changes are localized to 3 files
- No new functionality, only refactoring and test improvements
- No CUDA-specific logic changes
