# Agent Implementation Plan: Refactor CuBIE Caching for CUDASIM Compatibility

## Overview

This plan refactors the CuBIE caching module to work in CUDASIM mode by:
1. Removing CUDASIM stub classes from `cuda_simsafe.py`
2. Vendoring the numba `Cache` class for CUDASIM compatibility
3. Updating `cubie_cache.py` to use vendored/imported classes consistently
4. Removing `nocudasim` markers from cache tests

---

## Component 1: Vendored Cache Module

### Purpose
Provide a `Cache` base class that works in both CUDA and CUDASIM modes, since `CUDACache` from `numba.cuda.dispatcher` is not available in CUDASIM.

### Location
`src/cubie/vendored/numba_cuda_cache.py`

### Behavior
- Contains the `Cache` class vendored from `numba.cuda.core.caching`
- Must be marked with a comment: `# Vendored from numba-cuda on 2026-01-11`
- The `Cache` class orchestrates:
  - Creating a `CacheImpl` subclass instance
  - Managing an `IndexDataCacheFile` for index/data operations
  - Providing `load_overload` and `save_overload` methods
  - Providing `enable`, `disable`, and `flush` methods

### Dependencies
- `numba.cuda.core.caching.IndexDataCacheFile` (imports successfully in CUDASIM)
- `numba.cuda.core.caching.CacheImpl` (imports successfully in CUDASIM)
- `numba.cuda.core.caching._CacheLocator` (imports successfully in CUDASIM)

### Key Difference from numba's Cache
- CuBIE's `CUBIECache` doesn't use `py_func` initialization (uses system info instead)
- CuBIE already overrides `__init__` and `_index_key` in `CUBIECache`
- The vendored `Cache` only needs to provide the base structure

---

## Component 2: Updated cuda_simsafe.py

### Changes Required
1. **Remove** all `_Stub*` caching classes:
   - `_StubCacheLocator`
   - `_StubCacheImpl`
   - `_StubIndexDataCacheFile`
   - `_StubCUDACache`

2. **Remove** the conditional import block for caching that switches between stubs and real classes

3. **Add** unconditional imports from `numba.cuda.core.caching`:
   - `_CacheLocator`
   - `CacheImpl`
   - `IndexDataCacheFile`

4. **Add** import from vendored module:
   - `Cache` from `cubie.vendored.numba_cuda_cache`

5. **Update** `__all__` to reflect removed/added exports

6. **Remove** `_CACHING_AVAILABLE` flag (caching is always available now)

### Imports After Refactoring
```python
# Caching infrastructure - works in both CUDA and CUDASIM modes
from numba.cuda.core.caching import (
    _CacheLocator,
    CacheImpl,
    IndexDataCacheFile,
)
from cubie.vendored.numba_cuda_cache import Cache
```

---

## Component 3: Updated cubie_cache.py

### Changes Required
1. **Update imports** to use the consistent classes from `cuda_simsafe`:
   - Import `Cache` (vendored) instead of `CUDACache`
   - Continue importing `_CacheLocator`, `CacheImpl`, `IndexDataCacheFile`

2. **Update** `CUBIECache` to inherit from `Cache` instead of `CUDACache`

3. **Remove** the `create_cache` function's CUDASIM check:
   - Currently returns `None` when `is_cudasim_enabled()`
   - Should return a valid `CUBIECache` in CUDASIM mode
   - The cache will function but won't have compiled kernels to cache

4. **Remove** the `invalidate_cache` function's CUDASIM early return

5. **Update** `CUBIECache.__init__` to properly initialize from `Cache` base

### Expected Behavior
- `CUBIECache` works identically in both modes
- In CUDASIM mode, `save_overload` will be called but actual kernel data won't exist
- LRU eviction, flush operations, path management all work normally

---

## Component 4: Test Updates

### Files to Modify
1. `tests/test_cubie_cache.py`
2. `tests/batchsolving/test_cache_config.py`

### Changes Required

#### tests/test_cubie_cache.py
Remove `@pytest.mark.nocudasim` from:
- Line 131: `test_cubie_cache_init`
- Line 144: `test_cubie_cache_index_key`
- Line 174: `test_cubie_cache_path`

Update tests that check for `None` in CUDASIM mode:
- `test_create_cache_returns_none_in_cudasim` - should now return a valid cache
- `test_batch_solver_kernel_no_cache_in_cudasim` - update expected behavior

#### tests/batchsolving/test_cache_config.py
Remove `@pytest.mark.nocudasim` from these test classes:
- Line 120: `TestCUBIECacheMaxEntries`
- Line 140: `TestEnforceCacheLimitNoEviction`
- Line 175: `TestEnforceCacheLimitEviction`
- Line 223: `TestEnforceCacheLimitDisabled`
- Line 256: `TestEnforceCacheLimitPairs`
- Line 306: `TestCUBIECacheModeStored`
- Line 336: `TestFlushCacheRemovesFiles`
- Line 369: `TestFlushCacheRecreatesDirectory`
- Line 419: `TestCustomCacheDir`

### Expected Test Failures
Some tests may fail in CUDASIM mode if they depend on:
- Actual kernel compilation producing `.nbc` files
- CUDA context operations like `target_context.refresh()`

These failures are expected and document the limitation that:
> "In CUDASIM mode, caching infrastructure works but no compiled files are produced"

---

## Component 5: Package Init for Vendored Module

### Purpose
Create the vendored package directory structure

### Location
`src/cubie/vendored/__init__.py`

### Content
Empty or minimal docstring indicating this package contains vendored code.

---

## Integration Points

### With BatchSolverKernel
- `BatchSolverKernel` uses `create_cache()` to get a cache instance
- After refactoring, it will get a valid `CUBIECache` in CUDASIM mode
- The kernel compilation path in CUDASIM doesn't produce cacheable data, so save operations will effectively be no-ops for the actual kernel data

### With Solver
- `Solver` exposes cache configuration through `kernel.cache_config`
- No changes needed to Solver itself

### With create_cache function
- Must return a valid `CUBIECache` in CUDASIM mode
- Tests should be updated to expect a cache object, not `None`

---

## Edge Cases

1. **CUBIECacheImpl.reduce() in CUDASIM**
   - Currently calls `kernel._reduce_states()`
   - In CUDASIM, the kernel may not have this method
   - May need to handle `AttributeError` gracefully or let test fail to document limitation

2. **CUBIECacheImpl.rebuild() in CUDASIM**
   - Imports `_Kernel` from `numba.cuda.dispatcher`
   - This import fails in CUDASIM mode
   - May need conditional import or let test fail to document limitation

3. **load_overload with target_context**
   - Base `Cache.load_overload` calls `target_context.refresh()`
   - In CUDASIM, this may not work correctly
   - `CUBIECache` may need to override to handle CUDASIM

---

## Validation Approach

After implementation:
1. Run `NUMBA_ENABLE_CUDASIM=1 pytest tests/test_cubie_cache.py -v`
2. Run `NUMBA_ENABLE_CUDASIM=1 pytest tests/batchsolving/test_cache_config.py -v`
3. Verify tests pass or fail only for documented CUDASIM limitations
4. Document any tests that fail due to missing compiled kernel files

---

## File Summary

| File | Action |
|------|--------|
| `src/cubie/vendored/__init__.py` | CREATE - Package init |
| `src/cubie/vendored/numba_cuda_cache.py` | CREATE - Vendored Cache class |
| `src/cubie/cuda_simsafe.py` | MODIFY - Remove stubs, update imports |
| `src/cubie/cubie_cache.py` | MODIFY - Update inheritance, remove CUDASIM checks |
| `tests/test_cubie_cache.py` | MODIFY - Remove nocudasim markers, update expectations |
| `tests/batchsolving/test_cache_config.py` | MODIFY - Remove nocudasim markers |
