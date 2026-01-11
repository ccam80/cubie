# Agent Plan: CUDASIM Caching Refactor

## Overview

This plan refactors the CuBIE caching module to achieve CUDASIM compatibility by vendoring necessary numba-cuda caching classes rather than using minimal stub implementations.

## Components to Modify

### 1. cuda_simsafe.py - Caching Infrastructure Section

**Current State**: Contains stub classes (`_StubCacheLocator`, `_StubCacheImpl`, `_StubIndexDataCacheFile`, `_StubCUDACache`) that provide minimal no-op implementations in CUDASIM mode.

**Target State**: Contains vendored versions of numba-cuda caching classes that work in both modes:
- `IndexDataCacheFile` - Vendored from numba-cuda, handles file I/O for cache
- `_CacheLocator` - Abstract base class for locating cache files
- `CacheImpl` - Abstract base for serialization logic
- `Cache` - Base cache class with load/save operations

**Expected Behavior**:
- Classes should be fully functional for filesystem operations
- Serialization methods (`reduce`, `rebuild`) will fail when actual CUDA kernels are involved, which is expected
- The `dumps` function from `numba.cuda.serialize` should fallback to `pickle.dumps` in CUDASIM mode

**Integration Points**:
- `cubie_cache.py` imports these classes and extends them
- Tests import via `cubie_cache.py` which re-exports from `cuda_simsafe.py`

### 2. cubie_cache.py - Simplified Wrapper

**Current State**: Imports caching classes from `cuda_simsafe.py` which provides stubs in CUDASIM mode.

**Target State**: 
- Remove the `is_cudasim_enabled()` check in `create_cache()` that returns `None`
- Let the cache infrastructure be testable in CUDASIM mode
- Keep the check only where truly necessary (kernel serialization)

**Expected Behavior**:
- `CacheConfig` works identically in both modes
- `CUBIECacheLocator` creates paths and returns stamps in both modes
- `CUBIECacheImpl` can be instantiated; `reduce`/`rebuild` may fail on real kernels
- `CUBIECache` can be instantiated and perform file operations

### 3. test_cubie_cache.py - Remove CUDASIM Markers

**Current State**: Three tests marked with `@pytest.mark.nocudasim`:
- `test_cubie_cache_init` (line 131)
- `test_cubie_cache_index_key` (line 144)
- `test_cubie_cache_path` (line 174)

**Target State**: 
- Remove all `nocudasim` markers from cache tests
- Tests that exercise filesystem operations should pass
- Tests that require actual kernel compilation may fail in CUDASIM (expected)

## Architectural Changes

### Vendored Classes Structure

The vendored classes should be placed in `cuda_simsafe.py` under a clearly marked section:

```
# --- Vendored numba-cuda caching classes (2026-01-11) ---
# Source: numba_cuda/numba/cuda/core/caching.py
# These classes are vendored to enable CUDASIM compatibility.
# The original implementations depend on imports unavailable in CUDASIM mode.
```

### Serialization Fallback

The vendored `IndexDataCacheFile` uses `numba.cuda.serialize.dumps` for serialization. This import will fail in CUDASIM mode, so a fallback is needed:

```python
try:
    from numba.cuda.serialize import dumps
except ImportError:
    import pickle
    def dumps(obj):
        return pickle.dumps(obj)
```

### Cache Base Class

The vendored `Cache` class (`_Cache` in numba-cuda) provides the base implementation that `CUDACache` (also vendored) extends. Key methods:
- `enable()` / `disable()` - Toggle caching
- `load_overload()` / `save_overload()` - Load/save operations
- `flush()` - Clear the cache

## Dependencies

### Required Imports in Vendored Code
- `os`, `pickle`, `hashlib`, `uuid`, `contextlib`, `errno` - Standard library
- `numba` - For `numba.__version__` in cache versioning
- `pathlib.Path` - For path operations

### Removed Dependencies
- `numba.cuda.misc.appdirs` - Not needed, CuBIE uses custom paths
- `numba.cuda.core.config` - Not needed, CuBIE has own config

## Edge Cases

### 1. No CUDA Context in CUDASIM
- The `load_overload` method calls `target_context.refresh()` 
- In CUDASIM, there is no real context, but this call should not fail
- If it does, wrap with try/except

### 2. Kernel Serialization Failure
- `reduce()` calls `kernel._reduce_states()` which requires a real compiled kernel
- In CUDASIM, this will fail
- Tests should expect this failure when testing actual save operations

### 3. File Permission Errors on Windows
- The vendored code includes `_guard_against_spurious_io_errors` for Windows
- This should be preserved

## Data Structures

### IndexDataCacheFile
Manages index (`.nbi`) and data (`.nbc`) files:
- `_cache_path` - Directory for cache files
- `_index_name` - Name of index file (e.g., `system-disambig.nbi`)
- `_source_stamp` - Freshness indicator
- `_version` - Numba version for compatibility

### CacheImpl
Abstract implementation class:
- `_locator_classes` - List of locator classes to try (CuBIE overrides with empty list)
- `_locator` - The chosen locator instance
- `_filename_base` - Base name for cache files

## Test Expectations After Refactor

| Test | CUDASIM Expected Result |
|------|------------------------|
| `test_cache_locator_get_cache_path` | PASS |
| `test_cache_locator_get_source_stamp` | PASS |
| `test_cache_locator_get_disambiguator` | PASS |
| `test_cache_locator_from_function_raises` | PASS |
| `test_cache_impl_locator_property` | PASS |
| `test_cache_impl_filename_base` | PASS |
| `test_cache_impl_check_cachable` | PASS |
| `test_cubie_cache_init` | PASS (after removing nocudasim) |
| `test_cubie_cache_index_key` | PASS (after removing nocudasim) |
| `test_cubie_cache_path` | PASS (after removing nocudasim) |
| `test_batch_solver_kernel_no_cache_in_cudasim` | PASS |
| `test_cache_locator_instantiation_works` | PASS |
| `test_cache_impl_instantiation_works` | PASS |
| `test_create_cache_returns_none_when_disabled` | PASS |
| `test_create_cache_returns_none_in_cudasim` | NEEDS UPDATE - may now return cache |
| `test_create_cache_returns_cache_when_enabled` | PASS |
| `test_invalidate_cache_no_op_when_hash_mode` | PASS |
| `test_invalidate_cache_flushes_when_flush_mode` | PASS |

## Implementation Notes

### Vendoring Strategy
1. Copy relevant classes from `active_plans/numba_cuda_caching.py`
2. Remove dependencies on unavailable imports
3. Simplify to only what CuBIE needs
4. Mark with vendor date comment

### Classes to Vendor
From `numba_cuda_caching.py`:
- `IndexDataCacheFile` (lines 95-220) - File I/O operations
- `_CacheLocator` (lines 356-410) - Abstract locator base
- `Cache` (lines 222-354) - Main cache implementation

From `numba_cuda_dispatcher.py`:
- `CUDACache` pattern - Extend `Cache` with CUDA-specific behavior

### What NOT to Vendor
- All the specific locator implementations (`_InTreeCacheLocator`, etc.)
- CuBIE uses `CUBIECacheLocator` which has custom path logic
- IPython/Jupyter specific code

## Iteration Plan

1. **Step 1**: Add vendored classes to `cuda_simsafe.py`
   - Add serialization fallback
   - Add `IndexDataCacheFile`
   - Add `_CacheLocator` abstract base
   - Add `Cache` base class
   - Keep existing imports for non-CUDASIM mode

2. **Step 2**: Update imports and remove stubs
   - Remove `_StubCacheLocator`, `_StubCacheImpl`, `_StubIndexDataCacheFile`, `_StubCUDACache`
   - Update the conditional import section to use vendored classes for both modes

3. **Step 3**: Update `cubie_cache.py`
   - Remove CUDASIM early return in `create_cache()` if appropriate
   - Ensure all cache operations work up to serialization

4. **Step 4**: Update tests
   - Remove `@pytest.mark.nocudasim` from cache tests
   - Update `test_create_cache_returns_none_in_cudasim` expectation
   - Run tests, identify failures

5. **Step 5**: Iterate on failures
   - For each import failure, vendor the needed module
   - For each runtime failure, add appropriate error handling
   - Leave tests that fail due to no compiled file as expected failures
