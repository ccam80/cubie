# Agent Plan: Cache Review Comments and Architecture

## Overview

This plan addresses PR #485 review comments for the CuBIE caching module. The
work involves:

1. Addressing 6 specific review comments
2. Analyzing cache-flushing architecture
3. Making targeted code improvements

## Architecture Analysis Result

**No vendoring of IndexDataCacheFile is needed.** The class imports successfully
in CUDASIM mode (unlike the Cache class which required vendoring). The current
`invalidate_cache()` function has questionable logic that can be simplified
without additional vendoring.

---

## Component Changes

### Component 1: cuda_simsafe.py - Remove Non-Simsafe Imports

**Review Comment ID**: 2679282108

**Current State**:
```python
# Lines 166-173
# --- Caching infrastructure ---
# These classes work in both CUDA and CUDASIM modes
from numba.cuda.core.caching import (
    _CacheLocator,
    CacheImpl,
    IndexDataCacheFile,
)
from cubie.vendored.numba_cuda_cache import Cache
```

**Problem**: These caching imports are not "simsafe" utilities. They belong in
`cubie_cache.py` which is their actual consumer.

**Required Change**: Remove the caching imports from `cuda_simsafe.py`. Update
`cubie_cache.py` to import directly from their sources.

**Files Affected**:
- `src/cubie/cuda_simsafe.py` - Remove imports and __all__ entries
- `src/cubie/cubie_cache.py` - Update imports

---

### Component 2: cubie_cache.py - Fix Docstring

**Review Comment ID**: 2679186567

**Current State** (line 480-481):
```python
Returns
-------
CUBIECache or None
    CUBIECache instance if caching enabled and not in CUDASIM mode,
    None otherwise.
```

**Problem**: Docstring is outdated. With vendored Cache class, caching works in
CUDASIM mode too.

**Required Change**: Update docstring to:
```python
Returns
-------
CUBIECache or None
    CUBIECache instance if caching is enabled, None otherwise.
```

**Files Affected**:
- `src/cubie/cubie_cache.py`

---

### Component 3: cubie_cache.py - Remove Commented Code

**Identified via Review/Greptile**

**Current State** (lines 318-323):
```python
    ) -> None:
        # Caching not available in CUDA simulator mode
        # if not _CACHING_AVAILABLE:
        #     raise RuntimeError(
        #         "CUBIECache is not available in CUDA simulator mode. "
        #         "File-based caching requires a real CUDA environment."
        #     )
```

**Problem**: Commented-out code references removed `_CACHING_AVAILABLE` flag.

**Required Change**: Remove the commented-out block entirely.

**Files Affected**:
- `src/cubie/cubie_cache.py`

---

### Component 4: cubie_cache.py - Fix invalidate_cache Logic

**Review Comment ID**: 2679290544

**Current State** (line 528):
```python
def invalidate_cache(...) -> None:
    ...
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
        pass
```

**Problem**: Creates a full CUBIECache object just to immediately flush it.
This is wasteful and confusing. The CUBIECache constructor also creates the
cache directory, which is then immediately cleared.

**Required Change**: Simplify to directly compute and clear the cache path:
```python
def invalidate_cache(...) -> None:
    cache_config = CacheConfig.from_user_setting(cache_arg)
    if not cache_config.enabled:
        return
    if cache_config.mode != "flush_on_change":
        return

    # Compute cache path directly without creating CUBIECache
    if cache_config.cache_dir is not None:
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

**Files Affected**:
- `src/cubie/cubie_cache.py`

---

### Component 5: test_cubie_cache.py - Add Test Assertions

**Review Comment ID**: 2679186566

**Current State** (line 320):
```python
def test_invalidate_cache_flushes_when_flush_mode(tmp_path):
    """Verify invalidate_cache calls flush in flush_on_change mode."""
    from cubie.cubie_cache import invalidate_cache

    # In both CUDA and CUDASIM modes, invalidate_cache should work
    invalidate_cache(
        cache_arg="flush_on_change",
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456789012345678901234567890123456"
        "789012345678901234567890abcd",
    )
    # No assertion needed - just verify it runs without error
```

**Problem**: Test has no meaningful assertions.

**Required Change**: Add assertions that verify flush actually occurs:
```python
def test_invalidate_cache_flushes_when_flush_mode(tmp_path):
    """Verify invalidate_cache flushes cache in flush_on_change mode."""
    from cubie.cubie_cache import invalidate_cache, CUBIECache

    # Create a cache with a marker file
    cache = CUBIECache(
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456789...",
        custom_cache_dir=tmp_path,
    )
    marker_file = cache.cache_path / "marker.txt"
    cache.cache_path.mkdir(parents=True, exist_ok=True)
    marker_file.write_text("test")
    assert marker_file.exists()

    # invalidate_cache should remove the cache contents
    invalidate_cache(
        cache_arg="flush_on_change",
        system_name="test_system",
        system_hash="abc123",
        config_hash="def456789...",
        custom_cache_dir=tmp_path,  # Note: need to add this parameter
    )

    # Verify cache was flushed
    assert not marker_file.exists()
```

**Note**: The current `invalidate_cache` function doesn't accept
`custom_cache_dir`. This test improvement requires either:
1. Adding custom_cache_dir parameter to invalidate_cache, or
2. Testing against the default GENERATED_DIR location

**Files Affected**:
- `tests/test_cubie_cache.py`

---

### Component 6: test_cubie_cache.py - Fix Inconsistent Test

**Review Comment ID**: 2679186570

**Current State** (lines 186-200):
```python
def test_batch_solver_kernel_cache_in_cudasim(solverkernel):
    """Verify cache behavior in CUDASIM mode.

    With vendored caching infrastructure, a cache object can be
    attached in CUDASIM mode. The cache will function for
    infrastructure operations (eviction, flush) but no compiled
    kernel files are produced since compilation doesn't occur.
    """
    from cubie.cuda_simsafe import is_cudasim_enabled

    # Build the kernel to trigger cache attachment logic
    kernel = solverkernel.kernel

    # In both modes, cache behavior depends on caching_enabled setting
    # which is tested separately. This test just verifies no errors.
```

**Problem**: Test name and docstring claim to verify cache behavior, but test
has no assertions.

**Required Change**: Either:
1. Add meaningful assertions about the kernel/cache, or
2. Rename to clarify it's a smoke test

Recommended approach - add minimal but meaningful assertions:
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

**Files Affected**:
- `tests/test_cubie_cache.py`

---

## Integration Points

### Import Dependencies After Changes

**cuda_simsafe.py** will no longer export:
- `_CacheLocator`
- `CacheImpl`
- `IndexDataCacheFile`
- `Cache`

**cubie_cache.py** will import directly:
- `from numba.cuda.core.caching import _CacheLocator, CacheImpl, IndexDataCacheFile`
- `from cubie.vendored.numba_cuda_cache import Cache`

### Edge Cases

1. **Custom cache directory**: The simplified `invalidate_cache` must handle
   both custom and default cache paths
2. **Cache path doesn't exist**: rmtree should not raise if path missing
3. **Permission errors**: Maintain best-effort semantics with try/except

---

## Expected Behavior After Changes

1. **cuda_simsafe.py**: Pure CUDA simulation utilities, no caching imports
2. **cubie_cache.py**: All caching logic consolidated here with correct imports
3. **invalidate_cache()**: Simple, direct path calculation and flush
4. **Tests**: All have meaningful assertions verifying actual behavior

---

## Dependencies

- `numba.cuda.core.caching` - For _CacheLocator, CacheImpl, IndexDataCacheFile
- `cubie.vendored.numba_cuda_cache` - For Cache base class
- `cubie.odesystems.symbolic.odefile` - For GENERATED_DIR
- `shutil` - For rmtree in invalidate_cache
- `pathlib.Path` - For path operations
