# CuBIE Caching Implementation Refactor - Agent Plan

## Overview

This plan describes the refactoring of CuBIE's caching implementation to:
1. Clean separation of cache ownership from compile settings
2. CUDASIM mode compatibility for cache operations
3. Better integration with Numba-CUDA's caching infrastructure

---

## Task 1: Decontaminate BatchSolverKernel/BatchSolverConfig

### Current State

**Problem**: CacheConfig is currently part of the compile settings flow, but
caching is not a compile-time concern. The current code has:

1. `BatchSolverKernel.py` line 36 imports CacheConfig from BatchSolverConfig,
   but CacheConfig is defined in cubie_cache.py (import error exists)
2. `BatchSolverKernel.py` line 174 calls `CacheConfig(cache)` with a bool/str/Path
   but CacheConfig expects keyword arguments (enabled, mode, etc.)
3. Cache is created at build time in `build_kernel()` and attached to dispatcher
4. `cache_config` property delegates to `compile_settings.cache_config` but
   BatchSolverConfig doesn't have a cache_config field

### Target State

1. BatchSolverKernel instantiates a CUBIECache (or None) at `__init__` time
2. BatchSolverKernel stores the cache as `self._cache`
3. All cache-related properties/methods delegate to `self._cache`
4. BatchSolverConfig has no cache-related fields
5. CacheConfig remains in cubie_cache.py

### Components to Modify

#### cubie_cache.py

**CacheConfig class**:
- Keep in cubie_cache.py (no move needed)
- Fix `from_user_setting()` method (currently references `cache_path` but
  field is `cache_dir`)
- Ensure it works standalone (no CUDAFactory dependency issues)

**CUBIECache class**:
- Add `config` property returning CacheConfig instance
- Store CacheConfig internally as `self._config`
- Constructor should accept either individual parameters or a CacheConfig
- Add factory method `from_user_setting()` that creates CUBIECache from
  user's cache parameter (True/False/str/Path)

#### BatchSolverKernel.py

**Initialization changes**:
- Remove `self.cache_config = CacheConfig(cache)` at line 174
- Add cache instantiation logic in `__init__`:
  ```python
  self._cache = self._create_cache(cache, system)
  ```
- Create `_create_cache()` helper method that:
  - Returns None if cache=False or cache=None
  - Returns None if is_cudasim_enabled() (for now, Task 2 may change)
  - Creates and returns CUBIECache instance otherwise

**Property changes**:
- `cache_config` property should return `self._cache.config if self._cache else None`
- Add `cache` property returning `self._cache`
- Add `cache_enabled` property returning `self._cache is not None`

**Method changes**:
- `set_cache_dir()` should delegate to `self._cache.set_cache_dir(path)` if cache exists
- `_invalidate_cache()` should call `self._cache.flush_cache()` if mode is "flush_on_change"
- Remove `instantiate_cache()` method (cache is now created at init)
- `build_kernel()` should attach `self._cache` to the dispatcher instead of
  creating a new cache instance

#### BatchSolverConfig.py

**Changes**:
- Remove any cache_config field if it exists
- Ensure no imports from cubie_cache.py
- Keep only compile-critical fields (loop_fn, memory elements, compile_flags)

### Integration Points

- `BatchSolverKernel.__init__()` → creates cache
- `BatchSolverKernel.build_kernel()` → attaches cache to dispatcher
- `BatchSolverKernel._invalidate_cache()` → calls cache.flush_cache() if needed
- `BatchSolverKernel.set_cache_dir()` → updates cache directory
- `Solver` class → exposes cache properties via kernel

---

## Task 2: CUDASIM Mode Compatibility

### Current State

In CUDASIM mode:
- `_CacheLocator = object`
- `CacheImpl = object`
- `IndexDataCacheFile = None`
- `CUDACache = object`
- `_CACHING_AVAILABLE = False`

This means CUBIECacheLocator, CUBIECacheImpl, and CUBIECache cannot be
instantiated because their base classes are just `object` with no methods.

### Target State

Allow these operations in CUDASIM mode:
- Creating CacheConfig (already works - pure Python)
- Creating CUBIECacheLocator (path calculation is pure Python)
- Path generation (`get_cache_path()`, `get_source_stamp()`, `get_disambiguator()`)
- File operations (creating/removing cache files)
- Hash computation

Only actual kernel serialization requires CUDA context.

### Components to Modify

#### cuda_simsafe.py

Add stub classes for CUDASIM mode:

```python
# Caching infrastructure stubs for CUDASIM
class _StubCacheLocator:
    """Stub for _CacheLocator in CUDASIM mode."""

    def ensure_cache_path(self):
        path = self.get_cache_path()
        os.makedirs(path, exist_ok=True)

    def get_cache_path(self):
        raise NotImplementedError

    def get_source_stamp(self):
        raise NotImplementedError

    def get_disambiguator(self):
        raise NotImplementedError


class _StubCacheImpl:
    """Stub for CacheImpl in CUDASIM mode."""

    _locator_classes = []

    def reduce(self, data):
        raise NotImplementedError("Cannot reduce in CUDASIM mode")

    def rebuild(self, target_context, payload):
        raise NotImplementedError("Cannot rebuild in CUDASIM mode")

    def check_cachable(self, data):
        return False


class _StubIndexDataCacheFile:
    """Stub for IndexDataCacheFile in CUDASIM mode."""

    def __init__(self, cache_path, filename_base, source_stamp):
        self._cache_path = cache_path
        self._filename_base = filename_base
        self._source_stamp = source_stamp

    def flush(self):
        """Clear the index by saving empty dict."""
        pass  # No-op in stub

    def save(self, key, data):
        raise NotImplementedError("Cannot save in CUDASIM mode")

    def load(self, key):
        return None  # Always miss in CUDASIM


class _StubCUDACache:
    """Stub for CUDACache in CUDASIM mode."""

    _impl_class = None

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def load_overload(self, sig, target_context):
        return None  # Always miss

    def save_overload(self, sig, data):
        pass  # No-op
```

Update the conditional imports:
```python
if CUDA_SIMULATION:
    _CacheLocator = _StubCacheLocator
    CacheImpl = _StubCacheImpl
    IndexDataCacheFile = _StubIndexDataCacheFile
    CUDACache = _StubCUDACache
    _CACHING_AVAILABLE = False
else:
    # ... existing imports from numba
```

#### cubie_cache.py

Modify CUBIECache to handle CUDASIM mode gracefully:
- Check `_CACHING_AVAILABLE` before attempting save/load operations
- Path operations and configuration still work
- flush_cache() works (deletes files if they exist)
- enforce_cache_limit() works (just file operations)

The `__attrs_post_init__` on CUBIECacheLocator (line 177) is incorrect -
it's not an attrs class. Remove it.

### Expected Behavior in CUDASIM

| Operation | Behavior |
|-----------|----------|
| Create CacheConfig | Works normally |
| Create CUBIECache | Works, but marks itself as CUDASIM-limited |
| get_cache_path() | Returns valid path |
| flush_cache() | Removes files if they exist |
| enforce_cache_limit() | Removes oldest files |
| save_overload() | No-op (logged warning optional) |
| load_overload() | Returns None (cache miss) |

---

## Task 3: Review Numba-CUDA Mode Functionality

### Current State

CUBIECache has:
- Custom `enforce_cache_limit()` implementing LRU eviction
- Custom `flush_cache()` using shutil.rmtree
- Commented out `super().__init__()` because parent needs py_func

### Analysis Required

#### super().__init__() Issue

Numba's `Cache.__init__(self, py_func)` does:
```python
self._name = repr(py_func)
self._py_func = py_func
self._impl = self._impl_class(py_func)
# ... locator and cache_file setup
```

CUBIECache doesn't have a py_func because kernels are generated dynamically.
The current approach manually does what super().__init__() would do, which is
acceptable given the constraint.

**Resolution**: Keep the manual initialization but document why. Consider
adding a comment explaining the constraint.

#### flush() vs flush_cache()

Numba's Cache class has:
```python
def flush(self):
    self._cache_file.flush()
```

This clears the index file but doesn't remove data files. CUBIECache's
`flush_cache()` does a full directory wipe with shutil.rmtree.

**Decision needed**: Should flush_cache() use parent's flush() or keep
custom behavior?

**Recommendation**: Keep custom behavior but rename to clarify:
- `flush()` → clears index (delegates to parent/cache_file)
- `flush_cache()` → removes all files (current behavior)

Or use `clear_cache()` for full removal.

#### enforce_cache_limit() Optimization

Current implementation:
1. Lists all .nbi files
2. Sorts by mtime
3. Removes oldest files with their .nbc companions

Numba's IndexDataCacheFile stores entries in a single index file. To use
Numba's mechanics, we would:
1. Load the index
2. Identify entries to remove
3. Resave index without those entries
4. Remove orphaned .nbc files

**Recommendation**: Keep current implementation but add TODO for future
optimization. The current approach works correctly and file system operations
are not performance-critical.

### Components to Modify

#### cubie_cache.py

**CUBIECache class**:
- Add `flush()` method that calls `self._cache_file.flush()` (index only)
- Keep `flush_cache()` for full directory removal
- Add docstring explaining difference between flush() and flush_cache()
- Add comment near __init__ explaining why super().__init__() is not called
- Add TODO comment on enforce_cache_limit() for future Numba integration

---

## Task 4: Address Review Comments

### enforce_cache_limit() Review

**Current comment**:
> "we should use existing Numba IndexCacheFile mechanics to remove certain
> entries by index, or resave only certain indices"

**Analysis**:
IndexDataCacheFile has `save(key, data)` and `load(key)` methods. The flush()
method saves an empty index. There's no direct API for "remove entry by key".

To implement this properly:
1. Load index via `_load_index()`
2. Remove entries from dict
3. Call `_save_index()` with modified dict
4. Remove orphaned .nbc files

However, these methods are prefixed with `_` (private).

**Resolution**:
Add TODO comment acknowledging this is future work. Current implementation
is functionally correct and the optimization benefit is minimal for typical
cache sizes (10 entries).

### flush_cache() Review

**Current comment**:
> "Can't we just use existing Numba cache flush logic?"

**Analysis**:
Numba's `Cache.flush()` calls `self._cache_file.flush()` which clears the
index but leaves .nbc data files orphaned. CUBIECache's full directory wipe
is cleaner for the "flush_on_change" mode where we want complete reset.

**Resolution**:
- Add `flush()` method delegating to `self._cache_file.flush()` for index-only clear
- Keep `flush_cache()` for full removal
- Document the distinction in docstrings

---

## Dependencies and Integration

### Import Dependencies

After refactoring:

```
BatchSolverKernel.py
  ├── cubie_cache.py (CUBIECache, CacheConfig)
  └── BatchSolverConfig.py (BatchSolverConfig, ActiveOutputs)

cubie_cache.py
  ├── CUDAFactory.py (_CubieConfigBase)
  ├── cuda_simsafe.py (cache stubs/real classes)
  └── odesystems/symbolic/odefile.py (GENERATED_DIR)

cuda_simsafe.py
  └── (no cache imports from cubie_cache - one-way dependency)
```

### Test File Updates

Tests should verify:
1. Cache creation at kernel init (not build time)
2. Cache operations work in CUDASIM mode (except save/load)
3. flush() vs flush_cache() behavior difference
4. Cache property delegation works correctly

---

## Edge Cases

1. **cache=True with CUDASIM**: Cache object created but no-ops on save/load
2. **cache=False**: No cache object created, all cache methods return None/no-op
3. **cache=Path**: Custom directory used, CUBIECache created with that path
4. **Cache directory doesn't exist**: Created on first access
5. **Permission errors**: Caught and warned, cache disabled gracefully

---

## Validation Criteria

For detailed_implementer agent:

1. All existing tests in test_cubie_cache.py pass
2. All existing tests in test_cache_config.py pass
3. BatchSolverKernel can be instantiated with cache=True/False/Path
4. No CacheConfig in BatchSolverConfig
5. Cache operations (flush, enforce_limit) work in CUDASIM mode
6. Import of CacheConfig from BatchSolverConfig is fixed
