# Caching Strategy Comparison

## Overview

Two approaches exist for managing file-based kernel caching in CuBIE:

1. **Hash-Based Caching**: Cache per unique combination of constants and code
2. **Flush-on-Change Caching**: Use CUDAFactory invalidation to flush cache

---

## Option A: Hash-Based Caching (Current Implementation)

Cache files are keyed by a composite hash of the ODE system definition and
compile settings. Each unique configuration gets its own cache entry.

```python
def _index_key(self, sig, codegen):
    return (
        sig,
        codegen.magic_tuple(),
        self._system_hash,           # Hash of ODE equations + constants
        self._compile_settings_hash, # Hash of all compile settings
    )
```

### Pros

| Advantage | Description |
|-----------|-------------|
| **Multiple configs cached** | Switching between configurations (e.g., float32 ↔ float64) loads from cache instead of recompiling |
| **No stale cache risk** | Each configuration has a unique key; old configs remain valid |
| **Parallel development** | Multiple developers can work with different settings without invalidating each other's caches |
| **Reproducibility** | Same settings always produce same cache key |
| **Rollback friendly** | Reverting settings immediately uses previously cached kernel |

### Cons

| Disadvantage | Description |
|--------------|-------------|
| **Disk usage** | Each unique configuration creates new cache files; can accumulate over time |
| **Hash computation overhead** | Must traverse attrs classes and compute SHA256 on each `build()` call |
| **Complexity** | Requires custom hash function that handles arrays, nested attrs, callable fields |
| **Cache pollution** | Experimental configurations leave orphaned cache files |
| **No automatic cleanup** | Old/unused cache entries persist indefinitely |

---

## Option B: Flush-on-Change Caching

Use CUDAFactory's existing `_cache_valid` flag to trigger cache flush. When
any compile setting changes, delete all cached files and recompile.

```python
def _invalidate_cache(self):
    self._cache_valid = False
    self._flush_file_cache()  # Delete all cache files for this system

def _flush_file_cache(self):
    cache_dir = GENERATED_DIR / self.system.name / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
```

### Pros

| Advantage | Description |
|-----------|-------------|
| **Simple implementation** | Hook into existing `_invalidate_cache()` method |
| **Minimal disk usage** | Only one configuration cached at a time |
| **No hash computation** | Avoids complex hash function for attrs classes |
| **No orphaned files** | Cache is always current configuration only |
| **Easy debugging** | Clear 1:1 relationship between settings and cache |

### Cons

| Disadvantage | Description |
|--------------|-------------|
| **Recompilation on toggle** | Switching between configs (e.g., debug ↔ release) always recompiles |
| **Lost work** | Flushing discards potentially useful cached kernels |
| **Not parallel-safe** | Multiple processes with different settings fight over cache |
| **Slower iteration** | Parameter sweeps recompile on each change |
| **Wasted compilation** | Frequently-used configurations are recompiled repeatedly |

---

## Performance Comparison

| Scenario | Hash-Based | Flush-on-Change |
|----------|------------|-----------------|
| First run | Compile + cache | Compile + cache |
| Same settings, new session | **Load from cache** | **Load from cache** |
| Change precision | Compile + cache | Compile + **flush** |
| Revert precision | **Load from cache** | Compile + cache |
| Parameter sweep (N configs) | N compiles, then all cached | N × 2 compiles (oscillating) |

---

## Recommendation

**Use Hash-Based Caching** (Option A) for the following reasons:

1. **Common workflow support**: Users frequently toggle between float32/float64
   or adjust tolerances during development. Hash-based caching preserves work.

2. **Parameter exploration**: Research workflows involve sweeping parameters;
   caching each configuration accelerates iteration.

3. **Multi-user environments**: Shared systems benefit from independent cache
   entries per configuration.

4. **Alignment with numba-cuda**: The hash-based approach mirrors how numba
   handles its own caching, making the integration more idiomatic.

### Mitigations for Disk Usage

1. **Cache size limits**: Implement LRU eviction when cache exceeds threshold
2. **Manual cleanup**: Provide `clear_cache()` method for user-initiated cleanup
3. **TTL expiration**: Optionally expire cache entries older than N days
4. **Per-system isolation**: Cache files are already in `generated/<system>/`
   so deleting the system directory clears its cache

### Hybrid Approach (Future Enhancement)

Consider a hybrid where:
- Hash-based caching is default for development
- Flush-on-change can be enabled via `caching_mode='flush'` for
  production/CI where disk usage matters

```python
class BatchSolverConfig:
    caching_enabled: bool = True
    caching_mode: str = 'hash'  # 'hash' or 'flush'
```

---

## Conclusion

The hash-based approach has higher upfront complexity but provides better
user experience for CuBIE's primary use cases (research, parameter exploration,
iterative development). The disk usage concern is mitigated by cache isolation
and can be addressed with cleanup utilities if needed.

The flush-on-change approach would be simpler but would significantly degrade
the development experience, especially for users who frequently adjust settings.
