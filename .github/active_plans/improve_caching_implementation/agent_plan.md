# Agent Implementation Plan: Improve Caching Implementation

## Overview

This plan addresses four tasks for improving CuBIE's kernel caching infrastructure:

1. **Decontaminate BatchSolverKernel/BatchSolverConfig** - Separate cache settings from compile-critical parameters
2. **Review CUDASIM Compatibility** - Ensure cache operations work without CUDA intrinsics
3. **Review Functionality vs CUDASIM Mode** - Align with numba-cuda patterns
4. **Address PR Review Comments** - Resolve any pending review feedback

---

## Task 1: Decontaminate BatchSolverKernel/BatchSolverConfig

### Goal
Remove cache settings from compile-critical configuration by creating a dedicated `CacheConfig` class.

### Components to Create

#### 1.1 CacheConfig Class
**Location**: `src/cubie/batchsolving/BatchSolverConfig.py` (or new file `src/cubie/batchsolving/CacheConfig.py`)

**Expected Behavior**:
- An attrs class that holds cache-related configuration
- NOT a subclass of `CUDAFactoryConfig` (cache settings are not compile-critical)
- Contains fields for:
  - `enabled: bool` - Whether caching is enabled
  - `cache_path: Optional[Path]` - Custom cache directory (None = use default)
  - `source_stamp: Optional[Tuple]` - Timestamp/size tuple for cache validation

**Key Method**:
```
@classmethod
def from_cache_param(cls, cache: Union[bool, str, Path, None]) -> "CacheConfig"
```

This method parses the `cache` parameter that may be passed to BatchSolverKernel:
- `True` → Enable caching with default path
- `False` or `None` → Disable caching  
- `str` or `Path` → Enable caching at specified path

The parsing logic currently referred to as `_parse_cache_param` should be encapsulated here.

#### 1.2 BatchSolverKernel Modifications
**Location**: `src/cubie/batchsolving/BatchSolverKernel.py`

**Expected Changes**:
- Accept optional `cache` parameter in `__init__`
- Use `CacheConfig.from_cache_param(cache)` to construct config
- Store `CacheConfig` instance as `self._cache_config` (NOT in `BatchSolverConfig`)
- Cache settings should not affect `self.compile_settings.values_hash`

**Integration Points**:
- The `CacheConfig` instance should be used when:
  - Determining whether to check disk cache before build
  - Deciding where to save compiled kernels
  - Generating cache keys (using `config_hash` from compile settings)

### Data Structures

```
CacheConfig:
    enabled: bool = False
    cache_path: Optional[Path] = None
    source_stamp: Optional[Tuple[float, int]] = None  # (mtime, size)
    
    @classmethod
    def from_cache_param(cls, cache) -> CacheConfig
    
    @property
    def cache_directory(self) -> Optional[Path]
        """Return resolved cache directory or None if disabled."""
```

### Edge Cases
- `cache=True` but no valid cache directory → Fall back to user cache dir or disable
- `cache="/invalid/path"` → Log warning and disable caching
- Cache directory not writable → Disable caching gracefully

---

## Task 2: Review CUDASIM Compatibility

### Goal
Ensure cache infrastructure can be tested without CUDA hardware.

### Components to Review/Modify

#### 2.1 cuda_simsafe.py Additions
**Location**: `src/cubie/cuda_simsafe.py`

**Expected Additions**:
Minimal stubs that allow cache-related code to function in CUDASIM mode. The key insight is that most cache operations (file I/O, hashing, path generation) do NOT require CUDA intrinsics.

What may need stubs:
- If cache code references `cuda.current_context()` for device info → provide fake context
- If cache code uses device compute capability for cache keys → provide fake CC

**Pattern**: Follow existing `cuda_simsafe.py` pattern where `CUDA_SIMULATION` flag controls imports.

#### 2.2 Cache Operations Analysis

Operations that should work WITHOUT any CUDA dependency:
- SHA256 hashing of configuration values
- File system operations (mkdir, write, read)
- Pickle serialization/deserialization
- Index file management

Operations that MAY need CUDASIM handling:
- Getting compute capability for cache key
- Device context for unique identification

### Dependencies to Avoid

The cache system should NOT import:
- `numba.cuda.cudadrv.driver` directly (use cuda_simsafe wrappers)
- Any GPU-specific memory management
- PTX/SASS inspection functions

---

## Task 3: Review Functionality vs CUDASIM Mode

### Goal
Align with numba-cuda caching patterns while adapting for CuBIE's needs.

### numba-cuda Patterns to Adopt

#### 3.1 IndexDataCacheFile Pattern
**Source**: `numba_cuda/numba/cuda/core/caching.py`

The `IndexDataCacheFile` class provides:
- Separate index file (`.nbi`) and data files (`.nbc`)
- Source stamp validation
- Version checking
- Atomic writes with UUID temp files

**CuBIE Adaptation**:
- CuBIE can use this pattern directly or create similar class
- Replace function-based key (codebytes, cvarbytes) with `config_hash`
- Use CuBIE's existing `values_hash` infrastructure

#### 3.2 CacheImpl Pattern
**Source**: `numba_cuda/numba/cuda/core/caching.py`

The abstract `CacheImpl` provides:
- `reduce(data)` - Serialize data for storage
- `rebuild(target_context, reduced_data)` - Deserialize from storage
- `check_cachable(data)` - Determine if data can be cached

**CuBIE Adaptation**:
Create `CuBIECacheImpl` that:
- `reduce()`: Returns serializable representation of compiled kernel
- `rebuild()`: Reconstructs kernel from cached data
- `check_cachable()`: Returns True (CuBIE kernels are always cachable)

#### 3.3 _Kernel Serialization Pattern
**Source**: `numba_cuda/numba/cuda/dispatcher.py`

The `_Kernel` class uses `serialize.ReduceMixin`:
- `_reduce_states()` - Returns dict of serializable state
- `_rebuild()` - Classmethod to reconstruct from reduced state

**CuBIE Adaptation**:
`BatchSolverCache` (or equivalent) should implement:
- `_reduce_states()` - Return dict with `solver_kernel`, `compile_settings_hash`, etc.
- `_rebuild()` - Reconstruct the cached kernel

### Key Differences from numba-cuda

| Aspect | numba-cuda | CuBIE |
|--------|------------|-------|
| Cache Key | Function bytecode + closure | `config_hash` from compile settings |
| py_func | Required | Not available (dynamically generated) |
| Compute Capability | In key | Should be in key |
| Source Stamp | File mtime/size | N/A or custom stamp |

### Commented-out super().__init__() Issue

The prompt mentions a "commented out super().__init__() which had to be removed because the original needs py_func".

**Resolution Approach**:
- Don't inherit from numba-cuda's `Cache` class directly
- Instead, compose/wrap the caching infrastructure
- Create CuBIE-specific cache class that uses similar patterns but doesn't require `py_func`

---

## Task 4: Address PR Review Comments

### Goal
Resolve all pending review feedback.

### Approach
1. Query for PR review comments using GitHub MCP tools
2. Identify comments related to caching implementation
3. Address each comment with appropriate code changes

### Expected Categories of Feedback
- Code style (line length, docstrings, naming)
- Architecture concerns (separation of concerns)
- Test coverage gaps
- Documentation updates

---

## Integration Points

### With Existing CuBIE Infrastructure

```
CUDAFactory.config_hash
    ↓
CacheConfig.from_cache_param()
    ↓
CuBIECache (wraps IndexDataCacheFile pattern)
    ↓
BatchSolverKernel.build() / load_cached()
```

### With Buffer Registry
Cache operations should NOT interact with buffer_registry. Caching is purely about compiled kernel persistence.

### With Memory Manager  
Cache operations should NOT interact with memory manager. Cache deals with file I/O, not GPU memory.

---

## Test Expectations

### New Test Files
- `tests/batchsolving/test_cache_config.py`
  - Test `CacheConfig.from_cache_param()` with various inputs
  - Test cache path resolution
  
### Modified Test Files
- `tests/batchsolving/test_solver.py`
  - Add tests for cache=True/False behavior
  
### CUDASIM Compatibility
- All new cache tests should be runnable with `pytest -m "not nocudasim"`
- Cache file operations should work without GPU

---

## File Modification Summary

| File | Action | Description |
|------|--------|-------------|
| `src/cubie/batchsolving/BatchSolverConfig.py` | Modify | Add CacheConfig class |
| `src/cubie/batchsolving/BatchSolverKernel.py` | Modify | Use CacheConfig, remove cache from compile settings |
| `src/cubie/cuda_simsafe.py` | Modify | Add cache-related CUDASIM stubs if needed |
| `src/cubie/CUDAFactory.py` | Review | Ensure config_hash usable for cache keys |
| `tests/batchsolving/test_cache_config.py` | Create | New tests for CacheConfig |

---

## Dependencies

### External
- `pathlib.Path` - Path operations
- `pickle` - Serialization (via numba.cuda.serialize or custom)
- `hashlib` - SHA256 hashing
- `os` - File system operations

### Internal
- `cubie.CUDAFactory._CubieConfigBase` - For config patterns
- `cubie.cuda_simsafe.CUDA_SIMULATION` - For conditional behavior

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Cache invalidation bugs | Use existing `values_hash` infrastructure |
| CUDASIM incompatibility | Isolate CUDA-specific code, test in both modes |
| File permission errors | Graceful fallback to no-cache mode |
| Pickle compatibility | Pin serialization format, version checking |
