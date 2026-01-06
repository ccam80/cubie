# Agent Plan: File-Based Caching for BatchSolverKernel

## Overview

This plan describes the architectural approach for implementing file-based
caching of compiled CUDA kernels in CuBIE. The goal is to persist kernels
to disk so they can be reloaded across Python sessions, eliminating repeated
compilation time.

---

## Component Descriptions

### 1. CUBIECacheLocator

**Purpose**: Custom cache locator that directs cache files to the CuBIE
`generated/` directory structure instead of the default `__pycache__`.

**Expected Behaviour**:
- Returns cache path as `generated/<system_name>/cache/`
- Uses ODE system hash as source stamp for freshness checking
- Provides disambiguator based on compile settings hash
- Creates cache directory if it does not exist

**Key Methods**:
- `get_cache_path()` → `Path` to cache directory
- `get_source_stamp()` → hash representing system definition freshness
- `get_disambiguator()` → string hash of compile settings
- `ensure_cache_path()` → create directory if needed (inherited)

**Relationship to numba-cuda**: Subclasses `_CacheLocator` from
`numba.cuda.core.caching`. Does NOT use `_SourceFileBackedLocatorMixin`
since CuBIE kernels are not backed by a single source file.

### 2. CUBIECacheImpl

**Purpose**: Implementation class providing serialization logic for CuBIE
kernels.

**Expected Behaviour**:
- `reduce()` extracts serializable state from a compiled kernel
- `rebuild()` reconstructs a kernel from cached state
- `check_cachable()` returns True (CUDA kernels are always cachable)
- Uses `CUBIECacheLocator` instead of default locator classes

**Key Methods**:
- `reduce(kernel)` → dict of serializable kernel state
- `rebuild(target_context, payload)` → reconstructed `_Kernel` object
- `check_cachable(data)` → `True`

**Serialization Details**:
- Leverages numba's existing `_Kernel._reduce_states()` and `_Kernel._rebuild()`
- These handle PTX/cubin serialization automatically
- No need to implement custom serialization

**Relationship to numba-cuda**: Subclasses `CacheImpl` from
`numba.cuda.core.caching`. Overrides `_locator_classes` to use
`CUBIECacheLocator`.

### 3. CUBIECache

**Purpose**: Main cache class that coordinates loading and saving of
cached kernels.

**Expected Behaviour**:
- Initialized with ODE system and compile settings (not a py_func)
- Computes cache keys that incorporate system hash and settings hash
- Delegates file operations to `IndexDataCacheFile`
- Provides cache statistics (hits/misses)

**Key Methods**:
- `load_overload(sig, target_context)` → cached kernel or None
- `save_overload(sig, data)` → save kernel to disk
- `_index_key(sig, codegen)` → composite key including CuBIE-specific hashes

**Cache Key Construction**:
The cache key must uniquely identify a compilation configuration:
```
key = (
    signature,                    # Function signature (from numba)
    codegen.magic_tuple(),        # Target architecture
    system_definition_hash,       # ODE equations + constants
    compile_settings_hash,        # All attrs compile settings
    numba_version,                # For compatibility
)
```

**Relationship to numba-cuda**: Subclasses `Cache` from
`numba.cuda.core.caching`. Uses custom `CUBIECacheImpl`.

### 4. Compile Settings Hash Utility

**Purpose**: Compute a stable hash from attrs compile settings classes.

**Expected Behaviour**:
- Traverses the CUDAFactory object chain
- Collects all `compile_settings` attributes
- Serializes to stable representation
- Computes hash of serialized form

**Algorithm**:
```
1. For each CUDAFactory in the chain:
   a. Get compile_settings attrs class
   b. For each field in attrs class:
      - If primitive: add to hash input
      - If numpy array: add array.tobytes()
      - If nested attrs: recurse
      - If callable: skip (device functions not hashable)
2. Concatenate all values with separators
3. Return SHA256 hash
```

**Considerations**:
- Must handle `eq=False` fields (like `loop_fn`) by skipping them
- Must handle numpy arrays with `tobytes()` for deterministic hashing
- Must handle optional fields that may be None

### 5. CUDAFactory Integration

**Purpose**: Minimal modifications to attach custom cache to dispatchers.

**Expected Behaviour**:
- When `build()` creates a dispatcher via `cuda.jit()`, attach cache
- Pass system and compile settings to cache constructor
- Cache attachment is optional; graceful fallback if caching fails

**Integration Point** (in BatchSolverKernel.build_kernel):
```python
@cuda.jit(...)
def integration_kernel(...):
    ...

# Attach CuBIE-specific caching
if caching_enabled:
    cache = CUBIECache(self.system, self.compile_settings)
    integration_kernel._cache = cache
    # Note: don't call enable_caching() - we're replacing the cache directly

return integration_kernel
```

---

## Architectural Changes Required

### New File: `src/cubie/cubie_cache.py`

Contains:
- `CUBIECacheLocator` class
- `CUBIECacheImpl` class
- `CUBIECache` class
- `hash_compile_settings()` utility function

### Modified File: `src/cubie/batchsolving/BatchSolverKernel.py`

Changes:
- Import caching utilities
- Add cache attachment in `build_kernel()` method
- Optional: add `cache_enabled` configuration parameter

### Modified File: `src/cubie/CUDAFactory.py`

Changes:
- Add method to traverse object chain and collect compile settings
- Add property to access system hash (for subclasses that have systems)

---

## Integration Points with Current Codebase

### With CUDAFactory Pattern

The caching layer must respect CuBIE's existing cache invalidation:
- CUDAFactory._cache_valid tracks in-memory cache validity
- File-based cache is a secondary layer checked during `build()`
- When CUDAFactory._cache_valid is False, check file cache before compiling

### With ODEFile and GENERATED_DIR

Cache files should coexist with generated code:
```
generated/
├── system_name.py          # Generated dxdt code (existing)
└── system_name/
    └── cache/
        ├── kernel-1.nbi    # Index file
        └── kernel-1.1.nbc  # Data file
```

### With hash_system_definition

Reuse existing `hash_system_definition()` from `sym_utils.py`:
- Already captures ODE equations and constants
- Provides stable hash for cache key construction
- Called during SymbolicODE construction

---

## Expected Interactions Between Components

```
BatchSolverKernel.build_kernel()
    │
    ├─> cuda.jit() creates CUDADispatcher
    │
    ├─> CUBIECache(system, compile_settings) created
    │       │
    │       ├─> CUBIECacheImpl(py_func) initialized
    │       │       │
    │       │       └─> CUBIECacheLocator.from_function()
    │       │               │
    │       │               └─> Creates locator with system_name path
    │       │
    │       └─> IndexDataCacheFile initialized with cache_path
    │
    └─> Attach cache to dispatcher._cache
```

On cache lookup:
```
CUDADispatcher.compile(sig)
    │
    ├─> Check in-memory overloads
    │
    ├─> CUBIECache.load_overload(sig, target_context)
    │       │
    │       ├─> _index_key(sig, codegen)
    │       │       │
    │       │       ├─> Standard numba key components
    │       │       └─> + system_hash + compile_settings_hash
    │       │
    │       └─> IndexDataCacheFile.load(key)
    │               │
    │               └─> CUBIECacheImpl.rebuild() if found
    │
    └─> If not found: compile and save via save_overload()
```

---

## Data Structures

### Cache Index File Format (`.nbi`)

Pickle file containing:
```python
{
    key: data_filename,
    # key is tuple of (sig, arch, system_hash, settings_hash, ...)
}
```

### Cache Data File Format (`.nbc`)

Pickle file containing kernel state from `_Kernel._reduce_states()`:
```python
{
    'cooperative': bool,
    'name': str,
    'signature': tuple,
    'codelibrary': CodeLibrary,  # Contains PTX/cubin
    'debug': bool,
    'lineinfo': bool,
    'call_helper': object,
    'extensions': list,
}
```

### Compile Settings Hash Input

Ordered concatenation of:
```
precision=float32|
local_memory_elements=128|
shared_memory_elements=64|
compile_flags.save_state=True|
compile_flags.save_observables=False|
...
```

---

## Dependencies and Imports Required

```python
# From numba-cuda (internal, may change between versions)
from numba.cuda.core.caching import (
    Cache,
    CacheImpl,
    _CacheLocator,
    IndexDataCacheFile,
)
from numba.cuda.dispatcher import _Kernel

# From cubie
from cubie.odesystems.symbolic.sym_utils import hash_system_definition
from cubie.odesystems.symbolic.odefile import GENERATED_DIR

# Standard library
import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
```

---

## Edge Cases to Consider

### 1. No ODE System Available

Some CUDAFactory subclasses may not have an ODE system. In this case:
- Skip system hash component
- Use only compile settings hash
- May result in less specific cache keys

### 2. Compile Settings with Callable Fields

Fields like `loop_fn` are callable device functions and cannot be hashed:
- Skip these fields in hash computation
- Mark with `eq=False` in attrs definition (already done)
- Cache key still valid because callables are regenerated from settings

### 3. Cache Corruption

If cache files become corrupted:
- `rebuild()` will raise an exception
- Catch exception in `load_overload()`, return None
- System falls back to recompilation
- Optionally: delete corrupted files

### 4. CUDA Simulator Mode

When `NUMBA_ENABLE_CUDASIM=1`:
- numba-cuda caching may not work (FakeCUDAKernel)
- Detect simulator mode and disable file caching
- In-memory CUDAFactory caching still works

### 5. Version Mismatch

When numba or CUDA version changes:
- Include version in cache key
- Old cache files will not match new keys
- Files remain on disk but unused (cleanup is manual)

### 6. Concurrent Access

Multiple Python processes may access cache simultaneously:
- numba's IndexDataCacheFile uses atomic write patterns
- Race conditions handled by uuid-based temp files
- Worst case: redundant compilation, not corruption

---

## Configuration Options

Consider adding to `BatchSolverConfig` or solver-level settings:

```python
cache_enabled: bool = True       # Enable/disable file caching
cache_dir: Optional[Path] = None # Override default cache location
cache_debug: bool = False        # Enable cache hit/miss logging
```

---

## Validation Strategy

1. **Unit Tests**:
   - `CUBIECacheLocator.get_cache_path()` returns expected directory
   - `hash_compile_settings()` produces stable hashes
   - Hash changes when any setting changes

2. **Integration Tests**:
   - First run creates cache files
   - Second run loads from cache (verify via logging or timing)
   - Settings change causes recompilation

3. **Manual Testing**:
   - Run solver, exit Python, run again
   - Verify faster startup on second run
   - Inspect generated cache files

---

## Implementation Notes for detailed_implementer

### Priority Order

1. `hash_compile_settings()` utility - required for cache keys
2. `CUBIECacheLocator` - required for cache paths
3. `CUBIECacheImpl` - can mostly delegate to numba defaults
4. `CUBIECache` - ties everything together
5. `BatchSolverKernel` integration - attach cache to dispatcher

### Testing Approach

- Use CUDASIM mode for unit tests (no GPU required)
- Mock file system operations for isolation
- Integration tests need real compilation (slower)

### Documentation

- Add docstrings following numpydoc format
- Document cache file locations in user guide
- Note dependency on numba-cuda internals
