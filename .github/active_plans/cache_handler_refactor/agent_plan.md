# CacheHandler Refactor - Agent Plan

## Overview

This plan details the architectural changes required to complete the
CacheHandler refactor for a persistent cache interface in CuBIE. The cache
handler will manage file-based kernel caching following CuBIE patterns.

---

## Component 1: CacheConfig Enhancement

### Expected Behavior
CacheConfig is an attrs class that stores all cache-related configuration.
It should:
- Store cache_enabled, cache_mode, max_cache_entries, cache_dir, system_hash
- Provide `params_from_user_kwarg()` for parsing the single `cache` argument
- Inherit from `_CubieConfigBase` to gain update() and hashing capabilities

### Current State
CacheConfig already exists with the required fields. The `params_from_user_kwarg`
method exists but returns keys that need refinement for the update pattern.

### Required Changes
- Rename field `enabled` to `cache_enabled` for consistency with kwarg pattern
- Ensure update() method handles all cache-related kwargs properly
- Add `ALL_CACHE_PARAMETERS` constant set for kwarg filtering

### Integration Points
- Used by CubieCacheHandler for configuration storage
- Passed to `build_config()` utility for instantiation
- Updated via CubieCacheHandler.update()

---

## Component 2: CubieCacheHandler Refactor

### Expected Behavior
CubieCacheHandler manages the lifecycle of a CUBIECache instance:
- Instantiates with system_name, system_hash from ODE system
- Accepts cache kwargs that override CacheConfig defaults
- Provides `update()` method returning recognized parameters
- Provides `configured_cache(compile_settings_hash)` for run-time cache setup

### Current State
CubieCacheHandler exists with basic structure but:
- Uses `create_cache` with wrong signature (expects CacheConfig, not raw args)
- Update method doesn't follow return pattern (set of recognized keys)
- Missing proper system_name/system_hash propagation

### Required Changes

#### Initialization
- Accept `cache_arg` (bool/str/Path) plus `**kwargs` for cache settings
- Extract system_name, system_hash from kwargs or use defaults
- Use `build_config()` to create CacheConfig from merged parameters
- Instantiate CUBIECache via proper factory method

#### Update Method
- Accept `updates_dict=None, silent=False, **kwargs` signature
- Merge updates_dict and kwargs
- Call `config.update()` for cache settings
- If system_hash changes, update internal state
- Return set of recognized parameter names

#### configured_cache Method
- Accept compile_settings_hash parameter
- Update cache with current system_hash and compile_settings_hash
- Return the configured CUBIECache instance

### Integration Points
- Instantiated by BatchSolverKernel.__init__()
- Updated by BatchSolverKernel.update() when cache settings change
- Called by BatchSolverKernel.run() to get configured cache before launch

---

## Component 3: CUBIECacheLocator Path Refactor

### Expected Behavior
CUBIECacheLocator computes cache paths with hierarchical structure:
- Path format: `generated/<system_name>/<system_hash>/CUDA_cache/`
- System hash subdirectory groups entries by ODE system version
- Compile settings hash used only for disambiguation within directory

### Current State
Current path: `generated/<system_name>/CUDA_cache/`
Missing the system_hash subdirectory level.

### Required Changes
- Modify `get_cache_path()` to include system_hash in path
- Path becomes: `self._cache_path = GENERATED_DIR / system_name / system_hash / "CUDA_cache"`
- Update `set_system_hash()` to potentially update path (or require reinit)

### Data Structures
- `_system_name`: str - ODE system identifier
- `_system_hash`: str - Hash of ODE system definition  
- `_compile_settings_hash`: str - Hash of compile settings
- `_cache_path`: Path - Computed cache directory path

### Edge Cases
- Empty system_hash should use "default" or raise error
- Custom cache_dir should override entire path, not just base
- Path creation should be lazy (directory created on first write)

---

## Component 4: BatchSolverKernel Cache Integration

### Expected Behavior
BatchSolverKernel coordinates cache handler lifecycle:
- Instantiates CubieCacheHandler with system info and cache settings
- Updates cache handler when relevant settings change
- Configures cache with compile_settings_hash at each run()

### Current State
BatchSolverKernel has cache_handler but:
- Instantiation passes wrong kwargs structure (cache_settings may be None)
- Missing extraction of system_name and system_hash from ODE system
- run() method accesses cache_handler correctly but needs verification

### Required Changes

#### __init__ Method
- Extract system_name from `system.name` (with fallback to hash)
- Extract system_hash from `system.fn_hash` if SymbolicODE, else compute
- Build cache_settings dict from filtered kwargs
- Pass to CubieCacheHandler initialization

#### update Method
- Detect cache-related parameters in updates_dict
- Forward to cache_handler.update()
- Track recognized cache parameters

#### run Method
- Compute config_hash via `self.config_hash()` 
- Call `cache_handler.configured_cache(config_hash)` before kernel launch
- Attach returned cache to kernel dispatcher

### Integration Points
- Receives cache settings from Solver
- Owns CubieCacheHandler instance
- Provides cache_config property for external access

---

## Component 5: System Hash Extraction

### Expected Behavior
System hash uniquely identifies the ODE system definition:
- For SymbolicODE: use `fn_hash` attribute (already computed)
- For other BaseODE subclasses: compute from compile_settings hash
- Must be stable across sessions for cache hits

### Current State
- SymbolicODE has `fn_hash` attribute computed from equations
- BaseODE has no explicit hash; uses compile_settings via CUDAFactory
- BatchSolverKernel doesn't extract system hash

### Required Changes
- Add helper function or property to extract system hash from BaseODE
- For SymbolicODE: return `fn_hash`
- For BaseODE: return `compile_settings.values_hash`
- Add `system_name` property accessor if not present

### Dependencies
- Must work for both SymbolicODE and custom BaseODE subclasses
- Hash must be deterministic for same system definition

---

## Component 6: Test Updates

### Expected Behavior
Tests validate:
- Cache handler instantiation with various configurations
- Cache directory structure includes system_hash subdirectory
- Cache handler update method recognizes correct parameters
- Cache hit/miss behavior with different hashes

### Current Test State
`tests/test_cubie_cache.py` has tests for:
- CUBIECacheLocator path computation
- CUBIECacheImpl properties
- CUBIECache initialization and index keys
- create_cache and invalidate_cache functions

### Required Changes
- Update path assertions to expect `<system_name>/<system_hash>/CUDA_cache/`
- Add tests for CubieCacheHandler instantiation
- Add tests for CubieCacheHandler.update() return value
- Add tests for cache_handler.configured_cache() behavior
- Update BatchSolverKernel fixture to include cache_settings

### Test Fixtures Needed
- Mock ODE system with name and fn_hash
- Cache settings dictionaries for various configurations
- Temporary directories for cache file testing

---

## Architectural Dependencies

```
CacheConfig
    └── used by CubieCacheHandler
    
CubieCacheHandler
    ├── uses CacheConfig
    ├── creates CUBIECache
    └── used by BatchSolverKernel

CUBIECacheLocator
    └── used by CUBIECacheImpl
    
CUBIECacheImpl
    └── used by CUBIECache

CUBIECache
    └── attached to kernel dispatcher

BatchSolverKernel
    ├── owns CubieCacheHandler
    ├── extracts system info from BaseODE
    └── configures cache at run time
```

---

## Parameter Flow

### At Instantiation
```
User provides: cache=True, cache_mode='hash', cache_dir='/custom'
         ↓
Solver filters: cache_settings = {cache_mode, cache_dir, ...}
         ↓
BatchSolverKernel extracts: system_name, system_hash from system
         ↓
CubieCacheHandler: build_config(CacheConfig, required, **cache_settings)
         ↓
CacheConfig instance with merged settings
         ↓
CUBIECache created with config values
```

### At Update
```
User calls: solver.update(cache_mode='flush_on_change')
         ↓
Solver forwards to: kernel.update(...)
         ↓
BatchSolverKernel detects cache param, calls: cache_handler.update(...)
         ↓
CubieCacheHandler: config.update({cache_mode: ...})
         ↓
Cache updated or recreated as needed
```

### At Run
```
BatchSolverKernel.run() starts
         ↓
Compute: config_hash = self.config_hash()
         ↓
Get configured cache: cache_handler.configured_cache(config_hash)
         ↓
CubieCacheHandler: cache.set_hashes(system_hash, config_hash)
         ↓
Attach: kernel._cache = cache
         ↓
Launch kernel
```

---

## Constants and Parameters

### ALL_CACHE_PARAMETERS Set
Should include:
- `cache_enabled`
- `cache_mode`
- `max_cache_entries`
- `cache_dir`

Note: `system_hash` and `system_name` are derived from system, not user kwargs.

### Cache Mode Values
- `'hash'`: Content-addressed caching (default)
- `'flush_on_change'`: Clear cache when settings change

---

## Error Handling

### Invalid Cache Path
- If cache_dir is provided but not writable, warn and disable caching
- If path creation fails, continue without caching

### Missing System Hash
- If system has no fn_hash, compute from compile_settings
- If neither available, generate from system representation

### Cache Corruption
- Numba's cache handles corruption internally
- CuBIE layer should not interfere with Numba's error handling

---

## Edge Cases

1. **System without name**: Use first 12 chars of system_hash as name
2. **Empty compile_settings_hash**: Should never happen; raise if it does
3. **Cache disabled at init, enabled later**: update() should create cache
4. **Custom cache_dir with system_hash subdirs**: Preserve subdirectory structure
5. **CUDASIM mode**: Caching behavior may differ; handle gracefully
