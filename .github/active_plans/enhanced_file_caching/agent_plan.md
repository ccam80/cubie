# Agent Plan: Enhanced File-Based Caching

## Component Architecture

### 1. CacheConfig Class

A new attrs class in `BatchSolverConfig.py` that encapsulates all cache
configuration:

**Expected Behavior:**
- Stores `enabled` (bool), `mode` (str), `cache_dir` (Optional[Path]),
  `max_entries` (int)
- Default: enabled=True, mode='hash', cache_dir=None, max_entries=10
- When `cache_dir` is None, uses default GENERATED_DIR/<system>/cache/
- Validates mode is one of ('hash', 'flush_on_change')

**Integration Points:**
- Added as field to BatchSolverConfig
- Passed through to CUBIECache during kernel build
- Accessible via BatchSolverKernel properties

### 2. Enhanced Cache Parameter Parsing

BatchSolverKernel.__init__ needs to accept an enhanced `cache` parameter:

**Expected Behavior:**
- `cache=True`: Create CacheConfig(enabled=True, mode='hash')
- `cache=False`: Create CacheConfig(enabled=False)
- `cache='flush_on_change'`: Create CacheConfig(enabled=True,
  mode='flush_on_change')
- `cache=Path(...)` or `cache='path/string'`: Create CacheConfig(enabled=True,
  mode='hash', cache_dir=path)

**Integration Points:**
- Parsing occurs in BatchSolverKernel.__init__
- Result stored in BatchSolverConfig.cache_config
- Used in build_kernel() when attaching CUBIECache

### 3. LRU Eviction in CUBIECache

Add eviction logic to CUBIECache:

**Expected Behavior:**
- Before saving a new cache entry, check total file count in cache directory
- If count >= max_entries, evict oldest files (by mtime) until under limit
- Evict both .nbi and .nbc files together (they're paired)
- Only applies when mode='hash'

**Data Structures:**
- No new structures; uses filesystem for state
- Pairs .nbi/.nbc files by filename base

**Integration Points:**
- Called in CUBIECache before save_overload
- Receives max_entries from CacheConfig
- Respects existing cache_path from CUBIECacheLocator

### 4. Flush-on-Change Mode

Integrate with CUDAFactory._invalidate_cache():

**Expected Behavior:**
- When mode='flush_on_change', deleting cache files when _invalidate_cache()
  is called
- Uses shutil.rmtree on cache directory, then recreates empty directory
- Only one cache entry exists at a time in this mode

**Integration Points:**
- BatchSolverKernel overrides _invalidate_cache()
- Calls parent implementation, then flushes files if mode='flush_on_change'
- CUBIECache still handles save/load normally

### 5. set_cache_dir Method

New method on BatchSolverKernel:

**Expected Behavior:**
- Accepts Path or str for new cache directory
- Updates CacheConfig.cache_dir
- Calls _invalidate_cache() to trigger rebuild
- Next build uses new directory

**Dependencies:**
- Requires CacheConfig to be mutable (default attrs behavior)
- Must update compile_settings to trigger rebuild

### 6. Solver Pass-through Methods

Expose cache methods on Solver class:

**Expected Behavior:**
- Properties: `cache_enabled`, `cache_mode`, `cache_dir`
- Methods: `set_cache_dir(path)`
- All delegate to self.kernel

**Integration Points:**
- Added to Solver class in solver.py
- Mirror pattern of existing pass-through properties

### 7. Session Save/Load Functions

Two module-level functions in cubie/__init__.py or new session.py:

**save_session(solver, name):**
- Extract compile_settings from solver.kernel
- Serialize using pickle
- Save to GENERATED_DIR/sessions/<name>.pkl
- Store session name in a sessions index for lookup

**load_from_session(name):**
- Load from GENERATED_DIR/sessions/<name>.pkl
- Return compile_settings object (not full Solver)
- User creates new Solver and passes settings

**Data Structures:**
- Session file contains pickled compile_settings
- Session index optional (can scan directory)

**Edge Cases:**
- Session name conflicts: overwrite with warning
- Missing session: raise FileNotFoundError
- Incompatible version: let pickle raise naturally

## File Changes Summary

### Modified Files

1. **src/cubie/batchsolving/BatchSolverConfig.py**
   - Add CacheConfig attrs class
   - Add cache_config field to BatchSolverConfig

2. **src/cubie/batchsolving/BatchSolverKernel.py**
   - Add cache parameter parsing in __init__
   - Add set_cache_dir method
   - Override _invalidate_cache for flush mode
   - Update build_kernel to use CacheConfig

3. **src/cubie/batchsolving/solver.py**
   - Add cache pass-through properties
   - Add set_cache_dir method
   - Add cache parameter to __init__

4. **src/cubie/cubie_cache.py**
   - Add evict_lru method
   - Add max_entries parameter to CUBIECache.__init__
   - Add custom_cache_dir support to CUBIECacheLocator

5. **src/cubie/__init__.py**
   - Add save_session, load_from_session to exports

### New Files

1. **src/cubie/session.py**
   - save_session function
   - load_from_session function
   - SESSIONS_DIR constant

## Architectural Changes

### CacheConfig Integration

```
BatchSolverConfig
├── precision
├── loop_fn
├── local_memory_elements
├── shared_memory_elements
├── compile_flags
├── caching_enabled (deprecated, use cache_config)
└── cache_config: CacheConfig
    ├── enabled: bool
    ├── mode: str
    ├── cache_dir: Optional[Path]
    └── max_entries: int
```

### Cache Directory Structure

```
generated/
├── <system_name>/
│   ├── <system_name>.py
│   └── cache/
│       ├── <system>-<hash>.nbi
│       └── <system>-<hash>.1.nbc
└── sessions/
    ├── my_experiment.pkl
    └── baseline_run.pkl
```

## Expected Component Interactions

### Build with LRU Eviction

1. build_kernel() called
2. Create CUBIECache with max_entries from CacheConfig
3. On save_overload:
   a. Count files in cache_path
   b. If >= max_entries, evict oldest by mtime
   c. Save new entry

### Build with Flush-on-Change

1. Settings change via update()
2. _invalidate_cache() called on BatchSolverKernel
3. If mode='flush_on_change':
   a. Delete all files in cache_path
   b. Recreate empty directory
4. build_kernel() creates fresh cache

### Session Workflow

1. User creates Solver, runs experiments
2. User calls save_session(solver, "experiment_v1")
3. Later: settings = load_from_session("experiment_v1")
4. User creates new Solver with those settings

## Edge Cases

1. **Empty cache directory**: Create on first save
2. **Permission errors**: Catch OSError, log warning, continue without cache
3. **Concurrent access**: Use atomic file writes (uuid temp files)
4. **Invalid session file**: Let pickle raise, user handles
5. **Cache dir is symlink**: Follow symlink for operations
6. **Max entries = 0**: Disable LRU (keep all entries)
7. **Max entries = 1**: Immediate eviction after each unique config
