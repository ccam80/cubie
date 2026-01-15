# CellML Object Caching with LRU - Agent Plan

## Component Overview

This plan details the implementation of LRU caching for parsed CellML objects with argument-aware cache keys. The feature modifies the existing `CellMLCache` class and `load_cellml_model()` to support multiple configurations per model with automatic LRU eviction.

## Modified Component: CellMLCache Class

**File:** `src/cubie/odesystems/symbolic/parsing/cellml_cache.py`

**Purpose:** Manage LRU caching with argument-based cache keys for parsed CellML data structures using pickle and JSON manifest.

**Expected Behavior:**
- Compute cache key combining file hash and serialized arguments
- Maintain JSON manifest tracking up to 5 cache entries per model
- Load cached data for matching argument configurations
- Save parsing results with automatic LRU eviction when limit exceeded
- Update LRU order on cache hits
- Handle errors gracefully (corrupted cache, missing files)

### New Public Methods:

```python
def compute_cache_key(
    self,
    parameters: Optional[dict],
    observables: Optional[list],
    precision: PrecisionDType,
    name: str
) -> tuple[str, str]:
    """Compute cache key from file hash and arguments.
    
    Parameters
    ----------
    parameters : dict or None
        Parameters dict (only keys matter for parsing)
    observables : list or None
        Observables list
    precision : PrecisionDType
        Floating-point precision
    name : str
        Model name
    
    Returns
    -------
    tuple[str, str]
        (cache_key, args_hash) where cache_key = f"{file_hash}_{args_hash}"
    """

def load_from_cache(
    self,
    cache_key: str,
    args_hash: str
) -> Optional[dict]:
    """Load cached parse results for specific configuration.
    
    Updates LRU order in manifest on successful load.
    
    Parameters
    ----------
    cache_key : str
        Combined file_hash + args_hash key
    args_hash : str
        Arguments hash component
    
    Returns
    -------
    dict or None
        Cached data dictionary or None if not found/invalid
    """

def save_to_cache(
    self,
    cache_key: str,
    args_hash: str,
    args: dict,
    parsed_equations: ParsedEquations,
    indexed_bases: IndexedBases,
    all_symbols: dict,
    user_functions: Optional[dict],
    fn_hash: str,
    precision: PrecisionDType,
    name: str,
) -> None:
    """Save parse results to cache with LRU management.
    
    Evicts least recently used entry if manifest already has 5 entries.
    
    Parameters
    ----------
    cache_key : str
        Combined file_hash + args_hash key
    args_hash : str
        Arguments hash component for filename
    args : dict
        Serialized arguments for manifest metadata
    [... other parameters same as before ...]
    """
```

### New Internal Methods:

```python
def _serialize_args(
    self,
    parameters: Optional[dict],
    observables: Optional[list],
    precision: PrecisionDType,
    name: str
) -> tuple[dict, str]:
    """Serialize arguments for hashing.
    
    Returns
    -------
    tuple[dict, str]
        (args_dict, args_hash_str) - dictionary for manifest and hash
    """

def _load_manifest(self) -> dict:
    """Load manifest from disk or create empty structure.
    
    Returns
    -------
    dict
        Manifest structure with 'file_hash' and 'entries' list
    """

def _save_manifest(self, manifest: dict) -> None:
    """Save manifest to disk atomically.
    
    Parameters
    ----------
    manifest : dict
        Manifest structure to save
    """

def _validate_manifest(self, manifest: dict) -> bool:
    """Check if manifest file_hash matches current file.
    
    Returns
    -------
    bool
        True if manifest valid for current file
    """

def _update_lru_order(
    self,
    manifest: dict,
    args_hash: str
) -> dict:
    """Move entry to end of list (most recent).
    
    Returns
    -------
    dict
        Updated manifest
    """

def _evict_lru(self, manifest: dict) -> dict:
    """Remove least recently used entry (first in list).
    
    Deletes corresponding pickle file.
    
    Returns
    -------
    dict
        Updated manifest with entry removed
    """
```

### Modified Attributes:
- Remove: `cache_file` (single file path)
- Add: `manifest_file` (path to JSON manifest)
- Add: `cache_dir` (already exists, used for individual pickle files)

### Integration Points:
- Uses `GENERATED_DIR` pattern (no change)
- Uses `default_timelogger` for notifications
- Imports `ParsedEquations`, `IndexedBases` (no change)
- New import: `json` for manifest serialization

## Modified Component: load_cellml_model()

**File:** `src/cubie/odesystems/symbolic/parsing/cellml.py`

**Changes Required:**

### 1. Remove Cache Bypass Logic
**Before:** Lines 211: `use_cache = (parameters is None and observables is None)`  
**After:** Remove this logic - always attempt to use cache

**Rationale:** Arguments now part of cache key, so different parameters/observables get separate cache entries

### 2. Compute Cache Key from Arguments
**Before:** Cache check based only on file hash  
**After:** Compute cache key including all arguments

**Expected Flow:**
```python
def load_cellml_model(path, precision=np.float32, name=None, 
                     parameters=None, observables=None, ...):
    # Determine model name
    if name is None:
        name = Path(path).stem
    
    # Initialize cache manager
    cache = CellMLCache(model_name=name, cellml_path=path)
    
    # Compute cache key from file + arguments
    cache_key, args_hash = cache.compute_cache_key(
        parameters=parameters,
        observables=observables,
        precision=precision,
        name=name
    )
    
    # Try loading from cache
    cached_data = cache.load_from_cache(cache_key, args_hash)
    
    if cached_data is not None:
        # Cache hit - reconstruct SymbolicODE
        from cubie.odesystems.symbolic.symbolicODE import SymbolicODE
        
        ode = SymbolicODE(
            equations=cached_data['parsed_equations'],
            all_indexed_bases=cached_data['indexed_bases'],
            all_symbols=cached_data['all_symbols'],
            fn_hash=cached_data['fn_hash'],
            user_functions=cached_data['user_functions'],
            name=cached_data['name'],
            precision=precision,
        )
        default_timelogger.print_message(
            f"Loaded {name} from CellML cache (config {args_hash[:8]})"
        )
        return ode
    
    # Cache miss - print message
    default_timelogger.print_message(
        f"No CellML cache found for {name} with current args, parsing..."
    )
    
    # Continue with normal parsing
    # ... existing cellmlmanip code ...
```

### 3. Update Cache Save to Include Arguments
**Before:** Save without argument context  
**After:** Save with cache_key, args_hash, and serialized args

**Expected Flow:**
```python
def load_cellml_model(path, ...):
    # ... cache check code ...
    # ... existing parsing code ...
    
    # After parse_input returns results:
    sys_components = parse_input(...)
    index_map, all_symbols, functions, equations, fn_hash = sys_components
    
    # Serialize arguments for manifest
    args_dict, _ = cache._serialize_args(
        parameters=parameters,
        observables=observables,
        precision=precision,
        name=name
    )
    
    # Save to cache with LRU management
    cache.save_to_cache(
        cache_key=cache_key,
        args_hash=args_hash,
        args=args_dict,
        parsed_equations=equations,
        indexed_bases=index_map,
        all_symbols=all_symbols,
        user_functions=functions,
        fn_hash=fn_hash,
        precision=precision,
        name=name,
    )
    
    # Continue with SymbolicODE.create() as before
    symbolic_ode = SymbolicODE.create(...)
    return symbolic_ode
```

### 4. Update TimeLogger Messages
**Location:** Cache hit/miss notification points

**New Messages:**
```python
# On cache hit
default_timelogger.print_message(
    f"Loaded {name} from CellML cache (config {args_hash[:8]})"
)

# On cache miss
default_timelogger.print_message(
    f"No CellML cache found for {name} with current args, parsing..."
)
```

### 5. Remove user_functions Skip Logic
**Before:** Skip caching if user_functions provided  
**After:** Include user_functions in cached data (if picklable, otherwise cache without)

**Expected Behavior:**
- Attempt to cache even with user_functions
- If pickle fails during save, log warning and continue without caching
- No special handling needed in load_cellml_model()

## Data Structures

### Cached Data Dictionary (per configuration)
```python
{
    'cellml_hash': str,                    # SHA256 of source file
    'args_hash': str,                      # SHA256 of serialized args
    'parsed_equations': ParsedEquations,   # attrs frozen class
    'indexed_bases': IndexedBases,         # Contains 6 IndexedBaseMap
    'all_symbols': dict[str, sp.Symbol],   # Symbol mapping
    'user_functions': dict | None,         # May be None
    'fn_hash': str,                        # System hash
    'precision': np.dtype,                 # np.float32/64/16
    'name': str,                           # Model name
}
```

### Manifest Structure (JSON file)
```python
{
    'file_hash': str,          # SHA256 of CellML file
    'entries': [               # List ordered by LRU (oldest first, newest last)
        {
            'cache_key': str,           # "{file_hash}_{args_hash}"
            'args_hash': str,           # Arguments hash component
            'args': {                   # Serialized arguments
                'parameters': list[str] | None,  # Sorted parameter names
                'observables': list[str] | None, # Sorted observable names
                'precision': str,                # "float32", "float64", etc.
                'name': str                      # Model name
            },
            'last_accessed': float,     # Unix timestamp
            'cache_file': str,          # "cache_{args_hash}.pkl"
        },
        # ... up to 5 entries total
    ]
}
```

### Serialized Arguments Dictionary
```python
{
    'parameters': sorted(parameters.keys()) if parameters else None,
    'observables': sorted(observables) if observables else None,
    'precision': str(precision),  # Convert numpy dtype to string
    'name': name
}
```

### ParsedEquations (already exists, no changes)
```python
@attrs.define(frozen=True)
class ParsedEquations:
    ordered: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    state_derivatives: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    observables: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    auxiliaries: Tuple[Tuple[sp.Symbol, sp.Expr], ...]
    _state_symbols: frozenset[sp.Symbol]
    _observable_symbols: frozenset[sp.Symbol]
    _auxiliary_symbols: frozenset[sp.Symbol]
```

### IndexedBases (already exists, no changes)
```python
class IndexedBases:
    states: IndexedBaseMap
    parameters: IndexedBaseMap
    constants: IndexedBaseMap
    observables: IndexedBaseMap
    drivers: IndexedBaseMap
    dxdt: IndexedBaseMap
```

### IndexedBaseMap (already exists, no changes)
Each map contains:
- `base_name`: str
- `length`: int
- `real`: bool
- `base`: sp.IndexedBase
- `index_map`: dict[sp.Symbol, int]
- `ref_map`: dict[sp.Symbol, sp.Indexed]
- `symbol_map`: dict[str, sp.Symbol]
- `default_values`: dict[sp.Symbol, float]
- `defaults`: dict[str, float]
- `units`: dict[str, str]

## Dependencies

### New Imports for cellml_cache.py
```python
import json
import time
from pathlib import Path
from typing import Optional
import pickle
from hashlib import sha256

from cubie.odesystems.symbolic.parsing.parser import ParsedEquations
from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases
from cubie._utils import PrecisionDType
from cubie.time_logger import default_timelogger
```

### Imports for cellml.py (no changes needed)
```python
from .cellml_cache import CellMLCache  # Already exists
```

## Behavioral Requirements

### Cache Hit Behavior
1. User calls `load_cellml_model(path, parameters={'V': ...}, observables=['I_Na'])`
2. Cache manager computes file hash and args hash
3. Cache manager checks manifest for matching entry (by cache_key)
4. Cache manager unpickles data from `cache_{args_hash}.pkl`
5. Cache manager updates LRU order in manifest (moves entry to end)
6. TimeLogger prints: "Loaded {name} from CellML cache (config {args_hash[:8]})"
7. SymbolicODE constructed directly from cached components
8. Function returns SymbolicODE in <5 seconds

### Cache Miss with Available Slot
1. User calls `load_cellml_model(path, parameters={'V': ...})`
2. Cache check fails (args_hash not in manifest)
3. Manifest has <5 entries (no eviction needed)
4. TimeLogger prints: "No CellML cache found for {name} with current args, parsing..."
5. Function proceeds with normal cellmlmanip parsing (~2 minutes)
6. After `parse_input()` completes, results saved to `cache_{args_hash}.pkl`
7. Manifest updated with new entry (appended to end)
8. SymbolicODE created as normal
9. Function returns SymbolicODE

### Cache Miss with Full Manifest (5 entries)
1. User calls `load_cellml_model(path, observables=['new_obs'])`
2. Cache check fails (args_hash not in manifest)
3. Manifest has 5 entries (eviction required)
4. Cache manager evicts first entry (oldest/least recent)
5. TimeLogger prints: "Evicted LRU cache entry for {name}"
6. Corresponding pickle file deleted
7. Continue with parsing and cache save as above
8. New entry appended to manifest (now has 5 entries again)

### File Change Invalidation
1. User modifies CellML file (even whitespace change)
2. Next call to `load_cellml_model()` computes new file hash
3. Manifest file_hash doesn't match current file hash
4. **All cached entries invalidated** (entire manifest replaced)
5. All old pickle files remain but are orphaned (not loaded)
6. New manifest created with empty entries list
7. Parsing occurs, new cache entry created

### LRU Update on Cache Hit
1. User loads config A (cached)
2. User loads config B (cached)
3. User loads config A again (still cached)
4. Manifest order: [oldest, ..., B, A] (A moved to end)
5. Next eviction will remove entry at position 0 (not A or B)

### Error Recovery Behavior
1. Manifest file corrupted (invalid JSON)
2. `_load_manifest()` catches exception, returns empty manifest
3. Existing pickle files orphaned but not deleted
4. New manifest built from scratch
5. Subsequent loads work normally

6. Cache pickle corrupted
7. `load_from_cache()` catches exception, returns None
8. Triggers re-parsing
9. Re-parsing saves new valid cache
10. Subsequent loads work normally

## Testing Requirements

### Unit Tests (modified file: `tests/odesystems/symbolic/test_cellml_cache.py`)

**Test 1: Argument Serialization Consistency**
- Serialize same arguments in different orders
- Verify hash is identical
- Verify None parameters vs empty dict produce different hashes
- Verify sorted parameter keys produce consistent hash

**Test 2: Cache Key Computation**
- Compute cache key with different argument combinations
- Verify file_hash component is consistent
- Verify args_hash component changes with arguments
- Verify combined cache_key format is correct

**Test 3: LRU Eviction Logic**
- Create cache with 5 entries
- Add 6th entry
- Verify oldest entry evicted
- Verify corresponding pickle file deleted
- Verify manifest has exactly 5 entries

**Test 4: LRU Order Update**
- Cache 3 configurations
- Load config 1 (should move to end)
- Load config 2 (should move to end)
- Verify manifest order updated correctly

**Test 5: Manifest Validation**
- Load model, create cache
- Modify CellML file
- Attempt to load from cache
- Verify manifest invalidated (file_hash mismatch)
- Verify empty manifest returned

**Test 6: Multiple Configurations**
- Load model with parameters=None
- Load same model with parameters={'V': ...}
- Load same model with observables=['I_Na']
- Verify 3 separate cache entries created
- Verify each can be loaded independently

**Test 7: Corrupted Manifest Handling**
- Create valid cache
- Manually corrupt manifest JSON
- Attempt to load
- Verify graceful fallback (empty manifest returned)

**Test 8: Orphaned Cache Files**
- Create cache entries
- Manually delete manifest
- Load from cache (should rebuild manifest)
- Verify orphaned pickle files not loaded

### Integration Tests (add to `tests/odesystems/symbolic/test_cellml.py`)

**Test 1: Parameter Variation Caching**
- Load beeler_reuter_model with no parameters
- Load again with parameters={'V': ...}
- Verify both cached separately
- Verify cache hits on reload for each

**Test 2: Observable Variation Caching**
- Load model with default observables
- Load with custom observables list
- Verify separate cache entries
- Verify correct equations loaded for each

**Test 3: LRU Eviction Integration**
- Load model with 6 different parameter combinations
- Verify 6th load evicts 1st configuration
- Reload 1st configuration (should re-parse)
- Reload 2nd configuration (should hit cache)

**Test 4: File Change Invalidates All Configs**
- Cache 3 different configurations
- Modify CellML file
- Attempt to load any configuration
- Verify all cache entries invalidated
- Verify re-parsing occurs

**Test 5: Precision in Cache Key**
- Load model with precision=np.float32
- Load same model with precision=np.float64
- Verify separate cache entries
- Verify correct precision in each loaded ODE

## Edge Cases

### Edge Case 1: Identical Parameters, Different Values
**Scenario:** User loads model with `parameters={'V': 1.0}`, then `parameters={'V': 2.0}`  
**Expected Behavior:** Both use same cache entry (values don't affect parsing)  
**Implementation:** Serialize only parameter keys, not values

### Edge Case 2: Parameter Order Variation
**Scenario:** User provides `{'a': 1, 'b': 2}` vs `{'b': 2, 'a': 1}`  
**Expected Behavior:** Same cache entry (order doesn't matter)  
**Implementation:** Sort parameter keys before hashing

### Edge Case 3: Observable Order Variation
**Scenario:** User provides `['I_Na', 'I_K']` vs `['I_K', 'I_Na']`  
**Expected Behavior:** Same cache entry (order doesn't affect parsing)  
**Implementation:** Sort observable names before hashing

### Edge Case 4: Manifest Full During Save
**Scenario:** 5 cached configs exist, user adds 6th with different args  
**Expected Behavior:** Evict oldest (first in list), add new (at end)  
**Implementation:** `_evict_lru()` removes entry 0, deletes pickle, then append new

### Edge Case 5: Concurrent Access to Manifest
**Scenario:** Two processes try to update manifest simultaneously  
**Expected Behavior:** Last writer wins; manifest may miss one update but stays consistent  
**Rationale:** JSON writes are atomic for small files; rare race condition acceptable

### Edge Case 6: File Deleted Between Init and Load
**Scenario:** CellML file exists at init, deleted before `get_cellml_hash()`  
**Expected Behavior:** FileNotFoundError raised during hash computation  
**Implementation:** Existing behavior in `get_cellml_hash()` method

### Edge Case 7: Precision as np.dtype vs string
**Scenario:** User passes `np.float32` vs `"float32"` as precision  
**Expected Behavior:** Both serialize to same string, same cache key  
**Implementation:** `str(precision.dtype if hasattr(precision, 'dtype') else precision)`

### Edge Case 8: Orphaned Cache Files After Manifest Reset
**Scenario:** File changes, manifest invalidated, old pickles remain on disk  
**Expected Behavior:** Old pickles ignored (manifest is source of truth)  
**Cleanup:** Optional manual cleanup, not automatic

### Edge Case 9: Empty Parameters Dict vs None
**Scenario:** User provides `parameters={}` vs `parameters=None`  
**Expected Behavior:** Different cache entries (empty dict might have been filtered)  
**Implementation:** Serialize as `None` vs `[]` (sorted keys of empty dict)

### Edge Case 10: Max 5 Entries with Frequent Access Pattern
**Scenario:** User alternates between 6 different configurations  
**Expected Behavior:** 6th config never cached (always evicted on next access)  
**Acceptable:** User should use fewer configurations or increase limit (future enhancement)

## File Organization

### Modified Files
```
src/cubie/odesystems/symbolic/parsing/cellml_cache.py  (major changes)
src/cubie/odesystems/symbolic/parsing/cellml.py        (moderate changes)
```

### Modified Test Files
```
tests/odesystems/symbolic/test_cellml_cache.py         (new tests added)
tests/odesystems/symbolic/test_cellml.py               (integration tests added)
```

### Cache File Locations (auto-generated)
```
generated/<model_name>/
  cellml_cache_manifest.json           # LRU manifest (NEW)
  cache_<args_hash_1>.pkl              # Config 1 cache (NEW naming)
  cache_<args_hash_2>.pkl              # Config 2 cache (NEW naming)
  ...                                  # Up to 5 pickle files
  <model_name>.py                      # Existing codegen cache (unchanged)
```

### Removed Files
```
generated/<model_name>/cellml_cache.pkl  # Old single-file cache (replaced)
```

## Implementation Notes

### Note 1: LRU Order in Manifest
Entries list is ordered oldest-first, newest-last. Index 0 is always evicted when full.

### Note 2: Argument Hashing for Cache Keys
Use `json.dumps()` with `sort_keys=True` to ensure deterministic serialization across Python sessions.

### Note 3: Timestamp for LRU Tracking
Use `time.time()` for last_accessed timestamps. Update on both cache save and cache load.

### Note 4: Atomic Manifest Updates
Pattern:
1. Load manifest
2. Modify in memory
3. Write entire manifest atomically
4. No partial updates

### Note 5: Pickle Protocol Version
Use `pickle.HIGHEST_PROTOCOL` for performance (unchanged from current implementation).

### Note 6: Hash Computation
File hash: SHA256 of entire CellML file (unchanged)
Args hash: SHA256 of JSON-serialized args dict
Cache key: f"{file_hash}_{args_hash}"

### Note 7: TimeLogger Integration Pattern
- Use `default_timelogger.print_message()` for cache hit/miss/eviction notifications
- DO NOT register new events for cache operations
- DO NOT use `start_event()`/`stop_event()` for cache operations
- Pattern matches `cubie_cache.py` lines 401-415

### Note 8: Backward Compatibility
Existing single-file caches (cellml_cache.pkl) will be ignored. Users will need to re-parse once to populate new multi-config cache structure.

### Note 9: Manifest Schema Version
Consider adding `"version": 1` to manifest for future schema migrations (optional but recommended).

### Note 10: Cache File Naming
Use `cache_{args_hash}.pkl` pattern to avoid collisions and enable per-config deletion.

## Success Criteria

Implementation is complete when:
1. `CellMLCache` class modified with LRU manifest support
2. Argument-based cache key computation implemented
3. `load_cellml_model()` removes cache bypass logic (line 211)
4. `load_cellml_model()` passes arguments to cache key computation
5. Manifest tracks up to 5 cache entries with LRU eviction
6. All unit tests pass (8+ tests covering LRU, multi-config, eviction)
7. Integration tests demonstrate cache hits with different parameter/observable combinations
8. File change invalidates all cached configurations
9. LRU order updated on cache hits
10. Corrupted manifest/cache handling tested and working
11. Documentation strings complete for all modified functions
12. Code passes linting (flake8, ruff)
13. Backward compatibility note added (old caches ignored)
