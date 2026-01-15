# CellML Object Caching - Agent Plan

## Component Overview

This plan details the implementation of caching for parsed CellML objects. The feature adds a new `CellMLCache` class and modifies `load_cellml_model()` to check/save cache before/after parsing.

## New Component: CellMLCache Class

**File:** `src/cubie/odesystems/symbolic/parsing/cellml_cache.py`

**Purpose:** Manage serialization and deserialization of parsed CellML data structures using pickle.

**Expected Behavior:**
- Compute SHA256 hash of CellML file content for cache validation
- Check if valid cache exists for given file path and hash
- Load cached `ParsedEquations`, `IndexedBases`, and related data
- Save parsing results to cache after successful parse
- Handle errors gracefully (corrupted cache → re-parse)

**Public Interface:**
```python
class CellMLCache:
    """Manage disk-based caching of parsed CellML objects."""
    
    def __init__(self, model_name: str, cellml_path: str) -> None:
        """Initialize cache manager for a CellML model.
        
        Parameters
        ----------
        model_name : str
            Name used for cache directory (typically cellml filename stem)
        cellml_path : str
            Path to source CellML file
        """
    
    def get_cellml_hash(self) -> str:
        """Compute SHA256 hash of CellML file content.
        
        Returns
        -------
        str
            Hexadecimal hash string
        """
    
    def cache_valid(self) -> bool:
        """Check if cache file exists and hash matches current file.
        
        Returns
        -------
        bool
            True if cache exists and is current
        """
    
    def load_from_cache(self) -> Optional[dict]:
        """Load cached parse results from disk.
        
        Returns
        -------
        dict or None
            Dictionary with keys: 'parsed_equations', 'indexed_bases', 
            'all_symbols', 'user_functions', 'fn_hash', 'precision', 'name'
            Returns None if cache invalid or load fails
        """
    
    def save_to_cache(
        self,
        parsed_equations: ParsedEquations,
        indexed_bases: IndexedBases,
        all_symbols: dict,
        user_functions: Optional[dict],
        fn_hash: str,
        precision: PrecisionDType,
        name: str,
    ) -> None:
        """Save parse results to cache file.
        
        Parameters
        ----------
        parsed_equations : ParsedEquations
            Equation container from parse_input
        indexed_bases : IndexedBases
            Index maps from parse_input
        all_symbols : dict
            Symbol mapping from parse_input
        user_functions : dict or None
            User-provided functions (may be None)
        fn_hash : str
            System hash from parse_input
        precision : PrecisionDType
            Floating-point precision
        name : str
            Model name
        """
```

**Internal Attributes:**
- `cache_dir`: Path to model-specific cache directory (`generated/<model_name>/`)
- `cache_file`: Path to pickle file (`generated/<model_name>/cellml_cache.pkl`)
- `cellml_path`: Path to source CellML file
- `model_name`: Name used for organization

**Error Handling:**
- `pickle.UnpicklingError` → Log warning, return None from `load_from_cache()`
- `FileNotFoundError` → Return None from `load_from_cache()`
- `PermissionError` during save → Log warning, continue without caching
- Hash mismatch → Treat as cache miss, re-parse

**Integration Points:**
- Uses `GENERATED_DIR` from `odefile.py`
- Uses `default_timelogger` from `time_logger.py`
- Imports `ParsedEquations` from `parser.py`
- Imports `IndexedBases` from `indexedbasemaps.py`

## Modified Component: load_cellml_model()

**File:** `src/cubie/odesystems/symbolic/parsing/cellml.py`

**Changes Required:**

### 1. Add Cache Check at Start
**Before:** Function immediately calls `cellmlmanip.load_model()`  
**After:** Check cache first, load from cache if valid

**Expected Flow:**
```python
def load_cellml_model(path, precision=np.float32, name=None, ...):
    # Determine model name
    if name is None:
        name = Path(path).stem
    
    # Initialize cache manager
    cache = CellMLCache(model_name=name, cellml_path=path)
    
    # Try loading from cache
    if cache.cache_valid():
        cached_data = cache.load_from_cache()
        if cached_data is not None:
            # Cache hit - print message
            default_timelogger.print_message(
                f"Loaded {name} from CellML cache at: {cache.cache_file}"
            )
            
            # Reconstruct SymbolicODE from cached data
            ode = SymbolicODE(
                equations=cached_data['parsed_equations'],
                all_indexed_bases=cached_data['indexed_bases'],
                all_symbols=cached_data['all_symbols'],
                fn_hash=cached_data['fn_hash'],
                user_functions=cached_data['user_functions'],
                name=cached_data['name'],
                precision=precision,
            )
            return ode
    
    # Cache miss - print message
    default_timelogger.print_message(
        f"No CellML cache found for {name}, parsing from source..."
    )
    
    # Continue with normal parsing
    # ... existing cellmlmanip code ...
```

### 2. Add Cache Save After Parsing
**Before:** Function returns SymbolicODE directly  
**After:** Save intermediate results before creating SymbolicODE

**Expected Flow:**
```python
def load_cellml_model(path, ...):
    # ... cache check code ...
    # ... existing parsing code ...
    
    # After parse_input returns results:
    sys_components = parse_input(...)
    index_map, all_symbols, functions, equations, fn_hash = sys_components
    
    # Save to cache before creating SymbolicODE
    # No timing events needed - just save silently
    cache.save_to_cache(
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

### 3. TimeLogger Integration
**Location:** Within cache check/load/save flow

**Messages to Print:**
```python
# On cache hit
default_timelogger.print_message(
    f"Loaded {name} from CellML cache at: {cache.cache_file}"
)

# On cache miss
default_timelogger.print_message(
    f"No CellML cache found for {name}, parsing from source..."
)

# Optionally time cache save (no event registration needed)
# Cache save can use existing event timing if desired
```

### 4. Handle user_functions Special Case
**Issue:** User-provided callables may not be picklable  
**Solution:** Detect unpicklable functions, skip cache with warning

**Expected Behavior:**
- If `user_functions` provided to `load_cellml_model()`, skip caching
- Log informational message: "CellML caching disabled due to user_functions"
- Continue normal parsing flow
- Alternatively: attempt to cache, catch pickle error, warn and continue

## Data Structures

### Cached Data Dictionary
```python
{
    'cellml_hash': str,                    # SHA256 of source file
    'parsed_equations': ParsedEquations,   # attrs frozen class
    'indexed_bases': IndexedBases,         # Contains 6 IndexedBaseMap
    'all_symbols': dict[str, sp.Symbol],   # Symbol mapping
    'user_functions': dict | None,         # May be None
    'fn_hash': str,                        # System hash
    'precision': np.dtype,                 # np.float32/64/16
    'name': str,                           # Model name
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
from pathlib import Path
from typing import Optional
import pickle
from hashlib import sha256

from cubie.odesystems.symbolic.odefile import GENERATED_DIR
from cubie.odesystems.symbolic.parsing.parser import ParsedEquations
from cubie.odesystems.symbolic.indexedbasemaps import IndexedBases
from cubie._utils import PrecisionDType
from cubie.time_logger import default_timelogger
```

### New Imports for cellml.py
```python
from .cellml_cache import CellMLCache
```

## Behavioral Requirements

### Cache Hit Behavior
1. User calls `load_cellml_model(path)`
2. Cache manager computes hash of file at `path`
3. Cache manager checks if `generated/<name>/cellml_cache.pkl` exists
4. Cache manager reads first line, validates hash matches
5. Cache manager unpickles data dictionary
6. TimeLogger prints message: "Loaded {name} from CellML cache at: {path}"
7. SymbolicODE constructed directly from cached components
8. Function returns SymbolicODE in <5 seconds

### Cache Miss Behavior
1. User calls `load_cellml_model(path)`
2. Cache check fails (missing file or hash mismatch)
3. TimeLogger prints message: "No CellML cache found for {name}, parsing from source..."
4. Function proceeds with normal cellmlmanip parsing
5. After `parse_input()` completes, results pickled to cache (silently)
6. SymbolicODE created as normal
7. Function returns SymbolicODE (takes ~2 minutes first time)

### Cache Invalidation Behavior
1. User modifies CellML file (even whitespace change)
2. Next call to `load_cellml_model()` computes new hash
3. Hash doesn't match cached hash
4. Cache treated as invalid, re-parsing occurs
5. New cache file overwrites old one with updated hash

### Error Recovery Behavior
1. Cache file corrupted (unpickling fails)
2. `load_from_cache()` catches exception, logs warning
3. Returns None to trigger re-parsing
4. Re-parsing succeeds, overwrites corrupted cache
5. Subsequent loads work normally

## Testing Requirements

### Unit Tests (new file: `tests/odesystems/symbolic/test_cellml_cache.py`)

**Test 1: Cache Creation and Loading**
- Load CellML model fresh (no cache)
- Verify cache file created
- Verify "parsing from source" message printed
- Load same model again
- Verify loaded from cache (check for "Loaded from cache" message)
- Verify SymbolicODE instances equivalent

**Test 2: Hash-Based Invalidation**
- Load CellML model (creates cache)
- Modify CellML file content
- Load again
- Verify cache invalidated, re-parsing occurred

**Test 3: Cache Persistence**
- Load CellML model in one process
- Delete SymbolicODE instance
- Load again in same session
- Verify cache used

**Test 4: Corrupted Cache Handling**
- Load CellML model (creates cache)
- Manually corrupt cache file
- Load again
- Verify graceful fallback to re-parsing

**Test 5: Missing user_functions Handling**
- Load model with user_functions=None
- Verify caching works
- (Optional: test with actual user_functions, verify skip/warning)

### Integration Tests (add to `tests/odesystems/symbolic/test_cellml.py`)

**Test 1: Beeler-Reuter Model Caching**
- Use existing `beeler_reuter_model` fixture
- First load → verify "parsing from source" message
- Second load → verify "Loaded from cache" message
- Compare results from both loads

**Test 2: Multiple Models**
- Load `basic_ode.cellml` and `beeler_reuter_model_1977.cellml`
- Verify separate cache files created
- Verify no cross-contamination

## Edge Cases

### Edge Case 1: Precision Mismatch
**Scenario:** User loads model with `precision=np.float32`, cache exists with `precision=np.float64`  
**Expected Behavior:** Accept cached data, override precision during SymbolicODE construction  
**Rationale:** Precision doesn't affect parsing, only affects compilation later

### Edge Case 2: Parameter Override at Load Time
**Scenario:** User provides `parameters={'param': 1.0}` to `load_cellml_model()`  
**Expected Behavior:** Use cached equations, update parameter values after loading  
**Implementation:** SymbolicODE already supports updating parameters post-construction

### Edge Case 3: Concurrent Cache Access
**Scenario:** Two processes load same model simultaneously  
**Expected Behavior:** Both may write cache; last writer wins; no corruption  
**Rationale:** Pickle writes are atomic at OS level for small files

### Edge Case 4: CellML File Moved
**Scenario:** Cache created, CellML file moved to different path  
**Expected Behavior:** Cache miss, new cache created at new path  
**Implementation:** Cache is path-dependent (uses `model_name` from path)

### Edge Case 5: Generated Directory Missing
**Scenario:** User deletes `generated/` directory  
**Expected Behavior:** Directory auto-created on next save  
**Implementation:** `cache_dir.mkdir(parents=True, exist_ok=True)` in save method

## File Organization

### New File
```
src/cubie/odesystems/symbolic/parsing/cellml_cache.py
```

### Modified Files
```
src/cubie/odesystems/symbolic/parsing/cellml.py
src/cubie/odesystems/symbolic/parsing/__init__.py  (if needed for exports)
```

### New Test Files
```
tests/odesystems/symbolic/test_cellml_cache.py
```

### Modified Test Files
```
tests/odesystems/symbolic/test_cellml.py  (add cache integration tests)
```

### Cache File Locations (auto-generated)
```
generated/<model_name>/cellml_cache.pkl
```

## Implementation Notes

### Note 1: Pickle Protocol Version
Use `pickle.HIGHEST_PROTOCOL` for performance and forward compatibility.

### Note 2: Hash Computation
Read entire file into memory for hashing (acceptable for CellML files <10MB typically).

### Note 3: Symbol Identity Preservation
SymPy Symbol instances must maintain identity across pickle/unpickle for correct comparison in IndexedBases maps.

### Note 4: Backward Compatibility
This is a new feature; no backward compatibility concerns. If pickle format changes in future Python versions, cache will simply be invalidated and regenerated.

### Note 5: TimeLogger Integration Pattern
- Use `default_timelogger.print_message()` for cache hit/miss notifications
- DO NOT register new events for cache check/load
- DO NOT use `start_event()`/`stop_event()` for cache operations
- Pattern matches `cubie_cache.py` lines 401-415

### Note 6: Import Organization
Follow CuBIE conventions: use explicit imports in CUDAFactory-adjacent files, though `cellml_cache.py` is not a CUDAFactory subclass.

### Note 7: Cache Notification Messages
Use `default_timelogger.print_message()` for cache hit/miss notifications, consistent with `cubie_cache.py` pattern (lines 401-415):
- Cache hit: Print informational message with cache file path
- Cache miss: Print informational message that parsing will occur
- No event registration or timing needed for these notifications

## Success Criteria

Implementation is complete when:
1. `CellMLCache` class exists with all specified methods
2. `load_cellml_model()` checks cache before parsing
3. Cache saves after successful parse
4. All unit tests pass
5. Integration tests demonstrate <5s cache loads
6. Corrupted cache handling tested and working
7. Documentation strings complete for all new functions
8. Code passes linting (flake8, ruff)
