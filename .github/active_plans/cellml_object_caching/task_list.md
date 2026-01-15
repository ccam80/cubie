# Implementation Task List
# Feature: CellML Object Caching
# Plan Reference: .github/active_plans/cellml_object_caching/agent_plan.md

## Task Group 1: Create CellMLCache Class
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/odesystems/symbolic/odefile.py (lines 1-100)
- File: src/cubie/time_logger.py (entire file for TimeLogger interface)
- File: src/cubie/_utils.py (lines 1-50 for PrecisionDType)

**Input Validation Required**:
- model_name: Check type is str, not empty string
- cellml_path: Check type is str, file exists at path
- parsed_equations: Check type is ParsedEquations (attrs class validation)
- indexed_bases: Check type is IndexedBases
- precision: Validate against ALLOWED_PRECISIONS set
- name: Check type is str

**Tasks**:

1. **Create cellml_cache.py module**
   - File: src/cubie/odesystems/symbolic/parsing/cellml_cache.py
   - Action: Create
   - Details:
     ```python
     """Disk-based caching for parsed CellML objects.
     
     This module manages serialization and deserialization of parsed CellML
     data structures using pickle. Cache files are stored in the generated/
     directory alongside compiled code, with hash-based invalidation.
     """
     
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
   - Edge cases: Handle missing GENERATED_DIR, permission errors during write
   - Integration: Uses same GENERATED_DIR as ODEFile for consistency

2. **Implement CellMLCache.__init__()**
   - File: src/cubie/odesystems/symbolic/parsing/cellml_cache.py
   - Action: Create
   - Details:
     ```python
     class CellMLCache:
         """Manage disk-based caching of parsed CellML objects.
         
         Cache files are stored at generated/<model_name>/cellml_cache.pkl
         with the first line containing a SHA256 hash of the source CellML
         file for validation. The remainder of the file is a pickled
         dictionary containing ParsedEquations, IndexedBases, and related
         metadata needed to reconstruct SymbolicODE without re-parsing.
         """
         
         def __init__(self, model_name: str, cellml_path: str) -> None:
             """Initialize cache manager for a CellML model.
             
             Parameters
             ----------
             model_name : str
                 Name used for cache directory (typically cellml filename stem)
             cellml_path : str
                 Path to source CellML file
             
             Raises
             ------
             TypeError
                 If model_name or cellml_path are not strings
             ValueError
                 If model_name is empty string
             FileNotFoundError
                 If cellml_path does not exist
             """
             # Validate inputs (as specified above)
             if not isinstance(model_name, str):
                 raise TypeError(
                     f"model_name must be str, got {type(model_name).__name__}"
                 )
             if not isinstance(cellml_path, str):
                 raise TypeError(
                     f"cellml_path must be str, got {type(cellml_path).__name__}"
                 )
             if not model_name:
                 raise ValueError("model_name cannot be empty string")
             
             cellml_path_obj = Path(cellml_path)
             if not cellml_path_obj.exists():
                 raise FileNotFoundError(
                     f"CellML file not found: {cellml_path}"
                 )
             
             self.model_name = model_name
             self.cellml_path = cellml_path
             self.cache_dir = GENERATED_DIR / model_name
             self.cache_file = self.cache_dir / "cellml_cache.pkl"
     ```
   - Edge cases: Non-existent CellML file raises FileNotFoundError; empty model_name raises ValueError
   - Integration: Stores cache alongside ODEFile generated code in same directory

3. **Implement CellMLCache.get_cellml_hash()**
   - File: src/cubie/odesystems/symbolic/parsing/cellml_cache.py
   - Action: Create
   - Details:
     ```python
     def get_cellml_hash(self) -> str:
         """Compute SHA256 hash of CellML file content.
         
         Reads entire file content and computes hash for cache validation.
         Whitespace changes will change the hash.
         
         Returns
         -------
         str
             Hexadecimal hash string (64 characters)
         
         Raises
         ------
         FileNotFoundError
             If CellML file has been deleted since initialization
         IOError
             If file cannot be read
         """
         # Read file content
         with open(self.cellml_path, 'rb') as f:
             content = f.read()
         
         # Compute SHA256 hash
         hash_obj = sha256(content)
         return hash_obj.hexdigest()
     ```
   - Edge cases: File deleted between init and hash computation; unreadable file
   - Integration: Hash used for cache validation in cache_valid() method

4. **Implement CellMLCache.cache_valid()**
   - File: src/cubie/odesystems/symbolic/parsing/cellml_cache.py
   - Action: Create
   - Details:
     ```python
     def cache_valid(self) -> bool:
         """Check if cache file exists and hash matches current file.
         
         Reads first line of cache file and compares against current
         CellML file hash. Returns False if cache doesn't exist or
         hash mismatch.
         
         Returns
         -------
         bool
             True if cache exists and is current, False otherwise
         """
         # Check cache file exists
         if not self.cache_file.exists():
             return False
         
         try:
             # Read first line (hash)
             with open(self.cache_file, 'r', encoding='utf-8') as f:
                 first_line = f.readline().strip()
             
             # Extract hash (remove # prefix if present)
             stored_hash = first_line.lstrip('#')
             
             # Compute current hash
             current_hash = self.get_cellml_hash()
             
             # Compare hashes
             return stored_hash == current_hash
         
         except Exception:
             # Any error reading cache = invalid cache
             return False
     ```
   - Edge cases: Corrupted cache file returns False; missing cache file returns False; exception during read returns False
   - Integration: Used by load_cellml_model() to decide whether to use cache

5. **Implement CellMLCache.load_from_cache()**
   - File: src/cubie/odesystems/symbolic/parsing/cellml_cache.py
   - Action: Create
   - Details:
     ```python
     def load_from_cache(self) -> Optional[dict]:
         """Load cached parse results from disk.
         
         Reads pickled data from cache file. First line is hash comment,
         remainder is pickled dictionary. Returns None if cache invalid
         or unpickling fails.
         
         Returns
         -------
         dict or None
             Dictionary with keys: 'cellml_hash', 'parsed_equations',
             'indexed_bases', 'all_symbols', 'user_functions', 'fn_hash',
             'precision', 'name'. Returns None if load fails.
         """
         # Validate cache before attempting load
         if not self.cache_valid():
             return None
         
         try:
             # Read file, skip first line (hash comment)
             with open(self.cache_file, 'rb') as f:
                 # Read and discard first line
                 _ = f.readline()
                 # Unpickle remaining content
                 cached_data = pickle.load(f)
             
             # Verify expected keys present
             required_keys = {
                 'cellml_hash', 'parsed_equations', 'indexed_bases',
                 'all_symbols', 'user_functions', 'fn_hash',
                 'precision', 'name'
             }
             if not all(key in cached_data for key in required_keys):
                 default_timelogger.print_message(
                     "Cache file missing required keys, will re-parse"
                 )
                 return None
             
             return cached_data
         
         except pickle.UnpicklingError as e:
             default_timelogger.print_message(
                 f"Cache unpickling failed: {e}, will re-parse"
             )
             return None
         except Exception as e:
             default_timelogger.print_message(
                 f"Cache load error: {e}, will re-parse"
             )
             return None
     ```
   - Edge cases: Corrupted pickle returns None with warning; missing keys returns None; any exception returns None
   - Integration: Called by load_cellml_model() on cache hit

6. **Implement CellMLCache.save_to_cache()**
   - File: src/cubie/odesystems/symbolic/parsing/cellml_cache.py
   - Action: Create
   - Details:
     ```python
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
         
         Creates cache directory if needed, writes hash comment as first
         line, then pickles data dictionary. Silently continues if write
         fails (caching is opportunistic).
         
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
         try:
             # Create cache directory if needed
             self.cache_dir.mkdir(parents=True, exist_ok=True)
             
             # Compute current file hash
             cellml_hash = self.get_cellml_hash()
             
             # Build cache dictionary
             cache_data = {
                 'cellml_hash': cellml_hash,
                 'parsed_equations': parsed_equations,
                 'indexed_bases': indexed_bases,
                 'all_symbols': all_symbols,
                 'user_functions': user_functions,
                 'fn_hash': fn_hash,
                 'precision': precision,
                 'name': name,
             }
             
             # Write cache file: hash comment + pickled data
             with open(self.cache_file, 'wb') as f:
                 # Write hash as first line (text mode for comment)
                 f.write(f"#{cellml_hash}\n".encode('utf-8'))
                 # Pickle data
                 pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
         
         except PermissionError as e:
             default_timelogger.print_message(
                 f"Cannot write cache (permission denied): {e}"
             )
         except Exception as e:
             default_timelogger.print_message(
                 f"Cache save failed: {e}"
             )
     ```
   - Edge cases: Directory creation failure is handled silently; permission errors log warning; any exception during write is caught
   - Integration: Called by load_cellml_model() after successful parse

**Tests to Create**:
- Test file: tests/odesystems/symbolic/test_cellml_cache.py
- Test function: test_cache_initialization_valid_inputs
  - Description: Verify CellMLCache initializes with valid model_name and cellml_path
- Test function: test_cache_initialization_invalid_inputs
  - Description: Verify TypeError raised for non-string inputs, ValueError for empty model_name
- Test function: test_get_cellml_hash_consistent
  - Description: Verify hash is consistent for same file, changes when file changes
- Test function: test_cache_valid_missing_file
  - Description: Verify cache_valid() returns False when cache file doesn't exist
- Test function: test_cache_valid_hash_mismatch
  - Description: Verify cache_valid() returns False when hash doesn't match
- Test function: test_load_from_cache_returns_none_invalid
  - Description: Verify load_from_cache() returns None when cache invalid
- Test function: test_save_and_load_roundtrip
  - Description: Verify data saved to cache can be loaded back correctly
- Test function: test_corrupted_cache_returns_none
  - Description: Verify load_from_cache() handles corrupted pickle gracefully

**Tests to Run**:
- tests/odesystems/symbolic/test_cellml_cache.py::test_cache_initialization_valid_inputs
- tests/odesystems/symbolic/test_cellml_cache.py::test_cache_initialization_invalid_inputs
- tests/odesystems/symbolic/test_cellml_cache.py::test_get_cellml_hash_consistent
- tests/odesystems/symbolic/test_cellml_cache.py::test_cache_valid_missing_file
- tests/odesystems/symbolic/test_cellml_cache.py::test_cache_valid_hash_mismatch
- tests/odesystems/symbolic/test_cellml_cache.py::test_load_from_cache_returns_none_invalid
- tests/odesystems/symbolic/test_cellml_cache.py::test_save_and_load_roundtrip
- tests/odesystems/symbolic/test_cellml_cache.py::test_corrupted_cache_returns_none

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: Register TimeLogger Events
**Status**: [ ]
**Dependencies**: Groups [1]

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/cellml.py (lines 57-75)
- File: src/cubie/time_logger.py (entire file for registration pattern)

**Input Validation Required**:
None (event registration is declarative, no user input)

**Tasks**:

1. **Register cache timing events in cellml.py**
   - File: src/cubie/odesystems/symbolic/parsing/cellml.py
   - Action: Modify
   - Details:
     Add new event registrations after existing cellml timing events (after line 74):
     ```python
     # Existing events at lines 59-74
     default_timelogger.register_event(
         "codegen_cellml_load_model", "codegen",
         "Codegen time for cellmlmanip.load_model()"
     )
     # ... other existing events ...
     
     # NEW: Add cache events after line 74
     default_timelogger.register_event(
         "codegen_cellml_cache_check", "codegen",
         "Codegen time for checking CellML cache validity"
     )
     default_timelogger.register_event(
         "codegen_cellml_cache_load", "codegen",
         "Codegen time for loading from CellML cache"
     )
     default_timelogger.register_event(
         "codegen_cellml_cache_save", "codegen",
         "Codegen time for saving to CellML cache"
     )
     ```
   - Edge cases: None (module-level registration)
   - Integration: Events used in Task Group 3 to time cache operations

**Tests to Create**:
None (event registration tested implicitly by integration tests)

**Tests to Run**:
None

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: Modify load_cellml_model() for Cache Integration
**Status**: [ ]
**Dependencies**: Groups [1, 2]

**Required Context**:
- File: src/cubie/odesystems/symbolic/parsing/cellml.py (lines 105-377)
- File: src/cubie/odesystems/symbolic/symbolicODE.py (lines 142-212)
- File: src/cubie/odesystems/symbolic/parsing/parser.py (lines 1382-1629 - parse_input function)
- File: src/cubie/odesystems/symbolic/parsing/cellml_cache.py (entire file from Task Group 1)

**Input Validation Required**:
None (load_cellml_model already validates inputs; cache operations handle their own errors)

**Tasks**:

1. **Add import for CellMLCache**
   - File: src/cubie/odesystems/symbolic/parsing/cellml.py
   - Action: Modify
   - Details:
     Add import after existing imports (around line 55):
     ```python
     from cubie._utils import PrecisionDType
     from cubie.time_logger import default_timelogger
     from .cellml_cache import CellMLCache  # NEW
     ```
   - Edge cases: None
   - Integration: Makes CellMLCache available in load_cellml_model()

2. **Add cache check at start of load_cellml_model()**
   - File: src/cubie/odesystems/symbolic/parsing/cellml.py
   - Action: Modify
   - Details:
     Insert cache check logic after name determination (after line 205) and before cellmlmanip.load_model():
     ```python
     # Existing code at lines 204-206:
     if name is None:
         name = path_obj.stem
     
     # NEW: Cache check and load
     cache = CellMLCache(model_name=name, cellml_path=path)
     
     default_timelogger.start_event("codegen_cellml_cache_check")
     cache_is_valid = cache.cache_valid()
     default_timelogger.stop_event("codegen_cellml_cache_check")
     
     if cache_is_valid:
         default_timelogger.start_event("codegen_cellml_cache_load")
         cached_data = cache.load_from_cache()
         
         if cached_data is not None:
             # Reconstruct SymbolicODE from cached data
             # Import needed here to avoid circular import
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
             default_timelogger.stop_event("codegen_cellml_cache_load")
             default_timelogger.print_message(
                 f"Loaded {name} from CellML cache"
             )
             return ode
         
         default_timelogger.stop_event("codegen_cellml_cache_load")
     
     # Cache miss or invalid - continue with normal parsing below
     # Existing code continues at line 207: default_timelogger.start_event(...)
     ```
   - Edge cases: Cache invalid returns None and continues to parse; corrupted cache handled by load_from_cache()
   - Integration: Early return bypasses all cellmlmanip parsing; cache miss falls through to existing code

3. **Refactor load_cellml_model() to call parse_input directly**
   - File: src/cubie/odesystems/symbolic/parsing/cellml.py
   - Action: Modify
   - Details:
     Current implementation (lines 362-377) calls SymbolicODE.create() which internally calls parse_input(). We need to call parse_input() directly to cache intermediate results.
     
     REMOVE existing code at lines 362-377:
     ```python
     # OLD CODE TO REMOVE:
     from cubie.odesystems.symbolic.symbolicODE import SymbolicODE
     
     return SymbolicODE.create(
         dxdt=all_equations,
         states=initial_values if initial_values else None,
         parameters=parameters_dict if parameters_dict else None,
         constants=constants_dict if constants_dict else None,
         observables=observables,
         name=name,
         precision=precision,
         strict=False,
         state_units=state_units if state_units else None,
         parameter_units=parameter_units if parameter_units else None,
         observable_units=observable_units if observable_units else None,
     )
     ```
     
     REPLACE with direct parse_input() call and cache save:
     ```python
     # NEW CODE:
     from cubie.odesystems.symbolic.symbolicODE import SymbolicODE
     from cubie.odesystems.symbolic.parsing import parse_input
     
     # Register parsing event (same as SymbolicODE.create)
     default_timelogger.register_event(
         "symbolic_ode_parsing",
         "codegen",
         "Codegen time for symbolic ODE parsing",
     )
     
     # Parse equations into structured components
     default_timelogger.start_event("symbolic_ode_parsing")
     sys_components = parse_input(
         dxdt=all_equations,
         states=initial_values if initial_values else None,
         observables=observables,
         parameters=parameters_dict if parameters_dict else None,
         constants=constants_dict if constants_dict else None,
         drivers=None,
         user_functions=None,
         strict=False,
         state_units=state_units if state_units else None,
         parameter_units=parameter_units if parameter_units else None,
         constant_units=None,
         observable_units=observable_units if observable_units else None,
         driver_units=None,
     )
     index_map, all_symbols, functions, equations, fn_hash = sys_components
     default_timelogger.stop_event("symbolic_ode_parsing")
     
     # Save to cache before creating SymbolicODE
     default_timelogger.start_event("codegen_cellml_cache_save")
     cache.save_to_cache(
         parsed_equations=equations,
         indexed_bases=index_map,
         all_symbols=all_symbols,
         user_functions=functions,
         fn_hash=fn_hash,
         precision=precision,
         name=name,
     )
     default_timelogger.stop_event("codegen_cellml_cache_save")
     
     # Construct SymbolicODE directly (not via .create())
     symbolic_ode = SymbolicODE(
         equations=equations,
         all_indexed_bases=index_map,
         all_symbols=all_symbols,
         name=name,
         fn_hash=fn_hash,
         user_functions=functions,
         precision=precision,
     )
     
     return symbolic_ode
     ```
   - Edge cases: Cache save failure handled silently by CellMLCache.save_to_cache(); parse_input failures propagate as normal
   - Integration: Matches pattern in SymbolicODE.create() but adds cache save step; reconstruction logic matches cache load in Task 3.2

**Tests to Create**:
None (integration tests added in Task Group 4)

**Tests to Run**:
- tests/odesystems/symbolic/test_cellml.py::test_load_simple_cellml_model
- tests/odesystems/symbolic/test_cellml.py::test_load_complex_cellml_model
- tests/odesystems/symbolic/test_cellml.py::test_algebraic_equations_as_observables

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: Add Integration Tests
**Status**: [ ]
**Dependencies**: Groups [1, 2, 3]

**Required Context**:
- File: tests/odesystems/symbolic/test_cellml.py (lines 1-120)
- File: tests/fixtures/cellml/basic_ode.cellml (for test model path)
- File: tests/fixtures/cellml/beeler_reuter_model_1977.cellml (for test model path)
- File: src/cubie/odesystems/symbolic/parsing/cellml.py (entire file)
- File: src/cubie/odesystems/symbolic/parsing/cellml_cache.py (entire file)

**Input Validation Required**:
None (tests validate system behavior)

**Tasks**:

1. **Add cache hit test for basic_ode model**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify
   - Details:
     Add test after existing tests (around line 120):
     ```python
     def test_cache_used_on_reload(cellml_fixtures_dir, tmp_path, monkeypatch):
         """Verify CellML cache is used on second load of same model."""
         import shutil
         from pathlib import Path
         
         # Copy fixture to tmp directory so we can control generated/ location
         tmp_cellml = tmp_path / "basic_ode.cellml"
         shutil.copy(
             cellml_fixtures_dir / "basic_ode.cellml",
             tmp_cellml
         )
         
         # Change working directory to tmp_path for generated/ directory
         import os
         original_cwd = os.getcwd()
         try:
             os.chdir(tmp_path)
             
             # First load - creates cache
             ode1 = load_cellml_model(str(tmp_cellml), name="basic_ode")
             
             # Verify cache file created
             cache_file = tmp_path / "generated" / "basic_ode" / "cellml_cache.pkl"
             assert cache_file.exists(), "Cache file should exist after first load"
             
             # Second load - should use cache
             ode2 = load_cellml_model(str(tmp_cellml), name="basic_ode")
             
             # Verify both ODEs are equivalent
             assert ode1.num_states == ode2.num_states
             assert ode1.fn_hash == ode2.fn_hash
             assert len(ode1.indices.states.index_map) == len(ode2.indices.states.index_map)
         
         finally:
             os.chdir(original_cwd)
     ```
   - Edge cases: Working directory restoration in finally block
   - Integration: Verifies end-to-end cache save and load

2. **Add cache invalidation test**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify
   - Details:
     Add test after previous cache test:
     ```python
     def test_cache_invalidated_on_file_change(cellml_fixtures_dir, tmp_path):
         """Verify cache invalidates when CellML file content changes."""
         import shutil
         import os
         
         # Copy fixture to tmp directory
         tmp_cellml = tmp_path / "basic_ode.cellml"
         shutil.copy(
             cellml_fixtures_dir / "basic_ode.cellml",
             tmp_cellml
         )
         
         original_cwd = os.getcwd()
         try:
             os.chdir(tmp_path)
             
             # First load - creates cache
             ode1 = load_cellml_model(str(tmp_cellml), name="basic_ode")
             cache_file = tmp_path / "generated" / "basic_ode" / "cellml_cache.pkl"
             assert cache_file.exists()
             
             # Modify CellML file (add comment)
             with open(tmp_cellml, 'a') as f:
                 f.write("\n<!-- Modified for test -->\n")
             
             # Verify cache becomes invalid
             from cubie.odesystems.symbolic.parsing.cellml_cache import CellMLCache
             cache = CellMLCache("basic_ode", str(tmp_cellml))
             assert not cache.cache_valid(), "Cache should be invalid after file change"
             
             # Load again - should re-parse and update cache
             ode2 = load_cellml_model(str(tmp_cellml), name="basic_ode")
             
             # Verify new cache is valid
             assert cache.cache_valid(), "Cache should be valid after re-parse"
         
         finally:
             os.chdir(original_cwd)
     ```
   - Edge cases: File modification detection via hash
   - Integration: Verifies cache invalidation works correctly

3. **Add multiple models cache isolation test**
   - File: tests/odesystems/symbolic/test_cellml.py
   - Action: Modify
   - Details:
     Add test after previous cache test:
     ```python
     def test_cache_isolated_per_model(cellml_fixtures_dir, tmp_path):
         """Verify each model has separate cache file."""
         import shutil
         import os
         
         # Copy both fixtures to tmp directory
         tmp_basic = tmp_path / "basic_ode.cellml"
         tmp_br = tmp_path / "beeler_reuter_model_1977.cellml"
         shutil.copy(cellml_fixtures_dir / "basic_ode.cellml", tmp_basic)
         shutil.copy(
             cellml_fixtures_dir / "beeler_reuter_model_1977.cellml",
             tmp_br
         )
         
         original_cwd = os.getcwd()
         try:
             os.chdir(tmp_path)
             
             # Load both models
             ode_basic = load_cellml_model(str(tmp_basic), name="basic_ode")
             ode_br = load_cellml_model(str(tmp_br), name="beeler_reuter_model_1977")
             
             # Verify separate cache files exist
             cache_basic = tmp_path / "generated" / "basic_ode" / "cellml_cache.pkl"
             cache_br = tmp_path / "generated" / "beeler_reuter_model_1977" / "cellml_cache.pkl"
             
             assert cache_basic.exists(), "basic_ode cache should exist"
             assert cache_br.exists(), "beeler_reuter cache should exist"
             
             # Verify different models have different hashes
             assert ode_basic.fn_hash != ode_br.fn_hash
             assert ode_basic.num_states != ode_br.num_states
         
         finally:
             os.chdir(original_cwd)
     ```
   - Edge cases: None
   - Integration: Verifies cache isolation between different models

**Tests to Create**:
(Tests created inline above via modifications to test_cellml.py)

**Tests to Run**:
- tests/odesystems/symbolic/test_cellml.py::test_cache_used_on_reload
- tests/odesystems/symbolic/test_cellml.py::test_cache_invalidated_on_file_change
- tests/odesystems/symbolic/test_cellml.py::test_cache_isolated_per_model
- tests/odesystems/symbolic/test_cellml.py (full suite to verify no regressions)

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Summary

**Total Task Groups**: 4

**Dependency Chain**:
1. Task Group 1 (Create CellMLCache) - No dependencies
2. Task Group 2 (Register TimeLogger Events) - Depends on Group 1
3. Task Group 3 (Modify load_cellml_model) - Depends on Groups 1, 2
4. Task Group 4 (Integration Tests) - Depends on Groups 1, 2, 3

**Tests Overview**:
- **Unit tests**: 8 tests in test_cellml_cache.py validating CellMLCache class
- **Integration tests**: 3 tests in test_cellml.py validating end-to-end caching
- **Regression tests**: Existing test_cellml.py suite ensures no breaking changes

**Estimated Complexity**: Medium
- New CellMLCache class: ~150 lines
- Modifications to load_cellml_model(): ~80 lines changed/added
- Tests: ~150 lines total
- Total implementation: ~380 lines across 3 files

**Key Risks**:
1. SymPy symbol identity preservation across pickle/unpickle (mitigated by testing)
2. File system permissions during cache write (handled with graceful fallback)
3. Concurrent cache access (acceptable - atomic writes at OS level)

**Performance Expectations**:
- First load: ~120-160 seconds (unchanged - must parse)
- Cache hit load: <5 seconds (24x+ speedup)
- Cache overhead: <100ms (hash computation + validation)
