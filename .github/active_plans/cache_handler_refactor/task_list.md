# Implementation Task List
# Feature: CacheHandler Refactor for Persistent Cache Interface
# Plan Reference: .github/active_plans/cache_handler_refactor/agent_plan.md

## Task Group 1: CacheConfig Enhancement and ALL_CACHE_PARAMETERS
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/cubie_cache.py (lines 556-627, CacheConfig class)
- File: src/cubie/CUDAFactory.py (lines 80-235, _CubieConfigBase class)
- File: src/cubie/_utils.py (lines 723-833, build_config function)
- File: src/cubie/batchsolving/solver.py (lines 1-40, import section)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- cache_enabled: bool type check via attrs validator
- cache_mode: string in ('hash', 'flush_on_change') via attrs validator
- max_cache_entries: int >= 0 via getype_validator
- cache_dir: Optional[Path] or Optional[str] via attrs validator
- system_hash: str type check via attrs validator

**Tasks**:
1. **Add ALL_CACHE_PARAMETERS constant set**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     ```python
     # Add at module level, after imports, before CUBIECacheLocator class
     ALL_CACHE_PARAMETERS: Set[str] = {
         "cache_enabled",
         "cache_mode", 
         "max_cache_entries",
         "cache_dir",
     }
     ```
   - Edge cases: None - this is a constant set
   - Integration: Will be imported by Solver to filter cache-related kwargs

2. **Rename CacheConfig.enabled to cache_enabled**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     - Rename field `enabled` to `cache_enabled` at line 577
     - Update the `from_user_setting` classmethod if it references `enabled`
     - Update any references to `.enabled` in the module to `.cache_enabled`
     - Current code at line 577:
       ```python
       enabled: bool = field(
           default=False,
           validator=val.instance_of(bool),
       )
       ```
     - New code:
       ```python
       cache_enabled: bool = field(
           default=False,
           validator=val.instance_of(bool),
       )
       ```
   - Edge cases: Ensure backwards compatibility is NOT maintained (per plan requirements)
   - Integration: CubieCacheHandler uses this field

3. **Add system_name field to CacheConfig**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     - Add `system_name: str` field after `system_hash` field (around line 598)
     ```python
     system_name: str = field(
         default="",
         validator=val.instance_of(str),
     )
     ```
   - Edge cases: Empty string as default allows lazy initialization
   - Integration: Used by CubieCacheHandler to construct cache paths

4. **Export ALL_CACHE_PARAMETERS in cubie_cache imports**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     - Add `Set` to typing imports at top of file
     - Ensure ALL_CACHE_PARAMETERS is defined at module level

**Tests to Create**:
- Test file: tests/test_cubie_cache.py
- Test function: test_all_cache_parameters_contains_expected_keys
- Description: Verify ALL_CACHE_PARAMETERS contains cache_enabled, cache_mode, max_cache_entries, cache_dir
- Test function: test_cache_config_has_cache_enabled_field
- Description: Verify CacheConfig has cache_enabled field (not enabled)
- Test function: test_cache_config_has_system_name_field  
- Description: Verify CacheConfig has system_name field

**Tests to Run**:
- tests/test_cubie_cache.py::test_all_cache_parameters_contains_expected_keys
- tests/test_cubie_cache.py::test_cache_config_has_cache_enabled_field
- tests/test_cubie_cache.py::test_cache_config_has_system_name_field

**Outcomes**:
- Files Modified:
  * src/cubie/cubie_cache.py (15 lines changed)
  * tests/test_cubie_cache.py (32 lines changed)
- Functions/Methods Added/Modified:
  * ALL_CACHE_PARAMETERS constant added at module level
  * CacheConfig class: renamed `enabled` to `cache_enabled`, added `system_name` field
  * CubieCacheHandler.invalidate(): updated reference from `.enabled` to `.cache_enabled`
- Implementation Summary:
  Added ALL_CACHE_PARAMETERS constant set at module level containing the four cache parameter names. Added Set import to typing imports. Renamed enabled field to cache_enabled in CacheConfig class and updated the docstring. Added system_name field to CacheConfig with empty string default. Updated CubieCacheHandler.invalidate() to use cache_enabled instead of enabled.
- Issues Flagged: Note that create_cache() and invalidate_cache() functions reference non-existent CacheConfig.from_user_setting() method - this is pre-existing broken code, not related to this task. 

---

## Task Group 2: CUBIECacheLocator Path Refactor for Hierarchical Structure
**Status**: [x]
**Dependencies**: Group 1

**Required Context**:
- File: src/cubie/cubie_cache.py (lines 37-132, CUBIECacheLocator class)
- File: src/cubie/odesystems/symbolic/odefile.py (GENERATED_DIR constant)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- system_name: Must be non-empty string when used for path construction
- system_hash: Must be non-empty string when used for path construction
- If system_hash is empty, use "default" subdirectory

**Tasks**:
1. **Modify CUBIECacheLocator path construction to include system_hash**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     - Current code at line 68:
       ```python
       self._cache_path = GENERATED_DIR / system_name / "CUDA_cache"
       ```
     - New code:
       ```python
       # Use "default" if system_hash is empty
       hash_dir = system_hash if system_hash else "default"
       self._cache_path = GENERATED_DIR / system_name / hash_dir / "CUDA_cache"
       ```
   - Edge cases: 
     - Empty system_hash uses "default" subdirectory
     - Custom cache_dir overrides entire path (existing behavior preserved)
   - Integration: All cache file operations now use hierarchical structure

2. **Update set_system_hash to optionally rebuild path**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     - Current code at lines 110-118:
       ```python
       def set_system_hash(self, system_hash: str) -> None:
           """Update the system hash.
           ...
           """
           self._system_hash = system_hash
       ```
     - New code - add path update if using default location:
       ```python
       def set_system_hash(self, system_hash: str) -> None:
           """Update the system hash and refresh cache path.

           Parameters
           ----------
           system_hash
               New system hash to set.
           """
           self._system_hash = system_hash
           # Only update path if not using custom cache directory
           if not hasattr(self, '_custom_cache_dir') or \
              self._custom_cache_dir is None:
               hash_dir = system_hash if system_hash else "default"
               self._cache_path = (
                   GENERATED_DIR / self._system_name / hash_dir / "CUDA_cache"
               )
       ```
   - Edge cases: Custom cache_dir should NOT be affected by hash changes
   - Integration: CUBIECacheImpl.set_hashes delegates here

3. **Store custom_cache_dir flag in CUBIECacheLocator init**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     - Add `self._custom_cache_dir = custom_cache_dir` before path computation
     - This allows set_system_hash to know whether to update path

**Tests to Create**:
- Test file: tests/test_cubie_cache.py
- Test function: test_cache_locator_path_includes_system_hash
- Description: Verify cache path follows generated/<system_name>/<system_hash>/CUDA_cache/ structure
- Test function: test_cache_locator_empty_hash_uses_default
- Description: Verify empty system_hash uses "default" subdirectory
- Test function: test_cache_locator_custom_dir_not_affected_by_hash_change
- Description: Verify custom cache_dir is preserved when set_system_hash is called

**Tests to Run**:
- tests/test_cubie_cache.py::test_cache_locator_path_includes_system_hash
- tests/test_cubie_cache.py::test_cache_locator_empty_hash_uses_default
- tests/test_cubie_cache.py::test_cache_locator_custom_dir_not_affected_by_hash_change

**Outcomes**:
- Files Modified:
  * src/cubie/cubie_cache.py (12 lines changed)
  * tests/test_cubie_cache.py (47 lines added)
- Functions/Methods Added/Modified:
  * CUBIECacheLocator.__init__(): Added _custom_cache_dir storage, added system_hash subdirectory to path construction
  * CUBIECacheLocator.set_system_hash(): Updated to rebuild path when not using custom cache directory
- Implementation Summary:
  Modified CUBIECacheLocator to use hierarchical path structure with system_hash subdirectory. Path pattern is now `generated/<system_name>/<system_hash>/CUDA_cache/`. Empty system_hash uses "default" as the subdirectory name. The custom_cache_dir is stored in init and used by set_system_hash to determine whether to rebuild the path when the hash changes.
- Issues Flagged: None

---

## Task Group 3: CubieCacheHandler Refactor
**Status**: [x]
**Dependencies**: Groups 1, 2

**Required Context**:
- File: src/cubie/cubie_cache.py (lines 629-709, CubieCacheHandler class)
- File: src/cubie/cubie_cache.py (lines 465-504, create_cache function)
- File: src/cubie/cubie_cache.py (lines 556-627, CacheConfig class)
- File: src/cubie/_utils.py (lines 723-833, build_config function)
- File: src/cubie/CUDAFactory.py (lines 127-198, _CubieConfigBase.update method)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- cache_arg: Union[bool, str, Path, None] - no additional validation needed (handled by CacheConfig.params_from_user_kwarg)
- system_name: str - must be provided or default to system_hash[:12]
- system_hash: str - must be provided for cache to function correctly
- kwargs: Only ALL_CACHE_PARAMETERS keys are relevant

**Tasks**:
1. **Fix CubieCacheHandler.__init__ signature and instantiation**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     - Current broken code at line 656:
       ```python
       self._cache = create_cache(_config)
       ```
     - The `create_cache` function expects different arguments. Fix the initialization:
       ```python
       def __init__(
           self,
           cache_arg: Union[bool, str, Path] = None,
           system_name: str = "",
           system_hash: str = "",
           **kwargs
       ) -> None:
           # Convert single cache arg into cache_enabled, path kwargs
           config_params = CacheConfig.params_from_user_kwarg(cache_arg)
           kwargs.update(config_params)
           kwargs["system_name"] = system_name
           kwargs["system_hash"] = system_hash
           
           # Build CacheConfig using build_config utility
           _config = build_config(
               CacheConfig,
               {},  # No required params - all have defaults
               **kwargs
           )
           self.config = _config
           
           # Create cache if enabled
           if _config.cache_enabled:
               self._cache = CUBIECache(
                   system_name=system_name,
                   system_hash=system_hash,
                   config_hash="",  # Will be set at run time
                   max_entries=_config.max_cache_entries,
                   mode=_config.cache_mode,
                   custom_cache_dir=_config.cache_dir,
               )
           else:
               self._cache = None
       ```
   - Edge cases:
     - cache_arg=None or False: _cache is None
     - Empty system_name: Use first 12 chars of system_hash
   - Integration: Called by BatchSolverKernel.__init__

2. **Fix CubieCacheHandler.update to return set of recognized params**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     - Current code at lines 667-683:
       ```python
       def update(self, updates_dict, **kwargs) -> None:
           ...
           recognized, _ = self.config.update(updates_dict)
           self._cache.update_from_config(self.config)
           return recognized
       ```
     - Fix signature and add handling for None cache:
       ```python
       def update(
           self,
           updates_dict: Optional[Dict[str, Any]] = None,
           silent: bool = False,
           **kwargs
       ) -> Set[str]:
           """Update cache configuration and recreate cache if needed.

           Parameters
           ----------
           updates_dict
               Dictionary of configuration updates.
           silent
               Suppress errors for unrecognized parameters.
           **kwargs
               Additional configuration overrides.

           Returns
           -------
           Set[str]
               Set of recognized parameter names.
           """
           if updates_dict is None:
               updates_dict = {}
           updates_dict = updates_dict.copy()
           updates_dict.update(kwargs)
           
           if not updates_dict:
               return set()

           recognized, changed = self.config.update(updates_dict)
           
           # Update cache if it exists and settings changed
           if self._cache is not None and changed:
               self._cache.update_from_config(self.config)
           
           # Handle cache being enabled via update
           if "cache_enabled" in changed and self.config.cache_enabled:
               if self._cache is None:
                   self._cache = CUBIECache(
                       system_name=self.config.system_name,
                       system_hash=self.config.system_hash,
                       config_hash="",
                       max_entries=self.config.max_cache_entries,
                       mode=self.config.cache_mode,
                       custom_cache_dir=self.config.cache_dir,
                   )

           return recognized
       ```
   - Edge cases:
     - Cache disabled then enabled: create new cache
     - Cache None when update called: skip update_from_config
   - Integration: Called by BatchSolverKernel.update

3. **Fix CubieCacheHandler.configured_cache method**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     - Current code at lines 685-698 has potential None access:
       ```python
       def configured_cache(self, compile_settings_hash: str) -> Optional[CUBIECache]:
           self._cache.set_hashes(...)  # Will fail if _cache is None
           return self._cache
       ```
     - Fix to handle None cache:
       ```python
       def configured_cache(
           self, compile_settings_hash: str
       ) -> Optional[CUBIECache]:
           """Return a CUBIECache configured with current hashes.

           Parameters
           ----------
           compile_settings_hash
               Hash of compile settings for cache disambiguation.

           Returns
           -------
           CUBIECache or None
               Configured cache instance if enabled, else None.
           """
           if self._cache is None:
               return None
               
           self._cache.set_hashes(
               system_hash=self.config.system_hash,
               compile_settings_hash=compile_settings_hash,
           )
           return self._cache
       ```
   - Edge cases: Cache disabled returns None
   - Integration: Called by BatchSolverKernel.run

4. **Fix CubieCacheHandler.flush and invalidate methods**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     - Add None checks to flush() and invalidate():
       ```python
       def flush(self) -> None:
           """Flush the managed cache."""
           if self._cache is not None:
               self._cache.flush_cache()

       def invalidate(self) -> None:
           """Invalidate the managed cache if in flush_on_change mode."""
           if self._cache is None:
               return
           if self.config.cache_mode != "flush_on_change":
               return
           self.flush()
       ```
   - Edge cases: All methods must handle None cache gracefully
   - Integration: Called during cache invalidation flow

5. **Add missing imports to CubieCacheHandler section**
   - File: src/cubie/cubie_cache.py
   - Action: Modify
   - Details:
     - Ensure `Set`, `Dict`, `Any`, `Optional` are imported from typing
     - Current imports may be missing some of these

**Tests to Create**:
- Test file: tests/test_cubie_cache.py
- Test function: test_cache_handler_init_with_disabled_cache
- Description: Verify CubieCacheHandler initializes with None cache when cache_arg=False
- Test function: test_cache_handler_init_with_enabled_cache
- Description: Verify CubieCacheHandler creates CUBIECache when cache_arg=True
- Test function: test_cache_handler_update_returns_recognized_params
- Description: Verify update() returns set of recognized parameter names
- Test function: test_cache_handler_configured_cache_returns_none_when_disabled
- Description: Verify configured_cache() returns None when cache disabled
- Test function: test_cache_handler_configured_cache_sets_hashes
- Description: Verify configured_cache() updates system and compile hashes
- Test function: test_cache_handler_flush_handles_none_cache
- Description: Verify flush() does not error when cache is None
- Test function: test_cache_handler_enable_cache_via_update
- Description: Verify cache can be enabled via update(cache_enabled=True)

**Tests to Run**:
- tests/test_cubie_cache.py::test_cache_handler_init_with_disabled_cache
- tests/test_cubie_cache.py::test_cache_handler_init_with_enabled_cache
- tests/test_cubie_cache.py::test_cache_handler_update_returns_recognized_params
- tests/test_cubie_cache.py::test_cache_handler_configured_cache_returns_none_when_disabled
- tests/test_cubie_cache.py::test_cache_handler_configured_cache_sets_hashes
- tests/test_cubie_cache.py::test_cache_handler_flush_handles_none_cache
- tests/test_cubie_cache.py::test_cache_handler_enable_cache_via_update

**Outcomes**:
- Files Modified:
  * src/cubie/cubie_cache.py (45 lines changed)
  * tests/test_cubie_cache.py (85 lines added)
- Functions/Methods Added/Modified:
  * CubieCacheHandler.__init__(): Fixed to create CUBIECache directly when enabled
  * CubieCacheHandler.flush(): Added None check for cache
  * CubieCacheHandler.update(): Fixed to return Set[str], handle None cache and None updates_dict, handle enabling cache via update
  * CubieCacheHandler.configured_cache(): Fixed to return None when cache disabled
  * CubieCacheHandler.invalidate(): Fixed to handle None cache gracefully
- Implementation Summary:
  Refactored CubieCacheHandler to properly handle cache=None case throughout. The __init__ method now creates CUBIECache directly when cache_enabled is True, rather than calling the broken create_cache function. The update method properly returns Set[str] of recognized parameters and can enable caching at runtime. All methods that access self._cache now check for None first. Added Dict, Any to typing imports.
- Issues Flagged: None 

---

## Task Group 4: BatchSolverKernel Cache Integration
**Status**: [x]
**Dependencies**: Groups 1, 2, 3

**Required Context**:
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 1-50, imports)
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 125-210, __init__ method)
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 351-567, run method)
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 816-893, update method)
- File: src/cubie/batchsolving/BatchSolverKernel.py (lines 969-973, _invalidate_cache method)
- File: src/cubie/cubie_cache.py (lines 1-60, imports and ALL_CACHE_PARAMETERS)
- File: src/cubie/odesystems/baseODE.py (lines 42-135, BaseODE class)
- File: src/cubie/odesystems/symbolic/symbolicODE.py (lines 140-210, SymbolicODE.__init__)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- cache: Union[bool, str, Path] - passed to CubieCacheHandler
- cache_settings: Optional[Dict] - merged with cache kwargs (may be None)
- system: Must have `name` attribute (str or None) and optionally `fn_hash` attribute

**Tasks**:
1. **Import ALL_CACHE_PARAMETERS in BatchSolverKernel**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - Update import at line 24-25:
       ```python
       from cubie.cubie_cache import (
           CacheConfig,
           create_cache,
           invalidate_cache,
           CubieCacheHandler,
           ALL_CACHE_PARAMETERS,
       )
       ```
   - Integration: Needed for kwargs filtering

2. **Extract system_name and system_hash from ODE system**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - Add helper code in __init__ before cache_handler instantiation (around line 203):
       ```python
       # Extract system identification for cache
       system_name = getattr(system, 'name', None) or ""
       # For SymbolicODE, use fn_hash; otherwise use compile_settings hash
       if hasattr(system, 'fn_hash'):
           system_hash = system.fn_hash
       else:
           system_hash = system.config_hash if hasattr(system, 'config_hash') else ""
       
       # If system_name is empty, use first 12 chars of hash
       if not system_name and system_hash:
           system_name = system_hash[:12]
       ```
   - Edge cases:
     - System without name: use hash prefix
     - System without fn_hash (custom BaseODE): use config_hash
   - Integration: Passed to CubieCacheHandler

3. **Fix cache_handler instantiation in __init__**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - Current code at line 205:
       ```python
       self.cache_handler = CubieCacheHandler(cache, **cache_settings)
       ```
     - Problem: cache_settings may be None. Fix:
       ```python
       # Build cache settings dict from cache_settings and filtered kwargs
       if cache_settings is None:
           cache_settings = {}
       
       self.cache_handler = CubieCacheHandler(
           cache_arg=cache,
           system_name=system_name,
           system_hash=system_hash,
           **cache_settings
       )
       ```
   - Edge cases: cache_settings=None must not cause error
   - Integration: CubieCacheHandler now receives proper system info

4. **Update BatchSolverKernel.update to forward cache parameters**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Modify
   - Details:
     - In update() method, add cache parameter handling (around line 865):
       ```python
       # Forward cache-related updates to cache_handler
       cache_updates = {
           k: v for k, v in updates_dict.items() 
           if k in ALL_CACHE_PARAMETERS
       }
       if cache_updates:
           all_unrecognized -= self.cache_handler.update(
               cache_updates, silent=True
           )
       ```
   - Edge cases: No cache params in updates_dict is a no-op
   - Integration: Allows solver.update(cache_mode='flush_on_change')

5. **Verify run() method cache configuration is correct**
   - File: src/cubie/batchsolving/BatchSolverKernel.py
   - Action: Verify/Modify
   - Details:
     - Current code at lines 524-527:
       ```python
       config_hash = self.config_hash()
       self.kernel._cache = self.cache_handler.configured_cache(
           config_hash
       )
       ```
     - This is correct. Verify it handles None return from configured_cache.
       The kernel._cache assignment to None is acceptable (caching disabled).
   - Edge cases: None cache should not cause kernel launch error
   - Integration: Numba dispatcher handles None _cache gracefully

**Tests to Create**:
- Test file: tests/test_cubie_cache.py
- Test function: test_batch_solver_kernel_extracts_system_hash_from_symbolic_ode
- Description: Verify BatchSolverKernel extracts fn_hash from SymbolicODE
- Test function: test_batch_solver_kernel_uses_name_from_system
- Description: Verify BatchSolverKernel uses system.name for cache directory
- Test function: test_batch_solver_kernel_handles_none_cache_settings
- Description: Verify BatchSolverKernel works when cache_settings=None
- Test function: test_batch_solver_kernel_update_forwards_cache_params
- Description: Verify update(cache_mode='flush_on_change') is recognized

**Tests to Run**:
- tests/test_cubie_cache.py::test_batch_solver_kernel_extracts_system_hash_from_symbolic_ode
- tests/test_cubie_cache.py::test_batch_solver_kernel_uses_name_from_system
- tests/test_cubie_cache.py::test_batch_solver_kernel_handles_none_cache_settings
- tests/test_cubie_cache.py::test_batch_solver_kernel_update_forwards_cache_params

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/BatchSolverKernel.py (35 lines changed)
  * tests/test_cubie_cache.py (75 lines added)
- Functions/Methods Added/Modified:
  * BatchSolverKernel.__init__(): Updated import, changed cache_settings type from Optional[CacheConfig] to Optional[Dict[str, Any]], added system_name/system_hash extraction logic, fixed CubieCacheHandler instantiation with proper arguments
  * BatchSolverKernel.update(): Added cache parameter forwarding to cache_handler
- Implementation Summary:
  Added ALL_CACHE_PARAMETERS to imports. Changed cache_settings parameter type from Optional[CacheConfig] to Optional[Dict[str, Any]]. Added logic to extract system_name from system.name (with fallback to first 12 chars of hash) and system_hash from fn_hash (for SymbolicODE) or config_hash (for other BaseODE subclasses). Fixed cache_handler instantiation to handle None cache_settings and pass system_name/system_hash. Updated update() method to filter and forward cache-related parameters to cache_handler. Verified run() method correctly handles None return from configured_cache. Created 4 tests for BatchSolverKernel cache integration.
- Issues Flagged: None 

---

## Task Group 5: Solver Cache Keyword Argument Support
**Status**: [x]
**Dependencies**: Groups 1, 4

**Required Context**:
- File: src/cubie/batchsolving/solver.py (lines 1-50, imports)
- File: src/cubie/batchsolving/solver.py (lines 205-305, Solver.__init__)
- File: src/cubie/cubie_cache.py (ALL_CACHE_PARAMETERS constant)
- File: src/cubie/_utils.py (lines 241-286, merge_kwargs_into_settings)
- File: .github/context/cubie_internal_structure.md (entire file)

**Input Validation Required**:
- cache: Union[bool, str, Path] - primary cache toggle
- cache_mode: str in ('hash', 'flush_on_change') if provided in kwargs
- max_cache_entries: int >= 0 if provided in kwargs
- cache_dir: Optional[Path] if provided in kwargs

**Tasks**:
1. **Import ALL_CACHE_PARAMETERS in Solver**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     - Add import after line 36:
       ```python
       from cubie.cubie_cache import ALL_CACHE_PARAMETERS
       ```
   - Integration: Needed for kwargs filtering

2. **Add cache settings merging in Solver.__init__**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     - Add cache_settings parameter handling (around line 275, after loop_settings):
       ```python
       # Merge cache settings from kwargs
       cache_settings, cache_recognized = merge_kwargs_into_settings(
           kwargs=kwargs,
           valid_keys=ALL_CACHE_PARAMETERS,
           user_settings={},  # No explicit cache_settings param yet
       )
       ```
     - Update recognized_kwargs union (around line 280):
       ```python
       recognized_kwargs = (
           step_recognized
           | algorithm_recognized
           | output_recognized
           | memory_recognized
           | loop_recognized
           | cache_recognized
       )
       ```
   - Edge cases: No cache kwargs is fine - empty dict passed
   - Integration: Allows solve_ivp(cache_mode='hash')

3. **Pass cache_settings to BatchSolverKernel**
   - File: src/cubie/batchsolving/solver.py
   - Action: Modify
   - Details:
     - Update BatchSolverKernel instantiation (around line 288):
       ```python
       self.kernel = BatchSolverKernel(
           system,
           loop_settings=loop_settings,
           profileCUDA=profileCUDA,
           step_control_settings=step_settings,
           algorithm_settings=algorithm_settings,
           output_settings=output_settings,
           memory_settings=memory_settings,
           cache=cache,
           cache_settings=cache_settings,
       )
       ```
   - Edge cases: Empty cache_settings dict is valid
   - Integration: BatchSolverKernel merges into CubieCacheHandler

**Tests to Create**:
- Test file: tests/batchsolving/test_solver.py
- Test function: test_solver_accepts_cache_mode_kwarg
- Description: Verify Solver(system, cache_mode='flush_on_change') is recognized
- Test function: test_solver_accepts_max_cache_entries_kwarg
- Description: Verify Solver(system, max_cache_entries=5) is recognized
- Test function: test_solve_ivp_passes_cache_kwargs
- Description: Verify solve_ivp(system, y0, params, cache_mode='hash') works

**Tests to Run**:
- tests/batchsolving/test_solver.py::test_solver_accepts_cache_mode_kwarg
- tests/batchsolving/test_solver.py::test_solver_accepts_max_cache_entries_kwarg
- tests/batchsolving/test_solver.py::test_solve_ivp_passes_cache_kwargs

**Outcomes**:
- Files Modified:
  * src/cubie/batchsolving/solver.py (12 lines changed)
  * tests/batchsolving/test_solver.py (51 lines added)
- Functions/Methods Added/Modified:
  * Added import of ALL_CACHE_PARAMETERS from cubie.cubie_cache
  * Solver.__init__(): Added cache settings merging using merge_kwargs_into_settings, updated recognized_kwargs union to include cache_recognized, added cache_settings parameter to BatchSolverKernel instantiation
- Implementation Summary:
  Implemented cache keyword argument support in Solver class. Added import of ALL_CACHE_PARAMETERS from cubie.cubie_cache. Added cache settings merging after loop_settings merging to extract cache-related kwargs (cache_enabled, cache_mode, max_cache_entries, cache_dir). Updated recognized_kwargs to include cache_recognized so that strict=True mode properly recognizes cache parameters. Passed cache_settings dict to BatchSolverKernel instantiation. Created 3 tests: test_solver_accepts_cache_mode_kwarg verifies Solver accepts cache_mode kwarg with strict=True, test_solver_accepts_max_cache_entries_kwarg verifies max_cache_entries is recognized, test_solve_ivp_passes_cache_kwargs verifies solve_ivp convenience function properly forwards cache kwargs.
- Issues Flagged: None

---

## Task Group 6: Update Existing Tests for New Cache Path Structure
**Status**: [x]
**Dependencies**: Groups 1, 2, 3, 4, 5

**Required Context**:
- File: tests/test_cubie_cache.py (entire file)
- File: src/cubie/cubie_cache.py (entire file, for reference)
- File: .github/context/cubie_internal_structure.md (Testing Infrastructure section)

**Input Validation Required**:
- None - this is test updates only

**Tasks**:
1. **Update test_cache_locator_get_cache_path to expect hierarchical structure**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     - Current test at lines 53-62:
       ```python
       def test_cache_locator_get_cache_path():
           """Verify cache path is in generated/<system_name>/cache/."""
           locator = CUBIECacheLocator(
               system_name="test_system",
               system_hash="abc123",
               compile_settings_hash="def456",
           )
           path = locator.get_cache_path()
           assert "test_system" in path
           assert "cache" in path
       ```
     - Updated test:
       ```python
       def test_cache_locator_get_cache_path():
           """Verify cache path includes system_hash subdirectory."""
           locator = CUBIECacheLocator(
               system_name="test_system",
               system_hash="abc123",
               compile_settings_hash="def456",
           )
           path = locator.get_cache_path()
           assert "test_system" in path
           assert "abc123" in path  # system_hash in path
           assert path.endswith("CUDA_cache")
       ```
   - Edge cases: None
   - Integration: Tests new hierarchical path structure

2. **Update test_cubie_cache_path to expect hierarchical structure**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     - Current test at lines 171-179:
       ```python
       def test_cubie_cache_path():
           """Verify cache_path property returns expected path."""
           cache = CUBIECache(...)
           assert "test_system" in str(cache.cache_path)
           assert "cache" in str(cache.cache_path)
       ```
     - Updated test:
       ```python
       def test_cubie_cache_path():
           """Verify cache_path includes system_hash subdirectory."""
           cache = CUBIECache(
               system_name="test_system",
               system_hash="abc123",
               config_hash="def456789012345678901234567890123456"
               "789012345678901234567890abcd",
           )
           path_str = str(cache.cache_path)
           assert "test_system" in path_str
           assert "abc123" in path_str  # system_hash in path
           assert "CUDA_cache" in path_str
       ```
   - Edge cases: None
   - Integration: Tests CUBIECache uses new path structure

3. **Fix any imports in test file**
   - File: tests/test_cubie_cache.py
   - Action: Modify
   - Details:
     - Ensure imports include any new classes/constants needed for new tests:
       ```python
       from cubie.cubie_cache import (
           CUBIECacheLocator,
           CUBIECacheImpl,
           CUBIECache,
           CubieCacheHandler,
           CacheConfig,
           ALL_CACHE_PARAMETERS,
       )
       ```
   - Edge cases: None
   - Integration: Tests can use new components

**Tests to Create**:
- None in this group - only updates to existing tests

**Tests to Run**:
- tests/test_cubie_cache.py::test_cache_locator_get_cache_path
- tests/test_cubie_cache.py::test_cubie_cache_path
- tests/test_cubie_cache.py (entire file to verify no regressions)

**Outcomes**:
- Files Modified:
  * tests/test_cubie_cache.py (6 lines changed)
- Functions/Methods Added/Modified:
  * test_cache_locator_get_cache_path(): Updated assertions to verify system_hash in path and path ends with "CUDA_cache"
  * test_cubie_cache_path(): Updated assertions to verify system_hash in path and "CUDA_cache" in path
- Implementation Summary:
  Updated two existing tests to verify the new hierarchical cache path structure. test_cache_locator_get_cache_path now verifies that the path includes the system_hash ("abc123") and ends with "CUDA_cache". test_cubie_cache_path now verifies that the path includes both the system_name and system_hash, plus contains "CUDA_cache". Imports were already complete with CubieCacheHandler, CacheConfig, and ALL_CACHE_PARAMETERS already imported.
- Issues Flagged: None 

---

## Task Group 7: Integration Tests for Complete Cache Flow
**Status**: [x]
**Dependencies**: Groups 1, 2, 3, 4, 5, 6

**Required Context**:
- File: tests/test_cubie_cache.py (entire file)
- File: tests/conftest.py (fixture patterns)
- File: tests/system_fixtures.py (ODE system fixtures)
- File: src/cubie/batchsolving/solver.py (Solver class)
- File: src/cubie/cubie_cache.py (entire file)
- File: .github/context/cubie_internal_structure.md (Testing Infrastructure section)

**Input Validation Required**:
- None - this is integration test creation

**Tasks**:
1. **Create integration test for cache path with SymbolicODE**
   - File: tests/test_cubie_cache.py
   - Action: Modify (add new test)
   - Details:
     ```python
     def test_cache_handler_uses_symbolic_ode_fn_hash(three_state_linear):
         """Verify CubieCacheHandler uses fn_hash from SymbolicODE."""
         system = three_state_linear
         handler = CubieCacheHandler(
             cache_arg=True,
             system_name=system.name,
             system_hash=system.fn_hash,
         )
         
         assert handler.config.system_hash == system.fn_hash
         assert handler.config.system_name == system.name
         
         # Verify cache path includes the hash
         if handler.cache is not None:
             path_str = str(handler.cache.cache_path)
             assert system.fn_hash in path_str or "default" in path_str
     ```
   - Edge cases: System without fn_hash should work with empty hash
   - Integration: Tests complete flow from SymbolicODE to cache

2. **Create integration test for solver with cache configuration**
   - File: tests/test_cubie_cache.py
   - Action: Modify (add new test)
   - Details:
     ```python
     def test_solver_cache_configuration_flow(three_state_linear):
         """Verify Solver correctly configures cache handler."""
         from cubie import Solver
         
         solver = Solver(
             three_state_linear,
             algorithm="euler",
             cache=True,
             cache_mode="hash",
             max_cache_entries=5,
         )
         
         # Verify cache handler is configured
         assert solver.kernel.cache_handler is not None
         assert solver.kernel.cache_handler.config.cache_mode == "hash"
         assert solver.kernel.cache_handler.config.max_cache_entries == 5
     ```
   - Edge cases: None
   - Integration: Tests Solver -> BatchSolverKernel -> CubieCacheHandler chain

3. **Create integration test for cache update flow**
   - File: tests/test_cubie_cache.py
   - Action: Modify (add new test)
   - Details:
     ```python
     def test_solver_kernel_update_cache_mode(solverkernel):
         """Verify BatchSolverKernel.update forwards cache parameters."""
         # Initial state
         initial_mode = solverkernel.cache_handler.config.cache_mode
         
         # Update cache mode
         recognized = solverkernel.update(cache_mode="flush_on_change")
         
         assert "cache_mode" in recognized
         assert solverkernel.cache_handler.config.cache_mode == "flush_on_change"
     ```
   - Edge cases: None
   - Integration: Tests update flow through to cache handler

**Tests to Create**:
- Test file: tests/test_cubie_cache.py
- Test function: test_cache_handler_uses_symbolic_ode_fn_hash
- Description: Verify CubieCacheHandler extracts and uses fn_hash from SymbolicODE
- Test function: test_solver_cache_configuration_flow
- Description: Verify complete Solver -> cache handler configuration flow
- Test function: test_solver_kernel_update_cache_mode
- Description: Verify BatchSolverKernel.update forwards cache parameters

**Tests to Run**:
- tests/test_cubie_cache.py::test_cache_handler_uses_symbolic_ode_fn_hash
- tests/test_cubie_cache.py::test_solver_cache_configuration_flow
- tests/test_cubie_cache.py::test_solver_kernel_update_cache_mode
- tests/test_cubie_cache.py (entire file for full validation)

**Outcomes**:
- Files Modified:
  * tests/test_cubie_cache.py (48 lines added)
- Functions/Methods Added/Modified:
  * test_cache_handler_uses_symbolic_ode_fn_hash(): New integration test verifying CubieCacheHandler uses fn_hash from SymbolicODE
  * test_solver_cache_configuration_flow(): New integration test verifying Solver -> BatchSolverKernel -> CubieCacheHandler configuration chain
  * test_solver_kernel_update_cache_mode(): New integration test verifying BatchSolverKernel.update forwards cache parameters
- Implementation Summary:
  Added three new integration tests to verify the complete cache flow. test_cache_handler_uses_symbolic_ode_fn_hash uses the existing `system` fixture (which provides a SymbolicODE) to verify that CubieCacheHandler correctly extracts and uses the fn_hash for cache configuration. test_solver_cache_configuration_flow creates a Solver with cache kwargs and verifies they flow through to the cache handler. test_solver_kernel_update_cache_mode uses solverkernel_mutable to verify that update(cache_mode=...) properly forwards to the cache handler.
- Issues Flagged: None

---

# Summary

## Total Task Groups: 7

## Dependency Chain:
```
Group 1 (CacheConfig) 
    ↓
Group 2 (CUBIECacheLocator paths) ──┐
    ↓                               │
Group 3 (CubieCacheHandler) ←───────┘
    ↓
Group 4 (BatchSolverKernel integration)
    ↓
Group 5 (Solver kwarg support)
    ↓
Group 6 (Update existing tests)
    ↓
Group 7 (Integration tests)
```

## Tests to Create:
- Group 1: 3 new tests
- Group 2: 3 new tests
- Group 3: 7 new tests
- Group 4: 4 new tests
- Group 5: 3 new tests (in different test file)
- Group 6: 0 new tests (updates only)
- Group 7: 3 new integration tests

**Total: 23 new tests**

## Estimated Complexity:
- Groups 1-2: Low complexity (field renaming, path changes)
- Group 3: Medium complexity (handler refactoring with None handling)
- Groups 4-5: Medium complexity (integration wiring)
- Groups 6-7: Low complexity (test updates/additions)

## Files Modified:
1. `src/cubie/cubie_cache.py` - Major changes
2. `src/cubie/batchsolving/BatchSolverKernel.py` - Moderate changes
3. `src/cubie/batchsolving/solver.py` - Minor changes
4. `tests/test_cubie_cache.py` - Test additions/updates
5. `tests/batchsolving/test_solver.py` - Test additions
