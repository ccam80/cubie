# Implementation Task List
# Feature: Buffer Settings Plumbing
# Plan Reference: /home/runner/work/cubie/cubie/docs/plans/buffer_settings_plumbing.md

## Overview

This task list implements the "Buffer Settings Plumbing" feature which treats buffer location parameters as first-class compile settings, identical to other parameters like `dt_save`. The key principles from user feedback:

1. Each CUDAFactory owns its buffers and their locations
2. Buffer location kwargs are joined to each factory's existing argument list
3. Use existing `split_applicable_settings` and `merge_kwargs_into_settings` utilities
4. Add `update()` method to BufferRegistry following `update_compile_settings` pattern
5. Buffer locations follow the same code path as other compile settings

---

## Task Group 1: BufferRegistry.update() Method - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None

**Required Context**:
- File: src/cubie/buffer_registry.py (lines 122-273) - BufferRegistry class
- File: src/cubie/CUDAFactory.py (lines 565-635) - update_compile_settings method pattern
- File: src/cubie/_utils.py (lines 200-245) - merge_kwargs_into_settings utility

**Input Validation Required**:
- `factory`: Must exist in registry's _contexts (if not, return empty set for silent, raise for non-silent)
- `updates_dict`: Optional dict, defaults to empty dict
- `silent`: bool, defaults to False
- `**kwargs`: Additional key-value pairs to merge with updates_dict
- Location value validation: handled by existing BufferEntry validator (must be 'shared' or 'local')

**Tasks**:

### 1.1 Add update() method to BufferRegistry
- File: src/cubie/buffer_registry.py
- Action: Add new method
- Location: After `update_buffer()` method (around line 273)
- Details:
  ```python
  def update(
      self,
      factory: object,
      updates_dict: Optional[Dict[str, object]] = None,
      silent: bool = False,
      **kwargs: object,
  ) -> Set[str]:
      """Update buffer locations via compile-settings-like interface.

      Finds CUDABuffer objects where a keyword matches the pattern
      `{buffer_name}_location` and updates the location property.
      Invalidates the context's cached layouts on any change.

      Parameters
      ----------
      factory
          Factory instance whose buffers should be updated.
      updates_dict
          Optional mapping of parameter names to new values.
      silent
          When True, suppress errors for unrecognized parameters.
      **kwargs
          Additional parameter updates.

      Returns
      -------
      Set[str]
          Names of parameters that were recognized and updated.

      Raises
      ------
      KeyError
          If `silent=False` and unrecognized parameters exist.
      ValueError
          If location value is invalid ('shared' or 'local' required).
      """
      # Implementation logic:
      # 1. Merge updates_dict and kwargs into single dict
      # 2. If factory not in _contexts, return empty set (or raise if not silent)
      # 3. Get context for factory
      # 4. Build buffer_name -> entry mapping for pattern matching
      # 5. For each key-value pair in merged dict:
      #    a. Check if key ends with '_location'
      #    b. Extract buffer_name = key[:-9] (strip '_location')
      #    c. If buffer_name in context.entries:
      #       - Call update_buffer(buffer_name, factory, location=value)
      #       - Add key to recognized set
      # 6. If unrecognized params exist and not silent, raise KeyError
      # 7. Return recognized set
  ```
- Edge cases:
  - Factory not in registry: return empty set silently, or raise if silent=False and params provided
  - No params provided: return empty set immediately
  - Key doesn't end in '_location': not recognized by this method
  - Buffer name from pattern not registered: not recognized
  - Invalid location value: BufferEntry validator will raise ValueError
- Integration: Called by IVPLoop.update() to delegate buffer location updates

### 1.2 Add Set import to buffer_registry.py
- File: src/cubie/buffer_registry.py
- Action: Modify import statement
- Location: Line 3
- Details:
  ```python
  # Change from:
  from typing import Callable, Dict, Optional
  # To:
  from typing import Callable, Dict, Optional, Set
  ```

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 2: IVPLoop.update() Integration - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 855-901) - existing update() method
- File: src/cubie/buffer_registry.py - BufferRegistry.update() (to be added in Task Group 1)
- File: src/cubie/_utils.py (lines 200-245) - merge_kwargs_into_settings

**Input Validation Required**:
- `updates_dict`: Optional dict (already validated in existing code)
- `silent`: bool (already validated in existing code)
- `**kwargs`: Additional updates (already validated in existing code)
- No new validation needed - delegate to BufferRegistry.update() and update_compile_settings()

**Tasks**:

### 2.1 Modify IVPLoop.update() to delegate buffer location updates
- File: src/cubie/integrators/loops/ode_loop.py
- Action: Modify existing method
- Location: Lines 855-901
- Details:
  ```python
  def update(
      self,
      updates_dict: Optional[dict[str, object]] = None,
      silent: bool = False,
      **kwargs: object,
  ) -> Set[str]:
      """Update compile settings through the CUDAFactory interface.

      Parameters
      ----------
      updates_dict
          Mapping of configuration names to replacement values.
      silent
          When True, suppress warnings about unrecognized parameters.
      **kwargs
          Additional configuration updates applied as keyword arguments.

      Returns
      -------
      set
          Set of parameter names that were recognized and updated.
      """
      if updates_dict is None:
          updates_dict = {}
      updates_dict = updates_dict.copy()
      if kwargs:
          updates_dict.update(kwargs)
      if updates_dict == {}:
          return set()

      # Flatten nested dict values before distributing
      updates_dict, unpacked_keys = unpack_dict_values(updates_dict)

      # Delegate buffer location updates to registry first
      buffer_recognized = buffer_registry.update(
          self, updates_dict, silent=True
      )

      # Delegate remaining to compile settings
      recognised = self.update_compile_settings(updates_dict, silent=True)

      # Combine recognized sets
      all_recognised = recognised | buffer_recognized | unpacked_keys

      # Check for unrecognized params
      unrecognised = set(updates_dict.keys()) - (recognised | buffer_recognized)
      if not silent and unrecognised:
          raise KeyError(
              f"Unrecognized parameters in update: {unrecognised}. "
              "These parameters were not updated.",
          )

      return all_recognised
  ```
- Edge cases:
  - Buffer location param recognized by registry but not compile_settings: should work
  - Compile setting recognized but not buffer location: should work
  - Param recognized by both: buffer_registry updates buffer, compile_settings updates setting
  - Empty updates: return empty set
- Integration: Enables unified update API for both buffer locations and compile settings

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 3: Define Buffer Location Parameter Names - SEQUENTIAL
**Status**: [ ]
**Dependencies**: None (can run in parallel with Task Group 1)

**Required Context**:
- File: src/cubie/integrators/loops/ode_loop.py (lines 39-46) - ALL_LOOP_SETTINGS
- File: src/cubie/integrators/loops/ode_loop.py (lines 130-167) - IVPLoop.__init__ buffer location params

**Input Validation Required**:
- None - this is a constant definition

**Tasks**:

### 3.1 Add buffer location parameter names to ALL_LOOP_SETTINGS
- File: src/cubie/integrators/loops/ode_loop.py
- Action: Modify constant
- Location: Lines 39-46
- Details:
  ```python
  # Recognised compile-critical loop configuration parameters. These keys mirror
  # the solver API so helper utilities can consistently merge keyword arguments
  # into loop-specific settings dictionaries.
  ALL_LOOP_SETTINGS = {
      "dt_save",
      "dt_summarise",
      "dt0",
      "dt_min",
      "dt_max",
      "is_adaptive",
      # Buffer location parameters for loop-assigned buffers
      "loop_state_location",
      "loop_proposed_state_location",
      "loop_parameters_location",
      "loop_drivers_location",
      "loop_proposed_drivers_location",
      "loop_observables_location",
      "loop_proposed_observables_location",
      "loop_error_location",
      "loop_counters_location",
      "loop_state_summary_location",
      "loop_observable_summary_location",
  }
  ```
- Edge cases: None
- Integration: Allows `split_applicable_settings` and `merge_kwargs_into_settings` to filter these params

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 4: Tests for BufferRegistry.update() - PARALLEL
**Status**: [ ]
**Dependencies**: Task Group 1

**Required Context**:
- File: tests/test_buffer_registry.py (entire file) - existing test patterns
- File: src/cubie/buffer_registry.py - BufferRegistry class

**Input Validation Required**:
- Test that valid location values are accepted
- Test that invalid location values raise ValueError
- Test that unrecognized params raise KeyError when silent=False
- Test that unrecognized params are silently ignored when silent=True
- Test that recognized set is correctly returned

**Tasks**:

### 4.1 Add TestBufferRegistryUpdate test class
- File: tests/test_buffer_registry.py
- Action: Add new test class
- Location: After TestBufferUpdate class (around line 333)
- Details:
  ```python
  class TestBufferRegistryUpdate:
      """Tests for BufferRegistry.update() method."""

      @pytest.fixture(autouse=True)
      def fresh_registry(self):
          """Create a fresh registry for each test."""
          self.registry = BufferRegistry()
          self.factory = MockFactory()
          yield

      def test_update_empty_returns_empty_set(self):
          """Empty updates return empty set."""
          self.registry.register('buf', self.factory, 100, 'shared')
          result = self.registry.update(self.factory)
          assert result == set()

      def test_update_unregistered_factory_returns_empty(self):
          """Unregistered factory returns empty set."""
          other_factory = MockFactory()
          result = self.registry.update(other_factory, {'buf_location': 'shared'})
          assert result == set()

      def test_update_location_recognized(self):
          """Buffer location update is recognized."""
          self.registry.register('state', self.factory, 100, 'local')
          result = self.registry.update(
              self.factory, {'state_location': 'shared'}
          )
          assert 'state_location' in result

      def test_update_location_changes_entry(self):
          """Buffer location update changes entry."""
          self.registry.register('state', self.factory, 100, 'local')
          self.registry.update(self.factory, {'state_location': 'shared'})
          entry = self.registry._contexts[self.factory].entries['state']
          assert entry.location == 'shared'

      def test_update_invalidates_layout(self):
          """Buffer location update invalidates cached layout."""
          self.registry.register('state', self.factory, 100, 'shared')
          _ = self.registry.shared_buffer_size(self.factory)
          context = self.registry._contexts[self.factory]
          assert context._shared_layout is not None

          self.registry.update(self.factory, {'state_location': 'local'})
          assert context._shared_layout is None

      def test_update_unrecognized_silent_true(self):
          """Unrecognized params ignored when silent=True."""
          self.registry.register('state', self.factory, 100, 'local')
          result = self.registry.update(
              self.factory,
              {'unknown_param': 42},
              silent=True
          )
          assert 'unknown_param' not in result

      def test_update_unrecognized_silent_false_raises(self):
          """Unrecognized params raise KeyError when silent=False."""
          self.registry.register('state', self.factory, 100, 'local')
          with pytest.raises(KeyError, match="unknown_param"):
              self.registry.update(
                  self.factory,
                  {'unknown_param': 42},
                  silent=False
              )

      def test_update_via_kwargs(self):
          """Update works with kwargs."""
          self.registry.register('state', self.factory, 100, 'local')
          result = self.registry.update(
              self.factory, state_location='shared'
          )
          assert 'state_location' in result

      def test_update_multiple_buffers(self):
          """Multiple buffer locations can be updated at once."""
          self.registry.register('state', self.factory, 100, 'local')
          self.registry.register('params', self.factory, 50, 'local')
          result = self.registry.update(
              self.factory,
              {
                  'state_location': 'shared',
                  'params_location': 'shared',
              }
          )
          assert 'state_location' in result
          assert 'params_location' in result
          context = self.registry._contexts[self.factory]
          assert context.entries['state'].location == 'shared'
          assert context.entries['params'].location == 'shared'

      def test_update_invalid_location_raises(self):
          """Invalid location value raises ValueError."""
          self.registry.register('state', self.factory, 100, 'local')
          with pytest.raises(ValueError):
              self.registry.update(
                  self.factory, {'state_location': 'invalid'}
              )

      def test_update_nonlocation_param_not_recognized(self):
          """Params not ending in _location are not recognized."""
          self.registry.register('state', self.factory, 100, 'local')
          result = self.registry.update(
              self.factory, {'state': 'shared'}, silent=True
          )
          assert result == set()

      def test_update_partial_match_not_recognized(self):
          """Buffer name must fully match for recognition."""
          self.registry.register('loop_state', self.factory, 100, 'local')
          # 'state_location' doesn't match 'loop_state'
          result = self.registry.update(
              self.factory, {'state_location': 'shared'}, silent=True
          )
          assert result == set()
          # But 'loop_state_location' does
          result = self.registry.update(
              self.factory, {'loop_state_location': 'shared'}
          )
          assert 'loop_state_location' in result
  ```
- Edge cases: All covered in test cases above
- Integration: Validates BufferRegistry.update() implementation

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Task Group 5: Tests for IVPLoop.update() Buffer Integration - PARALLEL
**Status**: [ ]
**Dependencies**: Task Groups 1, 2, 3

**Required Context**:
- File: tests/integrators/loops/ - existing test directory structure
- File: src/cubie/integrators/loops/ode_loop.py - IVPLoop class

**Input Validation Required**:
- Test that buffer location updates via IVPLoop.update() work correctly
- Test that recognized set includes buffer location params
- Test integration with compile_settings updates

**Tasks**:

### 5.1 Add test file for IVPLoop buffer integration
- File: tests/integrators/loops/test_ode_loop_buffer_update.py
- Action: Create new test file
- Details:
  ```python
  """Tests for IVPLoop buffer location update integration."""

  import pytest
  import numpy as np

  from cubie.buffer_registry import buffer_registry, BufferRegistry
  from cubie.integrators.loops.ode_loop import IVPLoop, ALL_LOOP_SETTINGS
  from cubie.outputhandling.output_config import OutputCompileFlags


  @pytest.fixture
  def minimal_compile_flags():
      """Create minimal OutputCompileFlags for testing."""
      return OutputCompileFlags(
          save_state=True,
          save_observables=False,
          summarise_state=False,
          summarise_observables=False,
          save_counters=False,
      )


  @pytest.fixture
  def ivp_loop(minimal_compile_flags):
      """Create minimal IVPLoop for testing."""
      # Clear any existing registrations for isolation
      loop = IVPLoop(
          precision=np.float64,
          n_states=3,
          compile_flags=minimal_compile_flags,
          n_parameters=2,
          n_drivers=0,
          n_observables=0,
          n_error=3,
          n_counters=4,
          state_location='local',
          parameters_location='local',
      )
      yield loop
      # Clean up
      buffer_registry.clear_factory(loop)


  class TestALLLoopSettingsIncludesBufferLocations:
      """Tests that ALL_LOOP_SETTINGS includes buffer location params."""

      def test_loop_state_location_in_settings(self):
          assert 'loop_state_location' in ALL_LOOP_SETTINGS

      def test_loop_parameters_location_in_settings(self):
          assert 'loop_parameters_location' in ALL_LOOP_SETTINGS

      def test_loop_proposed_state_location_in_settings(self):
          assert 'loop_proposed_state_location' in ALL_LOOP_SETTINGS


  class TestIVPLoopUpdateBufferLocation:
      """Tests for IVPLoop.update() with buffer location params."""

      def test_update_buffer_location_recognized(self, ivp_loop):
          """Buffer location update is recognized by IVPLoop.update()."""
          result = ivp_loop.update({'loop_state_location': 'shared'})
          assert 'loop_state_location' in result

      def test_update_buffer_location_changes_registry(self, ivp_loop):
          """Buffer location update changes registry entry."""
          ivp_loop.update({'loop_state_location': 'shared'})
          context = buffer_registry._contexts[ivp_loop]
          assert context.entries['loop_state'].location == 'shared'

      def test_update_compile_setting_still_works(self, ivp_loop):
          """Compile settings update still works."""
          result = ivp_loop.update({'dt_save': 0.05})
          assert 'dt_save' in result
          assert ivp_loop.dt_save == 0.05

      def test_update_mixed_params(self, ivp_loop):
          """Mixed buffer location and compile setting updates work."""
          result = ivp_loop.update({
              'loop_state_location': 'shared',
              'dt_save': 0.02,
          })
          assert 'loop_state_location' in result
          assert 'dt_save' in result

      def test_update_unrecognized_raises(self, ivp_loop):
          """Unrecognized params raise KeyError."""
          with pytest.raises(KeyError):
              ivp_loop.update({'completely_unknown': 42})

      def test_update_unrecognized_silent(self, ivp_loop):
          """Unrecognized params ignored when silent=True."""
          result = ivp_loop.update(
              {'completely_unknown': 42},
              silent=True
          )
          assert 'completely_unknown' not in result
  ```
- Edge cases: All covered in test cases above
- Integration: Validates IVPLoop.update() correctly delegates to BufferRegistry.update()

### 5.2 Ensure tests/integrators/loops/__init__.py exists
- File: tests/integrators/loops/__init__.py
- Action: Create if not exists
- Details: Empty file
- Edge cases: None
- Integration: Required for pytest to discover tests in subdirectory

**Outcomes**:
[Empty - to be filled by taskmaster agent]

---

## Summary

### Total Task Groups: 5

### Dependency Chain:
```
Task Group 1 (BufferRegistry.update()) ──┐
                                         ├──> Task Group 4 (Registry Tests)
Task Group 3 (ALL_LOOP_SETTINGS)    ──┐  │
                                     ├──┼──> Task Group 2 (IVPLoop.update())
                                     │  │                   │
                                     │  └───────────────────┼──> Task Group 5 (Loop Tests)
                                     └──────────────────────┘
```

### Parallel Execution Opportunities:
- Task Groups 1 and 3 can run in parallel (no dependencies between them)
- Task Groups 4 and 5 can run in parallel once their dependencies complete

### Estimated Complexity:
- Task Group 1: Medium (new method following established pattern)
- Task Group 2: Low (modification to existing method)
- Task Group 3: Low (constant definition)
- Task Group 4: Medium (comprehensive test suite)
- Task Group 5: Medium (integration tests with fixtures)

### Key Design Decisions Implemented:
1. **BufferRegistry.update()** follows exact pattern of `CUDAFactory.update_compile_settings()`
2. **Pattern matching**: `{buffer_name}_location` suffix identifies buffer location params
3. **No separate dicts**: Buffer locations joined to existing ALL_LOOP_SETTINGS
4. **Unified update path**: IVPLoop.update() delegates to both registry and compile_settings
5. **Cache invalidation**: Automatic via existing `update_buffer()` method
