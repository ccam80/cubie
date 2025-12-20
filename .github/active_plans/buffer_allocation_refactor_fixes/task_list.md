# Implementation Task List
# Feature: Buffer Allocation Refactor Fixes
# Plan Reference: .github/active_plans/buffer_allocation_refactor_fixes/agent_plan.md

## Overview

This task list implements three interconnected fixes to the buffer allocation system:
1. Audit and fix buffer name/parameter name mismatches
2. Remove ALL_BUFFER_LOCATION_PARAMETERS special handling
3. Implement three-parameter allocator for cross-location aliasing

All tasks are designed for surgical, minimal changes following CuBIE conventions.

---

## Task Group 1+2 (COMBINED): Buffer Name Audit and Fixes - SEQUENTIAL
**Status**: [x]
**Dependencies**: None
**Priority**: CRITICAL

**Description**: Audit all buffer registrations to ensure buffer names match their corresponding location parameter names exactly. Fix any mismatches found.

**Required Context**:
- File: /home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop.py (lines 171-221, 232)
- File: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_dirk.py (lines 242-250)
- File: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_erk.py (entire file - similar pattern to DIRK)
- File: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_firk.py (entire file - similar pattern to DIRK)
- File: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_rosenbrock_w.py (entire file - similar pattern to DIRK)
- File: /home/runner/work/cubie/cubie/src/cubie/integrators/matrix_free_solvers/newton_krylov.py (lines 228-255)
- File: /home/runner/work/cubie/cubie/src/cubie/integrators/matrix_free_solvers/linear_solver.py (entire file - similar pattern to newton_krylov)
- File: /home/runner/work/cubie/cubie/src/cubie/integrators/SingleIntegratorRunCore.py (lines 32-44)
- File: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py (lines 617-690 - update() method)

**Input Validation Required**: None (this is an audit and correction task)

**Tasks**:

### TG1.1: Audit IVPLoop Buffer Names vs Parameters
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop.py
- **Action**: AUDIT (no changes yet)
- **Details**:
  1. Extract all buffer_registry.register() calls (lines 171-221)
  2. Map buffer names to their expected parameter names:
     - Buffer: 'state' → Parameter: 'state_location' ✓
     - Buffer: 'proposed_state' → Parameter: 'state_proposal_location' ❌ MISMATCH
     - Buffer: 'parameters' → Parameter: 'parameters_location' ✓
     - Buffer: 'drivers' → Parameter: 'drivers_location' ✓
     - Buffer: 'proposed_drivers' → Parameter: 'drivers_proposal_location' ❌ MISMATCH
     - Buffer: 'observables' → Parameter: 'observables_location' ✓
     - Buffer: 'proposed_observables' → Parameter: 'observables_proposal_location' ❌ MISMATCH
     - Buffer: 'error' → Parameter: 'error_location' ✓
     - Buffer: 'counters' → Parameter: 'counters_location' ✓
     - Buffer: 'state_summary' → Parameter: 'state_summary_location' ✓
     - Buffer: 'observable_summary' → Parameter: 'observable_summary_location' ✓
     - Buffer: 'dt' → No location parameter (always local) ✓
     - Buffer: 'accept_step' → No location parameter (always local) ✓
  3. Identify mismatches: proposed_state, proposed_drivers, proposed_observables
  4. Check ODELoopConfig usage at line 232 for consistency

**Edge cases**: None
**Integration**: This audit identifies which naming convention to standardize on

### TG1.2: Fix IVPLoop Parameter Names (Prefer "proposed_*")
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop.py
- **Action**: Modify
- **Details**:
  Change parameter names to match buffer names:
  ```python
  # Line 142: Change parameter name
  state_proposal_location: str = 'local',
  # TO:
  proposed_state_location: str = 'local',
  
  # Line 145: Change parameter name
  drivers_proposal_location: str = 'local',
  # TO:
  proposed_drivers_location: str = 'local',
  
  # Line 147: Change parameter name
  observables_proposal_location: str = 'local',
  # TO:
  proposed_observables_location: str = 'local',
  
  # Line 177: Update register call to use new parameter name
  buffer_registry.register(
      'proposed_state', self, n_states,
      state_proposal_location, precision=precision
  )
  # TO:
  buffer_registry.register(
      'proposed_state', self, n_states,
      proposed_state_location, precision=precision
  )
  
  # Line 189: Update register call
  buffer_registry.register(
      'proposed_drivers', self, n_drivers,
      drivers_proposal_location, precision=precision
  )
  # TO:
  buffer_registry.register(
      'proposed_drivers', self, n_drivers,
      proposed_drivers_location, precision=precision
  )
  
  # Line 197: Update register call
  buffer_registry.register(
      'proposed_observables', self, n_observables,
      observables_proposal_location, precision=precision
  )
  # TO:
  buffer_registry.register(
      'proposed_observables', self, n_observables,
      proposed_observables_location, precision=precision
  )
  
  # Line 232: Update ODELoopConfig field names
  proposed_state_location=state_proposal_location,
  # TO:
  proposed_state_location=proposed_state_location,
  
  # Line 235: Update ODELoopConfig field names
  proposed_drivers_location=drivers_proposal_location,
  # TO:
  proposed_drivers_location=proposed_drivers_location,
  
  # Line 237: Update ODELoopConfig field names
  proposed_observables_location=observables_proposal_location,
  # TO:
  proposed_observables_location=proposed_observables_location,
  ```
- **Edge cases**: 
  - Ensure docstring updates for renamed parameters (lines 75-94)
  - ODELoopConfig attrs class may need corresponding field renames
- **Integration**: These parameter names must also be updated in ALL_BUFFER_LOCATION_PARAMETERS

### TG1.3: Update ALL_BUFFER_LOCATION_PARAMETERS
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/SingleIntegratorRunCore.py
- **Action**: Modify
- **Details**:
  Update parameter name set to match corrected IVPLoop parameters:
  ```python
  # Lines 32-44: Update the set
  ALL_BUFFER_LOCATION_PARAMETERS = {
      "state_location",
      "state_proposal_location",  # OLD MISMATCH
      "parameters_location",
      "drivers_location",
      "drivers_proposal_location",  # OLD MISMATCH
      "observables_location",
      "observables_proposal_location",  # OLD MISMATCH
      "error_location",
      "counters_location",
      "state_summary_location",
      "observable_summary_location",
  }
  # TO:
  ALL_BUFFER_LOCATION_PARAMETERS = {
      "state_location",
      "proposed_state_location",  # FIXED
      "parameters_location",
      "drivers_location",
      "proposed_drivers_location",  # FIXED
      "observables_location",
      "proposed_observables_location",  # FIXED
      "error_location",
      "counters_location",
      "state_summary_location",
      "observable_summary_location",
  }
  ```
- **Edge cases**: None
- **Integration**: This constant will be removed in Task Group 3, but must be correct for intermediate testing

### TG1.4: Audit Newton-Krylov Buffer Names
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/matrix_free_solvers/newton_krylov.py
- **Action**: AUDIT (no changes)
- **Details**:
  1. Extract all buffer_registry.register() calls (lines 228-255)
  2. Map buffer names to expected parameter names:
     - Buffer: 'newton_delta' → Parameter: 'delta_location' ❌ POTENTIAL ISSUE
     - Buffer: 'newton_residual' → Parameter: 'residual_location' ❌ POTENTIAL ISSUE
     - Buffer: 'newton_residual_temp' → Parameter: 'residual_temp_location' ❌ POTENTIAL ISSUE
     - Buffer: 'newton_stage_base_bt' → Parameter: 'stage_base_bt_location' ❌ POTENTIAL ISSUE
  3. Determine: Should buffer names include "newton_" prefix or should parameters?
  4. Decision: Per agent_plan.md Edge Case 1, prefer buffer names WITHOUT factory prefix
     - Buffer names should be: 'delta', 'residual', 'residual_temp', 'stage_base_bt'
     - OR parameters should be: 'newton_delta_location', etc.
  5. Recommendation: Keep current parameter names (without newton_ prefix) and CHANGE buffer names

**Edge cases**: Factory-specific prefixes should be avoided for internal buffers
**Integration**: Decision affects whether to rename buffers or parameters

### TG1.5: Fix Newton-Krylov Buffer Names (Remove "newton_" Prefix)
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/matrix_free_solvers/newton_krylov.py
- **Action**: Modify
- **Details**:
  Change buffer names to match parameter names (remove "newton_" prefix):
  ```python
  # Line 228-234: Change buffer name
  buffer_registry.register(
      'newton_delta',
      self,
      n,
      delta_location,
      precision=precision
  )
  # TO:
  buffer_registry.register(
      'delta',
      self,
      n,
      delta_location,
      precision=precision
  )
  
  # Line 235-241: Change buffer name
  buffer_registry.register(
      'newton_residual',
      self,
      n,
      residual_location,
      precision=precision
  )
  # TO:
  buffer_registry.register(
      'residual',
      self,
      n,
      residual_location,
      precision=precision
  )
  
  # Line 242-248: Change buffer name
  buffer_registry.register(
      'newton_residual_temp',
      self,
      n,
      residual_temp_location,
      precision=precision
  )
  # TO:
  buffer_registry.register(
      'residual_temp',
      self,
      n,
      residual_temp_location,
      precision=precision
  )
  
  # Line 249-255: Change buffer name
  buffer_registry.register(
      'newton_stage_base_bt',
      self,
      n,
      stage_base_bt_location,
      precision=precision
  )
  # TO:
  buffer_registry.register(
      'stage_base_bt',
      self,
      n,
      stage_base_bt_location,
      precision=precision
  )
  
  # Line 297-301: Update get_allocator calls to use new buffer names
  alloc_delta = buffer_registry.get_allocator('newton_delta', self)
  alloc_residual = buffer_registry.get_allocator('newton_residual', self)
  alloc_residual_temp = buffer_registry.get_allocator('newton_residual_temp', self)
  # TO:
  alloc_delta = buffer_registry.get_allocator('delta', self)
  alloc_residual = buffer_registry.get_allocator('residual', self)
  alloc_residual_temp = buffer_registry.get_allocator('residual_temp', self)
  
  # Search entire file for any other references to 'newton_delta', 
  # 'newton_residual', 'newton_residual_temp', 'newton_stage_base_bt' 
  # and update to new names
  ```
- **Edge cases**: 
  - All references in build() method must be updated
  - Any allocator assignments must use new names
- **Integration**: Affects compiled CUDA device function variable names

### TG1.6: Audit LinearSolver Buffer Names
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/matrix_free_solvers/linear_solver.py
- **Action**: AUDIT (read entire file)
- **Details**:
  1. Search for all buffer_registry.register() calls
  2. Map buffer names to parameter names
  3. Identify any mismatches similar to Newton-Krylov pattern
  4. Document findings (may have similar "linear_" or "krylov_" prefixes)

**Edge cases**: May need similar prefix removal as Newton-Krylov
**Integration**: Findings determine if TG1.7 is needed

### TG1.7: Fix LinearSolver Buffer Names (If Needed)
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/matrix_free_solvers/linear_solver.py
- **Action**: Modify (conditional on TG1.6 findings)
- **Details**:
  1. Apply same pattern as TG1.5 if mismatches found
  2. Remove factory-specific prefixes from buffer names
  3. Ensure parameter names match buffer names exactly
  4. Update all get_allocator() calls to use new buffer names

**Edge cases**: File may already be correct
**Integration**: Maintains consistency with Newton-Krylov approach

### TG1.8: Audit Algorithm Buffer Names
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_dirk.py
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_erk.py
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_firk.py
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_rosenbrock_w.py
- **Action**: AUDIT (read files)
- **Details**:
  1. For each algorithm file, extract buffer_registry.register() calls
  2. Map buffer names to parameter names:
     - DIRK: 'stage_increment' → 'stage_increment_location', 
             'accumulator' → 'accumulator_location',
             'stage_base' → 'stage_base_location' (if present)
     - ERK: 'stage_rhs' → 'stage_rhs_location',
            'stage_accumulator' → 'stage_accumulator_location'
     - FIRK: 'stage_driver_stack' → 'stage_driver_stack_location',
             'stage_state' → 'stage_state_location'
     - Rosenbrock: 'stage_store' → 'stage_store_location',
                    'cached_auxiliaries' → 'cached_auxiliaries_location'
  3. Verify all names match (no "dirk_", "erk_", etc. prefixes)
  4. Document any mismatches

**Edge cases**: May have nested solver buffer registrations (via get_child_allocators)
**Integration**: Confirms algorithm buffer naming is already correct or needs fixes

### TG1.9: Fix Algorithm Buffer Names (If Needed)
- **File**: Various algorithm files (conditional on TG1.8 findings)
- **Action**: Modify (conditional)
- **Details**:
  1. Apply corrections identified in TG1.8
  2. Ensure consistency across all algorithm implementations
  3. Update get_allocator() calls if buffer names change
  4. Update ALL_ALGORITHM_STEP_PARAMETERS in base_algorithm_step.py if needed

**Edge cases**: Must maintain consistency across all four algorithm families
**Integration**: Ensures uniform naming convention across integrators

### TG1.10: Update ODELoopConfig Attrs Class
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop_config.py
- **Action**: Modify
- **Details**:
  1. Locate ODELoopConfig attrs class definition
  2. Rename fields to match IVPLoop parameter changes:
     ```python
     proposed_state_location: str = attrs.field(default='local')
     proposed_drivers_location: str = attrs.field(default='local')
     proposed_observables_location: str = attrs.field(default='local')
     ```
  3. Verify field names match IVPLoop.__init__ parameters exactly
  4. Update any references to old field names in same file

**Edge cases**: Config may be used in other locations
**Integration**: Config changes must match IVPLoop signature

**Outcomes**: 
- [x] All buffer names verified to match parameter names exactly
- [x] IVPLoop parameter names corrected (proposed_* convention)
- [x] Newton-Krylov buffer names corrected (no "newton_" prefix)
- [x] LinearSolver buffer names corrected (no "lin_" prefix)
- [x] Algorithm buffer names verified/corrected (ERK, Rosenbrock fixed; DIRK, FIRK already correct)
- [x] ALL_BUFFER_LOCATION_PARAMETERS updated with correct names
- [x] ODELoopConfig already had correct field names (proposed_*)

**Files Modified**:
- src/cubie/integrators/loops/ode_loop.py (24 lines changed)
  * Renamed parameters: state_proposal_location → proposed_state_location
  * Renamed parameters: drivers_proposal_location → proposed_drivers_location
  * Renamed parameters: observables_proposal_location → proposed_observables_location
  * Updated buffer registry calls to use new parameter names
  * Updated ODELoopConfig instantiation
  * Updated docstrings
- src/cubie/integrators/SingleIntegratorRunCore.py (3 lines changed)
  * Updated ALL_BUFFER_LOCATION_PARAMETERS constant with new naming
- src/cubie/integrators/matrix_free_solvers/newton_krylov.py (8 lines changed)
  * Removed newton_ prefix: newton_delta → delta
  * Removed newton_ prefix: newton_residual → residual
  * Removed newton_ prefix: newton_residual_temp → residual_temp
  * Removed newton_ prefix: newton_stage_base_bt → stage_base_bt
- src/cubie/integrators/matrix_free_solvers/linear_solver.py (4 lines changed)
  * Removed lin_ prefix: lin_preconditioned_vec → preconditioned_vec
  * Removed lin_ prefix: lin_temp → temp
- src/cubie/integrators/algorithms/generic_erk.py (6 lines changed)
  * Removed erk_ prefix: erk_stage_rhs → stage_rhs
  * Removed erk_ prefix: erk_stage_accumulator → stage_accumulator
  * Removed erk_ prefix: erk_stage_cache → stage_cache
- src/cubie/integrators/algorithms/generic_rosenbrock_w.py (8 lines changed)
  * Removed rosenbrock_ prefix: rosenbrock_stage_rhs → stage_rhs
  * Removed rosenbrock_ prefix: rosenbrock_stage_store → stage_store
  * Removed rosenbrock_ prefix: rosenbrock_cached_auxiliaries → cached_auxiliaries
  * Removed rosenbrock_ prefix: rosenbrock_stage_cache → stage_cache

**Algorithms Verified as Correct**:
- generic_dirk.py: Buffers already match parameters (stage_increment, accumulator, stage_base) ✓
- generic_firk.py: Buffers already match parameters (stage_increment, stage_driver_stack, stage_state) ✓
- backwards_euler.py: No algorithm-specific buffers registered ✓
- crank_nicolson.py: No algorithm-specific buffers registered ✓

**Implementation Summary**:
All buffer names now match their location parameter names exactly, following the pattern:
- Buffer name: `{name}`
- Parameter name: `{name}_location`
- No factory-specific prefixes (newton_, lin_, erk_, rosenbrock_)
- Proposal buffers use `proposed_*` pattern (not `*_proposal`)

**Issues Flagged**: None - all changes were straightforward renames

---

## Task Group 3: Remove ALL_BUFFER_LOCATION_PARAMETERS - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 1+2 (must complete buffer name fixes first)
**Priority**: HIGH

**Description**: Remove the ALL_BUFFER_LOCATION_PARAMETERS constant and associated filtering logic, allowing location parameters to flow naturally through the initialization chain like other factory parameters.

**Required Context**:
- File: /home/runner/work/cubie/cubie/src/cubie/integrators/SingleIntegratorRunCore.py (lines 32-44, 350-380)
- File: /home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop.py (lines 129-165, 222-250)

**Input Validation Required**: None (plumbing refactor only)

**Tasks**:

### TG3.1: Remove ALL_BUFFER_LOCATION_PARAMETERS Constant
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/SingleIntegratorRunCore.py
- **Action**: Delete
- **Details**:
  ```python
  # Lines 29-44: DELETE these lines entirely
  # Buffer location parameters that can be specified at Solver level.
  # These parameters control whether specific buffers are allocated in
  # shared or local memory within CUDA device functions.
  ALL_BUFFER_LOCATION_PARAMETERS = {
      "state_location",
      "proposed_state_location",  # (or whatever the correct name is after TG1+2)
      "parameters_location",
      "drivers_location",
      "proposed_drivers_location",  # (or whatever the correct name is after TG1+2)
      "observables_location",
      "proposed_observables_location",  # (or whatever the correct name is after TG1+2)
      "error_location",
      "counters_location",
      "state_summary_location",
      "observable_summary_location",
  }
  # (blank lines)
  # DELETE ALL OF THE ABOVE
  ```
- **Edge cases**: Check for any imports or references to this constant elsewhere
- **Integration**: Must verify no other files import or reference this constant

### TG3.2: Remove Filtering Logic in instantiate_loop()
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/SingleIntegratorRunCore.py
- **Action**: Modify
- **Details**:
  Replace filtering logic with natural kwarg flow:
  ```python
  # Lines 352-362: CURRENT CODE
  # Extract buffer location kwargs from loop_settings
  buffer_location_kwargs = {
      key: loop_settings[key]
      for key in ALL_BUFFER_LOCATION_PARAMETERS
      if key in loop_settings
  }
  
  loop_kwargs = dict(loop_settings)
  # Remove buffer location kwargs from loop_settings (we'll pass them separately)
  for key in ALL_BUFFER_LOCATION_PARAMETERS:
      loop_kwargs.pop(key, None)
  
  # TO: DELETE ABOVE, REPLACE WITH
  loop_kwargs = dict(loop_settings)
  
  # Lines 376: CURRENT CODE
  **buffer_location_kwargs,
  
  # TO: DELETE (already included in loop_kwargs naturally)
  ```
  
  After changes, the flow should be:
  ```python
  loop_kwargs = dict(loop_settings)
  
  # Build the loop with individual parameters (new API)
  loop_kwargs.update(
      precision=precision,
      n_states=n_states,
      compile_flags=compile_flags,
      n_parameters=n_parameters,
      n_drivers=n_drivers,
      n_observables=n_observables,
      n_error=self.n_error,
      n_counters=n_counters,
      state_summary_buffer_height=state_summaries_buffer_height,
      observable_summary_buffer_height=observable_summaries_buffer_height,
      # Location parameters naturally included from loop_settings
  )
  if "driver_function" not in loop_kwargs:
      loop_kwargs["driver_function"] = driver_function
  
  # ... rest of instantiation
  ```
- **Edge cases**: 
  - Ensure loop_settings contains ALL desired parameters (no silent drops)
  - Verify no parameter conflicts between loop_settings and explicit parameters
- **Integration**: Location parameters now flow identically to other factory parameters

### TG3.3: Update Docstring Comments
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/SingleIntegratorRunCore.py
- **Action**: Modify
- **Details**:
  1. Remove or update any docstring references to "special handling" of location parameters
  2. Update class docstring if it mentions ALL_BUFFER_LOCATION_PARAMETERS
  3. Ensure parameter descriptions are accurate for natural flow

**Edge cases**: None
**Integration**: Documentation matches new simpler architecture

### TG3.4: Verify IVPLoop Accepts Location Parameters
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop.py
- **Action**: VERIFY (no changes expected)
- **Details**:
  1. Confirm IVPLoop.__init__ signature includes all location parameters (lines 129-165)
  2. Verify parameters have default values (enables optional specification)
  3. Confirm parameters are used in buffer_registry.register() calls (lines 171-221)
  4. Test that location parameters can be omitted (defaults apply) or specified (override defaults)

**Edge cases**: None expected (IVPLoop already designed for this)
**Integration**: Validates that removal of filtering doesn't break instantiation

**Outcomes**:
- [ ] ALL_BUFFER_LOCATION_PARAMETERS constant deleted
- [ ] Filtering logic removed from instantiate_loop()
- [ ] Location parameters flow naturally in loop_kwargs
- [ ] Docstrings updated to reflect simpler architecture
- [ ] IVPLoop instantiation verified to work with natural flow

---

## Task Group 4: Three-Parameter Allocator Implementation - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Group 3 (location parameters must flow naturally first)
**Priority**: HIGH

**Description**: Enhance allocator architecture to support three allocation sources (shared_parent, persistent_parent, shared_fallback), enabling cross-location aliasing where parent and child have different memory locations.

**Required Context**:
- File: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py (lines 75-122, 267-326, 454-492)
- File: /home/runner/work/cubie/cubie/.github/context/cubie_internal_structure.md (CUDAFactory pattern section)

**Input Validation Required**: None (internal refactor)

**Tasks**:

### TG4.1: Modify CUDABuffer.build_allocator() Signature
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Details**:
  Add third parameter for shared fallback allocation:
  ```python
  # Lines 75-100: CURRENT SIGNATURE
  def build_allocator(
      self,
      shared_slice: Optional[slice],
      persistent_slice: Optional[slice],
      local_size: Optional[int],
  ) -> Callable:
  
  # TO: ADD shared_fallback_slice parameter
  def build_allocator(
      self,
      shared_slice: Optional[slice],
      persistent_slice: Optional[slice],
      shared_fallback_slice: Optional[slice],
      local_size: Optional[int],
  ) -> Callable:
      """Compile CUDA device function for buffer allocation.
  
      Generates an inlined device function that allocates this buffer
      from the appropriate memory region based on which parameters are
      provided.
  
      Parameters
      ----------
      shared_slice
          Slice into shared memory for aliasing shared parent, or None.
      persistent_slice
          Slice into persistent local memory for aliasing persistent parent, 
          or None.
      shared_fallback_slice
          Slice into shared memory for new shared allocation when parent
          cannot be aliased, or None.
      local_size
          Size for local array allocation, or None if not local.
  
      Returns
      -------
      Callable
          CUDA device function: 
          (shared_parent, persistent_parent, shared_fallback) -> array
      """
      # Compile-time constants captured in closure
      _use_shared = shared_slice is not None
      _use_persistent = persistent_slice is not None
      _use_shared_fallback = shared_fallback_slice is not None  # NEW
      _shared_slice = shared_slice if _use_shared else slice(0, 0)
      _persistent_slice = (
          persistent_slice if _use_persistent else slice(0, 0)
      )
      _shared_fallback_slice = (  # NEW
          shared_fallback_slice if _use_shared_fallback else slice(0, 0)
      )
      _local_size = local_size if local_size is not None else 1
      _precision = self.precision
  
      @cuda.jit(device=True, inline=True, **compile_kwargs)
      def allocate_buffer(shared_parent, persistent_parent, shared_fallback):
          """Allocate buffer from appropriate memory region."""
          if _use_shared:
              array = shared_parent[_shared_slice]
          elif _use_persistent:
              array = persistent_parent[_persistent_slice]
          elif _use_shared_fallback:  # NEW
              array = shared_fallback[_shared_fallback_slice]
          else:
              array = cuda.local.array(_local_size, _precision)
          return array
  
      return allocate_buffer
  ```
- **Edge cases**: 
  - Only one of (_use_shared, _use_persistent, _use_shared_fallback) should be True for aliasing
  - Local allocation is fallback when all three are None/False
- **Integration**: Allocator signature now has three parameters instead of two

### TG4.2: Enhance build_shared_layout() to Compute Fallback Slices
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Details**:
  Track both aliased and fallback shared allocations:
  ```python
  # Line 267: CURRENT SIGNATURE
  def build_shared_layout(self) -> Dict[str, slice]:
  
  # TO: Return tuple with aliased and fallback layouts
  def build_shared_layout(self) -> Tuple[Dict[str, slice], Dict[str, slice]]:
      """Compute slice indices for shared memory buffers.
  
      Implements cross-location aliasing:
      - If parent is shared and has sufficient remaining space, alias
        slices within parent (returned in first dict)
      - If parent is shared but too small, allocate new shared space
        (returned in second dict as fallback)
      - If parent is local, allocate new shared space
        (returned in second dict as fallback)
      - Multiple aliases consume parent space first-come-first-serve
  
      Returns
      -------
      Tuple[Dict[str, slice], Dict[str, slice]]
          (aliased_layout, fallback_layout) - two mappings of buffer 
          names to shared memory slices. A buffer appears in exactly 
          one of the two dicts if it's shared.
      """
      offset = 0
      layout = {}  # Aliased or primary shared allocations
      fallback_layout = {}  # Fallback shared allocations
      self._alias_consumption.clear()
  
      # Process non-aliased shared buffers first (these go in primary layout)
      for name, entry in self.entries.items():
          if entry.location != 'shared' or entry.aliases is not None:
              continue
          layout[name] = slice(offset, offset + entry.size)
          self._alias_consumption[name] = 0
          offset += entry.size
  
      # Track fallback offset separately
      fallback_offset = 0
  
      # Process aliased buffers
      for name, entry in self.entries.items():
          if entry.aliases is None:
              continue
  
          parent_entry = self.entries[entry.aliases]
  
          if parent_entry.is_shared:
              # Parent is shared - check if space available AND buffer is shared
              consumed = self._alias_consumption.get(entry.aliases, 0)
              available = parent_entry.size - consumed
  
              if entry.is_shared and entry.size <= available:
                  # Alias fits within parent - use primary layout
                  parent_slice = layout[entry.aliases]
                  start = parent_slice.start + consumed
                  layout[name] = slice(start, start + entry.size)
                  self._alias_consumption[entry.aliases] = (
                      consumed + entry.size
                  )
              elif entry.is_shared:
                  # Parent too small - allocate in fallback layout
                  fallback_layout[name] = slice(
                      fallback_offset, fallback_offset + entry.size
                  )
                  fallback_offset += entry.size
              # else: not shared, will be handled by persistent/local
          elif entry.is_shared:
              # Parent is local, but child is shared - use fallback layout
              fallback_layout[name] = slice(
                  fallback_offset, fallback_offset + entry.size
              )
              fallback_offset += entry.size
          # else: buffer not shared, skip
  
      return layout, fallback_layout
  ```
- **Edge cases**:
  - A shared buffer appears in EITHER layout OR fallback_layout, never both
  - Non-shared buffers don't appear in either dict
  - Fallback offset starts at 0 (independent from main offset)
- **Integration**: Method now returns tuple instead of single dict

### TG4.3: Update BufferGroup._shared_layout to Store Both Layouts
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Details**:
  Change _shared_layout to store tuple of layouts:
  ```python
  # Line 147-149: CURRENT DEFINITION
  _shared_layout: Optional[Dict[str, slice]] = attrs.field(
      default=None, init=False
  )
  
  # TO: Store tuple of (aliased_layout, fallback_layout)
  _shared_layout: Optional[Tuple[Dict[str, slice], Dict[str, slice]]] = attrs.field(
      default=None, init=False
  )
  
  # Add helper property to access fallback layout
  @property
  def shared_fallback_layout(self) -> Dict[str, slice]:
      """Return fallback shared memory layout."""
      if self._shared_layout is None:
          self._shared_layout = self.build_shared_layout()
      return self._shared_layout[1]
  
  # Add helper property to access primary layout
  @property
  def shared_primary_layout(self) -> Dict[str, slice]:
      """Return primary (aliased) shared memory layout."""
      if self._shared_layout is None:
          self._shared_layout = self.build_shared_layout()
      return self._shared_layout[0]
  ```
- **Edge cases**: 
  - All accesses to _shared_layout must be updated to use tuple indexing
  - Properties provide cleaner access
- **Integration**: Affects all methods that access _shared_layout

### TG4.4: Update shared_buffer_size() Method
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Details**:
  Calculate total shared memory from both layouts:
  ```python
  # Lines 437-448 (in BufferGroup): CURRENT CODE
  def shared_buffer_size(self) -> int:
      """Return total shared memory elements for this parent."""
      if self._shared_layout is None:
          self._shared_layout = self.build_shared_layout()
      if not self._shared_layout:
          return 0
      return max(s.stop for s in self._shared_layout.values())
  
  # TO: Sum both primary and fallback layouts
  def shared_buffer_size(self) -> int:
      """Return total shared memory elements for this parent.
      
      Includes both primary (aliased) and fallback shared allocations.
      """
      if self._shared_layout is None:
          self._shared_layout = self.build_shared_layout()
      
      primary_layout, fallback_layout = self._shared_layout
      
      primary_size = 0
      if primary_layout:
          primary_size = max(s.stop for s in primary_layout.values())
      
      fallback_size = 0
      if fallback_layout:
          fallback_size = max(s.stop for s in fallback_layout.values())
      
      return primary_size + fallback_size
  ```
- **Edge cases**: 
  - Handle empty layouts (no shared buffers)
  - Both layouts contribute to total
- **Integration**: Total shared memory calculation now accounts for fallback

### TG4.5: Update get_allocator() to Provide Three Slices
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Modify
- **Details**:
  Provide shared_fallback_slice to build_allocator:
  ```python
  # Lines 454-492 (in BufferGroup): CURRENT CODE
  def get_allocator(self, name: str) -> Callable:
      """Generate CUDA device function for buffer allocation."""
      if name not in self.entries:
          raise KeyError(f"Buffer '{name}' not registered for parent.")
  
      entry = self.entries[name]
  
      # Ensure layouts are computed
      if self._shared_layout is None:
          self._shared_layout = self.build_shared_layout()
      if self._persistent_layout is None:
          self._persistent_layout = self.build_persistent_layout()
      if self._local_sizes is None:
          self._local_sizes = self.build_local_sizes()
  
      # Determine allocation source
      shared_slice = self._shared_layout.get(name)
      persistent_slice = self._persistent_layout.get(name)
      local_size = self._local_sizes.get(name)
  
      return entry.build_allocator(
          shared_slice, persistent_slice, local_size
      )
  
  # TO: Extract slices from both shared layouts
  def get_allocator(self, name: str) -> Callable:
      """Generate CUDA device function for buffer allocation."""
      if name not in self.entries:
          raise KeyError(f"Buffer '{name}' not registered for parent.")
  
      entry = self.entries[name]
  
      # Ensure layouts are computed
      if self._shared_layout is None:
          self._shared_layout = self.build_shared_layout()
      if self._persistent_layout is None:
          self._persistent_layout = self.build_persistent_layout()
      if self._local_sizes is None:
          self._local_sizes = self.build_local_sizes()
  
      # Determine allocation source
      primary_layout, fallback_layout = self._shared_layout
      
      shared_slice = primary_layout.get(name)  # Aliasing parent
      shared_fallback_slice = fallback_layout.get(name)  # New shared allocation
      persistent_slice = self._persistent_layout.get(name)
      local_size = self._local_sizes.get(name)
  
      return entry.build_allocator(
          shared_slice, persistent_slice, shared_fallback_slice, local_size
      )
  ```
- **Edge cases**:
  - A buffer should have at most ONE of (shared_slice, shared_fallback_slice) non-None
  - Verify mutual exclusivity in implementation
- **Integration**: Allocator now receives correct slice for shared aliasing vs fallback

### TG4.6: Update IVPLoop.build() for Three-Parameter Allocators
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop.py
- **Action**: Modify
- **Details**:
  1. Locate IVPLoop.build() method (search for "def build")
  2. Find where allocators are called (they receive shared_parent, persistent_parent)
  3. Add shared_fallback array creation and passing:
     ```python
     # In build() method, locate allocator calls:
     # CURRENT PATTERN (example):
     state = state_allocator(shared_mem, persistent_mem)
     
     # TO: Add third parameter
     shared_fallback = cuda.shared.array(fallback_size, precision)
     state = state_allocator(shared_mem, persistent_mem, shared_fallback)
     
     # Where fallback_size is computed from buffer_registry
     fallback_size = buffer_registry.shared_fallback_buffer_size(self)
     ```
  4. Ensure all allocator invocations updated
  5. shared_fallback array is independent from shared_mem

**Edge cases**:
  - fallback_size may be 0 (no fallback needed)
  - Must allocate separate shared array for fallback
- **Integration**: All buffer allocations now pass three arrays

### TG4.7: Add shared_fallback_buffer_size() Method
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: Create
- **Details**:
  Add method to BufferGroup and BufferRegistry:
  ```python
  # In BufferGroup class, add after shared_buffer_size():
  def shared_fallback_buffer_size(self) -> int:
      """Return fallback shared memory elements for this parent.
      
      Returns size of shared_fallback array needed for cross-location
      aliasing scenarios.
      """
      if self._shared_layout is None:
          self._shared_layout = self.build_shared_layout()
      
      primary_layout, fallback_layout = self._shared_layout
      
      if not fallback_layout:
          return 0
      return max(s.stop for s in fallback_layout.values())
  
  # In BufferRegistry class, add public wrapper:
  def shared_fallback_buffer_size(self, parent: object) -> int:
      """Return fallback shared memory elements for a parent.
  
      Parameters
      ----------
      parent
          Parent instance to query.
  
      Returns
      -------
      int
          Fallback shared memory elements needed.
      """
      if parent not in self._groups:
          return 0
      return self._groups[parent].shared_fallback_buffer_size()
  ```
- **Edge cases**: Returns 0 when no fallback needed
- **Integration**: Used by IVPLoop and algorithms to size fallback array

### TG4.8: Update Algorithm build() Methods for Three-Parameter Allocators
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_dirk.py
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_erk.py
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_firk.py
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_rosenbrock_w.py
- **Action**: Modify (all files)
- **Details**:
  1. Locate build() method in each algorithm class
  2. Find allocator invocations (pattern: allocator(shared, persistent))
  3. Add shared_fallback parameter:
     ```python
     # CURRENT PATTERN:
     buffer = allocator(shared_mem, persistent_mem)
     
     # TO:
     buffer = allocator(shared_mem, persistent_mem, shared_fallback)
     ```
  4. Ensure shared_fallback is created/passed from parent or as parameter
  5. Repeat for ALL algorithm files

**Edge cases**:
  - Algorithms may receive shared_fallback from IVPLoop
  - Or may create their own if they have independent buffers
- **Integration**: All algorithm allocations updated for new signature

### TG4.9: Update Newton-Krylov and LinearSolver for Three-Parameter Allocators
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/matrix_free_solvers/newton_krylov.py
- **File**: /home/runner/work/cubie/cubie/src/cubie/integrators/matrix_free_solvers/linear_solver.py
- **Action**: Modify
- **Details**:
  1. Locate build() method
  2. Find allocator invocations
  3. Add shared_fallback parameter to all allocator calls
  4. Ensure fallback array is created and passed

**Edge cases**: May be nested inside algorithm build()
**Integration**: Solver allocations updated for new signature

**Outcomes**:
- [ ] CUDABuffer.build_allocator() accepts shared_fallback_slice
- [ ] Allocator device function has three-parameter signature
- [ ] build_shared_layout() returns (primary, fallback) tuple
- [ ] shared_buffer_size() accounts for both layouts
- [ ] get_allocator() provides correct slices
- [ ] shared_fallback_buffer_size() method added
- [ ] IVPLoop.build() updated for three-parameter allocators
- [ ] All algorithm build() methods updated
- [ ] All solver build() methods updated
- [ ] Cross-location aliasing fully functional

---

## Task Group 5: Integration Testing & Verification - SEQUENTIAL
**Status**: [ ]
**Dependencies**: Task Groups 1+2, 3, 4 (all implementation complete)
**Priority**: CRITICAL

**Description**: Verify all changes work correctly through comprehensive testing of buffer allocation, location propagation, and cross-location aliasing.

**Required Context**:
- File: /home/runner/work/cubie/cubie/tests/test_buffer_registry.py (entire file)
- File: /home/runner/work/cubie/cubie/tests (search for buffer location tests)
- File: /home/runner/work/cubie/cubie/src/cubie/batchsolving/solver.py (solver API entry points)

**Input Validation Required**: Test coverage must validate that user inputs flow correctly

**Tasks**:

### TG5.1: Run Existing Buffer Registry Tests
- **File**: None (command line)
- **Action**: TEST
- **Details**:
  ```powershell
  # From repository root
  pytest tests/test_buffer_registry.py -v
  ```
  Expected results:
  - All existing tests pass
  - No regressions in buffer registration
  - Aliasing logic works correctly
  - Cross-location aliasing scenarios tested

**Edge cases**: Tests may need updates if they verify old allocator signature
**Integration**: Baseline validation before new tests

### TG5.2: Run Existing Solver Tests
- **File**: None (command line)
- **Action**: TEST
- **Details**:
  ```powershell
  # Search for solver tests that use buffer locations
  pytest tests/batchsolving/test_solver.py -v -k location
  
  # If no matches, run full solver test suite
  pytest tests/batchsolving/test_solver.py -v
  ```
  Expected results:
  - Solver initialization with location parameters works
  - Buffer locations propagate through init chain
  - No errors from removed ALL_BUFFER_LOCATION_PARAMETERS

**Edge cases**: May reveal missing test coverage
**Integration**: Validates end-to-end user API

### TG5.3: Create Test for Buffer Name/Parameter Matching
- **File**: /home/runner/work/cubie/cubie/tests/test_buffer_registry.py
- **Action**: Create (add test)
- **Details**:
  Add test at end of file:
  ```python
  def test_buffer_parameter_name_matching():
      """Verify all buffer names match their location parameter names.
      
      This test ensures that buffer_registry.update() can correctly
      find buffers when location parameters are specified.
      """
      from cubie.integrators.loops.ode_loop import IVPLoop
      from cubie.integrators.algorithms.generic_dirk import DIRKStep
      from cubie.integrators.matrix_free_solvers.newton_krylov import NewtonKrylovSolver
      from cubie.outputhandling import OutputCompileFlags
      import numpy as np
      
      precision = np.float32
      n_states = 3
      compile_flags = OutputCompileFlags(save_state=True)
      
      # Test IVPLoop buffer names
      loop = IVPLoop(
          precision=precision,
          n_states=n_states,
          compile_flags=compile_flags,
          state_location='local',
          proposed_state_location='local',  # Must match buffer name
          parameters_location='local',
          # ... etc
      )
      
      # Verify update() recognizes all location parameters
      recognized = buffer_registry.update(
          loop,
          state_location='shared',
          proposed_state_location='shared',  # Must match buffer 'proposed_state'
          parameters_location='shared',
      )
      
      assert 'state_location' in recognized
      assert 'proposed_state_location' in recognized
      assert 'parameters_location' in recognized
      
      # Test Newton-Krylov buffer names
      newton = NewtonKrylovSolver(
          precision=precision,
          n=n_states,
          delta_location='local',  # Must match buffer 'delta'
          residual_location='local',  # Must match buffer 'residual'
      )
      
      recognized = buffer_registry.update(
          newton,
          delta_location='shared',
          residual_location='shared',
      )
      
      assert 'delta_location' in recognized
      assert 'residual_location' in recognized
  ```
- **Edge cases**: Test must use actual corrected names from TG1+2
- **Integration**: Validates TG1+2 fixes

### TG5.4: Create Test for Cross-Location Aliasing
- **File**: /home/runner/work/cubie/cubie/tests/test_buffer_registry.py
- **Action**: Create (add test)
- **Details**:
  Add comprehensive aliasing test:
  ```python
  def test_cross_location_aliasing():
      """Verify cross-location aliasing works with three-parameter allocators.
      
      Tests scenarios:
      - Parent shared + child shared (space available) → alias parent
      - Parent shared + child shared (no space) → fallback shared
      - Parent local + child shared → fallback shared
      - Parent persistent + child persistent → alias parent
      """
      import numpy as np
      from cubie.buffer_registry import buffer_registry
      
      class MockParent:
          pass
      
      parent = MockParent()
      precision = np.float32
      
      buffer_registry.clear_parent(parent)
      
      # Scenario 1: Parent shared, child shared, space available
      buffer_registry.register(
          'parent1', parent, 10, 'shared', precision=precision
      )
      buffer_registry.register(
          'child1', parent, 5, 'shared', aliases='parent1', precision=precision
      )
      
      group = buffer_registry._groups[parent]
      primary, fallback = group.build_shared_layout()
      
      assert 'parent1' in primary
      assert 'child1' in primary  # Should alias within parent
      assert 'child1' not in fallback
      
      parent_slice = primary['parent1']
      child_slice = primary['child1']
      assert child_slice.start >= parent_slice.start
      assert child_slice.stop <= parent_slice.stop
      
      # Scenario 2: Parent shared, child shared, no space
      buffer_registry.clear_parent(parent)
      buffer_registry.register(
          'parent2', parent, 3, 'shared', precision=precision
      )
      buffer_registry.register(
          'child2', parent, 5, 'shared', aliases='parent2', precision=precision
      )
      
      group = buffer_registry._groups[parent]
      primary, fallback = group.build_shared_layout()
      
      assert 'parent2' in primary
      assert 'child2' not in primary
      assert 'child2' in fallback  # Should use fallback
      
      # Scenario 3: Parent local, child shared
      buffer_registry.clear_parent(parent)
      buffer_registry.register(
          'parent3', parent, 10, 'local', precision=precision
      )
      buffer_registry.register(
          'child3', parent, 5, 'shared', aliases='parent3', precision=precision
      )
      
      group = buffer_registry._groups[parent]
      primary, fallback = group.build_shared_layout()
      
      assert 'parent3' not in primary
      assert 'parent3' not in fallback
      assert 'child3' not in primary
      assert 'child3' in fallback  # Parent local, child shared → fallback
      
      # Scenario 4: Verify allocator has three parameters
      allocator = buffer_registry.get_allocator('child3', parent)
      import inspect
      sig = inspect.signature(allocator.py_func)
      params = list(sig.parameters.keys())
      assert len(params) == 3
      assert 'shared_parent' in params
      assert 'persistent_parent' in params
      assert 'shared_fallback' in params
  ```
- **Edge cases**: 
  - Test multiple children aliasing same parent
  - Test all four scenarios from user stories
- **Integration**: Validates TG4 implementation

### TG5.5: Test Init Path Location Propagation
- **File**: /home/runner/work/cubie/cubie/tests/batchsolving (create new file or add to existing)
- **Action**: Create test
- **Details**:
  Test that location parameters flow from solve_ivp → Solver → SingleIntegratorRun → IVPLoop:
  ```python
  def test_location_parameter_init_propagation(three_state_linear):
      """Verify buffer location parameters propagate through init chain."""
      from cubie import solve_ivp
      from cubie.buffer_registry import buffer_registry
      import numpy as np
      
      system = three_state_linear
      t_span = (0.0, 1.0)
      
      # Call solve_ivp with location parameters
      result = solve_ivp(
          system,
          t_span,
          state_location='shared',
          proposed_state_location='shared',
          error_location='shared',
      )
      
      # Verify buffers were created with correct locations
      # (This requires access to the solver's IVPLoop instance)
      # May need to add debugging/inspection API
      
      assert result is not None
      # Further assertions depend on available inspection methods
  ```
- **Edge cases**: May need solver API enhancement for inspection
- **Integration**: Validates TG3 natural parameter flow

### TG5.6: Test Update Path Location Changes
- **File**: /home/runner/work/cubie/cubie/tests/batchsolving (create or add to existing)
- **Action**: Create test
- **Details**:
  Test that solver.update() correctly updates buffer locations:
  ```python
  def test_location_parameter_update_propagation(three_state_linear):
      """Verify buffer location updates trigger cache invalidation and rebuild."""
      from cubie import Solver
      from cubie.buffer_registry import buffer_registry
      import numpy as np
      
      system = three_state_linear
      t_span = (0.0, 1.0)
      
      # Create solver with initial locations
      solver = Solver(
          system,
          algorithm='explicit_euler',
          state_location='local',
      )
      
      # Solve once
      result1 = solver.solve(t_span)
      
      # Update location
      solver.update(state_location='shared')
      
      # Solve again (should rebuild with new location)
      result2 = solver.solve(t_span)
      
      # Both should succeed
      assert result1 is not None
      assert result2 is not None
      
      # Verify cache was invalidated and rebuilt
      # (Requires inspection API or internal testing)
  ```
- **Edge cases**: Verify cache invalidation actually occurs
- **Integration**: Validates update path works with natural flow

### TG5.7: Run Full Integration Test Suite
- **File**: None (command line)
- **Action**: TEST
- **Details**:
  ```powershell
  # Run all tests to check for regressions
  pytest tests/ -v --durations=10
  
  # Check for specific failures
  # If CUDA tests fail without GPU, that's expected
  # Focus on non-CUDA tests passing
  ```
  Expected results:
  - No regressions in existing functionality
  - New tests pass
  - Buffer allocation works across all scenarios

**Edge cases**: Some tests may fail due to CUDA unavailability (acceptable)
**Integration**: Final validation of all changes

### TG5.8: Manual Verification of Allocator Signature
- **File**: /home/runner/work/cubie/cubie/src/cubie/buffer_registry.py
- **Action**: INSPECT
- **Details**:
  1. Manually verify compiled allocator signatures:
     ```python
     # In Python REPL or test
     from cubie.buffer_registry import buffer_registry
     import numpy as np
     
     class TestParent:
         pass
     
     parent = TestParent()
     buffer_registry.register('test', parent, 10, 'shared', precision=np.float32)
     allocator = buffer_registry.get_allocator('test', parent)
     
     # Inspect signature
     import inspect
     sig = inspect.signature(allocator.py_func)
     print(sig)  # Should show: (shared_parent, persistent_parent, shared_fallback)
     ```
  2. Verify parameter count is 3
  3. Verify parameter names match expected

**Edge cases**: None
**Integration**: Confirms TG4 allocator signature change

**Outcomes**:
- [ ] All existing buffer_registry tests pass
- [ ] All existing solver tests pass
- [ ] Buffer name/parameter matching test passes
- [ ] Cross-location aliasing test passes
- [ ] Init path propagation test passes
- [ ] Update path propagation test passes
- [ ] Full test suite passes (no regressions)
- [ ] Allocator signature manually verified

---

## Summary

### Dependency Chain
```
TG1+2 (Audit & Fix Names)
    ↓
TG3 (Remove Filter)
    ↓
TG4 (Three-Parameter Allocator)
    ↓
TG5 (Testing & Verification)
```

### Total Task Groups: 5
### Total Individual Tasks: 37

### Complexity Estimates
- **TG1+2**: Moderate (10 tasks, mostly rename operations)
- **TG3**: Simple (4 tasks, deletion and flow simplification)
- **TG4**: Complex (9 tasks, architectural change with signature updates)
- **TG5**: Moderate (8 tasks, comprehensive testing)

### Parallel Execution Opportunities
- **TG1.4-1.9** (Algorithm/Solver audits) can be done in parallel
- **TG4.8-4.9** (Algorithm/Solver allocator updates) can be done in parallel
- **TG5.3-5.6** (Test creation) can be done in parallel

### Estimated Effort
- **TG1+2**: 2-3 hours (careful audit and systematic renaming)
- **TG3**: 30 minutes (straightforward removal)
- **TG4**: 3-4 hours (careful architectural change, many call sites)
- **TG5**: 2 hours (test creation and verification)

**Total**: 8-10 hours of focused implementation

### Files Expected to Change
1. `/home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop.py`
2. `/home/runner/work/cubie/cubie/src/cubie/integrators/loops/ode_loop_config.py`
3. `/home/runner/work/cubie/cubie/src/cubie/integrators/SingleIntegratorRunCore.py`
4. `/home/runner/work/cubie/cubie/src/cubie/integrators/matrix_free_solvers/newton_krylov.py`
5. `/home/runner/work/cubie/cubie/src/cubie/integrators/matrix_free_solvers/linear_solver.py`
6. `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_dirk.py`
7. `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_erk.py`
8. `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_firk.py`
9. `/home/runner/work/cubie/cubie/src/cubie/integrators/algorithms/generic_rosenbrock_w.py`
10. `/home/runner/work/cubie/cubie/src/cubie/buffer_registry.py`
11. `/home/runner/work/cubie/cubie/tests/test_buffer_registry.py`
12. Possibly: `/home/runner/work/cubie/cubie/tests/batchsolving/test_solver.py`

### Success Criteria (from agent_plan.md)
1. ✅ All buffer names match their parameter names exactly
2. ✅ NO ALL_BUFFER_LOCATION_PARAMETERS constant exists
3. ✅ Location parameters flow naturally through init chain
4. ✅ Location parameters update correctly through update chain
5. ✅ Cross-location aliasing works in all scenarios
6. ✅ All existing tests pass
7. ✅ New aliasing tests added and passing
8. ✅ User can specify buffer locations via solve_ivp, Solver.__init__, Solver.solve, Solver.update
9. ✅ Buffer location updates trigger cache invalidation and rebuild
