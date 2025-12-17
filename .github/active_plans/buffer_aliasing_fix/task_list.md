# Implementation Task List
# Feature: Buffer Aliasing Fix
# Plan Reference: .github/active_plans/buffer_aliasing_fix/agent_plan.md

## Overview

This task list addresses incorrect shared memory aliasing in `generic_dirk.py`. After thorough analysis of all algorithm files:

- **DIRK**: Has aliasing bugs that need fixing
- **ERK**: Aliasing logic is correct (verified)
- **FIRK**: No internal aliasing (verified)
- **Rosenbrock**: No internal aliasing (verified)

The core issue is that DIRK's device function unconditionally aliases `stage_base` from `stage_accumulator` when `multistage=True`, ignoring whether `stage_accumulator` is actually in shared memory. Similarly, `increment_cache` and `rhs_cache` unconditionally alias from `solver_scratch`, which always exists in shared memory from the parent's perspective, so this is actually correct behavior.

After analysis, the primary bug is in `stage_base` aliasing logic.

---

## Task Group 1: Fix DIRKBufferSettings Memory Accounting - SEQUENTIAL
**Status**: [x]
**Dependencies**: None

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 114-316)

**Input Validation Required**:
- No new validation needed; existing validators are sufficient

**Tasks**:

### 1.1 Update stage_base_aliases_accumulator Property
- File: src/cubie/integrators/algorithms/generic_dirk.py
- Action: Modify
- Lines: 206-211
- Details:
  ```python
  @property
  def stage_base_aliases_accumulator(self) -> bool:
      """Return True if stage_base can alias first slice of accumulator.

      Only valid when multistage, BOTH stage_base AND accumulator are 
      in shared memory.
      """
      return (self.multistage 
              and self.use_shared_accumulator 
              and self.use_shared_stage_base)
  ```
- Edge cases: Single-stage methods, mixed shared/local configurations
- Integration: Used by shared_memory_elements and shared_indices properties

### 1.2 Update shared_memory_elements Property
- File: src/cubie/integrators/algorithms/generic_dirk.py
- Action: Modify
- Lines: 214-232
- Details:
  The current logic incorrectly handles the case where accumulator is local but stage_base is shared. Fix:
  ```python
  @property
  def shared_memory_elements(self) -> int:
      """Return total shared memory elements required.

      Includes accumulator, solver_scratch, and stage_increment if shared.
      stage_base is counted separately when it cannot alias accumulator.
      """
      total = 0
      if self.use_shared_accumulator:
          total += self.accumulator_length
      total += self.solver_scratch_elements  # Always included
      if self.use_shared_stage_increment:
          total += self.n
      # stage_base needs separate allocation when:
      # - shared AND (single-stage OR accumulator is local)
      if self.use_shared_stage_base and not self.stage_base_aliases_accumulator:
          total += self.n
      return total
  ```
- Edge cases: 
  - `multistage=True, accumulator=local, stage_base=shared` → separate allocation
  - `multistage=True, accumulator=shared, stage_base=shared` → aliases
  - `multistage=False, stage_base=shared` → separate allocation
- Integration: Used by DIRKStep.shared_memory_required

### 1.3 Update local_memory_elements Property
- File: src/cubie/integrators/algorithms/generic_dirk.py
- Action: Modify  
- Lines: 234-249
- Details:
  The current logic is mostly correct but should be clarified:
  ```python
  @property
  def local_memory_elements(self) -> int:
      """Return total local memory elements required.

      Includes buffers configured with location='local'.
      solver_scratch is not included as it is always shared from parent.
      """
      total = 0
      if not self.use_shared_accumulator:
          total += self.accumulator_length
      if not self.use_shared_stage_increment:
          total += self.n
      # stage_base needs local storage when local
      if not self.use_shared_stage_base:
          total += self.n
      return total
  ```
- Edge cases: All local configuration
- Integration: Used by DIRKStep.local_scratch_required

### 1.4 Update local_sizes Property for stage_base
- File: src/cubie/integrators/algorithms/generic_dirk.py
- Action: Modify
- Lines: 252-270
- Details:
  The current logic sets `stage_base_size=0` when multistage, which is incorrect if stage_base is local. Fix:
  ```python
  @property
  def local_sizes(self) -> DIRKLocalSizes:
      """Return DIRKLocalSizes instance with buffer sizes.

      The returned object provides nonzero sizes suitable for
      cuda.local.array allocation.
      """
      # stage_base needs local allocation when not aliasing accumulator
      # and configured for local memory
      if self.use_shared_stage_base:
          stage_base_size = 0  # Will use shared memory
      elif self.multistage and not self.use_shared_accumulator:
          # Can alias local accumulator in device function
          stage_base_size = 0
      else:
          stage_base_size = self.n
      return DIRKLocalSizes(
          stage_increment=self.n,
          stage_base=stage_base_size,
          accumulator=self.accumulator_length,
          solver_scratch=self.solver_scratch_elements,
          increment_cache=0,  # Always 0 since solver_scratch shared
          rhs_cache=0,        # Always 0 since solver_scratch shared
      )
  ```
- Edge cases: Local accumulator with local stage_base (can alias locally)
- Integration: Used in device function for cuda.local.array sizing

### 1.5 Update shared_indices Property for stage_base
- File: src/cubie/integrators/algorithms/generic_dirk.py
- Action: Modify
- Lines: 273-315
- Details:
  Fix the stage_base slice allocation to handle the case where accumulator is local but stage_base is shared:
  ```python
  @property
  def shared_indices(self) -> DIRKSliceIndices:
      """Return DIRKSliceIndices instance with shared memory layout.

      The returned object contains slices for each buffer's region
      in shared memory. Local buffers receive empty slices.
      """
      ptr = 0

      if self.use_shared_accumulator:
          accumulator_slice = slice(ptr, ptr + self.accumulator_length)
          ptr += self.accumulator_length
      else:
          accumulator_slice = slice(0, 0)

      # solver_scratch always included in shared memory layout
      solver_scratch_slice = slice(ptr, ptr + self.solver_scratch_elements)
      ptr += self.solver_scratch_elements

      if self.use_shared_stage_increment:
          stage_increment_slice = slice(ptr, ptr + self.n)
          ptr += self.n
      else:
          stage_increment_slice = slice(0, 0)

      # stage_base: alias accumulator, separate shared, or local
      if self.stage_base_aliases_accumulator:
          # Alias first n elements of accumulator
          stage_base_slice = slice(
              accumulator_slice.start,
              accumulator_slice.start + self.n
          )
      elif self.use_shared_stage_base:
          # Separate shared allocation
          stage_base_slice = slice(ptr, ptr + self.n)
          ptr += self.n
      else:
          # Local allocation
          stage_base_slice = slice(0, 0)

      return DIRKSliceIndices(
          stage_increment=stage_increment_slice,
          stage_base=stage_base_slice,
          accumulator=accumulator_slice,
          solver_scratch=solver_scratch_slice,
          local_end=ptr,
      )
  ```
- Edge cases: All combinations of parent/child locations
- Integration: Used in device function for shared memory slicing

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_dirk.py (25 lines changed)
- Functions/Methods Modified:
  * stage_base_aliases_accumulator property - now checks both accumulator AND stage_base locations
  * shared_memory_elements property - counts stage_base separately when not aliasing
  * local_memory_elements property - always counts stage_base when local
  * local_sizes property - returns 0 for stage_base when aliasing local accumulator
  * shared_indices property - handles separate shared allocation for stage_base
- Implementation Summary:
  Fixed all five BufferSettings properties to correctly handle mixed location configurations.
- Issues Flagged: None

---

## Task Group 2: Fix DIRKStep Device Function Allocation - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 1

**Required Context**:
- File: src/cubie/integrators/algorithms/generic_dirk.py (lines 616-800)

**Input Validation Required**:
- None; compile-time constants control branching

**Tasks**:

### 2.1 Add stage_base Shared Slice to Compile-Time Constants
- File: src/cubie/integrators/algorithms/generic_dirk.py
- Action: Modify
- Lines: 677-687
- Details:
  Add `stage_base_slice` extraction from `shared_indices`:
  ```python
  # Unpack slice indices for shared memory layout
  shared_indices = buffer_settings.shared_indices
  stage_increment_slice = shared_indices.stage_increment
  stage_base_slice = shared_indices.stage_base  # ADD THIS LINE
  accumulator_slice = shared_indices.accumulator
  solver_scratch_slice = shared_indices.solver_scratch
  ```
- Edge cases: None
- Integration: Used in device function branching

### 2.2 Add Aliasing Flags as Compile-Time Constants
- File: src/cubie/integrators/algorithms/generic_dirk.py
- Action: Modify
- Lines: 670-675
- Details:
  Add `stage_base_aliases_accumulator` flag:
  ```python
  # Unpack boolean flags as compile-time constants
  stage_increment_shared = buffer_settings.use_shared_stage_increment
  stage_base_shared = buffer_settings.use_shared_stage_base
  accumulator_shared = buffer_settings.use_shared_accumulator
  stage_base_aliases = buffer_settings.stage_base_aliases_accumulator  # ADD
  ```
- Edge cases: None
- Integration: Used in device function branching

### 2.3 Fix stage_base Allocation Logic in Device Function
- File: src/cubie/integrators/algorithms/generic_dirk.py
- Action: Modify
- Lines: 790-799
- Details:
  Replace the current incorrect logic:
  ```python
  # Current (INCORRECT):
  if multistage:
      stage_base = stage_accumulator[:n]
  else:
      if stage_base_shared:
          stage_base = shared[:n]
      else:
          stage_base = cuda.local.array(stage_base_local_size, precision)
          for _i in range(stage_base_local_size):
              stage_base[_i] = numba_precision(0.0)
  ```
  
  With correct logic:
  ```python
  # Correct: Check aliasing eligibility based on BOTH parent and child locations
  if stage_base_aliases:
      # Both accumulator and stage_base are shared; alias first slice
      stage_base = stage_accumulator[:n]
  elif multistage and not accumulator_shared and not stage_base_shared:
      # Both local; can alias local accumulator
      stage_base = stage_accumulator[:n]
  elif stage_base_shared:
      # Separate shared allocation (accumulator local or single-stage)
      stage_base = shared[stage_base_slice]
  else:
      # Separate local allocation
      stage_base = cuda.local.array(stage_base_local_size, precision)
      for _i in range(stage_base_local_size):
          stage_base[_i] = numba_precision(0.0)
  ```
- Edge cases:
  - `accumulator=shared, stage_base=shared, multistage=True` → alias shared
  - `accumulator=local, stage_base=shared, multistage=True` → separate shared
  - `accumulator=shared, stage_base=local, multistage=True` → separate local
  - `accumulator=local, stage_base=local, multistage=True` → alias local
  - `multistage=False, stage_base=shared` → separate shared
  - `multistage=False, stage_base=local` → separate local
- Integration: Core device function allocation

**Outcomes**: 
- Files Modified:
  * src/cubie/integrators/algorithms/generic_dirk.py (18 lines changed)
- Functions/Methods Modified:
  * build_step - added stage_base_aliases and stage_base_slice compile-time constants
  * step device function - fixed stage_base allocation logic with correct branching
- Implementation Summary:
  Added stage_base_aliases and stage_base_slice as compile-time constants.
  Replaced incorrect multistage-only aliasing with proper 4-way branching.
- Issues Flagged: None

---

## Task Group 3: Add Tests for DIRK Buffer Aliasing - PARALLEL
**Status**: [x]
**Dependencies**: Task Groups 1, 2

**Required Context**:
- File: tests/integrators/algorithms/test_buffer_settings.py (lines 143-252)

**Input Validation Required**:
- None; test file

**Tasks**:

### 3.1 Add Test for stage_base Aliasing with Mixed Locations
- File: tests/integrators/algorithms/test_buffer_settings.py
- Action: Modify (add tests to TestDIRKBufferSettings class)
- Details:
  Add new test methods:
  ```python
  def test_stage_base_aliases_requires_both_shared(self):
      """stage_base should only alias when BOTH accumulator and stage_base shared."""
      # Case 1: Both shared - should alias
      settings_both_shared = DIRKBufferSettings(
          n=3,
          stage_count=4,
          accumulator_location='shared',
          stage_base_location='shared',
      )
      assert settings_both_shared.stage_base_aliases_accumulator is True

      # Case 2: accumulator local, stage_base shared - should NOT alias
      settings_acc_local = DIRKBufferSettings(
          n=3,
          stage_count=4,
          accumulator_location='local',
          stage_base_location='shared',
      )
      assert settings_acc_local.stage_base_aliases_accumulator is False

      # Case 3: accumulator shared, stage_base local - should NOT alias
      settings_base_local = DIRKBufferSettings(
          n=3,
          stage_count=4,
          accumulator_location='shared',
          stage_base_location='local',
      )
      assert settings_base_local.stage_base_aliases_accumulator is False

  def test_shared_memory_counts_stage_base_when_not_aliased(self):
      """Shared memory should include stage_base when it cannot alias."""
      linear_settings = LinearSolverBufferSettings(n=3)
      newton_settings = NewtonBufferSettings(
          n=3,
          linear_solver_buffer_settings=linear_settings,
      )
      # accumulator local, stage_base shared -> needs separate shared allocation
      settings = DIRKBufferSettings(
          n=3,
          stage_count=4,
          accumulator_location='local',
          stage_base_location='shared',
          newton_buffer_settings=newton_settings,
      )
      # Should include: solver_scratch + stage_base (n=3)
      expected = newton_settings.shared_memory_elements + 3
      assert settings.shared_memory_elements == expected

  def test_shared_indices_stage_base_separate_allocation(self):
      """shared_indices should allocate stage_base separately when needed."""
      linear_settings = LinearSolverBufferSettings(n=3)
      newton_settings = NewtonBufferSettings(
          n=3,
          linear_solver_buffer_settings=linear_settings,
      )
      settings = DIRKBufferSettings(
          n=3,
          stage_count=4,
          accumulator_location='local',
          stage_base_location='shared',
          newton_buffer_settings=newton_settings,
      )
      indices = settings.shared_indices
      solver_size = newton_settings.shared_memory_elements
      
      # stage_base should have its own slice after solver_scratch
      assert indices.stage_base == slice(solver_size, solver_size + 3)
      assert indices.accumulator == slice(0, 0)  # local

  def test_local_sizes_stage_base_with_local_accumulator(self):
      """local_sizes.stage_base should be 0 when aliasing local accumulator."""
      settings = DIRKBufferSettings(
          n=3,
          stage_count=4,
          accumulator_location='local',
          stage_base_location='local',
      )
      sizes = settings.local_sizes
      # Can alias local accumulator in device function
      assert sizes.stage_base == 0
  ```
- Edge cases: All combinations covered
- Integration: Extends existing test class

**Outcomes**: 
- Files Modified:
  * tests/integrators/algorithms/test_buffer_settings.py (80 lines added)
- Functions/Methods Added:
  * test_stage_base_aliases_requires_both_shared - tests aliasing condition
  * test_shared_memory_counts_stage_base_when_not_aliased - tests memory accounting
  * test_shared_indices_stage_base_separate_allocation - tests shared_indices layout
  * test_local_sizes_stage_base_with_local_accumulator - tests local_sizes
- Implementation Summary:
  Added 4 comprehensive test methods covering all mixed location configurations.
- Issues Flagged: None

---

## Task Group 4: Update Instrumented Device Function - SEQUENTIAL
**Status**: [x]
**Dependencies**: Task Group 2

**Required Context**:
- File: tests/integrators/algorithms/instrumented/generic_dirk.py (lines 247-419)

**Input Validation Required**:
- None

**Tasks**:

### 4.1 Add stage_base Aliasing Flag and Slice to Instrumented DIRK
- File: tests/integrators/algorithms/instrumented/generic_dirk.py
- Action: Modify
- Lines: 250-268
- Details:
  Add `stage_base_aliases` flag and `stage_base_slice` extraction. After line 253:
  ```python
  # Unpack boolean flags as compile-time constants
  stage_increment_shared = buffer_settings.use_shared_stage_increment
  stage_base_shared = buffer_settings.use_shared_stage_base
  accumulator_shared = buffer_settings.use_shared_accumulator
  solver_scratch_shared = buffer_settings.use_shared_solver_scratch
  stage_base_aliases = buffer_settings.stage_base_aliases_accumulator  # ADD

  # Unpack slice indices for shared memory layout
  shared_indices = buffer_settings.shared_indices
  stage_increment_slice = shared_indices.stage_increment
  stage_base_slice = shared_indices.stage_base  # ADD THIS LINE
  accumulator_slice = shared_indices.accumulator
  solver_scratch_slice = shared_indices.solver_scratch
  ```
- Edge cases: None
- Integration: Used in device function branching

### 4.2 Fix stage_base Allocation Logic in Instrumented Device Function
- File: tests/integrators/algorithms/instrumented/generic_dirk.py
- Action: Modify
- Lines: 410-419
- Details:
  Replace the current incorrect logic:
  ```python
  # Current (INCORRECT):
  # Alias stage base onto first stage accumulator or allocate locally
  if multistage:
      stage_base = stage_accumulator[:n]
  else:
      if stage_base_shared:
          stage_base = shared[:n]
      else:
          stage_base = cuda.local.array(stage_base_local_size, precision)
          for _i in range(stage_base_local_size):
              stage_base[_i] = numba_precision(0.0)
  ```
  
  With correct logic matching the source file fix:
  ```python
  # Correct: Check aliasing eligibility based on BOTH parent and child locations
  if stage_base_aliases:
      # Both accumulator and stage_base are shared; alias first slice
      stage_base = stage_accumulator[:n]
  elif multistage and not accumulator_shared and not stage_base_shared:
      # Both local; can alias local accumulator
      stage_base = stage_accumulator[:n]
  elif stage_base_shared:
      # Separate shared allocation (accumulator local or single-stage)
      stage_base = shared[stage_base_slice]
  else:
      # Separate local allocation
      stage_base = cuda.local.array(stage_base_local_size, precision)
      for _i in range(stage_base_local_size):
          stage_base[_i] = numba_precision(0.0)
  ```
- Edge cases: Same as Task 2.3
- Integration: Mirrors fix in source file

**Outcomes**: 
- Files Modified:
  * tests/integrators/algorithms/instrumented/generic_dirk.py (18 lines changed)
- Functions/Methods Modified:
  * build_step - added stage_base_aliases and stage_base_slice compile-time constants
  * step device function - fixed stage_base allocation logic matching source file
- Implementation Summary:
  Mirrored all fixes from the source file to the instrumented version.
- Issues Flagged: None

---

## Summary

| Group | Description | Execution | Est. Complexity |
|-------|-------------|-----------|-----------------|
| 1 | Fix DIRKBufferSettings memory accounting | SEQUENTIAL | Medium |
| 2 | Fix DIRKStep device function allocation | SEQUENTIAL | Medium |
| 3 | Add tests for DIRK buffer aliasing | PARALLEL | Low |
| 4 | Update instrumented device function | SEQUENTIAL | Low |

### Dependency Chain
```
Group 1 (BufferSettings) 
    ↓
Group 2 (Device Function)
    ↓
Groups 3, 4 (Tests, Instrumented) [PARALLEL]
```

### Parallel Execution Opportunities
- Groups 3 and 4 can execute in parallel after Groups 1 and 2 complete

### Key Observations
1. **ERK, FIRK, Rosenbrock verified correct** - No changes needed
2. **DIRK is the only file with aliasing bugs** - Focused fix
3. **increment_cache/rhs_cache aliasing is actually correct** - solver_scratch is always provided from shared memory by the parent, so slicing from it is valid
4. **stage_base aliasing is the primary bug** - Must check both parent (accumulator) and child (stage_base) locations

---

# Implementation Complete - Ready for Review

## Execution Summary
- Total Task Groups: 4
- Completed: 4
- Failed: 0
- Total Files Modified: 4

## Task Group Completion
- Group 1: [x] Fix DIRKBufferSettings Memory Accounting - Complete
- Group 2: [x] Fix DIRKStep Device Function Allocation - Complete
- Group 3: [x] Add Tests for DIRK Buffer Aliasing - Complete
- Group 4: [x] Update Instrumented Device Function - Complete

## All Modified Files
1. src/cubie/integrators/algorithms/generic_dirk.py (43 lines changed)
2. tests/integrators/algorithms/test_buffer_settings.py (80 lines added)
3. tests/integrators/algorithms/instrumented/generic_dirk.py (18 lines changed)

## Flagged Issues
None identified during implementation.

## Handoff to Reviewer
All implementation tasks complete. Task list updated with outcomes.
Ready for reviewer agent to validate against user stories and goals.

---

*This document is for use by the taskmaster agent.*
