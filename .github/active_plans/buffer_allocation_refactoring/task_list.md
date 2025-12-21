# Implementation Task List
# Feature: Buffer Allocation Refactoring - Instrumented Algorithm Files
# Plan Reference: .github/active_plans/buffer_allocation_refactoring/agent_plan.md

## Summary

After detailed comparison of the main algorithm files and their instrumented
counterparts, all 4 files identified for update already contain the required
changes. The buffer location parameters and conditional buffer_kwargs creation
patterns are already synchronized with the main implementation.

## Verification Results

### 1. generic_erk.py (Instrumented)
**Status**: [x] Already synchronized

**Main file pattern** (`src/cubie/integrators/algorithms/generic_erk.py`):
- Lines 351-352: `stage_rhs_location` and `stage_accumulator_location` params
- Lines 424-432: Conditional buffer_kwargs creation

**Instrumented file** (`tests/integrators/algorithms/instrumented/generic_erk.py`):
- Lines 81-82: Has `stage_rhs_location` and `stage_accumulator_location` params
- Lines 93-102: Has conditional buffer_kwargs creation with same pattern:
  ```python
  buffer_kwargs = {
      'n': n,
      'stage_count': tableau.stage_count,
  }
  if stage_rhs_location is not None:
      buffer_kwargs['stage_rhs_location'] = stage_rhs_location
  if stage_accumulator_location is not None:
      buffer_kwargs['stage_accumulator_location'] = stage_accumulator_location
  buffer_settings = ERKBufferSettings(**buffer_kwargs)
  ```

**Outcomes**: File already matches main implementation. No changes required.

---

### 2. generic_dirk.py (Instrumented)
**Status**: [x] Already synchronized

**Main file pattern** (`src/cubie/integrators/algorithms/generic_dirk.py`):
- Lines 456-458: `stage_increment_location`, `stage_base_location`,
  `accumulator_location` params
- Lines 531-542: Conditional buffer_kwargs creation

**Instrumented file** (`tests/integrators/algorithms/instrumented/generic_dirk.py`):
- Lines 90-92: Has `stage_increment_location`, `stage_base_location`,
  `accumulator_location` params
- Lines 97-108: Has conditional buffer_kwargs creation with same pattern:
  ```python
  buffer_kwargs = {
      'n': n,
      'stage_count': tableau.stage_count,
  }
  if stage_increment_location is not None:
      buffer_kwargs['stage_increment_location'] = stage_increment_location
  if stage_base_location is not None:
      buffer_kwargs['stage_base_location'] = stage_base_location
  if accumulator_location is not None:
      buffer_kwargs['accumulator_location'] = accumulator_location
  buffer_settings = DIRKBufferSettings(**buffer_kwargs)
  ```

**Outcomes**: File already matches main implementation. No changes required.

---

### 3. generic_firk.py (Instrumented)
**Status**: [x] Already synchronized

**Main file pattern** (`src/cubie/integrators/algorithms/generic_firk.py`):
- Lines 400-402: `stage_increment_location`, `stage_driver_stack_location`,
  `stage_state_location` params
- Lines 479-492: Conditional buffer_kwargs creation

**Instrumented file** (`tests/integrators/algorithms/instrumented/generic_firk.py`):
- Lines 103-105: Has `stage_increment_location`, `stage_driver_stack_location`,
  `stage_state_location` params
- Lines 110-122: Has conditional buffer_kwargs creation with same pattern:
  ```python
  buffer_kwargs = {
      'n': n,
      'stage_count': tableau.stage_count,
      'n_drivers': n_drivers,
  }
  if stage_increment_location is not None:
      buffer_kwargs['stage_increment_location'] = stage_increment_location
  if stage_driver_stack_location is not None:
      buffer_kwargs['stage_driver_stack_location'] = stage_driver_stack_location
  if stage_state_location is not None:
      buffer_kwargs['stage_state_location'] = stage_state_location
  buffer_settings = FIRKBufferSettings(**buffer_kwargs)
  ```

**Outcomes**: File already matches main implementation. No changes required.

---

### 4. generic_rosenbrock_w.py (Instrumented)
**Status**: [x] Already synchronized

**Main file pattern** (`src/cubie/integrators/algorithms/generic_rosenbrock_w.py`):
- Lines 386-388: `stage_rhs_location`, `stage_store_location`,
  `cached_auxiliaries_location` params
- Lines 451-463: Conditional buffer_kwargs creation

**Instrumented file** (`tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`):
- Lines 86-88: Has `stage_rhs_location`, `stage_store_location`,
  `cached_auxiliaries_location` params
- Lines 96-107: Has conditional buffer_kwargs creation with same pattern:
  ```python
  buffer_kwargs = {
      'n': n,
      'stage_count': tableau.stage_count,
      'cached_auxiliary_count': 0,
  }
  if stage_rhs_location is not None:
      buffer_kwargs['stage_rhs_location'] = stage_rhs_location
  if stage_store_location is not None:
      buffer_kwargs['stage_store_location'] = stage_store_location
  if cached_auxiliaries_location is not None:
      buffer_kwargs['cached_auxiliaries_location'] = cached_auxiliaries_location
  buffer_settings = RosenbrockBufferSettings(**buffer_kwargs)
  ```

**Outcomes**: File already matches main implementation. No changes required.

---

## Conclusion

All 4 instrumented algorithm files already have the optional buffer location
parameters in their `__init__` methods and correctly implement the conditional
buffer_kwargs creation pattern. The refactoring has already been applied to
these files.

**Task Groups**: 0 (no tasks required)
**Dependencies**: None
**Parallel Execution Opportunities**: N/A
**Estimated Complexity**: Minimal - verification only
