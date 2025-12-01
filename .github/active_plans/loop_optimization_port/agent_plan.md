# Loop Optimization Port: Agent Plan

## Overview

This plan describes porting optimized loop indexing patterns from `tests/all_in_one.py` to the production algorithm modules. The key insight is that restructuring loop iteration order and coefficient access patterns enables the CUDA compiler to fully unroll loops and embed constants.

## Components to Modify

### 1. generic_erk.py - ERKStep.build_step()

**Current Location:** `src/cubie/integrators/algorithms/generic_erk.py`

**Source of Optimized Code:** `tests/all_in_one.py` lines 1127-1453 (`erk_step_inline_factory`)

**Action Required:** The step function inside `build_step()` should be a **direct copy** of the optimized ERK step from `all_in_one.py`, with the following modifications:

1. **Remove** all memory location selection logic (lines 1191-1198 and 1273-1297 in all_in_one.py):
   - Remove `stage_rhs_in_shared`, `stage_accumulator_in_shared`, `stage_cache_in_shared` flags
   - Remove `stage_cache_aliases_rhs`, `stage_cache_aliases_accumulator` logic
   - Keep the **current module's** memory allocation pattern:
     ```python
     stage_rhs = shared[:n]
     stage_accumulator = cuda.local.array(accumulator_length, dtype=precision)
     if multistage:
         stage_cache = stage_rhs  # FSAL cache alias
     ```

2. **Keep** the optimized loop structure:
   ```python
   for prev_idx in range(stages_except_first):
       stage_offset = prev_idx * n
       stage_idx = prev_idx + int32(1)
       matrix_col = stage_rhs_coeffs[prev_idx]

       for successor_idx in range(stages_except_first):
           coeff = matrix_col[successor_idx+int32(1)]
           row_offset = successor_idx * n
           for idx in range(n):
               increment = stage_rhs[idx]
               stage_accumulator[row_offset + idx] += coeff * increment
   ```

3. **Keep** the inline state conversion:
   ```python
   for idx in range(n):
       stage_accumulator[base] = (stage_accumulator[base] * dt_scalar + state[idx])
       base += int32(1)
   ```

4. **Keep** the driver function calls (current module has them, all_in_one.py uses placeholders)

**Expected Behavior:**
- Coefficient access via `stage_rhs_coeffs[prev_idx]` returns a column of the tableau matrix
- Inner loop accumulates contributions from current stage to all successor accumulators
- State conversion happens inline before evaluating next stage RHS

### 2. generic_dirk.py - DIRKStep.build_step()

**Current Location:** `src/cubie/integrators/algorithms/generic_dirk.py`

**Source of Optimized Code:** `tests/all_in_one.py` lines 711-1120 (`dirk_step_inline_factory`)

**Action Required:** Adapt the optimized loop pattern while keeping current memory allocation:

1. **Keep** current memory allocation (shared memory for accumulator and solver scratch)

2. **Adopt** the optimized accumulator filling pattern:
   ```python
   for stage_idx in range(int32(1), stage_count):
       prev_idx = stage_idx - int32(1)
       successor_range = stage_count - stage_idx

       # Fill accumulators with previous step's contributions
       for successor_offset in range(successor_range):
           successor_idx = stage_idx + successor_offset
           base = (successor_idx - int32(1)) * n
           for idx in range(n):
               state_coeff = stage_rhs_coeffs[successor_idx][prev_idx]
               contribution = state_coeff * stage_rhs[idx] * dt_scalar
               stage_accumulator[base + idx] += contribution
   ```

3. **Key difference from current:** The DIRK in all_in_one.py uses `stage_rhs_coeffs[successor_idx][prev_idx]` (row-based access) which is already what the current module uses via `a_flat`. Verify the access pattern is optimized.

**Expected Behavior:**
- Stage accumulator fills with contributions from previous stages
- Newton solver receives properly accumulated base state
- Final output accumulation follows tableau structure

### 3. generic_rosenbrock_w.py - GenericRosenbrockWStep.build_step()

**Current Location:** `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

**No direct equivalent in all_in_one.py** - Apply similar optimization principles:

1. **Review** the stage loop (lines 535-665) for similar optimization opportunities
2. **Optimize** coefficient access patterns where applicable
3. **Ensure** loop structure enables compiler unrolling

**Key Optimization Opportunities:**
- The `for predecessor_offset in range(stage_idx)` loops could potentially be restructured
- The `a_coeffs` and `C_coeffs` access patterns should be reviewed

**Current Pattern to Preserve:**
- Linear solver integration
- Jacobian caching
- Time derivative handling

### 4. generic_firk.py - FIRKStep.build_step()

**Current Location:** `src/cubie/integrators/algorithms/generic_firk.py`

**No direct equivalent in all_in_one.py** - Apply similar optimization principles:

1. **Review** the stage accumulation loop (lines 460-518) for optimization
2. **Consider** restructuring the coefficient accumulation pattern
3. **Maintain** the Kahan summation for accuracy

**Key Optimization Opportunities:**
- The nested `for stage_idx... for contrib_idx` pattern could be optimized
- The `stage_rhs_coeffs` access pattern should enable unrolling

### 5. Instrumented Copies

**Location:** `tests/integrators/algorithms/instrumented/`

**Files to Update:**
- `generic_erk.py`
- `generic_dirk.py`
- `generic_rosenbrock_w.py`
- `generic_firk.py`

**Action Required:** After modifying source files, replicate all changes to instrumented copies, preserving only the logging instrumentation additions.

## Integration Points

### With Tableau Classes

The optimized code relies on tableau methods:
- `tableau.typed_columns(tableau.a, numba_precision)` - Returns columns for ERK
- `tableau.typed_rows(tableau.a, numba_precision)` - Returns rows for DIRK
- `tableau.a_flat(numba_precision)` - Returns flattened matrix

**Verify** these methods exist and return appropriate tuple structures.

### With Memory Management

**DO NOT MODIFY** memory allocation patterns. Current patterns:
- ERK: `stage_rhs` in shared, `stage_accumulator` in local
- DIRK: Both accumulator and solver scratch in shared
- Rosenbrock: Stage buffers in shared
- FIRK: All buffers in shared

### With Step Controllers

No changes to controller interface. Step functions continue to return `int32` status codes.

## Edge Cases

1. **Single-stage methods**: Both ERK and DIRK handle `multistage` flag; ensure optimization doesn't break single-stage case

2. **FSAL optimization**: First-Same-As-Last caching must continue to work with new loop structure

3. **Error accumulation**: Both accumulating and direct-capture error patterns must work

4. **b_row/b_hat_row shortcuts**: Direct assignment from stage accumulator must use correct indices

## Dependencies

- `cubie.cuda_simsafe.all_sync`, `activemask` for warp-vote operations
- `numba.cuda`, `int16`, `int32` for CUDA types
- Tableau classes from respective `*_tableaus.py` modules

## Validation Strategy

After implementation:
1. Run existing algorithm tests: `pytest tests/integrators/algorithms/`
2. Run instrumented tests: `pytest tests/integrators/algorithms/instrumented/`
3. Verify no functional changes (only loop restructuring)

## File Permissions Summary

| File | Action |
|------|--------|
| `src/cubie/integrators/algorithms/generic_erk.py` | MODIFY step function |
| `src/cubie/integrators/algorithms/generic_dirk.py` | MODIFY step function |
| `src/cubie/integrators/algorithms/generic_rosenbrock_w.py` | MODIFY step function |
| `src/cubie/integrators/algorithms/generic_firk.py` | MODIFY step function |
| `tests/integrators/algorithms/instrumented/generic_erk.py` | MODIFY to sync |
| `tests/integrators/algorithms/instrumented/generic_dirk.py` | MODIFY to sync |
| `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py` | MODIFY to sync |
| `tests/integrators/algorithms/instrumented/generic_firk.py` | MODIFY to sync |
