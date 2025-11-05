# Agent Implementation Plan: Last-Step Caching for RODAS*P and RadauIIA5

## Overview

Implement compile-time optimization for Runge-Kutta tableaus where final solution weights match a row in the coupling matrix. This eliminates redundant accumulation operations by directly copying pre-computed stage increments.

## Component Modifications

### 1. ButcherTableau Base Class Enhancement

**File**: `src/cubie/integrators/algorithms/base_algorithm_step.py`

**New Properties to Add**:

#### Property: `b_matches_a_row`
- **Type**: `Optional[int]`
- **Returns**: Row index (0-based) where `a[row]` equals `b`, or None if no match
- **Behavior**: 
  - Iterate through rows of `a` matrix
  - For each row, compare all elements with corresponding elements in `b`
  - Use tolerance of 1e-15 for floating-point comparison
  - Return index of matching row (prefer last if multiple matches)
  - Return None if no match found

#### Property: `b_hat_matches_a_row`
- **Type**: `Optional[int]`
- **Returns**: Row index where `a[row]` equals `b_hat`, or None if no match or no b_hat
- **Behavior**:
  - Return None immediately if `b_hat` is None
  - Otherwise, same logic as `b_matches_a_row` but comparing with `b_hat`

**Implementation Notes**:
- Use `@property` decorator for lazy evaluation
- Compare only up to `stage_count` elements (ignore padding)
- Handle different row lengths gracefully
- Do NOT cache the result (tableaus are immutable frozen attrs classes)

### 2. Generic Rosenbrock-W Step Optimization

**File**: `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

**Changes in `build_step()` method**:

**Current behavior** (lines ~494-500):
```python
for idx in range(n):
    increment = stage_increment[idx]
    proposed_state[idx] += solution_weight * increment
    if has_error:
        error[idx] += error_weight * increment
```

**New behavior**:
1. Before the device function definition, check tableau properties:
   ```python
   b_row = tableau.b_matches_a_row
   b_hat_row = tableau.b_hat_matches_a_row
   ```

2. Inside device function, add compile-time branches:
   - If `b_row is not None`: Use direct copy from `stage_store[b_row * n : (b_row + 1) * n]` for proposed_state
   - If `b_hat_row is not None`: Use direct copy for error calculation
   - Otherwise: Use existing accumulation loop

**Compile-time branch structure**:
```python
# Pseudo-code showing branch structure
if b_row is not None:  # Compile-time constant
    # Direct copy path for proposed_state
    stage_slice = stage_store[b_row * n : (b_row + 1) * n]
    for idx in range(n):
        proposed_state[idx] = state[idx] + stage_slice[idx]
else:
    # Existing accumulation path
    for stage_idx in range(stage_count):
        # ... accumulation logic
```

**Integration points**:
- Access `stage_store` buffer (already available in scope)
- Proposed state still initialized to `state[idx]` at line ~339
- Direct copy adds the stored increment, same as accumulation result
- Error estimate follows similar pattern with `b_hat_row`

**Expected behavior**:
- RODAS4P: `b_row = 5` (last row, 0-indexed), `b_hat_row = 4`
- RODAS5P: `b_row = 7` (last row), `b_hat_row = 6`
- Other Rosenbrock tableaus: `b_row = None`, fallback to accumulation

### 3. Generic FIRK Step Optimization

**File**: `src/cubie/integrators/algorithms/generic_firk.py`

**Changes in `build_step()` method**:

**Current behavior** (lines ~360-372):
```python
for comp_idx in range(n):
    solution_acc = typed_zero
    error_acc = typed_zero
    for stage_idx in range(stage_count):
        rhs_value = stage_rhs_flat[stage_idx * n + comp_idx]
        solution_acc += solution_weights[stage_idx] * rhs_value
        if has_error:
            error_acc += error_weights[stage_idx] * rhs_value
    proposed_state[comp_idx] = state[comp_idx] + dt_value * solution_acc
    if has_error:
        error[comp_idx] = dt_value * error_acc
```

**New behavior**:
1. Check tableau properties before device function
2. Add compile-time branches for direct access to `stage_rhs_flat`

**Note for FIRK**: 
- FIRK accumulates from RHS values (`stage_rhs_flat`), not increments
- For RadauIIA5, last row of `a` equals `b`, so last stage's contribution IS the solution
- Direct copy: `proposed_state[idx] = state[idx] + dt_value * stage_rhs_flat[b_row * n + idx]`

**Expected behavior**:
- RadauIIA5: `b_row = 2` (last row of 3-stage method)
- Gauss-Legendre: `b_row = None`, use accumulation

### 4. Generic ERK Optimization

**File**: `src/cubie/integrators/algorithms/generic_erk.py`
**Instrumented File**: `tests/integrators/algorithms/instrumented/generic_erk.py`

**Changes in `build_step()` method**:

Apply the same optimization pattern as for Rosenbrock-W:
- Check `tableau.b_matches_a_row` and `tableau.b_hat_matches_a_row` before device function
- Add compile-time branches to use direct copy when properties return non-None
- Access pre-computed stage values from storage arrays

**Current accumulation** (lines ~296-310):
```python
for idx in range(n):
    increment = stage_rhs[idx]
    proposed_state[idx] += solution_weights[stage_idx] * increment
    if has_error:
        error[idx] += error_weights[stage_idx] * increment
```

**Optimized path**:
- If `b_row is not None`: Copy directly from `stage_cache` or stage storage
- If `b_hat_row is not None`: Copy directly for error calculation

**Instrumented counterpart**:
- Apply identical changes to `tests/integrators/algorithms/instrumented/generic_erk.py`
- Maintain exact parallel structure with main implementation

### 5. Generic DIRK Optimization

**File**: `src/cubie/integrators/algorithms/generic_dirk.py`
**Instrumented File**: `tests/integrators/algorithms/instrumented/generic_dirk.py`

**Changes in `build_step()` method**:

Apply the same optimization pattern:
- Check tableau properties before device function
- Add compile-time branches for direct copy
- Access stage values from appropriate storage buffers

**Instrumented counterpart**:
- Apply identical changes to `tests/integrators/algorithms/instrumented/generic_dirk.py`
- Maintain exact parallel structure with main implementation

## Data Structures

### No New Data Structures Required

The optimization uses existing structures:
- `ButcherTableau.a`, `.b`, `.b_hat` (already exists)
- `stage_store` buffer in Rosenbrock methods (already allocated)
- `stage_rhs_flat` buffer in FIRK methods (already allocated)
- `stage_rhs` buffer in ERK methods (already allocated)
- Stage storage in DIRK methods (already allocated)

All generic algorithms have appropriate storage for stage increments/RHS values that can be accessed directly when row equality is detected.

## Expected Interactions

### Compilation Flow

1. Algorithm `__init__()` receives tableau instance
2. Algorithm `build_step()` called by CUDAFactory
3. Inside `build_step()`:
   - Access `tableau.b_matches_a_row` and `tableau.b_hat_matches_a_row`
   - These properties are Python constants at compile time
   - Numba's JIT compiler folds branches based on constants
   - Dead code elimination removes unused branch
4. Resulting device function contains only optimal path

### Runtime Behavior

- **No runtime checks**: Branch already eliminated by compiler
- **No performance penalty**: Optimal code for each tableau
- **Numerical equivalence**: Direct copy produces identical result to accumulation

## Edge Cases to Handle

### Edge Case 1: No embedded error estimate
- `b_hat` is None
- `b_hat_matches_a_row` returns None
- Only optimize proposed_state, leave error path as-is
- **Handling**: Check `has_error` flag, only apply error optimization when both conditions met

### Edge Case 2: Tableau with row padding
- Some tableaus pad rows with zeros to uniform length
- Comparison should only consider first `stage_count` elements
- **Handling**: Slice both `a[row]` and `b` to `[:stage_count]` before comparison

### Edge Case 3: Multiple matching rows
- Theoretically possible (though unlikely in practice)
- **Handling**: Return last matching row index (most likely to be the final stage)

### Edge Case 4: Near-equality (floating point)
- Tableau coefficients may have decimal representation errors
- **Handling**: Use tolerance of 1e-15 in comparison (tighter than computation but loose enough for definitions)

### Edge Case 5: Empty tableau or single stage
- `stage_count == 1` means only one row
- **Handling**: Comparison still works; if `a[0] == b`, optimization applies

## Dependencies

### Internal Dependencies
- `ButcherTableau` class (base_algorithm_step.py)
- `RosenbrockTableau` inherits from ButcherTableau
- `FIRKTableau` inherits from ButcherTableau
- Generic algorithm implementations import tableaus

### External Dependencies
- None (uses existing Numba compilation)

### Import Changes
- No new imports required
- All needed functionality already available

## Testing Strategy

### Unit Tests for Tableau Properties

**New test file**: `tests/integrators/algorithms/test_tableau_properties.py`

**Tests to add**:
1. Test `b_matches_a_row` returns correct index for RODAS4P (expect 5)
2. Test `b_matches_a_row` returns correct index for RODAS5P (expect 7)
3. Test `b_matches_a_row` returns correct index for RadauIIA5 (expect 2)
4. Test `b_matches_a_row` returns None for tableaus without match (e.g., ROS3P)
5. Test `b_hat_matches_a_row` returns correct index for RODAS4P (expect 4)
6. Test `b_hat_matches_a_row` returns None when b_hat is None
7. Test floating-point tolerance in comparison

### Integration Tests

**Existing test files**: Modify or add to existing algorithm tests

**Tests to add**:
1. Run RODAS4P solver and verify results match CPU reference
2. Run RODAS5P solver and verify results match CPU reference
3. Run RadauIIA5 solver and verify results match CPU reference
4. Verify no unnecessary accumulation occurs (inspect compiled code structure if feasible)
5. Test across different precisions (float32, float64)

### Numerical Validation

**Approach**:
- Use existing `cpu_reference/algorithms.py` implementations
- Run same problem with GPU (optimized) and CPU (reference)
- Assert results match within tolerance (use existing test patterns)
- Test across different precisions (float32, float64)

### Instrumented Algorithm Updates

**Critical**: All changes to generic algorithms must be mirrored in instrumented versions:
- `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py`
- `tests/integrators/algorithms/instrumented/generic_firk.py`
- `tests/integrators/algorithms/instrumented/generic_erk.py`
- `tests/integrators/algorithms/instrumented/generic_dirk.py`

**Validation**:
- Ensure instrumented versions maintain exact parallel structure
- Any compile-time branch added to main algorithm must appear in instrumented version
- Keep variable names and logic flow identical

## Computational Efficiency Goals

### Elimination of Unnecessary Operations

**RODAS4P (6 stages)**:
- Without optimization: 6 multiplications + 6 additions per state variable for proposed_state
- With optimization: Direct copy from stage storage (no accumulation)
- Same savings for error estimate

**RODAS5P (8 stages)**:
- Without optimization: 8 multiplications + 8 additions per state variable
- With optimization: Direct copy from stage storage
- Same savings for error estimate

**RadauIIA5 (3 stages)**:
- Without optimization: 3 multiplications + 3 additions per state variable
- With optimization: Direct copy from stage storage

### Measurement

- Validation via numerical equivalence tests (not timing tests)
- Confirm no unnecessary accumulation through code inspection
- Verify compile-time branch elimination in generated device functions

## Validation Checklist

Before marking implementation complete:

- [ ] `ButcherTableau.b_matches_a_row` property implemented and tested
- [ ] `ButcherTableau.b_hat_matches_a_row` property implemented and tested
- [ ] Rosenbrock-W step optimization implemented (main and instrumented)
- [ ] FIRK step optimization implemented (main and instrumented)
- [ ] ERK step optimization implemented (main and instrumented)
- [ ] DIRK step optimization implemented (main and instrumented)
- [ ] All existing tests pass (no regressions)
- [ ] New unit tests for tableau properties pass
- [ ] Integration tests confirm numerical equivalence
- [ ] CPU reference validation confirms correctness
- [ ] Linting passes (flake8, ruff)
- [ ] Documentation updated if needed (likely just code comments)

## Documentation Updates

### Code Comments
- Add docstring to new tableau properties explaining the optimization
- Add comments in generic algorithm implementations explaining the branch logic
- Reference issue #163 in relevant code comments

### User Documentation
- No user-facing documentation changes needed (transparent optimization)
- Optional: Add note to CHANGELOG.md about performance improvement

### Developer Documentation
- Consider adding note to AGENTS.md or internal structure doc about the optimization
- Explain the property-based detection pattern for future contributors

## Risk Mitigation

### Risk: Numerical differences due to operation reordering
**Mitigation**: 
- Test with strict tolerance against CPU reference
- Test across multiple precisions
- If differences found, document and assess acceptability

### Risk: Compile-time branch not eliminated, runtime overhead
**Mitigation**:
- Inspect generated CUDA PTX/SASS to verify branch elimination
- Use Numba's constant folding features explicitly if needed
- Fallback: Use template-style separate functions if branches persist

### Risk: Breaking change for subclassed tableaus
**Mitigation**:
- Properties have default behavior (None) for non-matching tableaus
- No breaking changes to existing API
- Frozen attrs classes prevent mutation issues

### Risk: Future tableau definitions don't get optimization
**Mitigation**:
- Properties are automatic; any new tableau with matching rows gets optimization
- Document the property in ButcherTableau docstring for visibility
- Consider CI check or warning for tableaus that could benefit

## Implementation Order

1. **Phase 1**: Tableau properties
   - Add `b_matches_a_row` and `b_hat_matches_a_row` to ButcherTableau
   - Write unit tests for properties
   - Verify all existing tests still pass

2. **Phase 2**: Rosenbrock optimization
   - Modify `generic_rosenbrock_w.py` build_step()
   - Modify `tests/integrators/algorithms/instrumented/generic_rosenbrock_w.py` in parallel
   - Add/update integration tests for RODAS4P and RODAS5P
   - Verify numerical equivalence

3. **Phase 3**: FIRK optimization
   - Modify `generic_firk.py` build_step()
   - Modify `tests/integrators/algorithms/instrumented/generic_firk.py` in parallel
   - Add/update integration tests for RadauIIA5
   - Verify numerical equivalence

4. **Phase 4**: ERK optimization
   - Modify `generic_erk.py` build_step()
   - Modify `tests/integrators/algorithms/instrumented/generic_erk.py` in parallel
   - Add/update integration tests
   - Verify numerical equivalence

5. **Phase 5**: DIRK optimization
   - Modify `generic_dirk.py` build_step()
   - Modify `tests/integrators/algorithms/instrumented/generic_dirk.py` in parallel
   - Add/update integration tests
   - Verify numerical equivalence

6. **Phase 6**: Documentation and cleanup
   - Add code comments
   - Update CHANGELOG.md
   - Final validation pass

## Success Criteria

- All validation checklist items completed
- No unnecessary accumulation when row equality exists
- Zero regressions in existing tests
- All generic algorithms (ERK, DIRK, FIRK, Rosenbrock-W) support optimization
- Instrumented versions updated in parallel with main implementations
- Code review approval
- User stories acceptance criteria met
