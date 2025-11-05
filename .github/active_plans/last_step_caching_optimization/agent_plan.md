# Agent Plan: Last-Step Caching Optimization

## Overview

Implement compile-time optimization for Runge-Kutta tableaus where solution weights (b) or error weights (b_hat) match rows of the A matrix. This allows reusing already-computed stage states instead of performing redundant weighted accumulations.

**Target Tableaus:**
- RODAS3P, RODAS4P, RODAS5P (Rosenbrock-W family)
- RadauIIA5 (Fully Implicit RK)

**Performance Goal:** Reduce per-step computation by 5-15% for affected methods without compromising numerical accuracy.

## Component 1: ButcherTableau Property Extensions

**Location:** `src/cubie/integrators/algorithms/base_algorithm_step.py`

**Purpose:** Add detection logic for tableaus where b or b_hat match rows of A.

### New Properties

#### Property: `can_reuse_stage_for_solution`
- **Returns:** `bool`
- **Behavior:** Returns True if any row of the A matrix matches the b weights (ignoring trailing zeros and final increments)
- **Detection Logic:**
  - Iterate through A rows (index i from 0 to stage_count-1)
  - For each row, check if first `i` elements of A[i] match first `i` elements of b
  - Handle floating-point comparison with tolerance (e.g., abs(a - b) < 1e-12)
  - Return True if any match found
  - Return False otherwise

#### Property: `b_matching_row_index`
- **Returns:** `Optional[int]`
- **Behavior:** Returns the index of the A row that matches b, or None if no match
- **Detection Logic:**
  - Similar to `can_reuse_stage_for_solution` but returns the index
  - If multiple matches exist, return the first one found
  - Cache result to avoid repeated computation

#### Property: `can_reuse_stage_for_error`
- **Returns:** `bool`
- **Behavior:** Returns True if b_hat is defined and matches a row of A
- **Detection Logic:**
  - Only proceed if `self.b_hat is not None`
  - Similar logic to `can_reuse_stage_for_solution` but comparing against b_hat
  - Return False if no b_hat exists

#### Property: `bhat_matching_row_index`
- **Returns:** `Optional[int]`
- **Behavior:** Returns the index of the A row that matches b_hat, or None
- **Detection Logic:**
  - Returns None if b_hat is None
  - Otherwise, similar to `b_matching_row_index`

### Implementation Notes

- **Floating-Point Tolerance:** Use tolerance of 1e-12 for comparisons to handle representation issues
- **Caching:** Consider using `@property` with internal cached attributes to avoid repeated comparisons
- **Pattern Matching:** For RODAS methods, A[i] matches b for first i elements, with remaining elements being 0 except possibly final stages with weight 1.0
- **Edge Cases:** Single-stage methods (can't match), methods without embedded estimates (b_hat checks return False)

### Expected Results

When implemented:
- `RODAS3P_TABLEAU.can_reuse_stage_for_solution` → True
- `RODAS3P_TABLEAU.b_matching_row_index` → 4 (5th row, 0-indexed)
- `RODAS4P_TABLEAU.can_reuse_stage_for_solution` → True
- `RODAS4P_TABLEAU.b_matching_row_index` → 4
- `RODAS4P_TABLEAU.can_reuse_stage_for_error` → True
- `RODAS4P_TABLEAU.bhat_matching_row_index` → 4
- `RODAS5P_TABLEAU.can_reuse_stage_for_solution` → True
- `RODAS5P_TABLEAU.b_matching_row_index` → 5
- `RADAU_IIA_5_TABLEAU.can_reuse_stage_for_solution` → True
- `RADAU_IIA_5_TABLEAU.b_matching_row_index` → 2 (last stage)

## Component 2: GenericRosenbrockWStep Optimization

**Location:** `src/cubie/integrators/algorithms/generic_rosenbrock_w.py`

**Purpose:** Modify the build_step method to generate optimized CUDA kernel when tableau supports reuse.

### Compilation-Time Detection

In `build_step` method, after extracting tableau:
```python
can_reuse = tableau.can_reuse_stage_for_solution
reuse_row_idx = tableau.b_matching_row_index if can_reuse else None
can_reuse_error = tableau.can_reuse_stage_for_error
error_reuse_row_idx = tableau.bhat_matching_row_index if can_reuse_error else None
```

### Kernel Generation Changes

**Current Accumulation Logic (lines ~387-389, 494-500):**
```python
# Stage 0
proposed_state[idx] += stage_increment[idx] * solution_weights[0]
if has_error:
    error[idx] += stage_increment[idx] * error_weights[0]

# Stages 1..s
proposed_state[idx] += solution_weight * increment
if has_error:
    error[idx] += error_weight * increment
```

**Optimized Logic (when `can_reuse` is True):**

If `can_reuse` and `stage_idx == reuse_row_idx`:
- The stage_state at this stage already contains `state + A[reuse_row_idx] @ stage_increments`
- Copy this to proposed_state
- Add remaining stage increments with their weights

Detailed behavior:
1. For stages 0 to `reuse_row_idx-1`: Continue normal accumulation into proposed_state
2. At stage `reuse_row_idx`:
   - Copy `stage_slice[idx]` to a temporary buffer or directly to proposed_state base
   - The stage_slice already contains the weighted sum from A[reuse_row_idx]
   - This corresponds to the first `reuse_row_idx` elements of b
3. For stages `reuse_row_idx+1` to `stage_count-1`:
   - Add increments with weights b[stage_idx]
   - These are typically just the final stage(s) with weight 1.0

### Specific Code Modifications

**Alternative Approach (cleaner):**
- Keep the accumulation loop structure
- But for stages 0 to `reuse_row_idx`, set solution_weight to 0.0 (skip accumulation)
- At stage `reuse_row_idx`, copy stage_slice to proposed_state before accumulation
- Continue normal accumulation for remaining stages

**Pseudo-code:**
```python
if can_reuse:
    # At stage reuse_row_idx, after computing stage_slice
    if stage_idx == reuse_row_idx:
        for idx in range(n):
            proposed_state[idx] = stage_slice[idx]
    
    # Then accumulate only for stages > reuse_row_idx
    if stage_idx > reuse_row_idx:
        solution_weight = solution_weights[stage_idx]
        for idx in range(n):
            proposed_state[idx] += solution_weight * stage_increment[idx]
    # For stages <= reuse_row_idx, skip accumulation (already in proposed_state)
else:
    # Original accumulation logic
    for idx in range(n):
        proposed_state[idx] += solution_weight * increment
```

**Error Estimate Optimization:**
Similarly, if `can_reuse_error` is True:
- Skip error accumulation for stages 0 to `error_reuse_row_idx`
- At stage `error_reuse_row_idx`, copy from appropriate stage state buffer
- Accumulate remaining stages normally

### Integration with Existing Code

- **Stage 0 special handling:** Preserve existing stage 0 logic (lines ~345-390)
- **Stage loop:** Modify lines ~394-500 to include conditional logic
- **Shared memory layout:** No changes needed to buffer allocation
- **Driver function calls:** No changes
- **Linear solver calls:** No changes

### Expected Behavior Changes

- **Numerical Results:** Bit-for-bit identical to original (reusing exact same computed values)
- **Performance:** Fewer FLOPs per step (skip ~n * reuse_row_idx multiplications and additions)
- **Memory Access:** Similar memory traffic (reading stage_slice vs stage_store multiple times)

## Component 3: FIRKStep Optimization

**Location:** `src/cubie/integrators/algorithms/generic_firk.py`

**Purpose:** Optimize RadauIIA5 and other FIRK tableaus where last stage state equals proposed state.

### Compilation-Time Detection

In `build_step` method (around line 204):
```python
can_reuse = tableau.can_reuse_stage_for_solution
reuse_row_idx = tableau.b_matching_row_index if can_reuse else None
```

### Kernel Generation Changes

**Current Accumulation Logic (lines ~360-372):**
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

**Optimized Logic (when `can_reuse` is True and `reuse_row_idx == stage_count - 1`):**

For RadauIIA5, the last stage state is exactly the proposed state:
```python
if can_reuse and reuse_row_idx == stage_count - 1:
    # The stage_state from last stage is already the solution
    # stage_state was computed at lines ~327-338 in the loop
    # For last stage, copy it to proposed_state
    for comp_idx in range(n):
        proposed_state[comp_idx] = stage_state[comp_idx]
    
    # Error estimate still needs full accumulation (b_hat differs for RadauIIA5)
    if has_error:
        error_acc = typed_zero
        for stage_idx in range(stage_count):
            rhs_value = stage_rhs_flat[stage_idx * n + comp_idx]
            error_acc += error_weights[stage_idx] * rhs_value
        error[comp_idx] = dt_value * error_acc
else:
    # Original logic
```

**Note:** The FIRK implementation computes stage_state in local memory during the stage loop (lines ~327-338). We need to preserve the final stage_state to copy to proposed_state.

### Specific Code Modifications

**Challenge:** stage_state is reused for each stage (it's a local array). Need to preserve it from the last stage.

**Solution:** 
- Keep the existing loop that computes stage_state for each stage
- After the loop completes, if `can_reuse` and it's the last stage, stage_state still holds the final stage values
- Copy it to proposed_state before the current accumulation logic

**Modified structure:**
```python
# Existing stage computation loop (lines ~314-358)
for stage_idx in range(stage_count):
    # ... compute stage_state ...
    # ... evaluate observables and dxdt at stage_state ...

# After loop, stage_state contains the last stage
if can_reuse and reuse_row_idx == stage_count - 1:
    for comp_idx in range(n):
        proposed_state[comp_idx] = stage_state[comp_idx]
else:
    # Original accumulation
    for comp_idx in range(n):
        solution_acc = typed_zero
        for stage_idx in range(stage_count):
            rhs_value = stage_rhs_flat[stage_idx * n + comp_idx]
            solution_acc += solution_weights[stage_idx] * rhs_value
        proposed_state[comp_idx] = state[comp_idx] + dt_value * solution_acc

# Error always computed via accumulation (RadauIIA5 b_hat doesn't match A row)
if has_error:
    for comp_idx in range(n):
        error_acc = typed_zero
        for stage_idx in range(stage_count):
            rhs_value = stage_rhs_flat[stage_idx * n + comp_idx]
            error_acc += error_weights[stage_idx] * rhs_value
        error[comp_idx] = dt_value * error_acc
```

### Expected Behavior Changes

- **RadauIIA5:** Skip ~3n multiply-adds for solution computation
- **Other FIRK methods:** No change (optimization not triggered)
- **Numerical Results:** Identical to original implementation

## Data Structures and Dependencies

### No New Data Structures Required

The optimization leverages existing buffers:
- **Rosenbrock:** `stage_store` buffer already contains stage states
- **FIRK:** `stage_state` local array already contains final stage

### Dependencies

- **ButcherTableau:** All algorithms depend on tableau properties
- **Floating-point precision:** Detection must handle precision correctly
- **Numba compilation:** Compile-time flags must be accessible in device function generation

## Edge Cases and Error Handling

### Edge Case 1: Single-Stage Methods
- **Behavior:** `can_reuse_stage_for_solution` returns False (no prior stages to match)
- **Handling:** Falls through to original accumulation logic

### Edge Case 2: Methods Without Embedded Estimates
- **Behavior:** `can_reuse_stage_for_error` returns False
- **Handling:** Only optimize solution, not error

### Edge Case 3: Floating-Point Comparison Tolerance
- **Issue:** Tableau coefficients may not be exactly representable
- **Handling:** Use tolerance of 1e-12 in row-matching logic
- **Test:** Verify RODAS*P tableaus are detected correctly across precisions

### Edge Case 4: Future Tableaus
- **Behavior:** New tableaus automatically benefit if structure matches
- **Handling:** No special code needed; properties detect automatically

### Edge Case 5: FSAL Interaction
- **Issue:** Existing FSAL logic caches first stage, this optimizes later stages
- **Handling:** No conflict; FSAL operates on stage 0, this optimization on stage s-2 or s-1
- **Note:** Both optimizations can coexist

## Architectural Integrations

### Integration Point 1: Compilation Pipeline
- **Component:** CUDAFactory pattern in algorithm step classes
- **Integration:** Tableau properties accessed during `build()` method
- **Data Flow:** Tableau → Properties → build_step → Compiled kernel

### Integration Point 2: Tableau Registry
- **Component:** ROSENBROCK_TABLEAUS, FIRK_TABLEAU_REGISTRY dictionaries
- **Integration:** Existing tableaus automatically gain new properties
- **Data Flow:** User selects tableau by name → Retrieved from registry → Properties available

### Integration Point 3: Test Infrastructure
- **Component:** Algorithm test suites
- **Integration:** Add tests verifying optimization correctness
- **Data Flow:** Test instantiates solver with RODAS*P → Verifies results match baseline

## Expected Interactions Between Components

### Interaction 1: Tableau → Algorithm
- **Flow:** Algorithm queries tableau properties during compilation
- **Dependencies:** Algorithm must check properties before generating kernel
- **Coupling:** Loose - algorithms work with any tableau, properties are optional hints

### Interaction 2: Rosenbrock ↔ FIRK
- **Flow:** Independent implementations, shared base tableau class
- **Dependencies:** Both use ButcherTableau properties
- **Coupling:** None - each algorithm implements optimization differently

### Interaction 3: Optimization ↔ FSAL
- **Flow:** Independent optimizations
- **Dependencies:** None - operate on different stages
- **Coupling:** None expected, but should verify in testing

## Validation and Testing Strategy

### Validation 1: Numerical Accuracy
- **Method:** Run same problem with and without optimization
- **Metric:** Bit-for-bit identical results (floating-point permutation acceptable)
- **Test Cases:** RODAS3P, RODAS4P, RODAS5P, RadauIIA5 on various systems

### Validation 2: Performance Improvement
- **Method:** Benchmark step execution time
- **Metric:** 5-15% reduction in per-step time for RODAS*P
- **Test Cases:** Medium-to-large systems (n=100, n=1000) with stiff dynamics

### Validation 3: Compilation Success
- **Method:** Verify kernels compile without errors
- **Metric:** No Numba compilation failures
- **Test Cases:** All precisions (float32, float64), all affected tableaus

### Validation 4: Edge Case Handling
- **Method:** Test non-optimizable tableaus
- **Metric:** Original behavior preserved
- **Test Cases:** Backward Euler, ROS3P with modified coefficients, single-stage methods

### Validation 5: FSAL Compatibility
- **Method:** Test tableaus with FSAL + this optimization
- **Metric:** No conflicts, both optimizations active
- **Test Cases:** (Future - after FSAL implementation)

## Implementation Notes

### Code Style Consistency
- Follow existing pattern: `first_same_as_last` property as reference
- Use descriptive property names: `can_reuse_stage_for_solution` not `b_matches_a`
- Add docstrings explaining mathematical basis
- Include references to tableau structure in comments

### Compile-Time vs Runtime
- All optimization decisions at compile-time (during `build()`)
- No runtime branching in generated CUDA kernels
- Use Python conditionals in build_step to generate different code paths

### Documentation Requirements
- Update docstrings for affected methods
- Add comments explaining why optimization is valid
- Reference issue #163 in commit messages
- Document expected performance improvements

### Backward Compatibility
- No breaking changes to public API
- Existing code continues to work
- Optimization is transparent to users
- Results remain numerically equivalent

## Summary of Modifications

### Files to Modify
1. `src/cubie/integrators/algorithms/base_algorithm_step.py` - Add properties to ButcherTableau
2. `src/cubie/integrators/algorithms/generic_rosenbrock_w.py` - Optimize build_step
3. `src/cubie/integrators/algorithms/generic_firk.py` - Optimize build_step

### Files Unchanged
- Tableau definition files (generic_rosenbrockw_tableaus.py, generic_firk_tableaus.py) - properties auto-detect
- Loop implementations
- Step controllers
- Solver interface
- Output handling

### Testing Files to Add/Modify
- Add tests verifying optimization correctness
- Add performance benchmarks
- Verify all tableaus compile successfully
