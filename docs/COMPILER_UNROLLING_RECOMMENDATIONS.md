# Compiler Unrolling Recommendations for DIRK Implementation

## Problem Statement

The outer loop in the DIRK implementation (`for prev_idx in range(stages_except_first)`) at lines 1276-1355 in `tests/all_in_one.py` is not being automatically unrolled by the compiler, despite having a compile-time known bound.

**Goal:** Provide structural changes to indexing and loop structure (no manual unrolls) that aid automatic compiler unrolling.

---

## Root Cause Analysis

### Why the Compiler Doesn't Unroll

1. **Loop-carried dependency through `stage_rhs`**
   - Each iteration reads `stage_rhs` (contains `k_i` from previous iteration)
   - Computes new stage value
   - Writes new `k_i` to `stage_rhs`
   - Next iteration cannot begin until current completes

2. **Complex loop body** (80+ lines)
   - Multiple nested loops
   - Conditional branches
   - Function calls with potential side effects
   - Array operations (slicing, indexing)

3. **Compiler heuristics**
   - Code size concerns (unrolling would replicate 80+ lines per stage)
   - Complexity exceeds typical unrolling thresholds

---

## Structural Changes to Aid Unrolling

### Phase 1: Simplify Indexing Patterns ⭐⭐⭐

**Priority:** HIGH  
**Effort:** LOW  
**Expected Impact:** 15-25% performance, 30-50% chance of triggering unrolling

#### 1.1 Remove Array Slicing (Line 1301)

**Current Code:**
```python
# Line 1301
stage_base = stage_accumulator[stage_offset:stage_offset + n]
for idx in range(n):
    stage_base[idx] += state[idx]
```

**Issue:** Array slicing creates a view/reference that obscures memory access pattern from compiler

**Recommended Change:**
```python
# Combine into single indexed loop
for idx in range(n):
    stage_base[idx] = stage_accumulator[stage_offset + idx] + state[idx]
```

**Benefit:**
- Explicit memory access pattern
- No indirection through slice object
- Compiler can analyze access pattern directly

#### 1.2 Pre-compute Offsets at Factory Scope

**Current Code:**
```python
# Lines 1276-1278
for prev_idx in range(stages_except_first):
    stage_offset = int32(prev_idx * n)      # Runtime computation
    stage_idx = prev_idx + int32(1)          # Runtime computation
    matrix_col = explicit_a_coeffs[prev_idx]
```

**Issue:** Even simple arithmetic adds complexity for unroller

**Recommended Change:**
```python
# At factory scope (outside device function):
def dirk_step_inline_factory(...):
    # ... existing setup ...
    
    # Pre-compute as compile-time constant tuples
    stage_offsets = tuple(int32(i * n) for i in range(stages_except_first))
    stage_indices = tuple(int32(i + 1) for i in range(stages_except_first))
    
    @cuda.jit(...)
    def step(...):
        # ... 
        
        for prev_idx in range(stages_except_first):
            # Direct constant lookups (no arithmetic)
            stage_offset = stage_offsets[prev_idx]
            stage_idx = stage_indices[prev_idx]
            matrix_col = explicit_a_coeffs[prev_idx]
            # ...
```

**Benefit:**
- Replaces runtime arithmetic with constant tuple indexing
- Compiler sees simple pattern: `tuple[loop_var]`
- Easier to unroll (no computation in loop header)

#### 1.3 Flatten 2D Coefficient Arrays

**Current Code:**
```python
# Line 1279
matrix_col = explicit_a_coeffs[prev_idx]

# Line 1283
coeff = matrix_col[successor_idx + int32(1)]
```

**Issue:** Two-level indirection: `explicit_a_coeffs[i][j]`

**Recommended Change:**
```python
# At factory scope: flatten to 1D array
def dirk_step_inline_factory(...):
    # ... existing setup ...
    
    explicit_a_coeffs = tableau.explicit_terms(numba_precision)
    
    # Flatten to 1D (row-major order)
    explicit_a_flat = []
    for col_idx in range(stage_count):
        for row_idx in range(stage_count):
            explicit_a_flat.append(explicit_a_coeffs[col_idx][row_idx])
    explicit_a_flat = tuple(explicit_a_flat)
    
    @cuda.jit(...)
    def step(...):
        # ...
        
        for prev_idx in range(stages_except_first):
            for successor_idx in range(stages_except_first):
                # Single-level indexing with computed offset
                flat_idx = prev_idx * stage_count + successor_idx + int32(1)
                coeff = explicit_a_flat[flat_idx]
                # ...
```

**Benefit:**
- Single-level indexing: `array[computed_index]`
- Simpler memory access pattern
- Reduces pointer chasing

---

### Phase 2: Reduce Loop Complexity 🔧

**Priority:** MEDIUM  
**Effort:** MEDIUM  
**Expected Impact:** Additional 5-10% performance

#### 2.1 Lift Compile-Time Conditionals

**Current Code:**
```python
for prev_idx in range(stages_except_first):
    # ...
    
    if has_driver_function:  # Compile-time constant
        driver_function(...)
    
    # ...
    
    if stage_implicit[stage_idx]:  # Runtime data
        nonlinear_solver(...)
```

**Issue:** Mixing compile-time and runtime conditionals in loop body

**Recommended Change:**
```python
# At factory scope: create conditional helper functions
if has_driver_function:
    @cuda.jit(device=True, inline=True)
    def maybe_update_drivers(stage_time, driver_coeffs, proposed_drivers):
        driver_function(stage_time, driver_coeffs, proposed_drivers)
else:
    @cuda.jit(device=True, inline=True)
    def maybe_update_drivers(stage_time, driver_coeffs, proposed_drivers):
        pass  # No-op, will be optimized away

# In device function:
for prev_idx in range(stages_except_first):
    # ...
    maybe_update_drivers(stage_time, driver_coeffs, proposed_drivers)
    # Compiler can inline and eliminate dead branches
```

**Benefit:**
- Uniform call site (no conditional in loop)
- Compiler can inline and eliminate no-op version
- Reduces branch complexity

#### 2.2 Simplify Accumulation Pattern

**Current Code:**
```python
for idx in range(n):
    increment = stage_rhs[idx]
    if accumulates_output:
        proposed_state[idx] += solution_weight * increment
    elif b_row == stage_idx:
        proposed_state[idx] = stage_base[idx]
```

**Issue:** Data-dependent branching in inner loop

**Alternative (if beneficial):**
```python
# Use predicated execution pattern
accumulate_flag = int32(1) if accumulates_output else int32(0)
assign_flag = int32(1) if b_row == stage_idx else int32(0)

for idx in range(n):
    increment = stage_rhs[idx]
    # Predicated accumulate
    proposed_state[idx] += accumulate_flag * solution_weight * increment
    # Predicated assign (using selp-like pattern)
    proposed_state[idx] = (assign_flag * stage_base[idx] + 
                           (1 - assign_flag) * proposed_state[idx])
```

**Note:** This may not always be beneficial - benchmark before applying

---

### Phase 3: Break Loop-Carried Dependency 🚀

**Priority:** LOW (High effort, only if Phase 1-2 insufficient)  
**Effort:** HIGH  
**Expected Impact:** 30-50% performance, 70-90% chance of unrolling  
**Cost:** O(stages × n) extra memory

#### 3.1 Buffer All RHS Values

**Current Pattern:**
```python
for prev_idx in range(stages_except_first):
    # Stream previous k (read stage_rhs)
    for successor_idx in range(stages_except_first):
        contribution = coeff * stage_rhs[idx] * dt_scalar
        stage_accumulator[...] += contribution
    
    # Compute new stage (write stage_rhs)
    dxdt_fn(..., stage_rhs, ...)
```

**Alternative Pattern:**
```python
# Allocate buffer for all RHS values
all_stage_rhs = cuda.local.array(stages_except_first * n, numba_precision)

# Pass 1: Compute all stages (store k's separately)
for stage_idx in range(int32(1), stage_count):
    prev_idx = stage_idx - int32(1)
    
    # ... build stage_base from accumulator ...
    
    # Compute stage
    if stage_implicit[stage_idx]:
        nonlinear_solver(...)
        for idx in range(n):
            stage_base[idx] += diagonal_coeff * stage_increment[idx]
    
    # Compute derivatives and store in dedicated buffer
    rhs_offset = prev_idx * n
    observables_function(...)
    dxdt_fn(..., all_stage_rhs[rhs_offset:rhs_offset + n], ...)
    
    # Accumulate to solution
    for idx in range(n):
        if accumulates_output:
            proposed_state[idx] += solution_weight * all_stage_rhs[rhs_offset + idx]

# Pass 2: Stream all k's to accumulators (now fully independent)
for prev_idx in range(stages_except_first):
    rhs_offset = prev_idx * n
    for successor_idx in range(stages_except_first):
        if successor_idx >= prev_idx:  # Only stream to future stages
            coeff = explicit_a_coeffs[prev_idx][successor_idx + int32(1)]
            for idx in range(n):
                target_offset = successor_idx * n
                stage_accumulator[target_offset + idx] += (
                    coeff * all_stage_rhs[rhs_offset + idx] * dt_scalar
                )
```

**Benefits:**
- Breaks loop-carried dependency
- Pass 2 loop can potentially be unrolled (all iterations independent)
- More explicit data flow

**Drawbacks:**
- Requires O(stages × n) extra memory
- More complex code structure
- Two-pass algorithm

---

## Implementation Roadmap

### Step 1: Apply Phase 1 Changes (Recommended Starting Point)

1. Remove array slicing (line 1301)
2. Pre-compute stage offsets and indices
3. Flatten explicit_a_coeffs to 1D

**Expected outcome:** Simpler code, easier for compiler to analyze, may trigger unrolling

### Step 2: Measure and Evaluate

1. Compile and examine PTX/SASS output
2. Profile performance with representative DIRK tableaus
3. Check if unrolling occurred

### Step 3: Apply Phase 2 if Needed

If Phase 1 doesn't achieve unrolling:
1. Lift compile-time conditionals
2. Simplify accumulation patterns

### Step 4: Consider Phase 3 Only if Critical

If profiling shows this loop is a significant bottleneck AND Phase 1-2 don't achieve unrolling:
1. Implement RHS buffering
2. Validate correctness thoroughly
3. Measure memory impact

---

## Example: Phase 1 Implementation

```python
def dirk_step_inline_factory(
    nonlinear_solver,
    dxdt_fn,
    observables_function,
    driver_function,
    n,
    prec,
    tableau,
):
    """Create inline DIRK step device function matching generic_dirk.py."""
    
    numba_precision = numba_from_dtype(prec)
    typed_zero = numba_precision(0.0)
    
    # Extract tableau properties
    n_arraysize = n
    accumulator_length_arraysize = int(max(tableau.stage_count-1, 1) * n)
    double_n = 2 * n
    n = int32(n)
    stage_count = int32(tableau.stage_count)
    stages_except_first = stage_count - int32(1)
    
    # ===== PHASE 1 OPTIMIZATION: Pre-compute offsets =====
    stage_offsets = tuple(int32(i * n) for i in range(stages_except_first))
    stage_indices = tuple(int32(i + 1) for i in range(stages_except_first))
    
    # Compile-time toggles
    has_driver_function = driver_function is not None
    has_error = tableau.has_error_estimate
    multistage = stage_count > 1
    
    # Extract and type tableau coefficients
    stage_rhs_coeffs = tableau.typed_columns(tableau.a, numba_precision)
    explicit_a_coeffs = tableau.explicit_terms(numba_precision)
    
    # ===== PHASE 1 OPTIMIZATION: Flatten 2D to 1D =====
    explicit_a_flat = []
    for col_idx in range(stage_count):
        for row_idx in range(stage_count):
            explicit_a_flat.append(explicit_a_coeffs[col_idx][row_idx])
    explicit_a_flat = tuple(explicit_a_flat)
    
    solution_weights = tableau.typed_vector(tableau.b, numba_precision)
    error_weights = tableau.error_weights(numba_precision)
    if error_weights is None or not has_error:
        error_weights = tuple(typed_zero for _ in range(stage_count))
    stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)
    diagonal_coeffs = tableau.diagonal(numba_precision)
    
    # ... rest of setup ...
    
    @cuda.jit([...], device=True, inline=True, **compile_kwargs)
    def step(
        state,
        proposed_state,
        parameters,
        driver_coeffs,
        drivers_buffer,
        proposed_drivers,
        observables,
        proposed_observables,
        error,
        dt_scalar,
        time_scalar,
        first_step_flag,
        accepted_flag,
        shared,
        persistent_local,
        counters,
    ):
        # ... buffer allocation ...
        # ... stage 0 computation ...
        
        # Stages 1-s loop
        for prev_idx in range(stages_except_first):
            # ===== PHASE 1: Direct lookups instead of arithmetic =====
            stage_offset = stage_offsets[prev_idx]
            stage_idx = stage_indices[prev_idx]
            
            # Stream previous stage's RHS into accumulators
            for successor_idx in range(stages_except_first):
                # ===== PHASE 1: Flat array access =====
                flat_idx = prev_idx * stage_count + successor_idx + int32(1)
                coeff = explicit_a_flat[flat_idx]
                row_offset = successor_idx * n
                for idx in range(n):
                    contribution = coeff * stage_rhs[idx] * dt_scalar
                    stage_accumulator[row_offset + idx] += contribution
            
            stage_time = (
                current_time + dt_scalar * stage_time_fractions[stage_idx]
            )
            
            if has_driver_function:
                driver_function(
                    stage_time,
                    driver_coeffs,
                    proposed_drivers,
                )
            
            # ===== PHASE 1: No slicing, direct indexed loop =====
            for idx in range(n):
                stage_base[idx] = stage_accumulator[stage_offset + idx] + state[idx]
            
            diagonal_coeff = diagonal_coeffs[stage_idx]
            
            if stage_implicit[stage_idx]:
                status_code |= nonlinear_solver(
                    stage_increment,
                    parameters,
                    proposed_drivers,
                    stage_time,
                    dt_scalar,
                    diagonal_coeffs[stage_idx],
                    stage_base,
                    solver_scratch,
                    counters,
                )
                for idx in range(n):
                    stage_base[idx] += diagonal_coeff * stage_increment[idx]
            
            observables_function(
                stage_base,
                parameters,
                proposed_drivers,
                proposed_observables,
                stage_time,
            )
            
            dxdt_fn(
                stage_base,
                parameters,
                proposed_drivers,
                proposed_observables,
                stage_rhs,
                stage_time,
            )
            
            solution_weight = solution_weights[stage_idx]
            error_weight = error_weights[stage_idx]
            for idx in range(n):
                increment = stage_rhs[idx]
                if accumulates_output:
                    proposed_state[idx] += solution_weight * increment
                elif b_row == stage_idx:
                    proposed_state[idx] = stage_base[idx]
                
                if has_error:
                    if accumulates_error:
                        error[idx] += error_weight * increment
                    elif b_hat_row == stage_idx:
                        error[idx] = stage_base[idx]
        
        # ... finalization ...
        
        return status_code
    
    return step
```

---

## Expected Outcomes

### Phase 1 Implementation

**Code changes:**
- ~10 lines modified in factory function
- ~3 lines modified in device function loop

**Complexity:**
- Minimal increase (pre-computation happens at factory scope)
- Device function becomes slightly simpler (fewer operations in loop)

**Performance:**
- 15-25% improvement expected
- 30-50% chance of triggering automatic unrolling

**Risk:**
- Very low (straightforward transformations)
- Easy to validate (compare outputs before/after)

### If Unrolling Doesn't Occur After Phase 1

The fundamental blocker is the **loop-carried dependency through `stage_rhs`**. Breaking this dependency (Phase 3) is the only guaranteed way to enable unrolling, but requires significant restructuring and memory overhead.

**Recommendation:** Accept current performance for debug script. Only pursue Phase 3 if this code becomes production-critical and profiling identifies it as a bottleneck.

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-10  
**Target File:** `tests/all_in_one.py` lines 939-1394
